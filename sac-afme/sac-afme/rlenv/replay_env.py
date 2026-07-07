"""
rlenv/replay_env.py — vectorized log-replay POMDP environment
==============================================================
M synchronized parallel episodes over logged trajectories (episodes are
synchronized so the warmup phase can be batched; the slight sample correlation
is irrelevant with segment randomization).

Multirate schedule: one RL step = one measurement EPOCH = `uwb_stride`
base-rate substeps (50 Hz chain, 10 Hz UWB by default). The filter receives
z=None on substeps (prediction) and z on the epoch (solve) — see wfme.py.
Actions therefore fire exactly when new information arrives.

Observation  o_t = [ |nu|_{t-L+1..t} ]  running-normalized   (approx. information state)
Action       a in [-1,1]^2 → N = round(mid_N + half_N * a1), lam = mid_l + half_l * a2
             (tanh range mapping — lower bounds structurally satisfied;
              cf. RL-AKF exponential mapping for PSD constraints)
Reward       r = -min( scale * ||p_gt - p_hat||^2 , clip )      (RL-AKF)
done         always False (segments are truncations → bootstrap; infinite-horizon)
"""
import torch
from filter.wfme import WeightedFME


class VectorReplayEnv:
    def __init__(self, cfg, dataset, device, seed=0):
        self.cfg, self.ds, self.dev = cfg, dataset, device
        self.M = cfg.n_envs
        self.L = cfg.L_obs
        self.fme = WeightedFME(cfg, device, self.M)
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        self.meas_sig = torch.tensor(cfg.meas_sigma, device=device)
        M = self.M
        self.ti = torch.zeros(M, dtype=torch.long, device=device)
        self.t = torch.zeros(M, dtype=torch.long, device=device)
        self.sigma = torch.zeros(M, 1, device=device)
        self.stack = torch.zeros(M, self.L, device=device)
        self.step_in_ep = 0
        # running normalization of |nu| (global scalars, EMA)
        self.nu_mean = torch.tensor(1.0, device=device)
        self.nu_var = torch.tensor(1.0, device=device)
        self.mom = 0.001
        # action mapping constants
        self.N_mid = 0.5 * (cfg.N_max + cfg.N_min)
        self.N_half = 0.5 * (cfg.N_max - cfg.N_min)
        self.l_mid = 0.5 * (1.0 + cfg.lam_min)
        self.l_half = 0.5 * (1.0 - cfg.lam_min)
        self.default_N = torch.full((M,), float(cfg.N_default), device=device)
        self.default_l = torch.full((M,), float(cfg.lam_default), device=device)

    # ────────────────────────────── helpers
    def map_action(self, a):
        """a [M,2] in [-1,1] → (N [M], lam [M]); ablation flags honored."""
        cfg = self.cfg
        N = torch.round(self.N_mid + self.N_half * a[:, 0]).clamp(cfg.N_min, cfg.N_max)
        lam = (self.l_mid + self.l_half * a[:, 1]).clamp(cfg.lam_min, 1.0)
        if cfg.ablation_fix_lambda:
            lam = torch.ones_like(lam)
        if cfg.ablation_fix_N:
            N = torch.full_like(N, float(cfg.N_default))
        return N, lam

    def _measure(self):
        """noisy z = 4 UWB ranges at current pointers (per-episode sigma)."""
        _, rc, _, _ = self.ds.get(self.ti, self.t)
        return rc + self.sigma * torch.randn(self.M, rc.shape[1],
                                             generator=self.rng, device=self.dev)

    def _epoch(self, N, lam):
        """advance one measurement epoch = `stride` base-rate substeps.
        z=None on substeps (prediction), z on the final substep (solve)."""
        stride = max(1, self.cfg.uwb_stride)
        nu = None
        for i in range(stride):
            self.t += 1
            up, _, _, _ = self.ds.get(self.ti, self.t)
            z = self._measure() if i == stride - 1 else None
            s_hat, nu_i, _ = self.fme.step(up, z, N, lam)
            if nu_i is not None:
                nu = nu_i
        return s_hat, nu

    def _obs(self):
        std = torch.sqrt(self.nu_var).clamp(min=1e-3)
        return ((self.stack - self.nu_mean) / std).clone()

    def _push_nu(self, nu):
        nn = torch.linalg.vector_norm(nu / self.meas_sig, dim=1)   # whitened norm
        m = nn.mean()
        self.nu_mean = (1 - self.mom) * self.nu_mean + self.mom * m
        self.nu_var = (1 - self.mom) * self.nu_var + self.mom * (nn - self.nu_mean).pow(2).mean()
        self.stack = torch.roll(self.stack, 1, dims=1)
        self.stack[:, 0] = nn

    # ────────────────────────────── episode control (synchronized)
    def reset(self):
        cfg = self.cfg
        seg = (cfg.warmup_steps + cfg.episode_len + 1) * max(1, cfg.uwb_stride)
        self.ti, self.t = self.ds.sample_segments(self.M, seg, self.rng)
        lo, hi = cfg.uwb_sigma_range
        self.sigma = lo + (hi - lo) * torch.rand(self.M, 1, generator=self.rng,
                                                 device=self.dev)
        # filter init at GT + small noise
        _, _, _, gt0 = self.ds.get(self.ti, self.t)
        s0 = gt0 + cfg.init_pos_noise * torch.randn(self.M, cfg.state_dim,
                                                    generator=self.rng, device=self.dev)
        self.fme.reset(torch.arange(self.M, device=self.dev), s0)
        self.stack.zero_()
        # ── warmup: default params, no transitions (epochs) ──
        for _ in range(cfg.warmup_steps):
            _, nu = self._epoch(self.default_N, self.default_l)
            self._push_nu(nu)
        self.step_in_ep = 0
        return self._obs()

    def step(self, a):
        """a [M,2] → (obs' [M,L], reward [M], done [M], info)"""
        cfg = self.cfg
        N, lam = self.map_action(a)
        s_hat, nu = self._epoch(N, lam)                      # one epoch = one RL step
        _, _, p_gt, _ = self.ds.get(self.ti, self.t)         # error at epoch time
        err2 = (p_gt - s_hat[:, 0:3]).pow(2).sum(dim=1)
        reward = -torch.clamp(cfg.reward_scale * err2, max=cfg.reward_clip)
        self._push_nu(nu)
        self.step_in_ep += 1
        ep_end = self.step_in_ep >= cfg.episode_len
        done = torch.zeros(self.M, device=self.dev)          # truncation → bootstrap
        obs = self._obs()
        info = {"err": err2.sqrt(), "N": N, "lam": lam, "ep_end": ep_end}
        return obs, reward, done, info
