"""
rlenv/replay_env.py — vectorized log-replay POMDP environment
==============================================================
M synchronized parallel episodes over logged trajectories (episodes are
synchronized so the warmup phase can be batched; the slight sample correlation
is irrelevant with segment randomization).

Multirate schedule: one RL step = one measurement EPOCH = `uwb_stride`
base-rate substeps. The filter receives z=None on substeps (prediction) and z
on the epoch (solve) — see wfme.py.

Observation (POMDP, no true state):
  per-step feature  o_t = [ nu_1, nu_2, nu_3, nu_4, N_hat_{t-1}, lam_hat_{t-1} ]
    nu_i    : per-anchor innovation, whitened then clipped to [-resid_clip, resid_clip]
              (a dropped / NLOS anchor shows up as its OWN channel — the reason
              the residual VECTOR is used instead of a scalar RMSE)
    N_hat   : previous horizon,  min-max normalized  2*(N-N_min)/(N_max-N_min)-1
    lam_hat : previous lambda,   min-max normalized  2*(lam-lam_min)/(1-lam_min)-1
  stacked over the last L steps -> obs = flatten -> dim (meas_dim+2)*L.

Action  a in [-1,1]^2 -> N = round(mid_N + half_N a1), lam = mid_l + half_l a2.
Reward  r = -min( ||p_gt - p_hat|| , reward_clip )  (pure L2 error + safety clip).
done    always False (segments are truncations -> bootstrap; infinite-horizon).
"""
import torch
from filter.wfme import WeightedFME


class VectorReplayEnv:
    def __init__(self, cfg, dataset, device, seed=0):
        self.cfg, self.ds, self.dev = cfg, dataset, device
        self.M = cfg.n_envs
        self.L = cfg.L_obs
        self.nz = cfg.meas_dim
        self.feat = cfg.meas_dim + 2                      # per-step feature width
        self.fme = WeightedFME(cfg, device, self.M)
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(seed)
        self.meas_sig = torch.tensor(cfg.meas_sigma, device=device)
        M = self.M
        self.ti = torch.zeros(M, dtype=torch.long, device=device)
        self.t = torch.zeros(M, dtype=torch.long, device=device)
        self.sigma = torch.zeros(M, 1, device=device)
        self.stack = torch.zeros(M, self.L, self.feat, device=device)  # 0 = newest
        self.step_in_ep = 0
        self.N_mid = 0.5 * (cfg.N_max + cfg.N_min)
        self.N_half = 0.5 * (cfg.N_max - cfg.N_min)
        self.l_mid = 0.5 * (1.0 + cfg.lam_min)
        self.l_half = 0.5 * (1.0 - cfg.lam_min)
        self.default_N = torch.full((M,), float(cfg.N_default), device=device)
        self.default_l = torch.full((M,), float(cfg.lam_default), device=device)

    # ────────────────────────────── action / normalization
    def map_action(self, a):
        cfg = self.cfg
        N = torch.round(self.N_mid + self.N_half * a[:, 0]).clamp(cfg.N_min, cfg.N_max)
        lam = (self.l_mid + self.l_half * a[:, 1]).clamp(cfg.lam_min, 1.0)
        if cfg.ablation_fix_lambda:
            lam = torch.ones_like(lam)
        if cfg.ablation_fix_N:
            N = torch.full_like(N, float(cfg.N_default))
        return N, lam

    def _norm_N(self, N):
        cfg = self.cfg
        return 2.0 * (N - cfg.N_min) / max(cfg.N_max - cfg.N_min, 1e-6) - 1.0

    def _norm_lam(self, lam):
        cfg = self.cfg
        return 2.0 * (lam - cfg.lam_min) / max(1.0 - cfg.lam_min, 1e-6) - 1.0

    # ────────────────────────────── measurement (dropout via NaN rows)
    def _measure(self):
        """noisy z = 4 UWB ranges at current pointers (per-episode sigma).
        Anchor dropout is a dataset scenario: a dropped anchor's clean range
        is NaN, so the measurement row is NaN -> the filter's innovation gate
        excludes it (as in DI-FME's intermittent-dropout handling)."""
        _, rc, _, _ = self.ds.get(self.ti, self.t)
        return rc + self.sigma * torch.randn(self.M, rc.shape[1],
                                             generator=self.rng, device=self.dev)

    def _epoch(self, N, lam):
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

    # ────────────────────────────── observation assembly
    def _push_feature(self, nu, N, lam):
        c = self.cfg.resid_clip
        r = (nu / self.meas_sig).clamp(-c, c)
        r = torch.nan_to_num(r, nan=0.0)                 # dropped anchor -> 0 channel
        feat = torch.cat([r, self._norm_N(N).unsqueeze(1),
                          self._norm_lam(lam).unsqueeze(1)], dim=1)   # [M,feat]
        self.stack = torch.roll(self.stack, 1, dims=1)
        self.stack[:, 0] = feat

    def _obs(self):
        return self.stack.reshape(self.M, self.L * self.feat).clone()

    # ────────────────────────────── episode control (synchronized)
    def reset(self):
        cfg = self.cfg
        seg = (cfg.warmup_steps + cfg.episode_len + 1) * max(1, cfg.uwb_stride)
        self.ti, self.t = self.ds.sample_segments(self.M, seg, self.rng)
        lo, hi = cfg.uwb_sigma_range
        self.sigma = lo + (hi - lo) * torch.rand(self.M, 1, generator=self.rng,
                                                 device=self.dev)
        _, _, _, gt0 = self.ds.get(self.ti, self.t)
        s0 = gt0 + cfg.init_pos_noise * torch.randn(self.M, cfg.state_dim,
                                                    generator=self.rng, device=self.dev)
        self.fme.reset(torch.arange(self.M, device=self.dev), s0)
        self.stack.zero_()
        for _ in range(cfg.warmup_steps):
            _, nu = self._epoch(self.default_N, self.default_l)
            self._push_feature(nu, self.default_N, self.default_l)
        self.step_in_ep = 0
        return self._obs()

    def step(self, a):
        cfg = self.cfg
        N, lam = self.map_action(a)
        s_hat, nu = self._epoch(N, lam)                      # one epoch = one RL step
        _, _, p_gt, _ = self.ds.get(self.ti, self.t)
        err = (p_gt - s_hat[:, 0:3]).norm(dim=1)             # L2 position error [m]
        reward = -torch.clamp(err, max=cfg.reward_clip)      # pure -||e|| + safety clip
        self._push_feature(nu, N, lam)
        self.step_in_ep += 1
        ep_end = self.step_in_ep >= cfg.episode_len
        done = torch.zeros(self.M, device=self.dev)          # truncation → bootstrap
        obs = self._obs()
        info = {"err": err, "N": N, "lam": lam, "ep_end": ep_end}
        return obs, reward, done, info
