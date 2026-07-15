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
        self.n_rng = len(cfg.anchors)                     # UWB channel count (4)
        if cfg.obs_channel_norms:
            assert cfg.meas_dim == self.n_rng + 6, \
                "channel-norm obs assumes z = [UWB x4 | attitude x3 | gyro x3]"
            self.feat = cfg.n_obs_groups + 2              # [g_uwb,g_att,g_gyr,N,lam]
            # obs_group_scale may carry 3 entries (UWB, att, gyro) for backward
            # compat; when the gyro group is dropped we only keep the first
            # n_obs_groups so the width matches the 2-channel head.
            self.grp_scale = torch.tensor(
                cfg.obs_group_scale[:cfg.n_obs_groups], device=device).view(1, -1)
        else:
            self.feat = cfg.meas_dim + 2                  # legacy per-channel width
        self.fme = WeightedFME(cfg, device, self.M)
        # DI-FME 기준선 필터 (동일 측정 스트림, 고정 N/lam) — advantage 보상용
        self.ref = WeightedFME(cfg, device, self.M) if cfg.ref_monitor_N > 0 else None
        self.refN = torch.full((self.M,), float(max(cfg.ref_monitor_N, cfg.N_min)),
                               device=device)
        self.refL = torch.ones(self.M, device=device)
        self._s_ref = None
        self._prevN = None
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
        """noisy z [M,meas_dim] = [4 UWB ranges | 3 attitude | 3 gyro].
        UWB part is UNCHANGED: per-episode sigma randomization + NLoS
        scale/bias + dropout-NaN (NaN row -> the filter's innovation gate
        excludes it, DI-FME intermittent-dropout handling).
        IMU part (meas_dim=10 fusion mode): synthesized ONLINE from the GT
        log. run_datagen records the drone's TRUE attitude/rates under real
        physical disturbance (set_masses payload / apply_forces wind), so the
        disturbance response is physically present in these channels; only
        the SENSOR noise (FCU attitude est. / gyro grade, cfg.meas_sigma[4:])
        is added here."""
        _, rc, _, gt = self.ds.get(self.ti, self.t)
        scale = 1.0
        bias = 0.0
        if hasattr(self.ds, "noise_scale"):                 # NLoS per-anchor σ↑
            scale = self.ds.noise_scale[self.ti, self.t]    # [M,4]
        if hasattr(self.ds, "range_bias"):                  # NLoS multipath bias
            bias = self.ds.range_bias[self.ti, self.t]      # [M,4]
        z_uwb = rc + bias + self.sigma * scale * torch.randn(
            self.M, rc.shape[1], generator=self.rng, device=self.dev)
        if self.nz <= rc.shape[1]:                          # legacy UWB-only mode
            return z_uwb
        sig_imu = self.meas_sig[rc.shape[1]:self.nz]        # [6] att3 + gyro3
        z_imu = gt[:, 6:12] + sig_imu * torch.randn(
            self.M, self.nz - rc.shape[1], generator=self.rng, device=self.dev)
        return torch.cat([z_uwb, z_imu], dim=1)

    def _epoch(self, N, lam):
        stride = max(1, self.cfg.uwb_stride)
        nu = None
        for i in range(stride):
            self.t += 1
            up, _, _, _ = self.ds.get(self.ti, self.t)
            z = self._measure() if i == stride - 1 else None
            s_hat, nu_i, _ = self.fme.step(up, z, N, lam)
            if self.ref is not None:
                self._s_ref, _, _ = self.ref.step(up, z, self.refN, self.refL)
            if nu_i is not None:
                nu = nu_i
        return s_hat, nu

    # ────────────────────────────── observation assembly
    def _push_feature(self, nu, N, lam):
        c = self.cfg.resid_clip
        r = torch.nan_to_num(nu / self.meas_sig, nan=0.0)   # whitened, UNclipped
        if self.cfg.obs_channel_norms:
            # per-GROUP whitened innovation norms (user option 3): the policy
            # sees "how wrong is each sensor FAMILY" scale-free, so it can
            # tell UWB trouble (NLoS/dropout) from dynamics trouble (payload/
            # wind -> attitude+gyro innovations rise; measured x2.1 in window).
            nr = self.n_rng
            g_uwb = r[:, 0:nr].norm(dim=1, keepdim=True) / (nr ** 0.5)
            g_att = r[:, nr:nr + 3].norm(dim=1, keepdim=True) / (3.0 ** 0.5)
            if getattr(self.cfg, "obs_drop_gyro", False):
                # GYRO DROPPED from the observation (2026-07-15). The gyro
                # innovation carries almost no disturbance information (mean
                # Cohen's d ~0.4 vs UWB ~1.9 across all patterns/scenarios):
                # it measures angular RATE, whereas wind/payload induce a
                # low-frequency attitude BIAS that differentiation misses,
                # while amplifying manoeuvre noise. The WFME still USES the
                # gyro measurement for state estimation; it is removed only
                # from the policy observation.
                head = torch.cat([g_uwb, g_att], dim=1)              # [M,2]
            else:
                g_gyr = r[:, nr + 3:nr + 6].norm(dim=1, keepdim=True) / (3.0 ** 0.5)
                head = torch.cat([g_uwb, g_att, g_gyr], dim=1)       # [M,3]
            # normalize by the Isaac-measured CALM-FLIGHT innovation level, so
            # mu reads as a multiple of nominal (1 = calm, 2-4 = disturbance).
            head = head / self.grp_scale
            if getattr(self.cfg, "obs_squared_stat", False):
                # frozen-NIS: eps = nu' Sbar^{-1} nu / d  (see config note);
                # squaring the gbar-normalised RMS norm gives exactly that.
                head = head ** 2
            if getattr(self.cfg, "obs_log_compress", False):
                # LOG COMPRESSION (2026-07-13), replaces the hard clip:
                #   mu~ = log(1+mu) / (denom + log(1+mu))  in [0,1)
                # bounded (no clip needed), MONOTONE (a mu=22 outlier stays
                # distinguishable from mu=4, which the old clip destroyed), and
                # tail-compressing (a single agile-flight transient -- attitude
                # p99 ~ 17 -- cannot dominate the input). mu is ratio-scale, so
                # the log is the natural transform.
                lg = torch.log1p(head.clamp(min=0.0))
                head = lg / (float(self.cfg.obs_log_denom) + lg)
            else:
                head = head.clamp(max=c)                  # legacy hard clip
        else:
            head = r.clamp(-c, c)                                     # legacy
        feat = torch.cat([head, self._norm_N(N).unsqueeze(1),
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
        if self.ref is not None:
            self.ref.reset(torch.arange(self.M, device=self.dev), s0.clone())
        self._prevN = torch.full((self.M,), float(self.default_N), device=self.dev) \
            if not torch.is_tensor(self.default_N) else \
            self.default_N.clone().float().expand(self.M).clone()
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
        # ── reward: ABSOLUTE error only (user decision — no baseline terms).
        ec = torch.clamp(err, max=cfg.reward_clip)
        if getattr(cfg, "reward_mode", "sq") == "sq":
            # normalized quadratic: (e/e0)^2 with e0 ~ nominal error level.
            # For e<1 a RAW square SHRINKS the absolute signal (0.4^2-0.25^2
            # = 0.10/step) to the same order as the entropy term (alpha=0.08);
            # dividing by e0=0.2 restores it (~2.4/step) without changing the
            # optimal policy (positive scaling).
            e0 = getattr(cfg, "reward_err_scale", 0.2)
            rn = torch.clamp(err, max=2.0) / e0
            reward = -torch.clamp(rn * rn,
                                  max=getattr(cfg, "reward_sq_clip", 100.0))
        else:
            reward = -ec             # legacy linear
        # DI-FME parallel filter = MONITOR ONLY (logging / eval table), not reward:
        err_ref = (p_gt - self._s_ref[:, 0:3]).norm(dim=1) \
            if self.ref is not None else err
        if cfg.act_smooth_coef > 0:                          # regime steps, not jitter
            reward = reward - cfg.act_smooth_coef \
                * (N.float() - self._prevN).abs() / max(cfg.N_max - cfg.N_min, 1)
        self._prevN = N.float().clone()
        self._push_feature(nu, N, lam)
        self.step_in_ep += 1
        ep_end = self.step_in_ep >= cfg.episode_len
        done = torch.zeros(self.M, device=self.dev)          # truncation → bootstrap
        obs = self._obs()
        info = {"err": err, "err_ref": err_ref, "N": N, "lam": lam, "ep_end": ep_end}
        return obs, reward, done, info
