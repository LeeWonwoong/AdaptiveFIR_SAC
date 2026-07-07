"""
filter/wfme.py — Time-weighted Finite Memory Estimator (PURE FME, masked batch)
================================================================================
Vectorized over M parallel environments. Fixed ring-buffer length W = N_max
(measurement EPOCHS) so all tensor shapes are static; the agent's (N_t, lam_t)
enter only through the lag-weight mask
        w_j = lam_t^j * 1[j < N_t],      j = 0 (newest epoch) .. W-1 (oldest).

PURE-FME contract (paper positioning):
  - The estimate is the weighted LS over the window rows ONLY — no prior, no
    blending with any other estimator. Unbiasedness Lemma: for any admissible
    (N, lam) with Omega > 0, K Hbar = Phi(k, m)  =>  E[s_hat] = s under the
    frozen linear model (verified numerically in tests, T2).
  - The tiny RELATIVE Tikhonov rows (eps ~ 1e-8) are numerical rank safety for
    the batched QR only; their bias is ~1e-8 and they are NOT a prior.
  - When a measurement is rejected by the innovation gate, or the active
    window lacks rank, the step outputs the propagated PREDICTION s_pred
    (time-update only) — standard filtering practice, no estimator switch.
  - The auxiliary EKF exists ONLY for the literature warmup: it serves until
    the gain-existence condition filled_valid >= N_min latches (handover),
    and never intervenes afterwards.

Measurement-epoch schedule (multirate, caller-driven):
  The plant/control chain runs at the base rate dt (50 Hz). UWB epochs arrive
  every `uwb_stride` steps (10 Hz). The CALLER decides the schedule by the
  argument z: step(u, z=None) is a prediction substep (compose the frozen
  chain, output s_pred); step(u, z) is an epoch (push one composed slot,
  solve). This resolves the short-window collinearity (v*t vs 0.5*g*th*t^2)
  within pure FME: the window spans N*stride*dt = 0.8..2.0 s in TIME while N
  in [8, 20] keeps its frozen semantics as the number of measurement epochs.

Per-epoch slot j holds the COMPOSED quantities over the stride substeps:
  Abar_j = A_{t_j} ... A_{t_{j-1}+1},   ubar_j = sum of chained pseudo-inputs,
  C_j, z~_j at the epoch time  =>  the decimated system is linear again and
  the classical FME algebra applies unchanged (anchored at the ACTIVE window
  start m = k - N_t + 1, per-env N via a Phi/d reset mask, fp64 weighted QR).

Warmup / handover (condition-based, not time-based): aux EKF serves ONLY
until filled_valid >= N_min (~N_min epochs); afterwards the filter serves the
growing window N_eff = min(N_agent, filled_valid), reaching the full
[N_min, N_max] range within N_max epochs (T5).
"""
import torch
from .uav_model import UAVModel


class WeightedFME:
    def __init__(self, cfg, device, M):
        self.cfg = cfg
        self.dev = device
        self.M = M
        self.W = cfg.N_max                 # ring length in measurement EPOCHS
        self.nx = cfg.state_dim
        self.nz = cfg.meas_dim
        self.model = UAVModel(cfg, device)
        self.eps = cfg.ridge_eps
        sig = torch.tensor(getattr(cfg, "meas_sigma", (1.0,) * cfg.meas_dim),
                           dtype=torch.float32, device=device)
        assert sig.numel() == cfg.meas_dim, "meas_sigma length != meas_dim"
        self.Dw = (1.0 / sig)                       # channel whitening weights
        self.n_rng = min(4, cfg.meas_dim)           # gate applies to range channels
        self._alloc()

    def _alloc(self):
        M, W, nx, nz = self.M, self.W, self.nx, self.nz
        d = self.dev
        # lag-indexed EPOCH buffers: index j=0 newest
        self.A_buf = torch.zeros(M, W, nx, nx, device=d)   # Abar into epoch (k-j)
        self.u_buf = torch.zeros(M, W, nx, device=d)       # composed pseudo-input
        self.C_buf = torch.zeros(M, W, nz, nx, device=d)
        self.z_buf = torch.zeros(M, W, nz, device=d)       # pseudo-measurement
        self.w_valid = torch.zeros(M, W, device=d)         # 1 if slot filled & not gated
        self.s_hat = torch.zeros(M, nx, device=d)
        self.filled = torch.zeros(M, dtype=torch.long, device=d)
        self.lag = torch.arange(W, device=d).float()
        self.eyeh = torch.eye(nx, device=d)
        # substep chain accumulators (composed since the last epoch)
        self.A_acc = self.eyeh.expand(M, nx, nx).clone()
        self.u_acc = torch.zeros(M, nx, device=d)
        # auxiliary EKF — WARMUP ONLY (pre-handover); never used afterwards
        self.aux_P = 0.1 * self.eyeh.expand(M, nx, nx).clone()
        q = [1e-6] * 3 + [5e-4] * 3 + [1e-5] * 3 + [5e-4] * 3
        self.aux_Q = torch.diag(torch.tensor(q, device=d)) * max(1, self.cfg.uwb_stride)
        self.aux_R = torch.diag(torch.tensor(
            getattr(self.cfg, "meas_sigma", (0.05,) * nz),
            dtype=torch.float32, device=d) ** 2)
        self.handed = torch.zeros(M, dtype=torch.bool, device=d)   # handover latch
        self.s_lin = torch.zeros(M, nx, device=d)    # linearization anchor track (aux EKF)

    # -------------------------------------------------- reset
    def reset(self, idx, s0):
        self.A_buf[idx] = 0.0
        self.u_buf[idx] = 0.0
        self.C_buf[idx] = 0.0
        self.z_buf[idx] = 0.0
        self.w_valid[idx] = 0.0
        self.s_hat[idx] = s0
        self.filled[idx] = 0
        self.A_acc[idx] = self.eyeh
        self.u_acc[idx] = 0.0
        self.aux_P[idx] = 0.1 * self.eyeh
        self.handed[idx] = False
        self.s_lin[idx] = s0

    # -------------------------------------------------- one filter step
    @torch.no_grad()
    def step(self, u_prev, z, N, lam):
        """
        u_prev [M,4]    control acting from step k-1 -> k (base rate dt)
        z      [M,4]|None   UWB at k (None = prediction substep, no epoch)
        N, lam [M]      agent parameters (epochs / lag weight)
        returns (s_hat [M,12], nu [M,4]|None, s_pred [M,12])
        """
        m = self.model
        # frozen linearization along the ANCHOR track (aux-EKF trajectory,
        # EFIR practice); self_anchor=True = paper-faithful ablation where the
        # delivered estimate itself seeds the linearization (DI-FME eq.(7)).
        s_prev = self.s_hat if self.cfg.self_anchor else self.s_lin
        A = m.jac_f(s_prev, u_prev)                     # [M,12,12]
        s_pred = m.f(s_prev, u_prev)
        u_t = s_pred - torch.bmm(A, s_prev.unsqueeze(-1)).squeeze(-1)
        self.A_acc = torch.bmm(A, self.A_acc)
        self.u_acc = torch.bmm(A, self.u_acc.unsqueeze(-1)).squeeze(-1) + u_t
        # tangent time-update of the DELIVERED state (pure model propagation
        # along the frozen linearization — used on substeps / rank-deficient)
        s_hold = torch.bmm(A, self.s_hat.unsqueeze(-1)).squeeze(-1) + u_t

        if z is None:                                   # ── prediction substep ──
            if not self.cfg.self_anchor:
                self.s_lin = s_pred                     # advance anchor track
            s_hat = self._clamp(s_pred if self.cfg.self_anchor else s_hold)
            self.s_hat = s_hat
            return s_hat, None, s_pred

        # ── measurement epoch ──
        C = m.jac_h(s_pred)                             # [M,4,12]
        nu = z - m.h(s_pred)                            # exact nonlinear innovation
        # per-anchor validity: dropped anchor (scenario) arrives as NaN range;
        # NLOS / outlier arrives as a gate-exceeding residual. Both are
        # excluded ROW-WISE (DI-FME intermittent-dropout handling), so a single
        # bad anchor never poisons the other three.
        anch_ok = torch.isfinite(nu) & (nu.abs() < self.cfg.innov_gate)   # [M,4]
        nu = torch.nan_to_num(nu, nan=0.0)
        aw = anch_ok.float().unsqueeze(-1)              # [M,4,1]
        C = C * aw                                      # zero dropped-anchor rows
        z_t = nu + torch.bmm(C, s_pred.unsqueeze(-1)).squeeze(-1)
        z_t = z_t * anch_ok.float()                     # keep dropped rows at 0
        # epoch counts if at least one anchor row survived (rank handled downstream)
        ok = anch_ok.any(dim=1).float()                 # [M]

        # push composed slot; reset accumulators
        self.A_buf = torch.roll(self.A_buf, 1, dims=1); self.A_buf[:, 0] = self.A_acc
        self.u_buf = torch.roll(self.u_buf, 1, dims=1); self.u_buf[:, 0] = self.u_acc
        self.C_buf = torch.roll(self.C_buf, 1, dims=1); self.C_buf[:, 0] = C
        self.z_buf = torch.roll(self.z_buf, 1, dims=1); self.z_buf[:, 0] = z_t
        self.w_valid = torch.roll(self.w_valid, 1, dims=1); self.w_valid[:, 0] = ok
        self.filled = torch.clamp(self.filled + 1, max=self.W)
        Aep = self.A_acc.clone()                        # epoch transition (for aux P)
        self.A_acc = self.eyeh.expand(self.M, self.nx, self.nx).clone()
        self.u_acc = torch.zeros(self.M, self.nx, device=self.dev)

        # auxiliary EKF — always advanced: it is the linearization anchor
        # track (and the pre-handover output). Gated epochs: time update only.
        P_pred = torch.bmm(torch.bmm(Aep, self.aux_P),
                           Aep.transpose(1, 2)) + self.aux_Q
        Sm = torch.bmm(torch.bmm(C, P_pred), C.transpose(1, 2)) + self.aux_R
        K = torch.bmm(P_pred, torch.linalg.solve(Sm, C).transpose(1, 2))
        okc = ok.unsqueeze(1)
        s_ekf = s_pred + okc * torch.bmm(K, nu.unsqueeze(-1)).squeeze(-1)
        P_upd = torch.bmm(self.eyeh - torch.bmm(K, C), P_pred)
        self.aux_P = ok.view(-1, 1, 1) * P_upd + (1 - ok.view(-1, 1, 1)) * P_pred
        if not self.cfg.self_anchor:
            self.s_lin = s_ekf

        # pure FME solve on the epoch window
        s_fme = self._solve(N.float(), lam)
        s_fme = torch.where(torch.isfinite(s_fme), s_fme, s_pred)   # numeric guard

        # handover latch: gain existence  filled_valid >= N_min
        fv = self.w_valid.sum(dim=1)
        self.handed = self.handed | (fv >= float(self.cfg.N_min))
        # PURE output rule: pre-handover -> aux EKF (literature warmup, the
        # aux filter's ONLY output duty); post-handover -> the weighted-LS
        # solve whenever the active window has rank (a GATED current epoch is
        # simply EXCLUDED from the window — the solve over the remaining rows
        # is still the pure FME, no output substitution); rank-deficient
        # window -> tangent time-update of the delivered state. No estimator
        # blending anywhere.
        N_eff = torch.minimum(N.float(), fv).clamp(min=1.0)
        rows = (self._weights(N_eff, lam) > 0).float().sum(dim=1)
        enough = (rows * self.nz >= self.nx)
        use_fme = self.handed & enough
        fallback = torch.where(self.handed.unsqueeze(1), s_hold, s_ekf)
        s_hat = torch.where(use_fme.unsqueeze(1), s_fme, fallback)
        s_hat = self._clamp(s_hat)

        self.s_hat = s_hat                               # delivered estimate (xyz claim)
        return s_hat, nu, s_pred

    # -------------------------------------------------- internals
    def _clamp(self, s):
        """physical projection of the weakly observable blocks (constrained
        estimation; keeps the next linearization point physical)."""
        if not self.cfg.state_clamp:
            return s
        c = self.cfg
        return torch.cat([s[:, 0:3],
                          s[:, 3:6].clamp(-c.clamp_vel, c.clamp_vel),
                          s[:, 6:8].clamp(-c.clamp_att, c.clamp_att),
                          s[:, 8:9],
                          s[:, 9:12].clamp(-c.clamp_rate, c.clamp_rate)], dim=1)

    def _weights(self, N, lam):
        """w_j = lam^j * 1[j < N] * valid   -> [M,W]  (j = epoch lag)"""
        j = self.lag.unsqueeze(0)
        mask = (j < N.unsqueeze(1)).float()
        return torch.pow(lam.unsqueeze(1).clamp(min=1e-3), j) * mask * self.w_valid

    def _solve(self, N, lam):
        """
        Chronological recursion over epoch slots (oldest c=0 .. newest c=W-1):
            Phi_0 = I, d_0 = 0 ;  Phi_c = Abar_c Phi_{c-1},  d_c = Abar_c d_{c-1} + ubar_c
        rows:  H_c = C_c Phi_c,  y_c = z~_c - C_c d_c    (sqrt-weighted, stacked)
        solve: fp64 weighted QR lstsq (+ eps Tikhonov rows for numerical rank
               safety only);  s_hat = Phi_final s0 + d_final.
        Anchor = ACTIVE window start m = k - N + 1 (classical FME); N clipped
        to filled_valid so the chain never touches empty slots (safe for any
        caller: step, oracle grids, probes).
        """
        M, W, nx, nz = self.M, self.W, self.nx, self.nz
        Ab = self.A_buf.double()
        ub = self.u_buf.double()
        Cb = self.C_buf.double()
        zb = self.z_buf.double()
        eye = self.eyeh.double()
        eyeM = eye.expand(M, nx, nx)
        Phi = eyeM.clone()
        d = torch.zeros(M, nx, device=self.dev, dtype=torch.float64)
        Hs, ys = [], []
        fv = self.w_valid.sum(dim=1).double()
        N_eff = torch.minimum(N.double().round(), fv).clamp(min=1.0)
        start_lag = (N_eff.long() - 1).clamp(0, W - 1)
        w = self._weights(N_eff.float(), lam).double()
        for c in range(W):
            lag = W - 1 - c
            if c > 0:
                A = Ab[:, lag]
                Phi = torch.bmm(A, Phi)
                d = torch.bmm(A, d.unsqueeze(-1)).squeeze(-1) + ub[:, lag]
            is_start = (start_lag == lag)
            Phi = torch.where(is_start.view(M, 1, 1), eyeM, Phi)
            d = torch.where(is_start.view(M, 1), torch.zeros_like(d), d)
            H = torch.bmm(Cb[:, lag], Phi)
            y = zb[:, lag] - torch.bmm(Cb[:, lag], d.unsqueeze(-1)).squeeze(-1)
            # sqrt(time weight) x channel whitening 1/sigma  (positive row
            # weights: the unbiasedness Lemma holds unchanged)
            sw = w[:, lag].sqrt().view(M, 1, 1) * self.Dw.double().view(1, -1, 1)
            Hs.append(sw * H)                        # sw: [M,nz,1] broadcast
            ys.append(sw.squeeze(-1) * y)
        Hb = torch.cat(Hs, dim=1)                        # [M, nz*W, 12]
        yb = torch.cat(ys, dim=1)
        # numerical-rank-safety rows ONLY (relative eps ~1e-8; not a prior)
        row_scale = Hb.pow(2).sum(dim=(1, 2)).div(nx).sqrt().clamp(min=1e-9)
        reg = (self.eps * row_scale).view(M, 1, 1) * eyeM
        Hb = torch.cat([Hb, reg], dim=1)
        yb = torch.cat([yb, torch.zeros(M, nx, device=self.dev,
                                        dtype=torch.float64)], dim=1)
        s0 = torch.linalg.lstsq(Hb, yb.unsqueeze(-1)).solution.squeeze(-1)
        s_hat = torch.bmm(Phi, s0.unsqueeze(-1)).squeeze(-1) + d
        return s_hat.float()
