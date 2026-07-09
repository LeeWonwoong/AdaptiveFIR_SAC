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
  - When a measurement row is invalid (NaN dropout), or the active
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
        self.n_rng = min(4, cfg.meas_dim)           # UWB range channel count
        self._alloc()

    def _alloc(self):
        M, W, nx, nz = self.M, self.W, self.nx, self.nz
        d = self.dev
        # lag-indexed EPOCH buffers: index j=0 newest
        self.A_buf = torch.zeros(M, W, nx, nx, device=d)   # Abar into epoch (k-j)
        self.u_buf = torch.zeros(M, W, nx, device=d)       # composed pseudo-input
        self.C_buf = torch.zeros(M, W, nz, nx, device=d)
        self.z_buf = torch.zeros(M, W, nz, device=d)       # pseudo-measurement
        self.sp_buf = torch.zeros(M, W, nx, device=d)      # anchor (prediction) state at epoch
        self.w_valid = torch.zeros(M, W, device=d)         # 1 if slot filled & row-valid (NaN dropout)
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
        self.sp_buf[idx] = 0.0
        self.w_valid[idx] = 0.0
        self.s_hat[idx] = s0
        self.filled[idx] = 0
        self.A_acc[idx] = self.eyeh
        self.u_acc[idx] = 0.0
        self.aux_P[idx] = 0.1 * self.eyeh
        # self_anchor ablation: hand over immediately (skip EKF warmup) so the
        # filter self-linearizes from t=0 with no stable seed — reproduces the
        # small-N divergence. Deployed filter (self_anchor=False) warms up.
        self.handed[idx] = bool(self.cfg.self_anchor)
        self.s_lin[idx] = s0

    # -------------------------------------------------- one filter step
    @torch.no_grad()
    def step(self, u_prev, z, N, lam, Np=None):
        """
        u_prev [M,4]    control acting from step k-1 -> k (base rate dt)
        z      [M,4]|None   UWB at k (None = prediction substep, no epoch)
        N, lam [M]      agent parameters (epochs / lag weight)
        Np     [M]|None mini-batch size (two-stage stage-1 length; None = N,
                        i.e. classical batch — backward compatible)
        returns (s_hat [M,12], nu [M,4]|None, s_pred [M,12])
        """
        m = self.model
        # ── LINEARIZATION ANCHOR (classical extended-FIR, self-sustaining) ──
        # The FIR estimator linearizes along its OWN delivered trajectory once
        # it has taken over (DI-FME eq.(7): A_t, C_t evaluated at the previous
        # FIR estimate s_hat_{t-1}). The auxiliary EKF is a WARMUP-ONLY device:
        # it provides the linearization anchor and the output ONLY until the
        # window is long enough to solve (handover), then it is switched OFF
        # and never runs again. After handover the horizon fills, step by step,
        # with Jacobians seeded by FIR estimates until — within N epochs — the
        # whole window is FIR-anchored and the filter is fully self-sustaining.
        #   pre-handover  (~handed): anchor = aux-EKF track   (s_lin)
        #   post-handover ( handed): anchor = own estimate    (s_hat)
        # per-env because each environment hands over at its own epoch.
        hb = self.handed.view(-1, 1)                    # [M,1]
        s_prev = torch.where(hb, self.s_hat, self.s_lin)
        A = m.jac_f(s_prev, u_prev)                     # [M,12,12]
        s_pred = m.f(s_prev, u_prev)
        u_t = s_pred - torch.bmm(A, s_prev.unsqueeze(-1)).squeeze(-1)
        self.A_acc = torch.bmm(A, self.A_acc)
        self.u_acc = torch.bmm(A, self.u_acc.unsqueeze(-1)).squeeze(-1) + u_t
        # tangent time-update of the DELIVERED state (pure model propagation
        # along the frozen linearization — used on substeps / rank-deficient)
        s_hold = torch.bmm(A, self.s_hat.unsqueeze(-1)).squeeze(-1) + u_t

        if z is None:                                   # ── prediction substep ──
            # advance the aux-EKF anchor track ONLY while still warming up;
            # once handed over the anchor rides the FIR estimate (s_hold).
            self.s_lin = torch.where(hb, self.s_lin, s_pred)
            s_hat = self._clamp(s_hold)
            self.s_hat = s_hat
            return s_hat, None, s_pred

        # ── measurement epoch ──
        C = m.jac_h(s_pred)                             # [M,4,12]
        nu = z - m.h(s_pred)                            # exact nonlinear innovation
        # per-row validity = FINITENESS ONLY: a dropped anchor (scenario)
        # arrives as a NaN range and is excluded ROW-WISE (DI-FME
        # intermittent-dropout handling), so a lost anchor never poisons the
        # other rows.
        # NO magnitude gate (REMOVED by design, 2026-07-09): the final
        # scenario mix (nominal / mass_step / sustained_wind / turbulence)
        # contains no NLoS outliers — a large UWB innovation there is
        # MODEL-ERROR signal (payload sag, wind push) that (a) the window LS
        # needs in order to correct the drifting prediction and (b) the SAC
        # policy needs as its observation; a magnitude gate would censor
        # both. The old gate_esc escalation existed only to un-stick the
        # gate's own all-rows-rejected blackout in the pre-IMU (dynamics-only
        # attitude-drift) era; with IMU fusion that failure mode is gone.
        anch_ok = torch.isfinite(nu)
        nu = torch.nan_to_num(nu, nan=0.0)
        aw = anch_ok.float().unsqueeze(-1)              # [M,nz,1]
        C = C * aw                                      # zero dropped rows
        z_t = nu + torch.bmm(C, s_pred.unsqueeze(-1)).squeeze(-1)
        z_t = z_t * anch_ok.float()                     # keep dropped rows at 0
        # epoch counts if at least one row survived (rank handled downstream)
        ok = anch_ok.any(dim=1).float()                 # [M]

        # push composed slot; reset accumulators
        self.A_buf = torch.roll(self.A_buf, 1, dims=1); self.A_buf[:, 0] = self.A_acc
        self.u_buf = torch.roll(self.u_buf, 1, dims=1); self.u_buf[:, 0] = self.u_acc
        self.C_buf = torch.roll(self.C_buf, 1, dims=1); self.C_buf[:, 0] = C
        self.z_buf = torch.roll(self.z_buf, 1, dims=1); self.z_buf[:, 0] = z_t
        self.sp_buf = torch.roll(self.sp_buf, 1, dims=1); self.sp_buf[:, 0] = s_pred
        self.w_valid = torch.roll(self.w_valid, 1, dims=1); self.w_valid[:, 0] = ok
        self.filled = torch.clamp(self.filled + 1, max=self.W)
        Aep = self.A_acc.clone()                        # epoch transition (for aux P)
        self.A_acc = self.eyeh.expand(self.M, self.nx, self.nx).clone()
        self.u_acc = torch.zeros(self.M, self.nx, device=self.dev)

        # ── auxiliary EKF — WARMUP ONLY ──
        # Runs to provide the pre-handover anchor + output. Frozen the instant
        # an environment hands over: no measurement update, no covariance
        # growth, no anchor advance afterwards (the FIR trajectory takes over).
        # We still compute it every epoch (cheap, static shapes) but MASK its
        # effect to pre-handover environments so a handed-over env's EKF state
        # can never leak back into the delivered estimate or the linearization.
        warm = (~self.handed).view(-1, 1, 1).float()    # [M,1,1] 1 while warming
        P_pred = torch.bmm(torch.bmm(Aep, self.aux_P),
                           Aep.transpose(1, 2)) + self.aux_Q
        Sm = torch.bmm(torch.bmm(C, P_pred), C.transpose(1, 2)) + self.aux_R
        K = torch.bmm(P_pred, torch.linalg.solve(Sm, C).transpose(1, 2))
        okc = ok.unsqueeze(1)
        s_ekf = s_pred + okc * torch.bmm(K, nu.unsqueeze(-1)).squeeze(-1)
        P_upd = torch.bmm(self.eyeh - torch.bmm(K, C), P_pred)
        aux_P_next = ok.view(-1, 1, 1) * P_upd + (1 - ok.view(-1, 1, 1)) * P_pred
        # freeze aux covariance once handed over
        self.aux_P = warm * aux_P_next + (1.0 - warm) * self.aux_P

        # pure FME solve on the epoch window
        if Np is None:                      # RL interface passes only (N, lam);
            Np = torch.full_like(N, float(getattr(self.cfg, "Np_fix", 4)))
        s_fme = self._solve(N.float(), lam, Np=Np)
        s_fme = torch.where(torch.isfinite(s_fme), s_fme, s_pred)   # numeric guard

        # handover latch: gain existence  filled_valid >= N_min (one-way, never
        # un-latches — once the FIR takes over it stays self-sustaining).
        fv = self.w_valid.sum(dim=1)
        was_handed = self.handed.clone()
        # handover once the buffer holds N_max valid epochs: every N in
        # [N_min, N_max] is then fully available (no growing-window transient,
        # no N clipping). Warmup runs N_max epochs so this is satisfied before
        # the first SAC action. (SEFFB: a horizon-N filter is used only when N
        # samples are buffered — here we wait for the largest N.)
        handover_len = float(getattr(self.cfg, "handover_len", self.cfg.N_max))
        self.handed = self.handed | (fv >= handover_len)

        # advance the aux-EKF ANCHOR track only for still-warming environments;
        # a handed-over env's linearization is carried by its own estimate and
        # must not be overwritten by the (now frozen) EKF.
        self.s_lin = torch.where(was_handed.view(-1, 1), self.s_lin, s_ekf)

        # ── PURE output rule ──
        #   pre-handover  -> aux EKF (warmup; the EKF's ONLY output duty)
        #   post-handover -> weighted-LS FME solve when the active window has
        #                    rank; a GATED current epoch is simply EXCLUDED
        #                    from the window (solve over remaining rows is still
        #                    pure FME); rank-deficient window -> tangent
        #                    time-update of the delivered state. No blending.
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

    def _solve(self, N, lam, Np=None):
        """
        DI-FME-original TWO-STAGE solve over the epoch window [m, k]:

          Stage 1 (mini-batch, eq.(18)):  the OLDEST Np epochs [m, i],
              i = m + Np - 1.  Weighted LS in the window-start frame,
              H_j = C_j Phi_{m,j},  y_j = z~_j - C_j d_j,
              w_j = lam^(lag_j - lag_i)  (so the global epoch weight is
              lam^(k-j) after stage-2 decay).  Solved by fp64 QR (the SAME
              pseudo-inverse as eq.(18), computed at condition kappa instead
              of kappa^2).  s_i = Phi_{m,i} s_m + d_i ;  information
              Om_i = Phi^{-T} (H^T W H) Phi^{-1}   (G_i = Om_i^{-1}).
          Stage 2 (iteration, eq.(19)-(22) / Shmaliy recursion, information
              form, extended with lam as per-epoch information decay):
              for l = i+1 .. k:
                s^- = Abar_l s + ubar_l ;  Om^- = lam * Abar^{-T} Om Abar^{-1}
                Om  = Om^- + C_l^T C_l ;   K = Om^{-1} C_l^T
                s   = s^- + K (z~_l - C_l s^-)
              gated/dropped epoch (no valid rows): s = s^-, Om = Om^-.

          Np = None or Np >= N  ->  mini-batch covers the whole window
          (classical batch FME; backward compatible with all callers).
        Linear-limit equivalence (Shmaliy S18-S24): stage-1+2 is the RLS
        factorization of the same weighted window LS, so deadbeat (T2) holds
        for any admissible (N, Np, lam). Vectorized; per-env (N, Np) masks.
        """
        M, W, nx, nz = self.M, self.W, self.nx, self.nz
        dev = self.dev
        Ab = self.A_buf.double()
        ub = self.u_buf.double()
        Cb = self.C_buf.double()
        zb = self.z_buf.double()
        eyeM = self.eyeh.double().expand(M, nx, nx)

        fv = self.w_valid.sum(dim=1).double()
        N_eff = torch.minimum(N.double().round(), fv).clamp(min=1.0)
        full_batch = (Np is not None) and bool((Np <= 0).all())
        if Np is None or full_batch:
            # None -> classical batch (=N);  <=0 sentinel -> full-batch mode:
            # mini-batch spans the WHOLE window, no stage-2 iteration.
            Np_eff = N_eff.clone()
        else:
            Np_eff = torch.minimum(Np.double().round().clamp(min=1.0), N_eff)
        start_lag = (N_eff.long() - 1).clamp(0, W - 1)              # window start m
        bound_lag = (N_eff.long() - Np_eff.long()).clamp(0, W - 1)  # mini-batch end i

        # ── pass A: chain + mini-batch rows (QR) + information accumulation ──
        Phi = eyeM.clone()
        d = torch.zeros(M, nx, device=dev, dtype=torch.float64)
        Macc = torch.zeros(M, nx, nx, device=dev, dtype=torch.float64)
        Phi_b = eyeM.clone()
        d_b = torch.zeros(M, nx, device=dev, dtype=torch.float64)
        s_bar = torch.zeros(M, nx, device=dev, dtype=torch.float64)  # anchor at window start
        Hs, ys = [], []
        vlag = self.w_valid.double()
        lam_d = lam.double().clamp(min=1e-3)

        for c in range(W):
            lag = W - 1 - c
            lag_t = torch.full((M,), lag, dtype=torch.long, device=dev)
            in_win = (lag_t <= start_lag)
            if c > 0:
                A = Ab[:, lag]
                Phi_n = torch.bmm(A, Phi)
                d_n = torch.bmm(A, d.unsqueeze(-1)).squeeze(-1) + ub[:, lag]
                Phi = torch.where(in_win.view(M, 1, 1), Phi_n, Phi)
                d = torch.where(in_win.view(M, 1), d_n, d)
            is_start = (lag_t == start_lag)
            Phi = torch.where(is_start.view(M, 1, 1), eyeM, Phi)
            d = torch.where(is_start.view(M, 1), torch.zeros_like(d), d)
            s_bar = torch.where(is_start.view(M, 1), self.sp_buf[:, lag].double(),
                                s_bar)

            C = Cb[:, lag]
            H = torch.bmm(C, Phi)
            y = zb[:, lag] - torch.bmm(C, d.unsqueeze(-1)).squeeze(-1)
            in_mb = in_win & (lag_t >= bound_lag)
            w_mb = torch.pow(lam_d, (lag_t - bound_lag).clamp(min=0).double())
            w_mb = torch.where(in_mb, w_mb * vlag[:, lag], torch.zeros_like(w_mb))
            sw = w_mb.sqrt().view(M, 1, 1)
            Hs.append(sw * H)
            ys.append(sw.squeeze(-1) * y)
            Macc = Macc + w_mb.view(M, 1, 1) * torch.bmm(H.transpose(1, 2), H)
            at_b = (lag_t == bound_lag) & in_win
            Phi_b = torch.where(at_b.view(M, 1, 1), Phi, Phi_b)
            d_b = torch.where(at_b.view(M, 1), d, d_b)

        Hb = torch.cat(Hs, dim=1)
        yb = torch.cat(ys, dim=1)
        # eq.(18) mini-batch estimate by the NORMAL EQUATIONS with a plain
        # inverse (NO SVD): solve (He^T He + rho I) delta = He^T y'.
        # DEVIATION solve around the anchor (linearization) trajectory:
        #   delta = s_m - s_bar,  y' = y - H s_bar.
        # In the observable subspace this is the identical LS solution (affine
        # equivariance -> unbiasedness/deadbeat unchanged, T2). The Tikhonov
        # ridge rho = ridge_eps * tr(He^T He)/ne GUARANTEES invertibility of
        # the observable-block normal matrix and leaves the near-null
        # (unobservable) directions at delta ~ 0 -> they follow the dynamics
        # propagation (the K->0 behavior of the original iterative form).
        ne = getattr(self.cfg, "est_dim", 6)   # measurement-corrected block [p, v]
        yb = yb - torch.bmm(Hb, s_bar.unsqueeze(-1)).squeeze(-1)
        He = Hb[:, :, 0:ne]                     # observable-block columns
        eyeE = torch.eye(ne, device=dev, dtype=torch.float64).expand(M, ne, ne)
        HtH = torch.bmm(He.transpose(1, 2), He)                    # [M,ne,ne]
        Hty = torch.bmm(He.transpose(1, 2), yb.unsqueeze(-1))      # [M,ne,1]
        rho = (self.eps * HtH.diagonal(dim1=1, dim2=2).sum(-1).div(ne)
               .clamp(min=1e-12))
        HtH_reg = HtH + rho.view(M, 1, 1) * eyeE                   # invertible
        d6 = torch.linalg.solve(HtH_reg, Hty).squeeze(-1)
        delta = torch.zeros(M, nx, device=dev, dtype=torch.float64)
        delta[:, 0:ne] = d6
        s_m = s_bar + delta

        # ── full-batch mode: Np <= 0 sentinel -> single-stage window LS IS the
        #    estimate; skip stage-2 entirely. With Np_eff == N_eff the batch
        #    spans the whole window, so Phi_b, d_b are the window-start (m)
        #    transition and s_k = Phi_b s_m + d_b.
        if full_batch:
            s_full = torch.bmm(Phi_b, s_m.unsqueeze(-1)).squeeze(-1) + d_b
            return s_full.float()
        s_run = torch.bmm(Phi_b, s_m.unsqueeze(-1)).squeeze(-1) + d_b
        # boundary information on the estimated block only (6x6):
        # Om6 = Phi6^{-T} Macc[0:ne,0:ne] Phi6^{-1}
        eyeE = torch.eye(ne, device=dev, dtype=torch.float64).expand(M, ne, ne)
        M6 = Macc[:, 0:ne, 0:ne]
        mscale = M6.diagonal(dim1=1, dim2=2).sum(-1).div(ne).clamp(min=1e-12)
        M6 = M6 + (self.eps * mscale).view(M, 1, 1) * eyeE
        P6 = Phi_b[:, 0:ne, 0:ne]
        X = torch.linalg.solve(P6.transpose(1, 2), M6)
        Om_run = torch.linalg.solve(P6.transpose(1, 2),
                                    X.transpose(1, 2)).transpose(1, 2)
        Om_run = 0.5 * (Om_run + Om_run.transpose(1, 2))

        # ── pass B: stage-2 iteration over lags < bound_lag ──
        it_any = (bound_lag > 0).any()
        if it_any:
            for c in range(W):
                lag = W - 1 - c
                lag_t = torch.full((M,), lag, dtype=torch.long, device=dev)
                in_it = (lag_t < bound_lag)
                if not in_it.any():
                    continue
                A = Ab[:, lag]
                C = Cb[:, lag]
                zt = zb[:, lag]
                ok = vlag[:, lag]
                ne = getattr(self.cfg, "est_dim", 6)
                # full 12-dim tangent propagation of the state
                s_pr = torch.bmm(A, s_run.unsqueeze(-1)).squeeze(-1) + ub[:, lag]
                # 6-dim error recursion on the estimated block: since the
                # attitude/rate deviations are structurally zero, their
                # coupling terms vanish and the error transition is A[0:6,0:6].
                A6 = A[:, 0:ne, 0:ne]
                C6 = C[:, :, 0:ne]
                Xo = torch.linalg.solve(A6.transpose(1, 2), Om_run)
                Om_pr = torch.linalg.solve(A6.transpose(1, 2),
                                           Xo.transpose(1, 2)).transpose(1, 2)
                Om_pr = lam_d.view(M, 1, 1) * 0.5 * (Om_pr + Om_pr.transpose(1, 2))
                Om_up = Om_pr + torch.bmm(C6.transpose(1, 2), C6)
                # K = (Om + rho I)^{-1} C6^T by a plain inverse (NO eigh):
                # the Tikhonov ridge guarantees invertibility; near-null
                # (unobservable) directions get vanishing gain -> propagation.
                eyeE2 = torch.eye(ne, device=dev, dtype=torch.float64).expand(M, ne, ne)
                rho2 = (self.eps * Om_up.diagonal(dim1=1, dim2=2).sum(-1).div(ne)
                        .clamp(min=1e-12))
                Om_reg = Om_up + rho2.view(M, 1, 1) * eyeE2
                Kt = torch.linalg.solve(Om_reg, C6.transpose(1, 2))
                innov = zt - torch.bmm(C, s_pr.unsqueeze(-1)).squeeze(-1)
                corr6 = torch.bmm(Kt, innov.unsqueeze(-1)).squeeze(-1)
                s_up = s_pr.clone()
                s_up[:, 0:ne] = s_pr[:, 0:ne] + corr6
                okv = (ok > 0).view(M, 1)
                s_new = torch.where(okv, s_up, s_pr)
                Om_new = torch.where(okv.view(M, 1, 1), Om_up, Om_pr)
                s_run = torch.where(in_it.view(M, 1), s_new, s_run)
                Om_run = torch.where(in_it.view(M, 1, 1), Om_new, Om_run)

        return s_run.float()
