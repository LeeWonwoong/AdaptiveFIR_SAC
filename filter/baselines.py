"""
filter/baselines.py — comparison estimators (batched over M trajectories)
==========================================================================
All support the multirate schedule: step(u, z) with z=None is a prediction
substep; z given is a measurement epoch (matching WeightedFME's contract).

  EKF        : extended Kalman filter (predict every substep, update on epochs)
  UKF        : additive unscented KF (same schedule)
  FixedFME   : WeightedFME with constant (N, lam)  — grid baseline
  DIFME      : Dynamic Iterative-FME (Lee et al.) — overrides ONLY _solve with
               the information-form iterative window recursion + dynamic gain
               alpha_t (gamma=0.3); shares the pure-FME step machinery
               (epoch slots, gate->prediction fallback, warmup aux EKF,
               clamp). Faithful reimplementation on the shared frozen-
               linearization buffers; verify against the original
               implementation before camera-ready.
  RuleFME    : innovation-ratio threshold rule — the hand-crafted adaptive
               competitor (switches responsive/smoothing presets).
All share UAVModel (nominal mass) so every method sees the same model mismatch.
"""
import torch
from .uav_model import UAVModel
from .wfme import WeightedFME


def proc_Q(cfg, acc=None, gyro=None):
    """Discrete process covariance from a plant accel/gyro σ. Per-epoch velocity
    variance from an accel white sequence over the dt/4 substeps ≈ σ_a² dt²/4.
    acc/gyro default to the TRUE nominal (config.proc_acc_std/proc_gyro_std);
    the practitioner KF passes UNDER-stated values."""
    dt = cfg.dt
    a = getattr(cfg, "proc_acc_std", 0.30) if acc is None else acc
    gg = getattr(cfg, "proc_gyro_std", 0.05) if gyro is None else gyro
    qa = (a ** 2) * dt * dt / 4.0
    qg = (gg ** 2) * dt * dt / 4.0
    return [1e-7] * 3 + [qa] * 3 + [1e-6] * 3 + [qg] * 3


def _kf_stats(cfg, oracle):
    """(q_diag, r_sigma) for a recursive baseline. oracle → correct statistics
    (Q from true q0, R from meas_sigma); else PRACTITIONER (Q under-scaled by
    ekf_Q_scale, R from the datasheet ekf_R_sigma)."""
    if oracle:
        return proc_Q(cfg), list(getattr(cfg, "meas_sigma", (0.12,) * cfg.meas_dim))
    sc = getattr(cfg, "ekf_Q_scale", 0.40)
    q = proc_Q(cfg, acc=cfg.proc_acc_std * sc, gyro=cfg.proc_gyro_std * sc)
    return q, [getattr(cfg, "ekf_R_sigma", 0.10)] * cfg.meas_dim


# ══════════════════════════════════════════════════════════════ EKF
class EKF:
    def __init__(self, cfg, device, M, q_diag=None, r_diag=None, oracle=False):
        self.cfg, self.dev, self.M = cfg, device, M
        self.model = UAVModel(cfg, device)
        nx = cfg.state_dim
        q0, r0 = _kf_stats(cfg, oracle)
        q = q_diag if q_diag is not None else q0
        self.Q = torch.diag(torch.tensor(q, device=device))
        sig = r_diag if r_diag is not None else r0
        self.R = torch.diag(torch.tensor(sig, device=device) ** 2)
        self.eye = torch.eye(nx, device=device)
        self.s = torch.zeros(M, nx, device=device)
        self.P = 0.1 * self.eye.expand(M, nx, nx).clone()

    def reset(self, idx, s0):
        self.s[idx] = s0
        self.P[idx] = 0.1 * self.eye

    @torch.no_grad()
    def step(self, u_prev, z, *_):
        m = self.model
        A = m.jac_f(self.s, u_prev)
        s_pred = m.f(self.s, u_prev)
        # fp64 covariance path + symmetrization (same hardening as wfme aux-EKF):
        # fp32 (I-KC)P intermittently loses positive-definiteness on CUDA ->
        # one-step NaN blowups observed in 7/50 eval rollouts while tracking at
        # 0.02-0.4 m error. fp64 + P=(P+P^T)/2 removes the failure mode.
        Ad = A.double()
        Pd = torch.bmm(torch.bmm(Ad, self.P.double()), Ad.transpose(1, 2)) \
            + self.Q.double()
        self.P = (0.5 * (Pd + Pd.transpose(1, 2))).float()
        if z is None:                                   # prediction substep
            self.s = s_pred
            return self.s, None, s_pred
        C = m.jac_h(s_pred)
        nu = z - m.h(s_pred)
        # [수정B] per-anchor dropout: a missing anchor arrives as NaN range.
        # Zero its measurement row → that channel does prediction-only (no NaN
        # poisoning). ALL anchors missing → C=0 → K=0 → pure time update, so the
        # WRONG nominal model integrates unchecked (honest divergence). NOTE the
        # EKF keeps its FIXED R (=LoS σ, [수정C]): on an NLoS burst nu is finite
        # so the corrupted anchor is FULLY trusted → error amplifies.
        valid = torch.isfinite(nu)                       # [M,4]
        nu = torch.nan_to_num(nu, nan=0.0)
        C = C * valid.float().unsqueeze(-1)              # drop missing rows
        Cd, Pd = C.double(), self.P.double()
        S = torch.bmm(torch.bmm(Cd, Pd), Cd.transpose(1, 2)) + self.R.double()
        K = torch.bmm(Pd, torch.linalg.solve(S, Cd).transpose(1, 2))
        self.s = s_pred + torch.bmm(K, nu.double().unsqueeze(-1)).squeeze(-1).float()
        Pu = torch.bmm(self.eye.double() - torch.bmm(K, Cd), Pd)
        self.P = (0.5 * (Pu + Pu.transpose(1, 2))).float()
        return self.s, nu, s_pred


# ══════════════════════════════════════════════════════════════ UKF
class UKF:
    def __init__(self, cfg, device, M, q_diag=None, r_diag=None,
                 alpha=0.5, beta=2.0, kappa=0.0, oracle=False):
        self.cfg, self.dev, self.M = cfg, device, M
        self.model = UAVModel(cfg, device)
        nx, nz = cfg.state_dim, cfg.meas_dim
        self.nx, self.nz = nx, nz
        q0, r0 = _kf_stats(cfg, oracle)
        q = q_diag if q_diag is not None else q0
        self.Q = torch.diag(torch.tensor(q, device=device))
        sig = r_diag if r_diag is not None else r0
        self.R = torch.diag(torch.tensor(sig, device=device) ** 2)
        lam = alpha ** 2 * (nx + kappa) - nx
        self.lam = lam
        self.ns = 2 * nx + 1
        Wm = torch.full((self.ns,), 1.0 / (2 * (nx + lam)), device=device)
        Wc = Wm.clone()
        Wm[0] = lam / (nx + lam)
        Wc[0] = lam / (nx + lam) + (1 - alpha ** 2 + beta)
        self.Wm, self.Wc = Wm, Wc
        self.s = torch.zeros(M, nx, device=device)
        self.P = 0.1 * torch.eye(nx, device=device).expand(M, nx, nx).clone()

    def reset(self, idx, s0):
        self.s[idx] = s0
        self.P[idx] = 0.1 * torch.eye(self.nx, device=self.dev)

    def _sigma(self, s, P):
        nx = self.nx
        jitter = 1e-8 * torch.eye(nx, device=self.dev)
        L = torch.linalg.cholesky((self.lam + nx) * (P + jitter))
        pts = [s]
        for i in range(nx):
            pts.append(s + L[:, :, i])
        for i in range(nx):
            pts.append(s - L[:, :, i])
        return torch.stack(pts, dim=1)                       # [M,ns,nx]

    @torch.no_grad()
    def step(self, u_prev, z, *_):
        m, M, nx, nz = self.model, self.M, self.nx, self.nz
        X = self._sigma(self.s, self.P)
        Xf = m.f(X.reshape(-1, nx),
                 u_prev.unsqueeze(1).expand(-1, self.ns, -1).reshape(-1, 4)
                 ).reshape(M, self.ns, nx)
        s_pred = (self.Wm.view(1, -1, 1) * Xf).sum(1)
        dX = Xf - s_pred.unsqueeze(1)
        P_pred = torch.einsum("s,msi,msj->mij", self.Wc, dX, dX) + self.Q
        if z is None:                                   # prediction substep
            self.s, self.P = s_pred, P_pred
            return self.s, None, s_pred
        Zf = m.h(Xf.reshape(-1, nx)).reshape(M, self.ns, nz)
        z_pred = (self.Wm.view(1, -1, 1) * Zf).sum(1)
        dZ = Zf - z_pred.unsqueeze(1)
        # [수정B] per-anchor dropout: zero the missing channels' innovation
        # deviations → they contribute only R to Pzz (invertible) and 0 to Pxz,
        # so K has no column there and nu is 0 → prediction-only on missing
        # channels (all missing → K=0 → pure time update).
        valid = torch.isfinite(z)                        # [M,nz]
        dZ = dZ * valid.float().unsqueeze(1)             # zero missing-channel devs
        nu = torch.nan_to_num(z - z_pred, nan=0.0)
        Pzz = torch.einsum("s,msi,msj->mij", self.Wc, dZ, dZ) + self.R
        Pxz = torch.einsum("s,msi,msj->mij", self.Wc, dX, dZ)
        K = torch.linalg.solve(Pzz, Pxz.transpose(1, 2)).transpose(1, 2)
        self.s = s_pred + torch.bmm(K, nu.unsqueeze(-1)).squeeze(-1)
        self.P = P_pred - torch.bmm(torch.bmm(K, Pzz), K.transpose(1, 2))
        return self.s, nu, s_pred


# ══════════════════════════════════════════════════════════════ Fixed FME
class FixedFME(WeightedFME):
    def __init__(self, cfg, device, M, N=None, lam=None):
        super().__init__(cfg, device, M)
        self.N_fix = torch.full((M,), float(N if N else cfg.N_default), device=device)
        self.l_fix = torch.full((M,), float(lam if lam else cfg.lam_default), device=device)

    def step(self, u_prev, z, *_):
        return super().step(u_prev, z, self.N_fix, self.l_fix)


# ══════════════════════════════════════════════════════════════ DI-FME
class DIFME(WeightedFME):
    """Dynamic Iterative-FME (Lee et al.): the window estimate is re-derived
    each epoch via the information-form recursion — batch init on the oldest
    Np epoch slots (uniform LS), then iterative updates with dynamic-gain
    blending
        alpha_t = 1 / (1 + gamma * ||s_minus - s_cur||^2)
        G*      = (1 - alpha) G_prev + alpha G_new
    Implemented as a _solve override on the shared pure-FME step machinery
    (same epoch slots, gate/rank fallbacks, warmup aux EKF, clamp), so the
    comparison isolates exactly the estimator recursion. N fixed = N_default,
    gamma = 0.3 (values verified from the DI-FME paper)."""

    def __init__(self, cfg, device, M, N=None, gamma=0.3, Np=4):
        super().__init__(cfg, device, M)
        self.Nwin = int(N if N else cfg.N_default)
        self.gamma = gamma
        self.Np = Np
        self.N_fix = torch.full((M,), float(self.Nwin), device=device)
        self.l_fix = torch.full((M,), 1.0, device=device)

    def step(self, u_prev, z, *_):
        return super().step(u_prev, z, self.N_fix, self.l_fix)

    @torch.no_grad()
    def _solve(self, N, lam, Np=None):
        Mb, W, nx, nz = self.M, self.W, self.nx, self.nz
        eye = self.eyeh
        fv = self.w_valid.sum(dim=1)
        Nw = int(min(self.Nwin, max(1.0, float(fv.min().item()))))
        Np = min(self.Np, Nw)
        Ab, ub, Cb, zb = self.A_buf, self.u_buf, self.C_buf, self.z_buf
        # ── batch init: uniform LS over the OLDEST Np epoch slots ──
        Phi = eye.expand(Mb, nx, nx).clone()
        d = torch.zeros(Mb, nx, device=self.dev)
        S = torch.zeros(Mb, nx, nx, device=self.dev)
        b = torch.zeros(Mb, nx, device=self.dev)
        init_lags = list(range(Nw - 1, Nw - 1 - Np, -1))     # oldest -> newer
        for k, lag in enumerate(init_lags):
            if k > 0:
                Ai = Ab[:, lag]
                Phi = torch.bmm(Ai, Phi)
                d = torch.bmm(Ai, d.unsqueeze(-1)).squeeze(-1) + ub[:, lag]
            H = torch.bmm(Cb[:, lag], Phi)
            y = zb[:, lag] - torch.bmm(Cb[:, lag], d.unsqueeze(-1)).squeeze(-1)
            wv = self.w_valid[:, lag].view(Mb, 1, 1)
            S = S + wv * torch.bmm(H.transpose(1, 2), H)
            b = b + wv.squeeze(-1) * torch.bmm(H.transpose(1, 2),
                                               y.unsqueeze(-1)).squeeze(-1)
        md = S.diagonal(dim1=1, dim2=2).mean(dim=1).clamp(min=1e-9).view(-1, 1, 1)
        S = S + 1e-3 * md * eye                              # relative jitter
        s0 = torch.linalg.solve(S, b.unsqueeze(-1)).squeeze(-1)
        s_i = torch.bmm(Phi, s0.unsqueeze(-1)).squeeze(-1) + d
        G = torch.linalg.inv(S)
        G_prev = G.clone()
        # ── iterative information-form updates over the newer epochs ──
        for lag in range(Nw - Np - 1, -1, -1):
            Ai, Ci, zi = Ab[:, lag], Cb[:, lag], zb[:, lag]
            s_minus = torch.bmm(Ai, s_i.unsqueeze(-1)).squeeze(-1) + ub[:, lag]
            AGA = torch.bmm(torch.bmm(Ai, G), Ai.transpose(1, 2))
            mg = AGA.diagonal(dim1=1, dim2=2).mean(dim=1).clamp(min=1e-9).view(-1, 1, 1)
            Ginfo = torch.bmm(Ci.transpose(1, 2), Ci) + \
                torch.linalg.inv(AGA + 1e-6 * mg * eye)
            mi = Ginfo.diagonal(dim1=1, dim2=2).mean(dim=1).clamp(min=1e-9).view(-1, 1, 1)
            G_new = torch.linalg.inv(Ginfo + 1e-6 * mi * eye)
            dss = (s_minus - s_i).pow(2).sum(dim=1)
            alpha = (1.0 / (1.0 + self.gamma * dss)).view(-1, 1, 1)
            G_star = (1 - alpha) * G_prev + alpha * G_new
            innov = zi - torch.bmm(Ci, s_minus.unsqueeze(-1)).squeeze(-1)
            K = torch.bmm(G_star, Ci.transpose(1, 2))
            wv = self.w_valid[:, lag].view(-1, 1)
            s_i = s_minus + wv * torch.bmm(K, innov.unsqueeze(-1)).squeeze(-1)
            G_prev, G = G_new, G_new
        return s_i


# ══════════════════════════════════════════════════════════════ rule-based
class RuleFME(WeightedFME):
    """NIS-style threshold rule: if short/long innovation-norm ratio exceeds
    `thresh`, switch to the responsive preset (N_min, lam_min); else the
    smoothing preset (N_max, 1.0). The obvious hand-crafted competitor."""

    def __init__(self, cfg, device, M, thresh=2.0, K_s=3, K_l=30):
        super().__init__(cfg, device, M)
        self.thresh = thresh
        self.short = torch.zeros(M, K_s, device=device)
        self.long = torch.zeros(M, K_l, device=device)
        self.last_N = torch.full((M,), float(cfg.N_default), device=device)
        self.last_lam = torch.full((M,), 1.0, device=device)

    def reset(self, idx, s0):
        super().reset(idx, s0)
        self.short[idx] = 0.0
        self.long[idx] = 0.0

    def step(self, u_prev, z, *_):
        # decide from PREVIOUS epochs' innovations (causal)
        s_mean = self.short.mean(dim=1)
        l_mean = self.long.mean(dim=1).clamp(min=1e-6)
        hot = (s_mean / l_mean > self.thresh).float()
        N = hot * self.cfg.N_min + (1 - hot) * self.cfg.N_max
        lam = hot * self.cfg.lam_min + (1 - hot) * 1.0
        s_hat, nu, s_pred = super().step(u_prev, z, N, lam)
        if nu is not None:                              # epoch: update stats
            self.last_N, self.last_lam = N, lam
            nn = torch.linalg.vector_norm(nu, dim=1)
            self.short = torch.roll(self.short, 1, 1); self.short[:, 0] = nn
            self.long = torch.roll(self.long, 1, 1); self.long[:, 0] = nn
        return s_hat, nu, s_pred
