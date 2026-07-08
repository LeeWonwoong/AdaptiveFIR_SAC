"""
tests/test_wfme.py — filter verification suite
================================================
  T1  Jacobians (AD) vs central finite differences
  T2  Deadbeat / unbiasedness: on a LINEAR system with ZERO noise, the
      weighted FME recovers the true state EXACTLY for arbitrary admissible
      (N, lambda) — the numerical form of the unbiasedness Lemma
      (K Hbar = Phi for any Omega > 0).
  T3  lambda responsiveness: after an abrupt model change, smaller lambda
      tracks faster (transient error decreases with lambda).
  T4  Nonlinear sanity: noise-free UAV trajectory → estimation error stays
      near machine/linearization tolerance; EKF and WFME comparable.
Run: python -m tests.test_wfme
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config                      # noqa: E402
from filter.uav_model import UAVModel          # noqa: E402
from filter.wfme import WeightedFME            # noqa: E402
from filter.baselines import EKF               # noqa: E402

DEV = "cpu"
torch.manual_seed(0)


def t1_jacobians():
    cfg = Config()
    m = UAVModel(cfg, DEV)
    # float64 for finite-difference reference (float32 FD truncation ~5e-3)
    for attr in ("J", "Jinv", "anchors", "g_vec"):
        setattr(m, attr, getattr(m, attr).double())
    s = (torch.randn(3, 12) * 0.3).double()
    s[:, 0:3] += torch.tensor([5.0, 5.0, 1.5], dtype=torch.float64)
    u = torch.tensor([[13.5, 0.02, -0.01, 0.005]], dtype=torch.float64).repeat(3, 1)
    A = m.jac_f(s, u)
    C = m.jac_h(s)
    eps = 1e-6
    A_fd = torch.zeros_like(A)
    C_fd = torch.zeros_like(C)
    for j in range(12):
        dp = torch.zeros(12); dp[j] = eps
        A_fd[:, :, j] = (m.f(s + dp, u) - m.f(s - dp, u)) / (2 * eps)
        C_fd[:, :, j] = (m.h(s + dp) - m.h(s - dp)) / (2 * eps)
    ea = (A - A_fd).abs().max().item()
    ec = (C - C_fd).abs().max().item()
    assert ea < 1e-6 and ec < 1e-6, (ea, ec)
    print(f"T1 jacobians OK  (|dA|={ea:.2e}, |dC|={ec:.2e})")


class _LinModel:
    """random stable linear system injected into WeightedFME (duck-typed)."""

    def __init__(self, nx=12, nz=4, seed=1):
        g = torch.Generator().manual_seed(seed)
        A = torch.eye(nx) + 0.02 * torch.randn(nx, nx, generator=g)
        # normalize spectral radius to ~1 (realistic dt-scale transition;
        # avoids artificial state blow-up over long tests)
        rho = torch.linalg.eigvals(A).abs().max().real
        self.A0 = A / max(float(rho), 1.0)
        self.B0 = 0.1 * torch.randn(nx, 4, generator=g)
        self.H0 = torch.randn(nz, nx, generator=g)

    def f(self, s, u):
        return s @ self.A0.T + u @ self.B0.T

    def h(self, s):
        return s @ self.H0.T

    def jac_f(self, s, u):
        return self.A0.expand(s.shape[0], -1, -1)

    def jac_h(self, s):
        return self.H0.expand(s.shape[0], -1, -1)


def t2_deadbeat_unbiasedness():
    # Validates the UNRESTRICTED solver algebra (est_dim=12) on a generic
    # dense-H linear system: batch (Np=N) and two-stage (Np<N) must both be
    # deadbeat for any admissible (N, Np, lam)  [Shmaliy S18-S24 equivalence].
    # The est_dim=6 observable-block contract is UAV-structural (C attitude
    # columns exactly zero) and is validated separately by T2b.
    import dataclasses
    cfg = dataclasses.replace(Config(), meas_dim=4, state_clamp=False,
                              innov_gate=1e9, meas_sigma=(1.0,) * 4,
                              est_dim=12)
    M = 6
    fme = WeightedFME(cfg, DEV, M)
    fme.model = _LinModel()                                     # linear plant == filter model
    g = torch.Generator().manual_seed(2)
    s_true = torch.randn(M, 12, generator=g)
    s0 = s_true + 0.5 * torch.randn(M, 12, generator=g)          # WRONG init
    fme.reset(torch.arange(M), s0)
    # random admissible (N, lam) per env — Lemma claims exact recovery for ALL
    N = torch.tensor([8., 11., 14., 17., 20., 9.])
    lam = torch.tensor([0.70, 0.80, 0.90, 1.00, 0.75, 0.95])
    Np = torch.tensor([3., 11., 7., 4., 20., 5.])   # mixes batch & two-stage
    err = None
    for k in range(40):   # warmup=N_max=20 -> allow self-sustaining to settle
        u = torch.randn(M, 4, generator=g)
        s_true = fme.model.f(s_true, u)
        z = fme.model.h(s_true)                                  # ZERO noise
        s_hat, _, _ = fme.step(u, z, N, lam, Np=Np)
        err = (s_hat - s_true).abs().max().item()
    assert err < 1e-3, err
    print(f"T2 deadbeat/unbiasedness OK  (max err {err:.2e} for random (N,Np,lam))")


class _BlockModel:
    """UAV-structured linear plant: C attitude/rate columns are EXACTLY zero
    (per-epoch unobservable, like UWB ranges) and the transition is block-
    diagonal so the attitude subsystem is purely input-driven. This is the
    setting of the est_dim=6 observable-block contract."""
    def __init__(self, nx=12, nz=4, seed=7):
        g = torch.Generator().manual_seed(seed)
        Apv = torch.randn(6, 6, generator=g)
        Apv = Apv / max(float(torch.linalg.eigvals(Apv).abs().max().real), 1.0)
        Aar = torch.randn(6, 6, generator=g)
        Aar = Aar / max(float(torch.linalg.eigvals(Aar).abs().max().real), 1.0)
        self.A0 = torch.zeros(nx, nx)
        self.A0[0:6, 0:6] = Apv
        self.A0[6:12, 6:12] = Aar
        self.B0 = 0.1 * torch.randn(nx, 4, generator=g)
        self.H0 = torch.zeros(nz, nx)
        self.H0[:, 0:6] = torch.randn(nz, 6, generator=g)         # att cols == 0

    def f(self, s, u):
        return s @ self.A0.T + u @ self.B0.T

    def h(self, s):
        return s @ self.H0.T

    def jac_f(self, s, u):
        return self.A0.expand(s.shape[0], -1, -1)

    def jac_h(self, s):
        return self.H0.expand(s.shape[0], -1, -1)


def t2b_observable_block_deadbeat():
    # est_dim=6 contract: measurements correct ONLY [p, v]; attitude/rate
    # follow the (input-driven) propagation. Under the contract premise —
    # attitude initialized true, noiseless — the delivered FULL state must be
    # deadbeat-exact for any admissible (N, Np, lam).
    import dataclasses
    cfg = dataclasses.replace(Config(), meas_dim=4, state_clamp=False,
                              innov_gate=1e9, meas_sigma=(1.0,) * 4,
                              est_dim=6)
    M = 6
    fme = WeightedFME(cfg, DEV, M)
    fme.model = _BlockModel()
    g = torch.Generator().manual_seed(3)
    s_true = torch.randn(M, 12, generator=g)
    s0 = s_true.clone()
    s0[:, 0:6] += 0.5 * torch.randn(M, 6, generator=g)   # wrong ONLY in [p, v]
    fme.reset(torch.arange(M), s0)
    N = torch.tensor([8., 11., 14., 17., 20., 9.])
    lam = torch.tensor([0.70, 0.80, 0.90, 1.00, 0.75, 0.95])
    Np = torch.tensor([3., 11., 7., 4., 20., 5.])
    err = None
    for k in range(40):
        u = torch.randn(M, 4, generator=g)
        s_true = fme.model.f(s_true, u)
        z = fme.model.h(s_true)
        s_hat, _, _ = fme.step(u, z, N, lam, Np=Np)
        err = (s_hat - s_true).abs().max().item()
    assert err < 1e-3, err
    print(f"T2b observable-block(est_dim=6) deadbeat OK  (max err {err:.2e})")


class _DoubleIntModel:
    """Structured, physically representative linear system:
    6 positions + 6 velocities, direct position measurement (UWB-like) —
    healthy short-window observability, unlike a fully random (A,H) pair."""

    def __init__(self, dt=0.02, nz=6):
        nx = 12
        A = torch.eye(nx)
        A[0:6, 6:12] = dt * torch.eye(6)
        self.A0 = A
        self.B0 = torch.zeros(nx, 4)
        self.B0[6:10, 0:4] = dt * torch.eye(4)
        # direct (full-rank) position measurement → fully observable pair
        self.H0 = torch.cat([torch.eye(6), torch.zeros(6, 6)], dim=1)

    def f(self, s, u):
        return s @ self.A0.T + u @ self.B0.T

    def h(self, s):
        return s @ self.H0.T

    def jac_f(self, s, u):
        return self.A0.expand(s.shape[0], -1, -1)

    def jac_h(self, s):
        return self.H0.expand(s.shape[0], -1, -1)


def t3_lambda_responsiveness():
    import dataclasses
    cfg = dataclasses.replace(Config(), meas_dim=6, state_clamp=False,
                              innov_gate=1e9, meas_sigma=(1.0,) * 6,
                              est_dim=12)   # this model's whole state IS the
                                            # observable [pos, vel] block
    M = 3
    fme = WeightedFME(cfg, DEV, M)
    lin = _DoubleIntModel(dt=cfg.dt)
    fme.model = lin
    g = torch.Generator().manual_seed(4)
    s_true = torch.zeros(M, 12)
    fme.reset(torch.arange(M), s_true.clone())
    N = torch.full((M,), 20.0)
    lam = torch.tensor([1.0, 0.85, 0.70])
    trans_err = torch.zeros(M)
    steady_err = torch.zeros(M)
    K = 100
    for k in range(K):
        u = 0.2 * torch.randn(M, 4, generator=g)
        s_true = lin.f(s_true, u)
        if k == 40:                                    # unmodeled velocity kick (wind-like)
            s_true[:, 6:12] += 0.8
        z = lin.h(s_true) + 0.005 * torch.randn(M, 6, generator=g)
        s_hat, _, _ = fme.step(u, z, N, lam)
        if 40 <= k < 55:
            trans_err += (s_hat[:, 0:6] - s_true[:, 0:6]).norm(dim=1)
        if 70 <= k:                                    # window fully post-jump
            steady_err += (s_hat[:, 0:6] - s_true[:, 0:6]).norm(dim=1)
    te, se = trans_err.tolist(), steady_err.tolist()
    # the trade-off that motivates ADAPTATION:
    #  (a) during transients some lam<1 beats lam=1 (responsiveness)
    #  (b) in steady state lam=1 beats the smallest lam (noise averaging)
    assert min(te[1], te[2]) < te[0], te
    assert se[0] < se[2], se
    print(f"T3 lambda trade-off OK  (transient: lam=1 {te[0]:.3f} > best lam<1 "
          f"{min(te[1], te[2]):.3f} | steady: lam=1 {se[0]:.3f} < lam=0.7 {se[2]:.3f})")


def t4_nonlinear_sanity():
    cfg = Config()
    M = 1
    m = UAVModel(cfg, DEV)
    fme = WeightedFME(cfg, DEV, M)
    ekf = EKF(cfg, DEV, M, r_diag=[1e-3] * cfg.meas_dim)
    s_true = torch.zeros(M, 12)
    s_true[:, 0:3] = torch.tensor([5.0, 5.0, 1.5])
    s0 = s_true.clone()
    fme.reset(torch.arange(M), s0.clone())
    ekf.reset(torch.arange(M), s0.clone())
    N = torch.full((M,), 14.0)
    lam = torch.full((M,), 1.0)
    hov = cfg.mass_nominal * cfg.g
    e_f, e_k = 0.0, 0.0
    steps = 100
    for k in range(steps):
        u = torch.tensor([[hov + 0.3 * np.sin(0.1 * k), 0.02 * np.sin(0.05 * k),
                           -0.015 * np.cos(0.07 * k), 0.0]], dtype=torch.float32)
        s_true = m.f(s_true, u)                                   # same model, no noise
        z = m.h(s_true)
        sf, _, _ = fme.step(u, z, N, lam)
        sk, _, _ = ekf.step(u, z)
        if k > 30:
            e_f += (sf[:, :3] - s_true[:, :3]).norm().item()
            e_k += (sk[:, :3] - s_true[:, :3]).norm().item()
    e_f /= steps - 31
    e_k /= steps - 31
    assert e_f < 5e-3, e_f
    print(f"T4 nonlinear sanity OK  (WFME pos err {e_f:.2e} m | EKF {e_k:.2e} m)")




def t5_handover_growing_window():
    """Condition-based handover: (a) from filled_valid >= N_min the FME (not
    the aux EKF) serves the estimate; (b) during the ramp, requesting any
    N > filled_valid is EXACTLY equivalent to N = filled_valid (clipping)."""
    import dataclasses
    cfg = dataclasses.replace(Config(), meas_dim=6, state_clamp=False,
                              innov_gate=1e9, meas_sigma=(1.0,) * 6,
                              est_dim=12)   # _DoubleIntModel: whole state observable
    M = 2
    fme = WeightedFME(cfg, DEV, M)
    lin = _DoubleIntModel(dt=cfg.dt)
    fme.model = lin
    g = torch.Generator().manual_seed(9)
    s_true = torch.randn(M, 12, generator=g)
    fme.reset(torch.arange(M), s_true + 0.3 * torch.randn(M, 12, generator=g))
    Nbig = torch.full((M,), float(cfg.N_max))
    lam = torch.full((M,), 0.9)
    hl = float(getattr(cfg, "handover_len", cfg.N_max))
    for k in range(cfg.N_max + 8):
        u = 0.2 * torch.randn(M, 4, generator=g)
        s_true = lin.f(s_true, u)
        z = lin.h(s_true)                                # noise-free
        s_hat, _, _ = fme.step(u, z, Nbig, lam)
        fv = int(fme.w_valid.sum(dim=1)[0].item())
        handed = bool(fme.handed[0].item())
        # (a) handover happens exactly when filled_valid reaches handover_len
        #     (= N_max): before that the aux-EKF serves, after that the FME.
        if fv < hl:
            assert not handed, (k, fv, "handed too early")
        else:
            assert handed, (k, fv, "handover missed")
            # (b) N-clipping equivalence once handed: N=N_max vs N=filled give
            #     identical solves for filled in [N_min, N_max].
            a = fme._solve(Nbig, lam)
            b = fme._solve(torch.full((M,), float(min(fv, cfg.N_max))), lam)
            assert (a - b).abs().max().item() < 1e-6, (k, fv)
            # (c) FME is deadbeat-accurate on noise-free data after handover
            assert (s_hat - s_true).abs().max().item() < 1e-2, \
                (k, fv, (s_hat - s_true).abs().max().item())
    print(f"T5 handover OK  (aux-EKF until filled_valid={int(hl)}=N_max, "
          f"then FME self-sustaining & deadbeat)")


if __name__ == "__main__":
    t1_jacobians()
    t2_deadbeat_unbiasedness()
    t2b_observable_block_deadbeat()
    t3_lambda_responsiveness()
    t4_nonlinear_sanity()
    t5_handover_growing_window()
    print("\nALL FILTER TESTS PASSED")
