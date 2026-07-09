"""
filter/uav_model.py — 12-state quadrotor model + UWB range measurement
=======================================================================
State  s = [p(3), v(3), eta(3), omega(3)]   (world ENU, body FLU)
Input  u = [T, tau_x, tau_y, tau_z]         (physical: N, Nm — datagen converts
                                             PX4 setpoints via calibration.json)
Meas   z = [d_1..d_4; eta; omega] in R^10    UWB ranges to fixed anchors +
             FCU attitude estimate + gyro rates (standard onboard channels;
             renders every FME window fully observable — see config.meas_sigma).

Discrete model (DI-FME, Sec. II — Euler step, dt = cfg.dt):
  p'     = p + v*T
  v'     = v + ( g_vec + (1/m) R(eta) f_T ) * T ,  f_T = [0,0,+T_thrust] (FLU up),
                                                   g_vec = [0,0,-g]
  eta'   = eta + W(eta) omega * T              (Euler-rate kinematic matrix;
                                                the paper's "W = diag(I)" is a typo)
  omega' = omega + J^{-1}( tau - omega x J omega ) * T

Linearization: step-wise frozen 1st-order Taylor around the running estimate
(EFIR convention). Jacobians via torch.func.jacrev + vmap (verified against
finite differences in tests/test_wfme.py).

All functions are batched over leading dim M and run in FP32 (filter path is
kept FP32 even when NN forward uses TF32 — rhukf precision policy).
"""
import torch
from torch.func import jacrev, vmap


class UAVModel:
    def __init__(self, cfg, device, mass=None):
        self.cfg = cfg
        self.dev = device
        self.dt = cfg.dt
        self.m = float(mass if mass is not None else cfg.mass_nominal)
        self.g = cfg.g
        self.J = torch.tensor([cfg.Ixx, cfg.Iyy, cfg.Izz],
                              dtype=torch.float32, device=device)          # diag inertia
        self.Jinv = 1.0 / self.J
        self.anchors = torch.tensor(cfg.anchors, dtype=torch.float32,
                                    device=device)                          # [4,3]
        self.g_vec = torch.tensor([0.0, 0.0, -self.g], dtype=torch.float32,
                                  device=device)
        # batched jacobians (built lazily; vmap over batch dim)
        self._jac_f_s = vmap(jacrev(self._f_single, argnums=0))
        self._jac_h_s = vmap(jacrev(self._h_single))

    # ---------------- single-sample dynamics (used by AD) ----------------
    def _f_single(self, s, u):
        p, v, eta, om = s[0:3], s[3:6], s[6:9], s[9:12]
        phi, th, psi = eta[0], eta[1], eta[2]
        # clip pitch/roll away from singularity (matches repo UKF practice)
        lim = 1.2
        phi = torch.clamp(phi, -lim, lim)
        th = torch.clamp(th, -lim, lim)
        cph, sph = torch.cos(phi), torch.sin(phi)
        cth, sth = torch.cos(th), torch.sin(th)
        cps, sps = torch.cos(psi), torch.sin(psi)
        # R body->world (ZYX yaw-pitch-roll)
        R = torch.stack([
            torch.stack([cps * cth, cps * sth * sph - sps * cph, cps * sth * cph + sps * sph]),
            torch.stack([sps * cth, sps * sth * sph + cps * cph, sps * sth * cph - cps * sph]),
            torch.stack([-sth,       cth * sph,                   cth * cph]),
        ])                                                                    # [3,3]
        thrust_b = torch.stack([torch.zeros((), device=s.device, dtype=s.dtype),
                                torch.zeros((), device=s.device, dtype=s.dtype),
                                u[0]])
        acc = self.g_vec + (R @ thrust_b) / self.m
        # Euler-rate matrix W(eta): eta_dot = W om
        tt = sth / cth.clamp_min(1e-3) if False else torch.tan(th)
        sec = 1.0 / cth.clamp(min=0.35)     # bounded secant (|th|<1.2 → cth>0.36)
        W = torch.stack([
            torch.stack([torch.ones_like(phi), sph * tt, cph * tt]),
            torch.stack([torch.zeros_like(phi), cph, -sph]),
            torch.stack([torch.zeros_like(phi), sph * sec, cph * sec]),
        ])
        tau = u[1:4]
        om_dot = self.Jinv * (tau - torch.cross(om, self.J * om, dim=0))
        T = self.dt
        return torch.cat([p + v * T,
                          v + acc * T,
                          eta + (W @ om) * T,
                          om + om_dot * T])

    def _h_single(self, s):
        d = s[0:3].unsqueeze(0) - self.anchors                                # [4,3]
        rng = torch.linalg.vector_norm(d, dim=1)                               # [4] UWB
        return torch.cat([rng, s[6:9], s[9:12]], dim=0)                        # [10] +att+gyro

    # ---------------- batched public API ----------------
    def f(self, s, u):
        """s [M,12], u [M,4] -> [M,12] (nonlinear propagation)."""
        return vmap(self._f_single)(s, u)

    def h(self, s):
        """s [M,12] -> [M,10]: UWB 4 ranges + IMU (attitude 3 + gyro 3).
        UWB rows observe position only; attitude rows (FCU estimate) observe
        eta = s[6:9]; gyro rows observe omega = s[9:12]. The attitude/rate
        selector rows make the previously-null Jacobian columns full rank, so
        the windowed normal matrix C^T C is invertible and est_dim=12 no longer
        diverges (the yaw-unobservable failure of UWB-only)."""
        d = s[:, None, 0:3] - self.anchors[None]                               # [M,4,3]
        rng = torch.linalg.vector_norm(d, dim=2)                               # [M,4]
        att = s[:, 6:9]                                                        # [M,3] eta
        gyro = s[:, 9:12]                                                      # [M,3] omega
        return torch.cat([rng, att, gyro], dim=1)                             # [M,10]

    def jac_f(self, s, u):
        """A_i = df/ds | (s,u)   -> [M,12,12]"""
        return self._jac_f_s(s, u)

    def jac_h(self, s):
        """C_i = dh/ds | s -> [M,10,12] (range rows: unit LOS; att/rate rows: selectors)"""
        return self._jac_h_s(s)

    # convenience: pseudo-input / pseudo-measurement (affine corrections)
    def pseudo_input(self, s_lin, u, A):
        """u_tilde = f(s_lin,u) - A s_lin   (absorbs B u + Taylor offset; FM-SLAC style)"""
        return self.f(s_lin, u) - torch.bmm(A, s_lin.unsqueeze(-1)).squeeze(-1)

    def pseudo_meas(self, z, s_lin, C):
        """z_tilde = z - h(s_lin) + C s_lin  ->  z_tilde ≈ C s + v"""
        return z - self.h(s_lin) + torch.bmm(C, s_lin.unsqueeze(-1)).squeeze(-1)
