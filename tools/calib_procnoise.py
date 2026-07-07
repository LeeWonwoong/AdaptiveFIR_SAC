"""
tools/calib_procnoise.py — calibrate the plant process noise q0
================================================================
Root cause of the earlier null result: the synthetic plant was almost
deterministic (position diffuses ~mm over a 0.6 s window vs σ_meas=0.12 m), so
the finite-memory optimum N_opt was effectively infinite → Greedy saturated at
N_max and the infinite-memory EKF was unbeatable.

Shmaliy: a real system's process is NOT deterministic, so full-horizon
averaging is sub-optimal and N_opt is FINITE. DI-FME's N=14 is that finite
optimum for a real quadrotor. This script finds the process-noise σ (proc_acc_
std, with proc_gyro_std coupled) for which the NOMINAL Greedy median N_opt ≈ 14
at σ_meas=0.12 — i.e. the world consistent with DI-FME.

Greedy here uses λ=1 (the DI-FME/UFIR setting) and the N grid [8..N_max]; the
statistic is the per-step argmin-N of GT position error on NOMINAL trajectories.

Usage: python -m tools.calib_procnoise [--sweep 0.05,0.1,0.2,0.35,0.5,0.8,1.2]
                                       [--m 6] [--T 1000] [--target 14]
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from rlenv.synth import generate_traj
from filter.wfme import WeightedFME

GYRO_RATIO = 0.3          # proc_gyro_std = GYRO_RATIO * proc_acc_std


def _nominal_scenario(cfg, pattern, seed):
    return {"type": "nominal", "pattern": pattern, "duration_s": cfg.traj_duration_s,
            "seed": int(seed), "mass": None, "gusts": [], "sustained": None,
            "dropouts": [], "nlos_burst": [], "turbulence": [], "heldout": False}


def _build(cfg, M, T, seed0):
    """generate M nominal trajectories in-memory → stacked tensors."""
    pats = cfg.flight_patterns
    gts, us = [], []
    for i in range(M):
        sc = _nominal_scenario(cfg, pats[i % len(pats)], seed0 + i)
        arr = generate_traj(cfg, sc, np.random.default_rng(sc["seed"]))
        gts.append(arr["gt"][:T]); us.append(arr["u"][:T])
    gt = torch.tensor(np.stack(gts), dtype=torch.float32)     # [M,T,12]
    u = torch.tensor(np.stack(us), dtype=torch.float32)
    anch = torch.tensor(cfg.anchors, dtype=torch.float32)
    rc = torch.linalg.vector_norm(gt[:, :, None, 0:3] - anch[None, None], dim=3)
    return gt, u, rc


def _greedy_N(cfg, gt, u, rc, sig, Ng, dev):
    """per-step argmin-N (λ=1) of GT position error → all chosen N (post-warmup)."""
    M, T = gt.shape[0], gt.shape[1]
    g = torch.Generator(device=dev).manual_seed(7)
    z = rc + sig * torch.randn(M, T, 4, generator=g, device=dev)
    flt = WeightedFME(cfg, dev, M)
    flt.reset(torch.arange(M, device=dev),
              gt[:, 0] + cfg.init_pos_noise * torch.randn(M, 12, device=dev))
    defN = torch.full((M,), 14.0, device=dev); defL = torch.full((M,), 1.0, device=dev)
    warm = cfg.warmup_steps
    chosen = []
    for t in range(1, T):
        s, _, _ = flt.step(u[:, t - 1], z[:, t], defN, defL)
        if t <= warm:
            continue
        best = (gt[:, t, 0:3] - s[:, 0:3]).pow(2).sum(1)
        bs = s.clone(); bN = defN.clone()
        for n in Ng:
            sc = flt._solve(torch.full((M,), float(n), device=dev), defL)
            e = (gt[:, t, 0:3] - sc[:, 0:3]).pow(2).sum(1)
            b = e < best
            best = torch.where(b, e, best); bs[b] = sc[b]
            bN = torch.where(b, torch.full_like(bN, float(n)), bN)
        flt.s_hat = bs
        chosen.append(bN.cpu().numpy())
    return np.concatenate(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="0.05,0.1,0.2,0.35,0.5,0.8,1.2")
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--sig", type=float, default=0.12)
    ap.add_argument("--target", type=float, default=14.0)
    args = ap.parse_args()
    dev = "cpu"; torch.manual_seed(0)
    qs = [float(x) for x in args.sweep.split(",")]
    cfg0 = Config()
    Ng = list(range(cfg0.N_min, cfg0.N_max + 1, 2))
    print(f"process-noise sweep (nominal, σ_meas={args.sig}, N grid {Ng[0]}..{Ng[-1]}, "
          f"target median N≈{args.target:.0f})")
    print(f"  {'proc_acc_std':>12s} {'gyro':>6s} | {'medN':>5s} {'meanN':>6s} "
          f"{'p10':>4s} {'p90':>4s}  {'satur%':>6s}")
    rows = []
    for q in qs:
        cfg = Config()
        cfg.proc_acc_std = q
        cfg.proc_gyro_std = GYRO_RATIO * q
        gt, u, rc = _build(cfg, args.m, args.T, seed0=1000)
        N = _greedy_N(cfg, gt, u, rc, args.sig, Ng, dev)
        med, mean = np.median(N), N.mean()
        sat = 100.0 * (N >= cfg.N_max).mean()
        rows.append((q, med, mean, sat))
        print(f"  {q:12.3f} {GYRO_RATIO*q:6.3f} | {med:5.1f} {mean:6.1f} "
              f"{np.percentile(N,10):4.0f} {np.percentile(N,90):4.0f}  {sat:6.1f}")
    # recommend q0 with median closest to target
    best = min(rows, key=lambda r: abs(r[1] - args.target))
    print(f"\n  → q0 ≈ proc_acc_std={best[0]:.3f} (proc_gyro_std={GYRO_RATIO*best[0]:.3f}) "
          f"gives median N={best[1]:.1f} (closest to {args.target:.0f})")
    print("  set config.proc_acc_std / proc_gyro_std to these and regenerate data.")


if __name__ == "__main__":
    main()
