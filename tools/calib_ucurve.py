"""
tools/calib_ucurve.py — calibrate q0 to the FIXED-N RMSE U-curve bottom = N=14
==============================================================================
CORRECT calibration target (ROADMAP Phase 0-1). The earlier calib_procnoise.py
targeted the wrong statistic (per-step greedy-argmin MEDIAN), which is insensitive
to q (saturates) and is NOT the Shmaliy finite-memory optimum.

Shmaliy's N_opt is  argmin_N  RMSE( FixedFME(N, λ=1) )  over a FIXED window N —
the balance point between too-small-N noise variance and too-large-N staleness
bias (the latter exists only if the plant has honest process noise q>0). We sweep
q = proc_acc_std (with proc_gyro_std = GYRO_RATIO·q) on NOMINAL trajectories and
find the q0 whose fixed-N U-curve bottoms at N=14 (DI-FME's experimental choice),
verifying it is a TRUE U (RMSE rises on BOTH sides of the minimum).

Measurement: z = range_clean + σ_LoS·randn,  σ_LoS = cfg.meas_sigma (0.12).
Filter: FixedFME(N, λ=1), self_anchor=False (aux-EKF-anchored linearization).

Usage:
  python -m tools.calib_ucurve --sweep 0.5,1.0,1.5,2.0,2.5,3.0 --m 8 --T 1500
  python -m tools.calib_ucurve --q 1.5 --plot            # single-q U-curve + png
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from rlenv.synth import generate_traj
from filter.baselines import FixedFME

GYRO_RATIO = 0.3
NGRID = [8, 10, 12, 14, 16, 18, 20]


def _nominal_scenario(cfg, pattern, seed):
    return {"type": "nominal", "pattern": pattern, "duration_s": cfg.traj_duration_s,
            "seed": int(seed), "mass": None, "gusts": [], "sustained": None,
            "dropouts": [], "nlos_burst": [], "turbulence": [], "heldout": False}


def _build(cfg, M, T, seed0):
    """M nominal trajectories in-memory → (gt, u, range_clean) tensors."""
    pats = cfg.flight_patterns
    gts, us = [], []
    for i in range(M):
        sc = _nominal_scenario(cfg, pats[i % len(pats)], seed0 + i)
        arr = generate_traj(cfg, sc, np.random.default_rng(sc["seed"]))
        gts.append(arr["gt"][:T]); us.append(arr["u"][:T])
    gt = torch.tensor(np.stack(gts), dtype=torch.float32)
    u = torch.tensor(np.stack(us), dtype=torch.float32)
    anch = torch.tensor(cfg.anchors, dtype=torch.float32)
    rc = torch.linalg.vector_norm(gt[:, :, None, 0:3] - anch[None, None], dim=3)
    return gt, u, rc


def _ucurve(cfg, gt, u, rc, sig, dev, seed=1234):
    """fixed-N RMSE curve over post-warmup steps for each N in NGRID (λ=1)."""
    M, T = gt.shape[0], gt.shape[1]
    g = torch.Generator(device=dev).manual_seed(seed)
    z = rc + sig * torch.randn(M, T, 4, generator=g, device=dev)
    warm = cfg.warmup_steps
    rmse = {}
    for N in NGRID:
        flt = FixedFME(cfg, dev, M, N=N, lam=1.0)
        flt.reset(torch.arange(M, device=dev),
                  gt[:, 0] + cfg.init_pos_noise * torch.randn(M, 12, device=dev))
        sse, cnt = 0.0, 0
        for t in range(1, T):
            s, _, _ = flt.step(u[:, t - 1], z[:, t])
            if t <= warm:
                continue
            e2 = (gt[:, t, 0:3] - s[:, 0:3]).pow(2).sum(1)   # [M] squared pos err
            sse += float(e2.sum().item()); cnt += M
        rmse[N] = float(np.sqrt(sse / max(cnt, 1)))
    return rmse


def _argmin_and_shape(rmse):
    Ns = NGRID
    vals = [rmse[N] for N in Ns]
    i = int(np.argmin(vals))
    Nopt = Ns[i]
    left_up = i == 0 or vals[i - 1] > vals[i]
    right_up = i == len(Ns) - 1 or vals[i + 1] > vals[i]
    true_u = (i not in (0, len(Ns) - 1)) and left_up and right_up
    return Nopt, true_u, left_up, right_up


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="")
    ap.add_argument("--q", type=float, default=None)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--sig", type=float, default=None)
    ap.add_argument("--target", type=int, default=14)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default="results/calib")
    args = ap.parse_args()
    dev = "cpu"; torch.manual_seed(0)

    cfg0 = Config()
    sig = args.sig if args.sig is not None else float(cfg0.meas_sigma[0])
    if args.sweep:
        qs = [float(x) for x in args.sweep.split(",")]
    elif args.q is not None:
        qs = [args.q]
    else:
        qs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"fixed-N U-curve calibration  (nominal, σ_LoS={sig:.2f}, λ=1, "
          f"N grid {NGRID}, target bottom N={args.target})")
    hdr = "  " + f"{'q':>6s} " + " ".join(f"N{n:>2d}" for n in NGRID) + \
          f"   {'N_opt':>5s} {'trueU':>5s}"
    print(hdr)
    results = {}
    for q in qs:
        cfg = Config()
        cfg.proc_acc_std = q
        cfg.proc_gyro_std = GYRO_RATIO * q
        gt, u, rc = _build(cfg, args.m, args.T, seed0=1000)
        rmse = _ucurve(cfg, gt, u, rc, sig, dev)
        results[q] = rmse
        Nopt, trueU, lu, ru = _argmin_and_shape(rmse)
        cells = " ".join(f"{rmse[n]:.3f}" for n in NGRID)
        flag = "  ← bottom=14" if (Nopt == args.target and trueU) else \
               ("  (bottom=%d)" % Nopt)
        print(f"  {q:6.2f} {cells}   {Nopt:5d} {str(trueU):>5s}{flag}")

    # recommend q0: prefer exact target with true-U, else nearest N_opt to target
    def score(q):
        Nopt, trueU, *_ = _argmin_and_shape(results[q])
        return (abs(Nopt - args.target), 0 if trueU else 1)
    q0 = min(qs, key=score)
    Nopt, trueU, lu, ru = _argmin_and_shape(results[q0])
    print(f"\n  → q0 ≈ proc_acc_std={q0:.3f} (proc_gyro_std={GYRO_RATIO*q0:.3f}): "
          f"N_opt={Nopt}, trueU={trueU} (left_up={lu}, right_up={ru})")
    if Nopt == args.target and trueU:
        print(f"     ✓ fixed-N U-curve bottoms at N={args.target} with rise on both sides.")
        print(f"     set config.proc_acc_std={q0:.3f}, proc_gyro_std={GYRO_RATIO*q0:.3f}")
    else:
        print(f"     ✗ target N={args.target} true-U not hit on this grid — refine --sweep.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.outdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        for q in qs:
            ax.plot(NGRID, [results[q][n] for n in NGRID], marker="o",
                    label=f"q={q:.2f}")
        ax.axvline(args.target, color="gray", ls="--", lw=1, alpha=.7)
        ax.set_xlabel("fixed horizon N"); ax.set_ylabel("nominal RMSE [m]")
        ax.set_title(f"fixed-N U-curve (σ_LoS={sig:.2f}, λ=1)")
        ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
        p = os.path.join(args.outdir, "ucurve_nominal.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f"  plot → {p}")


if __name__ == "__main__":
    main()
