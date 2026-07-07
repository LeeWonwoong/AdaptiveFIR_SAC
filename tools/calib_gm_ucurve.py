"""
tools/calib_gm_ucurve.py — A-1: does a TIME-CORRELATED (OU) measurement bias
create an INTERNAL fixed-N U-curve floor?  (Phase-0 last mechanism)
============================================================================
Phase-0 NO-GO root cause: white measurement noise always averages down √N, so
the nominal fixed-N RMSE U-curve is MONOTONE-decreasing → N_opt pins at N_max.
The only untried measurement-side fix is a time-correlated per-anchor bias
(discrete Gauss-Markov / OU, rlenv.dataset.gauss_markov_bias): older epochs in
the window carry a bias that has WANDERED away from the current value → stale →
including too much old data hurts → a FINITE N_opt appears.

This tool builds NOMINAL trajectories (LoS only), applies OU bias with LoS
parameters, and sweeps (σ_b, τ) to find the combo whose fixed-N U-curve bottoms
INTERNALLY in (N_min, N_max) with a rise on both sides (a true U).

  z = range_clean + b_OU(σ_b, τ) + σ_LoS·randn        (λ=1, FixedFME)

Usage:
  python -m tools.calib_gm_ucurve --sig 0.12 --m 12 --T 1500 \
      --std 0.03,0.05,0.08 --tau 0.3,0.5,1.0,2.0 --plot
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from filter.baselines import FixedFME
from rlenv.dataset import gauss_markov_bias
from tools.calib_ucurve import _build, NGRID, _argmin_and_shape


def _ucurve_gm(cfg, gt, u, rc, sig, std, tau, dev, seed=1234):
    """fixed-N RMSE curve with OU bias b(σ_b=std, τ=tau) + white σ_LoS."""
    M, T = gt.shape[0], gt.shape[1]
    g = torch.Generator(device=dev).manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    std_f = np.full((M, T, 4), std, np.float64)
    tau_f = np.full((M, T, 4), tau, np.float64)
    b = torch.tensor(gauss_markov_bias(std_f, tau_f, cfg.dt, rng,
                                       clip=cfg.gm_bias_clip), device=dev)
    z = rc + b + sig * torch.randn(M, T, 4, generator=g, device=dev)
    warm = cfg.warmup_steps
    out = {}
    for N in NGRID:
        flt = FixedFME(cfg, dev, M, N=N, lam=1.0)
        flt.reset(torch.arange(M, device=dev),
                  gt[:, 0] + cfg.init_pos_noise * torch.randn(M, 12, device=dev))
        sse, cnt = 0.0, 0
        for t in range(1, T):
            s, _, _ = flt.step(u[:, t - 1], z[:, t])
            if t <= warm:
                continue
            e2 = (gt[:, t, 0:3] - s[:, 0:3]).pow(2).sum(1)
            sse += float(e2.sum().item()); cnt += M
        out[N] = float(np.sqrt(sse / max(cnt, 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", type=float, default=None)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--std", default="0.03,0.05,0.08")
    ap.add_argument("--tau", default="0.3,0.5,1.0,2.0")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--outdir", default="results/calib")
    args = ap.parse_args()
    dev = "cpu"; torch.manual_seed(0)

    cfg = Config()
    sig = args.sig if args.sig is not None else float(cfg.meas_sigma[0])
    stds = [float(x) for x in args.std.split(",")]
    taus = [float(x) for x in args.tau.split(",")]

    # nominal trajectories (built once, reused for every (σ_b, τ))
    gt, u, rc = _build(cfg, args.m, args.T, seed0=1000)

    print(f"A-1: OU-bias fixed-N U-curve  (nominal, σ_LoS={sig:.2f}, λ=1, "
          f"N grid {NGRID}, M={args.m}, T={args.T})")
    # white-noise baseline (σ_b=0) for reference — should be monotone-decreasing
    base = _ucurve_gm(cfg, gt, u, rc, sig, 0.0, 1.0, dev)
    bN, bU, *_ = _argmin_and_shape(base)
    print("  " + f"{'σ_b':>5s} {'τ[s]':>5s}  " + " ".join(f"N{n:>2d}" for n in NGRID)
          + f"   {'N_opt':>5s} {'trueU':>5s}")
    cells = " ".join(f"{base[n]:.3f}" for n in NGRID)
    print(f"  {'0.00':>5s} {'--':>5s}  {cells}   {bN:5d} {str(bU):>5s}   (white ref)")

    results = {}
    hits = []
    for std in stds:
        for tau in taus:
            rm = _ucurve_gm(cfg, gt, u, rc, sig, std, tau, dev)
            results[(std, tau)] = rm
            Nopt, trueU, lu, ru = _argmin_and_shape(rm)
            internal = trueU and cfg.N_min < Nopt < cfg.N_max
            flag = "  ← INTERNAL floor" if internal else ""
            if internal:
                hits.append((std, tau, Nopt))
            cells = " ".join(f"{rm[n]:.3f}" for n in NGRID)
            print(f"  {std:5.2f} {tau:5.2f}  {cells}   {Nopt:5d} {str(trueU):>5s}{flag}")

    print("\n  SUMMARY:")
    if hits:
        for std, tau, Nopt in hits:
            print(f"    ✓ INTERNAL floor N_opt={Nopt} at σ_b={std}, τ={tau}s "
                  f"(→ candidate LoS params)")
    else:
        print("    ✗ no INTERNAL floor in (N_min,N_max) for any (σ_b,τ) swept "
              "— LoS OU alone does not unpin N_opt; the lever (if any) is the "
              "NLoS fast-τ window (see A-2/A-3).")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.outdir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        ax.plot(NGRID, [base[n] for n in NGRID], marker="s", color="k",
                lw=1.4, label="white σ_b=0 (ref)")
        for (std, tau), rm in results.items():
            Nopt = _argmin_and_shape(rm)[0]
            ax.plot(NGRID, [rm[n] for n in NGRID], marker="o", alpha=.8,
                    label=f"σ_b={std}, τ={tau}s (N*={Nopt})")
        ax.set_xlabel("fixed horizon N"); ax.set_ylabel("nominal RMSE [m]")
        ax.set_title(f"A-1 OU-bias fixed-N U-curve (σ_LoS={sig:.2f})")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3); fig.tight_layout()
        p = os.path.join(args.outdir, "ucurve_gm_nominal.png")
        fig.savefig(p, dpi=120); plt.close(fig)
        print(f"  plot → {p}")


if __name__ == "__main__":
    main()
