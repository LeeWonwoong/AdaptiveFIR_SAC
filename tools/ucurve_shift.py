"""
tools/ucurve_shift.py — per-regime fixed-N U-curve overlay (ROADMAP Phase 0-2)
=============================================================================
Paper motivation figure. With q0 calibrated so the NOMINAL fixed-N RMSE U-curve
bottoms at N=14 (tools.calib_ucurve), overlay the U-curve computed over the
DISTURBANCE WINDOW of each regime. Expectation (Shmaliy Fig.10 spirit):
  - turbulence_burst : bottom shifts LEFT (N≈7-9) — dump stale data fast when the
                       plant process noise spikes.
  - nlos_burst       : bottom shifts LEFT — a biased/outlier anchor pollutes the
                       window; a shorter N flushes it sooner.
  - anchor_dropout   : bottom shifts RIGHT / high-N — hold the pre-dropout
                       4-anchor geometry to ride out the GDOP loss.

Fixed-N grid {8,10,12,14,16,18,20}, λ=1, self_anchor=False. Measurement layer
(σ_LoS, NLoS bias/σ, dropout NaN) comes straight from the dataset — same data the
decomposition table (tools.adapt_signal) uses, so the two are consistent.

Outputs → results/calib/ucurve_shift.png  + printed bottom-N table.
Usage: python -m tools.ucurve_shift [--data_dir data_diag] [--split train] [--T 2000]
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from rlenv.dataset import TrajDataset
from filter.baselines import FixedFME
from tools.adapt_signal import (type_trajs, window_mask, nominal_mask, rmse, _reset)

NGRID = [8, 10, 12, 14, 16, 18, 20]
# (regime type, window label or None, post-window margin [s], plot color)
REGIMES = [
    ("nominal",          None,               0.0, "tab:gray"),
    ("turbulence_burst", "turbulence_burst", 0.3, "tab:red"),
    ("nlos_burst",       "nlos_burst",       0.3, "tab:orange"),
    ("anchor_dropout",   "anchor_dropout",   0.3, "tab:blue"),
]


def run_fixed_err(cfg, ds, N, z, dev):
    """FixedFME(N, λ=1) position-error series [n,T]."""
    flt = FixedFME(cfg, dev, ds.n, N=N, lam=1.0); _reset(flt, cfg, ds, dev)
    errs = np.zeros((ds.n, ds.T))
    for t in range(1, ds.T):
        s, _, _ = flt.step(ds.u[:, t - 1], z[:, t])
        errs[:, t] = (ds.gt[:, t, 0:3] - s[:, 0:3]).norm(dim=1).cpu().numpy()
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_diag")
    ap.add_argument("--split", default="train")
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--outdir", default="results/calib")
    args = ap.parse_args()
    cfg = Config(); cfg.data_dir = args.data_dir
    dev = "cpu"; torch.manual_seed(0)
    ds = TrajDataset(cfg, args.split, dev)
    if args.T < ds.T:
        for a in ("gt", "u", "m_true", "wind", "range_clean", "noise_scale", "range_bias"):
            setattr(ds, a, getattr(ds, a)[:, :args.T])
        ds.T = args.T
    counts = {t: len(type_trajs(ds, t)) for t, _, _, _ in REGIMES}
    print(f"[ucurve_shift] {ds.n} trajs x {ds.T} steps  counts={counts} "
          f"(q0=proc_acc_std={cfg.proc_acc_std}, σ_LoS={cfg.meas_sigma[0]})")

    # measurement draw (same construction as adapt_signal)
    base = torch.tensor(cfg.meas_sigma, device=dev).view(1, 1, 4)
    g = torch.Generator(device=dev).manual_seed(1234)
    z = ds.range_clean + ds.range_bias + base * ds.noise_scale * torch.randn(
        ds.n, ds.T, 4, generator=g, device=dev)

    # fixed-N error series (once per N, reused for every regime mask)
    err = {}
    for N in NGRID:
        err[N] = run_fixed_err(cfg, ds, N, z, dev)
        print(f"  fixed N={N:2d} done")

    nom_all = nominal_mask(cfg, ds)
    # per-regime mask restricted to that regime's trajectories
    curves = {}
    for name, label, margin, _ in REGIMES:
        rows = type_trajs(ds, name)
        if not rows:
            continue
        rset = np.zeros(ds.n, bool); rset[rows] = True
        if label is None:
            mask = nom_all & rset[:, None]
        else:
            mask = window_mask(cfg, ds, label, margin) & rset[:, None]
        if not mask.any():
            continue
        curves[name] = np.array([rmse(err[N], mask) for N in NGRID])

    # ── report ──
    print("\nfixed-N U-curve RMSE [m] over each regime's window (λ=1):")
    print("  " + f"{'regime':17s}" + " ".join(f"N{n:>2d}" for n in NGRID) + "  N_opt")
    for name, *_ in REGIMES:
        if name not in curves:
            continue
        c = curves[name]; nopt = NGRID[int(np.argmin(c))]
        cells = " ".join(f"{v:.3f}" for v in c)
        print(f"  {name:17s}{cells}  {nopt:>4d}")
    if "nominal" in curves and "turbulence_burst" in curves:
        dn = NGRID[int(np.argmin(curves['nominal']))]
        dt = NGRID[int(np.argmin(curves['turbulence_burst']))]
        print(f"\n  nominal bottom N={dn}  →  turbulence_burst bottom N={dt}  "
              f"(ΔN={dt-dn:+d}; expect leftward/negative)")

    # ── figure ──
    os.makedirs(args.outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for name, label, margin, color in REGIMES:
        if name not in curves:
            continue
        c = curves[name]
        nopt = NGRID[int(np.argmin(c))]
        ax.plot(NGRID, c, marker="o", color=color, label=f"{name} (N*={nopt})")
        ax.scatter([nopt], [c.min()], color=color, s=80, zorder=5,
                   edgecolor="k", linewidth=0.6)
    ax.axvline(14, color="k", ls="--", lw=1, alpha=.5, label="DI-FME N=14")
    ax.set_xlabel("fixed horizon N"); ax.set_ylabel("window RMSE [m]")
    ax.set_title(f"Regime-dependent U-curve shift (q0={cfg.proc_acc_std}, σ_LoS={cfg.meas_sigma[0]})")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    p = os.path.join(args.outdir, "ucurve_shift.png")
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"\nfigure → {p}")


if __name__ == "__main__":
    main()
