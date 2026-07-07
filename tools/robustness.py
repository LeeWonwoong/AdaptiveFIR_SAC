"""
tools/robustness.py — the plain question, simply
================================================
Noisy UWB + in-flight disturbances: does ADAPTING N make the FIR more robust
than a fixed filter (and vs the recursive EKF)?

Filters (apples-to-apples on the SAME measurement draw):
  EKF                 recursive baseline (fixed R)
  FIR N=8 / 14 / 20   fixed window: agile / DI-FME / smooth
  RuleFME             ADAPTIVE, realistic (NIS ratio → short N in disturbance)
  Greedy              ADAPTIVE oracle (per-step best N,λ) = upper bound

Metrics per scenario: overall RMSE, nominal vs disturbance-WINDOW RMSE, and the
PEAK error inside disturbance windows (robustness = lower peak / faster settle).

Usage: python -m tools.robustness [--data_dir data_diag] [--T 1500]
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
from filter.baselines import EKF, FixedFME, RuleFME
from datagen.scenario import disturbance_intervals
from tools.adapt_signal import (type_trajs, window_mask, nominal_mask, rmse,
                                run_recursive, run_fixed, run_greedy, _reset)

SCEN = [("turbulence_burst", "turbulence_burst"),
        ("nlos_burst", "nlos_burst"),
        ("anchor_dropout", "anchor_dropout")]


def run_rule(cfg, ds, z, dev):
    flt = RuleFME(cfg, dev, ds.n); _reset(flt, cfg, ds, dev)
    errs = np.zeros((ds.n, ds.T))
    for t in range(1, ds.T):
        s, _, _ = flt.step(ds.u[:, t - 1], z[:, t])
        errs[:, t] = (ds.gt[:, t, 0:3] - s[:, 0:3]).norm(dim=1).cpu().numpy()
    return errs


def peak(errs, mask):
    """mean over trajectories of the per-traj max error inside the mask."""
    v = []
    for i in range(errs.shape[0]):
        e = errs[i][mask[i]]; e = e[np.isfinite(e)]
        if e.size:
            v.append(e.max())
    return float(np.mean(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_diag")
    ap.add_argument("--split", default="train")
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--outdir", default="results/robustness")
    args = ap.parse_args()
    cfg = Config(); cfg.data_dir = args.data_dir
    dev = "cpu"; torch.manual_seed(0)
    ds = TrajDataset(cfg, args.split, dev)
    if args.T < ds.T:
        for a in ("gt", "u", "m_true", "wind", "range_clean", "noise_scale", "range_bias"):
            setattr(ds, a, getattr(ds, a)[:, :args.T])
        ds.T = args.T
    os.makedirs(args.outdir, exist_ok=True)

    base = torch.tensor(cfg.meas_sigma, device=dev).view(1, 1, 4)
    g = torch.Generator(device=dev).manual_seed(1234)
    z = ds.range_clean + ds.range_bias + base * ds.noise_scale * torch.randn(
        ds.n, ds.T, 4, generator=g, device=dev)

    lines = []
    def P(s=""): lines.append(s); print(s)
    P(f"[robustness] {ds.n} trajs x {ds.T} steps  σ_LoS={cfg.meas_sigma[0]}  "
      f"N∈[{cfg.N_min},{cfg.N_max}]  (data={args.data_dir})")
    P("  running EKF / FIR8 / FIR14 / FIR20 / RuleFME(adaptive) / Greedy(oracle) ...")

    filt = {
        "EKF": run_recursive(cfg, ds, z, dev, EKF),
        "FIR8": run_fixed(cfg, ds, 8, 1.0, z, dev),
        "FIR14": run_fixed(cfg, ds, 14, 1.0, z, dev),
        "FIR20": run_fixed(cfg, ds, 20, 1.0, z, dev),
        "RuleFME": run_rule(cfg, ds, z, dev),
    }
    grd, chN, _, _ = run_greedy(cfg, ds, z, dev)
    filt["Greedy"] = grd
    names = ["EKF", "FIR8", "FIR14", "FIR20", "RuleFME", "Greedy"]

    nom = nominal_mask(cfg, ds)
    P("\nRMSE [m] — nominal (all trajs) and each disturbance WINDOW")
    P(f"  {'region':16s} " + " ".join(f"{n:>8s}" for n in names))
    P(f"  {'nominal':16s} " + " ".join(f"{rmse(filt[n], nom):8.3f}" for n in names))
    winmasks = {}
    for sname, label in SCEN:
        rows = type_trajs(ds, sname)
        if not rows:
            continue
        rset = np.zeros(ds.n, bool); rset[rows] = True
        wm = window_mask(cfg, ds, label, 0.3) & rset[:, None]
        winmasks[sname] = wm
        if wm.any():
            P(f"  {sname:16s} " + " ".join(f"{rmse(filt[n], wm):8.3f}" for n in names))

    P("\nPEAK error [m] inside each disturbance window (robustness = lower)")
    P(f"  {'window':16s} " + " ".join(f"{n:>8s}" for n in names))
    for sname, _ in SCEN:
        if sname in winmasks and winmasks[sname].any():
            P(f"  {sname:16s} " + " ".join(f"{peak(filt[n], winmasks[sname]):8.3f}"
                                            for n in names))

    # adaptive N behaviour: does Greedy actually shrink N in disturbance windows?
    P("\nGreedy chosen-N: nominal vs window (does it adapt?)")
    Nn = chN[nom]; Nn = Nn[np.isfinite(Nn)]
    P(f"  nominal  meanN={Nn.mean():4.1f}")
    for sname, _ in SCEN:
        if sname in winmasks and winmasks[sname].any():
            Nd = chN[winmasks[sname]]; Nd = Nd[np.isfinite(Nd)]
            P(f"  {sname:16s} meanN={Nd.mean():4.1f}  (ΔN={Nd.mean()-Nn.mean():+.1f})")

    # ── plain verdict ──
    P("\nVERDICT (plain):")
    best_fixed = lambda m: min(rmse(filt["FIR8"], m), rmse(filt["FIR14"], m),
                               rmse(filt["FIR20"], m))
    for sname, _ in SCEN:
        if sname not in winmasks or not winmasks[sname].any():
            continue
        m = winmasks[sname]
        e_ekf = rmse(filt["EKF"], m); e_bf = best_fixed(m)
        e_rule = rmse(filt["RuleFME"], m); e_grd = rmse(filt["Greedy"], m)
        adapt_vs_fixed = 100 * (e_bf - e_grd) / e_bf
        rule_vs_fixed = 100 * (e_bf - e_rule) / e_bf
        P(f"  {sname:16s} bestFIR={e_bf:.3f} EKF={e_ekf:.3f} | "
          f"Greedy {adapt_vs_fixed:+.0f}% vs bestFIR, "
          f"Rule {rule_vs_fixed:+.0f}% vs bestFIR, "
          f"{'EKF wins' if e_ekf < min(e_bf,e_rule,e_grd) else 'FIR-family wins'}")

    # ── one clear figure: error traces on a disturbance example per scenario ──
    fig, axes = plt.subplots(len(SCEN), 1, figsize=(9, 8), sharex=True)
    tt = np.arange(ds.T) * cfg.dt
    for ax, (sname, label) in zip(axes, SCEN):
        rows = type_trajs(ds, sname)
        if not rows:
            continue
        ex = rows[0]
        for n, col in [("EKF", "tab:green"), ("FIR20", "tab:blue"),
                       ("RuleFME", "tab:orange"), ("Greedy", "tab:red")]:
            ax.plot(tt, filt[n][ex], lw=0.9, color=col, label=n)
        for (t0, t1, lab) in disturbance_intervals(ds.metas[ex].get("scenario", {})):
            if lab == label:
                ax.axvspan(t0, t1, color="gray", alpha=0.15)
        ax.set_ylabel("pos err [m]"); ax.set_title(sname, fontsize=9)
        ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=4)
    axes[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "robustness.png"), dpi=110); plt.close(fig)
    with open(os.path.join(args.outdir, "robustness.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nfigure+summary → {args.outdir}/")


if __name__ == "__main__":
    main()
