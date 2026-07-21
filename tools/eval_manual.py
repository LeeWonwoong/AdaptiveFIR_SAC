#!/usr/bin/env python3
"""
Evaluate EKF / UKF / FME / AFME on MANUAL (pilot-in-the-loop) trajectories.

Unlike tools/eval_seeds.py this does NOT group by cfg.heldout_plan (manual
flights are free-form); each trajectory is reported individually using the
windows stored in its own meta_XXXX.json, with the same constants and the
same 2-40 s window as the paper (tools/_common.py).

All filters are causal, so post-processing the recorded stream is bit-exact
equivalent to running them online during the flight.

  python3 tools/eval_manual.py --data_dir data_manual \
      --ckpt results/v12_50k/ckpt.pt --seed 13 --outdir figures_manual
"""
import argparse
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (load_cfg, make_dataset, make_runner, load_agent, run_all,
                     eval_slice, rmse3, err_norm, moving_rms, moving_avg,
                     COLORS, METHODS, EVAL_T0, EVAL_T1, Q_EKF, Q_UKF, FME_N)


def meta_windows(meta):
    """[(start, end, label, shade_color)] from a manual/scripted meta."""
    sc = meta.get("scenario", meta)
    out = []
    if sc.get("sustained"):
        for w in sc["sustained"]:
            out.append((w["start_s"], w["start_s"] + w["duration_s"],
                        "sustained wind", "#d62728"))
    if sc.get("mass"):
        m = sc["mass"]
        out.append((m["onset_s"], m["onset_s"] + m["duration_s"],
                    "payload attached", "#ff7f0e"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="engine --out dir containing <split>/traj_*.npz")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--outdir", default="figures_manual")
    ap.add_argument("--smooth", type=float, default=1.5)
    ap.add_argument("--label-y", type=float, default=0.975)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    cfg = load_cfg(a.data_dir)
    ds, M = make_dataset(cfg, a.device)          # TrajDataset globs the split dir
    run = make_runner(cfg, ds, a.device, a.seed)
    agent = load_agent(cfg, a.ckpt, a.device)
    res = run_all(run, cfg, a.device, M, agent)
    sl = eval_slice(cfg, ds.T)
    t = np.arange(ds.T) * cfg.dt
    t_end = min(EVAL_T1, ds.T * cfg.dt)      # short manual flights: clamp axis

    metas = []
    for f in sorted(glob.glob(os.path.join(a.data_dir, a.split, "meta_*.json"))):
        with open(f) as fh:
            metas.append(json.load(fh))
    assert len(metas) == M, f"{len(metas)} metas vs {M} trajs"

    # ---------------------------------------------------------- per-traj table
    print(f"# manual evaluation | window {EVAL_T0:g}-{t_end:g}s | "
          f"Q={Q_EKF:g} FME N={FME_N} | noise seed {a.seed} | 3-D RMSE [m]")
    hdr = " ".join(f"{m:>7}" for m in METHODS)
    print(f"{'traj':>5} {'type':>15} | {hdr} | in-window ({'/'.join(METHODS)})")
    agg = {m: [] for m in METHODS}
    for i, meta in enumerate(metas):
        sc = meta.get("scenario", meta)
        vals = {m: rmse3(res[m]["evec"], [i], sl) for m in METHODS}
        for m in METHODS:
            agg[m].append(vals[m])
        wins = meta_windows(meta)
        if wins:
            k = np.zeros(ds.T, dtype=bool)
            for (s0, s1, _, _) in wins:
                k[int(s0 / cfg.dt):int(s1 / cfg.dt)] = True
            k[:sl.start] = False; k[sl.stop:] = False
            iw = "/".join(f"{np.sqrt((np.asarray(res[m]['evec'])[i][k]**2).sum(-1).mean()):.3f}"
                          for m in METHODS)
        else:
            iw = "-"
        print(f"{i:>5} {sc.get('type','?'):>15} | "
              + " ".join(f"{vals[m]:7.3f}" for m in METHODS) + f" | {iw}")
    print(f"{'mean':>5} {'':>15} | "
          + " ".join(f"{np.mean(agg[m]):7.3f}" for m in METHODS))

    order_ok = all(np.mean(agg['AFME']) < np.mean(agg['FME'])
                   < min(np.mean(agg['EKF']), np.mean(agg['UKF']))
                   for _ in [0])
    print(f"# mean ordering KF > FME > AFME: {'OK' if order_ok else 'CHECK'}")

    # ---------------------------------------------------------- figures
    N, L = res["AFME"]["N"], res["AFME"]["lam"]
    for i, meta in enumerate(metas):
        sc = meta.get("scenario", meta)
        wins = meta_windows(meta)
        title = f"Manual flight #{i} ({sc.get('type', '?')})"

        # RMSE time series (all four filters)
        fig, ax = plt.subplots(figsize=(4.6, 2.7))
        lo, hi = np.inf, -np.inf
        for m in METHODS:
            c = moving_rms(err_norm(res[m]["evec"])[i], cfg, a.smooth)
            lo, hi = min(lo, c[sl].min()), max(hi, c[sl].max())
            ax.plot(t, c, "--" if m == "FME" else "-", color=COLORS[m],
                    lw=2.2 if m == "AFME" else 1.5,
                    label=m + (" (proposed)" if m == "AFME" else ""))
        for (s0, s1, lab, shade) in wins:
            ax.axvspan(s0, s1, color=shade, alpha=0.10)
        span = hi - lo
        ax.set_ylim(max(0, lo - 0.08 * span),
                    hi + (0.24 if wins else 0.08) * span)
        import matplotlib.transforms as mtr
        tr = mtr.blended_transform_factory(ax.transData, ax.transAxes)
        for (s0, s1, lab, _) in wins:
            ax.text(0.5 * (s0 + s1), a.label_y, lab, transform=tr,
                    ha="center", va="top", fontsize=6.5, color="#555555",
                    style="italic")
        ax.set_title(title, fontsize=10.5, pad=22)
        ax.set_xlabel("time [s]"); ax.set_ylabel("RMSE [m]")
        ax.set_xlim(EVAL_T0 + 1, t_end); ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, ncol=4, frameon=False, loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0,
                  handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, f"manual{i:02d}_rmse.pdf"),
                    bbox_inches="tight", dpi=150)
        fig.savefig(os.path.join(a.outdir, f"manual{i:02d}_rmse.png"),
                    bbox_inches="tight", dpi=130)
        plt.close(fig)

        # N / lambda adaptation
        fig, ax = plt.subplots(2, 1, figsize=(4.6, 3.4), sharex=True)
        ax[0].plot(t, moving_avg(N[i], cfg, 0.3), color=COLORS["AFME"],
                   lw=1.8, label="AFME (proposed)")
        ax[0].axhline(FME_N, color=COLORS["FME"], ls="--", lw=1.3,
                      label=f"FME ($N{{=}}{FME_N}$)")
        ax[1].plot(t, moving_avg(L[i], cfg, 0.3), color=COLORS["AFME"], lw=1.8)
        for (s0, s1, lab, _) in wins:
            ax[0].axvspan(s0, s1, color="#888", alpha=0.13)
            ax[1].axvspan(s0, s1, color="#888", alpha=0.13)
        ax[0].set_title(title, fontsize=10.5, pad=22)
        ax[0].set_ylabel(r"$N_k$"); ax[1].set_ylabel(r"$\lambda_k$")
        ax[1].set_ylim(cfg.lam_min - 0.02, 1.0)
        ax[1].set_xlabel("time [s]"); ax[1].set_xlim(EVAL_T0 + 1, t_end)
        ax[0].grid(alpha=0.25); ax[1].grid(alpha=0.25)
        ax[0].legend(fontsize=7.5, ncol=2, frameon=False, loc="lower center",
                     bbox_to_anchor=(0.5, 1.0), borderaxespad=0.0)
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, f"manual{i:02d}_adapt.pdf"),
                    bbox_inches="tight", dpi=150)
        fig.savefig(os.path.join(a.outdir, f"manual{i:02d}_adapt.png"),
                    bbox_inches="tight", dpi=130)
        plt.close(fig)

    print(f"# wrote {2 * M} figures to {a.outdir}/")


if __name__ == "__main__":
    main()
