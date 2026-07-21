#!/usr/bin/env python3
"""
3-D trajectory figure: ground truth + estimator tracks + UWB anchors, with
the disturbance window highlighted and an optional zoomed panel -- the same
composition as the trajectory figures in recent UWB-UAV papers (GT solid,
estimators dashed/dotted, anchors as yellow squares).

  python3 tools/plot_traj3d.py --data_dir data_manual15 \
      --ckpt results/v12_50k/ckpt.pt --seed 13 --traj 2 --outdir figures_manual

  # all trajectories, no zoom panel
  python3 tools/plot_traj3d.py --data_dir data_manual15 --ckpt ... --no-zoom
"""
import argparse
import glob
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from _common import (load_cfg, make_dataset, make_runner, load_agent, run_all,
                     COLORS, METHODS, Q_EKF, Q_UKF, FME_N)

GT_GREEN = "#0bb04a"


def meta_windows(sc):
    out = []
    if sc.get("sustained"):
        for w in sc["sustained"]:
            out.append((w["start_s"], w["start_s"] + w["duration_s"],
                        "sustained wind"))
    if sc.get("mass"):
        m = sc["mass"]
        out.append((m["onset_s"], m["onset_s"] + m["duration_s"],
                    "payload attached"))
    return out


def draw_scene(ax, cfg, gt, est, wins, dt, stride, title, zoom=None):
    anchors = np.array([list(a) for a in cfg.anchors], dtype=float)
    # anchor frame: markers + drop lines + dashed ground rectangle
    ax.scatter(anchors[:, 0], anchors[:, 1], anchors[:, 2], marker="s",
               s=48, c="#ffd400", edgecolors="k", linewidths=0.6,
               depthshade=False, label="UWB anchors", zorder=5)
    for (axx, ayy, azz) in anchors:
        if azz > 0:
            ax.plot([axx, axx], [ayy, ayy], [0, azz], ls="--", lw=0.7,
                    color="#39c0d4", alpha=0.7)
    xs = [anchors[:, 0].min(), anchors[:, 0].max()]
    ys = [anchors[:, 1].min(), anchors[:, 1].max()]
    rect = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]),
            (xs[0], ys[1]), (xs[0], ys[0])]
    ax.plot([p[0] for p in rect], [p[1] for p in rect], 0, ls="--", lw=0.7,
            color="#39c0d4", alpha=0.7)

    T = gt.shape[0]
    mask = np.zeros(T, dtype=bool)
    for (s0, s1, _) in wins:
        mask[int(s0 / dt):min(int(s1 / dt), T)] = True

    # estimator tracks (downsampled, thin) UNDER the ground truth
    order = [m for m in METHODS if m in est]
    styles = {"EKF": ":", "UKF": "-.", "FME": "--", "AFME": "-"}
    for m in order:
        e = est[m][::stride]
        ax.plot(e[:, 0], e[:, 1], e[:, 2], styles[m], color=COLORS[m],
                lw=1.6 if m == "AFME" else 1.0,
                alpha=0.95 if m == "AFME" else 0.8,
                label=m + (" (proposed)" if m == "AFME" else ""))

    # ground truth: normal part green, disturbance part dark red + thicker
    seg = np.stack([gt[:-1, :3], gt[1:, :3]], axis=1)
    segmask = mask[:-1]
    lc0 = Line3DCollection(seg[~segmask], colors=GT_GREEN, linewidths=2.4)
    ax.add_collection3d(lc0)
    if segmask.any():
        lc1 = Line3DCollection(seg[segmask], colors="#b3121b", linewidths=3.0)
        ax.add_collection3d(lc1)
        c = gt[mask][:, :3].mean(0)
        ax.text(c[0], c[1], gt[mask][:, 2].max() + 0.55, wins[0][2],
                color="#b3121b", fontsize=8.5, style="italic", ha="center")
    ax.plot([], [], color=GT_GREEN, lw=2.4, label="Ground truth")
    if segmask.any():
        ax.plot([], [], color="#b3121b", lw=3.0, label="GT (disturbance)")
    ax.scatter(*gt[0, :3], marker="o", s=30, c="k", depthshade=False)
    ax.text(gt[0, 0], gt[0, 1], gt[0, 2] + 0.25, "start", fontsize=7.5)

    if zoom is None:
        ax.set_xlim(xs[0] - 0.5, xs[1] + 0.5)
        ax.set_ylim(ys[0] - 0.5, ys[1] + 0.5)
        ax.set_zlim(0, max(anchors[:, 2].max(), gt[:, 2].max()) + 0.5)
    else:
        p = gt[mask][:, :3] if mask.any() else gt[:, :3]
        pad = 0.6
        ax.set_xlim(p[:, 0].min() - pad, p[:, 0].max() + pad)
        ax.set_ylim(p[:, 1].min() - pad, p[:, 1].max() + pad)
        ax.set_zlim(max(0, p[:, 2].min() - pad), p[:, 2].max() + pad)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.set_zlabel("z (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    try:
        ax.set_box_aspect((np.ptp(ax.get_xlim()), np.ptp(ax.get_ylim()),
                           np.ptp(ax.get_zlim())))
    except Exception:
        pass
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--traj", type=int, default=-1,
                    help="trajectory index; -1 = all")
    ap.add_argument("--outdir", default="figures_manual")
    ap.add_argument("--stride", type=int, default=3,
                    help="downsampling of estimator tracks (plot clarity)")
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--elev", type=float, default=24.0)
    ap.add_argument("--azim", type=float, default=-60.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    cfg = load_cfg(a.data_dir)
    ds, M = make_dataset(cfg, a.device)
    run = make_runner(cfg, ds, a.device, a.seed)
    agent = load_agent(cfg, a.ckpt, a.device)
    res = run_all(run, cfg, a.device, M, agent,
                  q_ekf=Q_EKF, q_ukf=Q_UKF, fme_N=FME_N)
    metas = [json.load(open(f)) for f in
             sorted(glob.glob(os.path.join(a.data_dir, a.split, "meta_*.json")))]

    gt_all = ds.gt.cpu().numpy() if hasattr(ds.gt, "cpu") else np.asarray(ds.gt)
    idxs = range(M) if a.traj < 0 else [a.traj]
    for i in idxs:
        sc = metas[i].get("scenario", metas[i])
        wins = meta_windows(sc)
        gt = gt_all[i]
        # evec = gt - est (evaluate.py L153), so est = gt - evec
        est = {m: gt[:, :3] - np.asarray(res[m]["evec"])[i] for m in METHODS}

        panels = 1 if (a.no_zoom or not wins) else 2
        fig = plt.figure(figsize=(4.6 * panels + 0.4, 4.4))
        ax = fig.add_subplot(1, panels, 1, projection="3d")
        draw_scene(ax, cfg, gt, est, wins, cfg.dt, a.stride,
                   f"Manual flight #{i} ({sc.get('type', '?')})")
        ax.view_init(elev=a.elev, azim=a.azim)
        ax.legend(fontsize=6.5, loc="upper left", frameon=False,
                  borderaxespad=0.1)
        if panels == 2:
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            draw_scene(ax2, cfg, gt, est, wins, cfg.dt, max(1, a.stride // 2),
                       "disturbance window (zoom)", zoom=True)
            ax2.view_init(elev=a.elev, azim=a.azim)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.outdir, f"traj3d_{i:02d}.{ext}"),
                        bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"wrote traj3d_{i:02d}.pdf/.png")


if __name__ == "__main__":
    main()
