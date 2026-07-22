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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from _common import (load_cfg, make_dataset, make_runner, load_agent, run_all,
                     COLORS, METHODS, Q_EKF, Q_UKF, FME_N)

GT_GREEN = "#0bb04a"


def add_flat_highlight(fig, ax, pts, mask, label):
    """UIFM-SLAC-style FLAT yellow rectangle over the disturbance segment.

    Drawn in the axes 2-D screen space (not a 3-D volume): the segment
    points are projected with the CURRENT view and their screen bounding
    box is shaded. Call AFTER ax.view_init (a canvas draw settles the
    projection first).
    """
    if not mask.any():
        return
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d import proj3d
    fig.canvas.draw()
    p = pts[mask]
    x2, y2, _ = proj3d.proj_transform(p[:, 0], p[:, 1], p[:, 2],
                                      ax.get_proj())
    disp = ax.transData.transform(np.column_stack([x2, y2]))
    fr = ax.transAxes.inverted().transform(disp)
    fx0, fy0 = fr.min(0) - 0.015
    fx1, fy1 = fr.max(0) + 0.015
    rect = mpatches.Rectangle(
        (fx0, fy0), fx1 - fx0, fy1 - fy0, transform=ax.transAxes,
        facecolor="#fff3a3", alpha=0.45, edgecolor="#5a5a5a",
        linewidth=0.9)
    # Axes3D asks every artist for a depth via do_3d_projection(); a plain
    # 2-D patch lacks it. Return a huge depth so the panel is treated as
    # the FARTHEST artist -> drawn first -> stays behind the 3-D tracks.
    rect.do_3d_projection = lambda *a, **k: 1e9
    ax.add_patch(rect)
    ax.text2D(0.5 * (fx0 + fx1), fy1 - 0.012, label,
              transform=ax.transAxes, ha="center", va="top",
              fontsize=8.5, style="italic", color="#6b5b00", zorder=1)


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


def draw_scene(ax, cfg, gt, est, wins, dt, title, zoom=None):
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
        e = est[m]
        ax.plot(e[:, 0], e[:, 1], e[:, 2], styles[m], color=COLORS[m],
                lw=1.5 if m == "AFME" else 0.8,
                alpha=0.95 if m == "AFME" else 0.65,
                label=m + (" (proposed)" if m == "AFME" else ""))

    # ground truth: single thin black line
    ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], "-", color="black", lw=1.1,
            label="Ground truth", zorder=4)
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
        # keep the FULL vertical scale (0 .. anchor plane): a tight z-crop
        # stretches the axis and makes centimetre-level z jitter look wild
        ax.set_zlim(0, max(anchors[:, 2].max(), p[:, 2].max()) + 0.3)
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
    ap.add_argument("--stride", type=int, default=0,
                    help="0 (default) = draw ONLY the post-measurement-update "
                         "estimates at the 10 Hz UWB epochs; prediction "
                         "sub-steps between updates are never drawn. "
                         "Any n>0 = every n-th 50 Hz step.")
    ap.add_argument("--filters", default="EKF,UKF,FME,AFME",
                    help="comma list of estimator tracks to draw (fewer = "
                         "clearer; the quantitative comparison lives in the "
                         "error-time-series figure, not here)")
    ap.add_argument("--est-smooth", type=float, default=0.0,
                    help="optional display-only moving average [s] on the "
                         "estimator tracks (0 = raw estimates)")
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
        # evec = gt - est (evaluate.py L153), so est = gt - evec.
        # Display sampling: estimates are drawn ONLY at the 10 Hz UWB
        # measurement epochs (stride = cfg.uwb_stride). The 50 Hz record
        # also contains the prediction-only sub-steps BETWEEN updates;
        # plotting those adds a drift-and-jump sawtooth that makes the
        # tracks look far noisier than the estimator output actually is.
        us = int(getattr(cfg, "uwb_stride", 5)) if a.stride == 0 else a.stride
        sel = [m.strip() for m in a.filters.split(",") if m.strip() in METHODS]
        if a.stride == 0:
            # measurement-update instants only: evaluate.py fires updates at
            # is_epoch = (t % uwb_stride == 0) with t starting from 1, i.e.
            # t = us, 2*us, ...  Index 0 is the INIT state (no measurement),
            # so start sampling at us.
            est = {m: (gt[:, :3] - np.asarray(res[m]["evec"])[i])[us::us]
                   for m in sel}
        else:
            est = {m: (gt[:, :3] - np.asarray(res[m]["evec"])[i])[::us]
                   for m in sel}
        if a.est_smooth > 0:                     # optional display smoothing
            w = max(1, int(round(a.est_smooth / (cfg.dt * us))))
            if w > 1:
                ker = np.ones(w) / w
                est = {m: np.stack([np.convolve(e[:, j], ker, mode="same")
                                    for j in range(3)], 1)
                       for m, e in est.items()}

        panels = 1 if (a.no_zoom or not wins) else 2
        fig = plt.figure(figsize=(4.6 * panels + 0.4, 4.4))
        ax = fig.add_subplot(1, panels, 1, projection="3d")
        draw_scene(ax, cfg, gt, est, wins, cfg.dt,
                   f"Manual flight #{i} ({sc.get('type', '?')})")
        ax.view_init(elev=a.elev, azim=a.azim)
        ax.legend(fontsize=6.5, loc="upper left", frameon=False,
                  borderaxespad=0.1)
        _mask = np.zeros(gt.shape[0], dtype=bool)
        for (_s0, _s1, _l) in wins:
            _mask[int(_s0 / cfg.dt):min(int(_s1 / cfg.dt), gt.shape[0])] = True
        add_flat_highlight(fig, ax, gt[:, :3], _mask,
                           wins[0][2] if wins else "")
        if panels == 2:
            ax2 = fig.add_subplot(1, 2, 2, projection="3d")
            draw_scene(ax2, cfg, gt, est, wins, cfg.dt,
                       "disturbance window (zoom)", zoom=True)
            ax2.view_init(elev=a.elev, azim=a.azim)
            add_flat_highlight(fig, ax2, gt[:, :3], _mask,
                               wins[0][2] if wins else "")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.outdir, f"traj3d_{i:02d}.{ext}"),
                        bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"wrote traj3d_{i:02d}.pdf/.png")


if __name__ == "__main__":
    main()
