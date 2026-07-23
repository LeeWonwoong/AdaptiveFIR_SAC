#!/usr/bin/env python3
"""
Generate the two paper figures for one seed -- one per disturbance scenario.

  fig_wind       wind scenario:    error / N_k / lambda_k stacked vertically
  fig_payload    payload scenario: error / N_k / lambda_k stacked vertically

Each figure has three panels sharing one time axis:
  [0] 3-D position error of all four estimators (EKF, UKF, FME, AFME)
  [1] adapted horizon N_k of AFME vs the fixed FME window
  [2] adapted forgetting factor lambda_k of AFME vs the fixed FME value
The shared x-axis makes the disturbance shading line up across the panels.

Curves are the RMS across the three flight patterns of the scenario, shown as
a centred moving RMS (--smooth, default 1.5 s).  The x-range equals the RMSE
evaluation window (2-40 s), so figures and table describe the same interval.
Set FIG_PATTERN in tools/_common.py (or pass --fig-pattern helical/figure8/
waypoint) to draw ONE representative pattern instead of the all-pattern RMS.

  python3 tools/make_figs.py --data_dir data_isaac_v12 \
      --ckpt results/v12_50k/ckpt.pt --seed 13 --outdir figures/
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 7.5, "axes.labelsize": 7.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 0.9, "pdf.fonttype": 42,
})

from _common import (DISPLAY, load_cfg, scenario_index, make_dataset,
                     load_agent, run_all_seeded, default_method_seeds,
                     SEED, FIG_PATTERN, select_pattern,
                     eval_slice, err_norm, moving_rms,
                     moving_avg, METHODS,
                     COLORS, EVAL_T0, EVAL_T1,
                     WIND_WINDOWS, PAYLOAD_WINDOWS, Q_EKF, Q_UKF,
                     Q_EKF_DIST, Q_UKF_DIST, R_EKF, R_UKF,
                     R_EKF_DIST, R_UKF_DIST, FME_N, FME_LAM)


def _annotate(ax, wins, label, color, size, y=0.975):
    """Write `label` inside each shaded disturbance band."""
    if not label:
        return
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for (s0, s1) in wins:
        ax.text(0.5 * (s0 + s1), y, label, transform=tr,
                ha="center", va="top", fontsize=size, color=color,
                style="italic")


def _legend(ax, mode, ncol):
    """Place the legend so it never sits on top of the curves."""
    if mode == "none":
        return
    if mode == "above":
        ax.legend(fontsize=7.5, ncol=ncol, frameon=False,
                  loc="lower center", bbox_to_anchor=(0.5, 1.0),
                  borderaxespad=0.0, handlelength=1.6,
                  columnspacing=1.1, handletextpad=0.5)
    else:
        ax.legend(fontsize=8, ncol=2, frameon=False, loc="upper right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    # per-method noise-seed overrides; a method without an override uses the
    # SEED_* constant from tools/_common.py, then --seed. Methods sharing a
    # seed still share ONE measurement stream.
    ap.add_argument("--seed-ekf", type=int, default=None)
    ap.add_argument("--seed-ukf", type=int, default=None)
    ap.add_argument("--seed-fme", type=int, default=None)
    ap.add_argument("--seed-afme", type=int, default=None)
    ap.add_argument("--no-lam", action="store_true",
                    help="fig3 = N panel only (use with the lam-fixed "
                         "N-only formulation, where the lambda trace is a "
                         "constant line)")
    ap.add_argument("--avg-seeds", dest="avg_seeds", default="",
                    help='e.g. "1-20": draw Monte-Carlo averaged curves '
                         "(paired streams within each seed are unchanged)")
    ap.add_argument("--fig-pattern", default=FIG_PATTERN,
                    choices=["helical", "figure8", "waypoint"],
                    help="draw ONE representative flight pattern instead of "
                         "the RMS across all three (default: FIG_PATTERN "
                         "in tools/_common.py)")
    ap.add_argument("--pattern-sweep", action="store_true",
                    help="write every variant instead of just --fig-pattern: "
                         "fig{2,3}_<scen>_all (3-pattern RMS) plus "
                         "_helical/_figure8/_waypoint; the filter rollouts "
                         "run only once")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--smooth", type=float, default=1.5)
    ap.add_argument("--adapt-smooth", type=float, default=0.3,
                    help="moving-average window [s] for the N_k/lambda_k "
                         "traces in fig3")
    ap.add_argument("--raw-alpha", type=float, default=0.0,
                    help="opacity of the raw per-step N_k/lambda_k trace "
                         "drawn behind the smoothed line (0 = off; e.g. "
                         "0.3 with --adapt-smooth 0.8 for a raw underlay)")
    ap.add_argument("--wind-label", default="sustained wind",
                    help="text drawn inside the wind shading ('' to disable)")
    ap.add_argument("--payload-label", default="payload attached",
                    help="text drawn inside the payload shading ('' to disable)")
    ap.add_argument("--label-size", type=float, default=7.5,
                    help="font size of the payload label")
    ap.add_argument("--wind-label-size", type=float, default=6.5,
                    help="font size of the (smaller) wind label")
    ap.add_argument("--label-color", default="#555555")
    ap.add_argument("--label-y", type=float, default=0.975,
                    help="vertical position of the band labels "
                         "(1.0 = top of the axes)")
    ap.add_argument("--legend", default="above",
                    choices=["above", "inside", "none"],
                    help="legend placement; 'above' avoids overlapping curves")
    ap.add_argument("--width", type=float, default=4.2)
    ap.add_argument("--q-ekf", "--q-ekf-nom", dest="q_ekf",
                    type=float, default=Q_EKF,
                    help="EKF process noise in nominal flight")
    ap.add_argument("--q-ukf", "--q-ukf-nom", dest="q_ukf",
                    type=float, default=Q_UKF,
                    help="UKF process noise in nominal flight")
    ap.add_argument("--q-ekf-dist", type=float, default=Q_EKF_DIST,
                    help="EKF process noise under disturbance (wind/payload)")
    ap.add_argument("--q-ukf-dist", type=float, default=Q_UKF_DIST,
                    help="UKF process noise under disturbance (wind/payload)")
    # KF measurement noise (R), scalar x datasheet meas_sigma, same nominal /
    # disturbance split as Q.
    ap.add_argument("--r-ekf", "--r-ekf-nom", dest="r_ekf",
                    type=float, default=R_EKF,
                    help="EKF meas-noise scale in nominal flight")
    ap.add_argument("--r-ukf", "--r-ukf-nom", dest="r_ukf",
                    type=float, default=R_UKF,
                    help="UKF meas-noise scale in nominal flight")
    ap.add_argument("--r-ekf-dist", type=float, default=R_EKF_DIST,
                    help="EKF meas-noise scale under disturbance (wind/payload)")
    ap.add_argument("--r-ukf-dist", type=float, default=R_UKF_DIST,
                    help="UKF meas-noise scale under disturbance (wind/payload)")
    ap.add_argument("--fme-n", type=int, default=FME_N)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    # the wind / payload figures are disturbance scenarios, so the EKF / UKF
    # curves must be drawn with the SAME disturbance-regime (Q,R) rollout the
    # RMSE table (eval_seeds) scores them with; else figure and table disagree.
    split_q = (a.q_ekf_dist != a.q_ekf) or (a.q_ukf_dist != a.q_ukf) \
        or (a.r_ekf_dist != a.r_ekf) or (a.r_ukf_dist != a.r_ukf)

    os.makedirs(a.outdir, exist_ok=True)
    cfg = load_cfg(a.data_dir)
    scen = scenario_index(cfg)
    ds, M = make_dataset(cfg, a.device)
    agent = load_agent(cfg, a.ckpt, a.device)

    # --avg-seeds "1-20": Monte-Carlo averaged curves. Evaluation methodology
    # is unchanged (within each seed all filters share ONE measurement
    # stream = paired comparison); averaging across seeds only removes the
    # realization-specific wiggle from the DISPLAYED curves, the standard MC
    # presentation in the filtering literature. evec entries are replaced by
    # sqrt(mean over seeds of squared error) per step; N/lam are averaged.
    # seed precedence: --seed-<m> CLI flag > SEED_<M> in _common.py > --seed
    method_seeds = default_method_seeds()
    method_seeds.update({m: s for m, s in
                         [("EKF", a.seed_ekf), ("UKF", a.seed_ukf),
                          ("FME", a.seed_fme), ("AFME", a.seed_afme)]
                         if s is not None})
    if a.avg_seeds:
        if method_seeds:
            ap.error("--seed-ekf/--seed-ukf/--seed-fme/--seed-afme are "
                     "single-seed options; drop --avg-seeds to use them")
        lo, hi = (int(x) for x in a.avg_seeds.split("-"))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [a.seed]
    res = None
    for si, sd in enumerate(seeds):
        r1 = run_all_seeded(cfg, ds, a.device, M, agent,
                            seed=sd, method_seeds=method_seeds,
                            q_ekf=a.q_ekf, q_ukf=a.q_ukf,
                            r_ekf=a.r_ekf, r_ukf=a.r_ukf, fme_N=a.fme_n,
                            q_ekf_dist=a.q_ekf_dist if split_q else None,
                            q_ukf_dist=a.q_ukf_dist if split_q else None,
                            r_ekf_dist=a.r_ekf_dist if split_q else None,
                            r_ukf_dist=a.r_ukf_dist if split_q else None)
        if res is None:
            res = {m: {k: (np.asarray(v, dtype=float) ** 2
                           if k.startswith("evec") else
                           np.asarray(v, dtype=float))
                       for k, v in d.items()} for m, d in r1.items()}
        else:
            for m, d in r1.items():
                for k, v in d.items():
                    res[m][k] += (np.asarray(v, dtype=float) ** 2
                                  if k.startswith("evec") else
                                  np.asarray(v, dtype=float))
    for m, d in res.items():
        for k in d:
            d[k] /= len(seeds)
            if k.startswith("evec"):
                d[k] = np.sqrt(d[k])          # per-step RMS error over seeds
    if len(seeds) > 1:
        print(f"[make_figs] curves averaged over {len(seeds)} seeds "
              f"({seeds[0]}..{seeds[-1]})")

    def evec_for(m, scenario):
        """EKF/UKF use the disturbance-regime (Q,R) rollout in the wind/payload
        figures; every other filter (and the nominal case) uses the single
        rollout."""
        d = res[m]
        if scenario != "nominal" and "evec_dist" in d:
            return d["evec_dist"]
        return d["evec"]

    t = np.arange(ds.T) * cfg.dt     # x-axis: time in seconds
    xticks = [10, 20, 30, 40]
    methods = list(METHODS)          # EKF, UKF, FME, AFME -- always all four

    # per-method line style / colour for the error panel:
    #   EKF -> black dotted   UKF -> its colour dashed
    #   FME -> dash-dot       AFME -> solid (proposed, thicker)
    STYLE = {"EKF": ":", "UKF": "--", "FME": "-.", "AFME": "-"}

    labels = {"wind": (a.wind_label, a.wind_label_size),
              "payload": (a.payload_label, a.label_size)}

    wind_speed = next((r[1] for r in cfg.heldout_plan
                       if r[0] == "sustained_wind"), None)
    cases = [
        ("wind", f"Wind ({wind_speed:g} m/s)" if wind_speed else "Wind",
         scen["wind"], WIND_WINDOWS, "#d62728"),
        ("payload", "Payload", scen["payload"], PAYLOAD_WINDOWS, "#ff7f0e"),
    ]

    # ---- per-scenario figures (paper convention) ----
    # fig2_<scen>: ONE panel, localization error only, x-axis in time steps k
    # fig3_<scen>: two stacked panels N_k / lambda_k sharing the same axis
    k0, k1 = int((EVAL_T0 + 1) / cfg.dt), int(EVAL_T1 / cfg.dt)
    tk = np.arange(ds.T)                     # x-axis: time step k
    kticks = [500, 1000, 1500, 2000]
    N, L = res["AFME"]["N"], res["AFME"]["lam"]
    # --pattern-sweep: same rollouts, but the plot loop below runs once per
    # (pattern, suffix) variant; (None, "_all") is the 3-pattern RMS.
    if a.pattern_sweep:
        pat_list = [(None, "_all"), ("helical", "_helical"),
                    ("figure8", "_figure8"), ("waypoint", "_waypoint")]
    else:
        pat_list = [(a.fig_pattern, "")]
    for (pat, sfx), (key, title, idx, wins, shade) in (
            (p, c) for p in pat_list for c in cases):
        # --fig-pattern: one representative trajectory instead of the
        # all-pattern RMS (idx then has a single element, mean(0) is a no-op)
        idx = select_pattern(cfg, idx, pat)
        if a.pattern_sweep:            # self-describing titles in sweep mode
            title = f"{title} – {pat or 'all-pattern RMS'}"
        wins_k = [(s0 / cfg.dt, s1 / cfg.dt) for (s0, s1) in wins]
        _lab, _sz = labels[key]

        # ── fig2: localization error, single panel ──
        fig, ax = plt.subplots(figsize=(a.width, 2.6),
                               constrained_layout=True)
        lo, hi = np.inf, -np.inf
        for m in methods:
            c = moving_rms(np.sqrt((err_norm(evec_for(m, key))[idx] ** 2).mean(0)),
                           cfg, a.smooth)
            lo, hi = min(lo, c[k0:k1].min()), max(hi, c[k0:k1].max())
            ax.plot(tk, c, STYLE[m],
                    color="black" if m == "EKF" else COLORS[m],
                    lw=2.0 if m == "AFME" else 1.2,
                    label=DISPLAY[m]
                    + (" (proposed)" if m == "AFME" else ""))
        for (w0, w1) in wins_k:
            ax.axvspan(w0, w1, color=shade, alpha=0.12, zorder=0)
        _annotate(ax, wins_k, _lab, a.label_color, _sz, a.label_y)
        span = hi - lo
        ax.set_ylim(max(0.0, lo - 0.08 * span), hi + 0.12 * span)
        ax.set_xlim(k0, k1)
        ax.set_xticks(kticks)
        ax.set_ylabel("Localization error [m]")
        ax.set_xlabel("time step ($k$)")
        ax.grid(alpha=0.25)
        ax.legend(ncol=len(methods), loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), frameon=False,
                  columnspacing=1.0, handlelength=1.6, handletextpad=0.4)
        fig.suptitle(title, y=1.16, fontsize=11)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.outdir, f"fig2_{key}{sfx}.{ext}"),
                        bbox_inches="tight", dpi=150)
        plt.close(fig)

        # ── fig3: adaptation, N_k over lambda_k (or N only) ──
        nrows = 1 if a.no_lam else 2
        fig, ax = plt.subplots(nrows, 1,
                               figsize=(a.width, 2.2 if a.no_lam else 3.1),
                               sharex=True, constrained_layout=True)
        if nrows == 1:
            ax = [ax]
        _bt = __import__('matplotlib.transforms',
                         fromlist=['x']).blended_transform_factory
        nraw = N[idx].mean(0)
        nc = moving_avg(nraw, cfg, a.adapt_smooth)
        if a.raw_alpha > 0:      # raw per-step trace as a faint underlay
            ax[0].plot(tk, nraw, "-", color="k", lw=0.6,
                       alpha=a.raw_alpha, zorder=1)
        ax[0].plot(tk, nc, "-", color="k", lw=1.3, zorder=2)
        ax[0].axhline(a.fme_n, ls="--", color=COLORS["FME"], lw=0.9)
        ax[0].text(0.995, a.fme_n, f"FME $N{{=}}{a.fme_n}$",
                   transform=_bt(ax[0].transAxes, ax[0].transData),
                   ha="right", va="bottom", fontsize=6.5, color=COLORS["FME"])
        ax[0].set_ylabel(r"$N_k$")
        # y-range from the raw trace when it is visible, so it isn't clipped
        nb = nraw if a.raw_alpha > 0 else nc
        nlo, nhi = nb[k0:k1].min(), max(nb[k0:k1].max(), a.fme_n)
        ax[0].set_ylim(max(cfg.N_min - 2, nlo - 2), min(cfg.N_max, nhi + 2))
        # N is a window length -> integer ticks only (no 7.5 / 12.5 labels)
        from matplotlib.ticker import MaxNLocator
        ax[0].yaxis.set_major_locator(MaxNLocator(integer=True))

        if not a.no_lam:
            lraw = L[idx].mean(0)
            if a.raw_alpha > 0:
                ax[1].plot(tk, lraw, "-", color="k", lw=0.6,
                           alpha=a.raw_alpha, zorder=1)
            ax[1].plot(tk, moving_avg(lraw, cfg, a.adapt_smooth), "-",
                       color="k", lw=1.3, zorder=2)
        if not a.no_lam:
            ax[1].axhline(FME_LAM, ls="--", color=COLORS["FME"], lw=0.9)
        if not a.no_lam:
            ax[1].text(0.995, FME_LAM, f"FME $\\lambda{{=}}{FME_LAM:g}$",
                       transform=_bt(ax[1].transAxes, ax[1].transData),
                       ha="right", va="bottom", fontsize=6.5,
                       color=COLORS["FME"])
        if not a.no_lam:
            ax[1].set_ylabel(r"$\lambda_k$")
            ax[1].set_ylim(cfg.lam_min - 0.02, 1.0)
        ax[-1].set_xlabel("time step ($k$)")

        for A in ax:
            for (w0, w1) in wins_k:
                A.axvspan(w0, w1, color=shade, alpha=0.12, zorder=0)
            A.set_xlim(k0, k1)
            A.grid(alpha=0.25)
        ax[0].set_xticks(kticks)
        _annotate(ax[0], wins_k, _lab, a.label_color, _sz, a.label_y)
        ax[0].set_title(title, fontsize=11, pad=6)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.outdir, f"fig3_{key}{sfx}.{ext}"),
                        bbox_inches="tight", dpi=150)
        plt.close(fig)

    outabs = os.path.abspath(a.outdir)
    qnote = (f"Q_EKF={a.q_ekf:g}/{a.q_ekf_dist:g} Q_UKF={a.q_ukf:g}/{a.q_ukf_dist:g} "
             f"R_EKF={a.r_ekf:g}/{a.r_ekf_dist:g} R_UKF={a.r_ukf:g}/{a.r_ukf_dist:g}"
             " (nominal/disturb)" if split_q
             else f"Q_EKF={a.q_ekf:g} Q_UKF={a.q_ukf:g} "
                  f"R_EKF={a.r_ekf:g} R_UKF={a.r_ukf:g}")
    if method_seeds:
        snote = "seeds " + " ".join(
            f"{m}={method_seeds.get(m, a.seed)}" for m in METHODS)
    else:
        snote = f"seed {a.seed}"
    pnote = ("pattern sweep (all/helical/figure8/waypoint)" if a.pattern_sweep
             else f"pattern {a.fig_pattern}" if a.fig_pattern
             else "all-pattern RMS")
    nfig = 4 * len(pat_list)
    print(f"wrote {nfig} figures to {outabs} "
          f"({snote}, {pnote}, {a.smooth:g}s moving RMS, "
          f"{a.adapt_smooth:g}s N/lam avg + raw alpha {a.raw_alpha:g}, "
          f"filters: {', '.join(methods)}, {qnote})")
    for _pat, sfx in pat_list:
        for key in ("wind", "payload"):
            for pfx in ("fig2", "fig3"):
                print(f"  {os.path.join(outabs, f'{pfx}_{key}{sfx}.png')}")


if __name__ == "__main__":
    main()
