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

from _common import (load_cfg, scenario_index, make_dataset, make_runner,
                     load_agent, run_all, eval_slice, err_norm, moving_rms,
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
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--smooth", type=float, default=1.5)
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
    run = make_runner(cfg, ds, a.device, a.seed)
    agent = load_agent(cfg, a.ckpt, a.device)
    res = run_all(run, cfg, a.device, M, agent,
                  q_ekf=a.q_ekf, q_ukf=a.q_ukf,
                  r_ekf=a.r_ekf, r_ukf=a.r_ukf, fme_N=a.fme_n,
                  q_ekf_dist=a.q_ekf_dist if split_q else None,
                  q_ukf_dist=a.q_ukf_dist if split_q else None,
                  r_ekf_dist=a.r_ekf_dist if split_q else None,
                  r_ukf_dist=a.r_ukf_dist if split_q else None)

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

    # ---- combined per-scenario figure: 3 rows (error / N_k / lambda_k) ----
    # One figure per scenario (wind, payload). The three panels share the same
    # time axis so the disturbance shading lines up vertically across error,
    # horizon and forgetting-factor traces.
    k0, k1 = int((EVAL_T0 + 1) / cfg.dt), int(EVAL_T1 / cfg.dt)
    N, L = res["AFME"]["N"], res["AFME"]["lam"]
    for key, title, idx, wins, shade in cases:
        fig, ax = plt.subplots(3, 1, figsize=(a.width, 4.8), sharex=True,
                               constrained_layout=True)

        # [0] 3-D position error -- all four estimators
        lo, hi = np.inf, -np.inf
        for m in methods:
            c = moving_rms(np.sqrt((err_norm(evec_for(m, key))[idx] ** 2).mean(0)),
                           cfg, a.smooth)
            lo, hi = min(lo, c[k0:k1].min()), max(hi, c[k0:k1].max())
            ax[0].plot(t, c, STYLE[m],
                       color="black" if m == "EKF" else COLORS[m],
                       lw=1.6 if m == "AFME" else 1.0,
                       label=m + (" (proposed)" if m == "AFME" else ""))
        ax[0].set_ylabel("Localization error [m]")
        span = hi - lo
        ax[0].set_ylim(max(0.0, lo - 0.08 * span), hi + 0.10 * span)

        # [1] adapted horizon N_k vs the fixed FME window
        nc = moving_avg(N[idx].mean(0), cfg, 0.3)
        ax[1].plot(t, nc, "-", color="k", lw=1.3)
        ax[1].axhline(a.fme_n, ls="--", color=COLORS["FME"], lw=0.9)
        ax[1].text(0.995, a.fme_n, f"FME $N{{=}}{a.fme_n}$",
                   transform=__import__('matplotlib.transforms',fromlist=['x']).blended_transform_factory(ax[1].transAxes,
                                                           ax[1].transData),
                   ha="right", va="bottom", fontsize=6.5, color=COLORS["FME"])
        ax[1].set_ylabel(r"$N_k$")
        nlo, nhi = nc[k0:k1].min(), max(nc[k0:k1].max(), a.fme_n)
        ax[1].set_ylim(max(cfg.N_min - 2, nlo - 2), min(cfg.N_max, nhi + 2))

        # [2] adapted forgetting factor lambda_k vs the fixed FME value
        ax[2].plot(t, moving_avg(L[idx].mean(0), cfg, 0.3), "-", color="k", lw=1.3)
        ax[2].axhline(FME_LAM, ls="--", color=COLORS["FME"], lw=0.9)
        ax[2].text(0.995, FME_LAM, f"FME $\\lambda{{=}}{FME_LAM:g}$",
                   transform=__import__('matplotlib.transforms',fromlist=['x']).blended_transform_factory(ax[2].transAxes,
                                                           ax[2].transData),
                   ha="right", va="bottom", fontsize=6.5, color=COLORS["FME"])
        ax[2].set_ylabel(r"$\lambda_k$")
        ax[2].set_ylim(cfg.lam_min - 0.02, 1.0)
        ax[2].set_xlabel("time [s]")

        # shared: disturbance shading on every row (vertical alignment is the
        # point of the stacked layout), plus common x-range / ticks / grid
        for A in ax:
            for (s0, s1) in wins:
                A.axvspan(s0, s1, color=shade, alpha=0.12, zorder=0)
            A.set_xlim(EVAL_T0 + 1, EVAL_T1)
            A.grid(alpha=0.25)
        ax[0].set_xticks(xticks)
        ax[0].set_title(title, fontsize=11, pad=16)
        _lab, _sz = labels[key]
        _annotate(ax[0], wins, _lab, a.label_color, _sz, a.label_y)

        # single figure legend above the top panel: the four error curves plus
        # the fixed-FME horizon line, de-duplicated so "AFME (proposed)" (which
        # labels both the green error curve and the black N_k/lambda_k trace)
        # appears only once.
        h0, l0 = ax[0].get_legend_handles_labels()
        fig.legend(h0, l0, ncol=len(l0), loc="upper center",
                   bbox_to_anchor=(0.5, 1.045), frameon=False,
                   columnspacing=1.0, handlelength=1.5, handletextpad=0.4)

        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(a.outdir, f"fig_{key}.{ext}"),
                        bbox_inches="tight", dpi=150)
        plt.close(fig)

    outabs = os.path.abspath(a.outdir)
    qnote = (f"Q_EKF={a.q_ekf:g}/{a.q_ekf_dist:g} Q_UKF={a.q_ukf:g}/{a.q_ukf_dist:g} "
             f"R_EKF={a.r_ekf:g}/{a.r_ekf_dist:g} R_UKF={a.r_ukf:g}/{a.r_ukf_dist:g}"
             " (nominal/disturb)" if split_q
             else f"Q_EKF={a.q_ekf:g} Q_UKF={a.q_ukf:g} "
                  f"R_EKF={a.r_ekf:g} R_UKF={a.r_ukf:g}")
    print(f"wrote 2 figures to {outabs}\\ "
          f"(seed {a.seed}, {a.smooth:g}s moving RMS, "
          f"filters: {', '.join(methods)}, {qnote})")
    for key in ("wind", "payload"):
        print(f"  {os.path.join(outabs, f'fig_{key}.png')}")


if __name__ == "__main__":
    main()
