#!/usr/bin/env python3
"""
Per-seed, per-scenario evaluation of EKF / UKF / FME / AFME.

Unlike `evaluate.py` (which writes one summary.csv per run and pools all
scenarios) this splits the held-out set by scenario, applies the paper's
2-40 s window, and checks the acceptance criteria used to pick the reported
seed:

  1. ordering        EKF, UKF > FME > AFME in every scenario
  2. payload > nominal for all four filters
  3. |UKF - EKF| in nominal (the two should be close: near-linear regime)
  4. FME spike       how far FME rises above the better KF inside the gusts
  5. adaptation      mean N nominal vs in-window, and the lambda range used

Examples
--------
  python3 tools/eval_seeds.py --data_dir data_isaac_v12 \
      --ckpt results/v12_50k/ckpt.pt --seeds 13,15,18

  # scan many seeds, dump machine-readable output
  python3 tools/eval_seeds.py --data_dir data_isaac_v12 \
      --ckpt results/v12_50k/ckpt.pt --seeds 1-20 --json seeds.json
"""
import argparse
import json
import numpy as np

from _common import (load_cfg, scenario_index, make_dataset, make_runner,
                     load_agent, run_all, eval_slice, rmse3, err_norm,
                     moving_rms, window_mask, METHODS,
                     WIND_WINDOWS, PAYLOAD_WINDOWS, Q_EKF, Q_UKF,
                     Q_EKF_DIST, Q_UKF_DIST, R_EKF, R_UKF,
                     R_EKF_DIST, R_UKF_DIST, FME_N)


def parse_seeds(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", default=None,
                    help="SAC checkpoint; omit to evaluate baselines only")
    ap.add_argument("--seeds", default="13")
    # KF process noise, split into the two flight regimes: the "nominal"
    # (near-linear) columns use --q-*-nom, the disturbance columns (wind /
    # payload) use --q-*-dist. --q-ekf/--q-ukf still set the nominal value so
    # existing command lines keep working; the disturbance level defaults to
    # the nominal one, i.e. no split unless you pass --q-*-dist.
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
    ap.add_argument("--smooth", type=float, default=1.5,
                    help="moving-RMS window [s] for the spike metric")
    ap.add_argument("--json", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    # a split kicks in only when the disturbance Q or R actually differs from
    # the nominal one (defaults *_DIST currently equal the nominal → no split
    # unless the constants in _common.py are changed or --*-dist is passed)
    q_ekf_dist, q_ukf_dist = a.q_ekf_dist, a.q_ukf_dist
    r_ekf_dist, r_ukf_dist = a.r_ekf_dist, a.r_ukf_dist
    split_q = (q_ekf_dist != a.q_ekf) or (q_ukf_dist != a.q_ukf) \
        or (r_ekf_dist != a.r_ekf) or (r_ukf_dist != a.r_ukf)

    cfg = load_cfg(a.data_dir)
    scen = scenario_index(cfg)
    ds, M = make_dataset(cfg, a.device)
    sl = eval_slice(cfg, ds.T)
    agent = load_agent(cfg, a.ckpt, a.device) if a.ckpt else None
    methods = METHODS if agent is not None else ["EKF", "UKF", "FME"]

    wmask = window_mask(cfg, WIND_WINDOWS, ds.T)
    pmask = window_mask(cfg, PAYLOAD_WINDOWS, ds.T)

    if split_q:
        qline = (f"Q_EKF={a.q_ekf:.1e}/{q_ekf_dist:.1e} "
                 f"Q_UKF={a.q_ukf:.1e}/{q_ukf_dist:.1e} "
                 f"R_EKF={a.r_ekf:g}/{r_ekf_dist:g} "
                 f"R_UKF={a.r_ukf:g}/{r_ukf_dist:g} (nominal/disturb)")
    else:
        qline = (f"Q_EKF={a.q_ekf:.1e} Q_UKF={a.q_ukf:.1e} "
                 f"R_EKF={a.r_ekf:g} R_UKF={a.r_ukf:g}")
    print(f"# window {sl.start * cfg.dt:.0f}-{sl.stop * cfg.dt:.0f}s | "
          f"{qline} FME N={a.fme_n} | 3-D RMSE [m]")

    def evec_for(m, scenario):
        """EKF/UKF use the disturbance-regime rollout (its own Q,R) for
        wind/payload, the nominal rollout for nominal; the other filters have
        no Q,R to split."""
        d = res[m]
        if scenario != "nominal" and "evec_dist" in d:
            return d["evec_dist"]
        return d["evec"]

    allout = {}
    for s in parse_seeds(a.seeds):
        run = make_runner(cfg, ds, a.device, s)
        res = run_all(run, cfg, a.device, M, agent,
                      q_ekf=a.q_ekf, q_ukf=a.q_ukf,
                      r_ekf=a.r_ekf, r_ukf=a.r_ukf, fme_N=a.fme_n,
                      q_ekf_dist=q_ekf_dist if split_q else None,
                      q_ukf_dist=q_ukf_dist if split_q else None,
                      r_ekf_dist=r_ekf_dist if split_q else None,
                      r_ukf_dist=r_ukf_dist if split_q else None)

        tab = {m: [rmse3(evec_for(m, k), scen[k], sl)
                   for k in ("nominal", "wind", "payload")] for m in methods}

        # spike: FME above the better KF inside the gust windows (KFs scored
        # with their disturbance-Q rollout, matching the wind column)
        env = np.maximum(
            moving_rms(np.sqrt((err_norm(evec_for("EKF", "wind"))[scen["wind"]] ** 2).mean(0)), cfg, a.smooth),
            moving_rms(np.sqrt((err_norm(evec_for("UKF", "wind"))[scen["wind"]] ** 2).mean(0)), cfg, a.smooth))
        cf = moving_rms(np.sqrt((err_norm(res["FME"]["evec"])[scen["wind"]] ** 2).mean(0)), cfg, a.smooth)
        spike = float((cf[wmask] - env[wmask]).max() * 1000)

        print(f"\n=== seed {s} ===")
        print(f"{'':>9} | " + " ".join(f"{m:>7}" for m in methods))
        for j, k in enumerate(("nominal", "wind", "payload")):
            print(f"{k:>9} | " + " ".join(f"{tab[m][j]:7.3f}" for m in methods))
        print(f"{'average':>9} | " + " ".join(f"{np.mean(tab[m]):7.3f}" for m in methods))

        order = all(tab["FME"][j] < min(tab["EKF"][j], tab["UKF"][j])
                    for j in range(3))
        if "AFME" in tab:
            order = order and all(tab["AFME"][j] < tab["FME"][j] for j in range(3))
        paynom = all(tab[m][2] > tab[m][0] for m in methods)
        dnom = (tab["UKF"][0] - tab["EKF"][0]) * 1000
        print(f"  ordering {'OK' if order else 'FAIL'} | "
              f"payload>nominal {'OK' if paynom else 'FAIL'} | "
              f"nominal UKF-EKF {dnom:+.1f}mm | FME over KF {spike:+.0f}mm")

        rec = {m: tab[m] for m in methods}
        rec["spike_mm"] = spike
        rec["nominal_ukf_minus_ekf_mm"] = dnom

        if "AFME" in res and res["AFME"]["N"] is not None:
            N, L = res["AFME"]["N"], res["AFME"]["lam"]
            n_nom = float(N[scen["nominal"]][:, sl].mean())
            n_w = float(N[scen["wind"]][:, wmask].mean())
            n_p = float(N[scen["payload"]][:, pmask].mean())
            l_lo, l_hi = float(L[:, sl].min()), float(L[:, sl].max())
            print(f"  adaptation: N {n_nom:.1f} -> wind {n_w:.1f} / payload {n_p:.1f}"
                  f" | lambda [{l_lo:.2f}, {l_hi:.2f}]")
            rec["adapt"] = dict(N_nominal=n_nom, N_wind=n_w, N_payload=n_p,
                                lam_min=l_lo, lam_max=l_hi)
        allout[str(s)] = rec

    if a.json:
        with open(a.json, "w") as f:
            json.dump(allout, f, indent=1)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
