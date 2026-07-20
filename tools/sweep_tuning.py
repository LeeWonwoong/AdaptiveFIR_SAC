#!/usr/bin/env python3
"""
Baseline tuning sweeps used to select the constants reported in the paper.

The selection protocol is the SAME for every baseline: minimise the
NOMINAL-flight RMSE (the disturbances are, by assumption, unknown at tuning
time).  That is how the FME horizon N=10 was chosen, so the Kalman filters
are tuned the same way instead of being handed an arbitrary Q.

  # each Kalman filter's own Q optimum (this is what fixes Q_EKF / Q_UKF)
  python3 tools/sweep_tuning.py --data_dir data_isaac_v12 --mode kf_q

  # UKF sigma-point spread (shows the result is insensitive to alpha)
  python3 tools/sweep_tuning.py --data_dir data_isaac_v12 --mode ukf_alpha

  # fixed-horizon sweep behind the FME baseline
  python3 tools/sweep_tuning.py --data_dir data_isaac_v12 --mode fme_n

Add --all-scenarios to print wind/payload alongside nominal (useful to check
that the nominal-optimal choice does not accidentally break the ordering).
"""
import argparse
import numpy as np

from _common import (load_cfg, scenario_index, make_dataset, make_runner,
                     make_ekf, make_ukf, make_fme, eval_slice, rmse3,
                     err_norm, moving_rms, window_mask, WIND_WINDOWS)


def _row(cfg, sl, scen, ev, all_scen):
    n = rmse3(ev, scen["nominal"], sl)
    if not all_scen:
        return f"{n:8.4f}", n
    w = rmse3(ev, scen["wind"], sl)
    p = rmse3(ev, scen["payload"], sl)
    return f"{n:8.4f} {w:8.4f} {p:8.4f}", n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mode", required=True,
                    choices=["kf_q", "ukf_alpha", "fme_n"])
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--all-scenarios", action="store_true")
    ap.add_argument("--q-grid", default="1e-3,1.5e-3,2e-3,3e-3,4e-3,6e-3")
    ap.add_argument("--n-grid", default="6,8,9,10,11,12,14")
    ap.add_argument("--alpha-grid", default="0.5,0.8,1.0")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    cfg = load_cfg(a.data_dir)
    scen = scenario_index(cfg)
    ds, M = make_dataset(cfg, a.device)
    run = make_runner(cfg, ds, a.device, a.seed)
    sl = eval_slice(cfg, ds.T)

    hdr = "nominal" if not a.all_scenarios else "nominal     wind  payload"
    print(f"# seed {a.seed}, window {sl.start * cfg.dt:.0f}-{sl.stop * cfg.dt:.0f}s, "
          f"3-D RMSE [m]  (selection criterion: nominal)")

    if a.mode == "kf_q":
        grid = [float(x) for x in a.q_grid.split(",")]
        print(f"\n{'Q':>9} | {'EKF ' + hdr:>26} | {'UKF ' + hdr:>26}")
        best = {"EKF": (1e9, None), "UKF": (1e9, None)}
        for q in grid:
            _, _, _, ee = run.run(make_ekf(cfg, a.device, M, q))
            _, _, _, eu = run.run(make_ukf(cfg, a.device, M, q))
            se, ne = _row(cfg, sl, scen, ee, a.all_scenarios)
            su, nu = _row(cfg, sl, scen, eu, a.all_scenarios)
            for k, v, s in (("EKF", ne, q), ("UKF", nu, q)):
                if v < best[k][0]:
                    best[k] = (v, s)
            print(f"{q:>9.1e} | {se:>26} | {su:>26}")
        print(f"\n-> nominal optimum:  EKF Q={best['EKF'][1]:.1e} ({best['EKF'][0]:.4f})"
              f"   UKF Q={best['UKF'][1]:.1e} ({best['UKF'][0]:.4f})")

    elif a.mode == "ukf_alpha":
        grid = [float(x) for x in a.alpha_grid.split(",")]
        n = 12
        print(f"\n{'alpha':>6} {'n+lam':>7} {'W0':>7} | {'UKF ' + hdr:>26} | corr(EKF)")
        _, _, _, ee = run.run(make_ekf(cfg, a.device, M))
        ce = err_norm(ee)[:, sl].ravel()
        for al in grid:
            lam = al * al * n - n
            w0 = lam / (n + lam)
            _, _, _, eu = run.run(make_ukf(cfg, a.device, M, alpha=al))
            su, _ = _row(cfg, sl, scen, eu, a.all_scenarios)
            cu = err_norm(eu)[:, sl].ravel()
            print(f"{al:>6.2f} {n + lam:>7.2f} {w0:>7.2f} | {su:>26} | "
                  f"{np.corrcoef(ce, cu)[0, 1]:.4f}")
        print("\n-> a correlation near 1.0 means the two Kalman variants are "
              "effectively the same filter here: the UWB range is close to\n"
              "   linear over the operating uncertainty and 6 of the 10 "
              "measurement channels enter linearly.")

    else:  # fme_n
        grid = [int(x) for x in a.n_grid.split(",")]
        T = ds.T
        mask = window_mask(cfg, WIND_WINDOWS, T)
        _, _, _, ee = run.run(make_ekf(cfg, a.device, M))
        _, _, _, eu = run.run(make_ukf(cfg, a.device, M))
        kf_env = np.maximum(
            moving_rms(np.sqrt((err_norm(ee)[scen["wind"]] ** 2).mean(0)), cfg, 1.5),
            moving_rms(np.sqrt((err_norm(eu)[scen["wind"]] ** 2).mean(0)), cfg, 1.5))
        print(f"\n{'N':>4} | {hdr:>26} | {'wind peak':>9} {'over KF':>8}")
        for N in grid:
            _, _, _, ev = run.run(make_fme(cfg, a.device, M, N))
            s, _ = _row(cfg, sl, scen, ev, a.all_scenarios)
            c = moving_rms(np.sqrt((err_norm(ev)[scen["wind"]] ** 2).mean(0)), cfg, 1.5)
            print(f"{N:>4} | {s:>26} | {c[mask].max():9.3f} "
                  f"{(c[mask] - kf_env[mask]).max() * 1000:+7.0f}mm")
        print("\n-> 'over KF' is how far the fixed-horizon FME rises above the "
              "better Kalman filter inside the gust windows (1.5 s moving RMS).")


if __name__ == "__main__":
    main()
