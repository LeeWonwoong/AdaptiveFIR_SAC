#!/usr/bin/env python3
"""
Emit the LaTeX RMSE table for one seed (same numbers as tools/eval_seeds.py).

  python3 tools/make_table.py --data_dir data_isaac_v12 \
      --ckpt results/v12_50k/ckpt.pt --seed 13 --out table_rmse.tex

  # add the per-axis columns
  python3 tools/make_table.py ... --per-axis
"""
import argparse
import numpy as np

from _common import (DISPLAY, load_cfg, scenario_index, make_dataset,
                     load_agent, run_all_seeded, default_method_seeds,
                     SEED, eval_slice, rmse3, METHODS,
                     EVAL_T0, EVAL_T1, Q_EKF, Q_UKF, Q_EKF_DIST, Q_UKF_DIST,
                     R_EKF, R_UKF, R_EKF_DIST, R_UKF_DIST, FME_N)

ROWS = [("nominal", "Nominal"), ("wind", "Wind"), ("payload", "Payload")]


def bold_min(vals, fmt="{:.3f}"):
    lo = min(vals.values())
    return [(r"\textbf{" + fmt.format(vals[m]) + "}") if vals[m] == lo
            else fmt.format(vals[m]) for m in METHODS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    # per-method noise-seed overrides (same convention as make_figs.py);
    # a method without an override uses the SEED_* constant from
    # tools/_common.py, then --seed.
    ap.add_argument("--seed-ekf", type=int, default=None)
    ap.add_argument("--seed-ukf", type=int, default=None)
    ap.add_argument("--seed-fme", type=int, default=None)
    ap.add_argument("--seed-afme", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-axis", action="store_true")
    ap.add_argument("--q-ekf", "--q-ekf-nom", dest="q_ekf",
                    type=float, default=Q_EKF)
    ap.add_argument("--q-ukf", "--q-ukf-nom", dest="q_ukf",
                    type=float, default=Q_UKF)
    ap.add_argument("--q-ekf-dist", type=float, default=Q_EKF_DIST)
    ap.add_argument("--q-ukf-dist", type=float, default=Q_UKF_DIST)
    ap.add_argument("--r-ekf", "--r-ekf-nom", dest="r_ekf",
                    type=float, default=R_EKF)
    ap.add_argument("--r-ukf", "--r-ukf-nom", dest="r_ukf",
                    type=float, default=R_UKF)
    ap.add_argument("--r-ekf-dist", type=float, default=R_EKF_DIST)
    ap.add_argument("--r-ukf-dist", type=float, default=R_UKF_DIST)
    ap.add_argument("--fme-n", type=int, default=FME_N)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    # match eval_seeds: wind/payload scored with the disturbance-regime (Q,R)
    split_q = (a.q_ekf_dist != a.q_ekf) or (a.q_ukf_dist != a.q_ukf) \
        or (a.r_ekf_dist != a.r_ekf) or (a.r_ukf_dist != a.r_ukf)

    cfg = load_cfg(a.data_dir)
    scen = scenario_index(cfg)
    ds, M = make_dataset(cfg, a.device)
    agent = load_agent(cfg, a.ckpt, a.device)
    # seed precedence: --seed-<m> CLI flag > SEED_<M> in _common.py > --seed
    method_seeds = default_method_seeds()
    method_seeds.update({m: s for m, s in
                         [("EKF", a.seed_ekf), ("UKF", a.seed_ukf),
                          ("FME", a.seed_fme), ("AFME", a.seed_afme)]
                         if s is not None})
    res = run_all_seeded(cfg, ds, a.device, M, agent,
                         seed=a.seed, method_seeds=method_seeds,
                         q_ekf=a.q_ekf, q_ukf=a.q_ukf,
                         r_ekf=a.r_ekf, r_ukf=a.r_ukf, fme_N=a.fme_n,
                         q_ekf_dist=a.q_ekf_dist if split_q else None,
                         q_ukf_dist=a.q_ukf_dist if split_q else None,
                         r_ekf_dist=a.r_ekf_dist if split_q else None,
                         r_ukf_dist=a.r_ukf_dist if split_q else None)
    sl = eval_slice(cfg, ds.T)

    def evec_for(m, scenario):
        """EKF/UKF use the disturbance-regime (Q,R) rollout for wind/payload."""
        d = res[m]
        if scenario != "nominal" and "evec_dist" in d:
            return d["evec_dist"]
        return d["evec"]

    body = []
    per_scen = {}
    for key, disp in ROWS:
        vals = {m: rmse3(evec_for(m, key), scen[key], sl) for m in METHODS}
        per_scen[key] = vals
        body.append(f"{disp} & " + " & ".join(bold_min(vals)) + r" \\")

    avg = {m: float(np.mean([per_scen[k][m] for k, _ in ROWS])) for m in METHODS}

    if split_q:
        kf_desc = (f"the Kalman filters use $Q={a.q_ekf:g}I_{{12}}$ (EKF) / "
                   f"$Q={a.q_ukf:g}I_{{12}}$ (UKF) in nominal flight and "
                   f"$Q={a.q_ekf_dist:g}$ / ${a.q_ukf_dist:g}$ under disturbance, "
                   f"with measurement noise scaled by "
                   f"{a.r_ekf:g}/{a.r_ekf_dist:g} (EKF) and "
                   f"{a.r_ukf:g}/{a.r_ukf_dist:g} (UKF) "
                   r"(nominal/disturbance)")
    else:
        kf_desc = (f"both Kalman filters use $Q={a.q_ekf:g}I_{{12}}$, "
                   r"each selected by a grid search minimising nominal-flight RMSE")
    cap = (r"Position RMSE over the "
           f"{EVAL_T0:g}--{EVAL_T1:g}"
           r"\,s evaluation window (m), averaged over the three flight "
           r"patterns of each scenario. The FME baseline uses the fixed "
           f"horizon $N{{=}}{a.fme_n}$; {kf_desc}. "
           r"Bold: best.")

    tex = "\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + cap + "}",
        r"\label{tab:rmse}",
        r"\begin{tabular}{lcccc}", r"\hline",
        r"Scenario & " + " & ".join(DISPLAY[m] for m in METHODS)
        + r" \\", r"\hline",
        *body, r"\hline",
        r"Average & " + " & ".join(bold_min(avg)) + r" \\",
        r"\hline", r"\end{tabular}", r"\end{table}"])

    print(tex)
    if method_seeds:
        print("% seeds: " + " ".join(
            f"{m}={method_seeds.get(m, a.seed)}" for m in METHODS))

    if a.per_axis:
        print("\n% per-axis breakdown (x / y / z)")
        for key, disp in ROWS:
            line = [disp]
            for m in METHODS:
                e = np.asarray(evec_for(m, key))[scen[key]][:, sl, :]
                ax_ = [float(np.sqrt((e[:, :, j] ** 2).mean())) for j in range(3)]
                line.append(" / ".join(f"{v:.3f}" for v in ax_))
            print("%  " + " | ".join(line))

    if a.out:
        with open(a.out, "w") as f:
            f.write(tex + "\n")
        print(f"\n% wrote {a.out}")


if __name__ == "__main__":
    main()
