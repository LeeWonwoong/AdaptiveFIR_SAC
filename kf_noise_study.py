#!/usr/bin/env python3
"""
kf_noise_study.py — EKF vs UKF under (i) mis-specified noise statistics and
(ii) large-error regimes.

WHY THIS EXISTS
---------------
R and Q are NOT known in practice. A deployed filter is handed a datasheet
sigma and a hand-tuned Q, both of which are wrong to some degree. The honest
question is therefore not "which filter wins when we hand it the truth" but
"which filter degrades more gracefully when we hand it something wrong".

PROTOCOL (this is the part that makes the comparison fair)
---------------------------------------------------------
* Q and R are STATISTICAL ASSUMPTIONS ABOUT THE PLANT AND THE SENSORS, so both
  filters are always given the SAME ones. Handing the two estimators different
  beliefs about the same sensor is not a comparison of estimators.
* alpha/beta/kappa are UKF-INTERNAL sigma-point knobs with no EKF counterpart,
  so they are tuned to the UKF's own best. That gives the UKF its best shot,
  which is what a fair protocol requires.
* Whatever ordering falls out, falls out. Report it.

USAGE
    python3 kf_noise_study.py --data data_isaac_v8/heldout
"""
import argparse
import dataclasses
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from config import Config                       # noqa: E402
from filter.baselines import EKF, UKF           # noqa: E402

TRUE_SIGMA = (0.10, 0.02, 0.01)   # uwb [m], attitude [rad], gyro [rad/s]


def make_measurements(gt, anchors, seeds):
    """Synthesize the sensor stream with the TRUE noise (this never changes —
    only the filter's BELIEF about it does)."""
    T = gt.shape[0]
    out = []
    for sd in seeds:
        r = np.random.default_rng(sd)
        d = (gt[:, None, 0:3] - anchors[None]).norm(dim=2).numpy()
        d = d + TRUE_SIGMA[0] * r.standard_normal((T, 4))
        att = gt[:, 6:9].numpy() + TRUE_SIGMA[1] * r.standard_normal((T, 3))
        gyr = gt[:, 9:12].numpy() + TRUE_SIGMA[2] * r.standard_normal((T, 3))
        out.append(np.concatenate([d, att, gyr], axis=1))
    return torch.tensor(np.stack(out, 1).astype(np.float32))


def run_filter(kind, gt, u, Z, cfg, q_val, r_sigma, alpha=0.5, beta=2.0,
               kappa=0.0, mask=None):
    """r_sigma = the sigma the FILTER BELIEVES (may differ from TRUE_SIGMA)."""
    T, M = gt.shape[0], Z.shape[1]
    c = dataclasses.replace(cfg, n_envs=M)
    f = (EKF(c, "cpu", M) if kind == "EKF"
         else UKF(c, "cpu", M, alpha=alpha, beta=beta, kappa=kappa))

    f.Q = torch.diag(torch.tensor([q_val] * 12)).float()
    rr = torch.tensor([r_sigma[0]] * 4 + [r_sigma[1]] * 3 + [r_sigma[2]] * 3) ** 2
    f.R = torch.diag(rr).float()

    torch.manual_seed(11)
    f.reset(torch.arange(M), gt[0:1].expand(M, -1).clone() + 0.02 * torch.randn(M, 12))

    E = torch.zeros(M, T, 3)
    for t in range(1, T):
        s, _, _ = f.step(u[t - 1:t].expand(M, -1), Z[t] if t % 5 == 0 else None)
        E[:, t] = gt[t, 0:3].unsqueeze(0) - s[:, 0:3]

    if mask is None:
        mask = torch.zeros(T, dtype=torch.bool)
        mask[300:T - 100] = True
    return float(torch.sqrt((E[:, mask] ** 2).sum(-1).mean(1)).mean())


def load(base, tid, anchors, seeds, T=2500):
    z = np.load(f"{base}/traj_{tid:04d}.npz")
    gt = torch.tensor(z["gt"][:T], dtype=torch.float32)
    u = torch.tensor(z["u"][:T], dtype=torch.float32)
    return gt, u, make_measurements(gt, anchors, seeds)


def window_masks(base, tid, T=2500, dt=0.02):
    """Return (inside_window, outside_window) boolean masks."""
    sc = json.load(open(f"{base}/meta_{tid:04d}.json"))["scenario"]
    inside = torch.zeros(T, dtype=torch.bool)
    wins = []
    if sc.get("sustained"):
        s = sc["sustained"]
        wins = s if isinstance(s, list) else [s]
        wins = [(w["start_s"], w["start_s"] + w["duration_s"]) for w in wins]
    elif sc.get("mass"):
        m = sc["mass"]
        wins = [(m["onset_s"], m["onset_s"] + m["duration_s"])]
    for a, b in wins:
        inside[int(a / dt):min(T, int(b / dt))] = True
    outside = ~inside
    outside[:300] = False
    return inside, outside


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_isaac_v8/heldout")
    ap.add_argument("--nominal", type=int, default=0)
    ap.add_argument("--wind", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    cfg = Config()
    anchors = torch.tensor(cfg.anchors)
    seeds = tuple(range(args.seeds))

    gt0, u0, Z0 = load(args.data, args.nominal, anchors, seeds)
    gt2, u2, Z2 = load(args.data, args.wind, anchors, seeds)
    win_in, _ = window_masks(args.data, args.wind)

    # ---- 1. UKF sigma-point knobs: give the UKF its best shot -------------
    best_a, best_v = 0.5, 9.9
    for a in (0.3, 0.5, 0.7, 0.9, 1.0):
        v = run_filter("UKF", gt0, u0, Z0, cfg, 2e-3, TRUE_SIGMA, alpha=a)
        if v < best_v:
            best_a, best_v = a, v
    print(f"UKF sigma-point tuning: best alpha = {best_a}  ({best_v:.4f})\n")

    # ---- 2. R MIS-SPECIFICATION (same wrong R to BOTH filters) ------------
    print("R mis-specification — both filters believe the SAME (wrong) sigma_uwb")
    print(f"{'believed':>9} │ {'EKF':>7} │ {'UKF':>7} │ winner")
    print("─" * 42)
    for r_uwb in (0.05, 0.08, 0.10, 0.15, 0.20, 0.30):
        rs = (r_uwb, TRUE_SIGMA[1], TRUE_SIGMA[2])
        e = run_filter("EKF", gt0, u0, Z0, cfg, 2e-3, rs)
        k = run_filter("UKF", gt0, u0, Z0, cfg, 2e-3, rs, alpha=best_a)
        tag = "TRUE" if abs(r_uwb - TRUE_SIGMA[0]) < 1e-9 else ""
        w = "UKF" if k < e else "EKF"
        print(f"{r_uwb:>9.2f} │ {e:>7.4f} │ {k:>7.4f} │ {w}  {tag}")

    # ---- 3. Q MIS-SPECIFICATION ------------------------------------------
    print("\nQ mis-specification — both filters use the SAME Q")
    print(f"{'Q':>9} │ {'EKF':>7} │ {'UKF':>7} │ winner")
    print("─" * 42)
    for q in (5e-4, 1e-3, 2e-3, 5e-3, 1e-2):
        e = run_filter("EKF", gt0, u0, Z0, cfg, q, TRUE_SIGMA)
        k = run_filter("UKF", gt0, u0, Z0, cfg, q, TRUE_SIGMA, alpha=best_a)
        print(f"{q:>9.0e} │ {e:>7.4f} │ {k:>7.4f} │ {'UKF' if k < e else 'EKF'}")

    # ---- 4. LARGE-ERROR REGIME (inside the disturbance window) -----------
    # The UKF's advantage is curvature: it should only appear once the error is
    # big enough that the EKF's linearization of the RANGE model breaks down.
    print("\nInside the gust window (large errors) — same Q, same R")
    e = run_filter("EKF", gt2, u2, Z2, cfg, 2e-3, TRUE_SIGMA, mask=win_in)
    k = run_filter("UKF", gt2, u2, Z2, cfg, 2e-3, TRUE_SIGMA, alpha=best_a,
                   mask=win_in)
    print(f"  EKF {e:.4f} │ UKF {k:.4f} │ {'UKF' if k < e else 'EKF'} wins")


if __name__ == "__main__":
    main()
