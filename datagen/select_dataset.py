#!/usr/bin/env python3
"""
datagen/select_dataset.py — offline pool curation for RL training
==================================================================
Workflow (fully DECOUPLED from RL training, as designed):
  1) Run Isaac Sim datagen at any earlier time into a RAW POOL:
       ISAACSIM_PYTHON datagen/run_datagen.py --headless 1 --out data_raw
       python datagen/commander.py --n_train 400 --n_heldout 100
  2) Curate a subset for training (this tool):
       python -m datagen.select_dataset --pool data_raw --out data \
              --n_train 200 --n_heldout 50
  3) Train / evaluate as usual (train.py reads only npz files).

Per-trajectory quality gates (pure numpy — no torch):
  G1 integrity   : finite arrays, length >= 95% of scenario duration,
                   airborne thrust (mean u1 > 0.3 * m*g after takeoff)
  G2 workspace   : positions inside the anchor workspace + margin
                   (also catches ENU/NED frame mistakes immediately)
  G3 consistency : nominal-segment 1-step model propagation error
                   med ||f(gt_k, u_k) - gt_{k+1}|| < thresh (default 1 cm)
                   -> validates u calibration & frame conventions per file
  G4 signal      : for disturbance scenarios, disturbed/nominal propagation
                   error ratio >= dist_ratio_min (the injected disturbance
                   actually bit — otherwise it is a mislabeled nominal)

Selection: within each split, group by scenario type, allocate quotas
proportional to cfg.scenario_probs over the types available, then fill each
quota with the LOWEST nominal-consistency error first (cleanest logs).
Shortfalls are topped up across types by score. Selected files are copied and
RENUMBERED (traj_0000.. contiguous) so TrajDataset loads them unchanged.
A full selection_report.csv (every pool file, all metrics, pass/fail reasons,
selected split) is written next to the output.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config                                 # noqa: E402
from datagen.scenario import disturbance_intervals        # noqa: E402


# ── nominal-model 1-step propagation (numpy mirror of filter/uav_model.f,
#    nominal mass, NO wind — this is exactly what the filter believes) ──
def _f_nom(s, u, dt, cfg):
    p, v, eta, om = s[0:3], s[3:6], s[6:9], s[9:12]
    ph, th, ps = np.clip(eta[0], -1.2, 1.2), np.clip(eta[1], -1.2, 1.2), eta[2]
    cp, sp = np.cos(ph), np.sin(ph)
    ct, st = np.cos(th), np.sin(th)
    cy, sy = np.cos(ps), np.sin(ps)
    R = np.array([[cy * ct, cy * st * sp - sy * cp, cy * st * cp + sy * sp],
                  [sy * ct, sy * st * sp + cy * cp, sy * st * cp - cy * sp],
                  [-st, ct * sp, ct * cp]])
    acc = np.array([0, 0, -cfg.g]) + R @ np.array([0, 0, u[0]]) / cfg.mass_nominal
    tt, sec = np.tan(th), 1.0 / max(np.cos(th), 0.35)
    W = np.array([[1, sp * tt, cp * tt], [0, cp, -sp], [0, sp * sec, cp * sec]])
    J = np.array([cfg.Ixx, cfg.Iyy, cfg.Izz])
    om_dot = (u[1:4] - np.cross(om, J * om)) / J
    return np.concatenate([p + v * dt, v + acc * dt,
                           eta + (W @ om) * dt, om + om_dot * dt])


def _prop_errors(gt, u, dt, cfg):
    """||f_nom(gt_k,u_k) - gt_{k+1}|| for all k -> [T-1]"""
    T = gt.shape[0]
    e = np.zeros(T - 1)
    for k in range(T - 1):
        e[k] = np.linalg.norm(_f_nom(gt[k], u[k], dt, cfg) - gt[k + 1])
    return e


def analyze(npz_path, cfg, args):
    """returns dict of metrics + pass/fail reasons for one trajectory."""
    r = {"file": os.path.basename(npz_path), "reasons": []}
    meta_path = npz_path.replace("traj_", "meta_").replace(".npz", ".json")
    try:
        z = np.load(npz_path)
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        sc = meta.get("scenario", {})
    except Exception as e:
        r["reasons"].append(f"load_fail:{e}")
        r.update(type="?", pattern="?", ok=False)
        return r
    gt, u, t = z["gt"], z["u"], z["t"]
    r["type"] = sc.get("type", "?")
    r["pattern"] = sc.get("pattern", "?")
    r["T"] = int(gt.shape[0])
    dt = float(meta.get("consts", {}).get("dt", cfg.dt))

    # G1 integrity
    if not (np.isfinite(gt).all() and np.isfinite(u).all()):
        r["reasons"].append("nonfinite")
    dur = float(sc.get("duration_s", cfg.traj_duration_s))
    if gt.shape[0] < 0.95 * dur / dt:
        r["reasons"].append(f"short:{gt.shape[0]}")
    k0 = int(args.skip_s / dt)                       # skip takeoff transient
    hover = cfg.mass_nominal * cfg.g
    if u[k0:, 0].mean() < 0.3 * hover:
        r["reasons"].append("not_airborne")

    # G2 workspace (frame check)
    p = gt[k0:, 0:3]
    lo, hi, m = 0.0, 10.0, args.ws_margin
    if p.size and ((p[:, 0] < lo - m).any() or (p[:, 0] > hi + m).any() or
                   (p[:, 1] < lo - m).any() or (p[:, 1] > hi + m).any() or
                   (p[:, 2] < 0.1).any() or (p[:, 2] > 6.0).any()):
        r["reasons"].append("workspace")

    # G3/G4 consistency & disturbance signal (subsample for speed)
    stride = max(1, gt.shape[0] // args.check_pts)
    idx = np.arange(k0, gt.shape[0] - 1, stride)
    e = np.array([np.linalg.norm(_f_nom(gt[k], u[k], dt, cfg) - gt[k + 1])
                  for k in idx])
    tt = t[idx]
    dist_mask = np.zeros(len(idx), bool)
    for (a, b, _) in disturbance_intervals(sc):
        dist_mask |= (tt >= a) & (tt <= b)
    nom = e[~dist_mask]
    dis = e[dist_mask]
    r["nom_err"] = float(np.median(nom)) if nom.size else float("nan")
    r["dist_err"] = float(np.median(dis)) if dis.size else float("nan")
    r["ratio"] = (r["dist_err"] / r["nom_err"]
                  if nom.size and dis.size and r["nom_err"] > 0 else float("nan"))
    if nom.size and r["nom_err"] > args.prop_err_thresh:
        r["reasons"].append(f"consistency:{r['nom_err']:.4f}")
    if r["type"] in ("mass_step", "gust", "sustained_wind", "mixed"):
        if dis.size and np.isfinite(r["ratio"]) and r["ratio"] < args.dist_ratio_min:
            r["reasons"].append(f"weak_signal:{r['ratio']:.2f}")

    r["ok"] = len(r["reasons"]) == 0
    return r


def select_split(rows, n_want, cfg, rng):
    """scenario-balanced pick: quotas ~ cfg.scenario_probs over available
    types; within type sort by nominal consistency (ascending)."""
    ok = [r for r in rows if r["ok"]]
    by_type = {}
    for r in ok:
        by_type.setdefault(r["type"], []).append(r)
    for v in by_type.values():
        v.sort(key=lambda r: (r["nom_err"] if np.isfinite(r["nom_err"]) else 9e9))
    types = [s for s in cfg.scenario_types if s in by_type]
    probs = np.array([cfg.scenario_probs[cfg.scenario_types.index(s)]
                      for s in types], dtype=float)
    probs = probs / probs.sum() if probs.sum() > 0 else probs
    quota = {s: int(round(n_want * p)) for s, p in zip(types, probs)}
    picked = []
    for s in types:
        picked += by_type[s][:quota[s]]
    # top-up shortfall across all remaining, cleanest first
    rest = sorted([r for r in ok if r not in picked],
                  key=lambda r: (r["nom_err"] if np.isfinite(r["nom_err"]) else 9e9))
    picked += rest[:max(0, n_want - len(picked))]
    return picked[:n_want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data_raw", help="raw pool root (train/, heldout/)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_heldout", type=int, default=50)
    ap.add_argument("--prop_err_thresh", type=float, default=0.01,
                    help="G3: nominal 1-step propagation median [m]")
    ap.add_argument("--dist_ratio_min", type=float, default=3.0,
                    help="G4: disturbed/nominal propagation-error ratio")
    ap.add_argument("--ws_margin", type=float, default=1.5)
    ap.add_argument("--skip_s", type=float, default=2.0)
    ap.add_argument("--check_pts", type=int, default=400,
                    help="subsampled consistency points per trajectory")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = Config()
    rng = np.random.default_rng(args.seed)

    report = []
    for split, n_want in (("train", args.n_train), ("heldout", args.n_heldout)):
        pool_dir = os.path.join(args.pool, split)
        files = sorted(glob.glob(os.path.join(pool_dir, "traj_*.npz")))
        if not files:
            print(f"[select] WARNING: no files in {pool_dir} — skipping split")
            continue
        rows = [analyze(f, cfg, args) for f in files]
        for r, f in zip(rows, files):
            r["path"] = f
            r["split_pool"] = split
        picked = select_split(rows, n_want, cfg, rng)
        sel_names = {r["file"] for r in picked}
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        for old in glob.glob(os.path.join(out_dir, "traj_*.npz")) + \
                glob.glob(os.path.join(out_dir, "meta_*.json")):
            os.remove(old)
        for i, r in enumerate(picked):
            shutil.copy(r["path"], os.path.join(out_dir, f"traj_{i:04d}.npz"))
            mp = r["path"].replace("traj_", "meta_").replace(".npz", ".json")
            if os.path.exists(mp):
                shutil.copy(mp, os.path.join(out_dir, f"meta_{i:04d}.json"))
        for r in rows:
            r["selected"] = split if r["file"] in sel_names else ""
        report += rows
        n_ok = sum(r["ok"] for r in rows)
        types = {}
        for r in picked:
            types[r["type"]] = types.get(r["type"], 0) + 1
        print(f"[select:{split}] pool {len(rows)} | passed {n_ok} | "
              f"picked {len(picked)}/{n_want} | mix {types}")

    rp = os.path.join(args.out, "selection_report.csv")
    os.makedirs(args.out, exist_ok=True)
    with open(rp, "w") as f:
        f.write("file,split_pool,selected,type,pattern,T,nom_err,dist_err,"
                "ratio,ok,reasons\n")
        for r in report:
            f.write(f"{r['file']},{r.get('split_pool','')},{r.get('selected','')},"
                    f"{r.get('type','')},{r.get('pattern','')},{r.get('T',0)},"
                    f"{r.get('nom_err',float('nan')):.5f},"
                    f"{r.get('dist_err',float('nan')):.5f},"
                    f"{r.get('ratio',float('nan')):.2f},{r['ok']},"
                    f"\"{';'.join(r['reasons'])}\"\n")
    print(f"[select] report → {rp}")


if __name__ == "__main__":
    main()
