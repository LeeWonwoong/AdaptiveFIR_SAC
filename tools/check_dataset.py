#!/usr/bin/env python3
"""
Dataset QC: verify every generated trajectory stays inside the anchor hull,
below the anchor plane, and within the intended speed band.

  python3 tools/check_dataset.py --data_dir data_isaac_v13
  python3 tools/check_dataset.py --data_dir data_isaac_v13 --split heldout -v

Exit code 1 if any trajectory fails, so it can gate a pipeline.
"""
import argparse
import glob
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config                                        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", default="both", choices=["train", "heldout", "both"])
    ap.add_argument("--margin", type=float, default=0.0,
                    help="required clearance inside the anchor square [m]")
    ap.add_argument("--v-max", dest="v_max", type=float, default=2.5,
                    help="horizontal speed gate [m/s]; v15 modulated gusts "
                         "legitimately reach ~3.9 (peak ~18 m/s shove), so "
                         "use 4.0 for v15-profile datasets")
    ap.add_argument("--z-margin", type=float, default=0.3,
                    help="required clearance below the anchor plane [m]")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    cfg = Config()
    anch = np.array([list(x) for x in cfg.anchors], float)
    x0, x1 = anch[:, 0].min() + a.margin, anch[:, 0].max() - a.margin
    y0, y1 = anch[:, 1].min() + a.margin, anch[:, 1].max() - a.margin
    z_top = anch[:, 2].max() - a.z_margin

    print(f"anchor square x {x0:g}-{x1:g}, y {y0:g}-{y1:g} | "
          f"anchor plane {anch[:, 2].max():g} m -> z limit {z_top:g} m")

    splits = ["train", "heldout"] if a.split == "both" else [a.split]
    bad_total = 0
    for sp in splits:
        files = sorted(glob.glob(os.path.join(a.data_dir, sp, "traj_*.npz")))
        if not files:
            print(f"\n[{sp}] no trajectories found")
            continue
        print(f"\n[{sp}] {len(files)} trajectories")
        bad = []
        for f in files:
            g = np.load(f)["gt"]
            v = g[:, 3:6]
            hp = np.linalg.norm(v[:, :2], axis=1)
            issues = []
            if g[:, 0].min() < x0 or g[:, 0].max() > x1:
                issues.append(f"x {g[:, 0].min():.2f}~{g[:, 0].max():.2f}")
            if g[:, 1].min() < y0 or g[:, 1].max() > y1:
                issues.append(f"y {g[:, 1].min():.2f}~{g[:, 1].max():.2f}")
            if g[:, 2].max() > z_top:
                issues.append(f"z max {g[:, 2].max():.2f}")
            if hp.max() > a.v_max:
                issues.append(f"v max {hp.max():.2f}")
            name = os.path.basename(f)
            if issues:
                bad.append((name, issues))
                mf = f.replace("traj_", "meta_").replace(".npz", ".json")
                pat = "?"
                if os.path.exists(mf):
                    pat = json.load(open(mf))["scenario"].get("pattern", "?")
                print(f"  ✗ {name} ({pat}): " + ", ".join(issues))
            elif a.verbose:
                print(f"  ✓ {name}: x {g[:, 0].min():.2f}~{g[:, 0].max():.2f} "
                      f"y {g[:, 1].min():.2f}~{g[:, 1].max():.2f} "
                      f"z {g[:, 2].min():.2f}~{g[:, 2].max():.2f} "
                      f"v {np.median(hp):.2f}")
        bad_total += len(bad)
        print(f"  → {len(files) - len(bad)}/{len(files)} pass"
              + (f", {len(bad)} FAIL" if bad else "  ✓ all clear"))

    sys.exit(1 if bad_total else 0)


if __name__ == "__main__":
    main()
