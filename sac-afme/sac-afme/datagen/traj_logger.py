"""
datagen/traj_logger.py — 50 Hz trajectory logger for the Isaac Sim datagen.
Collects rows inside run_datagen.py (same process as the sim) and writes the
EXACT schema produced by rlenv/synth.py, so rlenv/dataset.py loads either
source unchanged:
    traj_XXXX.npz : t [T], u [T,4], gt [T,12], m_true [T], wind [T,3]
    meta_XXXX.json: {scenario, consts}
Frames: gt is world-ENU / body-FLU, eta = ZYX yaw-pitch-roll [roll,pitch,yaw],
matching filter/uav_model.py.
"""
import json
import os
import numpy as np


class TrajLogger:
    def __init__(self, out_root):
        self.out_root = out_root
        self.active = False
        self.rows = None
        self.meta = None

    def start(self, scenario: dict, consts: dict):
        self.rows = {k: [] for k in ("t", "u", "gt", "m_true", "wind")}
        self.meta = {"scenario": scenario, "consts": consts}
        self.active = True

    def log(self, t, u_phys, gt12, m_true, wind_vel):
        if not self.active:
            return
        self.rows["t"].append(float(t))
        self.rows["u"].append(np.asarray(u_phys, dtype=np.float32))
        self.rows["gt"].append(np.asarray(gt12, dtype=np.float32))
        self.rows["m_true"].append(float(m_true))
        self.rows["wind"].append(np.asarray(wind_vel, dtype=np.float32))

    def stop_save(self, split: str, traj_id: int):
        """write npz + meta; returns path or None if nothing logged."""
        if not self.active or len(self.rows["t"]) == 0:
            self.active = False
            return None
        d = os.path.join(self.out_root, split)
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"traj_{traj_id:04d}.npz")
        np.savez_compressed(
            path,
            t=np.asarray(self.rows["t"], dtype=np.float32),
            u=np.stack(self.rows["u"]),
            gt=np.stack(self.rows["gt"]),
            m_true=np.asarray(self.rows["m_true"], dtype=np.float32),
            wind=np.stack(self.rows["wind"]))
        with open(os.path.join(d, f"meta_{traj_id:04d}.json"), "w") as f:
            json.dump(self.meta, f, indent=1)
        self.active = False
        n = len(self.rows["t"])
        print(f"[logger] saved {path} ({n} rows = {n * 0.02:.1f}s)", flush=True)
        return path
