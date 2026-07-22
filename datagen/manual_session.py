#!/usr/bin/env python3
"""
Pilot-in-the-loop recording session (manual flight, human on the sticks).

Replaces commander.py for MANUAL flights: the human flies the vehicle through
QGroundControl (USB gamepad or a real RC transmitter presented as a USB
joystick -- see datagen/MANUAL_FLIGHT.md), while this node only

  1. publishes the scenario JSON        (/datagen/scenario)
  2. detects "airborne" from /gt/odometry (or waits for Enter)
  3. sends "start_log", counts SIMULATION time by counting gt messages
     (one message per 0.02 s of sim time -- RTF-invariant, same trick as
     commander.py), prints a live status line with the disturbance windows
  4. sends "stop_save" after --duration seconds -> traj_XXXX.npz identical
     in format to the scripted dataset
  5. offers to record another trajectory without restarting Isaac.

The engine (run_datagen.py) is completely unchanged: it applies wind/mass at
the scenario windows keyed to time-since-start_log, and logs u from the PX4
setpoints regardless of who pilots.

Examples (engine already running, see datagen/launch_manual.sh)
---------------------------------------------------------------
  # nominal manual flight, auto-start when airborne
  python3 datagen/manual_session.py --mode nominal

  # manual protocol: 12 m/s wind in the 6-11 s window (15 s flight)
  python3 datagen/manual_session.py --mode wind

  # payload pickup: +70 % in the 6-11 s window
  python3 datagen/manual_session.py --mode mass

  # start logging on Enter instead of the altitude trigger
  python3 datagen/manual_session.py --mode wind --start enter
"""
import argparse
import json
import math
import sys
import threading
import queue

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)
from std_msgs.msg import String
from nav_msgs.msg import Odometry

GT_DT = 0.02          # one /gt/odometry message per 0.02 s of SIM time


def parse_windows(spec):
    """'6:10,26:10' -> [(6.0, 10.0), (26.0, 10.0)]  (start_s, duration_s)"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        out.append((float(a), float(b)))
    return out


def build_scenario(a, rng):
    """Scenario dict with EXACTLY the schema of the scripted held-out rows
    (see data_isaac_v12/heldout/meta_0004.json / meta_0007.json)."""
    sc = {"type": "nominal", "pattern": "manual", "duration_s": a.duration,
          "seed": int(rng.integers(0, 2**31 - 1)),
          "mass": None, "gusts": [], "sustained": None, "dropouts": [],
          "nlos_burst": [], "turbulence": [], "cm_regime": [],
          "heldout": True, "ambient_turb_std": a.ambient_turb,
          "manual": True,
          "traj_id": a.traj_id, "split": a.split}
    if a.mode == "wind":
        sc["type"] = "sustained_wind"
        dir_rad = (float(rng.uniform(0, 2 * math.pi))
                   if (a.wind_dir is None or a.wind_dir < 0)
                   else math.radians(a.wind_dir))
        sc["sustained"] = [
            {"speed": a.wind_speed, "vert_ratio": a.wind_vert,
             "dir_rad": dir_rad, "start_s": s, "duration_s": d}
            for (s, d) in parse_windows(a.windows)]
    elif a.mode == "mass":
        sc["type"] = "mass_step"
        sc["mass"] = {
            "delta": a.mass_delta,
            "duration_s": a.mass_duration,
            "onset_s": a.mass_onset,
            "com_offset": a.com_offset,
            "com_dir": float(rng.uniform(0, 2 * math.pi)),
            "impulse_z": float(rng.uniform(0.6, 1.2)),
            "impulse_xy": float(rng.uniform(0.3, 0.7)),
            "impulse_dir": float(rng.uniform(0, 2 * math.pi)),
        }
    return sc


def fmt_windows(sc):
    if sc["type"] == "sustained_wind":
        return " + ".join(f"wind {w['speed']:g} m/s @ {w['start_s']:g}-"
                          f"{w['start_s'] + w['duration_s']:g}s"
                          for w in sc["sustained"])
    if sc["type"] == "mass_step":
        m = sc["mass"]
        return (f"mass +{m['delta'] * 100:.0f}% @ {m['onset_s']:g}-"
                f"{m['onset_s'] + m['duration_s']:g}s")
    return "no disturbance (ambient turbulence only)"


class ManualSession(Node):
    def __init__(self, a):
        super().__init__("manual_session")
        self.a = a
        self.rng = np.random.default_rng(a.seed)
        self.pub_scn = self.create_publisher(String, "/datagen/scenario", 10)
        self.pub_ctl = self.create_publisher(String, "/datagen/control", 10)
        qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE,
                            history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Odometry, "/gt/odometry", self._cb_gt, qos_be)

        self.z = 0.0
        self.gt_n = 0                 # message count -> sim time (RTF-safe)
        self.gt_seen = False

        # non-blocking keyboard: a reader thread feeds a queue
        self.keys = queue.Queue()
        threading.Thread(target=self._stdin_reader, daemon=True).start()

        # session state machine
        self.state = "announce"       # announce -> wait -> logging -> saved
        self.t_air = None             # sim time when altitude trigger armed
        self.t_log0 = None            # sim time at start_log
        self.last_line = -1
        self.scenario = None
        self.timer = self.create_timer(0.05, self._tick)

    # ------------------------------------------------------------ callbacks
    def _cb_gt(self, msg):
        self.z = float(msg.pose.pose.position.z)
        self.gt_n += 1
        self.gt_seen = True

    def _stdin_reader(self):
        for line in sys.stdin:
            self.keys.put(line.strip().lower())

    def _key(self):
        try:
            return self.keys.get_nowait()
        except queue.Empty:
            return None

    @property
    def sim_t(self):
        return self.gt_n * GT_DT

    def _ctl(self, cmd):
        m = String(); m.data = cmd
        self.pub_ctl.publish(m)

    # ------------------------------------------------------------ main loop
    def _tick(self):
        a = self.a
        if self.state == "announce":
            if not self.gt_seen:
                return                            # engine still booting
            self.scenario = build_scenario(a, self.rng)
            m = String(); m.data = json.dumps(self.scenario)
            self.pub_scn.publish(m)
            print(f"\n=== trajectory #{a.traj_id} ({a.mode}) ===")
            print(f"    {fmt_windows(self.scenario)}")
            print(f"    duration {a.duration:.0f}s, split '{a.split}'")
            if a.start == "enter":
                print(">>> take off in QGC, then press ENTER to start logging")
            else:
                print(f">>> take off; logging auto-starts once z > "
                      f"{a.alt_trigger:.1f} m for {a.hold:.0f} s "
                      f"(or press ENTER to force)")
            self.state = "wait"
            self.t_air = None

        elif self.state == "wait":
            forced = self._key() is not None
            if a.start != "enter":
                if self.z > a.alt_trigger:
                    if self.t_air is None:
                        self.t_air = self.sim_t
                    armed = (self.sim_t - self.t_air) >= a.hold
                else:
                    self.t_air = None
                    armed = False
            else:
                armed = False
            if forced or armed:
                self._ctl("start_log")
                self.t_log0 = self.sim_t
                self.last_line = -1
                print(f"[log] START (z = {self.z:.2f} m) -- fly!")
                self.state = "logging"

        elif self.state == "logging":
            t = self.sim_t - self.t_log0
            sec = int(t)
            if sec != self.last_line:
                self.last_line = sec
                tag = ""
                sc = self.scenario
                if sc["type"] == "sustained_wind":
                    for w in sc["sustained"]:
                        if w["start_s"] <= t <= w["start_s"] + w["duration_s"]:
                            tag = f"  << WIND {w['speed']:g} m/s ACTIVE >>"
                elif sc["type"] == "mass_step":
                    mm = sc["mass"]
                    if mm["onset_s"] <= t <= mm["onset_s"] + mm["duration_s"]:
                        tag = f"  << PAYLOAD +{mm['delta']*100:.0f}% ATTACHED >>"
                print(f"[log] t = {sec:3d}/{a.duration:.0f}s  z = {self.z:5.2f} m{tag}")
            if t >= a.duration:
                self._ctl("stop_save")
                print(f"[log] STOP -> {a.split}/traj_{a.traj_id:04d}.npz")
                self.state = "ask"
                print(">>> record another? [y]es same mode / [n]o quit "
                      "(keep flying or land, your choice)")

        elif self.state == "ask":
            k = self._key()
            if k in ("y", "yes", ""):
                a.traj_id += 1
                self.state = "announce"
            elif k in ("n", "no", "q", "quit"):
                print("[session] done. (engine keeps running; Ctrl-C the "
                      "launcher or send 'shutdown' to stop it)")
                if a.shutdown_engine:
                    self._ctl("shutdown")
                raise SystemExit(0)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["nominal", "wind", "mass"],
                    default="nominal")
    ap.add_argument("--traj_id", type=int, default=0,
                    help="first trajectory id (files are traj_XXXX.npz)")
    ap.add_argument("--split", default="heldout",
                    help="subfolder of the engine's --out dir "
                         "(default 'heldout' so TrajDataset/tools load it)")
    ap.add_argument("--duration", type=float, default=None,
                    help="default: 8 s for wind (straight traverse), "
                         "15 s otherwise (square lap)")
    ap.add_argument("--ambient-turb", type=float, default=0.3,
                    help="ambient OU airflow std [m/s]. Manual default 0.3 "
                         "(scripted runs keep 0.5): the constant buffeting at "
                         "0.5 made hand-flying needlessly hard, and a quieter "
                         "baseline RAISES the disturbance-window contrast the "
                         "policy keys on. Note the deviation in the paper.")
    # start trigger
    ap.add_argument("--start", choices=["auto", "enter"], default="auto")
    ap.add_argument("--alt-trigger", type=float, default=0.5,
                    help="altitude [m] that arms auto-start")
    ap.add_argument("--hold", type=float, default=2.0,
                    help="seconds above the trigger before logging starts")
    # wind protocol (defaults = the paper's held-out row)
    ap.add_argument("--wind-speed", type=float, default=11.5,
                    help="manual protocol: 10 m/s, low end of the v13 training "
                         "range [9,15], pilotable; scripted heldout keeps 12")
    ap.add_argument("--wind-dir", type=float, default=180.0,
                    help="deg (default 180: pushes -x, i.e. INWARD/tailwind on "
                         "laps started at the white puck); pass -1 for random")
    ap.add_argument("--wind-vert", type=float, default=0.25)
    ap.add_argument("--windows", default=None,
                    help="'start:dur,start:dur' in seconds after start_log")
    # mass protocol (defaults = the paper's held-out row)
    ap.add_argument("--mass-delta", type=float, default=0.70)
    ap.add_argument("--mass-onset", type=float, default=6.0)
    ap.add_argument("--mass-duration", type=float, default=3.0)
    ap.add_argument("--com-offset", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shutdown-engine", action="store_true",
                    help="send 'shutdown' to the engine when quitting")
    a = ap.parse_args()
    # mode-dependent protocol defaults (v13):
    #   wind -> 8 s single straight traverse ALONG the wind axis, gust 3.5-6.5 s
    #   mass/nominal -> 15 s square lap, window 6-11 s
    # Different durations CANNOT share a dataset folder (the loader stacks
    # trajectories into one tensor), so record wind and square runs into
    # SEPARATE --out directories.
    if a.duration is None:
        a.duration = 8.0 if a.mode == "wind" else 15.0
    if a.windows is None:
        a.windows = "4:2" if a.mode == "wind" else "6:3"

    rclpy.init()
    node = ManualSession(a)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
