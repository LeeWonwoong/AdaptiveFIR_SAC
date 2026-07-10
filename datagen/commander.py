#!/usr/bin/env python3
"""
datagen/commander.py — trajectory orchestrator for Isaac Sim data generation
==============================================================================
Runs in a NORMAL ROS2 python env (Terminal 2) while run_datagen.py runs inside
Isaac Sim (Terminal 1). Per trajectory it:
  1. samples a scenario (datagen/scenario.py — same sampler as Tier-0 synth)
  2. publishes it to the sim  (/datagen/scenario)
  3. arms PX4 + enters offboard, climbs to the pattern start point
  4. starts logging (/datagen/control "start_log"), flies the pattern for
     duration_s with 50 Hz TrajectorySetpoints (same reference shapes as
     rlenv/synth._ref so Tier-0 and Isaac datasets are directly comparable)
  5. stops+saves ("stop_save"), resets the world ("reset"), repeats.

Offboard mechanics (heartbeat -> DO_SET_MODE(1,6) + ARM, z<0 = up in PX4
local NED) mirror the user's verified Issacsim-rhukf/online_rl_main.py.

[VERIFY-IN-SIM] items (one-time live checks; the dataset validation
checklist in README will catch violations immediately):
  - ENU->NED mapping ned=(y_enu, x_enu, -z_enu), yaw_ned = pi/2 - yaw_enu:
    confirm logged gt positions stay inside the anchor workspace [0,10]^2.
  - PX4 local origin == Isaac world origin (vehicle spawns at (0,0,0.07)).
  - QoS on /fmu/in/* (default depth-10 here; switch to the repo's profile if
    setpoints are not accepted).

Usage:
  python datagen/commander.py --n_train 200 --n_heldout 50 [--px4_ns ""]
"""
import argparse
import json
import os
import sys
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint,
                          VehicleCommand)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config                                # noqa: E402
from datagen.scenario import sample_scenario             # noqa: E402
from rlenv.synth import _ref                             # noqa: E402  (shared patterns)


def enu_to_ned(p_enu, v_enu=None, yaw_enu=0.0):
    p_ned = np.array([p_enu[1], p_enu[0], -p_enu[2]])
    v_ned = None if v_enu is None else np.array([v_enu[1], v_enu[0], -v_enu[2]])
    yaw_ned = float(np.pi / 2.0 - yaw_enu)
    return p_ned, v_ned, yaw_ned


class Commander(Node):
    TICK = 0.02                                   # 50 Hz

    def __init__(self, args):
        super().__init__("datagen_commander")
        self.args = args
        self.cfg = Config()
        self.rng = np.random.default_rng(args.seed)

        qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                            durability=DurabilityPolicy.VOLATILE,
                            history=HistoryPolicy.KEEP_LAST, depth=5)
        ns = args.px4_ns
        self.pub_offboard = self.create_publisher(
            OffboardControlMode, f"{ns}/fmu/in/offboard_control_mode", 10)
        self.pub_traj = self.create_publisher(
            TrajectorySetpoint, f"{ns}/fmu/in/trajectory_setpoint", 10)
        self.pub_cmd = self.create_publisher(
            VehicleCommand, f"{ns}/fmu/in/vehicle_command", 10)
        self.pub_scenario = self.create_publisher(String, "/datagen/scenario", 10)
        self.pub_control = self.create_publisher(String, "/datagen/control", 10)
        self.create_subscription(Odometry, "/gt/odometry", self._cb_gt, qos_be)

        # trajectory queue: (traj_id, split, heldout)
        self.queue = [(i, "train", False) for i in
                      range(args.start_id, args.start_id + args.n_train)]
        self.queue += [(i, "heldout", True) for i in range(args.n_heldout)]
        self.q_idx = 0

        self.gt_pos = np.zeros(3)                 # ENU (from sim GT)
        self.gt_seen = False
        self.state = "INIT"
        self.state_t = 0.0
        self.fly_t = 0.0
        self.scenario = None
        self.retry_count = 0
        self.MAX_RETRY = 3
        # ── live-log state (online_rl_main-style console feedback) ──
        self._airborne = False           # takeoff detected this traj
        self._flag_state = {}            # disturbance-window ON/OFF edges
        self._last_fly_log = -1e9        # throttle for FLY tracking line
        self.arm_resend_t = 0.0

        self.timer = self.create_timer(self.TICK, self._tick)
        self.get_logger().info(
            f"commander: {args.n_train} train + {args.n_heldout} heldout "
            f"trajectories, alt={args.alt} m")

    # ────────────────────────────── ROS helpers
    def _cb_gt(self, msg):
        self.gt_pos[:] = [msg.pose.pose.position.x,
                          msg.pose.pose.position.y,
                          msg.pose.pose.position.z]
        self.gt_seen = True

    def _send_offboard(self):
        m = OffboardControlMode()
        m.position = True
        m.timestamp = 0
        self.pub_offboard.publish(m)

    def _send_setpoint_enu(self, p_enu, v_enu=None, yaw_enu=0.0):
        p, v, yaw = enu_to_ned(p_enu, v_enu, yaw_enu)
        m = TrajectorySetpoint()
        m.position = [float(p[0]), float(p[1]), float(p[2])]
        nan = float("nan")
        m.velocity = ([float(v[0]), float(v[1]), float(v[2])] if v is not None
                      else [nan, nan, nan])
        m.acceleration = [nan, nan, nan]
        m.yaw = yaw
        m.timestamp = 0
        self.pub_traj.publish(m)

    def _vehicle_cmd(self, command, p1, p2=0.0):
        m = VehicleCommand()
        m.command = command
        m.param1 = float(p1)
        m.param2 = float(p2)
        m.target_system = 1
        m.target_component = 1
        m.source_system = 1
        m.source_component = 1
        m.from_external = True
        m.timestamp = 0
        self.pub_cmd.publish(m)

    def _arm_offboard(self):
        self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
        self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def _control(self, s):
        m = String()
        m.data = s
        self.pub_control.publish(m)

    # ────────────────────────────── pattern reference (ENU, shared with synth)
    def _pattern_ref(self, t):
        c = np.array([self.args.cx, self.args.cy, self.args.alt])
        p, v, yaw = _ref(self.scenario["pattern"], t, c=c)
        return p, v, yaw

    # ────────────────────────────── live-log helpers (online_rl_main style)
    @staticmethod
    def _scenario_desc(sc):
        d = []
        if sc.get("mass"):
            _mm = sc["mass"]
            _w = (f"+{_mm['duration_s']:.1f}s" if "duration_s" in _mm else " (to end)")
            d.append(f"payload +{100 * _mm['delta']:.0f}% "
                     f"@ t={_mm['onset_s']:.1f}s{_w}")
        if sc.get("sustained"):
            su = sc["sustained"]
            _w = (f" @ t={su['start_s']:.1f}s+{su['duration_s']:.1f}s"
                  if "start_s" in su else " (full traj)")
            d.append(f"sustained wind {su['speed']:.1f} m/s{_w}")
        for g in sc.get("gusts", []):
            d.append(f"gust {g['speed']:.1f} m/s @ {g['start_s']:.1f}s+{g['duration_s']:.1f}s")
        for tb in sc.get("turbulence", []):
            d.append(f"turb x{tb['boost']:.1f} @ {tb['start_s']:.1f}s+{tb['duration_s']:.1f}s")
        return "; ".join(d) if d else "clean (no disturbance)"

    def _active_flags(self, t):
        """disturbance windows active at traj time t -> {name: bool}."""
        sc = self.scenario or {}
        f = {}
        if sc.get("mass"):
            _mm = sc["mass"]
            _me = _mm["onset_s"] + _mm.get("duration_s", 1e9)
            f[f"PAYLOAD+{100 * _mm['delta']:.0f}%"] = _mm["onset_s"] <= t <= _me
        if sc.get("sustained"):
            su = sc["sustained"]
            _on = (su["start_s"] <= t <= su["start_s"] + su["duration_s"]) \
                if "start_s" in su else True
            f[f"WIND {su['speed']:.0f}m/s"] = _on
        for i, g in enumerate(sc.get("gusts", [])):
            f[f"GUST{i} {g['speed']:.0f}m/s"] = \
                g["start_s"] <= t <= g["start_s"] + g["duration_s"]
        for i, tb in enumerate(sc.get("turbulence", [])):
            f[f"TURB{i} x{tb['boost']:.1f}"] = \
                tb["start_s"] <= t <= tb["start_s"] + tb["duration_s"]
        return f

    # ────────────────────────────── state machine (50 Hz)
    def _tick(self):
        self.state_t += self.TICK

        if self.state == "INIT":
            if self.q_idx >= len(self.queue):
                self.get_logger().info("all trajectories done — shutting down sim")
                self._control("shutdown")
                rclpy.shutdown()
                return
            traj_id, split, heldout = self.queue[self.q_idx]
            _ho_idx = sum(1 for q in self.queue[:self.q_idx] if q[2]) \
                if heldout else None            # heldout 서수 → heldout_plan 소비
            self.scenario = sample_scenario(self.cfg, self.rng, heldout=heldout,
                                            heldout_idx=_ho_idx)
            payload = dict(self.scenario)
            payload.update({"traj_id": traj_id, "split": split})
            m = String()
            m.data = json.dumps(payload)
            self.pub_scenario.publish(m)
            self._airborne = False
            self._flag_state = {}
            self._last_fly_log = -1e9
            self.get_logger().info(
                f"┏━ [{self.q_idx + 1}/{len(self.queue)}] traj {traj_id} ({split}) "
                f"━ {self.scenario['type']} / {self.scenario['pattern']} "
                f"/ {self.scenario['duration_s']:.0f}s")
            self.get_logger().info(
                f"┗━ 외란: {self._scenario_desc(self.scenario)}")
            self._goto("STREAM")

        elif self.state == "STREAM":                 # offboard heartbeat >= 1.5 s
            self._send_offboard()
            p0, _, yaw0 = self._pattern_ref(0.0)
            self._send_setpoint_enu(p0, None, yaw0)
            if self.state_t > 1.5:
                self._arm_offboard()
                self.arm_resend_t = self.state_t
                self.get_logger().info("  [TAKEOFF] Arm + OFFBOARD 전환 시도")
                self._goto("ASCEND")

        elif self.state == "ASCEND":                 # climb & settle at start point
            self._send_offboard()
            p0, _, yaw0 = self._pattern_ref(0.0)
            self._send_setpoint_enu(p0, None, yaw0)
            if self.state_t - self.arm_resend_t > 2.0:      # re-send arm (repo habit)
                self._arm_offboard()
                self.arm_resend_t = self.state_t
                if not self._airborne:
                    self.get_logger().info("  [TAKEOFF] Arm 재시도")
            # 이륙 감지 (online_rl_main 계승: 감지 순간 1회 로그)
            if not self._airborne and self.gt_seen and self.gt_pos[2] > 0.5:
                self._airborne = True
                self.get_logger().info(
                    f"  [TAKEOFF] 이륙 감지! alt={self.gt_pos[2]:.2f} m")
            # 상승 진행 로그 (2s 주기)
            if self.gt_seen and int(self.state_t / 2.0) != \
                    int((self.state_t - self.TICK) / 2.0):
                d0 = float(np.linalg.norm(self.gt_pos - p0))
                self.get_logger().info(
                    f"  [ASCEND {self.state_t:4.1f}s] alt={self.gt_pos[2]:.2f} m, "
                    f"시작점까지 {d0:.2f} m (tol {self.args.settle_tol})")
            settled = self.gt_seen and \
                np.linalg.norm(self.gt_pos - p0) < self.args.settle_tol
            if settled and self.state_t > 4.0:
                self.get_logger().info(
                    f"  ✅ 시작점 정착 (d={float(np.linalg.norm(self.gt_pos - p0)):.2f} m)"
                    f" → FLY, 로깅 ON")
                self._control("start_log")
                self.fly_t = 0.0
                self._goto("FLY")
            elif self.state_t > 30.0:
                self.get_logger().warn("ascend timeout — resetting & retrying")
                self._control("reset")
                self._goto("RESET_WAIT", retry=True)

        elif self.state == "FLY":                    # log & fly the pattern
            self._send_offboard()
            p, v, yaw = self._pattern_ref(self.fly_t)
            self._send_setpoint_enu(p, v, yaw)
            self.fly_t += self.TICK
            # ── 외란 창 전이 로그 (online_rl_main의 🔴 Attack ON / 🟢 OFF 계승):
            #    시나리오의 각 외란 창이 열리고 닫히는 순간을 1회씩 찍는다.
            flags = self._active_flags(self.fly_t)
            for name, on in flags.items():
                prev = self._flag_state.get(name, False)
                if on and not prev:
                    self.get_logger().warn(
                        f"  🔴 외란 ON  @ t={self.fly_t:5.1f}s: {name}")
                elif prev and not on:
                    self.get_logger().info(
                        f"  🟢 외란 OFF @ t={self.fly_t:5.1f}s: {name}")
                self._flag_state[name] = on
            # ── 추적 로그 (2s 주기): 위치·추적오차·활성 외란 ──
            if self.gt_seen and self.fly_t - self._last_fly_log >= 2.0:
                self._last_fly_log = self.fly_t
                terr = float(np.linalg.norm(self.gt_pos - p))
                on_txt = " ".join(n for n, v in flags.items() if v) or "nominal"
                self.get_logger().info(
                    f"  [FLY {self.fly_t:5.1f}/{self.scenario['duration_s']:.0f}s] "
                    f"pos=({self.gt_pos[0]:5.2f},{self.gt_pos[1]:5.2f},"
                    f"{self.gt_pos[2]:4.2f}) 추적err={terr:4.2f} m │ {on_txt}")
            # ── in-flight crash/runaway detection (online_rl_main heritage:
            #    its LEARNING state applied SOFT/WARM/HARD resets on crash).
            #    Under the FINAL strong disturbances (payload +60~90 %,
            #    wind 15~20 m/s) PX4 can lose the fight — ground contact or
            #    failsafe drift. Abort WITHOUT stop_save (the un-saved log is
            #    discarded: the next start_log re-opens fresh rows), world-
            #    reset, and retry the same traj slot (counter-capped below).
            crashed = self.gt_seen and self.fly_t > 1.0 and (
                self.gt_pos[2] < 0.15 or
                np.linalg.norm(self.gt_pos[:2]
                               - np.array([self.args.cx, self.args.cy])) > 12.0)
            if crashed:
                self.get_logger().warn(
                    f"crash/runaway at t={self.fly_t:.1f}s "
                    f"(alt={self.gt_pos[2]:.2f} m) — discard log & retry")
                self._control("reset")
                self._goto("RESET_WAIT", retry=True)
            elif self.fly_t >= self.scenario["duration_s"]:
                self._control("stop_save")
                self._goto("SAVE_WAIT")

        elif self.state == "SAVE_WAIT":              # give the sim a beat to write
            if self.state_t > 1.0:
                self._control("reset")
                self.q_idx += 1
                self.retry_count = 0                 # clean save -> reset cap
                self._goto("RESET_WAIT")

        elif self.state == "RESET_WAIT":             # world reset + PX4 re-settle
            if self.state_t > self.args.reset_wait:
                self.gt_seen = False
                self._goto("INIT")

    def _goto(self, s, retry=False):
        if retry:
            # retry cap = the practical stand-in for online_rl_main's
            # HARD_RESET tier: if the same traj slot keeps failing (PX4
            # failsafe state a world-reset cannot clear, un-flyable scenario
            # draw, ...), skip it after MAX_RETRY instead of looping forever.
            # A fresh scenario is re-sampled on each retry (rng advances).
            self.retry_count += 1
            if self.retry_count >= self.MAX_RETRY:
                self.get_logger().warn(
                    f"traj slot [{self.q_idx + 1}/{len(self.queue)}]: "
                    f"{self.retry_count} consecutive failures — SKIPPING")
                self.q_idx += 1
                self.retry_count = 0
        self.state = s
        self.state_t = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=200)
    ap.add_argument("--n_heldout", type=int, default=50)
    ap.add_argument("--start_id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--px4_ns", type=str, default="",
                    help="PX4 topic namespace ('' = bare /fmu/...)")
    ap.add_argument("--cx", type=float, default=5.0, help="pattern center East [m]")
    ap.add_argument("--cy", type=float, default=5.0, help="pattern center North [m]")
    ap.add_argument("--alt", type=float, default=1.5, help="cruise altitude (ENU up) [m]")
    ap.add_argument("--settle_tol", type=float, default=0.4)
    ap.add_argument("--reset_wait", type=float, default=6.0)
    args = ap.parse_args()

    rclpy.init()
    node = Commander(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
