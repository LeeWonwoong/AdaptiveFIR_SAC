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
                          VehicleCommand, VehicleStatus)

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

        # PX4 feedback. Without this the commander armed BLINDLY and re-sent the
        # command every 2 s: right after an Isaac world reset PX4's EKF2 has to
        # re-converge, arming is REJECTED until the pre-flight checks pass, and
        # the old code simply spammed until it happened to succeed (the "Arm
        # 재시도" loop). We now wait for the checks, then verify the result.
        # PX4 publishes /fmu/out/* BEST_EFFORT + VOLATILE. A TRANSIENT_LOCAL
        # subscriber is QoS-INCOMPATIBLE with a VOLATILE publisher, so the
        # previous profile received NOTHING — that is why the log said
        # "pre-flight unknown". VOLATILE fixes it.
        qos_px4 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST, depth=1)
        # PX4 >= v1.16 publishes VERSIONED topics (vehicle_status_v1); older
        # firmware uses the unversioned name. Subscribe to both — whichever
        # exists will fire, the other stays silent. (This is why the previous
        # log showed armed=False while the vehicle was visibly flying: the
        # subscription pointed at a topic that this PX4 never publishes.)
        for _topic in (f"{ns}/fmu/out/vehicle_status_v1",
                       f"{ns}/fmu/out/vehicle_status"):
            self.create_subscription(VehicleStatus, _topic,
                                     self._cb_status, qos_px4)
        self.px4_armed = False
        self.px4_ready = False          # pre-flight checks pass
        self.px4_offboard = False
        self._arm_tries = 0

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
        self.sim_now = 0.0
        self.fly_t0_sim = -1.0
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
        _st = msg.header.stamp
        _t = float(_st.sec) + float(_st.nanosec) * 1e-9
        if _t > 0.0:
            self.sim_now = _t

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

    def _cb_status(self, msg):
        self.px4_armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self.px4_offboard = (msg.nav_state ==
                             VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        # newer px4_msgs expose the pre-flight result; fall back gracefully
        self.px4_ready = bool(getattr(msg, "pre_flight_checks_pass", True))

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
            _su0 = sc["sustained"]
            _sl = _su0 if isinstance(_su0, list) else [_su0]
            _wtxt = ", ".join(f"{w['start_s']:.1f}+{w['duration_s']:.1f}s"
                              if "start_s" in w else "full" for w in _sl)
            d.append(f"sustained wind {_sl[0]['speed']:.1f} m/s "
                     f"[{len(_sl)}win: {_wtxt}]")
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
            _su0 = sc["sustained"]
            _sl = _su0 if isinstance(_su0, list) else [_su0]
            _on = any((w["start_s"] <= t <= w["start_s"] + w["duration_s"])
                      if "start_s" in w else True for w in _sl)
            f[f"WIND {_sl[0]['speed']:.0f}m/s"] = _on
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
                                            train_idx=(None if heldout else traj_id),
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

        elif self.state == "STREAM":
            # Stream the offboard heartbeat + setpoint, and WAIT for PX4 to be
            # ready before arming. PX4 requires (i) setpoints already flowing
            # (>=2 Hz) BEFORE the OFFBOARD switch and (ii) the EKF2 estimate to
            # have converged after the Isaac reset. Arming earlier is simply
            # rejected. 3 s of heartbeat + the pre-flight flag removes the
            # retry loop entirely; the 8 s cap keeps a stuck run from hanging.
            self._send_offboard()
            p0, _, yaw0 = self._pattern_ref(0.0)
            self._send_setpoint_enu(p0, None, yaw0)
            _ready = self.px4_ready and self.state_t > 3.0
            if _ready or self.state_t > 8.0:
                if not _ready:
                    self.get_logger().warn(
                        "  [TAKEOFF] pre-flight 미확인 — 8s 경과, 강행")
                self._arm_offboard()
                self.arm_resend_t = self.state_t
                self._arm_tries = 1
                self.get_logger().info(
                    f"  [TAKEOFF] Arm + OFFBOARD 전환 (heartbeat {self.state_t:.1f}s, "
                    f"pre-flight {'OK' if self.px4_ready else 'unknown'})")
                self._goto("ASCEND")

        elif self.state == "ASCEND":
            # TWO-PHASE ascend: (1) climb straight up at the CURRENT xy to the
            # pattern altitude, (2) only then slide horizontally to the start
            # point. Commanding a combined 3-m-lateral + 4-m-vertical jump at
            # the instant of arming is what intermittently left the vehicle
            # parked at the pattern centre with the horizontal error frozen at
            # ~R0: PX4 accepted the z component but the lateral transition was
            # never engaged. Splitting the motion removes that failure mode,
            # and the nav_state now in the log tells us the mode if it recurs.
            self._send_offboard()
            p0, _, yaw0 = self._pattern_ref(0.0)
            # Ambient-wind-aware settle gate: at 3.5 m/s a hovering quad
            # wanders 0.9-1.4 m, so the still-air 0.4 m tolerance is
            # unreachable — that (not arming) caused the endless ASCEND loop.
            _turb = float(self.scenario.get("ambient_turb_std", 0.0))
            _tol = max(self.args.settle_tol, 0.45 * _turb)
            if self.gt_pos is not None and self.gt_pos[2] < 0.8 * p0[2]:
                _sp = np.array([self.gt_pos[0], self.gt_pos[1], p0[2]])  # phase 1
            else:
                _sp = p0                                                  # phase 2
            self._send_setpoint_enu(_sp, None, yaw0)
            if int(self.state_t) % 4 == 0 and abs(self.state_t - int(self.state_t)) < 0.03:
                self.get_logger().info(
                    f"  [ASCEND] armed={self.px4_armed} offboard={self.px4_offboard}")
            if (self.px4_armed and not self.px4_offboard
                    and self.state_t - self.arm_resend_t > 2.0):
                self._vehicle_cmd(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.arm_resend_t = self.state_t
                self.get_logger().warn("  [ASCEND] OFFBOARD 재전환 (nav_state 미달)")
            # Re-send ONLY if PX4 actually reports it is not armed/offboard.
            # (The old code re-sent unconditionally every 2 s, which is what
            # produced the "Arm 재시도" spam even on runs that were fine.)
            if (not (self.px4_armed and self.px4_offboard)
                    and self.state_t - self.arm_resend_t > 2.0):
                self._arm_offboard()
                self.arm_resend_t = self.state_t
                self._arm_tries += 1
                if not self._airborne:
                    self.get_logger().warn(
                        f"  [TAKEOFF] Arm 재시도 #{self._arm_tries} "
                        f"(armed={self.px4_armed}, offboard={self.px4_offboard}, "
                        f"pre-flight={self.px4_ready})")
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
                    f"시작점까지 {d0:.2f} m (tol {_tol:.2f})")
            # Under an ambient airflow a hovering quad WANDERS: at 3.5 m/s the
            # position error oscillates 0.9-1.4 m, so the still-air tolerance
            # of 0.4 m is unreachable — this, not arming, was the endless
            # ASCEND loop. Widen the gate with the scenario's own wind level.
            settled = self.gt_seen and \
                np.linalg.norm(self.gt_pos - p0) < _tol
            if settled and self.state_t > 4.0:
                self.get_logger().info(
                    f"  ✅ 시작점 정착 (d={float(np.linalg.norm(self.gt_pos - p0)):.2f} m)"
                    f" → FLY, 로깅 ON")
                self._control("start_log")
                self.fly_t = 0.0
                self.fly_t0_sim = -1.0
                self._goto("FLY")
            elif self.state_t > 30.0:
                # Do NOT reset: if the altitude is reached the vehicle is fine,
                # merely being pushed around by the wind. Start the pattern —
                # the tracking loop converges within a couple of seconds, and
                # evaluation trims the first 6 s anyway. (The old reset+retry
                # is what silently ATE trajectories h2/h4 from the last gate.)
                if self.gt_seen and abs(self.gt_pos[2] - p0[2]) < 0.6:
                    d0 = float(np.linalg.norm(self.gt_pos - p0))
                    self.get_logger().warn(
                        f"  [ASCEND] 정착 미달(d={d0:.2f}>{_tol:.2f})이지만 고도 도달"
                        f" — 패턴 시작 (초반 과도는 평가에서 절삭)")
                    self._control("start_log")
                    self.fly_t = 0.0
                    self.fly_t0_sim = -1.0
                    self._goto("FLY")
                else:
                    self.get_logger().warn("ascend timeout — resetting & retrying")
                    self._control("reset")
                    self._goto("RESET_WAIT", retry=True)

        elif self.state == "FLY":                    # log & fly the pattern
            self._send_offboard()
            # ── SIM-TIME pattern phase (fix 2026-07-13) ──────────────────
            # fly_t used to advance by the WALL-clock tick. The commander is
            # paced by real time while Isaac runs at speed-up x (RTF): at an
            # achieved RTF of 2.33 the pattern therefore unfolded 2.33x
            # SLOWER in simulation time. Measured on the v9 dataset: mean
            # speed 0.88-1.13 m/s vs the commanded 2.0, bank 2-4 deg vs the
            # commanded ~8, and T = 2.33 x the planned 50 s. This single bug
            # slowed every "faster trajectory" round (1.2x, 1.35x) and is why
            # the payload x,y signature never materialised. The phase now
            # advances with the simulation clock taken from the GT odometry
            # header stamps; the wall tick remains only as a fallback when
            # stamps are absent (sim_now stays 0).
            if self.sim_now > 0.0:
                if self.fly_t0_sim < 0.0:
                    self.fly_t0_sim = self.sim_now
                    self.get_logger().info(
                        "  [FLY] pattern phase driven by SIM time (odometry stamps)")
                self.fly_t = self.sim_now - self.fly_t0_sim
            else:
                if self.fly_t0_sim < 0.0:
                    self.fly_t0_sim = 0.0
                    self.get_logger().warn(
                        "  [FLY] odometry stamps unavailable — wall-clock fallback")
                self.fly_t += self.TICK
            p, v, yaw = self._pattern_ref(self.fly_t)
            self._send_setpoint_enu(p, v, yaw)
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
