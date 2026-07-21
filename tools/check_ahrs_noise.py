#!/usr/bin/env python3
"""
Measure the PX4 EKF2 attitude-estimation error against Isaac ground truth,
to justify the synthesized attitude-measurement noise of the paper:

    z_att = gt_att + xi,  xi ~ N(0, sigma^2 I),  sigma = 0.01 rad

If the measured EKF2 RMS error is BELOW sigma, the synthesized noise
upper-bounds a realistic AHRS feed and the reported accuracies are
conservative -- which is the sentence this tool produces evidence for.

Run it NEXT TO a flying session (engine + QGC + pilot, or the scripted
commander). It only subscribes; nothing is modified:

    python3 tools/check_ahrs_noise.py --duration 60

Protocol suggestion: hover ~10 s first (per-axis MEAN should be ~0 there;
a constant yaw offset of +-90/180 deg would indicate a frame-convention
mismatch -- see the note printed with the results), then fly normally.

Frames: PX4 reports attitude NED/FRD; the ground truth is ENU/FLU.
Euler (ZYX) mapping used here:  roll_flu = roll_frd,
pitch_flu = -pitch_frd,  yaw_enu = wrap(pi/2 - yaw_ned).
Angular rate: (wx, -wy, -wz).
"""
import argparse
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)
from nav_msgs.msg import Odometry

try:
    from px4_msgs.msg import VehicleAttitude, VehicleAngularVelocity
    try:
        from px4_msgs.msg import VehicleOdometry
    except ImportError:
        VehicleOdometry = None
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "px4_msgs import failed -- source the ROS2 workspace that "
        "commander.py uses (same environment).\n" + str(e))

GT_DT = 0.02


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_to_euler_zyx(x, y, z, w):
    """[x,y,z,w] quaternion -> (yaw, pitch, roll), aerospace ZYX.
    Pure numpy (the system scipy is binary-incompatible with numpy 2.x)."""
    # roll (x-axis)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    # pitch (y-axis), clamped for numerical safety
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(s)
    # yaw (z-axis)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw, pitch, roll


class AhrsCheck(Node):
    def __init__(self, a):
        super().__init__("ahrs_check")
        self.a = a
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(Odometry, "/gt/odometry", self._cb_gt, qos)
        ns = a.px4_ns.rstrip("/")
        for topic in (f"{ns}/fmu/out/vehicle_attitude",
                      f"{ns}/fmu/out/vehicle_attitude_v1"):
            self.create_subscription(VehicleAttitude, topic, self._cb_att, qos)
        for topic in (f"{ns}/fmu/out/vehicle_angular_velocity",
                      f"{ns}/fmu/out/vehicle_angular_velocity_v1"):
            self.create_subscription(VehicleAngularVelocity, topic,
                                     self._cb_rate, qos)
        # fallback: many PX4 uXRCE-DDS configs do not publish
        # vehicle_angular_velocity but DO publish vehicle_odometry,
        # whose angular_velocity field carries the same EKF2 body rates
        if VehicleOdometry is not None:
            for topic in (f"{ns}/fmu/out/vehicle_odometry",
                          f"{ns}/fmu/out/vehicle_odometry_v1"):
                self.create_subscription(VehicleOdometry, topic,
                                         self._cb_odom, qos)

        self.att_px4 = None          # latest EKF2 euler (ENU/FLU-converted)
        self.rate_px4 = None
        self.gt_n = 0
        self.t0 = None
        self.err_att = []            # [roll, pitch, yaw] error rows
        self.err_rate = []
        self.last_print = -5.0

    # ---------------------------------------------------------- PX4 side
    def _cb_att(self, m):
        # px4 q = [w, x, y, z], NED/FRD
        yaw, pitch, roll = quat_to_euler_zyx(m.q[1], m.q[2], m.q[3], m.q[0])
        self.att_px4 = np.array([roll, -pitch, wrap(math.pi / 2 - yaw)])

    def _cb_rate(self, m):
        self.rate_px4 = np.array([m.xyz[0], -m.xyz[1], -m.xyz[2]])
        self.rate_src = "vehicle_angular_velocity"

    def _cb_odom(self, m):
        if getattr(self, "rate_src", None) == "vehicle_angular_velocity":
            return                       # prefer the dedicated topic if alive
        w = m.angular_velocity           # body FRD
        self.rate_px4 = np.array([w[0], -w[1], -w[2]])
        self.rate_src = "vehicle_odometry"

    # ------------------------------------------------------------ GT side
    def _cb_gt(self, m):
        self.gt_n += 1
        t = self.gt_n * GT_DT
        if self.att_px4 is None:
            return
        if self.t0 is None:
            self.t0 = t
            print("[ahrs] both sources alive -- collecting "
                  f"{self.a.duration:.0f}s (fly or hover)")
        q = m.pose.pose.orientation
        yaw, pitch, roll = quat_to_euler_zyx(q.x, q.y, q.z, q.w)
        gt_att = np.array([roll, pitch, yaw])
        e = np.array([wrap(d) for d in (self.att_px4 - gt_att)])
        self.err_att.append(e)
        if self.rate_px4 is not None:
            w = m.twist.twist.angular
            self.err_rate.append(self.rate_px4 - np.array([w.x, w.y, w.z]))

        el = t - self.t0
        if el - self.last_print >= 5.0:
            self.last_print = el
            rms = np.sqrt((np.array(self.err_att[-250:]) ** 2).mean(0))
            print(f"[ahrs] t={el:4.0f}s  n={len(self.err_att):5d}  "
                  f"last-5s RMS r/p/y = "
                  f"{rms[0]:.4f}/{rms[1]:.4f}/{rms[2]:.4f} rad")
        if el >= self.a.duration:
            self._report()
            raise SystemExit(0)

    # ------------------------------------------------------------- report
    def _report(self):
        E = np.array(self.err_att)
        print("\n=== EKF2 attitude error vs ground truth "
              f"({len(E)} samples @ 50 Hz) ===")
        names = ["roll ", "pitch", "yaw  "]
        rms_all = []
        for i, nm in enumerate(names):
            mean, std = E[:, i].mean(), E[:, i].std()
            rms = float(np.sqrt((E[:, i] ** 2).mean()))
            rms_all.append(rms)
            # 1/e autocorrelation time (crude): first lag below 1/e
            x = E[:, i] - mean
            den = float((x * x).sum())
            tau = float("nan")
            if den > 0:
                for lag in range(1, min(len(x) - 1, 500)):
                    r = float((x[:-lag] * x[lag:]).sum()) / den
                    if r < 1 / math.e:
                        tau = lag * GT_DT
                        break
            print(f"  {nm}: mean {mean:+.4f}  std {std:.4f}  RMS {rms:.4f} rad"
                  f"  ({math.degrees(rms):.2f} deg)   corr-time ~{tau:.2f} s")
        worst = max(rms_all)
        sig = self.a.sigma
        print(f"\n  synthesized sigma = {sig:.3f} rad "
              f"({math.degrees(sig):.2f} deg)")
        if worst < sig:
            print(f"  -> EKF2 worst-axis RMS {worst:.4f} < sigma:  "
                  "synthesized noise UPPER-BOUNDS the AHRS error. "
                  "Reported accuracies are conservative.  [OK]")
        else:
            print(f"  -> EKF2 worst-axis RMS {worst:.4f} >= sigma: "
                  "inspect per-axis means first (a large constant mean = "
                  "frame-convention offset, not estimator noise).")
        if abs(E[:, 2].mean()) > 0.3:
            print("  [!] yaw mean is large -- likely the NED->ENU yaw "
                  "convention; re-check the mapping printed in the header "
                  "before quoting numbers.")
        if self.err_rate:
            W = np.array(self.err_rate)
            rms = np.sqrt((W ** 2).mean(0))
            src = getattr(self, "rate_src", "?")
            pooled = float(np.sqrt((W ** 2).mean()))
            print(f"  angular-rate error RMS x/y/z = "
                  f"{rms[0]:.4f}/{rms[1]:.4f}/{rms[2]:.4f} rad/s "
                  f"(pooled {pooled:.4f}; source: {src}; "
                  f"synthesized sigma = 0.005)")
        else:
            print("  angular-rate: no PX4 rate source received -- "
                  "check `ros2 topic list | grep -E 'angular|odometry'`")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--sigma", type=float, default=0.01,
                    help="synthesized attitude noise std [rad] to compare against")
    ap.add_argument("--px4_ns", default="",
                    help="PX4 topic namespace, e.g. /px4_1 (empty for none)")
    a = ap.parse_args()
    rclpy.init()
    node = AhrsCheck(a)
    print("[ahrs] waiting for /gt/odometry and /fmu/out/vehicle_attitude ...")
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
