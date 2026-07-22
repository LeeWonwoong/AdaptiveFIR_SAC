#!/usr/bin/env python
"""
datagen/run_datagen.py — Isaac Sim + Pegasus + PX4 data-generation engine
===========================================================================
Adapted from the user's Issacsim-rhukf/run_sim.py (attack machinery removed;
scenario-driven wind + TRUE-MASS injection + 50 Hz dataset logging added).

!! REQUIRES the Isaac Sim python environment (ISAACSIM_PYTHON) with Pegasus
!! and px4_msgs — NOT runnable in a plain python env. The logic mirrors the
!! user's verified run_sim.py; items marked [VERIFY-IN-SIM] must be checked
!! once live (frame conventions), using the dataset validation checklist in
!! README (1-step propagation error must be ~1e-3 m on nominal segments).

Terminal 1:  ISAACSIM_PYTHON datagen/run_datagen.py --headless 1 --out data
Terminal 2:  python datagen/commander.py --n_train 200 --n_heldout 50

ROS topics (commander -> sim):
  /datagen/scenario   String(JSON): scenario dict + {"traj_id", "split"}
  /datagen/control    String: "start_log" | "stop_save" | "reset" | "shutdown"
Sim -> commander:
  /gt/odometry        50 Hz ground truth (commander uses it for progress checks)
"""
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--headless", type=int, default=1)
parser.add_argument("--speed", type=float, default=1.0,
                    help="requested real-time factor (lockstep permitting)")
parser.add_argument("--px4_ns", type=str, default="auto")
parser.add_argument("--out", type=str, default="data")
_pre_args, _ = parser.parse_known_args()

# ── Isaac Sim bootstrap (must precede pxr/omni imports) ──
from isaacsim import SimulationApp                                   # noqa: E402
simulation_app = SimulationApp({"headless": bool(_pre_args.headless)})

import carb                                                           # noqa: E402
import omni.timeline                                                  # noqa: E402
import omni.usd                                                       # noqa: E402
import numpy as np                                                    # noqa: E402
from omni.isaac.core.world import World                               # noqa: E402
from omni.isaac.core.prims import RigidPrimView                       # noqa: E402
from scipy.spatial.transform import Rotation                          # noqa: E402

import rclpy                                                          # noqa: E402
from rclpy.node import Node                                           # noqa: E402
from rclpy.qos import (QoSProfile, ReliabilityPolicy, DurabilityPolicy,
                       HistoryPolicy)                                 # noqa: E402
from std_msgs.msg import String                                       # noqa: E402
from nav_msgs.msg import Odometry                                     # noqa: E402
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint  # noqa: E402

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS  # noqa: E402
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend, PX4MavlinkBackendConfig)                       # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import (
    Multirotor, MultirotorConfig)                                     # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import (
    PegasusInterface)                                                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datagen.wind import WindModel                                    # noqa: E402
from datagen.traj_logger import TrajLogger                            # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_calibration(path=None):
    with open(path or os.path.join(_HERE, "calibration.json")) as f:
        return json.load(f)


# PX4 body frame is FRD; our model/state is FLU.  [VERIFY-IN-SIM]
# tau_flu = (tau_x, -tau_y, -tau_z)_frd ; thrust: |z_frd| points up in FLU.
FRD_TO_FLU = np.array([1.0, -1.0, -1.0])


def to_physical_u(thrust_sp, torque_sp, calib):
    """PX4 normalized setpoints -> physical [T(N), tau_x, tau_y, tau_z (Nm)]
    (same mapping as the user's env/ukf_filter.to_physical_u, + FRD->FLU)."""
    u = np.zeros(4)
    u[0] = abs(thrust_sp[2]) * calib["C_thrust"]
    tau = torque_sp * FRD_TO_FLU
    u[1] = tau[0] * calib["C_torque_xy"]
    u[2] = tau[1] * calib["C_torque_xy"]
    u[3] = tau[2] * calib["C_torque_z"]
    return u


class DatagenApp:
    def __init__(self, args):
        self.args = args
        self.sim_time = 0.0
        self.physics_dt = 1.0 / 250.0            # cfg.physics_hz
        self.log_every = 5                        # 250 Hz / 5 = 50 Hz (cfg.log_hz)
        self.calib = load_calibration()
        self.mass_nominal = float(self.calib.get("mass", 1.372))

        self.cmd_thrust = np.zeros(3)
        self.cmd_torque = np.zeros(3)

        self.scenario = None                      # active scenario dict
        self.split = "train"
        self.traj_id = 0
        self.traj_t0 = 0.0                        # sim_time at start_log
        self.mass_applied = False
        self.needs_reset = False
        self.stop_sim = False

        self.logger = TrajLogger(args.out)
        # Load the PROJECT config by absolute path. A bare `import config`
        # inside Isaac's python.sh resolves to the bundled cv2/config.py
        # (pip_prebundle shadows the project root) and crashes the loader.
        import importlib.util as _ilu
        _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "config.py")
        _spec = _ilu.spec_from_file_location("afir_project_config", _cfg_path)
        _mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
        _c = _mod.Config()
        self._cfg = _c
        self.amb_turb = float(getattr(_c, "ambient_turb_std", 0.0))
        self.amb_bw = float(getattr(_c, "ambient_turb_bw", 2.0))
        self.wind = WindModel({}, turb_intensity=self.amb_turb,
                              turb_bw=self.amb_bw)

        # ── ROS ──
        rclpy.init()
        self.ros_node = Node("datagen_engine")
        self.gt_pub = self.ros_node.create_publisher(Odometry, "/gt/odometry", 10)
        self.ros_node.create_subscription(String, "/datagen/scenario",
                                          self._cb_scenario, 10)
        self.ros_node.create_subscription(String, "/datagen/control",
                                          self._cb_control, 10)
        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=5)
        ns = self._resolve_ns(args.px4_ns)
        self.ros_node.create_subscription(
            VehicleThrustSetpoint, f"{ns}/fmu/out/vehicle_thrust_setpoint",
            lambda m: self.cmd_thrust.__setitem__(slice(None), m.xyz[:3]), px4_qos)
        self.ros_node.create_subscription(
            VehicleTorqueSetpoint, f"{ns}/fmu/out/vehicle_torque_setpoint",
            lambda m: self.cmd_torque.__setitem__(slice(None), m.xyz[:3]), px4_qos)
        carb.log_warn(f"[datagen] PX4 namespace = '{ns}'")

        # ── world / vehicle (same as user's run_sim.py) ──
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])
        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0, "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe})
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        self.vehicle = Multirotor(
            "/World/quadrotor", ROBOTS["Iris"], 0, [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat(),
            config=config_multirotor)
        self.world.reset()
        self.stage = omni.usd.get_context().get_stage()

        self.body_view = None
        self._setup_body_view()

        # Floor guides + overview camera for GUI runs (HEADLESS=0). Never
        # created in headless datagen, so the generated data is unchanged.
        if not int(getattr(args, "headless", 1)):
            self._setup_manual_visuals()

    # ---------------- ROS callbacks ----------------
    def _cb_scenario(self, msg):
        try:
            d = json.loads(msg.data)
            self.scenario = d
            self.split = d.get("split", "train")
            self.traj_id = int(d.get("traj_id", self.traj_id))
            # Light ambient wind. Held-out rows carry an EXPLICIT value from
            # heldout_plan (so the paper trajectories are deterministic);
            # training rows draw one per trajectory from the configured range.
            if "ambient_turb_std" in d:
                _ti = float(d["ambient_turb_std"])           # planned
            else:
                _tr = getattr(self._cfg, "ambient_turb_std_range", None)
                if _tr:
                    _rng = np.random.default_rng(int(d.get("seed", 0)) + 9973)
                    _ti = float(_rng.uniform(*_tr))
                else:
                    _ti = self.amb_turb
                d["ambient_turb_std"] = _ti                  # log into the meta
            self.wind = WindModel(d, seed=int(d.get("seed", 0)),
                                  turb_intensity=_ti, turb_bw=self.amb_bw)
            self.mass_applied = False
            self._set_mass(self.mass_nominal)
            carb.log_warn(f"[datagen] scenario #{self.traj_id} "
                          f"({d.get('type')}, {d.get('pattern')}, {self.split})")
        except Exception as e:
            carb.log_error(f"scenario parse fail: {e}")

    def _cb_control(self, msg):
        cmd = msg.data.strip()
        if cmd == "start_log":
            consts = dict(mass_nominal=self.mass_nominal,
                          dt=self.physics_dt * self.log_every,
                          source="isaac_pegasus_px4",
                          calibration=self.calib)
            self.logger.start(self.scenario or {}, consts)
            self.traj_t0 = self.sim_time
        elif cmd == "stop_save":
            self.logger.stop_save(self.split, self.traj_id)
        elif cmd == "reset":
            self.needs_reset = True
        elif cmd == "shutdown":
            self.stop_sim = True

    def _resolve_ns(self, configured):
        if configured != "auto":
            return configured
        try:  # best-effort autodetect (mirrors user's repo)
            time.sleep(1.0)
            for name, _ in self.ros_node.get_topic_names_and_types():
                if name.endswith("/fmu/out/vehicle_thrust_setpoint"):
                    return name.rsplit("/fmu", 1)[0]
        except Exception:
            pass
        return ""

    # ---------------- physics helpers ----------------
    def _setup_manual_visuals(self):
        """Floor guides + tilted overview camera (GUI runs only).

        Visual-only USD prims (no physics APIs), so nothing collides and the
        logged data is bit-identical to a headless run.

          * bright-green rectangle .... UWB anchor square (hard boundary)
          * orange poles + ticks ...... anchor positions; ticks at 1.5 / 2.0 m
                                        (green, recommended band) and 2.5 m
                                        (orange, ceiling below the 3 m plane)
          * cyan square ............... 4.4 m lap guide for the payload /
                                        nominal runs (one lap ~14 s at the
                                        measured 1.25 m/s cruise)
          * magenta centre line ....... 5 m straight traverse for the wind
                                        runs, along the wind axis (y = mid).
                                        White puck = START (upwind end): with
                                        dir 180 deg the gust pushes -x, so
                                        flying east->west needs no lateral
                                        correction at all.

        Camera tilt/height are env-tunable so the pilot can trade off the
        top-down view (x-y placement) against the side view (altitude):
            MANUAL_CAM_TILT=50 MANUAL_CAM_H=15 bash datagen/launch_manual.sh ...
        """
        try:
            import os as _os
            from pxr import UsdGeom, Gf
            stage = self.stage
            root = "/World/ManualAids"
            UsdGeom.Scope.Define(stage, root)

            anchors = [tuple(a) for a in self._cfg.anchors]
            xs = [a[0] for a in anchors]
            ys = [a[1] for a in anchors]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)

            def _cube(name, pos, half, color):
                p = UsdGeom.Cube.Define(stage, f"{root}/{name}")
                xf = UsdGeom.Xformable(p.GetPrim())
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
                xf.AddScaleOp().Set(Gf.Vec3f(*half))   # Cube size=2 -> half-extent
                p.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

            # ── anchor square (hard boundary) ──
            t, zs = 0.06, 0.02
            green = (0.10, 0.90, 0.20)
            _cube("edge_ymin", (cx, y0, zs), ((x1 - x0) / 2 + t, t, zs), green)
            _cube("edge_ymax", (cx, y1, zs), ((x1 - x0) / 2 + t, t, zs), green)
            _cube("edge_xmin", (x0, cy, zs), (t, (y1 - y0) / 2 + t, zs), green)
            _cube("edge_xmax", (x1, cy, zs), (t, (y1 - y0) / 2 + t, zs), green)

            # ── anchors: pole + head, plus altitude ticks on the tall pair ──
            for i, (ax, ay, az) in enumerate(anchors):
                h = max(az, 0.4)
                _cube(f"anchor{i}_pole", (ax, ay, h / 2),
                      (0.03, 0.03, h / 2), (1.0, 0.55, 0.10))
                _cube(f"anchor{i}_head", (ax, ay, az if az > 0 else 0.4),
                      (0.09, 0.09, 0.09), (1.0, 0.35, 0.05))
                if az <= 0:
                    continue
                for zt, col in ((1.5, (0.1, 0.9, 0.2)), (2.0, (0.1, 0.9, 0.2)),
                                (2.5, (1.0, 0.55, 0.1))):
                    if zt < az:
                        _cube(f"tick{i}_{int(zt * 10)}", (ax, ay, zt),
                              (0.16, 0.16, 0.015), col)

            # ── AIRBORNE 3-D reference paths (v13.1): the guide the pilot
            # actually FLIES, drawn at altitude with the desired z profile
            # baked in, plus ">" chevrons showing the travel direction.
            # (Floor markings could not convey the z choreography.)
            import math as _m

            def _chevron(name, pos, yaw_deg, color, size=0.16):
                """">"-shaped direction marker at `pos`, tip pointing
                toward `yaw_deg` (0 = +x, CCW)."""
                for si, off in ((1, 150.0), (2, -150.0)):
                    ang = _m.radians(yaw_deg + off)
                    cxy = (pos[0] + 0.5 * size * _m.cos(ang),
                           pos[1] + 0.5 * size * _m.sin(ang), pos[2])
                    p = UsdGeom.Cube.Define(stage, f"{root}/{name}_{si}")
                    xf = UsdGeom.Xformable(p.GetPrim())
                    xf.ClearXformOpOrder()
                    xf.AddTranslateOp().Set(Gf.Vec3d(*cxy))
                    xf.AddRotateZOp().Set(yaw_deg + off)
                    xf.AddScaleOp().Set(Gf.Vec3f(0.5 * size, 0.022, 0.022))
                    p.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])

            def _dotted_path(name, pts, color, half=0.035):
                for k, q in enumerate(pts):
                    _cube(f"{name}_{k:03d}", tuple(q), (half, half, half), color)

            # ① lap path (mass / nominal): 4.4 m square at altitude, CCW from
            #   the SW corner, with the z choreography built in:
            #     side 1 (S, +x): climb 1.5 -> 1.9   (be HIGH before pickup)
            #     side 2 (E, +y): hold 1.9           (6 s pickup lands here)
            #     side 3 (N, -x): descend 1.9 -> 1.5
            #     side 4 (W, -y): hold 1.5
            margin = 1.8
            wx0, wx1 = x0 + margin, x1 - margin
            wy0, wy1 = y0 + margin, y1 - margin
            cyan = (0.15, 0.65, 1.00)
            side = wx1 - wx0
            npt = 12                              # dots per side
            lap = []
            for k in range(npt):                  # S side: climb
                s = k / npt
                lap.append((wx0 + s * side, wy0, 1.5 + 0.4 * s))
            for k in range(npt):                  # E side: hold high
                lap.append((wx1, wy0 + (k / npt) * side, 1.9))
            for k in range(npt):                  # N side: descend
                s = k / npt
                lap.append((wx1 - s * side, wy1, 1.9 - 0.4 * s))
            for k in range(npt):                  # W side: hold low
                lap.append((wx0, wy1 - (k / npt) * side, 1.5))
            _dotted_path("lap3d", lap, cyan)
            _cube("lap_startpuck", (wx0, wy0, 1.5), (0.10, 0.10, 0.10),
                  (1.0, 1.0, 1.0))                # white = START (SW corner)
            wmid = 0.5 * (wx0 + wx1)
            for nm, pos, yaw in (
                    ("lapdir_s", (wmid, wy0, 1.72), 0.0),
                    ("lapdir_e", (wx1, 0.5 * (wy0 + wy1), 1.9), 90.0),
                    ("lapdir_n", (wmid, wy1, 1.68), 180.0),
                    ("lapdir_w", (wx0, 0.5 * (wy0 + wy1), 1.5), 270.0)):
                _chevron(nm, pos, yaw, cyan)

            # ② wind path: 5 m straight traverse at y = cy, east -> west
            #   (with dir=180 deg the gust pushes -x: downwind leg, no lateral
            #   correction), one gentle z arch 1.6 -> 2.0 -> 1.5 -> 1.7 so the
            #   vertical channel is exercised too.
            lx0, lx1 = cx - 2.5, cx + 2.5
            mag = (1.00, 0.25, 0.85)
            nwp = 22
            wpath = []
            for k in range(nwp + 1):
                s = k / nwp                       # 0 = east start, 1 = west end
                xw = lx1 - 5.0 * s
                zw = 1.7 + 0.3 * _m.sin(2 * _m.pi * s) - 0.1 * s
                wpath.append((xw, cy, zw))
            _dotted_path("wind3d", wpath, mag)
            _cube("wind_startpuck", (lx1, cy, wpath[0][2]),
                  (0.11, 0.11, 0.11), (1.0, 1.0, 1.0))   # white = START (east)
            for kf in (0.25, 0.5, 0.75):
                q = wpath[int(kf * nwp)]
                _chevron(f"winddir_{int(kf * 100)}", q, 180.0, mag)

            # ── tilted overview camera ──
            import math as _math
            tilt = float(_os.environ.get("MANUAL_CAM_TILT", "42"))
            hcam = float(_os.environ.get("MANUAL_CAM_H", "13.5"))
            dback = _math.tan(_math.radians(tilt)) * (hcam - 1.0)
            cam = UsdGeom.Camera.Define(stage, f"{root}/OverviewCam")
            xf = UsdGeom.Xformable(cam.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy - dback, hcam))
            xf.AddRotateXOp().Set(tilt)
            try:
                from omni.kit.viewport.utility import get_active_viewport
                vp = get_active_viewport()
                if vp is not None:
                    vp.camera_path = f"{root}/OverviewCam"
            except Exception as e:
                carb.log_warn(f"[manual-aids] camera switch failed ({e}); pick "
                              f"{root}/OverviewCam in the viewport menu")
            carb.log_warn(
                f"[manual-aids] anchor square {x0:g}-{x1:g} m | lap square "
                f"{wx0:g}-{wx1:g} m (side {wx1 - wx0:g}) | wind line "
                f"x {lx0:g}->{lx1:g} at y={cy:g} | cam tilt {tilt:g} deg h={hcam:g}")
        except Exception as e:
            carb.log_warn(f"[manual-aids] skipped: {e}")

    def _setup_body_view(self):
        try:
            self.body_view = RigidPrimView(
                prim_paths_expr="/World/quadrotor/body", name="body_view")
            self.world.scene.add(self.body_view)
            self.body_view.initialize()
            try:
                self._inertia_nom = np.array(self.body_view.get_inertias())
            except Exception:
                self._inertia_nom = None
        except Exception as e:
            carb.log_error(f"body_view init failed: {e}")
            self._inertia_nom = None

    def _set_mass(self, m, com=None, inertia_scale=None):
        """TRUE payload injection (scenario 'mass_step'); PX4 keeps believing
        the nominal airframe — the model mismatch we want.

        A payload is not attached at the centre of mass, so besides the scalar
        mass we shift the CoM and grow the inertia tensor. The offset CoM makes
        the thrust vector produce a parasitic torque, which the attitude loop
        must fight — this is what puts REAL model error on x and y as well,
        instead of the artificial z-only disturbance a mass-only change gives.
        """
        try:
            if not self.body_view:
                return
            self.body_view.set_masses(np.array([m], dtype=np.float32))
            if com is not None:
                self.body_view.set_coms(
                    np.array([com], dtype=np.float32).reshape(1, 3),
                    np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
            if inertia_scale is not None and self._inertia_nom is not None:
                self.body_view.set_inertias(
                    (self._inertia_nom * float(inertia_scale)).astype(np.float32))
        except Exception as e:
            carb.log_error(f"payload injection failed: {e}")

    def _do_reset(self):
        self._set_mass(self.mass_nominal, com=[0.0, 0.0, 0.0], inertia_scale=1.0)
        self.mass_applied = False
        self.world.reset()
        self.needs_reset = False
        self.sim_time = 0.0
        carb.log_warn("[datagen] world reset")

    # ---------------- main loop ----------------
    def run(self):
        self.timeline.play()
        speed = max(0.1, float(self.args.speed))
        wall_start = time.time()
        step = 0
        physics_hz = int(1.0 / self.physics_dt)
        render_interval = max(1, physics_hz // 60)

        while simulation_app.is_running() and not self.stop_sim:
            if self.needs_reset:
                self._do_reset()
                wall_start = time.time()
                step = 0
                continue

            t_traj = self.sim_time - self.traj_t0 if self.logger.active else 0.0

            # scenario-driven TRUE mass
            m_true = self.mass_nominal
            sc = self.scenario or {}
            if sc.get("mass") and self.logger.active:
                _m = sc["mass"]
                _end = _m["onset_s"] + _m.get("duration_s", 1e9)   # legacy: to end
                _in_win = _m["onset_s"] <= t_traj <= _end
                if _in_win:
                    m_true = self.mass_nominal * (1.0 + _m["delta"])
                    if not self.mass_applied:
                        _off = float(_m.get("com_offset", 0.0))
                        _dir = float(_m.get("com_dir", 0.0))
                        _com = [_off * np.cos(_dir), _off * np.sin(_dir), 0.0]
                        self._set_mass(m_true, com=_com,
                                       inertia_scale=1.0 + _m["delta"])
                        self.mass_applied = True
                        carb.log_warn(
                            f"[MASS] 🔴 PICKUP {self.mass_nominal:.3f} → "
                            f"{m_true:.3f} kg (+{100 * _m['delta']:.0f}%) "
                            f"@ t={t_traj:.1f}s — PX4는 nominal 신뢰 (실물리 payload)")
                elif self.mass_applied and t_traj > _end:
                    self._set_mass(self.mass_nominal, com=[0.0, 0.0, 0.0],
                                   inertia_scale=1.0)
                    self.mass_applied = False
                    carb.log_warn(
                        f"[MASS] 🟢 RELEASE → {self.mass_nominal:.3f} kg "
                        f"@ t={t_traj:.1f}s (창 {_m['onset_s']:.1f}"
                        f"~{_end:.1f}s 종료)")

            # scenario-driven wind drag force (applied in world frame)
            w_vel, w_force = self.wind.get(t_traj, self.physics_dt)
            if self.body_view is not None and np.any(w_force):
                self.body_view.apply_forces(
                    np.array([w_force], dtype=np.float32), is_global=True)

            # ── 상태 로그 (2 sim-s 주기, run_sim.py [RTF] 계승):
            #    달성 배속 + 활성 바람 세기·항력을 함께 — "실제로 돌고 있고
            #    외란이 실제로 인가되고 있음"을 콘솔에서 확인 가능하게.
            if step % (physics_hz * 2) == 0:
                _now = time.time()
                if not hasattr(self, "_rtf_t0"):
                    self._rtf_t0, self._rtf_s0 = _now, self.sim_time
                else:
                    _dw = _now - self._rtf_t0
                    _ds = self.sim_time - self._rtf_s0
                    if _dw > 0.5:
                        _rtf = _ds / _dw
                        _lag = "" if _rtf >= 0.95 * speed else "  ← compute 한계"
                        _wv = float(np.linalg.norm(w_vel))
                        _wtxt = (f" │ 💨 wind {_wv:.1f} m/s, "
                                 f"drag {float(np.linalg.norm(w_force)):.2f} N"
                                 if self.logger.active and _wv > 0.5 else "")
                        _log = ("REC" if self.logger.active else "idle")
                        print(f"[RTF] {_rtf:.2f}x/{speed:.1f}x "
                              f"(sim={self.sim_time:.0f}s, {_log}){_wtxt}{_lag}",
                              flush=True)
                    self._rtf_t0, self._rtf_s0 = _now, self.sim_time

            do_render = (not self.args.headless) and (step % render_interval == 0)
            self.world.step(render=do_render)
            self.sim_time += self.physics_dt
            step += 1

            # ── 50 Hz: GT publish + dataset row ──
            if step % self.log_every == 0:
                st = self.vehicle.state
                # Pegasus state: position ENU, attitude quat [x,y,z,w],
                # linear_velocity world-ENU, angular_velocity body-FLU
                # eta = [roll, pitch, yaw] (ZYX) to match filter/uav_model.py
                eul = Rotation.from_quat(st.attitude).as_euler("ZYX")   # [yaw,pitch,roll]
                eta = np.array([eul[2], eul[1], eul[0]])
                gt12 = np.concatenate([st.position, st.linear_velocity,
                                       eta, st.angular_velocity])
                u_phys = to_physical_u(self.cmd_thrust, self.cmd_torque, self.calib)
                self.logger.log(t_traj, u_phys, gt12, m_true, w_vel)

                msg = Odometry()
                msg.header.stamp = self.ros_node.get_clock().now().to_msg()
                msg.header.frame_id = "world"
                msg.pose.pose.position.x = float(st.position[0])
                msg.pose.pose.position.y = float(st.position[1])
                msg.pose.pose.position.z = float(st.position[2])
                msg.pose.pose.orientation.x = float(st.attitude[0])
                msg.pose.pose.orientation.y = float(st.attitude[1])
                msg.pose.pose.orientation.z = float(st.attitude[2])
                msg.pose.pose.orientation.w = float(st.attitude[3])
                msg.twist.twist.linear.x = float(st.linear_velocity[0])
                msg.twist.twist.linear.y = float(st.linear_velocity[1])
                msg.twist.twist.linear.z = float(st.linear_velocity[2])
                self.gt_pub.publish(msg)

            rclpy.spin_once(self.ros_node, timeout_sec=0.0)

            # real-time pacing (headless has no render throttle; protects
            # PX4 lockstep — same mechanism as the user's run_sim.py)
            target_wall = wall_start + self.sim_time / speed
            now = time.time()
            if now < target_wall:
                time.sleep(target_wall - now)

        carb.log_warn("[datagen] shutting down")
        self.timeline.stop()
        simulation_app.close()


def main():
    app = DatagenApp(_pre_args)
    app.run()


if __name__ == "__main__":
    main()
