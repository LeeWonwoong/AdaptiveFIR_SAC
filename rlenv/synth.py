"""
rlenv/synth.py — Tier-0 synthetic trajectory generator
=======================================================
Produces .npz files with EXACTLY the same schema as the Isaac Sim datagen,
so the whole pipeline (dataset → env → SAC → evaluate) can be developed and
smoke-tested without Isaac Sim, then re-pointed at real sim data.

Plant : full 12-state nonlinear quadrotor (TRUE mass m_true(t), wind drag
        acceleration, small process noise), sub-stepped at dt_int = dt/4.
Control: geometric-ish PD using NOMINAL mass (no integrator) → mass steps and
        wind create realistic tracking + model-mismatch signatures.
Filter model mismatch comes from: m_true != m_nominal, wind force absent from
the filter model, process noise. (decision #5: filter keeps DI-FME model.)

Schema per trajectory  traj_XXXX.npz :
  t        [T]      u [T,4] (thrust N, torques Nm — the model's physical input)
  gt       [T,12]   [p,v,eta,omega]
  m_true   [T]      wind [T,3]
meta_XXXX.json : scenario dict + model constants + anchors.
"""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config                             # noqa: E402
from datagen.scenario import sample_scenario          # noqa: E402
from datagen.wind import WindModel                    # noqa: E402


# ─────────────────────────────── plant dynamics (numpy, single traj)
def _R(eta):
    ph, th, ps = eta
    cp, sp = np.cos(ph), np.sin(ph)
    ct, st = np.cos(th), np.sin(th)
    cy, sy = np.cos(ps), np.sin(ps)
    return np.array([
        [cy * ct, cy * st * sp - sy * cp, cy * st * cp + sy * sp],
        [sy * ct, sy * st * sp + sy * 0 + cy * cp, sy * st * cp - cy * sp],
        [-st, ct * sp, ct * cp]])


def _Weul(eta):
    ph, th, _ = eta
    cp, sp = np.cos(ph), np.sin(ph)
    tt, sec = np.tan(th), 1.0 / max(np.cos(th), 0.35)
    return np.array([[1, sp * tt, cp * tt],
                     [0, cp, -sp],
                     [0, sp * sec, cp * sec]])


def _plant_step(s, u, dt, m, J, g_vec, wind_acc, wn):
    p, v, eta, om = s[0:3], s[3:6], s[6:9], s[9:12]
    R = _R(eta)
    acc = g_vec + R @ np.array([0, 0, u[0]]) / m + wind_acc + wn[:3]
    om_dot = (u[1:4] - np.cross(om, J * om)) / J + wn[3:6]
    return np.concatenate([p + v * dt, v + acc * dt,
                           eta + (_Weul(eta) @ om) * dt, om + om_dot * dt])


# ─────────────────────────────── reference patterns (position, velocity, yaw)
def _ref(pattern, t, c=np.array([5.0, 5.0, 1.5]), R0=3.0, w=0.675):
    # w 0.50 -> 0.675 (2026-07-13, final): 1.35x faster on the SAME radius.
    # Centripetal demand a_h = R0*w^2 = 1.37 m/s^2 -> commanded bank ~7.9 deg.
    # MEASURED at 1.2x (gate v8): the heavier payload vehicle still sagged to
    # ~2.3 deg inside the window, so 1.2x was not enough for an x,y signature;
    # 1.35x pushes the commanded bank ~26% higher while cruise speed stays a
    # modest 2.0 m/s -- an agile but plausible indoor flight.
    # WHY THIS MATTERS: a payload mass error acts ALONG THE THRUST AXIS, so it
    # only reaches x,y through the tilt: a_xy = |dT/m| * sin(theta). Measured
    # |dT/m| = 5.18 m/s^2. In the measured payload window the vehicle banked
    # only 1-2 deg (a heavier vehicle tracks the reference more sluggishly and
    # banks LESS), so the horizontal share was ~0.12 m/s^2 -- BELOW the
    # 0.18 m/s^2 nominal residual. That is the whole reason the payload looked
    # like a pure z-axis disturbance. At 6.3 deg the share is 0.57 m/s^2,
    # ~3x the residual, and the payload shows up on all three axes.
    if pattern == "hover":
        return c, np.zeros(3), 0.0
    if pattern == "circle":
        p = c + np.array([R0 * np.cos(w * t), R0 * np.sin(w * t), 0])
        v = np.array([-R0 * w * np.sin(w * t), R0 * w * np.cos(w * t), 0])
        return p, v, np.arctan2(v[1], v[0])
    if pattern == "figure8":
        # figure-8 WITH altitude variation (z-sweep coupled to the lobe) so the
        # non-coplanar anchors get vertical excitation — helps velocity/attitude
        # observability under UWB-only. Planar 8 (z=0) left as "figure8_flat".
        z = 0.8 * np.sin(w * t)
        vz = 0.8 * w * np.cos(w * t)
        p = c + np.array([R0 * np.sin(w * t), 0.5 * R0 * np.sin(2 * w * t), z])
        v = np.array([R0 * w * np.cos(w * t), R0 * w * np.cos(2 * w * t), vz])
        return p, v, np.arctan2(v[1], v[0])
    if pattern == "figure8_flat":
        p = c + np.array([R0 * np.sin(w * t), 0.5 * R0 * np.sin(2 * w * t), 0])
        v = np.array([R0 * w * np.cos(w * t), R0 * w * np.cos(2 * w * t), 0])
        return p, v, np.arctan2(v[1], v[0])
    if pattern == "helical":
        # 3D helical climb/descent: horizontal circle (excites roll/pitch via
        # centripetal tilt) + sustained vertical motion (activates z-observation
        # against the non-coplanar anchors). Observability of attitude through
        # UWB position measurements stays alive at every instant. PRIMARY.
        zc = 1.5 + 1.0 * np.sin(0.18 * w * t)           # slow vertical sweep
        vz = 1.0 * 0.18 * w * np.cos(0.18 * w * t)
        p = c + np.array([R0 * np.cos(w * t), R0 * np.sin(w * t), zc - c[2]])
        v = np.array([-R0 * w * np.sin(w * t), R0 * w * np.cos(w * t), vz])
        return p, v, np.arctan2(v[1], v[0])
    if pattern == "figure8_3d":
        # inclined figure-8: lateral 8 + coupled altitude oscillation. Frequent
        # heading reversals give rich attitude excitation; the tilted plane
        # keeps z varying against non-coplanar anchors. SECONDARY.
        z = 0.8 * np.sin(w * t)                          # altitude coupled to lobe
        vz = 0.8 * w * np.cos(w * t)
        p = c + np.array([R0 * np.sin(w * t), 0.5 * R0 * np.sin(2 * w * t), z])
        v = np.array([R0 * w * np.cos(w * t), R0 * w * np.cos(2 * w * t), vz])
        return p, v, np.arctan2(v[1], v[0])
    if pattern == "waypoint":
        wps = c + np.array([[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]])
        seg, per = 6.0, 24.0
        tm = t % per
        i = int(tm // seg)
        f = (tm % seg) / seg
        a, b = wps[i], wps[(i + 1) % 4]
        return a + (b - a) * f, (b - a) / seg, np.arctan2((b - a)[1], (b - a)[0])
    # aggressive: fast fig-8 + altitude bob
    wa = 1.1
    p = c + np.array([R0 * np.sin(wa * t), 0.5 * R0 * np.sin(2 * wa * t),
                      0.5 * np.sin(0.8 * t)])
    v = np.array([R0 * wa * np.cos(wa * t), R0 * wa * np.cos(2 * wa * t),
                  0.4 * np.cos(0.8 * t)])
    return p, v, np.arctan2(v[1], v[0])


def _cm_gain(t, dyn_intervals, ramp=0.4):
    """smooth attitude-activity gain g(t)∈[0,1]: 1 inside a dynamic segment
    (cosine-ramped at both ends so position/attitude stay continuous), 0 in calm."""
    for (a, b) in dyn_intervals:
        if a <= t <= b:
            up = 0.5 - 0.5 * np.cos(np.pi * min((t - a) / ramp, 1.0))
            dn = 0.5 - 0.5 * np.cos(np.pi * min((b - t) / ramp, 1.0))
            return min(up, dn)
    return 0.0


def _cm_ref(t, gain, c=np.array([5.0, 5.0, 1.5]), R0=3.0, w=1.0):
    """calm = hover at c; dynamic = fast figure-8 + altitude bob, amplitude
    scaled by `gain` → attitude activity turns on/off with the regime."""
    osc = np.array([R0 * np.sin(w * t), 0.5 * R0 * np.sin(2 * w * t),
                    0.4 * np.sin(0.8 * t)])
    vosc = np.array([R0 * w * np.cos(w * t), R0 * w * np.cos(2 * w * t),
                     0.4 * 0.8 * np.cos(0.8 * t)])
    p = c + gain * osc
    v = gain * vosc
    return p, v, float(np.arctan2(v[1], v[0])) if gain > 1e-3 else 0.0


# ─────────────────────────────── PD controller (nominal mass — no integrator)
def _controller(s, p_ref, v_ref, yaw_ref, m_nom, J, g):
    p, v, eta, om = s[0:3], s[3:6], s[6:9], s[9:12]
    kp, kd = 4.0, 3.2
    a_des = kp * (p_ref - p) + kd * (v_ref - v)
    a_des = np.clip(a_des, -6, 6)
    f_w = m_nom * (a_des + np.array([0, 0, g]))                 # desired world force
    fz = max(f_w[2], 0.2 * m_nom * g)
    T = np.linalg.norm(f_w)
    cy, sy = np.cos(yaw_ref), np.sin(yaw_ref)
    # small-angle attitude from desired lateral force
    th_des = np.clip((cy * f_w[0] + sy * f_w[1]) / fz, -0.45, 0.45)
    ph_des = np.clip((sy * f_w[0] - cy * f_w[1]) / fz, -0.45, 0.45)
    eta_des = np.array([ph_des, th_des, yaw_ref])
    e = eta_des - eta
    e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi
    kp_a, kd_a = 22.0, 6.0
    tau = J * (kp_a * e - kd_a * om) + np.cross(om, J * om)
    tau = np.clip(tau, -1.2, 1.2)
    return np.array([np.clip(T, 0, 3.0 * m_nom * g), *tau])


# ─────────────────────────────── trajectory rollout
def generate_traj(cfg: Config, scenario: dict, rng: np.random.Generator):
    dt, sub = cfg.dt, 4
    dti = dt / sub
    T = int(scenario["duration_s"] / dt)
    J = np.array([cfg.Ixx, cfg.Iyy, cfg.Izz])
    g_vec = np.array([0, 0, -cfg.g])
    wind = WindModel(scenario, seed=scenario["seed"])

    if scenario.get("type") == "tag_commonmode":
        p0 = np.array([5.0, 5.0, 1.5])                       # calm hover start
    else:
        p0, _, _ = _ref(scenario["pattern"], 0.0)
    s = np.zeros(12)
    s[0:3] = p0 + rng.normal(0, 0.05, 3)
    m_nom = cfg.mass_nominal

    t_arr = np.zeros(T)
    u_arr = np.zeros((T, 4))
    gt = np.zeros((T, 12))
    mt = np.zeros(T)
    wv = np.zeros((T, 3))

    mass_onset_k = None
    if scenario.get("mass"):
        mass_onset_k = int(scenario["mass"]["onset_s"] / dt)
    cm_dyn = [(seg["start_s"], seg["start_s"] + seg["duration_s"])
              for seg in scenario.get("cm_regime", []) if seg.get("mode") == "dynamic"]
    is_cm = scenario.get("type") == "tag_commonmode"

    for k in range(T):
        t = k * dt
        m_true = m_nom
        if scenario.get("mass") and t >= scenario["mass"]["onset_s"]:
            m_true = m_nom * (1.0 + scenario["mass"]["delta"])
        w_vel, w_force = wind.get(t, dt)
        wind_acc = w_force / m_true

        if is_cm:
            p_ref, v_ref, yaw_ref = _cm_ref(t, _cm_gain(t, cm_dyn))
        else:
            p_ref, v_ref, yaw_ref = _ref(scenario["pattern"], t)
        u = _controller(s, p_ref, v_ref, yaw_ref, m_nom, J, cfg.g)

        # turbulence burst: boost the TRUE process-noise σ during the interval
        # (the filters keep believing nominal q0 → fixed-Q KF lags).
        q_boost = 1.0
        for tb in scenario.get("turbulence", []):
            if tb["start_s"] <= t <= tb["start_s"] + tb["duration_s"]:
                q_boost = max(q_boost, tb["boost"])
        acc_std = cfg.proc_acc_std * q_boost
        gyro_std = cfg.proc_gyro_std * q_boost

        # payload-coupling instant: inject a velocity impulse (z-sink + lateral),
        # exactly the UIFM-SLAC Scenario-2 signature (sudden CoG change).
        if mass_onset_k is not None and k == mass_onset_k:
            mm = scenario["mass"]
            s[5] -= mm.get("impulse_z", 0.0)                      # downward z velocity
            ang = mm.get("impulse_dir", 0.0)
            s[3] += mm.get("impulse_xy", 0.0) * np.cos(ang)       # lateral x
            s[4] += mm.get("impulse_xy", 0.0) * np.sin(ang)       # lateral y

        # log BEFORE stepping: (gt_k, u_k) with u_k acting k -> k+1
        t_arr[k], u_arr[k], gt[k], mt[k], wv[k] = t, u, s, m_true, w_vel

        for _ in range(sub):
            wn = np.concatenate([rng.normal(0, acc_std, 3),
                                 rng.normal(0, gyro_std, 3)])
            s = _plant_step(s, u, dti, m_true, J, g_vec, wind_acc, wn)
        s[6:8] = np.clip(s[6:8], -1.1, 1.1)

    return dict(t=t_arr, u=u_arr, gt=gt, m_true=mt, wind=wv)


def generate_dataset(cfg: Config, out_root=None, n_train=None, n_heldout=None,
                     verbose=True):
    out_root = out_root or cfg.data_dir
    n_train = n_train if n_train is not None else cfg.n_train_traj
    n_heldout = n_heldout if n_heldout is not None else cfg.n_heldout_traj
    rng = np.random.default_rng(cfg.seed)
    consts = dict(mass_nominal=cfg.mass_nominal, g=cfg.g, dt=cfg.dt,
                  Ixx=cfg.Ixx, Iyy=cfg.Iyy, Izz=cfg.Izz,
                  anchors=[list(a) for a in cfg.anchors], source="synthetic_tier0")
    for split, n, ho in (("train", n_train, False), ("heldout", n_heldout, True)):
        d = os.path.join(out_root, split)
        os.makedirs(d, exist_ok=True)
        for i in range(n):
            sc = sample_scenario(cfg, rng, heldout=ho, train_idx=(None if ho else i))
            arrs = generate_traj(cfg, sc, np.random.default_rng(sc["seed"]))
            np.savez_compressed(os.path.join(d, f"traj_{i:04d}.npz"),
                                **{k: v.astype(np.float32) for k, v in arrs.items()})
            with open(os.path.join(d, f"meta_{i:04d}.json"), "w") as f:
                json.dump({"scenario": sc, "consts": consts}, f, indent=1)
            if verbose and (i + 1) % 20 == 0:
                print(f"[synth] {split} {i + 1}/{n}")
    if verbose:
        print(f"[synth] done → {out_root}/train, {out_root}/heldout")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_train", type=int, default=None)
    ap.add_argument("--n_heldout", type=int, default=None)
    args = ap.parse_args()
    c = Config()
    generate_dataset(c, args.out, args.n_train, args.n_heldout)
