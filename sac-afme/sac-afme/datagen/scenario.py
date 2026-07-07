"""
datagen/scenario.py — scenario sampler shared by the Tier-0 synthetic
generator (rlenv/synth.py) and the Isaac Sim datagen (run_datagen.py).
Pure python — no Isaac imports.

Each trajectory gets a scenario dict (fully recorded in meta.json):
  {type, pattern, duration_s, seed,
   mass:  {delta, onset_s}                        (mass_step / mixed)
   gusts: [{speed, dir_rad, start_s, duration_s}] (gust / mixed)
   sustained: {speed, dir_rad}                    (sustained_wind / mixed)
   dropouts: [{anchors:[...], start_s, duration_s}] (anchor_dropout / mixed)}
"""
import numpy as np


def sample_scenario(cfg, rng: np.random.Generator, heldout: bool = False) -> dict:
    stype = rng.choice(cfg.scenario_types, p=np.array(cfg.scenario_probs))
    pattern = rng.choice(cfg.flight_patterns)
    dur = float(cfg.traj_duration_s)
    sc = {"type": str(stype), "pattern": str(pattern), "duration_s": dur,
          "seed": int(rng.integers(0, 2 ** 31 - 1)),
          "mass": None, "gusts": [], "sustained": None, "dropouts": [],
          "heldout": bool(heldout)}

    mass_rng = cfg.heldout_mass_delta_range if heldout else cfg.mass_delta_range
    gust_rng = cfg.heldout_gust_speed_range if heldout else cfg.gust_speed_range

    def _mass():
        return {"delta": float(rng.uniform(*mass_rng)),
                "onset_s": float(rng.uniform(*cfg.mass_onset_frac) * dur)}

    def _dropouts():
        n = int(rng.integers(cfg.dropout_count_range[0],
                             cfg.dropout_count_range[1] + 1))
        n_anch = len(cfg.anchors)
        out, t0 = [], 0.15 * dur
        for _ in range(n):
            d = float(rng.uniform(*cfg.dropout_duration_range))
            start = float(rng.uniform(t0, max(t0 + 0.1, 0.85 * dur - d)))
            k = int(rng.integers(1, cfg.dropout_max_anchors + 1))
            anchors = sorted(int(a) for a in
                             rng.choice(n_anch, size=min(k, n_anch), replace=False))
            out.append({"anchors": anchors, "start_s": start, "duration_s": d})
            t0 = start + d + 1.0
        return out

    def _gusts():
        n = int(rng.integers(cfg.gust_count_range[0], cfg.gust_count_range[1] + 1))
        out, t0 = [], 0.15 * dur
        for _ in range(n):
            d = float(rng.uniform(*cfg.gust_duration_range))
            start = float(rng.uniform(t0, max(t0 + 0.1, 0.85 * dur - d)))
            out.append({"speed": float(rng.uniform(*gust_rng)),
                        "dir_rad": float(rng.uniform(0, 2 * np.pi)),
                        "start_s": start, "duration_s": d})
            t0 = start + d + 1.0
        return out

    if stype == "mass_step":
        sc["mass"] = _mass()
    elif stype == "gust":
        sc["gusts"] = _gusts()
    elif stype == "sustained_wind":
        sc["sustained"] = {"speed": float(rng.uniform(*cfg.sustained_speed_range)),
                           "dir_rad": float(rng.uniform(0, 2 * np.pi))}
    elif stype == "anchor_dropout":
        sc["dropouts"] = _dropouts()
    elif stype == "mixed":
        sc["mass"] = _mass()
        sc["gusts"] = _gusts()
        if rng.random() < 0.5:
            sc["dropouts"] = _dropouts()
    return sc


def disturbance_intervals(sc: dict):
    """[(t0, t1, label), ...] for figure shading / segment metrics."""
    out = []
    if sc.get("mass"):
        out.append((sc["mass"]["onset_s"], sc["duration_s"], "mass_step"))
    for g in sc.get("gusts", []):
        out.append((g["start_s"], g["start_s"] + g["duration_s"], "gust"))
    if sc.get("sustained"):
        out.append((0.0, sc["duration_s"], "sustained_wind"))
    for dp in sc.get("dropouts", []):
        out.append((dp["start_s"], dp["start_s"] + dp["duration_s"], "anchor_dropout"))
    return out
