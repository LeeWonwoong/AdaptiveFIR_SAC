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
          "nlos_burst": [], "turbulence": [],
          "heldout": bool(heldout)}

    mass_rng = cfg.heldout_mass_delta_range if heldout else cfg.mass_delta_range
    gust_rng = cfg.heldout_gust_speed_range if heldout else cfg.gust_speed_range
    nlos_bias_rng = cfg.heldout_nlos_bias_range if heldout else cfg.nlos_bias_range

    def _mass():
        return {"delta": float(rng.uniform(*mass_rng)),
                "onset_s": float(rng.uniform(*cfg.mass_onset_frac) * dur),
                "impulse_z": float(rng.uniform(*cfg.mass_impulse_z)),
                "impulse_xy": float(rng.uniform(*cfg.mass_impulse_xy)),
                "impulse_dir": float(rng.uniform(0, 2 * np.pi))}

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

    def _turbulence():
        """반복 난류: 2~4 intervals where the plant process-noise σ is boosted
        ×3~5 (the synth reads scenario['turbulence'] in its rollout)."""
        n = int(rng.integers(cfg.turb_count_range[0], cfg.turb_count_range[1] + 1))
        out, t0 = [], 0.12 * dur
        for _ in range(n):
            d = float(rng.uniform(*cfg.turb_duration_range))
            hi = 0.90 * dur - d
            if t0 > hi:
                break
            start = float(rng.uniform(t0, max(t0 + 0.1, hi)))
            out.append({"start_s": start, "duration_s": d,
                        "boost": float(rng.uniform(*cfg.turb_boost_range))})
            t0 = start + d + 1.0
        return out

    def _nlos_burst():
        """NLoS burst: one anchor's σ jumps LoS→NLoS with a positive multipath
        bias, intermittently for 2~4 s. Per-anchor (σ, bias) applied by the
        dataset noise model — the filters keep believing R=LoS."""
        n = int(rng.integers(cfg.nlos_count_range[0], cfg.nlos_count_range[1] + 1))
        n_anch = len(cfg.anchors)
        out, t0 = [], 0.12 * dur
        for _ in range(n):
            d = float(rng.uniform(*cfg.nlos_duration_range))
            hi = 0.90 * dur - d
            if t0 > hi:
                break
            start = float(rng.uniform(t0, max(t0 + 0.1, hi)))
            out.append({"anchor": int(rng.integers(0, n_anch)),
                        "start_s": start, "duration_s": d,
                        "sigma": float(cfg.nlos_sigma),
                        "bias_m": float(rng.uniform(*nlos_bias_rng))})
            t0 = start + d + 0.8
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
    elif stype == "nlos_burst":
        sc["nlos_burst"] = _nlos_burst()
    elif stype == "turbulence_burst":
        sc["turbulence"] = _turbulence()
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
    for nb in sc.get("nlos_burst", []):
        out.append((nb["start_s"], nb["start_s"] + nb["duration_s"], "nlos_burst"))
    for tb in sc.get("turbulence", []):
        out.append((tb["start_s"], tb["start_s"] + tb["duration_s"], "turbulence_burst"))
    return out
