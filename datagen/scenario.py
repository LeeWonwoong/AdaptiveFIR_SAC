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


def sample_scenario(cfg, rng: np.random.Generator, heldout: bool = False,
                    heldout_idx=None) -> dict:
    plan = getattr(cfg, "heldout_plan", None)
    _plan_row = None
    _plan_win = ()          # explicit (start_s, duration_s) windows, if planned
    _plan_extra = {}        # {"turb": ambient wind, "com": payload CoM offset}
    if heldout and plan and heldout_idx is not None:
        _plan_row = plan[heldout_idx % len(plan)]
        stype = _plan_row[0]
        if len(_plan_row) > 3 and _plan_row[3]:
            _plan_win = tuple(_plan_row[3])
        if len(_plan_row) > 4 and _plan_row[4]:
            _plan_extra = dict(_plan_row[4])
    else:
        stype = rng.choice(cfg.scenario_types, p=np.array(cfg.scenario_probs))
    pattern = (_plan_extra["pattern"] if "pattern" in _plan_extra
               else rng.choice(cfg.flight_patterns))
    if "pattern" in _plan_extra:                 # held-out rows pin the pattern
        pattern = str(_plan_extra["pattern"])
    dur = float(cfg.traj_duration_s)
    sc = {"type": str(stype), "pattern": str(pattern), "duration_s": dur,
          "seed": int(rng.integers(0, 2 ** 31 - 1)),
          "mass": None, "gusts": [], "sustained": None, "dropouts": [],
          "nlos_burst": [], "turbulence": [], "cm_regime": [],
          "heldout": bool(heldout)}
    if "turb" in _plan_extra:
        sc["ambient_turb_std"] = float(_plan_extra["turb"])

    mass_rng = cfg.heldout_mass_delta_range if heldout else cfg.mass_delta_range
    gust_rng = cfg.heldout_gust_speed_range if heldout else cfg.gust_speed_range
    nlos_bias_rng = cfg.heldout_nlos_bias_range if heldout else cfg.nlos_bias_range

    def _mass():
        _mr = (_plan_row[1], _plan_row[2]) if (_plan_row and
                                               _plan_row[0] == "mass_step") else mass_rng
        if _plan_win:                            # explicit paper window
            _on, _du = float(_plan_win[0][0]), float(_plan_win[0][1])
        else:
            _on = float(rng.uniform(*cfg.mass_onset_frac) * dur)
            _dr = getattr(cfg, "mass_window_duration_range", None)
            _du = float(rng.uniform(*_dr)) if _dr else float(dur - _on)
        _du = min(_du, dur - _on - 2.0)          # release inside the traj
        if "com" in _plan_extra:
            _cm = float(_plan_extra["com"])
        else:
            _co = getattr(cfg, "mass_com_offset_range", None)
            _cm = float(rng.uniform(*_co)) if _co else 0.0
        _cdir = float(rng.uniform(0, 2 * np.pi))
        return {"delta": float(rng.uniform(*_mr)),
                "duration_s": _du,
                "onset_s": _on,
                # payload is NOT attached at the CoM: offset magnitude [m] and
                # direction in the body xy-plane (parasitic torque -> x,y error)
                "com_offset": _cm,
                "com_dir": _cdir,
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
            b = float(rng.uniform(*cfg.turb_boost_range))
            out.append({"start_s": start, "duration_s": d, "boost": b,
                        "wind_sigma": float(
                            getattr(cfg, "turb_wind_sigma_per_boost", 2.5) * b)})
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

    def _cm_regime():
        """alternating calm↔dynamic flight segments (start calm). Dynamic
        segments = attitude-active maneuvering → the tag-side common-mode bias
        switches to its fast/large OU regime there. Recorded so the measurement
        layer (dataset) and the trajectory generator (synth) both read them."""
        segs, t, mode = [], 0.0, "calm"
        while t < dur - 0.5:
            lo, hi = (cfg.cm_calm_dur_range if mode == "calm"
                      else cfg.cm_dyn_dur_range)
            d = min(float(rng.uniform(lo, hi)), dur - t)
            segs.append({"start_s": float(t), "duration_s": float(d), "mode": mode})
            t += d
            mode = "dynamic" if mode == "calm" else "calm"
        return segs

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
        _sd = float(rng.uniform(*getattr(cfg, "sustained_duration_range", (8.0, 15.0))))
        _s0 = float(rng.uniform(*getattr(cfg, "sustained_onset_frac", (0.20, 0.50))) * dur)
        _s0 = min(_s0, max(0.0, 0.9 * dur - _sd))          # keep window inside traj
        _sr = (_plan_row[1], _plan_row[2]) if (_plan_row and
                _plan_row[0] == "sustained_wind") else cfg.sustained_speed_range
        _vr = getattr(cfg, "wind_vertical_ratio", None)
        # UPDRAFT ONLY (gate finding 2026-07-10, n=5 trajs): an updraft
        # component reliably moves the in-window z-optimal horizon down to
        # N~6 (z RMSE 0.110-0.129 @N6 vs 0.143-0.188 @N10), while a
        # DOWNdraft produces NO z model error in this sim (z stays optimal
        # at N=10-14). Scenario is therefore defined as wind with an updraft
        # component (thermal / terrain-induced flow) — stated in the paper.
        _vert = float(rng.uniform(*_vr)) if _vr else 0.0
        _nw = int(getattr(cfg, "wind_n_windows", 1))
        _spd = float(rng.uniform(*_sr)); _dir = float(rng.uniform(0, 2 * np.pi))
        if _plan_win:                              # explicit paper windows
            sc["sustained"] = [
                {"speed": _spd, "vert_ratio": _vert, "dir_rad": _dir,
                 "start_s": float(_w[0]), "duration_s": float(_w[1])}
                for _w in _plan_win]
        elif _nw <= 1:
            sc["sustained"] = {"speed": _spd, "vert_ratio": _vert,
                               "dir_rad": _dir,
                               "start_s": _s0, "duration_s": _sd}
        else:
            # N non-overlapping windows spread across the trajectory, each with
            # the SAME wind vector, gap >= 6 s. Slots split [0.10, 0.92]*dur.
            _wins = []
            _lo, _hi = 0.10 * dur, 0.92 * dur
            _slot = (_hi - _lo) / _nw
            for _k in range(_nw):
                _d = float(rng.uniform(*getattr(cfg, "sustained_duration_range",
                                                (8.0, 15.0))))
                _d = min(_d, _slot - 6.0)                 # leave >=6 s gap
                _base = _lo + _k * _slot
                _st = float(rng.uniform(_base, _base + _slot - _d))
                _wins.append({"speed": _spd, "vert_ratio": _vert,
                              "dir_rad": _dir,
                              "start_s": _st, "duration_s": _d})
            sc["sustained"] = _wins        # LIST of windows
    elif stype == "anchor_dropout":
        sc["dropouts"] = _dropouts()
    elif stype == "nlos_burst":
        sc["nlos_burst"] = _nlos_burst()
    elif stype == "turbulence_burst":
        sc["turbulence"] = _turbulence()
    elif stype == "tag_commonmode":
        sc["cm_regime"] = _cm_regime()
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
        _m = sc["mass"]
        _end = _m["onset_s"] + _m["duration_s"] if "duration_s" in _m \
            else sc["duration_s"]                 # legacy: persists to the end
        out.append((_m["onset_s"], _end, "mass_step"))
    for g in sc.get("gusts", []):
        out.append((g["start_s"], g["start_s"] + g["duration_s"], "gust"))
    if sc.get("sustained"):
        su = sc["sustained"]
        _sl = su if isinstance(su, list) else [su]     # 1-or-N windows
        for _w in _sl:
            if "start_s" in _w:  # windowed (2026-07-09+): anchor at the TRUE onset
                out.append((_w["start_s"], _w["start_s"] + _w["duration_s"],
                            "sustained_wind"))
            else:                # legacy full-trajectory sustained
                out.append((0.0, sc["duration_s"], "sustained_wind"))
    for dp in sc.get("dropouts", []):
        out.append((dp["start_s"], dp["start_s"] + dp["duration_s"], "anchor_dropout"))
    for nb in sc.get("nlos_burst", []):
        out.append((nb["start_s"], nb["start_s"] + nb["duration_s"], "nlos_burst"))
    for tb in sc.get("turbulence", []):
        out.append((tb["start_s"], tb["start_s"] + tb["duration_s"], "turbulence_burst"))
    for seg in sc.get("cm_regime", []):
        if seg.get("mode") == "dynamic":
            out.append((seg["start_s"], seg["start_s"] + seg["duration_s"],
                        "tag_commonmode"))
    return out
