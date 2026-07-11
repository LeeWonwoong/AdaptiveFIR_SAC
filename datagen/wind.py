"""
datagen/wind.py — WindModel ported from the user's Issacsim-rhukf/run_sim.py.
Extended to (a) report the wind VELOCITY vector (logged in the dataset)
in addition to the drag force, and (b) accept a scenario dict (gust list +
sustained) rather than a single mode.
"""
import numpy as np


class WindModel:
    def __init__(self, scenario: dict, area=0.04, Cd=1.28, rho=1.225,
                 turb_intensity=0.0, turb_bw=2.0, seed=0):
        self.sc = scenario or {}
        self.A, self.Cd, self.rho = area, Cd, rho
        self.ti, self.tb = turb_intensity, turb_bw
        self.rng = np.random.default_rng(seed)
        self._ts = np.zeros(3)
        self._tw = np.zeros(3)   # windowed turbulence-burst OU state

    def wind_velocity(self, t, dt):
        """world-frame wind velocity vector [3] at sim time t."""
        w = np.zeros(3)
        sus = self.sc.get("sustained")
        if sus:
            _sl = sus if isinstance(sus, list) else [sus]   # 1-or-N windows
            for _su in _sl:
                _in = True
                if "start_s" in _su:             # windowed (default); missing
                    _in = _su["start_s"] <= t <= \
                        _su["start_s"] + _su["duration_s"]  # keys = legacy full
                if _in:
                    d = np.array([np.cos(_su["dir_rad"]),
                                  np.sin(_su["dir_rad"]), 0.0])
                    w += _su["speed"] * d
                    w[2] += _su["speed"] * float(_su.get("vert_ratio", 0.0))
                    break        # windows are non-overlapping; one active max
        for g in self.sc.get("gusts", []):
            if g["start_s"] <= t <= g["start_s"] + g["duration_s"]:
                d = np.array([np.cos(g["dir_rad"]), np.sin(g["dir_rad"]), 0.0])
                # half-cosine gust profile (repo)
                V = (g["speed"] / 2.0) * (1 - np.cos(
                    np.pi * (t - g["start_s"]) / g["duration_s"]))
                w += V * d
        # ── turbulence bursts (scenario['turbulence']): INSIDE each window a
        #    Dryden-like OU wind fluctuation is physically active (Isaac gap
        #    fix 2026-07-09: previously these windows were synth-only process-
        #    noise boosts and produced NOMINAL flights in Isaac).
        #    variance-preserving OU (x = a x + sqrt(1-a^2) sigma xi) so the
        #    recorded wind_sigma IS the true stationary wind-speed std [m/s].
        boost_sig = 0.0
        for tw in self.sc.get("turbulence", []):
            if tw["start_s"] <= t <= tw["start_s"] + tw["duration_s"]:
                boost_sig = max(boost_sig, float(
                    tw.get("wind_sigma", 2.5 * tw.get("boost", 0.0))))
        a = np.exp(-self.tb * dt)
        if boost_sig > 0.0:
            self._tw = a * self._tw + np.sqrt(max(1.0 - a * a, 0.0)) \
                * boost_sig * self.rng.standard_normal(3)
            # gust-factor clip (~1.5 sigma): Gaussian tails x quadratic drag
            # otherwise produce instant-crash peaks (measured 58 m/s = 105 N).
            nrm = float(np.linalg.norm(self._tw))
            cap = 1.5 * boost_sig
            if nrm > cap:
                self._tw *= cap / nrm
            w += self._tw
        else:
            self._tw *= a                      # decay after the window closes
            if np.linalg.norm(self._tw) > 0.1:
                w += self._tw
        if self.ti > 0:
            a2 = np.exp(-self.tb * dt)
            self._ts = a2 * self._ts + (1 - a2) * self.ti * self.rng.standard_normal(3)
            w += self._ts
        return w

    def drag_force(self, wind_vel):
        v = np.linalg.norm(wind_vel)
        if v < 1e-6:
            return np.zeros(3)
        return 0.5 * self.rho * v ** 2 * self.Cd * self.A * (wind_vel / v)

    def get(self, t, dt):
        w = self.wind_velocity(t, dt)
        return w, self.drag_force(w)
