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

    def wind_velocity(self, t, dt):
        """world-frame wind velocity vector [3] at sim time t."""
        w = np.zeros(3)
        sus = self.sc.get("sustained")
        if sus:
            d = np.array([np.cos(sus["dir_rad"]), np.sin(sus["dir_rad"]), 0.0])
            w += sus["speed"] * d
        for g in self.sc.get("gusts", []):
            if g["start_s"] <= t <= g["start_s"] + g["duration_s"]:
                d = np.array([np.cos(g["dir_rad"]), np.sin(g["dir_rad"]), 0.0])
                # half-cosine gust profile (repo)
                V = (g["speed"] / 2.0) * (1 - np.cos(
                    np.pi * (t - g["start_s"]) / g["duration_s"]))
                w += V * d
        if self.ti > 0:
            a = np.exp(-self.tb * dt)
            self._ts = a * self._ts + (1 - a) * self.ti * self.rng.standard_normal(3)
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
