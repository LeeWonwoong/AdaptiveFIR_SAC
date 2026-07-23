"""
Shared helpers for the paper-analysis tools (tools/*.py).

Conventions fixed here so every tool reports the SAME numbers as the paper:

  * evaluation window : 2 s -- 40 s  (EVAL_T0/EVAL_T1)
                        start skips the growing-window ramp of the fixed-N
                        baselines, end matches the time-series figures.
  * metric            : 3-D position RMSE, pooled over the three flight
                        patterns of a scenario.
  * baselines         : EKF / UKF  Q = 3e-3 I12 (each filter's own nominal
                        grid optimum), UKF alpha=0.5 beta=2 kappa=0 and
                        R_uwb x 0.85; FME fixed N = 10, lambda = 1.
  * noise             : noise_seed = 1234 + seed*101, identical stream for
                        every filter in a given seed.

Run every tool from the repository root, e.g.

    python3 tools/eval_seeds.py --data_dir data_isaac_v12 \
        --ckpt results/v12_50k/ckpt.pt --seeds 13,15,18
"""
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------------------------------------------------- defaults
EVAL_T0 = 2.0            # s -- start of the RMSE window
EVAL_T1 = 40.0           # s -- end of the RMSE window (== figure x-limit)
# Noise seeds -- THIS module is the single source of truth, exactly like the
# baseline (Q,R) tuning below. SEED is the default noise seed of the
# single-seed tools (make_figs / make_table); the SEED_* overrides give one
# method its OWN measurement-noise realisation (None = use SEED). The CLI
# flags (--seed / --seed-ekf ...) still win over these constants.
SEED = 12
SEED_EKF = 1         # per-method overrides; None -> SEED (paired stream)
SEED_UKF = 2
SEED_FME = 2
SEED_AFME = 2
# Baseline tuning -- THIS module is the single source of truth; evaluate.py
# imports these constants so the online-play run and every tools/ analysis use
# the SAME baseline configuration.
Q_EKF = 1.5e-3           # nominal grid optimum, see tools/sweep_tuning.py
Q_UKF = 5e-3             # nominal grid optimum (same value, independently found)
Q_EKF_DIST = 5e-3        # EKF process noise under disturbance (wind/payload)
Q_UKF_DIST = 6e-3        # UKF process noise under disturbance (wind/payload)

R_EKF = 1.2             # EKF meas-noise scale x meas_sigma (nominal)
R_UKF = 0.9              # UKF meas-noise scale x meas_sigma (nominal)
R_EKF_DIST = 1.2        # EKF meas-noise scale under disturbance (wind/payload)
R_UKF_DIST = 0.9         # UKF meas-noise scale under disturbance (wind/payload)

UKF_ALPHA = 0.3         # standard range 0 < alpha <= 1
UKF_BETA = 2.0
UKF_KAPPA = 0.0
UKF_R_UWB_SCALE = 1.2   # UKF's own grid optimum on the UWB block of R
FME_N = 11               # fixed-horizon baseline (nominal grid optimum)
FME_LAM = 0.9

# disturbance windows of the held-out set, (start_s, end_s)
WIND_WINDOWS = [(6.0, 16.0), (26.0, 36.0)]
PAYLOAD_WINDOWS = [(15.0, 33.0)]

COLORS = {"EKF": "#d62728", "UKF": "#ff7f0e",
          "FME": "#1f77b4", "AFME": "#2ca02c"}
METHODS = ["EKF", "UKF", "FME", "AFME"]
# Paper display names -- internal keys (JSON, res dicts, SEED_* names) stay
# "AFME" for compatibility; only rendered strings use the paper name.
DISPLAY = {"EKF": "EKF", "UKF": "UKF", "FME": "FME", "AFME": "DRLA-FME"}

# Figure pattern selection -- the time-series figures (make_figs) normally
# draw the RMS across the three flight patterns of a scenario. Set this to
# one pattern name ("helical" / "figure8" / "waypoint") to draw that single
# representative trajectory instead; None keeps the all-pattern RMS.
# The CLI flag (--fig-pattern) still wins over this constant.
FIG_PATTERN = "None"


# ------------------------------------------------------------------ set-up
def load_cfg(data_dir):
    """parse_cli() reads sys.argv, so hand it a clean argv."""
    from config import Config, parse_cli
    saved = sys.argv
    sys.argv = ["_", "--data_dir", data_dir]
    try:
        cfg = parse_cli(Config())
    finally:
        sys.argv = saved
    return cfg


def scenario_index(cfg):
    """{'nominal': [...], 'wind': [...], 'payload': [...]} from heldout_plan."""
    alias = {"nominal": "nominal",
             "sustained_wind": "wind",
             "mass_step": "payload"}
    out = {}
    for i, row in enumerate(cfg.heldout_plan):
        key = alias.get(row[0], row[0])
        out.setdefault(key, []).append(i)
    return out


def select_pattern(cfg, idx, pattern):
    """Restrict scenario indices `idx` to one named flight pattern.

    `pattern` is the heldout_plan extra["pattern"] name ("helical",
    "figure8", "waypoint"); falsy -> `idx` unchanged (all-pattern RMS).
    """
    if not pattern:
        return list(idx)
    sel = [i for i in idx
           if cfg.heldout_plan[i][4].get("pattern") == pattern]
    if not sel:
        have = sorted({cfg.heldout_plan[i][4].get("pattern") for i in idx})
        raise ValueError(f"pattern {pattern!r} not in scenario "
                         f"(available: {', '.join(have)})")
    return sel


def make_dataset(cfg, dev="cpu"):
    from rlenv.dataset import TrajDataset
    ds = TrajDataset(cfg, "heldout", dev)
    return ds, ds.n


def make_runner(cfg, ds, dev, seed):
    from evaluate import Runner
    return Runner(cfg, ds, dev, noise_seed=1234 + int(seed) * 101)


def load_agent(cfg, ckpt, dev="cpu"):
    from rl.sac import SACAgent
    agent = SACAgent(cfg, obs_dim=cfg.obs_dim, device=dev)
    agent.load(ckpt)
    return agent


# ----------------------------------------------------------------- filters
def make_ekf(cfg, dev, M, q=Q_EKF, r_scale=R_EKF):
    from filter.baselines import EKF
    # r_diag is a std; the filter squares it internally (R = diag(sig**2)).
    r = [s * r_scale for s in cfg.meas_sigma]
    return EKF(cfg, dev, M, q_diag=[q] * 12, r_diag=r)


def make_ukf(cfg, dev, M, q=Q_UKF, r_scale=R_UKF, alpha=UKF_ALPHA,
             beta=UKF_BETA, kappa=UKF_KAPPA, r_uwb_scale=UKF_R_UWB_SCALE):
    from filter.baselines import UKF
    r = [s * r_scale for s in cfg.meas_sigma]
    f = UKF(cfg, dev, M, q_diag=[q] * 12, r_diag=r,
            alpha=alpha, beta=beta, kappa=kappa)
    if r_uwb_scale != 1.0:
        # extra UWB-block trust factor, applied on top of the overall r_scale
        rd = torch.tensor([(cfg.meas_sigma[i] * r_scale) ** 2 for i in range(10)])
        rd[:4] *= r_uwb_scale
        f.R = torch.diag(rd).float().to(dev)
    return f


def make_fme(cfg, dev, M, N=FME_N, lam=FME_LAM):
    from filter.baselines import FixedFME
    return FixedFME(cfg, dev, M, N=N, lam=lam)


def make_afme(cfg, dev, M):
    from filter.wfme import WeightedFME
    return WeightedFME(cfg, dev, M)


def run_all(run, cfg, dev, M, agent=None,
            q_ekf=Q_EKF, q_ukf=Q_UKF, r_ekf=R_EKF, r_ukf=R_UKF,
            fme_N=FME_N, skip=(),
            q_ekf_dist=None, q_ukf_dist=None,
            r_ekf_dist=None, r_ukf_dist=None):
    """Run every filter on one noise realisation.

    Returns {name: dict(evec=[M,T,3], N=[M,T] or None, lam=[M,T] or None)}.

    `q_ekf`/`q_ukf` and `r_ekf`/`r_ukf` are the process- and measurement-noise
    levels used for the nominal (near-linear) flight regime. If ANY of the
    `*_dist` levels are given the EKF / UKF are ALSO run with the disturbance
    regime (each `*_dist` falling back to its nominal value when None) and the
    result is stored under an extra `evec_dist` key on the same entry, so a
    caller can score the nominal scenario with `evec` and the disturbance
    scenarios with `evec_dist`. When all `*_dist` are None (default) no extra
    rollouts happen and behaviour is unchanged.
    """
    res = {}
    if agent is not None and "AFME" not in skip:
        _, na, la, ev = run.run(make_afme(cfg, dev, M),
                                policy=lambda o: agent.act(o, deterministic=True))
        res["AFME"] = dict(evec=ev,
                           N=_np(na), lam=_np(la))
    if "EKF" not in skip:
        _, _, _, ev = run.run(make_ekf(cfg, dev, M, q_ekf, r_ekf))
        res["EKF"] = dict(evec=ev, N=None, lam=None)
        if q_ekf_dist is not None or r_ekf_dist is not None:
            qd = q_ekf if q_ekf_dist is None else q_ekf_dist
            rd = r_ekf if r_ekf_dist is None else r_ekf_dist
            _, _, _, evd = run.run(make_ekf(cfg, dev, M, qd, rd))
            res["EKF"]["evec_dist"] = evd
    if "UKF" not in skip:
        _, _, _, ev = run.run(make_ukf(cfg, dev, M, q_ukf, r_ukf))
        res["UKF"] = dict(evec=ev, N=None, lam=None)
        if q_ukf_dist is not None or r_ukf_dist is not None:
            qd = q_ukf if q_ukf_dist is None else q_ukf_dist
            rd = r_ukf if r_ukf_dist is None else r_ukf_dist
            _, _, _, evd = run.run(make_ukf(cfg, dev, M, qd, rd))
            res["UKF"]["evec_dist"] = evd
    if "FME" not in skip:
        _, _, _, ev = run.run(make_fme(cfg, dev, M, fme_N))
        res["FME"] = dict(evec=ev, N=None, lam=None)
    return res


def default_method_seeds():
    """The SEED_* constants above as {method: seed}, overrides only."""
    return {m: s for m, s in
            [("EKF", SEED_EKF), ("UKF", SEED_UKF),
             ("FME", SEED_FME), ("AFME", SEED_AFME)] if s is not None}


def run_all_seeded(cfg, ds, dev, M, agent=None, seed=SEED, method_seeds=None,
                   **kw):
    """run_all(), but each filter may use its OWN noise seed.

    `method_seeds` maps {"EKF": s, ...}; methods not listed fall back to
    `seed`.  When None (default) the module-level SEED_* constants are used.
    Methods sharing a seed share ONE Runner, i.e. the same measurement-noise
    realisation (the paired-comparison convention is kept); methods with
    distinct seeds see distinct noise streams.
    """
    if method_seeds is None:
        method_seeds = default_method_seeds()
    groups = {}
    for m in METHODS:
        groups.setdefault(int(method_seeds.get(m, seed)), []).append(m)
    res = {}
    for sd in sorted(groups):
        run = make_runner(cfg, ds, dev, sd)
        skip = tuple(m for m in METHODS if m not in groups[sd])
        res.update(run_all(run, cfg, dev, M, agent, skip=skip, **kw))
    return res


def _np(x):
    if x is None:
        return None
    return x.numpy() if torch.is_tensor(x) else np.asarray(x)


# ----------------------------------------------------------------- metrics
def eval_slice(cfg, T=None):
    k0 = int(EVAL_T0 / cfg.dt)
    k1 = int(EVAL_T1 / cfg.dt)
    if T is not None:
        k1 = min(k1, T)
    return slice(k0, k1)


def rmse3(evec, idx, sl):
    """3-D position RMSE over trajectories `idx` and time slice `sl`."""
    e = np.asarray(evec)[idx][:, sl, :]
    return float(np.sqrt((e ** 2).sum(-1).mean()))


def err_norm(evec):
    """[M,T] per-step 3-D error magnitude."""
    return np.linalg.norm(np.asarray(evec), axis=2)


def moving_rms(x, cfg, win_s):
    """Centred moving RMS with a `win_s` second window."""
    w = max(1, int(round(win_s / cfg.dt)))
    k = np.ones(w) / w
    return np.sqrt(np.convolve(np.asarray(x) ** 2, k, mode="same"))


def moving_avg(x, cfg, win_s):
    """Centred moving average (for N_k / lambda_k traces)."""
    w = max(1, int(round(win_s / cfg.dt)))
    k = np.ones(w) / w
    return np.convolve(np.asarray(x, dtype=float), k, mode="same")


def pattern_rms(evec, idx):
    """RMS across the flight patterns of one scenario -> [T]."""
    en = err_norm(evec)[idx]
    return np.sqrt((en ** 2).mean(0))


def window_mask(cfg, windows, T):
    m = np.zeros(T, dtype=bool)
    for (a, b) in windows:
        m[int(a / cfg.dt):int(b / cfg.dt)] = True
    return m
