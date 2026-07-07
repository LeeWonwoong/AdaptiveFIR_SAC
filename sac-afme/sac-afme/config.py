"""
config.py — SAC-AFME unified configuration
============================================
Single dataclass shared by: datagen (Isaac Sim), synthetic generator,
filter, log-replay env, SAC training, and evaluation.

Design references (see README):
  - Action = filter parameters output directly, constrained by range mapping
    (RL-AKF, Gao et al. 2020: exponential mapping for PSD; ours: tanh range map)
  - N >= dim(state) convention (Lee et al., IEEE TITS 2025 / FM-SLAC Remark 1)
  - lambda in (0,1]: Omega > 0 keeps the unbiasedness Lemma valid.
"""
from dataclasses import dataclass, field, asdict
import argparse
import json
import os


@dataclass
class Config:
    # ══════════════════════════════════════════════════════════
    #  General
    # ══════════════════════════════════════════════════════════
    seed: int = 42
    device: str = "auto"                # "auto" | "cuda" | "cpu"
    outdir: str = "results/run0"
    data_dir: str = "data"

    # ══════════════════════════════════════════════════════════
    #  UAV model (must match datagen/calibration.json 'drone')
    #  State s = [p(3), v(3), eta(3), omega(3)]  (ENU / FLU, yaw-pitch-roll ZYX)
    # ══════════════════════════════════════════════════════════
    dt: float = 0.02                    # 50 Hz logging / filter rate (decision #2)
    mass_nominal: float = 1.372         # kg — what the FILTER believes
    g: float = 9.81
    Ixx: float = 0.02663
    Iyy: float = 0.02663
    Izz: float = 0.05049
    # UWB anchors (DI-FME layout scaled to workspace)  [4,3]
    anchors: tuple = ((0.0, 0.0, 0.0), (10.0, 0.0, 3.0),
                      (10.0, 10.0, 0.0), (0.0, 10.0, 3.0))
    # measurement suite [FROZEN SPEC]: z = 4 UWB ranges ONLY (paper-identical).
    # The ESTIMATOR internally solves the full 12-state window LS exactly as
    # DI-FME eq.(8)-(18); the DELIVERABLE (reward/metrics/output claim) is the
    # position xyz — localization. The weakly observable (v,eta,omega) blocks
    # of the LS solution are internal byproducts, never delivered and (with
    # self_anchor=False) never fed back into the linearization.
    meas_sigma: tuple = (0.05, 0.05, 0.05, 0.05)       # UWB [m] (nominal; per-episode randomized)

    # ══════════════════════════════════════════════════════════
    #  Weighted FME (filter)
    # ══════════════════════════════════════════════════════════
    N_min: int = 8                      # >= dim(s)/meas-rank margin (TITS'25 convention)
    N_max: int = 20                     # ring-buffer length W (fixed shape)
    lam_min: float = 0.7                # Omega > 0 (unbiasedness Lemma premise)
    ridge_eps: float = 1e-8             # RELATIVE Tikhonov rows (numerical rank safety ONLY; bias ~1e-8)
    # PURE FME: no prior/regularization toward any prediction. The weak-
    # observability variance of (v,eta,omega) under UWB-only short windows is
    # made HARMLESS by anchoring the linearization on the auxiliary-EKF track
    # (self_anchor=False, EFIR practice): the LS byproducts never feed back.
    # self_anchor=True = paper-faithful ablation (DI-FME eq.(7) linearizes at
    # its own estimate) — reproduces the instability the original authors
    # report: "too small N led to unstable estimation".
    self_anchor: bool = False
    uwb_stride: int = 1                 # measurement epoch every `stride` filter steps
                                        # (1 = every 50 Hz step, the frozen design; >1 supported
                                        #  as an optional realism knob — N counts epochs)
    innov_gate: float = 2.0             # per-channel |nu| gate [m]: gated epoch row EXCLUDED from window
    state_clamp: bool = True            # physical projection of [v,att,rate] after solve (UAV)
    clamp_vel: float = 30.0
    clamp_att: float = 1.2
    clamp_rate: float = 20.0
    # default (non-adaptive) parameters for warmup & fixed-FME baselines
    N_default: int = 14                 # matches DI-FME experimental choice
    lam_default: float = 1.0

    # ══════════════════════════════════════════════════════════
    #  POMDP / log-replay environment
    # ══════════════════════════════════════════════════════════
    L_obs: int = 10                     # innovation-norm stack length (approx. information state)
    episode_len: int = 400              # RL steps per segment (epochs; 8 s @ 50 Hz, stride=1)
    warmup_steps: int = 8               # aux-EKF-only phase, in EPOCHS (= N_min; handover: filled_valid>=N_min)
    n_envs: int = 64                    # vectorized log-replay envs
    uwb_sigma_range: tuple = (0.03, 0.10)   # per-episode measurement-noise randomization [m]
    reward_scale: float = 100.0         # = 1/sigma_e^2, sigma_e=0.1 m  (r = -scale*||e||^2, RL-AKF)
    reward_clip: float = 100.0          # clip per-step cost (protects entropy auto-tuning)
    init_pos_noise: float = 0.05        # filter init: GT + N(0, sigma)

    # ══════════════════════════════════════════════════════════
    #  SAC (CleanRL-style)
    # ══════════════════════════════════════════════════════════
    total_steps: int = 300_000          # vector-env steps (x n_envs transitions)
    start_random_steps: int = 1_000     # pure exploration before policy actions
    learning_starts: int = 2_000        # transitions before updates
    updates_per_step: int = 8           # SGD updates per vector step
    batch_size: int = 512
    buffer_size: int = 1_000_000
    gamma: float = 0.95
    tau: float = 0.005
    lr: float = 3e-4
    weight_decay: float = 0.0           # AdamW decoupled decay (0 => Adam과 동일; 필요시 1e-4)
    hidden: int = 128
    autotune_alpha: bool = True
    target_entropy_scale: float = 1.0   # target_entropy = -scale * act_dim
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    eval_every: int = 5_000             # vector steps
    ckpt_every: int = 20_000
    ablation_fix_lambda: bool = False   # True → lambda := 1 (N-only ablation)
    ablation_fix_N: bool = False        # True → N := N_default (lambda-only ablation)

    # ══════════════════════════════════════════════════════════
    #  Scenario management (shared: synth generator & Isaac Sim datagen)
    # ══════════════════════════════════════════════════════════
    scenario_types: tuple = ("nominal", "mass_step", "gust", "sustained_wind", "mixed")
    scenario_probs: tuple = (0.25, 0.25, 0.25, 0.15, 0.10)
    flight_patterns: tuple = ("hover", "circle", "figure8", "waypoint", "aggressive")
    traj_duration_s: float = 40.0
    # mass_step
    mass_delta_range: tuple = (0.10, 0.40)     # +10~40 %
    mass_onset_frac: tuple = (0.15, 0.55)      # onset within trajectory
    # gust  (WindModel half-cosine profile, ported from user's repo)
    gust_speed_range: tuple = (4.0, 10.0)      # m/s
    gust_duration_range: tuple = (1.0, 5.0)    # s
    gust_count_range: tuple = (1, 3)
    # sustained wind
    sustained_speed_range: tuple = (2.5, 7.0)  # m/s
    # held-out (outside training ranges → generalization claim)
    heldout_mass_delta_range: tuple = (0.45, 0.60)
    heldout_gust_speed_range: tuple = (11.0, 14.0)
    # dataset sizes
    n_train_traj: int = 200
    n_heldout_traj: int = 50

    # ══════════════════════════════════════════════════════════
    #  Isaac Sim datagen (only used by datagen/*)
    # ══════════════════════════════════════════════════════════
    physics_hz: int = 250
    log_hz: int = 50                    # 250/50 = every 5 physics steps
    px4_ns: str = "auto"
    flight_alt: float = -0.0            # commander uses its own NED convention (see commander.py)
    anchor_keepout_m: float = 1.0       # trajectories keep >= 1 m from anchors (Jacobian conditioning)

    # ── derived ──
    state_dim: int = 12
    meas_dim: int = 4
    act_dim: int = 2

    def __post_init__(self):
        assert self.N_min * self.meas_dim >= self.state_dim + 2, "N_min too small for gain existence"
        assert self.N_max >= self.N_min
        assert 0.0 < self.lam_min <= 1.0
        self.warmup_steps = max(self.warmup_steps, self.N_min)
        self.n_anchors = len(self.anchors)

    # ---------- helpers ----------
    def resolve_device(self):
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        d = {k: v for k, v in asdict(self).items()}
        with open(path, "w") as f:
            json.dump(d, f, indent=2, default=str)


def parse_cli(cfg: Config = None) -> Config:
    """rhukf-style: dataclass defaults + argparse overrides (only common knobs)."""
    cfg = cfg or Config()
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default=cfg.outdir)
    p.add_argument("--data_dir", type=str, default=cfg.data_dir)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--device", type=str, default=cfg.device)
    p.add_argument("--total_steps", type=int, default=cfg.total_steps)
    p.add_argument("--n_envs", type=int, default=cfg.n_envs)
    p.add_argument("--batch", type=int, default=cfg.batch_size)
    p.add_argument("--updates_per_step", type=int, default=cfg.updates_per_step)
    p.add_argument("--L_obs", type=int, default=cfg.L_obs)
    p.add_argument("--fix_lambda", action="store_true")
    p.add_argument("--fix_N", action="store_true")
    p.add_argument("--episode_len", type=int, default=cfg.episode_len)
    p.add_argument("--start_random", type=int, default=cfg.start_random_steps)
    p.add_argument("--learning_starts", type=int, default=cfg.learning_starts)
    a, _ = p.parse_known_args()
    cfg.outdir, cfg.data_dir, cfg.seed = a.outdir, a.data_dir, a.seed
    cfg.device, cfg.total_steps, cfg.n_envs = a.device, a.total_steps, a.n_envs
    cfg.batch_size, cfg.updates_per_step, cfg.L_obs = a.batch, a.updates_per_step, a.L_obs
    cfg.ablation_fix_lambda, cfg.ablation_fix_N = a.fix_lambda, a.fix_N
    cfg.episode_len, cfg.start_random_steps = a.episode_len, a.start_random
    cfg.learning_starts = a.learning_starts
    return cfg
