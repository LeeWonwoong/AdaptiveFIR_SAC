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
    # PLANT process noise (the TRUE stochastic forcing the synth injects each
    # substep). CALIBRATED so nominal Greedy N_opt median ≈ 14 at σ_meas=0.12 —
    # i.e. the finite-horizon-optimal world of Shmaliy (a real system's process
    # is NOT deterministic, so full-horizon averaging is sub-optimal). The
    # model-based filters (EKF/UKF Q) BELIEVE this nominal value and never learn
    # the turbulence boost — the honest limit of a fixed-Q KF.
    # NOTE: the N_opt≈14 calibration target is UNREACHABLE here — N_opt is
    # floored ~16 by range-measurement nonlinearity over the maneuver (position
    # is the double-integral of the accel noise AND is directly observed by UWB,
    # so process noise perturbs weakly-observed velocity, not the position N_opt).
    # These values are the DEMONSTRATION operating point that makes the fixed-Q
    # EKF lag under turbulence while nominal stays sane.
    proc_acc_std: float = 1.50          # m/s^2  (nominal plant accel process noise)
    proc_gyro_std: float = 0.45         # rad/s^2 (angular process noise)
    # PRACTITIONER mis-tuning of the recursive baselines (Shmaliy: a real KF
    # user does NOT know the true noise statistics — KF only leads for β∈
    # [0.7,1.6]). The default EKF/UKF use a datasheet R (UNDER-states σ) and a Q
    # that UNDER-states the process noise (never anticipates turbulence). The
    # oracle-KF (correct R=meas_sigma, correct Q=q0) is reported alongside as
    # the honest upper bound.
    ekf_R_sigma: float = 0.10           # practitioner datasheet σ [m] (< true 0.12~0.45)
    ekf_Q_scale: float = 0.40           # practitioner process-σ = ekf_Q_scale * q0 (~q0/2.5)
    # UWB anchors (DI-FME layout scaled to workspace)  [4,3]
    anchors: tuple = ((0.0, 0.0, 0.0), (10.0, 0.0, 3.0),
                      (10.0, 10.0, 0.0), (0.0, 10.0, 3.0))
    # measurement suite [FROZEN SPEC]: z = 4 UWB ranges ONLY (paper-identical).
    # The ESTIMATOR internally solves the full 12-state window LS exactly as
    # DI-FME eq.(8)-(18); the DELIVERABLE (reward/metrics/output claim) is the
    # position xyz — localization. The weakly observable (v,eta,omega) blocks
    # of the LS solution are internal byproducts, never delivered and (with
    # self_anchor=False) never fed back into the linearization.
    # UWB LoS σ realised at INFME level (R≈0.014 m²; NLoS bursts raise it to
    # R≈0.2 m² per anchor, INFME-adjacent). This is the FIXED noise statistic
    # every model-based filter (FME whitening, EKF/UKF R) BELIEVES — it does NOT
    # know the NLoS σ jump, which is the whole point of [수정C].
    meas_sigma: tuple = (0.12, 0.12, 0.12, 0.12)       # UWB LoS [m] (nominal; per-episode randomized)

    # ══════════════════════════════════════════════════════════
    #  Weighted FME (filter)
    # ══════════════════════════════════════════════════════════
    N_min: int = 8                      # >= dim(s)/meas-rank margin (TITS'25 convention)
    N_max: int = 20                     # ring-buffer length W (fixed shape). At the
                                        # CALIBRATED process noise q0 the nominal N_opt≈14
                                        # (DI-FME choice) sits mid-range → real headroom.
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
    # ── Observation (POMDP): per-step feature o_t = [nu_1..nu_4, N_hat, lam_hat]
    #    stacked over L steps -> obs_dim = (meas_dim + 2) * L.
    #    nu   : per-anchor innovation (residual), clipped to [-resid_clip, +resid_clip]
    #    N_hat: previous horizon,   min-max normalized to [-1,1]
    #    lam_hat: previous lambda,  min-max normalized to [-1,1]  (max is 1.0)
    #    Residual VECTOR (not scalar RMSE) so the agent distinguishes a single
    #    anchor fault [0.4,0,0,0] from a global degradation [0.2,0.2,0.2,0.2]
    #    -> essential for the anchor-dropout scenario.
    L_obs: int = 6                      # sliding-window length (steps)
    resid_clip: float = 1.0             # per-anchor residual clip [m]
    episode_len: int = 150              # RL steps per segment (shorter -> higher disturbance
                                        # density + more episodes before alpha settles)
    warmup_steps: int = 8               # aux-EKF-only phase, in EPOCHS (= N_min; handover: filled_valid>=N_min)
    n_envs: int = 64                    # vectorized log-replay envs
    uwb_sigma_range: tuple = (0.08, 0.16)   # per-episode LoS σ randomization [m] (brackets 0.12)
    # Reward: r = -||p_gt - p_hat||  (pure localization error, L2 distance),
    # with a safety clip only (numerical guard, NOT reward engineering).
    reward_clip: float = 10.0           # clip per-step |cost| at 10 m (protects entropy auto-tuning)
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
    hidden: int = 64                    # MLP width (36-dim obs -> 64x64 sufficient)
    autotune_alpha: bool = True
    target_entropy_scale: float = 0.5   # target_entropy = -scale * act_dim (0.5 keeps
                                        # more exploration; 1.0 collapsed alpha->0 here)
    alpha_min: float = 0.02             # floor so entropy never fully vanishes
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    eval_every: int = 5_000             # vector steps
    ckpt_every: int = 20_000
    ablation_fix_lambda: bool = False   # True → lambda := 1 (N-only ablation)
    ablation_fix_N: bool = False        # True → N := N_default (lambda-only ablation)

    # ══════════════════════════════════════════════════════════
    #  Scenario management (shared: synth generator & Isaac Sim datagen)
    # ══════════════════════════════════════════════════════════
    # FOCUSED SET for the IIR<FIR<AFIR study (DI-FME Table I pattern: nominal
    # ≈-tied, disturbance windows FIR≫IIR & AFIR≫FIR). The lever is MEASUREMENT-
    # side (established: plant kicks barely dent UWB localization):
    #   nominal          LoS σ, no fault          → IIR≈FIR at N_opt (Shmaliy)
    #   turbulence_burst process-noise q×3~5, 2~5s → EKF (fixed Q) LAGS (main)
    #   nlos_burst       one anchor σ↑ + bias 2~4s → EKF over-trusts (main)
    #   anchor_dropout   one anchor out 3~5 s      → geometry loss (FIR tier=DI-FME)
    scenario_types: tuple = ("nominal", "turbulence_burst",
                             "nlos_burst", "anchor_dropout")
    scenario_probs: tuple = (0.15, 0.35, 0.25, 0.25)
    # flight patterns (NO 'aggressive': high-G maneuvers break the 1st-order
    # Taylor linearization the FIR relies on — out of scope by design).
    flight_patterns: tuple = ("hover", "circle", "figure8", "waypoint")
    traj_duration_s: float = 40.0
    # ── DISTURBANCE SCENARIOS (paper-motivated: sharp events + recovery,
    #    NOT high-G maneuvers — 1st-order Taylor linearization stays valid).
    #    The (N,lambda) benefit is the UFIR transient/noise tradeoff
    #    (1_FIR_filter Fig.3/5): a sharp event lowers N_opt (dump the polluted
    #    old data fast), a calm segment raises N_opt (average out noise). ──
    # payload coupling (UIFM-SLAC Scenario 2): STEP mass jump + a z-sink /
    # lateral velocity impulse at the coupling instant -> position error spike.
    mass_delta_range: tuple = (0.40, 0.70)     # STEP +40~70 % (was gentle 10~40)
    mass_onset_frac: tuple = (0.30, 0.60)      # coupling instant within trajectory
    mass_impulse_z: tuple = (0.6, 1.2)         # downward velocity impulse [m/s] at coupling
    mass_impulse_xy: tuple = (0.3, 0.7)        # lateral velocity impulse [m/s]
    # gust: SHARP impulsive gusts (FM-SMC abrupt disturbance), not slow ramps
    gust_speed_range: tuple = (8.0, 14.0)      # m/s (stronger)
    gust_duration_range: tuple = (0.2, 0.6)    # s (sharp impulse, was 1~5 s)
    gust_count_range: tuple = (2, 4)
    # sustained wind
    sustained_speed_range: tuple = (2.5, 7.0)  # m/s
    # anchor dropout: PROLONGED full outage (UIFM-SLAC Scenario 3 = ~7 s NLOS;
    # DI-FME intermittent). One anchor lost -> GDOP worsens (3-anchor GDOP up
    # to ~11 near workspace edges); large N holds the pre-dropout 4-anchor data
    # to ride it out. This is the clearest (N,lambda) benefit case.
    # ── TURBULENCE BURST (주력): the plant's process noise q is boosted ×3~5
    #    for 2~5 s, 2~4 times. The TRUE trajectory becomes erratic; a fixed-Q
    #    EKF/UKF (believes nominal q0) UNDER-weights measurements → lags; the FIR
    #    ignores noise statistics → stays robust; AFIR sees innovation variance
    #    rise and SHORTENS N. This is the Shmaliy FIR>KF mechanism (process-side,
    #    accumulates in the recursive filter). Amplitude bounded → Taylor valid.
    turb_count_range: tuple = (2, 4)
    turb_duration_range: tuple = (2.0, 5.0)    # s per turbulence burst
    turb_boost_range: tuple = (5.0, 8.0)       # σ multiplier (eff 7.5~12 m/s²): below
                                               # ~7 the EKF still tracks (position is
                                               # UWB-observed) → no IIR<FIR separation
    # ── ANCHOR DROPOUT (주력): one anchor fully lost 3~5 s (range→NaN). EKF/UKF
    #    do prediction-only on the missing rows ([수정B]) → the WRONG nominal
    #    model integrates unchecked → honest divergence; FME excludes the row
    #    and the surviving 3-anchor window still localizes (large N holds the
    #    pre-dropout 4-anchor epochs).
    dropout_count_range: tuple = (1, 2)
    dropout_duration_range: tuple = (3.0, 5.0) # s (prolonged but bounded)
    dropout_max_anchors: int = 1               # one anchor fully lost (NLoS-like)
    # ── NLoS BURST (주력): one anchor's σ jumps LoS(0.12)→NLoS(0.45, R≈0.2 m²)
    #    with a positive multipath bias, intermittently for 2~4 s. The model-
    #    based filters keep believing R=LoS → EKF OVER-TRUSTS the corrupted
    #    anchor and its error amplifies; FME's bounded batch-LS limits the
    #    damage and AFIR can down-weight (λ, or drop N).
    nlos_count_range: tuple = (2, 4)
    nlos_duration_range: tuple = (2.0, 4.0)    # s per NLoS burst
    nlos_sigma: float = 0.45                   # NLoS σ [m] (R≈0.20 m², INFME-adjacent)
    nlos_bias_range: tuple = (0.3, 0.5)        # m positive multipath bias (< innov_gate 2.0)
    # held-out (outside training ranges → generalization claim)
    heldout_mass_delta_range: tuple = (0.70, 0.90)
    heldout_gust_speed_range: tuple = (14.0, 18.0)
    heldout_nlos_bias_range: tuple = (0.5, 0.7)  # stronger NLoS bias (still < gate)
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

    @property
    def obs_dim(self):
        return (self.meas_dim + 2) * self.L_obs      # [nu_1..4, N_hat, lam_hat] x L

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
