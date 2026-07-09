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
    anchors: tuple = ((0.0, 0.0, 0.0), (10.0, 0.0, 4.0),
                      (10.0, 10.0, 0.0), (0.0, 10.0, 4.0))
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
    meas_sigma: tuple = (0.12, 0.12, 0.12, 0.12,      # UWB LoS [m]
                         0.02, 0.02, 0.02,           # IMU attitude [rad] (FCU estimate)
                         0.01, 0.01, 0.01)           # IMU gyro [rad/s]

    # ══════════════════════════════════════════════════════════
    #  Weighted FME (filter)
    # ══════════════════════════════════════════════════════════
    N_min: int = 4                      # observability rank-12 reached at N=2 (UWB+IMU); 4 for noise-reduction margin
    N_max: int = 20                     # ring-buffer length W (fixed shape). At the
                                        # CALIBRATED process noise q0 the nominal N_opt≈14
                                        # (DI-FME choice) sits mid-range → real headroom.
    lam_min: float = 0.7                # Omega > 0 (unbiasedness Lemma premise)
    # Inversion policy: PLAIN inverse (never SVD). ridge_eps is a RELATIVE
    # Tikhonov floor on the observable-block normal matrices (stage-1 He^T W He
    # and stage-2 Om). Principle (mode 3): keep it MINIMAL (1e-8) so the solve
    # stays deadbeat/unbiased (T2) — regularization then only guards numerical
    # rank. If a genuine singularity shows up in practice, RAISE it (mode 1,
    # e.g. 1e-5): invertibility is guaranteed at the cost of an O(ridge_eps)
    # deadbeat bias. Just change this one value; the code path is identical.
    ridge_eps: float = 1e-8             # 1e-8 = principled (deadbeat); 1e-5 = singularity-safe
    est_dim: int = 12                   # FULL 12-state corrected (UWB+IMU fusion makes C full-rank); attitude/
                                        # rate follow control-driven dynamics (delta == 0) —
                                        # the '위치만 추정' contract, structural not a prior
    Np_fix: int = 4                     # mini-batch (stage-1) length, USER-SET, not an RL
                                        # action: pure-FME estimates are Np-invariant
                                        # (iterative == RLS factorization of the batch,
                                        # Shmaliy S18-S24; verified V1), so Np only sets the
                                        # numerical init path. Rank floor: ceil(est_dim/q)=2
                                        # epochs; 4 gives comfortable stage-1 conditioning.
    full_batch_flag: float = -1.0       # Np sentinel: if the Np passed to the filter is <= 0
                                        # (e.g. this value), stage-2 iteration is SKIPPED and the
                                        # estimate is the SINGLE full-window batch LS over N
                                        # (SAC-chosen). Any positive Np => two-stage batch+iter.
    _unused_rcond: float = 1e-6         # (deprecated: SVD truncation removed; plain inverse now)
                                        # below rcond*smax are UNOBSERVABLE in the window and
                                        # receive zero correction (follow the dynamics) — the
                                        # literal Moore-Penrose semantics, not a prior.
    # PURE FME, classical self-sustaining extended-FIR:
    #   - The auxiliary EKF is WARMUP-ONLY. It serves (anchor + output) until
    #     the window can solve (filled_valid >= N_min), then switches OFF.
    #   - After handover the FIR linearizes along its OWN delivered trajectory
    #     (DI-FME eq.(7): A_t, C_t at s_hat_{t-1}); the horizon fills with
    #     FIR-anchored Jacobians and within N epochs is fully self-sustaining.
    # self_anchor=True = ablation: SKIP the EKF warmup and self-linearize from
    #   t=0 (no stable seed) — reproduces the instability the original authors
    #   report ("too small N led to unstable estimation"; ~1e2 km divergence),
    #   which is the MOTIVATION for the observability-based N_min, not a mode
    #   the deployed filter uses.
    self_anchor: bool = False
    uwb_stride: int = 5                 # measurement 10Hz (DW1000 TWR x4 anchors); prediction stays 50Hz
                                        # (1 = every 50 Hz step, the frozen design; >1 supported
                                        #  as an optional realism knob — N counts epochs)
    innov_gate: float = float("inf")    # DEPRECATED/unused — magnitude gate REMOVED
                                        # (final mix has no NLoS outliers; model-error
                                        #  innovations ARE the adaptation signal and must
                                        #  reach both the LS and the SAC observation).
                                        # Field kept only for arg-compat of old tools/tests.
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
    obs_channel_norms: bool = True      # True → per-GROUP whitened innovation norms
                                        # [‖ν_UWB‖/σ, ‖ν_att‖/σ, ‖ν_gyro‖/σ, N̂, λ̂] x L
                                        # (user-selected option 3; groups let SAC tell
                                        #  "UWB trouble (dropout/NLoS)" from "IMU trouble").
                                        # False → legacy per-channel residual vector.
    n_obs_groups: int = 3               # UWB / attitude / gyro
    obs_group_scale: tuple = (1.0, 8.0, 12.0)
                                        # ISAAC-MEASURED nominal whitened-norm levels
                                        # (uwb ~1.1-1.3 | att ~2-16 pattern-dep | gyro
                                        #  ~10-22): IMU innovations are MODEL-error
                                        # dominated, so sigma-whitening alone puts them
                                        # at 3-40 and the 4-sigma clip would PERMANENTLY
                                        # saturate the IMU groups. Obs = norm/scale =
                                        # "x nominal level"; disturbance reaches 2-3.5x
                                        # (measured: wind uwb x3, turb gyro x2-3) with
                                        # no saturation. Filter LS whitening unchanged.
    resid_clip: float = 4.0             # whitened-residual clip [sigma units].
                                        # WAS 1.0 -> SATURATION BUG: clip applies AFTER
                                        # sigma-whitening, so nominal N(0,1) channels sat at
                                        # 1 sigma -> group norm ceiling 1.0 while nominal sits
                                        # ~0.92-0.95 (measured): the x2.1 disturbance rise was
                                        # compressed into [0.95,1.0], invisible to SAC.
                                        # 4 sigma keeps nominal ~0.92 and lets disturbance
                                        # reach ~2 with headroom.
    episode_len: int = 150              # RL steps per segment (shorter -> higher disturbance
                                        # density + more episodes before alpha settles)
    warmup_steps: int = 20              # aux-EKF-only phase in EPOCHS. Set to N_max so that,
                                        # by the first SAC action, filled_valid == N_max and ANY
                                        # N in [N_min, N_max] is fully available (no growing-window
                                        # transient, no N clipping) — the SEFFB principle that a
                                        # horizon-N filter is used only once N samples are buffered.
    n_envs: int = 64                    # vectorized log-replay envs
    uwb_sigma_range: tuple = (0.08, 0.16)   # per-episode LoS σ randomization [m] (brackets 0.12)
    # Reward: r = -||p_gt - p_hat||  (pure localization error, L2 distance),
    # with a safety clip only (numerical guard, NOT reward engineering).
    reward_clip: float = 10.0           # clip per-step |cost| at 10 m (protects entropy auto-tuning)
    reward_mode: str = "sq"             # "sq": r = -clip(||e||)^2  (DEFAULT)
                                        #   quadratic cost == direct RMSE minimization
                                        #   (the reported metric IS the square metric),
                                        #   and it amplifies the disturbance-window signal
                                        #   ~3x vs linear WITHOUT any baseline term:
                                        #   window err 0.55 m vs nominal 0.18 m ->
                                        #   linear ratio 3.1x, squared ratio 9.4x.
                                        # "abs": legacy r = -clip(||e||).
    ref_monitor_N: int = 14             # >0: run a parallel FIXED N=14, lam=1 filter
                                        # (= DI-FME) on the identical stream for LOGGING
                                        # ONLY — the live "rmse vs DI-FME" column and the
                                        # eval table. NEVER enters the reward (user
                                        # decision: reward stays absolute-error-based).
                                        # 0 = off (saves ~2x env compute).
    act_smooth_coef: float = 0.005      # small |dN| penalty (units of N-range) so the
                                        # learned N(t) is regime-STEPS, not jitter — the
                                        # paper figure needs Shmaliy-style plateaus.
    train_scenario_types: tuple = ("nominal", "sustained_wind")
                                        # dataset WHITELIST (FOCUSED RUN, user decision:
                                        # single best scenario + nominal for contrast).
                                        # sustained_wind has the steepest window curve
                                        # (N=12 in-window costs 2.1x vs N*=6) -> the
                                        # clearest N(t) swing figure. () = use all types.
    init_pos_noise: float = 0.05        # filter init: GT + N(0, sigma)

    # ══════════════════════════════════════════════════════════
    #  SAC (CleanRL-style)
    # ══════════════════════════════════════════════════════════
    total_steps: int = 300_000          # vector-env steps (x n_envs transitions)
    start_random_steps: int = 6_000     # pure exploration before policy actions.
                                        # WAS 1k -> the buffer barely sampled small-N
                                        # actions inside disturbance windows before
                                        # exploitation began. 6k vector steps x n_envs
                                        # covers the whole (N,lam) box in windows, so the
                                        # Q-function KNOWS N=6-in-window is good before
                                        # the policy commits ("explore first" without
                                        # corrupting the reward).
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
    target_entropy_scale: float = 1.0   # target_entropy = -scale * act_dim (0.5 keeps
                                        # more exploration; 1.0 collapsed alpha->0 here)
    alpha_min: float = 0.08             # exploration floor. WAS 0.02 -> alpha hit the
                                        # floor by ~2.5k steps and the policy froze into
                                        # the best STATIC compromise (N=11, lam=0.83,
                                        # rmse 0.182 = fixed-filter optimum) before ever
                                        # discovering the CONDITIONAL gain (windows are
                                        # ~10-15 % of steps; gain ~0.005 avg reward is
                                        # invisible without sustained exploration).
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
    # ── FINAL scenario mix (container STAGE-2 gate, 2026-07-09):
    #    payload mass_step is the PRIMARY adaptation driver (N* 14→6 at both
    #    +75 % and +100 %, magnitude-robust); sustained strong wind 15~20 m/s
    #    is SECONDARY (N* 14→4; 10~12 m/s is direction-ambiguous → avoided);
    #    turbulence_burst is kept as the IIR<FIR<AFIR chain scenario
    #    (Shmaliy Fig.10 reproduction: EKF 0.360 > FIR 0.165 > AFIR 0.134).
    #    anchor_dropout / nlos_burst EXCLUDED from the default mix (IMU fusion
    #    bridges 1-anchor loss → no N* shift; 2-anchor loss = collapse, not
    #    adaptation). Sampler branches remain available for ablations.
    scenario_types: tuple = ("nominal", "mass_step",
                             "sustained_wind", "turbulence_burst")
    scenario_probs: tuple = (0.15, 0.40, 0.25, 0.20)
    # flight patterns (NO 'aggressive': high-G maneuvers break the 1st-order
    # Taylor linearization the FIR relies on — out of scope by design).
    flight_patterns: tuple = ("helical", "figure8", "waypoint")
    traj_duration_s: float = 40.0
    # ── DISTURBANCE SCENARIOS (paper-motivated: sharp events + recovery,
    #    NOT high-G maneuvers — 1st-order Taylor linearization stays valid).
    #    The (N,lambda) benefit is the UFIR transient/noise tradeoff
    #    (1_FIR_filter Fig.3/5): a sharp event lowers N_opt (dump the polluted
    #    old data fast), a calm segment raises N_opt (average out noise). ──
    # payload coupling (UIFM-SLAC Scenario 2): STEP mass jump + a z-sink /
    # lateral velocity impulse at the coupling instant -> position error spike.
    mass_delta_range: tuple = (0.60, 0.90)     # STEP +60~90 % (centre +75 %: N* 14→6 measured)
    mass_onset_frac: tuple = (0.30, 0.60)      # coupling instant within trajectory
    mass_impulse_z: tuple = (0.6, 1.2)         # downward velocity impulse [m/s] at coupling
    mass_impulse_xy: tuple = (0.3, 0.7)        # lateral velocity impulse [m/s]
    # gust: SHARP impulsive gusts (FM-SMC abrupt disturbance), not slow ramps
    gust_speed_range: tuple = (15.0, 20.0)     # m/s — <15 m/s shifts N* the WRONG way
                                               # (measured: 10 m/s → N*=20, 12 m/s → ±2 ambiguous;
                                               #  ≥15 m/s → N* 14→4). PX4-compensation-exceeding only.
    gust_duration_range: tuple = (4.0, 8.0)    # s (sustained-style; the validated WIN was ~7 s)
    gust_count_range: tuple = (2, 4)
    # sustained wind
    # WINDOWED sustained wind (2026-07-09): whole-trajectory wind teaches the
    # policy nothing about TRANSITIONS (a constant N=4 suffices within such a
    # trajectory); adaptation value lives at regime edges. A mid-trajectory
    # window (validated N_opt shift used a ~7 s window) gives calm -> onset ->
    # N shrink -> offset -> N recover, twice per trajectory.
    sustained_onset_frac: tuple = (0.20, 0.50)   # window start within trajectory
    sustained_duration_range: tuple = (8.0, 15.0)  # s (>= validated ~7 s window)
    sustained_speed_range: tuple = (15.0, 20.0)  # m/s — must exceed PX4 compensation so GT
                                               # actually deflects (6 m/s legacy data: vel-std
                                               # 1.258→1.293, i.e. fully compensated → useless)
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
    turb_wind_sigma_per_boost: float = 2.0     # Isaac mapping: turbulent WIND std [m/s]
                                               # = knob x boost -> 10~16 m/s buffeting (clip 1.5σ)
                                               # (variance-preserving OU; must be validated
                                               #  in Isaac: crash rate & GT deflection)
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
    # ── TIME-CORRELATED (Gauss-Markov / OU) per-anchor measurement bias
    #    [Phase-0 last mechanism]. White measurement noise always averages down
    #    √N, so N_opt pins at N_max (Phase-0 NO-GO). Real UWB multipath/NLoS
    #    error is TIME-CORRELATED (geometry-dependent, slowly wandering), which
    #    breaks the averaging gain and is the only untried measurement-side way
    #    to make N_opt finite. Discrete OU per (traj, anchor):
    #       b[k+1] = (1 - dt/τ)·b[k] + N(0, σ_w),   σ_w = σ_b·√(1-(1-dt/τ)²)
    #    → zero-mean, stationary std σ_b, correlation time τ. Replaces the
    #    old within-burst CONSTANT bias (which had no N-lever: every epoch in a
    #    window was identically polluted). LoS baseline = nearly static fine
    #    systematic error; NLoS burst = fast-wandering (τ < window) — the lever.
    gm_bias: bool = True                       # enable OU measurement bias
    los_bias_std: float = 0.04                 # σ_b LoS baseline [m] (fine systematic)
    los_bias_tau: float = 3.0                  # τ LoS [s] (nearly static, 2~4 s)
    nlos_bias_std: float = 0.22                # σ_b NLoS burst [m] (0.15~0.30)
    nlos_bias_tau: float = 0.5                 # τ NLoS [s] (0.3~0.8 s; < N·dt window → wanders)
    gm_bias_clip: float = 1.5                  # |b| clamp [m] (stay < innov_gate 2.0)
    gm_bias_seed: int = 20260708               # fixed OU draw → reproducible across tools
    # ── TAG-SIDE COMMON-MODE bias [Phase-0 FINAL test]. Documented UWB tag
    #    errors — DW1000 received-power-dependent timestamp shift, tag antenna
    #    radiation-pattern range offset, tag clock drift — are COMMON to all
    #    anchors and vary TIME-CORRELATED with the drone's attitude (antenna
    #    orientation / RX power change as the body tilts). Unlike single-anchor
    #    NLoS this hits every anchor at once, so it is NOT geometrically rejected
    #    → the A-1 all-anchor-OU world, but physically grounded and spec-relevant.
    #      range[a] += b_common(k)·s_a + b_a(k)
    #    b_common : tag OU whose (σ_b, τ) is tied to attitude activity —
    #      calm (hover/cruise):  quasi-static (σ_b small, τ large)
    #      dynamic (attitude-active): fast/large (σ_b large, τ < N·dt window)
    #    s_a : per-anchor sensitivity (antenna-pattern heterogeneity), b_a: the
    #    per-anchor NLoS OU (kept). Regime = calm↔dynamic flight segments.
    cm_bias: bool = True                       # enable tag-side common-mode OU
    # cm_mode: "common"      = coherent b_common·s_a (clock/RX-power: same wander
    #                          on every anchor → geometrically absorbed, no N-lever)
    #          "independent" = per-anchor INDEPENDENT OU on ALL anchors in dynamic
    #                          segments (antenna-pattern vs bearing: each anchor its
    #                          own attitude-driven wander → A-1's per-anchor world)
    cm_mode: str = "common"
    cm_calm_std: float = 0.04                  # σ_b calm [m] (quasi-static)
    cm_calm_tau: float = 3.0                   # τ calm [s]
    cm_dyn_std: float = 0.20                   # σ_b dynamic [m] (A-1 success point)
    cm_dyn_tau: float = 0.4                    # τ dynamic [s] (< N·dt → wanders)
    cm_sens_range: tuple = (0.7, 1.3)          # per-anchor sensitivity s_a (antenna pattern)
    cm_calm_dur_range: tuple = (2.5, 4.0)      # s per calm segment
    cm_dyn_dur_range: tuple = (2.0, 3.5)       # s per dynamic segment
    # held-out (outside training ranges → generalization claim)
    heldout_mass_delta_range: tuple = (0.90, 1.10)
    heldout_gust_speed_range: tuple = (20.0, 24.0)
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
    meas_dim: int = 10                  # UWB 4 + IMU attitude 3 + gyro 3
    act_dim: int = 2

    @property
    def obs_dim(self):
        if self.obs_channel_norms:
            return (self.n_obs_groups + 2) * self.L_obs   # [g_uwb, g_att, g_gyro, N_hat, lam_hat] x L
        return (self.meas_dim + 2) * self.L_obs      # legacy: [nu_1..nz, N_hat, lam_hat] x L

    def __post_init__(self):
        assert self.N_min * self.meas_dim >= self.state_dim + 2, "N_min too small for gain existence"  # 4*10=40 >= 14 OK
        assert self.N_max >= self.N_min
        assert 0.0 < self.lam_min <= 1.0
        self.warmup_steps = max(self.warmup_steps, self.N_max)   # handover completed for ALL N
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
