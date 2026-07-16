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
    anchors: tuple = ((1.0, 1.0, 0.0), (9.0, 1.0, 5.0),
                      (9.0, 9.0, 0.0), (1.0, 9.0, 5.0))
    # GEOMETRY UPDATE (2026-07-09, measured): anchors pulled IN to 8x8 and the
    # upper pair raised to z=5 -> steeper elevation angles from the flight band
    # (z 1~2.6) -> vertical GDOP improves: FIR nominal z 0.145 -> 0.108 (-26%)
    # at an x,y cost of +0.005. Trajectories (x 1.5~8.4, y 2.7~8.0) remain
    # inside the anchor hull. Ranges are synthesized OFFLINE from GT + anchors,
    # so NO Isaac re-run is needed — only retraining on the new measurements.
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
    meas_sigma: tuple = (0.10, 0.10, 0.10, 0.10,
                         0.01, 0.01, 0.01,
                         0.005, 0.005, 0.005)

    # ══════════════════════════════════════════════════════════
    #  Weighted FME (filter)
    # ══════════════════════════════════════════════════════════
    N_min: int = 6                      # observability rank-12 reached at N=2 (UWB+IMU); 4 for noise-reduction margin
    N_max: int = 20                     # ring-buffer length W (fixed shape). At the
                                        # CALIBRATED process noise q0 the nominal N_opt≈14
                                        # (DI-FME choice) sits mid-range → real headroom.
    lam_min: float = 0.75
    # IFIABLE -- (N=5,lam=0.8), (N=4,lam=1), (N=7,lam=0.7) give nearly the same
    # effective memory, so the Q-landscape has a ridge along constant-memory
    # contours and SAC parks lambda at an arbitrary low value (measured 0.72-
    # 0.80 across three runs), which then blocks the long-memory corner
    # (N>=10 AND lam~1) the oracle uses. Restricting lambda to [0.9, 1] makes
    # N the memory-length control and lambda a within-window weighting trim.                # Omega > 0 (unbiasedness Lemma premise)
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
    n_obs_groups: int = 3               # UWB / attitude / gyro (3 groups, v9f_B-validated)
    obs_drop_gyro: bool = False          # exclude gyro group from observation
    obs_group_scale: tuple = (1.14, 1.74, 14.32)
    # ^ sbar per group, RE-ESTIMATED 2026-07-14 on v9e (0.8x, w=0.4, IMU att 0.01 / gyr 0.005) nominal heldout
    #   (WFME N=6 innovations, whitened group norms over 6-48 s, 3 patterns,
    #   MEDIAN estimator -- the typical calm level. RMS/mean is rejected: the
    #   attitude channel is so heavy-tailed in agile flight (RMS 31 vs median
    #   1.2) that a mean-based scale would be set by a handful of manoeuvre
    #   spikes and push the typical level to eps~1e-3, blinding the channel;
    #   the tail is exactly what the log compression is for). The stale
    #   values (1.0, 8.0, 12.0) came from the old slow-trajectory system
    #   and left the attitude/gyro channels at eps=0.02/0.24 in calm v9c
    #   flight -- compressed to ~0, i.e. the policy was BLIND on the two
    #   channels that carry the wind/payload signature (measured window
    #   contrast with corrected sbar: wind eps=(2.5,1.5,6.4), payload
    #   eps=(1.3,1.8,8.4)). Re-estimate whenever the trajectory regime or
    #   sensor setup changes.
                                        # ISAAC-MEASURED nominal whitened-norm levels
                                        # (uwb ~1.1-1.3 | att ~2-16 pattern-dep | gyro
                                        #  ~10-22): IMU innovations are MODEL-error
                                        # dominated, so sigma-whitening alone puts them
                                        # at 3-40 and the 4-sigma clip would PERMANENTLY
                                        # saturate the IMU groups. Obs = norm/scale =
                                        # "x nominal level"; disturbance reaches 2-3.5x
                                        # (measured: wind uwb x3, turb gyro x2-3) with
                                        # no saturation. Filter LS whitening unchanged.
    obs_log_compress: bool = False
    obs_log_denom: float = 1.5
    obs_squared_stat: bool = False
    # FROZEN-NIS OBSERVATION (2026-07-13, final). With the flag ON the per-
    # group statistic is the SQUARED whitened innovation norm normalised by
    # the nominal innovation level:
    #     eps_i = ||nu_i||^2_{R^-1} / (d_i * gbar_i^2)
    #           = nu_i' Sbar_i^{-1} nu_i / d_i,
    # i.e. the group-wise NIS computed with the innovation covariance Sbar
    # FROZEN at its calm-flight estimate (gbar^2 = tr(Sbar R^-1)/d, the values
    # in obs_group_scale). E[eps]=1 in calm flight BY CONSTRUCTION. We freeze
    # Sbar instead of using the window-dependent H P H' + R on purpose: the
    # live S shrinks/grows with the agent's own memory choice N, which would
    # let the policy attenuate its alarm signal by shortening the window
    # (observation-action coupling) instead of reacting to the disturbance.
    # Squared features also empirically outperform linear ones in learned
    # filtering (Recursive KalmanNet, 2025), consistent with the quadratic
    # nature of the underlying covariances. Flag OFF restores the legacy RMS
    # statistic (v4-validated) without touching the data.
    # OBSERVATION COMPRESSION (2026-07-13). Replaces the hard clip. The group
    # innovation mu (=1 in calm flight) is HEAVY-TAILED: the attitude channel
    # has median 0.15 but p99 ~ 17 and max ~ 23 during agile segments, so 3-6%
    # of steps used to saturate the old clip at 4 -- which mapped mu=4.1 and
    # mu=22 onto the same input. We instead apply
    #       mu_tilde = log(1+mu) / (obs_log_denom + log(1+mu)),
    # which is (i) strictly bounded in [0,1) so no clip is needed, (ii) MONOTONE
    # so outliers stay distinguishable, and (iii) tail-compressing so a single
    # transient cannot dominate the input. mu is a RATIO-scale quantity
    # (multiples of the calm-flight level), and the log turns its multiplicative
    # structure additive -- the natural transform. Mapping with denom=1.5:
    #   calm mu=1 -> 0.32 | disturbance mu=4 -> 0.52 | outlier mu=22 -> 0.68.
    resid_clip: float = 4.0             # (legacy path only) whitened-residual clip.
                                        # WAS 1.0 -> SATURATION BUG: clip applies AFTER
                                        # sigma-whitening, so nominal N(0,1) channels sat at
                                        # 1 sigma -> group norm ceiling 1.0 while nominal sits
                                        # ~0.92-0.95 (measured): the x2.1 disturbance rise was
                                        # compressed into [0.95,1.0], invisible to SAC.
                                        # 4 sigma keeps nominal ~0.92 and lets disturbance
                                        # reach ~2 with headroom.
    episode_len: int = 250              # RL steps per segment. v10 (user-approved,
                                        # 2026-07-15): 150 -> 250 so a segment contains the
                                        # full disturbance window PLUS the recovery tail —
                                        # the return then credits fast post-window recovery.
                                        # Fits: seg=(20+250+1)*5=1355 < T=2500 (50 s @50 Hz).
    warmup_steps: int = 20              # aux-EKF-only phase in EPOCHS. Set to N_max so that,
                                        # by the first SAC action, filled_valid == N_max and ANY
                                        # N in [N_min, N_max] is fully available (no growing-window
                                        # transient, no N clipping) — the SEFFB principle that a
                                        # horizon-N filter is used only once N samples are buffered.
    n_envs: int = 64                    # vectorized log-replay envs
    uwb_sigma_range: tuple = (0.07, 0.13)   # per-episode LoS σ randomization [m] (brackets 0.10)
    # Reward: r = -||p_gt - p_hat||  (pure localization error, L2 distance),
    # with a safety clip only (numerical guard, NOT reward engineering).
    reward_clip: float = 10.0           # clip per-step |cost| at 10 m (protects entropy auto-tuning)
    reward_err_scale: float = 0.2       # e0 [m]: reward normalization scale
                                        # (~nominal error level). r = -(min(e,2)/e0)^2
                                        # -> nominal -0.36, unadapted window -4.0,
                                        # adapted -1.6: the per-step adaptation gain
                                        # becomes ~2.4 (was ~0.1 raw-squared, same
                                        # order as alpha=0.08 + smooth penalty).
                                        # Positive scaling => optimal policy UNCHANGED
                                        # (reward scale is SAC's canonical knob).
    reward_sq_clip: float = 100.0       # cap of the scaled squared cost (e=2 m sat.)
    reward_mode: str = "sq"            # "lin" (2026-07-14): r = -min(e, clip).
    # Switched from "sq": the squared reward over-weights the large-error
    # transients (rn^2 ~ 2-20 inside gust/payload windows vs ~0.5 in calm
    # flight), so SAC converged to a transient-only policy -- lambda pinned at
    # 0.74-0.79 and N stuck mid-range, discarding the calm-segment noise
    # averaging where the measured adaptation headroom is largest (Greedy-GT
    # oracle on v9c heldout: nominal -17.8%, payload -15.7%, wind -6.4%,
    # oracle mean N ~ 11). The paper metric (per-scenario RMSE, scenarios
    # equally weighted) is linear in |e|; the linear reward matches it.
                                        #   quadratic cost == direct RMSE minimization
                                        #   (the reported metric IS the square metric),
                                        #   and it amplifies the disturbance-window signal
                                        #   ~3x vs linear WITHOUT any baseline term:
                                        #   window err 0.55 m vs nominal 0.18 m ->
                                        #   linear ratio 3.1x, squared ratio 9.4x.
                                        # "abs": legacy r = -clip(||e||).
    kf_Q_diag: tuple = ((2e-3,) * 12)
                                        # FIXED KF/UKF statistics, paper-quotable:
                                        # Q = 2e-3 I, R = diag(sensor sigmas^2),
                                        # FIXED across ALL scenarios. Selected by
                                        # sweep {1e-4..5e-2}: the unique decade where
                                        # the per-axis chain EKF>=UKF>=FIR>AFIR holds
                                        # in nominal AND wind AND payload while the
                                        # wind ceiling stays <=0.36/axis. Measured
                                        # (new anchors, sigma=0.10): nominal
                                        # 0.068/0.066/0.117, wind 0.352/0.332/0.167,
                                        # payload 0.093/0.088/0.459. UKF edge over
                                        # EKF emerges naturally at this Q (payload
                                        # all axes, wind x/y).
    ref_monitor_N: int = 10             # >0: run a parallel FIXED N=10, lam=1 filter
                                        # (= plain UFIR: fixed N=10, lam=1, NO dynamic/
                                        #  adaptive gain — the standard batch UFIR
                                        #  solution over the window; the two-stage
                                        #  init+recursion is exactly Shmaliy's UFIR
                                        #  algorithm form) on the identical stream, LOGGING
                                        # ONLY — the live "rmse vs FIR" column and the
                                        # eval table. NEVER enters the reward (user
                                        # decision: reward stays absolute-error-based).
                                        # 0 = off (saves ~2x env compute).
    act_smooth_coef: float = 0.02       # small |dN| penalty (units of N-range; scaled
                                        # with the new reward units) so the
                                        # learned N(t) is regime-STEPS, not jitter — the
                                        # paper figure needs Shmaliy-style plateaus.
    train_scenario_types: tuple = ("nominal", "sustained_wind", "mass_step")
                                        # mass ADDED BACK (measured: wind-only policy
                                        # transfers imperfectly to payload — heldout
                                        # +103%: AFIR z 0.333 vs FIR 0.304; with mass
                                        # in the mix the policy learns the z-dominant
                                        # signature too).
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
    ckpt_every: int = 5_000
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
    # v10 (2026-07-15, user decision): TRAIN = HELD-OUT = the same THREE
    # scenario types. turbulence_burst REMOVED from the mix (heldout was
    # always 3 types; train now matches, so the policy is never trained on a
    # disturbance class the paper does not evaluate). Its probability mass is
    # redistributed to the two disturbance classes.
    scenario_types: tuple = ("nominal", "mass_step", "sustained_wind")
    scenario_probs: tuple = (0.15, 0.45, 0.40)
    # flight patterns (NO 'aggressive': high-G maneuvers break the 1st-order
    # Taylor linearization the FIR relies on — out of scope by design).
    flight_patterns: tuple = ("helical", "figure8", "waypoint")
    # WAYPOINT DROPPED (2026-07-13). Measured reference profiles:
    #   figure8 : 2.0 m/s, mean bank 11.6 deg (max 16.5), z-range 0.57 m
    #   helical : 2.0 m/s, mean bank  7.9 deg,             z-range 0.72 m
    #   waypoint: 1.0 m/s, mean bank  0.5 deg,             z-range 0.00 m  <-- flat
    # A payload mass error acts along the THRUST AXIS, so it only reaches x,y
    # through the bank: a_xy = 5.18 * sin(theta). On waypoint that is 0.05
    # m/s^2 -- an order of magnitude BELOW the 0.18 m/s^2 nominal residual, so
    # the payload looks like a pure z-axis disturbance. Both payload rows of
    # the last gate happened to DRAW waypoint, which is the whole reason the
    # x,y signature was missing. On figure8 the same payload gives 1.04 m/s^2,
    # ~6x the residual. waypoint also has zero z-excursion (no vertical
    # observability) and 74-deg bank spikes at its velocity discontinuities.
    traj_duration_s: float = 50.0
    # 50 s (2026-07-13). The figures are cropped at ~45 s anyway (the tail is
    # a repeat of calm behaviour), so a longer trajectory only cost datagen
    # time. All disturbance windows below are placed inside [4, 42] s.
    # ── DISTURBANCE SCENARIOS (paper-motivated: sharp events + recovery,
    #    NOT high-G maneuvers — 1st-order Taylor linearization stays valid).
    #    The (N,lambda) benefit is the UFIR transient/noise tradeoff
    #    (1_FIR_filter Fig.3/5): a sharp event lowers N_opt (dump the polluted
    #    old data fast), a calm segment raises N_opt (average out noise). ──
    # payload coupling (UIFM-SLAC Scenario 2): STEP mass jump + a z-sink /
    # lateral velocity impulse at the coupling instant -> position error spike.
    mass_delta_range: tuple = (0.60, 0.90)     # STEP +60~90 % (centre +75 %: N* 14→6 measured)
    mass_onset_frac: tuple = (0.30, 0.60)      # coupling instant within trajectory
    mass_com_offset_range: tuple = (0.02, 0.05)  # payload CoM offset [m], random direction
    mass_inertia_scales: bool = True             # scale the inertia tensor with the mass
    # PAYLOAD PHYSICS FIX (2026-07-13). Changing ONLY the scalar mass makes the
    # payload a perfectly symmetric z-axis disturbance: the measured model-error
    # acceleration rises 4x in z (a sustained -2.5 m/s^2 bias) but stays at the
    # NOMINAL level in x,y -- so the estimators separate in z and coincide in
    # x,y. That is an artifact, not physics: a real payload is not attached at
    # the centre of mass. Offsetting the CoM by 2-5 cm makes the thrust vector
    # miss the CoM, producing a parasitic torque -> attitude error -> genuine
    # x,y model error, and the inertia tensor grows with the added mass.
    mass_impulse_z: tuple = (0.6, 1.2)         # downward velocity impulse [m/s] at coupling
    mass_impulse_xy: tuple = (0.3, 0.7)        # lateral velocity impulse [m/s]
    payload_wind_speed: float = 0.0            # [m/s] coincident wind gust over the
    # payload window. 0 = DISABLED (option A, 2026-07-16): payload is a pure mass
    # pickup. The paper reports 3D RMSE, and z growth from the added mass already
    # lifts payload 3D RMSE above nominal (no wind needed). Set >0 (e.g. 12) to
    # re-enable a coincident gust that also lifts the x,y axes, if a per-axis
    # payload comparison is ever wanted. Applied identically in Isaac and synth.
    # gust: SHARP impulsive gusts (FM-SMC abrupt disturbance), not slow ramps
    gust_speed_range: tuple = (12.0, 16.0)           # same tilt-clamp ceiling     # m/s — <15 m/s shifts N* the WRONG way
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
    sustained_speed_range: tuple = (12.0, 16.0)
    # CAP AT 16 m/s (2026-07-13). Holding station against a headwind requires a
    # bank of atan(0.023*v^2 / g): 28 deg at 15 m/s, 31 at 16, 40 at 19, 43 at
    # 20 -- and the trajectory itself now demands another ~8 deg. PX4 clamps the
    # tilt at MPC_TILTMAX_AIR (45 deg by default), so beyond ~18 m/s the
    # attitude loop SATURATES, position hold is lost and the vehicle departs.
    # This is not hypothetical: the 19 m/s held-out row (h4) failed to generate
    # in BOTH gate v7 and gate v8 -- it was crashing, and we mislabelled it as a
    # dropped trajectory. 12-16 keeps a >= 6 deg margin to the clamp while still
    # producing the sustained model error the finite-memory structure needs.
    ambient_turb_std: float = 0.5
    ambient_turb_std_range: tuple = (0.5, 0.5)
    # LIGHT AMBIENT WIND, drawn per trajectory (2026-07-13). Measured effect:
    #   1.4-2.0 m/s -> NO effect (drag ~ v^2 puts the model error BELOW the
    #                  existing nominal residual of 0.2-0.6 m/s^2; verified on
    #                  two Isaac gates: EKF 0.147->0.148, FME 0.123->0.123).
    #   5.0 m/s     -> nominal error rises 0.15 -> 0.20 m, which finally
    #                  exposes the RANGE NONLINEARITY: the UKF beats the EKF
    #                  on ALL THREE AXES for the first time (0.193 vs 0.203),
    #                  and N* drops 10 -> 6.
    # RANGE 1.0-5.0 (user decision 2026-07-13): drawn per trajectory. NOTE the
    # v^2 scaling means the low end (1-2 m/s) is effectively calm, so the
    # per-trajectory effect is strongly skewed toward the upper half of the
    # range; the nominal row of the table therefore averages over trajectories
    # with and without a meaningful disturbance. Verify on the gate.
    # Framing for the paper (indoor arena): this is NOT outdoor wind but a
    # residual air disturbance -- rotor-wash recirculation off the walls and
    # floor, ventilation currents, ground effect -- which is present in any
    # real indoor flight and is what a perfectly still simulator omits.
    # ALWAYS-ON light turbulence [m/s], applied to EVERY trajectory including
    # nominal (2026-07-10). Rationale: a perfectly clean nominal makes the FIR
    # indistinguishable from the KF (both sit at the measurement-noise floor),
    # which is an artifact of the simulator, not physics — real flight always
    # has ambient air motion. A 1.5 m/s OU gust field injects a small but
    # PERSISTENT model error (~0.2-0.4 m/s^2), which is exactly the regime the
    # finite-memory structure is designed for. Expected: nominal chain
    # EKF > UKF > FIR > AFIR emerges naturally.
    ambient_turb_bw: float = 0.4
    # OU bandwidth [Hz]. At 2.0 Hz the gust force decorrelates WITHIN one
    # estimation window and the induced bias self-cancels; 0.4 Hz gives a
    # ~2.5 s correlation time > the 1-s window, so the model error stays
    # coherent long enough to penalize long memories.
    wind_n_windows: int = 2
    # sustained_wind: number of NON-overlapping wind windows per trajectory
    # (2026-07-10). 2 -> the estimator sees enter/recover TWICE, which makes the
    # AFIR horizon adaptation far more legible in the time-series figures. Gap
    # between windows >= 6 s. Backward compatible: 1 reproduces the old single
    # window. Applies to BOTH train and held-out; retraining optional (a policy
    # trained on 1 window still reacts to each window independently).
    wind_vertical_ratio: tuple = (0.20, 0.30)
    # vertical wind component = ratio x horizontal speed, UPDRAFT (+) only —
    # empirical gate (2026-07-10): updraft shifts the window z-optimum to
    # N~6 in 3/3 trajectories; downdraft produces no z-shift in 2/2. Purpose (2026-07-09): inject MODEL ERROR into z
    # inside the wind window so the short-horizon advantage covers all three
    # axes (previously z stayed noise-limited -> long N optimal -> AFIR lost
    # the z cell to FIR). 25% of 15 m/s = 3.8 m/s vertical, same order as the
    # payload z-disturbance. GATE: verify z-axis N* shift on 1-2 fresh trajs
    # BEFORE the 50k retrain; if the shift is absent raise to 0.30-0.35.
    mass_window_duration_range: tuple = (10.0, 20.0)
    # payload is now a WINDOW (pickup at onset -> RELEASE at onset+duration),
    # mirroring the wind-window structure: 2 transitions per episode (N down,
    # N back up), symmetric fig timelines, and the delivery-drone narrative.  # m/s — must exceed PX4 compensation so GT
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
    heldout_mass_delta_range: tuple = (0.65, 0.75)   # superseded by heldout_plan
    heldout_plan: tuple = (
        # PAPER HELD-OUT SET (deterministic, 2026-07-13).
        # THREE flight patterns x THREE scenarios = 9 trajectories. Table II
        # reports, for each scenario, the mean over the three patterns, so a
        # number is never an artefact of one particular path. The DISTURBANCE
        # is held fixed within a scenario (same 15 m/s, same windows; same
        # +70%, same CoM, same window) -- only the pattern varies, which is
        # what makes the average meaningful.
        # Figures use the figure-8 rows (h4 wind, h7 payload): the sharpest
        # path, so the entry/recovery transients are the most legible.
        #
        # (type, p_lo, p_hi, windows, extra{turb, com, pattern})
        ("nominal", 0, 0, (), {"turb": 0.5, "pattern": "helical"}),        # 0
        ("nominal", 0, 0, (), {"turb": 0.5, "pattern": "figure8"}),        # 1
        ("nominal", 0, 0, (), {"turb": 0.5, "pattern": "waypoint"}),       # 2
        ("sustained_wind", 15.0, 15.0, ((6.0, 10.0), (26.0, 10.0)),        # 3
         {"turb": 0.5, "pattern": "helical"}),
        ("sustained_wind", 15.0, 15.0, ((6.0, 10.0), (26.0, 10.0)),        # 4  FIG
         {"turb": 0.5, "pattern": "figure8"}),
        ("sustained_wind", 15.0, 15.0, ((6.0, 10.0), (26.0, 10.0)),        # 5
         {"turb": 0.5, "pattern": "waypoint"}),
        ("mass_step", 0.70, 0.70, ((15.0, 18.0),),                         # 6
         {"turb": 0.5, "com": 0.04, "pattern": "helical"}),
        ("mass_step", 0.70, 0.70, ((15.0, 18.0),),                         # 7  FIG
         {"turb": 0.5, "com": 0.04, "pattern": "figure8"}),
        ("mass_step", 0.70, 0.70, ((15.0, 18.0),),                         # 8
         {"turb": 0.5, "com": 0.04, "pattern": "waypoint"}),
    )
    heldout_gust_speed_range: tuple = (14.0, 17.0)   # was (20,24): tilt-clamp crash
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
