---
name: phase0-tag-commonmode-final-nogo
description: Phase-0 FINAL pre-registered test (tag-side common-mode colored noise) → NO-GO in BOTH coherent & per-anchor-independent modes; noise-model hunt COMPLETE. IIR<FIR fails in dynamic (EKF always wins); N_opt lever absent/artifactual.
metadata:
  type: project
---

Phase-0 FINAL, pre-registered "real last" test (2026-07-08): **tag-side
common-mode colored measurement noise**, motivated by documented UWB tag errors
(DW1000 RX-power-dependent timestamp shift, tag antenna radiation pattern, clock
drift) that are common across anchors and attitude-coupled — the user's rebuttal
that A-1's all-anchor-OU world is spec-relevant, not out of spec.

Built an integrated pipeline (all committed): `tag_commonmode` scenario with
smooth calm↔dynamic attitude segments (`datagen/scenario.py` `_cm_regime`,
`rlenv/synth.py` `_cm_gain`/`_cm_ref` — dynamic |ω|≈2.4 vs calm ≈1.2), a
measurement-layer common-mode component `range[a] += b_common(k)·s_a + b_a(k)`
(`rlenv/dataset.py`, config `cm_*`), dataset `data_cm`, and `tools/cm_test.py`
running 4 measurements + the pre-registered gate. `filter/wfme.py` untouched,
T1–T5 PASS.

**Pre-registered GATE: LEARNABLE% ≥ 10% AND dynamic IIR<FIR.  RESULT: NO-GO
(both modes fail both conditions).** Noise-model hunt is COMPLETE.

- **cm_mode="common"** (coherent b_common·s_a, cross-anchor corr 0.97 — clock/
  RX-power): dynamic fixed-N U-curve monotone → N_opt=20 (no shift); 3-label
  LEARNABLE%=0.0% (best-fixed=(20,1)=regime-oracle everywhere); dynamic **EKF
  0.318 < FIRbest 0.433 (EKF LEADS)**; common-mode corr(anchor-mean innov,N*)≈0.
  Reason: a coherent range offset on all anchors is geometrically ABSORBED as a
  smooth position/clock offset that any N tracks and the KF handles best.
- **cm_mode="independent"** (per-anchor OU on all anchors in dynamic segments —
  antenna-pattern-vs-bearing, the strongest form of the hypothesis): dynamic
  **EKF 0.425 < FIRbest 0.581 (EKF LEADS)**; FIR14 catastrophically blows up
  (7.6–10.1 — pure-FME no-prior fragility to persistent per-anchor bias); U-curve
  N=20 destabilizes (0.59→1.04) so the mid-range "floor" is a large-N-instability
  artifact, not a noise/staleness tradeoff; ΔN=+2 (not leftward).

**Decisive invariant:** across coherent AND independent tag colored noise, the
IIR<FIR chain FAILS in the dynamic regime — the recursive KF (motion prior +
recursive averaging) handles smoothly time-correlated bias BETTER than the
prior-less finite window. Reconciles with A-1 (`calib_gm_ucurve` per-anchor indep
OU on nominal hover → clean N_opt=10): A-1 measured only the FME U-curve SHAPE in
isolation; it never checked the KF, which still wins on absolute RMSE. So A-1's
"finite N_opt" was real but never implied AFIR>KF.

**How to apply:** the (N,λ)/AFIR-beats-KF thesis does NOT hold under ANY tag-side
colored-noise model in the frozen over-observable 4-UWB position setup. Per the
pre-registered FAIL branch, STOP noise-model hunting entirely. Decision now is the
user's: (1) REDEFINE the problem — ≤3 anchors / GPS-denied so position is weakly
observable (the A-1 finite-N world where the KF might also degrade), or (2) pivot
to a **turbulence-only** framing (process-side fixed-Q KF lag, the one regime that
ever showed in-window IIR<FIR — see [[adaptation-lever-is-measurement-side]]
UPDATE 07c). See [[phase0-gm-bias-hunt-end]], [[phase0-nogo-no-N-lever]].
