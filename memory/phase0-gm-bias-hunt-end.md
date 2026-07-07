---
name: phase0-gm-bias-hunt-end
description: Phase-0 last mechanism (time-correlated OU measurement bias) → NO-GO; mechanism hunt ENDED. OU unpins N_opt only under ALL-anchor corruption; physical single-anchor NLoS stays over-observable → LEARNABLE%≈0.
metadata:
  type: project
---

Phase-0 FINAL mechanism (2026-07-08): **time-correlated (Gauss-Markov / OU)
per-anchor measurement bias** — the last untried measurement-side lever, since
white noise always averages √N (root cause of [[phase0-nogo-no-N-lever]]).
Discrete OU b[k+1]=(1-dt/τ)b[k]+N(0,σ_w), zero-mean, stationary std σ_b,
correlation time τ. Added `rlenv.dataset.gauss_markov_bias` + config `gm_bias`/
`los_bias_std=0.04`/`los_bias_tau=3.0`/`nlos_bias_std=0.22`/`nlos_bias_tau=0.5`;
REPLACES the old within-burst CONSTANT NLoS bias. `filter/wfme.py` untouched,
T1–T5 PASS. Tools: `tools/calib_gm_ucurve.py` (A-1), `tools/ucurve_shift.py`
(+recovery-tail, A-2), `tools/gm_decomp.py` (3-label decomp + rolling-mean corr
+ recovery time, A-3).

**VERDICT: NO-GO — mechanism hunt ENDED.** GO gate was LEARNABLE%≥10% (combined)
OR recovery-tail alone ≥20%; got ~0% on every axis.

Evidence (data_diag, σ_LoS=0.12, N∈[8,20]):
- **A-1 (all-anchor nominal OU) — the mechanism IS real:** white ref N_opt=20
  (monotone); σ_b=0.20, τ=0.4s applied to ALL 4 anchors → INTERNAL true-U floor
  at **N_opt=10**; smaller τ pushes N_opt to the N=8 edge. So time-correlated
  bias DOES break the √N averaging gain and unpin N_opt — WHEN it corrupts every
  anchor. (σ_b=0.04 LoS is far too weak/slow to bend it — needs σ_b ≳ 1.5×σ_white.)
- **A-2/A-3 (physical single-anchor NLoS OU, σ_b=0.22 τ=0.5) — FAILS:** window,
  recovery-tail, AND nominal all bottom at N_opt=20 (no leftward shift). 3-label
  LEARNABLE% = 0.0% everywhere (best-fixed = regime-oracle = (N=20,λ=1) in every
  regime). New corr(rolling-mean ν, N*) ≈ 0 (|r|≤0.11, multiR≤0.20); N*mean=15.1
  identical across nominal/burst/tail. Recovery: EKF beats FIR-best & AFIR in
  BOTH burst (0.245 vs 0.422/0.343) and tail (0.168 vs 0.224/0.177) — no
  recovery-speed lever, and the IIR<FIR chain still fails.

**Why (identical to the Phase-0 wall):** one corrupted anchor among three clean
ones is geometrically REJECTED — 4 UWB ranges keep position over-observable, so
averaging still wins and N_opt pins at the ceiling. The OU only bites when the
correlated bias hits (nearly) ALL anchors at once (A-1). Three independent
mechanisms (constant NLoS bias, plant/process kicks, OU bias) now converge to
the same structural verdict.

**How to apply:** the (N,λ)-adaptation thesis is NOT recoverable under the frozen
4-UWB over-observable spec by any measurement fault that spares ≥3 anchors. It
needs WEAK position observability — fewer anchors, or a whole-body / multi-anchor
correlated NLoS (GPS-denied), which A-1 shows would work but is a DIFFERENT
problem than the frozen spec. Per the branch decision, STOP mechanism hunting and
switch framing (path B). Note recovery-SPEED is NOT a viable path-B lever here
either (EKF wins burst+tail), so path B must change the problem (weaken
observability) or the claim, not just the metric. See
[[phase0-nogo-no-N-lever]], [[adaptation-lever-is-measurement-side]].
