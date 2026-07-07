---
name: phase0-nogo-no-N-lever
description: Phase 0 NO-GO — at σ_LoS=0.12 with 4-range instantaneous position observability, N_opt is pinned at N_max in every regime; LEARNABLE% ≈ 0%.
metadata:
  type: project
---

Phase 0 (2026-07-07) verdict: **NO-GO on the (N,λ) adaptation framing.** Do not
proceed to Phase 1 (physics upgrade) as-is.

Evidence (data_diag, σ_LoS=0.12, N∈[8,20], filter/wfme.py untouched, T1–T5 PASS):
- Nominal fixed-N RMSE U-curve is MONOTONE-decreasing → N_opt = N_max = 20 at
  every physically valid process-noise q (1.5–10); the q-knob barely moves it.
- EVERY regime window (turbulence_burst / nlos_burst / anchor_dropout) also
  bottoms at N=20 — no leftward shift (results/calib/ucurve_shift.png).
- LEARNABLE% ≈ 0% all scenarios: best-fixed = regime-oracle = (N=20, λ=1). The
  17–29% "luck%" is unrealizable per-step greedy noise, not a learnable signal.
- Nominal EKF=0.084 vs FIR14=0.192 (EKF wins 2.3×); EKF also wins the dropout
  window; corr(innovation energy, N*) ≈ 0.

**Why:** 4 UWB ranges make 3-D position instantaneously OVER-observable each
epoch, so every regime is variance-limited (more averaging = lower position
RMSE) and N_opt sits at the ceiling. A finite N_opt (true-U bottom=14) appears
only in a bias-limited corner (σ≈0.03, q≈10) — unreachable at spec σ=0.12
without unphysical q≈40 (plant diverges by q≈50). This is why the ROADMAP warns
"q 노브로도 안 갈리면 물리 효과를 넣어도 안 갈릴 가능성이 크다": Phase-1 physics
(drag/rotor-lag/mass) are also dynamics-side model errors and won't defeat
instantaneous position observability.

**How to apply:** the (N,λ)-lever thesis needs position to be WEAKLY observable
(fewer anchors, higher σ, or scoring weakly-observable states) to create
differential N_opt. Otherwise pivot framing to auto-tuning / recovery-speed /
robustness (ROADMAP GO-criterion parenthetical). See
[[adaptation-lever-is-measurement-side]], [[core-recovered-from-session-logs]].
