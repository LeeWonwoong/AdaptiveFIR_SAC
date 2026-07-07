# Memory index

- [Adaptation lever is measurement-side](adaptation-lever-is-measurement-side.md) — (N,λ) lever lives in measurement degradation (dropout/outliers/noise), not plant kicks; C dropped; IIR<FIR<AFIR chain needs harsher measurement-side faults + a fixed EKF baseline.
- [Core recovered from session logs](core-recovered-from-session-logs.md) — "Add files via upload" reverted core to an older API; recover by replaying prior-session Edit chains from ~/.claude/projects/*.jsonl.
- [Phase 0 NO-GO: no N lever](phase0-nogo-no-N-lever.md) — at σ_LoS=0.12 with 4-range instantaneous position observability, N_opt pinned at N_max in every regime; LEARNABLE%≈0%; do not proceed to Phase 1 as-is.
- [Phase 0 GM-bias hunt END](phase0-gm-bias-hunt-end.md) — last mechanism (time-correlated OU measurement bias) NO-GO; OU unpins N_opt only under ALL-anchor corruption (A-1 N_opt=10), physical single-anchor NLoS stays over-observable (LEARNABLE%≈0); mechanism hunt ended, pivot to path B.
- [Phase 0 tag-common-mode FINAL NO-GO](phase0-tag-commonmode-final-nogo.md) — pre-registered last test (tag-side colored noise, common & per-anchor-independent) NO-GO both modes; IIR<FIR fails in dynamic (EKF always wins a smoothly time-correlated bias); noise-model hunt COMPLETE → decide problem-redefinition (≤3 anchors/GPS-denied) vs turbulence-only framing.
