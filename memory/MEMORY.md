# Memory index

- [Adaptation lever is measurement-side](adaptation-lever-is-measurement-side.md) — (N,λ) lever lives in measurement degradation (dropout/outliers/noise), not plant kicks; C dropped; IIR<FIR<AFIR chain needs harsher measurement-side faults + a fixed EKF baseline.
- [Core recovered from session logs](core-recovered-from-session-logs.md) — "Add files via upload" reverted core to an older API; recover by replaying prior-session Edit chains from ~/.claude/projects/*.jsonl.
- [Phase 0 NO-GO: no N lever](phase0-nogo-no-N-lever.md) — at σ_LoS=0.12 with 4-range instantaneous position observability, N_opt pinned at N_max in every regime; LEARNABLE%≈0%; do not proceed to Phase 1 as-is.
