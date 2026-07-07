---
name: core-recovered-from-session-logs
description: The newer core API was lost via "Add files via upload" reverting to an older version; recovered by replaying prior-session Edit chains from ~/.claude/projects logs.
metadata:
  type: project
---

On 2026-07-07 the committed core (`config.py`, `rlenv/synth.py`, `datagen/scenario.py`,
`rlenv/dataset.py`, `filter/baselines.py`, plus `evaluate.py`, `rlenv/replay_env.py`)
had reverted to an OLDER API via a "Add files via upload" commit, while the untracked
Phase-0 tools (`tools/adapt_signal.py`, `tools/calib_procnoise.py`) and reference data
`data_diag/` needed the NEWER API (proc-noise q-knob wired into plant, turbulence_burst/
nlos_burst scenarios, NLoS measurement layer `range_bias`/`noise_scale`, `EKF(oracle=)`).

**Why:** the user's "Add files via upload" workflow can overwrite session work with a
stale snapshot, silently desyncing core modules from tools/data.

**How to apply:** recover by mining `~/.claude/projects/-home-acsl-projects-AdaptiveFIR-SAC/*.jsonl`
— parse tool_use `Write`/`Edit` payloads, replay the Edit chain (oldString→newString, in
log order) on the committed file as base. 100% oldString match ⇒ committed tree was the
pre-edit base and the replay is faithful. Then validate: T1–T5 (`python tests/test_wfme.py`),
q-knob live (GT vel std scales with `proc_acc_std`), `adapt_signal` runs on `data_diag`.
Recovered core committed 825fdcf. `filter/wfme.py` must stay untouched (T1–T5 gate).
Calibrated defaults then in config: `proc_acc_std=1.50`, `meas_sigma=0.12`,
`ekf_R_sigma=0.10`, `ekf_Q_scale=0.40`. See [[adaptation-lever-is-measurement-side]].
