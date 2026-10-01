---
id: W003
status: waiting
updated: 2026-10-01
notes: [247, 248, 249, 250, 251, 263]
---

# Vision encoder numerics, memory, and compile boundary

## Goal
Localize the Cosmos multi-image/VLM output instability and the vision-engine memory excess versus the old direct engine,
and find a vision engine that is stable, exact, low-memory and non-regressing.

## Current state
- Variation is in the encoder payload, not downstream: freezing the final encoder embedding removes logit divergence for the
  affected request (note 247). A control FP16 vision engine is stable across E batch; FP32 merger/norms are not required (note 248).
- Qwen3-VL merger activation is tanh in upstream v0.10.0/v0.10.1 but exact GELU in the HF reference; merger-only exact GELU makes
  multi-image repeats hash-stable but costs ~120 MiB TRT memory (note 249).
- Explicit FP32 erf decomposition fuses into fc1 and recovers the 120 MiB (548.3 -> 428.3 MiB) with exact math (note 250). Isolated E timing
  is 1.2-4.8% faster (note 251). Under the corrected Release/calibration contract it is not uniformly non-regressing (multi-image -2.22%,
  vision-heavy TPOT worse) and fixed-length identity vs native fails 3/5 although through-EOS matches (note 263). Faster E shifts D cohorts
  (vision-heavy 24.0 -> 21.6). Not promoted.
- Compile boundary (note 263): old direct engine 444,873,728 B / 368 layers; native exact ONNX 574,899,200 B / 392 layers.
- Default retained: the native exact-GELU engine; Cosmos primary vision engine is `vision-exact-gelu` (note 331, rebuilt with Release `visual_build`, note 273).
- Bounded-synchronization memcheck passes on multi-image serving with 0 errors; the unbounded sanitizer still crashes with host signal 11 (note 247).
- Fresh equal-cap (64) vLLM on four text traces: earlier latency win does not hold; Current throughput -3.82% balanced, -0.42% decode-heavy, -2.42% bimodal, +8.85% long-prefill (note 247).

## Conclusions
- 247 — Encoder payload localized as the numerical source; bounded memcheck clean; equal-cap vLLM throughput -3.8% to +8.9% across four text traces.
- 248 — Control vision engine stable across E batch; control full12 -2.05% vs prior (not promoted); compact-telemetry section left unfilled (see 271/272).
- 249 — Merger GELU tanh vs exact origin; exact-GELU engine hash-stable, +120 MiB.
- 250 — FP32 erf decomposition recovers the 120 MiB; mixed four-case serving, not promoted.
- 251 — E isolation: erf faster in isolation, +1.52% six-run crossover; formation coupling; lower-memory candidate only (corrected by 263 under valid contract).
- 263 — Under corrected contract erf engine not uniformly non-regressing; fixed-length identity fails 3/5; not promoted.

## Open questions
- Non-regressing erf (or equivalent 368-layer) candidate on four cases then full 12x3.
- Unrestricted E/P/D sanitizer (signal 11) and cross-shape exact identity.
- Controlled resident-D continuity experiment under E/P while preserving first-token progress -> W014; global D-continuity rule tried and removed in note 303 (W008), terminal cost not run (W005).
- Why status is waiting: later items kept the native exact-GELU engine and did not return to the memory recovery.

## Artifacts
- `.local/current/cosmos/vision-engine` (present; resolves to `vision-exact-gelu/engine/visual`)
- `.local/results/v0101-forward-port/closure-gates` (present)
- `.local/results/v0101-forward-port/vision-precision-20260908` (present)
- `.local/results/v0101-forward-port/erf-merger-20260908` (present)
- `.local/results/v0101-forward-port/encoder-isolation-20260908` (present)
- `.local/results/v0101-forward-port/vision-recovery-20260909` (present)
- `.local/artifacts/v0101-forward-port/cosmos-reason2-2b/vision-canonical-release` (present)
- `.local/scratch/vision-erf-fp32-20260908` (deleted; note 370)
