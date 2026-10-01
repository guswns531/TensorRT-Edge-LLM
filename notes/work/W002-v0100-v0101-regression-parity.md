---
id: W002
status: waiting
updated: 2026-10-01
notes: [244, 245, 246, 257, 258, 259, 260, 261, 262, 268, 269, 274, 275]
---

# v0.10.0 vs v0.10.1 regression and parity

## Goal
Attribute and close the throughput/latency gap between the v0.10.0 fork and the v0.10.1 port under an equal
build, engine, calibration and request contract.

## Current state
- Governing result (note 275): with byte-identical TensorRT engines, 12 workloads, 3 repeats, v0.10.1 trails v0.10.0 by
  0.44% (V0), 1.43% (V1), 0.39% (V2) geomean token throughput. Current V1/V2 beat frozen vLLM on 12/12 (+11.52%/+11.71%).
  The per-workload oracle is 1.04% above current V2.
- Memory: 9,399 MiB (v0.10.1) vs 9,581 MiB (v0.10.0), same 3,584 MiB KV pool; old runtime had 2 vision-heavy allocation
  failures in 74 attempts, new 0 in 72 (note 275). Four-case gate: V0 -0.10%, V1 -2.36%, V2 +1.87% (note 274).
- Corrections: the build-type finding (note 259: old binaries Release/-O3, v0.10.1 comparison build empty build type) retracts
  attributions in 244, 257, 258; the calibration-asset finding (note 261: 80 of 319 warmup requests pointed at a deleted
  image path, client counted submissions as successes) invalidates note 260 VLM results and earlier Release VLM comparisons.
  Note 262's 108 validated runs (-4.56%/-2.48%/-2.52% vs old; frozen vLLM +7.8/+11.5/+12.1%) were the first valid
  full12 and are themselves superseded by 275's same-engine comparison.
- Not regressions: KV pool/payload and page-table metadata (BS64 49.3 -> 12.5 us under Release) (notes 259, 262); prebuilt vs JIT XQA
  bit-identical in 36/36 shapes (note 245). Calibration coverage did not block Scalar authority; the 42.6 ms mixed span difference
  is accounted for by less E/P/D overlap (15.6% -> 10.5%), not D fragmentation (notes 268, 269).
- Cross-release greedy hashes differ on decode-heavy, long-prefill, bimodal (note 275); within a release/policy/workload cell
  output is deterministic. Exact cross-engine identity is not closed.

## Conclusions
- 244 — Artifact audit: no single scheduler fault; frozen vLLM used client cap 80 vs 64 in four text workloads (build/calibration confounds found later; partly retracted by 259/261).
- 245 — XQA prebuilt vs JIT bit-identical (36/36 shapes); P graph replay 0 in the one balanced serving run checked; graph-on and canonical adapter ordering not promoted.
- 246 — Graph-miss causes found; step-14 argmax near-tie reproduced; startup memcheck failure traced to undersized deepstack buffer (fixed 79e8b12), post-fix sanitizer run incomplete.
- 257 — Same paged-pool layout old/new; 6-run policy screen (invalid: build without Release) (corrected by 259).
- 258 — Legacy pair eligibility option and KV metadata microbenchmark (unoptimized core) (corrected by 259).
- 259 — Release/-O3 and CUDA arch contract restored; prior old/new attributions retracted.
- 260 — Release runs and 322 tests pass, E workspace +124 MiB; VLM suite invalid (corrected by 261).
- 261 — Deleted-asset warmup requests and unchecked responses fixed; 319/319 validated.
- 262 — 108 validated runs: -4.56/-2.48/-2.52% vs old; old vision engine restores multi-image but fails identity (superseded by 275 for parity numbers).
- 268 — Same-engine mixed n=1: -1.56% throughput, P+D selections 9 -> 3, overlap 15.6% -> 10.5%.
- 269 — Decomposition: less overlap, P favored over P+D under all-late; not calibration or packing.
- 274 — Four-workload gate: V0 parity, V1 -2.36%, V2 +1.87%, memory -182 MiB.
- 275 — Full12 canonical comparison: near release parity, better memory robustness.

## Open questions
- Paired action/membership telemetry for old/new V1 and V2 (bimodal, vision-heavy, poisson, text-heavy, mixed, multi-image, late-vision).
  Not run; later V3 work (W005) changed the policy rather than closing this comparison.
- First cross-release token divergence with identical row order, ownership and bindings, compared at logits (3/12 workloads).
- Allocation ledger for the 182 MiB reduction.
- Why status is waiting: no later item picked up these three diagnostics; they were bypassed by W005 policy work.

## Artifacts
- `.local/results/v0101-forward-port/version-policy-full12-20260909` (present)
- `.local/results/v0101-forward-port/v0101-canonical-fourcase-v0-v1-20260909` (present)
- `.local/results/v0101-forward-port/validated-parity-20260909` (present)
- `.local/results/v0101-forward-port/version-activity-20260909` (present)
- `.local/builds/v0101-release` (present)
- `.local/worktrees/v0100-reference` (present)
- `.local/v010-forward-build` (deleted; note 370, old v0.10.0 Release binary)
- `.local/results/baselines/vllm-frozen-12x3` (dangling symlink, target absent; note 370 cleanup)
