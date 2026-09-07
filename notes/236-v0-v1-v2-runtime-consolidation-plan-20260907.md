# V0/V1/V2 Runtime Consolidation Plan

## Scope

The production phase scheduler retains exactly three policy variants over one
shared execution mechanism:

| Variant | Exact CUDA registry | Contextual Scalar RLS | Scalar H=2 transition |
|---|---:|---:|---:|
| V0 Exact | yes | no | no |
| V1 Scalar | yes | yes | no |
| V2 Scalar+T | yes | yes | yes |

Zero-start and generic calibration remain initialization conditions rather than
additional policy variants.  Workload names are never policy inputs.

## Preserved mechanism

The cleanup must not alter independent E/P/D TensorRT contexts, E/P/D/copy CUDA
streams, stable indexed KV/page ownership, vision leases, packed and chunked
prefill, continuous admission, CUDA graphs, memory accounting, request DAG
transitions, CUDA timing, or activity telemetry.

## Removed experiments

The current branch removes Effect-Vector/Decomposed Scalar, Completion-Vector,
completion conformal authority, Oracle H1, selective-fidelity selection, and
their policy telemetry and benchmark drivers.  Their implementation remains
recoverable from branch `codex/v010-policy-experiments-archive` at commit
`4efcdae8fb5abfbe95f2d1b30606701fffb1a9d0`.  Historical result notes remain in
the repository as evidence and are not production dependencies.

## Canonical policy contract

The serving process accepts one policy setting:

```text
TRT_EDGELLM_PHASE_POLICY=exact|scalar|scalar-transition
```

The policy changes only cost authority and bounded transition evaluation.
Candidate formation, feasibility, SLO deadlines, memory constraints, canonical
row order, dispatch leases, and GPU execution are identical across variants.

## Validation contract

Every cleanup stage must pass build and targeted unit tests.  V0/V1/V2 must
preserve candidate membership, request row order, selected action identity, and
greedy output.  Major stages run balanced, long-prefill, 48.8 req/s saturation,
multi-image, and Poisson sentinels.  The final stage runs the complete 12-trace
gate, with a 3% median throughput and latency regression threshold.  A fresh
vLLM comparison is required only for the final binary because the workload and
vLLM implementation do not change during cleanup.

## Frozen baseline

The corrected V2 candidate is the baseline described in
`235-multi-image-slo-contract-and-frozen-branch-results-20260907.md`.  It is
ahead of the previous champion by 2.29% throughput geometric mean across the
12 traces and ahead of the frozen vLLM throughput by 17.99%.  These numbers are
engineering baselines, not final paper claims: most full-suite points are
single runs and the vLLM data is frozen.

