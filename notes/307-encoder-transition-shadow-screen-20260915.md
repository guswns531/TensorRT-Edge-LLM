# 307. Encoder E-to-P-to-D transition shadow

Date: 2026-09-15. Branch: `codex/v0101-phase-forward-port`.
Parent source state: retained dirty research tree after note306.
Status: **transition mechanism and shadow telemetry validated; active authority is not promoted**.

## 1. Question and scope

Note306 showed that choosing an encoder cohort from E-plus-P drain alone can improve Cosmos multi-image but regress
mixed and vision-heavy service. A present-D guard cannot prevent this because the affected decode rows do not exist
until E completes and releases P, and P completes and releases D.

This step therefore implements the smallest explicit transition preview:

```text
candidate E cohort
    -> predicted E completion
    -> deterministic E-to-P DAG transition
    -> predicted P completion
    -> deterministic P-to-D DAG transition
    -> one successor D service boundary
```

The preview uses only requests and outstanding work in the current server snapshot. It does not predict external
arrivals, inspect a model/workload name, change the engine, change the KV/page allocator, change P128, or change the
physical E4/P8/D64 capabilities. This campaign is shadow-only: the existing Full encoder candidate is still
dispatched.

## 2. Implementation

### 2.1 Deterministic transition primitive

`phaseFormationPlanner` now exposes `PhaseEncoderTransitionPreview` and
`phaseFormationPreviewEncoderTransition(...)`. The result contains:

- feasibility and the selected encoder rows;
- existing and successor decode rows;
- projected P-ready, D-ready, and one-boundary D-complete times;
- accumulated uncertainty;
- projected vision/KV release bytes.

The implementation reuses the existing completion-boundary DAG transition code. It validates request ownership,
stage legality, finite physical estimates, and pre-existing decode owners. Formation and ownership transitions are
deterministic; only E/P/D physical durations come from runtime measurement.

### 2.2 Encoder preparation shadow

`PhaseThreeCoordinator` adds `transition-shadow` beside disabled, E-plus-P shadow, and the non-promoted active mode.
For each compatible FIFO encoder frontier it evaluates E1/E2/E4 candidates with:

```text
robust E p95
    + external P-drain estimate
    + one successor D-component p95
```

The existing drain dynamic program compares these candidate transition costs over the visible E frontier. It records
the selected logical encoder size, successor D rows, transition horizon, coverage misses, selection changes, and
applied changes. Applied changes must remain zero in this mode.

### 2.3 Server-state cache correctness

Caching only by pending encoder request IDs was insufficient: the same E frontier can be revisited after P, D, or an
outstanding context changes. The retained cache signature therefore includes:

- pending encoder request IDs;
- busy and in-flight phase kind;
- P request IDs and token counts;
- D request IDs and context lengths;
- in-flight execution and action IDs.

An unchanged polling state is reused, while an actual server-state transition forces reevaluation. This prevents the
1,000-plus repeated evaluations seen in the first diagnostic implementation without freezing a stale E decision.

### 2.4 Public measurement contract

The smoke runtime accepts:

```text
TRT_EDGELLM_MEASUREMENT_ENCODER_PREPARATION_POLICY=transition-shadow
```

It exports the transition-selected E rows, projected successor D rows, projected transition horizon, coverage
misses, selection changes, and applied changes. The lifetime-admission runner and analyzer retain this variant; the
analysis schema is version 6.

## 3. Verification

The retained source passed:

- targeted formation-planner tests: 19/19;
- full `unitTestRuntime`: 659 pass, two environment skips;
- Python admission/analysis contract tests: 14/14;
- relevant C++ and Python pre-commit hooks;
- `git diff --check`.

The two runtime skips are the existing isolated-metadata benchmark and the single-GPU NCCL ownership test. The
canonical build image is
`nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3`.
The campaign manifest records the exact source patch, binary, engine, trace, and command identities.

A fresh paired disabled-versus-shadow Cosmos multi-image run was attempted after this campaign, but both cells
stopped before container startup because host `nvidia-smi` returned driver communication error code 9. This is an
environment failure rather than a serving result. The failed diagnostic is retained at
`.local/results/dynamic-encoder-transition-shadow-parity-20260915`; paired dispatch-hash validation remains open.

## 4. Final six-cell shadow screen

Results are one real HTTP run per cell after generic model/GPU calibration. Latencies are milliseconds. Because
shadow does not apply its choice, differences from the prior three-run Fixed Full median are run-to-run variance,
not transition-policy gain.

| Model | Workload | token/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | E selector evaluations | Suggested E | Coverage misses |
|---|---|---:|---:|---:|---:|---:|---|---:|
| Gemma | mixed | 707.42 | 279.39 / 647.27 | 25.80 / 35.46 | 1467.14 / 2265.66 | 11 | E2 x10, E3 x1 | 0 |
| Gemma | vision-heavy | 549.91 | 392.36 / 634.65 | 30.71 / 42.07 | 1548.81 / 2143.76 | 17 | E2 x16, E3 x1 | 1 |
| Gemma | multi-image | 381.68 | 420.45 / 698.16 | 25.26 / 40.51 | 1203.40 / 1476.71 | 7 | E3 x4, E2 x3 | 3 |
| Cosmos | mixed | 1162.71 | 811.98 / 2018.46 | 33.91 / 62.48 | 2372.24 / 2502.58 | 11 | E3 x8, E2 x2, E4 x1 | 9 |
| Cosmos | vision-heavy | 724.60 | 1618.41 / 3024.71 | 39.90 / 78.26 | 3191.63 / 3348.62 | 14 | E3 x7, E1 x5, E4 x2 | 1 |
| Cosmos | multi-image | 306.91 | 260.44 / 307.10 | 8.26 / 10.14 | 516.59 / 520.96 | 3 | E2 x3 | 0 |

The analyzer reports the following one-boundary projection:

| Model / workload | Projected successor D rows, mean / p95 | Projected transition horizon, mean / p95 |
|---|---:|---:|
| Gemma mixed | 8.0 / 17.5 | 265.92 / 315.39 ms |
| Gemma vision-heavy | 9.69 / 19.25 | 265.20 / 269.90 ms |
| Gemma multi-image | 8.33 / 10.7 | 306.57 / 307.05 ms |
| Cosmos mixed | 4.5 / 5.85 | 299.84 / 299.84 ms |
| Cosmos vision-heavy | 20.62 / 29.4 | 549.17 / 575.80 ms |
| Cosmos multi-image | 2.0 / 2.0 | 423.72 / 423.72 ms |

Coverage distributions exclude failed estimates. Cosmos mixed has only two covered evaluations out of eleven, so
its apparent D-row distribution is not representative of the whole trace.

## 5. Fixed Full and frozen vLLM context

| Model / workload | Fixed Full token/s | Transition shadow token/s | Delta | Frozen vLLM token/s | Shadow vs vLLM |
|---|---:|---:|---:|---:|---:|
| Gemma mixed | 718.54 | 707.42 | -1.55% | 703.81 | +0.51% |
| Gemma vision-heavy | 546.17 | 549.91 | +0.68% | 559.83 | -1.77% |
| Gemma multi-image | 382.75 | 381.68 | -0.28% | 381.34 | +0.09% |
| Cosmos mixed | 1173.31 | 1162.71 | -0.90% | 923.32 | +25.93% |
| Cosmos vision-heavy | 724.71 | 724.60 | -0.02% | 577.19 | +25.54% |
| Cosmos multi-image | 299.26 | 306.91 | +2.56% | 243.90 | +25.84% |

The fixed comparison is a prior three-run median while this shadow screen is one run. The frozen vLLM result is
reused because the external model, trace, and serving contract did not change. Its engine/runtime/memory contract is
different, so this is a system comparison, not estimator causality. Cosmos remains substantially ahead. Gemma is
near throughput parity but is not uniformly ahead in latency: in particular its multi-image TTFT/E2E remain worse
than the frozen vLLM result.

## 6. Actual trajectory remains the Full path

The final Cosmos multi-image run dispatched:

```text
E: E3 -> E2                         2 dispatches
P: P1 -> P1 -> P1 -> P2            4 dispatches
D: D1 -> D3 -> D5 ...               33 dispatches
```

All five requests then remain in D5 for most decode rounds. The shadow selector proposed E2, but applied changes are
zero, so the physical E3/E2 trajectory is unchanged. Exact generated token traces were deterministic in every cell.

This distinction matters: the +2.56% Cosmos multi-image number cannot be attributed to the new selector. It only
shows that shadow computation did not destroy the existing coherent path in that run.

## 7. Findings

### 7.1 The transition abstraction is correct but the horizon is too short

The implementation can project E completion, P release, D release, one D cohort, and ownership release. It cannot
yet represent the mechanism behind the retained Cosmos E1 result:

```text
several small E producer completions
    -> several P releases
    -> downstream D refill wait
    -> first useful D4 cohort
```

A single candidate E cohort followed by one P and one D boundary sees only two successor rows in the final
multi-image snapshot. It therefore prefers E2, although fixed E1 produced four ready decode rows before the first
useful D service in the retained causal run. The missing information is not a learned model coefficient; it is a
missing deterministic producer-frontier transition.

### 7.2 The recommendation is not stable enough for authority

Across three Cosmos multi-image diagnostics, the shadow recommendation changed between E2, E1, and E2. The cache
fix eliminated repeated evaluations of an unchanged state, but it cannot repair sensitivity to physical calibration
and an incomplete horizon. Active use would therefore turn small measurement variation into a different batch
trajectory.

### 7.3 Decode-cost coverage is the second blocker

Cosmos mixed missed covered successor-D cost evidence in nine of eleven evaluations. Larger projected D cohorts can
fall outside the currently covered global-decode component estimate. A profile-free selector must interpolate from
capability-covered physical measurements with explicit uncertainty; it must not silently convert missing coverage
into a confident E decision.

### 7.4 Shadow CPU cost is observable

Scheduler decision p95 was 104--149 microseconds on Gemma and 39--243 microseconds on Cosmos in this one-run screen.
This includes the whole existing decision path, not only the new preview, but several cells exceed the intended
50-microsecond gate. Candidate transition states should be incrementally maintained rather than reconstructing and
hashing request vectors on every changed frontier.

## 8. Architecture decision

Retain:

- the generic deterministic E-to-P-to-D transition primitive;
- explicit transition and coverage telemetry;
- the complete server-state cache signature;
- shadow-only integration;
- stable KV/page and vision ownership as hard feasibility state.

Do not promote:

- `transition-shadow` to active authority;
- the current one-successor-boundary objective;
- a Cosmos-specific E1 rule;
- a workload-specific table;
- a full12 performance campaign before the six-cell gate passes.

## 9. Next gate

The next implementation should extend mechanism, not add a model/workload heuristic:

1. Represent the already-visible encoder producer frontier and its compatible cohorts in `ProjectedState`.
2. Advance multiple currently queued E completions and their P releases until the first useful downstream D-refill
   boundary. Do not predict future external arrivals.
3. Reuse the real decode candidate/refill constructor so E1 can be credited when it deterministically creates D4,
   rather than treating each successor D row as an immediately dispatched singleton.
4. Add capability-bounded interpolation and uncertainty for uncovered successor-D sizes. A missing estimate keeps
   the candidate shadow-ineligible rather than selecting it through fallback.
5. Cache aggregate transition state incrementally and require scheduler p95 below 50 microseconds.
6. Repeat Cosmos multi-image at least three times. The selector must consistently recover the coherent E1/D4
   opportunity and must avoid small-E fragmentation in mixed and vision-heavy.
7. Verify paired shadow dispatch hashes: action type, request membership/order, E/P/D shape, and requested start skew
   must be identical to disabled mode.
8. Only then run an active six-cell 3x screen; expand to full12 only if throughput and TTFT/TPOT/E2E gates pass.

The result is therefore a useful intermediate success: the scheduler now has an explicit, tested physical-to-DAG
transition interface. The experiment also demonstrates why one E-to-P-to-D boundary is not sufficient to control
encoder granularity. The remaining requirement is a bounded multi-producer/refill transition, not more
workload-specific tuning.
