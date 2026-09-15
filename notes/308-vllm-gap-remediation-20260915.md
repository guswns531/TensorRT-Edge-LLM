# 308. vLLM gap decomposition and phase-runtime remediation

Date: 2026-09-15. Branch: `codex/v0101-phase-forward-port`.
Status: **Superseded by the completed three-repeat full-12 result in note 310. GPU execution resumed; the final
campaign combines lifetime encoded admission, CUDA graphs, graph/eager policy-evidence sharing, and final-prefill
sampling refill visibility.**

## 1. Conclusion before changing policy

The retained traces do not show a GPU-idle problem. They show a work-granularity problem.

- E/P/D all-idle time is only 1.0--3.0% in the six activity traces.
- Current launches more phase actions than a unified-iteration runtime because fixed P128 chunks and exact D cohorts
  form separate TensorRT actions.
- Every retained production trace used eager execution: P and D graph entries, captures, and hits are all zero.
- The strongest counterexample is the persistent prefill wavefront. It preserves request continuity, but an
  underfilled live cohort excludes later compatible continuation rows. Disabling wavefront entirely improves some
  traces and seriously regresses another, so global removal is not a valid mechanism change.

The priority is therefore:

```text
useful work per action
    -> repeated-shape launch cost
    -> admission/KV utilization
    -> E-to-P-to-D transition quality
    -> duplicate context/workspace memory
```

This ordering does not add a workload label or a model-specific action rule.

## 2. Retained execution-density evidence

The following values were regenerated from
`.local/results/gemma4-v3-activity-diagnostic-20260913` with the updated analyzer. An action boundary is one emitted
P/D dispatch metric; encoder dispatches are reported separately, so this is a conservative lower bound on all GPU
submission boundaries.

| Workload | All-idle | E/P/D overlap | P+D action boundaries / generated token | CUDA graph hits |
|---|---:|---:|---:|---:|
| balanced | 1.09% | 8.66% | 0.0559 | 0 |
| bimodal | 1.02% | 13.95% | 0.1054 | 0 |
| long-prefill | 1.20% | 32.77% | 0.1112 | 0 |
| mixed | 2.46% | 16.71% | 0.1185 | 0 |
| multi-image | 2.62% | 9.60% | 0.4062 | 0 |
| vision-heavy | 2.99% | 13.41% | 0.2171 | 0 |

Multi-image has 7.3 times as many P/D boundaries per generated token as balanced. More overlap cannot by itself
remove this cost: the GPU is already busy, and overlapping small actions can further fragment the following cohort.

## 3. P-cohort refill mechanism

### 3.1 Behavior

The scheduler now retains the persistent wavefront but fills unused cohort capacity from rows that are already
ready and satisfy the existing mechanism compatibility contract:

```text
live P cohort: request 1 at continuation offset 128, capacity P4
ready queue:   requests 2/3/4 at the same compatible continuation frontier

before: P1(request 1)
after:  P4(requests 1/2/3/4)
```

Compatibility still requires:

- the same text or external-embedding producer class;
- the same initial-versus-continuation stage;
- a compatible packed/ragged chunk shape and token budget;
- the existing exclusivity and page-growth eligibility checks;
- canonical priority and stable row ordering.

The cohort still expires at the existing bounded turn limit. A decode action can still be selected at every P128
completion boundary. This is not multi-chunk monopolization and does not turn P128 into P256/P512.

### 3.2 Configuration and telemetry

- Production `PhaseServingRuntime` enables compatible refill by default.
- The research smoke path uses the same default.
- `TRT_EDGELLM_DISABLE_PREFILL_COHORT_REFILL=1` provides a same-binary baseline.
- Every P dispatch exports `prefill_cohort_size` and `prefill_cohort_refill_rows`.
- The activity analyzer reports refill dispatch count, admitted rows, refill-batch distribution, action density, and
  P/D CUDA graph coverage.

The A/B switch changes only cohort membership. Chunk size, engine profiles, policy model, KV allocator, admission,
request trace, and SLO contract remain identical.

## 4. CUDA graph gate

Graph capture/replay was already implemented for independent P and D contexts, including separate execution-cost
variants. It was not active in the retained campaign. The diagnostic runner now accepts:

```bash
ENABLE_CUDA_GRAPHS=1
```

and emits graph entries, hits, misses, captures, evictions, and hit rate. `BUILD_ROOT`, `MODEL_ROOT`, `TRACE_ROOT`,
`CALIBRATION_TRACE`, `REPEATS`, and `PHASE_POLICY` can be overridden without editing the script.

Graphs must not be credited until a fresh run shows nonzero captures and hits. The first graph experiment keeps
cohort refill enabled because denser repeated P shapes should increase graph reuse. The causal matrix is:

| Cell | Cohort refill | Graphs |
|---|---:|---:|
| mechanism baseline | off | off |
| density only | on | off |
| density plus launch optimization | on | on |

## 5. KV and admission status

KV is not the first unresolved difference.

- The runtime already uses stable paged ownership and logical row remapping.
- Production phase serving already uses headroom reservation instead of full-output reservation.
- It reserves prompt plus 128 output tokens per request and guarantees full tails only for a bounded growth-owner
  cohort sized from D capability.
- The memory-horizon supplier uses allocatable rather than physical pages.
- The related scheduler/memory tests passed in this campaign.

KV capacity can still change admission and batch formation, but changing the allocator again before measuring refill
would confound two mechanisms. The next KV experiment is therefore an unchanged-code budget sweep, not a new cache
layout.

## 6. Encoder transition status

The generic E-to-P-to-D completion/DAG preview is implemented and validated in shadow mode. It is not production
authority. The retained screen established two blockers:

1. one producer boundary cannot credit several visible small-E completions that later form the first useful D
   cohort;
2. uncovered successor-D sizes make the recommendation unstable.

This work remains behind P refill and graph validation. Promoting the incomplete E transition model could improve
one Cosmos multi-image trajectory while regressing mixed or vision-heavy service.

## 7. Verification completed without a GPU

- SM86 TensorRT/CUDA build: `unitTestRuntime` and `llm_phase_context_smoke` passed compilation.
- Focused wavefront/refill/producer-class tests: 4/4 passed.
- Scheduler, memory broker, and three-phase policy tests: 189/189 passed.
- Python analyzer compilation and retained six-workload replay: passed.
- Shell syntax and `git diff --check`: passed.

The full runtime binary contains 662 tests. In the driverless container, 622 passed and two skipped; 38 tests that
allocate CUDA streams or memory failed with `CUDA driver is a stub library`. These are environment failures, not
valid regression results. Host diagnostics show loaded NVIDIA 610.43.02 modules but no `/dev/nvidia*` nodes, and
`nvidia-smi` exits with code 9.

## 8. Required fresh measurement

After GPU device nodes return, use the newly built binary in `.local/builds/v0101-validation` and run the same
request trace three times per cell:

```bash
BUILD_ROOT=.local/builds/v0101-validation \
RESULT_ROOT=.local/results/gemma4-vllm-gap-20260915/baseline \
DISABLE_PREFILL_COHORT_REFILL=1 REPEATS=3 \
CASES="balanced long-prefill bimodal mixed vision-heavy multi-image" \
benchmarks/phase_serving/run_gemma_v3_activity_diagnostic.sh

BUILD_ROOT=.local/builds/v0101-validation \
RESULT_ROOT=.local/results/gemma4-vllm-gap-20260915/refill \
REPEATS=3 CASES="balanced long-prefill bimodal mixed vision-heavy multi-image" \
benchmarks/phase_serving/run_gemma_v3_activity_diagnostic.sh

BUILD_ROOT=.local/builds/v0101-validation \
RESULT_ROOT=.local/results/gemma4-vllm-gap-20260915/refill-graphs \
ENABLE_CUDA_GRAPHS=1 REPEATS=3 \
CASES="balanced long-prefill bimodal mixed vision-heavy multi-image" \
benchmarks/phase_serving/run_gemma_v3_activity_diagnostic.sh
```

Promotion requires all of the following:

- exact greedy token identity against the same-binary baseline;
- lower P dispatch count and higher mean P batch where refill opportunities occur;
- no regression above 3% in throughput, TTFT mean/p95, TPOT mean/p95, or E2E mean/p95;
- nonzero graph hits in the graph cell and a lower host submission gap;
- no action-fidelity, KV ownership, page-growth, or producer-class violations;
- the frozen vLLM row is reused only when its engine, model, trace, client concurrency, and serving contract remain
  unchanged.

## 9. Remaining order

1. Run the three-cell refill/graph matrix and retain a manifest.
2. Expand successful cells to the full 12-workload suite and compare every mean/p95 latency metric with frozen vLLM.
3. Sweep allocatable KV pages with the winning binary; do not change reservation policy in the same experiment.
4. Extend the E transition shadow to multiple visible producer boundaries and a real successor D-refill cohort.
5. Measure shared-weight context/workspace allocations and graph memory before considering context reuse.
6. Run selected Nsight traces to separate kernel service, launch gap, and overlap interference.

The immediate implementation is intentionally narrow: improve phase execution density while preserving stable
ownership and global action semantics. It addresses the strongest retained evidence without introducing a model,
workload, or batch-size lookup rule.
