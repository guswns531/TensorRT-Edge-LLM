# 306. Dynamic encoder preparation and E-to-D transition screen

Date: 2026-09-15. Branch: `codex/v0101-phase-forward-port`.
Parent commit: `a516dcdc194b44ea7cddd2ae08c84f1f1b2d4ca3` with the retained working-tree implementation.
Status: **shadow mechanism validated; active six-workload promotion failed; full12 was intentionally not run**.

## 1. Scope and contract

This experiment asks whether the physical E4 capability can remain fixed while the runtime chooses a smaller
logical encoder preparation cohort from current CUDA cost evidence. It does not change the engine, quantization,
KV pool, stable page ownership, prefill chunk, decode batch capability, request trace, or explicit SLO contract.
The selector never sees a model name or workload label.

The implementation is opt-in through
`TRT_EDGELLM_MEASUREMENT_ENCODER_PREPARATION_POLICY=shadow|active`. `shadow` computes and records a choice while
retaining the original batch. `active` applies that choice at a drained measurement boundary. The default remains
disabled. The fixed-E reference is note305.

The canonical build uses the TensorRT 26.06 image
`sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3`. The source was rebuilt after removing
two ineffective diagnostic guards. Final identities are:

- smoke binary: `de7b65a5482ae8476d1156f6330aff238fbd8114ad6a10b45dcca8393fba001b`
- runtime unit-test binary: `7f9e56d5695ad4fa61c81f139d0be7ac42d86ce0d30ed97360764d3caf652986`
- plugin: `76e79540a4fb5d148c0c0e4f245e91608a173066ef779b1e05fd3c41152f5447`

The performance runs preceded that final rebuild but correspond to the retained preparation selector behavior.
The removed guards produced zero decisions and therefore did not alter those runs' logical policy.

## 2. Implementation

### 2.1 Policy boundary

`PhaseThreeCoordinator` exposes three preparation modes:

```text
disabled  -> existing physical E4 candidate
shadow    -> evaluate E1/E2/E4; dispatch existing candidate
active    -> evaluate E1/E2/E4; dispatch selected prefix
```

The mode can change only after every request and GPU consumer drains. It cannot be enabled together with the old
static cost-aware encoder batching owner. This preserves one policy authority.

### 2.2 Observable inputs

For each compatible FIFO encoder frontier, the coordinator:

1. enumerates power-of-two batches plus the deployment limit;
2. gets an encoder estimate from the runtime CUDA cost tracker, using covering/interpolated estimates only when an
   exact entry is unavailable;
3. gets the external prefill drain estimate for the same rows and prompt size;
4. minimizes the robust E-plus-P drain over the complete current frontier;
5. records coverage, selected rows, predicted drain, selection changes, and applied changes.

The policy has no per-model constants. Engine batch limits are capabilities. Runtime CUDA observations are physical
evidence rather than a workload profile.

### 2.3 Same-frontier cache

The first shadow implementation reevaluated an unchanged frontier on every CPU polling iteration. Cosmos
multi-image produced 1,473 evaluations for only a few physical frontiers. The retained implementation keys the
choice by canonical pending request IDs and reuses it until that frontier changes. The same workload then produced
three evaluations. This fixes scheduler hot-path amplification without changing a choice.

## 3. Verification

- relevant pre-commit hooks: pass
- Python analysis and policy tests: 17/17 pass
- targeted global-scheduler C++ tests: 45/45 pass
- final full runtime suite: 657 pass and two environment skips
- final source after diagnostic-guard removal: smoke and runtime unit target rebuilt successfully
- `git diff --check`: pass

The trajectory analyzer now reads both retained layouts: current `run-001/gateway.log[.gz]` plus
`activity-intervals.csv`, and the older `activity/run-N-events.jsonl` layout.

## 4. Cached shadow result

Each cell is one real HTTP trace run after generic model/GPU calibration. Latencies are milliseconds. The fixed-E
comparison is a prior retained run, so small differences are not paired significance claims.

| Model | Workload | token/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | Distinct frontiers | Shadow choice |
|---|---|---:|---:|---:|---:|---:|---|
| Gemma | mixed | 717.66 | 264.97 / 585.32 | 25.61 / 34.66 | 1448.98 / 2180.22 | 8 | E2 dominant |
| Gemma | vision-heavy | 545.89 | 383.63 / 663.23 | 31.36 / 43.61 | 1559.98 / 2141.83 | 17 | E2 dominant |
| Gemma | multi-image | 391.53 | 440.73 / 720.14 | 23.89 / 39.53 | 1181.23 / 1478.38 | 7 | E2/E3 |
| Cosmos | mixed | 1160.18 | 753.79 / 2029.81 | 35.21 / 60.04 | 2376.42 / 2497.76 | 11 | E1--E4 |
| Cosmos | vision-heavy | 723.73 | 1553.85 / 3002.38 | 41.48 / 81.81 | 3184.32 / 3360.38 | 16 | E1--E4 |
| Cosmos | multi-image | 302.80 | 274.81 / 314.83 | 8.04 / 9.87 | 524.01 / 528.23 | 3 | E1 |

The shadow result is useful: it independently recovers the small-E opportunity previously found by fixed-E
experiments. It does not establish that every proposed choice is safe to activate.

## 5. Active screen and promotion decision

| Model | Workload | fixed token/s | active token/s | delta | TTFT mean delta | TPOT mean delta | E2E mean delta |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma | mixed | 718.54 | 725.77 | +1.0% | +0.7% | -0.6% | -0.6% |
| Gemma | vision-heavy | 546.17 | 543.39 | -0.5% | +0.3% | +0.8% | +1.0% |
| Gemma | multi-image | 382.75 | 368.50 | -3.7% | +3.1% | +4.7% | +3.4% |
| Cosmos | mixed | 1173.31 | 1151.01 | -1.9% | +2.8% | +3.9% | +0.8% |
| Cosmos | vision-heavy | 724.71 | 705.36 | -2.7% | +4.7% | +8.1% | +3.7% |
| Cosmos | multi-image | 299.26 | 313.00 | **+4.6%** | **-7.2%** | **-3.6%** | **-6.9%** |

Negative latency deltas are improvements. Cosmos multi-image is a real positive result, but the policy does not pass
the six-workload gate. It roughly doubled encoder dispatches in mixed/heavy cases and regressed their downstream
decode service. It is therefore not promoted, and running full12 would spend GPU time without satisfying the
predeclared screening gate.

The external serving contract did not change, so vLLM was not rerun. Against the frozen references in note305,
active Cosmos remains ahead in token throughput by about 24.7% mixed, 22.2% vision-heavy, and 28.3% multi-image,
with lower reported TTFT/TPOT/E2E means and tails. Gemma is not uniformly ahead: active is about +3.1% mixed but
-2.9% vision-heavy and -3.4% multi-image in token throughput, and several Gemma TTFT tails remain worse. These are
system comparisons with different runtimes, not estimator-only causal comparisons.

## 6. Why present-D protection cannot repair this failure

Two diagnostic variants attempted to protect decode service using, respectively, the existing no-SLO service-age
signal and the concrete queued/in-flight/sampling D set. Both guards fired zero times.

This is not an implementation mystery. The preparation decision occurs before the chosen E work has completed:

```text
E preparation decision
        |
        v
E completion -> P ready -> P completion -> D ready
```

At the decision boundary, the harmed D cohort often does not yet exist. A guard that asks whether D is resident can
never observe that future transition. The diagnostic guards were removed rather than retained as inert complexity.

The result directly supports execution--formation coupling: the E decision changes the release order into P and D,
which changes later cohort shape and dispatch count.

## 7. Fixed mechanism trajectory evidence

The repaired analyzer replays retained fixed E1/E2/Full runs under the same trace and engine contracts. This is
causal fixed-mechanism evidence, but it is not a GPU checkpoint replay: TensorRT/CUDA tensor and context state cannot
currently be restored at an arbitrary branch.

### 7.1 Cosmos multi-image

| E mode | E dispatches | P dispatches | D dispatches | First ready rows | First D batch | First-ready to full D cohort | token/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 run 1 | 5 | 5 | 32 | 4 | 4 | about 203 ms | 310.69 |
| E1 run 2 | 5 | 5 | 32 | 4 | 4 | about 205 ms | 310.80 |
| E2 run 1 | 3 | 4 | 33 | 1 | 1 | about 170 ms | 309.04 |
| E2 run 2 | 3 | 4 | 33 | 1 | 1 | about 170 ms | 311.38 |
| Full run 1 | 2 | 4 | 34 | 1 | 1 | about 115 ms | 296.10 |
| Full run 2 | 2 | 4 | 33 | 1 | 1 | about 124 ms | 301.48 |

E1 does more encoder work, but the downstream scheduler holds the first decode until four rows are ready. That saves
one or two full decode turns and improves the request path. A predictor that only minimizes E-plus-P drain cannot
represent this result.

### 7.2 Gemma scaled multi-image

| E mode | E dispatch range | P dispatch range | D dispatch range | token/s range |
|---|---:|---:|---:|---:|
| E1 | 20 | 31--33 | 53--56 | 361.5--368.4 |
| E2 | 12 | 26--28 | 47--58 | 374.5--393.9 |
| Full | 8--9 | 27--32 | 42--45 | 374.1--381.3 |

E2 wins one repeat, but its dispatch range and output trajectory are not stable enough to promote. This also shows
why a model-name rule such as `Cosmos -> E1` or `Gemma -> E2` would be a poor design.

## 8. Cosmos completion decomposition

The existing timeline already records decode done, sampling submit, sampling event ready, sampling collected, and
token committed. Offline pairing shows sampling submit and decode done are within about one microsecond; the small
negative delta reflects host record ordering rather than meaningful GPU work. The dominant variable interval is
sampling-submit to sampling-event-ready:

| Variant | Workload | submit-to-ready mean | p95 | ready-to-collect mean | collect-to-commit mean |
|---|---|---:|---:|---:|---:|
| shadow | mixed | 3.086 ms | 5.104 ms | 0.0014 ms | 0.058 ms |
| shadow | vision-heavy | 2.409 ms | 3.694 ms | 0.0011 ms | 0.069 ms |
| shadow | multi-image | 0.138 ms | 0.247 ms | 0.0005 ms | 0.018 ms |
| active | mixed | 1.558 ms | 2.365 ms | 0.0014 ms | 0.063 ms |
| active | vision-heavy | 5.931 ms | 8.795 ms | 0.0013 ms | 0.816 ms |
| active | multi-image | 0.334 ms | 1.034 ms | 0.0005 ms | 0.020 ms |

The client JSON and state-commit path are not the dominant delay, even though active vision-heavy also has a visible
0.816 ms collect-to-commit mean. Submit-to-ready still combines sampling GPU execution, GPU queueing/contention,
event visibility, and the polling interval, so it does not prove that the sampler kernel alone is slow. Kernel-level
attribution needs a sampling-stream CUDA interval or Nsight trace.

## 9. Correctness status

Gemma exact-output determinism remains a separate blocker. The fixed-E reference itself varies across repeats, and
E1/E2 change the number of differing request outputs; the active preparation screen therefore cannot attribute those
differences to its policy. Existing Gemma BS1 P128/P512 controlled runs are greedy-token exact, while concurrent
HTTP traces are not. Same-snapshot Gemma logit/row/ownership localization has not been completed.

A separate Cosmos diagnostic did localize one repeated divergence to the vision encoder's final embedding: fixed
embedding replay made the tested downstream logits and tokens identical. That evidence must not be relabeled as a
Gemma result or as a proof for every stable-KV access.

## 10. Architecture decision

Retain:

- physical E4 capability;
- cached shadow preparation evaluation;
- runtime CUDA E and P estimates;
- deterministic request-DAG/formation primitives;
- stable KV and vision ownership;
- disabled-by-default active authority.

Do not retain or add:

- model/workload labels;
- a present-D guard at the pre-E boundary;
- a static `E1 when small workload` rule;
- inactive diagnostic recovery knobs;
- full12 performance claims from a failed six-workload gate.

## 11. Next implementation gate

The next candidate must evaluate a bounded observable transition rather than E-plus-P drain alone:

```text
candidate E prefix
    -> robust physical E completion
    -> deterministic P-ready transition
    -> robust P completion
    -> D-ready cohort formation
    -> one successor D service boundary
```

Only request state already present in the snapshot and already-outstanding events may be used. External arrivals are
not predicted. The implementation should enter shadow first and must show all of the following before active use:

1. chooses E1 for the retained Cosmos multi-image frontier where D4 formation is observed;
2. avoids the E1/E2 fragmentation seen in Cosmos mixed and vision-heavy;
3. preserves Gemma fixed-E results within the existing three-percent screen;
4. adds less than 50 microseconds p95 scheduler cost;
5. has identical canonical request membership and row order in shadow mode;
6. passes the six-workload gate before a full12 run.

This is the smallest justified next step. The current experiment proves both the opportunity and why an immediate
E-plus-P optimizer is insufficient; it does not justify a larger learned policy or workload-specific tuning.
