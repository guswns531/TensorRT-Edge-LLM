# M2: Directional injection benchmark and Gate A

Date: 2026-09-01
Status: complete
Scope: M2 mechanism and characterization only; production action selection remains unchanged

## 1. Outcome

M2 implements a research-only directional launch controller, records incumbent/newcomer completion vectors in one
CUDA epoch, and runs all six requested phase directions at the five requested offsets `0/25/50/75/90%`.

The central result is that requested launch order is not equivalent to measured GPU execution order. On the RTX 3080
and the current TensorRT contexts, only two concrete actual directions formed multiple usable in-flight buckets:

| Actual GPU direction | Actual buckets | Maximum completion-vector change | Gate A |
|---|---:|---:|---:|
| P -> D | 3 | 18.84% | pass |
| E -> D | 2 | 20.94% | pass |
| D -> P | 0 | unavailable in the controlled first plan |
| D -> E | 0 | unavailable in the controlled first plan |
| E -> P | 0 | one overlap existed beyond the accepted 0--90% range |
| P -> E | 0 | unavailable in the controlled first plan |

Gate A therefore passes: actual in-flight timing materially changes completion and makespan for realizable directions.
It also rejects a stronger but incorrect assumption that all six requested directions can be modeled as symmetric
in-flight actions. M3 must retain requested action identity, but use actual GPU direction/offset and a serial-realization
fallback instead of inventing overlap for an enqueue pair whose engine intervals do not intersect.

## 2. Runtime and measurement structure

```text
fixed real-request frontier
        |
        v
requested direction + target fraction
        |
        +-- P/D: worker launch order and bounded host delay
        |
        `-- E pair: one-shot mature-frontier hold
                    + E/P or E/D launch order
        |
        v
independent TensorRT context enqueue
        |
        v
common CUDA epoch
  incumbent engine [start, end]
  newcomer  engine [start, end]
        |
        +-- actual direction from measured start order
        +-- actual offset / isolated incumbent reference
        +-- incumbent/newcomer duration and slowdown
        +-- overlap and pair makespan
        `-- host completion-visibility span
```

The host-requested delay is never reported as an achieved GPU offset. A sample is accepted only when both engine
intervals overlap and its measured offset is within `0.13` of the nearest actual bucket. CUDA start order may reverse
the requested direction; the artifact preserves both `requested_direction` and `direction`.

The E-incumbent path needs one extra mechanism. If D is the requested newcomer, prerequisite P work is allowed to
mature D while the queued E cohort is held. Once the selected E+D plan launches, that frontier hold is permanently
released. The same one-shot rule applies to telemetry: only the first controlled plan is tagged, so later natural
overlaps cannot contaminate the target cell. Production never enables this controller.

## 3. Completion-vector results

Final combined artifact:

```text
.local/m2-directional-20260901/final-v2/directional-injection-artifact.json
.local/m2-directional-20260901/final-v2/directional-injection-samples.csv
```

Accepted actual cells:

| Direction | Actual bucket | Samples | Actual fraction median | Incumbent ms | Newcomer ms | Makespan ms | Overlap ms | Makespan p95-median ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E -> D | 0.50 | 1 | 0.481 | 23.051 | 14.481 | 25.536 | 11.996 | 0 |
| E -> D | 0.75 | 1 | 0.753 | 22.397 | 9.665 | 26.974 | 5.088 | 0 |
| P -> D | 0.25 | 2 | 0.360 | 14.064 | 8.819 | 14.064 | 8.819 | 0.077 |
| P -> D | 0.50 | 1 | 0.515 | 12.226 | 8.298 | 14.998 | 5.526 | 0 |
| P -> D | 0.75 | 6 | 0.738 | 13.642 | 9.269 | 14.156 | 8.844 | 1.771 |

The `P -> D` zero-offset request had an actual floor near 0.36 because the second context cannot materialize its first
GPU work at the host enqueue timestamp. The 90% P->D request started after the measured incumbent had ended and is
correctly retained as a non-overlap sample but excluded from Gate A.

For E pairs, the distinction is stronger:

- E1+D32 produced actual E->D overlap only around 0.48 and 0.75. Later targets became serial realizations.
- E4+D32 with a large real image did not overlap at all; D's first GPU work appeared after the encoder engine interval.
- D->E used a roughly 6.5 ms incumbent D. Encoder submission did not materialize GPU work before D completed.
- E->P/P->E similarly failed to form accepted 0--90% in-flight buckets in the one-shot controlled plan.

These are measured action-realization limits, not missing host enqueue calls. They explain why earlier action-level
telemetry could report an E+D/E+P selection while fine-grained engine intervals showed little or no overlap.

## 4. Correctness and workload contract

The matrix used real OpenAI-compatible HTTP requests through the phase IPC gateway and the Cosmos Reason2 FP16 path.

| Pair trace | Request contract | Exact token hash across its five targets |
|---|---|---|
| P/D | 96 requests, 5,824 prompt, 6,656 generated tokens | `6c93d3585b084cec8eb31179d6e08d3475c86c96c62ed1a0c5f0e8d5843026b1` |
| E1/D32 | 64 requests, 16,928 prompt, 6,176 generated tokens | `ebfbebd6668458e083340f0112ccb361218ac56afbc4e5787f17a79e778b064a` |
| E/P phase-order variants | 56 requests, 1,320 prompt, 56 generated tokens | stable within each trace order |

Ready GPU memory was about 9,237 MiB. Peak memory was 9,237 MiB for P/D, up to 9,267 MiB for the large-image E1/D
trace, and about 9,247 MiB for the small-image E/P traces.

All selected 30 logs passed the unified event validator:

- zero decision/dispatch/completion correlation gaps
- zero invalid common-epoch GPU intervals
- zero action-fidelity violations
- exact generated/captured token counts

The five-offset cells are engineering characterization runs with one process per cell. The completion artifact exposes
p50, p95, and `p95-median` uncertainty, but paper-grade confidence intervals require selected-cell repeats.

## 5. Analysis correctness fixes found during M2

M2 found and fixed four measurement errors that would otherwise produce misleading conclusions.

1. `run_id=phase-ipc` repeats across processes. Completion joins now include the source log path, preventing cross-run
   execution-ID collisions. The strict multi-log validator uses the same source-scoped identity.
2. Requested direction is stored independently from actual CUDA start order.
3. Non-overlapping intervals are not accepted as in-flight offset buckets.
4. Only the first controlled injection per source run contributes a completion vector; later natural pair actions are
   excluded from the requested target cell.

Stable P/D execution IDs and per-phase plan/action identities were also required to prevent residual augmentation from
rewriting the incumbent chain. Allowed outstanding sets now expand monotonically only after the augmenting plan has an
observable dispatch.

## 6. Code map

| Path | M2 responsibility |
|---|---|
| `cpp/runtime/phase/mechanism/phaseUnifiedEvent.h` | six directions, pair matching, requested injection metadata |
| `cpp/runtime/scheduling/phaseDispatchWorker.{h,cpp}` | P/D launch order, delay, stable per-phase identity |
| `cpp/runtime/scheduling/independentPhase{Coordinator,AsyncServer}.{h,cpp}` | directional control forwarding |
| `cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}` | one-shot E frontier, launch ordering, actual dispatch lineage |
| `examples/llm/llm_phase_context_smoke.cpp` | research environment parsing and event serialization |
| `benchmarks/phase_serving/build_encoder_overlap_trace.py` | phase-first/encoder-first/interleaved real-request frontiers |
| `benchmarks/phase_serving/run_directional_injection_matrix.py` | six requested directions by five targets |
| `benchmarks/phase_serving/analyze_directional_injection.py` | source-safe completion join, actual offsets, Gate A |
| `benchmarks/phase_serving/manifests/directional_injection_artifact_v1.json` | artifact contract |
| `tests/python-unittests/test_directional_injection.py` | actual order, completion vector, non-overlap rejection |
| `unittests/phaseUnifiedEventTest.cpp` | direction endpoints and unordered phase-pair matching |

## 7. Validation performed

- C++ build: `llm_phase_context_smoke` and `unitTest`
- focused C++ tests after the final one-shot filter adjustment: 31/31 passed
- analyzer direct unit invocation: all three tests passed
- Python byte compilation: passed using explicit `/tmp` bytecode paths
- runtime boundary validator: passed
- JSON schema parse and `git diff --check`: passed
- 30 real GPU logs validated together: 54,137 events, 18,140 matched executions, zero action-fidelity failures,
  zero missing dispatch/completion records, and 18,140 valid common-epoch GPU intervals

## 8. Gate A decision and M3 constraints

Gate A passes because E->D and P->D timing changes pair completion by 20.94% and 18.84%, respectively. The M3 action
enumerator should therefore retain incumbent-aware completion-vector prediction, but it must not assume a dense,
symmetric six-direction action space.

M3 requirements derived from M2:

1. Keep requested action and realized execution as separate identities.
2. Represent `serial_realization` when independently enqueued engine intervals do not overlap.
3. Use actual start-skew buckets only after GPU completion; online selection uses dispatch age plus uncertainty.
4. Do not generate reverse-direction overlap value from the forward direction's evidence.
5. Permit 0/1/2 outstanding contexts, but reject E+P+D and same-context re-enqueue.
6. Update an overlap predictor only for action-fidelity samples with positive measured overlap.
7. Preserve a serial fallback for unavailable or uncertain directions.

M3 can now proceed without workload-specific shape rules. The sparse realization surface is itself an observable runtime
property learned from completion feedback, not a workload label or external cost registry.
