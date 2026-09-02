# M1: In-flight shadow snapshot and unified phase event chain

Date: 2026-09-01
Status: complete
Scope: M1 only; the production selector and dispatch policy are unchanged

## 1. Outcome

M1 exposes the currently outstanding E/P/D executions to the global coordinator and emits one append-only event
stream that correlates scheduler decisions, per-phase dispatches, common-CUDA-epoch engine intervals, and host-side
completion visibility.

The final 384-request Cosmos long-lived run produced:

- 1,147 decision records
- 1,298 dispatch records
- 1,298 matching completion records
- 1,298 common-epoch GPU engine intervals
- zero dispatch/completion correlation gaps
- zero action-fidelity violations
- the exact frozen M0 token trace SHA256
  `5c0aa22ee6aa52638fa1ea09c5d4b7fc2ec225a7870d8870aaf1693bd5fa6e21`

No new action selector is active. M1 is mechanism and shadow telemetry for the M2 directional-injection experiment.

## 2. Runtime structure

```text
IndependentPhaseAsyncServer
        |
        | queue counts + in-flight P/D snapshot
        v
IndependentPhaseCoordinator
        |
        v
PhaseDispatchWorker -----------------------------+
  P/D execution_id = stable dispatch index       |
  plan_id may change under residual augmentation |
  status = submitted/running/completion_ready    |
                                                  |
PhaseThreeCoordinator                             |
  adds E execution snapshot                       |
  E execution_id uses a disjoint namespace       |
  captures ready E/P/D work                       |
  records selected plan and incumbent work       |
  observes dispatch/completion transitions       |
        |                                         |
        +-------------------+---------------------+
                            v
                PhaseActivityTimelineRecorder
                  one synchronized CUDA epoch
                  E/P/D engine intervals only
                            |
                            v
          PHASE_SCHEDULER_EVENT <tab> JSON
             decision -> dispatch -> completion
```

The ordinary production path pays no CUDA-event timeline cost unless
`TRT_EDGELLM_PHASE_ACTIVITY_PREFIX` is set. `TRT_EDGELLM_EMIT_PHASE_METRICS=1` enables the unified shadow record;
the selector still uses the existing production policy.

## 3. Identity model

M1 separates three identities that must not be conflated.

| Identity | Lifetime | Purpose |
|---|---:|---|
| `decision_id` / `snapshot_id` | one arbitration boundary | reconstruct the state seen by the selector |
| `plan_id` / `action_id` | one selected action lease | describe the legal outstanding phase set |
| `execution_id` | one physical phase submission | correlate dispatch and completion even if its lease is augmented |

This distinction was required by a real failure found during M1. A P execution can start under a P-only plan and
later become the incumbent in an explicitly selected `P -> E+P` action. Replacing its plan ID destroys the original
dispatch/completion chain; retaining only the old plan falsely reports the explicit augmentation as hidden overlap.

The final implementation therefore keeps `execution_id` stable and maintains an effective allowed-outstanding set per
execution. It expands that set only after the newly selected plan has an observable dispatched member. A decision that
fails before dispatch cannot relax fidelity.

## 4. In-flight snapshot

Each decision record contains:

- ready E/P/D row counts and selected cohort shape
- outstanding phase mask
- each incumbent phase and stable `execution_id`
- current and possibly augmented plan/action IDs
- dispatch host timestamp and age
- `submitted`, `running`, or `completion_ready` status
- canonically ordered `request_ids` for every incumbent execution
- phase-local work: encoder rows, prefill rows/tokens, decode rows/context tokens

GPU running progress is not inferred from host elapsed time. M1 exposes dispatch age and CUDA-event readiness; M2
will measure start-offset buckets directly instead of pretending host age is exact remaining GPU work.

## 5. Unified event contract

The scheduler record prefix is `PHASE_SCHEDULER_EVENT`, not `PHASE_EVENT`. The latter is already the public IPC
token/completion protocol.

### Decision

```text
run_id, event_id
decision_id, snapshot_id, plan_id, action_id
ready, inflight[], selected_cohort, request_ids
candidates[] with request_ids, selected_action_id
outstanding_before_mask, planned_outstanding_mask
```

M1 records the selected candidate in `candidates[]`. Full bounded-frontier emission is an M2/M3 addition because the
current coordinator constructs several candidate families in separate mechanism layers.

### Dispatch

```text
decision_id, snapshot_id, plan_id, action_id
execution_id, phase, cohort, request_ids
action_direction
outstanding_before_mask, planned_outstanding_mask, observed_outstanding_mask
enqueue_host_ns, action_fidelity
```

`action_direction` distinguishes idle launch from the six ordered pairwise injections:

```text
E->P  P->E  E->D  D->E  P->D  D->P
```

### Completion

```text
decision_id, snapshot_id, plan_id, action_id
execution_id, phase, cohort, request_ids
gpu_start_us, gpu_end_us, gpu_duration_us
completion_visible_host_ns, completion_status
observed_outstanding_mask, action_fidelity
```

`gpu_start_us` and `gpu_end_us` are elapsed from the same synchronized CUDA event epoch. Matching is restricted to
engine intervals (`encoder_engine`, `*_dispatch`, `*_residual_dispatch`); sampling intervals deliberately do not satisfy
the completion contract.

The same ordered request lineage is copied from the selected candidate into its physical dispatch and completion.
The unified log can therefore validate membership without joining the older `PHASE_METRIC` stream.

## 6. Code map

| Path | M1 responsibility |
|---|---|
| `cpp/runtime/phase/mechanism/phaseUnifiedEvent.h` | snapshot/event/action-direction types and stable schema version |
| `cpp/runtime/scheduling/phaseDispatchWorker.{h,cpp}` | P/D execution state, stable execution ID, event readiness |
| `cpp/runtime/scheduling/independentPhaseCoordinator.{h,cpp}` | snapshot forwarding |
| `cpp/runtime/scheduling/independentPhaseAsyncServer.{h,cpp}` | queue state plus in-flight snapshot at arbitration |
| `cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}` | E snapshot, decision/dispatch/completion transitions, lease augmentation fidelity, CUDA interval correlation |
| `examples/llm/llm_phase_context_smoke.cpp` | append-only JSON serialization |
| `benchmarks/phase_serving/manifests/phase_event_schema_v1.json` | machine-readable V1 event contract |
| `benchmarks/phase_serving/validate_phase_scheduler_events.py` | chain, ordering, interval, and fidelity validator |
| `unittests/phaseUnifiedEventTest.cpp` | schema names, activity masks, and six direction mappings |

## 7. Final real-request validation

Artifact:

```text
.local/transition-aware-20260830/p107-inflight-m1-engine-interval/
```

Contract:

- model: `nvidia/Cosmos-Reason2-2B`, FP16 engine and KV
- real OpenAI-compatible HTTP requests through the PhaseAsyncServer IPC gateway
- 384 requests: 192 text and 192 vision
- 133,134 prompt tokens and 17,568 requested/generated output tokens
- fixed P chunk 128, P cap 8, D cap 64, E cap 8, max in-flight 80
- greedy, ignore EOS

Correctness and event results:

| Check | Result |
|---|---:|
| generated/captured tokens | 17,568 / 17,568 |
| token SHA vs M0 | exact |
| decisions | 1,147 |
| physical E/P/D executions | 80 / 221 / 997 |
| dispatch/completion pairs | 1,298 / 1,298 |
| common-epoch GPU intervals | 1,298 |
| exact event-to-activity interval matches | 1,298 |
| sampling intervals excluded | 1,216 |
| action-fidelity failures | 0 |

The final schema/lineage smoke artifact is:

```text
.local/transition-aware-20260830/p108-inflight-m1-request-lineage/
```

It sent 32 real HTTP requests (24 text and 8 vision), generated all 1,696 requested tokens, and emitted 149 decisions
plus 154 exact dispatch/completion pairs. All 154 completions had common-epoch GPU intervals, all event members carried
ordered `request_ids`, and the strict validator reported zero correlation and action-fidelity failures. This small run
is a schema/runtime smoke, while the 384-request p107 run remains the performance and activity characterization.

Observed decisions included 42 incumbent-aware augmentation opportunities in the prior final-smoke run. The final
engine-interval run observed the same mechanism families; their exact counts vary with asynchronous timing.

Final one-run performance sanity result:

| Metric | M1 common-epoch run |
|---|---:|
| generated token/s | 587.636 |
| TTFT mean / p95 | 1127.146 / 3322.508 ms |
| TPOT mean / p95 | 24.633 / 38.365 ms |
| E2E mean / p95 | 2245.659 / 4007.561 ms |
| peak VRAM | 9455 MiB |

Compared with the three-run frozen M0 myopic aggregate, this single run is +0.53% in token throughput, -0.79% TTFT
mean, -2.46% TPOT mean, and -1.41% E2E mean. This is a non-regression sanity check, not a performance claim: repeat
count and activity-event instrumentation differ. A policy comparison remains invalid until M2/M3 use the same event
contract on both sides.

## 8. Activity-mask characterization

The opt-in activity recorder measured the following 29.864 s active-span window:

| Item | Time | Ratio |
|---|---:|---:|
| no E/P/D/Copy work (`0000`) | 12.577 s | 42.11% |
| any E/P/D work | 17.287 s | 57.89% |
| E duty | 5.167 s | 17.30% |
| P duty | 7.772 s | 26.03% |
| D duty | 8.662 s | 29.00% |
| E+P | 1.902 s | 6.37% |
| E+D | 0.024 s | 0.08% |
| P+D | 2.388 s | 8.00% |
| E+P+D | 0 | 0% |
| Copy | 0 | 0% |

Copy is zero because this run bound direct external vision output storage; no encoder-output copy kernel was necessary.
The 42.11% mask-idle time is a stream-work definition, not hardware-idle proof. It includes host/request gaps within the
active span and motivates M2 injection experiments, but it does not by itself imply every idle gap is safely fillable.

## 9. Test results

- focused phase tests: 49/49 passed
- full C++ suite before the final common-epoch addition: 1,156 passed, 42 skipped, 3 failed
  - isolated `RopeWriteKvPrefill.AccuracyFp8` passed
  - `InitializeYarnRopeCosSin.Accuracy` and `InitializeMRopeCosSin.Accuracy` retained pre-existing numerical tolerance
    failures unrelated to M1
- production long-lived exact-output gate: passed
- strict unified-event validator with request lineage, action fidelity, and GPU intervals: passed
- runtime feature ownership entry for the new mechanism header: added

## 10. M2 entry point

M2 can now use the event chain without changing production policy:

1. add controlled injection delay/offset to the six directions;
2. emit the full bounded candidate frontier for those decisions;
3. derive incumbent and newcomer completion vectors from common-epoch intervals;
4. bucket actual injection offsets, not requested host delays;
5. repeat selected E1/E2/E4/E8 with P8 and D32 points;
6. decide Gate A: whether injection timing materially changes completion or action ranking.

Until Gate A passes, no in-flight predictor or learned active selector should be promoted.
