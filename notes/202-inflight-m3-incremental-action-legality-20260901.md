# M3: Incremental action enumerator and legality

Date: 2026-09-01
Status: complete
Scope: deterministic mechanism and telemetry; production action-value selection remains unchanged

## 1. Outcome

M3 converts the sparse directional behavior measured in M2 into a bounded action contract. The scheduler no longer
needs to infer legality from action names or scattered coordinator branches. Every dispatch lease now owns one stable
incremental identity over:

```text
phase-local cohort identity
        +
outstanding-before -> planned-outstanding transition
        +
requested launch direction/order
        +
requested or incumbent-age start-skew bucket
```

The legality layer is deterministic and independent of the policy cost model. It permits at most two E/P/D contexts,
rejects same-context re-enqueue and E+P+D, and represents no dispatch as an explicit action.

## 2. Bounded V1 frontier

The enumerator consumes at most one deterministic phase-local cohort per E/P/D context.

| Outstanding at the decision boundary | Returned frontier |
|---|---|
| none | NO_DISPATCH, each ready singleton, both orders for each ready pair |
| one phase | NO_DISPATCH, each distinct legal newcomer |
| two phases | NO_DISPATCH only |
| invalid E+P+D snapshot | empty frontier |

With E, P, and D all ready, the largest idle frontier is ten actions:

```text
NO_DISPATCH
E, P, D
E->P, P->E
E->D, D->E
P->D, D->P
```

Duplicate entries for one phase, a zero candidate identity, and non-E/P/D phases invalidate the input frontier. This
keeps candidate count bounded without workload labels or shape-specific policy conditions.

## 3. Legality authority

`PhaseIncrementalLegalityReason` records the first structural reason an action is illegal:

- invalid outstanding mask
- malformed NO_DISPATCH
- missing newcomer
- same-context re-enqueue
- third-context/E+P+D attempt
- incumbent mismatch
- requested direction mismatch
- planned outstanding mismatch
- dependency, context, or TensorRT shape failure

Policy code cannot override these results. `PhaseGlobalDispatchPlan::permits()` now also requires a legal incremental
action and exact agreement between its planned set and the execution lease's allowed set.

Residual augmentation uses the same authority:

```text
P running + D cohort -> P->D
D running + P cohort -> D->P
P/D running + E      -> P->E or D->E
pair already running + third phase -> reject
same phase already running + same phase -> reject
```

Row and stable-slot matching from the previous lease implementation remains in force. M3 adds transition legality; it
does not weaken ownership fidelity.

## 4. Direction and start-skew identity

The supported start-skew buckets are `unknown`, `0`, `25`, `50`, `75`, `90`, and `serial_realization`. Initial pair
launches normally use offset 0. A residual action maps incumbent host-observed age over its robust reference duration
to the nearest bucket. M2 controlled injection retains its requested bucket exactly.

`direction` is ordered. For example, `P->D@50` and `D->P@50` have different action IDs. Changing only the skew bucket
also changes the action ID. This prevents the M2 mistake of sharing evidence between reverse enqueue requests whose
actual CUDA realization differs.

After completion, the directional analyzer assigns either:

```text
overlap + actual offset bucket
```

or:

```text
serial_realization
```

Non-overlapping engine intervals are never admitted to an overlap bucket.

## 5. Runtime and telemetry integration

Every coordinator-owned execution lease is materialized through the M3 legality function. Initial E/P/D, initial pair,
and residual pair paths all carry the identity.

Unified decision, dispatch, and completion records now include:

- `incremental_action_id`
- `requested_action_direction`
- `requested_start_skew_percent`
- existing planned/observed outstanding masks and `action_fidelity`

The validator checks identity continuity across dispatch and completion, disallows masks outside E/P/D, and rejects an
E+P+D mask. Multi-process validation remains source-log scoped because `run_id=phase-ipc` is reused by every server
process.

## 6. Code map

| Path | M3 responsibility |
|---|---|
| `cpp/runtime/phase/mechanism/phaseIncrementalAction.h` | action key, buckets, legality reasons, enumerator API |
| `cpp/runtime/scheduling/phaseIncrementalAction.cpp` | bounded enumeration, legality, identity hashing |
| `cpp/runtime/phase/execution/phaseActionPlan.h` | incremental identity owned by every dispatch lease |
| `cpp/runtime/scheduling/phaseGlobalScheduler.cpp` | lease binding and residual augmentation validation |
| `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | initial/residual direction and skew binding |
| `cpp/runtime/phase/mechanism/phaseUnifiedEvent.h` | incremental event identity fields |
| `examples/llm/llm_phase_context_smoke.cpp` | JSON event serialization |
| `benchmarks/phase_serving/validate_phase_scheduler_events.py` | mask and identity-chain validation |
| `benchmarks/phase_serving/analyze_directional_injection.py` | overlap versus serial realization classification |
| `unittests/phaseIncrementalActionTest.cpp` | complete 0/1/2-context frontier and rejection tests |

## 7. Validation

### Build and unit tests

- TensorRT 11/CUDA 13.3 container build: `llm_phase_context_smoke` and `unitTest` passed.
- M3-focused plus adjacent mechanism tests after the duplicate-frontier guard: 84/84 passed.
- Python analyzer/validator tests: 5/5 passed.
- Full C++ suite: 1,208 tests executed; 1,163 passed and 42 skipped. Three unrelated CUDA accuracy tests initially
  failed. A clean-process rerun passed `RopeWriteKvDecodeTreeAttention.AccuracyFp8`; the two pre-existing
  `InitializeYarnRopeCosSin.Accuracy` and `InitializeMRopeCosSin.Accuracy` tolerance failures remained. M3 changes no
  RoPE kernel or reference path.

### Real Cosmos HTTP smoke

M3 reran the existing controlled P->D 50% trace:

- 96 real HTTP requests
- 5,824 prompt tokens
- 6,656 generated and captured tokens
- exact prior token hash: `6c93d3585b084cec8eb31179d6e08d3475c86c96c62ed1a0c5f0e8d5843026b1`
- 2,739 unified events; all 2,739 carried incremental identity, requested direction, and skew
- 913 matched executions and 913 common-epoch GPU intervals
- zero missing dispatch/completion, action-fidelity failures, or triple-phase masks

Artifact root:

```text
.local/m3-incremental-20260901/pd-o50/
```

The one-run mechanism smoke versus the matching M2 cell was:

| Metric | M2 | M3 | Delta |
|---|---:|---:|---:|
| request/s | 15.313 | 15.203 | -0.72% |
| TTFT mean ms | 21.434 | 20.291 | -5.33% |
| TTFT p95 ms | 31.465 | 31.511 | +0.15% |
| TPOT p95 ms | 16.214 | 16.208 | -0.04% |
| E2E p95 ms | 1339.521 | 1341.376 | +0.14% |

This is a one-run correctness/telemetry smoke, not a performance promotion result. The production selector and request
contract did not change, so the frozen vLLM comparator is not rerun for M3.

## 8. Boundary of M3 and M4

M3 decides which state transitions are legal and gives them stable identity. It does not predict the completion vector
or actively choose a new policy. M4 is now responsible for:

1. consuming an M3 action and M2 measured realization,
2. replaying the earliest concrete completion rather than the whole pair makespan,
3. applying request DAG and ownership transitions at that boundary,
4. rebuilding the next 0/1/2-outstanding snapshot deterministically,
5. retaining `serial_realization` when the requested pair did not physically overlap.

This preserves the research separation: deterministic mechanism defines legal actions; the later oracle/learned policy
only ranks those actions.
