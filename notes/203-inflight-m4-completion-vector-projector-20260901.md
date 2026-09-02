# M4: Completion-vector replay and deterministic H1 projector

Date: 2026-09-01
Status: complete
Scope: replay/oracle mechanism only; production action selection remains unchanged

## 1. Outcome

M4 replaces the old whole-action replay boundary with one deterministic transition at the earliest predicted or
measured E/P/D completion. It consumes the bounded M3 action identity, a replay completion vector, the post-dispatch
in-flight state, and stable request ownership. It produces the next request/DAG state, persistent ownership state,
remaining completion vector, and complete M3-legal action frontier.

```text
M3 legal incremental action
          +
replay completion vector
          +
post-dispatch request/ownership snapshot
          |
          v
robust earliest component
          |
          +-- retire exactly one execution lease
          +-- apply E -> P, P -> D, or D -> D/Complete
          +-- update TTFT/token milestone
          +-- allocate/reclaim stable vision and KV ownership
          +-- preserve the unfinished context and residual completion
          `-- enumerate the next bounded M3 frontier
```

This is the missing semantic bridge between M2's measured component times and M3's legal incremental actions. It is
not a learned predictor and does not change the production policy.

## 2. Why the old H2 boundary was wrong

The previous formation planner evaluates a concrete pair using one pair makespan. Only after that whole makespan does
it append a known D boundary:

```text
old replay
  max(E completion, D completion)
       -> append another D service boundary
```

The actual independent-context runtime exposes a decision opportunity when either component completes:

```text
E [=========================]
D     [=====]
      ^ D completion
      |
      +-- E remains outstanding
      +-- D request advances or releases KV
      `-- a P or new D cohort may now be legal
```

Treating this as one indivisible action both delays the next legal decision and loses the identity of the remaining
context. Appending a D after the pair also misrepresents a D component that already completed before E/P.

The M4 projector rejects a pair completion vector that omits either context. It advances to the minimum robust
component completion, subtracts only that mean time from the other component, and carries the other execution lease
into the successor snapshot.

## 3. State and transition contract

### 3.1 Completion vector

Each action has exactly one component per outstanding context:

```text
PhaseCompletionVector {
  action_id,
  [phase, execution_id, mean_completion_us, uncertainty_us, incumbent]
}
```

The projector verifies:

- the M3 action is legal;
- completion-vector and M3 action identities match;
- in-flight work exactly matches the planned outstanding mask;
- there are one or two unique E/P/D components;
- every component maps to one execution lease;
- every completed request belongs to the matching request DAG stage.

The robust boundary is `mean + beta * uncertainty`; simulated clock progress and residual mean completion use the
component mean. This retains uncertainty for the future M5 SLO ranker without incorrectly adding a confidence margin
to physical elapsed time.

### 3.2 Request DAG

The first implementation models the production milestones needed by H1 replay:

| Completed phase | Request transition | Milestone |
|---|---|---|
| E | `Encoder -> Prefill` | vision payload becomes P-ready |
| P | `Prefill -> Decode` or `Complete` | first token/TTFT observed |
| D | `Decode -> Decode` or `Complete` | one output token/TPOT service observed |

No future external arrival is predicted. Ready cohorts are rebuilt only from requests already present in the immutable
snapshot and from the selected completion's deterministic DAG transition.

### 3.3 Persistent ownership

Ownership is recomputed from stable request leases at the input boundary and then transitioned exactly once:

| Boundary | Vision lease | KV lease |
|---|---|---|
| E completion | allocate/retain | unchanged |
| P completion | reclaim after consumption | allocate/retain |
| non-final D completion | unchanged | retain |
| final P/D completion | reclaim if owned | reclaim |

The projector separately records current vision/KV bytes and cumulative reclaimed bytes. TensorRT workspace remains
outside this request-persistent ownership model and stays under the existing hard feasibility mechanism.

### 3.4 Next frontier

In-flight request IDs are excluded from ready cohort formation. The remaining ready requests are hashed in canonical
request order into at most one E, P, and D phase-local candidate. M3 then enumerates the next legal frontier.

Important examples:

```text
E+D, E completes first
  outstanding = D
  E requests  = P-ready
  next legal  = NO_DISPATCH or D->P

P+D, D completes first with more output remaining
  outstanding = P
  D request   = D-ready again
  next legal  = NO_DISPATCH or P->D

E+D, D completes final token first
  outstanding = E
  KV lease    = reclaimed
  existing P  = legal E->P newcomer
```

The successor can contain zero or one outstanding context after a completion; it never creates E+P+D.

## 4. Replay predictor boundary

`PhaseReplayCompletionPredictor` is an exact action-ID map for measured/oracle vectors. It intentionally does not
interpolate and has no TTL or external registry. Its purpose is to validate projector and M5 oracle semantics before
the M6 residual RLS model is allowed to generalize across continuous features.

```text
M2 measured common-epoch completion vector
                 |
                 v
exact M3 incremental_action_id map
                 |
                 v
M4 deterministic projector
```

An unknown or illegal action returns no prediction. A production learned predictor is still out of scope for M4.

## 5. Actual Cosmos replay

> 2026-09-01 correction: 최초 artifact는 incremental boundary를 incumbent 시작점으로 두었다. M5 audit에서 newcomer
> GPU start가 실제 H1 origin이어야 함을 확인해 `pd-o50-h1-replay-residual.json`으로 교정했다. 아래 whole-action
> overrun과 earliest-phase count는 변하지 않지만, projected boundary는 incumbent 전체 시간이 아니라 boundary에서 남은
> residual completion이다. corrected projected boundary median은 3.986 ms이고 범위는 1.350--8.257 ms다.

The M3 real-request `P->D@50%` Cosmos log was replayed without rerunning the model:

```text
.local/m3-incremental-20260901/pd-o50/prefill_to_decode-o50/run-001/gateway.log
        ->
.local/m4-projector-20260901/pd-o50-h1-replay.json
```

The source run contains 96 real HTTP requests, 5,824 prompt tokens, 6,656 generated tokens, and the exact frozen token
hash recorded by M3. M4 joined every natural directional pair in that log, not only the one controlled injection.

| Replay metric | Result |
|---|---:|
| Directional completion vectors | 74 |
| Earliest phase: D | 67 |
| Earliest phase: P | 7 |
| Pair-wide completion later than H1 boundary | 74 / 74 |
| Whole-action overrun median | 8.292 ms |
| Whole-action overrun p95 | 9.125 ms |
| Whole-action overrun range | 1.443--9.714 ms |
| Invalid successor mask | 0 |
| E+P+D successor | 0 |

This is direct evidence that a whole-pair completion is not the runtime's next useful decision boundary. In all 74
measured pairs, one context remained outstanding for a material interval after the other completed. Decode was the
earliest component in 90.5% of pairs, so appending D only after the pair makespan is especially mismatched for this
trace.

The replay result is a mechanism characterization, not a throughput claim. The M4 code is shadow/replay-only and the
request contract and production selector are unchanged, so the frozen vLLM and 12-workload performance results remain
the relevant comparison until M5 activates an oracle H1 selector.

## 6. Code map

| Path | M4 responsibility |
|---|---|
| `cpp/runtime/phase/mechanism/phaseIncrementalProjector.h` | request/DAG, ownership, completion-vector, projector, replay-predictor contract |
| `cpp/runtime/scheduling/phaseIncrementalProjector.cpp` | validation, robust earliest event, exact transition, residual vector, next frontier |
| `unittests/phaseIncrementalProjectorTest.cpp` | serial/pair replay, ownership, refill, and whole-action mismatch tests |
| `benchmarks/phase_serving/analyze_incremental_h1_replay.py` | actual unified-event completion join and H1 artifact |
| `tests/python-unittests/test_incremental_h1_replay.py` | measured earliest-boundary and artifact tests |
| `cpp/CMakeLists.txt` | stable phase-runtime object ordering |
| `benchmarks/phase_serving/manifests/v010_feature_owners.json` | Current feature ownership classification |

## 7. Validation

- Host C++ syntax checks: projector and unit-test translation units passed.
- TensorRT 11.0/CUDA 13.3 SM86 build: `unitTest` linked successfully.
- C++ focused/adjacent gate: 32/32 passed.
  - M4 projector: 7/7
  - M3 legality: 7/7
  - prior formation/H2: 14/14
  - unified event chain: 4/4
- Python replay/directional tests: 7/7 passed.
- Actual Cosmos log: 74/74 vectors projected with no invalid or triple-phase successor.
- Python byte compilation with an external bytecode path: passed.
- `git diff --check`: passed.

The CMake cache had lost three discovered CUDA paths while being regenerated. The build was recovered by explicitly
setting the CUDA 13.3 runtime include, CURAND include, and CUDART library already present in the TensorRT image. This
was an environment/cache issue; the new projector compiled cleanly both in a host syntax check and the complete CUDA
target build.

## 8. M4 boundary and M5 plan

M4 answers: "Given one legal action and its completion vector, what is the next legal observable state?" It does not
answer: "Which legal action should be selected?"

M5 should now:

1. construct a replay completion vector for every legal action in the same immutable snapshot;
2. project each action to its earliest completion boundary;
3. rank only valid successors in lexicographic order:
   `hard feasibility -> robust SLO violation -> useful progress -> H1 efficiency`;
4. compare oracle H1, myopic, legacy-compatible, and current H2 using the same runtime and exact request contract;
5. first validate controlled counterexamples, then run the 12-workload gate and 39/48.8/97.5 req/s load points;
6. retain M4 as shadow-only if oracle H1 cannot improve SLO goodput without regressing the workload gate.

Only after M5 establishes actionable H1 regret should M6 replace the exact replay map with a direction-specific
continuous residual RLS predictor.
