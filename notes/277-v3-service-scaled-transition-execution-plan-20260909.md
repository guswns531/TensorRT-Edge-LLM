# 277. V3 service-scaled transition execution plan

## Decision

Proceed with the V3 direction defined in note276, with one strict rule: V3
changes internal urgency units, not execution-cost priors or externally visible
SLO time. The implementation must separate those meanings before any active
policy experiment.

V3 remains:

```text
V2 contextual Scalar RLS
  + V2 deterministic H=2 transition
  + service-epoch-normalized internal urgency
  + mechanism-only bounded candidate frontier
```

It does not introduce a new predictor, Completion-Vector, workload profile,
external cost registry, or learned SLO.

The minimum release candidate is note276 variant D (`MechanismFrontier`). The
request-service-relative RLS feature and all-late normalization remain optional
ablations and are not bundled into the first V3 result.

## 1. Required semantic split

The current code has similarly valued constants with different meanings. They
must not be replaced together merely because they are expressed in milliseconds.

| Current quantity | Current role | V3 treatment |
|---|---|---|
| `decodeQueueWaitTargetUs=2000` | internal D queue pressure, drain and wait control | replace with frozen D service reference |
| `prefillQueueWaitTargetUs=5000` | internal P pressure and synthetic text TTFT fallback | split; normalize P pressure and remove the synthetic text deadline |
| `globalDecodeTpotTargetUs` | composition-root absolute TPOT contract | keep in microseconds with explicit provenance |
| per-request `tpotTargetUs` | request override for absolute TPOT | keep unchanged |
| per-request `ttftTargetUs` | request override for absolute TTFT | keep unchanged |
| `visionTtftTargetUs` | vision composition-root absolute TTFT | keep unchanged |
| `globalColdDecodeMs=2` | cold execution-cost prior | keep as cost; never reinterpret as urgency |
| static/runtime P/D cost points | physical execution estimate | use as service-reference sources |

This split is the primary correctness prerequisite. A global search-and-replace
of `2 ms` or `5 ms` would silently mix policy time, physical cost, and external
service contracts.

## 2. Current implementation map

The relevant existing substrate is already substantial:

- `phasePolicyMode.h` owns the V0/V1/V2 public policy identity.
- `phaseDeadline.h` already defines protected kinds, service-reference
  provenance, per-request isolated reference cost, and elapsed service time.
- `phaseServiceClock.h` already transports request submission and last-token
  commit clocks through unified events.
- `phaseQueueScheduler.cpp` produces runtime exact/interpolated/covering,
  static, derived-isolated, and cold cost references.
- `phaseGlobalScheduler.cpp` already contains shadow service-normalized scoring,
  but its previously rejected replacement authority stays disabled.
- V2 already owns the contextual Scalar decision and bounded deterministic
  transition path.

The missing V3 mechanism is persistent phase service epochs and a clean
separation between `hasExplicitSlo/absoluteSlackUs` and
`serviceAgeQuanta`.

### Fixed-time consumers to audit

| Site | Current behavior | Stage |
|---|---|---|
| queue snapshot P pressure | unspecified TTFT falls back to 5 ms | N2-a |
| queue snapshot D pressure | unspecified TPOT pressure uses 2 ms | N1 |
| queue snapshot D slack | request target or global absolute TPOT | preserve |
| prefill row ordering | unspecified TTFT becomes a 5 ms deadline | N2-a |
| prefill overlap allowance | residual 2 ms D budget | N1 |
| dynamic decode drain | 2 ms feasibility and remaining budget | N1 |
| recent decode pressure | observed interval divided by 2 ms | N1 |
| WAIT/refill | D target and residual pressure | N1 audit |
| expired-P decode suppression | standalone D may be deleted | N2-b |
| vision coordinator | explicit vision TTFT and decode TPOT | preserve |
| cold decode fallback | 2 ms physical cost estimate | preserve |

The N0 audit must identify every transitive consumer of
`recentDecodeTpotPressure`, because encoder admission, vision capacity, and
memory/admission controllers may consume it even when they do not read the
2 ms constant directly.

## 3. V3 state contract

Add one focused state type, preferably next to the existing service clock and
deadline contracts:

```cpp
struct PhaseServiceReference
{
    double serviceUs{};
    PhaseServiceReferenceSource source{};
    uint64_t epoch{};
    bool valid{};
};

struct PhaseServiceState
{
    double readyWaitUs{};
    double serviceAgeQuanta{};
    PhaseServiceReference reference;
    bool hasExplicitSlo{};
    double absoluteSlackUs{};
};
```

The scheduler must not encode `hasExplicitSlo` indirectly as a finite fallback
number. For V0/V1/V2, zero targets retain their existing inheritance semantics.
For V3, an unspecified text TTFT is explicitly represented as no deadline.
The composition root records whether a global TPOT or vision TTFT came from a
user/config contract rather than a built-in fallback.

### Epoch lifetime

```text
D epoch: previous token commit -> next token commit/completion
P epoch: row becomes P-ready -> real P progress/transition/cancel
E epoch: request becomes E-ready -> real E progress/cancel
```

The epoch start timestamp and reference are captured once before candidate
ranking. A later cost observation can update the next epoch only. Candidate
preview and transition rollout receive copies and cannot mutate the online
epoch.

Cold references may provide bounded internal fairness during startup, with
their provenance exposed in telemetry. They do not become evidence for unsafe
overlap or masquerade as runtime measurement.

## 4. Target decision pipeline

```text
request DAG + ready E/P/D + ownership + outstanding contexts
                            |
                            v
              canonical mechanism cohorts
                            |
                            v
           immutable per-request service epochs
                            |
                            v
             bounded hard-feasible frontier
       dependency / shape / inflight / ownership / memory
                            |
                            v
             explicit absolute-SLO safety
                  only where configured
                            |
                            v
              normalized service recovery
          age / frozen isolated service reference
                            |
                            v
             V2 Scalar + H=2 transition value
                            |
                            v
                         dispatch
```

Service recovery makes an action eligible; it does not make strict oldest-first
the final policy. Candidate efficiency and transition value remain responsible
for avoiding cohort fragmentation.

## 5. Causal implementation sequence

Each stage uses one source commit and one signed commit. A failed performance
gate is retained as a result and the stage is not folded into V3.

### S0 — Freeze V2 and add telemetry only

Code changes:

- extend unified decision events with service reference value, source, epoch,
  service age, explicit-SLO flag, and absolute slack;
- add counters for pseudo-TTFT use, D pressure over one quantum, D candidate
  suppression, P-only streak while D-ready, and D-ready non-D time;
- record the provenance of global/request SLO values;
- add an analyzer that reports reference coverage and fixed-target/service-cost
  ratios.

Invariants:

- no candidate, ordering, or selected action changes;
- paired dispatch hashes match the pre-change V2 control;
- scheduler decision p95 remains within repeat noise.

Experiments:

- controlled mixed trace with the retained long D gap;
- balanced, text-heavy, mixed, and multi-image, three repeats;
- derive the exact N1/N2 consumer list from observed events.

### S1 — Service-reference and epoch mechanism

Code changes:

- introduce `PhaseServiceReference` and `PhaseServiceState`;
- maintain per-request P/D epoch state in the queue/server and E epoch state in
  the coordinator;
- capture canonical reference costs before candidate ranking;
- reset only on the real progress boundaries defined above;
- expose snapshot accessors without changing V2 selection.

Unit gates:

- provenance order and cold fallback;
- candidate-independent denominator;
- monotonic service age within an epoch;
- observation changes only the next epoch;
- correct P/D/E reset, cancellation, and request-ID reuse behavior;
- snapshot/preview does not mutate epochs.

S1 remains shadow-only. V2 must be dispatch-identical with the feature enabled
or disabled.

### S2 — N1 decode service scale

Change only internal D queue control:

- queue pressure;
- dynamic decode batch drain feasibility;
- remaining WAIT/refill budget;
- overlap interference allowance;
- recent decode service pressure and its downstream consumers.

Do not change:

- `globalColdDecodeMs`;
- explicit TPOT slack in microseconds;
- P 5 ms behavior;
- candidate pruning;
- Scalar features or H=2 transition.

Run focused unit tests, deterministic replay, mixed 3--5 repeats, and then the
four-workload gate. Stop on >3% workload or >1% geometric-mean throughput
regression, output mismatch, action-fidelity failure, or material TPOT p95
regression.

### S3 — N2-a no synthetic text TTFT

- an unspecified text TTFT produces infinite absolute slack and
  `hasExplicitSlo=false`;
- P-local formation uses P-ready service age;
- the request first-token path uses submission age divided by robust remaining
  E+P+first-D service;
- prefill row ordering uses explicit TTFT first when present, then normalized
  service age and canonical stable tie-breaking;
- explicit vision and per-request TTFT remain absolute;
- retain the old candidate frontier for this stage.

The critical test is that text requests no longer become falsely expired after
5 ms while vision requests still preserve E-to-first-token TTFT protection.

### S4 — N2-b mechanism-only frontier

- preserve standalone P and D if hard-feasible;
- remove pseudo-expiration and pressure from action construction/pruning;
- apply explicit SLO safety and normalized urgency only in the global selector;
- keep the current bounded top-k candidate limit and deterministic ordering;
- record preserved-but-not-selected and preserved-selected actions.

Passing S2, S3, and S4 creates the minimum V3 policy. Only then add
`kServiceScaledTransition`, the `service-scaled-transition` parser name, CLI
help, replay support, and manifest identity.

### S5 — V3 primary promotion

Compare same-binary V0/V1/V2/V3 with identical engine, requests, calibration,
memory, graphs, P128, P8/D64/E4, and output contract.

First gate:

```text
balanced
text-heavy
mixed
multi-image
```

Use three repeats per policy, alternating order. If it passes, run full12 with
three repeats. Reuse frozen vLLM only because the engine-facing contract is
unchanged.

Primary metrics:

- token/request throughput;
- TTFT, TPOT, E2E mean/p95;
- joint-SLO request/token goodput;
- P/D/E dispatch and cohort distributions;
- D service gap p50/p95/p99/max;
- service-age and reference-source distributions;
- suppressed/preserved/selected candidate counts;
- action fidelity and output hashes;
- scheduler p50/p95/max and GPU memory.

### S6 — Optional N3 RLS feature

Only after the core V3 comparison is frozen, replace pair-relative slack with
request-service-relative slack in a separate branch. Keep the model dimension
and all other features fixed where possible.

Promote only if held-out error or action regret improves and the four-workload
and full12 serving gates do not regress. Otherwise the final V3 uses the exact
V2 Scalar representation.

### S7 — Optional N4 all-late ranking

Run only if V3 telemetry shows repeated regret when every candidate is already
late. Use the lexicographic normalized violation rule from note276. Do not
reactivate the rejected service-normalized replacement authority.

## 6. Tests by source boundary

| Boundary | Required validation |
|---|---|
| policy parser | V0/V1/V2 unchanged; V3 round-trip name |
| service state | source, epoch freeze, monotonicity, reset, cancel/reuse |
| queue snapshot | explicit SLO separated from service age |
| batch formation | canonical cohort reference independent of candidate |
| global selector | hard feasibility first, explicit SLO second, service recovery third |
| transition | copied epoch/reference; no arbitrary-arrival prediction |
| three-phase coordinator | vision TTFT spans E/P and E epoch resets on real progress |
| async server | last-token commit starts D epoch; sampling visibility does not forge a commit |
| unified event | schema-compatible optional V3 telemetry |
| HTTP path | exact tokens, cancellation, continuous admission, multi-image |

The minimum C++ test set extends `phaseQueueSchedulerTest`,
`phaseRuntimeCostTrackerTest`, `phaseUnifiedEventTest`, and
`independentPhaseAsyncServerTest`. Coordinator-specific E/vision cases remain
in the existing three-phase tests. Python tests cover manifest/replay parsing
and analysis schema.

## 7. Promotion criteria

V3 may replace V2 as the canonical production comparison only when all are
true:

1. no output identity, ownership, action-fidelity, or memory correctness
   failure;
2. no four-workload throughput regression worse than 3%;
3. no four-workload geometric-mean throughput regression worse than 1%;
4. mixed, text-heavy, and multi-image TPOT p95 remains within repeat noise or
   improves;
5. full12 geometric-mean throughput and joint-SLO goodput preserve or improve
   V2;
6. no new workload-level E2E p95 regression beyond the 3% gate;
7. scheduler p95 overhead remains below 50 us and does not materially increase
   host launch gaps;
8. candidate count remains bounded and output hashes remain deterministic;
9. no model, GPU, workload-name, or request-class special case is added.

Cross-release action identity is diagnostic. Stable service outcome without
retuning is the portability gate.

## 8. Expected interpretation

The experiment can produce three useful outcomes:

| Result | Interpretation |
|---|---|
| V3 improves and is robust | fixed wall-clock urgency caused policy amplification |
| V3 changes actions but does not improve | service scaling alone is insufficient; retain V2 and use telemetry to locate regret |
| V3 rarely changes actions | keep service state as diagnostics; do not claim a policy contribution |

No outcome justifies workload-specific thresholds. V3's research question is
whether one dimensionless internal service contract transfers across measured
runtime cost changes while absolute user SLO semantics remain intact.

## 9. Artifact and commit sequence

Retain campaigns under:

```text
.local/results/v0101-forward-port/v3-service-scale-20260909/
  s0-telemetry/
  s1-service-epoch/
  s2-decode-scale/
  s3-no-pseudo-ttft/
  s4-mechanism-frontier/
  s5-four-workload/
  s5-full12/
  s6-rls-ablation/
  s7-all-late-ablation/
```

Planned signed commits are:

1. telemetry/schema with no action change;
2. service-reference epoch mechanism with shadow validation;
3. decode service-scaled queue control;
4. explicit-SLO and P-service separation;
5. mechanism-only frontier and named V3 policy;
6. replay/analyzer and promotion results;
7. optional RLS/all-late changes only when separately promoted.

Failed experiments remain result artifacts and notes, not dormant production
branches or environment-variable combinations.

## 10. Immediate next action

Start with S0 only. Produce a machine-readable consumer inventory and a paired
V2 dispatch-identity report before adding active service epochs. This makes the
first code change observational and gives S2/S3 an exact causal baseline.
