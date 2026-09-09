# 276. V3 service-scaled transition policy plan

## Status

This document defines the next policy generation. It does not claim that V3 is
implemented, validated, or promoted. The production default remains V0/V1/V2
until every promotion gate in this document passes.

V3 addresses a specific portability and policy-amplification problem exposed by
the v0.10.0 to v0.10.1 forward port: the execution model adapts to measured GPU
cost, but several queue-urgency and candidate-eligibility decisions still use
fixed wall-clock constants. A small runtime timing change can therefore cross a
fixed threshold, remove an action before global selection, and produce a much
larger scheduling-trajectory change.

The central hypothesis is:

> Online execution models alone do not make a scheduler portable if internal
> urgency and queue-control decisions remain tied to fixed wall-clock constants.

V3 keeps externally meaningful SLOs in absolute time and expresses internal
scheduling pressure in units of immutable, isolated GPU service cost.

## 1. Policy lineage and name

| ID | Canonical name | Execution estimate | Decision horizon | Internal service scale |
|---|---|---|---|---|
| V0 | Exact | exact CUDA registry and conservative fallback | immediate | fixed current policy |
| V1 | Contextual Scalar | V0 plus contextual scalar RLS advantage | immediate | fixed current policy |
| V2 | Scalar + Transition | V1 | bounded deterministic request/DAG transition | fixed current policy |
| V3 | Service-Scaled Transition | V1 | same bounded deterministic transition | execution-cost normalized |

The intended runtime name is `service-scaled-transition`. V3 is not a new
Completion-Vector, Decomposed-Scalar, service-normalized replacement authority,
or workload profile. Its learned reward and bounded transition substrate remain
those of V2.

The minimum contract required to call a policy V3 is:

1. explicit external SLOs remain absolute deadlines;
2. a request without an explicit SLO is not assigned a synthetic deadline;
3. phase-local urgency uses service-epoch-normalized age instead of fixed 5/2 ms
   queue targets;
4. fallback timing policy cannot hard-delete an otherwise executable action;
5. the V2 scalar estimator and deterministic transition evaluator remain the
   efficiency authority after feasibility and service protection.

RLS feature changes and normalized all-late ranking are staged V3 experiments,
not prerequisites silently bundled into the first V3 implementation. They are
promoted only if their separate gates pass.

## 2. Evidence motivating V3

The retained mixed trace measured approximately:

```text
P GPU service: 33 ms
D GPU service: 8.41 ms
```

The current internal and external time scales are:

| Quantity | Current value | Approximate service quanta |
|---|---:|---:|
| text P fallback | 5 ms | 0.15 P execution |
| D queue target | 2 ms | 0.24 D execution |
| D TPOT target | 80 ms | 9.5 D executions |

The 2 ms decode scale is used by more than one mechanism: queue pressure,
dynamic decode drain selection, residual wait budget, recent TPOT pressure, and
several adaptive batching decisions. Replacing only one occurrence would leave
inconsistent urgency definitions in the same scheduler.

In the retained v0.10.1 mixed diagnostic, one decode service interval lasted
424.732 ms. Only 2.16% of the interval lacked recorded E/P/D work. Six later P
decisions had 34--38 D rows ready while the P hard guard suppressed standalone
D. Restoring D alone did not guarantee its selection, showing that candidate
availability and action ranking are distinct problems.

This evidence motivates service scaling and mechanism-only candidate
preservation. It does not prove that always choosing D, minimizing wall-clock
wait, or maximizing cross-version action agreement is correct.

## 3. External service contract

The primary V3 campaign freezes the following contract before any policy A/B:

```text
Vision-request TTFT: 500 ms
All-request decode TPOT: 80 ms
Text-request TTFT: unspecified
Runtime E2E deadline: none
```

The scope is important. `Vision TTFT = 500 ms` does not implicitly assign a
500 ms TTFT to text-only requests. Text requests without request metadata have
no absolute TTFT deadline in V3.

Per-request metadata continues to override the composition-root contract:

```json
{
  "metadata": {
    "phase_scheduling": {
      "ttft_target_ms": 500,
      "tpot_target_ms": 80,
      "priority": 0
    }
  }
}
```

Absolute SLO semantics do not depend on GPU speed. A request with an explicit
deadline is checked in microseconds:

```text
absolute_slack = deadline - age
robust_completion = predicted_completion + uncertainty
safe iff robust_completion + guard <= absolute_slack
```

The campaign manifest, runtime telemetry, and result summary must record the
contract and its source. A hidden global fallback must not be described as a
request-provided SLO.

## 4. Internal service state

### 4.1 Required types

V3 should expose one explicit service-state contract instead of reusing an
ambiguous `pressure` or `slack` value:

```cpp
struct PhaseServiceReference
{
    double serviceUs;
    PhaseServiceReferenceSource source;
    uint64_t epoch;
    bool valid;
};

struct PhaseServiceState
{
    double readyWaitUs;
    double serviceAgeQuanta;
    PhaseServiceReference reference;
    bool hasExplicitSlo;
    double absoluteSlackUs;
};
```

Exact names may follow project conventions, but the two meanings must remain
separate:

```text
absoluteSlackUs    -> external SLO safety only
serviceAgeQuanta   -> internal urgency, fairness, and recovery only
```

### 4.2 Reference provenance

Allowed reference sources, in descending preference, are:

1. runtime exact isolated cost;
2. runtime interpolated isolated cost;
3. runtime covering isolated cost;
4. static isolated profile;
5. bounded cold prior.

Selected overlap timing, queue residence, request deadline, and candidate
ranking output are not valid reference costs. Cold references remain visible in
telemetry and must not masquerade as measured evidence.

### 4.3 Candidate independence

The denominator is computed before candidate ranking from a mechanism-only
canonical cohort. Every candidate compared at one decision boundary uses the
same request reference. The selected candidate cannot change its own urgency
denominator.

For decode, the canonical cohort is the largest currently compatible mechanism
cohort containing the protected request, bounded by the engine profile and
stable ownership. Its robust isolated cost supplies `C_D_ref`.

For prefill, the canonical cohort is the compatible phase-local P cohort
containing the protected request, using its concrete producer class, chunk,
past-KV range, and engine limits. Its robust isolated cost supplies `C_P_ref`.

### 4.4 Service-epoch stability

Snapshot immutability alone is insufficient. A batch-size or reference-source
change must not make a request less urgent while it continues waiting.

V3 freezes the reference for one service epoch:

- decode epoch: starts at the previous token commit and ends at the next token
  commit or request completion;
- prefill epoch: starts when the row becomes P-ready and ends on real prefill
  progress, phase transition, cancellation, or completion;
- encoder epoch: starts when E becomes request-ready and ends on encoder
  progress, cancellation, or completion.

A newly available exact observation takes effect at the next service epoch, not
retroactively in the current one. This makes service age monotonic:

```text
Q_D = elapsed_since_last_token / frozen_C_D_ref
Q_P_local = elapsed_since_P_ready / frozen_C_P_ref
```

An integrated `sum(dt / C_ref(t))` clock may be retained as a shadow diagnostic,
but it is not required for the first active implementation.

### 4.5 First-token state is not P queue state

V3 keeps two distinct producer clocks:

```text
P local formation:
    P-ready wait / frozen P service reference

Request first-token path:
    age since submission /
    robust remaining E + P + first-D reference work
```

The former controls P batching and phase-local fairness. The latter protects a
request's E-to-P-to-first-token critical path. They must not be collapsed back
into one pseudo-TTFT threshold.

## 5. V3 selection semantics

The target pipeline is:

```text
ready E/P/D, outstanding contexts, request DAG, ownership
                         |
                         v
              mechanism-only candidates
                         |
                         v
                 hard feasibility
        dependency / TRT shape / inflight / memory
                         |
                         v
          explicit absolute-SLO safety, if supplied
                         |
                         v
        service-epoch-normalized urgency and recovery
                         |
                         v
          contextual Scalar RLS + H=2 transition
                         |
                         v
                      dispatch
```

### 5.1 Hard pruning

Only actions that cannot legally execute are removed:

- dependency not ready;
- unsupported TensorRT profile or binding shape;
- execution context already in flight;
- KV or vision ownership conflict;
- memory capacity or reservation failure;
- invalid request membership, row ordering, or stable slot.

Queue age, fallback deadline, predicted profitability, and phase preference are
not hard-feasibility conditions.

### 5.2 Explicit SLO path

If at least one candidate protects a request with an explicit SLO, existing
absolute robust completion checks remain authoritative. If a safe frontier is
nonempty, unsafe candidates do not replace it merely because their normalized
age or scalar efficiency is attractive.

### 5.3 No-SLO path

A request without an explicit SLO is never marked deadline-expired. It
contributes normalized service age. The first implementation substitutes that
dimensionless pressure at the current pressure-control points:

```text
serviceAgeQuanta < 1  -> normal V2 efficiency selection
serviceAgeQuanta >= 1 -> service recovery is eligible
```

`1` means one estimated normal service opportunity has been missed. It is a
dimensionless policy boundary, not a claim of parameter-free scheduling. The
initial gate does not tune this value by workload, model, or release.

Service age must not become unconditional strict oldest-first authority. V3
retains bounded cohort efficiency and transition reasoning so one marginal age
difference cannot fragment every batch.

### 5.4 Mechanism-only frontier

The expired-P guard may remain as audit telemetry during development, but a
fallback pseudo-SLO cannot delete standalone D from the V3 production frontier.
Explicit SLO risk is evaluated globally rather than through local phase-action
deletion.

Preserving a candidate is not equivalent to selecting it. V3 records:

- newly preserved candidates per decision;
- preserved D selected count;
- preserved D losing to P, P+D, E, or E+D;
- candidate construction and selection CPU cost.

### 5.5 Relationship to V2 transition

V3 reuses the existing bounded deterministic transition:

```text
candidate action
  -> concrete completion boundary
  -> request DAG transition
  -> newly ready E/P/D work
  -> canonical cohort formation
  -> KV/vision ownership transition
  -> second bounded boundary
```

The transition does not forecast arbitrary future arrivals. Service references
used by a rollout are copied from the source snapshot/epoch and cannot be
recomputed from the candidate that the rollout is evaluating.

## 6. Relationship to the contextual Scalar RLS

The V3 core keeps the V2 scalar reward:

```text
normalized advantage =
    (serial reference - observed action horizon) / serial reference
```

It also keeps the same generic calibration and online update contract. Batch
size, batch capacity, phase cost ratio, context bucket, work size, graph use,
residual anchor, and pair-relative slack remain observable features.

Current V2 already represents slack as approximately:

```text
minimumSlack / pairSerialCost
```

Therefore the RLS experiment is not `raw versus normalized`. It is:

```text
V2: pair-execution-relative slack
V3-RLS ablation: request-service-relative slack
```

The active V3 core initially retains the V2 feature so N1/N2 measure only
service scaling and frontier semantics. The RLS feature is changed only in the
separate N3 gate. No Completion-Vector, Decomposed Scalar, hierarchical RLS,
external registry, workload label, or model-name branch is introduced.

## 7. Staged implementation and experiment plan

Intermediate letters identify causal experiments, not permanent public policy
names.

| Variant | Internal queue scale | Synthetic no-SLO deadline | Policy pruning | RLS slack feature |
|---|---|---|---|---|
| A Current V2 | fixed 5/2 ms | present | current | pair-relative |
| B QueueScale | D service-scaled | present | current | pair-relative |
| C NoPseudoSlo | P/D service-scaled | absent | current | pair-relative |
| D MechanismFrontier | P/D service-scaled | absent | removed | pair-relative |
| E V3-RLS | P/D service-scaled | absent | removed | request-service-relative |

The target V3 release candidate is D unless E independently passes its gate.
Normalized all-late selection is not included in this table because it is a
later optional experiment.

### N-1. Freeze the contract

- Record the scoped 500/80/unspecified contract in the campaign manifest.
- Record source commit and dirty state, binary and engine identities, model,
  precision, KV pool, batch limits, chunk, graph mode, calibration, and request
  hashes.
- Keep request payloads unchanged for the first A/B; this avoids conflating a
  scheduler change with an HTTP contract change.
- Keep the previous service-normalized replacement authority disabled.

### N0. Telemetry-only characterization

No action may change. Add or derive per-decision telemetry for:

```text
fixed P fallback / P reference
fixed D target / D reference
explicit TPOT / D reference
reference source and service epoch
P pseudo-expired fraction
D pressure > 1 fraction
D candidate suppression and reason
P-only streak while D-ready
D-ready-but-non-D duration
D service-gap mean/p95/p99/max
all-late branch frequency
host decision p50/p95/max
```

Run current v0.10.1 V2 on the controlled mixed trace and the four-workload
screen: balanced, text-heavy, mixed, and multi-image. Reuse retained v0.10.0 V2
telemetry where its schema is complete; do not alter the protected comparison
worktree.

### N1. Normalize decode queue control

- Introduce the frozen decode service reference.
- Replace every policy use of the 2 ms internal decode scale, not the explicit
  80 ms TPOT contract.
- Audit dynamic decode batching, drain urgency, remaining wait budget, overlap
  interference allowance, TPOT telemetry pressure, refill/WAIT, admission, and
  P-ready control consumers.
- Keep P 5 ms behavior, candidate frontier, RLS features, and transition
  unchanged.
- First run unit/replay tests, then mixed 3--5 repeats, then the four-workload
  gate. Stop if the gate fails.

### N2-a. Remove the text pseudo-TTFT

- Do not convert an unspecified text TTFT into an absolute 5 ms deadline.
- Add/freeze the P-local service epoch.
- Use P-ready service age for local formation and request first-token service age
  for global producer protection.
- Retain explicit vision/per-request TTFT deadlines unchanged.
- Keep the old candidate frontier for this sub-step so the pseudo-SLO effect is
  isolated from frontier expansion.

### N2-b. Promote the mechanism-only frontier

- Preserve standalone P and D whenever they are hard-feasible.
- Move explicit SLO and urgency comparison to the global selector.
- Bound the total candidate set at the existing global candidate limit.
- Record candidate count, newly preserved action identity, selected count, and
  host decision overhead.
- Repeat mixed and the four-workload gate before full12.

Passing N1, N2-a, and N2-b establishes the minimum V3 contract.

### N3. RLS representation ablation

- Keep the same feature count if practical; replace only the slack projection.
- Compare pair-relative slack with request-service-relative slack.
- Reset and replay the same generic calibration for every process.
- Measure held-out prediction error, confidence/authority readiness, action
  disagreement, scheduler cost, and serving metrics.
- Promote the new feature only if it improves or preserves the full promotion
  gate. Otherwise V3 retains the existing V2 RLS representation.

### N4. Optional all-late normalization

Run only if N1--N3 leave a measured all-late regret. Do not revive the rejected
service-normalized replacement objective globally.

Candidate ranking is lexicographic:

1. number of additionally harmed explicitly protected requests;
2. maximum normalized additional violation;
3. upper-tail normalized additional violation;
4. Scalar+Transition efficiency;
5. stable candidate ID.

Every normalized violation uses an immutable request service reference. Do not
combine TTFT and TPOT in an unbounded weighted sum.

### N5. Final contract and cleanup

- Add `service-scaled-transition` to the runtime policy-mode parser only after
  N2 passes.
- Development environment flags are temporary ablation controls. Final V3 must
  be one named policy contract rather than a required collection of flags.
- Remove or keep shadow-only code according to its diagnostic value; do not
  enable the rejected service-normalized replacement authority.
- Update reproducible replay tooling and manifests.
- Commit each causal stage separately with DCO sign-off.

## 8. Correctness and unit gates

Required tests include:

### Service reference

- exact/interpolated/covering/static/cold provenance;
- selected candidate cannot change the denominator;
- every candidate at one snapshot shares the canonical request reference;
- reference changes apply only at the next service epoch;
- service age is monotonic within an epoch;
- token commit and P/E progress reset the correct epoch;
- cancel/release removes service state without reuse leakage.

### SLO semantics

- an unspecified SLO produces no finite absolute deadline;
- an explicit request SLO overrides the composition-root target;
- vision TTFT spans E to P without resetting submission age;
- decode TPOT spans last token commit to projected next token;
- E2E remains measurement-only;
- absolute-SLO safe candidates cannot be replaced by unsafe efficiency work.

### Candidate semantics

- SLO/pressure cannot remove a hard-feasible P or D action in V3;
- dependency, shape, inflight, ownership, and memory violations still reject;
- candidate frontier remains bounded and deterministically ordered;
- preview/shadow evaluation does not mutate online-learning or probe state;
- planned action equals the actual outstanding context set.

### V2 preservation

- V0/V1/V2 behavior and parser names remain unchanged;
- V3 with service scaling disabled is dispatch-identical to V2 in controlled
  deterministic replay;
- transition/DAG, KV release, vision release, and row-order tests remain valid;
- output token identity passes every controlled and HTTP gate.

## 9. Performance and promotion gates

### 9.1 Controlled gate

Use fixed snapshots/traces that expose:

- the retained 424.732 ms mixed decode-service gap;
- P-ready with D34--38;
- P-only, D-only, and P+D alternatives;
- E/P first-token critical path;
- dense-D and small-D cohort cases;
- multi-image trajectory fragmentation.

Measure equal useful-work completion, protected token completion, next cohort,
dispatch count, cumulative GPU work, and service debt. Independent fresh runs
are not described as same-state counterfactual replay.

### 9.2 Four-workload gate

Run balanced, text-heavy, mixed, and multi-image with three repeats per variant,
alternating run order where practical. Every row uses the same binary, engine,
request trace, calibration, memory, graph, batch, chunk, and output contract.

Stop a stage before full12 if any of the following occurs:

- token/output identity failure;
- action-fidelity failure;
- workload throughput regression worse than 3%;
- geometric-mean throughput regression worse than 1%;
- mixed, text-heavy, or multi-image TPOT p95 regression outside repeat noise;
- macro joint-SLO goodput regression;
- material scheduler-host p95 overhead regression;
- unbounded candidate growth or service-reference invalidity.

### 9.3 Full12 promotion

After the four-workload gate passes, run all twelve retained workloads with at
least three repeats:

```text
short
balanced
decode-heavy
long-prefill
bimodal
text-heavy
mixed
poisson
vision-heavy
wave-drain
multi-image
late-vision
```

Report:

- request and generated-token throughput;
- TTFT, TPOT, and E2E mean/p95;
- joint-SLO request/token goodput;
- phase dispatch and cohort distributions;
- P pseudo-expiration and D candidate suppression;
- P-only streak while D-ready;
- D service-gap mean/p95/p99/max;
- all-late and action-reason distribution;
- RLS observation/readiness/error when applicable;
- service-reference coverage/provenance;
- scheduler decision p50/p95/max;
- peak and retained GPU memory;
- exact output hashes.

The retained frozen vLLM result may be reused while the model, engine-facing
request/output contract, capacity, memory, and HTTP path remain unchanged. Rerun
vLLM if one of those contracts changes. V3 must be compared with V0, V1, and V2
from the same current binary.

### 9.4 Version robustness

Cross-runtime action similarity is diagnostic, not the primary objective. A
correct controller may choose different actions when measured execution costs
truly differ.

Primary version-robustness evidence is stable SLO goodput, latency, throughput,
and low action regret without retuning dimensionless policy semantics. Secondary
evidence includes P+D frequency, P-only streak, D-ready-but-non-D time,
all-late fraction, normalized pressure distributions, and trajectory agreement.

The frozen v0.10.0 reference worktree remains read-only. Implementing V3 in a
v0.10.0 comparison lineage requires an explicitly authorized backport and its
own commit, binary, manifest, and results. Do not modify the frozen reference in
place.

## 10. Baseline and expected comparison

The retained canonical full12 baseline is note275. Current v0.10.1 V2 is 0.39%
below v0.10.0 V2 in geometric-mean token throughput and 11.71% above frozen
vLLM, with 12/12 throughput wins. It also wins E2E mean and p95 against frozen
vLLM in 12/12, but not every TTFT or TPOT cell.

V3 is not justified by a requirement to increase overlap count. It succeeds if
it reduces timing-constant-induced policy amplification while retaining or
improving end-to-end service. In particular, it should recover decode continuity
without fragmenting P/E formation or losing V2's multi-image gains.

## 11. Risks and stop conditions

| Risk | Detection | Response |
|---|---|---|
| D selected too frequently | smaller D cohorts, P fragmentation, throughput loss | stop N1/N2; inspect recovery semantics |
| P/vision TTFT regression | first-token service age and E/P trajectory | retain explicit SLO safety; do not add workload rules |
| reference discontinuity | non-monotonic age, source churn | enforce service-epoch freeze |
| cold reference error | cold-only selections or unstable first epoch | fail closed or retain V2 until valid reference |
| circular policy input | denominator differs by candidate/selection | reject in unit test |
| candidate explosion | count/host p95 increase | retain bounded top-k mechanism frontier |
| prior service-authority regression returns | TTFT gain with TPOT/E2E loss | keep replacement authority disabled |
| no natural action changes | zero selected preserved candidates | retain telemetry; do not claim performance contribution |
| cross-version actions differ for valid cost reasons | lower regret despite lower agreement | prioritize service outcome over identity |

No workload name, model name, GPU name, prompt class label, or per-workload
threshold may be added to make a gate pass.

## 12. Non-goals

V3 does not:

- eliminate real user-provided TTFT or TPOT contracts;
- introduce a runtime E2E deadline;
- predict full incumbent/newcomer completion vectors;
- add a new external cost registry;
- forecast arbitrary future arrivals;
- change stable indexed-paged KV ownership;
- change P128, P8/D64/E4 engine capabilities, or memory capacity;
- tune dimensionless thresholds independently per workload;
- require more overlap or equal action traces across runtime versions;
- modify the frozen v0.10.0 comparison worktree.

## 13. Planned source and artifact boundaries

Likely source touchpoints are:

```text
cpp/runtime/phase/policy/
  phasePolicyMode.h
  phaseDeadline.h or a new focused service-state header

cpp/runtime/scheduling/
  phaseQueueScheduler.{h,cpp}
  phaseGlobalScheduler.cpp
  phaseThreeCoordinator.{h,cpp}
  independentPhaseAsyncServer.{h,cpp}

examples/llm/
  phaseSchedulerOptions.inc
  llm_phase_context_smoke.cpp

unittests/cpp/runtime/scheduling/
tests/python-unittests/
benchmarks/phase_serving/
```

New retained campaigns belong under:

```text
.local/results/v0101-forward-port/v3-service-scale-20260909/
```

Every retained campaign requires command/config, source dirty state, binary and
engine identity, workload and repeat count, summary paths, and a reference back
to this plan. Scratch and failed runs go under `.local/scratch/` and cleanup
remains dry-run/reference-driven.

## 14. Deliverables

V3 is complete only when all of the following exist:

1. a named `service-scaled-transition` policy with V0/V1/V2 unchanged;
2. explicit service-reference and service-epoch contracts;
3. no synthetic deadline for unspecified text TTFT;
4. mechanism-only bounded action frontier;
5. unit and controlled replay results;
6. three-repeat four-workload promotion result;
7. three-repeat full12 result if the four-workload gate passes;
8. V0/V1/V2/V3 same-binary comparison;
9. frozen-vLLM comparison or a documented valid reuse decision;
10. final architecture/result note stating which N3/N4 experiments were
    promoted or rejected.

Until these conditions pass, V3 remains an opt-in research policy and V2 remains
the canonical production comparison.
