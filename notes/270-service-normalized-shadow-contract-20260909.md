# Service-normalized shadow: contract and first implementation

## Status

Implemented an **offline prototype**, not an active runtime policy or a completed
V2 integration. Production scheduling, SLO values, RLS, engines and KV allocation
are unchanged. No fresh GPU benchmark was run in this step.

Files:

- `benchmarks/phase_serving/service_normalized_shadow.py`: pure snapshot evaluator
  and retained-event observability audit.
- `tests/python-unittests/test_service_normalized_shadow.py`: synthetic contract tests.
- Existing note269 mixed logs are the only real traces analyzed.

This intentionally separates a testable evaluation contract from the runtime
producer. Existing metadata must not be silently stretched into missing
request-level physical outcomes.

## Representation

Every snapshot contains one fixed protected request set, a common work target,
and canonical isolated execution references. For each request:

- D milestone: next token; elapsed time since the previous host-visible token.
- E/P milestone: first token; elapsed time since submission, preserved across E→P.
- `reference_us`: snapshot-fixed isolated execution reference, not queueing time
  and not an action-specific or contended execution cost.
- `reference_cohort_signature`: identifies the canonical reference cohort.

Every hard-feasible candidate must provide a predicted relative service completion
and uncertainty for **every protected request**, including requests outside its
immediate batch. Each candidate must cover the same work target, although its
predicted horizon duration may differ. Unknown service beyond the modeled horizon
is rejected rather than assigned zero delay.

The producer must ensure completions are relative to the decision's host-monotonic
service clock, including realization/completion-visibility delay as appropriate.
The consumer validates schema consistency, not the truth of caller-provided timing
estimates or opaque work/cohort signatures. GPU epoch timestamps cannot be passed
as host timestamps merely by changing the clock-domain label.

## Calculations and authority

For request r, reference c, elapsed w, predicted delay t and uncertainty u:

```text
current normalized age     = w / c
additional normalized age  = (t + u) / c
age at service             = (w + t + u) / c
```

All three are retained; using only the increment would cancel accumulated waiting.
There is no urgency clamp. Missing/zero/nonfinite references and numeric overflow
are errors, not silent fallback values.

The output includes max/mean age at service, max additional age, and the full
request vector. A Pareto set over max age, mean age and equal-work horizon is
reported. An explicitly named diagnostic lexicographic choice is also reported
for controlled comparison; it has **no production authority**. No claim is made
that this max→mean→horizon ordering is the final policy or is starvation-free.

Important limitation: this is a first-service-milestone diagnostic. It does not
model subsequent recurrent decode tokens, arbitrary future arrivals, recurrent
service debt resets, or long-run trajectory utility. It cannot by itself promote
a policy, establish SLO goodput, or justify replacing V2.

## Tests

11 new tests pass:

1. Old decode urgency is preserved rather than canceled by increments.
2. Old prefill urgency can reverse the diagnostic preference.
3. Uniform scaling of all times preserves normalized scores and ranking.
4. Unselected requests still require service projections.
5. Action-specific denominators are rejected.
6. Invalid references, queue-derived references and mixed clocks are rejected.
7. Duplicate request identities are rejected.
8. Unequal work targets and insufficient/nonfinite horizons are rejected.
9. Completion uncertainty contributes to predicted service age.
10. E→P preserves first-token milestone; encoder completion cannot substitute for it.
11. Evaluating a snapshot does not mutate it.

The fixtures are synthetic P→D versus D→P examples, not observed performance gains.
Uniform scaling is a useful consistency invariant; it is not evidence of
cross-model/GPU generalization under nonuniform kernel/tactic changes.

## Retained-log audit

Inputs: note268 old/current same-engine V1 mixed runs, one each.
Output: `.local/results/v0101-forward-port/version-activity-20260909/service-normalized-coverage-v1.json`.

| Coverage | Old runtime | Current |
|---|---:|---:|
| Measurement decisions | 121 | 126 |
| Explicit local guard audits | 0 | 126 |
| Standalone D suppressed | unavailable | 33 |
| Suppressed while P and D both expired | unavailable | 25 |
| Complete service-normalized snapshot | 0 | 0 |
| Scored real counterfactual decisions | 0 | 0 |

Old suppression is **unknown**, not zero. The new snapshot field is absent from
both binaries, as expected; the audit refuses to derive missing elapsed time from
configured SLO slack or to copy the chosen action's outcome to unchosen actions.
These logs contain neither complete per-request successor service projections nor
a reconstructed mechanism-only frontier. Therefore no claimed action improvement
or throughput/latency benefit is reported in this step.

## Next runtime integration, in order

1. Add a gated immutable diagnostic snapshot at the existing V2 decision boundary.
   Reuse request clocks; retain submission age across E→P and next-token age for D.
   Keep clock observations separate from queue-ready timestamps.
2. Preserve a bounded shadow mechanism frontier before policy pruning. Reuse
   actual compatibility/ownership checks and canonical batching. Do not simply
   toggle the active guard or call a mutating preview on the live scheduler.
3. Expose snapshot-fixed isolated references and cost provenance. Sample scarcity
   must be visible; do not infer missing reference costs from SLO targets.
4. Adapt existing H2 service predictions to the per-request contract where valid.
   Existing aggregate protected-D estimates are insufficient to fill every row by
   duplication. Mark unsupported projections unavailable, including residual paths
   that V2 currently excludes.
5. Run V1/V2 old/current repeated mixed diagnostics in alternating order; record
   coverage, normalized outcomes, frontier changes, host instrumentation overhead,
   E/P/D masks and request-level TTFT/TPOT/E2E mean/p95.
6. Only after valid projections and a controlled same-state/equal-work replay,
   consider active service-normalized selection. Reuse frozen vLLM while its
   workload/output contract remains unchanged.

The full runtime integration, repeated GPU comparison, restored candidate catalog,
and active gate are **not completed** by this prototype. No new RLS model or
additional serving SLO is introduced.

## Commands

From the v0.10.1 worktree:

```bash
LLM_SDK_DIR=$PWD python3 -m unittest discover -s tests/python-unittests -p test_service_normalized_shadow.py -v
python3 benchmarks/phase_serving/service_normalized_shadow.py --snapshot INPUT.json --output NEW_RESULT.json
python3 benchmarks/phase_serving/service_normalized_shadow.py --events OLD_EVENTS.jsonl CURRENT_EVENTS.jsonl --output NEW_COVERAGE.json
```

Outputs use exclusive creation to avoid overwriting retained results. No model,
engine or previous experiment artifact was deleted or modified.
