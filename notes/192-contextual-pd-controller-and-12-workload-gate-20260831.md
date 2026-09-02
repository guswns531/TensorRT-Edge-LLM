# Contextual P+D Controller and 12-Workload Gate

## Scope

This revision replaces exact-key policy authority for P+D overlap with a
process-local, low-dimensional contextual value head. Exact CUDA timing
remains the execution/debug model. The contextual head does not own global
E/P/D action selection and does not use workload labels, external registries,
or TTL state.

The evaluation compares `TRT_EDGELLM_CONTEXTUAL_PD=active` with
`TRT_EDGELLM_CONTEXTUAL_PD=disabled` using the same binary, engine, trace,
warmup, and synchronized decode-sampling path.

## Implemented control path

```text
ready DAG / slack / ownership snapshot
                |
                v
     deterministic feasibility
                |
                v
       ordinary global selector
                |
                v
     contextual P+D value head
       - veto negative overlap
       - promote only during bounded recovery
       - never control an E/external-P critical path
                |
                v
             dispatch
                |
                v
 CUDA observation -> exact model + RLS update
```

The controller uses 16 continuous features, recursive least squares
uncertainty, a lower-confidence-bound decision value, and an EMA residual.
The bounded recovery path may use the posterior mean only when the selected
action is already late and P+D either protects decode-dominated work or drains
all currently ready final-chunk P work. It cannot override producer critical
paths.

## Final architecture boundary

The current architecture is **E-aware but not yet an E-contextual model**.
This distinction is intentional.

```text
 Request DAG and stable ownership
 text: P -> D -> D ...
 VLM : E -> external-P -> D -> D ...
             |
             v
 ┌──────────────────────────────────────────────────────────┐
 │ Mechanism plane                                          │
 │ dependency readiness, stable KV/vision leases,           │
 │ TensorRT profile compatibility, single-inflight context  │
 └──────────────────────────┬───────────────────────────────┘
                            v
 ┌──────────────────────────────────────────────────────────┐
 │ Global policy authority                                  │
 │ bounded E/P/D candidates, robust request slack, memory,  │
 │ serial/overlap/WAIT choice, outstanding-set fidelity     │
 └──────────────────────────┬───────────────────────────────┘
                            v
 ┌──────────────────────────────────────────────────────────┐
 │ P+D contextual value head                                │
 │ continuous features -> RLS mean/uncertainty -> LCB       │
 │ may veto P+D; may promote only bounded late recovery     │
 │ no authority while E/external-P is on critical path      │
 └──────────────────────────┬───────────────────────────────┘
                            v
                         Dispatch
                            |
              CUDA start/end events per action
                            v
 ┌──────────────────────────────────────────────────────────┐
 │ Observation plane                                        │
 │ action-fidelity + candidate-parity check                 │
 │ exact execution-cost update + contextual RLS update      │
 └──────────────────────────────────────────────────────────┘
```

E is already represented in dependency checks, protected first-token slack,
outstanding-context feasibility, E+P/E+D candidates, and the producer
critical-path guard. E is **not** yet represented by learned E+P or E+D value
heads. Therefore the current implementation cannot claim a unified learned
E/P/D controller; it is a globally E-aware scheduler with one promoted P+D
contextual component.

### Deterministic mechanism versus learned policy

Correctness never depends on the online model. The deterministic mechanism
continues to enforce:

1. `P_r` is dispatchable only after its required `E_r` completes.
2. KV or vision storage is reclaimable only after every GPU consumer
   completes.
3. Each TensorRT execution context has at most one in-flight enqueue.
4. Candidate shapes must fit the selected TensorRT optimization profile.
5. The physical outstanding context set must match the selected global
   action.

The contextual model only ranks or vetoes already legal actions. If evidence
is absent or uncertain, the ordinary global selector remains the safe
fallback.

## Detailed contextual P+D implementation

### Continuous feature projection

An exact P+D action is projected into 16 bounded features:

1. bias;
2. log-scaled isolated P duration;
3. log-scaled isolated D duration;
4. P share of serial P+D work;
5. shorter/longer phase duration ratio;
6. log-scaled P batch size;
7. log-scaled D batch size;
8. normalized P chunk length;
9. normalized P context bucket;
10. normalized D context bucket;
11. protected slack divided by serial work;
12. residual-augmentation flag;
13. P-residual anchor flag;
14. D-residual anchor flag;
15. primary CUDA-graph variant flag;
16. secondary CUDA-graph variant flag.

No workload name, request-rate label, or exact `(P batch, D batch, context)`
policy key is an input. Exact keys remain available only to the execution
cost/debug plane.

### Prediction and update

For feature vector `x`, the process-local RLS model predicts:

```text
mean        = theta^T x
uncertainty = sqrt(residual_variance * (1 + x^T covariance x))
LCB         = mean - confidence_beta * uncertainty
```

The normalized observation reward is:

```text
reward = (serial_reference_work - observed_overlap_makespan)
         / serial_reference_work
```

and is clipped before the rank-one RLS update. Residual variance uses a
bounded online average. The default forgetting factor is one: there is no TTL,
external cost registry, persisted workload profile, or background training
phase. Model state starts empty with the server process.

### When evidence is collected and consumed

1. Candidate construction projects the current P+D state and obtains
   `mean`, `uncertainty`, and `LCB`.
2. The global selector first performs dependency, memory, TensorRT-shape, and
   robust-SLO checks.
3. In normal safe states, positive P+D selection requires conservative LCB
   evidence. Negative evidence can veto a selected overlap.
4. Posterior-mean recovery is permitted only when every feasible action is
   already late and either decode violation dominates or the P action drains
   all currently ready final-chunk P work.
5. Recovery is disabled whenever E work or external-P lineage is on the
   first-token producer path.
6. After the dispatched kernels finish, CUDA-event measurements produce the
   exact action makespan and serial reference work.
7. The observation updates the contextual model only if selected-action
   identity, candidate parity, and actual outstanding-set fidelity all match.
   A mismatched execution is rejected rather than used as training evidence.

The next scheduler decision immediately sees the updated posterior. There is
no offline profile import and no periodic retraining boundary.

### Modes and code locations

| Component | Location | Responsibility |
|---|---|---|
| Contextual API/model | `cpp/runtime/phase/policy/phaseContextualPdModel.h`, `cpp/runtime/scheduling/phaseContextualPdModel.cpp` | feature projection, RLS posterior, uncertainty, recovery guards |
| Runtime cost plane | `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h`, `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` | keep exact cost and contextual evidence separate |
| Global candidate/selection | `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}`, `phaseGlobalScheduler.cpp` | create legal candidates and retain final E/P/D authority |
| Outstanding-context residual path | `cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}` | apply the same authority boundary while another phase is outstanding |
| Completion feedback | `cpp/runtime/scheduling/phaseDispatchWorker.cpp`, `phaseQueueScheduler.cpp` | propagate CUDA timing and update only faithful observations |
| Runtime configuration/telemetry | `examples/llm/llm_phase_context_smoke.cpp` | `disabled`, `shadow`, `active`; environment configuration and counters |
| Tests | `unittests/phaseRuntimeCostTrackerTest.cpp`, `phaseGlobalSchedulerTest.cpp`, `phaseQueueSchedulerTest.cpp` | learning sign, uncertainty, safety, candidate and action fidelity |

The public modes are controlled by `TRT_EDGELLM_CONTEXTUAL_PD`. Shadow mode
collects predictions and disagreements without changing action choice. Active
mode enables the bounded value-head authority described above.

External vision-origin prefill remains excluded from the P+D head. E-aware
action authority is now provided by independent E+P and E+D contextual heads;
their architecture and validation are recorded in
`notes/193-contextual-ep-ed-controller-and-gate-20260831.md`. The P+D head still
cannot delay or reorder E work.

## Validation history

- The first direct candidate-cost override fragmented P/D cohorts in bimodal
  and Poisson traces.
- Restricting the contextual model to local value authority recovered those
  traces.
- A broad deadline-recovery rule improved the 48.8 request/s saturation point
  but regressed long prefill.
- Requiring either decode-dominated recovery or complete ready-P drain kept
  the saturation benefit and recovered long-prefill parity.
- Contextual prediction was removed from external-P candidates after VLM
  traces exposed an action-authority leak.

## Final 12-workload active-versus-disabled gate

Positive percentages are improvements. Throughput is higher-is-better; all
latency columns have their signs normalized so positive is also better.
The primary run is three repeats. Initially failing traces were repeated five
times with alternating active/disabled launch order.

| Workload | Token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Token identity | Result |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| short | +0.59% | -2.01% | +2.39% | +2.84% | +18.24% | +0.85% | +0.59% | exact | pass |
| balanced | -1.66% | +2.59% | +0.15% | -1.87% | -2.66% | -1.56% | -1.42% | exact | pass |
| decode-heavy | -0.48% | +0.36% | -0.27% | -0.64% | -0.67% | -0.47% | -0.36% | exact | pass |
| long-prefill | +0.12% | +0.19% | +5.14% | +0.27% | +1.45% | +0.21% | -0.54% | exact | pass |
| bimodal | -1.57% | -1.26% | -1.91% | -1.15% | +0.42% | -1.58% | -2.79% | exact | pass |
| text-heavy | +0.16% | +0.33% | -0.35% | -0.15% | +0.26% | -0.08% | -0.15% | exact | pass |
| mixed | -0.11% | -2.04% | -0.09% | +12.56% | +10.00% | +8.05% | -0.09% | exact | pass |
| vision-heavy | -0.08% | +0.06% | -0.18% | +0.04% | -0.24% | +0.07% | -0.08% | exact | pass |
| poisson | +3.71% | -1.80% | +1.02% | +3.29% | -0.31% | +4.76% | +3.86% | exact | pass |
| wave/drain | +0.01% | -3.44% | -0.57% | -2.48% | -8.82% | -5.12% | -0.33% | exact | external-P/no-op |
| multi-image | +9.18% | -5.89% | +3.18% | -7.55% | -11.68% | -5.41% | +2.43% | unstable | external-P/no-op |
| late-vision | -0.23% | -0.01% | -0.26% | -0.22% | -0.18% | -0.15% | -0.19% | exact | pass |

Ten policy-eligible/numerically stable traces pass the strict 3% regression
gate. The two remaining traces are external-P workloads for which the
contextual head has zero policy authority. Their active and disabled paths
show the same scheduling bifurcation: wave/drain alternates between roughly
210 ms and 330--427 ms TTFT modes, while five-request multi-image alternates
between roughly 0.49 s and 0.71 s completion modes. The multi-image baseline
also changes one FP16 greedy branch in one disabled repeat. These are recorded
as VLM timing/numerical determinism blockers rather than P+D policy failures;
claiming a strict numerical 12/12 pass would be inaccurate.

## Frozen vLLM comparison

The same existing vLLM 12-trace run is reused; no redundant vLLM rerun was
performed. Current token throughput is higher in 11/12 traces (wave/drain is
-0.9%), mean E2E latency is lower in 12/12, and E2E p95 is lower in 10/12.

| Workload | Current token/s vs vLLM | Current / vLLM TTFT mean (ms) | Current / vLLM TPOT mean (ms) | Current / vLLM E2E mean (ms) | Current / vLLM E2E p95 (ms) |
|---|---:|---:|---:|---:|---:|
| short | +26.4% | 86.9 / 175.1 | 13.3 / 13.4 | 329.2 / 425.0 | 408.9 / 503.7 |
| balanced | +4.9% | 64.2 / 147.4 | 12.1 / 15.2 | 1098.3 / 1437.8 | 1697.4 / 2244.1 |
| decode-heavy | +8.2% | 65.1 / 154.5 | 10.6 / 14.0 | 2802.3 / 3768.8 | 4270.8 / 5812.4 |
| long-prefill | +4.9% | 2069.3 / 2947.6 | 27.2 / 32.3 | 4403.4 / 5694.2 | 6316.4 / 7860.8 |
| bimodal | +2.2% | 1892.0 / 2513.5 | 18.2 / 23.7 | 4387.7 / 5641.8 | 9045.5 / 10328.2 |
| text-heavy | +19.4% | 342.4 / 427.7 | 25.2 / 29.3 | 1645.5 / 1943.7 | 1726.9 / 2037.8 |
| mixed | +17.2% | 822.9 / 882.4 | 25.4 / 47.1 | 2022.7 / 3004.8 | 2646.5 / 3140.9 |
| vision-heavy | +12.2% | 1560.2 / 1710.7 | 22.1 / 63.7 | 2386.1 / 4119.1 | 3718.2 / 4229.4 |
| poisson | +9.4% | 229.7 / 580.9 | 21.6 / 19.5 | 1590.1 / 1811.0 | 2043.7 / 2266.6 |
| wave/drain | -0.9% | 337.8 / 253.1 | 7.3 / 12.4 | 574.3 / 638.5 | 710.0 / 649.4 |
| multi-image | +3.9% | 283.3 / 265.7 | 7.7 / 12.2 | 539.9 / 642.6 | 629.0 / 653.9 |
| late-vision | +7.3% | 174.9 / 156.0 | 9.3 / 7.4 | 1510.1 / 1574.8 | 1822.0 / 1954.5 |

### Repeated 48.8 request/s saturation result

The earlier single-run statement that Current slightly exceeded vLLM
throughput is not retained. Both runtimes were rerun five times on the same
materialized trace. A second Current run also matched the client
`max-in-flight=80` used by vLLM; this equal-cap result is the headline below.

| Metric | Current median | Current min--max | vLLM median | vLLM min--max | Current advantage |
|---|---:|---:|---:|---:|---:|
| Request throughput | 40.472 req/s | 39.327--40.633 | 40.751 req/s | 40.642--40.935 | -0.68% |
| Token throughput | 3507.62 tok/s | 3408.37--3521.51 | 3531.74 tok/s | 3522.34--3547.68 | -0.68% |
| TTFT mean | 94.57 ms | 81.42--134.58 | 50.49 ms | 49.52--51.25 | -87.31% |
| TTFT p95 | 282.57 ms | 253.26--369.91 | 81.17 ms | 78.61--83.52 | -248.12% |
| TPOT mean | 13.60 ms | 13.50--14.62 | 13.74 ms | 13.40--13.82 | +1.04% |
| TPOT p95 | 15.71 ms | 15.53--17.09 | 17.92 ms | 17.33--18.07 | +12.36% |
| E2E mean | 1251.04 ms | 1234.64--1366.58 | 1223.00 ms | 1193.21--1230.10 | -2.29% |
| E2E p95 | 2000.51 ms | 1960.36--2147.94 | 2146.16 ms | 2089.86--2181.45 | +6.79% |

vLLM passes the joint SLO (`TTFT <= 500 ms`, `TPOT <= 50 ms`,
`E2E <= 2500 ms`) for 100% of requests in all five runs. Current passes 100%
in three runs and 287/288 requests (99.653%) in two runs. Median request
goodput is therefore 40.472 req/s for Current and 40.751 req/s for vLLM.

The equal-cap conclusion is nuanced: raw throughput is near parity with vLLM
ahead by 0.68%; Current retains better TPOT and E2E p95, but loses mean TTFT,
TTFT p95, and mean E2E. This is not a P+D kernel-cost regression. It exposes a
server admission/backpressure sensitivity at the 64--80 active-request
boundary.

For diagnosis, retaining the earlier Current client cap of 64 produces 40.639
req/s, 29.85/46.21 ms TTFT mean/p95, 13.42/15.26 ms TPOT, and
1172.24/1917.37 ms E2E, with 100% joint-SLO pass in all five runs. That cap is
not a fair headline against vLLM at 80, but it establishes that bounded active
admission recovers the lost latency. The next implementation should reproduce
this effect inside the server using instantaneous memory/service/slack state,
not an external workload-specific cap.

Current uses 7,799 MiB at this text-only point versus vLLM's 9,041 MiB, a
1,242 MiB (13.7%) reduction under these frozen configurations. This memory
comparison includes each runtime's full resident allocation and must not be
interpreted as KV-only savings.

Artifacts:

- `.local/transition-aware-20260830/p54-contextual-pd-saturation-repeat-5x/current`
- `.local/transition-aware-20260830/p54-contextual-pd-saturation-repeat-5x/vllm`
- `.local/transition-aware-20260830/p54-contextual-pd-saturation-repeat-5x/slo-goodput.json`
- `.local/transition-aware-20260830/p55-contextual-pd-saturation-matched-5x/current`
- `.local/transition-aware-20260830/p55-contextual-pd-saturation-matched-5x/slo-goodput.json`

## Verification

- Focused unit tests: 170/170 passed.
- `git diff --check`: passed.
- Token identity: exact for all stable text traces and the repeated
  wave/drain trace.
- Known blocker: multi-image FP16 greedy identity is not deterministic even
  within the disabled baseline.

## Next gate

### P+D freeze and observability

1. Move the cap-64 benefit into profile-free server admission. Admit work only
   while predicted decode service and protected slack remain safe; use current
   ready state and observed service cost rather than a fixed workload mode.
2. Repeat cap 64/72/80 as a characterization sweep, then require dynamic
   admission at client cap 80 to recover cap-64 TTFT without reducing median
   goodput below the equal-cap baseline.
3. Add explicit per-request contextual prediction/selection telemetry to the
   HTTP benchmark artifact so external-P zero-authority is machine-checked.
4. Replace tiny VLM percentage gates with a repeated-arrival version of the
   same trace or a confidence-interval equivalence test; do not tune policy to
   five-request timing modes.
5. Measure whole-selector and contextual-head CPU decision latency separately;
   retain the hot-path gate of p95 below 50 microseconds for the value head.

### E-aware contextual completion

6. Add an E+D value head with E batch/input-token cost, D cohort/context,
   first-token slack, and successor E/D formation features.
7. Add an E+P value head with vision payload lifetime, P external-lineage
   state, E/P isolated costs, and successor cohort preservation.
8. Keep one global selector. E+D, E+P, and P+D heads provide `(mean,
   uncertainty)` values; none may directly dispatch or override deterministic
   feasibility.
9. Permit only bounded E+D and E+P exploration under large protected slack.
   Do not explore E+P+D in the first version.
10. Extend observation gating so CUDA action fidelity and ownership transitions
   must match before any E-related posterior update.

### Validation and research gates

11. Re-run the controlled E1/E2/E4/E8 overlap matrix and require the same
   selector to choose profitable small-E overlap and reject harmful large-E
   fragmentation without workload labels.
12. Re-run all 12 production traces and the repeated 48.8 request/s point.
    Compare against Current-P+D, the best static policy, clean v0.10, and the
    frozen vLLM result.
13. Report SLO goodput, TTFT/TPOT/E2E mean and p95, token throughput, memory,
    action-fidelity violations, selection regret, and value-head CPU cost.
14. Promote the unified model only if it preserves the P+D gate, improves at
    least one natural E-heavy trace, and remains within 3% on every stable
    workload. External-P numerical nondeterminism remains a separate
    correctness project rather than a reason for workload-specific policy
    tuning.
