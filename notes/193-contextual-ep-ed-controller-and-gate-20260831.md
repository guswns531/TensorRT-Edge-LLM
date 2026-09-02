<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Contextual E+P/E+D Controller: Architecture, Implementation, and Gate

## Scope and conclusion

This revision extends the process-local contextual P+D controller to the two
encoder pair actions, E+P and E+D. The implementation is complete through
candidate projection, independent online models, SLO-safe action authority,
faithful CUDA-event feedback, configuration, telemetry, unit tests, controlled
GPU validation, and a paired 12-workload run.

The main architectural result is:

```text
deterministic mechanism decides legality
                 +
independent continuous value heads estimate profitability
                 +
the global selector retains SLO and action-fidelity authority
```

The implementation does **not** introduce workload names, an exact-key policy
table, an external cost registry, TTLs, or offline policy import. P+D, E+P, and
E+D share feature math but never share learned evidence.

The controlled traces exercise active E+P/E+D decisions and show the expected
shape-dependent signs. The short production traces do not yet collect enough
faithful encoder-pair observations to make either E head ready. Consequently,
the 12-workload paired result is primarily a no-regression/cold-start gate, not
evidence that the E heads improve natural production traces. This distinction
is essential.

## Final architecture

```text
                          Global snapshot
                E/P/D ready state, slack, ownership,
                 outstanding contexts and completion events
                                  |
                                  v
                    Hard feasibility and invariants
             DAG dependency / TRT shape / memory / single-inflight
                                  |
                                  v
                         Bounded candidates
              E, P, D, WAIT, P+D, E+P, E+D (no triple action)
                                  |
                 +----------------+----------------+
                 |                |                |
                 v                v                v
          exact CUDA cost    16-D continuous   protected request
          and uncertainty      projection       completion/slack
          (mechanism data)   (policy state)       (hard safety)
                 |                |
                 |       +--------+--------+
                 |       |        |        |
                 |       v        v        v
                 |      P+D      E+P      E+D
                 |      RLS      RLS      RLS
                 |      head     head     head
                 |       |        |        |
                 |       +--------+--------+
                 |                |
                 |          mean / uncertainty
                 |          LCB = mean - beta*sigma
                 |                |
                 +----------------+----------------+
                                  |
                                  v
                      Global SLO-safe selector
               negative ready head may veto pair overlap
               positive ready head may promote only when
               hard-feasible and no worse for protected SLO
                                  |
                                  v
                       Explicit action lease/dispatch
                                  |
                  planned outstanding set == actual set
                                  |
                                  v
                       CUDA start/end observations
                                  |
                                  v
               normalized equal-work action advantage
                                  |
              update only the matching P+D / E+P / E+D head
```

This preserves the mechanism/policy boundary:

- The exact cost plane models concrete engine actions and remains available
  for feasibility, diagnostics, and scheduler timing.
- The contextual plane generalizes decisions across nearby continuous shapes.
- The global selector owns deadlines, dependency ordering, ownership, and the
  actual outstanding-context configuration.
- The coordinator updates a model only after CUDA events prove that the
  observed pair action was faithfully executed.

## Continuous representation

All pair actions use the same bounded 16-dimensional projection:

1. bias;
2. log primary isolated time;
3. log secondary isolated time;
4. primary share of serial work;
5. shorter/longer phase duration ratio;
6. normalized primary batch size;
7. normalized secondary batch size;
8. normalized pair work size;
9. primary context bucket;
10. secondary context bucket;
11. protected slack/serial-work ratio;
12. residual-augmentation indicator;
13. P-anchor indicator;
14. D-anchor indicator;
15. primary CUDA-graph indicator;
16. secondary CUDA-graph indicator.

For P+D, primary/secondary mean P/D. For E+P they mean E/P, and for E+D they
mean E/D. The posterior is independent per family, so a profitable E1+D32
sample cannot make E8+P8 or P+D more aggressive.

Each head uses fixed-size recursive least squares. With feature vector `x`,
posterior mean `theta`, and covariance `P`:

```text
mean        = theta^T x
uncertainty = sqrt(residual_variance * (1 + x^T P x))
LCB         = mean - beta * uncertainty
```

The update uses a Sherman-Morrison/RLS step and performs no allocation on the
scheduler hot path. The default readiness threshold is four faithful
observations. State is process-local and is reset only with the serving epoch;
there is no TTL, persistent registry, or workload-profile lookup.

## Reward and the corrected serial reference

The learned reward is normalized equal-work compression:

```text
reward = (isolated_pair_serial_us - observed_pair_makespan_us)
         / isolated_pair_serial_us
```

For E+P and E+D, `isolated_pair_serial_us` is the isolated cost of the actual
encoder batch plus the isolated cost of the actual P or D batch. It is not the
throughput-oriented encoder reference that multiplies singleton E cost by the
batch size. The first implementation reused that throughput reference and
incorrectly reported roughly `+0.96` reward for E8+D32. The corrected reference
produces an action-level compression reward and keeps execution-cost accounting
separate from policy learning.

## Authority rules

The public modes are independent:

```text
TRT_EDGELLM_CONTEXTUAL_PD=disabled|shadow|active
TRT_EDGELLM_CONTEXTUAL_EP=disabled|shadow|active
TRT_EDGELLM_CONTEXTUAL_ED=disabled|shadow|active
```

Each head also accepts family-specific `MIN_OBSERVATIONS` and
`CONFIDENCE_BETA` settings. These are confidence controls, not workload
profiles.

The action rules are:

1. The ordinary global selector first chooses among hard-feasible candidates.
2. Cold evidence cannot control the action; the existing bounded calibration
   probe is the only exploration path.
3. A ready active head with non-positive LCB may veto its selected pair action.
4. A ready active head with positive LCB may promote its pair only when the
   candidate remains hard-feasible and its robust protected-request violation
   is no worse than the currently selected action.
5. If both E+P and E+D are safe and positive, the larger LCB wins.
6. `shadow` records disagreement but never changes selection.
7. Experimental static overlap percentages remain later overrides solely for
   controlled characterization; they are not part of the production policy.

There is no E+P+D action. One TensorRT execution context remains single-inflight,
and an action lease names the complete outstanding context set until the next
decision boundary.

## Code map

| Area | Files | Responsibility |
|---|---|---|
| Pair feature/model API | `cpp/runtime/phase/policy/phaseContextualPdModel.h` | common pair kind/input, 16-D representation, RLS contract |
| Pair model implementation | `cpp/runtime/scheduling/phaseContextualPdModel.cpp` | projection, prediction, uncertainty, RLS update |
| Three independent heads | `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h`, `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` | separate P+D, E+P, E+D posterior state and telemetry |
| Candidate state | `cpp/runtime/scheduling/phaseQueueScheduler.h` | pair features, readiness, mean, uncertainty, LCB, serial reference |
| Global E action policy | `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | E+P/E+D candidate projection, veto/promotion, best-LCB arbitration |
| Faithful completion feedback | `cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}` | preserve selected feature/reference and update after both CUDA events complete |
| Environment and telemetry | `examples/llm/llm_phase_context_smoke.cpp` | EP/ED modes, confidence controls, PHASE_METRIC counters |
| Tests | `unittests/phaseRuntimeCostTrackerTest.cpp` | common projection and independent-head evidence |

The source filename retains `PdModel` to avoid a risky mechanical rename in the
current forward-port. Its model class and projection are now pair-generic; the
P+D-specific helper guards remain P+D-only.

## Correctness and build validation

The final focused suite passes:

```text
177 tests from 5 suites: PASS
```

The suite covers the contextual model, runtime cost tracker, global scheduler,
queue scheduler, and pair-model isolation. The C++ runtime and
`llm_phase_context_smoke` build successfully against TensorRT 11.0/CUDA 13.3.
`git diff --check` is clean.

Every final real-request run reports:

```text
vision_global_action_fidelity_violations = 0
```

## Controlled GPU validation

Artifacts:

- `.local/transition-aware-20260830/p57-contextual-eped-correct-reference/`
- `.local/transition-aware-20260830/p63-contextual-ep-correct-reference/`
- `.local/transition-aware-20260830/p0-controlled-e32/traces/ep-d8.json`
- `.local/transition-aware-20260830/p0-controlled-e32/traces/ed-d32.json`

| Controlled case | Observations | Learned mean | LCB | Last reward | Pair selections | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| E1+P8 | 14 | +0.269 | +0.038 | +0.259 | 5 positive, 9 negative | converges to a small confident gain |
| E8+P8 | 4 | +0.002 | -0.234 | +0.068 | 0 positive | no confident gain; not promoted |
| E1+D32 | 50 | +0.356 | +0.321 | +0.410 | 28 positive | small E+D is profitable |
| E8+D32 | 11 | +0.246 | +0.119 | +0.264 | 8 positive, 1 negative | immediate compression is positive on this trace |

The E8 results are not evidence that every large-E overlap is profitable. The
head learns immediate equal-work compression; the existing global formation
horizon and protected-request completion model still own successor
fragmentation and SLO effects. E8+P8 reaches the readiness threshold but its
LCB remains negative, so uncertainty prevents promotion. The E+P run uses a
safe-probe interval of one only to obtain controlled coverage; production keeps
the ordinary bounded probe interval.

## Paired 12-workload gate

The authoritative extension comparison uses the same latest binary, tied
P8/D64 engine, trace, container, launch settings, and three repeats:

- baseline: P+D active, E+P/E+D disabled;
- active: P+D, E+P, and E+D active.

Positive percentages are improvements. Latency signs are normalized so that a
positive value means lower latency.

| Workload | Token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Active deterministic |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| short | +0.05% | -3.42% | -0.22% | +1.69% | -2.64% | -1.90% | -1.51% | yes |
| balanced | +1.60% | +1.37% | -0.95% | +2.11% | +3.30% | +2.00% | +1.70% | yes |
| decode-heavy | -0.18% | +0.26% | +0.45% | -0.25% | +0.78% | -0.02% | -0.29% | yes |
| long-prefill | -0.62% | -0.25% | -6.08% | -1.04% | -2.16% | -0.50% | -4.11% | yes |
| bimodal | -0.12% | +0.26% | +1.84% | -0.15% | +3.48% | +0.11% | +0.53% | yes |
| text-heavy | +0.06% | -0.70% | +0.27% | +0.45% | -0.32% | +0.14% | +0.15% | yes |
| mixed | -0.30% | +0.95% | +2.36% | +1.87% | -2.10% | +0.33% | +2.93% | no, both modes unstable |
| vision-heavy | +0.30% | -1.67% | -1.78% | +2.01% | +0.26% | +0.81% | +0.52% | yes |
| poisson | +0.07% | +0.08% | -0.05% | -0.22% | +0.29% | -0.00% | -0.30% | yes |
| wave/drain | +0.67% | -7.49% | +0.73% | +4.00% | +1.02% | -1.53% | +0.54% | yes |
| multi-image | +0.25% | +0.13% | -0.13% | +4.46% | +6.04% | +0.25% | +0.23% | no |
| late-vision | +0.21% | -0.27% | -0.12% | +0.21% | +0.23% | +0.20% | +0.21% | yes |

Artifacts:

```text
.local/transition-aware-20260830/p59-contextual-eped-final-12x3/
```

### Why this is a cold-start gate

The maximum final per-run E-head telemetry was:

| Workload class | E+P observations | E+D observations | Ready/policy selections |
|---|---:|---:|---:|
| five text-only traces | 0 | 0 | 0 |
| late-vision, mixed, text-heavy | 0 | 1 | 0 |
| poisson | 0 | 3 | 0 |
| multi-image | 1 | 1 | 0 |
| wave/drain | 2 | 2 | 0 |
| vision-heavy | 0 | 0 | 0 |

The benchmark starts a new server for every repeat, while the production model
is intentionally process-local. With a four-observation readiness threshold,
no natural trace reached ready state. Therefore the table must not be presented
as an E-policy speedup. It proves that enabling cold E heads does not alter
legality or action fidelity; most differences are run variance.

### Targeted five-run repeats

The largest initial tail differences were rerun with alternating baseline and
active launch order, five times each.

| Workload | Token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| long-prefill | +0.17% | +0.46% | -6.72% | +0.43% | -0.32% | +0.35% | -0.96% |
| wave/drain | -0.98% | +5.77% | +0.21% | -2.35% | -0.18% | +2.18% | -0.12% |
| multi-image | +0.17% | +1.23% | -3.86% | +1.48% | -3.49% | +0.15% | -0.73% |

Artifacts:

```text
.local/transition-aware-20260830/p60-contextual-eped-targeted-5x/
```

Long-prefill TTFT p95 remains noisy even though it has no encoder requests and
the E heads make zero selections. Wave/drain and multi-image also retain known
small-sample/bimodal timing or FP16 greedy instability. These are not attributed
to E+P/E+D action authority. A strict numerical `12/12 under 3%` claim would be
incorrect; the extension passes the mechanism/action-fidelity gate and the
controlled active-policy gate, while the natural-policy benefit gate remains
open.

## Frozen vLLM anchor

The vLLM server was not rerun because the materialized traces and vLLM
implementation are unchanged. The following uses the frozen vLLM measurements
as a directional anchor, not as the paired E-head ablation. Current values are
from the combined-active p59 run.

| Workload | Current token/s vs vLLM | Current/vLLM TTFT mean (ms) | Current/vLLM TPOT mean (ms) | Current/vLLM E2E mean (ms) | Current/vLLM E2E p95 (ms) |
|---|---:|---:|---:|---:|---:|
| short | +20.7% | 98.0 / 175.1 | 13.3 / 13.4 | 347.0 / 425.0 | 430.7 / 503.7 |
| balanced | -5.2% | 315.9 / 147.4 | 13.8 / 15.2 | 1493.4 / 1437.8 | 2334.5 / 2244.1 |
| decode-heavy | -0.4% | 703.3 / 154.5 | 11.6 / 14.0 | 3705.3 / 3768.8 | 5783.7 / 5812.4 |
| long-prefill | -2.7% | 3269.6 / 2947.6 | 29.7 / 32.3 | 5797.3 / 5694.2 | 8221.5 / 7860.8 |
| bimodal | -2.7% | 2849.6 / 2513.5 | 18.9 / 23.7 | 5469.2 / 5641.8 | 11562.0 / 10328.2 |
| text-heavy | -12.6% | 182.6 / 427.7 | 17.6 / 29.3 | 1105.9 / 1943.7 | 1432.8 / 2037.8 |
| mixed | +0.8% | 376.5 / 882.4 | 22.2 / 47.1 | 1393.0 / 3004.8 | 1763.0 / 3140.9 |
| vision-heavy | +7.9% | 739.6 / 1710.7 | 25.3 / 63.7 | 1714.3 / 4119.1 | 1974.8 / 4229.4 |
| poisson | -20.1% | 131.4 / 580.9 | 16.1 / 19.5 | 1254.5 / 1811.0 | 2274.7 / 2266.6 |
| wave/drain | +0.8% | 285.2 / 253.1 | 9.4 / 12.4 | 578.7 / 638.5 | 704.9 / 649.4 |
| multi-image | +1.5% | 367.8 / 265.7 | 9.5 / 12.2 | 662.8 / 642.6 | 708.6 / 653.9 |
| late-vision | +4.8% | 124.8 / 156.0 | 9.5 / 7.4 | 1489.2 / 1574.8 | 1866.2 / 1954.5 |

The frozen comparison shows that the current launch contract is not uniformly
better than vLLM, especially at balanced/text-heavy/Poisson throughput and
several TTFT points. This is not caused by the E-head extension: the paired
baseline has almost identical throughput and the natural E heads make no
policy selections.

The separately repeated matched-capacity 48.8 request/s result remains in
`notes/192-contextual-pd-controller-and-12-workload-gate-20260831.md`: Current
median is 40.472 req/s versus vLLM 40.751 req/s, with better Current TPOT p95 and
E2E p95 but substantially worse TTFT. E+P/E+D do not participate in that
text-only trace.

## What is complete and what remains

Complete:

1. pair-generic continuous feature projection;
2. independent P+D, E+P, and E+D process-local RLS heads;
3. uncertainty-aware LCB and per-family modes;
4. SLO-safe veto/promotion inside global authority;
5. corrected equal-work encoder-pair reward;
6. CUDA-event completion feedback and action-fidelity checks;
7. telemetry and focused unit coverage;
8. controlled active E+P/E+D validation;
9. paired cold-start 12-workload gate and targeted repeats;
10. frozen vLLM comparison without redundant reruns.

Remaining priorities:

1. **Long-lived natural trace.** Run one server process long enough for each E
   head to cross the four-observation threshold; report opportunity, ready,
   selection-change, regret, and SLO-goodput rather than restarting the model
   every repeat.
2. **Bounded calibration coverage.** Improve only the existing safe probe
   cadence using uncertainty, protected slack, and already-observed completion
   events. Do not add E-size or workload-name rules.
3. **Formation-aware reward attribution.** Keep immediate pair compression as
   the learned target, but measure whether global successor formation vetoes a
   positive immediate E head and whether that veto reduces two-action regret.
4. **Determinism isolation.** Reproduce mixed, wave, and multi-image divergent
   rows with canonical row order and fixed overlap action. Separate FP16 tactic
   sensitivity from ownership/correctness failures.
5. **Equal-contract vLLM rerun only after Current changes.** The frozen anchor
   is sufficient for this E-head ablation. A new HTTP comparison is warranted
   after natural E policy selections or admission/runtime changes alter the
   serving contract.
6. **Hot-path cost.** Measure feature projection, three predictions, candidate
   selection, and RLS update p50/p95. Keep the total scheduler decision p95
   below the existing host-gap budget.

The next promotion criterion is not “E heads are implemented”; that is done.
It is:

```text
on a long-lived, workload-label-free production trace,
E+P/E+D become ready and change actions,
action fidelity remains exact,
and joint SLO-goodput improves without regressing the 12-trace paired gate.
```
