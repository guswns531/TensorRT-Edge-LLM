# M6 Hierarchical Pair/Direction Completion Model

## 1. Outcome

M6 completion prediction now retains all six legal ordered action identities without requiring six isolated cold
models to learn from scratch. The runtime learns one canonical completion posterior per unordered phase pair and one
ordered posterior per launch direction.

```text
P+D canonical posterior (P,D) ---+-- shrink --> P->D (incumbent,newcomer)
                                `-- shrink --> D->P (incumbent,newcomer)

E+P canonical posterior (E,P) ---+------------> E->P / P->E
E+D canonical posterior (E,D) ---+------------> E->D / D->E
```

Production action selection is unchanged. The hierarchy is still shadow-only and cannot legalize an action, change a
deadline decision, dispatch a context, or reclaim an ownership lease.

## 2. Why the hierarchy is needed

Six directions are legal across different requests and cohorts, but they are not equally likely to produce valid GPU
overlap in a natural trace. Keeping six fully independent heads made a cold reverse direction completely unknown even
when its phase pair had abundant measurements. Merging both directions into one model would instead erase the measured
asymmetry between adding a newcomer to P and adding P to an incumbent D.

The hierarchy separates these concerns:

- pair posterior: transferable interference structure for P+D, E+P, or E+D;
- direction posterior: launch-order and residual-progress correction;
- deterministic feasibility: dependency, TensorRT profile, ownership, and single-inflight invariants.

No workload name, exact shape key, external registry, TTL, or offline profile selects the blend.

## 3. Canonical phase semantics

Pair-common heads never mix component meaning. Their component order is always canonical:

| Pair | Primary | Secondary |
|---|---|---|
| P+D | P | D |
| E+P | E | P |
| E+D | E | D |

Direction heads use `(incumbent,newcomer)`. For a reverse action such as D->P, the runtime maps the pair prediction
from `(P,D)` to `(D,P)` before blending. The same mapping is applied to references, observed completions, and errors.

This prevents a common model from accidentally learning that a P completion and D completion are interchangeable.

## 4. Shrinkage rule

For each completion component, let the canonical pair posterior be `(mu_p, sigma_p)` and the ordered-direction
posterior be `(mu_d, sigma_d)`. With `n_d` accepted direction observations and a global pseudo-count `k=4`:

```text
w_d = n_d / (n_d + k)
mu  = (1 - w_d) * mu_p + w_d * mu_d

sigma^2 = (1 - w_d)^2 * sigma_p^2
        + w_d^2       * sigma_d^2
        + w_d(1-w_d)  * (mu_p - mu_d)^2
```

The disagreement term prevents a newly diverging direction from becoming overconfident. A cold direction has
`w_d=0` and immediately falls back to a ready pair posterior. As accepted direction evidence grows, its posterior
gradually takes authority. `k` is global rather than workload- or shape-specific.

## 5. Observation boundary

Both hierarchy levels consume the same common-H1 CUDA label:

```text
H1 = newcomer launch / residual augmentation boundary
completion_i = component CUDA done - H1
```

The observation is admitted only if both isolated references are finite and positive and both completion labels are
finite and non-negative. Wrapper calibration and both underlying RLS models use this exact boundary. This fixed an
initial telemetry bug where rejected zero-reference samples were counted only in the wrapper.

Residual P+D had a separate wiring omission. `PhaseThreeCoordinator::dispatchGlobalPrefillDecodeResidual()` built the
scalar contextual features but did not attach incumbent/newcomer completion references. The coordinator now supplies
the active phase's predicted remaining time and the newcomer phase time. In a fresh D->P run this changed valid D->P
completion observations from 0 to 29.

## 6. Code map

| Path | Responsibility |
|---|---|
| `cpp/runtime/phase/policy/phaseContextualPdModel.h` | hierarchical estimate evidence fields and blend contract |
| `cpp/runtime/scheduling/phaseContextualPdModel.cpp` | uncertainty-aware pair/direction shrinkage |
| `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h` | three pair models, six direction models, global pseudo-count |
| `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` | canonical mapping, paired observation, calibration boundary |
| `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | residual P+D completion-reference propagation |
| `examples/llm/llm_phase_context_smoke.cpp` | pair and hierarchical direction calibration JSON |
| `benchmarks/phase_serving/analyze_contextual_shadow.py` | schema 3 pair/direction summaries |
| `benchmarks/phase_serving/run_directional_injection_matrix.py` | shadow-only completion learning during controlled injection |

## 7. Validation

### 7.1 Build and unit tests

- TensorRT 26.06 build: pass
- contextual/runtime focused C++ tests: 14/14 pass
- Python analyzer/directional/oracle-matrix tests: 11/11 pass
- cold reverse direction inherits pair posterior: pass
- direction posterior gradually overrides pair posterior: pass
- pair disagreement increases uncertainty: pass
- invalid labels are rejected at the shared boundary: pass
- `git diff --check`: pass

### 7.2 Six-direction controlled run

One 50% requested-offset process was run for each of the six directions. The unified event chain observed every ordered
direction at least once:

| Actual event direction | Events |
|---|---:|
| P->D | 43 |
| D->P | 117 |
| E->P | 11 |
| P->E | 23 |
| E->D | 6 |
| D->E | 24 |

Across 3,936 dispatch/completion executions there were zero action-fidelity failures, zero dispatches without
completion, and zero completions without dispatch. Requested direction still does not guarantee profitable engine
interval overlap: at the selected 50% target only actual P->D produced accepted material-overlap buckets. This preserves
the M2 distinction between a legal/requested direction and its realized GPU interval.

After fixing label admissibility, valid completion observations were available for five directions in the six-process
matrix. D->P was then rerun after residual-reference wiring and produced 29 valid ready-capable observations. In that
fresh process:

| Direction | Observations | Ready | Incumbent MAE | Newcomer MAE |
|---|---:|---:|---:|---:|
| D->P | 29 | 28 | 0.249 ms | 0.999 ms |
| P->D | 16 | 13 | 2.920 ms | 1.074 ms |

The canonical P+D pair contained 45 accepted observations. Its nominal ready interval coverage was 95.1% for canonical
P and 97.6% for canonical D in this one process. These are calibration observations, not an active-policy speedup.

Artifacts:

```text
.local/inflight-m6-hierarchical-20260901/direction-o50-final/
.local/inflight-m6-hierarchical-20260901/decode-to-prefill-final/
```

### 7.3 Balanced real-request shadow run

The final Cosmos balanced trace retained the frozen exact token hash and used 9,237 MiB ready/peak GPU memory.

| Mode | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|
| same-binary model disabled | 4,153.35 | 66.78 / 165.66 | 13.32 / 15.09 | 1,203.00 / 1,893.84 |
| hierarchical M6 shadow | 4,208.98 | 65.21 / 166.15 | 13.13 / 14.47 | 1,185.99 / 1,830.07 |

The one-run delta versus disabled was +1.34% token/s, -2.35% TTFT mean, +0.30% TTFT p95, -1.42% TPOT mean,
-4.06% TPOT p95, -1.41% E2E mean, and -3.37% E2E p95. Since the shadow model has no policy authority, these values are
run variance/overhead evidence and are not claimed as model speedup.

Scheduler decision latency was 229.19 us mean and 415.21 us p95, versus the frozen disabled p95 of 405.52 us. The p95
increase is about 2.4%. There were zero action-fidelity failures across 738 dispatch/completion executions.

## 8. Gate decision

The hierarchy fixes sparse-direction cold start structurally, but Gate C remains **not passed**.

| Requirement | Status |
|---|---|
| pair-to-cold-direction transfer | pass in unit/runtime substrate |
| six legal/observable directions | pass in unified event matrix |
| valid completion labels for both P+D directions | pass after residual fix |
| exact action fidelity | pass |
| same-snapshot multi-action ranking regret | not evaluable |
| natural all-direction calibration with repeats | incomplete |
| false-safe under positive protected slack | not evaluable |
| active-policy 12-workload promotion | deliberately not started |

## 9. Next step

1. Extend the controlled replay artifact so multiple legal actions sharing an exact snapshot receive measured labels;
   compute top-1 agreement and normalized H1 ranking regret.
2. Repeat selected P+D/E+P/E+D directional cells at multiple realized start skews and estimate confidence intervals.
3. Calibrate only global or pair-family uncertainty, never workload-specific thresholds.
4. Repeat balanced, vision-heavy, and 39/48.8/97.5 saturation points five times to bound shadow overhead.
5. Promote completion prediction into the M4 projector only after Gate C passes; then run the active 12-workload gate.

Update: full candidate-frontier telemetry and same-snapshot common-work ranking are implemented and evaluated in
`notes/210-inflight-m6-same-snapshot-ranking-20260901.md`. The repeated balanced pilot reached 10/10 contextual top-1
agreement with zero normalized H1 regret, but promotion remains blocked on broader repeated alternatives, interval
calibration, false-safe evidence, and E+P/E+D coverage.

Frozen vLLM results remain reusable because model, precision, HTTP request contract, and output contract did not change.
