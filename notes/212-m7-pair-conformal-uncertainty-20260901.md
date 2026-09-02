# M7 Pair-Family Conformal Completion Uncertainty

## 1. Outcome

M6's completion means remain the same hierarchical pair/direction RLS model. M7 adds one bounded online conformal
uncertainty scale per unordered phase pair:

```text
P-D scale: P->D and D->P residuals
E-P scale: E->P and P->E residuals
E-D scale: E->D and D->E residuals
```

There is no direction-, batch-shape-, trace-, request-class-, or workload-specific threshold. No wall-clock TTL,
external registry, persisted profile, or offline model is used. The implementation is disabled by default and was
evaluated in shadow mode.

The controlled real-request pilot made the calibrated interval evaluable for all six ordered directions. Across 380
chronological held-out completion vectors, every component reached at least 98.7% coverage against a 95% target. All
runs preserved exact token identity and observed zero false-safe decisions. This passes the narrow uncertainty pilot,
but it does not yet promote M7 to active scheduling: E/P has only seven held-out vectors, and broad production traces
still need shadow validation.

## 2. Online algorithm

For a raw hierarchical prediction with incumbent/newcomer uncertainties `sigma_i`, confidence multiplier `beta`, and
observed completions `y_i`, the label is one joint nonconformity score:

```text
score_t = max_i |y_i - mean_i| / (beta * sigma_i)
```

Using the maximum gives one scale for the entire two-component completion vector rather than independently tuning two
intervals. For pair family `f`, the next prediction uses the finite-sample conformal quantile of prior scores only:

```text
rank_f = ceil((n_f + 1) * target_coverage)
scale_f = clamp(quantile(score_1 ... score_n, rank_f), 1, 8)
calibrated_sigma_i = scale_f * raw_sigma_i
```

The current label is inserted only after coverage and false-safe telemetry has been evaluated. Consequently every
reported conformal observation is chronological held-out evidence. Scores are accepted only after the underlying RLS
completion model is ready. A bounded 128-observation window replaces old evidence by observation count, not time.

The real-request pilot used one common configuration for all three pair families:

```text
target coverage       0.95
minimum observations  8
score window           128
minimum scale          1.0
maximum scale          8.0
```

The checked-in default minimum remains 16; the shorter value was used to obtain a held-out suffix inside the bounded
controlled traces. The next broad traces must determine whether 16 is practical without persisted calibration.

## 3. Implementation

| Path | Responsibility |
|---|---|
| `cpp/runtime/phase/policy/phaseContextualPdModel.h` | calibration config, estimate, pair-vector telemetry, bounded calibrator |
| `cpp/runtime/scheduling/phaseContextualPdModel.cpp` | finite-sample quantile, scale application, post-prediction score insertion |
| `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h` | three independent pair-family calibrators |
| `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` | bidirectional pair sharing, raw-vs-calibrated boundary, reset |
| `examples/llm/llm_phase_context_smoke.cpp` | environment configuration and scale/held-out JSON telemetry |
| `benchmarks/phase_serving/run_directional_injection_matrix.py` | common conformal CLI controls and manifest contract |
| `benchmarks/phase_serving/analyze_contextual_shadow.py` | chronological held-out coverage and per-run scale summary |
| `benchmarks/phase_serving/manifests/phase_event_schema_v1.json` | candidate scale/readiness fields |
| `unittests/phaseRuntimeCostTrackerTest.cpp` | pre-update ordering and opposite-direction sharing tests |

Runtime controls:

```text
TRT_EDGELLM_COMPLETION_CONFORMAL=0|1
TRT_EDGELLM_COMPLETION_CONFORMAL_MIN_OBSERVATIONS=N
TRT_EDGELLM_COMPLETION_CONFORMAL_WINDOW=N
TRT_EDGELLM_COMPLETION_CONFORMAL_TARGET=p
```

## 4. Real-request held-out result

The matrix used 25%, 50%, and 75% requested offsets with two independent process repeats per cell. Direction identity
continued to follow measured execution order, not the requested experiment label.

| Ordered direction | Held-out vectors | Incumbent coverage | Newcomer coverage |
|---|---:|---:|---:|
| P -> D | 69 | 100.0% | 100.0% |
| D -> P | 212 | 100.0% | 100.0% |
| E -> P | 2 | 100.0% | 100.0% |
| P -> E | 5 | 100.0% | 100.0% |
| E -> D | 13 | 100.0% | 100.0% |
| D -> E | 79 | 98.7% | 100.0% |

Pair-family terminal scale distribution across independent runs:

| Pair | Ready runs | Median scale | Maximum scale |
|---|---:|---:|---:|
| P-D | 6 | 1.387 | 6.407 |
| E-P | 10 | 1.000 | 1.000 |
| E-D | 6 | 3.717 | 4.704 |

The different scales are learned continuous residual statistics, not configured pair rules. E-D required consistently
wider intervals. P-D was usually close to one but contained two difficult process-local prefixes. E-P did not need
widening in the fresh pilot, even though the earlier M6 matrix exposed under-coverage; more E-P held-out samples are
therefore required before active use.

Raw versus calibrated examples:

| Direction/component | Raw ready coverage | Calibrated held-out coverage |
|---|---:|---:|
| D -> E incumbent | 89.4% | 98.7% |
| E -> D newcomer | 94.1% | 100.0% |
| P -> D incumbent/newcomer | 93.1% / 93.1% | 100.0% / 100.0% |

All relevant process-level false-safe counters were zero before calibration. Since M7 only widens uncertainty, it did
not create a new false-safe event. Dedicated conformal-only false-safe counters are now emitted for subsequent fresh
runs.

## 5. Correctness, overhead, and performance scope

Exact generated-token hashes remained stable in every repeat:

- E/P: 56 tokens per run,
  `1ce07cd3506a9139e5ec26798b256568b541d071b5227de6ec8cb9c9d577f1a5`;
- E/D: 6,176 tokens per run,
  `2c9e7624bed45a92567232b6802fd8da60e1da0595014a70cc6d2947042f28d5`;
- P/D: 6,656 tokens per run,
  `6c93d3585b084cec8eb31179d6e08d3475c86c96c62ed1a0c5f0e8d5843026b1`.

Peak memory stayed at 9,237--9,250 MiB. Scheduler decision p95 over the combined 12,098 decisions was 93.41 us. By
matrix it was 50.08 us for E/P, 97.10 us for E/D, and 64.17 us for P/D. Quantile evaluation is not the dominant host
cost; candidate formation remains larger in the production balanced trace.

This stage is shadow-only. Throughput differences across requested-offset cells describe the injected action and are
not a conformal speedup claim. vLLM was not rerun because the model, trace contract, serving policy, and output
contract did not change; frozen vLLM anchors remain reserved for the later active-policy comparison.

## 6. Validation

- TensorRT 26.06/CUDA build: pass;
- contextual/runtime focused C++ tests: 15/15 pass;
- Gate-C Python analyzer/replay tests: 18/18 pass;
- six-direction conformal held-out analysis: evaluable;
- exact output identity: pass in 18/18 real-request repeats;
- material P->D overlap: two accepted offset buckets;
- `git diff --check`, Python bytecode, and JSON schema parse: pass.

Artifacts:

```text
.local/inflight-m7-conformal-20260901/
  contextual-summary-combined.json
  prefill-to-decode/contextual-summary.json
  encoder-to-prefill-heldout/contextual-summary.json
  encoder-to-decode/contextual-summary.json
```

## 7. Gate and next step

| Requirement | Status |
|---|---|
| pair-family calibration without workload labels | pass |
| chronological pre-update evaluation | pass |
| all six directions held out | pass |
| minimum held-out component coverage >= 95% | pass in pilot |
| exact output identity | pass |
| observed false-safe | 0 |
| enough E/P held-out samples | **fail: 7 total** |
| broad balanced/vision/saturation shadow validation | pending |
| active M7 / 12-workload promotion | blocked |

Next, keep the calibrated completion model in shadow mode and run balanced, vision-heavy, and offered-load
39/48.8/97.5 traces. Those longer natural traces must answer three questions before active scheduling:

1. Does default minimum 16 become ready without persisted state?
2. Does the calibrated coverage remain near 95% without scales saturating at the maximum?
3. Are conformal-only false-safe counts still zero while scheduler p95 remains acceptable?

Only after that gate should calibrated completion readiness become a requirement for active H1 selection and the full
12-workload promotion suite begin.
