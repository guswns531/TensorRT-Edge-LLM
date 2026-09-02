# M7 natural-trace promotion gate

## Decision

M7 completion-vector authority remains **opt-in and not promoted**. Natural
E/P calibration, conformal safety, action fidelity, and scheduler overhead all
improved enough to exercise the active path, but the required saturation gate
did not pass. In particular, the five-run 48.8 req/s result reached only
34.943 raw req/s and 17.968 joint-SLO goodput req/s. The gate required at least
40 raw req/s and 99% joint-SLO pass. Consequently the conditional active-M7
12-workload rerun was not executed; the existing shadow 12x3 result remains the
current full-suite reference.

This is a policy promotion result, not a correctness failure. Every fresh run
retained exact generated-token identity, and uncalibrated pair families always
fell back to the existing M6 completion bounds.

## What changed

### Natural E staging is now scheduler-visible

Asynchronous vision preparation is a mechanism stage rather than an implicit
encoder dispatch. `PhaseThreeCoordinator` retains a prepared encoder cohort and
presents it to the same global candidate frontier as P and D. The global
selector can therefore observe natural E, E+P, and E+D opportunities before GPU
submission. A prepared host-side cohort is not counted as an outstanding E
context; E becomes outstanding only after the vision context is submitted.
This preserves the execution-lease invariant:

```text
host preprocess -> prepared E cohort -> global candidate/lease -> GPU E submit
                                      \-> E+P or E+D candidate
```

This fixed the earlier mismatch in which host preparation could make a P+D
lease appear to have an unexpected E context.

### Bounded natural calibration

Warmup calibration chooses the least-observed contextual pair direction rather
than competing only by sparse exact CUDA keys. It still uses only the current
ready frontier, protected slack, and already-outstanding completions. It does
not use a workload name, E-size rule, wall-clock TTL, persisted registry, or
future-arrival predictor.

### Separate calibration and authority

`PhaseContextualCompletionCalibrationConfig` now has two independent controls:

- `enabled`: collect and apply conformal uncertainty in shadow prediction;
- `active`: permit a ready and calibrated completion vector to replace the
  legacy protected-completion bound used by selection.

The example runtime exposes the second control through
`TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE=1`, and the benchmark runner exposes
`--completion-conformal-active`. Both default to off. Authority is also gated
per candidate by `ready && uncertaintyCalibrated`; therefore E/D, which did not
obtain enough natural observations, automatically retained its M6 bound.

### Scheduler hot-path optimization

The three-phase coordinator previously rebuilt the P/D frontier three times:

```text
preview P/D + preview P-only + preview D-only
```

It now builds the immutable P/D frontier once and reuses its serial P and D
candidates when constructing E+P and E+D. This does not change rows, costs, or
selection semantics.

The primary balanced scheduler p95 fell from 424.548 us to 151.208 us, a 64.4%
reduction. Five-run load medians were 167.187 us at 39, 182.069 us at 48.8, and
182.614 us at 97.5 req/s. This is much lower than the previous metrics-heavy
539.73 us result, though still above the controlled-matrix 93.41 us reference.

## Validation

- TensorRT 26.06/CUDA build: pass.
- Focused C++ tests: 72/72 pass.
- Python runner syntax: pass.
- `git diff --check`: pass.
- natural action-fidelity violations: 0.
- exact generated-token identity: pass in every natural and load repeat.
- vision semantic request completion: pass.

## Natural shadow calibration

The natural traces used the production OpenAI-compatible HTTP adapter, four
adapter workers, P8/D64, fixed P chunk 128, stable slots 80, asynchronous vision
preparation, E batch cap 8, and independent TensorRT contexts. Per-dispatch JSON
was disabled for primary latency/throughput measurements and enabled only for
the held-out calibration analysis.

With 512 excluded vision-heavy warmup requests, natural calibration observed:

| Direction | Observations | Result |
|---|---:|---|
| E -> P | 36 | ready |
| P -> E | 4 | pair-shared support only |
| E -> D | 4 | insufficient |
| D -> E | 0 | insufficient |
| P -> D | 37 | ready |
| D -> P | 0 | insufficient |

E/P pair calibration accumulated 40 observations and its conformal scale
reached the configured upper bound of 8.0. The chronological held-out E -> P
interval coverage was 95% for the incumbent and 100% for the newcomer over 20
post-calibration observations. Conformal false-safe count was zero. P -> D also
had zero conformal false-safe observations. E/D remained unready and was not
given authority.

The scale of 8.0 is an important warning: natural E/P completion variance is
substantially larger than the raw RLS posterior predicts. The calibrated bound
is safe in this trace, but can be conservative.

## Focused active A/B

The following single-process checks compare the new active authority with the
matching natural shadow run. They are a functional promotion pilot, not the
final repeated performance claim.

| Workload | Metric | Shadow | Active M7 | Delta |
|---|---|---:|---:|---:|
| balanced | token/s | 4452.43 | 4540.37 | +1.97% |
| balanced | TTFT mean / p95 ms | 69.32 / 166.67 | 67.52 / 163.94 | -2.59% / -1.64% |
| balanced | TPOT mean / p95 ms | 12.38 / 13.95 | 12.12 / 13.48 | -2.04% / -3.38% |
| balanced | E2E mean / p95 ms | 1125.13 / 1736.44 | 1101.49 / 1713.37 | -2.10% / -1.33% |
| vision-heavy | token/s | 655.96 | 701.62 | +6.96% |
| vision-heavy | TTFT mean / p95 ms | 1484.42 / 3221.84 | 1468.22 / 3097.61 | -1.09% / -3.86% |
| vision-heavy | TPOT mean / p95 ms | 51.46 / 85.00 | 42.38 / 71.17 | -17.64% / -16.27% |
| vision-heavy | E2E mean / p95 ms | 3453.03 / 3684.08 | 3108.26 / 3458.38 | -9.98% / -6.13% |

These two runs justified exercising the load gate, but they are too small to
override the repeated saturation result below.

## Active M7 request-load gate

Each point used five fresh processes. Latency cells are the median across runs
of `mean / p95` in milliseconds. Joint SLO is TTFT <= 500 ms, TPOT <= 50 ms,
and arrival-based E2E <= 2500 ms.

| Offered req/s | Raw req/s | Token/s | SLO pass | SLO goodput req/s | TTFT | TPOT | E2E |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 39.0 | 33.633 | 2914.89 | 98.96% | 33.230 | 46.79 / 187.78 | 15.19 / 19.67 | 1338.46 / 2359.16 |
| 48.8 | 34.943 | 3028.39 | 51.74% | 17.968 | 242.46 / 490.46 | 17.63 / 20.33 | 1743.34 / 2842.97 |
| 97.5 | 37.975 | 3291.13 | 27.43% | 10.417 | 331.01 / 515.86 | 17.28 / 19.84 | 1797.57 / 2851.06 |

Against the frozen vLLM anchors:

| Offered req/s | Active M7 raw delta | Active M7 goodput delta |
|---:|---:|---:|
| 39.0 | -2.10% | -3.28% |
| 48.8 | -14.58% | -56.08% |

The 48.8 comparison reuses frozen vLLM raw/goodput 40.908/40.908 req/s. The
model, trace, output, memory, and SLO contracts did not change, so vLLM was not
rerun.

## Authority-only control

To isolate M7 authority from the rest of the current runtime, 48.8 req/s was
repeated five times with the identical binary and warmup but with authority
disabled. Shadow produced:

- raw throughput: 34.862 req/s;
- joint-SLO pass: 52.08%;
- goodput: 17.977 req/s.

Active M7 produced 34.943 req/s, 51.74%, and 17.968 req/s. The differences are
negligible. Therefore the saturation failure is not caused primarily by the
new conformal completion authority. It is a current runtime/policy substrate
regression already present with authority disabled. Conversely, the focused
vision improvement cannot yet be claimed as a robust M7 gain.

## Why the 12-workload active gate did not run

The agreed sequence was conditional: activate M7 and run the 12-workload gate
only after natural calibration, false-safe, scheduler-overhead, and saturation
gates pass. The 48.8 point failed both raw-throughput and joint-SLO thresholds,
and the authority-only control reproduced the failure. Running 12x3 at this
point would spend GPU time without producing a promotable configuration.

The latest completed full-suite reference therefore remains
`.local/inflight-m7-current-12x3-20260902/`, documented in Note 213. It used M7
shadow only and retained exact identity across all repeats.

## Next implementation order

1. Keep completion authority off by default. Preserve the opt-in path for
   controlled and focused experiments.
2. Decompose the 48.8 knee using the already-added decode-cycle timestamps:
   GPU completion visibility, sampling, state commit, candidate formation,
   selector, TRT submit, and launch gap.
3. Compare the current binary with the P55 40.472 req/s anchor under the same
   builder lifecycle and warmup contract. The authority-only A/B shows that
   tuning conformal scale is not the first fix.
4. Restore at least 40 raw req/s and 99% joint-SLO pass at 48.8 without losing
   the 97.5 overload-tail behavior.
5. Re-run active/shadow 3x5. Only if that passes, run active M7 over the same
   12 workloads for three fresh processes each and compare against P52, the
   frozen vLLM suite, and the shadow configuration.

## Artifacts

- low-overhead natural shadow:
  `.local/m7-natural-low-overhead-20260902/`
- E/P 512-request detailed calibration:
  `.local/m7-natural-ep512-detailed-20260902/`
- focused active pilot:
  `.local/m7-active-focused-20260902/`
- active 39/48.8/97.5 five-run gate:
  `.local/m7-active-load-3x5-20260902/`
- 48.8 authority-off control:
  `.local/m7-shadow-preview-reuse-load48-5x-20260902/`
- existing shadow 12x3 reference:
  `.local/inflight-m7-current-12x3-20260902/`
