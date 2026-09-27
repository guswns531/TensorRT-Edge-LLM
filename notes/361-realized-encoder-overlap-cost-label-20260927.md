# Realized Encoder-Overlap Labels for Online Phase Costs

## Result

The external-encoder background cost key now uses the observed CUDA-interval intersection for P/D cost observations when
the phase activity observer is attached. A planned E-active snapshot still describes the candidate at selection time;
it is no longer treated as proof that E physically overlapped the later P/D execution. If an E interval is still open and
its intersection cannot yet be decided, that action's cost/RLS observation is deferred instead of being assigned a
possibly wrong label.

After this change, a fresh full24 campaign plus two repeats completed 72/72 cells. All 24 three-run throughput medians
and all 72 individual runs beat the frozen vLLM anchor. Against the historical 9db3 Current median, every workload is
within -0.3% or better; this is a large improvement over the previous realized-key iteration's Gemma long-prefill
regression. The serving default remains the recorded 9db3 binary because output-quality approval is still unresolved,
and the exact interval observer is currently attached through the activity-recorder path rather than proven as an
always-on production mechanism.

## What the observer measures

The activity recorder places the E, P, and D phase-event intervals in one CUDA context and one event epoch. For each
completed P/D dispatch, `PhaseDispatchWorker` checks whether an encoder interval intersects the P interval, the D
interval, or both. The cost-model observation key uses that realized overlap bit. The planned E-active bit remains
available for candidate generation and is emitted separately for comparison.

This is stream-work interval overlap, not proof that the kernels were simultaneously resident on SMs. The next hardware
characterization can correlate these labels with Nsight kernel intervals and SM activity, but the serving cost label
now reflects actual E/P/D stream intervals rather than an asynchronously maintained Boolean snapshot.

In the measurement epoch, the full24 screen recorded 27,505 dispatches: all 27,505 had a resolved physical-overlap
label and none were deferred. E overlap was observed on 346 dispatches while the action-time planned E-active bit was
true on only 20; the difference comes from E work beginning or remaining in flight after the P/D candidate snapshot.
Most differences were in mixed/vision traces. Text-only balanced and long-prefill traces recorded no physical E
overlap in the measurement epoch.

## Full24 final measurements

Throughput values are 3-run medians. `TTFT`, `TPOT`, and `E2E` columns are mean/p95 percentage deltas against frozen
vLLM; negative latency deltas favor Current. `9db3` is the historical Current throughput median. Positive latency
deltas mean Current is slower on that metric.

| Model / workload | tok/s | Δ vs vLLM / 9db3 | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 |
|---|---:|---:|---:|---:|---:|
| Cosmos / balanced | 4472.9 | +3.6 / +0.6% | -46.6 / -41.7% | +2.0 / +1.5% | -3.4 / -1.8% |
| Cosmos / bimodal | 2001.6 | +6.9 / +2.7% | +19.4 / +45.4% | -22.8 / -14.1% | -9.9 / -3.6% |
| Cosmos / decode-heavy | 5245.8 | +6.2 / +0.1% | -48.6 / -48.5% | -3.3 / -3.3% | -5.4 / -4.4% |
| Cosmos / late-vision | 2538.7 | +17.3 / +0.7% | -49.7 / -48.7% | -13.9 / -13.3% | -18.9 / -14.7% |
| Cosmos / long-prefill | 1312.6 | +16.8 / +0.1% | +2.5 / -16.2% | -28.1 / -28.4% | -15.5 / -17.2% |
| Cosmos / mixed | 1174.5 | +27.2 / +3.1% | -13.3 / -26.8% | -27.3 / -27.9% | -22.4 / -20.9% |
| Cosmos / multi-image | 313.3 | +28.5 / +0.6% | +3.9 / -24.3% | -37.7 / -42.9% | -20.7 / -22.0% |
| Cosmos / poisson | 2044.3 | +14.8 / +1.0% | -41.6 / -14.8% | -10.2 / -11.5% | -17.1 / -14.3% |
| Cosmos / short | 2453.7 | +19.9 / +1.9% | -51.9 / -26.9% | +0.1 / -3.8% | -23.1 / -19.1% |
| Cosmos / text-heavy | 2016.0 | +56.0 / +1.3% | -52.7 / -48.7% | +15.4 / -16.7% | -14.2 / -34.2% |
| Cosmos / vision-heavy | 733.2 | +27.0 / +2.8% | -12.9 / -14.2% | -36.9 / -35.4% | -25.7 / -21.6% |
| Cosmos / wave-drain | 98.0 | +2.2 / -0.0% | -0.6 / -28.0% | -34.6 / -34.9% | -21.1 / -21.7% |
| Gemma / balanced | 1268.0 | +64.4 / +0.4% | -34.5 / -13.7% | -35.2 / -30.2% | -35.7 / -35.1% |
| Gemma / bimodal | 844.9 | +40.8 / -0.3% | +23.2 / +13.3% | -18.6 / +12.5% | -24.7 / -29.2% |
| Gemma / decode-heavy | 1381.5 | +70.0 / +0.3% | -41.7 / -14.7% | -37.7 / -36.8% | -38.1 / -37.8% |
| Gemma / late-vision | 1527.2 | +54.1 / -0.3% | -0.5 / +54.6% | -39.9 / -39.6% | -38.3 / -39.9% |
| Gemma / long-prefill | 629.4 | +25.8 / +1.6% | +26.0 / +13.6% | -21.9 / -22.2% | -14.1 / -12.4% |
| Gemma / mixed | 762.1 | +8.3 / +2.4% | -10.1 / +17.8% | -6.0 / +4.5% | -6.1 / -1.3% |
| Gemma / multi-image | 413.2 | +8.4 / +6.7% | +72.6 / +132.3% | -11.3 / +18.1% | +3.0 / +17.1% |
| Gemma / poisson | 935.7 | +37.2 / +0.7% | +4.4 / +116.4% | -21.6 / -17.2% | -20.8 / -19.1% |
| Gemma / short | 853.9 | +50.5 / -0.0% | -38.7 / -12.1% | -21.1 / -1.5% | -27.5 / -23.8% |
| Gemma / text-heavy | 903.5 | +123.3 / +1.1% | -89.4 / -90.3% | -6.5 / -5.0% | -54.9 / -72.4% |
| Gemma / vision-heavy | 576.6 | +3.0 / +0.9% | +38.5 / +102.4% | -5.3 / +10.1% | +3.8 / -7.2% |
| Gemma / wave-drain | 97.1 | +4.8 / -0.2% | +43.2 / +53.5% | -58.5 / -56.4% | -37.8 / -36.6% |

Latency wins against vLLM: TTFT mean 15/24, TTFT p95 15/24, TPOT mean 21/24, TPOT p95 19/24, E2E mean 22/24, and
E2E p95 23/24. Throughput alone is universally ahead in this fixed-output workload suite; the claim is not universal
latency or production-quality superiority. Across the 24 traces, 14 had identical output-token hashes over three runs
and 10 differed; semantic output equivalence was not approved.

## Reproducibility and validation

- Screen: `.local/results/observed-encoder-interval-full12-screen-20260927`
- Additional repeats: `.local/results/observed-encoder-interval-full12-additional-20260927`
- Both manifests use the same source snapshot, runtime binary, engines, generic calibration, traces, and frozen vLLM
  references; each has 24/24 and 48/48 completed cells respectively, no failed cells.
- C++ `PhaseActivityTimelineTest.*`, `PhaseGlobalCostModelTest.*`, and `PhaseQueueSchedulerTest.*`: 189/189 passed.
- The activity observer is attached only when a `PhaseActivityTimelineRecorder` is configured. Without one, metrics
  validity remains false and the runtime falls back to the planned E-active state. A lightweight always-on physical
  interval tracker and its overhead are not yet implemented or validated.
- No model/GPU-specific action rule or new SLO was introduced. Frozen vLLM data was reused; its contract was unchanged.

## Next steps

1. Profile the observer-enabled path against an uninstrumented paired run to bound host/event overhead.
2. Design a lightweight always-on E interval tracker using the existing preallocated E and P/D CUDA event lifecycle;
   avoid recording/exporting full activity CSVs in production.
3. Add end-to-end tests for event-complete-before-dispatch, E-start-during-P/D, E-finish-during-P/D, and pending E
   intervals that must be deferred.
4. After the always-on tracker is validated, rerun full24 three times and retain the frozen vLLM baseline comparison.
