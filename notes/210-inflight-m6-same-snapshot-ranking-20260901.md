# M6 Same-Snapshot Contextual Ranking Gate

## 1. Outcome

M6 now exports the complete policy-neutral action frontier and joins measured alternatives only when their immutable
execution snapshot is exactly identical. The resulting artifact can evaluate contextual top-1 action ranking and
normalized H1 regret without a workload label, an approximate state join, or an unobserved counterfactual.

```text
immutable E/P/D + ownership snapshot
              |
              +--> full legal candidate frontier + pre-update M6 posterior
              |
              +--> selected action --> CUDA component completion vector
              |
              `--> exact cross-run join --> empirical common-work milestone
```

The three-run-per-policy Cosmos balanced pilot produced:

| Item | Result |
|---|---:|
| Unified decisions | 4,548 |
| Dispatch/completion executions | 4,718 |
| GPU intervals | 4,718 |
| Action-fidelity failures | 0 |
| Exact multi-action snapshots | 78 |
| H1-comparable snapshots | 78 |
| Contextual ranking-evaluable frontiers | 10 |
| Contextual top-1 agreement | **10 / 10** |
| Mean/max normalized H1 regret | **0 / 0** |
| Fully repeated multi-action snapshots | 1 / 78 |

This makes the ranking gate evaluable for the first time. It is still a pilot, not a production-promotion pass: most
multi-action states contain different D batch materializations rather than repeated measurements of every alternative,
and E+P/E+D natural coverage and false-safe evidence remain insufficient.

## 2. Full candidate-frontier telemetry

Previously, a unified decision event retained only the selected P/D candidate. Cross-run CUDA labels could show that
another action was better, but the model prediction for that alternative had already been discarded. The following
path now retains the full frontier while leaving production selection unchanged:

| Path | Responsibility |
|---|---|
| `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}` | retain the full P/D/P+D preview frontier |
| `cpp/runtime/scheduling/independentPhaseAsyncServer.{h,cpp}` | expose the last preview without policy authority |
| `cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}` | merge P/D candidates with E/E+P/E+D candidates by stable action ID |
| `cpp/runtime/phase/mechanism/phaseUnifiedEvent.h` | attach pre-update contextual completion posterior and references |
| `examples/llm/llm_phase_context_smoke.cpp` | serialize the evidence into unified decision JSON |
| `benchmarks/phase_serving/manifests/phase_event_schema_v1.json` | document optional M6 candidate evidence |

The model history is deliberately not part of the execution snapshot signature. The signature answers whether two
actions saw the same executable work and ownership state; each prediction frontier separately preserves the policy and
its process-local posterior history.

## 3. Empirical objective correction

The original analyzer compared actions only when the phase that completed first was identical. This was safe but too
restrictive for a serial-vs-pair action:

```text
serial P       : P completes at 12.4 ms
overlap P + D  : D completes at 13.7 ms, P completes at 15.5 ms
```

Comparing the first completion would compare P progress with D progress. Dropping the sample would instead discard the
only directly comparable fact: both actions advance the same P work. The analyzer now uses this rule:

1. If alternatives materialize the same component frontier and have a stable first phase, compare first completion.
2. If component frontiers differ, require exactly one common phase and compare that phase's CUDA completion.
3. If there is no common phase or more than one ambiguous common milestone, reject the snapshot.

This is not a relaxed snapshot match. Request IDs, token counts, context lengths, in-flight state, and ownership counts
must still hash to the same exact signature. Only the scalar objective is projected onto equal work.

The coverage builder therefore retains every selected action's per-phase component completion vector. The analyzer
also emits per-frontier action predictions, empirical labels, top-1 identity, and normalized regret for causal audit.

## 4. Controlled-injection negative result

A P-to-D 75% injection produced a real 75.76% interval overlap and exact output identity. Compared with the serial
control, it reduced throughput and worsened decode-tail latency:

| Mode | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|---:|
| serial | 1,097.97 | 26.45 / 32.40 | 7.90 / 9.94 | 504.83 / 1,343.95 |
| 75.76% P->D overlap | 1,015.20 | 24.24 / 34.45 | 13.69 / 21.49 | 544.14 / 1,358.80 |

Both generated the same 6,656 tokens and exact token hash. However, independent executions did not reach an identical
ready-request snapshot: even with the same requested injection point, early GPU timing changed downstream ready IDs.
The exact join correctly produced zero multi-action snapshots. This confirms that approximate cross-run state matching
must not be used as a counterfactual label source.

## 5. Repeated balanced results

All six measured runs retained the frozen token hash
`f51d448d5824038f5237cdee55dc50799de5c5cb5a228fcf1166ff1ed30875ca` and used 9,237 MiB.

| Policy | Runs | token/s median | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|---:|
| unchanged myopic + M6 shadow | 3 | 4,166.33 | 67.68 / 166.28 | 13.27 / 14.70 | 1,197.12 / 1,855.36 |
| bounded safe-probe + M6 shadow | 3 | 3,910.96 | 59.12 / 182.17 | 14.31 / 15.99 | 1,279.51 / 2,022.09 |

Safe-probe improved TTFT mean by 12.65%, but reduced throughput by 6.13% and regressed TTFT p95, TPOT, and E2E by
roughly 7.8--9.6%. It remains a bounded label collector and must not become the serving default.

The exact common-P snapshot with repeated alternatives was stable:

| Action | Samples | median P completion |
|---|---:|---:|
| serial P | 2 | 12.432 ms |
| P+D | 3 | 15.468 ms |

All five contextual frontiers covering this snapshot ranked serial P first. Across all ranking-evaluable frontiers the
final common-work definition achieved 10/10 top-1 agreement and zero measured regret.

## 6. Code and validation

New or extended analysis paths:

- `benchmarks/phase_serving/build_oracle_h1_snapshot_coverage.py`: schema 2 component vectors and full prediction
  frontiers;
- `benchmarks/phase_serving/analyze_oracle_h1_snapshot_coverage.py`: common-work milestone, contextual top-1, normalized
  regret, and per-decision audit records;
- `benchmarks/phase_serving/run_directional_injection_matrix.py`: independent overlap-percentage control;
- `tests/python-unittests/test_oracle_h1_snapshot_coverage.py`;
- `tests/python-unittests/test_oracle_h1_snapshot_analysis.py`.

Validation completed:

- focused Python coverage/ranking/shadow tests: 10/10 pass;
- previous full-frontier C++ focused tests: 11/11 pass;
- unified event validation: 4,718 executions, zero missing dispatch/completion and zero fidelity failures;
- all balanced and injection comparisons retained exact token identity.

Artifacts:

```text
.local/inflight-m6-gate-c-ranking-20260901/
  gate-c-balanced/coverage-repeat3.json
  gate-c-balanced/analysis-repeat3-common-work.json
  m6-shadow-frontier*/
  m6-shadow-probe-frontier*/
  pd-o75-serial/
  pd-o75-overlap/
```

## 7. Gate C status and next step

| Gate C requirement | Status |
|---|---|
| exact action fidelity | pass |
| continuous interpolation without exact policy key | implemented |
| same-snapshot ranking evaluable | pass |
| contextual pilot top-1/regret | 10/10, zero regret |
| repeated P vs P+D alternative | pass at one snapshot |
| promotion-quality alternative coverage | incomplete |
| nominal completion interval calibration | partial |
| low SLO false-safe rate | not yet evaluable across all families |
| E+P/E+D multi-skew repeated coverage | incomplete |
| active 12-workload promotion | deliberately not started |

The next step is still M6 calibration rather than workload-specific tuning:

1. repeat selected P+D, E+P, and E+D cells at multiple realized start skews;
2. calibrate one global or pair-family uncertainty scale from pre-update residual coverage;
3. report predicted-safe/false-safe against protected slack by pair family;
4. rerun balanced, vision-heavy, and 39/48.8/97.5 saturation points with the calibrated shadow;
5. only then enable active ranking and execute the complete 12-workload promotion gate.

vLLM was not rerun because the model, quantization, engine-facing request trace, output contract, and production policy
did not change. The frozen vLLM result remains the comparison anchor for the later active-policy gate.

Update: the repeated E/P and E/D calibration matrix is complete in
`notes/211-inflight-m6-six-direction-calibration-20260901.md`. All six directions now have sufficient observations and
zero observed false-safe events, but three completion components remain under the nominal 95% interval target. Active
promotion therefore remains blocked on uncertainty calibration and broader repeated alternatives.
