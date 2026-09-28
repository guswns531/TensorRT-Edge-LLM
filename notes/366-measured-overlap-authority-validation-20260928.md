# Measured Overlap Cost Outranks the Contextual P+D Head

## Outcome

`c7a1b71` lets a directly measured overlap price (`runtime_exact` or `runtime_residual`) take precedence over the
contextual P+D head's lower-confidence-bound reprice. The head keeps authority only for shapes with no measured
overlap cost. Combined with `19d8aa8` (note 365), this removes the Gemma long-prefill bimodality: 8/8 A/B runs in the
high mode (614-630 tok/s), and 612-620 across the full24 x3 revalidation. Full24 x3 shows no regression elsewhere:
72/72 cells, every run above frozen vLLM, throughput geomean +1.11% versus 9db3 and +0.02% versus `1fab188`.

Neither commit is promoted to the serving default here; that decision still depends on the output-quality gate
(note 364), which this change does not touch.

## Change

At the three P+D pricing sites (complete candidate and residual augmentation in `PhaseQueueScheduler`, residual
augmentation in `PhaseThreeCoordinator`), `contextualScalarAuthorityApplied` is now gated on the overlap cost being
unmeasured. The encoder-pair head (E+P, E+D) is unchanged. Unit test
`PhaseQueueSchedulerTest.MeasuredOverlapCostOutranksContextualAuthority` covers the control (pessimistic ready head
overrides an unmeasured shape) and the fix (residual samples of the shape make the measured price authoritative).
`unitTestRuntime` 728 passed, 2 optional skipped.

Why this works when `19d8aa8` alone did not: the transfer priced 45-70 complete P+D candidates per run correctly,
but the head, whose learned mean reward on this trace is slightly negative on every binary including 9db3, repriced
them at about `reference * 1.08` and prefill-only won. Removing the reprice for measured shapes lets the transferred
price reach the selector.

## Gemma long-prefill A/B (`gemma-long-prefill-authority-ab-20260928`, 8 rotated blocks)

| Binary | Median tok/s | Range | Runs >= 610 | P+D dispatches (median) |
|---|---:|---|---:|---:|
| `c7a1b71` | 615.5 | 614-630 | 8/8 | 92 |
| 9db3 | 618.2 | 604-628 | 6/8 | 114 |
| `19d8aa8` | 611.5 | 588-635 | 4/8 | 74 |

Median throughput equals 9db3 within noise; variance collapses from about 16 to about 5 tok/s.

## Full24 x3 (`measured-authority-full24-3x-20260928`)

Median over three runs; percentages against frozen vLLM, historical 9db3, and `1fab188` (note 364).

| Workload | tok/s | vs vLLM | vs 9db3 | vs 1fab188 | Range |
|---|---:|---:|---:|---:|---|
| Cosmos balanced | 4520.6 | +4.7% | +1.7% | -0.9% | 4501-4526 |
| Cosmos bimodal | 1974.4 | +5.4% | +1.3% | -2.0% | 1970-2017 |
| Cosmos decode-heavy | 5218.2 | +5.7% | -0.5% | -1.3% | 5208-5247 |
| Cosmos late-vision | 2534.8 | +17.1% | +0.6% | +0.2% | 2515-2544 |
| Cosmos long-prefill | 1325.8 | +18.0% | +1.1% | +0.5% | 1293-1327 |
| Cosmos mixed | 1175.2 | +27.3% | +3.2% | -0.5% | 1162-1184 |
| Cosmos multi-image | 315.2 | +29.2% | +1.2% | +0.5% | 313-323 |
| Cosmos poisson | 2059.9 | +15.7% | +1.8% | +0.9% | 2059-2060 |
| Cosmos short | 2398.7 | +17.2% | -0.4% | -2.3% | 2388-2402 |
| Cosmos text-heavy | 2060.3 | +59.4% | +3.5% | +1.5% | 2036-2067 |
| Cosmos vision-heavy | 736.0 | +27.5% | +3.2% | -0.0% | 735-741 |
| Cosmos wave-drain | 98.0 | +2.2% | -0.1% | -0.0% | 98-98 |
| Gemma balanced | 1269.8 | +64.6% | +0.6% | -0.1% | 1264-1271 |
| Gemma bimodal | 852.9 | +42.1% | +0.7% | +0.9% | 850-854 |
| Gemma decode-heavy | 1380.4 | +69.9% | +0.2% | +0.2% | 1374-1382 |
| Gemma late-vision | 1522.4 | +53.7% | -0.6% | -0.4% | 1520-1531 |
| Gemma long-prefill | 613.4 | +22.6% | -1.0% | +4.2% | 612-620 |
| Gemma mixed | 775.7 | +10.2% | +4.2% | +0.8% | 760-786 |
| Gemma multi-image | 399.4 | +4.7% | +3.1% | -1.3% | 398-408 |
| Gemma poisson | 946.4 | +38.8% | +1.8% | +1.1% | 942-949 |
| Gemma short | 848.3 | +49.5% | -0.7% | -0.6% | 848-863 |
| Gemma text-heavy | 896.9 | +121.6% | +0.3% | -0.5% | 895-906 |
| Gemma vision-heavy | 580.2 | +3.6% | +1.6% | +0.1% | 572-581 |
| Gemma wave-drain | 97.3 | +5.0% | -0.0% | +0.0% | 97-97 |

Latency medians versus `1fab188` are within +-5% everywhere except Gemma mixed TTFT mean (+15%, p95 not
separately checked here) and Cosmos short TTFT mean (+5%). These are noted, not resolved.

## Output audit

Cross-repeat exact token agreement is 1513/1513 for Cosmos and 593/632 for Gemma (note 364 binary: 588/632), with
0 integrity issues. The change does not touch numerics; the Gemma difference is within the batch-shape variation
already characterized in note 364.

## Provenance

- Binary `.local/baselines/measured-authority-c7a1b71-20260928` was built from the source of `c7a1b71` before the
  pre-commit clang-format pass; the committed tree differs only in whitespace.
- Same engines, traces, generic calibration, and frozen vLLM references as note 362.

## Next work

1. Contextual head reference bias (note 365, mechanism 2): residual observations use the remaining-work reference.
   With measured prices now authoritative this is less urgent, but it still misprices unmeasured shapes.
2. Prefill consolidation (prefill us/token is 105 on this trace versus 101 in the best runs).
3. Gemma mixed TTFT +15% mean: check p95 and whether it is the head no longer deferring P+D under slack pressure.
4. Output-quality gate remains the promotion blocker (note 364).
