# Gemma 4 graph, lifetime admission, and sampling-refill full-12

## Outcome

The final campaign completed all twelve real-request HTTP workloads three times with no failed cells. It combines
the retained P8/D24/E4 engine, stable paged KV ownership, lifetime-aware encoded-vision admission, independent
TensorRT contexts, CUDA graph replay, graph/eager policy-evidence sharing, and final-prefill sampling visibility in
the decode refill horizon. The scheduler receives no workload label.

Against the unchanged one-run vLLM 0.28 capacity reference, Current wins generated-token throughput on 9/12
workloads and has 23.87% higher geometric-mean throughput. It wins mean TPOT on 11/12 and mean E2E on 8/12. It is
not a Pareto winner: TTFT mean wins only 5/12, TTFT p95 wins 4/12, and Current uses about 620 MiB more peak GPU
memory in the VLM cells.

The three throughput losses are bimodal (-0.85%), long-prefill (-12.69%), and vision-heavy (-2.57%). The strongest
remaining problem is first-token progress, not empty GPU time or decode kernel service.

## Architecture exercised

```text
HTTP request/DAG
      |
      +--> E queue -- independent vision TensorRT context --+
      |                                                     |
      +--> P queue -- independent packed-P TensorRT context +--> sampling event --> D-ready
      |                                                     |
      +--> D queue -- independent decode TensorRT context --+
                         |
                         +-- stable indexed-paged KV ownership

Global selector
  feasibility -> service/slack protection -> contextual physical value
  exact execution key keeps graph/eager variants separate
  contextual policy features share graph/eager evidence

Lifetime admission
  initial vision storage: 4 requests
  tracked downstream ownership horizon: up to 12 requests
  KV pool and E/P/D engine capabilities are unchanged
```

Automatic calibration capture populates graph variants before the measurement epoch. Serving does not learn a
separate policy merely because the same kernels are launched through graph replay. A final P sampling ticket is
treated as an observable producer of a future D row, allowing the scheduler to compare D-now with the concrete
sampling-completion horizon.

## Full comparison

Current values are three-run medians or medians of per-run request statistics. vLLM values are the retained
single-run capacity reference. Latencies are milliseconds.

| Workload | tok/s Current / vLLM | TTFT mean/p95 Current | TTFT vLLM | TPOT mean/p95 Current | TPOT vLLM | E2E mean/p95 Current | E2E vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 1181.67 / 771.46 | 89.8 / 214.0 | 134.3 / 234.3 | 16.60 / 18.20 | 23.76 / 24.56 | 1473.6 / 2288.4 | 2128.0 / 3244.8 |
| bimodal | 595.08 / 600.16 | 2208.6 / 5676.8 | 317.4 / 878.4 | 19.77 / 33.12 | 28.83 / 37.42 | 4820.2 / 10146.1 | 4389.9 / 9582.0 |
| decode-heavy | 1306.03 / 812.43 | 88.6 / 224.1 | 153.6 / 249.7 | 15.22 / 15.79 | 23.04 / 23.41 | 3938.6 / 6034.9 | 6006.2 / 9096.0 |
| late-vision | 1463.29 / 990.82 | 144.8 / 371.0 | 137.7 / 243.3 | 14.26 / 14.32 | 22.55 / 22.55 | 2185.6 / 2806.2 | 3367.5 / 4416.8 |
| long-prefill | 436.76 / 500.26 | 2226.8 / 3171.5 | 543.9 / 1442.0 | 22.59 / 26.70 | 36.37 / 44.60 | 4115.2 / 6015.5 | 3554.6 / 5867.0 |
| mixed | 717.65 / 703.81 | 272.6 / 635.1 | 276.4 / 410.6 | 25.97 / 35.00 | 26.48 / 32.39 | 1454.1 / 2214.0 | 1473.8 / 2162.6 |
| multi-image | 394.75 / 381.34 | 387.1 / 663.7 | 188.9 / 227.2 | 25.85 / 40.82 | 29.70 / 35.54 | 1171.5 / 1465.9 | 1109.6 / 1300.3 |
| poisson | 889.52 / 681.95 | 132.1 / 404.7 | 119.8 / 169.8 | 21.59 / 26.11 | 26.02 / 29.23 | 1654.6 / 2991.4 | 1968.9 / 3502.6 |
| short | 796.84 / 567.55 | 99.5 / 224.5 | 155.4 / 242.3 | 22.58 / 30.14 | 26.39 / 30.11 | 544.8 / 899.5 | 695.4 / 1068.2 |
| text-heavy | 852.39 / 404.66 | 179.7 / 496.8 | 1658.7 / 4312.0 | 22.54 / 27.88 | 22.47 / 27.54 | 1354.6 / 1743.8 | 2815.0 / 5848.8 |
| vision-heavy | 545.45 / 559.83 | 410.4 / 758.6 | 280.6 / 390.8 | 29.79 / 40.16 | 30.59 / 40.54 | 1550.9 / 2186.1 | 1420.7 / 2138.3 |
| wave-drain | 96.82 / 92.62 | 260.6 / 330.1 | 178.2 / 204.4 | 10.25 / 11.63 | 22.48 / 24.58 | 577.1 / 596.0 | 875.0 / 884.4 |

## Geometric-mean result

Positive improvement means Current is better; latency improvement accounts for lower-is-better direction.

| Metric | Current | vLLM | Current improvement | Workload wins |
|---|---:|---:|---:|---:|
| token throughput | 646.24 tok/s | 521.69 tok/s | +23.87% | 9/12 |
| TTFT mean | 270.15 ms | 241.59 ms | -11.82% | 5/12 |
| TTFT p95 | 582.50 ms | 411.88 ms | -41.43% | 4/12 |
| TPOT mean | 19.80 ms | 26.27 ms | +24.63% | 11/12 |
| TPOT p95 | 24.77 ms | 30.31 ms | +18.30% | 8/12 |
| E2E mean | 1669.39 ms | 2042.19 ms | +18.26% | 8/12 |
| E2E p95 | 2450.68 ms | 3120.58 ms | +21.47% | 7/12 |

Compared with the earlier one-run packed V3 full-12 in note 295, Current improves geometric-mean throughput by
14.85%, TTFT mean by 26.87%, TTFT p95 by 34.62%, E2E mean by 5.49%, and E2E p95 by 12.76%. This trades away some
decode continuity: TPOT mean and p95 are about 20.0% and 21.3% higher than that earlier low-admission run, although
they remain better than vLLM in aggregate.

## VLM mechanism isolation

Against the prior three-run Fixed Full lifetime-admission result, the latest stack is approximately throughput
neutral on mixed (-0.12%) and vision-heavy (-0.13%), while improving multi-image by 3.14%. Multi-image also improves
TTFT mean/p95 by 7.42%/2.72% and E2E mean/p95 by 2.63%/1.40%; its TPOT mean is 2.09% worse. The graph/refill changes
therefore recover a real multi-image fragmentation loss without explaining the remaining vision-heavy gap.

## Activity and bottleneck evidence

| Workload | all idle | E/P/D overlap | P dispatch / mean BS | D dispatch / mean BS | E dispatch / mean BS | P/D graph hit |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 1.24% | 8.26% | 43 / 1.95 | 297 / 18.10 | 0 / 0 | 37.4% / 36.5% |
| bimodal | 1.05% | 11.26% | 172 / 1.58 | 1000 / 9.62 | 0 / 0 | 24.5% / 46.1% |
| long-prefill | 1.09% | 26.04% | 293 / 1.54 | 576 / 9.33 | 0 / 0 | 18.1% / 64.2% |
| mixed | 1.95% | 25.80% | 74 / 2.00 | 149 / 19.22 | 17 / 1.88 | 33.6% / 44.8% |
| multi-image | 2.01% | 29.83% | 28 / 2.43 | 46 / 13.48 | 8 / 2.50 | 41.1% / 58.8% |
| vision-heavy | 2.06% | 24.12% | 69 / 2.64 | 155 / 15.48 | 20 / 2.40 | 34.5% / 49.0% |
| wave-drain | 65.65% | 6.16% | 40 / 1.70 | 145 / 4.28 | 8 / 2.50 | 38.8% / 44.2% |

All ordinary continuous-arrival workloads leave only about 1--2% all-idle time. Wave-drain deliberately separates
arrival waves and therefore spends 65.65% of the measurement span idle; it is not evidence that the saturated
scheduler fails to submit work.

Long-prefill and bimodal are dominated by P128 action fragmentation. Long-prefill performs 293 P dispatches with
mean P batch 1.54 and only 18.1% P graph hits. The retained profiled P512 experiment reduced long-prefill P
dispatches by about 71% and approached vLLM throughput, but a globally fixed P512 chunk fragmented VLM decode
cohorts. The next implementation must choose a transition-safe chunk from observable ready/slack/cohort state; it
must not switch on a workload name.

Vision-heavy remains an E-to-first-token problem. Its GPU is busy and its TPOT is already competitive; additional
unconditional overlap would move contention rather than remove the critical path. Multi-image demonstrates the
opposite trade-off: lifetime admission and refill make large downstream cohorts and win throughput, but TTFT and
TPOT p95 still trail vLLM.

## Correctness and claim limits

- Focused graph/refill tests pass 5/5; related scheduler, global selector, three-phase, memory, and contextual-model
  tests pass 246/246.
- Every run generated the requested fixed output count and completed HTTP service successfully.
- Gemma token hashes vary across fresh Current repeats, including prior baselines. This campaign does not claim
  exact cross-runtime greedy identity.
- Current is measured three times per workload; frozen vLLM is one retained run per workload. This is a strong
  engineering validation, not a confidence-interval paper claim.
- Current VLM peak is 9461--9469 MiB versus 8843 MiB for the frozen vLLM cells. Throughput gains are not equal-memory
  claims.

## Next implementation gates

1. Rebuild the retained multi-profile P512/P128 capability and make chunk selection depend on bounded successor-D
   formation and service slack rather than a workload label.
2. Gate P512 first on long-prefill and bimodal causal replay, then ensure mixed, vision-heavy, and multi-image keep
   P128 when the longer action would fragment their next D cohort.
3. Shorten the E-to-first-token path using already-observable completion events and bounded encoded ownership;
   avoid increasing lifetime capacity solely to maximize throughput.
4. Repeat only the winning Current candidate and frozen vLLM at least five times on the three loss workloads and
   the three VLM workloads.
5. Resolve canonical Gemma numerical determinism separately from scheduling performance.

Structured artifacts are in
`.local/results/gemma4-vllm-gap-20260915/graphs-lifetime-final-refill-confirm/`.

After validation, the run-002 and run-003 raw `gateway.log` files were removed for all twelve workloads, reclaiming
633,196,173 bytes. All three per-request summaries, aggregate files, activity analyses, comparison tables, and the
run-001 raw gateway log for each workload remain. This campaign is therefore reproducible at the summary level and
retains one raw diagnostic trace per workload without retaining redundant verbose logs.
