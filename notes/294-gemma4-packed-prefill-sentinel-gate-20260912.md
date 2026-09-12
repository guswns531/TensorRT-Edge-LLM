# Gemma 4 packed-prefill sentinel gate

Date: 2026-09-12

## Outcome

Gemma 4 packed prefill is a valid and useful engine capability, but the learned V1/V2 policies do not pass the VLM
promotion gate. A follow-up same-engine V3 Service-scaled Transition run recovers the mixed and vision-heavy
regressions and is the seven-workload geometric-mean champion. V0 Exact remains better on long-prefill and
multi-image, so V3 is the strongest promotion candidate rather than a universal winner. The full twelve-workload
repeated campaign remains deferred until V3 is repeated and the multi-image regression is understood.

The fixed contract is:

- Gemma 4 E2B INT4-AWQ backbone with FP16 KV, PLE, embeddings, and LM head;
- independent TensorRT E/P/D contexts on one RTX 3080;
- E4, packed P8, D24, fixed 128-token chunks;
- 24 stable indexed-paged KV owners and 96 pages;
- identical engine, request traces, calibration, memory limit, and runtime binary across V0/V1/V2/V3;
- no numeric TTFT/TPOT SLO target, while retaining the correctness-oriented prefill TTFT hard guard;
- one diagnostic repeat per point.

Raw results and their manifest are under
`.local/results/gemma4-packed-prefill-g4-20260912/`.

## Seven-workload sentinel result

Token throughput is generated tokens per second. Percentages are relative to V0 Exact.

| Workload | V0 Exact | V1 Scalar | V1 vs V0 | V2 Scalar+Transition | V2 vs V0 |
|---|---:|---:|---:|---:|---:|
| short | 793.68 | 804.56 | +1.37% | 793.66 | -0.00% |
| balanced | 1,193.30 | 1,203.56 | +0.86% | 1,204.42 | +0.93% |
| decode-heavy | 1,314.23 | 1,320.30 | +0.46% | 1,315.60 | +0.10% |
| long-prefill | 436.06 | 426.90 | -2.10% | 430.20 | -1.34% |
| mixed | 437.54 | 365.27 | -16.52% | 389.14 | -11.06% |
| vision-heavy | 326.19 | 249.07 | -23.64% | 253.01 | -22.43% |
| multi-image | 207.23 | 162.63 | -21.52% | 146.16 | -29.47% |

The text-only points remain within 2.1% of V0 and V1 slightly improves three of them. The learned policies therefore
do not expose a packed-P kernel regression. Their failure is isolated to phase interaction under VLM traffic.

## Same-engine V3 comparison

V3 was subsequently run with the same packed engine, binary, 49-request generic calibration, traces, and runtime
limits. Only `TRT_EDGELLM_PHASE_POLICY` changed to `service-scaled-transition`. This is a policy-only comparison;
it must not be confused with the earlier dense P4/D24/E8 V3 campaign.

| Workload | V0 Exact | V1 Scalar | V2 Scalar+Transition | V3 Service-scaled Transition | V3 vs V0 |
|---|---:|---:|---:|---:|---:|
| short | 793.68 | **804.56** | 793.66 | 789.92 | -0.47% |
| balanced | 1,193.30 | 1,203.56 | **1,204.42** | 1,181.99 | -0.95% |
| decode-heavy | 1,314.23 | **1,320.30** | 1,315.60 | 1,293.43 | -1.58% |
| long-prefill | **436.06** | 426.90 | 430.20 | 423.84 | -2.80% |
| mixed | 437.54 | 365.27 | 389.14 | **478.90** | +9.45% |
| vision-heavy | 326.19 | 249.07 | 253.01 | **359.54** | +10.22% |
| multi-image | **207.23** | 162.63 | 146.16 | 194.22 | -6.28% |

The per-workload token-throughput winner is V1 for short and decode-heavy, V2 for balanced, V0 for long-prefill
and multi-image, and V3 for mixed and vision-heavy. No workload label is provided to any policy.

Positive percentages below mean V3 is better. Latency improvement is `(V0 / V3 - 1)`, while throughput improvement
is `(V3 / V0 - 1)`.

| Workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | -0.47% | +6.96% | +6.67% | -0.80% | +6.69% | -0.16% | -4.42% |
| balanced | -0.95% | +3.26% | -1.64% | -1.39% | -1.88% | -1.17% | -1.22% |
| decode-heavy | -1.58% | +2.85% | +6.53% | -1.61% | -2.05% | -1.59% | -1.85% |
| long-prefill | -2.80% | -7.05% | -9.69% | +2.35% | +1.77% | -2.77% | -3.82% |
| mixed | +9.45% | +35.13% | +17.65% | -2.59% | +0.67% | +14.04% | +13.36% |
| vision-heavy | +10.22% | +9.54% | +6.46% | +25.67% | -0.01% | +12.39% | +9.25% |
| multi-image | -6.28% | -10.80% | -12.66% | +17.54% | +14.36% | -4.69% | -9.77% |

Across all seven sentinels, geometric means are:

| Policy | tok/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|
| V0 Exact | 554.18 | 446.98/961.51 | 17.73/21.44 | 1,900.44/2,957.20 |
| V1 Scalar | 502.37 | 510.56/1,067.83 | 18.59/21.93 | 2,119.93/3,337.38 |
| V2 Scalar+Transition | 499.75 | 504.05/1,045.98 | 18.45/22.09 | 2,129.60/3,325.23 |
| **V3 Service-scaled Transition** | **559.29** | **426.27/948.00** | **16.87/20.88** | **1,862.11**/2,958.94 |

Relative to V0, V3 improves geometric-mean token throughput by 0.92%, TTFT mean/p95 by 4.86%/1.43%, TPOT
mean/p95 by 5.11%/2.65%, and E2E mean by 2.06%. E2E p95 is 0.06% worse, which is indistinguishable in a one-run
diagnostic. Relative to V1 and V2, V3 improves every geometric-mean metric; the gains range from 5.01% to 19.77%.

The scheduler summaries explain the recovery direction without proving causality. On mixed, V3 selects 174
event-backed WAIT actions and 47 overlaps, compared with 70/17 for V1 and 11/18 for V2. On vision-heavy it selects
17 WAITs and 38 overlaps, while V1 selects 0/6 and V2 0/22. Action-fidelity violations remain zero. Service-scaled
urgency therefore restores useful progress choices that sparse scalar E-pair authority suppressed. Multi-image is
the counterexample: V3 reduces the V1/V2 loss but its 21 WAITs and seven overlaps do not recover V0's trajectory.
The next comparison must use request-ready/dispatch trajectories rather than add a workload-specific V3 rule.

Peak allocation remains 9,377--9,383 MiB across V0--V3. The V3 improvement is not purchased with a different KV
pool, slot count, engine, or context allocation.

### V0 latency and memory

These are service-side latencies measured after the HTTP client sends a request. They must not be compared directly
with the arrival-inclusive values in the earlier equal-24 campaign.

| Workload | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | Peak MiB |
|---|---:|---:|---:|---:|
| short | 103.36/242.81 | 22.60/29.74 | 550.27/864.35 | 9,377 |
| balanced | 90.30/214.30 | 16.39/17.57 | 1,454.77/2,234.69 | 9,377 |
| decode-heavy | 91.45/237.89 | 15.04/15.48 | 3,896.57/5,946.82 | 9,377 |
| long-prefill | 2,185.16/3,091.94 | 23.04/26.08 | 4,117.16/5,767.15 | 9,377 |
| mixed | 984.82/2,934.11 | 19.26/25.06 | 1,896.42/3,409.74 | 9,383 |
| vision-heavy | 1,669.46/3,354.39 | 16.40/22.56 | 2,320.99/3,795.54 | 9,379 |
| multi-image | 1,162.58/2,016.83 | 13.59/17.44 | 1,583.89/2,306.87 | 9,379 |

All requests completed their fixed output-token contract without OOM. Peak allocation is about 9,375--9,383 MiB,
roughly 350 MiB below the earlier dense-P4/D24/E8 frontier. This difference includes the E4-versus-E8 vision-engine
capacity change, so it is an architectural-frontier measurement rather than a pure packed-P memory ablation.

## External references

> **Superseded capacity comparison:** the frozen vLLM numbers below retain `max_num_seqs=8` and must not be used as
> the final comparator for the later 24-in-flight traces. The corrected vLLM capacity sweep, full twelve-workload
> result, and revised conclusions are in
> [note 295](295-gemma4-vllm-capacity-frontier-20260912.md).

The frozen equal-24 vLLM result uses the same model, request traces, fixed output lengths, and client-side maximum of
24, so token throughput remains a useful external reference. Its internal sequence cap is 8 and it uses a different
engine/runtime. It was not rerun because that contract did not change.

| Workload | Packed V3 | Frozen vLLM | V3 vs vLLM | Earlier dense V3 P4/D24/E8 | Packed vs dense V3 |
|---|---:|---:|---:|---:|---:|
| short | 789.92 | 288.38 | +173.92% | 660.70 | +19.56% |
| balanced | 1,181.99 | 329.72 | +258.48% | 1,107.78 | +6.70% |
| decode-heavy | 1,293.43 | 330.17 | +291.75% | 1,274.64 | +1.47% |
| long-prefill | 423.84 | 261.31 | +62.20% | 401.92 | +5.45% |
| mixed | 478.90 | 270.00 | +77.37% | 645.58 | -25.82% |
| vision-heavy | 359.54 | 285.40 | +25.98% | 445.65 | -19.32% |
| multi-image | 194.22 | 231.73 | -16.19% | 292.05 | -33.50% |

Packed V3 beats frozen vLLM in six of seven sentinels, but multi-image remains 16.19% behind. The dense comparison
is not policy-only: dense uses E8 and tiered E/P while packed uses fully independent E4/P8/D24. Packed V3 improves
the four text traces by 1.47--19.56% but loses 19.32--33.50% on the vision-heavy pair. Packed P8 therefore opens a
stronger text frontier, while reducing E8 to E4 and removing the tiered preparation/formation path gives back much
of the earlier dense V3 VLM advantage. The next work must recover E formation without discarding the packed-P
engine or reintroducing model-specific policy constants.

## Equal-runtime text-only dense versus packed

A new dense engine was built at exactly P4/D24 with 96 KV pages. Both engines were then run without loading a vision
engine, using the same 35-request text-only generic calibration, 24 stable slots, V0 Exact policy, request traces,
binary, and runtime options. This removes the earlier E8/E4 and VLM-calibration confounds. Negative latency deltas
mean packed is faster.

| Workload | Dense P4 tok/s | Packed P8 tok/s | tok/s delta | TTFT mean/p95 delta | TPOT mean/p95 delta | E2E mean/p95 delta |
|---|---:|---:|---:|---:|---:|---:|
| short | 690.20 | 782.57 | +13.38% | -31.73%/-45.26% | -8.57%/-1.51% | -13.95%/-15.59% |
| balanced | 1,125.32 | 1,188.56 | +5.62% | -18.09%/-44.31% | -5.65%/-10.77% | -6.79%/-8.83% |
| decode-heavy | 1,295.75 | 1,310.31 | +1.12% | -16.60%/-39.88% | -1.23%/-2.85% | -1.63%/-2.98% |
| long-prefill | 425.47 | 439.03 | +3.19% | -3.41%/-3.46% | -4.19%/-4.94% | -3.64%/-2.80% |

Packed improves every reported throughput and latency metric in all four text sentinels. GPU allocation also falls
from 9,117 MiB to 8,585 MiB, a 532 MiB reduction under the same slots/pages contract. This is not a KV-capacity
change. The packed engine's profile-local activation/workspace requirement is smaller even though its P capacity is
larger.

The dense P4/D24 engine was also tested with the same separate E4 vision context. It initializes the independent
text P/D contexts but OOMs when adding the dense vision-prefill context. The same failure occurs with retained D32
and D48 dense engines. Packed P can use one profile-local P context for text and vision, whereas the dense VLM path
requires another text-engine context. On this 10 GiB GPU the fair VLM result is therefore a feasibility distinction:

```text
dense:  E + text-P + vision-P + D  -> OOM
packed: E + unified packed-P + D   -> 9.38 GiB and completes
```

This explains why the earlier tiered dense champion reused contexts: it could not provide the same fully independent
frontier within the device budget.

## Calibration diagnosis

The original generic calibration has 49 requests: 26 P, 14 E, and 9 resident-D requests. P+D obtains 46--49
observations, while E+P is usually observed only once and E+D zero times. The four-observation authority threshold is
therefore reached for P+D but not for the sparse E directions. V1/V2 then make conservative serial decisions in the
measured VLM epoch while V0 can use exact measured costs.

Two diagnostic calibrations were tested without changing any workload or runtime setting.

| Calibration | Requests | E+P probes | E+D probes | V1 mixed | V1 vision-heavy | V1 multi-image | V2 mixed | V2 vision-heavy | V2 multi-image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| original, two cycles | 49 | usually 1 | 0 | 365.27 | 249.07 | 162.63 | 389.14 | 253.01 | 146.16 |
| original layout, four cycles | 95 | 1--4 | 0--1 | 428.79 | 264.22 | 178.87 | 432.59 | 241.95 | 171.95 |
| separated pair epochs | 160 | 2--4 | 0--1 | 393.41 | 281.72 | 195.60 | 411.29 | 253.30 | 173.97 |
| V0 Exact reference | 49 | not applicable | not applicable | 437.54 | 326.19 | 207.23 | 437.54 | 326.19 | 207.23 |

Longer calibration partially recovers mixed and multi-image, but the result is non-monotonic. The separated trace
improves V1 vision-heavy and multi-image while making mixed worse than the ordinary four-cycle calibration. It also
fails to create deterministic E+D coverage. This rejects a simple “more warmup fixes VLM” explanation and provides
no basis for promoting a fixed pair-calibration sequence. The experimental pair-epoch source option was therefore
not retained.

## Execution interpretation

V0 and V1/V2 share stable KV ownership, P8 ragged packing, CUDA graphs, contexts, queues, and memory. Action-fidelity
violations are zero. The visible difference is decision authority:

- V0 selects measured safe overlap and bounded WAIT actions in natural VLM traffic.
- V1/V2 have a well-trained P+D model but sparse E-pair posteriors, so they suppress most E overlap and WAIT actions.
- V2 recovers some mixed traffic through deterministic successor reasoning, but multi-image can form more E batches
  and extend the E-to-P critical path.
- Increasing calibration changes learned decisions but does not reliably improve them, indicating that physical
  feature/label transfer across E shapes is also involved.

Token traces are deterministic within each individual run, but cross-policy token hashes are not identical for many
workloads. Dense-versus-packed controlled tests with identical scheduling were greedy-token exact; cross-policy
identity remains a separate canonical row-order/FP16 numerical gate and is not claimed here.

## Promotion decision and next work

1. Retain V0 Exact as the conservative packed reference and promote V3 as the learned candidate for repeated Gemma
   4 evaluation. V3 is the aggregate champion, but not a universal per-workload winner.
2. Do not run the full twelve-workload V1/V2 campaign; both fail the VLM sentinel gate. Compare V0 and V3 next.
3. Preserve the completed equal-runtime text comparison as the packed-P causal gate; repeat it three times before a
   citable claim.
4. Measure E-ready, E-start, E-complete, P-ready, and first-P-start trajectories for V0 and V3 on scaled
   multi-image and vision-heavy waves.
5. Replace sparse direction authority only if a low-dimensional shared E-pair prior can be validated without a
   workload label or fixed model-specific overlap rule. Otherwise keep learned policy active only for P+D and let
   exact physical costs govern E transitions.
6. Repeat V0 and V3 three times, add arrival-inclusive TTFT/E2E, then expand to the
   remaining five workloads and refresh vLLM only if the external contract changes.

The immediate optimization target is no longer “make P larger.” It is to preserve the packed text gains while
recovering E cohort formation and E-to-P critical-path progress under the same profile-free contract.
