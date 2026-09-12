# Gemma 4 packed-prefill sentinel gate

Date: 2026-09-12

## Outcome

Gemma 4 packed prefill is a valid and useful engine capability, but the learned V1/V2 policies do not yet pass the
VLM promotion gate. V0 Exact is the current packed-engine champion. The full twelve-workload repeated campaign is
intentionally deferred until the learned VLM regression is understood; multiplying a failing sentinel result would
not establish a useful promotion claim.

The fixed contract is:

- Gemma 4 E2B INT4-AWQ backbone with FP16 KV, PLE, embeddings, and LM head;
- independent TensorRT E/P/D contexts on one RTX 3080;
- E4, packed P8, D24, fixed 128-token chunks;
- 24 stable indexed-paged KV owners and 96 pages;
- identical engine, request traces, calibration, memory limit, and runtime binary across V0/V1/V2;
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

The frozen equal-24 vLLM result uses the same model, request traces, fixed output lengths, and client-side maximum of
24, so token throughput remains a useful external reference. Its internal sequence cap is 8 and it uses a different
engine/runtime. It was not rerun because that contract did not change.

| Workload | Packed V0 | Frozen vLLM | V0 vs vLLM | Earlier dense P4/D24/E8 | V0 vs dense |
|---|---:|---:|---:|---:|---:|
| short | 793.68 | 288.38 | +175.22% | 660.70 | +20.13% |
| balanced | 1,193.30 | 329.72 | +261.91% | 1,107.78 | +7.72% |
| decode-heavy | 1,314.23 | 330.17 | +298.05% | 1,274.64 | +3.11% |
| long-prefill | 436.06 | 261.31 | +66.87% | 401.92 | +8.49% |
| mixed | 437.54 | 270.00 | +62.05% | 645.58 | -32.23% |
| vision-heavy | 326.19 | 285.40 | +14.29% | 445.65 | -26.81% |
| multi-image | 207.23 | 231.73 | -10.57% | 292.05 | -29.04% |

Packed V0 beats frozen vLLM in six of seven sentinels, but multi-image remains 10.57% behind. The dense comparison
is not policy-only: dense uses E8 and tiered E/P while packed uses fully independent E4/P8/D24. It nevertheless shows
that packed P8 opens a substantially stronger text frontier and that the next work must recover VLM formation rather
than increase P throughput further.

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

1. Retain V0 Exact as the packed-engine default champion for further Gemma 4 evaluation.
2. Do not run the full twelve-workload V1/V2 repeated promotion campaign yet; the VLM sentinel gate is failed.
3. Preserve the completed equal-runtime text comparison as the packed-P causal gate; repeat it three times before a
   citable claim.
4. Measure E-ready, E-start, E-complete, P-ready, and first-P-start trajectories for V0 and V1/V2 on scaled
   multi-image and vision-heavy waves.
5. Replace sparse direction authority only if a low-dimensional shared E-pair prior can be validated without a
   workload label or fixed model-specific overlap rule. Otherwise keep learned policy active only for P+D and let
   exact physical costs govern E transitions.
6. Repeat V0 and the surviving learned candidate three times, add arrival-inclusive TTFT/E2E, then expand to the
   remaining five workloads and refresh vLLM only if the external contract changes.

The immediate optimization target is no longer “make P larger.” It is to preserve the packed text gains while
recovering E cohort formation and E-to-P critical-path progress under the same profile-free contract.
