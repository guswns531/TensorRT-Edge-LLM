<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Thor IPC polling and greedy-sampling experiments

## Decision

Keep the existing busy-poll IPC loop and generic two-stage `selectAllTopK` greedy path for the throughput server. An
opt-in 50 microsecond poll backoff and a greedy-only argmax kernel were implemented, built, unit-tested, and measured on
Qwen3.8-27B. Both failed at least one real HTTP throughput gate and were removed from production source.

The source tree after this experiment contains only this note. Machine-readable results remain under
`data/qwen38/results/next`.

## Candidate A: 50 microsecond IPC idle backoff

The phase IPC backend busy-polls CUDA completion and sampling events. In a representative server lifecycle it consumed one
CPU core and called `poll()` 228,587,353 times. The candidate slept for 50 microseconds only when one loop iteration made no
progress, while preserving the existing ingress, dispatch, sampling, and serialization order.

Build and correctness:

- Thor development image built successfully.
- Focused phase server and recurrent-state tests: 6/6 passed.
- Every HTTP run completed 64/64 requests with the exact configured output count.

The backoff reduced poll calls from about 228.6 million to 3.22 million, a 70.9x reduction. Mean CPU/SOC power fell from
19.48 W to 17.45 W and mean board input power fell from 93.24 W to 85.61 W in the measured short/wave campaign.

It still failed the throughput gate:

| workload | busy poll | 50 us backoff | change |
| --- | ---: | ---: | ---: |
| short burst | 156.76 tok/s | 156.18 tok/s | -0.37% |
| wave burst | 177.02 tok/s | 156.83 tok/s | -11.41% |

The sleep delayed detection of decode completion and sampling refill. A timer-based wait is therefore not suitable while
requests or GPU work are active. An idle-only sleep remains a possible power feature, but it does not improve serving
throughput and should not be part of the performance preset.

## Candidate B: greedy-only argmax

The generic `selectAllTopK(topK=1)` path materializes an FP32 copy of the full logits tensor before its two reduction
stages. The candidate kept the same eight-block-per-row reduction but scanned logits directly, removing the full temporary
write/read when callers requested only indices. Top-k values and non-greedy paths were unchanged.

Build and correctness:

- `llm_build`, `llm_bench`, `llm_inference`, `unitTest`, and the phase server built successfully.
- Sampling and return-all-top-k tests: 14/14 passed on Thor.
- A new 17-row, 8,193-vocabulary correctness case crossed all eight reduction lanes.
- Every HTTP request completed with the exact fixed output-token count.

Results:

| workload | reference | greedy argmax | change |
| --- | ---: | ---: | ---: |
| short burst | 156.76 tok/s | 157.06 tok/s | +0.19% |
| wave burst | 177.02 tok/s | 149.49 tok/s | -15.55% |
| decode heavy | 213.11 tok/s | 175.57 tok/s | -17.62%* |

`*` The decode-heavy reference is the previous locked-clock three-run campaign rather than an immediately interleaved
lifecycle, so it is corroborating evidence rather than a strict paired A/B. The short/wave comparison used fresh server
lifecycles from the same session.

The generic FP32 staging traffic did not prove to be the bottleneck. Removing it lowered GPU power and did not improve the
stable short case. The existing path likely benefits from its highly parallel, contiguous staged reduction on this large
248K vocabulary.

## Important measurement finding

Across several fresh development images, throughput entered two repeatable regimes even with GPU and EMC clocks locked:

- short burst: roughly 153-157 tok/s, occasionally transitioning to about 172 tok/s;
- wave burst: roughly 146-157 tok/s, occasionally transitioning to about 174-178 tok/s;
- decode heavy: roughly 175 tok/s, later reaching about 194 tok/s.

The transition also occurred after rejected state-copy experiments, so it is not specific to argmax or memory-copy code.
Qwen3.8 hybrid serving disables dynamic decode batching; online decode-cost observations therefore do not explain the
transition. The stronger signal is an intermittent dispatch/refill gap: the fast regime has higher GPU power and lower
TPOT without changing the D32 engine or clocks.

Future kernel A/B results must not compare a cold low-utilization lifecycle with a later high-utilization lifecycle. Use
interleaved fresh lifecycles or explicitly prime until decode batch distribution and TPOT stabilize.

## Next direction

Instrument dispatch gaps rather than changing another kernel:

1. Emit per-dispatch decode batch size, queue wait, sampling-ready delay, and time from `decodeDone` to the next enqueue.
2. Compare one slow and one fast lifecycle with the same short and wave inputs.
3. If the gap is sampling-event notification, replace timer sleeps with event-driven wakeup from CUDA completion.
4. If the gap is cohort refill policy, prime or persist the stable D32 cohort and remove the D32-plus-tail transition.
5. Only resume kernel work after the dispatch utilization regime is held constant.

Machine-readable summaries:

- `data/qwen38/results/next/legacy-state-short-wave/summary.json`
- `data/qwen38/results/next/ipc-idle50-short-wave/summary.json`
- `data/qwen38/results/next/greedy-argmax-short-wave/summary.json`
- `data/qwen38/results/next/greedy-argmax-decode-heavy/summary.json`
- `data/qwen38/results/next/core-repeats-phase-throughput/summary.json`
