<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Thor decode-gap instrumentation and sampling overlap

## Result

The Qwen3.8 short-burst low-throughput regime is dominated by the D32 TensorRT execution itself, not by changing decode
batch sizes. Sampling is the next visible gap, but overlapping it on a second CUDA stream slows the main engine more than
it hides. Async sampling was therefore removed.

Opt-in timing instrumentation is retained behind `TRT_EDGELLM_PHASE_TIMING_METRICS=1`. The normal production path does not
take timestamps or add timing JSON. When enabled, the gateway `/metrics` endpoint returns the latest cumulative snapshot.

## Instrumentation

The phase server records:

- sampling ticket submit-to-ready time;
- sampled-token ready-to-next-decode-dispatch time per row;
- decode batch-size histogram;
- cumulative decode GPU milliseconds and sample count.

The completion event carries the cumulative snapshot, and the Python gateway stores the newest snapshot for `/metrics`.
Successive snapshots can be subtracted to obtain one workload run without restarting the server.

Enable it with:

```bash
PHASE_TIMING_METRICS=1 experiments/qwen38_thor/run_phase_server.sh thor-throughput
curl http://127.0.0.1:8001/metrics
```

The final image `tensorrt-edge-llm:qwen38-phase-gap-metrics-final` built successfully. Focused phase tests passed 6/6 and
the gateway passed Python bytecode compilation.

## Three consecutive short-burst runs

All runs used the same server process, locked clocks, 64 requests, concurrency 32, 128 input tokens, and 64 fixed output
tokens. Every run completed 64/64 requests.

| run | output tok/s | D32 dispatches | mean decode GPU | mean sampling ready | mean ready-to-dispatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 155.21 | 124 | 176.05 ms | 11.76 ms | 13.31 ms/row |
| 2 | 155.89 | 124 | 175.76 ms | 11.60 ms | 13.30 ms/row |
| 3 | 155.93 | 124 | 175.72 ms | 11.61 ms | 13.30 ms/row |

Each run had the same tail histogram: one D3, one D4, one D28, one D29, and 124 D32 dispatches. The stable D32 cohort is
therefore working as intended. Batch fragmentation does not explain this regime.

The approximate critical path per steady D32 turn is:

```text
TensorRT D32 decode       175.7-176.1 ms
sampling submit -> ready   11.6-11.8 ms
refill ready -> dispatch   ~13.3 ms
```

The row-level ready-to-dispatch sum repeats the same batch wait for each row; it is diagnostic and must not be added 32
times to wall-clock latency.

## Rejected async sampling

The candidate moved logits copy, top-1 reduction, and token D2H to a dedicated non-default stream. The main decode stream
waited only until the logits D2D copy completed, allowing the next TensorRT D32 enqueue to overlap top-1 and D2H.

It was correct but slower on the first fixed-output short run:

| path | output tok/s | mean decode GPU | TPOT median |
| --- | ---: | ---: | ---: |
| serialized baseline | 155.21-155.93 | 175.7-176.1 ms | 190.2-190.7 ms |
| async sampling | 150.56 | 182.37 ms | 197.0 ms |

Sampling submit-to-ready remained about 11.8 ms while TensorRT decode increased about 6.3 ms. Top-1 and D32 compete for
SM/cache/shared-LPDDR resources, so overlap converts a visible host gap into a larger engine slowdown. The candidate was
removed from source.

## Interpretation of older fast results

The older locked campaign reported 187.43 short-burst tok/s and 213.11 decode-heavy tok/s. Current clocks are still locked
at GPU GPC 1.575 GHz, GPU NVD 1.692 GHz, EMC 4.266 GHz, CPU 2.601 GHz, and MAXN. Current D32 execution is nevertheless about
176 ms and stable across repeated runs. The older fast result implies a materially lower D32 engine time; host dispatch
gaps alone cannot account for the full difference.

Future comparisons must record cumulative decode GPU time in addition to HTTP throughput. A run with different engine
latency is not a valid runtime-scheduler A/B even when clock sysfs values match.

## Next performance target

Move from runtime micro-optimizations to an engine-level Qwen3.8 D64 experiment:

1. Build an asymmetric dense P4/D64 engine while preserving the current P4 prefill profile.
2. Use 64 stable slots and retain the D64 recurrent cohort on GPU.
3. Measure D32 and D64 decode-only GPU latency at context lengths 128, 512, 2,048, and 4,096.
4. Accept D64 only when `64 / latency` beats current D32 aggregate throughput without exceeding the unified-memory budget.
5. Repeat short, wave, and decode-heavy HTTP three times with timing metrics enabled.

The D64 path targets the dominant 176 ms TensorRT phase directly. More sampling or host-copy changes are unlikely to move
the overall result until engine throughput improves.
