SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Prefill CUDA graph recurring-shape policy

## Outcome

The independent prefill and decode TensorRT contexts now capture a binding
shape when it recurs even if another shape ran between its first and second
occurrences. The live request is still executed exactly once. Separate graph
limits allow the prefill cache to remain small on the RTX 3080 while decode
keeps the larger cache that produces most of the launch-overhead benefit.

The recommended Cosmos P4/D64 setting for the current 10 GB environment is:

```text
--cudaGraph --maxPrefillCudaGraphs 4 --maxDecodeCudaGraphs 64
```

This is a memory-constrained recommendation for the measured engine, not a
model-independent production default.

## Safe recurring-shape capture

```text
first occurrence of shape A
  -> enqueueV3(A)
  -> remember exact addresses + shapes

shape B may run

second occurrence of shape A
  -> enqueueV3(A) exactly once
  -> synchronize only A's owning phase stream
  -> capture enqueueV3(A), but do not launch the captured graph
  -> cache graph A

later shape A
  -> cudaGraphLaunch(A)
```

The post-enqueue capture is necessary because TensorRT requires the current
dynamic shape to be warmed before stream capture. Not launching the new graph
on that turn prevents a duplicate KV write. Consecutive recurring shapes keep
the lower-overhead capture-and-launch path added in the previous step.

The candidate observation table is bounded to four times the graph limit.
Old one-off binding states are evicted FIFO, preventing host-memory growth in a
long-lived server with highly variable sequence lengths.

## Phase-local memory control

`llm_phase_bench` and the Cosmos matrix runner accept:

```text
--maxPrefillCudaGraphs N
--maxDecodeCudaGraphs N
```

When omitted, each inherits `--maxCudaGraphs`. Capture failure, graph-launch
failure, candidate eviction, cache-limit bypass, and post-enqueue capture are
reported separately for each execution context.

## Shape bucketing experiment rejected

An initial experiment converted 128-token prefill tails into 64/32/16/8 token
buckets and restricted prefill batch sizes to powers of two. It did not pad or
write fake KV tokens, but changing chunk boundaries still changed FP16
accumulation and added TensorRT executions.

On the paired 288-request trace:

| Metric | Existing fixed-128 | Shape bucketing | Change |
| --- | ---: | ---: | ---: |
| Prefill dispatches | 277 | 477 | +72.2% |
| Generated token/s | 4396.3 | 3370.9 | -23.3% |
| TTFT median | 4456.7 ms | 7506.1 ms | +68.4% |
| TPOT p95 | 13.180 ms | 15.844 ms | +20.2% |
| Output mismatches | 0 | 24 / 288 | correctness gate failed |

The scheduler bucketing changes were removed. The final implementation does
not change batching, prompt chunks, tensor shapes, KV lengths, or model math.

## Final P4/D64 result

Environment: RTX 3080 10 GB, Cosmos-Reason2-2B FP16 indexed-paged engine,
independent contexts, 64 slots, fixed-128 prefill, 288 requests, requested
outputs 96 to 384 tokens.

The reference is the immediately-consecutive-only CUDA graph policy. The final
run uses the exact same materialized trace.

| Metric | Consecutive-only | Recurring-shape P4/D64 | Change |
| --- | ---: | ---: | ---: |
| Generated token/s | 4396.34 | 4399.60 | +0.074% |
| TTFT median | 4456.72 ms | 4457.39 ms | +0.015% |
| TTFT p95 | 9438.83 ms | 9427.34 ms | -0.122% |
| TPOT p95 | 13.180 ms | 13.138 ms | -0.324% |
| E2E p95 | 11959.69 ms | 11950.96 ms | -0.073% |
| End free GPU memory | 65.4 MiB | 51.4 MiB | -14.0 MiB |
| Output mismatches | 0 | 0 / 288 | pass |

The throughput and latency differences are within run-to-run noise. The useful
change is cache coverage:

| Context | Consecutive-only replay | Recurring-shape replay | Final cache |
| --- | ---: | ---: | ---: |
| Prefill | 10 / 285 (3.5%) | 59 / 285 (20.7%) | 4 graphs |
| Decode | 1104 / 1209 (91.3%) | 1133 / 1209 (93.7%) | 60 graphs |

The final run had no capture failure or graph-launch failure. Prefill evicted
26 old observation candidates from its bounded host table.

## Cache-limit sweep

Prefill limits 2, 4, and 8 were tested with decode fixed at 64:

| Prefill limit | Prefill graph launches | End free memory | Observation |
| ---: | ---: | ---: | --- |
| 2 | 10 | 59.4 MiB | no improvement over consecutive-only prefill |
| 4 | 59 | 51.4 MiB | recommended balance |
| 8 | 124 | 43.4 MiB | two decode capture failures; too little headroom |

The engine already consumes almost all 10 GB. A byte-budgeted graph cache or a
smaller engine/KV/context allocation is required before raising these limits.

## Code and artifacts

- `cpp/runtime/exec/engineExecutor.{h,cpp}`: recurring-shape capture, bounded
  candidate table, phase cache statistics.
- `examples/llm/llm_phase_bench.cpp`: independent prefill/decode graph limits.
- `scripts/cosmos_reason2/run_real_request_kv_matrix.py`: reproducible CLI
  forwarding and status fields.
- Final artifacts:
  `.local/cosmos-reason2-2b/cudagraph-prefill-frequency/final-p4/`.
- Rejected bucketing artifacts:
  `.local/cosmos-reason2-2b/cudagraph-prefill-buckets/`.

## Next step

The graph byte budget is implemented and measured in
[note 63](63-cuda-graph-byte-budget-20260811.md). Next, sweep the budget Pareto
frontier across P1/P2/P4/P8 and D16/D32/D64. Scheduler-level shape changes
should remain disabled unless they preserve exact greedy output and compensate
for their extra prefill turns.
