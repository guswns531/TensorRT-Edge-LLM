SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# CUDA graph budget Pareto sweep

## Recommendation

For the measured Cosmos-Reason2-2B FP16 indexed-paged engine on the RTX 3080,
the best current operating point is P4/D64 with a 16 MiB prefill graph budget
and a 128 MiB decode graph budget:

```text
--prefillBatch 4 --decodeBatch 64
--cudaGraph
--maxPrefillCudaGraphs 4 --maxDecodeCudaGraphs 64
--maxPrefillCudaGraphMiB 16 --maxDecodeCudaGraphMiB 128
--cudaGraphChargeMiB 4
```

It retains effectively all of the unbounded graph performance while recovering
124 MiB of device headroom. Compared with graph-off, it improves throughput and
tail latency while still leaving 175.4 MiB free after the run.

## Workload and coverage

- Same 288-request materialized real-request trace for every run.
- Requested outputs: 96 to 384 tokens; fixed-128 chunked prefill.
- 64 indexed-paged stable slots; independent prefill/decode TensorRT contexts.
- Prefill batch: 2/4/8; decode batch: 32/64.
- Graph modes: off, count-only unbounded, decode budgets 128/160/192/224 MiB.
- Prefill byte budget remained 16 MiB for all budgeted cases.
- 36 primary runs, followed by two extra repeats of P4/D64 graph-off,
  unbounded, and 128 MiB.
- All 42 processes completed successfully; no OOM or runtime failure.
- Exact comparison found zero mismatches across 8,640 primary graph-enabled
  outputs. All nine P4/D64 confirmation outputs also matched request by request.

## P4/D64 three-run confirmation

| Metric | Graph off | Unbounded graph | 128 MiB decode budget |
| --- | ---: | ---: | ---: |
| Generated token/s | 4346.54 | 4380.91 | 4378.94 |
| TTFT median | 4484.90 ms | 4480.61 ms | 4487.97 ms |
| TTFT p95 | 9496.45 ms | 9472.28 ms | 9480.30 ms |
| TPOT median | 12.425 ms | 12.371 ms | 12.383 ms |
| TPOT p95 | 13.320 ms | 13.196 ms | 13.200 ms |
| E2E median | 6972.26 ms | 6950.16 ms | 6958.31 ms |
| E2E p95 | 12056.26 ms | 12004.53 ms | 11996.58 ms |
| End free GPU memory | 319.4 MiB | 51.4 MiB | 175.4 MiB |
| Token/s coefficient of variation | 0.183% | 0.129% | 0.099% |

128 MiB versus graph-off:

- generated-token throughput: +0.745%;
- TTFT p95: -0.170%;
- TPOT p95: -0.895%;
- E2E p95: -0.495%;
- device headroom cost: 144 MiB.

128 MiB versus unbounded graph:

- generated-token throughput: -0.045%;
- TTFT p95: +0.085%;
- TPOT p95: +0.034%;
- E2E p95: -0.066%;
- device headroom gain: +124 MiB.

The latency differences between 128 MiB and unbounded are below one tenth of a
percent and smaller than normal run-to-run variation. The memory difference is
large and deterministic.

## Graph coverage at the recommended point

| Context | Cached graphs | Replays | Normal enqueues | Accounted bytes |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 4 | 59 | 226 | 16.0 MiB |
| Decode | 31 | 930 | 279 | 126.1 MiB |

The unbounded configuration cached 60 decode graphs, but the additional 29
graphs increased decode replays only from 930 to 1,133. Those extra replays
produced 0.045% more mean throughput at a cost of 124 MiB.

## Batch-size comparison

Single-run results below compare graph-off with count-only unbounded graph.

| P/D | Graph-off token/s | Unbounded token/s | Gain | Off free | Graph free |
| --- | ---: | ---: | ---: | ---: | ---: |
| P2/D32 | 3092.1 | 3180.0 | +2.84% | 413.4 MiB | 265.4 MiB |
| P2/D64 | 4208.5 | 4234.2 | +0.61% | 363.4 MiB | 83.4 MiB |
| P4/D32 | 3113.6 | 3213.1 | +3.20% | 369.4 MiB | 237.4 MiB |
| P4/D64 | 4346.8 | 4376.7 | +0.69% | 319.4 MiB | 51.4 MiB |
| P8/D32 | 3094.9 | 3190.2 | +3.08% | 283.4 MiB | 139.4 MiB |
| P8/D64 | 4316.1 | 4350.0 | +0.78% | 233.4 MiB | 45.4 MiB |

Interpretation:

- D64 is decisively better than D32 for this long-output, decode-heavy trace.
  P4/D64 produces about 36% more tokens/s than P4/D32 and also has much lower
  TTFT and TPOT tails.
- P4 is the best prefill cap. P2 leaves more memory but underuses the GPU; P8
  allocates larger phase I/O and slightly increases interference without
  improving throughput.
- CUDA graph gives a larger percentage gain with D32 because launch overhead is
  a larger fraction of smaller decode work. D64 is already GPU-work dominated,
  so its graph benefit is smaller but still repeatable.

## Decode budget sweep

Each cell is `generated token/s / end free MiB`.

| Budget | P2/D32 | P2/D64 | P4/D32 | P4/D64 | P8/D32 | P8/D64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 MiB | 3175/273 | 4257/223 | 3213/237 | 4376/175 | 3189/143 | 4357/93 |
| 160 MiB | 3175/265 | 4233/191 | 3213/237 | 4373/143 | 3191/139 | 4354/63 |
| 192 MiB | 3175/265 | 4233/159 | 3212/237 | 4371/111 | 3189/139 | 4346/45 |
| 224 MiB | 3174/265 | 4232/125 | 3213/237 | 4373/81 | 3190/139 | 4349/45 |

Larger budgets do not improve P4/D64 throughput in this trace. The most useful
decode shapes are admitted within 128 MiB; later graph variants have low
frequency and poor memory efficiency. D32 similarly saturates its useful
variant set before the higher budget points.

## Throughput/headroom Pareto frontier

| Operating point | Token/s | End free memory | Use case |
| --- | ---: | ---: | --- |
| P4/D64 unbounded | 4376.7 | 51.4 MiB | absolute single-run throughput |
| P4/D64, decode 128 MiB | 4376.4 | 175.4 MiB | recommended graph mode |
| P4/D64 graph-off | 4346.8 | 319.4 MiB | maximum P4/D64 safety margin |
| P2/D64 graph-off | 4208.5 | 363.4 MiB | more memory, lower throughput |

The 128 MiB point dominates all 160/192/224 MiB P4/D64 points: it is faster in
the measured run and leaves more memory. The unbounded point adds negligible
performance and should not be used on this 10 GB GPU.

Even graph-off P4/D64 leaves only 319.4 MiB after execution. A 512 MiB free
memory requirement is therefore impossible with the current engine, phase I/O,
context workspace, embedding, and KV configuration; graph tuning alone cannot
meet it.

## Artifacts

- Primary sweep:
  `.local/cosmos-reason2-2b/cudagraph-budget-pareto/{graph-off,unbounded,decode-*mib}/`.
- P4/D64 confirmation:
  `.local/cosmos-reason2-2b/cudagraph-budget-pareto/confirm/`.
- Every directory contains the materialized trace, request summary, dispatch
  cost table, kernel-group cost table, page-pressure model, status CSV, and
  per-case logs.

## Next step

Make 16/128 MiB the Cosmos benchmark preset, then implement a global free-memory
reserve. That policy should stop both context caches when device free memory
approaches a configured floor, even if a phase-local accounting budget remains.
