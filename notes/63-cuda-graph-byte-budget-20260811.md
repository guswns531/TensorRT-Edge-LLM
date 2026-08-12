SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Independent CUDA graph byte budget

## Outcome

The independent prefill and decode TensorRT contexts now enforce separate CUDA
graph count and device-memory budgets. The cache tracks a conservative byte
charge for every graph and stops capture before the next graph is likely to
exceed the phase budget. A graph that exceeds the budget after capture is
destroyed, and the logical inference still executes exactly once.

The measured Cosmos P4/D64 balance on the RTX 3080 is:

```text
--cudaGraph
--maxPrefillCudaGraphs 4 --maxDecodeCudaGraphs 64
--maxPrefillCudaGraphMiB 16 --maxDecodeCudaGraphMiB 224
--cudaGraphChargeMiB 4
```

## Why measured delta needs a minimum charge

`cudaMemGetInfo()` is sampled immediately before and after graph capture and
instantiation. Some CUDA graph allocations are lazy: they become visible on
the first launch or when an internal CUDA pool grows. In the first 1 MiB
pressure experiment, several accepted graphs reported a zero immediate delta,
although end-of-run device use grew by 66 MiB.

The final accounting therefore uses:

```text
graph charge = max(capture-time free-memory delta,
                   configured minimum charge,
                   prior average graph charge when available)
```

The minimum is configurable because graph topology and TensorRT tactics vary.
Four MiB is the conservative value derived from this Cosmos engine; it is not
a universal CUDA constant.

## Enforcement flow

```text
recurring binding state
  |
  +-- count cache full --------------------------> enqueueV3
  |
  +-- remaining byte budget < estimated charge -> enqueueV3
  |
  +-- capture graph
        |
        +-- measured charge exceeds budget
        |     -> destroy graph
        |     -> mark this binding budget-rejected
        |     -> preserve the already executed request, or enqueue once
        |
        +-- within budget -> cache and replay
```

Consecutive capture has not executed the request yet, so a budget rejection
falls back to one normal enqueue. Non-consecutive capture happens after the
current normal enqueue, so rejection returns success without another enqueue.
This distinction prevents duplicate KV writes.

## Pressure validation

A 96-request P4/D32 trace used a deliberately impossible 1 MiB phase budget
with a 4 MiB minimum charge.

| Context | Captured graphs | Normal enqueue | Budget bypasses |
| --- | ---: | ---: | ---: |
| Prefill | 0 | 50 | 50 |
| Decode | 0 | 325 | 325 |

All 96 outputs exactly matched the existing reference. End free memory was
369.4 MiB, compared with 309.4 MiB in the measurement-only prototype that
incorrectly admitted zero-delta graphs. This validates capture-before-allocation
bypass and the normal-enqueue fallback.

## P4/D64 real-request result

Environment: RTX 3080 10 GB, Cosmos-Reason2-2B FP16 indexed-paged engine,
independent contexts, 64 slots, fixed-128 prefill, 288 requests, requested
outputs 96 to 384 tokens. Two runs used the same materialized trace.

| Metric | Unbounded mean | Byte-budget mean | Change |
| --- | ---: | ---: | ---: |
| Generated token/s | 4397.84 | 4388.55 | -0.211% |
| TTFT median | 4460.16 ms | 4470.75 ms | +0.237% |
| TTFT p95 | 9431.42 ms | 9445.59 ms | +0.150% |
| TPOT p95 | 13.144 ms | 13.154 ms | +0.077% |
| E2E p95 | 11956.06 ms | 11971.61 ms | +0.130% |
| End free GPU memory | 51.4 MiB | 81.4 MiB | +30.0 MiB |

Both budgeted runs matched all 288 reference outputs. Results were highly
repeatable: their durations differed by only 0.055 ms.

Final per-context state:

| Context | Graphs | Replays | Accounted bytes | Budget |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 4 | 59 / 285 | 16.0 MiB | 16 MiB |
| Decode | 53 | 1045 / 1209 | 220.1 MiB | 224 MiB |

The byte budget recovered 30 MiB of actual headroom for a 0.21% throughput
cost. The engine itself leaves only about 325 MiB before phase execution, so
the original 512 MiB headroom objective cannot be met through graph policy
alone; engine workspace, KV capacity, or model precision must also change.

## Telemetry and configuration

Each context reports:

- accounted graph bytes and configured byte budget;
- configured minimum per-graph charge;
- capture-time budget rejections;
- pre-capture budget bypasses;
- count-limit bypasses separately from byte-limit bypasses.

The Cosmos matrix runner forwards the common and phase-specific graph count,
MiB budget, and minimum-charge options and records them in `status.csv`.

## Code and artifacts

- `cpp/runtime/exec/engineExecutor.{h,cpp}`: byte accounting, preflight,
  post-capture rejection, destruction accounting, and statistics.
- `examples/llm/llm_phase_bench.cpp`: phase-specific MiB budgets and minimum
  charge CLI.
- `scripts/cosmos_reason2/run_real_request_kv_matrix.py`: experiment plumbing.
- Main artifacts:
  `.local/cosmos-reason2-2b/cudagraph-byte-budget/final-p4d64-16-224*/`.
- Pressure artifacts:
  `.local/cosmos-reason2-2b/cudagraph-byte-budget/tiny-1mib-final/`.

## Next step

The budget Pareto sweep is complete in
[note 64](64-cuda-graph-budget-pareto-sweep-20260811.md). The selected point is
P4/D64 with 16 MiB prefill and 128 MiB decode budgets. Next, expose a global
free-memory reserve that stops both context caches as device headroom approaches
a configured floor.
