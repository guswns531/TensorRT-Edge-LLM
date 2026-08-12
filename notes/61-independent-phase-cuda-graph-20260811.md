SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Independent prefill/decode CUDA graph

## Outcome

The current phase runtime can now opt into a separate CUDA graph cache for the
independent prefill and decode TensorRT execution contexts. The two contexts
continue to share one CUDA primary context, engine weights, and indexed-paged KV
pool, but each graph is captured and replayed only on its owning phase stream.

The first Cosmos P4/D64 real-request measurements show a modest but repeatable
decode-heavy improvement. Across two paired 288-request runs, generated-token
throughput increased by 0.97% on average, TPOT p95 fell by 2.50%, and E2E p95
fell by 1.15%. All 576 paired outputs were identical to graph-off execution.

## Execution flow

```text
one CUDA primary context / one physical GPU
|
+-- prefill stream -- prefill TRT context -- prefill graph cache
|                    profile 0             key = all I/O addresses + shapes
|
+-- decode stream --- decode TRT context --- decode graph cache
                     profile 1              key = all I/O addresses + shapes

shared below both contexts:
  ICudaEngine + weights + embedding + indexed-paged KV page pool
```

For one binding state, execution progresses as follows:

```text
new address/shape state
  -> enqueueV3 once
  -> remember the successful binding snapshot

same state immediately repeated
  -> synchronize only that phase stream
  -> cudaStreamBeginCapture
  -> TensorRT enqueueV3 is recorded, not run
  -> instantiate graph
  -> launch graph once for the current request
  -> cache graph

later matching state
  -> cudaGraphLaunch

different or uncached state
  -> enqueueV3 fallback
```

The initial normal enqueue is required because TensorRT can defer work after an
optimization-profile or dynamic-shape change. Capture is attempted only when
the immediately preceding successful execution has the exact same complete
binding snapshot. This is deliberately more conservative than capturing every
first-seen batch size: it avoids running a live KV update twice.

## Code locations

- `cpp/runtime/exec/engineExecutor.{h,cpp}`
  - opt-in automatic capture
  - binding-snapshot validation
  - per-context graph cache limit
  - capture, graph-launch, enqueue, failure, and limit-bypass counters
  - graph-launch failure cleanup and safe enqueue fallback
- `cpp/common/trtUtils.cpp`
  - destroys partially created graph resources on capture/enqueue failure
- `examples/llm/llm_phase_bench.cpp`
  - `--cudaGraph` and `--maxCudaGraphs`
  - independent-context-only validation
  - separate prefill/decode graph statistics
  - before/after CUDA memory reporting
- `scripts/cosmos_reason2/run_real_request_kv_matrix.py`
  - `--cuda-graph` and `--max-cuda-graphs` forwarding for reproducible traces

The feature is opt-in and does not change the legacy `EngineExecutor::execute()`
path unless `enableAutomaticCudaGraphCapture()` is called. The phase benchmark
rejects CUDA graph mode with a shared TensorRT context.

## Fixed-shape P4/D32 result

Environment: RTX 3080 10 GB, Cosmos-Reason2-2B FP16, indexed-paged engine,
independent contexts, prefill 128 tokens, decode BS32, five warmups and 20
measured iterations.

| Metric | Graph off | Graph on | Change |
| --- | ---: | ---: | ---: |
| Sequential makespan median | 27.8395 ms | 27.0193 ms | -2.95% |
| Overlap makespan median | 24.4531 ms | 23.4680 ms | -4.03% |
| Overlap prefill median | 24.4132 ms | 23.4322 ms | -4.02% |
| Overlap decode median | 13.9581 ms | 13.4625 ms | -3.55% |
| End-of-run GPU memory | 9434.9 MiB | 9442.9 MiB | +8.0 MiB |

Each phase created one graph. After the first enqueue and capture-launch, each
context reported 51 graph launches and no capture or launch failures. The
measured graph-cache overhead for these two fixed variants was about 8 MiB.
The selected P16/D64/page-pool256 engine already leaves less than 512 MiB free
on this GPU; graph caching therefore needs a bounded variant count.

## Real-request P4/D64 result

Workload: 288 requests, 64 stable slots, fixed-128 chunked prefill, variable
prompts, requested outputs 96 to 384 tokens, greedy decoding. Each paired run
uses the exact same materialized arrival trace for graph off and on.

| Metric | Repeat 1 off / on | Repeat 2 off / on | Mean graph change |
| --- | ---: | ---: | ---: |
| Generated token/s | 4306.8 / 4359.5 | 4089.2 / 4118.5 | +0.97% |
| TTFT median | 4547.7 / 4487.7 ms | 4761.5 / 4818.8 ms | -0.06% |
| TTFT p95 | 9633.0 / 9500.3 ms | 10169.1 / 10067.6 ms | -1.19% |
| TPOT median | 12.622 / 12.380 ms | 13.376 / 13.177 ms | -1.70% |
| TPOT p95 | 13.485 / 13.272 ms | 14.479 / 13.984 ms | -2.50% |
| E2E median | 7067.7 / 6964.3 ms | 7444.2 / 7413.8 ms | -0.94% |
| E2E p95 | 12209.6 / 12048.2 ms | 12911.6 / 12786.5 ms | -1.15% |

Correctness comparison uses request ID, output-token count, finish reason, and
complete output text. Both repeats had zero mismatches.

The decode context cached 59 and 61 variants and launched graphs 1,104 times in
the two runs. It needed only 104-105 normal enqueues, so about 91% of its 1,208
engine executions used a graph. Prefill cached only one or two variants and
launched them 6-10 times because prompt remainder and initial/continuation
states create more shape diversity. The primary gain therefore comes from
decode CPU launch-overhead reduction, as expected.

## Cache pressure and fallback

A 96-request P4/D32 smoke with `--maxCudaGraphs 1` completed successfully. The
prefill context recorded 42 cache-limit bypasses and decode recorded 317; both
fell back to normal enqueue without a capture or graph-launch failure. This
confirms that the cache limit is a performance policy, not an admission or
correctness condition.

## Reproduction

Fixed-shape graph run:

```bash
./build/examples/llm/llm_phase_bench \
  --engineDir .local/cosmos-reason2-2b/sweep-engines/engine-fp16-paged-p16-d64-b256-m80 \
  --prefillBatch 4 --decodeBatch 32 --slotCount 64 \
  --inputLen 128 --prefillChunkSize 128 --pastKVLen 0 \
  --warmup 5 --iterations 20 --trtContextMode independent \
  --cudaGraph --maxCudaGraphs 128
```

Real-request matrix runner adds:

```bash
python3 scripts/cosmos_reason2/run_real_request_kv_matrix.py \
  ... --context-modes independent --cuda-graph --max-cuda-graphs 128
```

Artifacts are under `.local/cosmos-reason2-2b/cudagraph-real/` and the fixed
CSV files are `.local/cosmos-reason2-2b/cudagraph-p4d32-fixed-final-{off,on}.csv`.

## Remaining work

1. Run three or more process-level repeats for every P/D combination used by
   the scheduler cost table; the current real-request result is two paired runs
   plus one earlier direct pair.
2. Add a graph-memory budget, not only a variant-count limit. The number of
   graphs that fits depends on TensorRT tactics and graph topology.
3. Prefill shape bucketing was tested and rejected: it increased prefill turns,
   regressed throughput, and failed exact-output comparison. The safe recurring-
   shape capture policy and phase-local limits are documented in
   [note 62](62-prefill-cuda-graph-frequency-policy-20260811.md).
4. Move graph policy configuration from the benchmark wiring into the eventual
   production async server configuration surface.
5. Compare CUDA graph on/off against vLLM production again. This patch closes
   part of launch-overhead parity but does not change scheduling or kernel cost.
