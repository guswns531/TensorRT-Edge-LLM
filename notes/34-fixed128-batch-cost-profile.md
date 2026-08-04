# Fixed-128 phase batch cost profile

## Purpose

This experiment removes adaptive chunk selection from the calibration path. Every
prefill dispatch contains at most 128 tokens per request. The resulting CUDA-event
tables are intended to be inputs to a later dynamic-batching policy, not a policy by
themselves.

The profile separates three effects:

1. Prefill kernel-group cost as the actual prefill batch grows.
2. Decode kernel-group cost as the actual decode batch grows and KV context changes.
3. Resource contention and request-tail behavior when independent TensorRT prefill
   and decode contexts run concurrently in one shared CUDA context.

## Engine and memory setup

- Model: Gemma 4 E2B text decoder, INT4 backbone and FP16 KV cache.
- GPU: RTX 3080 10 GB (SM86).
- Engine limits: `maxBatchSize=16`, `maxInputLength=128`,
  `maxKVCacheCapacity=2048`, indexed KV cache enabled.
- Runtime: one CUDA context, independent prefill/decode TensorRT execution contexts,
  independent non-blocking streams, and one 16-slot stable indexed KV pool.
- Fixed prefill chunk: 128 tokens. A 512-token prompt is four 128-token prefill
  dispatches; the engine never receives a 512-token prefill shape.

Gemma 4 E2B's FP16 KV allocation is approximately 84 MiB per physical slot, so the
16-slot pool reserves approximately 1.31 GiB. TensorRT reports 358 MiB of prefill
workspace and 66.1 MiB of decode workspace for the I128 engine. A BS16/I1024 engine
was also built, but its 2.8 GiB prefill workspace made two-context runtime startup
fail on the 10 GB card. The I128 calibration engine starts and completes P16/D16
workloads.

The phase batch caps are not added together when validating a continuous workload.
They are queue limits, not eagerly allocated slots. Admission still enforces at most
16 live requests. The legacy fixed-shape microbenchmark continues to require
disjoint P and D slots. Its warmup now reuses slot IDs sequentially when the two caps
sum to more than the pool size.

## Workloads

- Isolated prefill: 48 burst requests, prompt 128, output 1, P cap
  1/2/4/8/16.
- Isolated decode: 24 burst requests, prompt 128, output 24, D cap
  1/2/4/8/16.
- Concurrent combinations: 48 burst requests, prompt 512, output 16, caps
  P/D = 1/15, 2/14, 4/12, 6/10, 8/8, 10/6, 12/4, 14/2, 15/1.

The concurrent cases set the overlap token budget to `P cap * 128`. This does not
change the fixed chunk size; it permits the requested multi-row prefill batch to run
at the same time as decode. Earlier runs with a 128-token overlap budget only
overlapped P1 and therefore measured queue-cap partitioning rather than the intended
Pn/Dm GPU contention.

## Isolated phase cost

The table uses the dominant full-batch `*_engine` CUDA-event samples. Times are GPU
milliseconds.

| Phase | Actual BS | Median | p95 | Observation |
|---|---:|---:|---:|---|
| Prefill 128 | 1 | 18.285 | 18.329 | latency floor |
| Prefill 128 | 2 | 24.477 | 24.518 | 1.34x time for 2x requests |
| Prefill 128 | 4 | 41.625 | 41.807 | throughput still improves |
| Prefill 128 | 8 | 74.597 | 74.608 | near throughput saturation |
| Prefill 128 | 16 | 142.834 | 143.633 | little gain over BS8 |
| Decode | 1 | 6.274 | 6.348 | decode-only dispatch |
| Decode | 2 | 6.356 | 6.414 | almost free second row |
| Decode | 4 | 6.671 | 6.755 | good batching region |
| Decode | 8 | 13.916 | 13.951 | kernel/tactic cost step |
| Decode | 16 | 13.990 | 14.023 | nearly same time as BS8 |

Prefill request throughput rises from 54.44 req/s at BS1 to 104.44 at BS8, but only
to 108.59 at BS16. Decode request throughput rises from 6.20 req/s at BS1 to 24.78
at BS16. This suggests a first policy should normally cap prefill around 8 unless
prefill queue age is high, while decode should prefer either the 4-row low-latency
region or the 8-16-row high-throughput region.

## Concurrent request results

| P cap | D cap | req/s | TTFT p95 ms | TPOT p95 ms | E2E p95 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 10.40 | 4340 | 26.92 | 4555 |
| 2 | 14 | 14.23 | 3099 | 33.19 | 3323 |
| 4 | 12 | **15.40** | 2853 | 44.93 | **3070** |
| 6 | 10 | 14.64 | 2938 | 49.98 | 3231 |
| 8 | 8 | 14.13 | 2966 | 48.80 | 3356 |
| 10 | 6 | 15.15 | **2836** | 52.86 | 3124 |
| 12 | 4 | 13.69 | 3057 | 61.51 | 3460 |
| 14 | 2 | 10.10 | 3870 | 88.35 | 4700 |
| 15 | 1 | 6.52 | 5604 | 153.12 | 7303 |

P4/D12 maximizes throughput and has the best E2E p95. P10/D6 has the best TTFT in
this particular burst workload, but sacrifices decode tail latency. P2/D14 is a
useful decode-tail-biased point. The extremes starve one queue and are poor general
defaults.

An actual P4/D12 concurrent dispatch shows about 52.68 ms mean prefill-engine time,
37.36 ms mean decode-engine time, 53.17 ms mean makespan, and 0.411 mean overlap
ratio. Actual P8/D8 dispatches show 97.65/54.12/97.65 ms and 0.358 respectively.
These are contention costs; the scheduler must not predict concurrent service time
by simply adding or taking the maximum of isolated costs.

Requested caps and realized batches differ as slots change phase. For example,
P10/D6 produced P2/D6, P4/D6, P8/D6, and P10/D4 concurrent dispatches during this
trace. The cost tables therefore key records by actual P/D batch, not only scenario
name.

## Generated data

- `request-summary.csv`: request throughput and mean/median/p95 TTFT, TPOT, E2E.
- `kernel-group-summary.csv`: group cost by scenario and actual P/D batch.
- `kernel-cost-table.csv`: the detailed cost key, including 128-token chunk size and
  average decode context length.
- `dispatch-summary.csv`: queue waits, phase GPU time, makespan, and overlap ratio by
  actual P/D dispatch combination.

The committed aggregate files live under
`notes/results/gemma4-bs16-i128-fixed128-batch-cost/`. Raw per-dispatch and per-request
CSVs remain local and can be regenerated with:

```bash
python3 scripts/gemma4_e2b_indexed/run_phase_batch_cost_suite.py \
  --bench build/examples/llm/llm_phase_bench \
  --engine-dir /tmp/gemma4-e2b/engine-indexed-bs16-i128 \
  --output-dir notes/results/gemma4-bs16-i128-fixed128-batch-cost
```

## Using the table in a future scheduler

A first custom policy can enumerate feasible `(P batch, D batch)` candidates and
look up isolated and concurrent p95 costs. It should reject candidates that violate
the oldest request's TTFT/TPOT budget, then choose the remaining candidate with the
highest predicted completed-token utility. Missing table cells should fall back to
the nearest smaller measured batch, never an optimistic interpolation.

Before turning this into an online policy, repeat the suite for prompt distributions
128/512/1024 and decode context buckets around 128/512/1536. The current detailed
table records decode context, but this run's concurrent requests cluster around a
512-token prompt and only 16 output steps. Repeating three times and storing run ID,
GPU clocks, temperature, and p95 confidence intervals is also required before using
small differences such as P4/D12 versus P10/D6 as a stable production decision.
