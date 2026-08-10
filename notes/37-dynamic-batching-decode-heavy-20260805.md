# Dynamic batching: decode-heavy workload

The experiments use `.local/gemma4-e2b/engine-indexed-slots32-p8-d32-i512`,
which has `maxPrefillBatchSize=8`, `maxDecodeBatchSize=32`, fixed-128
prefill chunks, 32 physical slots, and KV capacity 512. Each run uses one
CUDA context and independent prefill/decode TensorRT contexts. CUDA-event
kernel-group data is emitted for every dispatch.

## Batch matrix

The completed matrix is under
`.local/gemma4-e2b/results/dynamic-batch-decode-heavy-20260805/`:

- Prefill limits P1/P2/P4/P8, prompt 128, output 1
- Decode limits D8/D16/D24/D32, prompt 128, output 32
- warmup 2, measured iterations 3

Request-level p95 results:

| case | TTFT p95 (ms) | TPOT p95 (ms) | E2E p95 (ms) |
|---|---:|---:|---:|
| P1 | 783.00 | 0.00 | 783.01 |
| P2 | 525.25 | 0.00 | 525.26 |
| P4 | 453.08 | 0.00 | 453.08 |
| P8 | 416.78 | 0.00 | 416.79 |
| D8 | 2662.81 | 65.47 | 4088.92 |
| D16 | 1808.01 | 37.67 | 2466.38 |
| D24 | 1515.82 | 28.09 | 2003.87 |
| D32 | 1402.77 | 24.32 | 1863.03 |

The longer decode run is `.local/gemma4-e2b/dynamic-decode64-*`:

- 32 requests, prompt 128, output 64, context-aware admission
- 59 full D32 dispatches, 67 decode dispatches total
- D32 `decode_engine` median/p95: 14.489 / 14.687 ms
- D32 `decode_prepare` median/p95: 0.018 / 0.020 ms
- D32 `decode_sample` median/p95: 0.161 / 0.166 ms
- TTFT median/p95: 167.140 / 300.068 ms
- E2E median/p95: 1220.166 / 1231.579 ms
- throughput: 25.425 requests/s, 1627.229 tokens/s
- independent-context makespan speedup: 1.156x

The scheduler's observed decode batch distribution was
`D1,D7,D9,D15,D17,D23,D25,D31,D32`, with mean 30.09 and maximum 32. This is
the workload to use for the first dynamic-batching policy table.

The overlap cases in the original suite were not included: they hard-code a
512-token prompt plus 16 output tokens, which exceeds the reduced engine's
512-token KV capacity. They require the 2048-capacity engine or a regenerated
overlap workload with prompt length <= 128.
