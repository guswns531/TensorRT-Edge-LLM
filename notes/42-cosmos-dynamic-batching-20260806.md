# Dynamic batching experiments: Cosmos FP16 and phase scheduler

Date: 2026-08-06
GPU: RTX 3080 10 GB, Docker TensorRT 26.06, CUDA 13.3
Model: `nvidia/Cosmos-Reason2-2B`, FP16, no quantization

## Direct Cosmos FP16 text batching

The same text-only workload was run with 64 generated tokens per request. Each engine used `maxInputLen=1024` and `maxKVCacheCapacity=2048`; the batch profile was symmetric because the current public `llm_inference` path pre-fills the whole request batch before decoding it.

| Batch | Prefill ms | Decode tok/s (aggregate) | Decode ms/token | Peak VRAM |
|---:|---:|---:|---:|---:|
| 1 | 10.116 | 172.7 | 5.789 | 5.11 GB |
| 2 | 14.434 | 347.9 | 2.874 | 5.72 GB |
| 4 | 16.210 | 685.9 | 1.458 | 6.23 GB |
| 8 | 19.356 | 1364.5 | 0.733 | 7.39 GB |

All requests completed successfully. BS8 leaves approximately 2.5 GB on the 10 GB GPU. These are aggregate decode rates across all active requests, not per-request rates.

Artifacts are under `.local/cosmos-reason2-2b/`: `text-bs{1,2,4,8}-long-{output,profile}.json` and the corresponding engine directories.

## Independent prefill/decode phase load

Cosmos is a Qwen3-VL/deepstack model, and `llm_phase_bench` v1 intentionally rejects deepstack inputs. To measure genuinely asymmetric phase batches, the existing indexed Gemma engine (`maxPrefillBatchSize=8`, `maxDecodeBatchSize=32`, 32 stable slots) was used. This measures the scheduler/stream behavior separately from the Cosmos model arithmetic.

Workload: 32 generated requests, arrival rate 60 req/s, prompt length uniformly 64–128 tokens, output length uniformly 64–96 tokens, fixed 128-token prefill chunk, independent TensorRT contexts, shared CUDA context, and CUDA-event kernel-group recording.

| Prefill / decode limit | TTFT median / p95 (ms) | E2E median / p95 (ms) | achieved tok/s |
|---|---:|---:|---:|
| 1 / 4 | 75.1 / 116.4 | 4387.3 / 4607.8 | 535.6 |
| 2 / 8 | 180.0 / 437.0 | 4388.1 / 4663.1 | 521.8 |
| 4 / 16 | 179.8 / 442.5 | 2301.5 / 2484.2 | 895.1 |
| 8 / 24 | 182.1 / 452.5 | 1741.7 / 1911.5 | 1106.1 |

The 8/24 case reached decode batch 24 in 67 dispatches. Decode batch 4/8/16/24 engine-group medians under overlap were approximately 9.23/17.35/17.56/17.93 ms; corresponding prefill engine medians were 19.44/17.83/17.75/17.92 ms. CUDA-event overlap fractions were 0.31/0.42/0.43/0.40.

Interpretation: a small prefill limit keeps prompt admission responsive, but a small decode limit leaves requests decoding for several seconds. Increasing decode capacity produces a large E2E and throughput improvement, while TTFT p95 rises because larger decode batches compete with new prefills. This supports a policy that keeps prefill smaller than decode and chooses the pair using TTFT/TPOT targets rather than maximizing either batch independently.

Kernel-group and request-level CSVs are in `.local/cosmos-reason2-2b/dynamic-batching/phase-load/`, for example `p8-d24-n32-kernel.csv`, `p8-d24-n32-load.csv`, and `p8-d24-n32-load-dispatch.csv`.
