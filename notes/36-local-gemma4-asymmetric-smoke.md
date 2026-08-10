# Gemma 4 E2B local asymmetric engine smoke

2026-08-05 GPU validation after moving model and generated artifacts under `.local/`.

## Artifacts

- Model: `google/gemma-4-E2B-it`, revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- Quantization: INT4-AWQ backbone, FP16 embedding/PLE/KV, 128 wikitext samples, seed 0
- Indexed ONNX: `.local/gemma4-e2b/onnx-indexed/llm`
- Main engine: `.local/gemma4-e2b/engine-indexed-slots32-p8-d32-i128`
  - `maxBatchSize=32`, `maxPrefillBatchSize=8`, `maxDecodeBatchSize=32`
  - `maxKVCacheCapacity=2048`, `indexed_kv_cache=true`
- Reduced-capacity workload engine: `.local/gemma4-e2b/engine-indexed-slots32-p8-d32-i512`
  - Same phase profiles, `maxKVCacheCapacity=512`

## GPU results (RTX 3080 10 GB)

The 2048-capacity engine loads and runs with one slot. Independent TensorRT
contexts share one CUDA context and completed the two-request smoke workload:

- independent concurrent makespan: 20.108 ms (sequential 24.123 ms, 1.20x)
- TTFT median: 18.205 ms; E2E median: 37.256 ms
- CUDA-event kernel samples are in `.local/gemma4-e2b/smoke-independent-p1-d1-kernels.csv`.

The same 2048-capacity engine with 32 physical slots cannot fit two independent
contexts on this 10 GB card; allocation fails with CUDA OOM. This is physical
KV storage (`32 x 2048`) plus the Gemma PLE table and two TensorRT workspaces,
not an engine/profile correctness failure.

The 512-capacity engine was used to exercise the requested high decode batch:

- workload: 32 requests, prompt 128, output 16, fixed prefill chunk 128
- topology: one CUDA context, independent prefill/decode TensorRT contexts
- dispatch histogram reached `decode=32` 11 times and `prefill=8` three times
- TTFT median/p95: 167.195 / 299.828 ms
- E2E median/p95: 510.940 / 522.331 ms
- achieved throughput: 58.252 requests/s, 932.039 tokens/s
- concurrent makespan: 77.565 ms vs sequential 90.070 ms (1.16x)
- kernel-group samples: `.local/gemma4-e2b/smoke-p8-d32-context-kernels.csv`

This confirms the asymmetric profile and independent-context path. The 512
engine is a workload-fit experiment; use the 2048 engine for the intended long
KV-capacity baseline after reducing physical slots or moving to a larger GPU.
