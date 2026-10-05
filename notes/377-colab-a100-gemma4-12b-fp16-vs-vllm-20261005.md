# 377 — Colab A100: Gemma 4 12B FP16 phase serving vs vLLM (2026-10-05)

## Outcome
- First run of the v0.11.0 port on SM80 (A100-SXM4-40GB, driver 580.82.07 / CUDA 13.0, TensorRT 11.0.0.114 from the
  CUDA apt repo). C++ unit tests: 2196 passed, 58 skipped (NVFP4/Blackwell, FP8 XQA, multi-GPU), 0 failed after the
  flaky-test fix below; `test_attention_plugin.py` 182 passed, 13 skipped, 1 known upstream failure (same as SM86).
- Gemma-4-12B-it FP16 (no quantization) serves correctly only after two phase-runtime fixes (below). Against vLLM
  0.31.0 (bf16) on 10 of the 12 workloads, phase serving is slower: generated tok/s geomean ratio 0.854, range
  0.72 (short) to 0.97 (decode-heavy). wave-drain and late-vision were not run (campaign stopped by request).
- The gap is structural for this model, not a tuning residue: Gemma 4 unified uses vision-block (bidirectional image)
  attention, which the attention plugin supports only for entry-padded, history-free prefill. The engine therefore
  cannot use packed or chunked prefill; every prefill is a separate padded dispatch that re-reads 24 GB of weights,
  whereas vLLM folds prefill tokens into decode steps.

## Fixes made on this branch
| Area | Defect | Fix |
|---|---|---|
| `phaseServingRuntime.cpp`, `llm_phase_context_smoke.cpp`, `pipelineIO` | Phase prefill never filled `vision_block_ids`; the engine read uninitialized block IDs and treated text as bidirectional blocks (e.g. 17x23 answered 459) | `preparePrefillVisionBlockIds()` runs `generateVisionBlockIds` on the staged prefill IDs, mirroring `llmRankRuntime`; prefix-continued rows fail loudly |
| `phaseServingRuntime.cpp`, `phaseSchedulerOptions.inc` | `enableRaggedPrefillBatching` was tied to `packedPrefill`, so entry-padded engines only batched equal-length prefill rows (almost always batch 1) | Always enabled; entry-padded engines right-pad mixed lengths. Concurrent outputs are token-identical to sequential (9/9) |
| `phaseKVActiveViewTest.cpp` | Reads after async metadata uploads on non-blocking streams without a sync (14/20 failures on A100) | Stream sync before each D2H read; 100/100 passes |

## Engine contract (fits 40 GB)
- Export: non-packed (packed is rejected with vision-block attention). Gemma's sqrt(hidden) embedding scale forbids
  `--reuse-tied-lm-head`, so the 2 GB embedding table and LM head are both resident.
- `llm_build --maxInputLen 1600 --maxKVCacheCapacity 2048 --maxBatchSize 24 --maxPrefillBatchSize 4
  --maxDecodeBatchSize 24 --maxKVPoolPages 224 --allowKVPoolUndercommit`.
- Why: without chunked prefill the whole prompt (max 1543 tokens) must fit `maxInputLen`, and each of the three
  phase contexts reserves rows x maxInputLen activations (8 x 2048 was 2.6 GB per context and OOMed). Bounded SWA
  reserves 19 pages per slot (more than full mode's 16 at 2048), so full SWA mode with an undercommitted pool is
  the only fit: 224 pages = 28.7K tokens at about 336 KB/token (40 SWA layers x 8 KV heads x d256).
- Serving: V3 retained contract (`run_serving_comparison.py`), D24 / P4 / E4, in-flight 24. Batched vision prefill
  and vision prefix prefill are disabled for this engine (packed-only, and request-local block IDs respectively).
- Warmup: generic calibration trace with 256-token prefill rows (`calibration-p4-d24-e4-t256.json`); with 1024-token
  rows the measured cost table made the scheduler fragment short prompts into batch-1 prefill (short: 316 vs 368 tok/s).

## Results (generated tok/s, median of 3; TTFT/TPOT medians of per-run means, ms)
| Workload | Phase | vLLM | Ratio | TTFT phase / vLLM | TPOT phase / vLLM |
|---|---:|---:|---:|---:|---:|
| balanced | 446.6 | 508.3 | 0.88 | 608 / 559 | 42.3 / 35.8 |
| mixed | 336.5 | 344.8 | 0.98 | 649 / 903 | 51.6 / 40.8 |
| vision-heavy | 320.3 | 425.4 | 0.75 | 659 / 507 | 52.5 / 38.3 |
| multi-image | 273.6 | 350.2 | 0.78 | 575 / 376 | 47.1 / 34.3 |
| long-prefill | 302.0 | 342.4 | 0.88 | 2050 / 1052 | 52.2 / 53.1 |
| bimodal | 521.7 | 572.9 | 0.91 | 365 / 345 | 37.5 / 30.3 |
| decode-heavy | 652.9 | 670.5 | 0.97 | 371 / 333 | 27.6 / 27.0 |
| short | 366.4 | 506.9 | 0.72 | 309 / 267 | 46.3 / 28.3 |
| text-heavy | 385.4 | 458.7 | 0.84 | 807 / 737 | 45.4 / 36.1 |
| poisson | 406.4 | 474.2 | 0.86 | 580 / 345 | 41.4 / 37.0 |

Every cell completed all requests with the exact fixed output length. Decode alone is not the bottleneck: a
batch-24 decode step is 22 ms (weight-read floor about 16 ms); raw engine prefill is 40 ms at 64 tokens and 131 ms at
1024 tokens.

## Quality
Greedy, 48 tokens, 6 text + 3 image prompts vs HF bf16: phase FP16 matches 5/9 exactly (common-prefix fraction 0.77);
image answers stay correct but diverge after 10-25 tokens with synonyms. HF FP16 itself matches 5/9 and returns empty
answers on images (vision FP16 overflow). Open: the exporter passes `vision_block_ids` to all 48 layers, while HF
Gemma 4 applies the bidirectional overlay only on sliding layers (global layers stay causal); this may explain part of
the image divergence and needs an A/B.

## Comparison contract
- Same client (`openai_trace_client.py`), traces (`build_serving_workloads.py`, seed 20261005), warmup trace, in-flight
  24 and `ignore_eos` for both systems; a fresh server per cell.
- vLLM 0.31.0 (torch 2.13+cu132): bf16, max-model-len 2048, max-num-seqs 24, chunked prefill with 4096 batched tokens,
  async scheduling, prefix caching off, multimodal processor cache off.
- Phase is FP16 and vLLM bf16 (Gemma's native dtype); the dtype is not identical.

## Retained paths
- `.local/results/a100-gemma4-12b-fp16/full12-x3` (validation; 10 workloads x 3, `STOPPED.txt`)
- `.local/results/a100-gemma4-12b-fp16/quality` (diagnostic; HF/TRT greedy outputs)
- `.local/artifacts/colab-a100/gemma-4-12b-it/engine-b24-p4-in1600-kv2048-pool224`
