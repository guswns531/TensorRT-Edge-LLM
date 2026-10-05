# 378 — Colab A100: Cosmos-Reason2-2B and Gemma 4 E2B FP16 phase serving vs vLLM (2026-10-05)

Follows note 377 (Gemma 4 12B) on the same machine, toolchain and harness.

## Outcome
- Both 3080-era models run unquantized (FP16) on the A100 with D64 engines. Phase serving vs vLLM 0.31.0 (bf16),
  generated tok/s, one run per cell, geomean over 12 workloads:

| Model | All 12 | Text-only (5) | With vision (7) |
|---|---:|---:|---:|
| Cosmos-Reason2-2B (fresh TRT server per cell) | 1.030 | 0.927 | 1.110 |
| Gemma 4 E2B | 1.105 | 0.690 | 1.547 |

- Every cell completed all requests with the exact fixed output length.
- Cosmos had been answering differently from HF since export: the loader replaced the checkpoint's own `lm_head`
  with the embedding table (fixed in `f857bca`). After the fix, TRT FP16 matches HF bf16 on 7/9 greedy prompts,
  the same agreement HF fp16 has with HF bf16, and all three image answers match exactly.
- The gains are on vision traffic; text-only traffic is slower than vLLM, most for E2B (long-prefill 0.48).

## Fixes made for these runs
| Commit | Defect |
|---|---|
| `f857bca` | `load_weights` tied `lm_head` to `embed_tokens` whenever `tie_word_embeddings` was set, overwriting a distinct `lm_head.weight` in the checkpoint (Cosmos-Reason2-2B: max abs diff 0.031). Layer dumps matched HF at every decoder layer (cos >= 0.9997); only the head differed. Gemma 4 E2B and 12B ship no `lm_head` and are unaffected |
| `eaac6bd` | The shared phase prefill PipelineIO was sized for the primary profile only (8 x 256); a 4 x 1024 vision-prefill batch overflowed it |

The Cosmos engine also drops the separate vision-prefill profile. With a vision chunk (1024) larger than the text
chunk, the runtime disables chunked vision prefill (`allowChunkedVisionPrefill`), and a two-image prompt of 2771
tokens (woman_and_dog alone is 2752 tokens at native resolution) fits neither profile. A uniform 512-token chunk lets
images chunk like text, which is exact for Qwen3-VL's causal attention.

## Engine contracts
| | Cosmos-Reason2-2B | Gemma 4 E2B |
|---|---|---|
| Export | packed prefill, chunk cap 1024, FP16 | packed prefill, chunk cap 256, FP16, no audio |
| `llm_build` | input 1024, KV 8192, batch 80, P8 / D64, chunk 512, `--maxKVPoolPages 1536 --allowKVPoolUndercommit` | input 1024, KV 2048, batch 64, P8 / D64, chunk 256, full KV pool |
| Vision | `--maxImageTokens 22528 --maxImageTokensPerImage 2816`, encoder budget 22528 | `--maxImageTokens 1120` (4 x 280) |
| Serving | stable slots 80, in-flight 64, P8 / D64 / E4 | stable slots 64, in-flight 64, P8 / D64 / E4 |
| vLLM | max-model-len 8192, max-num-seqs 64 | max-model-len 2048, max-num-seqs 64, audio off |

Sizing: a Cosmos KV page (128 tokens, 28 layers x 8 KV heads x d128) is 14.7 MB, so 80 slots x 8192 tokens would
need 75 GB; 1536 pages (22.6 GB, 196K tokens) fit. Text traces use 288 requests for the four bulk workloads so D64
sees more than one wave (`build_serving_workloads.py --bulk-requests 288`).

## Results (generated tok/s, 1 run)
| Workload | Cosmos TRT | Cosmos vLLM | Ratio | E2B TRT | E2B vLLM | Ratio |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 3606 | 4190 | 0.86 | 2232 | 3350 | 0.67 |
| mixed | 811 | 754 | 1.08 | 1395 | 434 | 3.21 |
| vision-heavy | 569 | 523 | 1.09 | 1281 | 696 | 1.84 |
| multi-image | 402 | 333 | 1.21 | 601 | 382 | 1.57 |
| long-prefill | 2180 | 2341 | 0.93 | 998 | 2079 | 0.48 |
| bimodal | 4913 | 5184 | 0.95 | 3142 | 4119 | 0.76 |
| decode-heavy | 6187 | 6717 | 0.92 | 4285 | 5430 | 0.79 |
| short | 2812 | 2868 | 0.98 | 1882 | 2311 | 0.81 |
| text-heavy | 1517 | 1003 | 1.51 | 1693 | 591 | 2.87 |
| poisson | 1436 | 1414 | 1.02 | 1219 | 1385 | 0.88 |
| wave-drain | 102 | 101 | 1.01 | 100 | 99 | 1.02 |
| late-vision | 2670 | 2822 | 0.95 | 2024 | 2282 | 0.89 |

- E2B vision mixes: vLLM's mean TTFT is 2-4.7 s on mixed/text-heavy (TRT about 1 s), with its multimodal processor
  cache disabled. Where vLLM spends that time (preprocessing or encoder scheduling) was not profiled.
- E2B text: prefill dominates. In balanced, 146 of 302 prefill dispatches are batch 1, and a dispatch of 300 tokens
  or fewer costs about 17 ms of GPU time (10 tok/ms) against 41 tok/ms at 1500+ tokens. In long-prefill, prefill
  and decode GPU time sum to 18.5 s of a 26.7 s run, so the GPU idles at least 8.2 s. Decode steps at batch 64 take 8.6-11 ms. The fixed cost per small packed prefill and the
  formation of batch-1 prefill are the open items.

## Shared-server mode (`run_serving_comparison.py --reuse-server`)
One server and one calibration per system, then all 12 workloads in order, each after a 16-request unmeasured warmup
of its own trace. A full Cosmos campaign (both systems) took 336 s against roughly 70 min with a fresh server per cell.
TRT results match fresh-server cells within 5% on 11 workloads, but `short` fell to 901 tok/s (fresh: 2705 and 2922):
one request, sent right after decode-heavy, waited 1.05 s for its first token while the other 47 finished by 0.39 s.
Phase scheduler state carried over from the previous workload can starve a new prefill; this is a serving issue, not
only a measurement artifact. vLLM keeps little cross-request state, so its shared-server cells are used as is.

## Quality (greedy, 48 tokens, 6 text + 3 image prompts)
| Candidate | vs HF bf16 exact | Common prefix |
|---|---:|---:|
| Cosmos TRT FP16 (after `f857bca`), sequential and 9-way concurrent | 7/9 | 0.89 |
| Cosmos HF fp16 | 7/9 | 0.97 |
| E2B TRT FP16, sequential and 9-way concurrent | 6/9 (all text) | 0.70 |
| E2B HF fp16 | 9/9 | 1.00 |

E2B image answers diverge from HF after a few tokens and once claims only the dog is clearly visible; image
preprocessing (resize/normalization) parity with HF has not been checked.

## Retained paths
- `.local/results/a100-gemma4-e2b-fp16/full12-x1` (validation; `summary-table.txt`)
- `.local/results/a100-cosmos-reason2-2b-fp16/full12-x1-fresh-trt`, `full12-x1-reuse` (validation)
- `.local/results/a100-{cosmos-reason2-2b,gemma4-e2b}-fp16/quality` (diagnostic)
- `.local/scratch/debug-cosmos` (diagnostic; layer dumps that located the lm_head defect)
