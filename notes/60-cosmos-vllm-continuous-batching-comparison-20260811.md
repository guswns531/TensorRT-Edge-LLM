SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Cosmos vLLM continuous-batching comparison

## Conclusion

The direct comparison changes the interpretation of the earlier clean-upstream
results. vLLM production continuous batching is much faster than the current
independent TensorRT runtime for short and medium outputs. The gap closes as
decode becomes dominant: at output 12x the current P4/D64 runtime is 1.0% faster
in generated-token throughput, but vLLM still has better TTFT/E2E and uses about
1.4 GiB less GPU memory with the same raw FP16 KV capacity.

The current design therefore has a viable long-decode throughput point, but it
does not yet beat vLLM as a general-purpose online server. The most important
missing parity item is CUDA graph support in the independent-context path.

## Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3080, 10,240 MiB, SM86 |
| Driver | 610.43.02 |
| Model | `nvidia/Cosmos-Reason2-2B`, revision `9ce19a1...` |
| Weights | unquantized, cast to FP16 |
| KV | FP16 |
| vLLM image | `vllm/vllm-openai@sha256:c2f3b1...` |
| vLLM stack | vLLM 0.27.1, Transformers 5.15.0, PyTorch 2.13.0+cu130 |
| vLLM mode | language-model-only, max seqs 80, token budget 8192 |
| vLLM scheduler | chunked prefill on, prefix cache off |
| current engine | indexed-paged, maxBatch80, P16/D64 capable, page pool256 |
| current runtime | independent TensorRT contexts, fixed-128 prefill, P4/D64 |

Prefix caching is intentionally disabled. The 288-request workload repeats the
same twelve prompts 24 times, so enabling vLLM's default prefix caching would
remove prefill work only on the vLLM side.

The primary vLLM comparison uses its production compile and CUDA graph path.
The current phase runtime uses its existing graph-disabled real-request path.
This is a best-runtime end-to-end comparison, not a scheduler-only ablation.
The separate eager result below quantifies the graph/compile effect.

## End-to-end scaling

Each vLLM number is the median of three processes after a concurrent warmup.
N48/N288 output1x-4x current values are the existing single-run independent D32
measurements. The output8x/12x current values were rerun three times with P4/D64
for this comparison.

| Trace | Current tok/s | vLLM tok/s | vLLM throughput | Current E2E med/p95 | vLLM E2E med/p95 | vLLM E2E change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| N48 output1x | 820 | 1,992 | +142.8% | 934 / 1,223 ms | 440 / 501 ms | -52.9% / -59.1% |
| N48 output2x | 1,357 | 2,698 | +98.9% | 1,108 / 1,503 ms | 635 / 752 ms | -42.7% / -49.9% |
| N48 output4x | 1,963 | 3,100 | +57.9% | 1,414 / 1,893 ms | 904 / 1,255 ms | -36.1% / -33.7% |
| N288 output1x | 833 | 2,408 | +188.9% | 3,846 / 7,137 ms | 1,610 / 2,269 ms | -58.1% / -68.2% |
| N288 output2x | 1,769 | 3,399 | +92.1% | 3,917 / 6,549 ms | 2,199 / 3,289 ms | -43.9% / -49.8% |
| N288 output4x | 2,426 | 4,119 | +69.8% | 5,302 / 9,229 ms | 3,391 / 5,177 ms | -36.1% / -43.9% |
| N288 output8x | 4,260 | 4,400 | +3.3% | 5,654 / 9,059 ms | 4,993 / 8,423 ms | -11.7% / -7.0% |
| N288 output12x | **4,475** | 4,428 | **current +1.0%** | 6,990 / 11,783 ms | **6,506 / 11,599 ms** | -6.9% / -1.6% |

Generated throughput uses each runtime's actual completion token count. EOS
termination differs slightly across backends, so requested-token throughput is
not substituted. The request CSV preserves both values.

## Output8x/12x three-run details

### Output8x

| Runtime | Tok/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 |
| --- | ---: | ---: | ---: | ---: |
| current P4/D64 | 4,260 | 3,283 / 7,163 ms | **15.19 / 17.97 ms** | 5,654 / 9,059 ms |
| vLLM production | **4,400** | **2,937 / 6,518 ms** | 15.87 / **17.12 ms** | **4,993 / 8,423 ms** |

### Output12x

| Runtime | Tok/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 |
| --- | ---: | ---: | ---: | ---: |
| current P4/D64 | **4,475** | 4,055 / 9,028 ms | **14.81 / 16.94 ms** | 6,990 / 11,783 ms |
| vLLM production | 4,428 | **3,421 / 8,600 ms** | 15.49 / 16.94 ms | **6,506 / 11,599 ms** |

The current runtime's long-decode advantage is in TPOT and aggregate decode
throughput. vLLM admits and prefills the burst more efficiently, so its TTFT and
E2E remain lower even when current wins token throughput.

## CUDA graph ablation

On N48/output4x:

| Runtime | Generated tok/s | E2E median/p95 |
| --- | ---: | ---: |
| vLLM eager, graph/compile off | 1,255 | 2,071 / 3,140 ms |
| current independent D32 | 1,963 | 1,414 / 1,893 ms |
| vLLM production | 3,100 | 904 / 1,255 ms |

vLLM production is 147% faster than its eager path. This is larger than the
independent-context overlap gain previously measured inside the current runtime.
CUDA graph parity must therefore precede any strong scheduler-level claim.

## Memory-matched comparison

Cosmos FP16 KV consumes 112 KiB per token. Both memory-matched paths reserve:

```text
32,768 tokens * 112 KiB = 3,584 MiB
```

| Runtime | Raw KV | Observed process VRAM | Headroom |
| --- | ---: | ---: | ---: |
| current P4/D64 | 3,584 MiB | 9,588 MiB | 278 MiB |
| vLLM production | 3,584 MiB | 8,174 MiB | 1,692 MiB |

These are `nvidia-smi` process samples, not allocator peak telemetry. They still
show a 1,414 MiB structural difference. The current log reports prefill/decode
context workspaces of about 1,088/513 MiB in addition to engine and KV memory.
The memory-matched vLLM output12x run produced 4,433 tok/s, indistinguishable
from its 41,056-token automatic-pool result of 4,428 tok/s. Extra KV capacity
was not responsible for its performance.

## Fairness limitations

- vLLM includes localhost HTTP, JSON, and per-token streaming overhead. The
  current C++ harness injects the materialized trace directly into the runtime.
- vLLM production enables compile/CUDA graphs; current real-request graphs are
  disabled. The eager ablation exposes this difference rather than hiding it.
- Both paths execute text only. vLLM disables the visual module; current does
  not run the visual engine but still owns Cosmos deepstack decoder inputs.
- The same messages, arrival offsets, output limits, greedy settings, model
  checkpoint, FP16 weight dtype, and FP16 KV dtype are used.
- Exact tokens can diverge near EOS because the backends are numerically and
  tokenization-implementation distinct. Actual generated tokens are reported.

## Engineering implications

1. Add CUDA graph support to the independent P/D execution path and rerun this
   exact matrix before tuning more scheduler thresholds.
2. Reduce independent-context memory, preferably with phase-specific engine
   profiles/workspaces or explicit workspace sharing where TensorRT permits it.
3. Replace seed-bucket prefill selection with largest-compatible-bucket or
   token-budget batching. vLLM's short-output TTFT result shows that P7 observed
   batching plus fixed-128 chunks leaves substantial prefill efficiency unused.
4. Keep P4/D64 as a long-output candidate: output12x proves that its decode path
   can match vLLM throughput, but route short/medium traffic differently.
5. Add a scheduler decision based on expected remaining output and queue age;
   the crossover is between output4x and output8x in this trace family.

## Reproduction artifacts

```text
scripts/cosmos_reason2/manage_vllm_docker.sh
scripts/cosmos_reason2/run_vllm_trace_bench.py
.local/vllm-cosmos-reason2-2b/
  manifest.json
  cache/
  eager-n48-m4/
  production-n48-m1 ... production-n288-m12/
  memory-matched-n288-m12/
  current-repeat-n288-m8-run1 ... run3/
  current-repeat-n288-m12-run1 ... run3/
```
