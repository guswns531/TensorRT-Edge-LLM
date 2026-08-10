# Cosmos Reason2 2B FP16 VLM smoke and BS2 result

Date: 2026-08-06
Model: `nvidia/Cosmos-Reason2-2B`
Revision: `9ce19a195e423419c349abfc86fd07178b230561`
Checkpoint SHA256: `fa5a6e6ef4fce40216b185cc48a3b24d31637ac3e2ba69c107ed1f389c1e6ede`

The checkpoint was used without quantization. The decoder backbone, embedding, visual encoder, and KV cache are FP16. The ONNX export had dynamic `Reduce*` axes that TensorRT could not parse, so the generated LLM ONNX copy in `.local/cosmos-reason2-2b/onnx-fp16-llm-patched/` replaces those semantically-`[-1]` axes with one INT64 initializer. The visual export uses the optimized graph in `.local/cosmos-reason2-2b/onnx-fp16-visual-opt/`.

## Engines

| Engine | Profiles | Visual image-token profile |
|---|---|---:|
| `engine-fp16-b1-i1024-kv2048` | max batch 1, prefill 1, decode 1 | 512 |
| `engine-fp16-b2-i1024-kv2048` | max batch 2, prefill 2, decode 2 | 2048 |

Both were built in `nvcr.io/nvidia/tensorrt:26.06-py3` with the repo plugin and then executed in the same image with `--gpus all`. Container GPU check: RTX 3080, driver 610.43.02, 10240 MiB.

## Results

Single-image request (`woman_and_dog.jpeg`, 16 generated tokens):

- vision encoder: 27.38 ms for 486 image tokens
- LLM prefill: 18.284 ms for 504 computed tokens
- decode: 5.829 ms/token, 171.6 tokens/s
- peak GPU: 5114 MB
- output: `A woman sits on the beach, smiling as she shares a joyful high-five with`

Two-request image batch (`woman_and_dog.jpeg` + `red_panda.jpeg`, 12 tokens/request):

- vision encoder: 49.04 ms for 980 image tokens
- LLM prefill: 47.410 ms for 1017 computed tokens
- decode aggregate: 2.798 ms/token, 357.4 tokens/s
- decode GPU average in profile: 6.11 ms per decode run
- peak GPU: 5720 MB
- both responses completed successfully

The first BS2 attempt intentionally used `maxPrefillBatchSize=1`; runtime correctly attempted a batch-2 prefill and TensorRT rejected the shape. Rebuilding with a batch-2 prefill profile fixed the issue. This is a profile/configuration mismatch, not an OOM or model failure.

Generated JSON outputs and profiles are under `.local/cosmos-reason2-2b/` (`vlm-smoke-*` and `vlm-bs2-*`).
