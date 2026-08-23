<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-27B Thor comparison

This experiment compares three implementations on one Jetson AGX Thor:

1. TensorRT Edge-LLM `v0.10.0` (`71dd1bae`)
2. the independent-phase branch built on that tag
3. stock vLLM with the official Qwen3.8 NVFP4 recipe

## Fairness contract

- Model family: `Qwen/Qwen3.8-27B`, text-only first
- Weight precision: NVFP4
- KV precision: FP8 when both runtimes support it; FP16 is a separate A/B
- Context cap: 32,768 tokens for the initial serving comparison
- Sampling: greedy, EOS ignored, fixed output length
- Prefix caching and speculative decoding: disabled for the vanilla comparison
- MTP: reported as a separate best-throughput comparison
- Hardware: one Thor GPU, MAXN, no unrelated CUDA workloads
- Metrics: output token/s, TTFT, TPOT/ITL, E2E, peak unified memory, and exact token count
- Repetitions: one warmup followed by at least three fresh process lifecycles in alternating order

The upstream and phase images use the same TensorRT 10.16, CUDA 13.2, compiler,
Python dependencies, and the SM110 CuTe DSL archive tracked by v0.10.0. This
avoids reusing the older v0.9.1 archive whose generated function ABI does not
match the v0.10 runner headers.

## Build images

Create the detached upstream worktree at the path used by the script, then run:

```bash
experiments/qwen38_thor/build_images.sh
```

The default output tags are:

- `tensorrt-edge-llm:qwen38-upstream-v010`
- `tensorrt-edge-llm:qwen38-phase`

The vLLM baseline starts from `nvcr.io/nvidia/vllm:26.06-py3`. Qwen's current
recipe uses an NVFP4 checkpoint, FP8 KV cache, and optionally three MTP draft
tokens. Validate model loading in the stock image before deciding whether a
newer source overlay is required.

The original checkpoint has 18 weight shards totaling 55,562,855,904 bytes.
Keep one local copy for Edge-LLM quantization/export; do not duplicate it per
runtime.

## Current phase-path gap

The current independent-phase coordinator requires a `packed_prefill` engine.
That export contract accepts only attention-only models with FP16 KV and head
dimension 128. Qwen3.8 has 48 GDN layers and attention head dimension 256, so
the phase path cannot serve it yet even though the upstream runtime can.

The implementation sequence is therefore:

1. establish upstream and vLLM vanilla baselines from the same Inferact NVFP4
   checkpoint;
2. add dense-prefill support to the independent coordinator;
3. add stable GDN recurrent/conv state ownership with phase-local active views;
4. keep decode-cohort state resident so steady decode does not gather and
   scatter every layer state on every token;
5. compare vanilla first, then report built-in MTP as a separate best-throughput
   ceiling.

## Initial Thor results

The shared Inferact checkpoint is 26,381,249,312 bytes across seven weight
files. It loads in both runtimes and produces the exact greedy smoke response
`THOR READY`.

The upstream engine uses B8, input 2,048, KV capacity 4,096, and 256 FP16 KV
pages. The phase candidate uses global B32, dense P1, D32, the same input/KV
limits, and 1,024 pages. Decode throughput below is aggregate batch throughput;
`llm_bench` prints `1 / latency` even for a multi-row batch.

| engine | phase | batch | E2E time | aggregate throughput |
| --- | --- | ---: | ---: | ---: |
| upstream v0.10 | prefill 2,048 | 1 | 770.9 ms | 2,656.6 tok/s |
| upstream v0.10 | prefill 2,048 | 8 | 12,592.5 ms | 162.6 tok/s |
| upstream v0.10 | decode, past 2,048 | 1 | 96.7 ms | 10.3 tok/s |
| upstream v0.10 | decode, past 2,048 | 8 | 109.9 ms | 72.8 tok/s |
| phase P1/D32 | prefill 2,048 | 1 | 594.1 ms | 3,447.0 tok/s |
| phase P1/D32 | decode, past 2,048 | 8 | 149.2 ms | 53.6 tok/s |
| phase P1/D32 | decode, past 2,048 | 16 | 174.3 ms | 91.8 tok/s |
| phase P1/D32 | decode, past 2,048 | 32 | 215.2 ms | 148.7 tok/s |

The first vLLM HTTP smoke uses 16 requests at concurrency 8 with 128 input and
64 fixed output tokens. It reaches 51.1 output tok/s, 812.8 ms median TTFT, and
144.2 ms median TPOT. NVFP4 uses the real
`FlashInferCutlassNvFp4LinearKernel`; the requested SM110 CuTeDSL GDN prefill
backend is rejected and falls back to Triton/FLA.

The asymmetric hybrid builder and runtime page-table profile validation now
accept the P1/D32 engine. The remaining serving blocker is stable ownership of
the 48 recurrent and conv state pairs; KV page remapping alone is insufficient.

## Development overlay

After changing the builder or runtime validation, rebuild only the affected
targets on top of the complete phase image:

```bash
docker build \
  -f experiments/qwen38_thor/Dockerfile.overlay \
  -t tensorrt-edge-llm:qwen38-phase-dev .
```
