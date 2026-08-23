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
