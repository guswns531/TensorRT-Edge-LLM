# Upstream v0.11.0 Clean Baseline on RTX 3080 (SM86)

## Outcome

Upstream v0.11.0 (`95515c2`) builds and exports cleanly, but on the 10 GiB RTX 3080 it cannot serve our Gemma4 E2B
AWQ contract at all and serves Cosmos-Reason2-2B only at batch 8 (our fork: 64). The memory and SM86 work in the
fork is therefore a port requirement, not an optimization to re-evaluate.

## Setup

- Worktree `.local/worktrees/upstream-v0110` (detached `v0.11.0`), build `.local/builds/upstream-v0110`
  (TensorRT 11.0 container, SM86, `-DCUTE_DSL_ARTIFACT_TAG=sm_86`; the configure also needs `--gpus all` for
  `CUDA_DRIVER_LIB`).
- Export venv `.local/cache/tools/export-v0110-venv` (torch 2.13.0+cu130, transformers 5.14.1, modelopt 0.45.0,
  onnx 1.19.0), created with the container's pip via `pip --python` because neither host nor container Python has
  `ensurepip`.
- ONNX and engines under `.local/artifacts/v0110-upstream/`; build and smoke logs alongside.
- Published server wheel `tensorrt-edgellm[server]==0.11.0` in `.local/cache/tools/server-v0110-venv`
  (system site packages for TensorRT 11.0 bindings); not yet exercised.

## Findings

1. **No KV-pool undercommit.** `llm_build` rejects `--maxKVPoolPages 192` for Gemma batch 24 x 2048: "must be zero or
   at least the minimum active pages (384)". The fork builds and serves that contract with 192 pages.
2. **Gemma fixed memory does not fit.** At runtime init the PLE table (4.7 GB) and a PLE output buffer sized
   `layers x maxBatch x maxSeq x pleHidden` live on the GPU next to the 2.1 GB engine and 0.8 GB embedding table.
   With the vision engine loaded, init fails with `cudaMalloc ... out of memory` at batch 24, 16, and 8.
3. **Gemma text path fails on SM86 at any batch.** Text-only (no vision engine) initializes at batch 8 and batch 1,
   then every decode CUDA-graph warmup fails in a plugin (`pluginV3Runner.cpp:252`, no plugin message) and each
   request returns `finish_reason: error`. Batch 1 rules out memory; the attention plugin selected FMHA backend 2.
   Not localized further.
4. **Cosmos fits only at batch 8.** Batch 32 and 16 fail init with out of memory; batch 8 initializes and generates
   correct text on the 4-request smoke. The fork serves Cosmos at prefill 8 / decode 64 with 256 pages.

## Implications for the port

- Carry the fork's KV undercommit, host-resident PLE / external INT4 FFN weights, and SM86 Gemma attention path;
  without them the ported runtime cannot run the retained Gemma contract on this GPU.
- A like-for-like upstream throughput baseline exists only for Cosmos at batch 8; Gemma has no runnable upstream
  baseline on this hardware.
