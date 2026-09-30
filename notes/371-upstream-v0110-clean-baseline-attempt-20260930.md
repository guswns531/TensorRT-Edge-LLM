# Upstream v0.11.0 Clean Baseline on RTX 3080 (SM86)

## Outcome

Upstream v0.11.0 (`95515c2`) builds, exports, and generates correct text for both models on the 10 GiB RTX 3080,
but only at small batch: Gemma4 E2B AWQ at batch 4 and Cosmos-Reason2-2B at batch 8 with the vision engine loaded
(the fork serves batch 24 and prefill 8 / decode 64). The fork's memory work is therefore a port requirement.

Correction (same day): an earlier revision reported a Gemma plugin failure on SM86 at every batch. The cause was our
build configuration: v0.11.0 exports Gemma's INT4 projections as `Int4GroupwiseGemmPluginV2` (CuTe DSL kernels), but
`ENABLE_CUTE_DSL` defaults to `fmha`, so the INT4 group was not compiled and every INT4 enqueue returned -1 without a
message. Building with `-DENABLE_CUTE_DSL=ALL` and rebuilding the engines fixes it.

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
3. **Silent INT4 V2 failure without `-DENABLE_CUTE_DSL=ALL`.** Localized with a temporary instrumented plugin
   (`.local/worktrees/upstream-v0110-debug`): `CUTE_DSL_INT4_FP16_GEMM_ENABLED not compiled`. Neither the builder nor
   engine load reports the missing kernel group; the flag is documented only for the ONNX-less builder. Engines built
   against the incomplete plugin also fail in `onShapeChange` after the plugin is fixed and must be rebuilt.
4. **Batch limits with the corrected build** (`fit-status.txt`, vision engine loaded, 4-request smoke): Gemma 24, 16,
   and 8 fail init with out of memory, 4 runs; Cosmos 64, 32, and 16 fail init, 8 runs.
5. **Server contract.** The published `tensorrt-edgellm[server]==0.11.0` wheel serves Gemma at batch 4 with
   `--enable-in-flight-batching`, but its chat schema forbids extra fields, so `ignore_eos` and `return_token_ids`
   are rejected (HTTP 400), and it has no `/version` or `/metrics`. A fixed-output comparison with our traces is not
   possible against stock upstream; in-flight batching also joins only requests with equal `max_tokens`.

## Implications for the port

- Carry the fork's KV undercommit and host-resident PLE / external INT4 FFN weights; without them the ported
  runtime cannot run the retained Gemma contract (batch 24) on this GPU.
- Build the port with `-DENABLE_CUTE_DSL=ALL` (or at least `fmha;int4_fp16_gemm`) if its exporter emits the V2 INT4
  plugin; decide whether to keep V1 INT4 (note 368 numerics) or adopt V2.
- An upstream serving baseline must run with EOS allowed (both systems), since stock upstream cannot ignore EOS.
