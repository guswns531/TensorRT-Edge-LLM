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
2. **Gemma fixed memory does not fit.** At runtime init the PLE table (4.7 GB, GPU-resident in the fork too) and a
   PLE output buffer sized `layers x maxBatch x maxSeq x pleHidden` live on the GPU next to the 2.1 GB engine and 0.8 GB embedding table.
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

## Upstream serving baseline (`v0110-upstream-serving-20260930`)

Stock published wheel `tensorrt-edgellm[server]==0.11.0`, `--enable-in-flight-batching`, largest batch that fits
(Gemma 4, Cosmos 8), `--max-queued-requests 256`, fixed output through the upstream runtime switch
`EDGELLM_IGNORE_EOS=1` on the server process (the HTTP schema forbids `ignore_eos`; verified: a one-word prompt runs
to exactly `max_tokens`). Client: the frozen vLLM trace client with `return_token_ids`/`ignore_eos` removed from the
payload and `/version`/`/metrics` optional (`.local/scratch/v0110-upstream-serve-20260930/`), same traces, warmup,
and in-flight limits as the frozen vLLM contract (Gemma 24 / warmup 8, Cosmos 64 / warmup 64). One run per cell;
every request returned 200 at exactly its requested length. Seven Cosmos vision cells were rerun after a first pass
rejected their absolute host image paths (media root allowed only under `/workspace`). "Ours" is the tip full24 x3
median (note 369); vLLM is the frozen anchor.

| Workload | Upstream tok/s | Ours tok/s | Ours / upstream | vLLM / upstream | Upstream TTFT mean | Fixed-length requests |
|---|---:|---:|---:|---:|---:|---:|
| Cosmos balanced | 150.6 | 4446.7 | 29.5x | 28.7x | 32127 ms | 288/288 |
| Cosmos bimodal | 144.7 | 2018.6 | 14.0x | 12.9x | 59157 ms | 288/288 |
| Cosmos decode-heavy | 156.4 | 5238.5 | 33.5x | 31.6x | 92234 ms | 288/288 |
| Cosmos late-vision | 999.2 | 2517.9 | 2.5x | 2.2x | 2212 ms | 32/32 |
| Cosmos long-prefill | 142.4 | 1315.9 | 9.2x | 7.9x | 33967 ms | 288/288 |
| Cosmos mixed | 293.2 | 1181.0 | 4.0x | 3.1x | 6026 ms | 64/64 |
| Cosmos multi-image | 319.1 | 313.6 | 1.0x | 0.8x | 256 ms | 5/5 |
| Cosmos poisson | 153.6 | 2047.2 | 13.3x | 11.6x | 14152 ms | 64/64 |
| Cosmos short | 197.6 | 2392.1 | 12.1x | 10.4x | 2454 ms | 48/48 |
| Cosmos text-heavy | 293.2 | 2028.7 | 6.9x | 4.4x | 5918 ms | 64/64 |
| Cosmos vision-heavy | 425.2 | 732.6 | 1.7x | 1.4x | 3501 ms | 64/64 |
| Cosmos wave-drain | 99.0 | 98.0 | 1.0x | 1.0x | 199 ms | 20/20 |
| Gemma balanced | 122.4 | 1262.6 | 10.3x | 6.3x | 12881 ms | 64/64 |
| Gemma bimodal | 105.9 | 857.9 | 8.1x | 5.7x | 26587 ms | 64/64 |
| Gemma decode-heavy | 125.7 | 1376.8 | 10.9x | 6.5x | 37350 ms | 64/64 |
| Gemma late-vision | 425.8 | 1528.5 | 3.6x | 2.3x | 5345 ms | 32/32 |
| Gemma long-prefill | 98.6 | 616.9 | 6.3x | 5.1x | 16286 ms | 64/64 |
| Gemma mixed | 195.9 | 772.5 | 3.9x | 3.6x | 4647 ms | 64/64 |
| Gemma multi-image | 249.4 | 411.7 | 1.7x | 1.5x | 889 ms | 20/20 |
| Gemma poisson | 116.7 | 932.1 | 8.0x | 5.8x | 11364 ms | 64/64 |
| Gemma short | 119.2 | 845.8 | 7.1x | 4.8x | 2955 ms | 48/48 |
| Gemma text-heavy | 212.1 | 898.1 | 4.2x | 1.9x | 4509 ms | 64/64 |
| Gemma vision-heavy | 209.4 | 573.2 | 2.7x | 2.7x | 3565 ms | 64/64 |
| Gemma wave-drain | 93.1 | 97.2 | 1.0x | 1.0x | 285 ms | 20/20 |


Geomean over 24 cells: ours 5.28x upstream, frozen vLLM 4.16x upstream.

Upstream loses mainly through batch size (4/8 versus 24/64) and its in-flight admission rule: a request joins a
running batch only with equal `max_tokens`, so heterogeneous traces stall (`stalls_incompatible` reached 164k on
Cosmos decode-heavy) and TTFT means reach tens of seconds. It matches us only where batching does not matter:
wave-drain (both) and Cosmos multi-image (319 vs 314 tok/s, ahead of vLLM's 0.8x).

## Implications for the port

- Carry the fork's KV undercommit, phase-shaped Gemma4 PLE output buffers (`PhaseServingRuntime` builds one prefill
  preprocessor sized to a packed chunk and one decode preprocessor sized to batch x 1, sharing one GPU PLE table,
  instead of one `layers x maxBatch x maxSeq` buffer), and external INT4 FFN weights; without them the ported runtime
  cannot run the retained Gemma contract (batch 24) on this GPU. The PLE table itself is GPU-resident in both.
- Build the port with `-DENABLE_CUTE_DSL=ALL` (or at least `fmha;int4_fp16_gemm`) if its exporter emits the V2 INT4
  plugin; decide whether to keep V1 INT4 (note 368 numerics) or adopt V2.
- Upstream fixed-output serving runs are possible through `EDGELLM_IGNORE_EOS=1` on the server process.
