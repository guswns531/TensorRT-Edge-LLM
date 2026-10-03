---
id: W016
status: active
updated: 2026-10-03
notes: [371, 372, 373, 374]
---

# Upstream v0.11.0 baseline and port

## Goal
Establish what stock upstream v0.11.0 achieves on the RTX 3080 (SM86, 10 GiB) with the retained Gemma 4 E2B and Cosmos-Reason2-2B contracts, and port the fork's phase-serving runtime and memory work onto v0.11.0.

## Current state
- Stock v0.11.0 (`95515c2`) builds, exports and generates correct text for both models, but only at Gemma batch 4 and Cosmos batch 8 with the vision engine loaded; the fork serves batch 24 and prefill 8 / decode 64. Gemma batch 24/16/8 and Cosmos 64/32/16 fail runtime init with `cudaMalloc` out of memory (note 371).
- Note 371 was created in `57b05187` and revised in place on 2026-09-30: corrections in `54d7aa4c` (Gemma failure cause, batch limits) and `1fe437dd` (fork Gemma4 PLE mechanism), and the serving baseline added in `fca7bd04`. The state here is the corrected one: the Gemma failure is not a plugin failure at every batch; it was our build omitting `-DENABLE_CUTE_DSL=ALL`, so the `Int4GroupwiseGemmPluginV2` kernels were not compiled and every INT4 enqueue returned -1 silently. The PLE table (4.7 GB) is GPU-resident in the fork too; the fork's difference is phase-shaped PLE output buffers, not table placement.
- `llm_build` rejects `--maxKVPoolPages 192` at batch 24 x 2048 (minimum 384); the fork builds that contract (note 371).
- Upstream serving baseline (published `tensorrt-edgellm[server]==0.11.0`, in-flight batching, `EDGELLM_IGNORE_EOS=1`, one run per cell, all requests fixed-length): our tip full24 x3 median is 5.28x upstream geomean over 24 cells, frozen vLLM 4.16x; parity only on wave-drain and Cosmos multi-image (319 vs 314 tok/s). Loss drivers: batch size 4/8 vs 24/64, and in-flight joining only with equal `max_tokens` (note 371).
- Port requirements (note 371): fork KV undercommit, phase-shaped Gemma4 PLE output buffers, external INT4 FFN weights; build with `-DENABLE_CUTE_DSL=ALL` (or `fmha;int4_fp16_gemm`).
- Port on branch `codex/v0110-phase-forward-port` (`.local/worktrees/v0110-port`; notes 372-374 live on that branch): merge `83f5f768` of v0.11.0 into fork tip `fca7bd0`, ragged-ABI rewiring, and the W016 recovery fixes, regrouped into six commits on top of `f6c2f094`, HEAD `f0849333` with note 374.
- Port serving (full24 x3, binary `9f911c16`): faster than frozen vLLM in all 24 cells (0/72 runs below, geomean +27.9%); vs the v0.10.1 tip +0.7% overall, Gemma +2.7%, Cosmos -1.3% (note 374).
- Port output quality: MMLU zero-shot serving 51.27%, equal to the port batch-1 reference; above v0.10.1's 50.36% because the doubled Gemma `<bos>` is fixed (note 374).
- Recovery fixes (multi-slot KV page-table upload, one-copy ragged metadata, true packed prefill) recovered Gemma and Cosmos prefill-heavy cells but not the Cosmos decode-heavy gap of 1.7-2.7% to v0.10.1 (note 374).
- External Codex review found no defect on the validated phase-serving path; six port-introduced defects on other paths are fixed, four fork-inherited ones are left open by decision (note 374).

## Conclusions
- 371 — Upstream v0.11.0 baseline on SM86 (batch 4 Gemma / 8 Cosmos), silent INT4 V2 failure traced to missing `ENABLE_CUTE_DSL=ALL`, upstream serving baseline 5.28x slower than the fork tip, and port requirements. Revised in place in three later commits on 2026-09-30.
- 372 — Merge resolution log for the v0.11.0 port (branch `codex/v0110-phase-forward-port`).
- 373 — Port milestone 1: all targets build, C++ unit tests pass; no engines yet (same branch).
- 374 — Port recovery fixes and external review: all 24 cells faster than vLLM (+27.9%), +0.7% vs v0.10.1 with Cosmos -1.3%, MMLU parity, six review defects fixed (same branch).

## Open questions
- Cosmos decode-heavy cells remain 1.7-2.7% below the v0.10.1 tip; a same-day rerun of both binaries is needed to separate the version gap from day-to-day variance before profiling (note 374).
- The six review fixes after `9f911c16` were validated by tests and four smoke cells only; a full24 and MMLU rerun on the final tree would close that gap (note 374).
- Ordinary `llm_inference` on packed engines rejects rows above the chunk cap and multimodal input instead of chunking them (note 374).
- Choose V1 versus V2 INT4 for the port, in light of the GEMV/GEMM numerics -> W013 (note 368).
- Exercise the published server wheel at larger batch (the 24-cell baseline ran only at Gemma 4 / Cosmos 8; note 371).
- Promotion of the port as the default serving binary is outside this line -> W010.

## Artifacts
- `.local/worktrees/upstream-v0110` (present)
- `.local/builds/upstream-v0110` (present)
- `.local/artifacts/v0110-upstream` (present)
- `.local/results/v0110-upstream-serving-20260930` (present)
- `.local/cache/tools/export-v0110-venv` and `server-v0110-venv` (present)
- `.local/worktrees/v0110-port` (present; branch `codex/v0110-phase-forward-port`)
- `.local/artifacts/v0110-port` (present; port Gemma and Cosmos ONNX and engines)
- `.local/baselines/v0110-port-f6c2f094-20261001`, `.local/baselines/v0110-port-9f911c16-20261001` (present)
- `.local/results/v0110-port-full24-3x-20261001`, `v0110-port-fix-full24-3x-20261001`, `v0110-port-mmlu-20261001`, `v0110-port-fix-mmlu-20261001`, `v0110-port-codex-review-20261003` (present, diagnostic)
- `.local/worktrees/upstream-v0110-debug` (not present)
