---
id: W016
status: active
updated: 2026-10-01
notes: [371]
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
- Port in progress on branch `codex/v0110-phase-forward-port` (`.local/worktrees/v0110-port`, outside this branch's notes): merge commit `83f5f768` of v0.11.0 (`95515c2`) into fork tip `fca7bd0` (base v0.10.1 `e8b2952`), 47 conflicted files, plus fixes. Its notes 372 (resolution log) and 373 (milestone 1) report a build of all targets with `-DENABLE_CUTE_DSL=ALL` on SM86 and 0 failures in the C++ unit-test executables; no engines were exported or built from that tree at milestone 1. Milestone 2 is open: `.local/results/v0110-port-smoke-20261001` (diagnostic) completed 0/6 cells, and note 372 records phase KV active-view / ragged-ABI rewiring as not done. No serving result, so no port performance claim.

## Conclusions
- 371 — Upstream v0.11.0 baseline on SM86 (batch 4 Gemma / 8 Cosmos), silent INT4 V2 failure traced to missing `ENABLE_CUTE_DSL=ALL`, upstream serving baseline 5.28x slower than the fork tip, and port requirements. Revised in place in three later commits on 2026-09-30.

## Open questions
- Port the KV undercommit, phase-shaped Gemma4 PLE buffers and external INT4 FFN weights onto v0.11.0 and re-measure batch limits (note 371).
- Choose V1 versus V2 INT4 for the port, in light of the GEMV/GEMM numerics → W013 (note 368).
- Exercise the published server wheel at larger batch (the 24-cell baseline ran only at Gemma 4 / Cosmos 8; note 371).
- Finish milestone 2 on `codex/v0110-phase-forward-port`: export -> build -> inference per AGENTS.md, then batch-invariance/MMLU gates and full24 x3 (branch note 373).
- Promotion of the current tip as the default serving binary is outside this line → W010.

## Artifacts
- `.local/worktrees/upstream-v0110` (present)
- `.local/builds/upstream-v0110` (present)
- `.local/artifacts/v0110-upstream` (present)
- `.local/results/v0110-upstream-serving-20260930` (present)
- `.local/cache/tools/export-v0110-venv` and `server-v0110-venv` (present)
- `.local/worktrees/v0110-port` (present; branch `codex/v0110-phase-forward-port`, merge committed, not in note 371)
- `.local/worktrees/upstream-v0110-debug` (not present)
