---
id: W009
status: done
updated: 2026-10-01
notes: [312, 313, 314, 315, 316, 321, 323, 326, 332, 334, 335, 336, 337]
---

# Workspace memory modes and owner-aware runtime

## Goal
Reduce Gemma 4 / Cosmos serving memory (resident vision E substrate, phase workspaces, duplicated KV) without workload-specific rules, and convert the freed memory into throughput against frozen vLLM.

## Current state
- Gemma 4 E2B current-vs-vLLM memory gap is ~608-626 MiB (VLM) / 828-892 MiB (text), dominated by the separately resident 784 MiB vision E substrate; independent P/D costs only 38 MiB (note 312).
- Workspace modes (independent / tiered_ep / shared_ep / shared_ed / auto) exist (note 314). Tiered E3/E4 was not promoted (notes 313, 334). Two-slab, tiered and P4/D8 follow-ups were not promoted; final ABBA shows no clean 3% non-regression, and Cosmos independent costs +440-556 MiB with resident-text decode regression (note 334).
- Early single-screen claims (315, 316 KV192 +32.6%/+33.1%, 323 "12/12", 326 Gemma 9/12 and Cosmos 11/12 wins) are not citable: 323-329 were corrected by note 331 (frozen-vLLM Cosmos means recomputed, single-repeat Full24 not statistics). Corrected Full24 x3 under note 331: Gemma wins 10/12, Cosmos 6/12 tok/s; the "12/12" claim is unsupported.
- PipelineIOPool (321) was removed in 326 for P/D contention; unpooled shared_ep is canonical (note 326).
- Owner-aware runtime (note 336): unique physical KV owners cut Gemma KV 1008 -> 432 MiB (peak -576 MiB; 15 of 35 layers are physical owners, 20 borrow donor KV, notes 332/336), grow-only image scratch (~72 MiB), shadow-only resident-decode observation. 48/48 cells ran; Gemma shared peak ~9210 MiB; Gemma independent full12 became runnable.
- Throughput-first follow-up (note 337, corrects 336's "not all beat vLLM"): `40536eb` x3 (72/72) wins tok/s on 24/24 medians, +25.83% geomean; `9db3ed3` full24 x1 wins 24/24 (Gemma +35.15%, Cosmos +16.37%); probe-off rejected. Continued in W010 (confirmation, default promotion).

## Conclusions
- 312 — Memory excess vs vLLM is the 784 MiB resident E substrate; merging P/D is the wrong first target.
- 313 — Tiered E3/E4 cuts fresh ready memory 9445 -> 9375 MiB; not promoted (ran without lifetime admission).
- 314 — Added phase workspace modes and headroom override; shared_ep reclaims 175 MiB (design/verification plan).
- 315 — Single mixed-trace screen: shared_ep Gemma -0.4%, Cosmos +2.5% tok/s (single screen; superseded by 334).
- 316 — Gemma KV 192 pages via shared_ep: bimodal +32.6%, long-prefill +33.1%, peak 9825 MiB (screen only; later KV owners made this moot, see 336).
- 321 — PipelineIOPool and non-owning reshape (superseded by 326, removed).
- 323 — Memory-transfer optimization, headroom 96 -> 48 MiB; "12/12" claim (corrected by 331).
- 326 — Unified unpooled shared_ep, multimodal eager dispatch, 50 ms decode-burst grace; win counts (corrected by 331).
- 332 — CPU audit: donor-KV physical owners (576 MiB) and demand-based image scratch (~72 MiB) are unimplemented savings.
- 334 — Workspace-mode final screen: 18/24 cells (Gemma independent 6 startup OOM); no clean non-regression; two-slab/tiered not promoted.
- 335 — Plan for owner-aware runtime: KV owner, scratch, admission vs resident-decode protection.
- 336 — Implemented KV owner/scratch/shadow observation; 48/48 cells (partly corrected by 337).
- 337 — Throughput-first common path: 40536eb x3 24/24 wins; 9db3ed3 x1 24/24 wins.

## Open questions
- Gemma exact-output identity across modes/batch shapes -> W013 (INT4 batch-shape cause, note 368).
- Resident-aware active scheduling and phase-wise completion retirement -> W014 (scheduler redesign).
- Promotion/default selection of 9db3ed3 and its repeats -> W010.
- Gemma multi-image stable margin (+0.08% in note 337) was not re-isolated here; later full24 x3 (note 369) shows +7.9% vs vLLM, so effectively closed.

## Artifacts
- `.local/results/memory-attribution-20260915` (present)
- `.local/results/gemma4-tiered-vision-engine-20260915` (present)
- `.local/results/owner-aware-runtime-20260926` (present)
- `.local/baselines/owner-kv-5a5e3ca-20260926`, `.local/baselines/owner-demand-34a8967-20260926` (present)
- `.local/artifacts/v0101-forward-port/workspace-corrected-20260926` (present)
- `.local/results/runtime-contract-revalidation-20260926` (present)
- `.local/results/workspace-mode-screen-20260920`, `.local/results/gemma-kv192-screen-20260920`, `.local/results/pipeline-io-pooling-screen-20260920` (deleted; note 370)
