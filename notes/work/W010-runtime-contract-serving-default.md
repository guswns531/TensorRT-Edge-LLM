---
id: W010
status: active
updated: 2026-10-01
notes: [330, 331, 333, 338, 339, 369]
---

# Runtime contract revalidation and serving-default promotion

## Goal
Repair observation/lifetime contracts, revalidate Gemma 4 and Cosmos on Full24 x3 (2 models x 12 workloads x 3 repeats) against frozen vLLM, and decide which binary becomes the serving default behind `.local/current/...`.

## Current state
- Runtime-contract fixes (note 331): P1 tail misclassified as decode, packed-decode plugin workspace, inflight cancel deferred release, graph invalidation on workspace swap, shared config resolution. Full24 x3 completed 72/72 (6,435 requests). Corrected tok/s wins vs vLLM: Gemma 10/12, Cosmos 6/12. Promotion withheld (Gemma exact output 561/632, ~21 MiB headroom). Note 331 corrects claims of 328-329 (including the 12/12 claim, single-repeat Full24 as statistics, and Cosmos frozen-vLLM means recomputed from 35 raw runs; baseline `.local/results/review-correction-20260926/cosmos-vllm-frozen-raw-corrected.json`).
- Candidate `5520216` (remove decode-only M-RoPE blocking of next encoder admission) raised Cosmos vision-heavy throughput ~82% but resident text E2E mean +345%; rejected and reverted, `43c680a` kept (note 333).
- `9db3ed3` + workspace-corrected engines re-run twice more: geomean +35.73% Gemma, +16.64% Cosmos vs frozen vLLM; 71/72 runs win (Cosmos wave-drain 92.68 tok/s loses -3.28%) (note 338).
- Serving default linked (note 339): `.local/current/active/runtime`, `.../gemma4/runtime`, `.../cosmos/runtime` all point to `.local/baselines/throughput-9db3ed3-20260926/bin` (verified now). Cleanup binary `d3a27e1` (-460 net lines) held throughput (+0.18%) but showed repeated TTFT signals, so not promoted. Highest portability risk: static D cost table and its clamp (note 339).
- Tip `863a6d4` (`.local/baselines/int4-force-gemm-863a6d4-20260929`) Full24 x3: 72/72 above vLLM, geomean +27.05% vs vLLM, +0.98% vs 9db3, -0.12% vs c7a1b71; Cosmos 1513/1513 and Gemma 589/632 repeat-exact (note 369). It is a promotion candidate; `.local/registry/current.json` records it as "candidate, awaiting approval" and the runtime pointers still target 9db3ed3.

## Conclusions
- 330 — Plan: fix observation contracts, shared config resolution, E/P exclusivity vs storage lease; frozen reference binary 949334c; seven-metric reporting.
- 331 — Contract/lifetime fixes and Full24 x3 (72/72); promotion withheld; corrects 323-329.
- 333 — M-RoPE/encoder-admission root cause; 5520216 rejected, 43c680a kept.
- 338 — 9db3ed3 two more full24 repeats; best throughput candidate, not a latency champion; dead-code audit.
- 339 — 9db3ed3 linked as serving default; d3a27e1 cleanup (-460 lines) not promoted; portability audit.
- 369 — Tip 863a6d4 full24 x3 promotion validation; candidate pending explicit approval.

## Open questions
- Explicit approval to repoint `.local/current/active/runtime` (and gemma4/cosmos `runtime`) to 863a6d4, update `.local/registry/current.json`, and mark `.local/results/tip-full24-3x-20260929` citable (note 369). Closes with that approval.
- Interleaved 9db/d3 TTFT control (note 339): superseded in practice by the tip comparison in 369; confirm whether still wanted.
- Static D prior/clamp portability and startup capability identity -> W011.
- Gemma output identity gate -> W013 (cause identified in note 368; 369 reports 589/632 repeat-exact).
- Cosmos wave-drain last-wave D cohort split (note 338) and Gemma multi-image margin: not re-investigated; 369 shows Cosmos wave-drain +2.2%, Gemma multi-image +7.9% vs vLLM.

## Artifacts
- `.local/current/active/runtime`, `.local/current/gemma4/runtime`, `.local/current/cosmos/runtime` -> 9db3ed3 (present)
- `.local/registry/current.json` (present; tip entry "candidate, awaiting approval")
- `.local/baselines/throughput-9db3ed3-20260926` (present)
- `.local/baselines/int4-force-gemm-863a6d4-20260929`, `.local/results/tip-full24-3x-20260929` (present)
- `.local/results/runtime-contract-revalidation-20260926`, `.local/results/throughput-frontier-20260926`, `.local/results/throughput-confirmation-20260926` (present)
- `.local/baselines/serving-cleanup-d3a27e1-20260926`, `.local/results/current-default-cleanup-20260926`, `.local/results/review-correction-20260926` (present)
