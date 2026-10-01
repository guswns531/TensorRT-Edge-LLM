---
id: W006
status: waiting
updated: 2026-10-01
notes: [284, 285, 287, 288, 290, 291, 292, 293, 294, 296, 297, 298]
---

# Gemma 4 E2B port, packed prefill, and batch frontier

## Goal
Port Gemma 4 E2B AWQ to the v0.10.1 phase runtime (export -> build -> inference), establish its engine batch frontier, and implement packed/chunked prefill so the V3 policy can serve it competitively; also size the Cosmos batch frontier.

## Current state
- Gemma 4 E2B AWQ runs export -> build -> inference on RTX 3080 in legacy and E/P/D phase runtimes with exact output identity; it needs INT4 GEMM plugin v1, and dense non-packed prefill (head dim 256/512) until packed prefill landed (note 284).
- The visual engine had inherited a 512 soft-token default instead of the model's 280; after the fix mixed prompt tokens are 1193 (not 2101) (note 288, corrects 285's mixed numbers).
- Engine frontier: P4/D16/E2 (note 291) was extended to compiled P4/D32/E8 with active D24, 24 owners, tiered E/P arena once headroom gate was relaxed to 192 MiB (note 292). Limiting allocation is phase-context workspace, not KV (note 291).
- True packed/chunked prefill (d256/d512, sliding, shared-KV) is implemented; multi-chunk greedy output matches dense exactly; packed text P beats dense by 1-13% and saves 532 MiB (notes 293, 294).
- Chunk profiles: P128 stays default; P512 is opt-in (+12.8-13.9% text long-prefill, -3.5% VLM, +534 MiB) (note 297). One engine with P512 primary and P128 auxiliary keeps P128 performance but chunk 512 regresses multi-image 9.8% (D dispatches 199 -> 253, fill 3.12 -> 2.45; GPU interference not isolated) at +638 MiB text / +506 MiB VLM peak vs standalone P128, with VLM chunk-512 calibration unconverged; a composition-root capability error that disabled chunked vision prefill was fixed (note 298).
- Direct vision output binding removed all encoder-output D2D copies (24 -> 0) (note 296).
- vLLM comparison claims in 288-294 were superseded: the seq8 vLLM control understated vLLM by 65-161%; with the seq24/KV480/P4096/sparse-graph control, packed V3 won 7/12 (+7.86% geomean) (note 295, supersedes 292/294). Current vLLM comparison state is in W007.
- Cosmos frontier: P8/D64/E4 stays default; D80 rejected (decode-heavy -15.94%), P12 OOM, P16/D64 +4-5% mixed/vision-heavy but -9% long-prefill (note 287).
- Hash differences across cross-profile and repeated Gemma runs are INT4 batch-shape numerics, handled in W013.

## Conclusions
- 284 — Gemma 4 E2B AWQ validated through export/build/inference in legacy and phase runtimes; fixed PLE transfer and external-prefill gaps.
- 285 — P2/D4 engine: V0-V3 outputs identical; V3 +12.75% mixed throughput vs V0 (corrected by 288 for prompt-token inflation).
- 287 — Cosmos P8/D64/E4 stays default; D80 and P12 rejected, P16 shape-dependent.
- 288 — Soft-token default fixed (280 vs 512); V3 +22.4% warm / +33.7% cold over V0 on HTTP mixed.
- 290 — Heuristic audit separating capabilities from heuristics; one-run full12 vs optimized vLLM wins 9/12 throughput, loses vision-heavy, multi-image and wave-drain (-0.7%); TTFT mean wins 2/12 (control later superseded by 295).
- 291 — Frontier P4/D16/E2 (886.4 tok/s; D16 is the main gain, P2/D16 +60.0% over P2/D8); P4/D32 misses 512 MiB headroom (479); workspace is the limiter.
- 292 — P4/D32/E8, D24, 24 owners with headroom 192 MiB; claimed 2.21x geomean over vLLM (superseded by 295).
- 293 — Packed/chunked prefill implemented and exact vs dense; natural E formation reaches only E2.
- 294 — Packed V3 recovers V1/V2 sentinel regressions, 7-workload geomean +0.92% vs V0 (vLLM comparison superseded by 295).
- 296 — Direct vision output binding: encoder D2D copies 24 -> 0.
- 297 — P512 capability is opt-in; P128 default.
- 298 — Narrow/wide profile engine preserves P128; chunk 512 regresses multi-image; capability bug fixed.

## Open questions
- P512 VLM calibration coverage and D fragmentation with longer chunks; chunk length as a selector action -> W008 (notes 311, 319).
- Cross-schedule/batch-shape output divergence -> W013.
- vLLM full12 repeats, frozen-control caveats -> W007.
- Memory-reduced engine (owner-unique KV, shared E/P workspace) -> W009.
- Dual-profile Cosmos prefill design (P1-P8 / P9-P16, proposed in 287) was not picked up by any later item; this is why the status is waiting rather than done.

## Artifacts
- `.local/artifacts/models/gemma-4-e2b-it-awq` (present)
- `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq` (present)
- `.local/results/gemma4-packed-prefill-g4-20260912` (present)
- `.local/results/gemma4-e2b-awq-full12-20260911` (present)
- `.local/results/v0101-forward-port/batch-frontier-20260911` (present)
- `.local/artifacts/binaries/profiled-prefill-20260913` (present)
- `.local/results/gemma4-e2b-awq-batch-frontier-20260911` (deleted; note 370)
- `.local/results/gemma4-profiled-prefill-20260913` (deleted; note 370)
