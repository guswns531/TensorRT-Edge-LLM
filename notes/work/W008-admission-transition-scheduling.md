---
id: W008
status: done
updated: 2026-10-01
notes: [299, 300, 301, 302, 303, 304, 305, 306, 307, 311, 318, 319, 322]
---

# Lifetime/dynamic admission and phase transition scheduling

## Goal
Diagnose why Gemma V3 trails vLLM on long-prefill, bimodal and vision-rich traces, and decide which admission, encoder-overlap and prefill-chunk mechanisms (byte/lifetime admission, small-E, E->P->D transitions, dynamic chunking, decode-burst optimization) belong in the runtime.

## Current state
- Gemma deficits were admission and formation, not scheduler cost: long-prefill/bimodal hit the full 96-page KV reservation (submit-to-admit ~1.9 s), vision traces waited 0.7-1.7 s before E start with mean E batch 1.4-1.7; 5-8 ms host submission spans dwarf scheduler cost (note 299, corrects 295's "not KV capacity").
- Capacity A/Bs: KV 192 pages removes admission wait but hurts long-prefill (-27.8%, TTFT x2) at that engine; encoded capacity 4 -> 12 gives +34-52% VLM throughput but +38-86% TPOT (note 300). A producer-class mixing defect in popBatch was fixed (note 300).
- Opt-in byte/lifetime encoded admission matches a static slot window on Gemma and lets Cosmos large admission complete, but Cosmos vision-heavy TPOT +53%; kept opt-in because memory-feasible admission is not service-efficient admission (note 302). It is retained (with the release-contract query as opt-in ownership scoring, decode-stage diagnostics and a drained-boundary vision-chunk diagnostic) while the two D-priority rules (local E yield, global D-continuity) were removed; ownership scoring gave ~+/-1% and forced chunk128 was not adopted (notes 303, 304).
- ~72% of the Cosmos heavy decode-cycle increase (28.1 -> 43.0 ms/token) is D ready-queue wait; forced 128-token vision chunking on Cosmos cut throughput 11-15% (note 303).
- Fixed small-E (E1/E2) gives real overlap but is not the final policy; active dynamic E selection and the shadow E->P->D transition candidate were not promoted (scheduler p95 up to 243 us, unstable recommendations) (notes 305-307).
- Dynamic prefill chunk screen: no candidate promoted at the time; fixed P512 best for long-prefill, D-ready P128 cap best for multi-image (+11.61%) (note 311). Later single-screen results: adaptive P128/P256/P512 on Gemma multi-image TPOT mean -63.7% (27.89 -> 10.13 ms; E2E p95 +3.3%) (note 319, partly addresses 311's open item; adaptive chunking is off in the current default, note 339); online max-service-rate decode-burst optimizer, Cosmos poisson TTFT 300 -> 224 ms (note 322, revises 320). Note 318 added a 6-feature RLS PhaseTransitionPredictor replacing decodeBurstLimit=8 and maxOverlapPrefillTokens=128 (160/160 unit tests, no end-to-end measurement in 318 itself; 322 builds on it).
- 318/319/322 are single screens; their headline comparisons (including 322's "Gemma vision-heavy victory", 554 vs vLLM 560 tok/s) were not confirmed by the Full24 x3 revalidation (note 331: Gemma vision-heavy 498.63 tok/s, -10.93% vs vLLM; Cosmos poisson -9.66%) and are not governing numbers. Current serving default and its measured state: W010; vLLM numbers: W007.

## Conclusions
- 299 — Gemma V3 deficits traced to KV reservation, E-start delay and small E batches (corrects 295).
- 300 — KV and encoded-capacity A/Bs: trade-offs, no single static setting wins; popBatch producer mixing fixed.
- 301 — Plan for opt-in lifetime/byte encoded admission, compared on both models.
- 302 — Lifetime admission works memory-wise, not service-wise; opt-in, not default.
- 303 — Four interventions (local E yield, global D-continuity, ownership release credit, forced Cosmos chunk128) showed no speedup; first two removed, ownership scoring opt-in, chunk128 diagnostic-only; Cosmos TPOT cause is D queue wait.
- 304 — Architecture synthesis: keep V3 + lifetime admission; filling free memory does not optimize TPOT.
- 305 — Small-E progressive overlap is real but fixed small-E is not final (superseded by 306 for active selection).
- 306 — Active dynamic E-preparation selection failed six-workload promotion; shadow mechanism valid.
- 307 — E->P->D transition shadow validated mechanically; not promoted (instability, p95 243 us).
- 311 — Dynamic prefill chunk screen: none promoted; chunk length becomes a global-selector action (partly addressed by 319).
- 318 — PhaseTransitionPredictor replaces decodeBurstLimit=8 and maxOverlapPrefillTokens=128; unit tests only in this note.
- 319 — Unified action space with P128/P256/P512 chunking: Gemma multi-image TPOT -63.7%, single screen.
- 322 — Online service-rate horizon optimizer; Cosmos poisson TTFT 224 ms, single screen (headline claims not confirmed by 331).

## Open questions
- Gemma batch-shape output divergence blocking exact-output gates -> W013.
- Workspace/memory modes and shared E/P exclusivity behind the admission results -> W009; M-RoPE/encoder-admission lifetime fix -> W010 (note 333).
- Realized-overlap cost labels, event-driven decisions, E->P->D transition pricing and scheduler p95 -> W014 (notes 360-367).
- Prefill formation and ingress effects on P cohorts -> W012.

## Artifacts
- `.local/results/gemma4-v3-activity-diagnostic-20260913` (present)
- `.local/results/gemma4-v3-capacity-ab-20260913` (present)
- `.local/results/gemma4-vllm-gap-20260915/baseline` (present)
- `.local/results/service-admission-20260914` (deleted; note 370)
- `.local/results/lifetime-encoded-admission-20260913` (deleted; note 370)
- `.local/results/small-encoder-overlap-20260914` (deleted; note 370)
- `.local/results/gemma4-dynamic-prefill-transition-20260915` (deleted; note 370)
- `.local/results/unified-action-p512-screen-v3-20260920` (deleted; note 370)
