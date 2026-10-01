---
id: W014
status: active
updated: 2026-10-01
notes: [360, 361, 362, 363, 364, 365, 366, 367]
---

# Realized overlap cost labels and scheduler redesign

## Goal
Make the online P/D cost key reflect physical E/P/D interval overlap, remove the Gemma long-prefill throughput bimodality, and decide whether the scheduler's learned P+D authority can be simplified into one pricing path.

## Current state
- Cost key: P/D observations are keyed by the realized CUDA-interval E overlap (tri-state: overlap / no overlap / deferred; deferred observations are discarded, not replayed); candidate prediction still uses the planned E-active bit. Planned E-active was true on 16-20 dispatches while realized overlap was on 346-354 of about 27.5k (notes 361, 362, 363).
- Label-path defects fixed: encoder-only recorder retired E intervals during the prefill query of a D-anchored dispatch; planned serial reference reused under a realized-context mismatch; decode-service sample and calibration key E handling (note 363). Startup measured decode-service selection (`5a9b417`, `1ae84b3`) does not beat the observe-only control and regresses Cosmos balanced by 5-8%; not promoted (note 363).
- Observer overhead: encoder-only observer +0.75% and full +1.69% throughput geomean versus off, so no recorder should be removed for performance (note 364).
- Gemma long-prefill gap vs 9db3 is variance, not a mean regression: pooled 77 runs 9db3 622.6 tok/s (sd 8.4) vs post-9db3 606.2 (sd about 16), P(9db3 > current)=0.64 (note 365). Not caused by the label fixes (4-arm ablation, note 364).
- Fix: a directly measured overlap price (`runtime_exact` / `runtime_residual`) now outranks the contextual P+D head (`c7a1b71`), with the residual-compression transfer `19d8aa8`. Gemma long-prefill 8/8 in the high mode (median 615.5, range 614-630, sd about 5). Full24 x3: 72/72 cells above vLLM, geomean +1.11% vs 9db3 and +0.02% vs `1fab188` (note 366).
- Redesign premise falsified: with the head disabled, full24 x3 geomean is -1.82%, 12 workloads lose (e.g. Cosmos balanced -6.5%, Gemma mixed -6.6%), only Gemma long-prefill wins (+5.1%, 644.9 tok/s); decode-row slowdown under overlap makes makespan-only pricing wrong. Overlap-by-default and head removal are withdrawn (note 367).
- Also tested in note 367: epoch-gated residual previews (`41f287f`) cut selector calls (Gemma long-prefill 62,082 -> 614) with throughput within noise; opt-in-only idle host wait (`519ddb6`, default off because Cosmos wave-drain TTFT +4.5%); separate complete-P+D head (`1055f0e`) -4.9% and reverted; `policy_reset` rejected; measured-reference labels (`50cf646`) -4.4% and reverted (`a48513d`).
- The tip binary (`863a6d4`) matches `c7a1b71` within noise (geomean -0.12%, note 369 in → W010). Output-quality gate was the promotion blocker (notes 362-366) and is narrowed in → W013.

## Conclusions
- 360 — Context key for external-encoder background plus E-active state-lifetime fix; full24 x3 all above vLLM, Gemma long-prefill -7.0% vs 9db3 in that screen. Cited campaign directories later deleted (note 370).
- 361 — Cost labels use realized CUDA-interval E overlap; full24 x3 every workload within -0.3% or better versus 9db3 (campaign citation corrected in 363).
- 362 — Tri-state labels (overlap / none / deferred); Gemma long-prefill -2.6% vs 9db3 (corrected from +0.7% in place); encoder-only recorder by default in VLM serving.
- 363 — Label/analyzer fixes; startup decode-service campaigns not promoted; startup SIGSEGV with tokenizer DOM corruption recorded (unreproduced); corrects 361/362 citations.
- 364 — Fix revalidation 72/72; observer overhead negligible; long-prefill bimodality is pre-existing; batch-1 reference shows Gemma 60% exact vs Cosmos 100%.
- 365 — Bimodality is variance; `19d8aa8` correct but not mode-changing alone; contextual P+D head bias mechanism identified.
- 366 — Measured overlap price outranks the head; removes bimodality; full24 x3 no regression.
- 367 — Head-off falsifies overlap-by-default and head removal; decision-loop fix, idle wait (opt-in), separate-model and measured-reference experiments reverted; head bias remains a real but unfixed problem.

## Open questions
- Fix the head's reference bias (residual observations rewarded against remaining-work reference; offline relabel shows 62% negative planned-reference labels vs 0% with measured standalone medians) without losing the head's wins on the other 12 workloads. Needs a standalone cost that exists for complete-P+D shapes (covering/interpolated or a per-token prefill and decode-bucket model); exact-key medians are insufficient (note 367).
- Any pair-cost objective must price decode-row slowdown (decode 14 ms alone vs about 44 ms under overlap), not only makespan (note 367).
- Validate the E-active interference coefficients on Gemma multi-image and vision-heavy; WAIT for E formation stays with the formation policy (note 367).
- Per-phase completion (Stage 2) and event-driven wake-ups for time-based waits; idle host wait stays opt-in (note 367).
- Stage 1 decision elision (`9ba95bc`) was superseded by `41f287f`; the `FormationPolicy` seam (stage 3) was not implemented.
- Startup SIGSEGV with tokenizer DOM corruption (`compute-sanitizer` / ASan repeat) (note 363), unresolved.
- Gemma mixed TTFT mean +15% after `c7a1b71` (p95 not checked, note 366).
- Whether `19d8aa8` survives is undecided (note 367).

## Artifacts
- `.local/baselines/measured-authority-c7a1b71-20260928` (present)
- `.local/baselines/realized-label-fix-1fab188-20260928` (present)
- `.local/results/measured-authority-full24-3x-20260928` (present)
- `.local/results/headoff-full24-3x-20260929` (present)
- `.local/results/epoch-preview-ab-20260929` (present)
- `.local/results/gemma-long-prefill-authority-ab-20260928` (present)
- `.local/results/realized-encoder-label-full12-screen-20260927` and `tri-state-encoder-label-full12-screen-20260927` (deleted; note 370)
- `.local/results/startup-service-holistic-clean-20260928` and `startup-service-boundary-20260928` (deleted; note 370)
