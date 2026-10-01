---
id: W011
status: done
updated: 2026-10-01
notes: [340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350]
---

# Startup calibration and decode service model

## Goal
Replace the static D (decode) cost prior with startup-measured capability/service costs, and decide whether startup-chosen decode partitions (dense vs split) improve serving over the static table.

## Current state
- Opt-in startup calibration (D/P/E probes, measured D batching): 48/48 cells but regressed vs same-binary baseline (Gemma geomean throughput -2.84%, Cosmos -0.21%; startup 17-18 s / 16-17 s); not promoted (note 340).
- Factorization (140 cells + 6 learning diagnostics): swapping the cost source alone regresses (balanced x3 median vs static-table baseline: Cosmos B -5.5% / C -4.3%, Gemma B -2.2% / C -1.3%; full12 geomean B: Gemma -1.37%, Cosmos -0.82%); extra probes recover only part; warmup shrink is near parity; chunk-aware probe-length bug fixed; coverage != learned quality (note 341, refines 340).
- Equal-work trials: Gemma D8 vs D4+D4 saves 2.84% GPU but is +0.85% slower host-inclusive; Cosmos D64 vs D32+D32 +70% slower; async control Gemma +0.59%, Cosmos +71.7%. D-split search ended, no split applied (notes 342, 343, 345).
- Startup-generated host decode-service model (no static table): balanced -4.11% Gemma, -4.24% Cosmos vs static; Gemma exact 61/64; not promoted (note 344).
- The service regression traced to label contamination: D3 host-service p95 (~27.7 ms) included a following P/P+D starting before sampling/commit, causing D1x3 selection (D1 dispatches 56-57 vs 20) (note 348). Interval guard, same-binary on/off 2x: +4.12% tokens/s (1160.48 -> 1208.35), D1 dispatches 57.5 -> 21, still -1.55% vs static; guard kept on by default (note 349).
- Predicted [4,3] vs realized rolling [4,4] is not a loss; DP first batch equals the service-density choice; remaining gap is 5-7 extra P dispatches (note 350, later attributed to client closed-loop feedback in 351, W012).
- Gemma D8 vs D1/D4 divergence is a top-2 margin of 0.0073 vs 0.109; graph and eager identical; reproduced by changing one decode turn; not new to the service model (notes 345-347). Cause identified later as INT4 batch-shape numerics (W013, note 368).
- Default remains the static-table serving binary (9db3ed3, W010); startup service variants were again not promoted in note 363 (W014).

## Conclusions
- 340 — Startup calibration regresses (-2.84%/-0.21%); not promoted.
- 341 — Cost-source swap alone already regresses (extra probes recover part, not all); probe-length bug fixed (refines 340).
- 342 — Decode trial planner (plan-only): 0 candidates with positive guarded saving.
- 343 — Equal-work trials: neither D split pays; split search ended.
- 344 — Startup-generated service model: -4% balanced, Gemma exact 61/64; not promoted.
- 345 — Async D8 vs D4+D4 no gain; Gemma divergence at output index 13 is batch-shape.
- 346 — Divergence margin 0.0073 vs 0.109; HTTP small splits realized 41/41; service -3.42% vs static.
- 347 — Single-turn decode change reproduces divergence; D3 dense (7.0 ms) beats D2+D1 (12.5) and D1x3 (18.6).
- 348 — Following-phase contamination of D service samples explains D1x3 choice.
- 349 — Interval guard +4.12% same-binary; kept on; [4,3] never realized (0/15, 0/18).
- 350 — [4,3] vs [4,4] not a loss; gap is extra P dispatches (re-attributed by 351).

## Open questions
- Output exact-identity gate (60-62/64 in 344, 349) -> W013.
- Extra P dispatches / P-candidate membership static vs service -> W012.
- Startup service and measured-label campaigns after 349 -> W014 (note 363).
- Static D prior/clamp portability to new model/GPU (note 339): no startup mechanism beat the static table here; closing evidence would be a new-model/GPU measurement (see W016 for the v0.11.0 port).

## Artifacts
- `.local/builds/v0101-validation` (present)
- `.local/current/active` (present; runtime = 9db3ed3)
- `.local/results/startup-calibration-20260927`, `.local/results/startup-factorization-20260927`, `.local/results/startup-service-v2-20260927` (deleted; note 370)
- `.local/results/decode-service-guard-ablation-20260927-{on,off}`, `.local/results/decode-single-turn-20260927`, `.local/results/decode-logits-20260927` (deleted; note 370)
- `.local/baselines/startup-service-v2-20260927`, `.local/baselines/startup-factorization-v2-20260927`, `.local/baselines/startup-autotune-step2-v2-20260927` (deleted; note 370)
