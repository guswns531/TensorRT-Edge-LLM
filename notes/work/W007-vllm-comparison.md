---
id: W007
status: waiting
updated: 2026-10-01
notes: [286, 289, 295, 308, 310, 317, 320, 324, 325, 327, 328, 329]
---

# vLLM comparison campaigns

## Goal
Establish a fair, repeatable comparison of the phase-serving runtime against vLLM (Gemma control 0.28.0; frozen Cosmos control 0.27.1) on Cosmos-Reason2-2B and Gemma 4 E2B over the 12-workload suite, decompose the gaps, and keep a frozen vLLM control.

## Current state
The single-run scorecards in notes 317-329 are not the current numbers; 328 and 329 carry audit errata, and the scorecards were superseded by re-measurement in 331, 338, 339 and 369.
- Frozen controls: Gemma vLLM = seq24 / KV480 / P4096 / sparse CUDA graphs, one run per workload (note 295); Cosmos vLLM 0.27.1 = frozen raw results, mostly 3 runs, vision-heavy 2 successes + 1 failed attempt preserved (notes 331, 338). The earlier seq8 Gemma control understated vLLM by 65-161% (note 295). vLLM is not batch-invariant either (note 368).
- vLLM gap is work-granularity (fixed P128 chunks, exact D cohorts, eager execution), not GPU idle (1-3%) (note 308).
- Runtime-contract revalidation at `43c680a` (Full24 x3, 72/72, 6,435 requests): token/s wins Gemma 10/12, Cosmos 6/12; Cosmos mixed -24.74%, multi-image -11.57%, poisson -9.66%, vision-heavy -32.35%, Gemma multi-image -2.34%, vision-heavy -10.93%; TTFT mean wins Gemma 4/12, Cosmos 6/12 (note 331; errata on 328/329, supersedes the 317-327 scorecards).
- After throughput work (note 337) the same contract at `9db3ed3` x3 (72/72): token/s geomean vs frozen vLLM Gemma +35.73%, Cosmos +16.64%, overall +25.82%; median wins 24/24, individual runs 71/72 (Cosmos wave-drain last repeat -3.28%) (notes 338, 339). Latency is not uniformly better: across 24 cells TTFT mean/p95 wins 14/15, TPOT 22/19, E2E 22/21; Gemma TTFT p95 geomean +10.05% worse (note 339).
- Current tip `863a6d4` Full24 x3: 72/72 runs above frozen vLLM, geomean +27.05% (note 369); Gemma cross-repeat exact 589/632, Cosmos 1513/1513 (note 369; cause in W013).
- Upstream v0.11.0 on the same hardware: ours 5.28x upstream, frozen vLLM 4.16x upstream (geomean over 24 cells, one run per cell) (note 371; tracked in W016).
- Caveats that still apply: frozen vLLM is reused, not fresh paired; Gemma vLLM is 1 run per workload; no paired confidence intervals; only RTX 3080 and these two models (notes 338, 339).
- Early claims withdrawn: 12/12 Gemma sweeps and "victory" scorecards (323, 327) stitched single-workload screens onto earlier rows, and 327 used different vLLM vision-heavy/multi-image values (505.78/283.02 vs frozen 559.83/381.34); 328's 22/24 was single-run, throughput-only with a wrong KV figure (192, not 480) and Gemma TTFT p95 +30.64% (note 328 audit); 329's 383.75/544.87 tok/s screen values are not substantiated by the retained Full-12 aggregates (note 329 audit; 331).

## Conclusions
- 286 — vLLM 0.28 supports Gemma4 (AWQ eager via compat view; compiled OOMs on 10 GB); Cosmos fresh full12: V3 wins tokens 12/12 (median +5.83%), TTFT trade-offs on bimodal/mixed/wave/multi-image.
- 289 — Optimized vLLM Gemma (eager backend + decoder graphs) is +67-79% over all-eager; became the Gemma control (later replaced by 295's seq24 control).
- 295 — Seq24/KV480/P4096/sparse-graph vLLM control; packed V3 wins 7/12 (+7.86%); supersedes 292/294 claims.
- 308 — Gap is work granularity not idle GPU; remediation plan (superseded by 310).
- 310 — Gemma full-12 x3 with lifetime admission, graphs, sampling refill: 9/12 tok/s (+23.87% geomean), TTFT 5/12 (superseded by 317, then 331).
- 317 — Gemma KV192 shared_ep scorecard: 11/12, +27.75% geomean, single run (corrected by 331).
- 320 — Cosmos shared_ep scorecard: 10/12, +12.95% geomean, single run (corrected by 331).
- 324 — Dual-model revalidation: Cosmos 11/12, Gemma 9/12; contradicts 323's 12/12 (corrected by 331).
- 325 — All-opts: Cosmos pooled_io 4249 tok/s, Gemma +27.41%; pooling helps text, hurts vision-heavy (pooling later removed, W009).
- 327 — Claimed Gemma 12/12 sweep with mixed vLLM baselines (superseded by 328, corrected by 331).
- 328 — 22/24 campaign with audit correction: single-run, throughput-only, seven-metric recomputation (corrected by 331).
- 329 — Gemma vision scaling and three lifecycle fixes; screen values unsubstantiated per audit (corrected by 331).

## Open questions
- Fresh paired vLLM runs (>=3-5 repeats, Gemma and Cosmos vision-heavy) to replace frozen single-run references; needed for citable status. No later item scheduled this.
- Gemma TTFT p95 and vision-heavy/multi-image latency deficits vs vLLM (E admission path) -> W014 (scheduler redesign), W012 (formation).
- Second GPU/platform repeat of the comparison. Not scheduled.
- Why waiting rather than done: the repeat/fresh-baseline questions in 286, 310, 338, 339 were not picked up by any later line.

## Artifacts
- `.local/results/tip-full24-3x-20260929` (present)
- `.local/results/review-correction-20260926` (present)
- `.local/results/gemma4-vllm-capacity-sweep-20260912` (present)
- `.local/results/gemma4-vllm-gap-20260915/graphs-lifetime-final-refill-confirm` (present)
- `.local/results/gemma4-e2b-awq-vllm028-graph-eager-20260911` (present)
- `.local/results/v0101-forward-port/vllm-028-cosmos-fresh-20260911` (present)
- `.local/results/dual-model-full24-clean-sweep-20260921-v2` (deleted; note 370)
- `.local/results/gemma-kv192-full12-20260920` (deleted; note 370)
