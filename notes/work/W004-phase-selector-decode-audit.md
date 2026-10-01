---
id: W004
status: done
updated: 2026-10-01
notes: [252, 253, 254, 255, 256, 264, 265, 266, 267]
---

# Phase selector inputs and decode readiness audit

## Goal
Explain why decode-ready work waits (mixed/vision-heavy TPOT tails) and whether widening the selector's candidate set or
horizon recovers it without workload rules.

## Current state
- Producer readiness is not the delay: commit -> ready ~1.4 us, ready -> next-D start averages ~23 ms with ~95.8% covered by E/P/D host spans, i.e. phase arbitration (note 252).
  A detailed snapshot's `max_slo_violation_us=0` is an uncomputed field (note 252).
- Only the P/D preview winner (plus E candidates) reaches the final selector; protected completions differ between preview and final (note 253, `PhaseGlobalCandidateAudit`).
- Vision-heavy, D-ready decisions without D (58 native), final D was absent in 43 and predicted SLO-late in 15; absence comes from the P TTFT hard guard at candidate generation (15/12 E-competing P/D stages, 0 with guard off) plus P/D winner-only truncation (23 still absent with guard off); E->P protection can raise D's violation above E's (one example); post-selection overrides 0 (note 254).
  Guard-off full12: long-prefill +5.58%, mixed +7.07%, poisson +5.53% throughput, but mixed TPOT p95 +20.35%, short -4.75% (TTFT mean +16.4%), balanced/decode-heavy TTFT p95 +16.3%/+14.4%; default guard-on kept. The -4.8% Poisson figure is a guard-on repeat vs an earlier single run.
- All negative, none promoted (default off or removed): forwarding standalone P and D alternatives (91 candidates, Mixed TPOT p95 +54%) (note 255);
  encoder-inclusive horizon (worse Mixed TPOT/E2E, removed) (note 256); preserve-expired-decode (mixed -1.71%, vision-heavy TPOT mean +9.9%/p95 +13.5%) (note 264).
- Restored standalone D was selected 0/23 times (all `all_late_efficiency_recovery`); 16 final actions still included D via P+D/E+D; vision-heavy D cohorts shrink 21.2 -> 17.3, mixed 32.2 -> 33.3 (note 265).
  Minimum-additional-lateness ranking disagrees with all-late efficiency but has large modeled compression loss; global replacement unjustified (note 266).
- Policy-neutral host-path optimization kept: 12/12 token hashes, Mixed +1.68%, Balanced -1.23% in ABBA (note 256).
- Same-state P2 vs P2+D48 replay not possible: audit capture lacks ownership/row state and the frontier did not recur in 3 full captures (note 267); needs an in-memory checkpoint/restore facility.
- The line's questions moved to V3 (W005): TTFT hard guard bypassed in V3 (note 367), decode-continuity terminal cost and preparation placement (note 281).

## Conclusions
- 252 — Decode-ready publication is fast; delay is phase arbitration, not producer latency.
- 253 — Preview winner only reaches final selector; audit API added, no behavior change.
- 254 — Selector audit: D suppressed by P TTFT hard guard at generation; guard-off mixed result, default kept.
- 255 — Opt-in P/D alternatives forwarding works mechanically but worsens Mixed tail; off.
- 256 — Encoder-inclusive horizon rejected; host-path optimization kept.
- 264 — Preserve-expired-decode candidate not promoted.
- 265 — Guard audit: restored D never selected standalone; continuity vs cohort-efficiency trade-off.
- 266 — Shadow ranking by lateness does not justify replacing all-late efficiency.
- 267 — Branch replay infeasible without checkpoint/restore; readiness checker and full capture added.

## Open questions
- Runtime barrier/checkpoint facility for KV, vision and policy state, then forced-branch replay with rotated A/B order -> W005 (forced branch replay listed in note 280 ledger).
- Bounded P/D feasible frontier and D-continuity cost in the selector -> W005 (note 281), W014 (note 367).

## Artifacts
- `.local/results/v0101-forward-port/ready-path-20260908` (present)
- `.local/results/v0101-forward-port/selector-audit-20260908` (present)
- `.local/results/v0101-forward-port/pd-frontier-20260908` (present)
- `.local/results/v0101-forward-port/encoder-horizon-20260908` (present)
- `.local/results/v0101-forward-port/expired-decode-20260909` (present)
- `.local/results/v0101-forward-port/decode-guard-audit-20260909` (present)
- `.local/results/v0101-forward-port/all-late-shadow-20260909` (present)
- `.local/results/v0101-forward-port/branch-readiness-20260909` (present)
