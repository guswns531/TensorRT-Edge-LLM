---
id: W012
status: waiting
updated: 2026-10-01
notes: [351, 352, 353, 354, 355]
---

# Prefill formation and backend ingress

## Goal
Explain why service-selected and static scheduling form different P (prefill) batches, and whether that difference is a scheduler cost-model effect, client arrival feedback, or server ingress ordering.

## Current state
- Extra P dispatches are not a consistent service-policy defect; client closed-loop feedback changes ingress/cohorts: closed-loop24 service 44.0 vs static 39.0 P dispatches, -2.60% tok/s; client64 (near-fixed ingress) service 38.3 vs static 44.7, +0.15%; not all of the difference is client-caused (note 351, corrects the 350 attribution).
- Identical HTTP send times do not give identical server admission: backend submit order flips (25<->26), ~200 ms admission differences and different P cohorts; runs 2/3 show an earlier server-internal P difference at request 3/12 (note 352).
- Ordered backend ingress (submit 0->63) gives throughput parity (service +0.19%) but P6/7 vs P5 divergence at the 2nd P dispatch persists in 3/3 runs, so ingress order alone does not explain formation (note 353).
- The P6/P5 split is traced to a sparse covering cost: two slow P1/37 samples raise the P1/33 service reference and flip the wavefront seed via service-age ranking; reproduced in a C++ test; no direct GPU-cost score selected P5 (note 354).
- Trusted-reference ablation (exclude untrusted sparse reference): mechanism works but early seed unchanged; throughput -0.46% (1243.62 vs 1237.87), P dispatches 37/38 (trusted) vs 41/38 (baseline); first P6/P5 divergence did not reproduce even in baseline, output exact 61/64; not promoted (note 355).
- The greedy output divergence seen here begins before any P membership difference (note 356, W013); its cause is INT4 batch-shape numerics (note 368), so output identity is not evidence about P formation.

## Conclusions
- 351 — Extra P dispatches are mostly closed-loop client feedback (corrects 350).
- 352 — Same HTTP send still gives different server ingress/admission order.
- 353 — Ordered ingress yields parity throughput but P6/P5 divergence remains.
- 354 — P6/P5 seed switch from sparse covering cost + service-age ranking.
- 355 — Trusted-reference ablation: -0.46%, no seed change; not promoted.

## Open questions
- P5/P6 forced-branch trajectory comparison from an identical pre-branch snapshot (or evidence that seed-action changes occur often in real serving) is the only thing that would reopen the P-reference trust rule (note 355). No later line picked this up; waiting because later work (W014) moved to realized-cost labels and formation policy without resolving it.
- Output identity gate -> W013.

## Artifacts
- `.local/results/prefill-arrival-openloop-20260927-static`, `.local/results/prefill-arrival-closed24-current-20260927-static` (deleted; note 370)
- `.local/results/prefill-ordered-ingress-v2-20260927-static`, `.local/results/prefill-ordered-ingress-20260927-static` (deleted; note 370)
- `.local/results/prefill-formation-age-20260927-service`, `.local/results/prefill-formation-cost-20260927-service` (deleted; note 370)
- `.local/results/prefill-trusted-20260927-baseline`, `.local/results/prefill-trusted-20260927-filtered` (deleted; note 370)
- Raw runs gone; numbers survive only in notes 351-355.
