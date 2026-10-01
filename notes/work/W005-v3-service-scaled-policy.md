---
id: W005
status: waiting
updated: 2026-10-01
notes: [270, 271, 272, 273, 276, 277, 278, 279, 280, 281, 282, 283]
---

# V3 service-scaled policy and heuristic elimination

## Goal
Replace V2's fixed internal-urgency constants (5/2 ms) and model/GPU-dependent policy constants with service-epoch-normalized quantities, producing a profile-free V3 phase-selection policy that beats V0/V2 and frozen vLLM without workload-specific thresholds.

## Current state
- Service-normalized age scoring was first built as an offline evaluator and then as runtime shadow scoring. Shadow covered 110/111 real decisions with 23 shadow/live disagreements; an active normalized selector failed the four-workload gate (mixed TPOT p95 +23%) and the Pareto-safe form made 0 replacements, so normalization stayed diagnostic (notes 270-273).
- V3 (service-scaled-transition) keeps V2's scalar RLS + H=2 transition and replaces the fixed urgency constants with frozen service-epoch-normalized age plus a mechanism-only candidate frontier (notes 276, 277). Fresh full12: +12.83% token/s over V0, +9.47% over V2 (geomean), 12/12 over fresh equal-contract vLLM at that time (+10.41%) (note 279).
- Recovery filter was overactive (invoked in 16-49% of decisions; disabling it raised core-four geomean 1209 -> 1301 token/s) and was defaulted off in the simplified V3 (notes 279, 281). Scheduler p95 decision cost above 50 us was recorded as open (note 279).
- Canonical simplified V3 (recovery off, legacy pair eligibility, encoder credit timer, vision exclusivity threshold, no-SLO hysteresis and fixed warmup removed) is performance-neutral vs same-day control (+0.02% geomean) and +14.20% token/s over frozen vLLM, 11/12 wins (note 281, corrects 279).
- Disclosed compatibility guards remain: the 25 ms encoder formation cap and encoder arbiter; automatic replacement of both failed the TPOT gate (VLM6 TPOT p95 -10.54%, wave/multi tails ~-42%/-22%; note 281), and deleting the wait and arbiter regressed wave/multi TPOT p95 by 22.35%/23.58% (note 283).
- Preparation/formation separation (persistent preparation worker, P/D dispatch during preparation, separate E cost) is opt-in only: compatibility mode matches frozen V3; multi-image TPOT p95 +33% worse with P/D dispatch; P4 needs detached prepared-input leases (notes 282, 283).
- V3 is the local serving default; later cross-model vLLM numbers and revalidation are tracked in W007 and W010. vLLM comparisons: 278 and 281 use frozen Cosmos references, 279 a fresh equal-contract run (vision-heavy 2/3 successes); all on the pre-revalidation runtime and not current.

## Conclusions
- 270 — Offline service-normalized shadow evaluator (11 tests); retained logs held 0 complete snapshots, so no counterfactual had been scored.
- 271 — Host-monotonic service clocks in V2 telemetry: snapshots in 115/115 decisions, 0 complete counterfactuals, 1092.8 token/s with output hash equal to V1.
- 272 — Runtime shadow covers 110/111 decisions with 23 disagreements; V2 vs V1 +2.00% geomean, not promoted universally.
- 273 — Canonical Release/SM86 build validated; active normalized selector failed the four-workload gate, Pareto-safe form made 0 replacements.
- 276 — V3 plan: service-scaled transition, frozen epoch-normalized age, mechanism-only frontier, N-1..N5 stages.
- 277 — Causal S0-S7 execution plan with semantic split of fixed-time constants (urgency / physical cost / absolute SLO).
- 278 — V3 without SLOs beats V2 request throughput 9/12 (+7.26% geomean) and frozen vLLM token throughput 12/12 (+9.21%); promoted as best no-SLO candidate.
- 279 — Fresh full12: V3 +12.83% over V0, +9.47% over V2, 12/12 over vLLM; recovery filter overactive (corrected by 281 for the simplified form).
- 280 — H0-H7 plan to remove model/GPU/arrival-dependent constants from V3.
- 281 — Simplified V3 neutral vs control (+0.02%), +14.20% over frozen vLLM (11/12); 25 ms E cap and arbiter kept.
- 282 — Opt-in preparation/formation separation substrate (P0-P3), defaults unchanged.
- 283 — Separation switches fail the cross-metric gate; removing 25 ms wait/arbiter unsafe; all stay opt-in, P4 needs detached prepared-input leases.

## Open questions
- Detached prepared-input leases and late E packing (P4), replacing the arrival EWMA, 25 ms wait and arbiter with global-selector candidates (E-now / prepared-E / event-backed WAIT): not implemented by any later note; W008 (305-307) covers only fixed small-E and shadow E->P->D, not promoted; note 367 (W014) states E wait stays with the formation policy.
- Decode-continuity terminal cost and forced wave/multi counterfactuals: no later note found; W014 (note 367) prices decode-row slowdown under overlap but does not run these.
- Second model/GPU repeat -> W006 (Gemma 4), W007 (vLLM comparison).
- Scheduler decision p95 above 50 us: not re-measured; note 367 (W014) cuts selector calls via epoch-gated previews, no p95 figure.

## Artifacts
- `.local/results/v0101-forward-port/service-shadow-v2-20260909` (present)
- `.local/results/v0101-forward-port/p0-p4-20260909` (present)
- `.local/results/v0101-forward-port/v3-service-scale-20260910` (present)
- `.local/results/v0101-forward-port/heuristic-elimination-20260911` (present)
- `.local/results/v0101-forward-port/preparation-separation-20260911` (present)
- `.local/builds/v0101-release` (present)
