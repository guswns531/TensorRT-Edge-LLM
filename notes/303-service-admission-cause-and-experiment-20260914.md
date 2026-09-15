# Service admission: cause, objective, and two-model experiment

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

**Final status:** 42 diagnostic performance cells and 4 post-cleanup HTTP smoke cells completed. Rejected
service-priority rules are removed; original V3 and opt-in lifetime admission remain. Sections below are
chronological revisions, not concurrent production policies. Architecture synthesis: [note 304](304-dynamic-admission-architecture-and-tradeoffs-20260914.md).

## Starting evidence

Note 302 established different trade-offs under the same byte/lifetime mechanism. Gemma heavy P dispatches
fell from 118.33 to 81.67 and D dispatches from 539 to 150. Cosmos heavy D dispatches fell from 115.67 to
72.33, but mean TPOT increased from 28.09 to 42.95 ms and mean E2E from 2538.35 to 3185.05 ms.
Both efficiency and resident delay must be evaluated. A memory budget alone cannot express that preference.

Physical owner retention also differs: the smoke path retains the Cosmos unsplit M-RoPE slab through D,
whereas Gemma can release prefill storage earlier. This affects memory feasibility; it does not establish
that storage retention caused the latency regression. No KV or encoder engine change is part of this test.

## Correcting the initial hypothesis

The first idea was to maximize observed B/T(B)^2, balancing throughput and reciprocal decode cycle time.
Source inspection and raw timing rejected this as the immediate intervention before any performance run.
Cosmos heavy repeat 1 has D GPU total 1337.86 -> 894.04 ms and mean dispatch GPU time 12.16 -> 12.42 ms.
Prefill total also falls 1905.60 -> 1693.01 ms. The large TPOT regression therefore cannot be attributed
primarily to a slower D kernel. Aggregate GPU savings and per-request service delays are different quantities.

Generic HTTP calibration also does not reliably form the engine's maximum D batch. Requiring that exact
shape would leave a curve optimizer without authority, while contracting on sparse smaller shapes would
risk trapping it in small batches. The unvalidated curve helper was removed before measurement.

The implemented `service` experiment protects a concrete ready decode cohort before forming additional
encoder work: if its waiting age exceeds its measured service quantum, defer E formation until the next
poll. Once D receives service its request service epoch advances, so E becomes eligible again. Existing
V3 selects among the remaining P/D actions. No new SLO milliseconds, request-count limit or learning model.
This is a restrictive admission experiment, not a claim that a local gate is the final global policy.

## Mechanism and authority

- Same V3 action selector, P128, engine capacities, KV allocation, request contract and generic calibration.
- Activation only after calibration drains, through `setEncodedAdmissionMode(..., service=true)`.
- Reuse the server's current decode service reference and ready age. Accept runtime exact, interpolated or
  covering references; reject cold/static-only evidence. Require a ready D row and no explicit request SLO.
- Reuse `serviceRecoveryAgeQuanta` already in V3 (one quantum in this contract); this preference remains
  explicit and is not presented as removal of all policy parameters. All times scale with measured service.
- Preserve full E batch construction after eligibility returns; no learned small count chops a batch.
- Emit decode age quanta and candidate reduction counts. No new probes or trace-derived startup data.

This experiment is opt-in and not promoted through `PhaseServingRuntimeConfig`. Production idle-slab
accounting and split-M-RoPE parity remain separate validation requirements from note 302.

## Verification and decision contract

Test measured-reference authority, service age and explicit-SLO exclusion. Build the
same binary for static, lifetime and service runs. First compare both models on heavy and mixed traces,
then extend only if admission actually changes and the trade-off is useful. Retain failures and negative
results. Report throughput, TTFT/TPOT/E2E mean and p95, capacity coverage and fragmentation. Reuse frozen
vLLM only under its previously documented comparison limitations.

Closed, manifest-listed logs in `.local/results/service-admission-20260914/` may be losslessly gzip-compressed
after analysis to stay within disk headroom. No engines, models, current pointers or worktrees may be deleted.

## Status

The C++ build and 217 scheduling/memory/server tests pass. Ten Python tests pass, including a synthetic
request-timeline decomposition. The initial non-GPU build linked unsuccessfully because the configured
libcuda path requires the NVIDIA runtime mount; the subsequent GPU-enabled build completed.

The two-model heavy screening campaign runs in `.local/results/service-admission-20260914/heavy-screen`.
Each static-base/lifetime/service cell shares a binary and post-calibration activation. No performance
improvement is claimed until that campaign completes and mechanism telemetry confirms actual intervention.

## Request-weighted attribution from retained runs

`analyze_decode_service_admission.py` matches request IDs across ready, start, done and token-commit events
only in measurement epoch 1. It reports each request's mean component first, then averages requests; the
host execution interval includes completion visibility and is not a pure CUDA kernel duration. The commit
interval includes sampling/collection, not just a sampler kernel. No unmatched stages or outstanding matched
decode stages remain in the inspected heavy cells.

Cosmos heavy, three existing repeats, milliseconds per generated decode token (request-weighted):

| Component | Static16 | Lifetime | Change |
|---|---:|---:|---:|
| D ready to D start | 11.419 | 22.092 | +10.673 |
| D start to host-observed done | 12.575 | 15.981 | +3.406 |
| Done to committed token | 4.117 | 4.913 | +0.796 |
| Sum | 28.111 | 42.986 | +14.875 |

Approximately 72% of this reconstructed cycle increase is queue waiting. Within text requests, ready wait
is 19.597 -> 35.890 ms; within vision requests, 8.693 -> 17.493 ms. The class split excludes the explanation
that a different text/vision mix alone caused the aggregate regression. It does not by itself distinguish
E interference, P interference, row selection or CPU scheduling inside that wait interval.

Gemma heavy has a different decomposition: ready wait 2.948 -> 8.259, host execution 10.614 -> 18.586,
commit gap 2.044 -> 4.570 ms. Its static estimate here uses repeats 1/2 only, excluding the previously
declared contaminated repeat 3; do not compare this two-repeat attribution as the clean three-repeat
serving table in note 302. These retained decompositions are in the prior campaign's
`decode-service-analysis.json`, with input log and analyzer hashes.

## Screening revision 1: local E yield is rejected

All six heavy screening cells completed. Each result is one run, not a repeated promotion gate.

| Model | Variant | Tok/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|
| Gemma | static4 | 310.83 | 1868.42 / 3940.02 | 14.11 / 22.97 | 2448.18 / 4231.97 |
| Gemma | lifetime | 562.52 | 394.11 / 670.86 | 29.80 / 44.16 | 1513.90 / 2129.91 |
| Gemma | local service yield | 550.93 | 332.06 / 625.89 | 32.94 / 46.96 | 1555.05 / 2104.15 |
| Cosmos | static16 | 673.04 | 1430.24 / 3274.24 | 26.82 / 37.17 | 2493.40 / 3589.89 |
| Cosmos | lifetime | 719.33 | 1547.36 / 3041.35 | 42.75 / 85.18 | 3210.34 / 3372.74 |
| Cosmos | local service yield | 706.46 | 1259.31 / 3120.64 | 52.81 / 86.78 | 3250.88 / 3427.66 |

The guard actually ran: 27/7399 candidate blocking evaluations for Gemma/Cosmos. These are poll counts,
not unique requests. Cosmos ready wait increased 17.71 -> 26.65 ms/token and completion-to-commit increased
5.18 -> 9.48 ms/token. Preventing E formation does not ensure the separate P/D decision serves D: it can
still dispatch P, and asynchronous preparation/completion changes future formation. This experiment does
not justify a throughput/delay improvement claim. The local gate and its helper were removed from source.
The executed tracked patch, runner and identities remain in `heavy-screen`.

## Revision 2: single policy authority

The experimental service switch now enables continuity protection in both uses of `PhaseGlobalScheduler`
(the P/D selector and the outer E/P/D selector). Byte feasibility still controls ownership; encoder formation
has no new service gate or new count limit.

After hard feasibility and existing deadline safety, inspect runtime-derived no-SLO decode references. If
a protected request has waited at least the existing service quantum, retain safe actions that actually
include that request's row in D and have the earliest robust predicted D completion. E+D/P+D remain eligible
if they provide that completion. A candidate merely mentioning the request in its protection metadata does
not qualify. If no suitable safe action exists, preserve the original frontier. Resetting the request's
service epoch after progress naturally restores normal efficiency selection.

This is a preference for an overdue token milestone, not a universal latency optimum. It may sacrifice
TTFT or future cohort size; repeated mixed and text controls are required. No measured throughput improvement
is assumed. `vision_admission_service_blocks` now counts selector-frontier restrictions, including previews,
and must not be compared as the same event count as revision 1's E formation blocks.

Source paths: `cpp/runtime/phase/policy/phaseGlobalScheduler.h`,
`cpp/runtime/scheduling/phaseGlobalScheduler.cpp`, queue/server forwarding methods and the coordinator's
drained activation. Unit tests cover actual row membership, overlap eligibility, hard feasibility, and
explicit-SLO exclusion. Runtime and benchmark contracts remain versioned by manifest/binary hash.

### Revision 2 screen: no promotion

The four-cell `global-heavy-screen` completed with 97/74 restriction evaluations for Gemma/Cosmos. Compared
with same-binary lifetime, Gemma output throughput 556.80 -> 543.75 tok/s, TPOT mean 31.09 -> 30.23 ms,
E2E mean 1537.80 -> 1563.88 ms. Cosmos throughput 731.74 -> 712.86, TPOT mean 40.07 -> 43.69,
E2E mean 3156.11 -> 3226.93. One run each; no all-metric benefit is established.

Cosmos D dispatches rose 76 -> 89, mean D batch fell 31.58 -> 26.97, and request-weighted ready wait rose
19.70 -> 22.26 ms. More eager D dispatch does not necessarily improve subsequent cohort service. This is
another reason to avoid promoting a local or global hard priority merely because its target metric sounds right.

## Revision 3: release-contract mismatch

Code audit found `nearReclaimBytes = visionPayloadBytes(prefillRequestIds)` in the coordinator's global
memory-horizon supplier. That awards total logical payload bytes even where `releasePrefillStorage()`
returns zero because unsplit M-RoPE retains the complete slab through decode. It also includes positional
bytes in split-lease paths although those positional bytes survive final prefill. This is a scoring/reference
mismatch, not evidence of an early physical free or a corrupted KV cache.

The `ownership` experimental mode changes only the release-potential value: query a payload method that
matches its actual release contract, returning zero for unsplit positional storage and only prefill-view bytes
otherwise. Existing global scheduling stays intact, with revision 2 continuity protection disabled.
Gemma without positional storage is a control: old/new logical release values are equal. Cosmos's legacy
unsplit path is the affected case. Model names are never consulted by this decision.

This remains **logical final-prefill release potential**, not guaranteed physical free bytes after the current
chunk. Shared owners, partial chunks and idle caches require a richer transition interface before an exact
physical reclaim prediction can be claimed. Admission continues to use the existing physical owner budget.

### Revision 3 repeated result: accounting semantics improved, no demonstrated speedup

`ownership-3x` completed all 24 cells (two models, two workloads, two variants, three alternating-order
repeats). Each row below averages three per-run means/p95 values; it is not a pooled request percentile.
The service-continuity restriction is off in both variants. Same binary, engine, KV, calibration and traces.

| Model / workload | Variant | token/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---|---:|---:|---:|---:|
| Gemma mixed | Lifetime | 718.24 | 266.97 / 607.39 | 25.82 / 34.59 | 1456.48 / 2192.68 |
| Gemma mixed | Ownership | 724.40 | 250.14 / 558.20 | 26.18 / 34.53 | 1446.43 / 2187.34 |
| Gemma heavy | Lifetime | 551.18 | 366.05 / 665.01 | 31.52 / 44.18 | 1550.59 / 2174.09 |
| Gemma heavy | Ownership | 549.41 | 376.60 / 672.03 | 31.42 / 44.74 | 1554.42 / 2137.02 |
| Cosmos mixed | Lifetime | 1155.00 | 812.39 / 1998.54 | 34.34 / 62.35 | 2383.58 / 2515.56 |
| Cosmos mixed | Ownership | 1145.83 | 774.79 / 2000.02 | 35.42 / 62.58 | 2403.73 / 2535.60 |
| Cosmos heavy | Lifetime | 724.96 | 1517.44 / 3003.17 | 42.70 / 82.03 | 3181.70 / 3356.15 |
| Cosmos heavy | Ownership | 724.64 | 1492.83 / 3004.31 | 43.14 / 81.04 | 3174.87 / 3344.12 |

Throughput changes are +0.86%/-0.32%/-0.79%/-0.04% in table order. Gemma's logically equivalent control
also varies, so small improvements are not evidence of policy benefit. Cosmos mean TPOT does not recover.
Memory-accounting correctness and scheduling performance must remain separate claims. No default promotion.

The matched host decomposition confirms the missing benefit. Cosmos heavy ready wait is 23.147 ->
22.898 ms/token, host execution envelope 16.072 -> 16.562, completion-to-commit 3.453 -> 3.650.
D dispatches 71.33 -> 74.00 and mean D rows 33.66 -> 32.45. Correcting release credit did not restore
decode continuity. These are averages of request-weighted per-run components, not additive p95 values.

Activity mask `0000` (none of the instrumented E/P/D/Copy work intervals active) averages 1.81/1.79%
for Gemma mixed lifetime/ownership, 2.12/2.10% Gemma heavy, 2.87/3.07% Cosmos mixed and 3.19/3.03%
Cosmos heavy. Low all-stream idle does not imply individual D requests receive timely service; it is not
an SM-occupancy measurement. Peak GPU memory averages 9384/9386, 9389/9389, 9728/9740 and 9851/9815 MiB
respectively. No KV capacity or precision changed and all 24 cells completed without OOM.

Correctness qualification: Gemma mixed lifetime has one token-trace hash across three runs, ownership has
three; both Gemma heavy variants have three. Both Cosmos workloads/variants have one per variant. Thus
request/output-count completion is established, but Gemma exact-repeat promotion is **not** passed.
Timing/formation/FP16 sensitivity is a hypothesis, not an excuse to declare ownership correctness proven.

Send-based E2E excludes client-cap wait. Gemma lifetime scheduled-arrival E2E mean/p95 is 2691.98/4006.54
ms mixed and 2962.62/4366.57 ms heavy; ownership 2672.33/3957.45 and 2953.84/4376.21. Cosmos's 64-request
in-flight contract has negligible send-delay difference here. Do not compare these arrival-relative values
to send-relative frozen vLLM columns.

## Revision 4 diagnostic: configured P128 does not mean identical vision granularity

The Cosmos engine has a text P128 profile and an auxiliary vision P1024 profile. The benchmark composition
root intentionally makes vision prefill atomic when that auxiliary row limit is larger. Gemma's selected
engine has P128 chunking. Thus all preceding `P128` contracts specify **text** chunk size; they do not imply
equal vision-prefill granularity across models. Within each paired admission comparison, this setting was
unchanged. Production `PhaseServingRuntime` also has a separate explicit chunking capability flag.

This is a documented historical performance choice (notes 243 and 298), not a newly discovered engine bug.
The builder's dynamic minimums allow shorter chunks, but execution still uses the selected auxiliary profile
and its mask/carrier limits. Earlier narrow-engine versus atomic-engine results changed more than one factor.

The `chunked` diagnostic enables existing vision chunking only at the drained measurement boundary, keeping
generic calibration, binary, engine, KV and lifetime admission identical. Cosmos tests actual atomic versus
128-token chunking; Gemma is an identity-mode control. This is a fixed-granularity causal screen, **not** an
adaptive-chunk performance claim. It changes the downstream runtime contract, so historical vLLM remains
a serving reference, not a matched intervention baseline. Full greedy identity and memory safety validation
are required before promoting a changed vision path.

### Revision 4 result: chunking reduces one gap but increases total work

`chunked-screen` completed 8/8 cells, one run per cell. Within each model the pair uses the same binary,
engine, calibration and lifetime admission. Gemma already chunks, so its pair is an identity-mode timing
control, not evidence that a second implementation of chunking helps or hurts. Times in milliseconds.

| Model / workload | Variant | token/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 |
|---|---|---:|---:|---:|---:|
| Gemma mixed | Existing chunk128 | 735.57 | 241.42 / 556.68 | 25.81 / 33.58 | 1422.28 / 2137.89 |
| Gemma mixed | Force chunk128 | 718.15 | 260.57 / 560.45 | 26.01 / 34.71 | 1456.26 / 2206.55 |
| Gemma heavy | Existing chunk128 | 556.51 | 366.41 / 721.99 | 31.69 / 45.81 | 1552.30 / 2133.83 |
| Gemma heavy | Force chunk128 | 548.67 | 373.19 / 734.27 | 31.76 / 46.39 | 1566.51 / 2181.69 |
| Cosmos mixed | Existing atomic | 1162.46 | 670.17 / 1942.56 | 37.55 / 57.74 | 2368.56 / 2491.52 |
| Cosmos mixed | Force chunk128 | 988.65 | 1030.02 / 2476.07 | 37.84 / 62.09 | 2792.46 / 2895.21 |
| Cosmos heavy | Existing atomic | 721.79 | 1586.96 / 2978.69 | 41.10 / 78.57 | 3207.39 / 3363.42 |
| Cosmos heavy | Force chunk128 | 638.55 | 1752.22 / 3416.79 | 45.74 / 72.98 | 3548.17 / 3769.84 |

Cosmos throughput falls 14.95% mixed and 11.53% heavy. The heavy TPOT p95 improves, but mean TPOT and
both E2E metrics worsen. This is not a useful broad promotion. No wider sweep is justified by this screen.

| Cosmos mechanism | Mixed atomic | Mixed chunk128 | Heavy atomic | Heavy chunk128 |
|---|---:|---:|---:|---:|
| P dispatches | 33 | 72 | 46 | 89 |
| P cumulative GPU ms | 1236.85 | 1638.53 | 1735.47 | 2204.20 |
| D dispatches | 75 | 101 | 78 | 104 |
| D mean rows | 38.19 | 28.36 | 30.77 | 23.08 |
| D cumulative GPU ms | 1305.82 | 1373.95 | 1059.13 | 1528.98 |
| D ready wait, request-weighted ms | 12.965 | 10.187 | 19.378 | 17.144 |
| D host execution envelope ms | 20.746 | 20.846 | 16.633 | 21.632 |
| D done-to-commit ms | 3.786 | 6.742 | 5.128 | 6.927 |

Chunking does reduce the ready queue interval. However, more P dispatches, smaller D cohorts and extra
completion/collection delay consume the benefit. The data separate **waiting interval** from **total token
cycle**, and do not support attributing the entire change to one attention kernel or to memory fragmentation.
The existing auxiliary profile remains in use; this is not an optimized narrow-profile chunking experiment.

Cosmos atomic/chunked outputs match exactly for all 64 requests in each workload (2928/2928 mixed and
2464/2464 heavy positional tokens). This single pair is not a sanitizer or repeated correctness proof.
Gemma identity-mode pairs match 60/64 requests mixed and 62/64 heavy (2818/2928 and 2440/2464 tokens).
Its cross-schedule exact-output gate remains open independently of the policy-performance conclusions.

## Final cleanup and promotion status

The unsuccessful local E yield had already been removed. The global D-continuity restriction, forwarding
methods, config flag, metric, two experiment-only tests and runner's `service` option are also removed after
the completed screens. Their exact executed patches and runner versions remain in each result manifest's
`source.patch` and `runner-source.py`; rerunning those historical cells requires those revisions, not the
cleaned current binary. Existing V3 recovery remains unchanged.

Retained: lifetime physical-byte admission, release-contract query and opt-in ownership scoring, request-level
decode decomposition, and a drained-boundary vision-chunk diagnostic. No new default D priority, model-name
branch, TTFT/TPOT target, batch-count constant, mandatory RLS fit or KV reduction. Rebuild and post-cleanup
HTTP smoke are recorded separately from the 42 pre-cleanup performance cells, never pooled as one binary.

The release credit is a late comparator after service compression (`PhaseGlobalScheduler::select`), and
automatic admission bytes are not yet the same physical ledger as the global memory-horizon supplier.
Therefore a corrected release value alone cannot be described as a complete service-aware memory scheduler.
See note 304 for the architectural boundary and next decision criteria.

### Post-cleanup verification

Final binary SHA256: `4bccbf3620a834abbc36f8701e61a43e13586d0f18c9c4eb0459393e80a8b680`.
`final-cleanup-smoke` completed 4/4 HTTP cells with zero reported failures. 261 C++ tests in five scheduling,
memory and server groups and 11 Python contracts pass. The C++ count is two lower because the two removed
priority-rule tests no longer describe any supported policy. Model/engine/KV identities remain unchanged.

| Lifetime, one smoke run | token/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms | Peak MiB |
|---|---:|---:|---:|---:|---:|
| Gemma mixed | 720.76 | 258.84 / 562.66 | 26.07 / 34.41 | 1455.31 / 2190.60 | 9385 |
| Gemma heavy | 547.86 | 370.90 / 661.58 | 31.54 / 43.93 | 1553.49 / 2146.68 | 9389 |
| Cosmos mixed | 1158.22 | 771.88 / 2019.09 | 34.82 / 63.35 | 2372.20 / 2494.26 | 9723 |
| Cosmos heavy | 730.37 | 1539.84 / 3027.21 | 40.95 / 81.86 | 3151.45 / 3323.42 | 9861 |

These confirm execution after cleanup; they are not a new 3x performance campaign or proof of exact output
identity. No full12/sanitizer run was performed in this turn. Failed-policy source/runner patches are retained,
not engines or a second runtime implementation. No artifacts other than explicitly permitted closed-log
compression were deleted. No commits or pushes were made.

The stage analyzer now reports unavailable decode-age telemetry as no observations (`mean=null`), rather
than silently treating a removed field as a zero-age sample. That reporting correction does not change any
serving metric or the request-matched cycle decomposition.
