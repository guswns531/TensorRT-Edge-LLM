# 265. Decode guard: candidate restoration versus selection and realization

## Outcome

Continues note264 without another policy change. Four diagnostic HTTP executions completed: mixed and vision-heavy, guard OFF/ON, one fresh process each. These runs explain mechanisms; they do not replace note264's three-repeat non-instrumented performance screen. Default remains OFF.

The restored standalone D candidate was selected zero times out of23 restored candidates. All23 local decisions used `all_late_efficiency_recovery`. Nevertheless16 final actions included D through P+D or E+D. Thus candidate restoration, standalone-D selection, and actual D service must be distinguished.

## Implementation

- `cpp/runtime/phase/policy/phaseGlobalScheduler.h`: optional `PhaseDecodeGuardAudit` with P/D expiry and candidate restored/suppressed flags. These flags describe candidate formation, not ranking or GPU completion.
- `cpp/runtime/scheduling/phaseQueueScheduler.cpp`: fill the guard audit only when an audit is requested, at the actual standalone-D candidate gate.
- `cpp/runtime/scheduling/phaseThreeCoordinator.cpp`: carry the local audit into the enclosing E/P/D selector audit. Direct P/D decisions retain their own audit.
- `examples/llm/llm_phase_context_smoke.cpp`: serialize `decode_guard_audit` alongside existing actual local/final selector inputs, independently of preview-frontier unions. Existing unrelated compact-telemetry edits remain preserved.
- `benchmarks/phase_serving/analyze_selector_audit.py`: count restoration, local selection, final selection and dispatch kinds joined by decision ID. Reject a restored candidate missing from actual local inputs. Unknown audits remain explicitly unavailable rather than counted as false.
- `benchmarks/phase_serving/analyze_encoder_serving_pair.py`: explicit `--host-ready-only` enables the existing request-local host-clock analysis without requiring GPU epoch timestamps. No inferred GPU timestamps are substituted.
- C++ tests verify guard flags and request/action equality on copied scheduler states. Python tests distinguish local-D selection from final encoder dispatch and reject malformed restored-candidate evidence.

A test initially compared two sequential previews on the same scheduler and failed. Existing preview selection can update safe-probe history, so this was not an equal-state A/B. The test now clones the scheduler before comparing audit ON/OFF. The runtime preview semantics were not changed or silently declared pure. Final C++ results:324 passed, one optional metadata benchmark skipped; Python11 passed.

## Experiment contract and provenance

Root: `.local/results/v0101-forward-port/decode-guard-audit-20260909/`.

Parent source commit `80e3e1d`, plus this diagnostic patch and previously preserved smoke telemetry changes. Release SM86 binary SHA256 `63c412fc9b32b2a1f80a7c0ef4461a301346cfc547ce3add4b38c66e5b86d102`. Plugin and native vision/text engines match note264. KV3584MiB, P8/D64/E4, chunk128, graph OFF, V1 scalar, legacy pair eligibility ON, outer P/D frontier OFF, fixed-output HTTP contract unchanged. Audit instrumentation is enabled, so these are not primary serving measurements.

Each run validates319 generic calibration responses, then64 measured requests. All four warmups passed. Mixed output2928 tokens/run and vision-heavy2464 tokens/run; OFF/ON full token hashes match for each workload. No engine rebuild, KV change or new model training target was introduced. Generic calibration input equality does not imply equal posterior: the guard option is also active during warmup.

Reproduce with note264 replay command plus `--phase-telemetry --telemetry-level audit --repeats 1`; add `--preserve-expired-decode-candidate` only for ON. Each output directory contains its exact command and input-contract manifests. Run `analyze_selector_audit.py` on the retained dispatch files and `analyze_encoder_serving_pair.py --host-ready-only --events FILE --output FILE`. Summaries are retained as `off-selector.json`, `on-selector.json`, `off-ready.json`, `on-ready.json`.

## Actual candidate/dispatch evidence

| Metric | mixed OFF | mixed ON | vision-heavy OFF | vision-heavy ON |
|---|---:|---:|---:|---:|
| Decision events | 121 | 110 | 166 | 171 |
| Guard audited | 121 | 110 | 166 | 170 |
| Both P/D deadlines expired | 20 | 7 | 25 | 16 |
| Standalone D suppressed by P guard | 31 | 25 | 40 | 27 |
| Standalone D restored | 0 | 7 | 0 | 16 |
| Restored standalone D selected locally | — | 0 | — | 0 |
| Restored → actual P+D dispatch | — | 2 | — | 7 |
| Restored → actual E+D dispatch | — | 4 | — | 3 |
| Restored → actual P dispatch | — | 1 | — | 6 |
| Restored decision lacking dispatch | — | 0 | — | 0 |

Suppression may exceed the both-expired count because P-expired/D-not-expired cases remain suppressed by design. The missing guard audit in one ON vision-heavy decision is reported as unavailable, not a negative encounter. Counts differ across trajectories and cannot be matched as if they were identical snapshots.

### Why restored D loses

All23 restored-candidate local audits report `all_late_efficiency_recovery`. `phaseGlobalScheduler.cpp` uses this branch when no candidate is deadline-safe and all remaining candidates share a nonzero known violation mask. Its first ranking key is service compression, then reclaim/lag and deterministic ID. It is not minimum additional SLO violation.

In mixed, four of seven restored D candidates had a lower predicted maximum violation than the chosen local action. In vision-heavy, seven of sixteen did. D was marked dominated in zero mixed and three vision-heavy cases. This is direct evidence of the objective being used, not proof that choosing D would improve serving.

Examples from `on-selector.json`:

- mixed decision9223372036854776313: D predicted violation629277.5us versus selected P+D663513.7us; local reason all-late efficiency, actual final E+D.
- vision-heavy decision9223372036854776365: D2427994.5us versus selected P2476861.8us; final P.

These values include existing deadline lateness; they are not GPU kernel durations. The mask groups protected kinds, so the branch name must not be interpreted as proof that every individual request has the same urgency.

No observed `post_select_override` or late final choice with an audited safe D occurred in the non-D subset. The problem observed here is not a hidden post-selection override. The local versus outer selection distinction remains important, but restored standalone D already loses locally in this sample.

## Request-local decode ready path

All times below are host-clock mean/p95 milliseconds. Resident transitions only:2800 mixed and2336 vision-heavy ready→next-D intervals per run. Sampling tables also include the final resident token; ready tables exclude requests that terminate. E/P/D coverage overlaps and must not be summed as exclusive utilization.

| Interval | mixed OFF | mixed ON | vision-heavy OFF | vision-heavy ON |
|---|---:|---:|---:|---:|
| Sampling submit→CPU handling | 2.963/5.067 | 2.140/6.852 | 3.246/6.247 | 3.689/8.097 |
| CPU handling→collect | 0.001/0.002 | 0.001/0.003 | 0.001/0.002 | 0.001/0.002 |
| Collect→commit | 0.014/0.022 | 0.014/0.026 | 0.232/0.032 | 0.018/0.033 |
| Ready→next D start | 21.471/170.909 | 18.079/112.798 | 24.001/177.886 | 17.580/121.951 |
| E host-span coverage while ready | 13.493/131.463 | 9.388/112.721 | 16.961/113.906 | 10.465/82.238 |
| P host-span coverage while ready | 9.811/72.173 | 10.884/74.981 | 10.814/71.465 | 9.195/41.577 |
| D host-span coverage while ready | 1.964/4.513 | 2.695/21.726 | 1.010/4.167 | 2.261/6.051 |
| Uncovered host span while ready | 0.237/0.476 | 0.212/0.468 | 0.182/0.507 | 0.154/0.351 |

Commit→ready mean is below0.001ms in all four runs. The uncovered interval is small compared with total ready wait, but it is not a direct measurement of scheduler CPU cost. Phase start→done spans include submission and completion visibility and do not establish SM occupancy, GPU idle, or causal blocking.

### Continuity versus cohort efficiency

Deduplicate `PHASE_TIMELINE decode_start` by dispatch index; each request in the batch emits a start, so raw event counts would overcount batches.

| Workload | OFF D dispatches / mean BS | ON D dispatches / mean BS |
|---|---:|---:|
| mixed | 89 / 32.18 | 86 / 33.30 |
| vision-heavy | 113 / 21.24 | 139 / 17.27 |

Vision-heavy has shorter request-ready waiting but more, smaller D executions. This is consistent with a continuity/formation trade-off; it does not prove a single-action causal mechanism because ON/OFF warmup and subsequent trajectories differ. Audit mode lacks the compact `PHASE_METRIC` records required by `summarize_compact_dispatch.py`; that tool rejects these files. Therefore no cumulative GPU-time comparison is claimed from this dataset.

## Serving metrics: diagnostic runs only

These single-run audit measurements are included for completeness, not promotion or comparison headlines. They visibly differ from the repeated primary screen, illustrating why the two evidence types must remain separate.

| Workload | Audit setting | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|
| mixed | OFF | 1086.17 | 753.69/2244.62 | 35.83/53.57 | 2413.24/2637.65 |
| mixed | ON | 1109.95 | 768.24/2262.88 | 38.13/65.50 | 2496.15/2616.68 |
| vision-heavy | OFF | 654.22 | 1411.27/3245.33 | 36.67/55.13 | 2860.50/3673.28 |
| vision-heavy | ON | 644.46 | 1439.42/3349.25 | 34.09/54.56 | 2818.62/3750.06 |

### Unchanged primary comparison versus frozen vLLM

Reuse note264's non-instrumented three-repeat results. No new vLLM run was necessary because the primary workload/model contract is unchanged. Do not compare its numbers directly to instrumentation-induced differences above.

| Workload | Primary variant | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|
| mixed | OFF | 1115.26 | 766.81/2149.76 | 36.17/54.72 | 2435.14/2580.39 |
| mixed | ON, rejected | 1096.20 | 778.05/2119.78 | 36.68/55.30 | 2454.73/2609.69 |
| mixed | frozen vLLM | 921.48 | 874.56/2541.43 | 46.97/84.02 | 3008.38/3140.85 |
| vision-heavy | OFF | 646.24 | 1407.91/3313.33 | 34.63/55.05 | 2814.33/3652.17 |
| vision-heavy | ON, rejected | 660.44 | 1426.07/3227.50 | 38.04/62.49 | 2952.84/3613.98 |
| vision-heavy | frozen vLLM | 579.20 | 1710.70/3691.37 | 63.70/119.58 | 4119.14/4229.37 |

## Next controlled step

1. Keep production OFF. Do not respond by forcing D unconditionally or adding workload labels.
2. Shadow-evaluate the all-late branch against minimum incremental protected delay on the same actual candidate inputs. Log which request class/phase pays the extra lateness, predicted compression sacrifice, and successor D cohort. This needs the full comparison inputs; current audit stores violation and domination, not every ranking term.
3. Preserve already-formed D cohort membership in the comparison. More frequent D dispatch alone is not an objective: vision-heavy shows a shorter wait with a21.24→17.27 batch reduction.
4. Separate generic warmup policy from measurement authority for a frozen-posterior diagnostic A/B if selection differences remain ambiguous. Keep normal end-to-end learning runs as the primary serving contract.
5. Only after a controlled branch shows benefit, rerun mixed/vision-heavy three times without audit overhead, then the full12 gate if latency and throughput pass. Frozen vLLM remains reusable under the same contract.

This turn localizes the decision rule and supplies the missing evidence. It does not establish that minimum-violation selection is universally better, does not complete a full12 revalidation, and does not claim a new performance improvement.
