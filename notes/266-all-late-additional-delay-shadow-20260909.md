# 266. All-late shadow: additional lateness versus service compression

## Contract

Continues note265. SLO values, actual selection policy, independent contexts, KV pool and engines are unchanged. This is an offline shadow ranking of the exact selector inputs captured during four diagnostic HTTP runs, not an Active replacement or a forced-branch experiment.

For protected completion r, let robust future completion be c_r (prediction plus nonnegative uncertainty), remaining slack s_r, and deadline guard g. The shadow measures:

`additional_r = max(0, c_r + g - s_r) - max(0, -s_r)`.

For nonnegative c_r and g, the equivalent implemented expression is `max(0, c_r + g - max(0, s_r))`. This avoids subtracting large accumulated lateness values. The candidate value is the maximum additional lateness across its protected completions. Nonfinite unbounded slack is ignored, as in the existing violation model. Candidates without explicit protected completions use the existing aggregate fallback.

Only all-late decisions are compared. Shadow selects minimum additional lateness, ties by higher existing service compression, then stable candidate ID. It uses only the existing hard-feasible, frontier-eligible, non-dominated candidates. It does not restore pruned candidates, invent future arrivals, or rerun preview methods. The actual selector continues to choose efficiency first under its existing all-late rule.

The measured quantity is not the difference between two maximum aggregate violations. Existing lateness is removed per protected completion before taking the maximum. SLO guard and uncertainty are retained. This is a hypothesis about useful ranking, not proof that it optimizes goodput.

## Implementation and tests

- `phaseGlobalScheduler.cpp`: compute additional lateness, exact selector compression and horizon only when an audit is requested.
- `phaseGlobalScheduler.h`: extend candidate audit with those values and primary/secondary batch sizes.
- `llm_phase_context_smoke.cpp`: serialize the new fields for both local P/D and final E/P/D inputs.
- `benchmarks/phase_serving/analyze_all_late_shadow.py`: rank without changing execution, retain actual/shadow candidate rows and predicted delay/efficiency differences.
- C++ tests cover already-expired aggregate slack, per-protected-completion uncertainty, and selection neutrality. Python tests enforce actual pruned-frontier membership, unchanged input authority, and nonfinite-input rejection.

Release build succeeded. C++325 passed, one optional metadata benchmark skipped; Python12 passed. Changed-file formatting checks passed. No new RLS outputs, weights, thresholds, SLO values or learned state are introduced.

## Provenance

Source parent `c583be8` plus this audit patch and pre-existing compact telemetry edits retained in the worktree. Binary SHA256 `ca42e909772ed8ae18537888058f53759bc78ef8d622d608eb82bca7159379b6`. Native vision/text engines and plugin match notes264–265. Artifact root: `.local/results/v0101-forward-port/all-late-shadow-20260909/`.

V1 scalar, P8/D64/E4, text chunk128, FP16 KV3584MiB, CUDA graph OFF, legacy pair eligibility ON, outer P/D frontier OFF. OFF/ON refers solely to the previously rejected standalone-D restoration option, not shadow authority. Both receive319 generic calibration requests in a fresh process. Detailed audit instrumentation remains outside the primary serving contract.

Reproduction: note265 replay command with new output roots; then `analyze_all_late_shadow.py EVENT_FILES --output OUTPUT.json`. Local P/D and final E/P/D are separate scopes and may refer to the same decision; do not sum them as independent actions.

## Limits

The stored batch sizes are the candidates' current cohorts, not predicted successor cohorts. Compression values compare the existing model's service-work/horizon ratio; they are not measured counterfactual GPU speedups. This shadow cannot establish actual next-D fragmentation, alternative completion timing or performance improvement. Those require subsequent controlled branch execution and repeated non-instrumented serving runs. Frozen vLLM comparisons remain note264's unchanged primary reference.

## Completed shadow comparison

All four diagnostic HTTP runs completed. All319 warmup responses succeeded per run. The64 measured requests generated2928 mixed or2464 vision-heavy tokens; all OFF/ON token hashes match their note265 references. Each setting has one diagnostic repeat; none is a new primary performance result. Full candidate records are retained in `off-shadow.json` and `on-shadow.json`.

The following means apply only to changed decisions. Additional-delay savings are model predictions in milliseconds; compression changes are mean relative changes in the model's work/horizon ratio, not measured throughput losses.

| Setting | Workload | Scope | All-late decisions | Shadow changes | Mean predicted added-delay saving ms | Mean compression change |
|---|---|---|---:|---:|---:|---:|
| OFF | mixed | local P/D | 20 | 0 | — | — |
| OFF | mixed | final E/P/D | 38 | 20 | 124.73 | -30.02% |
| OFF | vision-heavy | local P/D | 23 | 8 | 45.33 | -23.19% |
| OFF | vision-heavy | final E/P/D | 47 | 20 | 156.34 | -51.30% |
| ON | mixed | local P/D | 9 | 2 | 33.67 | -17.08% |
| ON | mixed | final E/P/D | 37 | 19 | 50.38 | -53.31% |
| ON | vision-heavy | local P/D | 16 | 9 | 27.42 | -30.05% |
| ON | vision-heavy | final E/P/D | 42 | 12 | 26.91 | -37.96% |

All recorded changes have strictly positive predicted added-delay savings; they are not merely different stable-ID tie breaks. Final changed actions remove D participation3/8/8/5 times in OFF mixed/OFF vision-heavy/ON mixed/ON vision-heavy, respectively. They add D participation1/0/3/5 times. Therefore this global maximum-additional-lateness objective is not synonymous with protecting resident decode continuity.

Examples of transition counts:

- OFF mixed final: P→E+P12 times, E+P→E2 times, and six other transitions once each.
- OFF vision-heavy final: P+D→E+P4, E→E+P4, P→E+P4, P→E3, P+D→P2, D→E2, P+D→D1.
- ON mixed local: P+D→D2.
- ON vision-heavy local: P+D→D4, P+D→P3, P→P+D1, P→D1.

The large shifts toward E/P work mean that replacing efficiency recovery globally could trade one type of lateness for another. It would be premature to enable this objective from disagreement counts alone.

### Low modeled efficiency-cost candidates

As a descriptive post-analysis slice, not a serving threshold, count changed final candidates retaining at least97% of the actual candidate's modeled compression. There are3 OFF mixed,1 OFF vision-heavy,1 ON mixed, and0 ON vision-heavy cases. This slice is not a newly introduced scheduler rule and does not prove those alternatives safe or faster.

One useful controlled-replay target is ON mixed decision9223372036854776328:

| Candidate | Cohort | Added lateness ms | Selection horizon ms | Compression |
|---|---|---:|---:|---:|
| Actual P | P2 | 46.821 | 43.963 | 1.1381 |
| Shadow P+D | P2 + D48 | 42.051 | 44.889 | 1.1146 |

Predicted added-delay improvement4.770ms; compression decreases about2.06%, and the current P cohort remains P2 while D48 gains service. The alternative was **not executed**, so neither successor cohort stability nor measured improvement is established. The absolute maximum violation actually rises795.760→802.295ms while incremental maximum lateness falls: removing accumulated lateness per protected completion changes the objective, rather than simply rescaling its old value.

## Diagnostic serving results

These executions use the existing actual policy, not the shadow winner. OFF/ON is the note264 restoration flag; differences cannot be attributed to activating the new objective because it was never activated.

| Workload | Audit setting | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|
| mixed | OFF | 1056.24 | 767.66/2243.30 | 40.89/59.55 | 2592.89/2725.50 |
| mixed | ON | 1084.03 | 725.19/2095.93 | 38.77/62.75 | 2476.72/2631.97 |
| vision-heavy | OFF | 663.22 | 1414.04/3269.86 | 40.32/61.48 | 3021.59/3647.80 |
| vision-heavy | ON | 668.64 | 1420.04/3291.53 | 39.83/65.59 | 3004.36/3610.78 |

### Retained primary baseline versus vLLM

Unchanged three-repeat note264 comparison, without audit instrumentation. No fresh vLLM or full12 run was performed this turn.

| Workload | Primary variant | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|
| mixed | Current OFF | 1115.26 | 766.81/2149.76 | 36.17/54.72 | 2435.14/2580.39 |
| mixed | frozen vLLM | 921.48 | 874.56/2541.43 | 46.97/84.02 | 3008.38/3140.85 |
| vision-heavy | Current OFF | 646.24 | 1407.91/3313.33 | 34.63/55.05 | 2814.33/3652.17 |
| vision-heavy | frozen vLLM | 579.20 | 1710.70/3691.37 | 63.70/119.58 | 4119.14/4229.37 |

## Decision and next step

Do not replace all-late efficiency recovery globally. The shadow finds real objective disagreements, but often at a large predicted efficiency sacrifice and with reduced D participation. Existing default remains unchanged; no performance promotion is claimed.

The next bounded task is repeated forced-branch comparison on a small number of identical snapshots such as P2 versus P2+D48, with request membership/ownership/frontier signatures equal before branching. Measure actual protected completion, D cohort sequence and equal-work horizon through reconvergence. Report per-kind protected delays, not only the global maximum. If identical replay cannot be established, treat new serving A/B as trajectory-level evidence rather than a same-state counterfactual. Only after this evidence supports a change should an Active variant enter the three-repeat latency/throughput gate and then full12.
