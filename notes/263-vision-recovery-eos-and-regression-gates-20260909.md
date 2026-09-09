# 263. Vision memory recovery: corrected calibration, EOS identity, and regression attribution

## Scope and decision

Continues [262](262-validated-v0101-full12-kv-and-engine-control-20260909.md). The primary question is whether the existing explicit-FP32-erf exact-GELU engine recovers the old vision efficiency under the corrected Release and calibration contract. No KV reduction, new RLS model, workload-specific scheduler rule, engine deletion, or text-engine rebuild is involved.

The four-case V1 screen completed 12 HTTP runs (three repeats each). The candidate is **not promoted**: memory improves but latency is not uniformly non-regressing, and full fixed-output greedy identity differs. This failed prerequisite means a second 108-run promotion suite would not establish a pass. The previous 108-run native baseline remains authoritative, not replaced by this partial candidate screen.

Artifacts: root .local/results/v0101-forward-port/vision-recovery-20260909/.
Exact commands, remapped input hashes and per-run validated warmup receipts accompany erf-screen/. The source tree remains the authorized .local/v0101-forward-port worktree; the clean upstream checkout is not edited.

## What changed this turn

- replay_retained_policy_commands.py: explicit --respect-eos removes both client --ignore-eos and Docker TRT_EDGELLM_IGNORE_EOS. Setting the latter to 0 would not disable it, since the runtime checks existence.
- --dispatch-telemetry emits existing per-dispatch metrics into unique policy/case/run paths. It is diagnostic-only and changes instrumentation overhead; these runs are not used as headline throughput results.
- compare_engine_outputs.py compares full token streams and streams through the first EOS separately. It rejects unsuccessful/incomplete token capture and different request membership. First-EOS agreement never changes the strict full-token verdict.
- Tests cover EOS configuration, nonmutation, membership rejection, and post-EOS differences that must remain strict failures.
- CMake formatting was normalized by the repository hook. No new runtime binary or scheduler semantic change was made this turn.

## Compiled graph explanation

| Engine | Device memory | Compiled layers | Boundary |
|---|---:|---:|---|
| Old direct engine | 444873728 bytes | 368 | Direct TensorRT graph, exact erf merger |
| Native exact ONNX | 574899200 bytes | 392 | Native ONNX Gelu, exact |
| Explicit erf ONNX | 449072128 bytes | 370 | FP32 erf arithmetic, FP16 public merger boundaries |

The native→erf difference is about120 MiB, with only about4 MiB left relative to the old direct engine. The retained inspectors and note250 show the changed fusion boundary; the old engine lacks detailed tactic IDs, so an exact per-tactic diff cannot be reconstructed.

This candidate was already exported, checked, built and inferred in note250. That older serving comparison preceded the corrected build/warmup contract, so it was remeasured here. We did not need another copy of weights or another engine build to test the hypothesis.

The KV pool remains FP16 3584 MiB, page128×256, stable slots80; P8/D64/E4 and text chunk128 remain unchanged. Both variants use the same Release binary/plugin and text engine recorded in note262. The GPU memory reduction is not attributable to smaller KV.

## Four-workload serving comparison

Current native and erf have three repeats each. Old v0.10.0 V1 is retained historical data (one repeat); frozen vLLM follows note262's retained contract. Means below are arithmetic run means; p95 is median of run p95, not a pooled quantile. A slow repeat can make mean exceed median-of-p95.

| Workload | 비교 대상 | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|
| mixed | old V1 | 1185.12 | 692.51/1949.30 | 30.39/39.54 | 2123.84/2404.71 |
| mixed | native V1 | 1094.38 | 739.86/2168.80 | 34.38/52.47 | 2344.74/2609.23 |
| mixed | erf V1 | 1096.89 | 713.40/2114.00 | 35.84/48.53 | 2364.31/2593.99 |
| mixed | frozen vLLM | 921.48 | 874.56/2541.43 | 46.97/84.02 | 3008.38/3140.85 |
| multi-image | old V1 | 312.12 | 267.92/304.71 | 7.71/8.62 | 506.92/512.37 |
| multi-image | native V1 | 308.13 | 291.20/306.81 | 8.46/11.02 | 553.53/518.58 |
| multi-image | erf V1 | 301.29 | 328.15/318.93 | 8.46/9.47 | 590.52/530.67 |
| multi-image | frozen vLLM | 244.52 | 259.81/402.58 | 12.42/16.32 | 644.30/653.90 |
| text-heavy | old V1 | 2131.93 | 368.38/1047.42 | 21.08/33.30 | 1471.99/1572.85 |
| text-heavy | native V1 | 1970.43 | 342.26/1108.72 | 24.58/36.11 | 1612.53/1699.68 |
| text-heavy | erf V1 | 2023.10 | 329.39/875.91 | 23.93/38.31 | 1560.93/1659.05 |
| text-heavy | frozen vLLM | 1634.76 | 421.58/1232.20 | 29.21/47.36 | 1943.42/2037.81 |
| vision-heavy | old V1 | 687.95 | 1334.86/2890.15 | 41.45/65.29 | 2955.64/3402.72 |
| vision-heavy | native V1 | 665.22 | 1397.07/3166.65 | 36.86/53.75 | 2863.45/3620.04 |
| vision-heavy | erf V1 | 668.72 | 1403.95/3182.50 | 40.17/59.48 | 2998.09/3617.49 |
| vision-heavy | frozen vLLM | 579.20 | 1710.70/3691.37 | 63.70/119.58 | 4119.14/4229.37 |

### Interpretation

- mixed: +0.23% throughput; TTFT and TPOT p95 improve, but TPOT mean and E2E mean do not. No universal recovery.
- text-heavy: +2.67% throughput and lower mean TTFT/TPOT/E2E; TPOT p95 increases (36.11→38.31ms).
- vision-heavy: +0.53% throughput but TPOT mean/p95 increase (36.86/53.75→40.17/59.48ms), and E2E mean increases.
- multi-image: -2.22% median throughput; repeats198.96/311.96/301.29 token/s. The slow first repeat yields mean E2E590.52ms. Excluding it would hide the instability.
- All four candidate median throughputs exceed frozen vLLM. This does not prove a statistically significant win or a same-time controlled comparison.
- Candidate repeat token hashes are stable within each of the four cases. Cross-engine full-token identity is a separate gate.
- Sampled peak run medians: mixed9451, multi-image9463, text-heavy9477, vision-heavy9529 MiB.

## Correctness contracts

The multi-image fixed32-token comparison initially found 3/5 full request matches versus native, but 5/5 through-EOS matches, for each of three candidate repeats. Request0 first differs at token30, request2 at token14 (zero-based), after their natural answer terminators. JSON: erf-output-identity.json. These are comparisons of captured fixed-length streams, not proof that actual early-termination execution has identical outputs.

Actual EOS-termination runs and diagnostic dispatch attribution are recorded below. They are kept separate from the ignoreEOS throughput contract. A diagnostic first-EOS match does not waive the existing exact-output promotion gate.

## Actual EOS termination: completed

All three engines completed three actual early-termination runs. All nine runs produced the same five request outputs, 36 captured tokens total/run, hash 3a1dd485b9f55dffbfe09f1b57bb0aa941b3b9428b46ffb023c03527f7453443. This is a small-trace correctness pass, not all-model semantic validation. No frozen vLLM comparison is made under this changed output contract.

| Engine | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|
| native | 290.82/309.96 | 13.53/25.69 | 351.13/384.87 |
| erf | 275.80/324.29 | 17.74/27.61 | 349.03/399.24 |
| old | 259.70/287.40 | 14.47/23.65 | 321.82/362.77 |

## Compact dispatch diagnosis: one repeat each, not headline performance

| Workload | Engine | E count/meanBS/GPU sum ms | P count/meanBS/GPU sum ms | D count/meanBS/GPU sum ms | D host-gap mean/p95 ms |
|---|---|---:|---:|---:|---:|
| mixed | native | 10/3.20/1304.44 | 35/1.94/1285.04 | 92/31.13/888.63 | 17.10/132.70 |
| text-heavy | native | 6/2.67/816.94 | 23/3.13/601.97 | 75/44.37/688.20 | 11.31/57.88 |
| vision-heavy | native | 16/3.00/2291.34 | 45/1.47/1520.33 | 100/24.00/901.26 | 25.47/180.18 |
| mixed | erf | 10/3.20/1265.15 | 35/1.94/1171.93 | 89/32.18/838.80 | 17.53/138.38 |
| text-heavy | erf | 6/2.67/667.88 | 24/3.00/955.72 | 84/39.62/800.97 | 9.14/23.33 |
| vision-heavy | erf | 18/2.67/1803.80 | 45/1.47/1671.24 | 111/21.62/1168.58 | 17.70/131.67 |

### Interpretation of compact telemetry

GPU spans can overlap; E+P+D sums are not wall time or utilization. Host gaps are completion-observation to next dispatch, not CUDA-complete to first kernel.

The vision-heavy diagnostic saved about487.54ms of E span, but P added150.91ms and D added267.32ms. D cohort fell24.00→21.62 and dispatches rose100→111. E also fragmented16→18 batches. This is a concrete execution/formation interaction candidate, not proof from a single instrumented trajectory of the production run's causal decomposition.

Mixed preserved E/P counts, with D92→89; text-heavy had faster E but P23→24 and D75→84. A faster E implementation changes more than isolated E service: subsequent queue readiness and overlap placement change. Different instrumentation levels and individual trajectories must not be pooled.

## Vision-heavy request-ready audit

A separate audit run per engine joined request-local sampling tickets and next decode starts. Both had2464 producer tickets and2400 next-decode transitions. Resident-decode producers account for2336 nonterminal next-decode transitions.

| Resident decode path | Native mean/p95 ms | Erf mean/p95 ms |
|---|---:|---:|
| Sampling submission→CPU handling | 2.159/0.399 | 1.729/5.228 |
| Handling→collect | 0.00113/0.00168 | 0.00089/0.00122 |
| Collect→token commit | 0.16449/0.01920 | 0.01360/0.02543 |
| Commit→ready | 0.00030/0.00051 | 0.00032/0.00066 |
| Ready→next D start | 28.592/192.587 | 33.244/196.133 |
| E host-span coverage while ready | 24.321/192.537 | 26.424/191.759 |
| P host-span coverage while ready | 14.158/86.435 | 16.026/85.770 |
| Uncovered host span while ready | 0.170/0.407 | 0.183/0.531 |

E/P coverage overlaps and must not be added. Host spans are not GPU utilization, and their complement is not a direct CPU-overhead measurement. Sampling submission→handling includes asynchronous waiting and is not the duration of the sampling kernel. Rare outliers can make means exceed p95.

For first D following prefill, ready→D mean/p95 was158.51/605.03ms native versus87.49/247.81ms erf. Thus first-token progress and resident decode continuity can move in opposite directions even in the same diagnostic. This motivates protecting both, not a workload-specific preference.

The audit did not enable a common GPU activity epoch. The older analyzer correctly failed its GPU-start/end analysis on missing gpu_start_us; no host timestamp was substituted. Only its existing analyze_ready_path host-clock join was used. Full retained summary: ready-path-host-summary.json. Absolute GPU idle/overlap duration remains outside this audit's evidence.

## Validation and promotion decision

- 29 new HTTP runs completed: four-case screen12, actual EOS9, compact diagnosis6, request audit2. Every warmup response set passed the guard.
- C++ unitTestRuntime:323 scheduled,322 passed,1 optional metadata benchmark skipped. An initial invocation of unitTestRuntimeState matched0 tests and is explicitly not a pass.
- Python replay/output/dispatch tests:10 passed.
- Changed-file pre-commit passed for benchmark scripts/tests and C++ files. CMake formatting hook changed whitespace and passed on rerun. This is not a claim of whole-repository --all-files completion.
- Existing engine source/export/build provenance remains notes249–250. No new model export or engine build occurred this turn.
- Normal-EOS multi-image cross-engine/repeat identity passed; fixed-output cross-engine identity did not. These gates remain separate.
- No candidate default was changed: erf saves memory but fails uniform latency promotion; old engine is faster in this small EOS trace but lacks broad numerical/shape validation.
- The final108-run native suite in note262 remains the reference. A new candidate full12/V0/V1/V2 gate is deliberately not called complete, since the prerequisite four-case no-regression screen already failed.

### Remaining implementation target

Prioritize a controlled resident-D continuity experiment under E/P execution, preserving first-token progress and canonical candidate formation. Do not rework the KV pool to explain this result. Compare current Scalar selection with deterministic request-slack protection on the same observable frontier; collect action/ready membership so a change in formation cannot masquerade as a pure selector gain. First prove a non-regressing candidate on these four cases, then rerun all12×V0/V1/V2×3. Frozen vLLM remains reusable only for the fixed-output contract, not these actual-EOS runs.

Unrestricted full E/P/D sanitizer and full cross-shape exact identity remain promotion requirements; a passing isolated encoder memcheck alone would not satisfy them.

### Sanitizer attempt this turn

An isolated native/erf encoder-pair memcheck used CUDA13.3 Compute Sanitizer with no force-synchronization-limit, real checkpoint weights, synthetic fixed512-patch inputs, E1/E2/E4, one warmup and one measured iteration each. The process was bounded by timeout120s. It exited124 before completing the matrix or emitting an error summary. One instrumented native E1 measurement took24386ms; this is tool overhead, not serving performance. encoder-memcheck.log and the retained command describe the attempt. **No sanitizer pass is claimed.** The earlier bounded full-serving test in note247 is a distinct result, not a substitute for unrestricted E/P/D coverage.
