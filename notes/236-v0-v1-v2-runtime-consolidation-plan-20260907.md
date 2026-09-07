# V0/V1/V2 Runtime Consolidation Plan

## Scope

The production phase scheduler retains exactly three policy variants over one
shared execution mechanism:

| Variant | Exact CUDA registry | Contextual Scalar RLS | Scalar H=2 transition |
|---|---:|---:|---:|
| V0 Exact | yes | no | no |
| V1 Scalar | yes | yes | no |
| V2 Scalar+T | yes | yes | yes |

Zero-start and generic calibration remain initialization conditions rather than
additional policy variants.  Workload names are never policy inputs.

## Preserved mechanism

The cleanup must not alter independent E/P/D TensorRT contexts, E/P/D/copy CUDA
streams, stable indexed KV/page ownership, vision leases, packed and chunked
prefill, continuous admission, CUDA graphs, memory accounting, request DAG
transitions, CUDA timing, or activity telemetry.

## Removed experiments

The current branch removes Effect-Vector/Decomposed Scalar, Completion-Vector,
completion conformal authority, Oracle H1, selective-fidelity selection, and
their policy telemetry and benchmark drivers.  Their implementation remains
recoverable from branch `codex/v010-policy-experiments-archive` at commit
`4efcdae8fb5abfbe95f2d1b30606701fffb1a9d0`.  Historical result notes remain in
the repository as evidence and are not production dependencies.

## Canonical policy contract

The serving process accepts one policy setting:

```text
TRT_EDGELLM_PHASE_POLICY=exact|scalar|scalar-transition
```

The policy changes only cost authority and bounded transition evaluation.
Candidate formation, feasibility, SLO deadlines, memory constraints, canonical
row order, dispatch leases, and GPU execution are identical across variants.

## Validation contract

Every cleanup stage must pass build and targeted unit tests.  V0/V1/V2 must
preserve candidate membership, request row order, selected action identity, and
greedy output.  Major stages run balanced, long-prefill, 48.8 req/s saturation,
multi-image, and Poisson sentinels.  The final stage runs the complete 12-trace
gate, with a 3% median throughput and latency regression threshold.  A fresh
vLLM comparison is required only for the final binary because the workload and
vLLM implementation do not change during cleanup.

## Frozen baseline

The corrected V2 candidate is the baseline described in
`235-multi-image-slo-contract-and-frozen-branch-results-20260907.md`.  It is
ahead of the previous champion by 2.29% throughput geometric mean across the
12 traces and ahead of the frozen vLLM throughput by 17.99%.  These numbers are
engineering baselines, not final paper claims: most full-suite points are
single runs and the vLLM data is frozen.

## Implemented architecture

The production path is now intentionally narrow:

```text
HTTP / continuous admission
            |
            v
request DAG and stable ownership
  text: P -> D -> ... -> complete
  VLM : E -> P -> D -> ... -> complete
  KV page lease / vision payload lease
            |
            v
one immutable E/P/D ready snapshot
            |
            v
bounded candidate mechanism
  E, P, D, E+P, E+D, P+D, WAIT
  dependency / TRT shape / single-inflight / memory feasibility
            |
            v
canonical policy mode
  V0: exact CUDA cost
  V1: exact + contextual scalar cost
  V2: V1 + deterministic H=2 transition value
            |
            v
lexicographic selector
  feasibility -> SLO safety -> efficiency
            |
            v
independent E / P / D TensorRT contexts
and E / P / D / copy CUDA streams
            |
            v
CUDA event completion -> request transition -> next snapshot
```

The learned model does not own correctness. It predicts only normalized action
value. Dependency, ownership, TensorRT profiles, request transitions, row
membership, and memory safety remain deterministic mechanisms. V2 does not
learn arbitrary future arrivals: it projects only action-induced state changes
and already-visible completion boundaries.

The implementation boundary is:

```text
cpp/runtime/phase/policy/phasePolicyMode.h
  canonical V0/V1/V2 enum and predicates

cpp/runtime/phase/cost/phaseRuntimeCostTracker.h
cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp
  exact registry and contextual scalar authority

cpp/runtime/phase/policy/phaseFormationPlanner.h
cpp/runtime/scheduling/phaseFormationPlanner.cpp
  deterministic bounded transition projection used only by V2

cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}
  common P/D candidate construction and global selection

cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}
  E/P/D DAG transitions, vision ownership, and V2 transition integration

examples/llm/llm_phase_context_smoke.cpp
  one public environment contract and HTTP-serving integration

benchmarks/phase_serving/run_policy_warmup_matrix.py
  same-binary V0/V1/V2 evaluation harness
```

## Cleanup result

The full experiment-rich implementation remains recoverable at archive branch
`codex/v010-policy-experiments-archive`, commit
`4efcdae8fb5abfbe95f2d1b30606701fffb1a9d0`. Relative to that archive, the
current worktree has:

```text
80 files changed
697 insertions
15,186 deletions
net reduction: 14,489 lines
```

Removed production-policy concepts include:

- Effect/Decomposed-Scalar and Completion-Vector authorities;
- completion conformal and selective-fidelity selection;
- offline Oracle-H1 policy and forced directional injection;
- global/contextual shadow modes and legacy selection-mode switches;
- latency-safe/balanced/throughput/long-prefill/auto serving presets;
- static overlap percentages and per-direction CUDA delay gates;
- external JSON injection for P/D, prefill-formation, and encoder costs;
- their schemas, analyzers, benchmark drivers, and unit tests.

Historical Markdown results are retained because they explain why the rejected
designs were rejected. They are not compiled or imported by the current
runtime. Global-disabled fallback code remains a mechanism compatibility path,
not a fourth production policy.

Three old local artifacts were also removed before the final measurement:

```text
.local/cosmos-reason2-2b/v091-fresh-20260820
.local/cosmos-reason2-2b/tied-gate-20260820
.local/clean-v010-build
```

Available filesystem space increased from about 65 MiB to about 12 GiB. The
current Cosmos model, current P128 engine, visual engine, vLLM environment, and
v0.10 build were preserved.

## Same-binary 12-workload comparison

All three variants used the same Cosmos-Reason2-2B model, engine, P128 chunk,
P8/D64/E4 limits, stable slots, admission, CUDA graphs, request traces, and
workload-agnostic generic calibration. The only policy setting changed was
`TRT_EDGELLM_PHASE_POLICY`. These are one-run engineering measurements; they
are not paper confidence intervals.

| workload | V0 tok/s | V1 tok/s | V2 tok/s | V2 vs V0 | V2 vs V1 | frozen vLLM tok/s | V2 vs vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 2514.5 | 2478.9 | 2495.5 | -0.76% | +0.67% | 1983.5 | +25.81% |
| balanced | 4562.7 | 4455.0 | 4454.1 | -2.38% | -0.02% | 4319.9 | +3.11% |
| decode-heavy | 5347.9 | 5319.2 | 5265.9 | -1.53% | -1.00% | 4854.3 | +8.48% |
| long-prefill | 1086.9 | 1220.4 | 1227.5 | +12.93% | +0.58% | 1120.9 | +9.51% |
| bimodal | 1888.0 | 1916.0 | 1952.6 | +3.42% | +1.91% | 1868.3 | +4.51% |
| text-heavy | 2108.4 | 2131.9 | 2113.6 | +0.25% | -0.86% | 1634.8 | +29.29% |
| mixed | 1125.0 | 1185.1 | 1168.9 | +3.90% | -1.37% | 921.5 | +26.85% |
| poisson | 2038.2 | 2065.8 | 1974.0 | -3.15% | -4.44% | 1800.1 | +9.66% |
| vision-heavy | 705.2 | 687.9 | 736.8 | +4.49% | +7.10% | 579.2 | +27.21% |
| wave-drain | 98.1 | 98.0 | 98.0 | -0.12% | -0.01% | 95.8 | +2.20% |
| late-vision | 2541.4 | 2552.3 | 2565.8 | +0.96% | +0.53% | 2359.2 | +8.76% |
| multi-image | 310.2 | 312.1 | 324.8 | +4.71% | +4.08% | 244.5 | +32.85% |

Summary of the one-run matrix:

- V2 versus V0 token-throughput geometric mean: `+1.81%`, 7/12 wins;
- V2 versus V1: `+0.56%`, 6/12 wins;
- V2 versus frozen vLLM: `+15.16%`, 12/12 wins;
- peak GPU memory: 9,237 MiB text-only and at most 9,665 MiB VLM;
- V0 is still the strongest low-variance baseline for balanced/decode-heavy;
- V1 exposes the value of scalar generalization, especially long-prefill;
- V2 is the most attractive research candidate, but is not a universal winner.

Replacing the four most variable VLM points with their later three-run medians
gives V2 `1176.2/730.9/98.0/317.4 tok/s` for mixed/vision-heavy/wave-drain/
multi-image. On that conservative table V2 is `-0.60%` geometric mean versus
the pre-cleanup P23 champion, every trace remains inside the `-3%` throughput
gate, and it remains `+14.93%` geometric mean and 12/12 ahead of frozen vLLM.
Thus the cleanup did not create a material throughput regression, but neither
did it make V2 consistently faster than the archived champion.

## Complete V0/V1/V2 latency table

The values below are `mean / p95` milliseconds. Lower is better.

| workload | variant | TTFT | TPOT | E2E | peak MiB |
|---|---|---:|---:|---:|---:|
| short | V0 | 83.0 / 169.9 | 13.62 / 26.81 | 328.6 / 407.4 | 9237 |
|  | V1 | 87.6 / 174.9 | 13.67 / 26.85 | 334.3 / 413.6 | 9237 |
|  | V2 | 87.1 / 176.6 | 13.37 / 26.05 | 328.4 / 408.4 | 9237 |
| balanced | V0 | 66.4 / 167.0 | 12.07 / 13.48 | 1095.7 / 1702.1 | 9237 |
|  | V1 | 67.2 / 170.2 | 12.34 / 13.87 | 1118.4 / 1739.7 | 9237 |
|  | V2 | 66.6 / 168.9 | 12.36 / 14.06 | 1117.9 / 1745.9 | 9237 |
| decode-heavy | V0 | 64.4 / 174.9 | 10.39 / 10.87 | 2751.8 / 4188.8 | 9237 |
|  | V1 | 63.6 / 174.5 | 10.46 / 11.11 | 2762.5 / 4223.6 | 9237 |
|  | V2 | 67.0 / 185.8 | 10.54 / 11.38 | 2787.8 / 4250.9 | 9237 |
| long-prefill | V0 | 2215.9 / 2913.5 | 30.16 / 36.72 | 4788.2 / 7025.0 | 9237 |
|  | V1 | 2059.2 / 2645.1 | 25.77 / 31.91 | 4249.5 / 6105.3 | 9237 |
|  | V2 | 2026.9 / 2790.7 | 25.73 / 30.43 | 4211.7 / 6063.9 | 9237 |
| bimodal | V0 | 1920.8 / 4219.1 | 18.78 / 29.41 | 4466.4 / 9312.8 | 9237 |
|  | V1 | 1950.3 / 4109.4 | 18.07 / 27.03 | 4420.6 / 9139.1 | 9237 |
|  | V2 | 1889.4 / 3983.0 | 18.00 / 28.91 | 4319.7 / 8955.6 | 9237 |
| text-heavy | V0 | 382.2 / 1011.9 | 21.17 / 34.36 | 1489.3 / 1598.5 | 9411 |
|  | V1 | 368.4 / 1047.4 | 21.08 / 33.30 | 1472.0 / 1572.9 | 9411 |
|  | V2 | 322.5 / 998.2 | 22.75 / 33.69 | 1488.2 / 1588.6 | 9399 |
| mixed | V0 | 644.1 / 2107.8 | 37.83 / 56.14 | 2357.6 / 2540.5 | 9395 |
|  | V1 | 692.5 / 1949.3 | 30.39 / 39.54 | 2123.8 / 2404.7 | 9483 |
|  | V2 | 696.5 / 2052.4 | 37.75 / 59.44 | 2378.4 / 2493.7 | 9395 |
| poisson | V0 | 179.6 / 671.4 | 21.31 / 40.07 | 1534.2 / 1991.7 | 9401 |
|  | V1 | 228.1 / 746.4 | 19.49 / 37.07 | 1475.5 / 1917.7 | 9387 |
|  | V2 | 256.9 / 817.4 | 20.31 / 35.77 | 1563.9 / 1987.7 | 9379 |
| vision-heavy | V0 | 1236.2 / 2819.1 | 35.29 / 57.30 | 2635.7 / 3249.5 | 9427 |
|  | V1 | 1334.9 / 2890.2 | 41.45 / 65.29 | 2955.6 / 3402.7 | 9547 |
|  | V2 | 1383.7 / 3036.9 | 43.03 / 82.37 | 3052.2 / 3318.8 | 9665 |
| wave-drain | V0 | 243.6 / 293.2 | 7.92 / 9.12 | 489.3 / 501.1 | 9395 |
|  | V1 | 243.3 / 297.1 | 8.01 / 11.56 | 491.7 / 504.8 | 9405 |
|  | V2 | 257.9 / 303.8 | 7.70 / 9.52 | 496.7 / 513.1 | 9389 |
| late-vision | V0 | 108.9 / 417.5 | 9.26 / 9.31 | 1435.6 / 1813.8 | 9387 |
|  | V1 | 120.5 / 425.5 | 9.20 / 9.25 | 1438.3 / 1803.9 | 9445 |
|  | V2 | 120.0 / 442.2 | 9.19 / 9.26 | 1436.4 / 1797.4 | 9387 |
| multi-image | V0 | 269.9 / 308.7 | 7.74 / 8.64 | 509.9 / 515.4 | 9379 |
|  | V1 | 267.9 / 304.7 | 7.71 / 8.62 | 506.9 / 512.4 | 9433 |
|  | V2 | 258.2 / 284.2 | 7.37 / 7.88 | 486.7 / 492.1 | 9395 |

Frozen vLLM reports medians rather than means. Under that like-for-like
median/p95 comparison, V2 has lower E2E median and p95 on all 12 traces. V2
also has lower TTFT p95 on all 12, lower TPOT median on 11/12, and lower TPOT
p95 on 9/12. The exceptions are important: mixed/vision-heavy/wave-drain show
that higher throughput or lower E2E does not imply every latency component is
better.

## Correctness and variance gate

Within each one-run result, every request completed and each aggregate marked
its token trace deterministic. Cross-policy hashes agreed for 8/12 traces.
They differed for mixed, vision-heavy, wave-drain, and multi-image. A separate
V2 three-run check found:

| workload | V2 tok/s samples | range | repeated token hash |
|---|---|---:|:---:|
| mixed | 1192.5, 1172.5, 1214.7 | 3.60% | no |
| vision-heavy | 679.6, 708.3, 672.8 | 5.28% | no |
| wave-drain | 98.06, 98.05, 98.04 | 0.02% | no |
| multi-image | 242.5, 203.9, 322.9 | 58.36% | yes |

The first divergent token was position 14 in the affected requests. The
divergence is therefore consistent with FP16 batch/row execution-order
sensitivity rather than missing tokens, ownership corruption, or an HTTP
contract failure. That distinction is useful for research correctness, but the
stricter production exact-output promotion gate still fails.

Multi-image is the opposite failure: token identity is stable, while execution
trajectory and performance are not. Its two non-converged calibration runs
were slow and the converged run was fast. A workload-agnostic bounded
calibration option was added to the harness. It repeats the same generic trace
only when the runtime reports non-convergence and stops immediately on
convergence. Three subsequent converged runs produced `309.3, 323.2, 317.4
tok/s`, reducing the range to `4.49%`. This is a useful stabilization, but not
a complete solution: the runtime's current convergence flag can still be true
for different required direction sets, and mixed/vision-heavy hashes remain
execution-order sensitive.

## Validation completed

```text
CMake configure in TensorRT 26.06 container: PASS
Targets llm_phase_context_smoke and unitTest: PASS
Focused C++ tests: 245/245 PASS
Focused Python harness/analysis tests: 21/21 PASS
Bounded-calibration subset: 13/13 PASS (included above)
git diff --check: PASS
phase event JSON schema parse: PASS
```

## Decision and remaining work

The cleanup itself is acceptable: it removes 14.5k net lines while preserving
the pre-cleanup champion within the 3% throughput gate and retaining a broad
lead over the unchanged frozen vLLM runtime. V2 remains the primary research
candidate, V0 is the conservative mechanism/cost baseline, and V1 is the
necessary ablation isolating contextual generalization.

The runtime should not yet be called production-promoted. The next work is
narrow and no new policy family is needed:

1. make calibration authority require a stable, declared direction frontier
   across consecutive full generic rounds rather than a per-round dynamic
   required set;
2. record a canonical per-request row/binding signature and isolate the token-14
   branch sensitivity from request ownership correctness;
3. stabilize multi-image E/P/D trajectory with the same V2 model and bounded
   transition, without workload-name conditions;
4. rerun only the four failing VLM traces until exact/variance gates pass, then
   rerun the 12-workload V2 gate;
5. rerun fresh vLLM only if the request/output contract changes; otherwise the
   frozen 3-run baseline remains the fair comparator.

No Completion-Vector, Decomposed-Scalar, selective policy, external cost
registry, workload label, or static per-trace profile should be reintroduced to
solve these gates.
