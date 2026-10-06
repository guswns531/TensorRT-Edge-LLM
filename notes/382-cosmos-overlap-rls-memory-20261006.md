<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Cosmos on A100: overlap, RLS quality, workspace memory and carried settings

This campaign tests the questions raised after note 381: whether E/P really overlaps, whether RLS
learns useful prices, whether workspace sharing reduces memory at an acceptable cost, and whether
settings carried from the user's Thor deployment penalize A100. Only A100 is measured here; no new
Thor execution or hardware identity claim is made for historical artifacts.

## Shared contract

Source/binary commit `dd773f05a7010634fa49e18543531bf9eff40c35`, clean measured code, same frozen runtime,
FP16 Cosmos checkpoint, LLM/vision engines and UUID traces as note 381. E8/P32/D256; KV 21 GiB;
encoded storage ceiling 2.5 GiB; managed memory limit 24 GiB. Same four workloads, concurrency 512,
1119-request generic calibration, no per-workload warmup, fixed output lengths ignoring EOS.
Memory limits do not include all weights/workspaces. No cleanup, default promotion, or quality-gate
waiver is performed. All comparisons require succeeded=requests, failed=0, and fixed output complete.

Artifact root: `.local/results/a100-cosmos-reason2-2b-fp16/capacity-resume/overlap-rls-memory/`.
`manifest.json` records configs, identities and evidence. Detailed CUDA event observation and full
telemetry are diagnostic only; performance runs use dispatch telemetry without the activity recorder.
RLS predictions remain active and process-local in each server. A server is reused across workloads
and repetitions, so knowledge carries forward in the recorded workload order.

## Measured phase overlap and RLS

The independent diagnostic uses `TRT_EDGELLM_PHASE_ACTIVITY_PREFIX` and full telemetry. Calibration
resets the activity recorder before measurement. CUDA event spans bracket phase engine execution;
they show concurrent phase intervals, not simultaneous kernel occupancy or an SM utilization profile.
`analyze_phase_activity.py` filters measured request lifecycles and produces intervals/segments.
Its phase totals include token sampling. A separate sweep of `encoder_engine` and P/D
`*_dispatch` intervals excludes sampling and verifies engine-only overlap.
No E+P+D triple overlap is observed in this run.

RLS prediction error is computed before applying the corresponding label. It is evaluated on selected
actions, not independent held-out or counterfactual samples. The warmup epoch is subtracted from the
measurement epoch; full telemetry emits cumulative snapshots after drains, so convergence time
inside a dispatch cannot be recovered. A ready flag means minimum observation count, not accuracy.
`analyze_contextual_adaptation.py --epoch-summary` reports this limitation explicitly.

| Phase interval | Measured duration | Fraction of E active time |
|---|---:|---:|
| E active | 28.685 s | 100% |
| E/P overlap | 10.583 s | 36.9% |
| E/D overlap | 5.110 s | 17.8% |

The total measured active-span window is 108.073 seconds. The existing activity analyzer confirms
the phase-overlap durations including sampling. The engine-only sweep confirms
E/P overlap of10.583 seconds (36.9% of E active) and E/D overlap of5.110 seconds (17.8%). Detailed recording lowers throughput
versus the uninstrumented runs; this is not the vLLM comparison configuration.

| RLS family | New measured observations | MAE | RMSE | Predicted profitable | False profitable |
|---|---:|---:|---:|---:|---:|
| PD | 473 | 0.178 | 0.250 | 272 | 21 |
| EP | 18 | 0.159 | 0.221 | 9 | 0 |
| ED | 24 | 0.262 | 0.329 | 2 | 1 |

Errors are dimensionless normalized advantage: `(reference_work - observed_makespan) / reference_work`,
not milliseconds or token error rates. Across calibration and measurement, directional samples end
at P→D820, D→P195, E→P22, E→D25, P→E0 and D→E0. Reverse E directions have no learned evidence.
All three families report zero rejected observations. The model is updating, but measured PD RMSE
0.250 and ED RMSE0.329 do not support a claim of consistently accurate or converged pricing.
EP has only18 new observations; its zero false-profitable labels among9 profitable predictions is
limited evidence. ED has one false-profitable label among only two such predictions.

Warmup PD RMSE is0.083, versus0.250 during measurement. Passing readiness at calibration end
does not establish accurate prices under the later serving load. No observations are rejected
for invalid numbers, but this numerical health check does not validate the reference/label model.

## Memory sharing and portability experiments

The current vision engine has one optimization profile. `tiered_ep` would fall back to shared E/P
with full serialization, so it is not counted as an independent candidate. The current builder
also restricts smaller visual profiles to Gemma4; Cosmos tiered E/P would require implementation
work, not just a new parameter. E/P sharing saves
1275074048 bytes (1.188 GiB) of workspace by reusing an arena of4659871744 bytes (4.340 GiB).
Measured total memory can change by more because active temporary buffer lifetimes also change.

The runtime does not import a Thor RLS posterior: every server starts a new process-local learner.
A100 batch capacities were already searched and its engines were built locally. Carried settings
remain possible portability issues, as identified in note339: compatibility decode cost tables,
25ms E formation cap, graph cache64, minimum RLS observations4 and confidence beta0.5.
`ENABLE_MEASURED_DECODE_BATCHING=1` does not itself remove the compatibility table: sparse coverage
can fall back to it. `MEASUREMENT_MEASURED_DECODE_COSTS=1` explicitly clears that table at the
calibration boundary while preserving measured history; this is tested separately.

E/D sharing reuses only19928576 bytes (19.005 MiB) of decode workspace. Its generic calibration
stalled for more than170 seconds with backend CPU100% and GPU0%; the isolated candidate was
interrupted, its backend terminated, and logs retained in `performance-shared_ed/failure.json`.
No measured requests completed, so no throughput is assigned. Root cause is not established;
this is a liveness failure requiring separate diagnosis, not a measured performance regression.

| Workspace mode | balanced | decode-heavy | mixed | vision-heavy | Geometric mean | Peak MiB | Repeats |
|---|---:|---:|---:|---:|---:|---:|---:|
| Independent | 5474 | 10642 | 964 | 671 | 2478 | 39832 | 3 |
| Shared E/P | 5305 | 10548 | 451 | 302 | 1662 | 35922 | 3 |
| Shared E/D | — | — | — | — | — | — | Calibration stalled |

All rates are generated tokens/s. Shared E/P reduces peak across workloads by3910MiB (3.818GiB),
but reduces mixed throughput53.2%, vision-heavy55.0% and geometric mean32.9%. Its mixed-cell
peak reduction is3998MiB (3.904GiB). The difference beyond the1.188GiB shared workspace is not
assigned solely to arena sharing: serialization also changes temporary buffer overlap/lifetimes.
Independent E/P remains the useful throughput setting for this workload.

| Carried setting ablation | Geometric mean tok/s | Peak MiB | Repeats |
|---|---:|---:|---:|
| Remove compatibility D table after calibration | 2451 | 39812 | 3 |
| E formation cap25ms → 0 | 2437 | 39824 | 1 |
| D graph cache64 → 128 | 2474 | 40032 | 1 |
| RLS minimum observations4 → 16 | 2447 | 39824 | 1 |
| RLS confidence beta0.5 → 1.0 | 2424 | 39828 | 1 |
| Original settings at campaign end | 2440 | 39840 | 1 |

Original settings give geomean2478 in the initial three-pass confirmation and2440 in the final
single-pass control (about1.5% apart). None of these ablations establishes a large systematic
throughput gain. Single-pass screens are diagnostics, not a statistically confirmed optimum.
Graph128 raises peak memory by200MiB with almost unchanged overall throughput. This campaign
does not exclude every possible Thor portability effect; cold priors, feature scales, CPU-adapter
capacity and engine tactics are not independently swept. Current host: six physical Xeon cores,
12 logical CPUs. No carried RLS posterior is loaded, and current source/build capabilities are
A100-specific. Removing the old D table alone does not explain the modest vLLM throughput gap.

## Higher-load follow-up

The prior campaign already raised concurrency64 to512, and runtime D capacity64 to256; an earlier
D512 engine diagnostic did not improve overall throughput and had memory/capture limitations
(note380). This follow-up tests external concurrency beyond512. The fixed larger traces contain
2048 balanced/decode-heavy requests and1024 mixed/vision-heavy requests each. Each trace is the
original UUID trace repeated twice, preserving messages, output lengths and arrival offsets, but
assigning a new unique image UUID to every occurrence. Both concurrency512 and1024 use these
identical larger traces, so the concurrency comparison is not confounded by request count.

TRT internal inflight/stable-slot limits remain256; increasing HTTP concurrency cannot raise the
engine D ceiling. vLLM retains its optimized FP16 S512/T16384 config, prefix/processor caches off,
21GiB KV and no cross-request image reuse. Each vLLM pass has a fresh server. The system order
is TRT→vLLM at512 and vLLM→TRT at1024. These are one-pass diagnostics, not three-pass promotion
evidence. vLLM is rerun because the workload contract changed; the old frozen result is not reused.

| Client concurrency | TRT geo tok/s | vLLM S512 geo tok/s | TRT/vLLM | TRT peak MiB | vLLM peak MiB |
|---:|---:|---:|---:|---:|---:|
| 512 | 2428 | 2293 | 1.059x | 39826 | 31870 |
| 1024 | 2431 | 2281 | 1.066x | 39828 | 31914 |

Client send/completion CSVs confirm actual peak HTTP inflight512 or1024 in every cell, not
merely a configured limit. TRT geomean changes+0.10% and vLLM S512−0.50% when concurrency doubles,
so this tested engine/KV contract is already near a throughput plateau at512. This is not proof
that every hardware, engine, workload or internal capacity is saturated. The earlier D512 engine
diagnostic is distinct from this external concurrency test.

For TRT, TTFT p95 rises from5392 to13439ms (balanced),7325 to18584ms (decode-heavy),28033 to43358ms
(mixed), and33179 to59548ms (vision-heavy). TTFT is measured after the client semaphore grants
submission; time waiting locally for that semaphore is excluded. Higher HTTP concurrency moves
more waiting into the server, so these numbers must not be interpreted as slower kernels.

At high-load512, a vLLM metrics snapshot during vision reports204 cumulative preemptions, including
calibration, and zero prefix/CPU multimodal cache hits. Preemption may be a contributor, but its
share of runtime and measured-only count are not established. A second vLLM S256/T16384 candidate
is screened at1024 client concurrency to avoid assuming that the old best batch remains optimal
under the changed load. The cache/precision/KV/workload contract is held fixed.

The completed S256 screen yields 5355/9435/895/611 tok/s (balanced/decode-heavy/mixed/vision-heavy),
geomean 2292.4 tok/s and peak 31810 MiB. At HTTP concurrency1024 this is
about0.49% above S512, with TRT ahead by
6.03%. This is the best of these two one-pass native
screens, not proof of a global optimum. Its final metrics contain324 cumulative preemptions,
including calibration and all four workloads; this cannot be compared directly with the earlier
mid-vision snapshot. All6144 requests complete, and observed HTTP inflight reaches1024 in each cell.

## Reproduction and disposition

Source environment with `.local/env.sh` and the Cosmos lineage `serving.env`. Use
`benchmarks/phase_serving/run_serving_comparison.py` with the retained config, matching output
folder, `--reuse-server --workload-warmup-requests 0`, and all four workloads in the order above.
Run `diagnostic-independent.json` once; `performance-independent.json`,
`performance-shared_ep.json` and `portability-measured-only.json` three times; the remaining
portability screens once. High-load configs512/1024 run both systems once; the additional
`high-load-1024-vllm-s256.json` runs vLLM once. Exact client/server commands are retained alongside
summaries. Each high-load candidate starts a fresh server. The excluded E/D calibration was
aborted manually, preserving logs and `failure.json`; do not treat it as a completed benchmark.

For full telemetry use `analyze_contextual_adaptation.py LOG --epoch-summary --output OUTPUT`
and `analyze_phase_activity.py --gateway-log LOG --intervals activity-intervals.csv --output-dir DIR`.
The separate `engine-only-overlap.json` unions encoder-engine and prefill/decode-dispatch event
intervals, excluding sampling; raw interval CSVs and full telemetry are retained for this conclusion.
Unused diagnostic shared-workspace configs are not evidence of executed full-telemetry runs.

Accepted measured runs complete76800 requests with zero failures and full fixed outputs. The
E/D liveness failure remains unresolved. Keep independent workspace: E/P sharing has a substantial
vision throughput penalty, and E/D sharing has little direct memory benefit even before resolving
its stall. RLS updates are real, but serving error and sparse E observations do not support a
claim of stable learning. Carried-setting screens and higher external load do not reveal a large
hidden throughput gain. The retained FP16/KV-matched TRT advantage remains modest and its peak
memory higher. No runtime promotion or output-quality waiver follows from these diagnostics.

Follow-up [note 383](383-cosmos-tiered-shared-memory-20261006.md) identifies and repairs the direct-event
workspace ownership retirement bug behind the E/D stall. It also separates the full-sharing penalty
from the single-retained-batch storage policy, and tests compact Qwen3-VL tiered profiles.
