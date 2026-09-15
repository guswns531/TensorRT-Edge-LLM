# Lifetime encoded admission: two-model results and trade-off assessment

## 1. Question and implemented change

The experiment asks whether the encoded-request window can be replaced by a model-neutral byte/lifetime
admission mechanism, rather than choosing a different winning request count for every workload. This is
not a new RLS predictor, a new SLO target, or automatic engine-profile selection. The implementation follows
[the implementation plan](301-lifetime-encoded-admission-two-model-plan-20260913.md).

The opt-in production field is `PhaseServingRuntimeConfig::enableLifetimeEncodedAdmission`. It flows into
`PhaseThreeCoordinatorConfig`; the default remains false. An explicit `maxEncodedBytes` ceiling is optional.
For the comparison, activation occurs only at the drained calibration/measurement boundary through
`PhaseThreeCoordinator::setEncodedAdmissionMode()`, so changing admission cannot change which generic
calibration requests are offered to the initial static controller.

| Implementation location | Responsibility |
|---|---|
| `cpp/runtime/scheduling/phaseMemoryBroker.{h,cpp}` | Pure saturated-arithmetic fitting-prefix helper |
| `cpp/runtime/scheduling/phaseVisionAdapter.{h,cpp}` | Shared physical-owner deduplication and observed prepared-storage peak |
| `cpp/runtime/scheduling/independentPhaseAsyncServer.{h,cpp}` | Collect active, queued and pending-attachment payload owners |
| `cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}` | Drained activation, E reservation, physical byte feasibility and pressure metrics |
| `cpp/runtime/scheduling/phaseServingRuntime.{h,cpp}` | Experimental opt-in C++ production configuration plumbing |
| `examples/llm/llm_phase_context_smoke.cpp` | Common-calibration measurement switch and telemetry |
| `benchmarks/phase_serving/run_lifetime_encoded_admission.py` | Restricted two-model HTTP experiments and immutable identities |
| `benchmarks/phase_serving/analyze_lifetime_encoded_admission.py` | Trace attribution, compressed evidence and explicit confound replacement |
| `tests/python-unittests/test_lifetime_encoded_admission.py` | CPU experimental-contract tests |

```text
Pending vision requests
        |
        v
E-compatible frontier: engine batch/media/token/geometry constraints
        |
        v
Retained physical owners + outstanding E reservations + new E reservation <= byte budget
        |
        v
Async preparation -> E context -> completed shared vision slab
                                      |
                                      v
                                    P queue -> independent P context
                                      |
                           GPU-safe final-P completion
                                      |
                        release embedding/deepstack views
                                      |
                     D queue -> independent D context -> completion
                                      |
                  release remaining M-RoPE and KV ownership
```

The automatic byte budget is sampled once at a drained activation boundary:

```text
budget = max(0, cudaMemGetInfo.freeBytes - maxObservedPreparedStorageBytes)
```

Current reservations and retained-owner accounting change on subsequent queue/ownership transitions.
The device free-memory query is not placed in the scheduling hot path. The budget itself is not continuously
resized. KV physical allocation, logical page limit, page table, addressing and eviction are unchanged within
each comparison. Existing fixed P128 chunks, V3 action evaluation, E/P/D independent contexts, E formation
wait and engine shape limits remain unchanged.

## 2. Why physical ownership, rather than logical payload sums

Rows from one encoder execution can share a batch slab. Summing a row's live embedding view does not tell
us whether the underlying allocation has actually become free. If one row finishes P but another row still
references the slab, the full slab remains charged once. `phaseVisionRetainedStorageBytes()` deduplicates
batch owners and split M-RoPE owners; it includes external tensors' capacities when there is no shared owner.

Accounting includes coordinator P-ready payloads, admitted requests, queued requests, and deferred
`pendingVisionPayload` attachments. E rows reserve bytes before the asynchronous preparation worker runs;
the reservation is transferred into retained storage on E completion, not prematurely released while the
GPU or worker can still consume it. Existing final-P completion callbacks remain the authority for releasing
prefill storage. Cancellation does not permit early reuse of a running consumer's allocation.

For a legacy unsplit M-RoPE payload, the shared batch owner may remain live throughout D even after its
embedding views are released. A split M-RoPE lease allows the embedding slab to disappear while its much
smaller positional storage remains. This experiment does not change the selected engines' lease policy.

Unknown payload geometry uses the existing singleton bootstrap; if no earlier size estimate exists, that
singleton reserves the whole budget. Known oversized requests are explicitly rejected instead of being
allowed to bypass the byte ceiling. This is an admission estimator, not a proof of allocation safety:
unseen geometries, allocator rounding, temporary image/preparation storage, other GPU processes and
unobserved engine workspaces can still invalidate a budget derived from earlier observations.

## 3. Comparison contract

The primary campaign is
`.local/results/lifetime-encoded-admission-20260913/corrected-3x`.

| Property | Gemma | Cosmos |
|---|---|---|
| Model | google/gemma-4-E2B-it | nvidia/Cosmos-Reason2-2B |
| Weights | INT4-AWQ | FP16 |
| E / text-P / D maximum batch | 4 / 8 / 24 | 4 / 8 / 64 |
| Vision-P admission maximum | 4 | 4 |
| Prefill chunk | 128 | 128 |
| Stable sequence slots | 24 | 80 |
| HTTP maximum in flight | 24 | 64 |
| Generic calibration requests | 49 | 239 |
| Initial/static-base encoded window | 4 | 16 |
| Static-large measured window | 12 | 80 |
| Lifetime measured window | no normal-path count credit | no normal-path count credit |

Three variants use the same compiled binary and the same engines for each model. All begin with that
model's same static-base calibration configuration; static-large and lifetime change only after calibration
drains. The offered generic requests are identical, but timing-sensitive GPU observations and therefore
posterior values need not be bit-identical across independent process executions.

The VLM workloads are the retained real HTTP `mixed`, `vision-heavy` and `multi-image` traces. Gemma's
multi-image trace has 20 requests; Cosmos's retained semantic multi-image trace has 5 requests. Neither is
a kernel-only benchmark, and these two traces must not be interpreted as identical cross-model work.
Each measured variant is attempted three times, with variant order reversed on the second repetition.
Full phase telemetry is enabled for every current variant. Primary CUDA graph settings and synchronous
decode-sampling setting are held constant. No additional serving SLO is supplied by this experiment.

The manifest records source commit and dirty patch, binary/engine/config/calibration/workload hashes,
container image digest, exact commands, completed cells and failed attempts. The preliminary binary differs
in deferred-payload accounting and its results are diagnostic only; they are never spliced into this campaign.

Final comparison binary SHA256:
`51ef6f8c61755f62cdb51bb57160bbdf3067adcb9144bd2589aa2b571bda64ca`.

Successful aggregate token counts and HTTP status are checked, and D useful-token counts must equal total
output tokens minus one first token per request. These checks do not establish exact cross-policy greedy
identity: independent FP16 execution schedules can produce different token hashes, and semantic/logit
validation is a separate gate.

## 4. Metrics and aggregation rules

Report token throughput, TTFT mean/p95, TPOT mean/p95, E2E mean/p95 and peak VRAM. Throughput is the
arithmetic mean of the three independent run throughput values; p95 columns average each run's request
p95, not the p95 of a merged distribution. Preserve run ranges rather than presenting three observations
as a high-confidence statistical estimate. Failed attempts are excluded from successful latency means,
but their attempted/successful counts and errors remain visible; an all-failed comparator has no latency value.

The analyzer also retains scheduled-arrival-based TTFT/E2E, E queue wait, E/P/D batch distributions,
scheduler decision duration, all 16 stream-envelope masks and byte-pressure counters. The standard client
latencies are measured from HTTP send, so they must not be substituted for scheduled-arrival latency under
client concurrency backpressure. Stream envelopes indicate outstanding stream work, not hardware SM utilization.

`vision_admission_byte_blocks` counts candidate-prefix reductions, not unique blocked requests or a time
duration. Separately reported maxima of retained bytes and reserved bytes must not be summed to claim a
simultaneous budget violation. `vision_downstream` still reports resident request metadata, including rows
whose vision storage has already released; it is not a live-embedding count.

## 5. Validation already completed

- C++ build and runtime test target completed successfully.
- 216 scheduling/async-server/memory tests passed, including byte-prefix fitting, exhaustion, unknown size,
  saturated arithmetic, duplicate/null payload accounting and GPU-safe lifetime regression coverage.
- Nine CPU benchmark contract tests passed: same pre-calibration settings, measurement-only activation,
  static comparator isolation, common explicit byte ceiling, restricted network-free backend, percentile
  behavior, losslessly compressed evidence, declared replacement provenance and slot-capability comparison.
- The final RuntimeState binary also passed 23 `StableKVPageManagerTest`/`KVPageTableTest` cases. The first
  attempt targeted the wrong test binary and matched zero cases; it is not counted as coverage. The proper
  page-table suite performs CUDA uploads, so it must not run concurrently with a performance measurement.

The unit tests for payload accounting use external CPU tensors; the repeated GPU encoder batches exercise
actual shared slabs. This is not a new sanitizer run or complete ownership proof across all cancellation races.

A scheduling mistake ran that short GPU page-table suite while the third Gemma vision-heavy static-base
cell was active. Conservatively exclude that cell from the uncontended performance estimate and replace
it with a separately manifested, clean targeted repeat after the primary campaign drains. Preserve the
original cell as diagnostic evidence rather than silently overwriting it. No further GPU tests are scheduled
concurrently with the campaign.

## 6. Repeated VLM results

The primary VLM campaign completed 54 attempts: 52 successful HTTP runs and two failed Cosmos static-large
vision-heavy attempts. Its third static-large vision-heavy attempt succeeded, so it is **1/3 successful**, not
an all-failed comparator. All lifetime cells completed 3/3. A clean separately manifested Gemma static-base
vision-heavy repeat replaces the declared confounded primary repeat in the following uncontended estimates;
the primary `summary.json` still retains its original values. Use `comparison.json` with the explicit exclusion
and supplement rather than silently interpreting that raw summary as the corrected comparison.

### 6.1 All serving metrics

Latency columns are **mean / p95 in milliseconds**; VRAM is the mean of per-run peaks in MiB. `Base` is
window 4 for Gemma and 16 for Cosmos; `Large` is window 12 for Gemma and 80 for Cosmos. Neither tested
large window is claimed to be an exhaustive optimal static policy.

| Model / workload | Admission | Successful / attempted | Output tok/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | Peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma mixed | Base | 3/3 | 537.18 | 648.75 / 1977.06 | 19.32 / 25.11 | 1571.75 / 2289.81 | 9375.00 |
| Gemma mixed | Large | 3/3 | 688.86 | 256.55 / 597.35 | 26.70 / 35.34 | 1472.08 / 2275.73 | 9381.67 |
| Gemma mixed | Lifetime | 3/3 | 721.39 | 263.64 / 561.76 | 25.87 / 34.86 | 1452.98 / 2195.57 | 9386.00 |
| Gemma vision-heavy | Base, clean replacement | 3/3 | 323.62 | 1693.97 / 3461.35 | 15.05 / 23.00 | 2303.16 / 3852.71 | 9377.00 |
| Gemma vision-heavy | Large | 3/3 | 464.53 | 636.33 / 1519.48 | 29.25 / 36.13 | 1750.67 / 2445.48 | 9381.00 |
| Gemma vision-heavy | Lifetime | 3/3 | 551.47 | 378.02 / 725.25 | 31.41 / 44.86 | 1555.77 / 2141.12 | 9389.67 |
| Gemma multi-image | Base | 3/3 | 209.70 | 992.61 / 1937.00 | 11.30 / 15.51 | 1342.87 / 2233.96 | 9377.00 |
| Gemma multi-image | Large | 3/3 | 303.25 | 540.57 / 973.22 | 23.70 / 29.05 | 1275.20 / 1545.67 | 9381.67 |
| Gemma multi-image | Lifetime | 3/3 | 370.16 | 464.57 / 751.05 | 25.90 / 42.30 | 1267.49 / 1547.48 | 9390.33 |
| Cosmos mixed | Base | 3/3 | 1120.99 | 737.95 / 2201.45 | 30.23 / 41.23 | 2207.45 / 2545.24 | 9593.00 |
| Cosmos mixed | Large | 3/3 | 1154.55 | 830.53 / 2042.53 | 33.68 / 61.73 | 2397.06 / 2513.69 | 9735.00 |
| Cosmos mixed | Lifetime | 3/3 | 1159.66 | 801.52 / 1999.74 | 34.08 / 61.73 | 2379.45 / 2506.78 | 9737.00 |
| Cosmos vision-heavy | Base | 3/3 | 667.91 | 1419.03 / 3340.37 | 28.09 / 38.17 | 2538.35 / 3622.78 | 9605.00 |
| Cosmos vision-heavy | Large, successful survivor only | 1/3 | 729.62 | 1553.80 / 2974.55 | 40.67 / 81.38 | 3157.53 / 3326.16 | 9835.00 |
| Cosmos vision-heavy | Lifetime | 3/3 | 723.30 | 1508.29 / 3007.53 | 42.95 / 81.07 | 3185.05 / 3349.47 | 9815.67 |
| Cosmos multi-image | Base | 3/3 | 300.00 | 275.39 / 318.92 | 8.18 / 10.03 | 528.82 / 532.81 | 9601.00 |
| Cosmos multi-image | Large | 3/3 | 300.61 | 276.70 / 318.55 | 8.10 / 9.94 | 527.73 / 531.90 | 9605.00 |
| Cosmos multi-image | Lifetime | 3/3 | 300.47 | 277.03 / 318.56 | 8.11 / 9.99 | 528.50 / 532.23 | 9607.00 |

Request throughput is also retained in `comparison.json` as `achieved_req_s_median`. It scales with output
throughput under each fixed-work trace: mixed has 45.75 output tokens/request, vision-heavy 38.5, multi-image
32. The lifetime request throughput is therefore 15.77 / 14.32 / 11.57 req/s for Gemma and 25.35 / 18.79 /
9.39 req/s for Cosmos, respectively. These are achieved rates, not offered-arrival-rate settings.

### 6.2 Lifetime relative to the initial static window

All changes are `100 * (lifetime / reference - 1)`: positive throughput is favorable, positive latency is a
regression. A tail improvement must not hide an average-latency loss.

| Model / workload | Tok/s change | TTFT mean / p95 change | TPOT mean / p95 change | E2E mean / p95 change |
|---|---:|---:|---:|---:|
| Gemma mixed | +34.29% | -59.36% / -71.59% | +33.92% / +38.81% | -7.56% / -4.12% |
| Gemma vision-heavy | +70.41% | -77.68% / -79.05% | +108.66% / +95.04% | -32.45% / -44.43% |
| Gemma multi-image | +76.52% | -53.20% / -61.23% | +129.24% / +172.73% | -5.61% / -30.73% |
| Cosmos mixed | +3.45% | +8.61% / -9.16% | +12.73% / +49.70% | +7.79% / -1.51% |
| Cosmos vision-heavy | +8.29% | +6.29% / -9.96% | +52.92% / +112.40% | +25.48% / -7.54% |
| Cosmos multi-image | +0.16% | +0.59% / -0.11% | -0.77% / -0.38% | -0.06% / -0.11% |

### 6.3 Lifetime relative to the tested larger static window

| Model / workload | Tok/s change | TTFT mean / p95 change | TPOT mean / p95 change | E2E mean / p95 change |
|---|---:|---:|---:|---:|
| Gemma mixed | +4.72% | +2.77% / -5.96% | -3.10% / -1.35% | -1.30% / -3.52% |
| Gemma vision-heavy | +18.71% | -40.59% / -52.27% | +7.36% / +24.18% | -11.13% / -12.45% |
| Gemma multi-image | +22.06% | -14.06% / -22.83% | +9.29% / +45.64% | -0.61% / +0.12% |
| Cosmos mixed | +0.44% | -3.49% / -2.10% | +1.20% / -0.01% | -0.73% / -0.27% |
| Cosmos vision-heavy, 1-success comparator | -0.87% | -2.93% / +1.11% | +5.60% / -0.38% | +0.87% / +0.70% |
| Cosmos multi-image | -0.05% | +0.12% / +0.00% | +0.18% / +0.52% | +0.15% / +0.06% |

The Cosmos vision-heavy survivor is not a reliable latency baseline: two of three attempts failed. The small
Cosmos mixed/multi-image deltas are not evidence of a statistically established speedup. Its important byte
admission benefit is successful execution under the pressure that defeated the large count window.

### 6.4 Text-only control: completed 12/12 HTTP runs

Both models' balanced text-only static-base/lifetime pairs completed three repeats. No vision requests are
present, so the admission change should be mechanism-neutral on this path. The small observed differences
are not attributed to a vision-specific speedup.

| Model | Admission | Output tok/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms | Peak MiB |
|---|---|---:|---:|---:|---:|---:|
| Gemma | Static-base | 1162.48 | 87.11 / 219.19 | 16.95 / 18.43 | 1501.47 / 2345.71 | 9373.00 |
| Gemma | Lifetime | 1174.40 | 87.31 / 218.99 | 16.77 / 18.23 | 1484.55 / 2297.38 | 9373.00 |
| Cosmos | Static-base | 4199.85 | 71.00 / 165.81 | 13.18 / 14.85 | 1189.65 / 1854.89 | 9601.00 |
| Cosmos | Lifetime | 4233.73 | 69.28 / 160.97 | 13.13 / 14.78 | 1183.45 / 1843.51 | 9601.67 |

| Lifetime vs text static-base | Tok/s change | TTFT mean / p95 change | TPOT mean / p95 change | E2E mean / p95 change |
|---|---:|---:|---:|---:|
| Gemma balanced | +1.03% | +0.23% / -0.09% | -1.11% / -1.05% | -1.13% / -2.06% |
| Cosmos balanced | +0.81% | -2.43% / -2.92% | -0.34% / -0.46% | -0.52% / -0.61% |

The measured controls do not show meaningful LLM degradation. They cover balanced only, not all text-only
traces from the historical 12-workload suite. See the independently manifested `text-control-3x` campaign.

### 6.5 Stronger Gemma comparator: static engine-slot window 24

The `static-slot-control-3x` campaign completed all nine runs. It uses the identical initial window-4 generic
calibration and switches only the measured static window to 24, the existing engine stable-slot capability.
It does not rebuild an engine or tune a separate count by workload. The lifetime values above are still the
original three runs, not newly selected best repeats.

| Gemma workload | Static-24 output tok/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms | Peak MiB |
|---|---:|---:|---:|---:|---:|
| mixed | 727.72 | 253.02 / 545.57 | 26.02 / 34.95 | 1441.81 / 2193.67 | 9385.67 |
| vision-heavy | 552.26 | 377.70 / 682.09 | 31.21 / 45.58 | 1548.36 / 2127.73 | 9389.00 |
| multi-image | 379.53 | 445.49 / 701.67 | 25.21 / 41.17 | 1227.08 / 1507.34 | 9387.67 |

| Lifetime vs static-24 | Tok/s change | TTFT mean / p95 change | TPOT mean / p95 change | E2E mean / p95 change |
|---|---:|---:|---:|---:|
| Gemma mixed | -0.87% | +4.20% / +2.97% | -0.59% / -0.26% | +0.78% / +0.09% |
| Gemma vision-heavy | -0.14% | +0.09% / +6.33% | +0.62% / -1.57% | +0.48% / +0.63% |
| Gemma multi-image | -2.47% | +4.28% / +7.04% | +2.73% / +2.76% | +3.29% / +2.66% |

This prevents the weak-baseline interpretation. Lifetime is near the stronger static-24 baseline, not faster
than it. Multi-image mean E2E has a ~3.3% loss and TTFT p95 ~7%; do not relabel these as an all-metric 3%
pass. Static-24 runs were measured later rather than interleaved with the original lifetime runs, so these
small differences are not an isolated attribution to byte-scan overhead.

Gemma has ample vision-byte headroom and the 24-request HTTP cap already bounds its live frontier.
Removing explicit count tuning recovers large-window behavior at modest cost. Cosmos's corresponding
capability-derived static-80 comparator is the primary Large variant; it is not consistently memory-safe
(two vision-heavy OOM attempts). Simply using every engine slot does not establish a universal safe window.

### 6.6 Repeated lifetime run spread

| Model / workload | Tok/s min–max | E2E mean min–max ms | Per-run E2E p95 min–max ms |
|---|---:|---:|---:|
| Gemma mixed | 716.34–724.18 | 1446.09–1463.93 | 2180.51–2225.27 |
| Gemma vision-heavy | 546.35–557.29 | 1543.51–1572.13 | 2105.83–2163.52 |
| Gemma multi-image | 367.71–373.39 | 1254.35–1280.93 | 1537.51–1563.65 |
| Cosmos mixed | 1154.75–1168.96 | 2356.18–2395.99 | 2484.44–2523.34 |
| Cosmos vision-heavy | 714.20–731.48 | 3134.61–3235.02 | 3317.18–3386.91 |
| Cosmos multi-image | 298.86–301.65 | 526.10–530.70 | 530.34–534.87 |

## 7. Physical admission and formation attribution

### 7.1 Lifetime high-water values

These are maxima across sampled records and repeats; they are not a continuous CUDA allocator trace.

| Model / workload | Auto budget peak MiB | Retained physical storage peak MiB | E reserved estimate peak MiB | Candidate-reduction counter peak | Resident downstream request peak |
|---|---:|---:|---:|---:|---:|
| Gemma mixed | 501.91 | 10.65 | 6.16 | 0 | 17 |
| Gemma vision-heavy | 501.91 | 12.12 | 6.16 | 0 | 24 |
| Gemma multi-image | 501.91 | 11.41 | 6.16 | 0 | 20 |
| Cosmos mixed | 426.06 | 323.69 | 35.06 | 0 | 32 |
| Cosmos vision-heavy | 425.06 | 420.22 | 35.03 | 2023 | 48 |
| Cosmos multi-image | 425.06 | 51.06 | 24.91 | 0 | 5 |

Gemma's byte pressure is inactive in these traces. Its improvement is primarily removal of an overly small
request-count gate; this is not proof that a learned optimal byte ceiling has been found. The compatible
E/P engine shapes, server slots and request frontier still bound concurrency. Cosmos retained storage is
over an order of magnitude larger, so its automatic admission actually encounters memory pressure.

All successful cells' last sampled retained-byte value is zero. This supports completion/lifetime accounting,
but does not alone certify every allocation was freed immediately after drain. Static cells can report zero
E reservation estimates when the runner initially returns unknown sizes; that is not zero preparation memory.

### 7.2 Dispatch and queue effects

The following compare static-base with lifetime. Counts and mean batch values are averaged per run; they
are not a token-weighted mean over all repeats.

| Model / workload | E dispatches Base → Lifetime | P dispatches Base → Lifetime | D dispatches Base → Lifetime | D mean BS Base → Lifetime | Mean E queue wait ms Base → Lifetime |
|---|---:|---:|---:|---:|---:|
| Gemma mixed | 25.00 → 18.33 | 101.33 → 73.00 | 293.33 → 147.67 | 9.77 → 19.40 | 961.60 → 35.27 |
| Gemma vision-heavy | 34.67 → 23.67 | 118.33 → 81.67 | 539.00 → 150.00 | 4.49 → 16.00 | 1966.54 → 56.79 |
| Gemma multi-image | 13.33 → 8.00 | 41.67 → 32.00 | 238.00 → 51.33 | 2.61 → 12.08 | 758.50 → 32.47 |
| Cosmos mixed | 11.67 → 10.00 | 34.33 → 35.00 | 80.67 → 70.00 | 35.60 → 40.93 | 904.21 → 618.16 |
| Cosmos vision-heavy | 21.67 → 15.67 | 45.33 → 45.33 | 115.67 → 72.33 | 20.83 → 33.19 | 1452.80 → 931.59 |
| Cosmos multi-image | 2.00 → 2.00 | 4.00 → 4.00 | 33.00 → 32.67 | 4.70 → 4.75 | 38.08 → 32.63 |

Gemma's smaller count window fragments both P formation and decode cohorts; removing it dramatically
reduces dispatch count. The larger batch costs more per decode iteration, which explains why TPOT can
increase even while TTFT, total completion throughput and E2E improve. This is a measured trade-off, not
a free all-metric optimization.

For Cosmos, P work/count barely changes in vision-heavy, while D cohorts grow and D dispatch count falls.
This reduces total drain time but produces a slower streaming experience for individual resident requests.
In vision-heavy, text E2E mean changes 2267.44 → 3269.19 ms and vision E2E mean 2628.66 → 3157.00 ms.
The average loss is not merely a changed class mix. In mixed, text E2E mean changes 2394.62 → 2457.45 ms,
and vision E2E mean 2020.27 → 2301.46 ms. The improved p95 and faster achieved throughput do not rescue
that mean-latency regression for an interactive-serving objective.

### 7.3 Activity and host-cost boundaries

Lifetime all-stream idle-envelope ratios are 1.82 / 2.30 / 1.92% for Gemma mixed/heavy/multi-image and
2.83 / 3.28 / 1.99% for Cosmos. E/P/D-only idle has the same rounded values here. Thus the throughput
change is not primarily recovery of a large idle stream gap. `analysis.json` preserves all 16 masks for each
repeat, so overlap and individual stream coverage can be reanalyzed without a fresh GPU run.

Measured lifetime `host_scheduler_decision_us` mean / p95 is 71.62 / 141.19, 68.59 / 158.70, 55.65 / 110.39
for Gemma, and 119.15 / 237.21, 111.83 / 239.92, 27.27 / 41.25 for Cosmos. These are whole existing scheduler
measurements, not an isolated cost of the new byte helper. The new physical-owner scan is bounded O(n²)
with temporary pointer vectors; it is not claimed to be zero-overhead or an incremental O(1) ownership ledger.

Scheduled-arrival E2E mean / p95 differs substantially for Gemma mixed (2683.62 / 3985.94 ms) and heavy
(2960.05 / 4372.10 ms), because the 24-request client cap can delay HTTP sends. Gemma multi-image is
1267.85 / 1547.78 ms. Cosmos allows all 64 offered requests in flight and has arrival E2E 2379.54 / 2506.85,
3185.10 / 3349.51, 528.88 / 532.49 ms. Do not compare these arrival values to send-based frozen vLLM
columns without also extracting that reference's arrival timestamps.

## 8. Warmup and learning conditions

Every analyzed Gemma calibration completed exactly 49 offered warmup requests; every Cosmos calibration
completed 239. The new admission mechanism has no RLS parameter vector and learns no capacity directly.
It uses geometry estimates, the adapter's observed storage maximum, active reservations and physical owner
lifetime. The preexisting V3/RLS controller remains active and may continue collecting serving observations.

Gemma's analyzed primary calibrations have P→D observations 105–107 and D→P exactly 4, but E→P only
2–3, E→D 0–1 and reverse E directions zero. None reports all contextual directions converged. Cosmos
P→D observations span 234–322, D→P 4–16, E→P 6–12 and E→D 1–11; 20/25 analyzed successful primary
calibrations report the contextual convergence flag. This campaign does not certify stable all-direction
E/P/D learning or equal posterior authority between models. It deliberately holds each model's existing
generic acquisition procedure fixed to isolate admission, instead of adding target-trace-derived warmup.

## 9. Frozen vLLM comparison

Gemma reuses the optimized retained vLLM 0.28 result at
`.local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12`:
24 sequences, 480 MiB FP16 KV, 4096 batched tokens and sparse graph buckets through 24. Each listed
Gemma vLLM workload has **one** retained run, not three. Cosmos reuses
`.local/results/v0101-forward-port/v3-service-scale-20260910/vllm-fresh-equal-summary.json`: FP16 weights/KV,
64 measured-request concurrency, fixed output/ignore EOS, 3.5 GiB KV, and 16-request vision warmup to
avoid all-at-once warmup OOM. Its vision-heavy reference has 2/3 successful runs; others here have 3/3.

These are unchanged-workload contextual references, not fresh randomized equal-memory pairs. Do not
attribute the entire vLLM difference to lifetime admission: the same-binary static-base already has substantial
runtime/packing gains over the older Cosmos reference. Current full telemetry also differs from the vLLM
instrumentation overhead.

| Model / workload | vLLM successful / attempted | vLLM output tok/s | vLLM TTFT mean / p95 ms | vLLM TPOT mean / p95 ms | vLLM E2E mean / p95 ms |
|---|---:|---:|---:|---:|---:|
| Gemma mixed | 1/1 | 703.81 | 276.41 / 410.58 | 26.48 / 32.39 | 1473.77 / 2162.56 |
| Gemma vision-heavy | 1/1 | 559.83 | 280.57 / 390.78 | 30.59 / 40.54 | 1420.68 / 2138.33 |
| Gemma multi-image | 1/1 | 381.34 | 188.86 / 227.16 | 29.70 / 35.54 | 1109.61 / 1300.30 |
| Gemma balanced text | 1/1 | 771.46 | 134.27 / 234.33 | 23.76 / 24.56 | 2128.03 / 3244.79 |
| Cosmos mixed | 3/3 | 923.32 | 868.73 / 2542.45 | 47.29 / 83.92 | 2998.29 / 3132.22 |
| Cosmos vision-heavy | 2/3 | 577.19 | 1630.87 / 3544.35 | 65.15 / 120.75 | 4087.81 / 4240.20 |
| Cosmos multi-image | 3/3 | 243.90 | 260.05 / 401.97 | 12.40 / 16.27 | 645.29 / 654.42 |
| Cosmos balanced text | 3/3 | 4315.77 | 112.25 / 254.04 | 12.20 / 13.58 | 1155.56 / 1771.35 |

| Lifetime relative to vLLM | Tok/s change | TTFT mean / p95 change | TPOT mean / p95 change | E2E mean / p95 change |
|---|---:|---:|---:|---:|
| Gemma mixed | +2.50% | -4.62% / +36.82% | -2.31% / +7.64% | -1.41% / +1.53% |
| Gemma vision-heavy | -1.49% | +34.74% / +85.59% | +2.68% / +10.65% | +9.51% / +0.13% |
| Gemma multi-image | -2.93% | +145.99% / +230.62% | -12.80% / +19.03% | +14.23% / +19.01% |
| Gemma balanced text | +52.23% | -34.98% / -6.55% | -29.43% / -25.77% | -30.24% / -29.20% |
| Cosmos mixed | +25.60% | -7.74% / -21.35% | -27.93% / -26.44% | -20.64% / -19.97% |
| Cosmos vision-heavy | +25.31% | -7.52% / -15.15% | -34.08% / -32.86% | -22.08% / -21.01% |
| Cosmos multi-image | +23.19% | +6.53% / -20.75% | -34.59% / -38.60% | -18.10% / -18.67% |
| Cosmos balanced text | -1.90% | -38.28% / -36.64% | +7.61% / +8.81% | +2.41% / +4.07% |

It is false that lifetime now beats vLLM in every metric or every model. Gemma multi-image TTFT and E2E
remain important deficits; Cosmos multi-image TTFT mean also remains slightly higher. The memory overhead
is not normalized away: retained Gemma vLLM VLM peak is 8843 MiB, while this current path is around
9386–9390 MiB. Most of that gap predates the admission change; lifetime versus same-binary static-large
adds only about 4–9 MiB in these Gemma cases.

## 10. Adoption and acceptable trade-off decision

Do **not** switch the shared default to lifetime mode yet. Keep one model-neutral opt-in algorithm rather
than selecting a different production policy by model/workload label. Both models show a meaningful
architectural effect, but not the same performance benefit:

- Gemma: count-gate removal improves formation, TTFT and throughput relative to the tested windows.
  Mixed is the most balanced trade-off against window 12. Vision-heavy buys ~11% average E2E and
  ~12% E2E p95 improvement at ~7% mean / ~24% p95 TPOT cost. Multi-image's ~22% throughput gain
  does not materially improve E2E against window 12 and costs ~46% TPOT p95; it is not an automatic
  interactive-serving promotion. The completed static-24 comparator is as good as or slightly better than
  lifetime; the large window-12 gains are not a fundamental dynamic-admission speedup.
- Cosmos: byte pressure makes the large-admission regime complete reliably in these three attempts,
  unlike count window 80. Against the smaller stable window 16, the ~25% average E2E and ~53% mean
  TPOT loss in vision-heavy is too large to accept for the current latency objective, despite throughput and
  p95 E2E improvement. The five-request multi-image case is essentially admission-neutral.

This is not a failed ownership mechanism; it is evidence that **memory-feasible admission is not
service-efficient admission**. Do not compensate by choosing a bespoke 200 MiB budget, a new per-model
count window, or a trace-specific SLO solely to make these tables look better.

## 11. Remaining evaluation and implementation steps

1. Two-model text-only balanced controls are completed, three runs per static-base/lifetime pair (section 6.4).
   Other historical text-only traces still require the later full-suite gate.
2. Gemma's static engine-slot window-24 comparison is completed (section 6.5). Cosmos's corresponding
   slot-limit comparator (80) is in the primary campaign. An interleaved paired recheck would be required to
   attribute the remaining small Gemma differences to the admission mechanism itself.
3. Add a deterministic downstream-service gate using the existing measured phase costs and known
   request frontier/completion events. Compare admit-now versus hold-for-known-completion under an
   equal-work bounded horizon; protect resident D continuity without inventing a new model-specific SLO.
4. Account for production idle cached slabs and validate the production Cosmos split M-RoPE lease path.
   Replace repeated physical-owner scans with an incremental ledger only after its lifetime contract is tested.
5. Only after both-model mean/tail trade-offs are acceptable, rerun the retained full 12 workloads per model
   and pressure/cancellation cases. The present three VLM traces are not a full-12 promotion gate.

## 12. Boundaries of the architectural claim

Removing a count gate allows admission to vary with payload size, current ownership and release timing.
It does not automatically find a latency-optimal resident decode cohort. In particular, a memory-feasible
large cohort can improve completion throughput while making each decode step slower. If this persists in
Cosmos, the next shared-model mechanism should evaluate measured downstream service/continuity cost,
not silently restore a per-workload winning encoded count or tune a new model-specific SLO.

The automatic budget presently participates in admission feasibility. Existing global memory-horizon
scoring still uses its earlier logical-byte accounting/explicit ceiling; it has not become a full physical
lifetime predictor. There is no dynamic CUDA pool resizing, KV-to-vision repartition, copy-stream redesign,
automatic E/P/D engine rebuilding, or workload classification in this change.

The smoke-harness contract explicitly retains zero idle vision slabs. Production `PhaseServingRuntime`
currently has its separate default idle-cache policy and automatically splits M-RoPE for Cosmos. Consequently,
the measured Cosmos smoke-harness unsplit ownership behavior must not be presented as a measurement of
that production split-lease configuration. Before promoting automatic lifetime admission in production,
idle pooled allocations must also be charged or reclaimed with a GPU-safe reuse policy, and the production
split-lease path must be tested directly. The experimental opt-in is not a completed production memory guarantee.

Retained `.gz` logs are lossless and can be decompressed to recover the originals. Models, engines and
worktrees are untouched; compression only covers closed manifest-listed successful or terminal failed logs.

## 13. Reproduction and retained evidence

The final aggregate comparison contains 73 uncontended successful performance cells: 52 VLM cells after
explicit replacement, 12 text controls and nine static-slot controls. The campaigns made 76 attempts;
74 completed HTTP successfully, one successful primary cell is excluded for concurrent GPU-test interference,
and two attempts failed OOM. All lifetime cells completed successfully. This covers four retained workloads
per model, not all 12 historical workloads.

Run commands:

```bash
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma cosmos --workloads mixed vision-heavy multi-image --repeats 3 \
  --result-root .local/results/lifetime-encoded-admission-20260913/corrected-3x

python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma --workloads vision-heavy --variants static-base --repeats 1 \
  --result-root .local/results/lifetime-encoded-admission-20260913/clean-control-recheck

python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma cosmos --workloads balanced --variants static-base lifetime --repeats 3 \
  --result-root .local/results/lifetime-encoded-admission-20260913/text-control-3x

python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma --workloads mixed vision-heavy multi-image --variants static-slot --repeats 3 \
  --result-root .local/results/lifetime-encoded-admission-20260913/static-slot-control-3x
```

Analyze supplemental campaigns first, then reconstruct the corrected comparison:

```bash
python3 benchmarks/phase_serving/analyze_lifetime_encoded_admission.py \
  --result-root .local/results/lifetime-encoded-admission-20260913/corrected-3x \
  --exclude-key gemma/vision-heavy/static-base/3 \
  --supplement-result-root .local/results/lifetime-encoded-admission-20260913/clean-control-recheck \
  --supplement-result-root .local/results/lifetime-encoded-admission-20260913/text-control-3x \
  --supplement-result-root .local/results/lifetime-encoded-admission-20260913/static-slot-control-3x
```

`comparison.json` preserves contributing root/key/aggregate hashes and the declared exclusion. `analysis.json`
preserves per-cell operation counts, all masks, pressure and arrival latency; `failure-analysis.json` preserves
both OOM error lines. Each manifest references notes 301/302. Executed runner source is archived as
`runner-source.py`, matching that campaign's runner hash even though the later static-slot comparator
extended the current script. Source remains on the active root v0.10.1 branch; no comparison worktree is edited.

Scripts do not rebuild engines, launch external-network containers, change `.local/current/` pointers, or
select a model-specific production winner. Campaign state remains `diagnostic`: the shared default is not
promoted and neither exact-output nor full-12 promotion gates are claimed.
