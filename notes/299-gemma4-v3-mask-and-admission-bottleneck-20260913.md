# Gemma 4 V3: vLLM deficits, phase masks, and admission bottlenecks

Date: 2026-09-13

## 1. Outcome and scope

V3 still loses generated-token throughput and mean/p95 E2E latency to the selected vLLM control on five of the
twelve retained Gemma workloads: long-prefill, bimodal, mixed, vision-heavy, and multi-image. Its mean TPOT remains
lower on all twelve. This is not evidence that every part of the first-token deficit is an encoder/kernel issue.

Six new V3 HTTP diagnostics locate two different forms of upstream blocking:

1. **Long-prefill/bimodal:** all 96 KV pages are reserved, even though the configured 24-owner ceiling is not
   reached. Server admission wait averages approximately 1.9 seconds. Full prompt-plus-output reservation and a
   smaller KV token pool constrain the executable request frontier.
2. **Vision-rich traffic:** KV pages remain available, but requests wait approximately 0.7--1.7 seconds before E
   starts. E batches average only 1.39--1.67 requests with an E4 engine. The four-payload cross-phase capacity and
   scheduling/formation decisions need to be separated before attributing this delay to encoder compute.
3. **P formation:** P1 decisions occur with multiple P-ready requests, but the recorded candidate frontier contains
   no larger P candidate in those snapshots. Tuning the V3 estimator cannot select an absent candidate.
4. **Runtime realization:** dispatch/submission spans average 5--8 ms, whereas scheduler computation is measured in
   tens of microseconds. Event envelopes can hide submission-induced gaps; a low `0000` ratio is not proof of high
   SM utilization.

No performance policy, chunk default, attention kernel, KV allocator, engine, or model was changed in this
diagnosis. Only reproducible diagnostic/analysis scripts were added. This is a diagnosis, not a promoted fix.

## 2. Keep three experiment contracts separate

| Evidence | Policy / engine | Status |
|---|---|---|
| Retained full-12, note 295 | V3, standalone packed P128, independent E4/P8/D24 | one diagnostic repeat |
| New mask diagnostic, this note | current binary, V3, same standalone P128/E4 engines, full telemetry | one diagnostic repeat per six workloads |
| Recent narrow/wide profile experiment, note 298 | **V0 Exact**, P128/P512 engines, tiered VLM arena | three diagnostic repeats |

The old full-12 binary was SHA-256 `191524728e17381eb0f43a210d1e8ec0e9bd95b93ac0e032103484e1f041224b`.
The current diagnostic binary is `3093ccfd006de6781921416265c2deccf06ebac4cdda1b1eb6e622d0f87c9835`, built from
runtime source committed at `a516dcdc194b44ea7cddd2ae08c84f1f1b2d4ca3`.

The current binary includes intervening direct vision-output and profile-routing changes. Therefore the fresh
diagnostic is not a telemetry-only, same-binary A/B against the old full-12. It identifies the current V3's
trajectory; differences from old performance can include runtime changes, telemetry perturbation, and variance.

Neither the Exact P512 text gain nor its VLM regression is a V3 policy result. No V3 P512 conclusion is claimed.

## 3. Where V3 does not beat vLLM: retained full-12

The frozen comparator is the corrected vLLM 0.28 seq24/KV480 MiB/P4096/sparse decoder-graph configuration from
[note 295](295-gemma4-vllm-capacity-frontier-20260912.md), not the superseded seq8 comparator. Requests, arrival
schedule, fixed output lengths, and client concurrency are shared; chat/template prompt totals and execution
backends differ. This is an E2E serving comparison, not exact cross-framework token identity or equal memory.

All deltas below are `(V3 / vLLM - 1) * 100`. Positive throughput is good; positive latency is bad. Mean and p95
are service-side request latency, not scheduled-arrival-inclusive latency.

| Workload | tok/s delta | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | +39.18% | -37.83% | -6.06% | -13.66% | -7.43% | -20.74% | -15.34% |
| balanced | +53.22% | -34.87% | -7.02% | -30.05% | -27.10% | -30.83% | -30.28% |
| decode-heavy | +59.21% | -42.12% | -10.56% | -33.64% | -32.50% | -34.07% | -33.39% |
| **long-prefill** | **-15.28%** | **+332.20%** | **+137.42%** | -38.11% | -42.53% | **+19.12%** | **+2.21%** |
| **bimodal** | **-1.93%** | **+576.24%** | **+535.25%** | -30.90% | -10.73% | **+7.89%** | **+6.65%** |
| text-heavy | +105.83% | -86.99% | -85.56% | -3.75% | -4.98% | -51.85% | -70.19% |
| **mixed** | **-31.96%** | **+163.66%** | **+507.44%** | -25.34% | -23.14% | **+12.83%** | **+39.08%** |
| **vision-heavy** | **-35.78%** | **+443.24%** | **+706.27%** | -57.33% | -44.35% | **+45.37%** | **+62.48%** |
| poisson | +29.54% | **+30.64%** | **+171.53%** | -18.24% | -14.97% | -16.06% | -13.69% |
| wave-drain | +1.43% | **+50.00%** | **+181.31%** | -64.31% | -60.24% | -41.03% | -8.91% |
| **multi-image** | **-49.07%** | **+590.15%** | **+916.56%** | -61.07% | -57.10% | **+49.77%** | **+96.63%** |
| late-vision | +46.00% | -5.33% | **+62.25%** | -36.86% | -36.66% | -35.57% | -36.55% |

Thus "wins throughput" is not "wins every metric": poisson/wave-drain still lose TTFT mean/p95, and late-vision
loses TTFT p95. Full absolute numbers remain in note 295. The five throughput deficits in absolute terms are:

| Workload | V3 tok/s | vLLM tok/s | V3 TTFT mean/p95 ms | vLLM TTFT mean/p95 ms |
|---|---:|---:|---:|---:|
| long-prefill | 423.84 | 500.26 | 2350.82 / 3423.61 | 543.92 / 1442.01 |
| bimodal | 588.56 | 600.16 | 2146.44 / 5579.99 | 317.41 / 878.39 |
| mixed | 478.90 | 703.81 | 728.78 / 2494.02 | 276.41 / 410.58 |
| vision-heavy | 359.54 | 559.83 | 1524.13 / 3150.77 | 280.56 / 390.78 |
| multi-image | 194.22 | 381.34 | 1303.41 / 2309.28 | 188.86 / 227.16 |

## 4. New diagnostic protocol and results

The six traces include all five throughput-loss workloads plus balanced as a positive control. Every cell starts a
fresh serving process, performs the same generic HTTP calibration, then runs its unchanged real-request trace.

- Gemma 4 E2B AWQ backbone, FP16 PLE/embedding/head/KV; same model and engines as standalone packed full-12.
- Independent E/P/D TensorRT contexts, **no tiered E/P workspace flag**.
- E4/P8/D24, fixed chunk 128, P token budget 1024, stable slots 24, KV pages 96.
- Global active `service-scaled-transition`, generic 49-request calibration, numeric request SLO absent.
- Before that HTTP calibration, the binary performs its generic P/D shape warm-up: 32 batches, 632 synthetic
  request rows, 60 overlap samples and 20 safe probes. The 49 HTTP requests are therefore not the entire
  initialization cost, and this is not zero-start learning. The E adapter is created after P/D shape warm-up.
- Retain TTFT hard guard, encoder formation wait 25 ms, encoded cross-phase capacity four.
- Graph capture/replay remains disabled; logs report zero P/D entries/captures/hits.
- CUDA-event activity plus full request/dispatch/scheduler telemetry enabled.
- Read-only input/build mounts; only each result cell is writable; container has no network and drops capabilities.
- One repeat each, **diagnostic**. All requested outputs were returned without OOM.

The calibration reports two calibrated required keys out of four and `calibration_converged=false` in these runs.
This is a useful diagnostic of the retained warm-up contract, not sufficient evidence for authority promotion or
learning convergence. It does not mean every P+D prediction is unknown.

| Workload | Output tokens | tok/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 5440 | 1157.12 | 88.19 / 220.17 | 17.10 / 18.71 | 1514.46 / 2362.30 | 9373 |
| long-prefill | 5440 | 431.40 | 2258.60 / 3203.22 | 23.04 / 26.23 | 4178.72 / 5998.16 | 9373 |
| bimodal | 9680 | 587.42 | 2192.92 / 5410.58 | 19.80 / 31.74 | 4806.60 / 10281.19 | 9373 |
| mixed | 2928 | 537.79 | 667.49 / 2035.79 | 18.79 / 24.54 | 1563.84 / 2319.25 | 9375 |
| vision-heavy | 2464 | 347.90 | 1535.54 / 3282.06 | 13.36 / 22.84 | 2087.77 / 3533.61 | 9375 |
| multi-image | 640 | 217.69 | 946.00 / 1672.24 | 10.52 / 14.20 | 1272.15 / 1986.55 | 9379 |

The five loss traces still lose throughput to the frozen vLLM control. Mixed and multi-image improve relative to
old V3, but this is not attributed solely to policy: the diagnostic uses a newer runtime binary and telemetry.
No fresh vLLM run is needed to inspect these same request traces; any new capacity/config comparison must explicitly
record its changed contract.

## 5. Mask definition and measurement limits

Bits are phase activity, **not SM affinity masks**:

```text
E = 0001    P = 0010    D = 0100    C = 1000
0011 = E+P    0101 = E+D    0110 = P+D    0000 = no recorded span
```

The recorder is reset at the end of generic HTTP calibration, before the `PHASE_EPOCH` measurement marker. The
analysis retains every interval in this measurement-reset CSV. It does not reuse the older lifecycle crop that
can exclude early encoder work before the first P dispatch. Request lifecycles are selected independently for
per-request stage analysis.

The denominator is first recorded interval start to last recorded interval end, including intervening gaps. It is
not process initialization, full HTTP wall-clock duration, SM occupancy, or an offered-load-normalized idle ratio.
Initial unrecorded waiting before the first interval and final unrecorded response drain are outside it.

P/D includes recorded engine/submission envelopes and sampling spans. `PhaseDispatchWorker::enqueueActivity()`
records an event, calls the host enqueue callback, then records the ending event. Consequently, an event envelope
can include host submission gaps and stream waits between kernels. A low `0000` value rules out long **uncovered
phase-envelope** gaps in this window; it does not rule out kernel-level idle inside the envelope.

C represents the dedicated encoder-output copy stream. Current direct output binding avoids those D2D copies. H2D
tokens/KV metadata and staging work still exist on phase/preparation streams; `C=0` does **not** mean zero memory
traffic or zero copy engines in use. Encoder preparation is not a separate E interval in this recorder.

### All sixteen exclusive mask states, percent of active span

| Mask | balanced | long-prefill | bimodal | mixed | vision-heavy | multi-image |
|---|---:|---:|---:|---:|---:|---:|
| 0000 | 1.092 | 1.195 | 1.020 | 2.460 | 2.986 | 2.621 |
| 0001 | 0 | 0 | 0 | 5.171 | 6.080 | 5.333 |
| 0010 | 12.452 | 25.092 | 10.679 | 25.436 | 24.935 | 25.279 |
| 0011 | 0 | 0 | 0 | 1.191 | 5.139 | 7.352 |
| 0100 | 77.794 | 40.944 | 74.351 | 50.221 | 52.587 | 57.168 |
| 0101 | 0 | 0 | 0 | 4.888 | 1.904 | 0.680 |
| 0110 | 8.662 | 32.769 | 13.951 | 10.627 | 6.369 | 1.567 |
| 0111 | 0 | 0 | 0 | 0.007 | 0 | 0 |
| 1000 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1001 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1010 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1011 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1100 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1101 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1110 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1111 | 0 | 0 | 0 | 0 | 0 | 0 |

Rounding can make the displayed sum slightly different from 100%. The tiny mixed `0111` span is not evidence of
a supported deliberate three-phase action: event-envelope/sampling tails can overlap across a decision boundary.
The runtime reports zero logical action-fidelity violations.

| Workload | Window ms | E duty | P duty | D duty | C duty | EPD overlap | Max uncovered gap ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 4699.40 | 0% | 21.11% | 86.46% | 0% | 8.66% | 0.25 |
| long-prefill | 12602.40 | 0% | 57.86% | 73.71% | 0% | 32.77% | 10.88 |
| bimodal | 16476.78 | 0% | 24.63% | 88.30% | 0% | 13.95% | 5.49 |
| mixed | 5441.48 | 11.26% | 37.26% | 65.74% | 0% | 16.71% | 3.33 |
| vision-heavy | 7079.55 | 13.12% | 36.44% | 60.86% | 0% | 13.41% | 5.53 |
| multi-image | 2926.12 | 13.37% | 34.20% | 59.41% | 0% | 9.60% | 2.84 |

Duty percentages are marginal, so overlapping phases are counted more than once. Exclusive mask states, not the
sum of phase duty values, define wall-clock coverage. Low uncovered idle does not imply that overlap has no value.

## 6. Long-prefill and bimodal: KV reservation/admission matters

The runtime reserves capacity at admission for `promptTokens + maxOutputTokens` in full-reservation mode. If
`ensureCapacity()` fails, the slot is released and the request remains unadmitted. A 24-slot configuration therefore
does not promise 24 executable requests with a 96-page undercommitted pool.

Using response prompt-token counts, output limits, and actual `server_admit`/`slot_released` events, the analysis
reconstructs full page reservations as `ceil((prompt + output_limit) / 128)`. This is strongest for text requests,
whose prompt counts are directly available; it is not a byte-accurate multimodal allocator trace.

| Workload | Peak admitted KV owners | Peak reconstructed pages | Minimum sampled free pages | Submit→admit mean/p95 ms | Admit→first token mean/p95 ms |
|---|---:|---:|---:|---:|---:|
| balanced | 24 | 52 | 45 | 0.016 / 0.027 | 81.59 / 213.58 |
| long-prefill | 12 | **96** | **0** | **1884.50 / 2730.47** | 338.56 / 864.29 |
| bimodal | 19 | **96** | **0** | **1926.68 / 5316.67** | 241.28 / 721.77 |
| mixed | 24 | 52 | 44 | 0.015 / 0.026 | 133.23 / 277.57 |
| vision-heavy | 20 | 37 | 59 | 0.013 / 0.026 | 165.13 / 271.92 |
| multi-image | 4 | 16 | 80 | 0.010 / 0.019 | 157.96 / 241.26 |

All reconstructed owners/pages return to zero at the end. Sampled free pages are dispatch-time snapshots and need
not capture the exact occupancy extremum; the reconstructed peak can therefore differ from `96 - sampled_min`.

The KV pool is 12,288 tokens; the frozen vLLM server reports 27,025 KV tokens. Full reservation can fill the former
with roughly twelve long requests before D24 is possible. This matches observed peak owners and maximum D12 on
long-prefill. There is no measured need to change from paged ownership to another allocator to explain this.

The earlier interpretation in note 295, "not KV capacity, E/P critical path," was too strong. These diagnostics
directly show a KV-capacity/reservation contribution to text TTFT. They do not quantify how much of the vLLM gap
would disappear with a larger pool; that requires a same-engine/runtime capacity A/B. The memory broker's zero
backpressure counter does not disprove page exhaustion in the independent server's admission path.

## 7. P candidate formation is a separate mechanism bottleneck

| Workload | P dispatches | Mean P BS | Useful P tokens/dispatch | D dispatches | Mean D BS |
|---|---:|---:|---:|---:|---:|
| balanced | 53 | 1.585 | 113.87 | 296 | 18.162 |
| long-prefill | 328 | 1.378 | 161.55 | 588 | 9.143 |
| bimodal | 177 | 1.531 | 167.16 | 1015 | 9.474 |
| mixed | 97 | 1.526 | 140.45 | 295 | 9.708 |
| vision-heavy | 103 | 1.767 | 168.94 | 474 | 5.063 |
| multi-image | 38 | 1.789 | 178.21 | 226 | 2.743 |

P8 is a capacity ceiling, not the observed batch size. Long-prefill has P1 in 257/328 dispatches (78.35%) and P8
only seven times. P128 makes many chunks, and admission/compatibility/ready timing keeps most batches far below
the 1024-token carrier budget.

Measured request-time scheduler snapshots contain the following P1 decisions with more than one P-ready row:

| Workload | P1 decisions with multiple P-ready rows | Those without any larger legal P frontier candidate |
|---|---:|---:|
| balanced | 13 | 13 |
| long-prefill | 82 | 82 |
| bimodal | 42 | 42 |
| mixed | 23 | 23 |
| vision-heavy | 30 | 30 |
| multi-image | 4 | 4 |

These are decision snapshots, not unique issued dispatches. They can include repeated/residual decisions. "Ready"
does not guarantee shape/initial-versus-continuation/exclusive compatibility. This observation therefore does not
prove that a larger P batch was legal, but it **does** show that the V3 selector was not offered one in these
snapshots. Next inspect compatible-row count and candidate formation before changing RLS scores.

The queue mechanism explicitly separates initial and continuation rows and filters exclusive and ragged shapes.
Some low BS is legitimate; removing those filters blindly would violate the runtime/kernel contract.

## 8. VLM: E queue lifetime dominates the first-token path

| Workload | Vision requests | E batches | Mean E BS | E=1/2/3/4 batch counts | Vision queue→E start mean/p95 ms | E start→done mean/p95 ms | E done→P ready mean ms | P ready→first P start mean/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mixed | 32 | 23 | 1.391 | 17 / 4 / 1 / 1 | **984.99 / 1982.18** | 49.35 / 84.51 | 0.003 | 33.87 / 140.69 |
| vision-heavy | 48 | 30 | 1.600 | 17 / 9 / 3 / 1 | **1743.43 / 3121.06** | 52.06 / 86.95 | 0.004 | 23.06 / 129.20 |
| multi-image | 20 | 12 | 1.667 | 5 / 6 / 1 / 0 | **713.00 / 1444.98** | 46.85 / 72.40 | 0.003 | 15.07 / 47.57 |

Request E start/done timestamps are host mechanism events, including realization/observation, not isolated GPU
kernel time. Separately, the recorded E engine averages 26.64/30.97/32.59 ms per batch on the three traces.
Queue statistics apply to vision requests; mixed/vision-heavy aggregate TTFT also includes text requests.

Three things coexist:

- KV has room: minimum sampled free pages are 44/59/80 in mixed/vision-heavy/multi-image.
- E occupies only 11.26--13.37% of the recorded envelope window.
- Vision requests wait far longer than one E execution; E4 is rarely formed.

This is a queue/formation/authority/lifetime diagnostic, not "the encoder kernel is simply too slow."
`encoderCapacityAvailable()` counts cross-phase downstream payloads, not just requests currently executing E.
The configured count limit is four, and payload lifetime extends through the consumer's prefill. Therefore an
idle E context cannot necessarily start new work until P releases enough vision ownership. More KV alone will
not fix this VLM case.

The first-to-last measured-metric differences also show sparse E-pair evidence:

| Workload | E-pair opportunities | No samples | Insufficient samples | Encoder dispatch deferral poll increments |
|---|---:|---:|---:|---:|
| mixed | 25 | 25 | 0 | 30232 |
| vision-heavy | 30 | 28 | 2 | 31385 |
| multi-image | 10 | 10 | 0 | 33574 |

These differences omit the very first measured dispatch and are not exact epoch totals. Deferral counts are
poll counts, not milliseconds or distinct missed E batches. They do not distinguish capacity blocking from all
arbiter/policy conditions. Sparse evidence, payload capacity, and physical profitability must be tested separately;
forcing all E overlap is not justified by this table.

## 9. Runtime realization versus policy CPU cost

| Workload | Dispatch/submission mean/p95 ms | Scheduler decision p95 us | D done→sampling collected mean/p95 ms | Token commit→next D start mean/p95 ms |
|---|---:|---:|---:|---:|
| balanced | 5.64 / 14.10 | 122.41 | 0.32 / 0.33 | 1.47 / 8.41 |
| long-prefill | 7.81 / 13.30 | 101.64 | 0.32 / 0.33 | 4.07 / 22.44 |
| bimodal | 5.16 / 12.45 | 92.21 | 0.31 / 0.32 | 1.30 / 7.50 |
| mixed | 5.83 / 13.05 | 106.99 | 1.69 / 8.57 | 3.38 / 26.61 |
| vision-heavy | 5.03 / 12.48 | 67.80 | 1.46 / 6.06 | 1.65 / 4.79 |
| multi-image | 4.99 / 7.47 | 54.48 | 1.48 / 6.08 | 0.88 / 1.21 |

Submission is `host_submission_end - host_dispatch_start`, not the duration of the `enqueueV3()` API alone. It
includes staging, callback work, and potentially multiple phases. Per-request D stage samples are token/request
weighted, not per-batch weighted. D start is a host timeline enqueue event, not the first GPU kernel timestamp.

The model's CPU arithmetic is not the largest measured host span. However the large submission envelope cannot
be declared additive GPU-idle time: some is hidden under outstanding GPU work, some can be inside the event
envelope, and synchronous sampling affects completion visibility. The graph-off contract and actual per-kernel
launch gaps remain an unquantified runtime opportunity.

### Nsight attempt: failed, not usable evidence

An additional same-config multi-image run used Nsight Systems 2026.1.3 with CUDA/NVTX tracing and no CPU sampling.
It exited with backend code 139 during startup shape warm-up, before HTTP health/measurement. A small `.nsys-rep`
was produced, but it contains no usable measured trace and is not used for mask, throughput, or kernel-idle claims.
The unprofiled six diagnostics completed successfully. No conclusion about kernel-level idle, memory bandwidth,
or Tensor Core utilization is drawn from this failed profiling run.

## 10. Next experiments: isolate causes, do not tune per workload

1. **Text capacity A/B:** same standalone P128 engine/runtime/policy and traces; vary only KV page budget while
   retaining D24/P8 and full reservation. Measure submit→admit, peak owners/pages, P/D BS, TTFT and E2E mean/p95.
   Do not reduce KV or change the allocator to hide undercapacity. Confirm engine/page-table capability before
   attempting a larger runtime pool; do not merely edit an engine config beyond its built limits.
2. **Vision lifetime A/B:** retain E4/P8/D24, chunk128, KV96 and policy. Vary only encoded downstream capacity
   four versus a memory-safe larger count. Keep E engine batch capacity unchanged. Record payload count/bytes,
   capacity-blocked time, E formation, release boundaries, TTFT and TPOT tails. This tests lifetime lookahead,
   not an E8 engine rebuild or a workload-specific overlap rule.
3. **Formation instrumentation:** in snapshots with multiple P-ready rows, expose eligible/compatible counts and
   why P2/P4/P8 is absent. Separate initial/continuation and vision/text compatibility from performance selection.
   Only then consider enlarging the bounded candidate frontier.
4. **Kernel realization:** obtain a successful measurement-window-only profiler capture. Quantify actual kernel
   union, launch gaps, submission, and sampling before claiming the event envelope represents occupied GPU time.
   Fix the profiler/startup interaction independently; do not silently change the performance contract.
5. **Promotion:** repeat promising same-runtime A/Bs at least three times, then all twelve. Always report TTFT,
   TPOT, E2E mean/p95, generated-token throughput, memory, output contract, calibration readiness, and actual masks.
   Reuse frozen vLLM only while its workload/comparison contract remains valid. No oracle workload label may enter
   the scheduler.

The architecture priority is **make useful work executable earlier**, not just "enable more overlap."
Stable paged ownership remains appropriate; pool undercommit/full reservation, vision payload lifetime, P
compatibility, and host realization are different mechanisms and should not be collapsed into an RLS-tuning issue.

## 11. Reproduction and retained data

- Script: `benchmarks/phase_serving/run_gemma_v3_activity_diagnostic.sh`.
- Analysis: `benchmarks/phase_serving/analyze_gemma_v3_bottleneck.py`.
- Result root: `.local/results/gemma4-v3-activity-diagnostic-20260913/`.
- Manifest: `manifest.json`; structured report: `bottleneck-analysis.json`.
- Each successful cell retains HTTP requests/summary, gateway telemetry, raw activity intervals/segments/summary,
  and `run-001/analysis/` derived data. The failed profiler attempt is under `nsight/multi-image/`.

```bash
bash benchmarks/phase_serving/run_gemma_v3_activity_diagnostic.sh
python3 benchmarks/phase_serving/analyze_gemma_v3_bottleneck.py \
  --result-root .local/results/gemma4-v3-activity-diagnostic-20260913
```

Run the reproduction under the configured GPU/container permission mechanism. Existing completed cells are
skipped; choose a distinct `RESULT_ROOT` for a new repeat. The script depends on the retained HTTP harness and input
trace paths recorded in the manifest. No models/engines/results are committed.

Validation: all six runs pass derived-data checks for exclusive mask sum 100%, complete measured lifecycles,
zero ending lease owners/pages, and useful D-token count equal to generated outputs minus first tokens.
First-chunk critical-path pairing and interpolated-p95 synthetic checks also pass.
Diagnostic script syntax and all applicable pre-commit checks pass. The YAPF hook needed the existing tool
environment's `platformdirs` on `PYTHONPATH`; no project dependency or runtime source was modified for that check.
