# 279. V3 P0--P6 validation results

## Outcome

P0--P6 are complete on the RTX 3080/Cosmos v0.10.1 environment. The campaign
used the same phase runtime binary, engines, 12 request traces, fixed-output
contract, generic calibration, batch limits, and memory limits for V0--V3.
No explicit TTFT or TPOT target was supplied to the phase runtime.

The principal result is that V3 remains the strongest of the four retained
policy generations, but the new causal ablation changes the interpretation of
why it wins:

```text
V0 Exact
  -> V1 Contextual Scalar
  -> V2 Scalar + deterministic transition
  -> V3 service-time-scaled no-SLO transition
```

Across the fresh three-repeat full-12 campaign, V3 improves token-throughput
geometric mean by **12.83% over V0**, **9.22% over V1**, and **9.47% over V2**.
It also improves geometric-mean TTFT p95 by 23.26% and E2E p95 by 11.87%
relative to V2. Fixed-output token traces match V0 on all 12 workloads for all
four variants.

The recovery filter is not the source of this gain. In a same-new-binary
ablation, disabling recovery improved the core-four token-throughput geometric
mean from 1,209.41 to 1,300.94 token/s. Audit telemetry shows that the default
recovery filter intervenes in 16.6%--48.7% of decisions, so it is not merely a
rare starvation escape. The evidence therefore supports service-time
normalization and removal of synthetic SLOs, while the current recovery policy
should be simplified or demoted after a full-12 promotion gate.

A fresh equal-contract vLLM campaign also completed. V3 wins token throughput
on 12/12 workloads with a 10.41% geometric-mean advantage. It wins E2E mean on
12/12 and E2E p95 on 11/12, but it does not dominate every latency metric.
vLLM completed only 2/3 vision-heavy repeats under the frozen 64-request,
3.5-GiB-KV contract; the failed repeat OOMed during the measured trace.

## P0--P1: fresh V0/V1/V2/V3 full-12 gate

All numbers are medians of three independent runs. Latency is reported in
milliseconds. The comparison uses the original P0 binary, before adding the
diagnostic recovery knobs; those knobs do not change default behavior.

| Workload | V2 token/s | V3 token/s | Token delta | TTFT p95 V2 -> V3 | TPOT p95 V2 -> V3 | E2E p95 V2 -> V3 |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 4,390.0 | 4,405.3 | +0.35% | 168.6 -> 156.1 | 13.98 -> 14.14 | 1,781.2 -> 1,742.3 |
| bimodal | 1,946.5 | 1,984.7 | +1.97% | 4,061.6 -> 3,986.4 | 29.39 -> 27.40 | 9,109.6 -> 8,932.1 |
| decode-heavy | 5,166.5 | 5,154.3 | -0.24% | 187.1 -> 154.9 | 11.37 -> 11.43 | 4,368.6 -> 4,339.8 |
| late-vision | 2,452.6 | 2,452.1 | -0.02% | 448.4 -> 458.3 | 9.65 -> 9.64 | 1,880.6 -> 1,880.6 |
| long-prefill | 1,190.1 | 1,344.8 | +13.00% | 2,642.7 -> 2,478.2 | 30.59 -> 26.74 | 5,985.9 -> 5,259.5 |
| mixed | 915.0 | 963.5 | +5.30% | 2,787.7 -> 2,564.0 | 36.41 -> 36.73 | 3,136.6 -> 2,955.2 |
| multi-image | 173.6 | 260.0 | +49.76% | 719.0 -> 400.8 | 13.01 -> 11.65 | 921.2 -> 615.1 |
| poisson | 1,698.2 | 1,902.1 | +12.00% | 1,532.2 -> 786.9 | 33.45 -> 40.71 | 2,415.6 -> 2,084.9 |
| short | 2,427.5 | 2,366.4 | -2.52% | 176.8 -> 191.9 | 25.15 -> 19.99 | 415.1 -> 413.8 |
| text-heavy | 1,480.3 | 1,813.5 | +22.51% | 1,951.2 -> 1,205.4 | 27.81 -> 37.73 | 2,231.3 -> 1,838.2 |
| vision-heavy | 548.0 | 630.8 | +15.10% | 4,014.4 -> 3,446.0 | 39.97 -> 38.31 | 4,426.0 -> 3,830.6 |
| wave-drain | 92.2 | 97.5 | +5.74% | 714.5 -> 531.7 | 13.03 -> 11.86 | 916.6 -> 738.6 |

The geometric-mean comparison is:

| Comparison | Token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 vs V0 | +12.83% | +12.00% | +22.22% | -4.72% | +0.03% | +5.19% | +15.57% |
| V3 vs V1 | +9.22% | +14.77% | +25.94% | -2.59% | -3.33% | +5.83% | +12.09% |
| V3 vs V2 | +9.47% | +11.78% | +23.26% | +0.60% | +1.31% | +7.68% | +11.87% |

Positive values mean V3 is better. V3 beats V2 on 9/12 token-throughput,
9/12 TTFT-mean, 10/12 TTFT-p95, 8/12 TPOT-mean, 7/12 TPOT-p95, 8/12
E2E-mean, and 11/12 E2E-p95 comparisons. V3 is therefore the aggregate
winner, not a pointwise dominator.

## P2: component attribution

Four diagnostic workloads were repeated three times with the same rebuilt
binary. Three configurations were compared:

- `final`: current V3 recovery and overdue-exploration suppression;
- `no-recovery`: service-time normalization remains, recovery is disabled;
- `always-frontier`: recovery is disabled and overdue unknown-overlap
  exploration is allowed.

| Variant | Core-four token/s geometric mean |
|---|---:|
| final | 1,209.41 |
| no-recovery | 1,300.94 |
| always-frontier | 1,296.72 |

| Workload | Final token/s | No-recovery token/s | Always-frontier token/s | Best interpretation |
|---|---:|---:|---:|---|
| balanced | 4,369.7 | 4,299.4 | 4,315.8 | recovery is 1.2--1.6% better |
| text-heavy | 1,942.2 | 1,980.5 | 1,980.2 | recovery costs about 2% |
| mixed | 961.6 | 1,098.6 | 1,099.5 | recovery costs about 14% |
| multi-image | 262.1 | 306.2 | 300.9 | recovery costs 14.8--16.8% |

No-recovery also lowers mixed TTFT p95 from 2,511.6 to 2,309.6 ms and E2E
p95 from 2,964.5 to 2,598.3 ms. On multi-image it lowers TTFT p95 from 397.2
to 308.3 ms and E2E p95 from 609.8 to 522.0 ms. Balanced is the counterexample:
removing recovery slightly lowers TTFT mean but regresses throughput, TPOT, and
E2E.

These results rule out the recovery filter as the main cause of V3's earlier
gain. The strongest retained V3 mechanism is the no-SLO service-scaled
representation and its changed candidate authority. The recovery rule is an
overactive secondary policy.

## P3: recovery sensitivity

Mixed and multi-image were repeated three times over recovery age and candidate
band values. The token-throughput geometric means are:

| Configuration | Token/s geometric mean |
|---|---:|
| no recovery | 579.98 |
| age 0.5 / band 1.0 | 469.13 |
| age 1.0 / band 0.5 | 495.59 |
| age 1.0 / band 1.0 | 502.08 |
| age 1.0 / band 2.0 | 568.37 |
| age 2.0 / band 1.0 | 522.21 |

The response is non-monotonic. A wider band approaches no-recovery throughput
and gives the best multi-image TPOT p95, but no one setting dominates. This is
not evidence for another workload-tuned constant. It is evidence that the
recovery filter is acting too early and too often to serve as a generic no-SLO
policy.

## P4: post-hoc SLO-goodput surface

The same 144 request-result files were re-evaluated without rerunning the GPU.
The grid contains TTFT thresholds 250/500/1,000/2,000 ms and TPOT thresholds
20/50/80/100 ms, with no E2E threshold: 2,304 run points and 768 aggregated
surface points.

Across surface points with positive goodput, V3 versus V2 has an 8.87%
normalized geometric gain, with 105 wins, 7 ties, and 80 losses. The per-trace
normalized change is:

| Workload | V3 vs V2 SLO-goodput surface |
|---|---:|
| balanced | -0.76% |
| bimodal | +4.03% |
| decode-heavy | -3.18% |
| late-vision | -2.69% |
| long-prefill | -2.03% |
| mixed | +3.08% |
| multi-image | +103.25% |
| poisson | +9.81% |
| short | -0.36% |
| text-heavy | +2.27% |
| vision-heavy | +18.25% |
| wave-drain | +20.29% |

This surface is the important qualification to the raw-throughput result.
V3's no-SLO policy moves the latency/throughput frontier, but does not dominate
all possible post-hoc utilities. `long-prefill` is a clear example: raw token
throughput improves 13.0% while the aggregate threshold surface falls 2.03%.
No workload label or threshold from this analysis was fed back into the policy.

## P5: fresh equal-contract vLLM

The vLLM comparison was rerun rather than reusing the older frozen results.
The contract was:

- identical measured JSON/image traces and fixed output lengths;
- EOS ignored on both runtimes;
- maximum 64 measured requests in flight;
- FP16 model and KV cache;
- vLLM KV budget 3.5 GiB and `max-num-seqs=80`;
- prefix caching disabled;
- one fresh vLLM server/container per independent repeat.

The initial 64-request all-at-once VLM warmup was not measured and caused an
artificial OOM, so VLM warmup was bounded to 16 requests. With that correction,
vision-heavy still OOMed in one of three measured repetitions. The table below
uses medians of the successful repetitions. Positive deltas mean V3 is better;
for latency, that means V3 is lower.

| Workload | Fresh vLLM successful | Token delta | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 3/3 | +15.65% | +49.83% | +25.26% | +4.10% | +17.14% | +19.78% | +16.02% |
| balanced | 3/3 | +2.07% | +42.91% | +38.56% | -2.49% | -4.11% | +2.61% | +1.64% |
| decode-heavy | 3/3 | +4.39% | +46.64% | +51.84% | +2.02% | +1.72% | +4.11% | +3.38% |
| long-prefill | 3/3 | +19.65% | -1.58% | +15.92% | +31.57% | +28.42% | +17.88% | +20.15% |
| bimodal | 3/3 | +5.97% | -20.72% | -51.50% | +24.81% | +26.87% | +9.87% | +0.98% |
| text-heavy | 3/3 | +40.32% | +58.24% | +42.83% | -41.26% | +8.96% | +12.22% | +27.01% |
| mixed | 3/3 | +4.35% | -1.51% | -0.85% | +42.74% | +56.23% | +27.41% | +5.65% |
| vision-heavy | 2/3 | +9.28% | +9.43% | +2.78% | +53.24% | +68.27% | +35.22% | +9.66% |
| poisson | 3/3 | +6.79% | +44.76% | +14.79% | +4.89% | +11.94% | +9.64% | +8.86% |
| wave-drain | 3/3 | +1.80% | -20.53% | -26.35% | +31.47% | +31.34% | +10.84% | -13.54% |
| multi-image | 3/3 | +6.61% | -20.92% | +0.29% | +18.79% | +28.39% | +8.80% | +6.01% |
| late-vision | 3/3 | +13.25% | +48.13% | +45.93% | +11.00% | +10.62% | +16.25% | +11.72% |

V3 wins token throughput 12/12, TTFT mean 7/12, TTFT p95 9/12, TPOT mean
10/12, TPOT p95 11/12, E2E mean 12/12, and E2E p95 11/12. The largest
remaining latency defects are bimodal TTFT, wave-drain TTFT/E2E p95, and
text-heavy TPOT mean. Therefore the supported claim is broad throughput and
E2E superiority, not universal per-metric dominance.

## P6: scheduler overhead and recovery invocation

Production runs were measured with dispatch telemetry disabled. The decision
cost is therefore not inflated by audit JSON serialization.

| Workload | Mean decision cost (us) | Run-p95 median (us) | Maximum (us) |
|---|---:|---:|---:|
| balanced | 121.33 | 219.03 | 337.73 |
| bimodal | 82.19 | 151.03 | 217.19 |
| decode-heavy | 98.54 | 212.52 | 363.53 |
| late-vision | 47.51 | 66.20 | 179.75 |
| long-prefill | 118.18 | 154.27 | 206.59 |
| mixed | 74.13 | 193.14 | 350.25 |
| multi-image | 30.81 | 71.72 | 102.02 |
| poisson | 72.82 | 215.73 | 373.64 |
| short | 59.46 | 127.20 | 174.12 |
| text-heavy | 94.80 | 242.44 | 410.31 |
| vision-heavy | 57.79 | 175.82 | 299.66 |
| wave-drain | 23.84 | 48.47 | 107.58 |

The earlier desired `p95 < 50 us` gate is met only by wave-drain. Although
phase actions are millisecond-scale and this cost is not the dominant current
GPU bottleneck, candidate materialization, snapshot traversal, and transition
preview need hot-path profiling and reduction before publication.

Recovery invocation was measured separately with audit telemetry enabled, so
the following runs are used only for policy attribution:

| Workload | Decisions | Recovery applications | Invocation rate |
|---|---:|---:|---:|
| balanced | 561 | 93 | 16.58% |
| text-heavy | 100 | 40 | 40.00% |
| mixed | 129 | 47 | 36.43% |
| multi-image | 76 | 37 | 48.68% |

This directly explains why tuning the recovery age or band can change an
entire scheduling trajectory. The default is not acting as a last-resort
starvation mechanism.

## Implementation added for this campaign

The runtime default remains unchanged. The following diagnostic controls were
added so the mechanism can be attributed without maintaining separate source
branches:

```text
TRT_EDGELLM_DISABLE_SERVICE_RECOVERY
TRT_EDGELLM_ALLOW_OVERDUE_EXPLORATION
TRT_EDGELLM_SERVICE_RECOVERY_AGE_QUANTA
TRT_EDGELLM_SERVICE_RECOVERY_BAND_QUANTA
```

The global policy configuration is copied consistently to the E/P/D
coordinators. Recovery-disable also covers the queue scheduler's expired-decode
preservation path. Unit tests cover validation and the disabled-recovery path.

Reusable analysis and replay support was added for:

- V0--V3 multi-policy replay and diagnostic environment overlays;
- post-hoc TTFT/TPOT SLO surfaces;
- production scheduler-decision overhead extraction;
- equal-contract vLLM command generation;
- repeated fresh-container HTTP execution with continue-on-error support;
- recovery application counts in V3 audit telemetry.

## Verification

- V0/V1/V2/V3 full-12: 48/48 policy/workload jobs succeeded, three repeats
  each, with 12/12 fixed-output identity for every variant.
- Rebuilt C++ targets: `llm_phase_context_smoke` and `unitTestRuntime` passed.
- Filtered C++ policy tests: 70/70 passed.
- Python contract and analyzer tests: 18/18 passed under `unittest`.
- Python syntax compilation: passed.
- `git diff --check`: passed.
- Fresh vLLM: 35/36 bounded-warmup measured repetitions succeeded; the sole
  failure was a measured vision-heavy OOM.

## Retained artifacts

Primary:

- `.local/results/v0101-forward-port/v3-service-scale-20260910/no-slo-v0-v3-full12-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260910/component-ablation-core4.json`
- `.local/results/v0101-forward-port/v3-service-scale-20260910/recovery-sensitivity-core2.json`
- `.local/results/v0101-forward-port/v3-service-scale-20260910/vllm-fresh-equal-summary.json`
- `.local/results/v0101-forward-port/v3-service-scale-20260910/v3-recovery-audit-core4/recovery-audit.json`

Supporting ablations:

- `.local/results/v0101-forward-port/v3-service-scale-20260910/ablation-v3-final-core4-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260910/ablation-v3-no-recovery-core4-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260910/ablation-v3-always-frontier-core4-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260910/sensitivity-age-0p5-core2-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260910/sensitivity-age-2p0-core2-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260910/sensitivity-band-0p5-core2-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260910/sensitivity-band-2p0-core2-3x`

Fresh vLLM raw runs are split across the `vllm-fresh-equal-*` result roots in
the same campaign directory. The summary records the final bounded-warmup
contract and success count.

## Decision and next gate

V3 remains the best retained no-SLO architecture, but the current recovery
filter should not be described as a validated generic policy. The next change
must not select a new age/band value from these workloads. The defensible order
is:

1. run `no-recovery` and a deliberately loose starvation-only fallback over
   the complete full-12 same-binary gate;
2. promote a simpler policy only if it preserves balanced behavior while
   retaining the mixed/multi-image gains;
3. profile and reduce scheduler decision p95, starting with snapshot traversal,
   candidate cloning/materialization, and transition rollout;
4. address the remaining bimodal and wave-drain tail counterexamples using
   observable request state, not workload labels;
5. repeat the final policy on a second GPU and model before claiming portable
   service normalization.

No default was changed by P0--P6 because the component evidence is not yet a
full-12 promotion result.
