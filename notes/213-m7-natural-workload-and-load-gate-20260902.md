# M7 Natural Workload and Request-Load Gate

## 결론

M7 pair-conformal completion uncertainty를 포함한 최신 통합 서버를 실제
HTTP request 계약으로 다시 측정했다. 12개 workload는 각각 fresh process
3회, request-load trace는 39/48.8/97.5 offered req/s에서 각각 5회 실행했다.

결론은 네 가지다.

1. 12개 workload와 15개 load 반복에서 token hash는 모두 반복 간 exact
   identity를 유지했고, VLM semantic gate도 통과했다.
2. 최신 Current는 frozen vLLM보다 token throughput이 8/12 workload에서
   높고, TTFT mean 10/12, TPOT p95 12/12, E2E mean/p95 10/12에서 짧다.
   하지만 long-prefill, bimodal, wave/drain, multi-image throughput은
   vLLM보다 낮다.
3. 이전 P+D best와 비교하면 최신 Current의 12-workload token throughput
   geometric mean은 3.64% 낮다. 특히 bimodal, multi-image, poisson과
   long-prefill이 회귀했다. 따라서 현재 결과는 promotion pass가 아니다.
4. 48.8 req/s의 최신 full-VLM-ready Current는 35.40 req/s이고, 이전
   P+D best 40.47 req/s와 frozen vLLM 40.91 req/s보다 낮다. 반면 97.5
   req/s에서 joint-SLO pass는 이전 27.92%에서 최신 median 78.82%로
   높아졌다. 최신 controller가 overload tail을 보호하는 대신 saturation
   knee 이전부터 service capacity를 과도하게 양보하는 trade-off가 생겼다.

M7 conformal calibration 자체가 회귀 원인은 아니다. 같은 full-VLM
계약에서 conformal을 끄면 throughput은 오히려 1.47% 낮아졌다. vision
context 상주 비용도 회귀의 일부만 설명한다. vision context를 적재하지
않으면 48.8 throughput은 1.10% 회복되지만 이전 best에는 도달하지 못했다.

## 실험 계약

- model: `nvidia/Cosmos-Reason2-2B`, FP16, 비양자화
- GPU: RTX 3080 10 GiB
- engine: P8/D64, fixed prefill chunk 128, stable slots 80
- execution: independent TensorRT E/P/D contexts, shared CUDA context
- server/client in-flight cap: 80/64 for the 12-workload gate, 80/80 for
  request-load traces
- request path: OpenAI-compatible HTTP gateway and asynchronous request adapter
- VLM path: E4, vision-prefill batch 4, asynchronous GPU preparation, encoder
  arbiter, vision-prefix prefill
- policy: profile-free Global active, P+D/E+P/E+D contextual heads active,
  workload labels absent, safe-probe multiplier zero
- M7: process-local pair conformal enabled, minimum observations 16, window
  128, target coverage 95%
- warmup: 64 requests, maximum 32 output tokens
- primary metrics: CUDA dispatch metrics disabled; client-side request timing
  and GPU memory sampling enabled
- statistics: 12 workloads use the median aggregate of three fresh processes;
  load points use the median aggregate of five fresh processes
- joint SLO: TTFT <= 500 ms, TPOT <= 50 ms, E2E <= 2500 ms

M7 completion uncertainty is observation/shadow authority only. It does not
promote or veto production actions in this gate. Therefore this experiment
measures the current M6 action policy plus M7 observation overhead, not an
active conformal policy promotion.

## Natural completion-calibration coverage

Before the primary gate, balanced, vision-heavy, and the three load points
were run with detailed phase metrics.

| Direction | Observations | Held-out | Incumbent coverage | Newcomer coverage | False-safe |
|---|---:|---:|---:|---:|---:|
| P -> D | 478 | 378 | 98.94% | 98.41% | 0 |
| D -> P | 22 | 22 | 100.00% | 100.00% | 0 |
| E -> P / P -> E | 0 | 0 | n/a | n/a | 0 |
| E -> D / D -> E | 0 | 0 | n/a | n/a | 0 |

Controlled six-direction calibration in Note 212 covered every direction,
but the natural active policy produced only P/D completion pairs. This is an
important promotion blocker: a process-local E pair model cannot become
confident from these natural traces without a bounded, SLO-safe calibration
path. Workload-name or E-batch lookup rules must not be added to solve this.

The metrics-heavy diagnostic measured whole-scheduler decision latency at
183.22 us mean and 539.73 us p95. This includes JSON telemetry and candidate
instrumentation and is not the primary serving hot-path result. It is still
above the desired p95 budget, so active M7 promotion remains blocked.

## Fresh Current: 12 workloads

Latency cells are `mean / p95` in milliseconds.

| Workload | Token/s | TTFT | TPOT | E2E | Peak MiB |
|---|---:|---:|---:|---:|---:|
| short | 2509.6 | 88.1 / 166.1 | 13.46 / 22.25 | 329.0 / 408.7 | 9237 |
| balanced | 4504.1 | 67.8 / 166.5 | 12.22 / 13.67 | 1109.6 / 1737.6 | 9237 |
| decode-heavy | 5223.3 | 65.1 / 174.9 | 10.66 / 11.35 | 2822.9 / 4316.3 | 9237 |
| long-prefill | 1111.0 | 2237.0 / 2845.9 | 28.64 / 33.55 | 4684.5 / 6666.7 | 9237 |
| bimodal | 1738.2 | 2142.9 / 4553.8 | 20.02 / 29.58 | 4912.3 / 10273.4 | 9237 |
| text-heavy | 1894.3 | 337.3 / 1150.2 | 26.01 / 41.05 | 1687.4 / 1781.0 | 9315 |
| mixed | 1038.3 | 775.7 / 2308.8 | 33.16 / 41.23 | 2349.2 / 2758.6 | 9323 |
| vision-heavy | 635.5 | 1552.7 / 3407.6 | 28.57 / 39.44 | 2659.4 / 3815.3 | 9363 |
| poisson | 1804.5 | 227.2 / 876.8 | 24.65 / 43.39 | 1798.4 / 2241.6 | 9321 |
| wave/drain | 95.1 | 379.7 / 512.8 | 9.14 / 12.38 | 662.9 / 720.3 | 9383 |
| multi-image | 208.4 | 396.3 / 565.4 | 9.85 / 13.81 | 702.3 / 766.6 | 9379 |
| late-vision | 2536.7 | 124.7 / 459.0 | 9.30 / 9.37 | 1457.4 / 1817.6 | 9311 |

Correctness:

- all 12 workloads: 3/3 repeat token hashes identical;
- all VLM workloads: semantic pass rate 100%;
- multi-image, which had shown a prior FP16 greedy branch instability, is
  deterministic in this gate;
- all requests completed and no invalid slot, ownership, or HTTP failure was
  observed.

Memory:

- text-only full-VLM-ready residency: 9237 MiB;
- maximum observed VLM residency: 9383 MiB;
- RTX 3080 10240 MiB 기준 headroom: 857--1003 MiB;
- identical text path without vision context: 7799 MiB;
- always-ready vision context cost: 1438 MiB.

The 1438 MiB is not a KV-cache increase. It is the integrated visual engine,
weights, workspace, and context residency. It explains only a small part of
the text throughput regression, but it removes much of the formerly reported
memory advantage over a full VLM vLLM server.

## Difference from the previous P+D best

The reference is the active P+D contextual 12x3 gate in
`p52-contextual-pd-final-12x3`. Positive latency percentages mean the latest
Current is faster; positive token percentages mean higher throughput.

| Workload | Token/s delta | TTFT mean | TPOT mean | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|
| short | +1.3% | +0.0% | -0.8% | +1.7% | +1.1% |
| balanced | -0.7% | -5.6% | -0.9% | -1.0% | -2.4% |
| decode-heavy | -0.8% | -1.0% | -1.2% | -1.1% | -1.5% |
| long-prefill | -5.6% | -8.1% | -5.2% | -6.4% | -5.5% |
| bimodal | -9.0% | -13.3% | -9.9% | -12.0% | -13.6% |
| text-heavy | -3.0% | +1.5% | -3.3% | -2.5% | -3.1% |
| mixed | -3.9% | +5.7% | -30.5% | -16.1% | -4.2% |
| vision-heavy | -3.5% | -1.7% | -28.0% | -12.3% | -3.8% |
| poisson | -8.4% | +1.1% | -14.0% | -13.1% | -9.7% |
| wave/drain | +0.1% | -11.3% | -26.8% | -15.7% | +0.4% |
| multi-image | -9.8% | -19.6% | -26.2% | -23.0% | -10.9% |
| late-vision | +0.3% | +28.7% | +0.2% | +3.5% | +0.2% |

Across the 12 workloads, token-throughput geometric mean is 3.64% below the
P52 reference and the median per-workload delta is -3.21%. Only short,
wave/drain, and late-vision are faster. The largest mechanism regressions are
not an E-head action effect: natural E-head selection remains absent. They are
associated with the newer outstanding-completion/WAIT and successor-preview
path plus full-VLM context residency.

## Frozen vLLM comparison

The trace/model/output contract is unchanged, so the frozen fresh vLLM 12x3
run is reused. No redundant vLLM execution was performed. Latency cells show
`Current mean/p95 vs vLLM mean/p95` in milliseconds.

| Workload | Token/s delta | TTFT Current vs vLLM | TPOT Current vs vLLM | E2E Current vs vLLM |
|---|---:|---:|---:|---:|
| short | +26.5% | 88.1/166.1 vs 175.1/264.0 | 13.46/22.25 vs 13.40/24.88 | 329.0/408.7 vs 425.0/503.7 |
| balanced | +4.3% | 67.8/166.5 vs 147.4/365.0 | 12.22/13.67 vs 15.20/17.40 | 1109.6/1737.6 vs 1437.8/2244.1 |
| decode-heavy | +7.6% | 65.1/174.9 vs 154.5/394.2 | 10.66/11.35 vs 14.00/15.03 | 2822.9/4316.3 vs 3768.8/5812.4 |
| long-prefill | -0.9% | 2237.0/2845.9 vs 2947.6/4257.2 | 28.64/33.55 vs 32.30/37.65 | 4684.5/6666.7 vs 5694.2/7860.8 |
| bimodal | -7.0% | 2142.9/4553.8 vs 2513.5/4291.8 | 20.02/29.58 vs 23.70/42.84 | 4912.3/10273.4 vs 5641.8/10328.2 |
| text-heavy | +15.9% | 337.3/1150.2 vs 427.7/1232.2 | 26.01/41.05 vs 29.30/47.36 | 1687.4/1781.0 vs 1943.7/2037.8 |
| mixed | +12.7% | 775.7/2308.8 vs 882.4/2541.4 | 33.16/41.23 vs 47.10/84.02 | 2349.2/2758.6 vs 3004.8/3140.9 |
| vision-heavy | +9.7% | 1552.7/3407.6 vs 1710.7/3691.4 | 28.57/39.44 vs 63.70/119.58 | 2659.4/3815.3 vs 4119.1/4229.4 |
| poisson | +0.2% | 227.2/876.8 vs 580.9/902.7 | 24.65/43.39 vs 19.50/45.67 | 1798.4/2241.6 vs 1811.0/2266.6 |
| wave/drain | -0.8% | 379.7/512.8 vs 253.1/418.6 | 9.14/12.38 vs 12.40/17.26 | 662.9/720.3 vs 638.5/649.4 |
| multi-image | -14.8% | 396.3/565.4 vs 265.7/402.6 | 9.85/13.81 vs 12.20/16.32 | 702.3/766.6 vs 642.6/653.9 |
| late-vision | +7.5% | 124.7/459.0 vs 156.0/631.5 | 9.30/9.37 vs 7.40/9.93 | 1457.4/1817.6 vs 1574.8/1954.5 |

Win counts:

| Metric | Current wins |
|---|---:|
| token throughput | 8/12 |
| TTFT mean / p95 | 10/12 / 9/12 |
| TPOT mean / p95 | 9/12 / 12/12 |
| E2E mean / p95 | 10/12 / 10/12 |

Token-throughput geometric mean is 4.57% above frozen vLLM, but this average
must not hide the multi-image -14.8% and bimodal -7.0% regressions. Current is
not uniformly better than vLLM.

## Request-load sweep

### Latest Current

Five-run medians are reported. Latency cells are `mean / p95` milliseconds.

| Offered req/s | Raw req/s | Token/s | SLO pass | SLO goodput req/s | TTFT | TPOT | E2E |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 39.0 | 33.161 | 2873.98 | 94.44% | 31.375 | 74.0 / 252.1 | 15.49 / 20.19 | 1395.5 / 2517.5 |
| 48.8 | 35.396 | 3067.69 | 87.50% | 30.849 | 226.9 / 481.8 | 17.18 / 20.06 | 1690.1 / 2732.3 |
| 97.5 | 37.121 | 3217.12 | 78.82% | 29.258 | 342.7 / 522.3 | 17.70 / 19.89 | 1847.0 / 2912.0 |

Failure attribution averaged over five runs:

| Offered req/s | Pass | TTFT only | E2E only | TTFT+E2E | Any TPOT failure |
|---:|---:|---:|---:|---:|---:|
| 39.0 | 272.6 | 0.0 | 15.4 | 0.0 | 0.0 |
| 48.8 | 248.8 | 8.4 | 28.0 | 2.8 | 0.0 |
| 97.5 | 224.6 | 24.0 | 32.4 | 7.0 | 0.0 |

The raw-throughput knee still exists, but the latest policy has flattened the
SLO-goodput curve: 31.38 -> 30.85 -> 29.26 req/s. This is much more robust at
97.5 offered req/s than the earlier controller, yet it reaches only 37.12 raw
req/s at that point.

### Historical anchors

The first historical Current row is the P52-era repeated baseline; the second
is the later P55 best at the matched 80 client cap. vLLM is the frozen five-run
baseline. The binaries/policies differ, so these are regression anchors rather
than same-binary component ablations.

| Runtime/version | Offered | Raw req/s | SLO pass | Goodput req/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Earlier Current | 39.0 | 34.470 | 100.00% | 34.470 | 34.2 / 129.0 | 13.92 / 17.36 | 1220.3 / 2088.4 |
| Latest Current | 39.0 | 33.161 | 94.44% | 31.375 | 74.0 / 252.1 | 15.49 / 20.19 | 1395.5 / 2517.5 |
| frozen vLLM | 39.0 | 34.355 | 100.00% | 34.355 | 39.5 / 64.8 | 10.28 / 12.82 | 918.5 / 1505.4 |
| P55 Current best | 48.8 | 40.472 | 99.65--100% | 40.472 | 94.6 / 282.6 | 13.60 / 15.71 | 1251.0 / 2000.5 |
| Latest Current | 48.8 | 35.396 | 87.50% | 30.849 | 226.9 / 481.8 | 17.18 / 20.06 | 1690.1 / 2732.3 |
| frozen vLLM | 48.8 | 40.908 | 100.00% | 40.908 | 51.4 / 84.3 | 13.53 / 17.66 | 1205.9 / 2121.3 |
| Earlier Current | 97.5 | 41.225 | 27.92% | 11.508 | 1268.4 / 2881.5 | 15.80 / 17.92 | 2607.2 / 4024.4 |
| Latest Current | 97.5 | 37.121 | 78.82% | 29.258 | 342.7 / 522.3 | 17.70 / 19.89 | 1847.0 / 2912.0 |

At 48.8, Latest Current is 12.54% below P55 raw throughput and 23.78% below
P55 request goodput. Against frozen vLLM it is 13.47% lower in raw throughput
and 24.59% lower in request goodput. This is a hard promotion failure.

At 97.5, however, Latest Current has 9.96% lower raw throughput than Earlier
Current but 154% higher SLO goodput. The policy is therefore not simply slower;
it has moved along the capacity/tail trade-off surface too far toward bounded
tail service before the 48.8 knee.

## Focused 48.8 A/B

| Variant | Repeats | Raw req/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| latest full-VLM, conformal on | 5 | 35.396 | 226.9 / 481.8 | 17.18 / 20.06 | 1690.1 / 2732.3 | 9237 |
| latest full-VLM, conformal off | 3 | 34.876 | 233.9 / 484.9 | 17.46 / 20.24 | 1721.5 / 2815.6 | 9237 |
| latest text-lean, conformal on | 3 | 35.787 | 222.4 / 472.0 | 16.92 / 20.23 | 1660.9 / 2647.6 | 7799 |
| latest full-VLM, legacy-compatible | 3 | 29.544 | 333.0 / 543.2 | 20.90 / 22.62 | 2118.3 / 3311.7 | 9237 |

Interpretation:

- conformal on/off is within run-to-run variation and disabling it does not
  recover performance;
- removing vision residency saves 1438 MiB and improves raw throughput by
  only 1.10%;
- replacing the complete selector with legacy-compatible arbitration is not a
  valid no-WAIT ablation and is 16.53% slower than Latest Current;
- the latest logs contain about 60 decode-refill WAIT selections per ordinary
  12-workload run and about 293 at 48.8, whereas the P55 best log records zero;
- this correlation is not sufficient to blame WAIT alone because
  legacy-compatible arbitration also changes candidate ranking and loses much
  more performance.

The next experiment must add a same-selector `WAIT shadow/no-authority` mode,
not substitute the legacy selector. That produces the missing causal A/B:
identical candidate builder, contextual heads, and action ranking, with only
the concrete WAIT action prevented from delaying dispatch.

## Promotion decision

M7 remains **shadow-only / not promoted**.

Passed:

1. controlled six-direction conformal coverage;
2. natural P/D coverage above the 95% target;
3. zero conformal false-safe observations in the measured natural set;
4. exact repeat token identity across the complete primary gate;
5. VLM semantic correctness.

Failed or incomplete:

1. natural E-related completion coverage is zero;
2. metrics-heavy whole-scheduler p95 is above the desired hot-path budget;
3. latest 12-workload throughput is 3.64% geometric-mean below P52;
4. 48.8 req/s raw throughput and SLO goodput regress materially versus both
   P55 Current and frozen vLLM;
5. full-VLM-ready memory is 1.438 GiB above the text-lean process.

## Next implementation order

1. Add a policy-neutral `global WAIT shadow` switch. Candidate generation and
   scoring remain active, but selected WAIT is recorded and immediate service
   proceeds. Repeat 39/48.8/97.5 five times.
2. Attribute each WAIT to `now` and `future` equal-work horizons, successor D
   cohort change, protected slack, and realized next dispatch. Reject WAIT when
   its realized two-action regret is positive.
3. Keep the overload tail benefit while restoring the 39/48.8 knee. The gate
   is: 48.8 median raw throughput >= 40 req/s, joint-SLO pass >= 99%, and no
   12-workload metric regression beyond 3% versus P52.
4. Add bounded E calibration only from already-outstanding completions and
   generous protected slack. Do not add workload names, fixed E-size rules,
   TTL, or an external registry.
5. After the mechanism gate passes, rerun the same 12x3 and 3x5 load matrix.
   Reuse frozen vLLM until model, trace, output, memory, or SLO contract changes.

## Artifacts

- primary Current 12x3:
  `.local/inflight-m7-current-12x3-20260902/`
- primary Current load 3x5:
  `.local/inflight-m7-current-load-5x-20260902/`
- natural M7 shadow diagnostics:
  `.local/inflight-m7-natural-20260902/`
- natural conformal summary:
  `.local/inflight-m7-natural-20260902/contextual-shadow-summary.json`
- 48.8 focused A/B:
  `.local/inflight-m7-ablation-20260902/`
- previous P+D 12x3 reference:
  `.local/transition-aware-20260830/p52-contextual-pd-final-12x3/`
- previous matched-capacity 48.8 best:
  `.local/transition-aware-20260830/p55-contextual-pd-saturation-matched-5x/`
- frozen vLLM 12x3:
  `.local/profile-free-global-20260827/r4-vllm-12x3/`
- frozen vLLM 39/48.8 repeat:
  `.local/transition-aware-20260830/p6-repeat-baseline/vllm/`
