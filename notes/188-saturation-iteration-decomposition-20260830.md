# 48.8 req/s Saturation Iteration Decomposition

## 결론

권장 실행 순서의 두 번째 단계로 Current의 48.8 offered req/s 경계를 request lifecycle, sampling ticket, dispatch host path, CUDA phase activity까지 연결해 분해했다.

가장 중요한 결론은 다음과 같다.

- 이번 trace에서 P/D GPU는 측정 구간의 95.11% 동안 작업 중이었다. 단순 GPU idle이 주병목은 아니다.
- Global selector 시간은 D dispatch 평균 0.154 ms, p95 0.272 ms이고, dispatch wall에서 CUDA-event GPU makespan을 제외한 realization residual은 평균 0.054 ms, p95 0.096 ms다. selector 또는 enqueue 이후 launch/observe residual만으로 48.8 경계의 수백 ms TTFT를 설명할 수 없다.
- server는 구성상 80 stable slot을 보유하지만 D64 cohort에 맞춰 실제 in-flight admission을 64로 내린다. 측정 중 active request가 64인 구간은 3906.1 ms였고, admission을 1 ms 넘게 기다린 162개 요청 전부가 정확히 active=64에서 submit됐다.
- 이 162개 요청의 server admission wait는 평균 298.015 ms, 최대 431.500 ms다. 따라서 48.8 경계의 TTFT 붕괴를 직접 만드는 위치는 `server submit -> active slot admission`이다.
- D sampling ticket의 CUDA work는 ticket당 평균 0.151 ms지만 `submit -> ready observed`는 평균 1.009 ms, p95 5.275 ms다. 약 0.859 ms 평균 residual은 sampling/event polling이 slot lifetime과 다음 decode-ready 시각을 늘리는 보조 병목이다.
- 80 admission을 강제로 허용하면 OOM이 아니라 decode cohort selection invariant에서 종료된다. eligible row 65개를 count한 뒤 cohort ID 집합은 D64로 제한되어 65번째 row를 찾지 못하고 `Eligible decode work disappeared during batch selection` 예외가 발생했다.

따라서 다음 구현은 workload별 selector tuning이 아니다. 먼저 D64를 초과하는 active sequence를 안전하게 유지하면서 매 iteration 최대 64 row만 선택하는 decode cohort formation 계약을 고쳐야 한다. 그 뒤 admission 64/72/80 A/B를 반복해 TTFT와 SLO-goodput 회복량을 측정한다.

## 실험 계약

- model: `nvidia/Cosmos-Reason2-2B`, 비양자화
- engine/runtime: Current P8/D64, fixed prefill chunk 128, 80 stable slots
- execution: independent P/D TensorRT contexts와 Global selector
- trace: P5와 동일한 materialized real-request trace, offered 48.8 req/s
- warmup: 64 requests, output 32 tokens
- measured requests: 288
- client maximum in-flight: 80
- joint SLO: TTFT <= 500 ms, TPOT <= 50 ms, E2E <= 2500 ms
- timeline: `PHASE_TIMELINE`, `PHASE_METRIC`, CUDA-event phase activity를 같은 dispatch/request ID로 연결
- vLLM 비교: workload와 계약이 같으므로 fresh 실행하지 않고 5-run frozen baseline을 재사용

이번 Current run은 상세 계측을 켠 단일 diagnostic run이다. raw throughput은 37.448 req/s로 비계측 5-run 평균 37.877 req/s보다 1.13% 낮았다. 원인 위치를 찾는 진단에는 사용하지만 promotion headline으로 사용하지 않는다.

## 추가한 lifecycle 계측

각 sampling ticket에는 stable sequence ID를 부여하고 다음 전이를 request별로 기록한다.

```text
prefill_done
  -> prefill_sampling_submit
  -> prefill_sampling_ready
  -> prefill_sampling_collected
  -> prefill_token_committed
  -> decode_ready

decode_done
  -> decode_sampling_submit
  -> decode_sampling_ready
  -> decode_sampling_collected
  -> decode_token_committed
  -> decode_ready 또는 slot_released
```

dispatch에는 다음 host/GPU 시각을 기록한다.

```text
scheduler decision duration
host dispatch start
host submission end
CUDA-event GPU makespan
host completion observation
```

`host submission duration`은 첫 enqueue 이후 GPU 실행과 겹칠 수 있으므로 GPU makespan과 더하지 않는다. 안전하게 해석 가능한 realization residual은 `dispatch wall - GPU makespan`이다.

## End-to-end 결과와 고정 vLLM 기준선

| Runtime | Runs | Raw req/s | SLO pass | Goodput req/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current, detailed instrumentation | 1 | 37.448 | 69.10% | diagnostic only | 262.21 / 673.01 | 16.02 / 18.04 | 1625.80 / 2415.74 |
| Current, frozen non-instrumented | 5 | 37.877 +/- 0.139 | 74.93 +/- 2.78 pp | 28.383 +/- 1.134 | 239.11 / 619.94 | 15.76 / 17.81 | 1580.55 / 2367.64 |
| vLLM, frozen | 5 | 40.908 +/- 0.029 | 100% | 40.908 +/- 0.029 | 51.44 / 84.27 | 13.53 / 17.66 | 1205.87 / 2121.30 |

latency는 모두 scheduled-arrival 기준이며 단위는 ms다. 계측 run의 runtime token hash는 기존과 같은 `f51d...75ca`이고 288/288 request가 완료됐다.

## GPU와 dispatch realization

### Stream activity

측정 active span은 7688.83 ms다.

| 구간 | 시간 (ms) | 비율 |
|---|---:|---:|
| Prefill active | 3129.885 | 40.71% |
| Decode active | 4190.262 | 54.50% |
| P+D overlap | 7.024 | 0.09% |
| E/P/D/Copy 모두 idle | 375.706 | 4.89% |

text-only trace이므로 E와 Copy activity는 없다. 823 dispatch 중 P는 293회, D는 530회다. D dispatch 사이 GPU gap은 평균 6.744 ms, p95 26.831 ms, 최대 80.256 ms이며 25 ms 초과 gap은 28회다.

### Dispatch host path

| Phase | Dispatches | Scheduler mean/p95 | Host submission mean/p95 | GPU mean/p95 | Wall mean/p95 | Residual mean/p95 |
|---|---:|---:|---:|---:|---:|---:|
| P | 293 | 0.304 / 0.616 | 4.693 / 4.987 | 10.666 / 12.661 | 10.746 / 12.735 | 0.080 / 0.153 |
| D | 530 | 0.154 / 0.272 | 1.409 / 2.036 | 7.763 / 8.422 | 7.817 / 8.494 | 0.054 / 0.096 |

단위는 ms다. P host submission은 약 4.7 ms지만 GPU service와 겹친다. dispatch wall이 GPU makespan보다 평균 0.05--0.08 ms만 길다는 점에서, 이번 trace의 큰 request queue는 `TRT enqueue -> kernel start`의 순수 launch gap으로 설명되지 않는다.

## Request critical path

288개 요청 전체의 주요 구간은 다음과 같다.

| 구간 | Mean (ms) | p95 (ms) | 해석 |
|---|---:|---:|---|
| Client scheduled arrival -> send | 72.455 | 269.494 | client in-flight cap에서 발생하는 앞단 대기 |
| Server submit -> admit | 167.635 | 390.967 | 가장 큰 TTFT 구성요소 |
| Admitted P initial queue | 3.312 | 11.531 | admission 후 P formation은 작음 |
| P active total | 12.901 | 21.374 | 실제 P GPU service |
| P chunk gaps | 0.059 | 0.390 | fixed-128 chunk 사이 gap은 작음 |
| P sampling submit -> ready | 0.351 | 2.469 | 첫 token sampling event |
| Frontend -> backend TTFT bookkeeping | 5.491 | 9.431 | HTTP/adapter 구간 |
| First token -> first D | 6.763 | 29.396 | initial decode transition |
| D active total/request | 697.791 | 1064.766 | 전체 output에 걸친 D GPU service 합 |
| D inter-dispatch gaps/request | 652.787 | 1137.167 | 전체 output에 걸친 gap 합 |
| Decode-ready queue/request | 571.215 | 1060.448 | ready 이후 다음 D에 포함될 때까지 합 |
| Commit -> decode-ready/request | 0.028 | 0.046 | state commit 후 queue publish는 작음 |
| Final commit -> slot release | 0.0008 | 0.0014 | release 자체는 사실상 즉시 |

`decode sampling submit -> ready`의 request당 합은 평균 88.082 ms지만 요청마다 여러 output token을 포함한다. ticket 단위 결과를 별도로 봐야 한다.

## Sampling ticket 분해

| Phase | Tickets | Submit -> ready mean/p95 | CUDA sampling mean/p95 | Ready observation residual mean/p95 |
|---|---:|---:|---:|---:|
| P | 257 | 0.383 / 2.470 | 0.025 / 0.037 | 0.370 / 2.444 |
| D | 530 | 1.009 / 5.275 | 0.151 / 0.222 | 0.859 / 5.095 |

단위는 ms다. `ready -> collect`는 request당 합 평균 0.119 ms, `collect -> commit`은 평균 1.083 ms다. GPU sampling kernel 자체보다 completion event가 server poll에서 관측될 때까지의 residual이 더 크다. 이 residual은 D ready 시간을 늦추고 slot lifetime을 간접 증가시키지만, 수백 ms admission wait의 직접 원인은 아니다.

## Pass/fail 인과 분리

계측 run에서 joint SLO pass는 199개, fail은 89개였다.

| Group | Client dispatch mean/p95 | Scheduled TTFT mean/p95 | Server admission mean/p95 | P initial queue mean | TPOT mean/p95 |
|---|---:|---:|---:|---:|---:|
| Pass | 9.709 / 99.742 | 114.443 / 471.190 | 81.955 / 367.904 | 4.096 | 16.818 / 18.094 |
| Fail | 212.752 / 362.356 | 592.596 / 732.859 | 359.212 / 406.808 | 1.561 | 14.226 / 17.460 |

실패 그룹은 오히려 TPOT가 더 짧고 admission 이후 P queue도 더 작다. 실패는 decode iteration 자체가 느려 threshold를 넘은 것이 아니라, 요청이 slot을 얻기 전에 오래 기다린 TTFT failure다.

## 64 active admission 경계

smoke runtime은 다음 순서로 effective admission을 정한다.

```text
requested max in-flight = min(80, stable slots 80) = 80
decode batch capacity   = 64
aligned admission       = 80 - (80 mod 64) = 64
```

실측 결과:

- 최대 active request: 64
- active=64 누적 시간: 3906.1 ms
- active=64에서 submit된 request: 162
- server admission wait > 1 ms인 request: 162
- 두 집합은 정확히 동일
- 해당 요청의 admission wait: 평균 298.015 ms, 최대 431.500 ms

즉 이번 boundary에서는 memory broker pressure나 page-pool exhaustion이 아니라 decode-aligned active cap이 backpressure를 만든다. stable slot 80개 중 최대 16개는 admission에 사용되지 않는다.

## 80-slot 강제 A/B와 종료 원인

`TRT_EDGELLM_DISABLE_DECODE_ALIGNED_ADMISSION=1`로 같은 trace를 실행했다. 이 실험은 성공적인 성능 A/B가 아니라 capacity-contract 진단이다.

실행은 active request가 65개가 된 첫 D64 이후 다음 예외로 종료됐다.

```text
terminate called after throwing an instance of 'std::runtime_error'
what(): Eligible decode work disappeared during batch selection
```

원인은 `PhaseQueueScheduler::selectBatch()`의 두 제한이 서로 다른 집합을 사용하기 때문이다.

1. `count`는 queue 전체의 eligible decode row 수와 max D batch의 최솟값으로 계산된다.
2. persistent decode cohort는 기존 ID를 유지하며 크기가 64가 되면 새 ID를 더 넣지 않는다.
3. active=65에서 `count`는 64가 될 수 있지만 cohort의 일부 ID가 이미 in-flight/비eligible이면 현재 eligible cohort member는 63개뿐일 수 있다.
4. loop는 64개를 요구하고 64번째 eligible cohort member를 찾지 못해 invariant 예외를 낸다.

따라서 admission 정렬은 불필요한 제한이 아니라 이 selection contract를 감추는 안전장치다. 다음 수정은 `count`를 eligible cohort intersection으로 계산하고, persistent cohort에서 빠진/완료된 row를 먼저 정리하며, capacity 밖 ready row는 다음 iteration까지 안정적으로 유지하는 방식이어야 한다.

## 기각하거나 좁힌 가설

1. **Global selector CPU가 포화 원인이다:** 기각. D p95 0.272 ms이며 수백 ms TTFT와 규모가 다르다.
2. **GPU가 자주 빈다:** 기각. P/D active 95.11%, all-idle 4.89%다.
3. **admission 후 P batching이 느리다:** 기각. P initial queue p95 11.53 ms다.
4. **slot release 호출 자체가 느리다:** 기각. final commit 이후 release p95 약 1.4 us다.
5. **sampling은 영향이 없다:** 좁힘. 직접 TTFT cliff는 아니지만 ticket당 event observation residual이 D 평균 0.859 ms라 slot lifetime을 늘릴 수 있다.
6. **80 stable slots을 즉시 모두 admission하면 된다:** 기각. 현재 D64 persistent cohort selection이 active>64를 안전하게 처리하지 못한다.

## 다음 권장 단계

다음 단계는 capacity-safe D64 cohort formation 수정과 controlled admission sweep이다.

1. active request가 D max batch보다 큰 unit test를 추가한다: active 65/72/80, D64, 일부 cohort row in-flight/complete 상태.
2. selection count를 `eligible && cohort member`의 실제 교집합에 맞춘다.
3. cohort에서 완료/더 이상 ready하지 않은 ID를 안전하게 교체하고 stable row order를 유지한다.
4. 64/72/80 admission에서 exact token identity와 sanitizer/OOB gate를 통과한다.
5. 동일 48.8 trace를 각 capacity에서 5회 반복해 scheduled TTFT, joint-SLO goodput, raw throughput, slot occupancy를 비교한다.
6. 그 다음 sampling completion visibility를 direct ready-queue handoff로 줄여 추가 goodput을 측정한다.

이 수정 전에는 workload별 WAIT, overlap, formation score를 더 튜닝하지 않는다.

## 검증

- Python analyzer tests: 8/8 passed in `nvcr.io/nvidia/pytorch:25.12-py3`
- C++ targeted tests: 183/183 passed
- TRT/CUDA build: `llm_phase_context_smoke`와 `unitTest` incremental build passed
- successful detailed run: 288/288 complete, exact runtime token hash stable
- admission80 diagnostic: expected failure reproduced and exact invariant located

## 산출물

- 집계 CSV: `benchmarks/phase_serving/results/saturation-iteration-decomposition-20260830.csv`
- 상세 Current run: `.local/transition-aware-20260830/p6-iteration-decomposition/current-48.8-instrumented`
- request attribution: `.local/transition-aware-20260830/p6-iteration-decomposition/attribution`
- E/P/D/Copy activity: `.local/transition-aware-20260830/p6-iteration-decomposition/activity-analysis`
- admission80 failure: `.local/transition-aware-20260830/p6-iteration-decomposition/current-48.8-admission80-instrumented`

