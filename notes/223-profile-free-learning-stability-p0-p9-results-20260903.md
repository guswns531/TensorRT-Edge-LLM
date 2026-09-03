# Profile-free Online Learning Stability: P0–P9 구현 및 결과

날짜: 2026-09-03

브랜치: `codex/v010-phase-forward-port`

구현 전 기준 commit: `3908810270dee65ac331e517c010b02095ada765`

계획: `notes/222-profile-free-learning-stability-p0-p9-plan-20260903.md`

## 1. 결론

P0–P9의 구현, 실제 GPU 검증, 12-workload gate와 정책 인과 분해를 완료했다. 가장 중요한 결과는 다음과 같다.

1. graph/tactic priming, exact CUDA execution cost, contextual policy posterior를 독립적으로 reset할 수 있게 됐다.
2. `graph_only`, `zero_start`, workload-independent `generic`, `trace_derived` 네 warm-up 계약을 구현했다.
3. 48.8 req/s text saturation에서 generic full controller는 `41.779 req/s`로 frozen vLLM의
   `40.908 req/s`보다 `2.13%` 빠르며 TTFT, TPOT, E2E mean/p95도 모두 낮다.
4. 이 우위의 주된 원인은 P+D contextual authority다. 같은 generic calibration에서 all-shadow는
   `39.188 req/s`, P+D-only는 `41.579 req/s`, full-active는 `41.779 req/s`다.
5. 반대로 vision-heavy에서는 full E/P/D active가 `655.03 tok/s`이고 P+D-only가 `692.63 tok/s`,
   all-shadow가 `694.98 tok/s`다. 현재 E+P/E+D head에 production authority를 주면 안 된다.
6. generic P+D는 text saturation 3회 중 2회에서 strict final stability를 만족했다. 관측된 보수적 상한은
   288번째 request, 약 7.4초다. 한 run은 마지막 window에서만 조건을 만족해 연속 두 window gate를 통과하지 못했다.
7. zero-start는 819 decisions 동안 P+D reward observation이 0이었다. 강제 exploration을 금지한 현재 안전 계약에서는
   serial fallback만으로 overlap counterfactual label을 얻을 수 없는 safe-learning deadlock이 존재한다.
8. 4-cycle generic vision calibration 뒤 실제 vision-heavy 측정 구간에서도 E+P/E+D 신규 reward label은 0이었다.
   E pair의 안정화와 자연 발생 일반화는 아직 증명되지 않았다.
9. P+D-only를 한 공통 configuration으로 적용한 12개 gate는 frozen vLLM 대비 token throughput 10/12 우세다.
   wave/drain은 `-2.08%`, multi-image는 `-5.36%`다. 따라서 “안정화 뒤 모든 workload에서 vLLM 우세”라는 최종 목표는
   아직 통과하지 않았다.

현재 production 승격 판단은 다음과 같다.

```text
P+D contextual head       active
E+P / E+D contextual head shadow
workload label            없음
persisted registry        없음
fixed P chunk             128
```

이는 workload별 fine-tuning이 아니다. 가족별 authority는 학습 증거와 causal gate를 통과한 정도가 다르기 때문에 분리한다.
E head도 같은 generic calibration과 online evidence로 promotion gate를 통과하면 같은 controller에서 active로 전환할 수 있다.

## 2. 최종 아키텍처

```text
                         request arrival
                               |
                               v
                    immutable ready snapshot
                 E-ready / P-ready / D-ready
                 slack / ownership / outstanding
                               |
                               v
                    hard feasibility filter
             dependency / TRT shape / memory / inflight
                               |
                               v
                      feasible phase actions
             E, P, D, E+D, E+P, P+D, bounded WAIT
                               |
              +----------------+----------------+
              |                                 |
              v                                 v
      exact execution-cost model       contextual action-value model
         CUDA event observations        RLS mean + uncertainty
              |                          + EMA residual
              +----------------+----------------+
                               |
                               v
                    SLO-safe action selector
                  legal first, safe second,
                  conservative value third
                               |
                               v
               exact leased-row materialization
             request id + stable KV slot id preserved
                               |
                               v
                 independent E/P/D TRT contexts
                    shared CUDA context/streams
                               |
                               v
                 CUDA completion and feedback
```

상태 소유권은 세 층으로 분리한다.

| State | Owner | Reset API | Measurement 시작 시 처리 |
|---|---|---|---|
| CUDA graph/TRT profile/allocator | execution coordinator | 별도 lifecycle | 유지 |
| exact action/decode CUDA cost | `PhaseRuntimeCostTracker` | `resetExecutionCostHistory()` | mode별 유지/초기화 |
| contextual RLS/completion/conformal | `PhaseRuntimeCostTracker` | `resetPolicyPosterior()` | mode별 유지/초기화 |
| queue age/cohort/telemetry | `PhaseQueueScheduler` | `resetSchedulingHistory()` | 항상 초기화 |

정확성 invariant는 policy와 분리한다.

```text
dispatch(P_r) -> required E_r completed
reclaim(x)    -> every GPU consumer of x completed
#inflight(context_phase) <= 1
selected candidate rows == dispatched request IDs and stable KV slots
```

## 3. P0–P9 구현 내용

### P0 — Baseline freeze

`benchmarks/phase_serving/manifests/profile_free_learning_p0_baseline.json`에 다음을 고정했다.

- repository/upstream commit
- GPU, driver, CUDA, TensorRT, compiler, container digest
- Cosmos model/engine/sidecar hash
- 12 workload와 48.8 trace의 실제 SHA-256
- P8/D64/E8, fixed P128, stable slot 80 계약
- generic text, 2-cycle VLM, 최종 4-cycle VLM calibration trace hash
- R8 Current와 frozen vLLM baseline

### P1 — State reset separation

변경 위치:

- `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h`
- `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp`
- `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}`

기존 `resetHistory(bool)`가 queue telemetry와 model state의 의미를 섞던 문제를 분리했다. 기존 API는 source compatibility
wrapper로 남겼다.

### P2 — Warm-up mode contract

`examples/llm/llm_phase_context_smoke.cpp`에 `TRT_EDGELLM_POLICY_WARMUP_MODE`를 추가했다.

| Mode | Graph/tactic priming | Policy seed | Measurement update |
|---|---:|---:|---:|
| graph_only | O | 없음, model disabled | X |
| zero_start | O | 없음 | O |
| generic | O | 공통 calibration trace | O |
| trace_derived | O | 측정 trace 반복 | O |

mode와 `measurement_epoch`는 calibration response와 `PHASE_METRIC`에 기록된다. calibration 종료 시 queue history를
초기화하고 CUDA activity timeline도 reset해 warm-up GPU 구간이 measurement activity에 섞이지 않게 했다.

### P3 — Workload-independent generic calibration

`benchmarks/phase_serving/build_generic_policy_calibration_trace.py`는 workload 이름이나 측정 trace를 보지 않고 다음 ready
shape를 만드는 고정 trace를 생성한다.

```text
P rows: 1 / 4 / 8, chunk 128
D rows: 8 / 32 / 64
E rows: 1 / 2 / 4
incumbent/newcomer 방향 교차
```

text trace는 2 cycles, VLM trace는 최종적으로 4 cycles를 사용했다. 2-cycle VLM은 일부 E direction에서 최소 4개
observation을 만들지 못했기 때문이다. 4-cycle은 576 calibration request이며 production workload마다 동일하다.

### P4 — Completion authority readiness

action-value readiness와 completion-time authority를 분리했다. ordered direction별로 다음을 모두 만족해야 contextual
completion estimate가 actual scheduling authority를 얻는다.

- ready calibration observation 수
- conformal calibration observation 수
- incumbent/newcomer held-out coverage
- target coverage 대비 허용 오차
- false-safe rate 상한

환경 변수:

```text
TRT_EDGELLM_COMPLETION_AUTHORITY_COVERAGE_TOLERANCE
TRT_EDGELLM_COMPLETION_AUTHORITY_MAX_FALSE_SAFE_RATE
```

준비되지 않은 estimate는 telemetry와 shadow learning에는 남지만 deterministic robust estimate를 대체하지 않는다.

### P5 — Adaptation curve

`benchmarks/phase_serving/analyze_contextual_adaptation.py`가 최신 measurement epoch만 선택하고 warm-up baseline을
차감해 다음 window를 생성한다.

```text
1-16 / 17-32 / 33-64 / 65-128 / 129-256 / 257+
```

각 family에서 observation, cumulative observation, MAE, RMSE, mean, uncertainty, LCB, predicted-safe,
false-safe를 보고한다. stability는 observation/RMSE/false-safe 조건을 연속 두 window에서 만족해야 하며 decision,
elapsed time, request frontier로 기록한다. 한때 안정됐다가 마지막 window에서 깨진 상태는 stable로 보고하지 않는다.

### P6 — Representative warm-up A/B와 action fidelity 수정

balanced, decode-heavy, vision-heavy, multi-image, late-vision, 48.8에서 네 mode를 실행했다. 이 과정에서 Global
selector가 lease한 row와 실제 `popBatch()`가 재형성한 row가 달라질 수 있는 mechanism bug를 발견했다.

원인:

```text
global candidate selects rows
        |
new ready/higher-priority work arrives
        |
legacy popBatch reforms rows
        |
planned action != actual dispatch
```

수정 후 `popGlobalCandidateBatch()`는 candidate의 request ID와 stable KV slot ID를 정확히 소비한다. request가
사라지거나 eligibility가 바뀌면 즉시 실패한다. multi-image와 wave/drain도 수정 후 3회 token hash가 동일했다.

### P7 — Vision critical-path and activity decomposition

full telemetry는 성능 headline이 아니라 원인 분석용이다. generic 4-cycle calibration이 끝난 뒤 activity epoch를 새로
시작한 vision-heavy 한 run의 clean measurement window는 `4018.0 ms`다.

| E/P/D working-state mask | Time ratio |
|---|---:|
| idle | 3.1618% |
| E only | 30.9771% |
| P only | 30.2262% |
| D only | 32.5863% |
| P+D | 3.0482% |
| E+D | 0.0004% |
| E+P / E+P+D / Copy-active | 0% |

```text
E duty        30.9775%
P duty        33.2744%
D duty        35.6349%
any E/P/D     96.8382%
```

따라서 이 trace의 문제는 GPU idle이 아니다. heavy E/P/D 단독 service가 거의 전체 GPU 시간을 차지하며 profitable
overlap 기회가 작다.

64 request의 request-path attribution은 다음과 같다.

| Vision request boundary | Mean / p95 ms |
|---|---:|
| E queue | 1555.15 / 3283.15 |
| E active | 140.97 / 350.84 |
| E result -> P-ready | 1.59 / 3.20 |
| initial P queue | 89.33 / 377.08 |
| P active | 29.61 / 37.67 |
| first token -> D | 58.42 / 213.86 |
| D ready queue | 600.82 / 966.20 |
| D active total | 288.60 / 315.89 |
| inter-D gap | 752.08 / 1218.57 |

host scheduler decision은 mean `45 us`, p95 `120 us`; action realization residual은 mean `167 us`, p95 `881 us`다.
vision TTFT의 주원인은 E->P handoff나 selector 계산이 아니라 E queue service order다. 동시에 resident text request의
TPOT은 vision pressure 아래 D continuity에서 손실된다.

### P8 — One-configuration 12-workload gate

최종 gate는 generic calibration 하나와 P+D active/E heads shadow를 사용한다. 모든 run은 fresh process 3회 중앙값이며
요구 output token을 완성했다.

| Workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak MiB | exact x3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 2516.75 | 84.79 | 167.59 | 13.56 | 26.71 | 328.41 | 406.77 | 9237 | O |
| balanced | 4526.73 | 64.90 | 168.83 | 12.14 | 13.56 | 1098.24 | 1691.08 | 9237 | O |
| decode-heavy | 5290.68 | 61.55 | 181.20 | 10.53 | 11.17 | 2780.46 | 4245.43 | 9237 | O |
| long-prefill | 1227.31 | 2036.44 | 2630.54 | 25.72 | 30.28 | 4222.10 | 5994.57 | 9237 | O |
| bimodal | 1989.98 | 1832.69 | 4201.24 | 17.51 | 27.98 | 4200.67 | 8892.29 | 9237 | O |
| text-heavy | 1937.12 | 332.32 | 1091.63 | 25.23 | 39.42 | 1633.57 | 1707.88 | 9351 | O |
| mixed | 1128.88 | 720.63 | 2094.15 | 38.42 | 61.84 | 2454.35 | 2571.75 | 9437 | O |
| vision-heavy | 692.63 | 1535.77 | 3257.10 | 35.15 | 67.36 | 3003.54 | 3494.20 | 9413 | O |
| poisson | 1927.91 | 228.80 | 898.17 | 21.28 | 37.14 | 1618.37 | 2042.64 | 9351 | O |
| wave/drain | 93.85 | 437.82 | 597.23 | 9.52 | 13.53 | 730.00 | 797.33 | 9303 | O |
| multi-image | 231.42 | 346.02 | 487.36 | 9.83 | 15.66 | 644.31 | 690.86 | 9303 | O |
| late-vision | 2545.48 | 122.73 | 442.45 | 9.24 | 9.29 | 1447.44 | 1811.41 | 9311 | O |

frozen vLLM 대비 token throughput은 다음과 같다. model, trace, HTTP arrival와 output contract가 바뀌지 않았으므로
중복 실행하지 않고 검증된 3-run anchor를 재사용했다.

| Workload | Current tok/s | vLLM tok/s | Current change |
|---|---:|---:|---:|
| short | 2516.75 | 1983.53 | +26.88% |
| balanced | 4526.73 | 4319.89 | +4.79% |
| decode-heavy | 5290.68 | 4854.34 | +8.99% |
| long-prefill | 1227.31 | 1120.88 | +9.49% |
| bimodal | 1989.98 | 1868.29 | +6.51% |
| text-heavy | 1937.12 | 1634.76 | +18.50% |
| mixed | 1128.88 | 921.48 | +22.51% |
| vision-heavy | 692.63 | 579.20 | +19.58% |
| poisson | 1927.91 | 1800.07 | +7.10% |
| wave/drain | 93.85 | 95.85 | -2.08% |
| multi-image | 231.42 | 244.52 | -5.36% |
| late-vision | 2545.48 | 2359.23 | +7.89% |

latency는 Current가 낮을 때 양수로 표시했다.

| Workload | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| short | +51.5% | +36.5% | -1.5% | -7.4% | +23.0% | +19.2% |
| balanced | +42.6% | +39.8% | -0.1% | -0.3% | +4.5% | +4.2% |
| decode-heavy | +47.8% | +42.5% | +3.9% | +3.4% | +6.0% | +5.0% |
| long-prefill | -6.6% | +8.8% | +19.9% | +18.3% | +9.0% | +9.0% |
| bimodal | -17.0% | -59.0% | +24.2% | +24.5% | +11.1% | +5.0% |
| text-heavy | +21.2% | +11.4% | +13.6% | +16.8% | +15.9% | +16.2% |
| mixed | +17.6% | +17.6% | +18.2% | +26.4% | +18.4% | +18.1% |
| vision-heavy | +10.2% | +11.8% | +44.8% | +43.7% | +27.1% | +17.4% |
| poisson | +47.8% | +0.5% | +4.1% | +18.7% | +10.1% | +9.9% |
| wave/drain | -73.2% | -42.7% | +23.4% | +21.6% | -14.4% | -22.8% |
| multi-image | -33.2% | -21.1% | +20.9% | +4.0% | -0.0% | -5.7% |
| late-vision | +20.0% | +29.9% | +6.7% | +6.4% | +8.2% | +7.3% |

Current가 vLLM보다 낮은 지표 수는 TTFT mean 8/12, TTFT p95 9/12, TPOT mean/p95 10/12,
E2E mean/p95 10/12다. 가장 명확한 미통과는 sparse vision wave와 multi-image다.

### P9 — Causal ablation

#### 48.8 req/s text saturation

| Policy state/authority | req/s | tok/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|---:|
| graph-only, no policy update | 39.448 | 3418.81 | 24.18 / 37.40 | 16.90 / 23.86 | 1450.01 / 2310.72 |
| zero-start online | 39.431 | 3417.36 | 24.55 / 37.50 | 16.86 / 24.20 | 1449.37 / 2322.65 |
| generic, all-shadow | 39.188 | 3396.28 | 27.75 / 43.93 | 17.06 / 24.13 | 1467.59 / 2351.24 |
| generic, P+D active | 41.579 | 3603.50 | 37.36 / 62.41 | 13.43 / 16.88 | 1179.94 / 1920.02 |
| generic, full E/P/D active | **41.779** | **3620.82** | 36.35 / 62.73 | **12.91 / 16.40** | **1133.88 / 1884.31** |
| trace-derived upper reference | 41.807 | 3623.24 | 36.08 / 61.57 | 12.96 / 15.89 | 1137.86 / 1866.50 |
| frozen vLLM | 40.908 | 3545.40 | 51.44 / 84.27 | 13.53 / 17.66 | 1205.87 / 2121.30 |

generic과 trace-derived 처리량 차이는 `0.07%`다. P+D-only는 all-shadow보다 `6.10%`, full-active는
all-shadow보다 `6.61%` 빠르다. E request가 없는 trace이므로 P+D가 인과적으로 성능을 만든다.

#### Vision-heavy

| Policy state/authority | tok/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | exact x3 |
|---|---:|---:|---:|---:|---:|
| graph-only | 615.07 | 1483.09 / 3556.05 | 39.63 / 51.13 | 3025.58 / 3922.38 | O |
| generic, full-active | 655.03 | 1482.29 / 3343.38 | 36.87 / 49.85 | 2924.70 / 3672.68 | O |
| generic, P+D-only | 692.63 | 1535.77 / 3257.10 | 35.15 / 67.36 | 3003.54 / 3494.20 | O |
| generic, all-shadow | **694.98** | **1299.90 / 3127.47** | 36.52 / 53.90 | **2741.25 / 3468.94** | O |
| trace-derived | 686.50 | 1461.80 / 3287.34 | **33.64 / 53.41** | 2811.50 / 3522.92 | X |
| frozen vLLM | 579.20 | 1710.70 / 3691.37 | 63.70 / 119.58 | 4119.14 / 4229.37 | O |

full-active는 P+D-only보다 `5.43%`, all-shadow보다 `5.75%` 느리다. generic calibration 뒤 measurement에서
E+P/E+D action-value 신규 observation이 0인 상태로 active authority를 허용한 것이 핵심 위험이다. trace-derived는
처리량 상한처럼 보이지만 3회 중 한 token hash가 달라 production promotion 대상이 아니다.

clean v0.10은 production async request adapter, independent E/P/D contexts, stable indexed ownership과 같은 HTTP
execution contract를 제공하지 않는다. 따라서 같은 policy-only 숫자로 표에 넣으면 서로 다른 mechanism을 비교하게 된다.
P9에서는 clean v0.10을 구조 baseline으로 유지하고, 동일 HTTP 계약의 numerical causal baseline은 graph-only/all-shadow로
정의했다.

## 4. 학습 안정화 결과

### 4.1 P+D at 48.8 req/s

generic calibration baseline은 run마다 181~182 P+D observations다. measurement에서 추가된 observation은
20, 26, 27개다.

| Run | Decisions | Final observations | Final-window RMSE | False-safe | Strict final stable | stability upper bound |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 788 | 202 | 0.082 | 0 | X | 측정 종료 뒤 한 window 더 필요 |
| 2 | 786 | 208 | 0.070 | 0 | O | request 287 / 7.341 s |
| 3 | 789 | 208 | 0.070 | 0 | O | request 287 / 7.407 s |

run 1도 최종 window 자체는 accurate/safe했지만 직전 window에서 measurement observation 누계가 3개라 연속 두
qualified window를 만들지 못했다. 따라서 3/3 안정화라고 과장하지 않는다.

### 4.2 Zero-start

zero-start는 819 decisions, 288 requests 동안 P+D/EP/ED observation이 모두 0이다. 현재 production 설정은
`TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER=0.0`으로 강제 exploration을 금지한다. 선택하지 않은 overlap의
reward는 관측할 수 없으므로 serial-only 안전 fallback이 스스로 overlap posterior를 학습하지 못한다.

```text
no prior -> high uncertainty -> choose serial
                         ^          |
                         |          v
                    no overlap label
```

해결은 arbitrary exploration이나 workload rule이 아니다. 별도의 bounded generic calibration, 또는 이미 outstanding인
CUDA completion과 충분한 request slack을 이용한 명시적 safe probe budget이 필요하다.

### 4.3 E pair

4-cycle generic VLM은 startup에서 E+P/E+D family에 5~7개 수준의 observation을 만들지만, 실제 vision-heavy 229
measurement decisions에서는 신규 E pair reward observation이 0이었다. 이는 다음을 의미한다.

- startup family sample 수만으로 natural-state coverage를 주장할 수 없다.
- completion conformal readiness와 action-value production authority는 별도 promotion gate가 필요하다.
- E active 결과가 all-shadow보다 나빴으므로 현재 E authority는 shadow가 맞다.

## 5. 구현 및 검증 파일

| 역할 | 위치 |
|---|---|
| reset/authority API | `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h` |
| exact cost와 posterior reset, conformal gate | `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` |
| exact candidate row materialization | `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}` |
| coordinator-side authority use | `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` |
| warm-up mode/epoch/activity reset | `examples/llm/llm_phase_context_smoke.cpp` |
| generic calibration trace generator | `benchmarks/phase_serving/build_generic_policy_calibration_trace.py` |
| warm-up/policy matrix runner | `benchmarks/phase_serving/run_policy_warmup_matrix.py` |
| adaptation analyzer | `benchmarks/phase_serving/analyze_contextual_adaptation.py` |
| P0 manifest | `benchmarks/phase_serving/manifests/profile_free_learning_p0_baseline.json` |
| Python tests | `tests/python-unittests/test_{generic_policy_calibration_trace,policy_warmup_matrix,contextual_adaptation_analysis}.py` |
| C++ tests | `unittests/phaseQueueSchedulerTest.cpp`, `unittests/phaseRuntimeCostTrackerTest.cpp` |

주요 artifact:

```text
.local/p9-load48-warmup-modes-20260903
.local/p9-load48-warmup-modes-r2-20260903
.local/p9-load48-learning-curves-full-20260903
.local/p9-full12-generic-vision4-20260903
.local/p9-full12-generic-pd-only-20260903
.local/p9-vision-heavy-warmup-ablation-20260903
.local/p9-vision-heavy-generic-pd-only-20260903
.local/p9-vision-heavy-generic-full-shadow-20260903
.local/p9-p7-vision4-full-epochfix-20260903
.local/p9-p7-vision4-epochfix-analysis-20260903
```

## 6. Promotion gate 판정

| Gate | Result | Evidence |
|---|---|---|
| state reset semantics | PASS | independent reset API/unit tests |
| warm-up mode provenance | PASS | mode + epoch emitted |
| action fidelity | PASS | exact request/slot lease consumption |
| output completion | PASS | every 12-workload run completed requested tokens |
| 12-workload deterministic output | PASS | selected P+D-only configuration 12/12 exact x3 |
| 48.8 generic vs trace-derived within 3% | PASS | throughput 0.07%; latency close |
| 48.8 generic vs vLLM | PASS | req/s +2.13%, all latency mean/p95 lower |
| generic P+D stable in every run | **FAIL** | strict final stability 2/3 |
| zero-start learns safely | **FAIL** | 0 labels in 819 decisions |
| generic E heads stable/generalized | **FAIL** | measurement E labels 0, active regression |
| 12-workload throughput vs vLLM | **FAIL** | 10/12; wave and multi-image lower |
| all latency axes vs vLLM | **FAIL** | bimodal, wave, multi-image and a few TTFT/TPOT cases |
| 512 MiB headroom | PASS | maximum measured 9437 MiB on 10240 MiB GPU |

## 7. 다음 계획

우선순위는 policy condition을 더 추가하는 것이 아니라 label acquisition과 authority promotion을 고치는 것이다.

### N1 — Family-specific promotion state machine

```text
disabled -> shadow-learning -> validated -> active
```

각 family/direction은 held-out error, conformal coverage, false-safe, action disagreement, natural-state observation을 모두
통과해야 active가 된다. startup observation count 하나만으로 E authority를 주지 않는다. 이 state machine은 workload
이름을 보지 않는다.

### N2 — Bounded safe label acquisition

- generic calibration의 고정 request budget을 줄이면서 direction coverage를 높인다.
- natural traffic에서는 arbitrary probe를 하지 않는다.
- 이미 outstanding인 completion event, 큰 positive slack, low memory pressure가 동시에 있는 경우에만 제한된 probe를 허용한다.
- probe 손실 budget과 cumulative regret를 명시적으로 제한한다.

### N3 — Learning curve를 성능 curve와 직접 연결

현재 analyzer에 window별 throughput/goodput, selected/shadow disagreement, predicted/actual action regret를 붙인다.

```text
x: elapsed seconds or completed requests
y1: held-out RMSE / uncertainty / false-safe
y2: SLO goodput
y3: vLLM-normalized performance
```

최종 목표는 단지 posterior가 수렴하는 것이 아니라 `stable=true` 이후 vLLM-normalized goodput이 1보다 크고 다시
떨어지지 않는다는 것을 5 fresh processes에서 보이는 것이다.

### N4 — Sparse vision service-order correction

wave/multi-image의 문제는 큰 E overlap 기회가 아니라 sparse arrival에서 E queue/formation wait가 TTFT를 지배하는 것이다.
workload rule 없이 oldest first-token critical-path slack과 observable next cohort gain만 사용해 bounded E wait를 결정한다.

### N5 — Re-run promotion sequence

1. controlled E1/E2/E4/E8 pair test
2. vision-heavy and multi-image family promotion A/B
3. 48.8 text 5 fresh processes
4. full 12-workload 3 fresh processes
5. only then frozen/fresh vLLM headline comparison

## 8. 최종 판단

이번 단계는 profile-free 방향을 더 명확하게 만들었다. P+D에서는 generic calibration이 trace-derived 성능과 사실상 같고
vLLM도 넘었다. 그러나 “온라인 학습을 켰다”는 사실만으로 안전한 production authority가 되지는 않는다. E family는
자연 traffic label이 없고 full-active가 실제로 회귀했다.

따라서 현재 증거가 지지하는 최종 구조는 deterministic feasibility mechanism + process-local generic P+D learner +
shadow E learners다. 다음 성공 조건은 E를 무조건 켜는 것이 아니라 family-specific evidence로 자동 promotion시키고,
그 안정화 이후 12 workload, 특히 wave/multi-image에서 vLLM을 넘는 것이다.
