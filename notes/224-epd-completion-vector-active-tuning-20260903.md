# E/P/D Completion-Vector Active 튜닝 및 승격 결과

날짜: 2026-09-03

브랜치: `codex/v010-phase-forward-port`

기준 문서: `notes/223-profile-free-learning-stability-p0-p9-results-20260903.md`

## 1. 결론

E/P/D completion-vector를 단순히 `active`로 켜는 대신, 실제 측정 분포에서 scalar prior보다 정확하고 conformal
coverage가 안전한 방향과 completion component만 점진적으로 scheduling authority를 얻도록 구현했다. 최종 정책은
다음과 같다.

```text
generic warm-up
    |
    +-- exact CUDA cost와 RLS/conformal posterior는 유지
    |
measurement epoch begin
    |
    +-- authority evidence만 reset
    |
    +-- 처음 8개의 held-out directional completion은 shadow validation
    |
    +-- coverage와 false-safe gate를 통과하면 promotion
    |
    +-- maturity × empirical error improvement만큼 연속적으로 blend
    |
    +-- recent window가 demotion tolerance를 벗어나면 scalar prior로 복귀
```

가장 신뢰할 수 있는 개선은 48.8 req/s text saturation 3회 반복이다. 최종 E/P/D completion-vector active는
`41.698 req/s`로 같은 binary의 scalar controller보다 `+0.47%`, frozen vLLM보다 `+1.93%` 빠르다. scalar 대비
TTFT mean/p95, TPOT mean/p95, E2E mean/p95도 모두 개선했다. vision-heavy 3회에서는 `715.42 tok/s`로 기존
P+D-only Current보다 `+3.29%`, frozen vLLM보다 `+23.52%` 빠르다.

12-workload broad gate에서는 frozen vLLM 대비 token throughput 10/12 우세다. wave/drain은 `-2.06%`, 5-request
multi-image는 `-17.91%`다. 두 trace의 측정 구간에는 completion-vector overlap action이 없었으므로 이를 고치기 위해
E/P/D model에 workload-specific rule을 추가하지 않는다. 특히 multi-image는 같은 contract에서도 약 `200`과
`232 tok/s` 두 실행 군집이 반복되는 encoder arrival/formation variance가 지배한다.

## 2. 기존 active 방식의 문제

이전 completion-vector active는 calibration이 준비되면 vector prediction이 즉시 scalar policy를 대체했다.

```text
ready == false                   ready == true
scalar completion 100%    ->    vector completion 100%
```

이 방식에는 네 문제가 있었다.

1. pair-family conformal calibration이 ordered direction의 transfer accuracy를 보장하지 않는다.
2. incumbent가 정확하고 newcomer가 부정확한 경우에도 두 component가 동시에 authority를 얻는다.
3. 최소 sample에서 authority가 0에서 1로 불연속적으로 바뀐다.
4. generic warm-up에서 검증된 distribution이 실제 measurement distribution과 같다고 가정한다.

## 3. 최종 아키텍처

### 3.1 execution model과 policy authority 분리

```text
CUDA event observations
        |
        +--------------------------+
        |                          |
        v                          v
exact/scalar action value     two-output completion RLS
efficiency horizon            incumbent/newcomer finish time
        |                          |
        |                    pair conformal interval
        |                          |
        +------------+-------------+
                     v
        held-out directional authority window
        coverage / false-safe / empirical error
                     |
                     v
            component-wise gradual blend
                     |
                     v
             SLO-safe candidate ranking
```

scalar head는 없애지 않는다. 동일한 실행 관측을 사용하더라도 scalar action-value는 service compression을 안정적으로
표현하고, completion vector는 protected request의 구체적인 완료 경계를 표현한다. 따라서 vector가 recent held-out
window에서 더 정확하다는 증거가 있는 만큼만 scalar prior를 수정한다.

### 3.2 bounded authority evidence

ordered direction별 최근 window에 다음을 저장한다.

- incumbent/newcomer interval coverage
- SLO-safe prediction 수와 false-safe 수
- completion-vector makespan absolute error
- scalar decision makespan absolute error
- incumbent/newcomer 각각의 vector/reference absolute error
- promotion/demotion 횟수와 현재 validation 상태

승격 조건은 다음과 같다.

```text
observations >= authorityMinimumObservations
incumbent coverage >= targetCoverage - promotionTolerance
newcomer coverage >= targetCoverage - promotionTolerance
falseSafeRate <= configured maximum
```

승격 뒤에는 더 넓은 `authorityDemotionCoverageTolerance`를 사용한다. 따라서 한 sample이 sliding window에서 빠지는
순간 policy가 promotion/demotion을 반복하는 것을 막는다.

### 3.3 empirical-risk 및 maturity blend

전체 makespan과 각 component의 weight를 독립적으로 계산한다.

```text
relativeImprovement
  = clamp((referenceError - completionError) / referenceError, 0, 1)

maturity
  = observations > minimum
  ? (observations - minimum) / observations
  : 0

effectiveWeight
  = configuredMaximumWeight * maturity * relativeImprovement
```

의미는 다음과 같다.

- vector가 scalar/reference보다 나쁘면 weight는 0이다.
- 최소 8개를 막 넘긴 순간에는 validation이 성공해도 weight는 0에서 시작한다.
- held-out evidence가 쌓일수록 authority가 연속적으로 증가한다.
- incumbent와 newcomer 중 하나만 개선되면 그 component만 바뀐다.

### 3.4 formation-aware uncertainty

completion-vector efficiency horizon은 두 component의 단순 최대 평균이 아니다.

```text
formationRisk = normalized producer batch fill
robustComponent = mean + formationRisk * uncertainty
vectorHorizon = max(robustIncumbent, robustNewcomer)
```

작은 producer batch에서는 관측된 mean overlap gain을 활용하고, 큰 E/P batch에서는 다음 cohort의 formation value를
보호하도록 uncertainty penalty가 연속적으로 커진다. workload 이름이나 `E1`, `P8` 같은 exact shape rule은 없다.

### 3.5 measurement-local validation epoch

generic warm-up 종료 시 다음을 구분한다.

| State | measurement 시작 시 |
|---|---|
| graph/TRT profile priming | 유지 |
| exact CUDA cost | 유지 |
| RLS completion posterior | 유지 |
| conformal scale | 유지 |
| completion authority evidence | **reset** |
| queue age/telemetry | reset |

즉 warm-up은 물리적으로 의미 있는 prior와 uncertainty scale을 제공하지만, 실제 serving distribution에서 최소 8개의
held-out completion이 다시 확인되기 전에는 scheduling decision을 바꾸지 못한다.

## 4. 구현 위치

| File | 역할 |
|---|---|
| `cpp/runtime/phase/policy/phaseContextualPdModel.h` | authority config/evidence와 completion blend API |
| `cpp/runtime/scheduling/phaseContextualPdModel.cpp` | formation-weighted horizon 및 component blend |
| `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h` | held-out authority API/window 선언 |
| `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` | promotion/demotion, empirical-risk, maturity, reset 구현 |
| `cpp/runtime/scheduling/phaseQueueScheduler.cpp` | P+D candidate와 protected completion에 vector authority 적용 |
| `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | E+P/E+D initial/residual action에 동일 authority 적용 |
| `examples/llm/llm_phase_context_smoke.cpp` | 환경 계약, epoch reset, JSON telemetry |
| `unittests/phaseRuntimeCostTrackerTest.cpp` | vector horizon, blend, held-out promotion/reset 검증 |

E observation join은 scheduler loop의 마지막 dispatch를 추측하지 않고 stable `planId`로 완료 observation을 연결한다.
이 수정이 없으면 다른 P/D dispatch가 끼어들 때 잘못된 E completion label이 RLS에 들어갈 수 있다.

## 5. 튜닝 과정

| 단계 | 변경 | 48.8 req/s 결과 | 판단 |
|---|---|---:|---|
| scalar control | completion authority 0 | `41.502 req/s` | 비교 기준 |
| per-component direct | component별 error authority | `41.658 req/s` | TPOT/E2E 개선, TTFT 회귀 |
| fixed 0.5 | 모든 validated vector 최대 50% | 약 `41.51 req/s` | 고정 weight 기각 |
| maturity | sample 수에 따라 연속 증가 | `41.674 req/s` | 전 latency 축 개선 |
| measurement-local reset | 실제 trace에서 held-out 재검증 | **`41.698 req/s`** | 최종 선택 |

`authorityMinimumObservations=16`도 시험했지만 한 run에서 `41.489 req/s`로 낮았다. 기본값 8을 유지한다. 고정 0.5나
16 같은 값을 workload별로 선택하지 않고, 한 공통 config에서 empirical error와 observation maturity가 실제 weight를
결정하게 했다.

## 6. 48.8 req/s saturation 반복 결과

### 6.1 최종 E/P/D active 대 scalar

| Metric | Scalar, 4-run median | Final vector, 3-run median | 변화 |
|---|---:|---:|---:|
| achieved req/s | 41.502 | **41.698** | **+0.47%** |
| token/s | 3596.85 | **3613.86** | **+0.47%** |
| TTFT mean | 37.693 ms | **37.020 ms** | **-1.79%** |
| TTFT p95 | 64.183 ms | **61.525 ms** | **-4.14%** |
| TPOT mean | 13.368 ms | **12.690 ms** | **-5.07%** |
| TPOT p95 | 16.745 ms | **15.561 ms** | **-7.07%** |
| E2E mean | 1175.716 ms | **1117.735 ms** | **-4.93%** |
| E2E p95 | 1905.170 ms | **1819.228 ms** | **-4.51%** |

최종 세 run은 `41.698 / 41.679 / 41.741 req/s`이고 output token hash가 모두 동일하다.

### 6.2 frozen vLLM과 비교

| Metric | frozen vLLM | Final vector | 변화 |
|---|---:|---:|---:|
| achieved req/s | 40.908 | **41.698** | **+1.93%** |
| TTFT mean | 51.44 ms | **37.02 ms** | **-28.0%** |
| TTFT p95 | 84.27 ms | **61.53 ms** | **-27.0%** |
| TPOT mean | 13.53 ms | **12.69 ms** | **-6.2%** |
| TPOT p95 | 17.66 ms | **15.56 ms** | **-11.9%** |
| E2E mean | 1205.87 ms | **1117.74 ms** | **-7.3%** |
| E2E p95 | 2121.30 ms | **1819.23 ms** | **-14.2%** |

## 7. 12-workload broad gate

단위는 generated token/s다. 48.8 saturation은 별도 반복 gate이고 이 표에는 포함하지 않는다. Final의
balanced/decode-heavy/short/long-prefill/bimodal/text-heavy/poisson/late-vision은 1회 broad gate,
mixed는 1회, vision-heavy/wave/multi-image는 3회 중앙값이다. Current P+D-only와 vLLM은 문서 223의 frozen 값을
재사용했다.

| Workload | Final E/P/D vector | Current P+D-only | 변화 | frozen vLLM | 변화 |
|---|---:|---:|---:|---:|---:|
| short | 2508.02 | 2516.75 | -0.35% | 1983.53 | +26.44% |
| balanced | 4476.65 | 4526.73 | -1.11% | 4319.89 | +3.63% |
| decode-heavy | 5361.97 | 5290.68 | +1.35% | 4854.34 | +10.46% |
| long-prefill | 1232.94 | 1227.31 | +0.46% | 1120.88 | +10.00% |
| bimodal | 1967.37 | 1989.98 | -1.14% | 1868.29 | +5.30% |
| text-heavy | 1955.32 | 1937.12 | +0.94% | 1634.76 | +19.61% |
| mixed | 1145.10 | 1128.88 | +1.44% | 921.48 | +24.27% |
| vision-heavy | 715.42 | 692.63 | +3.29% | 579.20 | +23.52% |
| poisson | 1988.90 | 1927.91 | +3.16% | 1800.07 | +10.49% |
| wave/drain | 93.88 | 93.85 | +0.03% | 95.85 | -2.06% |
| multi-image | 200.73 | 231.42 | -13.26% | 244.52 | -17.91% |
| late-vision | 2553.75 | 2545.48 | +0.33% | 2359.23 | +8.25% |

multi-image는 5 requests뿐이며 최종 세 run이 `200.73 / 231.93 / 200.06 tok/s`다. 기존 P+D-only 세 run도
`234.68 / 231.42 / 200.37 tok/s`로 동일한 이산적 변동을 보였다. 최종 측정 로그는
`contextual_predictions=0`, `overlaps=0`이다. 따라서 표의 `-13.26%`는 completion-vector가 선택한 잘못된 action의
인과 효과가 아니며, sparse asynchronous encoder arrival과 batch-formation variance를 더 긴 trace에서 별도로
평가해야 한다.

## 8. correctness 및 안정성

- 48.8 final 3/3 output token hash 동일
- vision-heavy final 3/3 output token hash 동일
- wave/drain final 3/3 output token hash 동일
- multi-image final 3/3 output token hash 동일
- E completion observation은 stable `planId`가 일치할 때만 join
- measurement epoch는 scheduler가 idle일 때만 authority evidence reset
- promotion 전 posterior/conformal state는 유지되지만 policy weight는 0
- invalid/unsafe component는 scalar/reference 쪽으로 자동 shrink

## 9. artifact

```text
.local/completion-active-tuning-20260903/scalar-load48-r4-v25
.local/completion-active-tuning-20260903/per-component-load48-r4-v24
.local/completion-active-tuning-20260903/epoch-reset-load48-r3-v35
.local/completion-active-tuning-20260903/epoch-reset-representative-v36
.local/completion-active-tuning-20260903/epoch-reset-vision-heavy-r3-v37
.local/completion-active-tuning-20260903/epoch-reset-sparse-vlm-r3-v38
.local/completion-active-tuning-20260903/final-12-gate-part1-v39
.local/completion-active-tuning-20260903/final-12-gate-part2-v40
.local/completion-active-tuning-20260903/final-12-gate-part3-v41
```

## 10. 승격 판단과 다음 작업

현재 최고 profile-free configuration은 다음이다.

```text
P+D contextual scalar/value head       active
E+P / E+D contextual scalar/value head active
E/P/D completion-vector authority      held-out validated + gradual
authority minimum observations         8
authority maximum blend                1.0
authority actual blend                 maturity × empirical improvement
authority validation scope             measurement-local direction
fixed P chunk                          128
workload label/exact shape policy       없음
external cost registry                 없음
```

이 configuration은 48.8 saturation과 vision-heavy에서 기존 Current보다 낫고, broad gate의 일반 workload를 대부분
유지한다. 다만 논문/production의 최종 12-workload promotion에는 다음이 남는다.

1. broad gate의 1회 항목을 3~5회 반복해 confidence interval을 고정한다.
2. sparse VLM은 최소 수십 request로 늘려 encoder arrival/formation variance와 policy effect를 분리한다.
3. wave/multi-image의 vLLM gap은 completion-vector가 아니라 encoder formation/ingress mechanism에서 해결한다.
4. authority promotion/change 횟수와 scalar 대비 decision regret을 natural workload별로 집계한다.
5. 두 번째 GPU/model에서 동일 feature/config가 다른 overlap profitability를 online으로 학습하는지 검증한다.

현재 구현에 workload별 fine-tuning을 더 넣는 것은 금지한다. 다음 성능 개선은 completion-vector weight를 임의로
조정하는 것이 아니라, sparse encoder workload의 observable pending preparation을 candidate formation에 포함하는 공통
mechanism과 충분히 긴 실제 request trace로 진행한다.
