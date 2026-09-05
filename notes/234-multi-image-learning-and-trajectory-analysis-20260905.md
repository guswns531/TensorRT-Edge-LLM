# Multi-image 학습 효과와 trajectory 파편화 P0--P6 결과

## 결론

이번 단계는 multi-image에서 관측한 `약 324 tok/s`와 `약 245 tok/s`의 큰 성능 차이를 단순한 RLS
학습 효과로 설명하지 않고, runtime lifecycle, exact CUDA cost, contextual RLS authority, E/P/D batch formation을
분리해 검증했다.

- 5-request multi-image는 같은 binary와 같은 generic calibration에서도 두 개의 trajectory로 갈린다.
  coherent trajectory는 D dispatch가 `32--33`회이고 약 `324 tok/s`, fragmented trajectory는 D dispatch가
  `54--60`회이고 약 `245 tok/s`다.
- zero-start 5회는 모두 coherent였다. 그러나 generic calibration 후 RLS posterior만 지운 `generic_reset`도
  `2/5`회 fragmented였다. 따라서 regression 전체를 RLS 탓으로 돌릴 수 없다. exact CUDA cost와 warm runtime
  history 자체가 action ordering을 바꿀 수 있다.
- full generic은 `4/5`회 fragmented였다. contextual RLS가 추가로 나쁜 선택을 만드는 경우가 있지만,
  contextual authority가 한 번도 적용되기 전에 이미 갈라지는 run도 있다.
- batch size는 원래부터 scalar/completion RLS feature에 포함되어 있었다. 이번에는 고정 `P8/D64` 정규화를
  실제 engine의 max P/D batch capability로 교체했다. workload별 batch rule은 추가하지 않았다.
- joint-fill/cost-balance interaction feature를 추가한 실험은 full12 geometric mean을 기존 champion 대비
  `-1.71%` 떨어뜨리고 multi-image `-10.30%`, text-heavy `-9.48%`를 만들었으므로 제거했다.
- opt-in successor guard는 같은 frontier에서 contextual 선택과 exact-cost fallback을 비교하도록 구현했지만,
  corrected 5회에서 실제 guard evaluation/override가 `0`이었다. 우연한 run-to-run trajectory 차이를 guard의
  효과로 주장하지 않고 production default를 바꾸지 않았다.
- 최종 16-feature 구현의 full12 1회 geometric mean은 이전 champion 대비 `-1.15%`, frozen vLLM 대비
  `+15.00%`이며 token throughput은 vLLM보다 `12/12` 높다. Poisson 3회 중앙값은 이전 champion 대비
  `+0.61%`로 회복됐다. text-heavy 3회 중앙값은 `-3.12%`로 3% gate를 `0.12%p` 초과했다.

이번 단계의 핵심 결론은 다음과 같다.

> Multi-image 성능의 큰 변동은 학습 전/후라는 한 축이 아니라 E completion과 P admission 순서가 첫 D1의
> criticality를 만들고, 그 D1을 보호하는 SLO policy가 이후 D cohort를 32회 또는 60회 trajectory로 증폭하는
> execution--formation coupling이다.

## 1. 질문별 직접 답변

### 1.1 왜 학습 전이 더 좋을 수 있는가

zero-start는 learned overlap을 쓰지 않는 단순한 상태가 아니다. 다음이 모두 cold다.

```text
zero-start
  exact CUDA execution-cost registry: empty
  contextual scalar posterior: reset
  completion/effect posterior: reset
  TensorRT/CUDA graph cache: warmup 계약에 따라 유지 또는 초기화
```

generic calibration은 반대로 여러 상태를 동시에 바꾼다.

```text
generic
  exact CUDA costs: populated
  contextual posterior: populated
  execution history: populated
  context/graph/runtime: warm
```

따라서 `zero-start > generic`이라는 결과만으로 RLS가 나쁘다고 결론 낼 수 없다. 이를 분리하기 위해
`generic_reset`을 추가했다.

```text
generic_reset
  generic calibration requests 실행
  exact CUDA costs와 warm runtime state 유지
  contextual/effect/completion policy posterior만 reset
```

5회 결과는 exact-cost lifecycle만으로도 coherent/fragmented trajectory가 모두 나옴을 보였다. RLS는 그 위에서
추가로 action을 바꾸지만 최초 원인은 아니다.

### 1.2 batch size가 RLS에 들어가는가

들어간다. scalar feature는 16차원이며 그중 primary/secondary batch fill이 각각 독립 feature다.

```text
feature[5] = log1p(primary batch size) / log1p(primary engine capacity)
feature[6] = log1p(secondary batch size) / log1p(secondary engine capacity)
```

P+D에는 P batch와 D batch가, E+P/E+D에는 E batch와 상대 phase batch가 들어간다. 또한 isolated phase cost,
cost ratio, context bucket, slack, residual progress, requested skew, outstanding phase도 함께 들어간다.

이번 수정 전 문제는 batch size가 없었던 것이 아니라 capacity 분모가 P8/D64로 고정되어 있었다는 점이다.
이제 다음 capability를 runtime config에서 전달한다.

- P capacity: `PhaseQueueSchedulerConfig::maxPrefillBatchSize`
- D capacity: `PhaseQueueSchedulerConfig::maxDecodeBatchSize`
- E capacity: `PhaseThreeCoordinatorConfig::maxEncoderBatchSize`

이는 batch target이나 workload heuristic이 아니라 engine shape capability의 연속 정규화다.

### 1.3 multi-image가 아주 잘 나온 run은 무엇이 다른가

최종 5회에서 coherent와 fragmented cluster가 분명히 갈렸다.

| Family | Runs | token/s 중앙 경향 | E dispatch | P dispatch | D dispatch | D GPU time |
|---|---:|---:|---:|---:|---:|---:|
| coherent | 2 | `324.36` | 2 | 4 | 32 | `213.7 ms` |
| fragmented | 3 | `244.85` | 2--3 | 4 | 54--60 | `354.8--391.3 ms` |

coherent run은 초기에 `E3+E2` 또는 `E2` 뒤 `E3+P1`처럼 producer progress가 모여 P/D transition이 형성됐다.
fragmented run은 `E1+E3+E1` 또는 `E4+E1`처럼 completion boundary가 달라졌고, P3가 대기하는 동안 D1이 먼저
ready가 됐다.

그 시점의 scheduler 판단은 workload 이름과 무관하게 합리적이었다.

```text
D1의 remaining TPOT slack  ~= 20 ms
P1의 predicted service      ~= 27 ms
P의 TTFT slack               ~= 305 ms
```

P를 먼저 실행하면 active D request의 TPOT budget을 넘길 수 있으므로 D1을 선택했다. 문제는 이 local SLO-safe
결정이 여러 번 반복되며 32-token decode가 작은 cohort로 분해된다는 점이다.

```text
E completion order
  -> P admission order
  -> first D1 ready boundary
  -> TPOT protection selects D1
  -> P remains queued
  -> subsequent D cohorts remain small
  -> D dispatch 32 -> 54--60
```

따라서 missing batch-size feature 하나로 해결할 문제가 아니다. completion-order와 future cohort formation을 함께
보는 transition problem이다.

## 2. 구현한 분리·관측 구조

### 2.1 runtime lifecycle A/B

`benchmarks/phase_serving/run_policy_warmup_matrix.py`에 다음을 추가했다.

- `generic_reset` mode
- 동일 matrix entry에 다른 trace를 주입하는 `--trace-override`
- opt-in `successor_guard` variant
- calibration client mode와 backend reset mode 분리

runtime의 `policy_reset`은 exact CUDA execution costs, graph cache, TensorRT context, allocator와 engine tactic을
유지하고 contextual/effect/completion posterior만 초기화한다. zero-start처럼 execution-cost history까지 지우지
않으므로 exact-cost와 RLS의 영향을 분리할 수 있다.

### 2.2 same-frontier non-contextual attribution

각 candidate에 `contextualScalarAuthorityApplied`를 기록한다. 동일 candidate vector를 복사한 뒤
`phaseRestoreNonContextualPolicy()`를 적용하면 다음은 그대로 남는다.

- DAG/shape/memory feasibility
- request membership와 row order
- stable slot/ownership state
- exact CUDA observation
- SLO slack

제거되는 것은 contextual scalar와 completion authority뿐이다. event에는 active H1과 non-contextual H1 action ID를
동시에 남긴다. 따라서 `learned action -> exact-cost fallback action` disagreement를 같은 frontier에서 측정할 수
있다.

### 2.3 trajectory analyzer

새 `benchmarks/phase_serving/analyze_multi_image_trajectory.py`는 HTTP aggregate, E/P/D/Copy CUDA-event interval,
scheduler event를 run별로 결합한다.

출력은 다음을 포함한다.

- token/s, TTFT/TPOT/E2E mean/p95
- E/P/D/Copy dispatch와 GPU time
- measurement epoch의 scheduler decision 수
- selected action의 scalar evidence 수
- active-vs-noncontextual disagreement와 action pair
- successor guard evaluation/override 수
- 5-request trace의 coherent/transitional/fragmented trajectory label

generic calibration과 measured trace가 같은 telemetry file에 있을 때 request-ID epoch boundary로 calibration
decision을 제외한다. 40/80-request scaled trace에는 5-request threshold를 잘못 적용하지 않고 `scaled`로 표시한다.

## 3. P0 -- exact cost와 RLS lifecycle 분해

동일 5-request trace를 각 mode에서 5회 실행했다.

| Mode | coherent | fragmented | token/s 범위 | 해석 |
|---|---:|---:|---:|---|
| zero-start | 5/5 | 0/5 | `318.89--319.84` | cold exact/RLS가 안정적으로 coherent |
| generic_reset | 3/5 | 2/5 | coherent `313--324`, fragmented `244` | exact cost/runtime history만으로도 분기 |
| generic | 1/5 | 4/5 | coherent `311`, fragmented `204--242` | RLS disagreement가 추가 악화 가능 |

generic run별 selected contextual evidence는 `0--3`회뿐이었다. fragmented run 중 하나는 contextual known action이
`0`회였으므로 RLS가 필요조건이 아니다.

## 4. P1--P3 -- successor guard 검증

### 4.1 첫 구현의 observer effect

처음에는 모든 multi-phase decision에서 H=2 successor evaluator를 실행했다. 5회 모두 fragmented였고 guard는
한 번도 override하지 않았다.

| Metric | 결과 |
|---|---:|
| guard evaluations | `63--85 / run` |
| guard overrides | `0` |
| median token/s | 약 `240.65` |
| median D dispatch | 약 `60` |

selector가 action을 바꾸지 않았는데 trajectory가 나빠졌으므로 expensive shadow planning의 CPU 시간과 decision
boundary perturbation 자체가 async runtime을 바꾼 것이다. 이는 `shadow == policy neutral`이라는 가정이 틀릴 수
있다는 중요한 negative result다.

### 4.2 corrected gated guard

비싼 transition evaluation은 contextual active action과 noncontextual fallback action이 실제로 다를 때만 실행하도록
제한했다.

```text
same frontier
  -> active H1
  -> remove contextual authority
  -> fallback H1
  -> identical action: no H2 work
  -> disagreement: bounded H2 comparison
```

corrected 5회는 4 coherent / 1 fragmented였지만 guard evaluation과 override는 전부 `0`이었다. 따라서 성능 회복은
guard의 인과적 효과가 아니라 run-to-run formation 차이다. 기능은 research opt-in으로 남기고 default는 끈다.

## 5. P4 -- batch feature 실험

### 5.1 유지한 변화

- 기존 16차원 feature 유지
- P/D/E batch fill 분모를 실제 runtime capability로 변경
- workload label, trace ID, exact batch lookup rule은 추가하지 않음

### 5.2 제거한 변화

joint batch fill과 `cost balance x joint fill` interaction 2개를 넣은 18-feature 실험은 다음 regression을 냈다.

| Metric/workload | 18-feature vs 이전 champion |
|---|---:|
| full12 geometric mean | `-1.71%` |
| multi-image | `-10.30%` |
| text-heavy | `-9.48%` |
| short | `-4.18%` |

동일 warmup sample budget에서 차원을 늘리면 posterior authority가 sparse evidence를 다른 방식으로 일반화하고,
async formation도 달라진다. 따라서 extra interaction은 제거했다. `feature richness` 자체를 최적화 목표로 두지
않는다.

## 6. P5 -- scaled multi-image의 학습 효과

### 6.1 40 requests

| Mode | token/s 중앙값 | 해석 |
|---|---:|---|
| zero-start | `418.24` | cold execution/policy |
| generic | `464.73` | `+11.1%`, generic evidence가 평균 throughput에 도움 |

generic 세 run의 D dispatch는 `35/76/105`회로 여전히 trajectory variance가 컸다. 높은 중앙 throughput이 일정한
cohort mechanism을 의미하지 않는다.

### 6.2 80 requests

| Mode | token/s 중앙값 | TTFT mean 중앙 경향 | TPOT mean 중앙 경향 | E2E mean 중앙 경향 |
|---|---:|---:|---:|---:|
| zero-start | `452.65` | 약 `1851 ms` | 약 `59.3 ms` | 약 `3688 ms` |
| generic | `427.02` | 약 `1713 ms` | 약 `43.0 ms` | 약 `3032 ms` |

generic은 raw token throughput 중앙값이 `-5.7%`지만 request latency는 더 좋았다. 즉 learned policy가 단순히
GPU token/s를 최대화한 것이 아니라 E/P first-token progress와 D continuity 사이의 trajectory를 바꾼다. 최종 평가는
raw throughput만이 아니라 joint-SLO goodput을 사용해야 한다.

## 7. P6 -- 최종 full12 및 frozen vLLM 비교

같은 engine, P128 chunk, max P8/D64/E4, admission/memory contract를 유지했다. vLLM trace/runtime contract가 바뀌지
않았으므로 기존 frozen fresh 측정값을 재사용했다.

| Workload | Final tok/s | 이전 champion 대비 | frozen vLLM 대비 | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 4480.8 | -1.51% | +3.90% | 65.6 / 170.1 | 12.26 / 13.59 | 1110.7 / 1716.8 |
| bimodal | 1896.6 | -1.65% | +3.07% | 1908.3 / 4122.9 | 18.11 / 27.36 | 4395.7 / 9191.9 |
| decode-heavy | 5198.9 | -2.11% | +4.81% | 64.0 / 178.8 | 10.72 / 11.50 | 2831.6 / 4337.2 |
| late-vision | 2543.6 | -0.13% | +7.93% | 119.2 / 425.8 | 9.28 / 9.34 | 1448.5 / 1813.2 |
| long-prefill | 1159.5 | -0.42% | +3.63% | 2158.6 / 2717.3 | 27.31 / 31.73 | 4481.9 / 6243.6 |
| mixed | 1160.7 | +2.81% | +55.24% | 711.3 / 2138.3 | 37.17 / 58.43 | 2393.8 / 2500.6 |
| multi-image | 315.5 | -1.92% | +32.79% | 208.9 / 297.8 | 9.39 / 12.78 | 500.1 / 506.9 |
| poisson | 1912.7 | -3.10% | +6.68% | 250.1 / 955.8 | 21.36 / 39.18 | 1629.1 / 2057.8 |
| short | 2441.4 | -1.93% | +22.84% | 90.5 / 178.0 | 13.55 / 27.13 | 336.4 / 417.4 |
| text-heavy | 2068.2 | -3.00% | +26.38% | 317.4 / 1020.6 | 23.19 / 34.45 | 1517.7 / 1615.1 |
| vision-heavy | 700.5 | -0.81% | +21.68% | 1319.4 / 3143.4 | 36.36 / 55.26 | 2774.5 / 3456.1 |
| wave-drain | 98.2 | +0.08% | +2.57% | 240.9 / 290.4 | 8.09 / 9.79 | 491.6 / 499.4 |

요약:

- 이전 champion 대비 token throughput geometric mean: `-1.15%`
- frozen vLLM 대비 token throughput geometric mean: `+15.00%`
- frozen vLLM token throughput 승리: `12/12`
- single-run 3% gate: `10/12` 엄격 통과, text-heavy와 Poisson이 경계

경계 두 workload를 3회 반복했다.

| Workload | 3회 token/s 중앙값 | 이전 champion 대비 | frozen vLLM 대비 | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| poisson | 1985.81 | +0.61% | +10.76% | 209.4 / 789.6 | 20.98 / 39.27 | 1548.6 / 1989.5 |
| text-heavy | 2065.60 | -3.12% | +26.22% | 319.2 / 1006.3 | 23.56 / 34.55 | 1524.9 / 1625.1 |

Poisson의 single-run 실패는 반복 변동으로 설명된다. text-heavy는 여전히 gate를 `0.12%p` 넘으므로 production
promotion 기준으로는 미해결 경계다. 또한 text-heavy 3회 중 한 번의 greedy token trace hash가 달라 exact
cross-run identity가 `false`였다. 이번 batch-capacity 변경 때문이라고 단정할 evidence는 없지만, exact promotion
gate를 위해 canonical row/binding ordering과 FP16 branch sensitivity를 별도로 확인해야 한다.

## 8. 최종 architecture 상태

```text
Ready E/P/D snapshot
        |
        v
Deterministic candidate mechanism
  dependency / TRT profile / single inflight
  stable KV+vision ownership / canonical membership
        |
        v
Exact CUDA cost registry --------------------+
        |                                     |
        v                                     v
16-D contextual scalar RLS              noncontextual H1 replay
  includes runtime-normalized E/P/D fill      |
        |                                     |
        +----------------+--------------------+
                         v
                 SLO-safe global selector
                         |
             optional disagreement-only H2 guard
                         |
                         v
                   E/P/D dispatch
                         |
                         v
           CUDA event + request trajectory log
```

production default는 V1 contextual scalar immediate selector다. successor guard와 richer transition selection은
opt-in/shadow다. 이번 결과는 workload별 fine tuning을 추가할 근거가 아니라, deterministic mechanism이 만들어내는
completion/ready ordering을 더 정확히 관측하고 제어해야 한다는 근거다.

## 9. 하지 않은 주장

- successor guard가 4/5 coherent를 만들었다고 주장하지 않는다. evaluation/override가 0이었다.
- generic calibration이 항상 throughput을 올린다고 주장하지 않는다. 40 requests에는 도움, 80 requests에는 raw
  throughput 손해와 latency 이득이 동시에 있었다.
- fragmented multi-image를 RLS만의 문제라고 주장하지 않는다. generic_reset에서도 발생했다.
- 18-feature model이 더 expressive하므로 낫다고 주장하지 않는다. 실제 full12에서 회귀했다.
- frozen vLLM 숫자는 이번에 재실행한 값이라고 표현하지 않는다. workload/runtime contract가 동일하여 재사용했다.

## 10. 다음 우선순위

### P7 -- E completion/ready boundary attribution

현재 telemetry에 다음 timestamp와 stable request IDs를 연결한다.

```text
E GPU complete
  -> completion visible
  -> vision lease published
  -> request enters P queue
  -> P candidate formed
  -> P selected/enqueued
  -> first D row ready
```

목표는 good/bad run의 최초 divergent boundary를 scheduler action이 아니라 request transition 수준에서 exact하게
찾는 것이다.

### P8 -- deterministic formation replay

동일 measured completion vector를 immutable snapshot에 주입해 다음을 재생한다.

- E completion order만 바꾼 branch
- P admission/order만 바꾼 branch
- first D1 protection이 시작되는 branch
- trajectory가 다시 같은 D cohort로 합쳐지는 지점

future arrival은 예측하지 않고 이미 ready이거나 outstanding인 work만 사용한다.

### P9 -- profile-free cohort protection

workload 이름이나 `multi-image` rule을 만들지 않는다. 후보는 다음 연속 상태로 평가한다.

- oldest D의 robust TPOT slack
- P 실행 후 새로 합류 가능한 D ready mass
- D1 now와 bounded wait/P-first 이후의 equal-work D rounds
- E/P completion visibility residual

정책 목표는 단순히 P를 D보다 먼저 실행하는 것이 아니라, SLO violation을 늘리지 않으면서 action-induced small-D
cohort lock-in을 피하는 것이다.

### P10 -- promotion gate

1. 5-request multi-image 20회에서 coherent probability와 tail 분포
2. scaled 40/80에서 joint-SLO goodput과 raw throughput 동시 비교
3. text-heavy exact-token identity와 3% boundary 재검증
4. full12 3회 geometric mean, 모든 workload regression `<=3%`
5. 동일 trace contract의 fresh vLLM은 최종 promotion 시 한 번만 재실행

## 11. 재현 artifact

- lifecycle A/B: `.local/p12-lifecycle-policy-ab-v2-20260905`
- original successor guard: `.local/p12-successor-guard-multi-20260905`
- gated successor guard: `.local/p12-successor-guard-multi-v2-20260905`
- rejected 18-feature full12: `.local/p12-batch-feature-full12-20260905`
- scaled current: `.local/p12-scaled-multi40-current-20260905`,
  `.local/p12-scaled-multi80-current-20260905`
- scaled zero-start: `.local/p12-scaled-multi40-zero-20260905`,
  `.local/p12-scaled-multi80-zero-20260905`
- final multi-image 5회: `.local/p12-final-multi-r5-20260905`
- final full12: `.local/p12-final-full12-20260905`
- boundary 3회: `.local/p12-final-boundary-r3-20260905`

`.local` artifact는 재현용 측정 산출물이며 Git에 포함하지 않는다.
