# Completion Physical Model, Transition Causality, and Warm-up Stability

날짜: 2026-09-04

브랜치: `codex/v010-phase-forward-port`

선행 문서:

- `notes/228-completion-authority-attribution-p0-p7-20260904.md`
- `notes/227-bounded-completion-calibration-full12-20260903.md`

## 1. 질문과 결론

이번 단계는 Completion-Vector를 더 공격적으로 활성화하는 대신 다음 세 질문을 분리한다.

1. completion physical model을 shadow로 실행하는 것 자체가 Scalar dispatch를 바꾸는가?
2. `multi-image`의 `D dispatch 33 -> 61`은 P+D completion-order 변경이 만드는 인과 효과인가?
3. generic warm-up을 몇 request 수행해야 RLS가 안정화되며, 더 많은 표본이 serving 성능을 단조롭게 높이는가?

현재 가장 명확한 결론은 세 번째 질문에 대한 부정이다. **모든 E/P/D 방향에 공통인 하나의 충분한 warm-up request
수는 없다.** P+D authority는 비교적 빨리 검증되지만 E+P/E+D는 훨씬 희소하고, 더 많은 generic warm-up이 raw
throughput과 TTFT/TPOT/E2E를 서로 반대 방향으로 움직이거나 특정 trace를 크게 악화할 수 있다.

따라서 production lifecycle은 다음이어야 한다.

```text
Cold start
  -> Scalar/exact policy로 즉시 serving
  -> completion physical model은 shadow observation
  -> direction별 posterior/uncertainty/held-out validation
  -> deterministic request-ready transition rollout에서만 평가
  -> repeated causal gate를 통과한 action만 bounded override
  -> 나머지는 항상 Scalar fallback
```

warm-up request 수는 workload별 tuning knob가 아니다. controlled evaluation과 optional generic bootstrap에만 사용하고,
production은 mandatory warm-up 없이 안전한 Scalar fallback에서 online evidence를 누적해야 한다.

## 2. 최종 아키텍처 경계

### 2.1 correctness mechanism

다음은 학습하지 않는다.

- request DAG dependency: vision request의 `E -> P -> D`, text request의 `P -> D`
- TensorRT profile/shape legality
- context별 single-inflight invariant
- stable KV/vision ownership과 GPU completion 이전 reclaim 금지
- 실제 ready row와 canonical request membership

### 2.2 physical observation model

RLS의 역할은 action value를 직접 결정하는 것이 아니라 실제 물리 결과를 추정하는 것이다.

```text
decision timestamp
  -> requested start skew
  -> actual first-kernel/start skew
  -> incumbent completion
  -> newcomer completion
  -> completion visibility
```

현재 completion-vector는 ordered direction별 incumbent/newcomer completion과 uncertainty를 유지하고, pair-common posterior와
direction posterior를 결합한다. exact CUDA-event epoch가 없는 observation은 counterfactual evidence로 승격하지 않는다.

### 2.3 deterministic transition evaluator

physical prediction 뒤의 request/DAG transition과 cohort formation은 학습 label에 섞지 않고 deterministic하게 적용한다.

```text
candidate action
  -> predicted physical completion boundaries
  -> completed E rows become P-ready
  -> completed P rows become D-ready
  -> completed D rows become next-token-ready or release ownership
  -> canonical next E/P/D cohort materialization
  -> exactly one more request-ready boundary
```

arbitrary future arrival은 예측하지 않는다. 현재 snapshot의 ready work와 이미 outstanding인 GPU completion만 사용한다.

### 2.4 policy authority

정책은 다음 순서를 유지한다.

```text
hard feasibility
  -> protected TTFT/TPOT/E2E slack
  -> Scalar candidate
  -> validated physical completion + bounded transition shadow
  -> confidently lower robust horizon일 때만 override
```

Completion 모델의 `enabled`와 `active`는 분리한다. Shadow는 model observation과 telemetry만 수행하며
`completion_changed_h1_action`이 0이어야 한다.

## 3. Warm-up 실험 계약

### 3.1 고정 조건

- model: `nvidia/Cosmos-Reason2-2B`
- 같은 engine/binary/process configuration
- workload 이름을 policy feature로 사용하지 않음
- fixed P chunk 128
- P max batch 8, D max batch 64, stable slots 80
- E max batch 4, encoded vision capacity 16
- generic text/VLM calibration trace만 반복
- 각 warm-up budget 뒤 measured trace는 새 process에서 실행

### 3.2 budget과 workload

warm-up budget:

- 0: cold start, measured request 중 online learning은 계속됨
- 424: 기존 P+D early-ready 지점
- 1,272: 기존 E+D early-ready 후보 지점
- 1,696: 현재 bounded calibration maximum

confirmation workload:

- `mixed`
- `vision-heavy`
- `multi-image`

각 점은 독립 process 3회다. 따라서 0은 learner 비활성화가 아니라 **사전 표본 0**을 뜻한다.

## 4. Warm-up 성능 결과

아래 값은 3회 run median이며 `CV`는 run별 generated token/s의 population coefficient of variation이다.

| warm-up | workload | token/s | CV | TTFT mean ms | TPOT p95 ms | E2E mean ms | P->D obs/ready | E->P obs/ready | E->D obs/ready |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | mixed | 1024.22 | 2.1% | 754.0 | 53.6 | 2527.7 | 0/0% | 0/0% | 0/0% |
| 0 | vision-heavy | 629.95 | 3.7% | 1420.4 | 58.0 | 3056.0 | 0/0% | 0/0% | 0/0% |
| 0 | multi-image | 320.76 | 0.1% | 231.6 | 11.6 | 494.1 | 0/0% | 0/0% | 0/0% |
| 424 | mixed | 1016.45 | 3.9% | 767.1 | 53.6 | 2542.1 | 100/67% | 6/0% | 7/0% |
| 424 | vision-heavy | 638.96 | 1.5% | 1356.2 | 54.2 | 2920.7 | 99/67% | 6/0% | 6/0% |
| 424 | multi-image | 313.86 | 11.1% | 254.9 | 10.7 | 506.6 | 99/33% | 8/0% | 6/0% |
| 1272 | mixed | 1016.44 | 0.1% | 772.6 | 51.8 | 2508.8 | 179/100% | 22/0% | 18/0% |
| 1272 | vision-heavy | 672.55 | 5.8% | 1372.6 | 67.7 | 3171.4 | 190/100% | 21/0% | 18/0% |
| 1272 | multi-image | 310.75 | 10.7% | 265.6 | 10.5 | 511.5 | 181/100% | 21/0% | 18/0% |
| 1696 | mixed | 1079.02 | 3.6% | 693.8 | 61.8 | 2423.7 | 229/100% | 27/33% | 24/0% |
| 1696 | vision-heavy | 653.94 | 3.5% | 1350.8 | 49.5 | 2889.8 | 235/100% | 28/67% | 25/0% |
| 1696 | multi-image | 248.74 | 12.8% | 348.0 | 10.8 | 611.7 | 218/100% | 27/33% | 21/0% |

`ready`는 단순 posterior sample-ready가 아니라 held-out `completion_authority_validated` 비율이다.

### 4.1 사전 표본 0 대비 변화

| warm-up | workload | token/s | TTFT mean | TPOT p95 | E2E mean |
|---:|---|---:|---:|---:|---:|
| 424 | mixed | -0.8% | +1.7% | +0.1% | +0.6% |
| 424 | vision-heavy | +1.4% | -4.5% | -6.6% | -4.4% |
| 424 | multi-image | -2.2% | +10.1% | -7.6% | +2.5% |
| 1272 | mixed | -0.8% | +2.5% | -3.4% | -0.8% |
| 1272 | vision-heavy | +6.8% | -3.4% | +16.7% | +3.8% |
| 1272 | multi-image | -3.1% | +14.7% | -9.8% | +3.5% |
| 1696 | mixed | +5.4% | -8.0% | +15.3% | -4.1% |
| 1696 | vision-heavy | +3.8% | -4.9% | -14.6% | -5.4% |
| 1696 | multi-image | -22.5% | +50.3% | -6.8% | +23.8% |

latency 열은 음수가 개선이다.

## 5. Warm-up 해석

### 5.1 posterior sample-ready와 policy-safe는 다르다

P+D는 1,272회에서 세 run 모두 authority validation을 통과했다. E+P는 1,696회에도 1/3 또는 2/3만 통과했고,
E+D는 0/3이다. 전체 `calibration_converged`는 모든 3-workload/4-budget run에서 false였다. contextual scalar policy는
대부분 수렴했지만 completion-policy held-out gate가 끝나지 않았기 때문이다.

### 5.2 더 많은 표본이 단조로운 성능 향상을 보장하지 않는다

1,272 vision-heavy는 token/s가 6.8% 증가했지만 TPOT p95가 16.7%, E2E 평균이 3.8% 악화됐다. 1,696 mixed도
token/s와 E2E 평균은 개선됐지만 TPOT p95는 15.3% 악화됐다. 이는 raw physical compression과 protected request
completion이 다른 목적임을 다시 확인한다.

### 5.3 generic warm-up도 trajectory sensitivity가 있다

multi-image의 token/s CV는 424/1,272/1,696에서 10.7--12.8%다. 같은 generic trace와 budget도 calibration의
safe-probe completion ordering과 online timing noise에 따라 다른 posterior/ready trajectory를 만든다. 따라서 고정 request
count만으로 authority를 켜면 안 된다.

### 5.4 권장 production default

- mandatory blocking warm-up: 0
- Scalar/exact serving: 즉시 시작
- physical completion observation: shadow
- P+D/E+P/E+D authority: direction별 held-out gate
- active override: repeated same-frontier evidence와 transition rollout gate를 모두 통과한 경우만
- 1,696 requests: 실험용 upper bound이지 production 권장값이 아님

### 5.5 blocking warm-up 시간 비용

gateway process의 첫/마지막 timestamp 차이에서 같은 workload의 budget 0 시간을 빼면 평균 추가 wall time은 다음과 같다.

| generic warm-up requests | 추가 wall time |
|---:|---:|
| 424 | 12.49 s |
| 1,272 | 35.44 s |
| 1,696 | 46.99 s |

대략 request당 28--29 ms이며 1,696회 full warm-up은 약 47초의 blocking startup 비용을 만든다. 그런데 이 비용을
지불해도 E+D authority는 0/3이고 multi-image가 악화될 수 있다. 따라서 mandatory full warm-up은 비용과 안정성 양쪽에서
정당화되지 않는다.

## 6. Scalar--Shadow confound 검사

같은 source manifest와 binary에서 Scalar와 Completion Shadow를 각각 실행했다. command 환경의 의도된 차이는
`TRT_EDGELLM_COMPLETION_CONFORMAL=0/1`이고 두 정책 모두
`TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE=0`이다.

| workload | Scalar token/s | Shadow token/s | delta | Scalar E2E mean | Shadow E2E mean | Shadow completion action changes |
|---|---:|---:|---:|---:|---:|---:|
| mixed | 1101.92 | 1027.71 | -6.7% | 2334.7 | 2527.6 | 0 |
| vision-heavy | 661.76 | 644.01 | -2.7% | 2692.7 | 2832.3 | 0 |
| multi-image | 323.44 | 247.05 | -23.6% | 491.7 | 614.8 | 0 |

이 표는 각 branch 1회이므로 성능 차이를 Shadow overhead로 확정하지 않는다. 실제로 선행 3회 matrix의 Shadow
multi-image는 Scalar 대비 -1.28%였고, 이번 warm-up sweep에서도 동일 budget의 multi-image token/s CV가 최대 12.8%였다.
이번 single-run의 큰 차이는 generic safe-probe/online RLS trajectory의 multimodality를 다시 보여준다.

직접적인 authority invariant는 통과했다. 세 Shadow event stream 모두 `completion_changed_h1_action=0`이다. scheduler host
decision cost는 다음과 같다.

| workload | Scalar mean/p95 us | Shadow mean/p95 us |
|---|---:|---:|
| mixed | 98.6/233.7 | 85.0/222.1 |
| vision-heavy | 72.2/201.7 | 67.9/189.8 |
| multi-image | 31.8/97.8 | 33.8/72.4 |

Shadow model 계산이 host p95를 일관되게 증가시킨 증거는 없다. 별도 process의 logical sequence는 exact하지 않았다. 이는
아래 항목을 기록한다는 기존 계획과 함께 다음처럼 해석한다.

- command/environment 차이
- `completion_changed_h1_action`
- logical decision/dispatch sequence
- scheduler host decision mean/p95/max
- token/s, TTFT, TPOT, E2E
- warm-up probe trajectory 차이

별도 process의 exact sequence equality 실패만으로 Shadow가 policy를 바꿨다고 판정하지 않는다. CUDA timing과 online
safe-probe의 작은 차이가 다음 ready boundary를 바꿀 수 있기 때문이다. 직접적인 invariant는 Shadow에서
`completion_changed_h1_action == 0`이고 completion authority가 dispatch selection에 적용되지 않는 것이다.

## 7. Multi-image P+D 인과 분기

E 정책과 candidate mechanism을 유지하고 measured phase에서 P+D experimental selection만 0%와 100%로 바꿔 각 3회
실행했다.

| branch | median token/s | TTFT mean | TPOT p95 | E2E mean | E dispatch | P dispatch | D dispatch | D GPU sum ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P+D serial | 320.17 | 227.1 | 12.72 | 495.36 | 3 | 4 | 33 | 221.88 |
| P+D overlap | 315.88 | 232.4 | 11.09 | 492.44 | 3 | 4 | 34 | 247.61 |

중앙값에서는 두 branch가 거의 같은 E2E를 보였고 overlap은 token/s -1.34%, E2E mean -0.59%다. 더 중요한 결과는
P+D serial의 한 run도 D dispatch 59회, D GPU sum 382.0 ms로 파편화됐다는 점이다. 나머지 두 serial run은 D33이다.
따라서 `P+D overlap`이라는 phase label만으로 기존 Completion Active의 D61 회귀를 재현하지 못했다. E/P completion
순서와 구체 request membership이 함께 고정되어야 한다.

exact candidate-frontier를 기준으로 두 대안이 각각 3회 이상 반복된 frontier는 0개였다. broader true-counterfactual join은
15 snapshot을 찾았지만 action disagreement가 0이어서 대안 action의 causal horizon 비교로 사용할 수 없다. 이 promotion
gate는 실패했다.

수집 항목은 다음과 같다.

- D dispatch count와 mean batch
- D GPU sum
- E/P dispatch와 request membership
- TTFT/TPOT/E2E
- identical snapshot/candidate frontier의 repeated alternative count

이 분기는 workload용 production rule이 아니다. E policy와 mechanism은 그대로 두고 P+D experimental action만 0/100%로
고정하는 research control이다.

## 8. 두 request-ready-boundary rollout

승격 전 evaluator는 다음 범위만 허용한다.

```text
boundary 0: 현재 concrete action
boundary 1: 그 action의 E/P/D physical completion으로 ready set 갱신
boundary 2: 새로 materialize된 canonical cohort 한 번 실행
```

다음은 금지한다.

- 아직 도착하지 않은 request 예측
- workload label 또는 trace별 threshold
- unbounded tree search
- learned formation/ownership transition
- physical prediction만으로 Scalar fallback 우회

초기 구현은 shadow telemetry만 생성한다. 같은 frontier에서 반복 대안과 multi-image causal branch가 future D fragmentation을
설명한 뒤에만 active promotion을 검토한다.

## 9. 구현 및 artifact

새 도구:

- `benchmarks/phase_serving/run_policy_warmup_sweep.py`
- `benchmarks/phase_serving/analyze_policy_warmup_sweep.py`
- `benchmarks/phase_serving/analyze_logical_dispatch_identity.py`
- `benchmarks/phase_serving/analyze_repeated_candidate_counterfactual.py`

실험 artifact:

- `.local/completion-transition-p8-p13-20260904/warmup-active-screen`
- `.local/completion-transition-p8-p13-20260904/warmup-active-confirm`
- `.local/completion-transition-p8-p13-20260904/identity-scalar`
- `.local/completion-transition-p8-p13-20260904/identity-shadow`
- `.local/completion-transition-p8-p13-20260904/multi-pd-serial`
- `.local/completion-transition-p8-p13-20260904/multi-pd-overlap`

## 10. Promotion gate

Completion transition override는 다음을 모두 만족하기 전까지 production default가 아니다.

1. Shadow에서 completion authority로 인한 action change 0.
2. 동일 candidate frontier에서 각 대안 action 최소 3회 common-CUDA-epoch observation.
3. immediate completion뿐 아니라 두 ready boundary의 realized regret 감소.
4. multi-image D dispatch/GPU sum과 E2E 회귀 회복.
5. 12-workload에서 Scalar 대비 throughput, TTFT, TPOT, E2E promotion gate 통과.
6. 같은 HTTP contract의 frozen vLLM 대비 SLO-goodput 우위 유지.
7. scheduler decision latency p95가 host submission gap을 유의하게 늘리지 않음.

현재 production 기본값은 계속 **Scalar**다.
