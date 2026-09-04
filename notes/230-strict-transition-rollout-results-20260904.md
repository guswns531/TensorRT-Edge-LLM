# Strict Transition Rollout 구현 및 검증 결과

## 결론

이번 단계는 Completion-Vector Active를 승격하지 않았다. 대신 그 전에 필요했던 실험 계약을 구현하고 실제로
검증했다.

> A richer, independently validated physical completion model can still hurt end-to-end performance when its decisions
> alter subsequent cohort formation.

현재 데이터가 지지하는 범위는 여기까지다. Completion-Vector를 Scalar보다 “더 정확한 policy”라고 부르지
않는다. 물리 completion 측정과 policy authority를 분리하고, branch 이후 request DAG와 cohort 변화를 두 개의
ready boundary까지만 재생하는 기반을 마련했다.

## 1. 구현한 것

### 1.1 Strict pre-branch identity

`PhaseUnifiedEvent`의 causal identity는 이제 다음을 모두 포함한다.

```text
ready E/P/D request IDs와 row order
prefill progress / decode KV length
outstanding phase와 request membership
candidate action kind/shape/request row order
KV slot/page/generation/committed-length ownership
vision payload lease lifetime
scalar cost/deadline policy inputs
```

마지막 `scalar_policy_state_signature`가 중요하다. queue/frontier가 같아도 이전 CUDA 관측으로 만들어진 scalar
cost state가 다르면 같은 policy state가 아니다. Shadow-only completion prediction은 이 signature에서 제외한다.

### 1.2 One-shot forced causal branch

다음 연구 전용 환경변수를 추가했다.

```text
TRT_EDGELLM_REPLAY_DECISION_SEQUENCE
TRT_EDGELLM_REPLAY_ACTION_KIND
```

지정된 policy decision에서 exact frontier에 존재하고 hard-feasible한 action만 한 번 강제할 수 있다. 없는 action,
불가능한 action은 조용히 fallback하지 않고 실패한다. event에는 `policy_decision_sequence`와
`causal_replay_forced`가 기록된다.

### 1.3 정확히 두 request-ready boundary의 deterministic replay

새 evaluator는 physical completion vector를 completion 시각 순으로 정렬한 뒤 정확히 두 개의 ready boundary만
재생한다.

```text
E completion -> request E-ready에서 P-ready로 이동
P completion -> request P-ready에서 D-ready로 이동, vision lease release
D completion -> D-ready 유지 또는 완료, 완료 시 KV lease release
```

동시 completion은 하나의 boundary로 합친다. completion이 한 시각뿐이면 두 번째 boundary에 임의의 future
arrival을 만들지 않고 첫 boundary를 반복한다. live queue와 allocator는 전혀 변경하지 않는 shadow mechanism이다.

### 1.4 Scaled real-request wave builder

기존 5-request semantic multi-image trace를 그대로 반복하면서 arrival offset만 wave 단위로 이동시키는 도구를
추가했다. 이번 검증은 8 waves, 40 requests, wave interval 150 ms를 사용했다. 원래 message/image/semantic check와
output length는 유지된다.

### 1.5 실험기 확장

기존 manifest의 workload 계약은 유지하면서 다음을 override할 수 있다.

```text
--warmup-requests
--trace
--backend-env NAME=VALUE
--telemetry-level counterfactual
```

## 2. Scalar–Shadow confound 검증

### 2.1 첫 strict identity에서 발견한 누락

multi-image cold-start를 Scalar 5회, Completion Shadow 5회 실행했다.

| 항목 | 결과 |
|---|---:|
| 공통 queue/frontier/ownership strict signature | 42 |
| cross-run matched pairs | 57 |
| candidate frontier/row-order 일치 | 57/57 |
| selected action 일치 | 56/57 |

유일한 불일치는 동일 frontier `P1 / D1 / P1+D1`에서 Scalar가 `D1`, Shadow가 `P1`을 선택한 경우였다.
Completion authority change는 양쪽 모두 0이었다. 원인은 completion policy가 아니라 같은 queue state에 도달하기
전 축적된 scalar CUDA cost history가 달랐던 것이다.

따라서 queue/frontier/ownership만 같은 것을 “동일 pre-branch state”라 부르면 안 된다.

### 2.2 Scalar policy-state signature 추가 후

새 바이너리로 Scalar 3회와 Completion Shadow 3회를 다시 실행했다.

| 항목 | 결과 |
|---|---:|
| 양쪽 completion H1 authority change | 0 |
| 공통 full strict signature | 2 |
| matched pairs | 7 |
| frontier/row-order 일치 | 7/7 |
| selected action 일치 | 7/7 |

공통 signature 수가 줄어든 것은 실패가 아니다. 실제 policy state까지 같아야 match하므로 올바른 결과다. 이
조건에서 Shadow가 action을 바꾼 증거는 없다.

3-run median은 다음과 같다. 이 작은 trace는 두 실행 모드가 공통적으로 약 235~323 token/s의 multimodal한
분포를 보여 성능 차이를 policy 효과로 해석할 수 없다.

| Policy | token/s | TTFT mean ms | TPOT p95 ms | E2E mean ms | E2E p95 ms |
|---|---:|---:|---:|---:|---:|
| Scalar | 292.352 | 261.404 | 12.165 | 531.065 | 546.863 |
| Completion Shadow | 302.094 | 251.855 | 11.787 | 518.658 | 528.961 |

## 3. Forced multi-image branch replay

Cold-start decision 2의 `encoder`와 `encoder_prefill`을 각각 5회 강제했다. 모든 run에서 요청한 action이 한 번만
강제됐고 output token hash는 동일했다.

| Branch | token/s median | TTFT mean median ms | TPOT p95 median ms | E2E mean median ms | D-containing decision records |
|---|---:|---:|---:|---:|---|
| forced E | 317.987 | 258.683 | 9.662 | 498.631 | 66, 33, 66, 33, 32 |
| forced E+P | 304.274 | 248.091 | 11.987 | 515.632 | 39, 37, 43, 32, 33 |

그러나 두 branch 사이에 공통 full strict pre-branch signature가 0개였다. 비동기 image preparation/admission이
decision 2 이전의 request membership과 scalar policy history를 바꿨기 때문이다. 따라서 위 표는 descriptive
결과일 뿐 causal effect가 아니다. analyzer의 `fully_repeated_multi_action_frontiers`도 정확히 0이며 이 결과를
killer evidence로 사용하지 않는다.

이 실패는 forced action 기능의 실패가 아니라 causal pairing 전제의 실패다. 다음 causal 실험은 process를 두 번
시작해 우연히 snapshot이 같아지기를 기대하지 않고, frozen snapshot에서 branch state를 clone하는 in-process
replay가 필요하다.

## 4. Common CUDA epoch physical completion 반복

E->P, requested start fraction 0.5를 같은 action execution에서 5회 측정했다. 다섯 실행 모두 실제 GPU overlap과
action fidelity는 참이었다.

| 물리량 | median | range |
|---|---:|---:|
| actual start fraction | 1.713 | 1.229–1.865 |
| E completion | 31.225 ms | 25.798–32.528 |
| P completion | 20.346 ms | 20.146–20.698 |
| pair makespan | 39.545 ms | 33.834–40.665 |
| serial-equivalent compression | -56.50% | -59.21%–-54.97% |

즉 common epoch에서 incumbent/newcomer completion vector는 5개 확보했지만 requested 50% launch는 하나도
허용 오차에 들어오지 않았다(`accepted_buckets=0`). 실제 P 시작이 E reference의 123~187% 지점까지 늦었다.
이는 Completion-Vector 학습 전에 realization model을 분리해야 한다는 직접 증거다. 요청 skew를 feature로 넣고
이 관측을 50% label로 학습하면 잘못된 모델이 된다.

## 5. Scaled multi-image 40-request 결과

8-wave trace를 Scalar와 Completion Active 각각 3회 실행했다. warmup은 0이고 Active는 serving 중 표본이 쌓인
후 일부 authority를 얻었다.

| Metric | Scalar | Completion Active | Active delta |
|---|---:|---:|---:|
| token/s | 399.861 | 391.960 | -1.98% |
| TTFT mean ms | 1018.057 | 1003.816 | -1.40% |
| TTFT p95 ms | 1723.877 | 1727.441 | +0.21% |
| TPOT mean ms | 33.412 | 31.446 | -5.88% |
| TPOT p95 ms | 53.642 | 48.929 | -8.79% |
| E2E mean ms | 2063.971 | 1978.643 | -4.13% |
| E2E p95 ms | 2391.491 | 2422.139 | +1.28% |

Active의 run별 H1 action change는 1/4/6회, completion authority application은 31/36/36회였다. D-containing
decision record는 Scalar 142/164/143, Active 124/153/157로 일관된 fragmentation 감소가 아니었다.

해석은 두 가지다.

1. richer physical model은 평균 TPOT/E2E를 개선할 수 있다.
2. 동시에 throughput과 E2E tail을 악화시킬 수 있으며, 현재 scalar completion reward만으로는 이 trade-off를
   안전하게 선택하지 못한다.

따라서 “Completion-Vector가 더 좋은 policy”라는 주장은 기각하고, transition rollout의 필요성만 유지한다.

## 6. Promotion gate 판정

| Gate | 상태 | 근거 |
|---|---|---|
| Scalar–Shadow logical neutrality | 통과(조건부) | full policy-state가 같은 7/7 action 일치 |
| 3–5 common-epoch physical observations | 부분 통과 | 5개 vector 확보, requested skew realization 실패 |
| strict forced branch causal pair | 실패 | 공통 full strict pre-branch 0 |
| scaled multi-image natural evidence | 통과 | 40 requests x 3, Active의 mixed mean/tail trade-off 재현 |
| two-boundary mechanism correctness | 통과 | DAG/vision/KV transition 단위 테스트 |
| transition rollout active promotion | 보류 | causal pair와 realization gate 미통과 |
| 12-workload active gate | 실행하지 않음 | 이전 gate 미통과 |
| fresh vLLM rerun | 실행하지 않음 | workload 계약이 바뀐 최종 후보가 없음 |

이는 계획을 중단한 것이 아니라 gate를 지킨 결과다. 근거 없이 rollout authority를 켜고 12개/vLLM 숫자를
생성하면 policy 효과와 confound가 다시 섞인다.

## 7. 다음 구현 순서

### N1. In-process frozen-snapshot branch clone

동일 coordinator snapshot, cost-model state, candidate frontier, KV/vision ownership을 immutable replay object로
복제한다. 두 branch는 live TensorRT를 동시에 실행하는 대신 이미 측정된 common-epoch physical vector를 주입해
두 ready boundary까지 deterministic하게 재생한다.

### N2. Realization predictor 분리

```text
requested skew -> predicted actual first-kernel skew
actual skew + shape -> incumbent/newcomer completion stretch
```

첫 모델이 target miss를 예측하지 못하면 overlap candidate를 authority 대상으로 올리지 않는다.

### N3. Scaled trace에서 transition prediction 검증

H1 change가 발생한 1/4/6개 지점을 frozen replay로 평가한다. 예측 대상은 immediate pair makespan이 아니라:

```text
두 ready-boundary까지 protected completion
successor E/P/D cohort IDs와 row order
D cohort size sequence / dispatch delta
vision/KV release
```

### N4. 제한적 Active gate

multi-image, vision-heavy, mixed 세 regression trace에서 다음을 모두 만족할 때만 rollout authority를 켠다.

```text
strict causal pair >= 3 per action
predicted action regret sign agreement >= 80%
false-safe = 0
throughput regression <= 3%
TTFT/TPOT/E2E p95 regression <= 3%
```

### N5. 그 이후에만 전체 평가

N4 통과 후 12-workload cross-order를 실행한다. 동일 trace 계약의 frozen vLLM 결과가 있으면 재사용하며, trace,
warmup, SLO 또는 runtime contract가 바뀐 경우에만 fresh vLLM을 실행한다.

## 8. 검증

```text
C++: PhaseFormationPlannerTest + PhaseUnifiedEventTest + PhaseGlobalSchedulerTest
     60/60 passed
Python: policy matrix runner + scaled-wave builder
        9/9 passed
Output token hash: 모든 5-request branch에서 동일
```

핵심 결론은 단순하다.

```text
physical completion model 정확도
          !=
end-to-end action 가치

action 가치 = physical completion
            + deterministic DAG transition
            + successor cohort formation
            + ownership release
```

그리고 이 식을 검증하려면 서로 다른 process run의 비슷한 snapshot이 아니라, scalar policy state까지 포함한
완전히 동일한 frozen pre-branch snapshot이 필요하다.
