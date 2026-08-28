# Phase cost TTL 및 drift fallback

## 목적

portable build/fleet prior가 오래됐거나 현재 node CUDA 실행시간과 지속적으로 어긋날 때 잘못된 deadline 및
WAIT/overlap 판단을 막는다. workload label이나 workload별 parameter는 추가하지 않는다.

## 선택 경로

```text
충분한 exact-key node-local cost
    ↓ 없으면
drift/TTL을 통과한 fleet prior
    ↓ 없으면
drift/TTL을 통과한 build prior
    ↓ 없으면
sparse exact-key local cost
    ↓ 없으면
기존 analytical/static fallback
```

phase 하나가 drift 상태가 되어도 다른 E/P/D/overlap prior는 유지한다. memory hard feasibility에는 timing
prior를 사용하지 않으므로 이 변경의 영향을 받지 않는다.

## 기본 gate

- portable fleet/non-exact build prior TTL: 30일
- exact engine/plugin build prior: TTL 면제, startup anchor와 drift로 검증
- node-local observation TTL: 7일
- drift window: phase당 최근 16개
- drift 진입: 최소 8표본의 median 상대 오차가 20% 초과
- drift 해제: median 상대 오차가 10% 미만
- 비정상 ratio 0.25 미만 또는 4.0 초과는 drift 판단에서 제외
- persisted drift state는 기본 1시간만 복원

## calibration과 production 분리

controlled startup/calibration 관측은 phase scale을 보정하며 drift 표본으로 사용하지 않는다. fresh anchor가
최소 표본을 채운 phase는 기존 local-only 상태를 해제한다. production 관측은 phase scale을 바꾸지 않고
drift와 exact-key recent window만 갱신한다.

## persistence

`PhaseNodeCostJournal`은 observation과 함께 최신 drift state를 bounded host queue로 넘긴다. background
worker가 `local_cost_snapshot.json`에 두 상태를 함께 atomic write한다. inference hot path에는 filesystem I/O가
추가되지 않았다.

## 환경 변수

```text
TRT_EDGELLM_PHASE_COST_DRIFT_MIN_SAMPLES
TRT_EDGELLM_PHASE_COST_DRIFT_ENTER_RATIO
TRT_EDGELLM_PHASE_COST_DRIFT_EXIT_RATIO
TRT_EDGELLM_PHASE_COST_PRIOR_TTL_HOURS
TRT_EDGELLM_PHASE_COST_NODE_TTL_HOURS
```

## 검증 결과

단위 및 scheduler regression은 다음 14개 관련 suite `236/236`을 통과했다.

- `PhaseCostKnowledgeTest.*`
- `PhaseQueueSchedulerTest.*`
- `PhaseGlobalSchedulerTest.*`
- `PhaseGlobalCostModelTest.*`
- `PhaseThreeCoordinatorPolicyTest.*`
- `IndependentPhaseAsyncServerTest.*`
- `PhaseContinuousLoadGeneratorTest.*`
- `PhaseKVActiveViewTest.*`
- `PhasePrefixReuseTest.*`
- `PhaseMemoryBrokerTest.*`
- `PhaseDispatchWorkerTest.*`
- `PhaseExecutionSafetyContractTest.*`
- `PhaseKernelGroupRecorderTest.*`
- `StableKVPageManagerTest.*`

실제 Cosmos Reason2 2B tied engine(P8/D64, FP16 KV)에서는 build prior의 D scale만 의도적으로 `0.5`로
왜곡했다. calibration warmup 직후에는 `D:no`였으므로 warmup 표본이 drift window를 오염하지 않았다.
production decode에서는 다음과 같이 동작했다.

| 시점 | D local-only | D drift 표본 | median observed/prior |
|---|---:|---:|---:|
| post-warmup | no | 0 | 1.000 |
| 첫 격리 | yes | 2 | 2.091 |
| 종료 snapshot | yes | 7 | 2.058 |

E/P/overlap은 끝까지 local-only로 전환되지 않았다. snapshot은 실제 engine SHA-256과 plugin SHA-256을
포함한 exact deployment fingerprint로 저장했다. 동일 fingerprint로 프로세스를 재시작한 결과 node-local
snapshot을 복원했고, post-warmup health는 `E:no/P:no/D:yes/O:no`였다. 따라서 잘못된 portable D timing은
재시작 직후부터 사용되지 않으며 다른 phase의 prior는 계속 사용할 수 있다.

이번 검증은 prior를 의도적으로 망가뜨린 safety test다. scheduler action, batch formation, KV ownership 또는
실제 workload policy는 바꾸지 않았으므로 기존 workload/vLLM 성능 수치는 갱신하지 않는다.

다음 단계는 uncertainty와 실제 queue opportunity를 이용해 controlled warmup/idle 구간에서만 overlap
candidate를 능동적으로 측정하는 것이다.
