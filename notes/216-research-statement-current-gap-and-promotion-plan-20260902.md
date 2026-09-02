# Research Statement와 Current 간 차이 및 promotion 계획

## 1. 문서 목적

이 문서는 다음 두 상태를 혼동하지 않기 위한 기준 문서다.

1. 현재 성능과 correctness가 검증된 최종 production controller
2. Research Statement가 제안하는 in-flight-aware completion-projection controller

둘은 별개의 시스템이 아니다. Independent E/P/D execution, stable ownership,
bounded action, CUDA feedback이라는 substrate는 공유한다. 핵심 차이는 어떤 online
model이 policy authority를 가지며, 실행 중인 incumbent의 residual state를 실제
production decision에 얼마나 직접 사용하느냐에 있다.

관련 기준 문서:

- `198-inflight-aware-incremental-phase-scheduling-design-20260901.md`: 최초 통합 설계
- `203-inflight-m4-completion-vector-projector-20260901.md`: completion-vector projector
- `208-inflight-m6-directional-completion-rls-shadow-20260901.md`: directional RLS shadow
- `212-m7-pair-conformal-uncertainty-20260901.md`: conformal uncertainty
- `214-m7-natural-promotion-gate-20260902.md`: active M7 promotion 실패 기록
- `215-final-profile-free-contextual-controller-20260902.md`: 현재 최종 production 결과

## 2. 결론

Research Statement는 현재 방향과 대략 다음 정도로 일치한다.

| 범위 | 정성적 일치도 | 판단 |
|---|---:|---|
| Runtime와 ownership substrate | 80--90% | 대부분 구현 및 검증됨 |
| 최종 production policy semantics | 55--65% | contextual이지만 advantage authority임 |
| Research Statement의 핵심 causal evidence | 45--50% | characterization은 강하고 active completion policy 증거는 부족 |

가장 중요한 차이는 다음 한 줄이다.

```text
Current production
  candidate overlap advantage를 예측
  -> 이미 SLO-safe인 whole action의 efficiency ranking을 변경

Research Statement target
  incumbent/newcomer completion vector를 예측
  -> request TTFT/TPOT milestone에 투영
  -> 실행 중인 context에 work를 추가할지 결정
```

따라서 현재 구현을 폐기하거나 처음부터 다시 만들 필요는 없다. 이미 구현한
completion model, incremental action lease, earliest-completion projector를 shadow
검증에서 production policy authority로 승격하는 작업이 남아 있다.

## 3. 공통 기반

### 3.1 Independent execution

현재와 Research Statement 모두 다음 구조를 사용한다.

```text
shared CUDA context
    |
    +-- TensorRT context E -- CUDA stream E
    +-- TensorRT context P -- CUDA stream P
    `-- TensorRT context D -- CUDA stream D
```

각 TensorRT context는 single-inflight invariant를 유지하고, 서로 다른 context는
bounded pair action에서 동시에 outstanding일 수 있다. V1 action space는
`E/P/D/WAIT/E+P/E+D/P+D`이며 `E+P+D`는 허용하지 않는다.

### 3.2 Stable ownership

Request identity와 execution row를 분리한다.

```text
request ID
  +-- stable KV slot/page lease
  +-- vision slab/output lease
  +-- request DAG state
  `-- current E/P/D metadata

temporary batch row
  `-- current dispatch에만 유효
```

Request가 다른 cohort로 재배치돼도 KV payload를 복사하지 않는다. Vision payload도
E completion 이후 P가 소비할 때까지 request-owned lease로 유지한다. GPU consumer
completion 전에는 어떤 persistent lease도 회수하지 않는다.

### 3.3 Deterministic mechanism

다음 항목은 learned policy가 바꿀 수 없다.

- request DAG dependency
- TensorRT optimization profile와 binding shape
- context별 single-inflight
- stable KV/page/vision ownership
- page reservation과 memory feasibility
- action lease와 actual dispatch/completion correlation
- canonical request/row order

Online model은 legal candidate의 value만 바꾼다.

### 3.4 Profile-free operation

두 방향 모두 workload 이름, 외부 cost registry, persisted policy state, TTL,
`text-heavy` 또는 `vision-heavy` mode를 사용하지 않는다. Policy input은 현재
observable state와 process-local CUDA observation뿐이다.

## 4. 구조적 차이

| 항목 | Current production | Research Statement target | 남은 차이 |
|---|---|---|---|
| Learned target | normalized serial-equivalent overlap advantage | incumbent/newcomer remaining completion vector | completion model authority 승격 |
| Candidate 시점 | whole action launch가 중심 | live incumbent에 ADD E/P/D 가능 | natural residual opportunity 필요 |
| In-flight feature | residual flag와 P/D anchor | elapsed fraction, start skew, co-execution state | continuous residual feature 부족 |
| Deadline projection | exact/covering robust cost와 protected completion | predicted completion vector를 request milestone에 투영 | calibrated vector가 아직 기본 authority 아님 |
| Exact cost model | eligibility, cold start, robust bound에도 관여 | measurement/audit와 physical reference 중심 | policy dependency 축소 필요 |
| WAIT | concrete outstanding event만 기다림 | 동일 | 거의 일치 |
| Formation | 별도 H2 mechanism, production authority off | completion boundary의 successor state로 통합 가능 | H1 증거 후 선택적으로 결합 |
| Feedback | action advantage와 completion telemetry 모두 수집 | action-faithful completion label만 학습 | target 단순화 필요 |

## 5. Current production controller

현재 최종 controller의 핵심은 pair family별 16차원 RLS advantage head다.

```text
P+D RLS
E+P RLS
E+D RLS
```

현재 feature는 다음 정보를 포함한다.

- isolated primary/secondary cost
- duration ratio
- primary/secondary batch fill
- chunk/work size
- primary/secondary context bucket
- minimum protected slack
- residual augmentation 여부와 residual anchor
- primary/secondary CUDA graph variant

Reward는 다음 normalized compression이다.

```text
(serial-equivalent work - observed makespan) / serial-equivalent work
```

RLS mean과 uncertainty에서 conservative LCB를 만들고, 이미 deterministic
feasibility와 deadline gate를 통과한 후보의 decision makespan을 바꾼다. 이 구조는
workload label 없이 인접 shape evidence를 공유한다는 장점이 있다.

그러나 다음 값은 16차원 policy feature에 직접 들어가지 않는다.

- continuous incumbent elapsed fraction
- predicted remaining fraction
- detailed launch skew
- successor ready mass와 formation delta
- ownership allocate/reclaim delta

Memory horizon과 protected completion은 candidate/selector mechanism에 존재하지만
advantage RLS의 직접 feature는 아니다.

## 6. Research Statement target controller

목표 구조는 learned execution model과 deterministic service projection을 분리한다.

```text
Observed snapshot
  ready E/P/D
  incumbent phase/cohort/age
  request slack
  ownership/memory
        |
        v
Deterministic legal incremental actions
  ADD E / ADD P / ADD D / pair launch / NO_DISPATCH
        |
        v
Completion predictor
  incumbent completion mean/uncertainty
  newcomer completion mean/uncertainty
        |
        v
Request-level projection
  next TTFT milestone
  next TPOT milestone
  ownership transition
  earliest next decision boundary
        |
        v
Lexicographic selector
  feasibility -> SLO safety -> service efficiency
```

Learned model은 GPU completion behavior만 예측한다. TTFT, TPOT, request progress,
memory safety는 runtime이 명시적으로 계산한다. 따라서 scheduling objective가
training label 안에 숨지 않고, workload별 reward tuning 없이 같은 execution model을
다른 SLO에도 사용할 수 있다.

## 7. 이미 존재하지만 아직 최종 authority가 아닌 구현

### 7.1 Incremental execution lease

현재 code는 실행 중인 single-phase candidate의 unfinished portion을 만들고 live P 또는
D lease를 E+P, E+D, P+D로 upgrade할 수 있다.

```text
phaseGlobalResidualCandidate(..., elapsedUs)
phaseGlobalAugmentedDispatchPlan(...)
```

Action identity에는 outstanding set, ordered direction, start-skew bucket,
candidate/request identity가 포함된다. Failed launch 또는 action mismatch observation은
online update에서 제외한다.

### 7.2 Completion-vector predictor

`PhaseContextualCompletionModel`은 deterministic isolated completion을 baseline으로
사용하고, incumbent와 newcomer 각각의 normalized residual을 RLS로 학습한다.
Direction별 posterior는 pair-common posterior와 hierarchical shrinkage로 결합된다.

### 7.3 Conformal uncertainty

Pair-family calibrator는 chronological pre-update residual로 incumbent/newcomer 전체를
보호하는 uncertainty scale을 계산한다. 다음 두 control은 분리돼 있다.

- `enabled`: shadow prediction과 calibration
- `active`: calibrated vector가 protected-completion authority를 대체

현재 최종 production 결과에서는 completion conformal authority가 기본으로 켜져
있지 않다.

### 7.4 Earliest-completion projector

`phaseProjectEarliestCompletion()`은 completion vector에서 가장 먼저 완료되는
component까지만 state를 전진시킨다. 완료되지 않은 context는 residual completion과
ownership을 가진 채 in-flight로 유지한다. 이는 Research Statement의 bounded
incremental successor semantics와 일치한다.

## 8. 현재 증거가 말하는 것

### 8.1 강하게 증명된 항목

1. **Shape-dependent overlap**
   - 동일 fixed work에서 E1+P8은 `+11.42%`, E2+P8은 `+13.85%`다.
   - E4+P8은 `-1.03%`, E8+P8은 `-8.00%`다.
   - Stage pair만으로 profitability를 결정할 수 없다.
2. **Execution--formation coupling**
   - Controlled E8/P8에서 serial은 encoder work를 5 dispatch로 처리한다.
   - Always-overlap은 later E cohort를 파편화해 8 dispatch가 필요하다.
3. **Independent context overlap의 실제 발생**
   - 일반 burst workload idle은 `1.31--3.92%`다.
   - actual overlap은 workload별 `0.64--39.50%`다.
   - text load에서 P+D는 `23.02 -> 28.98 -> 32.27%`로 증가한다.
4. **Stable ownership와 correctness**
   - exact action/dispatch correlation, stable KV/page lease, vision semantics를
     production HTTP path에서 검증했다.
5. **Current production 성능**
   - 최종 canonical 12-workload에서 frozen vLLM token throughput을 12/12 이긴다.
   - 48.8 req/s 5-run median은 Current `41.798`, vLLM `40.901 req/s`다.

### 8.2 아직 증명되지 않은 항목

1. Natural workload에서 residual progress가 action value를 얼마나 자주 바꾸는가.
2. Co-launched pair overlap과 live-context residual augmentation의 성능 기여 분리.
3. Completion-vector predictor가 incumbent interference를 충분히 정확히 예측하는가.
4. Calibrated completion authority가 current advantage controller보다 SLO goodput을
   높이는가.
5. Immediate-cost oracle보다 request-level completion projection이 낮은 regret을
   보이는가.
6. 같은 feature와 selector가 두 번째 model/GPU의 다른 interference surface에
   workload-specific tuning 없이 적응하는가.

### 8.3 M7 negative result의 의미

Note 214의 active M7는 48.8 req/s promotion gate를 통과하지 못했다. Active와 shadow
authority-only control의 차이는 미미했으므로 completion authority 자체가 주된
회귀 원인은 아니었다. 이후 provenance-aware sampling completion boundary와
production hot-path를 정리한 최종 Current가 48.8 성능을 회복했다.

따라서 M7 실패를 “completion prediction 방향이 틀렸다”로 해석하지 않는다. 다만
completion authority가 성능을 개선했다는 증거도 아직 없다. 회복된 Current를 새
frozen baseline으로 두고 policy-only A/B를 다시 해야 한다.

## 9. Activity 결과의 정확한 해석

Activity mask는 CUDA event로 감싼 E/P/D phase work interval이다. 다음을 보여준다.

```text
phase intervals overlap했는가?                 yes
planned action과 measured interval이 대응하는가? yes
GPU SM/tensor/DRAM 자원을 동시에 효율적으로 썼는가? unknown
live incumbent에 newcomer를 추가한 residual action인가? 별도 구분 필요
```

Overlap이 많다는 사실만으로 completion-aware incremental policy의 가치를 증명할 수
없다. 현재 telemetry는 최소한 다음 두 종류를 분리해야 한다.

```text
co-launch pair
  GPU idle boundary에서 P+D/E+P/E+D를 함께 시작

residual augmentation
  incumbent가 이미 outstanding일 때 ADD P/D/E
```

최종 high-load artifact에서는 `residual_augmentation_opportunities=0`이 관찰됐다.
따라서 high-load P+D `23--32%` overlap은 in-flight-aware novelty의 직접 증거로
사용하면 안 된다. Controlled injection과 natural residual opportunity density가
별도로 필요하다.

## 10. Claim maturity

| Claim | 상태 | 논문 사용 조건 |
|---|---|---|
| E/P/D overlap profitability is shape-dependent | 검증됨 | repeated controlled error bar 추가 |
| Dispatch can fragment future cohorts | 검증됨 | equal-work timeline 유지 |
| Stable ownership enables repeated regrouping | 검증됨 | correctness/mechanism contribution 가능 |
| Profile-free advantage controller is competitive | 검증됨 | Current 12-workload/vLLM 결과 사용 |
| Residual execution state changes the best production action | 부분 검증 | injection matrix와 natural frequency 필요 |
| Completion predictor is calibrated | shadow에서 부분 검증 | direction별 held-out coverage 필요 |
| Completion-vector authority improves scheduling | 미검증 | current frozen baseline과 policy-only A/B 필요 |
| In-flight-aware selector improves SLO goodput | 미검증 | load sweep와 same-runtime ablation 필요 |
| Same controller generalizes across GPU/model | 미검증 | second GPU/model 필요 |

Research Statement의 대괄호 문장은 위 표에서 검증되지 않은 claim으로 취급한다.
실험 gate를 통과하기 전에는 완료형 문장으로 바꾸지 않는다.

## 11. 구현 및 검증 순서

### P0. Current production baseline freeze

다음을 immutable comparison anchor로 고정한다.

- commit과 binary hash
- engine/model/trace hash
- P8/D64, fixed chunk 128, stable slots 80
- 12-workload 3-run result
- 39/48.8/97.5 load result
- frozen vLLM comparison
- E/P/D/Copy activity result

Completion-vector 실험은 같은 binary와 builder lifecycle에서 authority flag만 바꾼
policy-only A/B로 시작한다.

### P1. Residual observability contract

Unified event와 activity artifact에 다음 필드를 고정한다.

- `co_launch` 또는 `residual_augmentation`
- incumbent phase/execution ID
- newcomer phase/execution ID
- dispatch direction
- incumbent dispatch age
- requested/observed start-skew bucket
- incumbent/newcomer CUDA completion
- action fidelity rejection reason

Gate:

```text
runtime dispatch count == activity dispatch count
planned outstanding set == actual outstanding set
accepted observation action ID == completion action ID
```

### P2. Completion feature V2와 shadow collection

Workload label 없이 다음 continuous state를 추가한다.

- normalized incumbent dispatch age
- isolated reference 대비 elapsed ratio
- requested/observed launch skew
- incumbent/newcomer cost ratio
- current outstanding set
- protected minimum slack ratio

완료 전 CUDA progress를 실제 kernel progress라고 가정하지 않는다. Host-observed age와
완료된 kernel-group marker만 feature로 사용한다.

Completion model은 계속 shadow로 실행하고 selection을 바꾸지 않는다.

### P3. Controlled residual causal matrix

동일 fixed work에 대해 newcomer injection point를 바꾼다.

```text
incumbent progress target: 0 / 25 / 50 / 75 / near-complete%
directions: P->D, D->P, E->P, P->E, E->D, D->E
```

각 point에서 측정한다.

- incumbent completion delay
- newcomer completion
- pair makespan
- serial-equivalent compression
- TTFT/TPOT milestone projection error
- prediction mean/interval coverage

같은 candidate의 value가 residual state에 따라 바뀌는 sign reversal 또는 의미 있는
regret 차이가 없으면 in-flight residual을 central contribution으로 승격하지 않는다.

### P4. Natural opportunity density

12-workload와 load sweep에서 다음을 분리 집계한다.

- co-launch opportunities/selections
- residual augmentation opportunities/selections
- direction과 skew별 observation 수
- completion-aware와 current selector의 disagreement
- disagreement의 replay regret

Residual opportunity가 극히 드물면 completion predictor는 robustness mechanism으로
유지하고 논문 중심은 transition-aware whole-action scheduling으로 낮춘다.

### P5. Completion authority promotion gate

Shadow에서 active로 승격하려면 다음을 모두 만족해야 한다.

1. exact token/semantic correctness 유지
2. action fidelity violation `0`
3. calibrated direction에서 conformal false-safe `0`
4. incumbent/newcomer held-out interval coverage가 목표 coverage와 일치
5. scheduler p95가 frozen Current 대비 유의미하게 회귀하지 않음
6. 48.8 req/s raw throughput `>= 40 req/s`
7. 48.8 joint-SLO pass `>= 99%`
8. authority-off Current 대비 SLO goodput 또는 latency에서 반복 가능한 개선

Prediction이 ready하지 않거나 calibration gate를 통과하지 못한 direction은 Current의
robust bound로 자동 fallback한다.

### P6. Same-runtime full evaluation

다음 policy를 동일 runtime mechanism에서 비교한다.

| Policy | 목적 |
|---|---|
| Current advantage | frozen production baseline |
| Serial | overlap opportunity의 하한 |
| Always overlap | static overlap 반례 |
| Immediate-cost | current action만 보는 myopic baseline |
| Completion shadow | overhead-only control |
| Completion active | Research Statement target |
| Completion w/o uncertainty | uncertainty ablation |
| Completion w/o residual state | central premise ablation |
| New-work-only completion | incumbent completion 예측의 가치 |

Primary metric은 joint-SLO goodput이고 throughput, TTFT, TPOT, E2E mean/p95를 모두
보고한다. Trace/model/engine 계약이 같으면 frozen vLLM을 재사용하고, 계약이 바뀐
경우에만 fresh vLLM을 실행한다.

### P7. H2 successor의 조건부 결합

Completion-aware H1이 안정화되기 전에는 H2 authority를 켜지 않는다. H1 이후에도
immediate best와 short-horizon best가 반복적으로 달라지는 measured state가 있을 때만
earliest completion successor를 한 단계 추가한다. Future arrival은 예측하지 않는다.

### P8. Architecture와 generality

선택된 causal point에서 Nsight Systems/Compute로 다음을 측정한다.

- SM active와 concurrent kernels
- tensor utilization
- DRAM/L2 pressure
- host launch gap
- event visibility와 sampling completion delay

그 후 두 번째 GPU와 두 번째 model에서 같은 feature와 selector를 사용한다. 목표는
동일 action을 선택하는 것이 아니라 서로 다른 overlap frontier를 online completion
model이 학습하는지 확인하는 것이다.

## 12. 최종 promotion 판단

성공하면 최종 architecture는 다음과 같다.

```text
Independent E/P/D execution
        +
Stable lifetime-aware ownership
        +
Deterministic bounded incremental actions
        +
Online completion-vector prediction
        +
Request-level SLO projection
```

실패하거나 natural residual opportunity가 희박하면 현재 production architecture를
유지한다.

```text
Independent E/P/D execution
        +
Stable lifetime-aware ownership
        +
Profile-free contextual pair-action value controller
```

어느 경우에도 workload별 fine-tuning, workload name 분기, 외부 profile registry,
shape-specific policy table은 추가하지 않는다. Research Statement 방향의 가치는
복잡한 predictor를 구현했다는 사실이 아니라, 실제 production trace에서 residual
completion state가 action regret과 SLO goodput을 개선한다는 causal evidence로만
판단한다.
