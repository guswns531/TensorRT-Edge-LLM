# In-flight-aware incremental phase scheduling: design and implementation plan

## 1. 문서 목적

이 문서는 현재까지 구현하고 측정한 single-GPU E/P/D serving substrate 위에서 다음 scheduler 구조를 구현하기 위한 기준 문서다.

```text
Independent E/P/D TensorRT execution contexts
                  +
Stable KV / vision ownership
                  +
In-flight-aware incremental dispatch
                  +
Completion-vector prediction
                  +
Request milestone projection
```

목표는 workload 이름이나 `P8/D32`, `E1/P8` 같은 shape별 정책표를 사용하지 않고, 현재 observable state만으로 다음 dispatch를 결정하는 것이다. 외부 cost registry, TTL 기반 evidence expiry, workload profile별 fine-tuning은 사용하지 않는다.

이 문서는 다음 두 초안을 하나로 합치고, 실제 구현에서 발견된 제약을 반영한다.

- 논문 Section 1--4의 새 구조: global in-flight-aware stage scheduling
- 단계별 구현 및 평가 계획: state, action, predictor, projector, selector, injection benchmark
- [197번 결과](197-realized-transition-attribution-and-concrete-boundary-20260831.md): whole-H2 horizon과 실제 earliest legal D boundary의 불일치

## 2. 현재 구현 상태와 구조 변경의 이유

현재 작업 트리에는 다음 기반이 이미 있다.

- 동일 CUDA context 안의 independent E/P/D TensorRT execution contexts와 streams
- stable indexed/page KV ownership과 vision payload lifetime 관리
- global phase candidate generation과 execution lease
- P+D, E+P, E+D contextual action-value model
- equal-work H2 formation planner와 concrete sampling completion preview
- 선택 action과 실제 4-dispatch 전개의 realized transition attribution

최종 H2와 myopic의 장기 3회 A/B에서는 H2가 throughput `+0.28%`, TTFT mean `-2.98%`, E2E p95 `-2.01%`를 보였고 token hash도 모두 동일했다. 그러나 H2가 action을 바꾼 10개 episode 중 D budget이 알려진 9개 모두 local D budget을 위반했다. 예측 regret과 실제 first-D gap의 상관도 약했다.

원인은 다음 모델 차이다.

```text
현재 H2 모델
  whole action horizon complete
        -> concrete D boundary를 뒤에 추가

실제 runtime
  E/P/D 중 가장 먼저 끝나는 component
        -> execution lease 갱신
        -> residual augmentation 또는 새 dispatch 가능
        -> 다음 decision boundary
```

따라서 다음 단계는 임의 H=3 예측을 추가하는 것이 아니다. 실행 중인 work를 scheduler state에 직접 포함하고, 현재 action이 incumbent와 newcomer 각각의 완료시간을 어떻게 바꾸는지 예측해야 한다.

## 3. 핵심 연구 주장

`global scheduler` 자체를 novelty로 주장하지 않는다. 핵심은 다음과 같다.

> 하나의 GPU에서 E/P/D actor가 독립적으로 enqueue될 수 있을 때, 새 cohort의 추가는 새 work뿐 아니라 이미 실행 중인 work의 완료시간도 바꾼다. Scheduler는 bounded incremental action의 completion vector를 request milestone과 SLO에 투영해 다음 dispatch 또는 WAIT를 선택한다.

Phase action은 단순한 GPU work가 아니다.

```text
Phase action
    |
    +-- immediate GPU execution and interference
    +-- persistent KV / vision ownership transition
    +-- next legal decision boundary
    `-- near-future batch formation
```

Execution--Formation Coupling은 이미 측정된 중요한 현상으로 유지한다. 다만 formation H2는 새 기본 알고리즘의 중심이 아니라, H1 incremental decision만으로 부족함이 확인될 때 추가하는 bounded successor optimization으로 둔다.

## 4. 설계 원칙

### 4.1 Mechanism과 policy의 분리

Deterministic mechanism이 action legality와 lifetime correctness를 보장한다. Online policy는 legal action 가운데 가치 있는 action만 고른다.

```text
Mechanism
  dependency / TensorRT shape / single-inflight / ownership / memory
                          |
                          v
                    Legal actions
                          |
                          v
Policy
  completion prediction / SLO projection / action ranking
```

### 4.2 Workload label-free

Scheduler는 `text-heavy`, `vision-heavy`, `latency`, `throughput` 같은 profile을 입력으로 받지 않는다. 다음 observable state만 사용한다.

- ready E/P/D work
- request age와 TTFT/TPOT slack
- host가 관측한 outstanding context와 dispatch age
- 이미 제출된 completion events
- stable KV/vision ownership과 memory headroom
- phase cost와 observed interference
- 선택 action이 만드는 deterministic successor state

### 4.3 Bounded observable horizon

임의 future arrival을 예측하지 않는다. 현재 immutable ready work, 이미 outstanding인 completion event, 선택 action이 직접 만드는 state transition만 사용한다.

### 4.4 Planned action과 actual execution의 일치

선택 action은 한 번의 poll에서 enqueue할 작업만 뜻하지 않는다. 다음 completion boundary까지 허용되는 outstanding phase set을 뜻한다.

```text
planned outstanding set = actual outstanding set
```

이 invariant가 깨진 observation은 online model update에 사용하지 않는다.

## 5. Scheduler state

시점 `t`의 snapshot은 다음 세 종류로 나눈다.

### 5.1 Ready state

- phase별 compatible queue prefix
- E image/shape cohort와 batch capacity
- P packed rows, useful/padded tokens, fixed chunk 128 state
- D ready rows, graph bucket, context-length aggregate
- 이미 제출된 sampling completion이 만들 concrete D-ready preview

### 5.2 In-flight state

- E/P/D context별 idle, submitted, running, completion-visible 상태
- dispatch timestamp와 host-observed age
- batch/row/token/shape metadata
- execution lease와 action lineage
- start/end CUDA events와 coarse kernel-group milestones
- incumbent request IDs와 ownership handles

중요한 제한은 실행 중 TensorRT work의 정확한 GPU 진행률을 host가 실시간으로 직접 알 수 없다는 점이다. CUDA event elapsed time은 완료 전 online progress 값으로 사용할 수 없다. 초기 구현은 `dispatch 이후 host-observed age + action shape + event readiness`를 사용한다. kernel-group segmentation이 제공되는 경로에서는 완료된 milestone까지만 추가 관측으로 사용한다.

### 5.3 Request and resource state

- request DAG와 next milestone
- TTFT deadline, TPOT deadline, request age
- stable KV page lease와 예상 growth/reclaim
- vision slab lease와 P 소비 시점
- TensorRT context activation memory
- concurrent workspace lease
- CUDA graph buffers와 sampling/copy staging

## 6. Incremental action space

Action은 phase label만으로 식별하지 않는다. 다음 정보가 action identity에 포함된다.

- incumbent outstanding set
- 새로 추가할 phase와 cohort size
- dispatch direction: `D running -> add E`와 `E running -> add D`의 구분
- idle pair launch order
- 실제 start skew bucket
- selected TensorRT optimization profile/graph bucket

V1 action legality는 다음으로 제한한다.

| Current outstanding | Legal action |
|---|---|
| none | E, P, D, supported E+P/E+D/P+D pair launch, WAIT |
| one phase | 다른 phase 하나 추가, 또는 NO_DISPATCH |
| two phases | NO_DISPATCH |

추가 invariant:

- 같은 TensorRT execution context에는 동시에 두 enqueue를 허용하지 않는다.
- V1에서는 E+P+D를 허용하지 않는다.
- failed launch는 observation이나 policy feedback으로 기록하지 않는다.
- candidate membership은 `deadline -> age -> stable request ID`의 deterministic compatible prefix에서 만든다.
- policy는 bounded size 후보만 선택하며 임의 request 조합을 탐색하지 않는다.

## 7. Completion-vector predictor

### 7.1 예측 대상

새 action `a`를 적용했을 때 단일 makespan만 예측하지 않는다.

```text
CompletionVector(a) = {
  incumbent E remaining completion,
  incumbent P remaining completion,
  incumbent D remaining completion,
  newcomer E completion,
  newcomer P completion,
  newcomer D completion,
  uncertainty for each observed component
}
```

이 벡터에서 가장 이른 completion이 다음 실제 decision boundary 후보가 된다.

### 7.2 학습 label

선택하지 않은 action의 counterfactual slowdown은 online에서 관측할 수 없다. 학습 label은 실행한 action에 대해서만 다음처럼 정의한다.

```text
observed remaining completion
    = CUDA completion timestamp - decision timestamp
```

action fidelity가 확인되고, completion attribution이 exact-once이며, request/lease lineage가 일치하는 observation만 update한다.

### 7.3 모델 구조

첫 learned predictor는 CPU-only residual RLS/Bayesian linear model을 사용한다.

- direction-specific family: P->D, D->P, E->D, D->E, E->P, P->E
- compact continuous features: isolated cost, batch fill, duration ratio, context, slack, start skew, incumbent age, formation delta, ownership delta
- exact CUDA cost model은 execution/debugging reference로 유지
- low-dimensional model은 scheduling decision에 사용
- EMA residual은 thermal/clock/background drift correction에만 사용
- TTL과 external registry는 사용하지 않음

예측은 평균과 uncertainty를 함께 반환한다.

```text
robust completion = mean + beta(slack) * uncertainty
```

cold state에서는 안전한 serial 또는 제한된 controlled probe를 사용한다.

### 7.4 Oracle의 정의

Oracle은 production online counterfactual oracle이 아니다. 다음 중 하나로만 정의한다.

- 동일 snapshot을 재현하는 controlled injection/replay
- 미리 측정한 candidate 결과를 이용한 offline trace oracle
- 동일 runtime에서 legal action들을 반복 실행한 measured oracle

따라서 `Oracle H1`은 predictor와 selector를 분리 검증하기 위한 실험 도구다.

## 8. Deterministic request milestone projector

Predictor는 GPU component completion을 출력하고, projector는 이를 request 결과로 변환한다.

```text
Completion vector
       |
       v
Earliest completion event
       |
       +-- request DAG transition
       +-- TTFT/TPOT milestone update
       +-- KV/vision ownership transition
       +-- remaining incumbent state
       `-- next decision snapshot
```

Projector가 반드시 구분해야 하는 것은 `action 전체 완료`와 `다음 event 완료`다. 예를 들어 D가 E보다 먼저 끝나면 D completion 시점에 E는 계속 in-flight이고, 그 상태에서 새 P 또는 D 추가 가능성을 다시 평가한다. 이 구조가 현재 whole-H2 horizon 뒤 D를 붙이는 mismatch를 직접 제거한다.

Projector는 예측 모델이 아니라 deterministic mechanism이다. request DAG, execution lease, context single-inflight, stable ownership transition을 그대로 적용한다.

## 9. Feasibility와 memory

모든 candidate는 ranking 전에 다음 hard feasibility를 통과해야 한다.

- request dependency: vision request의 P는 E completion/lease readiness 이후만 가능
- TensorRT optimization profile과 binding shape 지원
- context별 최대 one in-flight
- stable KV/vision lease 유효성
- GPU consumer completion 전 reclaim 금지
- memory horizon 내 allocation 가능

Memory feasibility에는 KV와 vision만 포함하면 안 된다.

```text
M_required(a) =
  persistent KV growth
  + vision payload growth
  + context activation memory
  + non-aliasable concurrent workspace
  + CUDA graph buffers
  + sampling/copy staging
  - guaranteed reclaim
```

독립 TensorRT context가 항상 별도 workspace를 가져야 하는 것은 아니다. serial contexts는 activation/workspace lease를 재사용할 수 있지만, 실제 concurrent contexts는 서로 alias하지 않는 workspace가 필요하다. 모델 weights는 공유한다.

## 10. Selector policy

Weighted heuristic soup를 만들지 않고 lexicographic ordering을 유지한다.

1. **Hard feasibility**
2. **Robust request-level SLO violation 최소화**
3. **Urgency-normalized milestone progress 최대화**
4. **GPU service efficiency 및 ownership release**
5. **Stable deterministic tie-break**

SLO safety는 predictor uncertainty를 포함한다.

```text
Slack_FT(r, a)
  = TTFT_deadline
    - request_age
    - robust remaining critical path under a

Slack_TPOT(r, a)
  = next_token_deadline
    - robust next-D completion under a
```

`WAIT`는 별도 timer heuristic이 아니다. 당장 enqueue하지 않고 이미 존재하는 가장 가까운 useful completion event까지 기다리는 `NO_DISPATCH` action이다. CUDA event 자체가 CPU를 깨우지는 않으므로 completion worker 또는 bounded polling이 event readiness를 coordinator에 알린다. CUDA host callback을 사용할 경우 callback 안에서는 CUDA API를 호출하지 않는다.

## 11. 기존 구현의 재사용과 역할 변경

| 현재 구성요소 | 새 구조에서의 역할 |
|---|---|
| `PhaseThreeCoordinator` | request DAG와 completion/event-loop mechanism |
| `PhaseQueueScheduler` | deterministic compatible prefix와 bounded cohort materialization |
| `PhaseGlobalScheduler` | legal incremental action의 lexicographic ranker |
| execution lease/action plan | outstanding-set 전이와 action fidelity의 correctness authority |
| contextual P+D/E+P/E+D heads | learned completion predictor의 bootstrap/baseline |
| `PhaseFormationPlanner` | replay/oracle 도구와 선택적 H2 successor projection |
| realized transition tracker | predictor label, action fidelity, prediction error attribution |
| stable KV/vision ownership | memory feasibility와 deterministic projector substrate |
| exact phase cost model | precise execution/debugging reference; policy table로 사용하지 않음 |

현재 `H2 + first concrete completion boundary`는 production default에서 제거하지 않아도 되지만 research opt-in으로 유지한다. 새 H1 incremental projector가 promotion gate를 통과하기 전까지 production default는 기존 myopic transition-safe selector다.

## 12. Directional injection benchmark

Predictor와 action semantics를 검증하려면 pair 종류뿐 아니라 방향과 실제 start offset을 분리해야 한다.

대상:

- P running -> add D
- D running -> add P
- E running -> add D
- D running -> add E
- E running -> add P
- P running -> add E

목표 offset은 incumbent isolated duration의 0/25/50/75/90%다. 그러나 host sleep 시간으로 실제 GPU offset을 간주하지 않는다.

```text
incumbent CUDA start/end
candidate CUDA start/end
          |
          v
post-hoc actual start-offset calculation
          |
          v
accepted sample bucket
```

선택 지점은 Nsight Systems/CUPTI와 kernel-group marker로 검증한다. 이 실험은 다음 질문에 답해야 한다.

- injection timing에 따라 incumbent remaining time이 실제로 달라지는가?
- newcomer completion과 incumbent slowdown을 동시에 예측해야 하는가?
- direction/order/start skew가 action ranking을 바꾸는가?

## 13. 구현 단계

### M0. Baseline freeze

- 현재 myopic production default와 H2 research 결과 고정
- engine/model/trace/driver/binary SHA와 request/output contract 기록
- unified decision/dispatch/completion telemetry schema 고정

### M1. In-flight snapshot과 unified event log

Status: complete. 구현 및 검증 상세는 `notes/200-inflight-m1-shadow-snapshot-and-event-chain-20260901.md`를 참조한다.

- E/P/D context state, dispatch age, lease, request lineage 노출
- `decision -> enqueue -> GPU start -> GPU completion -> host visibility` timestamp 연결
- 새 selector는 shadow-only

### M2. Directional injection benchmark

Status: complete. 구현, 실제 CUDA 방향성 결과, Gate A 판정은
`notes/201-inflight-m2-directional-injection-benchmark-20260901.md`를 참조한다.

- 6개 direction과 5개 actual offset bucket 측정
- incumbent/newcomer completion vector와 uncertainty artifact 생성
- Gate A 판단

### M3. Incremental action enumerator와 legality

Status: complete. 0/1/2-context frontier, ordered direction/start-skew identity, execution-lease fidelity, 실제 Cosmos
검증은 `notes/202-inflight-m3-incremental-action-legality-20260901.md`를 참조한다.

- 0/1/2 outstanding 규칙 구현
- direction/order/start skew를 action identity에 포함
- 실행 lease가 planned/actual outstanding set fidelity 검증
- E+P+D와 same-context re-enqueue 명시적 거부

### M4. Replay predictor와 deterministic H1 projector

Status: complete. earliest-component replay, request DAG/ownership transition, actual Cosmos shadow replay 결과는
`notes/203-inflight-m4-completion-vector-projector-20260901.md`를 참조한다.

- 측정 oracle/replay predictor 연결
- earliest event에서 request DAG와 ownership transition 적용
- 다음 snapshot 재구성 단위 테스트
- 기존 whole-H2 append-D mismatch 재현 및 제거

### M5. Oracle H1 selector

Status: ranker, residual-boundary correction, progress-aware deterministic signature, cross-run coverage builder와 실제
Cosmos VLM repeat validation까지 완료했다. 동일 policy repeat에서는 7개 exact snapshot이 재현됐지만 multi-action exact
coverage는 0이므로 Gate B는 아직 평가 불가다. 구현과 판정 상세는
`notes/204-inflight-m5-oracle-h1-selector-20260901.md`를 참조한다. Production default는 변경하지 않았다.

- feasibility -> robust SLO -> progress -> efficiency ranker 연결
- myopic/legacy-compatible/current H2와 same-runtime policy-only A/B
- 12-workload와 load sweep으로 Gate B 판단

### M6. Learned residual RLS predictor

Status: hierarchical shadow implementation complete; Gate C not yet evaluable/passed. Three canonical pair-common
posteriors now shrink six ordered incumbent/newcomer completion posteriors, so a cold reverse direction can reuse pair
evidence without erasing direction asymmetry. Common-H1 CUDA labels, residual P+D reference propagation, calibration
telemetry, six-direction injection, and same-binary Cosmos validation are implemented. Same-snapshot oracle coverage
remains insufficient. See `notes/208-inflight-m6-directional-completion-rls-shadow-20260901.md` and
`notes/209-inflight-m6-hierarchical-direction-shrinkage-20260901.md`.

- direction-specific model을 shadow로 학습
- mean/uncertainty calibration, action ranking regret, false-safe rate 측정
- scheduler decision latency p95 측정
- Gate C 통과 뒤 active mode 허용

### M7. Active learned selector promotion

- 12-workload 전부 반복
- 39/48.8/97.5 req/s saturation points 반복
- SLO goodput, TTFT/TPOT/E2E mean/p95, throughput, peak VRAM, exact output 비교
- frozen vLLM은 contract가 같을 때 재사용하고 contract 변경 시만 fresh 실행

### M8. Optional H2 formation extension

H1에서 실제 regret이 남고 natural workload에서 formation-aware action change 기회가 충분할 때만 추가한다.

- arbitrary future arrival 예측 금지
- current immutable ready work와 이미 outstanding인 completion만 사용
- action change frequency와 SLO goodput gain이 없으면 characterization tool로 유지

## 14. Evaluation gates

### Gate A: in-flight effect

Directional injection timing이 incumbent/newcomer completion 또는 action ranking을 유의미하게 바꾸어야 한다. 변화가 거의 없다면 in-flight predictor를 단순화한다.

### Gate B: oracle objective

동일 runtime/kernels에서 Oracle H1이 current myopic보다 다음을 만족해야 한다.

- correctness와 token identity 유지
- 12-workload 핵심 latency/throughput gate에 구조적 회귀 없음
- saturation SLO goodput 개선 또는 동률
- planned/actual action fidelity 100%

Oracle도 개선하지 못하면 learned predictor를 구현하기 전에 objective/projector를 수정한다.

### Gate C: learned predictor

- completion error와 uncertainty가 calibration됨
- oracle action ranking과 충분히 일치
- SLO-unsafe action을 safe로 판단하는 false-safe rate가 낮음
- scheduler decision latency가 GPU action에 비해 무시 가능한 수준
- exact-key cold-state 없이 continuous feature interpolation이 작동

## 15. 평가 구성

### 15.1 Causal comparison

가장 중요한 비교는 같은 binary, engine, kernels, trace에서 policy만 바꾸는 것이다.

- serial/static policy
- always-overlap characterization
- legacy-compatible/current myopic
- current H2 research mode
- Oracle incremental H1
- learned incremental H1
- optional learned H1+H2

### 15.2 기존 12 workloads

기존 역할을 유지한다.

- short: tail overhead
- balanced: normal continuous batching
- decode-heavy: D refill/cohort
- long-prefill: packed-P formation
- bimodal: request ordering/fairness
- text-heavy: text/VLM coexistence
- mixed: integrated behavior
- vision-heavy: E pressure
- Poisson mixed: online arrival
- wave/drain: E critical path/starvation
- multi-image: small vision cohort
- late-vision D24: overlap placement control

모든 workload에서 throughput뿐 아니라 다음을 기록한다.

- E2E mean/p95
- TTFT mean/p95
- TPOT mean/p95
- SLO goodput와 failure attribution
- phase dispatch count와 batch distribution
- decision-to-enqueue 및 completion-to-next-action gap
- E/P/D/Copy activity mask와 overlap duration
- peak VRAM, KV/vision/workspace occupancy
- exact greedy output hash

### 15.3 vLLM 비교의 위치

vLLM 비교는 end-to-end competitiveness를 보여주는 secondary comparison이다. scheduler의 causal 효과는 같은 TensorRT runtime 안의 policy-only A/B로 먼저 증명한다. trace, HTTP contract, precision, output contract가 동일하면 frozen vLLM 결과를 재사용하고, 계약이 바뀌면 fresh run을 수행한다.

## 16. 논문 구조와 표현

권장 중심 용어는 다음과 같다.

- **Compiled Phase-Actor Substrate**
- **Stable Cross-Phase Ownership**
- **In-flight-aware Incremental Dispatch**
- **Completion-vector Prediction and Milestone Projection**

다음 표현은 피한다.

- `global scheduler 자체가 새롭다`
- `CUDA가 실행 중인 정확한 progress를 항상 관측한다`
- `독립 context는 항상 완전히 별도 workspace를 쓴다`
- `online oracle이 선택하지 않은 action의 실제 결과를 안다`
- `stable ownership 자체가 paged KV보다 새로운 allocator다`

논문의 policy 원칙은 다음 네 줄로 요약한다.

```text
Legality
    -> Request-level SLO safety
    -> Useful milestone progress
    -> GPU and ownership efficiency
```

## 17. 최종 결정

이 구조 변경은 진행한다. 최근 H2 결과는 formation 현상의 부재가 아니라, whole-horizon prediction과 실제 event-driven runtime semantics의 mismatch를 보여준다. In-flight-aware incremental dispatch는 그 mismatch를 직접 모델링하며, 현재 구현의 independent contexts, execution lease, realized attribution, stable ownership을 버리지 않고 역할을 명확히 재배치한다.

초기 성공 기준은 learned model이 아니다. 먼저 replay/oracle completion vector와 deterministic projector로 같은 runtime에서 더 좋은 H1 action을 선택할 수 있음을 보여야 한다. 그 gate를 통과한 뒤에만 online RLS predictor를 active policy로 승격한다.
