# Current-only 코드 정리와 추가 성능 최적화 계획

## 결론

현재 production 경로는 `fixed P128 + P8/D64 + stable indexed-paged KV + independent E/P/D contexts + profile-free Global scheduler`로 확정한다. 앞으로 workload별 profile을 다시 추가하지 않는다.

2026-08-28 실행 결과, rejected horizon 제거와 Global active hot-path의 Legacy policy 평가 생략,
production fixed-P128 wiring은 전체 성능 gate를 통과해 승격했다. 반면 batch-former 함수/clone 구조 변경,
inactive controller wiring 삭제, direct event callback은 일부 workload에서 지속적 회귀가 발생해
되돌렸다. 최종 12 workload x 3 및 의심 5 workload x 3 결과는
`notes/169-current-only-stage-gates-20260828.md`에 기록했다. 따라서 아래 단계는 완료 목록이 아니라,
성능 gate에 따라 일부만 승격된 설계 로드맵으로 읽어야 한다.

코드 정리는 바로 Legacy 분기를 삭제하는 방식으로 시작하면 안 된다. 현재 Global candidate 생성은 `PhaseQueueScheduler` 전체를 복사한 뒤 `globalSchedulerMode=disabled`로 바꿔 Legacy mechanism을 재사용한다. 따라서 Legacy policy와 batch formation mechanism이 아직 구조적으로 결합되어 있다.

안전한 순서는 다음과 같다.

```text
Current baseline 고정
        -> deterministic batch former 추출
        -> scheduler full-copy preview 제거
        -> Global-only production scheduler 전환
        -> Legacy/profile/negative experiment 삭제
        -> host allocation/metadata/token transport 최적화
        -> 12-workload gate
```

정리 자체의 목적은 LOC 감소만이 아니다. Global hot path가 더 이상 Legacy decision, scheduler clone, 사용하지 않는 adaptive/profile branch를 실행하지 않게 하여 host submission gap과 작은 batch timing 변동을 줄이는 것이 핵심이다.

## 현재 기준선

### 코드와 결과

- v0.10 worktree: `.local/upstream-v010`
- production runtime commit: `aa339ab`
- 결과 문서 commit: `04dc268`
- root benchmark/transport commit: `6c738d6`
- Current 결과: `.local/balanced-vllm-crossover-20260828/r53-production-regression-11x1`, `r54-production-regression-11x2`, `r59-final-balanced-production-3x`
- 최고 Legacy: `.local/profile-local-prefill-20260827/r1-legacy-disabled-12x3`
- vLLM: `.local/profile-free-global-20260827/r4-vllm-12x3`, C64 fresh 결과 `r39`, `r58`

### Current contract

- Cosmos Reason2-2B tied engine
- text/VLM 동일 runtime
- independent TensorRT E/P/D contexts, CUDA context 공유
- P8, D64, E4
- stable slots 80, decode-aligned effective admission 64
- fixed prefill chunk 128
- packed text prefill
- dynamic decode batching, stable decode cohort, online decode cost learning
- profile-free Global phase-action scheduling
- E/P/D, E+P, E+D, P+D와 bounded WAIT
- native completion callback
- external per-dispatch JSON은 profiling opt-in
- CUDA graph OFF
- prefix reuse OFF
- adaptive chunk/adaptive admission/memory-broker policy OFF

### 보존해야 할 성능

12-workload 기하평균에서 Current는 최고 Legacy 대비 throughput `+1.77%`, TTFT mean `-29.34%`, TTFT p95 `-18.59%`, E2E mean `-7.49%`, E2E p95 `-8.99%`다. 반면 TPOT mean/median은 각각 약 `+6.94%/+7.20%` 느리고 TPOT p95는 사실상 동일하다.

따라서 cleanup 이후 목표는 다음이다.

- balanced throughput `>= 4,517 tok/s`—현재 4,562.68의 1% 이내
- 모든 workload throughput 회귀 `<= 1%`
- balanced는 vLLM 4,333.52 tok/s보다 높게 유지
- exact token hash 유지
- TTFT/TPOT/E2E p95 회귀 `<= 3%`
- peak VRAM 증가 `<= 64 MiB`
- text-heavy/poisson TPOT는 Legacy 방향으로 개선

## 코드 복잡도 감사

핵심 파일 크기는 다음과 같다.

| 파일 | LOC |
|---|---:|
| `phaseQueueScheduler.cpp` | 3,748 |
| `phaseQueueScheduler.h` | 894 |
| `independentPhaseAsyncServer.cpp` | 2,110 |
| `independentPhaseAsyncServer.h` | 545 |
| `phaseThreeCoordinator.cpp` | 2,795 |
| `phaseThreeCoordinator.h` | 664 |
| `llm_phase_context_smoke.cpp` | 3,329 |
| `phaseQueueSchedulerTest.cpp` | 3,319 |

production smoke는 151개의 `TRT_EDGELLM_*` 환경 변수를 참조하지만 최종 command가 설정하는 runtime 변수는 29개다. 많은 변수는 실패한 A/B, Legacy profile, debug 또는 현재 비활성 기능을 위한 것이다.

핵심 문제는 단순 파일 크기보다 authority가 섞여 있다는 점이다.

```text
PhaseQueueScheduler::next()
    |
    +-- legacyQueueDecision()               # Active에서도 먼저 계산
    |
    +-- selectGlobalQueueAction()
          |
          +-- previewMechanismPlan()
          |      |
          |      +-- PhaseQueueScheduler 전체 복사
          |      +-- mode=Disabled
          |      +-- Legacy policy callback으로 P/D mechanism 실행
          |
          +-- Global feasibility/deadline/efficiency
```

최근 copy-on-write는 cost/history deep-copy를 줄였지만 queue, maps, cohort state와 candidate vectors의 scheduler 복제는 남아 있다. Legacy를 지우기 전에 이 dependency를 제거해야 한다.

## 반드시 유지할 Current mechanism

| 영역 | 유지할 기능 | 이유 |
|---|---|---|
| KV | stable slot/page lease, generation, append-only page ownership | compaction 없는 correctness 기반 |
| P | fixed-128, packed/ragged carrier, text/external compatibility | 현재 최적 engine contract와 semantic correctness |
| D | dynamic batch, stable cohort, completion-event WAIT, online context-bucket cost | balanced/decode-heavy 성능 핵심 |
| E | E4 batching, media/shape compatibility, async preparation | VLM production 경로 |
| Global | feasibility -> protected slack -> efficiency/lifetime | workload-label-free policy authority |
| 실행 | explicit outstanding-set lease, independent contexts, CUDA events | action fidelity와 overlap correctness |
| lifetime | vision payload/KV ownership 및 completion 후 release | memory correctness |
| transport | native callback, ready-token SSE coalescing | 현재 host-path 개선 |
| 계측 | 내부 CUDA-event learning, opt-in external metrics | online scheduler와 regression 분석 |

## 제거 후보 분류

### A. Current에서 OFF이고 실제 gate가 부정적이었던 기능

첫 cleanup 대상으로 적합하다.

| 기능 | 코드/설정 | 근거 |
|---|---|---|
| prefill continuation horizon | `enableGlobalPrefillContinuationHorizon`, `TRT_EDGELLM_GLOBAL_PREFILL_CONTINUATION_HORIZON` | negative gate, production OFF |
| incremental full decode-drain horizon | `enableGlobalIncrementalDecodeDrainHorizon`, 대응 env | 긴 horizon이 cohort를 과대평가, OFF |
| encoder queue horizon | `enableGlobalEncoderQueueHorizon`, 대응 env | vision-heavy 소폭 이득이나 wave tail 악화, OFF |
| phase-normalized ingress | 이미 revert | wave 개선 없음, multi-image 악화 |
| vision TTFT 350 override | benchmark-only rejected | hash 불안정, 500 유지 |
| direct CUDA graph priming/variant experiment | 이미 production patch에서 제거 | eager보다 느림 |

이 기능들의 config field, telemetry counter, preview helper, env parsing과 전용 unit test를 함께 제거한다. Current 결과가 바뀌면 안 된다.

### B. Mechanism 추출 후 삭제할 Legacy policy

바로 삭제하면 Global preview가 깨지므로 두 번째 단계다.

- `PhaseGlobalSchedulerMode::{kDisabled,kShadow}`
- `PhaseGlobalSelectionMode::kLegacyCompatibility`
- `PhaseSchedulerProfile::{kLatencySafe,kBalanced,kThroughputBalanced,kLongPrefill,kAuto}`
- `applySchedulerProfile()`
- production hot path의 `legacyQueueDecision()`
- Legacy phase debt/metrics policy authority
- shadow disagreement/parity telemetry
- Legacy prefill-formation timer와 external profile selection
- adaptive/stepwise admission controller
- Legacy decode refill 결과를 최종 action으로 쓰는 분기

단, Legacy가 제공하던 deterministic batch compatibility와 canonical ordering은 삭제하지 않고 새 batch former로 옮긴다.

### C. Fixed-P128 specialization 뒤 삭제할 prefill 실험 코드

Current는 adaptive chunk를 사용하지 않는다.

- 32/64/128 adaptive chunk candidate 선택
- queue-pressure chunk shrink
- completion split opt-in
- joint `(batch, chunk)` cost-aware search
- long-prefill profile preset
- chunk별 offline overlap shape 탐색

다음은 유지한다.

- final tail은 실제 남은 token 수만 실행
- initial/continuation class 구분
- packed useful-token carrier
- atomic multimodal P와 text/external compatibility
- fixed-128 wavefront cohort mechanism

### D. Current-only production binary에서 분리할 실험·debug 기능

- graph capture/cache 환경 변수
- prefix reuse 실험 경로
- tiered E8 vision execution
- adaptive encoded-capacity hysteresis
- homogeneous encoder batching experiment
- vision tensor dump와 M-RoPE copy microprofile
- offline external admission profile
- phase memory drain/broker policy
- dedicated/shared/class-specific P-context A/B switches

삭제 전에 `llm_phase_context_lab` 같은 별도 binary로 한 번 격리할 수 있다. Current-only binary의 결과가 고정되면 lab target과 코드는 archive branch의 Git history에만 남기고 삭제한다.

### E. 유지 여부를 재평가할 코드

#### Persistent decode select/page bindings

과거에는 metadata operation을 크게 줄였지만 dispatch timing이 빨라져 작은 batch가 증가해 기본 OFF였다. 지금은 full P cohort ingress와 decode-aligned admission으로 formation이 안정됐으므로 다시 A/B할 가치가 있다.

- persistent decode select: decode memset 1,694회 -> 1회였던 이력
- persistent page binding: active-row materialization의 약 70% 재사용 이력
- 이전 balanced에서 select는 약 +1%였지만 short/decode에서 불안정

새 gate에서는 action/batch trace가 ON/OFF 사이에서 같을 때만 kernel/metadata 효과를 비교한다. schedule이 달라지면 metadata 최적화가 아니라 formation 문제로 분류한다.

#### CUDA graph

현재 최종 graph A/B는 이득이 없었다. 코드 정리의 첫 성능 목표로 삼지 않는다. host scheduler와 shape stability가 고정된 뒤 exact hit bucket만 다시 검토한다.

## 새 Current-only 구조

### 1. Mechanism과 policy 분리

```text
ReadyState
   |
   v
PhaseBatchFormer                 # deterministic, policy 없음
   |-- formPrefill(P128, P8)
   |-- formDecode(D<=64)
   |-- formEncoder(E<=4)
   `-- compatibility/canonical ordering
   |
   v
CandidateSet                     # bounded views, 아직 queue mutation 없음
   |
   v
GlobalPhaseActionScheduler       # feasibility/slack/cost only
   |
   v
SelectedPlan
   |
   v
PhasePlanMaterializer            # stable lease + queue mutation 1회
   |
   v
E/P/D dispatch workers
```

Global preview는 scheduler clone 대신 immutable candidate view를 받는다. 후보를 선택한 뒤에만 queue와 in-flight ownership을 변경한다.

### 2. Production config 축소

현재 151개 env parser 대신 production API는 다음 세 종류만 받는다.

```text
Engine limits:
  max P/D/E, chunk/page/sequence capacity

Serving limits:
  slots, admission memory budget, adapter workers

Request/SLO defaults:
  TTFT, TPOT, output contract
```

P8/D64/E4/128은 가능하면 engine manifest에서 읽고, request-specific SLO는 request metadata로 전달한다. `global active`는 모드가 아니라 유일한 production scheduler가 된다.

### 3. Model/GPU별 static table 축소

현재 smoke에 Cosmos/RTX 3080 decode cost prior가 직접 들어 있다. 이를 다음처럼 바꾼다.

- engine/host manifest에 cold-start prior 저장
- unmeasured shape는 conservative interpolation
- controlled warmup CUDA event가 충분하면 online estimate가 authority
- encoder JSON도 동일 cost-store interface로 통합
- workload name/profile은 key에 포함하지 않음

## 추가 성능 향상 포인트

### 우선순위 1 — scheduler clone과 중복 Legacy decision 제거

현재 `next()`는 Active에서도 Legacy decision을 먼저 계산하고, Global P/D candidate마다 scheduler copy preview를 수행한다. 가장 먼저 제거할 host overhead다.

측정할 지표:

- `scheduler_decision_us` median/p95/max
- decision당 heap allocation 수/bytes
- candidate당 queue scan 수
- CUDA kernel 사이 host submission gap
- P/D dispatch count와 batch shape identity

성공 조건은 GPU batch sequence가 같으면서 host 시간이 줄어드는 것이다.

### 우선순위 2 — candidate storage와 queue scan 재사용

현재 여러 경로에서 `vector`, `unordered_set`, `find_if`, stable sort와 request-ID 재탐색을 반복한다.

- queue entry에 stable handle/index 부여
- snapshot epoch 동안 canonical P/D view 캐시
- request ID -> queue iterator/index table 유지
- 후보 request ID/slot ID storage를 bounded reusable arena로 변경
- D completion preview의 누적 vectors를 preallocated buffer로 변경
- candidate top-k를 생성 단계에서 제한

정확한 snapshot epoch가 바뀌면 view를 무효화한다. stale pointer를 허용하지 않는다.

### 우선순위 3 — Legacy보다 느린 D continuity 회복

Current는 전체 평균 TTFT/E2E가 좋지만 text-heavy와 poisson의 TPOT가 Legacy보다 느리다. workload label 대신 다음 action value를 추가한다.

```text
D continuation value
  = same cohort를 한 turn 더 유지해 절약하는
    row replacement + launch + future drain cost
```

- oldest first-token slack이 충분하고
- D cohort가 dense하며
- 다음 P/E가 즉시 resource를 release하지 않고
- D continuation이 TPOT tail을 보호하면

한 번의 D lease를 연속 유지한다. 고정 `text-heavy` 분기나 fixed D burst count는 추가하지 않는다.

우선 gate는 text-heavy, poisson, mixed, vision-heavy이며 balanced 결과를 함께 보호한다.

### 우선순위 4 — metadata cache 재평가

formation schedule을 먼저 고정한 뒤 persistent select/page binding을 다시 켠다. scheduler가 dispatch boundary를 보존하면 과거의 local metadata 절감이 E2E 이득으로 전환될 가능성이 있다.

### 우선순위 5 — token span callback과 pinned ring

현재 HTTP는 ready token을 coalesce하지만 runtime 내부에서는 여전히 token event 하나씩 생성할 수 있다.

- sampling completion이 같은 request의 ready token span을 전달
- bounded SPSC/MPSC ring 또는 preallocated event pool 사용
- token text/token ID storage의 per-token allocation 제거
- terminal event ordering은 sequence ID로 보존

TTFT는 첫 token 즉시 전달을 유지하고, 이미 완료된 후속 token만 묶는다.

### 우선순위 6 — page metadata delta update

Stable indexed-paged ownership은 유지한다. 매 dispatch 전체 row/page view를 재구성하는 대신 lease generation과 page-count delta만 업데이트한다. persistent binding A/B가 schedule-neutral일 때만 기본값으로 승격한다.

### 우선순위 7 — graph는 마지막

Current의 batch formation과 host path가 고정된 뒤 D32/D64처럼 반복되는 exact shape만 capture한다. unseen shape capture는 request path에서 하지 않는다. 이전 graph 실험이 eager보다 느렸으므로 이 단계 전에는 다시 활성화하지 않는다.

## 삭제 및 구현 단계

### Stage 0 — 복구점과 golden replay

1. 현재 commit을 local archive tag/branch로 보존한다.
2. 12 workload의 trace SHA, token hash, action/batch sequence와 metric을 manifest로 고정한다.
3. scheduler snapshot replay microbenchmark를 추가한다.
4. Git history 밖의 engine/result는 삭제하지 않는다.

### Stage 1 — rejected opt-in 제거

- P continuation horizon
- incremental D full-drain horizon
- E queue horizon
- 관련 env/config/telemetry/tests

이 단계는 production action sequence가 byte-for-byte 동일해야 한다.

### Stage 2 — `PhaseBatchFormer` 추출

- P/D/E deterministic formation을 policy와 분리
- immutable candidate view
- one-shot materialization
- full scheduler preview clone 제거
- canonical row ordering 및 stable lease invariant 유지

### Stage 3 — Global-only 전환

- disabled/shadow/legacy compatibility mode 제거
- Legacy profiles 및 policy 제거
- `legacyQueueDecision()` hot-path 호출 제거
- production config와 telemetry를 Current 기준으로 축소

### Stage 4 — fixed-P128 specialization

- adaptive chunk/joint shape code 제거
- fixed P128 + final tail + atomic external P만 유지
- P formation cost key 단순화

### Stage 5 — server/coordinator 실험 controller 정리

- adaptive/stepwise/external-profile admission 제거
- 비활성 memory drain, tiered E8, adaptive vision policy 제거
- 현재 E4/ownership/global action 경로만 유지

### Stage 6 — 성능 최적화

- reusable candidate arena
- stable queue handles
- metadata cache A/B
- token span callback
- D continuation value

### Stage 7 — 전체 gate

각 Stage마다 다음 순서로 검증한다.

```text
unit/build
  -> scheduler replay identity
  -> short + balanced + text-heavy + poisson
  -> mixed + wave + multi
  -> 전체 12 workload x3
  -> cached vLLM comparison
```

trace/HTTP contract와 vLLM 설정이 바뀌지 않으면 vLLM은 매 단계나 최종 cleanup stage에서 다시 실행하지
않고 trace SHA가 검증된 fresh 3회 결과를 재사용한다. model, trace, arrival/output 계약 또는 vLLM 설정이
바뀌는 architecture milestone에서만 fresh 전체 sweep을 실행한다.

## 삭제하면 안 되는 것

- stable indexed-paged allocator와 page lease generation
- text/external P compatibility
- M-RoPE/vision embedding placement correctness
- exact action outstanding-set invariant
- per-context inflight <= 1 invariant
- GPU completion 뒤 ownership reclaim invariant
- canonical row ordering
- request-level TTFT/TPOT hints
- online CUDA cost observations
- native completion cleanup order

## 기대 결과와 판단 기준

코드 규모 목표는 위 핵심 scheduling/server/smoke 약 1.77만 LOC의 25~35%를 제거하거나 production target 밖으로 이동하는 것이다. 실제 성능 향상 폭은 미리 가정하지 않고 scheduler microbenchmark와 CUDA submission gap으로 측정한다.

가장 가능성이 높은 성능 개선은 다음 두 가지다.

1. full scheduler clone/중복 Legacy decision 제거로 host gap 감소
2. D continuation value로 text-heavy/poisson의 TPOT와 throughput 회복

코드 삭제만으로 GPU kernel 시간이 줄지는 않는다. cleanup이 성능으로 연결되려면 Current와 동일한 batch를 더 적은 host work로 만들거나, 같은 SLO 안에서 더 dense한 D cohort를 형성해야 한다.

최종 목표는 다음이다.

```text
하나의 Current production path
  + deterministic batch formation
  + profile-free Global action selection
  + stable ownership
  + independent E/P/D execution
  + bounded online cost learning
```

Legacy와 실패한 실험은 별도 runtime mode로 유지하지 않고 Git history와 결과 문서로만 보존한다.
