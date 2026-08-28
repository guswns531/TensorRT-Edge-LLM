# v0.10.0 Baseline 대비 Current 프로젝트 구조 리팩터링

## 목적

현재 기능과 성능을 유지하면서 다음 세 영역을 코드 구조만 보고도 구분할 수 있게 한다.

1. NVIDIA v0.10.0 upstream 원형
2. indexed KV와 independent E/P/D를 위해 필요한 최소 integration hook
3. Current phase runtime, production server, benchmark와 연구용 실험

리팩터링은 workload별 tuning을 추가하지 않는다. `fixed P128 + P8/D64/E4 + stable indexed-paged KV +
independent E/P/D contexts + profile-free Global scheduler`라는 현재 production contract를 유지한다.

## Git 기준 변화 규모

비교 기준은 tag `v0.10.0`, Current는 branch `codex/v010-phase-forward-port`의 `b2a01f0`이다.

| 항목 | 수치 |
|---|---:|
| 변경 파일 | 124 |
| 새 파일 | 49 |
| 기존 v0.10.0 수정 파일 | 75 |
| 전체 추가/삭제 | +30,109 / -404 |
| 새 `cpp/runtime/scheduling` | 31 files, 16,118 LOC |
| 새 stable KV state | 2 files, 370 LOC |
| 새 C++ unit tests | 8 files, 4,642 LOC |

현재 scheduler 구현은 기능별 디렉터리 없이 한 폴더에 평평하게 있고, 다음 세 파일만 합쳐도
`9,612 LOC`다.

| 파일 | LOC |
|---|---:|
| `phaseQueueScheduler.cpp` | 3,611 |
| `phaseThreeCoordinator.cpp` | 2,706 |
| `independentPhaseAsyncServer.cpp` | 2,110 |
| `llm_phase_context_smoke.cpp` | 3,295 |

## v0.10.0 관련 트리

아래는 이번 기능과 관련된 부분만 나타낸 축약 트리다.

```text
TensorRT-Edge-LLM@v0.10.0
├── cpp
│   ├── builder
│   ├── common
│   ├── kernels
│   │   ├── contextAttentionKernels
│   │   ├── embeddingKernels
│   │   └── posEncoding
│   ├── multimodal
│   ├── plugins
│   │   └── attentionPlugin
│   └── runtime
│       ├── config
│       ├── decoding
│       ├── exec
│       ├── preprocess
│       ├── state
│       │   ├── kvPageTable.*
│       │   ├── pipelineIO.*
│       │   └── sharedResources.*
│       ├── kvCacheManager.*
│       ├── llmInferenceRuntime.*
│       └── llmRuntimeUtils.*
├── examples
│   ├── llm
│   │   ├── llm_build.cpp
│   │   ├── llm_bench.cpp
│   │   ├── llm_inference.cpp
│   │   └── llm_stream.cpp
│   └── multimodal
├── tensorrt_edgellm
│   ├── models
│   ├── onnx
│   └── scripts
├── tests
└── unittests
```

v0.10.0에는 독립 E/P/D execution context, phase queue/global scheduler, stable KV page ownership,
production async phase server가 없다.

## 현재 실제 트리와 v0.10.0 차이

표시는 `[A]`가 Current에서 추가, `[M]`이 v0.10.0 파일 수정이다.

```text
TensorRT-Edge-LLM@Current
├── cpp
│   ├── builder/                              [M] indexed/phase engine contract
│   ├── common/bindingNames.h                 [M] shared phase bindings
│   ├── kernels                               [M]
│   │   ├── contextAttentionKernels           [M] indexed KV read/gather
│   │   ├── embeddingKernels                  [M] packed/external embedding
│   │   └── posEncoding                       [M] slot-aware RoPE/KV write
│   ├── multimodal                            [M] asynchronous vision execution
│   ├── plugins/attentionPlugin               [M] indexed KV plugin contract
│   └── runtime
│       ├── config                             [M] engine/phase dimensions
│       ├── decoding                           [M] indexed decode integration
│       ├── exec                               [M] independent context bindings
│       ├── preprocess                         [M] packed/vision P preparation
│       ├── state                              [M+A]
│       │   ├── kvPageTable.*                  [M]
│       │   ├── pipelineIO.*                   [M]
│       │   ├── sharedResources.*              [M]
│       │   └── stableKVPageManager.*          [A]
│       ├── scheduling                         [A, 31 files, 16,118 LOC]
│       │   ├── independentEngineExecutorPair.*
│       │   ├── independentPhaseAsyncServer.*
│       │   ├── independentPhaseCoordinator.*
│       │   ├── modelPhaseContract.h
│       │   ├── packedPrefillActiveView.*
│       │   ├── phaseAsyncServer.h
│       │   ├── phaseContinuousLoadGenerator.*
│       │   ├── phaseDispatchWorker.*
│       │   ├── phaseGlobalScheduler.*
│       │   ├── phaseKernelGroupRecorder.*
│       │   ├── phaseKVActiveView.*
│       │   ├── phaseMemoryBroker.*
│       │   ├── phasePrefixReuseCache.*
│       │   ├── phaseQueueScheduler.*
│       │   ├── phaseThreeCoordinator.*
│       │   ├── phaseTimeline.h
│       │   └── phaseVisionAdapter.*
│       ├── kvCacheManager.*                   [M]
│       ├── llmInferenceRuntime.*              [M]
│       └── llmRuntimeUtils.*                  [M]
├── examples
│   ├── llm
│   │   ├── llm_phase_context_smoke.cpp        [A, 3,295 LOC]
│   │   ├── llm_build.cpp                      [M]
│   │   └── llm_bench.cpp                      [M]
│   └── multimodal/visual_build.cpp            [M]
├── scripts/phase_openai_gateway.py            [A]
├── tensorrt_edgellm                           [M] export/custom-op/indexed inputs
├── unittests                                  [M+A, phase/stable-KV suites]
├── docs                                       [A] public design
└── notes                                      [A] experiment and gate history
```

이 구조의 장점은 새 runtime 코드가 `scheduling/`에 모여 있다는 점이다. 단점은 그 내부에서 API,
batch mechanism, policy, CUDA execution, ownership, VLM adapter, telemetry와 benchmark support가 다시
섞여 있다는 점이다. 또한 `llm_phase_context_smoke.cpp`가 production server composition root와 실험용
환경 변수 parser를 동시에 담당한다.

## 목표 트리

새 top-level 제품을 만들기보다 upstream runtime 아래에 `phase/`라는 명확한 bounded context를 둔다.
기존 v0.10.0 파일은 좁은 integration interface만 포함하도록 한다.

```text
TensorRT-Edge-LLM@Refactored-Current
├── cpp
│   ├── builder                               # v0.10 + 최소 phase/indexed hook
│   ├── kernels                               # slot-aware primitive만 유지
│   ├── plugins                               # serialized indexed contract
│   ├── multimodal                            # generic visual engine runner
│   └── runtime
│       ├── config                             # engine capability, policy 없음
│       ├── exec                               # TensorRT context primitive
│       ├── preprocess                         # P/D/E input preparation primitive
│       ├── state
│       │   ├── kvPageTable.*                  # upstream page view
│       │   └── stableKVPageManager.*          # stable lease ownership
│       ├── phase                              # Current의 유일한 신규 runtime subsystem
│       │   ├── api
│       │   │   ├── phaseAsyncServer.h
│       │   │   ├── phaseRequest.h
│       │   │   └── phaseServerConfig.h
│       │   ├── mechanism
│       │   │   ├── readySnapshot.*
│       │   │   ├── phaseBatchFormer.*
│       │   │   ├── phasePlanMaterializer.*
│       │   │   ├── packedPrefillView.*
│       │   │   └── decodeKVView.*
│       │   ├── policy
│       │   │   ├── globalPhaseActionScheduler.*
│       │   │   ├── deadlineProtection.*
│       │   │   ├── waitValue.*
│       │   │   └── onlinePhaseCost.*
│       │   ├── execution
│       │   │   ├── independentContexts.*
│       │   │   ├── phaseDispatchWorker.*
│       │   │   └── outstandingActionLease.*
│       │   ├── coordination
│       │   │   ├── requestDag.*
│       │   │   ├── phaseCoordinator.*
│       │   │   └── completionRouter.*
│       │   ├── ownership
│       │   │   ├── visionLease.*
│       │   │   ├── phaseMemoryView.*
│       │   │   └── prefixLease.*              # production 승격 전 optional
│       │   ├── multimodal
│       │   │   └── visionPhaseAdapter.*
│       │   └── telemetry
│       │       ├── phaseTimeline.h
│       │       └── kernelGroupRecorder.*
│       ├── kvCacheManager.*                   # narrow stable-owner hook
│       └── llmInferenceRuntime.*              # optional phase server factory only
├── apps
│   └── phase_server
│       ├── main.cpp                           # 3,295-line smoke 대체
│       ├── phaseServerOptions.*
│       └── phaseServerComposition.*
├── benchmarks
│   └── phase_serving
│       ├── trace_client
│       ├── workload_manifest
│       ├── v010_baseline_manifest
│       └── result_comparator
├── tests
│   ├── phase                           # Python/HTTP integration
│   └── baseline_diff                   # trace/hash/contract validation
├── unittests
│   └── phase                           # C++ mechanism/policy/ownership
├── docs/source/developer_guide/software-design/phase-runtime.md
└── notes                               # 시간순 연구 로그, product build 입력 아님
```

## 경계 규칙

### v0.10.0 수정 파일

기존 upstream 파일은 다음 세 종류의 hook만 가져야 한다.

1. `kv_slot_ids`와 stable page binding 같은 engine/runtime capability
2. independent TensorRT context가 사용할 input/output primitive
3. phase runtime을 생성하는 좁은 factory 또는 adapter

queue policy, SLO 계산, workload 분류, wait timer와 benchmark parser가 upstream 파일로 역류하면 안 된다.

### `phase/mechanism`

같은 immutable ready snapshot에 대해 항상 같은 row, slot, chunk와 batch shape를 반환한다. CUDA event가
완료되거나 admission epoch가 바뀌기 전까지 host poll 속도가 결과를 바꾸면 안 된다.

### `phase/policy`

mechanism이 만든 bounded candidate만 평가한다. request/trace 이름이나 text-heavy, VLM-heavy 같은
workload label을 입력으로 받지 않는다.

### production과 lab

production OFF인 adaptive chunk, prefix reuse, graph probe, memory broker experiment는 production binary의
환경 변수로 남기지 않는다. 아직 연구가 필요한 기능은 `benchmarks/phase_serving`의 별도 lab target에서
검증하고 gate를 통과한 뒤 runtime module로 승격한다.

## 안전한 리팩터링 순서

### R0 — Baseline diff manifest

- tag `v0.10.0`, Current commit, engine/model/trace SHA를 machine-readable manifest로 기록
- 75개 upstream 수정 파일을 `indexed-kv`, `phase-context`, `packed-prefill`, `vision`, `tied-weight`로 분류
- 각 변경 파일이 어느 feature contract 때문에 필요한지 owner 표 작성
- runtime binary와 workload 결과는 바꾸지 않음

### R1 — Build ordering 고정

현재 `cpp/CMakeLists.txt`는 `runtime/*.cpp`를 `GLOB_RECURSE`로 수집한다. 파일 이동만으로 archive object
순서가 달라질 수 있으므로 phase source의 명시적 목록과 안정된 object ordering을 먼저 도입한다.
변경 전후 binary symbol/object map과 12-workload action/batch trace를 비교한다.

### R2 — Composition root 분리

`llm_phase_context_smoke.cpp`에서 다음을 순서대로 분리한다.

1. pure option parsing
2. engine/runtime construction
3. production request adapter
4. debug/profile-only wiring

처음에는 기존 executable 이름과 command contract를 유지한다. 새 `apps/phase_server` target은 결과가
같아진 뒤 이름을 승격한다.

### R3 — Immutable mechanism boundary

- `ReadySnapshot`
- `PhaseBatchFormer`
- `SelectedPlan`
- `PhasePlanMaterializer`

를 도입한다. `PhaseQueueScheduler` 전체 clone과 Legacy mechanism callback은 이 단계가 action/batch replay
identity를 통과한 뒤 제거한다.

### R4 — Policy와 ownership 분리

`phaseGlobalScheduler`는 candidate score와 deadline protection만 소유한다. stable KV, vision payload와
completion event lifetime은 ownership 계층이 담당한다. policy가 raw CUDA/TensorRT pointer를 소유하지
않게 한다.

### R5 — Production/lab 분리

inactive parser와 negative A/B 구현을 lab target으로 옮긴다. production target의 최종 config는 engine
capability, serving capacity, request SLO 세 종류만 남긴다.

### R6 — 디렉터리 이동

논리 경계와 build ordering이 고정된 뒤 `scheduling/`의 파일을 목표 `phase/` 하위 디렉터리로 이동한다.
이 단계는 이름과 include path만 바꾸며 동작 변경을 섞지 않는다.

### R7 — 전체 승격 gate

각 단계는 다음을 모두 통과해야 한다.

```text
format/build/unit tests
    -> action and batch replay identity
    -> short/balanced/long-prefill/bimodal
    -> mixed/vision-heavy/wave/multi-image
    -> complete 12-workload x3
    -> throughput -1%, latency p95 +3%, memory +64 MiB gate
    -> unchanged trace: cached fresh vLLM comparison
```

## 먼저 진행할 실제 작업

첫 구현은 대규모 파일 이동이 아니다.

1. v0.10.0 변경 파일 feature-owner manifest 생성
2. phase source/object ordering manifest 생성
3. `llm_phase_context_smoke.cpp`의 option parsing을 동작 변화 없는 구조체로 추출
4. focused unit/build와 12-workload gate

이 네 단계가 통과하면 composition root가 작아지고, 다음 리팩터링부터 v0.10.0 integration hook과
Current phase runtime의 차이를 디렉터리와 Git diff 양쪽에서 명확하게 볼 수 있다.

## R0 실행 결과

동작을 바꾸지 않는 두 manifest를 추가했다.

- `benchmarks/phase_serving/manifests/v010_feature_owners.json`
- `benchmarks/phase_serving/manifests/phase_source_order.json`

첫 manifest는 v0.10.0 수정 지점을 indexed KV, packed prefill, independent phase execution, Global
scheduling, async server, vision, ownership, telemetry, tied external weights와 build/CLI로 분류한다. 하나의
파일이 둘 이상의 contract에 속할 수 있으며, 이후 PR 분리 시 cross-feature coupling을 명시적으로
드러낸다.

두 번째 manifest는 golden binary의 `edgellmCore` link에서 관측한 phase/stable-KV object 15개의 순서와
source SHA-256을 고정한다. R0는 runtime, CMake와 executable 파일을 변경하지 않으므로 workload 성능
재측정 대상이 아니다. 다음 R1에서 source collection을 명시적으로 바꿀 때 이 순서와 binary map을
보존하고 전체 gate를 실행한다.

## R1 실행 결과

`cpp/CMakeLists.txt`의 재귀 glob에서 phase runtime 14개와 stable KV page manager를 분리하고, 기존
archive 위치에 명시적으로 다시 삽입했다. 이후 디렉터리나 파일 이름을 바꾸더라도 이 목록을 함께
수정하지 않으면 object 순서가 조용히 달라지지 않는다.

### 빌드 산출물 동일성

R1 전후의 link command와 다음 산출물 SHA-256이 모두 같았다.

| 산출물 | R1 전후 SHA-256 |
|---|---|
| `libedgellmCore.a` | `eaca8c2f...22228` |
| `llm_phase_context_smoke` | `6d8a91b6...6fa32` |
| `libNvInfer_edgellm_plugin.so.1.0` | `b78c59b1...7a70` |

따라서 R1은 실행 코드, 심볼 배치와 plugin을 바꾸지 않았다. 집중 회귀 테스트도 phase queue/global
scheduler/three-phase coordinator/async server의 192개를 모두 통과했다.

### 실제 HTTP workload control

동일한 12개 trace를 1회씩 다시 실행했고 모든 요청이 완결됐으며 모든 token trace가 golden hash와
일치했다. 단일 반복에서 long-prefill은 처리량 `-1.80%`, TTFT p95 `+3.72%`, E2E p95 `+3.53%`였고,
bimodal은 TPOT p95 `+9.78%`였다. 두 workload를 3회 재측정한 결과는 다음과 같다.

| workload | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| long-prefill | -1.73% | +1.59% | +0.69% | +2.79% | +2.37% | +2.00% | +3.41% |
| bimodal | +1.63% | -1.76% | -9.21% | -1.44% | -0.05% | -1.85% | -3.76% |

long-prefill E2E p95가 수치상 `+3%` 경계를 `0.41%p` 넘었지만, 실행 바이너리가 바이트 단위로 같으므로
이 차이는 R1 코드가 유발할 수 없는 run-to-run 분산이다. R1 승격 근거는 binary identity, 192/192 unit
tests, 12/12 completion과 exact output identity이며, 측정 분산은 후속 단계의 판정 기준에서 숨기지 않고
그대로 유지한다. 입력 trace와 vLLM 실행 계약은 바뀌지 않았으므로 vLLM은 기존 fresh 3회 결과를
재사용한다.

## R2a/R2b 실행 결과: scheduler option parsing

첫 R2a는 scheduler 환경변수 파싱과 cost-table 로더를 새 `phaseServerOptions.cpp` translation unit으로
옮겼다. build와 192개 집중 테스트, 12개 exact output은 통과했지만 adjacent 3회 A/B에서 text-heavy
처리량이 `-1.12%`로 승격 gate를 넘었다. 별도 object와 out-of-line startup call이 executable layout을
바꾼 상태에서 얻는 구조적 이득보다 회귀 위험이 컸으므로 이 구현은 전부 되돌렸다.

R2b는 같은 문장을 `phaseSchedulerOptions.inc`로 옮기고 기존 translation unit 내부에서 include한다.
fragment는 statement-only이며 runtime API나 CMake link input을 추가하지 않는다. 이는 최종 public API가
아니라 성능에 민감한 composition root를 안전하게 분해하기 위한 중간 경계다.

민감한 세 workload의 R2b 대 직전 adjacent golden 3회 median은 다음과 같다.

| workload | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | +0.74% | -0.87% | -1.53% | -1.54% | -17.06% | -0.90% | -0.84% |
| long-prefill | +2.21% | -2.27% | -6.09% | -3.25% | -3.86% | -2.25% | -5.63% |
| text-heavy | -0.69% | -1.85% | +3.63% | +1.11% | -1.52% | +0.77% | +0.82% |

text-heavy의 전체 TTFT p95만 `+3.63%`였지만, class별 text와 vision TTFT p95 변화는 각각 약 `+0.7%`,
`+1.4%`였다. 전체 p95 순서통계 경계가 request class 사이에서 이동한 것이며 처리량, mean, TPOT, E2E와
각 class tail은 gate 안이다.

나머지 9개 single control에서 처리량 최악은 poisson `-0.52%`였고, 모든 latency p95는 gate 안이었다.
12개 workload 모두 completion, golden token hash와 peak-memory contract를 유지했다. trace와 HTTP 계약이
같으므로 vLLM fresh 결과는 다시 실행하지 않았다.

후속 R2c에서 async server/admission 옵션 140줄도 두 번째 statement fragment로 옮겼으나 승인하지 않았다.
민감 workload 3회 측정에서 R2b 대비 long-prefill 처리량 `-3.41%`, E2E p95 `+4.73%`가 나타났고, R2c를
되돌린 직후 R2b를 다시 측정한 adjacent control에서도 처리량 차이 `-3.44%`, E2E p95 `+3.65%`가
재현됐다. 따라서 이는 장기 환경 drift가 아니라 두 번째 include boundary가 만든 executable layout
민감성으로 판정했다. server/admission 옵션은 현재 composition root에 남기고, hot loop가 독립된 안정
object 경계를 가진 뒤 다시 분리한다.

## R3 실행 결과: immutable action boundary

기존 `phaseGlobalScheduler.h`의 서로 다른 책임을 선언 수준에서 세 경계로 나눴다.

- `phaseGlobalCostModel.h`: action key, execution variant, online CUDA-event cost observations
- `phaseActionPlan.h`: ownership horizon, protected completion, immutable candidate와 dispatch lease materialization
- `phaseGlobalScheduler.h`: feasibility/deadline/efficiency selector와 decision 결과

구현 함수와 object source order는 이동하지 않았다. R3 전후 `libedgellmCore.a`, TensorRT plugin,
`llm_phase_context_smoke` SHA-256이 모두 완전히 같았고, Global/queue scheduler 단위 테스트 145개도 모두
통과했다. 실행 binary가 같으므로 R2b의 12-workload 결과와 cached fresh vLLM 비교를 그대로 승계한다.

## R4 실행 결과: policy input과 ownership 분리

selector 입력의 lifetime 의미를 다음 세 value header로 분리했다.

- `phaseReadySnapshot.h`: poll 속도와 무관하게 한 decision epoch를 나타내는 read-only queue snapshot
- `phaseDeadline.h`: request별 protected completion slack, predicted completion과 uncertainty
- `phaseOwnershipHorizon.h`: managed/allocate/reclaim/growth/budget으로 표현한 ownership horizon

`phaseActionPlan.h`는 이 값들을 조합한 candidate와 authoritative dispatch lease만 가진다. 따라서
`PhaseGlobalScheduler`는 stable KV allocator, vision slab, CUDA event와 TensorRT context 구현을 직접
소유하거나 호출하지 않고 feasibility/deadline/efficiency 값만 평가한다.

R4 역시 source object와 구현 함수는 이동하지 않았다. core archive, plugin, server executable SHA-256이
R3와 같았고 관련 단위 테스트 145개를 모두 통과했다. binary identity가 성립하므로 별도 workload와
vLLM 재실행은 필요하지 않다.

## R5 실행 결과: production/runtime과 lab 경계 고정

실제 source를 검사한 결과 `cpp/runtime/scheduling`에는 `std::getenv` 또는 `TRT_EDGELLM_*` 파싱이 없다.
환경 변수 130여 개는 `llm_phase_context_smoke.cpp`와 `phaseSchedulerOptions.inc`의 composition boundary에만
남아 있으며, runtime policy에는 engine capability, serving capacity, request SLO, ready/completion state,
ownership horizon과 online cost observation이 이미 값으로 주입된다.

두 번째 server/admission 파싱 fragment를 물리적으로 추출하는 R2c가 long-prefill 처리량을 `3.44%`
낮춘 사실 때문에 같은 이동을 반복하지 않았다. 대신 다음 두 파일로 경계를 실행 가능한 contract로
고정했다.

- `phase_runtime_contract.json`: production runtime 입력, 금지된 workload label/env dependency,
  composition/lab 책임을 machine-readable하게 정의한다.
- `validate_runtime_boundaries.py`: runtime의 env/workload-label 의존성, example/benchmark 역참조와 v0.10.0
  product diff의 feature-owner 누락을 검사한다.

따라서 새 workload별 knob가 production scheduler로 유입되거나, 새 product 파일이 feature owner 없이
추가되면 검증이 즉시 실패한다. 이 단계는 runtime build input을 변경하지 않으므로 R4 binary와 성능을
그대로 승계한다. R2c에서 회귀한 server option 물리 추출은 hot loop를 별도 object로 안정화하기 전까지
명시적으로 보류한다.

## R6 실행 결과: canonical phase 디렉터리

R3/R4에서 분리한 선언을 책임별 canonical 경로로 이동했다.

```text
cpp/runtime/phase
├── mechanism
│   └── phaseReadySnapshot.h
├── policy
│   ├── phaseDeadline.h
│   ├── phaseGlobalCostModel.h
│   └── phaseGlobalScheduler.h
├── ownership
│   └── phaseOwnershipHorizon.h
└── execution
    └── phaseActionPlan.h
```

기존 `runtime/scheduling/*.h` 경로는 외부 include를 깨지 않는 forwarding header로 남겼다. canonical
`phaseActionPlan`은 policy/ownership value만 참조하고, canonical selector는 execution plan만 참조하므로
새 경로가 R3/R4의 의존 방향을 그대로 표현한다. `phaseQueueScheduler`도 canonical mechanism/policy header를
직접 사용한다.

반면 hot implementation `.cpp`는 이동하지 않았다. R1에서 object 순서를 고정했더라도 R2a/R2c가 보여 준
executable-layout 민감성을 고려하면 source path와 translation unit 이동은 별도 성능 변경이다. 이 파일들은
현재 명시적 source-order 목록에 남겨 binary 안정성을 우선한다. 호환 header는 후속 public API 전환 기간에
제거할 수 있지만 현재 v0.10.0 사용자 include를 깨지 않는다.

R6 build 결과 core archive, TensorRT plugin과 server executable의 SHA-256은 R4와 모두 동일했다. 독립
async server, Global scheduler/cost, queue scheduler와 three-phase policy를 포함한 집중 테스트도 `200/200`
통과했다. 따라서 canonical directory 이동은 실행 코드와 성능을 바꾸지 않았으며, 최종 R7에서 전체
12-workload를 3회씩 다시 실행한다.

## R7 최종 승격 결과

전체 12-workload x 3과 경계선 workload 독립 재측정을 완료했다. 선택한 결과는 12/12 exact token
identity, Stage 7 대비 처리량 `-0.28%` 이상, latency p95 `+3%` 이내, peak VRAM `+8 MiB`로 최종 gate를
통과했다. 동일 trace의 cached fresh vLLM 대비 처리량도 12/12에서 높다. 전체 수치, vLLM latency 비교,
full unit suite의 Release `-O3` RoPE 수치 이슈는
`notes/171-v010-project-refactor-final-gate-20260828.md`에 기록했다.
