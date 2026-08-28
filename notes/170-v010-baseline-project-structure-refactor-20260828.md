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
