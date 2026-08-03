# 동일 CUDA context 기반 3-phase continuous-load 로드맵

## 목표 실행 모델

프로세스와 GPU당 CUDA primary context는 하나만 사용한다. phase마다 분리하는 대상은 CUDA context가 아니라
TensorRT execution context, user-managed workspace, phase-local I/O, CUDA stream이다.

```text
process
└─ one CUDA primary context
   ├─ encoder stream
   │  └─ TensorRT IExecutionContext E + workspace E + I/O E
   ├─ prefill stream
   │  └─ TensorRT IExecutionContext P + workspace P + I/O P
   └─ decode stream
      └─ TensorRT IExecutionContext D + workspace D + I/O D
```

TensorRT `IRuntime`, `ICudaEngine`, immutable weights와 indexed KV allocation은 공유한다. 이 구조는 CUDA-context
switch를 만들지 않으면서 TensorRT의 mutable profile/shape/binding state만 phase별로 격리한다.

## 단계

### 1. Context 의미와 resource isolation 확정

- 모호한 shared/independent context 이름을 TensorRT context 이름으로 변경한다.
- `cuStreamGetCtx`로 prefill/decode stream이 같은 `CUcontext`에 속하는지 construction time에 검증한다.
- independent 모드에서는 실제 `IExecutionContext`, workspace, I/O owner와 stream이 모두 달라야 한다.
- `EngineExecutor::createSibling()`은 같은 `ICudaEngine`에서 별도 `IExecutionContext`를 만든다는 것을
  runtime identity로 검증한다.
- Gemma PLE는 큰 immutable table만 공유하고 mutable output buffer/view는 phase별로 분리한다.

통과 조건:

- 같은 CUDA context + 별도 TensorRT context 검증 테스트
- 잘못된 CUDA-context 또는 aliased TensorRT resource는 dispatch 전에 거부
- shared TensorRT context는 host-serialized, independent TensorRT context만 concurrent enqueue
- phase-local PLE output 주소가 서로 다르고 PLE table은 복제되지 않음
- 실제 Gemma shared/independent smoke와 Compute Sanitizer 통과

### 2. Deterministic continuous-load generator

고정된 시작 batch 대신 실행 중에 요청을 지속적으로 도착시킨다.

- 요청 수, requests/s, prompt/output 길이 범위, seed를 CLI로 설정
- 고정 간격 arrival schedule과 seed 기반 길이 생성으로 재현성 확보
- GPU batch가 in-flight인 동안 non-blocking event poll과 새 request admission을 함께 진행
- stable slot이 없으면 bounded pending queue와 backpressure 사용
- request별 arrival, submit, first admission, first token, terminal timestamp 기록
- dispatch별 실제 prefill/decode batch size, queue wait, CUDA time, overlap ratio 기록

통과 조건:

- 동일 seed/config가 동일 schedule 생성
- queue saturation, pending admission, slot reuse, drain이 검증됨
- phase별 실제 batch-size histogram과 request latency CSV가 생성됨
- 실제 Gemma indexed engine에서 continuous admission부터 terminal까지 모두 완료

### 3. Three-phase coordinator

encoder worker와 prefill/decode worker의 독립 dispatch를 하나의 coordinator가 중재한다. phase별 deadline,
in-flight 제한, KV slot 교집합과 encoder-to-prefill handoff를 한 decision에서 다룬다.

### 4. Adaptive chunked prefill

decode queue wait, predicted CUDA cost와 overlap 효율을 이용해 prefill chunk를 동적으로 조절한다.

### 5. SLO 기반 customizable scheduler

TTFT/TPOT 목표, queue pressure, EWMA phase cost, tenant priority를 policy input으로 제공하고 기본 정책과 사용자
callback을 함께 지원한다.

### 6. 실제 encoder engine 연결

Gemma text-only baseline은 유지한다. VLM 실험에서는 같은 CUDA context 안에 별도 encoder
`IExecutionContext`/workspace/I/O를 두고 generic encoder callback을 실제 TensorRT enqueue로 교체한다.

### 7. SM resource backend

`Noop`, stream priority, 지원 가능한 CUDA Green Context, 연구용 SM-mask backend를 scheduler와 분리한다.

### 8. Kernel-group 계측과 engine segmentation

monolithic `enqueueV3()` 내부 kernel 사이에는 host CUDA event를 넣을 수 없다. Nsight/NVTX로 병목 group을 찾은
뒤 필요하면 engine/plugin 경계를 segment하고 segment별 event handoff를 추가한다.

### 9. 최종 load/performance gate

legacy/indexed, shared/independent TensorRT context, SM-controlled 모드의 TTFT, TPOT, throughput, queue wait,
batch-size distribution, overlap, peak VRAM, slot utilization과 sanitizer 결과를 비교한다.

## 이번 구현 범위

이번 작업은 1번과 2번까지다. 2번 결과가 실제로 prefill/decode queue를 동시에 채우고 서로 다른 batch를 만드는
것을 확인하기 전에는 three-phase coordinator나 SM 제어를 붙이지 않는다.

## 진행 상태

- [x] 1. 하나의 CUDA primary context와 phase별 TensorRT context/resource 격리
- [x] 2. deterministic continuous-load generator와 실제 Gemma queue batching
- [x] 3. Three-phase coordinator
- [x] 4. Adaptive chunked prefill
- [x] 5. SLO 기반 customizable scheduler
- [x] 6. 실제 encoder engine 연결
- [ ] 7. SM resource backend
- [x] 8. Phase-level kernel-group 계측과 segmentation
- [ ] 9. 최종 load/performance gate

1단계 결과는 [24번 노트](24-shared-cuda-independent-trt-contexts.md), 2단계 결과는
[25번 노트](25-deterministic-continuous-load.md)에 기록했다. 3, 4, 6, 8단계 결과와 TensorRT engine 내부
segmentation의 남은 경계는 [28번 노트](28-gemma4-vlm-three-phase-adaptive-segmentation.md)에 기록했다.
5단계의 요청별 TTFT/TPOT/priority 계약과 benchmark CLI는
[30번 노트](30-slo-customizable-scheduler.md)에 기록했다.
