# 목표 구조와 안전한 분리 경계

## 1. 목표 의미

여기서 E/P/D disaggregation은 같은 노드와 같은 GPU 안에서 실행 자원과 scheduling domain을 나누는 것이다.
단일 요청의 dependency는 유지한다.

```text
Request A: Encoder A -> Prefill A -> Decode A0 -> Decode A1 -> ...
Request B:             Encoder B -> Prefill B -> Decode B0 -> ...

가능한 overlap 예:
time ---->
decode A0  [==========]
encoder B      [===========]
prefill B                    [================]
decode A1     [=====]
```

`Encoder B`와 `Prefill B`도 같은 요청에서는 겹칠 수 없다. 입력 modality를 여러 독립 encoder로 나눌 수 있는
경우에만 vision/audio 간 별도 overlap 가능성을 추가 검토한다.

## 2. 권장 component 경계

```text
Request admission
      |
      v
PhaseScheduler <---- CompletionQueue / Timeline
  |       |       |
  v       v       v
Encoder  Prefill  Decode       phase queues
Worker   Worker   Worker
  |       |       |
stream E stream P stream D
context E context P context D
workspace E workspace P workspace D
      \      |      /
       Shared model/device resources
       Partitioned request/KV ownership
```

### 공통 scheduler interface

정책과 CUDA 실행을 분리한다.

```cpp
enum class InferencePhase
{
    kEncoder,
    kPrefill,
    kDecode,
};

struct SchedulerSnapshot
{
    // Queue age, batch/sequence sizes, last phase timings, active SM budget,
    // memory pressure, and request deadlines.
};

struct ScheduleDecision
{
    // Which jobs to admit, stream priority, resource partition, and next poll point.
};

class ISchedulingPolicy
{
public:
    virtual ScheduleDecision decide(SchedulerSnapshot const& snapshot) = 0;
};
```

실제 이름과 field는 구현 시 coding guideline에 맞춰 정한다. 핵심은 `ISchedulingPolicy`가 CUDA handle이나
TensorRT object를 직접 소유하지 않도록 하는 것이다.

### resource partition backend

```text
IComputePartitionBackend
  - NoopPartitionBackend
  - GreenContextPartitionBackend   (공식 CUDA API 후보)
  - LibSmCtrlPartitionBackend      (연구용/버전 고정 후보)
```

기본 backend는 `Noop`으로 두어 기존 동작을 보존한다. scheduler 검증과 SM partition 실험을 분리할 수 있다.

## 3. phase별 필요한 격리

| Resource | Encoder | Prefill | Decode |
|---|---|---|---|
| CUDA stream | 전용 | 전용 | 전용, 높은 priority 후보 |
| TensorRT execution context | 전용 | 전용 | 전용 |
| TRT context workspace | 전용 | 전용 | 전용 |
| input/output tensor | request/slot별 | phase batch별 | phase batch별 |
| KV write 영역 | 없음 | 소유 slot의 range | 소유 slot의 next position |
| sampling/D2H staging | 없음 | 전용 ring slot | 전용 ring slot |
| completion event | 전용 pool | 전용 pool | 전용 pool |

같은 serialized `ICudaEngine`에서 execution context를 두 개 만드는 방법이 가능하면 weight memory 중복을 줄일 수
있다. 현재 `EngineExecutor`가 runtime, engine, context를 한 객체로 묶으므로 먼저 “engine owner”와 “context
executor”의 수명을 분리하는 refactor가 필요하다.

## 4. KV cache를 동시 실행에 맞게 바꾸는 두 경로

### 경로 A: 기존 linear layout을 유지하는 slot partition

초기 PoC에 권장한다.

- `maxBatchSize`의 slot을 scheduler가 요청에 고정 배정
- phase batch를 만들 때 logical request ID에서 physical slot으로 mapping
- 요청이 끝나도 즉시 전체 cache compaction을 하지 않고 slot을 free list로 반환
- prefill/decode는 서로 다른 slot에만 write
- 동일 요청의 prefill 완료 event 이후 decode가 해당 slot을 사용

장점은 engine binding/layout 변경이 작다는 점이다. 단점은 hole과 fixed-capacity 낭비가 생기며, 현재 engine이
contiguous active batch를 가정하는 부분을 확인해야 한다.

### 경로 B: paged/block KV cache

장기적으로 continuous batching과 높은 utilization에 적합하지만 변경 범위가 크다.

- token/block allocator
- request-to-block table
- attention plugin/binding 변경
- prefix reuse와 eviction 정책 변경
- speculative decode rollback/commit 변경

이번 목표의 첫 단계에서 paged cache까지 동시에 도입하면 scheduler 문제와 cache correctness 문제를 분리하기
어렵다. linear slot partition으로 concurrency를 검증한 뒤 별도 phase로 진행한다.

## 5. 첫 번째 동시 실행 PoC

가장 작은 유효 범위는 다음이다.

1. vanilla model, batch size 1씩
2. 요청 A의 decode와 요청 B의 vision encoder만 overlap
3. vision/base workspace와 output buffer 분리
4. encoder stream과 decode stream 사이에는 불필요한 wait 없음
5. encoder B 완료 event를 prefill B stream이 wait
6. SM partition은 아직 끄고 concurrent execution 자체만 검증

그 다음 prefill context를 decode context와 분리하고 `decode A + prefill B`를 연다.

## 6. TensorRT layer-group scheduling의 제약

BulletServe는 PyTorch model의 `start_layer/end_layer`를 바꾸어 N개 layer씩 forward한다. 현재 TensorRT LLM은
전체 network가 한 engine이고 runtime은 `enqueueV3()` 한 번만 본다.

따라서 다음은 scheduler boundary가 아니다.

- TensorRT `IProfiler` callback
- NVTX layer range
- CUDA event를 enqueue 앞뒤에 기록하는 것

이들은 관측은 가능하지만 실행 중인 engine을 layer 경계에서 멈추거나 mask를 바꾸지 못한다.

layer group scheduling이 필요하면 다음 중 하나가 필요하다.

1. export/build 시 transformer layer를 여러 engine segment로 분할
2. segment 사이 hidden state tensor를 명시적 I/O로 연결
3. segment별 execution context/workspace와 event dependency를 구성
4. KV cache binding은 동일 request slot을 유지
5. segment 수 증가에 따른 launch, workspace, graph, memory traffic 비용을 측정

처음에는 whole-phase enqueue를 scheduling unit으로 삼고, 이후 4~8 layer group 등 몇 개의 굵은 segment를
실험하는 것이 현실적이다.

## 7. SM 분할 선택지

### CUDA Green Contexts

[CUDA Green Contexts 공식 문서](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN__CONTEXTS.html)는
SM resource를 분리해 별도 green context와 stream을 만드는 공식 API다. Tegra SoC를 포함한 architecture별
최소 partition/alignment를 query해야 한다. 다만 disjoint partition이라도 concurrent 실행이나 forward
progress가 항상 보장되는 것은 아니며, TensorRT context를 green context 위에서 생성·실행하는 호환성 PoC가
필요하다.

### libsmctrl

BulletServe의 `libsmctrl`은 CUDA stream 내부 구조의 mask field를 수정한다. 빠르고 stream별 mask를 바꿀 수
있지만 비공개 ABI에 의존한다.

- 실제 API 단위는 TPC mask다. 일반적으로 TPC 하나가 SM 둘과 연결되지만 GPU topology별 검증이 필요하다.
- CUDA/driver version별 offset이 달라진다.
- BulletServe 문서 안에서도 지원 버전 설명이 일치하지 않는다.
- Jetson/Thor와 최신 CUDA에서 별도 port 및 validation이 필요하다.
- TensorRT auxiliary stream까지 같은 partition을 적용하지 않으면 main stream mask만으로 충분하지 않다.
- CUDA graph capture/replay가 어느 stream resource를 보존하는지 backend별 검증이 필요하다.

따라서 제품 기본 경로로 바로 채택하지 말고 `LibSmCtrlPartitionBackend` 실험 feature로 격리한다.

### MPS

MPS는 BulletServe처럼 별도 process/context를 쓰는 경우 유용하다. 하지만 현재 목표는 같은 프로세스이므로 첫
선택은 아니다. 이후 process isolation이 필요해질 때 비교군으로 유지한다.

## 8. 기능별 rollout

| 단계 | 지원 | 의도적으로 제외 |
|---|---|---|
| A | 계측만, 기존 single stream | scheduling 변경 |
| B | encoder/decode 2-stream | prefill 병행, SM mask |
| C | prefill/decode 별도 context | speculative, graph |
| D | configurable scheduler | dynamic SM partition |
| E | fixed SM partition backend | adaptive model |
| F | adaptive partition | layer segmentation |
| G | layer-group scheduler | paged KV cache |
