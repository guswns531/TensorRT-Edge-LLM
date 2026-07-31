# CUDA event 계측과 scheduler 계획

## 1. 계측 목표

계측은 세 종류를 분리한다.

1. CPU queue 시간: admission부터 GPU 제출까지
2. GPU phase 시간: 해당 stream에서 phase 시작 event부터 끝 event까지
3. dependency/overlap: phase 간 event edge와 host monotonic timestamp

GPU event 하나로 queueing, host overhead, 실제 kernel 실행 시간을 모두 표현하려 하지 않는다.

## 2. event 사용 규칙

### timing event

- `cudaEventCreate()`로 timing 활성 event를 pool에 미리 생성
- phase stream에 `start`, workload, `end` 순서로 record
- scheduler thread는 `cudaEventQuery(end)`로 non-blocking poll
- 완료된 record만 `cudaEventElapsedTime(start, end)` 계산
- hot path에서 `cudaEventSynchronize()`나 `cudaDeviceSynchronize()` 금지

### dependency event

- `cudaEventCreateWithFlags(..., cudaEventDisableTiming)` 사용
- producer stream에 완료 event record
- consumer stream은 `cudaStreamWaitEvent()`로 wait
- host thread가 producer를 synchronize한 뒤 consumer를 제출하는 방식은 피한다

예:

```text
encoderStream: [encoder B] --record(E_B_done)
prefillStream:                 wait(E_B_done) [prefill B] --record(P_B_done)
decodeStream:                                          wait(P_B_done) [decode B0]
```

서로 다른 요청의 독립 phase 사이에는 event edge를 만들지 않는다.

## 3. 기록할 timeline record

```text
requestId
phase
iteration 또는 segmentId
batchSize
inputTokens
contextTokens
streamClass
streamPriority
resourcePartitionId
smOrTpcBudget
cpuEnqueueTimestamp
cpuCompletionTimestamp
gpuElapsedMs
queueElapsedMs
policyName
decisionReason
```

처음부터 각 CUDA kernel 이름을 scheduler record에 넣지 않는다. TensorRT 내부 kernel은 Nsight Systems와
TensorRT profiler로 별도 수집하고, online scheduler는 phase 또는 engine-segment 수준의 저비용 record만 쓴다.

## 4. event ring 설계

현재 global `Timer`는 같은 stage의 중첩 실행에 적합하지 않다. scheduler용으로 bounded ring을 둔다.

```text
free records -> submitted records -> completed records -> free records
```

- phase/stream별 SPSC 또는 전체 MPSC completion queue
- event pair는 매 iteration 생성/삭제하지 않고 재사용
- generation counter로 stale completion 방지
- ring이 가득 차면 inference를 막기보다 상세 timing sample을 drop하고 drop counter 증가
- correctness dependency event는 절대 drop하지 않음

## 5. 기본 제공 scheduler 안

정책 이름 예: `DeadlineAwareComplementaryPolicy`.

### 입력

- encoder/prefill/decode queue age
- 각 job의 input length, batch size, decode context length
- 최근 phase별 EWMA latency
- 현재 active resource partition
- TTFT/TPOT target
- GPU memory headroom

### 초기 규칙

1. decode deadline/TPOT 보호를 최우선으로 한다.
2. decode가 없으면 encoder 또는 prefill에 전체 resource를 준다.
3. decode가 있으면 가장 오래 기다린 encoder/prefill 중 하나만 함께 실행한다.
4. 세 phase 동시 실행은 2-phase 결과가 충분히 쌓일 때까지 금지한다.
5. fixed partition table로 시작하고, online predictor는 계측 데이터만 수집한다.
6. deadline miss가 연속 발생하면 decode budget을 한 단계 늘린다.
7. GPU OOM 위험이나 unknown shape이면 기존 직렬 경로로 fallback한다.

예시 decision table은 하드웨어 profile로 보정해야 하며 숫자를 제품 default로 고정하지 않는다.

| 상태 | 실행 |
|---|---|
| decode queue 비어 있음 | oldest encoder/prefill 단독 |
| decode deadline 여유 큼 | decode + oldest prefill/encoder |
| decode deadline 임박 | decode 단독 또는 decode budget 확대 |
| prefill TTFT 임박 | decode 최소 보장 + prefill 우선 |
| encoder와 prefill 모두 대기 | 요청 dependency와 deadline으로 하나 선택 |

## 6. customization 지점

configuration에서 다음을 주입 가능하게 한다.

- scheduling policy
- partition backend
- phase별 stream priority
- max concurrent phases
- timing sample rate
- TTFT/TPOT target
- fixed partition 후보 목록
- fallback 조건

library 사용자가 C++ policy를 구현할 수 있도록 interface를 제공하고, JSON config는 기본 제공 policy의 parameter만
설정하도록 제한하는 편이 안전하다.

## 7. offline profile에서 online adaptation까지

### 단계 1: isolated profile

phase별로 batch/input/context 길이를 sweep하고 전체 GPU에서 latency를 기록한다.

### 단계 2: fixed co-run profile

`decode + prefill`, `decode + encoder`를 고정 partition 후보별로 실행한다. isolated latency 대비 slowdown을
기록한다.

### 단계 3: lookup policy

가장 가까운 bucket의 profile을 사용해 deadline을 만족하는 후보 중 throughput이 높은 partition을 선택한다.

### 단계 4: online correction

실측/predicted 비율의 EWMA로 bucket을 보정한다. 학습이 불안정하거나 표본이 부족하면 fixed safe policy로
fallback한다.

BulletServe의 analytical model을 처음부터 복제하기보다 이 순서로 correctness와 predictor 오차를 분리한다.

## 8. 구현 phase

### Phase 0: 관측만 추가

- phase ID와 request ID가 있는 event ring
- 기존 single-stream 결과와 latency 변화 확인
- Nsight Systems NVTX range와 record correlation

### Phase 1: stream/context 소유권 분리

- `PhaseStreamSet`
- engine owner와 execution context 분리
- context별 workspace
- phase별 PipelineIO/staging
- event dependency API

### Phase 2: encoder/decode overlap

- 두 요청 pipeline
- SM partition 없음
- racecheck, output parity, overlap 확인

### Phase 3: prefill/decode overlap

- physical KV slot ownership
- compaction 비활성 또는 scheduler-controlled
- decode CUDA graph 비활성부터 시작

### Phase 4: policy abstraction과 기본 policy

- `ISchedulingPolicy`
- `DeadlineAwareComplementaryPolicy`
- metrics export 및 fallback

### Phase 5: partition backend

- `Noop`
- fixed Green Context 또는 libsmctrl 실험 backend
- main/aux stream resource 검증

### Phase 6: adaptive policy

- profile table
- online correction
- SLO guard

### Phase 7: engine segmentation

- layer group engine build/export
- segment event와 hidden-state handoff
- group size sweep

## 9. 피해야 할 구현

- 기존 caller stream만 세 변수로 복사하고 같은 `IExecutionContext`를 동시 호출
- 공유 `mSharedExecContextMemory`를 그대로 두고 engine overlap
- `cudaDeviceSynchronize()`로 dependency 해결
- batch compaction 중 다른 stream에서 같은 KV tensor 접근
- SM mask가 TensorRT auxiliary stream에도 적용됐다고 검증 없이 가정
- CUDA event timing 결과만으로 request queue/SLO 계산
- layer profiler callback을 preemption point로 취급
