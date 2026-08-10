# 수정 전 legacy와 수정 후 indexed/phase 경로 비교

이 문서는 “무엇이 빨라졌는가”보다 **무엇이 바뀌었고, 무엇은 그대로이며, 어떤 비용을 추가했는가**를 비교한다.
비교 대상은 한 단계가 아니라 다음 tier로 나눈다.

```text
L0  legacy: active batch row = physical KV row, handleRequest 직렬
L1  indexed-only: stable kv_slot_ids, 기존 실행 순서
L2  indexed + shared TRT: queue/stream 분리, TensorRT context 하나로 event 직렬화
L3  indexed + independent TRT: 같은 CUDA context, TRT context/workspace/I/O/stream 분리
L4  serving adapter: request context pack/scatter, stable lease, completion/requeue
L5  continuous scheduler: arrival, pending admission, dynamic P/D batching, policy telemetry
```

## 1. 전체 차이 요약

| 차원 | 수정 전 legacy | 수정 후 indexed/phase |
|---|---|---|
| API 실행 | `handleRequest()` 한 호출 안에서 encoder/prefill/decode 직렬 | 기존 API 유지 + opt-in `submit/poll/completion` async 경로 |
| CUDA context | 하나 | 하나를 계속 공유 |
| CUDA stream | 사실상 caller stream 중심 | encoder/prefill/decode phase별 stream |
| TensorRT context | base/runtime의 mutable context 재사용 | 실제 overlap에는 phase별 `IExecutionContext` |
| TRT workspace/I/O | 공유/재사용 | concurrent phase마다 별도 owner |
| KV 주소 | batch row가 곧 physical slot | `logical row -> kv_slot_ids -> physical slot` |
| eviction | surviving KV를 contiguous row로 D2D compaction | slot ID/length/request metadata만 재배열 |
| KV allocation | fixed contiguous | fixed contiguous 그대로; indexed는 indirection만 추가 |
| length 관리 | active batch row length 중심 | global physical length + phase-local gathered view |
| batching | 한 active batch 중심 | prefill queue와 decode queue가 각각 batch 구성 |
| prefill | 한 번에 긴 prompt 또는 제한적 chunk | fixed-128 chunk와 queue requeue |
| decode | batch가 줄면 KV compaction 후 재실행 | 한 token씩 stable slot을 pack/scatter |
| 계측 | stage/전체 latency 중심 | queue wait, CUDA event, phase/kernel-group, overlap ratio |
| VLM | synchronous multimodal path | encoder queue/event -> prefill -> decode coordinator |
| SM 제어 | 없음 | 아직 없음; backend interface 후보만 존재 |

핵심은 **L1이 memory allocator를 바꾼 것이 아니고, L2/L3가 execution ownership과 scheduling을 바꾼 것**이다.

## 2. KV cache 동작 방식 비교

### 2.1 수정 전: fixed contiguous + physical compaction

layer마다 다음 tensor를 미리 할당한다.

```text
[maxBatchSize, 2(K/V), numKVHeads, maxKVCacheCapacity, headDim]
```

legacy에서는 다음이 항상 같다.

```text
active row 0 -> physical slot 0
active row 1 -> physical slot 1
active row 2 -> physical slot 2
```

request B가 종료되어 row 1을 제거하면 살아 있는 C/D를 앞쪽으로 옮긴다.

```text
before: row0=A/slot0, row1=B/slot1, row2=C/slot2, row3=D/slot3
after:  row0=A/slot0, row1=C/slot1, row2=D/slot2

필요한 작업: layer별 KV D2D copy + length/state copy
```

이 방식은 active batch가 contiguous라는 TensorRT/plugin 계약에 맞추기 쉽지만, GPU가 decode를 읽는 중에
compaction이 시작되면 race가 된다.

### 2.2 수정 후: stable indexed-linear

physical tensor는 그대로 두고 mapping을 추가한다.

```text
physical: slot0=A | slot1=free | slot2=C | slot3=D
logical:  row0=A | row1=C    | row2=D
kv_slot_ids = [0, 2, 3]
```

RoPE/KV write, cache gather, XQA decode는 `batchIdx`가 아니라 `kv_slot_ids[batchIdx]`를 사용한다.

request B가 끝나도:

```text
KV tensor:  이동하지 않음
length:     [slot0 length, slot2 length, slot3 length]를 gather
free-list:  slot1 반환
logical row만 [A,C,D]로 갱신
```

따라서 중간 hole은 생기지만 외부 CUDA heap 파편화가 아니라 fixed slot 내부의 logical hole이다. 모든 slot 크기가
같아서 free-list 재사용은 단순하다.

### 2.3 length 관리

legacy는 active row length를 중심으로 업데이트한다. indexed/phase 경로는 다음 view를 사용한다.

```text
globalLengths[physical slot]
    ├─ gather(kv_slot_ids_P) -> P local lengths
    └─ gather(kv_slot_ids_D) -> D local lengths

P/D completion -> physical slot에 commit/increment
```

동시에 실행하는 prefill/decode slot 집합은 disjoint해야 한다. `kv_slot_ids`가 있어도 같은 physical slot을 두
stream에서 쓰면 안전하지 않다.

## 3. eviction과 copy volume

Gemma 4 E2B, FP16 KV, 35 attention layers, KV head 1 기준에서 physical slot 하나는 capacity 2048일 때 약
84 MiB다. legacy가 survivor를 이동할 때의 이론적 KV payload는 다음과 같다.

| survivor 길이 | 한 slot 이동 | 두 slot 이동 | indexed |
|---:|---:|---:|---:|
| 128 | 5.25 MiB | 10.50 MiB | 0 MiB KV D2D |
| 512 | 21.00 MiB | 42.00 MiB | 0 MiB KV D2D |
| 1536 | 63.00 MiB | 126.00 MiB | 0 MiB KV D2D |
| 2048 | 84.00 MiB | 168.00 MiB | 0 MiB KV D2D |

이 값은 제거되는 physical payload의 계산값이다. 실제 DRAM traffic과 latency는 Nsight Systems로 별도 확인해야
하며, 현재 문서의 0 MiB는 indexed branch가 `compactKVCacheBatched()`를 호출하지 않는다는 의미다.

### legacy 장점

- active rows가 항상 contiguous라 plugin과 기존 tensor shape가 단순하다.
- 작은 batch/짧은 KV에서는 compaction 비용이 작을 수 있다.
- 기존 `handleRequest()`와 CUDA graph 경로에 변화가 적다.

### indexed 장점

- survivor KV를 복사하지 않는다.
- request 순서가 바뀌어도 physical ownership이 안정적이다.
- prefill/decode가 서로 다른 logical batch를 구성할 수 있다.
- stable slot lease와 cancellation/requeue를 표현하기 쉽다.

### indexed 비용

- `kv_slot_ids` GPU input, gather/scatter, slot allocator metadata가 추가된다.
- batch=1·짧은 sequence에서는 indirection 비용 때문에 조금 느릴 수 있다.
- fixed allocation 자체는 그대로이므로 memory utilization은 개선되지 않는다.

## 4. 메모리 사용량 비교

### 4.1 legacy와 indexed-only

동일한 Gemma INT4 engine 조건에서 측정된 값이다.

| 항목 | legacy | indexed-only | 차이 |
|---|---:|---:|---:|
| serialized engine | 1,001,923,868 B | 1,002,044,972 B | +121,104 B, 약 +0.012% |
| TensorRT build peak | 4,609 MiB | 4,609 MiB | 동일 |
| actual text inference peak | 8,480 MiB | 8,486 MiB | +6 MiB |
| KV tensor layout | fixed contiguous | fixed contiguous | 동일 |

따라서 indexed-only는 memory-saving 기능이 아니다. 추가 메모리는 slot ID/length metadata와 plugin 경로 정도이며,
큰 비용은 여전히 KV capacity와 PLE/embedding/engine weights가 차지한다.

### 4.2 phase context 추가 비용

동일 CUDA context에서 TensorRT context/workspace를 늘리면 다음처럼 증가한다.

| 구성 | 관측 peak | headroom (10,240 MiB 기준) |
|---|---:|---:|
| indexed + shared TensorRT context | 7,898 MiB | 1,968 MiB |
| indexed + independent P/D contexts | 8,626 MiB | 1,240 MiB |
| 추가 context/workspace 관측 비용 | 약 728 MiB | - |

이 수치는 BS2, 긴 input/past KV phase benchmark 조건이다. tokenizer/network/full VLM 상태가 포함된 모든
production 상황의 보장은 아니다.

### 4.3 32-slot/2048-capacity 한계

`maxBatch=32`, physical slots=32, KV capacity=2048, independent P/D context는 RTX 3080 10GB에서 OOM이었다.
원인은 다음이 합쳐졌기 때문이다.

```text
FP16 KV: 32 slots * full capacity
+ INT4 engine weights
+ FP16 PLE/embedding
+ prefill workspace
+ decode workspace
+ independent TensorRT context 2개
```

그래서 dynamic P8/D32 실험에는 동일 phase profile이지만 KV capacity 512인 별도 engine을 사용했다. capacity512는
prompt128 + output256 workload에는 충분하지만 2048 baseline과 같은 memory contract가 아니다.

### 4.4 파편화

| 종류 | 수정 전 | 수정 후 |
|---|---|---|
| CUDA heap external fragmentation | runtime 중 allocate/free가 적어 낮음 | preallocated phase resources를 유지하면 낮음 |
| fixed-slot internal fragmentation | 존재 | 그대로 존재 |
| batch hole | compaction으로 제거 | logical hole을 유지하고 free-list 재사용 |
| token-level sharing | 불가능 | 여전히 불가능 |
| paged block reclaim | 없음 | 없음 |

짧은 request가 capacity 2048 중 128 token만 사용해도 slot 전체를 예약하는 점은 양쪽이 같다. 실제 memory efficiency를
높이려면 indexed 다음 단계로 paged/block KV가 필요하다.

## 5. 실행/동시성 비교

### 수정 전

```text
handleRequest()
  -> preprocess
  -> prefill enqueue
  -> synchronize / sample
  -> decode enqueue
  -> synchronize / sample
  -> compact / next request
```

같은 runtime context, shared workspace, shared I/O를 재사용하므로 단일 요청 실행에는 안전하지만 다른 요청의
encoder/prefill/decode를 pipeline처럼 겹치기 어렵다.

### 수정 후 shared mode

```text
prefill queue -> P stream -> P done event
decode queue  -> wait(P done) -> D stream
```

실제 GPU kernel overlap은 없지만, host는 event를 poll하며 다른 queue를 관리할 수 있다. 기존 TensorRT context
alias 위험을 피하는 호환 경로다.

### 수정 후 independent mode

```text
P stream: [prefill context P + workspace P + I/O P] =====
D stream:             [decode context D + workspace D + I/O D] ===
```

같은 CUDA context에서 실제 kernel overlap이 가능하다. 대신 workspace/I/O/PLE mutable output을 phase별로 나누고
VRAM headroom을 다시 확인해야 한다.

## 6. API와 batching 비교

| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 요청 입력 | `handleRequest()` blocking | `submit()`, `poll()`, `tryPopCompletion()`, `cancel()` V1 |
| admission | 호출 시점의 active batch | stable slot lease + pending queue |
| prefill | 한 batch의 prompt 처리 | prefill queue, fixed-128 chunk, continuation requeue |
| decode | 기존 active batch 반복 | decode queue가 매 step request를 다시 pack |
| output budget | batch 공통 제약이 강함 | request별 output length로 terminal/scatter |
| eviction | KV와 active row 함께 compact | request metadata/slot ID만 compact |
| backpressure | runtime caller가 처리 | pending admission과 slot exhaustion telemetry |
| public compatibility | 기존 경로 | legacy `handleRequest()` 기본값 유지 |

현재 async facade는 vanilla greedy text 중심이며 spec decode, LoRA, audio, logprobs, legacy streaming callback,
system-prompt cache 등은 거부한다. 따라서 “public API 완전 대체”가 아니라 안전한 opt-in serving boundary다.

## 7. 성능 비교

### 7.1 legacy vs indexed-only: 같은 실행 순서

actual text BS4, 서로 다른 stop 조건으로 active batch가 4→3→2→1로 줄어드는 workload다.

| 단계 | legacy | indexed-only | 변화 |
|---|---:|---:|---:|
| prefill median / p95 | 17.6551 / 17.8228 ms | 17.6067 / 17.8129 ms | -0.27% / -0.06% |
| generation median / p95 | 5.9976 / 6.8363 ms | 6.1302 / 6.9324 ms | +2.21% / +1.41% |
| 전체 GPU time | 4,844.62 ms | 4,899.61 ms | +1.14% |
| peak VRAM | 8,480 MiB | 8,486 MiB | +6 MiB |
| output/finish reason | 동일 | 동일 | parity |

해석: 짧은 survivor KV에서는 compaction을 제거해도 indirection/metadata 비용이 더 크게 보일 수 있다. indexed의
성공 기준은 standalone latency 향상이 아니라 eviction copy 제거와 phase scheduler가 사용할 ownership 기반이다.

### 7.2 shared vs independent TensorRT context

고정 shape phase benchmark에서 shared context는 안전한 직렬 fallback이라 median/p95 비용이 대체로 +0.3~+1.4%
범위였다. independent context는 실제 overlap으로 makespan을 줄였다.

| workload 크기 | independent makespan 개선 경향 |
|---|---:|
| BS1, prompt/past 128 | 약 10~12% |
| BS1, prompt/past 512 | 약 6~7% |
| BS1, prompt1024/past1536 | 약 4% |
| BS2, 짧은 phase | 약 9~10% |
| BS2, 긴 phase | 약 2~4% |

phase가 길어질수록 한 phase가 GPU를 오래 점유하므로 overlap의 상대 이득이 줄고 decode contention이 커진다.

### 7.3 dynamic batching에서 관찰된 최신 결과

capacity512 실험 엔진, fixed128, P8/D32 cap에서 실제 queue를 채운 결과다.

| 조합 | TTFT p95 | TPOT p95 | 해석 |
|---|---:|---:|---|
| P1 | 783 ms | - | prefill batching 부족 |
| P4 | 453 ms | - | prefill 효율 개선 |
| P8 | 417 ms | - | prefill cap 근처 |
| D8 | 2,663 ms | 65.47 ms | decode queue가 충분히 크지 않음 |
| D16 | 1,808 ms | 37.67 ms | batch 효율 증가 |
| D24 | 1,516 ms | 28.09 ms | 높은 decode 효율 |
| D32 | 1,403 ms | 24.32 ms | steady-state 최대 batch |

output256 workload에서는 D32 dispatch가 251회 유지되고 1,946.7 tokens/s를 기록했다. P4/D16 overlap은
1.295배, P8/D32 overlap은 1.162배 makespan speedup을 보였다. 이 값들은 legacy와 직접 비교한 값이 아니라
indexed phase scheduler 내부에서의 workload 측정이다.

## 8. 정확성과 안정성 비교

| 검증 | 수정 전 기준 | 수정 후 결과 |
|---|---|---|
| greedy text output | baseline 생성 | legacy/indexed byte parity |
| finish reason | contiguous batch | 4→3→2→1 eviction에서도 동일 |
| indexed slot mapping | 없음 | non-identity mapping 테스트 통과 |
| chunk length | active row 중심 | global physical + phase-local gather/commit |
| same context overlap | 암묵적 직렬 | alias 검사 후 shared/independent 명시 |
| VLM image placement | synchronous baseline | deterministic single/reverse multi-image parity |
| sanitizer | legacy 중심 | indexed/phase test 범위 통과; 최종 eviction Nsight gate는 계속 확인 |

수정 후에도 기존 `handleRequest()`는 기본 경로로 남겨 legacy ABI와 동작을 보존한다. 새 기능은 config/CLI opt-in
이다.

## 9. 장단점 판단

### 수정 전 legacy가 더 나은 경우

- 요청 수가 작고 batch가 정적이다.
- slot eviction이 거의 없거나 survivor KV가 짧다.
- 10GB 카드에서 VRAM headroom이 매우 부족하다.
- 기존 CUDA graph/handleRequest 호환성이 최우선이다.
- phase를 실제로 동시에 실행할 필요가 없다.

### 수정 후 indexed/phase가 더 나은 경우

- continuous admission과 중간 request 종료가 빈번하다.
- prefill과 decode queue가 서로 다른 batch size를 가져야 한다.
- 긴 survivor KV에서 batch compaction copy가 병목이다.
- decode SLO를 보호하면서 다른 request의 prefill/encoder를 overlap하고 싶다.
- stable request ownership과 event-based completion이 필요하다.

### 수정 후 새로 생긴 위험

- context/workspace/I/O alias를 잘못 분리하면 race/NaN이 발생한다.
- independent context가 VRAM을 추가로 사용한다.
- overlap이 항상 latency 개선을 의미하지 않는다.
- fixed-linear 내부 파편화는 해결되지 않는다.
- scheduler policy가 잘못되면 queue wait와 tail latency가 급격히 증가한다.
- SM mask/libsmctrl은 driver/TensorRT auxiliary stream까지 함께 검증해야 한다.

## 10. 현재 결론과 다음 비교 실험

현재까지 가장 타당한 해석은 다음이다.

```text
legacy
  -> 단순하고 memory overhead가 작음

indexed-only
  -> 약 1% 수준의 정상 실행 비용으로 stable ownership 확보
  -> eviction KV copy를 제거할 준비

indexed + independent phase
  -> context/workspace 비용을 지불하고 실제 overlap 확보

dynamic scheduler
  -> batch/queue/SLO를 조절해 overlap 이득을 request workload에 맞춤
```

다음 비교는 반드시 같은 trace와 같은 engine 조건에서 수행해야 한다.

1. legacy vs indexed-only: output128/512/1536 survivor eviction, 실제 D2D bytes와 latency
2. indexed shared vs indexed independent: 동일 arrival trace, TTFT/TPOT/E2E p95와 peak VRAM
3. P/D 후보: `(P1,D32), (P4,D16), (P8,D32)`를 동일 output budget으로 비교
4. capacity: 2048 baseline과 512 high-concurrency를 별도 표로 유지
5. scheduler: fixed128 lookup policy vs current adaptive policy, warmup 제외 3회 이상
6. 이후에만 SM partition과 layer segmentation을 추가

비교 결과를 해석할 때는 항상 다음 네 가지를 함께 표시한다.

```text
correctness parity
memory/headroom
request latency (TTFT/TPOT/E2E)
GPU work (phase/kernel time, queue wait, copy bytes)
```

이 네 축 중 하나만 좋아지고 나머지가 악화되면 “성능 향상”이 아니라 trade-off로 기록해야 한다.
