# Gemma 4 E2B / intra-GPU phase scheduler 프로젝트 회고

이 문서는 지금까지의 구현과 실험을 한 번에 다시 읽기 위한 source of truth다. 숫자가 서로 다른 이유는
측정 계층이 다르기 때문이다. `legacy/indexed`는 같은 실행 순서에서 KV 주소 방식의 비용을 비교하고,
`shared/independent TensorRT context`는 phase overlap의 비용과 이득을 비교하며, continuous-load는 실제 queue,
admission, sampling, terminal release까지 포함한다.

## 0. 한 문장 결론

현재 구현은 **하나의 CUDA primary context 안에서 encoder/prefill/decode를 서로 다른 CUDA stream과 독립 TensorRT
execution context로 실행할 수 있는 실험용 async serving 기반**이다. Stable indexed-linear KV cache는 KV tensor를
이동하지 않는 고정 physical slot ownership을 제공하고, 그 위에 prefill/decode queue, fixed-128 chunk,
dynamic batching, CUDA-event telemetry, independent-context overlap, VLM encoder handoff까지 연결했다.

다만 이것은 아직 “기존 public `handleRequest()`를 모든 기능에서 완전히 대체한 production server”가 아니다.
SM mask/Green Context backend, TensorRT engine 내부 layer-group segmentation, paged KV, 장시간 2048-capacity
32-slot 독립 실행은 아직 최종 단계가 아니다.

## 1. 처음의 목표와 현재의 범위

### 목표

```text
같은 프로세스 / 같은 GPU / 하나의 CUDA primary context
  ├─ encoder queue  -> encoder stream  -> vision TensorRT context
  ├─ prefill queue  -> prefill stream  -> LLM TensorRT context P
  └─ decode queue   -> decode stream   -> LLM TensorRT context D
```

한 요청의 dependency는 항상 지킨다.

```text
Request A: encoder -> prefill -> decode0 -> decode1 -> ...
Request B:          encoder -> prefill -> decode0 -> ...

가능한 overlap: A의 decode와 B의 encoder/prefill
불가능한 overlap: 같은 요청의 encoder와 prefill
```

### 지금 실제로 확인된 것

| 항목 | 상태 |
|---|---|
| Gemma 4 E2B INT4-AWQ text baseline | 실제 RTX 3080 inference 성공 |
| legacy/indexed greedy output parity | 통과 |
| stable indexed slot/length lifecycle | unit 및 실제 request smoke 통과 |
| fixed-128 chunked prefill | 실제 TensorRT 경로 통과 |
| prefill/decode queue와 dynamic batching | 실제 continuous load에서 확인 |
| 같은 CUDA context + 독립 TensorRT context overlap | 실제 GPU에서 확인 |
| VLM encoder queue/event/prefill handoff | 실제 vision engine trace에서 확인 |
| CUDA-event phase/kernel-group 계측 | phase group 단위 완료 |
| public async facade | V1 제한 범위 구현; drop-in `handleRequest()` 대체 아님 |
| adaptive scheduler | opt-in 실험 정책; fixed-128보다 항상 좋지는 않음 |
| SM mask/Green Context | 아직 backend 미완료 |
| TensorRT 내부 layer-group preemption | 아직 미구현 |
| paged/block KV | 아직 미구현 |

## 2. 환경과 baseline

### 고정된 환경

- GPU: NVIDIA GeForce RTX 3080 10 GB, SM86
- driver: 610.43.02
- CUDA: 13.3
- TensorRT: 11.0.0.114 / NGC `nvcr.io/nvidia/tensorrt:26.06-py3`
- quant/export: `nvcr.io/nvidia/pytorch:25.12-py3`
- model: `google/gemma-4-E2B-it`
- immutable revision: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`

### 왜 FP16이 아니라 INT4인가

Gemma 4 E2B의 FP16 backbone과 PLE/embedding/LM head/vision 데이터를 RTX 3080 10 GB에 동시에 올리기 어렵다.
그래서 backbone만 INT4-AWQ로 양자화하고 다음은 FP16으로 보존했다.

```text
INT4: transformer backbone weights
FP16: embedding, PLE, LM head, vision encoder, KV cache
```

quantization 조건은 WikiText calibration 128 samples, seed 0, CPU ModelOpt 경로다. 생성물은 Git에 넣지 않고
다음에 둔다.

```text
.local/gemma4-e2b/hf
.local/gemma4-e2b/quant-int4-awq
.local/gemma4-e2b/onnx-indexed/llm
.local/gemma4-e2b/engine-indexed-slots32-p8-d32-i128
.local/gemma4-e2b/engine-indexed-slots32-p8-d32-i512
```

### Gemma 4의 중요한 특성

Gemma 4 E2B는 text-only LLM이 아니라 vision encoder와 PLE를 포함할 수 있는 VLM이다. text-only baseline에서도
Gemma 4 입력 계약 때문에 `vision_block_ids` 같은 placeholder 입력을 결정적으로 채워야 한다. 누락하면 첫
attention부터 NaN이 전파될 수 있다.

또한 일부 full-attention head dimension 512는 RTX 3080의 기본 FMHA가 지원하지 않는다. TensorRT는 의도대로
FFPA prefill + XQA decode fallback을 사용한다. 따라서 로그의 `FMHA unsupported headSize=512`는 이 환경에서
실패가 아니라 선택된 fallback을 의미한다.

## 3. 현재 runtime을 이해하는 핵심: CUDA context와 TensorRT context

가장 중요한 개념 분리는 다음이다.

```text
프로세스 / GPU
└─ CUDA primary context 하나
   ├─ CUDA stream P
   │  └─ TensorRT IExecutionContext P + workspace P + I/O P
   ├─ CUDA stream D
   │  └─ TensorRT IExecutionContext D + workspace D + I/O D
   └─ CUDA stream E
      └─ TensorRT IExecutionContext E + workspace E + I/O E
```

- CUDA context: CUDA driver 수준의 실행 주소 공간/장치 context. 목표는 공유한다.
- CUDA stream: 같은 CUDA context 안에서 작업 순서와 비동기 실행 domain을 나눈다.
- TensorRT `IExecutionContext`: optimization profile, shape, binding address, workspace 상태를 가진 mutable
  실행 객체. 실제 concurrent enqueue에는 phase마다 별도 객체가 필요하다.
- `ICudaEngine`와 weight: immutable이므로 여러 TensorRT context가 공유할 수 있다.

### shared TensorRT context fallback

한 개의 TensorRT context와 두 stream만 쓰는 `shared/serialized` 모드도 있다. 이 모드는 queue와 stream은 나누지만
prefill 완료 event 전에는 decode의 `prepare()`와 profile/binding 변경을 시작하지 않는다.

```text
P stream: [prepare][enqueue] -------- [P done event]
D stream:                            wait -> [prepare][enqueue]
```

이 모드는 실제 GPU kernel overlap을 제공하지 않는다. 대신 기존 context를 안전하게 유지하면서 queue 분리,
non-blocking completion, batching 정책을 검증하는 fallback이다.

### independent TensorRT context

실제 overlap은 다음 네 identity가 모두 달라야 한다.

```text
context P != context D
workspace P != workspace D
phase I/O P != phase I/O D
stream P != stream D
```

CUDA context는 같아야 한다. stream만 다르고 TensorRT context/workspace/I/O가 alias이면 race나 profile overwrite가
발생하므로 construction time에 거부한다.

## 4. KV cache: 기존 방식과 indexed-linear 방식

### 4.1 기존 fixed contiguous cache

attention layer마다 다음 GPU tensor를 runtime 초기화 때 한 번 할당한다.

```text
[maxBatchSize, 2(K/V), numKVHeads, maxSequenceLength, headDim]
```

현재 Gemma 4 E2B의 KV는 FP16이고, layer마다 head dimension이 다를 수 있다. `past_key_values_i` 입력과
`present_key_values_i` 출력은 같은 physical tensor를 alias하므로 prefill 결과를 decode용으로 다시 복사하지
않는다.

주소는 대략 다음처럼 계산된다.

```text
physical address = cache_base
                 + batch_row * fixed_slot_stride
                 + (current_length + token_offset) * token_stride
```

장점:

- 주소 계산과 TensorRT binding이 단순하다.
- 정적인 batch/maximum length edge deployment에 예측 가능하다.
- runtime 중 반복 allocation이 없어 CUDA heap 외부 파편화가 작다.

단점:

- 실제 요청이 짧아도 `maxSequenceLength`를 예약한다.
- 실제 active batch가 작아도 `maxBatchSize` slot을 예약한다.
- 중간 request 종료 때 contiguous row를 맞추려고 KV D2D compaction을 한다.
- phase별 logical batch와 physical row가 달라지면 주소가 깨진다.
- 같은 KV를 다른 stream에서 읽는 동안 compaction하면 race가 난다.

### 4.2 stable indexed-linear cache

indexed-linear는 paged cache가 아니다. physical layout은 그대로 fixed-linear로 유지하고, logical row와 physical
slot을 분리한다.

```text
logical active rows:  row0=A, row1=C, row2=D
physical slots:      slot0=A, slot1=free, slot2=C, slot3=D
kv_slot_ids:         [0, 2, 3]
```

prefill/decode kernel은 `batchIdx` 대신 `kv_slot_ids[batchIdx]`로 physical offset을 계산한다. ONNX에는
`kv_slot_ids: INT32[B]` 입력이 추가되고, past KV의 첫 번째 차원은 active batch와 별도의 `kv_slots` symbolic
dimension으로 export된다.

### 4.3 request lifecycle

```text
Free -> Reserved -> Prefilling -> DecodeReady -> Decoding
     -> ReclaimPending -> Free
```

1. `KVSlotAllocator`가 deterministic lowest-free slot을 lease한다.
2. 첫 prefill chunk는 slot length를 0으로 초기화한다.
3. chunk가 끝나면 global physical length를 commit한다.
4. 마지막 chunk만 decode queue로 이동한다.
5. decode 한 step마다 해당 physical slot length를 1 증가시킨다.
6. request가 끝나면 GPU event 완료 후 slot을 free-list로 반환한다.

중간 slot이 비어도 KV 본체는 움직이지 않는다. logical row와 `kv_slot_ids`, phase-local length view만 바뀐다.

### 4.4 길이 view

```text
globalLengths[physical slot]
        ├─ gather(kv_slot_ids_P) -> prefill local lengths
        └─ gather(kv_slot_ids_D) -> decode local lengths

phase completion -> scatter/increment -> globalLengths
```

중요한 계약은 동시에 실행하는 P/D batch의 slot 집합이 disjoint하다는 것이다. 같은 slot을 두 stream에서 쓰면
indexed라도 안전하지 않다.

### 4.5 indexed-linear의 장점과 한계

장점:

- request eviction에서 surviving KV D2D copy를 제거한다.
- hole이 있어도 slot을 O(1) free-list로 재사용한다.
- phase별 logical batch와 physical KV ownership을 분리한다.
- paged attention보다 plugin/kernel 변경 범위가 작다.
- 기존 fixed tensor allocation과 XQA 내부 128-token page view를 유지할 수 있다.

한계:

- VRAM 사용량은 줄지 않는다. token 방향 internal fragmentation은 그대로다.
- slot 하나가 maximum capacity 전체를 예약한다.
- prefix cache zero-copy, host offload, page refcount를 제공하지 않는다.
- global physical length와 phase-local length를 완전히 분리하는 작업은 계속 점검 대상이다.
- v1은 vanilla text prefill/decode 중심이며 speculative, Mamba, host offload, 여러 `handleRequest()` 사이
  continuous admission 등은 제한한다.

### 4.6 메모리 파편화 관점

Gemma 4 E2B에서 capacity 2048 기준 physical slot 하나는 약 84 MiB다. sequence 128 token이 실제로 사용하는
KV는 약 5.25 MiB여도 84 MiB가 예약된다.

```text
capacity 2048, slot 1: 84 MiB reserved
length 128:              5.25 MiB useful, ~78.75 MiB internal waste
length 512:              21 MiB useful
length 1536:             63 MiB useful
```

이것은 CUDA heap external fragmentation이 아니라 fixed tensor 내부의 internal fragmentation이다. 이번 연구의
indexed-linear는 효율보다 stable ownership/compaction 제거를 먼저 확보한 것이다. token utilization이 핵심이
되면 다음 backend는 paged/block KV가 되어야 한다.

## 5. quantize → export → build → inference 흐름

```text
HF Gemma revision
  -> INT4-AWQ quantization (CPU ModelOpt)
  -> indexed ONNX export (`kv_slot_ids`, `kv_slots`)
  -> TensorRT build (P8/D32 profile)
  -> runtime load / greedy inference
  -> CUDA-event metrics / request CSV
```

중간에 발생한 대표 문제:

1. FP16 모델은 VRAM/weight 조건상 부적합했다.
2. 처음 비대칭 builder만 바꾸고 ONNX dimension을 그대로 두자 TensorRT profile이 `batch 8 != 32`로 실패했다.
3. ONNX export에서 past KV 첫 축을 `kv_slots`로 분리해 문제를 해결했다.
4. `present_key_values`는 active output row shape를 유지하며 plugin이 physical cache write를 처리하는 구조다.
5. `maxBatch=32, KV capacity=2048, 32 slots, independent contexts`는 RTX 3080 10GB에서 CUDA OOM이었다.
6. capacity 512 실험 엔진을 별도 생성해 prompt128 + output256 workload를 실행했다.

## 6. phase queue와 async lifecycle

### queue 흐름

```text
submit(request)
    -> stable slot lease
    -> prefill queue
    -> P batch pop (fixed-128 chunks)
    -> P TensorRT enqueue
    -> P done event
    -> final chunk면 decode queue
    -> D batch pop (one token per request)
    -> D TensorRT enqueue + sampling
    -> D done event
    -> unfinished requeue / finished release
```

실제 `PhaseQueueScheduler`, `PhaseDispatchWorker`, `PhaseRequestLifecycle`,
`PhaseContextServingFacade`가 이 host ownership을 나눈다. queue batch size는 phase별로 독립적이다. P8/D32라는
표현은 cap이며, 실제 dispatch는 queue 상태와 request 종료 시점에 따라 D7/D15/D31 등도 발생한다.

### chunked prefill

prompt 512, chunk128이면 다음 네 번의 prefill enqueue가 된다.

```text
chunk0: length 0 -> 128
chunk1: length 128 -> 256
chunk2: length 256 -> 384
chunk3: length 384 -> 512 -> DecodeReady
```

chunking은 prefill 총 비용을 줄이는 기능이 아니다. 긴 prefill을 여러 scheduling point로 나누어 decode queue가
오래 막히지 않게 하는 기능이다. 너무 작은 chunk는 TensorRT launch/attention 반복 비용 때문에 전체 처리량을
낮춘다.

### async completion

기존 synchronous `handleRequest()`는 enqueue 후 `cudaStreamSynchronize()`와 host token 반영을 한 함수에서 했다.
새 경로는 다음으로 나눈다.

```text
enqueue phase + D2H result + done event
                 ... host poll other queues ...
event ready -> complete phase -> requeue/release
```

따라서 CPU worker가 GPU가 끝날 때까지 막히지 않고 다른 queue admission을 진행할 수 있다. 단, in-flight slot과
output buffer는 event 완료 전까지 재사용할 수 없다.

## 7. VLM three-phase와 text-only 경로의 관계

### 실제 VLM topology

```text
JSON/image arrival
   -> encoder queue
   -> image preprocess + vision TensorRT context E
   -> request-owned embedding D2D copy
   -> encoder done event
   -> prefill queue + stable KV slot
   -> LLM prefill context P
   -> decode queue
   -> LLM decode context D
```

`PhaseThreeCoordinator`는 encoder/prefill/decode worker를 event poll로 중재한다. 실제 Gemma vision
`MultimodalRunner`를 encoder callback에 연결했고, 재사용되는 vision output이 다음 request에서 덮이지 않도록
request-owned GPU tensor로 복사한 뒤 event를 publish한다.

### VLM에서 확인한 것

- deterministic single/multi-image placement 진단 통과
- legacy/indexed 출력 byte parity
- 12-request real image trace 12/12 runtime completion
- encoder/prefill/decode kernel group 기록
- 4 rps + overlap budget 1024에서 실제 P/D overlap dispatch 확인

초기에 이미지 비교 prompt가 이미지를 다시 요구한 것은 indexed cache 오류가 아니라 sampling/모호한 prompt에서
나온 false negative였다. greedy single-image와 forward/reverse multi-image 진단에서 placeholder 수와 embedding
row 수가 일치하고 출력도 parity였다.

### text-only로 돌아온 이유

최근 dynamic batching 연구는 encoder semantic 변수와 image token 길이 변수를 제거하고 LLM queue만 관찰하기 위한
것이다. 따라서 최근 P/D matrix는 text-only이며, VLM 결과와 직접 latency 비교하면 안 된다.

## 8. 계측: 무엇을 측정하고 무엇을 측정하지 못하는가

### phase-level groups

```text
encoder: encoder_preprocess -> encoder_engine
prefill: prefill_prepare -> prefill_engine -> prefill_cache_commit -> prefill_sample
decode:  decode_prepare -> decode_engine -> decode_sample
```

각 group 앞뒤에 CUDA event를 기록하고 `kernelGroupCsv`에 GPU ms를 남긴다. queue wait은 host monotonic time,
GPU phase time은 event elapsed time으로 따로 기록한다.

### 중요한 한계

`prefill_engine`과 `decode_engine`은 각각 하나의 TensorRT `enqueueV3()`다. host에서 event를 추가해도 그 안의
transformer layer/kernel 사이를 preempt하거나 layer group별 시간을 직접 얻을 수 없다.

따라서 현재 “kernel-group segmentation 완료”라는 말은 **phase enqueue 주변 group segmentation**을 뜻한다.
진짜 layer-group scheduler를 만들려면 transformer layer를 여러 TensorRT engine segment로 export/build하고 hidden
state와 event를 segment 사이에 명시적으로 연결해야 한다.

## 9. 지금까지의 성능을 올바르게 읽는 법

### 9.1 indexed-only: 주소 안정성의 비용

actual text BS4 eviction workload에서:

- 전체 GPU time: indexed가 legacy보다 약 +1.14%
- prefill: 거의 동일
- decode: 약간 증가
- greedy output: byte parity

즉 indexed-linear는 “정상적인 prefill/decode를 빠르게 하는 최적화”가 아니다. 목적은 stable ownership과
compaction 제거이며, 긴 survivor KV eviction에서 benefit이 커질 가능성이 있다.

### 9.2 shared vs independent TensorRT context

- shared context는 안전 fallback이며 queue separation/async completion의 의미가 크다.
- independent context는 실제 kernel overlap을 허용한다.
- 짧은 phase는 overlap 면적이 커서 makespan speedup이 크다.
- 긴 prefill은 decode kernel을 밀어 TPOT을 악화시킬 수 있다.

기존 fixed-shape matrix에서 independent overlap은 scenario에 따라 약 2~12% makespan 개선을 보였지만, 개별
decode phase는 contention으로 느려질 수 있었다. 따라서 목표 함수는 makespan 하나가 아니라 TTFT, TPOT, E2E p95,
throughput을 함께 봐야 한다.

### 9.3 continuous load와 sustainable knee

초기 BS2/D2, prompt128, output8 workload에서 약 25 req/s까지는 비교적 안정적이었다. 30 req/s부터는 처리량이
25~26 req/s에 머물고 pending/TTFT tail만 증가했다. 1000 req/s burst는 queue를 채우는 offered load일 뿐,
실제 service throughput이 1000 req/s라는 의미가 아니다.

### 9.4 fixed-128 batch matrix

최근 capacity512 실험에서 얻은 request-level p95:

| case | TTFT p95 | TPOT p95 | 해석 |
|---|---:|---:|---|
| P1 | 783 ms | - | 작은 prefill batch, queue/arrival 영향 큼 |
| P2 | 525 ms | - | prefill 효율 개선 |
| P4 | 453 ms | - | 계속 개선 |
| P8 | 417 ms | - | prefill cap 근처 효율 |
| D8 | 2663 ms | 65.47 ms | decode batch 부족 |
| D16 | 1808 ms | 37.67 ms | decode 효율 개선 |
| D24 | 1516 ms | 28.09 ms | 높은 효율 |
| D32 | 1403 ms | 24.32 ms | queue가 충분히 찼을 때 유리 |

### 9.5 긴 decode와 overlap

- output256, 32 requests: D32 dispatch 251회, 1,946.7 tokens/s
- P4/D16 output128: overlap ratio 평균 0.430, independent speedup 1.295x
- P8/D32 output128: overlap ratio 평균 0.414, independent speedup 1.162x

overlap ratio가 높다고 항상 request E2E가 좋아지는 것은 아니다. prefill이 decode의 resource를 잠식하면 TPOT
tail이 늘 수 있다.

## 10. 설계 선택별 장단점

| 선택 | 장점 | 단점/위험 | 현재 판단 |
|---|---|---|---|
| fixed contiguous KV | 단순, 예측 가능, 기존 plugin 호환 | internal waste, compaction | baseline |
| indexed-linear KV | stable slot, eviction copy 0, 변경 작음 | token utilization 개선 없음 | 현재 v1 |
| paged/block KV | token 단위 효율, prefix/COW/eviction 유리 | allocator/plugin/metadata 대규모 변경 | 장기 후보 |
| shared TRT context | VRAM 적음, legacy 호환 | 실제 kernel overlap 불가 | fallback |
| independent TRT contexts | phase kernel overlap | workspace/I/O VRAM 증가, contention | opt-in primary experiment |
| fixed-128 chunk | 비용 예측 가능, scheduler table 작성 쉬움 | 짧은 prompt도 enqueue 분할 가능 | 현재 calibration 기준 |
| adaptive chunk | SLO에 맞출 가능성 | 잘못된 EWMA/launch penalty로 성능 악화 | 재설계 필요 |
| no SM partition | 구현/호환성 단순 | phase contention 제어 불가 | correctness 우선 |
| Green Context | 공식 CUDA resource partition 후보 | topology/alignment/호환성 검증 필요 | 후속 backend |
| libsmctrl | 빠른 stream mask 실험 | 비공개 ABI/버전 위험 | 연구용 only |
| one monolithic engine | build/runtime 단순 | 내부 layer preemption 불가 | 현재 |
| segmented engines | layer-group scheduling 가능 | hidden-state I/O/launch/workspace 비용 큼 | 장기 연구 |

## 11. 실패와 거기서 얻은 교훈

### profile dimension failure

builder에서 P8/D32 cap만 바꾸면 ONNX의 `batch` symbol이 모든 KV tensor와 묶여 있어 TensorRT profile이 거부한다.
phase active batch와 physical KV slot batch는 ONNX export 단계에서 분리해야 한다.

### same TensorRT context race

두 stream이 있어도 같은 `IExecutionContext`의 profile/binding/shape를 동시에 바꾸면 안전하지 않다.
`cudaStreamWaitEvent()`만으로 host-side mutable state race를 해결할 수 없다.

### memory OOM

P32/KV2048/independent contexts는 10GB에서 OOM이었다. active lease 2개와 physical KV allocation 2개는 다르다.
slot을 줄이는 것은 memory는 줄일 수 있지만 engine contract와 correctness를 깨뜨릴 수 있다. capacity와 physical
slot을 명시적으로 분리해 실험하고, 실제 workload에 맞는 engine profile을 따로 만들어야 한다.

### adaptive chunk failure

초기 adaptive policy는 queue wait만 보고 chunk를 64/32로 계속 줄여 TensorRT launch 수를 늘렸다. Gemma 4에서는
32-token prefill도 engine cost가 크게 줄지 않아 throughput과 TTFT가 악화됐다. chunk 후보는 discrete table과
launch penalty를 포함해야 한다.

### VLM semantic false negative

이미지 비교 request가 이미지를 다시 요구한 것은 output sampling과 prompt ambiguity 때문이었다. deterministic
greedy single/multi-image forward/reverse 회귀가 placement correctness를 판단하는 기준이다.

### overlap suite capacity mismatch

기존 overlap suite의 prompt512 + output16은 capacity512 engine에서 KV limit을 초과했다. workload generator가
engine capacity와 prompt/output budget을 먼저 검증해야 한다.

## 12. 현재 repository에서 찾아볼 위치

| 목적 | 주요 파일 |
|---|---|
| builder phase caps/profile | `cpp/builder/llmBuilder.{h,cpp}` |
| runtime config validation | `cpp/runtime/config/llmEngineConfig.{h,cpp}` |
| Gemma indexed ONNX shape | `tensorrt_edgellm/models/gemma4/modeling_gemma4_text.py` |
| phase benchmark/CLI | `examples/llm/llm_phase_bench.cpp` |
| engine CLI | `examples/llm/llm_build.cpp` |
| slot allocator/lifecycle | `cpp/runtime/kvSlotAllocator.*`, `cpp/runtime/scheduling/*` |
| phase context/resource safety | `cpp/runtime/exec/engineExecutor.*`, scheduling safety contracts |
| request pack/scatter | `cpp/runtime/scheduling/phaseContextBatchAdapter.*` |
| actual VLM handoff | `PhaseThreeCoordinator`, `Gemma4PhaseVisionAdapter` 관련 scheduling/multimodal 파일 |
| quant/export | `scripts/gemma4_e2b_indexed/run_quant_export.sh` |
| load experiments | `scripts/gemma4_e2b_indexed/run_phase_load_suite.py` |
| batch cost experiments | `scripts/gemma4_e2b_indexed/run_phase_batch_cost_suite.py` |
| real request experiments | `scripts/gemma4_e2b_indexed/run_phase_real_request_suite.py` |
| output artifacts | `.local/gemma4-e2b/` |

## 13. 권장 복습 순서

1. [현재 CUDA 실행 경로와 KV cache](01-current-cuda-and-kv-cache.md)
2. [KV cache 구조 선택지](06-kv-cache-design-options.md)
3. [Gemma indexed 구현](07-gemma4-e2b-int4-indexed-implementation.md)
4. [공유 CUDA/분리 TensorRT context](24-shared-cuda-independent-trt-contexts.md)
5. [chunked prefill과 slot length](09-chunked-prefill-and-slot-lengths.md)
6. [request lifecycle와 serving facade](12-continuous-request-lifecycle.md), [15번 노트](15-continuous-context-serving-facade.md)
7. [continuous load](25-deterministic-continuous-load.md), [real request](33-llm-real-request-scheduler-experiments.md)
8. [fixed-128 cost table](34-fixed128-batch-cost-profile.md)
9. [최근 decode-heavy/overlap 결과](37-dynamic-batch-decode-heavy-20260805.md), [38번 노트](38-long-decode-overlap-20260805.md)
10. 마지막으로 [scheduler 계획](03-instrumentation-and-scheduler-plan.md)과 실제 policy 구현을 비교한다.

각 문서를 읽을 때 다음 네 질문을 반복하면 된다.

1. 이 단계에서 logical request state와 physical GPU state는 무엇인가?
2. 어느 stream/context/workspace/I/O가 공유되고 어느 것이 독립인가?
3. CUDA event가 실제로 보장하는 dependency는 무엇이며, 측정하지 못하는 것은 무엇인가?
4. latency, throughput, memory, correctness 중 어떤 목표를 개선하고 어떤 비용을 지불했는가?

## 14. 다음 단계의 우선순위

### 단기: 측정의 신뢰도

1. fixed-128 batch table을 prompt 128/512, decode context 128/512/1536, output 128/256으로 반복한다.
2. 각 scenario를 process-level 3회 이상 돌리고 GPU clock, temperature, peak VRAM, confidence interval을 저장한다.
3. legacy/indexed eviction을 실제 4->3->2->1 trace에서 Nsight Systems와 compute-sanitizer로 비교한다.
4. 2048 engine은 작은 physical slot 수와 longer context를 따로 검증하고, 512 engine의 high-concurrency 결과와
   섞지 않는다.

### 중기: dynamic batching policy

각 후보 `(P batch, D batch, chunk, decode context)`에 대해 다음을 lookup한다.

```text
predicted TTFT
predicted TPOT
predicted E2E / makespan
queue wait
overlap slowdown
available slots / memory headroom
```

정책은 oldest request의 남은 SLO를 위반하는 후보를 제거하고, 남은 후보 중 completed-token utility가 높은 것을
선택해야 한다. missing cell은 낙관적 interpolation 대신 nearest smaller measured batch로 fallback한다.

### 장기: resource partition과 engine segmentation

1. `NoopPartitionBackend`를 기준으로 stream priority를 비교한다.
2. Green Context를 실제 TensorRT context/auxiliary stream과 검증한다.
3. libsmctrl은 연구용 backend로만 격리한다.
4. 실제 Nsight 병목 group이 명확해진 뒤 4~8 layer 단위 TensorRT segment를 별도 build한다.
5. stable slot ownership 계약을 유지한 채 paged KV backend를 추가한다.

## 15. 최종 판단 기준

다음 단계로 넘어가기 전에 최소한 아래를 동시에 확인해야 한다.

- output token/finish reason parity
- compute-sanitizer 0 error
- indexed eviction 중 KV D2D copy 0
- independent context에서 context/workspace/I/O/stream alias 0
- TTFT/TPOT/E2E p95가 목표 SLO 안에 있음
- queue saturation 시 pending이 무한히 증가하지 않음
- peak VRAM headroom 최소 512 MiB
- fixed/continuous/real-request workload에서 같은 결론 재현

이 조건을 통과하면 SM partition과 engine segmentation을 연구할 수 있다. 통과하지 못하면 scheduler를 더 복잡하게
만들기보다 slot/length ownership, memory budget, cost table의 잘못된 가정을 먼저 수정해야 한다.
