# KV cache 구조 선택지와 Edge-LLM 지원 전략

이 문서는 현재 TensorRT Edge-LLM의 KV cache가 실제로 어떻게 저장되고 갱신되는지 설명하고,
일반적으로 사용되는 KV cache 기법을 비교한 뒤, 같은 GPU 안에서 encoder/prefill/decode를 분리 실행하려면
어떤 구조로 확장하는 것이 안전한지 정리한다.

## 1. 먼저 구분해야 할 네 가지 축

KV cache 기법들은 서로 완전히 대체 관계가 아니다. 다음 네 축은 대부분 독립적으로 조합할 수 있다.

| 축 | 대표 선택지 | 해결하는 문제 |
|---|---|---|
| 주소 배치 | 고정 연속 배열, indexed-linear, paged/block, virtual-contiguous, ring buffer | 토큰의 K/V가 GPU 주소 어디에 놓이는가 |
| 재사용 정책 | 재사용 없음, layer sharing, prefix/radix cache, copy-on-write | 이미 계산한 K/V를 누가 같이 쓰는가 |
| 메모리 계층 | GPU only, GPU+host offload, 원격 KV 전송 | K/V가 어느 장치에 상주하는가 |
| 표현 형식 | FP16/BF16, FP8/INT8, 더 낮은 bit, MHA/GQA/MQA | 한 토큰의 K/V가 차지하는 용량과 정확도 |

예를 들어 `paged + prefix sharing + FP8 + host offload`는 가능한 조합이다. 반대로 “paged cache와
prefix cache 중 무엇을 택할 것인가”는 올바른 질문이 아니다. paged는 주소 배치이고 prefix cache는 재사용
정책이다.

## 2. KV cache의 크기와 의미

decoder-only Transformer에서 attention layer는 이전 token들의 key와 value를 저장한다. 현재 코드의 layer별
논리 shape는 다음과 같다.

```text
[batch slot, K_or_V, kv head, sequence position, head dimension]
```

고정 용량 방식의 총 메모리는 대략 다음 식으로 계산할 수 있다.

```text
sum_over_layers(
    max_batch
  * 2
  * num_kv_heads[layer]
  * max_kv_capacity
  * head_dim[layer]
  * bytes_per_element
)
```

GQA/MQA와 FP8은 token당 비용을 줄이지만, 빈 batch slot이나 쓰지 않은 sequence capacity의 고정 할당
낭비는 없애지 않는다.

## 3. 현재 Edge-LLM 방식: fixed contiguous slot

### 3.1 물리 할당

`KVCacheManager`는 attention layer마다 시작 시점에 다음 크기의 연속 GPU tensor 하나를 할당한다.

```text
[maxBatchSize, 2, numKVHeads, maxSequenceLength, headDim]
```

현재 allocator가 허용하는 element type은 FP16과 FP8이다. 요청 길이가 짧아도 `maxSequenceLength` 전체,
실제 batch가 작아도 `maxBatchSize` 전체를 미리 확보한다. 주소 계산은 단순하지만 내부 단편화와 예약 메모리
비용이 크다.

`SharedResources::createForLLM()`은 config의 모든 attention layer에 대해 manager를 생성한다.
`kvSharingDonors` recipient도 manager 차원에서는 layer tensor가 할당되고, binding에서 donor tensor를 대신
참조한다. 현재 layer sharing은 계산·binding 재사용은 제공하지만 recipient allocation까지 없애지는 않는다.

### 3.2 TensorRT binding과 in-place update

각 layer의 `past_key_values_i` 입력과 `present_key_values_i` 출력은 동일한 `combinedKV` tensor를 가리킨다.
prefill에서 만든 KV를 decode용 cache로 복사하지 않고 같은 주소에 계속 append한다.

engine profile의 가변 KV length는 이미 확보된 큰 tensor의 활성 view 길이일 뿐이다. 물리 allocation이 요청
길이에 맞춰 증감하는 것은 아니다.

### 3.3 write 위치

RoPE+KV write kernel은 batch index와 현재 logical length를 이용해 연속 cache 주소를 계산한다.

```text
write_start = kv_cache_end_length[batch] - query_sequence_length
write_pos   = write_start + token_offset
cache_addr  = cache_base + batch_index * fixed_batch_stride + head_and_token_offset
```

scheduler가 가진 임의의 global slot ID와 engine batch index가 다르면 현재 kernel은 올바른 slot에 쓸 수 없다.
이 점이 continuous batching과 phase별 독립 batch를 지원할 때 가장 중요한 제약이다.

### 3.4 length와 lifecycle

- prefill 전에는 reuse된 prefix 길이를 GPU length tensor에 넣는다.
- prefill 완료 후 실제 context length를 commit한다.
- decode step마다 slot의 length를 1씩 증가시킨다.
- sequence가 제거되면 `compactBatch()`가 모든 layer의 KV와 length를 앞 slot으로 실제 이동한다.

직렬 실행에서는 단순하지만 decode A가 읽는 중 prefill B가 compaction하면 주소와 데이터가 동시에 바뀐다.
따라서 같은 GPU의 phase 병렬화에서는 compaction을 사용할 수 없다.

### 3.5 현재 존재하는 재사용 특례

- **Cross-layer KV sharing:** `kvSharingDonors`로 recipient layer가 donor layer KV를 읽는다. 같은 request의
  layer 사이 재사용이지 여러 request의 prefix 공유가 아니다.
- **System prompt cache:** prefix KV를 별도 tensor에 복사해 두었다가 새 batch slot로 복원한다. 현재 capture는
  FP16에 제한되고, 여러 request가 같은 주소를 보는 zero-copy sharing은 아니다.
- **Speculative decoding:** 일반적으로 base와 draft가 별도 cache manager를 가진다. Gemma4 MTP assistant는
  target/base KV를 직접 공유하고 own cache를 만들지 않는다.
- **Sliding-window attention:** 읽는 과거 범위는 제한하지만 allocation은 full capacity이다. 오래된 KV를
  반환하거나 ring으로 덮어써서 메모리를 줄이는 구조는 아니다.

## 4. 일반적인 주소 배치 방식

### 4.1 Fixed contiguous cache

현재 방식이다. 주소 계산과 access가 단순하고 기존 TensorRT plugin 계약을 유지할 수 있다. 정적 batch와
예측 가능한 최대 길이를 쓰는 edge deployment에는 잘 맞는다. 반면 최대 batch와 length를 항상 예약하고,
물리 compaction이 concurrent phase와 충돌하며, zero-copy prefix sharing과 eviction이 어렵다.

### 4.2 Stable indexed-linear cache

큰 구조 변경 전에 가장 먼저 도입할 중간 단계다. cache는 slot별 연속 배열로 유지하되 engine batch row와
물리 slot을 분리한다.

```text
engine batch row -> kv_slot_ids[row] -> physical cache slot
```

sequence 종료 시 다른 slot을 복사하지 않고 free-list에 반환할 수 있고, prefill/decode scheduler가 서로 다른
logical batch를 만들 수 있다. paged attention보다 kernel 변경이 작고 sequence 내부 load는 연속이다.

단, slot 하나는 여전히 최대 capacity를 예약한다. arbitrary batch를 지원하려면 attention/KV write plugin에
`kv_slot_ids` indirection이 필요하다. kernel 수정 전 PoC는 batch=1 subview binding이나 연속 slot 묶음으로
제한된다. 이 프로젝트의 intra-GPU E/P/D 분리에는 첫 production 후보로 권장한다.

### 4.3 Paged/block cache

sequence KV를 고정 token 수의 block으로 나누고 global pool에서 필요할 때 할당한다. 각 sequence는 logical
block에서 physical block으로 가는 table을 가진다.

```text
sequence -> [physical block 17, 42, 9, ...]
```

실제 token 수에 가까운 만큼만 할당하고 삭제 시 해당 block만 반환하므로 compaction이 없다. full prefix
block의 refcount 공유, continuous batching, eviction, beam copy-on-write와 잘 맞는다.

대신 attention kernel이 block table을 해석해야 하고 allocator/refcount/partial block 처리가 필요하다. 현재
연속 `past/present` binding 계약도 크게 바뀐다. block이 크면 마지막 block 낭비가, 작으면 lookup과 metadata
overhead가 커진다. 장기적으로 높은 동시성과 prefix reuse가 중요할 때 적합하다.

### 4.4 Virtual-contiguous cache

CUDA VMM으로 sequence마다 큰 연속 virtual range를 예약하고 token이 늘 때 physical page를 map한다. kernel에는
연속 주소처럼 보여 block-table-aware kernel을 피할 수 있다. vAttention 계열의 접근이다.

physical memory의 demand allocation과 contiguous addressing을 함께 얻을 가능성이 있지만 VMM page granularity,
map/unmap latency, CUDA graph, TensorRT pointer/profile 호환성을 실측해야 한다. prefix 공유와 eviction 정책은
여전히 별도다. target edge platform의 driver 기능과 성능도 확인해야 한다.

### 4.5 Circular/ring cache

sliding-window layer에서 capacity를 window size로 제한하고 오래된 token 위치를 modulo로 덮어쓴다. sequence가
길어져도 메모리가 일정하지만 full-attention layer에는 적용할 수 없다. absolute logical position과 physical
position을 분리해야 하며, sliding/full layer가 섞이면 layer마다 backend가 달라진다.

현재 sliding-window의 “계산 범위 제한”에 ring storage를 추가하는 것은 별도 최적화 작업이다.

## 5. 일반적인 재사용·배치 방식

### 5.1 Prefix/radix cache

여러 request의 token prefix가 같을 때 이미 계산한 block을 재사용한다. radix tree나 hash index는 가장 긴
cached prefix를 찾는다. 안전한 cache key에는 최소한 다음 identity가 들어가야 한다.

- model/engine과 weight version
- tokenizer와 정확한 token IDs
- KV dtype 및 quantization scale 방식
- RoPE/position 설정
- LoRA adapter 또는 request별 weight identity
- multimodal embedding 등 prompt를 바꾸는 입력 identity

공유 block은 read-only와 refcount로 관리한다. 공유 prefix 끝의 partial block을 수정하면 다른 sequence가
오염되므로 copy-on-write가 필요하다. 완전히 채워진 block만 global index에 publish하면 규칙이 단순하다.

현재 system prompt의 “저장 후 복사”보다 발전된 방식이며 paged backend와 결합할 때 가장 효율적이다.
indexed-linear에서도 read-only prefix와 private suffix를 나눌 수 있지만 주소 처리가 복잡해진다.

### 5.2 Host offload와 tiered cache

GPU가 부족할 때 cold block을 host memory로 내리고 decode 전에 다시 가져온다. scheduler가 hot/cold와
prefetch deadline을 명시적으로 아는 편이 예측 가능하다.

Edge/SoC에서는 discrete GPU의 PCIe 모델을 그대로 가정하면 안 된다. CPU/GPU가 물리 메모리를 공유해도 page
migration, coherence, bandwidth 경쟁이 생긴다. target에서 block evict/prefetch latency, encoder와 transfer의
간섭, decode deadline 은닉 가능성, pinned/managed/device memory 차이를 측정해야 한다.

offload는 주소 layout과 별개지만 작은 block 단위로 이동 가능한 paged backend에서 구현하기 쉽다.

### 5.3 Disaggregated KV transfer

서로 다른 GPU나 프로세스의 prefill/decode worker라면 KV connector, RDMA/IPC, serialization이 필요하다. 그러나
이 프로젝트의 1차 목표인 **같은 프로세스, 같은 GPU**에서는 prefill 결과를 decode로 복사할 필요가 없다.

```text
{sequence_id, slot_or_block_table, committed_length, ready_event}
```

decode stream은 `ready_event`를 기다린 뒤 같은 주소를 읽는다. prefill 완료 전에 length나 DecodeReady를
publish하면 안 된다. 다른 프로세스로 확장할 때만 CUDA IPC/VMM handle과 process 간 lifetime protocol을
추가한다.

### 5.4 KV quantization/compression

FP8/INT8/low-bit KV는 용량과 bandwidth를 줄이며 layout과 독립적으로 적용할 수 있다. scale granularity와
storage, K/V 분포 차이, kernel dequantization, long-context accuracy, prefix metadata identity가 필요하다.
신규 backend는 현재 runtime이 지원하는 FP16/FP8부터 공통 계약으로 삼고 더 낮은 bit는 별도 기능으로 두는
편이 안전하다.

## 6. 현재 지원 범위

| 기능 | 현재 상태 | intra-GPU E/P/D 관점 |
|---|---|---|
| layer별 연속 cache | 지원 | 기본 storage로 재사용 가능 |
| FP16 / FP8 KV | 지원 | 신규 allocator도 먼저 두 dtype 지원 |
| logical valid length | 지원 | global slot과 phase-local length를 분리해야 함 |
| in-place past/present alias | 지원 | prefill→decode 복사가 필요 없음 |
| batch compaction | 지원 | concurrent phase에서는 금지해야 함 |
| cross-layer donor sharing | 지원 | recipient allocation 제거까지는 못 함 |
| system prompt 저장/복원 | 제한적 지원 | FP16 copy 방식, zero-copy 공유 아님 |
| sliding-window compute | 지원 | 물리 ring/reclaim은 미지원 |
| arbitrary slot indirection | 미지원 | `kv_slot_ids` 필요 |
| paged/block allocator | 미지원 | 장기 backend 후보 |
| cross-request zero-copy prefix | 미지원 | block refcount/COW 필요 |
| host/offload tier | 미지원 | paged 이후 단계가 적절 |
| 같은 GPU phase 간 KV 공유 | 구조상 가능 | 별도 context/I/O와 ownership protocol 필요 |

## 7. 우리 구조에 권장하는 abstraction

```text
KVCacheStorage
  - LinearStorage
  - LinearIndexedStorage
  - PagedStorage          (later)
  - VirtualStorage        (research)

KVSlotAllocator
  - reserve / release / stable slot lease

KVSequenceTable
  - sequence_id -> slot or block table
  - committed_length / phase / ownership

KVCacheView
  - TensorRT context에 binding할 주소와 metadata

KVReuseIndex              (later)
  - prefix lookup / refcount / copy-on-write

KVTierManager             (later)
  - residency / eviction / prefetch
```

storage와 scheduling policy를 분리해야 linear로 먼저 동작시킨 뒤 paged를 추가해도 scheduler lifecycle을
다시 만들지 않는다.

## 8. phase 분리를 위한 KV 상태 기계

```text
Free -> Reserved -> Prefilling -> DecodeReady -> Decoding
     -> ReclaimPending -> Free
```

필수 규칙:

1. `Reserved` 이후 slot/block은 sequence가 끝날 때까지 안정적으로 유지한다.
2. prefill stream의 KV write event가 끝난 뒤에만 `committed_length`와 `DecodeReady`를 publish한다.
3. decode는 ready event를 기다린 뒤 같은 storage view를 읽는다.
4. active reader/writer가 있는 slot은 compaction, eviction, reuse할 수 없다.
5. cancellation도 즉시 free하지 않고 마지막 GPU event 뒤 `ReclaimPending`에서 회수한다.
6. scheduler의 예약 길이와 GPU write가 끝난 committed 길이를 구분한다.

## 9. 현재 kernel에서 생기는 batch 문제

decode batch에 global slot `[7, 2, 11]`을 넣어도 현재 kernel은 engine row `[0, 1, 2]`를 물리 slot
`[0, 1, 2]`로 해석한다. 해결책은 다음과 같다.

1. **Batch=1 subview PoC:** context마다 해당 slot 시작 주소를 batch 1 view로 binding한다. kernel 변경이
   적지만 throughput 실험용이다.
2. **연속 slot lease:** phase batch가 연속 물리 slot을 갖도록 제한한다. admission과 fragmentation이 남는다.
3. **`kv_slot_ids` indirection:** plugin과 read/write kernel이 engine row에서 실제 slot ID를 조회한다.
   production indexed-linear backend에 권장한다.

paged backend에서는 그 다음 단계로 `block_table`과 per-sequence length를 조회한다.

## 10. 권장 구현 순서

### KV0 — invariant와 계측

- layout, length commit, compaction 지점을 테스트로 고정한다.
- NVTX/CUDA event에 `sequence_id`, `phase`, `slot`, `length`를 기록한다.
- system prompt, speculative decode, sliding layer는 첫 concurrent mode에서 비활성화한다.

### KV1 — stable slot, batch=1 phase PoC

- free-list 기반 slot lease를 추가하고 concurrent mode에서 compaction을 끈다.
- prefill/decode 별도 TensorRT context와 I/O를 만들고 slot subview를 binding한다.
- same-GPU handoff는 slot handle + committed length + CUDA event만 전달한다.

이 단계가 “prefill B와 decode A가 같은 GPU에서 겹치는가”를 가장 작은 변경으로 검증한다.

### KV2 — indexed-linear production path

- global slot length 배열과 `kv_slot_ids` binding을 추가한다.
- phase scheduler가 임의 sequence들로 batch를 구성하게 한다.
- attention read, RoPE/KV write, sampling 이후 length commit의 mapping을 일관되게 바꾼다.
- cancellation과 event-delayed reclamation을 구현한다.

### KV3 — paged backend

- block pool, block table, refcount allocator와 block-table-aware attention을 추가한다.
- partial last block과 OOM/admission policy를 구현한다.
- linear backend와 같은 sequence lifecycle API를 유지한다.

### KV4 — prefix reuse

- full block publish, radix/hash lookup, refcount, copy-on-write를 추가한다.
- key에 model/LoRA/RoPE/quantization/multimodal identity를 포함한다.

### KV5 — tier/offload

- cold block eviction과 deadline-aware prefetch를 추가한다.
- target device에서 encoder/transfer/decode의 bandwidth 간섭을 측정한다.

## 11. 선택 기준

- **phase concurrency를 먼저 증명:** stable indexed-linear
- **길이가 예측 가능하고 동시 요청이 적음:** indexed-linear를 최종 구조로 유지 가능
- **continuous batching, 큰 길이 편차, prefix reuse가 핵심:** paged/block
- **연속 kernel을 유지하며 demand allocation 실험:** CUDA VMM
- **sliding layer 메모리만 감소:** layer별 ring backend
- **같은 GPU prefill→decode:** KV 복사 대신 ownership + CUDA event handoff

SM mask/Green Context와 KV layout은 독립적으로 설계한다. SM partition은 kernel 실행 자원을 제한하고 KV
allocator는 주소와 lifetime을 관리한다. scheduler가 둘을 policy로 조합하되 cache correctness가 특정 SM
제어 backend에 의존해서는 안 된다.
