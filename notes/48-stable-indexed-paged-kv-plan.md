# Stable indexed-paged KV cache 설계와 구현 계획

## 결론

현재 `indexed-linear`는 batch eviction 때 KV를 복사하지 않지만, stable slot마다
`maxKVCacheCapacity` 전체를 예약한다. 다음 KV 경로는 stable request slot을 유지하면서
실제 저장 공간만 128-token page bundle의 전역 pool에서 할당하는 `indexed-paged`로 한다.

이 설계는 vLLM PagedAttention의 block-table 원리를 사용하지만 vLLM의 범용 serving
allocator를 그대로 이식하지 않는다. TensorRT Edge-LLM의 기존 XQA layout, 독립
prefill/decode TensorRT context, phase queue와 CUDA-event scheduler를 보존하는 것이
우선이다.

## 세 경로의 차이

| 항목 | legacy linear | indexed-linear | indexed-paged 목표 |
| --- | --- | --- | --- |
| logical row와 KV 위치 | 동일 | `row -> stable slot` | `row -> stable slot -> page bundles` |
| 요청당 물리 예약 | capacity 전체 | capacity 전체 | 실제 길이의 page 수 |
| 중간 eviction KV copy | 필요 | 없음 | 없음 |
| pool 외부 파편화 | 없음 | 없음 | 없음 |
| 내부 낭비 상한 | capacity-actual | capacity-actual | 마지막 127 tokens |
| prefix 공유 | 없음 | 없음 | v1 없음 |

## Ownership와 물리 주소

```text
request / phase row
        |
        v
stable slot lease                  host metadata
        |
        +---- ordered bundle IDs: [7, 2, 11]
                                  |  |   |
global page-bundle pool     ... [2] [7] [11] ...

한 bundle p를 각 attention layer에서 다음 두 XQA physical page로 확장한다.
  K page = 2 * p
  V page = 2 * p + 1

physical layer pool layout:
  [2 * numPageBundles, 128, Hkv, D]  (page-major NHD)
```

bundle은 모든 attention layer의 같은 token 구간에 대한 ownership 단위다. allocator가
layer마다 서로 다른 page ID를 관리하지 않으므로 page table upload와 transaction rollback이
단순해지고, donor-sharing layer도 같은 table을 사용할 수 있다.

## 메모리 계산: Cosmos Reason2-2B

조건은 28 layers, `Hkv=8`, `D=128`, FP16 KV다.

```text
한 128-token page bundle
= 28 * 2(K/V) * 8 * 128 * 128 * 2 bytes
= 14 MiB

기존 indexed-linear
= 16 slots * 2048 tokens/slot
= 256 bundles 상당
= 3584 MiB

초기 indexed-paged 후보
= 80 bundles
= 1120 MiB

고정 KV allocation 감소
= 2464 MiB
```

80은 engine이 지원하는 동시 slot 수가 아니라 memory admission 한도다. 예를 들어 짧은
decode 요청은 16 slots를 모두 사용할 수 있지만, 긴 요청이 많으면 scheduler는 남은 bundle
수를 보고 admission 또는 다음 prefill chunk를 지연해야 한다.

## Allocation transaction

page 할당은 GPU enqueue 전에 coordinator의 단일 ownership 경계에서 수행한다.

1. `targetLength = currentLength + phaseIncrement`를 계산한다.
2. `ceil(targetLength / 128)`과 현재 slot page 수의 차이를 구한다.
3. free-list가 부족하면 어떤 page도 변경하지 않고 dispatch를 보류한다.
4. 충분하면 가장 낮은 free bundle ID부터 stable slot에 붙인다.
5. 변경된 slot page-table row를 phase stream으로 upload한다.
6. stream은 page-table upload event를 기다린 뒤 TensorRT를 enqueue한다.
7. request 완료 후 모든 bundle을 free-list에 반환한다.

Prefill과 decode가 서로 다른 stream/context에서 실행되더라도 한 request slot의 page-table을
동시에 수정하지 않는다. phase coordinator가 slot ownership을 직렬화하고 CUDA event가 host
table update와 kernel read 사이의 lifetime을 보장한다.

## v1 지원 범위

- FP16 attention KV
- vanilla text prefill, fixed-128 chunked prefill, single-token decode
- stable slot eviction/reuse
- 하나의 CUDA context와 독립 prefill/decode TensorRT execution context
- XQA page table과 page-major NHD cache

다음 기능은 page lifetime 의미가 별도로 필요하므로 v1에서 거부한다.

- prefix/system-prompt cache와 refcount/COW
- speculative decoding
- host offload와 page preemption
- Mamba/recurrent state
- image deepstack/M-RoPE prefill의 page 단위 분할

## 구현 단계와 gate

### A. Host ownership foundation

- `KVPageBundleAllocator` 추가
- deterministic free-list, transactional growth, slot release
- `[slot, 2, maxPages]` XQA physical page table 생성
- exhaustion, invalid length, empty-slot release 단위 테스트

### B. Runtime storage와 binding

- engine config에 opt-in `paged_kv_cache=true`, pool bundle 수 추가
- KV layer마다 `[2 * poolBundles, 128, Hkv, D]` 실제 allocation
- logical TensorRT KV binding은 기존 capacity descriptor를 유지하되 실제 pool pointer를 bind
- `kv_page_ids` 또는 동등한 shared page-table input을 ONNX/plugin/registry에 연결
- runtime 시작 시 engine/config binding contract 검증

### C. Kernel 연결

- RoPE KV write가 `slot/token` 대신 page table을 조회
- chunked-prefill cache gather가 비연속 page를 읽음
- XQA decode에는 active row의 page table을 직접 전달
- donor-sharing layer가 동일 page-table ownership을 사용

### D. Scheduler 연결

- prefill chunk/decode enqueue 전 transactional page reservation
- pool 부족을 OOM이 아니라 backpressure/admission metric으로 기록
- request completion에서 page release
- CUDA event 이전에는 page를 재사용하지 않는 deferred free 적용

### E. 검증

- contiguous/indexed-linear와 FP16 결과 비교
- hole mapping `[3, 0, 2]`, page boundary 127/128/129, 중간 eviction
- sanitizer OOB/use-after-release
- eviction D2D bytes 0
- Cosmos 80-bundle 설정에서 KV 1120 MiB와 최소 512 MiB GPU headroom 확인
- indexed-linear 대비 isolated prefill/decode median과 p95 회귀 3% 이내

## 현재 구현 상태

단계 A가 구현되었다. 이 단계는 allocator와 page-table ownership만 제공하며 아직 실제 KV
tensor allocation을 줄이지 않는다. VRAM 감소를 주장하려면 B와 C가 모두 연결된 engine을
`export -> build -> inference` 순서로 검증해야 한다.
