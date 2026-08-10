# Cosmos indexed-paged KV 구현과 10GB GPU 검증

## 결론

Cosmos Reason2-2B text decoder에서 `indexed-paged` KV cache가 실제
`export -> build -> inference` 전 구간으로 연결되었다. 구현은 기본 경로를 바꾸지 않는 opt-in이다.

- `legacy-linear`: logical batch row가 고정 linear KV row를 직접 소유
- `indexed-linear`: request가 stable slot을 소유하지만 slot마다 최대 sequence 전체를 예약
- `indexed-paged`: request가 stable slot을 소유하고, 실제 KV는 필요한 128-token bundle만 전역 pool에서 소유

Cosmos 조건에서 고정 KV allocation은 3,584MiB에서 1,120MiB로 줄었다. 실제 프로세스 peak
VRAM도 9,294MiB에서 6,828MiB로 2,466MiB 감소했다. BS1 decode median은
6.2368ms에서 6.2555ms로 0.30%, p95는 6.2543ms에서 6.2948ms로 0.65% 증가했다.
따라서 현재 단일 real-request decode 결과는 3% 회귀 제한 안이다.

다만 prefill은 측정 sample이 한 번뿐이고, 별도로 빌드한 engine 사이의 greedy output이 긴 출력에서
갈라진다. 그러므로 전체 correctness/performance gate를 통과했다고 선언하지 않는다.

## 실행 구조

```text
prefill/decode logical batch row
                |
                v
       kv_slot_ids[row]
                |
                v
        stable request slot
                |
                v
 kv_page_ids[slot][K/V][logical page]
                |
                v
      physical K/V page ID
                |
                v
layer pool [2 * bundles, 128, Hkv, D]
```

예를 들어 stable slot 3이 bundle `[7, 2]`를 소유하면 page table은 다음과 같다.

```text
slot 3, logical page       0    1
K physical page          14    4
V physical page          15    5
```

K/V는 항상 하나의 bundle transaction으로 함께 할당한다. 모든 attention layer가 같은 bundle ID
순서를 사용하고, 각 layer에는 자신의 page-major storage가 있다.

## 메모리와 파편화

Cosmos 설정은 28 attention layers, `Hkv=8`, `D=128`, FP16 KV,
`maxBatch=16`, `capacity=2048`, page size 128, pool 80 bundles다.

```text
bundle bytes
= 28 layers * 2(K/V) * 8 heads * 128 tokens * 128 dim * 2 bytes
= 14 MiB

indexed-linear
= 16 slots * ceil(2048 / 128) bundles/slot * 14 MiB
= 256 bundles * 14 MiB
= 3,584 MiB

indexed-paged
= 80 bundles * 14 MiB
= 1,120 MiB
```

80 bundles는 slot 수가 아니다. 16 stable slots는 그대로 유지하며, pool은 전체 active request가 함께
쓸 수 있는 10,240 token의 물리 budget이다. 16개 request가 동시에 active라면 평균 640 token까지
보유할 수 있고, 한 request는 최대 2,048 token을 보유할 수 있다. 실제 admission 가능 여부는 현재
active length 분포에 따라 달라진다.

고정 크기 bundle만 free-list에서 할당하므로 외부 파편화는 없다. 물리 page ID가 흩어지는 것은
page table이 해결하므로 연속 공간 확보나 compaction이 필요하지 않다. 내부 파편화는 각 active
sequence의 마지막 page에만 생기며 request당 최대 127 token이다. 16 request의 최악 조건에서도
약 222.25MiB다. 반면 indexed-linear는 각 slot에서 `capacity - actualLength` 전체가 낭비될 수 있다.

page table 자체는 `16 * 2 * 16 * 4 = 2,048 bytes`다. decode enqueue 때 만드는 active-row page view도
최대 2,048 bytes여서 KV pool 크기와 비교하면 무시할 수 있다.

## allocation과 release 순서

```text
phase scheduler
  |
  +-- target length 계산
  +-- host allocator lock
  +-- 필요한 bundle 수를 transaction으로 reserve
  +-- 변경된 stable host page-table row 갱신
  +-- phase CUDA stream에 row H2D 복사
  +-- 같은 stream에서 TensorRT enqueue
  +-- completion CUDA event 확인
  +-- terminal/cancel이면 page bundle 반환
  +-- 마지막으로 stable slot을 admission 가능 상태로 반환
```

H2D source는 함수-local temporary가 아니라 manager lifetime 동안 주소가 유지되는
`mHostKVPageIds`를 사용한다. prefill/decode context가 서로 다른 stream을 사용해도 host allocator와
page ownership 갱신은 하나의 mutex로 직렬화한다.

phase server의 terminal/cancel 경로는 GPU completion event가 끝난 뒤 `onSlotRelease`를 호출한다.
이 callback이 physical page를 먼저 반환하고, 그 다음 logical stable slot allocator가 slot을 반환한다.
device page-table의 사용하지 않는 row는 즉시 읽히지 않으며, 같은 slot이 재사용될 때 새 row upload가
TensorRT enqueue보다 같은 stream에서 먼저 실행된다. ordinary `handleRequest()` batch eviction 경로는
해제한 row를 `-1`로 upload한다.

## 코드 위치

| 역할 | 주요 파일 |
| --- | --- |
| export flag와 runtime config 생성 | `tensorrt_edgellm/scripts/export.py`, `checkpoint/checkpoint_utils.py` |
| 공통 decoder ONNX input | `tensorrt_edgellm/models/default/modeling_default.py` |
| paged attention custom op/translation | `tensorrt_edgellm/models/ops.py`, `onnx/dynamo_translations.py` |
| builder profile와 pool budget | `cpp/builder/llmBuilder.cpp`, `examples/llm/llm_build.cpp` |
| engine/config binding contract | `cpp/runtime/config/llmEngineConfig.cpp`, `cpp/runtime/exec/registryBuilder.cpp` |
| physical layer pool | `cpp/runtime/kvCacheManager.cpp` |
| slot/page ownership과 lifecycle | `cpp/runtime/hybridCacheManager.cpp`, `cpp/runtime/kvPageBundleAllocator.cpp` |
| prefill KV write/gather | `cpp/kernels/posEncoding/applyRopeWriteKV.cu`, `cpp/kernels/contextAttentionKernels/utilKernels.cu` |
| paged XQA decode 연결 | `cpp/plugins/attentionPlugin/attentionPlugin.cpp` |
| phase enqueue reservation | `cpp/runtime/scheduling/phaseBatchState.cpp` |
| terminal page release | `cpp/runtime/scheduling/phaseRequestLifecycle.cpp`, `phaseContextServingFacade.cpp` |

모델별 변경은 없다. Cosmos가 사용하는 default decoder attention graph가 공통 paged op를 생성하고,
Cosmos 특유 deepstack/M-RoPE input은 기존 model adapter 경로를 유지한다. 현재 실험은 vanilla text-only
request이며 visual engine은 Cosmos runtime 초기화 계약을 만족하기 위해 함께 로드했지만 image token이나
deepstack feature는 넣지 않았다.

## opt-in 사용법

export는 두 flag를 함께 사용한다.

```text
tensorrt-edgellm-export ... --indexed-kv-cache --paged-kv-cache
```

build는 stable slot/profile batch와 별도로 physical pool budget을 지정한다.

```text
llm_build \
  --maxBatchSize=16 \
  --maxPrefillBatchSize=8 \
  --maxDecodeBatchSize=16 \
  --maxInputLen=1024 \
  --maxKVCacheCapacity=2048 \
  --kvCachePageBundles=80
```

`paged_kv_cache=false`가 기본값이며, legacy와 indexed-linear engine에는 `kv_page_ids` binding이나 paged
allocation이 추가되지 않는다.

## 검증 결과

실험 장비는 RTX 3080 10GB(SM86), TensorRT 11.0.0.114, CUDA 13.3 환경이다.

### unit/CUDA accuracy

- deterministic allocation/reuse, transaction exhaustion, invalid lifecycle: 통과
- 127/128/129 page boundary와 terminal eviction table: 통과
- multi-head paged RoPE KV write/read round-trip: 통과
- paged XQA headDim 32/64/128/256 reference accuracy, pass rate at `1e-3`: 통과
- custom attention scale paged XQA: 통과
- context attention 8 cases와 phase dispatch regression: 통과

### end-to-end

| case | result | prefill | decode/step | peak VRAM |
| --- | --- | ---: | ---: | ---: |
| indexed-linear, BS1, prompt 43, output 64 | 성공 | 14.9691ms | median 6.2368ms, p95 6.2543ms | 9,294MiB |
| indexed-paged, BS1, prompt 43, output 64 | 성공 | 14.9073ms | median 6.2555ms, p95 6.2948ms | 6,828MiB |
| indexed-paged, BS1, prompt 43, output 160 | 성공, page boundary 통과 | 19.6874ms | median 6.2716ms, p95 6.3130ms | 6,828MiB |
| indexed-paged, BS2, prompt 합 85, output 합 128 | 성공 | 15.3599ms | median 6.2716ms, p95 6.3083ms | 6,828MiB |

64-token BS1 비교에서 indexed-paged는 indexed-linear보다 prefill 0.41% 빠르고 decode median 0.30%,
p95 0.65% 느렸다. 한 번의 prefill 측정으로 속도 개선을 주장할 수는 없지만 paged indirection 비용이
현재 sample에서 3% 이내임은 확인했다.

## 아직 통과하지 않은 gate

1. 별도 tactic으로 빌드한 legacy/indexed/paged engine의 긴 greedy output은 일부 token에서 갈린다.
   paged output은 request에 따라 legacy 또는 indexed output과 일치하는 구간이 달랐다. paged XQA와 KV write의
   numerical unit test는 통과했지만, production gate에는 같은 logits의 단계별 tolerance 비교가 필요하다.
2. prefill median/p95는 workload를 100회씩 3세트 실행해야 한다. 현재 표의 prefill count는 1이다.
3. pool exhaustion은 allocator 수준에서 transactional error로 안전하게 멈추지만, online scheduler가 이를
   queue backpressure metric으로 바꾸는 정책은 아직 없다.
4. deferred free의 event ordering은 phase server completion event 경계에 연결했지만 sanitizer/Nsight로
   use-after-release와 eviction D2D 0 bytes를 다시 확인해야 한다.
5. v1은 FP16 vanilla text attention만 지원한다. prefix sharing/COW, speculative decode, host offload,
   Mamba state paging, image prefill page 분할은 지원하지 않는다.

## 다음 작업 순서

1. engine output logits를 선택한 decode step에서 dump해 indexed-linear와 indexed-paged를
   `atol=1e-2`, `rtol=1e-2`로 직접 비교한다.
2. real-request trace를 반복해 BS1/2/4/8 prefill과 BS1/2/4/8/16 decode의 median/p95 cost table을 만든다.
3. scheduler에 `availableBundles`, dispatch 예상 bundle 수, pool-pressure metric을 노출하고 부족하면
   prefill chunk 또는 admission을 지연한다.
4. compute-sanitizer와 Nsight Systems로 page boundary, terminal reuse, eviction copy 0을 검증한다.
5. 위 gate 통과 후에만 prefix page refcount/COW 또는 adaptive page budget을 설계한다.
