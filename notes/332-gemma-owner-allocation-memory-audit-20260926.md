<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 332. Gemma physical-owner allocation 및 preparation scratch 메모리 감사

## 1. 범위와 결론의 상태

이 문서는 [331 재검증 캠페인](331-runtime-contract-memory-and-full24-revalidation-20260926.md)의
frozen binary/engine으로 72회 HTTP 검증을 진행하는 동안 수행한 **CPU-only, read-only 코드·산출물 감사**다.
이 감사 때문에 production source, 엔진, 실행 중인 workload/config를 변경하지 않았다.
아래 두 개선은 **미구현 제안**이고, 절감량은 코드와 실제 artifact shape에서 계산한 값이다.
새 GPU 실험, 성능 측정, sanitizer 통과 또는 promotion을 의미하지 않는다.

| 다음 후보 | 계산상 절감 가능량 | 무엇을 유지하는가 | 아직 필요한 증명 |
|---|---:|---|---|
| Gemma donor KV의 physical-owner allocation | **576 MiB** | 192 pages, 128 tokens/page, FP16, D24/P8, 최대 길이 2048 | owner mapping·복사 경로·prefix semantics·정확성·실제 VRAM |
| 실제 이미지 크기에 따른 grow-only resize scratch | 현 12-workload 이미지 집합에서 **약 72.07 MiB** | 입력 이미지 지원 한도와 resize 연산, KV 전체, E+D/P+D 가능성 | allocation lifetime·growth 비용·allocator reserved memory·성능 |

첫 번째는 **사용 가능한 KV 용량을 줄이는 제안이 아니다.** 모델이 이미 donor cache를 읽는 layer에
별도로 확보해 둔 사용되지 않는 pool을 제거하는 것이다. 두 번째도 benchmark 이름으로 작은 image cap을
선택하는 방식이 아니라, 실제 요청의 이미지 치수를 보고 필요한 임시 저장 공간을 확보하는 방식이다.

코드 감사 시점 HEAD는 `ff75f71caa2528a899c3952e29601b39d052ff64`다. 실행 파일의 provenance는
각 campaign manifest를 따른다. 현재 checkout HEAD가 기존 binary의 build commit이라는 뜻은 아니다.

## 2. 사용한 증거

공통 결과 루트는 `.local/results/runtime-contract-revalidation-20260926/`다.

- `workspace-rebuild/gemma/build.log`, `workspace-rebuild/gemma/manifest.json`
- `production-final/gemma-mixed-graphs0.log`: 이전 Gemma plan을 사용한 production 확인
- `production-safety/gemma-mixed-graphs0.log`: workspace 수정 후 새 Gemma plan을 사용한 확인
- `full24-final-3x/gemma/shared_ep-predictor-on/repeat-001/{balanced,mixed,vision-heavy,multi-image}/`
  아래 `contract.json`, `run-001/gateway.log.gz`의 startup/최종 메모리 로그와 `PHASE_METRIC`
- `.local/results/gemma4-v3-capacity-ab-20260913/build-p192.log`, `build-manifest.json`
  — 과거 lineage 보조 증거이며 현재 보존된 old plan의 완전한 provenance가 아님
- `.local/artifacts/v0101-forward-port/workspace-corrected-20260926/gemma/`의 config와 safetensors headers
- `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-packed-p8-d24-kv2048-p192/`의 기존 config/plan
- `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/visual-e4-soft280/visual/config.json`
- `.local/results/gemma4-e2b-awq-full12-20260911/inputs/`에서 참조하는 실제 이미지 파일의 JPEG header 치수

현재 full24 결과는 감사 시점에 진행 중이었다. 이 문서의 일부 로그 확인을 72/72 완료 또는 전체 품질
통과로 확대 해석하지 않는다. 최종 실험 결과와 headroom 판정은 331의 최종 보고서를 따른다.

## 3. fresh plan 이후 약 70 MiB 증가의 위치

| 항목 | 기존 plan | workspace-corrected plan | 해석 |
|---|---:|---:|---|
| 보존된 `llm.engine` 크기 | 1,388,545,260 B | 1,461,395,724 B | +72,850,464 B = **69.475 MiB** |
| 첫 LLM context 생성 시 TRT-managed GPU 로그 | 1290 MiB | 1360 MiB | vision 적재 전부터 약 70 MiB 차이 |
| vision 적재 후 TRT-managed GPU 로그 | 1612 MiB | 1682 MiB | vision 증가분은 양쪽 모두 322 MiB |
| P activation | 183,647,744 B | 183,647,744 B | 변화 없음 |
| D activation | 28,401,664 B | 28,401,664 B | 변화 없음 |
| P Max Scratch | 55,193,600 B | 55,193,600 B | 변화 없음 |
| D Max Scratch | 25,387,520 B | 25,387,520 B | 변화 없음 |
| `config.json` | 기준 파일 | byte-identical | KV/capability 설정 변경 아님 |

새 build의 `Total Weights Memory`는 **1,426,686,436 B**다. 증가가 발생한 위치는 LLM plan의 상주
weight/layout 저장 영역이며, KV page 증가나 독립 context activation 증가로 설명되지 않는다.
정확히 어떤 TensorRT tactic 또는 weight reformat/packing이 추가됐는지는 현재 로그만으로 증명하지 못했다.

과거 p192 build log는 weights **1,353,548,132 B**를 기록해 차이 69.750 MiB로 runtime 관측을
뒷받침한다. 그러나 그 manifest의 `engine_bytes=1,538,544,564`는 현재 보존된 old plan의 크기와 다르다.
따라서 과거 build의 weight 숫자를 현재 old plan의 정확한 내부 layout이라고 단정하면 안 된다.
기록된 과거 source `a516dcdc194b44ea7cddd2ae08c84f1f1b2d4ca3`부터 감사 시점까지 LLM builder,
builderUtils, INT4 plugin source 차이는 없고 attention contract는 달라졌다. 새 autotuning도 수행됐다.
이 사실은 구체적 tactic 차이의 원인을 확정하는 증거는 아니다.

이후 실험에서 old/new plan을 섞어 비교한다면 binary/plugin뿐 아니라 engine SHA도 분리해야 한다.
correctness를 위해 새 plan을 사용한 결과의 메모리 비용을 숨기거나 이전 plan의 footprint로 대신하면 안 된다.

## 4. 현재 메모리 구조: 중복 여부와 줄일 수 있는 영역의 구분

아래는 완전한 NVML 총량 분해가 아니라, 확인 가능한 부분의 목록이다. 파일 tensor payload, TRT 보고값,
코드상 예약량의 측정 기준이 다르므로 전부 더해서 정확한 process peak라고 표시하지 않는다.

| 영역 | 확인한 크기/상태 | 판단 |
|---|---:|---|
| 새 LLM engine weights | 1,426,686,436 B | P/D마다 engine을 다시 적재하는 구조 아님 |
| PLE table | **4480 MiB**, `[262144,8960]` FP16 | P/D가 같은 `shared_ptr` table 사용 |
| 입력 embedding table | **768 MiB**, `[262144,1536]` FP16 | phase별 table 복제 아님 |
| external INT4 FFN payload | **765.703125 MiB**, 210 tensors | 한 manager가 P/D tensor map에 등록 |
| KV physical pools | **1008 MiB** 계산 | donor routing과 allocation의 불일치가 남아 있음 |
| shared E/P activation arena | **299.00390625 MiB** | `max(E,P)`, 두 영역의 합이 아님 |
| 별도 D activation | **27.0859375 MiB** | E+D/P+D를 위해 유지해야 함 |
| P PLE output | **17.5 MiB**, `[35,1,1024,256]` FP16 | shared table과 다른, phase-local 결과 |
| D PLE output | **0.41015625 MiB**, `[35,24,1,256]` FP16 | P와 overlap 중 동시에 필요 |
| resize raw/intermediate scratch | **95.062122 MiB** | 최악 입력 크기로 항상 예약됨 |
| Gemma retained vision output slab | 최대 **3.28125 MiB**, `[1120,1536]` FP16 | 현재 primary는 최대 한 retained batch |
| logits | P 8 MiB + D 24 MiB | 서로 다른 phase 결과, 임의 alias 불가 |
| compacted P sampling logits | 추가 **8 MiB** | 더 작은 후속 최적화 후보; 두 주요 제안에는 포함하지 않음 |
| CUDA graph cache | 관측 D entries 24, P entries 0 | entry별 GPU memory byte 계측 없음 |

P/D engine sharing은 [engineExecutor.cpp](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/exec/engineExecutor.cpp:191)의
`SharedEngineState`와 `createSibling()`으로 확인할 수 있다. context는 별개지만 engine/weight storage는 같다.
PLE 공유와 별도 결과는 [gemma4EmbeddingPreprocessor.cpp](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/preprocess/gemma4EmbeddingPreprocessor.cpp:70),
[smoke startup](/home/sslab/TensorRT-Edge-LLM/examples/llm/llm_phase_context_smoke.cpp:835)에서 확인된다.

단, **P/D 사이 weight 공유와 embedding↔LM-head tied weight 공유는 다른 문제다.** 현재 Gemma config의
`external_weight_files`에는 INT4 FFN만 있다.
[requiresTiedEmbedding()](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/state/externalWeightManager.cpp:921)는 이 config에서
false이므로 현재 engine이 입력 embedding과 내부 LM-head weight까지 deduplicate했다고 주장하지 않는다.
내부 LM-head/tactic별 추가 중복은 이 감사에서 계측하지 않았고, 별도의 절감량으로 합산하지 않았다.

CUDA graph는 user-managed activation과 binding 주소를 캡처한다.
[captureTRTCudaGraph()](/home/sslab/TensorRT-Edge-LLM/cpp/common/trtUtils.cpp:314)에는 graph마다 새로운 전체
P/D workspace를 만드는 코드가 없다. 따라서 `24 graphs × 전체 context workspace` 같은 계산은 틀리다.
CUDA/TRT 내부 graph memory와 allocator reservation은 별도 before/after 측정 전에는 미분류로 남긴다.

## 5. shared E/P, single storage, bounded-two storage는 서로 다른 개념이다

[configureSharedVisionContextMemory()](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/independentEngineExecutorPair.cpp:238)는
E와 P가 같은 base address의 activation arena를 사용하도록 한다. D는 다른 allocation이다.

```text
Activation/workspace                         Persistent encoder outputs

shared arena: [ E 또는 P ]                  slab A: E1이 생성 → P1이 소비
separate D:  [     D     ]                  slab B: 다음 E2 결과용 (bounded-two일 때)

가능: E+D, P+D                              A의 P consumer가 끝나기 전 A 재사용 금지
불가능: E+P, E+P+D                          A와 B는 서로 다른 output allocation
```

**두 slab을 허용해도 full E/P overlap이 복구되지 않는다.** 이 옵션은 encoder output의 lifetime과
다음 encoder preparation/admission을 더 느슨하게 할 뿐, shared activation의 E/P 상호 배제는 유지한다.
따라서 이 구조의 연구/성능 주장은 E+D 및 P+D 가능성과 full independent E/P/D를 구분해야 한다.

설정은 [phaseThreeCoordinator.cpp](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseThreeCoordinator.cpp:899)에서
`TRT_EDGELLM_SHARED_EP_SINGLE_STORAGE=1`이면 한 batch, `0`이면 두 batch로 해석한다.
primary full24의 확인한 Gemma 로그는 모두 `serialized=1 max_retained_batches=1`이다.
별도의 two-slab screen과 혼동하지 않는다.

실제 output ownership의 안전장치는 다음과 같다.

1. [phaseVisionPreparationWithinStorageBudget()](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseVisionAdapter.h:113)는
   shared workspace 조건에서 retained batch 수를 제한한다. 한 slab 모드는 downstream pending도 기다린다.
2. [acquireBatchStorage()](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseVisionAdapter.cpp:324)는
   `use_count()==1`, 즉 pool 자체만 가진 storage만 재사용한다. 다른 request가 소유한 A를 B로 덮어쓰지 않는다.
3. [Gemma runner](/home/sslab/TensorRT-Edge-LLM/cpp/multimodal/gemma4/gemma4ViTRunner.cpp:696)는 내부 output을 해제하고
   request-owned storage를 직접 bind한다.
4. [submitPrepared()](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseVisionAdapter.cpp:555)는 그 pointer로
   E를 실행하고, 각 payload는 같은 batch storage의 row view와 `shared_ptr` owner를 유지한다.
5. 준비/encoder 완료 event가 prerequisite이고, P consumer 완료 후
   [releasePrefillStorage()](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/independentPhaseAsyncServer.cpp:2291)로
   해당 ownership을 놓는다. cancel 경로도 현재 lifecycle fix의 completion 경계를 지켜야 한다.

Primary repeat001에서 확인한 최종 로그는 다음과 같다. 누적 bytes이지 동시에 사용한 peak가 아니다.

| Workload | slab allocations/reclaims | reuses | 누적 direct output bytes | output D2D | idle slabs |
|---|---:|---:|---:|---:|---:|
| balanced | 11 / 11 | 0 | 11,182,080 | 0 | 0 |
| mixed | 27 / 27 | 0 | 41,441,280 | 0 | 0 |
| vision-heavy | 27 / 27 | 0 | 56,549,376 | 0 | 0 |
| multi-image | 18 / 18 | 0 | 30,277,632 | 0 | 0 |

따라서 이 primary의 near-OOM을 "사용하지 않는 vision slab이 수백 MiB 쌓여서"라고 설명할 근거는 없다.
반대로 매번 reclaim하는 allocation churn은 별도 성능 이슈가 될 수 있지만, 한 slab 보존으로 증가하는
Gemma payload 자체는 몇 MiB 수준이다. Cosmos의 M-RoPE/deepstack lifetime에는 같은 숫자를 적용하지 않는다.

## 6. 제안 A: 논리 KV layer와 물리 owner를 분리

### 6.1 현재 불일치와 정확한 계산

[SharedResources::createForLLM()](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/state/sharedResources.cpp:107)는
35개 `kvLayerConfigs`를 그대로 `KVCacheManager::Config`에 전달한다. 현재 Config에는 donor map이 없다.
[KVCacheManager constructor](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/kvCacheManager.cpp:83)는 35개 모두에
`[2,192,128,Hkv,D]` Tensor를 생성한다.

하지만 [pipelineIO.cpp](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/state/pipelineIO.cpp:204)의 binding은
`kvSharingDonors[layer] >= 0`이면 해당 donor의 pool을 사용한다. Gemma config는 다음과 같다.

```text
logical layers 0..14:  owner = 자기 자신
logical layers 15..34:
  sliding 16개: owner 13, D=256
  global   4개: owner 14, D=512
Hkv = 1, dtype = FP16, pages = 192, page tokens = 128
```

```text
현재 전체 할당
  (28×256 + 7×512) × 192 × 128 × 2(K/V) × 2(FP16 bytes)
  = 1,056,964,608 bytes = 1008 MiB

15개 실제 owner만 할당
  (12×256 + 3×512) × 192 × 128 × 2 × 2
  = 452,984,832 bytes = 432 MiB

borrower 20개 private allocation 제거분
  (16×256 + 4×512) × 192 × 128 × 2 × 2
  = 603,979,776 bytes = 576 MiB
```

이 숫자는 실제 buffer allocation 코드의 payload 계산이지 측정한 NVML 절감량은 아니다.
allocator 정렬/driver/runtime 차이 때문에 process peak 감소량이 정확히 576 MiB일지는 검증해야 한다.
페이지 수를 192 아래로 줄이지 않으며, cache format, ownership lease, page IDs, context length도 유지한다.

### 6.2 caller 감사: 단순 alias 패치가 충분하지 않은 이유

| 경로 | 현재 동작 | owner-aware 변경 시 필요한 사항 |
|---|---|---|
| Vanilla phase P/D binding | donor13/14로 이미 routing | 기존 engine 입력 pointer semantics 유지 |
| `getCombinedKVCache`, `kPoolPtr`, `vPoolPtr`, `getSeparateKVCache` | logical index의 별도 Tensor 반환 | 모든 accessor를 canonical owner로 해석; donor와 shape/dtype 일치 검사 |
| Hybrid head-dim groups | 모든 logical layer를 copy/compact info에 등록 | **physical owner를 한 번만 등록**; 같은 pointer의 중복 쓰기 금지 |
| Legacy batch compaction | head-dim group에 대해 in-place kernel 실행 | alias 중복이 있으면 동일 주소를 여러 block이 갱신할 수 있어 반드시 제거 |
| Legacy system-prompt capture/restore | logical layer마다 별도 저장/복사 | public logical indexing과 물리 owner 저장을 구분; owner 한 번만 save/restore |
| Modern hybrid partial-KV snapshot | logical layer loop로 K/V 복사 | snapshot layout/shape validation과 restore 중복을 함께 정리 |
| Modern attention prefix reuse | page lease/reference 중심 | owner map 때문에 page ID나 reference count를 변경하지 않음 |
| Gemma MTP draft target mapping | target absolute layer의 cache accessor 사용 | absolute→local→canonical owner mapping 일관성 필요; 별도 MTP 검증 없이는 지원 주장 금지 |
| DSpark/DFlash draft, speculative acceptance | direct accessor 또는 head-dim groups 사용 | 비공유 모델 identity mapping 유지; speculative write group 중복 방지 |
| Layer debugger | separate K/V view 요청 | logical borrower를 보더라도 실제 donor 내용을 보여줘야 함 |
| Alpamayo action gather | layer별 combined cache gather | read-only 중복은 성능 문제; index/shape 의미 유지 |
| Qwen Omni TTS reset | layer별 combined cache memset | 현재 non-sharing identity 유지; 추후 공유 모델이면 owner별 한 번만 초기화 |
| Mamba recurrent/conv | 별도 manager/absolute-to-Mamba mapping | 변경하지 않음 |

정확한 코드 위치:

- [KV accessors](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/kvCacheManager.cpp:138)
- [Hybrid group 구성](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/hybridCacheManager.cpp:84)
- [Hybrid absolute→local accessor](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/hybridCacheManager.cpp:181)
- [Legacy compact](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/hybridCacheManager.cpp:351)
- [Legacy capture](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/hybridCacheManager.cpp:401),
  [restore](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/hybridCacheManager.cpp:450)
- [Snapshot layout validation](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/state/contextCache/hybridSnapshotStorage.cpp:134),
  [partial capture](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/state/contextCache/hybridSnapshotStorage.cpp:351),
  [partial restore](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/state/contextCache/hybridSnapshotStorage.cpp:377)
- [Gemma MTP target binding](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/state/pipelineIO.cpp:421)
- [DSpark draft](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/decoding/dsparkDecoder.cpp:198),
  [DFlash draft](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/decoding/dflashDecoder.cpp:136)
- [Layer debugger](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/debug/layerDebugger.cpp:107)
- [Alpamayo gather](/home/sslab/TensorRT-Edge-LLM/cpp/action/alpamayo1ActionRunner.cpp:294)
- [TTS resets](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/qwen3OmniTTSRuntime.cpp:4267)

현재 phase-serving source에서 `HybridCacheManager::compactBatch()` 호출은 발견하지 못했다.
`resetForNewSequences()`는 주로 length bookkeeping을 초기화하며 private borrower data를 생산하는 경로가 아니다.
이 때문에 current vanilla 실행에 대해 20개 private pool이 binding상 불필요하다는 결론은 강하다.
그러나 공통 manager를 바꾸면서 legacy/prefix/speculative 경로를 자동으로 안전하다고 가정하면 안 된다.

이번 getter/copy caller 검색에서는 별도 KV host-offload implementation을 발견하지 못했다.
이것을 저장소의 모든 가능한 offload 기능 부재에 대한 증명으로 쓰지 않는다. 향후 host snapshot/offload가
동일 accessor를 사용한다면 serialization format에 logical→owner map을 반영하고 중복 physical copy를 제거해야 한다.

Memory broker의 byte accounting도 함께 감사해야 한다.
[smoke 설정](/home/sslab/TensorRT-Edge-LLM/examples/llm/llm_phase_context_smoke.cpp:2711)은 현재
`numAttentionLayers × headDim` 형태로 page bytes를 계산하는 경로를 포함한다. Gemma의 heterogeneous head
크기 및 owner sharing을 반영한 **unique-owner 실제 allocation bytes**를 공통 API로 제공하는 편이 안전하다.
현재 실험의 broker가 비활성이어도 이후 memory-pressure 정책의 과대/과소 계산을 남기면 안 된다.

### 6.3 구현 전 정할 invariant

- logical attention index와 absolute decoder index를 바꾸지 않는다.
- owner map은 범위, cycle, donor shape/dtype compatibility를 검증한다. non-sharing 모델은 identity map이다.
- donor chain을 허용한다면 canonical root를 계산하고 allocator/getter/copy 모두 같은 root를 사용한다.
- logical layer 수와 physical owner 수를 별도 API/telemetry로 제공한다.
- allocation/free, compaction, snapshot restore는 owner마다 정확히 한 번 수행한다.
- owner lifetime은 engine/graph/모든 GPU consumer보다 길다. 요청마다 KV pointer를 바꾸지 않는다.
- page pool 및 stable lease는 현재와 동일하며 borrower 수만큼 page를 추가/감소시키지 않는다.
- 첫 구현을 vanilla에 한정하면 다른 모드는 명시적으로 기존 동작을 유지하거나 fail-closed한다.

## 7. 제안 B: 이미지 preparation scratch의 실제 수요 기반 성장

### 7.1 예약량과 실제 요청 치수

[allocateBuffer()](/home/sslab/TensorRT-Edge-LLM/cpp/multimodal/gemma4/gemma4ViTRunner.cpp:284)는 현재:

```text
raw: 4096 × 4096 × 3 × uint8                         = 48.000000 MiB
tmp: floor(sqrt(maxImagePixels × 4096²) × 1.25) × 3 × float32
     maxImagePixels = 280 × 3² × 16² = 645120        = 47.062122 MiB
합계                                                = 95.062122 MiB
```

[copyImageToDeviceAndResize()](/home/sslab/TensorRT-Edge-LLM/cpp/kernels/preprocessKernels/imageUtilKernels.cu:841)의
실제 scratch 요구량은 한 frame 기준 `rawH×rawW×channels` bytes와
`rawH×outW×channels×sizeof(float)`다. frame은 같은 stream에서 순차 처리한다.
[resize shape 함수](/home/sslab/TensorRT-Edge-LLM/cpp/multimodal/common/imageUtils.cpp:279)를 적용하면:

| 이미지 | 원본 W×H | resized W×H | raw MiB | tmp MiB | 합계 MiB |
|---|---|---|---:|---:|---:|
| `database_er.jpeg` | 1790×294 | 1968×288 | 1.506 | 6.621 | 8.127 |
| `giant_panda.jpeg` | 1000×1000 | 768×768 | 2.861 | 8.789 | 11.650 |
| `red_panda.jpeg` | 1000×747 | 912×672 | 2.137 | 7.796 | 9.934 |
| `woman_and_dog.jpeg` | 2048×1365 | 960×624 | 7.998 | 14.996 | 22.994 |

이 네 파일은 현재 Gemma full12 input JSON이 참조하는 이미지 집합이다. 그 집합의 high-water 필요량은
22.994385 MiB이므로 **95.062122−22.994385 = 72.067738 MiB**가 계산상 예약 여유다.
미래 요청, 다른 calibration 이미지 또는 더 큰 원본에도 같은 절감이 유지된다고 주장하지 않는다.

### 7.2 lifetime 및 graph 제약

preparation의 raw/tmp는 persistent vision output slab, KV pool, TRT activation arena와 별개의 버퍼다.
따라서 image scratch만 재설계하면서 P/D execution context나 graph의 activation pointer는 유지할 수 있다.

- 요청의 실제 raw/out 치수를 CPU에서 이미 알고 있으므로 필요한 byte 수를 계산할 수 있다.
- 현재 용량으로 충분하면 재사용하고, 부족할 때만 grow한다. workload 이름이나 특정 JPEG 이름으로 분기하지 않는다.
- 기존 최대 raw dimension 검사는 유지한다. 이미지 지원 한도를 작게 바꾸어 절감한 것처럼 표시하지 않는다.
- resize/upload/patchify의 마지막 GPU consumer가 끝나기 전 old allocation을 반환하면 안 된다.
- 현재 단일 encoder/preparation ownership과 event 완료 경계를 사용한다. E idle만으로 다른 copy consumer까지
  완료됐다고 추정하지 않는다.
- 동기 `cudaMalloc`/`cudaFree`로 매 요청 D를 중단시키면 안 된다. bounded amortized growth 또는 명시 stream의
  stream-ordered allocation과 completion-aware retirement가 필요하다.
- `cudaMallocAsync`를 사용해도 allocator reserved bytes가 계속 남을 수 있다. live allocated와 pool reserved,
  NVML peak를 각각 계측해야 실제 절감을 입증할 수 있다.
- P/D graph는 이 preparation scratch를 참조하지 않는다. 향후 preparation/E graph를 도입하면 pointer 변경 시
  해당 graph invalidation/recapture 정책을 별도로 추가해야 한다.
- 급격히 큰 이미지가 온 경우 growth와 동시 D memory 사용의 feasibility를 검증하고, OOM 대신 backpressure 또는
  admission 대기를 명확히 구현해야 한다. 이것은 batch/page capacity 축소와 다른 mechanism이다.

이 제안은 shared E/P를 E+P로 바꾸지 않으며, 현재 가능한 E+D/P+D를 금지하지도 않는다.
다만 실제 allocation overhead/stream dependency가 overlap 성능을 저하시키지 않는지는 측정해야 한다.

## 8. 우선순위를 낮춘 후보와 하지 않을 것

- P/D activation을 하나로 합치면 27 MiB 정도를 줄일 여지가 있어도 independent overlap을 잃는다. 추천하지 않는다.
- PLE table 양자화·모델 변경·KV dtype 변경·page 수 축소는 이 감사의 제안이 아니다.
- 출력 slab은 이미 direct output이며 idle 0이다. 여기서 수백 MiB 절감을 기대하지 않는다.
- Graph 개수를 먼저 줄이면 decode latency를 잃을 수 있다. graph byte attribution 없이 주요 절감 대상으로 삼지 않는다.
- Compacted P logits는 Gemma P8에서 추가 8 MiB다.
  [production sampling](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseServingRuntime.cpp:448)과
  [smoke sampling](/home/sslab/TensorRT-Edge-LLM/examples/llm/llm_phase_context_smoke.cpp:1312)에 별도 allocation이 있다.
  추후 row-indexed argmax로 float logits 복사를 피할 수 있지만, row order/tie-breaking/partial-P completion
  semantics 검증이 필요하다. 우선은 큰 두 후보를 분리 검증한다.
- 내부 tied LM-head/weight-layout 중복은 별도 export/build artifact 비교를 거쳐야 한다. 현재 증거로 추가 768 MiB
  절감을 확정하거나 두 제안의 절감량에 더하지 않는다.

## 9. 후속 구현·검증 계획

1. 현재 72-run을 동일 binary/engine/config로 완료하고 331에 결과를 고정한다. 이 문서는 미래 계획으로 연결한다.
2. owner mapping의 CPU contract와 physical-byte 계산 테스트를 먼저 추가한다. Gemma expected owners=15,
   logical layers=35, bytes=432 MiB, 제거분=576 MiB; Cosmos는 identity/no-change를 확인한다.
3. KV getter/owner iteration과 Hybrid copy group을 함께 수정한다. alias만 추가하는 최소 패치는 하지 않는다.
   snapshot·compaction·restore·debug·MTP 등 지원 범위를 명시한다.
4. fresh process에서 engine 동일, 페이지 수 동일, input/row order 동일 조건으로 allocation before/after를 측정한다.
   초기 KV 할당은 graph capture 이전에 끝내고 graph pointer lifetime을 보장한다.
5. Gemma P1/P8, D1/D24, mixed d256/d512, 중간 eviction, donor layers, reuse, cancel/drain, graph0/1을 검증한다.
   exact greedy identity와 실제 KV address alias를 확인하고 memcheck를 별도 실행한다.
6. legacy compact/prefix snapshot 경로를 지원한다면 duplicate-owner write를 검증하는 GPU 테스트를 추가한다.
   지원하지 않으면 opt-in vanilla 범위를 벗어날 때 명시적으로 거부/기존 경로 유지한다.
7. Cosmos identity mapping smoke로 공통 allocator 회귀가 없음을 확인한다. KV precision/pages는 변경하지 않는다.
8. 별도 변경으로 resize scratch high-water/actual/reserved memory telemetry를 넣고 grow-only 경로를 구현한다.
   작은→큰→작은 이미지, 여러 frame, cancel, two-slab preparation, copy event, D 동시 실행을 검증한다.
9. 현 이미지 집합과 최대 지원 이미지에 대해 VRAM·growth latency·host gap·E/D/P mask를 측정한다.
   계산상 72 MiB와 실제 peak 절감이 다르면 allocator/timing 차이를 설명한다.
10. 두 변경 각각에 대해 12-workload request throughput/token throughput, TTFT mean/p95, TPOT mean/p95,
    E2E mean/p95, output 품질, peak/free memory를 비교한다. 두 변경을 먼저 합쳐서 원인을 섞지 않는다.
11. 동일 serving contract면 frozen vLLM 결과를 재사용한다. 요청/출력 길이·batch capacity·image preprocessing
    의미가 바뀌지 않았음을 manifest에 기록한다. 비교 조건이 바뀌면 새 vLLM baseline이 필요하다.

최종적으로 주장할 수 있어야 하는 것은 “캐시를 작게 해서 들어갔다”가 아니라 다음이다.

> 같은 모델, 같은 KV 수용량, 같은 요청과 phase execution capability에서 실제 data owner와 lifetime에 맞춰
> physical allocation을 정리했고, 정확성과 latency/throughput을 유지하면서 GPU headroom을 확보했다.
