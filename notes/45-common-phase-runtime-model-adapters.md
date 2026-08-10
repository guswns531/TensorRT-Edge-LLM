# 공통 phase runtime과 모델별 adapter 분리 설계

## 목표

prefill/decode queue, dynamic batching, adaptive chunking, CUDA event 계측,
독립 TensorRT context, CUDA stream/SM backend는 모델과 무관한 공통 runtime으로
유지한다. 모델별로 다른 부분은 export된 TensorRT I/O 계약, encoder 출력,
embedding assembly, position encoding, PLE/deepstack 처리로 한정한다.

## 경계

```text
request
  -> common admission / phase queues
  -> ModelPhaseContract capability checks
  -> model adapter (tokens, embeddings, deepstack, M-RoPE)
  -> common TensorRT phase executor
  -> common KV ownership / metrics / completion
```

### 공통 runtime

- `IndependentEngineExecutorPair`: 하나의 CUDA primary context에서 prefill/decode
  TensorRT execution context와 context workspace를 분리한다.
- `PhaseQueueScheduler`, `PhaseDispatchWorker`, `PhaseThreeCoordinator`:
  prefill/decode/encoder queue, batch selection, overlap policy, admission 순서를
  모델 이름 없이 처리한다.
- `PhaseKernelGroupRecorder`: CUDA event로 phase kernel group의 duration과
  dispatch composition을 기록한다.
- `PhaseAsyncServer`: submit/poll/completion API와 request lifecycle을 소유한다.
- `HybridCacheManager`와 `PhaseBatchState`: indexed cache에서는 stable slot과
  eviction 없는 KV ownership을, legacy linear cache에서는 fixed row만 제공한다.
- SM control은 `Noop`/Green Context/연구용 SM-control 구현을 선택하는 backend로
  둔다. scheduler는 SM API를 직접 호출하지 않는다.

### 모델/engine contract

`ModelPhaseContract`는 engine config와 adapter가 phase runtime에 제공하는
capability 집합이다.

```text
indexedKVCache
hasPLE
numDeepstackFeatures
hasMRope
supportsChunkedPrefill
requiresAtomicMultimodalPrefill
supportsDynamicAdmission
maxPrefillChunkTokens
```

queue scheduler는 Gemma/Cosmos를 비교하지 않고 이 capability만 검사한다.

### 모델별 adapter

- Gemma4: visual embedding 1개, PLE table/output, dual RoPE, vision-block 제약.
- Qwen3-VL/Cosmos: main visual embedding, raw deepstack feature 3개,
  M-RoPE cos/sin, image-token expansion.
- 기타 VLM: `MultimodalRunner`의 공통 virtual API를 구현하되, phase adapter는
  encoder 결과의 lifetime과 phase TensorMap binding을 책임진다.

## KV 정책

indexed engine은 `kv_slot_ids`를 통해 logical row와 physical KV row를 분리한다.
따라서 batch eviction에서 metadata만 compact하고 KV D2D copy를 하지 않는다.
non-indexed engine은 logical row가 physical row이므로 fixed microbenchmark만
허용한다. production dynamic admission/eviction은 indexed engine 재-export가
필수다.

## 실제 구현 순서

1. capability contract를 추가하고 `llm_phase_bench`와 scheduler safety check가
   이를 사용하도록 한다.
2. 기존 `Gemma4PhaseVisionAdapter`의 lifecycle을 generic vision adapter 경계로
   옮긴다.
3. `Qwen3VLPhaseVisionAdapter`를 추가하여 encoder main embedding, deepstack
   feature, M-RoPE를 request-owned GPU storage로 복사한다.
4. packed prefill adapter와 phase TensorMap에 deepstack/M-RoPE를 전달한다.
5. PLE가 없는 모델의 decode callback을 허용하고, 모델 adapter가 필요한
   embedding preparation을 등록하게 한다.
6. Gemma indexed, Cosmos fixed text, Cosmos image trace를 각각 correctness와
   CUDA event 성능 smoke로 검증한다.

## 현재 구현/검증 상태

- 1--5는 `llm_phase_bench`와 공통 packed adapters에 연결되어 있다. Qwen3-VL
  adapter는 encoder 결과를 request-owned GPU Tensor로 복사한 뒤 completion
  event에서 prefill로 넘기며, M-RoPE batch는 v1에서 한 요청씩만 허용한다.
- TensorRT 11.0/CUDA 13.3 컨테이너에서 `llm_phase_bench` 전체 타깃 빌드와
  Cosmos FP16 text smoke를 통과했다. 새 Cosmos image trace는 코드 경로가
  준비됐지만 현재 체크포인트의 `indexed_kv_cache=false` engine이 stable
  admission/eviction 계약을 만족하지 않아 실행 전에 거부된다.
- 실제 image trace를 열려면 Cosmos decoder를 `indexed_kv_cache=true`로
  재-export/build하고, `num_deepstack_features=3` 및 M-RoPE binding이 같은
  config/engine에 존재하는지 확인한다. 그 뒤 Gemma와 동일한 JSON trace를
  independent mode로 실행한다.
- 공통 `unitTest`는 637개 중 597개 통과, 39개 skip였고 기존
  `InitializeMRopeCosSin.Accuracy` 1건이 CUDA 13.3 컨테이너에서
  `rotaryDim=128, base=10000, interleaved=0` 케이스의 한 원소 오차로
  실패했다. 새 adapter/phase 테스트 22개는 모두 통과했으며, 이 kernel
  baseline failure는 이번 adapter 변경 파일과 무관하게 별도 추적한다.
- 고정 phase benchmark는 TensorRT 첫 enqueue의 lazy initialization을
  순차/동시 비교에 섞지 않도록 두 모드를 먼저 prime한다. 따라서 과거
  `warmup=0, iterations=1` 기록의 6배 이상 speedup은 성능 결론으로 사용하지
  않고, 보정 후 Cosmos b2 p1/d1은 1.1876배였다.

## 불변식

- scheduler에는 모델명 분기가 없어야 한다.
- encoder output은 다음 encoder dispatch가 덮어쓸 수 있으므로 request/phase
  소유 GPU buffer로 복사한 뒤 completion event를 기록한다.
- TensorRT execution context, context workspace, mutable phase I/O는 prefill와
  decode가 공유하지 않는다.
- multimodal request는 encoder가 생성한 token IDs, image embedding, deepstack,
  M-RoPE가 모두 prefill 완료까지 살아 있어야 한다.
