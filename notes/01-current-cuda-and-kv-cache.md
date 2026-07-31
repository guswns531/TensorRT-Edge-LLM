# 현재 CUDA 실행 경로와 KV cache

## 1. 요청 실행 흐름

현재 중심 진입점은 `cpp/runtime/llmInferenceRuntime.cpp`의
`LLMInferenceRuntime::handleRequest()`다.

```text
handleRequest
  -> tokenizer / request 준비
  -> multiModalRuntimePreprocess
       -> vision/audio preprocess
       -> vision/audio TensorRT enqueueV3
  -> setUpForPrefillExecution
       -> KV cache length reset 또는 prefix cache restore
  -> runBaseModelPrefill
       -> embedding/preparation kernels
       -> profile 0 prepare
       -> base TensorRT enqueueV3
       -> KV length += context length
       -> sampling + D2H + stream synchronize
  -> while (...)
       -> VanillaDecoder::decodeStep 또는 speculative strategy
            -> embedding/preparation kernels
            -> profile 1 prepare
            -> base TensorRT enqueueV3
            -> KV length += 1
            -> sampling + D2H + stream synchronize
       -> batch eviction/compaction
```

관련 코드:

- `cpp/runtime/llmInferenceRuntime.cpp`: `handleRequest()`, `multiModalRuntimePreprocess()`,
  `runBaseModelPrefill()`, `setUpForPrefillExecution()`
- `cpp/runtime/decoding/vanillaDecoder.cpp`: vanilla decode 한 step
- `cpp/runtime/exec/engineExecutor.cpp`: profile 변경, binding, `enqueueV3()`, CUDA graph launch
- `cpp/runtime/preprocess/stepPreparer.cpp`: phase별 context length와 engine input 준비

현재 API가 받은 `cudaStream_t`가 encoder, prefill, decode, sampling, cache management에 계속 전달된다.
TensorRT는 engine에 auxiliary stream이 있으면 `EngineExecutor` 생성 시 non-blocking stream을 추가로 만들고
`IExecutionContext::setAuxStreams()`에 등록한다. 따라서 겉으로 보이는 caller stream 하나만 관찰해서는 실제
TensorRT 내부 실행 stream 전체를 대표하지 못할 수 있다.

## 2. TensorRT engine과 phase의 관계

prefill과 decode는 서로 다른 engine이 아니다.

- profile 0: prefill/context
- profile 1: decode/generation
- executor: 동일한 `EngineExecutor`
- TensorRT execution context: 동일한 `IExecutionContext`
- TensorMap/PipelineIO: 동일한 객체를 shape만 바꿔 재사용
- context workspace: 동일한 `mSharedExecContextMemory`

`EngineExecutor::prepare()`는 `setOptimizationProfileAsync(profileIndex, stream)` 후 모든 address와 shape를
다시 bind한다. `execute()`는 binding snapshot에 맞는 CUDA graph가 있으면 graph를 launch하고, 없으면
`enqueueV3(stream)`을 호출한다.

동일 `IExecutionContext`를 서로 다른 host thread/stream에서 동시에 prepare/enqueue하면 profile, shape,
binding address가 서로 덮어써진다. 이것이 prefill/decode stream 분리 전에 execution context 복제가 필요한
첫 번째 이유다.

## 3. 공유 TensorRT context workspace

`LLMInferenceRuntime::initializeCommon()`은 base, speculative strategy, vision, audio, action engine의 필요한
workspace 중 최댓값 하나만큼 `mSharedExecContextMemory`를 만들고 모든 context에 같은 주소를 준다. 코드 주석도
모든 engine이 직렬 실행하므로 공유할 수 있다고 명시한다.

동시 실행을 도입할 때는 적어도 동시에 실행 가능한 execution context마다 별도 workspace가 필요하다.

```text
기존:
  base context ----\
  vision context ---+--> one shared workspace
  audio context ----/

필요:
  encoder context ------> encoder workspace
  prefill context ------> prefill workspace
  decode context -------> decode workspace
```

workspace뿐 아니라 `PipelineIO`, embedding output, logits, host staging buffer, sampling buffer도 phase/request별
동시 접근 여부를 조사해야 한다.

## 4. KV cache의 실제 layout

`KVCacheManager`는 attention layer마다 GPU tensor 하나를 생성한다.

```text
[maxBatchSize, 2, numKVHeads_i, maxSequenceLength, headDim_i]
                    ^
                    K/V
```

- dtype: FP16 또는 FP8
- allocation: runtime 초기화 시 고정 크기로 한 번 할당
- layer별 `numKVHeads`, `headDim` 차이를 허용
- page table이나 block allocator 없음
- batch slot 하나가 해당 layer의 전체 `maxSequenceLength` 영역을 소유

메모리 사용량은 대략 다음과 같다.

```text
sum(layer_i)
  maxBatchSize * 2 * numKVHeads_i * maxSequenceLength * headDim_i * elementSize
```

`HybridCacheManager`는 attention KV cache와 Mamba recurrent/conv state를 하나의 layer routing 아래 묶는다.
KV length는 GPU의 `int32[maxBatchSize]` tensor 하나로 관리한다.

### 요청 시작

`resetForNewSequences()`가 prefix reuse 길이를 host에서 GPU length tensor로 복사한다.
모든 reuse length가 0이면 `mKVCacheAllEmpty = true`가 된다. 이 값은 initial prefill과 chunked/prefix-reuse
prefill의 binding shape를 구분하는 데 사용된다.

### prefill 완료

engine이 cache의 기존 length 다음 위치에 K/V를 쓴 후
`commitSequenceLength(contextLengths)`가 각 slot 길이를 증가시킨다.

### decode 한 step 완료

engine이 각 slot에 token 하나의 K/V를 쓴 후 `commitSequenceLength(1)`이 모든 active slot 길이를 1씩
증가시킨다.

### 요청 종료 및 batch 축소

끝난 slot이 생기면 `compactBatch()`가 살아 있는 slot을 앞쪽으로 이동한다.

- layer별 KV data 이동
- KV length tensor 이동
- Mamba recurrent/conv state 이동
- active batch size 변경

이는 current batch가 독점적으로 cache를 사용한다는 전제에는 단순하지만, 다른 stream에서 prefill/decode가
동시에 같은 batch 차원을 사용하면 compaction과 cache write가 충돌한다.

## 5. system prompt cache

system prompt cache는 main KV tensor의 일부를 별도 tensor들로 capture한 뒤 요청 slot로 restore하는 방식이다.
현재 FP16 KV cache만 capture/restore를 지원한다. 이 경로 역시 main cache와 copy kernel의 순서를 보장해야 하므로
초기 동시 실행 PoC에서는 비활성화하는 편이 안전하다.

## 6. 현재 timing 기반

`cpp/profiling/timer.{h,cpp}`에 이미 CUDA event 기반 `TIME_STAGE`가 있다.

- scope 시작/끝에 CUDA event 기록
- 결과 조회 시 end event만 synchronize
- `kLLM_PREFILL`, `kLLM_GENERATION` 등의 stage 제공

재사용할 수 있지만 현재 구현은 stage 이름마다 event pair 하나이고 전역 timer가 mutable map을 관리한다.
여러 phase가 동시에 같은 stage 이름으로 진행되는 MPSC 환경, request ID별 ring buffer, scheduler의 non-blocking
polling에는 그대로 적합하지 않다. 기존 metric은 유지하고 별도의 scheduler용 timeline을 추가하는 것이 안전하다.

## 7. 우선 읽을 코드 순서

1. `cpp/runtime/llmInferenceRuntime.cpp`
2. `cpp/runtime/decoding/vanillaDecoder.cpp`
3. `cpp/runtime/exec/engineExecutor.{h,cpp}`
4. `cpp/runtime/preprocess/stepPreparer.{h,cpp}`
5. `cpp/runtime/kvCacheManager.{h,cpp}`
6. `cpp/runtime/hybridCacheManager.{h,cpp}`
7. `cpp/kernels/kvCacheUtilKernels/`
8. `cpp/kernels/speculative/batchEvictKernels.cu`
9. `cpp/multimodal/multimodalRunner.{h,cpp}`와 사용할 encoder runner
10. `cpp/profiling/timer.{h,cpp}`
