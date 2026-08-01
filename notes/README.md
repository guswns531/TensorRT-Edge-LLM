# Intra-GPU Encoder/Prefill/Decode 분리 연구 노트

이 디렉터리는 TensorRT Edge-LLM에서 하나의 프로세스와 하나의 GPU를 사용하면서 다음 세 phase를
논리적·실행적으로 분리하기 위한 학습 및 구현 지도다.

1. multimodal encoder
2. LLM prefill
3. LLM decode

목표는 단일 요청의 세 단계를 동시에 실행하는 것이 아니다. 한 요청 안에서는
`encoder -> prefill -> decode` 데이터 의존성이 있으므로 순서를 지켜야 한다. 기대하는 효과는 서로 다른 요청의
phase를 pipeline처럼 겹치는 것이다. 예를 들면 요청 A의 decode와 요청 B의 encoder 또는 prefill을 같은 GPU에서
동시에 실행한다.

## 먼저 읽을 순서

1. [현재 CUDA 실행 경로와 KV cache](01-current-cuda-and-kv-cache.md)
2. [목표 구조와 안전한 분리 경계](02-intra-gpu-epd-architecture.md)
3. [CUDA event 계측과 scheduler 계획](03-instrumentation-and-scheduler-plan.md)
4. [BulletServe 구현 읽기 가이드](04-bulletserve-reading-guide.md)
5. [실험·검증 체크리스트](05-experiment-checklist.md)
6. [KV cache 구조 선택지와 지원 전략](06-kv-cache-design-options.md)

7. [Gemma 4 E2B INT4 / indexed-linear 구현과 실험 결과](07-gemma4-e2b-int4-indexed-implementation.md)

8. [Prefill/Decode dual-stream PoC와 queue scheduler](08-dual-stream-phase-scheduler.md)
9. [Chunked prefill과 physical-slot KV length view](09-chunked-prefill-and-slot-lengths.md)
10. [Phase batch binding과 CUDA event dispatch worker](10-phase-dispatch-worker.md)

11. [Production prefill/decode async completion 경계](11-async-phase-completion.md)
12. [Continuous request lifecycle과 stable slot lease](12-continuous-request-lifecycle.md)
## 지금 내린 핵심 결론

13. [실제 phase engine 연결과 shared/independent context 성능](13-phase-engine-connection-and-performance.md)

- 현재 KV cache는 paged cache가 아니다. attention layer별로
  `[maxBatch, 2, numKVHeads, maxSequenceLength, headDim]` 크기의 연속 GPU tensor를 미리 할당하고,
  batch slot과 sequence position으로 관리한다.
- 현재 `LLMInferenceRuntime::handleRequest()`는 encoder, prefill, 반복 decode를 한 호출 안에서 직렬로 수행한다.
- LLM prefill과 decode는 optimization profile만 0/1로 나뉘며 같은 `EngineExecutor`, 같은 TensorRT
  `IExecutionContext`, 같은 I/O buffer를 사용한다.
- base, draft, vision, audio, action engine은 “서로 직렬 실행한다”는 전제로 하나의 TensorRT context workspace를
  공유한다. 따라서 stream만 세 개로 바꾸는 것은 안전하지 않다.
- encoder는 이미 별도 TensorRT execution context를 가지므로 첫 번째 동시 실행 PoC의 좋은 출발점이다. 다만
  공유 workspace와 요청별 output buffer부터 분리해야 한다.
- prefill/decode 동시 실행에는 최소한 별도 TensorRT execution context, 별도 context workspace, 요청/phase별
  I/O buffer, 겹치지 않는 KV slot 소유권이 필요하다.
- 현재 직렬화된 TensorRT LLM engine은 한 번의 `enqueueV3()` 내부를 임의의 layer group 경계에서 멈출 수 없다.
  BulletServe처럼 layer-wise scheduling을 하려면 phase-level 동시 실행을 먼저 완성한 뒤, LLM을 여러 engine
  segment로 빌드하는 별도 설계가 필요하다.
- SM 제어는 구현을 scheduler와 분리된 backend로 둔다. 기본은 `Noop`, 공식 경로 후보는 CUDA Green Contexts,
  연구용 후보는 BulletServe의 `libsmctrl`이다.

## 범위 원칙

- 기존 `handleRequest()` 경로는 기본값으로 유지한다.
- 새 기능은 opt-in configuration 아래에서만 활성화한다.
- indexed-linear v1은 vanilla text prefill/decode와 batch eviction만 지원한다. speculative decoding,
  system prompt cache, host offload, Mamba, multimodal 실행, 여러 `handleRequest()` 사이 continuous admission은
  명시적으로 거부한다. CUDA graph는 smoke만 검증했다.
- 성능 개선보다 정확성, 메모리 소유권, 의존성 검증을 먼저 완료한다.
