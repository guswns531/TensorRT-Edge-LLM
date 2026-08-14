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
14. [Production decode context batching과 adapter 성능](14-production-decode-context-batching.md)
15. [Continuous request context serving facade와 KV 파편화](15-continuous-context-serving-facade.md)
16. [Production prefill context batching](16-production-prefill-context-batching.md)
17. [Pending admission과 backpressure](17-pending-admission-backpressure.md)
18. [실제 sampling과 반복 decode](18-actual-sampling-and-repeated-decode.md)
19. [CUDA event와 queue metrics](19-cuda-event-and-queue-metrics.md)
20. [Metrics 기반 adaptive scheduler](20-metrics-adaptive-scheduler.md)
21. [Single-context/two-stream 실행 안전 계약](21-single-context-two-stream-safety.md)
22. [Encoder queue/stream/event와 prefill handoff](22-encoder-phase-handoff.md)
23. [동일 CUDA context 기반 3-phase continuous-load 로드맵](23-shared-cuda-context-continuous-load-roadmap.md)
24. [공유 CUDA context와 분리 TensorRT context 구현 결과](24-shared-cuda-independent-trt-contexts.md)
25. [Deterministic continuous-load 구현과 Gemma 실험](25-deterministic-continuous-load.md)
26. [Phase continuous-load sweep 결과](26-phase-load-sweep.md)
27. [Legacy/indexed/phase/continuous 단계별 성능 분석](27-performance-tier-analysis.md)
28. [Gemma 4 VLM three-phase, adaptive prefill, kernel-group 결과](28-gemma4-vlm-three-phase-adaptive-segmentation.md)
29. [Gemma 4 VLM real-request 정확성 및 성능 비교](29-gemma4-vlm-real-request-performance.md)
30. [SLO 기반 customizable scheduler](30-slo-customizable-scheduler.md)
31. [SLO scheduler 100-request 성능 결과](31-slo-scheduler-performance.md)
32. [Production async server와 실제 three-context trace](32-production-async-three-context-server.md)
33. [LLM real-request scheduler 실험](33-llm-real-request-scheduler-experiments.md)
34. [프로젝트 전체 회고·동작 방식·장단점·다음 단계](39-project-retrospective.md)
35. [수정 전 legacy와 수정 후 indexed/phase 비교](40-before-after-legacy-indexed-comparison.md)
36. [모델 중립 independent TensorRT context pair](43-model-agnostic-independent-contexts.md)
37. [Cosmos non-indexed KV/deepstack phase 지원](44-cosmos-phase-support-20260806.md)
38. [공통 phase runtime과 모델별 adapter 분리 설계](45-common-phase-runtime-model-adapters.md)
39. [Cosmos 공정 재측정과 BS16 kernel-group cost table](46-cosmos-fair-phase-cost-20260806.md)
40. [Cosmos Reason2-2B indexed KV 구현과 image trace 결과](47-cosmos-indexed-kv-20260810.md)
41. [Stable indexed-paged KV cache 설계와 구현 계획](48-stable-indexed-paged-kv-plan.md)
42. [Cosmos indexed-paged KV 구현과 10GB GPU 검증](49-cosmos-paged-kv-implementation.md)
43. [Real-request indexed-linear/indexed-paged cost matrix](50-real-request-indexed-paged-cost-matrix.md)
44. [Cosmos 구현 및 vLLM 비교 계획](65-cosmos-vllm-implementation-comparison-plan-20260812.md)
45. [계획 구현 상태와 다음 실험](66-implementation-status-20260812.md)
46. [Cosmos dynamic scheduler 구현 및 비교 결과](67-cosmos-dynamic-scheduler-results-20260812.md)
47. [Cosmos paged-KV reservation 정책 구현과 실험](68-cosmos-page-reservation-policies-20260812.md)
48. [Cosmos 적응형 KV page-growth lease 구현과 실험](69-cosmos-adaptive-page-growth-20260812.md)
49. [Cosmos prefill batching, wavefront, cost model 실험](70-cosmos-prefill-batching-wavefront-20260812.md)
50. [Cosmos direct-overlap cost와 TPOT hard guard](71-cosmos-tpot-guard-direct-overlap-20260813.md)
51. [Cosmos ragged prefill 구현과 vLLM 격차 축소](72-cosmos-ragged-prefill-20260813.md)
52. [Cosmos ragged direct-cost admission과 controlled overlap probe](73-cosmos-ragged-cost-aware-admission-20260813.md)
53. [Cosmos direct-cost v5 coverage 0과 workload regression gate](74-cosmos-direct-cost-v5-regression-gates-20260813.md)
54. [Cosmos throughput-balanced preset과 TPOT hysteresis](75-cosmos-throughput-balanced-hysteresis-20260813.md)
55. [Cosmos 최신 scheduler와 vLLM 비교 및 발전 추이](76-cosmos-vllm-latest-progress-20260813.md)
56. [Cosmos exact-trace graph warmup과 short prefill budget](77-cosmos-trace-graph-warmup-prefill-budget-20260813.md)
57. [Production phase-shape CUDA graph priming](78-cosmos-phase-shape-graph-priming-20260813.md)
58. [Cosmos short TTFT 분해와 P8 검증](79-cosmos-short-ttft-breakdown-20260813.md)
59. [Cosmos completion-aware prefill 실험과 packed prefill 경계](80-cosmos-completion-bonus-and-packed-prefill-seam-20260813.md)
60. [Cosmos true packed prefill CUDA 기반](81-cosmos-packed-prefill-kernel-foundation-20260813.md)
61. [Cosmos packed prefill attention plugin 연결](82-cosmos-packed-prefill-attention-plugin-20260813.md)
62. [Cosmos packed prefill continuation과 variable-prefix gather](83-cosmos-packed-prefill-continuation-20260813.md)
63. [Cosmos packed prefill engine E2E와 real-request 결과](84-cosmos-packed-prefill-engine-e2e-20260813.md)
64. [Cosmos packed prefill 정확성 gate와 288-request trace 결과](85-cosmos-packed-prefill-correctness-and-288-traces-20260813.md)
65. [Cosmos packed-prefill cost model과 dynamic scheduler 연결](86-cosmos-packed-cost-model-and-dynamic-scheduler-20260813.md)
66. [Cosmos packed mixed-load dynamic batching 분석](87-cosmos-packed-mixed-load-dynamic-batching-20260813.md)
67. [Cosmos variable-length packed prefill과 backlog-aware chunk 정책](88-cosmos-variable-packed-prefill-20260813.md)
68. [Cosmos와 vLLM의 남은 성능 차이](89-cosmos-vllm-remaining-gaps-20260813.md)
69. [Cosmos production HTTP/SSE와 vLLM 공정 비교](93-cosmos-http-sse-vllm-fair-comparison-20260813.md)
70. [Cosmos 전체 workload 재검증](94-cosmos-full-workload-revalidation-20260814.md)
71. [Cosmos tied embedding/LM-head GPU 메모리 공유와 성능](95-cosmos-tied-embedding-lm-head-memory-20260814.md)
72. [Cosmos tied LM-head workload 확장 A/B](96-cosmos-tied-head-workload-ab-20260814.md)
73. [Cosmos tied LM-head CUDA Graph workload 재검증](97-cosmos-tied-head-cuda-graph-revalidation-20260814.md)
74. [Cosmos vLLM, clean upstream, Current 재측정](98-cosmos-vllm-upstream-current-recheck-20260814.md)
75. [Cosmos 다섯 workload Current, vLLM, clean upstream 비교](99-cosmos-five-workload-current-vllm-upstream-20260814.md)
76. [Cosmos vLLM attention/scheduler 옵션 sweep](100-cosmos-vllm-attention-scheduler-option-sweep-20260814.md)

- 기본 KV cache와 indexed-linear는 attention layer별로
  `[maxBatch, 2, numKVHeads, maxSequenceLength, headDim]` 크기의 연속 GPU tensor를 미리 할당한다.
  새 opt-in indexed-paged 경로는 stable slot을 유지하면서 128-token 전역 page-bundle pool에서 실제 KV를
  할당하며 Cosmos text decoder에서 export/build/inference까지 검증되었다.
- 호환 경로인 `LLMInferenceRuntime::handleRequest()`는 encoder, prefill, 반복 decode를 한 호출 안에서 직렬로
  수행한다. 새 online 경로는 `PhaseAsyncServer::submit()/poll()/tryPopCompletion()`만 사용한다.
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
- indexed-linear v1은 vanilla text prefill/decode와 batch eviction을 기본 지원한다. speculative decoding,
  system prompt cache, host offload, Mamba, 여러 `handleRequest()` 사이 continuous admission은 명시적으로
  거부한다. multimodal phase 실행은 독립 context와 model adapter가 있는 경우에만 opt-in이다. Qwen3-VL/Cosmos
  indexed decoder의 image trace까지 검증했으며, deepstack/M-RoPE image prefill은 현재 원자적 BS1로 제한한다.
  CUDA graph는 real-request와 1,024-request endurance까지 검증했다.
- indexed-paged v1은 FP16 vanilla text attention, 128-token page, stable slot release와 whole-request page-reservation
  backpressure를 지원한다. prefix refcount/COW, speculative decoding, host offload와 image prefill paging은 아직
  지원하지 않는다.
- 성능 개선보다 정확성, 메모리 소유권, 의존성 검증을 먼저 완료한다.
