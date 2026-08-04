# Production async server와 실제 three-context trace

## 구현 결과

PhaseAsyncServer는 기존 blocking LLMInferenceRuntime::handleRequest()를 사용하지 않는 event-loop API다.
호출자는 JSON/image 요청 하나의 소유권을 submit()으로 넘기고, 서버는 요청별 DecodingInferenceContext,
formatted prompt, image buffer, stable KV slot lease를 terminal event까지 보존한다.

외부 loop가 사용하는 API는 다음 네 개다.

1. submit(request, schedulingHints): 즉시 request ID와 현재 admission 상태를 반환한다.
2. poll(): encoder/prefill/decode CUDA event를 query하고 runnable queue를 dispatch한다.
3. tryPopCompletion(): 완료 순서대로 response를 이동해 반환한다.
4. cancel(requestId): queued 요청은 즉시, GPU in-flight 요청은 event 완료 후 slot을 반환한다.

V1은 요청당 batch 1, greedy vanilla generation을 지원한다. 여러 요청의 실제 batch는 요청 바깥에서 미리 묶는
방식이 아니라 encoder, prefill, decode queue가 각각 독립적으로 만든다. Spec decode, LoRA, audio, logprobs,
legacy streaming callback, stop string, system-prompt cache는 명시적으로 거부한다.

## 실행 topology

    JSON/image arrival
           |
           v
     PhaseAsyncServer  -- owns request/context until terminal
           |
           +--> encoder queue --> encoder stream --> vision IExecutionContext
           |                         |
           |                    CUDA event handoff
           v                         v
     prefill queue ---------> prefill stream --> LLM prefill IExecutionContext
           |                         |
           |                    stable indexed KV slot
           v                         v
     decode queue ----------> decode stream ---> LLM decode IExecutionContext
           |
           v
     completion queue

세 stream은 하나의 CUDA primary context를 공유한다. TensorRT IExecutionContext, mutable context workspace,
phase-local I/O, PLE output은 세 phase가 서로 alias하지 않는다. LLM engine weight와 PLE table 같은 immutable
데이터, indexed KV physical storage는 공유한다.

RTX 3080 10GB에서는 LLM context가 모든 optimization profile의 최대 workspace를 각각 할당하면 세 번째
encoder context를 수용할 수 없었다. TensorRT 11의 profile별 상한을 사용해 prefill은 716.0 MiB, decode는
16.5 MiB를 할당했다. encoder는 358.1 MiB다. 이 구성에서 full physical KV slot 4개를 유지하고 encoder context
생성 직전 free VRAM은 999.4 MiB였다.

중요한 correctness 조건은 engine의 maxBatchSize=4와 KV physical allocation을 일치시키는 것이다. 실험 중
KV slot tensor만 2개로 줄였을 때 embedding/PLE는 finite였지만 logits 전체가 NaN이 됐다. indexed mode에서
active lease가 2개인 것과 physical storage를 2개만 할당하는 것은 다른 의미다. active phase batch와 I/O는
BS1로 줄여도 KV physical slot 수는 engine 계약인 4를 유지한다.

Gemma 4 prefill은 text-only 요청에도 vision_block_ids를 요구한다. packed callback과 warmup 모두 token별
block ID를 결정적으로 채운다. 이 입력을 생략하면 첫 attention부터 NaN이 전파된다.

## 실제 JSON/image arrival trace

입력은 [gemma4_vlm_real_requests.json](../tests/test_cases/gemma4_vlm_real_requests.json)이다.
12 requests, 18 images, 4,648 image tokens, 788 generated tokens을 2 request/s arrival로 직접
PhaseAsyncServer에 넣었다. synthetic prompt length나 fake embedding은 사용하지 않았다.

    ./build/examples/llm/llm_phase_bench \
      --engineDir /tmp/gemma4-e2b/engine-indexed \
      --multimodalEngineDir /tmp/gemma4-e2b/engine-multimodal \
      --inputFile tests/test_cases/gemma4_vlm_real_requests.json \
      --prefillBatch 1 --decodeBatch 1 \
      --inputLen 1024 --prefillChunkSize 1024 --pastKVLen 512 \
      --traceArrivalRate 2 \
      --traceCsv notes/results/gemma4-phase-real-image-trace.csv

| metric | result |
|---|---:|
| completed | 12 / 12 |
| output tokens | 788 |
| E2E median | 1,208.0 ms |
| E2E p95 | 1,583.5 ms |
| E2E min / max | 212.2 / 1,636.0 ms |
| submit queue-delay median / p95 | 0.111 / 2.485 ms |

첫 두 text-only smoke 응답은 각각 Hello!와 Four.로 나왔고 logits NaN은 사라졌다. Image trace의 semantic
판정은 기존 synchronous baseline과 같다. 12건 중 10건이 strict reference를 만족한다. red panda와 giant
panda를 표로 비교하라는 한 요청이 두 번째 이미지를 다시 요구하고, woman/dog와 red panda 비교가 red panda를
wild bear cub로 부른다. 두 실패는 legacy와 indexed handleRequest()의 반복 결과에도 동일하므로 async
handoff나 multi-image placement 회귀로 보지 않는다.

Kernel-group CUDA-event 계측은 2,400 segments를 기록했다. 주요 median은 encoder preprocess 5.81 ms,
encoder engine 23.41 ms, prefill prepare 0.116 ms, prefill engine 41.60 ms, decode prepare 0.015 ms,
decode engine 6.93 ms, decode sample 0.021 ms다.

기본 2 request/s에서 multimodal prompt는 128-token overlap 한도보다 커서 scheduler가 prefill/decode를
직렬 plan으로 선택했다. 동시 실행 가능성은 4 request/s와 maxOverlapPrefillTokens=1024로 별도 검증했다.
이때 542-token prefill과 한 decode step이 실제 overlap plan 하나로 dispatch됐고 CUDA-event overlap ratio는
20.23%였다. 전체 12-request makespan은 6,370.4 ms에서 6,254.0 ms로 1.83% 감소했지만, 더 압축된 arrival로
E2E median은 2,515.4 ms까지 증가했다. 즉 context overlap 자체는 동작하지만 SLO 관점에서는 workload별
admission rate와 overlap 한도를 함께 조절해야 한다.

Raw response/latency CSV는 기본적으로 *.csv ignore 대상이라 local artifact로 남는다. 리뷰용 요약은 이
문서가 source of truth다.

## 현재 경계

- 새 online harness는 public handleRequest()를 전혀 호출하지 않는다.
- 기존 examples/API를 한 번에 깨지 않기 위해 blocking API symbol은 compatibility 경로로 남겨 두었다.
- 다음 productionization은 server construction을 별도 factory로 옮기고, gRPC/HTTP frontend가 같은
  submit/poll/completion 계약만 사용하게 하는 것이다.
- RTX 3080의 현재 profile에서 prefill/decode queue 최대 batch는 각각 1이다. 더 큰 independent batch는
  별도 engine profile 또는 더 큰 VRAM이 필요하다.
