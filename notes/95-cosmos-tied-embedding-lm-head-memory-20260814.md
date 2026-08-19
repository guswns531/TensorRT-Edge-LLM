# Cosmos tied embedding/LM-head GPU 메모리 공유

## 결론

Cosmos-Reason2-2B의 tied token embedding과 LM head를 GPU에 두 번 올리던 경로를 opt-in으로 바꿨다.
`--reuse-tied-lm-head`를 사용하면 FP16 weight를 LM-head GEMM에 적합한 `[hidden, vocab]` layout으로 한 번만
저장하고, embedding lookup이 같은 buffer를 strided gather로 읽는다. TensorRT engine의 LM-head input도 이
buffer를 non-owning alias로 직접 binding한다.

RTX 3080 10GB의 balanced real-request trace를 세 번 새 process에서 반복한 graph-off primary 결과는 다음과 같다.

- 실행 전 GPU 사용량: `8,946.9 -> 8,370.9MiB`, **576MiB 절감**
- 실행 후 GPU 사용량: `8,950.9 -> 8,376.9MiB`, **574MiB 절감**
- generated token throughput: `3,623.86 -> 3,616.41 token/s`, **0.206% 하락**
- TTFT median/p95: **0.185% / 0.211% 증가**
- TPOT median/p95: **0.349% / 0.405% 증가**
- E2E median/p95: **0.128% / 0.227% 증가**
- prefill/decode engine CUDA-event 평균: **0.372% / 0.089% 증가**

즉 10GB GPU 전체 사용량의 약 `6.4%`, 원래 중복된 FP16 vocabulary table 한 벌에 해당하는 메모리를 회수하면서
주요 graph-off latency와 처리량 회귀를 모두 0.5% 아래로 유지했다. CUDA graph 실험에서도 실행 후 582MiB를
절감했다. graph 결과의 처리량 `+2.30%`는 capture shape와 실행 순서 변동이 포함되므로 이 기능의 구조적 속도
향상으로 해석하지 않는다.

## 왜 두 벌이 있었나

Cosmos checkpoint는 `tie_word_embeddings=true`이므로 수학적으로 token embedding과 LM head가 같은 weight다.
하지만 기존 runtime은 서로 다른 소유권 경로로 적재했다.

```text
embedding.safetensors                    TensorRT engine
[vocab, hidden] FP16                     LM-head constant [hidden, vocab] FP16
          |                                              |
          +--> embedding lookup                         +--> logits GEMM
```

Cosmos의 `vocab=151,936`, `hidden=2,048`, FP16일 때 한 table은 대략 다음 크기다.

```text
151,936 * 2,048 * 2 bytes = 622,329,856 bytes = 593.5 MiB
```

allocator 정렬과 다른 작은 실행시점 변화 때문에 device 전체 관측 절감은 574--582MiB지만, 원인은 이 중복 table
한 벌이다. KV page pool, independent prefill/decode execution context, phase I/O, CUDA graph cache는 그대로다.

## 최종 동작 방식

```text
checkpoint tied weight [vocab, hidden]
                 |
                 | export 시 1회 transpose
                 v
embedding.safetensors: embedding_transposed [hidden, vocab]
                 |
                 | GPU allocation 1개
                 v
       +--------- shared FP16 buffer ----------+
       |                                        |
       | strided gather                         | non-owning Tensor alias
       v                                        v
embedding lookup                         TensorRT LM-head input
table[h * vocab + token]                 MatMul(hidden, [hidden,vocab])
       |                                        |
       +------------ hidden states ------------+--> logits
```

핵심은 runtime transpose가 없다는 점이다. embedding은 token 하나의 hidden vector가 연속하지 않은 layout을 읽으므로
기존 coalesced lookup보다 이론적으로 불리하지만, 전체 모델에서 차지하는 시간이 작아 real trace 회귀는 0.5%
미만이었다. LM-head GEMM은 기존 최적 layout을 그대로 받으므로 큰 prefill workspace를 추가하지 않는다.

## 실패한 첫 설계와 수정

첫 시도는 기존 `[vocab, hidden]` embedding buffer를 TensorRT input으로 주고 ONNX graph 안에 `Transpose`를
남겼다. engine의 constant weight는 약 593MiB 줄었지만 prefill activation workspace가 약 `84 -> 661.5MiB`로
증가했다. 그 결과 실행 전 device 사용량은 `8,946.9MiB`로 baseline과 같았고 graph-off에서 throughput은
`4.813%` 하락, prefill engine 평균은 `12.563%` 증가했다.

이 경로는 폐기했다. 최종 경로는 ONNX의 기존 2D transpose consumer를 export 때 제거하고 input shape를
`[hidden, vocab]`으로 canonicalize한다. 따라서 engine build의 prefill workspace는 다시 약 84MiB이고 TensorRT
managed weight만 `3,729 -> 3,136MiB`로 감소한다.

## 구현 위치

- `tensorrt_edgellm/scripts/export.py`: `--reuse-tied-lm-head` CLI와 v1 지원 범위 검증
- `tensorrt_edgellm/onnx/export.py`: tied 여부, embedding scale, FP8/reduced-vocab 제약 검증과 export 연결
- `tensorrt_edgellm/external_weights.py`: LM-head initializer를 external input으로 바꾸고 embedding source manifest 생성
- `tensorrt_edgellm/checkpoint/checkpoint_utils.py`: `embedding_transposed` `[hidden,vocab]` artifact 저장
- `cpp/runtime/state/externalWeightManager.*`: embedding GPU allocation을 engine binding 이름으로 non-owning alias
- `cpp/runtime/llmRuntimeUtils.cpp`: normal/transposed embedding artifact를 구분해 적재
- `cpp/kernels/embeddingKernels/embeddingKernels.cu`: text/image embedding의 transposed-layout gather
- `cpp/runtime/llmInferenceRuntime.cpp`, `examples/llm/llm_phase_bench.cpp`: embedding을 먼저 적재하고 alias 수명 연결

기본값은 꺼져 있으므로 기존 engine/runtime의 weight 소유권과 embedding layout은 변하지 않는다. v1은 tied FP16
embedding, scale 1, TP1, vanilla model만 지원한다. FP8 embedding, reduced vocab, tensor parallel, speculative
decoding은 export 단계에서 거부한다.

## 공정 성능 비교

고정 조건은 `nvidia/Cosmos-Reason2-2B` FP16, indexed-paged FP16 KV, 128-token page, 256 page bundles, 80 slots,
P8/D64, fixed chunk 128, independent prefill/decode TensorRT contexts, shared CUDA primary context, arrival 30 req/s다.
동일 binary와 동일 SHA-256 trace를 사용하고 각 variant를 세 번 새 lifecycle로 실행했다.

Trace는 288 requests, 25,872 prompt tokens, 23,712 generated tokens이며 SHA-256은
`290d3406...49d6538`이다.

| Graph-off primary | Baseline | Shared layout | 변화 |
| --- | ---: | ---: | ---: |
| Generated token/s | 3,623.863 | 3,616.415 | -0.206% |
| Request/s | 44.015 | 43.924 | -0.206% |
| TTFT median / p95 | 2,723.73 / 5,214.00ms | 2,728.76 / 5,224.98ms | +0.185% / +0.211% |
| TPOT median / p95 | 10.567 / 12.034ms | 10.603 / 12.083ms | +0.349% / +0.405% |
| E2E median / p95 | 3,620.85 / 5,965.69ms | 3,625.50 / 5,979.23ms | +0.128% / +0.227% |
| Prefill engine 평균 | 16.078ms | 16.138ms | +0.372% |
| Decode engine 평균 | 7.731ms | 7.738ms | +0.089% |
| Device used, 실행 전 / 후 | 8,946.9 / 8,950.9MiB | 8,370.9 / 8,376.9MiB | -576 / -574MiB |

`prepare` 구간은 절대값이 수십 microseconds라 상대 변화가 크게 보인다. prefill은 `0.033 -> 0.083ms`, decode는
`0.008 -> 0.013ms`로 증가했지만 E2E에 미치는 절대 영향은 작다.

| CUDA graph smoke | Baseline | Shared layout | 변화 |
| --- | ---: | ---: | ---: |
| Generated token/s | 3,563.474 | 3,645.492 | +2.302% |
| TTFT median / p95 | 2,771.83 / 5,186.35ms | 2,698.71 / 4,949.11ms | -2.638% / -4.574% |
| TPOT median / p95 | 10.192 / 11.949ms | 10.419 / 11.997ms | +2.230% / +0.396% |
| E2E median / p95 | 3,698.78 / 5,956.80ms | 3,605.19 / 5,913.58ms | -2.530% / -0.726% |
| Prefill / decode engine 평균 | 16.129 / 7.515ms | 16.563 / 7.556ms | +2.691% / +0.548% |
| Device used, 실행 전 / 후 | 8,946.9 / 9,180.9MiB | 8,370.9 / 8,598.9MiB | -576 / -582MiB |

graph variant는 cold dynamic capture이며 prefill graph hit가 `20.65% -> 16.67%`로 달라졌다. 그러므로 graph-off를
primary regression gate로 사용하고 graph 결과는 호환성과 메모리 smoke로만 본다.

## 정확성 상태와 gate

- checkpoint embedding과 원래 ONNX LM-head initializer는 byte-exact transpose였다.
- normal-layout CPU reference와 transposed-layout CUDA embedding lookup unit test가 exact하게 일치했다.
- real-request 양쪽 모두 288개 요청을 성공했고 generated token 총수 `23,712`, EOS/max-token 분포 `48/240`가
  같았다.
- 그러나 별도로 build한 baseline engine과 shared-layout engine의 greedy token sequence는 `96/288`만 완전히
  같았다. 첫 mismatch token 중앙값은 48.5다. constant GEMM을 runtime input GEMM으로 바꾸면서 TensorRT tactic과
  부동소수점 누산 순서가 달라진 영향으로 추정되며, 현재 결과만으로 exact greedy identity를 주장하지 않는다.

따라서 메모리 gate와 3% 성능 gate는 통과하지만 이 기능은 아직 **experimental opt-in**이다. 기본값으로 만들기
전에 logits allclose 비교와 Cosmos task/accuracy suite를 통과시켜 수치 차이가 허용 범위인지 확인해야 한다.

## 보존 artifact와 다음 단계

- 결과 요약: `.local/cosmos-reason2-2b/tied-head-ab-20260814/layout-summary.json`
- 최종 ONNX: `.local/cosmos-reason2-2b/onnx-fp16-packed-tied-layout/llm`
- 최종 engine: `.local/cosmos-reason2-2b/engine-fp16-packed-tied-layout-p8-d64-mb80-b256`
- baseline engine: `.local/cosmos-reason2-2b/engine-fp16-packed-p8-d64-mb80-b256`

다음 gate는 같은 hidden-state 입력의 baseline/shared LM-head logits를 FP32 기준과 함께 비교해 max/mean error와
top-1 margin flip rate를 측정하는 것이다. 그 뒤 short/balanced/decode-heavy/long-prefill 전체 workload를 다시
검증한다. 576MiB headroom은 KV page bundle 약 2배 확대, 더 큰 CUDA graph cache, 또는 long-prefill workspace 중
하나에 우선 배분하고 동시에 모두 늘리지는 않는다.
