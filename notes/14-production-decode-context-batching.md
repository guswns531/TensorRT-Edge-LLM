# Production decode context batching과 측정 결과

이번 단계는 서로 다른 요청의 `DecodingInferenceContext` row를 한 decode batch로 묶되, KV cache는 이동하지 않는 실행 경계를 추가한다.

## 데이터 흐름

```text
request A context row 1 -- slot 3 --+
                                     +--> phase-local rows [0, 1]
request B context row 0 -- slot 0 --+      slot_ids [3, 0]
                                            |
                                  TensorRT decode / XQA
                                            |
                                     CUDA event 완료
                                            |
                         token/length/finished state scatter
```

`PhaseContextBatchAdapter::packDecode()`는 source row의 token과 generation metadata를 packed context로 복사하고,
`PhaseBatchState`가 physical slot ID를 GPU에 올린 뒤 global slot length를 phase-local tensor로 gather한다.
`VanillaDecoder`는 context에 phase state가 있으면 이 length view로 `contextLengths`를 만들고, 성공한 decode
enqueue 뒤 해당 stable slot만 commit한다. 완료 event가 끝난 다음 `scatterDecode()`가 결과를 원래 context로
돌려놓고 legacy tensor binding을 복원한다.

KV tensor의 주소와 내용은 이 과정에서 바뀌지 않는다. logical row compaction 대신 작은 host metadata와
`kv_slot_ids`/length tensor만 갱신하므로 KV D2D copy와 cache compaction이 없다. 빈 physical slot은
free-list가 재사용하며, 살아 있는 slot의 cache는 그대로 유지된다.

## v1 안전 범위

현재 adapter는 vanilla text decode 전용이다. streaming channel/callback, logprobs, logit bias, LoRA, multimodal,
thinker output, layer debugger와 nested phase packing은 명시적으로 거부한다. sampling parameter와
`maxGenerateLength`가 다른 요청도 같은 packed batch에 섞지 않는다. adapter는 enqueue부터 completion까지
살아 있어야 하며 scatter는 CUDA completion 뒤에만 호출한다.

일반 `handleRequest()`는 기본 경로를 그대로 사용한다. 새 포인터가 null이면 기존 active-row KV length와
commit을 사용하므로 legacy engine 및 CUDA graph 동작은 바뀌지 않는다.

## RTX 3080 실제 Gemma 측정

환경은 기존과 같은 Gemma 4 E2B INT4-AWQ indexed engine, FP16 KV, CUDA graph 비활성 phase benchmark다.
BS2 prefill input 512와 BS2 decode past KV 512에서 warmup 20회 후 100회를 측정했다.

| 경로 | Sequential median / p95 | Overlap median / p95 |
|---|---:|---:|
| adapter 미사용 | 94.7323 / 95.1787 ms | 90.0710 / 90.4502 ms |
| adapter 사용 | 94.8705 / 95.3446 ms | 90.1028 / 90.6435 ms |
| 변화 | +0.15% / +0.17% | +0.04% / +0.21% |

adapter 사용 시 host pack median은 sequential/overlap에서 53.474/52.844 us,
scatter median은 12.902/12.197 us였다. overlap makespan speedup은 1.0529x다.
raw CSV는 `/tmp/gemma4-e2b/perf/phase/context-adapter-final-b2_i512_k512.csv`에 저장했다.

실제 `llm_basic` greedy 출력 SHA256은
`1918f649c96695ea807985d3e7a98c4257d3d429f3b9557277f4957472dfcb2a`로 기존 baseline과 byte 단위로 같다.
CUDA graph 4개도 정상 캡처됐고 Compute Sanitizer memcheck는 8개 phase/adapter 테스트에서 0 error였다.

## 다음 경계

이제 남은 production 작업은 queue admission이 source context row와 slot lease를 만들고,
decode completion에서 adapter scatter와 request별 종료 처리를 호출하는 serving facade다.
prefill은 prompt tensor 크기가 다르므로 chunk size bucket과 별도 prefill context adapter가 필요하다.
