# Production prefill context batching

`PhasePrefillContextBatchAdapter`는 서로 다른 request context의 prompt slice를 하나의 prefill batch로 pack한다.
각 row는 `requestId`, source context row, stable `kvSlotId`, `tokenOffset`, `tokenCount`를 유지한다.

```text
request A prompt[0:128]   slot 3 --+
request B prompt[0:128]   slot 0 --+--> device token_ids [2, 128]
                                           slot_ids [3, 0]
                                           lengths [0, 0]
                                                |
                                      Gemma PLE / TensorRT prefill
                                                |
                                      global slot length += 128
```

scheduler는 TensorRT shape가 같은 row만 묶도록 `(initial/continuation, chunkLength)`를 bucket key로 사용한다.
따라서 128-token initial chunk와 128-token continuation chunk, 또는 128/64-token final chunk가 같은 batch에
섞이지 않는다. adapter는 text-only full prompt를 요구하며 prefix-cache reuse, LoRA와 multimodal input은 v1에서
거부한다.

adapter는 pinned host staging과 device token tensor를 최대 batch/chunk 크기로 한 번 할당한다. pack 시 prompt의
현재 slice만 복사하고 `PhaseBatchState`가 stable slot length를 gather한다. CUDA event 완료 뒤 `complete()`가
adapter 진입 직전의 prefill TensorMap binding을 정확히 복원한다. KV tensor 자체는 이동하지 않는다.

실제 Gemma 4 E2B INT4 indexed engine에서 BS2/input512/chunk128을 실행해 request당 네 prefill turn과 한 packed
decode turn이 완료되고 모든 slot이 반환되는 것을 확인했다. raw smoke CSV는
`/tmp/gemma4-e2b/perf/phase/production-prefill-adapter-smoke.csv`다.

단위 테스트는 prompt slice `[12,13]`, `[22,23]`와 physical slot mapping `[3,0]`, global length commit,
exact binding restore를 GPU에서 검증한다. scheduler test는 서로 다른 initial/final/continuation bucket이 분리되는지
검증한다.
