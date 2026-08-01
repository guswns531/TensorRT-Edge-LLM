# 실제 sampling과 반복 decode

## 이전 smoke의 한계

이전 facade smoke는 TensorRT prefill/decode와 stable KV binding은 실행했지만 decode completion에서 상수 token
`0`을 source context에 넣고 즉시 종료했다. 따라서 logits, 다음 token embedding, 반복 decode 상태가 하나의
production loop로 연결됐다는 증거는 아니었다.

## 현재 데이터 흐름

```text
final prefill chunk
  -> TensorRT outputLogits [B, vocab]
  -> PhaseGreedySampler top-1 (GPU)
  -> selected IDs D2H (같은 prefill stream)
  -> CUDA event completion
  -> source tokenIds append / EOS·length 판정
  -> decode queue
       |
       v
PhaseContextBatchAdapter
  -> 각 source의 마지막 token을 pinned host [B]에 pack
  -> async H2D device [B, 1]
  -> Gemma 4 PLE embedding
  -> TensorRT decode + indexed stable-slot KV commit
  -> outputLogits top-1 + async D2H
  -> event completion 후 source row scatter
  -> unfinished row만 다음 decode batch로 requeue
```

`PhaseGreedySampler`는 phase마다 별도 instance를 둔다. selected ID, pinned host staging, top-K workspace를
prefill/decode stream 사이에서 공유하지 않으므로 두 phase가 independent context mode에서 겹쳐도 buffer race가
없다. V1은 greedy top-1 전용이다. temperature/top-k/top-p, logit bias, reduced-vocab mapping은 아직 facade에서
지원하지 않는다.

## 종료와 KV lease

첫 생성 token은 final prefill logits에서 나온다. 이 token이 EOS이거나 `maxGenerateLength`에 도달하면 요청을
decode queue에 넣지 않고 final-prefill completion에서 stable slot을 즉시 반환한다. decode token도 동일하게
EOS/length 상태를 source row에 기록하고, adapter scatter 뒤 terminal lifecycle이 slot을 반환한다.

KV length와 생성 token 수는 구분된다.

- final prefill sampling은 새 KV entry를 쓰지 않으므로 KV length는 prompt 길이 그대로다.
- sampled token을 입력으로 실행한 decode 한 번이 끝나면 KV length가 1 증가한다.
- `currentGenerateLengths`는 prefill sampling token부터 1 증가한다.

## 실제 Gemma 4 E2B INT4 smoke

indexed engine, RTX 3080에서 다음 조건으로 실행했다.

- request 2개
- prompt 512 token
- chunk size 128: prefill 4회
- generation budget 5 token: final-prefill token 1개 + decode 최대 4회
- EOS IDs `[1, 106]`
- separate TensorRT contexts와 prefill/decode streams

실제 logits top-1, `[B,1]` token staging, Gemma PLE, 반복 packed decode, source scatter, slot release까지 모두
통과했다. 1회 smoke의 timed synthetic phase workload는 sequential 139.4463 ms, independent-context concurrent
123.2374 ms로 1.1315x였다. 이 수치는 정식 3회/100 iteration gate가 아니라 연결 검증용 단일 sample이다.
raw CSV는 `/tmp/gemma4-e2b/perf/phase/actual-sampling-smoke.csv`에 저장했다.

## 테스트

- synthetic FP32 logits에서 GPU top-1 ID와 EOS 상태 비교
- decode adapter의 current token device tensor `[B,1]` 값 검증
