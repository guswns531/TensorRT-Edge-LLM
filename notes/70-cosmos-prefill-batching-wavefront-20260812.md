# Cosmos prefill batching, wavefront, cost model 실험

## 결론

Cosmos-Reason2-2B FP16 indexed-paged engine에서 fixed 128-token chunk를 유지한 채 prefill batch 크기를
CUDA-event cost table로 선택하는 경로를 구현했다. 기존 동작은 그대로 기본값이며, dynamic prefill,
wavefront cohort, TTFT recovery는 각각 opt-in이다.

범용 기본값은 아직 fixed-P2가 가장 안전하다. Dynamic-P는 처리량과 TTFT를 개선하지만 긴 prompt와 긴 output이
섞인 부하에서 decode TPOT tail을 크게 악화시킬 수 있다. 따라서 다음 두 profile을 구분한다.

- latency-safe: fixed chunk 128, fixed prefill token budget 256(P2), D64, independent TensorRT contexts
- throughput/TTFT: dynamic-P floor 2, cost model, 필요할 때만 wavefront 또는 TTFT recovery 활성화

## 구현 구조

요청의 최초 submit 시각은 prefill chunk 재큐잉 중에도 보존된다. 따라서 TTFT 압력은 각 chunk의 queue residence가
아니라 전체 요청 나이로 계산한다. queue residence는 별도 telemetry로 남는다.

```text
request submit
    |
    +-- submittedAt (불변) --------------------> cumulative TTFT age/slack
    |
    +-- prefill queue -> chunk 0 -> requeue -> chunk 1 -> ... -> decode queue
                           |          |
                           +----------+----------> per-dispatch queue residence

cost lookup key:
  (initial/continuation, chunk=128, past-KV upper bound,
   concurrent decode upper bound, candidate prefill BS)

selection:
  decode slack 안의 후보 중 tokens/ms 최대
  -> 후보가 없으면 decode interference 최소
  -> minDynamicPrefillBatchSize로 P1 과선택 방지
  -> prefillSloRecovery는 별도 opt-in
```

Wavefront는 최대 cohort 크기와 turn 수를 갖는다. 같은 request 집합을 여러 chunk 동안 전진시켜 past-KV frontier
편차를 줄이지만, cohort가 decode tail을 악화시킬 수 있으므로 dynamic-P와 독립된 옵션이다.

추가 telemetry는 initial/continuation/final row 수, past-KV min/mean/max/spread, 남은 prefill token,
cumulative request age, TTFT slack, 예측 prefill GPU 시간, 예측 decode slowdown, cohort 크기다. 이 값은 dispatch
CSV와 kernel-group CSV에 함께 기록된다.

## Cost table

실험 엔진은 `engine-fp16-paged-p16-d64-b256-m80`, RTX 3080 10GB, independent prefill/decode TensorRT
contexts, page size 128, fixed chunk 128이다. Bimodal과 long-prefill trace에서 P1/P2/P4/P8을 각각 96 requests로
측정해 104개 prefill point와 57개 decode point를 만들었다.

대표 prefill p95 GPU 시간은 다음과 같다.

| 상태 | concurrent D | P1 | P2 | P4 | P8 |
|---|---:|---:|---:|---:|---:|
| initial, past=0 | 0 | 13.48 ms | 14.74 ms | 21.21 ms | 35.44 ms |
| continuation, past<=512 | 0 | 15.62 ms | 18.39 ms | 28.23 ms | 49.47 ms |

P가 커질수록 prefill token 처리 효율은 좋아지지만 decode interference도 증가했다. 예를 들어 D32에서 initial
P1/P2/P4 slowdown p95는 3.54/4.80/8.36 ms, continuation past<=512에서는 3.81/5.51/10.50 ms였다.

생성기는 `scripts/cosmos_reason2/build_prefill_wavefront_cost_model.py`이며 결과는 로컬 실험 디렉터리의
`prefill-wavefront-20260812/cost-model.json`에 있다. 로컬 model/engine/실험 산출물은 Git에 넣지 않는다.

## Real-request 결과

모든 비교는 같은 engine, 96 requests, arrival 1000 req/s, P8/D64 engine limit, slots 64, page bundles 256,
fixed chunk 128, CUDA graph off 조건이다. 아래는 단일 seed의 fixed-P2 대비 변화다.

| workload | 후보 | 처리량 | TTFT p95 | TPOT p95 | E2E p95 |
|---|---|---:|---:|---:|---:|
| short | dynamic, no wavefront | +6.3% | -12.8% | +16.6% | -12.3% |
| balanced | dynamic, no wavefront | +4.4% | -9.6% | -4.2% | -4.7% |
| decode-heavy | dynamic, no wavefront | +2.5% | -5.6% | +2.8% | -2.7% |
| long-prefill | dynamic, no wavefront | +11.1% | -16.3% | +20.1% | -10.5% |
| bimodal | dynamic, no wavefront | +6.0% | -8.9% | +62.1% | -4.8% |

공격형 dynamic+wavefront+TTFT recovery를 세 seed로 반복했을 때 long-prefill 처리량 향상은
+6.3~+6.7%, TTFT p95 개선은 -8.7~-9.1%로 재현됐다. 반면 TPOT p95는 +20.5~+20.8%였다.
Balanced에서는 처리량 +2.9~+5.2%, TTFT -7.5~-10.7%, TPOT -4.3~-5.2%로 모든 주요 지표가 좋아졌다.

모든 variant의 96개 greedy output text가 해당 fixed-P2 결과와 정확히 일치했다. Past-KV spread는 wavefront가
long/bimodal에서 줄였지만, 그 자체가 E2E 개선을 보장하지는 않았다.
요약 원본 수치는 `results/cosmos-prefill-batching-20260812.csv`에 저장했다.

## Page reservation 연결

`fullReservationPromptThresholdTokens`를 추가했다. 0이면 기존 정책이며, 양수이면 threshold 이상의 긴 prompt는
항상 full reservation을 쓴다. threshold 미만의 짧은 요청만 headroom/bounded-overcommit과 adaptive growth
lease를 사용한다.

512-token threshold로 검증한 결과 long-prefill은 growth owner 0으로 완전히 안전한 full reservation을
유지했다. Bimodal은 짧은 요청에서 최대 10 growth owners가 형성됐고 정상 완료했지만 256-bundle pool에서는
추가 처리량 이득이 약 0.2%에 그쳤다. 현재 부하는 page tail보다 GPU 실행이 주 병목이라는 뜻이다.

## 권장 사용법

1. 기본 production은 fixed chunk 128 + fixed-P2로 시작한다.
2. balanced workload라면 dynamic-P floor 2를 먼저 켠다.
3. long-prefill 처리량/TTFT가 우선일 때만 wavefront와 `prefillSloRecovery`를 단계적으로 켠다.
4. TPOT p95 gate를 반드시 둔다. Bimodal에서는 throughput 개선만 보고 dynamic-P를 채택하면 안 된다.
5. cost table은 GPU, engine, context mode, chunk 크기, page 설정이 바뀔 때 다시 만든다.

## 검증

- TensorRT/CUDA build: `unitTest`, `llm_phase_bench` 성공
- Phase test: 57/57 통과
- Python scripts: `py_compile` 통과
- real-request: 5 workload x 다수 ablation, 모든 완료 output 동일
- page-aware hybrid: long/bimodal 정상 완료, pool release 후 allocated bundles 0
