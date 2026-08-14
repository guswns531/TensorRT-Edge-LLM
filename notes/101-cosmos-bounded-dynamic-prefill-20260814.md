# Cosmos bounded dynamic prefill chunk controller

## 결론

Cosmos-Reason2-2B의 independent prefill/decode context 경로에 profiled shape만 선택하는 bounded dynamic prefill
controller를 opt-in으로 연결했다. 첫 지원 조합은 fixed-128 engine 위의 `64/128`이며 기본값은 기존 fixed chunk다.

동일 tied embedding/LM-head engine, 동일 5개 real-request trace의 graph-off A/B에서 정상 SLO(`TPOT 50ms`)는
모두 128-token productive chunk를 유지했다. fixed-128 대비 처리량 차이는 `-0.24%` 이내, TTFT/TPOT/E2E 주요
차이는 `+0.73%` 이내였고 1,200개 요청의 greedy output text, output token 수와 finish reason이 모두 같았다.

TPOT target을 20ms로 낮춘 stress probe에서는 64-token productive row가 실제로 54개 선택되어 controller activation을
확인했다. 하지만 fixed-128보다 throughput `-0.30%`, TTFT p95 `+0.29%`, TPOT p95 `+2.61%`였으므로 단순
queue/TPOT threshold는 아직 fixed 최적점을 이기지 못한다. 따라서 bounded controller는 customization seam과 계측
기반으로 유지하고, production 기본 정책으로 승격하지 않는다.

## 정책

한 dispatch에서 사용하는 pressure는 다음과 같다.

```text
queue pressure = min(decode queue depth / max decode batch, 1)
TPOT pressure  = recent p95(decode queue wait + decode GPU event) / TPOT target
combined       = queue pressure * TPOT pressure

combined < threshold  -> 가장 큰 후보
combined >= threshold -> 가장 작은 후보
```

`--adaptiveChunkCandidates 64,128`처럼 export/build에서 검증한 길이만 후보로 받는다. 임의의 72, 96-token
productive row를 만들지 않으므로 TensorRT profile, CUDA graph와 kernel-group cost table의 shape 수가 무한히 늘지
않는다. 마지막 request tail은 후보보다 작아도 정확한 잔여 길이로 완료한다.

기본적으로 잔여 prefill이 최대 후보 안에 들어오면 한 번에 완료한다. pressure가 높더라도 100-token tail을
64+36으로 쪼개면 TensorRT enqueue가 하나 더 생겨 이득보다 손해가 컸기 때문이다. 이 동작은 실험용
`--adaptiveChunkSplitCompletion`으로만 켤 수 있다.

## 실행 흐름

```text
prefill queue                          decode queue
     |                                     |
     | remaining tokens                    | depth / Dmax
     v                                     v
+----------------+       recent decode CUDA-event p95
| bounded chunk  |<-------------------------+
| controller     |  queue pressure * TPOT pressure
+----------------+
     |
     +-- healthy ----------------------> 128-token profiled row
     +-- pressured -------------------->  64-token profiled row
     +-- final tail --------------------> exact remaining tokens
                                               |
                                               v
                                    packed prefill TRT context
```

prefill과 decode TensorRT execution context, CUDA stream, phase I/O는 기존처럼 서로 독립이다. controller는 KV
allocator나 context를 새로 만들지 않고, stable indexed-paged KV slot을 소유한 request의 다음 prefill row 길이만
결정한다.

## 구현 위치

- `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}`
  - bounded candidate 검증과 선택
  - recent decode pressure 수집
  - final completion 보호
  - 기존 continuous adaptive path와 하위 호환
- `cpp/runtime/scheduling/phaseDispatchWorker.cpp`
  - dispatch 시점의 queue/TPOT/combined pressure를 완료 metric으로 전달
- `examples/llm/llm_phase_bench.cpp`
  - candidate, threshold, completion-split CLI
  - 세 pressure를 `requests-dispatch.csv`에 기록
- `scripts/cosmos_reason2/run_real_request_kv_matrix.py`
  - real-request matrix CLI 연결
  - 실제 row 길이별 `prefill-chunk-table.csv` 생성
- `cpp/runtime/state/pipelineIO.cpp`
  - packed token carrier 용량을 per-row input limit가 아니라
    `maxPrefillBatchSize * maxPackedPrefillChunkTokens`로 검증

마지막 수정은 P8, max-chunk 256 engine의 2,048-token carrier를 per-row `maxInputLength=1024`와 잘못 비교해
거부하던 독립적인 contract bug를 해결한다. 2,049 tokens처럼 실제 profile을 넘는 carrier는 계속 거부한다.

## 동일 엔진 5-workload A/B

조건은 FP16 Cosmos-Reason2-2B, tied embedding/LM-head, indexed-paged FP16 KV 256 bundles, 80 slots, P8/D64,
independent TensorRT contexts, chunk max 128, token budget 1024, graph off, TPOT target 50ms다. 표의 변화율은
`dynamic / fixed - 1`이며 latency의 양수는 회귀다.

| Workload | Fixed / dynamic token/s | Throughput | TTFT med / p95 | TPOT med / p95 | E2E p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| short, 48 req | 1967.32 / 1965.09 | -0.113% | -0.257% / +0.155% | +0.730% / +0.594% | +0.098% |
| balanced, 288 req | 3977.55 / 3974.81 | -0.069% | -0.085% / +0.105% | -0.016% / +0.306% | +0.067% |
| decode-heavy, 288 req | 4511.90 / 4501.21 | -0.237% | +0.335% / +0.292% | +0.272% / +0.305% | +0.251% |
| long-prefill, 288 req | 1279.37 / 1279.25 | -0.010% | +0.060% / +0.018% | +0.004% / +0.142% | +0.009% |
| bimodal, 288 req | 1832.61 / 1831.49 | -0.061% | +0.030% / +0.064% | +0.035% / +0.132% | +0.066% |

정상 SLO에서는 모든 non-final productive row가 128이었다. 이는 “동적 기능이 실행되지 않았다”가 아니라 decode
pressure가 건강할 때 fixed-128과 같은 선택을 해 불필요한 추가 enqueue를 만들지 않았다는 뜻이다.

## Activation stress probe

balanced trace에서 TPOT target만 20ms로 낮추면 64-token productive row 54개와 128-token row 12개가 선택됐다.

| Metric | Fixed-128 | Dynamic 64/128 | 변화 |
| --- | ---: | ---: | ---: |
| Generated token/s | 3977.55 | 3965.69 | -0.30% |
| TTFT median / p95 | 1908.55 / 4392.38ms | 1914.05 / 4404.94ms | +0.29% / +0.29% |
| TPOT median / p95 | 17.322 / 20.522ms | 17.224 / 21.058ms | -0.57% / +2.61% |
| E2E p95 | 5402.01ms | 5407.90ms | +0.11% |

작은 chunk가 decode 간섭 시간을 줄이더라도 prefill dispatch 수, packing과 TensorRT enqueue가 늘어난다. 현재
부하에서는 이 추가 비용이 TPOT tail 이득보다 컸다.

## max-256 탐색 결과와 공정성 제한

P8/D64 max-chunk 256 engine도 export -> build -> inference 순서로 검증했다. packed carrier contract 수정 후 12개
correctness request는 기존 128/256 결과와 `0/12` mismatch였고, 실행 후 GPU free memory는 약 857MiB였다.

다만 이 max-256 engine은 tied LM-head export가 아니라 기존 non-tied ONNX에서 만들었다. 동일 binary라도 max-128
주 엔진보다 GPU weight가 약 576MiB 더 많고 TensorRT tactic도 다를 수 있으므로, 아래 관측은 정책 결론에 쓰지 않는다.

- fixed-256은 fixed-128 대비 workload에 따라 throughput `-2.25%`에서 `+7.68%`까지 변했다.
- completion split threshold sweep은 fixed-256 대비 최대 약 `+0.07%`였고, 적극적인 split은 TPOT p95를 최대
  `+4.91%` 악화시켰다.
- tied max-256을 동일 export 조건으로 다시 만들기 전에는 128/256 workload router를 채택하지 않는다.

## 검증과 artifact

- `PhaseQueueSchedulerTest.*`와 `PipelineIOTest.*`: 62개 통과
- Python matrix runner syntax check 통과
- 5-workload fixed/dynamic output identity: 1,200/1,200
- max-256 correctness: 12/12
- 원시 결과: `.local/cosmos-reason2-2b/adaptive-chunk-20260814/`
- max-256 ONNX/engine:
  - `.local/cosmos-reason2-2b/onnx-fp16-packed-max256/llm`
  - `.local/cosmos-reason2-2b/engine-fp16-packed-max256-p8-d64-mb80-b256`

## 다음 단계

다음 controller는 단순 threshold가 아니라 `(P batch, chunk, D batch, past-KV bucket)` kernel-group cost table로 후보별
예상 makespan과 decode debt를 계산해야 한다. 순서는 다음과 같다.

1. tied max-256 ONNX/engine을 동일 조건으로 export/build한다.
2. 64/128/256 각 shape의 solo/overlap 비용과 CUDA graph hit/miss 비용을 수집한다.
3. 후보마다 `predicted prefill completion gain - predicted decode debt - extra enqueue cost`를 계산한다.
4. replay simulator에서 후보 정책을 평가한 뒤 실제 5-workload 3회 반복 gate를 통과시킨다.
5. 이득이 재현되는 workload에서만 dynamic을 기본값으로 승격하고, 그 외에는 fixed-128을 유지한다.
