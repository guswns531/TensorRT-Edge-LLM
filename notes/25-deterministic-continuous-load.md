# Deterministic continuous-load 구현과 Gemma 실험

## 구현

`PhaseContinuousLoadGenerator`는 다음 입력으로 고정 간격 arrival schedule을 한 번 만든다.

- request count와 requests/s
- prompt/output token 최소·최대
- seed와 첫 request ID

같은 config와 seed는 request ID, arrival offset, prompt 길이, output budget이 모두 같은 schedule을 만든다. runtime
loop는 CUDA 작업을 blocking wait하지 않고 event를 poll하면서 도착 시간이 지난 요청을 `submitOrQueue()`로 넣는다.
stable KV slot이 없으면 bounded pending queue에서 기다리고, terminal request의 slot이 반환되면 FIFO로 admission된다.

```text
fixed-rate arrivals
       │
       ▼
submitOrQueue ── free slot ──► prefill queue ──► decode queue ──► terminal
       │                            │                 │              │
       └── no slot ─► pending FIFO ─┴── released slot ┘              └─ slot release

host loop: arrivals → cudaEventQuery/poll → dispatchNext → repeat
```

request CSV에는 scheduled arrival, actual submit/admission, first token, terminal, TTFT, E2E, TPOT을 기록한다.
dispatch CSV에는 실제 prefill/decode batch size, queue wait, CUDA event time, overlap ratio를 기록한다.

## 서로 다른 output 길이의 batching

기존 packed decode adapter는 같은 `maxGenerateLength`만 한 batch로 묶을 수 있었다. continuous load에서는 output
budget이 서로 다르므로 이 제약을 제거했다. packed context는 batch 내 최대 budget으로 buffer를 잡되, greedy sampler는
각 source row의 원래 budget으로 finished state를 계산한다. 따라서 짧은 요청은 먼저 terminal되고 stable slot을
반환하며, 살아남은 요청은 다음 decode batch에 다시 pack된다.

## 실행 명령

```bash
./build/examples/llm/llm_phase_bench \
  --engineDir /workspace/artifacts/gemma4-e2b/engine-indexed \
  --prefillBatch 2 --decodeBatch 2 \
  --inputLen 128 --prefillChunkSize 128 --pastKVLen 128 \
  --warmup 1 --iterations 3 --contextAdapter \
  --trtContextMode independent \
  --loadRequests 12 --arrivalRate 1000 \
  --loadPromptMin 128 --loadPromptMax 128 \
  --loadOutputMin 4 --loadOutputMax 8 --loadSeed 0 \
  --loadCsv /workspace/artifacts/gemma4-e2b/perf/phase/continuous-load.csv \
  --outputCsv /workspace/artifacts/gemma4-e2b/perf/phase/continuous-load-fixed.csv
```

load 측정 전에 prefill/decode를 각 1회 실행해 TensorRT profile lazy initialization을 제외한다.

## RTX 3080 / Gemma 4 E2B INT4 indexed 결과

| 항목 | 결과 |
|---|---:|
| 요청 / generated tokens | 12 / 82 |
| 처음부터 pending이었던 요청 | 8 |
| 전체 drain 시간 | 454.323 ms |
| achieved request throughput | 26.413 req/s |
| achieved generated-token throughput | 180.488 token/s |
| TTFT median / p95 | 225.629 / 362.146 ms |
| E2E median / p95 | 254.308 / 443.323 ms |
| fixed-shape independent overlap speedup | 1.1378x |

제시한 1000 req/s는 12개 요청을 짧게 밀어 넣어 queue를 포화시키는 offered load다. achieved throughput은 전체
request가 terminal될 때까지의 실제 service throughput이다.

실제 dispatch histogram:

| Prefill BS | Decode BS | 횟수 |
|---:|---:|---:|
| 2 | 0 | 5 |
| 1 | 0 | 1 |
| 0 | 2 | 26 |
| 0 | 1 | 16 |
| 1 | 2 | 1 |

prefill row 12개, decode row 70개로 총 generated token 82개와 일치한다. prefill/decode overlap은 1 dispatch였다.
즉 두 queue의 독립 batching과 실제 overlap은 확인됐지만, 기본 queue policy가 이 부하에서 overlap을 적극적으로
선택하지는 않는다. overlap 빈도를 늘리는 것은 5단계 SLO/metrics policy 튜닝 대상이다.

## KV 메모리 관점

physical KV allocation은 시작 시 고정되어 load 도중 늘거나 줄지 않는다. request 종료 시 logical registration과
slot lease만 해제하며 KV tensor를 compact하거나 복사하지 않는다. 이 실험은 4개의 physical slot으로 12개 요청을
drain했고 마지막에 4개 slot과 registration이 모두 반환됨을 검증했다. 따라서 외부 메모리 파편화나 KV D2D copy는
발생하지 않는다. 다만 slot 내부의 사용하지 않은 sequence capacity는 fixed-linear 방식의 내부 낭비로 남는다.

## 검증

- generator 단위 테스트 4개: seed 재현성, 다른 seed, arrival boundary, invalid config
- 전체 phase 테스트 31개 통과
- heterogeneous output budget의 packed decode/scatter 검증
- Compute Sanitizer 대상 테스트 7개, error 0
- 실제 Gemma request 12개가 정확한 output budget만큼 생성하고 전부 terminal
- request CSV: `/tmp/gemma4-e2b/perf/phase/continuous-load.csv`
- dispatch CSV: `/tmp/gemma4-e2b/perf/phase/continuous-load-dispatch.csv`

TensorRT가 Gemma의 head size 512에 대해 FMHA 미지원 로그를 내지만, engine은 의도대로 FFPA prefill과 XQA decode
fallback을 선택하며 실행은 성공한다.
