# SLO Saturation 반복 기준선 고정

## 결론

권장 실행 순서의 첫 단계로 Current와 vLLM의 SLO saturation 경계를 fresh-process 조건에서 각각 5회 반복했다. 단발 실행에서 관측한 방향은 유지됐지만 격차 크기는 달라졌다.

- 39.0 offered req/s에서는 두 runtime 모두 joint-SLO 100%를 통과하며 request goodput은 Current가 0.33% 높아 사실상 동률이다.
- 48.8 offered req/s에서는 Current의 평균 SLO pass가 74.93%, vLLM은 100%다. Current의 request goodput은 vLLM보다 30.62% 낮다.
- Current의 48.8 실패는 TPOT threshold 초과가 아니다. run당 평균 61.8개가 TTFT-only failure이고 TPOT가 포함된 failure는 0개다.
- Current의 client admission/dispatch delay p95는 약 215--236 ms인 반면 vLLM은 약 1.5--2.4 ms다. 현재 가장 먼저 분해할 경로는 GPU decode kernel 하나가 아니라 `slot/admission wait -> completion visibility -> sampling/state commit -> next submit` 전체다.
- 97.5 offered req/s에서 Current는 raw throughput을 41.23 req/s까지 올리지만 SLO pass는 27.92%, goodput은 11.51 req/s로 떨어진다. raw saturation과 SLO-safe capacity가 분명히 다르다.

따라서 다음 단계는 selector score 조정이 아니라 48.8 boundary의 request/iteration timeline 계측이다. 특히 TTFT queueing이 slot lifetime, admission gate, GPU completion visibility 중 어디서 생성되는지 분리해야 한다.

## 실험 계약

모든 실행은 같은 materialized real-request trace와 다음 계약을 사용했다.

- model: `nvidia/Cosmos-Reason2-2B`, 비양자화
- Current engine: P8/D64, fixed prefill chunk 128, 80 stable slots
- Current execution: independent TensorRT E/P/D contexts, Global selector, explicit overlap warmup 비활성
- vLLM: 동일 model/checkpoint와 memory-matched VLM container
- warmup: 64 requests, output 32 tokens
- measured requests: 288
- maximum client/server in-flight: 80
- joint SLO: TTFT <= 500 ms, TPOT <= 50 ms, E2E <= 2500 ms
- TTFT/E2E 기준점: client send 시각이 아니라 trace의 scheduled arrival 시각
- 반복: 각 점 5회, Current는 매 회 backend process 재시작, vLLM은 매 회 container 재시작
- 통계: 5회 산술평균과 Student-t 95% confidence interval half-width, 자유도 4

처음 Current 실행 한 번은 존재하지 않는 vision engine 경로를 지정해 request 측정 전에 종료됐다. 경로를 바로잡은 뒤 성공 run으로 덮어썼으며 아래 통계에는 포함하지 않았다.

## 반복 결과

`mean +/- CI95`이며 latency 단위는 ms다.

### Throughput과 joint-SLO goodput

| Runtime | Offered | Raw req/s | Raw token/s | SLO pass | Goodput req/s | Goodput token/s |
|---|---:|---:|---:|---:|---:|---:|
| Current | 39.0 | 34.470 +/- 0.029 | 2987.4 +/- 2.5 | 100.00% | 34.470 +/- 0.029 | 2987.4 +/- 2.5 |
| vLLM | 39.0 | 34.355 +/- 0.002 | 2977.4 +/- 0.2 | 100.00% | 34.355 +/- 0.002 | 2977.4 +/- 0.2 |
| Current | 48.8 | 37.877 +/- 0.139 | 3282.7 +/- 12.1 | 74.93 +/- 2.78 pp | 28.383 +/- 1.134 | 2424.3 +/- 105.5 |
| vLLM | 48.8 | 40.908 +/- 0.029 | 3545.4 +/- 2.5 | 100.00% | 40.908 +/- 0.029 | 3545.4 +/- 2.5 |
| Current | 97.5 | 41.225 +/- 0.399 | 3572.8 +/- 34.6 | 27.92 +/- 0.24 pp | 11.508 +/- 0.087 | 978.8 +/- 7.8 |

39.0에서는 Current의 request/token goodput이 vLLM보다 0.33% 높다. 48.8에서는 Current의 raw request throughput이 7.41% 낮고, SLO filtering 이후 request goodput은 30.62%, token goodput은 31.62% 낮다. 반대로 표현하면 vLLM request goodput은 Current보다 44.1% 높다.

이전 단발 실행의 48.8 goodput 격차는 vLLM 기준 +64.8%였다. 5회 평균에서는 +44.1%로 줄었으므로 단발 threshold 결과를 headline으로 사용해서는 안 된다. 다만 confidence interval보다 격차가 훨씬 커서 saturation 문제 자체는 반복 가능하다.

### Request latency

| Runtime | Offered | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current | 39.0 | 34.15 +/- 2.58 | 128.97 +/- 21.21 | 13.92 +/- 0.17 | 17.36 +/- 0.35 | 1220.31 +/- 16.39 | 2088.42 +/- 24.82 |
| vLLM | 39.0 | 39.49 +/- 0.44 | 64.79 +/- 1.65 | 10.28 +/- 0.03 | 12.82 +/- 0.09 | 918.50 +/- 3.19 | 1505.41 +/- 9.66 |
| Current | 48.8 | 239.11 +/- 5.57 | 619.94 +/- 30.98 | 15.76 +/- 0.05 | 17.81 +/- 0.19 | 1580.55 +/- 7.09 | 2367.64 +/- 17.13 |
| vLLM | 48.8 | 51.44 +/- 1.21 | 84.27 +/- 2.21 | 13.53 +/- 0.12 | 17.66 +/- 0.15 | 1205.87 +/- 11.76 | 2121.30 +/- 23.45 |
| Current | 97.5 | 1268.43 +/- 26.64 | 2881.46 +/- 49.88 | 15.80 +/- 0.16 | 17.92 +/- 0.36 | 2607.22 +/- 40.36 | 4024.39 +/- 67.12 |

39.0에서 Current는 TTFT mean이 13.52% 짧지만 TTFT p95는 99.07%, TPOT mean은 35.38%, E2E mean은 32.86% 길다. 둘 다 SLO를 통과하므로 Current의 빠른 일부 first-token과 vLLM의 더 균일한 tail/iteration service가 공존한다.

48.8에서 Current의 TTFT mean은 vLLM보다 364.8%, p95는 635.6% 길다. TPOT p95 차이는 0.89%에 불과하지만 TPOT mean은 16.49% 길고, E2E mean은 31.07% 길다. 따라서 직접적인 SLO failure trigger는 TTFT queueing이고, 느린 평균 decode iteration은 slot lifetime을 늘려 admission 지연을 간접적으로 키울 수 있다.

## Failure attribution

### Current 48.8

5회의 pass request 수는 각각 208, 219, 223, 219, 210개다. run당 평균 분류는 다음과 같다.

| Failure class | Mean requests/run |
|---|---:|
| Pass | 215.8 |
| TTFT only | 61.8 |
| E2E only | 6.6 |
| TTFT + E2E | 3.8 |
| TPOT가 포함된 모든 failure | 0.0 |

### Current 97.5

5회의 pass request 수는 80, 80, 81, 81, 80개다. run당 평균은 pass 80.4, TTFT-only 51.4, E2E-only 3.0, TTFT+E2E 153.2개이며 TPOT가 포함된 failure는 없다.

이 분류는 `TPOT가 충분히 빠르므로 decode path는 문제가 아니다`라는 뜻이 아니다. TPOT threshold 50 ms가 현재 16--18 ms 수준보다 느슨하므로, decode cycle의 작은 손해는 TPOT failure가 되기 전에 KV slot 보유시간과 first-token admission queue를 증가시킬 수 있다.

## Correctness와 메모리

- Current는 모든 15회 실행에서 288/288 output을 완료했고 runtime 내부 token hash가 동일했다.
- vLLM은 모든 10회 실행에서 288/288 output을 완료했고 vLLM 실행끼리 hash가 동일했다.
- 두 runtime의 token capture/output contract가 달라 이 단계에서는 cross-runtime exact token identity를 주장하지 않는다. 이는 성능 반복 기준선이며 기존 별도 semantic/correctness gate를 대체하지 않는다.
- peak VRAM: Current 9313 MiB, vLLM 9039 MiB. Current가 274 MiB 더 사용했다.

## 단발 결과와 달라진 이유

48.8 단발 Current는 SLO pass 66.32%, goodput 24.85 req/s였지만 5회 평균은 74.93%, 28.38 req/s다. 요청들이 TTFT 500 ms와 E2E 2500 ms threshold 부근에 있어 수 ms의 queue timing으로 pass count가 이동한다. 반면 raw throughput은 37.88 +/- 0.14 req/s로 안정적이다.

따라서 이후에는 다음 규칙을 적용한다.

1. saturation promotion 판단은 최소 5회 평균과 confidence interval로 한다.
2. raw throughput, joint-SLO goodput, failure reason을 함께 보고한다.
3. scheduled-arrival latency와 client send-relative latency를 섞지 않는다.
4. 같은 trace를 재사용할 때 vLLM은 매 구현 단계마다 재실행하지 않고 이 고정 기준선을 사용한다. workload, model, engine contract 또는 SLO 정의가 바뀔 때만 fresh vLLM을 다시 실행한다.

## 다음 권장 단계

48.8 offered req/s에서 다음 timestamp를 request 및 decode iteration별로 연결한다.

```text
scheduled arrival
  -> client admission / slot acquisition
  -> D enqueue
  -> first D kernel start
  -> D kernel complete
  -> completion observed by CPU
  -> sampling scheduled / complete
  -> request state committed
  -> next D candidate formed
  -> scheduler selects D
  -> TensorRT enqueue
  -> next D kernel start
```

분해 우선순위는 다음과 같다.

1. `scheduled arrival -> admission`과 free-slot visibility
2. `GPU complete -> CPU observe`
3. sampling과 state commit
4. candidate formation과 selector CPU time
5. TensorRT enqueue와 first-kernel launch gap
6. isolated D kernel-group cost와 vLLM iteration service 비교

이 분해가 끝나기 전에는 formation-aware score나 workload별 threshold를 추가하지 않는다.

## 산출물

- 집계 CSV: `benchmarks/phase_serving/results/slo-goodput-repeat-baseline-20260830.csv`
- Current raw runs: `.local/transition-aware-20260830/p6-repeat-baseline/current`
- vLLM raw runs: `.local/transition-aware-20260830/p6-repeat-baseline/vllm`
- joint-SLO per-run 분석: `.local/transition-aware-20260830/p6-repeat-baseline/joint-slo-runs.json`
