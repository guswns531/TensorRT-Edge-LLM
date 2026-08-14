# Cosmos vLLM, clean upstream, Current 재측정

## 결론

디스크 정리 후 동일한 Cosmos-Reason2-2B balanced real-request trace를 vLLM, clean upstream, Current에서 새로
측정했다. 모든 성공 run은 요청 288개, prompt 25,872 tokens, requested output 24,960 tokens, 실제 생성
23,712 tokens로 같다.

- Current default는 vLLM보다 generated token/s가 `+11.47%`, TPOT p95가 `-16.68%`, E2E p95가
  `-10.49%`였다.
- Current의 TTFT p95는 vLLM보다 `-11.50%`지만 TTFT median은 `+2.44%`로 느리다. throughput을 위해
  prefill을 모으는 구간의 median wait가 남아 있다.
- tied shared-layout은 Current default보다 처리량이 `-0.79%`지만 574MiB를 절감하고, vLLM 대비 처리량
  `+10.59%`를 유지했다.
- clean upstream fixed-BS8 oracle은 1,167.4 token/s로 과거 수치와 재현됐다. Current는 3.917배지만 이는
  kernel-only 비교가 아니라 continuous admission, D64 decode, independent phase overlap을 포함한 serving
  architecture 차이다.
- Current default의 세 run token/s 범위는 3.44%로 vLLM 0.34%, upstream 0.13%보다 크다. 평균 성능보다
  batch trajectory 안정화가 다음 scheduler 과제다.

## 고정 환경

| 항목 | 값 |
| --- | --- |
| GPU | RTX 3080 10GB, driver 610.43.02 |
| model | `nvidia/Cosmos-Reason2-2B`, revision `9ce19a1`, FP16 weights/KV |
| trace | SHA-256 `290d34061a173c13440e10247f55bd442a131e525a40c30a52d0b39a549d6538` |
| 요청 | 288 requests, 25,872 prompt, 24,960 requested output, 23,712 generated |
| arrival | 30 requests/s materialized trace |
| decoding | greedy, EOS enabled, prefix reuse disabled |
| 반복 | backend별 3 complete runs, 중앙값 |
| Current | commit `4e00b02`, TensorRT 11.0, P8/D64, 80 slots, indexed-paged KV |
| upstream | public commit `7f061f2`, fixed-linear BS8 |
| vLLM | 0.27.1, pinned image digest `c2f3b1b...9f31da2` |

Current와 vLLM은 localhost OpenAI-compatible HTTP/SSE client를 공유한다. vLLM은 64개 warm-up request 뒤
3회를 실행했다. Current는 각 반복마다 새 TensorRT/container lifecycle을 시작하고 frequency profile로 graph를
priming한 뒤 같은 client를 실행했다.

## HTTP/SSE 결과

latency는 median / p95다.

| Backend | token/s | req/s | TTFT | TPOT | E2E | GPU memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current default | **4,573.3** | **55.55** | 1,870.1 / **3,697.7ms** | **13.706 / 15.623ms** | 2,995.7 / 4,643.7ms | 9,184.9MiB |
| Current tied opt-in | 4,537.3 | 55.11 | 1,874.6 / 3,779.4ms | **13.564 / 15.558ms** | **2,954.4 / 4,637.3ms** | 8,610.9MiB |
| vLLM memory-matched | 4,102.7 | 49.83 | **1,825.6** / 4,178.2ms | 17.355 / 18.751ms | 3,363.6 / 5,187.7ms | 8,154MiB |

Current default 대비 vLLM의 변화율은 다음과 같다.

| metric | Current 변화 |
| --- | ---: |
| generated token/s, request/s | `+11.47%` |
| TTFT median / p95 | `+2.44%` / `-11.50%` |
| TPOT median / p95 | `-21.02%` / `-16.68%` |
| E2E median / p95 | `-10.94%` / `-10.49%` |

세 run token/s는 Current default `4,455.5--4,608.7`, tied `4,514.3--4,594.1`, vLLM
`4,100.3--4,114.2`다. Current의 가장 느린 run도 vLLM의 가장 빠른 run보다 8.3% 빠르므로 throughput 우위는
run ordering으로 설명되지 않는다. 다만 Current의 분산은 줄여야 한다.

## 과거 결과와 비교

같은 trace의 직전 값은 Current 4,391.2 token/s, vLLM 4,134.6 token/s였다. 이번 값은 각각 `+4.15%`,
`-0.77%`다. vLLM은 latency를 포함해 대부분 1.5% 안에서 재현됐다.

Current는 TPOT median/p95가 직전보다 약 16.9% 개선됐지만 TTFT median은 13.4% 나빠졌다. code diff에서 기본
engine의 kernel이나 scheduler 정책을 직접 바꾼 변경은 없었고, 실제 dispatch 수와 overlap 궤적이 run마다 달랐다.
따라서 이번 `+4.15%`를 새 kernel speedup으로 보지 않는다. queue timing, graph shape hit, online batch formation이
decode 쪽에 유리하게 형성된 결과이며 TTFT와 교환됐다.

Current의 graph는 매 run decode 53개, 약 220MiB를 사용하고 hit rate는 약 85--88%였다. prefill은 3개,
약 14MiB, hit rate 9--12%에 불과했다. capture/launch failure와 budget rejection은 모두 0이었다.

## Clean upstream 결과와 의미

public upstream에는 continuous admission과 token streaming이 없다. 같은 288 requests를 output length별 full BS8
36개 batch로 묶고, 실제 `llm_inference` batch wall time을 측정한 뒤 original arrivals에 clairvoyant
shortest-processing-time 순서로 replay했다. 이 조건은 upstream에 유리하다.

| metric | Clean upstream oracle | Current default |
| --- | ---: | ---: |
| generated token/s | 1,167.4 | 4,573.3 |
| request/s | 14.18 | 55.55 |
| TTFT median / p95 | 6,874 / 18,488ms, optimistic estimate | 1,870 / 3,698ms, client wall |
| TPOT median / p95 | 6.198 / 6.280ms, estimated batch wall | 13.706 / 15.623ms, client wall |
| E2E median / p95 | 7,460 / 19,278ms | 2,996 / 4,644ms |
| peak GPU memory | 7,348MiB | 9,185MiB |

Current의 3.917배 throughput은 fixed-linear KV를 indexed-paged로만 바꾼 효과가 아니다. upstream은 한 번에
BS8 batch 하나를 끝까지 처리하지만 Current는 stable slots에 continuous admission하고, prefill과 최대 D64
decode를 독립 TensorRT context/stream에서 겹친다. 반대로 upstream batch 안의 request는 prefill interruption이
없어 estimated TPOT은 더 낮다. 그러므로 upstream은 correctness와 fixed-batch kernel regression oracle로 유지하고
production throughput 목표는 vLLM과 비교하는 것이 맞다.

## 메모리 상황

raw KV budget은 Current와 vLLM 모두 약 3.5GiB로 맞췄다.

- Current default와 vLLM 차이: 약 `1,031MiB`
- tied layout과 vLLM 차이: 약 `457MiB`
- tied layout의 절감: `574MiB`

즉 최근 tied embedding/LM-head 공유가 vLLM과의 메모리 차이 절반 이상을 제거했다. 남은 차이는 independent
TensorRT context workspace, phase별 I/O, 234MiB CUDA Graph cache와 runtime allocation에서 찾아야 한다. tied는
성능상 충분하지만 exact greedy identity가 별도 engine build 사이에서 성립하지 않아 아직 기본값으로 둘 수 없다.

## 현재 판단

1. balanced throughput/TPOT/E2E에서는 Current가 vLLM보다 경쟁력이 있다.
2. TTFT median은 유일하게 vLLM이 낫다. prefill queue가 작은 cohort를 너무 오래 기다리지 않도록 해야 한다.
3. Current run-to-run 분산은 vLLM보다 크다. scheduler가 host timing과 completion order에 민감하다.
4. tied sharing은 메모리 대비 성능 trade-off가 좋지만 correctness gate가 먼저다.
5. upstream의 fixed-batch kernel은 빠르지만 production serving 구조로는 확장되지 않는다. upstream을 이기는 것보다
   vLLM의 continuous batching 기능과 안정성을 따라가는 것이 더 중요한 단계다.

## 향후 우선순위

1. **Tied correctness:** teacher-forced logits allclose와 Cosmos accuracy suite를 통과시켜 574MiB 절감을 default로
   전환할 수 있는지 판정한다.
2. **Batch trajectory 안정화:** 같은 trace를 10회 실행해 dispatch shape histogram, overlap order와 token/s
   coefficient of variation을 연결하고 deterministic cohort/timeout 경계를 만든다.
3. **TTFT-aware prefill batching:** decode TPOT 우위를 유지하면서 prefill cohort wait 상한을 두어 vLLM보다 느린
   TTFT median 44.5ms를 회수한다.
4. **Tied 전용 cost table:** isolated/overlap kernel table과 engine fingerprint를 연결한 뒤 TPOT hard guard를
   재검증한다.
5. **남은 457MiB:** context workspace, phase I/O, graph cache별 actual allocation을 계측하고, 낮은 prefill graph
   hit에 쓰이는 entry와 buffer부터 줄인다.
6. **Serving 기능 비교:** repeated-prefix/COW, cancellation, request length heterogeneity가 포함된 trace를 vLLM과
   같은 HTTP frontend에서 비교한다.
7. **전체 workload gate:** 위 변경 후 short/balanced/decode-heavy/long-prefill을 같은 날 세 backend에서 다시
   실행한다.

## Artifact

- 통합 요약: `.local/cosmos-reason2-2b/three-way-recheck-20260814/comparison.json`
- 표 형식: `.local/cosmos-reason2-2b/three-way-recheck-20260814/comparison.csv`
- Current default: `.local/cosmos-reason2-2b/three-way-recheck-20260814/current-balanced/`
- Current tied: `.local/cosmos-reason2-2b/three-way-recheck-20260814/current-tied-balanced/`
- vLLM: `.local/cosmos-reason2-2b/three-way-recheck-20260814/vllm-balanced/`
- clean upstream: `.local/cosmos-reason2-2b/three-way-recheck-20260814/upstream-balanced-r1..r3/`

vLLM container는 측정 후 중지했고 pinned image는 다음 재검증을 위해 보존했다. 이미지 설치 후 host disk 여유는
46GiB다.
