# Request-level E/P/D critical-path attribution

## 결론

정책을 바꾸지 않고 요청별 E/P/D host timeline을 완성한 뒤 Cosmos Reason2-2B의 기존 12개 real-request
HTTP workload를 모두 다시 계측했다. 이번 결과에서 가장 중요한 결론은 다음과 같다.

1. `fixed P128`과 P chunk 사이 공백은 현재 주 병목이 아니다. 가장 큰 평균 P chunk gap도 long-prefill의
   `4.6 ms`다.
2. long-prefill과 bimodal의 TTFT 문제는 P kernel 시간이 아니라 KV page pool 고갈 뒤의 admission 대기다.
   두 workload만 available KV page가 `0`까지 내려갔고, submit-to-admit가 각각 평균 `1,940.8 ms`와
   `1,791.6 ms`를 차지했다.
3. VLM burst의 TTFT는 E kernel보다 E queue가 지배한다. vision-heavy vision 요청은 E active가 평균
   `67.8 ms`인 반면 E queue가 `1,429.8 ms`다.
4. text-heavy/mixed/vision-heavy의 긴 E2E는 request별 D dispatch 사이 service gap이 크다. 이 값은 GPU
   idle 시간이 아니라 해당 request가 다른 cohort/phase에 서비스를 양보한 시간이다.
5. HTTP adapter/tokenization/image preparation은 기존 `vision_queued`보다 앞에 있다. burst workload에서
   이 미계측 전단은 평균 `88--183 ms`, vision tail에서는 `210--310 ms`까지 커진다.
6. 모든 workload에서 planned/actual action fidelity violation, page-growth wait, phase memory broker
   backpressure는 `0`이었다. 다만 이 실행의 memory broker는 disabled이고 stable page allocator 자체는
   동적으로 lease/release한다. 따라서 KV page가 0이 된 admission stall은 broker telemetry에 잡히지 않았다.

따라서 다음 최우선 구현은 workload별 score tuning이 아니다. pending request의 required/guaranteed KV
page, 현재 free page, first-token slack을 global snapshot에 넣고 initial admission block reason과 시간을
직접 측정한 뒤, 동일한 profile-free admission policy로 long-prefill/bimodal을 고치는 것이다.

## 구현

### C++ timeline 변경

기존 `IndependentPhaseAsyncServer`는 request별 최초 `decode_start`와 `decode_done`만 전달했다. 이후 decode
step은 timeline에서 제거되어 첫 D 이후 completion까지가 하나의 불명확한 tail로 남았다. 이 deduplication을
제거해 모든 D dispatch의 start/done과 dispatch index, batch size를 보존했다.

변경한 파일은 다음과 같다.

- `cpp/runtime/scheduling/independentPhaseAsyncServer.cpp`
- `cpp/runtime/scheduling/independentPhaseAsyncServer.h`
- `cpp/runtime/scheduling/phaseTimeline.h`

이 변경은 candidate, selector, admission, batch formation, CUDA dispatch를 바꾸지 않는다. timeline callback이
설정된 진단 실행에서만 추가 record가 외부로 전달된다.

### 분석기

`benchmarks/phase_serving/analyze_phase_timeline.py`를 추가했다. 분석기는 다음 문제를 처리한다.

- server shape warmup의 `request_index >= 1,000,000` 제외
- HTTP warmup과 production trace가 같은 request ID를 재사용하는 경우 completion으로 lifecycle을 나누고
  마지막 lifecycle 선택
- dispatch index로 반복 P/D start-done pairing
- text와 vision request 분리
- client `requests.csv`와 request ID를 결합해 backend 앞/뒤 HTTP 구간 분리
- 요청 CSV, all/text/vision aggregate JSON, client TTFT 상위 5% tail aggregate 생성
- incomplete lifecycle과 class mismatch 검출

단위 테스트는 text, vision, chunked P, warmup ID filtering, reused lifecycle, HTTP client join의 6개 경우를
검증한다.

## 시간 분해의 의미

첫 토큰은 첫 decode가 아니라 마지막 prefill sampling에서 생성된다. 따라서 TTFT와 E2E는 다음처럼 겹치지
않게 분해한다.

```text
HTTP send
  | frontend / request-adapter / tokenization / image preparation
backend arrival
  | [vision] E queue -> E active -> E/P handoff
  | server submit -> admission
  | P initial queue -> sum(P active) + between-chunk gaps
  | prefill sampling
first token
  | first D entry wait
  | sum(D active) + between-dispatch service gaps
  | completion tail
backend completion
  | gateway / HTTP completion delivery
HTTP completion
```

각 request에서 다음 두 invariant를 검사했다.

```text
backend TTFT
  = E/P critical-path components + critical_path_residual

backend E2E
  = backend TTFT + D components + e2e_residual
```

12개 workload의 모든 production request에서 두 residual은 부동소수점 합산 오차 수준이었다. 예를 들어
short smoke의 residual p95는 약 `1e-14 ms`였다.

`P/D active`는 host enqueue부터 CUDA completion 확인까지의 request-visible interval이다. 순수 GPU event
시간은 기존 `PHASE_METRIC`에 계속 남는다. `D gap`은 한 request의 연속 D dispatch 사이 시간이며 GPU가
놀았다는 뜻이 아니다. 다른 cohort, P/E action, sampling completion이 그 사이에 실행될 수 있다.

## 실행 조건과 correctness

- model: `nvidia/Cosmos-Reason2-2B`
- Current: independent E/P/D TensorRT contexts, shared CUDA context
- maximum phase batch: P8 / D64 / E4
- fixed prefill chunk: 128
- maximum in-flight: 64, stable slots: 80, KV pool pages: 256
- HTTP real request/arrival/output contract: 기존 12-workload trace와 동일
- 새 진단 플래그: `TRT_EDGELLM_EMIT_PHASE_METRICS=1`
- repeats: attribution용 각 1회

모든 12개 실행이 요청 수와 requested output token 수를 완성했고 이전과 같은 workload별 token hash를
냈다. 총 `443,677` timeline event를 분석했으며 production lifecycle은 모두 complete였다.

| workload | complete requests | timeline events including warmup |
|---|---:|---:|
| short | 48/48 | 6,656 |
| balanced | 288/288 | 55,536 |
| decode-heavy | 288/288 | 155,376 |
| long-prefill | 288/288 | 59,104 |
| bimodal | 288/288 | 95,752 |
| text-heavy | 64/64 | 11,584 |
| mixed | 64/64 | 10,728 |
| vision-heavy | 64/64 | 9,876 |
| poisson | 64/64 | 14,144 |
| wave/drain | 20/20 | 6,132 |
| multi-image | 5/5 | 5,037 |
| late-vision D24 | 32/32 | 13,752 |

빌드와 테스트 결과:

- `llm_phase_context_smoke`, `unitTest` 재빌드 통과
- `IndependentPhaseAsyncServerTest.*`: 25/25 통과
- Python attribution tests: 6/6 통과
- 전체 C++ suite: 1,143개 중 1,099 pass, 42 skip, 2 fail. 실패는 변경 전부터 SM86에서 재현된
  `InitializeYarnRopeCosSin.Accuracy`와 `InitializeMRopeCosSin.Accuracy` tolerance 두 건이다.

## 계측 실행의 client-visible 결과

다음 표는 per-dispatch timeline을 켠 진단 1회 값이다. 성능 headline이 아니라 attribution run 자체의
상태를 확인하는 표다.

| workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 2,465.24 | 83.49 | 170.61 | 13.90 | 27.13 | 334.54 | 416.19 |
| balanced | 4,461.21 | 67.45 | 165.31 | 12.34 | 13.65 | 1,119.66 | 1,726.35 |
| decode-heavy | 5,157.13 | 62.84 | 173.50 | 10.81 | 11.27 | 2,854.97 | 4,327.67 |
| long-prefill | 1,140.53 | 2,113.61 | 2,713.24 | 28.40 | 33.08 | 4,547.39 | 6,445.10 |
| bimodal | 1,873.96 | 1,940.46 | 3,935.30 | 18.54 | 29.89 | 4,494.70 | 9,376.30 |
| text-heavy | 1,916.49 | 325.74 | 1,117.44 | 25.84 | 39.37 | 1,659.74 | 1,758.54 |
| mixed | 1,112.71 | 732.24 | 2,098.40 | 32.45 | 42.11 | 2,269.57 | 2,559.99 |
| vision-heavy | 682.89 | 1,389.67 | 3,138.23 | 28.34 | 37.39 | 2,512.17 | 3,538.70 |
| poisson | 1,960.25 | 189.72 | 725.99 | 22.06 | 40.82 | 1,594.63 | 2,041.66 |
| wave/drain | 97.80 | 208.19 | 308.28 | 9.64 | 11.52 | 507.16 | 517.00 |
| multi-image | 278.37 | 244.38 | 343.00 | 9.83 | 11.99 | 549.17 | 571.81 |
| late-vision D24 | 2,487.07 | 115.76 | 458.24 | 9.49 | 9.55 | 1,475.27 | 1,853.61 |

## 계측 오버헤드

모든 D row의 매 dispatch record를 JSON line으로 gateway log에 쓰므로 이번 12개 gateway log만 약
`127 MiB`다. 비계측 Current 결과인 Note 178과 비교한 변화율은 다음과 같다. latency의 양수는 느려진
것이다.

| workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | -2.0% | -0.9% | +2.5% | +3.8% | +22.1% | +2.0% | +2.1% |
| balanced | -2.7% | +1.5% | +0.6% | +2.8% | +2.9% | +2.8% | +2.4% |
| decode-heavy | -2.4% | -2.7% | -3.8% | +2.6% | +1.7% | +2.4% | +1.9% |
| long-prefill | -2.1% | +1.2% | +1.3% | +2.7% | +4.3% | +2.0% | +2.4% |
| bimodal | -3.1% | +3.5% | -3.8% | +3.3% | +0.9% | +3.8% | +1.7% |
| text-heavy | -2.5% | +4.3% | +4.0% | +2.6% | +1.0% | +2.8% | +2.6% |
| mixed | -1.5% | +1.2% | +0.8% | +0.9% | +0.1% | +1.1% | +1.2% |
| vision-heavy | -1.5% | +0.8% | -1.3% | +5.6% | +1.0% | +2.4% | +1.2% |
| poisson | -0.5% | -4.3% | -2.3% | +0.8% | +0.8% | +0.3% | +0.9% |
| wave/drain | -0.1% | +1.9% | +1.7% | +1.5% | +1.0% | +1.7% | +1.3% |
| multi-image | -10.5% | +16.8% | +16.0% | +4.7% | -6.4% | +10.3% | +11.7% |
| late-vision D24 | -2.4% | +1.7% | +0.9% | +2.5% | +2.3% | +2.4% | +2.4% |

대부분의 throughput 영향은 `0.1--3.1%`다. multi-image는 요청이 5개뿐이고 warmup/formation 경계에
민감해 1회 계측값을 성능 gate로 사용하지 않는다. 해당 workload의 비계측 3회 결과는 `310.89 tok/s`,
TTFT p95 `295.66 ms`, E2E p95 `511.75 ms`다. production 기본값에서는 metrics callback을 연결하지
않으므로 이 오버헤드는 없다. 장기적으로 production sampling이 필요하면 per-dispatch JSON이 아니라
request completion 시 bounded aggregate 하나를 내보내야 한다.

## 평균 critical-path cost와 batch density

단위는 ms다. `F`는 HTTP send부터 backend arrival까지의 전단 차이, `A`는 server submit-to-admit,
`Eq/Ea`는 vision request에서만 계산한 E queue/active, `Pq/Pa`는 P initial queue/active, `Dq`는 first
token-to-first-D, `Da/Dg`는 D active/inter-dispatch gap이다. E와 D가 없는 request의 null 값은 평균에서
제외했다. PBS/DBS는 각 request가 경험한 dispatch batch size의 평균이다.

| workload | F | A | Eq | Ea | Pq | Pa | Dq | Da | Dg | PBS | DBS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 6.9 | 0.0 | - | - | 51.5 | 23.4 | 91.0 | 151.7 | 8.1 | 6.3 | 40.1 |
| balanced | 9.3 | 0.0 | - | - | 34.9 | 20.9 | 55.5 | 715.8 | 276.5 | 4.9 | 61.1 |
| decode-heavy | 7.3 | 0.0 | - | - | 31.2 | 21.5 | 56.0 | 2,353.3 | 377.5 | 5.2 | 61.4 |
| long-prefill | 3.9 | 1,940.8 | - | - | 51.4 | 110.4 | 68.9 | 921.2 | 1,437.7 | 3.2 | 34.2 |
| bimodal | 4.0 | 1,791.6 | - | - | 73.8 | 65.6 | 140.8 | 1,586.8 | 821.9 | 3.1 | 35.8 |
| text-heavy | 111.7 | 0.0 | 546.5 | 87.8 | 31.0 | 21.2 | 128.0 | 456.9 | 747.0 | 3.4 | 55.8 |
| mixed | 132.9 | 0.0 | 972.7 | 71.7 | 51.5 | 22.0 | 53.3 | 395.2 | 1,089.1 | 2.5 | 42.2 |
| vision-heavy | 149.6 | 0.0 | 1,429.8 | 67.8 | 85.1 | 24.8 | 96.8 | 313.1 | 712.0 | 1.5 | 25.4 |
| poisson | 22.4 | 0.0 | 367.9 | 76.3 | 34.8 | 18.4 | 55.8 | 592.3 | 755.5 | 2.2 | 49.2 |
| wave/drain | 30.8 | 0.0 | 32.8 | 80.2 | 29.2 | 31.0 | 6.9 | 209.2 | 83.9 | 1.4 | 4.8 |
| multi-image | 34.8 | 0.0 | 43.4 | 80.1 | 51.9 | 31.4 | 5.4 | 230.5 | 70.4 | 1.4 | 4.4 |
| late-vision D24 | 23.6 | 0.0 | 106.0 | 91.0 | 24.6 | 15.5 | 12.8 | 1,337.1 | 460.3 | 5.8 | 23.9 |

long-prefill/bimodal 외에는 admission 평균이 반올림 오차 수준이다. long-prefill과 bimodal은 P가 각각
평균 6.2/3.8번 chunk dispatch되지만 chunk 사이 gap은 4.6/2.9ms에 불과하다. TTFT를 줄이기 위해
P128을 바꾸는 것보다 admission을 먼저 고쳐야 한다.

balanced/decode-heavy는 DBS `61.1/61.4`로 D64에 거의 도달한다. long-prefill/bimodal은
`34.2/35.8`, text-heavy/mixed/vision-heavy는 `55.8/42.2/25.4`로 workload의 ready cohort와 phase
contention에 따라 자연스럽게 낮아진다. 이 정보는 workload label 없이 current ready state에서 직접
관측된 결과다.

## Memory와 action fidelity

| workload | minimum available KV pages | minimum available phase slots | memory backpressure | action fidelity violations |
|---|---:|---:|---:|---:|
| short | 188 | 16 | 0 | 0 |
| balanced | 121 | 0 | 0 | 0 |
| decode-heavy | 19 | 0 | 0 | 0 |
| long-prefill | 0 | 28 | 0 | 0 |
| bimodal | 0 | 10 | 0 | 0 |
| text-heavy | 80 | 0 | 0 | 0 |
| mixed | 109 | 16 | 0 | 0 |
| vision-heavy | 137 | 32 | 0 | 0 |
| poisson | 72 | 0 | 0 | 0 |
| wave/drain | 164 | 48 | 0 | 0 |
| multi-image | 164 | 48 | 0 | 0 |
| late-vision D24 | 188 | 36 | 0 | 0 |

KV pool은 128-token page 256개다. stable ownership은 request에 page를 lease하고 completion에서 반환한다.
long-prefill/bimodal은 slot이 남아 있는데 page만 0이 됐다. 즉 메모리 fragmentation이나 stable slot
exhaustion보다 page capacity/lifetime이 initial admission을 막은 것이다. decode-heavy는 19 page까지
내려가지만 0은 아니어서 같은 초 단위 admission stall이 없다.

`phase_memory_backpressure=0`은 메모리 문제가 없다는 뜻이 아니다. 현재 command에서 memory broker가
disabled라 decision counter가 움직이지 않은 것이다. page allocator의 hard feasibility는 별도 경로에서
계속 적용된다. 다음 계측은 `pending required pages`, `initial admission block reason`, `nearest guaranteed
reclaim`을 같은 snapshot에 넣어 이 간극을 없애야 한다.

## Workload별 분석

### 1. Short

- client TTFT mean/p95: `83.49 / 170.61 ms`
- backend TTFT 평균 `76.6 ms` 중 P initial queue가 `51.5 ms`, P active가 `23.4 ms`다.
- client TTFT tail request의 평균 P queue는 `140.7 ms`다. 짧은 prompt의 kernel보다 dispatch turn을
  기다리는 시간이 tail을 만든다.
- D는 평균 BS40.1이고 첫 D 진입 대기 `91.0 ms`, active `151.7 ms`, gap `8.1 ms`다.
- 비계측 Current는 cached fresh vLLM보다 throughput `+26.8%`, TTFT mean `-51.9%`, E2E p95
  `-19.1%`다. short 전용 knob를 추가할 이유는 없다.

### 2. Balanced

- client TTFT mean/p95: `67.45 / 165.31 ms`
- 평균 P queue/active는 `34.9 / 20.9 ms`, TTFT tail P queue 평균은 `160.4 ms`다.
- D 평균 BS는 `61.1`로 D64에 근접한다. D active `715.8 ms`와 request service gap `276.5 ms`가 E2E를
  구성한다.
- D batch formation은 이미 양호하다. 개선한다면 P oldest-slack dispatch와 D request service gap을
  건드려야 하며 D batch를 무조건 더 기다리게 하면 안 된다.
- 비계측 Current는 vLLM보다 throughput `+5.8%`, E2E mean/p95 `-5.3/-4.5%`다.

### 3. Decode-heavy

- client TTFT mean/p95: `62.84 / 173.50 ms`
- P queue/active `31.2 / 21.5 ms`; tail P queue `160.0 ms`다.
- 평균 D BS `61.4`, active `2,353.3 ms`, gap `377.5 ms`다. D gap은 D span의 약 14%로 balanced보다
  상대적으로 작다. D64 refill이 잘 작동하는 positive control이다.
- KV page는 최소 19까지 내려가지만 고갈되지 않는다.
- 비계측 Current는 vLLM보다 throughput `+6.4%`, TPOT mean/p95 `-3.9/-4.2%`, E2E p95 `-4.9%`다.

### 4. Long-prefill

- client TTFT mean/p95: `2,113.61 / 2,713.24 ms`
- submit-to-admit 평균 `1,940.8 ms`가 backend TTFT의 약 92%다. client TTFT 상위 5% 요청에서는 평균
  `2,851.9 ms`다.
- P active는 평균 `110.4 ms`, chunk gap은 `4.6 ms`다. fixed P128이나 packed P kernel이 TTFT
  2초의 원인이 아니다.
- available KV page가 0이지만 stable slot은 최소 28개 남는다. page lifetime/capacity가 admission을
  결정한다.
- D 평균 BS가 `34.2`로 낮아지고 D service gap이 `1,437.7 ms`다. page 고갈과 긴 P가 D cohort
  continuity에도 영향을 준다.
- Current는 vLLM보다 throughput/E2E는 좋지만 TTFT mean은 `+9.3%` 느리다. profile-free page-aware
  admission의 첫 promotion workload다.

### 5. Bimodal

- client TTFT mean/p95: `1,940.46 / 3,935.30 ms`
- submit-to-admit 평균은 `1,791.6 ms`, client TTFT 상위 5%에서는 `4,015.0 ms`다.
- P active/chunk gap은 `65.6 / 2.9 ms`뿐이다.
- KV page가 0까지 가고 stable slot은 10개 남는다. long prompt와 long-lived decode가 page ownership을
  길게 유지하면서 short request까지 admission 뒤에 세우는 것이 핵심이다.
- 평균 D BS `35.8`, D active/gap `1,586.8/821.9 ms`; tail request는 `2,597.4/1,631.2 ms`다.
- Current의 명확한 vLLM 약점이다. TTFT mean/p95가 vLLM보다 `+19.7/+54.8%` 느리다. 반면 throughput은
  `+5.1%`, E2E mean은 `-8.3%`이므로 전체 효율을 버리지 않고 admission ordering과 page guarantee만
  고쳐야 한다.

### 6. Text-heavy

- text client TTFT `149.9 ms`, backend TTFT `38.5 ms`: adapter/frontend가 평균 `111.4 ms`다.
- vision client TTFT `853.4 ms`: frontend `112.6`, E queue `546.5`, E active `87.8`, P queue `70.7`,
  P active `27.9 ms`가 주 구성이다.
- vision tail은 frontend `209.3`, E queue `804.9 ms`다. E batching의 kernel gain보다 request age가
  중요하다.
- text request의 D active/gap은 `511.4/911.1 ms`다. text first token은 빠르지만 VLM phase와 함께
  실행되는 동안 token service interval이 길어진다.
- Current는 vLLM보다 throughput `+20.3%`, TTFT mean/p95 `-25.9/-12.8%`, E2E p95 `-15.9%`다.

### 7. Mixed

- text client/backend TTFT는 `122.5/34.3 ms`; frontend/adapter가 `88.2 ms`다.
- vision client TTFT는 `1,342.0 ms`; frontend `177.5`, E queue `972.7`, E active `71.7`, P queue
  `86.9`, P active `27.9 ms`다.
- vision tail E queue는 평균 `1,904.7 ms`다.
- text D active/gap은 `518.2/1,724.5 ms`, vision은 `272.1/453.6 ms`다. 전체 D batch 평균 BS42.2다.
  gap은 GPU idle이 아니라 서로 다른 E/P/D/cohort 사이 request-level service gap이다.
- Current는 vLLM보다 throughput `+22.5%`, TPOT p95 `-49.9%`, E2E mean/p95 `-25.4/-19.5%`다.
  현재 architecture의 강점을 보이는 positive control이지만 text D continuity는 더 개선할 여지가 있다.

### 8. Vision-heavy

- vision client TTFT mean/p95: `1,824.3 / 3,190.8 ms`
- 평균 frontend `183.1`, E queue `1,429.8`, E active `67.8`, P queue `106.4`, P active `28.1 ms`다.
- tail vision request는 frontend `309.8`, E queue `2,865.1 ms`다. E active는 오히려 `38.6 ms`라 큰
  E batch가 kernel 효율은 높였지만 기다린 시간을 상쇄하지 못했다.
- text backend TTFT는 `36.7 ms`로 낮지만 client adapter 전단이 `49.0 ms` 추가된다. text D gap은
  `1,625.5 ms`다.
- Current는 vLLM보다 throughput `+19.7%`, E2E mean/p95 `-40.5/-17.3%`다. 다음 개선은 E batch size
  고정 tuning이 아니라 oldest vision slack에 의해 formation wait를 끝내는 것이다.

### 9. Poisson

- text client/backend TTFT `61.9/42.1 ms`, vision은 `573.3/543.1 ms`다. burst trace보다 frontend
  queue가 작아 text `19.8`, vision `30.3 ms`다.
- vision E queue/active `367.9/76.3 ms`, tail E queue `635.9 ms`다.
- 전체 평균 D BS는 `49.2`; text D active/gap `693.7/874.2 ms`, vision `288.1/399.1 ms`다.
- 동일 scheduler가 arrival smoothing에 따라 자연스럽게 latency 쪽으로 이동한 사례다.
- Current는 vLLM보다 throughput `+9.4%`, TTFT mean `-54.7%`, E2E p95 `-10.7%`다.

### 10. Wave/drain

- client TTFT mean/p95: `208.19 / 308.28 ms`
- 평균 E queue/active `32.8/80.2 ms`로 E active가 queue보다 크다. 전체 E batching을 즉시화할 필요는
  없다.
- tail 한 request의 E queue는 `167.2 ms`; batch gain 없는 마지막 request만 age/slack으로 보호하면 된다.
- D 평균 BS4.8, active/gap `209.2/83.9 ms`다. workload 자체가 작은 wave라 D64를 목표로 기다리면 안 된다.
- Current는 vLLM보다 throughput `+2.1%`, TTFT p95 `-27.6%`, E2E p95 `-21.4%`다.

### 11. Multi-image

- 요청이 5개뿐이므로 p95와 tail aggregate는 사실상 한 request를 가리킨다.
- 평균 E queue/active `43.4/80.1 ms`, P queue/active `51.9/31.4 ms`, D BS4.4다.
- 마지막 request의 E queue가 `216.8 ms`로 tail을 만든다. wave와 같은 bounded-oldest-wait 문제다.
- 이번 instrumentation 1회는 JSON/timing perturbation에 민감했으므로 성능 판단은 비계측 3회 결과를
  사용한다. 비계측 Current는 vLLM보다 throughput `+27.1%`, TTFT p95 `-26.6%`, E2E p95 `-21.7%`다.

### 12. Late-vision D24

- text client TTFT mean/p95 `39.5/51.0 ms`, vision `344.4/497.7 ms`다.
- vision 평균 frontend `61.4`, E queue `106.0`, E active `91.0`, P queue `55.9`, P active `25.7 ms`다.
- vision tail은 frontend `83.7`, E queue `206.7`, P queue `98.8 ms`다. 이미 돌고 있는 D24와 E/P
  placement가 tail에 직접 보인다.
- text D 평균 BS23.9, D active/gap `1,337.1/460.3 ms`다. 의도한 D24 cohort가 형성된다.
- Current는 vLLM보다 throughput `+8.0%`, TTFT p95 `-28.1%`, E2E p95 `-7.4%`다.

## 비계측 Current와 cached fresh vLLM

model, trace SHA, arrival/output contract가 바뀌지 않았으므로 vLLM은 Note 169/178의 fresh 3회 결과를
재사용한다. 아래 변화는 비계측 Current 기준이며 latency 음수는 Current가 낮다는 뜻이다.

| workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | +26.8% | -51.9% | -37.0% | +0.2% | -10.6% | -23.1% | -19.1% |
| balanced | +5.8% | -41.2% | -41.4% | -1.1% | -1.8% | -5.3% | -4.5% |
| decode-heavy | +6.4% | -45.2% | -42.8% | -3.9% | -4.2% | -5.7% | -4.9% |
| long-prefill | +3.1% | +9.3% | -7.2% | -13.9% | -14.4% | -3.9% | -4.5% |
| bimodal | +5.1% | +19.7% | +54.8% | -22.3% | -20.1% | -8.3% | -1.5% |
| text-heavy | +20.3% | -25.9% | -12.8% | -13.7% | -17.7% | -17.0% | -15.9% |
| mixed | +22.5% | -17.3% | -18.1% | -31.5% | -49.9% | -25.4% | -19.5% |
| vision-heavy | +19.7% | -19.4% | -13.9% | -57.9% | -69.0% | -40.5% | -17.3% |
| poisson | +9.4% | -54.7% | -17.7% | -1.3% | -11.4% | -11.7% | -10.7% |
| wave/drain | +2.1% | -19.1% | -27.6% | -23.6% | -33.9% | -21.8% | -21.4% |
| multi-image | +27.1% | -19.4% | -26.6% | -24.4% | -21.5% | -22.7% | -21.7% |
| late-vision D24 | +8.0% | -25.8% | -28.1% | -6.4% | -6.0% | -8.6% | -7.4% |

Current는 처리량과 E2E mean/p95에서 12/12 vLLM보다 좋다. 남은 cross-system 약점은 long-prefill TTFT
mean과 bimodal TTFT mean/p95이며, 이번 request attribution은 둘 다 KV-page-gated admission으로
수렴한다.

## 다음 구현 우선순위

### 1. Page-aware initial admission

workload label 없이 다음 상태를 global snapshot에 추가한다.

```text
pending request:
  required initial pages
  guaranteed near-term page growth
  first-token age/slack

page pool:
  free pages
  outstanding owners
  nearest guaranteed reclaim
```

그리고 pending admission마다 `admitted`, `blocked-pages`, `blocked-slot`, `blocked-SLO` reason과 block
duration을 기록한다. feasibility는 hard constraint로 유지하고, feasible request 사이 ordering만 oldest
first-token slack으로 결정한다. long request를 임의로 우회하는 workload heuristic이 아니라 current
ownership/lifetime과 request urgency만 사용한다.

첫 gate는 long-prefill/bimodal submit-to-admit mean/p95 감소, 기존 10 workload throughput ±3%, 모든
token identity, page OOB 0이다. 10GB headroom이 허용하면 KV page 256 대 320/384도 같은 mechanism에서
별도 capacity ablation으로 비교한다.

### 2. Adapter arrival stage

HTTP receive, adapter queue start/end, tokenization/image preparation start/end를 backend timeline과 같은 request
ID로 기록한다. burst에서 `F`가 100ms를 넘는 원인이 worker queue인지 tokenizer/image preparation인지
분리한다. 이 단계도 scheduling policy를 바꾸지 않는다.

### 3. Slack-bounded E formation

E wait의 예상 batch gain과 oldest vision first-token slack을 비교한다. wave/multi의 마지막 request와
vision-heavy tail을 같은 식으로 보호한다. workload name, static VLM profile, 고정 E credit을 사용하지 않는다.

### 4. Request-level D continuity

D64 형성 자체는 balanced/decode-heavy에서 이미 정상이다. 따라서 batch size를 더 키우는 대신 request별
inter-dispatch gap과 next-token slack을 selector에 제공해 mixed/text-heavy/vision-heavy에서 오래 밀린
request를 보호한다. gap을 0으로 만드는 것이 목적이 아니라 TPOT/E2E tail을 줄이면서 throughput을 보존하는
것이 gate다.

## 결과 위치

- 12-workload raw HTTP/phase logs:
  `.local/request-attribution-20260829/all-workloads-12x1`
- workload별 request CSV:
  `<workload>/run-001/attribution/request-attribution.csv`
- workload별 all/text/vision/tail aggregate:
  `<workload>/run-001/attribution/phase-attribution.json`
- full-decode smoke:
  `.local/request-attribution-20260829/smoke-short-full-decode`
