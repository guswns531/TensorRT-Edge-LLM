# Final Profile-Free Contextual Phase Controller

## 1. 결론

이번 단계는 workload 이름이나 고정 `E/P/D` shape rule을 사용하지 않는 최종
production 설정을 구현하고, 실제 OpenAI-compatible HTTP request trace로 다시
검증했다.

핵심 결과는 다음과 같다.

1. 최종 canonical 12-workload에서 token throughput은 frozen vLLM보다
   `12/12` workload에서 높다. 개선 범위는 `+0.94%`에서 `+25.87%`다.
2. 48.8 offered req/s의 동일 text trace를 fresh process 5회 실행한 결과,
   Current의 median request goodput은 `41.798 req/s`로 frozen vLLM의
   `40.901 req/s`보다 `2.19%` 높다. Current는 TTFT, TPOT, E2E의 mean과
   p95도 모두 더 짧다.
3. 39와 48.8 req/s에서는 joint SLO를 모든 요청이 통과했다. 97.5 req/s에서는
   pass rate median이 `40.97%`로 내려간다. 실패는 TPOT가 아니라 arrival
   backlog가 포함된 TTFT와 E2E에서 발생한다.
4. 최신 12-workload와 39/48.8/97.5 req/s activity diagnostic에서 일반 burst
   workload의 E/P/D idle은 `1.31--3.92%`였고, wave/drain의 `70.46%`만 의도된
   arrival gap이었다. 실제 pair overlap은 workload에 따라 `0.64--39.50%`였으며,
   offered load가 증가할 때 P+D overlap은 `23.02 -> 28.98 -> 32.27%`로
   증가했다. Independent context와 contextual controller가 실제로 동시 실행을
   만들고 있으며, overlap의 주된 역할은 빈 idle을 채우는 것보다 이미 busy한
   GPU의 makespan을 압축하는 것이다.
5. Legacy 대비로는 mixed, poisson, text-heavy, vision-heavy throughput이 아직
   낮다. 따라서 현재 결과는 vLLM comparison gate는 통과하지만 모든 historical
   baseline을 지배하는 최종점은 아니다.

## 2. 최종 아키텍처

```text
OpenAI-compatible HTTP requests
             |
             v
Production async request adapter
             |
             v
Request DAG and stable ownership
  text:       P -> D -> D ... -> release KV lease
  vision: E -> P -> D -> D ... -> release vision/KV leases
             |
             v
Global ready snapshot
  E/P/D rows, request slack, outstanding contexts,
  stable slots/pages, producer provenance, memory horizon
             |
             v
Deterministic feasibility
  dependency / TensorRT profile / one in-flight per context /
  stable ownership / page reservation / memory
             |
             v
Bounded candidate frontier
  E, P, D, WAIT, E+P, E+D, P+D
             |
             v
Profile-free contextual decision plane
  P+D RLS head | E+P RLS head | E+D RLS head
  continuous features -> mean / uncertainty / conservative value
             |
             v
Lexicographic selector
  feasibility -> protected TTFT/TPOT slack -> efficiency
             |
             v
Explicit action lease
  independent E/P/D TensorRT contexts, shared CUDA context
             |
             v
CUDA start/end events and sampling completion
       |                         |
       +-> exact cost tracker    +-> contextual online update
```

Production action space는 `E/P/D/E+P/E+D/P+D/WAIT`로 제한한다. `E+P+D`는
RTX 3080에서 candidate 수와 interference surface를 불필요하게 늘리므로 넣지
않았다.

### 2.1 Mechanism과 policy

Correctness는 deterministic mechanism이 보장한다.

- request DAG dependency
- TensorRT optimization profile와 binding shape
- context별 single-inflight
- canonical row와 stable KV/page ownership
- GPU consumer 완료 전 lease 회수 금지
- action lease와 dispatch/completion event correlation

Contextual model은 이미 legal한 후보의 value만 바꾼다. illegal action을 legal하게
만들거나 exact deadline cost를 덮어쓸 수 없다.

### 2.2 Process-local online controller

`P+D`, `E+P`, `E+D`는 각각 작은 RLS posterior를 갖는다. 입력은 workload label이
아니라 현재 observable continuous state다.

```text
isolated E/P/D cost
batch fill and chunk/context shape
protected TTFT/TPOT slack
ready mass and outstanding residual
successor fill delta
ownership allocate/reclaim transition
```

모델은 CPU hot path에서 동작한다. 외부 cost registry, persisted policy state,
wall-clock TTL, workload profile은 사용하지 않는다. Exact CUDA key table은
실행 계측과 audit에만 남고 production decision representation은 아니다.

### 2.3 H2 formation preview의 상태

Execution--formation coupling과 equal-work H2 mechanism은 구현돼 있지만 최종
production gate에서는 별도 authority를 켜지 않았다. 최종 mixed diagnostic의
formation episode는 `0`이었다. 즉 최종 12-workload 결과는 benchmark별 H2
override가 아니라 공통 contextual controller의 결과다. H2는 controlled research
mode로 유지한다.

## 3. 이번에 추가한 provenance-aware completion boundary

Decode sampling 완료를 CPU가 늦게 관찰하면 text-only saturation에서 다음 D
cohort가 늦어진다. 반대로 decode event를 항상 동기화하면 vision encoder가 만든
external P critical path를 가로막는다.

최종 정책은 request provenance를 이용한다.

```text
decode sampling ticket ready?
       |
       +-- external vision P ready or external producer outstanding
       |       -> cudaEventQuery only; E/P progress를 막지 않음
       |
       `-- text-only P/D frontier
               -> concrete decode event synchronize;
                  state commit과 다음 D formation을 즉시 진행
```

이는 workload 이름에 따른 분기가 아니다. 현재 ready P가 external producer를
소비하는지와 outstanding producer row가 있는지만 본다.

코드 위치:

| 책임 | 파일 |
|---|---|
| producer provenance snapshot | `cpp/runtime/phase/mechanism/phaseReadySnapshot.h` |
| external P classification | `cpp/runtime/scheduling/phaseQueueScheduler.cpp` |
| sampling event policy | `cpp/runtime/scheduling/independentPhaseAsyncServer.cpp` |
| policy API/config | `cpp/runtime/scheduling/independentPhaseAsyncServer.h` |
| correctness test | `unittests/independentPhaseAsyncServerTest.cpp` |

Static decode-aligned admission capacity도 기본 정책에서 제거했다. 필요할 때만
`TRT_EDGELLM_ENABLE_DECODE_ALIGNED_ADMISSION=1`로 opt-in한다. 기본 admission은
stable slot/page/memory/SLO feasibility로 결정된다.

## 4. Stable ownership와 KV memory

Current는 logical row와 physical KV ownership을 분리한다.

```text
logical active rows       [r7, r2, r9, ...]
                             | stable slot IDs
                             v
physical KV/page leases   [slot 3] [slot 0] [slot 8] ...
```

Request eviction은 logical row vector와 row-to-slot mapping만 compact한다. KV
payload를 다른 slot으로 복사하지 않는다. Stable page lease는 request가 완료되고
GPU consumer event가 끝난 뒤 반환된다. Vision payload도 같은 lifetime 원칙을
사용한다.

이번 final engine의 observed residency는 다음과 같다.

| 경로 | Peak MiB | RTX 3080 headroom |
|---|---:|---:|
| text-only workload, VLM-ready process | 9237 | 1003 |
| mixed | 9427 | 813 |
| multi-image | 9439 | 801 |
| vision-heavy | 9579 | 661 |
| wave/drain | 9451 | 789 |
| frozen vLLM text saturation | 9039 | 1201 |

48.8 text trace에서 Current는 vLLM보다 `198 MiB` 더 사용한다. 이는 KV 증가가
아니라 independent vision engine/context를 함께 상주시킨 공정한 VLM-ready
계약의 비용이다.

## 5. Final 12-workload gate

공통 조건:

- model: `nvidia/Cosmos-Reason2-2B`, FP16, non-quantized
- GPU: RTX 3080 10 GiB
- P8/D64, fixed P chunk 128, stable slots 80
- shared CUDA context, independent E/P/D TensorRT contexts
- async HTTP request adapter workers 4
- active contextual P+D/E+P/E+D
- workload label, external registry, TTL 없음
- fresh process 3회, 결과는 run aggregate median
- frozen vLLM은 model/trace SHA를 검증해 재사용

### 5.1 Throughput comparison

| Workload | Current tok/s | Legacy tok/s | vs Legacy | vLLM tok/s | vs vLLM |
|---|---:|---:|---:|---:|---:|
| balanced | 4360.33 | 3564.86 | +22.31% | 4319.89 | +0.94% |
| bimodal | 1942.11 | 1748.82 | +11.05% | 1868.29 | +3.95% |
| decode-heavy | 5252.29 | 4856.78 | +8.14% | 4854.34 | +8.20% |
| late-vision | 2548.10 | 2520.86 | +1.08% | 2359.23 | +8.01% |
| long-prefill | 1159.27 | 1061.68 | +9.19% | 1120.88 | +3.43% |
| mixed | 1079.84 | 1211.65 | -10.88% | 921.48 | +17.18% |
| multi-image | 281.10 | 248.07 | +13.31% | 244.52 | +14.96% |
| poisson | 2021.28 | 2082.84 | -2.96% | 1800.07 | +12.29% |
| short | 2496.59 | 2174.53 | +14.81% | 1983.53 | +25.87% |
| text-heavy | 1977.33 | 2097.87 | -5.75% | 1634.76 | +20.96% |
| vision-heavy | 705.30 | 711.09 | -0.81% | 579.20 | +21.77% |
| wave/drain | 97.79 | 97.54 | +0.25% | 95.85 | +2.02% |

Current는 frozen vLLM throughput을 12/12에서 이겼다. Legacy에는 8/12에서
이기고 mixed, poisson, text-heavy, vision-heavy에서 진다. 이 네 workload가
다음 최적화의 regression guard다.

### 5.2 Current request latency

각 latency cell은 `mean / p95 ms`다.

| Workload | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| balanced | 62.95 / 168.32 | 12.73 / 14.42 | 1147.58 / 1822.83 |
| bimodal | 1884.29 / 3855.15 | 18.18 / 29.75 | 4324.08 / 8895.07 |
| decode-heavy | 61.53 / 183.85 | 10.61 / 11.34 | 2801.64 / 4317.09 |
| late-vision | 123.10 / 415.49 | 9.23 / 9.28 | 1443.51 / 1809.48 |
| long-prefill | 2143.84 / 2930.67 | 27.54 / 32.79 | 4489.88 / 6408.84 |
| mixed | 735.49 / 2233.33 | 37.41 / 54.67 | 2463.48 / 2643.07 |
| multi-image | 271.32 / 358.68 | 8.75 / 12.48 | 543.52 / 567.82 |
| poisson | 249.95 / 798.23 | 19.89 / 32.18 | 1531.70 / 1969.91 |
| short | 91.87 / 171.44 | 13.28 / 26.39 | 331.84 / 410.51 |
| text-heavy | 334.95 / 1106.54 | 24.77 / 39.41 | 1602.63 / 1699.72 |
| vision-heavy | 1500.57 / 3067.22 | 33.87 / 69.73 | 3032.13 / 3463.16 |
| wave/drain | 265.22 / 309.71 | 8.05 / 9.89 | 509.76 / 517.70 |

Frozen 12-workload vLLM artifact는 median/p95만 보존하고 mean request latency를
보존하지 않았으므로, 이 표에 존재하지 않는 vLLM mean을 추정하지 않았다.

### 5.3 Correctness

- long-prefill와 multi-image: 3/3 exact token identity
- multi-image: semantic pass 100%
- wave/drain: semantic pass 100%; focused run에서 FP16 encoder batch/tactic에
  의한 두 token hash branch가 관찰됨
- invalid slot, use-after-release, request failure 없음

Wave의 semantic output은 맞지만 exact cross-run identity는 최종 논문용
promotion 전에 canonical vision numerical path를 한 번 더 고정해야 한다.

## 6. 48.8 req/s repeated comparison

동일 trace SHA256:

```text
c9d64ed7e84fea87352c9b4acedb309932c8f0e9cae71d527fc2f868ec90849d
```

각 runtime은 fresh process 5회다. Current는 5회 모두 24,960 output token과
동일 token hash를 생성했다.

| Metric | Current | frozen vLLM | Current improvement |
|---|---:|---:|---:|
| request goodput | 41.798 req/s | 40.901 req/s | +2.19% |
| token throughput | 3622.47 tok/s | 3544.72 tok/s | +2.19% |
| TTFT mean | 35.44 ms | 49.90 ms | 28.98% shorter |
| TTFT p95 | 60.94 ms | 81.32 ms | 25.06% shorter |
| TPOT mean | 12.91 ms | 13.49 ms | 4.26% shorter |
| TPOT p95 | 15.78 ms | 17.67 ms | 10.73% shorter |
| E2E mean | 1132.65 ms | 1200.91 ms | 5.68% shorter |
| E2E p95 | 1859.92 ms | 2122.52 ms | 12.37% shorter |
| peak memory | 9237 MiB | 9039 MiB | Current +198 MiB |

따라서 이전 단일 run의 `41.02 vs 40.91 req/s`는 우연한 outlier가 아니었다.
최종 정책의 5-run median은 오히려 `41.80 req/s`다.

## 7. Load sweep와 SLO attribution

Joint SLO:

```text
arrival TTFT <= 500 ms
TPOT         <= 50 ms
arrival E2E  <= 2500 ms
```

| Offered req/s | Raw/goodput req/s | Token/s | Pass | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 39.0 | 34.555 / 34.555 | 2994.73 | 100.00% | 32.37 / 56.81 | 11.06 / 13.61 | 974.63 / 1578.41 |
| 48.8 | 41.798 / 41.798 | 3622.47 | 100.00% | 35.44 / 60.94 | 12.91 / 15.78 | 1132.65 / 1859.92 |
| 97.5 | 47.723 / 19.456 | 4136.03 | 40.97% | 37.19 / 63.09 | 16.36 / 21.53 | 1389.52 / 2149.38 |

97.5의 failure count median:

| Category | Requests / 288 |
|---|---:|
| pass | 118 |
| TTFT only | 61 |
| E2E only | 2 |
| TTFT + E2E | 107 |
| any TPOT failure | 0 |

서버가 관측한 latency는 낮지만 client dispatch delay p95가 약 1.8--2.0초까지
증가한다. 즉 97.5 cliff는 decode kernel/TPOT failure가 아니라 offered load가
server capacity를 넘으면서 admission 이전 arrival backlog가 쌓이는 현상이다.

`benchmarks/phase_serving/analyze_slo_goodput.py`는 이제 joint failure reason을
전체 및 request class별로 저장한다.

## 8. E/P/D/Copy activity

### 8.1 이전 mixed `33.74%` 결과의 교정

이 문서의 이전 revision은 mixed trace의 idle을 `33.74%`로 보고했다. 이는
runtime 동작이 아니라 artifact 결합 오류였다. `sync`와 `worker-4` variant가 같은
`activity-{run}` prefix를 공유했고, 저장 단계에서 sync runtime log와 worker-4
activity CSV가 결합됐다. 그 결과 runtime phase summary의 dispatch `123`개 중
activity interval은 `68`개만 들어갔다.

최신 진단은 workload와 load point마다 고유 prefix를 사용한다. 또한 다음 coverage
invariant를 분석 전에 검사했다.

```text
runtime phase-summary dispatches == activity planned dispatches
planned pair actions == measured pair-action records
```

12개 workload와 3개 load point, 총 15개 실행 모두 첫 invariant가 exact match였다.
P+D도 모든 실행에서 planned/actual action count가 같았고 missed/unplanned action은
없었다. 따라서 이전 `33.74%` 수치는 폐기하고 아래 결과로 대체한다.

| Trace | Runtime/activity dispatches | Planned/actual P+D |
|---|---:|---:|
| short | 39 / 39 | 5 / 5 |
| balanced | 517 / 517 | 92 / 92 |
| decode-heavy | 1437 / 1437 | 133 / 133 |
| long-prefill | 1348 / 1348 | 664 / 664 |
| bimodal | 1822 / 1822 | 511 / 511 |
| text-heavy | 96 / 96 | 11 / 11 |
| mixed | 119 / 119 | 5 / 5 |
| vision-heavy | 146 / 146 | 3 / 3 |
| poisson | 183 / 183 | 13 / 13 |
| wave/drain | 146 / 146 | 6 / 6 |
| multi-image | 36 / 36 | 2 / 2 |
| late-vision | 203 / 203 | 1 / 1 |
| load 39.0 | 841 / 841 | 227 / 227 |
| load 48.8 | 635 / 635 | 214 / 214 |
| load 97.5 | 514 / 514 | 211 / 211 |

### 8.2 최신 Current 12-workload activity

각 workload를 최신 Current binary의 fresh process에서 한 번씩 실행했다. 이 표는
성능 promotion용 3-run median이 아니라 GPU activity 구조를 확인하기 위한 1회
diagnostic이다. Warmup과 calibration은 lifecycle dispatch correlation으로
제외했다.

Phase column은 각 phase의 inclusive activity이므로 overlap 구간이 중복 집계되어
합이 100%를 넘을 수 있다. `Any overlap`은 둘 이상의 E/P/D phase가 동시에 active인
mutually exclusive mask의 합이다.

| Workload | Active span ms | E | P | D | Copy | Idle | Any overlap |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 428.3 | 0.00% | 48.09% | 70.97% | 0.00% | 1.47% | 20.53% |
| balanced | 5466.9 | 0.00% | 38.03% | 76.80% | 0.00% | 2.51% | 17.34% |
| decode-heavy | 14328.2 | 0.00% | 17.46% | 90.11% | 0.00% | 2.48% | 10.05% |
| long-prefill | 21306.1 | 0.00% | 88.16% | 50.03% | 0.00% | 1.31% | 39.50% |
| bimodal | 23428.1 | 0.00% | 53.63% | 71.18% | 0.00% | 1.33% | 26.13% |
| text-heavy | 1724.3 | 24.35% | 42.02% | 40.10% | 0.00% | 2.83% | 9.30% |
| mixed | 2580.9 | 32.16% | 41.00% | 29.04% | 0.00% | 3.71% | 5.90% |
| vision-heavy | 3499.8 | 35.70% | 60.32% | 24.28% | 0.00% | 3.92% | 24.22% |
| poisson | 2370.4 | 17.76% | 39.17% | 50.50% | 0.00% | 2.66% | 10.09% |
| wave/drain | 6509.9 | 8.03% | 8.18% | 13.97% | 0.00% | 70.46% | 0.64% |
| multi-image | 489.6 | 26.89% | 26.53% | 51.01% | 0.00% | 2.85% | 7.27% |
| late-vision | 1820.2 | 9.67% | 19.28% | 75.14% | 0.00% | 3.20% | 7.29% |

일반 burst workload의 idle은 `1.31--3.92%`다. wave/drain의 `70.46%`는 네 개의
arrival wave 사이를 의도적으로 비운 trace 계약이며 scheduler가 실행 가능한 일을
놓친 idle로 해석하면 안 된다. 실제 계산 overlap은 long-prefill `39.50%`, bimodal
`26.13%`, vision-heavy `24.22%`까지 형성된다.

### 8.3 Mutually exclusive phase mask

Bit 정의는 `E=0001`, `P=0010`, `D=0100`, `Copy=1000`이다. 아래 각 행의
unrounded 값은 정확히 100%로 합산되며 표시 값은 소수 셋째 자리에서 반올림했다.

| Workload | 0000 idle | 0001 E | 0010 P | 0011 E+P | 0100 D | 0101 E+D | 0110 P+D | 0111 E+P+D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 1.466% | 0.000% | 27.560% | 0.000% | 50.443% | 0.000% | 20.531% | 0.000% |
| balanced | 2.513% | 0.000% | 20.690% | 0.000% | 59.458% | 0.000% | 17.339% | 0.000% |
| decode-heavy | 2.479% | 0.000% | 7.406% | 0.000% | 80.064% | 0.000% | 10.050% | 0.000% |
| long-prefill | 1.312% | 0.000% | 48.660% | 0.000% | 10.531% | 0.000% | 39.497% | 0.000% |
| bimodal | 1.326% | 0.000% | 27.492% | 0.000% | 45.047% | 0.000% | 26.134% | 0.000% |
| text-heavy | 2.834% | 24.347% | 32.715% | 0.000% | 30.801% | 0.000% | 9.303% | 0.000% |
| mixed | 3.707% | 28.802% | 35.093% | 3.355% | 26.495% | 0.000% | 2.548% | 0.000% |
| vision-heavy | 3.924% | 12.085% | 36.102% | 23.612% | 23.672% | 0.000% | 0.605% | 0.000% |
| poisson | 2.661% | 12.568% | 29.081% | 5.191% | 45.604% | 0.000% | 4.895% | 0.000% |
| wave/drain | 70.457% | 8.026% | 7.545% | 0.000% | 13.333% | 0.000% | 0.638% | 0.000% |
| multi-image | 2.846% | 26.887% | 19.260% | 0.000% | 43.733% | 0.000% | 7.274% | 0.000% |
| late-vision | 3.195% | 2.860% | 14.190% | 4.611% | 72.466% | 2.203% | 0.474% | 0.000% |

Copy bit가 포함된 `1000--1111` mask는 모든 workload에서 각각 `0.000%`다. Vision
encoder output은 별도 D2D output copy 없이 retained device-resident slab lease로
P에 전달된다. `E+P+D`가 0인 것은 production action space를 bounded pair action으로
제한한 설계와도 일치한다.

### 8.4 Offered load에 따른 activity 변화

동일 text trace에서 offered load만 높인 one-repeat diagnostic 결과다.

| Offered req/s | Active span ms | P | D | Idle | P+D |
|---:|---:|---:|---:|---:|---:|
| 39.0 | 8338.2 | 44.98% | 76.08% | 1.97% | 23.02% |
| 48.8 | 6943.3 | 52.75% | 74.00% | 2.23% | 28.98% |
| 97.5 | 6111.9 | 54.90% | 74.92% | 2.45% | 32.27% |

세 load point는 text-only이므로 표에 없는 E, E+P, E+D, E+P+D와 모든 Copy mask는
`0.00%`다. P와 D column은 inclusive 값이며 `P+D`가 양쪽에 포함된다.

Idle ratio는 `1.97 -> 2.23 -> 2.45%`로 아주 조금 증가하지만 absolute idle은 약
`164 -> 155 -> 150 ms`로 감소한다. 총 active span이 더 빠르게 짧아져 생긴
denominator effect다. 반면 P+D는 `23.02 -> 28.98 -> 32.27%`로 증가한다. 즉
controller는 pressure가 커질수록 실제 overlap을 더 많이 사용한다.

### 8.5 해석과 계측 한계

이번 결과는 “idle이 적으므로 overlap이 필요 없다”는 결론을 지지하지 않는다.
대부분의 trace에서 GPU는 이미 serial work만으로도 busy하다. 이때 overlap의 역할은
idle hole을 메우는 것이 아니라 같은 E/P/D work의 wall-clock makespan을 줄이는
것이다. 따라서 판단 기준은 overlap 비율 자체가 아니라 다음 값이어야 한다.

```text
isolated equivalent work / overlapped makespan
    subject to TTFT/TPOT slack and formation preservation
```

또한 activity mask는 CUDA event로 감싼 phase의 “작업 중인 시간 구간”이다. 이는
SM active, tensor-core utilization, DRAM bandwidth 또는 L2 pressure와 동일하지
않다. E/P/D interval이 겹쳐도 커널이 자원을 완전히 직렬화할 수 있고, 반대로 작은
시간 overlap이 높은 makespan gain을 만들 수도 있다. 이 구분은 selected point의
Nsight Systems/Compute 분석으로 검증해야 한다.

Planned action label만으로 physical overlap을 추정하지 않는다. Online reward는
CUDA-event start/end에서 관측한 equal-work compression으로 갱신하고, activity
artifact는 runtime dispatch coverage가 exact match일 때만 유효한 것으로 취급한다.

## 9. Validation

### 9.1 C++

- final post-format phase/scheduler focused tests: `261/261` pass
- full unitTest: `1240` tests 실행, `1195` pass, `42` skip
- full suite 최초 실패 3개 중 `RopeWriteKvPrefill.AccuracyFp8`은 isolated rerun
  pass
- `InitializeYarnRopeCosSin.Accuracy`와 `InitializeMRopeCosSin.Accuracy`는 isolated
  rerun에서도 기존 `1e-3` tolerance를 약 `0.0011--0.0012`로 초과

남은 두 test는 이번 phase scheduler/sampling completion 변경과 독립인 RoPE
initializer 경로다. 이 commit에서 tolerance를 넓혀 숨기지 않는다.

### 9.2 Python

새/변경 분석기 전체 targeted suite의 `33/33` test가 pass했다. 최종 pre-commit
자동 포맷 뒤 재실행한 직접 변경 8개 파일도 `28/28` pass했다.

- SLO goodput/failure attribution
- E/P/D/Copy activity analysis
- contextual shadow
- six-direction injection
- incremental H1 replay
- oracle H1 policy/matrix/snapshot coverage

### 9.3 End-to-end

- model/export/build 산출물은 기존 고정 Cosmos bundle을 사용
- actual runtime inference, HTTP adapter, sampling, VLM semantics를 통과
- 12 workload x 3와 load 3 points x 5를 fresh process로 완료
- 48.8 result 5/5 exact token identity

## 10. Artifact locations

```text
.local/final-contextual-provenance-canonical-20260902/
.local/final-contextual-provenance-comparison-20260902/
.local/final-contextual-provenance-load-3x5-20260902/
.local/final-current-activity-12x1-20260902/
.local/final-current-activity-load-3x1-20260902/
```

이전 `.local/final-contextual-activity-20260902/`는 variant 간 prefix 충돌이 있어
activity headline 근거로 사용하지 않는다. 위 두 `final-current-activity-*` root가
교정된 결과의 source of truth다.

주요 파일:

```text
comparison.md
slo-goodput.json
analysis/activity-summary.json
analysis/measured-segments.csv
formation-summary.json
```

## 11. 남은 작업

우선순위는 다음과 같다.

1. Legacy가 이기는 mixed, poisson, text-heavy, vision-heavy에서 D continuity,
   producer critical path, context residency를 같은 binary의 mechanism-only
   ablation으로 분리한다.
2. Wave/drain의 FP16 vision exact branch를 canonical encoder batch row와 tactic
   binding으로 고정한다. Semantic gate와 exact production gate를 분리 보고한다.
3. 97.5 overload에서 admission 이전 client backlog를 서버 SLO와 분리하고,
   overload rejection/deadline-aware admission의 goodput curve를 추가한다.
4. Natural E+P/E+D evidence density를 높이되 unsafe random exploration을 하지
   않는다. Current ready state와 이미 outstanding인 event만 사용한다.
5. Equal-memory curve를 위해 independent vision context lazy residency/context
   reuse를 구현한다. KV pool은 줄이지 않는다.
6. 두 번째 GPU와 두 번째 model에서 동일 contextual controller가 다른 overlap
   payoff surface에 적응하는지 확인한다.
7. 논문용 selected points는 Nsight Systems/Compute로 SM active, DRAM, L2,
   concurrent kernel, host launch gap을 반복 측정한다.
8. Activity harness는 workload/variant별 고유 prefix와 exact dispatch coverage
   검사를 필수 invariant로 유지한다.

최종 방향은 유지한다.

```text
Independent E/P/D execution
        +
stable lifetime-aware ownership
        +
one profile-free online contextual selector
```

다음 최적화에서도 workload별 fine-tuning, 외부 cost registry, TTL을 추가하지
않는다.
