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
4. 최종 mixed trace의 CUDA-event active span에서 E/P/D overlap은 존재하지만
   작다. E+P는 `3.47%`, P+D는 `0.001%`이고 E/P/D idle은 `33.74%`다.
   independent context가 있다고 concurrency가 자동으로 이익이 되는 것은
   아니다.
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

최종 mixed HTTP trace 한 번을 phase metrics와 CUDA event를 켜고 실행했다.

| Mask | Meaning | Time | Active-span ratio |
|---|---|---:|---:|
| 0000 | E/P/D/Copy idle | 831.396 ms | 33.74% |
| 0001 | E only | 744.768 ms | 30.23% |
| 0010 | P only | 317.107 ms | 12.87% |
| 0100 | D only | 485.122 ms | 19.69% |
| 0011 | E+P | 85.491 ms | 3.47% |
| 0110 | P+D | 0.031 ms | 0.001% |

Phase inclusive utilization은 E `33.70%`, P `16.34%`, D `19.69%`다. Copy는
`0%`다. 이 trace는 encoder output을 별도 D2D copy하지 않고 device-resident
vision lease로 P에 전달했다.

Planned P+D action은 두 번 있었지만 same-dispatch physical kernel overlap은
0회였다. 두 context enqueue가 legal해도 GPU resource availability와 launch order에
따라 커널 구간이 겹치지 않을 수 있다.

```text
logical overlap action != measured kernel overlap
```

따라서 online reward는 action label이 아니라 CUDA-event start/end의 measured
compression으로 갱신해야 한다.

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
.local/final-contextual-activity-20260902/
```

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
