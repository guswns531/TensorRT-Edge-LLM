# 동적 admission: 원인, 유지할 구조, 아직 해결하지 못한 것

Date: 2026-09-14. Active source: repository root, `codex/v0101-phase-forward-port`.

## 결론

**메모리가 허용하는 양과 서비스가 효율적인 양은 다르다.** 실제 byte·GPU consumer lifetime으로
고정 encoded-request 수를 대체하는 방향은 유효하지만, 남은 메모리를 모두 활용하면 TPOT까지 자동으로
최적화된다는 가정은 틀렸다. 또한 D를 우선 실행하거나 vision P를 무조건 잘게 나누는 규칙도 해결책이 아니었다.

이번 추가 실험에서 두 모델 모두를 개선하는 새 정책은 입증하지 못했다. 실패한 D 우선권 코드는 제거했고,
기존 V3와 lifetime admission을 유지한다. 메모리 안전성·계측·회수 계약 정합성의 진전과 성능 향상을 구분한다.
기존 lifetime admission의 이득은 [302](302-lifetime-encoded-admission-two-model-results-20260913.md),
이번 모든 개별 실행/원인 표는 [303](303-service-admission-cause-and-experiment-20260914.md)에 있다.

## 1. 무엇이 실제 원인이었나

### 1.1 Cosmos: 큰 배치는 GPU 일을 줄였지만 요청의 진행 간격을 늘렸다

기존 3회 heavy 결과를 request ID별로 연결했다. 각 요청의 토큰별 시간을 평균한 뒤 요청 간 평균을 냈다.

| Decode cycle 구성 | Static16 | Lifetime | 증가 |
|---|---:|---:|---:|
| Ready → D start | 11.419 ms | 22.092 ms | +10.673 ms |
| D start → host done | 12.575 ms | 15.981 ms | +3.406 ms |
| Host done → token commit | 4.117 ms | 4.913 ms | +0.796 ms |
| 재구성한 cycle 합 | 28.111 ms | 42.986 ms | +14.875 ms |

증가분의 약 72%가 ready queue 구간이다. Host done은 GPU kernel 종료 자체가 아니며 completion visibility가
포함된다. Commit 구간 역시 sampler kernel만이 아니라 수집/상태 반영을 포함한다. 이 분해를 CUDA service와
혼동하지 않는다. 한 repeat의 D GPU 합은 1337.86→894.04ms로 줄었다. 전체 GPU 비용 절감과 개별 서비스 개선은
다른 지표라는 직접적인 반례다.

### 1.2 Gemma: 같은 정책이어도 service cost와 병목 비중이 다르다

Gemma는 D 배치가 커지면서 GPU/host 실행 비용 증가도 컸다. 따라서 Cosmos에서 관측한 ready-wait 문제를
그대로 다른 모델에 적용할 수 없다. 알고리즘에 모델 이름을 넣을 것이 아니라, 같은 stage 계측을 사용해 현재
서비스 비용을 읽어야 한다. 두 모델 간 trace 토큰 수, precision, profile, calibration 양이 같다는 뜻은 아니다.

### 1.3 낮은 idle은 D continuity를 보장하지 않는다

이번 반복의 E/P/D/Copy `0000` 비율은 대략 1.8~3.2%다. GPU work interval은 거의 차 있지만 특정 D 요청은
E/P 실행, 다른 cohort 및 completion 반영을 기다릴 수 있다. 이는 SM utilization이 아니며, aggregate idle만
줄이는 정책으로 TPOT 문제를 해결할 수 있다는 근거도 아니다.

## 2. 구현하고 검증한 네 가지 개입

| 개입 | 시험 | 결과 | 결정 |
|---|---|---|---|
| D service-age 초과 시 E formation yield | 두 모델 heavy, 6 cells | P가 대신 실행되고 D 파편화/대기 악화 | 소스 제거 |
| Global selector에서 overdue D의 earliest completion 보호 | 두 모델 heavy, 4 cells | 처리량 약 −2.3~−2.6%; Cosmos D76→89 | 소스 제거 |
| Final-P 회수 가능 view만 reclaim credit에 반영 | 두 모델 mixed/heavy, 24 cells, 3회 | 처리량 변화 약 ±1%, TPOT 회복 없음 | 정합성 API 유지, opt-in |
| Cosmos atomic vision P를 같은 엔진에서 chunk128로 실행 | 두 모델 mixed/heavy, 8 cells | Cosmos 처리량 −15%/−12%; D wait는 줄지만 총 비용 증가 | 기본 경로 유지, 진단 옵션만 유지 |

Gemma의 마지막 비교는 양쪽 모두 chunk128인 identity-mode control이다. 약 1~2% 단일 실행 차이를 정책 이득으로
해석하지 않는다. 최종 cleanup 후 검증은 별도 manifest로 기록하며 위 42 cells에 섞지 않는다.

## 3. 아키텍처의 원칙

```text
Engine capability / prepared tensor geometry
                     │
                     ▼
Physical ownership accounting ── KV pages / vision slab / reserved E output
                     │             (마지막 GPU consumer 전에는 회수 금지)
                     ▼
Memory-safe admission ───────── 충분한 byte가 있는가?
                     │
                     ▼
Ready E/P/D + outstanding GPU state
                     │
                     ▼
기존 Global V3 selector ─────── 실제 service 비용 / request 진행 / cohort 구성
                     │
                     ▼
Independent TRT contexts / explicit CUDA dependencies
                     │
                     └──────── 완료 관측 → request transition / ownership 반납
```

메모리 관리자는 실행 가능성을 판단하고, Global selector만 실행 정책을 결정한다. 별도 E/D 우선권 controller를
새로 두면 로컬 목표가 서로 충돌할 수 있다. 이번 실패한 두 guard는 그 우려를 구체적인 데이터로 보여줬다.

### 현재 동적인 것

- E 후보의 준비된 shape와 byte 추정, outstanding reservation을 이용한 admission prefix 선택.
- 실제 공유 owner를 중복 계산하지 않는 retained storage accounting.
- 마지막 consumer 완료 후에만 ownership 반납.
- 초기화/학습이 끝난 drained boundary에서 GPU 여유 메모리와 관측 preparation burst로 예산 산정.
- 기존 V3의 runtime cost/RLS 및 request-ready state에 따른 action 선택.

### 아직 자동 최적화하지 않는 것

- 자동 byte 예산은 **측정 중 매번 cudaMemGetInfo로 재확장하는 장치가 아니다**. 예약/소비량은 동적이지만
  기본 예산은 boundary에서 정한다. 이는 hot-path 동기화와 외부 allocator 충돌 위험을 피하기 위한 구분이다.
- RLS가 encoded capacity의 최적값을 학습하지 않는다. Lifetime admission은 새 학습 모델이 아니다.
- Memory feasibility와 service-efficient admission은 아직 하나의 검증된 통합 최적화가 아니다.
- `prefillReleaseByteSize()`는 final-P view 해제 가능량이지 현재 chunk 종료 후 물리 free 보장값이 아니다.
- Global memory horizon의 logical ledger와 admission의 physical owner ledger는 완전히 통일되지 않았다.
- 모델별로 사용하는 atomic/chunked vision-P capability가 다르며, 이 선택 자체를 자동 최적화하지 않는다.

동적 budget은 준비 시 관측한 burst에 근거한다. 관측되지 않은 더 큰 media geometry, 다른 프로세스의 GPU
할당, 일시적 workspace 증가까지 포함한 무조건적인 OOM 방지 증명은 아니다. 요청 admission/reservation과
실제 allocation 실패 처리는 계속 별개로 검증해야 한다.

### 코드 위치와 책임

| 파일 / 함수 | 책임 | 이번 추가 범위 |
|---|---|---|
| `phaseThreeCoordinator.cpp::setEncodedAdmissionMode` | Drained boundary의 예산 산정 및 mode 적용 | 실패한 service-priority 옵션 제거; ownership scoring만 opt-in |
| `phaseMemoryBroker.cpp` | byte reservation으로 E candidate prefix 제한 | 이전 lifetime 구현 유지 |
| `phaseVisionAdapter.cpp::phaseVisionRetainedStorageBytes` | shared slab / positional owner의 보존량 계산 | 이전 lifetime 구현 유지 |
| `phaseVisionAdapter.cpp::prefillReleaseByteSize` | final-P가 해제할 수 있는 logical view 조회 | unsplit M-RoPE 0, 나머지는 prefill view 크기 |
| `independentPhaseAsyncServer.cpp::visionPrefillReleaseBytes` | candidate request IDs의 release potential 집계 | global memory supplier와 연결 |
| `independentPhaseAsyncServer.cpp::setChunkedVisionPrefill` | 모든 요청/GPU work가 빈 boundary에서 기존 기능 토글 | 같은 엔진 diagnostic용; 기본 chunk 규칙은 유지 |
| `phaseGlobalScheduler.cpp` | feasibility/recovery/efficiency 단일 selector | 실험용 D 우선권을 제거하여 원래 V3 보존 |
| `llm_phase_context_smoke.cpp` | 공통 calibration 이후 측정 모드 적용 | `ownership` / chunk diagnostic, telemetry |
| `run_lifetime_encoded_admission.py` | 같은 계약의 두 모델 반복 HTTP 실행 | archived revision별 manifest·hash·폐쇄 로그 압축 |
| `analyze_decode_service_admission.py` | measurement epoch의 request stage 연결 | ready wait / host execution / commit 및 CUDA 합 구분 |

파일은 `cpp/runtime/scheduling/`, benchmark는 `benchmarks/phase_serving/` 아래다. 측정용 mode와 public serving
기본값을 혼동하지 않는다. 제거한 실험 코드는 `.local/results/service-admission-20260914/*/source.patch`에만
재현 증거로 보존하며 실행 코드에는 새로운 D 우선권이 없다.

## 4. SLO를 없애도 목적함수는 남는다

명시적인 “TTFT 500ms, TPOT 50ms”를 없애는 것은 가능하다. 그러나 어느 정도의 TTFT 증가를 허용하고
TPOT/처리량을 개선할지는 여전히 정책의 가치 판단이다. 그 선호까지 학습이 자동으로 정해 준다고 주장하면
숨겨진 휴리스틱이 된다. 이번 실험에서도 가장 빠른 전체 drain과 가장 짧은 평균 token gap은 일치하지 않았다.

따라서 모델별 수치를 조정하지 않되, 현재의 service-normalized recovery라는 **공통 선호**는 드러내야 한다.
사용자 latency 선호 없이 모든 평균·tail·처리량을 동시에 최대화한다는 목표는 보장할 수 없다. 비교는 동일
trace의 여러 지표와 Pareto trade-off로 판단한다. Benchmark의 사후 평가 기준은 scheduler 입력과 분리한다.

## 5. 다음 구현은 무엇을 만족해야 하나

1. **Physical ownership snapshot 통일.** Candidate의 현재 chunk가 final-P인지, shared owner의 모든 P consumer가
   끝나는지, M-RoPE가 D까지 남는지 구분한다. Predicted free를 실제 allocation credit으로 앞당겨 쓰지 않는다.
2. **같은 일을 비교하는 service horizon.** D1-now와 D4-later 또는 E/P/D action을 비교할 때 남은 작업을 생략하지
   않는다. 기존 bounded transition machinery를 재사용하고, 작은 batch를 빨리 실행한 immediate gain만 보지 않는다.
3. **Actual completion부터 ready까지의 비용.** GPU reference 외에 sampling/commit/formation 구간을 분리해
   관측한다. 현 host telemetry로 GPU visibility와 sampling kernel을 완전히 분리했다는 주장은 하지 않는다.
4. **한 가지 공통 수정만 paired A/B.** 두 모델에 같은 규칙을 적용하고 mixed/heavy 및 text/multi-image control을
   통과하면 full12로 확장한다. 정적 best를 선택해 배포하는 모델별 rule table을 만들지 않는다.
5. **승격 gate.** Greedy/ownership 검증, mean/p95 TTFT·TPOT·E2E, 처리량, peak memory를 함께 보고 어느 지표를
   양보했는지 명시한다. 요청 수·output token을 줄이거나 KV를 축소해서 성공으로 보이게 하지 않는다.

위 목록은 남은 계획이다. 통합 physical horizon이나 새로운 service-optimal controller를 이번에 완성했다고
보고하지 않는다. 새 정책을 더 붙이기보다 잘못된 단순화를 제거하고 비교 가능한 관측점을 확보한 것이 이번 진전이다.

## 6. Frozen vLLM과 현 수준

아래 Current는 `ownership-3x`의 **Lifetime control** 3회 평균이다. vLLM은 note302의 최적화된 retained reference를
재사용한다. Gemma vLLM은 1회, Cosmos mixed 3회/heavy 2회 성공 결과이며 fresh/equal-memory 비교가 아니다.
모든 latency는 HTTP send 기준이며 client admission 대기는 제외한다. 전체 12개 재측정 결과가 아니다.

| Model / workload | Runtime | token/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---|---:|---:|---:|---:|
| Gemma mixed | Current | 718.24 | 266.97 / 607.39 | 25.82 / 34.59 | 1456.48 / 2192.68 |
| Gemma mixed | Frozen vLLM | 703.81 | 276.41 / 410.58 | 26.48 / 32.39 | 1473.77 / 2162.56 |
| Gemma heavy | Current | 551.18 | 366.05 / 665.01 | 31.52 / 44.18 | 1550.59 / 2174.09 |
| Gemma heavy | Frozen vLLM | 559.83 | 280.57 / 390.78 | 30.59 / 40.54 | 1420.68 / 2138.33 |
| Cosmos mixed | Current | 1155.00 | 812.39 / 1998.54 | 34.34 / 62.35 | 2383.58 / 2515.56 |
| Cosmos mixed | Frozen vLLM | 923.32 | 868.73 / 2542.45 | 47.29 / 83.92 | 2998.29 / 3132.22 |
| Cosmos heavy | Current | 724.96 | 1517.44 / 3003.17 | 42.70 / 82.03 | 3181.70 / 3356.15 |
| Cosmos heavy | Frozen vLLM | 577.19 | 1630.87 / 3544.35 | 65.15 / 120.75 | 4087.81 / 4240.20 |

Cosmos는 이 두 workload에서 모든 표 지표가 우세하다. Gemma mixed는 평균에서 근소 우세지만 tail이 약하고,
heavy는 모든 지표에서 부족하다. 이 차이를 이번 새 기능의 speedup으로 귀속하지 않는다. 특히 Gemma의
repeat/cross-policy token identity 미통과는 성능과 별도의 production blocker다.
