# Transition-aware Phase Action: P0--P5 구현 및 검증

## 결론

P0부터 P5까지 구현하고 RTX 3080에서 실제 HTTP request trace로 검증했다. 가장 중요한 결론은 세 가지다.

1. 동일한 phase pair라도 concrete batch shape에 따라 overlap의 손익이 반전된다. 같은 32 E request와 24 P request를 처리할 때 E1/P8의 always-overlap은 serial보다 처리량이 11.42% 높지만 E8/P8은 8.00% 낮다.
2. E8/P8의 손실은 단순 contention만이 아니다. serial은 encoder를 5회 실행하며 E8을 만들었지만 always-overlap은 8회 실행하고 최대 E6만 형성했다. 현재 action이 다음 batch formation을 바꾸는 execution--formation coupling을 같은 총 작업량에서 확인했다.
3. real-request load sweep에서 raw throughput은 부하와 함께 계속 증가하지만 joint SLO goodput은 39.0 req/s 이후 급락했다. Current의 SLO-safe capacity knee는 이 단일 trace 기준 39.0--48.8 offered req/s 사이이며, 48.8 req/s에서 vLLM과의 주된 차이는 raw 처리량보다 TPOT/E2E와 queueing이다.

이번 단계에서 formation-aware selector의 bounded successor horizon과 telemetry는 구현했지만, E8/P8 production trace에서 myopic choice를 실제로 바꾼 횟수는 0이었다. 따라서 controlled counterexample은 강하지만 새 selector가 그것을 자동으로 해결했다고 아직 주장하지 않는다. 이 부정 결과를 숨기지 않고 다음 selector 개선의 promotion gate로 사용한다.

## P0 — Controlled result freeze

이전 overlap matrix 일부는 비교하는 요청 수가 달라 causal evidence로 쓰기 어려웠다. 이번에는 모든 E batch 설정에서 다음 총 작업을 고정했다.

- EP: E request 32개 + P request 24개, 각 output 1 token
- ED: E request 32개 + resident decode request 32개, decode output 192 token
- E batch: 1/2/4/8
- action: serial 0% 또는 always-overlap 100%
- 각 점 3회 중앙값
- 동일 trace의 output token hash 일치, action-fidelity violation 0

### E+P

| E/P | Serial req/s | Always overlap req/s | 변화 | E 실행 serial → overlap | observed max E serial → overlap |
|---|---:|---:|---:|---:|---:|
| E1/P8 | 72.71 | 81.02 | +11.42% | 32 → 32 | 1 → 1 |
| E2/P8 | 133.33 | 151.80 | +13.85% | 16 → 17 | 2 → 2 |
| E4/P8 | 228.43 | 226.08 | -1.03% | 8 → 9 | 4 → 4 |
| E8/P8 | 237.48 | 218.48 | -8.00% | 5 → 8 | 8 → 6 |

E1/E2는 작은 encoder work를 prefill과 겹쳐 이익을 얻었다. E4부터 이익이 사라지고, E8에서는 action이 encoder queue를 너무 빨리 부분 drain해 큰 cohort를 깨뜨렸다. 이는 workload label이 아니라 ready rows, candidate shape, 이미 outstanding인 completion에서 관측할 수 있는 현상이다.

### E+D

| E/D | Serial token/s | Always overlap token/s | 변화 | TTFT mean 변화 | TPOT p95 변화 |
|---|---:|---:|---:|---:|---:|
| E1/D32 | 2827.39 | 3436.62 | +21.55% | -26.38% | -17.92% |
| E2/D32 | 3259.57 | 3509.98 | +7.68% | -12.99% | -7.29% |
| E4/D32 | 3563.84 | 3580.09 | +0.46% | +5.05% | -0.51% |
| E8/D32 | 3628.33 | 3625.13 | -0.09% | +10.39% | +0.07% |

작은 E는 D의 실행과 잘 겹치지만 E가 커질수록 GPU contention과 formation loss가 immediate compression을 상쇄한다. 따라서 `E+D`라는 phase label만으로 action profitability를 결정할 수 없다.

원본 고정 결과는 `benchmarks/phase_serving/results/transition-aware-controlled-overlap-20260830.csv`에 저장했다.

## P1 — Successor formation과 decode continuity telemetry

두 종류의 관측을 추가했다.

### Formation telemetry

`PhaseThreeCoordinatorMetrics`와 최종 `PHASE_METRIC`에 다음을 추가했다.

- lookahead 실행 횟수
- bounded horizon 안에 예측한 encoder row 수
- myopic 대비 최종 selection 변경 횟수
- formation 평가 뒤 P/D 선택 횟수
- formation 평가 뒤 overlap 선택 횟수
- 마지막 predicted rows와 horizon

이 값은 외부 cost registry나 workload profile에서 오지 않는다. 현재 ready E rows, online encoder inter-arrival EWMA, 이미 선택 가능한 P/D action의 측정 비용만 사용한다.

### Decode continuity telemetry

`analyze_phase_activity.py`에 stream-level D continuity를 추가했다.

- D dispatch count
- 인접 D kernel-group 사이 gap mean/p95/max
- 25 ms 초과 gap 수
- consecutive non-D action 최대 길이

기존 request attribution과 함께 사용하면 GPU dispatch 공백과 각 request가 실제로 받은 decode service continuity를 분리할 수 있다.

## P2 — Bounded formation-aware selector

Global selector의 feasibility와 SLO lexicographic ordering은 유지했다. efficiency 비교에만 bounded current-plus-successor horizon을 추가했다.

```text
현재 E가 under-filled이고 P 또는 D가 ready
        │
        ├─ 현재 P/D makespan 안에 이미 관측된 arrival process로 E row가 도착 가능한가?
        │
        └─ 가능하면 동일한 reference work를 세 action으로 비교

E first : E_now + P/D + E_newcomer
P/D first: P/D + E_combined
overlap : overlap(E_now, P/D) + E_newcomer
```

임의의 미래 request 도착은 예측하지 않는다. 관측 horizon은 현재 P/D action이 끝나는 시점으로 제한하며, EWMA는 다음 encoder 도착과 그 안에서 형성 가능한 row 수만 추정한다. configuration은 기본 false이고 Global active serving smoke에서만 opt-in된다. `TRT_EDGELLM_DISABLE_GLOBAL_FORMATION_AWARE`로 같은 binary에서 myopic A/B가 가능하다.

안전성 보완으로 E candidate가 비어 있을 때 successor 계산을 하지 않도록 명시적으로 검사한다.

## P3 — Static, myopic, formation-aware 비교

### Static envelope와 offline oracle

P0의 serial/always-overlap 두 action을 같은 trace에서 비교했다. 이 제한된 action set에서 best-static과 offline oracle은 동일하다.

| Trace | Serial | Always overlap | SLO-safe offline choice |
|---|---:|---:|---|
| E1/P8 | 72.71 req/s | 81.02 req/s | overlap |
| E8/P8 | 237.48 req/s | 218.48 req/s | serial |
| E1/D32 | 2827.39 token/s | 3436.62 token/s | overlap |
| E8/D32 | 3628.33 token/s | 3625.13 token/s | serial |

이는 shape-independent always-overlap이나 always-serial이 모두 oracle이 될 수 없음을 보인다.

### Production myopic 대 formation-aware

E8/P8에서 각각 5회 반복했다.

| Selector | median req/s | TTFT mean ms | TTFT p95 ms | E2E mean ms | E2E p95 ms |
|---|---:|---:|---:|---:|---:|
| Myopic | 238.20 | 119.02 | 224.90 | 119.24 | 225.25 |
| Formation-aware | 235.63 | 120.62 | 225.01 | 120.88 | 225.22 |

Formation-aware는 run당 2--3회 lookahead했지만 selection change는 5회 모두 0이었다. 두 결과의 차이는 큰 run-to-run bimodality 안에 있으며 selector 개선으로 해석할 수 없다. 현재 online selector는 controlled static collapse를 피하지만, 그 이유는 기존 feasibility/SLO/cost 선택도 serial 쪽을 택했기 때문이다. P2의 successor value가 실제 결정을 바꾸는 trace를 추가로 만들어야 한다.

## P4 — Legacy decode continuity 원인 검증

48.8 offered req/s boundary에서 동일 binary/engine/trace로 Current와 Legacy-compatible selector를 비교했다.

| Metric | Current | Legacy-compatible | 차이 |
|---|---:|---:|---:|
| raw request/s | 37.467 | 37.436 | -0.08% |
| raw token/s | 3247.15 | 3244.49 | -0.08% |
| SLO pass | 66.32% | 70.49% | +4.17 pp |
| request goodput/s | 24.848 | 26.387 | +6.20% |
| decode dispatch/request | 85.667 | 85.667 | 동일 |
| mean decode BS | 56.748 | 56.242 | Current +0.90% |
| cumulative decode gap mean/request | 653.14 ms | 653.36 ms | 사실상 동일 |
| cumulative decode gap p95/request | 1151.54 ms | 1163.51 ms | Current -1.03% |

이 boundary trace에서는 “Global이 D cohort를 깨서 vLLM보다 느리다”는 가설이 지지되지 않는다. Current와 Legacy의 D formation/continuity가 사실상 같고 raw throughput도 같다. goodput 차이는 500/2500 ms threshold 부근 요청의 단일-run 이동에 민감하므로 반복 전에는 causal improvement로 주장하지 않는다.

## P5 — Joint-SLO goodput load sweep

### 도구와 정의

`materialize_load_sweep.py`는 materialized trace에서 request 내용, prompt/output 길이, 순서를 보존하고 arrival offset만 multiplier로 나눈다.

`analyze_slo_goodput.py`는 다음 joint SLO를 모두 만족하는 완료 요청만 goodput으로 센다.

```text
TTFT <= 500 ms
TPOT <= 50 ms
E2E  <= 2500 ms
```

중요하게 TTFT와 E2E는 client가 실제 send한 시각이 아니라 trace의 `scheduled_arrival_us`부터 측정한다. admission 이전 client queueing을 제외하면 overload에서 raw throughput을 SLO goodput으로 오인하기 때문이다.

### Current load sweep

| Offered req/s | Raw req/s | Raw token/s | SLO pass | Goodput req/s | Goodput token/s |
|---:|---:|---:|---:|---:|---:|
| 19.5 | 18.382 | 1593.13 | 100.00% | 18.382 | 1593.13 |
| 29.3 | 26.692 | 2313.34 | 100.00% | 26.692 | 2313.34 |
| 39.0 | 34.337 | 2975.83 | 100.00% | 34.337 | 2975.83 |
| 48.8 | 37.467 | 3247.15 | 66.32% | 24.848 | 2123.14 |
| 97.5 | 40.929 | 3547.15 | 26.74% | 10.943 | 929.99 |
| 487.5 | 49.376 | 4279.29 | 24.31% | 12.001 | 1031.42 |

raw token/s만 보면 487.5 req/s가 최고지만 joint-SLO goodput은 39.0 req/s가 최고다. 이 결과 때문에 이후 selector 평가는 throughput 하나가 아니라 SLO goodput load curve를 사용해야 한다.

### Fresh vLLM boundary comparison

동일 materialized trace와 SLO 정의로 39.0과 48.8 req/s를 fresh 실행했다.

| Offered | Runtime | Raw req/s | Raw token/s | SLO pass | Goodput req/s | Goodput token/s |
|---:|---|---:|---:|---:|---:|---:|
| 39.0 | Current | 34.337 | 2975.83 | 100.00% | 34.337 | 2975.83 |
| 39.0 | vLLM | 34.360 | 2977.86 | 100.00% | 34.360 | 2977.86 |
| 48.8 | Current | 37.467 | 3247.15 | 66.32% | 24.848 | 2123.14 |
| 48.8 | vLLM | 40.948 | 3548.82 | 100.00% | 40.948 | 3548.82 |

39 req/s에서는 raw throughput parity이고 Current TTFT 평균은 더 짧았지만 TPOT/E2E는 더 길었다. 48.8 req/s에서는 vLLM이 100% SLO pass를 유지해 Current request goodput보다 64.8% 높았다. 이는 formation-aware E policy만으로 해결될 문제가 아니다. P4 결과와 함께 보면 이 trace의 다음 병목은 LLM iteration service cost, sampling/completion submission gap, 혹은 TensorRT decode kernel/launch overhead 쪽이다.

결과 CSV는 `benchmarks/phase_serving/results/slo-goodput-load-sweep-20260830.csv`에 저장했다. load sweep과 vLLM boundary는 현재 각 점 1회 exploratory run이므로 최종 논문 수치로 사용하기 전 선택점 5회 반복과 confidence interval이 필요하다.

## 구현 위치

| 영역 | 파일 | 역할 |
|---|---|---|
| Runtime | `cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}` | bounded successor horizon, configuration, counters |
| Runtime adapter | `examples/llm/llm_phase_context_smoke.cpp` | opt-in/env A/B와 PHASE_METRIC |
| C++ tests | `unittests/phaseGlobalSchedulerTest.cpp` | 동일 reference work에서 formation horizon이 myopic overlap을 거부하는지 검증 |
| Activity | `benchmarks/phase_serving/analyze_phase_activity.py` | D dispatch gap과 non-D streak |
| Load generation | `benchmarks/phase_serving/materialize_load_sweep.py` | arrival process만 scaling |
| SLO evaluation | `benchmarks/phase_serving/analyze_slo_goodput.py` | scheduled-arrival 기준 joint-SLO goodput |
| Python tests | `tests/python-unittests/test_phase_activity_analysis.py`, `test_phase_slo_goodput.py` | continuity/load/SLO 계산 검증 |

## Promotion 판단

P0--P5 구현과 diagnostic validation은 완료했다. 그러나 formation-aware selector의 production promotion은 보류한다.

- 통과: 동일 작업량에서 shape-dependent overlap 반전 재현
- 통과: action-fidelity violation 0, exact token hash 유지
- 통과: successor horizon과 decode continuity를 직접 관측 가능
- 통과: joint-SLO load sweep 및 동일 trace vLLM 비교 가능
- 미통과: formation-aware가 myopic selection을 실제 변경하며 oracle regret를 줄인 사례
- 미통과: saturation boundary에서 vLLM과 동등한 TPOT/E2E goodput

다음 단계는 workload별 rule을 추가하는 것이 아니다. 첫째, 이미 outstanding인 completion과 current ready rows만으로 selection change가 발생하는 최소 trace를 만들고 two-action regret를 검증한다. 둘째, 48.8 req/s에서 sampling completion부터 다음 D enqueue까지의 host/GPU gap과 decode kernel-group cost를 Nsight로 분해한다. 이 두 문제는 각각 policy와 runtime mechanism에 속하므로 분리해서 고쳐야 한다.
