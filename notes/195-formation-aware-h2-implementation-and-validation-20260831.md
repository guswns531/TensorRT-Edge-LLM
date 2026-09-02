<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Formation-Aware Equal-Work H=2: Implementation and Validation

## 1. 결론과 현재 promotion 상태

이번 revision은 workload 이름, exact shape rule, 외부 cost registry, TTL, 임의의 미래 arrival 예측 없이 다음 구조를 구현했다.

```text
deterministic mechanism
        +
process-local continuous action value
        +
observable equal-work H=2 transition
        +
one SLO-safe global selector
```

최종 residual cleanup 소스의 장기 natural trace A/B 결과는 다음과 같다.

| Metric | H=2 | Myopic | H=2 improvement |
|---|---:|---:|---:|
| token throughput | 591.111 tok/s | 591.847 tok/s | **-0.12%** |
| TTFT mean | 1005.023 ms | 1051.444 ms | **+4.42%** |
| TTFT p95 | 3172.059 ms | 3211.950 ms | **+1.24%** |
| TPOT mean | 27.280 ms | 26.151 ms | **-4.32%** |
| TPOT p95 | 40.478 ms | 39.685 ms | **-2.00%** |
| E2E mean | 2258.865 ms | 2244.215 ms | **-0.65%** |
| E2E p95 | 3783.075 ms | 3821.179 ms | **+1.00%** |
| peak VRAM | 9473 MiB | 9481 MiB | **-8 MiB** |

두 경로 모두 3/3 exact-token identity와 action-fidelity violation 0을 만족했다. H=2는 TTFT와 일부 tail을 개선하지만 TPOT와 mean E2E를 악화시키며 throughput은 parity다.

이전 `p78/p79`의 `+1.61%` 결과는 residual post-selector authority가 완전히 제거되기 전 binary였다. 최종 H=2의 성능 증거로 사용하지 않고 역사적 audit 자료로만 남긴다.

12-workload 3회 gate도 완료했지만 H=2는 strict promotion gate를 통과하지 못했다. 따라서 현재 상태는 다음과 같다.

```text
mechanism / correctness / telemetry     complete
P5 selected-point profiling             complete
production promotion                    rejected for this revision
production default                      myopic
H=2 research mode                       explicit opt-in
```

## 2. 설계 원칙

### 2.1 workload fine-tuning 금지

Production selector는 다음 이름을 보지 않는다.

- short, balanced, decode-heavy, vision-heavy 같은 benchmark label
- latency/throughput/VLM serving profile
- E1/P8, E8/P8 같은 shape별 수동 정책 key

입력은 현재 observable state뿐이다.

```text
E/P/D ready rows
canonical request and stable-slot order
outstanding execution set and concrete completion events
TTFT/TPOT protected slack
exact CUDA observations
contextual mean/uncertainty
ownership allocate/reclaim transition
```

### 2.2 mechanism과 policy 분리

```text
Mechanism
  dependency / TensorRT shape / stable ownership / single-inflight
  candidate row construction / action lease / CUDA event completion

Policy
  already-legal candidate의 decision value
  SLO-safe candidate 중 equal-work horizon comparison
```

Contextual model은 illegal action을 legal하게 만들 수 없고 exact completion cost를 덮어쓸 수 없다.

### 2.3 exact execution cost와 decision cost 분리

각 candidate는 두 비용을 갖는다.

```text
predictedMakespanUs
  exact CUDA observation 또는 conservative mechanism cost
  feasibility, protected completion, telemetry에 사용

decisionMakespanUs
  contextual LCB를 serial-equivalent work에 투영한 연속 값
  이미 safe한 후보의 ranking/H=2에만 사용
```

이 분리는 continuous model이 unseen shape를 interpolation하면서도 SLO protection의 exact boundary를 오염시키지 않게 한다.

## 3. 최종 action flow

```text
Ready/Outstanding Snapshot
          |
          v
Mechanism Candidate Builder
 E / P / D / WAIT / E+P / E+D / P+D
          |
          v
Hard Feasibility
 DAG / TRT profile / ownership / memory / single-inflight
          |
          v
Contextual Decision Projection
 P+D head / E+P head / E+D head
 mean, uncertainty, LCB -> decisionMakespanUs
          |
          v
Observable Equal-Work H=2
 current action + at most one successor
 same target rows for every candidate
          |
          v
One Global Selector
 bounded exploration -> deadline safety -> efficiency
          |
          v
Explicit Action Lease
 planned outstanding set == actual outstanding set
          |
          v
CUDA start/end events
          |
          +--> exact cost tracker
          +--> matching contextual head update
```

Production action set은 `E`, `P`, `D`, `WAIT`, `E+P`, `E+D`, `P+D`로 닫혀 있다. `E+P+D`는 candidate explosion과 RTX 3080 interference를 피하기 위해 지원하지 않는다.

## 4. Observable equal-work H=2

### 4.1 Snapshot

`PhaseFormationSnapshot`은 다음만 포함한다.

- epoch
- 현재 ready E/P/D row 수
- 이미 outstanding인 event ID, bounded completion horizon, 그 event가 unlock하는 row

아직 도착하지 않은 request는 절대 snapshot에 들어가지 않는다. 같은 snapshot은 stable fingerprint를 가지며 같은 replay 결과를 만든다.

### 4.2 Equal-work target

짧은 action이 단순히 적은 work를 처리했기 때문에 유리해지는 오류를 막기 위해 모든 first action이 같은 target work를 끝내야 한다.

```text
C_H2(a, W) = C(a) + min C(successor)

subject to:
  completed(a, successor) covers exactly W
  successor count <= 1
  no arbitrary future arrival
```

한 successor로 target을 덮지 못하는 sequence는 비교 대상에서 제외한다.

### 4.3 Execution-formation coupling

H=2가 다루는 핵심은 action 자체의 kernel compression뿐 아니라 action 뒤 남는 cohort다.

```text
current action
   -> ready queue consumption
   -> concrete completion visibility
   -> successor cohort shape
   -> next dispatch count/cost
```

초기 controlled fixed-work 결과는 이 필요성을 보여 주었다.

| Pair | Small producer | Large producer |
|---|---:|---:|
| E+P overlap vs serial | E1/P8 +11.42%, E2/P8 +13.85% | E4/P8 -1.03%, E8/P8 -8.00% |
| E+D overlap vs serial | E1/D32 +21.55%, E2/D32 +7.68% | E4/D32 +0.46%, E8/D32 -0.09% |

E8/P8에서는 serial이 encoder를 5회 dispatch하고 E8을 유지했지만 always-overlap은 8회 dispatch, max E6으로 파편화했다.

그러나 이 표는 현재 binary의 promotion 근거로 재사용하면 안 된다. 최종 single-selector 소스로 같은 trace를 다시 3회 실행한 결과는 다음과 같다.

| Pair | Serial | Always overlap | Overlap change |
|---|---:|---:|---:|
| E1/P8 | 131.280 req/s | 80.339 req/s | **-38.80%** |
| E8/P8 | 232.924 req/s | 223.994 req/s | **-3.83%** |
| E1/D32 | 3155.051 tok/s | 3360.644 tok/s | **+6.52%** |

E1/P8 부호가 바뀐 주원인은 overlap path가 느려진 것이 아니다. 초기와 최종의 always-overlap은 약 81 req/s로 거의 같지만, serial의 P duty가 약 513 ms에서 166 ms로 줄면서 serial throughput이 72.712에서 131.280 req/s로 상승했다. 즉 earlier characterization이 발견한 현상은 당시 runtime에는 유효했지만 이후 mechanism optimization이 action payoff surface 자체를 바꿨다.

이 결과는 오히려 workload label이나 고정 shape rule을 두면 안 된다는 주장을 강화한다. 동시에 현재 revision에서 E1/P8을 formation-aware policy benefit의 positive counterexample로 사용할 수 없음을 의미한다.

## 5. Continuous contextual heads

P+D, E+P, E+D는 같은 feature math를 사용하지만 posterior evidence는 공유하지 않는다.

주요 feature는 다음과 같다.

- isolated phase duration과 duration ratio
- normalized batch fill
- P chunk와 context bucket
- protected slack
- residual progress/anchor
- successor formation state
- execution variant

각 head는 process-local RLS uncertainty와 normalized equal-work reward를 사용한다.

```text
mean = theta^T x
uncertainty = sqrt(x^T P x)
LCB = mean - beta * uncertainty

decision cost
  = serial_reference * (1 - bounded LCB)
```

정책 state는 `unknown/known exact key`가 아니라 연속적인 confidence가 된다. 외부 파일 저장/불러오기, TTL expiry, workload-specific 초기값은 없다.

## 6. One selector authority

초기 구현에는 selector 이후 다음 override가 존재했다.

- contextual positive -> overlap promote
- contextual negative -> overlap veto
- safe probe -> overlap 강제
- residual augmentation -> 별도 강제 선택

최종 구조에서는 production 경로의 이 권한을 모두 제거했다.

```text
candidate generation:
  decisionCostKnown / decisionMakespanUs
  safeProbeEligible

global selector:
  bounded unknown probe frontier
  robust deadline-safe frontier
  minimum violation if every action is late
  equal-work service compression
```

Warmup calibration과 `experimental-*-overlap-percent`는 controlled characterization mode에만 남아 있으며 production policy가 아니다.

## 7. 코드 위치

| 영역 | 파일 | 책임 |
|---|---|---|
| Formation state/API | `cpp/runtime/phase/policy/phaseFormationPlanner.h` | immutable snapshot, known completion, target work, H=2 sequence |
| H=2 transition | `cpp/runtime/scheduling/phaseFormationPlanner.cpp` | deterministic transition, equal-work coverage, one successor |
| Candidate cost split | `cpp/runtime/phase/execution/phaseActionPlan.h` | exact execution cost와 contextual decision cost 분리 |
| Contextual projection | `cpp/runtime/phase/policy/phaseContextualPdModel.h`, `cpp/runtime/scheduling/phaseContextualPdModel.cpp` | 16-D feature, RLS, bounded decision makespan |
| P+D candidate | `cpp/runtime/scheduling/phaseQueueScheduler.cpp` | legacy-compatible P/D formation과 contextual decision input |
| E/P/D integration | `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | E+P/E+D, H=2 snapshot, residual augmentation, action lease |
| Single selector | `cpp/runtime/scheduling/phaseGlobalScheduler.cpp` | feasibility, bounded exploration, SLO, equal-work efficiency |
| Process-local heads | `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h`, `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` | P+D/E+P/E+D 독립 posterior와 CUDA feedback |
| Runtime telemetry | `examples/llm/llm_phase_context_smoke.cpp` | action, formation, contextual, fidelity, activity metric |
| Tests | `unittests/phaseFormationPlannerTest.cpp`, `unittests/phaseGlobalSchedulerTest.cpp`, `unittests/phaseQueueSchedulerTest.cpp` | replay/equal-work/SLO/exploration/canonical-order gate |

## 8. Correctness와 CPU validation

최종 residual cleanup 뒤 결과:

```text
Build targets:
  unitTest                PASS
  llm_phase_context_smoke PASS

Focused tests:
  183 tests / 4 suites    PASS

git diff --check          PASS
```

주요 unit invariant:

- canonical row/stable ownership order 불변
- concrete completion event만 successor state에 반영
- 같은 snapshot ID는 같은 replay identity
- unequal work candidate를 빠르다고 보상하지 않음
- contextual decision cost가 exact deadline을 우회하지 않음
- exact execution cost가 decision cost로 덮이지 않음
- unknown overlap은 explicit bounded probe일 때만 가능
- all-late recovery probe도 candidate guard를 통과해야 함
- action frontier가 설정된 upper bound를 넘지 않음

## 9. Long-lived natural trace contract

짧은 12-workload process restart에서는 E head가 readiness threshold를 넘지 못하므로, 같은 process에서 학습과 활용이 모두 일어나는 별도 trace를 만들었다.

```text
requests                 384
epochs                   6, permuted order
text / vision            192 / 192
prompt tokens            133,134
requested/generated      17,568 / 17,568
client max-in-flight      80
P/D/E caps               8 / 64 / 8
prefill chunk             128
trace SHA256              94d8d5299063b6ed2ffa808c1cc37edc152d1d35735399de4a69eb40afc1f207
```

Workload label은 trace generation/offline 분석에만 존재하고 scheduler input에는 없다.

## 10. Historical p78/p79 H=2 versus myopic 결과 — superseded

이 절의 수치는 residual post-selector authority가 완전히 제거되기 전 binary에서 얻은 역사적 결과다. 최종 promotion 수치로 사용하지 않는다. 최종 소스 A/B는 20절에 기록한다.

### 10.1 Aggregate

양쪽 모두 3/3에서 generated token 수 17,568과 token SHA256
`5c0aa22ee6aa52638fa1ea09c5d4b7fc2ec225a7870d8870aaf1693bd5fa6e21`가 정확히 일치했다.

Latency improvement는 양수일수록 H=2가 좋다는 의미다.

| Metric | H=2 | Myopic | Improvement |
|---|---:|---:|---:|
| token/s | 601.664 | 592.110 | +1.61% |
| req/s | 13.151 | 12.942 | +1.61% |
| TTFT mean | 970.892 | 999.617 | +2.87% |
| TTFT median | 570.866 | 562.245 | -1.53% |
| TTFT p95 | 2955.615 | 3070.945 | +3.76% |
| TPOT mean | 26.952 | 27.877 | +3.32% |
| TPOT median | 26.696 | 28.210 | +5.37% |
| TPOT p95 | 39.287 | 39.542 | +0.64% |
| E2E mean | 2200.697 | 2266.546 | +2.91% |
| E2E median | 1968.178 | 1963.392 | -0.24% |
| E2E p95 | 3704.080 | 3790.199 | +2.27% |

Median TTFT/E2E는 각각 1.53%/0.24% 나빠졌지만 mean과 p95, throughput은 개선됐다. 이 결과는 H=2를 무조건 promote하기보다 cross-workload/SLO gate가 필요한 이유이기도 하다.

### 10.2 Request class

| Class/metric | H=2 | Myopic | Improvement |
|---|---:|---:|---:|
| text TTFT mean | 239.777 | 248.147 | +3.37% |
| text TTFT p95 | 570.723 | 574.219 | +0.61% |
| text TPOT mean | 29.596 | 31.077 | +4.77% |
| text TPOT p95 | 40.303 | 40.279 | -0.06% |
| text E2E mean | 1939.405 | 1967.833 | +1.44% |
| text E2E p95 | 2599.237 | 2603.282 | +0.16% |
| vision TTFT mean | 1702.008 | 1751.088 | +2.80% |
| vision TTFT p95 | 3340.056 | 3359.242 | +0.57% |
| vision TPOT mean | 24.307 | 24.677 | +1.50% |
| vision TPOT p95 | 34.453 | 35.422 | +2.73% |
| vision E2E mean | 2455.537 | 2516.084 | +2.41% |
| vision E2E p95 | 3804.488 | 4099.011 | **+7.19%** |

가장 큰 이득은 vision E2E tail이고, text TPOT p95는 사실상 parity다.

## 11. Policy/action telemetry

3회 run의 median cumulative counter다.

| Counter | H=2 | Myopic |
|---|---:|---:|
| P+D overlap selections | 137 | 140 |
| WAIT selections | 15 | 20 |
| E+P selections | 8 | 5 |
| E+D selections | 0 | 0 |
| bounded safe probes | 51 | 64 |
| H=2 lookaheads | 288 | 0 |
| H=2 selection changes | 7 | 0 |
| formation post-policy overrides | **0** | 0 |
| action fidelity violations | **0** | **0** |

세 run에서 H=2가 myopic과 다르게 고른 23개 누적 decision의 first action은 다음과 같다.

```text
P             10
E+P            7
E              6
E+D/P+D        0 as the changed first action
```

즉 gain은 overlap을 무조건 늘린 결과가 아니다. WAIT/probe 수를 줄이고 일부 시점에는 E/P formation을 보존하거나 E+P를 선택한 결과다.

## 12. Stream activity와 CPU cost

Activity mask 정의:

```text
E = 0001
P = 0010
D = 0100
C = 1000
```

3회 median window ratio:

| Metric | H=2 | Myopic |
|---|---:|---:|
| all E/P/D idle | 43.197% | 43.144% |
| any E/P/D active | 56.803% | 56.856% |
| E+P active | 3.886% | 3.870% |
| E+D active | 0.378% | 0.152% |
| P+D active | 7.945% | 8.163% |
| encoder duty | 17.487% | 17.214% |
| prefill duty | 24.673% | 24.606% |
| decode duty | 27.717% | 27.163% |

Copy mask는 이 trace에서 0이다. 이는 copy가 없다는 일반 결론이 아니라 현재 activity source가 기록한 explicit copy interval이 없었다는 의미다.

H=2 planner 자체의 CPU 시간은 652 observable snapshot 기준:

```text
median 0.779 us
p95    1.238 us
p99    1.424 us
max    1.633 us
```

전체 scheduler decision metric은:

| Metric | H=2 | Myopic |
|---|---:|---:|
| median | 73.734 us | 69.818 us |
| p95 | 343.667 us | 302.724 us |
| p99 | 482.580 us | 413.898 us |

H=2 planner는 작지만 전체 hot path p95는 목표 50 us보다 크다. 다음 CPU 최적화 대상은 H=2 수학이 아니라 snapshot aggregate 재사용, candidate materialization, telemetry serialization이다.

## 13. 이전 H=2 revision과의 관계 — historical audit

이전 `p68`은 1회 run이며 contextual/safe-probe post-selector override가 남아 있었다.

| Metric | p68 old H=2, 1x | p78 unified H=2, 3x median |
|---|---:|---:|
| token/s | 595.837 | 601.664 |
| TTFT mean | 966.701 | 970.892 |
| TPOT mean | 28.803 | 26.952 |
| E2E mean | 2288.031 | 2200.697 |
| E2E p95 | 3813.922 | 3704.080 |
| post-policy overrides | 10 | 0 |

반복 수가 다르므로 직접 speedup claim으로 쓰지 않는다. 중요한 변화는 action authority가 하나로 줄었고 post-policy override가 0이라는 점이다.

## 14. vLLM 비교

### 14.1 기존 48.8 request/s 반복 anchor

동일 text-only trace의 이전 5회 equal-cap 결과는 유지한다.

| Metric | Current median | vLLM median | Current advantage |
|---|---:|---:|---:|
| request throughput | 40.472 req/s | 40.751 req/s | -0.68% |
| token throughput | 3507.62 tok/s | 3531.74 tok/s | -0.68% |
| TTFT mean | 94.57 ms | 50.49 ms | -87.31% |
| TTFT p95 | 282.57 ms | 81.17 ms | -248.12% |
| TPOT mean | 13.60 ms | 13.74 ms | +1.04% |
| TPOT p95 | 15.71 ms | 17.92 ms | +12.36% |
| E2E p95 | 2000.51 ms | 2146.16 ms | +6.79% |

Current는 TPOT/E2E tail이 좋지만 throughput과 TTFT는 vLLM보다 낮다. H=2 E scheduling은 이 text-only trace에 직접 참여하지 않는다.

### 14.2 새 384-request long-lived VLM trace

Fresh vLLM은 같은 request trace와 max-in-flight 80으로 두 번 시도했으나 모두 engine OOM으로 종료했다.

```text
attempt 1:
  free 123.44 MiB, requested 244.00 MiB
  process memory 9.51 GiB

attempt 2:
  free 179.44 MiB, requested 182.00 MiB
  process memory 9.46 GiB
```

실패 위치는 Qwen3-VL vision block의 MLP/GELU allocation이다. 따라서 이 trace에서 vLLM latency/throughput 숫자를 만들지 않았으며 결과는 `0/3 complete (CUDA OOM)`이다. Current H=2는 peak 9429 MiB로 3/3 완주했다.

이 결과는 Current가 vLLM보다 빠르다는 증거가 아니라, 동일 10GB 장기 VLM contract에서 Current가 완주하고 해당 vLLM 설정이 완주하지 못했다는 memory-sustainability 결과다.

## 15. GPU 복구 전 12-workload 상태 — superseded

마지막으로 완료된 12-workload 결과는 `p59`이며 E heads의 cold-start no-regression gate다. frozen vLLM 대비 Current token/s 방향은 다음과 같았다.

| Workload | Current vs vLLM token/s |
|---|---:|
| short | +20.7% |
| balanced | -5.2% |
| decode-heavy | -0.4% |
| long-prefill | -2.7% |
| bimodal | -2.7% |
| text-heavy | -12.6% |
| mixed | +0.8% |
| vision-heavy | +7.9% |
| poisson | -20.1% |
| wave/drain | +0.8% |
| multi-image | +1.5% |
| late-vision | +4.8% |

따라서 Current가 모든 workload에서 vLLM보다 빠르다는 결론은 아니다. 특히 balanced/text-heavy/Poisson이 남은 주요 target이다.

최종 single-selector 소스로 12 trace x 3회를 시작했지만 첫 short workload 전에 host `nvidia-smi`가 exit 9로 실패했다. 현재 상태:

```text
/dev/nvidia*             absent
NVIDIA kernel modules    loaded
PCIe RTX 3080            visible
running GPU containers   none
nvidia-smi               cannot communicate with driver
```

그래서 당시의 중단된 `p81` 임시 output은 유효한 run이 아니며 결과 표에 포함하지 않는다. 이후 GPU 복구 뒤 새 디렉터리에 완주한 `p81-final-h2-12x3` artifact와는 별개다. 최종 promotion 판단에는 21절의 완주 artifact만 사용한다.

## 16. GPU 복구 전 P5 상태 — superseded

`nsys` 2026.3.1은 host에 존재하지만 GPU device node가 사라진 뒤 selected-point profile을 새로 capture할 수 없다.

GPU 복구 후 다음 다섯 점만 capture한다.

1. E1/P8 serial
2. E1/P8 overlap
3. E8/P8 serial
4. E8/P8 overlap
5. E1/D32 overlap

필수 metric:

- SM active와 tensor utilization
- DRAM/L2 traffic
- concurrent kernel duration
- host launch gap
- action boundary와 다음 cohort formation

목표 질문은 `왜 E1/P8은 이득이고 E8/P8은 손해인가` 하나다.

## 17. GPU 복구 뒤 수행한 재개 순서

GPU driver/device node가 정상화된 뒤 다음 순서를 모두 수행했다.

1. `nvidia-smi`와 100MiB 이하 idle 상태 확인
2. final residual single-selector source로 long-lived H=2/myopic 각 3회 재확인
3. 최종 12-workload 3회 gate 재실행
4. unchanged trace의 frozen vLLM과 비교; contract가 바뀐 trace만 fresh vLLM 실행
5. selected five-point Nsight P5 capture
6. result promotion:
   - exact/semantic correctness
   - action fidelity violation 0
   - post-policy override 0
   - workload별 throughput와 TTFT/TPOT/E2E mean/p95
   - peak memory와 OOM

## 18. 다음 최적화 우선순위

GPU gate가 통과한 뒤 순서는 다음과 같다.

1. **Host decision path 최적화**: aggregate snapshot 재사용과 telemetry serialization 분리로 p95를 낮춘다.
2. **Balanced/text-heavy/Poisson 원인 분해**: vLLM 대비 TTFT와 throughput gap을 admission, completion visibility, sampling, TRT enqueue로 나눈다.
3. **Natural formation regret**: H=2가 바꾼 decision의 two-action regret와 SLO goodput을 직접 기록한다.
4. **Memory-normalized curve**: vLLM OOM을 단일 점이 아니라 equal-memory sustainable goodput curve로 검증한다.
5. **Second GPU/model**: 같은 selector가 다른 overlap sign을 online으로 학습하는지 확인한다.

## 19. p78 시점의 잠정 판단 — superseded

이번 구현으로 formation awareness는 더 이상 workload-specific heuristic이 아니다. 현재 ready work와 concrete event만을 사용한 bounded state transition이며, continuous action value와 exact mechanism cost가 분리되어 있다.

장기 trace에서 실제 selection change와 1.61% throughput, 2--4% mean/tail latency 개선이 관측되어 H=2 policy benefit의 첫 자연 발생 증거는 확보했다. 동시에 GPU utilization 총량은 거의 같았으므로, gain은 단순히 overlap을 더 많이 켠 결과가 아니라 **어떤 action을 언제 실행해 다음 formation과 SLO를 보존했는가**에서 나온다.

이 문장은 당시 재개 조건이었다. GPU 복구 후 final-source 재측정, 12-workload gate, Nsight P5를 완료했으며 최종 판단은 아래 20절 이후로 대체한다.

## 20. Final-source long-lived H=2 versus myopic

### 20.1 Contract와 correctness

```text
H=2 artifact     .local/transition-aware-20260830/p82-final-source-h2-3x/
myopic artifact  .local/transition-aware-20260830/p83-final-source-myopic-3x/
requests         384 per run
repeats          3
generated tokens 17,568 per run
token hash       5c0aa22ee6aa52638fa1ea09c5d4b7fc2ec225a7870d8870aaf1693bd5fa6e21
exact identity   3/3 in both variants
action fidelity  0 violations
post override    0 in both variants
```

### 20.2 Aggregate

개선율은 양수일수록 H=2가 좋다. Latency는 lower-is-better 방향으로 환산했다.

| Metric | H=2 | Myopic | H=2 improvement |
|---|---:|---:|---:|
| token/s | 591.111 | 591.847 | -0.12% |
| req/s | 12.920 | 12.937 | -0.12% |
| TTFT mean | 1005.023 ms | 1051.444 ms | +4.42% |
| TTFT median | 581.054 ms | 589.130 ms | +1.37% |
| TTFT p95 | 3172.059 ms | 3211.950 ms | +1.24% |
| TPOT mean | 27.280 ms | 26.151 ms | -4.32% |
| TPOT median | 27.521 ms | 25.047 ms | -9.88% |
| TPOT p95 | 40.478 ms | 39.685 ms | -2.00% |
| E2E mean | 2258.865 ms | 2244.215 ms | -0.65% |
| E2E median | 1898.076 ms | 1947.455 ms | +2.54% |
| E2E p95 | 3783.075 ms | 3821.179 ms | +1.00% |
| peak VRAM | 9473 MiB | 9481 MiB | -8 MiB |

이 결과는 H=2가 실제 action을 바꾸고 TTFT를 개선할 수 있음을 보여 주지만, end-to-end promotion benefit은 아니다. 같은 선택이 resident decode continuity를 희생해 TPOT를 악화시켰다.

### 20.3 Policy telemetry

세 run의 cumulative counter 범위는 다음과 같다.

| Counter | H=2 | Myopic |
|---|---:|---:|
| observable H=2 lookaheads | 224--300 | 0 |
| H=2 selection changes | 7--10 | 0 |
| E+P selections | 2--10 | 11--12 |
| E+D selections | 0 | 0 |
| P/D-family selections | 945--981 | 906--1116 |
| WAIT selections | 14--19 | 10--17 |
| bounded safe probes | 42--54 | 35--43 |
| post-policy overrides | 0 | 0 |
| action-fidelity violations | 0 | 0 |

Activity-mask median도 GPU를 더 많이 채운 결과가 아님을 보여 준다.

| Window ratio | H=2 | Myopic |
|---|---:|---:|
| all E/P/D idle | 43.866% | 43.963% |
| any E/P/D active | 56.134% | 56.037% |
| E+P active | 4.861% | 5.682% |
| E+D active | 0.099% | 0.024% |
| P+D active | 7.923% | 8.144% |
| E duty | 17.280% | 17.491% |
| P duty | 25.142% | 26.009% |
| D duty | 27.177% | 26.806% |

즉 H=2는 더 많은 overlap을 강제하지 않았다. P duty를 줄이고 E/D 배치를 재배치했지만 TPOT trade-off를 해소하지 못했다.

## 21. Final-source 12-workload H=2/myopic gate

```text
H=2 artifact     .local/transition-aware-20260830/p81-final-h2-12x3/
myopic artifact  .local/transition-aware-20260830/p84-final-myopic-12x3/
common config    P8 / D64 / E8 / chunk128 / stable slots80
repeats          3 per workload and policy
```

표의 latency 값은 `mean/p95 ms`다.

| Workload | H2 tok/s | Myopic tok/s | H2 vs M | H2 TTFT | M TTFT | H2 TPOT | M TPOT | H2 E2E | M E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 2393.616 | 2389.470 | +0.17% | 85.9/168.6 | 89.6/173.1 | 14.12/27.32 | 14.46/28.11 | 344.8/428.5 | 346.5/429.1 |
| balanced | 3961.590 | 4014.791 | -1.33% | 331.1/585.7 | 324.8/559.9 | 14.34/16.56 | 14.10/16.29 | 1554.3/2471.8 | 1528.0/2432.9 |
| decode-heavy | 4895.538 | 4860.095 | +0.73% | 694.7/1222.4 | 703.0/1232.2 | 11.49/12.35 | 11.58/12.53 | 3664.9/5797.5 | 3695.0/5764.6 |
| long-prefill | 1127.594 | 1120.680 | +0.62% | 3165.5/4143.8 | 3200.7/4138.5 | 28.26/33.69 | 28.41/33.13 | 5575.9/7953.3 | 5627.1/7928.2 |
| bimodal | 1727.047 | 1746.243 | -1.10% | 3022.0/6495.6 | 3054.6/6296.4 | 20.03/30.14 | 19.80/30.11 | 5791.7/11793.8 | 5775.1/11711.6 |
| text-heavy | 1438.893 | 1515.657 | **-5.06%** | 162.3/538.9 | 165.6/539.7 | 17.65/23.54 | 16.48/22.73 | 1091.3/1485.8 | 1031.0/1391.8 |
| mixed | 911.303 | 919.909 | -0.94% | 365.3/1237.7 | 371.3/1269.7 | 22.69/32.20 | 22.41/32.34 | 1380.6/1881.6 | 1387.4/1929.2 |
| vision-heavy | 653.707 | 613.771 | **+6.51%** | 510.0/1009.2 | 518.9/1143.9 | 28.66/37.34 | 29.73/38.76 | 1623.6/2404.4 | 1690.3/2495.3 |
| poisson | 1422.607 | 1441.786 | -1.33% | 110.9/505.7 | 111.8/449.2 | 16.71/24.13 | 16.25/22.22 | 1263.1/2310.4 | 1242.3/2277.2 |
| wave/drain | 97.766 | 97.818 | -0.05% | 262.3/316.8 | 257.9/316.8 | 9.99/13.01 | 9.37/12.49 | 572.1/617.8 | 557.2/615.7 |
| multi-image | 313.974 | 290.513 | **+8.08%** | 229.0/290.4 | 254.7/315.1 | 8.90/12.28 | 8.53/10.06 | 505.0/509.1 | 540.2/548.1 |
| late-vision | 2458.045 | 2453.762 | +0.17% | 125.1/442.4 | 122.6/436.6 | 9.60/9.67 | 9.62/9.69 | 1500.5/1875.8 | 1500.3/1879.0 |

### 21.1 Promotion 판정

H=2는 다음 positive signals를 보인다.

- decode-heavy throughput/TTFT/TPOT 소폭 개선
- vision-heavy throughput +6.51%, TTFT p95와 E2E 개선
- multi-image throughput +8.08%, TTFT/E2E 개선
- short와 late-vision은 거의 parity

하지만 strict gate를 통과하지 못한다.

- text-heavy throughput -5.06%, TPOT mean -7.11%, E2E p95 -6.76%
- multi-image TPOT p95 -22.01%
- vision-heavy TTFT p95가 좋아져도 반복 greedy hash는 안정적이지 않음
- Poisson TTFT p95 -12.56%, TPOT p95 -8.59%
- balanced/bimodal도 일관된 joint improvement가 아님

따라서 workload마다 H=2를 켜고 끄는 rule을 추가하지 않는다. 그것은 profile-free 목표를 훼손한다. 이번 revision의 production promotion은 실패로 판정한다.

### 21.2 Numerical gate

- exact deterministic: short, balanced, decode-heavy, long-prefill, bimodal, text-heavy, Poisson, multi-image, late-vision
- repeat-semantic pass지만 token hash variation: mixed, vision-heavy, wave/drain
- wave/drain은 H=2와 myopic 모두 같은 알려진 FP16 greedy branch instability를 보임

정책 corruption이나 ownership 오류의 증거는 없다. 그러나 production exact promotion에는 여전히 별도 numerical gate가 필요하다.

## 22. vLLM 비교

### 22.1 Frozen 12-workload anchor

12개 materialized trace와 vLLM implementation은 바뀌지 않았으므로 vLLM을 중복 실행하지 않았다. 아래 token/s 차이는 p59 보고서의 frozen vLLM anchor를 재사용한 방향성 비교이며, p59의 한 자리 반올림 비율로부터 환산했기 때문에 약 ±0.1% 오차가 있다.

| Workload | Final H2 vs frozen vLLM token/s |
|---|---:|
| short | +22.1% |
| balanced | -8.3% |
| decode-heavy | +0.6% |
| long-prefill | +0.6% |
| bimodal | -7.5% |
| text-heavy | -12.0% |
| mixed | -1.1% |
| vision-heavy | +11.4% |
| poisson | -20.9% |
| wave/drain | +2.0% |
| multi-image | +41.2% |
| late-vision | +4.2% |

H=2는 7/12 throughput 우세, 5/12 열세다. 특히 balanced, bimodal, text-heavy, Poisson이 남은 주요 gap이다. `p81`과 `p59`는 E capacity가 다르므로 이 표는 H=2 policy-only A/B가 아니라 현재 full configuration 대 frozen vLLM anchor다.

### 22.2 48.8 req/s equal-cap anchor

text-only 5회 결과는 그대로 유지한다.

| Metric | Current myopic | vLLM | Current advantage |
|---|---:|---:|---:|
| request throughput | 40.472 req/s | 40.751 req/s | -0.68% |
| token throughput | 3507.62 tok/s | 3531.74 tok/s | -0.68% |
| TTFT mean | 94.57 ms | 50.49 ms | -87.31% |
| TTFT p95 | 282.57 ms | 81.17 ms | -248.12% |
| TPOT mean | 13.60 ms | 13.74 ms | +1.04% |
| TPOT p95 | 15.71 ms | 17.92 ms | +12.36% |
| E2E p95 | 2000.51 ms | 2146.16 ms | +6.79% |

H=2 E scheduling은 이 text-only trace에 직접 참여하지 않는다.

### 22.3 Long-lived VLM memory sustainability

fresh vLLM은 같은 384-request trace와 max-in-flight 80에서 두 번 모두 vision MLP/GELU allocation OOM으로 실패했다.

```text
attempt 1  free 123.44 MiB, request 244.00 MiB, process 9.51 GiB
attempt 2  free 179.44 MiB, request 182.00 MiB, process 9.46 GiB
vLLM       0/3 complete
Current    3/3 complete, peak 9473--9481 MiB
```

이는 속도 우위가 아니라 10GB 동일 contract에서의 완주 가능성 차이다.

## 23. Final controlled overlap retest

Artifact:

```text
.local/transition-aware-20260830/p87-final-source-controlled/
```

| Pair | Serial | Always overlap | Change |
|---|---:|---:|---:|
| E1/P8 | 131.280 req/s | 80.339 req/s | -38.80% |
| E8/P8 | 232.924 req/s | 223.994 req/s | -3.83% |
| E1/D32 | 3155.051 tok/s | 3360.644 tok/s | +6.52% |

E1/P8의 역사적 `+11.42%`가 최종 소스에서 `-38.80%`로 바뀐 이유는 serial path가 개선됐기 때문이다.

| E1/P8 metric | Initial serial | Final serial | Initial overlap | Final overlap |
|---|---:|---:|---:|---:|
| throughput | 72.712 | 131.280 | 81.016 | 80.339 |
| P duty | 약 512.8 ms | 약 165.7 ms | 약 624.5 ms | 약 646.9 ms |

Always-overlap cost는 거의 그대로인데 serial P duty가 약 68% 줄었다. static overlap opportunity는 runtime mechanism이 발전하면 사라질 수 있다. 이 때문에 과거 shape payoff table을 production policy로 고정하면 안 된다.

## 24. P5 Nsight selected-point characterization

### 24.1 Capture contract

```text
GPU                  RTX 3080 10GB, SM86
Nsight Systems       2026.3.1
CAP_SYS_ADMIN        profiling container only
GPU metric interval  profiler default, about 100 us
artifact             .local/transition-aware-20260830/p85-nsys-selected/
CUDA/NVTX-only check .local/transition-aware-20260830/p86-nsys-cuda-only/
```

5개 `.nsys-rep`와 matching SQLite를 생성했다.

1. E1/P8 serial
2. E1/P8 overlap
3. E8/P8 serial
4. E8/P8 overlap
5. E1/D32 overlap

### 24.2 GPU metric window

각 client duration에 대응하는 마지막 GPU kernel window에서 계산했다.

| Point | SM active | Tensor active | DRAM read | Kernel busy | Mean concurrency | Max concurrency |
|---|---:|---:|---:|---:|---:|---:|
| E1/P8 serial | 23.1% | 16.2% | 21.2% | 85.2% | 1.05 | 4 |
| E1/P8 overlap | 27.1% | 18.7% | 27.2% | 91.3% | 1.07 | 5 |
| E8/P8 serial | 28.0% | 18.3% | 25.1% | 78.3% | 1.01 | 4 |
| E8/P8 overlap | 29.2% | 19.5% | 25.6% | 79.4% | 1.05 | 5 |
| E1/D32 overlap | 63.9% | 21.5% | 59.1% | 87.0% | 1.10 | 5 |

모든 point의 measured memcpy는 약 10.4 MB로 거의 같아 결과 차이를 KV compaction/copy 증가로 설명할 수 없다.

### 24.3 Interpretation과 perturbation

E1/P8 overlap은 serial보다 SM/Tensor/DRAM utilization과 kernel concurrency가 높다. 그러나 P stream duty가 166 ms에서 647 ms로 늘어 effective useful service는 크게 나빠진다. 높은 instantaneous utilization은 좋은 serving action의 충분조건이 아니다.

E8/P8은 overlap이 추가한 SM/Tensor utilization이 각각 1.2/1.2 percentage point에 불과하고 concurrency도 1.01에서 1.05로만 증가한다. 작은 concurrency gain이 interference와 formation cost를 상쇄하지 못한다.

E1/D32는 decode가 긴 resident work를 제공해 E+D active window를 만들고 최종 unprofiled run에서도 +6.52%를 유지했다.

중요한 측정 한계도 있다. GPU counter 및 CUDA trace capture에서 E1/P8 serial은 약 120 req/s, overlap은 약 80 req/s였고, unprofiled final 3회도 131/80 req/s로 같은 현재 부호를 확인했다. 다만 profiler 수치는 절대 성능 비교에 사용하지 않고 architecture counter에만 사용한다.

## 25. Production default와 opt-in contract

12-workload gate 실패 후 H=2를 production default로 승격하지 않았다.

```text
default Global active:
  myopic transition-safe selector

research H=2:
  TRT_EDGELLM_ENABLE_GLOBAL_FORMATION_AWARE=1

explicit myopic ablation:
  TRT_EDGELLM_DISABLE_GLOBAL_FORMATION_AWARE=1
```

Benchmark harness도 `--enable-global-formation-aware`와 `--disable-global-formation-aware`를 mutually exclusive하게 노출한다.

이 판정은 workload-specific fine-tuning을 피한다. text-heavy에서 끄고 vision-heavy에서 켜는 rule은 추가하지 않았다. 하나의 production algorithm은 myopic으로 유지하고, H=2는 다음 일반화된 policy revision이 전 workload gate를 통과할 때만 승격한다.

## 26. 최종 build/test gate

```text
llm_phase_context_smoke build  PASS
unitTest build                 PASS
focused CPU scheduler tests   183/183 PASS
Python harness py_compile      PASS
git diff --check              PASS
```

검증한 핵심 invariant:

- exact execution cost와 continuous decision cost 분리
- canonical row order와 stable ownership 유지
- concrete completion event만 H=2 successor에 반영
- equal-work target과 최대 one successor
- hard feasibility와 SLO protection을 policy보다 먼저 적용
- bounded exploration도 단일 selector 안에서만 선택
- production post-selector override 0
- planned action과 actual outstanding set fidelity
- H=2 on/off 동일 token identity가 가능한 trace에서 exact 3/3

## 27. 최종 판단과 다음 계획

이번 작업으로 설계와 mechanism은 완성됐다.

```text
Independent E/P/D execution
        +
Stable cross-phase ownership
        +
Continuous P+D/E+P/E+D action values
        +
Observable equal-work H=2 transition
        +
One feasibility/SLO/action selector
```

하지만 H=2 policy 자체는 production promotion에 실패했다. 가장 중요한 연구 결과는 다음 네 가지다.

1. formation effect와 action-dependent overlap payoff는 실제로 존재한다.
2. payoff surface는 runtime mechanism 최적화로 크게 변한다. 과거 E1/P8 positive point가 최종 소스에서는 사라졌다.
3. H=2는 TTFT 또는 vision-heavy throughput을 개선할 수 있지만 resident decode TPOT를 동시에 보호하지 못한다.
4. instantaneous SM utilization 증가와 serving goodput 증가는 동일하지 않다.

다음 구현 우선순위는 workload rule 추가가 아니다.

1. **SLO-constrained transition value**: H=2 successor cost에 oldest resident-D service gap을 직접 포함하되 TPOT는 hard robust constraint로 유지한다.
2. **Counterfactual regret telemetry**: 선택한 action과 안전한 대안의 two-action equal-work regret을 실제 완료 시점에 기록한다.
3. **Host path 최적화**: snapshot aggregate 재사용, telemetry serialization 분리, decision p95 감소.
4. **Balanced/text-heavy/Poisson 분해**: admission, completion visibility, sampling, state commit, candidate formation, TRT enqueue를 timestamp로 분리한다.
5. **Memory-normalized vLLM curve**: long-lived VLM OOM을 여러 memory/admission point의 sustainable SLO-goodput curve로 확장한다.
6. **Second GPU/model**: 동일 continuous selector가 다른 payoff surface에서도 workload label 없이 적응하는지 검증한다.

Promotion 조건은 유지한다.

```text
all legal/correctness invariants pass
AND each stable workload remains within 3%
AND at least one natural workload improves joint SLO goodput
AND no workload-specific label or shape rule is introduced
```

현재 권장 운영안은 **myopic production default + complete H=2 research opt-in**이다.

후속 SLO-constrained sequence guard, counterfactual regret telemetry, 그리고 새 12-workload gate는
`notes/196-slo-constrained-h2-regret-and-12-workload-gate-20260831.md`에 기록했다.
