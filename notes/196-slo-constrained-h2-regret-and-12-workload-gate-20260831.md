<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SLO-Constrained H=2, Counterfactual Regret, and 12-Workload Gate

## 1. 결론

`notes/195-formation-aware-h2-implementation-and-validation-20260831.md`의 첫 번째 후속 우선순위를 구현하고 검증했다.

```text
old H=2
  equal-work current + one successor
  protected completion per first action

this revision
  equal-work current + one successor
  all protected completion over the whole sequence
  oldest resident-D service budget over the whole sequence
  max protected violation minimization
  predicted counterfactual regret telemetry
```

장기 natural trace에서는 myopic 대비 throughput `+0.28%`, TTFT p95 `3.45%` 개선, E2E p95 `3.76%` 개선을 보였지만 TPOT median은 `6.52%` 악화됐다. 12-workload gate에서는 text-only 5종이 대부분 parity였으나 text-heavy VLM throughput `-5.76%`, multi-image TPOT p95 `+20.13%`, Poisson E2E p95 `+3.18%` 등의 회귀가 남았다.

따라서 최종 판정은 다음과 같다.

```text
mechanism and invariant tests       PASS
long-lived primary tail             mostly improved
12-workload strict promotion gate   FAIL
production default                  myopic
SLO-constrained H=2                 explicit opt-in
workload-specific exceptions        not added
```

## 2. 이번에 해결한 문제

이전 H=2는 같은 work horizon을 비교했지만, sequence 전체에서 resident decode가 언제 다시 service되는지를 직접 제한하지 않았다. 예를 들어 첫 action이 P이고 successor가 E이면 two-action makespan이 짧더라도 이미 decode 중인 request는 두 action 동안 기다릴 수 있었다.

이번 revision은 snapshot에 decode service budget을 추가한다.

```text
decodeServiceBudgetUs
  = current feasible frontier의 protected decode slack 중 최솟값

decodeServiceUs(sequence)
  = sequence에서 최초로 resident D를 service하는 robust completion

decodeServiceViolationUs
  = max(0, decodeServiceUs - decodeServiceBudgetUs)
```

각 sequence가 가진 모든 protected completion의 violation도 함께 계산한다.

```text
protectedViolationUs(sequence)
  = max_r [robustCompletion_r(sequence) - slack_r]^+

sequenceViolationUs
  = max(protectedViolationUs, decodeServiceViolationUs)
```

Oracle 및 H=2 선택 순서는 global selector의 all-late semantics와 맞췄다.

```text
1. minimum sequenceViolationUs
2. minimum equal-work H=2 makespan
3. minimum uncertainty
4. stable action order
```

이렇게 해야 `protected violation 우선, decode violation 차선`인 별도 lexicographic oracle과 실제 selector의 `max violation` 판단이 엇갈려 false regret을 만드는 문제가 없다.

## 3. Counterfactual regret 정의

H=2 planner가 평가한 safe action frontier 안에서 선택 action과 oracle action의 predicted equal-work 차이를 기록한다.

```text
regretUs
  = max(0, chosen.equalWorkMakespanUs - oracle.equalWorkMakespanUs)

valid regret sample
  = chosen과 oracle이 같은 minimum violation frontier에 있음
```

추가 telemetry:

- `globalFormationRegretSamples`
- `globalFormationPositiveRegrets`
- `globalFormationPredictedRegretUs`: process 누적 predicted regret
- `maxGlobalFormationPredictedRegretUs`
- `lastGlobalFormationOracleAction`
- `lastGlobalFormationDecodeViolationUs`

이 값은 아직 **실행 뒤 실측 counterfactual**이 아니다. 선택하지 않은 action을 동시에 실행할 수 없으므로 현재 값은 같은 process-local CUDA cost model과 deterministic transition을 이용한 predicted regret이다. 이후 selected probe/replay를 통해 realized regret과 분리해 검증해야 한다.

## 4. 코드 위치

| 파일 | 변경 책임 |
|---|---|
| `cpp/runtime/phase/policy/phaseFormationPlanner.h` | decode budget, sequence violation, regret API |
| `cpp/runtime/scheduling/phaseFormationPlanner.cpp` | sequence-wide protected completion, earliest D service, min-max violation oracle |
| `cpp/runtime/scheduling/phaseThreeCoordinator.h` | regret/oracle/decode-violation cumulative state |
| `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | snapshot decode budget, H=2 guard 적용, regret 기록 |
| `examples/llm/llm_phase_context_smoke.cpp` | JSON telemetry export |
| `unittests/phaseFormationPlannerTest.cpp` | D budget, all protected deadline, snapshot identity, regret tests |

기존 hard feasibility, stable slot ownership, single-inflight context, action lease는 바뀌지 않았다.

## 5. Unit/build validation

TensorRT 26.06 container에서 다음을 다시 빌드했다.

```text
unitTest build                 PASS
llm_phase_context_smoke build  PASS
```

Focused suites:

```text
PhaseFormationPlannerTest      9
PhaseGlobalSchedulerTest      32
PhaseQueueSchedulerTest      139
PhaseRuntimeCostTrackerTest    5
total                        185/185 PASS
```

새로 고정한 invariant:

- decode service budget이 snapshot fingerprint에 포함된다.
- 더 짧은 H=2 action이라도 resident-D service budget을 넘으면 선택되지 않는다.
- oracle은 첫 action뿐 아니라 successor까지 포함한 모든 protected completion을 보존한다.
- minimum-violation frontier가 다른 action 사이에는 positive regret을 만들지 않는다.
- action-fidelity violation과 post-selector override는 0을 유지한다.

## 6. Long-lived natural trace A/B

Artifacts:

```text
H=2     .local/transition-aware-20260830/p91-max-violation-h2-3x/
myopic  .local/transition-aware-20260830/p90-full-slo-myopic-3x/
```

공통 contract:

```text
requests per run       384
repeats                3
prompt tokens          133,134
generated tokens       17,568
P / D / E caps         8 / 64 / 8
prefill chunk          128
trace hash             94d8d5299063b6ed2ffa808c1cc37edc152d1d35735399de4a69eb40afc1f207
token hash             5c0aa22ee6aa52638fa1ea09c5d4b7fc2ec225a7870d8870aaf1693bd5fa6e21
exact identity         3/3 both variants
```

개선율은 양수일수록 H=2가 좋다.

| Metric | H=2 | Myopic | H=2 improvement |
|---|---:|---:|---:|
| token/s | 590.822 | 589.190 | +0.28% |
| req/s | 12.914 | 12.879 | +0.28% |
| TTFT mean | 1061.401 ms | 1092.296 ms | +2.83% |
| TTFT median | 592.823 ms | 594.539 ms | +0.29% |
| TTFT p95 | 3190.661 ms | 3304.558 ms | +3.45% |
| TPOT mean | 26.183 ms | 25.597 ms | -2.29% |
| TPOT median | 25.564 ms | 23.999 ms | **-6.52%** |
| TPOT p95 | 40.075 ms | 39.708 ms | -0.92% |
| E2E mean | 2220.930 ms | 2242.053 ms | +0.94% |
| E2E median | 1808.133 ms | 1813.194 ms | +0.28% |
| E2E p95 | 3794.215 ms | 3942.626 ms | +3.76% |
| peak VRAM | 9473 MiB | 9473 MiB | parity |

H=2 selection change는 run별 `6--8`회였다. action fidelity violation과 post-policy override는 모두 0이었다. Primary tail은 좋아졌지만 TPOT median 때문에 long trace alone으로도 joint promotion은 보수적으로 보류해야 한다.

## 7. 12-workload gate

Artifacts:

```text
H=2     .local/transition-aware-20260830/p92-max-violation-h2-12x3/
myopic  .local/transition-aware-20260830/p84-final-myopic-12x3/
config  P8 / D64 / E8 / chunk128 / slots80 / worker8
repeats 3 per workload
```

표에서 latency improvement는 양수일수록 H=2가 좋다. `TTFT`, `TPOT`, `E2E`는 `mean/p95`다.

| Workload | H2 tok/s | tok/s delta | H2 TTFT | TTFT improvement | H2 TPOT | TPOT improvement | H2 E2E | E2E improvement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 2409.5 | +0.84% | 85.0/167.7 | +5.44%/+3.24% | 14.15/23.56 | +2.18%/+19.30% | 343.0/425.8 | +1.04%/+0.77% |
| balanced | 4037.0 | +0.55% | 323.5/551.0 | +0.42%/+1.61% | 14.05/15.89 | +0.39%/+2.51% | 1519.0/2402.7 | +0.59%/+1.26% |
| decode-heavy | 4871.8 | +0.24% | 702.4/1256.1 | +0.08%/-1.90% | 11.56/12.33 | +0.14%/+1.60% | 3688.7/5763.7 | +0.17%/+0.01% |
| long-prefill | 1118.9 | -0.16% | 3204.2/4204.0 | -0.11%/-1.56% | 28.42/33.28 | -0.04%/-0.45% | 5633.6/8017.6 | -0.12%/-1.11% |
| bimodal | 1735.9 | -0.59% | 3001.3/6462.2 | +1.78%/-2.57% | 20.08/30.23 | -1.44%/-0.39% | 5749.7/11708.0 | +0.44%/+0.03% |
| text-heavy VLM | 1428.3 | **-5.76%** | 158.2/539.2 | +4.65%/+0.10% | 17.82/23.37 | **-7.56%**/-2.73% | 1088.0/1424.0 | **-5.24%**/-2.26% |
| mixed VLM | 914.5 | -0.59% | 355.1/1156.2 | +4.55%/+9.82% | 22.49/31.44 | -0.32%/+2.85% | 1371.3/1835.7 | +1.17%/+5.10% |
| vision-heavy | 610.5 | -0.53% | 539.9/1096.3 | -3.90%/+4.34% | 29.55/37.88 | +0.60%/+2.32% | 1707.3/2445.3 | -0.99%/+2.05% |
| Poisson mixed | 1410.0 | -2.20% | 114.2/449.9 | -2.12%/-0.16% | 16.83/22.79 | -3.49%/-2.48% | 1277.8/2349.7 | -2.78%/**-3.08%** |
| wave/drain | 97.8 | -0.02% | 258.2/316.7 | -0.14%/+0.04% | 9.43/12.47 | -0.67%/+0.14% | 550.3/615.1 | +1.26%/+0.11% |
| multi-image | 303.3 | +4.40% | 241.1/307.9 | +5.65%/+2.32% | 8.59/12.09 | -0.72%/**-16.75%** | 524.6/527.5 | +2.98%/+3.92% |
| late-vision D24 | 2448.4 | -0.22% | 122.7/445.6 | -0.08%/-2.03% | 9.64/9.70 | -0.21%/-0.18% | 1503.9/1883.4 | -0.24%/-0.23% |

Text-only 5종은 모든 primary metric이 대체로 ±3%이고 balanced는 joint improvement다. 그러나 VLM gate는 실패한다.

### 7.1 Strict gate failures

- `text-heavy VLM`: throughput `-5.76%`, TPOT mean `+8.18%`, E2E mean `+5.53%` latency regression.
- `vision-heavy`: TTFT mean `+4.05%`; median은 `+18.75%`지만 p95는 개선되어 분포 이동이 비단조적이다.
- `Poisson mixed`: TPOT mean `+3.61%`, E2E p95 `+3.18%`.
- `multi-image`: TPOT p95 `+20.13%`; throughput과 E2E는 개선됐다.
- `short`: TPOT median `+3.33%`이나 mean/p95는 개선됐다. Primary mean/p95 gate 기준으로는 통과하며 median warning으로 남긴다.

### 7.2 Numerical identity

모든 workload에서 H=2 token hash 집합은 myopic token hash 집합과 교집합을 가진다.

- deterministic 3/3: short, balanced, decode-heavy, long-prefill, bimodal, text-heavy, mixed, Poisson, multi-image, late-vision.
- repeat hash variation: vision-heavy, wave/drain.
- vision-heavy와 wave/drain은 myopic에서도 이미 같은 FP16 greedy branch variation을 보였다.
- mixed는 myopic에서 2개 hash였지만 이번 H=2 run에서는 1개 hash로 수렴했다.

따라서 새 ownership corruption 증거는 없지만 exact production gate는 numerical stability 문제를 별도로 계속 추적해야 한다.

## 8. 왜 한두 번의 H=2 변경이 큰 차이를 만들 수 있는가

12-workload median telemetry:

| Workload | H=2 lookahead | selection changes | positive predicted regrets |
|---|---:|---:|---:|
| text-heavy VLM | 35 | 1 | 0 |
| mixed VLM | 58 | 1 | 0 |
| vision-heavy | 84 | 4 | 0 |
| Poisson mixed | 97 | 1 | 0 |
| wave/drain | 4 | 0 | 0 |
| late-vision D24 | 3 | 0 | 0 |

Text-heavy는 median 기준 selection change가 1회뿐인데 decode dispatch가 myopic `141`에서 H=2 `166`으로 증가했다.

```text
one action change
  -> request completion visibility changes
  -> ready-row order changes
  -> later D cohort boundary changes
  -> 25 additional D dispatches
  -> same total tokens, lower throughput and higher TPOT/E2E mean
```

반대로 multi-image는 P dispatch `5 -> 4`, D dispatch `36 -> 33`으로 줄어 throughput이 `+4.40%` 개선됐지만 작은 request count 때문에 일부 decode step 지연이 TPOT p95를 크게 바꿨다.

이 결과는 action-induced formation coupling이 존재한다는 증거인 동시에, 현재 predicted equal-work regret가 긴 causal cascade를 충분히 값으로 나타내지 못한다는 증거다. 현재 12 gate에서 positive predicted regret가 0인 이유는 oracle과 선택 action의 immediate H=2 predicted cost가 같은 minimum-violation frontier에서 거의 같기 때문이다. 이는 장기 E2E effect가 0이라는 뜻이 아니다.

## 9. vLLM 비교 정책

이번 p92는 기존 12개와 trace, model, engine, request contract가 동일하고 H=2 policy만 바뀌었다. 따라서 vLLM을 다시 실행하지 않고 frozen anchor를 재사용한다. 새 trace나 output contract를 만들 때만 fresh vLLM을 실행한다.

이번 단계의 promotion 질문은 `H=2 vs same-binary myopic`이다. vLLM은 absolute serving 수준을 확인하는 secondary anchor이며, policy-only causal gate를 대신하지 않는다.

## 10. 최종 아키텍처 상태

```text
Ready Snapshot
  E/P/D rows, stable ownership, outstanding concrete events
        |
        v
Hard Feasibility
  DAG, TRT profile, memory, single-inflight, action lease
        |
        v
Continuous Action Values
  P+D / E+P / E+D process-local RLS + uncertainty
        |
        v
SLO-Constrained Equal-Work H=2 (opt-in)
  current + <=1 observable successor
  all protected completions
  oldest resident-D service budget
  no arbitrary future arrivals
        |
        v
One Global Selector
  min violation -> efficiency -> stable order
        |
        v
Explicit Dispatch and CUDA Events
        |
        +--> exact execution cost update
        +--> contextual action-value update
        +--> predicted H=2 regret telemetry
```

Correctness authority는 deterministic mechanism에 있고, model은 이미 legal하고 SLO-safe한 action의 value만 정한다. 외부 cost registry, TTL, workload label, shape별 수동 rule은 없다.

## 11. 다음 단계

이 결과 뒤에는 H=2 score에 또 다른 heuristic penalty를 붙이지 않는다. 우선순위는 다음과 같다.

1. **Realized transition attribution**: selection-change event마다 실제 다음 2--4 dispatch, D cohort, completion visibility, protected-request service gap을 묶어 기록한다. predicted regret와 realized local outcome을 분리한다.
2. **Cascade-bounded transition state**: arbitrary arrival prediction 없이 이미-ready row와 concrete completion event가 만드는 두 번째 cohort boundary까지 표현한다. action horizon을 무조건 H=3으로 늘리지 않고 causal boundary만 확장한다.
3. **Host path optimization**: planner 자체는 microsecond급이므로 snapshot aggregate와 candidate materialization을 재사용하고 verbose telemetry serialization을 hot path 밖으로 옮긴다.
4. **Text-heavy/Poisson saturation decomposition**: GPU completion, CPU visibility, sampling, state commit, candidate formation, selector, TRT enqueue, first kernel을 timestamp로 나눈다.
5. **Promotion rerun**: long-lived A/B 뒤 12-workload gate를 다시 실행한다. 모든 stable workload가 ±3%이고 적어도 하나의 joint SLO improvement가 있어야 default를 바꾼다.
6. **vLLM freshness rule**: trace/contract 변경 시에만 fresh vLLM을 실행하고, 동일 contract 반복에서는 frozen anchor를 사용한다.

현재 권장 운영안은 계속 다음과 같다.

```text
production  myopic transition-safe selector
research    SLO-constrained H=2 + regret telemetry opt-in
```

후속 realized attribution, concrete sampling boundary, strict final-action
dominance 구현과 장기 3회 A/B는
`notes/197-realized-transition-attribution-and-concrete-boundary-20260831.md`에 정리했다.
