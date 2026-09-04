# Outcome Fidelity × Transition Reasoning 구현 및 12-Workload Shadow 검증

날짜: 2026-09-04

브랜치: `codex/v010-phase-forward-port`

## 1. 결론

이번 단계는 predictor를 무조건 더 복잡하게 만드는 방향이 아니다. 학습하는 물리 정보의 양과 그 정보를 이용해
전개하는 transition horizon을 서로 분리했다.

```text
학습 fidelity
  Exact -> Scalar -> Effect Vector -> Completion Vector

reasoning horizon
  Immediate -> immutable physical boundary replay -> successor action (future gate)
```

구현과 검증 결과는 다음과 같다.

1. Scalar, Effect-Vector, Completion-Vector가 같은 immutable request/DAG/ownership snapshot과 같은 transition
   projector를 사용하도록 공통 `OutcomeEnvelope` 계약을 구현했다.
2. Scalar는 모르는 completion order를 임의로 만들지 않고 두 가능한 order를 모두 반환한다.
3. Effect-Vector는 `overall compression`, `incumbent stretch`, `completion-order margin` 세 normalized head만
   학습한다. absolute timestamp를 모두 맞혀야 하는 Completion보다 작은 decision interface다.
4. Completion-Vector는 같은 envelope/replay API에 연결했다. predictor가 달라도 correctness mechanism은 바뀌지 않는다.
5. 기존 Scalar가 production authority를 유지한 동일 실행에서 12-workload, 4,246 common-epoch pair samples를
   수집했다. Effect shadow는 4,168 samples에서 ready였고 confident completion order accuracy는 90.17%였다.
6. 그러나 E->P는 ready 160개 중 154개가 uncertainty 때문에 ambiguous였고, confident 6개는 모두 틀렸다.
   따라서 Effect/Completion transition authority를 full E/P/D에 바로 적용하지 않았다.

현재 가장 중요한 판정은 다음과 같다.

> Decomposed Effect-Vector는 P+D에서 유망하지만 E family에 대한 production authority는 아직 근거가 없다.
> Active V6 Selective를 workload별 rule로 채우지 않고, physical error와 transition regret gate가 통과한
> direction에만 권한을 주어야 한다.

## 2. 비교 family의 정확한 의미

| ID | Physical model | Transition | 현재 상태 |
|---|---|---|---|
| V0 Exact | exact CUDA registry, uncovered shape는 conservative fallback | 없음 | 기존 baseline 유지 |
| V1 Scalar | contextual normalized overlap advantage RLS | 없음 | 현재 production authority/champion |
| V2 Scalar+T | Scalar makespan을 두 completion-order envelope로 변환 | frozen two-boundary shadow 구현 |
| V3 Effect | compression + incumbent stretch + order margin | 없음 | RLS/telemetry 구현, shadow |
| V4 Effect+T | Effect envelope와 deterministic transition | frozen two-boundary shadow 구현 |
| V5 Completion+T | incumbent/newcomer completion vector | 같은 transition evaluator | shadow 구현 |
| V6 Selective | V1 default, richer model은 evidence가 충분한 decision만 escalation | 미승격 |

`V6 미승격`은 미구현 누락이 아니다. V6 trigger를 workload 이름이나 특정 P/D/E shape 조건으로 먼저 만들면
검증 결과를 policy에 다시 암기시키는 fine-tuning table이 된다. V2/V4/V5의 동일-snapshot regret가 V1보다
좋다는 증거가 생긴 뒤 그 disagreement/uncertainty만으로 trigger를 정의한다.

## 3. 최종 아키텍처

```text
                          Ready snapshot
               request DAG / E-P-D queues / ownership
                                |
                                v
                    deterministic feasibility
              dependency / TRT shape / memory / inflight
                                |
                                v
                       same candidate frontier
                                |
             +------------------+------------------+
             |                  |                  |
             v                  v                  v
       Scalar model        Effect model      Completion model
      advantage, sigma   compression/stretch  component finish
             |             / order, sigma       time, sigma
             +------------------+------------------+
                                |
                                v
                         OutcomeEnvelope
                one or multiple plausible orders
                                |
                                v
                   immutable frozen transition
             first physical completion boundary
                                |
             request stage + KV/vision lifetime update
                                |
                                v
                  second physical completion boundary
                                |
                                v
             robust horizon / ready E-P-D / ownership
                                |
                                v
                  SLO-safe authority gate (future)
```

중요한 제한도 명시한다. 현재 replay는 같은 pair에 포함된 두 physical component가 완료될 때까지의
`two-physical-boundary replay`다. 첫 boundary 뒤 새 successor TensorRT action을 실제로 materialize하는 full two-action
rollout은 아직 아니다. 현재 구현을 `complete two-action scheduler`라고 과장하지 않는다.

## 4. 구현 위치

| 역할 | 파일 |
|---|---|
| Effect-Vector의 세 RLS head | `cpp/runtime/phase/policy/phaseContextualPdModel.h` |
| Effect prediction/update | `cpp/runtime/scheduling/phaseContextualPdModel.cpp` |
| direction별 six-model tracker | `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h` |
| common-epoch effect labels | `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` |
| candidate effect shadow | `cpp/runtime/phase/execution/phaseActionPlan.h` |
| event snapshot effect fields | `cpp/runtime/phase/mechanism/phaseUnifiedEvent.h` |
| P+D effect prediction | `cpp/runtime/scheduling/phaseQueueScheduler.cpp` |
| E+P/E+D effect prediction | `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` |
| frozen snapshot/envelope API | `cpp/runtime/phase/mechanism/phaseIncrementalProjector.h` |
| frozen replay implementation | `cpp/runtime/scheduling/phaseIncrementalProjector.cpp` |
| JSON telemetry | `examples/llm/llm_phase_context_smoke.cpp` |
| schema | `benchmarks/phase_serving/manifests/phase_event_schema_v1.json` |
| common-epoch analyzer | `benchmarks/phase_serving/analyze_effect_vector_shadow.py` |

Effect label은 같은 accepted common CUDA epoch에서 다음과 같이 계산한다.

```text
compression       = (serial_reference - pair_makespan) / serial_reference
incumbent_stretch = (incumbent_completion - incumbent_reference) / incumbent_reference
order_margin      = (newcomer_completion - incumbent_completion) / serial_reference
```

Effect와 Completion은 선택된 실제 overlap에서 동일 label source를 받는다. 별도 프로세스 A/B가 아니므로 candidate,
row order, realized skew와 GPU contention이 정확히 같다.

## 5. Immutable replay correctness contract

frozen snapshot ID는 다음을 포함한다.

```text
epoch
in-flight phase/execution/request order
request stage / stable slot / generation progress
KV and vision ownership
scalar policy-state signature
complete legal candidate frontier and row order
```

freeze/replay는 다음 경우 즉시 실패한다.

- request ID 중복 또는 missing in-flight request
- request stage와 phase 불일치
- recomputed ownership과 snapshot ownership 불일치
- action ID가 frontier에 없거나 illegal
- predictor envelope의 action ID 불일치
- invalid/negative completion vector

live queue, TensorRT context, KV allocator와 vision lease는 replay에서 참조하거나 변경하지 않는다.

## 6. Candidate hard-gate audit

### 6.1 Correctness gate: 제거하지 않음

| Gate | 이유 |
|---|---|
| request DAG dependency | E가 필요한 요청은 E completion 전 P 불가 |
| one in-flight execution per TensorRT context | context 재진입 금지 |
| stable slot/page generation | released/reused KV ownership 접근 금지 |
| `allowChunkedPrefill` / `supportsChunkedPrefill` | model attention contract; Gemma vision atomic path 포함 |
| `exclusivePrefill` | 해당 request의 binding/semantic contract |
| pair action fidelity | planned action과 actual outstanding set 일치 |

### 6.2 Engine capability gate: scheduler policy가 아님

| Gate | 현재 값/의미 |
|---|---|
| max prefill profile | P8 |
| max decode profile | D64 |
| max encoder execution | E4 primary serving configuration |
| fixed prefill chunk | 128; current engine/profile contract |
| external vision prefill profile | 별도 TensorRT shape limit |
| candidate count | 최대 11; host decision 비용 bound |

이 값은 workload label이 아니라 build/runtime shape capability다. 다른 엔진으로 바꾸면 자동으로 다른 범위를 노출해야 한다.

### 6.3 Memory/admission gate: 유지

KV page reservation, vision payload bytes, TensorRT workspace와 page-pool pressure는 action feasibility를 제한한다.
이 gate는 action score와 분리하고, feasible action을 줄이는 역할만 한다.

### 6.4 Performance heuristic: 향후 제거/ablation 대상

```text
maxOverlapPrefillTokens = 128
maxOverlapPrefillBatchSize
maxConsecutiveOverlapBatches
decodeBurstLimit
maxPredictedOverlapPrefillMs
minObservedOverlapRatio
encoderBatchWaitUs / encoderCreditWaitUs
prefillBatchWaitUs
static decodeBatchCosts / encoderBatchCosts
```

현재 Global selector가 exact/contextual cost로 일부를 대체하지만 candidate formation 이전에 적용되는 cap은 learner가
보지 못하는 blind spot이다. 단, P chunk 128은 과거 최적점이면서 현재 engine contract이므로 이번 revision에서는
고정한다. 다음 ablation은 candidate를 갑자기 무제한으로 열지 않고 `correctness/capability` 밖의 gate만 하나씩
shadow frontier에 추가해 regret를 측정한다.

## 7. Initialization contract

| Mode | 의미 | 이번 상태 |
|---|---|---|
| I0 zero-start | RLS/exact overlap evidence 없이 시작 | 12-workload 실행 완료 |
| I1 generic calibration | 같은 model/GPU의 workload-agnostic calibration | primary 12-workload 실행 완료 |
| I2 trace-derived | 대상 trace로 충분한 evidence를 만든 upper bound | 새 Active contender가 없어 실행 보류 |

I2는 성능 claim이 아니라 ceiling이다. V2/V4/V5가 shadow regret gate를 통과해 실제 contender가 된 뒤 최종 두 후보만
I0/I2로 확장한다. 모든 모델을 trace-derived로 돌려 winner를 선택하면 workload fine-tuning이 된다.

## 8. 12-workload 동일 실행 결과

artifact:

```text
.local/effect-vector-full12-shadow-20260904
```

실행 contract:

- model: `nvidia/Cosmos-Reason2-2B`
- GPU: RTX 3080 10 GiB
- same tied P8/D64, fixed-P128, indexed-paged engine
- p10 command manifest와 같은 request trace, generic warmup, admission, memory
- 기존 V1 Scalar만 action authority 보유
- V3 Effect와 Completion은 같은 selected dispatch에서 shadow observation
- one repeat, counterfactual telemetry; performance promotion 수치가 아니라 model diagnostic

### 8.1 Serving result

양의 delta는 Current가 빠르다는 뜻이다. p10/vLLM은 frozen 3-run median이고 Current는 이번 1회이므로 3% 안팎 차이는
성능 결론으로 사용하지 않는다.

| Workload | Current tok/s | vs p10 | vs vLLM | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| short | 2525.1 | +0.97% | +27.30% | 89.3 / 169.6 | 13.10 / 21.93 | 326.4 / 405.7 |
| balanced | 4461.3 | -2.26% | +3.27% | 64.5 / 169.0 | 12.38 / 13.88 | 1119.1 / 1752.4 |
| decode-heavy | 5255.1 | -1.14% | +8.26% | 64.1 / 182.6 | 10.60 / 11.25 | 2800.9 / 4301.3 |
| long-prefill | 1190.5 | -0.68% | +6.21% | 2104.1 / 2591.3 | 26.64 / 31.17 | 4368.4 / 5978.9 |
| bimodal | 1950.5 | -0.08% | +4.40% | 1862.4 / 4211.0 | 17.63 / 28.19 | 4284.0 / 9298.9 |
| text-heavy | 2010.8 | -4.74% | +23.00% | 284.2 / 1111.2 | 24.95 / 37.87 | 1572.7 / 1657.7 |
| mixed | 1099.2 | -1.80% | +19.29% | 677.4 / 2269.2 | 37.63 / 52.97 | 2397.6 / 2620.2 |
| poisson | 2019.8 | +2.91% | +12.21% | 233.9 / 807.7 | 20.02 / 38.16 | 1516.6 / 1941.4 |
| vision-heavy | 644.4 | -9.32% | +11.25% | 1361.1 / 3312.2 | 38.56 / 53.38 | 2883.4 / 3733.6 |
| wave-drain | 98.1 | +0.03% | +2.35% | 239.0 / 293.0 | 8.07 / 11.55 | 489.0 / 500.6 |
| late-vision | 2553.6 | -0.05% | +8.24% | 122.9 / 415.7 | 9.23 / 9.30 | 1445.0 / 1805.3 |
| multi-image | 236.2 | -26.88% | -3.40% | 362.6 / 472.8 | 8.81 / 11.59 | 635.7 / 677.2 |

Geometric mean은 p10 대비 -3.93%, vLLM 대비 +9.87%다. 이 수치는 counterfactual JSON logging과 한 번의 run을
포함하므로 새 shadow 모델의 overhead라고 단정하지 않는다. 특히 multi-image는 이전에도 20% 이상 run variance가
있었다. 최종 contender가 정해진 뒤 telemetry-off 3~5회로 다시 측정한다.

### 8.2 Physical prediction aggregate

| Metric | Result |
|---|---:|
| common-epoch selected pair samples | 4,246 |
| Effect ready | 4,168 (98.16%) |
| Scalar makespan MAE | 12.009 ms |
| Completion makespan MAE | 13.153 ms |
| Effect compression MAE | 0.146 |
| Effect incumbent-stretch MAE | 0.206 |
| Effect order-margin MAE | 0.104 |
| Effect confident order accuracy | 90.17% (2,146 / 2,380) |
| Effect ambiguous order | 1,788 |

Completion의 aggregate makespan MAE는 Scalar보다 나쁘고 RMSE도 큰 outlier를 갖는다. 더 rich한 출력이라는 이유만으로
더 좋은 policy model이라 부를 수 없다.

### 8.3 Direction별 결과

| Direction | Samples | Ready | Confident | Ambiguous | Order accuracy | 해석 |
|---|---:|---:|---:|---:|---:|---|
| P->D | 4,006 | 4,006 | 2,372 | 1,634 | 90.39% | 충분한 primary evidence |
| E->P | 216 | 160 | 6 | 154 | 0% | 대부분 안전하게 ambiguous; authority 금지 |
| E->D | 3 | 2 | 2 | 0 | 100% | 표본이 너무 작음 |
| D->E | 9 | 0 | 0 | 0 | 0% | cold |
| P->E | 12 | 0 | 0 | 0 | 0% | cold |

E->P에서 실제 newcomer P는 216/216 모두 incumbent E보다 늦게 끝났다. Effect mean은 ready 160개 중 62개에서만
그 sign을 예측했지만 154개를 uncertainty interval crossing으로 ambiguous 처리했다. 이 uncertainty가 현재는 올바른
안전장치다. ambiguity를 강제로 positive/negative action으로 바꾸면 안 된다.

### 8.4 Workload별 shadow coverage

| Workload | Common epoch | Effect ready | Confident order accuracy | Scalar MAE ms | Completion MAE ms |
|---|---:|---:|---:|---:|---:|
| short | 118 | 118 | 85.7% | 5.077 | 4.648 |
| balanced | 250 | 250 | 83.6% | 5.050 | 4.022 |
| decode-heavy | 292 | 292 | 86.4% | 5.222 | 4.276 |
| long-prefill | 806 | 806 | 93.4% | 3.542 | 3.111 |
| bimodal | 480 | 480 | 89.2% | 4.263 | 3.697 |
| text-heavy | 283 | 272 | 92.2% | 18.585 | 16.642 |
| mixed | 307 | 296 | 86.3% | 17.992 | 15.539 |
| poisson | 458 | 448 | 92.5% | 13.476 | 12.519 |
| vision-heavy | 339 | 328 | 93.0% | 24.917 | 20.254 |
| wave-drain | 348 | 340 | 90.4% | 13.151 | 14.500 |
| late-vision | 294 | 282 | 95.1% | 26.029 | 58.409 |
| multi-image | 271 | 256 | 78.2% | 18.715 | 16.479 |

Completion은 9개 workload에서 makespan MAE가 Scalar보다 작지만 `wave-drain`과 특히 `late-vision`에서 나쁘다.
따라서 workload 평균으로 selector를 고르는 것이 아니라 동일 snapshot에서 실제 action regret sign을 비교해야 한다.

### 8.5 I0 zero-start와 I1 generic calibration 비교

zero-start artifact:

```text
.local/effect-vector-full12-zero-r2-20260904
```

I0는 `--warmup-requests 0 --policy-warmup-mode zero_start`로 실행했다. command template의
`TRT_EDGELLM_POLICY_WARMUP_MODE={policy_warmup_mode}`는 HTTP harness가 각 run을 시작할 때 실제 mode로 치환하며,
runtime event와 client aggregate에도 `zero_start`가 기록된 것을 확인했다.

| Workload | tok/s I0 | tok/s I1 | I1 vs I0 | TTFT mean/p95 I0 -> I1 ms | TPOT mean/p95 I0 -> I1 ms | E2E mean/p95 I0 -> I1 ms |
|---|---:|---:|---:|---:|---:|---:|
| short | 2434.7 | 2525.1 | +3.71% | 101.7/187.8 -> 89.3/169.6 | 12.90/25.97 -> 13.10/21.93 | 338.7/418.6 -> 326.4/405.7 |
| balanced | 4348.8 | 4461.3 | +2.59% | 59.6/177.4 -> 64.5/169.0 | 12.81/14.38 -> 12.38/13.88 | 1151.5/1781.4 -> 1119.1/1752.4 |
| decode-heavy | 5183.4 | 5255.1 | +1.38% | 59.8/202.2 -> 64.1/182.6 | 10.78/11.39 -> 10.60/11.25 | 2843.3/4328.9 -> 2800.9/4301.3 |
| long-prefill | 1244.3 | 1190.5 | -4.33% | 2024.4/2688.3 -> 2104.1/2591.3 | 25.08/29.99 -> 26.64/31.17 | 4157.9/5864.6 -> 4368.4/5978.9 |
| bimodal | 1929.9 | 1950.5 | +1.07% | 1857.2/4026.8 -> 1862.4/4211.0 | 18.14/30.20 -> 17.63/28.19 | 4316.6/9063.8 -> 4284.0/9298.9 |
| text-heavy | 2038.6 | 2010.8 | -1.37% | 256.4/961.2 -> 284.2/1111.2 | 25.14/38.02 -> 24.95/37.87 | 1547.4/1650.9 -> 1572.7/1657.7 |
| mixed | 1015.0 | 1099.2 | +8.29% | 773.2/2407.0 -> 677.4/2269.2 | 36.30/44.57 -> 37.63/52.97 | 2470.8/2816.1 -> 2397.6/2620.2 |
| poisson | 2020.6 | 2019.8 | -0.04% | 184.8/756.4 -> 233.9/807.7 | 21.25/39.67 -> 20.02/38.16 | 1531.3/1981.2 -> 1516.6/1941.4 |
| vision-heavy | 620.1 | 644.4 | +3.91% | 1464.1/3485.6 -> 1361.1/3312.2 | 34.71/47.80 -> 38.56/53.38 | 2807.8/3900.6 -> 2883.4/3733.6 |
| wave-drain | 97.9 | 98.1 | +0.23% | 216.7/302.9 -> 239.0/293.0 | 9.34/12.00 -> 8.07/11.55 | 506.2/539.6 -> 489.0/500.6 |
| late-vision | 2525.3 | 2553.6 | +1.12% | 122.1/459.3 -> 122.9/415.7 | 9.30/9.36 -> 9.23/9.30 | 1453.8/1825.7 -> 1445.0/1805.3 |
| multi-image | 321.0 | 236.2 | -26.41% | 217.6/294.4 -> 362.6/472.8 | 8.85/11.91 -> 8.81/11.59 | 492.0/498.2 -> 635.7/677.2 |

I1/I0 token throughput geometric mean은 `-1.22%`다. 따라서 generic calibration이 항상 serving 성능을 높인다는
주장은 현재 데이터로 성립하지 않는다. multi-image 한 번의 `-26.41%`와 알려진 큰 run variance가 geometric mean을
강하게 좌우하므로, 이 표는 calibration 효과의 최종 성능 결론이 아니라 warm-up sensitivity 진단이다.

| Learning metric | I0 zero-start | I1 generic |
|---|---:|---:|
| common-epoch samples | 1,327 | 4,246 |
| Effect ready | 1,234 (92.99%) | 4,168 (98.16%) |
| confident completion-order samples | 1,078 | 2,380 |
| confident completion-order accuracy | 88.31% | 90.17% |
| P->D samples / ready | 1,320 / 1,234 | 4,006 / 4,006 |
| E-family samples / ready | 7 / 0 | 240 / 162 |

I0에서도 긴 text trace 안에서 P+D는 빠르게 학습되어 Effect ready가 92.99%까지 올라갔다. 반면 짧거나 vision 중심인
trace는 해당 run 안에서 ready가 되기 전에 끝났고, E direction은 사실상 학습되지 않았다. I1은 coverage를 크게
늘리지만 E->P의 confident sign은 여전히 잘못되므로 `더 많은 sample`만으로 authority를 주지 않는다.

I0와 I1의 MAE를 직접 우열 비교하지 않는다. 두 mode가 선택한 action과 common epoch 집합 자체가 달라 같은 held-out
sample set이 아니기 때문이다. 다음 단계의 올바른 비교 단위는 동일 frozen snapshot/candidate의 repeated causal
outcome과 two-boundary regret다.

## 9. Validation

```text
C++ focused tests:
  PhaseIncrementalProjectorTest
  PhaseFrozenReplayTest
  PhaseContextualEffectModelTest
  PhaseUnifiedEventTest
  PhaseGlobalSchedulerTest
  62/62 passed

Python:
  test_effect_vector_shadow_analysis.py
  test_oracle_h1_policy_matrix.py
  9/9 passed

Actual GPU:
  I0 multi-image HTTP smoke: success, 5 requests / 160 tokens
  I0 complete 12-workload HTTP trace: 12/12 completed
  I1 complete 12-workload HTTP trace: 12/12 completed
```

## 10. Promotion decision

| Gate | Result |
|---|---|
| common immutable replay mechanism | PASS |
| Scalar unknown-order conservatism | PASS |
| Effect three-head online learning | PASS |
| Completion uses same transition API | PASS |
| P+D Effect readiness | PASS |
| E direction sample sufficiency | FAIL |
| Effect E->P confident order accuracy | FAIL |
| V2/V4/V5 same-snapshot action-regret improvement | NOT YET MEASURED |
| V4/V5 active 12-workload | BLOCKED BY PRIOR GATE |
| V6 Selective trigger | DEFERRED UNTIL DISAGREEMENT EVIDENCE |

따라서 production champion은 계속 V1 Scalar다. richer model은 shadow에서 더 많은 label을 모으지만 action authority는 없다.

## 11. 다음 순서

1. frozen snapshot을 실제 coordinator/server request-state export에 연결한다. 현재 projector 단위 테스트의 synthetic
   snapshot을 production decision snapshot으로 바꾼다.
2. V1/V2/V4/V5가 같은 frozen frontier의 모든 candidate를 평가하도록 하고, measured replay가 있는 candidate에 대해
   two-boundary regret sign과 false-safe를 계산한다.
3. P+D부터 `>=3 repeated causal outcomes`, regret sign agreement `>=80%`, false-safe `0`을 통과시키고 V2/V4를
   제한적으로 active A/B한다.
4. E direction은 generic calibration에서 natural E->P/E->D coverage가 생기도록 action shape를 늘리되, target workload
   trace는 사용하지 않는다.
5. V4가 V2보다 좋고 V5와 비슷할 때 Effect+T를 최종 후보로 선택한다. V5가 유의하게 좋을 때만 Completion+T를 선택한다.
6. 최종 두 후보만 I0/I1/I2와 cross-trace order를 실행한다.
7. 그 뒤 telemetry-off 12-workload 3회, unstable VLM 5회, fresh vLLM을 동일 HTTP contract로 실행한다.

연구 질문은 predictor의 복잡도 경쟁이 아니다.

> 현재와 다음 request-ready transition을 안전하게 선택하는 데 필요한 최소 physical information은 무엇인가?

이번 결과는 P+D에서 Effect-Vector가 그 최소 interface 후보가 될 수 있음을 보여 주지만, E/P/D 전체에 대한 답은 아직
아니다.
