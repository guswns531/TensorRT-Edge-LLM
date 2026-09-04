# Production Frozen Transition Models and Promotion Gates

## 1. 결론

이번 단계에서는 predictor의 표현력만 늘리는 대신, **같은 production decision snapshot을 여러 physical model이
재생(replay)하도록 만드는 공통 검증 기반**을 완성했다.

현재 production 기본값은 계속 **V1 Scalar Immediate**다.

- V0 Exact-only, V1 Scalar Immediate, V2 Scalar+Transition은 같은 binary와 candidate frontier로 비교 가능하다.
- V2는 multi-image와 일부 mixed/Poisson 상태에서 execution--formation coupling을 줄이는 유의미한 신호를 보였다.
- 그러나 3회 반복 mixed workload에서 V2 throughput이 V1보다 4.04% 낮았다. 모든 workload에 안전한 기본값으로
  승격할 수 없다.
- V4 Decomposed-Effect+Transition과 V5 Completion+Transition은 frozen production snapshot에서 shadow replay된다.
  물리 makespan MAE는 Scalar보다 낮았지만 action profitability sign과 false-safe gate를 통과하지 못했다.
- 따라서 V4/V5는 shadow로 유지하고, V6 Selective activation은 연기한다.

이는 richer model을 구현하지 못한 결과가 아니다. **구현은 되었지만 production authority gate가 이를 거부한
결과**다. workload-specific 예외 규칙을 넣어 수치를 맞추지 않는다는 원칙을 유지했다.

## 2. 최종 비교 축

| ID | Learned quantity | Transition reasoning | 현재 상태 |
|---|---|---|---|
| V0 Exact-only | exact CUDA registry | immediate | 구현 및 12-workload 실행 |
| V1 Scalar Immediate | normalized pair advantage | immediate | production champion |
| V2 Scalar+Transition | scalar advantage | 기존 deterministic two-boundary formation evaluator | 구현, 12-workload 및 selected-4 3회 실행 |
| V3 Decomposed Effect | compression/incumbent/newcomer effect | immediate | online shadow |
| V4 Effect+Transition | V3 physical envelope | frozen two-completion-boundary replay | 구현, shadow gate 실패 |
| V5 Completion+Transition | incumbent/newcomer completion | frozen two-completion-boundary replay | 구현, shadow gate 실패 |
| V6 Selective | confidence 기반 model authority | transition | 미활성; V4/V5 gate 통과 후 진행 |

여기서 V2의 transition과 V4/V5의 replay는 구분해야 한다.

- V2는 기존 scheduler의 scalar formation-aware H2 경로를 실제 선택에 사용한다.
- V4/V5는 동일 pre-dispatch state에서 현재 후보가 만들 완료 순서와 ready/ownership transition을 shadow로 재생한다.
- V4/V5 replay는 아직 모든 successor action을 다시 생성하여 equal-work terminal cost를 최소화하는 완전한 rollout은
  아니다. 선택된 action의 최대 두 physical completion boundary를 정확히 전개하는 단계다.

## 3. 구현 구조

```text
Production ready queues / request DAG / stable leases
                       |
                       v
          Frozen pre-dispatch snapshot
      request stage, row order, ready state,
      KV pages, vision bytes, outstanding work
                       |
             same candidate identity
                       |
       +---------------+----------------+
       |               |                |
       v               v                v
  Scalar envelope  Effect envelope  Completion envelope
       |               |                |
       +---------------+----------------+
                       |
                       v
      deterministic completion-boundary replay
       E complete -> P ready / vision retained
       P complete -> D ready / vision reclaimed
       D final    -> KV physically releasable
                       |
                       v
       sign, false-safe, error, state disagreement
                       |
                       v
                promotion gate
```

### 3.1 Frozen pre-dispatch snapshot

`PhaseFrozenDecisionSnapshot`은 dispatch 전에 다음을 고정한다.

- 모든 projected request의 stage, ready 여부, canonical request ID
- E/P/D phase별 outstanding work와 candidate가 새로 launch할 work
- stable KV slot/page ownership과 vision payload ownership
- 현재 ready row와 candidate action identity

요청 ID `0`도 유효하게 취급한다. 실제 HTTP trace 첫 요청이 0이므로 이를 sentinel로 거부하면 snapshot coverage가
인위적으로 낮아진다.

주요 구현:

- `cpp/runtime/phase/mechanism/phaseIncrementalProjector.h`
- `cpp/runtime/scheduling/phaseIncrementalProjector.cpp`
- `cpp/runtime/scheduling/phaseThreeCoordinator.cpp`

### 3.2 Production request/DAG export

`IndependentPhaseAsyncServer::projectedRequests()`가 현재 request stage를 projector 표현으로 내보낸다. 별도 synthetic
state를 만드는 것이 아니라 production queue와 stable lease의 실제 상태를 사용한다.

### 3.3 Exact physical reclaim

KV reclaim은 logical page count가 아니라 실제로 이 request 완료로 반환할 수 있는 page만 센다.

```text
refcount == 1  -> request 완료 시 physical reclaim 가능
refcount > 1   -> donor/prefix sharing 중이므로 reclaim 불가
```

`StableKVPageManager::releasablePages()`가 이를 계산한다. vision payload 해제와 함께 transition의 memory delta에
반영된다.

### 3.4 Telemetry와 analyzer

decision event에는 frozen snapshot coverage와 각 candidate의 scalar/effect/completion transition 결과가 기록된다.
completion event의 common GPU epoch와 연결하여 다음을 계산한다.

- frozen snapshot coverage
- common-epoch physical sample 수
- profitability sign agreement
- false-safe: 실제로 손해인데 model이 안전한 이득으로 판단한 횟수
- makespan MAE/p95
- model 간 ready row/reclaim transition disagreement
- model별 promotion gate

도구는 `benchmarks/phase_serving/analyze_transition_fidelity.py`다.

## 4. Physical-model production gate

Artifact:

```text
.local/transition-fidelity-full12-i1-20260904
```

I1 generic calibration, V1 active, Effect/Completion shadow, 12개 production HTTP workload에서 얻었다.

| 지표 | Scalar | Decomposed Effect | Completion |
|---|---:|---:|---:|
| common-epoch ready samples | 3,061 | 2,314 | 3,062 |
| sign agreement | 90.17% | 73.38% | 72.63% |
| false-safe | 35 | 25 | 26 |
| makespan MAE | 8.842 ms | 6.900 ms | 6.148 ms |
| promotion | fail | fail | fail |

기본 gate는 다음과 같다.

```text
samples >= 100
sign agreement >= 80%
false-safe == 0
```

핵심 해석은 **시간 오차가 낮다고 scheduling decision이 좋아지는 것은 아니다**라는 점이다. Completion은 MAE가
가장 낮지만 sign agreement가 가장 낮은 편이고 false-safe가 남아 있다. 현재 objective에서 더 정확한 timestamp
model을 바로 authority로 쓰면 오히려 위험하다.

### 4.1 방향별 data density

| Direction | Samples | Scalar sign | Effect sign | Completion sign |
|---|---:|---:|---:|---:|
| P -> D | 2,959 | 92.19% | 76.67% | 74.96% |
| E -> P | 136 | 31.11% | 2.22% | 4.44% |
| E -> D | 22 | 33.33% | 0.00% | 16.67% |
| D -> E | 4 | cold | cold | cold |

E 경로를 처음부터 같은 권한으로 활성화하면 안 되는 이유가 수치로 확인됐다. 문제는 workload label이 없어서가
아니라, 자연 serving에서 해당 direction의 counterfactual-quality evidence가 너무 희소하기 때문이다.

### 4.2 Workload별 sign agreement

| Workload | Scalar | Effect | Completion |
|---|---:|---:|---:|
| balanced | 94.3% | 84.8% | 75.0% |
| bimodal | 96.8% | 81.8% | 78.8% |
| decode-heavy | 94.7% | 91.0% | 84.1% |
| late-vision | 80.1% | 61.7% | 52.5% |
| long-prefill | 95.5% | 77.5% | 79.3% |
| mixed | 88.9% | 74.7% | 74.9% |
| multi-image | 93.0% | 70.9% | 71.3% |
| poisson | 92.1% | 74.5% | 70.2% |
| short | 84.2% | 76.8% | 77.2% |
| text-heavy | 75.1% | 60.2% | 60.2% |
| vision-heavy | 74.6% | 57.0% | 53.6% |
| wave-drain | 62.8% | 47.0% | 46.0% |

이 표를 workload별 rule로 사용하지 않는다. heterogeneous state에서 발생하는 일반화 실패를 찾는 diagnostic이다.

## 5. 같은 binary/engine/frontier에서 V0--V2 12-workload 비교

이 실험은 세 variant 모두 다음 조건을 공유한다.

```text
max-in-flight = 48
same engine / CUDA graph / admission / chunk P128
same I1 generic calibration family
same HTTP trace and candidate frontier
```

### 5.1 Token throughput

| Workload | V0 Exact | V1 Scalar | V2 Scalar+T | V1 vs V0 | V2 vs V1 |
|---|---:|---:|---:|---:|---:|
| short | 2,503.70 | 2,473.04 | 2,472.97 | -1.22% | 0.00% |
| balanced | 3,792.48 | 3,776.96 | 3,780.08 | -0.41% | +0.08% |
| decode-heavy | 4,458.91 | 4,431.53 | 4,460.44 | -0.61% | +0.65% |
| long-prefill | 1,157.31 | 1,194.97 | 1,185.82 | +3.25% | -0.77% |
| bimodal | 1,903.36 | 1,953.18 | 1,950.91 | +2.62% | -0.12% |
| text-heavy | 1,598.45 | 1,652.80 | 1,682.36 | +3.40% | +1.79% |
| mixed | 1,039.58 | 1,041.28 | 1,150.36 | +0.16% | +10.48% |
| poisson | 1,594.83 | 1,680.98 | 1,788.11 | +5.40% | +6.37% |
| vision-heavy | 687.69 | 672.72 | 658.81 | -2.18% | -2.07% |
| wave-drain | 95.65 | 95.64 | 97.81 | -0.02% | +2.27% |
| late-vision | 2,540.42 | 2,548.23 | 2,554.13 | +0.31% | +0.23% |
| multi-image | 306.89 | 197.65 | 203.53 | -35.60% | +2.98% |

단일 실행 aggregate는 다음과 같다.

- V1 vs V0: mean -2.07%, median +0.07%, 6/12 wins
- V2 vs V1: mean +1.83%, median +0.44%, 8/12 wins, 3% 초과 loss 0개
- V2 vs V0: mean -0.30%, median +1.40%, 8/12 wins

multi-image는 요청이 5개라 단일 실행 분산이 매우 크다. 따라서 위 표만으로 policy를 승격하지 않았다.

### 5.2 Selected-4 3회 반복

| Workload | V1 tok/s | V2 tok/s | Delta | V1 TTFT mean/p95 | V2 TTFT mean/p95 |
|---|---:|---:|---:|---:|---:|
| mixed | 1,104.89 | 1,060.26 | -4.04% | 421.0 / 1,088.2 | 420.3 / 1,087.9 |
| multi-image | 243.96 | 311.07 | +27.51% | 354.6 / 451.1 | 235.5 / 305.8 |
| poisson | 1,651.32 | 1,684.61 | +2.02% | 173.6 / 695.9 | 165.0 / 754.6 |
| vision-heavy | 690.20 | 700.03 | +1.42% | 897.2 / 2,129.5 | 808.3 / 1,897.2 |

| Workload | V1 TPOT mean/p95 | V2 TPOT mean/p95 | V1 E2E mean/p95 | V2 E2E mean/p95 |
|---|---:|---:|---:|---:|
| mixed | 27.41 / 38.31 | 25.38 / 35.86 | 1,783.0 / 2,446.4 | 1,637.0 / 2,004.8 |
| multi-image | 8.63 / 11.12 | 8.30 / 11.05 | 621.9 / 655.3 | 511.1 / 514.1 |
| poisson | 20.29 / 28.47 | 19.84 / 31.04 | 1,525.4 / 2,217.1 | 1,490.2 / 2,149.7 |
| vision-heavy | 32.69 / 53.08 | 34.82 / 59.03 | 2,264.7 / 3,148.9 | 2,132.1 / 2,870.3 |

V2는 transition reasoning의 잠재력을 보였지만 모든 축을 지배하지 않는다.

- mixed: throughput은 낮지만 TPOT와 E2E는 좋아진다.
- multi-image: throughput과 TTFT/E2E가 크게 좋아진다.
- poisson: throughput과 mean latency는 좋아지지만 TTFT/TPOT p95 일부가 나빠진다.
- vision-heavy: throughput, TTFT, E2E는 좋아지지만 TPOT는 나빠진다.

따라서 지금 필요한 것은 workload별 toggle이 아니라 **request-level SLO와 successor service를 같은 equal-work
horizon에서 비교하는 selector**다.

## 6. 예전 champion과의 차이: policy가 아니라 실험 계약

처음의 max-in-flight 48 결과를 과거 champion/vLLM 표와 직접 비교하면 성능이 퇴보한 것처럼 보인다. 그러나 과거
frozen champion은 다음 조건이었다.

```text
max-in-flight = 64
generic calibration = v7-small-d
text calibration = generic-text-v7-small-d.json
VLM calibration  = generic-vlm-v7-small-d.json
```

이번 V0/V1/V2 causal matrix는 `max-in-flight=48`과 이전 p9 calibration을 사용했다. 따라서 variant 간 상대 비교는
공정하지만 과거 절대 성능과는 계약이 다르다.

V1을 과거와 같은 `64 + v7`로 확인한 결과:

| Workload | New confirmation | Past frozen Current | 판단 |
|---|---:|---:|---|
| balanced | 4,601.67 tok/s | 4,461.3 tok/s | champion capacity 재현 |
| decode-heavy | 5,270.18 tok/s | 5,255.1 tok/s | 사실상 동일 |
| multi-image | 247.68 tok/s | 236.2 tok/s | 높은 분산 범위에서 재현 |

이 결과는 최근 코드가 예전 성능 경로를 잃은 것이 아니라는 것을 확인한다.

## 7. vLLM 비교의 올바른 사용

과거 frozen vLLM 비교 역시 `max-in-flight=64 + v7` 계약에서만 유효하다.

| Workload | Frozen Current tok/s | Current vs vLLM |
|---|---:|---:|
| short | 2,525.1 | +27.30% |
| balanced | 4,461.3 | +3.27% |
| decode-heavy | 5,255.1 | +8.26% |
| long-prefill | 1,190.5 | +6.21% |
| bimodal | 1,950.5 | +4.40% |
| text-heavy | 2,010.8 | +23.00% |
| mixed | 1,099.2 | +19.29% |
| poisson | 2,019.8 | +12.21% |
| vision-heavy | 644.4 | +11.25% |
| wave-drain | 98.1 | +2.35% |
| late-vision | 2,553.6 | +8.24% |
| multi-image | 236.2 | -3.40% |

새 V0/V1/V2 max-48 표와 위 vLLM 표를 교차 나눗셈하지 않는다. 동일 workload를 변경하지 않았으므로 vLLM을
매번 재실행할 필요도 없지만, 최종 후보를 승격할 때는 `64 + v7`, 동일 HTTP contract에서 fresh 반복한다.

## 8. Correctness와 검증

### 8.1 테스트

- Python: 12 passed
  - warm-up matrix variant contract
  - transition fidelity grouping, false-safe, promotion gate
  - effect-vector shadow analysis
- C++: 18 passed
  - incremental projector
  - frozen replay/action identity
  - stable KV page manager와 exact reclaim
- TensorRT 11.0.0 / CUDA 13.3 container full build 성공
- pre-commit: 변경 파일 전체 통과

### 8.2 남은 correctness gate

일부 반복에서 token trace hash가 달라졌다. 알려진 FP16 numerical/order sensitivity일 수 있지만 production 승격
전에는 다음을 분리해야 한다.

- stable request ID 기반 canonical row order
- 동일 candidate membership/binding shape
- 같은 KV page ownership hash
- paired greedy identity 또는 logit/top-k closeness

transition이 좋아 보인다는 이유로 이 gate를 완화하지 않는다.

## 9. 왜 모든 branch를 active로 만들지 않았나

`모두 진행`은 모든 실험적 model을 강제로 production에 켠다는 뜻으로 해석하지 않았다. correctness와 promotion
gate를 통과하지 않은 model을 켜면 workload-specific fine tuning을 피하려던 목표와 반대가 된다.

```text
법적 action인가?       deterministic mechanism
물리 예측이 검증됐나?  held-out common-epoch gate
SLO-safe한가?          request-level guard
전체 gate 통과?        production authority
```

현재 V4/V5는 두 번째 단계에서 실패한다. V6 Selective는 richer model이 적어도 일부 state domain에서 gate를 통과한
뒤에만 의미가 있다.

## 10. 다음 구현 계획

### P0. Strict paired correctness

Candidate frontier, row order, request membership, binding shape, KV/vision ownership hash를 paired dispatch signature로
고정한다.

### P1. Equal-work successor rollout

현재 completion-boundary replay 위에 successor candidate construction을 연결한다.

```text
a0 completion envelope
  -> deterministic DAG transition
  -> newly ready E/P/D
  -> same batch builder
  -> best legal a1
  -> equal-work terminal service/SLO cost
```

임의 미래 arrival은 예측하지 않고 현재 ready work와 이미 outstanding인 completion만 사용한다.

### P2. Controlled counterfactual replay

multi-image와 E/P fragmentation state에서 pre-branch snapshot을 완전히 같게 만든 뒤 각 branch를 3--5회 실제
실행한다. trajectory가 다시 같은 cohort state로 합쳐질 때까지 D dispatch 수, GPU service, TTFT/TPOT/E2E를 잰다.

### P3. Decomposed vs Completion shadow 재평가

Immediate makespan MAE가 아니라 equal-work two-boundary regret, false-safe, SLO violation으로 gate를 다시 평가한다.

### P4. Selective authority

V4 또는 V5가 검증된 state domain에서만 V6을 연다. workload 이름, trace ID, static E/P/D shape rule은 feature나
authority condition으로 사용하지 않는다.

### P5. Final full12 and vLLM

최종 후보만 `max-in-flight=64 + v7`, 동일 HTTP arrival/timeout/output contract로 3--5회 재실행한다. throughput뿐
아니라 TTFT/TPOT/E2E mean/p95와 joint-SLO goodput을 함께 비교한다.

## 11. 현재 권장 architecture

```text
Deterministic legality and ownership
              +
V1 contextual Scalar immediate authority
              +
production frozen transition shadow plane
              +
strict held-out promotion gates
```

즉 현재 winner는 V1이다. 하지만 V2의 multi-image/Poisson 결과와 V4/V5의 낮은 physical MAE는 다음 단계의 가치가
있음을 보여준다. 다음 목표는 model을 더 크게 만드는 것이 아니라, **같은 equal-work successor horizon에서 어떤
물리 정보가 실제 action regret을 줄이는지**를 증명하는 것이다.
