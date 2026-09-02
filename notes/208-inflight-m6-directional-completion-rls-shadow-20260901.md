# M6 Directional Completion-Vector RLS Shadow

## 1. Outcome

M6의 learned predictor substrate와 실제 CUDA-label shadow 경로를 구현했다. Production selector는 바꾸지 않았다.
구현은 다음 두 학습 평면을 분리한다.

```text
exact CUDA cost plane                 contextual decision plane
---------------------                 -------------------------
exact action key                      16-D continuous feature
median/p95 execution reference        six ordered directions
debug/build/runtime evidence          scalar advantage + two completion heads
          |                                      |
          +-------------- shadow only -----------+
```

각 ordered direction은 서로 증거를 공유하지 않는다.

- P -> D, D -> P
- E -> P, P -> E
- E -> D, D -> E

각 direction에는 기존 normalized action-advantage head와 별도로 incumbent/newcomer completion residual head가
있다. 따라서 completion predictor는 총 12개 RLS head다. 모델은 process-local이고 TTL, external registry,
workload label을 사용하지 않는다.

M6 구현은 완료했지만 Gate C는 통과하지 않았다. 자연 workload의 방향 coverage와 same-snapshot multi-action
oracle coverage가 부족하고, P -> D completion interval도 아직 목표 calibration에 도달하지 않았다. 따라서
active learned selector인 M7은 아직 허용하지 않는다.

## 2. Prediction target and normalization

M4와 동일하게 H1 boundary는 newcomer launch/augmentation boundary다.

```text
observed completion
    = component CUDA completion - H1 augmentation boundary
```

P+D worker는 `mDispatchStart` 또는 `mAugmentationStart`에서 각 P/D done event까지 별도로 측정한다. Residual
augmentation에서는 incumbent의 전체 duration이 아니라 H1 이후 남은 completion이 label이 된다. E+P/E+D는
direction과 이미 경과한 incumbent 시간을 사용해 encoder와 상대 phase completion을 incumbent/newcomer 순서로
정렬한다.

초기 component-local normalization은 packed-P reference가 0.06 ms까지 작아지는 경우 residual이 폭증했다.
workload rule을 추가하지 않고 두 component의 serial reference 합으로 정규화를 바꿨다.

```text
serial_ref = incumbent_ref + newcomer_ref
target_i   = (observed_completion_i - component_ref_i) / serial_ref
mean_i     = component_ref_i + serial_ref * predicted_residual_i
sigma_i    = serial_ref * predicted_sigma_i
```

이 방식은 cold state에서 isolated reference를 그대로 사용하고, 한쪽 component reference가 작거나 부정확해도
수치적으로 안정적이다.

## 3. Runtime architecture

```text
Phase snapshot / bounded candidate
               |
               v
phaseContextualPairFeatures (16-D)
               |
      +--------+---------+
      |                  |
scalar advantage     ordered completion model
RLS + uncertainty    incumbent RLS + newcomer RLS
      |                  |
      +--------+---------+
               |
        shadow prediction
               |
     actual CUDA completion vector
               |
       pre-update calibration
```

Correctness authority remains deterministic:

- action legality: `PhaseIncrementalAction`
- completion transition: M4 projector
- action fidelity: unified decision/dispatch/completion chain
- KV/vision ownership: stable leases and existing hard feasibility

Learned evidence cannot legalize an action, reclaim memory, or launch a third context.

## 4. Code map

| Path | M6 responsibility |
|---|---|
| `cpp/runtime/phase/policy/phaseContextualPdModel.h` | ordered direction identity, scalar RLS, two-output completion estimate/telemetry |
| `cpp/runtime/scheduling/phaseContextualPdModel.cpp` | continuous features, RLS update, pre-update calibration, serial-reference residual normalization |
| `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h` | six scalar heads and six two-output completion models |
| `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` | direction-specific prediction/observation/reset routing |
| `cpp/runtime/scheduling/phaseDispatchWorker.cpp` | common-H1 P/D component CUDA completion timestamps |
| `cpp/runtime/scheduling/phaseQueueScheduler.cpp` | P+D shadow prediction and exact plan-bound observation |
| `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | E+P/E+D direction, reference, and component observation |
| `examples/llm/llm_phase_context_smoke.cpp` | direction/component calibration telemetry emission |
| `benchmarks/phase_serving/analyze_contextual_shadow.py` | MAE/RMSE/coverage/false-safe/decision-latency report |

## 5. Correctness validation

### 5.1 Unit/build

- TensorRT 26.06/CUDA build: pass
- `PhaseContextual*:*PhaseRuntimeCostTracker*`: 16/16 pass
- Python M6 analyzer/policy matrix tests: 5/5 pass
- opposite direction evidence independence: pass
- incumbent/newcomer residual independence: pass
- pre-update rather than post-label calibration: pass

### 5.2 Actual Cosmos unified events

Final two-trace validation covered balanced and vision-heavy real HTTP requests.

| Metric | Result |
|---|---:|
| Decisions | 1,150 |
| Dispatches/completions | 1,189 / 1,189 |
| GPU intervals | 1,189 |
| Action fidelity failures | 0 |
| Dispatch without completion | 0 |
| Completion without dispatch | 0 |

The exact greedy token hashes remained the frozen hashes:

- balanced: `f51d448d...30875ca`
- vision-heavy: `cfc7a828...82329b1`

## 6. Shadow performance

Same-binary model-disabled control and M6 shadow are policy-equivalent; one-run differences below include ordinary run
variance and are not claimed as speedups.

| Trace | Mode | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|
| balanced | disabled control | 4,153.35 | 66.78 / 165.66 | 13.32 / 15.09 | 1,203.00 / 1,893.84 |
| balanced | final serial-normalized shadow | 4,241.95 | 66.22 / 165.88 | 13.01 / 14.23 | 1,175.63 / 1,829.61 |
| vision-heavy | disabled control | 652.12 | 1,504.37 / 3,302.37 | 27.25 / 38.42 | 2,592.38 / 3,704.34 |
| vision-heavy | completion shadow | 655.66 | 1,482.64 / 3,280.27 | 27.72 / 38.57 | 2,591.97 / 3,683.19 |

Final balanced shadow scheduler decision latency was mean 225.72 us and p95 413.30 us. The earlier same-binary disabled
control p95 was 405.52 us, so the observed p95 delta was about +1.9%. This is acceptable for shadow research but is
not yet the final active-path overhead result; five-run repeats are required before promotion.

## 7. Completion calibration

After changing from component-local to serial-reference normalization, balanced P -> D ready-only calibration was:

| Component | Ready observations | MAE | RMSE | 95% interval coverage |
|---|---:|---:|---:|---:|
| incumbent P | 27 | 1.895 ms | 3.699 ms | 88.9% |
| newcomer D | 27 | 1.004 ms | 1.889 ms | 88.9% |

This improved incumbent MAE from 5.295 ms under the unstable normalization. It is still below the desired calibration
quality: a nominal 95% interval covering 88.9% is under-calibrated.

Natural direction coverage remains sparse. Across the final balanced/vision run, selected completion observations were
dominated by P -> D. E -> D, E -> P, and P -> E had only 1--2 samples per process; D -> P and D -> E were absent.
This is expected from bandit feedback: an unselected action has no online label.

## 8. Gate C decision

Gate C is **not evaluable/passed** yet.

| Requirement | Status | Evidence |
|---|---|---|
| exact action fidelity | pass | 0/1,189 failures |
| continuous interpolation | implemented | no exact policy key in RLS path |
| completion error/calibration | partial | P -> D ready data exists; coverage 88.9% |
| low false-safe | not evaluable | protected slack produced no positive-safe completion decisions in this shadow |
| oracle action ranking agreement/regret | not evaluable | no same-snapshot multi-action labels |
| all six directions | not evaluable | natural trace coverage is sparse |
| decision latency | preliminary pass | p95 413.30 us, about +1.9% vs same-binary disabled control |

The analyzer deliberately reports `ranking_regret_evaluable=false` and therefore `gate_c_evaluable=false`; it does not
turn missing labels into a zero-regret claim.

## 9. Next step

M7 should not start in active mode yet. The next work is a bounded M6 calibration matrix, not workload fine-tuning.

Update: the pair-common plus direction-specific hierarchical shrinkage and residual D->P reference fix are implemented
and validated in `notes/209-inflight-m6-hierarchical-direction-shrinkage-20260901.md`. The remaining items below still
apply to Gate C promotion.

1. Replay controlled M2 direction injections with the final binary so every one of the six directions has at least four
   ready observations at multiple start-skew points.
2. Join same-snapshot legal actions through controlled replay to compute top-1 agreement and normalized ranking regret.
3. Calibrate uncertainty globally or by action family, without workload labels, until ready interval coverage is close
   to its nominal confidence and false-safe is bounded.
4. Repeat balanced/vision-heavy and 39/48.8/97.5 saturation points five times to bound scheduler p95 overhead.
5. Only after Gate C passes, wire the learned completion vector into the existing M4 projector and run M7's 12-workload
   active promotion gate. Frozen vLLM data remains reusable because the HTTP/model/output contract has not changed.
