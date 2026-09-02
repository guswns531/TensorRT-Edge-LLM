# Completion-vector projection P0–P7 구현 및 검증 결과

날짜: 2026-09-02  
브랜치: `codex/v010-phase-forward-port`  
대상: `nvidia/Cosmos-Reason2-2B`, RTX 3080 10 GiB, TensorRT 11.0/CUDA 13.3

## 1. 결론

P0부터 P7까지 계획된 구현, controlled characterization, natural trace 수집,
promotion gate, same-runtime ablation, 조건부 H2 판정을 완료했다.

핵심 결과는 다음과 같다.

1. E/P/D의 planned action과 실제 outstanding execution을 plan/execution ID 및
   GPU interval로 연결했고, 최종 108-run P6 검증에서 action-fidelity violation은
   `0/96,302 executions`였다.
2. Completion feature V2는 incumbent age, isolated-reference 대비 elapsed ratio,
   requested/observed skew, cost ratio, outstanding set, protected slack을 workload
   label 없이 투영한다.
3. P3 controlled matrix는 6개 direction 중 P→D, D→P, P→E 세 direction에서만
   유효한 offset coverage와 material effect를 얻었다. Host sleep 기반 injection은
   정확한 CUDA start offset을 보장하지 못하므로 나머지 direction의 부정 결과를
   overlap 부재로 해석하면 안 된다.
4. P4 natural workload에서는 co-launch와 residual opportunity가 충분히 있었지만,
   replay-regret median/p95가 모두 0 µs인 경우가 대부분이었다. 현재 natural
   counterfactual estimator는 selector authority를 정당화할 만큼 구별력이 없다.
5. P5 active는 authority-off보다 48.8 trace paired SLO goodput을 3/3 개선했지만,
   절대 gate인 40 req/s, 99% joint-SLO, held-out coverage를 통과하지 못했다.
6. P6 macro joint-SLO goodput은 Current 14.22 req/s가 가장 높았다. Always-overlap은
   raw throughput을 +3.97% 올렸지만 goodput은 14.05로 Current보다 낮았다.
7. 따라서 P7 H2는 활성화하지 않았다. P5 실패와 replay-regret median 0 µs 모두
   H2 진입 조건을 만족하지 않는다.

즉 completion-vector architecture와 측정 substrate는 구현·검증됐지만, 현재 active
policy는 production default로 promotion하지 않는다. Frozen Current가 계속 기본값이다.

## 2. 최종 아키텍처

```text
HTTP/IPC requests
       │
       ▼
PhaseThreeCoordinator
  ├─ deterministic request DAG / stable KV+vision ownership
  ├─ E/P/D candidate construction
  ├─ current outstanding execution lease
  └─ unified decision/dispatch/completion correlation
       │
       ▼
Hard feasibility
  dependency / TRT shape / memory / single-inflight
       │
       ▼
Contextual pair models (P+D, E+D, E+P)
  ├─ immediate advantage head
  ├─ completion-vector V2 head
  │    incumbent completion mean + uncertainty
  │    newcomer completion mean + uncertainty
  └─ chronological conformal calibration
       │
       ▼
SLO-safe selector
  Current(default) / completion shadow / completion active(ablation)
       │
       ▼
PhaseDispatchWorker → independent TensorRT E/P/D contexts
       │
       ▼
CUDA event completion
       ├─ exact runtime cost tracker update
       ├─ completion model update
       └─ unified fidelity/event record
```

정책이 틀려도 correctness가 깨지지 않도록 hard feasibility와 ownership은 학습
모델 밖의 deterministic mechanism으로 남겼다. Completion model이 ready하지 않거나
coverage가 부족하면 기존 robust Current estimate로 fallback한다.

## 3. 구현 위치

| 위치 | 구현 내용 |
|---|---|
| `cpp/runtime/phase/mechanism/phaseUnifiedEvent.h` | action mode/direction, execution identity, completion V2 field, fidelity reason |
| `cpp/runtime/phase/execution/phaseActionPlan.h` | plan에 requested direction/skew와 outstanding contract 유지 |
| `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | decision/dispatch/completion correlation, accumulated execution lease, residual augmentation, static P6 policy control |
| `cpp/runtime/scheduling/phaseDispatchWorker.cpp` | dispatch identity 및 GPU interval 전달 |
| `cpp/runtime/scheduling/phaseQueueScheduler.cpp` | phase-local execution/completion metadata 보존 |
| `cpp/runtime/phase/policy/phaseContextualPdModel.h` | completion vector, conformal state, policy ablation configuration |
| `cpp/runtime/scheduling/phaseContextualPdModel.cpp` | continuous feature projection, mean/uncertainty prediction, calibration/update |
| `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` | pair/direction별 online evidence와 completion telemetry |
| `examples/llm/llm_phase_context_smoke.cpp` | 환경 설정, unified JSON event, policy ablation wiring |
| `benchmarks/phase_serving/run_directional_injection_matrix.py` | 6 direction × 5 target offset × repeat 실행 및 cleanup |
| `benchmarks/phase_serving/analyze_directional_injection.py` | actual CUDA offset/effect 분석 |
| `benchmarks/phase_serving/analyze_completion_opportunities.py` | natural opportunity, coverage, disagreement, replay regret |
| `benchmarks/phase_serving/evaluate_completion_promotion.py` | P5와 P7 machine-readable gate |
| `benchmarks/phase_serving/analyze_completion_policy_matrix.py` | P6 9-policy HTTP metric/identity 집계 |
| `benchmarks/phase_serving/validate_phase_scheduler_events.py` | plain/gzip unified event contract 검증 |

## 4. P0 — Frozen baseline

P0는 signed commit `8116da0` (`bench: freeze completion projection baseline`)으로
고정했다. Manifest는
`benchmarks/phase_serving/manifests/completion_projection_p0_baseline.json`이다.

- upstream: v0.10.0, `71dd1bae...`
- frozen source commit: `306212e5...`
- max P/D/E batch: 8/64/8
- fixed prefill chunk: 128
- max stable slots: 80
- indexed-paged KV, 256 pages
- frozen 48.8 result: 41.798 req/s, joint-SLO 100%
- frozen vLLM: 40.901 req/s

중요한 점은 P0의 약 41.8 req/s와 이번 instrumented P4/P5의 약 33–35 req/s가
동등한 결과가 아니라는 것이다. P0은 immutable manifest/hash로 남겼고, 이번 P5는
새 계측 바이너리 내부에서 authority off/active만 바꾼 paired comparison이다.

## 5. P1 — Residual observability와 action fidelity

통합 event schema는 다음 세 종류를 같은 `run_id/plan_id/execution_id`로 연결한다.

```text
decision
  planned outstanding mask + selected action
       │
dispatch
  actual phase + enqueue timestamp + execution identity
       │
completion
  GPU start/end + observed outstanding mask + completion visibility
```

구현 중 발견하고 수정한 두 오류가 중요하다.

1. residual augmentation 뒤 incumbent completion이 최초 single mask와 비교되어
   false fidelity failure가 발생했다. 실행 lease가 허용한 accumulated mask와
   비교하도록 수정했다.
2. pair plan의 각 member는 phase-local action ID를 유지한다. pair completion에서
   selected pair action ID와 member action ID의 exact equality를 요구하지 않고,
   plan identity와 incremental identity를 검증하도록 수정했다. Single action은
   여전히 exact action ID를 요구한다.

최종 P6 validator 결과:

- events: 272,731
- decisions: 80,127
- executions: 96,302
- GPU intervals: 96,302
- dispatch without completion: 0
- completion without dispatch: 0
- action-fidelity violations: 0

## 6. P2 — Completion feature V2

V2는 exact workload key 대신 다음 continuous observable state를 사용한다.

- normalized incumbent dispatch age
- incumbent isolated reference 대비 elapsed ratio
- requested/observed launch skew
- incumbent/newcomer isolated cost ratio
- current outstanding phase mask
- minimum protected TTFT/TPOT slack ratio
- co-launch와 residual augmentation 구분
- incumbent와 newcomer completion target 분리

GPU kernel progress를 host elapsed time과 동일시하지 않는다. 완료된 CUDA marker와
host-observed age만 feature/label로 사용한다. External registry와 TTL은 사용하지 않으며,
evidence는 process-local online state다.

## 7. P3 — Controlled residual causal matrix

실험 구성은 6 direction × requested offset 0/25/50/75/90% × 3회, 총 90회다.
모든 cell에서 token trace는 deterministic했고 validation error는 0이었다.

| 항목 | 결과 |
|---|---:|
| unified events | 162,703 |
| executions | 54,766 |
| co-launch opportunities | 2,093 |
| residual opportunities | 1,099 |
| completion predictions ready | 2,129 |
| calibrated predictions | 981 (46.08%) |
| selector disagreement | 94/2,118 (4.44%) |
| replay regret mean / median / p95 | 39.18 / 0 / 0 µs |
| false-safe | 0 |
| action-fidelity failures | 0 |

Material effect를 확인한 direction은 3/6이다.

- D→P: 2 accepted buckets, 최대 약 17.23%
- P→D: 2 accepted buckets, 최대 약 10.93%
- P→E: 2 accepted buckets, 최대 약 124.43%의 큰 harmful effect
- E→P: accepted bucket은 있었으나 material range 없음
- E→D, D→E: accepted bucket 없음

이는 E overlap이 없다는 뜻이 아니다. Host sleep으로 newcomer enqueue를 늦춰도
encoder host preparation과 TensorRT submission latency 때문에 실제 GPU start가 target을
넘었다. 예를 들어 E→D target 0–90%가 actual median 약 107–196%에 형성됐다.
정확한 P3 후속 실험에는 CUDA event/barrier start gate가 필요하다.

## 8. P4 — Natural opportunity density

12개 production-like HTTP trace:

- events 30,938, decisions 9,338, executions 10,800
- co-launch opportunities 3,975, selections 1,462
- residual opportunities/selections 672/672
- completion-ready 4,142
- selector disagreement 2,395/4,142 = 57.82%
- replay regret 1,747 samples, mean/median/p95 = 0/0/0 µs
- action-fidelity failures 0

Load 39/48.8/97.5 req/s × 3회:

- events 24,893, decisions 6,977, executions 8,958
- co-launch opportunities 2,883, selections 1,981
- residual opportunities/selections 117/117
- selector disagreement 902/3,000 = 30.07%
- replay regret mean 24.07 µs, median/p95 0/0 µs
- action-fidelity failures 0

Natural opportunity 자체는 드물지 않다. 그러나 현재 replay는 동일
snapshot/action의 선택된 observation을 주로 재사용하므로 counterfactual separation이
부족하고 regret가 0으로 퇴화한다. 높은 disagreement만으로 authority를 켜면 안 된다.

## 9. P5 — Completion authority promotion gate

48.8 trace에서 shadow(off) 3회와 active 3회를 같은 계측 바이너리로 비교했다.

| Gate | 관측 | 결과 |
|---|---|---|
| exact token identity | 3/3 동일 hash | PASS |
| action fidelity | 0 failures | PASS |
| conformal false-safe | 0 | PASS |
| held-out coverage | D→P newcomer 74.60%, incumbent 88.89%; P→D 96.96/97.43% | FAIL |
| scheduler p95 | off 1,884.12 µs, active 1,923.20 µs; limit 1,978.32 | PASS |
| active raw throughput | median 34.41 req/s, required 40 | FAIL |
| active joint-SLO pass | median 51.04%, required 99% | FAIL |
| paired SLO goodput | off 13.48, active 17.56 req/s; active 3/3 win | PASS |

P5 최종 결과는 FAIL이다. Active는 paired improvement를 보였지만 절대 serving
capacity와 coverage 조건을 만족하지 못했다.

## 10. P6 — Same-runtime 9-policy evaluation

모든 정책은 같은 binary, engine, HTTP trace, warmup/calibration contract를 사용했다.
각 workload 1회이며 variance용 promotion 결과가 아니라 causal ablation이다.

| Policy | 12-workload macro throughput | Current 대비 | macro joint-SLO goodput | throughput wins | exact token identity |
|---|---:|---:|---:|---:|---:|
| Current | 24.12 | 0.00% | **14.22** | — | 12/12 |
| Serial | 23.84 | +2.01%* | 12.88 | 8/12 | 7/12 |
| Always overlap | **24.48** | +3.97% | 14.05 | 9/12 | 7/12 |
| Immediate cost | 23.77 | +1.00% | 13.80 | 7/12 | 7/12 |
| Completion shadow | 23.80 | +1.03% | 13.53 | 4/12 | 8/12 |
| Completion active | 23.66 | +0.33% | 13.90 | 8/12 | 7/12 |
| Completion w/o uncertainty | 23.93 | +1.27% | 13.57 | 7/12 | 9/12 |
| Completion w/o residual | 23.67 | +0.09% | 13.64 | 8/12 | 7/12 |
| New-work-only | 23.63 | -1.05% | 13.88 | 5/12 | 8/12 |

`Current 대비`는 workload별 percentage의 macro mean이라 absolute throughput 평균의
차이와 정확히 같지 않다. 또한 non-current policy의 cross-policy greedy identity가
7–9/12에 그쳤다. text-only fixed load P5에서는 exact identity가 유지됐으나, VLM
schedule/order가 바뀌는 trace에서는 FP16 row/reduction order 또는 binding shape에
의한 token divergence가 남아 있다. Production promotion을 막는 별도 correctness gate다.

### Current와 completion-active 상세 결과

| workload | Current req/s | Active req/s | 변화 | Current goodput | Active goodput | Serial req/s | Always req/s | Active TTFT mean/p95 | Active TPOT mean/p95 | Active E2E mean/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 98.20 | 91.61 | -6.71% | 98.20 | 91.61 | 89.85 | 94.49 | 135.4/222.0 | 15.2/25.9 | 427.5/518.0 |
| balanced | 37.14 | 37.60 | +1.26% | 8.25 | 8.36 | 37.44 | 38.18 | 93.6/204.4 | 17.3/19.7 | 1560.9/2443.3 |
| decode-heavy | 14.36 | 14.56 | +1.42% | 0.60 | 0.61 | 14.61 | 14.56 | 94.9/215.8 | 15.1/16.4 | 3990.4/6121.2 |
| long-prefill | 12.20 | 12.88 | +5.51% | 0.21 | 0.22 | 13.07 | 14.62 | 2241.4/2882.7 | 28.5/35.3 | 4654.0/6665.5 |
| bimodal | 11.20 | 11.31 | +0.93% | 0.70 | 0.94 | 11.54 | 11.90 | 2180.6/4654.0 | 19.2/26.8 | 4899.1/10190.3 |
| text-heavy | 29.00 | 29.16 | +0.57% | 12.23 | 12.30 | 30.56 | 30.03 | 602.7/1295.2 | 27.4/40.9 | 1989.4/2118.8 |
| mixed | 21.54 | 20.79 | -3.50% | 2.02 | 3.90 | 21.12 | 21.41 | 985.1/2550.3 | 33.3/40.1 | 2532.2/3007.8 |
| vision-heavy | 15.35 | 15.27 | -0.55% | 3.60 | 4.06 | 17.81 | 16.94 | 1644.6/3754.0 | 29.9/37.5 | 2782.6/4114.6 |
| poisson | 22.74 | 22.84 | +0.42% | 18.12 | 17.84 | 22.41 | 23.65 | 377.1/1146.8 | 23.1/41.3 | 1887.6/2377.1 |
| wave-drain | 3.02 | 3.02 | +0.01% | 3.02 | 3.02 | 3.02 | 3.02 | 312.4/376.1 | 8.8/12.2 | 584.0/591.8 |
| multi-image | 8.43 | 8.92 | +5.87% | 8.43 | 8.92 | 8.70 | 8.75 | 239.9/338.7 | 9.8/12.8 | 542.3/558.0 |
| late-vision | 16.22 | 16.02 | -1.22% | 15.21 | 15.02 | 16.00 | 16.21 | 147.1/485.3 | 10.2/10.3 | 1603.7/1995.1 |

고정 500ms TTFT/50ms TPOT/2.5s E2E SLO는 모든 12 trace의 원래 목적에 맞춘
개별 SLO가 아니므로 absolute macro goodput을 capacity로 해석하지 않는다. 같은 trace
안에서 policy 간 useful completion을 비교하는 P6 ablation metric으로만 사용한다.

### P6의 causal 해석

- Always-overlap은 long-prefill에서 14.62 req/s로 Current보다 높지만 vision-heavy의
  text/vision TPOT p95를 약 60.8/69.3ms까지 악화시킨다.
- Serial은 long-prefill formation을 잘 보존하지만 vision-heavy goodput이 0.56 req/s로
  떨어진다. Overlap 제거도 일반해가 아니다.
- Completion-active는 long-prefill +5.51%, multi-image +5.87%지만 short -6.71%, mixed
  -3.50%다. 한 configuration으로 Current를 지배하지 못했다.
- New-work-only는 decode-heavy 13.22, long-prefill 11.98 req/s로 active보다 크게 낮아
  overlap에서 incumbent completion cost도 함께 예측해야 함을 지지한다.
- Uncertainty 제거는 raw throughput을 올리는 경우가 있지만 macro goodput이 active보다
  낮고 bimodal tail이 악화된다. Uncertainty는 SLO protection에 필요하다.

## 11. P7 — Conditional H2 decision

Machine-readable gate 결과:

- P5 passed: false
- comparable decisions: 950
- selector disagreement: 18.21%
- replay-regret median: 0 µs
- H2 eligible: false

따라서 earliest-completion one-step successor authority는 실행하지 않았다. 이는 P7을
생략한 것이 아니라 계획의 stop condition을 적용한 결과다. Future arrival prediction도
추가하지 않았다.

## 12. 검증과 아티팩트

### Test

- C++ related unit tests: 31/31 PASS
- P3 90-run token trace: cell 내부 deterministic
- P4 event validation: 0 fidelity failures
- P5 fixed-load exact token identity: 3/3
- P6 event validation: 0/96,302 fidelity failures
- P6 cross-policy exact identity: Current 12/12, 나머지 7–9/12

### Local artifacts

- P3: `/home/sslab/TensorRT-Edge-LLM/.local/completion-p3-causal-matrix-v3-20260902`
- P4 12-workload: `/home/sslab/TensorRT-Edge-LLM/.local/completion-p4-natural-shadow-12x1-20260902`
- P4 load: `/home/sslab/TensorRT-Edge-LLM/.local/completion-p4-natural-shadow-load-3x3-20260902`
- P5 active: `/home/sslab/TensorRT-Edge-LLM/.local/completion-p5-active-load48.8-3x-20260902`
- P5/P7 gate: `completion-p5-active-load48.8-3x-20260902/p5-p7-promotion.json`
- P6 matrix: `/home/sslab/TensorRT-Edge-LLM/.local/completion-p6-policy-matrix-20260902/policy-matrix.json`
- P6 CSV: `/home/sslab/TensorRT-Edge-LLM/.local/completion-p6-policy-matrix-20260902/policy-matrix.csv`
- instrumented binary SHA256: `57462e80fe98fb34c3e26ab7116d01144d004f270c6276991dc7880ce0c77684`

## 13. 다음 계획

P0–P7 뒤의 우선순위는 다음과 같다.

1. **P0 performance recovery audit**: frozen 41.8 req/s와 instrumented 34.4 req/s의
   차이를 binary/source/trace/client dispatch delay로 분해한다. Completion policy를 더
   튜닝하기 전에 mechanism performance parity를 회복한다.
2. **CUDA-gated directional injection**: host sleep을 제거하고 incumbent stream의 CUDA
   event/barrier에 newcomer stream을 연결해 actual 0/25/50/75/90% start를 만든다.
3. **Counterfactual replay 개선**: 선택된 action만으로 regret 0이 되는 문제를 해결한다.
   동일 frozen snapshot을 serial/overlap 양쪽에 재생하는 bounded probe가 필요하다.
4. **Canonical VLM ordering**: `(graph bucket, context bucket, stable request ID)` 기반
   ordering과 동일 binding shape로 cross-policy exact greedy identity를 복구한다.
5. **Coverage 보정**: D→P newcomer 74.6% under-coverage 원인을 direction별 sample
   scarcity와 residual feature scale로 나눠 분석한다. Workload별 threshold는 추가하지 않는다.
6. **다시 P5**: 위 네 gate를 고친 뒤 동일 48.8 trace에서 off/active 최소 5회,
   vLLM frozen anchor와 비교한다.
7. **H2는 계속 조건부**: P5 통과 후 natural median regret가 100µs 이상일 때만 켠다.

새 정책은 workload 이름, 외부 cost registry, shape-specific rule table을 사용하지 않는다.
최종 판단 기준은 profile-free라는 설명이 아니라 exact correctness, SLO goodput,
action fidelity, calibrated uncertainty가 같은 configuration에서 동시에 성립하는가이다.
