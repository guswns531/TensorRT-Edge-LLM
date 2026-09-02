# Completion/residual scheduling 결정 검증 계획

날짜: 2026-09-02  
브랜치: `codex/v010-phase-forward-port`  
대상: `nvidia/Cosmos-Reason2-2B`, RTX 3080 10 GiB, TensorRT 11.0/CUDA 13.3

## 1. 문서 목적

이 문서는 P0--P7 결과와 후속 검토 의견을 바탕으로, completion-vector 및 residual-state
scheduling 연구를 계속 강하게 주장할지, Current controller의 제한된 veto/refinement로
축소할지를 결정하기 위한 실행 계획이다.

현재 상태를 가장 정확하게 요약하면 다음과 같다.

> Completion-vector architecture와 action-faithful measurement substrate는 구현됐다.
> 그러나 completion/residual information이 Current보다 나은 scheduling authority라는
> 주장은 아직 증명되지 않았다.

따라서 다음 단계의 목적은 새 heuristic을 추가해 특정 workload 숫자를 높이는 것이
아니다. 다음 세 질문에 causal evidence로 답하는 것이다.

1. 계측과 completion mechanism 자체의 비용을 제거한 뒤에도 새 policy의 효과가 남는가?
2. 동일 실행 상태에서 residual progress만 달라질 때 최선의 action이 실제로 달라지는가?
3. 그 정보를 사용한 selector가 Current/immediate/no-residual보다 decision regret와
   request-level SLO goodput을 줄이는가?

관련 기준 문서:

- `notes/215-final-profile-free-contextual-controller-20260902.md`: frozen Current
- `notes/216-research-statement-current-gap-and-promotion-plan-20260902.md`: 연구 목표와 Current의 차이
- `notes/217-completion-projection-p0-p7-results-20260902.md`: P0--P7 실제 결과

## 2. 현재 evidence와 남은 blocker

### 2.1 이미 확인된 것

- Independent E/P/D TensorRT context, stable KV/vision ownership, bounded action lease가
  실제 runtime에 존재한다.
- 96,302 executions에서 planned action과 dispatch/completion의 fidelity violation은 0이었다.
- Residual timing에 따라 D→P와 P→D가 각각 최대 약 17.23%, 10.93% 변했고,
  P→E는 최대 약 124.43%의 harmful effect를 보였다.
- Always-overlap은 macro raw throughput을 높였지만 Current보다 macro SLO goodput이 낮았다.
  즉 overlap capability와 profitability는 다르며 raw throughput maximization은 충분하지 않다.
- New-work-only는 decode-heavy와 long-prefill에서 completion-active보다 크게 낮았다.
  일부 interference-sensitive state에서는 incumbent completion을 함께 봐야 한다.
- Uncertainty를 제거하면 일부 raw throughput은 증가하지만 macro goodput과 tail이 나빠졌다.

### 2.2 아직 증명되지 않은 것

- 6 direction 모두에서 실제 GPU start progress가 통제된 residual sensitivity
- 선택하지 않은 action을 실제로 실행한 counterfactual value
- Completion+residual이 Current/immediate/no-residual보다 낮은 decision regret를 갖는다는 것
- Completion-active가 12-workload 전체에서 Current의 robustness를 보존한다는 것
- D→P newcomer를 포함한 direction별 calibration coverage
- Scheduling 순서가 달라도 equivalent inference임을 보장하는 cross-policy correctness

### 2.3 현재 수치가 경고하는 것

- Frozen P0 48.8 trace는 41.798 req/s였지만 instrumented runtime은 약 33--35 req/s였다.
  원인을 아직 instrumentation overhead라고 단정할 수는 없지만, 약 17% 차이를 해소하지
  않고 final system을 평가할 수 없다.
- Natural trace의 selector disagreement는 30--58%였으나 replay regret median/p95가 0 µs로
  퇴화했다. 현재 replay는 selected-action observation을 counterfactual처럼 재사용한다.
- P5 completion-active는 authority-off보다 paired goodput을 3/3 개선했지만, 40 req/s,
  99% joint-SLO, direction별 coverage gate를 통과하지 못했다.
- P6 macro joint-SLO goodput은 Current 14.22, completion-active 13.90이었다.
- `w/o residual`은 13.64로 active보다 약 1.9% 낮았지만 central claim을 받치기에는 반복성과
  causal attribution이 부족하다.

## 3. 연구 원칙과 고정 제약

### 3.1 Policy-free correctness

다음은 learned model이나 실험 mode가 바꿀 수 없다.

- request DAG dependency
- TensorRT optimization profile 및 binding legality
- context별 single-inflight
- stable indexed KV/page lease와 vision lease
- GPU consumer completion 전 reclaim 금지
- deterministic admission/memory feasibility
- planned action과 actual outstanding set의 fidelity

### 3.2 Profile-free policy

Policy에는 다음을 넣지 않는다.

- `short`, `balanced`, `vision-heavy` 같은 workload 이름
- workload별 threshold 또는 mode
- `P8/D64`, `E1/P8` 같은 exact shape rule
- 외부 cost registry, persisted learned state, TTL
- 미래의 미도착 request 예측

Policy input은 current ready state, request slack, current outstanding execution,
stable ownership state, process-local CUDA observation뿐이다.

### 3.3 평가 중 고정할 것

- Cosmos checkpoint/engine/hash
- max P/D/E batch 8/64/8
- fixed prefill chunk 128
- indexed-paged KV, stable slots/pages
- request trace와 arrival timestamp
- tokenizer/sampling seed와 greedy contract
- GPU clock/temperature/background load 기록

P chunk 128은 이번 연구 질문과 독립적이므로 변경하지 않는다. KV pool 크기나 page layout도
memory-pressure 전용 실험이 아니면 변경하지 않는다.

### 3.4 Mechanism과 policy의 분리

```text
Deterministic mechanism
  legal action / ownership / binding / single-inflight
                    │
                    ▼
Current observable snapshot
                    │
                    ▼
Execution models
  exact timing audit + immediate advantage + completion vector
                    │
                    ▼
Policy authority
  Current / full completion / conservative refinement
```

정확한 execution model이 scheduling authority일 필요는 없다. Completion predictor가
유용하더라도 Current 전체를 대체하지 못하면 제한된 veto/refinement로 사용할 수 있다.

## 4. 전체 실행 순서

```text
R0  Freeze and parity audit
 │
 ▼
R1  Low-overhead observation plane
 │
 ▼
R2  CUDA-gated six-direction characterization
 │
 ▼
R3  Deterministic snapshot fork and true counterfactual replay
 │
 ├── residual/completion value 없음 ──► measurement-only로 강등
 │
 ▼
R4  Calibration and canonical correctness
 │
 ▼
R5  Authority comparison
 │
 ├── full authority가 Current 지배 ──► Full completion controller
 ├── 일부 state에서만 유효 ─────────► Current + conservative veto/refinement
 └── 안정적 이득 없음 ──────────────► Current 유지, characterization contribution
 │
 ▼
R6  12-workload/load-sweep final evaluation
 │
 ▼
R7  Architectural profiling, memory-normalized result, paper freeze
```

각 단계는 앞 단계 gate를 통과한 뒤 진행한다. Gate 실패를 score/threshold 튜닝으로
우회하지 않는다.

## 5. R0 — Frozen baseline과 performance parity audit

### 5.1 가설

41.798 req/s와 33--35 req/s의 차이는 policy 자체가 아니라 source/binary, event logging,
client arrival, warmup, GPU state 또는 host hot-path 중 하나에서 발생했다.

### 5.2 비교 variant

동일 source tree와 동일 compiler/container에서 기능을 단계적으로 켠다.

1. `Current-minimal`: frozen Current semantics, completion code와 event emission off
2. `Current-events`: unified action event만 on
3. `Current-completion-shadow`: predictor/update on, authority off
4. `Current-completion-active`: authority on

가능하면 compile-time 제거와 runtime disable을 모두 측정해 dormant code cost와 event
serialization cost를 분리한다.

### 5.3 측정

- 48.8 req/s trace, 각 variant warmup 후 5회
- raw throughput, SLO goodput, TTFT/TPOT/E2E mean/p95
- scheduler decision mean/p95/max
- candidate generation, feature extraction, model prediction, event serialization 시간
- client offered/actual dispatch rate와 arrival drift
- GPU E/P/D/Copy busy mask, idle ratio, host launch gap
- peak VRAM, CPU utilization, log bytes/event 수
- source/binary/engine/trace/hash와 driver/clock/temperature

### 5.4 Gate

- `Current-minimal` median throughput이 frozen P0의 3% 이내
- `Current-events`와 `Current-completion-shadow`가 `Current-minimal` 대비 각각 3% 이내
- joint-SLO pass-rate 회귀 1 percentage point 이내
- token identity 및 action fidelity 100%
- scheduler p95가 기존 Current 대비 5% 또는 50 µs 중 큰 값 이내

### 5.5 실패 시 처리

- client/trace contract가 다르면 P0 manifest를 새 same-binary baseline으로 다시 freeze한다.
- event serialization이 원인이면 ring buffer와 run-end bulk flush로 변경한다.
- feature/model update가 원인이면 snapshot aggregate 재사용, update batching, shadow sampling을
  적용한다.
- TensorRT/engine 차이면 policy 실험을 중단하고 engine lifecycle parity부터 복구한다.

R0를 통과하기 전에는 completion scoring을 튜닝하지 않는다.

## 6. R1 — Low-overhead observation plane

### 6.1 목표

연구에 필요한 event를 유지하면서 production hot path의 synchronization, formatting,
allocation을 제거한다.

### 6.2 구현 범위

- decision/dispatch/completion은 고정 크기 POD record로 기록
- per-event JSON formatting과 file flush는 worker/run-end로 이동
- execution ID와 plan ID는 preallocated counter 사용
- model update는 CUDA completion을 기다리지 않고 완료 record consumer가 수행
- queue aggregate와 protected slack은 scheduler snapshot에서 한 번만 계산
- event level은 `minimal`, `research`, `debug`로 나누되 policy semantics는 동일

### 6.3 Gate

- `minimal`과 `research`가 동일 token/action trace를 생성
- research event loss 0, dispatch/completion mismatch 0
- research mode throughput 회귀 3% 이내
- decision p95 목표 50 µs는 별도 이상적 목표로 유지하되, 현재 runtime의 실측 baseline보다
  나쁜 임의 숫자를 강제하지 않는다. Primary gate는 paired relative overhead다.

## 7. R2 — CUDA-gated six-direction residual characterization

### 7.1 목표

Host sleep이 아니라 newcomer stream의 실제 GPU 시작을 통제해, 동일 pair/cohort에서
incumbent progress만 바꾼 causal matrix를 완성한다.

### 7.2 Gate mechanism

```text
incumbent stream
  start marker ─────────── incumbent TRT work ─────────── end marker

newcomer stream
  prepare/enqueue ── wait(gate) ── newcomer TRT work
                           ▲
                  calibrated gate release
```

Newcomer의 host preparation과 TensorRT enqueue는 gate 앞에서 끝낸다. Gate release는
incumbent start marker가 GPU에서 관측된 뒤 isolated duration의 목표 fraction에 맞춰
수행한다. 가능하면 기존 kernel-group boundary event를 사용하고, 정확한 내부 marker가
없는 action은 post-hoc actual start progress가 tolerance에 들어온 cell만 채택한다.

### 7.3 Matrix

- directions: P→D, D→P, P→E, E→P, E→D, D→E
- actual progress targets: 0/25/50/75/90%
- 각 cell 최소 5 valid repeats
- 동일 request set, cohort, engine, seed, ownership state
- serial reference도 각 cell과 같은 run block에서 측정

### 7.4 Acceptance

- actual start progress median이 target ±5 percentage points, 개별 repeat ±10pp
- invalid/late cell은 negative effect로 해석하지 않고 재측정
- direction별 incumbent completion shift, newcomer completion, equal-work makespan,
  request milestone shift를 보고
- material effect는 상대 3% 이상 또는 100 µs 이상 중 하나를 만족하고 95% confidence
  interval이 0을 넘는 경우로 정의

### 7.5 산출물

- six-direction progress-response curve
- material/non-material direction 분류
- residual feature를 제거하면 구분할 수 없는 matched-state pair
- Figure 후보: 같은 action/cohort, 다른 progress, 반대 또는 크게 다른 outcome

## 8. R3 — True counterfactual snapshot fork/replay

### 8.1 현재 replay의 문제

선택된 action의 observation을 다른 action의 결과처럼 재사용하므로 disagreement가 높아도
regret가 0으로 퇴화한다. 이것은 policy quality evidence가 아니다.

### 8.2 Snapshot 정의

Full GPU memory를 복제하지 않는다. 동일 deterministic trace prefix를 두 번 재생해 같은
decision frontier를 복원한다.

Snapshot hash에는 다음을 포함한다.

- ready E/P/D request ID와 phase metadata
- stable KV slot/page ID, per-request KV length
- vision lease ID와 lifetime state
- outstanding phase/action, dispatch age, graph/profile bucket
- request deadlines/slack와 sampling state
- engine/checkpoint/tokenizer/seed

### 8.3 Fork protocol

```text
identical trace prefix → frozen frontier S
                         ├─ branch A: serial/WAIT/current action
                         └─ branch B: overlap/incremental ADD action

두 branch는 같은 logical work frontier까지 진행한 뒤 비교
```

비교 horizon은 두 branch가 동일한 request phase set을 완료하고 outstanding set이 다시
합류하는 최초 경계다. 단순히 첫 action duration만 비교하지 않는다.

### 8.4 대상 state

1. R2의 각 material direction/progress cell
2. Natural 12-workload에서 disagreement가 발생한 state
3. 39/48.8/97.5 load에서 SLO boundary에 가까운 state

먼저 controlled state에서 direction당 20 pair 이상, 이후 natural state에서 pair family당
최소 50 snapshot을 replay한다.

### 8.5 Counterfactual value와 regret

Value는 weighted reward 하나로 숨기지 않고 lexicographic하게 비교한다.

1. correctness/fidelity violation
2. protected TTFT/TPOT/E2E violation 수와 최대 violation
3. 동일 work frontier의 completion time
4. serial-equivalent work 대비 makespan compression
5. ownership bytes×lifetime 및 peak memory

Oracle action은 위 순서에서 더 나은 branch다. Policy regret는 policy-selected branch와
oracle branch의 protected milestone delay 및 equal-work makespan 차이로 각각 보고한다.

### 8.6 Gate

- paired branch snapshot hash와 input token identity 100%
- action fidelity 100%, ownership violation 0
- Current, immediate, completion, no-residual, new-work-only 모두 같은 snapshot set 평가
- completion+residual의 median 또는 p95 regret가 적어도 한 material direction에서
  Current/immediate/no-residual보다 낮고, paired 95% CI가 0을 넘음
- natural state에서도 selection change가 0이 아니고 improvement 방향과 일치

R3에서 이득이 없으면 completion-vector를 full authority 후보에서 제거한다. 그래도 R2의
phenomenon은 characterization 결과로 유지할 수 있다.

## 9. R4 — Calibration과 canonical correctness

### 9.1 Calibration

Direction별로 incumbent와 newcomer를 분리해 다음을 보고한다.

- sample count와 chronological train/calibration/test split
- MAE, median/p95 absolute error
- nominal 90/95% interval의 empirical coverage
- uncertainty bin별 actual error
- false-safe와 false-conservative
- co-launch와 residual augmentation의 coverage 차이

Material direction의 held-out coverage는 nominal-5pp 이상, false-safe 0을 gate로 둔다.
Non-material 또는 sample이 부족한 direction에는 모델 권한을 주지 않고 Current로 fallback한다.
Workload별 calibration threshold는 만들지 않는다.

### 9.2 Canonical ordering

- batch row order는 `(graph/profile bucket, context-length bucket, stable request ID)`로 고정
- 같은 logical candidate는 같은 padding/binding bucket을 사용
- donor/shared KV와 multi-image placement도 stable request order 유지
- sampling completion과 ready-queue commit 순서를 stable execution ID로 정렬

정책이 batch size 자체를 바꾸는 경우까지 억지로 동일 binding shape로 만들지는 않는다.
대신 두 correctness mode를 분리한다.

1. **Mechanism equivalence mode**: 같은 candidate/action에서 binding/order를 고정해 exact 비교
2. **Policy performance mode**: policy가 다른 batch를 만들 수 있으며, repeat determinism과
   logits/top-k closeness를 함께 기록

### 9.3 Correctness gate

- 동일 policy 반복 exact token identity 100%
- 동일 action/candidate의 cross-policy mechanism identity 100%
- 12-workload cross-policy greedy identity를 목표로 하되, FP16 batch-shape 차이로 남는 경우
  first-divergence logit delta/top-k agreement를 공개하고 ownership corruption과 분리
- CUDA sanitizer/memcheck, invalid slot/page, use-after-release 0

## 10. R5 — Authority 구조 결정

R0--R4 결과 후 세 구조 중 하나만 선택한다.

### 10.1 Branch A: Full completion authority

```text
feasible actions
      │
completion mean/uncertainty + request projection
      │
lexicographic SLO-safe selector
      │
dispatch
```

다음 조건을 모두 만족할 때만 선택한다.

- R3에서 Current/immediate/no-residual보다 유의하게 낮은 counterfactual regret
- 48.8 same-binary 5회에서 Current 이상 median SLO goodput
- 12-workload에서 macro goodput Current 이상, workload별 회귀 3% 이내
- residual-sensitive workload 최소 2개에서 유의한 개선
- calibration/correctness/performance parity gate 통과

### 10.2 Branch B: Current + conservative completion refinement

Full authority는 실패하지만 R2/R3에서 일부 residual state의 value가 확인되면 선택한다.

```text
Current contextual controller
          │
       choice a
          │
completion projection
  ├─ confident incumbent/newcomer SLO risk → veto 또는 serial/WAIT
  ├─ confident lower regret alternative    → bounded refinement
  └─ uncertain/non-material                → Current 그대로
```

Completion model은 action을 처음부터 전역 rank하지 않는다. Current가 만든 선택을
counterfactual evidence가 충분한 state에서만 수정한다. Materiality는 workload name이나
exact key가 아니라 continuous feature, calibrated confidence, projected SLO risk로 판단한다.

Branch B gate:

- Current action의 90% 이상은 그대로 유지하거나, 변경률 자체보다 false-positive veto 0을 보장
- residual-sensitive state에서 regret 감소
- 12-workload macro goodput Current 이상, workload별 회귀 1--3% 이내
- scheduler overhead와 coverage gate 통과

### 10.3 Branch C: Measurement/characterization only

R3에서 completion/residual이 Current보다 안정적으로 낮은 regret를 만들지 못하면 선택한다.

- Production은 frozen Current 유지
- Completion vector는 online telemetry와 interference characterization에만 사용
- 논문 핵심은 independent phase substrate, action fidelity, shape/residual-dependent
  interference, raw-throughput와 SLO-goodput의 차이로 재정의
- Full authority 성능 향상 주장은 제거

이 분기는 실패를 숨기지 않기 위한 사전 stop condition이다.

## 11. R6 — Final 12-workload 및 load-sweep 평가

### 11.1 비교 정책

- Frozen Current
- 선택된 final policy: Branch A 또는 B
- Serial
- Always-overlap
- Immediate-cost
- Completion w/o uncertainty
- Completion w/o residual
- New-work-only
- vLLM

vLLM은 workload/model/trace가 동일한 iterative run에서는 frozen 결과를 재사용한다. Final
논문 freeze 때 동일 HTTP client, arrival trace, SLO, model precision으로 fresh 5회 재실행한다.

### 11.2 Workload

기존 12개를 그대로 유지한다.

- short
- balanced
- decode-heavy
- long-prefill
- bimodal
- text-heavy
- mixed
- vision-heavy
- poisson
- wave-drain
- multi-image
- late-vision

추가로 39/48.8/97.5 req/s와 saturation 전후 세 점을 각 5회 측정한다. 정책을 workload별로
바꾸지 않으며 단일 configuration을 사용한다.

### 11.3 필수 metric

- raw request/token throughput
- SLO goodput와 TTFT/TPOT/E2E failure attribution
- TTFT/TPOT/E2E mean, median, p95
- E/P/D dispatch count, batch distribution, useful/padded work
- E/P/D/Copy stream busy mask와 GPU idle/overlap 비율
- decode completion→CPU visibility→sampling→ready commit→next enqueue breakdown
- action distribution, disagreement, counterfactual regret
- predictor MAE/coverage/false-safe
- peak VRAM, KV/vision ownership bytes×lifetime, page pressure/backpressure
- scheduler CPU mean/p95/max

### 11.4 Final promotion gate

- token/correctness/fidelity gate 전부 통과
- same-binary Current 대비 macro SLO goodput 비열등 또는 우월
- 어떤 workload도 throughput, TTFT p95, TPOT p95, E2E p95에서 3%를 넘는 설명되지 않은 회귀 없음
- residual-sensitive subset에서 paired 95% CI 기준 유의한 regret/goodput 개선
- 48.8에서 Current와 frozen/fresh vLLM 모두와 공정한 비교
- raw throughput 개선만 있고 SLO goodput이 나빠지면 promotion 실패

## 12. R7 — Architectural profiling과 논문 evidence freeze

### 12.1 Nsight selected points

전체 matrix를 Nsight로 돌리지 않는다. 다음 상태만 선택한다.

- R2에서 overlap이 가장 이로운 residual cell
- R2에서 가장 해로운 residual cell
- same pair/cohort에서 progress만 달라 sign이 바뀌는 cell
- 48.8 Current와 final policy
- residual-sensitive natural workload 한 개

측정 항목:

- kernel group start/end와 concurrent interval
- SM active, tensor utilization, DRAM/L2 bandwidth
- launch gap과 CUDA completion visibility
- actual outstanding mask와 scheduler-planned mask
- Copy stream이 E/P/D critical path를 막는 구간

### 12.2 Memory-normalized comparison

Independent context overhead를 숨기지 않는다.

- 같은 GPU memory budget에서 최대 SLO goodput
- 같은 SLO에서 필요한 memory
- engine/context workspace, KV pool, vision slab, temporary activation을 분리
- 동적 ownership이 reclaim한 bytes와 lifetime

이번 completion 연구에서 KV layout을 다시 변경하지 않는다. MemoryHorizon의 causal
counterexample가 충분할 때만 별도 contribution으로 승격한다.

### 12.3 Paper claim freeze

최종 claim은 선택된 branch에 따라 달라진다.

Branch A:

> Residual-aware completion projection can safely control independent phase execution and
> reduce decision regret and SLO-goodput loss without workload-specific profiles.

Branch B:

> A robust contextual scheduler benefits from calibrated residual-completion projection as
> a selective veto/refinement in interference-sensitive states.

Branch C:

> Independent phase actions exhibit residual-dependent interference that static overlap and
> myopic work-conserving policies cannot characterize, although full online authority is not
> consistently beneficial.

어느 branch든 결과보다 강한 claim을 쓰지 않는다.

## 13. Experiment/report contract

모든 결과 디렉터리는 다음을 포함한다.

- `manifest.json`: source/binary/engine/model/trace/environment hash
- `runs.csv`: per-run throughput와 latency/goodput
- `requests.csv`: per-request TTFT/TPOT/E2E와 SLO failure reason
- `actions.jsonl.gz`: decision/dispatch/completion correlation
- `gpu_intervals.csv`: E/P/D/Copy start/end와 mask timeline
- `calibration.json`: direction/head별 error와 coverage
- `counterfactual.csv`: snapshot hash, two actions, oracle, policy regret
- `summary.md`: gate 결과와 해석

표의 상대 변화는 반드시 기준을 이름으로 명시한다.

- absolute macro average
- mean per-workload relative change
- geometric mean normalized value

를 같은 열에서 혼용하지 않는다. 반복 결과는 median과 95% confidence interval을 함께
보고한다.

## 14. 구현 순서와 예상 변경 위치

| 순서 | 주요 위치 | 변경 내용 |
|---|---|---|
| R0 | `benchmarks/phase_serving/`, smoke wiring | layered binary/runtime audit와 manifest |
| R1 | `phaseUnifiedEvent`, coordinator, logger | POD ring buffer, deferred serialization |
| R2 | dispatch worker, controlled injection benchmark | pre-enqueued newcomer CUDA gate, actual progress validator |
| R3 | coordinator snapshot/audit, replay harness | deterministic frontier hash와 two-branch replay |
| R4 | contextual model, queue scheduler, sampler commit | calibration report, canonical ordering/correctness |
| R5 | global selector | full authority 또는 conservative veto/refinement |
| R6 | HTTP benchmark/analyzers | 12-workload/load sweep와 vLLM final comparison |
| R7 | profiling/analyzers/notes | Nsight, memory-normalized figures, paper evidence |

각 단계는 별도 signed commit으로 남긴다. 구현 commit과 대규모 결과 artifact는 분리한다.
`.local`의 raw trace와 engine은 commit하지 않고, manifest/CSV/요약만 repository에 남긴다.

## 15. 중단 조건

다음 경우 completion full authority 개발을 중단한다.

1. R0 parity를 복구한 뒤에도 mechanism overhead가 3%를 지속적으로 넘고 제거 경로가 없음
2. R2에서 actual progress를 통제해도 residual effect가 반복되지 않음
3. R3 true replay에서 completion+residual regret가 Current/no-residual과 구분되지 않음
4. R4 calibration이 material direction에서 nominal-5pp를 반복적으로 충족하지 못함
5. R5/R6에서 개선이 특정 workload 이름이나 exact shape rule 없이는 재현되지 않음

이 경우 Current를 production으로 유지하고, completion mechanism은 measurement substrate로
정리한다. 반대로 full authority가 Current를 이기지 못하더라도 일부 state에서 causal value가
확실하면 Branch B의 selective refinement를 먼저 검증한다.

## 16. 다음 즉시 실행할 작업

다음 구현 turn은 R0만 수행한다.

1. frozen P0과 현재 source/binary/engine/client manifest 차이 생성
2. `Current-minimal/events/completion-shadow/active` 네 variant wiring
3. 48.8 trace 5회씩 동일 run block 측정
4. scheduler/event/client/GPU gap breakdown
5. parity gate 판정 및 signed commit

R0 결과가 나오기 전에는 CUDA gate, predictor threshold, full authority score를 동시에
수정하지 않는다. 다음 연구 분기 전체가 신뢰할 수 있는 same-runtime baseline 위에 서도록
하는 것이 첫 번째 목표다.
