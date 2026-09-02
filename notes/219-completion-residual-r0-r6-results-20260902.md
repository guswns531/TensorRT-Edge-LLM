# Completion/Residual R0--R6 구현 및 결정 결과

> **R7 update:** 이 문서의 R2 prepared-execution blocker는 이후 해결됐다.
> mapped semaphore, staged vision preparation, exact P/D dispatch preamble로 6-direction
> 30-cell fidelity를 검증한 최신 결과는
> `notes/220-prepared-directional-execution-r7-results-20260902.md`를 기준으로 한다.
> 아래 R0--R6 내용은 실패 원인과 branch decision을 보존하는 historical record다.

날짜: 2026-09-02  
브랜치: `codex/v010-phase-forward-port`  
대상: `nvidia/Cosmos-Reason2-2B`, RTX 3080 10 GiB, TensorRT 11.0/CUDA 13.3

## 1. 최종 결론

`notes/218-completion-residual-decisive-validation-plan-20260902.md`의 R0--R6를
사전 정의한 gate 순서대로 실행했다. 최종 판정은 **Branch C**다.

> Frozen Current를 production policy로 유지한다. Completion/residual predictor와
> CUDA directional gate는 measurement/characterization 기능으로만 남기며 scheduling
> authority로 승격하지 않는다.

이 결정은 새 정책의 일부 raw-throughput 승리만 골라서 판단한 것이 아니다.

- Current는 12개 workload에서 유일하게 exact token identity `12/12`를 지켰다.
- Current의 macro joint-SLO goodput은 `14.216 req/s`로 모든 비교 정책 중 가장 높았다.
- Always-overlap은 macro raw throughput을 Current보다 높였지만 macro joint-SLO goodput과
  output identity를 잃었다.
- Completion-active는 Current보다 macro raw throughput과 macro SLO goodput이 모두 낮았다.
- True counterfactual replay에서 동일 snapshot의 action disagreement가 없었으므로
  completion/residual authority의 decision-regret 개선을 입증하지 못했다.
- CUDA gate로 의도한 residual progress를 실제 GPU kernel start에 재현하는 R2 fidelity
  gate도 통과하지 못했다.

따라서 completion/residual information이 연구 계측으로 유용하다는 것과 production
authority로 유효하다는 것을 분리했다. 후자는 현재 evidence로 지지되지 않는다.

## 2. 단계별 판정

| 단계 | 구현/실험 | 결과 | Gate |
|---|---|---|---|
| R0 | Current minimal parity, 48.8 req/s 5회 | median `40.794 req/s`, frozen `41.798` 대비 `-2.40%`, hash 5/5 동일 | 통과 |
| R1 | `minimal/research/counterfactual` telemetry level과 side-channel | research median `39.646 req/s`, minimal 대비 `-2.81%`; event 약 60 MiB에서 3.3 MiB/run으로 감소 | 통과 |
| R2 | GPU-side six-direction directional gate | P→D 50% target이 actual `91.96%`; target error `41.96pp` | 실패, full matrix 중단 |
| R3 | deterministic snapshot fork/true replay analyzer | Current↔active matched 3, action disagreement 0, fidelity failure 0 | authority evidence 실패 |
| R4 | direction calibration/canonical row ordering | P→D held-out 1개뿐; 6 direction coverage 부족. 관련 unit test 14/14 통과 | authority coverage 실패 |
| R5 | predeclared branch selection | R2/R3/R4와 12-workload 결과에 따라 Branch C | Current 유지 |
| R6 | 동일 runtime 12-workload 정책 matrix와 load evidence 재집계 | Current가 macro SLO goodput 및 identity 최상 | 완료 |

R2 이후 모든 `30 cells x 5 repeats`를 억지로 실행하지 않은 것은 누락이 아니다. 계획서의
acceptance gate는 actual progress가 target median `±5pp`에 들어오는 cell만 채택하도록
정했다. 첫 대표 cell이 이 gate를 크게 벗어났으므로 원인을 숨긴 채 나머지 수치를
생성하지 않고, R3--R6의 promotion 판단으로 진행했다.

## 3. R0 — Frozen Current parity

### 3.1 결과

Latest binary의 completion authority와 상세 telemetry를 끈 minimal mode를 48.8 req/s에서
5회 실행했다.

```text
40.9296, 40.8808, 40.7387, 40.7943, 40.6309 req/s
median = 40.7943 req/s
frozen Current = 41.798 req/s
difference = -2.40%
```

모든 run의 greedy token hash는 다음과 같았다.

```text
23305d8cc53d1f4a191ae49a67edd7b6e81ca28971a497a2bd72056dcd6cb479
```

따라서 이전 instrumented run의 약 33--35 req/s는 completion policy 자체의 필연적 비용이
아니었다. 상세 per-decision JSON serialization과 hot-path request-vector 복제가 주된
원인이었다.

### 3.2 산출물

- `.local/completion-r0-parity-20260902/current-minimal`

## 4. R1 — Low-overhead observation plane

### 4.1 구현

Telemetry를 다음 세 level로 분리했다.

```text
full            detailed debugging and request timeline
research        compact decision/dispatch/completion side channel
counterfactual  deterministic snapshot/cohort data for paired replay
```

환경 변수는 다음과 같다.

```text
TRT_EDGELLM_PHASE_TELEMETRY_LEVEL
TRT_EDGELLM_PHASE_TELEMETRY_PATH
```

Research level은 policy semantics를 바꾸지 않고 다음 비용을 제거한다.

- detailed dispatch metric materialization
- scheduler hot path의 전체 request ID vector 복제
- gateway stdout의 대형 per-event JSON
- event별 flush/formatting

Compact event는 별도 side-channel JSONL에 기록하고 분석기는 이 파일을 직접 읽는다.

### 4.2 결과

최종 3회는 `39.6464`, `39.6995`, `39.3067 req/s`, median `39.6464 req/s`였다.
R0 minimal median 대비 `-2.81%`로 3% gate 안에 들어왔고 token hash도 동일했다.

이벤트 파일은 약 `60 MiB/run`에서 약 `3.3 MiB/run`, gateway log는 약 `12 KiB`로
줄었다.

### 4.3 산출물

- `.local/completion-r1-compact-snapshot-v3-20260902/current`

## 5. R2 — CUDA directional gate와 decision realization

### 5.1 구현 구조

Host `sleep`으로 newcomer submit을 늦추지 않고 GPU dependency로 방향과 시작 위치를
통제하도록 다음 경로를 추가했다.

```text
incumbent stream
  record(start_event) ───────── TensorRT kernels ───────── record(end)
               │
               ▼
high-priority gate stream
  wait(start_event) ── calibrated GPU delay ── record(gate_done)
                                                │
                                                ▼
newcomer E/P/D stream
  host prepare/enqueue ── wait(gate_done) ── TensorRT kernels
```

새 CUDA mechanism은 `PhaseCudaDirectionalGate`이며 다음 stream/event가 실제 runtime
component를 따라 노출된다.

- P/D: `PhaseDispatchWorker` → `IndependentPhaseCoordinator` → `IndependentPhaseAsyncServer`
- E: `PhaseVisionAdapter`의 encoder stream 및 encoder-start event
- E↔P/D: coordinator gate
- P↔D: worker gate

처음 실험한 mapped-memory wait-value gate는 progress guarantee가 없어 deadlock되었고 즉시
폐기했다. 최종 구현은 CUDA event chain만 사용한다.

### 5.2 대표 결과

P→D actual progress target 50%에서:

```text
requested progress = 50.00%
observed progress  = 91.96%
target error       = 41.96 percentage points
acceptance         = fail
```

### 5.3 해석

Gate dependency 자체는 GPU-side지만, TensorRT enqueue가 newcomer kernel을 GPU queue에
materialize하는 host 시간이 이미 incumbent 실행의 큰 부분을 소비했다. 즉 gate release를
50%로 설정해도 newcomer kernel은 그 시점에 실행 준비가 되어 있지 않았고 실제 시작은
약 92%까지 밀렸다.

```text
scheduler decision
        │
        ▼
TRT binding / enqueue / kernel materialization
        │              substantial host time
        ▼
GPU gate eligibility
        │
        ▼
actual newcomer kernel start
```

이 결과는 중요한 architecture evidence다.

> Decision quality와 decision realization은 별개의 병목이다. Independent context가 있어도
> newcomer work가 GPU queue에 일찍 materialize되지 않으면 arbitrary residual overlap을
> 실현할 수 없다.

따라서 현재 runtime에서 residual fraction을 강한 scheduling authority로 쓰는 것은
부정확하다. 다음 재시도 전에는 TensorRT enqueue preparation과 GPU launch dependency를
분리하거나 prepared execution을 미리 queue할 수 있는 mechanism이 필요하다.

### 5.4 산출물

- `.local/completion-r2-cuda-gate-smoke-v4-20260902`

## 6. R3 — True counterfactual replay

### 6.1 구현

`analyze_true_counterfactual_replay.py`를 추가했다. 다음 항목이 동일한 branch execution만
pairing한다.

- deterministic `snapshot_signature`
- exact request ID vector
- selected cohort/token work
- plan/dispatch/completion correlation

선택한 action의 GPU horizon과 action fidelity를 branch별로 비교한다. 기존의 selected-action
observation을 임의 counterfactual로 재사용하지 않는다.

### 6.2 결과

| Pair | Matched | Action disagreement | Fidelity failure | Median horizon delta |
|---|---:|---:|---:|---:|
| Current vs completion-active | 3 | 0 | 0 | `-61.523 us` |
| Current vs serial | 2 | 0 | 0 | `+8.789 us` |

Matched state에서 action disagreement가 0이므로 horizon의 작은 차이는 같은 action의 run
variance이지 policy regret 개선 증거가 아니다. 따라서 completion/residual이 실제 decision을
바꾸고 더 나은 결과를 냈다는 R3 gate는 통과하지 못했다.

### 6.3 산출물

- `.local/completion-r3-counterfactual-balanced-20260902.json`
- `.local/completion-r3-counterfactual-balanced-serial-20260902.json`

## 7. R4 — Calibration과 canonical correctness

### 7.1 Calibration analyzer

`analyze_completion_calibration.py`는 direction별 chronological held-out sample에서 다음을
계산한다.

- completion absolute error mean/median/p95
- uncertainty interval empirical coverage
- false-safe count
- direction/sample coverage

현재 유효 sample은 P→D 3개뿐이며 held-out은 1개다.

```text
completion components = 2
MAE / median           = 1301.834 us
p95                    = 1989.587 us
empirical coverage     = 1.0
false-safe             = 0
```

Coverage 수치는 2 component에 불과하므로 calibration 성공을 뜻하지 않는다. 특히 D→P를
포함한 6 direction coverage가 없으므로 authority gate는 실패다.

### 7.2 Canonical ordering

Batch row ordering은 다음 stable key로 canonicalize한다.

```text
(token_count / 128 context bucket, stable request ID)
```

기존 row affinity가 있으면 유지하고, unassigned row만 canonicalize한다. Prior affinity가
없는 batch는 전체를 canonicalize하며 prefill도 dispatch 전에 같은 규칙을 적용한다.

이 변경은 scheduling policy와 numerical determinism을 분리한다. 동일 legal cohort는 policy가
달라도 동일 row order/binding shape로 실행될 가능성을 높인다.

관련 unit test `18/18`이 통과했다.

- `PhaseRuntimeCostTrackerTest`: 5
- `PhaseDispatchWorkerTest`: 6
- `PhaseUnifiedEventTest`: 7
- 신규 canonical row ordering case 포함

48.8 단일 regression run은 `39.416 req/s`였고 R0/R1 variance band 안이었다. Token hash는
동일했다. 단일 run의 tail 수치는 promotion 근거가 아니라 smoke evidence로만 사용한다.

### 7.3 산출물

- `.local/completion-r4-calibration-20260902.json`
- `.local/completion-r4-canonical-regression-20260902/current`

## 8. R5 — Authority 선택

사전 정의된 gate를 적용하면 다음과 같다.

```text
R2 actual residual placement fidelity  fail
R3 causal action disagreement/regret   absent
R4 six-direction calibration coverage  insufficient
12-workload macro SLO goodput           Current wins
12-workload exact token identity        Current only 12/12
```

따라서 full completion authority나 Current 위의 conservative veto/refinement를 production에
승격할 근거가 없다. 새 threshold나 workload별 rule을 추가해 gate를 우회하지 않았다.

Branch C의 의미는 completion code를 삭제한다는 것이 아니다.

- production default: frozen profile-free Current
- research telemetry: enabled on demand
- completion/residual model: shadow measurement and offline analysis
- CUDA directional gate: controlled characterization only
- policy tuning: 중단

## 9. R6 — 12-workload policy matrix

동일 runtime/trace의 기존 frozen policy matrix를 재집계했다. R5에서 새 production policy를
승격하지 않았으므로 모든 policy를 최신 telemetry patch 위에서 불필요하게 재실행하지 않았다.
R0/R1 및 latest canonical regression으로 production-disabled mechanism이 Current semantics와
token hash를 바꾸지 않는 것을 확인했다.

| Policy | Macro throughput (req/s) | Workload-macro % vs Current | Macro joint-SLO goodput (req/s) | Exact identity |
|---|---:|---:|---:|---:|
| **Current** | `24.116` | `0.00%` | **`14.216`** | **12/12** |
| Always overlap | `24.482` | `+3.97%` | `14.053` | 7/12 |
| Completion active | `23.665` | `+0.33%` | `13.900` | 7/12 |
| Completion new-work-only | `23.631` | `-1.05%` | `13.880` | 8/12 |
| Completion no-residual | `23.672` | `+0.09%` | `13.636` | 7/12 |
| Completion no-uncertainty | `23.935` | `+1.27%` | `13.568` | 9/12 |
| Immediate cost | `23.772` | `+1.00%` | `13.797` | 7/12 |
| Serial | `23.843` | `+2.01%` | `12.879` | 7/12 |

두 가지 macro throughput 열은 aggregation 방식이 다르다. 첫 번째는 workload throughput의
산술 평균이고, 두 번째는 각 workload에서 Current 대비 percentage를 구한 뒤 평균한 값이다.
따라서 Completion active의 absolute macro throughput이 Current보다 작지만 workload-macro
percentage는 소폭 양수가 될 수 있다. Promotion metric은 SLO goodput과 correctness다.

### 9.1 해석

- Always-overlap의 raw throughput 승리는 overlap opportunity가 실재함을 보여준다.
- 그러나 SLO goodput과 exact identity가 나빠져 production 최적점은 아니다.
- Completion-active는 residual signal의 존재만으로 Current보다 나은 authority가 되지
  못했다.
- Uncertainty 제거는 일부 raw throughput을 올리지만 macro goodput을 더 악화시켰다.
- Serial은 raw throughput 평균이 높아 보이는 workload가 있어도 joint-SLO goodput이 가장
  낮다. 따라서 raw throughput만으로 policy를 평가하면 잘못된 결론을 낸다.

### 9.2 Load sweep과 vLLM anchor

Frozen Current 5-run load sweep에서:

| Offered load | Result |
|---:|---|
| 39 req/s | median token throughput `2994.73 token/s`, joint-SLO 100% |
| 48.8 req/s | median request goodput `41.798 req/s`, joint-SLO 100% |
| 97.5 req/s | raw 약 `47.7 req/s`, pass median 약 `41%`, goodput 약 `19.5 req/s` |

48.8의 frozen vLLM은 `40.901 req/s`이므로 frozen Current가 `+2.19%` 높다. Current의
TTFT/TPOT/E2E mean/p95도 모두 더 낮았다. 다만 R0/R1 latest-binary parity는 이 frozen
최고점보다 2.4--5.1% 낮으므로 논문 headline에서는 frozen result와 latest-instrumented
result를 혼용하지 않는다.

Frozen canonical 12 workload에서는 Current token throughput이 vLLM보다 12/12 높았고
범위는 `+0.94%`에서 `+25.87%`였다. 이 비교의 세부 조건과 idle/overlap 해석은
`notes/215-final-profile-free-contextual-controller-20260902.md`에 고정돼 있다.

### 9.3 산출물

- `.local/completion-r6-final-20260902/policy-matrix.json`
- `.local/completion-r6-final-20260902/policy-matrix.csv`
- `.local/final-contextual-provenance-load-3x5-20260902`

## 10. 최종 runtime architecture

```text
HTTP request arrival
       │
       ▼
Request DAG + stable KV/vision ownership
       │
       ▼
Global E/P/D ready snapshot
       │
       ├── deterministic feasibility
       │     dependency / TRT shape / single-inflight / memory
       │
       ├── production Current selector
       │     SLO safety / contextual value / bounded WAIT
       │
       └── research shadow plane
             compact telemetry
             completion/residual prediction
             exact snapshot signature
             counterfactual/calibration analysis
       │
       ▼
Explicit E/P/D or pair action lease
       │
       ▼
Independent TensorRT contexts, shared CUDA context
       │
       ├── E stream/context
       ├── P stream/context
       ├── D stream/context
       └── Copy stream
       │
       ▼
CUDA start/end/completion events
       │
       ├── production online cost update
       └── research side-channel update
```

Research directional injection은 이 production path 옆에만 존재한다.

```text
directional experiment request
       │
       ▼
incumbent start event ── gate stream ── newcomer wait
       │
       └── actual start progress/fidelity measurement
```

Default environment에서는 gate와 completion authority가 모두 꺼져 있다.

## 11. 무엇이 완료됐고 무엇은 완료되지 않았는가

### 완료

- latest binary frozen parity audit
- low-overhead research telemetry
- E/P/D stream 및 start-event 연결
- GPU-side directional gate implementation
- exact snapshot/cohort counterfactual analyzer
- direction calibration analyzer
- canonical row ordering
- relevant C++ unit test와 Python syntax test
- predeclared R5 branch decision
- 12-workload/load/vLLM evidence 재집계

### 의도적으로 승격하지 않은 것

- completion-aware full policy authority
- Current 위 completion veto/refinement
- residual-fraction-specific overlap rule
- workload별 completion threshold
- external cost registry/persisted learning/TTL

### R2에서 남은 mechanism blocker

현재 TensorRT execution은 newcomer host enqueue/materialization이 늦어서 arbitrary residual
progress target을 만들지 못한다. 다음에 residual scheduling 연구를 다시 열려면 우선 다음
중 하나가 필요하다.

1. prepared enqueue와 release dependency를 분리할 수 있는 runtime path
2. graph/execution instance를 미리 materialize한 뒤 CUDA event로 release하는 path
3. TensorRT enqueue를 phase worker의 critical path 밖에서 준비하는 double-buffered binding
4. action decision boundary를 kernel materialization 이전으로 이동

이 blocker를 해결하기 전에는 completion predictor의 score나 threshold를 더 튜닝하지 않는다.

## 12. 다음 권장 순서

R6 이후 production 개선은 completion policy가 아니라 다음 순서가 타당하다.

1. Current를 release candidate로 freeze하고 12-workload correctness gate를 유지한다.
2. R7에서는 Current만 Nsight profiling하여 host enqueue/materialization gap을 분해한다.
3. P/D/E 각각에 대해 decision → TRT enqueue → first kernel start를 측정한다.
4. Prepared execution이 실제로 first-kernel delay를 줄일 수 있을 때만 R2를 다시 연다.
5. R2 target fidelity가 통과한 뒤에만 six-direction causal matrix와 completion authority를
   재평가한다.
6. 논문에서는 completion controller의 성능 향상을 주장하지 않고, transition-aware
   measurement substrate와 decision-realization limitation을 정직하게 보고한다.

## 13. 재현 및 검증 요약

사용한 핵심 검증은 다음과 같다.

```text
Python analyzers:
  python3 -m py_compile \
    benchmarks/phase_serving/analyze_completion_calibration.py \
    benchmarks/phase_serving/analyze_true_counterfactual_replay.py

C++ unit tests:
  PhaseRuntimeCostTrackerTest.*
  PhaseDispatchWorkerTest.*
  PhaseUnifiedEventTest.*

Runtime smoke:
  48.8 req/s minimal/research/canonical regression
  P→D CUDA directional 50% target
```

결과 artifact는 repository의 `.local`에 보존하며 commit에는 source와 문서만 포함한다.
