# Realized transition attribution and concrete completion boundary

## 1. 목적

이 단계는 H=2 formation-aware selector가 myopic selector와 다른 action을 고른 뒤 실제 GPU 실행이 어떻게 전개되는지 측정한다. 새 workload profile이나 shape별 정책 rule을 추가하지 않는다.

핵심 질문은 세 가지다.

1. 선택 변경 뒤 실제 dispatch sequence와 D cohort는 무엇인가?
2. 이미 outstanding인 sampling CUDA event가 이후 D-ready를 만들 때 현재 snapshot이 이를 보호하는가?
3. H=2가 예측한 이득이 실제 local service와 end-to-end 결과에 대응하는가?

## 2. 구현 구조

```text
H2/myopic selection differs
          |
          v
RealizedEpisodeStart
  snapshot/action/regret/budget
          |
          v
Global execution lease validation
          |
          +-- actual dispatch #1
          +-- actual dispatch #2
          +-- actual dispatch #3
          +-- actual dispatch #4
          |
          v
CUDA completion visible on host
          |
          v
PHASE_FORMATION_EPISODE JSON
  actual action sequence
  actual E/P/D rows
  first D service gap
  D completion visibility
  bounded horizon completion
```

주요 위치:

- `cpp/runtime/phase/policy/phaseFormationPlanner.h`
  - realized episode schema와 bounded tracker
  - concrete completion의 readiness, service cost, TPOT budget
- `cpp/runtime/scheduling/phaseFormationPlanner.cpp`
  - 최대 4-dispatch attribution
  - augmented lease plan ID remap
  - host-visible completion과 local D violation 집계
  - `H2 + first concrete completion boundary`
  - strict action-regret dominance helper
- `cpp/runtime/scheduling/independentPhaseAsyncServer.cpp`
  - 이미 제출된 decode sampling ticket을 cumulative completion preview로 노출
  - median/p95 sampling latency와 request-owned context/slot/SLO metadata 사용
- `cpp/runtime/scheduling/phaseThreeCoordinator.cpp`
  - H2 snapshot에 첫 concrete sampling event 연결
  - final selected action 자체가 myopic보다 strict improvement일 때만 변경 허용
- `examples/llm/llm_phase_context_smoke.cpp`
  - aggregate `PHASE_METRIC`
  - episode별 `PHASE_FORMATION_EPISODE` JSON
- root benchmark harness의 `scripts/cosmos_reason2/summarize_formation_episodes.py`
  - 여러 gateway log의 episode JSON/CSV 집계

## 3. Realized tracker correctness

tracker는 policy authority가 아니다. 이미 선택되어 성공적으로 launch된 execution lease만 관찰한다.

보장하는 invariant:

- 선택 action이 첫 dispatch다.
- 최대 4개의 실제 dispatch만 attribution한다.
- failed launch는 기록하지 않는다.
- P/D residual augmentation으로 plan ID가 바뀌면 active episode도 remap한다.
- completion은 CUDA event가 host에서 관측된 시점으로 기록한다.
- drain 시 partial episode는 truncated로 명시한다.
- callback/summary drain은 exact-once다.

focused test는 최종 `189/189`를 통과했고, dominance helper 추가 뒤 planner test `14/14`를 다시 통과했다.

## 4. 첫 attribution 결과

Artifacts:

```text
p95  .local/transition-aware-20260830/p95-realized-jsonl-long-1x/
p96  .local/transition-aware-20260830/p96-realized-regressions-3x/
p97  .local/transition-aware-20260830/p97-realized-long-3x/
```

### 4.1 세 회귀 workload

| Workload | Episodes | D budget known | First D gap mean/p95 | First D rows median/max |
|---|---:|---:|---:|---:|
| text-heavy | 2 | 0 | 47.82/48.52 ms | 27.5/28 |
| vision-heavy | 10 | 0 | 95.84/117.32 ms | 20.5/26 |
| Poisson mixed | 2 | 0 | 45.94/45.95 ms | 30/30 |

모든 episode는 4 dispatch를 완전히 기록했고 D는 실제로 형성됐다. 그러나 당시 snapshot의 D budget은 모두 unknown이었다.

### 4.2 장기 trace 3회

```text
requests/run       384
episodes           21
truncated          0
decode serviced    21
decode budget      0 known
first D gap        mean 87.24 ms, median 96.82 ms, p95 111.58 ms
first D rows       median 22, p95 51, max 52
token hash         3/3 identical
```

H2 change는 주로 E/P boundary에서 발생했다. 현재 ready-D는 없었지만 이미 outstanding인 sampling work가 45--112ms 뒤 큰 D cohort를 만들었다. 따라서 `ready resident D only` protection은 충분하지 않았다.

## 5. Cascade-bounded concrete completion

임의 H=3나 future-arrival predictor를 추가하지 않았다. 다음처럼 제한했다.

```text
current immutable ready work
        |
        v
equal-work H2
  current + <=1 successor
        |
        v
first already-submitted sampling CUDA event
        |
        v
one conservative D service boundary
```

두 번째 cumulative preview는 첫 cohort를 포함하므로 중복 append하지 않는다. event ready time, uncertainty, D p95 service cost, request TPOT budget은 snapshot hash에도 포함된다.

이 구조는 다음을 예측하지 않는다.

- 외부 request arrival
- future workload class
- 아직 제출되지 않은 sampling
- arbitrary H=3 action Cartesian product

## 6. 발견한 action-authority 버그

초기 dominance gate는 myopic이 formation oracle보다 나쁜지만 확인했다. 이 조건은 oracle이 `encoder`인데 후단 Global selector가 고른 제3의 `prefill` action도 허용했다.

```text
formation oracle  encoder
myopic            encoder+prefill
final selector    prefill
old gate          allow, because myopic regret > 0
```

이 상태의 p99에서 7개 selection change가 모두 `prefill <- encoder_prefill`이었고 7개 모두 실제 D budget을 초과했다.

최종 gate는 final selected action을 직접 비교한다.

```text
replace myopic iff
  regret(final selected) < regret(myopic)
```

equal violation/equal horizon tie에서는 stable myopic ordering을 유지한다. 이는 workload heuristic이 아니라 동일 observable frontier에 대한 strict dominance 조건이다.

## 7. 최종 장기 A/B

Artifacts:

```text
final H2  .local/transition-aware-20260830/p101-final-dominance-h2-long-3x/
myopic    .local/transition-aware-20260830/p102-final-myopic-long-3x/
trace     permuted-6epoch.json
repeats   3
```

같은 binary, engine, model, trace, request contract를 사용했다.

| Metric | Final H2 | Myopic | H2 delta |
|---|---:|---:|---:|
| generated tok/s | 586.179 | 584.561 | +0.28% |
| TTFT mean | 1102.249 ms | 1136.089 ms | -2.98% |
| TTFT p95 | 3289.755 ms | 3391.840 ms | -3.01% |
| TPOT mean | 25.287 ms | 25.253 ms | +0.13% |
| TPOT p95 | 39.441 ms | 39.923 ms | -1.21% |
| E2E mean | 2250.373 ms | 2277.855 ms | -1.21% |
| E2E p95 | 4049.495 ms | 4132.749 ms | -2.01% |
| peak VRAM | 9473 MiB | 9481 MiB | parity |

token trace SHA256는 양쪽 모든 run에서 동일했다.

장기 aggregate에서는 H2가 joint improvement에 가깝다. TPOT mean만 0.13% 악화로 사실상 동률이고 나머지는 개선됐다.

## 8. 아직 해결되지 않은 mismatch

final H2 3회에서:

```text
realized episodes          10
known D budgets             9
actual D budget violations  9
first D gap mean/p95       103.75/137.81 ms
predicted myopic regret     mean 40.08 ms
regret vs D-gap corr        0.388
regret vs horizon corr      0.132
```

aggregate 성능은 개선됐지만 local D prediction은 아직 정확하지 않다. 원인은 현재 boundary가 complete H2 horizon 뒤 D를 append하는 반면 실제 D enqueue 가능 시점은 action의 phase별 completion과 outstanding-set/augmentation semantics에 의해 결정되기 때문이다.

```text
current model
  whole H2 horizon complete -> D

actual runtime
  E/P component completion
       + execution lease state
       + residual augmentation legality
       -> first permissible D enqueue
```

따라서 이 결과만으로 concrete-boundary H2를 production default로 승격하지 않는다.

## 9. 최종 상태

```text
production default
  myopic transition-safe selector

research opt-in
  contextual P+D/E+P/E+D
  + equal-work H2
  + first concrete sampling boundary
  + final-action strict dominance
  + realized 4-dispatch attribution
```

메커니즘 correctness와 telemetry는 완성됐다. 장기 A/B도 회귀 없이 통과했다. 그러나 local D service prediction mismatch 때문에 12-workload promotion gate 전에는 research opt-in을 유지한다.

## 10. 다음 순서

1. **Action-specific service boundary**: whole-horizon 대신 각 action의 earliest legal D augmentation/enqueue boundary를 execution lease에서 계산한다.
2. **Shadow validation**: policy는 바꾸지 않고 predicted first-D time과 realized gap의 error/correlation을 12 workload에서 측정한다.
3. **Promotion A/B**: error가 안정화된 뒤 long-lived와 12-workload H2/myopic gate를 재실행한다.
4. **Host path decomposition**: text-heavy/Poisson saturation에서 GPU completion, sampling visibility, state commit, candidate formation, selector, TRT enqueue, first kernel을 분리한다.
5. **vLLM rule**: trace와 HTTP/output contract가 같으므로 기존 frozen vLLM anchor를 재사용한다. contract가 바뀔 때만 fresh vLLM을 실행한다.
