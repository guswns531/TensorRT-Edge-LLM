# In-flight-aware scheduler M5: measured Oracle H1 selector

## 1. Outcome

M5의 measured/replay Oracle H1 ranker와 controlled objective validation을 구현했다. Production selector는 변경하지
않았다.

```text
M4 completion vector + deterministic projection
                         |
                         v
               measured H1 candidates
                         |
                         v
hard feasibility -> robust SLO -> progress -> efficiency
                         |
                         v
                 stable action choice
```

핵심 구현과 단위 검증은 통과했다. 하지만 12-workload와 saturation Gate B는 아직 **통과가 아니라 평가 불가**다.
현재 natural trace에는 동일 immutable snapshot의 여러 legal action을 모두 실제 실행한 counterfactual completion vector가
없다. 선택된 action의 completion만으로 Oracle을 만들면 oracle이 아니라 current policy replay가 되므로 그렇게 하지 않았다.

## 2. Scope and non-scope

M5 Oracle은 production online oracle이 아니다.

- 동일 snapshot에서 실제로 측정했거나 controlled replay로 제공한 candidate만 비교한다.
- 보지 못한 action의 completion을 exact-key fallback, workload rule 또는 외부 registry로 추측하지 않는다.
- request DAG, outstanding context와 ownership transition은 M4 projector 결과를 그대로 사용한다.
- production myopic selector, legacy-compatible control과 H2 research mode의 실행 경로는 변경하지 않는다.

따라서 이번 구현은 predictor와 objective를 분리 검증하는 oracle evaluation plane이다.

## 3. Code map

| Path | Responsibility |
|---|---|
| `cpp/runtime/phase/policy/phaseOracleH1Selector.h` | measured candidate, protected milestone와 decision contract |
| `cpp/runtime/scheduling/phaseOracleH1Selector.cpp` | lexicographic Oracle H1 ranker |
| `unittests/phaseOracleH1SelectorTest.cpp` | feasibility/SLO/progress/efficiency와 H1 counterexample |
| `benchmarks/phase_serving/analyze_oracle_h1_policy.py` | offline candidate replay와 myopic/legacy/H2 agreement 분석 |
| `benchmarks/phase_serving/build_oracle_h1_snapshot_coverage.py` | exact cross-run snapshot별 measured action/completion join |
| `benchmarks/phase_serving/validate_phase_scheduler_events.py` | signature/progress lineage와 common-epoch interval 검증 |
| `tests/python-unittests/test_oracle_h1_policy.py` | replay validation과 Gate B coverage guard |
| `tests/python-unittests/test_oracle_h1_snapshot_coverage.py` | exact/legacy coverage 및 corrupted signature test |
| `cpp/CMakeLists.txt` | stable phase runtime object order에 M5 selector 추가 |
| `benchmarks/phase_serving/manifests/v010_feature_owners.json` | M5 source ownership 기록 |

## 4. Objective semantics

### 4.1 Hard feasibility

다음 조건을 모두 만족해야 후보가 ranker에 들어간다.

- M3 incremental action legality
- M4 projection validity
- action ID와 completion vector action ID 일치
- dependency/context/TRT shape/ownership safety
- persistent-memory hard peak가 budget 이내

Near reclaim은 hard capacity로 사용하지 않는다. 실제로 release된 ownership bytes는 마지막 tie-break에서만 사용한다.

### 4.2 Robust SLO

각 protected milestone은 현재 boundary에서의 completion mean과 uncertainty를 갖는다.

```text
violation(a) = max_r [completion_r(a) + uncertainty_r(a) - slack_r]^+
```

명시된 completion/uncertainty가 없으면 M4 projection의 earliest boundary와 robust boundary를 사용한다. 명시적 zero
uncertainty는 zero로 유지된다.

### 4.3 Urgency-normalized progress

Earliest boundary에서 실제 완료되는 milestone만 progress로 인정한다.

```text
progress(a) = sum completed progress units / max(1 us, request slack)
```

이는 workload 이름이나 `E1/P8` 같은 shape rule이 아니다. Replay contract가 제공한 request milestone과 현재 slack만
사용한다.

### 4.4 Efficiency and deterministic tie-break

```text
efficiency(a) = reference GPU work / robust earliest boundary
```

동일 SLO와 progress에서 efficiency가 큰 후보, 실제 ownership release가 큰 후보, action ID/source index가 작은 후보 순으로
고른다. 최종 순서는 다음과 같다.

1. hard feasibility
2. minimum robust SLO violation
3. maximum urgency-normalized milestone progress
4. maximum H1 service efficiency
5. maximum observed ownership release
6. stable action identity

## 5. Controlled replay result

Artifact:

```text
.local/m5-oracle-h1-20260901/controlled-replay.json
.local/m5-oracle-h1-20260901/controlled-analysis-residual.json
```

M4 actual P/D earliest-boundary values와 controlled all-late/progress counterexample 네 개를 사용했다.

| Baseline | Oracle agreement | Oracle changes | Robust violation regret |
|---|---:|---:|---:|
| myopic | 0/4 | 4 | 6,024.243 us |
| legacy-compatible | 3/4 | 1 | 4,224.243 us |
| current H2 | 3/4 | 1 | 0 us |

이 표는 성능 결과가 아니다. Objective ordering이 의도한 candidate를 고르고 기존 policy choice와 다른 선택을 표현할 수
있는지 검증하는 controlled result다.

특히 whole pair가 15 ms 뒤 끝나더라도 D가 6 ms에 먼저 끝나면 Oracle은 6 ms H1 boundary에서 D milestone과 남은 P
in-flight를 평가한다. M4가 제거한 whole-H2 mismatch가 ranker에서 다시 생기지 않는다.

## 6. Validation

### C++

TensorRT 26.06 container, SM86 build에서 다음 target을 재빌드했다.

```text
unitTest
llm_phase_context_smoke
```

Focused and adjacent test result:

```text
PhaseGlobalSchedulerTest          32/32
PhaseIncrementalActionTest         7/7
PhaseIncrementalProjectorTest      7/7
PhaseOracleH1SelectorTest           6/6
PhaseUnifiedEventTest               5/5
total                              57/57
```

### Python

```text
M2--M5 focused Python tests      13/13
```

`git diff --check`와 feature-owner JSON parse도 통과했다.

## 7. Gate B decision

> 2026-09-01 follow-up: cross-policy actual execution으로 exact multi-action candidate coverage는 열렸다. 다만
> repeated-alternative coverage와 12-workload/saturation promotion은 아직 남아 있다. 최신 판정은
> `notes/206-inflight-m5-cross-policy-exact-coverage-20260901.md`를 따른다.

현재 판정:

```text
controlled objective semantics       PASS
action fidelity in controlled replay 100%
12-workload counterfactual coverage  MISSING
39/48.8/97.5 coverage                MISSING
Gate B promotion                     NOT EVALUABLE / NOT PASSED
production default                   unchanged myopic
```

12-workload를 단순히 myopic/H2로 다시 실행하는 것은 이 누락을 해결하지 않는다. 각 실행은 선택한 action만 관측하며,
첫 action divergence 이후 snapshot 자체가 달라진다. 서로 다른 snapshot의 completion을 같은 candidate frontier로 합치면
policy-only A/B가 아니다.

따라서 vLLM도 다시 실행하지 않았다. Runtime/request/output contract와 production policy가 바뀌지 않았으므로 frozen vLLM
anchor가 유효하며, 이번 단계의 질문은 same-runtime Oracle objective다.

## 8. Required continuation before learned active scheduling

M5에서 드러난 coverage 문제를 다음처럼 해결한다.

1. deterministic snapshot signature를 `ready rows + canonical request order + per-request P/D progress + in-flight
   lineage + ownership state`로 고정한다.
2. 같은 signature에서 policy가 실제 선택한 action/completion vector를 여러 repeated runs 사이에 join한다.
3. 둘 이상의 legal measured candidate가 모인 snapshot만 Oracle H1 episode로 인정한다.
4. coverage가 부족한 frontier는 ample SLO slack에서 bounded one-shot probe로 보완한다.
5. 12-workload와 39/48.8/97.5에서 action coverage, selection regret, SLO goodput을 함께 기록한다.

이 continuation은 arbitrary future arrival 예측이나 workload별 tuning이 아니다. 같은 observable snapshot의 측정된 action만
추가한다. 충분한 natural coverage가 생기기 전에는 M6 learned predictor를 active로 승격하지 않는다.

이 continuation의 실제 Cosmos 검증과 false-match 교정은
`notes/205-inflight-m5-exact-snapshot-coverage-20260901.md`에 기록했다.

## 9. Architectural conclusion

M5는 policy authority를 production path에 하나 더 추가하지 않았다.

```text
deterministic mechanism
  M3 legality + M4 projection

evaluation policy
  measured Oracle H1 ranker

production policy
  existing myopic transition-safe selector
```

가장 중요한 결과는 Oracle objective의 구현 가능성과 counterfactual coverage의 한계를 분리한 것이다. 선택된 action 하나의
로그만으로 Oracle 성능을 주장하지 않는 것이 이후 learned predictor의 검증 오염을 막는다.
