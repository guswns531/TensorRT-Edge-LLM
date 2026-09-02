# In-flight M5: cross-policy exact coverage and empirical H1

## 1. Outcome

동일 Cosmos 4-image frozen trace에 production myopic, Legacy-compatible, H2, bounded safe-probe를 실제로
실행했다. CUDA common-epoch와 exact output을 모두 만족한 실행만 최종 coverage에 포함했다.

```text
same immutable execution snapshot
        +
different measured legal actions
        +
common-epoch CUDA completion vectors
        |
        v
empirical H1 pilot oracle
```

최종 exact-output coverage:

| Item | Result |
|---|---:|
| Decisions joined | 203 |
| Snapshot signatures | 112 |
| Exact repeated snapshots | 61 |
| Exact multi-action snapshots | 4 |
| Same-first-phase H1-comparable snapshots | 3 |
| Fully repeated alternatives | 2 / 3 |
| Action-fidelity failures | 0 |
| Candidate coverage Gate | PASS |
| Promotion-quality coverage | **FAIL: one D-only alternative remains n=1** |

한 multi-action snapshot은 `D`와 `P`가 서로 다른 first-completion phase를 가지므로 H1 boundary Oracle에서
의도적으로 제외했다. 서로 다른 progress semantics를 실행시간 하나로 비교하지 않는다.

## 2. Measurement contract corrections

정확한 Oracle 자료에는 다음 두 환경 설정이 모두 필요하다.

```text
TRT_EDGELLM_EMIT_PHASE_METRICS=1
TRT_EDGELLM_PHASE_ACTIVITY_PREFIX=<per-run prefix>
```

첫 번째만 없으면 unified scheduler event가 생성되지 않는다. 두 번째가 없으면 completion event는 있지만 CUDA
common-epoch `gpu_start_us/gpu_end_us`가 없다. 따라서 초기 `policy-coverage-v1`과 `v2`는 Oracle 입력에서 제외했고,
두 조건을 모두 만족한 `policy-coverage-v3`만 사용했다.

Final event validation:

- Legacy/H2/safe-probe first runs: 90 decisions, 93 executions, 93 GPU intervals, fidelity failure 0
- H2 exact-output repeat: 27 decisions/executions/intervals, fidelity failure 0
- safe-probe repeat: 20 decisions, 23 executions/intervals, fidelity failure 0
- myopic repeat: 45 decisions/executions/intervals, fidelity failure 0

## 3. Exact-output policy runs

모든 final exact-output run의 token hash:

```text
9c76d5424999d700e3b9522b47b1837608dea2538674a7489d9a21a3ba0d21e0
```

H2 첫 실행 하나는 request 3의 여섯 번째 generated token부터 갈라졌다. 그러나 해당 실행의 formation lookahead와
selection change는 모두 0이었고, 바로 다음 H2 반복은 기준 hash와 일치했다. 따라서 이를 H2 action의 인과적 오류로
분류하지 않고 현재 FP16/runtime 반복 결정성 경고로 남긴다. Divergent 실행은 exact-output coverage에서 제외했다.

## 4. Request-level pilot performance

아래 수치는 4-request trace의 pilot이며 promotion 성능 결과가 아니다.

| Policy | Runs | gen tok/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|---:|
| Production myopic reference | 2 | 150.81 | 263.70 / 298.20 | 9.76 / 12.01 | 410.04 / 419.33 |
| Myopic natural repeat, serial path | 1 | 113.84 | 333.82 / 443.64 | 11.51 / 15.79 | 506.54 / 561.60 |
| Legacy-compatible | 1 | 113.46 | 332.18 / 439.75 | 12.08 / 16.53 | 513.36 / 563.38 |
| H2 exact repeat | 1 | 145.44 | 275.39 / 325.78 | 9.63 / 12.30 | 419.80 / 439.44 |
| Bounded safe-probe | 2 | 156.20 | 262.70 / 285.77 | 9.43 / 11.87 | 404.14 / 406.21 |

Safe-probe의 두 실행은 production myopic reference 대비 median generated token/s가 약 `+3.6%`, E2E mean이
약 `-1.4%`, E2E p95가 약 `-3.1%`였다. 하지만 4-request trace이고 내부 cost history가 달라질 수 있으므로 이
수치만으로 production default를 변경하지 않는다.

추가 myopic 반복은 동일 출력이면서도 E+P 대신 serial E, P504 대신 P513을 선택해 Legacy와 비슷한 성능 cluster로
이동했다. 따라서 기존 2-run reference만으로 분산을 작다고 가정하면 안 된다. 이 history-sensitive action variance 자체가
12-workload 반복 측정에서 확인해야 할 다음 항목이다.

## 5. Empirical H1 result

새 분석기:

```text
benchmarks/phase_serving/analyze_oracle_h1_snapshot_coverage.py
```

같은 exact snapshot에서 대안들이 동일 phase를 첫 번째로 완료할 때만 median H1 completion boundary를 비교한다.
정책 내부 online cost state는 execution signature에 넣지 않는다. 같은 execution state에서도 관측 history에 따라 같은
online policy가 다른 action을 고를 수 있으므로 모든 선택 관측을 보존해 regret을 계산한다.

| Policy | Exact snapshots | Selection observations | H1 agreement | Total pilot regret |
|---|---:|---:|---:|---:|
| myopic | 3 | 7 | 4 / 7 | 4.587 ms |
| Legacy-compatible | 2 | 2 | 1 / 2 | 4.072 ms |
| H2 exact repeat | 2 | 2 | 1 / 2 | 4.072 ms |
| safe-probe | 3 | 6 | 2 / 6 | 21.676 ms |

Myopic은 같은 execution snapshot 두 곳에서 cost-history-sensitive selection을 보였다. 이것은 signature 오류가 아니다.

```text
execution signature = action의 GPU 환경을 join하는 key
policy state         = 해당 실행에서 action을 고른 online knowledge
```

두 상태를 한 signature에 합치면 cross-policy counterfactual coverage가 다시 사라진다.

## 6. Important interpretation

Safe-probe는 full trace E2E가 가장 좋았지만, exact P+D snapshot의 H1 boundary는 D-only보다 약 10.6 ms 늦었다.

```text
H1 view
  D-only first completion wins

full trace view
  P+D also advances prefill
  -> successor formation and downstream lifetime can improve
```

따라서 이번 결과는 H1 selector를 바로 production에 적용하라는 증거가 아니다. 오히려 H1 completion과 H2 successor
value를 분리해야 한다는 실제 counterexample이다. 다음 selector는 feasibility/SLO invariant를 유지하면서 H1 observation을
runtime cost evidence로 사용하고, action 선택은 bounded successor transition까지 본다.

## 7. Artifacts

```text
.local/m5-oracle-h1-20260901/policy-coverage-v3/
  oracle-h1-coverage-exact-output.json
  oracle-h1-empirical-analysis.json
  legacy/
  h2-repeat/
  safe-probe/
  safe-probe-repeat/
  myopic-repeat/
```

## 8. Gate and next step

현재 상태:

```text
exact multi-action candidate coverage   PASS
pilot H1 regret                         EVALUABLE
all alternatives repeated >= 2          2 / 3
12-workload natural coverage            PASS: 108 exact multi-action snapshots
39/48.8/97.5 saturation coverage        NOT RUN
production promotion                    BLOCKED
```

12-workload 결과와 최종 promotion 판정은
`notes/207-inflight-m5-workload12-policy-coverage-20260901.md`에 이어서 기록했다.

다음 단계는 이 작은 trace에 추가 heuristic을 맞추는 것이 아니다.

1. 동일 event-enabled 계약으로 12-workload와 39/48.8/97.5 trace를 수집한다.
2. exact execution-state join과 policy-history-sensitive 선택을 모두 보존한다.
3. H1 cost와 H2 successor value를 별도 column으로 평가한다.
4. natural multi-action coverage가 없는 frontier에서만 bounded one-shot probe를 사용한다.
5. output identity, fidelity, repeated-alternative coverage를 통과한 뒤 M6 predictor activation을 검토한다.

동일 workload/계약에 대한 production policy가 아직 바뀌지 않았으므로 vLLM은 재실행하지 않았다. 기존 frozen vLLM
reference는 이후 12-workload promotion 비교에서 함께 사용한다.
