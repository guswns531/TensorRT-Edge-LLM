# In-flight M5 continuation: exact snapshot coverage

## 1. Outcome

M5 Oracle H1이 실제 counterfactual을 비교할 수 있도록 cross-run snapshot identity와 measured coverage builder를
완성했다. Production action selector는 바꾸지 않았다.

```text
E/P/D ready lineage + per-request P/D progress
                 +
in-flight phase/cohort/status + ownership state
                 |
                 v
      deterministic FNV-1a signature
                 |
                 v
same signature across measured policy runs
                 |
                 v
common-epoch completion vectors by action
```

실제 Cosmos 4-image trace를 두 번 실행한 최종 결과는 다음과 같다.

| Item | Result |
|---|---:|
| Requests / run | 4 |
| Prompt / output tokens | 2,034 / 64 |
| Unified decisions | 45 |
| Dispatch/completion executions | 49 / 49 |
| Common-epoch GPU intervals | 49 / 49 |
| Action-fidelity failures | 0 |
| Cross-run exact signatures | 7 |
| Exact repeated snapshots | 7 |
| Exact multi-action snapshots | 0 |
| Token trace deterministic | yes |
| Gate B candidate coverage | **false** |

이 결과는 실패가 아니라 올바른 coverage 판정이다. 같은 policy를 반복하면 동일 state가 재현될 수 있지만, 다른 legal
action을 실제 실행하지 않았으므로 counterfactual Oracle이라고 주장할 수 없다.

## 2. Why request IDs were insufficient

초기 signature는 ready request ID와 aggregate row count를 보존했지만 각 decode row의 현재 KV length를 보존하지
않았다. 같은 두 request가 decode를 한 step 수행한 뒤에도 ID와 row count는 같으므로 서로 다른 state가 같은 signature로
합쳐졌다.

초기 actual coverage는 다음 false positive를 만들었다.

```text
same apparent signature
  observation A: D
  observation B: P+D
```

두 observation을 조사하니 A와 B 사이에 decode context가 전진했다. 즉 action alternative가 아니라 서로 다른 iteration이었다.
이를 exact multi-action coverage로 세면 Gate B가 거짓 통과한다.

최종 signature는 다음 parallel vectors를 추가한다.

- `ready_prefill_request_ids[i]` + `ready_prefill_token_counts[i]`
- `ready_decode_request_ids[i]` + `ready_decode_context_lengths[i]`

그 결과 false multi-action count는 `1 -> 0`으로 제거됐고, 실제 cross-run repeated state 7개는 유지됐다.

## 3. Signature contract

포함하는 observable state:

- current outstanding E/P/D mask
- E/P/D ready rows와 token/context aggregate
- canonical ready request order
- per-request remaining prefill tokens와 decode KV length
- allocated/guaranteed page bundles와 vision payload bytes
- in-flight phase, status, cohort와 request lineage

의도적으로 제외하는 transient state:

- process-local execution/plan/event ID
- host timestamp와 dispatch age
- in-flight vector polling order

따라서 signature는 같은 replay state에는 안정적이고, request progress나 ownership feasibility가 다른 state에는 민감하다.

## 4. M4 residual-boundary correction

Coverage 구현 중 M4 offline analyzer의 H1 origin 오류도 교정했다. Incremental action의 decision boundary는 incumbent의
원래 GPU start가 아니라 **newcomer actual GPU start**다. 따라서 incumbent completion은 전체 실행 시간이 아니라 그
boundary에서 남은 residual이어야 한다.

Corrected actual artifact:

```text
.local/m5-oracle-h1-20260901/pd-o50-h1-replay-residual.json
```

| Metric | Corrected result |
|---|---:|
| P/D directional pairs | 74 |
| Earliest D / P | 67 / 7 |
| Projected H1 boundary median | 3.986 ms |
| Projected boundary range | 1.350--8.257 ms |
| Whole-action overrun median | 8.292 ms |
| Invalid/triple successors | 0 / 0 |

Controlled Oracle comparison도 residual origin으로 다시 생성했다.

| Baseline | Agreement | Robust violation regret |
|---|---:|---:|
| myopic | 0/4 | 6.024 ms |
| legacy-compatible | 3/4 | 4.224 ms |
| H2 | 3/4 | 0 ms |

H2의 한 disagreement는 violation regret이 아니라 progress/efficiency tie ordering 차이다.

## 5. Actual run artifacts

최종 검증 자료:

```text
.local/m5-oracle-h1-20260901/signature-repeat-vlm-v3/aggregate.json
.local/m5-oracle-h1-20260901/signature-repeat-vlm-v3/run-001/gateway.log
.local/m5-oracle-h1-20260901/signature-repeat-vlm-v3/run-002/gateway.log
.local/m5-oracle-h1-20260901/signature-repeat-vlm-v3/oracle-h1-coverage.json
```

Median actual serving metrics:

| Metric | Result |
|---|---:|
| generated token/s | 150.81 |
| TTFT mean / p95 | 263.70 / 298.20 ms |
| TPOT mean / p95 | 9.76 / 12.01 ms |
| E2E mean / p95 | 410.04 / 419.33 ms |
| peak VRAM | 9,275 MiB |

이 수치는 signature correctness smoke의 성능 기록이며 policy speedup 주장이 아니다.

## 6. Validation

- TensorRT 26.06 / CUDA 13.3 build: `unitTest`, `llm_phase_context_smoke` PASS
- C++ focused/adjacent: 223/223 PASS
- Python M2--M5: 13/13 PASS
- Actual event validator: 143 events, 49/49 common-epoch intervals, 0 fidelity failures
- Token trace: 2/2 exact SHA-256 identity
- JSON schema parse and `git diff --check`: PASS

## 7. Next step

Cross-policy actual 실행과 empirical H1 분석을 후속 완료했다.

```text
notes/206-inflight-m5-cross-policy-exact-coverage-20260901.md
```

Exact multi-action candidate coverage는 `0 -> 4`로 열렸지만, same-phase H1 비교 3개 중 모든 대안이 2회 이상
반복된 것은 2개다. 따라서 아래의 원래 12-workload promotion 단계는 여전히 남아 있다.

Gate B를 평가하려면 동일 exact signature에서 둘 이상의 measured action이 필요하다.

1. Production default는 유지한다.
2. 동일 frozen trace를 `myopic`, `legacy-compatible`, `H2`, bounded safe-probe policy로 반복한다.
3. 첫 divergence 이후의 다른 state를 강제로 합치지 않고 signature가 자연히 일치하는 episode만 join한다.
4. natural coverage가 부족하면 ample request slack에서 action당 one-shot bounded probe를 사용한다.
5. exact multi-action snapshot에서만 Oracle regret을 계산한다.
6. 12-workload와 39/48.8/97.5 load points의 coverage가 충분해진 뒤 M6 RLS predictor를 active로 검토한다.

현재 promotion 상태는 다음과 같다.

```text
mechanism correctness     PASS
signature repeatability  PASS
false-match rejection    PASS
multi-action coverage    MISSING
Gate B                   NOT EVALUABLE
production policy        unchanged
```
