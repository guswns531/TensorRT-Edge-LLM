# Profile-free Online Learning Stability: P0–P9 실행 계획

날짜: 2026-09-03

브랜치: `codex/v010-phase-forward-port`

기준 commit: `3908810270dee65ac331e517c010b02095ada765`

선행 결과: `notes/221-process-local-calibration-and-saturation-r8-results-20260902.md`

## 1. 목표

최종 목표는 workload 이름이나 workload별 정책 table 없이 같은 E/P/D controller가 현재 ready state,
request slack, ownership transition, CUDA observation만으로 안정화되는 것이다. 평가의 중심은 단순히
warm state의 최고 처리량이 아니다.

1. 새 process에서 학습이 언제 안정화되는가?
2. 안정화 이전에 SLO나 correctness를 훼손하지 않는가?
3. trace-derived warmup을 제거해도 같은 controller가 여러 workload에 일반화되는가?
4. 안정화 이후 동일 HTTP trace에서 vLLM보다 높은 SLO goodput과 낮은 TTFT/TPOT/E2E를 유지하는가?

외부 cost registry, persisted posterior, workload label은 사용하지 않는다. 모든 policy state는 process-local이고
fresh process마다 비어 있는 상태에서 시작한다.

## 2. 현재 발견한 평가 오염

현재 startup shape warmup은 두 역할을 동시에 수행한다.

```text
synthetic startup requests
       |
       +-- CUDA graph / tactic / allocator priming       (mechanism)
       |
       +-- exact execution-cost observations             (execution model)
       |
       +-- contextual RLS/completion/conformal updates   (policy model)
```

최근 기본 실행은 32개 batch, 1,628 synthetic request, 155 overlap sample을 만든 뒤
`resetHistory(true)`를 호출했다. 이 호출은 queue telemetry만 지우고 execution cost와 policy posterior를 모두
보존했다. 그 뒤 같은 production trace를 burst 형태로 128~256 request 재생했다. 따라서 결과는 persisted
profile은 아니지만 zero-shot도 아니며, workload-distribution-aware premeasurement calibration이다.

P0–P9는 이 세 상태를 분리한다.

```text
Mechanism state       : CUDA graph, TensorRT profile/tactic, allocator pools
Execution-cost state  : exact action/decode CUDA-event samples
Policy state          : contextual action value, completion RLS, conformal uncertainty
```

## 3. 네 warmup mode

| Mode | Mechanism priming | Generic policy calibration | Trace-derived calibration | Measurement 중 online update |
|---|---:|---:|---:|---:|
| `graph_only` | O | X | X | X |
| `zero_start` | O | X | X | O |
| `generic` | O | O, 모든 workload에 동일 | X | O |
| `trace_derived` | O | O | O, 측정 trace 순환 | O |

`trace_derived`는 현재 성능 상한/reference로만 유지한다. profile-free headline은 `generic`이고, cold adaptation은
`zero_start`로 평가한다. `graph_only`는 policy learning 자체의 causal baseline이다.

## 4. 안정화 정의

단순 observation count만으로 안정화됐다고 하지 않는다. ordered direction마다 다음 조건을 모두 만족해야 한다.

```text
action-value observations >= 4
completion ready observations >= 16
conformal uncertainty calibrated
held-out interval coverage >= target - tolerance
false-safe rate == 0, 또는 사전 정의한 매우 작은 upper bound 이하
rolling normalized prediction error가 연속 두 window에서 threshold 이하
rolling action disagreement/regret가 연속 두 window에서 안정
```

요청/decision index 구간은 최소 다음을 보고한다.

```text
1–16, 17–32, 33–64, 65–128, 129–256, 257+
```

각 구간은 family/direction별 observation 수, mean, uncertainty, lower confidence bound, actual reward,
absolute prediction error, interval coverage, false-safe, selected action, decision latency를 포함한다.

## 5. 승격 기준

### Correctness

- 기존 greedy token hash와 exact identity
- semantic check 통과
- action fidelity failure 0
- invalid ownership/page/vision lease 0
- CUDA sanitizer OOB/use-after-release 0

### Learning safety

- low-confidence action은 serial 또는 shadow fallback
- completion authority는 direction별 16개 observation과 uncertainty calibration 전에는 사용 금지
- false-safe 0을 우선 gate로 사용
- production traffic에서 exploration 강제 금지; exploration은 충분한 slack의 bounded calibration에만 허용

### Performance

- `generic`은 `trace_derived` 대비 각 대표 workload에서 throughput과 latency p95가 3% 이내
- `zero_start` cold prefix의 손실과 안정화 request 수를 별도 보고
- 안정화 이후 Current가 frozen vLLM보다 SLO goodput 우세
- TTFT/TPOT/E2E mean과 p95를 모두 보고하고 한 지표의 개선으로 다른 지표의 회귀를 숨기지 않음
- peak memory 최소 512 MiB headroom

## 6. P0–P9

### P0 — Baseline freeze

commit, engine/model hash, driver/CUDA/TensorRT, trace hash, command, result artifact를 manifest로 고정한다.
R8 48.8 req/s 5-repeat과 R7 12-workload를 변경 전 기준으로 사용한다.

### P1 — State reset 분리

다음 API를 독립시킨다.

```text
resetSchedulingHistory()
resetPolicyPosterior()
resetExecutionCostHistory()
```

CUDA graph/profile cache는 coordinator 소유이므로 위 reset의 영향을 받지 않는다. 기존
`resetHistory(bool)`은 source compatibility wrapper로 남긴다.

### P2 — Warmup mode 계약

`TRT_EDGELLM_POLICY_WARMUP_MODE`에 네 mode를 추가한다. mode는 calibration response, phase metric,
run manifest에 기록한다. measurement epoch 시작점에서 history와 필요한 model state를 정확히 초기화한다.

### P3 — Workload-independent generic calibration

한 고정 trace가 P/D, E/P, E/D 양방향을 모두 덮는다. workload의 production trace를 보지 않는다.

```text
P rows: 1 / 4 / 8, fixed chunk 128
D rows: 8 / 32 / 64
E rows: 1 / 2 / 4
pair direction: incumbent/newcomer 양방향
```

가능하면 temporary calibration buffers와 isolated request namespace를 사용한다. user KV, prefix cache,
sampling result, output token hash에 영향을 주면 generic calibration은 승격하지 않는다.

### P4 — Authority readiness

Action-value model readiness와 completion-authority readiness를 분리한다. completion authority는 direction별
evidence, conformal coverage, false-safe gate를 모두 통과한 family/direction에만 허용한다. 나머지는 shadow
prediction을 기록하되 robust deterministic estimate를 계속 사용한다.

### P5 — Adaptation curve

누적 최종값 대신 request/decision index별 curve를 만든다. cold, transition, steady 세 구간을 자동 판정하고
time-to-stability, requests-to-stability, steady prediction error, steady throughput/goodput을 보고한다.

### P6 — Representative warmup A/B

`balanced`, `decode-heavy`, `vision-heavy`, `multi-image`, `late-vision`, `48.8 saturation`을 네 mode로
3회 실행한다. 민감하거나 CV가 큰 경우 5회로 올린다. 같은 trace의 frozen vLLM 결과는 재사용하되 HTTP,
model, output contract가 달라지면 다시 실행한다.

### P7 — Vision critical-path decomposition

vision-heavy에서 다음 경계를 request별로 연결한다.

```text
arrival -> adapter ready -> E queue -> E start/end -> vision lease ready
        -> P queue -> P start/end -> first D -> first token
```

E evidence 부족, E formation wait, P/D interference, host preparation, TensorRT submission 중 어떤 항이 cold 및
steady tail을 만드는지 분리한다. workload-specific threshold는 추가하지 않는다.

### P8 — 12-workload gate

동일 `generic` configuration 하나로 12개 workload를 모두 실행한다. throughput, TTFT/TPOT/E2E mean/p95,
joint SLO goodput, memory, E/P/D/C activity mask, direction별 calibration/error와 cold/steady 결과를 기록한다.

### P9 — Causal ablation

같은 trace/engine/HTTP 계약에서 다음을 비교한다.

```text
clean v0.10
mechanism only / graph_only
P+D contextual only
full E/P/D shadow
full E/P/D active / generic
trace-derived upper reference
frozen vLLM
```

정책 구성마다 candidate mechanism, admission, graph state, chunk 128, max P/D/E shape를 같게 유지한다.
차이는 online decision authority 하나만 남긴다.

## 7. 해석 원칙

- `trace_derived`가 vLLM을 이겨도 profile-free 성공으로 간주하지 않는다.
- `generic`이 vLLM을 이기고 여러 workload에서 3% gate를 통과해야 main result로 사용한다.
- `zero_start`는 시작부터 vLLM을 이길 필요는 없지만 안정화까지의 손실 면적과 시간/request 수를 보고한다.
- warmup이 충분하지 않은 E direction을 P/D와 같은 수준으로 학습됐다고 표현하지 않는다.
- policy가 선택한 action과 CUDA timeline의 actual outstanding phase set이 다르면 해당 run은 무효다.
- 평균 성능뿐 아니라 process CV와 confidence interval을 사용한다.

## 8. 최종 성공 형태

```text
same implementation + same hyperparameters
             |
fresh process, no persisted model
             |
generic calibration or zero-start adaptation
             |
directional posterior converges safely
             |
steady SLO goodput > vLLM
TTFT / TPOT / E2E mean,p95 <= vLLM
             |
12-workload에서 workload-specific tuning 없이 유지
```

핵심 논문 메시지는 “warmup table을 잘 만들었다”가 아니다. deterministic mechanism은 legal action을
정하고, low-dimensional process-local model은 현재 state에서 valuable action을 학습하며, uncertainty가
충분히 줄기 전에는 authority를 얻지 못한다는 구조다.

## 9. 실행 결과

P0–P9 구현과 GPU 검증 결과는
`notes/223-profile-free-learning-stability-p0-p9-results-20260903.md`에 기록했다. 최종 판정은 P+D family는
active promotion 가능, E+P/E+D family는 shadow 유지다. 48.8 req/s text saturation은 frozen vLLM을 넘었지만
12-workload 전체 목표는 10/12 throughput 우세로 아직 미완료다.
