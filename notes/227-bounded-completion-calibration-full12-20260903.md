# Bounded Completion Calibration 및 12-Workload Screening

날짜: 2026-09-03

브랜치: `codex/v010-phase-forward-port`

선행 문서:

- `notes/225-minimal-physical-outcome-full12-gate-20260903.md`
- `notes/226-physical-outcome-epoch-full12-results-20260903.md`

## 1. 이번 단계의 결론

이번 단계의 목적은 completion-vector/RLS 수학을 다시 확장하는 것이 아니라, E/P/D physical-outcome
controller가 실제로 필요한 방향의 학습을 **유한한 generic warm-up 안에서 끝내도록** calibration lifecycle을
완성하는 것이었다.

변경 전에는 다음 두 문제가 결합되어 있었다.

1. 한 방향이 held-out validation을 실패하면 `evidence ready == false`가 영원히 유지되어 같은 방향을 계속 probe했다.
2. continuous completion model이 학습할 수 있는 방향도 sparse exact-key registry의 32-key 한도를 공유했다. 먼저 등장하는
   P+D 및 E+P shape가 registry를 채우면, 늦게 실행 가능해지는 E+D는 completion sample을 만들지 못했다.

변경 후에는 각 ordered direction이 다음의 명시적인 bounded state machine을 가진다.

```text
posterior_fit
      |
      v
uncertainty_calibration
      |
      v
authority_validation
      |
      v
complete
  |        |
  | pass   | fail
  v        v
validated  scalar/exact fallback
```

`complete`는 authority 승격을 뜻하지 않는다. 충분한 held-out evidence로 **승격 또는 기각 결정을 끝냈음**을 뜻한다.
따라서 실패한 방향도 무한 probe하지 않는다. 반대로 아직 complete가 아닌 continuous direction은 sparse exact-key 예산이
가득 차도 계속 sample을 받을 수 있다.

12-workload screening 결과는 다음과 같다.

- 이전 completion-active 대비 throughput 기하평균은 **+1.30%**, 9/12 workload 승리, 12/12가 -3% 이내다.
- p10 최고 구현 대비 기하평균은 **-0.82%**, 3/12 승리, 11/12가 -3% 이내다.
- frozen vLLM 대비 기하평균은 **+13.43%**, 12/12 workload에서 throughput이 높다.
- 가장 큰 미회복 구간은 `vision-heavy`: p10 대비 **-9.89%**다.
- `text-heavy`, `mixed`, `long-prefill`은 이전 completion-active 대비 각각 **+3.16%, +3.41%, +2.40%**다.
- 현재 결과는 `vision-heavy`만 3회이고 나머지는 1회 screening이다. 최종 promotion 증거가 아니라 regression 및
  방향성 검증이다.

즉 bounded calibration 구현은 의도대로 동작하고 기존 workload를 망가뜨리지 않았다. 하지만 completion-vector 자체가
모든 direction에서 실질적 scheduling authority를 얻었다는 뜻은 아니다. held-out 검증을 통과해도 aggregate empirical
blend가 scalar보다 낫지 않으면 weight 0으로 남는다. 이것이 안전한 fallback contract다.

## 2. 문제의 정확한 구조

### 2.1 서로 다른 두 calibration 공간

```text
Sparse exact CUDA registry                 Continuous completion model
---------------------------------          ----------------------------------
shape/context/action exact key             10--20차원 normalized feature
정밀 진단 및 fallback                      direction physical outcome 예측
bounded max keys = 32                      posterior/conformal/held-out stage
새 shape마다 key 필요                      기존 sample에서 interpolation 가능
```

이 둘은 목적이 다르므로 같은 admission budget을 쓰면 안 된다. exact registry가 가득 찼다는 사실은 continuous E+D
direction이 충분히 학습됐다는 뜻이 아니다.

변경 전 흐름은 다음과 같았다.

```text
P+D/E+P가 exact key 32개를 먼저 점유
             |
             v
E+D candidate가 나중에 등장
             |
             v
calibrationTarget = false
             |
             v
E->D completion observation = 0
```

변경 후에는 `needsCompletionCalibration`을 별도로 계산한다.

```text
calibrationTarget
 = exact-key admission
   OR continuous direction incomplete
```

따라서 exact diagnostic은 bounded인 채로 유지하면서 E 방향 posterior/conformal/authority evidence를 끝까지 수집한다.

### 2.2 ready와 complete의 분리

기존 `EvidenceReady()`는 다음을 모두 만족해야 참이었다.

```text
held-out 표본 충분
+ coverage/false-safe 검증 통과
```

검증 실패도 `false`이므로 warm-up은 이를 "표본 부족"으로 오인했다. 새 `EvidenceComplete()`는 held-out 표본 수가
minimum에 도달했는지만 나타낸다.

```text
EvidenceComplete == false  -> 더 측정
EvidenceComplete == true
    EvidenceReady == true  -> completion authority 사용 가능
    EvidenceReady == false -> bounded rejection, fallback 사용
```

### 2.3 chronological candidate ranking

warm-up probe 후보는 더 이상 scalar/exact sample count만 비교하지 않는다. 각 direction의 가장 이른 미완료 단계를 우선한다.

```text
rank = (stage, stage_progress_fraction, observations)

stage 0: posterior fit
stage 1: pair-family uncertainty/conformal calibration
stage 2: held-out authority validation
stage 3: complete
```

이렇게 해야 E->P posterior가 최소 4개를 채운 직후 probe가 멈추지 않고 conformal 및 held-out 구간까지 진행된다.

### 2.4 candidate_seen과 required의 분리

reverse/residual direction은 candidate prediction은 만들어도 execution invariant 때문에 실제 dispatch되지 않을 수 있다.

```text
candidate_seen = predictions > 0
required       = observations > 0
```

실행된 적 없는 방향은 scalar fallback으로 남고 bounded calibration 종료를 막지 않는다. 이 구분이 없으면 E/P/D ready
snapshot에 등장했다는 이유만으로 영원히 달성할 수 없는 direction을 기다리게 된다.

## 3. 구현 위치

| 파일 | 변경 |
|---|---|
| `cpp/runtime/phase/cost/phaseRuntimeCostTracker.h` | calibration stage/progress API와 evidence-complete contract 추가 |
| `cpp/runtime/scheduling/phaseRuntimeCostTracker.cpp` | posterior -> uncertainty -> authority -> complete 상태 계산 |
| `cpp/runtime/scheduling/phaseQueueScheduler.cpp` | P+D continuous calibration을 exact-key 예산과 분리 |
| `cpp/runtime/scheduling/phaseThreeCoordinator.cpp` | E+P/E+D 분리 및 stage-progress 기반 warm-up candidate ranking |
| `examples/llm/llm_phase_context_smoke.cpp` | direction별 stage/count/validation telemetry와 bounded convergence 판정 |
| `unittests/phaseRuntimeCostTrackerTest.cpp` | 네 단계의 순차 진행과 complete/ready 의미 검증 |
| `unittests/phaseQueueSchedulerTest.cpp` | exact-key registry가 가득 찬 뒤에도 continuous sample이 수집됨을 검증 |

HTTP trace harness의 반복 warm-up에는 별도의 중요한 수정이 필요했다. 원본 trace의 arrival offset을 그대로 여러 번 이어
붙이면 두 번째 cycle에서 offset이 0으로 돌아가 `max-in-flight requires requests ordered by arrival_offset_us`가 발생했다.
각 cycle에 전체 trace span만큼 offset을 더하고, 각 calibration round는 자신의 local origin으로 rebase하도록 고쳤다.
이 변경은 outer research harness `scripts/cosmos_reason2/run_vllm_trace_bench.py`에 있으며 upstream-v010 source commit과는
별도로 관리한다.

## 4. Controlled calibration progression

`vision-heavy`, generic warm-up, max output 128을 사용해 원인을 단계적으로 분리했다.

| 구성 | warm-up requests | E->P | E->D | 결과 |
|---|---:|---:|---:|---|
| 기존 exact-budget coupling | 424 | obs 26, held-out 6 | obs 0 | E+D starvation |
| exact-budget 분리 | 424 | obs 21, held-out 1 | obs 12, conformal 8 | E+D probe 회복 |
| 2 cycles | 848 | obs 33, held-out 13, complete | obs 22, held-out 2 | E->D authority 진행 중 |
| bounded final | 1,272 | obs 48, complete/validated | obs 31, complete/validated | required 4/4 complete |

실제 workload에 따라 1,272 또는 1,696 request에서 조기 종료했다. 최대치를 채웠기 때문에 종료한 것이 아니라, 실제
관측된 모든 required direction이 held-out decision을 끝낸 시점에 종료했다.

## 5. 12-workload 성능

### 5.1 Throughput 및 이번 구현의 절대 latency

`old`는 `notes/226`의 completion-active 3회 median, `p10`은 이전 최고 small-D 구현 3회 median, vLLM은 고정된 동일
trace 2--3회 결과다. 이번 값은 vision-heavy만 3회 median이고 나머지는 1회다. latency 열은 `mean/p95 ms`다.

| workload | New tok/s | vs old | vs p10 | vs vLLM | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 2531.5 | -0.28% | +1.22% | +27.63% | 87.6/165.1 | 13.16/22.06 | 325.8/404.8 |
| balanced | 4514.6 | +1.49% | -1.09% | +4.51% | 61.2/167.3 | 12.24/13.96 | 1102.7/1685.6 |
| decode-heavy | 5265.2 | -0.77% | -0.95% | +8.46% | 64.1/185.2 | 10.56/11.46 | 2794.1/4293.2 |
| long-prefill | 1284.2 | +2.40% | +7.13% | +14.57% | 1981.9/2544.0 | 24.10/28.77 | 4030.5/5659.5 |
| bimodal | 1986.8 | +0.91% | +1.79% | +6.34% | 1807.4/3769.9 | 17.70/28.95 | 4190.0/8479.2 |
| text-heavy | 2084.0 | +3.16% | -1.27% | +27.48% | 390.3/1039.3 | 21.40/34.42 | 1508.7/1615.6 |
| mixed | 1088.3 | +3.41% | -2.77% | +18.10% | 711.0/2178.0 | 34.29/41.25 | 2304.7/2623.1 |
| vision-heavy | 640.3 | +2.46% | -9.89% | +10.56% | 1400.8/3451.5 | 36.90/49.80 | 2857.1/3781.5 |
| poisson | 1916.0 | -0.57% | -2.38% | +6.44% | 207.3/839.4 | 22.12/39.55 | 1631.0/2054.0 |
| wave-drain | 98.1 | +2.04% | -0.02% | +2.30% | 247.2/291.1 | 7.81/11.39 | 489.4/499.2 |
| multi-image | 321.1 | +1.37% | -0.60% | +31.30% | 227.0/290.0 | 8.56/11.72 | 492.5/498.1 |
| late-vision | 2551.5 | +0.04% | -0.13% | +8.15% | 121.0/419.7 | 9.22/9.27 | 1442.0/1808.1 |

요약:

| 기준 | throughput 기하평균 | New 승리 | -3% 이내 |
|---|---:|---:|---:|
| 이전 completion-active | +1.30% | 9/12 | 12/12 |
| p10 최고 구현 | -0.82% | 3/12 | 11/12 |
| frozen vLLM | +13.43% | 12/12 | 12/12 |

### 5.2 Latency 변화율

각 셀은 `old / p10 / vLLM` 대비 변화다. latency이므로 음수가 개선이다. vLLM에는 과거 aggregate에 mean이 없어 동일
run의 request CSV에서 run mean을 복원한 뒤 run median을 사용했다.

| workload | TTFT mean Δ | TTFT p95 Δ | TPOT mean Δ | TPOT p95 Δ | E2E mean Δ | E2E p95 Δ |
|---|---:|---:|---:|---:|---:|---:|
| short | -1.3%/+1.9%/-49.9% | -2.5%/-3.5%/-37.5% | +0.2%/-2.3%/-1.5% | +0.1%/-16.7%/-11.3% | +0.2%/-0.9%/-23.7% | +0.4%/-0.6%/-19.6% |
| balanced | -2.7%/-5.1%/-58.3% | +1.3%/-0.1%/-54.2% | -1.5%/+1.6%/-19.0% | -0.2%/+3.3%/-19.8% | -1.5%/+1.1%/-23.3% | -3.3%/-0.6%/-24.9% |
| decode-heavy | +0.5%/+0.6%/-58.2% | +5.8%/+2.0%/-53.0% | +0.7%/+1.0%/-24.6% | +3.0%/+3.4%/-23.7% | +0.9%/+1.1%/-25.9% | +0.9%/+1.8%/-26.1% |
| long-prefill | -1.6%/-5.1%/-32.7% | -3.8%/-1.2%/-40.2% | -3.1%/-8.4%/-25.7% | -6.1%/-6.4%/-23.6% | -2.4%/-6.8%/-29.2% | -1.5%/-6.1%/-28.0% |
| bimodal | -2.3%/-2.7%/-28.0% | -5.0%/-2.0%/-12.2% | +0.7%/-2.1%/-25.2% | +2.5%/-0.6%/-32.4% | -1.2%/-2.6%/-25.5% | -4.3%/-4.6%/-17.9% |
| text-heavy | +35.7%/+29.1%/-7.4% | -0.6%/+5.5%/-15.7% | -14.2%/-6.9%/-26.8% | -8.7%/-5.1%/-27.3% | -3.9%/+1.3%/-22.4% | -2.4%/+1.8%/-20.7% |
| mixed | -3.2%/+3.1%/-18.7% | -7.6%/+11.4%/-14.3% | -6.2%/-8.7%/-27.0% | -18.0%/-22.8%/-50.9% | -4.8%/-3.5%/-23.4% | -3.4%/+2.4%/-16.5% |
| vision-heavy | -0.3%/+6.5%/-18.1% | -1.1%/+17.9%/-6.5% | -3.1%/-21.1%/-42.1% | -4.0%/-27.5%/-58.4% | -1.6%/-8.4%/-30.6% | -2.3%/+10.8%/-10.6% |
| poisson | -0.2%/+6.1%/-52.7% | -0.0%/+14.4%/-7.0% | +2.0%/+2.6%/-0.3% | -0.7%/-0.8%/-13.4% | +2.0%/+3.8%/-9.4% | +2.1%/+2.6%/-9.4% |
| wave-drain | -14.0%/-3.8%/-2.2% | -33.4%/-31.9%/-30.4% | -7.2%/-7.7%/-37.1% | +1.8%/-4.1%/-34.0% | -11.2%/-4.8%/-23.3% | -22.1%/-21.0%/-23.1% |
| multi-image | -12.3%/-8.9%/-12.6% | -2.3%/+2.3%/-28.0% | +11.1%/+10.2%/-31.1% | +24.5%/+26.3%/-28.2% | -1.1%/+0.6%/-23.6% | -1.4%/+0.6%/-23.8% |
| late-vision | +1.0%/-0.1%/-21.1% | +0.0%/+0.1%/-33.5% | +0.2%/+0.3%/+24.1% | +0.2%/+0.4%/-6.7% | +0.3%/+0.3%/-8.5% | +0.2%/+0.4%/-7.5% |

중요한 해석은 다음과 같다.

1. throughput 개선만 보고 promotion하면 안 된다. `text-heavy`는 old 대비 throughput과 E2E/TPOT은 좋아졌지만 TTFT
   mean이 +35.7%다.
2. `vision-heavy`는 old 대비는 전반적으로 개선됐으나 p10 대비 TTFT/E2E tail과 throughput을 회복하지 못했다.
3. `multi-image`는 TTFT/E2E가 좋아졌지만 TPOT는 old/p10보다 나빠졌다. request 수가 5개뿐인 작은 trace라 반복 검증이
   특히 필요하다.
4. vLLM 대비 throughput 및 대부분 latency는 크게 앞서지만, `late-vision` TPOT mean 하나는 +24.1%다. p95는 -6.7%로
   더 낮아 분포 및 single-token vision rows의 metric contract를 함께 봐야 한다.

## 6. Direction별 calibration 결과

모든 VLM screening trace에서 실제 관측된 required direction은 4/4 complete가 되었다. 반대 E 방향은 candidate는 보였지만
실제 observation이 0이면 required로 세지 않는다.

| workload | warm-up | E->P | E->D | P->D | D->P |
|---|---:|---|---|---|---|
| text-heavy | 1696 | 42, valid | 28, rejected | 736, valid | 16, valid |
| mixed | 1696 | 47, rejected | 31, valid | 724, valid | 17, valid |
| vision-heavy | 1272 | 48, valid | 31, valid | 765, valid | 18, valid |
| poisson | 1696 | 48, valid | 30, valid | 874, valid | 14, rejected |
| wave-drain | 1272 | 38, valid | 30, valid | 536, valid | 16, valid |
| multi-image | 1272 | 39, valid | 28, valid | 480, valid | 16, valid |
| late-vision | 1696 | 41, valid | 31, valid | 761, valid | 20, rejected |

여기서 `valid`는 held-out coverage/false-safe validation을 통과했다는 뜻이지 반드시 policy blend가 1이라는 뜻이 아니다.
이번 screening에서는 aggregate empirical completion weight가 대부분 0이었다. component incumbent/newcomer weight는 일부
non-zero였지만, 전체 makespan 기준으로 scalar fallback보다 안전한 우위가 없으면 controller는 기존 판단을 유지했다.

이 결과는 두 가지를 동시에 의미한다.

- calibration mechanism은 이제 E까지 완주한다.
- E까지 sample을 모았다는 사실만으로 active physical predictor가 자동으로 우월해지는 것은 아니다.

따라서 다음 연구 질문은 "E sample이 있는가"가 아니라 "어떤 physical component가 scalar 대비 실제 decision regret을
줄이는가"다.

## 7. 검증

### 7.1 단위 테스트

- phase 관련: **347/347 통과**
- 전체 C++: 1,257 executed 중 **1,213 통과, 42 skip, 2 실패**
- 실패 2건은 변경 전부터 재현되는 SM86 수치 허용오차 문제다.
  - `InitializeYarnRopeCosSin.Accuracy`
  - `InitializeMRopeCosSin.Accuracy`
- scheduler/calibration 변경과 관련된 신규 실패는 없다.

### 7.2 정적 검사

- 변경한 C++/CUDA-facing 7개 파일에 대해 pre-commit 통과
- `clang-format`, license insertion, codespell, line-ending 검사 통과
- `git diff --check` 통과
- outer HTTP harness는 `python3 -m py_compile` 통과

### 7.3 결과 위치

- controlled calibration: `.local/completion-stage-calibration-p0-20260903/vision-heavy-*`
- 12-workload screen: `.local/completion-stage-calibration-p0-20260903/full12-screen`
- vision-heavy 3회: `.local/completion-stage-calibration-p0-20260903/vision-heavy-3x`
- 전체 GTest JSON: `.local/completion-stage-calibration-p0-20260903/full-unit-tests.json`

## 8. 현재 아키텍처의 의미

현재 구조는 workload별 tuning table이 아니다.

```text
workload name/profile                  사용하지 않음
exact CUDA key                        fallback/diagnostic
continuous physical predictor         E/P/D direction 결과 예측
conformal + held-out validation        authority 경계
deterministic transition evaluator     formation/ownership
SLO feasibility                        action 안전성
```

학습 시작점은 generic warm-up에서 얻은 현재 process-local evidence다. 각 workload 이름에 맞춘 파일을 불러오지 않는다.
하지만 지금 screening은 trace별로 새 process를 시작하고 그 trace를 반복해 warm-up했으므로, 엄밀히 말하면 cross-trace
continuous online adaptation 실험은 아니다. 다음 단계에서는 한 process에 서로 다른 trace epoch를 순서대로 넣어 posterior가
안정화되고 workload 전환을 따라가는지를 검증해야 한다.

## 9. 남은 한계

1. 11개 workload는 1회 screening이라 run variance를 정량화하지 못했다.
2. p10 대비 `vision-heavy -9.89%`는 -3% promotion gate를 통과하지 못했다.
3. validation pass와 useful authority가 다르다. E direction의 aggregate blend 0이 많은 이유를 component error와 action trace로
   분해해야 한다.
4. warm-up 1,272--1,696 requests는 연구 검증에는 허용 가능하지만 production startup 비용으로는 크다.
5. trace별 process reset이므로 posterior transfer, drift adaptation, cold-start amortization은 아직 검증하지 않았다.
6. default async adapter에서는 일부 VLM token trace가 실행 순서에 따라 달라질 수 있다. canonical diagnostic mode와 semantic
   correctness를 분리해 관리해야 한다.

## 10. 다음 구현 및 실험 순서

### P1. 12-workload 3회 promotion rerun

이번 1회 screen에서 p10 대비 -3% 밖인 `vision-heavy`를 우선 5회, 나머지는 3회로 재실행한다. throughput뿐 아니라
TTFT/TPOT/E2E mean·p95와 token/semantic correctness를 함께 gate한다.

### P2. Authority-use attribution

각 action에 다음을 기록한다.

```text
scalar selected completion
completion-vector selected completion
actual completion
component blend weights
selected action changed 여부
two-action regret
```

held-out valid인데 aggregate weight 0인 방향이 실제로 불필요한지, 아니면 projection/evaluator가 이득을 버리는지 구분한다.

### P3. Vision-heavy p10 gap 복구

새 workload rule을 추가하지 않는다. E/P/D action trace에서 다음 공통 state quantity만 비교한다.

- E formation fill과 action-induced fragmentation
- oldest first-token critical path/slack
- E completion -> P-ready visibility delay
- P/D continuity 및 small-D refill
- planned action과 actual outstanding phase mask

### P4. Warm-up sample efficiency

posterior/conformal/held-out minimum을 임의로 줄이지 않고, shared -> pair -> direction hierarchical shrinkage로 E sample을 줄인다.
424/848/1272/1696 request checkpoint에서 prediction error, false-safe, action agreement를 비교한다.

### P5. Cross-trace online stability

단일 process에서 text-heavy -> vision-heavy -> poisson -> mixed 순으로 workload label 없이 전환한다. 각 epoch에서:

- posterior/uncertainty 변화
- authority promote/demote
- action selection 변화
- scheduler CPU p95
- SLO goodput

을 측정한다. 목표는 학습 안정화 이후 frozen vLLM보다 높은 성능을 유지하면서 새 trace로의 전환에도 catastrophic
regression이 없는 것이다.

## 11. Promotion 판단

현재 bounded completion calibration은 mechanism으로는 promotion 가능하다. 무한 warm-up과 E+D starvation을 제거했고,
검증 실패 시 안전하게 fallback한다.

그러나 completion-vector active policy 전체를 production default로 올리는 판단은 보류한다.

```text
mechanism gate      PASS
phase unit tests    PASS
full12 regression  PASS vs previous completion-active
vLLM throughput    PASS in 12/12 frozen comparisons
p10 parity         FAIL only vision-heavy (-9.89%)
repeat confidence  INCOMPLETE (11 workloads are 1 run)
```

다음 기준은 `vision-heavy` gap 복구와 3--5회 반복이다. 이 두 조건을 통과한 뒤에야 E/P/D completion-vector active를
현재 최고 production policy로 승격한다.
