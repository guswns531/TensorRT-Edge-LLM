# Physical-Outcome Calibration Epoch 및 12-Workload 검증

날짜: 2026-09-03

브랜치: `codex/v010-phase-forward-port`

선행 문서:

- `notes/224-epd-completion-vector-active-tuning-20260903.md`
- `notes/225-minimal-physical-outcome-full12-gate-20260903.md`

## 1. 결론

이번 단계에서는 completion-vector의 수학 모델을 더 복잡하게 만들지 않고, 이미 구현된 physical-outcome
controller가 올바른 calibration epoch에서 학습되고 실제 serving에서만 사용되도록 lifecycle을 고쳤다.

핵심 변경은 다음 세 가지다.

1. calibration 종료 전에 E/P/D/Copy/setup stream을 모두 동기화하고 activity interval을 완전히 drain한다.
2. queue/SLO/dispatch telemetry는 measurement epoch에서 초기화하지만, held-out CUDA completion authority evidence는
   보존한다.
3. async HTTP adapter의 빠른 text completion-order는 production 기본값으로 유지하고, vision/barrier canonical ordering은
   출력 재현성 진단용 opt-in으로 분리한다.

최종 결론은 명확하다.

- `default adapter ordering + completion active`는 frozen vLLM보다 token throughput이 **12/12 workload에서 높다**.
- 하지만 p10 대비 기하평균은 `-2.09%`이고, `text-heavy -4.30%`, `mixed -5.98%`,
  `vision-heavy -12.05%`이므로 completion-vector active를 production 기본값으로 승격하지 않는다.
- phase-aware canonical ordering은 exact token repeat를 **12/12**로 만들지만, `mixed`, `vision-heavy`, `wave-drain`의
  E/P formation과 latency를 악화시킨다. 따라서 correctness diagnostic mode로만 유지한다.
- vision-heavy 5회 A/B에서 scalar `668.33`, shadow `690.42`, active `669.88 tok/s`였지만 각 모드 run range가
  `7.9--8.7%`다. shadow의 겉보기 이득은 causal policy gain이 아니다.
- completion authority는 사실상 P -> D 한 방향에만 충분하다. E -> P/E -> D는 calibration sample이 부족해
  scheduling authority를 얻지 못한다. 현재 결과를 full E/P/D physical controller의 승리로 표현하면 안 된다.

따라서 현재 production 선택은 다음과 같다.

```text
default serving
  = async completion-order adapter
  + scalar contextual/exact-cost policy
  + completion-vector shadow 가능

experimental active
  = 위 구조
  + held-out conformal validation을 통과한 direction/component만 blend

diagnostic exactness
  = TRT_EDGELLM_CANONICAL_VISION_ADAPTER_ORDER=1
```

## 2. 최종 아키텍처

### 2.1 execution prediction과 policy evaluation 분리

```text
                         Ready snapshot
                 E/P/D queues, slack, ownership
                              |
                              v
                    Hard feasibility filter
            dependency / TRT shape / memory / inflight
                              |
                 +------------+-------------+
                 |                          |
                 v                          v
        Exact/scalar cost             Physical outcome
       stable fallback prior        completion-vector RLS
                 |                 incumbent / newcomer
                 |                          |
                 |                  conformal uncertainty
                 |                          |
                 +------------+-------------+
                              |
                    validated component blend
                              |
                              v
                 Deterministic state transition
             next formation / KV / vision ownership
                              |
                              v
                    SLO-safe action selector
                              |
                              v
                      E/P/D dispatch lease
                              |
                              v
                       CUDA observations
```

RLS는 workload의 이름이나 최종 policy reward를 직접 외우지 않는다. pair의 물리적 completion을 예측하고, scheduler는
그 결과와 deterministic successor state를 결합한다. exact/scalar 모델은 제거하지 않고 검증되지 않은 direction의
fallback으로 남는다.

### 2.2 calibration/measurement epoch

이전 구현의 문제는 다음과 같았다.

```text
warm-up completion이 아직 GPU/host queue에 남음
        |
        +-- scheduling telemetry reset
        +-- authority evidence reset
        |
        v
measurement 시작
        |
        +-- 늦게 도착한 warm-up completion이 measurement state를 오염
        +-- 또는 검증 evidence가 0이 되어 active model이 실제로 authority를 얻지 못함
```

수정 후 contract는 다음과 같다.

```text
calibration submit 종료
        |
        +-- synchronize E stream
        +-- synchronize P stream
        +-- synchronize D stream
        +-- synchronize Copy stream
        +-- synchronize setup stream
        +-- drain activity timeline
        +-- assert pending activity == 0
        |
        v
immutable physical observation boundary
        |
        +-- exact CUDA cost 유지
        +-- RLS posterior 유지
        +-- conformal scale 유지
        +-- held-out authority evidence 유지
        +-- queue wait / SLO / dispatch telemetry만 reset
        |
        v
measurement epoch
```

즉 학습 state를 무조건 유지하는 것이 아니다. GPU에서 완료가 확인된 physical observation만 유지하고,
workload-specific serving history는 제거한다.

### 2.3 adapter ordering의 두 모드

HTTP JSON/image parsing은 CPU future로 수행된다. image parsing이 text request admission을 막지 않도록 기본값은 완료된
future를 먼저 drain한다.

```text
production default
arrival:       V0  T1  T2  V3  barrier
ready order:       T1  T2      V0  V3
admission:         T1  T2      V0  V3  barrier

canonical diagnostic
text:          느린 vision을 우회 가능
vision:        V0 -> V3 ingress order 유지
barrier:       이전 모든 adapter task 완료 후 통과
```

canonical mode는 `TRT_EDGELLM_CANONICAL_VISION_ADAPTER_ORDER=1`로 켠다. 기존
`TRT_EDGELLM_DISABLE_IPC_ADAPTER_OUT_OF_ORDER=1`도 호환 경로로 남긴다. 두 환경변수를 동시에 설정하면 즉시
실패시켜 모호한 contract를 허용하지 않는다.

canonical mode에서 같은 drain turn에 준비된 입력은 `ingressSequence`로 stable sort한다. 단, 느린 vision future 때문에
뒤의 text future를 막지는 않는다. 이것이 전체 FIFO보다 낮은 비용으로 vision E-order만 재현하는 phase-aware ordering이다.

## 3. 구현 위치

이번 변경은 `examples/llm/llm_phase_context_smoke.cpp` 한 파일에 국한된다.

| 위치 | 구현 |
|---|---|
| `PhaseIpcInput` | adapter ingress sequence 추가 |
| `PhaseIpcOrderingClass` | text / vision / barrier 분류 |
| shape warm-up 종료 | P/D/Copy/setup stream synchronization |
| encoder calibration 종료 | E/P/D/Copy/setup stream synchronization |
| `calibration_end` | activity timeline drain 및 pending interval invariant |
| epoch reset | scheduling history만 reset, completion authority evidence 보존 |
| async adapter drain | production completion-order와 phase-aware canonical mode 분리 |
| final telemetry | `vision_canonical`, `unrestricted_out_of_order`, bypass 수 보고 |

correctness와 policy authority의 경계를 다음 invariant로 고정했다.

```text
measurement epoch 시작
  => outstanding E/P/D/Copy activity interval == 0

canonical barrier dispatch
  => 모든 이전 adapter input이 이미 materialized

completion authority 사용
  => physical observation completed
     and conformal validation passed
     and component empirical error improved
```

## 4. 실험 계약과 artifact

모델과 엔진은 모든 비교에서 동일하다.

- model: `nvidia/Cosmos-Reason2-2B`
- engine: tied engine, P8/D64, fixed prefill chunk 128, indexed-paged KV
- GPU: RTX 3080 10 GiB
- HTTP request trace 및 client contract: p10과 동일
- warm-up: text 260 requests, VLM 424 requests
- workload repeat: 기본 3회, wave-drain 5회
- vLLM: request trace와 configuration이 바뀌지 않아 frozen 3회 결과 재사용

주요 artifact:

```text
.local/minimal-completion-default-fair-full12-20260903
.local/minimal-completion-phase-canonical-fair-full12-20260903
.local/minimal-completion-authority-canonical-fair-full12-20260903
.local/minimal-completion-authority-frozen-multi10-20260903
.local/minimal-completion-authority-canonical-frozen-full12-20260903
.local/minimal-completion-default-vision-modes-20260903
.local/p10-final-small-d-12x3-20260903/generic
.local/profile-free-global-20260827/r4-vllm-12x3
```

`frozen calibration` 실험은 한 process/posterior에서 measurement를 반복하므로 학습 안정성 진단에는 유효하지만,
매 repeat마다 server/context를 재생성하는 p10과의 code-only 비교에는 사용하지 않는다. 아래 main table은 fresh lifecycle의
fair gate만 사용한다.

## 5. 전체 12-workload 처리량 결과

아래 Current는 **default adapter ordering + completion conformal active**다. 양수는 Current가 빠르다는 뜻이다.

| Workload | Current tok/s | vs p10 | vs frozen vLLM | Exact repeat | Peak MiB | Run range |
|---|---:|---:|---:|:---:|---:|---:|
| short | 2538.74 | +1.51% | +27.99% | yes | 9237 | 2.24% |
| long-prefill | 1254.02 | +4.62% | +11.88% | yes | 9237 | 2.82% |
| bimodal | 1968.96 | +0.87% | +5.39% | yes | 9237 | 0.41% |
| balanced | 4448.21 | -2.54% | +2.97% | yes | 9237 | 4.72% |
| decode-heavy | 5306.15 | -0.18% | +9.31% | yes | 9237 | 1.09% |
| text-heavy | 2020.16 | -4.30% | +23.58% | yes | 9357 | 1.14% |
| mixed | 1052.38 | -5.98% | +14.21% | no | 9427 | 7.88% |
| poisson | 1926.95 | -1.82% | +7.05% | yes | 9373 | 8.22% |
| vision-heavy | 624.97 | -12.05% | +7.90% | no | 9427 | 3.11% |
| wave-drain | 96.09 | -2.02% | +0.26% | no | 9427 | 1.18% |
| late-vision | 2550.50 | -0.17% | +8.11% | yes | 9407 | 0.61% |
| multi-image | 316.74 | -1.94% | +29.53% | no | 9359 | 3.58% |

요약:

- p10 3% throughput gate: 9/12 통과. `text-heavy`, `mixed`, `vision-heavy` 실패.
- frozen vLLM throughput: 12/12 우세.
- p10 대비 geometric mean: `-2.085%`.
- Current peak memory: text `9237 MiB`, VLM `9357--9427 MiB`; OOM 없음.
- exact repeat: 8/12. production completion-order가 E/P batch membership을 바꾸는 네 VLM trace는 FP16 greedy
  boundary에서 다른 token branch를 만들 수 있다.

## 6. Current 절대 latency

단위는 ms다. mean은 per-run mean의 median, p95는 per-run p95의 median이다.

| Workload | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| short | 88.72 | 169.31 | 13.14 | 22.03 | 324.97 | 403.08 |
| long-prefill | 2013.65 | 2643.88 | 24.87 | 30.63 | 4130.91 | 5745.58 |
| bimodal | 1849.08 | 3968.56 | 17.59 | 28.24 | 4241.58 | 8864.43 |
| balanced | 62.87 | 165.12 | 12.42 | 13.98 | 1119.44 | 1742.47 |
| decode-heavy | 63.76 | 174.92 | 10.49 | 11.13 | 2768.79 | 4253.36 |
| text-heavy | 287.53 | 1045.58 | 24.93 | 37.70 | 1569.29 | 1655.24 |
| mixed | 734.73 | 2356.62 | 36.56 | 50.28 | 2421.95 | 2714.81 |
| poisson | 207.62 | 839.57 | 21.69 | 39.82 | 1599.10 | 2012.29 |
| vision-heavy | 1404.47 | 3490.35 | 38.07 | 51.88 | 2904.74 | 3869.51 |
| wave-drain | 287.36 | 437.42 | 8.42 | 11.19 | 551.34 | 641.17 |
| late-vision | 119.75 | 419.66 | 9.20 | 9.25 | 1438.25 | 1804.10 |
| multi-image | 258.96 | 296.93 | 7.71 | 9.42 | 497.90 | 504.90 |

## 7. p10 대비 latency 변화

음수는 개선이다.

| Workload | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| short | +3.24% | -1.01% | -2.52% | -16.79% | -1.14% | -1.01% |
| long-prefill | -3.62% | +2.73% | -5.48% | -0.39% | -4.46% | -4.66% |
| bimodal | -0.47% | +3.13% | -2.74% | -3.07% | -1.43% | -0.27% |
| balanced | -2.44% | -1.43% | +3.13% | +3.50% | +2.62% | +2.74% |
| decode-heavy | +0.03% | -3.61% | +0.27% | +0.47% | +0.15% | +0.83% |
| text-heavy | -4.92% | +6.15% | +8.44% | +3.98% | +5.42% | +4.26% |
| mixed | +6.59% | +20.55% | -2.66% | -5.94% | +1.41% | +6.02% |
| poisson | +6.32% | +14.46% | +0.59% | -0.08% | +1.80% | +0.52% |
| vision-heavy | +6.75% | +19.19% | -18.58% | -24.46% | -6.90% | +13.42% |
| wave-drain | +11.86% | +2.34% | -0.64% | -5.79% | +7.20% | +1.47% |
| late-vision | -1.11% | +0.11% | +0.17% | +0.18% | +0.04% | +0.14% |
| multi-image | +3.87% | +4.73% | -0.80% | +1.47% | +1.71% | +1.96% |

completion active의 가장 분명한 trade-off는 `vision-heavy`다. resident decode TPOT는 크게 좋아지지만 E -> P first-token
critical path와 E2E tail이 악화된다. 이는 scalar reward 하나의 문제가 아니라, 현재 실제 authority가 P -> D에 집중되어
있어 decode continuity의 물리 예측만 정책에 강하게 반영되기 때문이다.

## 8. frozen vLLM 대비 latency 변화

vLLM artifact에는 mean이 없으므로 median/p95를 비교한다. 음수는 Current가 빠르다.

| Workload | TTFT median | TTFT p95 | TPOT median | TPOT p95 | E2E median | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| short | -64.80% | -35.86% | -6.00% | -11.48% | -23.89% | -19.98% |
| long-prefill | -30.96% | -37.90% | -23.74% | -18.66% | -25.35% | -26.91% |
| bimodal | -35.51% | -7.53% | -22.97% | -34.10% | -16.69% | -14.17% |
| balanced | -59.01% | -54.77% | -19.80% | -19.64% | -17.96% | -22.35% |
| decode-heavy | -60.46% | -55.63% | -26.89% | -25.93% | -25.81% | -26.82% |
| text-heavy | -50.67% | -15.14% | -18.57% | -20.40% | -19.63% | -18.77% |
| mixed | +7.52% | -7.27% | -15.56% | -40.16% | -10.50% | -13.56% |
| poisson | -79.54% | -6.99% | +3.77% | -12.82% | -11.92% | -11.22% |
| vision-heavy | -18.06% | -5.45% | -40.22% | -56.62% | -24.55% | -8.51% |
| wave-drain | +11.61% | +4.50% | -38.03% | -35.20% | -13.75% | -1.27% |
| late-vision | -26.05% | -33.55% | -7.37% | -6.83% | -8.02% | -7.69% |
| multi-image | +14.69% | -26.24% | -41.68% | -42.30% | -21.93% | -22.79% |

Current는 vLLM 대비 E2E median/p95 모두 12/12 우세다. 남은 열세는 mixed/multi-image의 TTFT median,
wave-drain의 TTFT median/p95, poisson의 TPOT median이다. 따라서 “모든 지표에서 vLLM 우세”는 아니지만,
throughput과 E2E는 전체 suite에서 우세하다.

## 9. canonical ordering A/B

phase-aware canonical mode는 같은 active completion policy에서 token repeat를 12/12로 만들었다.

| Workload | Default-order tok/s | Phase-canonical tok/s | Canonical vs p10 | Canonical exact |
|---|---:|---:|---:|:---:|
| short | 2538.74 | 2512.02 | +0.44% | yes |
| long-prefill | 1254.02 | 1197.42 | -0.11% | yes |
| bimodal | 1968.96 | 1956.03 | +0.21% | yes |
| balanced | 4448.21 | 4522.76 | -0.91% | yes |
| decode-heavy | 5306.15 | 5356.99 | +0.78% | yes |
| text-heavy | 2020.16 | 2052.98 | -2.74% | yes |
| mixed | 1052.38 | 1071.25 | -4.29% | yes |
| poisson | 1926.95 | 2012.55 | +2.54% | yes |
| vision-heavy | 624.97 | 657.99 | -7.40% | yes |
| wave-drain | 96.09 | 93.80 | -4.36% | yes |
| late-vision | 2550.50 | 2549.91 | -0.20% | yes |
| multi-image | 316.74 | 311.23 | -3.65% | yes |

canonical mode의 p10 대비 geometric mean은 `-1.68%`지만 workload별 gate는 `mixed`, `vision-heavy`,
`wave-drain`, `multi-image`에서 실패한다. 특히 wave-drain TTFT mean `+69.85%`, E2E mean `+42.46%`는 단순
수치 오차가 아니다. E ready 순서를 고정하면서 admission/formation 경계가 바뀐 결과다.

따라서 다음 두 목표는 같은 knob로 해결할 수 없다.

```text
maximum request-adapter parallelism and batching efficiency
                         !=
cross-run exact greedy token identity
```

KV ownership corruption이나 OOB의 증거는 없다. 동일 canonical execution에서 exact hash가 12/12 반복되므로,
non-canonical mode의 divergence는 request row/batch membership과 FP16 reduction/tactic order가 greedy boundary를
바꾸는 numerical determinism 문제로 해석한다.

## 10. vision-heavy scalar/shadow/active 5회 A/B

동일 engine, trace, warm-up 계약에서 process를 mode별로 하나씩 띄우고 measurement를 5회 반복했다.

| Mode | Median tok/s | Run values tok/s | Range | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| scalar-only | 668.33 | 706.86, 705.09, 668.33, 650.54, 660.70 | 8.66% | 1347.56 | 3179.73 | 35.83 | 55.07 | 2804.71 | 3606.20 |
| completion-shadow | 690.42 | 690.42, 706.10, 691.80, 685.32, 654.16 | 7.94% | 1314.01 | 3115.16 | 33.84 | 50.68 | 2648.48 | 3445.42 |
| completion-active | 669.88 | 669.88, 685.41, 672.35, 654.46, 634.20 | 8.07% | 1341.93 | 3221.66 | 34.52 | 48.01 | 2715.59 | 3584.95 |

shadow는 action selection을 바꾸지 않으므로 scalar 대비 `+3.30%` median을 model benefit으로 해석할 수 없다. 이 차이는
vision formation/runtime variance의 크기를 보여주는 control이다. active는 scalar 대비 throughput `+0.23%`,
TTFT mean `-0.42%`, E2E mean `-3.18%`, TPOT p95 `-12.81%`지만 TTFT p95는 `+1.32%`다. run range가 효과보다
훨씬 크므로 승격 근거가 아니다.

## 11. completion authority가 실제로 어디까지 작동했는가

calibration evidence를 reset하지 않게 고친 뒤 P -> D authority는 실제로 measurement에 전달된다.

대표 vision-heavy run의 calibration 종료 상태:

```text
P -> D
  authority window          128
  validated                 true
  incumbent blend           0.727
  newcomer blend            0.534

D -> P
  authority window          7
  validated                 false

E -> P
  contextual observations  16
  completion ready          12
  authority window          0
  validated                 false

E -> D / reverse E directions
  authority window          0
  validated                 false
```

P -> D는 workload마다 충분한 evidence를 얻지만 balanced의 일부 run에서는 conformal coverage gate를 통과하지 못해
validation이 달라진다. text-heavy/mixed/poisson/vision-heavy에서는 P -> D window가 대부분 128까지 찬다.

반면 E direction은 posterior가 겨우 ready가 된 시점에 calibration이 끝난다. uncertainty calibrator가 준비된 뒤 남는
held-out completion이 0--1개이므로 authority window가 차지 않는다. 따라서 현재 active 결과에서 E completion-vector가
vision scheduling을 직접 개선했다고 주장할 수 없다.

이 결과는 architecture를 폐기할 이유가 아니라 calibration contract의 미완성을 보여준다. 다음 단계는 warm-up request
수를 무작정 늘리는 것이 아니라, E pair posterior/calibrator/held-out authority를 서로 다른 sample split으로 명시하고
각 direction의 최소 evidence를 충족할 때까지만 bounded probe를 배치하는 것이다.

## 12. 검증 결과

- TensorRT/CUDA build: 성공.
- phase changed-scope C++ tests: **311/311 passed**.
- full C++ suite: **1211 passed, 42 skipped**.
- 기존 SM86 numerical tolerance failure 2개:
  - `InitializeYarnRopeCosSin.Accuracy`
  - `InitializeMRopeCosSin.Accuracy`
- 12-workload HTTP inference: 모든 request가 완료되고 requested output token 수를 충족.
- CUDA/TensorRT runtime crash, invalid slot, OOM: 없음.
- peak memory: 최대 9427 MiB로 10 GiB GPU에서 실행 성공.

두 full-suite failure는 이번에 수정한 IPC/calibration code와 무관하며 직전 기준에서도 동일했다.

## 13. Promotion 결정

### Production 기본값

다음을 유지한다.

- completion-order async adapter: 기본 on.
- scalar contextual/exact cost: scheduling authority.
- completion conformal/vector: 기본 off 또는 shadow.
- canonical vision ordering: 기본 off, diagnostic opt-in.

### 승격하지 않는 것

- completion-vector full active를 default로 승격하지 않는다.
- canonical ordering을 performance default로 승격하지 않는다.
- `vision-heavy`나 `multi-image` 이름을 확인하는 workload-specific rule을 추가하지 않는다.

### 유지할 구현

- hard calibration epoch drain/assertion.
- completed physical evidence retention.
- direction/component별 conformal validation과 empirical blend.
- phase-aware canonical diagnostic mode.
- scalar fallback과 deterministic formation/ownership evaluator.

## 14. 다음 작업

### P0 — E authority를 위한 bounded calibration split

```text
fit samples
    -> uncertainty calibration samples
        -> held-out authority samples
```

각 E -> P, P -> E, E -> D, D -> E direction이 최소 evidence를 얻도록 generic shape probe를 구성한다. workload
label은 입력으로 사용하지 않는다. calibration이 수렴하지 않으면 해당 direction은 shadow에 남긴다.

### P1 — requested/realized action contract

requested start skew와 실제 first-kernel start skew를 direction별로 저장하고, realization error를 completion-vector label에서
분리한다. 별도 거대 모델을 추가하기 전에 현재 telemetry로 둘의 상관관계를 먼저 측정한다.

### P2 — active authority의 policy-only causal trace

같은 physical posterior와 같은 ready snapshot을 replay하여:

```text
scalar selected action
completion selected action
actual two-action horizon
protected request completion
```

을 비교한다. selection change가 없으면 그 workload의 active 효과를 주장하지 않는다.

### P3 — vision formation variance 감소

arrival sequence를 강제하지 않고도 같은 GPU-ready E set에서 canonical batch identity를 만드는 방법을 구현한다. adapter
완료 순서와 scheduler batch identity를 분리하여 text bypass는 유지하고 E batch membership만 stable하게 만든다.

### P4 — 12-workload promotion gate 재실행

다음 조건을 모두 만족할 때만 completion active를 승격한다.

- 모든 workload throughput이 p10 대비 3% 이내.
- workload별 protected TTFT/TPOT/E2E p95가 3% 이상 악화되지 않음.
- E authority가 실제로 selection에 사용되며 false-safe gate를 만족.
- 동일 HTTP trace에서 frozen vLLM throughput/E2E 우세 유지.
- no OOM, no ownership invariant failure.

## 15. 최종 판단

physical-outcome architecture 자체는 더 명확해졌다. RLS가 직접 workload policy를 학습하는 대신 GPU completion을
예측하고, deterministic transition과 SLO evaluator가 action 가치를 계산하는 구조는 유지할 가치가 있다. 이번 단계에서
새로 확인된 한계는 모델 복잡도가 아니라 **authority data lifecycle과 E sample coverage**다.

현재 가장 좋은 배포 선택은 여전히 p10 계열 scalar/exact controller다. completion-vector는 P -> D에서 유용한 TPOT
개선 신호를 보였지만, full active는 p10의 broad gate를 통과하지 못했다. 다음 단계는 새로운 휴리스틱이나 더 큰 모델이
아니라, E direction을 포함한 bounded calibration split과 action-realization telemetry를 완성하여 같은 architecture 안에서
실제 causal selection gain을 만드는 것이다.
