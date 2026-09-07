# Multi-image SLO contract, frozen branch replay, and full12 gate

## 1. 목적

이 단계는 5-request multi-image trace가 약 `203--326 tok/s`의 양봉 분포를 보인 원인을 request lifecycle까지
분해하고, 다음 세 가설을 순서대로 검증했다.

1. P/D bounded formation이 실제 serial P frontier를 빠뜨리는가?
2. 첫 decode row를 잘못된 SLO로 보호하는가?
3. 동일 decode SLO가 three-phase coordinator와 semantic P/D queue에서 다르게 해석되는가?

workload 이름, image 수, 특정 batch shape를 production action rule로 사용하지 않았다.

## 2. 구현

### 2.1 Measured request lineage

`benchmarks/phase_serving/analyze_multi_image_trajectory.py`가 full telemetry의 마지막 measured lifecycle을 복구한다.
calibration과 measured trace가 request ID를 다시 0부터 할당하므로 단순 ID 필터가 아니라 각 ID의 마지막 complete
lifecycle을 선택한다.

출력은 다음을 포함한다.

- E/P/D 실제 cohort와 request membership
- first-token-ready부터 첫 D 시작까지의 시간
- 첫 D 전에 ready였던 request 수
- 연속 D1 prefix 길이
- 첫 full D cohort까지의 시간

`run_policy_warmup_matrix.py`는 명시적 backend environment override를 telemetry 기본값보다 나중에 적용한다. 따라서
counterfactual activity plumbing을 유지하면서 `full` request-lineage level을 정확히 요청할 수 있다.

### 2.2 Frozen completion-order branch replay

`PhaseFrozenBranchComparison`과 `phaseCompareFrozenOutcomeBranches()`를 추가했다. 동일한 immutable snapshot과 동일한
action ID에 두 개의 단일 physical completion outcome을 적용하고 다음을 비교한다.

```text
same frozen state + same action
        |                         |
        v                         v
  E finishes first          P/D finishes first
        |                         |
        +---- logical DAG/ownership signatures ----+
                              |
                              v
             first divergence and terminal reconvergence
```

logical signature에는 request stage, stable KV slot, remaining decode steps, generated-token count, vision/KV ownership,
ready state와 outstanding phase membership이 포함된다. elapsed time은 제외해 completion magnitude와 completion-order
transition을 분리한다.

### 2.3 Decode-formation frontier parity

P/D queue의 observable `final P -> D-ready` horizon은 overlap 전용 P frontier가 아니라 실제 serial P candidate를
사용한다. 예를 들어 serial P2와 overlap P1이 동시에 가능할 때 P2가 생성할 D2를 보존한다. successor D shape에
direct exact key가 없으면 trusted covering context estimate를 사용하되, 신뢰 가능한 두 successor cost가 모두 있을
때만 formation horizon에 authority를 준다.

### 2.4 Single decode-service contract

가장 큰 실제 문제는 구성 계층 간 SLO 불일치였다.

```text
TRT_EDGELLM_VISION_DECODE_TPOT_TARGET_MS=80
        |
        +--> PhaseThreeCoordinator: 80 ms
        |
        `--> PhaseQueueScheduler:    20 ms default   (before)
```

이 상태에서는 coordinator가 producer progress에 80ms 여유가 있다고 보더라도, semantic P/D selector는 새로 생긴
D1을 20ms마다 보호했다. P 한 번이 약 `27--36ms`이므로 D1/D1 alternating trajectory가 만들어지고 남은 P rows가
오랫동안 막혔다.

composition root에서 millisecond vision option과 microsecond global option의 precedence를 동일하게 적용했다.

```text
request metadata tpot_target_ms (per-row override)
                 |
                 v
configured common decode target
  VISION_DECODE_TPOT_TARGET_MS, otherwise GLOBAL_DECODE_TPOT_TARGET_US
                 |
        +--------+--------+
        v                 v
 three-phase pressure   semantic P/D deadline
```

이 변경은 workload별 tuning이 아니다. 같은 request/SLO contract를 두 policy layer가 동일하게 읽도록 만든
correctness/configuration fix다.

## 3. Causal 분석

### 3.1 Fragmented trajectory

P17 full telemetry의 나쁜 run은 다음과 같다.

```text
E: E4([0,1,2,3]) -> E1([4])
P: P1([4]) -> P1([3]) -> P1([1]) -> P2([0,2])
D: D1([4]), D1([3]), ... 23 singleton dispatches -> D3 -> D5
```

- token throughput: `244.46 tok/s`
- D dispatches: `55`
- D GPU time: `355.88ms`
- first D ready-to-start: `0.14ms`
- singleton prefix: `23`

### 3.2 Coherent trajectory

같은 trace의 좋은 run은 P가 진행되는 동안 첫 D request의 다음 sampling completion이 늦게 visible해져 후속 rows가
합쳐졌다.

- token throughput: `321--323 tok/s`
- D dispatches: `33`
- D GPU time: `221--235ms`
- singleton prefix: `0--1`

따라서 batch size는 RLS feature에 포함되어 있어도 이 실패를 직접 해결하지 못한다. RLS가 action 시점의 batch
shape와 cost ratio를 보지만, 서로 다른 계층의 deadline contract가 D1을 hard-safe action으로 만들면 learned score가
그 후보를 뒤집을 권한이 없기 때문이다.

### 3.3 기각한 first-D 가설

처음에는 첫 D가 first-token path이므로 TTFT로 보호해야 한다는 실험을 했지만 runtime contract를 추적한 뒤
기각했다. final prefill completion callback이 이미 sampling을 제출하고 첫 token을 만든다. 이후 D queue로 들어오는
row는 두 번째 output token이므로 TPOT 보호가 맞다. 관련 실험 코드는 최종 변경에서 제거했다.

이 과정에서 얻은 중요한 구분은 다음과 같다.

```text
incorrect fix: D의 의미를 TTFT로 재분류
correct fix:  같은 TPOT target을 모든 policy layer에 전달
```

## 4. Multi-image 반복 결과

모든 결과는 Cosmos-Reason2-2B, P128, P8/D64/E4, max-in-flight 64, 동일 v7-small-d generic calibration과
동일 HTTP trace/output contract를 사용한다.

### 4.1 SLO 통일 전

| Policy | runs | 좋은 trajectory | 중앙 token/s | 최저 token/s |
|---|---:|---:|---:|---:|
| Scalar production, P17/P18 계열 | 8 | 변동 | 약 256--322 | 203--244 |
| Scalar+Transition, P20 | 5 | 4 | 322.74 | 202.97 |

기존 H2는 좋은 run 비율을 높일 수 있지만 실패를 제거하지 못했다.

### 4.2 SLO 통일 후

| Policy | runs | token/s 중앙값 | 범위 | TTFT mean/p95 중앙 ms | TPOT mean/p95 중앙 ms | E2E mean/p95 중앙 ms |
|---|---:|---:|---:|---:|---:|---:|
| Scalar production, P21 | 5 | **323.84** | 292.61--324.52 | 259.06 / 285.15 | 7.46 / 8.62 | 490.81 / 493.55 |
| Scalar+Transition, P22 | 5 | 318.98 | 203.05--324.68 | 258.05 / 293.72 | 7.88 / 9.32 | 494.38 / 501.35 |

production Scalar는 catastrophic `203--244 tok/s` trajectory를 5/5에서 피했다. Scalar+Transition은 1/5에서
다시 `203.05 tok/s`로 떨어졌다. 따라서 H2 authority는 계속 비활성으로 두고 physical/frozen replay 진단에만
사용한다.

## 5. Full12 production gate

P23의 12-workload 1회 결과를 사용하고, variance가 컸던 Poisson은 P24 3회 중앙값으로 대체했다. 이전 champion은
note 234의 workload별 최종/반복값이며 frozen vLLM은 request trace와 runtime contract가 변하지 않아 재실행하지
않았다.

| workload | Current tok/s | 이전 champion 대비 | frozen vLLM 대비 | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| balanced | 4545.8 | +1.45% | +5.41% | 65.9 / 168.5 | 12.08 / 13.43 | 1095.1 / 1692.4 |
| bimodal | 1940.0 | +2.29% | +5.43% | 1901.5 / 4192.6 | 17.65 / 29.60 | 4320.1 / 8932.6 |
| decode-heavy | 5267.0 | +1.31% | +6.18% | 63.3 / 181.0 | 10.57 / 11.18 | 2793.4 / 4264.3 |
| late-vision | 2554.5 | +0.43% | +8.40% | 120.9 / 422.9 | 9.24 / 9.31 | 1444.8 / 1804.7 |
| long-prefill | 1237.1 | +6.69% | +10.56% | 2036.8 / 2543.0 | 25.30 / 31.09 | 4187.8 / 5933.6 |
| mixed | 1181.6 | +1.80% | +58.03% | 653.0 / 2059.5 | 37.52 / 56.21 | 2340.2 / 2466.7 |
| multi-image | 320.4 | +1.54% | +34.83% | 248.0 / 290.8 | 7.87 / 9.29 | 492.1 / 498.8 |
| poisson, 3x median | 1941.4 | -2.24% | +8.28% | 254.6 / 891.9 | 21.40 / 37.41 | 1608.4 / 2027.8 |
| short | 2503.9 | +2.56% | +25.99% | 86.2 / 177.0 | 13.37 / 26.10 | 327.5 / 406.6 |
| text-heavy | 2117.1 | +2.49% | +29.37% | 375.0 / 937.4 | 21.65 / 34.91 | 1483.3 / 1590.7 |
| vision-heavy | 734.7 | +4.88% | +27.62% | 1277.2 / 2964.6 | 49.05 / 77.95 | 3133.5 / 3316.8 |
| wave-drain | 98.0 | -0.15% | +2.45% | 280.9 / 585.8 | 8.58 / 13.95 | 546.9 / 785.9 |

요약:

- 이전 champion 대비 throughput geometric mean: **+1.90%**
- 이전 champion 대비 12/12가 `-3%` regression gate 통과, 10/12는 더 빠름
- frozen vLLM 대비 throughput geometric mean: **+17.54%**
- frozen vLLM throughput: **12/12 우세**
- greedy token hash: 모든 measured repeat에서 동일
- peak memory: `9495MiB` 이하

P23은 single run이므로 paper confidence interval이 아니다. 다만 multi-image 5회와 Poisson 3회가 가장 큰 두
variance 위험을 보완한다.

## 6. 검증

```text
C++ incremental build:
  llm_phase_context_smoke, unitTest PASS

C++ targeted tests:
  14/14 PASS
  - PhaseIncrementalProjectorTest.*
  - PhaseFrozenReplayTest.*
  - GlobalDecodeFormationUsesSerialPrefillWhenOverlapFrontierDiffers

Python tests:
  16/16 PASS
  - test_multi_image_trajectory.py
  - test_policy_warmup_matrix.py
```

## 7. 최종 architecture 판단

현재 production winner는 계속 `Contextual Scalar + deterministic feasibility/SLO selector`다.

```text
request DAG + E/P/D ready snapshot
              |
              v
deterministic candidate mechanism
  dependency / TRT shape / single inflight / stable ownership
              |
              v
one shared request-level SLO contract
              |
              +-----------------------+
              v                       v
exact CUDA execution registry    contextual Scalar RLS
                                  batch fill/cost ratio/
                                  context/slack features
              |                       |
              +-----------+-----------+
                          v
                 SLO-safe global selector
                          |
                          v
                    E/P/D dispatch
```

Frozen completion-vector and H2 transition components는 production authority가 아니라 causal validation substrate다.
더 복잡한 physical model이 반드시 더 좋은 policy를 만들지는 않는다. 이번 결과에서 가장 큰 이득은 model fidelity가
아니라 policy layers가 동일한 SLO contract를 사용하게 한 데서 나왔다.

## 8. 다음 계획

1. multi-image와 Poisson을 5--10회로 늘려 confidence interval을 확정한다.
2. configured target과 per-request metadata target을 startup/decision telemetry에 명시해 향후 split-brain을 자동 검출한다.
3. `Scalar+Transition`의 1/5 regression을 frozen branch replay로 재현하고 H2가 실제로 선택을 바꾼 경우만 causal하게
   분석한다. production authority는 그 전까지 비활성이다.
4. 48.8 req/s saturation을 공통 TPOT contract로 다시 5회 확인한다.
5. 동일 HTTP contract의 최종 paper run에서는 Current와 vLLM을 모두 fresh 5회 실행하고 bootstrap confidence
   interval, joint-SLO goodput, memory를 함께 보고한다.
6. 디스크 headroom 확보 후 selected multi-image run 하나만 full Nsight/lineage로 수집한다.

