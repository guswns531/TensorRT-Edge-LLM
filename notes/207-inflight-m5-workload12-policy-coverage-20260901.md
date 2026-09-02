# In-flight M5: 12-workload policy coverage and promotion decision

## 1. Outcome

M5의 작은 4-image pilot을 기존 frozen 12-workload 실제 HTTP request trace로 확장했다. 동일 binary, engine,
request trace, admission, warmup, phase calibration 및 CUDA-event 계측 계약을 유지하고 selector만 다음처럼 바꿨다.

```text
myopic production selector
        versus
bounded safe-probe selector
```

이번 실행의 목적은 safe-probe를 새 기본 정책으로 만드는 것이 아니라 동일한 immutable execution snapshot에서 서로 다른
legal action의 measured completion vector를 확보하는 것이다.

핵심 결론은 다음과 같다.

- 두 정책 모두 unified action fidelity와 dispatch/completion 무결성을 통과했다.
- exact multi-action snapshot은 작은 pilot의 4개에서 12-workload의 108개로 증가했다.
- safe-probe는 일부 VLM workload를 개선했지만 text-heavy formation workload를 크게 회귀시켰다.
- 12-workload 기하평균에서 safe-probe는 generated token/s가 `-0.40%`, E2E mean이 `+1.24%` 개선됐다.
  하나의 명확한 우세 정책이라고 볼 수 없다.
- cross-policy greedy token hash는 `7/12` workload만 일치했다. 따라서 production promotion gate는 실패다.
- empirical H1에서 myopic의 agreement/regret이 safe-probe보다 낫다. 하지만 mixed처럼 H1이 나빠도 full-trace E2E가
  좋아지는 workload가 있어 H1만을 production objective로 사용해서도 안 된다.

```text
safe-probe = useful counterfactual data collector
safe-probe != production policy
```

## 2. Experimental contract

소스 manifest:

```text
.local/current-only-cleanup-20260828/
  stage7-final-current-12x3/commands.json
```

실행기:

```text
benchmarks/phase_serving/run_oracle_h1_policy_matrix.py
```

공통 조건:

- model: `nvidia/Cosmos-Reason2-2B`
- same v0.10 forward-port binary and TensorRT engines
- same materialized JSON/image traces
- same HTTP adapter and output-token capture
- one measured repeat per workload in this coverage survey
- 64-request warmup and existing phase calibration contract
- `TRT_EDGELLM_EMIT_PHASE_METRICS=1`
- per-run `TRT_EDGELLM_PHASE_ACTIVITY_PREFIX`
- CUDA common-epoch start/end intervals
- exact immutable snapshot signature including ready request lineage, progress, ownership state, and in-flight work

이 결과는 one-repeat coverage survey다. 최종 performance promotion 수치가 아니며 선택된 점은 이후 반복해야 한다.

## 3. Unified-event correctness

| Metric | Myopic | Safe-probe |
|---|---:|---:|
| Decisions | 9,126 | 9,785 |
| Executions | 9,508 | 10,392 |
| GPU intervals | 9,508 | 10,392 |
| Dispatch without completion | 0 | 0 |
| Completion without dispatch | 0 | 0 |
| Action-fidelity failures | 0 | 0 |
| P→D overlap dispatches | 315 | 1,447 |
| E→D overlap dispatches | 22 | 33 |
| E→P overlap dispatches | 49 | 29 |

Safe-probe는 myopic보다 decision 수가 `+7.2%`, execution 수가 `+9.3%` 많고 P→D overlap은 약 `4.6x`
많다. 이 증가는 단순히 GPU overlap opportunity를 더 쓴 것이 아니라 일부 workload에서 batch fragmentation과 추가
dispatch를 만든다. 따라서 probe 횟수 자체를 성능 향상으로 해석하면 안 된다.

## 4. Full 12-workload request metrics

아래 `M → S`는 myopic에서 safe-probe로의 변화다. 모든 latency는 ms다.

| Workload | gen tok/s M→S | TTFT mean / p95 M→S | TPOT mean / p95 M→S | E2E mean / p95 M→S | Exact output |
|---|---:|---:|---:|---:|:---:|
| short | 1997.22 → 1954.70 | 127.26 / 201.44 → 128.17 / 209.84 | 15.17 / 24.68 → 15.21 / 24.88 | 416.58 / 514.87 → 416.47 / 510.56 | yes |
| balanced | 3192.16 → 3051.63 | 90.32 / 188.34 → 82.33 / 205.55 | 17.48 / 19.40 → 18.49 / 20.48 | 1580.43 / 2465.47 → 1659.23 / 2611.65 | yes |
| decode-heavy | 3751.01 → 3656.54 | 96.58 / 208.40 → 83.19 / 237.56 | 14.95 / 15.93 → 15.46 / 16.41 | 3955.10 / 6105.03 → 4071.24 / 6229.79 | yes |
| long-prefill | 1040.03 → 968.45 | 2350.69 / 2963.91 → 2539.36 / 3062.04 | 30.44 / 35.77 → 32.94 / 37.60 | 4971.51 / 6927.52 → 5352.91 / 7493.24 | yes |
| bimodal | 1619.82 → 1563.06 | 2184.74 / 4347.79 → 2307.51 / 5161.64 | 20.97 / 32.13 → 21.40 / 30.79 | 5119.97 / 10214.75 → 5339.81 / 11647.51 | yes |
| text-heavy | 1498.91 → 1541.07 | 609.81 / 1357.52 → 621.15 / 1365.98 | 28.81 / 42.93 → 26.74 / 38.37 | 2074.01 / 2208.33 → 1980.83 / 2134.83 | no |
| mixed | 955.18 → 1002.63 | 962.01 / 2523.82 → 914.69 / 2341.74 | 35.83 / 43.81 → 34.19 / 41.34 | 2639.34 / 2994.08 → 2503.61 / 2845.36 | no |
| vision-heavy | 601.22 → 629.75 | 1595.47 / 3559.11 → 1534.78 / 3381.80 | 33.52 / 40.78 → 30.84 / 39.25 | 2896.12 / 4025.61 → 2743.38 / 3833.29 | no |
| poisson | 1595.08 → 1627.26 | 360.95 / 953.88 → 360.30 / 942.69 | 25.63 / 45.66 → 24.46 / 42.42 | 1998.81 / 2556.64 → 1926.02 / 2454.94 | yes |
| wave/drain | 94.94 → 96.35 | 353.40 / 516.10 → 295.37 / 366.86 | 10.84 / 14.54 → 10.30 / 13.02 | 689.50 / 767.70 → 614.60 / 630.91 | no |
| multi-image | 256.01 → 246.42 | 329.75 / 375.01 → 260.49 / 390.64 | 9.25 / 10.67 → 11.46 / 14.15 | 616.38 / 624.82 → 615.70 / 647.73 | no |
| late-vision | 2011.06 → 2076.38 | 157.52 / 564.58 → 143.74 / 508.18 | 11.76 / 11.83 → 11.38 / 11.45 | 1842.05 / 2293.90 → 1774.46 / 2221.26 | yes |

Positive improvement 기준 집계:

| Metric | Geometric-mean improvement | Safe-probe wins |
|---|---:|---:|
| generated token/s | -0.40% | 6 / 12 |
| TTFT mean | +5.53% | 8 / 12 |
| TTFT p95 | +0.61% | 5 / 12 |
| TPOT mean | -0.59% | 6 / 12 |
| TPOT p95 | +0.35% | 7 / 12 |
| E2E mean | +1.24% | 8 / 12 |
| E2E p95 | +0.73% | 7 / 12 |

Exact-output 7-workload만 보면 safe-probe는 generated token/s `-2.08%`, E2E mean `-1.71%`, E2E p95
`-2.99%`다. 정확성 gate를 먼저 적용하면 성능 승격 근거가 더 약해진다.

## 5. Workload-level interpretation

### 5.1 Safe-probe가 손해인 영역

`balanced`, `decode-heavy`, `long-prefill`, `bimodal`은 generated token/s가 각각 `-4.40%`, `-2.52%`,
`-6.88%`, `-3.50%`다.

공통 원인은 다음과 같다.

```text
aggressive P+D probe
       |
       +-- more decision boundaries
       +-- more executions
       +-- disturbed packed-P / D-cohort formation
       `-- overlap H1 gain보다 successor loss가 커짐
```

특히 long-prefill은 probe의 현재 boundary 이득보다 packed prefill의 후속 formation 가치가 크다. 이 workload는
formation-aware successor objective가 필요한 negative control로 유지한다.

### 5.2 Safe-probe가 좋아진 영역

`mixed`, `vision-heavy`, `poisson`, `late-vision`은 throughput과 TTFT/TPOT/E2E가 함께 좋아졌다. `wave/drain`은
throughput 개선은 작지만 TTFT mean `16.42%`, TTFT p95 `28.92%`, E2E p95 `17.82%`가 개선됐다.

이 결과는 overlap 기회가 실제로 존재한다는 증거다. 그러나 `text-heavy`, `mixed`, `vision-heavy`, `wave/drain`,
`multi-image`는 cross-policy token hash가 달라 production correctness gate를 통과하지 못했다. 결과 차이가 곧 KV
ownership 오류라는 뜻은 아니지만, canonical row/order와 FP16 greedy boundary를 분리 검증하기 전에는 성능 이득으로
승격하지 않는다.

### 5.3 Multi-image의 상충

Multi-image는 TTFT mean이 `21.00%` 좋아졌지만 TPOT mean은 `23.93%`, TPOT p95는 `32.71%` 나빠졌고 E2E
p95도 `3.67%` 나빠졌다. encoder critical-path 진전만 보고 overlap을 고르면 decode continuity를 손상할 수 있다는
직접 사례다.

## 6. Exact snapshot coverage

Combined myopic + safe-probe coverage:

| Item | Result |
|---|---:|
| Decisions joined | 18,911 |
| Snapshot signatures | 18,614 |
| Exact repeated snapshots | 185 |
| Exact multi-action snapshots | 108 |
| Same-phase H1-comparable snapshots | 106 |
| Fully repeated alternatives | 1 / 106 |
| Different-alternative H1 phase snapshots | 1 |
| Unstable-action H1 phase snapshots | 1 |
| Action-fidelity failures | 0 |

워크로드별 exact multi-action / comparable snapshot:

| Workload | Multi-action | H1 comparable | Fully repeated |
|---|---:|---:|---:|
| short | 3 | 3 | 0 |
| balanced | 1 | 1 | 0 |
| decode-heavy | 3 | 3 | 0 |
| long-prefill | 17 | 16 | 0 |
| bimodal | 54 | 54 | 0 |
| text-heavy | 5 | 4 | 0 |
| mixed | 3 | 3 | 0 |
| vision-heavy | 9 | 9 | 1 |
| poisson | 6 | 6 | 0 |
| wave/drain | 6 | 6 | 0 |
| multi-image | 1 | 1 | 0 |
| late-vision | 0 | 0 | 0 |

Coverage breadth는 충분히 커졌지만 거의 모든 alternative가 한 번만 측정됐다. 따라서 empirical median은 pilot
diagnostic이며 promotion-quality oracle이 아니다.

## 7. H1 instability handling

같은 action ID가 반복 표본에서 서로 다른 phase를 첫 번째로 완료하는 action 하나가 발견됐다.

```text
sample A: E completes first
sample B: D completes first
```

이는 action fidelity 실패가 아니다. 두 phase completion boundary가 가까울 때 runtime variation으로 H1 순서가 뒤집힌
것이다. 기존 분석기는 이를 hard error로 종료했다. 이제 분석기는 다음처럼 처리한다.

1. action별 `completed_phase_counts`를 보존한다.
2. H1 phase가 표본 사이에서 바뀌면 `stable_h1_phase=false`로 표시한다.
3. 해당 snapshot은 same-phase empirical oracle/regret에서 제외한다.
4. 제외 수를 `unstable_action_h1_snapshots`로 별도 보고한다.

불안정한 H1을 임의로 다수결 phase에 합치지 않으므로 oracle regret을 낙관적으로 만들지 않는다.

## 8. Empirical H1 pilot

안정적인 106개 snapshot에서:

| Policy | Exact snapshots | Selection observations | H1 agreement | Mean regret | Total regret |
|---|---:|---:|---:|---:|---:|
| myopic | 83 | 113 | 68 / 113 (60.2%) | 318.5 us | 35.992 ms |
| safe-probe | 98 | 137 | 46 / 137 (33.6%) | 787.2 us | 107.845 ms |

Safe-probe의 H1 objective는 myopic보다 명확히 나쁘다. 하지만 mixed/VLM full trace에서 request E2E가 개선된 경우가
있으므로 다음 두 값을 분리해야 한다.

```text
H1(a) = first protected progress boundary
H2(a) = action completion + best observable successor transition
```

M6 policy는 H1을 measured runtime evidence로 쓰되 H1 minimum 하나만 action value로 사용하지 않는다.

## 9. vLLM comparison policy

이번 단계는 기존 12개 materialized trace와 HTTP/output contract를 변경하지 않았고 production selector도 바꾸지 않았다.
따라서 vLLM을 중복 실행하지 않았다. Frozen vLLM absolute reference는
`notes/193-contextual-ep-ed-controller-and-gate-20260831.md`에 유지한다.

이번 계측 run은 unified event emission과 safe-probe data collection이 활성화돼 있으므로 frozen vLLM과 headline
성능을 직접 섞지 않는다. M6 후보가 output identity와 same-runtime 12-workload gate를 통과한 뒤 동일 HTTP 조건의
Current와 frozen vLLM을 비교하며, request/model/output contract가 바뀌면 fresh vLLM을 실행한다.

## 10. Artifacts

```text
.local/m5-oracle-h1-20260901/workload12-v1/
  myopic/
  safe-probe/
  myopic-coverage.json
  myopic-empirical-analysis.json
  myopic-safe-probe-coverage.json
  myopic-safe-probe-empirical-analysis.json
```

각 workload directory에는 다음이 있다.

```text
worker-4/aggregate.json
worker-4/run-001/gateway.log
worker-4/activity/run-001-intervals.csv
worker-4/activity/run-001-segments.csv
worker-4/activity/run-001-summary.csv
```

## 11. Promotion decision

```text
event/action fidelity                       PASS
natural exact multi-action coverage         PASS
same-phase H1 pilot evaluable               PASS
all alternatives repeated >= 2              FAIL
cross-policy exact greedy output 12/12       FAIL (7/12)
safe-probe throughput no-regression          FAIL
safe-probe latency no-regression             FAIL
safe-probe production promotion              REJECT
```

Production default는 myopic으로 유지한다. Safe-probe는 bounded data-collection mode로만 남긴다.

## 12. Next implementation order

1. **Selected repeat coverage**: balanced, long-prefill, mixed, vision-heavy, wave/drain, late-vision을 우선 3--5회
   반복해 alternative별 median/p95와 H1 phase stability를 확보한다.
2. **Numerical determinism isolation**: hash가 달라진 5 workload에서 canonical D row order, binding shape, first
   divergent request/token을 기록해 scheduling numerical drift와 ownership corruption을 분리한다.
3. **Bounded H2 label**: arbitrary future arrival은 예측하지 않고 현재 action으로 유도되는 successor state와 이미
   outstanding인 completion event만 사용해 equal-work H2 label을 만든다.
4. **M6 contextual estimator shadow mode**: exact key를 policy authority로 쓰지 않고 continuous feature에서 H1/H2
   advantage와 uncertainty를 예측한다. 외부 cost registry나 TTL은 사용하지 않는다.
5. **No-workload-label policy gate**: workload 이름을 feature로 넣지 않고 동일 selector configuration으로 12개를
   다시 실행한다.
6. **Promotion order**: exact output → action fidelity → per-workload 3% no-regression → SLO goodput → frozen/fresh
   vLLM comparison 순서를 유지한다.

M5의 결과는 “safe-probe가 더 빠르다”가 아니다. workload-specific rule 없이도 exact execution state와 online history에서
action 결과를 수집할 수 있게 되었고, H1-only와 always-probe가 모두 불충분하다는 것을 12개 실제 request workload에서
확인했다는 것이다.

## 13. Selected independent repeats

Section 12의 첫 번째 항목을 바로 실행했다. `balanced`, `long-prefill`, `mixed`, `vision-heavy`, `wave/drain`,
`late-vision`에 대해 정책별로 독립 process 두 개를 더 실행해 최초 survey와 합쳐 정책별 3개 표본을 만들었다.

Runner에는 `--matrix-repeats`를 추가했다. 이는 benchmark 내부의 `--repeats`만 늘리는 방식과 다르다.

```text
matrix repeat 1
  independent server process
  independent warmup / online cost history
  independent CUDA activity prefix
  independent phase run ID

matrix repeat 2
  independent server process
  independent warmup / online cost history
  independent CUDA activity prefix
  independent phase run ID
```

이 구조는 activity artifact 덮어쓰기와 process-local cost-history 공유를 피한다. 새 lineage 동작을 포함한 Python test는
`9/9` 통과했다.

### 13.1 Three-run median comparison

아래 값은 정책별 3-run median이다. TTFT/TPOT/E2E는 `mean / p95 ms`다.

| Workload | gen tok/s M→S | TTFT M→S | TPOT M→S | E2E M→S |
|---|---:|---:|---:|---:|
| balanced | 3192.16 → 2928.20 | 89.2 / 188.3 → 79.0 / 218.9 | 17.48 / 19.26 → 19.46 / 21.92 | 1580.4 / 2464.4 → 1734.5 / 2752.8 |
| long-prefill | 1018.89 → 951.54 | 2373.2 / 3272.6 → 2576.5 / 3102.6 | 31.38 / 36.75 → 33.55 / 38.20 | 5074.0 / 7364.2 → 5444.9 / 7560.3 |
| mixed | 962.16 → 991.56 | 962.0 / 2521.4 → 931.0 / 2457.8 | 35.39 / 43.81 → 34.85 / 42.32 | 2622.3 / 2974.6 → 2556.9 / 2884.3 |
| vision-heavy | 595.63 → 624.33 | 1625.0 / 3604.4 → 1544.6 / 3468.0 | 33.03 / 40.52 → 31.72 / 39.84 | 2904.2 / 4062.4 → 2801.8 / 3871.1 |
| wave/drain | 94.94 → 96.34 | 353.4 / 516.2 → 298.2 / 363.6 | 10.87 / 14.54 → 10.12 / 12.90 | 689.5 / 767.7 → 613.8 / 630.9 |
| late-vision | 2011.06 → 2069.15 | 157.5 / 565.0 → 145.2 / 509.9 | 11.76 / 11.83 → 11.42 / 11.48 | 1842.1 / 2293.9 → 1780.7 / 2229.1 |

Safe-probe improvement:

| Workload | gen tok/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 |
|---|---:|---:|---:|---:|
| balanced | **-8.27%** | +11.42% / -16.25% | -11.29% / -13.81% | -9.75% / -11.71% |
| long-prefill | **-6.61%** | -8.57% / +5.19% | -6.93% / -3.96% | -7.31% / -2.66% |
| mixed | **+3.06%** | +3.22% / +2.52% | +1.52% / +3.40% | +2.50% / +3.04% |
| vision-heavy | **+4.82%** | +4.95% / +3.78% | +3.97% / +1.67% | +3.52% / +4.71% |
| wave/drain | **+1.48%** | +15.62% / +29.56% | +6.87% / +11.29% | +10.98% / +17.82% |
| late-vision | **+2.89%** | +7.83% / +9.74% | +2.91% / +2.89% | +3.33% / +2.82% |

부호 반전은 반복 후에도 유지된다. 따라서 다음 구현을 `if balanced`, `if vision-heavy` 같은 workload rule로 만들면 안
된다. 같은 selector가 ready mass, isolated phase cost ratio, slack, outstanding residual, successor fill delta를 보고 이
두 영역을 구분해야 한다.

### 13.2 Repeat correctness and numerical stability

| Workload | Unique myopic hashes / 3 | Unique safe-probe hashes / 3 | Shared hash exists |
|---|---:|---:|:---:|
| balanced | 1 | 1 | yes |
| long-prefill | 1 | 1 | yes |
| mixed | 3 | 3 | no |
| vision-heavy | 3 | 3 | no |
| wave/drain | 2 | 2 | yes |
| late-vision | 1 | 1 | yes |

Mixed와 vision-heavy는 같은 policy 안에서도 세 번 모두 token hash가 다르다. Wave/drain도 두 개 hash를 가진다.
그러므로 이 세 workload의 cross-policy hash 불일치를 safe-probe의 correctness failure로만 돌릴 수 없다. 먼저 VLM
row/binding/reduction order의 runtime repeat determinism을 해결해야 한다. 반대로 balanced, long-prefill, late-vision은
정책 내부와 정책 사이에서 exact output이 안정적이므로 performance causal anchor로 사용할 수 있다.

### 13.3 Expanded exact-state coverage

최초 12-workload survey와 24개 추가 독립 실행을 모두 합친 결과:

| Item | Before repeats | After repeats |
|---|---:|---:|
| Decisions | 18,911 | 36,207 |
| Snapshot signatures | 18,614 | 34,994 |
| Exact repeated snapshots | 185 | 679 |
| Exact multi-action snapshots | 108 | 436 |
| Same-phase H1-comparable snapshots | 106 | 424 |
| Fully repeated alternatives | 1 | 3 |
| Skipped H1 snapshots | 2 | 12 |
| Action-fidelity failures | 0 | 0 |

추가 반복은 coverage breadth를 크게 늘렸지만 promotion-quality repeated alternatives는 `1 → 3`만 늘렸다. 비동기
arrival/completion 때문에 전체 request trace를 다시 실행해도 request ID, progress, in-flight set이 모두 같은 exact
snapshot을 재현하기 어렵기 때문이다.

이 결과는 exact-key table을 production policy representation으로 쓰지 말아야 한다는 직접 근거다.

```text
exact snapshot
  good for: fidelity, counterfactual audit, corruption detection
  poor for: dense policy generalization

continuous contextual state
  good for: interpolation, uncertainty, online action value
```

Expanded empirical H1:

| Policy | Exact snapshots | Observations | Agreement | Mean H1 regret |
|---|---:|---:|---:|---:|
| myopic | 145 | 247 | 53.4% | 165.3 us |
| safe-probe | 394 | 793 | 40.4% | 870.6 us |

Safe-probe는 대안 표본을 많이 만들지만 H1 regret이 크다. M6는 probe를 매 opportunity에 실행하지 않고 uncertainty와
protected slack이 허용하는 bounded exploration에서만 사용해야 한다.

## 14. Revised next step after repeats

1. exact snapshot join은 correctness/audit plane으로 유지한다.
2. mixed/vision-heavy/wave의 same-policy numerical nondeterminism을 canonical row and binding trace로 분해한다.
3. M6는 workload label 없이 `P+D`, `E+D`, `E+P` action family별 low-dimensional contextual estimator를 shadow
   mode로 연결한다.
4. estimator label은 H1 alone이 아니라 bounded H2 successor value와 분리 보존한다.
5. 외부 cost registry와 TTL은 사용하지 않는다. process-local observation과 uncertainty만 사용한다.
6. action authority를 주기 전 decision latency, prediction calibration, held-out action regret를 측정한다.
7. authority 후에는 balanced/long-prefill 회귀 방지와 mixed/vision/wave/late 이득 보존을 같은 12-workload
   configuration으로 검증한다.
