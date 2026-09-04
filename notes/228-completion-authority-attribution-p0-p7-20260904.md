# Completion-Vector Authority Attribution: P0--P7

날짜: 2026-09-04

브랜치: `codex/v010-phase-forward-port`

선행 문서:

- `notes/225-minimal-physical-outcome-full12-gate-20260903.md`
- `notes/226-physical-outcome-epoch-full12-results-20260903.md`
- `notes/227-bounded-completion-calibration-full12-20260903.md`

## 1. 최종 결론

이번 단계는 completion-vector predictor가 물리적 E/P/D 완료 시각을 학습한다는 사실과, 그 예측을 실제 policy authority로
사용했을 때 12개 real-request HTTP workload가 좋아진다는 주장을 분리해 검증했다.

결론은 명확하다.

1. bounded generic calibration, direction별 authority attribution, common-CUDA-epoch 측정은 정상 동작한다.
2. Completion-Vector Active는 12-workload promotion gate를 통과하지 못했다. `mixed -6.06%`, `multi-image -24.27%`,
   `vision-heavy -4.02%` throughput 회귀가 있다.
3. 같은 completion 모델을 shadow로만 실행하면 macro throughput은 Scalar 대비 `+0.38%`지만 `vision-heavy -3.93%`라
   보수적인 전체 gate를 통과하지 못한다.
4. exact-snapshot H1 분석에서 Active의 immediate first-completion regret은 평균 `29.93 us`다. 이것만으로 4--24%의
   end-to-end 회귀를 설명할 수 없다.
5. 실제 큰 손실은 작은 수의 action 변경이 E/P 완료 순서를 바꾸고, 일부 request를 D로 먼저 진입시켜 미래 D cohort를
   파편화하는 execution--formation coupling에서 나온다.
6. 기존 bounded H2는 `mixed`를 회복하고 `vision-heavy` raw throughput을 높였지만, `multi-image`를 회복하지 못하고
   vision TPOT/E2E trade-off를 만들었다. 지금 형태로 승격하지 않는다.
7. 한 process에서 posterior를 유지하면 vision/poisson/mixed throughput이 shadow보다 각각 `+4.6%/+2.0%/+1.1%`
   좋아질 수 있으나, fresh-process Scalar보다 모든 workload가 안정적으로 좋아지는 것은 아니다.
8. 학습 효율은 방향별로 비대칭이다. P->D는 424 generic request에서 authority-ready지만 E->P는 약 848,
   E->D는 1,272--1,696 request가 필요하며 최종 blend도 약하다.
9. fresh vLLM 12x1 대비 Scalar Current는 token throughput과 joint-SLO goodput 12/12, E2E mean/p95 12/12에서 앞섰다.
   token throughput의 workload 기하평균 우위는 `+16.40%`다. 다만 Current/vLLM 반복 수가 3/1이므로 최종 논문용
   confidence interval은 아니다.

따라서 현재 production 기본값은 **Scalar 유지**다. Completion-vector는 계속 측정하고 검증하는 shadow physical-outcome
predictor로 남긴다. 이는 completion architecture를 폐기한다는 뜻이 아니라, learned physical outcome과 scheduling policy의
승격 lifecycle을 분리한다는 뜻이다.

```text
현재 production path
-----------------------
Hard feasibility
  -> Scalar/exact completion
  -> SLO-safe transition evaluator
  -> Dispatch

동시에 유지하는 research path
------------------------------
CUDA completion observation
  -> hierarchical completion RLS
  -> held-out authority validation
  -> shadow completion vector
  -> causal attribution artifact
```

## 2. P0--P7 질문과 판정

| 단계 | 질문 | 결과 | 판정 |
|---|---|---|---|
| P0 | generic calibration이 bounded하고 동일 binary에서 재현되는가 | direction state가 complete 또는 bounded reject로 종료 | 통과 |
| P1 | Scalar/Shadow/Active 중 full-12에서 누가 안정적인가 | Active 3개 workload 큰 회귀, Shadow vision -3.93% | Scalar 유지 |
| P2 | completion authority가 어디서 실제 action을 바꾸는가 | 43,474 decisions 중 H1 변경 989회, fidelity/false-safe 0 | attribution 통과 |
| P3 | 동일 snapshot의 실제 H1 outcome으로 causal 비교 가능한가 | 21 comparable, fully repeated 0; Active regret가 더 큼 | pilot만 가능 |
| P4 | vision 회귀가 immediate completion 오류인가 | D dispatch가 103->122, 33->61, 138->147로 증가 | formation이 주원인 |
| P5 | posterior carryover가 서로 다른 trace에 전이되는가 | 일부 개선, 일부 회귀; workload label 없이 상태 적응 확인 | 부분 통과 |
| P6 | E/P/D 방향이 얼마나 빨리 안정화되는가 | P->D 빠름, E->P/E->D 느리고 blend 약함 | E sample efficiency 미달 |
| P7 | 무엇을 default로 승격하고 vLLM 대비 어디에 있는가 | Scalar가 fresh vLLM token/SLO/E2E 12/12 우세 | Scalar default |

## 3. 비교한 정책의 정확한 의미

### 3.1 Scalar

기존 contextual scalar advantage와 exact CUDA timing을 사용한다. completion-vector RLS가 있어도 action choice에 completion
authority를 적용하지 않는다. 현재 가장 안전한 production reference다.

### 3.2 Completion Shadow

Scalar와 같은 action을 실행한다. completion-vector prediction, uncertainty, held-out validation, attribution telemetry만
수집한다. 따라서 모델 계산 비용과 measurement lifecycle의 overhead는 포함하지만 completion prediction이 정책을 바꾸지는
않는다.

### 3.3 Completion Active

held-out authority gate를 통과한 direction에서 scalar completion을 component별 robust completion-vector와 blend한다.
Feasibility와 SLO hard guard는 그대로 유지한다. workload 이름, 외부 cost registry, TTL, workload별 tuning table은 사용하지
않는다.

### 3.4 P10

Completion Active 이전의 small-D/formation 보정 최고 구현이다. 같은 binary family의 historical best reference이지만 이번
P1의 Scalar/Shadow/Active와 같은 policy switch만 바꾼 순수 3-way A/B는 아니다.

## 4. P0: bounded calibration lifecycle

P0 구현은 두 commit으로 고정했다.

- `2daf4e1 feat: bound phase completion calibration`
- `9ab24dc fix: isolate phase calibration epochs`

각 ordered direction은 다음 상태를 가진다.

```text
posterior fit
    -> uncertainty calibration
    -> held-out authority validation
    -> complete(validated or rejected)
```

`complete`와 `ready`를 분리한다. 표본이 충분하지만 validation을 실패한 direction은 `complete=true, ready=false`가 되어
scalar fallback으로 끝난다. 같은 방향을 무한 probe하지 않는다. warm-up observation은 별도 measurement epoch에 기록해
serving latency와 online evidence를 오염시키지 않는다.

실행 전제는 모든 workload에서 동일하다.

- generic calibration trace 사용
- workload name/profile 입력 없음
- trace별 static cost registry 로드 없음
- P chunk 128 고정
- P max batch 8, D max batch 64, stable slots 80
- E batch 4, max encoded vision 16
- independent E/P/D TensorRT contexts 및 같은 CUDA context

## 5. P1: 12-workload same-binary policy matrix

Scalar와 Shadow는 3회, Active는 3회이고 `vision-heavy`는 오염 가능성이 없는 clean 5회 결과를 사용했다. 아래 throughput은
generated token/s의 run median이다.

| workload | Scalar | Shadow | Active | P10 | Active vs Scalar | Shadow vs Scalar | P10 vs Scalar |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 4497.03 | 4513.92 | 4488.91 | 4564.31 | -0.18% | +0.38% | +1.50% |
| bimodal | 1979.19 | 1956.45 | 1984.87 | 1951.98 | +0.29% | -1.15% | -1.37% |
| decode-heavy | 5318.58 | 5297.87 | 5326.33 | 5315.67 | +0.15% | -0.39% | -0.05% |
| late-vision | 2534.86 | 2555.10 | 2544.19 | 2554.93 | +0.37% | +0.80% | +0.79% |
| long-prefill | 1237.64 | 1243.76 | 1249.08 | 1198.68 | +0.92% | +0.49% | -3.15% |
| mixed | 1139.57 | 1175.24 | 1070.49 | 1119.29 | -6.06% | +3.13% | -1.78% |
| multi-image | 323.35 | 319.20 | 244.86 | 323.01 | -24.27% | -1.28% | -0.10% |
| poisson | 1936.13 | 1996.60 | 1929.51 | 1962.66 | -0.34% | +3.12% | +1.37% |
| short | 2487.10 | 2510.56 | 2507.66 | 2500.93 | +0.83% | +0.94% | +0.56% |
| text-heavy | 2029.10 | 2080.48 | 2071.06 | 2110.89 | +2.07% | +2.53% | +4.03% |
| vision-heavy | 698.95 | 671.50 | 670.87 | 710.59 | -4.02% | -3.93% | +1.67% |
| wave-drain | 98.13 | 98.08 | 98.06 | 98.08 | -0.08% | -0.05% | -0.06% |

### 5.1 전체 latency와 memory

모든 latency 열은 `mean/p95 ms`다. peak memory는 run별 peak의 median이다.

| workload | policy | tok/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | peak MiB |
|---|---|---:|---:|---:|---:|---:|
| short | Scalar | 2487.10 | 92.6/172.9 | 13.19/27.22 | 333.3/412.2 | 9237 |
| short | Shadow | 2510.56 | 84.5/168.4 | 13.62/26.79 | 329.4/407.8 | 9237 |
| short | Active | 2507.66 | 88.0/173.5 | 13.45/26.46 | 330.1/408.5 | 9237 |
| balanced | Scalar | 4497.03 | 61.6/167.4 | 12.30/13.88 | 1110.6/1723.3 | 9237 |
| balanced | Shadow | 4513.92 | 63.9/166.5 | 12.23/13.87 | 1100.2/1716.5 | 9237 |
| balanced | Active | 4488.91 | 63.5/166.6 | 12.32/13.89 | 1111.7/1721.3 | 9237 |
| decode-heavy | Scalar | 5318.58 | 63.6/180.8 | 10.46/11.10 | 2763.6/4217.2 | 9237 |
| decode-heavy | Shadow | 5297.87 | 62.0/180.2 | 10.54/11.18 | 2777.7/4254.2 | 9237 |
| decode-heavy | Active | 5326.33 | 63.6/175.2 | 10.45/11.01 | 2761.1/4206.6 | 9237 |
| long-prefill | Scalar | 1237.64 | 2037.6/2566.8 | 25.19/29.72 | 4182.2/5815.1 | 9237 |
| long-prefill | Shadow | 1243.76 | 2026.6/2659.8 | 25.21/29.52 | 4170.8/5849.0 | 9237 |
| long-prefill | Active | 1249.08 | 2014.8/2860.3 | 25.07/30.61 | 4147.4/5893.0 | 9237 |
| bimodal | Scalar | 1979.19 | 1824.0/4076.8 | 17.56/28.92 | 4216.7/8825.0 | 9237 |
| bimodal | Shadow | 1956.45 | 1854.0/3931.4 | 17.83/28.31 | 4281.8/8915.3 | 9237 |
| bimodal | Active | 1984.87 | 1802.2/4195.8 | 17.51/28.71 | 4184.1/8914.1 | 9237 |
| text-heavy | Scalar | 2029.10 | 312.1/1086.2 | 24.31/39.10 | 1565.6/1652.3 | 9447 |
| text-heavy | Shadow | 2080.48 | 357.7/959.5 | 22.35/34.38 | 1509.2/1614.7 | 9467 |
| text-heavy | Active | 2071.06 | 312.4/972.3 | 23.09/35.65 | 1514.7/1615.1 | 9471 |
| mixed | Scalar | 1139.57 | 718.9/2157.1 | 33.14/51.19 | 2293.4/2501.5 | 9473 |
| mixed | Shadow | 1175.24 | 753.5/2139.9 | 30.78/38.72 | 2124.8/2425.5 | 9471 |
| mixed | Active | 1070.49 | 725.1/2292.7 | 35.37/43.76 | 2406.8/2663.7 | 9455 |
| vision-heavy | Scalar | 698.95 | 1331.0/3200.4 | 28.04/43.43 | 2438.8/3459.5 | 9447 |
| vision-heavy | Shadow | 671.50 | 1348.3/3235.6 | 33.53/46.70 | 2691.9/3591.5 | 9461 |
| vision-heavy | Active | 670.87 | 1346.1/3217.4 | 40.70/55.66 | 2954.4/3603.5 | 9449 |
| poisson | Scalar | 1936.13 | 244.8/1005.9 | 20.96/36.85 | 1602.0/2012.0 | 9455 |
| poisson | Shadow | 1996.60 | 202.4/730.7 | 21.09/38.85 | 1547.6/1979.9 | 9463 |
| poisson | Active | 1929.51 | 211.9/787.5 | 21.67/40.49 | 1600.7/2018.9 | 9461 |
| wave-drain | Scalar | 98.13 | 295.9/419.8 | 8.14/11.63 | 548.2/624.5 | 9447 |
| wave-drain | Shadow | 98.08 | 242.2/293.9 | 8.03/11.61 | 491.3/500.7 | 9467 |
| wave-drain | Active | 98.06 | 247.5/291.8 | 7.85/11.09 | 490.9/500.0 | 9453 |
| multi-image | Scalar | 323.35 | 250.0/286.6 | 7.73/9.56 | 489.8/494.5 | 9463 |
| multi-image | Shadow | 319.20 | 243.1/286.5 | 8.14/10.85 | 489.7/500.5 | 9455 |
| multi-image | Active | 244.86 | 354.3/447.3 | 8.59/11.53 | 620.5/653.2 | 9427 |
| late-vision | Scalar | 2534.86 | 120.7/426.0 | 9.28/9.33 | 1448.8/1819.3 | 9465 |
| late-vision | Shadow | 2555.10 | 124.7/436.8 | 9.22/9.28 | 1444.0/1805.1 | 9449 |
| late-vision | Active | 2544.19 | 112.0/431.1 | 9.25/9.30 | 1451.5/1812.4 | 9465 |

Aggregate 판정:

| 정책 | macro token throughput vs Scalar | throughput wins | macro joint-SLO goodput | exact token identity workloads |
|---|---:|---:|---:|---:|
| Scalar | reference | 0 | 18.998 req/s | 12/12 |
| Shadow | +0.38% | 7/12 | 19.260 req/s | 9/12 |
| Active | -2.53% | 6/12 | 17.875 req/s | 9/12 |
| P10 | +0.28% | 6/12 | 18.194 req/s | 7/12 |

토큰 identity는 같은 policy의 반복이 아닌 Scalar token hash 집합과의 exact equality다. FP16 row/tactic 순서에 따른 작은
numerical branch도 포함하므로 semantic correctness와 별도 gate지만, production default 변경에는 보수적으로 사용했다.

## 6. P2: authority-use attribution

`57ec8aa feat: attribute completion policy authority`에서 decision/candidate telemetry에 다음을 추가했다.

```text
scalar component completion
active blended component completion
robust uncertainty
direction/pair observation count
authority-ready / authority-applied
scalar -> active -> final action change
snapshot signature
actual dispatch/completion action fidelity
```

Active full-12 상세 실행 결과:

- scheduler decisions: 43,474
- completion predictions ready/calibrated: 12,587/12,334
- authority-applied candidates: 11,302
- completion authority 때문에 H1 action이 달라진 decision: 989/43,474 = 2.275%
- action-fidelity failures: 0
- conformal/authority false-safe: 0

즉 Active의 큰 회귀는 모든 decision이 바뀌어서가 아니다. 약 2.3%의 boundary decision이 request transition 순서를 바꾸고,
그 결과가 이후 여러 decode iteration에 전파된다.

## 7. P3: measured H1 causal replay와 measurement contract

### 7.1 첫 분석이 실패한 이유

초기 detailed log에는 `gpu_duration_us`만 있었고 `gpu_start_us/gpu_end_us`가 없었다. 서로 다른 action의 종료 시각을 같은
epoch에서 비교할 수 없으므로 90,043개의 `selected action has no common-epoch completion`을 만들었다. 이를 isolated
duration 추정값으로 채우지 않았다. causal artifact는 반드시 공통 CUDA-event epoch를 요구한다.

### 7.2 candidate frontier signature 수정

coordinator의 ready snapshot은 mechanism queue 전체를 소유하지 않는다. 특히 prepared vision batch는 pending queue에서 이미
빠졌지만 encoder candidate에는 남을 수 있다. 기존 signature는 이 상태에서 Scalar E1과 Active E4를 같은 snapshot으로
잘못 join했다.

수정 후 snapshot key는 다음 두 부분을 모두 포함한다.

```text
coordinator ready/ownership/outstanding signature
    +
sorted(candidate action kind, legal, candidate request IDs)
```

이 검증은 `build_oracle_h1_snapshot_coverage.py`와
`test_oracle_h1_snapshot_coverage.py::test_does_not_join_hidden_encoder_frontiers_with_same_ready_state`에 고정했다.

### 7.3 corrected exact-snapshot 결과

`mixed`, `multi-image`, `vision-heavy`를 activity-enabled Scalar/Active로 다시 실행했다.

- decisions: 25,184
- exact repeated snapshots: 31
- exact multi-action snapshots: 22
- comparable H1 snapshots: 21
- fully repeated alternative snapshots: 0
- action fidelity failures: 0
- promotion-quality coverage: false

| policy | H1 oracle agreement | mean empirical H1 regret |
|---|---:|---:|
| Scalar | 70.59% | 16.89 us |
| Completion Active | 22.50% | 29.93 us |

fully repeated alternative가 0이므로 이 결과는 unbiased oracle promotion 증거가 아니다. 하지만 기존 false E1/E4 join에서
나온 64 ms 가짜 regret는 제거됐고, 실제 immediate H1 regret가 수십 us임을 보여준다. 따라서 end-to-end 4--24% 회귀의
주원인을 immediate completion 오차 하나로 돌릴 수 없다.

## 8. P4: vision-heavy gap과 execution--formation coupling

동일 activity-enabled run에서 측정한 dispatch 수와 D GPU duration 합은 다음과 같다.

| workload | Scalar D dispatch | Active D dispatch | Scalar D GPU sum | Active D GPU sum | P dispatch Scalar/Active |
|---|---:|---:|---:|---:|---:|
| mixed | 103 | 122 | 899.3 ms | 1149.3 ms | 37/37 |
| multi-image | 33 | 61 | 220.9 ms | 399.3 ms | 4/4 |
| vision-heavy | 138 | 147 | 1232.1 ms | 1395.4 ms | 48/48 |

`multi-image`가 가장 명확하다.

```text
Scalar
E/P transition을 먼저 모음
  -> D cohort가 함께 시작
  -> D dispatch 33

Completion Active
일부 request P가 먼저 완료
  -> 부분 D cohort를 29 step drain
  -> 남은 P가 P+D로 뒤늦게 완료
  -> 다시 D cohort를 31 step drain
  -> D dispatch 61
```

현재 action의 physical completion을 조금 더 정확히 맞춰도 다음 request-ready boundary를 바꾸면 전체 future service cost가
커질 수 있다. 이 현상을 **Execution--Formation Coupling**, 그 부정적 사례를 **Action-Induced Batch Fragmentation**으로
정의한다.

### 8.1 bounded H2 결과

Completion Active에 기존 deterministic formation-aware H2를 결합해 회귀 3개를 한 번씩 재실행했다.

| workload | Scalar tok/s | Active tok/s | Active+H2 tok/s | H2 TTFT mean/p95 | H2 TPOT mean/p95 | H2 E2E mean/p95 |
|---|---:|---:|---:|---:|---:|---:|
| mixed | 1092.31 | 1042.09 | 1095.51 | 725.0/2231.4 | 34.45/50.39 | 2333.0/2592.9 |
| vision-heavy | 673.21 | 656.98 | 735.68 | 1226.4/2954.6 | 46.38/71.74 | 3004.4/3296.7 |
| multi-image | 321.67 | 231.88 | 232.88 | 374.6/482.8 | 8.69/11.37 | 644.1/686.6 |

H2는 mixed를 회복하고 vision raw throughput을 올렸지만 multi-image는 회복하지 못했다. vision-heavy도 TPOT mean/p95가
Scalar의 39.70/54.54 ms보다 나빠졌다. 현재 H2는 E->P->D 두 transition 뒤의 D cohort split을 충분히 표현하지 못한다.
새 threshold를 workload별로 넣지 않고 승격을 보류한다.

## 9. P5: cross-trace posterior transfer

`run_cross_trace_http_epochs.py`가 frozen HTTP command manifest의 backend suffix를 직접 읽도록 확장했다. 이로써 실험마다 긴
Docker/engine command를 재작성하지 않고도 정확히 같은 binary/config를 한 process에서 유지할 수 있다.

실행 순서는 다음과 같다.

```text
generic warm-up once
  -> text-heavy
  -> vision-heavy
  -> poisson
  -> mixed

posterior, EMA residual, authority state는 process 안에서 유지
workload name은 policy 입력으로 사용하지 않음
```

3-process 반복의 median 및 range:

| epoch | Active tok/s median (range) | Shadow tok/s median (range) | Active vs Shadow | Active TTFT p95 | Active TPOT p95 | Active E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| text-heavy | 2042.54 (1974.37--2072.88) | 2051.18 (2027.83--2086.71) | -0.42% | 1020.97 | 37.17 | 1641.95 |
| vision-heavy | 655.34 (640.30--696.05) | 626.46 (619.32--647.80) | +4.61% | 3241.36 | 49.53 | 3673.82 |
| poisson | 1915.36 (1860.25--1985.30) | 1878.21 (1874.19--1983.13) | +1.98% | 801.16 | 39.36 | 2016.84 |
| mixed | 1111.19 (1088.40--1128.69) | 1098.90 (1064.66--1109.90) | +1.12% | 2171.86 | 52.55 | 2568.41 |

Active carryover를 fresh-process Active P1과 비교하면 text/vision/poisson은 각각 `-1.38%/-2.32%/-0.73%`, mixed는
`+3.80%`다. posterior transfer는 실제로 action을 바꾸고 후반 trace에서 이득을 만들 수 있지만, 아직 order-independent한
안정성이나 모든 trace의 oracle 근접성을 보장하지 않는다.

## 10. P6: direction별 sample efficiency

424/848/1,272/1,696 generic calibration request에서 여러 run의 status snapshot을 합쳐 authority acquisition을 측정했다.

| direction | budget | authority-ready fraction | posterior obs median | model/reference abs-error ratio | incumbent/newcomer blend median |
|---|---:|---:|---:|---:|---:|
| P->D | 424 | 100% | 196.5 | 0.359 | 0.684/0.433 |
| P->D | 848 | 100% | 378.0 | 0.286 | 0.705/0.618 |
| P->D | 1,272 | 100% | 546.5 | 0.289 | 0.675/0.644 |
| P->D | 1,696 | 100% | 723.0 | 0.309 | 0.625/0.655 |
| E->P | 424 | 0% | 22.0 | 0.470 | 0.000/0.000 |
| E->P | 848 | 90.9% | 33.0 | 0.513 | 0.000/0.209 |
| E->P | 1,272 | 90.0% | 40.0 | 0.520 | 0.000/0.357 |
| E->P | 1,696 | 75.0% | 41.5 | 0.420 | 0.000/0.420 |
| E->D | 424 | 0% | 10.5 | n/a | 0.000/0.000 |
| E->D | 848 | 0% | 19.0 | 0.708 | 0.000/0.000 |
| E->D | 1,272 | 50.0% | 27.0 | 0.254 | 0.000/0.000 |
| E->D | 1,696 | 100% | 29.0 | 0.216 | 0.000/0.099 |

P->D는 빠르고 안정적으로 유효 authority를 얻는다. E->P는 validation 자체는 비교적 일찍 가능하지만 incumbent blend가
0으로 남는다. E->D는 4배의 warm-up budget에서도 newcomer blend가 약 0.1뿐이다. 따라서 하나의 global `sample_count >= N`
조건으로 E/P/D를 활성화하면 안 된다. 반대로 workload-specific threshold도 추가하지 않는다. direction마다 posterior
uncertainty와 held-out physical error로 authority를 얻는 현재 구조를 유지하되, sparse E observation을 더 효율적으로 만드는
shared/pair/direction shrinkage가 다음 과제다.

## 11. P7: fresh vLLM HTTP gate

이 절은 fresh vLLM 12-workload 실행 완료 후 같은 trace, 같은 request concurrency, 같은 SLO 정의로 채운다.

P7 실행 계약:

- vLLM 0.27.1, FP16 Cosmos-Reason2-2B
- workload마다 새 container/engine process
- prefix caching off, MM processor cache 0
- chunked prefill on, max batched tokens 8192, max sequences 80
- KV cache 3.5 GiB 고정
- max model length 2048, ignore EOS
- warm-up 64 requests x 32 output tokens
- SLO: TTFT 500 ms, TPOT 50 ms, E2E 2,500 ms
- Current는 P1 Scalar 3회 median, vLLM은 fresh 1회 기준점

### 11.1 결과

Current는 Scalar 3회 median이고 vLLM은 이번 fresh 1회다. `tok/s delta`는 Current가 높을수록 양수다. latency는 절대값을
나란히 표시했다.

| workload | Current tok/s | vLLM tok/s | delta | TTFT Current/vLLM mean | TTFT Current/vLLM p95 | TPOT Current/vLLM mean | TPOT Current/vLLM p95 | E2E Current/vLLM mean | E2E Current/vLLM p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | 4497.0 | 4312.7 | +4.3% | 61.6/112.4 | 167.4/292.8 | 12.30/12.21 | 13.88/13.63 | 1110.6/1155.2 | 1723.3/1785.4 |
| bimodal | 1979.2 | 1840.2 | +7.6% | 1824.0/1573.1 | 4076.8/2626.0 | 17.56/22.93 | 28.92/37.13 | 4216.7/4721.6 | 8825.0/9461.6 |
| decode-heavy | 5318.6 | 4960.4 | +7.2% | 63.6/120.9 | 180.8/323.3 | 10.46/10.96 | 11.10/11.56 | 2763.6/2959.6 | 4217.2/4477.5 |
| late-vision | 2534.9 | 2356.6 | +7.6% | 120.7/162.0 | 426.0/599.0 | 9.28/9.92 | 9.33/9.93 | 1448.8/1583.1 | 1819.3/1957.5 |
| long-prefill | 1237.6 | 1118.9 | +10.6% | 2037.6/1928.5 | 2566.8/2926.3 | 25.19/32.37 | 29.72/37.41 | 4182.2/4680.9 | 5815.1/6667.0 |
| mixed | 1139.6 | 747.7 | +52.4% | 718.9/1598.9 | 2157.1/3325.6 | 33.14/30.72 | 51.19/72.37 | 2293.4/2800.7 | 2501.5/3874.7 |
| multi-image | 323.3 | 237.6 | +36.1% | 250.0/268.4 | 286.6/411.6 | 7.73/12.67 | 9.56/16.41 | 489.8/661.1 | 494.5/671.7 |
| poisson | 1936.1 | 1792.9 | +8.0% | 244.8/453.7 | 1005.9/914.7 | 20.96/22.01 | 36.85/46.04 | 1602.0/1810.4 | 2012.0/2276.5 |
| short | 2487.1 | 1987.4 | +25.1% | 92.6/172.8 | 172.9/264.3 | 13.19/13.55 | 27.22/25.45 | 333.3/425.3 | 412.2/501.8 |
| text-heavy | 2029.1 | 1636.5 | +24.0% | 312.1/472.6 | 1086.2/1462.1 | 24.31/28.12 | 39.10/40.22 | 1565.6/1941.7 | 1652.3/2032.9 |
| vision-heavy | 699.0 | 575.7 | +21.4% | 1331.0/1764.8 | 3200.4/3782.3 | 28.04/63.60 | 43.43/115.16 | 2438.8/4172.4 | 3459.5/4258.2 |
| wave-drain | 98.1 | 95.7 | +2.5% | 295.9/270.0 | 419.8/454.9 | 8.14/12.25 | 11.63/17.31 | 548.2/649.8 | 624.5/678.8 |

### 11.2 SLO goodput와 memory

| workload | joint-SLO goodput Current/vLLM req/s | peak memory Current/vLLM MiB |
|---|---:|---:|
| balanced | 12.32/11.06 | 9237/9039 |
| bimodal | 0.90/0.04 | 9237/9383 |
| decode-heavy | 1.99/1.85 | 9237/9039 |
| late-vision | 17.57/14.29 | 9465/9225 |
| long-prefill | 0.35/0.00 | 9237/9323 |
| mixed | 11.81/1.53 | 9473/9871 |
| multi-image | 10.10/7.42 | 9463/9801 |
| poisson | 20.72/14.97 | 9455/9711 |
| short | 114.79/91.73 | 9237/9035 |
| text-heavy | 29.92/14.96 | 9447/9741 |
| vision-heavy | 4.50/0.00 | 9447/9739 |
| wave-drain | 3.01/2.99 | 9447/9827 |

요약:

- token throughput: Current 12/12 승리, workload 기하평균 `+16.40%`
- request throughput: Current 12/12 승리
- joint-SLO goodput: Current 12/12 승리
- E2E mean/p95: Current 12/12 모두 더 낮음
- TTFT mean: Current 9/12 승리; p95 10/12 승리
- TPOT mean/p95: Current 각각 10/12 승리
- peak memory: Current가 더 작거나 같은 workload 8/12

Current가 모든 개별 latency metric에서 이긴 것은 아니다. `bimodal` TTFT mean/p95, `long-prefill` TTFT mean,
`wave-drain` TTFT mean, `poisson` TTFT p95, `balanced/mixed` TPOT mean, `balanced/short` TPOT p95는 vLLM이 더 낮다.
그럼에도 E2E mean/p95 및 joint-SLO goodput은 12개 모두 Current가 높다.

해석할 때 세 제한을 유지한다.

1. Current는 3회 median, fresh vLLM은 1회이므로 이 표는 final statistical confidence interval이 아니다.
2. vLLM은 workload마다 새 process이고 Current P1도 run마다 gateway/backend가 새로 시작한다. serving state carryover는 없다.
3. `multi-image` 첫 vLLM attempt는 startup 직후 BrokenPipe로 실패했고 새 container의 두 번째 attempt가 성공했다. 실패 run은
   summary의 failure lineage에 남아 있고 성능 집계에서는 제외했다.

## 12. 구현 및 artifact 위치

| 파일 | 역할 |
|---|---|
| `benchmarks/phase_serving/run_cross_trace_http_epochs.py` | frozen backend manifest를 재사용하는 persistent cross-trace runner |
| `benchmarks/phase_serving/build_oracle_h1_snapshot_coverage.py` | common-epoch measured H1 join 및 candidate frontier identity 검증 |
| `benchmarks/phase_serving/analyze_completion_sample_efficiency.py` | budget별 posterior/authority/blend/error progression 분석 |
| `benchmarks/phase_serving/analyze_completion_vllm_gate.py` | Current policy matrix와 fresh vLLM HTTP suite 비교 |
| `tests/python-unittests/test_cross_trace_http_epochs.py` | manifest backend extraction 검증 |
| `tests/python-unittests/test_oracle_h1_snapshot_coverage.py` | hidden encoder frontier false join 회귀 테스트 |
| `tests/python-unittests/test_completion_sample_efficiency.py` | budget/direction aggregation 검증 |
| `tests/python-unittests/test_completion_vllm_gate.py` | throughput/latency/SLO/memory 비교 검증 |

실험 root:

```text
.local/completion-attribution-p0-p7-20260904/
├── p1/{scalar,shadow,active,active-clean}/
├── p1/policy-matrix.{json,csv}
├── p2-detailed/
├── p3-causal/
│   ├── activity-{scalar,active}/
│   └── regression-oracle-h1-{coverage,analysis}-v2.json
├── p4-active-h2/
├── p5-cross-trace/{active,shadow}-repeat-{1,2,3}/
├── p6-sample-efficiency/vision-heavy.json
└── p7-vllm-fresh-12x1/
```

## 13. Promotion decision

### 기본값

```text
Scalar action authority      ON
Completion measurement      ON when enabled for research/validation
Completion shadow           allowed
Completion active authority OFF by default
Formation H2                OFF by default
```

### Active 승격을 보류한 이유

1. 12-workload throughput gate에서 3개 workload가 -3% 밖이다.
2. immediate H1 accuracy가 좋아져도 future D formation 손실을 막지 못한다.
3. E direction sample acquisition과 blend가 P->D보다 훨씬 약하다.
4. cross-trace posterior carryover 효과가 순서/trace별로 일관적이지 않다.
5. H2가 모든 vision regression과 tail latency를 동시에 회복하지 못했다.

## 14. 다음 계획

### N0. Scalar default freeze

현재 Scalar를 release/reference configuration으로 고정하고 Completion Active를 실험 옵션으로 유지한다. 이후 변경은 항상
Scalar 12-workload 결과와 비교한다.

### N1. Transition rollout의 단위를 request-ready boundary로 확장

현재 H1/H2는 첫 completion 또는 다음 한 action을 본다. multi-image의 손실은 E->P->D 두 transition 뒤에 나타나므로,
arbitrary future arrival을 예측하지 않고 현재 outstanding event 및 현재 ready requests만으로 다음 두 request-ready boundary를
rollout한다.

```text
action a
  -> measured/predicted physical completions
  -> deterministic request DAG transitions
  -> newly-ready P/D rows
  -> successor cohort formation
  -> bounded terminal cost
```

### N2. Physical prediction과 transition value 분리 유지

RLS는 incumbent/newcomer completion stretch와 realization delay만 학습한다. formation, stable KV ownership, vision lease release는
deterministic state transition으로 계산한다. RLS가 action reward 전체를 직접 학습하게 되돌리지 않는다.

### N3. Candidate-conditioned counterfactual coverage 확보

동일 exact snapshot에서 대안 action별 최소 3회 common-epoch observation을 확보한다. production request를 위험하게 탐색하지
않고 controlled fixed-frontier trace 및 충분한 SLO slack 구간에서만 probe한다.

승격 조건:

```text
fully repeated alternatives >= 3 samples/action
action fidelity failures = 0
false-safe = 0
completion-vector H1 regret < Scalar H1 regret
```

### N4. E sample efficiency 개선

별도 workload rule 대신 shared structural prior -> pair-family residual -> direction residual 계층을 강화한다. E direction이
sample이 적을 때 shared/pair prior로 수축하고, 충분해지면 direction residual authority가 증가해야 한다.

### N5. Cross-trace order robustness

서로 다른 trace ordering과 반복 cycle로 posterior carryover를 검증한다.

```text
text -> vision -> poisson -> mixed
vision -> text -> mixed -> poisson
mixed -> poisson -> vision -> text
```

각 order에서 fresh Scalar 대비 throughput, TTFT/TPOT/E2E mean/p95, false-safe, action-change frequency를 기록한다.

### N6. SLO goodput load sweep

raw throughput이 아니라 TTFT/TPOT/E2E joint-SLO goodput의 low/knee/saturation/overload 곡선을 Scalar, transition rollout,
vLLM에 대해 비교한다. workload별 policy knob은 두지 않는다.

### N7. 최종 승격 기준

Completion/transition active 정책은 다음을 모두 만족할 때만 기본값 후보가 된다.

- 12/12 workload에서 Scalar throughput -3% 이내
- macro joint-SLO goodput Scalar 이상
- vision-heavy/multi-image/mixed의 TTFT/TPOT/E2E p95 비회귀
- fresh vLLM과 동일 HTTP contract에서 경쟁력 확인
- exact action fidelity failure 0, false-safe 0
- workload label, external cost registry, TTL, workload별 fine tuning 없음

## 15. 연구 메시지

이번 결과가 지지하는 주장은 "learned predictor가 항상 더 빠르다"가 아니다.

```text
A phase action changes
  (1) immediate physical completion,
  (2) protected-request completion,
  (3) persistent ownership lifetime, and
  (4) the formation of future executable cohorts.
```

따라서 정확한 physical completion predictor는 필요하지만 충분하지 않다. deterministic correctness가 legal action을 정하고,
validated physical model이 실행 결과를 예측하며, bounded transition evaluator가 SLO와 future formation을 함께 평가해야 한다.
이번 P0--P7은 그중 physical-model authority를 무조건 켜는 것이 왜 잘못인지, 그리고 다음 scheduler가 어떤 state transition을
명시적으로 표현해야 하는지를 실측으로 좁혔다.
