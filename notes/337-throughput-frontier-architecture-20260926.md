<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 337. 처리량 우선: 공통 실행 경로와 selector 정합성

## 목표와 기준

사용자는 336 결과 이후 모든 workload에서 vLLM 처리량을 이기는 것을 최우선으로 요청했다.
동일 요청/출력 길이/모델 정밀도/정확도 조건을 유지한다. 처리량을 늘리기 위해 요청을 누락하거나,
output을 줄이거나, workload 이름으로 policy를 고르지 않는다. TTFT/TPOT/E2E mean/p95와 메모리도
계속 보고하되 latency 손익과 throughput 승격 판단을 구분한다. 아래 검증된24개 조합의 우세를
임의의 workload/GPU에서의 무조건적 우세나 품질 gate 통과로 확장하지 않는다.

Root source `b6f8dc5`, C++ baseline `af78464`; 336 primary48 binary는 `34a8967`이다.
`af78464`는 기본 off인 resident diagnostic을 추가했고 이번 primary 실행에는 켜지 않는다.
고정 vLLM은 요청 계약이 같을 때만 재사용한다. Cosmos corrected raw reference를 사용한다.

## 실험과 구현 순서

1. **실현 비용 분리**: 같은 binary/engine에서 full research telemetry와 기존 dispatch telemetry를
   비교한다. Dispatch는 요청별 timeline/full candidate JSON callback을 생략하나 CUDA timing,
   RLS feedback와 phase dispatch metric은 유지한다. 계측 감소를 새로운 학습정책의 이득이라고
   부르지 않는다. Time-sensitive serving의 trajectory가 같을 것이라는 보장도 없다.
2. **Selector 정합성**: aggregate dominance가 최종 comparator에서 사용하는 service lag와
   ownership reclaim tie-break를 무시하여 최종 winner를 미리 제거할 수 있는지 재현한다.
   Workload-specific threshold 대신 같은 objective에서의 pruning invariant를 고친다.
3. **대표 screen**: 두 모델 balanced/mixed/vision-heavy/multi-image에 동일 independent 구성,
   같은 engine/KV/calibration/graph 범위를 사용한다. 기존 결과보다 나쁜 cell도 보존한다.
4. **탐색 비용 분리**: generic warmup은 유지하고 unknown serving overlap probe만 비활성화하는
   opt-in ablation을 추가한다. 336 Gemma independent의 E+P 선택은 mixed5/5, vision-heavy5/5,
   multi-image4/4 모두 measured decision이 아닌 bounded exploration이었다. Known exact/RLS
   후보와 실제 실행의 online update는 유지한다. 이 비교는 새로운 universal winner 주장도,
   zero-start 데이터 수집 문제가 해결됐다는 주장도 아니다.
5. **전체 검증**: 후보가 유망하면 두 모델 full12. 처리량 부족 cell과 작게 앞선 cell은 반복한다.
   성공 cell만 골라 합친 best-of 표를 단일 dynamic policy의 결과라고 부르지 않는다.

## 이번 단계에서 하지 않는 것

- 명시적 SLO나 workload별 E/P/D cap를 새로 튜닝하지 않는다.
- 계측 mask를 GPU SM utilization으로 해석하지 않는다.
- 같은 scalar horizon을 가진 serial P/D의 first-action uncertainty를 수정할 경우 별도 ablation으로
  분리한다. Dominance fix에 여러 다른 objective 변경을 함께 넣지 않는다.
- 기존 Gemma cross-mode exact-output 문제를 integrity 통과로 덮지 않는다.
- 디스크 여유 약3.8GiB이므로 새 엔진을 무작정 여러 개 만들거나 임의로 이전 artifact를 삭제하지 않는다.

Artifact root: `.local/results/throughput-frontier-20260926/`. 각 runner manifest가 실제 command,
source dirty hash, binary/engine identity, calibration, trace, requested/completed cells를 기록한다.
이 문서는 계획으로 시작하며 결과가 나온 뒤 같은 파일에 추가한다.

## 구현과 검증 경계

- Sampling: ready event 또는 synchronization 대상이 아닌 ticket에서 불필요한 전체 queue
  snapshot을 만들지 않는다. 기존 sync predicate의32조합을 유지하며 async/sync mode를 바꾸지 않는다.
- Selector: pruning 단계에도 기존 final comparator의 reclaim/service-lag 순서를 반영한다.
  동일 common-work horizon에서 큰 first-action uncertainty만으로 오래 기다린 P를 없애지 않는다.
  Horizon의 uncertainty 정의 자체는 이번 변경 대상이 아니다.
- Telemetry: runner의 `--telemetry-level full|dispatch`, 기본 full. Dispatch는 추가 RLS나 policy가
  아니라 연구용 instrumentation 감소다. 정확히는 unified snapshots/request timeline/counterfactual
  selects를 생략하며 PHASE_METRIC 자체는 원래도 drain 이후 직렬화했다.
- Probe ablation: `--serving-overlap-probes on|off`, 기본 on. off가 관측 기반 selector의 학습을
  모두 정지시키는 것은 아니지만 새로운 unknown pair의 serving evidence 획득은 막는다.
  Zero-start self-lock 위험을 숨기지 않는다. 기본값 승격은 별도 판단한다.

초기 compact-baseline screen 중 CPU C++ 빌드가 병행되었다. GPU inference는 직렬로 실행했지만
이 screen을 엄격한 host-idle paired confirmation이나 instrumentation-only causal gain이라고
부르지 않는다. 후속 GPU 비교 구간에서는 빌드를 끝내고 실행한다.

## 추가로 확인한 구조적 문제

### 같은 work frontier를 비교하지 않는 successor 비용

Serial P 후보가 final row 2개를 처리하고 overlap P 후보는 1개만 처리할 수 있을 때,
overlap horizon은 serial 후보에서 유도한 newly-produced D 비용을 더하면서 남은 P row의 비용을
누락할 수 있었다. 예를 들어 P2=10ms, P1=8ms, D1=2ms, 신규 D2=2.5ms,
합쳐진 D3=3ms, overlap(P1,D1)=8ms이면 기존 계산은 P-first13ms 대 overlap10.5ms다.
실제로 같은 P2+D work를 끝내려면 overlap8 + 남은 P1 8 + 신규 D2 2.5 =18.5ms다.
이는 측정 성능 숫자가 아닌 재현 가능한 synthetic cost counterexample이다.
후속 수정은 workload rule이 아니라 request/offset/token frontier의 동일성을 보장해야 한다.

### 독립 context와 독립 completion retirement는 다르다

`PhaseDispatchWorker::poll()`은 P와 D event가 모두 끝난 뒤 completion callback을 호출한다.
짧은 D가 먼저 완료돼도 긴 P가 끝나기 전에는 sampling/ready 전환이 지연될 수 있다.
단순 callback 선행은 안전하지 않다. KV length commit, scheduler active-row retirement,
sampling/logit lifetime까지 같은 phase의 completion과 함께 진행해야 한다.
원래 plan/event/metric lease를 pair 종료까지 유지하는 early retirement와,
같은 P 아래 여러 D iteration을 실행하는 기능은 서로 다른 변경이다.
후자는 per-invocation event/metric epoch와 residual authority, RLS label contract까지 필요하다.
이번 측정에는 아직 적용하지 않았다.

## 변경 이력과 unit 검증

| Commit | 범위 |
|---|---|
| `f612160` | sampling 완료 경로의 불필요한 queue snapshot 생략, 기존 synchronization predicate 유지 |
| `a4a1f2c` | dominance pruning과 최종 reclaim/service-lag tie-break의 정합성 |
| `1ec8952` | 기본 on serving probe ablation, calibration 유지, full/dispatch runner 선택 |
| `40536eb` | overlap이 생략한 canonical P/D 작업을 common horizon에 반영 |

`40536eb`는 singleton residual P 비용을 더하므로 실제로 재배칭 가능한 경우를 보수적으로 평가할 수 있다.
미측정 residual은 기존 cold prediction+uncertainty를 사용하며, 수학적으로 보장된 physical upper bound가
아니다. ID/slot/offset/count/prompt/class가 맞지 않으면 canonical work 완료 credit을 주지 않는다.
같은 frontier의 horizon 수식, 실제 dispatch, KV, scalar 학습 label/feature는 유지한다.
이번 변경은 모든 request별 완료시각의 정확성을 보장하는 수정은 아니다. 특히 다른 D frontier에서
보호 대상 D request가 실제 overlap에 포함됐는지에 대한 기존 completion 추정은 별도 점검 대상이다.

- Runtime: 706개 중704 pass, 선택적2 skip, fail0.
- Python runner: 당시33/33 pass (corrected Cosmos 기본 reference 경로 회귀 테스트 포함).
  이후 반복별 처리량 보고 테스트2개를 추가한 최종 검증은35/35 pass다.
- Scoped pre-commit, `git diff --check`: pass.
- 앞선 probe unit 실패2개는 preview의 cooldown 소모와 실제 TPOT slack을 반영하지 못한 fixture를
  수정한 뒤 전체 suite로 재검증했다. Production safety guard를 완화하지 않았다.
- 기존 global authority fixture는 오래된 P를 명시하여, legacy callback의 D 선호와 global의 P 선택을
  분리해 검증하도록 고쳤다. 새로운 dominance regression3개도 통과했다.

## 초기 전체 결과의 정확한 범위

`consistent-full24`는 `a4a1f2c` frozen binary, 두 모델×12 workload×1회다.
동일 independent workspace, dispatch telemetry, generic calibration, serving probe on을 사용한다.
호스트 C++ 빌드는 종료 후 실행했으며, 모든24 cell에서 처리량이 frozen vLLM보다 높았다.
Gemma 범위 +3.39~+120.44%, Cosmos +2.20~+55.40%다.
요청/출력길이/HTTP/capture integrity issue는0, first-EOS0이다.
이는72회 후속 반복 결과도, 모든 latency 승리도, semantic accuracy 통과도 아니다.

### 이전 구현과 출력 비교

| Workload | Cosmos raw / first-stop prefix 일치 | Gemma raw 일치 | Gemma first-stop prefix 일치 |
|---|---:|---:|---:|
| balanced | 288/288 / 288/288 | 60/64 | 60/64 |
| bimodal | 288/288 / 288/288 | 57/64 | 57/64 |
| decode-heavy | 288/288 / 288/288 | 56/64 | 56/64 |
| late-vision | 32/32 / 32/32 | 30/32 | 30/32 |
| long-prefill | 288/288 / 288/288 | 58/64 | 58/64 |
| mixed | 64/64 / 64/64 | 56/64 | 59/64 |
| multi-image | 5/5 / 5/5 | 15/20 | 16/20 |
| poisson | 64/64 / 64/64 | 50/64 | 51/64 |
| short | 48/48 / 48/48 | 47/48 | 47/48 |
| text-heavy | 64/64 / 64/64 | 56/64 | 57/64 |
| vision-heavy | 64/64 / 64/64 | 52/64 | 60/64 |
| wave-drain | 20/20 / 20/20 | 20/20 | 20/20 |
| 합계 | 1513/1513 / 1513/1513 | 557/632 | 571/632 |

비교 원본은336의 `owner-demand-full24-gpu` independent (`34a8967`) 대 `consistent-full24`다.
모델/engine/vision/config/calibration과 요청 trace identity가 같다. First-stop prefix는 종료 토큰까지
포함하며, 종료 토큰이 없으면 전체 출력이다. Gemma raw 차이75건 중61건은 종료 이전에도 다르다.
같은 `a4a1f2c` binary의 screen 대 full24에서도 대표4 workload는 raw193/212,
first-stop200/212 일치였다. 따라서 이번 코드만 원인이라고도, 무해한 FP16 수치차라고도 단정하지 않는다.
품질을 바꾸어 처리량을 얻었다는 증거는 없지만, Gemma exact/semantic 품질 gate 해결도 주장하지 않는다.

### 초기 full24 일곱 지표: Current / frozen vLLM (변화율)

336의 owner-demand independent와 비교해도24/24 처리량이 높았다. 기하평균 향상은
Gemma +10.11%, Cosmos +3.65%, 전체 +6.83%다. 두 비교의 engine/calibration/trace 객체는 같지만
**기존 full telemetry → 이번 dispatch telemetry**가 모든 cell에서 다르다. 따라서 이 개선은
계측/host/scheduler를 합친 결과이지 pruning 정책만의 순수 효과가 아니다.

| 비교 / 개선 cell 수 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 기존336 independent / vLLM | 18/24 | 9/24 | 12/24 | 21/24 | 20/24 | 21/24 | 15/24 |
| 초기 a4 full24 / vLLM | 24/24 | 16/24 | 15/24 | 21/24 | 18/24 | 22/24 | 21/24 |
| 초기 a4 full24 / 기존336 | 24/24 | 20/24 | 18/24 | 15/24 | 14/24 | 23/24 | 20/24 |

원본/공식/24개 상세 변화율은
`.local/results/throughput-frontier-20260926/a4-vs-owner-demand-independent.json`에 있다.

Latency 단위ms, 각1회. 낮은 latency와 높은 tok/s가 좋다. 아래 표는 `a4a1f2c`이며,
후속 `40536eb` 반복 결과와 섞어 평균내거나 best-of로 합치지 않는다.

| Model/variant/workload | Runs | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosmos/independent-predictor-on/balanced | 1 | 4424.71 / 4315.77 (+2.52%) | 58.34 / 112.40 (-48.10%) | 152.46 / 254.04 (-39.99%) | 12.59 / 12.20 (+3.24%) | 13.99 / 13.58 (+2.96%) | 1129.46 / 1154.35 (-2.16%) | 1779.97 / 1771.35 (+0.49%) |
| cosmos/independent-predictor-on/bimodal | 1 | 1973.37 / 1873.00 (+5.36%) | 1881.47 / 1548.74 (+21.48%) | 3957.75 / 2631.30 (+50.41%) | 17.08 / 22.59 (-24.37%) | 30.35 / 37.47 (-19.02%) | 4235.04 / 4642.60 (-8.78%) | 8763.38 / 9020.73 (-2.85%) |
| cosmos/independent-predictor-on/decode-heavy | 1 | 5231.02 / 4937.33 (+5.95%) | 63.37 / 118.57 (-46.56%) | 160.10 / 321.61 (-50.22%) | 10.64 / 11.01 (-3.33%) | 11.24 / 11.63 (-3.31%) | 2812.61 / 2969.42 (-5.28%) | 4306.63 / 4491.39 (-4.11%) |
| cosmos/independent-predictor-on/late-vision | 1 | 2543.42 / 2165.09 (+17.47%) | 130.56 / 250.16 (-47.81%) | 449.25 / 847.64 (-47.00%) | 9.27 / 10.77 (-13.91%) | 9.33 / 10.78 (-13.43%) | 1458.52 / 1792.55 (-18.63%) | 1813.24 / 2130.19 (-14.88%) |
| cosmos/independent-predictor-on/long-prefill | 1 | 1293.13 / 1123.89 (+15.06%) | 2018.69 / 1916.21 (+5.35%) | 2720.72 / 2947.28 (-7.69%) | 23.38 / 32.19 (-27.36%) | 28.44 / 37.37 (-23.89%) | 4003.88 / 4652.88 (-13.95%) | 5551.54 / 6586.48 (-15.71%) |
| cosmos/independent-predictor-on/mixed | 1 | 1141.50 / 923.32 (+23.63%) | 778.17 / 858.42 (-9.35%) | 2035.60 / 2542.45 (-19.94%) | 35.47 / 47.70 (-25.64%) | 64.28 / 83.92 (-23.40%) | 2414.66 / 2997.41 (-19.44%) | 2534.94 / 3132.22 (-19.07%) |
| cosmos/independent-predictor-on/multi-image | 1 | 325.96 / 243.90 (+33.65%) | 230.61 / 262.02 (-11.99%) | 284.61 / 401.97 (-29.20%) | 8.33 / 12.28 (-32.12%) | 11.24 / 16.27 (-30.88%) | 488.94 / 642.59 (-23.91%) | 490.69 / 654.42 (-25.02%) |
| cosmos/independent-predictor-on/poisson | 1 | 2030.49 / 1781.11 (+14.00%) | 276.16 / 435.59 (-36.60%) | 758.02 / 923.41 (-17.91%) | 19.61 / 22.35 (-12.26%) | 37.28 / 46.23 (-19.36%) | 1509.79 / 1815.85 (-16.86%) | 1951.68 / 2287.42 (-14.68%) |
| cosmos/independent-predictor-on/short | 1 | 2389.00 / 2046.18 (+16.75%) | 90.16 / 180.00 (-49.91%) | 202.03 / 256.73 (-21.30%) | 13.12 / 12.82 (+2.35%) | 26.41 / 24.13 (+9.45%) | 328.35 / 420.08 (-21.84%) | 412.43 / 492.75 (-16.30%) |
| cosmos/independent-predictor-on/text-heavy | 1 | 2008.31 / 1292.39 (+55.40%) | 397.15 / 843.35 (-52.91%) | 1115.82 / 2108.36 (-47.08%) | 22.33 / 19.29 (+15.78%) | 33.35 / 41.45 (-19.53%) | 1562.89 / 1806.03 (-13.46%) | 1658.44 / 2518.49 (-34.15%) |
| cosmos/independent-predictor-on/vision-heavy | 1 | 719.64 / 577.19 (+24.68%) | 1446.34 / 1630.87 (-11.31%) | 3043.63 / 3544.35 (-14.13%) | 45.44 / 65.15 (-30.26%) | 80.01 / 120.75 (-33.74%) | 3198.94 / 4087.81 (-21.74%) | 3377.93 / 4240.20 (-20.34%) |
| cosmos/independent-predictor-on/wave-drain | 1 | 97.93 / 95.82 (+2.20%) | 253.22 / 254.96 (-0.68%) | 307.84 / 420.78 (-26.84%) | 8.06 / 12.43 (-35.13%) | 10.93 / 17.27 (-36.70%) | 503.09 / 640.15 (-21.41%) | 514.23 / 650.51 (-20.95%) |
| gemma/independent-predictor-on/balanced | 1 | 1268.60 / 771.46 (+64.44%) | 84.71 / 134.27 (-36.91%) | 211.60 / 234.33 (-9.70%) | 15.41 / 23.76 (-35.12%) | 16.96 / 24.56 (-30.96%) | 1367.40 / 2128.03 (-35.74%) | 2106.80 / 3244.79 (-35.07%) |
| gemma/independent-predictor-on/bimodal | 1 | 844.26 / 600.16 (+40.67%) | 392.71 / 317.41 (+23.72%) | 1013.06 / 878.39 (+15.33%) | 23.28 / 28.83 (-19.23%) | 41.06 / 37.42 (+9.74%) | 3312.83 / 4389.95 (-24.54%) | 6951.93 / 9582.02 (-27.45%) |
| gemma/independent-predictor-on/decode-heavy | 1 | 1378.83 / 812.43 (+69.72%) | 92.20 / 153.61 (-39.98%) | 217.15 / 249.66 (-13.02%) | 14.35 / 23.04 (-37.71%) | 14.78 / 23.41 (-36.87%) | 3722.70 / 6006.17 (-38.02%) | 5689.22 / 9096.01 (-37.45%) |
| gemma/independent-predictor-on/late-vision | 1 | 1530.70 / 990.82 (+54.49%) | 137.11 / 137.71 (-0.44%) | 375.49 / 243.29 (+54.34%) | 13.54 / 22.55 (-39.97%) | 13.58 / 22.55 (-39.78%) | 2076.08 / 3367.53 (-38.35%) | 2668.47 / 4416.79 (-39.58%) |
| gemma/independent-predictor-on/long-prefill | 1 | 615.53 / 500.26 (+23.04%) | 666.40 / 543.92 (+22.52%) | 1644.16 / 1442.01 (+14.02%) | 28.72 / 36.37 (-21.05%) | 36.24 / 44.60 (-18.74%) | 3054.54 / 3554.56 (-14.07%) | 5404.39 / 5866.97 (-7.88%) |
| gemma/independent-predictor-on/mixed | 1 | 756.94 / 703.81 (+7.55%) | 252.06 / 276.41 (-8.81%) | 547.25 / 410.58 (+33.29%) | 24.84 / 26.48 (-6.20%) | 32.95 / 32.39 (+1.76%) | 1380.78 / 1473.77 (-6.31%) | 2121.95 / 2162.56 (-1.88%) |
| gemma/independent-predictor-on/multi-image | 1 | 400.26 / 381.34 (+4.96%) | 362.54 / 188.86 (+91.97%) | 677.85 / 227.16 (+198.40%) | 25.38 / 29.70 (-14.54%) | 39.71 / 35.54 (+11.73%) | 1149.43 / 1109.61 (+3.59%) | 1452.19 / 1300.30 (+11.68%) |
| gemma/independent-predictor-on/poisson | 1 | 925.43 / 681.95 (+35.70%) | 122.67 / 119.77 (+2.42%) | 314.44 / 169.82 (+85.16%) | 20.82 / 26.02 (-19.96%) | 24.56 / 29.23 (-15.96%) | 1589.65 / 1968.88 (-19.26%) | 2944.49 / 3502.62 (-15.93%) |
| gemma/independent-predictor-on/short | 1 | 843.73 / 567.55 (+48.66%) | 95.36 / 155.42 (-38.65%) | 217.65 / 242.31 (-10.18%) | 20.91 / 26.39 (-20.75%) | 27.78 / 30.11 (-7.73%) | 510.14 / 695.40 (-26.64%) | 829.30 / 1068.16 (-22.36%) |
| gemma/independent-predictor-on/text-heavy | 1 | 892.05 / 404.66 (+120.44%) | 184.08 / 1658.73 (-88.90%) | 473.11 / 4312.05 (-89.03%) | 21.26 / 22.47 (-5.37%) | 26.32 / 27.54 (-4.42%) | 1290.58 / 2814.97 (-54.15%) | 1632.47 / 5848.75 (-72.09%) |
| gemma/independent-predictor-on/vision-heavy | 1 | 578.80 / 559.83 (+3.39%) | 381.02 / 280.57 (+35.81%) | 653.08 / 390.78 (+67.12%) | 29.81 / 30.59 (-2.54%) | 40.84 / 40.54 (+0.72%) | 1503.23 / 1420.68 (+5.81%) | 2145.74 / 2138.33 (+0.35%) |
| gemma/independent-predictor-on/wave-drain | 1 | 97.31 / 92.62 (+5.06%) | 261.71 / 178.23 (+46.84%) | 312.72 / 204.42 (+52.98%) | 9.24 / 22.48 (-58.90%) | 11.41 / 24.58 (-53.57%) | 548.08 / 874.97 (-37.36%) | 557.40 / 884.36 (-36.97%) |

## 계측 계약과 frozen 비교 대상 상세 감사

감사 대상: source `40536eb564940e7a84239ee553ce5642856f3ea0`, 2026-09-26. 읽기 전용 감사이며 추가 실행/수정은 하지 않았다. 아래는 성능 수치가 아니라 코드 및 retained artifact 계약이다.

### 1. 정확히 무엇이 바뀌는가

두 모드 모두 `TRT_EDGELLM_EMIT_PHASE_METRICS=1`이다. `TRT_EDGELLM_PHASE_TELEMETRY_LEVEL`만 `full` → `dispatch`로 바꾼 경우의 차이는 다음과 같다. Runner 기본값은 여전히 `full`이다.

| 구분 | Full | Dispatch |
|---|---|---|
| P/D CUDA start/done elapsed timing, 공통 epoch makespan | 유지 | 유지 |
| 전체 `PhaseDispatchMetrics` 생성 및 내부 vector 보관 | 유지 | 유지 |
| Exact runtime cost 관측, P+D Scalar RLS update | 유지 | 유지 |
| E 단독 및 E+P/E+D runtime cost/RLS update | 유지 | 유지 |
| 실제 후보 생성·transition 예측·service recovery·최종 selector | 유지 | 유지 |
| sampling ticket latency와 queue/control state | 유지 | 유지 |
| HTTP token/completion 응답 및 client TTFT/TPOT/E2E | 유지 | 유지 |
| `PHASE_METRIC` JSON | 상세 필드 | 13개 compact 필드 |
| `PHASE_METRIC` JSON 생성 시점 | drain 후 | drain 후 |
| request별 `PHASE_TIMELINE` callback/JSON | 있음 | 없음 |
| unified decision/dispatch/completion callback/JSON | 있음 | 없음 |
| 상세 ready IDs·ownership hash·candidate snapshot·selector audit 출력 | 있음 | 없음 |
| 진단용 추가 Scalar/non-contextual selector 재실행 | 있음 | 없음 |
| formation episode JSON 출력 | 있음 | 없음 |
| formation episode 내부 tracker | 유지 | 유지 |
| `PHASE_ENCODER_METRIC` callback/JSON | 유지 | 유지 |
| E/P/D/Copy activity recorder | 별도 설정 | 별도 설정; 이번 runner에서는 유지 |
| resident decode shadow diagnostic | 별도 opt-in 시 가능 | unified callback이 없어 출력되지 않음 |

Compact 필드는 `dispatch_index`, `measurement_epoch`, `kind`, `prefill_batch`, `decode_batch`, `prefill_tokens`, `prefill_gpu_ms`, `decode_gpu_ms`, `makespan_gpu_ms`, `host_scheduler_decision_us`, `host_dispatch_start_us`, `host_submission_end_us`, `host_completion_us`다. Request IDs, 상세 cost-source/uncertainty/graph/KV/formation/RLS counters 등은 compact per-dispatch JSON에서 생략된다. **필드가 출력되지 않는 것과 내부 측정/학습이 제거되는 것은 다르다.**

설정/출력 근거:

- [llm_phase_context_smoke.cpp:2234](/home/sslab/TensorRT-Edge-LLM/examples/llm/llm_phase_context_smoke.cpp:2234): mode별 bool 및 metrics collection 설정.
- [llm_phase_context_smoke.cpp:2954](/home/sslab/TensorRT-Edge-LLM/examples/llm/llm_phase_context_smoke.cpp:2954): full request timeline, non-dispatch unified/formation callback, 공통 encoder callback.
- [llm_phase_context_smoke.cpp:3698](/home/sslab/TensorRT-Edge-LLM/examples/llm/llm_phase_context_smoke.cpp:3698): 두 모드 모두 phase-metric JSON은 serving drain까지 지연. 같은 위치 3705부터 compact 필드가 정의된다.
- [llm_phase_context_smoke.cpp:4287](/home/sslab/TensorRT-Edge-LLM/examples/llm/llm_phase_context_smoke.cpp:4287): owner poll thread에서 unified JSON 직렬화.
- [llm_phase_context_smoke.cpp:4610](/home/sslab/TensorRT-Edge-LLM/examples/llm/llm_phase_context_smoke.cpp:4610): timeline/encoder/formation JSON 처리.
- [llm_phase_context_smoke.cpp:3023](/home/sslab/TensorRT-Edge-LLM/examples/llm/llm_phase_context_smoke.cpp:3023): output thread는 token/completion object 직렬화 및 출력도 담당하지만, 이미 문자열인 research telemetry의 앞단 JSON 생성까지 대신하지 않는다.
- [run_lifetime_encoded_admission.py:482](/home/sslab/TensorRT-Edge-LLM/benchmarks/phase_serving/run_lifetime_encoded_admission.py:482): `--telemetry-level full|dispatch`, 기본 full.

### 2. 제거되는 계산과 남는 계산

`recordUnifiedDecision()`은 callback이 없으면 바로 반환한다. 따라서 dispatch에서는 ready/inflight 재조회, 상세 membership/ownership snapshot, candidate/audit 복사, 서명 hash, plan별 diagnostic state 저장과 두 번의 추가 selector 호출을 생략한다.

- [phaseThreeCoordinator.cpp:1943](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseThreeCoordinator.cpp:1943): callback 없는 경우 early return.
- [phaseThreeCoordinator.cpp:2086](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseThreeCoordinator.cpp:2086): 추가 Scalar select 및 non-contextual frontier 복사/선택. 실제 선택을 덮어쓰지 않는 진단값이다.
- [phaseThreeCoordinator.cpp:2114](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseThreeCoordinator.cpp:2114): unified in-flight transition 추적도 callback 없으면 반환. 이전/현재 execution 비교, activity interval 복사/검색, request별 dispatch/completion JSON payload와 상세 action-fidelity audit를 생략한다.
- [phaseGlobalScheduler.cpp:1163](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseGlobalScheduler.cpp:1163): selector는 const이고 선택용 candidate는 const다. 선택 과정의 audit pointer 분기는 출력 배열을 채울 뿐 후보/정책 상태를 바꾸지 않는다.

반대로 실제 candidate frontier와 V3 transition evaluator는 계속 계산한다. 일부 mechanism-audit용 후보 복사도 현재 코드에서 callback 밖에 존재하므로, **모든 진단 관련 CPU 비용을 제거했다고 말하면 안 된다**.

Formation tracker도 [phaseThreeCoordinator.cpp:2543](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseThreeCoordinator.cpp:2543)에서 계속 dispatch/episode를 관측한다. [phaseThreeCoordinator.cpp:2564](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseThreeCoordinator.cpp:2564)는 completed episode를 꺼낸 뒤 callback이 없으면 외부 전달만 생략한다.

### 3. RLS 학습·제어 입력이 유지되는 근거

- [phaseDispatchWorker.cpp:834](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseDispatchWorker.cpp:834): CUDA elapsed, phase completion, overlap ratio와 `mLastMetrics`를 만들고 `mScheduler.observeMetrics()`를 **출력 callback 이전에 무조건 호출**한다.
- [independentPhaseCoordinator.cpp:148](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/independentPhaseCoordinator.cpp:148): collection flag는 외부 조회용 `mMetrics.push_back`에만 적용된다. 이번 두 모드에서는 그 flag도 모두 true다.
- [phaseQueueScheduler.cpp:4450](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseQueueScheduler.cpp:4450): phase cost EWMA, decode cost, service age/TPOT 관측 유지.
- [phaseQueueScheduler.cpp:4577](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseQueueScheduler.cpp:4577): exact action cost 및 방향별 P+D contextual reward 업데이트. JSON 출력 여부를 읽지 않는다.
- [phaseThreeCoordinator.cpp:4038](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseThreeCoordinator.cpp:4038): E+P/E+D 측정 결합 및 4084/4093의 exact/RLS 업데이트. Unified callback과 무관하다.
- [phaseThreeCoordinator.cpp:4645](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseThreeCoordinator.cpp:4645): encoder 단독 cost 업데이트는 encoder metric callback 바깥이다.

기존 hard safety는 유지된다. [phaseDispatchWorker.cpp:270](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseDispatchWorker.cpp:270)의 launched phase-set 계약 및 [phaseThreeCoordinator.cpp:2481](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseThreeCoordinator.cpp:2481)의 global lease subset check는 detailed telemetry와 별개다. 다만 상세 unified fidelity 감사 출력은 사라지므로, dispatch 결과만으로 그 상세 감사까지 통과했다고 주장하면 안 된다.

### 4. 정책 중립성의 정확한 범위

이번 V3 `service-scaled-transition`에는 workload별 rule, 새 SLO, 새 RLS prior를 넣지 않는다. mode만 바꾸는 비교는 **계측/실현 비용 ablation**이다. 초기 calibration 설정, engine/graph/KV capacity, 실제 관측에 대한 학습 로직은 동일하다.

그러나 host 시간은 serving의 입력이기도 하다. callback/JSON 비용 감소로 다음 poll, completion visibility, admission, queue formation 시각이 바뀌면 이후 action과 posterior가 달라질 수 있다. 따라서 동일 알고리즘이지 동일 trajectory/동일 numerical output을 보장하는 설정은 아니다.

추가 주의: [phaseThreeCoordinator.cpp:1314](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/scheduling/phaseThreeCoordinator.cpp:1314)의 P/D-only fast path는 unified/formation callback 부재뿐 아니라 `!phasePolicyUsesTransition(mode)`도 요구한다. V3는 [phasePolicyMode.h:74](/home/sslab/TensorRT-Edge-LLM/cpp/runtime/phase/policy/phasePolicyMode.h:74)상 transition mode이므로 이번 변경이 그 fast path를 켜지는 않는다. V0/V1에는 이 조건이 다를 수 있으므로 모든 variant의 mechanism parity를 일반화하지 않는다.

최종 `40536eb` binary는 telemetry 외에도 selector/common-work 정합성 수정과 serving-probe-off ablation 기능을 포함하지만, primary 캠페인의 serving probe는 on이다. 최종 전체 성능 차이를 모두 telemetry 효과나 새로운 learned policy 효과로 귀속하면 안 된다. 초기 compact-baseline 구간의 동시 CPU build confound 역시 별도로 유지한다.

### 5. 실제 frozen vLLM 비교 대상

두 모델의 frozen vLLM 버전은 서로 다르다. 최신 설치 버전으로 이름을 덮어쓰지 않는다.

| 계약 | Cosmos frozen | Gemma frozen |
|---|---|---|
| 모델 | `nvidia/Cosmos-Reason2-2B` | `Chunity/gemma-4-E2B-it-AWQ-4bit` compatibility view |
| vLLM 실제 버전 | **0.27.1** | **0.28.0** |
| image identity | `sha256:c2f3b1b964e47809b722b5e75b61b1e7b39a50f70388cf2bf2418f16a9f31da2` | `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` |
| weights/compute | 비양자화 FP16, `--dtype float16` | INT4-AWQ checkpoint, `--dtype half` |
| KV | FP16, 3,758,096,384 bytes = 3,584 MiB | default KV dtype with half compute, 480 MiB |
| max model length | 2048 | 2048 |
| max sequences | 80 | 24 |
| client in-flight | 64 | 24 |
| batched-token budget | 8192 | 4096 |
| chunked prefill / prefix cache | on / off | on / off |
| image limit | 2 | 8 |
| graph/compiler contract | retained command 기본값; 강제 eager flag 없음 | `-cc.backend=eager` + CUDA graph sizes 1/2/4/8/16/24; `--enforce-eager`가 아님 |
| 기타 | image min/max pixels 516096; processor cache 0 | async scheduling on; expandable_segments allocator |
| 반복 | workload별 3회 시도; vision-heavy는 성공 2/실패 1, 나머지는 성공 3 | workload별 1회 diagnostic |

Cosmos 근거는 [corrected raw summary](/home/sslab/TensorRT-Edge-LLM/.local/results/review-correction-20260926/cosmos-vllm-frozen-raw-corrected.json), 그 안의 원본 [commands.json](/home/sslab/TensorRT-Edge-LLM/.local/results/v0101-forward-port/v3-service-scale-20260910/vllm-fresh-equal-full12-3x/short/run-001/commands.json), [version.json](/home/sslab/TensorRT-Edge-LLM/.local/results/v0101-forward-port/v3-service-scale-20260910/vllm-fresh-equal-full12-3x/short/run-001/client/version.json)이다. 별도 20260911 `vllm-028-cosmos-fresh` 캠페인과 혼동하면 안 된다. Warmup 수는 corrected raw의 origin별 값을 따른다. 전체를 같은 수로 덮어쓰지 않으며, vision-heavy의 실패를 성공 반복으로 대체하거나 3회 성공이라고 표현하지 않는다.

Gemma 근거는 [capacity manifest](/home/sslab/TensorRT-Edge-LLM/.local/results/gemma4-vllm-capacity-sweep-20260912/manifest.json), [run-config.sh](/home/sslab/TensorRT-Edge-LLM/.local/results/gemma4-vllm-capacity-sweep-20260912/run-config.sh), [selected version.json](/home/sslab/TensorRT-Edge-LLM/.local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12/balanced/version.json)이다. Selected aggregate는 warmup 8 requests × max32 tokens, 1 repeat를 기록한다. Compatibility view는 structurally absent shared-KV projection의 quantization metadata를 수정했고 weight 파일 자체는 변경하지 않은 경로다.

이 retained vLLM command들은 일반 HTTP server logging과 HTTP client/memory 측정을 사용한다. Current full의 per-decision candidate/ownership/phase JSON 연구 계측에 대응하는 추가 tracer는 없다. 따라서 **lower-diagnostic serving reference**라는 표현이 적절하다. 로그가 완전히 꺼졌다거나 내부 계측 비용이 0이라고 주장할 근거는 없다. Trace hash가 같아도 framework별 chat template/tokenization 및 generated output exact identity까지 같음을 뜻하지 않는다.

### 6. 현재 TensorRT engine 계약과 출처

완료된 final full24×3의 [manifest](/home/sslab/TensorRT-Edge-LLM/.local/results/throughput-frontier-20260926/final-full24-3x/manifest.json)는 source `40536eb`, binary `f2f99bf30189beafefde52257a152e670906cfb3338771fee4430db47b356dd8`, plugin `ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2`를 기록한다.

| 현재 engine | Gemma | Cosmos |
|---|---|---|
| decoder path | `workspace-corrected-20260926/gemma` | `workspace-corrected-20260926/cosmos` |
| SHA256 | `fef5210c22b0ceb064cdce07658f6dace7737d66cad7e35a3f625602f2c9405e` | `c4f873c30db785cb87aba4475cc80b935112f225c3e2204f22c2ade09356026d` |
| P / D cap | 8 / 24 | 8 / 64 |
| text chunk | 128 | 128 |
| KV dtype / max sequence / page-pool count | FP16 / 2048 / 192 | FP16 / 2048 / 256 |
| stable slots / in-flight | 24 / 24 | 80 / 64 |
| vision | E4, 280 output image-token capacity per image | E4, 512 output image-token capacity per image |
| generic calibration | 49 requests | 239 requests |

Gemma engine는 AWQ checkpoint와 external INT4 FFN sidecar를 사용하고, Cosmos는 비양자화 FP16 lineage다. HF config의 dtype 필드는 runtime engine 전체 precision을 뜻하지 않는다. Engine 및 sidecar/vision/config hash의 최종 권위는 위 manifest다. 이 engine이 vLLM과 동일 실행 바이너리·동일 배치/token budget이라는 뜻은 아니며, telemetry 제거가 그러한 기존 framework 차이를 없애지는 않는다.

정확한 결론: **학습을 끈 비교가 아니라, 동일 V3 제어 경로를 더 적은 연구 진단 비용으로 실행하는 비교**다. 학습/selector 개선, runtime 정합성 수정, 계측 감소는 성과의 서로 다른 항목으로 보고한다.

## 반복 중 확인한 원인 구분

`40536eb`의 Gemma multi-image 첫 실행은 `a4a1f2c` 첫 실행보다 처리량이 400.26→381.65 tok/s로
낮았다. 같은 20 requests, 6772 P tokens, 640 output tokens에서 E/P/D 실행 횟수는
8/28/43→10/31/47로 증가했다. P GPU event 합은 958.45→1004.48ms, E는 376.98→399.18ms,
D는 652.73→654.78ms다. Scheduler host 합 증가는 약1.16ms뿐이다. Fragmentation과 실행 순서
변화는 관측됐지만 단일 실행으로 residual-P 가격 산정이 원인이라고 결론내리지 않는다.

특히 현재 primary는 overlap P cap override=0(일반 P8 상속), chunk128 고정,
dynamic-P batching/cost-aware-P shape/oversized admission 비활성이다. 따라서 같은 snapshot의
serial/overlap P frontier는 통상 같아야 하며, residual-P singleton 비용은 보통 추가되지 않는다.
별도 preview 시각에 따른 age ordering 교차 가능성은 남는다. 반면 D는 concurrent-P 여부에 따른
component p95로 batch size를 선택하므로 residual-D가 생길 수 있다. Compact telemetry에는
residual-P/residual-D/unmatched-frontier 분해가 없어서 어느 경로가 활성화됐는지는 미확정이다.
따라서 residual P를 재배칭해 싸게 추정하는 변경을 성급하게 추가하지 않는다.

상세 원본: `.local/results/throughput-frontier-20260926/final40536-multi-image-repeat1-diagnostic.json`.
GPU event 합은 exclusive busy time이 아니며, host submission 합은 GPU 실행과 중첩될 수 있다.

## Residual E 완료 추정 수정과 탐색 대조 실험

탐색 off 감사에서 기존 오류를 발견했다. E가 실행 중인 P/D에 합류하는 residual candidate에서
exact cost가 없고 safeProbe도 false이면 incumbent 완료시간이0으로 남는다. 하지만 Scalar-ready
후보는 여전히 decisionCostKnown으로 feasibility를 통과할 수 있어 보호 대상이 즉시 완료되는 것처럼
평가될 수 있었다. Probe interval/slack 차단 때도 존재하던 경로로, off에서만 새로 생긴 오류는 아니다.
GPU dependency/ownership invariant가 깨졌다는 증거와는 다르며, 실제 trace 발생 빈도는 미확정이다.

`9db3ed3`은 해당 missing estimate에 기존 serial reference인 E isolated + incumbent remaining과
두 uncertainty 합을 사용한다. 추정값 존재 여부를 명시적으로 구분하여 exact/probe의 유효한0은
보존한다. 이는 보수적인 reference이지 overlap interference의 수학적 상한은 아니다.
E+P/E+D, probe off/interval/slack 차단, Scalar-ready feasibility와 잘못된 deadline-safe 방지를
테스트했다. Runtime708개 중706 pass, 선택적2 skip, Python35 pass, pre-commit pass.

탐색 on/off는 generic calibration과 선택된 실행의 exact/RLS 관측을 유지한다. 다만 E의 기존
safeProbe 블록은 eligibility뿐 아니라 physical makespan/uncertainty도 바꾸므로, 이 ablation을
오직 탐색 횟수만의 완전한 causal 분리라고 부르지 않는다. 새로운 shape에 대한 zero-start
self-lock 위험도 남는다. 두 모델 같은 설정을 사용하며 workload별 on/off best-of는 만들지 않는다.

Frozen binary: `.local/baselines/throughput-9db3ed3-20260926/manifest.json`.
대표4 workload×2모델 on/off screen은 각각 `residual-safe-probes-on`, `residual-safe-probes-off`에
기록한다. 이 후보의 screen은 아래40536eb 전체72회와 다른 바이너리이므로 결과를 섞지 않는다.

### 탐색 대조 결과: off는 채택하지 않음

두 screen 모두8/8 성공, 같은9db3ed3 binary/engine/calibration/trace다. 각1회이므로 통계적
인과확정은 아니지만, off가 전체 처리량을 개선한다는 근거는 없다. Default on을 유지한다.

| Model / workload | On tok/s | Off tok/s | Off / On | Frozen vLLM |
|---|---:|---:|---:|---:|
| Cosmos balanced | 4423.24 | 4467.03 | +0.99% | 4315.77 |
| Cosmos mixed | 1139.23 | 1148.82 | +0.84% | 923.32 |
| Cosmos multi-image | 311.65 | 308.34 | −1.06% | 243.90 |
| Cosmos vision-heavy | 709.48 | 707.23 | −0.32% | 577.19 |
| Gemma balanced | 1262.42 | 1264.37 | +0.15% | 771.46 |
| Gemma mixed | 737.67 | 739.47 | +0.24% | 703.81 |
| Gemma multi-image | 387.34 | 369.31 | −4.65% | 381.34 |
| Gemma vision-heavy | 575.55 | 554.96 | −3.58% | 559.83 |

Gemma vision-heavy의 P dispatch는76→84, D는146→135다. Multi-image는 P31→36,
D45→41이다. Off에서 D 평균 batch는 각각16.44→17.78,13.78→15.12로 더 조밀해졌지만
처리량은 낮아졌다. P row/token 총량은 같다.

Sampling interval을 제외한 실제 engine dispatch 교집합도 확인했다. Gemma mixed/vision-heavy/
multi-image의 E∩P는 on에서283.678/425.069/224.071ms, E∩D는209.320/360.259/33.790ms였다.
Off에서는 모두0ms다. Sampling을 포함하는 E/P/D activity mask에 남은 아주 작은 교집합을
encoder와 decoder engine의 실제 동시 실행으로 해석하지 않는다. On의 overlap이 추가 비용과
formation 변화를 상쇄한 패턴이지만, 두 실험의 token hash와 학습 trajectory도 달라 단일 변수의
결정적 인과 증명은 아니다.

추가 분해는 다음과 같다. 모두 Gemma on→off, serving epoch만 사용한다. P∩D 역시 sampling을
뺀 engine dispatch interval이며, E/P/D event 합은 중복 계산되는 시간이 포함된다.

| workload | E count | P+D dispatch | P∩D ms | idle % | E event sum ms | P event sum ms | D event sum ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 0→0 | 36→36 | 454.44→442.62 | 1.01→1.00 | 0→0 | 807.05→797.13 | 3814.75→3806.46 |
| mixed | 17→14 | 41→32 | 568.44→446.09 | 1.66→1.52 | 614.11→570.71 | 1977.89→1701.21 | 2322.38→2025.36 |
| vision-heavy | 23→20 | 39→32 | 610.28→460.28 | 2.31→1.77 | 934.12→856.36 | 2301.86→2056.16 | 2295.02→1862.14 |
| multi-image | 8→8 | 12→5 | 139.54→54.28 | 1.86→2.53 | 370.96→358.10 | 990.44→822.30 | 631.57→536.51 |

Vision-heavy는 off에서 idle과 각 phase event 비용까지 줄었지만 처리량은 낮아졌다. 따라서
idle 최소화, 개별 phase 시간 최소화, 큰 D cohort 중 하나만 최적화하는 것은 전체 처리량과 다르다.
Queue safe-probe 선택은 balanced13→0, mixed17→0, vision-heavy24→0, multi-image8→0이다.
이 count는 residual augmentation을 포함하는 P+D dispatch count와 다르다. 최종 vision summary의
starts/batches에는 calibration이 포함되므로 serving E count에는 activity CSV를 사용했다.

전체7지표와 원본 반복:
`.local/results/throughput-frontier-20260926/residual-safe-screen-report.{md,json,csv}`.
출력 integrity는 on/off 모두0 issue이며 semantic correctness는 별도다.
이후 나머지8 workload×2모델도 같은 default on으로 검증한다. 앞의8개와 이16개는 workload가
겹치지 않으며, 같은 binary/policy로 전체24개를 덮는 것이지 더 나은 실행을 골라 합치지 않는다.

## 40536eb 전체 3회 검증 완료

두 모델×12 workload×3회, **72/72 성공**, HTTP/요청 수/출력 수/capture integrity issue 0이다.
중앙값 처리량은 **24/24**에서 frozen vLLM을 넘었다. 전체 기하평균 향상은 약 **25.83%**,
Gemma **35.40%**, Cosmos **16.95%**다. 다만 **모든 반복이 이긴 것은23/24**다.

Gemma multi-image는 381.65 / 375.83 / 393.15 tok/s이며 vLLM은381.34다.
중앙값 +0.08%, 최저값 약−1.45%로 안정적인 승리 주장을 보류한다. 그다음 작은 중앙값 차이는
Cosmos wave-drain +2.33%, Gemma vision-heavy +2.61%, Cosmos balanced +2.86%다.
Fresh paired vLLM 반복이 아니므로 통계적 유의성이나 동일 시점 confidence interval을 주장하지 않는다.

| 개선 cell 수 / 24 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 40536eb / frozen vLLM | 24 | 14 | 15 | 22 | 19 | 22 | 21 |
| 40536eb / 이전336 independent | 24 | 18 | 18 | 15 | 15 | 22 | 20 |

이전336 independent 대비 처리량 기하평균 향상은 Gemma +9.47%, Cosmos +3.40%, 전체 +6.39%다.
최종 중앙값에서도24/24 처리량이 높다. 이전336은 full telemetry이므로 이 차이는 instrumentation과
공통 runtime/selector 수정을 합친 효과다. RLS 학습만의 향상으로 해석하지 않는다.

손해도 분리한다. Gemma bimodal은 이전336 대비 TPOT p95가20.931→40.522ms(+93.60%),
TPOT mean이15.431→23.081ms(+49.58%)로 나빠졌다. Gemma multi-image는 vLLM 대비
TTFT p95 +195.94%, mean +128.37%다. 따라서 처리량 승리가 latency Pareto 개선은 아니다.
개별72회는 frozen vLLM 대비71승, 이전336 대비70승이다. 두 번째 Cosmos long-prefill과
Gemma multi-image는 이전 결과보다도 각각−0.736%,−0.452%였으며 삭제하거나 제외하지 않았다.

Peak 사용량은 Gemma9405MiB, Cosmos9855MiB다. 모델/engine/정밀도/요청/출력길이/배치상한/KV를
바꾸지 않았으며, workload 이름에 따른 policy 분기를 추가하지 않았다.
최고 속도의 서로 다른 버전을 workload별로 섞은 결과가 아니다.

같은40536eb 세 반복에서 Cosmos는1513/1513 요청의 전체 token이 일치했다.
Gemma는632개 중571개 전체 token,584개 first-stop prefix가 세 반복 모두 일치했다.
즉61개 raw/48개 prefix가 하나 이상의 반복에서 달랐다. 무해한 numerical drift인지 실제 품질 문제인지
분리하지 못했으므로 **Gemma semantic/exact correctness 승격은 별도 미해결**이다.

결과:
- `.local/results/throughput-frontier-20260926/final-report.{md,json,csv}`
- `.local/results/throughput-frontier-20260926/final-output-audit.{md,json}`
- `.local/results/throughput-frontier-20260926/final-full24-3x/manifest.json`

Report는 모든 반복의 tok/s와 원본 SHA를 표시한다. 아래 latency mean은 mean-of-run-means,
p95와 처리량은 각3회 값의 중앙값이다. 단위ms, Current / frozen vLLM (변화율).

| Model/variant/workload | Runs | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosmos/independent-predictor-on/balanced | 3 | 4439.26 / 4315.77 (+2.86%) | 59.12 / 112.40 (-47.41%) | 151.91 / 254.04 (-40.20%) | 12.53 / 12.20 (+2.73%) | 13.92 / 13.58 (+2.45%) | 1124.35 / 1154.35 (-2.60%) | 1759.18 / 1771.35 (-0.69%) |
| cosmos/independent-predictor-on/bimodal | 3 | 1957.73 / 1873.00 (+4.52%) | 1872.47 / 1548.74 (+20.90%) | 4144.11 / 2631.30 (+57.49%) | 17.15 / 22.59 (-24.05%) | 27.93 / 37.47 (-25.46%) | 4252.09 / 4642.60 (-8.41%) | 9031.81 / 9020.73 (+0.12%) |
| cosmos/independent-predictor-on/decode-heavy | 3 | 5262.53 / 4937.33 (+6.59%) | 62.59 / 118.57 (-47.21%) | 212.18 / 321.61 (-34.03%) | 10.58 / 11.01 (-3.91%) | 11.14 / 11.63 (-4.18%) | 2797.67 / 2969.42 (-5.78%) | 4279.62 / 4491.39 (-4.71%) |
| cosmos/independent-predictor-on/late-vision | 3 | 2540.42 / 2165.09 (+17.34%) | 128.23 / 250.16 (-48.74%) | 438.65 / 847.64 (-48.25%) | 9.26 / 10.77 (-13.95%) | 9.35 / 10.78 (-13.32%) | 1455.53 / 1792.55 (-18.80%) | 1816.01 / 2130.19 (-14.75%) |
| cosmos/independent-predictor-on/long-prefill | 3 | 1298.74 / 1123.89 (+15.56%) | 1991.69 / 1916.21 (+3.94%) | 2744.73 / 2947.28 (-6.87%) | 23.33 / 32.19 (-27.52%) | 27.35 / 37.37 (-26.79%) | 3974.95 / 4652.88 (-14.57%) | 5497.00 / 6586.48 (-16.54%) |
| cosmos/independent-predictor-on/mixed | 3 | 1148.06 / 923.32 (+24.34%) | 717.23 / 858.42 (-16.45%) | 2065.35 / 2542.45 (-18.77%) | 37.49 / 47.70 (-21.40%) | 63.59 / 83.92 (-24.22%) | 2416.69 / 2997.41 (-19.37%) | 2514.52 / 3132.22 (-19.72%) |
| cosmos/independent-predictor-on/multi-image | 3 | 309.68 / 243.90 (+26.97%) | 276.33 / 262.02 (+5.46%) | 311.09 / 401.97 (-22.61%) | 7.61 / 12.28 (-38.02%) | 9.16 / 16.27 (-43.66%) | 512.21 / 642.59 (-20.29%) | 516.42 / 654.42 (-21.09%) |
| cosmos/independent-predictor-on/poisson | 3 | 2024.57 / 1781.11 (+13.67%) | 273.16 / 435.59 (-37.29%) | 750.82 / 923.41 (-18.69%) | 19.93 / 22.35 (-10.82%) | 38.25 / 46.23 (-17.28%) | 1526.04 / 1815.85 (-15.96%) | 1963.80 / 2287.42 (-14.15%) |
| cosmos/independent-predictor-on/short | 3 | 2424.11 / 2046.18 (+18.47%) | 109.82 / 180.00 (-38.99%) | 197.05 / 256.73 (-23.24%) | 11.14 / 12.82 (-13.10%) | 15.88 / 24.13 (-34.20%) | 332.60 / 420.08 (-20.82%) | 404.49 / 492.75 (-17.91%) |
| cosmos/independent-predictor-on/text-heavy | 3 | 2020.42 / 1292.39 (+56.33%) | 369.49 / 843.35 (-56.19%) | 1132.20 / 2108.36 (-46.30%) | 22.69 / 19.29 (+17.66%) | 33.63 / 41.45 (-18.87%) | 1556.14 / 1806.03 (-13.84%) | 1647.57 / 2518.49 (-34.58%) |
| cosmos/independent-predictor-on/vision-heavy | 3 | 714.89 / 577.19 (+23.86%) | 1434.55 / 1630.87 (-12.04%) | 3045.46 / 3544.35 (-14.08%) | 47.03 / 65.15 (-27.82%) | 82.19 / 120.75 (-31.93%) | 3244.45 / 4087.81 (-20.63%) | 3400.36 / 4240.20 (-19.81%) |
| cosmos/independent-predictor-on/wave-drain | 3 | 98.05 / 95.82 (+2.33%) | 245.32 / 254.96 (-3.78%) | 294.74 / 420.78 (-29.95%) | 8.18 / 12.43 (-34.18%) | 11.64 / 17.27 (-32.63%) | 498.85 / 640.15 (-22.07%) | 502.59 / 650.51 (-22.74%) |
| gemma/independent-predictor-on/balanced | 3 | 1262.89 / 771.46 (+63.70%) | 83.91 / 134.27 (-37.50%) | 212.55 / 234.33 (-9.29%) | 15.53 / 23.76 (-34.64%) | 16.94 / 24.56 (-31.05%) | 1376.89 / 2128.03 (-35.30%) | 2147.35 / 3244.79 (-33.82%) |
| gemma/independent-predictor-on/bimodal | 3 | 846.22 / 600.16 (+41.00%) | 398.12 / 317.41 (+25.43%) | 1048.72 / 878.39 (+19.39%) | 23.08 / 28.83 (-19.93%) | 40.52 / 37.42 (+8.29%) | 3301.19 / 4389.95 (-24.80%) | 7030.27 / 9582.02 (-26.63%) |
| gemma/independent-predictor-on/decode-heavy | 3 | 1361.12 / 812.43 (+67.54%) | 86.78 / 153.61 (-43.51%) | 204.38 / 249.66 (-18.14%) | 14.58 / 23.04 (-36.71%) | 15.00 / 23.41 (-35.93%) | 3778.16 / 6006.17 (-37.10%) | 5767.14 / 9096.01 (-36.60%) |
| gemma/independent-predictor-on/late-vision | 3 | 1530.58 / 990.82 (+54.48%) | 138.22 / 137.71 (+0.37%) | 375.59 / 243.29 (+54.38%) | 13.53 / 22.55 (-40.00%) | 13.57 / 22.55 (-39.81%) | 2076.23 / 3367.53 (-38.35%) | 2666.71 / 4416.79 (-39.62%) |
| gemma/independent-predictor-on/long-prefill | 3 | 622.28 / 500.26 (+24.39%) | 683.34 / 543.92 (+25.63%) | 1641.20 / 1442.01 (+13.81%) | 28.02 / 36.37 (-22.97%) | 35.46 / 44.60 (-20.49%) | 3015.71 / 3554.56 (-15.16%) | 5315.16 / 5866.97 (-9.41%) |
| gemma/independent-predictor-on/mixed | 3 | 742.69 / 703.81 (+5.52%) | 258.76 / 276.41 (-6.39%) | 511.56 / 410.58 (+24.60%) | 25.47 / 26.48 (-3.83%) | 34.35 / 32.39 (+6.08%) | 1412.10 / 1473.77 (-4.18%) | 2205.37 / 2162.56 (+1.98%) |
| gemma/independent-predictor-on/multi-image | 3 | 381.65 / 381.34 (+0.08%) | 431.29 / 188.86 (+128.37%) | 672.28 / 227.16 (+195.94%) | 25.58 / 29.70 (-13.87%) | 41.91 / 35.54 (+17.92%) | 1224.32 / 1109.61 (+10.34%) | 1541.68 / 1300.30 (+18.56%) |
| gemma/independent-predictor-on/poisson | 3 | 921.79 / 681.95 (+35.17%) | 126.14 / 119.77 (+5.32%) | 348.62 / 169.82 (+105.28%) | 20.75 / 26.02 (-20.23%) | 24.54 / 29.23 (-16.03%) | 1591.77 / 1968.88 (-19.15%) | 2944.52 / 3502.62 (-15.93%) |
| gemma/independent-predictor-on/short | 3 | 850.73 / 567.55 (+49.90%) | 95.04 / 155.42 (-38.85%) | 221.08 / 242.31 (-8.76%) | 21.29 / 26.39 (-19.32%) | 28.19 / 30.11 (-6.35%) | 516.29 / 695.40 (-25.76%) | 817.32 / 1068.16 (-23.48%) |
| gemma/independent-predictor-on/text-heavy | 3 | 896.14 / 404.66 (+121.45%) | 187.20 / 1658.73 (-88.71%) | 459.01 / 4312.05 (-89.36%) | 20.98 / 22.47 (-6.63%) | 26.13 / 27.54 (-5.10%) | 1280.77 / 2814.97 (-54.50%) | 1647.95 / 5848.75 (-71.82%) |
| gemma/independent-predictor-on/vision-heavy | 3 | 574.42 / 559.83 (+2.61%) | 399.47 / 280.57 (+42.38%) | 768.73 / 390.78 (+96.71%) | 29.65 / 30.59 (-3.05%) | 42.75 / 40.54 (+5.44%) | 1509.97 / 1420.68 (+6.28%) | 2043.03 / 2138.33 (-4.46%) |
| gemma/independent-predictor-on/wave-drain | 3 | 97.29 / 92.62 (+5.05%) | 256.42 / 178.23 (+43.87%) | 313.28 / 204.42 (+53.25%) | 9.45 / 22.48 (-57.97%) | 11.40 / 24.58 (-53.63%) | 549.28 / 874.97 (-37.22%) | 557.95 / 884.36 (-36.91%) |

## 최신 source 9db3ed3 최종 범위와 판정

Default serving probe **on**, 동일 independent E/P/D, chunk/graph/KV/engine/calibration을 유지했다.
On-screen8개와 remaining16개는 서로 겹치지 않는 workload 집합이며, 같은 binary/config로
**두 모델×12개, 총24개를 각1회** 검증했다. 24/24 성공, integrity issue0, 처리량24/24 우세다.
앞선40536eb 72회를 최신 binary의3회 반복이라고 세지 않는다.

| Workload | Gemma tok/s | vs vLLM | Cosmos tok/s | vs vLLM |
|---|---:|---:|---:|---:|
| balanced | 1262.42 | +63.64% | 4423.24 | +2.49% |
| bimodal | 831.74 | +38.59% | 1940.23 | +3.59% |
| decode-heavy | 1370.36 | +68.67% | 5242.34 | +6.18% |
| late-vision | 1526.62 | +54.08% | 2523.83 | +16.57% |
| long-prefill | 603.88 | +20.71% | 1291.09 | +14.88% |
| mixed | 737.67 | +4.81% | 1139.23 | +23.38% |
| multi-image | 387.34 | +1.57% | 311.65 | +27.78% |
| poisson | 929.41 | +36.29% | 2023.99 | +13.64% |
| short | 852.71 | +50.24% | 2426.53 | +18.59% |
| text-heavy | 897.43 | +121.77% | 1975.75 | +52.88% |
| vision-heavy | 575.55 | +2.81% | 709.48 | +22.92% |
| wave-drain | 97.29 | +5.04% | 98.10 | +2.38% |

처리량 비율 기하평균은 Gemma **+35.15%**, Cosmos **+16.37%**다.
TTFT mean/p95 개선은14/15개, TPOT는22/19개, E2E는22/21개로, 모든 latency 승리는 아니다.
원본7지표·각 실행값·trace hash·command·engine/binary 식별은
`.local/results/throughput-frontier-20260926/residual-safe-full24-report.{md,json,csv}`와
두 campaign manifest에 있다. 출력 감사는 `residual-safe-on-output-audit`,
`residual-safe-remaining-output-audit`다.

### 채택과 미해결을 분리

- **유지**: lazy sampling snapshot, selector tie-break 정합성, common-work residual accounting,
  missing residual-E completion fallback. 동일 source의 unit706 pass/optional2 skip, Python35 pass.
- **비교 실행 모드**: dispatch telemetry. Exact timing/RLS/selector는 유지하며 full research JSON 비용만
  줄였다. 연구 진단이 필요하면 full을 켤 수 있다. 연구 계측 감소를 새로운 learner의 성과로 부르지 않는다.
- **미채택**: serving probe off. 두 Gemma VLM에서 처리량이 악화되어 default on 유지.
- **미완료**: 최신9db3ed3의 full24×3, Gemma multi-image의 통계적으로 확실한 margin,
  Gemma token 차이의 semantic/수치 원인 분리. Frozen vLLM과 fresh paired CI도 이번 범위가 아니다.
- **자동 승격 없음**: `.local/current` artifact pointer를 바꾸지 않았다. 결과는 validation 상태다.

다음 구조적 개선 후보는 **phase별 completion retirement**다. 짧은 D가 끝났는데 긴 P의 pair barrier로
sampling/ready 전환이 기다리는 구간을 먼저 계측하고, KV length commit·scheduler row retirement·logit
lifetime을 함께 분리해야 한다. 한 P 아래 여러 D를 허용하는 확장은 별도 invocation event/lease와
RLS label 귀속이 필요하다. 이 변경의 구현 또는 성능 이득은 아직 주장하지 않는다.
