# Selector-stage 감사 연결 및 D-ready 지연 분석

## 1. 범위와 재현 계약

note253의 미완료 항목인 실제 selector 입력 기록, 경량 HTTP telemetry, Vision-heavy 원인 분류를 진행했다.
구현 커밋: `a7480e6` (기준 `14dc76c`). 기본 policy/엔진/KV 설정은 변경하지 않았다.

산출물: `.local/results/v0101-forward-port/selector-audit-20260908/`.
각 실험 디렉터리의 `commands.json`이 실행 명령 및 환경 변수의 기준이다. `manifest.json`에
바이너리·엔진 hash와 빌드 시점 tracked diff hash를 기록했다.
기존 미커밋 dispatch telemetry 변경 두 hunk는 실험 바이너리에 포함되지만 이번 구현 커밋에는
포함하지 않고 보존했다. 이번 측정은 해당 `dispatch` 모드를 사용하지 않는다.

- Cosmos-Reason2-2B FP16, 양자화 없음, P8/D64/E4, P chunk128, stable slots80, KV256 pages.
- 동일 HTTP arrival trace, client cap64, ignore-EOS, 고정 output budget.
- 매 프로세스 generic calibration: text239 또는 VLM319 requests × max32 tokens.
- 모델 export/build/inference는 notes249–251의 검증된 산출물 그대로 사용했다. 이번에는 runtime만 빌드했다.
- GPU 측정은 순차 실행. frozen vLLM은 계약이 같은 기존 결과를 재사용한다.
- 단발 계측 실행은 원인 진단용이다. 이를 비계측 serving 성능이나 통계적 유의성으로 해석하지 않는다.

## 2. 무엇을 구현했는가

### 실제 선택 시점의 값

`PhaseGlobalCandidateAudit`에 실제 action kind와 protected-phase violation mask를 추가했다.
`PhaseGlobalSelectionAudit`은 input별 평가와 실제 `PhaseGlobalDecision`을 함께 보존한다.
이는 logging 시점에서 selector를 재실행한 결과가 아니다.

`PhaseQueueScheduler::previewGlobalAction(audit)` → `IndependentPhaseAsyncServer` →
`PhaseThreeCoordinator`로 P/D 단계의 실제 선택 결과를 전달한다.
E가 경쟁하면 최종 E/P/D selector 감사와 앞 단계 P/D 감사를 함께 보존한다.
E가 없으면 P/D 선택 자체가 최종 감사가 된다.

`PhaseUnifiedEvent`는 immutable shared audit payload를 보유하며 JSON은 smoke harness에서 직렬화한다.
Runtime library에서 파일 I/O를 하지 않는다. callback이 없으면 candidate별 감사 벡터를 채우지 않는다.

필드:

- `selector_audit.inputs`: 최종 selector의 실제 입력. preview union이 아니다.
- `pd_selector_audit`: E와 비교하기 전 P/D 단계의 실제 입력과 선택. 해당 단계가 없으면 null.
- `hard_feasible`, `max_slo_violation_us`, `violation_mask`, `frontier_eligible`, `dominated`.
- `reason`, `selected_action_id`, `selected_violation_us`.
- `post_select_override`: 실제 dispatch action ID와 selector가 고른 ID가 다른지.

감사 없이 dispatch되는 경로는 `selector_audit=null`로 명시한다. forced calibration의 임의 선택을
일반 selector 판단으로 위장하지 않는다. 이번 measurement epoch의 decision은 모두 감사됐다.

기존 preview snapshot의 계산되지 않은 `max_slo_violation_us`는 NaN→JSON null로 변경했다.
이전 default0을 SLO-safe로 오해하는 문제를 막는다. `candidates`는 여전히 preview 목록이며,
실제 평가에는 반드시 `selector_audit`를 사용해야 한다.

### 경량 모드

`TRT_EDGELLM_EMIT_PHASE_METRICS=1`과 `TRT_EDGELLM_PHASE_TELEMETRY_LEVEL=audit`:

- full request/ownership snapshot은 생략한다.
- actual selector 감사와 producer request timeline을 기록한다.
- CUDA activity는 별도 `TRT_EDGELLM_PHASE_ACTIVITY_PREFIX`로 기록한다.

이 모드도 JSON/콜백/타임라인 비용이 있으므로 zero-overhead 계측은 아니다.
성능 표는 별도의 비계측 실행을 사용한다.

## 3. 테스트

- runtime/smoke 빌드 성공, TensorRT26.06, `TRT_PACKAGE_DIR=/opt/tensorrt`.
- C++ `PhaseGlobalSchedulerTest`, `PhaseQueueSchedulerTest`, `PhaseUnifiedEventTest`,
  `PhaseThreeCoordinatorPolicyTest`: **209/209 통과**.
- Python selector/producer/JSON schema 분석기: **7/7 통과**.
- Native 실제 calibration/measurement 이벤트 **2128개**를 JSON schema로 검증했다.
  미계산 preview violation은 null, 경량 모드에서 수집하지 않은 ownership hash는 0으로
  구분한다. 미수집 hash 0끼리 같다는 이유로 ownership identity를 주장하면 안 된다.
- 변경 파일 pre-commit 통과.
- 실제 P/D preview 선택 ID와 감사된 선택 ID 일치 테스트 추가.
- 분석기 테스트는 preview에 safe-D가 있더라도 실제 입력에 없으면 `absent`로 분류하는 것을 검증한다.

## 4. Vision-heavy 실제 선택 원인

최종 2단계 계측: `native-stages/`, `erf-stages/`, 각각 64requests/2464output tokens, 1회.
분석: `stages-analysis.json`; 원시 로그: `*-stages-events.jsonl`.

| 측정 항목 | Native exact | Explicit erf |
|---|---:|---:|
| 전체 decision / 감사됨 | 186 / 186 | 185 / 185 |
| D-ready 상태 decision | 179 | 175 |
| 그중 D를 포함한 action | 121 | 115 |
| 그중 D 없는 E/P action | 58 | 60 |
| D 없는 action: 최종 D 후보 부재 | 43 | 45 |
| D 없는 action: 최종 D도 예측 SLO 위반 | 15 | 15 |
| 선택 이유: 최소 위반 | 18 | 19 |
| 선택 이유: all-late efficiency recovery | 38 | 38 |
| 선택 이유: deadline-safe efficiency | 2 | 3 |
| safe-D를 두고 late action 선택 | 0 | 0 |
| selector 이후 dispatch override | 0 | 0 |

수치는 action decision 수이며 요청 수나 wall-time 비중이 아니다.
여기서 safe는 **현재 모델이 예측한 SLO-safe**이지 실제 SLO 달성 보장이 아니다.

### 원인 A: 후보 생성 단계의 TTFT hard guard

`phaseQueueScheduler.cpp::selectGlobalQueueAction()`:

```text
P queued && minimum P TTFT slack <= 0 && enablePrefillTtftHardGuard
    → prefillDeadlineExpired
    → D 단독 후보를 생성하지 않음
```

최종 E 단계의 후보 축약뿐 아니라 P/D 단계 자체에서도 D가 빠진다.
최종 E와 경쟁한 decision의 P/D 감사에서 D 부재는 native15회, erf12회였다.
E 경쟁이 없는 P-only decision도 separately 기록되므로 이 수치를 전체 D 부재 횟수로 읽으면 안 된다.
기존 `TRT_EDGELLM_PREFILL_TTFT_HARD_GUARD`는 값이 아니라 존재 여부로 켜진다.
대조 실험에서 끄려면 `=0`이 아니라 환경 변수 자체를 제거해야 한다.

### 원인 B: E→P 보호 제약 추가

native decision `9223372036854776317`:

| 단계 / 후보 | 예측 최대 SLO 위반 |
|---|---:|
| P/D 단계 D | 0 µs |
| 최종 D, E→P 지연 보호 추가 후 | 386727.05 µs |
| 최종 E | 380433.05 µs |

최종 selector는 `minimum_violation`으로 E를 골랐다. 앞 단계의 D=0만 보고 “safe D를 무시했다”고
판정하면 잘못된 결론이 된다. E 대신 D를 실행해 늦은 vision 첫 토큰을 더 미루는 비용이 반영됐다.

### 원인 C: all-late recovery는 최소 위반 선택과 다르다

native decision `9223372036854776313`은 P 위반72901.30µs, D55871.79µs였지만 P를 골랐다.
두 후보가 같은 protected **phase-kind mask**를 위반하여 `all_late_efficiency_recovery`가 적용됐다.
이 branch는 공통 위반 상태에서 efficiency를 우선한다. 동일 요청 집합이나 동일 실제 deadline이라는
뜻은 아니다. 따라서 “매번 위반 시간이 가장 작은 후보를 고른다”는 설명도 정확하지 않다.

이 세 메커니즘은 우리의 scheduling 계층이다. upstream v0.10.1 CUDA/GELU 또는 KV ownership
오류로 분류할 수 없다. 동시에 이 정책이 모든 workload에서 최적이라는 증거도 아니다.

주의: 기존 selector의 `hardFeasible()`에는 dependency/context/shape/memory뿐 아니라
**unknown overlap cost && no safe probe** 배제도 포함된다. 그러므로 감사의 `hard_feasible=false`만으로
엔진 capability가 없거나 GPU overlap이 불가능하다고 결론 내릴 수 없다.

## 5. Producer ready 경로 재확인

`ready-analysis.json`, row-weighted ms, initial P-produced D도 포함:

| 구간 mean / p95 | Native exact | Explicit erf |
|---|---:|---:|
| token commit → D-ready | 0.0015 / 0.0036 | 0.0015 / 0.0041 |
| D-ready → 다음 D host start | 26.7100 / 196.2328 | 24.0109 / 157.7776 |
| 이 대기 중 E/P/D host span으로 설명되지 않는 시간 | 0.8066 / 2.4644 | 0.6995 / 2.2890 |

현재 증거는 ready 상태 발행 자체보다 E/P 실행과 정책적 D 대기에 무게를 둔다.
host-span coverage는 GPU utilization이 아니며 phase별 합계는 overlap 때문에 더하면 안 된다.
sampling GPU 완료 시각과 CPU가 관측한 시각도 구분해야 한다.

## 6. 비계측 전체 workload 및 guard 대조

전체 12개를 각 1회 비계측 실행했다. **12/12 greedy token hash가 동일 native exact 엔진의 이전 실행과 일치**했다.

- before: note249 native exact, workload별 n=1.
- after: 이번 최종 감사 연결 바이너리, 감사 비활성, 동일 native exact, workload별 n=1.
- vllm: frozen 결과. balanced/bimodal/decode-heavy/long-prefill은 note247의 동일 client cap64 n=1,
  나머지는 note244의 n=3을 재사용한다. fresh vLLM 실행은 하지 않았다.
- 아래 latency는 client **send-relative**다. arrival→send 대기는 별도이며, 이를 포함한 전체
  scheduled-arrival 지연과 혼동하면 안 된다. 원본 requests.csv에 해당 timestamp를 보존했다.
- 구성 간 calibration 방법/비용은 동일하지 않다. 동일 serving 요청·출력 budget 비교이며,
  strict common-posterior 또는 cross-framework greedy identity 비교는 아니다.
- CSV: `full12-comparison.csv`, `full12-with-vllm.csv`; 전자는 동일 엔진 before/after의
  token identity, 메모리와 상대 지표를 포함한다. 후자는 frozen source 경로를 각 행에 기록한다.

| Workload | 구성 | token/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---|---:|---:|---:|---:|
| balanced | before | 4218.48 | 72.07 / 176.91 | 13.06 / 14.49 | 1184.37 / 1852.77 |
| balanced | after | 4124.55 | 68.43 / 172.74 | 13.44 / 15.14 | 1212.69 / 1915.36 |
| balanced | vllm | 4318.34 | 110.37 / 247.20 | 12.23 / 13.71 | 1154.26 / 1771.51 |
| bimodal | before | 1865.03 | 1957.13 / 4753.63 | 18.12 / 25.53 | 4485.65 / 9842.16 |
| bimodal | after | 1853.05 | 1970.33 / 4896.14 | 18.25 / 26.61 | 4531.50 / 10051.58 |
| bimodal | vllm | 1852.31 | 1573.46 / 2609.84 | 22.46 / 36.31 | 4689.19 / 9283.93 |
| decode-heavy | before | 4929.22 | 72.18 / 198.91 | 11.32 / 12.31 | 2992.46 / 4615.30 |
| decode-heavy | after | 4920.27 | 69.71 / 193.03 | 11.34 / 12.12 | 2999.11 / 4623.94 |
| decode-heavy | vllm | 4943.41 | 113.96 / 310.66 | 11.03 / 11.60 | 2969.50 / 4486.02 |
| late-vision | before | 2347.30 | 141.99 / 510.13 | 10.04 / 10.08 | 1579.89 / 1964.51 |
| late-vision | after | 2331.93 | 146.98 / 525.95 | 10.10 / 10.15 | 1594.01 / 1977.70 |
| late-vision | vllm | 2359.23 | 153.40 / 631.51 | 9.90 / 9.93 | 1576.03 / 1954.46 |
| long-prefill | before | 1230.80 | 2065.82 / 2662.93 | 25.30 / 29.44 | 4212.53 / 5702.41 |
| long-prefill | after | 1245.57 | 2038.82 / 2616.33 | 24.85 / 30.30 | 4143.37 / 5829.47 |
| long-prefill | vllm | 1127.42 | 1925.36 / 3027.37 | 31.92 / 36.89 | 4643.91 / 6579.62 |
| mixed | before | 1032.01 | 750.66 / 2275.65 | 34.54 / 41.27 | 2378.52 / 2761.06 |
| mixed | after | 1007.69 | 793.57 / 2380.34 | 34.08 / 42.74 | 2415.64 / 2838.40 |
| mixed | vllm | 921.48 | 874.56 / 2541.43 | 46.97 / 84.02 | 3008.38 / 3140.85 |
| multi-image | before | 291.79 | 248.92 / 329.16 | 9.47 / 12.71 | 542.39 / 547.92 |
| multi-image | after | 293.47 | 243.57 / 324.23 | 9.55 / 12.80 | 539.47 / 544.92 |
| multi-image | vllm | 244.52 | 259.81 / 402.58 | 12.42 / 16.32 | 644.30 / 653.90 |
| poisson | before | 1839.59 | 254.35 / 1000.62 | 22.30 / 40.75 | 1712.11 / 2176.63 |
| poisson | after | 1729.19 | 248.65 / 942.27 | 24.88 / 42.24 | 1849.08 / 2312.92 |
| poisson | vllm | 1800.07 | 438.11 / 902.68 | 22.19 / 45.67 | 1800.22 / 2266.55 |
| short | before | 2386.54 | 100.60 / 179.92 | 13.09 / 22.52 | 344.87 / 426.71 |
| short | after | 2378.73 | 104.59 / 183.35 | 12.88 / 22.64 | 346.61 / 428.57 |
| short | vllm | 1983.53 | 174.92 / 263.97 | 13.36 / 24.88 | 426.71 / 503.71 |
| text-heavy | before | 1839.21 | 324.59 / 1110.47 | 27.46 / 41.64 | 1738.07 / 1813.24 |
| text-heavy | after | 1806.60 | 336.15 / 1212.49 | 27.51 / 41.71 | 1754.61 / 1832.85 |
| text-heavy | vllm | 1634.76 | 421.58 / 1232.20 | 29.21 / 47.36 | 1943.42 / 2037.81 |
| vision-heavy | before | 635.13 | 1443.18 / 3415.08 | 31.92 / 39.14 | 2688.94 / 3810.97 |
| vision-heavy | after | 627.82 | 1436.05 / 3392.26 | 35.10 / 50.56 | 2787.64 / 3844.31 |
| vision-heavy | vllm | 579.20 | 1710.70 / 3691.37 | 63.70 / 119.58 | 4119.14 / 4229.37 |
| wave-drain | before | 97.34 | 266.46 / 402.43 | 9.61 / 13.76 | 564.24 / 618.59 |
| wave-drain | after | 96.12 | 270.45 / 373.95 | 9.40 / 13.23 | 561.86 / 623.24 |
| wave-drain | vllm | 95.85 | 252.76 / 418.60 | 12.43 / 17.26 | 637.86 / 649.42 |

### 판정

- 처리량: before 대비 11/12가 ±3% 범위. Poisson만 -6.00%로 단발 회귀 경고.
- 일부 latency p95도 3%를 넘겼다. 예: balanced TPOTp95 +4.49%, E2Ep95 +3.38%.
  따라서 **전체 지표 3% 무회귀 gate는 아직 통과가 아니다**. n=1이라 구현 원인으로 단정하지 않는다.
- frozen vLLM 대비 token/s 우세 8/12. balanced -4.49%, decode-heavy -0.47%,
  Poisson -3.94%, late-vision -1.16%; 모든 workload에서 vLLM을 이긴 상태는 아니다.
- 감사 기능 추가는 성능 최적화 변경이 아니다. 이번 수치 차이를 감사 API의 speedup으로 주장하지 않는다.
- Native/erf 계측 4회는 primary 표에서 제외했다. 계측 on/off의 token 결과는 동일 엔진 안에서
  일치했지만 실행 타이밍은 달라질 수 있다.

## 7. TTFT hard-guard 대조 실험

동일 native 엔진, trace, cap, calibration 요청을 유지하고 해당 환경 변수의 존재 여부만 바꾼다.
AB/BA/AB 순서로 guard-on/off 각 3회 비계측 실행한다. fresh process마다 calibration은 다시 수행되므로
동일 학습 posterior를 강제한 counterfactual replay는 아니다. 별도 off-audit 1회로 후보가 살아나는지도 확인한다.
기본값 변경이나 workload-specific 최적화는 이 실험에 포함하지 않는다.

### Vision-heavy 3회 대조 결과

| Run | token/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|
| guard-on-r1 | 634.84 | 1412.32/3355.27 | 35.17/49.79 | 2766.74/3805.94 |
| guard-on-r2 | 634.71 | 1408.82/3365.22 | 33.01/38.86 | 2685.86/3801.02 |
| guard-on-r3 | 632.91 | 1418.43/3370.58 | 36.53/53.50 | 2829.49/3809.12 |
| on-median | 634.71 | 1412.32/3365.22 | 35.17/49.79 | 2766.74/3805.94 |
| guard-off-r1 | 637.58 | 1411.91/3426.57 | 32.34/39.43 | 2672.40/3795.62 |
| guard-off-r2 | 625.27 | 1467.98/3454.04 | 34.19/47.59 | 2791.31/3859.99 |
| guard-off-r3 | 636.35 | 1413.80/3357.98 | 32.28/39.48 | 2672.70/3799.00 |
| off-median | 636.35 | 1413.80/3426.57 | 32.34/39.48 | 2672.70/3799.00 |

median은 세 run의 metric을 다시 median한 값이다. pooled request p95가 아니다.
Guard-off는 on 대비 token/s +0.26%, TPOT mean -8.04%, TPOTp95 -20.71%,
E2E mean -3.40%, E2Ep95 -0.18%. TTFT mean +0.10%, TTFTp95 +1.82%.
각 run의 TPOT tail 분산이 크므로 효과 크기가 확정됐다고 주장하지 않는다.
6개 run의 output token hash는 모두 동일했다.

별도 고정 평가 기준인 `TTFT<=500ms AND TPOT<=80ms`를 request CSV에 적용하면 on은
21/64,21/64,21/64, off는 21/64,20/64,21/64였다. 이 illustrative joint-SLO의 통과 요청 수는
개선되지 않았다. 이는 내부 request별 deadline 설정을 그대로 재현한 지표는 아니며,
TPOT tail 개선만으로 SLO capacity 개선을 주장할 수 없다는 점을 확인하는 보조 지표다.

별도 off-audit: 177/177 decision 감사, D-ready170회, D 없는 action53회.
P/D 단계의 **D 후보 부재가 0회**가 됐다. 최종 E 단계에서는 여전히 D 단독 후보가 없는 선택23회가
남는다. 따라서 TTFT hard guard에 의한 제거와 상위 단계의 P/D winner-only 축약은 서로 다른 메커니즘이다.
Off에서도 safe-D를 두고 late action 선택한 사례와 post-select override는 0회였다.

이 결과는 guard를 끄면 candidate availability가 바뀐다는 인과 증거다. 모델이 예측한 SLO와 실제
요청 SLO는 구분해야 하며, 이것만으로 guard-off가 전역적으로 더 안전하다고 결론 내리지 않는다.

## 8. 전체 12개 guard-off 재검증

`full12-guard-off/`, 각 1회, telemetry off. 비교 대상은 6절의 `full12-native/`다.
12/12에서 요청 순서 기준 greedy token hash가 일치했다. 같은 binary/engine/calibration 요청을
사용하며 posterior 자체는 fresh process에서 다시 학습한다. 아래 latency 단위는 ms다.
기계 판독 비교는 `full12-guard-comparison.json/csv`에 저장했다.

| Workload | Off token/s | On 대비 처리량 | Off TTFT mean/p95 | Off TPOT mean/p95 | Off E2E mean/p95 |
|---|---:|---:|---:|---:|---:|
| balanced | 4065.83 | -1.42% | 70.98/200.90 | 13.63/15.75 | 1230.20/1899.70 |
| bimodal | 1867.25 | +0.77% | 1965.30/4434.56 | 17.48/23.10 | 4468.91/9415.54 |
| decode-heavy | 4909.02 | -0.23% | 69.16/220.80 | 11.37/12.19 | 3005.02/4613.12 |
| late-vision | 2329.35 | -0.11% | 142.64/535.72 | 10.13/10.20 | 1593.63/1979.79 |
| long-prefill | 1315.12 | +5.58% | 1957.41/2763.20 | 23.17/28.33 | 3923.18/5520.00 |
| mixed | 1078.95 | +7.07% | 794.33/2345.63 | 35.83/51.44 | 2469.52/2660.57 |
| multi-image | 289.59 | -1.32% | 252.24/333.03 | 9.49/12.72 | 546.35/552.31 |
| poisson | 1824.75 | +5.53% | 230.66/794.17 | 23.21/41.76 | 1719.06/2198.06 |
| short | 2265.78 | -4.75% | 121.77/206.63 | 12.38/20.81 | 356.51/437.62 |
| text-heavy | 1786.95 | -1.09% | 319.08/1191.73 | 28.30/42.90 | 1773.23/1861.33 |
| vision-heavy | 626.65 | -0.19% | 1430.32/3415.99 | 37.79/51.59 | 2920.76/3857.26 |
| wave-drain | 97.00 | +0.92% | 240.13/369.17 | 10.29/13.34 | 558.99/608.39 |

### 일괄 변경을 채택하지 않는 이유

- Short: 처리량 -4.75%, TTFT mean +16.42%, p95 +12.70%.
- Balanced: TTFTp95 +16.30%, TPOTp95 +4.02%. Decode-heavy도 TTFTp95 +14.39%.
- Long-prefill: 처리량 +5.58%, E2E mean/p95 약 -5.31%지만 TTFTp95 +5.61%.
- Mixed: 처리량 +7.07%, E2Ep95 -6.27%지만 TPOTp95 +20.35%, E2E mean +2.23%.
- Poisson은 이 단발 비교에서 전반적으로 개선됐다. 하지만 다른 trace의 손실을 무시하고
  Poisson 전용 guard 예외를 넣지는 않는다.
- 추가 Vision-heavy off 실행은 TPOT mean 37.79ms로 7절의 off 3회보다 나빴다.
  따라서 앞선 TPOT 개선은 반복 조건에 민감하며 보장된 효과로 표현할 수 없다.

결정: **기본 guard-on 유지**, guard-off는 실험 ablation으로만 보관한다.
이는 guard가 최적이라는 증명이 아니라, 현재 증거로 단순 제거를 production에 승격할 수 없다는 뜻이다.

## 9. Poisson guard-on 추가 반복

6절 단발 회귀를 확인하려고 같은 설정의 fresh process 2회를 추가했다.
원본 `full12-native` 1회와 `poisson-confirmation` 2회를 합친 총 3회이며,
세 run 모두 output token hash가 동일하다. 반복별 결과를 먼저 보고 metric별 median을 계산했다.

| Run | token/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|
| original | 1729.19 | 248.65/942.27 | 24.88/42.24 | 1849.08/2312.92 |
| repeat1 | 1750.92 | 237.88/953.09 | 23.58/39.01 | 1766.94/2214.63 |
| repeat2 | 1821.85 | 234.18/1002.12 | 23.41/42.04 | 1749.94/2214.69 |
| run-median | 1750.92 | 237.88/953.09 | 23.58/42.04 | 1766.94/2214.69 |

Median token/s는 이전 note249 단발 1839.59 대비 -4.82%, frozen vLLM 1800.07 대비 -2.73%다.
반복 범위 1729.19–1821.85로 변동이 있지만 회귀 경고를 해소하지 못했다.
단, 이전 버전도 같은 시점에서 반복한 A/B가 아니므로 감사 API가 원인이라고 단정할 수 없다.
E2E mean/p95는 frozen vLLM보다 낮지만 TPOT mean은 높다. 처리량 하나로 승패를 단정하지 않는다.

## 10. 완료 범위와 남은 성능 gate

이번 작업에서 완료한 것:

1. P/D 및 E/P/D 두 단계 actual selector audit 연결과 preview/actual 구분.
2. 경량 audit telemetry, producer-ready 분석, 선택 이유 분석기 및 테스트.
3. Native/erf 각각 실제 후보·SLO·선택 원인 추적. KV/vision 엔진 변경 없이 관측 경로 추가.
4. 비계측 default full12와 guard-off full12 총 24회, Vision-heavy guard on/off 추가 6회,
   Poisson 추가 2회 완료. 계측 진단 5회는 primary 성능 표에서 제외했다.
5. 기존 결과 및 계약이 맞는 frozen vLLM과 TTFT/TPOT/E2E mean/p95 비교.
6. Runtime 변경 `a7480e6` 커밋. 기존 미커밋 dispatch-only 변경과 note248 초안은 보존했다.

**완료하지 못한 목표는 전체 workload/latency 무회귀 및 모든 경우 vLLM 우세다.**
이를 달성했다고 보고하거나 guard-off를 새 champion으로 지정하지 않는다.
성능 변경 없이 계측부터 확정한 이번 단계와, 이후 실제 selector 정책 변경은 분리한다.

다음 구현은 workload label별 분기가 아니라 다음 두 메커니즘을 같은 snapshot에서 대조해야 한다.

- P/D에서 winner 하나만 넘기지 않고 bounded feasible frontier를 최종 E/P/D 선택에 전달한다.
  동일 후보 수 상한과 CPU decision p95를 유지하면서 D 후보의 전달 누락을 분리 검증한다.
- TTFT 만료 시 D를 생성 단계에서 제거하는 정책과, D를 유지하되 E/P/D 전체 critical-path
  위험으로 판단하는 정책을 비교한다. 기존 guard 제거 실험과 동일하다고 취급하지 않는다.

우선 Poisson/Short/Mixed에서 이전 binary와 교대 반복해 host/runtime 변화와 정책 효과를
분리한 뒤 전체12로 확장한다. 엔진, KV pool, calibration 요청, graph 지원, client cap은 고정한다.
메모리/엔진 재빌드로 현상을 섞거나 workload별 최적 설정을 합쳐 단일 성능으로 주장하지 않는다.
