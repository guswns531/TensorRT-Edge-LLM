<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 338. 최신 처리량 후보 추가 2회 검증과 코드 정리 감사

## 요청과 고정 계약

최신 `9db3ed3`을 두 모델×12 workload에서2회 더 실행한다. 337의 같은 바이너리 첫24회와
합쳐 workload별3회로 판단한다. 이번 요청은 불필요한 코드의 확인이며 삭제·정책 변경은 하지 않는다.
GPU 검증 중 runtime/build를 수정하거나 C++ 빌드를 병행하지 않는다.

- Frozen binary: `.local/baselines/throughput-9db3ed3-20260926/bin`.
- Smoke SHA256: `c60e6a06e290cee734b1af725ae3fab09d283236731a14493c9c60a668d5d947`.
- Plugin SHA256: `ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2`.
- Source checkout at launch: `bd4c581`, clean. Serving source는 `9db3ed3`.
- Same engines: `workspace-corrected-20260926/gemma` and `cosmos`.
- Same V3, independent E/P/D, dispatch telemetry, serving probes on, text chunk128, D graph on/P graph off.
- Same batch/KV/admission/calibration/trace contract as337; no workload-specific setting change.
- Fresh process and same generic calibration procedure per cell; no posterior transferred from another workload.
- vLLM references are frozen unchanged-contract results, not fresh paired measurements.

Preflight: RTX3080 10GiB, GPU memory1MiB, utilization0%, temperature60°C, no running Docker container.
Disk available about3.2GiB. No model/engine rebuild, no artifact deletion.

## 원본과 추가 실행의 식별

| Source | Cells | Meaning |
|---|---:|---|
| `throughput-frontier-20260926/residual-safe-probes-on` | 8 | first run, representative4×2 models |
| `throughput-frontier-20260926/residual-safe-remaining16` | 16 | first run, remaining8×2 models |
| `throughput-confirmation-20260926/additional48` | 48 | additional2×full12×2 models |

경로는 모두 `.local/results/` 아래다. 서로 다른 root의 `repeat-001`은 같은 실행이 아니다.
합칠 때 original root/cell/aggregate hash를 유지한다. 첫24개와 추가48개만 합치며 `40536eb`나
`a4a1f2c`, 탐색 off 결과는 섞지 않는다. Group 결과의 median들을 평균내지 않고72개 raw aggregate를
직접 집계한다. Binary/plugin/engine/calibration/trace와 실행 설정의 일치도 확인한다.

재현 명령:

```bash
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma cosmos --full12 --variants independent \
  --transition-predictors on --repeats 2 \
  --build-root .local/baselines/throughput-9db3ed3-20260926/bin \
  --binary-source-commit 9db3ed3 \
  --gemma-engine-dir .local/artifacts/v0101-forward-port/workspace-corrected-20260926/gemma \
  --cosmos-engine-dir .local/artifacts/v0101-forward-port/workspace-corrected-20260926/cosmos \
  --result-root .local/results/throughput-confirmation-20260926/additional48 \
  --telemetry-level dispatch --serving-overlap-probes on --compress-closed-logs
```

## 판단 기준

1. Completed/requested48/48, 고정 출력·HTTP/capture integrity 확인.
2. 같은 source의72개 합산: 처리량 중앙값/최저/최고/반복별 승패, TTFT·TPOT·E2E mean/p95.
3. Frozen vLLM 및 같은 계약의 이전 후보와 비교. 다른 engine/계측/정밀도의 역사적 최고 숫자를
   무조건 같은 순위에 넣지 않는다. 계측이 다른336은 시스템 전체 변화 비교로 명시한다.
4. 특히 Gemma multi-image, vision-heavy, Cosmos balanced/wave-drain의 작은 margin과 반복 변동 확인.
5. Gemma token 차이를 별도 기록. 처리량 우세는 semantic correctness나 모든 latency 우세가 아니다.
6. 최신이 비교군 전체를 모든 workload/지표에서 이기지 않으면 단일 절대 champion이라고 부르지 않는다.

## 코드 정리 원칙

- 기본 off는 dead code의 증거가 아니다. Caller·CLI/config·테스트·benchmark 사용을 같이 확인한다.
- V0/V1/V2 비교 경로와 V3 내부 exact/Scalar fallback의 역할을 구분한다.
- 후보 제거/잔여 작업/완료시간 보정 및 regression test는 유지한다.
- 실패한 ablation과 출력 불일치의 근거는 보존한다.
- 우선순위는 사용되지 않는 compatibility wrapper → 중복 configuration 작성 → 실험용 진단 경계 정리다.
- 삭제 후보마다 파일/호출 근거/조건/회귀 테스트를 적는다. 이번에는 실제 삭제하지 않는다.

## 코드 감사: 삭제 후보와 보존 경계

이 절은 소스 읽기·caller 검색에 근거한다. 삭제 후 성능을 측정한 결과가 아니다. 이번 실행의
serving binary는 그대로 두며 아래 항목을 자동 삭제하지 않는다.

### 이미 정리된 것

`cpp/runtime/phase/policy/phasePolicyMode.h`의 지원 정책은 V0/V1/V2/V3다.
과거 Completion/Decomposed/Selective 정책을 대량으로 다시 삭제할 상황은 아니다.
`run_policy_warmup_matrix.py`에 남은 옛 환경변수 이름들은 구형 command를 재생할 때 제거하는
호환성 sanitizer이며 실행 모델 구현이 아니다.

### 낮은 위험의 구조 정리

| 대상 | 범위 | 판단 / 필요한 검증 |
|---|---:|---|
| `cpp/runtime/scheduling/{phaseActionPlan,phaseDeadline,phaseGlobalCostModel,phaseGlobalScheduler,phaseOwnershipHorizon,phaseReadySnapshot}.h` | 6개, 각21줄, 총126줄 | canonical header로 전달만 하는 shim. 5개는 tracked caller가 없고 GlobalScheduler만 구현·테스트의2개 include가 있음. 내부 include 이관 후 삭제 가능하지만 외부 include 호환성이 깨질 수 있음 |
| `PhaseOptimizationContext.prefillTokens` | 필드·대입 약4줄 | 생산 코드에서 대입만 하고 읽지 않음. struct layout/API 변경을 감안해야 함 |
| `phaseThreeCoordinator.cpp`의 `mediaFull`, `multiMediaReady` | 동일식·중복 조건 | 한 이름으로 통합 가능. compiler가 이미 제거할 수 있으므로 성능 개선으로 주장하지 않음 |
| `llm_phase_context_smoke.cpp`의 protected-completion JSON serializer | 동일한 약21줄이2곳 | full telemetry 두 출력 경로의 schema/value 동일성 검증 후 공통 helper로 통합 |
| benchmark 파일 identity/SHA 함수 | 6–7줄씩 여러곳 | 공통화 가능하나 작은 유틸을 합치기 위한 import 복잡도가 더 클 수 있음. 우선순위 낮음 |

6개 shim 외에 `phaseAsyncServer.h`도 alias 성격이지만 상위 통합 API 이름이므로 자동 삭제
대상으로 묶지 않는다. 위 line count는 source 정리 규모이지 GPU memory 절감량이 아니다.

### 의미 있는 후보: 사용되지 않는 queue-residence 학습

`PhaseTransitionPredictor`에는 live deterministic burst/overlap-token controller와 shadow decode
queue-residence RLS가 함께 있다. 후자는 매 D 완료 시 `observeDecodeQueueWait()`로 갱신하지만,
`predictDecodeQueueWait()` 및 관련 telemetry의 tracked consumer는 unit test뿐이다.

```text
유지: recommendedDecodeBurst / recommendedOverlapPrefillTokens
유지: PhaseContextualPdModel의 주력 Scalar RLS, exact CUDA timing
분리 후보: decode queue-residence shadow RLS update / storage / prediction API
```

- 구현: `cpp/runtime/scheduling/phaseTransitionPredictor.cpp:74–180`.
- 선언·상태: `cpp/runtime/phase/policy/phaseTransitionPredictor.h`.
- update caller: `cpp/runtime/scheduling/phaseQueueScheduler.cpp:4604–4608`.
- `ShadowQueueLearningCannotChangeBurstOrOverlap` 테스트가 학습과 실제 추천의 독립성을 검증한다.
- 약160–220줄의 코드·선언이 분리/삭제 후보이며 정확한 diff 산정 전 추정이다.

이 부분이 가장 실질적인 단순화 후보다. 다만 공개 accessor가 있어 외부 consumer 부재까지
증명한 것은 아니다. CPU overhead 감소가 실제 enqueue timing을 바꿀 수 있으므로 제거 시에는
새 바이너리의 all12 회귀를 별도로 확인해야 한다. 현재 검증 바이너리에 이 삭제를 소급 적용하지 않는다.

### 사용 중인 연구 경로: retirement 결정이 먼저

Encoder preparation의 `e-dynamic-shadow`, `e-transition-shadow`, `e-dynamic-active`는 runner에서
여전히 opt-in으로 호출된다. Note306에서 active promotion은 실패했고307은 transition shadow를
연구용으로 유지한다. Coordinator의 selector/helper/state 약250–350줄은 분리 후보지만 dead code는
아니다. 종료하기로 결정하면 runner·tests·notes의 재현 계약과 함께 정리해야 한다. 공통 formation
primitive나 encoder batch 선택 전체를 삭제하면 안 된다.

v0.10.0 pair-eligibility 호환 옵션도 `compare_version_activity.py` 및
`replay_retained_policy_commands.py`의 실제 caller가 있다. 버전 비교 재현을 폐기하지 않는 한 보존한다.

### 이름 때문에 삭제하면 안 되는 것

`legacyQueueDecision` / Global Disabled 경로는 V3의 `previewMechanismPlan()`이 재사용한다.

```text
V3 candidate preview
  → scheduler clone
  → Global Disabled + explicit phase kind
  → next()의 공통 batch construction
```

따라서 legacy라는 이름만 보고 제거하면 최신 P/D candidate formation도 깨진다.
다음도 유지한다.

- V0/V1/V2 동일 runtime ablation, exact/covering/cold fallback, uncertainty.
- Serving probe on/off: 학습 evidence 수집과 탐색 ablation 기능이다.
- Full telemetry: dispatch mode에서 줄였지만 보호 누락·row order·false-safe 진단에 필요하다.
- KV/vision lease, context single-inflight, event, shape, memory feasibility 검사.
- `phaseSchedulerOptions.inc`의36개 D cost fallback: cold service reference, admission, dynamic
  decode sizing이 아직 소비한다. 제거는 단순 정리가 아니라 별도 policy 변경이다.

### 구조 개선이나 삭제와 별개인 것

`phaseQueueScheduler.cpp`의 transition-predictor env와 `independentPhaseAsyncServer.cpp`의
resident-decode-shadow env는 composition에서 config로 전달하는 편이 경계가 명확하다.
현재 동작을 유지하며 caller를 이관해야 하므로 별도 작은 리팩터링으로 다룬다.

또한 현재 runner가 `.local/results/v0101-forward-port/replay-tools/`의 세 Python 파일을 실제로
사용한다. `run_phase_http_trace_bench.py`, `run_phase_openai_gateway.py`, `run_vllm_trace_bench.py`는
오래된 결과 디렉터리에 있다는 이유로 지우면 안 된다. Tracked benchmark로 이관하되 HTTP·token·
timing 계약을 유지하는 작업이 필요하다.

세 파일 비교 결과 HTTP trace harness와 vLLM client에는 tracked 대응 파일이 없다. Gateway는
`scripts/phase_openai_gateway.py`와 이름만 유사하다. 현재 replay gateway에는 calibration 제어,
SSE 준비 토큰 최대64개 묶음 전송, backend 종료 오류 전파와 drain lifecycle이 있고 tracked gateway에는
이들이 없다. 반대로 tracked에는 handler 종료 시 cancel이 있다. 단순 경로 변경은 calibration과
TTFT/TPOT 측정 계약을 바꾼다. 먼저 현재 소스를 byte-identical하게 보존·이관하고 lifecycle 통합은
별도 변경으로 검증해야 한다.

권장 순서는 성능 확정 → 중복 shim/serializer 정리 → shadow queue learner 분리 → env composition
경계 정리다. 연구 기능 retirement나 source 이관을 한 번에 섞지 않는다.

## 성능 결과

추가48/48회와 기존 같은 바이너리24회를 합쳐 **72/72회, 두 모델×12 workload×3회**를 완료했다.
실패·누락0, HTTP/count/capture integrity issue0, first-token EOS0이다. 실험 중 설정 변경이나 실패 실행
대체는 하지 않았다. 아래 비교는 사전 보관된 동일 workload 계약의 frozen vLLM이며 fresh paired baseline이 아니다.

| 모델 | 처리량 기하평균 향상 vs vLLM | 중앙값 우세 workload | 개별 실행 우세 | Peak VRAM 범위 |
|---|---:|---:|---:|---:|
| Gemma AWQ | +35.73% | 12/12 | 36/36 | 9,385–9,405 MiB |
| Cosmos FP16 | +16.64% | 12/12 | 35/36 | 9,739–9,857 MiB |
| 전체 | +25.82% | 24/24 | 71/72 | — |

Peak 범위는 workload별 세 실행의 최대 사용량을 모은 min/max다. 새 engine나 KV capacity 변경으로
얻은 결과가 아니다. vLLM Gemma reference는 workload별1회, Cosmos는 대부분 성공3회이며 vision-heavy는
성공2회와 실패 시도1회를 보존한다. 이번3회만으로 paired95% CI나 보편적인 승리를 주장하지 않는다.

중요한 예외는 **Cosmos wave-drain의 마지막 반복**이다. `98.10 / 98.02 / 92.68 tok/s`, frozen vLLM은
`95.82 tok/s`였다. 중앙값은 +2.29%지만 마지막 실행은 −3.28%다. 이 실행도 그대로 포함했다.
Gemma multi-image는 `387.34 / 395.67 / 387.02`, vLLM `381.34`, 중앙값 +1.57%다.
Gemma vision-heavy는 `575.55 / 571.27 / 564.62`, vLLM `559.83`, 중앙값 +2.04%다.

### 일곱 지표 요약

변화율은 workload별 비율의 기하평균이다. 처리량은 양수, latency는 음수가 개선이다.
괄호는 개선 workload 수이며, 좋은 전체 평균이 일부 workload 회귀를 숨기지 않도록 함께 표시한다.

| 모델 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma | +35.73% (12) | −15.49% (5) | +10.05% (4) | −25.01% (12) | −16.88% (8) | −25.58% (10) | −27.36% (10) |
| Cosmos | +16.64% (12) | −25.63% (9) | −22.89% (11) | −16.91% (10) | −21.46% (11) | −15.00% (12) | −15.78% (11) |
| 전체 개선 수 /24 | 24 | 14 | 15 | 22 | 19 | 22 | 21 |

따라서 처리량 전체 중앙값 우세는 확인됐지만 **모든 지표 우세는 아니다**. 특히 Gemma TTFT p95는
기하평균도 악화다. Multi-image의 요청 latency와 vision-heavy의 평균 E2E 등은 아직 별도 손익이 있다.

### 기존 후보보다 최고인가

| 기준 | 기준 반복 | Gemma 처리량 변화 | Cosmos 처리량 변화 | 전체 변화 | 빨라진 cell |
|---|---:|---:|---:|---:|---:|
| 직전 `40536eb` | 3 | +0.25% | −0.27% | −0.009% | 10/24 |
| 초기 `a4a1f2c` | 1 | −0.33% | −0.51% | −0.42% | 11/24 |
| Note336 independent `34a8967` | 1 | +9.75% | +3.13% | +6.38% | 24/24 |

`40536eb` 대비 전체 성능은 사실상 비슷한 관측치다. 통계적 동등성을 증명한 것은 아니지만,
최신이 모든 경우에서 더 빠르다는 근거도 없다. 최신 `9db3ed3`은 residual E 후보의 보호 완료시간이
0으로 남는 문제를 고친 정합성 수정까지 포함하면서 앞선 처리량 수준을 유지한 후보로 해석한다.

`a4a1f2c`는 common-work 비용 보정 전이고 각1회라 숫자가 더 높다고 바로 되돌릴 이유는 아니다.
336 비교는 같은 engine/trace/calibration이지만 full→dispatch 계측 차이도 포함하므로 순수 RLS 또는
단일 selector 수정의 효과라고 부르지 않는다. 세 비교의24행×7지표 원본 비율은
`historical-comparison.json`에 보존했다.

| 최신 / 직전405 지표 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전체 기하평균 변화 | −0.009% | −0.19% | +0.26% | +0.46% | +1.50% | +0.007% | +0.05% |
| 최신이 나은 cell /24 | 10 | 16 | 12 | 11 | 10 | 13 | 10 |

**결론: 검증된 최고 수준의 처리량 후보이지만 단일 절대 최고·모든 지표 champion은 아니다.**
따라서 기존 정책·진단 경로를 대량 제거할 근거로 쓰지 않는다. 정합성 수정과 재현성은 유지하고,
증거가 명확한 중복/미사용 보조 학습만 다음 리팩터링 후보로 삼는다.

### 출력 반복성과 품질 경계

| 모델 | 서로 다른 요청 수 | 세 번 raw exact | 세 번 through-first-stop exact |
|---|---:|---:|---:|
| Cosmos | 1,513 | 1,513/1,513 | 1,513/1,513 |
| Gemma | 632 | 573/632 | 587/632 |

Gemma59개 요청의 raw 출력이 세 반복 중 하나 이상에서 달랐고, 이 중45개는 첫 종료 토큰까지도 달랐다.
분모는3배로 부풀리지 않은 unique request 수다. HTTP/capture 통과와 semantic accuracy는 별개이며,
기존 Gemma 수치/row-order/ownership 관련 정확성 과제가 이번에도 해결됐다고 할 수 없다.

`.local/current` 승격이나 source/artifact 삭제는 하지 않았다.

### 유일한 반복 패배: Cosmos wave-drain

아래 실행 순서는 기존1회, 추가1회, 추가2회다. 모두20개 요청, P12,300 tokens,
D620 row-tokens, 출력640 tokens이며 요청별 output hash도 일치한다. 원본은 `residual-safe-remaining16`
및 `additional48`의 Cosmos wave-drain aggregate, request CSV, compressed gateway log와 activity trace다.

| 항목 | 기존1회 | 추가1회 | 추가2회 |
|---|---:|---:|---:|
| tok/s | 98.104 | 98.017 | 92.679 |
| client duration ms | 6,523.709 | 6,529.472 | 6,905.547 |
| E2E mean / p95 ms | 495.82 / 502.12 | 497.98 / 505.53 | 563.10 / 693.15 |
| TTFT mean / p95 ms | 240.56 / 294.75 | 242.94 / 300.47 | 246.55 / 332.33 |
| TPOT mean / p95 ms | 8.234 / 11.747 | 8.227 / 11.664 | 10.211 / 15.880 |
| P dispatch | 16 | 16 | 16 |
| D dispatch | 125 | 129 | 192 |
| mean D batch | 4.960 | 4.806 | 3.229 |
| P GPU event 합계 ms | 736.64 | 748.67 | 554.30 |
| D GPU event 합계 ms | 818.32 | 885.06 | 1,409.57 |
| E interval 합계 ms | 667.07 | 668.93 | 627.56 |
| E/P/D active union ms | 1,884.84 | 1,904.78 | 2,297.74 |
| 모든 stream idle ms | 4,624.90 | 4,610.92 | 4,578.91 |
| activity-window idle 비율 | 71.05% | 70.77% | 66.59% |

Activity mask는 CUDA event 작업 구간이지 SM utilization이 아니다. Activity window와 HTTP client
duration은 시작·종료 경계가 같지 않을 수 있으며, phase별 시간 합계에는 overlap이 있어 union과 다르다.

손실 대부분은 마지막5개 요청 wave에서 관측됐다.

| 마지막 wave | 기존1회 | 추가1회 | 추가2회 |
|---|---:|---:|---:|
| D dispatch | 31 | 32 | 93 |
| D batch 구성 | D5×31 | D1×1 + D5×30 + D4×1 | D1×62 + D3×31 |
| D GPU event 합계 ms | 199.34 | 219.71 | 623.02 |
| E2E mean ms | 494.97 | 499.38 | 712.64 |
| request19 TTFT ms | 254.87 | 259.94 | 677.00 |
| request19 E2E ms | 492.18 | 498.10 | 874.28 |
| 두 번째 E start − 첫 E start ms | 114.53 | 115.52 | 561.77 |

전체 P shape multiset은 모두 `P1×12 + P2×4`지만 좋은 실행의 wave별 순서는 `P1,P1,P1,P2`,
느린 실행은 `P1,P1,P2,P1`이었다. 마지막 wave에서는 `P2+D1` 뒤 D3/D1이 반복됐고 마지막 요청이
늦게 첫 토큰을 받은 뒤 단독 decode로 끝났다. 따라서 단순 P 실행 횟수가 아니라 ready 순서와
다음 D cohort 형성의 차이가 중요하다.

Mask의 E∩P 시간은 `332.62/329.46/73.93ms`, E∩D는 `0/0/139.21ms`, P∩D는
`20.83/82.72/103.66ms`였다. E engine 횟수는 `9/9/8`이나 compact activity에는 각 E batch의
membership이 없으므로 E shape까지 단정하지 않는다.

동일 D5 GPU median은 `6.416/6.418/6.416ms`로 같다. Client dispatch p95는
`0.214/0.185/0.126ms`, scheduler 총비용은 `3.843/3.986/3.506ms`, host submission 총비용은
`126.58/134.38/119.70ms`여서 느린 실행의 전체 비용 증가를 설명하지 않는다.
종료 로그의 growth waits0도 memory pressure의 직접 증거를 제공하지 않지만, 종료 요약만으로
모든 순간 memory pressure를 완전히 배제하지는 않는다.

안전한 결론은 **달라진 E/P 순서와 overlap placement에 동반된 D cohort 분할이 추가 GPU service와
drain tail을 설명한다**는 것이다. 왜 그 선택이 발생했는지는 현재 compact telemetry만으로
generic calibration posterior, async readiness, selector/probe를 구분할 수 없다. 실제 회귀 사례로
보존하며 timing noise라고 버리거나 설정을 바꿔 재실행으로 대체하지 않는다.

따라서 다음 성능 작업은 이 마지막 wave를 full diagnostic mode로 분리 재현해 decision 원인을
확인하는 것이다. Full 계측의 host perturbation도 대조해야 하며, 이 문제 확인 전 비교/계측 경로를
지우면 원인 분석이 어려워진다.

### 합산·감사 재현

```bash
python3 benchmarks/phase_serving/report_workspace_revalidation.py \
  --merged-campaign 'latest3x=.local/results/throughput-frontier-20260926/residual-safe-probes-on,.local/results/throughput-frontier-20260926/residual-safe-remaining16,.local/results/throughput-confirmation-20260926/additional48' \
  --output-prefix .local/results/throughput-confirmation-20260926/latest3x-report

python3 benchmarks/phase_serving/audit_phase_campaign_outputs.py \
  --campaign .local/results/throughput-frontier-20260926/residual-safe-probes-on \
  --additional-campaign .local/results/throughput-frontier-20260926/residual-safe-remaining16 \
  --additional-campaign .local/results/throughput-confirmation-20260926/additional48 \
  --output-prefix .local/results/throughput-confirmation-20260926/combined-output-audit
```

Strict reporter는 같은 frozen serving source/hash, plugin/container/runner/replay-tools, engine/vision/
sidecar/calibration, trace, command/env를 요구한다. Command에서 cell output directory와 대응 bind mount만
정규화한다. Source checkout 메타데이터가 달라도 같은 frozen binary이면 허용한다. 완료 manifest가 없는
aggregate, 실패 cell, 중복 raw source는 성공 반복으로 세지 않는다.

Output auditor는 source별 HTTP/count/capture integrity와 동일 trace/stop contract만 확인하며 runtime
동일성은 reporter의 역할이다. Raw/through-first-stop hash는 request별 모든 반복이 일치해야 일치로
센다. Through-first-stop에는 첫 종료 토큰이 포함되고 종료 토큰이 없으면 전체 출력이다. Gemma는
engine EOS 외에 기존 diagnostic stop 집합 `{1,50,106}`도 사용한다. 이 지표는 semantic accuracy가 아니다.

이번 도구 수정은 inference binary를 바꾸지 않았다. CPU unit tests: merge14, output audit19,
기존 runner35, 합계68개 통과. 신규 테스트는 source repeat 충돌, 불일치 contract, 실패/누락, raw3회
집계 및 종료 이후만 다른 출력의 구분을 포함한다.

### 전체24개 일곱 지표

단위ms. 각 칸은 Current / frozen vLLM (변화율)이다. 평균 latency는 세 run mean의 산술평균,
처리량과 p95는 세 run 값의 중앙값이다. 아래 p95는 세 실행의 모든 요청을 합친 pooled p95가 아니다.

| Model/variant/workload | Runs | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosmos/independent-predictor-on/balanced | 3 | 4444.18 / 4315.77 (+2.98%) | 58.37 / 112.40 (-48.07%) | 149.92 / 254.04 (-40.98%) | 12.49 / 12.20 (+2.42%) | 13.98 / 13.58 (+2.92%) | 1120.28 / 1154.35 (-2.95%) | 1746.61 / 1771.35 (-1.40%) |
| cosmos/independent-predictor-on/bimodal | 3 | 1949.34 / 1873.00 (+4.08%) | 1880.19 / 1548.74 (+21.40%) | 4412.67 / 2631.30 (+67.70%) | 17.01 / 22.59 (-24.69%) | 28.22 / 37.47 (-24.69%) | 4262.94 / 4642.60 (-8.18%) | 9306.06 / 9020.73 (+3.16%) |
| cosmos/independent-predictor-on/decode-heavy | 3 | 5242.34 / 4937.33 (+6.18%) | 64.61 / 118.57 (-45.51%) | 211.46 / 321.61 (-34.25%) | 10.60 / 11.01 (-3.66%) | 11.28 / 11.63 (-2.95%) | 2804.45 / 2969.42 (-5.56%) | 4298.31 / 4491.39 (-4.30%) |
| cosmos/independent-predictor-on/late-vision | 3 | 2520.10 / 2165.09 (+16.40%) | 128.45 / 250.16 (-48.65%) | 469.30 / 847.64 (-44.63%) | 9.35 / 10.77 (-13.12%) | 9.42 / 10.78 (-12.63%) | 1468.39 / 1792.55 (-18.08%) | 1830.60 / 2130.19 (-14.06%) |
| cosmos/independent-predictor-on/long-prefill | 3 | 1311.19 / 1123.89 (+16.67%) | 1990.70 / 1916.21 (+3.89%) | 2673.08 / 2947.28 (-9.30%) | 22.92 / 32.19 (-28.80%) | 26.60 / 37.37 (-28.82%) | 3938.68 / 4652.88 (-15.35%) | 5362.07 / 6586.48 (-18.59%) |
| cosmos/independent-predictor-on/mixed | 3 | 1139.23 / 923.32 (+23.38%) | 801.62 / 858.42 (-6.62%) | 2039.60 / 2542.45 (-19.78%) | 35.15 / 47.70 (-26.32%) | 62.44 / 83.92 (-25.59%) | 2413.12 / 2997.41 (-19.49%) | 2545.99 / 3132.22 (-18.72%) |
| cosmos/independent-predictor-on/multi-image | 3 | 311.49 / 243.90 (+27.71%) | 276.13 / 262.02 (+5.38%) | 308.04 / 401.97 (-23.37%) | 7.63 / 12.28 (-37.89%) | 9.17 / 16.27 (-43.61%) | 512.50 / 642.59 (-20.24%) | 513.43 / 654.42 (-21.54%) |
| cosmos/independent-predictor-on/poisson | 3 | 2023.99 / 1781.11 (+13.64%) | 346.82 / 435.59 (-20.38%) | 788.69 / 923.41 (-14.59%) | 19.29 / 22.35 (-13.71%) | 36.53 / 46.23 (-20.99%) | 1564.62 / 1815.85 (-13.84%) | 1958.48 / 2287.42 (-14.38%) |
| cosmos/independent-predictor-on/short | 3 | 2408.34 / 2046.18 (+17.70%) | 100.07 / 180.00 (-44.41%) | 197.86 / 256.73 (-22.93%) | 12.12 / 12.82 (-5.46%) | 20.40 / 24.13 (-15.46%) | 330.18 / 420.08 (-21.40%) | 406.68 / 492.75 (-17.47%) |
| cosmos/independent-predictor-on/text-heavy | 3 | 1990.47 / 1292.39 (+54.01%) | 360.51 / 843.35 (-57.25%) | 1134.06 / 2108.36 (-46.21%) | 23.50 / 19.29 (+21.84%) | 38.19 / 41.45 (-7.87%) | 1573.20 / 1806.03 (-12.89%) | 1673.70 / 2518.49 (-33.54%) |
| cosmos/independent-predictor-on/vision-heavy | 3 | 713.01 / 577.19 (+23.53%) | 1433.07 / 1630.87 (-12.13%) | 3085.57 / 3544.35 (-12.94%) | 47.13 / 65.15 (-27.66%) | 80.97 / 120.75 (-32.94%) | 3242.76 / 4087.81 (-20.67%) | 3407.25 / 4240.20 (-19.64%) |
| cosmos/independent-predictor-on/wave-drain | 3 | 98.02 / 95.82 (+2.29%) | 243.35 / 254.96 (-4.55%) | 300.47 / 420.78 (-28.59%) | 8.89 / 12.43 (-28.45%) | 11.75 / 17.27 (-32.00%) | 518.97 / 640.15 (-18.93%) | 505.53 / 650.51 (-22.29%) |
| gemma/independent-predictor-on/balanced | 3 | 1262.42 / 771.46 (+63.64%) | 84.90 / 134.27 (-36.77%) | 211.52 / 234.33 (-9.73%) | 15.52 / 23.76 (-34.65%) | 16.97 / 24.56 (-30.90%) | 1376.73 / 2128.03 (-35.31%) | 2148.97 / 3244.79 (-33.77%) |
| gemma/independent-predictor-on/bimodal | 3 | 847.12 / 600.16 (+41.15%) | 394.61 / 317.41 (+24.32%) | 1006.08 / 878.39 (+14.54%) | 23.33 / 28.83 (-19.07%) | 41.57 / 37.42 (+11.08%) | 3317.29 / 4389.95 (-24.43%) | 7005.88 / 9582.02 (-26.89%) |
| gemma/independent-predictor-on/decode-heavy | 3 | 1377.52 / 812.43 (+69.56%) | 86.64 / 153.61 (-43.60%) | 217.32 / 249.66 (-12.95%) | 14.38 / 23.04 (-37.59%) | 14.77 / 23.41 (-36.91%) | 3725.69 / 6006.17 (-37.97%) | 5697.27 / 9096.01 (-37.37%) |
| gemma/independent-predictor-on/late-vision | 3 | 1531.62 / 990.82 (+54.58%) | 138.43 / 137.71 (+0.53%) | 374.20 / 243.29 (+53.81%) | 13.54 / 22.55 (-39.93%) | 13.58 / 22.55 (-39.80%) | 2078.74 / 3367.53 (-38.27%) | 2667.16 / 4416.79 (-39.61%) |
| gemma/independent-predictor-on/long-prefill | 3 | 619.51 / 500.26 (+23.84%) | 659.23 / 543.92 (+21.20%) | 1642.47 / 1442.01 (+13.90%) | 28.89 / 36.37 (-20.58%) | 36.21 / 44.60 (-18.82%) | 3062.51 / 3554.56 (-13.84%) | 5382.98 / 5866.97 (-8.25%) |
| gemma/independent-predictor-on/mixed | 3 | 744.20 / 703.81 (+5.74%) | 250.22 / 276.41 (-9.48%) | 515.53 / 410.58 (+25.56%) | 25.27 / 26.48 (-4.60%) | 34.16 / 32.39 (+5.49%) | 1395.80 / 1473.77 (-5.29%) | 2192.59 / 2162.56 (+1.39%) |
| gemma/independent-predictor-on/multi-image | 3 | 387.34 / 381.34 (+1.57%) | 413.31 / 188.86 (+118.85%) | 689.17 / 227.16 (+203.38%) | 25.28 / 29.70 (-14.89%) | 41.09 / 35.54 (+15.63%) | 1196.99 / 1109.61 (+7.88%) | 1493.89 / 1300.30 (+14.89%) |
| gemma/independent-predictor-on/poisson | 3 | 929.41 / 681.95 (+36.29%) | 126.85 / 119.77 (+5.91%) | 387.69 / 169.82 (+128.29%) | 20.55 / 26.02 (-21.01%) | 24.44 / 29.23 (-16.38%) | 1576.22 / 1968.88 (-19.94%) | 2921.88 / 3502.62 (-16.58%) |
| gemma/independent-predictor-on/short | 3 | 854.10 / 567.55 (+50.49%) | 92.16 / 155.42 (-40.71%) | 219.80 / 242.31 (-9.29%) | 20.72 / 26.39 (-21.48%) | 28.07 / 30.11 (-6.75%) | 502.96 / 695.40 (-27.67%) | 815.23 / 1068.16 (-23.68%) |
| gemma/independent-predictor-on/text-heavy | 3 | 894.05 / 404.66 (+120.94%) | 175.37 / 1658.73 (-89.43%) | 422.40 / 4312.05 (-90.20%) | 21.36 / 22.47 (-4.93%) | 26.06 / 27.54 (-5.36%) | 1288.31 / 2814.97 (-54.23%) | 1651.70 / 5848.75 (-71.76%) |
| gemma/independent-predictor-on/vision-heavy | 3 | 571.27 / 559.83 (+2.04%) | 371.81 / 280.57 (+32.52%) | 662.19 / 390.78 (+69.45%) | 30.18 / 30.59 (-1.33%) | 44.59 / 40.54 (+9.99%) | 1502.15 / 1420.68 (+5.73%) | 2074.79 / 2138.33 (-2.97%) |
| gemma/independent-predictor-on/wave-drain | 3 | 97.29 / 92.62 (+5.04%) | 251.15 / 178.23 (+40.91%) | 312.56 / 204.42 (+52.90%) | 9.55 / 22.48 (-57.51%) | 11.41 / 24.58 (-53.59%) | 547.18 / 874.97 (-37.46%) | 558.55 / 884.36 (-36.84%) |

### 반복별 처리량 원본

First8/first16과 additional48의 source-qualified repeat ID를 보존한다. 모든 반복은 같은 binary지만
새 프로세스에서 generic calibration을 다시 수행하므로 posterior까지 같다는 뜻은 아니다.

| Model/variant/workload | Per-repeat tok/s (change; beats frozen) | Min | Max | Frozen | Wins/runs |
|---|---|---:|---:|---:|---:|
| cosmos/independent-predictor-on/balanced | source-01:residual-safe-probes-on/repeat-001: 4423.24 (+2.49%; yes); source-03:additional48/repeat-001: 4490.49 (+4.05%; yes); source-03:additional48/repeat-002: 4444.18 (+2.98%; yes) | 4423.24 | 4490.49 | 4315.77 | 3/3 |
| cosmos/independent-predictor-on/bimodal | source-02:residual-safe-remaining16/repeat-001: 1940.23 (+3.59%; yes); source-03:additional48/repeat-001: 1957.71 (+4.52%; yes); source-03:additional48/repeat-002: 1949.34 (+4.08%; yes) | 1940.23 | 1957.71 | 1873.00 | 3/3 |
| cosmos/independent-predictor-on/decode-heavy | source-02:residual-safe-remaining16/repeat-001: 5242.34 (+6.18%; yes); source-03:additional48/repeat-001: 5222.12 (+5.77%; yes); source-03:additional48/repeat-002: 5281.29 (+6.97%; yes) | 5222.12 | 5281.29 | 4937.33 | 3/3 |
| cosmos/independent-predictor-on/late-vision | source-02:residual-safe-remaining16/repeat-001: 2523.83 (+16.57%; yes); source-03:additional48/repeat-001: 2516.58 (+16.23%; yes); source-03:additional48/repeat-002: 2520.10 (+16.40%; yes) | 2516.58 | 2523.83 | 2165.09 | 3/3 |
| cosmos/independent-predictor-on/long-prefill | source-02:residual-safe-remaining16/repeat-001: 1291.09 (+14.88%; yes); source-03:additional48/repeat-001: 1337.62 (+19.02%; yes); source-03:additional48/repeat-002: 1311.19 (+16.67%; yes) | 1291.09 | 1337.62 | 1123.89 | 3/3 |
| cosmos/independent-predictor-on/mixed | source-01:residual-safe-probes-on/repeat-001: 1139.23 (+23.38%; yes); source-03:additional48/repeat-001: 1152.48 (+24.82%; yes); source-03:additional48/repeat-002: 1139.13 (+23.37%; yes) | 1139.13 | 1152.48 | 923.32 | 3/3 |
| cosmos/independent-predictor-on/multi-image | source-01:residual-safe-probes-on/repeat-001: 311.65 (+27.78%; yes); source-03:additional48/repeat-001: 311.49 (+27.71%; yes); source-03:additional48/repeat-002: 309.98 (+27.09%; yes) | 309.98 | 311.65 | 243.90 | 3/3 |
| cosmos/independent-predictor-on/poisson | source-02:residual-safe-remaining16/repeat-001: 2023.99 (+13.64%; yes); source-03:additional48/repeat-001: 1840.81 (+3.35%; yes); source-03:additional48/repeat-002: 2046.66 (+14.91%; yes) | 1840.81 | 2046.66 | 1781.11 | 3/3 |
| cosmos/independent-predictor-on/short | source-02:residual-safe-remaining16/repeat-001: 2426.53 (+18.59%; yes); source-03:additional48/repeat-001: 2394.27 (+17.01%; yes); source-03:additional48/repeat-002: 2408.34 (+17.70%; yes) | 2394.27 | 2426.53 | 2046.18 | 3/3 |
| cosmos/independent-predictor-on/text-heavy | source-02:residual-safe-remaining16/repeat-001: 1975.75 (+52.88%; yes); source-03:additional48/repeat-001: 1990.47 (+54.01%; yes); source-03:additional48/repeat-002: 2009.03 (+55.45%; yes) | 1975.75 | 2009.03 | 1292.39 | 3/3 |
| cosmos/independent-predictor-on/vision-heavy | source-01:residual-safe-probes-on/repeat-001: 709.48 (+22.92%; yes); source-03:additional48/repeat-001: 713.01 (+23.53%; yes); source-03:additional48/repeat-002: 715.88 (+24.03%; yes) | 709.48 | 715.88 | 577.19 | 3/3 |
| cosmos/independent-predictor-on/wave-drain | source-02:residual-safe-remaining16/repeat-001: 98.10 (+2.38%; yes); source-03:additional48/repeat-001: 98.02 (+2.29%; yes); source-03:additional48/repeat-002: 92.68 (-3.28%; no) | 92.68 | 98.10 | 95.82 | 2/3 |
| gemma/independent-predictor-on/balanced | source-01:residual-safe-probes-on/repeat-001: 1262.42 (+63.64%; yes); source-03:additional48/repeat-001: 1256.78 (+62.91%; yes); source-03:additional48/repeat-002: 1264.94 (+63.97%; yes) | 1256.78 | 1264.94 | 771.46 | 3/3 |
| gemma/independent-predictor-on/bimodal | source-02:residual-safe-remaining16/repeat-001: 831.74 (+38.59%; yes); source-03:additional48/repeat-001: 847.12 (+41.15%; yes); source-03:additional48/repeat-002: 850.92 (+41.78%; yes) | 831.74 | 850.92 | 600.16 | 3/3 |
| gemma/independent-predictor-on/decode-heavy | source-02:residual-safe-remaining16/repeat-001: 1370.36 (+68.67%; yes); source-03:additional48/repeat-001: 1381.41 (+70.03%; yes); source-03:additional48/repeat-002: 1377.52 (+69.56%; yes) | 1370.36 | 1381.41 | 812.43 | 3/3 |
| gemma/independent-predictor-on/late-vision | source-02:residual-safe-remaining16/repeat-001: 1526.62 (+54.08%; yes); source-03:additional48/repeat-001: 1534.90 (+54.91%; yes); source-03:additional48/repeat-002: 1531.62 (+54.58%; yes) | 1526.62 | 1534.90 | 990.82 | 3/3 |
| gemma/independent-predictor-on/long-prefill | source-02:residual-safe-remaining16/repeat-001: 603.88 (+20.71%; yes); source-03:additional48/repeat-001: 619.51 (+23.84%; yes); source-03:additional48/repeat-002: 621.79 (+24.29%; yes) | 603.88 | 621.79 | 500.26 | 3/3 |
| gemma/independent-predictor-on/mixed | source-01:residual-safe-probes-on/repeat-001: 737.67 (+4.81%; yes); source-03:additional48/repeat-001: 754.79 (+7.24%; yes); source-03:additional48/repeat-002: 744.20 (+5.74%; yes) | 737.67 | 754.79 | 703.81 | 3/3 |
| gemma/independent-predictor-on/multi-image | source-01:residual-safe-probes-on/repeat-001: 387.34 (+1.57%; yes); source-03:additional48/repeat-001: 395.67 (+3.76%; yes); source-03:additional48/repeat-002: 387.02 (+1.49%; yes) | 387.02 | 395.67 | 381.34 | 3/3 |
| gemma/independent-predictor-on/poisson | source-02:residual-safe-remaining16/repeat-001: 929.41 (+36.29%; yes); source-03:additional48/repeat-001: 927.63 (+36.03%; yes); source-03:additional48/repeat-002: 932.10 (+36.68%; yes) | 927.63 | 932.10 | 681.95 | 3/3 |
| gemma/independent-predictor-on/short | source-02:residual-safe-remaining16/repeat-001: 852.71 (+50.24%; yes); source-03:additional48/repeat-001: 854.10 (+50.49%; yes); source-03:additional48/repeat-002: 866.88 (+52.74%; yes) | 852.71 | 866.88 | 567.55 | 3/3 |
| gemma/independent-predictor-on/text-heavy | source-02:residual-safe-remaining16/repeat-001: 897.43 (+121.77%; yes); source-03:additional48/repeat-001: 893.87 (+120.89%; yes); source-03:additional48/repeat-002: 894.05 (+120.94%; yes) | 893.87 | 897.43 | 404.66 | 3/3 |
| gemma/independent-predictor-on/vision-heavy | source-01:residual-safe-probes-on/repeat-001: 575.55 (+2.81%; yes); source-03:additional48/repeat-001: 571.27 (+2.04%; yes); source-03:additional48/repeat-002: 564.62 (+0.86%; yes) | 564.62 | 575.55 | 559.83 | 3/3 |
| gemma/independent-predictor-on/wave-drain | source-02:residual-safe-remaining16/repeat-001: 97.29 (+5.04%; yes); source-03:additional48/repeat-001: 97.32 (+5.07%; yes); source-03:additional48/repeat-002: 97.28 (+5.03%; yes) | 97.28 | 97.32 | 92.62 | 3/3 |
