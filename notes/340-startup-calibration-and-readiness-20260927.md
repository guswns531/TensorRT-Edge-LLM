# Startup calibration / serving readiness — 2026-09-27

## 목적과 범위

모델 가중치·독립 E/P/D context를 적재한 뒤, 요청을 받기 전에 현재 엔진/GPU에서 비용을 측정한다.
외부 cost registry와 workload 이름에 따른 설정 선택은 사용하지 않는다.
기존 champion 기본값은 유지하며, 새 경로는 `--startup-calibration`으로 명시적으로 검증한다.

시작 시 관측 가능한 물리적 비용과 serving 중에만 알 수 있는 arrival/queue 상태를 분리한다.
짧은 calibration으로 모든 shape의 p95, 최적 배치, 최적 KV pool 또는 RLS 수렴을 보장하지 않는다.

## 구현·검증 순서

1. capability-derived D/P probe frontier: geometric batch + 실제 maximum, 짧은/긴 prompt.
   full-output page reservation과 sequence 한계를 만족하는 shape만 실행한다.
2. auto 모드에서 과거 고정 D 비용표를 제거하고 process-local measured batching 사용.
   비용이 없으면 기존 largest-available fallback, 임의로 작은 batch를 선호하지 않는다.
3. graph 준비 후 graph/eager를 구분해 비용을 수집한다. 같은 tracker를 serving에서도 사용한다.
4. 개별 probe의 실제 관측 coverage를 확인하며 반복한다. 시간 예산은 신규 probe admission만
   중단하고 이미 시작한 GPU 작업과 lease는 반드시 drain한다.
5. generic image의 E calibration을 준비 전 실행한다. 미지원/누락 shape를 숨기지 않는다.
6. report를 저장한 후 ready를 알린다. strict 모드는 coverage 누락 시 ready를 내보내지 않는다.
7. CPU contract tests → C++ tests → GPU startup smoke → 동일 binary off/on 비교.
   throughput, TTFT/TPOT/E2E mean/p95, memory, startup 비용을 분리 기록한다.
8. 두 모델 full12 검증 전에는 기본 runtime pointer를 변경하지 않는다.

## 고정되는 안전 경계

- 기존 엔진의 profile, 물리 KV pool, stable slots, context binding은 변경하지 않는다.
- graph capture 이후 KV/phase I/O 주소를 재할당하지 않는다.
- graph cache 최대 크기와 실행 최대 배치는 다르다. 현재 graph priming mechanism을 사용한다.
- E formation wait와 arrival-dependent batching은 serving 관측으로 판단한다.
- 새로운 사용자 SLO는 추가하지 않는다. startup 시간 예산은 SLO가 아니라 준비 비용 상한이다.
- 준비 완료는 선택한 frontier의 관측 충족이다. 모든 엔진 shape/overlap 방향 학습 완료가 아니다.
- sample p95는 소수 표본의 경험적 통계이며 tail confidence guarantee가 아니다.
- synthetic calibration 이후 one-request HTTP marker는 기존 calibration-end 제어 경계를 실행하기
  위한 것이다. 49/239개 trace calibration을 대신하는 학습 수렴 기준으로 해석하지 않는다.

## 시작 전 자동화로 해결되지 않는 부분

pool/vision/graph memory 전체의 최적 배분, encoder arrival wait, chunk의 workload 최적값은
startup isolated timing만으로 결정할 수 없다. 이 구현은 물리 pool 확대/축소나 배치 상한
최적화기를 포함하지 않는다. 실제 memory headroom과 지원 frontier를 보고하고 serving에
기존 lifetime admission을 유지한다. 무리하게 모든 가정을 제거했다고 주장하지 않는다.

## 구현 위치

- `cpp/runtime/scheduling/phaseServingExecutionOptions.{h,cpp}`:
  startup 옵션 검증, engine/slot/page/sequence 제약을 만족하는 probe 계획.
- `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}`:
  준비 판정과 serving이 동일한 graph/eager variant를 조회하도록 연결.
- `examples/llm/llm_phase_context_smoke.cpp`:
  HTTP phase backend의 준비 orchestration, 기존 D 비용표 비활성화, 실측 D batching,
  P/D coverage 반복, E calibration, drain, JSON report, strict ready gate.
- `benchmarks/phase_serving/run_lifetime_encoded_admission.py`:
  `--startup-calibration` 및 `independent-startup` A/B variant.
  독립 E/P/D, chunk128, engine, 실제 HTTP request trace, KV 크기는 동일하게 유지.
- `benchmarks/phase_serving/report_startup_calibration.py`:
  같은 binary의 baseline/new 비교, 누락 pair 검출, 전체 latency 지표와 준비 시간.
- `benchmarks/phase_serving/analyze_lifetime_encoded_admission.py`:
  compact dispatch schema의 vanilla D token accounting 수정. `decode_tokens`가 있으면
  우선 사용하고, 생략된 경우에만 `decode_batch`를 사용한다.

이 단계의 orchestration은 실제 HTTP benchmark backend에 연결했다. 별도
`PhaseServingRuntime::create()` production API가 같은 startup 절차를 실행하도록
통합한 것은 아니다. 해당 API의 자동 준비까지 완료했다고 해석하면 안 된다.

## 무엇을 자동화했고 무엇을 유지했는가

| 항목 | 새 모드의 동작 | 남아 있는 경계 |
|---|---|---|
| D timing | 고정 비용표를 사용하지 않고 CUDA 실측 tracker 조회 | 미관측 key는 largest-available fallback |
| P/D probe batch | geometric batch 및 허용 maximum에서 자동 생성 | engine/사용자 cap을 확대하지 않음 |
| context coverage | 짧은 prompt와 full-output KV에 맞는 긴 prompt target | cost key는 기존 512-token bucket; 같은 key의 기존 관측은 재사용 |
| graph | 기존 graph priming 후 해당 graph/eager key로 관측 | cache 상한/재할당 정책을 새로 최적화하지 않음 |
| cold P/E 비용 | 관측된 key는 공유 tracker의 비용을 사용 | 미관측 shape의 cold fallback 상수는 남음 |
| calibration 요청 수 | D/P key coverage까지 필요한 synthetic 요청 반복 | 최소 관측수/시간 예산은 안전·신뢰도 설정이며 자동 수렴 증명 아님 |
| E coverage | generic image의 E1/E2/E4 직접 관측 | 모든 image resolution 또는 overlap 방향 coverage가 아님 |
| memory | physical KV 범위 안에서 probe 구성·headroom 보고 | KV/vision/graph 간 최적 재분배 구현 아님 |
| ready | strict coverage 부족 시 ready 금지 | process liveness HTTP endpoint를 새로 구현한 것은 아님 |
| E wait/chunk | 기존 serving 동작 유지 | arrival-dependent 최적값을 startup에서 확정하지 않음 |

`rounds=0`인 report row는 같은 cost key의 기존 관측을 사용했다는 뜻이다.
특히 Cosmos D64의 짧은/긴 prompt target이 모두 context bucket1이면 동일 관측을
재사용할 수 있다. 그 긴 prompt 자체를 새로 실행했다고 주장하면 안 된다.
`frontier_covered`는 이 key coverage를 뜻하며 exact geometry 전수 측정, tail 안정성,
RLS posterior 수렴 또는 모든 overlap 방향 authority 확보와 다르다.

## 구현 중 발견·수정한 문제

1. Gemma는 scheduler용 `estimateInputTokens()`가 0일 수 있지만 encoder profile용
   `estimateProfileInputTokens()`는 유효하다. 기존 E calibration의 무조건 양수 전제는
   Gemma를 거부했다. 비용 key의 단위를 바꾸지 않고 profile capacity 검증을 별도로 적용했다.
2. graph 관측이 없다는 이유로 eager 관측을 가져오면 serving과 준비 판정이 달라진다.
   현재 execution-variant supplier를 공통 조회하도록 수정했다.
3. compact dispatch log에서 `decode_tokens`가 생략되어 기존 분석기가 token mismatch를
   잘못 보고했다. 실제 balanced HTTP 출력은 양쪽 모두 5,440개, D token은 5,376개로 일치했다.
4. 실행 중 binary가 바뀌면 기존 campaign identity guard가 중단한다. `smoke-r2`는
   이 보호 장치로 중단된 diagnostic이며 성능 자료에 포함하지 않는다.

## 검증 계약

- 기반 source: `b850927054bdd9b192f2e052f7ba0736ecc04e76` + campaign의 `source.patch`.
- 최종 smoke SHA256: `0ef5a1aa79ecff9c298780920543ee73e4d287d61a55656229914c1797c04b93`.
- plugin SHA256: `ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2`.
- 보존 binary: `.local/baselines/startup-calibration-20260927/bin`.
- GPU: RTX3080 10GB. 두 기존 엔진의 export/build lineage는 그대로이며 kernel/weight 변경 없음.
- C++ runtime: 704 pass / 2 optional skip. startup/execution-option XML subset: 10 pass.
- Python: runner/analyzer 41 pass, startup report 3 pass.
- `smoke-r3`: Gemma/Cosmos short 각각 strict startup와 HTTP 추론 통과.
  준비 보정 시간은 각각 17.767초/16.631초. 모델 적재와 전체 process-to-ready 시간은 포함하지 않음.
- full12 A/B root: `.local/results/startup-calibration-20260927/full12-paired`.
  `independent`와 `independent-startup`, 두 모델 ×12 ×2 =48 cells, 각 1회.
- 기존 vLLM은 request/engine precision/output contract가 바뀌지 않아 frozen 결과를 재사용.
  fresh vLLM을 실행했다고 주장하지 않는다.
- 기본 `.local/current/*/runtime`과 promotion registry는 변경하지 않는다.

## 첫 원인 분해

Gemma balanced에서 P dispatch는 41→46, D dispatch는 293→321,
D 평균 batch는 18.35→16.75였다. Decode-heavy는 D 872→960,
평균 batch 18.64→16.93이었다. 같은 D useful token 수를 더 많은 dispatch로 처리했다.

하지만 이 결과를 D 비용표 교체만의 효과라고 단정할 수 없다.
기본 shape warmup도 함께 달라졌다:

| 모델 | 기존 shape-warmup requests | 자동 shape-warmup requests | 기존/자동 P+D observations |
|---|---:|---:|---:|
| Gemma | 776 | 632 | 76 / 60 |
| Cosmos | 3,416 | 1,568 | 376 / 164 |

여기에 자동 경로의 추가 D/P probe와 E fixture calibration이 들어가고, HTTP generic
49/239 requests는 1-request epoch marker로 바뀐다. 따라서 이 비교는 **startup 계약
전체의 A/B**이며 physical cost/학습량/초기 posterior 각각의 순수 ablation이 아니다.
총 calibration이 반드시 줄었다고 말할 수도 없다. Shape warmup 수만으로 전체 준비 비용을
비교하지 말고 `startup.json`과 gateway의 calibration 로그를 함께 봐야 한다.

## 재현 명령

```bash
LLM_SDK_DIR="$PWD" python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma cosmos --full12 \
  --variants independent independent-startup --repeats 1 \
  --build-root .local/baselines/startup-calibration-20260927/bin \
  --result-root .local/scratch/startup-calibration-replay --compress-closed-logs

python3 benchmarks/phase_serving/report_startup_calibration.py \
  --result-root .local/results/startup-calibration-20260927/full12-paired --include-dispatch

python3 benchmarks/phase_serving/report_workspace_revalidation.py \
  --campaign startup=.local/results/startup-calibration-20260927/full12-paired \
  --output-prefix .local/results/startup-calibration-20260927/full12-paired/frozen-comparison
```

## 승격 판단

준비 기능의 구현/실행 성공과 기존 champion 대체 여부는 별개의 gate다.
첫 Gemma full12에서 throughput 기하평균 -2.84%, short -10.63%였으므로 자동 보정 경로는
기본값으로 승격하지 않는다. 전체 최종 수치와 남은 작업은 아래 최종 결과에 기록한다.

## 최종 full12 결과

48/48 cells 완료, HTTP 실패·누락 0, 두 variant 모두 동일 request contract.
각 cell 1회이므로 유의성/95% CI/3회 promotion gate를 주장하지 않는다.

| 모델 | 자동/기존 throughput 기하평균 변화 | throughput 개선 workload | 준비 보정 구간 |
|---|---:|---:|---:|
| Gemma AWQ | -2.84% | 2/12 | 17.82–18.31초 |
| Cosmos FP16 | -0.21% | 7/12 | 16.66–16.87초 |

준비 시간은 모델 적재가 끝난 뒤의 보정 구간이다. 전체 서버 시작 시간이나 HTTP generic
warmup까지 포함한 두 방식의 time-to-ready 비교가 아니다.

아래는 자동 보정 / 같은-binary 기존 calibration의 변화율이다. Throughput은 양수,
latency는 음수가 개선이다. TTFT/TPOT/E2E의 mean과 p95를 모두 보존했다.

| Model/workload/repeat | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Startup ms | Peak MiB old/new |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma/balanced/1 | -5.04% | -3.58% | -1.21% | +1.78% | +1.52% | +1.63% | +2.09% | 17823.3 | 9385/9353 |
| gemma/mixed/1 | -0.56% | -4.89% | -5.50% | +2.64% | -0.54% | +0.78% | +0.53% | 17851.8 | 9399/9399 |
| gemma/vision-heavy/1 | -2.29% | -3.22% | -2.50% | +2.29% | +2.17% | +1.40% | +4.44% | 17859.9 | 9405/9403 |
| gemma/multi-image/1 | +0.49% | +4.40% | +0.45% | -0.04% | -0.80% | +1.42% | +2.61% | 17841.9 | 9403/9403 |
| gemma/long-prefill/1 | +1.37% | +16.34% | -2.53% | -5.69% | -5.78% | -1.48% | -5.15% | 18311.0 | 9385/9353 |
| gemma/bimodal/1 | -1.05% | -1.61% | -1.41% | +3.75% | +11.76% | +1.28% | -1.46% | 17863.8 | 9389/9353 |
| gemma/decode-heavy/1 | -5.20% | +1.57% | +5.30% | +1.26% | +1.59% | +1.44% | +1.39% | 17882.8 | 9385/9353 |
| gemma/short/1 | -10.63% | +4.84% | -1.99% | +2.73% | -4.30% | +4.04% | +3.90% | 17874.1 | 9385/9353 |
| gemma/text-heavy/1 | -3.27% | -1.75% | +5.40% | +2.33% | +2.21% | +2.04% | +1.35% | 17904.0 | 9395/9395 |
| gemma/poisson/1 | -6.68% | +10.09% | -10.13% | +0.75% | +3.93% | +1.71% | -1.62% | 17885.1 | 9389/9389 |
| gemma/wave-drain/1 | -0.08% | -3.12% | +0.51% | +4.37% | +1.77% | +0.82% | -0.02% | 17868.7 | 9391/9391 |
| gemma/late-vision/1 | -0.38% | -0.37% | -2.99% | +0.91% | +1.17% | +0.83% | +1.71% | 17884.4 | 9395/9395 |
| cosmos/balanced/1 | -10.11% | -1.56% | -4.22% | +10.57% | +11.49% | +9.76% | +12.04% | 16682.9 | 9733/9553 |
| cosmos/mixed/1 | -0.76% | -5.11% | -2.31% | +1.92% | +1.91% | -0.96% | -0.62% | 16674.7 | 9841/9841 |
| cosmos/vision-heavy/1 | +0.27% | -5.05% | -0.56% | +4.45% | -0.70% | -0.41% | -0.68% | 16681.4 | 9855/9853 |
| cosmos/multi-image/1 | +4.37% | -14.79% | -7.00% | +7.98% | +26.74% | -4.24% | -4.23% | 16680.6 | 9731/9599 |
| cosmos/long-prefill/1 | +7.34% | -4.81% | +5.55% | -9.39% | -11.49% | -7.26% | -7.13% | 16696.2 | 9741/9553 |
| cosmos/bimodal/1 | +3.14% | -1.28% | -6.92% | +1.46% | +17.25% | -2.14% | -4.76% | 16698.3 | 9731/9553 |
| cosmos/decode-heavy/1 | -4.00% | +2.57% | -11.65% | +1.81% | +2.18% | +1.80% | +2.22% | 16693.8 | 9731/9553 |
| cosmos/short/1 | -5.47% | -10.60% | -1.07% | +18.39% | +24.73% | +4.71% | +5.47% | 16690.9 | 9731/9553 |
| cosmos/text-heavy/1 | +0.90% | +2.44% | -5.90% | +0.26% | -5.73% | -0.87% | -0.28% | 16873.7 | 9731/9659 |
| cosmos/poisson/1 | +3.27% | -17.02% | -25.42% | +2.46% | +9.17% | -3.40% | -1.66% | 16691.0 | 9741/9633 |
| cosmos/wave-drain/1 | +0.49% | -11.53% | -8.83% | +4.59% | +17.38% | -3.87% | -5.21% | 16663.6 | 9739/9605 |
| cosmos/late-vision/1 | -0.75% | +3.91% | +4.04% | +0.59% | +0.46% | +0.88% | +0.71% | 16671.5 | 9731/9625 |

### frozen vLLM 대비

| 모델/초기화 | Throughput 기하평균 차이 | Throughput 우세 | E2E mean 우세 | E2E p95 우세 |
|---|---:|---:|---:|---:|
| Gemma 기존 calibration | +35.62% | 12/12 | 10/12 | 10/12 |
| Gemma 자동 calibration | +31.77% | 12/12 | 10/12 | 9/12 |
| Cosmos 기존 calibration | +16.47% | 12/12 | 12/12 | 12/12 |
| Cosmos 자동 calibration | +16.23% | 11/12 | 11/12 | 11/12 |

원시 수치와 모든 지표의 vLLM 비교는
`.local/results/startup-calibration-20260927/full12-paired/frozen-comparison.{json,md,csv}`,
같은-binary A/B는 `startup-comparison.{json,md}`에 있다.
VLM output-contract 차이와 Gemma semantic/exact output gate는 이전과 동일하게 미해결이며,
fixed-output throughput 우세를 production quality 통과로 바꾸어 해석하지 않는다.

### 준비 실패 처리

`expected-budget-rejection`: Gemma에 1ms 예산을 주면
`status=degraded`, `frontier_covered=false`, `budget_exhausted=true`를 저장하고
`Startup coverage incomplete; serving readiness withheld`로 시작을 거부했다.
본 성능 캠페인의 실패가 아니라 별도의 expected-negative test다.

총 소요는 약 743ms였다. 예산은 신규 calibration 요청의 admission을 제한하며,
이미 시작한 작업의 drain과 graph 준비를 강제 중단하는 wall-clock timeout은 아니다.

`eager-smoke`: Cosmos short를 CUDA graph off로 실행해 HTTP 추론 및 coverage를 통과했다.
준비 보정 16.261초, D graph entries=0, D 관측 variant는 모두 eager였다.
검증 종료 후 GPU는 1MiB/0% utilization, 실행 중인 benchmark 컨테이너는 없었다.

## 최종 판단과 미완료 범위

- 구현한 opt-in HTTP startup 경로와 full12 ×2모델 A/B 검증은 완료했다.
- **기존 기본값 유지.** 자동 보정은 관측을 이식 가능하게 만들지만 현재 방식은
  cohort formation과 초기 policy evidence를 바꿔 일부 핵심 workload에서 회귀한다.
- 이 단계에서 production native API 통합, KV/vision/graph 최적 용량 자동 배분,
  모든 fallback 상수 제거, RLS 수렴 기반 readiness까지 완료한 것은 아니다.
- 다음 단계는 같은 calibration cohort/학습량을 유지한 채
  (a) D 비용표만 교체, (b) E/P/D 초기 exact 관측만 추가, (c) 학습량만 축소를 분리한다.
  그 전에는 어느 하나가 이번 회귀의 단독 원인이라고 주장하지 않는다.
- Startup readiness의 physical-key coverage와 policy authority readiness를 별도 상태로
  보고해야 한다. 특히 overlap 방향별 관측·held-out error가 없어도 key coverage는 통과할 수 있다.
- 일반화는 두 모델/한 GPU에서만 확인했다. 다른 GPU에서의 최적성·안정성을 보장하지 않는다.
