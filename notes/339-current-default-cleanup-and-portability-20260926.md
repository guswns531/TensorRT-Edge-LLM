<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 339. 검증된 Current 기본 연결, 불필요한 코드 정리와 모델·GPU 이식성 감사

## 1. 요청과 상태

사용자 요청은 다음 세 가지다.

1. 현재 검증된 최고 수준의 버전을 기본 설정으로 연결한다.
2. 우리가 추가했던 코드 중 사용되지 않는 구현과 중복을 제거한다.
3. 모델·양자화·GPU가 달라지면 문제가 될 수 있는 휴리스틱과 고정 설정을 확인한다.

이번 작업은 새로운 정책이나 batch 설정으로 성능을 다시 최적화하는 작업이 아니다. 기본 경로는
Note338에서 3회 검증한 `9db3ed3`의 계약을 사용한다. 정리한 소스 `d3a27e1`도 full24와 선택 반복12회,
총36회 검증했다. **처리량 수준은 유지됐으나 반복 TTFT 신호가 남아 정리 바이너리의 승격은 보류했다.**
기본 설정과 소스 정리는 완료했지만 latency 무회귀를 모두 통과했다고 주장하지 않는다.
기존 검증 결과를 새 바이너리의 성능 결과로 소급하지 않는다.

| 대상 | 최종 상태 |
|---|---|
| 기본 serving 바이너리 | 기존 검증된 immutable `9db3ed3`에 연결 |
| 기본 engine | Note338에서 검증한 workspace-corrected Gemma/Cosmos engine에 연결 |
| 소스 정리 | 미사용 보조 learner, 전달용 header, serializer 중복, 동일 조건 중복 제거 |
| 정리 후 새 바이너리 | 36회 validation 완료; TTFT 신호로 승격 보류, 기본9db 유지 |
| 모델·GPU 이식성 | 실제 V3 caller/config를 추적한 read-only 감사 완료 |
| Gemma semantic/exact-output gate | 미해결; 처리량 승격과 분리 |

## 2. 기본 설정으로 선택한 이유와 한계

근거는 [Note338](338-throughput-three-repeat-confirmation-and-code-audit-20260926.md)이다.
동일 바이너리의 첫24회와 추가48회를 합쳐 두 모델 × 12 workload × 3회, 총72회를 확인했다.

| 모델 | frozen vLLM 대비 처리량 기하평균 | 중앙값 우세 | 개별 실행 우세 |
|---|---:|---:|---:|
| Gemma AWQ | +35.73% | 12/12 | 36/36 |
| Cosmos FP16 | +16.64% | 12/12 | 35/36 |
| 전체 | +25.82% | 24/24 | 71/72 |

`9db3ed3`은 직전 `40536eb` 3회 결과 대비 전체 처리량 변화가 −0.009%였다. 따라서 최신이
모든 cell과 모든 지표에서 절대 최고라는 결론은 아니다. **Residual E의 보호 완료시간 누락을
수정하면서 기존 최고 수준의 처리량을 유지한 검증 후보**이므로 기본으로 사용한다.

다음 한계는 승격 이후에도 그대로 남는다.

- Cosmos wave-drain의 한 반복은 frozen vLLM보다 −3.28%였다. 마지막 wave의 D cohort 분할과
  추가 GPU service가 관측됐으며, 이 실행을 제거하거나 좋은 실행으로 대체하지 않는다.
- 모든 latency가 우세하지 않다. 전체24개에서 TTFT mean/p95 우세는14/15개, TPOT mean/p95는
  22/19개, E2E mean/p95는22/21개였다. Gemma TTFT p95의 기하평균은 +10.05% 악화였다.
- Cosmos1,513개 요청은 세 반복의 raw output이 모두 같았지만, Gemma632개 중 raw exact는573개,
  through-first-stop exact는587개였다. HTTP·출력 길이·capture 검사는 semantic accuracy 검사가 아니다.
- vLLM은 동일 workload 계약의 frozen reference다. 이번 승격이 fresh paired 비교나 통계적 동등성
  검증을 새로 수행했다는 뜻은 아니다.

## 3. 기본 경로와 재현 계약

```text
.local/current/active -> gemma4
  └─ runtime -> .local/baselines/throughput-9db3ed3-20260926/bin

.local/current/gemma4/
  ├─ runtime -> 검증된 공통 serving binary
  └─ engine  -> workspace-corrected-20260926/gemma

.local/current/cosmos/
  ├─ runtime   -> 검증된 공통 serving binary
  └─ engine-vp -> workspace-corrected-20260926/cosmos
```

`benchmarks/phase_serving/run_lifetime_encoded_admission.py`의 기본 실행은 다음 계약을 사용한다.

- V3 `service-scaled-transition`, 독립 E/P/D execution context/workspace.
- Serving overlap probe on, transition-predictor option on, dispatch telemetry.
- Text chunk128, P token budget1024, D CUDA graph on/cache maximum64, P graph off.
- Generic calibration; 각 cell은 새 프로세스에서 같은 절차를 수행한다. posterior까지 동일하다는 뜻은 아니다.
- Gemma E4/P8/D24, slots24/inflight24/KV192pages; Cosmos E4/P8/D64,
  slots80/inflight64/KV256pages. 모델별 precision/engine/vision capacity는 유지한다.
- 기본 serving binary는 mutable build directory가 아닌 `.local/current/active/runtime`으로 해결한다.
- 기본 engine도 각 model lineage의 `current` pointer에서 해결한다. 이전 flat pointer만 보고 전체
  보호 집합을 판단하지 않는다.

승격 기록은 `.local/registry/current.json`의 `serving_default`/`model_lineages`와 frozen binary manifest를
기준으로 보존한다. 기존 `artifacts` flat-pointer metadata는 역사적 참조 감사용으로 보존하며
현재 serving의 authoritative 설정으로 사용하지 않는다.
`AGENTS.md`에도 이 기본 실행 경로, immutable binary 원칙, 이식성·품질 한계를 명시한다.
일반 inference entrypoint와 upstream `handleRequest()`의 모든 기본 동작을 V3로 바꿨다는 의미는 아니다.

벤치마크의 `IGNORE_EOS=1`은 요청된 고정 output work를 맞추기 위한 계약이다. 일반 production
요청의 종료 정책으로 무조건 적용해야 하는 설정이 아니며, throughput 비교와 semantic 검증을
분리해야 하는 이유이기도 하다.

현재 기본 계약 확인 및 실행:

```bash
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py --full12 --dry-run
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --full12 --repeats 3 --compress-closed-logs \
  --result-root .local/scratch/current-serving-check
```

기본 model은 Gemma/Cosmos 둘 다이고 `--models gemma`로 하나만 선택할 수 있다. 기본 workload는
mixed/vision-heavy/multi-image 세 개이므로 전체 검증에는 `--full12`를 넣는다.
과거 static/shared/tiered 구성은 `--variants`로 명시해야 실행된다. 상세 원인 분석에는
`--telemetry-level full`을 사용하되 dispatch-only 성능과 같은 계측 계약이라고 간주하지 않는다.
기본 scratch 경로에 다른 source/command 결과가 있으면 재개를 거부하므로 새 캠페인에는 별도 경로를 준다.

## 4. 제거한 코드와 유지한 경계

### 4.1 제거·통합 대상

| 대상 | 변경 | 동작 경계 |
|---|---|---|
| Scheduling forwarding header6개 | canonical `runtime/phase/...` include로 이관 후 제거 | 내부 caller를 이관한다. 외부에서 옛 include 경로를 썼다면 경로 변경이 필요하다 |
| Decode queue-residence 보조 RLS | 학습 update, 상태, 예측 API와 전용 테스트 제거 | 예측값의 production consumer가 없던 보조 learner다. 주력 contextual Scalar RLS가 아니다 |
| `PhaseOptimizationContext.prefillTokens` | 읽지 않는 필드와 대입 제거 | 사용되지 않는 입력만 제거 |
| Protected-completion JSON serializer | 중복 lambda를 공통 helper로 통합 | Full telemetry의 기존 field/value 의미 유지 |
| `mediaFull` / `multiMediaReady` | 동일식과 중복 조건 통합 | E batch readiness 판단의 의미 유지 |

제거한 forwarding header는 `phaseActionPlan.h`, `phaseDeadline.h`, `phaseGlobalCostModel.h`,
`phaseGlobalScheduler.h`, `phaseOwnershipHorizon.h`, `phaseReadySnapshot.h`의
`cpp/runtime/scheduling/` 경로다. 실제 canonical 구현은 `cpp/runtime/phase/` 아래에 남는다.
Header6개의 기존 분량은126줄이다.

소스 커밋 `d3a27e1`: 20파일, 추가156줄/삭제616줄, **순감소460줄**이다. 이 중 C++ runtime/examples는
추가29줄/삭제410줄로 **순감소381줄**이다. 테스트는 제거된 learner 전용 검증을 정리하고 기본 설정/
provenance 검증을 추가해 순감소100줄이다. 본 보고서의 설명 분량은 코드 감소량에 넣지 않는다.

보조 queue learner는 D 완료 후 queue residence를 갱신했지만 정책 선택에서 값을 읽지 않았다.
제거 후에도 `PhaseTransitionPredictor`의 stateless burst/overlap-token fallback controller는 남긴다.
이 controller는 지원하는 legacy/opt-in 경로용이며, 이름이 비슷한 `PhaseContextualPdModel`의
주력 P+D/E+P/E+D Scalar RLS와 구분한다.

이 보조 learner의 설정6개(minimum observations, confidence beta, initial covariance/residual variance,
forgetting factor, latency clip)도 제거됐다. **활성 정책의 휴리스틱6개를 없앤 것이 아니라** 정책에서
사용하지 않는 학습기의 설정 표면을 줄인 것이다.

외부에서 삭제한 보조 learner API/config field를 직접 사용했다면 소스 수정이 필요하다.
해당 내부 클래스의 layout도 달라지므로 모든 custom runtime caller를 재빌드해야 한다. Binary/API 호환성을
보장하는 정리는 아니며, 보장하려는 것은 유지된 실행 정책식과 correctness constraint다.

CPU update 제거가 GPU kernel이나 policy formula를 바꾸지 않더라도 host submission timing은 달라질 수
있다. 따라서 logical no-op이라는 이유로 성능 검증을 생략하거나 개선을 선결론으로 쓰지 않는다.

### 4.2 의도적으로 남긴 코드

- V0/V1/V2 동일 runtime ablation과 V3의 exact/covering/cold fallback, Scalar RLS, uncertainty.
- `previewMechanismPlan()`이 재사용하는 Global Disabled + forced phase의 공통 batching mechanism.
- Serving probe on/off, full telemetry, action fidelity·row ordering·출력 불일치 진단.
- KV/vision ownership, dependency, context single-inflight, CUDA completion, shape/memory 검증.
- 아직 실제 opt-in caller가 있는 encoder preparation 연구 경로와 버전 비교 호환 경로.
- 실제로 실행되는 `.local/results/v0101-forward-port/replay-tools/`의 HTTP harness/gateway/client.
  이것은 위치가 오래된 결과 디렉터리라는 이유만으로 지울 수 있는 로그가 아니다.
- Hardcoded D cost fallback. 아래 감사에서 가장 큰 이식성 위험으로 분류하지만, 제거하면 live
  batch/service 정책이 바뀌므로 이번 중복·미사용 정리에 섞지 않는다.

소스 삭제와 model/engine/result artifact 삭제는 별개의 작업이다. 이번 코드 정리를 이유로 과거
실험 근거나 source worktree를 삭제하지 않는다.

## 5. 감사 방법: 숫자가 존재하는가와 실제 정책이 사용하는가를 구분

감사는 Note338의 `additional48/manifest.json`에 보존된 effective environment와 current source의
caller를 함께 확인했다. 기본 off나 과거 이름만으로 dead code라고 판정하지 않는다.

네 종류로 나누면 오해가 줄어든다.

| 종류 | 예 | 관리 방향 |
|---|---|---|
| Correctness invariant | dependency, event completion 전 reclaim 금지, context single-inflight | 유지; 성능 knob처럼 제거하지 않음 |
| Engine capability / resource contract | 지원 batch/shape, KV pages, TRT workspace, graph storage | engine·장비 identity에 묶어 검증 |
| Active performance/estimator assumption | D cost prior, cold cost, chunk128, E wait cap, RLS confidence | 실측·coverage·이식성 ablation 대상 |
| Bypassed legacy / optional policy | legacy burst grace, arbiter max defer, 기존 overlap-token gate | current V3와 구분; 지원 경로 retirement 결정 전에는 보존 |

아래 위치는 감사 시점의 source line 기준이다. 이후 정리로 line이 이동하면 함수/필드 이름을 같이
검색한다. 값의 존재만으로 그 값이 지금 throughput 손실을 만들었다고 주장하지 않는다.

## 6. Active V3의 이식성 위험 순위

| 위험도 | 값 / 구현 | 실제 소비와 예상 위험 | 근거 위치 |
|---|---|---|---|
| 높음 | D static cost36개, 약6.238–9.861ms, context1024/2048, BS1–64 | Dynamic D formation과 cold service reference가 사용한다. 새 GPU·model·quantization의 cost curve가 달라지면 cohort/normalized age가 잘못될 수 있다 | `examples/llm/phaseSchedulerOptions.inc:179`; `phaseQueueScheduler.cpp:411`, `:4001`, `:4064` |
| 높음 | D observation correction: isolated±25%, contended upper16×, 최소8samples/window32 | Frozen 설정은 measured-decode-batching off, dynamic/component-observation on이다. 실측이 충분해도 이 경로에서는 static prior의±25% 밖 isolated cost로 완전히 교체되지 않는다 | `phaseQueueScheduler.h:483–493`; `.cpp:4147–4162`; options`:165–170` |
| 높음 | Cold P0.02ms/token, D2ms, E50ms, external P50ms, uncertainty2ms, E safety margin5ms | Coverage가 없을 때 live fallback이다. 명시적 SLO를 없애도 service normalization·early choice·probe를 물리적 prior가 바꾼다 | `phaseQueueScheduler.h:409`; `phaseThreeCoordinator.h:119`, `:238–240`; `phaseGlobalCostModel.h:176`; queue`.cpp:370`, `:424`; three`.cpp:3091`, `:3157`, `:3420` |
| 높음 | Generic calibration49 Gemma /239 Cosmos; graph warmup BS1,2,4,8,12,…64 | Workload label별 tuning은 아니지만 model별 초기 knowledge budget은 다르다.49/239개가 새로운 모델에서도 충분하다는 보장은 없다 | `run_lifetime_encoded_admission.py`의 `model_config`, `WARMUP_DECODE_BATCHES`, `command_for` |
| 높음: 자원 | E4/P8/D24 또는64, slots24/80, inflight24/64, KV192/256pages, E input cap1120/8192 | 검증된 deployment capacity이지 자동 최적점이 아니다. Layer/head/KV dtype/vision resolution에 따라 같은 page 수와 batch의 bytes가 달라진다 | runner `model_config`와 env construction; options`:76–89` |
| 중간~높음 | Lifetime vision budget = 활성화 시점 free bytes − 관측된 maximum prepared-storage bytes | Count credit보다 ownership에 가깝지만 한 번의 memory snapshot/reserve다. 뒤늦은 library/graph allocation, 더 큰 미관측 shape, 다른 프로세스까지 보장하지 않는다 | `phaseThreeCoordinator.cpp:1118–1145`, `:4829` |
| 중간 | Text chunk128, P total-token budget1024, adaptive chunking off | 현재 실행 granularity를 직접 제한한다. Faster/slower kernel과 host launch cost가 바뀌면 amortization/interference trade-off도 바뀐다 | runner `command_for`; options`:86`, `:107`; `phaseServingExecutionOptions.cpp:54–98` |
| 중간 | E formation maximum wait25ms | Global에서도 사용하지만 무조건25ms sleep은 아니다. Arrival EWMA와 E+P cost gain을 비교한 WAIT의 상한이다. Absolute cap은 다른 service scale에서 재검토 필요 | runner `TRT_EDGELLM_VISION_ENCODER_BATCH_WAIT_US`; three`.cpp:5124–5227` |
| 중간 | E arrival EWMA α.2, E token bucket1024, D context bucket512 | Update speed와 evidence granularity의 가정이다. 다른 shape에서 너무 거칠거나 희소해질 수 있다 | three`.cpp:1162`, `:1269`, `:3087`; queue`.h:488` |
| 중간 | Scalar RLS14features, readiness4, β.5, covariance4, variance.04, forgetting1, reward clip1 | Workload 규칙은 아니지만 sample/confidence 가정이다. Forgetting1의 누적 posterior가 장비 drift에 자동으로 적절히 대응한다고 보장할 수 없다 | `phaseContextualPdModel.h:66`, `:123–132` |
| 중간 | RLS log time scale1000µs/4, context bucket/4 clip4, work quantum128 | 연속 feature도 normalization/clipping 선택을 포함한다. 아주 큰 context나 극단적인 kernel cost가 feature에서 서로 구분되지 않을 수 있다 | `phaseContextualPdModel.cpp:99–131` |
| 중간 | Serving probes on, slack multiplier3, interval0, calibration maximum16keys, exact overlap4samples/+2% gain floor | Frozen interval0은 decision-count 간격을 제거한다. No-SLO slack은 무한대이므로 slack 검사만으로 보수성이 생기는 것은 아니다. 다른 feasibility/evidence 조건은 유지된다 | runner env; queue`.h:415–420`; three`.cpp:3393–3433`; `phaseGlobalCostModel.h:174–177` |
| 중간 | D graph maximum64 /P graphs0와 대표 batch warmup | Cache memory, capture/launch benefit은 GPU·model에 따라 다르다.64는 policy/resource budget이지 correctness 상수는 아니다 | runner graph options; `phaseServingExecutionOptions.cpp:117–164` |
| 중간 | Vision 전 text-prefix prefill 최소128tokens | Feature는 켜져 있지만 짧은 prefix는 별도 P launch를 하지 않는다. Prefix가 짧은 모델의 E+prefix-P 기회를 제한할 수 있다 | `phaseThreeCoordinator.h:194`; `.cpp:1190–1205` |
| 낮음~중간 | Request adapter workers4, pending cap1024, sampling cold200µs/window32 | CPU 수/tokenizer/preprocessing/IPC가 달라지면 producer readiness와 batching이 달라진다. Sampling prior와 실제 worker capacity를 구분한다 | runner env; smoke`.cpp:1900`; `independentPhaseAsyncServer.h:235–236` |
| 설계 bound | 최대11 candidates, bounded transition horizon, realized diagnostic4dispatches | CPU 비용을 제한하는 설계다. 더 큰 execution frontier에서 최적 후보를 누락하는지는 별도 검증 필요. Diagnostic horizon과 policy horizon은 다른 값이다 | `phaseGlobalScheduler.h:67`; three`.h:147–149` |

### 6.1 가장 먼저 확인할 D prior 문제

현재 계약에서 `enableDynamicDecodeBatching=true`, `enableDecodeComponentObservation=true`지만
`enableMeasuredDecodeBatching`은 켜지지 않는다. 따라서 다음 두 경로를 혼동하면 안 된다.

```text
Global action prediction:
  process-local exact/covering timing + Scalar RLS

일부 D cohort construction:
  static D table
    + 관측된 component cost를 제한된 범위로 보정
    → remaining-row drain DP
    → batch 선택
```

Isolated correction은 `prior × .75`와 `prior × 1.25` 사이로 clamp된다.
예를 들어 새 모델의 실제 D cost가 prior의 절반이라면, 이 component correction만으로 실제 값에
도달할 수 없다. 이 예는 코드 수식의 의미이며 새 GPU에서 실제로 그 회귀가 발생했다는 측정은 아니다.
반대로 sparse coverage를 이유로 D batch를 줄이지 않는 fallback은 유지할 가치가 있다.

Static table을 이번 cleanup에서 그냥 지우지 않은 이유는 실제 batch 선택이 바뀌기 때문이다.
다음 정책 변경에서 runtime-cost coverage와 static/clamp 사용 빈도를 계측한 뒤 분리 A/B해야 한다.

### 6.2 E formation25ms의 정확한 의미

Global active에서 partial E batch가 준비되고, capacity/geometry/input limit 때문에 이미 막힌 상태가
아니며, 다음 arrival의 EWMA가 있을 때 다음을 비교한다.

```text
지금 dispatch:
  current E+P critical cost + singleton E+P critical cost

잠시 WAIT:
  predicted interarrival remainder + larger E+P critical cost
```

대기 예측은 `25ms − oldest wait`로 제한된다. Current/singleton/future E 비용이 확보되고,
대기 후 E+P horizon이 유리하며 보호 slack을 침범하지 않을 때만 WAIT한다.
따라서25ms를 없앤다고 모든 vision 요청의 TTFT가25ms 줄어든다는 해석은 틀리다.
다만 이 상한 자체는 device service scale에서 유도된 값이 아니므로 모델·GPU 변경 시 재검토 대상이다.

### 6.3 Batch size는 RLS에 포함된다

`phaseContextualPairFeatures()`는 실제 primary/secondary batch size를 `log1p(batch) /
log1p(capacity)`로 정규화해 feature에 넣는다. Capacity는 실제 P/D 설정에서 전달된다.
Time ratio, measured cost, work size, context bucket, slack ratio, residual anchor도 포함한다.

즉 `E1/P8`과 `E4/P8`이 같은 phase label이라는 이유로 완전히 같은 입력을 받지는 않는다.
그러나14차원 projection이 모든 shape 차이를 완벽히 표현하거나, 같은 batch fill이 다른 모델에서도
같은 interference를 뜻한다는 보장은 없다. Model/quantization 이름을 넣는 대신 새 engine에서 측정한
실행 비용과 새로운 process-local posterior로 판단한다.

## 7. 선언은 있지만 현재 V3의 직접 정책이 아닌 값

### 7.1 `maxOverlapPrefillTokens=128`

`phaseServingExecutionOptions.cpp:92`에서 설정하고 legacy/default 또는 opt-in cost-aware overlap
경로에서 사용한다. 현재 GlobalV3는 다음 순서로 동작한다.

1. `phaseQueueScheduler.cpp:3593–3599`: Global active이고 external drain이 없으면
   `legacyQueueDecision()`을 호출하지 않는다.
2. `previewMechanismPlan():1855–1865`: Global Disabled clone에 explicit phase kind를 설치하고
   metrics policy를 끈 채 공통 batch를 만든다.
3. Global overlap candidate 생성(`:2444`)에는128-token 조건이 없다.
4. Frozen 계약에서 `enableCostAwareOverlapAdmission`은 off다.

따라서 **기존 overlap-token gate128과 현재 실제 text chunk128을 같은 제한으로 설명하면 안 된다.**
전자는 legacy/opt-in portability 항목이고 후자는 현재 실행 granularity다.

### 7.2 Decode burst grace20ms와 stateless transition controller

`PhaseTransitionPredictor`에 burst grace20ms, burst1–16, D queue≤2일 때 burst2,
overlap slowdown을 decode step의20%로 추정하는 식이 남는다. 정리 후 구현은
`phaseTransitionPredictor.cpp:36–113`에 있다.

이 식의 호출자는 기본/metrics decision과 optional external drain 경로이며, 위 GlobalV3 구성에서는
직접 phase authority가 아니다. `enableTransitionPredictor=1`이라는 env만으로 이 controller가
Global 선택을 결정한다고 해석하면 안 된다. 이번에 제거한 queue-residence learner는 이 controller에도
출력을 공급하지 않았다. 지원하는 fallback API를 유지하기 위해 stateless controller는 남겼다.

### 7.3 Encoder arbiter/max defer

Legacy arbiter에는 initial50ms, safety5ms, text guard250ms, pressure.9, maximum defer500ms가 있다.
`startNextEncoder()`는 Global이 E batch를 승인했으면 arbiter를 통과하지 않고 approved decision을
사용한다(`phaseThreeCoordinator.cpp:4121–4126`). 따라서 max defer500ms를 현재 GlobalV3의
vision starvation 정책이라고 설명하면 틀리다.

다만 같은 config의 **initial E50ms와 margin5ms는 Global cold prediction/probe에도 재사용**된다.
Arbiter가 bypass된다는 이유로 이 두 물리적 시간 prior까지 inactive로 분류하면 안 된다.

### 7.4 명시적 SLO와 recovery

V3에서 외부 vision TTFT가 명시되지 않으면 built-in2.5s를0으로 바꾼다
(`phaseThreeCoordinator.cpp:921–924`). Request TPOT/TTFT와 명시적 global TPOT가 없으면
P/D의 absolute slack은 infinity가 된다(`phaseQueueScheduler.cpp:593–666`).
따라서 built-in TPOT20ms/vision TTFT2.5s가 현재 no-SLO 계약의 숨은 절대 deadline은 아니다.

`serviceRecoveryAgeQuanta=1`, `serviceRecoveryBandQuanta=1`은 별도 enable flag가 필요하다.
Frozen env에는 service recovery/normalized authority enable이 없으며 composition은 off로 둔다
(`llm_phase_context_smoke.cpp:1773–1789`). 선언된 값만으로 active라고 세지 않는다.
**No explicit SLO와 no heuristic은 다른 주장**이다. Active service cost prior와 capacity 선택은 남는다.

### 7.5 Memory/admission의 optional 정책

Adaptive/stepwise admission, phase memory broker, external drain은 frozen 실행에서 켜지지 않는다.
KV reservation은 기본 full output reservation이다(`independentPhaseAsyncServer.h:219`).
Headroom128tokens/growth8의 config가 있어도 headroom 모드를 선택하지 않으면 그 정책은 사용하지 않는다.

현재의 lifetime vision admission은 실제 retained/reserved byte를 사용한다. KV pool 전체가 vision과
자유롭게 재분할되는 무제한 동적 allocator라는 뜻은 아니다. Engine/KV/graph/workspace의 고정 예산과
vision payload lifetime accounting을 구분해야 한다.

## 8. 지금 제거하면 안 되는 correctness/capability 경계

다음은 경험적 성능 heuristic과 다르다.

- Vision 결과를 소비하는 P는 유효한 dependency/event handoff와 payload ownership을 가져야 한다.
- KV slot/page와 vision storage는 모든 GPU consumer가 끝나기 전에 재사용·회수하지 않는다.
- 같은 TensorRT context에 두 enqueue를 동시에 outstanding으로 만들지 않는다.
- Global이 승인한 outstanding phase set을 초과하는 실제 실행은 거부한다.
- Batch/chunk/encoder input shape는 해당 engine optimization profile과 binding 범위 안이어야 한다.
- Admission/growth reservation은 실제 가용 page/byte 및 보장된 성장량을 만족해야 한다.
- E/P/D/Copy stream 간 데이터 의존성은 event나 같은 stream 순서로 보장한다.
- Multimodal chunking은 모델의 position/cache/embedding placement 계약이 검증된 경로에서만 허용한다.

Portability 개선은 이 검사를 약하게 만드는 것이 아니라 **측정된 physical cost와 capability 정보를
정책에 전달하는 방법을 개선하는 것**이어야 한다.

## 9. 후속 우선순위

### P0. Cleanup 승격을 위한 동시점 paired control

Unit/build/full24와6개 선택 workload의 추가 반복은 완료했다. 기존9db3ed3을 기본으로 유지하며,
TTFT 신호가 남은 workload에서9db/d3를 같은 시점에 교차 반복해 source 영향과 실행 변동을 분리한다.
HTTP/count integrity와 output exact, throughput, TTFT/TPOT/E2E mean/p95를 따로 비교한다.
정리로 CPU timing이 달라질 수 있으므로 logical equivalence만으로 benchmark 통과를 대신하지 않는다.

### P1. D prior의 portability부터 계측

Static/cold/runtime cost 사용 비율, correction clamp hit, coverage miss를 모은다.
Runtime D cost가 충분히 덮는 범위에서는 prior 대체를 검토하되, sparse coverage에서 더 작은 D만
선택해 fragmentation을 만드는 문제를 방지한다. E25ms를 먼저 무조건 제거하는 것보다 우선순위가 높다.

### P2. Startup capability와 learning identity를 명시

Model checkpoint/quantization, engine/plugin hash, GPU architecture, graph policy와 execution capacity를
하나의 runtime identity로 검증한다. 다른 engine/GPU에서 옛 posterior를 그대로 재사용하지 않는다.
Capacity/resource 선택과 scheduling policy를 별도 설정층으로 유지한다.

### P3. Cold physical prior와 calibration 종료 조건 개선

작은 isolated calibration으로 실제 cost scale을 확보한 뒤 uncertainty/coverage로 authority를 평가한다.
49/239requests는 역사적 재현 계약으로 남기되, 다른 모델에서도 고정 개수만 채우면 충분하다고 가정하지
않는다. E/P/D별 부족한 evidence와 safe probe 비용, time-to-benefit를 함께 보고한다.

### P4. E formation과 chunk를 각각 분리 A/B

같은 두 모델, 같은 engine frontier, 같은 workload와 memory 조건에서 E wait cap의 service-scaled
대안을 검증한다. Fixed chunk128 변경은 다른 ablation으로 분리한다. MaxOverlap token legacy gate를
바꾸는 것과 current fixed chunk를 바꾸는 것을 혼동하지 않는다.

### P5. 더 큰 frontier와 다른 GPU/model에서 같은 알고리즘 검증

더 큰 graph cache나 candidate 수는 무조건 개선이 아니라 memory/CPU cost와 recall의 trade-off다.
새 장비에서 별도 workload별 정책을 만들지 않고도 capacity·measured cost·uncertainty로 선택이 달라지는지
검증한다. 이번 두 모델/RTX3080 결과만으로 전 장비에서 vLLM을 이긴다고 주장하지 않는다.

## 10. Cleanup 검증 결과 기록 위치

**Full24와 TTFT 의심 구간의 선택 반복 검증 모두 완료.** 정리본의 latency 무회귀 승격은 보류한다.

### 10.1 완료된 build/CPU 검증

- Source: `d3a27e1`, 이전 source `0de93de`. 빌드 당시 source diff와 동일한 C++를 이후 서명 커밋했다.
- Frozen candidate: `.local/baselines/serving-cleanup-d3a27e1-20260926/bin`.
- `llm_phase_context_smoke`, `llm_inference`, `unitTestRuntime` 빌드 성공.
- Runtime suite: 702개 중700통과, 기존 optional2skip, 실패0.
- Python: runner38, report merge14, output audit19, 총71통과.
- Scoped pre-commit와 `git diff --check` 통과. 저장소 전체 architecture boundary gate의 기존 drift가
  해결됐다는 주장은 아니다.
- 두 stateless controller 함수의 본문은 이전 source와 SHA256이 동일하다.
  `6be3fc93c4277a921bba9a6783a6e8d452549afa562fa75b8fcd2fda79d817d9`.
- 새 runner의 default dry-run24개는 기존72회 effective environment와 전부 일치한다.
  Engine/vision/sidecar/config/calibration/trace/plugin/container/replay-tools도 동일하고 binary/runner만 변경됐다.

Smoke binary SHA256: `a92c6cdf041b071130cb8b8ac593aab0e97b4f336437eda7706997b9b45bf375`.
Production binary SHA256: `a2a8cbdea4e1a541fc2cbcfe8ec0ba3c7f393b661d079c0820fc45e2e1b173ee`.
Plugin SHA256: `ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2` (이전과 동일).

Full24는 `.local/results/current-default-cleanup-20260926/full24`에 기록한다.
기존3회와 새1회는 서로 다른 바이너리이므로 하나의 repeat pool로 합치지 않는다.
vLLM은 동일 계약의 frozen 결과를 재사용한다. Gemma TTFT 등 반복 범위 밖 latency가 나타나는 cell은
추가 검증 대상으로 표시하며 처리량만으로 무회귀라고 판정하지 않는다.

### 10.2 첫 full24 결과: 정리 후1회와 기존3회 비교

24/24완료, 실패0, 누락0. 아래는24개 workload의 비율에 대한 기하평균이다.
Throughput은 높을수록, latency는 낮을수록 좋다. 이전3회는 평균 지표의 경우 run mean 평균,
throughput/p95는 run 중앙값이며 새 full24는 각각1회다. 통계적 동등성 검정 결과가 아니다.

| 지표 | 정리 전9db3ed3 대비 | frozen vLLM 대비 |
|---|---:|---:|
| tok/s | +0.205% | +26.081% |
| TTFT mean | +1.986% | −19.149% |
| TTFT p95 | −0.986% | −8.786% |
| TPOT mean | −1.565% | −22.303% |
| TPOT p95 | −1.746% | −20.614% |
| E2E mean | −0.336% | −20.733% |
| E2E p95 | −0.654% | −22.298% |

- vLLM 처리량 우세24/24. 이전9db3ed3 대비 처리량 우세12/24, 3% 초과 하락0.
- Gemma 처리량: 이전 대비+0.226%, vLLM 대비+36.042%.
- Cosmos 처리량: 이전 대비+0.184%, vLLM 대비+16.850%.
- Peak VRAM: Gemma9,385–9,407MiB, Cosmos9,739–9,853MiB. Cleanup은 GPU memory 최적화가 아니다.
- Fresh24 HTTP/count/capture integrity 통과. 기존72회와 합친96회에서도 issue0/first-token EOS0.
- Across4회·두 바이너리 출력: Cosmos1,513/1,513 raw/stop exact,
  Gemma raw562/632, through-stop577/632. 이전3회에서 같던 요청 중 raw11개/stop10개가 추가로 달랐다.
  이 차이를 cleanup correctness failure로 단정할 수 없지만, Gemma 품질 gate가 해결됐다고도 할 수 없다.

처리량 유지와 달리 아래6개 workload의 TTFT 중 최소 한 지표는 이전 aggregate보다3% 이상 높고,
기존3회 maximum도 벗어났다. 이들은각각2회 추가해 총3회로 확인했다.

| 모델 | Workload | 첫 실행에서 표시된 지표 |
|---|---|---|
| Gemma | vision-heavy | TTFT mean+15.60%, p95+28.13% |
| Gemma | text-heavy | TTFT mean+7.52%, p95+3.53% |
| Gemma | long-prefill | TTFT mean+3.97% |
| Cosmos | mixed | TTFT mean+7.77% |
| Cosmos | short | TTFT p95+3.50% |
| Cosmos | wave-drain | TTFT mean+8.27% |

TPOT/E2E에는 같은 조건의 range-outlier가 없었다. 이6개는 최초 screen으로 보존하며, 재실행에서
좋은 값이 나와도 최초 값을 버리지 않는다. 전체24개를 새로3회씩 실행했다는 뜻은 아니다.

결과 파일은 `.local/results/current-default-cleanup-20260926/` 아래의
`comparison-report.{json,md,csv,manifest.json}`, `cleanup-vs-previous3x.json`,
`combined-output-audit.{json,md}`다. 첫 full24 보고서는 후속 반복으로 덮어쓰지 않는다.

### 10.3 Gemma 선택 반복: TTFT 신호 재확인

`gemma-ttft-confirmation6`의 추가2회와 최초 full24의 해당3개를 합쳤다. 모든 실행을 포함하며
좋은 반복만 선택하지 않았다. 아래 변화는 정리 전9db3ed3의3회 aggregate 대비다.

| Workload | tok/s | TTFT mean | TTFT p95 |
|---|---:|---:|---:|
| vision-heavy | +0.40% | +6.52% | +15.37% |
| text-heavy | −0.35% | +6.19% | +3.53% |
| long-prefill | −1.16% | +0.69% | −0.38% |

Vision-heavy/text-heavy의 신호는 남았고 long-prefill의 최초 신호는 재현되지 않았다.
Vision-heavy 처리량은573.56/552.13/578.76tok/s로 중앙값은 frozen vLLM보다 높지만,
552.13은 frozen559.83보다−1.37%다. 중앙값 우세를 개별 실행 전승으로 표현하지 않는다.
이3개 workload의 TPOT/E2E에서는3% 회귀가 없었다.

**기본 바이너리는9db3ed3 유지, 정리본d3a27e1 승격은 보류한다.** 소스 정리와 기본 계약 연결은
완료됐지만 latency 무회귀 검증이 모두 통과한 것은 아니다. 정책식/제약은 같더라도 제거된 CPU 작업,
copy layout, readiness timing이나 새 프로세스 calibration의 변동이 trajectory를 바꿀 수 있다.
현재 데이터는 동일 시점 interleaved paired control이 아니므로 cleanup이 원인이라고 단정하지 않는다.
신호를 해소하려면 다음 검증에서 해당두 workload의9db/d3를 교차 반복해야 하며,
성능 타이밍을 맞추기 위한 쓸모없는 learner/busy-work를 임의로 다시 넣지는 않는다.

### 10.4 Cosmos 선택 반복과 최종 비교

`cosmos-ttft-confirmation6`도 최초 실행을 포함해 해당3개를각각3회로 확인했다.

| Workload | 3회 tok/s | tok/s 변화 | TTFT mean 변화 | TTFT p95 변화 |
|---|---|---:|---:|---:|
| mixed | 1137.64 / 1135.54 / 1136.08 | −0.28% | +8.87% | +1.27% |
| short | 2368.87 / 2426.15 / 2414.99 | +0.28% | +6.26% | −0.24% |
| wave-drain | 97.90 / 97.65 / 97.57 | −0.37% | +3.26% | +3.15% |

Mixed TTFT mean은863.90/874.03/880.28ms로 기존3회 maximum821.91ms를 모두 초과했다.
Short p95의 최초 신호는 재현되지 않았으며 short mean은 기존 run 범위 안이다.
Wave-drain mean의 변화도 기록하되,3회 중 한 번이 큰 초기 신호였다는 점을 함께 본다.
승격 보류는 Gemma뿐 아니라 Cosmos mixed의 반복 TTFT 신호에도 근거한다.

최종 집계는 **정리본36회(6개 cell×3회 +18개 cell×1회), 기존72회(24개 cell×3회)**다.
반복한6개를 workload 가중치3배로 세지 않고 각 cell의 aggregate를 만든 뒤24개 비율을 기하평균한다.
선택 반복은 처음 관측된 TTFT 편차에 따라 정해졌으므로 전 workload의 균일3회 검증이나 독립적인
통계적 검정으로 해석하지 않는다.

| 지표 | 정리 전9db3ed3 대비 | frozen vLLM 대비 |
|---|---:|---:|
| tok/s | +0.182% | +26.052% |
| TTFT mean | +1.370% | −19.638% |
| TTFT p95 | −1.564% | −9.318% |
| TPOT mean | −1.250% | −22.054% |
| TPOT p95 | −2.430% | −21.166% |
| E2E mean | −0.303% | −20.707% |
| E2E p95 | −0.533% | −22.203% |

- 처리량 vLLM 집계 우세24/24, 개별 실행35/36. 기존9db3ed3 대비 우세12/24.
- Gemma 처리량: 이전 대비+0.052%, vLLM 대비+35.805%.
- Cosmos 처리량: 이전 대비+0.313%, vLLM 대비+17.000%.
- Peak: Gemma9,385–9,407MiB, Cosmos9,739–9,855MiB. GPU memory 감소를 주장하지 않는다.
- 새36회/기존72회 합계108회 HTTP/count/capture 오류0, first-token EOS0.
- 모든 available4~6회를 비교한 Cosmos 출력은1,513/1,513 raw/stop exact.
  Gemma는 raw550/632, through-stop567/632. 기존3회에서 같던 요청 중 추가로23/20개가 달랐다.
  반복 수가 늘면 all-repeat exact 조건이 더 엄격해지므로 이 수만으로 바이너리 변경의 인과성을 주장하지 않는다.

최종 보고서는 같은 result root의 `final-comparison-report.{json,md,csv,manifest.json}`,
`final-cleanup-vs-previous3x.json`, `final-output-audit.{json,md}`다.7지표 전체24개 표, per-run 값,
source-qualified repeat 식별자와 artifact hash를 포함한다.

### 10.5 최종 handoff

- 기본 실행: 검증된9db3ed3 binary + workspace-corrected engines + V3 independent/probes-on/dispatch.
- 소스: `d3a27e1` 정리 완료; 정리 바이너리는 validation artifact로 보존하고 기본 pointer는 바꾸지 않았다.
- 미완료 gate: 정리본 latency 무회귀 승격, Gemma semantic/exact 검증, 새 GPU/model 이식성 실측.
- 모델/엔진/결과/source worktree 삭제0. 제거한 소스는Git parent에서 복구 가능하다.
- GPU postflight:1MiB, utilization0%,58°C; 실행 중Docker container0.
- Disk available 약1.8GiB. 새 engine을 만들거나 기존 결과를 삭제하지 않았다.

## 결론

기본으로 삼은 것은 **정합성 수정까지 포함하면서3회 검증된 최고 수준의 처리량 후보**다.
이번 정리는 주력 RLS와 실행·소유권 mechanism을 유지하며 읽지 않는 보조 학습과 중복을 제거한다.

이식성 측면에서 가장 시급한 것은 encoder25ms 하나보다 **모델별로 다시 측정되지 않은 static D prior와
그 prior에 제한되는 online correction**이다. V3는 명시적 SLO 없이 동작하지만, physical prior,
calibration coverage, engine/memory capacity, fixed chunk와 confidence 가정까지 자동화된 것은 아니다.
Active와 bypassed 상수를 구분한 뒤 하나씩 검증하는 것이 구조를 단순하게 유지하는 방향이다.
