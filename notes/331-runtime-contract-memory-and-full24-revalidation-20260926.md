<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 331. Runtime contract·메모리 lifetime 수정과 dual-model Full24 재검증

작성일: 2026-09-26. 활성 브랜치: `codex/v0101-phase-forward-port`.

상태: **구현 및 단위 검증 기록. 최종 GPU 성능·승격 판정은 pending.**

이 문서는 [330 실행 계획](330-runtime-contract-revalidation-plan-20260926.md)의 후속이다.
현재 controller를 새로운 학습기로 교체하는 작업이 아니라, 기존 결과를 해석하는 데 필요한
관측 label, configuration, CUDA graph lifetime, benchmark contract를 먼저 바로잡는 작업이다.
아래에서 `implemented`, `tested`, `pending`을 구분한다. Full24는 두 모델 각각 Full12이며,
한 설정에서 Full24 ×3은 72개 model/workload/repeat cell을 뜻한다.

## 1. 이번 작업의 범위와 완료 상태

| 항목 | 구현 상태 | 이 문서 작성 시 검증 상태 |
|---|---|---|
| Decode queue 관측 feature/label 일치 | 완료 | 단위 검증 완료 |
| Smoke/production 실행 옵션 resolver 공유 | 완료 | 단위 검증 완료; GPU entrypoint 비교 판정 pending |
| Production decode graph 초기화 연결 | 완료 | 모델당3-request graph on/off smoke 완료; Full-suite quality 판정 pending |
| Shared workspace 교체 시 graph invalidation | 완료 | graph generation/binding 단위 검증 완료; 실제 재현·재발 gate pending |
| Packed-prefill 1-token tail의 P/D 분류 수정 | 완료 (`5075ae6`) | 동일 binary/plugin 교체 전후 3-request GPU fixture에서 첫 EOS 오류 제거; 확장 gate pending |
| Shared E/P activation lease와 vision slab retention 분리 | 완료 | helper 단위 검증 완료; 실제 two-slab lifecycle pending |
| Predictor/workspace 동일 binary A/B runner | 완료 | CPU contract test 완료; GPU screening 진행 대상 |
| Failed/missing campaign fail-closed | 완료 | CPU 검증 완료 |
| Cosmos frozen vLLM raw provenance·평균 복구 | 완료 | 12 trace / 35 successful runs 검증 완료 |
| 두 모델 Full12 ×3 | 미완료 | pending; 전체 승리/무회귀 주장 금지 |
| 새로운 physical handoff learner | **구현하지 않음** | 이번 범위 밖 |

주요 구현 커밋:

- `3ac060b`: observation/execution contract, slab budget interface, runner·reporter 정리.
- `70743a2`: 실제 production initialization smoke runner.
- `6e2c518`: 누락/실패 campaign의 nonzero exit와 명시적 completion 상태.
- `af1be70`: mean/median, failed-cell, trace provenance 처리.
- `a32f7f2`: workspace 변경 시 CUDA graph와 captured-shape cache 무효화.
- `7f3617c`: Cosmos vLLM raw 결과의 평균 및 trace identity 복구.
- `334f4cb`: sanitizer/lifecycle 확인용 minimal-startup을 명시적 diagnostic 옵션으로 분리.
- `5075ae6`: profile-local carrier를 사용해 packed-prefill 1-token tail을 decode로 오인하지 않도록 수정.

이 목록은 source 변경의 기록이다. 실행 binary와 source의 동일성은 각 campaign manifest의
binary/plugin/source/dirty hash로 별도 확인한다.

## 2. 현재 architecture: 무엇을 그대로 두었는가

```text
HTTP trace / request adapter
           │
           ▼
    request DAG + ready queues
           │
           ├── E: independent TRT encoder context
           ├── P: independent TRT prefill context
           └── D: independent TRT decode context
           │
           ▼
    global action selection
    ├── existing exact CUDA timing + contextual scalar controller
    ├── deterministic dependency / memory / context feasibility
    └── service-scaled burst / overlap-token controls
           │
           ▼
    independent streams + CUDA completion
           │
           ├── stable indexed-paged KV ownership
           ├── retained vision output leases
           └── workspace-generation-aware CUDA graph replay
```

- CUDA context 공유와 TensorRT execution context 독립은 유지한다.
- KV allocator, stable slot/page ownership, attention kernels, model weights와 quantization은
  이번 변경에서 교체하지 않았다. 다만 검증 도중 드러난 attention **plugin의 P/D dispatch 분류**는
  `5075ae6`에서 수정했다. 아래9.2절은 이 correctness 변경을 별도 설명한다.
- 기존 contextual scalar overlap controller도 유지한다. 이번 `transition predictor on/off`
  A/B는 **전체 contextual RLS on/off가 아니다**. 아래 queue-observation 및 burst/token-limit
  경로의 authority를 비교하는 축이다.
- Workload 이름으로 policy를 선택하지 않는다. 다만 이를 모든 설정이 자동 학습되었다는 뜻으로
  해석해서는 안 된다. 뒤의 고정 설정과 남은 heuristic을 함께 기록한다.

## 3. Observation contract 수정

구현 위치:

- [`phaseTransitionPredictor.h`](../cpp/runtime/phase/policy/phaseTransitionPredictor.h)
- [`phaseTransitionPredictor.cpp`](../cpp/runtime/scheduling/phaseTransitionPredictor.cpp)
- [`phaseQueueScheduler.cpp`](../cpp/runtime/scheduling/phaseQueueScheduler.cpp)

### 3.1 Feature 정의를 하나로 통일

`PhaseDecodeQueueState`는 `batchRows`와 `contextTokens`를 명시한다.
`contextTokens`는 생성할 output token 수가 아니라 cohort의 KV context length 합이다.
관측과 조회는 동일한 `phaseDecodeQueueFeatures()`를 사용한다.

```text
features = [1, batchRows / 64, summedContextTokens / 65536]
label    = decode-ready queue residence in microseconds
```

기존 training과 inference의 batch normalization 차이를 없앴다. 이 수치는 feature normalization이며,
engine batch capability나 fixed workload class를 정의하지 않는다.

### 3.2 Queue residence를 physical handoff cost로 사용하지 않는다

`observeDecodeQueueWait()`의 label은 scheduler가 만든 queue 체류시간이다. CPU submission,
GPU completion visibility, sampling, E→P physical handoff 각각을 분리한 service measurement가 아니다.
따라서 이를 `decode burst를 키우면 amortize할 수 있는 물리적 transition overhead`로 사용하는
feedback 경로를 제거했다. E/P 경로의 대응되지 않는 label을 독립 physical predictor로 설명하는
것도 중단했다.

현재 queue model은 관측/진단 API를 유지하지만, 이 값으로 physical handoff를 대체하지 않는다.
기존 scalar action-value controller가 없어졌다는 의미는 아니다.

### 3.3 Burst 계산은 완전 학습형이 아니다

`effectiveDecodeBurstLimit()`는 관측된 phase service time 또는 cold fallback을 사용하고,
`recommendedDecodeBurst()`가 bounded 후보를 평가한다. `dispatchOverheadUs`는 현재
**200µs의 미보정 기준값**이며 queue residence에서 학습한 값이 아니다.

남은 값에는 burst 후보 1–16, small-cohort 분기, cold D/P service fallback,
grace 기본 20ms, overlap-token 제한 등이 있다. Queue RLS 자체도 minimum observations=4,
confidence beta=0.5, forgetting factor=0.99 등의 설정이 남는다.

이번 수정의 성과는 heuristic을 모두 제거한 것이 아니라 **학습된 물리량과 설정값을 더 이상
혼동하지 않도록 contract를 정확하게 만든 것**이다. 새 physical handoff learner나 automatic
grace tuning의 성능은 이번 결과에서 주장할 수 없다.

## 4. Smoke와 production의 설정·graph 초기화

구현 위치:

- [`phaseServingExecutionOptions.cpp`](../cpp/runtime/scheduling/phaseServingExecutionOptions.cpp)
- [`independentPhaseCoordinator.cpp`](../cpp/runtime/scheduling/independentPhaseCoordinator.cpp)
- [`phaseServingRuntime.cpp`](../cpp/runtime/scheduling/phaseServingRuntime.cpp)
- [`llm_phase_context_smoke.cpp`](../examples/llm/llm_phase_context_smoke.cpp)
- [`run_production_phase_smoke.sh`](../benchmarks/phase_serving/run_production_phase_smoke.sh)

공통 resolver가 fixed/adaptive chunk, candidate list, overlap limit, cost-aware shape,
decode grace, graph enable/capture limit을 해석하고 effective contract를 로그에 남긴다.
입력 범위와 engine capability를 확인한다. Graph disabled는 online capture도 비활성화한다.

`prepareServingGraphs()`는 request admission 전, coordinator가 drained 상태일 때만 수행한다.
Production에도 실제 decode input을 stage한 뒤 graph를 prime하는 경로를 연결했다.
Graph workspace 설정이 끝난 뒤 capture해야 하며, pointer가 바뀌면 이전 capture는 유효하지 않다.

두 entrypoint가 resolver와 mechanism을 공유한다고 해서 request adapter 전체가 동일해진 것은 아니다.
HTTP campaign은 IPC/smoke serving 경로이고 production API initialization/lifecycle은 별도로
실행해야 한다. 실제 production graph on/off에서 모델당3-request 확인은 완료했지만,
이 작은 smoke를 Full12 quality 및 production 모든 lifecycle의 통과로 확대하지 않는다.

### 역사적 grace 설정의 중요한 차이

September 21 Full24 v1 manifest에는 두 모델 모두 `GRACE_PERIOD_US=150000`이 적혀 있다.
그러나 frozen `949334c`의 smoke 경로는 이 환경변수를 읽지 않았다. 당시 읽는 곳은
`phaseServingRuntime.cpp:169`의 production 경로뿐이며, smoke는 predictor header 기본값
**20000µs**로 실행되었다. Production은 당시 override가 없으면 50000µs를 사용했다.

따라서 새 코드에서 150000을 실제로 소비하게 만들고 이를 “과거와 같은 설정”이라고 부르면 안 된다.
이번 공정 A/B의 canonical grace는 **20ms**다. 150ms 실험은 별도 정책 변경으로 표기해야 한다.

## 5. Workspace 메모리와 overlap의 trade-off

`IndependentEngineExecutorPair`는 context를 합치는 것이 아니라 activation workspace 배치를
바꾼다. 아래 식은 activation workspace만의 개념식이며 model weights, KV, I/O, CUDA graphs,
vision payload와 allocator overhead는 포함하지 않는다.

| Workspace 모드 | activation workspace 개념식 | 허용 execution frontier |
|---|---|---|
| independent | `W_D + W_P + W_E` | 별도 lease와 policy가 허용하면 E/P/D 독립 실행 |
| shared_ep | `W_D + max(W_P, W_E)` | E와 P engine은 상호 배타; E+D 및 P+D 가능 |
| tiered_ep | `W_D + max(align(W_P)+W_E_small, W_E_large)` | small E와 P는 분리; large E는 P와 상호 배타 |

구현: [`independentEngineExecutorPair.cpp`](../cpp/runtime/scheduling/independentEngineExecutorPair.cpp).

Shared E/P가 메모리를 줄이는 대신 E+P 실행 frontier를 제거하는 것은 명시적인 trade-off다.
“context 독립을 유지한다”와 “모든 pair가 동시에 실행 가능하다”는 같은 뜻이 아니다.
Tiered는 실제 small/large visual optimization profile이 필요하다. 한 profile만 가진 engine을
shared로 fallback하여 실행한 결과를 tiered의 효과라고 평가하지 않는다. 새 benchmark runner는
필요한 tiered capability가 없는 경우 이를 명시적으로 거부한다.

진행 중인 screening에서는 Gemma independent mode가 OOM이 나거나 shared mode보다 작은 D graph
cache만 형성하는 경우가 있었다(관측한 D cache coverage: independent10, shared24).
따라서 independent/shared를 graph availability까지 동일한 A/B라고 단정할 수 없다.
수정 shared 경로도 이전보다 약90MiB의 peak 증가가 관측되었으며, 실제 D graph coverage 복원과
함께 해석해야 한다. 이번 graph-lifetime 수정 자체를 순수 VRAM 절감으로 보고하지 않는다.
이 수치는 screening 관측이지 최종 Full12 memory envelope가 아니며, 결과표에 source/config와
graph coverage를 함께 확정해야 한다.

### 5.1 Activation lease와 vision output slab lease는 다르다

E engine이 끝나면 activation workspace lease는 해제할 수 있다. 반면 encoder output은
후속 P consumer가 끝날 때까지 살아 있어야 한다.

```text
E engine completion ───────────► E activation workspace 재사용 가능
      │
      └── vision output slab ──► P ready queue ──► P consume completion ──► 회수
```

Single-storage 경로는 retained payload가 있는 동안 다음 E preparation까지 막는다. 이는 E/P
activation exclusivity보다 강한 제약이다. 메모리 peak 억제에는 유리하지만 E→P→다음 E 진행을
직렬화하여 producer pipeline을 줄일 수 있다.

`TRT_EDGELLM_SHARED_EP_SINGLE_STORAGE=1`은 기본 1-slab bound를 유지한다.
`=0`은 **무제한 allocation이 아니라 최대2 retained batch/slab** 실험을 활성화한다.
`phaseVisionPreparationWithinStorageBudget()`와 retained-storage accounting을 준비/dispatch
경로 양쪽에서 사용한다. 실제 E/P workspace mutual exclusion은 계속 유지한다.

이 interface와 helper 검증은 완료했지만, 두 slab을 실제 request로 채운 상태에서
cancel/release/reuse를 반복하는 GPU lifecycle 판정은 pending이다. 2-slab이 반드시 빠르거나
10GB GPU에서 모든 workload에 맞는다는 결론은 아직 없다.

## 6. 발견한 CUDA graph/workspace lifetime 문제와 수정

초기 screening에서 이전 contract와 달리 P graph cap4 및 모든 decode batch의 synthetic warmup을
사용했다. 이 과정에서 startup OOM과 shared-EP illegal access가 발생했다.
이 run들은 `.local/results/runtime-contract-revalidation-20260926/{prior,screen}` 및
`diagnostic-status.json`에 **diagnostic**으로 보존한다. 기존 primary 성능과 비교하는 표에
실패를 숨기거나 성공한 cell만 전체 결과인 것처럼 포함하지 않는다.

문제 경로는 capture한 graph가 예전 workspace pointer를 유지하는데 shared/tiered arena 재배치가
기존 allocation을 해제하는 것이다. API의 현재 context pointer만 바꿔도 기존 graph의 captured
pointer가 갱신되지는 않는다.

수정 내용:

1. Context memory replacement 전에 해당 stream이 idle인지 확인하고 graph cache를 retire한다.
2. `setContextMemory()`가 graph cache·traced binding을 비우고 context memory generation을 증가시킨다.
3. Graph binding identity에 memory generation을 포함한다.
4. Coordinator가 generation 변경을 감지해 captured shape와 observation cache를 무효화한다.
5. Workspace rebind는 coordinator가 drained 상태여야 한다.
6. Smoke initialization에서 workspace 설정과 graph 준비의 순서를 바로잡는다.

구현 위치:

- [`engineExecutor.cpp`](../cpp/runtime/exec/engineExecutor.cpp)
- [`engineExecutor.h`](../cpp/runtime/exec/engineExecutor.h)
- [`independentEngineExecutorPair.cpp`](../cpp/runtime/scheduling/independentEngineExecutorPair.cpp)
- [`independentPhaseCoordinator.cpp`](../cpp/runtime/scheduling/independentPhaseCoordinator.cpp)

단위 테스트는 stale generation과 binding-cache 불일치를 검증한다. 그것만으로 실제 모든 CUDA
graph 재배치 경로가 안전하다고 확정하지 않는다. 실제 P graph cap4 재현 fixture 및 sanitizer
확인이 최종 확인 항목이다. Primary P graph cap0은 historical contract 복원이지 이 버그를
고치지 않고 숨기는 대체 수단이 아니다.

## 7. 고정 benchmark contract

실행기: [`run_lifetime_encoded_admission.py`](../benchmarks/phase_serving/run_lifetime_encoded_admission.py).

| 항목 | Gemma | Cosmos |
|---|---|---|
| Model | Gemma 4 E2B INT4-AWQ lineage | Cosmos-Reason2-2B FP16 lineage |
| E max | 4 | 4 |
| P max / chunk | 8 / 128 | 8 / 128 |
| D max | 24 | 64 |
| Stable slots / measured inflight | 24 / 24 | 80 / 64 |
| Actual KV pool pages | 192 | 256 |
| Generic calibration requests | 49 | 239 |
| E formation wait | 25000µs | 25000µs |
| Decode burst grace | 20000µs | 20000µs |
| Primary graph contract | graph enabled; P cap0 / D cap64 | graph enabled; P cap0 / D cap64 |
| Shared E/P slab default | 1 | 1 |

KV192/256은 실제 engine config에서 검사한다. vLLM의 KV MiB 설정을 Current page count로
읽지 않는다. Model-specific engine capability와 request capacity를 숨기지 않되, workload마다
이 값을 바꿔서 승자를 고르는 방식은 사용하지 않는다.

고정 engine/vision 경로는 명시적 CLI override를 지원하지만 파일 존재 여부에 따라 다른 engine을
자동 선택하지 않는다. Gemma primary는 `engine-packed-p8-d24-kv2048-p192`와
`visual-e4-soft280/visual`; Cosmos primary는 `engine-p8-d64-kv256-p128-vp1024-atomic`와
`vision-exact-gelu/engine/visual`이다. 모든 hash와 실제 full path는 manifest에 기록한다.

Calibration의 decode batch 목록은 기존 sparse 목록
`1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64`를 model D cap 이하로 자른다.
이는 serving graph bucket preparation과 다른 작업이다. 모든 1..D batch를 warmup request로
추가해 약 2천 요청 규모가 된 실패 run을 과거 generic calibration과 같은 것으로 취급하지 않는다.

September 21 v2는 E wait2ms였으므로 위25ms canonical과 같지 않다. Frozen binary가 같아도
history-versus-new 숫자 차이를 그대로 source regression으로 해석하지 않는다.

### Manifest·재실행·실패 처리

- Source commit/dirty diff, binary/plugin, engine/sidecar, vision, calibration, trace 및 replay tool identity 기록.
- Effective environment와 실제 command 보존.
- 기존 결과 재사용은 cell contract identity가 같을 때만 허용. Aggregate 존재만으로 skip하지 않는다.
- Requested/completed/missing/failed cell을 따로 센다. 하나라도 누락/실패면 runner nonzero exit.
- Failed/partial cell을 reporter가 승리 집계에 포함하지 않는다.
- 한 token hash는 repeatability 미검증이다. 여러 동일 hash도 관측된 일치이지 일반적인 determinism 증명은 아니다.

## 8. Historical results 정정과 frozen vLLM 재사용

[328](328-dual-model-24-clean-sweep-20260921.md),
[329](329-gemma-vision-scaling-and-concurrency-resolution-20260921.md)에 errata를 추가했다.
원래 기록은 삭제하거나 덮어쓰지 않았다.

- Full24 v1/v2는 각 cell 1회였으므로 반복 검증/통계적 승리로 승격하지 않는다.
- Throughput 우세와 TTFT/TPOT/E2E 각각의 우세를 구분한다.
- Note329의 targeted 숫자를 raw가 남은 다른 Full12 run에 섞지 않는다.
- E8 startup OOM은 E8 실행 성공이나 performance point가 아니다.

### 8.1 Cosmos vLLM raw lineage 복구

Historical flattened summary:

`.local/results/v0101-forward-port/v3-service-scale-20260910/vllm-fresh-equal-summary.json`

원본 successful runs는 같은 parent의 다음 네 root에 남아 있다.

- `vllm-fresh-equal-full12-3x`
- `vllm-fresh-equal-resume`
- `vllm-fresh-equal-remaining4`
- `vllm-fresh-equal-vision-third`

12 workload의 **35 successful raw aggregate 모두** 현재 canonical trace SHA256과 일치한다.
Vision-heavy는 성공2/실패1이며 나머지는 각 성공3이다. 실패는 성공치로 impute하지 않는다.
vLLM 0.27.1, FP16 weights/KV, KV budget3.5GiB, measured inflight64, fixed output/ignore EOS,
chunked prefill, prefix caching disabled contract가 commands/version artifact에 남아 있다.

Warmup은 앞7개(short/balanced/decode-heavy/long-prefill/bimodal/text-heavy/mixed)에서64,
나머지5개(vision-heavy/poisson/wave-drain/multi-image/late-vision)에서16이다.
“Vision workload는 모두16 warmup”이라는 historical summary 설명은 정확하지 않았다.

원본 flattened summary의 `*_mean_of_run_means_ms`는 실제로 run mean의 median이었다.
새 artifact에서 arithmetic mean으로 재계산했다. 예를 들어 Cosmos text-heavy의 historical
TTFT mean939.538ms는 arithmetic mean843.347ms, E2E mean1946.624ms는1806.027ms다.
Throughput과 p95 median은 원본 raw 집계와 일치했다.

새 immutable baseline:

`.local/results/review-correction-20260926/cosmos-vllm-frozen-raw-corrected.json`

복구 도구 [`rederive_frozen_vllm.py`](../benchmarks/phase_serving/rederive_frozen_vllm.py)는 원본 SHA,
canonical trace SHA, 각 run의 command/version identity와 warmup contract를 보존한다.
기존 summary는 수정하지 않았다. Trace가 같다는 이유만으로 execution configuration이나
generated token의 exact identity까지 같다고 주장하지 않는다.

### 8.2 새 reporter의 집계 규칙

- Throughput: run throughput의 median.
- TTFT/TPOT/E2E mean: 각 run mean의 arithmetic mean.
- TTFT/TPOT/E2E p95: 각 run p95의 median; 모든 request를 합친 pooled p95와 다르다.
- Peak memory: retained run의 peak 중 최대.
- 성공/실패/반복 수 및 baseline trace identity는 성능 값과 별도로 표시.

```bash
python3 benchmarks/phase_serving/report_workspace_revalidation.py \
  --campaign prior-p0=.local/results/runtime-contract-revalidation-20260926/prior-p0 \
  --cosmos-vllm .local/results/review-correction-20260926/cosmos-vllm-frozen-raw-corrected.json \
  --output-prefix .local/results/runtime-contract-revalidation-20260926/prior-p0-corrected-report
```

Frozen Gemma reference는
`.local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12`다.
Cosmos/Gemma의 frozen vLLM version과 모델/양자화가 서로 같다는 비교가 아니다.
새 trace/output/image contract를 만들지 않는 한 기존 raw를 재사용하며, hardware 상태의 fresh
동시 비교나 새로운 repeat confidence가 확보되었다고 표현하지 않는다.

## 9. 완료한 테스트와 미해결 테스트

| 테스트 | 결과 | 해석 |
|---|---|---|
| Runtime C++ | 393 pass / 1 skip | Skip은 `PhaseKVActiveViewTest.IsolatedMetadataCostBenchmark` |
| Common C++ | 119/120 pass | `NormalizeImage.Accuracy` 실패; 전체 green 아님 |
| NormalizeImage 별도5회 | 3 pass / 2 fail | 재현되는 intermittent failure; 이번 graph 수정 탓이라고 단정하지 않음 |
| State C++ | 78 pass | KV/state 단위 검증 |
| Python contracts | 32 pass = runner/reporter28 + lifecycle4 | CPU 검증; 실제 GPU lifecycle 대체 아님 |
| Packed-prefill classification | 4 pass | 새 P1/D1 carrier predicate 테스트; GPU arithmetic 전체 검증 아님 |

`5075ae6` 이후 `unit-tests-single-token-fix.log`에서 새 predicate4개 및 runtime393 pass/1 skip을
재확인했다. Common/State는 앞의 기록을 유지하며 Common failure를 새 runtime 성공으로 덮지 않는다.

로그:

- `.local/results/runtime-contract-revalidation-20260926/unit-tests-workspace-fix.log`
- `.local/results/runtime-contract-revalidation-20260926/unit-tests-image-repeat-state.log`
- `.local/results/runtime-contract-revalidation-20260926/unit-tests-single-token-fix.log`
- 초기/중간 `unit-tests.log`, `unit-tests-final.log`는 다른 revision의 기록일 수 있으므로
  최신 graph-workspace 결과와 섞지 않는다.

실제 `submit → progress → cancel → drain → reuse`와 busy cancellation contract를 확인하는
[`check_phase_ipc_lifecycle.py`](../benchmarks/phase_serving/check_phase_ipc_lifecycle.py)를 추가했다.
실제 model/GPU를 사용한 최종 lifecycle 결과와 sanitizer 판정은 pending으로 남긴다.

### 9.1 Fixed-output 성능 성공과 output identity는 다른 gate다

성능 trace는 fixed output / `ignore_eos` 조건이다. 따라서 조기에 EOS에 해당하는 token이
나왔더라도 요청 길이를 끝까지 채우면 throughput contract는 만족할 수 있다. 그것이 정상적인
사용자 답변 품질이나 exact greedy identity까지 통과했다는 뜻은 아니다.

Gemma request33/45/57의 확인에서 prior predictor-on과 new predictor-on의 첫 token은106이었고,
new predictor-off는 `Here`로 시작하는 출력 차이가 관측되었다. Captured token ID의 position-wise
agreement는 약95–99%였지만 **exact identity는 아니었다**. 이 차이를 “FP16이므로 무해하다”거나
“출력 길이가 같으므로 correctness 통과”라고 결론 내리지 않는다. 별도 semantic/logit 또는
동일-context 재현 검증이 필요하다. 이후 아래9.2절의 deterministic 분류 버그가 확인되었으므로,
해당 divergence를 단순 numerical boundary 문제라고 가정했던 설명은 철회한다.

Production graph on/off 모델당3-request smoke의 output 확인은 제한적인 성공이다. 위의 workload
출력 divergence와 독립적인 관측이며, 어느 한쪽을 다른 쪽의 correctness 증명으로 사용하지 않는다.

### 9.2 확인된 root cause: 129-token prompt의 마지막 P1이 decode로 분류됨

`5075ae6` 이전 `AttentionPlugin::enqueueImpl()`는 다음 조건으로 packed prefill을 구분했다.

```text
packed enabled && physicalBatch == 1 && runtimeSequenceLength > 1
```

Prompt129를 P128 chunk로 처리하면 실제 dispatch는 `[128, 1]`이 된다. 마지막 physical binding은
`[1,1,C]`이지만 의미는 **KV start128부터 query1개를 추가하는 prefill continuation**이다.
기존 조건은 이를 decode로 분류했다. Runtime은 올바른 query length1과 KV start128을 전달했지만,
decode 경로는 context length1을 cumulative KV end로 해석하여 KV start를 반영하지 않고 position0을
사용했다. Ragged multi-row continuation에서 같은 prompt가 다르게 보인 이유도 이 분류 조건과
연결된다. 이는 policy score를 튜닝해서 해결할 문제가 아니다.

수정은 기존 profile-local packed chunk carrier를 분류 전에 읽는다.

```text
P profile-local capacity128 + actual tokens1 → packed P continuation
D profile-local capacity1   + actual tokens1 → decode
```

구현:

- [`attentionPlugin.cpp`](../cpp/plugins/attentionPlugin/attentionPlugin.cpp)
- [`packedPrefillContract.h`](../cpp/plugins/attentionPlugin/packedPrefillContract.h)
- [`packedPrefillContractTest.cpp`](../unittests/cpp/plugins/attentionPlugin/packedPrefillContractTest.cpp)

Profile-local carrier가 없는 engine과 max chunk capacity 자체가1인 P profile은 여전히 P1/D1을
구별할 수 없다. 이번 수정이 이 경우까지 해결했다고 주장하지 않는다. Prompt, chunk128, batching
policy, KV metadata, weight와 attention kernel 산술을 바꾸지 않고 올바른 경로를 선택하게 한 수정이다.

실제 single-concurrency fixture:

`.local/results/runtime-contract-revalidation-20260926/single-token-prefill-repro/{before,after}/result.json`

| 검증 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 동일 prompt3개 / prompt token 수 | 3 / 129 | 3 / 129 |
| 관측 P dispatch | 각 `[128,1]` | 각 `[128,1]` |
| Calibration / CUDA graphs | 0 / off | 0 / off |
| 첫 token ID | `[106,106,106]` (EOS) | `[8291,8291,8291]` (non-EOS) |
| 중복 prompt의 token sequence 일치 | true | true |
| Fixture의 first-token EOS oracle | fail, 3/3 EOS | pass, 3/3 non-EOS |

두 실행의 binary SHA는 `35bae17c…`로 같고 plugin만 `d3c3c679… → caddc28a…`로 바뀌었다.
따라서 이 fixture에서는 수정 전후 변화가 plugin 분류 수정에 연결된다. 다만 이는 first-token
EOS/metadata oracle 통과이며 full semantic correctness, 다른 엔진과 exact identity, graph-on
P1 fixture, paired ragged output agreement까지 모두 입증한 것은 아니다. 성능 비교도 아니다.

이 발견 때문에 **plugin 수정 전 `screen-p0`와 prior의 성능 결과는 diagnostic**으로 남긴다.
Singleton P1 발생 빈도가 policy/cohort에 따라 달라지므로 pre-fix on/off 차이를 순수 정책 효과로
승격하면 안 된다. 수정 plugin을 고정한 새 campaign이 필요하다.

### 9.3 수정 plugin 이후 production-final: 네 번의 실제 API smoke

Manifest:

`.local/results/runtime-contract-revalidation-20260926/production-final/manifest.json`

Source `5075ae6`, production binary `eebe94d0…`, plugin `caddc28a…`를 기록했다.
Shared E/P, chunk128, grace20ms, explicit SLO 없음, calibration0, online capture off 조건이다.

| Model | Graph flag | 사전 decode captures | 완료 requests | 기록된 error | 같은 모델 graph on/off 출력 |
|---|---:|---:|---:|---:|---|
| Cosmos | 0 | 0 | 3 | 0 | exact |
| Cosmos | 1 | 64 | 3 | 0 | exact |
| Gemma | 0 | 0 | 3 | 0 | exact |
| Gemma | 1 | 24 | 3 | 0 | exact |

비교 필드는 output IDs/text/token count, prompt count, finish reason이다. 두 모델 모두 과일3개,
panda 식별, 숫자5 질문을 확인했고 모든 요청이 EOS로 끝났다. 이는 위 fixture처럼 EOS를 무시한
고정 길이 성공만을 확인한 것이 아니다.

Graph-on은 measurement 전에 synthetic GPU graph priming을 거쳤고 graph-off는 그렇지 않다.
따라서 이 smoke의 latency 감소를 pure graph replay speedup으로 보고하지 않는다. Capture count는
확인되지만 별도 graph replay hit counter는 이 artifact에 없다. 각 flag1회/3request이고 generic
RLS calibration도 없으므로 Full12 HTTP 성능이나 learning-quality gate를 대체하지 않는다.

## 10. Activity mask를 safety assertion으로 사용할 때

E=0001, P=0010, D=0100, Copy=1000은 **stream activity envelope**다. P/D sampling도 해당 phase
activity에 포함될 수 있다. 그러므로 shared E/P에서 모든 `E&P mask duration=0`을 요구하면
올바른 sampling overlap까지 잘못된 workspace overlap으로 취급할 수 있다.

Prior-p0 Gemma vision-heavy에서 관측한0.207092ms E+P 구간은 다음과 같다.

```text
P TRT dispatch end          565.240417ms
P sampling                 565.262939 → 565.676880ms
E encoder engine start     565.469788ms
```

P TRT workspace 사용은 E 시작 전에 끝났다. Shared arena safety의 직접 gate는
`encoder_engine ∩ prefill_dispatch = 0`이며 sampling buffer lifetime은 별도 invariant다.
이 prior-p0 6개 cell에서 해당 engine overlap은0, action fidelity/cost-key parity violation도0이었다.
이것은 frozen reference의 제한된 관측이며 수정본 Full12나 sanitizer 통과를 의미하지 않는다.

## 11. GPU 결과: pre-fix diagnostic과 최종 gate 분리

### 11.1 `screen-p0` 중간 결과 — P1 plugin 수정 이전, 19/24 성공

이 표는 최종 `5075ae6`의 결과가 아니다. Binary `35bae17c…`, plugin `d3c3c679…`의
`screen-p0`이며 manifest는 **partial, requested24/completed19/failed5, exit1**이다.
Gemma independent의 balanced/off, vision-heavy/on·off, multi-image/on·off가 실패했다.
각 성공 cell은1회이고 workload도 모델당3종뿐이다.

아래 값은 raw aggregate이며 latency 단위는ms, memory는MiB다. Mean/p95를 각각 표시하므로
throughput을 포함해 일곱 serving 지표를 모두 포함한다. `on/off`는 transition predictor 설정이다.

| Model / workspace / predictor / workload | tok/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | peak MiB |
|---|---:|---:|---:|---:|---:|
| Cosmos / independent / off / balanced | 4239.40 | 62.73 / 160.92 | 12.73 / 14.68 | 1145.20 / 1778.41 | 9737 |
| Cosmos / independent / off / multi-image | 309.90 | 243.44 / 296.71 | 8.67 / 11.26 | 512.14 / 515.92 | 9743 |
| Cosmos / independent / off / vision-heavy | 701.87 | 1434.35 / 3091.77 | 47.88 / 83.32 | 3272.20 / 3418.72 | 9859 |
| Cosmos / independent / on / balanced | 4280.58 | 61.21 / 156.14 | 12.65 / 14.06 | 1134.90 / 1779.67 | 9745 |
| Cosmos / independent / on / multi-image | 295.79 | 265.57 / 316.18 | 8.62 / 11.08 | 532.76 / 540.59 | 9737 |
| Cosmos / independent / on / vision-heavy | 708.99 | 1479.62 / 3125.82 | 44.84 / 80.68 | 3214.96 / 3410.27 | 9851 |
| Cosmos / shared_ep / off / balanced | 4385.75 | 62.40 / 150.19 | 12.24 / 13.46 | 1103.81 / 1710.21 | 9297 |
| Cosmos / shared_ep / off / multi-image | 175.68 | 377.68 / 623.87 | 6.76 / 7.39 | 587.22 / 846.55 | 9297 |
| Cosmos / shared_ep / off / vision-heavy | 390.08 | 2275.03 / 5509.22 | 8.47 / 12.08 | 2615.19 / 5748.82 | 9297 |
| Cosmos / shared_ep / on / balanced | 4292.01 | 62.98 / 145.90 | 12.58 / 13.96 | 1131.73 / 1774.11 | 9297 |
| Cosmos / shared_ep / on / multi-image | 175.86 | 379.43 / 624.05 | 6.71 / 7.35 | 587.38 / 846.02 | 9297 |
| Cosmos / shared_ep / on / vision-heavy | 389.88 | 2325.83 / 5543.26 | 8.42 / 12.44 | 2662.50 / 5778.49 | 9297 |
| Gemma / independent / on / balanced | 1167.67 | 130.49 / 605.63 | 16.27 / 17.92 | 1490.90 / 2219.56 | 9871 |
| Gemma / shared_ep / off / balanced | 1215.99 | 150.81 / 627.39 | 15.53 / 16.56 | 1448.61 / 2131.79 | 9785 |
| Gemma / shared_ep / off / multi-image | 377.47 | 311.36 / 528.18 | 29.56 / 45.71 | 1227.70 / 1552.16 | 9789 |
| Gemma / shared_ep / off / vision-heavy | 473.97 | 635.18 / 1612.13 | 32.92 / 51.37 | 1860.87 / 4128.09 | 9789 |
| Gemma / shared_ep / on / balanced | 1201.02 | 137.22 / 606.78 | 15.78 / 17.36 | 1452.86 / 2144.97 | 9785 |
| Gemma / shared_ep / on / multi-image | 372.15 | 323.36 / 510.26 | 29.11 / 44.88 | 1225.88 / 1541.00 | 9789 |
| Gemma / shared_ep / on / vision-heavy | 511.46 | 435.92 / 1109.69 | 33.89 / 50.31 | 1664.16 / 2210.29 | 9789 |

원본과 frozen vLLM 대비 일곱 지표의 paired comparison:

- `.local/results/runtime-contract-revalidation-20260926/screen-p0/manifest.json`
- `.local/results/runtime-contract-revalidation-20260926/screen-p0-interim.{json,csv,md,manifest.json}`

Interim reporter는 복구된 Cosmos baseline을 사용한다. 그러나 raw identity를 검증했다고 해서
pre-fix Current output-quality 문제가 사라진 것은 아니므로 final vLLM 승리 표로 사용하지 않는다.

이 제한 안에서 다음 실험을 정하는 단서는 남는다.

1. Cosmos vision-heavy에서 independent/on은 shared/on보다 throughput+81.85%, TTFT p95−43.61%,
   E2E p95−40.98%였지만 **TPOT mean+432.36%, E2E mean+20.75%**였다. 메모리는554MiB 늘었다.
   Producer progress와 resident decode service의 trade-off를 한 지표로 합치면 안 된다.
2. Cosmos multi-image에서 같은 비교는 throughput+68.20%, E2E mean−9.30%, p95−36.10%였지만
   TPOT mean+28.48%, p95+50.75%였고 memory+440MiB였다. 큰 cohort를 빨리 준비하는 방향이
   항상 token 간 지연까지 개선하는 것은 아니다.
3. Cosmos balanced는 independent/on과 shared/on의 throughput 차이가−0.27%에 그쳤는데
   memory는448MiB 늘었다. 따라서 모든 경우에 independent가 더 낫다는 결론도 아니다.
4. Gemma vision-heavy shared/on은 off보다 throughput+7.91%, E2E p95−46.46%였으나1회다.
   Balanced/multi-image에서는 throughput이 각각−1.23%/−1.41%였다. P1 분류 오류까지 있는
   revision이므로 이것을 predictor의 보편적 승리나 실패로 승격하지 않는다.

최소 다음 비교는 same engine/trace/calibration의 수정 plugin 조건에서 이루어져야 한다.
특히 two-slab 실험은 shared의 memory 이점을 일부 유지하면서 producer 병목을 줄일 수 있는지
검증하는 가설이며, 위 independent 수치를 그대로 기대 개선치로 옮기지 않는다.

### 11.2 최종 GPU gate — 아직 pending

다음 표에는 최종 결과가 확보된 후에만 raw artifact를 연결한다. 현재 빈 항목을 historical
single-run 수치나 targeted 최고 수치로 채우지 않는다.

| Gate | 요구 evidence | 상태 |
|---|---|---|
| Prior/current 동일 canonical contract | 같은 engine·trace·calibration·graphs·memory, source만 구분 | pending |
| Predictor on/off | 동일 binary에서 burst/token-limit authority만 변경 | pending |
| Independent/shared_ep | 같은 KV pool, engine frontier 및 memory peak를 함께 비교 | pending |
| Shared 1-slab/2-slab | 실제 retained batches, release/reuse, peak, serving latency | pending |
| Tiered | genuine multi-profile visual engine; small/large lease 검증 | pending |
| Production graph on/off | 실제 API initialization 및 request outputs | 모델당3-request smoke 완료; Full-suite quality pending |
| P graph cap4 safety regression | 기존 stale-workspace 문제의 실제 재발 여부 | pending |
| P1 continuation 확장 gate | 수정 plugin의 graph-on·ragged paired identity·Full12 재검증 | singleton graph-off fixture 완료; 나머지 pending |
| Cancel/continuous admission/reuse | 실제 GPU lifecycle, no invalid ownership | pending |
| Two-model Full12 ×3 | 72 cells per selected fixed configuration; seven metrics+memory | pending |
| Cross-variant output identity | Fixed-output 성공과 분리한 captured token/semantic 비교 | Gemma divergence 관측; exact gate 미통과 |
| Promotion | correctness + repeated regression review + known failure disclosure | 미결정 |

## 12. 다음 판단 원칙

1. 먼저 수정본의 동일-contract screening과 lifecycle을 완료한다. 성능이 나쁘더라도 실패나
   contract mismatch를 빼고 가장 좋은 cell만 모아 champion을 만들지 않는다.
2. Shared activation arena가 절약한 memory와 single-slab barrier가 잃은 pipeline opportunity를
   분리해서 본다. 2-slab은 이 가설의 실험이지 검증된 최적값이 아니다.
3. One fixed configuration을 선택한 뒤 두 모델 Full12 ×3을 수행한다. Workload별 E wait,
   grace, graph cap, predictor mode를 바꿔서 성능을 맞추지 않는다.
4. Physical handoff 학습이 필요하면 GPU/CPU boundary label contract부터 새로 정의한다.
   이번 queue-wait RLS를 이름만 바꿔 physical transition model이라고 부르지 않는다.
5. 최종 표는 throughput과 TTFT/TPOT/E2E mean/p95, peak memory를 각각 보인다.
   vLLM 대비 all-workload 또는 all-metric 개선 여부는 completed campaign raw로만 판정한다.

현재 결론: **관측·실행·평가 contract와 workspace graph lifetime은 구현 수준에서 정리했고,
raw baseline 오류도 복구했다. 추가로 P1 tail의 잘못된 decode 분류를 실제 GPU fixture로 재현하고
수정했다. 새 controller의 우월성, shared memory 정책의 최적성, 두 모델 전체 workload 무회귀는
아직 증명하지 않았다.**

## 13. 최종 재검증 전 추가 correctness 수정

### 13.1 Packed attention의 physical decode workspace (`55b3c8e`)

Packed P carrier는 `[1,totalTokens]`지만 D carrier는 `[batch,1]`이다. 기존 workspace 계산은
packed 설정이면 양쪽 모두 physical batch 1로 계산했다. 실제 D의 Q scratch에는 다음 크기가 필요하다.

| Shape | 기존 plugin 선언 전체 bytes | 실제 D Q scratch bytes |
|---|---:|---:|
| Gemma D24/H8/d256 | 8,704 | 98,304 |
| Gemma D24/H8/d512 | 16,896 | 196,608 |
| Cosmos D64/H16/d128 | 9,600 | 262,144 |

이는 plugin의 크기 contract 오류다. 더 큰 전체 TRT workspace가 존재하므로 과거 모든 실행에서
CUDA OOB가 발생했다고 단정할 수는 없다. 수정은 실제 physical batch×sequence로 scratch를
계산하며 packed P의 compact allocation은 유지한다. KV pool 용량이나 page allocator 변경이 아니다.
기존 직렬화 engine이 수정된 크기를 안전하게 반영한다는 보장이 없어, 같은 ONNX와 builder
capability로 새 engine을 생성한다. 기존 engine과 current pointer는 보존한다.

재현 절차: `scripts/rebuild_phase_attention_workspace.py` (기본 dry-run, `--execute`로 실행).
산출물: `.local/artifacts/v0101-forward-port/workspace-corrected-20260926/{gemma,cosmos}`.
Sidecar는 hardlink하며 model weights를 중복 복사하지 않는다. 엔진 재빌드가 포함되므로 이후
성능 차이를 scheduler source만의 pure A/B로 해석하지 않는다.

### 13.2 취소 접수와 GPU ownership 회수의 분리 (`cf196e4`, `43c680a`)

수정 전 Cosmos lifecycle에서는 매 token마다 취소를 시도해도 다음 D가 이미 in-flight여서
128회 시도 중 접수가 없었다. 이는 busy-window starvation이며 KV corruption 증거는 아니다.
Gemma에서는 2회 중 1회 busy 후 접수되어 survivor32/readmission16 tokens를 완료했다.

수정 후 `cancel()==true`는 취소 의사 접수다. GPU와 sampling consumer가 끝날 때까지 request,
KV, vision payload를 유지하며 안전한 경계에서만 회수한다. 추가 D와 정상 completion은 억제한다.
Three-phase의 request ID 및 downstream memory accounting도 실제 회수까지 유지한다.
회수 직전에 scheduler와 sampling consumer가 없음을 검사한다. 이 경로의 통합 GPU 재검증은
아래 최종 결과로 별도 확정한다.

### 13.3 단위 검증과 sanitizer 판정

`unit-tests-safety-final.log`: plugin 7/7, runtime 688 pass/2 optional skip.
Python at this checkpoint: baseline/report28, lifecycle14, singleton5, output-audit9, rebuild4 tests pass.
기존 `NormalizeImage.Accuracy`의 반복 간 실패는 별개로 남긴다(앞 절 참조).

수정 전 Gemma minimal-startup memcheck는 ready 이후 target이 비정상 종료했다.
`ERROR SUMMARY: 0 errors`만으로 통과라고 하지 않는다. 당시 token/cancel/reuse 경로를
완료하지 못했으며 실패 로그를 보존한다.

### 13.4 두-slab 1차 screen

`two-slab-screen`: 두 모델×balanced/vision-heavy/multi-image 6/6 실행 완료.
P1 수정 plugin을 사용했고 아직 workspace 재빌드 전이다. HTTP/request-count/full-token-capture
integrity 오류0, 첫 EOS anomaly0이다. 이전 pre-P1 screen에는 해당 anomaly가5건 있었다.
두-slab은 `max_retained_batches=2` 설정이지만 실제 concurrent retained peak2를 직접 증명하는
카운터는 없다. 설정값과 실제 점유 관측을 혼동하지 않는다.

Peak MiB: Gemma balanced9785/vision-heavy9795/multi-image9795;
Cosmos9345/9379/9343. 단일 run이며 최종 default로 승격하지 않는다.
성능 일곱 지표는 `two-slab-report.md`, 요청별 검사는 `two-slab-quality.md`에 보존한다.
동일 새 engine의 one-slab/ two-slab 비교와 Full24×3 완료 뒤 최종 결론을 갱신한다.

## 14. Workspace-corrected engines와 최종 lifecycle evidence

### 14.1 재빌드 identity와 공정성 제한

두 모델 모두 기존 ONNX와 같은 builder configuration으로 새 plan을 만들었다. 기존 plan과
`.local/current/`는 변경하지 않았다. `builder_config.json`은 기존과 byte-identical이며 KV pool,
FP16 KV, quantization, batch capability를 줄이지 않았다. 생성 절차는
`scripts/rebuild_phase_attention_workspace.py`, build manifest는 campaign의
`workspace-rebuild/{gemma,cosmos}/manifest.json`에 있다. Build-only manifest의
`built_not_inference_validated`는 build 단계의 상태이며 이후 검증은 아래 별도 artifact에 기록한다.

| Model | 새 engine 디렉터리 | Engine SHA256 |
|---|---|---|
| Gemma | `.local/artifacts/v0101-forward-port/workspace-corrected-20260926/gemma` | `fef5210c22b0ceb064cdce07658f6dace7737d66cad7e35a3f625602f2c9405e` |
| Cosmos | `.local/artifacts/v0101-forward-port/workspace-corrected-20260926/cosmos` | `c4f873c30db785cb87aba4475cc80b935112f225c3e2204f22c2ade09356026d` |

새 runtime/plugin의 C++ source build는 `43c680a`다. 최종 반복 campaign은 `47a9cf7` checkout에서
시작했고 source dirty 상태, binary/plugin SHA, engine SHA를 manifest에 분리 기록한다.
Report/helper/note 변경은 실행 중인 binary 변경과 동일하지 않다. Runner는 매 cell에서 실행 파일과
plugin hash가 달라지면 실패하도록 되어 있다.

Gemma에는 예상 밖의 resident weight 증가가 있었다.

| 항목 | 보존된 이전 plan | 새 plan |
|---|---:|---:|
| Engine 파일 bytes | 1,388,545,260 | 1,461,395,724 |
| TRT LLM managed memory 로그 MiB | 1,290 | 1,360 |
| Vision 로드 후 managed memory 로그 MiB | 1,612 | 1,682 |
| P context workspace bytes | 183,647,744 | 183,647,744 |
| D context workspace bytes | 28,401,664 | 28,401,664 |

Plan 파일 증가는 약69.48MiB, resident managed-memory 차이는 약70MiB다. **KV 증가나 P/D context
workspace 증가가 아니다.** Attention workspace 선언 수정 후 새로 빌드하며 tactic/weight packing이
달라졌을 가능성은 있지만 특정 tactic이 원인이라고 입증하지는 않았다. 과거 September13 build
manifest의 engine size는 현재 보존된 이전 plan과 일치하지 않으므로 그 build weight log를 정확히
이전 artifact의 측정값으로 가져오지 않는다. Cosmos의 D workspace21,548,544bytes는 유지됐다.

따라서 새 binary의 성능을 frozen prior와 비교할 수는 있어도, 그 차이를 controller 변경만의 효과라고
말할 수는 없다. 이번에는 correctness 수정, graph coverage 변화, plan 재빌드가 함께 들어갔다.

### 14.2 실제 GPU cancel → survivor → readmission

`lifecycle-safety/{gemma,cosmos}/result.json`의 두 모델 모두 성공했다.

| Model | Cancel 시도 / busy 거절 | Survivor output | Readmission output | D graph 준비 |
|---|---:|---:|---:|---:|
| Gemma | 1 / 0 | 32 tokens | 16 tokens | 24 |
| Cosmos | 1 / 0 | 32 tokens | 16 tokens | 64 |

이는 cancellation intent를 접수한 뒤 outstanding GPU/sampling consumer가 끝날 때까지 ownership을
보존하는 실제 경로의 검사다. 취소 접수 직후 메모리를 회수하는 것이 아니다. 모든 cancellation
interleaving을 증명하는 것은 아니지만 이전 Cosmos의 busy-only starvation은 이 fixture에서 재현되지
않았다. 실행 contract는 `lifecycle-new-contract.json`과 개별 result의 launched identity에 보존했다.

### 14.3 새 plan의 production API graph on/off

`production-safety/manifest.json`과 네 개 실행 artifact를 확인했다. 두 모델×graph on/off 각각
3requests, 총12requests가 정상 EOS로 완료됐다. 모델별 paired request의 IDs, text, token count,
prompt count, finish reason은 모두 같다. 같은 세 질문에 대해 재빌드 이전 `production-final`과도
출력이 일치했다. D graph 준비 수는 Gemma0/24, Cosmos0/64다.

이 검사는 production entry point가 새 plan에서 graph 설정을 실제로 반영한다는 evidence다.
Small smoke의 output identity이며 Full12의 cross-repeat exact identity나 HTTP latency gate를
대체하지 않는다. Graph priming의 warm-up 차이 때문에 이 네 실행으로 pure graph speedup도 주장하지 않는다.

### 14.4 Sanitizer: 오류0 문자열은 통과가 아니다

Gemma의 기존 minimal-startup memcheck와 새 plan Cosmos의 memcheck가 모두 ready 이후 첫 E 요청
부근에서 target abnormal exit로 끝났다. 각각 `lifecycle-memcheck/gemma`,
`lifecycle-safety-memcheck/cosmos`에 실패를 보존했다. `ERROR SUMMARY: 0 errors`가 출력되어도
cancel/reuse fixture를 끝내지 못했으므로 **sanitizer gate는 실패/미확정**이다.

Kernel journal의 CPU SIGSEGV instruction을 동일 container의 ELF offset과 대조하면 두 실행 모두
glibc `__pthread_rwlock_rdlock+0x15`의 `mov 0x18(%rdi),%edx`다. 잘못된 lock pointer가 보이지만
caller stack이 아직 없으므로 TensorRT, sanitizer, 우리 runtime 중 어느 쪽의 원인이라고 단정하지
않는다. 이 정보는 CPU crash 위치를 좁힌 것이지 GPU memory safety를 증명한 것이 아니다.
Text-only fixture와 bounded backtrace로 vision execution 의존성 및 caller를 추가 확인한다.

### 14.5 출력 반복 동일성과 request 완료는 별도 gate

중간 Gemma mixed R1/R2에서는64requests 중56개가 전체 token sequence 동일했다. 첫 stop token을
포함한 prefix까지 비교하면58개가 같다. 두 요청은 정상 답변/종료 이후 강제 `ignore_eos` 출력만
달랐고 나머지6개는 실제 응답 안에서 분기했다. 두-image 요청에서는 Dog→Red Panda 순서가
유지됐지만 이것을 full semantic identity로 취급하지 않는다.

8개 raw-divergent 요청 모두 P1 tail이 없었다. 유일한 P1 tail 요청29는 두 반복에서 전체 출력이
같았다. 반면 첫 token divergence에 대응하는 D batch는 다음처럼 달랐다.

| Request ID | 첫 분기 token index (0-based) | R1 → R2 D batch |
|---|---:|---|
| 2 | 18 | 8 → 24 |
| 10 | 26 | 6 → 24 |
| 46 | 20 | 22 → 23 |
| 48 | 22 | 22 → 23 |
| 51 | 20 | 21 secondary graph → 20 primary graph |
| 59 | 30 | 5 → 11 |
| 60 | 30 | 5 → 4 |
| 61 | 20 | 7 → 15 |

특히 text requests2/10은 P membership/shape가 같았다. 다음 진단 축은 동일 요청/KV 상태에서
D binding batch와 graph/eager만 바꾸는 controlled comparison이다. 현재 logits가 없으므로
이를 benign FP16 rounding이라고 단정하지 않으며 exact-output promotion은 통과로 표시하지 않는다.

## 15. 선택된 기본 계약: 두 모델 Full12 ×3 완료

`full24-final-3x`는 **72/72 완료, failed/missing0**이다. Source build `43c680a`, binary
`aab919194709253bdc377830653e1c201abf4a0b34967469cf0db0832cb1552b`, plugin
`ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2`를 사용했다.
Primary 완료 뒤 같은 binary/plugin을
`.local/baselines/runtime-contract-43c680a-20260926/bin/`에 보존했다.

이후 M-RoPE admission 후보 `5520216`은 targeted screen에서 resident text latency 회귀로
기각했고 `1ebaf63`으로 그 변경만 되돌렸다. `git diff 43c680a -- cpp unittests`는 비어 있다.
따라서 아래 표는 선택된 기본 runtime의 결과이며, 기각 후보의 좋은 수치를 섞지 않는다.
자세한 causal 검사는 [333](333-encoder-admission-mrope-lifetime-fix-20260926.md)에 있다.

### 15.1 전체 일곱 지표: Current / frozen vLLM (변화율)

Throughput은 높을수록 좋고 모든 latency는 낮을수록 좋다. 단위는 token/s와ms다.
Mean은 run별 mean의 산술평균, throughput/p95는 run별 수치의 중앙값이다. Pooled request p95나
confidence interval이 아니다. Current는 각3회, Gemma frozen vLLM은1회이며 Cosmos는
대부분3회·vision-heavy2회 성공+1회 실패다. 실패를 성공으로 치환하거나 vLLM까지 모두3회라고
표시하지 않는다. 기존 계약이 유지된 vLLM raw를 재사용했고 이번에 fresh vLLM은 실행하지 않았다.

| Model/variant/workload | Runs | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosmos/shared_ep-predictor-on/balanced | 3 | 4264.36 / 4315.77 (-1.19%) | 65.88 / 112.40 (-41.39%) | 163.02 / 254.04 (-35.83%) | 12.67 / 12.20 (+3.86%) | 14.47 / 13.58 (+6.50%) | 1141.89 / 1154.35 (-1.08%) | 1793.42 / 1771.35 (+1.25%) |
| cosmos/shared_ep-predictor-on/bimodal | 3 | 1910.88 / 1873.00 (+2.02%) | 1884.44 / 1548.74 (+21.68%) | 4027.70 / 2631.30 (+53.07%) | 16.93 / 22.59 (-25.03%) | 27.10 / 37.47 (-27.69%) | 4245.18 / 4642.60 (-8.56%) | 8924.95 / 9020.73 (-1.06%) |
| cosmos/shared_ep-predictor-on/decode-heavy | 3 | 4991.78 / 4937.33 (+1.10%) | 67.71 / 118.57 (-42.90%) | 161.37 / 321.61 (-49.82%) | 10.78 / 11.01 (-2.08%) | 11.40 / 11.63 (-1.96%) | 2851.77 / 2969.42 (-3.96%) | 4355.96 / 4491.39 (-3.02%) |
| cosmos/shared_ep-predictor-on/late-vision | 3 | 2425.67 / 2165.09 (+12.04%) | 115.28 / 250.16 (-53.92%) | 443.39 / 847.64 (-47.69%) | 9.71 / 10.77 (-9.85%) | 9.79 / 10.78 (-9.19%) | 1505.66 / 1792.55 (-16.00%) | 1901.53 / 2130.19 (-10.73%) |
| cosmos/shared_ep-predictor-on/long-prefill | 3 | 1313.27 / 1123.89 (+16.85%) | 1969.66 / 1916.21 (+2.79%) | 2546.50 / 2947.28 (-13.60%) | 22.35 / 32.19 (-30.56%) | 26.45 / 37.37 (-29.22%) | 3869.56 / 4652.88 (-16.84%) | 5285.02 / 6586.48 (-19.76%) |
| cosmos/shared_ep-predictor-on/mixed | 3 | 694.89 / 923.32 (-24.74%) | 1064.57 / 858.42 (+24.01%) | 3458.87 / 2542.45 (+36.04%) | 10.64 / 47.70 (-77.70%) | 15.44 / 83.92 (-81.60%) | 1580.06 / 2997.41 (-47.29%) | 3686.67 / 3132.22 (+17.70%) |
| cosmos/shared_ep-predictor-on/multi-image | 3 | 215.69 / 243.90 (-11.57%) | 314.79 / 262.02 (+20.14%) | 518.97 / 401.97 (+29.11%) | 6.92 / 12.28 (-43.60%) | 7.58 / 16.27 (-53.42%) | 529.42 / 642.59 (-17.61%) | 736.33 / 654.42 (+12.52%) |
| cosmos/shared_ep-predictor-on/poisson | 3 | 1608.98 / 1781.11 (-9.66%) | 353.20 / 435.59 (-18.92%) | 1840.57 / 923.41 (+99.32%) | 14.13 / 22.35 (-36.81%) | 18.42 / 46.23 (-60.15%) | 1407.74 / 1815.85 (-22.47%) | 2096.52 / 2287.42 (-8.35%) |
| cosmos/shared_ep-predictor-on/short | 3 | 2286.71 / 2046.18 (+11.76%) | 98.14 / 180.00 (-45.48%) | 208.14 / 256.73 (-18.93%) | 12.78 / 12.82 (-0.31%) | 21.08 / 24.13 (-12.64%) | 335.01 / 420.08 (-20.25%) | 427.40 / 492.75 (-13.26%) |
| cosmos/shared_ep-predictor-on/text-heavy | 3 | 1349.71 / 1292.39 (+4.44%) | 412.51 / 843.35 (-51.09%) | 1828.46 / 2108.36 (-13.28%) | 12.15 / 19.29 (-36.99%) | 15.44 / 41.45 (-62.75%) | 1072.30 / 1806.03 (-40.63%) | 2037.52 / 2518.49 (-19.10%) |
| cosmos/shared_ep-predictor-on/vision-heavy | 3 | 390.47 / 577.19 (-32.35%) | 2289.59 / 1630.87 (+40.39%) | 5484.20 / 3544.35 (+54.73%) | 8.46 / 65.15 (-87.02%) | 12.36 / 120.75 (-89.76%) | 2627.20 / 4087.81 (-35.73%) | 5708.65 / 4240.20 (+34.63%) |
| cosmos/shared_ep-predictor-on/wave-drain | 3 | 94.87 / 95.82 (-0.99%) | 297.21 / 254.96 (+16.57%) | 499.27 / 420.78 (+18.65%) | 7.00 / 12.43 (-43.68%) | 8.22 / 17.27 (-52.43%) | 514.13 / 640.15 (-19.69%) | 717.37 / 650.51 (+10.28%) |
| gemma/shared_ep-predictor-on/balanced | 3 | 1203.25 / 771.46 (+55.97%) | 139.97 / 134.27 (+4.25%) | 608.01 / 234.33 (+159.47%) | 15.68 / 23.76 (-34.00%) | 17.33 / 24.56 (-29.45%) | 1448.48 / 2128.03 (-31.93%) | 2144.43 / 3244.79 (-33.91%) |
| gemma/shared_ep-predictor-on/bimodal | 3 | 808.80 / 600.16 (+34.76%) | 728.29 / 317.41 (+129.45%) | 1319.47 / 878.39 (+50.21%) | 20.20 / 28.83 (-29.94%) | 35.97 / 37.42 (-3.89%) | 3446.81 / 4389.95 (-21.48%) | 7090.75 / 9582.02 (-26.00%) |
| gemma/shared_ep-predictor-on/decode-heavy | 3 | 1317.72 / 812.43 (+62.19%) | 516.77 / 153.61 (+236.41%) | 1477.86 / 249.66 (+491.94%) | 14.29 / 23.04 (-37.96%) | 15.03 / 23.41 (-35.82%) | 4133.93 / 6006.17 (-31.17%) | 5773.86 / 9096.01 (-36.52%) |
| gemma/shared_ep-predictor-on/late-vision | 3 | 1496.75 / 990.82 (+51.06%) | 132.78 / 137.71 (-3.58%) | 426.04 / 243.29 (+75.12%) | 13.80 / 22.55 (-38.78%) | 13.97 / 22.55 (-38.06%) | 2110.24 / 3367.53 (-37.34%) | 2750.48 / 4416.79 (-37.73%) |
| gemma/shared_ep-predictor-on/long-prefill | 3 | 503.19 / 500.26 (+0.59%) | 1528.53 / 543.92 (+181.02%) | 3803.05 / 1442.01 (+163.73%) | 24.72 / 36.37 (-32.04%) | 33.88 / 44.60 (-24.03%) | 3611.43 / 3554.56 (+1.60%) | 7542.20 / 5866.97 (+28.55%) |
| gemma/shared_ep-predictor-on/mixed | 3 | 720.57 / 703.81 (+2.38%) | 255.74 / 276.41 (-7.48%) | 465.69 / 410.58 (+13.42%) | 27.56 / 26.48 (+4.07%) | 36.65 / 32.39 (+13.18%) | 1497.34 / 1473.77 (+1.60%) | 2234.18 / 2162.56 (+3.31%) |
| gemma/shared_ep-predictor-on/multi-image | 3 | 372.42 / 381.34 (-2.34%) | 318.02 / 188.86 (+68.39%) | 527.86 / 227.16 (+132.37%) | 29.43 / 29.70 (-0.92%) | 45.18 / 35.54 (+27.14%) | 1230.27 / 1109.61 (+10.87%) | 1546.38 / 1300.30 (+18.92%) |
| gemma/shared_ep-predictor-on/poisson | 3 | 867.69 / 681.95 (+27.24%) | 168.44 / 119.77 (+40.63%) | 618.31 / 169.82 (+264.09%) | 21.20 / 26.02 (-18.52%) | 26.39 / 29.23 (-9.70%) | 1655.24 / 1968.88 (-15.93%) | 2913.67 / 3502.62 (-16.81%) |
| gemma/shared_ep-predictor-on/short | 3 | 804.75 / 567.55 (+41.79%) | 117.86 / 155.42 (-24.17%) | 315.09 / 242.31 (+30.03%) | 20.67 / 26.39 (-21.68%) | 26.12 / 30.11 (-13.25%) | 532.23 / 695.40 (-23.46%) | 809.01 / 1068.16 (-24.26%) |
| gemma/shared_ep-predictor-on/text-heavy | 3 | 855.07 / 404.66 (+111.30%) | 165.05 / 1658.73 (-90.05%) | 426.43 / 4312.05 (-90.11%) | 22.22 / 22.47 (-1.13%) | 27.47 / 27.54 (-0.24%) | 1321.45 / 2814.97 (-53.06%) | 1689.78 / 5848.75 (-71.11%) |
| gemma/shared_ep-predictor-on/vision-heavy | 3 | 498.63 / 559.83 (-10.93%) | 497.48 / 280.57 (+77.31%) | 1132.11 / 390.78 (+189.70%) | 33.96 / 30.59 (+11.04%) | 51.46 / 40.54 (+26.92%) | 1741.81 / 1420.68 (+22.60%) | 2379.54 / 2138.33 (+11.28%) |
| gemma/shared_ep-predictor-on/wave-drain | 3 | 96.87 / 92.62 (+4.58%) | 209.04 / 178.23 (+17.29%) | 327.39 / 204.42 (+60.15%) | 10.66 / 22.48 (-52.58%) | 13.53 / 24.58 (-44.95%) | 539.47 / 874.97 (-38.34%) | 576.91 / 884.36 (-34.76%) |

### 15.2 해석: 모든 workload·모든 지표에서 이긴 결과가 아니다

| 모델 | tok/s 승리 | TTFT mean / p95 승리 | TPOT mean / p95 승리 | E2E mean / p95 승리 |
|---|---:|---:|---:|---:|
| Gemma | 10/12 | 4/12 · 1/12 | 10/12 · 9/12 | 8/12 · 8/12 |
| Cosmos | 6/12 | 6/12 · 6/12 | 11/12 · 11/12 | 12/12 · 7/12 |

Gemma의 decode-heavy는 throughput+62.19%, E2E p95−36.52%지만 TTFT p95+491.94%다.
Cosmos vision-heavy는 E2E mean−35.73%, TPOT p95−89.76%지만 throughput−32.35%,
TTFT p95+54.73%, E2E p95+34.63%다. Throughput 또는 E2E mean 하나로 전면 승리를 주장하면
이 상충관계를 놓친다. 이번 수치는 source-only 개선률이 아니라 최종 실행 계약 대 frozen vLLM이다.

### 15.3 출력·메모리 promotion 제한

6,435 request instances의 HTTP/count/token-capture integrity 검사는 통과했고 first-token EOS
flag는0이었다. 하지만 강제 길이 출력 완료는 semantic correctness나 exact-output gate와 다르다.

| 모델 | 서로 다른 request positions | 3회 raw exact | 첫 stop 포함 prefix exact |
|---|---:|---:|---:|
| Cosmos | 1,513 | 1,513/1,513 | 1,513/1,513 |
| Gemma | 632 | 561/632 | 569/632 |

Gemma의71개 raw /63개 prefix request positions는 반복 중 적어도 한 번 다르다. Wave-drain만
workload 전체 raw exact이며 나머지11개는 하나 이상의 차이가 있다. 분기 원인을 logits 없이
FP16 rounding 또는 KV 오류로 단정하지 않는다. Exact-output promotion은 미통과다.

Peak VRAM은 Cosmos9,299MiB, Gemma9,849–9,853MiB다. SMI total은10,240MiB지만 driver reservation
약365–366MiB를 제외한 usable budget은 약9,874–9,875MiB다. 따라서 추정 headroom은
Cosmos575–576MiB, Gemma최악21–22MiB다. 다른 시각에 읽은 rounded SMI 항목을 peak와
동시 측정한 값처럼 취급하지 않는다. Gemma는 역사적512MiB 목표를 크게 밑돌며 OOM이 이번에
없었다는 것만으로 production capacity를 안전하다고 할 수 없다.

### 15.4 산출물과 후속 범위

Campaign root: `.local/results/runtime-contract-revalidation-20260926/`.

- `full24-final-3x/manifest.json`: 실행 contract, 원본72회, hashes.
- `full24-final-report.{json,csv,md,manifest.json}`: frozen vLLM 비교와 각 run의 범위/편차.
- `full24-final-report-scope.md`: repeatability와 headroom 해석.
- `full24-quality-audit.{json,md}`: 요청별 capture 및 stop-prefix exact.
- `mrope-gate-screen/`: 기각된5520216 후보의6회 screen; primary 표와 합치지 않는다.
- [332](332-gemma-owner-allocation-memory-audit-20260926.md): 다음 메모리 개선 후보의 코드 감사.

다음 memory 후보인 donor-KV 중복 할당576MiB와 image scratch 약72MiB는 아직 **계산·설계**다.
이번에 절감한 실측량으로 보고하지 않는다. KV capacity 감소 없이 접근할 수 있지만 physical owner
alias/copy 안전성과 dynamic scratch의 GPU consumer lifetime을 별도 구현·검증해야 한다.
