<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 336. Owner-aware runtime: 중복 메모리 제거와 독립 실행 가능성 검증

## 1. 질문과 범위

사용자 목표는 workload별 설정표 없이 현재 요청/자원 상태에 대응하여 두 모델의 모든 workload에서
vLLM보다 좋은 성능을 내는 것이다. 이 목표를 이미 달성했다고 주장하지 않는다. 이번 변경은
그 목표의 기반인 **물리 메모리 ownership, 실제 입력 수요에 따른 버퍼 성장, resident decode 관측**을
분리한다. 단순히 메모리를 줄이면 처리량/latency가 자동으로 좋아진다는 가정도 하지 않는다.

계획은 [335](335-owner-aware-dynamic-runtime-plan-20260926.md), 이전 workspace 반례는
[334](334-workspace-memory-mode-final-screen-20260926.md), admission 반례는
[333](333-encoder-admission-mrope-lifetime-fix-20260926.md)에 있다.

이번에 추가한 로직에는 workload 이름이나 부하별 lookup table이 없다. 기존 25ms encoder formation
wait, 20ms decode grace, engine batch/profile 한도는 바꾸지 않았다. **기존 휴리스틱을 모두 없앤
정책이라는 뜻은 아니다.** 모델이 선언한 KV sharing 관계와 실제 image shape는 capability/data이며
workload label이 아니다.

## 2. 변경 A: logical layer와 physical KV owner 분리

커밋 `5a5e3ca`: `fix: Allocate shared KV pools once per canonical owner`.

```text
logical attention layer
  ├── own layer ───────────────────────┐
  └── borrowed layer → canonical owner ┤
                                      ▼
                         unique physical K/V page pool
                                      │
                         same slot/page lease and capacity
```

- `cpp/runtime/kvCacheManager.{h,cpp}`: donor range/cycle/shape/dtype를 allocation 전에 검증한다.
  donor chain을 canonical owner로 해석하고 owner만 할당한다. 논리 layer 수와 getter indexing은
  유지하되 실제 tensor는 owner를 가리킨다. 빈 donor map은 identity mapping이다.
- `cpp/runtime/hybridCacheManager.{h,cpp}`: compaction/spec copy group도 owner를 한 번만 처리한다.
  동일 storage에 중복 in-place copy하지 않는다. legacy snapshot의 borrower는 owner의 비소유 view다.
  undercommitted pool을 legacy identity-slot 방식으로 넘겨 읽는 경로는 범위 검사에서 거부한다.
- `cpp/runtime/state/sharedResources.cpp`: ordinary/spec/draft cache config에 donor schema를 전달한다.
- `cpp/runtime/state/contextCache/hybridSnapshotStorage.{h,cpp}`: partial prefix snapshot의
  allocation/copy/byte accounting도 unique owner 기준이다.
- `cpp/runtime/qwen3OmniTTSRuntime.cpp`: cache reset도 physical owner만 수행한다.
- `examples/llm/llm_phase_context_smoke.cpp`: 실제 layer별 shape를 합한 `bytesPerPage()`로 broker를
  구성하고 `PHASE_KV_ALLOCATION`을 기록한다. 모든 layer를 단일 head dimension으로 계산하지 않는다.

| 모델 | 논리 layer / 물리 owner | 유지한 pool pages | 이전 KV | 변경 KV | 차이 |
|---|---:|---:|---:|---:|---:|
| Gemma | 35 / 15 | 192 | 1008 MiB | 432 MiB | −576 MiB |
| Cosmos | 28 / 28 | 256 | 3584 MiB | 3584 MiB | 0 |

이는 pooling capacity/정밀도를 줄인 것이 아니다. Gemma의 20개 borrower에 할당되던 사용하지 않는
storage를 없앤 것이다. Cosmos에는 해당 sharing이 없으므로 메모리가 줄지 않는 것이 정상이다.
Gemma KV-only screen의 NVML peak도 balanced 9849→9273MiB, VLM 9853→9277MiB로 정확히
576MiB 감소했다. Cosmos는 모든 대표 cell에서 9299MiB로 유지됐다.

Unit coverage: RuntimeState 82/82, ContextCache 173/173, Runtime 688 pass/2 optional skip.
새 owner 관련 memcheck 4개, error 0. Alias/chains, FP16/FP8, d256/d512, move, partial snapshot,
compaction을 포함한다. Spec/MTP의 모델 수준 E2E를 이번에 새로 검증했다는 뜻은 아니다.

## 3. 변경 B: 이미지 수요 기반 grow-only scratch

커밋 `34a8967`: `perf: Grow Gemma resize scratch from actual image demand`.

`cpp/multimodal/gemma4/gemma4ViTRunner.{h,cpp}`의 `Gemma4ResizeScratch`가 처리한다.

```text
actual raw H×W / resized H×W
        ↓ checked byte calculation
required <= retained capacity? ─ yes → reuse
        │ no
        ▼
cudaMallocAsync(replacements)
        ↓ both allocations succeeded
same-stream / completion-event ordered retirement of old buffers
        ↓
resize kernels → last-use event → subsequent reuse / destruction
```

- raw bytes = raw H × raw W × channels.
- horizontal-pass temporary bytes = raw H × output W × channels × sizeof(float).
- resize가 필요 없는 identity copy는 scratch를 요구하지 않는다.
- 부족할 때만 성장하고 작아진 입력에서도 capacity를 재사용한다. 정상 hot path에서 synchronous
  malloc/free, default stream, device-wide synchronize를 추가하지 않았다.
- 두 replacement를 확보한 뒤 old allocation을 반환하므로 두 번째 allocation OOM 시 기존 capacity를
  보존한다. event는 호출자의 stream handle보다 오래 유지된다. stream이 달라져도 ordering을 보장한다.
- memory pool 미지원 장비만 기존 startup 고정 할당으로 돌아간다. 기존 4096 resize 입력 한도를
  줄이지 않는다. preprocessing graph capture는 명시적으로 지원하지 않는다.
- E output/vision payload/KV는 이 scratch와 별도 ownership이다. P/D graph binding을 바꾸지 않는다.

기존 startup allocation은 99,679,860 bytes(약95.06MiB)다. 현재 calibration image의 scratch는
24,111,360 bytes(약22.994MiB)다. **논리 buffer 약72.066MiB 감소와 NVML 감소는 다르다.**
CUDA pool은 더 큰 단위로 reserve하고 다른 runtime allocation과 공유한다. 첫 measured balanced
shared_ep peak는 9209MiB로 KV-only 9273MiB보다 64MiB 감소했다.

CPU 4개/GPU 5개 새 테스트 9/9 통과. 예상 OOM 테스트를 제외한 GPU 4개는 memcheck error 0.
예상 OOM rollback은 일반 GPU unit에서 통과했다. 모든 이미지 크기의 성능을 입증한 것은 아니다.
성장 시 old+new transient headroom이 필요하며, 이후 큰 이미지의 OOM에 대한 scheduler admission
backpressure는 아직 연결하지 않았다. grow-only이므로 이미 커진 buffer를 작은 요청마다 축소하지 않는다.
Cross-stream scratch sequencing은 runner 전체를 concurrently reentrant하게 만드는 기능이 아니다.
Identity helper의8192×1 unit test는 scratch 미사용을 검사한 것이며 그 image shape의 engine 전체 지원
증명이 아니다. Pool used/reserved는 device-wide 값이며 raw/tmp의 owner별 byte와 구분한다.

## 4. 변경 C: resident decode 관측과 policy authority 분리

커밋 `af78464`: `feat: Observe resident decode service without policy authority`.

Decode 후보 queue만으로는 GPU 실행 중 또는 sampling completion 대기 중인 resident request를
전부 볼 수 없다. 그런데 이 요청의 다음 token은 이미 GPU에서 계산됐을 수도 있다. 이를 무조건
`E time + D time`으로 평가하면 다음 token과 그 다음 token의 horizon을 혼동한다.

따라서 `TRT_EDGELLM_RESIDENT_DECODE_SHADOW=1`은 **진단 전용**이다. 기본값 off이며
`authority_applied=false`, `observation_point=post_selection`으로 기록한다. 기존 action, admission,
candidate eligibility를 바꾸지 않는다. Shadow 기록 비용은 0이 아니므로 primary 성능 실행에는 켜지 않는다.

- token commit에서 측정된 isolated D reference와 epoch를 고정한다. exact/covering 조회의 target은
  B1이나 실제 reference shape가 정확히 B1이라는 주장은 하지 않는다. static/cold fallback을
  만들어 넣지 않고 측정이 없으면 그 epoch 내내 unknown이다.
- queued / inflight / sampling / capacity_wait / unknown을 구분한다.
- uncapped queue membership과 batch cap/eligibility로 제한된 candidate-ready를 따로 기록한다.
- 마지막 token commit 기준 age를 measured reference로 정규화한다. TTL, workload-specific SLO,
  새 timer, fake ready row를 추가하지 않는다.
- candidate의 existing protected-completion에 포함됐는지 기록한다. 미포함이라는 사실만으로
  지연의 원인이나 active guard의 안전성을 입증하지 않는다.

구현 위치: `cpp/runtime/phase/mechanism/phaseResidentDecodeSnapshot.h`,
`cpp/runtime/scheduling/independentPhaseAsyncServer.{h,cpp}`,
`phaseQueueScheduler.{h,cpp}`의 read-only uncapped membership,
`phaseThreeCoordinator.{h,cpp}`의 post-selection emitter다.
`benchmarks/phase_serving/analyze_resident_decode_shadow.py`는 measurement epoch1만 기본 분석하며
epoch가 없으면 unscoped로 표시한다. 관측 횟수와 unique request ID를 구분하고 다른 실행을 합쳐
percentile을 만들지 않는다. Callback은 single owner-thread/non-reentrant contract를 따른다.

## 5. 실험 contract와 보존 자료

Root: `.local/results/owner-aware-runtime-20260926/`.

| 단계 | Binary source | 구성 | 범위 |
|---|---|---|---|
| baseline-screen | e8164e0 | 변경 전, shared_ep | 두 모델 × 대표4 ×1 |
| kv-only-screen | 5a5e3ca | owner KV만, shared_ep | 두 모델 × 대표4 ×1 |
| owner-demand-full24-gpu | 34a8967 | owner KV + demand scratch | 두 모델 ×12 × shared_ep/independent ×1 |

대표4는 balanced/mixed/vision-heavy/multi-image다. `owner-demand-full24`라는 앞선 시도는 sandbox의
NVML 접근 실패(exit9)로 **engine을 한 번도 실행하지 못한 infrastructure failure**다. 실패 manifest를
보존하고 GPU 접근이 가능한 환경의 `owner-demand-full24-gpu`로 분리했다. 성능/OOM 실패와 섞지 않는다.

각 campaign manifest에 source dirty state, 실제 binary source/sha, engine/sidecar/config/trace/
calibration hash, command/environment와 요청 cell 전체를 기록했다. 실행 중 source에서 shadow를
개발해도 frozen 34a8967 binary는 변하지 않는다. engine/export 변경은 없고 기존 producer provenance를
유지했다. `source_commit`과 `binary_source_commit`을 혼동하지 않는다.
공통 generic calibration trace/요청 수를 유지해도 workspace가 허용하는 overlap frontier가 달라지면
실제 probe/학습 관측은 달라진다. 이 비교는 동일 posterior를 강제한 순수 GPU-workspace microbenchmark가
아니라 같은 초기화 입력에서 각 실행 능력을 사용하는 E2E 비교다.

Frozen binaries: `.local/baselines/owner-kv-5a5e3ca-20260926/`,
`.local/baselines/owner-demand-34a8967-20260926/`. Build/test logs는 campaign `logs/`에 보존한다.
current pointer는 바꾸지 않았다. 삭제·push도 수행하지 않았다.

같은 요청/output contract의 frozen vLLM을 사용한다. Cosmos reference는
`.local/results/review-correction-20260926/cosmos-vllm-frozen-raw-corrected.json`이다.
이전 flattened summary가 아니다. Gemma reference는 기존 selected-seq24/kv480/p4096/g24 full12다.

## 6. 해석 원칙

1. shared_ep는 E/P **context**를 합치는 것이 아니라 workspace lifetime을 배타적으로 사용하는 방식이다.
   independent는 별도 workspace로 E/P overlap의 **가능성**을 열며 concurrency를 강제하지 않는다.
2. KV-only는 Gemma memory를 줄이는 변경이다. Cosmos의 single-run 처리량 차이를 이 변경의
   인과적 speedup으로 주장하지 않는다. Cosmos memory는 변하지 않았다.
3. baseline/owner stage의 singleton 비교는 screening이다. 반복 CI나 모든 지표 승리를 주장하지 않는다.
4. greedy token capture/integrity, semantic correctness, repeat exactness는 별개다. 이전 Gemma
   exact-output divergence를 이번 allocation unit test 통과만으로 해결됐다고 하지 않는다.
5. 모든 workload에 한 configuration을 적용한 결과를 각각 공개한다. workload별 best mode를 조합하여
   하나의 동적 scheduler 결과로 표현하지 않는다.

### 6.1 KV-only screen의 전후 지표

`100 × (KV-only / baseline − 1)`이며 각 1회다. throughput은 +, latency는 −가 좋다.
절대값과 frozen vLLM 7지표는 `kv-only-report.{md,csv,json}`에 있다.

| 모델/workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma balanced | +0.14% | +8.26% | +1.96% | −1.16% | −1.16% | −0.04% | −0.74% |
| Gemma mixed | −8.58% | −8.08% | −13.74% | −1.75% | −3.15% | −3.12% | −2.87% |
| Gemma vision-heavy | −2.41% | −4.81% | −4.30% | +4.58% | +0.38% | +2.10% | +0.11% |
| Gemma multi-image | −2.42% | +6.75% | −6.46% | −0.13% | −0.90% | +1.60% | +1.00% |
| Cosmos balanced | +7.13% | +15.40% | +6.21% | −7.91% | −10.37% | −6.77% | −8.38% |
| Cosmos mixed | −4.06% | +3.75% | +3.73% | −1.37% | −0.17% | +2.35% | +3.25% |
| Cosmos vision-heavy | +0.32% | −2.43% | −0.41% | −1.59% | −4.35% | −2.32% | −0.41% |
| Cosmos multi-image | +0.17% | −0.17% | −0.02% | −0.21% | −0.35% | −0.18% | −0.10% |

Cosmos balanced에서 P formation 자체가 175 dispatch/mean BS1.92→118/BS2.85로 변했다.
KV bytes가 같은 negative-control 경로의 +7.13%를 owner allocation의 직접 성능 이득으로 볼 수 없다.

Gemma mixed는 모든 latency가 개선됐지만 throughput은 하락했다. E/P/D GPU event 합은
589/1907/2646→583/1853/2441ms, D dispatch는 155→150, mean BS는18.48→19.09다.
activity window는4191→4005ms로 줄었으나 HTTP duration은4262→4662ms로 늘었다.
HTTP-duration minus activity-window는70.5→656.8ms다. ingress/tail/transport 중 원인은 미확정이다.
이 차이를 GPU KV kernel 회귀라고 결론 내리지 않는다.

기존 HTTP client concurrency cap도 유지했다. Serving TTFT/TPOT/E2E는 실제 전송 이후 latency이며
client concurrency queue 대기는 별도 `client_dispatch_delay`다. offered-arrival부터의 사용자 latency와
혼동하지 않는다. Gemma mixed의 client dispatch p95가 약3초인 점도 raw 결과에 남아 있다.

### 6.2 Cosmos에서 아직 남은 lifetime/admission 병목

KV-only shared_ep 측정 epoch 기준이다. GPU busy는 stream 작업 envelope이지 SM utilization이 아니다.

| workload | E/P/D dispatch | mean E/P/D BS | E queue mean/max ms | E/P/D duty | all-idle |
|---|---|---|---|---|---:|
| mixed | 12/35/376 | 2.67/1.94/7.62 | 2010/3992 | 21.55/21.14/56.39% | 2.69% |
| vision-heavy | 16/47/500 | 3.00/1.40/4.80 | 2730/5553 | 23.53/21.25/53.64% | 2.84% |
| multi-image | 3/4/93 | 1.67/1.25/1.67 | 248/551 | 17.35/14.06/66.44% | 2.15% |

Mixed 11/11, vision-heavy15/15의 다음 E preparation은 이전 cohort 마지막 D 완료 후
0.06–0.19ms 내 시작했다. 이전 마지막 P 완료보다는 약198–242ms 늦다. Embedding을 P가 소비한
뒤에도 M-RoPE storage가 D 동안 살아 있어 single-storage의 `visionPayloadBytes()>0` 판정이 유지된다.
관련 경로는 `phaseThreeCoordinator.cpp`의 후보/실제 encoder 준비 gate와
`phaseVisionAdapter.h`의 downstream-pending admission이다.

E/P workspace의 동시 실행 배타성은 필요하지만 D 종료까지 다음 image preparation을 막아야 한다는
것과는 별개다. 그런데 이 gate를 무조건 없애면 기존333의 text 회귀가 재현될 수 있다. 따라서
**물리 lease feasibility와 resident D 서비스 보호를 분리**하는 것이 다음 과제다. 지금 관측만으로
최적 fairness나 next-token cost를 확정하지 않는다.

Mixed E+D/P+D duty는1.41/0.37%, vision-heavy0.99/0.27%다. 측정 가능한 action-fidelity violation,
memory backpressure는0, D graph64개에 miss/eviction0이다. D admitted cohort는 작지만 서비스는
빠르다. 예를 들어 mixed Current/vLLM TPOT p95는13.40/83.92ms인 반면 TTFT p95는3821/2542ms다.
즉 기존 resident D latency와 새 vision TTFT 사이 trade-off이며 단순한 D starvation이 아니다.
Baseline Cosmos vision-heavy gateway에는 `PHASE_EPOCH`가 없어서 해당 log의 epoch-only dispatch
counts를 만들어 비교하지 않았다. 그 cell의 activity CSV와 HTTP 지표는 별도로 유효하다.

### 6.3 Gemma decode-heavy의 후보 선택 반례

독립/shared 비교에서 serving E dispatch는 양쪽 모두0이다. Independent의 P 총GPU시간은
741.07ms로 shared827.23ms보다 작다. D 평균GPU/dispatch도12.51ms로 shared12.74ms보다 작다.
그런데 independent의 D dispatch는1084회/mean BS15.00이고 shared는873회/BS18.62다.
Admission→첫P 최대대기는5392.87/1353.21ms이며, TTFT p95는5572.71/1469.94ms다.
즉 이 셀의 차이는 느린 P kernel이나 E contention보다는 늦어진 P 진입과 이후 D cohort 분해로 나타난다.

재현할 snapshot: independent decode-heavy `host_monotonic_ns=1563778258509298`.

- P14 ready, P8 후보 hard-feasible/frontier-eligible지만 `dominated=true`, D10선택.
- P reference50.915ms는 measured covering이고 oldest P wait2783.52ms는 약54.67 service quanta다.
- P8/D10/P+D 후보 모두 protected set `{8,19}`와 reference/source/elapsed tuple이 동일하다.
  explicit SLO도 없고 duplicate/invalid reference도 없다. Canonical-set mismatch라는 가설은 기각했다.
- P8/D10의 selection horizon/reference work는 모두86494.018µs다. Top-level uncertainty는
  P4598.175µs, D755.929µs이며 D가 scalar dominance에서 P를 제거한다. 비교는
  `phaseGlobalScheduler.cpp`의 `dominates()`와 pruning, 그 이후 service-normalized override 순서다.
- Request-normalized projected maximum은 P57.74390, D/P+D58.00987이다. 그러나 P는 D의 projected
  age를14.511로 만들고 D 선택은2.141로 유지하므로 per-request Pareto dominance는 성립하지 않는다.

설정도 정확히 구분해야 한다. 이 runner는 `service-scaled-transition` representation을 사용하지만
`TRT_EDGELLM_ENABLE_SERVICE_RECOVERY`, `TRT_EDGELLM_SERVICE_NORMALIZED_AUTHORITY`를 보내지 않는다.
따라서 smoke.cpp의 명시적 config 연결상 두 opt-in selector extension은 꺼져 있다.
이는281에서 recovery를 기본 경로에서 제거한 contract이며, V3라는 이름만으로 두 extension이 켜졌다고
생각하면 안 된다. 로그 `service_normalized_authority=false`만으로 이를 증명한 것은 아니다. 그 flag는
실제로 selection을 override했을 때만true다. 설정은 manifest와 source로 따로 확인했다.

그렇다고 flag를 켜면 해결되는 것도 아니다. 기본 recovery band1.0에서는 두 후보의 maximum 차이
0.266이 band 안이므로 모두 남고, 뒤의 scalar dominance에서 P가 다시 제거된다. Pure normalized
Pareto override도 P/D trade-off를 해결하지 않는다. **새 상수를 조정하면 해결된다는 근거는 없다.**
Protected completion trade-off를 무시하는 pruning과 최종 fairness objective를 별도로 재현/평가해야 한다.

두 실행은 generic49이지만 calibration convergence가 false, exact1/2, contextual2/3이었다.
Shared/independent의 target keys26/32, P+D probes27/32, E+D2/0, E+P0/3으로 posterior가 다르다.
이 때문에 workspace 변경만의 causal slowdown이나 새로운 KV 구현의 성능 회귀라고 단정하지 않는다.

## 7. 최종 실험 결과와 남은 단계

전체48/48 실행 완료, 실패/OOM0, 요청/token capture integrity issue0, first-token EOS anomaly0이다.
각1회이며 반복/semantic/exact identity gate 통과가 아니다. 아래 latency는ms, 처리량은token/s다.
각 셀은 Current / frozen vLLM (상대 변화)이며 latency는음수가 좋다.

### 7.0 전체48 결과: Current / frozen vLLM

Retained/requested cells: 48/48. Full-12 coverage and repeat counts are independent of metric wins.

- cosmos/independent-predictor-on: 12/12 workloads; minimum 1 repeats; Full-12 ×3 coverage: no.
- cosmos/shared_ep-predictor-on: 12/12 workloads; minimum 1 repeats; Full-12 ×3 coverage: no.
- gemma/independent-predictor-on: 12/12 workloads; minimum 1 repeats; Full-12 ×3 coverage: no.
- gemma/shared_ep-predictor-on: 12/12 workloads; minimum 1 repeats; Full-12 ×3 coverage: no.

| Model/variant/workload | Runs | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosmos/independent-predictor-on/balanced | 1 | 4181.93 / 4315.77 (-3.10%) | 62.64 / 112.40 (-44.27%) | 150.59 / 254.04 (-40.72%) | 12.95 / 12.20 (+6.16%) | 14.52 / 13.58 (+6.88%) | 1164.23 / 1154.35 (+0.86%) | 1833.36 / 1771.35 (+3.50%) |
| cosmos/independent-predictor-on/bimodal | 1 | 1871.42 / 1873.00 (-0.08%) | 1937.04 / 1548.74 (+25.07%) | 4410.53 / 2631.30 (+67.62%) | 17.51 / 22.59 (-22.46%) | 28.16 / 37.47 (-24.85%) | 4372.28 / 4642.60 (-5.82%) | 9147.90 / 9020.73 (+1.41%) |
| cosmos/independent-predictor-on/decode-heavy | 1 | 4965.36 / 4937.33 (+0.57%) | 66.19 / 118.57 (-44.17%) | 174.04 / 321.61 (-45.89%) | 10.84 / 11.01 (-1.57%) | 11.48 / 11.63 (-1.27%) | 2861.83 / 2969.42 (-3.62%) | 4384.84 / 4491.39 (-2.37%) |
| cosmos/independent-predictor-on/late-vision | 1 | 2433.54 / 2165.09 (+12.40%) | 131.45 / 250.16 (-47.45%) | 447.85 / 847.64 (-47.16%) | 9.67 / 10.77 (-10.16%) | 9.75 / 10.78 (-9.54%) | 1517.17 / 1792.55 (-15.36%) | 1894.69 / 2130.19 (-11.06%) |
| cosmos/independent-predictor-on/long-prefill | 1 | 1288.62 / 1123.89 (+14.66%) | 1982.13 / 1916.21 (+3.44%) | 2627.02 / 2947.28 (-10.87%) | 23.00 / 32.19 (-28.54%) | 27.27 / 37.37 (-27.01%) | 3936.82 / 4652.88 (-15.39%) | 5428.73 / 6586.48 (-17.58%) |
| cosmos/independent-predictor-on/mixed | 1 | 1120.90 / 923.32 (+21.40%) | 883.31 / 858.42 (+2.90%) | 2041.37 / 2542.45 (-19.71%) | 33.48 / 47.70 (-29.81%) | 60.43 / 83.92 (-27.99%) | 2445.76 / 2997.41 (-18.40%) | 2590.21 / 3132.22 (-17.30%) |
| cosmos/independent-predictor-on/multi-image | 1 | 300.46 / 243.90 (+23.19%) | 279.34 / 262.02 (+6.61%) | 314.61 / 401.97 (-21.73%) | 7.93 / 12.28 (-35.38%) | 9.45 / 16.27 (-41.88%) | 525.27 / 642.59 (-18.26%) | 532.16 / 654.42 (-18.68%) |
| cosmos/independent-predictor-on/poisson | 1 | 1979.89 / 1781.11 (+11.16%) | 258.89 / 435.59 (-40.57%) | 662.93 / 923.41 (-28.21%) | 20.26 / 22.35 (-9.37%) | 38.24 / 46.23 (-17.29%) | 1529.96 / 1815.85 (-15.74%) | 1968.99 / 2287.42 (-13.92%) |
| cosmos/independent-predictor-on/short | 1 | 2290.88 / 2046.18 (+11.96%) | 105.49 / 180.00 (-41.39%) | 196.36 / 256.73 (-23.51%) | 11.89 / 12.82 (-7.20%) | 16.27 / 24.13 (-32.55%) | 337.26 / 420.08 (-19.71%) | 426.68 / 492.75 (-13.41%) |
| cosmos/independent-predictor-on/text-heavy | 1 | 1957.03 / 1292.39 (+51.43%) | 329.85 / 843.35 (-60.89%) | 1162.30 / 2108.36 (-44.87%) | 24.16 / 19.29 (+25.24%) | 36.17 / 41.45 (-12.73%) | 1586.38 / 1806.03 (-12.16%) | 1681.34 / 2518.49 (-33.24%) |
| cosmos/independent-predictor-on/vision-heavy | 1 | 699.77 / 577.19 (+21.24%) | 1470.05 / 1630.87 (-9.86%) | 3076.33 / 3544.35 (-13.20%) | 47.09 / 65.15 (-27.73%) | 82.26 / 120.75 (-31.88%) | 3282.07 / 4087.81 (-19.71%) | 3454.03 / 4240.20 (-18.54%) |
| cosmos/independent-predictor-on/wave-drain | 1 | 97.89 / 95.82 (+2.16%) | 242.13 / 254.96 (-5.03%) | 296.51 / 420.78 (-29.53%) | 8.57 / 12.43 (-31.05%) | 11.68 / 17.27 (-32.36%) | 507.73 / 640.15 (-20.69%) | 513.98 / 650.51 (-20.99%) |
| cosmos/shared_ep-predictor-on/balanced | 1 | 4254.02 / 4315.77 (-1.43%) | 63.73 / 112.40 (-43.30%) | 158.15 / 254.04 (-37.74%) | 12.64 / 12.20 (+3.65%) | 14.10 / 13.58 (+3.81%) | 1137.82 / 1154.35 (-1.43%) | 1764.47 / 1771.35 (-0.39%) |
| cosmos/shared_ep-predictor-on/bimodal | 1 | 1902.16 / 1873.00 (+1.56%) | 1916.80 / 1548.74 (+23.77%) | 4656.07 / 2631.30 (+76.95%) | 17.00 / 22.59 (-24.71%) | 27.05 / 37.47 (-27.82%) | 4303.37 / 4642.60 (-7.31%) | 9581.12 / 9020.73 (+6.21%) |
| cosmos/shared_ep-predictor-on/decode-heavy | 1 | 4951.78 / 4937.33 (+0.29%) | 65.41 / 118.57 (-44.83%) | 182.44 / 321.61 (-43.27%) | 10.85 / 11.01 (-1.42%) | 11.65 / 11.63 (+0.20%) | 2870.55 / 2969.42 (-3.33%) | 4411.71 / 4491.39 (-1.77%) |
| cosmos/shared_ep-predictor-on/late-vision | 1 | 2426.61 / 2165.09 (+12.08%) | 116.03 / 250.16 (-53.62%) | 442.76 / 847.64 (-47.77%) | 9.70 / 10.77 (-9.90%) | 9.78 / 10.78 (-9.27%) | 1505.70 / 1792.55 (-16.00%) | 1900.60 / 2130.19 (-10.78%) |
| cosmos/shared_ep-predictor-on/long-prefill | 1 | 1288.29 / 1123.89 (+14.63%) | 1998.23 / 1916.21 (+4.28%) | 2611.75 / 2947.28 (-11.38%) | 22.73 / 32.19 (-29.39%) | 27.34 / 37.37 (-26.84%) | 3933.30 / 4652.88 (-15.47%) | 5365.94 / 6586.48 (-18.53%) |
| cosmos/shared_ep-predictor-on/mixed | 1 | 667.70 / 923.32 (-27.69%) | 1105.27 / 858.42 (+28.76%) | 3658.07 / 2542.45 (+43.88%) | 9.87 / 47.70 (-79.31%) | 13.35 / 83.92 (-84.09%) | 1580.51 / 2997.41 (-47.27%) | 3859.05 / 3132.22 (+23.20%) |
| cosmos/shared_ep-predictor-on/multi-image | 1 | 207.57 / 243.90 (-14.90%) | 308.74 / 262.02 (+17.83%) | 525.01 / 401.97 (+30.61%) | 7.46 / 12.28 (-39.19%) | 8.21 / 16.27 (-49.56%) | 540.14 / 642.59 (-15.94%) | 770.64 / 654.42 (+17.76%) |
| cosmos/shared_ep-predictor-on/poisson | 1 | 1606.80 / 1781.11 (-9.79%) | 360.19 / 435.59 (-17.31%) | 1845.04 / 923.41 (+99.81%) | 14.19 / 22.35 (-36.52%) | 18.19 / 46.23 (-60.66%) | 1421.11 / 1815.85 (-21.74%) | 2102.48 / 2287.42 (-8.09%) |
| cosmos/shared_ep-predictor-on/short | 1 | 2334.13 / 2046.18 (+14.07%) | 99.34 / 180.00 (-44.81%) | 199.90 / 256.73 (-22.13%) | 12.23 / 12.82 (-4.58%) | 19.72 / 24.13 (-18.25%) | 329.47 / 420.08 (-21.57%) | 418.23 / 492.75 (-15.12%) |
| cosmos/shared_ep-predictor-on/text-heavy | 1 | 1445.61 / 1292.39 (+11.86%) | 416.33 / 843.35 (-50.63%) | 1650.41 / 2108.36 (-21.72%) | 13.01 / 19.29 (-32.53%) | 18.18 / 41.45 (-56.14%) | 1120.69 / 1806.03 (-37.95%) | 1864.76 / 2518.49 (-25.96%) |
| cosmos/shared_ep-predictor-on/vision-heavy | 1 | 382.80 / 577.19 (-33.68%) | 2382.23 / 1630.87 (+46.07%) | 5363.04 / 3544.35 (+51.31%) | 8.70 / 65.15 (-86.65%) | 13.05 / 120.75 (-89.19%) | 2731.58 / 4087.81 (-33.18%) | 5562.31 / 4240.20 (+31.18%) |
| cosmos/shared_ep-predictor-on/wave-drain | 1 | 94.85 / 95.82 (-1.01%) | 253.95 / 254.96 (-0.39%) | 500.18 / 420.78 (+18.87%) | 7.09 / 12.43 (-42.92%) | 8.24 / 17.27 (-52.32%) | 473.82 / 640.15 (-25.98%) | 717.54 / 650.51 (+10.30%) |
| gemma/independent-predictor-on/balanced | 1 | 1224.13 / 771.46 (+58.68%) | 150.37 / 134.27 (+11.99%) | 612.80 / 234.33 (+161.52%) | 15.47 / 23.76 (-34.90%) | 16.93 / 24.56 (-31.09%) | 1439.99 / 2128.03 (-32.33%) | 2104.98 / 3244.79 (-35.13%) |
| gemma/independent-predictor-on/bimodal | 1 | 759.32 / 600.16 (+26.52%) | 1392.60 / 317.41 (+338.74%) | 3573.03 / 878.39 (+306.77%) | 15.43 / 28.83 (-46.47%) | 20.93 / 37.42 (-44.07%) | 3856.57 / 4389.95 (-12.15%) | 9587.16 / 9582.02 (+0.05%) |
| gemma/independent-predictor-on/decode-heavy | 1 | 1092.32 / 812.43 (+34.45%) | 1140.82 / 153.61 (+642.65%) | 5572.71 / 249.66 (+2132.10%) | 14.09 / 23.04 (-38.83%) | 14.91 / 23.41 (-36.32%) | 4712.54 / 6006.17 (-21.54%) | 9711.95 / 9096.01 (+6.77%) |
| gemma/independent-predictor-on/late-vision | 1 | 1494.24 / 990.82 (+50.81%) | 152.49 / 137.71 (+10.73%) | 443.64 / 243.29 (+82.35%) | 13.56 / 22.55 (-39.86%) | 13.60 / 22.55 (-39.69%) | 2094.87 / 3367.53 (-37.79%) | 2673.54 / 4416.79 (-39.47%) |
| gemma/independent-predictor-on/long-prefill | 1 | 599.92 / 500.26 (+19.92%) | 962.66 / 543.92 (+76.98%) | 2083.51 / 1442.01 (+44.49%) | 25.19 / 36.37 (-30.76%) | 32.43 / 44.60 (-27.30%) | 3056.38 / 3554.56 (-14.02%) | 5273.74 / 5866.97 (-10.11%) |
| gemma/independent-predictor-on/mixed | 1 | 663.46 / 703.81 (-5.73%) | 303.30 / 276.41 (+9.73%) | 852.21 / 410.58 (+107.56%) | 25.01 / 26.48 (-5.56%) | 35.10 / 32.39 (+8.38%) | 1440.58 / 1473.77 (-2.25%) | 2215.02 / 2162.56 (+2.43%) |
| gemma/independent-predictor-on/multi-image | 1 | 377.54 / 381.34 (-1.00%) | 425.39 / 188.86 (+125.24%) | 698.28 / 227.16 (+207.39%) | 24.91 / 29.70 (-16.14%) | 39.44 / 35.54 (+10.96%) | 1197.54 / 1109.61 (+7.92%) | 1480.41 / 1300.30 (+13.85%) |
| gemma/independent-predictor-on/poisson | 1 | 794.53 / 681.95 (+16.51%) | 189.81 / 119.77 (+58.48%) | 583.55 / 169.82 (+243.62%) | 22.86 / 26.02 (-12.14%) | 27.64 / 29.23 (-5.42%) | 1796.24 / 1968.88 (-8.77%) | 3263.63 / 3502.62 (-6.82%) |
| gemma/independent-predictor-on/short | 1 | 742.91 / 567.55 (+30.90%) | 237.58 / 155.42 (+52.86%) | 740.12 / 242.31 (+205.44%) | 17.96 / 26.39 (-31.93%) | 25.47 / 30.11 (-15.39%) | 592.37 / 695.40 (-14.82%) | 1148.04 / 1068.16 (+7.48%) |
| gemma/independent-predictor-on/text-heavy | 1 | 848.11 / 404.66 (+109.59%) | 199.40 / 1658.73 (-87.98%) | 508.15 / 4312.05 (-88.22%) | 21.93 / 22.47 (-2.40%) | 26.50 / 27.54 (-3.75%) | 1339.47 / 2814.97 (-52.42%) | 1720.27 / 5848.75 (-70.59%) |
| gemma/independent-predictor-on/vision-heavy | 1 | 497.17 / 559.83 (-11.19%) | 406.43 / 280.57 (+44.86%) | 813.68 / 390.78 (+108.22%) | 31.35 / 30.59 (+2.50%) | 46.36 / 40.54 (+14.35%) | 1583.29 / 1420.68 (+11.45%) | 2272.94 / 2138.33 (+6.30%) |
| gemma/independent-predictor-on/wave-drain | 1 | 91.76 / 92.62 (-0.93%) | 307.63 / 178.23 (+72.60%) | 696.22 / 204.42 (+240.58%) | 11.64 / 22.48 (-48.20%) | 13.98 / 24.58 (-43.11%) | 668.54 / 874.97 (-23.59%) | 944.21 / 884.36 (+6.77%) |
| gemma/shared_ep-predictor-on/balanced | 1 | 1210.52 / 771.46 (+56.91%) | 143.81 / 134.27 (+7.10%) | 619.20 / 234.33 (+164.24%) | 15.61 / 23.76 (-34.30%) | 16.98 / 24.56 (-30.86%) | 1448.05 / 2128.03 (-31.95%) | 2138.57 / 3244.79 (-34.09%) |
| gemma/shared_ep-predictor-on/bimodal | 1 | 716.42 / 600.16 (+19.37%) | 768.64 / 317.41 (+142.16%) | 3152.50 / 878.39 (+258.90%) | 20.78 / 28.83 (-27.90%) | 40.47 / 37.42 (+8.16%) | 3398.69 / 4389.95 (-22.58%) | 7138.66 / 9582.02 (-25.50%) |
| gemma/shared_ep-predictor-on/decode-heavy | 1 | 1323.32 / 812.43 (+62.88%) | 247.83 / 153.61 (+61.33%) | 1469.94 / 249.66 (+488.77%) | 14.44 / 23.04 (-37.33%) | 15.07 / 23.41 (-35.63%) | 3903.74 / 6006.17 (-35.00%) | 5735.22 / 9096.01 (-36.95%) |
| gemma/shared_ep-predictor-on/late-vision | 1 | 1504.34 / 990.82 (+51.83%) | 130.45 / 137.71 (-5.27%) | 412.03 / 243.29 (+69.36%) | 13.83 / 22.55 (-38.68%) | 13.97 / 22.55 (-38.05%) | 2111.09 / 3367.53 (-37.31%) | 2737.65 / 4416.79 (-38.02%) |
| gemma/shared_ep-predictor-on/long-prefill | 1 | 610.95 / 500.26 (+22.13%) | 810.80 / 543.92 (+49.07%) | 1677.05 / 1442.01 (+16.30%) | 26.82 / 36.37 (-26.27%) | 35.14 / 44.60 (-21.21%) | 3047.52 / 3554.56 (-14.26%) | 5250.45 / 5866.97 (-10.51%) |
| gemma/shared_ep-predictor-on/mixed | 1 | 721.87 / 703.81 (+2.57%) | 203.59 / 276.41 (-26.35%) | 416.56 / 410.58 (+1.46%) | 27.30 / 26.48 (+3.09%) | 36.12 / 32.39 (+11.53%) | 1438.77 / 1473.77 (-2.38%) | 2253.17 / 2162.56 (+4.19%) |
| gemma/shared_ep-predictor-on/multi-image | 1 | 382.05 / 381.34 (+0.18%) | 295.30 / 188.86 (+56.36%) | 499.82 / 227.16 (+120.03%) | 29.81 / 29.70 (+0.37%) | 45.85 / 35.54 (+29.02%) | 1219.43 / 1109.61 (+9.90%) | 1554.23 / 1300.30 (+19.53%) |
| gemma/shared_ep-predictor-on/poisson | 1 | 877.56 / 681.95 (+28.68%) | 162.86 / 119.77 (+35.97%) | 552.53 / 169.82 (+225.36%) | 20.85 / 26.02 (-19.85%) | 25.88 / 29.23 (-11.45%) | 1627.63 / 1968.88 (-17.33%) | 2891.07 / 3502.62 (-17.46%) |
| gemma/shared_ep-predictor-on/short | 1 | 798.27 / 567.55 (+40.65%) | 117.82 / 155.42 (-24.19%) | 312.91 / 242.31 (+29.14%) | 20.82 / 26.39 (-21.13%) | 25.97 / 30.11 (-13.73%) | 536.06 / 695.40 (-22.91%) | 812.69 / 1068.16 (-23.92%) |
| gemma/shared_ep-predictor-on/text-heavy | 1 | 844.53 / 404.66 (+108.70%) | 169.02 / 1658.73 (-89.81%) | 583.97 / 4312.05 (-86.46%) | 22.34 / 22.47 (-0.56%) | 26.79 / 27.54 (-2.71%) | 1331.57 / 2814.97 (-52.70%) | 1699.61 / 5848.75 (-70.94%) |
| gemma/shared_ep-predictor-on/vision-heavy | 1 | 456.29 / 559.83 (-18.49%) | 559.00 / 280.57 (+99.24%) | 1445.29 / 390.78 (+269.84%) | 35.86 / 30.59 (+17.23%) | 48.13 / 40.54 (+18.70%) | 1886.20 / 1420.68 (+32.77%) | 3962.36 / 2138.33 (+85.30%) |
| gemma/shared_ep-predictor-on/wave-drain | 1 | 96.85 / 92.62 (+4.57%) | 205.71 / 178.23 (+15.41%) | 324.55 / 204.42 (+58.77%) | 10.85 / 22.48 (-51.73%) | 13.77 / 24.58 (-43.98%) | 542.02 / 874.97 (-38.05%) | 575.83 / 884.36 (-34.89%) |

### 7.1 승수, 메모리, request-class 손익

다음은 frozen vLLM보다 수치가 좋은 workload 수다. 각 Current cell 1회이므로 통계적 승리/동률
판정이 아니다. 예를 들어 +0.18% throughput도 단순 승수에는 포함하지만 향상 확정으로 해석하지 않는다.

| 모델 / workspace | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma shared_ep | 11/12 | 4/12 | 1/12 | 9/12 | 8/12 | 10/12 | 9/12 |
| Gemma independent | 8/12 | 1/12 | 1/12 | 11/12 | 9/12 | 10/12 | 5/12 |
| Cosmos shared_ep | 6/12 | 7/12 | 6/12 | 11/12 | 10/12 | 12/12 | 7/12 |
| Cosmos independent | 10/12 | 8/12 | 11/12 | 10/12 | 11/12 | 11/12 | 10/12 |

| 모델 | Shared peak MiB | Independent peak MiB | 해석 |
|---|---:|---:|---|
| Gemma | 9209–9213 | 9385–9405 | 기존 shared 9849–9853 대비 −640MiB; independent full12 실행 가능 |
| Cosmos | 9299 | 9739–9853 | identity KV owners로 절감 없음; independent는 +440–554MiB |

드라이버 reserve 이후 사용 가능한 약9874MiB 기준 Cosmos independent 최악의 headroom은 약21MiB다.
이번 trace가 통과했어도 모든 입력 크기에 OOM 여유가 충분하다는 뜻이 아니다.

Independent/shared의 12-workload 기하평균 throughput은 Gemma −3.11%, Cosmos +18.02%다.
하지만 Cosmos 평균 E2E는 +10.06%, TPOT mean/p95는 +41.53%/+53.35%다. 특히 전체 평균이
서로 다른 request class의 손익을 숨긴다.

| Cosmos workload / request class | 지표 | Shared ms | Independent ms | 변화 |
|---|---|---:|---:|---:|
| mixed / text | E2E mean | 808.01 | 2529.10 | +213.0% |
| mixed / text | E2E p95 | 876.61 | 2594.91 | +196.0% |
| mixed / vision | E2E p95 | 4069.32 | 2420.86 | −40.5% |
| vision-heavy / text | E2E mean | 803.23 | 3362.51 | +318.6% |
| vision-heavy / text | E2E p95 | 878.53 | 3452.65 | +293.0% |
| vision-heavy / vision | E2E p95 | 5806.88 | 3442.66 | −40.7% |

Independent가 vision 요청을 일찍 진행시키는 이점과 resident text의 비용을 동시에 확인했다.
여기에는 GPU contention뿐 아니라 queue/cohort trajectory 변화가 포함된다. 손실 전부를 overlap
하나에 인과 귀속하지 않는다. Workload마다 승리한 workspace만 골라 단일 policy의 결과처럼 합치지 않는다.

이번 cross-mode 출력도 완전한 exact identity가 아니다. Gemma mixed는 raw50/64,
첫 stop까지53/64 요청만 동일하고 decode-heavy는59/64다. Integrity0과 semantic/exact 동일성은
별개다. 자세한 request ID와 token 비교는 `owner-demand-quality.{json,md}`에 보존했다.

### 7.2 이 단계로 해결했다고 주장하지 않는 것

- grow-only scratch는 KV/vision 사이의 동적 pool 재분배가 아니다. KV page count는 그대로다.
- independent workspace 복구는 실행 **선택지**의 복구이지 항상 overlap을 실행하는 새 정책이 아니다.
- resident shadow는 controller가 아니라 관측이다. sampling completion을 정확히 모델링한 active
  protection이나 admission 해제는 아직 아니다.
- 12-workload를 실행했다는 사실과 모든 workload/all metric에서 이겼다는 판정은 다르다.
- 기존 full24×3의43c680a 결과를 이번34a8967의 반복 실험으로 합치지 않는다.

### 7.3 다음 active 변경의 최소 contract

1. **Memory feasibility**: encoder workspace와 input scratch의 실제 마지막 consumer,
   P가 소비한 vision embedding, D가 보유한 M-RoPE metadata를 각각 다른 lease로 판단한다.
   M-RoPE가 남았다는 이유만으로 모든 다음 E 준비를 막거나, 반대로 전체 gate를 제거하지 않는다.
2. **Service obligations**: ready queue뿐 아니라 in-flight와 sampling resident의 다음 token 의무를
   snapshot에 포함한다. sampling 중인 요청에 이미 끝난 D GPU 비용을 다시 더하지 않는다.
   GPU completion, sampling visibility, host commit 시간축을 구분한다.
3. **Action realization**: E의 host/GPU preparation도 실제 delay를 만드는 작업이다. global action
   선택 밖에서 준비를 먼저 실행하고 나중에 E kernel만 평가하지 않도록 action 경계를 검토한다.
4. **Bounded evaluation**: 현재 알려진 completion/ready boundary와 measured service만 사용해
   기존 V3 evaluator에서 비교한다. workload label, 미래 arrival 예언, 새 고정 wait/SLO 숫자를 넣지 않는다.
   측정이 없는 경우 unknown을 유지하며 unknown을0cost 또는 안전한 positive evidence로 취급하지 않는다.
5. **Promotion**: 동일 snapshot의 action/lifetime 검사 → text/vision class별 mean/p95 → 동일 configuration
   full12×2 반복 순서다. 전체 throughput 증가가 resident text regression을 숨기면 승격하지 않는다.

이 순서는 구현·검증할 contract이며 성능 우위에 대한 수학적 보장이 아니다. 명시적 SLO를 없애더라도
TTFT와 TPOT 사이 어떤 fairness를 선택할 것인지의 정책 문제는 남는다.

## 8. Correctness와 diagnostic 검증

- `af78464` 빌드: smoke/production/Runtime targets 성공. Runtime unit 697개 중695 pass,
  optional benchmark/NCCL2 skip. 새 resident decode7개가 포함된다.
- Python resident analyzer8개, runner29개 총37/37 통과. Source pre-commit/diff 검사를 통과했다.
- Gemma singleton continuation: 기존 `e8164e0`와 `34a8967` 각각 prompt129를 `[128,1]`로
  처리한 duplicate request3개에서 전 token output 일치. 두 binary 사이도3/3 일치했다.
  CUDA graph off, max-inflight1, zero-start correctness test이며 full12의 exact gate 대체가 아니다.
- `34a8967` Gemma/Cosmos IPC lifecycle: 첫 token 이후 decode cancel 수락, surviving request와
  readmission completion 통과. Encoder/sampling pending cancel, 실제 KV page 재사용,
  vision payload reclamation, sanitizer 검증까지 포함하는 테스트는 아니다.
- Primary48 performance는 `34a8967`이다. 이후 `af78464`의 opt-in resident 진단은 별도 campaign이며,
  무거운 JSON logging이 포함되므로 두 수치를 이어 붙여 performance 개선이라고 하지 않는다.

### 8.1 Resident shadow 실측

`resident-shadow`는 두 모델 × mixed × shared_ep/independent ×1, 4/4 실행 완료다.
분석기에서 모든 record의 `authority_applied=false`를 검사했다. 아래는 calibration을 제외한
measurement epoch1만 분석한 것이다. Coverage는 frontier 내 어느 후보의 protected decode list에도
포함되지 않은 sampling 요청을 센다. 표의 sampling 수는 **request-observation**이며 request 수나
시간 비율이 아니다. 같은 요청이 여러 decision에서 다시 관측될 수 있다.

| 모델 / mode | Decision snapshots | Resident observations | Sampling observations | 보호목록 미포함 sampling | Sampling unique IDs | Sampling age p50/p95 (service quanta) |
|---|---:|---:|---:|---:|---:|---:|
| Gemma shared | 259 | 4936 | 1420 | 1420 | 63 | 3.066 / 15.200 |
| Gemma independent | 247 | 4821 | 1643 | 1643 | 62 | 2.862 / 8.708 |
| Cosmos shared | 389 | 3085 | 27 | 27 | 26 | 8.884 / 21.702 |
| Cosmos independent | 94 | 3927 | 523 | 523 | 48 | 7.010 / 15.491 |

각 실행의 unique resident request는64개다. 모든 sampling observation에 measured reference가
있었고 candidate coverage unknown은0이었다. Cosmos 전체 observation 중 shared16개,
independent27개는 reference가 없어 unknown으로 유지됐다. 이들을 age0으로 채우지 않았다.
Warmup shadow records는 각각315/240/1095/492개 제외했다.

이 결과는 **ready queue와 protected candidate만으로 resident decode 전체를 대표하지 못한다**는
관측을 지지한다. 하지만 sampling request에 추가 D GPU time을 부과해야 한다는 결론은 아니다.
GPU에서 이미 계산된 token이 commit 대기일 수 있고, host logging이 completion visibility를 늦출 수
있다. 따라서 sampling age의 절대값을 무계측 production TPOT나 GPU stall이라고 해석하지 않는다.
다음 active 설계는 다음-token completion과 그 이후 service obligation을 구별해야 한다.

분석: `resident-shadow-analysis.json`, 원본 compressed log SHA와 analyzer SHA 포함.
Primary 성능 표는 shadow off로 고정되어 이 계측 오버헤드와 섞이지 않는다.

## 9. 최종 판정

- 구현 완료: physical-owner KV allocation/copy/accounting, 실제 입력 기반 async scratch 성장,
  policy-neutral resident 관측 및 재현 분석기.
- 검증 완료: primary48/48, 대표 before/KV-only 각8/8, resident diagnostic4/4,
  위 단위·memcheck·P1·decode lifecycle 범위.
- 미달: 두 모델 모든 workload/모든 latency 지표에서 vLLM 우위, repeat confidence interval,
  Gemma cross-mode exact output identity, resident-aware active scheduling 및 ownership 기반 admission.
- 유지: production 기본 정책/휴리스틱 값, KV capacity, 엔진, current pointers. 새 workload별 모드는 없다.

다음 최우선은 더 큰 batch나 shape별 규칙이 아니라, **independent 실행 능력을 유지하며 request별
남은 service 의무와 실제 lease feasibility를 같은 action 평가에 연결하는 것**이다. Gemma P starvation과
Cosmos resident text 지연을 함께 검증해야 한다. 한쪽 처리량 상승만으로 완료/승격하지 않는다.
