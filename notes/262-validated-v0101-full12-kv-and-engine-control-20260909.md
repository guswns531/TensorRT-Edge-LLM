# 262. v0.10.1 validated full12: KV·빌드·calibration parity와 남은 회귀

## 결론과 완료 범위

2026-09-09 기준 12 workload × V0/V1/V2 × 3회 = **108회 HTTP 측정**을 완료했다. 108회 모두 warmup HTTP/SSE 응답을 검증했다. 별도로 clean upstream v0.10.1과 current의 실제 KV page-table 구현을 각각 5회, 총 10회 비교했다. 성능 회귀를 모두 해결했다는 뜻은 아니다.

| 정책 | 기존 v0.10.0 동일 정책 대비 처리량 기하평균 | frozen vLLM 대비 | vLLM보다 높은 workload |
|---|---:|---:|---:|
| V0 Exact | -4.56% | +7.82% | 10/12 |
| V1 Scalar | -2.48% | +11.54% | 12/12 |
| V2 Scalar+Transition | -2.52% | +12.11% | 12/12 |

V2는 처리량 종합 후보, V1은 일부 latency에서 더 나은 후보다. 단일 정책이 모든 지표에서 우세하지 않다. v0.10.0은 대부분 1회 역사적 측정이고 current는 3회다. 위 차이를 버전 변경만의 인과 효과나 통계적으로 확정된 승리로 표현하지 않는다.

## 1. 이번에 수정한 것

### 빌드 조건 복구

v0.10.0의 실제 실행 파일은 Release(-O3/-DNDEBUG)였지만 비교 중인 v0.10.1 빌드는 build type이 비어 있었다. CMake가 명시적 CUDA architecture도 덮어썼다. CMake를 수정해 명시적 SM86을 유지하고 Release로 재빌드했다. 엔진은 그대로 유지했다. 이 조건을 검증하지 않은 O0 87회 결과는 보존하되 최종 비교에서 제외한다.

### calibration 입력과 성공 계약 복구

retained generic VLM trace의 80개 vision 요청이 제거된 upstream-v010 경로의 이미지를 참조했다. 기존 client는 warmup 응답 실패를 검사하지 않아 319개 제출을 성공으로 오해할 수 있었다. 당시 encoder 시작 횟수는 85가 아니라 5였다.

원본 trace는 수정하지 않고 경로를 명시적으로 remap한다. 로컬 media 존재와 SHA256을 사전 검증하고 입력 snapshot을 보존한다. 원본 이미지와 remap 이미지의 Git blob은 동일하다. guarded client가 warmup과 measurement 모두의 HTTP200, SSE 오류 없음, 응답 개수를 검사한다. 수정 후 319/319 성공, encoder 시작/완료 85를 확인했다. 첫 Release suite도 이 calibration 결함이 있어 최종 비교에서 제외한다.

### 이전 pair eligibility 선택 가능

PhaseThreeCoordinator에 preserveLegacyPairEligibility를 opt-in으로 추가했다. external residual pair와 E/P exclusive 조건의 이전 후보 범위를 재현한다. workspace 안전 조건을 완화하지 않는다. 환경변수 TRT_EDGELLM_LEGACY_PAIR_ELIGIBILITY는 존재 여부를 검사하므로 0을 넣는 것이 OFF가 아니다. OFF는 unset이다. 최종 비교에서는 ON이다.

### 통계 집계 수정

기존 outer harness는 이름이 mean_of_run_means인 필드까지 median으로 집계했다. 새 report는 개별 run으로부터 mean은 산술평균으로 다시 계산한다. 처리량과 run p95는 run 사이 median이다. 이 때문에 평균이 median-of-run-p95보다 커질 수 있다. 별도 request-distributions 표는 모든 반복의 request CSV를 합쳐 진짜 pooled mean/p95를 계산한다.

frozen vLLM vision-heavy는 원시 CSV가 3회 중 2회만 남아 pooled 표에서 제외했다. aggregate 표에는 보존된 3회 요약을 사용한다. missing 데이터를 보간하거나 2회를 3회로 표시하지 않는다.

## 2. 실행 계약과 재현 위치

- 모델: Cosmos-Reason2-2B FP16, FP16 KV, 양자화 없음.
- independent E/P/D contexts, slots80, KV256 pages, P8/D64/E4.
- text chunk128, vision prefill profile1024, client concurrency64, ignoreEOS.
- decode sampling synchronization ON, 실제 CUDA graph OFF. 과거 retained 비교 조건에 맞춘 것이며 graph 최적화 ceiling 측정은 아니다.
- generic text239 / VLM319 warmup, 매 run fresh process. 같은 calibration 요청이 동일 posterior를 보장하지는 않는다.
- KV reservation mode는 full. growth_owners=0으로 headroom per-step growth 정책 비교가 아니다.
- GPU RTX3080 10GiB, driver610.43.02, TensorRT26.06 container.
- current runtime SHA256: 6c02172bcc0e14caa71c4f93d84b5b0f77d7d1ffd2f94431a0cfc0f87ead951f.
- plugin SHA256: 2a31e79d0d729202a005ccf08ad457bae9808ebe3d0b53e28505fb5322661566.
- text engine SHA256: 084d039248e08dc192a57ebf38d521ee1fdca2f037a865f55ac0b0b904214b0b.
- vision engine SHA256: 4ede4ddfcea4dadf8d5c507cf1c99bc6c5e3a36197ba4ba2a53e11f72434e2c4.

결과 루트: .local/results/v0101-forward-port/validated-parity-20260909/ (저장소 root 기준).
manifest.json, full12/commands.json, full12/input-contract.json, 개별 warmup-validation.json, completion.json에 실행 계약과 완료 증거가 있다. report/summary와 report/request-distributions의 JSON/CSV/Markdown을 보존한다.

## 3. 코드 아키텍처와 KV 차이

| 층 | v0.10.0 current → v0.10.1 current | 해석 |
|---|---|---|
| Scalar/formation | phaseContextualPdModel, phaseRuntimeCostTracker, phaseFormationPlanner 동일 | 이번에 새 RLS 정책을 만든 것이 아님 |
| stable ownership | stableKVPageManager, phaseKVActiveView, phaseMemoryBroker 동일 | stable lease/undercommit 구조 유지 |
| underlying pool | layer당 [2,pages,128,Hkv,D] 동일 | FP16 KV layout/용량 변경 아님 |
| tensor view | slot-shaped non-owning alias 제거, pool-shaped owner 사용 | alias 제거가 큰 GPU allocation 절감/증가를 뜻하지 않음 |
| upstream page table | clean v0.10.0/v0.10.1 kvPageTable.cpp 동일 | upstream 버전 변화에 의한 page-table 변경이라는 가설 지지 안 됨 |
| current table staging | pinned 3-slot ring, event query, ring 소진 때 host wait | clean의 단일 staging/event sync와 비교할 대상 |
| XQA | build-time NVRTC serialized cubin 경로 | per-request JIT라고 해석하면 안 됨 |
| E engine | builder/tactic/workspace 차이 남음 | KV와 독립적인 메모리·실행시간 원인 후보 |

clean/current 모두 dirty-row table upload와 metadata compaction을 한다. '우리만 KV payload copy가 없다'고 주장하지 않는다. current는 stable ownership, phase-local P/D metadata, undercommit admission을 추가한다. pool은 init 시 고정 할당하며 동적으로 물리 GPU pool을 늘리고 줄인 실험이 아니다.

FP16, 28 layers, Hkv8, D128, 256 pages ×128 tokens:
**KV = 3,758,096,384 bytes = 3584 MiB**, 과거/현재 동일하다.

80 slots ×2048 tokens 전량을 보장하면 1280 pages, 순수 KV만 17920 MiB다. clean upstream은 undercommit을 금지하므로 이 GPU에서 동일 slots80/KV256의 clean 전체 서버 실행은 불가능하다. 아래 page-table microbench를 clean 전체 서버 E2E 결과로 대체해서 해석하지 않는다.

frozen vLLM equal-cap은 block16 ×2048 blocks, current는 page128 ×256으로 모두 32768 tokens/동일 FP16 KV bytes다. page granularity는 다르지만 이번 v0.10.1 변경으로 생긴 차이는 아니다.

### 실제 page-table 단독 5회 비교

100 warmup + 1000 updates/run, 16 uploads마다 stream synchronization, 마지막 GPU K/V page ID 검증. 표는 5개 run median과 p95의 각각 median이다. FP16 KV payload나 attention GPU 시간은 포함하지 않는다. BS1 one/all은 동일 workload를 별도 실행한 두 조건이다.

| BS | dirty | clean median/p95 µs | current median/p95 µs |
|---:|---|---:|---:|
| 1 | none | 0.018/0.019 | 0.018/0.019 |
| 1 | one | 3.431/3.696 | 1.836/2.017 |
| 1 | all | 3.365/3.657 | 1.829/1.984 |
| 8 | none | 0.020/0.021 | 0.019/0.020 |
| 8 | one | 3.498/3.694 | 1.929/1.981 |
| 8 | all | 3.488/3.762 | 2.200/2.335 |
| 32 | none | 0.026/0.027 | 0.023/0.026 |
| 32 | one | 3.444/3.680 | 2.212/2.268 |
| 32 | all | 3.861/4.133 | 3.375/3.599 |
| 64 | none | 0.034/0.036 | 0.028/0.031 |
| 64 | one | 3.399/3.677 | 2.621/2.726 |
| 64 | all | 4.804/5.182 | 4.786/5.149 |

작은 dirty update에서는 current host 시간이 낮지만 BS64 all은 거의 같다. 이것은 metadata 준비 비용 차이이며 전체 KV bandwidth 또는 E2E 이득으로 일반화하지 않는다.

별도 active-view BS64 stable 준비는 persistent binding OFF 12.466/12.755µs, ON 7.667/7.920µs였다. 두 조건 table upload는 8192 bytes로 동일하다. 줄어든 것은 host binding 비용이지 KV bytes가 아니다. O0에서는 OFF 49.303µs였으므로 build type confound가 host hot path에서 실제로 컸다.

## 4. 메모리 분석

| 항목 | 과거 | 현재 | 차이 |
|---|---:|---:|---:|
| KV pool | 3584 MiB | 3584 MiB | 0 |
| P workspace | 301993472 bytes | 동일 | 0 |
| D workspace | 21548544 bytes | 동일 | 0 |
| E workspace | 444873728 bytes | 574899200 bytes | +124.002 MiB |
| balanced ready/peak 기록 | 9237 MiB | 9399 MiB | +162 MiB |

나머지 약38 MiB는 산술 잔여분이며 특정 allocator 원인으로 확정하지 않았다. 새 exact-GELU와 새 tanh-GELU 비교의 약120 MiB는 위 old→new124 MiB와 다른 기준이다. 이전 exact-GELU 엔진도 더 작은 workspace를 사용하므로 exact GELU 자체가 반드시 이 비용을 요구한다고 할 수 없다.

full12의 run-median peak 기록은 최대9725 MiB다. 10240 MiB 기준 약515 MiB이지만 이것은 run-median이고 순간 peak 전부의 최소 headroom 보장이 아니다. KV를 축소하지 않았다. engine-only control은 아래 별도 기록한다.

## 5. latency: 처리량 우세와 구분

pooled CSV가 완전한 11 workload에서 vLLM 대비 기하평균 latency 비율:

| 정책 | TTFT mean/p95 변화 | TPOT mean/p95 변화 | E2E mean/p95 변화 |
|---|---:|---:|---:|
| V1 | -23.95% / -17.79% | -14.65% / -17.98% | -12.96% / -9.53% |
| V2 | -22.80% / -17.72% | -15.32% / -18.56% | -12.96% / -9.55% |

V1/V2 E2E mean은 이 11개에서 모두 낮다. E2E p95는 V1 9/11, V2 10/11이다. TTFT mean은 둘 다 7/11, TPOT mean은 8/11이다. 따라서 '모든 latency 승리'라고 할 수 없다.

기존 동일 정책 대비 12 workload pooled E2E mean/p95 기하평균은 V1 +2.98/+5.82%, V2 +2.53/+5.86%다. throughput 평균만 보고 tail parity gate를 통과했다고 할 수 없다.

client send 기준 latency와 scheduled arrival 기준 latency를 분리한다. client64 제한으로 send 전 대기가 생길 수 있다. raw throughput은 open-loop 무제한 SLO capacity와 같지 않다. 부록 pooled 표의 arrival TTFT/E2E도 확인해야 한다.

## 6. 반복 분산과 correctness

36 current cells 모두 각각 3회 token trace hash가 동일했다. 이는 동일 정책 반복의 재현성 증거이며 cross-engine exact identity, sanitizer 또는 모든 cache lifetime 안전을 새로 증명한 것은 아니다.

multi-image V0 처리량은 238.20–275.59, V1은 240.26–308.80 token/s로 변동했다. V2는 297.07–305.97이었다. 단일 smoke에서 V0가 높고 learned 정책이 낮았던 결과를 '학습 때문에 항상 나빠짐'으로 일반화하지 않는다. 반복에서 V0도 느린 상태, V1도 빠른 상태가 나왔다. encoder/ready boundary와 D cohort, host timing을 함께 봐야 한다.

long-prefill은 V1이 과거보다 개선(1220.36→1296.01), balanced는 거의 동일(4455.01→4455.74)이다. 반면 mixed/text-heavy/vision-heavy와 일부 tail에는 회귀가 남는다. 하나의 KV 원인으로 모든 workload를 설명할 수 없다.

## 7. 검증과 남은 순서

- Release phase unit tests 322개 통과, 별도 metadata benchmark 통과.
- replay contract Python tests 6개 통과; Python compile, git diff --check 통과.
- 모델/ONNX를 변경하거나 새 engine을 export/build하지 않았다. 기존 엔진의 inference 비교다.
- full pre-commit, sanitizer, fresh old full-engine suite는 이번 완료 범위가 아니다.
- 예전 full text engine은 삭제되어 현재 fresh old-full-engine ABBA 비교가 불가능하다. retained 값을 fresh라고 표시하지 않는다.
- 먼저 아래 old vision engine-only 반복으로 E workspace/실행시간 원인을 분리한다.
- 다음은 mixed/text-heavy/vision-heavy의 E→P→D ready boundary와 D GPU service/host gap을 같은 입력으로 분해한다.
- per-case throughput 3% 및 TTFT/TPOT/E2E tail gate를 다시 적용한다. aggregate -2.5%가 이 gate 전체 통과는 아니다.
- graph/sampling sync 최적화는 별도 계약의 후속 실험으로 하고 기존 frozen baseline과 혼합하지 않는다.

## 8. 이전 vision engine만 교체한 추가 통제 실험

호환성 probe 1회 후 multi-image V0/V1/V2 각 3회, 총 9회 완료. text engine, runtime binary, calibration 요청, KV와 정책 설정은 유지하고 E engine만 변경했다. 새 generic calibration으로 인한 posterior 차이도 엔진 변경의 downstream 효과에 포함되므로 GPU kernel-only 실험은 아니다. 순차 실행이며 ABBA가 아니다.

| 정책 | 새 E token/s | 이전 E token/s | 변화 | 이전 E TTFT/TPOT/E2E mean ms | 새/이전 E peak MiB (run median) |
|---|---:|---:|---:|---:|---:|
| V0 | 238.51 | 321.52 | +34.80% | 250.97/7.84/494.11 | 9571/9459 |
| V1 | 308.13 | 321.51 | +4.34% | 260.98/7.66/498.43 | 9577/9465 |
| V2 | 299.33 | 304.41 | +1.69% | 242.01/8.80/514.70 | 9593/9453 |

| 이전 E 정책 | pooled TTFT mean/p95 ms | pooled TPOT mean/p95 ms | pooled E2E mean/p95 ms |
|---|---:|---:|---:|
| V0 | 250.97/293.55 | 7.84/9.26 | 494.11/505.52 |
| V1 | 260.98/305.43 | 7.66/8.79 | 498.43/518.12 |
| V2 | 242.01/345.80 | 8.80/12.24 | 514.70/556.55 |


**출력 동일성 gate 미통과이므로 이전 E 엔진을 기본값으로 승격하지 않았다.** V0/V1 각각 첫 반복은 current와 5/5 request identity였지만 후속 반복에서는 4/5였다. request 2의 giant panda 응답은 첫 EOS 이전 동일하며, 첫 차이는 0-based token14이다. ignoreEOS=true 고정32-token trace에서 EOS 이후 계속 생성한 영역의 차이다. 정상 EOS serving에서 같은 영향을 주는지 별도 검증해야 하며, 이를 ownership corruption 또는 harmless numerical difference 중 하나로 단정하지 않는다.

Raw: old-vision-control/commands.json 및 각 run request CSV/summary/warmup-validation.json. 기존 E engine: .local/atomic-packed-vision-runtime-20260824/direct-visual-max2048/visual/visual.engine.

이전 E engine SHA256: ea090592762a96d2dc82f10c167b60f4f85fa6ccc4810cb362cef22556fcf99d.

해석: 메모리 절감과 multi-image 처리량 회복이 동일 current 코드에서 관측되어 E engine/builder 경로가 유력한 잔여 원인이다. KV 알고리즘을 되돌리지 않고도 개선된다. 그러나 5-request 소규모 workload이며 모든 VLM workload의 회복이나 cross-engine exact gate를 증명하지 않는다. 다음은 이전 builder/tactic/workspace 계약을 새 engine 생성에 재현하고 EOS-aware 및 ignoreEOS exact 검증을 모두 수행하는 것이다.

전체 완료 범위: 주 실험108회 + E 호환성1회 + E-only control9회 + KV microbench10회. 주 실험과 control 결과는 합산 평균하지 않는다.


## 부록 A. 모든 84개 aggregate 비교와 정책별 변화

아래는 report/summary.md의 전체 내용이다. p95는 반복별 p95의 median이며 pooled p95는 부록 B다.

### Legacy parity comparison

Missing cells: 0

Old and vLLM are retained measurements; current is fresh. Not a paired KV-only experiment.

Latency means are recomputed as arithmetic means from individual runs when available; the historical outer harness used medians even for fields named mean. p95 and throughput are medians across runs, not pooled p95.

| Workload | Variant | n | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|---:|
| balanced | current V0 | 3 | 4478.86 | 62.93/164.87 | 12.43/13.64 | 1123.29/1736.30 |
| balanced | current V1 | 3 | 4455.74 | 63.77/170.84 | 12.36/14.02 | 1115.43/1756.43 |
| balanced | current V2 | 3 | 4483.84 | 64.58/170.12 | 12.34/13.65 | 1115.00/1718.85 |
| balanced | frozen vLLM | 1 | 4318.34 | 110.37/247.20 | 12.23/13.71 | 1154.26/1771.51 |
| balanced | old V0 | 1 | 4562.71 | 66.38/166.95 | 12.07/13.48 | 1095.71/1702.06 |
| balanced | old V1 | 1 | 4455.01 | 67.22/170.24 | 12.34/13.87 | 1118.36/1739.66 |
| balanced | old V2 | 1 | 4454.09 | 66.62/168.90 | 12.36/14.06 | 1117.87/1745.87 |
| bimodal | current V0 | 3 | 1866.86 | 1955.93/4250.99 | 18.67/29.56 | 4518.55/9691.72 |
| bimodal | current V1 | 3 | 1888.32 | 1920.04/4454.60 | 18.08/26.21 | 4407.73/9662.13 |
| bimodal | current V2 | 3 | 1947.09 | 1895.00/4261.30 | 17.84/28.16 | 4318.21/9203.96 |
| bimodal | frozen vLLM | 1 | 1852.31 | 1573.46/2609.84 | 22.46/36.31 | 4689.19/9283.93 |
| bimodal | old V0 | 1 | 1887.97 | 1920.80/4219.12 | 18.78/29.41 | 4466.36/9312.81 |
| bimodal | old V1 | 1 | 1916.03 | 1950.27/4109.43 | 18.07/27.03 | 4420.64/9139.12 |
| bimodal | old V2 | 1 | 1952.58 | 1889.41/3983.02 | 18.00/28.91 | 4319.67/8955.61 |
| decode-heavy | current V0 | 3 | 5288.30 | 61.98/180.00 | 10.55/11.01 | 2789.16/4229.44 |
| decode-heavy | current V1 | 3 | 5228.98 | 65.96/184.51 | 10.61/11.19 | 2804.37/4278.75 |
| decode-heavy | current V2 | 3 | 5220.10 | 62.71/178.91 | 10.68/11.29 | 2819.79/4291.11 |
| decode-heavy | frozen vLLM | 1 | 4943.41 | 113.96/310.66 | 11.03/11.60 | 2969.50/4486.02 |
| decode-heavy | old V0 | 1 | 5347.89 | 64.41/174.86 | 10.39/10.87 | 2751.84/4188.84 |
| decode-heavy | old V1 | 1 | 5319.19 | 63.59/174.46 | 10.46/11.11 | 2762.53/4223.61 |
| decode-heavy | old V2 | 1 | 5265.93 | 67.02/185.83 | 10.54/11.38 | 2787.79/4250.94 |
| late-vision | current V0 | 3 | 2458.34 | 119.67/427.57 | 9.58/9.63 | 1492.44/1875.92 |
| late-vision | current V1 | 3 | 2481.74 | 127.07/444.57 | 9.49/9.53 | 1486.47/1858.01 |
| late-vision | current V2 | 3 | 2480.39 | 128.78/449.64 | 9.48/9.55 | 1487.27/1857.16 |
| late-vision | frozen vLLM | 3 | 2359.23 | 153.40/631.51 | 9.90/9.93 | 1576.03/1954.46 |
| late-vision | old V0 | 1 | 2541.43 | 108.94/417.45 | 9.26/9.31 | 1435.60/1813.83 |
| late-vision | old V1 | 1 | 2552.25 | 120.51/425.48 | 9.20/9.25 | 1438.26/1803.93 |
| late-vision | old V2 | 1 | 2565.78 | 119.99/442.22 | 9.19/9.26 | 1436.44/1797.38 |
| long-prefill | current V0 | 3 | 1054.32 | 2271.96/3193.26 | 30.85/36.64 | 4905.05/6983.79 |
| long-prefill | current V1 | 3 | 1296.01 | 1953.69/2516.40 | 23.87/27.58 | 3979.69/5551.72 |
| long-prefill | current V2 | 3 | 1281.68 | 1975.12/2604.92 | 24.31/28.89 | 4036.01/5785.09 |
| long-prefill | frozen vLLM | 1 | 1127.42 | 1925.36/3027.37 | 31.92/36.89 | 4643.91/6579.62 |
| long-prefill | old V0 | 1 | 1086.89 | 2215.91/2913.49 | 30.16/36.72 | 4788.21/7024.97 |
| long-prefill | old V1 | 1 | 1220.36 | 2059.20/2645.14 | 25.77/31.91 | 4249.51/6105.34 |
| long-prefill | old V2 | 1 | 1227.47 | 2026.92/2790.71 | 25.73/30.43 | 4211.67/6063.90 |
| mixed | current V0 | 3 | 1108.19 | 645.88/2127.83 | 38.70/56.07 | 2396.09/2602.99 |
| mixed | current V1 | 3 | 1094.38 | 739.86/2168.80 | 34.38/52.47 | 2344.74/2609.23 |
| mixed | current V2 | 3 | 1104.56 | 703.18/2201.65 | 35.54/49.67 | 2353.70/2580.52 |
| mixed | frozen vLLM | 3 | 921.48 | 874.56/2541.43 | 46.97/84.02 | 3008.38/3140.85 |
| mixed | old V0 | 1 | 1125.01 | 644.12/2107.82 | 37.83/56.14 | 2357.59/2540.51 |
| mixed | old V1 | 1 | 1185.12 | 692.51/1949.30 | 30.39/39.54 | 2123.84/2404.71 |
| mixed | old V2 | 1 | 1168.87 | 696.49/2052.37 | 37.75/59.44 | 2378.39/2493.71 |
| multi-image | current V0 | 3 | 238.51 | 345.97/462.11 | 8.56/11.20 | 611.40/670.66 |
| multi-image | current V1 | 3 | 308.13 | 291.20/306.81 | 8.46/11.02 | 553.53/518.58 |
| multi-image | current V2 | 3 | 299.33 | 285.59/322.14 | 7.77/9.40 | 526.59/534.38 |
| multi-image | frozen vLLM | 3 | 244.52 | 259.81/402.58 | 12.42/16.32 | 644.30/653.90 |
| multi-image | old V0 | 1 | 310.23 | 269.92/308.67 | 7.74/8.64 | 509.86/515.43 |
| multi-image | old V1 | 1 | 312.12 | 267.92/304.71 | 7.71/8.62 | 506.92/512.37 |
| multi-image | old V2 | 1 | 324.84 | 258.23/284.15 | 7.37/7.88 | 486.72/492.09 |
| poisson | current V0 | 3 | 1968.21 | 195.58/674.09 | 22.03/41.01 | 1598.95/2069.23 |
| poisson | current V1 | 3 | 1913.90 | 214.62/773.39 | 21.61/39.67 | 1591.88/2048.47 |
| poisson | current V2 | 3 | 1955.88 | 241.91/845.99 | 21.04/39.69 | 1591.15/2034.22 |
| poisson | frozen vLLM | 3 | 1800.07 | 438.11/902.68 | 22.19/45.67 | 1800.22/2266.55 |
| poisson | old V0 | 1 | 2038.23 | 179.58/671.38 | 21.31/40.07 | 1534.23/1991.72 |
| poisson | old V1 | 1 | 2065.85 | 228.08/746.43 | 19.49/37.07 | 1475.47/1917.67 |
| poisson | old V2 | 1 | 1974.04 | 256.87/817.44 | 20.31/35.77 | 1563.87/1987.70 |
| short | current V0 | 3 | 2469.70 | 88.12/172.07 | 13.54/27.18 | 334.06/415.73 |
| short | current V1 | 3 | 2436.30 | 89.07/181.12 | 13.42/26.27 | 332.93/413.13 |
| short | current V2 | 3 | 2469.30 | 89.51/177.73 | 13.43/26.36 | 333.59/412.61 |
| short | frozen vLLM | 3 | 1983.53 | 174.92/263.97 | 13.36/24.88 | 426.71/503.71 |
| short | old V0 | 1 | 2514.51 | 83.02/169.91 | 13.62/26.81 | 328.62/407.37 |
| short | old V1 | 1 | 2478.95 | 87.64/174.85 | 13.67/26.85 | 334.31/413.56 |
| short | old V2 | 1 | 2495.52 | 87.14/176.59 | 13.37/26.05 | 328.43/408.37 |
| text-heavy | current V0 | 3 | 1985.04 | 269.83/1090.62 | 25.58/39.20 | 1591.63/1698.86 |
| text-heavy | current V1 | 3 | 1970.43 | 342.26/1108.72 | 24.58/36.11 | 1612.53/1699.68 |
| text-heavy | current V2 | 3 | 2004.88 | 358.68/1133.31 | 23.88/35.10 | 1602.59/1666.51 |
| text-heavy | frozen vLLM | 3 | 1634.76 | 421.58/1232.20 | 29.21/47.36 | 1943.42/2037.81 |
| text-heavy | old V0 | 1 | 2108.44 | 382.16/1011.91 | 21.17/34.36 | 1489.29/1598.46 |
| text-heavy | old V1 | 1 | 2131.93 | 368.38/1047.42 | 21.08/33.30 | 1471.99/1572.85 |
| text-heavy | old V2 | 1 | 2113.64 | 322.52/998.21 | 22.75/33.69 | 1488.18/1588.64 |
| vision-heavy | current V0 | 3 | 663.41 | 1304.81/3284.49 | 36.61/59.42 | 2755.30/3649.12 |
| vision-heavy | current V1 | 3 | 665.22 | 1397.07/3166.65 | 36.86/53.75 | 2863.45/3620.04 |
| vision-heavy | current V2 | 3 | 668.70 | 1461.46/3221.29 | 42.62/77.25 | 3131.12/3523.97 |
| vision-heavy | frozen vLLM | 3 | 579.20 | 1710.70/3691.37 | 63.70/119.58 | 4119.14/4229.37 |
| vision-heavy | old V0 | 1 | 705.16 | 1236.24/2819.14 | 35.29/57.30 | 2635.67/3249.46 |
| vision-heavy | old V1 | 1 | 687.95 | 1334.86/2890.15 | 41.45/65.29 | 2955.64/3402.72 |
| vision-heavy | old V2 | 1 | 736.80 | 1383.75/3036.91 | 43.03/82.37 | 3052.22/3318.75 |
| wave-drain | current V0 | 3 | 97.73 | 267.75/311.58 | 8.31/9.35 | 525.26/525.42 |
| wave-drain | current V1 | 3 | 97.62 | 267.15/320.32 | 8.18/11.66 | 520.60/532.44 |
| wave-drain | current V2 | 3 | 97.63 | 292.26/448.99 | 8.25/11.45 | 548.13/658.55 |
| wave-drain | frozen vLLM | 3 | 95.85 | 252.76/418.60 | 12.43/17.26 | 637.86/649.42 |
| wave-drain | old V0 | 1 | 98.08 | 243.62/293.19 | 7.92/9.12 | 489.26/501.12 |
| wave-drain | old V1 | 1 | 97.97 | 243.32/297.12 | 8.01/11.56 | 491.66/504.84 |
| wave-drain | old V2 | 1 | 97.96 | 257.94/303.75 | 7.70/9.52 | 496.70/513.07 |

## Within-version policy effect

Historical comparisons are not causal paired experiments.

| Workload | Version | Policy | Throughput vs V0 |
|---|---|---|---:|
| balanced | old | V1 | -2.36% |
| balanced | old | V2 | -2.38% |
| balanced | current | V1 | -0.52% |
| balanced | current | V2 | +0.11% |
| bimodal | old | V1 | +1.49% |
| bimodal | old | V2 | +3.42% |
| bimodal | current | V1 | +1.15% |
| bimodal | current | V2 | +4.30% |
| decode-heavy | old | V1 | -0.54% |
| decode-heavy | old | V2 | -1.53% |
| decode-heavy | current | V1 | -1.12% |
| decode-heavy | current | V2 | -1.29% |
| late-vision | old | V1 | +0.43% |
| late-vision | old | V2 | +0.96% |
| late-vision | current | V1 | +0.95% |
| late-vision | current | V2 | +0.90% |
| long-prefill | old | V1 | +12.28% |
| long-prefill | old | V2 | +12.93% |
| long-prefill | current | V1 | +22.92% |
| long-prefill | current | V2 | +21.56% |
| mixed | old | V1 | +5.34% |
| mixed | old | V2 | +3.90% |
| mixed | current | V1 | -1.25% |
| mixed | current | V2 | -0.33% |
| multi-image | old | V1 | +0.61% |
| multi-image | old | V2 | +4.71% |
| multi-image | current | V1 | +29.19% |
| multi-image | current | V2 | +25.50% |
| poisson | old | V1 | +1.36% |
| poisson | old | V2 | -3.15% |
| poisson | current | V1 | -2.76% |
| poisson | current | V2 | -0.63% |
| short | old | V1 | -1.41% |
| short | old | V2 | -0.76% |
| short | current | V1 | -1.35% |
| short | current | V2 | -0.02% |
| text-heavy | old | V1 | +1.11% |
| text-heavy | old | V2 | +0.25% |
| text-heavy | current | V1 | -0.74% |
| text-heavy | current | V2 | +1.00% |
| vision-heavy | old | V1 | -2.44% |
| vision-heavy | old | V2 | +4.49% |
| vision-heavy | current | V1 | +0.27% |
| vision-heavy | current | V2 | +0.80% |
| wave-drain | old | V1 | -0.11% |
| wave-drain | old | V2 | -0.12% |
| wave-drain | current | V1 | -0.11% |
| wave-drain | current | V2 | -0.10% |

## Observed GPU memory

Sampled process/device peak, not allocator accounting. Missing frozen values are not inferred.

| Workload | Variant | Peak MiB |
|---|---|---:|
| balanced | current V0 | 9399 |
| balanced | current V1 | 9399 |
| balanced | current V2 | 9399 |
| balanced | frozen vLLM | 8959 |
| balanced | old V0 | 9237 |
| balanced | old V1 | 9237 |
| balanced | old V2 | 9237 |
| bimodal | current V0 | 9399 |
| bimodal | current V1 | 9399 |
| bimodal | current V2 | 9399 |
| bimodal | frozen vLLM | 9365 |
| bimodal | old V0 | 9237 |
| bimodal | old V1 | 9237 |
| bimodal | old V2 | 9237 |
| decode-heavy | current V0 | 9399 |
| decode-heavy | current V1 | 9399 |
| decode-heavy | current V2 | 9399 |
| decode-heavy | frozen vLLM | 8967 |
| decode-heavy | old V0 | 9237 |
| decode-heavy | old V1 | 9237 |
| decode-heavy | old V2 | 9237 |
| late-vision | current V0 | 9603 |
| late-vision | current V1 | 9575 |
| late-vision | current V2 | 9617 |
| late-vision | frozen vLLM | 9071 |
| late-vision | old V0 | 9387 |
| late-vision | old V1 | 9445 |
| late-vision | old V2 | 9387 |
| long-prefill | current V0 | 9399 |
| long-prefill | current V1 | 9399 |
| long-prefill | current V2 | 9399 |
| long-prefill | frozen vLLM | 9245 |
| long-prefill | old V0 | 9237 |
| long-prefill | old V1 | 9237 |
| long-prefill | old V2 | 9237 |
| mixed | current V0 | 9583 |
| mixed | current V1 | 9599 |
| mixed | current V2 | 9601 |
| mixed | frozen vLLM | 9799 |
| mixed | old V0 | 9395 |
| mixed | old V1 | 9483 |
| mixed | old V2 | 9395 |
| multi-image | current V0 | 9571 |
| multi-image | current V1 | 9577 |
| multi-image | current V2 | 9593 |
| multi-image | frozen vLLM | 9035 |
| multi-image | old V0 | 9379 |
| multi-image | old V1 | 9433 |
| multi-image | old V2 | 9395 |
| poisson | current V0 | 9579 |
| poisson | current V1 | 9577 |
| poisson | current V2 | 9573 |
| poisson | frozen vLLM | 9497 |
| poisson | old V0 | 9401 |
| poisson | old V1 | 9387 |
| poisson | old V2 | 9379 |
| short | current V0 | 9399 |
| short | current V1 | 9399 |
| short | current V2 | 9399 |
| short | frozen vLLM | 9035 |
| short | old V0 | 9237 |
| short | old V1 | 9237 |
| short | old V2 | 9237 |
| text-heavy | current V0 | 9569 |
| text-heavy | current V1 | 9601 |
| text-heavy | current V2 | 9659 |
| text-heavy | frozen vLLM | 9791 |
| text-heavy | old V0 | 9411 |
| text-heavy | old V1 | 9411 |
| text-heavy | old V2 | 9399 |
| vision-heavy | current V0 | 9587 |
| vision-heavy | current V1 | 9623 |
| vision-heavy | current V2 | 9725 |
| vision-heavy | old V0 | 9427 |
| vision-heavy | old V1 | 9547 |
| vision-heavy | old V2 | 9665 |
| wave-drain | current V0 | 9603 |
| wave-drain | current V1 | 9571 |
| wave-drain | current V2 | 9585 |
| wave-drain | frozen vLLM | 9619 |
| wave-drain | old V0 | 9395 |
| wave-drain | old V1 | 9405 |
| wave-drain | old V2 | 9389 |

## Output fidelity

Within-current identity below is across observed repeats, not proof of semantic correctness.

| Workload | Policy | Current repeat hashes equal | Old/current hash sets equal |
|---|---|---|---|
| balanced | V0 | True | False |
| balanced | V1 | True | False |
| balanced | V2 | True | False |
| bimodal | V0 | True | False |
| bimodal | V1 | True | False |
| bimodal | V2 | True | False |
| decode-heavy | V0 | True | False |
| decode-heavy | V1 | True | False |
| decode-heavy | V2 | True | False |
| late-vision | V0 | True | False |
| late-vision | V1 | True | False |
| late-vision | V2 | True | False |
| long-prefill | V0 | True | False |
| long-prefill | V1 | True | False |
| long-prefill | V2 | True | False |
| mixed | V0 | True | False |
| mixed | V1 | True | False |
| mixed | V2 | True | False |
| multi-image | V0 | True | True |
| multi-image | V1 | True | True |
| multi-image | V2 | True | False |
| poisson | V0 | True | False |
| poisson | V1 | True | False |
| poisson | V2 | True | False |
| short | V0 | True | False |
| short | V1 | True | False |
| short | V2 | True | False |
| text-heavy | V0 | True | False |
| text-heavy | V1 | True | False |
| text-heavy | V2 | True | False |
| vision-heavy | V0 | True | False |
| vision-heavy | V1 | True | False |
| vision-heavy | V2 | True | False |
| wave-drain | V0 | True | False |
| wave-drain | V1 | True | False |
| wave-drain | V2 | True | False |

## Per-policy throughput summary

Geometric mean of workload ratios; partial rows are explicitly counted. No confidence interval inferred.

| Policy | Matched workloads | vs old same policy | vs frozen vLLM | Wins vs vLLM |
|---|---:|---:|---:|---:|
| V0 | 12/12 | -4.56% | +7.82% | 10 |
| V1 | 12/12 | -2.48% | +11.54% | 12 |
| V2 | 12/12 | -2.52% | +12.11% | 12 |

## Observed repeat variation

Run minima/maxima, not confidence intervals or pooled request percentiles.

| Workload | Variant | token/s min–max | E2E run-mean min–max ms |
|---|---|---:|---:|
| balanced | current V0 | 4363.27–4510.42 | 1107.18–1146.71 |
| balanced | current V1 | 4419.88–4509.72 | 1102.59–1128.79 |
| balanced | current V2 | 4348.11–4542.21 | 1094.22–1144.99 |
| bimodal | current V0 | 1862.20–1876.08 | 4502.74–4528.64 |
| bimodal | current V1 | 1862.56–1937.68 | 4271.64–4483.62 |
| bimodal | current V2 | 1944.99–1951.40 | 4307.15–4330.21 |
| decode-heavy | current V0 | 5226.54–5301.59 | 2771.67–2817.24 |
| decode-heavy | current V1 | 5205.34–5269.91 | 2783.68–2823.58 |
| decode-heavy | current V2 | 5176.78–5232.83 | 2804.28–2840.86 |
| late-vision | current V0 | 2454.84–2461.15 | 1484.67–1497.77 |
| late-vision | current V1 | 2479.34–2484.28 | 1485.90–1487.60 |
| late-vision | current V2 | 2472.25–2483.30 | 1486.88–1487.95 |
| long-prefill | current V0 | 1048.16–1088.28 | 4785.88–4982.56 |
| long-prefill | current V1 | 1293.19–1309.77 | 3949.03–3995.85 |
| long-prefill | current V2 | 1260.85–1300.33 | 3968.03–4111.55 |
| mixed | current V0 | 1104.37–1121.15 | 2348.47–2456.35 |
| mixed | current V1 | 1068.81–1126.06 | 2199.02–2448.24 |
| mixed | current V2 | 1059.01–1132.59 | 2278.94–2488.10 |
| multi-image | current V0 | 238.20–275.59 | 563.89–635.66 |
| multi-image | current V1 | 240.26–308.80 | 513.22–633.87 |
| multi-image | current V2 | 297.07–305.97 | 516.99–532.85 |
| poisson | current V0 | 1924.48–2022.73 | 1540.88–1650.00 |
| poisson | current V1 | 1912.72–2000.75 | 1540.85–1620.41 |
| poisson | current V2 | 1927.27–1966.35 | 1558.74–1625.92 |
| short | current V0 | 2461.64–2483.44 | 332.04–335.74 |
| short | current V1 | 2422.91–2443.88 | 330.16–335.27 |
| short | current V2 | 2445.46–2470.02 | 331.57–337.28 |
| text-heavy | current V0 | 1966.88–2009.72 | 1571.55–1609.35 |
| text-heavy | current V1 | 1902.64–2026.77 | 1556.40–1680.33 |
| text-heavy | current V2 | 1894.02–2026.01 | 1557.53–1680.82 |
| vision-heavy | current V0 | 662.52–668.12 | 2705.03–2800.10 |
| vision-heavy | current V1 | 656.26–673.89 | 2631.28–3082.80 |
| vision-heavy | current V2 | 649.73–692.38 | 2910.57–3332.92 |
| wave-drain | current V0 | 97.55–97.75 | 511.06–549.56 |
| wave-drain | current V1 | 97.39–97.64 | 516.58–528.26 |
| wave-drain | current V2 | 97.39–97.67 | 521.73–571.38 |

## Throughput matrix (token/s)

| Workload | old V0 | old V1 | old V2 | current V0 | current V1 | current V2 | frozen vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | 4562.71 | 4455.01 | 4454.09 | 4478.86 | 4455.74 | 4483.84 | 4318.34 |
| bimodal | 1887.97 | 1916.03 | 1952.58 | 1866.86 | 1888.32 | 1947.09 | 1852.31 |
| decode-heavy | 5347.89 | 5319.19 | 5265.93 | 5288.30 | 5228.98 | 5220.10 | 4943.41 |
| late-vision | 2541.43 | 2552.25 | 2565.78 | 2458.34 | 2481.74 | 2480.39 | 2359.23 |
| long-prefill | 1086.89 | 1220.36 | 1227.47 | 1054.32 | 1296.01 | 1281.68 | 1127.42 |
| mixed | 1125.01 | 1185.12 | 1168.87 | 1108.19 | 1094.38 | 1104.56 | 921.48 |
| multi-image | 310.23 | 312.12 | 324.84 | 238.51 | 308.13 | 299.33 | 244.52 |
| poisson | 2038.23 | 2065.85 | 1974.04 | 1968.21 | 1913.90 | 1955.88 | 1800.07 |
| short | 2514.51 | 2478.95 | 2495.52 | 2469.70 | 2436.30 | 2469.30 | 1983.53 |
| text-heavy | 2108.44 | 2131.93 | 2113.64 | 1985.04 | 1970.43 | 2004.88 | 1634.76 |
| vision-heavy | 705.16 | 687.95 | 736.80 | 663.41 | 665.22 | 668.70 | 579.20 |
| wave-drain | 98.08 | 97.97 | 97.96 | 97.73 | 97.62 | 97.63 | 95.85 |
## 부록 B. Pooled request distributions

Mean and p95 are computed across all measured requests from all repeats. Arrival-relative values include client admission waiting; ordinary values start at send. These p95 values differ from the median of per-run p95 values. No confidence interval is inferred.

Unavailable historical rows (not estimated from partial repeats): [{"workload": "vision-heavy", "variant": "frozen vLLM", "expected_runs": 3, "available_csvs": 2}]

| Workload | Variant | Requests | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | Arrival TTFT mean/p95 | Arrival E2E mean/p95 |
|---|---|---:|---:|---:|---:|---:|---:|
| balanced | current V0 | 864 | 62.93/167.14 | 12.43/14.08 | 1123.29/1739.19 | 1932.32/4109.97 | 2992.69/5015.83 |
| balanced | current V1 | 864 | 63.77/171.54 | 12.36/13.96 | 1115.43/1757.59 | 1895.34/4098.99 | 2947.00/4998.41 |
| balanced | current V2 | 864 | 64.58/170.31 | 12.34/14.01 | 1115.00/1750.86 | 1901.05/4069.56 | 2951.47/4996.95 |
| balanced | frozen vLLM | 288 | 110.37/247.20 | 12.23/13.71 | 1154.26/1771.51 | 2048.71/4322.57 | 3092.61/5171.24 |
| balanced | old V0 | 288 | 66.38/166.95 | 12.07/13.48 | 1095.71/1702.06 | 1900.38/4045.00 | 2929.72/4890.89 |
| balanced | old V1 | 288 | 67.22/170.24 | 12.34/13.87 | 1118.36/1739.66 | 1905.28/4122.33 | 2956.41/4986.73 |
| balanced | old V2 | 288 | 66.62/168.90 | 12.36/14.06 | 1117.87/1745.87 | 1907.31/4133.95 | 2958.56/4995.65 |
| bimodal | current V0 | 864 | 1955.93/4269.40 | 18.67/29.55 | 4518.55/9672.07 | 8198.73/18814.80 | 10761.36/21720.75 |
| bimodal | current V1 | 864 | 1920.04/4482.03 | 18.08/27.03 | 4407.73/9576.26 | 8019.73/18179.23 | 10507.42/21312.73 |
| bimodal | current V2 | 864 | 1895.00/4263.57 | 17.84/28.51 | 4318.21/9229.41 | 7881.04/17879.09 | 10304.24/20811.89 |
| bimodal | frozen vLLM | 288 | 1573.46/2609.84 | 22.46/36.31 | 4689.19/9283.93 | 9081.54/19527.67 | 12197.27/22091.78 |
| bimodal | old V0 | 288 | 1920.80/4219.12 | 18.78/29.41 | 4466.36/9312.81 | 8209.98/18472.69 | 10755.54/21412.59 |
| bimodal | old V1 | 288 | 1950.27/4109.43 | 18.07/27.03 | 4420.64/9139.12 | 8075.74/18343.98 | 10546.11/21208.65 |
| bimodal | old V2 | 288 | 1889.41/3983.02 | 18.00/28.91 | 4319.67/8955.61 | 7909.75/17990.86 | 10340.02/20643.63 |
| decode-heavy | current V0 | 864 | 61.98/181.51 | 10.55/11.16 | 2789.16/4240.07 | 4778.31/10319.05 | 7505.49/12945.39 |
| decode-heavy | current V1 | 864 | 65.96/181.27 | 10.61/11.28 | 2804.37/4289.75 | 4779.58/10412.47 | 7517.99/13071.65 |
| decode-heavy | current V2 | 864 | 62.71/178.35 | 10.68/11.32 | 2819.79/4317.20 | 4798.03/10463.77 | 7555.12/13100.02 |
| decode-heavy | frozen vLLM | 288 | 113.96/310.66 | 11.03/11.60 | 2969.50/4486.02 | 5129.99/11058.28 | 7985.52/13858.32 |
| decode-heavy | old V0 | 288 | 64.41/174.86 | 10.39/10.87 | 2751.84/4188.84 | 4729.55/10223.27 | 7416.99/12798.29 |
| decode-heavy | old V1 | 288 | 63.59/174.46 | 10.46/11.11 | 2762.53/4223.61 | 4700.03/10266.16 | 7398.97/12854.68 |
| decode-heavy | old V2 | 288 | 67.02/185.83 | 10.54/11.38 | 2787.79/4250.94 | 4732.24/10371.79 | 7453.01/12969.55 |
| late-vision | current V0 | 96 | 119.67/443.52 | 7.19/9.64 | 1492.44/1877.56 | 120.77/444.50 | 1493.55/1879.56 |
| late-vision | current V1 | 96 | 127.07/458.19 | 7.12/9.55 | 1486.47/1858.77 | 128.15/459.01 | 1487.55/1860.94 |
| late-vision | current V2 | 96 | 128.78/461.56 | 7.11/9.55 | 1487.27/1860.76 | 129.95/462.11 | 1488.45/1863.17 |
| late-vision | frozen vLLM | 96 | 156.02/634.52 | 7.43/9.94 | 1574.79/1957.60 | 157.07/634.96 | 1575.84/1960.22 |
| late-vision | old V0 | 32 | 108.94/417.45 | 6.95/9.31 | 1435.60/1813.83 | 110.06/417.56 | 1436.72/1816.06 |
| late-vision | old V1 | 32 | 120.51/425.48 | 6.90/9.24 | 1438.26/1803.93 | 121.62/426.24 | 1439.37/1805.28 |
| late-vision | old V2 | 32 | 119.99/442.22 | 6.89/9.26 | 1436.44/1797.38 | 121.09/443.13 | 1437.53/1798.84 |
| long-prefill | current V0 | 864 | 2271.96/3264.73 | 30.85/36.54 | 4905.05/7053.48 | 10163.94/20853.80 | 12797.04/22769.21 |
| long-prefill | current V1 | 864 | 1953.69/2589.15 | 23.87/28.35 | 3979.69/5562.34 | 8702.77/17054.62 | 10728.77/18445.17 |
| long-prefill | current V2 | 864 | 1975.12/2612.27 | 24.31/29.34 | 4036.01/5663.50 | 8877.84/17296.73 | 10938.73/18703.75 |
| long-prefill | frozen vLLM | 288 | 1925.36/3027.37 | 31.92/36.89 | 4643.91/6579.62 | 9934.14/19540.43 | 12652.68/21365.02 |
| long-prefill | old V0 | 288 | 2215.91/2913.49 | 30.16/36.72 | 4788.21/7024.97 | 9745.84/20288.89 | 12318.14/22182.83 |
| long-prefill | old V1 | 288 | 2059.20/2645.14 | 25.77/31.91 | 4249.51/6105.34 | 9096.72/18217.81 | 11287.02/19675.66 |
| long-prefill | old V2 | 288 | 2026.92/2790.71 | 25.73/30.43 | 4211.67/6063.90 | 9031.08/18011.40 | 11215.83/19522.64 |
| mixed | current V0 | 192 | 645.88/2103.67 | 38.70/60.28 | 2396.09/2603.49 | 645.95/2103.73 | 2396.16/2603.58 |
| mixed | current V1 | 192 | 739.86/2221.40 | 34.38/52.11 | 2344.74/2652.32 | 739.93/2221.47 | 2344.81/2652.40 |
| mixed | current V2 | 192 | 703.18/2201.22 | 35.54/52.64 | 2353.70/2692.33 | 703.28/2201.27 | 2353.80/2692.41 |
| mixed | frozen vLLM | 192 | 882.35/2542.58 | 47.10/84.64 | 3004.83/3146.73 | 882.42/2542.65 | 3004.90/3146.79 |
| mixed | old V0 | 64 | 644.12/2107.82 | 37.83/56.14 | 2357.59/2540.51 | 644.19/2107.88 | 2357.66/2540.59 |
| mixed | old V1 | 64 | 692.51/1949.30 | 30.39/39.54 | 2123.84/2404.71 | 692.58/1949.36 | 2123.91/2404.80 |
| mixed | old V2 | 64 | 696.49/2052.37 | 37.75/59.44 | 2378.39/2493.71 | 696.53/2052.43 | 2378.43/2493.74 |
| multi-image | current V0 | 15 | 345.97/462.17 | 8.56/11.23 | 611.40/671.04 | 346.43/462.61 | 611.86/671.61 |
| multi-image | current V1 | 15 | 291.20/456.91 | 8.46/12.05 | 553.53/665.53 | 291.59/457.26 | 553.92/665.84 |
| multi-image | current V2 | 15 | 285.59/325.74 | 7.77/9.95 | 526.59/538.28 | 285.96/325.97 | 526.95/538.51 |
| multi-image | frozen vLLM | 15 | 265.69/430.68 | 12.16/17.12 | 642.57/654.70 | 266.20/431.26 | 643.08/654.95 |
| multi-image | old V0 | 5 | 269.92/308.67 | 7.74/8.64 | 509.86/515.43 | 270.33/308.97 | 510.26/515.74 |
| multi-image | old V1 | 5 | 267.92/304.71 | 7.71/8.62 | 506.92/512.37 | 268.37/304.92 | 507.36/512.62 |
| multi-image | old V2 | 5 | 258.23/284.15 | 7.37/7.88 | 486.72/492.09 | 258.56/284.64 | 487.04/492.51 |
| poisson | current V0 | 192 | 195.58/688.67 | 22.03/41.35 | 1598.95/2102.57 | 195.66/688.73 | 1599.03/2102.64 |
| poisson | current V1 | 192 | 214.62/778.70 | 21.61/40.40 | 1591.88/2053.96 | 214.70/778.80 | 1591.96/2054.04 |
| poisson | current V2 | 192 | 241.91/858.95 | 21.04/39.73 | 1591.15/2058.15 | 241.99/859.03 | 1591.22/2058.21 |
| poisson | frozen vLLM | 192 | 580.94/1120.16 | 19.49/45.08 | 1810.99/2445.77 | 581.04/1120.33 | 1811.09/2445.84 |
| poisson | old V0 | 64 | 179.58/671.38 | 21.31/40.07 | 1534.23/1991.72 | 179.65/671.47 | 1534.31/1991.77 |
| poisson | old V1 | 64 | 228.08/746.43 | 19.49/37.07 | 1475.47/1917.67 | 228.16/746.50 | 1475.55/1917.75 |
| poisson | old V2 | 64 | 256.87/817.44 | 20.31/35.77 | 1563.87/1987.70 | 256.95/817.52 | 1563.95/1987.76 |
| short | current V0 | 144 | 88.12/171.84 | 13.54/29.54 | 334.06/415.03 | 88.21/171.86 | 334.15/415.12 |
| short | current V1 | 144 | 89.07/183.77 | 13.42/28.22 | 332.93/413.82 | 89.15/183.86 | 333.02/413.88 |
| short | current V2 | 144 | 89.51/178.52 | 13.43/28.62 | 333.59/414.24 | 89.59/178.69 | 333.67/414.35 |
| short | frozen vLLM | 144 | 175.07/265.62 | 13.36/26.39 | 424.98/502.78 | 175.15/265.68 | 425.06/502.92 |
| short | old V0 | 48 | 83.02/169.91 | 13.62/26.81 | 328.62/407.37 | 83.10/170.01 | 328.70/407.44 |
| short | old V1 | 48 | 87.64/174.85 | 13.67/26.85 | 334.31/413.56 | 87.72/174.93 | 334.39/413.65 |
| short | old V2 | 48 | 87.14/176.59 | 13.37/26.05 | 328.43/408.37 | 87.23/176.72 | 328.53/408.47 |
| text-heavy | current V0 | 192 | 269.83/1096.67 | 25.58/39.67 | 1591.63/1706.24 | 269.89/1096.74 | 1591.68/1706.31 |
| text-heavy | current V1 | 192 | 342.26/1114.88 | 24.58/35.99 | 1612.53/1751.40 | 342.33/1114.94 | 1612.60/1751.47 |
| text-heavy | current V2 | 192 | 358.68/1135.97 | 23.88/35.56 | 1602.59/1752.11 | 358.75/1136.04 | 1602.65/1752.19 |
| text-heavy | frozen vLLM | 192 | 427.67/1443.63 | 29.25/48.62 | 1943.68/2043.47 | 427.74/1443.69 | 1943.74/2043.53 |
| text-heavy | old V0 | 64 | 382.16/1011.91 | 21.17/34.36 | 1489.29/1598.46 | 382.23/1011.97 | 1489.36/1598.52 |
| text-heavy | old V1 | 64 | 368.38/1047.42 | 21.08/33.30 | 1471.99/1572.85 | 368.45/1047.48 | 1472.06/1572.94 |
| text-heavy | old V2 | 64 | 322.52/998.21 | 22.75/33.69 | 1488.18/1588.64 | 322.60/998.29 | 1488.25/1588.71 |
| vision-heavy | current V0 | 192 | 1304.81/3317.38 | 36.61/57.07 | 2755.30/3647.02 | 1304.88/3317.46 | 2755.37/3647.08 |
| vision-heavy | current V1 | 192 | 1397.07/3174.40 | 36.86/66.24 | 2863.45/3627.61 | 1397.15/3174.49 | 2863.53/3627.68 |
| vision-heavy | current V2 | 192 | 1461.46/3228.36 | 42.62/78.22 | 3131.12/3650.19 | 1461.53/3228.43 | 3131.19/3650.52 |
| vision-heavy | old V0 | 64 | 1236.24/2819.14 | 35.29/57.30 | 2635.67/3249.46 | 1236.31/2819.21 | 2635.74/3249.52 |
| vision-heavy | old V1 | 64 | 1334.86/2890.15 | 41.45/65.29 | 2955.64/3402.72 | 1334.93/2890.22 | 2955.71/3402.78 |
| vision-heavy | old V2 | 64 | 1383.75/3036.91 | 43.03/82.37 | 3052.22/3318.75 | 1383.82/3036.97 | 3052.29/3318.81 |
| wave-drain | current V0 | 60 | 267.75/346.74 | 8.31/13.31 | 525.26/584.94 | 267.83/346.80 | 525.34/585.02 |
| wave-drain | current V1 | 60 | 267.15/328.57 | 8.18/11.66 | 520.60/540.94 | 267.23/328.66 | 520.68/541.03 |
| wave-drain | current V2 | 60 | 292.26/454.77 | 8.25/11.93 | 548.13/665.68 | 292.34/454.86 | 548.21/665.77 |
| wave-drain | frozen vLLM | 60 | 253.06/418.84 | 12.43/17.26 | 638.52/649.42 | 253.16/418.92 | 638.61/649.51 |
| wave-drain | old V0 | 20 | 243.62/293.19 | 7.92/9.12 | 489.26/501.12 | 243.70/293.25 | 489.34/501.18 |
| wave-drain | old V1 | 20 | 243.32/297.12 | 8.01/11.56 | 491.66/504.84 | 243.40/297.20 | 491.74/504.88 |
| wave-drain | old V2 | 20 | 257.94/303.75 | 7.70/9.52 | 496.70/513.07 | 258.02/303.86 | 496.78/513.17 |
