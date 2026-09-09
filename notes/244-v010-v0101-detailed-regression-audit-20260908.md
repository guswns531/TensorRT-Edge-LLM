# v0.10 → v0.10.1: 실행 계약·지연·정책·수치 차이 재감사

> 후속 검증 주의(2026-09-09): note259에서 old Release/current 미최적화 build 차이,
> note261에서 삭제된 VLM calibration 이미지 경로와 warmup 실패 미검증을 발견했다.
> 아래 원시 수치는 보존하지만 공정한 버전/정책 인과 효과로 인용하기 전 각 run의 실제
> build flags와 warmup 성공을 확인해야 한다. 모든 과거 run이 동일 문제였다고 소급 단정하지는 않는다.
> Old/frozen vLLM 자체의 수치를 이 문제만으로 무효화하는 것도 아니다.

Date: 2026-09-08. Current source: `77c7662` (runtime changes `4099f76`). Compared source tree: root `codex/v010-phase-forward-port`, `f4eb53e`. **새 GPU 실험이나 코드 변경 없이** 보관된 request CSV, aggregate, command manifest, measurement events 및 Git diff를 분석했다.

## 1. 요약과 이전 표현의 정정

현재 문제는 단일 scheduler 실패가 아니다. **동일한 정책 모델 위에서 바뀐 compiled execution 경로**, **prefill formation 차이**, **VLM first-token/decode trade-off**, **비교 계약의 불완전한 정렬**을 구분해야 한다.

- old V1은 `.local/results/current/v1-full12` → `.local/v012-clean-full12-v1-20260907`의 **1회 engineering reference**다. 전 역사 중 각 workload 최고점을 모은 oracle이나 반복 검증된 절대 champion이라고 부르지 않는다.
- current V1은 `atomic-vp1024-final-full12-r3`의 3회; current V0/V2 및 old V0/V1/V2는 각 1회다. 작은 변화는 확정적 순위가 아니다.
- old/current V1 **12개 전부 TRT 환경 변수 목록과 값이 동일**하다. binary, plugin, engine, source 및 일부 자동 설정이 다르다. 같은 환경 변수는 같은 실행을 보장하지 않는다.
- 5개 text trace는 byte hash 동일. 7개 VLM trace는 경로 때문에 hash가 달랐다. `file://`를 실제 이미지 SHA256으로 정규화한 JSON은 **12/12 동일**했다. 옛 `.local/upstream-v010` 경로가 없으면 root의 동일 상대경로 retained asset을 사용해 비교했으며, 이미지가 바뀌었다는 증거는 없었다.
- frozen vLLM의 balanced/decode-heavy/long-prefill/bimodal은 client max-in-flight **80**, current/old phase HTTP는 **64**다. short는 vLLM 48 vs phase 64지만 총 48개라 실효 cap은 같다. VLM도 총 요청 수 또는 cap64 이내다. 따라서 앞의 4개는 **같은 offered trace, 서로 다른 client cap** 비교이며 완전한 동일 admission 계약이라고 표현하면 안 된다.
- P/D CUDA graph cache는 old V1 12개 로그와 current V1 36개 로그 모두 **captures=0, hits=0**다. “같은 graph 지원”과 “실제 replay 사용”은 다르다. 이번 회귀를 graph 제거로 설명할 수 없다.
- TTFT/E2E는 **send 기준**이다. saturation trace의 client concurrency wait를 포함하는 arrival 기준 지연도 별도로 보고한다. p95끼리 더하지 않고 request별 timestamp를 먼저 차분한 뒤 percentile을 계산했다.

## 2. 코드/아키텍처에서 실제로 달라진 것

| 층 | old v0.10 | current v0.10.1 | 판단 |
|---|---|---|---|
| Scalar RLS | phaseContextualPdModel | 파일 동일 | 회귀를 RLS 모델 변경으로 설명할 근거 없음 |
| Global selector | phaseGlobalScheduler | 파일 동일 | 동일 모델도 observation/후보가 달라지면 선택은 달라질 수 있음 |
| Stable KV | indexed-paged lease/pool | phaseKV* 파일 동일, 256 pages | pool 축소/compaction 재도입이 원인은 아님 |
| P/D queue | 동일 batching core | decode-block 플래그 추가, 나머지 주로 formatting | shared-E/D 안전장치; 실제 설정에서 활성 여부 확인 필요 |
| Residual external P+D | external lineage 평가 제외 | Scalar 평가 허용 | current만의 정책 권한 차이; old parity와 개선 실험을 분리해야 함 |
| E/P/D workspace | independent arenas | independent arenas + 선택적 E/D 공유 API | 최종 manifest는 E/D 공유를 켜지 않음 |
| Vision-P | atomic P1024 | 전용 profile 있으면 atomic, 없으면 P128 chunk | 최종 atomic 경로는 old 구조에 가까움 |
| Vision output | runner output를 통해 shape/할당 참조 | output spec, 내부 storage 해제, request-owned output 강제 | 메모리 절감이나 ownership correctness를 별도 검증해야 함 |
| Encoder formation | 기존 frontier | 최초 payload 크기 미확정이면 E1 bootstrap guard | unknown payload에서 처음 E batch를 작게 만들 수 있음; 안전 guard 삭제보다 estimator 보완 우선 |
| Executor | concrete EngineExecutor | 추상 interface + TrtEngineExecutor factory | 구조 변경은 확인, virtual dispatch 비용이 수 ms 회귀 원인이라는 증거 없음 |
| Decode attention | 사전 빌드 XQA cubin | build-time NVRTC JIT + engine cubin serialization | SM86 지원. 커널 구현/컴파일/수치 차이를 우선 점검할 대상 |
| Export/plan | 기존 ONNX/plugin 계약 | 새 checkpoint exporter 및 plugin 계약 | last_token_ids [B,1] vs packed [1,B] 차이가 이전 rebuild를 막았음 |

중요한 코드 위치:

- `cpp/runtime/scheduling/phaseThreeCoordinator.cpp`: residual Scalar gate (~2186), E/D exclusion, unknown-payload E1 guard (~3889).
- `cpp/runtime/scheduling/phaseVisionAdapter.cpp`: internal-output release (~218), spec-based allocation (~414), external binding requirement (~465).
- `cpp/runtime/scheduling/independentEngineExecutorPair.cpp`: shared-E/D API (~277); 존재 자체가 활성 사용을 뜻하지 않음.
- `cpp/runtime/exec/engineExecutor.cpp`: TrtEngineExecutor interface implementation (~76).
- `cpp/plugins/attentionPlugin/attentionPlugin.cpp`: XQA JIT build/load (~775–850).
- `cpp/kernels/decodeAttentionKernels/decoderXQAJitCompiler.cpp`: SM86 허용 및 compile contract (~250).
- `examples/llm/llm_phase_context_smoke.cpp`: graph opt-in (~1687), vision chunk selection (~1706).

`phaseServingRuntime`의 신규 production wrapper는 코드 차이에 포함되지만 이 HTTP trace는 smoke/IPC adapter를 사용한다. 호출되지 않는 wrapper의 코드량을 hot-path overhead로 세면 안 된다. Likewise NVFP4/MoE/Blackwell 변경은 Cosmos FP16/SM86 실행의 원인이라고 자동 분류하지 않는다.

## 3. old/current V0·V1·V2 처리량

단위 generated token/s. 같은 열의 old/current는 model/trace 양은 같지만 engine과 일부 generated token 내용이 다르다. 이 표는 forward-port 결과이며 policy-only A/B가 아니다.

| Workload | old V0 | new V0 | old V1 | new V1 | old V2 | new V2 |
|---|---:|---:|---:|---:|---:|---:|
| short | 2514.51 | 2363.89 | 2478.95 | 2347.45 | 2495.52 | 2250.86 |
| balanced | 4562.71 | 4143.44 | 4455.01 | 4164.17 | 4454.09 | 4032.69 |
| decode-heavy | 5347.89 | 4920.29 | 5319.19 | 4898.64 | 5265.93 | 4748.28 |
| long-prefill | 1086.89 | 1011.42 | 1220.36 | 1223.75 | 1227.47 | 1200.39 |
| bimodal | 1887.97 | 1815.40 | 1916.03 | 1869.52 | 1952.58 | 1837.88 |
| text-heavy | 2108.44 | 1906.91 | 2131.93 | 1872.49 | 2113.64 | 1910.48 |
| mixed | 1125.01 | 1145.71 | 1185.12 | 1078.53 | 1168.87 | 1137.51 |
| poisson | 2038.23 | 1790.52 | 2065.85 | 1878.52 | 1974.04 | 1768.90 |
| vision-heavy | 705.16 | 673.92 | 687.95 | 668.94 | 736.80 | 705.21 |
| wave-drain | 98.08 | 97.09 | 97.97 | 96.31 | 97.96 | 96.30 |
| late-vision | 2541.43 | 2384.85 | 2552.25 | 2399.25 | 2565.78 | 2377.51 |
| multi-image | 310.23 | 298.51 | 312.12 | 298.40 | 324.84 | 298.53 |

V1/V0의 current 기하평균 +1.45%는 long-prefill +20.99%의 영향이 크다. **long-prefill을 제외한 11개의 V1/V0 기하평균은 -0.16%**다. Scalar가 전반을 크게 가속한다고 확대 해석하면 안 된다. 동시에 long-prefill의 TTFT/TPOT/E2E 개선은 보존할 가치가 있다.

## 4. V1 latency: old → current

ms. old 1회, new는 run-level mean/p95의 median. 평균 열을 pooled all-request mean으로 오해하지 않는다.

| Workload | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| short | 87.64 → 109.05 | 174.85 → 188.46 | 13.67 → 12.92 | 26.85 → 22.27 | 334.31 → 352.18 | 413.56 → 434.29 |
| balanced | 67.22 → 71.21 | 170.24 → 174.29 | 12.34 → 13.28 | 13.87 → 15.03 | 1118.36 → 1201.16 | 1739.66 → 1882.52 |
| decode-heavy | 63.59 → 71.79 | 174.46 → 207.18 | 10.46 → 11.37 | 11.11 → 12.24 | 2762.53 → 3009.52 | 4223.61 → 4639.49 |
| long-prefill | 2059.20 → 2068.31 | 2645.14 → 2727.59 | 25.77 → 25.34 | 31.91 → 29.45 | 4249.51 → 4221.91 | 6105.34 → 5870.92 |
| bimodal | 1950.27 → 1987.13 | 4109.43 → 4537.04 | 18.07 → 18.16 | 27.03 → 26.00 | 4420.64 → 4504.74 | 9139.12 → 9628.46 |
| text-heavy | 368.38 → 314.58 | 1047.42 → 1107.21 | 21.08 → 26.65 | 33.30 → 39.82 | 1471.99 → 1687.72 | 1572.85 → 1773.05 |
| mixed | 692.51 → 714.19 | 1949.30 → 2213.64 | 30.39 → 33.16 | 39.54 → 40.55 | 2123.84 → 2273.13 | 2404.71 → 2649.30 |
| poisson | 228.08 → 218.41 | 746.43 → 764.79 | 19.49 → 22.81 | 37.07 → 41.01 | 1475.47 → 1660.85 | 1917.67 → 2125.02 |
| vision-heavy | 1334.86 → 1342.22 | 2890.15 → 3167.36 | 41.45 → 30.52 | 65.29 → 37.38 | 2955.64 → 2532.63 | 3402.72 → 3606.61 |
| wave-drain | 243.32 → 248.30 | 297.12 → 398.29 | 8.01 → 9.89 | 11.56 → 13.90 | 491.66 → 555.92 | 504.84 → 615.70 |
| late-vision | 120.51 → 130.03 | 425.48 → 493.15 | 9.20 → 9.81 | 9.25 → 9.86 | 1438.26 → 1535.31 | 1803.93 → 1922.13 |
| multi-image | 267.92 → 233.13 | 304.71 → 317.70 | 7.71 → 9.40 | 8.62 → 12.39 | 506.92 → 524.70 | 512.37 → 535.91 |

### 워크로드별 진단 (관측과 해석 구분)

1. **short**: throughput -5.30%, TTFT mean +24.4%인데 TPOT mean -5.5% / p95 -17.1%. 모든 decode kernel이 느려졌다는 설명과 맞지 않는다. first-token scheduling/launch/initial cohort를 먼저 본다.
2. **balanced**: throughput -6.53%, TPOT mean +7.6%, E2E mean +7.4%. steady serving cycle 비용 상승과 부합한다. 아래 P128→atomic 진단은 P fragmentation을 보여주지만 old→new 원인 증명과는 구분한다.
3. **decode-heavy**: throughput -7.91%, TPOT mean +8.8%, p95 +10.2%. XQA/GEMM service와 D cycle host gap을 분해할 우선 workload다. 같은 KV pool이라는 사실이 KV read kernel 성능 동일성을 뜻하지 않는다.
4. **long-prefill**: throughput +0.28% 근처, E2E mean -0.6%, p95 -3.8%. current Scalar가 old의 P-heavy 성능을 거의 회복했다. 광범위한 selector 교체로 이 성과를 잃지 말아야 한다.
5. **bimodal**: throughput -2.43%, TTFT p95 +10.4%, TPOT p95 -3.8%. 평균 decode 비용보다 long/short request ordering과 tail 분포를 봐야 한다.
6. **text-heavy**: throughput -12.17%, TTFT mean -14.6%, TPOT mean +26.4%, E2E mean +14.7%. 더 빠른 first token과 더 느린 resident decode가 공존한다. 평균 TTFT만 최적화하면 잘못된 결론이 나온다.
7. **mixed**: throughput -8.99%, TTFT mean +3.1%, E2E p95 +10.2%. 현재 V1/V2의 class별 효과도 다르다. 아래 별도 표 참조. 반복 출력 불일치도 함께 추적한다.
8. **poisson**: throughput -9.07%, TTFT mean -4.2%, TPOT mean +17.1%, E2E mean +12.6%. 신규 요청 progress를 얻고 decode continuity를 잃었을 가능성; request별 D gap 없이 원인 확정은 금지한다.
9. **vision-heavy**: throughput -2.76%지만 TPOT mean -26.4%, E2E mean -14.3%; E2E p95는 +6.0%. current가 전체적으로 나빠졌다는 단일 순위는 틀리다. average와 last drain/tail trade-off다.
10. **wave/drain**: throughput -1.70% 수준인데 TTFT p95 +34.0%, E2E p95 +22.0%. 의도된 arrival idle 때문에 총 throughput은 queue-delay 악화를 숨긴다. first E start 및 wave별 TTFT가 중요하다.
11. **late-vision**: throughput -6.00%, TPOT/E2E 약 +6.6–6.7%. 큰 D cohort의 지속 cycle을 통제 실험으로 사용하기 좋다. 일부 vision output length=1 요청의 TPOT는 해석에서 제외한다.
12. **multi-image**: throughput -4.39%, TTFT mean -13.0%, TPOT mean +21.9%, p95 +43.8%. 먼저 시작했지만 이후 D가 느려졌다. 요청이 5개라 작은 배치 변화/수 ms 차이에 민감하고 반복 수를 늘려야 한다.

## 5. measurement epoch만으로 balanced 원인 재분해

반복 범위도 확인했다. Current balanced 3회는 4125.27–4262.81 token/s, decode-heavy는
4864.71–4964.66, text-heavy는 1871.37–1899.13이었다. 각각 old 4455.01/5319.19/2131.93보다
세 실행 모두 낮다. 단순 current repeat 잡음만으로 격차를 없앨 수 없다는 근거지만,
old는 여전히 1회이므로 cross-day thermal/host variance까지 기각한 통계 검정은 아니다.

`telemetry-balanced-p128` vs `telemetry-balanced-atomic`, 각 1회. `PHASE_EPOCH kind=measurement` 이후 completion만 집계했다. 두 실험은 **v0.10.1 내부의 엔진/profile 차이**다. old v0.10 비교로 바꿔 부르면 안 된다.

| Metric | P128-only | atomic P1024 | 의미 |
|---|---:|---:|---|
| prompt token mass | 25,872 | 25,872 | 동일 work |
| P dispatch | 128 | 160 | +25.0% |
| P mean rows | 2.625 | 2.100 | -20.0% |
| useful prompt tokens/P dispatch | 202.125 | 161.700 | -20.0% |
| P cumulative GPU ms | 1976.08 | 2219.79 | +243.71 ms (+12.33%) |
| P mean service ms | 15.438 | 13.874 | 개별 실행은 작고 빨라짐, 총비용은 증가 |
| D dispatch | 477 | 475 | 거의 동일 |
| D mean rows | 51.723 | 51.941 | 거의 동일 |
| D cumulative GPU ms | 4282.23 | 4195.94 | -86.29 ms |
| D mean service ms | 8.977 | 8.834 | 소폭 감소 |
| action fidelity failures | 0 | 0 | 이 기록에서는 illegal implicit action 증거 없음 |
| activity idle | 10.596% | 11.112% | +0.516 percentage points |
| P+D activity | 15.629% | 13.835% | -1.794 percentage points |

**가장 구체적인 현상은 P formation 악화**다. “decode가 느려서 balanced가 느림”은 이 두 진단에는 맞지 않는다. 전체 suite 3회에서 이 dispatch 차이가 반복되는지는 아직 확인하지 않았다. GPU duration 합은 overlap 때문에 wall time에 그대로 더하면 안 된다. Activity에는 sampling 구간도 있어 completion-duration 합과 일치하지 않는다.

## 6. 수치/출력 검증: cross-engine 차이와 run-to-run 차이는 별개

request_id로 CSV를 join하여 token list 자체를 비교했다. 단순 hash serialization 차이 또는 completion row order 때문인지 먼저 배제했다.

| Workload | old run1 vs current run1 다른 요청 | current 3회 중 token list가 달라진 요청 |
|---|---:|---:|
| short | 16/48 | 0/48 |
| balanced | 168/288 | 0/288 |
| decode-heavy | 240/288 | 0/288 |
| long-prefill | 168/288 | 0/288 |
| bimodal | 192/288 | 0/288 |
| text-heavy | 25/64 | 0/64 |
| mixed | 20/64 | 7/64 |
| poisson | 29/64 | 3/64 |
| vision-heavy | 12/64 | 5/64 |
| wave/drain | 1/20 | 0/20 |
| late-vision | 24/32 | 0/32 |
| multi-image | 0/5 | 1/5 |

Text repeated prompts의 첫 divergence 위치가 request 반복마다 9/20/16/14 등으로 반복되고 current 내부에서는 안정적이다. 이는 deterministic execution/export numerical shift 가설과 부합하나 정확한 원인은 logits/attention comparison이 있어야 확정된다. multi-image는 old/current 첫 run이 5/5 동일한데 current 다른 repeat의 request 2가 달라진다. 두 종류의 문제를 하나의 KV 버그 또는 하나의 FP16 현상으로 묶으면 안 된다.

현재 engine inspector 두 파일은 layer-name-only이며 old 979/new 980 layers다. 이 파일이 **최종 atomic SHA와 묶인 inspector가 아니므로** 해당 layer-count를 최종 회귀 원인 증거로 사용하지 않는다. 정확한 engine SHA, tensor dtype/shape, tactic ID, plugin/JIT cubin hash에 묶인 새 inspection이 필요하다.

## 7. 메모리: 무엇이 같고 다른가

| Metric | old V1 | current V1 | 차이 |
|---|---:|---:|---|
| ready used MiB | 9237 | 9279 | +42 |
| P workspace bytes | 301993472 | 301993472 | 동일 (startup logs) |
| D workspace bytes | 21548544 | 21548544 | 동일 |
| E workspace bytes | 444873728 | 444873728 | 동일 |
| mixed peak median MiB | 9483 | 9401 | -82 |
| vision-heavy peak median MiB | 9547 | 9377 | -170 |
| KV pool | 256 pages, FP16 | 256 pages, FP16 | unchanged ownership capacity |

이전 +582 MiB 문제는 복구 전 v0.10.1과의 차이, +218 MiB는 current P128-only→atomic 차이, **+42 MiB**는 이번 old V1→atomic ready 차이다. 서로 다른 reference를 혼합하지 않는다. Current는 resident baseline은 소폭 증가했지만 VLM transient peak는 줄었다. +42 MiB 전체를 JIT/engine/weights 어느 하나로 단정할 allocation trace는 없다. KV 용량을 줄이는 것은 이 원인을 해결하는 방법이 아니다.

## 8. V2는 무엇을 개선하고 희생하는가

Current vision-heavy V2는 V1보다 token/s +5.42%지만 TPOT mean 30.52→52.87 ms, p95 37.38→87.85 ms, E2E mean 2532.63→3269.63 ms다. E2E p95는 3606.61→3438.68 ms로 감소한다. **전체 drain을 앞당기면서 다수 요청 완료를 늦추는 분포 변화**와 부합한다. Counterfactual 반복 없이 transition evaluator 때문이라고 확정하지 않는다.

Class별:

| Case / class | V1 TTFT mean | V2 TTFT mean | V1 TPOT p95 | V2 TPOT p95 | V1 E2E mean | V2 E2E mean |
|---|---:|---:|---:|---:|---:|---:|
| mixed / text | 92.04 | 83.24 | 42.24 | 70.64 | 2389.27 | 2512.59 |
| mixed / vision | 1336.98 | 1156.74 | 36.71 | 63.25 | 2159.88 | 2325.33 |
| vision-heavy / text | 58.36 | 55.34 | 46.50 | 90.72 | 2220.08 | 3341.95 |
| vision-heavy / vision | 1770.01 | 1681.72 | 35.90 | 85.10 | 2638.62 | 3245.53 |

V2 promotion은 throughput가 아니라 joint TTFT/TPOT/E2E guard로 판단해야 한다. 이것은 workload label을 scheduler 입력에 넣으라는 뜻이 아니라, 모든 request의 stage slack과 completion risk를 동일하게 보호하라는 뜻이다.

## 9. vLLM 비교: send 기준 상세

Frozen vLLM 3회, current V1 3회. vLLM mean은 retained per-request CSV에서 각 run mean을 재계산한 median이다. Text 4개 workload는 앞서 설명한 client-cap 차이가 있으므로 competitive configuration 비교이지 equal-cap causal 비교는 아니다. vLLM token IDs는 수집되지 않아 exact-output identity 비교 불가.

| Workload | vLLM → current token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 1983.53 → 2347.45 | 174.92 → 109.05 | 263.97 → 188.46 | 13.36 → 12.92 | 24.88 → 22.27 | 426.71 → 352.18 | 503.71 → 434.29 |
| balanced | 4319.89 → 4164.17 | 146.79 → 71.21 | 365.02 → 174.29 | 15.12 → 13.28 | 17.40 → 15.03 | 1437.43 → 1201.16 | 2244.11 → 1882.52 |
| decode-heavy | 4854.34 → 4898.64 | 153.43 → 71.79 | 394.23 → 207.18 | 14.00 → 11.37 | 15.03 → 12.24 | 3772.64 → 3009.52 | 5812.44 → 4639.49 |
| long-prefill | 1120.88 → 1223.75 | 2944.21 → 2068.31 | 4257.24 → 2727.59 | 32.43 → 25.34 | 37.65 → 29.45 | 5696.02 → 4221.91 | 7860.82 → 5870.92 |
| bimodal | 1868.29 → 1869.52 | 2509.06 → 1987.13 | 4291.79 → 4537.04 | 23.67 → 18.16 | 42.84 → 26.00 | 5622.52 → 4504.74 | 10328.21 → 9628.46 |
| text-heavy | 1634.76 → 1872.49 | 421.58 → 314.58 | 1232.20 → 1107.21 | 29.21 → 26.65 | 47.36 → 39.82 | 1943.42 → 1687.72 | 2037.81 → 1773.05 |
| mixed | 921.48 → 1078.53 | 874.56 → 714.19 | 2541.43 → 2213.64 | 46.97 → 33.16 | 84.02 → 40.55 | 3008.38 → 2273.13 | 3140.85 → 2649.30 |
| poisson | 1800.07 → 1878.52 | 438.11 → 218.41 | 902.68 → 764.79 | 22.19 → 22.81 | 45.67 → 41.01 | 1800.22 → 1660.85 | 2266.55 → 2125.02 |
| vision-heavy | 579.20 → 668.94 | 1710.70 → 1342.22 | 3691.37 → 3167.36 | 63.70 → 30.52 | 119.58 → 37.38 | 4119.14 → 2532.63 | 4229.37 → 3606.61 |
| wave-drain | 95.85 → 96.31 | 252.76 → 248.30 | 418.60 → 398.29 | 12.43 → 9.89 | 17.26 → 13.90 | 637.86 → 555.92 | 649.42 → 615.70 |
| late-vision | 2359.23 → 2399.25 | 153.40 → 130.03 | 631.51 → 493.15 | 9.90 → 9.81 | 9.93 → 9.86 | 1576.03 → 1535.31 | 1954.46 → 1922.13 |
| multi-image | 244.52 → 298.40 | 259.81 → 233.13 | 402.58 → 317.70 | 12.42 → 9.40 | 16.32 → 12.39 | 644.30 → 524.70 | 653.90 → 535.91 |

## 10. 예정 arrival 기준 E2E (client 대기 포함)

`completed_us - scheduled_arrival_us`를 request별로 계산. TPOT는 send/arrival 기준 변경에 영향이 없고, TTFT/E2E에는 client admission delay가 추가된다. Offered schedule과 실제 dispatch를 구분하며 새 측정 없이 CSV에서 복원했다.

| Workload | old mean / p95 | current mean / p95 | vLLM mean / p95 |
|---|---:|---:|---:|
| short | 334.39 / 413.65 | 352.26 / 434.33 | 426.78 / 503.81 |
| balanced | 2956.41 / 4986.73 | 3176.42 / 5383.42 | 3266.63 / 5264.11 |
| decode-heavy | 7398.97 / 12854.68 | 8055.38 / 13997.54 | 8576.89 / 14464.09 |
| long-prefill | 11287.02 / 19675.66 | 11381.80 / 19577.31 | 12766.52 / 21458.65 |
| bimodal | 10546.11 / 21208.65 | 10674.13 / 21661.14 | 12036.70 / 21908.49 |
| text-heavy | 1472.06 / 1572.94 | 1687.79 / 1773.13 | 1943.50 / 2037.90 |
| mixed | 2123.91 / 2404.80 | 2273.20 / 2649.36 | 3008.46 / 3140.92 |
| poisson | 1475.55 / 1917.75 | 1660.93 / 2125.08 | 1800.30 / 2266.63 |
| vision-heavy | 2955.71 / 3402.78 | 2532.70 / 3606.67 | 4119.21 / 4229.44 |
| wave-drain | 491.74 / 504.88 | 555.99 / 615.79 | 637.96 / 649.51 |
| late-vision | 1439.37 / 1805.28 | 1536.36 / 1923.67 | 1577.05 / 1956.39 |
| multi-image | 507.36 / 512.62 | 525.15 / 536.17 | 644.83 / 654.32 |

Balanced current send-E2E mean 1201.16 ms vs arrival-E2E mean 3176.42 ms다. frozen vLLM은 arrival mean 3266.63 ms, p95 5264.11 ms이고 current p95 5383.42 ms다. 따라서 current의 낮은 send-latency만으로 모든 arrival-relative tail도 우수하다고 결론 내릴 수 없다.

## 11. 개선 우선순위 재검토

| 우선순위 | 변경 전 확인/실험 | 통과 기준 / 다음 판단 |
|---|---|---|
| P0: 비교 계약 잠금 | engine/plugin/cubin/tokenizer hashes; 이미지 content hash; max-in-flight; mean/p95 정의; graph on/off manifest | 12 trace 내용 동일 확인은 완료. Equal-cap vLLM이 필요하면 4개만 별도 측정; frozen 수치는 유지 |
| P1: correctness 분리 | 동일 Q/K/V·동일 KV page table에서 old prebuilt vs new JIT XQA; logits/top-k 및 FP16 error, first divergence 위치 | kernel difference vs exporter/GEMM vs ownership 구분. 옛 serialized engine를 새 plugin에 직접 로드하지 않음 |
| P2: P formation parity | 동일 ready snapshot, request IDs와 row order에서 P candidate 비교; measurement-only useful tokens/dispatch; fixed P128 유지 | 128→160 dispatch가 반복되면 enqueue timing/formation mechanism부터 수정. Shape만 줄이는 정책 금지 |
| P3: decode cycle | balanced/decode-heavy/late-vision의 GPU D service, GPU-completion→sampling-ready→next-enqueue, JIT launch dimensions/compilation options | D GPU가 증가하면 kernel/tactic, host gap이면 completion/submit 경로. 예측 score 튜닝은 뒤로 |
| P4: VLM formation/ownership | unknown payload E1 guard 발동 횟수, E/P starts, direct-output lease consumers, text vs vision D gap | guard를 삭제하지 말고 payload estimator 정확도 개선; 4 repeat-failing traces를 replay/sanitizer |
| P5: policy ablation | 같은 atomic engine에서 external-residual Scalar 권한만 on/off; V0/V1/V2 교차 반복 | long-prefill 이득 보존 + text-heavy/poisson TPOT 회복. V2는 평균·tail 모두 gate |
| P6: CUDA graph 공통 개선 | 정확한 graph opt-in, graph-capture/replay coverage/host cost, old와 current를 모두 graph-on으로 비교 | 현재 eager 회귀와 독립된 최적화. Capture cost·메모리·binding correctness 포함 |

우선 P1–P3가 핵심이다. 새 RLS 구조나 workload별 tuning을 추가할 단계가 아니다. XQA 변경은 **확인된 코드 차이**이고 느려짐의 원인은 **가설**이다. P fragmentation은 **단일 진단의 실측 현상**이며 full12 반복 원인은 **추가 검증 대상**이다. 이 경계를 유지한다.

## 12. 재현 입력과 한계

- old V0/V1/V2: `.local/results/current/v{0,1,2}-full12`.
- current: `.local/results/v0101-forward-port/atomic-vp1024-final-full12-{v0,v2}`, `atomic-vp1024-final-full12-r3`.
- frozen vLLM: `.local/results/baselines/vllm-frozen-12x3`.
- measurement telemetry: `.local/results/v0101-forward-port/telemetry-balanced-{p128,atomic}/generic/balanced/worker-4/activity/run-001-events.jsonl`.
- request CSV: `generic/<case>/worker-4/run-NNN/client/run-001/requests.csv`; vLLM uses `<case>/run-NNN/client/run-001/requests.csv`.
- output analysis CSV: `.local/results/v0101-forward-port/244-old-current-vllm-latency-audit.csv`.
- Old engine config/plan referenced by old commands is no longer at its original path. Workspace comparison uses retained startup logs; the current source root is a source-level comparison, not a proof of old binary source identity.
- No source/runtime modifications, no new GPU serving runs, no new SLO thresholds, no claim of semantic failure solely from token disagreement.
