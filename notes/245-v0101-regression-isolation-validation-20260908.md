# v0.10.1 회귀 분리 검증: XQA, 실행 구간, graph, VLM 순서

Date: 2026-09-08. Branch: `codex/v0101-phase-forward-port`.
Diagnostic tools and measurement-start instrumentation commit: `a0f4c28`.

## 목적과 변경 범위

[244](244-v010-v0101-detailed-regression-audit-20260908.md)의 원인 분리 계획을 실행한다.
새 workload-specific policy나 RLS parameter는 추가하지 않는다. Serving binary는 기존 `4099f76` runtime을
그대로 사용하며, 이번 source 변경은 재현 가능한 XQA 비교 도구와 분석기 테스트다.
Production 기본 설정을 바꾸는 것과 실험 옵션을 켜는 것을 구분한다. 마지막에는 benchmark adapter에
measurement-start graph counter 로그만 추가하여 warmup hit와 serving hit를 분리한다.

Retained artifacts: `.local/results/v0101-forward-port/regression-resolution/`.
Atomic engine SHA256: `084d039248e08dc192a57ebf38d521ee1fdca2f037a865f55ac0b0b904214b0b`.
FP16 Cosmos-Reason2-2B, RTX 3080, P8/D64/E4, text chunk128, stable slots80, KV pages256 유지.
Generic calibration: text239/VLM319 requests. Fresh process마다 calibration을 다시 실행한다.
Frozen vLLM은 재사용하며, note244의 four-text client admission 차이는 그대로 명시한다.

## 1. 동일 Q/K/V와 page table: prebuilt 대 JIT XQA

`benchmarks/phase_serving/xqaParityProbe.cpp`를 각각 old/current headers와 static archive에 독립 link했다.
`XQALaunchParams`의 ABI가 달라 old archive를 new headers로 호출하지 않는다.
Hq16/Hkv8/D128, FP16, 128-token page, reversed physical page mapping, row별 가변 context length,
deterministic input seeds17/31. Sliding-window, RoPE write, donor sharing, full model logits는 이 검사 범위 밖이다.

Batch1/8/32/64 × context128/512/1536 ×3회, 실행 순서 교대. 각 실행 warmup20/timed100 CUDA event.
**36/36 FP16 원소값 완전 일치, finite, max absolute error0.** 독립 수학 reference 대비 정확성 증명은 아니다.

GPU event median의 run median, 단위 µs:

| D batch | context | prebuilt | JIT | 차이 |
|---:|---:|---:|---:|---:|
| 1 | 128 | 7.552 | 8.192 | +8.47% |
| 1 | 512 | 13.088 | 14.112 | +7.82% |
| 1 | 1536 | 26.624 | 27.312 | +2.58% |
| 8 | 128 | 8.192 | 9.216 | +12.50% |
| 8 | 512 | 28.672 | 29.312 | +2.23% |
| 8 | 1536 | 76.416 | 76.800 | +0.50% |
| 32 | 128 | 39.808 | 40.720 | +2.29% |
| 32 | 512 | 100.000 | 100.352 | +0.35% |
| 32 | 1536 | 288.128 | 288.768 | +0.22% |
| 64 | 128 | 70.656 | 71.744 | +1.54% |
| 64 | 512 | 192.368 | 193.152 | +0.41% |
| 64 | 1536 | 567.296 | 568.320 | +0.18% |

작은 shape의 상대 차이는 있지만 절대 차이는 약1µs이고 큰 batch/context에서는 1% 미만이다.
따라서 **이번 검사만으로 JIT XQA가 전체 6–8% 회귀나 token divergence의 원인이라고 볼 수 없다.**
이것은 XQA 전체 무혐의 판정이 아니다. 이벤트 구간은 host launch starvation도 포함할 수 있고
process 간 clock 차이도 있어 1µs 내외 차이를 pure kernel instruction 차이로 단정하지 않는다.

재현은 TensorRT container 안에서 `TRT_PACKAGE_DIR=/opt/tensorrt`,
`LD_LIBRARY_PATH=/opt/tensorrt/lib:/usr/local/cuda/lib64`를 설정한 뒤:

```bash
bash benchmarks/phase_serving/build_xqa_parity.sh \
  /workspace /workspace/.local/v010-forward-build \
  /workspace/.local/v0101-forward-port /workspace/.local/v0101-forward-build-make \
  /workspace/.local/results/v0101-forward-port/regression-resolution/xqa
python3 benchmarks/phase_serving/run_xqa_parity.py \
  --probe-dir /workspace/.local/results/v0101-forward-port/regression-resolution/xqa \
  --output-dir /workspace/.local/results/v0101-forward-port/regression-resolution/xqa/results \
  --repeats 3
```

`xqa/results/manifest.json`에는 실행 명령과 binary SHA256, `summary.csv`에는 mean/max error와
각 run median/p95가 있다. 출력 FP16 파일도 보관한다. 재실행 시 새 output directory를 사용한다.

## 2. P formation 소스 비교

Root v0.10 source와 current `phaseQueueScheduler.cpp`를 직접 비교했다. P batch selection은
formatting 외 알고리즘 변화가 없으며, 실질적 추가는 optional decode-dispatch blocking이다.
이 block은 shared/exclusive encoder 경로를 위한 것으로 default independent frontier의 새로운 P packing 규칙이 아니다.

Note244의 P128→atomic 진단에서 같은 prompt25872 tokens에 P dispatch128→160은 실측됐지만,
**같은 ready snapshot에서 서로 다른 generator가 나온다는 증거는 아니다.**
Engine latency / prefill completion / admission timing이 successor ready cohort를 바꾸는 경로를 먼저 의심한다.
Old/current runtime의 same-snapshot branch replay는 아직 별도 gate로 남아 있다.

## 3. Full timeline: 4개 워크로드

`timeline-commands.json`으로 balanced/decode-heavy/mixed/multi-image 각각1회.
`EMIT_PHASE_METRICS=1`, `PHASE_TELEMETRY_LEVEL=full`, file output을 사용했다.
기존 `analyze_phase_timeline.py`가 production request의 마지막 complete lifecycle을 선택하여
calibration의 같은 request ID 재사용을 제외한다. 288/288, 288/288, 64/64, 5/5 attribution 완료.

**Observer effect가 크다.** Balanced3248.04 token/s는 uninstrumented V1 median4164.17보다22.0% 낮다.
Decode-heavy3970.87도4898.64보다18.9% 낮다. Full 로그는 병목 후보를 분해하는 데만 사용하며,
원래 uninstrumented host overhead 또는 정확한 old/current regression 원인으로 그대로 인용하지 않는다.

Request별 누적 stage 시간을 전체 decode row-step 수로 나눈 값, 단위 ms:

| Workload | row-steps | sampling submit→ready | ready→collect | collect→commit | decode-ready queue |
|---|---:|---:|---:|---:|---:|
| balanced | 24672 | 0.1872 | 0.0107 | 0.0536 | 5.0943 |
| decode-heavy | 74592 | 0.1866 | 0.0106 | 0.0528 | 2.4261 |
| mixed | 2864 | 3.1878 | 0.0109 | 0.0552 | 27.0896 |
| multi-image | 155 | 1.6599 | 0.0022 | 0.0068 | 1.8885 |

이는 per-request **전체 누적시간을 한 iteration 시간으로 잘못 읽지 않기 위한 정규화**다.
Queue wait에는 실제 다른 E/P/D 작업, batching/WAIT, host 경로가 모두 들어간다.
Sampling submit→ready도 GPU sampling kernel time만이 아니라 실행 대기/완료 확인을 포함한다.
Table 항들을 독립 GPU wall-time처럼 합산할 수 없다.

Decode-only dispatch의 GPU event mean / host submission mean / scheduler decision mean(ms):

| Workload | GPU event | host submission | scheduler decision |
|---|---:|---:|---:|
| balanced | 8.218 | 2.131 | 0.693 |
| decode-heavy | 8.921 | 1.865 | 0.702 |
| mixed | 9.150 | 2.044 | 0.682 |

Host와 GPU 구간은 겹칠 수 있으므로 더하면 안 된다. GPU common epoch와 CPU observation의
cross-clock calibration 및 first-kernel timestamp는 이 로그만으로 확보되지 않았다.
다음 계측은 request별 JSON 출력 대신 bounded in-memory timestamps/aggregate로 낮은 perturbation을 검증해야 한다.

## 4. CUDA graph 전체 검사

`graph-commands.json`: capture enabled, P graph limit32/D64; 같은 engine/calibration/trace.
측정 전 generic shape warmup으로 capture하고 serving에서는 캐시된 graph만 replay한다.
Captured shape coverage에 따라 miss는 eager fallback이다.
Eager 대비 calibration cost key/probe 수도 달라질 수 있으므로 pure launch-cost ablation이 아니라
**graph를 켠 전체 serving configuration의 비교**다.

첫 balanced/mixed sentinel은 실제 graph hit를 확인했다. Balanced log에서 P captures2/hits436,
D captures9/hits618이다. 이 count는 calibration 포함이며 measurement-only hit rate로 쓰면 안 된다.
Balanced ready memory9279→9303MiB, +24MiB였다.

### 4.1 전체12 graph-on screening 결과

Graph1회 vs eager3회 median, frozen vLLM3회. 작은 차이는 통계적 승리 판정이 아니다.
Latency는 send 기준 ms. Eager latency 상세는 note243/244, 아래는 graph의 값이다.

| Workload | eager token/s | graph token/s | 변화 | frozen vLLM token/s | graph TTFT mean / p95 | graph TPOT mean / p95 | graph E2E mean / p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | 2347.45 | 2361.41 | +0.59% | 1983.53 | 104.81 / 188.35 | 12.77 / 22.00 | 346.43 / 425.93 |
| balanced | 4164.17 | 4199.99 | +0.86% | 4319.89 | 66.76 / 170.74 | 13.17 / 15.19 | 1188.46 / 1913.10 |
| decode-heavy | 4898.64 | 5022.33 | +2.52% | 4854.34 | 70.65 / 198.22 | 11.06 / 11.77 | 2928.95 / 4477.24 |
| long-prefill | 1223.75 | 1261.15 | +3.06% | 1120.88 | 2009.33 / 2591.16 | 24.50 / 28.04 | 4090.90 / 5601.56 |
| bimodal | 1869.52 | 1814.91 | -2.92% | 1868.29 | 2024.80 / 4161.39 | 18.77 / 29.77 | 4631.15 / 9602.08 |
| text-heavy | 1872.49 | 1865.19 | -0.39% | 1634.76 | 305.88 / 1137.46 | 27.12 / 40.70 | 1699.68 / 1789.15 |
| mixed | 1078.53 | 1118.73 | +3.73% | 921.48 | 695.55 / 2096.05 | 39.77 / 64.62 | 2481.73 / 2608.66 |
| poisson | 1878.52 | 1855.93 | -1.20% | 1800.07 | 217.61 / 763.53 | 23.19 / 40.08 | 1702.93 / 2174.44 |
| vision-heavy | 668.94 | 666.85 | -0.31% | 579.20 | 1348.01 / 3218.46 | 31.16 / 37.40 | 2558.39 / 3607.44 |
| wave-drain | 96.31 | 96.28 | -0.03% | 95.85 | 244.33 / 394.80 | 9.89 / 13.71 | 550.89 / 611.60 |
| late-vision | 2399.25 | 2474.83 | +3.15% | 2359.23 | 129.89 / 498.35 | 9.50 / 9.55 | 1491.49 / 1863.20 |
| multi-image | 298.40 | 299.72 | +0.44% | 244.52 | 232.17 / 315.69 | 9.36 / 12.32 | 522.45 / 533.61 |

Throughput geometric mean **+0.77%**, 7/12 positive. Balanced는 sentinel4126.90과 screening4199.99로
방향 자체가 달라 미세 개선을 확정할 수 없다. Mixed는 E2Emean2273.13→2481.73(+9.17%),
TPOTp95 40.55→64.62(+59.37%)여서 처리량만 보고 default로 승격하지 않는다.
Graph에서도 oldV1 balanced4455.01/decode-heavy5319.19에는 미달한다.

### 4.2 Graph coverage의 추가 함정

모든12 로그에서 P hits가 정확히436이다. P captures2/D captures9이며 D hits는429~1383이다.
전체 종료 counter에는 warmup이 섞여 있으므로 **P hit436을 production P가436번 replay됐다고 해석하면 안 된다.**
이 구분을 위해 `llm_phase_context_smoke.cpp`에 calibration-end/measurement-start counter snapshot을 추가했다.
측정 시작과 종료의 차이로 serving hits/misses를 계산할 수 있다. GPU 동기화나 hot-loop logging은 추가하지 않는다.

증분 빌드 후 balanced1회(`graph-epoch-confirm`)에서 실제 구분을 확인했다:

| Phase | measurement 시작 hits / misses | 종료 hits / misses | serving hit / miss | serving hit rate |
|---|---:|---:|---:|---:|
| P | 436 / 243 | 436 / 406 | 0 / 163 | 0% |
| D | 436 / 327 | 635 / 605 | 199 / 278 | 41.72% |

**이 balanced serving에서 P graph replay는 실제0회였다.** 전체12에서의 constant436과 부합하지만
다른11개 production hit rate를 이 한 번으로 확정하지 않는다. `TrtEngineExecutor::computeBindingHash()`는
profile index + 모든 tensor binding address + shape를 포함한다. Generic warmup과 실제 packed P의
어느 binding field가 달라지는지는 추가 진단 대상이다. Hash에서 address를 제거하는 식으로 correctness를 희생하면 안 된다.

Confirmation은4230.30 token/s, TTFT66.50/169.64ms, TPOT13.11/15.54ms,
E2E1181.97/1878.96ms였다. Eager와 token hash는 동일하다. 기존 sentinel/full12와 합쳐 선택적으로
가장 좋은 숫자만 인용하지 않으며, counter 추가 자체의 성능 이득이라고 해석하지 않는다.

## 5. VLM canonical adapter order: 4개 ×3회

Graph off, 나머지는 eager V1 그대로, `TRT_EDGELLM_CANONICAL_VISION_ADAPTER_ORDER=1`만 변경했다.
이는 image preprocessing 완료 결과를 ingress 순서대로 전달하는 설정이다. E batch, P/D row order,
CUDA kernel reduction order를 모두 고정하는 옵션은 아니다.

| Workload | eager → canonical token/s | canonical TTFT mean / p95 | canonical TPOT mean / p95 | canonical E2E mean / p95 | 3회 중 token이 달라진 request: eager → canonical |
|---|---:|---:|---:|---:|---:|
| mixed | 1078.53 → 1082.94 | 759.21 / 2289.83 | 32.59 / 40.22 | 2285.07 / 2627.15 | 7/64 → 2/64 |
| poisson | 1878.52 → 1859.69 | 231.31 / 876.51 | 22.59 / 42.63 | 1688.75 / 2150.01 | 3/64 → 2/64 |
| vision-heavy | 668.94 → 646.44 | 1435.53 / 3316.69 | 31.37 / 38.73 | 2656.62 / 3732.30 | 5/64 → 1/64 |
| multi-image | 298.40 → 261.94 | 334.99 / 386.26 | 8.89 / 12.00 | 606.81 / 609.30 | 1/5 → 1/5 |

Request ID별로 `output_token_ids`를 비교하여 하나라도 다른 ID 수를 셌다. Completed row 순서나
aggregate hash만 비교한 값이 아니다. Mixed/poisson/vision-heavy의 변동 request 수는 줄었지만
**4/4 exact-repeat gate는 여전히 실패**다. 세 번의 표본으로 불일치 확률 개선을 확정하지 않는다.
Vision-heavy throughput -3.36%, multi-image -12.22%, multi-image TTFTmean +43.69%다.
따라서 canonical adapter를 production 기본값으로 채택하지 않는다.

결론은 "adapter order가 전부의 원인"도 "ownership 오류가 없다"도 아니다. Preprocessing ordering만
고정하는 것으로는 부족하고, 동일 token prefix에서 logits/top-k 차이와 E/P/D batch membership을
같이 대조해야 한다. Unknown-payload E1 guard는 safety invariant로 유지했다. 이번 로그에 guard
발동 counter가 없어 현재 Cosmos에서 실제 발동 횟수를 확정할 수 없다.

## 6. 검증과 잔여 gate

- XQA comparison helper + timeline analyzer: Python unit tests14개 통과.
- `unitTestRuntime`: PhaseQueueScheduler139 + PhaseFormationPlanner17 + PhaseKVActiveView2 +
  IndependentPhaseAsyncServer28 = 186개 통과. 별도 state binary에서0개가 선택된 시도는 검증 실적에 포함하지 않는다.
- JIT XQA B64/context512/reversed page mapping: compute-sanitizer memcheck error0.
  Sanitizer instrumentation의 실행시간은 성능 비교에서 제외한다. Full VLM sanitizer 통과를 의미하지 않는다.
- 새 도구: pre-commit license/format/static checks 통과.
- Production runtime policy / KV allocator / memory budget 변경 없음.
- XQA representative parity 통과는 full-engine logits identity나 VLM ownership correctness를 대체하지 않는다.
- Full telemetry observer effect 때문에 low-overhead cycle attribution은 아직 필요하다.
- V0/V1/V2 기존 비교와 frozen vLLM 상세12 표는 note243/244 유지. 작은 차이를 반복 유의성 없이 승리로 선언하지 않는다.

## 7. 실행 상태와 다음 순서

| Audit 항목 | 이번에 확인한 것 | 아직 필요한 것 |
|---|---|---|
| P0 비교 계약 | binary/engine identity 보관; serving vs calibration graph counter 분리 | equal-client-cap fresh vLLM4개는 별도 미실행 |
| P1 XQA/numerics | isolated XQA36/36 exact, representative memcheck0 | full-model first-divergence logits/top-k; old/new exporter/GEMM/RoPE 경로 |
| P2 formation | P selection source parity; 측정-only P fragmentation 자료 유지 | 동일 snapshot/request IDs에서 branch replay |
| P3 cycle | full timeline4개에서 sampling/queue/submission 구분 | full logging의 observer effect를 제거한 계측 |
| P4 VLM | canonical-order4×3 실패를 확인하여 기본값 유지 | guard 발동 counter, same-prefix numerical trace, full VLM sanitizer |
| P5 policy | 기존 V0/V1/V2 결과 재검토; workload별 tuning 추가 안 함 | mechanism gate 후 동일-contract 교차 반복 |
| P6 graph | full12 screening + 실제 serving P0%/D41.72% 확인 | binding miss 원인, 안전한 generic shape capture coverage, 전체 반복 gate |

지금 가장 직접적인 성능 후보는 **P graph가 실제 serving에서 재사용되지 않는 이유를 분리하는 것**이다.
그다음 low-overhead GPU/host cycle, correctness의 first-logit divergence를 병행한다.
Graph를 무조건 켜거나 canonical adapter를 강제하면 전체 workload 평균/tail을 보존하지 못했다.
따라서 이번 단계는 default 성능 최적화 완료가 아니라, 효과 없는 대안 두 개를 배제하고 다음 구현 대상을
구체화한 검증이다. Old 최고 성능 복구 및 exact-repeat production promotion은 아직 완료되지 않았다.
