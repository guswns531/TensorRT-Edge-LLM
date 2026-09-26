# 334 — Workspace mode 최종 screen: 메모리, graph coverage, latency trade-off

작성일: 2026-09-26. 이 문서는 저장된 HTTP 결과와 GPU telemetry를 CPU로 재분석한 기록이다.
이 문서 작성 과정에서 GPU 실행, 빌드, 새 성능 측정은 하지 않았다.

## 1. 판정 요약

**24개 요청 cell 중 18개는 완료했고, Gemma independent 6개는 startup OOM으로 실패했다.**
Manifest의 최종 상태는 `partial`, exit code는 `1`이다. 18개 성공만 보고 전체 24개 검증을
통과했다고 표현하지 않는다. 모델별 3개 workload × 1회 screen이며, Full12 × 3회 검증이 아니다.

- Gemma는 동일 KV 192 pages와 D24 graph coverage를 유지했을 때 `shared_ep`만 실행되었다.
  Independent OOM을 피하려고 KV, D batch, graph 수를 자동으로 줄이지 않았다.
- Cosmos는 동일 KV 256 pages와 D64 graph coverage로 두 mode 모두 실행되었다.
  Independent peak는 shared보다 **440–556 MiB 높았다**.
- Cosmos independent는 vision-heavy 처리량과 tail을 크게 개선했지만 resident text의 decode
  latency를 크게 악화시켰다. **E/P/D 실행 자유도가 커지는 것과 모든 요청 latency가 좋아지는 것은 다르다.**
- 완료된 18개에서 측정된 action-fidelity violation은 0이었다. Shared E/P의 engine 실행 overlap은
  0이었다. Envelope의 극소 E∩P 비율은 sampling 등 계측 범위가 포함된 값이므로 workspace 위반으로
  해석하지 않는다.
- `predictor-on/off`는 contextual Scalar RLS 활성/비활성 실험이 아니다. 7절의 authority 구분이 필요하다.
- 기본 설정을 변경하거나 workload별 mode를 고르는 근거로 이 1회 screen을 사용하지 않는다.

M-RoPE-only admission barrier를 제거한 `5520216` 후보는 별도 실험에서 resident text 회귀로
거부되었다. [노트 333](333-encoder-admission-mrope-lifetime-fix-20260926.md)의 결과와 이 screen을
섞지 않는다. 이 screen은 해당 변경을 되돌린 runtime과 같은 `43c680a` 바이너리를 사용했다.

## 2. 재현 contract와 artifact identity

저장소 루트는 `/home/sslab/TensorRT-Edge-LLM`이다. 아래 경로는 모두 그 루트 기준이다.

| 자료 | 경로 |
|---|---|
| 실행 manifest 및 raw root | `.local/results/runtime-contract-revalidation-20260926/workspace-final-screen` |
| 7개 serving 지표와 frozen vLLM 비교 | `.local/results/runtime-contract-revalidation-20260926/workspace-final-report.{json,csv,md}` |
| 분석 identity / baseline hash | `.local/results/runtime-contract-revalidation-20260926/workspace-final-report.manifest.json` |
| HTTP/output integrity audit | `.local/results/runtime-contract-revalidation-20260926/workspace-final-quality.{json,md}` |
| 실행 driver 전체 로그 | `.local/results/runtime-contract-revalidation-20260926/workspace-final-screen.log` |

각 cell 상대 경로는 다음과 같다.

```text
workspace-final-screen/
  {gemma,cosmos}/
    {shared_ep,independent}-predictor-{on,off}/
      repeat-001/
        {balanced,vision-heavy,multi-image}/
          aggregate.json
          driver.log
          run-001/
            gateway.log.gz
            activity-summary.csv
            activity-intervals.csv
            client/run-001/requests.csv
```

실패 cell은 `aggregate.json`이 없고 gateway log가 압축 전 `gateway.log`로 남을 수 있다.

| 항목 | 값 |
|---|---|
| Campaign source commit | `985ff1a9f274732ce8bdf57aa3ea784ea55dd17d` |
| Campaign source state | clean; tracked diff SHA256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Binary source commit | `43c680a2a7e4dea4922c33e9c61f93628b6a4d95` |
| Binary SHA256 | `aab919194709253bdc377830653e1c201abf4a0b34967469cf0db0832cb1552b` |
| Plugin SHA256 | `ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2` |
| Runner SHA256 | `98e93f721feaf1b94cf7c3bb59b5ad6fc9074bd7802ac2a1f7b5d4aaf1b61c55` |
| Container digest | `nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3` |

| 설정 | Gemma | Cosmos |
|---|---|---|
| Model | `google/gemma-4-e2b-it` AWQ lineage | `nvidia/Cosmos-Reason2-2B` FP16 lineage |
| Text engine | `.local/artifacts/v0101-forward-port/workspace-corrected-20260926/gemma` | `.local/artifacts/v0101-forward-port/workspace-corrected-20260926/cosmos` |
| Engine SHA256 | `fef5210c22b0ceb064cdce07658f6dace7737d66cad7e35a3f625602f2c9405e` | `c4f873c30db785cb87aba4475cc80b935112f225c3e2204f22c2ade09356026d` |
| Vision engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/visual-e4-soft280/visual` | `.local/artifacts/v0101-forward-port/cosmos-reason2-2b/vision-exact-gelu/engine/visual` |
| Vision engine SHA256 | `094d84bdc2cd8f5d338b33d19f321bab8ebbbb275fb242a001047605c8684f69` | `4ede4ddfcea4dadf8d5c507cf1c99bc6c5e3a36197ba4ba2a53e11f72434e2c4` |
| E / P / D maximum rows | 4 / 8 / 24 | 4 / 8 / 64 |
| Fixed prefill chunk | 128 | 128; separate engine vision profile supports up to 1024 |
| Stable slots / max in-flight | 24 / 24 | 80 / 64 |
| KV pool pages | **192** | **256** |
| Generic calibration requests | 49 | 239 |
| Graph setting | enabled; P cap 0 / D cap 64 | enabled; P cap 0 / D cap 64 |
| Encoder formation wait | 25,000 µs | 25,000 µs |
| Decode burst grace | 20,000 µs | 20,000 µs |
| Single-storage flag | `TRT_EDGELLM_SHARED_EP_SINGLE_STORAGE=1` | same |

공통 policy는 `service-scaled-transition` / global active다. 요청은 동일 HTTP trace의 고정 output
길이와 `ignore_eos` contract로 실행했다. Sparse decode warm-up batch 목록은 Gemma
`1,2,4,8,12,16,20,24`, Cosmos `1,2,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64`다.
Sparse warm-up 입력 목록과 최종 실제 graph cache entry 수를 혼동하지 않는다.

Generic calibration artifact:

- Gemma: `.local/results/gemma4-packed-prefill-g4-20260912/generic-p8-d24-e4.json`,
  SHA256 `b7e576ab7d77387e24e6e1f19a67b12b23e7c534b9d26cc40423644e7eba3bc4`.
- Cosmos: `.local/results/v0101-forward-port/heuristic-elimination-20260911/v3-profile-free-canonical-full12-3x/inputs/generic-vlm-v7-small-d-468359b8d4ab2649-7baed8c23852af32-81ba98613783027f.json`,
  SHA256 `3be86e786f9ae0344a08d4001953a67cb9f64ef3e34ed09d909136aa905f8f3c`.

각 workload trace hash와 모든 command/environment는 manifest에 저장되어 있다. 같은 모델 안에서
workspace와 predictor flag 이외 engine, trace, calibration, KV page 수를 바꾸지 않았다.

## 3. 절대 serving 지표: 성공한 18개 cell

모든 latency 단위는 **ms**, peak는 **MiB**, 처리량은 **generated token/s**다.
표의 `Pctl`은 predictor flag를 뜻하며 prefill batch와 무관하다.
모든 행은 **1회 측정**이다. 평균은 요청 latency의 산술평균이고 p95는 해당 실행의 요청 분포에서
구한 값이다. 여러 실행을 합친 p95나 confidence interval이 아니다.

### 3.1 Gemma shared E/P workspace

| Mode | Pctl | Workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_ep | on | balanced | 1204.753 | 137.087 | 605.641 | 15.729 | 17.277 | 1448.507 | 2138.323 | 9849 |
| shared_ep | off | balanced | 1207.087 | 148.073 | 620.682 | 15.604 | 16.883 | 1452.622 | 2152.152 | 9849 |
| shared_ep | on | vision-heavy | 536.069 | 321.936 | 705.697 | 33.996 | 50.602 | 1583.951 | 2130.778 | 9853 |
| shared_ep | off | vision-heavy | 499.265 | 497.627 | 1115.230 | 33.612 | 52.066 | 1717.168 | 2381.125 | 9853 |
| shared_ep | on | multi-image | 298.407 | 656.260 | 968.156 | 23.578 | 35.854 | 1387.183 | 1788.869 | 9853 |
| shared_ep | off | multi-image | 365.137 | 350.356 | 593.590 | 29.332 | 45.881 | 1259.656 | 1574.162 | 9853 |

### 3.2 Cosmos shared E/P workspace

| Mode | Pctl | Workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_ep | on | balanced | 4258.206 | 63.113 | 151.031 | 12.671 | 14.141 | 1141.162 | 1783.304 | 9299 |
| shared_ep | off | balanced | 4234.839 | 66.380 | 151.443 | 12.717 | 14.161 | 1147.035 | 1793.953 | 9299 |
| shared_ep | on | vision-heavy | 392.099 | 2293.813 | 5471.897 | 8.428 | 12.399 | 2630.265 | 5694.455 | 9299 |
| shared_ep | off | vision-heavy | 390.211 | 2313.441 | 5496.169 | 8.361 | 12.388 | 2647.540 | 5718.577 | 9299 |
| shared_ep | on | multi-image | 219.885 | 260.932 | 452.411 | 7.110 | 8.099 | 481.332 | 666.023 | 9299 |
| shared_ep | off | multi-image | 174.842 | 402.790 | 642.948 | 6.967 | 7.791 | 618.768 | 860.444 | 9299 |

### 3.3 Cosmos independent E/P/D workspace

| Mode | Pctl | Workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak MiB |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| independent | on | balanced | 4197.853 | 61.381 | 160.290 | 12.863 | 14.265 | 1156.410 | 1818.415 | 9739 |
| independent | off | balanced | 4214.608 | 61.388 | 150.839 | 12.760 | 14.185 | 1146.917 | 1797.766 | 9767 |
| independent | on | vision-heavy | 711.802 | 1304.113 | 3008.458 | 50.528 | 82.673 | 3221.368 | 3380.476 | 9739 |
| independent | off | vision-heavy | 703.306 | 1429.423 | 3139.602 | 47.549 | 81.213 | 3255.726 | 3415.483 | 9855 |
| independent | on | multi-image | 303.381 | 278.007 | 309.986 | 7.913 | 9.285 | 523.325 | 527.122 | 9739 |
| independent | off | multi-image | 303.515 | 277.600 | 309.870 | 7.923 | 9.296 | 523.227 | 526.833 | 9739 |

### 3.4 성공으로 숨겨 처리하지 않은 실패 cell

| Model / mode | Pctl | Workload | 결과 / serving 지표 |
|---|---|---|---|
| Gemma independent | on | balanced | Startup `cudaMalloc` OOM; N/A |
| Gemma independent | off | balanced | Startup `cudaMalloc` OOM; N/A |
| Gemma independent | on | vision-heavy | Startup `cudaMalloc` OOM; N/A |
| Gemma independent | off | vision-heavy | Startup `cudaMalloc` OOM; N/A |
| Gemma independent | on | multi-image | Startup `cudaMalloc` OOM; N/A |
| Gemma independent | off | multi-image | Startup `cudaMalloc` OOM; N/A |

모두 gateway health-ready 이전 실패다. Gateway에는
`CUDA runtime error in cudaMalloc(&data, memoryCapacity): out of memory`, driver에는
health HTTP 500 및 `backend exited with code 139`가 남았다. 이는 측정 중 요청 error 6개라는 뜻이
아니라 **서버 시작에 실패한 실험 6개**라는 뜻이다. 처리량을 0으로 넣거나 성공한 shared 결과로
대체하지 않는다. 이 screen의 실패만으로 independent가 모든 모델/더 큰 GPU에서 불가능하다고
일반화하지 않는다.

## 4. 실제 graph coverage와 hidden fallback 점검

단순히 `aggregate.json`이 존재한다는 것만으로 graph contract 성공을 선언하지 않았다.
각 성공 gateway log의 최종 `Phase CUDA graph cache:` 및 CUDA error를 함께 확인했다.

| Model / mode | 완료 / 요청 | 실제 P entries / captures | 실제 D entries / captures | D misses | Evictions | 상태 |
|---|---:|---:|---:|---:|---:|---|
| Gemma shared_ep, on/off | 6 / 6 | 0 / 0 | 24 / 24 | 0 | 0 | 완료; D graph 축소 없음 |
| Gemma independent, on/off | 0 / 6 | N/A | N/A | N/A | N/A | Startup OOM; serving 미시작 |
| Cosmos shared_ep, on/off | 6 / 6 | 0 / 0 | 64 / 64 | 0 | 0 | 완료; D graph 축소 없음 |
| Cosmos independent, on/off | 6 / 6 | 0 / 0 | 64 / 64 | 0 | 0 | 완료; D graph 축소 없음 |

성공한 18개 log에는 OOM, CUDA runtime error, `[ERROR]`, terminate marker가 발견되지 않았다.
P graph miss가 있는 것은 `MAX_PREFILL_GRAPHS=0`인 **의도된 eager P 경로**다. D graph allocation
실패를 숨기고 eager로 내려간 결과와 다르다. 이 확인은 해당 18개 실행 범위의 health 확인이지,
다른 shape/load의 메모리 안전성이나 sanitizer 통과를 대신하지 않는다.

메모리 비교도 같은 관점으로 읽는다.

- Cosmos predictor-on의 independent는 모든 3개 workload에서 shared 대비 **+440 MiB**였다.
- Predictor-off는 balanced +468, vision-heavy +556, multi-image +440 MiB였다.
- 이 값은 전체 GPU peak의 차이다. Workspace 외에 실제 cohort, retained payload와 allocation peak
  시점도 달라지므로 전부 순수 E/P workspace bytes라고 귀속할 수 없다.
- 모든 비교에서 KV page 수는 동일했다. 줄어든 KV 때문에 shared가 적게 쓰는 결과가 아니다.
- `shared_ep`도 TensorRT context 자체를 하나로 합친 것이 아니다. E와 P의 실행 workspace를
  공유하여 서로 배타적으로 사용하고, D context/workspace는 별도로 유지하는 mode다.

## 5. Cosmos: 동일 predictor flag에서 workspace만 바꾼 수치 차이

아래 값은 `100 × (independent/shared_ep − 1)`이다. 처리량은 양수가 좋고 latency는 음수가 좋다.
반복 오차 범위는 아직 없으므로 작은 수치 차이로 우열을 확정하지 않는다.

| Pctl | Workload | tok/s Δ | TTFT mean Δ | TTFT p95 Δ | TPOT mean Δ | TPOT p95 Δ | E2E mean Δ | E2E p95 Δ | Peak Δ MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| on | balanced | -1.42% | -2.74% | +6.13% | +1.52% | +0.88% | +1.34% | +1.97% | +440 |
| off | balanced | -0.48% | -7.52% | -0.40% | +0.33% | +0.17% | -0.01% | +0.21% | +468 |
| on | vision-heavy | +81.54% | -43.15% | -45.02% | +499.52% | +566.76% | +22.47% | -40.64% | +440 |
| off | vision-heavy | +80.24% | -38.21% | -42.88% | +468.72% | +555.59% | +22.97% | -40.27% | +556 |
| on | multi-image | +37.97% | +6.54% | -31.48% | +11.31% | +14.64% | +8.72% | -20.86% | +440 |
| off | multi-image | +73.59% | -31.08% | -51.80% | +13.73% | +19.32% | -15.44% | -38.77% | +440 |

Vision-heavy에서 token/s가 80% 이상 증가해도 E2E 평균은 22–23% 나빠졌다. 모순이 아니다.
Throughput은 전체 trace drain 시간에 크게 좌우되지만 E2E 평균은 각 요청의 완료시간을 같은 비중으로
집계한다. 늦게 끝나는 vision tail을 앞당기는 동시에 이전에 빨리 끝나던 text를 길게 지연시킬 수 있다.

### 5.1 Vision-heavy의 요청 class별 latency

Cosmos trace는 text 16개, vision 48개다. 같은 campaign cell의 `by_request_class`를 사용했다.

| Mode | Pctl | Class / N | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| shared_ep | on | text / 16 | 61.955 | 127.186 | 11.510 | 12.847 | 719.485 | 791.162 |
| independent | on | text / 16 | 47.646 | 64.919 | 59.545 | 91.071 | 3297.111 | 3381.679 |
| shared_ep | on | vision / 48 | 3037.766 | 5486.471 | 7.401 | 8.741 | 3267.191 | 5699.043 |
| independent | on | vision / 48 | 1722.935 | 3031.413 | 47.522 | 81.419 | 3196.120 | 3352.871 |
| shared_ep | off | text / 16 | 91.830 | 169.697 | 11.641 | 13.546 | 752.383 | 827.190 |
| independent | off | text / 16 | 65.463 | 182.743 | 59.840 | 91.687 | 3329.701 | 3415.175 |
| shared_ep | off | vision / 48 | 3053.977 | 5563.988 | 7.267 | 8.521 | 3279.259 | 5775.960 |
| independent | off | vision / 48 | 1884.076 | 3156.871 | 43.451 | 79.792 | 3231.068 | 3428.023 |

Predictor-on의 text E2E mean은 719.49 → 3297.11 ms로 약 **4.58배**가 된다. Vision E2E mean은
3267.19 → 3196.12 ms로 약간 감소하고 p95는 5699.04 → 3352.87 ms로 크게 감소한다.
따라서 independent의 headline throughput만으로 promotion하면 resident decode 서비스 악화를
놓친다. 이 결과는 [노트 333](333-encoder-admission-mrope-lifetime-fix-20260926.md)에서 관찰한
"admission 확대가 vision tail과 resident D에 상반된 영향"과 일관되지만, 두 실험은 변경 대상이 달라
동일 causal A/B라고 합치지 않는다.

## 6. 측정 epoch의 dispatch/cohort와 실제 stream 활동

Warm-up을 제외하고 `PHASE_EPOCH`의 measurement epoch 1 이후만 집계했다.
E/P/D 셀은 `dispatch 수 / 평균 batch / 최대 batch`다. P+D 한 action은 P와 D에 각각 한 번 포함한다.

### 6.1 전체 성공 cell의 E/P/D formation

| Model | Mode | Pctl | Workload | E | P | D |
|---|---|---|---|---|---|---|
| Gemma | shared_ep | on | balanced | 0 / — / — | 42 / 2.000 / 8 | 300 / 17.920 / 24 |
| Gemma | shared_ep | off | balanced | 0 / — / — | 44 / 1.909 / 8 | 297 / 18.101 / 24 |
| Gemma | shared_ep | on | vision-heavy | 20 / 2.400 / 4 | 99 / 1.838 / 8 | 139 / 17.266 / 24 |
| Gemma | shared_ep | off | vision-heavy | 16 / 3.000 / 4 | 92 / 1.978 / 5 | 174 / 13.793 / 24 |
| Gemma | shared_ep | on | multi-image | 7 / 2.857 / 3 | 37 / 1.838 / 3 | 110 / 5.636 / 16 |
| Gemma | shared_ep | off | multi-image | 7 / 2.857 / 4 | 37 / 1.838 / 4 | 51 / 12.157 / 20 |
| Cosmos | shared_ep | on | balanced | 0 / — / — | 174 / 1.931 / 8 | 477 / 51.723 / 64 |
| Cosmos | shared_ep | off | balanced | 0 / — / — | 177 / 1.898 / 8 | 481 / 51.293 / 64 |
| Cosmos | independent | on | balanced | 0 / — / — | 177 / 1.898 / 8 | 480 / 51.400 / 64 |
| Cosmos | independent | off | balanced | 0 / — / — | 174 / 1.931 / 8 | 483 / 51.081 / 64 |
| Cosmos | shared_ep | on | vision-heavy | 16 / 3.000 / 4 | 48 / 1.375 / 7 | 500 / 4.800 / 18 |
| Cosmos | shared_ep | off | vision-heavy | 16 / 3.000 / 4 | 50 / 1.320 / 5 | 504 / 4.762 / 18 |
| Cosmos | independent | on | vision-heavy | 15 / 3.200 / 4 † | 43 / 1.535 / 8 | 83 / 28.916 / 56 |
| Cosmos | independent | off | vision-heavy | 16 / 3.000 / 4 † | 43 / 1.535 / 8 | 89 / 26.966 / 55 |
| Cosmos | shared_ep | on | multi-image | 2 / 2.500 / 4 | 4 / 1.250 / 2 | 62 / 2.500 / 4 |
| Cosmos | shared_ep | off | multi-image | 3 / 1.667 / 3 | 5 / 1.000 / 1 | 93 / 1.667 / 3 |
| Cosmos | independent | on | multi-image | 2 / 2.500 / 4 † | 4 / 1.250 / 2 | 32 / 4.844 / 5 |
| Cosmos | independent | off | multi-image | 2 / 2.500 / 4 † | 4 / 1.250 / 2 | 32 / 4.844 / 5 |

† **Instrumentation 차이:** Cosmos independent에는 `PHASE_ENCODER_METRIC` record가 없었다.
E를 0으로 간주하지 않고 `PHASE_SCHEDULER_EVENT`의 `event_kind=dispatch`, `phase=encoder`와
request membership으로 복구했다. 독립 경로에 E dispatch metric이 없는 상태에서 encoder GPU time,
queue wait 같은 다른 E metric까지 0이라고 채우면 안 된다. 해당 값이 필요하면 다른 유효한 timestamp
출처를 명시하거나 N/A로 남긴다. Balanced는 요청 자체가 text-only이고 실제 encoder dispatch도 없다.

Vision-heavy Cosmos의 E batch 수는 거의 유지되지만 D는 shared의 500/504회에서 independent의
83/89회로 줄었다. 동시에 평균 D batch는 4.8에서 27–29로 커졌다. 이 변화는 "동일 dispatch를
겹치기만 한" 실험이 아니라 **workspace mode가 허용하는 실행/준비 시점이 다음 cohort까지 바꾸는**
결과다. 더 큰 D batch와 적은 dispatch가 개별 token 간격 개선을 보장하지 않는 것은 5절에서 확인된다.

### 6.2 Activity mask와 E/P engine overlap

비트는 E=`0001`, P=`0010`, D=`0100`, C=`1000`이다. 아래 E∩P, E∩D, P∩D 비율은 해당 두
비트가 동시에 set인 **inclusive union 비율**이므로 합이 100%일 필요가 없고 triple 구간이 있으면
중복 집계된다. Idle은 계측 epoch에서 모든 bit가 0인 구간이다. SM utilization이 아니다.

| Model | Mode | Pctl | Workload | Idle % | E∩P envelope % | E∩D % | P∩D % | E/P engine overlap ms |
|---|---|---|---|---:|---:|---:|---:|---:|
| Gemma | shared_ep | on | balanced | 1.4046 | 0 | 0 | 10.5094 | 0 |
| Gemma | shared_ep | off | balanced | 1.3477 | 0 | 0 | 10.4583 | 0 |
| Gemma | shared_ep | on | vision-heavy | 1.9292 | 0.0038 | 17.9196 | 9.6520 | 0 |
| Gemma | shared_ep | off | vision-heavy | 1.6423 | 0 | 16.5161 | 8.0991 | 0 |
| Gemma | shared_ep | on | multi-image | 1.3414 | 0 | 12.8611 | 5.4989 | 0 |
| Gemma | shared_ep | off | multi-image | 1.9324 | 0 | 15.8894 | 8.1215 | 0 |
| Cosmos | shared_ep | on | balanced | 3.7006 | 0 | 0 | 27.9097 | 0 |
| Cosmos | shared_ep | off | balanced | 3.7134 | 0 | 0 | 29.4175 | 0 |
| Cosmos | independent | on | balanced | 3.8172 | 0 | 0 | 26.4868 | 0 |
| Cosmos | independent | off | balanced | 3.6538 | 0 | 0 | 26.5203 | 0 |
| Cosmos | shared_ep | on | vision-heavy | 2.8590 | 0 | 0.9453 | 0.2168 | 0 |
| Cosmos | shared_ep | off | vision-heavy | 2.8253 | 0 | 0.9295 | 0.7267 | 0 |
| Cosmos | independent | on | vision-heavy | 3.4287 | 6.0113 | 29.8284 | 13.0913 | 206.0831 |
| Cosmos | independent | off | vision-heavy | 3.2449 | 6.8789 | 31.2966 | 14.9039 | 238.6034 |
| Cosmos | shared_ep | on | multi-image | 1.9743 | 0 | 0 | 0 | 0 |
| Cosmos | shared_ep | off | multi-image | 2.1705 | 0 | 0 | 0 | 0 |
| Cosmos | independent | on | multi-image | 1.8801 | 12.0647 | 0 | 2.3829 | 58.4396 |
| Cosmos | independent | off | multi-image | 1.9517 | 12.0671 | 0 | 2.4732 | 58.3854 |

Engine overlap는 engine 실행을 감싸는 interval끼리의 교집합이며 envelope와 구분했다.
Shared Gemma의 0.0038% E∩P는 sampling 등을 포함한 phase envelope 때문에 생긴 것이고 E/P engine
workspace 동시 사용의 증거가 아니다. 반대로 independent Cosmos의 E/P engine overlap는 해당 mode에서
허용되는 정상적인 실행 자유도다. 관측한 global/vision action-fidelity violation counter는 모두 0이었다.

Copy bit가 0인 이 경로는 direct encoder output을 사용한다. 그 사실을 "GPU 내부의 모든 memory copy가
없다"로 바꾸어 말하지 않는다. 이 계측에 등록한 stream 작업만 mask에 들어간다.

## 7. Predictor on/off가 의미하는 것과 의미하지 않는 것

환경변수는 `TRT_EDGELLM_ENABLE_TRANSITION_PREDICTOR=0/1`이다. 다음 세 가지를 구분해야 한다.

1. `PhaseTransitionPredictor`는 decode enqueue-to-dispatch **queue residence**를 관측한다.
   이는 kernel/host handoff 비용이 아니고 현재 policy가 만든 waiting time이다.
2. `recommendedDecodeBurst()`와 `recommendedOverlapPrefillTokens()`는 deterministic advisory
   계산을 한다. Queue-residence posterior를 물리적 handoff 비용으로 다시 넣지 않는다. Global V3의
   action authority와 이 local advisory도 동일하지 않다.
3. Contextual Scalar pair RLS의 `observeContextualDirection()` 및 exact CUDA cost tracker는 별도 경로다.
   이 flag를 끈다고 Scalar RLS 전체가 없어지는 것은 아니다.

관련 구현 위치:

- `cpp/runtime/scheduling/phaseQueueScheduler.cpp`: constructor의 flag 입력,
  `effectiveDecodeBurstLimit()`, `effectiveOverlapPrefillTokens()`, `observeDispatch()`.
- `cpp/runtime/scheduling/phaseTransitionPredictor.cpp`: `recommendedDecodeBurst()` 및
  `recommendedOverlapPrefillTokens()`.
- `cpp/runtime/phase/policy/phaseTransitionPredictor.h`: queue state/label contract,
  uncalibrated `dispatchOverheadUs` 설명.

따라서 이 screen을 **"학습 on/off로 인한 성능 향상"**, **"RLS가 batching을 학습해서 개선"**이라고
요약하면 잘못이다. 동일 generic calibration 입력도 각 실행의 timing/formation까지 같게 만들지는
않는다. Flag가 바꾸는 advisory/관측 overhead와 실제 trajectory 변화, run-to-run 변동을 아직 분리하지
않았다. Gemma vision-heavy는 on 쪽이, multi-image는 off 쪽이 빠른 단일 결과만으로 workload별 flag를
선택하지 않는다.

## 8. Frozen vLLM 비교와 correctness 범위

vLLM은 이번 screen에서 새로 실행하지 않았다. 동일 trace provenance를 확인한 frozen 결과를 재사용했다.

- Gemma: `.local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12`.
- Cosmos: `.local/results/review-correction-20260926/cosmos-vllm-frozen-raw-corrected.json`,
  SHA256 `594383e90294d68c503e4b84bb4e9848b368af1a6c72d63175029587fc72eec8`.
  역사적 flattened summary의 mean/median 혼동을 원본 raw에서 바로잡은 별도 immutable artifact다.

모든 18개 row의 7개 지표별 절대값, baseline값, 상대변화는 `workspace-final-report.md`에 보존되어 있다.
핵심 비교는 다음과 같다.

- Gemma shared/on balanced: vLLM 대비 tok/s +56.17%, E2E mean -31.93%, p95 -34.10%지만
  TTFT p95는 +158.46%다. 이 screen에서 "모든 지표 우세"가 아니다.
- Gemma shared/on vision-heavy: tok/s -4.24%, E2E mean +11.49%; E2E p95만 -0.35%다.
- Gemma shared/on multi-image: tok/s -21.75%, E2E mean +25.02%, p95 +37.57%다.
- Cosmos independent/on vision-heavy: vLLM 대비 tok/s +23.32%, E2E mean -21.20%, p95 -20.28%이고
  이 row의 TTFT/TPOT 평균과 p95도 더 낮다. 그러나 shared current 대비 resident text regression이 크므로
  vLLM 우세만으로 current default promotion을 정당화하지 않는다.
- Cosmos independent/on balanced: tok/s -2.73%, E2E mean +0.18%, p95 +2.66%다.
- Cosmos independent/on multi-image: tok/s +24.39%, E2E mean -18.56%, p95 -19.45%지만
  TTFT mean은 +6.10%다.

완료 18개 output audit에서 HTTP error, 기대 output count 불일치, 불완전 token ID capture 등
정의된 integrity issue는 0이었다. First-EOS anomaly flag도 0이었다. 실패 6개는 별도 unresolved
startup failure로 남긴다. First-EOS가 없다는 것은 그 뒤 EOS가 없다는 뜻도, semantic correctness의
증명도 아니다. Fixed-length/ignore-EOS는 serving 부하 contract다.

각 cell이 1회이므로 raw exact token repeatability와 first-stop-prefix repeatability는 모두
`not_tested`다. 이전 aggregate에 `token_trace_deterministic=true`가 저장되어 있어도 1회 run hash로
determinism을 증명하지 않는다. 이번 screen은 mode 간 exact token identity gate도 대체하지 않는다.

## 9. 다음 결정을 위한 제한된 결론

이번 결과는 메모리 mode의 trade-off를 구체적으로 보여준다.

```text
shared E/P workspace
  → 낮은 peak memory, Gemma fixed-KV contract 실행 가능
  → E/P engine 배타 실행 + shared admission 경로
  → Cosmos resident D latency는 낮지만 vision tail이 길 수 있음

independent E/P/D workspace
  → Cosmos +440–556 MiB, Gemma startup OOM
  → 실제 E/P overlap 및 다른 downstream cohort 형성
  → Cosmos vision tail/throughput 개선, resident D latency 큰 악화
```

이는 policy가 resident decode continuity와 새 vision admission을 함께 조절해야 한다는 증거다.
단순히 allocator가 허용하는 한 E를 더 빨리 넣거나 workspace를 더 쓰는 것만으로는 충분하지 않다.
반대로 이 screen의 shared barrier를 "M-RoPE가 남아 있으므로 물리적으로 새 E가 불가능하다"고
정당화해서도 안 된다. 물리적 lease 안전성과 서비스 backpressure의 역할은 별개다.

현재 1회 screen만으로 default mode, KV 크기, workload별 predictor flag를 변경하지 않는다.
Full12 × 3회 primary 결과와 별도 memory/frontier 실험을 구분해 보존한다.

## 10. 후속 완료: shared E/P retained storage 1→2 screen

이 절부터는 1–9절의 24-cell matrix와 **별도 실행**이다. 기존 표를 덮어쓰지 않는다.
추가 분석 원본은 `.local/results/runtime-contract-revalidation-20260926/two-slab-final-analysis.md`다.
Raw root는 같은 상위 경로의 `two-slab-final-screen`, 7개 지표와 output audit은
`two-slab-final-report.{json,md}`, `two-slab-final-quality.{json,md}`에 있다.

### 10.1 동일한 것과 바뀐 것

두 모델 × balanced/vision-heavy/multi-image × 1회, 총 **6/6개 HTTP 실행이 완료**되었다.
즉시 single-storage 대조는 1–9절 `workspace-final-screen`의 `shared_ep-predictor-on` 6개다.
별도 반복 대조는 `full24-final-3x`에서 같은 6개 workload의 3회 결과를 사용했다.

세 campaign 모두 실제 executable은 source `43c680a`, binary SHA256 `aab91919…552b`, plugin
SHA256 `ddabc5df…d6c2`로 동일하다. Checkout은 screen과 two-slab이 `985ff1a`, 반복 대조가
`47a9cf7`이지만 checkout을 binary identity 대신 쓰지 않는다. 모델별 engine, vision, sidecar,
calibration, 일치하는 trace hash, runner와 container가 동일한 것을 확인했다.

Effective environment 차이는 **`TRT_EDGELLM_SHARED_EP_SINGLE_STORAGE=1 → 0`**뿐이다.
Startup log는 `serialized=1 max_retained_batches=1 → 2`로 해석된 것을 보여준다.
이는 bounded encoder-output storage를 두 개까지 retain하도록 허용한 것이지,
동시에 실행하는 E/P workspace arena를 두 개로 만든 것이 아니다. E/P workspace 배타 실행,
E+D/P+D 가능성, KV 192/256 pages, graph P0/D64, E4/P8/D24·64, direct output,
release-after-P와 `VISION_IDLE_SLABS=0`은 유지했다.

### 10.2 절대 7개 지표와 peak

모든 행은 1회, latency ms / memory MiB다.

| Model / workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cosmos / balanced | 4206.823 | 66.594 | 157.314 | 12.814 | 14.433 | 1157.095 | 1820.911 | 9347 |
| Cosmos / multi-image | 298.366 | 224.754 | 299.627 | 9.598 | 11.198 | 522.292 | 532.713 | 9347 |
| Cosmos / vision-heavy | 705.770 | 1236.625 | 2974.037 | 52.666 | 82.338 | 3218.020 | 3413.996 | 9383 |
| Gemma / balanced | 1214.173 | 153.803 | 617.299 | 15.462 | 16.825 | 1444.029 | 2115.437 | 9849 |
| Gemma / multi-image | 375.675 | 354.209 | 556.495 | 27.748 | 41.134 | 1214.392 | 1506.452 | 9857 |
| Gemma / vision-heavy | 543.081 | 341.634 | 639.695 | 32.819 | 48.187 | 1557.386 | 2044.942 | 9859 |

### 10.3 즉시 single-storage screen 대비

모든 값은 `100 × (two/single − 1)`이다.

| Model / workload | tok/s Δ | TTFT mean Δ | TTFT p95 Δ | TPOT mean Δ | TPOT p95 Δ | E2E mean Δ | E2E p95 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosmos / balanced | -1.21% | +5.52% | +4.16% | +1.13% | +2.07% | +1.40% | +2.11% |
| Cosmos / multi-image | +35.69% | -13.86% | -33.77% | +35.00% | +38.26% | +8.51% | -20.02% |
| Cosmos / vision-heavy | +80.00% | -46.09% | -45.65% | +524.89% | +564.06% | +22.35% | -40.05% |
| Gemma / balanced | +0.78% | +12.19% | +1.92% | -1.69% | -2.62% | -0.31% | -1.07% |
| Gemma / multi-image | +25.89% | -46.03% | -42.52% | +17.68% | +14.73% | -12.46% | -15.79% |
| Gemma / vision-heavy | +1.31% | +6.12% | -9.35% | -3.46% | -4.77% | -1.68% | -4.03% |

### 10.4 별도 3회 single-storage 대조 대비

Latency mean은 run mean의 산술평균, 그 외는 기존 report의 run 간 median 집계다.
대조가 3회여도 two-slab 쪽은 여전히 1회이므로 반복 통계 검증으로 승격하지 않는다.

| Model / workload | tok/s Δ | TTFT mean Δ | TTFT p95 Δ | TPOT mean Δ | TPOT p95 Δ | E2E mean Δ | E2E p95 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosmos / balanced | -1.35% | +1.08% | -3.50% | +1.15% | -0.23% | +1.33% | +1.53% |
| Cosmos / multi-image | +38.33% | -28.60% | -42.27% | +38.63% | +47.79% | -1.35% | -27.65% |
| Cosmos / vision-heavy | +80.75% | -45.99% | -45.77% | +522.73% | +566.10% | +22.49% | -40.20% |
| Gemma / balanced | +0.91% | +9.88% | +1.53% | -1.39% | -2.91% | -0.31% | -1.35% |
| Gemma / multi-image | +0.88% | +11.38% | +5.42% | -5.71% | -8.97% | -1.29% | -2.58% |
| Gemma / vision-heavy | +8.91% | -31.33% | -43.50% | -3.37% | -6.36% | -10.59% | -14.06% |

Gemma multi-image의 +25.89% headline은 즉시 single 1회와 비교한 값이다. 반복 single의 tok/s
범위는 369.89–377.34이고 two-slab 375.67은 그 안에 있다. 반복 대조에 대해서는 +0.88%뿐이며
TTFT mean은 오히려 +11.38%다. 어느 reference를 택하느냐에 따라 큰 개선으로 보이는 사례를
그대로 보존해야 한다. Gemma vision-heavy도 즉시 대조 대비 +1.31%와 반복 대조 대비 +8.91%를
구분한다. 동일 binary는 버전 차이를 제거하지만 GPU 시간적 상태/formation trajectory 변동까지
제거하지 않는다.

### 10.5 Frozen vLLM 대비

8절의 trace-matched frozen baseline 재사용이며 새 vLLM 측정이 아니다.

| Model / workload | tok/s Δ | TTFT mean Δ | TTFT p95 Δ | TPOT mean Δ | TPOT p95 Δ | E2E mean Δ | E2E p95 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cosmos / balanced | -2.52% | -40.75% | -38.07% | +5.05% | +6.25% | +0.24% | +2.80% |
| Cosmos / multi-image | +22.33% | -14.22% | -25.46% | -21.82% | -31.16% | -18.72% | -18.60% |
| Cosmos / vision-heavy | +22.28% | -24.17% | -16.09% | -19.17% | -31.81% | -21.28% | -19.49% |
| Gemma / balanced | +57.39% | +14.55% | +163.44% | -34.92% | -31.50% | -32.14% | -34.81% |
| Gemma / multi-image | -1.49% | +87.55% | +144.97% | -6.58% | +15.74% | +9.44% | +15.85% |
| Gemma / vision-heavy | -2.99% | +21.77% | +63.70% | +7.30% | +18.85% | +9.62% | -4.37% |

Cosmos multi-image/vision-heavy는 이 한 번의 vLLM 비교에서 7개 지표가 모두 좋다. 그러나 동일
current single-storage 대비 TPOT와 일부 mean E2E는 크게 악화되므로 두 비교를 함께 보여야 한다.

### 10.6 Class별 trade-off

Cosmos vision-heavy에서 two-slab의 text 16개 E2E mean은 **3320.880 ms**, p95는 **3415.836 ms**다.
즉시 single-storage의 719.485 / 791.162 ms와 비교하면 resident text가 상당히 지연된다.
Text TPOT mean/p95도 11.510 / 12.847 → **59.208 / 90.049 ms**다.

Vision 48개의 two-slab E2E mean/p95는 **3183.734 / 3387.806 ms**로 single의
3267.191 / 5699.043 ms보다 낮다. Vision TTFT mean/p95는 **1618.685 / 2983.564 ms**,
TPOT mean/p95는 **50.485 / 81.680 ms**다. 첫 토큰과 trace tail을 당기는 대신 token 간격이
커지는 상반된 효과가 있다. 이것은 all-request Pareto improvement가 아니다.

### 10.7 메모리와 실제 slab high-water의 한계

| Model / workload | Two peak MiB | Single screen | Single 3회 | 누적 slab allocations = reclaims | 누적 direct-output bytes | 최대 sampled logical payload bytes |
|---|---:|---:|---:|---|---:|---:|
| Cosmos / balanced | 9347 | 9299 | 9299 | 30 = 30 | 724598784 | 96403456 |
| Cosmos / multi-image | 9347 | 9299 | 9299 | 34 = 34 | 772898816 | 91160576 |
| Cosmos / vision-heavy | 9383 | 9299 | 9299 | 47 = 47 | 1183285248 | 121569280 |
| Gemma / balanced | 9849 | 9849 | 9849 | 11 = 11 | 11182080 | 3194880 |
| Gemma / multi-image | 9857 | 9853 | 9853 | 18 = 18 | 30277632 | 7188480 |
| Gemma / vision-heavy | 9859 | 9853 | 9853 | 31 = 31 | 56549376 | 8005632 |

Cosmos peak 증가는 +48/+48/+84 MiB, Gemma는 +0/+4/+6 MiB다.
**실제 동시 physical slab object high-water는 이 바이너리에서 export하지 않는다.** 설정 cap 2는
관측된 동시 live slab 수 2라는 뜻이 아니다. 누적 allocation/reclaim은 calibration을 포함한 process
전체 수치이며 동시 resident 수나 measurement-only allocation 수가 아니다.

`PhaseVisionAdapter::retainedStorageBatches()`는 external owner가 있는 storage object를 세지만
그 값을 gauge/high-water로 출력하지 않는다. Decision의 `vision_payload_bytes`는 logical payload
accounting이며 alias/view와 metadata가 포함될 수 있어 unique physical capacity나 object 수의
증명이 아니다. Terminal retained/downstream gauge가 0인 것도 실행 중 retain이 없었다는 뜻이 아니다.

6개 terminal summary 모두 direct output D2D operation/bytes 0, idle slab/bytes 0,
allocations=reclaims, 누적 released storage bytes=direct-output bytes였다. 이는 정상 drain의
lifetime 관찰이지 arbitrary cancellation leak-proof나 sanitizer 통과가 아니다.

### 10.8 Fidelity와 output 검사

기존 validator로 compressed log의 action-fidelity, GPU interval, residual contract를 검사했다.
Compact decision schema를 허용했으며 다음 count는 calibration을 포함한 process 전체다.

| Model / workload | Events | Dispatch | Completion | Fidelity failure | Unmatched dispatch/completion | Validator error |
|---|---:|---:|---:|---:|---|---:|
| Cosmos / balanced | 3458 | 1232 | 1232 | 0 | 0 / 0 | 0 |
| Cosmos / multi-image | 1620 | 593 | 593 | 0 | 0 / 0 | 0 |
| Cosmos / vision-heavy | 1935 | 709 | 709 | 0 | 0 / 0 | 0 |
| Gemma / balanced | 1792 | 613 | 613 | 0 | 0 / 0 | 0 |
| Gemma / multi-image | 1045 | 361 | 361 | 0 | 0 / 0 | 0 |
| Gemma / vision-heavy | 1514 | 522 | 522 | 0 | 0 / 0 | 0 |

Output audit는 6/6 cell, 505 requests, integrity issue 0, first-EOS anomaly 0이다.
Semantic은 `not_evaluated`, cross-repeat token identity는 각 1회이므로 `not_tested`다.
Exact greedy parity/cancellation/memcheck gate 통과로 바꿔 표현하지 않는다.

**판정: 기본 single-storage 유지.** 두 slab는 추가 mechanism 후보로 남긴다. 실제 object/byte
high-water와 완료 후 reclaim boundary를 계측하고, same-contract alternating 1↔2 순서로 최소 3회,
가능하면 5회 비교가 필요하다. Cosmos resident D 회귀를 먼저 해석한 뒤 Full12 양모델과
candidate 계약의 cancellation/sanitizer 검증으로 확대한다. `5520216` 제거 후보를 자동으로 다시
적용하지 않으며, 앞으로 acceptance된 admission 수정이 있다면 별도 source/engine identity로 검증한다.

## 11. 후속 완료: 실제 two-profile vision engine의 tiered mode

### 11.1 Contract

Raw root는 `tiered-final-screen` 및 `tiered-final-vision-screen`이다. 각각
`tiered-final-report/quality.{json,md}`, `tiered-final-vision-report/quality.{json,md}`가 있다.
모두 `.local/results/runtime-contract-revalidation-20260926/` 아래에 있다.

앞 절과 같은 `43c680a` 바이너리/동일 plugin, Gemma text engine, KV **192 pages**, calibration 49개,
P8/P128/D24, graph cap P0/D64를 유지했다. 다만 **vision engine은 two-profile engine으로 변경**했다.

- Path: `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/visual-tiered-e3-e4-soft280/visual`
- Vision SHA256: `8ed5cc8e7702b5d8264164418ba1fce19ffc61a0944e0e174822f24f4b467e7e`.
- 같은 two-profile engine을 shared_ep / tiered_ep / independent 세 mode에 모두 사용했다.

따라서 이 세 mode 간 비교는 같은 vision engine이다. 그러나 3절 one-profile vision engine의
shared 결과와 숫자만 바로 비교해 workspace 효과로 귀속하지 않는다.

### 11.2 Balanced: HTTP 2/3 성공이지만 tiered graph contract는 실패

| Mode | HTTP 결과 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared_ep | 완료 | 1208.319 | 148.882 | 634.002 | 15.631 | 16.753 | 1455.211 | 2145.066 | 9769 |
| tiered_ep | 완료, graph OOM fallback | 1186.330 | 134.809 | 612.324 | 16.041 | 17.341 | 1475.259 | 2189.889 | 9871 |
| independent | Startup OOM | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

| Mode | P entries / captures | D entries / captures | D hits / misses | Graph OOM warning | 판정 |
|---|---:|---:|---|---:|---|
| shared_ep | 0 / 0 | 24 / 24 | 682 / 0 | 0 | 이 balanced cell graph contract 유지 |
| tiered_ep | 0 / 0 | **11 / 11** | 559 / **131** | **13** | D24 coverage 불충족; diagnostic only |
| independent | N/A | N/A | N/A | Startup allocation OOM | serving 미시작 |

Tiered는 `instantiateCudaGraph` OOM 13회 후 일부 decode shape가 eager fallback으로 실행되었다.
따라서 HTTP aggregate가 생겼더라도 **같은 graph coverage의 성공 결과가 아니다.** 위 7개 serving
지표는 실패를 숨기지 않기 위한 diagnostic 기록이다. Tiered overhead나 gain의 공정한 primary
비교로 채택하지 않는다. Runner의 HTTP complete count와 추가 graph-health gate는 구분한다.

### 11.3 실제 vision-heavy: 0/3 완료, 실패 원인이 서로 다름

| Mode | 관측 실패 | 시점 / 해석 |
|---|---|---|
| shared_ep | `Visual optimization profile 1 has no valid context memory` | Profile 1 선택 시 memory binding 누락; 통합 correctness 문제 |
| tiered_ep | Graph OOM 13회 이후 `cudaMalloc` OOM | 서버 시작/graph fallback 후 실제 vision 처리 중 allocation 실패 |
| independent | `cudaMalloc` OOM | Startup 단계에서 실패 |

세 mode 모두 `aggregate.json`이 없어 7개 serving 지표는 N/A다. Shared 실패는 OOM과 다르다.
Two-profile engine의 profile 1 context memory binding 문제이며, 이 결과로 shared가 물리적으로
해당 engine을 실행할 수 없다고 결론 내리지 않는다. Balanced는 vision workload가 아니므로 이
profile 선택 결함을 노출하지 못한 것이다.

**수정 후 재검증 pending:** shared multi-profile memory binding 수정은 별도 진행 중이다.
수정 완료/빌드/재실행 전에는 해결됐다고 기재하지 않고, 위 실패 로그는 보존한다. 새 성공 여부는
새 source/binary identity와 같은 two-profile engine의 actual vision trace로 추가해야 한다.

## 12. 후속 완료: P4/D8 graph-cache diagnostic

### 12.1 실행 범위

Raw root는 **`workspace-graph-p4-d8-corrected`**, report는 `workspace-graph-final-report.{json,md}`,
audit는 `workspace-graph-final-quality.{json,md}`다. 모두 같은 상위 결과 디렉토리 아래에 있다.
이름이 비슷한 이전 `workspace-graph-p4-d8` 결과를 최종 값으로 사용하지 않는다.

여기서 **P4/D8은 최대 graph cache entry 수**다. Prefill batch 4 / decode batch 8 실험이 아니다.
실제 batch cap은 Gemma P8/D24, Cosmos P8/D64이며 KV192/256 pages, one-profile vision engine,
same `43c680a` binary/plugin/calibration을 유지했다. Mode는 shared_ep, predictor-on이다.
Primary와 달라진 graph cap은 P0/D64 → **P4/D8**이다.

두 모델의 balanced 한 번씩, **2/2 HTTP 완료**다. 다른 workload나 반복 gate를 통과한 것은 아니다.

| Model | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma | 1182.171 | 136.092 | 603.418 | 16.057 | 17.547 | 1474.591 | 2192.311 | 9773 |
| Cosmos | 4115.952 | 70.044 | 164.570 | 13.012 | 14.564 | 1177.883 | 1849.070 | 9193 |

### 12.2 실제 graph 사용과 health

최종 count는 calibration을 포함한 process 전체다.

| Model | P entries / captures | P hits / misses | D entries / captures | D hits / misses | Evictions |
|---|---:|---|---:|---|---:|
| Gemma | 4 / 4 | 124 / 134 | 8 / 8 | 436 / 201 | 0 |
| Cosmos | 4 / 4 | 468 / 379 | 8 / 8 | 774 / 1110 | 0 |

두 log 모두 CUDA error/OOM 및 action-fidelity violation 0이다. 이 경우의 miss는 P4/D8 cap을
명시적으로 줄여 capture되지 않은 shape를 eager로 실행한 것이며, 11절 tiered의 OOM-induced
coverage 손실과 다르다. 관측된 graph entries와 정상 종료는 이 두 targeted 경로의 검증이다.
모든 profile/shape 또는 cancellation에서 graph lifetime 문제가 없다는 증명은 아니다.

Output audit 352 requests의 integrity issue와 first-EOS anomaly는 0이다. Semantic 미평가,
단일 repeat exact identity 미검증 상태는 동일하다.

### 12.3 같은 screen의 P0/D64 balanced 대비

| Model | tok/s Δ | TTFT mean Δ | TTFT p95 Δ | TPOT mean Δ | TPOT p95 Δ | E2E mean Δ | E2E p95 Δ | Peak Δ MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma | -1.87% | -0.73% | -0.37% | +2.09% | +1.56% | +1.80% | +2.53% | -76 |
| Cosmos | -3.34% | +10.98% | +8.96% | +2.69% | +2.99% | +3.22% | +3.69% | -106 |

Peak는 줄었지만 P graph를 추가하는 동시에 D graph coverage를 줄인 복합 변경이다.
따라서 "P graph가 성능을 악화시켰다"거나 "P graph를 켜면 메모리가 감소한다"고 단독 원인으로
귀속하면 안 된다. 이 targeted diagnostic만으로 primary P0/D64를 변경하지 않는다.

## 13. 후속 판정과 아직 남은 gate

- Two-slab는 fixed-KV 조건에서 실행 가능하지만 Cosmos resident decode/mean E2E regression이 크다.
  유망한 throughput 수치만으로 default 승격하지 않는다.
- Genuine tiered는 balanced HTTP 성공과 graph contract 성공이 다르다는 것을 드러냈다. 실제
  vision-heavy는 현재 실패이며 shared profile binding 수정 후 재검증이 남았다.
- P4/D8 graph path는 두 balanced cell에서 정상 동작했으나 decode coverage 감소의 latency 비용이 있다.
  전 workload와 반복 검증이 아니다.
- 1–9절 workspace screen, 10절 two-slab, 11절 two-profile engine, 12절 graph cache screen은
  서로 다른 contract다. 한 표의 최고값을 골라 "단일 설정의 Full12 개선"이라고 합성하지 않는다.
- 실제 unique storage high-water, 반복 exact output/semantic 평가, candidate별 cancellation/sanitizer,
  새로운 shared multi-profile binding 수정 검증은 별도 gate로 남는다.

## 14. 수정 후 완료: shared E/P의 실제 multi-profile binding 재검증

이 절은 11.3절 및 13절에 남겨 둔 **shared multi-profile binding 재검증 pending을 해소**한다.
앞의 실패 결과를 삭제하거나 성공으로 다시 분류하지 않는다. Tiered/independent OOM 및 나머지
promotion gate가 해결됐다는 뜻도 아니다.

### 14.1 수정 범위와 재현 identity

기존 shared helper는 선택된 profile 0의 workspace만 할당·연결했다. Gemma vision runner는 실제
image shape에 따라 profile 1을 자동 선택하므로, profile 1의 memory binding이 비어 있는 것이
11.3절 실패의 원인이었다. 수정은 다음 두 동작으로 한정했다.

- 여러 vision profile이면 `max(P workspace, all-vision-profile workspace maximum)` 크기로
  shared arena를 만들고, 기존 `MultimodalRunner::setContextMemory()`로 **모든 profile**에 연결한다.
- 기존 `setContextMemoryForProfile(visionProfile, ..., setupStream)` 호출은 유지하여 초기 선택
  profile과 setup 순서를 보존한다. Arena 해제 전 P graph-cache invalidation도 유지한다.

단일-profile 엔진은 이전과 같은 profile별 size query, allocation, binding 경로를 사용한다.
KV pool, P/D batch cap, graph cap, scheduler policy, tiered allocation은 변경하지 않았다.
따라서 primary 72회의 single-profile 실행 mechanism은 유지하지만, **그 72회를 새 바이너리로
다시 실행했다고 쓰지는 않는다.** 이번 결과는 새 바이너리와 two-profile 엔진의 targeted 2-cell 검증이다.

| 항목 | 값 |
|---|---|
| Source / binary source commit | `e8164e04758d8dde84ca9a43a6873627bc43bd92` |
| Binary SHA256 | `c07a91bc1164fa66e00daaaa09800e42607e84ae92e3f3591528e9aee702f79d` |
| Plugin SHA256, 변경 없음 | `ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2` |
| Gemma text engine SHA256 | `fef5210c22b0ceb064cdce07658f6dace7737d66cad7e35a3f625602f2c9405e` |
| Two-profile vision engine SHA256 | `8ed5cc8e7702b5d8264164418ba1fce19ffc61a0944e0e174822f24f4b467e7e` |
| Runtime | shared_ep, predictor-on, retained storage 1, P8/P128/D24, KV192 pages |
| Graph cap / calibration | P0/D64 / 동일 generic 49 requests |
| Scope | Gemma balanced + vision-heavy, 각 1회, **2/2 완료**, 총 serving 128 requests |

Raw root는 `.local/results/runtime-contract-revalidation-20260926/shared-all-profiles-fixed`이다.
같은 상위 디렉토리의 `shared-all-profiles-report.{json,md,manifest.json}` 및
`shared-all-profiles-quality.{json,md}`를 사용한다. Manifest의 dirty 항목은 당시 작성 중인 notes이며,
command, engine/config/trace/calibration hash와 binary source commit은 별도로 기록되어 있다.

### 14.2 Profile 1이 실제 실행됐는가

**그렇다. 단순한 HTTP 성공이나 profile 1의 사전 binding만을 근거로 하지 않는다.**
`vision-heavy/run-001/gateway.log.gz`의 압축 해제 후 line 번호로 다음 증거를 연결했다.

1. Line 13260: serving 중 TensorRT의 `Switching optimization profile from: 0 to 1` 로그.
   Startup의 P/D context 설정 로그(line 51)와 별개의 전환이다.
2. 첫 serving E batch는 request IDs `[4, 0, 8, 12]`, batch index 9, E4이다. 앞의 E batch 1–8은
   calibration이다. 이 네 요청의 실제 JPEG 크기와 `gemma4ResizeTarget()` 규칙을 대조하면 다음과 같다.

| Request | Image | 원본 W×H | Resize W×H | Soft tokens | Patch tokens |
|---|---|---:|---:|---:|---:|
| 4 | red_panda.jpeg | 1000×747 | 912×672 | 266 | 2394 |
| 0 | woman_and_dog.jpeg | 2048×1365 | 960×624 | 260 | 2340 |
| 8 | giant_panda.jpeg | 1000×1000 | 768×768 | 256 | 2304 |
| 12 | database_er.jpeg | 1790×294 | 1968×288 | 246 | 2214 |
| 합계 | 동일 E batch | — | — | **1028** | **9252** |

Profile 0의 최대치는 soft 840 / patch 7560이고 profile 1은 soft 1120 / patch 10080이다.
따라서 이 batch는 profile 0에 들어갈 수 없고 profile 1이 필요하다. 여기서 1028/9252는
입력 이미지와 현재 resize/3×3 pooling 규칙에서 **재구성한 shape**다. Gemma의 해당
`PHASE_ENCODER_METRIC.input_tokens` 필드는 실제로 0으로 기록되어 있어, 그 필드가 1028을
측정했다고 주장하지 않는다.

3. Line 16764의 encoder dispatch와 line 16845의 encoder completion은 동일 execution ID
   `4611686018427387913` 및 같은 request IDs를 갖는다. Completion status는 `success`,
   공통 GPU epoch의 실행시간은 **59.923 ms**다.
4. Line 16899의 `PHASE_ENCODER_METRIC`도 batch index 9 / E4의 실행시간 **59.930 ms**를 기록한다.
   이후 serving 중 1→0→1 profile 전환과 다음 encoder completions도 이어진다.

따라서 기존에 실패했던 큰 profile의 실제 GPU enqueue와 completion까지 검증했다.
첫 큰-profile preparation은 약 **645.374 ms**, preparation 포함 GPU interval은 약 **705.745 ms**다.
이는 순수 encoder execution 59.930 ms와 구분해야 한다. 이 한 번의 관측만으로 preparation 지연을
workspace binding 비용이나 특정 lazy initialization 하나에 전부 귀속하지 않는다.

### 14.3 Graph coverage, 오류 및 메모리

Graph count는 calibration을 포함한 process 전체다. P0는 의도적으로 prefill graph를 끈 contract다.

| Workload | P entries / captures | P hits / misses | D entries / captures | D hits / misses | Evictions | Peak MiB |
|---|---:|---|---:|---|---:|---:|
| balanced | 0 / 0 | 0 / 225 | **24 / 24** | **770 / 0** | 0 | 9851 |
| vision-heavy | 0 / 0 | 0 / 274 | **24 / 24** | **633 / 0** | 0 | 9865 |

두 log 모두 CUDA error, OOM warning, invalid profile-memory 오류가 없다. P/D 및 vision의
action-fidelity violation counter도 모두 0이다. 따라서 tiered의 13회 graph OOM과 달리,
이번 shared 실행은 **D1–D24 사전 graph coverage를 보존했고 관측된 D miss도 0**이다.
모든 graph eviction/lifetime 경로나 모든 shape의 correctness를 증명하는 것은 아니다.

두 run의 초기 arena 로그는 `total=313528832 prefill=183647744 vision=313528832` bytes다.

| 비교 대상 | Shared arena bytes | 이번 수정 결과와의 관계 |
|---|---:|---|
| 결함이 있던 two-profile profile-0-only binding | 226679296 | 모든 profile 지원에 필요한 크기보다 작았음 |
| 수정된 all-profile shared binding | **313528832** | 올바른 all-profile maximum |
| Primary 72회의 원래 single-profile vision engine | 313528320 | 수정된 two-profile arena와 **512 bytes** 차이 |

결함 상태 대비 약 82.826 MiB 증가는 누락했던 profile-1 용량을 반영한 것이며, primary single-profile
대비 동일한 규모의 메모리 증가가 아니다. Process peak 차이를 arena 크기 하나로 환원하지도 않는다.

GPU total은 10240 MiB지만 driver reservation 약 365–366 MiB를 빼면 usable은 약 9874–9875 MiB다.
따라서 peak 시 추정 headroom은 balanced **23–24 MiB**, vision-heavy **9–10 MiB**뿐이다.
이는 usable-minus-observed-used 추정치이지 peak 순간 `memory.free`의 동시 측정치가 아니다.
OOM이 없었더라도 **512 MiB headroom gate는 통과하지 못했고**, 안정적인 production 여유가
확보됐다고 결론 내릴 수 없다.

### 14.4 7개 serving 지표 및 기존 결과와의 비교

Latency 단위는 ms이며 두 Current 행 모두 1회다. Frozen vLLM은 동일 trace의 기존 1회 결과를
재사용했다. 새로운 vLLM 측정이나 confidence interval은 없다.

| Workload / runtime | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced / 수정 Current | 1223.462 | 138.149 | 609.822 | 15.453 | 16.387 | 1428.662 | 2109.720 |
| balanced / frozen vLLM | 771.458 | 134.269 | 234.327 | 23.758 | 24.562 | 2128.027 | 3244.795 |
| Current Δ vs vLLM | +58.59% | +2.89% | +160.24% | -34.96% | -33.28% | -32.86% | -34.98% |
| vision-heavy / 수정 Current | 517.171 | 441.014 | 1010.407 | 33.685 | 51.862 | 1668.627 | 2336.417 |
| vision-heavy / frozen vLLM | 559.827 | 280.565 | 390.784 | 30.588 | 40.544 | 1420.679 | 2138.326 |
| Current Δ vs vLLM | -7.62% | +57.19% | +158.56% | +10.12% | +27.92% | +17.45% | +9.26% |

Primary `full24-final-report.json`의 single-profile 3회 집계와 비교하면 아래와 같다.
집계는 latency mean의 mean-of-run-means, throughput/p95의 repeat median이며, runner summary의
모든 항목 단순 평균을 대신 사용하지 않았다. **One-profile/old binary 3회 대 two-profile/new binary
1회**이므로 binding 수정만의 causal speedup이나 반복 gate로 해석하지 않는다.

| Workload | tok/s Δ | TTFT mean Δ | TTFT p95 Δ | TPOT mean Δ | TPOT p95 Δ | E2E mean Δ | E2E p95 Δ | Peak Δ MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | +1.68% | -1.30% | +0.30% | -1.45% | -5.43% | -1.37% | -1.62% | +2 |
| vision-heavy | +3.72% | -11.35% | -10.75% | -0.82% | +0.78% | -4.20% | -1.81% | +12 |

Output audit는 128 requests의 capture integrity issue 0, first-EOS anomaly 0이다.
**Semantic correctness는 미평가**, 단일 repeat이므로 exact repeat identity도 `not_tested`다.

이번 결론은 **shared multi-profile binding correctness와 targeted graph contract 복구**다.
Vision-heavy는 여전히 vLLM보다 7개 serving 지표 모두 나쁘고 headroom도 더 얇다. 따라서 이 수정은
필요한 기능 결함 해결이지만, tiered 정책 승격이나 전 workload 성능 개선의 증거는 아니다.

## 15. 최종 single-profile 바이너리 screen 및 fresh ABBA 비교

### 15.1 무엇을 검증했고, 무엇을 검증하지 않았는가

14절의 multi-profile 기능 수정이 기존 single-profile 경로를 바꾸지 않았는지 확인하기 위해
최종 `e8164e0` 바이너리로 두 모델 × 3 workload screen을 추가하고, Gemma의 큰 변동 두 workload는
이전 `43c680a` 바이너리를 다시 실행하여 ABBA 순서로 비교했다.

- Supplementary screen: `single-profile-final-screen`, 두 모델 balanced/vision-heavy/multi-image, 총 6회.
- Fresh ABBA: `paired-profile-guard-old-a` → `new-a` → `new-b` → `old-b`,
  각 단계에서 Gemma vision-heavy와 multi-image를 실행. **8/8 완료**, workload/바이너리당 2회.
- Old는 source `43c680a2a7e4dea4922c33e9c61f93628b6a4d95`,
  binary SHA256 `aab919194709253bdc377830653e1c201abf4a0b34967469cf0db0832cb1552b`.
- New는 source `e8164e04758d8dde84ca9a43a6873627bc43bd92`,
  binary SHA256 `c07a91bc1164fa66e00daaaa09800e42607e84ae92e3f3591528e9aee702f79d`.
- Plugin SHA256은 양쪽 모두 `ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2`.
  8개 `contract.json`의 effective environment, model engine/config/sidecars/vision/calibration/trace identity,
  runner identity를 비교하여 동일함을 확인했다. checkout HEAD가 아니라 실제 launched binary identity로 나눈다.
- Shared E/P, retained storage 1, predictor-on, 동일 P0/D64 graph 설정이다. Two-slab/독립 workspace나
  새로운 vision engine으로 바꾼 비교가 아니다.

`git diff 43c680a e8164e0 -- cpp examples`의 실행 코드 차이는
`independentEngineExecutorPair.{cpp,h}`의 all-vision-profile workspace initialization뿐이다.
Single-profile일 때 기존 profile별 query/binding 경로를 유지하고 serving hot path를 추가하지 않았다.
그러나 이것을 **성능 동일성이 실증됐다는 증거로 사용하지 않는다**.
Runtime trajectory, 측정 변동 및 다른 잠재 원인을 분리해야 하며, 아래 나쁜 결과도 그대로 남긴다.

Latency는 ms. 반복 집계는 기존 canonical `full24-final-report.json`과 똑같이
**mean 항목은 mean-of-run-means, throughput/p95는 repeat median**을 사용한다.
Runner의 전체 항목 산술평균을 canonical 3회 집계 대신 사용하지 않는다.
Fresh old/new가 각 2회라 여기서는 median과 mean의 수치가 같지만, 정의는 구분한다.
One-off supplementary screen은 fresh new 2회와 합치지 않는다.

### 15.2 Supplementary 최종 screen 6개 결과

각 행 1회이며, throughput만 골라 비교하지 않는다.

| 대상 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cosmos / balanced | 4315.76 | 64.11 | 152.27 | 12.43 | 13.98 | 1120.09 | 1736.87 | 9299 |
| cosmos / multi-image | 219.90 | 261.61 | 451.79 | 7.11 | 8.09 | 482.00 | 665.57 | 9299 |
| cosmos / vision-heavy | 408.12 | 2140.79 | 5286.54 | 8.93 | 13.76 | 2504.64 | 5510.04 | 9299 |
| gemma / balanced | 1206.04 | 146.93 | 617.74 | 15.64 | 16.99 | 1454.50 | 2135.54 | 9849 |
| gemma / multi-image | 310.17 | 439.27 | 974.61 | 30.38 | 42.11 | 1381.11 | 1580.70 | 9853 |
| gemma / vision-heavy | 487.09 | 488.63 | 1379.32 | 34.84 | 45.55 | 1784.59 | 3923.93 | 9853 |

**Gemma multi-image 310.17 tok/s, vision-heavy E2E p95 3923.93 ms도 포함한다.**
두 수치를 제외하거나 다른 run의 좋은 값으로 덮어쓰지 않는다.

Canonical primary 3회 기준에 대한 변화율은 다음과 같다.
양수 throughput은 유리하고 양수 latency는 불리하다.

| 대상 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| cosmos / balanced | +1.21% | -2.68% | -6.60% | -1.91% | -3.33% | -1.91% | -3.15% |
| cosmos / multi-image | +1.95% | -16.89% | -12.95% | +2.68% | +6.82% | -8.96% | -9.61% |
| cosmos / vision-heavy | +4.52% | -6.50% | -3.60% | +5.63% | +11.31% | -4.67% | -3.48% |
| gemma / balanced | +0.23% | +4.97% | +1.60% | -0.23% | -1.95% | +0.42% | -0.41% |
| gemma / multi-image | -16.71% | +38.13% | +84.63% | +3.24% | -6.82% | +12.26% | +2.22% |
| gemma / vision-heavy | -2.31% | -1.78% | +21.84% | +2.58% | -11.48% | +2.46% | +64.90% |

6개 모두 같은 모델/workload의 primary peak와 동일했다. 메모리/엔진 contract를 바꿔 성능을 개선한
실험은 아니다. Gemma multi-image는 canonical throughput 대비 **-16.71%**, E2E mean **+12.26%**이고,
vision-heavy E2E p95는 **+64.90%**다. 따라서 이 screen 하나로 회귀 없음/3% parity를 선언할 수 없다.

### 15.3 Fresh ABBA 8회 원시 결과 — 나쁜 run 포함

아래는 각 raw `client/run-001/summary.json`에서 직접 읽은 값이다.
모든 행 GPU peak는 **9853 MiB**다.

| 대상 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| old-a / vision-heavy | 402.61 | 1010.25 | 1984.51 | 31.02 | 49.25 | 2148.12 | 3522.69 |
| old-a / multi-image | 384.63 | 299.15 | 512.25 | 28.74 | 44.65 | 1190.24 | 1522.10 |
| new-a / vision-heavy | 504.76 | 482.39 | 1118.89 | 33.25 | 50.91 | 1690.47 | 2221.91 |
| new-a / multi-image | 298.26 | 651.18 | 920.70 | 23.41 | 34.25 | 1376.73 | 1761.43 |
| new-b / vision-heavy | 543.71 | 265.67 | 518.74 | 34.68 | 50.74 | 1543.18 | 2073.12 |
| new-b / multi-image | 378.64 | 306.66 | 523.75 | 29.51 | 45.63 | 1221.41 | 1547.28 |
| old-b / vision-heavy | 502.25 | 468.91 | 1127.81 | 34.40 | 52.50 | 1714.24 | 2324.58 |
| old-b / multi-image | 376.66 | 311.30 | 539.38 | 29.45 | 45.59 | 1224.24 | 1535.19 |

### 15.4 Fresh old 2회 대 fresh new 2회 집계

Supplementary screen과 historical primary는 이 집계에 넣지 않았다.

| 대상 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision-heavy / old 2회 | 452.43 | 739.58 | 1556.16 | 32.71 | 50.87 | 1931.18 | 2923.63 |
| vision-heavy / new 2회 | 524.23 | 374.03 | 818.82 | 33.96 | 50.83 | 1616.83 | 2147.51 |
| vision-heavy / new Δ vs old | +15.87% | -49.43% | -47.38% | +3.83% | -0.09% | -16.28% | -26.55% |
| multi-image / old 2회 | 380.65 | 305.23 | 525.81 | 29.10 | 45.12 | 1207.24 | 1528.64 |
| multi-image / new 2회 | 338.45 | 478.92 | 722.22 | 26.46 | 39.94 | 1299.07 | 1654.36 |
| multi-image / new Δ vs old | -11.09% | +56.91% | +37.35% | -9.08% | -11.48% | +7.61% | +8.22% |

관측 결과는 한 방향으로 일치하지 않는다.

- **Vision-heavy:** new throughput +15.87%, TTFT mean -49.43%, E2E mean -16.28%, E2E p95 -26.55%.
  하지만 TPOT mean은 +3.83%. Old-a 자체가 402.61 tok/s 및 E2E mean 2148.12 ms로 나빴다.
- **Multi-image:** new throughput **-11.09%**, TTFT mean **+56.91%**, E2E mean **+7.61%**,
  E2E p95 **+8.22%**. TPOT는 좋아졌다. New-a 298.26 tok/s를 지우면 안 된다.
- Old-b와 new-b multi-image는 각각 376.66/378.64 tok/s로 가깝지만, 좋은 쌍만 골라 parity를
  주장하지 않는다. Old-a/new-a의 반대 방향도 전체 결과에 포함한다.
- **이 2회 집계는 3% non-regression gate 통과가 아니다.** 표본이 적고 초기 trajectory의 변동이 크다.
  Multi-image의 관측 회귀는 명시적으로 남으며, 단일-profile 코드 경로가 유지됐다는 이유만으로
  측정값을 noise로 확정하지 않는다. 반대로 이 결과만으로 all-profile initialization 수정이
  serving regression의 직접 원인이라고 확정할 수도 없다.

### 15.5 Historical primary와 fresh old/new를 섞지 않고 비교

Primary 3회 기준값은 다음과 같다. 이 표는 runner 평균이 아니라 canonical report 값이다.

| 대상 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma vision-heavy / primary 3회 | 498.63 | 497.48 | 1132.11 | 33.96 | 51.46 | 1741.81 | 2379.54 |
| Gemma multi-image / primary 3회 | 372.42 | 318.02 | 527.86 | 29.43 | 45.18 | 1230.27 | 1546.38 |

Fresh 집계의 primary 대비 변화율:

| 대상 | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vision-heavy / fresh old 2회 | -9.27% | +48.67% | +37.46% | -3.70% | -1.14% | +10.87% | +22.87% |
| vision-heavy / fresh new 2회 | +5.13% | -24.82% | -27.67% | -0.01% | -1.23% | -7.18% | -9.75% |
| multi-image / fresh old 2회 | +2.21% | -4.02% | -0.39% | -1.12% | -0.15% | -1.87% | -1.15% |
| multi-image / fresh new 2회 | -9.12% | +50.59% | +36.82% | -10.10% | -11.61% | +5.59% | +6.98% |

Historical same-binary 변동도 참고하되, 나쁜 final run을 없애는 근거로 삼지 않는다.

- Primary Gemma vision-heavy E2E p95의 3회 범위는 **2335.08–4008.98 ms**다.
  Supplementary new 3923.93 ms는 그 범위 안이지만 primary median 대비 +64.90%라는 사실은 유지한다.
- 이전 `workspace-final-screen`의 old Gemma multi-image도 **298.41 tok/s**였다.
  따라서 298–310 tok/s 정도의 trajectory가 new binary에서만 처음 나타난 것은 아니다.
- Fresh vision-heavy client dispatch delay p95는 old-a **5168.62 ms**, new-a **3902.05 ms**,
  new-b **3727.21 ms**, old-b **3651.88 ms**다. Multi-image는 0.71–0.79 ms 수준이다.
  이는 client arrival/admission/queue trajectory 차이를 더 분석할 신호이며, 실제 요청 latency 차이를
  무효화하거나 client/runtime 중 한 곳으로 원인을 단정하는 증거는 아니다.

### 15.6 최종 판정과 잔여 사항

기능상 all-profile binding 복구(14절)와 전 workload 성능 무회귀 증명은 별개의 gate다.

1. **기능 수정은 유지 가능**하되, primary 72회를 최종 `e8164e0`에서 재실행했다고 적지 않는다.
   이번 최종 바이너리 추가 coverage는 6-cell screen 및 Gemma 2-workload fresh ABBA다.
2. 모든 지표 3% 이내라는 주장은 **보류**한다. Multi-image fresh two-repeat regression과
   supplementary의 나쁜 두 결과를 다음 검토 대상으로 남긴다.
3. 첫 번째 후속 비교는 같은 조건에서 반복 수와 순서를 늘리고, calibration 후 action/cohort
   sequence 및 request completion을 대조하는 것이다. 불리한 run만 재시도하여 교체하지 않는다.
4. Gemma 9853 MiB peak의 얇은 usable headroom, tiered/independent OOM, greedy identity 미통과,
   실제 simultaneous slab high-water 미계측 등 앞 절의 한계는 이번 추가 screen으로 해결되지 않았다.
   Semantic correctness나 sanitizer/cancellation coverage도 성능 표에서 새로 추론하지 않는다.
5. vLLM은 workload/engine 비교 contract가 바뀌지 않아 frozen reference를 재사용한다.
   `single-profile-final-report.{json,md}`에 6개 행의 7지표 vLLM 비교가 있으며, 이번 ABBA는
   old/new Current의 변화 원인을 좁히기 위한 추가 실험이다.

재현 자료:

- `.local/results/runtime-contract-revalidation-20260926/single-profile-final-report.{json,md,manifest.json}`
- `.local/results/runtime-contract-revalidation-20260926/paired-profile-guard-{old-a,new-a,new-b,old-b}/manifest.json`
- 각 cell의 `contract.json`, `aggregate.json`, `run-001/client/run-001/summary.json`
- 같은 결과 디렉토리의 `paired-profile-guard-analysis.md`는 이 절의 독립 사본이다.
