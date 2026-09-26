# 333 — Encoder admission과 M-RoPE lifetime 분리: 수정 전 원인 분석

작성일: 2026-09-26. 이 문서는 이미 저장된 결과를 CPU로 읽어 분석했다.

1–9절은 수정 직후의 원인 분석 기록을 보존한다. 이후 완료된 `5520216` targeted 검증과
명확한 latency trade-off는 **10절**에 추가했다. Targeted 결과를 Full12 반복 검증으로 읽으면 안 된다.
최종 screen 판정은 **`candidate_rejected` / default not promoted**다. 10.6절의 회귀 gate 결정을 따른다.

## 1. 결론과 현재 상태

Cosmos의 shared E/P single-storage 경로에서 **prefill이 끝난 뒤 decode가 사용하는 M-RoPE 데이터까지
다음 encoder 준비의 차단 조건에 포함**되어 있었다. Encoder output slab의 lifetime과 M-RoPE의
decode lifetime이 다시 admission 조건에서 합쳐진 것이다.

수정 전 `43c680a` 바이너리의 Cosmos vision-heavy에서는 15개의 인접 encoder cohort 전환 모두
다음 E 준비가 이전 cohort 전체 요청 완료 후 0.081–0.384 ms에 시작했다. 이전 cohort의 마지막
prefill 완료부터는 평균 205.23 ms를 추가로 기다렸다. Mixed도 10/10회 같은 패턴이었다.
반면 Gemma는 모든 해당 전환에서 이전 cohort의 decode 종료 전에 다음 E 준비를 시작했다.

`5520216ba53e5892b0ead75742aed8410b09423e`는 준비 차단 조건의 logical payload 검사를
`byteSize()`에서 **prefill용 embedding/deepstack 존재 여부**로 바꾼 최소 수정이다.
물리적 retained slab 수, E/P shared workspace exclusion, completion 이전 ownership 유지는 바꾸지 않았다.

**이 문서 작성 시점의 수정 상태:** 코드와 테스트가 추가되었지만, 이 수정의 빌드·GPU 검증 및
수정 후 성능 측정은 아직 하지 않았다. 아래 모든 수치는 수정 전 결과다. 이후 검증 결과는 별도 절에
추가해야 하며 수정 전 표를 덮어쓰지 않는다.

이 분석은 Cosmos의 긴 vision tail을 설명하는 구체적인 mechanism defect를 찾은 것이지,
전체 tail이 반드시 사라지거나 모든 workload가 빨라진다는 증명은 아니다.

## 2. 결과·바이너리 identity와 raw 경로

저장소 루트는 `/home/sslab/TensorRT-Edge-LLM`이다. 분석 캠페인 두 개는 다음과 같다.

- 수정 전 current: `/home/sslab/TensorRT-Edge-LLM/.local/results/runtime-contract-revalidation-20260926/full24-final-3x`
- frozen prior 대조: `/home/sslab/TensorRT-Edge-LLM/.local/results/runtime-contract-revalidation-20260926/prior-p0`

각 root의 `manifest.json`이 source/binary/plugin/engine/trace/config identity의 원본이다.

| 항목 | Frozen prior | 수정 전 current |
|---|---|---|
| Binary source commit | `949334c5d2a6bca085e843558837a36e66bf5617` | `43c680a2a7e4dea4922c33e9c61f93628b6a4d95` |
| Campaign source commit | `6e2c518a0639422ade45d1cf1c6d5e205b97fb85` | `47a9cf7616570d00cd8d3b8a142f8af7dc5cb042` |
| Binary SHA256 | `3217f49f76f07faa0e91f65bc9926dbdd43212a7bb7ba3250863a165879f3146` | `aab919194709253bdc377830653e1c201abf4a0b34967469cf0db0832cb1552b` |
| Plugin SHA256 | `d3c3c679794814e83bfd9468b05bc81ab08adc8c7966f3317102b020b67af2e5` | `ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2` |
| Campaign source dirty state | 있음; 정확한 파일 목록은 manifest | clean |

Binary source commit과 campaign source commit을 구분한다. Prior 실행 당시 source가 dirty였다고 해서
그 변경을 이미 frozen binary가 포함한 것으로 해석하지 않는다.

Current 엔진 경로:

- Gemma: `.local/artifacts/v0101-forward-port/workspace-corrected-20260926/gemma`
  — engine SHA256 `fef5210c22b0ceb064cdce07658f6dace7737d66cad7e35a3f625602f2c9405e`
- Cosmos: `.local/artifacts/v0101-forward-port/workspace-corrected-20260926/cosmos`
  — engine SHA256 `c4f873c30db785cb87aba4475cc80b935112f225c3e2204f22c2ade09356026d`

Prior 엔진 경로:

- Gemma: `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-packed-p8-d24-kv2048-p192`
  — engine SHA256 `68ace712ec66a9a493dd19e7148b8f8eb069872a87004399ef4ba25a35ffc3c9`
- Cosmos: `.local/artifacts/v0101-forward-port/text/cosmos-reason2-2b/engine-p8-d64-kv256-p128-vp1024-atomic`
  — engine SHA256 `084d039248e08dc192a57ebf38d521ee1fdca2f037a865f55ac0b0b904214b0b`

각 campaign root 아래 분석한 cell의 정확한 상대 경로는 다음과 같다.

| 모델·워크로드 | Cell path | Current | Prior |
|---|---|---|---|
| Gemma balanced | `gemma/shared_ep-predictor-on/repeat-001/balanced` | 있음 | 있음 |
| Gemma mixed | `gemma/shared_ep-predictor-on/repeat-001/mixed` | 있음 | 없음 |
| Gemma vision-heavy | `gemma/shared_ep-predictor-on/repeat-001/vision-heavy` | 있음 | 있음 |
| Gemma multi-image | `gemma/shared_ep-predictor-on/repeat-001/multi-image` | 있음 | 있음 |
| Cosmos balanced | `cosmos/shared_ep-predictor-on/repeat-001/balanced` | 있음 | 있음 |
| Cosmos mixed | `cosmos/shared_ep-predictor-on/repeat-001/mixed` | 있음 | 없음 |
| Cosmos vision-heavy | `cosmos/shared_ep-predictor-on/repeat-001/vision-heavy` | 있음 | 있음 |
| Cosmos multi-image | `cosmos/shared_ep-predictor-on/repeat-001/multi-image` | 있음 | 있음 |

각 cell에서 사용한 파일:

- `aggregate.json`: HTTP serving 지표와 GPU peak memory.
- `run-001/gateway.log.gz`: `PHASE_EPOCH`, `PHASE_METRIC`, `PHASE_ENCODER_METRIC`,
  `PHASE_TIMELINE`, `PHASE_SCHEDULER_EVENT`, graph cache log.
- `run-001/activity-intervals.csv`: 실제 계측된 stream 작업 interval.
- `run-001/activity-summary.csv`: mask별 epoch 비율.
- `run-001/client/run-001/requests.csv`: 요청별 raw 출력 및 HTTP 결과의 원본.

예를 들어 핵심 Cosmos vision-heavy timeline 원본은 다음 파일이다.

`/home/sslab/TensorRT-Edge-LLM/.local/results/runtime-contract-revalidation-20260926/full24-final-3x/cosmos/shared_ep-predictor-on/repeat-001/vision-heavy/run-001/gateway.log.gz`

Prior와 current의 비교 가능한 6개 cell은 **trace SHA256이 모두 동일**하다. 그러나 바이너리,
plugin, 엔진 및 실제 graph coverage가 다르므로 이 표는 순수 predictor-only 또는 storage-only A/B가 아니다.
특히 prior는 one-token prefill-tail 수정 전 lineage다. 정확한 correctness 범위는 노트 331을 함께 본다.

## 3. 측정 범위와 해석 규칙

이 문서는 전체 72회 최종 집계가 아니라 **두 모델 × 네 workload의 repeat 1**을 분석한다.
Full12 캠페인 이름에 `3x`가 포함되어 있어도 아래 값은 3회 중앙값이 아니다.

분석은 로그의 `PHASE_EPOCH {"epoch":1,"kind":"measurement"}` 이후를 사용한다.
Warm-up encoder batch, queue wait, P/D dispatch를 섞지 않았다. `PHASE_METRIC`의 P/D batch가
양수인 record를 해당 phase dispatch로 집계하고, E는 측정 epoch의 `PHASE_ENCODER_METRIC`으로 집계했다.
하나의 P+D action은 P와 D 각각에 한 번씩 포함된다.

지표 정의:

- E queue wait: 요청별 `encoder_start - vision_queued`.
- `encoder_start`는 coordinator의 **encoder 준비 시작**이다. 첫 encoder kernel 시작과 같지 않다.
- D ready wait: 각 `decode_start`보다 앞선 가장 최근 같은 요청의 `decode_ready`와의 차이.
- D dispatch gap: 시간순 D dispatch GPU interval 사이의 `max(0, next_start - previous_end)`.
  Ready D가 없는 drain/arrival 구간도 포함하므로 단독으로 starvation으로 부르면 안 된다.
- E/P engine overlap: `encoder_engine`과 P의 `*_dispatch` interval 교집합 합.
- Mask duty: 전체 measurement activity window를 분모로 사용한다.
- Queue p95는 요청/iteration raw sample에 선형 보간을 적용한 descriptive p95다.

두 가지 telemetry 함정도 피했다.

1. `prefill_release`는 `PhaseThreeCoordinator::dispatchReadyPrefill()`에서 요청을 LLM admission queue로
   넘긴 시각이다. 실제 final-prefill storage release가 아니다. 짧은 `encoder_done → prefill_release`를
   짧은 physical lease lifetime의 증거로 사용하지 않는다.
2. `PHASE_METRIC`의 vision aggregate는 record emission 때 `ipcThreePhase->metrics()`로 읽는다.
   그 record의 과거 `host_completion_us`와 완전히 동시인 ownership snapshot이라고 간주하지 않는다.
   Warm-up 누적 guard counter를 measurement-only 대기 횟수로 해석하지도 않는다.

## 4. 수정 전 current: phase batch와 overlap

E/P/D 칸은 `dispatch 수 / 평균 batch / 최대 batch`이다.

| 모델·워크로드 | E | P | D | Idle | E+D | P+D |
|---|---:|---:|---:|---:|---:|---:|
| Gemma balanced | 0 | 40 / 2.10 / 8 | 295 / 18.22 / 24 | 1.39% | 0% | 10.09% |
| Gemma mixed | 16 / 2.00 / 4 | 85 / 1.74 / 7 | 216 / 13.26 / 24 | 1.48% | 8.49% | 12.04% |
| Gemma vision-heavy | 16 / 3.00 / 4 | 92 / 1.98 / 5 | 174 / 13.79 / 24 | 1.65% | 16.54% | 8.14% |
| Gemma multi-image | 7 / 2.86 / 4 | 35 / 1.94 / 4 | 46 / 13.48 / 20 | 1.77% | 16.80% | 6.62% |
| Cosmos balanced | 0 | 170 / 1.98 / 8 | 484 / 50.98 / 64 | 3.62% | 0% | 30.16% |
| Cosmos mixed | 11 / 2.91 / 4 | 37 / 1.84 / 8 | 344 / 8.33 / 33 | 3.02% | 1.32% | 0.35% |
| Cosmos vision-heavy | 16 / 3.00 / 4 | 50 / 1.32 / 8 | 499 / 4.81 / 18 | 2.88% | 0.93% | 0.15% |
| Cosmos multi-image | 2 / 2.50 / 3 | 5 / 1.00 / 1 | 63 / 2.46 / 3 | 2.05% | 0% | 1.54% |

8개 모두 global/vision scheduler counter 및 event record 기준 **action-fidelity violation 0**이다.
`encoder_engine` 대 P dispatch의 실제 overlap도 **모두 0 ms**다.

Gemma mixed의 mask에서 E&P 및 EPD envelope 비율이 각각 **0.004%** 관측된다. P sampling도 P bit에
포함되므로, encoder engine과 P sampling의 겹침을 E/P TensorRT workspace violation으로 해석하지 않는다.
나머지 current 7개 cell의 E&P envelope 비율은 0이다.

Copy mask는 이 계측 범위에서 0이다. Direct output 경로의 encoder-output D2D가 없다는 것과
시스템 전체의 H2D/D2H가 전혀 없다는 것은 다른 주장이다.

### 4.1 측정 epoch의 GPU service 합

단위는 ms이며 E는 preparation envelope가 아닌 `execution_gpu_ms`를 사용한다.
P/D는 각각 phase GPU event 시간 합이다. 겹친 시간이 있으므로 세 합을 E2E makespan으로 더하면 안 된다.

| 모델·워크로드 | E 합 / dispatch 평균 | P 합 / dispatch 평균 | D 합 / dispatch 평균 |
|---|---:|---:|---:|
| Gemma balanced | 0 / — | 825.43 / 20.64 | 3790.69 / 12.85 |
| Gemma mixed | 580.68 / 36.29 | 1819.68 / 21.41 | 2920.08 / 13.52 |
| Gemma vision-heavy | 862.91 / 53.93 | 2134.36 / 23.20 | 2911.17 / 16.73 |
| Gemma multi-image | 365.33 / 52.19 | 813.16 / 23.23 | 824.85 / 17.93 |
| Cosmos balanced | 0 / — | 2827.00 / 16.63 | 4234.90 / 8.75 |
| Cosmos mixed | 954.51 / 86.77 | 955.45 / 25.82 | 2263.18 / 6.58 |
| Cosmos vision-heavy | 1429.73 / 89.36 | 1323.71 / 26.47 | 3187.36 / 6.39 |
| Cosmos multi-image | 151.35 / 75.67 | 135.34 / 27.07 | 391.49 / 6.21 |

## 5. Queue wait와 cohort 전환의 직접 증거

### 5.1 E queue 대기와 D ready 대기

단위는 ms다. D ready wait는 대부분 짧고 일부 매우 긴 값이 섞인 분포이므로 mean이 p95보다 클 수 있다.

| 모델·워크로드 | E queue mean / p95 / max | D ready wait mean / p95 / max | D dispatch gap mean / p95 / max |
|---|---:|---:|---:|
| Gemma balanced | — | 1.120 / 7.503 / 36.271 | 1.802 / 8.498 / 56.457 |
| Gemma mixed | 150.074 / 869.678 / 1245.576 | 8.838 / 44.475 / 130.301 | 7.076 / 35.215 / 141.570 |
| Gemma vision-heavy | 370.281 / 932.574 / 1147.026 | 9.982 / 59.211 / 130.373 | 10.731 / 65.468 / 140.831 |
| Gemma multi-image | 117.376 / 272.216 / 289.479 | 8.649 / 66.667 / 177.297 | 16.528 / 93.150 / 142.208 |
| Cosmos balanced | — | 2.605 / 6.628 / 122.991 | 2.953 / 7.644 / 124.410 |
| Cosmos mixed | 1934.780 / 3439.359 / 3802.299 | 2.291 / 0.267 / 126.481 | 5.767 / 0.623 / 194.949 |
| Cosmos vision-heavy | 2785.761 / 5248.119 / 5634.861 | 1.543 / 0.199 / 97.352 | 5.831 / 0.527 / 207.560 |
| Cosmos multi-image | 146.651 / 334.422 / 334.422 | 0.442 / 0.086 / 43.473 | 2.347 / 0.274 / 122.542 |

Cosmos vision-heavy의 긴 tail은 D가 계속 ready인데 보통 실행되지 못하는 형태보다,
E가 늦어져 P와 후속 D cohort 공급이 가늘어진 형태와 더 일치한다. D maximum capacity는 64지만
이 trace의 D 최대는 18, 평균은 4.81이다. 반면 동일 모델의 text balanced는 D 평균 50.98이다.
단순히 D cap을 늘려 해결할 현상이 아니다.

### 5.2 이전 E cohort의 final P와 전체 완료에 대한 다음 E 준비 시각

각 E dispatch의 request membership을 `PHASE_SCHEDULER_EVENT`에서 얻고, 같은 요청들의
마지막 `prefill_done`과 `completion`을 timeline에서 대응했다. 다음 cohort의 가장 이른
`encoder_start`와 비교한다. 따라서 request를 서로 다른 cohort에 임의로 대응한 값이 아니다.

| 모델·워크로드 | 인접 cohort 전환 수 | 이전 final P→다음 E 준비 평균 | 이전 전체 완료→다음 E 준비 평균 | 이전 전체 완료→다음 E 범위 | 이전 decode 완료 전 다음 E |
|---|---:|---:|---:|---:|---:|
| Gemma mixed | 15 | 37.051 ms | −892.845 ms | −1296.485 ~ −481.957 ms | 15/15 |
| Gemma vision-heavy | 15 | 30.459 ms | −1132.067 ms | −1587.429 ~ −537.029 ms | 15/15 |
| Gemma multi-image | 6 | 0.025 ms | −1013.774 ms | −1418.008 ~ −586.880 ms | 6/6 |
| Cosmos mixed | 10 | 210.541 ms | 0.136 ms | 0.105 ~ 0.154 ms | 0/10 |
| Cosmos vision-heavy | 15 | 205.231 ms | 0.178 ms | 0.081 ~ 0.384 ms | 0/15 |
| Cosmos multi-image | 1 | 196.164 ms | 0.158 ms | 0.158 ms | 0/1 |

Cosmos vision-heavy의 첫 다섯 전환:

| 이전 E cohort request IDs | Final P→다음 E 준비 | 전체 완료→다음 E 준비 |
|---|---:|---:|
| `[4]` | 230.091 ms | 0.081 ms |
| `[8,12,0,22]` | 222.170 ms | 0.158 ms |
| `[24,23,20]` | 197.601 ms | 0.251 ms |
| `[16,27,26]` | 202.994 ms | 0.142 ms |
| `[29,25,28]` | 203.994 ms | 0.148 ms |

첫 전환을 measurement 시작 상대 시각으로 풀면 다음과 같다.

```text
req4 encoder 준비 시작       10.42 ms
req4 encoder 완료            55.58 ms
req4 LLM queue로 전달        55.64 ms  (prefill_release; storage release 아님)
req4 prefill 시작           139.52 ms
req4 final prefill 완료     170.15 ms
req4 요청 전체 완료         400.16 ms
다음 cohort encoder 준비    400.24 ms
```

이것은 engine workspace의 E/P exclusion만으로 요구되는 대기가 아니다. 마지막 P가 끝난 후에도
약 230 ms의 decode 구간을 기다렸기 때문이다. 다만 timeline과 코드 분석만으로 수정 후 가능한
throughput 증가량을 계산하지 않는다. 다른 arbitration, preparation, batch formation이 남아 있다.

## 6. Frozen prior와 current 비교

### 6.1 Throughput와 memory

각각 repeat 1이다. `prior-p0`는 고정된 old binary를 같은 runtime contract로 재실행한 대조이지,
과거 모든 캠페인 중 최고 성능의 oracle을 뜻하지 않는다.

| 모델·워크로드 | Prior token/s | Current token/s | 변화 | Prior→current peak MiB |
|---|---:|---:|---:|---:|
| Gemma balanced | 1176.22 | 1223.14 | +3.99% | 9695→9849 |
| Gemma vision-heavy | 489.29 | 498.63 | +1.91% | 9699→9853 |
| Gemma multi-image | 365.38 | 377.34 | +3.27% | 9699→9853 |
| Cosmos balanced | 4136.45 | 4264.92 | +3.11% | 9297→9299 |
| Cosmos vision-heavy | 390.89 | 390.33 | −0.14% | 9297→9299 |
| Cosmos multi-image | 172.99 | 215.69 | +24.68% | 9297→9299 |

Mixed current peak는 Gemma 9853 MiB, Cosmos 9299 MiB다. Prior mixed는 측정하지 않았다.

### 6.2 Current의 serving 7개 지표

Latency는 ms다. 모델 간 trace 요청 수/출력 계약까지 같다는 의미가 아니므로 모델 간 절대 처리량을
아키텍처 우열로 직접 비교하지 않는다. 예를 들어 balanced는 Gemma 64요청, Cosmos 288요청이고,
multi-image는 Gemma 20요청, Cosmos 5요청이다.

| 모델·워크로드 | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma balanced | 1223.14 | 145.37 | 617.97 | 15.45 | 16.68 | 1436.97 | 2130.19 |
| Gemma mixed | 631.96 | 349.73 | 1283.96 | 28.65 | 41.47 | 1631.08 | 2870.46 |
| Gemma vision-heavy | 498.63 | 498.98 | 1118.76 | 33.63 | 52.04 | 1719.02 | 2379.54 |
| Gemma multi-image | 377.34 | 311.00 | 527.86 | 29.57 | 45.76 | 1227.73 | 1552.71 |
| Cosmos balanced | 4264.92 | 69.55 | 169.57 | 12.58 | 14.47 | 1138.26 | 1782.74 |
| Cosmos mixed | 664.16 | 1131.00 | 3694.38 | 9.90 | 13.67 | 1606.22 | 3929.36 |
| Cosmos vision-heavy | 390.33 | 2305.54 | 5522.90 | 8.43 | 12.36 | 2642.56 | 5762.29 |
| Cosmos multi-image | 215.69 | 303.24 | 518.97 | 6.96 | 7.58 | 518.96 | 736.33 |

### 6.3 Prior의 serving 7개 지표

| 모델·워크로드 | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemma balanced | 1176.22 | 84.83 | 227.13 | 16.33 | 17.84 | 1445.92 | 2246.31 |
| Gemma vision-heavy | 489.29 | 496.79 | 1129.21 | 34.26 | 53.40 | 1736.99 | 2299.78 |
| Gemma multi-image | 365.38 | 326.89 | 498.89 | 30.41 | 46.95 | 1269.65 | 1590.72 |
| Cosmos balanced | 4136.45 | 63.27 | 158.67 | 13.05 | 14.47 | 1173.38 | 1851.93 |
| Cosmos vision-heavy | 390.89 | 2310.27 | 5536.38 | 8.39 | 11.98 | 2645.23 | 5760.54 |
| Cosmos multi-image | 172.99 | 383.09 | 637.77 | 6.95 | 7.72 | 598.51 | 860.88 |

Throughput가 개선된 셀도 모든 latency가 개선되지는 않았다. 특히 Gemma balanced TTFT p95는
227.13→617.97 ms로 악화했다. Cosmos multi-image는 요청 5개뿐이므로 한 번의 큰 개선을 안정적인
일반 성능 향상으로 승격하지 않는다.

### 6.4 Prior 대비 phase formation과 D gap

P/D/E 칸은 prior→current다. Mean BS는 같은 phase의 dispatch 평균이다.

| 모델·워크로드 | E dispatch / mean BS | P dispatch / mean BS | D dispatch / mean BS | D gap p95 | E+D duty | P+D duty |
|---|---|---|---|---:|---:|---:|
| Gemma balanced | 0→0 / — | 47→40 / 1.79→2.10 | 297→295 / 18.10→18.22 | 12.67→8.50 ms | 0→0% | 8.89→10.09% |
| Gemma vision-heavy | 16→16 / 3.00→3.00 | 92→92 / 1.98→1.98 | 177→174 / 13.56→13.79 | 78.92→65.47 ms | 13.86→16.54% | 8.68→8.14% |
| Gemma multi-image | 7→7 / 2.86→2.86 | 36→35 / 1.89→1.94 | 47→46 / 13.19→13.48 | 117.50→93.15 ms | 14.62→16.80% | 4.98→6.62% |
| Cosmos balanced | 0→0 / — | 193→170 / 1.74→1.98 | 482→484 / 51.19→50.98 | 7.64→7.64 ms | 0→0% | 25.90→30.16% |
| Cosmos vision-heavy | 16→16 / 3.00→3.00 | 47→50 / 1.40→1.32 | 500→499 / 4.80→4.81 | 0.60→0.53 ms | 0.97→0.93% | 0.25→0.15% |
| Cosmos multi-image | 3→2 / 1.67→2.50 | 5→5 / 1.00→1.00 | 93→63 / 1.67→2.46 | 0.28→0.27 ms | 0→0% | 0→1.54% |

Cosmos vision-heavy는 old/new 양쪽에서 작은 D cohort와 낮은 overlap이 거의 그대로다. 따라서 이
문제는 이번 predictor 수정을 통해 새로 생긴 현상이라고 단정할 수 없다. Multi-image의 큰 차이는
즉시 D 실행 속도보다는 E cohort 3→2와 D dispatch 93→63의 formation 변화가 동반된 결과다.

### 6.5 실제 graph coverage 차이

Vision-heavy 로그 기준:

| 모델·버전 | 측정 시작 D entries | 시작 hits/misses | 종료 hits/misses | 측정 중 hits/misses |
|---|---:|---:|---:|---:|
| Gemma prior | 10 | 331 / 66 | 440 / 134 | 109 / 68 |
| Gemma current | 24 | 390 / 0 | 564 / 0 | 174 / 0 |
| Cosmos prior | 64 | 1403 / 0 | 1903 / 0 | 500 / 0 |
| Cosmos current | 64 | 1442 / 0 | 1941 / 0 | 499 / 0 |

모두 P graph entries는 0이다. Gemma에서는 실제 D graph coverage가 달라졌기 때문에 메모리 증가와
성능 변화를 predictor 하나에 귀속할 수 없다. 이 비교에서 Gemma peak는 +154 MiB, Cosmos는
+2 MiB이며 **net VRAM 감소 결과가 아니다**. 전체 메모리 항목별 attribution은 노트 332를 참조한다.

## 7. 코드 수준 원인

수정 전 두 encoder 준비 gate는 다음 입력을 사용했다.

```text
phaseVisionPreparationWithinStorageBudget(
    shared_workspace,
    max_retained_batches,
    actual_retained_encoder_slabs,
    ready_prefill_exists OR server.visionPayloadBytes() > 0)
```

관련 구현:

- `cpp/runtime/scheduling/phaseVisionAdapter.cpp`, `PhaseVisionPayload::byteSize()`:
  outputEmbedding + deepstackFeatures + **mropeCosSin**의 logical bytes를 합한다.
- 같은 파일의 `prefillByteSize()`는 embedding/deepstack만 센다.
- `releasePrefillStorage()`는 split lease일 때 embedding/deepstack/storageOwner를 해제하되 M-RoPE는 유지한다.
  Legacy tied slab은 decode가 끝날 때까지 whole-slab ownership을 유지한다.
- `cpp/runtime/scheduling/independentPhaseAsyncServer.cpp`, `visionPayloadBytes()`:
  active와 pending 요청의 전체 `byteSize()`를 합한다.
- `cpp/runtime/scheduling/phaseVisionAdapter.h`, `phaseVisionPreparationWithinStorageBudget()`:
  single-storage에서는 downstream flag가 true이면 다음 준비를 허용하지 않는다.
- `cpp/runtime/scheduling/phaseThreeCoordinator.cpp`, `poll()`와 `startNextEncoder()`:
  후보 준비 경로와 실제 준비 시작 모두 위 전체-byte 기반 flag를 사용했다.

따라서 split M-RoPE 상태에서는 다음과 같은 불필요한 의존성이 생겼다.

```text
E output slab ── final P complete ── release
M-RoPE       ────────────────────── D ... D ── release
                                  │
next E admission                  └── total payload bytes 조건 때문에 계속 차단
```

이 차단은 predictor가 cost를 잘 예측하는지와 별개다. 후보를 만들기 전에 준비 단계가 막히므로,
추가 E+D overlap 학습이나 E batch cap 증가만으로 해당 경로를 되살릴 수 없다.

## 8. `5520216` 수정과 보존한 invariant

새 query `IndependentPhaseAsyncServer::hasVisionPrefillPayload()`는 다음을 모두 확인한다.

- active request의 `visionPayload`
- prefix/suffix 연결을 기다리는 active request의 `pendingVisionPayload`
- admission 대기 중인 `PendingRequest::visionPayload`

각 payload가 존재하고 `prefillByteSize() > 0`일 때만 true다. 두 preparation gate에서 동일한 query를
사용하며 `!mReadyPrefill.empty()` 조건도 유지한다.

보존한 조건:

1. 실제 physical slab을 아직 consumer가 보유하면 `retainedStorageBatches()` cap이 계속 차단한다.
2. 아직 P가 소비해야 하는 embedding/deepstack는 pooled/unpooled 여부와 무관하게 single-storage를 차단한다.
3. Ready-prefill queue에 작업이 남으면 기존 보수적 single-storage 조건을 유지한다.
4. Legacy tied M-RoPE slab은 decode 종료까지 storage owner를 유지하므로 계속 차단된다.
5. Shared E/P TensorRT workspace exclusion은 그대로다. 이번 변경은 E와 P engine의 동시 실행을 허용하지 않는다.
6. 취소나 완료 시점을 앞당겨 reference를 버리지 않는다. GPU consumer completion 이전의 lease를
   강제로 해제하여 capacity를 얻는 수정이 아니다.
7. maxRetained=2의 physical cap과 independent workspace의 기존 동작도 유지한다.

수정 파일:

- `cpp/runtime/scheduling/independentPhaseAsyncServer.h`
- `cpp/runtime/scheduling/independentPhaseAsyncServer.cpp`
- `cpp/runtime/scheduling/phaseThreeCoordinator.cpp`
- `cpp/runtime/scheduling/phaseVisionAdapter.h`
- `unittests/cpp/runtime/scheduling/phaseVisionStoragePolicyTest.cpp`

추가 unit case는 M-RoPE-only payload, unpooled embedding/deepstack, legacy tied slab,
physical consumer lease/cap 1·2를 다룬다. 기존 4개와 합쳐 총 8개다.
이 문서 작성 시 source formatting과 `git diff --check`만 확인했고, **새 테스트 실행은 아직 하지 않았다**.
Unit helper 검증은 실제 CUDA cancellation/lifetime 안전성 증명을 대체하지 않는다.

## 9. 다음 검증: 수정 전 결과와 분리하여 추가할 것

1. 빌드 후 `PhaseVisionStoragePolicyTest.*`와 관련 runtime/state 테스트 실행.
2. 기존 primary 캠페인을 중간에 변경하지 않고 frozen binary 결과로 완료.
3. 같은 엔진/config에서 두 모델 × balanced/vision-heavy/multi-image의 수정 후 targeted run.
4. Cosmos의 다음 E 준비가 이전 cohort의 **final P 이후이면서 decode 완료 전** 시작하는지 확인.
   E queue, D cohort, 실제 E+D duty, 7개 serving 지표, peak VRAM을 함께 비교.
5. `encoder_engine` 대 P dispatch overlap 0, action-fidelity 0, output count/transport integrity를 확인.
   Raw exact와 stop-inclusive exact를 별도로 보고 semantic correctness를 추정하지 않는다.
6. In-flight cancellation, surviving request, request teardown, workspace graph reuse를 다시 확인.
7. Targeted 결과가 정당화하면 같은 contract로 Full12 × 3을 다시 실행. 성능 개선뿐 아니라 tail 회귀와
   메모리 증가도 함께 판단한다.

수정 후 raw path, binary/plugin hash, 반복 수, 표와 gate 결과는 이 절 뒤에 추가한다.
현재는 수정 후 성능, 안전성, vLLM 우세를 주장하지 않는다.

## 10. 추가 검증 — `5520216` targeted 6개 cell 완료

이 절은 위의 수정 전 분석 이후에 추가했다. 빌드와 GPU 실행은 main agent가 수행했고,
이 분석은 완료된 raw artifact만 읽었다.

### 10.1 실행·품질 contract

새 campaign root:

`/home/sslab/TensorRT-Edge-LLM/.local/results/runtime-contract-revalidation-20260926/mrope-gate-screen`

두 모델 각각 balanced/vision-heavy/multi-image, `shared_ep-predictor-on/repeat-001`의 **6/6 cell이
완료**했으며 manifest의 실패 record는 0이다. 비교 상대는 2절의 `full24-final-3x` **repeat 1**이다.
새 테스트에서 mixed는 실행하지 않았다.

- Source 및 binary source: `5520216ba53e5892b0ead75742aed8410b09423e`.
- Binary SHA256: `e8dfbde880c6000d9f75ce05579c2d38ac06bbd7cc3a3d476583ab5ad14fef63`.
- Plugin SHA256: `ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2` — 수정 전과 동일.
- 두 모델 모두 engine, engine config, vision engine/config, calibration SHA256이 수정 전과 같다.
- E4, P8/P128, D24(Gemma)/D64(Cosmos), shared E/P single-storage, predictor on,
  graph on/P graph cap 0/D graph cap 64, generic calibration 49/239 계약을 유지했다.
- Manifest dirty state는 노트 변경만 포함한다. Runtime 변경을 dirty 상태로 덧씌워 비교한 것이 아니다.

검증 로그:

- `.local/results/runtime-contract-revalidation-20260926/build-mrope-gate.log`
- `.local/results/runtime-contract-revalidation-20260926/unit-tests-mrope-gate.log`

Runtime unit suite: **694개 실행, 692 pass, 2 skip**. Skip은 isolated metadata benchmark와
NCCL resource test이며, 이 수치를 전체 end-to-end correctness proof로 해석하지 않는다.

새 output audit:

- `.local/results/runtime-contract-revalidation-20260926/mrope-gate-screen-quality-audit.json`
- `.local/results/runtime-contract-revalidation-20260926/mrope-gate-screen-quality-audit.md`

6개 cell 모두 HTTP/count/token-capture integrity issue 0, first-token EOS flag 0이다.
각 workload가 한 번뿐이므로 cross-repeat raw/prefix exact는 **not_tested**다.
이는 semantic quality 또는 수정 전후 exact greedy identity 검증을 대체하지 않는다.

### 10.2 7개 serving 지표와 memory

다음은 수정 후 단일 실행의 절대값이다. Latency 단위는 ms다.

| 모델·워크로드 | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma balanced | 1208.81 | 142.63 | 605.82 | 15.74 | 17.45 | 1454.71 | 2145.25 | 9849 |
| Gemma vision-heavy | 536.45 | 308.16 | 596.31 | 34.38 | 51.64 | 1580.37 | 2110.43 | 9853 |
| Gemma multi-image | 362.77 | 346.40 | 614.89 | 29.33 | 45.44 | 1255.60 | 1553.85 | 9853 |
| Cosmos balanced | 4187.94 | 62.51 | 136.33 | 12.92 | 14.31 | 1160.39 | 1821.77 | 9315 |
| Cosmos vision-heavy | 710.87 | 1210.37 | 2917.67 | 52.21 | 79.33 | 3177.80 | 3399.07 | 9341 |
| Cosmos multi-image | 312.55 | 210.77 | 294.24 | 9.47 | 12.86 | 504.35 | 511.65 | 9315 |

수정 전 repeat 1 대비 변화율이다. **Throughput만 양수가 유리하고 latency는 음수가 유리**하다.

| 모델·워크로드 | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma balanced | −1.17% | −1.88% | −1.97% | +1.87% | +4.60% | +1.23% | +0.71% | 0 MiB |
| Gemma vision-heavy | +7.59% | −38.24% | −46.70% | +2.26% | −0.77% | −8.07% | −11.31% | 0 MiB |
| Gemma multi-image | −3.86% | +11.38% | +16.49% | −0.82% | −0.71% | +2.27% | +0.07% | 0 MiB |
| Cosmos balanced | −1.81% | −10.13% | −19.61% | +2.69% | −1.13% | +1.94% | +2.19% | +16 MiB |
| Cosmos vision-heavy | +82.12% | −47.50% | −47.17% | +519.57% | +541.79% | +20.26% | −41.01% | +42 MiB |
| Cosmos multi-image | +44.91% | −30.49% | −43.30% | +36.09% | +69.71% | −2.82% | −30.51% | +16 MiB |

**All-metric 승리가 아니다.** 특히 Cosmos vision-heavy는 throughput과 TTFT/tail이 개선되지만
TPOT p95는 12.36→79.33 ms, E2E mean은 2642.56→3177.80 ms로 악화했다.
더 많은 요청을 빨리 decode에 넣고 tail을 줄이는 대신 기존 decode 요청의 진행을 크게 늦췄다.
기존의 잘못된 eligibility barrier가 우연히 decode 우선 효과를 냈다는 것과 그 barrier가 올바른
메모리 mechanism이었다는 것은 다른 주장이다. Barrier를 복원해 정책 역할을 대신하게 만들면 안 된다.

Gemma는 이번 M-RoPE coupling의 직접 대상이 아니다. Gemma vision-heavy의 +7.59%를 이번 수정의
확정적인 causal benefit으로 귀속하지 않는다. Generic warm-up 이후 online policy와 arrival/formation
경로가 달라질 수 있고, multi-image에는 반대 방향의 결과가 관측됐다. 반복 확인이 필요하다.

### 10.3 Mechanism 수정은 실제 timeline에서 확인됨

수정 후 E/P/D는 `dispatch 수 / 평균 BS / 최대 BS`다.

| 모델·워크로드 | E | P | D | Idle | E+D | P+D |
|---|---:|---:|---:|---:|---:|---:|
| Gemma balanced | 0 | 42 / 2.00 / 8 | 296 / 18.16 / 24 | 1.48% | 0% | 10.69% |
| Gemma vision-heavy | 20 / 2.40 / 4 | 99 / 1.84 / 5 | 139 / 17.27 / 24 | 1.73% | 17.53% | 9.53% |
| Gemma multi-image | 7 / 2.86 / 4 | 36 / 1.89 / 4 | 53 / 11.70 / 20 | 1.78% | 17.44% | 7.16% |
| Cosmos balanced | 0 | 183 / 1.84 / 8 | 480 / 51.40 / 64 | 3.60% | 0% | 27.28% |
| Cosmos vision-heavy | 15 / 3.20 / 4 | 48 / 1.38 / 8 | 69 / 34.78 / 56 | 2.81% | 21.72% | 0.61% |
| Cosmos multi-image | 2 / 2.50 / 3 | 4 / 1.25 / 2 | 32 / 4.84 / 5 | 1.83% | 6.64% | 0% |

6개 cell 모두 E/P engine interval overlap **0 ms**, global/vision/event fidelity violation **0**이다.
이번 screen의 E&P envelope ratio도 모두 0이다. Shared E/P exclusion을 풀어서 얻은 개선이 아니다.

Cosmos vision-heavy는 E dispatch 16→15, P 50→48인데 D가 **499→69회**로 줄고 평균 BS가
**4.81→34.78**로 커졌다. E GPU 합은 1429.73→1509.38 ms로 오히려 늘었지만,
D GPU 합은 3187.36→1265.97 ms로 줄었다. 개별 D dispatch 평균은 6.39→18.35 ms로 길어졌다.
이는 원래 지나치게 작게 공급되던 D cohort를 회복하는 효과와, 큰 D 및 E/P 간섭으로 token 주기가
늘어나는 효과가 동시에 발생했음을 보여준다.

| 항목 | 수정 전 | 수정 후 |
|---|---:|---:|
| Cosmos vision-heavy E queue mean / p95 | 2785.76 / 5248.12 ms | 1310.02 / 2666.87 ms |
| Cosmos vision-heavy D ready wait mean / p95 | 1.54 / 0.20 ms | 20.77 / 96.24 ms |
| Cosmos vision-heavy D dispatch gap mean / p95 | 5.83 / 0.53 ms | 31.39 / 147.35 ms |
| Cosmos vision-heavy E+D duty | 0.93% | 21.72% |
| Cosmos vision-heavy final P→다음 E 준비 평균 | 205.231 ms | 0.045 ms |
| Cosmos vision-heavy 이전 D 완료 전 다음 E 시작 | 0/15 | 14/14 |
| Cosmos multi-image E queue mean / p95 | 146.65 / 334.42 ms | 45.14 / 89.80 ms |
| Cosmos multi-image final P→다음 E 준비 | 196.164 ms | 0.013 ms |
| Cosmos multi-image 이전 D 완료 전 다음 E 시작 | 0/1 | 1/1 |

수정 후 Cosmos vision-heavy의 다음 E 준비는 이전 cohort 전체 완료보다 평균 **1580.44 ms 먼저**
시작했다. 범위는 450.35–2428.59 ms 먼저였다. Multi-image도 380.90 ms 먼저 시작했다.
즉 `P completed → E preparation eligible`라는 의도한 lifetime 분리가 실제 요청 timeline에서 확인됐다.

그러나 D ready 대기와 dispatch gap이 크게 증가한 것도 같은 raw timeline에서 확인된다.
이제는 E 공급이 묶여 있던 mechanism 병목을 넘어, 활성 E/P/D 사이의 service 균형 문제가 노출된다.

### 10.4 Cosmos vision-heavy의 request class별 손익

같은 trace의 text 16개, vision 48개를 나눈다. 단위는 ms다.

| Class | 지표 | 수정 전 | 수정 후 |
|---|---|---:|---:|
| Text | TTFT mean / p95 | 72.66 / 140.08 | 73.03 / 226.21 |
| Text | TPOT mean / p95 | 11.70 / 13.38 | 58.61 / 85.38 |
| Text | E2E mean / p95 | 738.64 / 810.22 | 3285.56 / 3397.12 |
| Vision | TTFT mean / p95 | 3049.83 / 5591.33 | 1589.49 / 3015.66 |
| Vision | TPOT mean / p95 | 7.33 / 8.78 | 50.08 / 78.45 |
| Vision | E2E mean / p95 | 3277.20 / 5821.16 | 3141.88 / 3390.84 |

Aggregate E2E mean이 악화한 핵심은 **기존 text resident의 decode continuity 손실**이다.
Text E2E mean은 약 +345%, p95는 약 +319%다. Vision E2E mean은 약 −4.1%, p95는 약 −41.8%다.
따라서 전체 throughput 또는 vision tail 하나만 선택하여 최종 production winner로 결론 내리지 않는다.

### 10.5 다음 gate

이 결과는 최소 mechanism 수정의 타당성과 overlap/frontier 회복 가능성을 뒷받침한다.
동시에 no-SLO policy에서 resident decode를 얼마나 보호할 것인지가 남았음을 보여준다.
후속 admission 후보를 설계할 때 다음을 분리해야 한다. 현재 후보의 실행 지속 여부는 10.6절을 따른다.

- 반복 Full12: 후속 후보가 targeted regression gate를 먼저 통과한 경우에만 동일 frozen contract로
  평균·p95·7개 지표와 memory 모두 검증.
- 요청 class별 trade-off: vision TTFT 개선이 text TPOT/E2E에 전가되는지 확인.
- Physical service vs queue residence: 큰 D batch의 GPU 비용과 E/P 사이 D service gap을 분리.
- Memory/lifetime: M-RoPE를 decode까지 유지한 상태의 peak 증가와 cancellation/teardown 안전성 확인.
- Existing scalar/transition selector: 새로 실행 가능한 E/P action이 resident D의 service urgency와
  protected continuation을 어떻게 반영하는지 분석. 학습 모델 변경이나 새로운 heuristic이 이미
  필요하다고 단정하지 않는다.

### 10.6 Screen fail-stop 결정: `candidate_rejected`

판정: **`5520216`은 default로 승격하지 않는다.** Cosmos vision-heavy의 text E2E mean 약 +345%,
p95 약 +319%는 다른 성능을 유지한다는 regression gate를 명확히 통과하지 못한다.
큰 throughput 향상만으로 이 회귀를 무시하거나 두 번째 Full12 × 3을 바로 실행하여 승격을 시도하지 않는다.

Main agent가 정한 처리 순서:

1. `5520216` source commit과 실험 binary, targeted raw 결과는 후보 분석용으로 보존한다.
2. 이미 진행 중인 bounded text sanitizer 진단이 끝난 뒤 해당 runtime admission 변경을 되돌린다.
3. 선택된 기본 runtime contract는 `43c680a`와 완료된 primary 72회 결과를 유지한다.
4. 다른 correctness 수정, 재현 가능한 runner/report/output-audit 도구는 유지한다.
5. 이번 분석만을 근거로 추가 workload flag나 per-workload tuning을 넣지 않는다.

Rollback을 `1ebaf63`으로 완료했다. `git diff 43c680a -- cpp unittests`가 비어 있음을 확인했고
기본 serving binaries도 보존한43c680a artifact로 복원했다. 후보 source `5520216`과 binaries는
`.local/baselines/runtime-contract-mrope-candidate-5520216-20260926/bin/`에 보존한다.
후보의 bounded text-only memcheck는 실제 cancel/survivor/readmission, exit0, error0으로
완료했으나 이것이 위 latency regression을 상쇄하지는 않는다.

구조적 교훈은 둘을 동시에 인정하는 것이다.

- Decode-only M-RoPE를 encoder slab 보유로 간주하는 것은 **물리적 메모리 안전성의 필수 조건이 아니다**.
  Lifetime에 대한 코드 설명을 그대로 유지해서는 안 된다.
- 그러나 기존 제한은 결과적으로 새 vision admission을 늦춰 **resident D를 보호하는 backpressure**로
  작동했다. 그 정책 효과를 대체하지 않고 allocator 조건만 완화하면 text latency가 크게 무너진다.

따라서 후속 연구는 `allocator correctness`와 `admission/service policy`를 분리해야 한다.
정확한 physical lease는 그대로 모델링하면서, 새 E/P가 이미 진행 중인 D에 미치는 service cost와
cohort 형성 효과를 별도 admission 정책에서 다룰 필요가 있다. 이 screen은 그 정책을 구현하거나
검증한 결과가 아니다.

현재 판정은 **“불필요한 lifetime coupling 확인, frontier 회복 확인, 심한 resident-D 회귀로 후보 기각”**이다.
전체 Full12 또는 vLLM 대비 전면 개선을 주장하지 않는다.
