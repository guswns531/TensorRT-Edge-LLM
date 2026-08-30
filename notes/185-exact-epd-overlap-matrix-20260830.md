# Exact E/P/D Overlap Matrix

## 결론

비어 있던 큰 prefill batch와 encoder batch 조합을 실제 HTTP request 수준에서 모두 채웠다. 이번 측정은 설정값만 바꾼 synthetic kernel benchmark가 아니다. Cosmos-Reason2-2B FP16 엔진, 실제 이미지 파일, 독립 E/P/D TensorRT context, CUDA event 기반 stream activity recorder를 사용했다.

핵심 결과는 다음과 같다.

- text-only에서는 현재의 큰 P/D batch가 이미 효율적이다. P4/D32, P8/D32, P8/D64에서 강제 P+D overlap은 3회 반복 기준 throughput과 latency를 동시에 개선하지 못했다.
- E+P는 pure encoder/prefill burst에서 실제 병렬성 가치가 크다. E1/P8의 50% overlap은 3회 중앙 req/s를 15.16% 높이고 TTFT p95를 10.54% 줄였다. 다만 TTFT 평균은 4.70% 늘었다.
- E+D는 decode가 충분히 큰 VLM 부하에서 더 균형적이다. E1/D32 100%는 throughput +2.43%, TPOT p95 -2.44%, E2E p95 -2.40%였다. TTFT 평균은 12.20% 늘었다.
- mixed E4/P8/D32에서는 E+D 100%가 E+P 100%보다 낫다. serial 대비 throughput +1.42%, TPOT p95 -1.39%, E2E 평균 -0.67%, E2E p95 -1.41%였고 TTFT 평균은 12.29% 늘었다.
- 실제 E8 실행도 32개 tiny-image backlog에서 확인했다. E8/D32의 E+D 100%는 3회 중앙값에서 throughput +0.51%, TPOT p95 -0.54%, E2E p95 -0.50%였지만 TTFT 평균/p95가 각각 5.18%/3.74% 나빠졌다. 더구나 E batch 수가 5회에서 7회로 늘어 static overlap은 승격하지 않는다.
- fresh vLLM VLM production과 동일 trace로 비교하면 선택한 Current 지점은 세 trace 모두 처리량과 평균·p95 latency가 우세했다. 대신 Current는 독립 TensorRT context와 vision runner 때문에 486~548 MiB 더 많은 peak VRAM을 사용했다.
- 81개 screening point, 9개 VLM 3회 확인점, 8개 text 3회 확인점 모두 요청한 output token을 100% 생성했다. 같은 trace 안의 token hash는 하나였고 action-fidelity violation은 0이었다.

이 결과는 overlap을 항상 켜야 한다는 뜻이 아니다. 독립 context의 가치는 workload별 static mode가 아니라 현재 action shape에서 serial, E+P, E+D, P+D 중 실제 이득이 있는 실행을 고를 수 있다는 데 있다.

## 구현

### E+P와 E+D opportunity envelope

기존 연구용 `TRT_EDGELLM_EXPERIMENTAL_OVERLAP_PERCENT`는 P+D만 제어했다. 다음 두 opt-in knob를 추가했다.

- `TRT_EDGELLM_EXPERIMENTAL_ENCODER_PREFILL_OVERLAP_PERCENT=0..100`
- `TRT_EDGELLM_EXPERIMENTAL_ENCODER_DECODE_OVERLAP_PERCENT=0..100`

두 knob의 기본값은 `-1`이며 production policy를 그대로 보존한다. 0~100을 주면 hard-feasible opportunity stream에서 deterministic accumulator로 정확한 장기 선택 비율을 만든다. E+P와 E+D 강제 knob를 동시에 켜는 것은 거부한다. 한 실험에서 어떤 overlap kind를 강제했는지 모호해지는 것을 막기 위해서다.

변경 위치는 다음과 같다.

- `cpp/runtime/scheduling/phaseThreeCoordinator.h`
  - 설정, opportunity/selection counter, 독립 accumulator
- `cpp/runtime/scheduling/phaseThreeCoordinator.cpp`
  - E+P/E+D candidate hard-feasibility 검사
  - deterministic percentage selection
  - 선택하지 않은 opportunity의 fallback 재평가
- `examples/llm/llm_phase_context_smoke.cpp`
  - 환경변수 parsing
  - 최종 `PHASE_METRIC` telemetry

추가 telemetry는 다음과 같다.

- `vision_global_experimental_encoder_prefill_opportunities`
- `vision_global_experimental_encoder_prefill_selections`
- `vision_global_experimental_encoder_decode_opportunities`
- `vision_global_experimental_encoder_decode_selections`

실험 knob는 production slack/cost soft gate를 우회하지만 dependency, ownership, TensorRT profile, memory, outstanding-context compatibility 같은 hard invariant는 우회하지 않는다.

### Text-prefill과 external-prefill의 비대칭 profile cap

E8 실험을 확장하면서 공통 P queue가 text P8 profile과 external/vision P4 profile을 같은 상한으로 취급하는 결함을 발견했다. 8개 tiny-image 결과가 동시에 E에서 P로 넘어오면 external P8을 만들 수 있었고, TensorRT enqueue 직전에 `packed prefill logical batch is out of range`로 거부됐다.

이를 workload heuristic 없이 shape feasibility로 수정했다.

- `PhaseQueueSchedulerConfig::maxExternalPrefillBatchSize`를 추가했다. 0은 기존 text P 상한을 상속한다.
- snapshot, dynamic candidate selection, packed batch pop, formation preview가 prefill class별 상한을 동일하게 사용한다.
- HTTP runtime은 visual engine의 실제 `maxSupportedVisionPrefillBatchSize`를 scheduler에 전달한다.
- text P8과 external P4가 같은 scheduler 구현에서 각각 정확히 형성되는 단위 테스트를 추가했다.

이 cap은 policy가 아니다. TensorRT optimization profile의 실행 가능 shape를 candidate generator에 정확히 반영하는 mechanism이다.

### 실제 request trace 생성기

`benchmarks/phase_serving/build_encoder_overlap_trace.py`는 세 trace family를 만든다.

- `ep`: vision 8개와 1-token text prefill 24개가 동시에 도착한다.
- `ed`: output 192인 resident decode D개가 먼저 시작하고 300 ms 뒤 vision 8개가 도착한다.
- `mixed`: resident decode D개가 먼저 시작하고 300 ms 뒤 vision 8개와 text prefill 24개가 같이 도착한다.

Vision request는 실제 `woman_and_dog.jpeg`를 `file://` request content로 전달한다. Text/vision 요청은 같은 timestamp에서 한 class가 먼저 유리해지지 않도록 교차 배치한다.

## 환경과 고정 조건

| 항목 | 값 |
|---|---|
| GPU | NVIDIA RTX 3080 10 GiB |
| Model | `nvidia/Cosmos-Reason2-2B` |
| Precision | FP16, 양자화 없음 |
| Runtime | TensorRT 11.0.0 / CUDA 13.3 container |
| LLM engine | max P8 / D64 / batch80 |
| Vision engine | configured max E8; 실제 batch는 image-token profile과 arrival formation에 따라 E1~E8 |
| KV | stable indexed-paged, 256 pages |
| Stable slots | 80 |
| Prefill chunk | fixed 128 |
| Global scheduler | active |
| Warmup overlap probe | disabled |
| EOS | ignored, requested output length 고정 |
| Request adapter | async HTTP/SSE, 8 workers |
| Activity | E=0001, P=0010, D=0100, Copy=1000 |

P8은 현재 LLM engine의 물리적 max text-prefill batch다. External/vision prefill은 별도 P4 profile을 사용한다. Visual engine 자체는 E8까지 실행 가능하지만, `woman_and_dog.jpeg`처럼 image token 수가 큰 요청은 total-token profile 때문에 한 실행에서 E4가 상한이다. 작은 실제 PNG와 충분한 backlog를 사용했을 때는 telemetry에서 E8 한 번 실행을 직접 확인했다. 즉 configured E8, profile-feasible E8, observed E8은 구분해야 한다.

## 전체 실험 수

Screening은 총 81점이다.

| Family | 조합 | 점 수 |
|---|---|---:|
| E+D | E 1/2/4 × D 8/32 × 0/50/100% | 18 |
| E+P | E 1/2/4 × P 2/4/8 × 0/50/100% | 27 |
| Mixed | E4 × P 2/4/8 × D 8/32 × E+P/E+D × 0/50/100% | 36 |

Screening 뒤 대표 9개 VLM point와 대표 8개 text point를 각각 3회 반복했다. 추가로 실제 E8 single-run 6점과 E8/D32 2점의 3회 확인을 수행했다. Fresh vLLM은 주 비교 trace 세 종류와 E8 진단 trace를 각각 3회 실행했다.

## Correctness와 실행 fidelity

| Gate | 결과 |
|---|---:|
| Screening output complete | 81/81 |
| Screening action-fidelity violation | 0 |
| Trace별 Current token hash 종류 | 각 1개 |
| VLM 3회 확인 token deterministic | 9/9 |
| Text 3회 확인 token deterministic | 8/8 |
| Peak Current VLM VRAM | 9,325~9,389 MiB |
| OOM / invalid request / timeout | 0 |

`requested 100%`는 전체 GPU 시간의 100%가 overlap이라는 뜻이 아니다. E와 P 또는 D가 동시에 ready인 hard-feasible decision boundary를 모두 overlap으로 선택한다는 뜻이다. 실제 active overlap은 E+D에서 약 2~8%, E+P에서 약 2~45%였다.

## Text P+D screening: P2/P4/P8 × D8/D32/D64

각 cell은 `0% / 50% / 100%` 순서다.

| P | D | token/s | TTFT p95 ms | TPOT p95 ms | E2E p95 ms |
|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 1055.7 / 1072.6 / 1078.1 | 247.9 / 270.2 / 287.1 | 7.694 / 7.541 / 7.493 | 1981.5 / 1950.7 / 1939.9 |
| 2 | 32 | 3427.3 / 3419.8 / 3423.3 | 244.9 / 279.6 / 284.2 | 9.321 / 9.123 / 9.122 | 2401.6 / 2401.3 / 2399.6 |
| 2 | 64 | 5613.2 / 5587.8 / 5470.7 | 309.2 / 363.6 / 417.0 | 11.301 / 10.907 / 11.146 | 2922.9 / 2929.4 / 2992.8 |
| 4 | 8 | 1124.9 / 1130.8 / 1135.4 | 162.8 / 175.8 / 184.1 | 7.222 / 7.159 / 7.137 | 1862.5 / 1850.0 / 1842.5 |
| 4 | 32 | 3683.4 / 3717.4 / 3699.6 | 162.2 / 173.7 / 188.1 | 8.663 / 8.474 / 8.508 | 2234.6 / 2212.9 / 2223.7 |
| 4 | 64 | 6136.8 / 6169.8 / 6143.6 | 168.7 / 199.5 / 226.9 | 10.350 / 10.079 / 10.103 | 2673.3 / 2654.3 / 2667.7 |
| 8 | 8 | 1162.9 / 1166.6 / 1173.7 | 111.7 / 125.0 / 124.4 | 6.989 / 6.967 / 6.899 | 1801.3 / 1795.4 / 1784.8 |
| 8 | 32 | 3838.3 / 3839.5 / 3843.6 | 113.3 / 122.1 / 128.4 | 8.295 / 8.241 / 8.276 | 2143.6 / 2143.4 / 2140.7 |
| 8 | 64 | 6411.5 / 6428.9 / 6381.6 | 111.2 / 123.1 / 138.3 | 9.906 / 9.816 / 9.862 | 2558.3 / 2551.3 / 2569.6 |

단발 screening만 보면 작은 이득점이 많아 보인다. 그러나 대표점을 3회 반복하면 큰 P/D에서 그 이득은 사라졌다.

| P/D | Setting | token/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 |
|---|---:|---:|---:|---:|---:|
| P2/D8 | 0% | 1057.88 | 125.45 / 247.60 | 7.617 / 7.680 | 402.96 / 1977.28 |
| P2/D8 | 100% | 1075.10 | 147.27 / 290.91 | 7.445 / 7.497 | 418.19 / 1944.61 |
| P4/D32 | 0% | 3696.10 | 80.09 / 153.36 | 8.490 / 8.627 | 945.59 / 2226.45 |
| P4/D32 | 50% | 3683.50 | 90.49 / 178.68 | 8.456 / 8.553 | 952.53 / 2232.36 |
| P8/D32 | 0% | 3845.70 | 60.59 / 113.77 | 8.209 / 8.286 | 900.08 / 2139.82 |
| P8/D32 | 100% | 3836.80 | 69.12 / 125.70 | 8.181 / 8.267 | 903.68 / 2144.62 |
| P8/D64 | 0% | 6439.98 | 370.60 / 2170.62 | 9.338 / 9.478 | 1732.05 / 2488.76 |
| P8/D64 | 50% | 6390.28 | 375.72 / 2176.42 | 9.567 / 9.712 | 1771.19 / 2566.25 |

P8/D64 trace는 D64와 late P48, 총 112요청이다. Client max-inflight 80이므로 두 admission wave가 생기고 TTFT에 client dispatch delay가 포함된다. 0%와 50%가 동일한 조건이므로 A/B 자체는 유효하지만 다른 80-request trace의 TTFT와 직접 비교해서는 안 된다.

## E+D screening

| E | D | Overlap | token/s | vs 0% | TTFT p95 | TPOT p95 | E2E p95 | 실제 E+D |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 0 / 50 / 100 | 934.8 / 957.4 / 957.3 | 0 / +2.43 / +2.41% | 503.8 / 469.5 / 467.8 | 8.532 / 8.331 / 8.332 | 1651.1 / 1612.2 / 1612.1 | 0 / 5.90 / 8.34% |
| 1 | 32 | 0 / 50 / 100 | 3196.5 / 3259.0 / 3270.0 | 0 / +1.96 / +2.30% | 426.9 / 431.3 / 424.4 | 9.903 / 9.718 / 9.686 | 1921.7 / 1885.3 / 1879.5 | 0 / 5.57 / 7.61% |
| 2 | 8 | 0 / 50 / 100 | 949.9 / 957.8 / 958.0 | 0 / +0.83 / +0.86% | 453.1 / 446.3 / 440.8 | 8.398 / 8.327 / 8.327 | 1624.9 / 1611.4 / 1611.4 | 0 / 4.36 / 6.58% |
| 2 | 32 | 0 / 50 / 100 | 3228.6 / 3252.5 / 3270.5 | 0 / +0.74 / +1.30% | 352.9 / 403.1 / 401.6 | 9.814 / 9.727 / 9.684 | 1903.0 / 1889.2 / 1879.2 | 0 / 3.23 / 4.70% |
| 4 | 8 | 0 / 50 / 100 | 952.6 / 956.2 / 960.7 | 0 / +0.38 / +0.86% | 423.2 / 433.7 / 424.1 | 8.374 / 8.342 / 8.300 | 1620.4 / 1613.9 / 1606.4 | 0 / 2.34 / 4.98% |
| 4 | 32 | 0 / 50 / 100 | 3255.2 / 3250.9 / 3263.6 | 0 / -0.13 / +0.26% | 378.9 / 395.3 / 380.6 | 9.730 / 9.747 / 9.702 | 1887.5 / 1890.5 / 1882.3 | 0 / 3.71 / 2.08% |

E가 커질수록 encoder 자체가 GPU를 더 오래 점유하고 D와의 contention이 커진다. E1은 작은 kernel group을 D의 빈 공간에 넣는 효과가 있지만 E4는 단독 실행 자체가 약 90 ms이므로 throughput gain이 1% 아래로 줄었다.

3회 반복 E1/D32 100% 대 0%는 다음과 같다.

- throughput: +2.43%
- TTFT mean: +12.20%
- TTFT p95: -1.18%
- TPOT p95: -2.44%
- E2E mean: -1.48%
- E2E p95: -2.40%

## E+P screening

| E | P | req/s 0/50/100 | TTFT mean 0/50/100 | TTFT p95 0/50/100 | 실제 overlap 100% |
|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 59.37 / 69.22 / 72.07 | 176.0 / 255.7 / 233.8 | 484.9 / 459.6 / 441.7 | 42.50% |
| 1 | 4 | 66.44 / 76.95 / 74.16 | 136.8 / 189.5 / 177.3 | 417.1 / 414.4 / 428.6 | 38.55% |
| 1 | 8 | 67.56 / 77.02 / 73.09 | 150.0 / 164.1 / 206.8 | 413.4 / 374.4 / 435.2 | 42.01% |
| 2 | 2 | 64.43 / 66.85 / 68.40 | 164.3 / 282.3 / 268.7 | 448.9 / 475.3 / 465.5 | 24.02% |
| 2 | 4 | 69.15 / 77.63 / 75.24 | 159.0 / 211.0 / 262.4 | 439.9 / 410.3 / 422.3 | 35.40% |
| 2 | 8 | 73.72 / 78.75 / 79.67 | 156.2 / 226.8 / 204.8 | 410.8 / 403.9 / 399.4 | 44.58% |
| 4 | 2 | 64.38 / 66.46 / 70.42 | 173.7 / 288.3 / 254.4 | 474.2 / 460.3 / 451.4 | 28.60% |
| 4 | 4 | 68.19 / 73.61 / 73.39 | 179.6 / 247.2 / 274.6 | 448.9 / 413.7 / 433.6 | 30.22% |
| 4 | 8 | 73.10 / 78.81 / 80.89 | 166.3 / 194.8 / 240.7 | 416.9 / 384.5 / 374.0 | 43.67% |

Pure E+P trace는 모든 요청의 output이 1 token이므로 `token/s == req/s`, TPOT는 0이다. 이 표는 E/P drain과 TTFT를 보는 실험이지 decode throughput 실험이 아니다.

3회 반복에서 다음 두 점을 확인했다.

| E/P | Setting | req/s | TTFT mean / p95 | E2E mean / p95 |
|---|---:|---:|---:|---:|
| E1/P8 | 0% | 66.98 | 153.36 / 416.46 | 153.58 / 416.48 |
| E1/P8 | 50% | 77.14 | 160.57 / 372.57 | 160.75 / 372.59 |
| E4/P8 | 0% | 74.82 | 164.13 / 399.75 | 164.40 / 399.82 |
| E4/P8 | 100% | 76.50 | 229.00 / 397.53 | 229.31 / 397.55 |

E1/P8 50%는 좋은 Pareto point지만 E4/P8 100%는 아니다. E4/P8은 req/s +2.24%를 위해 TTFT 평균을 39.52% 희생했다.

## Mixed screening

각 cell의 값은 `0% / 50% / 100%`다. E는 모두 4다.

| Kind | P | D | token/s | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | 실제 overlap 100% |
|---|---:|---:|---:|---:|---:|---:|---:|
| E+D | 2 | 8 | 879.3 / 884.1 / 889.6 | 493.8 / 467.8 / 488.8 | 9.208 / 9.162 / 9.105 | 1779.1 / 1769.7 / 1758.8 | 5.76% |
| E+D | 2 | 32 | 2910.4 / 2891.9 / 2916.0 | 481.8 / 494.6 / 509.4 | 10.961 / 11.027 / 10.932 | 2119.8 / 2133.8 / 2116.3 | 4.95% |
| E+D | 4 | 8 | 922.1 / 921.9 / 928.2 | 458.4 / 437.0 / 439.3 | 8.789 / 8.790 / 8.728 | 1700.0 / 1700.3 / 1688.4 | 4.14% |
| E+D | 4 | 32 | 3062.4 / 3102.2 / 3120.5 | 440.2 / 450.9 / 435.7 | 10.408 / 10.273 / 10.210 | 2014.9 / 1988.8 / 1976.9 | 4.71% |
| E+D | 8 | 8 | 934.1 / 947.4 / 948.0 | 454.6 / 420.1 / 428.9 | 8.674 / 8.549 / 8.547 | 1677.6 / 1654.2 / 1653.0 | 4.72% |
| E+D | 8 | 32 | 3147.9 / 3172.2 / 3201.1 | 444.2 / 427.2 / 412.1 | 10.103 / 10.027 / 9.936 | 1959.5 / 1944.2 / 1926.7 | 5.41% |
| E+P | 2 | 8 | 872.1 / 885.7 / 891.0 | 549.3 / 481.4 / 471.9 | 9.286 / 9.142 / 9.088 | 1794.1 / 1766.4 / 1755.9 | 7.55% |
| E+P | 2 | 32 | 2909.8 / 2933.2 / 2933.8 | 453.8 / 466.9 / 436.4 | 10.946 / 10.868 / 10.862 | 2120.9 / 2103.4 / 2102.2 | 5.77% |
| E+P | 4 | 8 | 916.7 / 934.4 / 929.4 | 468.0 / 435.0 / 419.4 | 8.843 / 8.674 / 8.720 | 1709.7 / 1677.2 / 1686.5 | 7.36% |
| E+P | 4 | 32 | 3093.2 / 3089.2 / 3133.1 | 395.3 / 415.1 / 403.6 | 10.302 / 10.320 / 10.174 | 1994.4 / 1996.8 / 1968.8 | 6.20% |
| E+P | 8 | 8 | 944.7 / 946.4 / 954.6 | 424.2 / 412.5 / 403.0 | 8.575 / 8.563 / 8.489 | 1659.1 / 1656.0 / 1641.6 | 8.77% |
| E+P | 8 | 32 | 3181.7 / 3210.7 / 3217.8 | 427.8 / 394.3 / 406.2 | 9.995 / 9.905 / 9.882 | 1939.0 / 1920.8 / 1916.5 | 8.42% |

Mixed E4/P8/D32를 3회 반복한 결과는 다음과 같다.

| Action | token/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 |
|---|---:|---:|---:|---:|
| Serial | 3165.18 | 101.95 / 432.53 | 9.948 / 10.050 | 1054.93 / 1948.89 |
| E+P 100% | 3207.64 | 136.54 / 385.54 | 9.810 / 9.911 | 1071.35 / 1923.36 |
| E+D 100% | 3210.05 | 114.48 / 414.31 | 9.807 / 9.910 | 1047.90 / 1921.32 |

E+P는 TTFT tail을 가장 많이 줄이지만 평균 first-token path를 더 크게 희생한다. E+D는 throughput과 decode tail을 비슷하게 개선하면서 전체 평균 E2E도 개선한다. 따라서 이 shape에서는 E+D가 더 균형적인 action이다.

## 실제 E8 profile 실험

E8을 단순 설정값이 아니라 실제 TensorRT execution batch로 확인하기 위해 작은 실제 PNG와 32개 동시 vision request를 사용했다. 8개 요청만 보냈을 때는 host arrival 순서 때문에 E8, E6+E2, E4+E4가 run마다 달라졌다. 32개 backlog에서 serial은 E8을 안정적으로 형성했다.

### E8 + D32, 3회 중앙값

| Action | token/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | E batch 수 / max | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| Serial | 3601.25 | 102.21 / 244.36 | 8.716 / 8.818 | 934.73 / 1712.53 | 5 / 8 | 9325 |
| E+D 100% | 3619.65 | 107.50 / 253.51 | 8.666 / 8.770 | 935.23 / 1703.97 | 7 / 8 | 9339 |

E+D 100%의 변화는 다음과 같다.

- throughput: +0.51%
- TTFT mean/p95: +5.18% / +3.74%
- TPOT mean/p95: -0.57% / -0.54%
- E2E mean/p95: +0.05% / -0.50%
- peak VRAM: +14 MiB

Overlap이 decode tail을 조금 줄였지만 encoder 실행은 5회에서 7회로 파편화됐다. action 선택이 다음 도착/queue drain 시점까지 바꾸기 때문이다. 따라서 E8/D32 static 100%는 Pareto point가 아니다. online selector는 E+D의 service compression뿐 아니라, 그 action 때문에 잃게 되는 미래 E formation gain도 함께 보아야 한다.

### E8 + P8, 단발 진단

| Action | req/s | TTFT mean / p95 | E batch 수 / observed max |
|---|---:|---:|---:|
| Serial | 265.70 | 92.75 / 162.39 | 4 / 8 |
| E+P 50% | 259.41 | 107.00 / 173.81 | 6 / 8 |
| E+P 100% | 232.74 | 119.90 / 218.15 | 8 / 5 |

E+P 100%는 throughput -12.40%이고 실제 최대 E batch도 5로 내려갔다. 큰 E backlog가 있는 동안에는 E와 P를 겹치는 것보다 E8 formation을 먼저 보존하는 편이 낫다.

### E8 vLLM 진단의 해석 제한

같은 64-request HTTP trace를 vLLM 0.27.1에서도 3회 실행했다. vLLM 중앙값은 1681.00 token/s, TTFT 759.22/2273.59 ms, TPOT 18.791/18.855 ms, E2E 2553.10/3671.62 ms, peak 9853 MiB였다. Current serial은 수치상 throughput +114.23%, E2E p95 -53.36%다.

그러나 이 값은 headline 공정 비교로 사용하지 않는다. E8을 만들기 위해 사용한 tiny PNG는 Current에서 작은 native vision shape로 처리됐지만, 기존 vLLM production container는 `mm-processor-kwargs`로 516,096 pixel을 강제해 vision request당 prompt token이 549였다. 동일 HTTP request이지만 encoder work가 동일하지 않다. 이 결과가 보여주는 것은 vLLM 서버가 같은 arrival/output contract를 완주했다는 것과, image preprocessing/token contract를 맞추지 않으면 VLM 수치를 직접 비교하면 안 된다는 점이다. 공정한 vLLM 비교의 주 결과는 바로 아래의 세 trace 표를 유지한다.

## Fresh vLLM 비교

vLLM 0.27.1 VLM production container를 한 번 fresh start하고 세 새 trace를 각각 3회 실행했다. Model은 같은 local FP16 checkpoint, `max-num-seqs=80`, `max-num-batched-tokens=8192`, chunked prefill on, prefix cache off다. Current와 vLLM 모두 실제 HTTP arrival와 ignore-EOS를 사용했다.

| Workload | System | token/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | Peak MiB |
|---|---|---:|---:|---:|---:|---:|
| E+D D32 | Current E1+D32 100% | 3269.08 | 116.25 / 421.18 | 9.588 / 9.682 | 1581.39 / 1879.05 | 9369 |
| E+D D32 | vLLM production | 2967.42 | 148.95 / 564.89 | 10.404 / 10.468 | 1738.70 / 2071.22 | 8821 |
| Pure E+P | Current E1+P8 50% | 77.14 req/s | 160.57 / 372.57 | n/a | 160.75 / 372.59 | 9351 |
| Pure E+P | vLLM production | 52.46 req/s | 167.23 / 536.60 | n/a | 167.46 / 536.65 | 8821 |
| Mixed D32 | Current E4+P8+D32, E+D 100% | 3210.05 | 114.48 / 414.31 | 9.807 / 9.910 | 1047.90 / 1921.32 | 9369 |
| Mixed D32 | vLLM production | 2938.54 | 143.27 / 622.39 | 10.585 / 10.632 | 1152.11 / 2099.40 | 8883 |

Current의 vLLM 대비 이점은 다음과 같다.

| Workload | Throughput | TTFT mean | TTFT p95 | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|
| E+D D32 | +10.17% | -21.95% | -25.44% | -7.51% | -9.05% | -9.28% |
| Pure E+P | +47.03% | -3.99% | -30.57% | n/a | -4.01% | -30.57% |
| Mixed D32 | +9.24% | -20.10% | -33.43% | -6.79% | -9.04% | -8.48% |

이번 fresh vLLM은 Current보다 486~548 MiB 적은 peak VRAM을 사용했다. 이전 text-only 비교에서 Current가 더 적은 메모리를 썼던 결과와 반대인 이유는 Current가 이번에는 visual engine과 독립 E/P/D TensorRT context를 모두 적재했기 때문이다. 이는 overlap의 성능 이득과 별개로 context/workspace reuse가 여전히 필요한 이유다.

## 무엇을 production에 적용할 것인가

실험 percentage를 workload별로 고정하면 안 된다. 이번 곡선은 production heuristic을 만들기 위한 profile tuning이 아니라 online selector가 배워야 하는 action-space의 ground truth다.

관측된 규칙은 다음과 같다.

1. 큰 P8/D32와 P8/D64의 P+D는 serial을 유지한다.
2. E1처럼 짧은 encoder action은 D32와 overlap할 가치가 있다.
3. E4처럼 긴 encoder action은 E+P보다 E+D가 mixed workload에서 더 안전하다.
4. overlap으로 TTFT p95가 좋아져도 TTFT mean이 나빠질 수 있다. 평균과 tail을 모두 보호해야 한다.
5. 후보의 실제 overlap 비율이 높다는 사실만으로 좋은 action은 아니다. isolated work 대비 makespan compression과 protected request delay를 같이 봐야 한다.

Production selector는 다음 조건을 사용해야 한다.

```text
hard feasibility
  -> predicted protected-slack violation 최소화
  -> deadline-equivalent 후보 중 observed service compression 최대화
  -> mean critical-path slowdown guard
```

즉 static `E1/D32=100%` 규칙을 넣지 않는다. Process-local observation key에 E/P/D batch shape, prompt/context bucket, outstanding set을 넣고, 실제 CUDA completion에서 makespan과 phase별 slowdown을 갱신한다. 알려지지 않은 shape는 충분한 slack이 있을 때만 bounded probe한다.

## 남은 정확한 실험

현재 engine에서 가능한 text P8, external P4, encoder E8, decode D64까지 실제 batch를 확인했다. 다음은 이번 matrix의 누락이 아니라 새 profile 계약 또는 새 연구 질문이다.

- 큰 semantic image에서도 E8을 허용하는 visual total-token profile 재구축과 E4 두 번 실행 비교
- E+P와 E+D를 동시에 후보로 두고 online selector가 고르는 production `-1` 재검증
- Copy stream이 실제로 활성화되는 host/device staging 또는 dynamic slab resize trace
- memory pressure 70/80/90%에서 동일 matrix의 feasibility 변화
- 12-workload regression suite에서 새로운 selector의 cross-workload gate

## 결과 파일

- 전체 81점: `benchmarks/phase_serving/results/exact-epd-overlap-matrix-20260830.csv`
- VLM 3회 확인 9점: `benchmarks/phase_serving/results/exact-epd-overlap-confirmation-20260830.csv`
- Text 3회 확인 8점: `benchmarks/phase_serving/results/exact-text-overlap-confirmation-20260830.csv`
- Fresh vLLM 비교: `benchmarks/phase_serving/results/exact-epd-vllm-comparison-20260830.csv`
- 실제 E8 실행: `benchmarks/phase_serving/results/exact-e8-overlap-20260830.csv`
- E8 vLLM 진단: `benchmarks/phase_serving/results/exact-e8-vllm-diagnostic-20260830.csv`
- Trace 생성기: `benchmarks/phase_serving/build_encoder_overlap_trace.py`
