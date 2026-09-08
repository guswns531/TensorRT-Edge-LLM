# Encoder-inclusive horizon 검증

## 가설

Note255의 frontier 전달은 동작했지만 Mixed tail이 악화됐다. 현재 selector는 P/D에
`predictedHorizonUs`가 있으면 그것을 쓰고, E에는 없으면 immediate action cost를 사용한다.
따라서 같은 최종 selector 안에서 비교 시간 범위가 다를 수 있다.

이것은 관측된 코드 차이이며, 회귀의 확정 원인이라고 미리 단정하지 않는다.

## 최소 변경

`phaseGlobalExtendEncoderHorizon()`은 기존 P/D horizon을 E 하나만큼 확장한다.

- Serial E: E cost + 최저 serial P/D horizon.
- Serial P/D 및 P+D: 기존 horizon + E cost.
- E+P/E+D: pair cost + 대응하는 serial P/D horizon의 잔여 비용.
- 공통 reference: E isolated work + serial P/D reference의 최대값.
- Deadline/uncertainty, candidate membership, memory, KV, TRT shape는 변경하지 않는다.

이는 bounded service-cost 근사다. 현재 ready encoder 이후 발생할 모든 P/D token trajectory를
정확하게 시뮬레이션한 request-union oracle은 아니다. RLS cost uncertainty도 그대로 사용한다.
텍스트-only와 live residual augmentation, 강제 warmup probe에는 적용하지 않는다.

옵션 `TRT_EDGELLM_GLOBAL_ENCODER_HORIZON=1`은 기본 off다. Note255의
`TRT_EDGELLM_GLOBAL_PD_FRONTIER=1`과 독립적으로 비교한다. 변수는 존재 여부로 파싱되므로
off는 제거해야 한다. 모델별 또는 workload별 분기는 추가하지 않는다.

## 실험 계약

Source base `89e1149`, 기존 Cosmos FP16 native engines와 generic calibration, P8/D64/E4,
P128, KV256 pages, slots80, client64, ignore-EOS 유지. Fresh process마다 동일 calibration
요청을 주되 posterior 자체가 동일하다고 주장하지 않는다. 계측 없이 same-binary 비교한다.
결과: `.local/results/v0101-forward-port/encoder-horizon-20260908/`.

첫 검증은 Mixed에서 baseline → horizon → frontier+horizon의 3개 구성이다.
효과가 있는 구성이 나오면 교대 반복 후 전체12로 확대한다. 실패한 구성을 기본으로 승격하거나
워크로드별 승자를 조합해 하나의 정책 성능으로 보고하지 않는다.

## Mixed 결과 및 기각

각 1회, telemetry off. 실험 binary SHA256:
`1dfaa89832edc39dd38c8ab45409694aa6081f5c36ca62e09bdf627f60477104`.

| 구성 | token/s | TTFT mean/p95 (ms) | TPOT mean/p95 (ms) | E2E mean/p95 (ms) |
|---|---:|---:|---:|---:|
| Baseline | 1047.85 | 731.45/2184.81 | 34.91/43.34 | 2383.24/2724.12 |
| Horizon | 1051.65 | 717.56/2220.94 | 43.55/70.01 | 2656.88/2769.32 |
| Frontier + Horizon | 1046.46 | 681.21/2198.09 | 44.32/71.09 | 2647.31/2782.34 |

Horizon 두 구성 모두 TPOT/E2E를 악화시켰다. 단발 결과지만 승격 근거가 없으므로 구현을
production 소스에서 제거하고 `rejected-horizon.patch`로 결과 디렉토리에 보존했다.
Patch의 core API/config에 example 환경 연결 한 줄을 추가하면 동일 옵션을 재구성할 수 있다.
이번 실패를 반복 재검증된 causal 결론으로 과장하지 않는다.

Horizon-only에서는 최종 frontier에 대응 단독 phase가 없는 E-overlap을 변환하지 않는다.
따라서 완전한 모든-candidate normalization은 아니다. Combined에서도 실패했으므로 단순히
공통 분모를 추가하는 것만으로 실행–formation/latency trade-off를 해결했다고 주장할 수 없다.

## 이어지는 host 경로 최적화

정책 변경은 기각하고 다음 policy-neutral 최적화로 이동했다.

1. GlobalQueueSelection의 후보 vector를 preview 저장소로 move한다. Request IDs/slot vector의
   deep copy를 제거하고 선택된 candidate도 move한다.
2. 선택된 candidate를 return 직전까지 const reference로 사용해 중간 사본을 없앤다.
3. 최종 E/P/D에서 formation evaluator가 후보를 변경하지 않았으면 첫 selector 결과와 audit을
   재사용한다. Formation이 실제 실행된 경우에는 기존처럼 다시 select한다.

Workload 이름, deadline, RLS, 후보/row ordering, engine, KV pool은 바꾸지 않는다.
이 변경은 compiler나 TensorRT kernel 최적화가 아니라 scheduler CPU 비용 감소다.
실제 serving 효과는 아래 별도 재측정으로 판정한다.

## Host 최적화 full12 결과

각 1회, instrumentation off. 12/12 greedy token hash가 note254와 일치했다.
Before는 note254의 동일 engine/cap/config 단발 기록이다. vLLM은 해당 문서의 동일 계약 frozen 결과를
재사용한다. 과거 before와 이번 after를 교대 측정한 결과가 아니므로 아래 차이는 먼저 관측치로 본다.

| Workload | Variant | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|
| balanced | before | 4124.55 | 68.43/172.74 | 13.44/15.14 | 1212.69/1915.36 |
| balanced | host-after | 4325.55 | 71.59/172.43 | 12.72/14.62 | 1153.89/1765.15 |
| balanced | frozen vLLM | 4318.34 | 110.37/247.20 | 12.23/13.71 | 1154.26/1771.51 |
| bimodal | before | 1853.05 | 1970.33/4896.14 | 18.25/26.61 | 4531.50/10051.58 |
| bimodal | host-after | 1860.84 | 1938.72/4286.55 | 18.41/25.88 | 4482.52/9607.41 |
| bimodal | frozen vLLM | 1852.31 | 1573.46/2609.84 | 22.46/36.31 | 4689.19/9283.93 |
| decode-heavy | before | 4920.27 | 69.71/193.03 | 11.34/12.12 | 2999.11/4623.94 |
| decode-heavy | host-after | 4996.52 | 72.58/186.39 | 11.12/11.93 | 2945.20/4477.45 |
| decode-heavy | frozen vLLM | 4943.41 | 113.96/310.66 | 11.03/11.60 | 2969.50/4486.02 |
| late-vision | before | 2331.93 | 146.98/525.95 | 10.10/10.15 | 1594.01/1977.70 |
| late-vision | host-after | 2372.35 | 133.69/520.13 | 9.93/9.98 | 1556.72/1943.43 |
| late-vision | frozen vLLM | 2359.23 | 153.40/631.51 | 9.90/9.93 | 1576.03/1954.46 |
| long-prefill | before | 1245.57 | 2038.82/2616.33 | 24.85/30.30 | 4143.37/5829.47 |
| long-prefill | host-after | 1239.48 | 2044.46/2624.96 | 24.99/29.67 | 4167.10/5668.18 |
| long-prefill | frozen vLLM | 1127.42 | 1925.36/3027.37 | 31.92/36.89 | 4643.91/6579.62 |
| mixed | before | 1007.69 | 793.57/2380.34 | 34.08/42.74 | 2415.64/2838.40 |
| mixed | host-after | 1080.18 | 803.96/2323.15 | 37.71/61.76 | 2543.05/2683.48 |
| mixed | frozen vLLM | 921.48 | 874.56/2541.43 | 46.97/84.02 | 3008.38/3140.85 |
| multi-image | before | 293.47 | 243.57/324.23 | 9.55/12.80 | 539.47/544.92 |
| multi-image | host-after | 292.64 | 247.38/327.57 | 9.47/12.71 | 540.80/546.20 |
| multi-image | frozen vLLM | 244.52 | 259.81/402.58 | 12.42/16.32 | 644.30/653.90 |
| poisson | before | 1729.19 | 248.65/942.27 | 24.88/42.24 | 1849.08/2312.92 |
| poisson | host-after | 1815.25 | 222.50/815.65 | 23.98/40.93 | 1755.42/2211.23 |
| poisson | frozen vLLM | 1800.07 | 438.11/902.68 | 22.19/45.67 | 1800.22/2266.55 |
| short | before | 2378.73 | 104.59/183.35 | 12.88/22.64 | 346.61/428.57 |
| short | host-after | 2344.64 | 99.02/188.72 | 13.58/22.23 | 348.14/429.95 |
| short | frozen vLLM | 1983.53 | 174.92/263.97 | 13.36/24.88 | 426.71/503.71 |
| text-heavy | before | 1806.60 | 336.15/1212.49 | 27.51/41.71 | 1754.61/1832.85 |
| text-heavy | host-after | 1810.09 | 331.07/1117.31 | 27.70/41.74 | 1758.37/1837.07 |
| text-heavy | frozen vLLM | 1634.76 | 421.58/1232.20 | 29.21/47.36 | 1943.42/2037.81 |
| vision-heavy | before | 627.82 | 1436.05/3392.26 | 35.10/50.56 | 2787.64/3844.31 |
| vision-heavy | host-after | 638.87 | 1404.64/3369.85 | 31.84/39.08 | 2647.84/3782.19 |
| vision-heavy | frozen vLLM | 579.20 | 1710.70/3691.37 | 63.70/119.58 | 4119.14/4229.37 |
| wave-drain | before | 96.12 | 270.45/373.95 | 9.40/13.23 | 561.86/623.24 |
| wave-drain | host-after | 96.25 | 255.27/403.86 | 10.33/13.88 | 575.63/621.07 |
| wave-drain | frozen vLLM | 95.85 | 252.76/418.60 | 12.43/17.26 | 637.86/649.42 |

## 변경 전 재빌드와 ABBA 비교

두 바이너리를 보관/재빌드하여 A1(Balanced,Mixed) → B1 → B2 → A2 순서로 실행했다.
A는 변경 전, B는 host 최적화 후다. 각 workload/variant n=2, 모두 greedy token hash가 일치했다.
A SHA256 `cc0524f27a813944e10ff7936210fa87c232baec9f6b781345f1b74c930d0e22`,
B SHA256 `b6cc00c88e5485a8df8823e44752e250000a03884489f1fdac8e62910dc9246a`.

| Run | Workload | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|
| paired-after-r1 | balanced | 4189.48 | 71.74/175.96 | 13.19/15.14 | 1193.92/1842.33 |
| paired-after-r1 | mixed | 1044.75 | 736.40/2318.17 | 34.96/43.61 | 2391.11/2734.71 |
| paired-after-r2 | balanced | 4130.13 | 67.05/175.27 | 13.42/15.69 | 1208.85/1911.65 |
| paired-after-r2 | mixed | 1044.58 | 752.73/2323.31 | 33.60/42.66 | 2356.38/2736.13 |
| paired-before-r1 | balanced | 4250.92 | 71.20/175.75 | 12.96/14.91 | 1175.17/1859.69 |
| paired-before-r1 | mixed | 1028.25 | 732.08/2304.57 | 35.51/41.90 | 2395.64/2766.56 |
| paired-before-r2 | balanced | 4172.72 | 66.62/176.46 | 13.29/15.02 | 1197.00/1869.35 |
| paired-before-r2 | mixed | 1026.65 | 750.10/2334.99 | 34.73/41.26 | 2381.70/2784.67 |

각 구성의 두 run metric 평균(두 값 median과 동일) 기준 변화:

| Workload | token/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|
| balanced | -1.23% | 0.71%/-0.28% | 1.35%/3.03% | 1.29%/0.67% |
| mixed | 1.68% | 0.47%/0.04% | -2.40%/3.73% | -0.62%/-1.45% |

처리량은 양수가 개선, latency는 음수가 개선이다. Run p95의 평균이지 pooled request p95가 아니다.
Mixed의 소폭 처리량 개선은 두 번 모두 관측됐다. 그러나 Mixed TPOTp95와 Balanced 지표는
전체 무회귀 성공을 주장할 만큼 좋아지지 않았다. n=2라 유의한 우월성/95% CI를 주장하지 않는다.
이 비교가 과거 note254 대비 +4.87% Balanced headline보다 우선한다.

## 최종 상태

- Horizon 두 정책은 기각했고 소스에서 제거했다. 남은 runtime 변경은 move 및 중복 select 제거뿐이다.
- 모델/엔진/KV/후보 정책을 바꾸지 않았다. V1/V2 선택 기능은 유지한다.
- 이번 turn은 horizon 3회, host full12 12회, rebuilt-before/after 교대 8회로 총 23회 HTTP 실행을 마쳤다.
- Full12의 단발 처리량은 frozen vLLM보다 12/12 높았지만, 이를 반복 우세나 전 지표 우세로 확대하지 않는다.
- Mixed 교대 비교에서 작은 처리량 이득은 남았지만 **모든 workload의 성능·latency 개선 목표는 미달성**이다.
- 추가 queue/selector 정책은 이 결과를 빌미로 기본화하지 않는다. 다음 큰 성능 변경 전에는
  calibration key 수(7/8)와 D cohort가 바뀌는 원인을 분리해야 한다. 이번 n=2 후속 비교에서도
  학습 posterior를 동일하게 고정한 policy replay는 아니었다.
- 비용을 줄인 코드라는 사실만으로 요청 latency의 개선을 보장하지 않는다. 이 점이 이번 결과의 핵심 제한이다.
- 최종 재빌드의 smoke SHA256은 실험한 B와 동일하다. Move 후 request/slot 보존 검증을 포함한
  scheduler 테스트 **209/209 통과**, 변경 파일 pre-commit 통과. 모든 GPU 실행은 종료했다.
