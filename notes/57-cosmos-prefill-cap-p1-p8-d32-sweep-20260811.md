SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# Cosmos P1~P8/D32 independent prefill-cap sweep

## 결론

48-request short-output workload의 포화 구간에서는 `P1/D32 independent`가 가장 좋았다. P8 대비
throughput은 200 req/s에서 `+8.5%`, 1,000 req/s에서 `+8.8%`였고, clean upstream 대비로도 각각
`+12.2%`, `+6.1%`였다. 낮은 부하인 30 req/s에서는 모든 P cap이 실제 P1로 실행되어 차이가 없었다.

P1의 이득은 작은 prefill kernel 자체보다 scheduler의 128-token overlap 경계에서 나온다. fixed chunk가
128이고 `maxOverlapPrefillTokens=128`이므로 P1 candidate는 decode와 joint dispatch되지만 P2 이상은
decode-only 결정을 유발한다. P2는 overlap을 잃으면서도 batching amortization이 작아 가장 나빴다.

P1은 throughput, TTFT와 E2E tail을 개선하지만 TPOT median은 P8보다 약 2배가 된다. 따라서 하나의
고정 P를 모든 요청에 적용하기보다 decode queue가 있을 때 P1 overlap을 선택하고, decode가 비었거나
TPOT SLO가 급할 때 P4 prefill/decode burst를 선택하는 정책이 다음 구현 후보이다.

## 실험 조건

| item | value |
| --- | --- |
| model/precision | `nvidia/Cosmos-Reason2-2B`, FP16 |
| engine/KV | indexed-paged, page128, stable slots 32, raw KV 1,792 MiB |
| execution | independent TensorRT prefill/decode contexts, one CUDA context |
| requested cap | P1/P2/P4/P8, D32 |
| coarse coverage | P3/P5/P6/P7 각 1회 |
| prefill chunk | fixed 128 tokens |
| requests/output | 48 requests, 총 1,040 generated tokens |
| arrivals | Poisson 30/200/1,000 req/s, seed 7 |
| repeats | primary P1/P2/P4/P8 각 3회, 실행 순서 회전 |

36개 primary case와 12개 coarse case가 모두 return code 0이었다. 각 primary case에서 request 48개와
generated tokens 1,040개를 다시 확인했다.

## Primary 3-run median

### Throughput와 clean upstream

| arrival | P1 | P2 | P4 | P8 | clean upstream |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 772.50 | 772.03 | 773.02 | 772.82 | 568.87 |
| 200 | **888.64** | 623.54 | 818.46 | 818.74 | 792.12 |
| 1,000 | **889.80** | 627.15 | 817.65 | 817.84 | 838.58 |

단위는 generated token/s이다. 200/1,000 req/s의 P1은 clean upstream보다 각각 `12.2%`, `6.1%`
빠르다. 30 req/s에서는 P1도 upstream보다 `35.8%` 빠르지만 P8 결과와 실질적으로 같다.

### P1 대 P8 latency

| arrival | metric | P1 | P8 | P1 delta |
| ---: | --- | ---: | ---: | ---: |
| 200 | TTFT median / p95 | 461.78 / 789.60 ms | 653.08 / 1,013.77 ms | `-29.3% / -22.1%` |
| 200 | TPOT median / p95 | 17.54 / 18.00 ms | 8.65 / 12.57 ms | `+102.8% / +43.2%` |
| 200 | E2E median / p95 | 824.82 / 970.84 ms | 870.97 / 1,169.02 ms | `-5.3% / -17.0%` |
| 1,000 | TTFT median / p95 | 518.27 / 906.78 ms | 740.51 / 1,061.26 ms | `-30.0% / -14.6%` |
| 1,000 | TPOT median / p95 | 17.53 / 17.95 ms | 8.62 / 11.66 ms | `+103.3% / +54.0%` |
| 1,000 | E2E median / p95 | 903.61 / 1,096.92 ms | 936.45 / 1,227.29 ms | `-3.5% / -10.6%` |

P1은 prefill을 빠르게 admission하고 decode와 계속 겹치므로 TTFT와 aggregate completion은 좋아진다.
반면 decode step 사이마다 약 17 ms의 joint prefill makespan이 끼어 streaming TPOT은 나빠진다. 이
workload의 output은 최대 32 tokens라 TTFT/tail 이득이 TPOT 손실보다 컸다. 긴 output workload에서는
별도 검증이 필요하다.

## Dispatch 원인 분석

| arrival | cap | dispatches | joint ratio | mean observed D | observed max P/D |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | P1 | 85 | 64.7% | 11.81 | P1/D22 |
| 200 | P2 | 200 | 8.5% | 5.48 | P2/D15 |
| 200 | P4/P8 | 146 | 10.3% | 7.46 | P4/D20 |
| 1,000 | P1 | 85 | 64.7% | 11.81 | P1/D21 |
| 1,000 | P2 | 205 | 5.9% | 5.36 | P2/D15 |
| 1,000 | P4/P8 | 151 | 7.3% | 7.24 | P4/D18 |

P1은 56 prefill rows를 하나씩 joint dispatch해 decode batch가 자연스럽게 D21~D22까지 성장한다.
전체 dispatch 수도 P8의 146~151회에서 85회로 줄었다. P2는 256-token candidate가 overlap 경계를
넘어 decode-only burst와 작은 prefill batch가 번갈아 발생하고, 전체 dispatch가 200회 이상으로
증가한다. P4/P8은 overlap은 적지만 실제 P4 batching으로 실행 횟수를 amortize해 P2보다 회복한다.

P3/P5/P6/P7의 1회 coarse 결과도 같은 형태였다. 200 req/s throughput은 P3 742.9, P5~P7
818.8~819.2 token/s이고, 1,000 req/s는 P3 740.1, P5~P7 817.3~818.7 token/s였다. P5 이상은
실제 observed P가 P4를 넘지 않아 P4/P8과 같았다.

## 다음 scheduler 정책

현재 결과로 권장하는 첫 정책안은 다음과 같다.

1. decode queue가 있고 `128-token P1`의 predicted overlap이 허용되면 P1/D-ready를 joint dispatch한다.
2. decode queue가 비면 최대 P4로 prefill을 묶어 initial admission을 amortize한다.
3. TPOT pressure가 target을 넘으면 joint P1을 잠시 중단하고 decode-only burst를 실행한다.
4. P2/P3는 128-token overlap 경계를 넘으면서 P4 amortization도 얻지 못하므로 선택하지 않는다.
5. 긴 output trace에서 TPOT penalty를 재검증한 뒤 P1 joint burst 길이와 decode rescue 조건을 정한다.

## 재현 자료

```text
.local/cosmos-reason2-2b/prefill-cap-sweep-p1-p8-d32/
  r30-repeat1..3/
  r200-repeat1..3/
  r1000-repeat1..3/
  coarse-r30/
  coarse-r200/
  coarse-r1000/
  repeat-summary.csv
```
