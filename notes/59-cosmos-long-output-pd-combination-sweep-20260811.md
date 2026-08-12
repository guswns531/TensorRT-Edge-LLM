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

# Cosmos long-output P/D combination sweep

## 결론

288-request saturated trace에서 output multiplier를 8×와 12×까지 늘려 `P/D` 조합을 확인했다.
큰 output에서는 D cap이 주효과이고, P8/P10/P16의 차이는 거의 사라졌다. 낮은 P도 추가로 실행한
결과 P4가 throughput과 E2E의 가장 안정적인 compromise였다.

현재 workload에서 권장 조합은 다음과 같다.

```text
long-output saturation: P4/D64
보수적 운영 cap:         P8/D56 또는 P8/D64
TTFT를 우선하는 실험:   P1/D64
```

P4/D64는 P8/D64 대비 output8×에서 `+0.1%`, output12×에서 `+0.9%` throughput이고, E2E median은
각각 약 `-2.8%`, `-3.2%`였다. P1은 joint overlap 비율이 높아졌지만 prefill queue residence가
커져 P4보다 느렸다.

## 실험 조건

| item | value |
| --- | --- |
| workload | 288 requests, Poisson λ=1,000 req/s, seed7 |
| model | `nvidia/Cosmos-Reason2-2B`, FP16 |
| engine | indexed-paged, maxBatch80, stable slots80, page pool256, page128 |
| contexts | independent TensorRT prefill/decode contexts, shared CUDA context |
| prefill chunk | fixed 128 tokens |
| output | multiplier 8× (max256) 및 12× (max384) |
| primary caps | P8/P10/P16 × D32/D48/D56/D64 |
| lower-P follow-up | P1/P4 × D56/D64 |

전체 32 case가 return code 0이었다. page pool exhaustion과 CUDA 오류는 없었다. output8×의 observed
admission pressure는 약 0.66~0.68, output12×는 약 0.77~0.84였다.

## Primary sweep: P8/P10/P16

### Output 8×

| P/D | tok/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 | observed P/D |
| --- | ---: | ---: | ---: | ---: | --- |
| P8/D32 | 3,046.8 | 4,618 / 10,445 ms | 26.22 / 29.85 ms | 8,281 / 12,998 ms | P7/D32 |
| P8/D48 | 3,777.4 | 3,679 / 8,345 ms | 18.41 / 22.01 ms | 6,571 / 10,290 ms | P7/D48 |
| P8/D56 | 4,039.4 | 3,392 / 7,692 ms | 16.25 / 19.83 ms | 6,144 / 9,636 ms | P7/D56 |
| P8/D64 | **4,251.7** | 3,314 / 7,260 ms | 14.04 / 18.10 ms | 5,824 / 9,103 ms | P7/D64 |
| P10/D56 | 4,033.0 | 3,400 / 7,707 ms | 16.23 / 19.87 ms | 6,155 / 9,652 ms | P7/D56 |
| P16/D64 | 4,237.2 | 3,327 / 7,292 ms | 14.10 / 18.21 ms | 5,848 / 9,137 ms | P7/D64 |

P8/P10/P16은 모두 observed P7에서 멈췄다. queue seed bucket과 continuation/initial bucket 조건이
실제 batch를 결정하므로 requested P cap을 10 또는 16으로 올려도 batch가 커지지 않았다.

### Output 12×

| P/D | tok/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 | observed P/D |
| --- | ---: | ---: | ---: | ---: | --- |
| P8/D32 | 3,134.4 | 5,430 / 13,615 ms | 25.56 / 28.80 ms | 10,562 / 17,269 ms | P7/D32 |
| P8/D48 | 3,903.8 | 4,500 / 10,685 ms | 18.88 / 21.26 ms | 8,454 / 13,758 ms | P7/D48 |
| P8/D56 | 4,174.4 | 4,136 / 9,771 ms | 16.93 / 19.39 ms | 7,785 / 12,732 ms | P7/D56 |
| P8/D64 | **4,421.8** | 3,860 / 9,125 ms | 15.35 / 17.42 ms | 7,242 / 11,951 ms | P7/D64 |
| P10/D56 | 4,165.2 | 4,143 / 9,791 ms | 16.97 / 19.45 ms | 7,800 / 12,760 ms | P7/D56 |
| P16/D64 | 4,401.0 | 3,885 / 9,179 ms | 15.41 / 17.53 ms | 7,287 / 12,011 ms | P7/D64 |

output12×에서도 D64가 D56보다 약 5.9% 빠르고, D32보다 약 41.1% 빠르다. output8×에서는 D64가
D56보다 약 5.3% 빠르다. 이전 output4×에서 D56이 최고였던 것과 달리, output이 길어지면서 D64의
큰 decode batch가 충분히 유지된다.

## Lower-P follow-up

| output | P/D | tok/s | TTFT median | E2E median/p95 | observed P/D | joint ratio |
| ---: | --- | ---: | ---: | ---: | --- | ---: |
| 8× | P1/D56 | 3,902.1 | 3,952 ms | 6,430 / 9,987 ms | P1/D56 | 35.5% |
| 8× | P4/D56 | 4,058.4 | 3,478 ms | 6,031 / 9,570 ms | P4/D56 | 22.7% |
| 8× | P1/D64 | 4,100.6 | 3,806 ms | 6,075 / 9,465 ms | P1/D64 | 38.8% |
| 8× | P4/D64 | **4,255.1** | 3,284 ms | 5,662 / 9,071 ms | P4/D64 | 25.8% |
| 12× | P1/D56 | 4,059.0 | 4,743 ms | 8,060 / 13,076 ms | P1/D56 | 23.8% |
| 12× | P4/D56 | 4,211.0 | 4,319 ms | 7,510 / 12,560 ms | P4/D56 | 15.9% |
| 12× | P1/D64 | 4,302.9 | 4,418 ms | 7,506 / 12,279 ms | P1/D64 | 27.0% |
| 12× | P4/D64 | **4,463.5** | 4,067 ms | 7,013 / 11,815 ms | P4/D64 | 18.0% |

P1은 P4보다 joint dispatch 비율이 높지만, 긴 output에서는 그 overlap이 prefill queue wait를
줄이지 못한다. P4는 prefill을 조금 더 amortize하면서 D64 decode stream을 유지해 가장 균형이 좋다.

## Scheduler 해석

이 결과에서 P/D 정책은 다음처럼 정리된다.

1. output이 짧으면 D32/D48도 충분하며 P cap 차이는 작다.
2. output4×에서는 D56이 좋은 saturation point였다.
3. output8×/12×에서는 D64가 계속 살아 있어 D64를 우선 선택한다.
4. requested P10/P16은 observed P7에 도달하지 못해 큰 의미가 없었다.
5. 긴 output saturation에서는 P4/D64를 기본 후보로 두고, queue가 얕거나 TTFT SLO가 우선일 때 P1/D64를 별도 평가한다.
6. page pressure가 0.8을 넘기기 시작하므로 output16× 이상에서는 page pool/admission backpressure를 함께 측정해야 한다.

## 재현 자료

```text
.local/cosmos-reason2-2b/
  pd-output-sweep-n288-m8/
  pd-output-sweep-n288-m12/
  pd-output-sweep-n288-m8-p1p4/
  pd-output-sweep-n288-m12-p1p4/
  pd-output-sweep-summary.csv
```
