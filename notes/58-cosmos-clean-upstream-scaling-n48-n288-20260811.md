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

# Clean upstream / current D8 / current D32 scaling comparison

## 결론

이전과 동일한 세 비교군을 유지하면서 request 수와 output token을 함께 늘렸다.

```text
clean upstream fixed-linear D8
current indexed-paged independent D8
current indexed-paged independent D32
```

같은 page128/stable-slot32 current engine에서 D32는 output length가 길어질수록 확실히 유리해졌다.
N=288 기준으로 output 1×에서는 upstream보다 여전히 `-18.3%` 느리지만, output 2×에서는 `+54.6%`,
output 4×에서는 `+107.6%` 빨랐다. D8은 모든 큰 saturation case에서 upstream보다 느렸다.

## 실험 조건

| item | value |
| --- | --- |
| model | `nvidia/Cosmos-Reason2-2B`, FP16 |
| current engine | indexed-paged, page128, stable slots32, maxBatch32 |
| current contexts | independent TensorRT prefill/decode contexts, shared CUDA context |
| current caps | requested P8/D8 and P8/D32, fixed-128 prefill chunk |
| clean upstream | unmodified fixed-linear `maxBatch=8`, D8 |
| requests | 48/96/192/288 (12-request trace repeated) |
| output | 1×/2×/4×, max output 32/64/128 tokens |
| arrivals | Poisson λ=1,000 req/s, seed7; finite burst trace |
| current repeats | exploratory first pass, every case return code 0 |

`1,000 req/s`는 지속적인 1초 부하가 아니라 finite burst trace의 Poisson parameter다. 마지막 arrival은
N48에서 33.2ms, N96에서 83.7ms, N192에서 176.9ms, N288에서 278.3ms였다. 이후 queue drain을
측정했다.

## Throughput scaling

단위는 generated token/s이다. 괄호는 clean upstream 대비 current의 변화다.

| requests | output | clean upstream | current D8 | current D32 |
| ---: | ---: | ---: | ---: | ---: |
| 48 | 1× | 838.6 | 740.4 (`-11.7%`) | 820.3 (`-2.2%`) |
| 48 | 2× | 929.3 | 959.6 (`+3.3%`) | 1,356.7 (`+46.0%`) |
| 48 | 4× | 949.3 | 1,030.3 (`+8.5%`) | 1,962.8 (`+106.8%`) |
| 96 | 1× | 1,013.6 | 734.3 (`-27.5%`) | 828.6 (`-18.2%`) |
| 96 | 2× | 1,140.9 | 926.5 (`-18.8%`) | 1,557.5 (`+36.5%`) |
| 96 | 4× | 1,166.8 | 1,022.3 (`-12.4%`) | 2,204.5 (`+88.9%`) |
| 192 | 1× | 1,018.1 | 732.7 (`-28.0%`) | 832.4 (`-18.2%`) |
| 192 | 2× | 1,143.8 | 910.4 (`-20.4%`) | 1,708.3 (`+49.4%`) |
| 192 | 4× | 1,168.4 | 1,018.1 (`-12.9%`) | 2,363.4 (`+102.3%`) |
| 288 | 1× | 1,019.6 | 733.0 (`-28.1%`) | 833.4 (`-18.3%`) |
| 288 | 2× | 1,144.7 | 906.7 (`-20.8%`) | 1,769.4 (`+54.6%`) |
| 288 | 4× | 1,168.9 | 1,013.3 (`-13.3%`) | 2,426.1 (`+107.6%`) |

핵심 전환점은 request 수보다 output length다. N을 48에서 288로 늘리면 queue가 포화되지만, output
1×에서는 각 request의 decode work가 짧아 D32가 큰 batch를 오래 유지하지 못한다. output 2×부터는
decode queue가 충분히 길어져 D32의 KV reuse와 per-step amortization이 나타난다.

## N=288 latency 비교

| output | metric | clean upstream | current D8 | current D32 |
| ---: | --- | ---: | ---: | ---: |
| 1× | E2E median / p95 | 2,428 / 5,637 ms | 4,689 / 8,114 ms | 3,846 / 7,137 ms |
| 2× | E2E median / p95 | 4,188 / 10,222 ms | 7,511 / 13,224 ms | 3,917 / 6,549 ms |
| 4× | E2E median / p95 | 7,473 / 19,207 ms | 12,903 / 22,989 ms | 5,302 / 9,229 ms |

N288/output4×에서 current D32는 upstream보다 throughput `+107.6%`, E2E median `-29.1%`, p95
`-52.0%`다. 반면 output1×에서는 D32도 upstream보다 throughput `-18.3%`, E2E median `+58.4%`다.

## 해석

- D8은 stable indexed KV의 compaction 제거만으로는 clean upstream의 full BS8 fixed-linear 경로를
  이기기 어렵다.
- D32의 이득은 request 수 자체보다 decode queue가 충분히 길고 output token이 충분히 많을 때 발생한다.
- output 1×/short request는 `D8` 또는 작은 decode cap이 latency에 유리하다.
- output 2× 이상이고 queue가 포화되면 `D32`가 throughput과 tail latency 모두 유리하다.
- 이번 current는 page128/stable-slot32라 D32의 이득을 보수적으로 측정했다. P10/D56의 maxBatch80
  결과는 별도 saturation regime이며 이 표와 직접 합치지 않는다.

## Clean upstream 방법과 한계

upstream은 output 길이별 homogeneous BS8 wall cost를 실제 fixed-linear binary로 측정하고, 원래 arrival
offset 위에서 shortest-processing-time oracle로 replay했다. N48 partial batch와 N96 full BS8 cost를
측정해 N192/N288에 적용했다. runtime 초기화와 tokenizer overhead는 제외되어 upstream에 유리하다.

output4×에서는 runtime별 EOS 조기 종료로 generated token 수가 조금 달라진다. 따라서 CSV에는
`generated_tokens`와 `requested_tokens`를 모두 보존했다. 위 표는 기존 비교와 일관되게 실제 generated
token/s를 사용한다.

## 다음 선택

현재 scheduler의 첫 정책은 다음처럼 두는 것이 합리적이다.

1. short output 또는 decode queue가 얕으면 D8을 사용한다.
2. queue depth와 예상 output length가 threshold를 넘으면 D32로 승격한다.
3. output 2×/4× saturation에서는 D32를 기본값으로 두고, TTFT pressure가 커질 때만 D8로 낮춘다.
4. 다음에는 실제 sustained arrival(수천 요청 또는 10~30초)에서 이 threshold를 재검증한다.

## 재현 자료

```text
.local/cosmos-reason2-2b/clean-compare-scaling/
  n48-m1 ... n288-m4/
  upstream-inputs/
  upstream-measurements/
  scaling-comparison-summary.csv
  scaling-compact-summary.csv
```
