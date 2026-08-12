SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Cosmos independent load sweep: P8/D8 대 P8/D32

> 후속 P1~P8 sweep에서 short-output 포화 부하의 새 최적점은 `P1/D32 independent`로 갱신됐다.
> 자세한 결과는 `notes/57-cosmos-prefill-cap-p1-p8-d32-sweep-20260811.md`를 참고한다.

## 결론

Independent TensorRT context의 주효과는 낮은 부하에서 latency를 보호하는 것이고, D32의 주효과는
모든 부하에서 TPOT과 aggregate throughput을 개선하는 것이다. 두 효과를 합친 `P8/D32 independent`가
세 부하 모두 가장 좋았다.

Clean upstream을 같은 48-request/output-1x trace에 추가하면 `P8/D32 independent`는 30 req/s에서
upstream보다 `36.0%`, 200 req/s에서 `3.3%` 빠르고, 1,000 req/s에서는 `2.5%` 느리다. 따라서
current의 강점은 낮은 부하의 continuous admission/overlap이고, 높은 포화 부하에서는 clean upstream의
고정 BS8 kernel 경로가 아직 조금 더 효율적이다.

| arrival | best throughput | observed P/D | TTFT med/p95 | TPOT med/p95 | E2E med/p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 30 req/s | 773.64 token/s | P1/D16 | 48.84 / 112.56 ms | 13.60 / 16.91 ms | 296.60 / 470.68 ms |
| 200 req/s | 818.02 token/s | P4/D20 | 653.11 / 1,014.21 ms | 8.66 / 12.58 ms | 871.32 / 1,170.21 ms |
| 1,000 req/s | 817.95 token/s | P4/D18 | 741.33 / 1,061.72 ms | 8.60 / 11.61 ms | 937.09 / 1,227.00 ms |

30 req/s에서 independent 효과가 가장 크고, queue가 포화되면 independent의 throughput 효과는
3~5%로 줄어든다. 반대로 D32는 independent에서도 D8 대비 처리량을 9~11%, TPOT median을
33~44% 개선했다.

## 실험 조건

| item | value |
| --- | --- |
| model | `nvidia/Cosmos-Reason2-2B`, FP16 |
| engine | indexed-paged, page pool 128, stable slots 32 |
| raw KV | 1,792 MiB |
| requests | 48, 동일한 12-request set을 4회 반복 |
| output | 원래 분포의 1x |
| arrival | Poisson 30 / 200 / 1,000 req/s, seed 7 |
| prefill | fixed-128 chunk, requested P8 |
| decode | requested D8 또는 D32 |
| context | shared-serialized 또는 independent-concurrent |
| repeats | 각 case 3회, 실행 순서 교차 |

arrival rate별 세 반복은 byte-identical materialized trace를 사용했다. 세 rate 사이에는 요청 내용과
output 길이는 동일하고 arrival offset만 다르다. 총 36 case가 모두 return code 0이었다. page pool
exhaustion은 없었으며 최대 observed admission pressure는 28.1%였다.

## 전체 3-run median

### 30 req/s

| D cap | context | token/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 | observed P/D | joint ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | shared | 610.49 | 99.84 / 162.61 | 36.64 / 61.00 | 755.86 / 1,337.09 | P1/D8 | 41.0% |
| 8 | independent | 697.49 | 47.11 / 111.97 | 21.57 / 35.97 | 487.60 / 855.64 | P1/D8 | 39.7% |
| 32 | shared | 724.64 | 89.99 / 180.82 | 19.37 / 23.22 | 429.15 / 716.82 | P1/D18 | 61.1% |
| 32 | independent | 773.64 | 48.84 / 112.56 | 13.60 / 16.91 | 296.60 / 470.68 | P1/D16 | 48.2% |

30 req/s에서는 요청이 decode 사이로 계속 도착해 P+D joint dispatch가 40~61%다. independent는 D8에서
shared 대비 throughput `+14.3%`, TTFT median `-52.8%`, TPOT median `-41.2%`, E2E median
`-35.5%`다. D32에서도 throughput `+6.8%`, E2E median `-30.9%`다.

### 200 req/s

| D cap | context | token/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 | observed P/D | joint ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | shared | 716.54 | 698.05 / 1,060.75 | 13.86 / 23.27 | 981.64 / 1,377.27 | P4/D8 | 9.0% |
| 8 | independent | 747.86 | 654.81 / 1,014.91 | 12.95 / 20.00 | 928.84 / 1,315.70 | P4/D8 | 9.0% |
| 32 | shared | 778.39 | 698.33 / 1,063.00 | 9.43 / 15.26 | 918.43 / 1,235.58 | P4/D20 | 10.3% |
| 32 | independent | 818.02 | 653.11 / 1,014.21 | 8.66 / 12.58 | 871.32 / 1,170.21 | P4/D20 | 10.3% |

200 req/s에서는 initial pending이 16이고 joint ratio가 약 10%로 떨어진다. independent의 throughput
효과는 D8 `+4.4%`, D32 `+5.1%`다. D32 independent는 D8 independent 대비 throughput `+9.4%`,
TPOT median `-33.1%`, E2E p95 `-11.1%`이며 TTFT는 사실상 동일하다.

### 1,000 req/s

| D cap | context | token/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 | observed P/D | joint ratio |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | shared | 714.58 | 775.24 / 1,088.49 | 16.05 / 24.68 | 1,163.87 / 1,432.69 | P4/D8 | 6.3% |
| 8 | independent | 737.90 | 746.01 / 1,059.45 | 15.37 / 21.56 | 1,118.44 / 1,386.31 | P4/D8 | 6.3% |
| 32 | shared | 788.20 | 770.33 / 1,090.88 | 9.28 / 14.52 | 966.82 / 1,275.69 | P4/D18 | 7.3% |
| 32 | independent | 817.95 | 741.33 / 1,061.72 | 8.60 / 11.61 | 937.09 / 1,227.00 | P4/D18 | 7.3% |

1,000 req/s에서는 independent의 throughput 효과가 D8 `+3.3%`, D32 `+3.8%`다. D32 independent는
D8 independent 대비 throughput `+10.9%`, TPOT median `-44.0%`, E2E median `-16.2%`다.

## Clean upstream 비교

Clean upstream은 fixed-linear `maxBatch=8` engine과 수정하지 않은 `llm_inference`를 사용했다.
48개 request의 output max별 개수는 일부가 8의 배수가 아니므로 padding하지 않고 4개 full BS8과
4개 partial batch를 실행했다. 모든 batch는 같은 output max끼리만 묶었다.

| arrival | upstream | current D8 independent | current D32 independent |
| ---: | ---: | ---: | ---: |
| 30 req/s | 568.87 token/s | 697.49 (`+22.6%`) | 773.64 (`+36.0%`) |
| 200 req/s | 792.12 token/s | 747.86 (`-5.6%`) | 818.02 (`+3.3%`) |
| 1,000 req/s | 838.58 token/s | 737.90 (`-12.0%`) | 817.95 (`-2.5%`) |

모든 구현이 실제로 1,040 tokens를 생성했으므로 이 표는 token count까지 동일하다. current는 실제
arrival queue를 online으로 처리한다. upstream은 measured batch wall cost를 original arrival 위에
replay하되 ready batch 중 실제 비용이 가장 짧은 것을 먼저 선택하는 clairvoyant SPT oracle이다.
upstream의 engine/tokenizer 초기화와 warmup도 제외했다. 따라서 비교 정책은 여전히 upstream에 유리하다.

| arrival | metric | upstream | current D32 independent | delta |
| ---: | --- | ---: | ---: | ---: |
| 30 | makespan | 1,828.20 ms | 1,344.29 ms | `-26.5%` |
| 30 | E2E median / p95 | 640.96 / 1,652.18 ms | 296.60 / 470.68 ms | `-53.7% / -71.5%` |
| 200 | makespan | 1,312.93 ms | 1,271.36 ms | `-3.2%` |
| 200 | E2E median / p95 | 649.73 / 1,286.53 ms | 871.32 / 1,170.21 ms | `+34.1% / -9.0%` |
| 1,000 | makespan | 1,240.19 ms | 1,271.47 ms | `+2.5%` |
| 1,000 | E2E median / p95 | 606.75 / 1,234.91 ms | 937.09 / 1,227.00 ms | `+54.5% / -0.6%` |

30 req/s에서는 homogeneous batch의 마지막 request를 기다려야 하는 upstream보다 current의 즉시
admission과 independent overlap이 유리하다. 200/1,000 req/s에서는 batch가 빠르게 준비되므로
clairvoyant SPT가 짧은 request를 앞에 배치해 upstream E2E median을 크게 낮춘다. 반면 current는
arrival/FIFO 기반이라 median은 높지만 p95는 200과 1,000 req/s 모두 upstream보다 낮다.

upstream public API는 first-token을 반환하지 않으므로 upstream TTFT/TPOT은 CUDA prefill과 batch
wall cost로 추정한 값이다. 위 표는 직접 측정 가능한 aggregate makespan과 E2E completion을 중심으로
해석해야 한다.

이후 모든 성능 sweep은 다음 세 열을 함께 유지한다.

1. clean upstream fixed-linear
2. current shared-serialized
3. current independent-concurrent

KV/batch 변화는 각 context mode에서 별도 행으로 추가한다. clean upstream이 지원하지 않는 D32 이상은
upstream D8을 고정 기준으로 두고, 동일 generated-token count와 arrival trace를 확인한 뒤 비교한다.

## 두 효과의 분리

### Independent context 효과

| arrival | D8 throughput | D8 TTFT med | D8 E2E med | D32 throughput | D32 TTFT med | D32 E2E med |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | +14.3% | -52.8% | -35.5% | +6.8% | -45.7% | -30.9% |
| 200 | +4.4% | -6.2% | -5.4% | +5.1% | -6.5% | -5.1% |
| 1,000 | +3.3% | -3.8% | -3.9% | +3.8% | -3.8% | -3.1% |

joint dispatch에서 independent overlap median은 0.338~0.356이었다. overlap 자체가 사라진 것이
아니라, 부하가 높을수록 전체 dispatch 중 joint dispatch 비율이 감소해 service-level 효과가 작아진다.

### D32 cap 효과

Independent context를 고정했을 때 D32 대 D8은 다음과 같다.

| arrival | throughput | TTFT median | TPOT median | E2E median / p95 |
| ---: | ---: | ---: | ---: | ---: |
| 30 | +10.9% | +3.7% | -37.0% | -39.2% / -45.0% |
| 200 | +9.4% | -0.3% | -33.1% | -6.2% / -11.1% |
| 1,000 | +10.9% | -0.6% | -44.0% | -16.2% / -11.5% |

requested D32가 실제 D32를 의미하지는 않았다. observed maximum은 D16~D20이다. stable slot 32개
안에서 prefill과 decode request가 공존하고 prompt/output 길이가 서로 달라 모든 slot이 같은 순간
decode-ready가 아니기 때문이다. 그래도 D8 cap을 제거하는 것만으로 명확한 이득이 발생했다.

## 종합 선택

`P8/D32 independent`는 세 부하 모두 throughput과 E2E가 가장 좋다. 낮은 부하에서 D32가 TTFT
median을 D8 대비 3.7% 늘렸지만 절대 차이는 1.73ms이고, TPOT/E2E 이득이 훨씬 크다. 이 48-request
short-output workload에서는 P8/D32를 기본 cap으로 두고 다음 runtime policy를 적용할 수 있다.

- queue가 얕아도 independent context를 유지해 새 prefill의 TTFT를 보호한다.
- decode-ready가 8을 넘으면 D32 cap 안에서 가능한 row를 즉시 묶는다.
- D32를 채우기 위한 추가 wait는 두지 않는다. observed D16~D20만으로 이미 이득이 확인됐다.
- 더 긴 output과 288-request saturation에서는 기존 결과대로 D48~D56 승격을 별도 적용한다.

## 재현 자료

```text
.local/cosmos-reason2-2b/independent-load-sweep-p8d8-p8d32/
  r30-repeat1..3/
  r200-repeat1..3/
  r1000-repeat1..3/
  repeat-summary.csv
  clean-upstream-summary.csv
.local/cosmos-reason2-2b/clean-upstream-short48/
  cost-repeat1..3/
  r30-repeat1..3/
  r200-repeat1..3/
  r1000-repeat1..3/
```
