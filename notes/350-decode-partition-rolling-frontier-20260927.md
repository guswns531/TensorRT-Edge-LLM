<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 350. Decode DP partition과 실제 rolling frontier의 차이

## 검토 대상

[349](349-decode-service-interval-guard-20260927.md)에서 interval guard를 켜자 D1×3
회귀는 사라졌지만 Gemma balanced의 throughput은 동일 binary static보다 1.55% 낮았다.
On run의 예측 `[4,3]`/`[4,4]` partition은 실제 두 dispatch와 일치하지 않았다.
이 불일치를 곧바로 성능 버그로 간주해 residual frontier를 lease할 것인지 확인했다.

같은 retained HTTP trace를 재실행하지 않고
`analyze_decode_partition_realization.py`에 다음 진단을 추가했다.

- Frontier를 모두 한 token 전진시키기 전 실제 dispatch한 row 수와 중복 row 수.
- 첫 frontier drain 시점까지 추가로 commit된 token 수.
- 직접 관측된 dense/후보 비용이 있는 decision에서 `service cost / batch rows`를
  최소화하는 선택과 기존 DP의 첫 batch 선택이 같은지.

후보의 `service_samples`와 `gpu_samples`가 있고 dense cost가 알려진 결정만 두 번째
비교에 포함했다. Unknown dense를 실행해 evidence를 모으는 bootstrap은 비교 대상에서
제외했다. 분석기 신규 단위 테스트 2개와 기존 3개가 통과했다.

## 같은 binary trace 재분석

| 경로/run | 비용 coverage가 있는 결정 | DP 첫 batch = service-density 선택 | 비자명한 split | 계획 partition 그대로 실현 | Extra token이 frontier drain까지 commit된 split |
|---|---:|---:|---:|---:|---:|
| Guard on 1 | 218 | 218 | 15 | 0 | 15 |
| Guard on 2 | 222 | 222 | 18 | 0 | 18 |
| Guard off 1 | 236 | 236 | 38 | 38 | 0 |
| Guard off 2 | 236 | 236 | 37 | 37 | 0 |

Guard off의 D1 반복 partition은 그대로 실현됐지만 느렸다. Guard on의 `[4,3]`은
대개 실제 `[4,4]`가 됐다. On 1의 15개 `[4,3]` snapshot 중 14개는 `[4,4]`,
1개는 `[4,6]`이었다. On 2의 16개 `[4,3]` 중 15개는 `[4,4]`, 1개는 `[4,6]`이었다.
각 `[4,4]` 실행은 원래 frontier 7개 모두의 첫 token을 commit할 때까지 8개
token을 처리했다. 이는 같은 trace의 연속된 snapshot이므로 각 행을 독립 실험으로
세면 안 된다.

예를 들어 on 1의 dispatch 1589는 ready IDs `[52,58,57,55,59,46,63]`에서
`[4,3]`을 예측해 `[52,58,57,55]`를 먼저 실행했다. Sampling 후 다음 ready
frontier가 `[59,46,63,52,58,57,55]`가 됐고 dispatch 1590은
`[52,63,59,46]`을 실행했다. 원래 7개 모두 첫 token을 받기 전에 request 52가
두 번째 token을 받은 것이다. 이 시점의 후보 robust service는 D4 7.102 ms,
D3 7.039 ms, D7 14.143 ms였고 `[4,3]` 계획 총합은 14.141 ms다.
실제 7개 첫 token과 request 52의 추가 token까지 commit된 horizon은 13.521 ms였다.
계획값은 p95/uncertainty를 포함하고 실측값은 단일 trajectory이므로 두 숫자의
차이를 비용 예측 오차로 단정하지 않는다. 그러나 **partition 불일치 자체가 유용한
작업의 손실이라는 가정은 반증된다**.

On 2의 나머지 2개 `[4,4]` snapshot은 실제 `[4,16]`/`[4,7]`이었고 새로 ready된
row가 끼었다. `[4,16]`의 frontier drain은 약 105 ms로 길었다. 이는 모든 mismatch가
이득이라는 주장도 막는다. 새 arrival 및 P completion을 포함한 request transition을
보지 않고 단순히 planned-vs-actual batch 수만으로 좋고 나쁨을 정할 수 없다.

## 정책 변경에 대한 결론

현재 covered decision 218/222/236/236건에서 단순 `service cost / rows` 최소 선택은
기존 equal-work DP와 첫 batch를 모두 동일하게 골랐다. 따라서 이 trace에서 DP를
service-density 식으로 바꾸는 것만으로는 action이 바뀌지 않는다. 반대로 residual
frontier를 강제로 lease하면 실제 rolling D4가 만든 추가 token을 막고, 새로 ready된
작업을 지연시킬 수 있다. 이 결과만으로 lease가 유리하다고 판단할 수 없다.

현재 DP의 `predictedDecodePartition`은 **counterfactual equal-work 비용 설명**이지
다음 dispatch까지 예약한 실행 계약이 아니다. Telemetry와 문서에서는 이 둘을
분리해 해석해야 한다. 정책 변경 전에는 동일 state의 forced D4→D3와 D4→D4를
같은 token 총량 및 다음 ready boundary까지 비교해야 한다.

## 남은 static 처리량 격차의 더 직접적인 신호

349의 동일 binary guard-on 두 run과 static 두 run은 모두 prefill useful tokens
6035, decode row-token work 5376으로 같다.

| 경로/run | P dispatch | P 단독 dispatch 수 | P dispatch duration 합 | D dispatch | D dispatch duration 합 |
|---|---:|---:|---:|---:|---:|
| Guard on 1 | 44 | 30 | 892.71 ms | 303 | 3737.91 ms |
| Guard on 2 | 46 | 32 | 926.61 ms | 305 | 3716.02 ms |
| Static 1 | 39 | 23 | 815.23 ms | 294 | 3779.15 ms |
| Static 2 | 39 | 23 | 819.10 ms | 294 | 3790.93 ms |

Guard-on의 D duration 합은 오히려 static보다 작지만, P dispatch가 5~7회 많고
P duration 합이 77~108 ms 크다. 이것이 처리량 격차와 함께 나타난 더 강한
mechanism 후보이다. 다만 P formation을 변화시킨 원인이 D action timing인지,
startup calibration posterior인지, host submission 차이인지는 이 trace 집계만으로
분리되지 않는다. 별도 snapshot 또는 paired trace 분석이 필요하다.

Prefill row appearance는 두 경로의 매 run에서 정확히 84회다. 각 요청의 P 실행 횟수도
44개 요청 × 1회, 20개 요청 × 2회로 동일하다. 추가 P dispatch는 추가 prompt work가
아니라 같은 row work를 더 작은 batch로 나눈 것이다. 첫 P membership 차이는
request 28 근처에서 나타났다. Static은 `[28,29]`를 함께 실행했고 guard-on은
`[28]` 다음 `[29,30]`이었다. 즉 같은 요청 집합의 cohort 경계가 바뀌었다.

## 다음 단계

1. 같은 ready snapshot에서 P candidate membership/packed-token utilization과
   선택·submission 시각을 static과 service로 대조해 추가 P dispatch가 어디서
   발생하는지 찾는다.
2. Equal-work frontier lease는 지금 적용하지 않는다. 필요하면 D4→D3와
   D4→D4의 equal-total-work forced branch 실험을 먼저 한다.
3. Batch-shape에 따른 Gemma exact token divergence가 남았으므로 throughput
   improvement만으로 promotion하지 않는다.

새 GPU benchmark나 vLLM 실험은 하지 않았다. 분석 출력은 각 retained
`.local/results/decode-service-guard-ablation-20260927-{on,off}/`
`realization-work-aware-repeat-00{1,2}.json`에 있다. Production scheduler,
engine, `.local/current/*`는 변경하지 않았다.
