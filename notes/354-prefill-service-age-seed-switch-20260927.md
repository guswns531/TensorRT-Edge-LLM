<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 354. P6/P5 seed switch의 원인: 희소 covering cost와 service-age ranking

## 질문과 진단 계약

[353](353-deterministic-backend-ingress-20260927.md)은 static/service 양쪽의 backend
`server_submit` 순서를 0→63으로 고정해도 초기 P6/7 대 P5 차이가 재현됨을 보였다.
이번에는 Global 후보가 참조하는 `previewMechanismPlan()`에 opt-in P formation 진단을
붙였다. `--prefill-formation-diagnostic`을 켠 full scheduler decision event에만
다음을 기록한다.

- Wavefront seed request ID와 padded chunk, 기존 active cohort ID.
- Queue 순서의 ready request IDs/남은 tokens/queue wait.
- 각 row의 P service reference와 정규화된 service age.
- Compatible row IDs, 검토한 batch/chunk의 useful tokens·GPU cost·feasibility,
  최종 선택 chunk/batch.

기본 scheduler 정책·engine·KV·`.local/current`는 변경하지 않았다. Global preview가
만든 diagnostic을 audit→unified decision event로 전달한다. 관련 C++ 테스트 3개와
Python runner 계약 테스트 50개가 통과했고 TensorRT Docker에서 실행 파일·unit test를
빌드했다.

Primary GPU 데이터는 ordered backend ingress, Gemma AWQ balanced 64 requests,
RTX 3080 10 GiB, TensorRT 26.06, 같은 binary SHA256
`74a618be6019118b90969fcd209f74276a3be3753691ec841ca250d0288ac236`의
static/service 각 2개 독립 process다. 각 manifest에 command/engine/trace/source
patch와 raw/full telemetry를 보존한다.

- `.local/results/prefill-formation-age-20260927-static/manifest.json`
- `.local/results/prefill-formation-age-20260927-service/manifest.json`

첫 구현(`prefill-formation-cost-20260927-*`)은 queue-age만 기록했고 첫 P 차이가
후반으로 이동했다. 그 결과만으로 원인을 확정하지 않고 service-reference/age를
추가해 새 binary로 다시 측정했다. 계측 자체가 host timing을 바꿀 수 있으므로
두 binary의 결과를 섞어 성능 효과로 주장하지 않는다.

## 같은 ready work에서도 seed가 바뀜

두 primary run 모두 첫 P membership 차이는 두 번째 P dispatch다. Static은
`[1,2,3,12,13,14]` P6, service는 `[1,2,12,13,14]` P5다. Run 2에서는
P-ready IDs와 남은 token counts가 양쪽 모두 정확히 같았다. Run 1에서는 service
ready set에 request 15가 추가되어 완전한 ready snapshot parity는 없었다.
두 경로 모두 serial P action이고 active wavefront cohort는 비어 있었다.

| Run 2, request | 경로 | Queue wait | P reference | Service age = wait/reference |
|---|---|---:|---:|---:|
| 1 (33 tokens) | Static | 5.379 ms | 41.644 ms | 0.1284 |
| 3 (38 tokens) | Static | 5.115 ms | 18.451 ms | 0.2766 |
| 1 (33 tokens) | Service | 4.975 ms | 16.826 ms | 0.2941 |
| 3 (38 tokens) | Service | 4.747 ms | 17.999 ms | 0.2631 |

Request 1은 양쪽 모두 request 3보다 약간 오래 기다렸다. 하지만 static에서는
request 1의 큰 reference 때문에 정규화된 age가 작아져 seed 3을 고른다.
Chunk38에는 길이 38의 request 3/15와 짧은 rows가 함께 들어가 P6이 된다.
Service에서는 seed 1의 chunk33이 선택되어 길이38의 rows가 빠지고 P5가 된다.

이 짧은 text P에서는 `shape_candidates`가 빈 배열이었다. 현재 코드의 생산적인
adaptive chunk 후보 조건을 만족하지 않아 batch별 GPU 효율 루프가 실행되지
않는다. 따라서 이번 P6/P5 차이를 "GPU 비용 점수가 직접 P5를 선택했다"고
설명하면 틀리다. **GPU 비용은 service reference를 통해 urgency seed에 간접적으로
작용했고, seed/chunk compatibility가 batch membership을 바꿨다.**

## Reference가 달라진 구체 증거

Static 두 run의 calibration 시점 P1/37 eager 단독 실행은 각각
`15.008/40.220 ms`, `14.786/41.639 ms`의 두 표본을 만들었다. Service에서는
첫 P1/37 eager 단독 표본이 `14.279/14.824 ms`였고, 다른 P1/37은
P+D overlap으로 `15.911/15.858 ms`였다. Overlap은 별도 action key다.

`makePrefillServiceReference()`는 먼저 minimum sample 수를 만족하는 measured
P cost를 찾는다. 없으면 service-scaled 경로에서 `estimateCoveringPrimary()`를
사용한다. 이 fallback은 `actionMinimumSamples=4` trust gate를 요구하지 않는다.
따라서 P1/33의 reference가 static에서 두 표본뿐인 P1/37의 높은 robust cost
약 40~42 ms의 영향을 받을 수 있다. Service의 P1/33 reference 약 16~17 ms는
동일한 느린 eager P1/37 표본 경로가 없었다. 정확히 어느 covering key가 선택됐는지
이번 telemetry는 기록하지 않았으므로 그 key까지 단정하지 않는다.

통제 C++ 테스트는 같은 ready P1/33, P3/38에서 P1/128의 안정적인 네 표본만 있을
때와, 희소한 P1/37의 `15/42 ms` 두 표본을 추가했을 때를 비교했다.
Seed가 request 1→3으로 뒤집히는 것을 확인했다. 즉 희소 covering-cost observation이
실제 P batch formation에 영향을 줄 수 있다는 메커니즘은 재현됐다.

## 성능과 판단

이번 diagnostic 두 run 평균 tokens/s는 static 1227.75, service 1221.61로
service가 약 0.50% 낮다. Service의 TTFT/E2E mean과 p95는 소폭 낮았다.
Exact output은 run별 60/64, 64/64 requests 일치해 quality gate는 불안정하다.
이 결과만으로 P6를 강제하거나 P5가 잘못됐다고 말할 수 없다. 이전 353의
진단 off 세 run에서는 service throughput이 사실상 static과 동률이었다.

다음 후보는 **P service reference의 신뢰도 경계**를 분리하는 것이다. 예를 들어
trusted direct/covering estimate가 없을 때 희소 worst-case sample을 urgency
denominator에 곧바로 넣지 않고, 불확실성이 명시된 generic reference를 유지하는
opt-in shadow ablation을 먼저 한다. 같은 ordered ingress에서 seed agreement,
TTFT/TPOT/E2E, output identity를 확인한 뒤에만 active/default 변경을 검토한다.
이는 모델별 P batch 규칙이나 고정 wait를 새로 넣는 방향이 아니다. 새 vLLM/full12
비교와 production 승격은 이번 진단에 포함하지 않는다.
