<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 353. Ordered backend ingress로 P formation을 다시 분리

## 문제와 구현

[352](352-server-ingress-admission-order-20260927.md)에서 client64는 같은 HTTP send
시각을 만들었지만 gateway handler thread가 backend stdin에 쓰는 순서가
static/service 사이에서 뒤집혔다. 따라서 client arrival을 맞추는 것만으로는
같은 backend ingress가 아니었다.

`run_ordered_phase_gateway.py`는 기존 retained HTTP/SSE gateway의 EventBroker를
진단 실행에서만 감싼다. HTTP handler가 어떤 순서로 들어와도 request index
0,1,2,… 순서로 backend stdin에 submit하고 응답은 기존 queue/비동기 SSE로 계속
받는다. Calibration `begin/status/end` ack 이후 next index를 0으로 되돌린다.
빠진 앞 요청을 무한정 기다리지 않고 request timeout에 실패시킨다. 기존 gateway,
server E/P/D, engine/KV, client JSON, scheduled arrival는 변경하지 않았다.
`run_lifetime_encoded_admission.py --ordered-backend-ingress`로만 이 adapter를
선택하며 기본 serving/benchmark 경로는 그대로다.

첫 버전은 adapter가 자체 로그를 별도 thread의 backend timeline 출력과 동시에
stdout에 써 `server_submit` 두 줄을 훼손했다. 요청 자체는 완료했지만
`.local/results/prefill-ordered-ingress-20260927-static/`는 **telemetry-invalid
diagnostic**으로 보존하고 성능·인과 결론에서 제외했다. Adapter 자체 출력을 제거한
v2 결과만 아래에 사용한다.

`analyze_prefill_arrival_coupling.py --require-ordered-server-submit`은 measurement
request 64개의 timeline ID와 client CSV ID 일치를 확인하고, `server_submit` 순서가
정확히 0→63이 아니면 실패한다. Ordered gate 단위 테스트 2개, runner 계약 49개,
paired timeline 테스트 3개 및 pre-commit이 통과했다.

## 재실행 계약

Gemma AWQ, RTX 3080 10 GiB, NVIDIA driver 610.57.04, TensorRT 26.06,
balanced 64 requests/6035 prompt tokens/5440 generated tokens다. Static과
measured-service 모두 동일 binary SHA256
`1372aa36d7e427ca614651846b37699f290a69780960a950a0e7a6eae40d4afb`와
ordered gateway SHA256
`997304d937ccf90062c09a6c3787f01b3342f16c079c734e1338b483a1710bfa`를
사용했다. Client max in-flight는 64, backend active slots는 기존 24다.
Static/service 각 3개 독립 process를 순차 실행했고, 여섯 run 모두 64/64 완료 및
measurement `server_submit` 순서 0→63을 확인했다.

재현 runner 추가 인자는 `--client-max-in-flight 64 --ordered-backend-ingress
--telemetry-level full --decode-partition-diagnostic --startup-budget-ms 120000`이다.
각 manifest가 command, source patch, binary/engine/trace/gateway hash를 보존한다.

- `.local/results/prefill-ordered-ingress-v2-20260927-{static,service}/`
- `.local/results/prefill-ordered-ingress-v2-repeat-20260927-{static,service}/`

## 결과와 P divergence

아래 지표는 3 run 평균, latency는 HTTP send 기준 ms mean/p95다. 이 낮은 반복 수로
통계적 우위나 production promotion을 주장하지 않는다.

| 경로 | tokens/s | TTFT | TPOT | E2E | P dispatch(run별) |
|---|---:|---:|---:|---:|---|
| Static | 1219.50 | 1210.97 / 2779.15 | 15.88 / 17.76 | 2531.97 / 4092.24 | 40/43/43 |
| Service | 1221.78 | 1205.81 / 2766.32 | 15.76 / 17.18 | 2517.83 / 4048.83 | 40/37/42 |

처리량 차이는 service +0.19%로 사실상 parity다. 세 paired run의 client send
차이 p95는 0.053/0.083/0.061 ms이고 backend submit 순서 차이는 0이다.
그래도 첫 P membership 차이는 모든 run의 두 번째 P dispatch에서 발생했다.

| Run | Static 첫 차이 | Service 첫 차이 | 같은 P-ready ID/token count? |
|---|---|---|---|
| 1 | `[1,2,3,12,13,14]` P6 | `[1,2,12,13,14]` P5 | 예, `[1..14]` |
| 2 | `[1,2,3,12,13,14,15]` P7 | `[1,2,12,13,14]` P5 | 아니오, 1개 차이 |
| 3 | `[1,2,3,12,13,14,15]` P7 | `[1,2,12,13,14]` P5 | 아니오, 1개 차이 |

첫 run의 두 decision은 모두 serial P, outstanding mask 0이다. 공통 ready item의
남은 token count도 동일했다. 그러나 static은 chunk38/P6을, service는
chunk33/P5를 선택했다. Decision telemetry의 protected P request도 각각
3과 1로 달랐다. 같은 backend submit 순서와 동일한 ready 목록만으로 같은
선택이 보장되지 않는다는 직접 증거다. Queue age, active wavefront cohort,
P cost posterior 등 **전체 scheduler state는 아직 같다고 증명하지 않았다**.
따라서 P5 선택을 버그로 단정하거나 P6를 강제하지 않는다. 실제 전체 P dispatch
수도 service가 세 run 평균 39.7회로 static 42.0회보다 적었다.

Static/service exact greedy output은 run별 64/64, 58/64, 60/64 requests가
일치했다. 첫 run만 exact였으므로 quality gate는 계속 미통과다.
이번 ordered-ingress 계약의 vLLM 결과도 없으므로 외부 우위 주장은 하지 않는다.

## 다음 좁은 작업

첫 P decision에서 wavefront cohort IDs, seed 후보, 후보 chunk/batch별 useful
tokens·실측/covering GPU cost·선택 score, per-request queue age를 **진단 모드에만**
기록한다. 그다음 P6/P5 차이가 input state, cost posterior, candidate formation,
selector 중 어느 층에서 생겼는지 구분한다. 동일한 완전한 snapshot의 forced branch가
가능하기 전에는 모델별 P batch 강제 규칙이나 고정 대기시간을 추가하지 않는다.
