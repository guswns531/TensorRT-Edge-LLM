<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 355. 희소 P covering reference의 신뢰도 경계 실험

## 가설과 opt-in 변경

[354](354-prefill-service-age-seed-switch-20260927.md)에서는 두 표본뿐인 느린
P1/37 관측이 P1/33의 service reference를 키워 wavefront seed를 바꿀 수 있음을
확인했다. 이번 실험은 그 경로가 실제 balanced serving에서 유리한지를 분리한다.

`--trusted-prefill-service-covering`은 P urgency reference에만 적용하는 진단
ablation이다. 기존 `estimateCoveringPrimary()`는 그대로 두고, opt-in 경로에서는
cost tracker의 `actionMinimumSamples`를 충족하는 개별 관측 shape만 먼저 남긴 후
covering geometry를 계산한다. 희소한 가까운 shape가 제외되면 충분히 관측된 더 큰
shape를 사용할 수 있다. 동일한 표본 수 gate를 최종 aggregate에만 적용하면 희소
shape가 trusted shape를 가린 채 전체 estimate가 사라지므로, 필터를 covering 이전에
적용했다. 기본 정책, decode/overlap cost, engine, KV와 candidate frontier는 바뀌지
않는다. C++ 통제 테스트는 P1/37 표본 2개에서는 seed flip이 사라지고 4개에서는
다시 나타남을 확인한다.

## Paired 실행 계약

Gemma 4 E2B AWQ, balanced 64 text requests/6035 prompt tokens/5440 output tokens,
RTX 3080 10 GiB, TensorRT 26.06에서 baseline과 opt-in을 각각 두 독립 process로
실행했다. 두 캠페인의 바이너리 SHA256은 동일한
`0c19b751aa3d19da06340a15b57ff60f7c37e8924295fffea3ab935920b66d9d`이다.
각 캠페인은 generic startup calibration, independent E/P/D, client max-in-flight
64, ordered backend ingress, full telemetry, P formation diagnostic을 사용했다.
유일한 환경 차이는 opt-in의
`TRT_EDGELLM_TRUSTED_PREFILL_SERVICE_COVERING=1`이다. 양쪽 모든 run에서 backend
`server_submit` 순서는 0→63으로 검증했다. Raw logs, source dirty patch, engine·
trace hash와 실행 명령은 각 manifest에 보존한다.

- `.local/results/prefill-trusted-20260927-baseline/manifest.json`
- `.local/results/prefill-trusted-20260927-filtered/manifest.json`
- `.local/results/prefill-trusted-20260927-filtered/pair-{001,002}.json`

## Formation과 request 결과

첫 P6/P5 divergence가 이번에는 재현되지 않았다. 네 run 모두 처음 두 P batch가
`[0]`, `[1,2,12,13,14]` P5였고, 다음 `[3,15]` P2까지 같았다. Opt-in은 early
reference를 실제로 바꿨다. 첫 P1/P3 동시-ready decision에서 baseline의 reference는
run별 req1/req3 `17.14/17.91`, `16.69/18.27 ms`였고, opt-in은
`17.89/17.89`, `17.98/17.98 ms`였다. 하지만 두 경로 모두 req1을 seed로
선택했다. P membership의 첫 차이는 run 1의 9번째 P dispatch, run 2의 18번째
P dispatch에서야 나타났다. 해당 위치의 ready P IDs는 양쪽 동일했지만 전체
scheduler snapshot parity를 주장하지 않는다.

아래 값은 독립 run 두 개의 산술평균이다. 요청 측 latency는 HTTP send 기준이며
mean/p95 단위는 ms다. 모든 run은 64/64 요청과 5440 output tokens를 완료했다.

| 경로 | tokens/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | P dispatch |
|---|---:|---:|---:|---:|---|
| Baseline | 1243.62 | 1198.40 / 2746.22 | 15.65 / 16.99 | 2501.34 / 4033.69 | 41/38 |
| Trusted P cover | 1237.87 | 1199.16 / 2732.49 | 15.64 / 17.22 | 2499.56 / 4026.41 | 37/38 |

Opt-in 처리량은 평균 0.46% 낮다. TTFT/E2E tail은 소폭 낮고 TPOT p95는 높아,
두 반복의 변동 범위에서 명확한 승리라고 볼 수 없다. 두 경로의 peak GPU memory는
모든 run에서 9393 MiB였다. Request별 greedy output exact agreement는 같은 번호의
run에서 각각 61/64, 61/64였다. 이전부터 남아 있는 출력 동일성 gate는 이번에도
통과하지 않았다.

## 판단

희소 reference를 제외하는 메커니즘은 동작하지만, 자연 trace의 이번 sample에서는
초기 seed가 바뀌지 않았고 E2E 성능 이득도 검증되지 않았다. 따라서 이 설정은
**기본값으로 승격하지 않는다.** [354](354-prefill-service-age-seed-switch-20260927.md)의
P6/P5 차이를 항상 희소 reference 탓으로 돌리거나 P6가 P5보다 좋다고 결론 내리지
않는다. 같은 workload/engine 계약이므로 frozen vLLM 결과를 재사용하며, 새 vLLM
실행이나 외부 성능 우위 주장은 이 진단에 포함하지 않는다.

다음에 이 정책을 다시 평가하려면, 먼저 동일한 완전한 pre-branch snapshot에서
P5/P6의 forced-branch trajectory를 비교하거나 실제 서비스에서 충분한 빈도로
seed action change가 발생한다는 증거가 필요하다. 그 전에는 P reference 신뢰도
규칙을 더 튜닝하기보다 출력 동일성의 원인을 별도 correctness gate로 추적한다.
