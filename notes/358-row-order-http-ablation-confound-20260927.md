<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 358. HTTP row-order ablation의 scheduler-state confound

## 실험 목적과 구성

[357](357-decode-row-order-ablation-20260927.md)의 HTTP 비교는 두 arm에서 실제
decode cohort와 admission 시점이 달라졌다. 이 때문에 측정된 throughput 차이와
token output 차이를 row-order 단독 효과로 해석할 수 없다. 다음 단계에서 필요한
조건을 정리하기 위해 두 번의 paired run과 decision/dispatch telemetry를 다시
대조했다.

동일 binary
`c936d9c170c1f6d6d1c0f75d7443338dc16beb278e7e6dfd0c87deff30a78bb5`, engine,
trace SHA256 `fb5bfd84ad542221a0975ef4aeec3dae6fe4f04bcc1ce527a52100b7a0dc012a`,
generic startup service calibration, independent E/P/D, 24 stable slots, 64 in-flight
HTTP client 및 ordered backend submit을 사용했다. 설정 차이는 decode row-order
mode 하나다. Affinity mode는 가능한 한 기존 request-to-row 위치를 보존하고,
canonical mode는 각 D dispatch 전 context bucket/request ID로 정렬한다.

## Admission 및 batch membership이 실제로 달라짐

Repeat 1에서는 처음 세 P cohort가 양쪽에서 같았고 P dispatch 수도 모두 40이었다.
그러나 D dispatch 수는 affinity 294회, canonical 308회였으며 P/D 동시 완료와
sampling 시점 차이가 후속 ready state를 바꿨다. 해당 run의 64 output 중
affinity/canonical은 62개가 exact였다.

Repeat 2에서는 P row order가 dispatch index 17에서 처음 달라졌고 P membership은
index 23에서 달라졌다. 더 이른 두 번째 D dispatch부터 affinity는 8 rows
`[0,1,2,3,12,13,14,15]`, canonical은 4 rows `[0,1,2,12]`를 선택했다. 이
차이는 canonical worker 정렬이 첫 multi-row D action에서 실행되기 전의 ready
membership 변화다. 그 뒤 request 1/13은 양쪽 모두 token index 18에서 갈렸고,
decode cohort membership 이력은 이미 달랐다. 따라서 이번 HTTP A/B는 동일
pre-decode state에서 row order만 바꾼 실험이 아니다.

## 성능과 기존 vLLM anchor

두 독립 run 평균은 다음과 같다.

| Mode | Current generated tok/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|
| Retain affinity | 1236.67 | 1190.79 / 2739.70 | 15.60 / 16.87 | 2490.32 / 4019.98 |
| Canonical every D | 1232.06 | 1198.99 / 2737.13 | 15.65 / 17.14 | 2500.87 / 4037.29 |

Canonical은 처리량 평균이 0.37% 낮다. 지연 지표도 혼합되어 있어 이 표본으로
일반적인 성능 우열을 말하지 않는다. 두 모드 모두 peak GPU memory 9393 MiB,
64/64 request 완료, 5440 generated tokens를 기록했다.

같은 trace SHA256의 frozen vLLM capacity run은 generated throughput 771.53 tok/s,
5440/5440 output tokens였다. 이 한 balanced workload anchor에 대해 현재 affinity
mode는 약 60.3% 높고 canonical은 약 59.8% 높다. 이 비교는 두 row-order mode의
선택 근거가 아니며, 12-workload 전체 승리를 의미하지 않는다. vLLM은 runtime 또는
trace 계약이 바뀌지 않아 다시 실행하지 않았다.

## 연구 판단

이 결과는 CPU admission/gateway timing과 asynchronous ready transition이 HTTP
paired trace에서 snapshot parity를 깨뜨릴 수 있음을 재확인했다. Canonical row
ordering은 serving default로 올리지 않고 opt-in 진단으로 둔다. 다음 인과 실험은
HTTP admission에 의존하지 않는다. 고정 prompt cohort를 한 번 prefill하고, page
aligned KV prefix를 stable slot 간 공유한 뒤, 같은 pre-decode page ownership·token
prefix·request membership으로 동일 D batch를 각각 실행한다. 각 branch가 decode로
KV를 쓰는 suffix는 분리한다. Row permutation 하나만 달리해 logits, selected token,
CUDA event duration을 비교한다. 그 fixture가 완성되기 전에는 row order를 성능
정책으로 채택하지 않는다.

Raw cells and commands:

- `.local/results/decode-row-order-20260927-affinity/`
- `.local/results/decode-row-order-20260927-canonical/`
