<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Controlled E 및 overlap startup calibration

## 목적

기존 startup warmup은 text P/D shape만 실행했다. 따라서 동일 engine에서 portable cost bundle을 읽더라도
E와 E+D는 node-local CUDA 표본이 없었고, portable overlap 표본 수만으로 현재 노드의 concurrency를 신뢰할
수 있었다. 이번 단계는 workload profile이나 외부 cost registry를 추가하지 않고 다음 두 조건을 분리한다.

```text
portable prior       = 첫 예측과 uncertainty에 사용
exact node coverage  = 현재 process의 controlled warmup이 직접 확인
```

portable prior가 이미 `eligible`이어도 local 표본이 minimum sample 수보다 작으면 calibration probe를 계속한다.
production traffic에서는 기존 slack multiplier와 probe interval을 그대로 사용하므로 startup 전용 권한이 사용자
요청의 overlap exploration으로 확대되지 않는다.

## 구현

### Cost oracle

`PhaseCostOracle`에 `localSampleCount()`와 `localOverlapDiagnostic()`을 추가했다. layered diagnostic은 local,
fleet/build prior를 합친 scheduler 예측용이고, local diagnostic은 현재 노드에서 overlap을 검증했는지 판단하는
calibration coverage용이다.

### Candidate와 action fidelity

`PhaseGlobalActionCandidate::calibrationProbe`를 일반 `safeProbeEligible`과 분리했다. warmup selector는 portable
cost의 유무가 아니라 local sample 수가 가장 적은 E+D/P+D candidate를 고른다. 선택 결과는 기존
`PhaseGlobalDispatchPlan`과 outstanding execution lease를 그대로 통과하므로, calibration 때문에 별도 enqueue
경로나 암묵적 overlap을 만들지 않는다.

### Encoder shape 생성

기본 E shape는 power-of-two와 exact maximum이다.

```text
max E8 -> E1, E2, E4, E8
max E6 -> E1, E2, E4, E6
```

하지만 실제 가능한 최대값은 설정의 `maxEncoderBatchSize`, encoded in-flight capacity, calibration image 한
건의 measured input tokens와 visual engine total-input-token limit의 최솟값이다. 큰 이미지가 E2까지만 맞으면
E4/E8을 잘못 요청하지 않는다. 명시적인 batch list도 이 물리 한계를 넘으면 즉시 실패한다.

### Production state 격리

IPC composition root는 production coordinator를 만들기 전에 임시 `PhaseThreeCoordinator`에서 synthetic VLM
요청을 실행한다.

1. 각 E shape의 부족한 local sample만 isolated 실행한다.
2. decode cohort를 만든 뒤 같은 E shape를 E+D로 실행한다.
3. 모든 completion/token과 prefix lease를 drain한다.
4. calibration epoch와 warmup selector를 끈다.
5. cost oracle은 보존하되 scheduler queue/latency telemetry를 reset한다.
6. 깨끗한 production three-phase coordinator를 생성한다.

따라서 실제 TensorRT context, stream, workspace 및 CUDA event 비용은 측정하지만 synthetic queue age나 request
state는 serving에 남지 않는다. build bundle snapshot도 이 단계 뒤에 기록되어 E/P/D 및 overlap record를 한
파일에 담는다.

## 제어 변수

```text
TRT_EDGELLM_PHASE_ENCODER_CALIBRATION_IMAGE
TRT_EDGELLM_PHASE_ENCODER_CALIBRATION_BATCHES
TRT_EDGELLM_PHASE_ENCODER_CALIBRATION_SAMPLES
TRT_EDGELLM_DISABLE_PHASE_ENCODER_DECODE_CALIBRATION
```

첫 변수만 opt-in이다. batch list를 생략하면 물리적으로 가능한 대표 shape를 자동 생성하고, sample 수를
생략하면 phase anchor와 overlap minimum 중 큰 값을 사용한다. E+D만 끄고 isolated E anchor만 만들 수도 있다.

## 실제 Cosmos Reason2-2B 검증

RTX 3080 10GB, Cosmos Reason2-2B FP16 tied LLM engine, independent P/D context와 실제 visual engine을 사용했다.
작은 calibration image는 visual context bucket 1이어서 E1/E2/E4/E8이 모두 실제로 형성됐다.

첫 process는 prior 없이 다음을 완료했다.

| 항목 | 결과 |
|---|---:|
| E shape | 1, 2, 4, 8 |
| isolated E 실행 | 8 |
| E+D 실행 | 8 |
| synthetic vision requests | 60 |
| OOM/profile 위반 | 0 |

생성 bundle에는 각 E shape 4개 관측과 E1+D1, E2+D2, E4+D4, E8+D8 각각 2개 관측이 기록됐다. E+D
makespan 범위는 약 9.46~10.33 ms였다. E1의 첫 isolated 실행은 약 14 ms cold-start였고 이후 isolated 실행은
약 7.3~8.0 ms였다. raw observation을 보존하므로 이 분산은 숨겨지지 않고 robust uncertainty에 반영된다.

두 번째 process는 첫 bundle을 compatible build prior로 먼저 로드했다. 그럼에도 local coverage를 별도로
판단해 동일한 E/E+D calibration을 다시 실행했고 다음 anchor를 얻었다.

```text
E scale       = 1.010, samples = 16
overlap scale = 1.000, samples = 16
```

즉 portable prior는 초기 예측에 사용됐지만 현재 노드에서 측정했다는 증거를 대신하지 않았다. 작은 이미지
검증 전에 사용한 실제 큰 VLM 이미지에서는 measured token bound가 E1/E2만 선택했고, 이 경우에도 profile
밖의 E4를 실행하지 않았다.

## 회귀 검증 범위

- portable overlap prior와 local coverage 분리 단위 테스트
- prior가 이미 eligible이어도 warmup이 P+D local probe를 선택하는 scheduler 테스트
- representative E shape 생성, 정렬, 중복 제거와 범위 오류 테스트
- 실제 Cosmos `llm_phase_context_smoke` E1/E2/E4/E8 및 E+D 실행

최종 phase runtime 관련 14개 suite는 `240/240` 통과했고 `unitTest` 및 `llm_phase_context_smoke` target도
TensorRT 11/CUDA 13.3 환경에서 재빌드했다.

이번 변경은 opt-in startup calibration과 cost knowledge만 바꾼다. production batch formation, action selector,
KV ownership 및 workload request trace는 바꾸지 않았으므로 동일한 vLLM workload를 다시 실행하지 않았다.
다음 성능 재측정은 static decode table을 generated bundle로 대체하는 promotion gate와 함께 수행한다.
