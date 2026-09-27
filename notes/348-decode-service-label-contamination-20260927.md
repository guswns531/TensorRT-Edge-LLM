<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 348. Decode service 비용 표본의 후속 phase 혼입

## 질문

[347](347-single-turn-decode-branch-20260927.md)에서 약 246-token context의 동일 작업량을
강제로 비교했을 때 D3 한 번은 7.00 ms, D1×3은 18.57 ms였다. 그런데 HTTP measured-service
경로는 tail에 D1×3을 선택하고 static보다 decode dispatch 수를 늘렸다. 이번에는 policy를
바꾸지 않고 **선택 순간의 후보별 실제 비용·표본·graph key**를 기록해 이유를 확인했다.

## 계측 계약

`captureDecodePartitionTrace`가 켜진 진단 실행에서만 각 ready frontier의 batch 1..N에 대해
다음 값을 dispatch metric에 복사한다.

- context bucket 및 실행 variant(graph/eager)
- host-service sample count, median, p95, uncertainty, DP가 사용한 robust cost
- CUDA GPU sample count, robust cost, covering estimate 사용 여부

이는 `selectDecodeBatchSize()`의 선택 값을 변경하지 않는다. Default serving에서는 벡터가
비어 있고 추가 lookup을 하지 않는다. `PhaseDispatchWorker`는 plan의 진단값을 metrics로
전달하고 full HTTP telemetry만 JSON에 직렬화한다. C++ 비용/partition 단위 테스트 2개가
통과했고, TensorRT Docker에서 `llm_phase_context_smoke`와 `unitTestRuntime`을 재빌드했다.

모든 새 HTTP 결과는 동일 source+patch, binary SHA256
`00a6118ad63a94360a753dcb148a27a141adced0d4b47d1c7eb2a1028d5cba4f`,
동일 Gemma engine, 64 요청/5440 출력 tokens, RTX 3080/driver 610.57.04,
TensorRT 26.06 container에서 나왔다. 명령, source patch, 전체 hash와 trace는 각
`.local/results/decode-cost-*-20260927/manifest.json`에 저장했다. 결과 상태는 diagnostic이며
각 run은 독립 process지만 반복 수가 작다.

## 동일 binary balanced HTTP 결과

TTFT/TPOT/E2E는 HTTP send 기준 ms의 mean/p95다. 각 행은 단일 run이므로 아래 차이를
통계적 우열 또는 production promotion으로 해석하지 않는다.

| 경로/run | tokens/s | TTFT | TPOT | E2E | D dispatch | D1 dispatch |
|---|---:|---:|---:|---:|---:|---:|
| Static shadow 1 | 1232.57 | 82.36 / 212.01 | 15.59 / 17.06 | 1381.55 / 2165.15 | 294 | — |
| Static shadow 2 | 1215.92 | 86.41 / 212.79 | 15.78 / 17.49 | 1400.50 / 2179.83 | 295 | — |
| Service, D1 outlier run | 1215.54 | 81.13 / 205.05 | 15.64 / 17.17 | 1382.86 / 2134.68 | 310 | 20 |
| Service, D3 outlier run 1 | 1164.59 | 79.27 / 207.45 | 15.85 / 17.26 | 1402.32 / 2171.20 | 318 | 56 |
| Service, D3 outlier run 2 | 1159.24 | 80.24 / 210.11 | 15.96 / 17.35 | 1411.73 / 2188.08 | 319 | 57 |

Paired repeat 1/2의 service tokens/s는 static보다 각각 -5.52%/-4.66%였다.
두 run 모두 exact output identity는 62/64 requests, 5355/5440 token positions다.
따라서 처리량 개선 여부와 별개로 output-quality gate는 계속 미통과다.
이번 진단은 새 vLLM 실행 또는 full12를 포함하지 않는다.

## 선택이 달라진 실제 비용

| 상태 | Batch | Service median | Service p95 | Uncertainty | DP 사용 비용 | 표본 |
|---|---:|---:|---:|---:|---:|---:|
| D3 outlier run 1, tail | D1 | 6.30 | 6.46 | 0.35 | 6.65 | 32 |
| D3 outlier run 1, tail | D3 | 6.57 | 27.63 | 21.06 | 27.63 | 31 |
| D3 outlier run 2, tail | D1 | 6.31 | 6.43 | 0.35 | 6.67 | 32 |
| D3 outlier run 2, tail | D3 | 6.58 | 27.76 | 21.18 | 27.76 | 31 |
| D1 outlier run, 첫 measurement | D1 | 6.30 | 29.24 | 22.94 | 29.24 | 32 |

단위는 ms다. 두 D3 outlier run에서 같은 context bucket 1의 D1×3 예상 비용은
약 19.95–20.00 ms로, D3 robust cost 27.63–27.76 ms보다 낮았다. 실제 DP는 각각
`[1,1,1]`을 28/33개 snapshot에서 택했다. 그 결과 D1 dispatch가 56/57회였고
tokens/s가 1165/1159였다. 반대로 D1 outlier run에서는 D1 cost가 29.24 ms로
급등해 `[4,4]` 4건 및 `[4,3]` 27건을 택했고 tokens/s는 1215.54였다.

`selectionServiceMs = max(p95, median + uncertainty)`이다. D3 median 자체는 통제
실험의 D3 시간과 비슷했다. **높은 p95가 dense batch를 탈락시킨 직접적인 수치 원인**이다.
첫 measurement에서 보였던 D1 p95 29.24 ms는 startup probe의 D1 p95 6.23 ms와
다르며, 요청 흐름의 추가 관측에 의해 rolling window가 바뀐 값이다.

## 긴 표본의 시간 순서

`gateway.log[.gz]`의 `PHASE_METRIC` host dispatch 시작과 `PHASE_TIMELINE` token commit을
request ID로 연결했다. 아래 duration은 metric dispatch-start→마지막 token commit의
근사치다. 서비스 collector의 정확한 prepare-start→state-commit과 경계가 수십~수백 µs
다를 수 있다.

| Run/dispatch | D GPU | Commit까지 | 후속 P 또는 P+D 시작 | 당시 metric의 concurrent P |
|---|---:|---:|---:|---|
| D3 outlier 1 / D3 #1643 | 14.82 ms | 28.07 ms | D 시작 약 20.11 ms 뒤 | false |
| D3 outlier 1 / D3 #1649 | 14.94 ms | 27.77 ms | D 시작 약 20.29 ms 뒤 | false |
| D1 outlier / D1 #1672 | 12.66 ms | 29.25 ms | D 시작 약 20.85 ms 뒤 | false |
| D1 outlier / D1 #1682 | 12.65 ms | 29.24 ms | D 시작 약 20.89 ms 뒤 | false |

이 네 dispatch는 시작 당시 `concurrentPrefillActive=false`여서 host-service 수집 대상이다.
그러나 D 이후 sampling/token commit 전에 다음 P 또는 P+D가 시작했다. 현재 collector는
dispatch 시점에만 P/E overlap을 거르고, sample 끝 시점의 후속 phase 시작을 확인하지
않는다. 따라서 **"isolated decode service" key에 후속 phase가 섞인 label이 들어갈 수
있다**. 긴 값의 모든 원인을 후속 P라고 단정하지는 않는다. D GPU 자체도 평소 약 6.4 ms보다
길었으며 GPU clock, dispatch 실현, tactic 등 다른 요인이 남아 있다. 다만 sampling까지
이어지는 host-service 구간이 후속 phase와 겹쳤다는 시간 순서는 직접 확인됐다.

## Partition 실현성과 해석

| Service run | 비자명한 split snapshots | First batch match | 전체 frontier 동일 순서 실현 |
|---|---:|---:|---:|
| D1 outlier | 31 | 31 | 2 |
| D3 outlier 1 | 36 | 36 | 36 |
| D3 outlier 2 | 37 | 37 | 37 |

D1 반복은 보통 계획한 frontier 그대로 실행됐지만 총 dispatch가 많고 느렸다. D4+D3
계획은 다음 sampling/requeue가 끼면서 29/31번 다른 trajectory로 갔다. 즉 현재 DP에는
두 개의 독립된 문제 가능성이 있다: (1) 현재 action의 host-service label이 후속 phase로
오염될 수 있음, (2) 맞게 예측한 partition도 asynchronous request transition에서
그대로 실현되지 않을 수 있음. 첫 번째 run의 성능이 나았다는 사실은 오염된 D1 비용을
정책 prior로 삼으라는 뜻이 아니다. 그것은 우연히 불리한 D1×3 선택을 막은 것이다.

## 다음 구현 기준

1. Decode service sample은 dispatch 시작 시점뿐 아니라 **sampling/state commit까지의
   interval 전체**에서 후속 E/P/D 시작 여부를 검사한다. 혼입된 표본은 isolated cost에
   넣지 않거나 별도 contended cost로 구분한다. 이 필터의 reject 수와 비용곡선을 기록한다.
2. 동일 binary/engine/trace에서 필터 전후 D1/D3 cost, partition, throughput,
   TTFT/TPOT/E2E와 exact output을 비교한다. D1과 D3 둘 다 안정적으로 직접 관측돼야 한다.
3. 그다음에도 D4+D3처럼 planned partition의 실현율이 낮으면, DP의 equal-work
   action horizon과 실제 requeue boundary를 맞추는 수정 여부를 판단한다.
4. Gemma 전용 D3 강제 규칙, static D 비용표 복귀, p95를 임의로 낮추는 임계값은
   이 진단의 해결책으로 사용하지 않는다.

이번 결과는 "D3 kernel이 느리다"가 아니라 **학습 label의 lifetime과 DP의 평가
horizon이 실제 비동기 실행 경계와 맞지 않는다**는 근거다. 아직 필터 수정 후 full12,
vLLM 비교, production 승격을 주장하지 않는다.
