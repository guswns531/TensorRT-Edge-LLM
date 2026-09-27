<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 349. Decode service interval guard: 동일 binary on/off 검증

## 문제와 수정

[348](348-decode-service-label-contamination-20260927.md)에서 D3의 host-service median은
약 6.6 ms인데 p95는 약 27.7 ms였다. 긴 표본 두 건은 D3 dispatch 시작 시점에는
`concurrentPrefillActive=false`였지만 sampling/state commit 전에 다음 P가 시작했다.
그 값이 `isolated` 비용표에 들어가 D1×3 split을 선택하게 했다.

이번 변경은 기존 `onMetrics`의 dispatch-start 필터를 유지하면서, sample의
prepare-start부터 token-state commit까지 후속 E/P/D의 시작 여부를 추가로 확인한다.
새 phase가 이미 완료됐다면 마지막 dispatch-start timestamp로, 아직 실행 중이면
`coordinator.busy()`로 감지한다. 외부 encoder는 시작 timestamp와 active 상태를
별도로 확인한다. 혼입 sample은 isolated decode-service tracker에 넣지 않는다.
허용·거부 건수는 startup report와 full HTTP telemetry에 기록한다.

기본값은 interval guard **on**이다. `TRT_EDGELLM_DISABLE_DECODE_SERVICE_INTERVAL_GUARD=1`은
동일 binary A/B를 위한 명시적 연구 ablation이며, 일반 정책이나 모델별 batch 규칙은
추가하지 않는다. `phaseDecodeServiceSampleIsolated()`의 경계/후속 phase/in-flight/
encoder 조건을 C++ 단위 테스트로 검사했고, Python harness의 opt-in 전달 계약도
테스트했다. 기본 V3 경로는 measured host-service를 사용하지 않으므로 이 필터가
스케줄러 선택을 바꾸지 않는다.

## 공정 비교 계약

Primary A/B는 RTX 3080 10 GiB, NVIDIA driver 610.57.04, TensorRT Docker 26.06에서
같은 Gemma AWQ engine, 같은 balanced JSON 요청 64개와 5,440 출력 tokens, 같은 chunk/
batch/KV/graph/calibration/full telemetry를 사용한다. 차이는 interval guard on/off 또는
static-shadow 정책뿐이다. 세 campaign의 binary SHA256은 모두
`f1a4154fdb7c64d0d60ad34603949470e11c98d459f237553e0df6d3273494b1`이다.
각 run은 새 process이고 각 모드 2회다. 명령, source patch, hash, startup, HTTP CSV,
compressed gateway log는 아래 manifest에 있다.

- `.local/results/decode-service-guard-ablation-20260927-off/manifest.json`
- `.local/results/decode-service-guard-ablation-20260927-on/manifest.json`
- `.local/results/decode-service-guard-ablation-20260927-static/manifest.json`

재현은 같은 runner 인자로 `--models gemma --workloads balanced --repeats 2
--startup-budget-ms 120000 --telemetry-level full --decode-partition-diagnostic
--compress-closed-logs`를 사용한다. `--variants independent-autotune-service`를 on/off
모두에 쓰며 off에서는 명령 앞에 위 환경변수만 설정한다. Static은
`--variants independent-autotune-shadow`다. `.local/current/*`는 변경하지 않았다.

## HTTP 결과

TTFT/TPOT/E2E는 HTTP send 기준 ms의 두 run 평균이며 각 칸은 mean / p95다.
Tokens/s 역시 두 run 평균이다. 2회이므로 신뢰구간이나 promotion 근거가 아니다.

| 경로 | tokens/s | TTFT | TPOT | E2E | D dispatch 평균 | D1 평균 |
|---|---:|---:|---:|---:|---:|---:|
| Service, guard off | 1160.48 | 80.10 / 209.76 | 15.95 / 17.30 | 1410.95 / 2185.31 | 319.5 | 57.5 |
| Service, guard on | 1208.35 | 80.14 / 213.36 | 15.85 / 17.26 | 1400.47 / 2205.69 | 304.0 | 21.0 |
| Static shadow | 1227.42 | 82.86 / 211.91 | 15.67 / 17.13 | 1388.66 / 2170.04 | 294.0 | 18.0 |

Guard on은 off 대비 tokens/s `+4.12%`, TPOT mean `-0.61%`, E2E mean `-0.74%`다.
반면 TTFT p95는 `+1.72%`, E2E p95는 `+0.93%`로 나빠졌다. Static 대비 guard on은
tokens/s `-1.55%`, TTFT mean `-3.28%`, TPOT mean `+1.12%`, E2E mean `+0.85%`,
E2E p95 `+1.64%`다. 즉 오염 제거가 처리량 회귀 대부분을 회복했지만 모든 latency
지표를 동시에 개선하지는 않았다.

이전 binary의 필터 전후 비교도 비슷한 방향이었지만 primary 수치는 위의 동일 binary
A/B만 사용한다. Driver가 다른 frozen vLLM을 이번 진단의 직접 수치 비교로 재사용하지
않았고 새 vLLM이나 full12도 실행하지 않았다.

## 비용곡선과 표본 처리

| 동일 binary run | D3 p95, 첫 measurement | D3 표본 수 | D1 p95, 첫 measurement | startup 허용/거부 | 종료 시 허용/거부 |
|---|---:|---:|---:|---:|---:|
| Off 1 | 27.66 ms | 30 | 6.42 ms | 1138 / 0 | 1604 / 0 |
| Off 2 | 27.57 ms | 30 | 6.43 ms | 1162 / 0 | 1628 / 0 |
| On 1 | 6.75 ms | 16 | 6.18 ms | 791 / 9 | 1139 / 49 |
| On 2 | 6.75 ms | 14 | 6.15 ms | 817 / 12 | 1172 / 45 |

Guard on에서 종료 시 D3 p95는 각각 6.60/6.67 ms였고, 보정 coverage는 두 run 모두
성공했다. 허용·거부 값은 startup부터 누적된 카운터다. Off와 on의 허용 건수 차이
전체를 `rejectedInterleaved`로 해석하면 안 된다. 선택한 batch와 관측 trajectory도
달라서 생성된 후보 sample 수가 달랐다.

이 동일 binary 비교는 긴 표본이 robust D3 cost를 높이고 D1×3을 선택하게 만든다는
348의 원인 해석과 일치한다. Off에서는 D1 반복 partition이 각 run 38/37개,
on에서는 D1 반복 partition이 0개였다. D1 dispatch는 58/57에서 21/21로 줄었다.

## Cosmos 단일 smoke

같은 필터 기본값으로 Cosmos balanced 한 셀도 실행했다. Startup frontier coverage를
통과했고 288 requests/24,960 output tokens를 모두 완료했다. Startup service sample은
932건 허용/12건 거부, 종료 시 누적 1389건 허용/48건 거부였다. 처리량은
4108.55 tokens/s, TTFT mean/p95는 64.38/160.70 ms, TPOT 12.81/14.29 ms,
E2E 1150.11/1776.60 ms였다. 이 한 셀에는 동일 binary의 Cosmos off/static 대조가
없으므로 성능 향상이나 exact-output parity를 주장하지 않는다. Manifest는
`.local/results/decode-service-guard-cosmos-smoke-20260927/manifest.json`이다.

## 남은 두 gate

첫째, guard on에도 DP가 `[4,3]` 또는 `[4,4]`를 예측한 snapshot이 15/18개 있었다.
첫 batch는 모두 계획대로 나갔지만, 동일 frontier의 전체 partition이 계획대로
실현된 것은 **0/15, 0/18**이다. Sampling/requeue 후 이미 한 token 진행한 row가
다시 선택돼 equal-work DP의 평가 horizon과 실제 실행 horizon이 다르다.
Static보다 남은 약 1.55% 처리량 손실은 이 현상과 연관될 수 있지만, 이 값 전부를
partition mismatch의 인과효과로 귀속할 수는 없다.

둘째, exact greedy output은 static과 guard-off/on 사이에 요청 60~62/64개만 일치했다.
두 run 모두 5,440 tokens를 생성했지만 output-quality gate는 여전히 실패한다.
Batch-shape별 logit divergence는 347에서 단일 턴에도 재현됐다. 따라서 이번
throughput 회복은 production promotion 또는 correctness 승인 근거가 아니다.

## 다음 작업

1. Service 관측 필터는 유지하고 더 많은 workload/모델에서 허용·거부 비율과 startup
   coverage를 확인한다. Filter가 probe 부족을 만들면 threshold를 임의 완화하지 말고
   안전한 isolated probe를 늘리는 쪽으로 수정한다.
2. 별도로 `[4,3]`의 실제 requeue boundary를 모델링하거나 다음 action까지의 비용을
   평가해 planned partition과 실제 trajectory를 맞춘다. D3 강제나 static 비용표
   복귀 같은 모델 전용 규칙은 사용하지 않는다.
3. Output-quality gate를 통과할 때까지 기본 runtime `.local/current` 승격과
   vLLM 대비 최종 성능 claim을 보류한다.
