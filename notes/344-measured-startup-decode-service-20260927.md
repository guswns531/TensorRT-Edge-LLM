<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 344. Startup-generated decode service model

## Goal and scope

기존 수동 D 비용표를 기본값으로 두고 변경 허가만 주는 방식에서 벗어난다.
새 opt-in 경로는 시작부터 D static table을 제거하고 현재 process의 측정으로 초기 상태를 구성한다.
Champion은 외부 비교용으로 보존한다. 새 경로 내부에서 legacy 비용으로 fallback하지 않는다.
이번 범위는 D batch selection이다. E/P/overlap 전체 자동 최적화 또는 모든 휴리스틱 제거 완료가 아니다.

## Implementation

- `PhaseRuntimeCostTracker`: CUDA action 비용과 분리된 host decode service sample window를 유지한다.
  Key는 batch/context bucket/실제 graph variant이며 runtime 동안 계속 갱신된다.
- `IndependentPhaseAsyncServer`: 실제 decode dispatch의 prepare-start timestamp와 matching sampling ticket을
  연결하고 token collect/state commit 후 host elapsed를 기록한다. P/D overlap 및 encoder-active로 표시된
  dispatch는 수집에서 제외한다. 이는 GPU action의 isolated 표시에 따른 필터이며, sampling 대기 중 다른
  작업이 시작될 가능성까지 제거한 실험실 isolated latency는 아니다.
- `PhaseQueueScheduler`: 새 모드는 constructor에서 static D table을 제거한다. Host service p95/uncertainty로
  현재 ready rows의 동일 작업량 partition을 비교한다. GPU 예측 필드는 별도 CUDA 비용으로 유지한다.
  미측정 remainder를 큰 shape로 대체하거나 비례 보간하지 않는다.
- Dense shape가 미측정이면 현재 ready rows를 engine cap 안에서 실행하고 관측한다.
  이는 모델별 fixed batch fallback이 아닌 work-conserving bootstrap이다. 미래 arrival을 기다리지 않는다.
- `PhaseStartupCalibrationOptions`: `TRT_EDGELLM_STARTUP_DECODE_SERVICE=1`은 calibration enabled를 요구하고,
  plan-only 및 delayed GPU-only activation과 혼용을 거부한다.
- Startup probe는 GPU/P coverage와 함께 host-service coverage를 확인한다. JSON에 source, median/p95,
  uncertainty/sample count를 남긴 뒤 ready를 알린다. 현재 research harness는 require-coverage=true로
  미충족 시 readiness를 보류한다. 일반 production entrypoint 전체로의 배포는 아직 아니다.
- Runner variant: `independent-autotune-service`. Chunk128, engine, KV pool, original generic warmup은 유지한다.

## Remaining assumptions

정확한 graph variant와 context bucket이 일치하는 직접 관측만 batch 선택에 사용한다.
Bucket 크기, 최소 sample 수, rolling window, uncertainty 계산은 기존 tracker 설정을 재사용한다.
배치 비용 합산은 짧은 horizon 근사이며 P 삽입/미래 cohort를 정확히 예측하는 rollout이 아니다.
동일 비용이면 큰 첫 batch를 선택한다. 명시적 SLO 없이 기존 V3 phase policy와 함께 동작한다.
V3의 다른 phase 비용 prior와 recovery/formation 규칙은 이번 변경의 제거 대상이 아니다.

## Minimal validation plan

1. Unit: static prior 무시, GPU-only와 host-service 선택 구분, 미측정 remainder/shape 처리, GPU model 분리.
2. 두 모델 `balanced`만, 같은 새 binary/engine에서 startup-shadow(static D)와 startup-service를 각1회 비교.
   공통 probe/warmup 구성은 유지하지만 정책이 달라 관측 trajectory까지 같다고 주장하지 않는다.
3. Throughput 및 TTFT/TPOT/E2E mean/p95 기록. Full12와 fresh vLLM은 하지 않는다.
   기존 vLLM은 contract가 같은 frozen 결과만 참고한다. 단일 반복은 promotion evidence가 아니다.
4. `.local/current/*/runtime`은 변경하지 않는다. 성능 차이는 기존에 복귀할 내부 조건이 아니라
   다음 개선 과제를 결정하는 외부 evaluation 결과다.

## First validation failure and correction

첫 binary (`0cd290586f652e93976948b0c4beb2e128200c30668b5979ba1fbf03444af50c`)의
static-shadow 2개 셀은 완료했지만 service 2개 셀은 startup coverage가 부족해 ready를 거부했다.
결과는 `.local/results/startup-service-20260927/balanced`에 실패 상태 그대로 보존한다.

GPU model은 selected/observed key 및 action fidelity가 일치하는 sample만 인정한다.
처음 host-service collector에는 같은 필터가 없어 GPU 3개/host 4개 상태가 발생했다.
Host coverage만 확보되자 D2/D4 등의 dense candidate가 partition 비교에서 빠지고,
이미 알려진 작은 batch만 실행되어 GPU의 네 번째 sample을 영구히 못 얻었다.
이는 모델별 튜닝 문제가 아니라 측정과 선택 사이 bootstrap 계약의 오류다.

수정은 두 가지다.

- Host 관측에도 GPU와 동일한 selected/observed key, candidate parity, action fidelity 조건을 적용한다.
- Dense shape의 host/GPU coverage가 모두 없으면 ready cohort를 실행하여 관측을 확보한다.
  Static table로 복귀하지 않는다. Cost-history reset은 host-service 관측도 함께 초기화한다.

회귀 단위 테스트는 host-only dense coverage가 있어도 GPU coverage를 확보하기 전에는 분할하지 않는지 검사한다.
수정본은 기존 artifact를 덮어쓰지 않고 `startup-service-v2-20260927`에 별도로 실행한다.

## Reproduction

```bash
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma cosmos --workloads balanced \
  --variants independent-autotune-shadow independent-autotune-service \
  --repeats 1 --startup-budget-ms 120000 \
  --build-root .local/baselines/startup-service-v2-20260927/bin \
  --binary-source-commit 6c42dfb --compress-closed-logs \
  --result-root .local/results/startup-service-v2-20260927/balanced
```

Source: `6c42dfb` + campaign `source.patch`. Binary SHA256:
`58d950ea4dd5a81ba56ee0b430ce6459abadbd0e9c13410413c6e3925a4ce167`.
Engine/plugin/workload identity 및 모든 command는 campaign manifest에 기록한다.
동일 binary로 비교하며, 이전 binary 결과와 혼합 평균하지 않는다.

## Minimal HTTP results

수정 후4/4 cells 완료. 모델별 balanced1회이며 full12/repeat3 통과 주장이 아니다.
Shadow는 기존 static D 비용표 유지 + startup GPU calibration/plan-only이고,
Service는 static D 비용표 없음 + 측정 host-service partition 선택이다.
Engine, KV, chunk128, E/P/D 상한, generic HTTP calibration 구성은 동일하다.
정책이 startup부터 달라지므로 calibration의 실행 순서/관측값까지 동일하지는 않다.

### Startup coverage

| 모델 | Static startup, s | Service startup, s | 추가 D probe 요청, 양쪽 동일 | Service coverage | Static D table |
|---|---:|---:|---:|---|---|
| Gemma | 19.469 | 21.555 | 115 | true | false |
| Cosmos | 21.436 | 23.417 | 187 | true | false |

시간은 startup report의 내부 calibration 구간이며 model load부터 HTTP ready까지의 전체 시간이 아니다.
추가 probe 요청 수는 generic shape warmup/encoder calibration/HTTP warmup을 모두 합한 수가 아니다.
Gemma D1/2/4/8/16/24, Cosmos D1/2/4/8/16/32/64의 계획된 context frontier를 확인했다.
모든 context/모든 중간 batch를 전수 보정한 것은 아니다. 미측정 shape는 serving 중 관측한다.

### Serving metrics

Latency는 ms, 각 셀은 mean / p95. 기존 aggregate의 TTFT/E2E는 **HTTP send 기준**이다.
클라이언트 concurrency 대기까지 포함한 scheduled-arrival latency는 다음 표에서 별도로 기록한다.
TPOT는 request별 `(E2E - TTFT)/(output_tokens-1)`이며 개별 inter-token gap p95가 아니다.

| 모델/경로 | tok/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | peak MiB |
|---|---:|---:|---:|---:|---:|
| Gemma static | 1249.92 | 84.79 / 211.85 | 15.65 / 17.33 | 1387.52 / 2165.55 | 9389 |
| Gemma service | 1198.61 | 76.78 / 210.30 | 15.77 / 16.86 | 1394.14 / 2163.98 | 9385 |
| Gemma frozen vLLM | 771.46 | 134.27 / 234.33 | 23.76 / 24.56 | 2128.03 / 3244.79 | — |
| Cosmos static | 4388.17 | 60.20 / 152.68 | 12.69 / 14.23 | 1139.43 / 1782.93 | 9741 |
| Cosmos service | 4202.15 | 58.12 / 155.58 | 13.23 / 14.75 | 1179.48 / 1847.21 | 9725 |
| Cosmos frozen vLLM | 4315.77 | 112.40 / 254.04 | 12.20 / 13.58 | 1154.35 / 1771.35 | — |

| Service의 static 대비 변화 | tok/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|
| Gemma | -4.11% | -9.45% / -0.73% | +0.75% / -2.70% | +0.48% / -0.07% |
| Cosmos | -4.24% | -3.46% / +1.90% | +4.21% / +3.65% | +3.52% / +3.61% |

위 변화율은 `(service/static - 1)`이다. 단일 run이므로 반복 분산/통계적 유의성은 확인하지 않았다.
vLLM은 새 실행이 아닌 frozen 결과이며 service throughput은 Gemma +55.37%, Cosmos -2.63%다.
전체7개 지표의 상대 변화는 `startup-service-v2-20260927/frozen-comparison.{md,json}`에 저장했다.
원본 참조는 campaign manifest의 `frozen_vllm` 및 comparison JSON을 따른다.

### Arrival queue accounting

Retained `requests.csv`에서 arrival→first token/completion을 재계산했다.
p95는 client aggregate와 같은 선형 보간 percentile이다.

| 모델/경로 | Arrival TTFT mean/p95, ms | Arrival E2E mean/p95, ms | Client wait mean/p95, ms |
|---|---:|---:|---:|
| Gemma static | 1216.95 / 2783.63 | 2519.67 / 4096.31 | 1132.15 / 2692.53 |
| Gemma service | 1205.16 / 2772.01 | 2522.51 / 4104.41 | 1128.38 / 2723.98 |
| Cosmos static | 1939.14 / 4208.24 | 3018.37 / 5086.31 | 1878.94 / 4146.71 |
| Cosmos service | 2021.83 / 4394.03 | 3143.20 / 5272.65 | 1963.71 / 4352.01 |

Cosmos send 기준 TTFT mean은 감소했지만 arrival 기준은 증가했다.
따라서 이를 전체 사용자 대기 개선으로 표현하지 않는다.

## Mechanism evidence and limits

Activity interval CSV에서 sampling 구간을 제외한 dispatch interval을 집계했다.
시간 합은 겹치는 구간을 각각 더한 event duration이며 elapsed serving time/SM utilization이 아니다.

| 모델/경로 | D dispatches | D duration sum, ms | P dispatches | P duration sum, ms |
|---|---:|---:|---:|---:|
| Gemma static | 296 | 3803.80 | 41 | 828.64 |
| Gemma service | 325 | 3983.60 | 44 | 909.56 |
| Cosmos static | 477 | 4261.01 | 187 | 2965.56 |
| Cosmos service | 488 | 4192.92 | 197 | 2963.56 |

- Gemma: D dispatch +9.80%, D duration +4.72%, P dispatch도 증가했다.
  더 잘게 실행된 trajectory가 회귀와 함께 관측되었다. 고정 snapshot causal replay는 아니므로
  partition 선택 하나의 인과효과로 전부 설명하지 않는다.
- Cosmos: D dispatch +2.31%인데 D duration 합은 -1.60%다.
  GPU duration 감소가 전체 처리량 증가를 보장하지 않는다. Formation/host gap/overlap 배치 위치를
  분해해야 하며 이번 집계만으로 특정 host 병목을 확정할 수 없다.
- Host-service cost에는 sampling/state commit이 포함되지만 다음 candidate 형성, 다음 enqueue까지의
  모든 비용은 포함되지 않는다. 개별 batch service를 더한 DP와 실제 asynchronous trajectory는 다르다.
- Gemma startup context1에서 D4 median6.786ms, D8 median13.065ms였지만
  p95는7.020/14.347ms다. Tail/uncertainty 기준은 median 기준과 partition 순위가 달라질 수 있다.
  이 예만으로 실제 모든 split의 원인이라고 주장하지 않는다.

## Correctness and validation scope

- C++ targeted4 suites217 tests, Python45 tests, pre-commit 모두 통과.
- Gemma64 requests/5440 tokens, Cosmos288 requests/24960 tokens 모두 완료했다.
- Static/service exact token identity: Cosmos288/288 requests, 24960/24960 token positions 일치.
  Gemma61/64 requests, 5141/5440 positions 일치로 **exact output parity 미통과**다.
  Shape/order 수치 영향인지 별도 오류인지는 이번 측정으로 확정하지 않는다. Semantic correctness 통과로
  간주하지 않으며 default promotion/quality 승인을 보류한다.
- 두 모델 `.local/current/*/runtime`은 `throughput-9db3ed3-20260926/bin`으로 유지했다.
  실험 경로 내부에는 static table fallback이 없다. 기본 배포 보류와 실험 정책 fallback은 다른 개념이다.
- 실험 종료 후 실행 container 없음, GPU1MiB. Engine/model 재생성이나 삭제 없음.

## Next bounded step

1. 수동 D 표 없는 구조는 유지한다. 이번 처리량 회귀를 이유로 내부에서 옛 표를 다시 쓰지 않는다.
2. 새 전체 campaign 대신 Gemma의 **같은 ready rows에 대한 dense/split + 공통 successor** 한 점을
   async server 실제 경로로 비교한다. `prepare→commit`과 `commit→next enqueue`를 분리해
   startup service 합산이 놓치는 cohort/launch 비용을 찾는다.
3. 같은 작은 replay에서 Gemma3개 divergent request를 BS1 및 canonical membership/order와 비교한다.
   출력 변화 원인을 확인하기 전 throughput만으로 승격하지 않는다.
4. 확인된 비용 항목만 startup evaluator에 추가하고 Cosmos 한 점으로 모델별 특수 규칙이 아닌지 확인한다.
   그 전에는 full12나 fresh vLLM을 반복하지 않는다.

이번 단계의 결론은 **자동 초기 상태 생성/운영 경로 확보**, 아직 **성능 최적화 완료 아님**이다.
이후 목표는 측정된 현재 장비 비용에서 설정을 고르는 것이며 기존 수동 설정을 이겨야만 변경 권한을
주는 구조가 아니다. 다만 측정 기반이라는 이유만으로 목적함수/평가 contract까지 자동으로 올바른 것은 아니다.
