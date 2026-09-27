<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# 342. Startup autotuning step 1: decode trial planner

후속 직접 비교 결과: [343. Equal-work decode trials](343-startup-decode-equal-work-trials-20260927.md).

## 범위

사용자의 요청은 시작 시 짧은 자동 실험으로 실행 설정을 고르되, 한 단계씩 필수 실험만 수행하는 것이다.
341의 measured-decode 전환은 모든 경우에 이득이 아니었으므로 측정값을 곧바로 정책에 적용하지 않는다.
이번 단계는 **측정 → 동일 row 수의 비교 후보 생성 → 보고서 저장**까지만 구현한다.
자동 설정 확정, production readiness gate 완성, 성능 향상을 뜻하지 않는다.

기존 engine, KV pool, E/P/D maximum batch, chunk128, graph 설정과 static decode table을 유지한다.
기존 shape/HTTP warmup도 축소하지 않는다. 추가 startup probe는 RLS posterior에 영향을 줄 수 있으므로
`plan_only`는 후보 생성기의 정책 비적용이지, probe 없는 실행과 dispatch identity를 보장하는 말이 아니다.

## 구현

- `phaseServingExecutionOptions.{h,cpp}`: 측정된 dense D batch와 최대 두 번의 split D dispatch를 열거한다.
  예: D24 대 D8+D16. 모든 후보의 row 합은 동일하며, 미측정 remainder를 보간하지 않는다.
- `llm_phase_context_smoke.cpp`: trusted startup cost를 context bucket과 graph/eager variant별로 묶고
  `decode_trial_plan`을 startup JSON에 기록한다. 서로 다른 bucket/variant 비용은 섞지 않는다.
- `run_lifetime_encoded_admission.py`: opt-in `independent-autotune-shadow` 변형을 제공한다.
  `TRT_EDGELLM_STARTUP_PLAN_ONLY=1`은 calibration enabled를 요구하고 measured-policy 활성화와 동시 사용을 거부한다.
- 출력에는 dense reference, split batches, estimated GPU cost, uncertainty sum, guarded saving을 남긴다.
  `policy_applied=false`, `status=requires_equal_work_validation`을 명시한다.

`guarded saving = dense median - dense uncertainty - sum(split median + split uncertainty)`이다.
이는 다음 실험의 우선순위를 정하기 위한 **bucketed GPU proxy**일 뿐 E2E 이득이나 통계적 신뢰구간이 아니다.
같은 bucket이어도 정확한 KV length는 다를 수 있다. Host 제출 비용, sampling, 다음 P/D cohort 변화도 포함하지 않는다.
실제로 동일 KV length/row set에서 dense와 split을 실행해야 자동 선택 근거가 된다.

## 이번 검증 범위

1. 후보 row 보존, asymmetric remainder, missing reference, uncertainty, 잘못된 입력 및 정책 활성화 금지 단위 테스트.
2. Gemma/Cosmos 각각 `short` HTTP smoke 1회. Startup coverage 및 trial report 생성과 요청 완료 확인.
3. Full12와 fresh vLLM은 실행하지 않는다. 이번에는 serving winner를 바꾸지 않으며 성능 A/B 결론도 내리지 않는다.

결과 경로: `.local/results/startup-autotune-step1-20260927/`.
진단 바이너리: `.local/baselines/startup-autotune-step1-20260927/bin`.
기본 `.local/current/*/runtime`은 변경하지 않는다.

## 실행 결과

소스는 `d0e168c8e6e3246f50da99bfd6bc7e7ddb74eb33` + 이번 변경이다.
실행 당시 patch, binary/plugin/engine hash, command와 workload identity는 `smoke/manifest.json` 및
`smoke/source.patch`에 보존했다. 바이너리 SHA256은
`7e260afd135c918cd4a6d9971b269532195722af5b2c86bb9800fae992a8dbb5`이다.

- 관련 C++ 16 tests 통과, Python runner 44 tests 통과, pre-commit 통과.
- 두 HTTP cell 성공, failure 0. 두 모델 모두 48 requests / 1,040 output tokens 완료.
- 두 모델 모두 `frontier_covered=true`, `plan_only=true`, `static_decode_table=true`,
  `measured_decode_at_measurement=false`, `policy_applied=false` 확인.
- 보고서의 모든 후보에서 row 합 보존 및 동일 context bucket/variant 내 측정 batch 사용을 확인했다.

| 항목 | Gemma | Cosmos |
|---|---:|---:|
| Startup 보고서 elapsed | 19.740 s | 21.675 s |
| 추가 decode-probe requests | 115 | 187 |
| Context/variant/target-row groups | 12 | 13 |
| Split 후보 수 | 8 | 10 |
| Guarded saving > 0 후보 | 0 | 0 |

Elapsed는 기존 코드의 startup timer 구간이며 **모델 로드부터 HTTP measurement-ready까지의 전체 시간이나
새 planner의 추가 실행 비용이 아니다**. Decode-probe requests 역시 HTTP generic warmup 및 다른 warmup 요청을
모두 합친 숫자가 아니다. 기존 calibration 경로는 D 외 P/E evidence도 수집한다.

Gemma의 가장 근접한 후보는 context bucket 1에서 D8 대 D4+D4이다.
Dense GPU proxy 13.078 ms, split 13.188 ms로 차이가 작지만 uncertainty를 포함한 guarded saving은 -1.932 ms이다.
Cosmos의 가장 근접한 split은 bucket 1의 D64 대 D32+D32이며 guarded saving은 -5.927 ms이다.
**이 결과로 split 우위를 주장하거나 자동 적용하지 않는다.** 또한 dense를 항상 최대 batch로 실행해야 한다는
결론도 아니다. 이 모델은 대기, P/D 간 우선순위, request별 latency와 미래 cohort를 평가하지 않는다.

### HTTP smoke 기록 — 성능 A/B 아님

| 지표 | Gemma | Cosmos |
|---|---:|---:|
| Throughput (output token/s) | 777.879 | 2,425.775 |
| Request/s | 35.902 | 111.959 |
| TTFT mean / p95 (ms) | 97.918 / 224.972 | 90.465 / 190.573 |
| TPOT mean / p95 (ms) | 23.399 / 29.751 | 13.120 / 23.354 |
| E2E mean / p95 (ms) | 564.540 / 940.365 | 330.748 / 413.118 |

각 1회이며 동시점 baseline 반복이 없다. 기존 champion/vLLM 대비 향상·회귀 판정에는 사용하지 않는다.
특히 Gemma client dispatch delay p95는 833.023 ms로 기록됐으므로 GPU service 단독 결과로 해석하면 안 된다.
Default binary/policy는 기존 champion 그대로이며 이 결과는 `diagnostic`이다.

재현 명령:

```bash
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma cosmos --workloads short \
  --variants independent-autotune-shadow --repeats 1 \
  --startup-budget-ms 120000 \
  --build-root .local/baselines/startup-autotune-step1-20260927/bin \
  --binary-source-commit d0e168c --compress-closed-logs \
  --result-root .local/results/startup-autotune-step1-20260927/smoke
```

## 다음 단계 및 확대 조건

1. 이번 후보 중 소수만 동일 row/KV length, 동일 graph availability에서 dense/split 직접 비교한다.
   GPU event뿐 아니라 host 포함 drain 시간과 다음 cohort까지 기록한다. 후보 생성만으로 설정을 확정하지 않는다.
   우선 Gemma D8 대 D4+D4, Cosmos D64 대 D32+D32 두 점만 검사한다. 전자는 근접 후보이고 후자는
   split이 불리하다는 proxy 예측을 확인하는 통제점이다. 유리한 split이 없으면 dense를 유지하고 탐색을 종료한다.
2. 개선 근거가 있는 후보만 opt-in으로 적용하고 `short`, `balanced` 같은 관련 회귀점부터 검증한다.
3. 이후 P/E/overlap 자동 실험을 각각 별도 단계로 확장한다. 여러 knob를 한 번에 바꾸지 않는다.
4. 실제 기본 정책을 승격할 때만 두 모델 full12와 기존 contract가 동일한 frozen vLLM을 비교한다.
   모든 비교에서 throughput, TTFT/TPOT/E2E mean/p95를 보고한다.
