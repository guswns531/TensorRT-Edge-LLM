<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Startup calibration 원인 분리

## 목적과 변경 범위

Note 340의 combined startup 회귀를 비용 원천, 추가 관측, shape warmup,
HTTP warmup으로 분리한다. 현재 promoted runtime/engine/KV/정책 기본값은 유지한다.
모델/engine export 변경 없이 동일 engine을 사용한다. 새 코드 경로는 opt-in이다.

## 순차 비교 계약

| ID | Runner variant | 측정 시 D 비용 | 추가 startup probe | Shape warmup | HTTP warmup |
|---|---|---|---|---|---|
| A | independent | 기존 | 없음 | 기존 | 기존 |
| B | independent-measured | 실측 | 없음 | 기존 | 기존 |
| C | independent-calibrated | 실측 | P/D + E | 기존 | 기존 |
| D | independent-compact-shapes | 실측 | P/D + E | capability-derived | 기존 |
| E | independent-compact-http | 실측 | P/D + E | capability-derived | epoch marker 1개 |

B–E는 calibration 동안 기존 D 비용 정책을 유지하고, 모든 요청/stream을 drain한
calibration_end에서 비용 원천만 변경한다. Exact 관측과 RLS posterior는 reset하지 않는다.
A/B의 *요청과 shape 구성*은 동일하지만 비동기 GPU 실행 timing과 실제 posterior가
bit-identical하다는 주장은 하지 않는다. 강제 replay가 없는 live A/B라는 한계가 있다.

C/B는 추가 probe가 만드는 exact 관측과 contextual 관측을 함께 평가한다.
따라서 이를 'exact 관측만의 효과'라고 표현하지 않는다. D/C는 shape warmup 구성,
E/D는 HTTP warmup 수를 분리한다. 기존 independent-startup은 호환성을 위해 유지하지만
새 primary matrix에는 섞지 않는다: 그 경로는 warmup 중에도 실측 D 정책을 쓴다.

## 검증

1. CPU 계약: A/B 환경 한 필드 차이, B/C 기존 shape 유지, C/D shape 제거만,
   D/E HTTP warmup 수만 변경, combined mode 혼용 거부.
2. C++: idle boundary만 비용 원천 전환 허용; 관측을 보존한 measured/static decision identity.
3. 기존 두 모델 × full12 × A–E, 같은 바이너리/engine/chunk/KV/HTTP trace로 각 1회 진단.
4. 핵심 회귀 balanced/short/decode-heavy와 이득 long-prefill/multi-image의 후속 반복은
   1차 결과를 보고 선택한다. 1회 결과로 promotion하지 않는다.
5. 처리량, TTFT/TPOT/E2E mean/p95, P/D dispatch/mean batch, peak memory,
   calibration 방향별 coverage를 보고한다. Frozen vLLM은 계약 불변이므로 재사용한다.

Coverage와 policy readiness는 분리한다. 관측수 충족은 held-out prediction error,
action ranking 안정성 또는 RLS 수렴을 보장하지 않는다. 보고서는
`policy_stability_validated=false`를 명시한다. Warmup 축소를 성능/안정성 gate 없이
기본값으로 승격하지 않는다.

## 실행 상태

주 비교 120/120 HTTP cells 성공, 실패 0. 각 cell 1회이며 전체 3회 gate는 아니다.
핵심 지점 추가 20회와 full-telemetry 학습 진단 6회도 모두 성공했다.
결과는 `.local/results/startup-factorization-20260927/`에 저장한다.
현재 champion `.local/current/*/runtime`은 변경하지 않는다.

첫 Cosmos C smoke는 30초 보정 예산을 소진해 E 관측 전에 strict readiness가 거부됐다.
해당 실패는 `smoke/`에 보존하고 성능 비교에서 제외한다. 기존 학습량을 보존한 상태에서
budget truncation이 원인 분리를 방해하지 않도록 primary matrix는 전부 120초 준비 예산을
사용한다. 이는 진단의 종료 한도이며 서빙 정책/SLO/모델별 최적화 값이 아니다.

120초 재확인(`smoke-budget120/`)에서는 시간 예산이 아닌 D32/context2 coverage가
실패했다. Prompt896의 prefill은 7 chunks인데 기존 probe output14는 이를 한 chunk로
계산했다. 기존 D 정책 아래서는 초기 decode row가 먼저 끝나 목표 cohort 관측이 0개였다.
새 계산은 prefillTurns=ceil(rows/Pmax), chunks=prompt/chunk로 두고
output=2*prefillTurns*chunks+minimumSamples+2를 예약한다.
Prompt+output이 KV/sequence capacity에 들어가는 최대 chunks를 역산한다.
이는 calibration 요청 길이 수정이며 serving chunk/batch/정책 변경이 아니다.
엄밀한 cohort 보장은 실제 cost-key coverage로 확인한다.

수정 후 `smoke-chunk-aware/`는 21.357초에 coverage를 충족했다. 수정 전 120초
budget의 실제 경과는 56.525초였고, D32/context2에서 32 rounds를 수행해도 관측이
0개였다. 따라서 30초 실패를 '정상 보정에 원래 30초 이상 필요하다'로 해석하면 안 된다.
긴 prompt896의 output은 14에서 62로 바뀌었고 수정 후 해당 probe는 1 round에 충족했다.
실패 runs는 coverage withheld diagnostic이며 CUDA 메모리 오류의 증거로 쓰지 않는다.

## 구현 경계

```text
동일 engine / 동일 capability / 동일 V3
                  │
       원래 shape 및 HTTP warmup
       + 선택적 startup P/D/E probes
                  │
       모든 요청과 GPU activity drain
                  │
          calibration_end
                  │
       A: 기존 D cost table 유지
       B–E: 실측 D cost로 전환
                  │
       같은 measurement HTTP trace
```

- `phaseQueueScheduler.{h,cpp}`: `useMeasuredDecodeCosts()`는 pending/active/inflight가
  모두 없는 상태에서만 허용한다. D table을 비우고 measured batching을 켜며 exact
  tracker/RLS posterior를 보존한다. Warmup 재실행이나 learner reset이 아니다.
- `phaseServingExecutionOptions.{h,cpp}`: startup probes와 measurement 시점 전환을
  분리한다. 새 boolean `TRT_EDGELLM_MEASUREMENT_MEASURED_DECODE_COSTS`는 probe 없이도
  사용할 수 있고 invalid 값은 거부한다. Probe output reservation은 위 chunk 계산을 쓴다.
- `llm_phase_context_smoke.cpp`: calibration drain 뒤 전환한다. Startup JSON은
  startup 중 table 상태와 deferred activation을 별도로 기록한다.
  `policy_stability_validated=false`는 coverage 성공 시에도 유지한다.
- `run_lifetime_encoded_admission.py`: A–E를 같은 binary에서 실행하고 각 cell의
  `calibration_contract`와 HTTP warmup 수를 manifest에 남긴다. 기존 combined startup
  mode와 factorized variant의 혼용은 거부한다.
- `report_startup_calibration.py`: 임의 두 variant의 paired 비교 및 warmup coverage,
  P/D dispatch 비교를 지원한다. Probe 없는 B의 startup duration은 unknown/null이며
  0초 보정이라고 표시하지 않는다.
- `analyze_contextual_adaptation.py`: gzip full telemetry를 읽고 compact log를 거부한다.
  관측 0개인 family의 MAE/RMSE/false-safe rate는 unknown이며 오차 0으로 처리하지 않는다.

현재 구현은 HTTP benchmark backend의 startup/measurement 경로다.
Native production `PhaseServingRuntime::create()`에 자동 optimizer를 연결한 것은 아니다.
Readiness validation과 production promotion은 별도 gate로 남는다.

## 재현 계약과 provenance

| 항목 | 계약 |
|---|---|
| Source | `35fdafd1f701c807b59b3d42178920bb4500b98d` + 각 campaign의 `source.patch` |
| Runtime | `.local/baselines/startup-factorization-v2-20260927/bin` |
| Smoke SHA256 | `31083a8a77954dd646472a9059b158168ce9624fe1faf44d8d01fc2cbdb38d76` |
| Plugin SHA256 | `ddabc5df4d481bc2440d77a46862565f12343a8db8ba00ee34e42496dddad6c2` |
| Gemma | AWQ backbone, FP16 KV; E4/P8/D24, slots24, KV192 pages, sequence2048 |
| Cosmos | FP16; E4/P8/D64, slots80, KV256 pages, sequence2048 |
| 공통 | chunk128, P token budget1024, independent E/P/D, D graphs on/P graph cache0, V3, no explicit SLO |
| Preparation | 공통 budget120000ms, serving probes on, fixed-output ignore EOS |
| HTTP warmup | A–D: Gemma49/Cosmos239; E:1 epoch marker |
| Telemetry | 주 성능 비교 dispatch; learning diagnostic만 full |

Container digest, engine/model identity, full commands, dirty state, runner source/hash,
workload 및 frozen vLLM 경로는 campaign manifest에 있다. 최종 commit이 원래 binary의
clean build source였다고 소급 표기하지 않는다. Frozen vLLM은 계약이 같아 재사용했고
이번에 새로 측정하지 않았다. Gemma semantic/exact-output 미해결 gate와 기존 VLM
output-contract 제한은 그대로이며 throughput 비교로 면제하지 않는다.

```bash
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma cosmos --full12 --repeats 1 \
  --variants independent independent-measured independent-calibrated \
    independent-compact-shapes independent-compact-http \
  --startup-budget-ms 120000 \
  --build-root .local/baselines/startup-factorization-v2-20260927/bin \
  --binary-source-commit 35fdafd --compress-closed-logs \
  --result-root .local/results/startup-factorization-20260927/full12
```

## Full12 1회: 비용 원천과 warmup 효과

아래는 같은 binary의 A 대비 workload별 비율의 기하평균이다. 처리량은 +가 좋고,
latency는 −가 좋다. Tail 비율의 기하평균이지 전체 요청을 합친 p95가 아니다.

| 모델/variant | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | 처리량 승리 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma B | −1.37% | −0.67% | −2.18% | +1.41% | +3.34% | +1.03% | +0.85% | 5/12 |
| Gemma C | −0.46% | +0.22% | +1.05% | +0.24% | +1.38% | +0.44% | +0.40% | 3/12 |
| Gemma D | −0.98% | +1.39% | −0.30% | +0.80% | +1.46% | +0.85% | +0.55% | 3/12 |
| Gemma E | −0.86% | +1.83% | +0.14% | +0.59% | +1.80% | +0.77% | +0.74% | 4/12 |
| Cosmos B | −0.82% | −4.38% | −2.21% | +1.78% | −1.87% | +0.50% | +0.44% | 5/12 |
| Cosmos C | −0.37% | −3.81% | −6.23% | +1.57% | +0.85% | +0.15% | −0.54% | 6/12 |
| Cosmos D | −0.31% | −0.39% | −2.46% | +2.50% | +2.66% | +0.36% | −0.08% | 4/12 |
| Cosmos E | +1.16% | −2.04% | −8.48% | +0.21% | +1.30% | −1.48% | −1.67% | 8/12 |

平均はほぼ parityでも、Gemma shortとmixed、Cosmos balancedで符号/大きさが異なる。
Cosmos Eだけを採用するなどモデル名で切り替える設定にはしない。
全 workloadの全7指標、peak memory、coverage、P/D dispatchは以下に保持する。

- [B/A](../.local/results/startup-factorization-20260927/full12/measured-vs-baseline.md)
- [C/A](../.local/results/startup-factorization-20260927/full12/calibrated-vs-baseline.md)
- [D/A](../.local/results/startup-factorization-20260927/full12/compact-shapes-vs-baseline.md)
- [E/A](../.local/results/startup-factorization-20260927/full12/compact-http-vs-baseline.md)
- [C/B](../.local/results/startup-factorization-20260927/full12/calibrated-vs-measured.md)
- [D/C](../.local/results/startup-factorization-20260927/full12/shapes-vs-calibrated.md)
- [E/D](../.local/results/startup-factorization-20260927/full12/http-vs-shapes.md)
- [Frozen vLLM 포함 절대값 전체 표](../.local/results/startup-factorization-20260927/full12/frozen-comparison.md)

### 무엇이 실제로 달라졌나

1. **비용 교체만으로도 회귀한다.** Gemma balanced A→B: P dispatch37→40,
   D294→321, D mean BS18.286→16.748. D1은18→45회이지만 대부분 마지막 1/4에
   몰린다(17→43). 전 구간에서 작은 배치만 썼다는 설명이 아니라 drain/tail fragmentation이다.
   처리한 P6035/D5376 tokens는 같다.
2. **Cosmos balanced는 다른 경로다.** A→B P181→206, mean P1.86→1.63인 반면
   D476→474, mean D51.83→52.05다. D 비용 변경이 P/D 선택 timing과 P formation을
   바꾼다는 증거이며 모든 회귀를 D microbatch로 설명할 수 없다. 같은 snapshot forced
   replay가 아니므로 정확한 한 decision의 인과관계까지 확정하지 않는다.
3. **추가 관측은 일부 cohort를 회복한다.** Cosmos multi-image B→C에서 D42→31,
   mean D3.69→5, 처리량277.43→312.09 tok/s. A는 D32/mean4.84/310.79 tok/s다.
   단 5요청 trace이므로 이것 하나로 full-suite 효과를 일반화하지 않는다.
4. **P packing이 좋아지는 방향도 있다.** Gemma long-prefill A→C P227→214,
   mean P1.99→2.11, 처리량599.28→610.16 tok/s. D296→307 증가와 공존한다.
   큰 D 배치만이 목표가 아니며 전체 trajectory를 봐야 한다.
5. **Host 비용도 늘 수 있다.** Gemma balanced scheduler decision mean은
   A66.89/B91.45/C110.14µs, Cosmos는166.8/208.6/233.7µs다. 측정형 lookup/선택 경로의
   overhead를 무시할 수 없지만 이것만으로 전체 latency 차이를 설명하지 않는다.
6. **처리량 parity는 tail parity가 아니다.** Gemma wave/drain C/A는 tok/s−0.16%지만
   TTFT mean+7.82%, E2E p95+8.91%다. Arrival-limited workload의 elapsed throughput만
   보고 승격하면 안 된다.

Gemma C/balanced의 startup D median은 BS1/4/8/16/24에서 약5.97/6.53/12.92/12.86/
13.11ms다. 기존 flat table과 다른 비용 곡선이 GPU drain-cost 최소화의 선택을 바꿀 수 있다.
이는 kernel/tactic 원인까지 확인한 결과는 아니다. `selectDecodeBatchSize()`의 목적은
예측 GPU drain cost이며 request E2E/후속 P formation 전체가 아니다. 더 정확한 isolated
cost와 더 좋은 serving policy를 동치로 두면 안 된다는 것이 이번 결과의 핵심이다.

### vLLM 대비 해석

A는 이번 full12 단회에서도 두 모델 모두 처리량12/12 우세다. C는 Gemma12/12,
Cosmos11/12이며 Cosmos balanced가4180.26 대4315.77 tok/s(−3.14%)다.
Gemma D/E는 vision-heavy에서 frozen vLLM보다 각각−0.10%/−0.36%다.
그렇다고 A가 모든 latency에서도 우세한 것은 아니다. Gemma multi-image A의
TTFT mean344.35 대188.86ms, E2E p95 1453.72 대1300.30ms는 여전히 열세다.
Frozen baseline 비교는 fresh paired 반복이나 CI가 아니며 전체 성능 승리를 주장하지 않는다.

## 다음 결정

기존 champion은 유지한다. Startup coverage 자동화와 chunk-aware probe 수정은 유효하지만,
실측 D 비용의 무조건 적용/HTTP warmup 삭제/모든 모델 자동 최적화 완료를 선언할 근거는 없다.
다음 개선은 workload별 preset 추가가 아니라 **동일 ready work에 대한 D 비용 선택이
후속 P formation과 drain tail에 미치는 영향**을 bounded transition에서 다루는 것이다.
그 전에 추가 반복과 학습 오차 관측으로 이번 단회 차이의 재현성을 확인한다.

## 준비 시간과 반복 확인

Startup JSON의 post-load preparation median은 다음과 같다. Engine load 이전 시간과
그 뒤 HTTP warmup 시간을 포함한 process-to-serving-ready 전체 시간이 아니다.
특히 D/E 차이에 HTTP warmup 감소 시간이 나타나지 않는 것이 정상이다.

| 모델 | C 기존 shape | D compact shape | E compact HTTP | C→D 준비 감소 |
|---|---:|---:|---:|---:|
| Gemma | 19.524s | 18.770s | 18.777s | 0.754s |
| Cosmos | 21.474s | 16.649s | 16.645s | 4.825s |

72개 startup-enabled primary cells 모두 planned frontier coverage를 충족했다.
Gemma probe requests115, Cosmos187이며 HTTP warmup 요청과 별개다.
E는 HTTP49/239→1을 줄이지만 startup GPU 학습을 없애는 zero-start가 아니다.

Memory peak는 B–E 전체에서 Gemma9353–9403MiB, Cosmos9553–9857MiB였다.
C/A paired 차이는 Gemma−4~+4MiB, Cosmos−24~+2MiB로 이번 주 회귀를 KV 용량 변화로
설명할 수 없다. E의 최소값 감소는 보정 경로 차이와 측정 peak의 결과이지 pool 크기를
줄인 결과가 아니다. nvidia-smi 표본 peak이며 순간 allocator peak를 완전히 포착한다고
주장하지 않는다. OOM은 없었지만 이것만으로 모든 입력 shape의 headroom을 보장하지 않는다.

### Balanced 3회

Primary1 + 추가2회. Throughput/p95는 run median, latency mean은 run means의
산술평균이다. Frozen vLLM 반복을 새로 수행하지 않았으며 CI를 주장하지 않는다.

| 모델/variant | tok/s (min–max) | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|
| Gemma A | 1267.13 (1264.84–1270.44) | 84.09 / 207.45 | 15.45 / 16.80 | 1371.27 / 2128.14 |
| Gemma B | 1238.67 (1236.43–1256.28) | 80.64 / 211.76 | 15.60 / 17.01 | 1378.95 / 2137.90 |
| Gemma C | 1250.37 (1248.10–1250.96) | 84.85 / 212.59 | 15.67 / 17.01 | 1387.80 / 2153.56 |
| Cosmos A | 4405.26 (4391.25–4444.31) | 57.82 / 156.15 | 12.68 / 14.22 | 1133.62 / 1784.79 |
| Cosmos B | 4161.77 (4125.77–4173.11) | 54.96 / 147.70 | 13.58 / 15.20 | 1210.08 / 1912.02 |
| Cosmos C | 4215.24 (4180.26–4262.94) | 54.33 / 149.69 | 13.33 / 15.07 | 1187.56 / 1864.64 |

Cosmos B/C는 3회 모두 A의 최저값보다 낮다. 추가 probe가 B보다 일부 회복하지만
A를 회복하지 못한다. C는 frozen vLLM4315.77 대비−2.33%, A는+2.07%다.
Gemma C도 A보다 처리량이 약1.32% 낮아 단회 우연으로만 설명하기 어렵다.
두 모델 모두 TTFT와 TPOT/E2E가 다르게 움직이므로 startup coverage만으로 승격하지 않는다.

3회 dispatch 수 범위도 같은 방향이다. Gemma A의 P37/D293–294에 비해 B는
P40–41/D310–321, C는 P42–45/D314–316이다. Cosmos A의 P181–186에 비해 B는
P199–221, C는 P194–204지만 D는 세 구성 모두474–484 범위에 겹친다.
즉 Gemma의 P/D fragmentation과 Cosmos의 P formation 악화는 추가 반복에서도 관찰됐다.

### Gemma short 3회: 첫 개선 해석 철회

| Variant | tok/s median (min–max) | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|
| A | 845.26 (766.83–871.04) | 95.56 / 221.05 | 21.55 / 28.20 | 524.22 / 826.16 |
| C | 842.58 (838.05–850.25) | 92.00 / 220.03 | 21.34 / 27.92 | 514.57 / 827.86 |

첫 pair의 C/A+9.88%는 3회 median에서−0.32%로 바뀌었다. A의 첫 run이 유난히
느렸으며 추가 두 run은 더 빨랐다. C의 관측 범위가 좁기는 하지만 3회로 분산 감소나
학습 안정성을 입증하지 않는다. 따라서 'startup 관측으로 short가 약10% 빨라졌다'는
주장을 하지 않는다. TTFT/E2E mean 개선과 E2E p95 소폭 악화도 함께 보존한다.

### Gemma text-heavy 3회: shape warmup 축소

| Variant | tok/s median (min–max) | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|
| C | 894.70 (873.06–900.43) | 171.65 / 428.08 | 21.34 / 26.28 | 1285.33 / 1607.42 |
| D | 893.31 (835.33–900.52) | 185.23 / 488.72 | 21.65 / 25.73 | 1313.54 / 1655.19 |

첫 D/C throughput−6.63%도 3회 median에서는−0.15%로 축소됐다. 반면 TTFT mean/p95와
E2E mean/p95는 더 크므로 'shape warmup 축소가 모든 지표에서 무해하다'는 결론도 아니다.
3회로 noisy live trajectory의 분포를 충분히 추정하지 못한다는 점을 유지한다.

성능 자료는 140cells(주120+추가20)를 binary/engine/runner/trace/command identity 검증 후
[통합 비교표](../.local/results/startup-factorization-20260927/merged-comparison.md)에 합쳤다.
Repeat가 겹치는 cell만 3회이며 나머지는 1회다. Full telemetry 학습 진단은 command와
overhead가 다르므로 이 성능 집계에 합치지 않는다.

## 학습 진단: coverage와 학습 품질은 달랐다

`learning-diagnostic/`에서 두 모델 mixed × A/C/E 총6cells를 별도로 실행했다.
Full telemetry의 실행/직렬화 overhead가 다르므로 이 수치를 primary 성능에 합치지 않는다.

### 먼저 수정한 분석 계약

현재 backend는 drain할 때 기록을 직렬화하며, 당시의 contextual 누적 counters를 과거
dispatch 행에 붙인다. 따라서 이 행들에 dispatch index bin을 씌우면 모든 신규 관측이
첫16 decisions에 생긴 것처럼 잘못 보인다. **이 로그로 학습의 시간 경과나 수렴 시점을
측정할 수 없다.** 다음과 같이 처리했다.

- `--epoch-summary`: calibration 직전/serving 종료의 누적 차이만 계산한다.
- elapsed-to-stability와 request-frontier-to-stability를 만들지 않는다. Stability는 unknown이다.
- 누적 snapshot 반복을 발견하면 기본 curve 분석을 거부한다. 이 검사는 명백한 오해를
  차단하는 방어선이지, 모든 과거 로그가 per-dispatch snapshot임을 증명하는 검사가 아니다.
- compact telemetry는 오차 counters가 없으므로 학습 오차 분석을 거부한다.
- 관측0은 RMSE0이나 fully stable이 아니라 unknown이다.

이번 6개는 calibration-end 응답의 방향별 관측수 합과 epoch baseline을 대조했다.
Errors는 업데이트 **직전** 예측과 실제 선택된 overlap label의 차이이며 independently
held-out state나 선택하지 않은 action의 counterfactual error가 아니다. Reward 단위는
normalized compression이다. 아래 false-safe는 모델 LCB>0인데 관측 reward≤0인 경우이며
메모리 correctness 실패나 request SLO 위반 횟수가 아니다.

| 모델/variant | warmup 관측 PD/EP/ED | serving 관측 PD/EP/ED | serving RMSE PD/EP/ED | false-safe / predicted-safe PD; EP; ED |
|---|---:|---:|---:|---|
| Gemma A | 107 / 3 / 0 | 38 / 1 / 5 | .233 / .275 / .272 | 3/33; 0/0; 2/3 |
| Gemma C | 181 / 4 / 0 | 45 / 0 / 5 | .246 / unknown / .340 | 4/35; 0/0; 1/1 |
| Gemma E | 103 / 0 / 0 | 36 / 5 / 4 | .347 / .310 / .387 | 1/20; 1/2; 1/2 |
| Cosmos A | 455 / 11 / 7 | 23 / 0 / 7 | .246 / unknown / .113 | 0/19; 0/0; 0/0 |
| Cosmos C | 528 / 10 / 7 | 15 / 0 / 8 | .360 / unknown / .041 | 0/6; 0/0; 0/8 |
| Cosmos E | 206 / 0 / 0 | 12 / 4 / 4 | .453 / .287 / .184 | 1/6; 0/0; 0/1 |

0/0은 false-safe가 없었다는 성능 보장이 아니라 positive-LCB 평가 표본이 없다는 뜻이다.
각 variant는 선택하는 action/state가 다르므로 이 RMSE 표를 같은 held-out set에서의
predictor 순위로 해석하지 않는다. Full telemetry 각 cell의 `adaptation.json`에 raw sums,
epoch baseline, direction별 counts를 보존했다.

### 해석

1. C는 두 모델의 PD warmup 표본을 늘리지만 serving PD RMSE를 낮추지는 않았다.
   단순 표본수 부족만으로 B/C 회귀를 설명할 수 없다. Warmup distribution과 serving
   distribution, 그리고 selected-action bias를 분리해야 한다.
2. E에서는 두 모델 모두 warmup EP/ED accepted observations가0이다. Startup에서
   E isolated/exact key coverage가 성공해도 E contextual RLS authority가 준비됐다는
   뜻이 아니다. E+D probe 실행 수와 accepted RLS label 수 역시 같은 지표가 아니다.
3. Gemma E PD warmup RMSE는 약.047로 A의.204보다 낮지만 serving은.347 대.233이다.
   쉬운 준비 표본에 잘 맞는 것을 실제 서빙 일반화나 학습 수렴으로 착각하면 안 된다.
4. Cosmos C의 ED serving RMSE.041은 유망하지만8표본/단일 mixed run이다.
   E 방향 전체나 full12 안정성을 입증하지 않는다.
5. Warmup 축소로 time-to-ready를 줄이는 기술적 가능성과 그 뒤 policy 안정성은 별개다.
   이번 결과로 `policy_stability_validated=true`를 만들지 않는다.

## 최종 검증과 판정

- C++ 최종707tests: **705 passed, 2 optional skipped**, failures0.
  `.local/results/startup-factorization-20260927/final-runtime-tests.xml`.
- Python 계약/보고서/학습 분석 tests: **52 passed**.
- Primary120 + targeted repeat20 + learning6: **146 successful HTTP cells**.
  Coverage 수정 확인 smoke1은 별도이며, 초기 실패 smoke2는 실패 diagnostic으로 보존한다.
- 모델/engine/KV 변경 없음. 기존 엔진의 export/build lineage를 재사용했으며 이번 검증은
  runtime startup 및 inference 변경 검증이다. 새 모델 export 성공을 주장하지 않는다.
- 두 모델의 default pointer는 `.local/baselines/throughput-9db3ed3-20260926/bin` 유지.
  자동 보정 variant는 **승격하지 않는다**. Unchanged frozen vLLM 비교표는 통합 report에 있다.

이번 작업은 비용 원천 전환, 추가 물리 관측, 두 종류 warmup 축소를 분리하고 실제 두 모델
full12에서 검증한 것이다. '모델 로드 전에 잠깐 측정하면 모든 정책/메모리 설정이 자동으로
최적화된다'는 목표를 완료한 것은 아니다.

다음 우선순위는 세 가지다.

1. 실측 D-cost를 사용하는 decision이 P formation/drain tail까지 보는지 동일 snapshot으로
   검증한다. GPU drain-cost 단독 목표와 V3 transition 목표의 불일치를 먼저 다룬다.
2. 학습 curves가 필요하면 observation 시점에 작은 counter snapshot을 저장하고 drain 후
   출력한다. 현재와 같은 종료 counters를 dispatch 시각에 소급 연결하지 않는다.
3. E readiness는 isolated coverage와 accepted contextual evidence를 별도 gate로 둔다.
   고정 HTTP warmup을 없애려면 모델명 preset이 아니라 generic held-out shape/방향에서
   uncertainty와 prediction quality를 확인해야 한다. 그 이후에만 production create/readiness에
   연결하고 full12 반복 promotion을 수행한다.
