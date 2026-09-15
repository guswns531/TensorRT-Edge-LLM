# Small E batches and progressive overlap

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

## Question and contract

Can E1/E2, submitted progressively alongside existing P/D work, improve request progress over forming E4?
This is distinct from note303's rejected D-priority rule and from changing vision prefill chunk size.

The physical E4 engine, P/D capabilities, KV pool, model precision, HTTP traces, generic calibration and
V3 action authority remain unchanged. The new preparation limit is activated only after calibration drains.
`lifetime` retains the existing E4 frontier, `e1` prepares at most one logical request and `e2` at most two.
E1 is a logical request batch, not necessarily one image or one crop. No request or output work is removed.
Gemma retains chunk128 vision P; Cosmos retains atomic vision P.

## Why preparation is the first boundary

Asynchronous encoder preparation packs one selected batch before the global dispatch selector sees it.
Splitting an already prepared E4 into arbitrary E1 views would require different shape/ownership mechanisms.
`encoderPreparationBatchLimit` instead constrains the existing FIFO/geometry/byte-feasible prefix before
preparation, preserving the oldest request and every existing memory/dependency guard. Its maximum remains
bounded by `maxEncoderBatchSize`; zero restores the full physical frontier. A full preparation quantum does
not wait for the unused physical batch slots to fill. Final dispatch is still authorized by the global selector.

These E1/E2 controls are **fixed diagnostic policies, not a completed automatic joint shape/action selector**.
They first establish whether exposing smaller E shapes creates useful real overlap under the existing policy.
No extra overlap probes or workload-specific warmup are added. An insufficient E cost posterior can still
make the selector conservative; no-overlap results must be reported rather than counted as a capability failure.

## Execution plan

1. Build one binary; CPU contracts and scheduling/memory tests.
2. Same-model paired HTTP screen: E4 frontier / E1 / E2, mixed, vision-heavy and multi-image on both models.
3. Inspect measurement-only E histograms, E/P/D/C masks, E/P and E/D overlap, D cohort sizes, queue delays,
   mean/p95 TTFT/TPOT/E2E and throughput. Verify request counts, token agreement and final retained storage.
4. Repeat useful comparisons in alternating order; do not select one noisy best run as the policy winner.
5. Only then promote bounded pre-preparation shape proposals into an online selector. Compare full/half/one
   using the same remaining encoder work and successor P/D cohorts; immediate pair compression alone is not
   sufficient. Preserve unknown-cost fallback and final dispatch authority; preparation is not GPU overlap.

Use `.local/results/small-encoder-overlap-20260914/` for manifests, identities, patches, summaries and raw evidence.
Closed manifest-listed gateway logs may be losslessly gzip-compressed there; no engines, models or worktrees
may be deleted. Frozen vLLM from note302 is reusable as an unchanged external-contract reference, not fresh
equal-memory evidence. The first screen is not full12, a causal repeated-snapshot experiment, or a sanitizer gate.

## 구현 경계와 동작 방식

이번 변경은 엔진을 작게 다시 만드는 것이 아니다. 같은 E4 엔진에 보내기 전 **논리 요청 prefix의 준비 단위**만
1/2로 제한한다. Multi-image 요청 하나는 여러 이미지/crop을 포함할 수 있다. E1을 image1 또는 crop1과 혼동하지 않는다.

```text
E-ready FIFO / geometry / byte capacity
                   │
                   ▼
preparation prefix: full / at most 2 / at most 1
                   │  request IDs와 ownership 확정
                   ▼
asynchronous preparation ─────── 그동안 허용된 P/D 실행
                   │
                   ▼
prepared E batch + current P/D ready/outstanding
                   │
                   ▼
기존 Global V3 selector ─────── E / E+P / E+D / P / D / WAIT
                   │
                   ▼
E 완료 → 해당 요청 P-ready → P 완료 → D-ready
```

- `cpp/runtime/scheduling/phaseThreeCoordinator.h`: `encoderPreparationBatchLimit`와 drained-boundary setter.
- `phaseThreeCoordinator.cpp`: `nextEncoderBatchIndices()`의 capacity prefix와 batch-full 판정에 준비 단위 적용.
  0은 원래 full frontier이며 physical max보다 큰 값은 거부한다. 실행/요청 lease가 남아 있으면 setter도 거부한다.
- `examples/llm/llm_phase_context_smoke.cpp`: 공통 calibration이 완전히 끝난 뒤
  `TRT_EDGELLM_MEASUREMENT_ENCODER_PREPARATION_ROWS=1/2`를 적용한다.
- `benchmarks/phase_serving/run_lifetime_encoded_admission.py`: `lifetime/e1/e2`는 같은 lifetime admission,
  physical E4, P/D/KV/HTTP 계약을 사용한다. e1/e2의 차이는 위 measurement 환경변수뿐이다.
- `analyze_lifetime_encoded_admission.py`: E GPU 합계, 실행 histogram, E&P/E&D epoch 비율과 E-active 조건부 overlap을
  별도로 기록한다. `analyze_decode_service_admission.py`는 완료 관측→sampling 관측→collect→commit을 분리한다.

핵심 제약은 그대로다. E execution context에는 하나만 inflight이며, 같은 요청의 P는 E 완료 전 실행하지 않는다.
E1 실행 중 다른 요청의 P/D와 겹칠 수 있지만 E1 여러 개를 같은 E context에 동시에 실행하는 구조는 아니다.
E1은 E 내부 layer/crop을 중간에 끊는 preemption도 아니다. 각 E1이 완료되면 다른 요청보다 먼저 P/D로 전진할 수 있다.

이 비교는 순수한 GPU overlap 인과 실험이 아니다. 준비 단위를 줄이면 queue 소비 순서와 batch-full 시점도 바뀐다.
예를 들어 E1은 한 요청이 있으면 full quantum이므로, 비어 있는 E4 슬롯을 채우려고 formation wait를 적용하지 않는다.
기존 25ms 최대 wait 설정 자체를 바꾸지는 않았지만 그 설정이 발동하는 상태는 달라진다. 성능 차이를 전부 overlap
compression으로 돌리지 않고 E dispatch 증가, 실제 service, downstream cohort, host completion을 함께 해석해야 한다.

## 동적 선택으로 이어갈 때의 설계

이번 결과로 `모델이 Cosmos이면 E1` 같은 분기를 넣지 않는다. 고정 E1도 최종 답이 아니다. 필요한 선택 위치는
GPU enqueue 직전만이 아니라 **asynchronous preparation 직전**이다. 이미 준비된 E4를 나누는 방식은 binding과
공유 payload의 lifetime을 추가로 바꿔야 하므로 이번 단계에서는 하지 않는다.

1. 같은 FIFO/geometry/byte-feasible snapshot에서 full/half/one prefix를 최대 3개 만든다. 남은 요청은 제거하지 않는다.
2. 각 후보를 동일한 유한 ready-work frontier에 대해 비교한다. E1 하나와 E4 하나의 시간만 비교하면 처리한 요청 수가
   달라 불공정하다. 남은 E 요청의 후속 dispatch, P-ready 시점, D cohort와 completion visibility 비용을 포함해야 한다.
3. 현재 관측된 E service/cost ratio, 준비 비용, pending sampling, P/D ready mass를 사용한다. 미래 도착을 예언하거나
   workload 이름을 쓰지 않는다. E1의 최초 응답 이득과 resident D/text 지연을 분리해서 평가한다.
4. 준비 전에 prefix를 결정해 ownership을 reserve한다. 준비 중 상태가 바뀔 수 있으므로 GPU enqueue 때 기존 Global
   selector가 최신 outstanding/feasibility로 다시 승인한다. 준비 선택을 overlap 실행 약속으로 취급하지 않는다.
5. 먼저 shadow로 full/half/one의 제안과 equal-work 예측 오차를 기록한다. 비용 근거가 부족하면 기존 full frontier를 유지한다.
   이 fallback은 E1을 지원하지 않는다는 뜻이 아니다. 새 모델별 threshold, 명시적 SLO, 별도 E/D 우선권은 추가하지 않는다.
6. 두 모델의 mixed/heavy/multi-image에서 TTFT·TPOT·E2E trade-off를 확인한 뒤 full12로 확대한다. 같은 출력/엔진/
   calibration 계약, request-level token agreement, 실제 overlap action fidelity를 별도 gate로 유지한다.

이것은 다음 구현 계획이며 이번 고정 quantum 비교만으로 검증된 online joint selector라고 주장하지 않는다.

## 실험 식별과 검증 범위

- Source: `codex/v0101-phase-forward-port`, HEAD `a516dcdc194b44ea7cddd2ae08c84f1f1b2d4ca3` + dirty patch.
- Binary SHA256: `84c396de771b4aaeb83e80bc070fc0310f20389ce1cddccd002c5f361afac66a`.
- Tracked patch SHA256: `e41bf6e51e4c7d1a2c8a00a777c1ff559fa7628b76555c12e2fa30c690944dca`.
- 각 campaign의 `manifest.json`, `source.patch`, `runner-source.py`에 Docker digest, plugin/binary/engine/config/
  calibration hash, cell별 명령을 저장했다. 분석 도구의 후속 CPU-only 변경은 실행 바이너리를 바꾸지 않았다.
- Gemma: INT4 AWQ, physical E4, text P8 / vision P4, D24, slot24, FP16 KV96 pages, vision P chunk128.
- Cosmos: FP16, physical E4, text P8 / vision P4, D64, slot80, FP16 KV256 pages, vision P atomic.
- 각 cell은 새 프로세스로 같은 모델별 generic calibration recipe를 실행한다. Gemma49 / Cosmos239 requests.
  같은 calibration 입력/설정이지, 독립 실행의 CUDA 측정값과 RLS posterior가 bitwise 동일하다는 뜻은 아니다.
- 같은 모델의 세 variant는 같은 no-explicit-SLO V3, lifetime byte admission, HTTP trace, output contract,
  P/D capabilities와 KV를 사용한다. 모델 간 precision/trace 토큰/engine shape까지 같다는 비교는 아니다.
- Mixed/heavy는 각각 64 requests. Multi-image는 Gemma20 / Cosmos5 requests로, 특히 Cosmos 결과는 작은 trace다.
  12-workload 전체 재검증이 아니며 sustained arrival/load sweep, fresh vLLM 반복도 아니다.
- Test: C++ scheduling/memory/runtime suite 261 passed; Python benchmark contracts 13 passed; relevant pre-commit
  및 `git diff --check` 통과. 기존 엔진을 사용하는 runtime-only 변경이며 새로운 모델 export/build claim은 없다.

재현 명령:

```bash
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma cosmos --workloads mixed vision-heavy multi-image \
  --variants lifetime e1 e2 --repeats 1 --compress-closed-logs \
  --result-root .local/results/small-encoder-overlap-20260914/screen
python3 benchmarks/phase_serving/run_lifetime_encoded_admission.py \
  --models gemma cosmos --workloads mixed vision-heavy multi-image \
  --variants e2 e1 lifetime --repeats 2 --compress-closed-logs \
  --result-root .local/results/small-encoder-overlap-20260914/confirmation
```

두 번째 campaign은 variant 순서를 반대로 시작하고 그 다음 repeat에서 다시 반전한다. 두 campaign의 독립된
repeat-001을 덮어쓰지 않고 source root를 유지하여 총 3회로 집계한다. 3회는 방향 확인용이며 CI/significance를
확정하는 충분한 반복이라고 주장하지 않는다. 지연 p95는 각 run의 request p95를 낸 뒤 run 간 평균한다.
처리량 역시 run별 전체 생성 token/s의 산술평균이다. 표의 전체 지연은 client send 기준이며 arrival-relative는
별도로 기록한다. E/P/D/C mask는 stream 작업 구간이지 SM occupancy/DRAM utilization이 아니다.

```bash
python3 benchmarks/phase_serving/analyze_lifetime_encoded_admission.py \
  --result-root .local/results/small-encoder-overlap-20260914/confirmation
python3 benchmarks/phase_serving/analyze_lifetime_encoded_admission.py \
  --result-root .local/results/small-encoder-overlap-20260914/screen \
  --supplement-result-root .local/results/small-encoder-overlap-20260914/confirmation
python3 benchmarks/phase_serving/analyze_decode_service_admission.py \
  --result-root .local/results/small-encoder-overlap-20260914/screen
python3 benchmarks/phase_serving/analyze_decode_service_admission.py \
  --result-root .local/results/small-encoder-overlap-20260914/confirmation
```

`screen/comparison.json`이 세 repeat의 serving 집계와 baseline 대비 delta를 담는다. 각 root의 `analysis.json`은
cell별 E histogram/mask/메모리/arrival-relative latency, `decode-service-analysis.json`은 request별 cycle 분해다.

## 최종 결과: 54 cells, 각 조합 3회

18-cell screen + 36-cell confirmation 모두 완료, 실패 0. 측정 HTTP 요청 **2,529개**, 고정 출력 토큰
**104,256개**가 완료됐다. Calibration 요청은 이 수에 포함하지 않는다. 작은 E로 점진적 진행/실제 overlap이
가능하다는 것은 확인했지만, 두 모델 모든 workload를 개선하는 고정 quantum은 찾지 못했다. 기본값은 변경하지 않는다.

이하 `Full`은 기존 lifetime admission + 최대 E4 frontier이며 모든 dispatch가 E4라는 뜻은 아니다.
E1/E2도 같은 lifetime admission이다. 처리량은 token/s, 지연은 ms, `평균 / p95` 표기다.

### 전체 serving 지표

| 모델 / workload | 준비 단위 | token/s | TTFT 평균 / p95 | TPOT 평균 / p95 | E2E 평균 / p95 | 평균 peak MiB |
|---|---|---:|---:|---:|---:|---:|
| Gemma mixed | Full | 718.54 | 253.18 / 571.55 | 26.36 / 35.19 | 1459.73 / 2222.58 | 9383.67 |
| Gemma mixed | E1 | 700.15 | 260.05 / 626.43 | 26.71 / 35.47 | 1473.56 / 2277.82 | 9379.00 |
| Gemma mixed | E2 | 730.44 | 252.26 / 554.64 | 25.80 / 34.64 | 1428.43 / 2148.61 | 9383.67 |
| Gemma heavy | Full | 546.17 | 379.28 / 715.58 | 31.51 / 44.72 | 1566.24 / 2197.77 | 9389.00 |
| Gemma heavy | E1 | 537.33 | 402.74 / 837.20 | 32.36 / 46.71 | 1618.91 / 2224.23 | 9383.00 |
| Gemma heavy | E2 | 549.06 | 351.74 / 666.12 | 32.21 / 47.53 | 1561.97 / 2231.82 | 9387.00 |
| Gemma multi-image | Full | 382.75 | 418.11 / 682.21 | 25.32 / 40.68 | 1203.12 / 1486.72 | 9389.67 |
| Gemma multi-image | E1 | 364.98 | 428.13 / 714.61 | 26.44 / 38.96 | 1247.66 / 1492.03 | 9379.67 |
| Gemma multi-image | E2 | 375.03 | 426.78 / 742.93 | 26.04 / 41.04 | 1233.99 / 1498.11 | 9381.00 |
| Cosmos mixed | Full | 1173.31 | 811.93 / 1993.51 | 33.10 / 61.06 | 2355.32 / 2477.39 | 9756.33 |
| Cosmos mixed | E1 | 1140.64 | 648.59 / 1931.00 | 39.98 / 63.28 | 2414.04 / 2547.86 | 9598.33 |
| Cosmos mixed | E2 | 1144.19 | 759.32 / 2028.75 | 36.19 / 62.80 | 2405.86 / 2531.23 | 9663.00 |
| Cosmos heavy | Full | 724.71 | 1512.10 / 2991.89 | 42.52 / 81.77 | 3173.82 / 3337.90 | 9832.33 |
| Cosmos heavy | E1 | 717.15 | 1236.15 / 3091.60 | 51.18 / 81.63 | 3169.33 / 3383.36 | 9600.33 |
| Cosmos heavy | E2 | 722.52 | 1522.13 / 3039.81 | 42.93 / 81.65 | 3192.73 / 3362.06 | 9815.00 |
| Cosmos multi-image | Full | 299.26 | 276.90 / 321.12 | 8.12 / 9.98 | 528.47 / 534.26 | 9598.33 |
| Cosmos multi-image | E1 | 310.08 | 197.56 / 294.10 | 10.10 / 13.20 | 510.56 / 514.31 | 9601.00 |
| Cosmos multi-image | E2 | 309.57 | 238.11 / 299.87 | 8.81 / 11.73 | 511.29 / 516.16 | 9601.00 |

### Full 대비 변화율

모든 지연 열은 음수가 개선, 처리량은 양수가 개선이다. Pareto winner가 없는 행에서 지표 하나로 승격하지 않는다.

| 모델 / workload | 단위 | token/s | TTFT 평균 / p95 | TPOT 평균 / p95 | E2E 평균 / p95 |
|---|---|---:|---:|---:|---:|
| Gemma mixed | E1 | −2.56% | +2.71 / +9.60% | +1.34 / +0.80% | +0.95 / +2.49% |
| Gemma mixed | E2 | +1.66% | −0.37 / −2.96% | −2.12 / −1.57% | −2.14 / −3.33% |
| Gemma heavy | E1 | −1.62% | +6.18 / +17.00% | +2.71 / +4.43% | +3.36 / +1.20% |
| Gemma heavy | E2 | +0.53% | −7.26 / −6.91% | +2.23 / +6.28% | −0.27 / +1.55% |
| Gemma multi-image | E1 | −4.64% | +2.40 / +4.75% | +4.40 / −4.23% | +3.70 / +0.36% |
| Gemma multi-image | E2 | −2.02% | +2.07 / +8.90% | +2.83 / +0.88% | +2.57 / +0.77% |
| Cosmos mixed | E1 | −2.78% | −20.12 / −3.14% | +20.78 / +3.64% | +2.49 / +2.84% |
| Cosmos mixed | E2 | −2.48% | −6.48 / +1.77% | +9.35 / +2.84% | +2.15 / +2.17% |
| Cosmos heavy | E1 | −1.04% | −18.25 / +3.33% | +20.37 / −0.17% | −0.14 / +1.36% |
| Cosmos heavy | E2 | −0.30% | +0.66 / +1.60% | +0.96 / −0.15% | +0.60 / +0.72% |
| Cosmos multi-image | E1 | +3.61% | −28.65 / −8.41% | +24.42 / +32.27% | −3.39 / −3.74% |
| Cosmos multi-image | E2 | +3.45% | −14.01 / −6.61% | +8.59 / +17.55% | −3.25 / −3.39% |

3회 처리량 min–max도 보존한다. Gemma heavy/multi-image의 분산이 크므로 +0.53% 같은 차이는 뚜렷한 개선이라고
판정하지 않는다. Gemma mixed E2는 세 실행 모두 Full 범위보다 높지만 token identity gate가 별도로 남아 있다.

| 모델 / workload | Full min–max | E1 min–max | E2 min–max |
|---|---:|---:|---:|
| Gemma mixed | 717.41–719.31 | 695.86–706.02 | 727.91–734.72 |
| Gemma heavy | 532.93–553.87 | 515.49–553.62 | 542.52–561.03 |
| Gemma multi-image | 374.11–392.88 | 361.54–368.36 | 356.73–393.88 |
| Cosmos mixed | 1167.16–1180.24 | 1136.84–1144.80 | 1138.81–1154.17 |
| Cosmos heavy | 722.09–729.77 | 713.60–723.06 | 716.93–730.02 |
| Cosmos multi-image | 296.10–301.48 | 308.74–310.80 | 308.30–311.38 |

## 실제 overlap과 formation

E histogram은 **3회 실행 합계**, E GPU 합과 D count/BS는 **run별 값의 평균**이다.
`E∩(P∪D)/E`는 E가 작업 중인 시간 중 P 또는 D도 작업 중인 비율이다. EP/ED는 전체 measurement epoch가 분모다.
EP와 ED는 triple 구간이 있으면 중복될 수 있으므로 두 비율을 더해 조건부 비율로 쓰지 않는다.

| 모델 / workload | 단위 | E histogram 합계 | E GPU 합 ms | E 겹침 조건부 % | EP / ED epoch % | D count / 평균 BS |
|---|---|---|---:|---:|---:|---:|
| Gemma mixed | Full | E1×30 E2×15 E3×8 E4×3 | 619.15 | 73.10 | 6.03 / 5.09 | 149.33 / 19.18 |
| Gemma mixed | E1 | E1×96 | 643.72 | 52.60 | 3.91 / 4.22 | 160.00 / 17.90 |
| Gemma mixed | E2 | E1×24 E2×36 | 610.22 | 55.62 | 2.87 / 5.63 | 152.33 / 18.80 |
| Gemma heavy | Full | E1×33 E2×10 E3×17 E4×10 | 896.59 | 54.09 | 4.00 / 6.75 | 153.33 / 15.66 |
| Gemma heavy | E1 | E1×144 | 999.40 | 59.29 | 7.72 / 5.32 | 149.67 / 16.05 |
| Gemma heavy | E2 | E1×22 E2×61 | 913.64 | 59.77 | 6.10 / 6.08 | 152.00 / 15.80 |
| Gemma multi-image | Full | E1×3 E2×11 E3×9 E4×2 | 377.55 | 59.58 | 13.48 / 0.10 | 48.67 / 12.74 |
| Gemma multi-image | E1 | E1×60 | 402.51 | 40.04 | 6.40 / 2.90 | 58.33 / 10.66 |
| Gemma multi-image | E2 | E1×12 E2×24 | 380.95 | 50.48 | 8.04 / 3.30 | 54.67 / 11.37 |
| Cosmos mixed | Full | E2×4 E3×16 E4×10 | 1021.70 | 35.63 | 13.25 / 1.39 | 67.33 / 42.56 |
| Cosmos mixed | E1 | E1×96 | 1157.02 | 35.23 | 14.81 / 1.11 | 68.00 / 42.12 |
| Cosmos mixed | E2 | E2×48 | 1054.19 | 53.33 | 12.30 / 9.72 | 71.33 / 40.17 |
| Cosmos heavy | Full | E1×3 E2×2 E3×31 E4×11 | 1558.28 | 44.44 | 16.77 / 3.74 | 77.67 / 30.96 |
| Cosmos heavy | E1 | E1×144 | 1830.98 | 49.47 | 20.87 / 5.87 | 73.00 / 32.88 |
| Cosmos heavy | E2 | E2×72 | 1594.51 | 32.33 | 14.26 / 0.93 | 73.00 / 32.89 |
| Cosmos multi-image | Full | E1×2 E2×1 E3×1 E4×2 | 163.69 | 40.24 | 13.09 / 0.00 | 33.67 / 4.61 |
| Cosmos multi-image | E1 | E1×15 | 210.52 | 78.06 | 32.55 / 0.00 | 32.33 / 4.79 |
| Cosmos multi-image | E2 | E1×3 E2×6 | 167.79 | 39.23 | 11.78 / 1.22 | 33.33 / 4.65 |

관측과 해석을 구분하면 다음과 같다.

1. **작은 E의 실제 overlap은 가능하다.** E1에서도 E/P와 E/D 작업 구간이 관측된다. 다만 Cosmos multi-image의
   E1 이득은 E+D가 아니라 주로 E+P에서 관측됐다. 요청한 action 비율과 실제 구간 overlap도 구분해야 한다.
2. **겹침 비율만 최대화하면 안 된다.** Cosmos mixed E2의 E 겹침은 35.63→53.33%인데 처리량은 −2.48%다.
   Gemma mixed E2는 겹침이 73.10→55.62%로 줄었는데 전체 지표는 개선됐다.
3. **작은 E는 전체 E 비용을 늘릴 수 있다.** Cosmos heavy E1의 E GPU 합은 1558.28→1830.98ms(+17.5%)다.
   Gemma multi-image는 D dispatch 48.67→58.33, BS12.74→10.66으로 fragmentation도 커졌다.
4. **GPU 일의 합과 request completion도 다르다.** Cosmos multi-image E1은 E GPU 합이 28.6% 늘었지만
   E/P 겹침과 더 빠른 request progression으로 E2E 평균이 3.39% 낮았다. 이는 인과 snapshot replay가 아닌
   end-to-end trajectory 비교이며, 그 중 어떤 항의 독립 기여인지 완전히 분리한 결과는 아니다.
5. 전체 `0000` idle은 cell별 1.76~4.67%다. E1은 Cosmos mixed의 평균 idle을 2.75→4.19%, heavy를
   3.10→4.47%로 늘렸다. Context overlap과 GPU 빈 구간/host 간격은 같은 척도가 아니다.

## 첫 토큰 이득은 어디서 생기고 누가 비용을 내는가

Cosmos의 request class별 3회 평균이다. 앞 표의 전체 TTFT 감소가 text 요청까지 동일하게 좋아졌다는 뜻은 아니다.

| Workload | 단위 | 요청 | TTFT 평균 | TPOT 평균 | E2E 평균 |
|---|---|---|---:|---:|---:|
| Mixed | Full | text | 124.65 | 40.92 | 2427.79 |
| Mixed | E1 | text | 147.30 | 41.77 | 2494.30 |
| Mixed | E2 | text | 148.68 | 41.44 | 2482.28 |
| Mixed | Full | vision | 1499.21 | 25.28 | 2282.85 |
| Mixed | E1 | vision | 1149.88 | 38.19 | 2333.79 |
| Mixed | E2 | vision | 1369.95 | 30.95 | 2329.44 |
| Heavy | Full | text | 68.99 | 58.46 | 3256.00 |
| Heavy | E1 | text | 86.84 | 58.43 | 3284.65 |
| Heavy | E2 | text | 120.44 | 58.03 | 3278.79 |
| Heavy | Full | vision | 1993.14 | 37.20 | 3146.43 |
| Heavy | E1 | vision | 1619.26 | 48.76 | 3130.89 |
| Heavy | E2 | vision | 1989.36 | 37.89 | 3164.05 |

Cosmos mixed E1은 vision TTFT를 23.3% 앞당기는 대신 text TTFT를 18.2%, text E2E를 2.7% 늘렸다.
반면 vision E2E도 2282.85→2333.79ms로 늘었다. 첫 응답을 앞당겼다고 이후 요청 서비스까지 개선되는 것은 아니다.

TPOT 정의에도 주의한다. 고정 출력 N에서 `TPOT=(E2E−TTFT)/(N−1)`이므로, 끝나는 시간이 비슷한 상태에서 첫 토큰만
앞당겨도 TPOT는 커진다. 따라서 Cosmos multi-image의 E1을 TPOT+24.42% 하나로 전면 실패라고 하거나,
TTFT−28.65% 하나로 전면 성공이라고 하지 않는다. 두 사용자 경험이 다른 trade-off다.

특히 **E queue 자체가 빨라진 것은 아니다**. Cosmos mixed E의 평균 queue wait는 Full592.97→E1 933.70ms,
heavy는 913.49→1300.41ms로 증가했다. 작은 E가 뒤쪽 encoder 요청을 더 기다리게 하면서도, 앞서 끝난 요청은
큰 E 묶음 전체를 기다리지 않고 P로 전진할 수 있다. 이 두 현상을 분리해야 한다.

### Decode cycle: CPU commit만의 문제가 아니다

요청마다 token cycle 평균을 먼저 낸 다음 요청 평균, 마지막으로 3회 평균을 사용한다. 단위 ms.
`D start→host done`은 순수 CUDA kernel 시간이 아니고 완료 관측 지연을 포함한다.

| Workload | 단위 | Ready→D start | D start→host done | Done→token commit | 그 중 Done→sampling 관측 | Collect→commit |
|---|---|---:|---:|---:|---:|---:|
| Mixed | Full | 15.964 | 14.571 | 2.581 | 2.517 | 0.063 |
| Mixed | E1 | 10.042 | 16.560 | 13.392 | 13.333 | 0.058 |
| Mixed | E2 | 17.758 | 15.711 | 2.761 | 2.695 | 0.065 |
| Heavy | Full | 19.983 | 17.491 | 5.071 | 4.895 | 0.175 |
| Heavy | E1 | 12.959 | 21.008 | 17.247 | 16.838 | 0.408 |
| Heavy | E2 | 18.942 | 16.885 | 7.260 | 6.584 | 0.675 |
| Multi-image | Full | 0.849 | 7.092 | 0.163 | 0.142 | 0.020 |
| Multi-image | E1 | 3.197 | 6.815 | 0.170 | 0.148 | 0.021 |
| Multi-image | E2 | 1.627 | 7.154 | 0.279 | 0.259 | 0.019 |

Sampling event 관측→collect는 mixed/heavy 약 0.0011~0.0013ms, multi-image 약 0.0005ms다. 큰 손실의 대부분은
collect 함수 뒤가 아니라 **sampling 완료가 host에 관측되기 전**이다. 이것은 GPU sampling 실행, GPU 대기/경합,
CPU poll/다른 dispatch 작업이 섞인 구간이다. GPU event와 polling timeline을 더 연결하기 전에는 sampler kernel
자체가 10ms 느려졌다고 단정하지 않는다. 구현 위치는 `IndependentPhaseAsyncServer::processSamplingTickets()`,
`completeSamplingTicket()`, `processTicket()`이며, 이번에는 관측 분해만 추가하고 이 실행 경로를 수정하지 않았다.

## 메모리와 correctness

- KV page 수/precision/allocator는 세 variant에서 동일하다. E1의 peak VRAM 감소는 KV 축소 실험이 아니다.
- Cosmos mixed peak는 9756.33→9598.33MiB, heavy는 9832.33→9600.33MiB다. 동시에 실제 admission ledger의
  peak retained vision storage 평균은 mixed323.69→36.74MiB, heavy382.29→96.88MiB로 줄었다.
  전체 VRAM 차이는 allocator reuse/workspace/관측 시점도 포함하므로 이 숫자를 단순 합산하지 않는다.
- Gemma peak 변화는 대체로 4~10MiB 수준이다. Gemma payload와 Cosmos unsplit M-RoPE storage lifetime이
  다르므로, E1에서 Cosmos처럼 수백 MiB가 줄어야 정상이라는 주장은 맞지 않는다.
- 모든 54 cells의 마지막 phase metric에서 retained vision bytes=0이었다. 이는 해당 ledger drain 증거이며
  sanitizer 통과나 모든 CUDA allocator의 누수 부재를 대신하지 않는다. 이번에 sanitizer는 실행하지 않았다.
- Cosmos heavy Full의 byte-block 누계는 세 run에서 1449/0/1, E1/E2는 0이다. Poll 횟수 기반 누계라 이를
  blocked-request 수로 읽지 않는다. Full에서 한 run의 memory backpressure가 활성화됐다는 것도 confound로 보존한다.
- Cosmos 27 cells의 request ID별 전체 output token string은 같은 workload의 첫 Full reference와 모두 일치했다.
- Gemma는 exact identity 미통과다. 아래는 첫 Full run 대비 **다른 출력의 request 수**이며 nonreference 반복 순서다.

| Gemma workload | 요청 수/run | Full의 추가 2회 | E1 3회 | E2 3회 |
|---|---:|---|---|---|
| Mixed | 64 | 2, 4 | 6, 7, 7 | 13, 11, 10 |
| Heavy | 64 | 9, 4 | 7, 7, 8 | 11, 9, 10 |
| Multi-image | 20 | 2, 2 | 4, 5, 4 | 2, 5, 2 |

고정 token 수와 HTTP 성공을 semantic/exact correctness와 혼동하지 않는다. 기존 Full 반복에도 차이가 있지만
그 사실만으로 새 quantum 경로의 KV/row/shape 문제가 없다고 결론 내릴 수 없다. Gemma 성능 표는 diagnostic이며,
same-snapshot logits/row/ownership 비교와 deterministic replay를 거치기 전 production 승격하지 않는다.

## Frozen vLLM 비교

새 측정은 하지 않았다. External request/model contract가 유지된 reference를 재사용한다.

- Gemma: `.local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12/`.
  vLLM v0.28, seq24 / KV480MiB / P4096 / graph24, 해당 표는 workload별 기존 1회다.
- Cosmos: `.local/results/v0101-forward-port/v3-service-scale-20260910/vllm-fresh-equal-summary.json`.
  FP16, client64, KV3.5GiB, vision warmup16. Mixed/multi-image 3회 성공, heavy는 3회 시도 중 2회 성공 집계다.
- 엔진/runtime/graph/calibration 및 메모리 footprint가 다른 system comparison이다. Fresh paired/equal-memory
  causal claim은 하지 않으며, Cosmos의 기존 큰 우위를 이번 작은 E 기능 하나의 향상으로 돌리지 않는다.

| 모델 / workload | vLLM token/s | TTFT 평균 / p95 | TPOT 평균 / p95 | E2E 평균 / p95 |
|---|---:|---:|---:|---:|
| Gemma mixed | 703.81 | 276.41 / 410.58 | 26.48 / 32.39 | 1473.77 / 2162.56 |
| Gemma heavy | 559.83 | 280.57 / 390.78 | 30.59 / 40.54 | 1420.68 / 2138.33 |
| Gemma multi-image | 381.34 | 188.86 / 227.16 | 29.70 / 35.54 | 1109.61 / 1300.30 |
| Cosmos mixed | 923.32 | 868.73 / 2542.45 | 47.29 / 83.92 | 2998.29 / 3132.22 |
| Cosmos heavy | 577.19 | 1630.87 / 3544.35 | 65.15 / 120.75 | 4087.81 / 4240.20 |
| Cosmos multi-image | 243.90 | 260.05 / 401.97 | 12.40 / 16.27 | 645.29 / 654.42 |

| 모델 / workload | Full 처리량 vs vLLM | E1 vs vLLM | E2 vs vLLM |
|---|---:|---:|---:|
| Gemma mixed | +2.09% | −0.52% | +3.78% |
| Gemma heavy | −2.44% | −4.02% | −1.92% |
| Gemma multi-image | +0.37% | −4.29% | −1.65% |
| Cosmos mixed | +27.07% | +23.54% | +23.92% |
| Cosmos heavy | +25.56% | +24.25% | +25.18% |
| Cosmos multi-image | +22.70% | +27.13% | +26.93% |

Cosmos의 E1/E2는 세 workload의 표에 있는 모든 지연 지표에서도 이 frozen vLLM보다 낮다. 하지만 Gemma는 아니다.
Gemma mixed E2는 처리량+3.78%, E2E 평균−3.08%/p95−0.65%지만 TTFT p95+35.09%, TPOT p95+6.96%로 tail이
남아 있다. Gemma heavy/multi-image는 이 실험으로 vLLM의 지연 우위를 해결하지 못했다.

## 최종 결정

**E를 최대 batch로만 실행할 필요는 없고, E1/E2로도 실제 overlap과 점진적 P/D 전진이 가능하다. 그러나 작은 E를
일괄 고정하는 것은 최종 정책이 아니다.** Cosmos multi-image의 E2는 E1과 비슷한 E2E 이득에 TPOT 손실이 더 작고,
Gemma mixed에서는 E2가 매력적이다. 이것을 workload별 hard-coded 선택표로 구현하지 않는다.

이번에 완료한 것은 준비 단위 제어 mechanism, post-calibration 공정 비교, 54-run 진단과 계측 분해다.
아직 완료하지 않은 것은 pre-preparation full/half/one의 online joint selection, Gemma exact correctness gate,
전체 12-workload promotion gate다. 기존 default를 유지하고, 위 동적 선택 계획을 다음 단계로 남긴다.
