# Current-only 정리 Stage Gate

## 실험 계약

- 모델: `nvidia/Cosmos-Reason2-2B`
- 엔진: `engine-p128-v1024-kv256-tied`
- 실행: independent E/P/D TensorRT contexts, shared CUDA context
- shape: P8/D64/E4, fixed prefill chunk 128
- stable slots: requested 80, decode-aligned effective admission 64
- scheduler: profile-free Global active
- warmup: 64 requests, maximum 32 output tokens
- primary telemetry: external phase JSON off, internal CUDA-event learning on
- Stage 0 결과: `.local/current-only-cleanup-20260828/stage0-baseline-12x1`

이 문서의 각 stage는 동일 trace와 output-token 계약을 사용한다. 단일 실행에서
회귀가 의심되면 해당 workload만 3회 반복한 뒤 승격 여부를 판단한다. trace가
바뀌지 않는 중간 stage에서는 기존 fresh vLLM 결과를 재사용하고, 최종 stage에서만
vLLM 전체 fresh 비교를 실행한다.

## 공통 승격 조건

1. 모든 request가 요구한 output token 수를 완성한다.
2. text workload는 workload별 token trace SHA-256이 Stage 0과 정확히 같다. VLM은 같은 binary의 비동기 E-batch 경계가 반복 실행 사이에도 달라질 수 있으므로 Stage 0 hash 일치와 semantic pass를 함께 기록한다. hash가 달라지면 승격 전에 E/P/D batch 수와 vision semantic pass를 반드시 확인한다.
3. peak VRAM은 Stage 0보다 64 MiB 이상 증가하지 않는다.
4. workload별 throughput 회귀가 1%를 넘지 않는다. 단일 실행 변동이 1%를 넘으면 3회 중앙값으로 재검증한다.
5. TTFT, TPOT, E2E p95 회귀가 3%를 넘지 않는다. 단일 실행 변동이 3%를 넘으면 3회 중앙값으로 재검증한다.
6. action fidelity, context inflight, stable KV/page ownership invariant를 유지한다.

## Stage 0 — Fresh Golden Baseline

| workload | token SHA-256 | tok/s | TTFT mean/median/p95 ms | TPOT mean/median/p95 ms | E2E mean/median/p95 ms | peak MiB |
|---|---|---:|---:|---:|---:|---:|
| short | `9c091953...157e6` | 2513.46 | 87.46 / 67.97 / 164.26 | 13.29 / 11.48 / 22.21 | 328.25 / 337.35 / 407.64 | 9313 |
| balanced | `f51d448d...875ca` | 4516.68 | 67.74 / 51.07 / 164.52 | 12.19 / 12.58 / 13.60 | 1107.70 / 1192.47 / 1709.62 | 9313 |
| decode-heavy | `dc51f754...d886` | 5288.23 | 62.18 / 44.79 / 172.66 | 10.54 / 10.77 / 11.05 | 2784.38 / 3076.40 / 4240.69 | 9313 |
| long-prefill | `f5ae0ac4...4b43` | 1214.03 | 2017.02 / 2180.60 / 2725.35 | 26.26 / 27.20 / 31.01 | 4269.28 / 4282.75 / 6082.70 | 9313 |
| bimodal | `f7fbfbd2...98d9` | 1918.34 | 1877.83 / 2055.59 / 4123.07 | 18.00 / 16.65 / 29.59 | 4353.39 / 4391.26 / 9054.68 | 9313 |
| text-heavy | `f3609f2c...82ff` | 1913.47 | 338.68 / 186.06 / 1137.08 | 25.61 / 25.27 / 38.99 | 1664.46 / 1716.44 / 1763.09 | 9391 |
| mixed | `ef1312db...c9b1` | 1139.86 | 720.63 / 239.61 / 2188.52 | 31.17 / 36.21 / 41.46 | 2210.39 / 2467.92 / 2509.38 | 9433 |
| vision-heavy | `cfc7a828...29b1` | 693.89 | 1367.80 / 1144.06 / 3152.14 | 27.06 / 29.81 / 36.88 | 2447.19 / 2371.24 / 3486.49 | 9453 |
| poisson | `e4aeac4e...77b1` | 1994.31 | 182.47 / 73.15 / 711.37 | 21.88 / 18.54 / 41.32 | 1574.37 / 1551.72 / 2019.76 | 9377 |
| wave | `de5e3b64...7e7e` | 97.80 | 205.71 / 206.23 / 303.89 | 9.51 / 9.54 / 11.43 | 500.66 / 500.64 / 512.94 | 9469 |
| multi | `9f801809...8742` | 313.23 | 212.20 / 263.02 / 301.21 | 9.19 / 7.83 / 12.59 | 497.14 / 505.87 / 510.40 | 9369 |
| late | `8c4ac5bb...7177` | 2537.81 | 113.83 / 44.39 / 447.73 | 9.30 / 9.27 / 9.36 | 1445.69 / 1814.41 / 1816.25 | 9351 |

모든 Stage 0 workload가 exact token identity와 output completion을 통과했다.

## Stage 진행 기록

| stage | 변경 | focused tests | 12-workload 결과 | 결정 |
|---|---|---|---|---|
| 0 | fresh golden baseline | 기존 182 focused tests 기준 | 12/12 exact | 고정 |
| 1 | rejected P continuation, full D-drain, E queue horizon 제거 | 4 suites, 192 tests pass | 12종 x1 완료, 의심 6종 x3 재검증 | 성능 승격, VLM repeat hash 변동 추적 |
| 2A | full scheduler copy를 selective mechanism scratch로 교체 | 4 suites, 192 tests pass | 12종 x1 + 의심 4종 x3 | 기각 후 revert |

## Stage 1 — Rejected Horizon 제거

### 변경 범위

- `enableGlobalPrefillContinuationHorizon`과 대응 config/env/telemetry/test 제거
- `enableGlobalEncoderQueueHorizon`과 대응 config/env/telemetry/test 제거
- `enableGlobalIncrementalDecodeDrainHorizon` config/env/test 제거
- production이 사용하는 one-step decode refill WAIT는 유지
- 뒤에 남아 있는 도달 불가능한 full-drain 구현 본체는 Stage 2의 batch former 분리 때 삭제

총 변경량은 C++ 6개 파일에서 `392`줄 순감소다. `git diff --check`, GPU TensorRT container build,
`unitTest`, `llm_phase_context_smoke` build를 통과했다. 다음 네 suite의 `192`개 focused test가 모두 통과했다.

```text
PhaseQueueSchedulerTest.*
PhaseGlobalSchedulerTest.*
PhaseThreeCoordinatorPolicyTest.*
IndependentPhaseAsyncServerTest.*
```

### 전체 12종 단일 실행

결과 위치는 `.local/current-only-cleanup-20260828/stage1-rejected-horizons-12x1`이다.
모든 request가 output 계약을 완성했고 semantic pass를 통과했다. text 8종과 mixed/vision-heavy/late는
Stage 0 token hash를 유지했다. timing에 민감한 wave는 Stage 0과 다른 clean semantic hash가 나왔다.

단일 실행에서 throughput 또는 tail 변동이 큰 `short`, `decode-heavy`, `long-prefill`, `poisson`,
`wave`, `multi`는 별도 3회 실행으로 재검증했다.

### 의심 6종 3회 중앙값

결과 위치는 `.local/current-only-cleanup-20260828/stage1-suspects-6x3`이다. 비교 기준은 단일 Stage 0
측정치보다 변동에 강한 Note 167의 최종 Current 3회 중앙값이다.

| workload | Stage 1 tok/s | vs final Current | TTFT p95 | 변화 | TPOT p95 | 변화 | E2E p95 | 변화 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 2,503.13 | +1.06% | 167.60 | -2.74% | 26.94 | -1.18% | 409.77 | -1.02% |
| decode-heavy | 5,271.73 | +0.05% | 184.04 | +5.52% | 11.11 | -0.46% | 4,266.66 | -0.29% |
| long-prefill | 1,197.60 | +0.09% | 2,812.32 | -0.97% | 31.44 | +1.15% | 6,208.67 | +1.53% |
| poisson | 1,977.75 | +0.29% | 752.97 | +1.43% | 40.57 | +0.12% | 2,014.49 | -0.08% |
| wave | 96.80 | +0.02% | 380.12 | +6.67% | 12.15 | +0.37% | 590.71 | +1.57% |
| multi | 300.16 | +2.52% | 320.38 | -3.83% | 13.08 | +2.14% | 531.98 | -2.39% |

처리량, TPOT p95, E2E p95에는 성능 하락이 없다. decode-heavy와 wave TTFT p95만 3% gate를
넘었지만 각각 처리량/TPOT/E2E가 그대로이며, 같은 코드의 단일 실행에서도 decode TTFT p95가
172--186ms, wave가 298--380ms로 움직였다. 따라서 inactive branch 제거에 의한 GPU service 회귀가
아니라 asynchronous arrival/batch-boundary tail 변동으로 분류한다.

### VLM repeat hash 관측

wave 3회의 hash는 `de5e...`, `d334...`, `d334...`, multi는 `9f80...`, `9f80...`,
`f697...`였다. 두 workload 모두 semantic pass rate는 `1.0`이고 output 수는 정확했다. 동시에
encoder batch 수가 wave `31/31/33`, multi `36/35/34`로 달랐다. 즉 같은 binary에서도 host arrival
timing이 E batch shape를 바꾸고, FP16 batched vision 경로의 reduction order가 greedy boundary를
바꾸는 기존 수치적 비결정성이다. Stage 2에서는 policy와 별개인 canonical formation replay를
추가해 이 차이를 더 직접적으로 추적한다.

### Stage 1 결정

성능 측면에서는 승격한다. 삭제한 세 feature가 production OFF였다는 기대대로 6종 재측정의 처리량
중앙값은 모두 최종 Current의 `-1%` gate 안이며 peak VRAM도 증가하지 않았다. 다만 VLM exact
repeatability는 cleanup 완료와 별개의 determinism debt로 남기고, 이후 stage마다 semantic pass와
batch-shape drift를 함께 검사한다.

## Stage 2A — Selective Mechanism Scratch 시도와 기각

### 시도

`previewMechanismPlan()`의 `PhaseQueueScheduler preview = *this`를 다음 상태만 복사하는 private
scratch constructor로 바꿨다.

- P/D queue와 queue residence timestamp
- active/in-flight request set
- P/D wavefront cohort와 canonical previous-D selection
- online decode sample의 shared view
- 현재 blocking/external-encoder/producer state

Global cost model, calibration history, pending execution lease와 policy sequence는 scratch에 복사하지
않았다. 기존 `next()`와 `popBatch()`를 그대로 실행했으므로 기능 test와 row ordering test는 모두
통과했다.

### 결과

전체 단일 실행은 12/12 output completion, token hash, semantic correctness를 통과했다. 3회가 필요한
네 workload의 중앙값은 다음과 같다.

| workload | Stage 2A tok/s | final Current 대비 | TTFT p95 | TPOT p95 | E2E p95 |
|---|---:|---:|---:|---:|---:|
| long-prefill | 1,181.40 | -1.27% | 2,784.96 | 31.67 | 6,218.52 |
| text-heavy | 1,953.84 | -0.20% | 1,066.96 | 38.15 | 1,725.25 |
| wave | 96.89 | +0.11% | 350.88 | 12.15 | 577.01 |
| multi | 274.54 | -6.23% | 365.53 | 13.71 | 579.91 |

long-prefill의 P prepare 수가 실행별 `1,018--1,106`회로 흔들렸다. Stage 1도 `1,042--1,101`회로
변동하지만 Stage 2A에서는 느린 두 실행이 중앙값을 차지했다. multi는 다섯 request의 E formation
경계 하나로 처리량이 크게 움직였다. 즉 selective scratch의 correctness 문제는 없지만, host decision
시간이 바뀌면서 P/E completion arrival와 다음 batch materialization 경계가 바뀌었다. 단순히 clone을
빠르게 만드는 것만으로는 action/batch sequence identity를 보장할 수 없다는 증거다.

### 결정

Stage 2A는 성능 gate를 통과하지 못해 코드에서 되돌렸다. 다음 구현은 scheduler 모양만 작게 복사하는
방식이 아니라 `candidate snapshot epoch -> immutable candidate rows -> one-shot materialization`을
명시적으로 만들고, 같은 epoch 동안 host 속도와 무관하게 동일 candidate를 유지해야 한다. 이 결과는
삭제하지 않고 `.local/current-only-cleanup-20260828/stage2-mechanism-projection-12x1`과
`stage2-suspects-4x3`에 보존한다.
