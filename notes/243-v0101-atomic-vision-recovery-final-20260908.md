# v0.10.1 atomic vision prefill 복원 및 최종 측정 정리

Date: 2026-09-08. Development branch: `codex/v0101-phase-forward-port`.

**Comparison-contract follow-up:** [note 244](244-v010-v0101-detailed-regression-audit-20260908.md)
audits the raw artifacts. Frozen vLLM uses client max-in-flight 80 versus phase HTTP 64 in four text workloads;
the comparison is not equal-admission. Both old/current full12 phase runs have zero P/D graph captures/hits.
The recorded TTFT/E2E exclude client concurrency wait; note 244 adds arrival-relative metrics and request-level
cross-engine token comparisons. These qualifications supersede any broader same-contract interpretation below.

## 결론과 상태

이번 수정은 새 scheduler가 아니라 forward-port에서 달라진 실행 경로 두 곳을 복원한다. 전용 vision-P profile이 있는 엔진에서는 external prefill을 atomic하게 처리하고, residual P+D 경로에서 external lineage만으로 Scalar 평가를 제외하지 않는다. KV allocator/pool/precision은 변경하지 않았다.

**완료된 실행과 production promotion을 구분한다.** V1 전체 12-workload × 3회, V0/V2 전체 각각 1회를 완료했다. V1 반복 출력 hash는 mixed/poisson/vision-heavy/multi-image에서 불일치하므로 exact-output promotion은 실패 상태다. v0.10 최고치 복구도 아직 완료됐다고 할 수 없다.

## 구현과 실행 아키텍처

- `cpp/runtime/scheduling/phaseThreeCoordinator.cpp::dispatchGlobalPrefillDecodeResidual`: external prefill도 P+D contextual prediction/observation 후보에 포함한다. pending/encoding/busy encoder/preparation에 따른 producer-critical-path 보호, SLO/feasibility와 context single-inflight는 유지한다. external이라는 이력만으로 학습 권한을 막지 않는다.
- `examples/llm/llm_phase_context_smoke.cpp`: `packedPrefill && !hasVisionPrefillProfile()`일 때만 chunked vision prefill을 설정한다. P1024 profile이 있는 실행은 atomic external P, 없는 엔진은 P128 chunk 경로다. 이 자동 선택은 HTTP/IPC benchmark adapter의 설정이며 public serving API의 명시적 chunk 설정을 일괄 변경한 것은 아니다.

```text
하나의 CUDA context
  E TensorRT context / E stream
    -> request-owned vision lease -> encoded-ready queue
  Text-P context (P128) + Vision-P sibling (P1024) / P stream
    -> 같은 P workspace arena 재사용 (P끼리는 동시 실행하지 않음)
    -> stable indexed-paged KV lease
  D TensorRT context / D stream
    -> D64 cohort -> sampling completion -> ready queue
  Copy stream
    -> pinned asynchronous staging / explicit event dependency

V0 Exact / V1 Scalar / V2 Scalar+Transition
  같은 candidate / ownership / graph / SLO mechanism
  서로 다른 decision-cost authority 및 bounded transition 평가
```

기존 tied-head 및 profile-local plugin workspace 수정은 유지한다. +582 MiB 문제와 이번 +218 MiB는 구분해야 한다. 후자는 전용 P1024 실행을 지원하기 위한 P workspace 증가이며 KV 증가가 아니다.

## 실험 계약과 재현

- Model: nvidia/Cosmos-Reason2-2B, FP16. RTX 3080 10GB, TensorRT 26.06 container.
- P8/D64/E4, stable slots 80, KV pages 256 (~3584 MiB), KV capacity 2048, text chunk 128.
- atomic engine: `.local/v0101-forward-artifacts/cosmos-reason2-2b/engine-p8-d64-kv256-p128-vp1024-atomic`.
- llm.engine SHA256: `084d039248e08dc192a57ebf38d521ee1fdca2f037a865f55ac0b0b904214b0b`.
- P128-only comparison SHA256: `c420acaa60ef65a0c76569afc12104e764290c0c4c7fd9d6af46c3e9a6154634`.
- Same HTTP traces, ignore-EOS output contract, generic calibration and memory limits. Text calibration 239 requests, VLM 319. Calibration reports with E-pair probes zero do **not** establish converged encoder overlap learning.
- Export artifact and engine build were validated before runtime measurement; no new export is required for these runtime-only changes. Earlier export/build provenance remains in note 242 and artifact directories.
- Source: `4099f76` (base `05fe00d` plus the two changes described above). All final atomic variants use the same built binary.
- Interrupted V2 used deleted `/tmp/v010-bench-tools` scripts. Recovery scripts are retained in `.local/results/v0101-forward-port/replay-tools/`, restored from Git object `1e4807e` (`scripts/cosmos_reason2/`); the retained base manifest rewrites only those script paths. This is archived benchmark tooling, not a new serving implementation.
- VLLM is frozen historical reference, reused because request/model/precision contract is unchanged. It is not a fresh paired repetition; small deltas are directional, not statistical superiority claims.

## V1 전체 12-workload 결과

Latency 단위 ms. 아래 mean은 run-level mean의 median이고 p95는 run p95의 median이다. Aggregate key 이름의 `mean_of_run_means`와 달리 harness는 `statistics.median`을 사용한다. Throughput은 run token/s의 median. 서로 다른 run의 요약값을 한 가상 run으로 해석하거나 percentile을 합산하지 않는다.

| Workload | token/s | req/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | peak MiB (median) | 3회 hash 일치 |
|---|---:|---:|---:|---:|---:|---:|:---:|
| short | 2347.45 | 108.34 | 109.05 / 188.46 | 12.92 / 22.27 | 352.18 / 434.29 | 9279 | 일치 |
| balanced | 4164.17 | 48.05 | 71.21 / 174.29 | 13.28 / 15.03 | 1201.16 / 1882.52 | 9279 | 일치 |
| decode-heavy | 4898.64 | 18.84 | 71.79 / 207.18 | 11.37 / 12.24 | 3009.52 / 4639.49 | 9279 | 일치 |
| long-prefill | 1223.75 | 14.12 | 2068.31 / 2727.59 | 25.34 / 29.45 | 4221.91 / 5870.92 | 9279 | 일치 |
| bimodal | 1869.52 | 12.19 | 1987.13 / 4537.04 | 18.16 / 26.00 | 4504.74 / 9628.46 | 9279 | 일치 |
| text-heavy | 1872.49 | 35.33 | 314.58 / 1107.21 | 26.65 / 39.82 | 1687.72 / 1773.05 | 9377 | 일치 |
| mixed | 1078.53 | 23.57 | 714.19 / 2213.64 | 33.16 / 40.55 | 2273.13 / 2649.30 | 9401 | 불일치 |
| poisson | 1878.52 | 25.73 | 218.41 / 764.79 | 22.81 / 41.01 | 1660.85 / 2125.02 | 9377 | 불일치 |
| vision-heavy | 668.94 | 17.38 | 1342.22 / 3167.36 | 30.52 / 37.38 | 2532.63 / 3606.61 | 9377 | 불일치 |
| wave-drain | 96.31 | 3.01 | 248.30 / 398.29 | 9.89 / 13.90 | 555.92 / 615.70 | 9347 | 일치 |
| late-vision | 2399.25 | 16.63 | 130.03 / 493.15 | 9.81 / 9.86 | 1535.31 / 1922.13 | 9351 | 일치 |
| multi-image | 298.40 | 9.32 | 233.13 / 317.70 | 9.40 / 12.39 | 524.70 / 535.91 | 9323 | 불일치 |

## 처리량 비교: 실행 경로 복구와 정책 변경을 혼동하지 않기

Historical v0.10/vLLM anchors are the rounded values retained in note 242. Ratios therefore have rounding uncertainty. P128 vs atomic changes engine/profile/vision chunking and is not a policy-only ablation.

| Workload | vs P128 V1 | vs v0.10 best | vs frozen vLLM |
|---|---:|---:|---:|
| short | -0.11% | -5.30% | 18.35% |
| balanced | -1.89% | -6.53% | -3.60% |
| decode-heavy | -1.38% | -7.91% | 0.91% |
| long-prefill | 3.73% | 0.27% | 9.18% |
| bimodal | 3.79% | -2.43% | 0.07% |
| text-heavy | 8.21% | -12.17% | 14.54% |
| mixed | 10.10% | -8.99% | 17.04% |
| poisson | 5.64% | -9.07% | 4.36% |
| vision-heavy | 12.80% | -2.76% | 15.49% |
| wave-drain | 0.58% | -1.72% | 0.53% |
| late-vision | -2.85% | -6.00% | 1.70% |
| multi-image | 13.97% | -4.39% | 22.04% |

Geometric mean: atomic V1 vs P128 V1 **+4.24%**, vs historical v0.10 **-5.64%**, vs frozen vLLM **+8.06%**. vLLM throughput 우세는 11/12이고 balanced는 -3.60%다. 따라서 “모든 회귀 해결” 또는 “모든 workload에서 vLLM 우세”는 아니다.

## 메모리와 회귀 해석

P128 ready 9061 MiB -> atomic ready 9279 MiB: **+218 MiB**. P workspace는 79,695,360 -> 301,993,472 bytes, D는 21,548,544 bytes. KV pages/lease/precision은 동일하다. 전체 V1 peak median 최대 9401 MiB로 10,240 MiB에서 약 839 MiB 여유다. 이는 표본화한 GPU-used 값이지 allocator fragmentation의 정밀 측정은 아니다.

Atomic 복원은 vision-heavy +12.80%, multi-image +13.97%, mixed +10.10%, text-heavy +8.21%로 P128-only보다 개선됐다. text-only는 혼합 결과다. balanced 별도 3회에서는 4272.53 token/s median이었지만 최종 전체 suite는 4164.17이다. 더 높은 별도 값을 최종값으로 선택하지 않는다.

Balanced diagnostic windows: P128 wall 6069.72 ms / idle 10.596% / P+D 15.629%; atomic wall 6363.24 ms / idle 11.112% / P+D 13.835%. 이 단일 진단은 timing/cohort interaction의 후보 근거이며 원인 확정은 아니다. 원본 event 파일에는 calibration도 포함되므로 파일 전체 P/D completion row count를 measurement-only dispatch count로 사용하면 안 된다.

## Correctness와 남은 gate

전체 V1 36회 중 실제 관측 peak 최댓값은 poisson run-002의 **9409 MiB**다. 따라서 최악 관측 headroom은 **831 MiB**이고, 위 839 MiB는 workload별 median peak 기준이다.

1. V1 반복 hash 불일치 4개: mixed, poisson, vision-heavy, multi-image. 요청 수와 생성 token 수가 맞아도 greedy identity와는 다르다.
2. P128 text-heavy request 20 단독 BS1 5회는 동일했다. 이는 batching/FP16 sensitivity 가설을 지지하지만 KV corruption을 배제하지 않는다. Logit 비교, canonical row-order replay 및 sanitizer가 아직 필요하다.
3. V0/V2 단일 실행의 deterministic=true는 반복 검증이 아니다. Cross-policy identity는 별도로 비교한다.
4. Old v0.10과의 performance gap은 남아 있다. 이번 자료만으로 exporter/tactic/encoder/host 중 한 원인으로 확정하지 않는다.
5. 본 수정의 default promotion은 보류한다. 프로파일별 benchmark adapter 경로 복원과 산출물 정리를 완료한 것이지 전 workload correctness/performance gate가 모두 통과한 것은 아니다.

## 다음 작업 (새 정책 추가 전)

1. 실패한 네 trace에서 request membership / row order / KV lease signature를 고정한 replay와 logits 비교로 수치 차이와 ownership 오류를 분리한다.
2. balanced 및 old-v0.10 gap은 같은 lifecycle, shape, graph coverage와 measurement-only timeline으로 분해한다. Calibration rows 제외를 강제한다.
3. 수정이 생긴 경우 영향받는 trace를 반복하고, 통과 후 12-workload를 재검증한다. frozen vLLM 계약이 바뀔 때만 fresh baseline을 측정한다.
4. V0/V1/V2 모두 동일 실행 substrate를 유지한다. Workload-specific policy knob 또는 KV 축소로 회귀를 감추지 않는다.

## 최종 V0/V1/V2 비교 (재개 완료)

V0/V2는 각각 1회, V1은 3회다. V2 short는 중단 전 결과를 보존하고 나머지 11개를 재시작 후 측정했다. V2 재개 중 CPU 소스 검사도 수행했으므로 미세한 차이를 strict idle-host paired estimate로 해석하지 않는다. 이는 screening이며 V2의 확정적 우열은 같은 세션의 교차 반복이 필요하다.

| Workload | V0 token/s | V1 token/s | V2 token/s | V1/V0 | V2/V0 | 모든 반복·정책 hash 동일 |
|---|---:|---:|---:|---:|---:|:---:|
| short | 2363.89 | 2347.45 | 2250.86 | -0.70% | -4.78% | 일치 |
| balanced | 4143.44 | 4164.17 | 4032.69 | 0.50% | -2.67% | 일치 |
| decode-heavy | 4920.29 | 4898.64 | 4748.28 | -0.44% | -3.50% | 일치 |
| long-prefill | 1011.42 | 1223.75 | 1200.39 | 20.99% | 18.68% | 일치 |
| bimodal | 1815.40 | 1869.52 | 1837.88 | 2.98% | 1.24% | 일치 |
| text-heavy | 1906.91 | 1872.49 | 1910.48 | -1.81% | 0.19% | 일치 |
| mixed | 1145.71 | 1078.53 | 1137.51 | -5.86% | -0.72% | 불일치 |
| poisson | 1790.52 | 1878.52 | 1768.90 | 4.92% | -1.21% | 불일치 |
| vision-heavy | 673.92 | 668.94 | 705.21 | -0.74% | 4.64% | 불일치 |
| wave-drain | 97.09 | 96.31 | 96.30 | -0.80% | -0.81% | 일치 |
| late-vision | 2384.85 | 2399.25 | 2377.51 | 0.60% | -0.31% | 일치 |
| multi-image | 298.51 | 298.40 | 298.53 | -0.04% | 0.01% | 불일치 |

V0 대비 geometric mean: V1 **1.45%**, V2 **0.74%**. 모든 반복과 정책의 hash가 동일한 workload는 **8/12**다. Hash list 길이(1회 vs 3회)를 직접 비교하지 않고 전체 hash 집합의 원소 수로 검사했다.

### V0 latency 상세 (1회)

| Workload | req/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 |
|---|---:|---:|---:|---:|
| short | 109.10 | 105.48 / 180.86 | 13.15 / 23.07 | 351.80 / 433.78 |
| balanced | 47.81 | 70.39 / 175.17 | 13.40 / 15.30 | 1212.79 / 1921.55 |
| decode-heavy | 18.92 | 66.38 / 195.74 | 11.35 / 12.06 | 3001.96 / 4584.55 |
| long-prefill | 11.67 | 2386.96 / 3204.39 | 32.45 / 38.21 | 5161.44 / 7561.37 |
| bimodal | 11.84 | 2005.19 / 4189.00 | 19.03 / 28.55 | 4636.88 / 9372.94 |
| text-heavy | 35.98 | 292.41 / 908.30 | 26.53 / 41.24 | 1652.92 / 1763.64 |
| mixed | 25.04 | 648.41 / 1860.67 | 38.99 / 65.24 | 2395.25 / 2546.08 |
| poisson | 24.53 | 231.80 / 933.88 | 23.95 / 41.58 | 1773.60 / 2235.13 |
| vision-heavy | 17.50 | 1328.57 / 3174.82 | 30.37 / 37.23 | 2513.24 / 3581.86 |
| wave-drain | 3.03 | 240.11 / 420.11 | 10.27 / 13.84 | 558.36 / 658.47 |
| late-vision | 16.53 | 130.05 / 496.38 | 9.87 / 9.92 | 1544.27 / 1933.95 |
| multi-image | 9.33 | 233.20 / 317.76 | 9.39 / 12.36 | 524.42 / 535.83 |

### V2 latency 상세 (1회)

| Workload | req/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 |
|---|---:|---:|---:|---:|
| short | 103.89 | 125.76 / 201.19 | 12.85 / 23.44 | 372.21 / 456.10 |
| balanced | 46.53 | 73.02 / 174.21 | 13.66 / 15.47 | 1235.87 / 1939.32 |
| decode-heavy | 18.26 | 72.76 / 204.46 | 11.77 / 12.53 | 3112.17 / 4759.18 |
| long-prefill | 13.85 | 2105.62 / 2875.63 | 26.15 / 31.97 | 4324.97 / 5960.86 |
| bimodal | 11.99 | 1977.49 / 4428.13 | 18.63 / 28.35 | 4553.18 / 9539.97 |
| text-heavy | 36.05 | 281.65 / 993.79 | 26.52 / 40.60 | 1645.67 / 1746.40 |
| mixed | 24.86 | 619.99 / 1860.02 | 40.43 / 66.23 | 2418.96 / 2564.29 |
| poisson | 24.23 | 242.29 / 988.71 | 23.84 / 42.98 | 1783.79 / 2240.04 |
| vision-heavy | 18.32 | 1275.13 / 3087.88 | 52.87 / 87.85 | 3269.63 / 3438.68 |
| wave-drain | 3.01 | 246.85 / 393.89 | 9.87 / 13.72 | 552.84 / 606.45 |
| late-vision | 16.48 | 130.35 / 494.98 | 9.90 / 9.96 | 1547.87 / 1939.29 |
| multi-image | 9.33 | 232.90 / 317.11 | 9.41 / 12.37 | 524.50 / 535.53 |

## 검증 및 보관

- Source commit: `4099f76`, signed-off.
- Final targeted runtime tests: **168/168 passed** (`PhaseQueueSchedulerTest.*:PhaseThreeCoordinatorPolicyTest.*:PhaseRuntimeCostTrackerTest.*`).
- Modified C++ files: pre-commit license / clang-format / codespell / line-ending checks passed; `git diff --check` passed.
- Full HTTP runs: V1 36, V0 12, V2 12, all expected workload result aggregates present. HTTP execution completion is not an exact-token promotion pass.
- Retained root: `.local/results/v0101-forward-port/`.
- Primary: `atomic-vp1024-final-full12-r3`, `atomic-vp1024-final-full12-v0`, `atomic-vp1024-final-full12-v2`.
- Machine-readable summary: `atomic-final-v0-v1-v2.csv` (same numbers as this report, explicit repeats and all-policy fidelity).
- Recovery tooling: `replay-tools/`; no engine/model/token is tracked in Git. Existing referenced diagnostics remain retained; no engines were deleted in this completion step.

V1을 반복 검증된 비교 기준으로 유지한다. V0도 일부 workload에서 더 빠르고 V2는 전체 winner가 아니므로 workload별로 좋은 숫자만 조합한 “최종 성능”을 만들지 않는다. 다음 해결 대상은 four-trace numerical/ownership isolation과 balanced 및 historical-v0.10 parity이며, 본 보고서는 이를 미해결로 명시한다.
