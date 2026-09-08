# Bounded P/D frontier 전달 실험

## 목적과 변경 범위

Note254에서 분리한 두 원인 중 **P/D winner-only 전달**만 변경한다.
TTFT hard guard, KV/page pool, engine, chunk, RLS features/labels는 변경하지 않는다.

`PhaseThreeCoordinatorConfig::enableGlobalPdFrontier`는 기본 false다.
실험 example에서는 `TRT_EDGELLM_GLOBAL_PD_FRONTIER=1`로 켠다. 환경 변수 존재 여부를
사용하므로 끌 때는 `=0`이 아니라 변수를 제거해야 한다.

`phaseThreeCoordinator.cpp::dispatchGlobalAction()`에서:

1. 기존 P/D preview winner와 frontier를 같은 호출에서 받는다.
2. E-ready가 있고 idle-context 선택일 때만 단독 P와 D를 각 최대 하나 보존한다.
3. Winner와 candidate ID가 같으면 추가하지 않는다. 새로운 batch/row ordering은 만들지 않는다.
4. 각 대안에 자신의 service time 및 uncertainty를 사용해 E→P protected completion을 추가한다.
5. 기존 최종 selector와 dispatch 경로를 그대로 사용한다.

최대 두 후보만 추가하며 selector의 기존 maxCandidates=11은 유지한다.
새 E+P+D나 새 overlap shape는 추가하지 않는다. residual augmentation 및 강제 warmup probe 선택은
제외한다. E가 없는 text-only 선택도 기존 P/D selector를 그대로 사용한다.

이는 모든 P/D overlap 후보를 전달하는 완전한 frontier 통합이 아니라 **단독 P/D 전달 누락을
분리하는 최소 ablation**이다. E 단독 후보의 기존 보호 입력도 그대로 둔다. 전체 request-union
보호 입력의 통일이나 TTFT guard 재설계는 별도 변경이어야 한다.

## 재현 계약

- Source base: `532f00b`, branch `codex/v0101-phase-forward-port`.
- Binary SHA256: `6ba4b28b39a7a4dc3a958a8f155d7d2bf1fa034d3205a21bcd01b3ca6c43851d`.
- Artifact root: `.local/results/v0101-forward-port/pd-frontier-20260908/`.
- Same FP16 Cosmos engines and P8/D64/E4, P128, KV256 pages, slots80 as note254.
- Same generic calibration request traces; fresh process per run. Same calibration requests do not imply
  identical learned posterior, because runtime timing can differ.
- `run_policy_warmup_matrix.py` commands are retained in each `commands.json`.
- First `audit/` smoke used an unrecognized events-path environment variable; output correctness was checked,
  but it is not the retained event capture. `audit-final/` uses the actual `PHASE_TELEMETRY_PATH` setting.
- Primary off/on results have telemetry disabled. Frozen vLLM is reused under note254's unchanged contract.

## 검증 상태

Runtime/smoke and unitTestRuntime builds passed. Existing scheduler regression tests: **209/209 passed**.
New behavior is checked through actual two-stage audit inputs, rather than preview-only snapshots.
The initial Vision-heavy smoke produced the same greedy token hash as note254.

## 결과

### 실제 전달 검증

`audit-final` measurement epoch: 198개 decision 모두 감사됨. 이 중 E/P/D 2단계 선택 68회에서
P/D upstream의 단독 P/D 후보 **91개 모두 최종 selector에 동일 candidate ID로 전달**됐다.
누락은 0개, 최종 후보 수 최대 5개다. 최종 단계가 upstream winner 대신 다른 단독 P/D 후보를
선택한 경우는 **21회**다. 이는 선택 가능한 경로를 실제로 확장했음을 확인하지만 성능 이득의 증명은 아니다.

D-ready194회 중 D 포함 action129회, non-D65회. Non-D에서 D absent43회, SLO-late21회,
safe1회였다. 상위 단계 전달 누락을 없애도 upstream guard에서 이미 제거한 후보는 되살아나지 않는다.
safe-D를 두고 late action을 선택한 사례와 post-select override는 0회였다.
구성 비교용 단발 계측이므로 note254의 decision count와 엄밀한 matched-snapshot 차이로 해석하지 않는다.

### 비계측 same-binary off/on 및 frozen vLLM

각 off/on은 1회다. 3/3 greedy token hash 일치. Frozen vLLM은 note254의 동일 계약 결과를
재사용한다. latency 단위 ms, mean/p95. On/off는 전체 off 세 trace 후 on 세 trace 순서로
실행했으므로 교대 반복에 의한 시간 drift 통제까지 완료한 실험은 아니다.

| Workload | Variant | token/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---|---:|---:|---:|---:|
| short | off | 2358.07 | 100.09/182.40 | 13.51/27.43 | 349.85/431.70 |
| short | on | 2376.92 | 102.04/182.94 | 13.07/22.64 | 346.70/428.60 |
| short | frozen vLLM | 1983.53 | 174.92/263.97 | 13.36/24.88 | 426.71/503.71 |
| mixed | off | 1050.94 | 750.90/2283.62 | 33.39/41.39 | 2344.28/2719.57 |
| mixed | on | 1048.15 | 791.59/2407.66 | 40.31/63.78 | 2629.91/2766.73 |
| mixed | frozen vLLM | 921.48 | 874.56/2541.43 | 46.97/84.02 | 3008.38/3140.85 |
| poisson | off | 1750.09 | 238.83/953.25 | 23.82/39.62 | 1779.14/2229.03 |
| poisson | on | 1753.07 | 237.04/956.78 | 23.59/38.95 | 1764.84/2210.96 |
| poisson | frozen vLLM | 1800.07 | 438.11/902.68 | 22.19/45.67 | 1800.22/2266.55 |

Short는 E가 없어서 이 옵션의 새 후보 확장 경로를 실행하지 않는다. 따라서 short의 +0.80%
처리량이나 TPOT 차이를 새 정책의 이득이라고 주장하지 않는다. 반복 변동을 보는 negative control이다.

Mixed는 처리량 -0.27%로 비슷하지만 TTFTp95 약 +5.43%, TPOTp95 약 +54.10%,
E2E mean 약 +12.18%로 승격 기준을 충족하지 못했다. Poisson 처리량은 +0.17%로 사실상 비슷하다.
Frozen vLLM 대비 on 처리량은 Short +19.83%, Mixed +13.75%, Poisson -2.61%다.
vLLM 대비 일부 우세와 기존 current 대비 개선은 별개의 판정이다.

## 결론과 다음 단계

**메커니즘 전달 검증은 성공, 성능 승격은 보류**다. 기본 `enableGlobalPdFrontier=false`를 유지한다.
이번에는 원래 전체12 재실험을 통과했다고 주장하지 않는다. 회귀 진단용 세 trace에서 이미 Mixed
tail 문제가 나타났으므로 전체12 확대 전에 동일 snapshot의 목적함수/보호 대상 일관성을 검토한다.

이번 수정은 E 후보 보호 입력을 기존 P/D winner에서 가져오는 동작까지 바꾸지 않았다.
대안 후보가 추가된 상태에서 각 candidate가 같은 request 집합과 같은 service horizon을 보호하는지
검증해야 한다. 전달이 올바르다고 selector objective까지 공정해진 것은 아니다.

권장 다음 작업:

1. Mixed on/off를 교대 반복하고 D service interval 및 actual selected candidate를 함께 대조한다.
2. 같은 snapshot에서 후보별 protected request/phase 집합과 reference horizon이 동일 계약인지 확인한다.
3. 차이가 확인되면 request-union 보호 입력을 공통 생성하는 최소 수정만 한다. TTFT guard 변경은
   여기에 섞지 않고 별도 ablation으로 둔다.
4. 위 진단 세 trace의 latency gate 통과 후에만 전체12를 확대한다.

Scheduler CPU decision p95 및 true GPU start-lag gate는 이번 경량 audit에서 새로 측정하지 않았다.
최대 후보 수 5라는 결과만으로 host overhead 무회귀를 주장하지 않는다.
