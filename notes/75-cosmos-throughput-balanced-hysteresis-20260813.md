# Cosmos throughput-balanced preset과 TPOT hysteresis

## 결론

direct-cost v5에서 검증한 production 설정을 `throughput-balanced` scheduler preset으로 묶었다. 이 preset은
다음 네 기능을 함께 켠다.

- direct overlap cost가 있는 shape만 static cap을 초과해 overlap한다.
- cost model coverage miss는 decode-only로 보수적으로 처리한다.
- TPOT hard guard가 연속 overlap과 예측 decode debt를 제한한다.
- 최근 decode service p95가 목표에 가까워지면 static-128 정책으로 폴백하고, 충분히 낮아진 뒤에만 복구한다.

기존 `balanced` preset은 dynamic prefill/decode batch 선택 실험용이므로 의미를 바꾸지 않았다.
`throughput-balanced`는 검증한 P/D 상한을 그대로 사용하는 production preset이다. 현재 권장 조합은 independent
TensorRT contexts, P4/D64, fixed chunk 128, ragged padded-token budget 256이다.

## controller 동작

각 decode dispatch가 끝날 때 다음 service proxy를 rolling window에 넣는다.

```text
decode service sample = oldest decode queue wait + decode CUDA-event GPU time
pressure = rolling p95(sample) / TPOT target
```

기본 window는 32 dispatch, cold-start 최소 sample은 8이다. pressure가 0.8 이상이면 latency-safe fallback으로
들어가고 0.6 이하가 되어야 복구한다. 서로 다른 enter/exit threshold를 사용하므로 경계에서 매 dispatch마다
정책이 뒤집히는 현상을 막는다.

fallback은 prefill/decode overlap 전체를 끄지 않는다. `maxOverlapPrefillTokens=128` 이내의 기존 overlap은 유지하고,
cap을 넘기 위해 direct cost를 조회해야 하는 후보만 decode-only로 바꾼다. 따라서 안전 모드는 이전 static-128과
같은 admission 경계를 사용한다.

dispatch CSV의 `latency_safe_fallback`으로 각 dispatch의 상태를 확인할 수 있다. runner status에는 preset이
암시적으로 켠 hard guard, direct-cost requirement, cost-aware admission, hysteresis도 `effective_*` 필드로 남긴다.

## 동일 production trace GPU 결과

조건은 Cosmos-Reason2-2B FP16 indexed-paged, RTX 3080 10GB, 하나의 CUDA primary context를 공유하는 independent
prefill/decode TensorRT contexts, P4/D64, 80 stable slots, 256 page bundles, CUDA graph P4/D64 budget, 288 requests,
arrival 1000 req/s, output token 합계 24,960, fixed chunk 128, ragged token budget 256이다. 이전 production 실행의
materialized trace를 직접 재사용해 prompt, output length, arrival offset을 고정했다.

### 기본 TPOT target 50ms

3회 결과의 median은 다음과 같다.

| metric | manual cost-aware v5 | throughput-balanced | 변화 |
| --- | ---: | ---: | ---: |
| generated token/s | 4100.5 | 4114.0 | +0.33% |
| TTFT p95 | 4403.7 ms | 4339.4 ms | -1.46% |
| TPOT p95 | 19.061 ms | 18.841 ms | -1.15% |
| E2E p95 | 5397.3 ms | 5367.7 ms | -0.55% |

반복 처리량은 4114.0/4114.3/4108.7 token/s였다. 각 실행에서 direct-cost evaluation 48회, guard defer 7회,
coverage miss 0회였고 fallback은 한 번도 켜지지 않았다. dispatch service proxy p95는 31.54~31.63ms로 40ms
enter threshold보다 낮았다. 기존 manual v5와 모든 288개 request의 output token 수와 text가 일치했다.

### 강제 pressure 검증: TPOT target 25ms

같은 trace에서 target만 25ms로 낮추면 20ms가 enter threshold, 15ms가 exit threshold가 된다.

| metric | target 50ms | target 25ms fallback | 변화 |
| --- | ---: | ---: | ---: |
| generated token/s | 4114.0 | 3393.6 | -17.51% |
| TTFT p95 | 4339.4 ms | 5925.3 ms | +36.55% |
| TPOT p95 | 18.841 ms | 16.863 ms | -10.50% |
| E2E p95 | 5367.7 ms | 6429.5 ms | +19.78% |

38번째 dispatch에서 fallback에 진입했고 596 dispatch 동안 유지한 뒤, decode pressure가 내려간 634번째
dispatch에서 복구했다. 이 구간에는 cap 초과 cost-aware overlap이 없었다. 처리량과 TTFT를 희생해 decode tail을
보호하는 의도한 동작이며, hysteresis transition은 정확히 두 번이었다. target 25ms 실행도 모든 output token/text가
기준과 일치했다.

## 사용 방법

`llm_phase_bench`에는 다음 옵션을 사용한다.

```text
--schedulerProfile throughput-balanced
--schedulerCostJson <cost-model-v5.json>
--tpotHysteresisEnterRatio 0.8
--tpotHysteresisExitRatio 0.6
--tpotHysteresisWindow 32
--minTpotHysteresisSamples 8
```

real-request runner에서도 같은 이름의 `--scheduler-profile throughput-balanced`와 kebab-case hysteresis 옵션을
지원한다. cost JSON은 필수다. production rollout에서는 먼저 50ms target을 유지하고, trace별 dispatch service
p95와 request TPOT p95를 함께 관찰한 뒤 threshold를 조정해야 한다.

artifact는 `.local/cosmos-reason2-2b/throughput-balanced-hysteresis-20260813/`에 있다.
