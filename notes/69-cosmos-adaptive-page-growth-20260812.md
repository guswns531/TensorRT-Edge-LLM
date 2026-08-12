# Cosmos 적응형 KV page-growth lease 구현과 실험

## 결론

고정된 growth lease 수 대신 decode queue/CUDA-event 지연으로 동시 성장 요청 수를 조절하는 opt-in controller를
구현했다. Bimodal mixed 288-request workload에서는 `full` reservation 3회 중앙값 대비 처리량 `+9.51%`, TTFT p95
`-8.67%`, E2E p95 `-10.62%`를 얻었다. 그 대가로 TPOT p95는 `+12.70%` 악화됐다.

Long-prefill에서는 처리량 `-11.36%`, TTFT p95 `+17.98%`, E2E p95 `+13.59%`로 나빠졌다. decode TPOT만 보는
controller가 prompt 구성과 prefill backlog를 구분하지 못하고 두 workload 모두 growth limit을 32까지 올린 것이
원인이다. 따라서 `full`은 계속 기본값이며, adaptive bounded-overcommit은 짧은/긴 요청이 섞인 throughput 우선
workload의 opt-in 실험 기능으로 둔다.

## 구현

admission 안전성의 상한 `Lmax`와 현재 runnable 성장 요청 수 `Lrun`을 분리했다.

```text
admission safety (변하지 않음)
  sum(base reservations) + sum(largest Lmax tails) <= page pool

runtime controller (decode dispatch마다 관측)
  pressure = (decode queue wait + decode CUDA-event GPU time) / TPOT target
  pressure_ewma = alpha * pressure + (1 - alpha) * previous

  pressure_ewma >= high  -> Lrun += step, up to Lmax
  pressure_ewma <= low   -> Lrun -= step, down to Lmin
```

`Lrun`을 줄이더라도 이미 base 경계를 넘어 성장 중인 sticky owner는 terminal/cancel 전까지 lease를 빼앗지 않는다.
따라서 controller downshift가 page exhaustion이나 진행 중 요청의 교착을 만들지 않는다. 새 요청은 owner 수가 낮아진
limit 아래로 내려갈 때까지 기다린다.

기본 controller 값은 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| EWMA alpha | 0.2 |
| scale-up / scale-down pressure | 0.8 / 0.35 |
| 조정 주기 | decode dispatch 8회 |
| 조정 step | 4 requests |
| benchmark Lmin / Lmax | 1 / 32 |
| benchmark TPOT target | 10 ms |

production config는 `PhasePageReservationConfig`에 있고 기본 `enableAdaptiveGrowthRequests=false`라 기존 동작을
바꾸지 않는다. benchmark에서는 `--adaptivePageGrowth`, `--minPageGrowthRequests`,
`--pageGrowthTpotTargetMs`로 켠다. 일반 phase scheduler의 `--tpotTargetMs`와 별도이므로 한 controller의 튜닝이 다른
controller의 SLO를 암묵적으로 바꾸지 않는다.
matrix runner는 `--adaptive-page-growth`, `--min-page-growth-requests`, `--growth-tpot-target-ms`를 제공한다.

## 코드 위치

| 기능 | 위치 |
| --- | --- |
| controller config/state, safe sticky lease 조정 | `cpp/runtime/scheduling/phaseContextServingFacade.{h,cpp}` |
| dispatch telemetry schema | `cpp/runtime/scheduling/phaseQueueScheduler.h` |
| real-request CLI와 raw CSV | `examples/llm/llm_phase_bench.cpp` |
| matrix 실행 및 controller telemetry 집계 | `scripts/cosmos_reason2/run_real_request_kv_matrix.py` |
| lease 증가와 owner 유지 단위 테스트 | `unittests/phaseDispatchWorkerTest.cpp` |

## 성능 결과

공통 조건은 Cosmos-Reason2-2B FP16 indexed-paged, RTX 3080 10GB, independent prefill/decode TensorRT contexts,
P4/D64, fixed chunk 128, prefill token budget 256, page pool 256 bundles, 64 slots, 288 requests, CUDA graph
reserve 256 MiB다.

### Bimodal mixed, 3회

| policy | token/s median | TTFT p95 median ms | TPOT p95 median ms | E2E p95 median ms | initial pending |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 1,493.34 | 21,359.31 | 17.84 | 24,333.91 | 233 |
| bounded1 adaptive L1..32, target 10 ms | 1,635.32 | 19,506.74 | 20.10 | 21,749.06 | 231 |
| adaptive 변화 | **+9.51%** | **-8.67%** | +12.70% | **-10.62%** | -2 |

adaptive 세 run의 처리량 범위는 1,629.90~1,641.70 token/s였다. controller limit의 중앙값/최대는 32/32,
owner 최대는 32였다. 즉 초기 L1이 혼잡 pressure를 보고 빠르게 확장했고, bounded admission으로 두 요청을 더 일찍
받은 효과가 전체 완료시간을 줄였다.

### Long-prefill

| policy | token/s | TTFT p95 ms | TPOT p95 ms | E2E p95 ms | initial pending |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 912.23 | 24,269.09 | 18.66 | 25,875.16 | 253 |
| bounded1 adaptive L1..32, target 10 ms | 808.65 | 28,633.50 | 19.09 | 29,389.72 | 252 |
| adaptive 변화 | **-11.36%** | +17.98% | +2.29% | +13.59% | -1 |

이 workload에서도 controller limit 중앙값/최대는 32/32, owner 중앙값은 31이었다. 추가 admission은 한 요청뿐인데
더 많은 request가 physical pages와 decode service를 공유해 긴 prefill의 완료 순서와 TTFT를 악화시켰다. 단순 TPOT
pressure만으로는 이 상황을 bimodal의 유익한 overcommit과 구분할 수 없다.

## 정확성·메모리 판정

- full과 adaptive의 288개 greedy output `(request_id, token count, finish reason, text)`는 bimodal과 long-prefill
  모두 정확히 일치했다.
- 모든 run 종료 시 page pool은 `allocated=0`, `available=256`이었다.
- peak CUDA 사용량은 약 9,574.9 MiB, free headroom은 약 299.4 MiB로 pool의 물리 크기는 줄지 않았다.
- adaptive 정책이 줄이는 것은 reserved-but-unused 논리 fragmentation이며, engine/context/graph/page-pool의 고정
  device allocation은 그대로다.

## 다음 판단

현재 controller는 안전하고 효과가 있는 workload가 있지만 범용 자동 정책은 아니다. 다음 scheduler 개선은
`prefill queue tokens`, `oldest prefill wait`, prompt/output ratio와 page pressure를 함께 feature로 넣고 다음 gate를
적용해야 한다.

1. long-prefill을 감지하면 `full` 또는 낮은 admission overcommit으로 복귀한다.
2. bimodal에서만 TPOT budget 안에서 growth limit을 올린다.
3. 어떤 workload에서도 full 대비 TPOT p95 +10% 초과 또는 E2E p95 +3% 초과면 자동 rollback한다.

원시 결과와 run별 값은 `notes/results/cosmos-adaptive-page-growth-20260812.csv`에 기록했다.
