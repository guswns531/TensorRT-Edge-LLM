# Stable D64 CUDA graph replay 검증

## 결론

48.8 offered req/s real-request trace에서 decode CUDA graph가 `0 hit`였던 이유는 capture/key 실패가 아니라
primary 성능 계약에서 graph를 의도적으로 비활성화했기 때문이다. 기존 opt-in 경로로 D64를 startup에서
capture하면 exact binding replay가 정상 동작하고, isolated D64 decode GPU service는 `8.3251 -> 8.0459 ms`,
즉 `3.35%` 짧아진다.

그러나 동일 trace 5회 end-to-end 결과는 eager보다 좋아지지 않았다.

| 5-run mean 또는 run-stat 중앙값 | Eager Current | D64 graph | 변화 |
|---|---:|---:|---:|
| raw request/s, mean | 37.865 | 37.757 | -0.29% |
| joint-SLO pass, mean | 75.97% | 74.03% | -1.94 pp |
| SLO goodput request/s, mean | 28.769 | 27.965 | -2.79% |
| scheduled TTFT mean (ms) | 185.849 | 189.403 | +1.91% |
| scheduled TTFT p95 (ms) | 412.433 | 414.991 | +0.62% |
| TPOT mean (ms) | 15.771 | 15.919 | +0.94% |
| TPOT p95 (ms) | 17.882 | 17.874 | -0.04% |
| scheduled E2E mean (ms) | 1531.400 | 1543.805 | +0.81% |
| scheduled E2E p95 (ms) | 2340.468 | 2345.069 | +0.20% |
| peak GPU memory (MiB) | 9081 | 9085 | +4 |

따라서 CUDA graph를 production default로 승격하지 않는다. 이번 결과는 graph replay의 correctness와
isolated opportunity를 확인했지만, 현재 saturation bottleneck을 해결하지 못했다. 다음 최적화는 graph bucket을
더 늘리는 것이 아니라 sampling/token staging 또는 admission-visible slot lifetime을 줄이는 경로여야 한다.

## 실험 계약

- model: `nvidia/Cosmos-Reason2-2B`, FP16, 비양자화
- engine: P8/D64, fixed packed-prefill chunk 128, 80 stable slots
- effective server admission: 64
- execution: independent TensorRT P/D contexts, profile-free Global selector
- trace: `load-0.05x.json`
- trace SHA256: `c9d64ed7e84fea87352c9b4acedb309932c8f0e9cae71d527fc2f868ec90849d`
- offered load: 48.8 request/s
- measured work/run: 288 requests, 25,872 prompt tokens, 24,960 output tokens
- client max in-flight: 80
- warmup: 64 requests, maximum 32 output tokens
- joint SLO: scheduled TTFT <= 500 ms, TPOT <= 50 ms, scheduled E2E <= 2500 ms
- graph capture: startup warmup에서 D64 한 shape만 capture
- online capture: 비활성
- prefill graph: 비활성
- 반복: fresh backend process 5회
- vLLM: workload 계약이 같으므로 `notes/187-slo-repeat-baseline-20260830.md`의 frozen 5-run 결과를 재사용

## 0 hit 원인 분리

기존 P7/P8 명령에는 `TRT_EDGELLM_CAPTURE_PHASE_GRAPHS`가 없었다. 따라서 다음 최종 통계는 정상적인
eager-only 상태다.

```text
prefill entries=0 hits=0 captures=0
decode  entries=0 hits=0 captures=0
```

Opt-in D64 실험은 다음 계약을 추가했다.

```text
TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=1
TRT_EDGELLM_MAX_PREFILL_GRAPHS=0
TRT_EDGELLM_MAX_DECODE_GRAPHS=1
TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES=64
```

warmup이 D64의 고정 TensorRT binding address와 shape를 capture했고 production에서는 새로운 shape를
동기 capture하지 않았다. 5회 모두 D64 graph replay가 발생했다.

| Graph telemetry | Mean | 95% CI half-width |
|---|---:|---:|
| decode graph hits/run | 247.2 | 2.0 |
| decode graph misses/run | 704.4 | 37.0 |
| hit / execute | 25.98% | - |

miss에는 startup smoke와 warmup도 포함된다. 그래도 D64-only graph가 production decode의 일부에만
적용된다는 방향은 분명하다.

## Correctness

모든 graph 실행은 288/288 request와 24,960/24,960 output token을 완료했다. 5회 token trace hash는 모두
다음 eager 기준과 같았다.

```text
f51d448d5824038f5237cdee55dc50799de5c5cb5a228fcf1166ff1ed30875ca
```

따라서 stable KV ownership, page-table content update, sampling output은 graph replay로 바뀌지 않았다.

## End-to-end 반복 결과

`mean +/- Student-t CI95`, latency는 각 run aggregate의 대표값이다.

| Runtime | Raw request/s | SLO pass | Goodput request/s |
|---|---:|---:|---:|
| Eager Current | 37.865 +/- 0.122 | 75.97 +/- 4.97 pp | 28.769 +/- 1.930 |
| D64 graph | 37.757 +/- 0.365 | 74.03 +/- 8.14 pp | 27.965 +/- 3.315 |
| vLLM frozen | 40.908 +/- 0.029 | 100% | 40.908 +/- 0.029 |

D64 graph는 vLLM보다 raw throughput이 `7.70%`, SLO goodput이 `31.64%` 낮다. workload, model, SLO가
변하지 않았으므로 vLLM을 다시 실행하지 않았다.

### SLO failure attribution

| Failure class | Eager mean requests/run | D64 graph mean requests/run |
|---|---:|---:|
| Pass | 218.8 | 213.2 |
| TTFT only | 58.4 | 63.8 |
| E2E only | 6.4 | 6.4 |
| TTFT + E2E | 4.4 | 4.6 |
| TPOT 포함 | 0.0 | 0.0 |

graph가 TPOT threshold failure를 만들지는 않았다. goodput 변화는 여전히 500 ms TTFT 경계 근처의
admission/queue timing이 몇 request를 이동시킨 결과다.

## Isolated D64 결과

동일 executable, context, D64 binding에서 20회 warmup 후 100회를 CUDA event로 측정했다.

| Variant | D64 decode GPU ms | 변화 |
|---|---:|---:|
| eager `enqueueV3` | 8.3251 | - |
| CUDA graph replay | 8.0459 | -3.35% |

graph는 kernel launch sequence의 device service를 실제로 줄였다. 하지만 약 `0.279 ms/D64 dispatch`의
isolated 절감은 다음 이유로 end-to-end capacity 개선으로 이어지지 않았다.

1. D64 graph hit는 전체 decode execute의 약 26%뿐이다.
2. 이 trace의 joint-SLO 실패는 TPOT 50 ms 초과가 아니라 admission에서 누적된 TTFT queueing이다.
3. graph/eager는 online cost model에서 서로 다른 execution variant다. 제한된 warmup 표본에서는 graph가
   줄인 launch 비용보다 action timing과 SLO threshold 분산이 더 크게 보일 수 있다.
4. P6에서 확인한 sampling event observation residual과 slot lifetime은 graph replay가 직접 제거하지 않는다.

## D32/D48/D64 coverage smoke

D64 miss를 줄이기 위해 D32/D48/D64 세 shape를 startup에서 capture한 smoke도 실행했다.

| Variant | Raw request/s | Decode graph hits | Decode misses | Peak MiB |
|---|---:|---:|---:|---:|
| D64-only, first smoke | 37.343 | 250 | 717 | 9085 |
| D32/D48/D64 | 37.898 | 250 | 718 | 9089 |

D32/D48을 추가했지만 hit 수가 사실상 늘지 않았다. production cohort가 정확히 32 또는 48인 횟수가 거의
없기 때문이다. exact-shape graph를 더 많이 capture하는 방식은 현재 trace의 해결책이 아니며, 4 MiB의 추가
cache footprint만 만들었다.

## 발견한 lifecycle 주의점

현재 smoke executable은 semantic server 전에 controlled phase microbenchmark도 실행한다.
`TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=1`이면 이 microbenchmark의 D1 graph가 같은 executor cache에 남고,
semantic warmup의 D64 graph가 추가된다. 그래서 `MAX_DECODE_GRAPHS=1`인데 최종 executor entry가 2개로
보일 수 있다. production replay correctness나 이번 measured interval에는 영향을 주지 않았지만, graph cache
limit을 엄밀한 메모리 계약으로 사용할 때는 controlled-smoke cache와 serving cache를 분리해야 한다.

이 문제는 graph가 promotion gate를 통과하지 못했으므로 이번 단계에서 production 코드를 복잡하게 만들며
고치지 않는다. 다시 graph를 승격할 때는 server-owned cache lifecycle과 per-shape hit telemetry를 먼저 정리한다.

## 판단

이번 단계에서 새 CUDA graph 구현은 필요하지 않았다. 기존 코드가 stable D64 binding을 정확하게 capture하고
replay했다. 중요한 negative result는 다음과 같다.

```text
D64 graph replay works
  -> isolated D64 GPU service -3.35%
  -> exact token identity preserved
  -> only ~26% decode coverage
  -> raw throughput -0.29%
  -> SLO goodput -2.79%
```

따라서 primary Current는 graph OFF를 유지한다. 다음 우선순위는 이미 실패한 host callback을 반복하거나 graph
bucket을 늘리는 것이 아니라, device-resident token/sampling state와 request completion 사이에서 실제 slot release를
앞당길 수 있는 최소 경로를 설계하고 A/B하는 것이다.

## 산출물

- summary CSV: `benchmarks/phase_serving/results/stable-d64-cuda-graph-ab-20260830.csv`
- D64 5-run raw: `.local/transition-aware-20260830/p9-decode-graph/graph-d64-5run`
- D64 SLO analysis: `.local/transition-aware-20260830/p9-decode-graph/slo-goodput.json`
- multi-bucket smoke: `.local/transition-aware-20260830/p9-decode-graph/graph-d32-d48-d64-smoke`
- eager comparison: `.local/transition-aware-20260830/p7-capacity-safe-cohort/admission64-r5`
- frozen vLLM: `notes/187-slo-repeat-baseline-20260830.md`
