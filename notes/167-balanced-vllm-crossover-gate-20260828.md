# Balanced vLLM crossover와 12-workload production 회귀 검증

## 결론

고정 packed-prefill chunk `128`을 유지하면서, workload 이름이나 사전 profile을 보지 않는 공통 serving mechanism만 개선했다. 최종 Cosmos Reason2-2B balanced real-request trace에서 Current는 vLLM보다 처리량이 `5.29%` 높고 TTFT, TPOT, E2E의 mean/median/p95 아홉 지표가 모두 낮다.

| balanced, C64, 3회 중앙값 | Current | vLLM | Current 변화 |
|---|---:|---:|---:|
| generated token/s | 4,562.677 | 4,333.524 | **+5.29%** |
| TTFT mean (ms) | 64.672 | 112.968 | **-42.75%** |
| TTFT median (ms) | 47.917 | 96.477 | **-50.33%** |
| TTFT p95 (ms) | 165.070 | 280.571 | **-41.17%** |
| TPOT mean (ms) | 12.065 | 12.128 | **-0.52%** |
| TPOT median (ms) | 12.507 | 12.526 | **-0.15%** |
| TPOT p95 (ms) | 13.367 | 13.516 | **-1.10%** |
| E2E mean (ms) | 1,094.425 | 1,149.778 | **-4.81%** |
| E2E median (ms) | 1,170.230 | 1,234.823 | **-5.23%** |
| E2E p95 (ms) | 1,693.421 | 1,766.133 | **-4.12%** |
| peak GPU memory (MiB) | 9,313 | 9,039 | +274 |

Current의 세 반복은 exact token hash가 모두 `f51d448d5824038f5237cdee55dc50799de5c5cb5a228fcf1166ff1ed30875ca`로 같았다. vLLM HTTP 응답은 이 harness에서 exact token ID를 제공하지 않으므로 동일한 exact-token gate를 적용하지 못했지만, 세 실행 모두 요구한 output token 수를 완성했다.

이 결과가 의미하는 범위는 분명하다.

- balanced에서는 처리량뿐 아니라 아홉 latency 통계가 모두 vLLM보다 좋다.
- 전체 12개 workload의 처리량도 모두 vLLM보다 높다.
- 이전 Current 대비 throughput 보존 gate는 wave/drain의 `-1.00%`를 포함해 모두 `-3%` 이내이거나 개선이다.
- 모든 workload의 모든 latency 지표까지 vLLM보다 좋다는 뜻은 아니다. short TPOT p95와 bimodal TTFT/E2E median은 아직 vLLM이 낫다.
- vision-heavy vLLM 값은 GPU OOM으로 두 번만 완료되어 다른 3회 결과보다 신뢰 구간이 약하다.

## 고정한 실험 계약

- 모델: `nvidia/Cosmos-Reason2-2B`
- Current engine: `.local/profile-local-prefill-20260827/engine-p128-v1024-kv256-tied`
- vision engine: `.local/atomic-packed-vision-runtime-20260824/direct-visual-max2048/visual`
- Current: independent TensorRT E/P/D execution contexts, P8/D64, stable indexed KV slots
- packed-prefill chunk: **항상 128**
- requested active capacity: 80; decode-aligned effective capacity: 64
- HTTP/client concurrency: balanced, decode-heavy, long-prefill, bimodal은 64; 나머지는 trace 계약을 유지
- E batch capacity: 4, maximum encoded requests: 16
- global scheduler: profile-free active mode
- common decode service target: 45 ms
- vision first-token target: 500 ms
- primary benchmark: 외부 per-dispatch JSON telemetry off, 내부 CUDA event 학습 유지
- CUDA graph: off
- balanced text trace에서도 동일 VLM runtime footprint를 비교하기 위해 Current vision engine을 적재
- vLLM: workload 계약이 변한 C64 trace만 fresh process로 3회 재실행하고, 계약이 바뀌지 않은 trace는 기존 fresh 결과를 재사용

Balanced trace SHA-256은 양쪽 모두 `290d34061a173c13440e10247f55bd442a131e525a40c30a52d0b39a549d6538`이다.

## 구현한 공통 mechanism

### 1. Decode-aligned admission

요청 capacity를 임의 workload별 상수로 고르지 않고 decode engine의 실제 batch capacity에 맞는 완전한 cohort로 내린다.

```text
requested capacity 80, D64  -> effective 64
requested capacity 80, D32  -> effective 64
requested capacity 80, D16  -> effective 80
requested <= D capacity     -> requested 유지
```

이 변경은 D64에서 `64 + 16`처럼 작은 residual cohort가 계속 생성되는 것을 막는다. 메모리 자체를 덜 쓰기 위한 정책이 아니라, stable KV lease admission과 decode service shape를 맞춰 tail fragmentation을 줄이는 mechanism이다. 필요하면 `TRT_EDGELLM_DISABLE_DECODE_ALIGNED_ADMISSION`으로 분리 A/B할 수 있다.

구현 위치:

- `.local/upstream-v010/cpp/runtime/scheduling/independentPhaseAsyncServer.{h,cpp}`
- `.local/upstream-v010/examples/llm/llm_phase_context_smoke.cpp`

### 2. Full prefill-cohort ingress

이전 IPC ingress는 P capacity의 절반만 공개하여 P8 엔진에 한 번에 네 요청만 scheduler snapshot에 보이는 경우가 있었다. 이제 parsing은 계속 비동기이지만 각 arbitration boundary에서 최대 한 개의 완전한 P cohort를 공개한다.

```text
pending HTTP requests
        |
        | async parse / image preparation
        v
ingress staging queue
        |
        | expose min(pending, P capacity=8)
        v
global E/P/D scheduler snapshot
```

이는 adaptive chunk 크기나 workload label이 아니다. P engine이 지원하는 batch shape를 candidate generator가 실제로 볼 수 있게 하는 mechanism parity 수정이다.

### 3. Native completion callback

PhaseThreeCoordinator 사용 시 token/completion이 중간 polling queue에 한 번 더 들어가던 경로를 제거했다. coordinator가 completion lifetime cleanup을 먼저 수행한 뒤 server callback으로 직접 전달한다.

```text
GPU completion event
        |
        v
PhaseThreeCoordinator lifetime cleanup
  - request ID
  - per-request TPOT target
  - downstream vision ownership metadata
        |
        v
native server callback -> HTTP response broker
```

이로써 text-only와 VLM이 같은 completion mechanism을 사용하며, vision engine을 적재한 text trace도 polling overhead를 지지 않는다.

구현 위치:

- `.local/upstream-v010/cpp/runtime/scheduling/phaseThreeCoordinator.{h,cpp}`
- `.local/upstream-v010/examples/llm/llm_phase_context_smoke.cpp`

### 4. Scheduler preview copy-on-write와 vacuous decision elision

Global candidate preview는 scheduler mechanism state를 정확히 복제해야 하지만, cost sample window와 최근 TPOT/decode GPU sample은 preview에서 읽기만 한다. 이 큰 history를 매 후보마다 deep copy하지 않고 공유하고, 실제 관측이 변경할 때만 detach한다.

또한 local runnable phase가 P 또는 D 하나뿐이고 admission에서 memory ownership을 이미 예약했다면, 선택지가 하나인 Global policy comparison만 생략한다. batch formation, canonical ordering, CUDA cost observation은 그대로 통과한다.

이는 policy authority를 우회하지 않는다. E/P/D 외부 action이 outstanding이거나 memory가 사전 예약되지 않은 경우에는 feasibility/global decision 경로를 그대로 사용한다.

구현 위치:

- `.local/upstream-v010/cpp/runtime/scheduling/phaseGlobalScheduler.{h,cpp}`
- `.local/upstream-v010/cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}`

### 5. Incremental WAIT 비교 수정

WAIT preview가 별도의 Global selector를 다시 호출해 현재 D batch를 만드는 대신, 동일 snapshot의 decode mechanism이 선택한 batch size와 canonical rows를 직접 사용한다. 따라서 비교 대상은 policy가 재구성한 근사 batch가 아니라 실제 `NOW` dispatch가 된다.

```text
NOW  = D(selected rows) + residual drain
WAIT = event wait + D(future rows) + future residual drain
```

`future total work`와 `now single dispatch`를 비교하는 오류를 피하고, protected TPOT slack과 uncertainty guard는 유지한다. unknown overlap exploration은 production 기본값에서 끄고 명시적 warm-up에서만 허용한다.

### 6. Primary telemetry와 HTTP transport 비용 분리

- per-dispatch kernel-group JSON은 `--emit-phase-metrics`일 때만 출력한다.
- primary 성능 측정에서는 JSON formatting/file I/O를 제거하지만 CUDA event와 내부 online cost learning은 유지한다.
- HTTP SSE는 별도 타이머를 만들지 않고 이미 준비된 동일 요청 token만 non-blocking으로 최대 64개 coalesce한다.
- terminal/error event를 다음 loop로 정확히 넘겨 token ordering과 completion ordering을 보존한다.
- benchmark summary에 TTFT/TPOT/E2E mean/median/p95를 모두 저장한다.

루트 구현 위치:

- `scripts/cosmos_reason2/run_parallel_vision_adapter_gate.py`
- `scripts/cosmos_reason2/run_phase_openai_gateway.py`
- `scripts/cosmos_reason2/run_phase_http_trace_bench.py`
- `scripts/cosmos_reason2/run_vllm_fresh_trace_suite.py`
- `scripts/cosmos_reason2/run_vllm_trace_bench.py`

## 12-workload throughput 결과

`Previous`는 이번 공통 mechanism 변경 직전의 profile-free Global 구현이다. `Final`은 두 번의 전체 회귀와 한 번의 balanced 최종 재확인을 합친 3회 중앙값이다. Balanced Final은 별도 최종 production 3회 결과를 사용했다.

| workload | Previous Current | Final Current | Final 변화 | vLLM | Final vs vLLM |
|---|---:|---:|---:|---:|---:|
| short | 2,470.33 | 2,476.78 | +0.26% | 1,983.53 | **+24.87%** |
| balanced | 3,928.89 | 4,562.68 | **+16.13%** | 4,333.52 | **+5.29%** |
| decode-heavy | 4,918.65 | 5,269.36 | **+7.13%** | 4,965.43 | **+6.12%** |
| long-prefill | 1,168.45 | 1,196.54 | +2.40% | 1,130.44 | **+5.85%** |
| bimodal | 1,905.12 | 1,945.12 | +2.10% | 1,840.15 | **+5.70%** |
| text-heavy | 1,955.05 | 1,957.84 | +0.14% | 1,634.76 | **+19.76%** |
| mixed | 1,125.80 | 1,137.66 | +1.05% | 921.48 | **+23.46%** |
| vision-heavy | 682.04 | 693.31 | +1.65% | 579.20 | **+19.70%** |
| poisson | 1,954.64 | 1,972.08 | +0.89% | 1,800.07 | **+9.56%** |
| wave/drain | 97.76 | 96.78 | -1.00% | 95.85 | **+0.97%** |
| multi-image | 284.78 | 292.78 | +2.81% | 244.52 | **+19.74%** |
| late-vision D24 | 2,502.38 | 2,544.23 | +1.67% | 2,359.23 | **+7.84%** |

## TTFT 전체 결과

단위는 ms이며 각 칸은 `mean / median / p95`이다.

| workload | Previous Current | Final Current | vLLM |
|---|---:|---:|---:|
| short | 83.53 / 68.03 / 170.92 | 88.02 / 83.20 / 172.32 | 174.92 / 195.47 / 263.97 |
| balanced | 63.07 / 29.70 / 206.96 | **64.67 / 47.92 / 165.07** | 112.97 / 96.48 / 280.57 |
| decode-heavy | 329.03 / 291.62 / 790.48 | **66.04 / 52.74 / 174.42** | 117.82 / 98.55 / 315.40 |
| long-prefill | 3,055.51 / 3,539.18 / 4,035.32 | 2,037.65 / 2,227.78 / 2,839.89 | 1,911.14 / 1,903.09 / 2,885.78 |
| bimodal | 2,752.83 / 3,278.23 / 5,768.37 | 1,888.18 / 2,422.77 / 3,812.51 | 1,566.84 / 1,605.17 / 2,641.77 |
| text-heavy | 319.96 / 163.91 / 1,068.85 | 321.60 / 166.67 / 1,107.13 | 421.58 / 308.32 / 1,232.20 |
| mixed | 727.92 / 259.88 / 2,185.28 | 706.53 / 255.17 / 2,089.07 | 874.56 / 270.36 / 2,541.43 |
| vision-heavy | 1,406.72 / 1,198.27 / 3,245.17 | 1,374.90 / 1,161.51 / 3,116.65 | 1,710.70 / 1,433.00 / 3,691.37 |
| poisson | 193.19 / 80.25 / 758.27 | 197.90 / 82.61 / 742.34 | 438.11 / 323.25 / 902.68 |
| wave/drain | 208.47 / 208.91 / 306.04 | 232.28 / 227.54 / 356.37 | 252.76 / 229.66 / 418.60 |
| multi-image | 232.83 / 225.98 / 344.63 | 213.28 / 218.91 / 333.15 | 259.81 / 229.49 / 402.58 |
| late-vision D24 | 115.43 / 44.67 / 454.76 | 114.00 / 43.59 / 455.70 | 153.40 / 61.38 / 631.51 |

## TPOT 전체 결과

단위는 ms이며 각 칸은 `mean / median / p95`이다.

| workload | Previous Current | Final Current | vLLM |
|---|---:|---:|---:|
| short | 13.83 / 11.68 / 26.98 | 13.34 / 11.26 / 27.26 | 13.36 / 12.07 / 24.88 |
| balanced | 18.61 / 18.33 / 26.63 | **12.07 / 12.51 / 13.37** | 12.13 / 12.53 / 13.52 |
| decode-heavy | 13.19 / 13.09 / 17.08 | **10.56 / 10.80 / 11.16** | 10.96 / 11.22 / 11.56 |
| long-prefill | 27.35 / 28.67 / 32.34 | 26.64 / 27.73 / 31.08 | 32.11 / 33.27 / 37.05 |
| bimodal | 17.98 / 16.61 / 29.72 | 18.03 / 16.43 / 29.61 | 23.10 / 21.50 / 37.08 |
| text-heavy | 25.33 / 24.55 / 39.42 | 25.22 / 24.72 / 38.82 | 29.21 / 29.66 / 47.36 |
| mixed | 31.32 / 35.84 / 41.96 | 32.15 / 36.24 / 41.75 | 46.97 / 47.38 / 84.02 |
| vision-heavy | 27.29 / 29.14 / 37.70 | 28.25 / 30.77 / 37.08 | 63.70 / 65.13 / 119.58 |
| poisson | 21.99 / 18.88 / 41.07 | 21.75 / 18.72 / 40.52 | 22.19 / 18.32 / 45.67 |
| wave/drain | 9.67 / 9.62 / 11.59 | 9.68 / 9.54 / 12.11 | 12.43 / 13.17 / 17.26 |
| multi-image | 9.72 / 9.52 / 13.06 | 9.53 / 9.73 / 12.81 | 12.42 / 13.21 / 16.32 |
| late-vision D24 | 9.43 / 9.41 / 9.49 | 9.28 / 9.26 / 9.35 | 9.90 / 9.92 / 9.93 |

## E2E 전체 결과

단위는 ms이며 각 칸은 `mean / median / p95`이다.

| workload | Previous Current | Final Current | vLLM |
|---|---:|---:|---:|
| short | 333.32 / 343.13 / 414.98 | 334.40 / 342.94 / 413.99 | 426.71 / 440.37 / 503.71 |
| balanced | 1,590.89 / 1,491.99 / 2,569.53 | **1,094.43 / 1,170.23 / 1,693.42** | 1,149.78 / 1,234.82 / 1,766.13 |
| decode-heavy | 3,671.25 / 3,679.56 / 5,626.48 | **2,795.61 / 3,091.93 / 4,279.01** | 2,956.65 / 3,278.47 / 4,467.25 |
| long-prefill | 5,408.73 / 5,615.89 / 7,739.81 | 4,334.81 / 4,335.12 / 6,115.20 | 4,638.66 / 4,648.76 / 6,590.08 |
| bimodal | 5,236.57 / 5,207.72 / 10,853.34 | 4,347.45 / 4,452.92 / 8,917.02 | 4,722.74 / 3,976.43 / 9,363.54 |
| text-heavy | 1,621.06 / 1,679.61 / 1,723.80 | 1,624.48 / 1,676.81 / 1,721.94 | 1,943.42 / 2,017.39 / 2,037.81 |
| mixed | 2,229.02 / 2,495.77 / 2,531.51 | 2,231.05 / 2,490.74 / 2,511.94 | 3,008.38 / 2,924.99 / 3,140.85 |
| vision-heavy | 2,493.93 / 2,422.54 / 3,541.81 | 2,480.02 / 2,386.15 / 3,493.79 | 4,119.14 / 4,107.52 / 4,229.37 |
| poisson | 1,597.56 / 1,562.54 / 2,048.55 | 1,582.29 / 1,546.64 / 2,016.07 | 1,800.22 / 1,757.78 / 2,266.55 |
| wave/drain | 508.35 / 507.13 / 523.54 | 532.77 / 520.62 / 581.55 | 637.86 / 637.72 / 649.42 |
| multi-image | 534.04 / 530.08 / 558.85 | 515.87 / 508.65 / 544.98 | 644.30 / 640.41 / 653.90 |
| late-vision D24 | 1,465.45 / 1,840.23 / 1,842.47 | 1,443.70 / 1,810.55 / 1,812.81 | 1,576.03 / 1,950.80 / 1,954.46 |

## 원인 분석

### Balanced가 교차한 이유

Balanced의 큰 변화는 P chunk를 바꾸거나 balanced 전용 knob를 추가해서 나온 것이 아니다.

1. D64와 admission64가 일치해 decode residual tail을 제거했다.
2. P8 전체 cohort가 한 decision snapshot에 보이므로 packed prefill formation을 회복했다.
3. 단일 runnable phase에서 결과가 정해진 selector preview/deep-copy 비용을 제거했다.
4. native completion과 SSE ready-token coalescing으로 host response path의 작은 반복 비용을 줄였다.
5. primary run에서 외부 JSON telemetry를 꺼 측정 코드가 serving 결과를 오염하지 않게 했다.

그 결과 이전 Current 대비 balanced 처리량은 `+16.13%`, TPOT p95는 `26.63 -> 13.37 ms`, E2E p95는 `2,569.53 -> 1,693.42 ms`로 좋아졌다.

### 다른 workload가 유지된 이유

- decode-heavy는 완전한 D64 cohort 효과를 직접 받아 처리량 `+7.13%`, TTFT p95 `790.48 -> 174.42 ms`로 개선됐다.
- long-prefill은 P8 ingress와 mechanism-equivalent batch formation으로 처리량 `+2.40%`, TTFT p95 `4,035.32 -> 2,839.89 ms`로 개선됐다.
- bimodal은 throughput과 tail은 좋아졌지만 vLLM보다 TTFT와 E2E median이 아직 길다. 이는 average GPU efficiency보다 request ordering/fairness 개선이 다음 과제임을 뜻한다.
- VLM mixed/vision-heavy/multi-image는 P128과 vision ownership을 그대로 두었고 throughput을 유지하거나 개선했다.
- wave/drain은 유일한 이전 Current 대비 throughput 회귀지만 `-1.00%`로 보존 gate 안이며 vLLM보다는 `+0.97%` 높다.

### Wave/drain의 남은 tail

profiling run에서 E formation은 양쪽 모두 `E1 x4 + E4 x4`, 총 8 batch/20 requests, median batch 2.5였다. 즉 더 큰 E batch를 기다려 얻은 이득이 아니라, 각 5-request wave의 마지막 singleton E가 ready P/D 뒤에서 약 180--225 ms 기다리는 것이 tail 원인이다.

vision TTFT target을 500 ms에서 350 ms로 내린 A/B는 wave를 개선하지 못했고 VLM exact token hash 변동까지 만들었다. phase-normalized ingress도 wave를 개선하지 못하고 multi-image를 악화시켰다. 두 변경은 모두 되돌렸으며 최종 production 설정에는 없다.

## 기각한 실험

| 실험 | 관측 | 결정 |
|---|---|---|
| phase-normalized ingress | wave 개선 없음, multi-image 손실 | 제거 |
| vision TTFT target 350 ms | wave tail 개선 없음, exact hash 불안정 | 500 ms 유지 |
| direct CUDA graph priming/variant split | balanced 약 4,464 tok/s, TPOT p95 약 14.00 ms로 최종 eager보다 나쁨 | production 패치에서 제거, graph off |
| 외부 per-dispatch metrics 상시 출력 | profiling에는 유용하나 primary host path 오염 | opt-in으로 전환 |

## Correctness와 테스트

- Current balanced 3회 exact token identity 통과
- 전체 Current regression 반복에서 exact deterministic output 확인
- C++ focused tests: 4 suites, 182 tests 통과
- `llm_phase_context_smoke`와 `unitTest` build 통과
- Python adapter/gateway tests: 18개 통과
- gateway ready-token coalescing unit tests 추가
- 변경 Python scripts `py_compile` 통과
- CUDA sanitizer 전체 재실행은 이번 성능 gate에서 하지 않았으며, stable KV ownership 자체는 이전 indexed/paged correctness gate 결과를 사용했다.

## 결과 위치

- 최종 balanced Current 3회: `.local/balanced-vllm-crossover-20260828/r59-final-balanced-production-3x`
- fresh balanced vLLM C64 3회: `.local/balanced-vllm-crossover-20260828/r39-vllm-fresh-c64-3x`
- fresh decode-heavy/long-prefill/bimodal vLLM C64 3회: `.local/balanced-vllm-crossover-20260828/r58-vllm-fresh-c64-3x`
- Current 12-workload 회귀 1회+2회: `.local/balanced-vllm-crossover-20260828/r53-production-regression-11x1`, `.local/balanced-vllm-crossover-20260828/r54-production-regression-11x2`
- wave kernel-group profile: `.local/balanced-vllm-crossover-20260828/r56-wave-profile-metrics`
- 기각한 target350 A/B: `.local/balanced-vllm-crossover-20260828/r57-vision-ttft350-wave-multi-3x`

## 다음 우선순위

1. Bimodal request ordering을 workload label 없이 oldest protected slack과 canonical cohort ordering으로 개선한다.
2. Short D의 TPOT p95를 악화시키지 않는 bounded small-D refill을 별도 correctness/performance gate로 검증한다.
3. Wave singleton E는 E batch target 변경이 아니라 outstanding P/D의 predicted remaining time을 first-token critical path에 포함해 해결한다.
4. 각 변경은 balanced crossover와 12-workload `-3%` preservation gate를 동시에 통과해야 promotion한다.
5. CUDA graph는 exact shape hit rate와 capture lifecycle이 production request path 밖에서 보장되기 전에는 다시 기본 활성화하지 않는다.

핵심은 workload마다 fine-tuning하는 것이 아니다. 동일한 admission, ownership, candidate formation, measured service cost, request slack으로 현재 ready state를 처리하고, balanced를 포함한 교차 workload gate로만 promotion 여부를 결정한다.
