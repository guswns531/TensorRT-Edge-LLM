# Controlled P+D overlap saturation sweep

## 1. 질문

기존 real-request 결과만 보면 P+D overlap이 거의 발생하지 않았고, 관측된 한 shape는 serial보다 느렸다.
그러나 이는 이미 batching과 queue formation이 잘 된 production workload만 본 결과일 수 있다. 따라서 다음 두 질문을
분리해서 확인했다.

1. 두 TensorRT context에 동시에 실행할 준비가 된 P와 D가 계속 존재한다면 GPU 수준 overlap은 이득인가?
2. 실제 request trace에서 scheduler가 만드는 overlap 비율이 증가할 때 E2E 성능은 어떻게 변하는가?

첫 질문은 request arrival, SLO slack, admission, page readiness를 제거한 controlled engine sweep으로 측정했다. 두 번째
질문은 기존 balanced/long-prefill real-request trace에서 safe probe 빈도를 강제로 높여 screening했다.

## 2. 측정 정의

서로 다른 값을 혼동하지 않는다.

- `requested_pairs`: 전체 P+D pair 중 두 context를 같은 CUDA event gate에서 시작한 pair의 비율
- `observed_pairs`: 실제로 concurrent dispatch한 pair의 비율
- `active_wall_overlap`: 전체 pair makespan 중 P와 D CUDA event interval의 교집합이 차지한 비율
- `real-trace EPD overlap`: 실제 request trace의 E/P/D active wall time 중 둘 이상이 동시에 active인 시간
- `speedup`: 같은 controlled series의 0% pair-overlap makespan을 해당 point의 makespan으로 나눈 값

`requested_pairs=100%`라도 `active_wall_overlap=100%`는 아니다. 긴 phase가 짧은 phase보다 더 오래 실행되므로 두
interval의 교집합만 overlap으로 센다.

## 3. 구현

`examples/llm/llm_phase_context_smoke.cpp`에 opt-in 진단을 추가했다.

- `TRT_EDGELLM_PHASE_OVERLAP_SWEEP=1`
  - 실제 prefill/decode TensorRT execution context를 사용한다.
  - 20회 warmup 후 point당 100 pair를 측정한다.
  - requested pair 비율을 `0/25/50/75/100%`로 고정한다.
  - CUDA event로 P start/end, D start/end, pair makespan을 직접 측정한다.
  - sweep만 실행하고 unrelated semantic smoke는 건너뛴다.
- `TRT_EDGELLM_PHASE_OVERLAP_SWEEP_DECODE_BATCH=N`
  - `1 <= N <= engine maxDecodeBatchSize` 범위에서 D batch를 선택한다.
  - stable indexed page lease와 실제 KV binding을 N row에 맞게 준비한다.

production 기본값에는 영향이 없다. 첫 환경 변수가 없으면 기존 smoke 및 server 경로가 그대로 실행된다.

## 4. 공통 환경

- GPU: RTX 3080 10 GiB, SM86
- runtime image: `nvcr.io/nvidia/tensorrt:26.06-py3`
- TensorRT: 11.0.0
- CUDA: 13.3 container runtime
- model: `nvidia/Cosmos-Reason2-2B`, FP16
- engine: max batch 80, P8/D64, packed P128, paged stable KV, 256 pages
- controlled P: packed initial P2, total 128 tokens (`96 + 32`)
- controlled D: past/context length 128, batch `1/8/32/64`
- contexts: one CUDA context, independent P and D TensorRT execution contexts and CUDA streams

## 5. D1 반복 측정

D1은 전체 sweep를 3회 실행했다. 아래는 각 point의 3회 평균이다.

| requested pair overlap | active wall overlap | mean makespan | speedup | makespan 감소 |
|---:|---:|---:|---:|---:|
| 0% | 0.000% | 16.2770 ms | 1.000x | 0.00% |
| 25% | 13.157% | 15.6466 ms | 1.040x | 3.87% |
| 50% | 27.403% | 15.0373 ms | 1.082x | 7.62% |
| 75% | 42.908% | 14.4103 ms | 1.130x | 11.47% |
| 100% | 59.684% | 13.8018 ms | 1.179x | 15.21% |

세 반복에서 100% point의 makespan은 `13.8093/13.7844/13.8117 ms`였다. 결과가 안정적이며 overlap
비율에 따라 거의 단조롭게 좋아졌다.

그러나 개별 phase는 느려졌다.

- serial P: 약 10.316 ms
- fully overlapped P: 약 13.798 ms, 약 +33.8%
- serial D: 약 5.960 ms
- fully overlapped D: 약 8.237 ms, 약 +38.2%

즉 overlap은 개별 request의 phase latency를 줄인 것이 아니다. 두 phase의 합 `P + D`를 하나의 wall-clock
makespan으로 압축해 pair throughput을 높였다.

## 6. decode batch별 saturation curve

각 D batch에서 같은 P shape와 0/25/50/75/100% dispatch pattern을 사용했다. D1은 3회 평균, 나머지는
동일 process 조건의 screening 1회다.

### 6.1 Makespan

| D batch | 0% | 25% | 50% | 75% | 100% |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.2770 | 15.6466 | 15.0373 | 14.4103 | 13.8018 |
| 8 | 16.4440 | 15.8252 | 15.2085 | 14.5919 | 13.9667 |
| 32 | 17.4826 | 16.7728 | 16.0880 | 15.3935 | 14.7090 |
| 64 | 18.6112 | 17.7942 | 16.9819 | 16.1666 | 15.3608 |

단위는 ms/pair다.

### 6.2 0% 대비 speedup

| D batch | 25% | 50% | 75% | 100% |
|---:|---:|---:|---:|---:|
| 1 | 1.040x | 1.082x | 1.130x | 1.179x |
| 8 | 1.039x | 1.081x | 1.127x | 1.177x |
| 32 | 1.042x | 1.087x | 1.136x | 1.189x |
| 64 | 1.046x | 1.096x | 1.151x | 1.212x |

### 6.3 Active wall overlap

| D batch | 25% | 50% | 75% | 100% |
|---:|---:|---:|---:|---:|
| 1 | 13.157% | 27.403% | 42.908% | 59.684% |
| 8 | 13.440% | 27.974% | 43.715% | 60.896% |
| 32 | 14.411% | 30.056% | 47.190% | 65.808% |
| 64 | 15.564% | 32.673% | 51.414% | 72.207% |

D64에서는 serial P/D가 각각 `10.310/8.301 ms`였고 fully overlapped P/D는 `15.358/11.092 ms`였다.
P는 약 49%, D는 약 34% 느려졌지만 pair makespan은 `18.611 -> 15.361 ms`, 즉 17.5% 감소했다.

## 7. Real-request overlap-ratio screening

production scheduler의 cost/slack 선택을 그대로 둔 경우와, unknown P+D probe를 더 자주 허용한 경우를 같은 trace로
비교했다. 이는 각 variant 1회의 screening이며 promotion용 x3 gate가 아니다.

### 7.1 Balanced trace

| variant | tok/s | TTFT p95 | TPOT p95 | E2E p95 | real EPD overlap | P+D selections |
|---|---:|---:|---:|---:|---:|---:|
| production | 4,398.98 | 164.41 | 14.00 | 1,762.27 | 0.044% | 0 |
| serial | 4,442.09 | 163.35 | 13.74 | 1,736.99 | 0.038% | 0 |
| forced interval 32 | 4,367.50 | 166.49 | 13.99 | 1,786.38 | 2.350% | 17 |
| forced interval 8 | 4,341.65 | 168.16 | 14.12 | 1,791.08 | 1.725% | 17 |
| forced interval 2 | 4,395.67 | 165.19 | 13.92 | 1,762.45 | 2.429% | 28 |
| forced interval 1 | 4,341.04 | 165.38 | 14.34 | 1,825.39 | 2.115% | 31 |

cached fresh vLLM balanced는 `4,333.52 tok/s`다. 같은 workload를 다시 실행하지 않았으며 위 screening point들은
모두 cached vLLM보다 약 `+0.17% ~ +2.51%`다. current production x3 결과는 `4,475.67 tok/s`였으므로 이번
single-run screening의 절대값을 promotion 숫자로 사용하지 않는다.

interval은 overlap 비율을 직접 지정하는 knob가 아니다. decision sequence 기준 cooldown이므로 queue readiness와 phase
duration에 따라 실제 overlap 비율이 비단조적이다.

### 7.2 Long-prefill trace

| variant | tok/s | TTFT p95 | TPOT p95 | E2E p95 | real EPD overlap | P+D selections |
|---|---:|---:|---:|---:|---:|---:|
| serial | 1,254.22 | 2,574.16 | 28.68 | 5,792.14 | 0.000% | 0 |
| forced interval 32 | 1,251.03 | 2,692.38 | 28.98 | 5,806.29 | 0.058% | 2 |
| forced interval 8 | 1,221.54 | 2,874.72 | 30.25 | 6,084.87 | 0.028% | 2 |
| forced interval 2 | 1,235.79 | 2,780.34 | 29.47 | 6,018.51 | 0.101% | 4 |
| forced interval 1 | 1,189.80 | 2,898.34 | 32.03 | 6,314.81 | 0.060% | 2 |

cached fresh vLLM long-prefill은 `1,130.44 tok/s`다. long-prefill에서는 P duty가 높아도 page admission과 phase
readiness 때문에 P와 D가 동시에 runnable인 기회가 거의 없었다. probe 빈도를 높이는 것만으로 overlap을 만들 수 없다.

## 8. 왜 controlled와 real request의 결론이 다른가

controlled sweep의 이득 난 shape와 real trace의 손해 난 shape는 같지 않다.

### Controlled favorable shape

- initial packed P2, total 128 tokens
- P past KV 0
- D context 128
- P와 D가 매 pair마다 동시에 ready
- host queue/admission/wait가 없음

### 이전 real trace의 대표 unprofitable shape

- P1 continuation 계열
- D47 수준의 큰 active cohort
- 더 큰 past-KV/context bucket
- isolated work 약 `9.02 ms`
- observed overlap makespan 약 `10.47 ms`

따라서 `enable_overlap=true` 같은 전역 결론은 틀리다. 다음 key별로 직접 관측해야 한다.

```text
(P class,
 P rows,
 P useful tokens,
 P past-KV bucket,
 D rows,
 D context bucket,
 execution variant)
```

같은 D batch라도 initial/continuation P, context length, CUDA graph/eager variant에 따라 SM, memory bandwidth,
workspace contention이 달라진다.

## 9. 결론

### 확인된 사실

1. Independent TensorRT contexts는 실제 Cosmos 커널에서 유효한 동시 실행을 만든다.
2. favorable short-context P+D shape에서는 overlap 비율이 높아질수록 pair makespan이 단조롭게 감소했다.
3. 이번 범위의 최대 이득은 P128+D64, 100% pair overlap의 약 `1.212x`였다.
4. 그 과정에서 개별 P/D latency는 30~50% 늘 수 있다. throughput 이득과 request SLO는 별도로 보호해야 한다.
5. production trace에서 overlap이 낮은 주원인은 CUDA가 overlap을 못 하는 것이 아니라 simultaneous-ready 기회와 shape별
   수익성이 제한되기 때문이다.
6. 실제 trace에서 무작정 probe를 늘리면 balanced/long-prefill 모두 성능이 나빠졌다.

### Scheduler 의미

목표는 overlap ratio 최대화가 아니다.

```text
select P+D only if
    measured overlap makespan < serial residual horizon
and per-request phase slowdown <= protected slack
and memory/context feasibility holds
```

즉 high-overlap favorable shape는 적극적으로 사용하되, continuation P + long-context D처럼 직접 관측상 손해인 shape는
serial로 남겨야 한다. process-local CUDA event cost model이 shape key를 사용하는 이유가 이 결과로 확인됐다.

## 10. 다음 실험

1. actual request trace에 short-context D cohort를 먼저 resident하게 만들고, short initial P2가 계속 도착하는
   `overlap-friendly` trace를 추가한다.
2. 같은 trace에서 production selector, forced 25/50/75/100% opportunity selector를 x3 비교한다.
3. throughput뿐 아니라 mean/p95 E2E, TTFT, TPOT와 request별 critical-path interference를 함께 본다.
4. continuation P와 D context `128/512/1024/1536`의 controlled table을 만든다.
5. measured beneficial bucket만 production overlap eligibility로 승격하고 12-workload gate를 다시 실행한다.

## 11. 검증

- `llm_phase_context_smoke` TensorRT 11/CUDA 13.3 build 통과
- D1 sweep 3회 완료
- D8/D32/D64 sweep 완료
- 최종 sweep-only D64 경로 exit code 0
- 진단 옵션 없는 production `TRT_EDGELLM_SEMANTIC_ONLY=1` Cosmos greedy semantic smoke 통과
- 모든 controlled point에서 actual requested pair count가 정확히 `0/25/50/75/100%`와 일치

