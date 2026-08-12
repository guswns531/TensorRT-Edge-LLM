SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Cosmos 구현 및 vLLM 비교 계획

## 목표

현재 Cosmos-Reason2-2B FP16 경로는 긴 decode 포화 workload에서 vLLM과 비슷한 generated-token
throughput을 보이지만, 짧고 중간 output에서는 vLLM보다 1.7~2.9배 느리다. 다음 작업의 목표는 단일 최고
수치를 찾는 것이 아니라 차이를 다음 세 축으로 분해하고, 각 변경의 효과를 같은 materialized trace에서 증명하는
것이다.

1. 실행 구조: shared/sequential, independent/sequential, independent/overlap, CUDA graph
2. KV 구조: fixed-linear, indexed-linear, indexed-paged
3. scheduling: static cap, queue-depth, token-budget, measured-cost/SLO policy

Cosmos text-only를 주 실험 모델로 고정한다. 공통 scheduler와 cache API는 모델 중립으로 구현하되, 모델별
attention/KV binding 검증은 별도 correctness gate로 유지한다.

## 현재 기준점

현재 권장점은 다음과 같다.

```text
model/cache: Cosmos-Reason2-2B FP16 / FP16 indexed-paged KV
execution:   one CUDA primary context, independent prefill/decode TRT contexts
scheduler:   fixed-128 chunk, P4/D64
graph:       prefill 16 MiB, decode 128 MiB, minimum charge 4 MiB
workload:    288 requests, requested output 96~384, 64 stable slots
result:      4,378.94 generated token/s
```

CUDA graph가 graph-off 대비 제공한 이득은 약 0.75%다. 따라서 짧은 output의 vLLM 격차는 launch overhead가
아니라 admission, prefill batch 형성, phase 선택, context/workspace 메모리에서 먼저 찾아야 한다.

## 측정 계약

### 두 종류의 공정 비교

서로 다른 경계를 한 표에 섞지 않는다.

| 비교 | 목적 | 경계 |
| --- | --- | --- |
| production E2E | 사용자가 실제로 얻는 결과 | JSON 수신부터 마지막 token까지, 양쪽 모두 동일한 transport |
| runtime core | CUDA/TRT scheduler 자체 비교 | 미리 tokenize/materialize한 요청이 runtime queue에 들어간 시점부터 |

현재 vLLM은 localhost HTTP이고 current는 direct C++ injection이므로 current에 유리한 차이가 있다. production
E2E 비교에서는 current async server에도 동일한 HTTP/JSON streaming front end를 사용한다. runtime-core 비교는
양쪽 모두 tokenizer와 transport를 제외할 수 있을 때만 별도 표로 제시한다.

### 고정 조건

- 같은 Cosmos checkpoint revision, FP16 weights, FP16 KV, greedy sampling, prefix cache off
- 동일한 materialized request ID, arrival timestamp, prompt, requested output limit
- warmup trace와 측정 trace 분리, 각 case 새 process에서 최소 3회
- GPU clock, temperature, driver, image digest, engine hash, commit을 manifest에 기록
- requested cap이 아니라 dispatch CSV의 observed P/D batch histogram을 결과로 사용
- actual generated tokens, EOS 차이, admission reject를 숨기지 않음
- throughput과 함께 TTFT/TPOT/E2E p50/p95/p99 및 SLO goodput을 항상 기록
- VRAM은 process peak와 `cudaMemGetInfo()` headroom을 함께 기록하고 측정 방식이 다른 값을 직접 빼지 않음

## Workload suite

고정 1,000 req/s burst만 사용하지 않고 prompt/output 구성과 offered load를 분리한다.

| 이름 | prompt tokens | requested output | 주로 확인할 병목 |
| --- | ---: | ---: | --- |
| interactive-short | 64~256 | 16~64 | admission, TTFT, 작은 decode batch |
| chat-balanced | 256~1024 | 64~256 | prefill/decode 균형 |
| long-prefill | 1024~2048 | 32~128 | chunk와 prefill blocking |
| decode-heavy | 128~512 | 256~512 | D48/D64 효율과 TPOT |
| bimodal-mixed | short/long 혼합 | 32~512 | head-of-line blocking과 fairness |

각 workload는 48-request smoke, 288-request steady run, 1,024-request endurance run을 갖는다. 먼저 단독 phase
cost로 sustainable token rate를 구한 후 offered load를 25%, 50%, 75%, 90%, 105%로 만든다. req/s만 같게 두면
prompt/output 분포가 달라질 때 GPU 부하가 달라지므로 주 비교축으로 사용하지 않는다. Poisson seed는 최소 세 개를
사용하고, 추가로 burst-on/burst-off trace를 둔다.

## Ablation ladder

모든 기능을 켠 current와 upstream/vLLM만 비교하면 원인을 알 수 없으므로 다음 순서로 누적 비교한다.

| ID | KV | TRT context/실행 | graph | scheduler | 질문 |
| --- | --- | --- | --- | --- | --- |
| A0 | upstream 방식 | clean upstream E2E | upstream default | upstream | 원본 기준은 얼마인가 |
| A1 | fixed-linear | shared/sequential | off | static | 기존 runtime 비용은 얼마인가 |
| A2 | indexed-linear | shared/sequential | off | static | compaction 제거 자체의 비용은 얼마인가 |
| A3 | indexed-paged | shared/sequential | off | static | page table과 메모리 절감의 비용/효과는 얼마인가 |
| A4 | indexed-paged | independent/sequential | off | static | context/workspace만 분리한 비용은 얼마인가 |
| A5 | indexed-paged | independent/overlap | off | static | 실제 overlap 순효과는 얼마인가 |
| A6 | indexed-paged | independent/overlap | 16/128 MiB | static P4/D64 | CUDA graph 순효과는 얼마인가 |
| A7 | indexed-paged | independent/overlap | 16/128 MiB | dynamic | scheduler 순효과는 얼마인가 |
| V0 | vLLM paged KV | production continuous batching | production | vLLM | 외부 기준점 |

A4와 A5는 반드시 같은 independent contexts, buffers, workspace를 사용한다. 단지 dispatch를 sequential 또는
overlap으로 바꿔 context 분리 비용과 overlap 이득을 혼동하지 않는다.

메모리 비교는 두 번 수행한다.

1. equal raw-KV capacity: 양쪽에 같은 token 수의 FP16 KV를 제공
2. equal process-VRAM budget: 같은 최대 VRAM 안에서 각 runtime이 수용 가능한 slot/token 수를 비교

## 구현 단계

### 1. 재측정과 안전 경계

먼저 current graph preset과 vLLM production을 위 workload의 short/balanced/decode-heavy 세 종류에서 정확히
재측정한다. 동시에 CUDA graph cache가 phase-local budget 이내여도 전체 GPU headroom을 소모하지 않도록 global
free-memory reserve를 추가한다.

구현 위치 후보:

- `cpp/runtime/exec/engineExecutor.{h,cpp}`: capture 직전 global free-memory floor 확인
- `examples/llm/llm_phase_bench.cpp`: 정책 CLI와 telemetry
- `scripts/cosmos_reason2/run_real_request_kv_matrix.py`: manifest/status 전달

초기 기본 reserve는 256 MiB로 실험하고, 256/384/512 MiB의 graph hit-rate와 성능을 비교한다. reserve 도달은
요청 실패가 아니라 graph capture 중단과 정상 `enqueueV3()` fallback이어야 한다.

### 2. Kernel-group cost model 완성

현재 CUDA event recorder를 scheduler 입력으로 사용할 수 있는 형태로 정규화한다.

```text
Cp(B, chunk, initial/continuation, prompt bucket)
Cd(B, KV-length bucket)
Co(P shape, D shape) = measured overlap makespan and contention
Cm(action) = extra pages, phase I/O, graph/workspace pressure
```

prefill은 B=1/2/4/8, fixed chunk=128을 먼저 측정한다. decode는 B=1/2/4/8/16/24/32/48/56/64와 KV
length bucket 128/512/1024/1536/2048을 측정한다. 전체 Cartesian product를 real trace로 반복하지 않고 단독
phase table로 후보를 줄인 뒤 P={1,2,4,8}, D={16,32,48,64} joint case만 측정한다.

cost table에는 median만이 아니라 p95, sample count, graph hit/fallback, overlap ratio를 저장한다. sample이 부족한
shape는 더 작은 인접 bucket의 보수적인 p95로 fallback한다.

### 3. Dynamic decode batching

가장 먼저 decode batch 선택만 동적으로 만든다. fixed P4/D64를 control로 유지한다.

- decode queue depth와 oldest TPOT slack을 읽음
- `Cd(B, length bucket)`에서 후보 B의 완료시간을 예측
- 더 큰 batch를 기다리는 이득보다 oldest request의 TPOT 손실이 커지면 즉시 dispatch
- page-pool pressure가 높으면 admission보다 기존 decode를 우선하여 page 회수를 앞당김
- 실제 미래 EOS 길이는 사용하지 않음. requested max-output은 명시적 hint 실험에서만 사용

비교 정책은 `static D64`, `queue-depth threshold`, `cost-aware deadline` 세 개다. 실제 output 길이를 미리 아는
oracle policy는 달성 가능한 상한만 보여주며 primary 결과로 사용하지 않는다.

### 4. Token-budget prefill batching

짧은 output 격차를 줄이는 핵심 단계다. 우선 chunk=128은 고정하여 batch formation 효과만 분리한다.

- 현재 queue front의 한 bucket만 보는 선택을 전체 queue의 compatible bucket 후보 검색으로 교체
- initial/continuation, chunk shape, model contract가 같은 row만 묶음
- row cap뿐 아니라 `prefill batch tokens <= budget`을 적용
- oldest TTFT deadline과 priority aging을 지키면서 가장 GPU 효율적인 compatible batch를 선택
- long-prefill 한 요청이 short prompt들을 막지 않도록 queue scan 범위를 제한하고 aging으로 starvation 방지

후보 budget은 128/256/512/1024 tokens다. 이 단계에서 64/256 adaptive chunk를 동시에 켜지 않는다. fixed-128
정책이 검증된 다음에만 chunk 64/128/256을 별도 ablation으로 추가한다.

### 5. Cost/SLO 기반 phase 선택

prefill과 decode batch builder가 안정된 후 phase 선택을 다음 세 action 중 고르게 한다.

```text
prefill only
decode only
prefill + decode overlap
```

각 action은 측정된 `Cp/Cd/Co`로 makespan을 예측하고 다음 점수를 최소화한다.

```text
predicted GPU time
+ TTFT deadline penalty
+ TPOT deadline penalty
+ page-pressure penalty
+ starvation/aging penalty
```

overlap은 `max(Cp, Cd)`로 가정하지 않고 반드시 측정된 `Co`를 사용한다. 두 phase가 같은 SM과 memory bandwidth를
경쟁하므로 overlap이 sequential보다 느린 shape도 정상적으로 거부해야 한다. EWMA online 보정은 offline table과
실측 차이가 일정 임계값을 넘을 때만 작은 범위에서 적용한다.

### 6. 메모리 및 context 최적화

scheduler 성능과 분리하여 다음을 하나씩 비교한다.

1. graph global reserve와 benefit-per-byte 기반 graph admission
2. prefill/decode phase-local I/O buffer의 실제 peak lifetime 축소
3. TensorRT context persistent/workspace 메모리 측정 및 phase-specific profile로 줄일 수 있는지 검증
4. indexed-paged internal fragmentation, 마지막 partial page, reserved-but-unused page 계측
5. page pressure에 따른 pending admission과 finished-request 우선 회수

두 independent context 사이에 workspace를 임의로 공유하지 않는다. 동시 실행 중 같은 workspace를 사용하면 data
race가 발생한다. TensorRT가 명시적으로 허용하는 allocator/lifetime 경계가 확인된 경우에만 실험한다.

### 7. Production surface와 회귀 검증

선택된 정책을 benchmark 전용 CLI에 남기지 않고 production async server config에 노출한다. 기본값은 기존 동작을
보존하며 dynamic policy는 opt-in으로 시작한다.

- request cancellation, timeout, overload, pending admission
- 1,024-request endurance와 repeated slot reuse
- CUDA sanitizer: OOB, use-after-release, invalid page/slot
- indexed mode에서 KV compaction/D2D copy가 없는지 Nsight Systems 확인
- graph on/off와 scheduler on/off의 request별 greedy output 완전 일치
- process restart를 포함한 세 번 이상의 재현성 검증

## 결과 판정 기준

### Correctness와 안정성

- current 내부 ablation은 request ID, output tokens, finish reason, output text가 정확히 일치
- vLLM과는 수치 경로가 다르므로 actual generated-token count와 EOS 차이를 별도 기록
- OOM, page exhaustion으로 인한 비정상 종료, slot/page leak, duplicate KV write 0건
- 1,024-request run 뒤 allocated page와 active slot이 모두 0으로 복귀

### 성능

새 scheduler는 static P4/D64 대비 다음 조건을 만족해야 채택한다.

- 다섯 workload 중 최소 세 개에서 throughput/TTFT/TPOT의 Pareto 개선
- decode-heavy throughput 회귀 3% 이내
- interactive-short와 chat-balanced의 TTFT p95 중 적어도 하나를 15% 이상 개선
- 어느 workload에서도 TPOT p95 또는 E2E p95를 10% 이상 악화시키지 않음
- overload에서는 단순 throughput보다 SLO goodput과 bounded queue growth를 우선

vLLM에 대한 1차 목표는 short/medium throughput 차이의 최소 25%를 줄이면서 output8x/12x의 동급 throughput을
유지하는 것이다. 이것은 완료 판정용 engineering target이며 현재 달성 결과가 아니다.

### 메모리

- graph-enabled 권장점에서 최소 256 MiB hard reserve 유지
- 동일 raw-KV capacity에서 current의 비-KV overhead를 구성 요소별로 설명 가능하게 계측
- equal-VRAM 비교에서 indexed-paged가 fixed/indexed-linear보다 더 많은 active tokens 또는 slots를 제공

## 권장 실행 순서

```text
exact current-vLLM rerun
  -> global memory reserve
  -> phase/joint cost table
  -> dynamic decode batching
  -> fixed-128 token-budget prefill
  -> cost/SLO phase selector
  -> optional chunk-size policy
  -> context/workspace memory reduction
  -> production async server integration
```

첫 구현 단위는 global memory reserve와 exact short/balanced/decode-heavy 재측정이다. 그 다음 dynamic decode만
적용하여 효과를 분리하고, 이후 token-budget prefill을 더한다. 이렇게 해야 short-output 개선이 batch formation,
phase selection, CUDA graph 중 어디서 발생했는지 설명할 수 있다.

## 2026-08-12 실행 결과

계획의 1~5단계와 7단계 안정성 일부를 구현하고 실제 GPU에서 검증했다. 상세 수치와 판정은
[Cosmos dynamic scheduler 구현 및 비교 결과](67-cosmos-dynamic-scheduler-results-20260812.md)에 기록한다.

- CUDA graph global reserve, phase-local graph budget, kernel-group cost JSON을 구현했다.
- fixed-128 prefill token budget, measured-cost decode batch 선택, SLO/page-pressure phase policy를 구현했다.
- 288-request long-prefill에서 발견한 page-pool exhaustion을 whole-request page reservation backpressure로 수정했다.
- 1,024-request 종료 뒤 stable slot과 page bundle이 모두 반환되는 것을 runtime assertion으로 확인했다.
- 현재 채택 후보는 `P4/D64 + fixed-128 + prefill token budget 256 + graph reserve 256 MiB`다.
- dynamic decode와 combined adaptive 정책은 현재 cost table/목표에서 static decode보다 느려 기본 정책으로 채택하지 않는다.
