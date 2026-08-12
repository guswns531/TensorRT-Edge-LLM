SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Cosmos dynamic scheduler 구현 및 비교 결과

## 최종 결론

현재 권장 정책은 Cosmos-Reason2-2B FP16 indexed-paged engine에서 다음 조합이다.

```text
CUDA context:         one shared primary context
TensorRT contexts:    independent prefill/decode contexts
streams:              independent nonblocking prefill/decode streams
engine caps:          P4 / D64
prefill:              fixed chunk 128, batch-token budget 256
decode:               static cap D64
CUDA graph:           prefill 16 MiB, decode 128 MiB, global free reserve 256 MiB
KV:                   256 page bundles, 128 tokens/page, 64 stable slots
admission:            whole-request conservative page reservation
```

`dynamic decode + cost/SLO phase selector`는 구현과 단위 검증은 끝났지만 현 cost table에서는 이득이 없었다.
default로 승격하지 않고 opt-in 실험 경로로 남긴다.

## 구현 위치

| 기능 | 위치 |
| --- | --- |
| automatic CUDA graph cache와 global reserve | `cpp/runtime/exec/engineExecutor.{h,cpp}` |
| capture 실패 graph 정리 | `cpp/common/trtUtils.cpp` |
| fixed-128 token budget, dynamic decode, page-pressure policy | `cpp/runtime/scheduling/phaseQueueScheduler.{h,cpp}` |
| whole-request page reservation/backpressure | `cpp/runtime/scheduling/phaseContextServingFacade.{h,cpp}` |
| page 수 계산과 최종 physical pool 통계 | `cpp/runtime/hybridCacheManager.{h,cpp}` |
| 실제 CLI, CUDA-event telemetry, 자원 반환 assertion | `examples/llm/llm_phase_bench.cpp` |
| trace materialization과 결과 table | `scripts/cosmos_reason2/run_real_request_kv_matrix.py` |
| cost JSON 생성 | `scripts/cosmos_reason2/build_phase_scheduler_cost_model.py` |
| five-workload source 생성 | `scripts/cosmos_reason2/make_scheduler_workload.py` |
| vLLM 동일 trace 재생 | `scripts/cosmos_reason2/run_vllm_trace_bench.py` |

## 동작 구조

```text
request arrival
    |
    v
tokenize -> required pages = ceil((prompt + requested max output) / 128)
    |
    +-- stable slot과 page reservation 여유 있음 --> active prefill queue
    |
    +-- 하나라도 부족 ---------------------------> bounded pending queue
                                                      |
request finish/cancel <--- decode queue <--- prefill chunks
    |                         ^                |
    +-- physical pages 반환 --+                +-- physical page는 실제 경계에서만 lazy allocate
    +-- conservative reservation 반환
```

reservation과 allocation은 다르다. admission reservation은 미래 최악의 요청 길이를 보장하는 논리적 budget이고,
physical allocation은 실제 KV 길이가 128-token 경계를 넘을 때만 일어난다. 따라서 early EOS에서는 reserved-but-unused
internal fragmentation이 생기지만, 이미 admission된 요청이 decode 중 page-pool exhaustion으로 죽는 일은 막는다.

## Static 대비 정책 결과

모든 Current 수치는 같은 materialized trace의 단일 GPU run이다. short의 static/token256은 추가로 각각 3회 실행해
같은 방향을 확인했다.

| workload | policy | token/s | TTFT p95 ms | TPOT p95 ms | E2E p95 ms |
| --- | --- | ---: | ---: | ---: | ---: |
| short 48 | static | 1,257.81 | 657.54 | 10.36 | 799.38 |
| short 48 | token256 | 1,613.59 | 401.21 | 16.46 | 615.93 |
| short 48 | dynamic decode | 1,202.03 | 676.33 | 12.35 | 817.66 |
| short 48 | token256 + dynamic/phase | 1,449.44 | 461.97 | 16.61 | 674.08 |
| balanced 288 | static | 3,107.70 | 6,339.79 | 11.36 | 7,114.32 |
| balanced 288 | token256 | 3,648.15 | 5,162.55 | 16.71 | 6,093.25 |
| decode-heavy 288 | static | 4,336.24 | 9,585.71 | 13.39 | 12,147.15 |
| decode-heavy 288 | token256 | 4,372.12 | 9,456.53 | 13.35 | 11,928.92 |
| long-prefill 288 | static | 720.83 | 31,850.68 | 14.81 | 32,981.20 |
| long-prefill 288 | token256 | 911.78 | 24,284.45 | 18.64 | 25,887.12 |
| bimodal 288 | static | 1,355.31 | 24,332.51 | 14.36 | 26,405.75 |
| bimodal 288 | token256 | 1,495.71 | 21,319.80 | 17.82 | 24,292.48 |

token256은 다섯 workload 모두에서 throughput과 TTFT/E2E를 개선했다. 대신 short, balanced, long-prefill,
bimodal에서 TPOT을 악화시킨다. 처음 정의한 “어느 workload도 TPOT p95 10% 이상 악화 금지” gate는 통과하지
못한다. 그러므로 이것은 throughput/TTFT 우선 preset이지 universal default가 아니다. decode-heavy에서는
throughput +0.83%, TPOT -0.3%로 3% 회귀 gate를 통과한다.

## vLLM memory-matched 비교

vLLM 0.27.1은 FP16 weights/KV, raw KV 3.5 GiB, max sequences 80, token budget 8192, chunked prefill, CUDA graph,
prefix cache off로 실행했다. vLLM은 localhost HTTP streaming, Current는 direct C++ trace injection이므로 transport
경계는 Current에 유리하다. 아래 수치는 production-E2E 동등 비교가 아니라 현재 구현 간 관측 비교다.

| workload | Current token/s | vLLM token/s | gap | Current/vLLM TTFT p95 ms | Current/vLLM TPOT p95 ms | Current/vLLM E2E p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1,613.59 | 2,007.75 | -19.63% | 401.21 / 259.47 | 16.46 / 25.35 | 615.93 / 499.13 |
| balanced | 3,648.15 | 4,109.84 | -11.23% | 5,162.55 / 4,156.24 | 16.71 / 18.71 | 6,093.25 / 5,200.48 |
| decode-heavy | 4,372.12 | 4,411.88 | -0.90% | 9,456.53 / 8,640.49 | 13.35 / 17.01 | 11,928.92 / 11,648.91 |
| long-prefill | 911.78 | 1,095.39 | -16.76% | 24,284.45 / 19,629.57 | 18.64 / 40.45 | 25,887.12 / 21,366.47 |
| bimodal | 1,495.71 | 1,751.43 | -14.60% | 21,319.80 / 17,231.56 | 17.82 / 39.23 | 24,292.48 / 20,333.91 |

Current는 decode-heavy throughput이 사실상 동급이고 모든 workload에서 TPOT p95가 더 좋다. 반면 admission/prefill
formation 때문에 TTFT와 E2E는 여전히 vLLM이 좋다. 다음 최적화는 decode kernel 자체보다 conservative page
reservation, prefill profile/workspace, phase choice와 HTTP production surface에 집중해야 한다.

## CUDA graph 메모리 결과

decode-heavy token256에서 reserve sweep 결과는 다음과 같다.

| global reserve | token/s | free VRAM | cached graphs | 판정 |
| ---: | ---: | ---: | --- | --- |
| 0 MiB | 4,429.74 | 179.4 MiB | P4 / D30 | 가장 빠르나 headroom 부족 |
| 256 MiB | 4,372.12 | 299.4 MiB | P2 / D2 | 권장 안전점 |
| 384 MiB | 4,378.78 | 319.4 MiB | P0 / D0 | base footprint가 reserve보다 커 graph off |
| 512 MiB | 4,372.20 | 319.4 MiB | P0 / D0 | base footprint가 reserve보다 커 graph off |

256 MiB 정책은 unrestricted보다 약 1.3% 느리지만 free memory를 120 MiB 더 남긴다. graph-off base headroom 자체가
약 319 MiB라 384/512 MiB hard reserve는 현재 두-context engine footprint에서는 달성 불가능하다.

## Backpressure와 endurance

page-aware admission 전 long-prefill 288-request run은 `KV page-bundle pool is exhausted`로 abort했다. 수정 후 static,
token256, bimodal 네 run 모두 완료됐다. long-prefill에서는 253/288 요청, bimodal에서는 233/288 요청이 초기 pending이
되었고 observed reservation pressure는 1.0까지 올라갔다. 이는 page exhaustion을 숨긴 것이 아니라 실패를 bounded
queue wait으로 바꾼 결과다.

최종 short 1,024-request endurance 결과:

```text
completions:             1024 / 1024
generated throughput:   1,652.35 token/s
TTFT median / p95:      5,678.11 / 11,702.39 ms
TPOT median / p95:      17.28 / 18.15 ms
max pending depth:      898
max reserved bundles:   114 / 256
final stable slots:     64 / 64 available
final physical pages:   allocated 0, available 256 / 256
final CUDA free memory: 299.4 MiB
```

## 검증

- TensorRT 26.06 / CUDA 13.3 / SM86 release build 성공
- 관련 C++ test 46/46 성공
- selected pre-commit hooks 전체 성공
- Python compile 성공
- request별 Current static/token256 output 완전 일치
- CUDA graph capture/launch failure 0
- 288 long/mixed 및 1,024 endurance에서 slot/page leak 0

CUDA sanitizer와 final Nsight Systems “KV compaction/D2D 0” 재검증은 아직 남아 있다.

## 다음 구현 우선순위

1. `PhaseAsyncServerConfig`에 production policy와 page-reservation mode를 노출한다.
2. full max-output reservation 외에 configurable headroom/fraction과 bounded overcommit을 실험한다.
3. `Co(P,D)` overlap cost를 cost model에 넣고 observed P/D shape별로 phase action을 학습한다.
4. prefill/decode engine profile을 더 비대칭으로 축소해 1.6 GiB의 두 context workspace 비용을 줄인다.
5. 동일 HTTP/JSON streaming front end로 Current-vLLM production E2E를 다시 측정한다.
