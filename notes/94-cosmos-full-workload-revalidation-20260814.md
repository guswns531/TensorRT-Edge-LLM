SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Cosmos 전체 workload 재검증

## 결론

Cosmos-Reason2-2B의 Current phase runtime을 short, balanced, decode-heavy, long-prefill 네 workload에서
localhost HTTP/JSON/SSE까지 포함해 각각 세 번 새로 실행했다. 모든 Current 반복은 요청 실패가 없었고 workload별
생성 token 수도 세 번 모두 같았다. shared CUDA primary context 안에서 prefill/decode는 서로 다른 TensorRT
execution context와 non-blocking stream을 사용했다.

같은 trace hash와 HTTP client로 보존된 vLLM 3회 결과와 비교하면 Current generated token/s는 네 workload에서
각각 `+21.83%`, `+6.20%`, `+9.24%`, `+29.81%`다. balanced TPOT p95는 vLLM보다 0.206ms 느리고
decode-heavy TTFT median은 209ms 느리다. 그 외 주요 median/p95 latency는 Current가 더 낮다.

이번 결과는 workload routing의 필요성도 확인한다. short/balanced/decode-heavy는 128-token contract의 P8/D64
graph engine이 적합하다. long-prefill은 256-token contract의 P4/D64 graph-off engine이 적합하며, P8 x 256은
2,048 packed tokens가 되어 1,024-token engine profile을 초과하므로 유효한 설정이 아니다.

## 고정 조건

- GPU: GeForce RTX 3080 10GB, driver 610.43.02
- TensorRT container: `nvcr.io/nvidia/tensorrt:26.06-py3`, TensorRT 11.0.0
- model: `nvidia/Cosmos-Reason2-2B`, FP16 weights, FP16 KV
- KV: indexed-paged, 128-token page bundle, 256 bundles, 80 stable slots
- decoding: greedy, EOS enabled, prefix reuse disabled
- transport: localhost OpenAI-compatible chat-completions HTTP/SSE
- topology: one CUDA context, independent prefill/decode TensorRT contexts and streams
- repetitions: three fresh backend lifecycles per Current workload

| Workload | Trace SHA-256 | Requests | Prompt tokens | Requested output | Current output per run |
| --- | --- | ---: | ---: | ---: | ---: |
| short | `3f689bc3...294e459` | 48 | 4,312 | 1,040 | 1,040 |
| balanced | `290d3406...49d6538` | 288 | 25,872 | 24,960 | 23,712 |
| decode-heavy | `68f523e9...a349af` | 288 | 25,872 | 74,880 | 57,480 |
| long-prefill | `7d713d97...79886ec3` | 288 | 213,024 | 24,960 | 24,432 |

## HTTP E2E 결과

값은 완전한 세 번 실행의 중앙값이다. 변화율은 `(Current / vLLM - 1)`이며 latency는 음수가 개선이다.

| Workload | Current tok/s | vLLM tok/s | 변화 | Current req/s | vLLM req/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | **2,462.7** | 2,021.4 | **+21.83%** | **113.66** | 93.30 |
| balanced | **4,391.2** | 4,134.6 | **+6.20%** | **53.33** | 50.22 |
| decode-heavy | **4,845.6** | 4,435.8 | **+9.24%** | **24.28** | 22.42 |
| long-prefill | **1,422.0** | 1,095.4 | **+29.81%** | **16.76** | 13.01 |

decode-heavy와 long-prefill은 backend 간 EOS 위치가 조금 달랐다. Current/vLLM median output은 각각
57,480/57,110과 24,432/24,240이다. 따라서 token/s뿐 아니라 request/s와 latency도 함께 보고 판단해야 한다.

| Workload | TTFT Current med / p95 | TTFT vLLM med / p95 | Current p95 변화 |
| --- | ---: | ---: | ---: |
| short | 149.3 / **205.6ms** | **147.9** / 258.6ms | **-20.48%** |
| balanced | **1,648.9 / 3,853.1ms** | 1,838.0 / 4,118.8ms | **-6.45%** |
| decode-heavy | 3,673.8 / **8,112.9ms** | **3,465.2** / 8,587.4ms | **-5.53%** |
| long-prefill | **7,548.1 / 15,108.7ms** | 9,969.9 / 19,629.6ms | **-23.03%** |

| Workload | TPOT Current med / p95 | TPOT vLLM med / p95 | Current p95 변화 |
| --- | ---: | ---: | ---: |
| short | **8.930 / 14.614ms** | 12.625 / 25.493ms | **-42.68%** |
| balanced | **16.503** / 18.785ms | 17.135 / **18.579ms** | +1.11% |
| decode-heavy | **12.388 / 13.245ms** | 15.475 / 16.595ms | **-20.19%** |
| long-prefill | **22.833 / 23.786ms** | 35.029 / 40.454ms | **-41.20%** |

| Workload | E2E Current med / p95 | E2E vLLM med / p95 | Current p95 변화 |
| --- | ---: | ---: | ---: |
| short | **333.6 / 405.3ms** | 433.9 / 493.5ms | **-17.86%** |
| balanced | **3,062.7 / 4,846.6ms** | 3,352.3 / 5,162.7ms | **-6.12%** |
| decode-heavy | **5,780.1 / 10,664.6ms** | 6,520.5 / 11,617.6ms | **-8.20%** |
| long-prefill | **9,473.1 / 16,387.5ms** | 12,881.7 / 21,366.5ms | **-23.30%** |

## Dynamic batching과 kernel-group 관측

아래 값은 첫 번째 반복의 scheduler dispatch와 CUDA-event kernel-group CSV에서 얻었다. overlap 비율은 한
scheduler dispatch에 prefill과 decode가 모두 있었던 비율이다. engine 평균은 다양한 batch/context shape를
합친 값이므로 workload 간 절대 kernel cost table 대신 실제 실행 형태를 보는 지표다.

| Workload | 설정 | 관측 max P / D | Overlap dispatch | Page peak | Prefill engine 평균 | Decode engine 평균 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| short | chunk128, graph | 8 / 47 | 18.9% | 56 / 256 | 25.56ms | 8.92ms |
| balanced | chunk128, graph | 8 / 64 | 31.0% | 143 / 256 | 15.86ms | 9.41ms |
| decode-heavy | chunk128, graph | 8 / 64 | 11.8% | 190 / 256 | 15.51ms | 8.97ms |
| long-prefill | chunk256, graph-off | 4 / 36 | 62.7% | 238 / 256 | 26.38ms | 14.73ms |

- balanced/decode-heavy에서 D64가 각각 322회/752회 형성되어 큰 decode batch profile이 실제 online trace에서
  충분히 사용됐다.
- long-prefill은 whole-request full reservation이 page pool을 93.0%까지 사용했다. 80 slots와 D64 engine이
  있어도 active decode는 D36에서 끝났다. 이 workload의 다음 admission 실험은 bounded-overcommit/growth lease다.
- long-prefill의 overlap 비율이 가장 높고, prefill/decode engine event 합이 wall time보다 크게 나타난다. 긴
  prefill을 작은 P4 wave로 보내면서 이미 admitted된 decode를 계속 진행한 결과다.

## 메모리와 CUDA graph

각 graph workload는 로그가 보존되도록 한 번 더 memory probe했다. 아래는 `cudaMemGetInfo` 기반 전체 device used
memory이며 다른 GPU process가 없는 상태였다.

| Workload | 실행 전 | 실행 후 | Headroom | Prefill graph hit | Decode graph hit |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 8,946.9MiB | 9,058.9MiB | 815.4MiB | 37.5% | 73.86% |
| balanced | 8,946.9MiB | 9,184.9MiB | 689.4MiB | 6.45% | 87.55% |
| decode-heavy | 8,946.9MiB | 9,184.9MiB | 689.4MiB | 24.86% | 89.00% |
| long-prefill | 9,658.9MiB | 9,664.9MiB | 209.4MiB | off | off |

128-contract engine은 right-sized packed phase I/O 뒤 세 workload 모두 512MiB headroom gate를 통과한다. decode
graph는 53 shapes, 약 220MiB에서 제한되며 balanced/decode-heavy에서 높은 replay hit를 보였다. 반면 prefill은
shape 다양성 때문에 16MiB budget 안에서 세 graph만 유지하고 hit rate가 낮다.

256-contract engine은 prefill workspace가 128MiB이고 engine/profile 자체도 커서 graph 없이도 headroom이
209MiB뿐이다. 이 경로에 graph를 추가하거나 P8 x 256 I/O를 할당해서는 안 된다. vLLM의 보존된 memory-matched
peak는 약 8,180MiB였지만 측정 API가 `nvidia-smi` process sample로 달라 직접 peak 동등 비교로 보지는 않는다.

## 재현성과 비교 한계

vLLM Docker image는 현재 host에 없고 root filesystem 여유가 13GB뿐이다. 이전 재-pull은 layer extraction에서
공간 부족으로 실패했으며 Docker reclaimable space도 거의 없다. 모델과 엔진 artifact를 임의 삭제하지 않기 위해
vLLM은 다음 보존된 3회 결과를 사용했다. trace hash, HTTP client, prompt token, output token은 다시 확인했으며
숫자를 보간하거나 재구성하지 않았다.

- short/decode-heavy: `.local/vllm-cosmos-reason2-2b/latest-comparison-20260813/`
- balanced: `.local/vllm-cosmos-reason2-2b/throughput-balanced-trace-20260813/`
- long-prefill: `.local/vllm-cosmos-reason2-2b/exact-20260812/long-prefill/`

Current의 새 결과는 `.local/cosmos-reason2-2b/workload-revalidation-20260814/`에 있다. 반복 실행은
`scripts/cosmos_reason2/run_phase_http_trace_bench.py`가 gateway/backend를 매회 새로 시작하고, health를 기다린
뒤 exact trace client를 실행하며, aggregate JSON/CSV를 만든다. backend가 준비 전에 종료하면 gateway가 종료
코드와 backend 로그를 노출한다.

## 다음 단계

1. long-prefill에서 full reservation과 bounded-overcommit/growth lease를 같은 3회 gate로 비교해 D36 제한과
   209MiB headroom 안에서 page wait를 줄인다.
2. prefill graph는 임의 shape capture 대신 실제 빈도가 높은 packed `(rows,totalTokens,pastKV)` bucket만 priming해
   16MiB budget의 낮은 hit rate를 개선한다.
3. balanced TPOT p95가 vLLM보다 0.206ms 느린 구간에 decode hard guard를 적용하되 6.2% 처리량 우위를 지키는지
   확인한다.
4. host 공간을 확보한 뒤 같은 네 trace의 vLLM image/version을 새로 고정하고 같은 날 3회 교차 실행한다.
5. 위 gate 뒤에 prefix-sharing trace와 heterogeneous long-KV cost model을 포함한 production router 회귀 suite를
   상시 실행하도록 묶는다.
