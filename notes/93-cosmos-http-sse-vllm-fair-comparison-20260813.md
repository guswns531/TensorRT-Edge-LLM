SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

# Cosmos production HTTP/SSE comparison

## Result

The Current phase runtime now accepts real localhost HTTP/JSON requests through an OpenAI-compatible streaming
gateway. The gateway forwards only admission commands to the C++ event loop; incremental token IDs, UTF-8 text deltas,
EOS state, and timestamps return from `PhaseAsyncServer::tryPopToken()`. Prefill and decode continue to execute in two
independent TensorRT execution contexts and non-blocking CUDA streams inside one CUDA context.

Both backends used the same materialized 288-request trace, SHA-256
`290d34061a173c13440e10247f55bd442a131e525a40c30a52d0b39a549d6538`, greedy decoding, EOS enabled, prefix reuse
disabled, FP16 weights/KV, 80 active sequence capacity, and 3.5GiB raw KV capacity. Each backend produced exactly
25,872 prompt tokens and 23,712 completion tokens. Values are medians of three complete runs.

| metric | Current HTTP/SSE | vLLM HTTP/SSE | Current change |
| --- | ---: | ---: | ---: |
| generated token/s | **4,448.3** | 4,134.6 | **+7.59%** |
| request/s | **54.03** | 50.22 | **+7.59%** |
| TTFT median | **1,630.7ms** | 1,838.0ms | **-11.28%** |
| TTFT p95 | **3,806.0ms** | 4,118.8ms | **-7.59%** |
| TPOT median | **16.261ms** | 17.135ms | **-5.10%** |
| TPOT p95 | 18.725ms | **18.579ms** | +0.79% |
| E2E median | **3,020.1ms** | 3,352.3ms | **-9.91%** |
| E2E p95 | **4,782.6ms** | 5,162.7ms | **-7.36%** |

Current throughput across the three runs was 4,469.1, 4,434.7, and 4,448.3 token/s. The direct-C++ streaming runs had
a median 4,507.4 token/s, so localhost HTTP/JSON/SSE costs about 1.31% of throughput in this workload. It does not erase
the phase scheduler advantage. TPOT p95 is the only metric where vLLM remains slightly better, by 0.146ms.

## Implementation

- `cpp/runtime/scheduling/phaseAsyncServer.{h,cpp}` owns an opt-in token event queue. It computes text deltas by decoding
  the cumulative generated prefix, avoiding invalid byte/subword fragments, and drains final tokens before completion.
- `examples/llm/llm_phase_bench.cpp --ipcServer` consumes newline-delimited admission commands with non-blocking raw
  reads. It emits token/completion JSON events while the same production `submit()/poll()` path drives the GPU.
- `scripts/cosmos_reason2/run_phase_openai_gateway.py` provides `/health`, `/version`, `/metrics`, and
  `/v1/chat/completions`, and converts phase events into OpenAI SSE chunks. Its accept backlog is 1,024 for burst traces.
- `scripts/cosmos_reason2/run_vllm_trace_bench.py --include-request-index` lets the exact same client address the
  deterministic phase trace fixture. Production request bodies remain normal messages; the index is benchmark routing
  metadata, not a scheduling hint.
- `scripts/cosmos_reason2/run_real_request_kv_matrix.py --token-streaming` measures token decoding and event-queue cost
  in direct runtime experiments.

## Scope and caveats

The vLLM image could not be restored for a new fourth run because Docker layer extraction exhausted the host root
filesystem. The comparison therefore reuses the preserved three-run vLLM artifacts whose trace hash, EOS output count,
and HTTP client are exact matches. No vLLM number was reconstructed or extrapolated.

The gateway is a benchmark/production-surface reference, not yet a hardened public server: it lacks authentication,
TLS, request-body security limits, cancellation on client disconnect, arbitrary prompt-to-fixture routing, and graceful
multi-run lifecycle management. Those do not affect the measured completed trace, but must be implemented before an
internet-facing deployment.

Artifacts:

- Current HTTP clients: `.local/cosmos-reason2-2b/http-e2e-fair-20260813/client-r1`, `client-r2b`, `client-r3b`
- Current direct token streaming: `.local/cosmos-reason2-2b/eos-stream-fair-20260813/current-r1..r3`
- vLLM HTTP clients: `.local/vllm-cosmos-reason2-2b/throughput-balanced-trace-20260813`
