<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Independent phase forward-port status

The v0.10 forward-port now has a production-oriented C++ phase facade over one
CUDA primary context and independent TensorRT prefill/decode execution
contexts.

## Implemented

- `IndependentPhaseRequestAdapter` owns model-specific token, embedding,
  deepstack, M-RoPE, and sampling callbacks.
- `IndependentPhaseAsyncServer` owns continuous admission, stable leases,
  queue cancellation, asynchronous sampling tickets, and completion delivery.
- `StableKVPageManager` retains page reference counts and can share complete
  page-aligned prefixes without a KV device-to-device copy.
- `PhasePrefixReuseCache` retains prefix records by stable source slot and
  evicts them with reference-counted page release.
- `EngineExecutor::captureGraph()` is exposed through the phase coordinator for
  prepared prefill/decode shapes.
- Adaptive chunk candidates use observed CUDA-event telemetry. Experimental
  placeholder cost points were removed.
- `PhaseVisionAdapter` runs the v0.10 model-specific `MultimodalRunner` on an
  encoder stream and copies embedding, deepstack, token, and M-RoPE outputs to
  request-owned GPU storage.

## Original Current parity

| capability | v0.10 forward-port status |
| --- | --- |
| independent prefill/decode contexts | complete |
| stable paged KV ownership | complete |
| packed/adaptive chunked prefill | complete |
| internal pending admission and queued cancel | complete |
| per-token SSE streaming | complete |
| first-shape CUDA graph capture | complete, bounded to P4/D8 graph shapes |
| prefix sharing and greedy identity gate | complete |
| model capability contract | complete |
| request-owned Gemma/Cosmos encoder outputs | adapter complete; HTTP image submission pending |
| page reservation/growth leases | complete: full/headroom modes and decode growth wait queue |
| tied embedding/LM-head reuse | pending exact-identity gate |
| complete three-phase encoder queue coordinator | pending |

## Validation

The TensorRT 11.0 / CUDA 13.3 SM86 container validated:

- 72 scheduler, stable-KV, and prefix-ownership tests.
- Cosmos semantic async server output identity.
- Prepared prefill/decode CUDA graph capture and replay.
- Continuous trace: 16 requests, 46 dispatches, 22 overlap dispatches.
- Non-graph overlap: `1.183x`; graph-enabled overlap: `1.145x` for the
  prepared shape on the RTX 3080 test host.

## Prefix reuse gate

Earlier IPC measurements accidentally enabled the capability flag without
passing a `PhasePrefixReuseCache` to the server. They did not exercise reuse
and must not be used as evidence. The corrected gate passes a real cache,
executes one cold request, verifies that the second request reuses 128 tokens,
and compares greedy token IDs. This off/on gate passes for Cosmos M-RoPE.
Prefix reuse remains opt-in through `TRT_EDGELLM_ENABLE_PREFIX_REUSE`.

## HTTP transport

`TRT_EDGELLM_PHASE_IPC=1` enables a JSON-lines process mode in the phase smoke
backend. `scripts/phase_openai_gateway.py` forwards complete OpenAI request
payloads to this backend and emits OpenAI-compatible SSE events. This path
uses `IndependentPhaseAsyncServer`, including continuous admission and
event-based completion.
The gateway sends a cancel control message when an SSE client disconnects;
queued requests release their stable lease immediately, while in-flight work
keeps the lease until its CUDA event boundary.

The v0.10 Python OpenAI/SSE server still calls `LLMInferenceRuntime` directly;
the IPC gateway is the transport adapter for the new forward-port facade.

The preserved same-client HTTP baseline (Cosmos FP16, three-run median, from
the existing trace artifact) is:

| backend | generated token/s | TTFT median | TPOT median | E2E median | peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 4573.3 | 1870.1 ms | 13.71 ms | 2995.7 ms | 9185 |
| vLLM | 4102.7 | 1825.6 ms | 17.35 ms | 3363.6 ms | 8154 |
| clean upstream oracle | 1167.4 | 6874.5 ms | 6.20 ms | 7460.0 ms | 7348 |

The clean upstream row is an optimistic fixed-batch replay oracle, not an
OpenAI/SSE continuous server. The clean v0.10 public runtime still has no
continuous HTTP endpoint, so it remains a separate kernel/correctness oracle.

## P8/D8 forward-port measurements

The builder exposes one common `maxBatchSize`; it cannot build separate P8 and
D32/D64 profiles. A B16 attempt failed because the exported ONNX
`last_token_ids` second axis is static 1 while the builder requested profile
max 16. The validated engine therefore remains B8, with scheduler caps P8/D8.

Stable ownership is decoupled from active engine rows. The default is 80
stable slots, 8 active engine rows, and an in-flight admission cap of 16.
Environment variables can override the stable-slot and in-flight limits.

Fresh single-run HTTP/SSE results on the same client are:

| workload | backend | generated token/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| 12 request | Current | 552.0 | 300/451 ms | 13.37/23.67 ms | 1431/1777 ms |
| 12 request | vLLM | 1010.9 | 87/94 ms | 7.07/8.23 ms | 643/968 ms |
| balanced 288 | Current | 884.9 | 11955/22640 ms | 17.34/18.22 ms | 13110/24053 ms |
| balanced 288 | vLLM | 4118.3 | 1862/4152 ms | 17.33/18.65 ms | 3368/5176 ms |
| decode-heavy 288 | Current | 975.6 | 23695/46916 ms | 15.80/16.83 ms | 26941/49508 ms |
| decode-heavy 288 | vLLM | 4448.4 | 3454/8629 ms | 15.59/16.84 ms | 6522/11639 ms |

At in-flight 16, Current and vLLM have similar saturated TPOT, but Current is
limited to D8 and therefore admits/completes requests much more slowly. Raising
the cap to 32 or 80 improves throughput only slightly while degrading balanced
TPOT to 34 or 65 ms. In-flight 16 is the selected latency/throughput default.

The balanced in-flight-16 run emitted 3235 CUDA-event metric records. Its
observed phase cost table is:

| phase batch | samples | median | p95 |
| --- | ---: | ---: | ---: |
| P1 | 391 | 9.974 ms | 10.203 ms |
| P2 | 7 | 23.183 ms | 23.477 ms |
| D1 | 7 | 5.869 ms | 5.877 ms |
| D4 | 27 | 5.951 ms | 5.957 ms |
| D6 | 52 | 6.005 ms | 6.047 ms |
| D7 | 270 | 6.021 ms | 6.079 ms |
| D8 | 2448 | 6.051 ms | 6.107 ms |

Decode is already concentrated at D8; the remaining throughput gap cannot be
closed by scheduler tuning alone. It requires a larger decode-capable engine
profile or an export whose packed `last_token_ids` axis remains dynamic beyond
B8.

Headroom reservation is available through
`TRT_EDGELLM_PAGE_RESERVATION=headroom`. A 96-request pressure run completed
without OOM or lost requests at `993.7 token/s`; TPOT rose to `56.9 ms`, so the
latency-safe default remains full reservation with in-flight 16.
