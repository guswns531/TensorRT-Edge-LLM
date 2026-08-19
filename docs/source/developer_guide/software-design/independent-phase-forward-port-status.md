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
- Adaptive chunk candidates and dynamic prefill batch cost points are enabled
  in the Cosmos semantic smoke path.

## Validation

The TensorRT 11.0 / CUDA 13.3 SM86 container validated:

- 68 scheduler, phase-shell, and prefix-ownership tests.
- Cosmos semantic async server output identity.
- Prepared prefill/decode CUDA graph capture and replay.
- Continuous trace: 16 requests, 46 dispatches, 22 overlap dispatches.
- Non-graph overlap: `1.183x`; graph-enabled overlap: `1.145x` for the
  prepared shape on the RTX 3080 test host.

## Explicit capability gate

Cosmos uses M-RoPE. Prefix reuse requires the adapter to provide absolute
position metadata for the suffix after a shared page boundary. The generic
stable-page cache is implemented and unit-tested, but the Cosmos adapter keeps
`supportsPageAlignedPrefixReuse` disabled until that position contract is
implemented. This avoids silently changing greedy output.

The current correctness-safe policy also admits at most one active reader of a
published source prefix. This prevents a packed dispatch from mixing a shared
suffix row with another row that aliases the same physical page. Sequential
prefix reuse is deterministic; multi-reader page sharing remains a future
optimization after a plugin-level aliasing contract is added.

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
OpenAI/SSE continuous server. These values therefore remain the comparison
baseline. A fresh 12-request IPC HTTP trace (same OpenAI/SSE client) measured
`905` generated tokens, `362.4 token/s`, TTFT median/p95
`1872.1/2430.1 ms`, and E2E median/p95 `1874.8/2432.8 ms`.

The same trace against the local vLLM container measured `988` generated
tokens, `1010.9 token/s`, TTFT median/p95 `87.4/93.7 ms`, and E2E median/p95
`643.4/967.8 ms`. The IPC backend emits sampled tokens after a completion
ticket, so its TPOT and token count are not directly comparable to vLLM's
per-token streaming path. The clean v0.10 public runtime still has no
continuous HTTP endpoint; its fixed-batch replay remains the clean oracle
until a matching HTTP adapter is added.
