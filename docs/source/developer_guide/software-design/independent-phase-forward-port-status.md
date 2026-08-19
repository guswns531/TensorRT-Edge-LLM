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
OpenAI/SSE continuous server. These values therefore remain the comparison
baseline. The corrected IPC backend emits each token after its sampling event
rather than replaying every token at final completion. A fresh 12-request HTTP
trace measured `905` generated tokens, `364.4 token/s`, TTFT median/p95
`207.5/1409.9 ms`, TPOT median/p95 `17.83/21.66 ms`, and E2E median/p95
`1852.7/2417.9 ms`.

The same trace against the local vLLM container measured `988` generated
tokens, `1010.9 token/s`, TTFT median/p95 `87.4/93.7 ms`, and E2E median/p95
`643.4/967.8 ms`, with TPOT median/p95 `7.07/8.23 ms`. The token counts differ
(`905` versus `988`), so this is a serving-path comparison, not a fixed-output
kernel comparison. The clean v0.10 public runtime still has no
continuous HTTP endpoint; its fixed-batch replay remains the clean oracle
until a matching HTTP adapter is added.
