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
- `PhaseThreeCoordinator` owns encoder admission/cancellation and hands a
  completed request-owned payload to the continuous prefill/decode server.
- Phase-sized I/O keeps the decode context at sequence length one instead of
  duplicating prefill-sized embedding, hidden-state, and deepstack buffers.

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
| request-owned Gemma/Cosmos encoder outputs | complete for C++ and OpenAI-style HTTP image submission |
| page reservation/growth leases | complete: full/headroom modes and decode growth wait queue |
| tied embedding/LM-head reuse | complete; controlled BS1 greedy identity and 3% performance gates pass |
| complete three-phase encoder queue coordinator | complete, including queued cancel and one encoder in flight |

## Validation

The TensorRT 11.0 / CUDA 13.3 SM86 container validated:

- 67 phase-scheduler tests in the latest focused gate, including exclusive
  multimodal chunking; the broader scheduler/stable-KV/prefix suites also pass.
- Cosmos semantic async server output identity.
- Prepared prefill/decode CUDA graph capture and replay.
- Continuous trace: 16 requests, 46 dispatches, 22 overlap dispatches.
- Non-graph overlap: `1.183x`; graph-enabled overlap: `1.145x` for the
  prepared shape on the RTX 3080 test host.
- Cosmos image request: independent encoder context/stream -> request-owned
  embedding and M-RoPE -> 128-token chunked prefill -> decode completion.

The Cosmos gate initially exposed two independent host-side defects: the
three-phase path had not populated `formattedRequests`, and a function call
could move the request payload before reading its token vector. Both are now
fixed. Image embeddings and deepstack tensors are viewed from the cumulative
image-token offset for each prefill chunk, so later chunks do not restart from
feature row zero.

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
When `TRT_EDGELLM_VISION_ENGINE_DIR` is set, OpenAI `image_url` content with a
local path or `file://` URL is decoded into an image buffer and submitted to
`PhaseThreeCoordinator`; remote/data URLs are rejected with a structured SSE
error. The request retains the same cancellation and completion path as text.
The gateway sends a cancel control message when an SSE client disconnects;
queued requests release their stable lease immediately, while in-flight work
keeps the lease until its CUDA event boundary.

The v0.10 Python OpenAI/SSE server still calls `LLMInferenceRuntime` directly;
the IPC gateway is the transport adapter for the new forward-port facade.

The fresh HTTP image gate emitted eight token deltas followed by `[DONE]`,
reported `prompt_tokens=501`, and showed four prefill chunks
(`128/128/128/117`) before decode. This verifies actual JSON parsing, encoder
handoff, chunk-offset image embedding placement, and SSE usage accounting.

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

## B8 to B32 forward-port measurements

The builder still exposes one common `maxBatchSize`, while the scheduler can
independently cap prefill and decode rows. A fresh export corrected an obsolete
artifact problem: `last_token_ids` now has dynamic `token_batch` and
`num_selected` axes. The graph did not require a source change. This enabled
B16/KV2048 and B32/KV1024 engines and P/D caps of P8/D16 and P8/D32.

Stable ownership is decoupled from active engine rows. The default is 80
stable slots, 8 active engine rows, and an in-flight admission cap of 16.
Environment variables can override the stable-slot and in-flight limits.

The earlier B8 results below are retained as the scaling baseline:

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
closed by scheduler tuning alone. The larger profiles validate that diagnosis.

The fixed-output balanced trace contains 288 requests, 25,872 prompt tokens,
and 24,960 generated tokens per run. Three fresh process lifecycles give:

| configuration | generated token/s | TTFT med/p95 | TPOT med/p95 | E2E med/p95 | ready/peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| B16 P8/D8 | 869.8 | 13246/25685 ms | 17.55/18.46 ms | 14760/27233 ms | 9293/9299 |
| B16 P4/D16 | 1297.8 | 8802/17136 ms | 11.78/12.58 ms | 9874/18126 ms | 9293/9299 |
| B16 P8/D16 | 1300.4 | 8841/17092 ms | 11.77/12.57 ms | 9861/18161 ms | 9293/9299 |
| B32 P8/D32 | 1921.4 | 5612/11112 ms | 15.66/16.88 ms | 6842/12188 ms | 8459/8465 |

P8/D16 is the B16 throughput winner; P4/D16 trades 0.2% throughput for a
slightly lower median TTFT. B32 increases throughput by 47.8% over B16
P8/D16. Its smaller KV capacity and a smaller TensorRT tactic also leave about
1.78 GiB of 10 GiB device headroom.

The scheduler cost table now covers D1-D16 at the 2,048-token context contract
and D17-D32 at the 1,024-token contract. CUDA-event p95 grows from 6.238 ms at
D1 to 6.862 ms at D16, then from 6.927 ms at D17 to 7.484 ms at D32. A
50,000-us decode queue target is selected for the saturated fixed-output
traces so the scheduler can form the efficient upper decode bucket.

## Asymmetric P8/D64 recovery

The builder and runtime now preserve separate prefill/decode profile limits.
The validated Cosmos engine uses global batch 80, P8, D64, KV capacity 2,048,
and 256 physical page bundles. An explicit `allow_kv_pool_undercommit`
contract decouples worst-case profile address space from physical pool size.
Inactive global page-table rows start empty, stable allocator leases are the
only source of live page IDs, and the public `handleRequest()` runtime rejects
this phase-server-only engine mode.

Phase-local I/O allocates P8 prefill and D64 decode buffers rather than two B80
maximum-shape sets. The tied engine reports 8,081/8,087 MiB ready/peak usage.
On the 288-request fixed-output balanced trace, graph-off P8/D64 with 64
in-flight requests gives a three-run median of 3,693.3 generated token/s,
2,337/5,066 ms TTFT, 15.29/16.64 ms TPOT, and 3,617/6,114 ms E2E. Relative to
the prior B32 P8/D32 gate, throughput improves 92.2% and peak memory falls
378 MiB.

Cold first-shape CUDA Graph capture regresses the single-run result by 2.7%, so
graph-off remains selected until shapes are primed before measurement. Raising
in-flight admission from 64 to 80 creates a recurring D64+D16 round and lowers
throughput. An opt-in stable decode cohort is implemented and tested, but is
not selected yet because rolling prefill admission must be coordinated with
cohort replacement.

Sampling-aware decode refill now closes the async-completion seam that caused a
steady D64/D16 alternation. When a partial decode tail plus completed decode
sampling tickets can reconstruct the configured target, the server defers only
that tail dispatch until the CUDA event is collected. The default target is
zero, preserving prior behavior.

With target 64 and in-flight capacity 80, balanced fixed-output throughput is
3,889.9 token/s across three fresh lifecycles. The D16 steady-state bucket is
eliminated, TTFT median falls to 2,035.8 ms, and peak memory remains 8,087 MiB.
The tradeoff is median TPOT rising from the in-flight64 latency profile's
15.29 ms to 18.23 ms. Short and decode-heavy smoke results are 1,698.5 and
4,476.6 token/s respectively.

An additional adaptive-admission opt-in starts with the latency profile,
enters throughput mode when pending backlog appears, and exits only after the
pending queue is empty and active requests fall to 64 or fewer. Balanced gives
3,905.7 token/s across three fresh lifecycles, with exactly two transitions per
run. Short remains in latency mode; decode-heavy spends 1,132 dispatches in
throughput mode and returns to latency mode for its final 337 dispatches.
Mode, transition count, and sampling-refill wait count are emitted with every
phase metric.

The IPC event loop no longer sleeps for a fixed millisecond after every poll.
It yields only when input, server polling, metrics, tokens, and completions all
make no progress. Sampling uses ticket-owned pinned host buffers and a reusable
CUDA-event slot pool; all ready tickets are collected in one pass rather than
being blocked behind the queue front.

Balanced fixed-output throughput rises from 3,905.7 to 4,346.8 token/s. A
fresh engine-specific cost model gives 4,364.3 token/s and 17.57 ms TPOT p95.
The host gap falls from about 1.00 seconds to 0.28--0.32 seconds, with two
sampling events reused roughly 560 times per run. Fresh native v0.9.1 reaches
4,468.1 token/s under the same fixed-output HTTP trace, leaving v0.10 within
2.32% throughput while using 332 MiB less peak memory.

The production IPC path now primes representative decode buckets before it
announces readiness. For D64 the buckets are D8/D16/D32/D48/D64. When CUDA
Graphs are enabled, the warmup captures only these prepared shapes and then
freezes runtime capture; an unseen production shape uses `enqueueV3` instead
of synchronously capturing on its latency-critical first request. Per-dispatch
metric serialization and retention are disabled by default and remain
available with `TRT_EDGELLM_EMIT_PHASE_METRICS=1`. Scheduler CUDA-event
telemetry remains active when external retention is disabled.

On fresh HTTP traces with the P8/D64 engine, startup priming and graph freezing
produce 2,362.2/4,531.2/5,275.5 generated token/s for short, balanced, and
decode-heavy workloads. The corresponding memory-matched vLLM 0.27.1 results
are 1,976.6/4,283.7/4,831.0 token/s, and fresh native v0.9.1 results are
1,353.4/4,468.1/4,911.5 token/s. Current therefore wins throughput in all
three workload classes. Decode-heavy median TTFT remains 0.8% behind v0.9.1
and 8.7% behind vLLM, while its TTFT p95, TPOT, E2E, and throughput are all
better. A P16/D64 experiment selected a slower TensorRT tactic and reduced
balanced throughput to 4,278 token/s, so it was rejected and its engine was
removed.

The next serving pass batches native token/completion records after the next
GPU dispatch has been enqueued, retains a D64 cohort while the other 16 stable
slots stage P8 prefills, and refines static decode costs from confident
decode-only observations. Online correction is active only in adaptive
throughput mode, uses a 32-sample window in 512-token context buckets, requires
eight samples, and is bounded to +/-25% of the static prior. Short throughput
improves from 2,362.2 to 2,410.4 token/s and median TPOT from 11.91 to 11.26 ms.
Balanced/decode-heavy throughput remains within 1% of the previous selected
median, while representative TTFT/TPOT smokes improve.

`EngineExecutor` now reports CUDA Graph hits, misses, captures, failures, and
frequency/LRU evictions. `TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES` accepts a
bounded comma-separated shape profile. An eight-shape frequency experiment
used 22 MiB more GPU memory and reduced short throughput from 2,406 to 2,360
token/s, so the default D8/D16/D32/D48/D64 profile remains selected.

The remaining decoder headroom can host the existing Cosmos visual engine and
its 806 MiB external-weight arena. Decoder plus vision uses 9,305 MiB at ready.
Repeated two-image HTTP traces now release request IDs through
`PhaseThreeCoordinator` and pass three times with an identical token trace.
Unbounded image bursts exhausted memory because request-owned encoded payloads
accumulated behind the LLM. The coordinator therefore keeps at most two encoded
GPU payloads downstream by default and leaves additional requests in the CPU
encoder queue. A 16-image burst then completes all 256 requested tokens at
92.3 token/s with a 9,481 MiB peak. The remaining 394 MiB is below the text-only
512 MiB safety gate but completed without OOM; vision serving should retain
this stricter encoded-payload backpressure.

Headroom reservation is available through
`TRT_EDGELLM_PAGE_RESERVATION=headroom`. A 96-request pressure run completed
without OOM or lost requests at `993.7 token/s`; TPOT rose to `56.9 ms`, so the
latency-safe default remains full reservation with in-flight 16.

## Tied embedding and LM-head reuse

`--reuse-tied-lm-head` has been forward-ported as an experimental opt-in. The
exporter stores one FP16 `[hidden, vocab]` table, removes the LM-head transpose,
and exposes the head weight as a fixed engine input. The runtime embedding
loader uses a strided gather while `ExternalWeightManager` publishes a
non-owning alias of the same GPU allocation to TensorRT. FP8 embeddings,
reduced vocabularies, tensor parallelism, and speculative decoding are
rejected by the v1 export contract.

The v0.10 validation exercised the required sequence end to end:

1. exported a Cosmos packed-prefill ONNX with the tied source manifest;
2. built a TensorRT 11 B8/KV2048/page128 engine;
3. ran semantic requests through independent prefill/decode contexts.

The CUDA transposed-lookup reference test and seven ONNX externalization tests
also pass. The final paired gate uses byte-identical baseline/tied ONNX model
files and one shared serialized TensorRT engine, so tactic selection and engine
bytes cannot confound the comparison. Only the runtime weight manifest and
embedding sidecar differ.

On the 48-request controlled BS1 greedy gate, all 4,160 output token IDs match
and both variants produce token-trace SHA-256
`b14d1c789b567b4dc231444871c6c7a1b8fa3d05e879c4cc8a97a4afdc7d9480`.
The full baseline embedding transpose, baseline LM-head sidecar, and tied
embedding are also byte-exact equal.

On the 288-request fixed-output HTTP trace, three fresh process lifecycles per
variant give a 594 MiB ready/peak memory reduction. Tied throughput is 0.192%
higher; TTFT median/p95 changes by -0.549%/-0.378%, TPOT by
+0.140%/+0.058%, and E2E by +0.020%/-0.115%. All performance dimensions are
inside the 3% gate. EOS is disabled only for this performance gate because an
EOS-enabled online trace produces different batch evolution even across
repeated runs of the same engine.

## Production tied-engine contract

The materialization helper now writes `tied_engine_contract` into
`config.json`. It requires byte-identical baseline/tied ONNX files and records
the serialized engine SHA-256 in the audit manifest. Runtime startup checks the
engine byte count, a four-position 1 MiB CRC32 fingerprint, and the tied
embedding allocation byte count before publishing the non-owning LM-head
alias. This catches stale or mixed engine directories without adding a full
multi-gigabyte hash pass to every process startup.

The new contract loaded the B8 tied engine and completed semantic inference in
4.11 seconds wall time. An older tied directory without the contract is
rejected with a specific materialization error. The helper's contract and ONNX
mismatch behavior have Python unit coverage.

## Multimodal row ownership

Cosmos vision payloads remain chunkable, but each image request is now marked
`exclusivePrefill`. Chunks from different image requests cannot occupy one
packed prefill batch because their request-owned embedding/deepstack/M-RoPE
views are not yet represented by a multi-row placement descriptor. Decode no
longer binds the vision payload after the final prefill chunk.

A real two-image OpenAI-style HTTP trace completes 2/2 requests with 1,017
prompt tokens and 32 generated tokens. The one-run smoke result is 70.4
token/s, 315.5 ms median TTFT, 9.23 ms median TPOT, and 453.9 ms median E2E.
