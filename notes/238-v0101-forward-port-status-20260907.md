# v0.10.1 phase forward-port status

> This compile-only checkpoint is superseded by
> [239-v0101-forward-port-results-20260907.md](239-v0101-forward-port-results-20260907.md). The GPU became available
> after this note was written; the later report records the completed export, build, inference, semantic VLM, and
> 12-workload V0/V1/V2 validation.

## Scope

This branch forward-ports the retained V0/V1/V2 phase-serving architecture onto the NVIDIA TensorRT Edge-LLM
v0.10.1 source layout. The development worktree is `.local/v0101-forward-port` on
`codex/v0101-phase-forward-port`, based on upstream commit `e8b29522938901f6df19ebeedd4b69bc8edbcd97`.

The historical v0.10.0 worktree remains a read-only comparison source. It is not used as the development root.

## Architecture mapping

v0.10.1 moved the production runtime behind `RuntimeCoordinator`, `LLMRankRuntime`, and the abstract
`EngineExecutor` interface. The forward-port therefore keeps the v0.10.1 facade and extends its mechanisms rather
than restoring the old monolithic runtime.

```text
v0.10.1 request facade
        |
        v
RuntimeCoordinator / LLMRankRuntime
        |
        +-- phase policy and request-DAG mechanisms
        |
        +-- EngineExecutor sibling contexts
        |      +-- shared TensorRT engine/runtime
        |      +-- independent execution context
        |      +-- profile-local workspace and CUDA graph cache
        |
        +-- phase-local PipelineIO and tensor registry
        |
        +-- canonical v0.10.1 paged KV pool
               +-- stable request lease
               +-- active-row to page mapping
               +-- no KV movement during logical compaction
```

The first retained commit, `fa058e5`, adds the phase scheduling, telemetry, stable ownership, benchmark, and test
substrate. The current integration batch adds the v0.10.1 execution/configuration adapters needed to compile that
substrate against the new runtime interfaces.

## Implemented in the current integration batch

- `EngineExecutor::createSibling()` creates an execution context that shares the deserialized TensorRT engine but
  owns its execution context, profile-local device workspace, CUDA graph cache, and context identity.
- Profile-specific workspace sizing and graph-cache statistics are exposed without weakening the v0.10.1 abstract
  executor interface.
- `InferenceDims` and `LLMEngineConfig` represent asymmetric prefill/decode limits, packed-prefill metadata, vision
  prefill profiles, and KV undercommit capability.
- `PipelineIO` and the tensor registry allocate phase-specific shapes and the packed-prefill chunk carrier.
- `KVPageTable` uses pinned staging slots and explicit cross-stream CUDA-event handoff for asynchronous page-table
  publication.
- The multimodal adapter exposes profile-specific workspace/token limits and request-owned output storage.
- Segmented vision/deep-stack embeddings are materialized into reusable GPU scratch before entering the canonical
  v0.10.1 embedding path. This preserves semantics for the first port; zero-copy segmented kernels remain a later
  performance task.
- The phase smoke executable follows the v0.10.1 external-weight ownership contract.

## Validation completed without a working CUDA driver

Build environment:

- image: `nvcr.io/nvidia/tensorrt:26.06-py3`
- TensorRT: 11.0.0
- CUDA toolkit: 13.3
- target architecture: SM86
- CUTE DSL disabled for the local Ampere build
- CUDA driver API linked through the toolkit stub for compile/link validation only

Successful build targets:

- `llm_phase_context_smoke`
- `unitTestRuntime`
- `unitTestRuntimeState`

Host-only tests:

- 245 phase scheduler, candidate formation, event, cost-model, memory-broker, prefix-reuse, and load-generator tests
  passed.
- 3 of 6 `StableKVPageManagerTest` cases passed.
- The other 3 stable-KV cases require a real `cudaMalloc` for page-table tensors and stopped with
  `CUDA driver is a stub library`; they did not report an ownership assertion failure.

The host currently reports an NVIDIA driver initialization failure, and the container cannot initialize CUDA.
Consequently, the three CUDA-backed stable-KV tests, phase smoke runtime, export/build/inference validation, and
performance comparison are not yet executable. Stub-linked success is compile/link evidence only.

## Remaining forward-port order

1. Commit the independent-context/configuration integration after formatting and pre-commit validation.
2. Port packed-prefill and indexed-paged builder, plugin, kernel, and Python export contracts while preserving the
   v0.10.1 model registry and attention paths.
3. Connect stable page leases and phase-local page-table publication to `LLMRankRuntime` request admission,
   completion, cancellation, and continuous admission.
4. Connect `PhaseAsyncServer` and the production request adapter to `RuntimeCoordinator` rather than the removed
   v0.10.0 runtime facade.
5. Connect encoder completion to the prefill-ready queue and retain independent E/P/D TensorRT contexts with one
   shared CUDA primary context.
6. Restore the GPU driver and run unit/sanitizer checks, then `export -> build -> inference` on the retained Cosmos
   model and fixed engines.
7. Re-run V0/V1/V2 under the same binary, engine, request traces, calibration, and memory limit; reuse the frozen
   vLLM results only where that contract is unchanged.
8. Record per-workload throughput, TTFT, TPOT, E2E mean/p95, phase occupancy, overlap, idle time, memory, and SLO
   goodput before promoting the v0.10.1 port.

## Promotion gates

- Legacy-compatible V0 output and stable-ownership V0 output must match for greedy decoding.
- V0/V1/V2 must use the same candidate frontier and differ only in the retained estimator/transition policy.
- Logical eviction must not launch KV compaction or KV D2D copy.
- Planned phase action must match the actual outstanding E/P/D context set.
- Scheduler and stable-ownership host tests must pass; CUDA-backed tests must pass on the real driver.
- No retained 12-workload regression may be hidden by an aggregate average.
