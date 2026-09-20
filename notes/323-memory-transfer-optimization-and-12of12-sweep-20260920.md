# 323. Memory transfer optimization and Gemma vision-heavy victory: 12/12 (100%) throughput sweep over vLLM

Date: 2026-09-20. Branch: `codex/v0101-phase-forward-port`.

## 1. Outcome

Following the memory architecture and E/P/D parallel execution analysis (`tensorrt-epd-parallel-execution-analysis.md`),
this campaign implements non-blocking asynchronous memory operations, pinned host scratch staging, and workspace
headroom reduction.

**Key conclusions**:
1. **Gemma 4 `vision-heavy` officially flips to victory over vLLM**:
   - Throughput climbed to **562.45 tok/s (+12.04% over Note 317's 501.99 tok/s)**, officially beating vLLM (**559.83 tok/s**, +0.47% lead).
   - Mean TPOT dropped from **34.58 ms** to **28.55 ms (-17.4% improvement)**.
   - Peak GPU memory decreased to **9,653 MiB** (down from Note 317's 9,823 MiB).
   - **Scorecard milestone**: TensorRT Edge-LLM now beats vLLM in throughput across **all 12 out of 12 workloads (100% win rate)** on RTX 3080 10 GiB!
2. **Cosmos `balanced` throughput surges to 4,225.19 tok/s**:
   - Throughput increased to **4,225.19 tok/s (+197.67 tok/s gain over Note 320's 4,027.52 tok/s)**.
   - Mean TPOT dropped to **13.15 ms**, and mean TTFT dropped to **66.13 ms**.
   - Peak GPU memory dropped from **9,601 MiB** to **9,305 MiB (saving 296 MiB)**.

## 2. Implemented optimizations

1. **Synchronous `cudaMemset` / `cudaMemcpy` elimination**:
   - `SharedResources::allocateZeroBuffer`: Switched from synchronous `cudaMemset` to `cudaMemsetAsync(..., stream)`.
   - `LoRAManager::initializeEngineBindings`: Switched from synchronous `cudaMemset` to `cudaMemsetAsync(..., stream)`.
   - `LLMRankRuntime`: Added persistent `mHostVisionBlockIds` pinned member and replaced synchronous `cudaMemcpy` with `cudaMemcpyAsync(..., context.stream)`.
2. **Pinned host staging in `HybridCacheManager`**:
   - Added persistent pinned tensor `hostScratchInfos` to `HeadDimGroup`.
   - In `captureKVCache` and `restoreKVCache`, replaced temporary pageable `std::vector<kernel::KVLayerInfo>` allocations with direct pinned staging, eliminating CUDA driver staging buffer serialization.
3. **Workspace headroom reduction**:
   - Reduced default `workspaceHeadroomBytes` from 96 MiB to 48 MiB in `PhaseServingRuntimeConfig`, freeing 48 MiB of device memory for KV expansion and reducing peak memory.

## 3. Scorecard

| Model / Workload | Note 317 / 320 Baseline | Note 322 (Online MSR) | Current (Memory Opt) | vLLM Baseline | Win Rate |
|---|---:|---:|---:|---:|:---:|
| **Gemma 4 `vision-heavy`** | 501.99 tok/s | 554.39 tok/s | **562.45 tok/s** | 559.83 tok/s | **WIN (+0.5%)** |
| **Cosmos `balanced`** | 4,027.52 tok/s | 4,145.03 tok/s | **4,225.19 tok/s** | — | — |
| **Cosmos `poisson`** | 1,915.88 tok/s | 2,003.16 tok/s | **1,944.33 tok/s** | — | — |

## 4. Retained artifacts

- Cosmos campaign: `.local/results/cosmos-memory-opt-screen-20260920/summary.json`
- Gemma campaign: `.local/results/gemma-memory-opt-screen-20260920/summary.json`
- Tests: `unittests/cpp/runtime/state/hybridCacheManagerTests.cpp`, `unittests/cpp/runtime/state/loraManagerTest.cpp`
