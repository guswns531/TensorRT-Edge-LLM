# 319. Unified action space: Dynamic prefill chunking (P128/P256/P512) and TPOT protection

Date: 2026-09-20. Branch: `codex/v0101-phase-forward-port`.

## 1. Scope and motivation

Note 297 established that widening the packed-prefill engine contract from P128 to P512 provides a 13.9% throughput
opportunity on text workloads, but introduces a severe dilemma under vision-language workloads (such as `multi-image`):
- **Fixed 128 chunk**: Safe for decode TPOT, but fragments large vision prompts into excessive TensorRT dispatches,
  inflating TTFT and E2E latency.
- **Fixed 512 chunk**: Faster prompt execution, but blocks decode turns for extended periods, inflating mean TPOT by
  +9.4% and TPOT p95 by +14.4% due to compute and DRAM contention.

To resolve this trade-off, this campaign implements the **Unified Action Space**: expanding the scheduler's action
frontier across chunk dimensions $\{128, 256, 512\}$. Guided by `PhaseTransitionPredictor` and live decode queue
pressure, the scheduler dynamically selects 128 tokens when decode work is pending (shielding TPOT), and expands to
512 tokens when decode is idle or backlog is high (maximizing prefill throughput).

## 2. Implementation details

### 2.1 Runtime configuration and composition root

1. **`PhaseServingRuntimeConfig`**:
   - `enableAdaptivePrefillChunking{false}`
   - `adaptivePrefillChunkCandidates{}`
   - `enableCostAwarePrefillShapeSelection{false}`
2. **Environment overrides**:
   - `TRT_EDGELLM_ENABLE_ADAPTIVE_PREFILL_CHUNKING`: Toggles adaptive chunk selection.
   - `TRT_EDGELLM_ADAPTIVE_PREFILL_CHUNK_CANDIDATES`: Comma-separated candidate shapes (e.g. `128,256,512`).
   - `TRT_EDGELLM_ENABLE_COST_AWARE_PREFILL_SHAPE`: Enables cost-aware shape selection when offline profiles exist.
3. **`llm_phase_context_smoke.cpp` workspace fix**:
   - Added support for `TRT_EDGELLM_PHASE_WORKSPACE_MODE=shared_ep` in the benchmark server binary.
   - Enables P512 prefill context (734 MiB) and vision context (313 MiB) to share a single 734 MiB arena,
     preventing out-of-memory errors on 10 GiB GPUs.

## 3. Benchmark results on P512 engine (`multi-image`)

Screening campaign comparing `shared_ep` (fixed 128 chunk) against `unified_action` (dynamic 128/256/512 chunking)
on `google/gemma-4-e2b-it` using `engine-profiled-p8x512-p8x128-d24-kv2048-p96`:

| Metric | `shared_ep` (fixed 128) | `unified_action` (adaptive) | Delta |
|---|---:|---:|---:|
| **TPOT mean (ms)** | 27.89 | **10.13** | **-63.7%** |
| **TPOT p95 (ms)** | 43.16 | **12.48** | **-71.1%** |
| **E2E mean (ms)** | 1,269.73 | **1,028.84** | **-19.0%** |
| **E2E p95 (ms)** | 1,566.03 | 1,617.32 | +3.3% |
| **Peak GPU memory (MiB)** | 9,813 | **9,805** | -8 MiB |

### Key takeaways:
1. **Dramatic TPOT improvement**: Dynamically throttling prefill chunk size when decode work is active slashed TPOT
   by **63.7%** (from 27.89 ms down to 10.13 ms) and TPOT p95 by **71.1%** (from 43.16 ms down to 12.48 ms).
2. **End-to-end latency reduction**: Average request turnaround time improved by **19.0%** (1,028 ms vs 1,270 ms).
3. **Memory safety preserved**: Peak memory remained bounded at 9,805 MiB on the 10 GiB RTX 3080 via the `shared_ep`
   workspace arena.

## 4. Retained artifacts

- Results: `.local/results/unified-action-p512-screen-v3-20260920/summary.json`
- Manifest: `.local/results/unified-action-p512-screen-v3-20260920/manifest.json`
