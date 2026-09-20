# 314. Phase workspace modes and dynamic headroom for KV expansion

Date: 2026-09-20. Branch: `codex/v0101-phase-forward-port`.

## 1. Scope and motivation

Note 312 established that Current's process memory exceeds vLLM by 618--1,048 MiB on RTX 3080 10 GiB despite holding
264 MiB less physical KV cache (12,288 tokens vs vLLM's 27,025 tokens). The primary driver is the independently
materialized vision execution substrate: 784 MiB resident at startup, including a 299 MiB encoder workspace.

Previously, `PhaseServingRuntime` selected the workspace mode via a hardcoded heuristic:
if `freeBytes >= visionBytes + 96 MiB`, it unconditionally allocated an independent 299 MiB E workspace. On a 10 GiB GPU,
this forced Mode 1 (`independent`), consuming 501 MiB across E (299 MiB), P (175 MiB), and D (27 MiB) workspaces,
leaving no memory headroom to expand the KV cache pool. Furthermore, the 96 MiB workspace headroom constant was
hardcoded without configuration or environment overrides.

This note documents the design and implementation of explicit, configurable phase workspace modes and dynamic
headroom controls in `PhaseServingRuntimeConfig`, enabling principled memory-versus-concurrency trade-offs and
unlocking KV cache expansion.

## 2. Workspace modes and memory footprint

| Mode | E/P overlap | E/D overlap | P/D overlap | Total workspace | Savings vs independent |
|---|:---:|:---:|:---:|---:|---:|
| `independent` | Yes | Yes | Yes | 501 MiB | Baseline (0 MiB) |
| `tiered_ep` | Small-E only | Yes | Yes | ~391 MiB | +110 MiB |
| `shared_ep` | No (serialized) | Yes | Yes | ~326 MiB | +175 MiB |
| `shared_ed` | Yes | No (serialized) | Yes | ~474 MiB | +27 MiB |
| `auto` | Dynamic | Dynamic | Yes | Dynamic | Resolved at startup |

### 2.1 Mode details

- **`independent`**: Allocates independent TensorRT context memory for E (299 MiB), P (175 MiB), and D (27 MiB).
  All pairwise overlaps (E+P, E+D, P+D) remain physically available.
- **`tiered_ep`**: Uses one E/P arena sized at `max(aligned(P) + small_E, large_E)`. Small vision requests
  (profile 0) can overlap prefill; large vision requests (profile 1) require mutual exclusion with prefill.
- **`shared_ep`**: Prefill and vision encoder share one arena sized at `max(P, E) = 299 MiB`. E and P actions
  are serialized (`serializeAllEncoderPrefill = true`). Note 308 showed E+P overlap accounts for only 2--8% of
  epoch time, making this an attractive trade-off for KV capacity.
- **`shared_ed`**: Decode and vision encoder share one arena sized at `max(D, E) = 299 MiB`. E and D actions
  are serialized (`serializeAllEncoderDecode = true`), and decode batch capacity is bounded under memory pressure.

## 3. Configuration and environment overrides

`PhaseServingRuntimeConfig` introduces:

```cpp
enum class PhaseWorkspaceMode
{
    kAuto,
    kIndependent,
    kTieredEp,
    kSharedEp,
    kSharedEd,
};

PhaseWorkspaceMode workspaceMode{PhaseWorkspaceMode::kAuto};
size_t workspaceHeadroomBytes{96U * 1024U * 1024U};
```

Runtime environment overrides:
- `TRT_EDGELLM_PHASE_WORKSPACE_MODE`: `auto`, `independent`, `tiered_ep` (or `tiered`), `shared_ep`, `shared_ed`
- `TRT_EDGELLM_WORKSPACE_HEADROOM_BYTES`: Integer byte override for the safety margin (default 96 MiB)
- Compatibility fallbacks: `TRT_EDGELLM_TIERED_VISION_CONTEXT_MEMORY=1` -> `kTieredEp`,
  `TRT_EDGELLM_SHARED_VISION_DECODE_CONTEXT_MEMORY=1` -> `kSharedEd`.

## 4. KV expansion impact

For Gemma 4 E2B AWQ on RTX 3080 10 GiB (FP16 KV, 128 tokens/page = 2.25 MiB/page):

| Scenario | Reclaimed memory | Additional KV pages | Total KV capacity | vs vLLM (27,025 tok) |
|---|---:|---:|---:|---:|
| Current baseline (independent) | 0 MiB | 0 | 12,288 tokens (96 pages) | 45.5% |
| `shared_ep` | 175 MiB | +77 pages | 22,144 tokens (173 pages) | 81.9% |
| `shared_ep` + 48 MiB headroom | 223 MiB | +99 pages | 24,960 tokens (195 pages) | 92.4% |
| `shared_ep` + 0 MiB headroom | 271 MiB | +120 pages | 27,648 tokens (216 pages) | **102.3%** |

Reclaiming 175--271 MiB eliminates the primary structural barrier that previously prevented Current from matching
vLLM's 27,025-token capacity on 10 GiB devices.

## 5. Verification plan

1. Verify compilation and existing unit tests (`unitTestRuntime`, `unitTestCommon`).
2. Verify that `TRT_EDGELLM_PHASE_WORKSPACE_MODE` correctly selects each of the four modes at startup.
3. Validate memory consumption via `nvidia-smi` / `cudaMemGetInfo` telemetry across all modes.
4. Run paired 6-workload screens under identical lifetime-admission contracts to measure the throughput/latency
   impact of `shared_ep` and `tiered_ep` vs `independent`.
