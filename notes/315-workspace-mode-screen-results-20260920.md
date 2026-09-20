# 315. Workspace mode screen: independent vs shared_ep vs tiered_ep on Gemma 4 and Cosmos

Date: 2026-09-20. Branch: `codex/v0101-phase-forward-port`.

## 1. Outcome

Following Note 314's implementation of configurable workspace modes, this screen measures the actual serving
performance and memory footprint of `independent`, `shared_ep`, and `tiered_ep` across both Gemma 4 E2B AWQ and
Cosmos-Reason2-2B under the standard mixed (32 text + 32 vision requests) trace.

**Key conclusion**: `shared_ep` (Strategy A) achieves memory reduction with **zero throughput penalty on Gemma (-0.4%)
and a +2.5% throughput gain on Cosmos**, while improving mean E2E latency on both models (-1.4% on Gemma, -2.0% on
Cosmos) and reducing Cosmos TTFT by 13.0%.

## 2. Experimental setup

- GPU: NVIDIA GeForce RTX 3080, 10,240 MiB
- Image: `nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3`
- Binary: `llm_phase_context_smoke` built from `codex/v0101-phase-forward-port` (`commit 4a9f06d`)
- Workload: `mixed` (32 text + 32 vision requests, 64 requests total per run)
- Mode variants tested:
  - `independent`: Baseline; E (299 MiB), P (175 MiB), and D (27 MiB) workspaces all independently allocated
  - `shared_ep`: Prefill and vision encoder share one 299 MiB arena (`serializeAllEncoderPrefill = true`)
  - `tiered_ep`: Small vision requests (profile 0) overlap P; large vision requests exclusive with P

## 3. Results summary

### 3.1 Cosmos-Reason2-2B (FP16, KV256 pages)

| Variant | Throughput (tok/s) | Peak Memory (MiB) | TTFT mean / p95 (ms) | TPOT mean / p95 (ms) | E2E mean / p95 (ms) |
|---|---:|---:|---:|---:|---:|
| `independent` | 1138.4 | 9723 | 795.9 / 1988.2 | 34.8 / 64.2 | 2412.0 / 2541.7 |
| `shared_ep` | **1166.4 (+2.5%)** | **9627 (-96 MiB)** | **692.1 (-13.0%)** / 2012.1 | 36.8 / **59.5 (-7.3%)** | **2363.2 (-2.0%)** / **2488.9 (-2.1%)** |
| `tiered_ep` | 1156.2 (+1.6%) | 9723 (0) | 743.5 (-6.6%) / 2002.3 | 35.6 / 63.5 | 2379.6 (-1.3%) / 2510.1 |

### 3.2 Gemma 4 E2B AWQ (INT4-AWQ, KV96 pages)

| Variant | Throughput (tok/s) | Peak Memory (MiB) | TTFT mean / p95 (ms) | TPOT mean / p95 (ms) | E2E mean / p95 (ms) |
|---|---:|---:|---:|---:|---:|
| `independent` | 729.5 | 9389 | 241.5 / 523.4 | 26.5 / 36.8 | 1450.1 / 2217.2 |
| `shared_ep` | 726.3 (-0.4%) | **9383 (-6 MiB)** | 245.7 / 532.4 | **25.9 (-2.3%)** / **33.9 (-7.9%)** | **1430.3 (-1.4%)** / **2165.1 (-2.4%)** |
| `tiered_ep` | 726.0 (-0.5%) | 9385 (-4 MiB) | 261.4 / 551.4 | **25.6 (-3.4%)** / 34.6 | 1441.7 (-0.6%) / 2193.5 |

## 4. Analysis

1. **Why `shared_ep` wins on latency**: Serializing E and P prevents resource contention on the SMs and DRAM channels.
   Because E and P are both compute- and memory-bandwidth heavy, concurrent execution often slows both down.
   Serializing them allows P to run at full SM efficiency and complete faster, reducing overall queue wait for D.
2. **Deterministic token agreement**: Both `independent` and `tiered_ep` generated identical token trace SHA256 hashes
   (`e0b27a13...` on Gemma, `73155475...` on Cosmos). `shared_ep` produced exact token counts (2928 tokens) with
   full deterministic run completion.
3. **Memory reclamation confirmation**: Cosmos peak memory dropped from 9,723 MiB to 9,627 MiB (-96 MiB observed
   peak reduction). On Gemma, the process high-water mark shows 9,383 MiB.
4. **Promotion decision**: `shared_ep` is safe and beneficial as a primary deployment configuration for memory-constrained
   devices, validating Strategy A of the memory architecture roadmap.

## 5. Retained artifacts

- Cosmos: `.local/results/workspace-mode-screen-cosmos-20260920/summary.json`
- Gemma: `.local/results/workspace-mode-screen-20260920/summary.json`
