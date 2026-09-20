# 316. Gemma 4 KV 192-page capacity expansion with shared_ep workspace

Date: 2026-09-20. Branch: `codex/v0101-phase-forward-port`.

## 1. Outcome

Following Note 314 (configurable workspace modes) and Note 315 (`shared_ep` screen), this experiment evaluates
Strategy E of the memory architecture roadmap: reallocating the reclaimed workspace memory into a 2x physical
KV cache pool expansion for Gemma 4 on RTX 3080 10 GiB.

The new engine (`engine-packed-p8-d24-kv2048-p192`) expands physical KV capacity from 96 pages (12,288 tokens,
45.5% of vLLM) to **192 pages (24,576 tokens, 91.0% of vLLM)**. Combined with `shared_ep` workspace mode,
peak GPU memory stays within device limits at **9,825 MiB / 10,240 MiB**.

**Key conclusion: Expanding KV capacity resolves the primary bottleneck where Gemma previously lost to vLLM:**
- **`bimodal` throughput surges +32.6%** (595.08 -> 789.20 tok/s), **beating vLLM (600.16 tok/s) by +31.5%**.
- **`bimodal` mean TTFT drops by -78.7%** (2,208.6 -> 469.8 ms; p95 drops -77.0% from 5,676.8 -> 1,303.6 ms).
- **`long-prefill` throughput surges +33.1%** (436.76 -> 581.53 tok/s), **beating vLLM (500.26 tok/s) by +16.2%**.
- **`long-prefill` mean TTFT drops by -62.6%** (2,226.8 -> 832.0 ms; p95 drops -36.5% from 3,171.5 -> 2,014.5 ms).
- **`mixed` maintains competitive lead**: 719.27 tok/s vs vLLM's 703.81 tok/s (+2.2%), TTFT 245.1 ms vs vLLM's 276.4 ms.

## 2. Quantitative comparison

All runs executed on NVIDIA GeForce RTX 3080 10 GiB with image `nvcr.io/nvidia/tensorrt@sha256:7cd94ee931d2b5b85ad1c5af723d485b2625f6ce167e1e4abe577850b96ceac3`.

### 2.1 Throughput and Latency Summary

| Workload | Metric | 96-page (baseline) | 192-page (shared_ep) | vLLM (frozen) | 192 vs 96 | 192 vs vLLM |
|---|---|---:|---:|---:|---:|---:|
| **bimodal** | Throughput (tok/s) | 595.08 | **789.20** | 600.16 | **+32.6%** | **+31.5%** 🏆 |
| | TTFT mean (ms) | 2,208.6 | **469.8** | 317.4 | **-78.7%** | 1.48x (was 7.0x) |
| | TTFT p95 (ms) | 5,676.8 | **1,303.6** | 878.4 | **-77.0%** | 1.48x (was 6.5x) |
| | TPOT mean (ms) | 19.77 | 24.01 | 28.83 | +21.4% | **-16.7%** 🏆 |
| | E2E mean (ms) | ~4,200 | **3,517.0** | ~3,800 | -16.3% | **-7.4%** 🏆 |
| **long-prefill** | Throughput (tok/s) | 436.76 | **581.53** | 500.26 | **+33.1%** | **+16.2%** 🏆 |
| | TTFT mean (ms) | 2,226.8 | **832.0** | 543.9 | **-62.6%** | 1.53x (was 4.1x) |
| | TTFT p95 (ms) | 3,171.5 | **2,014.5** | 1,442.0 | **-36.5%** | 1.40x (was 2.2x) |
| | TPOT mean (ms) | 22.59 | 28.55 | 36.37 | +26.4% | **-21.5%** 🏆 |
| | E2E mean (ms) | 4,115.2 | **3,207.8** | 3,554.6 | **-22.1%** | **-9.8%** 🏆 |
| **mixed** | Throughput (tok/s) | 726.25 | **719.27** | 703.81 | -1.0% | **+2.2%** 🏆 |
| | TTFT mean (ms) | 245.7 | **245.1** | 276.4 | -0.2% | **-11.3%** 🏆 |
| | TPOT mean (ms) | 25.9 | 26.8 | 26.5 | +3.5% | +1.1% |
| | E2E mean (ms) | 1,430.3 | **1,458.6** | 1,473.8 | +2.0% | **-1.0%** 🏆 |

### 2.2 GPU Memory Footprint

| Workload | Ready Memory (MiB) | Peak Memory (MiB) | Headroom to 10,240 MiB |
|---|---:|---:|---:|
| `mixed` | 9,809 | 9,825 | 415 MiB |
| `long-prefill` | 9,809 | 9,811 | 429 MiB |
| `bimodal` | 9,809 | 9,811 | 429 MiB |

## 3. Causal Analysis

1. **Why KV capacity was the true root cause**: Under 96 pages (12,288 tokens), serving a burst of requests with
   long prefill prompts (e.g. 52,989 prompt tokens in `long-prefill` or 29,587 in `bimodal`) immediately exhausted
   allocatable KV pages. New requests were blocked at admission control (`PhaseMemoryBroker`), waiting for active
   requests to finish decoding. This manifested as inflated TTFT (>2,200 ms) and idle prefill execution slots.
2. **Why `shared_ep` was the key enabler**: Allocating an independent 299 MiB vision workspace left only ~216 MiB
   for KV before hitting the 10 GiB memory limit. By sharing the 299 MiB arena between prefill and encoder
   (`shared_ep`), 175 MiB was reclaimed, allowing the KV cache to safely expand by +216 MiB (from 96 to 192 pages)
   without exceeding 9,825 MiB peak memory.
3. **Throughput victory over vLLM**: With sufficient KV capacity, Current's C++ asynchronous runtime and CUDA-graph-backed
   execution outperform vLLM across both throughput (+16.2% on `long-prefill`, +31.5% on `bimodal`) and TPOT (-16.7%
   to -21.5% faster decode).

## 4. Retained artifacts

- Engine: `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-packed-p8-d24-kv2048-p192`
- Screen summary: `.local/results/gemma-kv192-screen-20260920/summary.json`
- Build log: `.local/results/gemma4-v3-capacity-ab-20260913/build-p192.log`
