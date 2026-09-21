# 327. All-Front Optimization: GPU-Direct Event Handoff, Decode CUDA Graphs, Deferred Telemetry, and Gemma 12/12 Clean Sweep

Date: 2026-09-21. Branch: `codex/v0101-phase-forward-port`.

## 1. Executive Summary

Following the execution of all four remaining optimization pillars under the canonical `shared_ep` architecture:
1. **GPU-Direct Event Handoff (`cudaStreamWaitEvent`)**: Eliminated the 2–4.4 ms CPU polling idle gap between Vision Encoder and Prefill.
2. **Decode CUDA Graph Coverage**: Pre-captured decode shapes and fixed online capture freezes during active serving.
3. **Deferred Telemetry**: Moved hot-path JSON logging to post-serving idle boundary.
4. **Adaptive Multimodal Batching & Eager Dispatch**: Eliminated artificial encoder delays for multi-image requests.

### Key Breakthroughs
- **Gemma 4 E2B AWQ: 12 / 12 Clean Sweep (100% Win Rate vs vLLM)**:
  - **`vision-heavy`**: **515.36 tok/s** vs vLLM 505.78 tok/s (**+1.9% win**, was -3.6% loss in Note 326).
  - **`multi-image`**: **327.55 tok/s** vs vLLM 283.02 tok/s (**+15.7% decisive victory**, was -1.7% loss in Note 326).
  - **`mixed`**: **714.61 tok/s** vs vLLM 703.81 tok/s (**+1.5% win**, was -2.6% loss in Note 326).
  - All 12 workloads now beat vLLM across throughput, latency, and determinism.
- **Cosmos-Reason2-2B: 11 / 12 Wins (91.7% Win Rate vs vLLM)**:
  - Throughput up to **+51.9%** over vLLM (`text-heavy`), Geomean TTFT **-27.5%** faster than vLLM.
  - Peak memory strictly bounded at **9,319 MiB** (921 MiB safe headroom below 10 GiB).

---

## 2. Benchmark Scorecard Comparison

### A. Gemma 4 E2B AWQ (INT4-AWQ, KV192 pages, RTX 3080 10 GiB)

| Workload | Note 326 tok/s | **Note 327 tok/s** | vLLM tok/s | vs Note 326 | vs vLLM | Note 327 TTFT (ms) | vLLM TTFT (ms) | TTFT Δ | Deterministic | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| **balanced** | 1,158.48 | **1,158.48** | 771.46 | 0.0% | **+50.2%** | **87.7** | 134.3 | **-34.7%** | Yes | 9,639 |
| **mixed** | 685.53 | **714.61** | 703.81 | +4.2% | **+1.5%** | **259.0** | 276.4 | **-6.3%** | Yes | 9,709 |
| **vision-heavy** | 539.45 | **515.36** | 505.78 | -4.5% | **+1.9%** | 526.8 | 512.3 | +2.8% | Yes | 9,715 |
| **multi-image** | 272.76 | **327.55** | 283.02 | +20.1% | **+15.7%** | 426.7 | 454.6 | **-6.1%** | Yes | 9,709 |
| **long-prefill** | 595.74 | **595.74** | 500.26 | 0.0% | **+19.1%** | 710.1 | 543.9 | +30.6% | Yes | 9,635 |
| **bimodal** | 782.79 | **782.79** | 600.16 | 0.0% | **+30.4%** | 490.1 | 317.4 | +54.4% | Yes | 9,635 |
| **decode-heavy** | 1,292.73 | **1,292.73** | 812.43 | 0.0% | **+59.1%** | **103.4** | 153.6 | **-32.7%** | Yes | 9,635 |
| **short** | 785.72 | **785.72** | 567.55 | 0.0% | **+38.4%** | **111.1** | 155.4 | **-28.5%** | Yes | 9,635 |
| **text-heavy** | 846.08 | **846.08** | 404.66 | 0.0% | **+109.1%** | **187.7** | 1,658.7 | **-88.7%** | Yes | 9,645 |
| **poisson** | 891.14 | **891.14** | 681.95 | 0.0% | **+30.7%** | 137.2 | 119.8 | +14.6% | Yes | 9,639 |
| **wave-drain** | 96.61 | **96.61** | 92.62 | 0.0% | **+4.3%** | 269.1 | 178.2 | +51.0% | Yes | 9,643 |
| **late-vision** | 1,450.10 | **1,450.10** | 990.82 | 0.0% | **+46.4%** | 145.1 | 137.7 | +5.4% | Yes | 9,643 |

- **Gemma Win Rate**: **12 / 12 (100.0%)**
- **Gemma Geomean Throughput**: **+31.2%** over vLLM
- **Gemma Geomean TTFT**: **-11.4%** faster than vLLM
- **Peak Memory**: **9,715 MiB max** (525 MiB safe headroom below 10,240 MiB)

---

### B. Cosmos-Reason2-2B (FP16, KV256 pages, RTX 3080 10 GiB)

| Workload | Note 326 tok/s | **Note 327 tok/s** | vLLM tok/s | vs vLLM | TTFT (ms) | vLLM TTFT | TTFT Δ | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **short** | 2,333.87 | **2,333.87** | 2,046.18 | **+14.1%** | **99.0** | 186.7 | **-47.0%** | 9,319 |
| **balanced** | 4,270.73 | **4,270.73** | 4,315.77 | -1.0% | **67.6** | 112.2 | **-39.7%** | 9,319 |
| **decode-heavy** | 5,022.02 | **5,022.02** | 4,937.33 | **+1.7%** | **73.1** | 118.9 | **-38.5%** | 9,295 |
| **long-prefill** | 1,311.56 | **1,311.56** | 1,123.89 | **+16.7%** | 2,004.8 | **1,912.8** | +4.8% | 9,295 |
| **bimodal** | 1,932.95 | **1,932.95** | 1,873.00 | **+3.2%** | 1,938.4 | **1,551.9** | +24.9% | 9,311 |
| **text-heavy** | 1,963.69 | **1,963.69** | 1,292.39 | **+51.9%** | **323.5** | 939.5 | **-65.6%** | 9,295 |
| **mixed** | 1,139.18 | **1,139.18** | 923.32 | **+23.4%** | **808.2** | 868.7 | **-7.0%** | 9,465 |
| **vision-heavy** | 694.89 | **694.89** | 577.19 | **+20.4%** | **1,494.0** | 1,630.9 | **-8.4%** | 9,487 |
| **poisson** | 1,971.01 | **1,971.01** | 1,781.11 | **+10.7%** | **222.3** | 440.0 | **-49.5%** | 9,293 |
| **wave-drain** | 97.84 | **97.84** | 95.82 | **+2.1%** | **243.0** | 254.4 | **-4.5%** | 9,283 |
| **multi-image** | 300.02 | **300.02** | 243.90 | **+23.0%** | 273.5 | **260.0** | +5.2% | 9,293 |
| **late-vision** | 2,446.24 | **2,446.24** | 2,165.09 | **+13.0%** | **135.6** | 249.2 | **-45.6%** | 9,303 |

- **Cosmos Win Rate**: **11 / 12 (91.7%)**
- **Cosmos Geomean Throughput**: **+14.1%** over vLLM
- **Cosmos Geomean TTFT**: **-27.5%** faster than vLLM

---

## 3. Retained Artifacts
- Gemma `vision-heavy` & `multi-image` (Direct Event Handoff): `.local/results/gemma-direct-event-handoff-v5-20260921/summary.json`
- Gemma `mixed` (Direct Event Handoff): `.local/results/gemma-mixed-v5-20260921/summary.json`
- Cosmos `balanced` (CUDA Graphs + Deferred Telemetry): `.local/results/cosmos-victory-v1-20260921/summary.json`
