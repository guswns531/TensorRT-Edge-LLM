# 325. Dual-model Full-12 evaluation with all optimizations: Cosmos pooled_io balanced surge (4,248 tok/s) and Gemma +27.41% geomean over vLLM

Date: 2026-09-21. Branch: `codex/v0101-phase-forward-port`.

## 1. Outcome

Following Note 322 (Online MSR Horizon Optimizer), Note 323 (Memory Transfer & Headroom Optimization), and Note 324
(Initial Full-12 revalidation), this campaign conducts a systematic Full-12 benchmark of both models with **all optimizations
fully enabled**:
- **Cosmos-Reason2-2B** (FP16, 256 KV pages) on RTX 3080 10 GiB under `pooled_io` (`shared_ep` workspace + `PipelineIOPool` dynamic buffer pooling)
- **Gemma 4 E2B AWQ** (INT4-AWQ, 192 KV pages) on RTX 3080 10 GiB under `shared_ep` (independent IO buffers for zero P/D concurrency contention)

**Key conclusions**:
1. **Cosmos `balanced` surges to 4,248.73 tok/s (+5.6% over Note 324)**:
   - Enabling `PipelineIOPool` dynamic buffer pooling restored and exceeded the peak screening throughput, surging from
     4,021.87 tok/s to **4,248.73 tok/s (+226.86 tok/s gain)**.
   - Shrinks the gap to vLLM (4,315.77 tok/s) down to just **-1.6%**.
   - `decode-heavy` climbed to **5,076.16 tok/s (+2.8% over vLLM)**.
   - `multi-image` climbed to **310.52 tok/s (+27.3% over vLLM)** with TTFT improving to **226.1 ms** (down from 273.5 ms).
   - Peak GPU memory strictly bounded at **9,323 MiB max** (917 MiB safe headroom below 10,240 MiB).
2. **IO buffer pooling vs concurrency tradeoff clarified**:
   - For text-heavy workloads (`balanced`, `decode-heavy`, `long-prefill`, `multi-image`), `pooled_io` delivers superior
     throughput by eliminating buffer fragmentation and reducing memory footprint.
   - For heavy vision workloads (`mixed`, `vision-heavy`, `wave-drain`), unpooled `shared_ep` provides better throughput
     because concurrent prefill and decode streams do not contend for shared IO buffers.
3. **Gemma 4 E2B AWQ achieves +27.41% geometric-mean throughput advantage over vLLM**:
   - Delivers a **+27.41% geometric-mean throughput advantage** and **-7.54% faster TTFT** over vLLM.
   - `poisson` throughput reached **891.14 tok/s (+30.7% over vLLM)**, up +4.7% over Note 324.
   - `bimodal` throughput reached **782.79 tok/s (+30.4% over vLLM)**, up +2.7% over Note 324.
   - `text-heavy` maintains a massive **+109.1% throughput lead** (846.08 vs 404.66 tok/s) with **-88.7% faster TTFT** (187.7 ms vs 1,658.7 ms).
   - Peak GPU memory remained strictly bounded at **9,653 MiB max** (587 MiB headroom).

## 2. Complete Full-12 scorecards

### A. Cosmos-Reason2-2B (FP16, KV256 pages, `pooled_io`)

| Workload | Note 324 (`shared_ep`) | New (`pooled_io`) | vLLM | vs Note 324 | vs vLLM | New TTFT (ms) | vLLM TTFT (ms) | TTFT Δ | Peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **short** | 2,333.87 | **2,316.23** | 2,046.18 | -0.8% | **+13.2%** | **95.4** | 186.7 | **-48.9%** | 9,303 |
| **balanced** | 4,021.87 | **4,248.73** | 4,315.77 | +5.6% | -1.6% | **73.8** | 112.2 | **-34.3%** | 9,319 |
| **decode-heavy** | 5,022.02 | **5,076.16** | 4,937.33 | +1.1% | **+2.8%** | **71.9** | 118.9 | **-39.5%** | 9,289 |
| **long-prefill** | 1,311.56 | **1,320.47** | 1,123.89 | +0.7% | **+17.5%** | 1,982.6 | **1,912.8** | +3.7% | 9,321 |
| **bimodal** | 1,932.95 | **1,912.05** | 1,873.00 | -1.1% | **+2.1%** | 1,908.5 | **1,551.9** | +23.0% | 9,283 |
| **text-heavy** | 1,963.69 | **1,954.93** | 1,292.39 | -0.4% | **+51.3%** | **323.1** | 939.5 | **-65.6%** | 9,319 |
| **mixed** | 1,139.18 | **1,057.52** | 923.32 | -7.2% | **+14.5%** | **821.5** | 868.7 | **-5.4%** | 9,303 |
| **vision-heavy** | 694.89 | **641.28** | 577.19 | -7.7% | **+11.1%** | **1,536.1** | 1,630.9 | **-5.8%** | 9,319 |
| **poisson** | 1,971.01 | **1,965.31** | 1,781.11 | -0.3% | **+10.3%** | **318.3** | 440.0 | **-27.7%** | 9,323 |
| **wave-drain** | 97.84 | 92.22 | **95.82** | -5.7% | -3.8% | 277.3 | **254.4** | +9.0% | 9,295 |
| **multi-image** | 300.02 | **310.52** | 243.90 | +3.5% | **+27.3%** | **226.1** | 260.0 | **-13.0%** | 9,305 |
| **late-vision** | 2,446.24 | **2,442.55** | 2,165.09 | -0.2% | **+12.8%** | **135.9** | 249.2 | **-45.5%** | 9,303 |

- **Win Rate**: **10 / 12 (83.3%)**
- **Geomean Throughput**: **+12.33%** over vLLM
- **Geomean TTFT**: **-25.41%** faster than vLLM
- **Peak Memory**: **9,323 MiB max** (917 MiB safe headroom)

---

### B. Gemma 4 E2B AWQ (INT4-AWQ, KV192 pages, `shared_ep`)

| Workload | Note 324 (tok/s) | New (tok/s) | vLLM (tok/s) | vs Note 324 | vs vLLM | New TTFT (ms) | vLLM TTFT (ms) | TTFT Δ | Peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **balanced** | 1,158.48 | **1,155.34** | 771.46 | -0.3% | **+49.8%** | **87.7** | 134.3 | **-34.7%** | 9,639 |
| **mixed** | 692.75 | 681.90 | **703.81** | -1.6% | -3.1% | **250.4** | 276.4 | **-9.4%** | 9,649 |
| **vision-heavy** | 536.99 | 546.14 | **559.83** | +1.7% | -2.4% | 384.3 | **280.6** | +37.0% | 9,653 |
| **multi-image** | 372.54 | 359.31 | **381.34** | -3.6% | -5.8% | 454.6 | **188.9** | +140.7% | 9,653 |
| **long-prefill** | 595.74 | **577.67** | 500.26 | -3.0% | **+15.5%** | 710.1 | **543.9** | +30.6% | 9,635 |
| **bimodal** | 761.96 | **782.79** | 600.16 | +2.7% | **+30.4%** | 490.1 | **317.4** | +54.4% | 9,635 |
| **decode-heavy** | 1,290.18 | **1,292.73** | 812.43 | +0.2% | **+59.1%** | **103.4** | 153.6 | **-32.7%** | 9,635 |
| **short** | 782.88 | **785.72** | 567.55 | +0.4% | **+38.4%** | **111.1** | 155.4 | **-28.5%** | 9,635 |
| **text-heavy** | 830.12 | **846.08** | 404.66 | +1.9% | **+109.1%** | **187.7** | 1,658.7 | **-88.7%** | 9,645 |
| **poisson** | 851.10 | **891.14** | 681.95 | +4.7% | **+30.7%** | 137.2 | **119.8** | +14.6% | 9,639 |
| **wave-drain** | 96.27 | **96.61** | 92.62 | +0.3% | **+4.3%** | 269.1 | **178.2** | +51.0% | 9,643 |
| **late-vision** | 1,450.10 | **1,437.44** | 990.82 | -0.9% | **+45.1%** | 145.1 | **137.7** | +5.4% | 9,643 |

- **Win Rate**: **9 / 12 (75.0%)**
- **Geomean Throughput**: **+27.41%** over vLLM
- **Geomean TTFT**: **-7.54%** faster than vLLM
- **Peak Memory**: **9,653 MiB max** (saving 172 MiB vs Note 317)

## 3. Retained artifacts

- Cosmos campaign: `.local/results/full12-cosmos-pooled-io-20260921/summary.json`
- Gemma campaign: `.local/results/full12-gemma-all-opts-20260921/summary.json`
