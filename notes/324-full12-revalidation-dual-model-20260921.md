# 324. Dual-model Full-12 revalidation: 11/12 Cosmos victory (+13.59% tput, -27.87% TTFT) and +27.16% Gemma geomean over vLLM

Date: 2026-09-21. Branch: `codex/v0101-phase-forward-port`.

## 1. Outcome

Following Note 322 (Online MSR Horizon Optimizer) and Note 323 (Memory Transfer & Headroom Optimization), this
campaign conducts an exhaustive end-to-end re-evaluation of all 12 workloads across both models:
- **Gemma 4 E2B AWQ** (192 KV pages) on RTX 3080 10 GiB under `shared_ep`
- **Cosmos-Reason2-2B** (FP16, 256 KV pages) on RTX 3080 10 GiB under `shared_ep`

**Key conclusions**:
1. **Cosmos-Reason2-2B improves to 11 / 12 (91.7%) throughput win rate over vLLM**:
   - `wave-drain` flipped from a loss in Note 320 (92.24 tok/s, -3.7%) to a **win (97.84 tok/s, +2.1%)**, with TTFT dropping to 243.0 ms (-4.5% vs vLLM).
   - Geometric-mean throughput advantage expanded to **+13.59%** over vLLM (was +12.95% in Note 320).
   - Geometric-mean TTFT advantage expanded to **-27.87%** faster than vLLM (was -24.43% in Note 320).
   - Cosmos `poisson` TTFT reached **222.3 ms** (-49.5% vs vLLM 440.0 ms, and -25.9% vs Note 320's 300.1 ms).
   - Peak GPU memory remained strictly bounded at **9,487 MiB max** (753 MiB safe headroom below 10,240 MiB).
2. **Gemma 4 E2B AWQ sustains decisive dominance (+27.16% geomean throughput, -9.79% TTFT)**:
   - Delivers a **+27.16% geometric-mean throughput advantage** and **-9.79% faster TTFT** over vLLM.
   - `vision-heavy` throughput improved from 501.99 tok/s (Note 317) to **536.99 tok/s (+7.0%)**, with TTFT improving from 422.9 ms to **372.9 ms**.
   - Peak GPU memory dropped from 9,825 MiB to **9,653 MiB max** (saving 172 MiB, leaving 587 MiB headroom).

## 2. Complete Full-12 scorecards

### A. Cosmos-Reason2-2B (FP16, KV256 pages)

| Workload | Note 320 tok/s | New tok/s | vLLM tok/s | vs Note 320 | vs vLLM | New TTFT (ms) | vLLM TTFT (ms) | TTFT Δ | Peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **short** | 2,384.00 | **2,333.87** | 2,046.18 | -2.1% | **+14.1%** | **99.0** | 186.7 | **-47.0%** | 9,319 |
| **balanced** | 4,027.52 | 4,021.87 | **4,315.77** | -0.1% | -6.8% | **64.4** | 112.2 | **-42.6%** | 9,313 |
| **decode-heavy** | 4,991.96 | **5,022.02** | 4,937.33 | +0.6% | **+1.7%** | **73.1** | 118.9 | **-38.5%** | 9,295 |
| **long-prefill** | 1,312.26 | **1,311.56** | 1,123.89 | -0.1% | **+16.7%** | 2,004.8 | **1,912.8** | +4.8% | 9,295 |
| **bimodal** | 1,899.06 | **1,932.95** | 1,873.00 | +1.8% | **+3.2%** | 1,938.4 | **1,551.9** | +24.9% | 9,311 |
| **text-heavy** | 1,981.54 | **1,963.69** | 1,292.39 | -0.9% | **+51.9%** | **323.5** | 939.5 | **-65.6%** | 9,295 |
| **mixed** | 1,120.49 | **1,139.18** | 923.32 | +1.7% | **+23.4%** | **808.2** | 868.7 | **-7.0%** | 9,465 |
| **vision-heavy** | 699.51 | **694.89** | 577.19 | -0.7% | **+20.4%** | **1,494.0** | 1,630.9 | **-8.4%** | 9,487 |
| **poisson** | 1,915.88 | **1,971.01** | 1,781.11 | +2.9% | **+10.7%** | **222.3** | 440.0 | **-49.5%** | 9,293 |
| **wave-drain** | 92.24 | **97.84** | 95.82 | +6.1% | **+2.1%** | **243.0** | 254.4 | **-4.5%** | 9,283 |
| **multi-image** | 301.96 | **300.02** | 243.90 | -0.6% | **+23.0%** | 273.5 | **260.0** | +5.2% | 9,293 |
| **late-vision** | 2,481.71 | **2,446.24** | 2,165.09 | -1.4% | **+13.0%** | **135.6** | 249.2 | **-45.6%** | 9,303 |

- **Win Rate**: **11 / 12 (91.7%)** (up from 10/12 in Note 320)
- **Geomean Throughput**: **+13.59%** over vLLM
- **Geomean TTFT**: **-27.87%** faster than vLLM
- **Peak Memory**: **9,487 MiB max**

---

### B. Gemma 4 E2B AWQ (INT4-AWQ, KV192 pages)

| Workload | Note 317 tok/s | New tok/s | vLLM tok/s | vs Note 317 | vs vLLM | New TTFT (ms) | vLLM TTFT (ms) | TTFT Δ | Peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **balanced** | 1,169.55 | **1,158.48** | 771.46 | -0.9% | **+50.2%** | **92.5** | 134.3 | **-31.1%** | 9,635 |
| **mixed** | 708.37 | 692.75 | **703.81** | -2.2% | -1.6% | **248.2** | 276.4 | **-10.2%** | 9,647 |
| **vision-heavy** | 501.99 | 536.99 | **559.83** | +7.0% | -4.1% | 372.9 | **280.6** | +32.9% | 9,653 |
| **multi-image** | 387.09 | 372.54 | **381.34** | -3.8% | -2.3% | 404.5 | **188.9** | +114.2% | 9,653 |
| **long-prefill** | 581.57 | **595.74** | 500.26 | +2.4% | **+19.1%** | 743.5 | **543.9** | +36.7% | 9,635 |
| **bimodal** | 784.85 | **761.96** | 600.16 | -2.9% | **+27.0%** | 446.8 | **317.4** | +40.8% | 9,635 |
| **decode-heavy** | 1,290.33 | **1,290.18** | 812.43 | -0.0% | **+58.8%** | **97.7** | 153.6 | **-36.4%** | 9,635 |
| **short** | 782.27 | **782.88** | 567.55 | +0.1% | **+37.9%** | **108.7** | 155.4 | **-30.1%** | 9,635 |
| **text-heavy** | 855.34 | **830.12** | 404.66 | -2.9% | **+105.1%** | **169.7** | 1,658.7 | **-89.8%** | 9,645 |
| **poisson** | 858.09 | **851.10** | 681.95 | -0.8% | **+24.8%** | 141.3 | **119.8** | +17.9% | 9,643 |
| **wave-drain** | 96.81 | **96.27** | 92.62 | -0.6% | **+3.9%** | 272.6 | **178.2** | +52.9% | 9,641 |
| **late-vision** | 1,456.55 | **1,450.10** | 990.82 | -0.4% | **+46.4%** | 143.7 | **137.7** | +4.4% | 9,645 |

- **Win Rate**: **9 / 12 (75.0%)**
- **Geomean Throughput**: **+27.16%** over vLLM
- **Geomean TTFT**: **-9.79%** faster than vLLM
- **Peak Memory**: **9,653 MiB max** (saving 172 MiB vs Note 317)

## 3. Retained artifacts

- Campaign results: `.local/results/full12-memory-opt-validation-20260920/summary.json`
- Manifest: `.local/results/full12-memory-opt-validation-20260920/manifest.json`
