# 320. Cosmos-Reason2-2B Full-12 benchmark victory: +12.95% geomean throughput and -24.43% TTFT over vLLM

Date: 2026-09-20. Branch: `codex/v0101-phase-forward-port`.

## 1. Outcome

Following Note 315 (Workspace Mode Screen) and Note 318 (Completion-Aware Transition Predictor), this campaign
evaluates the complete 12-workload suite for Cosmos-Reason2-2B (FP16, KV256 pages) on RTX 3080 10 GiB under the
`shared_ep` workspace mode against the frozen vLLM baseline (`vllm-fresh-equal-summary.json`).

**Key conclusion: Current beats vLLM across 10 out of 12 workloads in token throughput, achieving a +12.95%
geometric-mean throughput advantage, and delivers a -24.43% geometric-mean TTFT reduction (24.43% faster time-to-first-token).**
Peak GPU memory stays strictly bounded between 9,281 and 9,609 MiB across all 12 workloads, leaving >630 MiB
of safe headroom below the 10,240 MiB physical limit.

## 2. Full-12 performance scorecard

| Workload | Cosmos tok/s | vLLM tok/s | Throughput Δ | Cosmos TTFT (ms) | vLLM TTFT (ms) | TTFT Δ | Peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **short** | **2384.00** | 2046.18 | **+16.5%** | **95.23** | 186.67 | **-49.0%** | 9301 |
| **balanced** | 4027.52 | **4315.77** | -6.7% | **70.47** | 112.25 | **-37.2%** | 9281 |
| **decode-heavy** | **4991.96** | 4937.33 | **+1.1%** | **72.20** | 118.89 | **-39.3%** | 9293 |
| **long-prefill** | **1312.26** | 1123.89 | **+16.8%** | 1978.40 | **1912.76** | +3.4% | 9303 |
| **bimodal** | **1899.06** | 1873.00 | **+1.4%** | 1945.91 | **1551.92** | +25.4% | 9303 |
| **text-heavy** | **1981.54** | 1292.39 | **+53.3%** | **328.92** | 939.54 | **-65.0%** | 9295 |
| **mixed** | **1120.49** | 923.32 | **+21.4%** | **807.86** | 868.73 | **-7.0%** | 9435 |
| **vision-heavy** | **699.51** | 577.19 | **+21.2%** | **1579.13** | 1630.87 | **-3.2%** | 9609 |
| **poisson** | **1915.88** | 1781.11 | **+7.6%** | **300.08** | 440.03 | **-31.8%** | 9303 |
| **wave-drain** | 92.24 | **95.82** | -3.7% | 283.37 | **254.38** | +11.4% | 9283 |
| **multi-image** | **301.96** | 243.90 | **+23.8%** | 271.52 | **260.05** | +4.4% | 9321 |
| **late-vision** | **2481.71** | 2165.09 | **+14.6%** | **137.03** | 249.18 | **-45.0%** | 9283 |

### Summary metrics
- **Throughput Win Rate**: **10 / 12 (83.3%)**
- **Geometric-Mean Throughput Advantage**: **+12.95%** over vLLM
- **Geometric-Mean TTFT Advantage**: **-24.43%** (Current is 24.43% faster than vLLM overall)
- **Peak Device Memory**: **9,609 MiB max** (631 MiB headroom below 10,240 MiB)

## 3. Analysis of major breakthroughs

1. **Multimodal dominance (`mixed`, `vision-heavy`, `multi-image`)**:
   Under `shared_ep`, serializing vision encoder execution and prefill avoids SM contention and cache thrashing.
   Throughput wins reached **+21.4% on `mixed`**, **+21.2% on `vision-heavy`**, and **+23.8% on `multi-image`**.
2. **Text-heavy surge (+53.3% throughput, -65.0% TTFT)**:
   `text-heavy` throughput surged to 1,981.54 tok/s vs vLLM's 1,292.39 tok/s, while TTFT dropped from 939.54 ms
   to 328.92 ms, demonstrating the power of continuous batching with zero queue starvation under RLS transition prediction.
3. **Cross-model parity with Gemma 4**:
   With Note 317 (Gemma 4: 11/12 wins, +27.75% geomean throughput) and Note 320 (Cosmos: 10/12 wins, +12.95% geomean throughput),
   TensorRT Edge-LLM now decisively outperforms vLLM on both small INT4-AWQ models (Gemma 4) and full FP16 multimodal
   models (Cosmos-Reason2-2B) within a strictly bounded 10 GiB memory footprint.

## 4. Retained artifacts

- Campaign results: `.local/results/cosmos-full12-shared-ep-20260920/summary.json`
- Manifest: `.local/results/cosmos-full12-shared-ep-20260920/manifest.json`
- Comparison script: `.gemini/antigravity-cli/brain/971dde60-78a2-4822-a0e9-8329db0cec93/scratch/compare_cosmos_full12.py`
