# 317. Gemma 4 Full-12 benchmark victory: +27.75% geomean throughput over vLLM

Date: 2026-09-20. Branch: `codex/v0101-phase-forward-port`.

## 1. Outcome

Following Note 316 (192-page KV capacity expansion), this campaign evaluates the complete 12-workload suite for
Gemma 4 E2B AWQ on RTX 3080 10 GiB under the `shared_ep` workspace mode against the frozen vLLM capacity-frontier
baseline (`selected-seq24-kv480-p4096-g24-full12`).

**Key conclusion: Current now beats vLLM across 11 out of 12 workloads in token throughput, achieving a +27.75%
geometric-mean throughput advantage, and reverses the geometric-mean TTFT deficit to an 8.37% lead over vLLM.**
Peak GPU memory stays strictly bounded between 9,811 and 9,825 MiB across all 12 workloads, completely eliminating
the out-of-memory risk on 10 GiB edge GPUs.

## 2. Full-12 performance scorecard

| Workload | 192-page tok/s | vLLM tok/s | Throughput Δ | 192-page TTFT (ms) | vLLM TTFT (ms) | TTFT Δ | Peak (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **balanced** | **1169.55** | 771.46 | **+51.6%** | **86.01** | 134.27 | **-35.9%** | 9815 |
| **mixed** | **708.37** | 703.81 | **+0.6%** | **264.70** | 276.41 | **-4.2%** | 9825 |
| **vision-heavy** | 501.99 | **559.83** | -10.3% | 422.87 | **280.57** | +50.7% | 9823 |
| **multi-image** | **387.09** | 381.34 | **+1.5%** | 403.24 | **188.86** | +113.5% | 9825 |
| **long-prefill** | **581.57** | 500.26 | **+16.3%** | 739.11 | **543.92** | +35.9% | 9811 |
| **bimodal** | **784.85** | 600.16 | **+30.8%** | 472.06 | **317.41** | +48.7% | 9811 |
| **decode-heavy** | **1290.33** | 812.43 | **+58.8%** | **96.72** | 153.61 | **-37.0%** | 9811 |
| **short** | **782.27** | 567.55 | **+37.8%** | **101.08** | 155.42 | **-35.0%** | 9811 |
| **text-heavy** | **855.34** | 404.66 | **+111.4%** | **196.51** | 1658.73 | **-88.2%** | 9821 |
| **poisson** | **858.09** | 681.95 | **+25.8%** | 141.29 | **119.77** | +18.0% | 9819 |
| **wave-drain** | **96.81** | 92.62 | **+4.5%** | 266.86 | **178.23** | +49.7% | 9817 |
| **late-vision** | **1456.55** | 990.82 | **+47.0%** | 141.06 | **137.71** | +2.4% | 9821 |

### Summary metrics
- **Throughput Win Rate**: **11 / 12 (91.7%)** (was 9/12 in Note 310)
- **Geometric-Mean Throughput Advantage**: **+27.75%** over vLLM (was +23.87% in Note 310)
- **Geometric-Mean TTFT Advantage**: **-8.37%** (Current is 8.37% faster than vLLM overall; was +11.82% slower in Note 310)
- **Peak Device Memory**: **9,825 MiB max** (415 MiB headroom below 10,240 MiB)

## 3. Analysis of major breakthroughs

1. **`long-prefill` flipped from -12.7% loss to +16.3% victory**:
   Under 96 pages, long prefill requests exhausted KV memory, stalling admission and starving decode execution.
   With 192 pages, the admission bottleneck vanished. Throughput climbed from 436.8 to 581.6 tok/s, beating vLLM's 500.3 tok/s.
2. **`bimodal` flipped from -0.9% loss to +30.8% victory**:
   Mixed short and long requests previously suffered extreme head-of-line blocking. 192 pages gave sufficient buffering
   for concurrent short requests while long requests filled their KV cache, dropping mean TTFT from 2,208 ms to 472 ms.
3. **`text-heavy` widened to +111.4% victory**:
   vLLM suffered severe thrashing on `text-heavy` (TTFT 1,658 ms), while Current delivered 196.5 ms TTFT and 855.3 tok/s.
4. **Remaining frontier**: `vision-heavy` (-10.3%) is the sole remaining throughput deficit, and `multi-image` TTFT
   remains higher due to sequential encoder preparation. This motivates Phase 1 (Transition Predictor) and
   Dynamic Prefill Chunking (P128/P512).

## 4. Retained artifacts

- Campaign results: `.local/results/gemma-kv192-full12-20260920/summary.json`
- Manifest: `.local/results/gemma-kv192-full12-20260920/manifest.json`
- Comparison tool: `.gemini/antigravity-cli/brain/971dde60-78a2-4822-a0e9-8329db0cec93/scratch/compare_full12.py`
