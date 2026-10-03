# 375 — v0.11.0 port: final-tree validation (2026-10-03)

Revises the limits of note 374, whose throughput and MMLU numbers were measured on `9f911c16` before the six review
fixes.

## Outcome
- Final tree `f0849333` (binary `.local/baselines/v0110-port-f0849333-20261003`) keeps output quality: MMLU
  zero-shot serving 51.27% (7193/14031), equal to the port batch-1 reference (51.27%), same prediction on 14029/14031.
- Throughput is unchanged from `9f911c16` within run-to-run noise on 23 of 24 cells (Gemma geomean −0.16%, Cosmos
  +0.27%). Against frozen vLLM the geomean is +28.0% (Gemma +40.4%, Cosmos +16.6%); against the v0.10.1 tip +0.7%
  (Gemma +2.5%, Cosmos −1.1%).
- **Cosmos balanced is now below frozen vLLM**: median 4251.9 vs 4315.8 tok/s (−1.5%), all three runs below
  (4251.9, 4301.3, 4246.5). These are the only 3 of 72 runs below vLLM (minimum run ratio 0.984). The cell has declined
  monotonically through the port: v0.10.1 tip 4446.7, pre-fix port 4372.6, `9f911c16` 4334.9, final 4251.9 (medians).
  The last step (−1.9%) is outside the other Cosmos cells' movement (−0.2% to +2.1%), so it is not explained by noise
  alone; its cause is not established.

## Throughput (full24 ×3, generated tok/s, median of 3)
Same engines, traces, repeat count and serving contract as note 374; methodology in
`benchmarks/phase_serving/EXPERIMENT_METHODOLOGY.md`.

| Model | Workload | vLLM | v0.10.1 | Port pre-fix `f6c2f094` | **Port final `f0849333`** | Final vs vLLM | Final vs v0.10.1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma | balanced | 771.5 | 1262.6 | 1271.9 | **1296.8** | +68.1% | +2.7% |
| Gemma | bimodal | 600.2 | 857.9 | 927.2 | **929.8** | +54.9% | +8.4% |
| Gemma | decode-heavy | 812.4 | 1376.8 | 1391.5 | **1400.6** | +72.4% | +1.7% |
| Gemma | late-vision | 990.8 | 1528.5 | 1538.3 | **1548.8** | +56.3% | +1.3% |
| Gemma | long-prefill | 500.3 | 616.9 | 640.8 | **643.4** | +28.6% | +4.3% |
| Gemma | mixed | 703.8 | 772.5 | 759.2 | **778.2** | +10.6% | +0.7% |
| Gemma | multi-image | 381.3 | 411.7 | 413.1 | **403.1** | +5.7% | -2.1% |
| Gemma | poisson | 681.9 | 932.1 | 950.8 | **958.3** | +40.5% | +2.8% |
| Gemma | short | 567.5 | 845.8 | 864.1 | **900.5** | +58.7% | +6.5% |
| Gemma | text-heavy | 404.7 | 898.1 | 905.4 | **920.7** | +127.5% | +2.5% |
| Gemma | vision-heavy | 559.8 | 573.2 | 578.5 | **583.3** | +4.2% | +1.8% |
| Gemma | wave-drain | 92.6 | 97.2 | 97.2 | **97.3** | +5.0% | +0.0% |
| Cosmos | balanced | 4315.8 | 4446.7 | 4372.6 | **4251.9** | -1.5% | -4.4% |
| Cosmos | bimodal | 1873.0 | 2018.6 | 1991.7 | **1979.1** | +5.7% | -2.0% |
| Cosmos | decode-heavy | 4937.3 | 5238.5 | 5157.2 | **5139.0** | +4.1% | -1.9% |
| Cosmos | late-vision | 2165.1 | 2517.9 | 2475.1 | **2478.6** | +14.5% | -1.6% |
| Cosmos | long-prefill | 1123.9 | 1315.9 | 1289.7 | **1313.1** | +16.8% | -0.2% |
| Cosmos | mixed | 923.3 | 1181.0 | 1163.6 | **1172.7** | +27.0% | -0.7% |
| Cosmos | multi-image | 243.9 | 313.6 | 306.0 | **307.9** | +26.3% | -1.8% |
| Cosmos | poisson | 1781.1 | 2047.2 | 2030.0 | **2050.9** | +15.1% | +0.2% |
| Cosmos | short | 2046.2 | 2392.1 | 2313.9 | **2393.5** | +17.0% | +0.1% |
| Cosmos | text-heavy | 1292.4 | 2028.7 | 1988.7 | **2022.1** | +56.5% | -0.3% |
| Cosmos | vision-heavy | 577.2 | 732.6 | 730.1 | **732.3** | +26.9% | -0.0% |
| Cosmos | wave-drain | 95.8 | 98.0 | 97.8 | **97.9** | +2.1% | -0.1% |
| **Gemma geomean** | | | | | | **+40.4%** | **+2.5%** |
| **Cosmos geomean** | | | | | | **+16.6%** | **-1.1%** |
| **All 24 geomean** | | | | | | **+28.0%** | **+0.7%** |

Movement against `9f911c16` (note 374): Cosmos balanced −1.9%, short +2.1%, mixed +1.3%; Gemma text-heavy −1.5%;
all other cells within ±0.7%.

## Open
- Cosmos balanced is the decode-dominated 288-request Cosmos trace at in-flight 64. Its gap to v0.10.1 (−4.4%) and to
  vLLM (−1.5%) is the remaining port regression; a same-day A/B of `9f911c16` against `f0849333` on that cell, then a
  decode-step profile, would separate the review-fix contribution (on the Cosmos path the candidate is the attention
  plugin's per-enqueue workspace bound check and workspace recomputation; Cosmos has no shared-KV layers) from
  day-to-day variance.

## Retained paths
- `.local/results/v0110-port-final-full24-3x-20261003` (diagnostic)
- `.local/results/v0110-port-final-mmlu-20261003` (diagnostic; batch-1 reference reused from `v0110-port-mmlu-20261001`)
- `.local/baselines/v0110-port-f0849333-20261003` (frozen binary with manifest)
