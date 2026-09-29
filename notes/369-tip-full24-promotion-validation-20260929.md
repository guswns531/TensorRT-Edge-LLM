# Tip Full24 x3 Promotion Validation

## Outcome

The current tip (binary `.local/baselines/int4-force-gemm-863a6d4-20260929`, source `863a6d4`; later commits change
only the harness and notes) completed Gemma + Cosmos full24 x3 under the note 362 contract with default settings
(INT4 GEMV override unset, idle host wait off). 72/72 cells, every run above frozen vLLM, no integrity issues.
Throughput equals `c7a1b71` (note 366) within noise: geomean -0.12%, no workload with separated run ranges. Combined
with the note 368 output-quality result, this binary is a promotion candidate for `.local/current/active/runtime`.

Campaign: `.local/results/tip-full24-3x-20260929` (`summary.json`, `output-audit.{json,md}`). vLLM and 9db3 anchors are
the frozen references used in notes 362 and 366.

## Throughput (median of three runs)

| Workload | tok/s | vs vLLM | vs 9db3 | vs c7a1b71 | Range |
|---|---:|---:|---:|---:|---|
| Cosmos balanced | 4446.7 | +3.0% | +0.0% | -1.6% | 4390-4534 |
| Cosmos bimodal | 2018.6 | +7.8% | +3.6% | +2.2% | 1998-2020 |
| Cosmos decode-heavy | 5238.5 | +6.1% | -0.1% | +0.4% | 5227-5252 |
| Cosmos late-vision | 2517.9 | +16.3% | -0.1% | -0.7% | 2517-2534 |
| Cosmos long-prefill | 1315.9 | +17.1% | +0.3% | -0.7% | 1302-1323 |
| Cosmos mixed | 1181.0 | +27.9% | +3.7% | +0.5% | 1163-1190 |
| Cosmos multi-image | 313.6 | +28.6% | +0.7% | -0.5% | 312-314 |
| Cosmos poisson | 2047.2 | +15.0% | +1.2% | -0.6% | 2039-2073 |
| Cosmos short | 2392.1 | +16.9% | -0.7% | -0.3% | 2386-2497 |
| Cosmos text-heavy | 2028.7 | +57.0% | +1.9% | -1.5% | 2015-2044 |
| Cosmos vision-heavy | 732.6 | +26.9% | +2.7% | -0.5% | 732-743 |
| Cosmos wave-drain | 98.0 | +2.2% | -0.1% | +0.0% | 98-98 |
| Gemma balanced | 1262.6 | +63.7% | +0.0% | -0.6% | 1260-1289 |
| Gemma bimodal | 857.9 | +42.9% | +1.3% | +0.6% | 850-859 |
| Gemma decode-heavy | 1376.8 | +69.5% | -0.1% | -0.3% | 1375-1385 |
| Gemma late-vision | 1528.5 | +54.3% | -0.2% | +0.4% | 1525-1531 |
| Gemma long-prefill | 616.9 | +23.3% | -0.4% | +0.6% | 613-619 |
| Gemma mixed | 772.5 | +9.8% | +3.8% | -0.4% | 767-793 |
| Gemma multi-image | 411.7 | +7.9% | +6.3% | +3.1% | 403-414 |
| Gemma poisson | 932.1 | +36.7% | +0.3% | -1.5% | 927-947 |
| Gemma short | 845.8 | +49.1% | -1.0% | -0.3% | 833-850 |
| Gemma text-heavy | 898.1 | +121.9% | +0.4% | +0.1% | 897-903 |
| Gemma vision-heavy | 573.2 | +2.4% | +0.4% | -1.2% | 573-575 |
| Gemma wave-drain | 97.2 | +4.9% | -0.1% | -0.0% | 97-97 |

Geomean: +27.05% versus vLLM, +0.98% versus 9db3, -0.12% versus `c7a1b71`. Latency means versus `c7a1b71` are within
+-3% except Cosmos text-heavy (TTFT -9.3%, TPOT +7.5%), Gemma multi-image (TTFT -6.9%), and Cosmos short (TTFT +5.3%).
Gemma long-prefill stays in the high mode (613-619).

## Output audit

72/72 cells, 0 integrity issues, 0 first-EOS anomalies. Cross-repeat exact agreement: Cosmos 1513/1513, Gemma
589/632 (note 366: 593/632); the Gemma difference is the INT4 batch-shape numerics of note 368, which do not change
MMLU accuracy.

## Promotion

Pending explicit approval: point `.local/current/active/runtime` (and the gemma4/cosmos `runtime` links) at this
binary, record the switch in `.local/registry/current.json`, and mark this campaign `citable`.
