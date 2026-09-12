# Gemma 4 vLLM capacity frontier and packed V3 full-12 comparison

Date: 2026-09-12

## Outcome

The previous Gemma 4 vLLM control was well optimized for an eight-request burst, but it was not a fair capacity
baseline for the later 24-in-flight traces. Raising only the external client concurrency had left the vLLM engine at
`max_num_seqs=8`, 160 MiB of KV cache, a 1,024-token scheduling budget, and CUDA Graph buckets through batch eight.
That contract understated vLLM throughput by 64.6--160.7% on the seven workloads previously compared with packed
V3.

A new capacity sweep selects this vLLM 0.28 contract for the current 24-in-flight campaign:

```text
max_num_seqs              = 24
KV cache                  = 480 MiB, FP16
max_num_batched_tokens    = 4096
chunked prefill           = enabled
async scheduling          = enabled
decoder CUDA Graph sizes  = 1, 2, 4, 8, 16, 24
prefix caching            = disabled
maximum images/request    = 8
maximum model length      = 2048
```

The stronger control materially changes the conclusion. Packed TensorRT V3 retains higher twelve-workload
geometric-mean token throughput, lower TPOT, and lower E2E latency, but it no longer wins every workload. vLLM is
substantially better on long-prefill and the vision-rich mixed, vision-heavy, and multi-image traces. The earlier
claim that packed V3 was 97.4% faster in seven-workload geometric-mean throughput used the capacity-limited vLLM
control and is superseded by this note.

All results in this note are diagnostic one-run measurements. They establish the configuration frontier and expose
the remaining architectural gap; they are not citable confidence-interval results.

## Fixed comparison contract

| Item | Packed TensorRT V3 | vLLM capacity control |
|---|---|---|
| GPU | NVIDIA GeForce RTX 3080, 10,240 MiB | same |
| Model | Gemma 4 E2B INT4-AWQ backbone | same source checkpoint compatibility view |
| Activations / KV | FP16 / FP16 | FP16 / FP16 |
| Maximum sequence | 2,048 | 2,048 |
| External concurrency | 24 workers, 24 in flight | same |
| Stable/active owners | 24 indexed-paged owners | `max_num_seqs=24` |
| KV capacity | 96 pages x 128 = 12,288 tokens | 480 MiB = 27,025 tokens |
| Prefill | fixed 128-token chunks, packed P8 | chunked, 4,096-token iteration budget |
| Decode | D24, independent TensorRT context | continuous batching through 24 |
| Vision | independent E4 context | eager encoder in the unified vLLM engine |
| Decode graphs | TensorRT/CUDA Graph runtime path | sizes 1/2/4/8/16/24 |
| Policy | V3 Service-scaled Transition, generic 49-request calibration | vLLM V1 scheduler |
| Prefix reuse | disabled for this comparison | disabled |

The request files, arrival schedule, fixed output-token counts, HTTP client implementation, and client concurrency
are shared. The serving frameworks use their own chat/template tokenization, so their reported prompt-token totals
differ slightly. Generated-token throughput and request latency are therefore end-to-end serving comparisons, not
an equal-prefill-token kernel comparison. vLLM does not retain token IDs in this harness, so exact cross-framework
greedy identity is not established.

The vLLM image is `vllm/vllm-openai:v0.28.0` at digest
`sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`. Full TorchInductor and encoder CUDA
Graph paths were already rejected on this 10 GiB device because they OOM during compilation or capture. The chosen
`backend=eager` is not `--enforce-eager`: vLLM graph partitioning and piecewise/full decoder CUDA Graph replay remain
enabled.

## Capacity sweep

The first sweep used balanced, decode-heavy, vision-heavy, and multi-image. Values are generated tokens/s.

| vLLM configuration | Balanced | Decode-heavy | Vision-heavy | Multi-image |
|---|---:|---:|---:|---:|
| seq16, KV320 MiB, P1024, graphs <=16 | 593.64 | 617.28 | 335.32 | 313.97 |
| seq24, KV480 MiB, P1024, graphs <=24 | 769.84 | 812.84 | 435.73 | **381.97** |
| seq24, KV480 MiB, P2048, graphs <=24 | **773.11** | **813.89** | 438.91 | 381.65 |
| **seq24, KV480 MiB, P4096, graphs <=24** | 771.53 | 812.86 | **544.70** | 381.05 |
| seq24, KV480 MiB, P8192, graphs <=24 | 771.33 | 813.55 | 426.10 | 382.11 |

Moving from 16 to 24 sequences is the dominant text/decode improvement. P2048 changes text throughput by less than
0.5%. P4096 is selected because it raises vision-heavy throughput by 24.1% over P2048 while leaving balanced,
decode-heavy, and multi-image within 0.4%. P8192 reverses that VLM gain. This is an engine-level scheduling-capacity
frontier, not a workload label supplied to the scheduler.

The selected server reports 27,025 GPU KV tokens, or 13.2 full-length 2,048-token sequences. That is more KV token
capacity than packed V3's 12,288-token pool. The larger vLLM number does not provide 24 simultaneous full-length
requests; `max_num_seqs=24` is a scheduler ceiling and actual admission remains token-capacity constrained.

### Dense CUDA Graph rejection

Capturing every decode size from one through 24 was also tested at seq24/KV480/P4096.

| Workload | Sparse graphs | Dense 1--24 graphs | Dense delta |
|---|---:|---:|---:|
| balanced | 771.53 | 773.04 | +0.20% |
| decode-heavy | 812.86 | 813.82 | +0.12% |
| vision-heavy | 544.70 | 297.17 | -45.45% |
| multi-image | 381.05 | 380.54 | -0.13% |

Dense capture grows graph memory from about 0.31 GiB to 0.65 GiB and raises the observed peak from 8,783 to 9,183
MiB. Its negligible text gain does not justify the memory cost or vision-heavy regression, so the sparse graph set
remains the selected control.

## Why the old vLLM comparison was weak

| Workload | Frozen seq8 vLLM | New seq24/P4096 vLLM | Improvement |
|---|---:|---:|---:|
| short | 288.38 | 567.55 | +96.81% |
| balanced | 329.72 | 771.46 | +133.97% |
| decode-heavy | 330.17 | 812.43 | +146.06% |
| long-prefill | 261.31 | 500.26 | +91.44% |
| mixed | 270.00 | 703.81 | +160.67% |
| vision-heavy | 285.40 | 559.83 | +96.16% |
| multi-image | 231.73 | 381.34 | +64.56% |

The old result was not fraudulent: it was the retained optimum for its original eight-request contract. It became
an invalid peak-performance comparator when later traces admitted 24 requests. The mismatch was the internal vLLM
sequence/KV/graph capacity, not AWQ, sampling, or an all-eager execution path.

## Full twelve-workload result

Positive throughput delta means packed V3 is faster. A negative latency delta means packed V3 is lower latency.

| Workload | Packed V3 tok/s | vLLM tok/s | Throughput delta | TTFT mean delta | TPOT mean delta | E2E mean delta |
|---|---:|---:|---:|---:|---:|---:|
| short | 789.92 | 567.55 | +39.18% | -37.83% | -13.66% | -20.74% |
| balanced | 1,181.99 | 771.46 | +53.22% | -34.87% | -30.05% | -30.83% |
| decode-heavy | 1,293.43 | 812.43 | +59.21% | -42.12% | -33.64% | -34.08% |
| long-prefill | 423.84 | 500.26 | -15.28% | +332.20% | -38.11% | +19.12% |
| bimodal | 588.56 | 600.16 | -1.93% | +576.24% | -30.90% | +7.89% |
| text-heavy | 832.93 | 404.66 | +105.83% | -86.99% | -3.75% | -51.85% |
| mixed | 478.90 | 703.81 | -31.96% | +163.68% | -25.35% | +12.83% |
| vision-heavy | 359.54 | 559.83 | -35.78% | +443.24% | -57.33% | +45.37% |
| poisson | 883.36 | 681.95 | +29.54% | +30.64% | -18.24% | -16.06% |
| wave-drain | 93.94 | 92.62 | +1.43% | +50.00% | -64.31% | -41.03% |
| multi-image | 194.22 | 381.34 | -49.07% | +590.15% | -61.07% | +49.77% |
| late-vision | 1,446.55 | 990.82 | +46.00% | -5.33% | -36.86% | -35.57% |

Packed V3 wins token throughput on seven workloads and vLLM wins five. Packed V3 has lower mean TPOT on all
twelve, reflecting the stronger compiled decode path. vLLM has lower mean TTFT on seven, especially the
long-prefill and vision-rich traces, reflecting faster first-token admission and encoder/prefill progress. Packed V3
has lower mean and p95 E2E on seven workloads; vLLM wins both on the same five throughput-loss workloads.

### Absolute latency

| Workload | Packed TTFT mean/p95 | vLLM TTFT mean/p95 | Packed TPOT mean/p95 | vLLM TPOT mean/p95 | Packed E2E mean/p95 | vLLM E2E mean/p95 |
|---|---:|---:|---:|---:|---:|---:|
| short | 96.63/227.64 | 155.42/242.31 | 22.79/27.87 | 26.39/30.11 | 551.14/904.30 | 695.40/1,068.16 |
| balanced | 87.44/217.88 | 134.27/234.33 | 16.62/17.91 | 23.76/24.56 | 1,472.00/2,262.27 | 2,128.03/3,244.79 |
| decode-heavy | 88.92/223.30 | 153.61/249.66 | 15.29/15.81 | 23.04/23.41 | 3,959.63/6,059.05 | 6,006.17/9,096.01 |
| long-prefill | 2,350.82/3,423.61 | 543.92/1,442.01 | 22.51/25.63 | 36.37/44.60 | 4,234.28/5,996.35 | 3,554.56/5,866.97 |
| bimodal | 2,146.44/5,579.99 | 317.41/878.39 | 19.92/33.40 | 28.83/37.42 | 4,736.10/10,219.31 | 4,389.95/9,582.02 |
| text-heavy | 215.79/622.73 | 1,658.73/4,312.05 | 21.63/26.16 | 22.47/27.54 | 1,355.34/1,743.34 | 2,814.97/5,848.75 |
| mixed | 728.78/2,494.02 | 276.41/410.58 | 19.77/24.89 | 26.48/32.39 | 1,662.90/3,007.76 | 1,473.77/2,162.56 |
| vision-heavy | 1,524.13/3,150.77 | 280.56/390.78 | 13.05/22.56 | 30.59/40.54 | 2,065.21/3,474.32 | 1,420.68/2,138.33 |
| poisson | 156.47/461.13 | 119.77/169.82 | 21.27/24.85 | 26.02/29.23 | 1,652.63/3,023.13 | 1,968.88/3,502.62 |
| wave-drain | 267.34/575.05 | 178.23/204.42 | 8.02/9.77 | 22.48/24.58 | 516.01/805.60 | 874.97/884.36 |
| multi-image | 1,303.41/2,309.28 | 188.86/227.16 | 11.56/15.25 | 29.70/35.54 | 1,661.86/2,556.75 | 1,109.61/1,300.30 |
| late-vision | 130.37/394.73 | 137.71/243.29 | 14.24/14.29 | 22.55/22.55 | 2,169.73/2,802.58 | 3,367.53/4,416.79 |

The twelve-workload geometric means are:

| Metric | Packed V3 | vLLM capacity control | Packed delta |
|---|---:|---:|---:|
| token throughput | 562.68 tok/s | 521.69 tok/s | +7.86% |
| TTFT mean | 369.42 ms | 241.59 ms | +52.91% |
| TTFT p95 | 890.98 ms | 411.88 ms | +116.32% |
| TPOT mean | 16.50 ms | 26.27 ms | -37.19% |
| TPOT p95 | 20.42 ms | 30.31 ms | -32.64% |
| E2E mean | 1,766.29 ms | 2,042.19 ms | -13.51% |
| E2E p95 | 2,809.14 ms | 3,120.58 ms | -9.98% |

Packed V3 peaks at 9,383 MiB and the selected vLLM run peaks at 8,843 MiB, so packed V3 spends up to 540 MiB more
GPU memory. Its 7.86% aggregate throughput, 37.19% mean-TPOT, and 13.51% mean-E2E advantages are therefore not
equal-memory claims. Conversely, vLLM's larger KV token pool means its lower allocation is not caused by starving
KV; it primarily reflects the cost of packed V3's independent TensorRT E/P/D context frontier.

## Interpretation and next gates

1. Keep seq24/KV480/P4096/sparse graphs as the new frozen vLLM Gemma control. Do not use the seq8 numbers for later
   24-in-flight performance claims.
2. The strongest packed V3 asset is decode: TPOT wins all twelve workloads. The largest deficit is E/P critical-path
   service, not decode batching or KV capacity.
3. Prioritize vision admission/E formation and long-prefill first-token progress. Multi-image and vision-heavy lose
   35.8--49.1% throughput even though their post-first-token TPOT is much lower.
4. Preserve P4096 as a vLLM comparison setting only, not as a new TensorRT policy constant. It is a vLLM iteration
   token budget and has no direct equivalence to TensorRT's 128-token packed chunks.
5. Repeat the selected full-12 at least three times before a citable comparison. Preserve request-class TTFT/TPOT,
   memory, and prompt-token totals in addition to aggregate throughput.
6. Add exact output capture or a framework-neutral token validation pass before claiming semantic parity.

## Retained artifacts

| Artifact | Path |
|---|---|
| Reproduction script | `.local/results/gemma4-vllm-capacity-sweep-20260912/run-config.sh` |
| Capacity sweep | `.local/results/gemma4-vllm-capacity-sweep-20260912/seq*` |
| Selected vLLM full-12 | `.local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12` |
| Dense-graph diagnostic | `.local/results/gemma4-vllm-capacity-sweep-20260912/seq24-kv480-p4096-gdense24` |
| Packed V3 full-12 | `.local/results/gemma4-packed-prefill-g4-20260912/sentinel-v3-1x/service-scaled-transition` |
