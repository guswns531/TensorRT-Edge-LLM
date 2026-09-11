# Gemma 4 E2B AWQ vLLM CUDA Graph Optimization

## Scope

This campaign replaces the initial all-eager vLLM control in note 288 with a stronger vLLM 0.28 configuration for
the same Gemma 4 E2B AWQ model, RTX 3080, image/text HTTP traces, and 64 generated tokens per request. It answers a
narrow question: how much of the prior TensorRT advantage remains after enabling the vLLM optimizations that fit in
10 GiB?

The retained comparison still has one correctness limitation. The vLLM client reports the expected 512 generated
tokens per run but does not retain token IDs, so this campaign cannot establish exact cross-framework greedy identity.

## Frozen contract

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3080, 10,240 MiB |
| Driver | 610.43.02 |
| vLLM image | `vllm/vllm-openai:v0.28.0` |
| Image digest | `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14` |
| Model | `Chunity/gemma-4-E2B-it-AWQ-4bit` compatibility view |
| Model config SHA-256 | `a16ce998e2da17ea500d93ef27ba0b5c3d89ceec421afa124dc14f7c3784f171` |
| Quantization config SHA-256 | `aa83d3b328c31dd5dc8b85ada6cf1f5bd980bcf8d819efef515cc4c12d7e4015` |
| Precision | INT4 AWQ weights, FP16 activations and KV |
| Requests | 8 concurrent HTTP requests, 64 output tokens each |
| Repeats | 3 after 8 warmup requests |
| Max sequence length | 2,048 |
| Max sequences | 8 |
| KV allocation | 160 MiB |
| Prefix caching | Disabled |
| Image/audio capability | One image and zero audio per prompt |

The selected server-side execution settings are:

```text
--max-num-batched-tokens 1024
--enable-chunked-prefill
--async-scheduling
-cc.backend=eager
--cudagraph-capture-sizes 1 2 4 8
```

`-cc.backend=eager` is intentionally different from `--enforce-eager`. It prevents TorchInductor code generation,
which does not fit on this GPU, while retaining vLLM graph partitioning and both piecewise and full CUDA Graph replay.
The server captured four piecewise and four full graphs using about 0.35 GiB.

## Optimization search

| Candidate | Result | Decision |
|---|---|---|
| All eager | Fits, but leaves both compilation and CUDA Graph disabled | Replaced |
| Full TorchInductor | OOM during compilation/autotuning after model load | Reject on 10 GiB |
| `backend=eager` plus decoder graphs | Fits and gives the dominant improvement | Select |
| P1024 to P2048 | At most +0.71% mixed throughput, but worse text throughput and mixed TTFT | Keep P1024 |
| Async scheduling | Noise-level at an eight-request burst; useful general contract | Select |
| Disable unused audio | Saves about 0.5 GiB without a material performance loss | Select for image-only trace |
| `torch_shm` multimodal IPC | Mixed TTFT improves about 1.5%, but text throughput regresses about 0.6% | Keep direct RPC globally |
| Automatic encoder CUDA Graph | OOM while capturing 70--1024-token, up-to-14-image paths | Reject |
| Encoder graph restricted to 280 tokens/E1 | Still OOM on a 474 MiB allocation, including with KV reduced to 64 MiB | Reject |
| Disable chunked prefill | Gemma 4 warns unsupported; P1024 is also below max sequence length | Reject |
| Explicit AWQ-Marlin forcing | `auto_awq` already selects Marlin for supported CUDA layers | No new variant |

The encoder graph failures are capacity failures rather than unsupported-model failures. vLLM 0.28 recognizes Gemma
4 as encoder-graph capable, but even the single 280-token/E1 capture leaves only about 0.4 GiB free before requesting
another 474 MiB activation. Reducing the KV allocation by 96 MiB and enabling expandable allocator segments did not
make the capture fit. The practical 10 GiB frontier is therefore eager vision encoding plus graphed language-model
execution.

## vLLM results

All values are medians across three runs, except memory, which is the maximum observed peak.

| Variant | Trace | tok/s | req/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms | GPU MiB |
|---|---|---:|---:|---:|---:|---:|---:|
| Initial eager | Text | 208.328 | 3.255 | 116.665 / 121.933 | 37.136 / 37.444 | 2456.415 / 2457.293 | 7,795 |
| Graph P1024 | Text | 348.142 | 5.440 | 95.193 / 96.357 | 21.763 / 21.802 | 1466.657 / 1469.863 | 8,725 |
| Graph P1024 + async, audio off | Text | **348.879** | **5.451** | **95.382 / 96.466** | **21.709 / 21.750** | **1463.658 / 1467.021** | **8,141** |
| Graph P2048 | Text | 347.312 | 5.427 | 100.125 / 101.935 | 21.746 / 21.778 | 1470.283 / 1473.685 | 8,807 |
| Graph P1024 + async + SHM | Text | 346.929 | 5.421 | 99.485 / 101.229 | 21.783 / 21.818 | 1471.974 / 1475.259 | 8,191 |
| Initial eager | Mixed | 182.274 | 2.916 | 339.601 / 451.484 | 37.684 / 38.734 | 2657.057 / 2730.633 | 7,859 |
| Graph P1024 | Mixed | **326.107** | **5.095** | **157.324 / 181.110** | **22.058 / 22.433** | **1549.393 / 1562.102** | 8,773 |
| Graph P1024 + async, audio off | Mixed | 325.716 | 5.089 | 159.610 / 184.120 | 22.092 / 22.469 | 1551.395 / 1564.525 | 8,231 |
| Graph P2048 | Mixed | 328.044 | 5.126 | 170.967 / 181.935 | 22.005 / 22.537 | 1557.029 / 1560.136 | 8,807 |
| Graph P1024 + async + SHM | Mixed | 325.866 | 5.092 | 157.303 / 181.436 | 22.120 / 22.490 | 1550.789 / 1563.514 | **8,191** |

P2048 is the isolated mixed-throughput maximum, but its +0.71% throughput relative to the selected P1024 async
configuration costs +7.1% mean TTFT. It also regresses text throughput by 0.45%. P1024 is the better single static
setting. Async scheduling and direct RPC are retained because their differences at this burst size are noise-level,
they avoid the text regression seen with SHM, and async scheduling is the appropriate higher-load contract.

Relative to the initial eager control, the selected vLLM configuration improves:

| Trace | Token throughput | Mean TTFT | Mean TPOT | Mean E2E | Memory |
|---|---:|---:|---:|---:|---:|
| Text | +67.47% | -18.24% | -41.54% | -40.42% | +346 MiB |
| Mixed | +78.69% | -53.00% | -41.37% | -41.61% | +372 MiB |

The earlier all-eager vLLM result was therefore not a sufficiently optimized performance baseline.

## TensorRT V3 versus optimized vLLM

The TensorRT values are the warmup-matched HTTP V3 values from note 288. The vLLM comparator is the selected P1024,
async, image-only configuration. Positive throughput means TensorRT is faster; negative TPOT/E2E means TensorRT has
lower latency.

| Trace | Token throughput | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | GPU memory |
|---|---:|---:|---:|---:|---:|
| Text | **+44.25%** | +212.38% / +483.04% | **-65.54% / -64.58%** | **-47.81% / -30.72%** | +902 MiB |
| Mixed | **+34.04%** | +127.41% / +279.18% | **-63.30% / -60.84%** | **-43.69% / -25.30%** | +816 MiB |

The conclusion is now more precise than the earlier eager comparison:

- TensorRT V3 still has materially higher sustained generation throughput, TPOT, and completed-request E2E latency.
- vLLM reaches the first token much sooner. TensorRT's current four-owner admission and independent phase pipeline
  delay first-token progress even though the subsequent decode path is much faster.
- TensorRT spends about 0.8--0.9 GiB more GPU memory for the independent E/P/D context frontier.
- The next fair optimization target is TensorRT TTFT/admission, not claiming a larger decode advantage against an
  intentionally eager vLLM configuration.

## Retained artifacts

| Artifact | Path |
|---|---|
| Initial eager results | `.local/results/gemma4-e2b-awq-vllm028-equal-http-20260911` |
| Optimized search and results | `.local/results/gemma4-e2b-awq-vllm028-graph-eager-20260911` |
| Reproduction manifest | `.local/results/gemma4-e2b-awq-vllm028-graph-eager-20260911/manifest.json` |

## Next comparison gates

1. Use P1024, decoder CUDA Graph, async scheduling, and image-only capability as the new frozen vLLM Gemma control.
2. Re-run vLLM only when the trace/load/output-length contract changes; reuse these results for unchanged traces.
3. Extend the comparison to sustained arrival traces and larger decode cohorts. The current traces are simultaneous
   eight-request bursts, not continuous-arrival saturation tests.
4. Measure TensorRT admission-to-first-P and first-P-to-first-D delays against vLLM to explain the remaining TTFT gap.
5. Revisit encoder CUDA Graph only after reducing model/context residency by at least 0.5 GiB; shrinking KV alone is
   insufficient.
6. Preserve vLLM prefix caching as disabled until an explicit shared-prefix workload is evaluated.

