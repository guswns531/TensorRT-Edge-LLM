<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Thor recurrent-state copy batching experiments

## Decision

Keep the existing per-segment `cudaMemcpyAsync` implementation together with stable decode-cohort residency. Two ways of
batching Qwen3.8 recurrent and convolution state copies were implemented and measured on Jetson AGX Thor; both regressed
real HTTP serving and were removed from the production source.

The experiment artifacts remain under `data/qwen38/results/next`. The source tree after this experiment contains only this
note; neither rejected copy implementation is enabled.

## Motivation

The resident-cohort path already performs no recurrent-state copy for an unchanged decode cohort. A cohort remap still
issues one device copy per changed row, recurrent layer, and state kind. Qwen3.8-27B has 48 recurrent layers, so the legacy
path can issue:

```text
changed rows * 48 layers * (recurrent + convolution) = changed rows * 96 copy calls
```

The hypothesis was that reducing host API/launch count would help Thor. The byte count was intentionally unchanged.

## Candidate A: fused SM copy kernel

The first candidate uploaded stable/active segment descriptors and compact row mappings, then copied every layer and state
kind with one CUDA kernel per gather or scatter. Aligned state rows used 16-byte vector loads/stores. Initial zeroing was
also fused.

Correctness and build status:

- SM110 CUDA build: passed.
- `llm_build`, `llm_bench`, `llm_inference`, `unitTest`, and `llm_phase_context_smoke`: passed.
- `PhaseRecurrentStateActiveViewTest.*`: 3/3 passed on Thor.
- Every HTTP request completed with the exact configured output-token count.

For one warmup plus three measured runs, the fused server reported 119 prefill scatter launches, 119 prefill zero launches,
76 decode gather launches, and 65 decode scatter launches. The equivalent copied/zeroed row count would require hundreds
of thousands of per-layer operations in the legacy implementation.

Despite that launch reduction, the fused kernel moved traffic through the SM path and competed with inference for shared
LPDDR bandwidth. It failed the performance gate.

## Candidate B: CUDA 13 batched DMA

The second candidate removed the custom kernel and used CUDA 13.2 `cudaMemcpyBatchAsync`, preserving copy-engine semantics.
Initial zeroing stayed on the legacy path. It also passed the build, the same 3/3 GPU tests, and exact HTTP completion.

The CUDA batch setup cost for 96 independently allocated state segments was still larger than the legacy calls on this
workload. It also failed the performance gate and was removed.

## Locked-clock HTTP A/B

Contract:

- Jetson AGX Thor MAXN
- GPU GPC 1.575 GHz, EMC 4.266 GHz, min equal to max
- Qwen3.8-27B NVFP4, dense P4/D32, `thor-throughput`
- streaming HTTP, ignore EOS, deterministic fixed output length
- one short-burst warmup per fresh server lifecycle
- three measured runs per case
- 64/64 requests succeeded in every run

| implementation | short burst tok/s | wave burst tok/s | short vs legacy | wave vs legacy |
| --- | ---: | ---: | ---: | ---: |
| legacy per-segment D2D | 156.76 | 177.02 | baseline | baseline |
| fused SM copy | 154.41 | 146.79 | -1.50% | -17.08% |
| CUDA batched DMA | 153.15 | 145.98 | -2.30% | -17.54% |

Median latency moved in the same direction:

| implementation | short TTFT / TPOT / E2E | wave TTFT / TPOT / E2E |
| --- | --- | --- |
| legacy | 1,119.5 / 189.1 / 13,031.1 ms | 712.9 / 168.8 / 16,757.6 ms |
| fused SM copy | 1,107.2 / 192.4 / 13,224.5 ms | 776.1 / 205.6 / 20,261.3 ms |
| CUDA batched DMA | 1,118.3 / 194.0 / 13,335.7 ms | 799.6 / 206.9 / 20,388.3 ms |

Machine-readable summaries:

- `data/qwen38/results/next/legacy-state-short-wave/summary.json`
- `data/qwen38/results/next/fused-state-short-wave/summary.json`
- `data/qwen38/results/next/batched-state-short-wave/summary.json`

## Interpretation

Launch count was not the active bottleneck after stable decode-cohort residency reached roughly 95%. The remaining remaps
move about 155 MiB per Qwen3.8 stable slot across 96 separately allocated tensors. Replacing optimized D2D operations with
an SM kernel was especially harmful on Thor's shared memory fabric. CUDA batch submission preserved DMA behavior but added
batch descriptor processing without reducing bytes or allocation fragmentation.

Do not retry copy batching unless the state layout first becomes a small number of contiguous slot-major arenas. With the
current per-layer allocation, stable residency plus legacy D2D copies is faster.

## Related decisions and next steps

- Logical-vocabulary strided sampling was already neutral on Thor and remains reverted.
- Do not add a per-token GPU gather around the current D32 token mirror; the payload is only 128 bytes.
- Next low-risk experiment: pool request-owned vision payload allocations and release them after final prefill completion.
- Next high-impact architecture experiment: contiguous slot-major recurrent arenas. This must include a TensorRT binding
  strategy that avoids recreating 96 independent copy descriptors.
- Native packed FMHA remains a separate milestone because it changes the attention plugin contract and needs numerical as
  well as performance validation.
