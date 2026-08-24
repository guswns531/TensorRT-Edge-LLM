<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Thor Qwen3.8 decode kernel memory-order analysis

## Scope

This analysis profiles one graph-off Qwen3.8-27B NVFP4 decode at batch 32 and past-KV length 128 on the locked-clock
Jetson AGX Thor. The engine is `phase-b32-p4-d32-kv4096`. Nsight Systems 2026.2 provides the kernel time distribution and
Nsight Compute 2026.1 provides hardware counters for the dominant GDN and NVFP4 kernels.

The unprofiled component latency was 172.18 ms. Nsight replay timings are not performance results; only their counters and
launch geometry are used below.

## Kernel time distribution

| kernel family | time share | instances | mean per instance |
| --- | ---: | ---: | ---: |
| GDN large-batch state update | 33.6% | 96 | 1.188 ms |
| NVFP4 block-scaled GEMM | 28.6% | 480 | 0.203 ms |
| FP16 tensor-core GEMM | 21.0% | 96 | 0.743 ms |
| LM-head GEMM | 7.0% | 2 | 11.810 ms |
| NVFP4 stream-K GEMM | 5.4% | 128 | 0.143 ms |
| attention | 0.9% | 32 | 0.099 ms |
| causal conv decode | 0.9% | 96 | 0.033 ms |

The first five families consume 95.6% of GPU kernel time. Attention and KV access are not the current decode bottleneck.

## GDN grid and address order

Measured launch:

```text
grid  = (1536, 1, 1) = N32 * HV48
block = (256, 1, 1)  = 8 warps
state = h0[N, HV, K128, V128], FP32
```

One CTA owns one `(request, recurrent-head)` state matrix. Its base global address is conceptually:

```text
base = ((request * HV + hv) * K * V) * sizeof(float)
addr(k, v) = base + (k * V + v) * sizeof(float)
```

The CTA processes V in four `TILE_V=32` tiles. The cp.async thread layout is `(32,8)` with four FP32 values per copy.
Each eight-thread subgroup reads 8 x 16 bytes, one contiguous 128-byte segment of a K row. A warp covers four K rows.
The state write maps `v_write = tidx % 32`, so each warp also writes one contiguous 128-byte V segment. Q/K loads use the
first 128 threads over contiguous K. These are coalesced patterns.

Shared state uses `(K128, V32, stages2)` with V stride 1 and padded K-row stride 36. The four-value padding is intended to
avoid repeated shared-memory bank alignment. The compute mapping keeps four neighboring V values in each warp subgroup and
reduces across K with shuffle operations.

## GDN measured bottleneck

| metric | value |
| --- | ---: |
| duration | 1.15 ms |
| compute throughput | 18.26% |
| memory throughput | 18.31% |
| L1 hit rate | 1.98% |
| L2 hit rate | 59.98% |
| registers/thread | 128 |
| dynamic shared memory/block | 38.08 KiB |
| theoretical / achieved occupancy | 33.33% / 33.10% |
| active warps/scheduler | 3.99 |
| eligible warps/scheduler | 0.20 |
| scheduler cycles with no eligible warp | 85.71% |
| average cycles between issued instructions | 27.88 |
| L1TEX scoreboard stall | 16.1 cycles, 57.7% of issue interval |

The kernel is not peak-bandwidth-bound and is not dominated by divergent lanes (`29.25` non-predicated threads per warp).
It is latency-bound with too few eligible warps. Register use limits each SM to two CTAs, and outstanding state loads leave
most schedulers without a ready warp.

### Double-buffer ordering issue

The kernel allocates two stages and initially commits two cp.async groups, but every V-tile starts with:

```python
cute.arch.cp_async_wait_group(0)
cute.arch.barrier()
```

`wait_group(0)` waits until no group remains outstanding. On tile 0 it therefore waits for both tile 0 and the already
prefetched tile 1. After prefetching tile 2, the next iteration again waits for all pending groups. The two-stage buffer
does not overlap the next global load with current-tile compute as intended.

The first isolated A/B should use `cp_async_wait_group(NUM_STAGES - 1)`, which is `wait_group(1)`: wait for the oldest
current tile while allowing one future tile group to remain in flight. This changes scheduling only, not tensor layout or
numerics.

### Other GDN layout candidates

1. Shorten register live ranges before forcing a register cap. A raw `maxrregcount` is unsafe because spilling would add
   more state traffic. Moving from 128 to about 85 registers/thread is required to permit three CTAs per SM.
2. Split V tiles into an extra grid dimension. This reloads small Q/K/scalar inputs but removes the four-tile serial loop
   and can reduce shared memory/live ranges. It should be tested only after fixing wait-group ordering.
3. Export the existing small-batch kernel as a second AOT symbol and test it at N=32. It uses eight CTAs per state and
   128-thread blocks, trading redundant Q/K loads for more independent scheduling. The current AOT artifact exports only
   `run_large_batch`; the Python test threshold does not affect the runtime wrapper.
4. Do not transpose the global state first. Current global loads and stores are already coalesced; a transpose would be a
   large contract change without evidence that transaction count is the problem.

## NVFP4 GEMM grid and access behavior

The profiled stream-K variant launches:

```text
grid  = (40, 1, 1)
block = (256, 1, 1)
cluster size = 2
CTA tile includes M128
```

| metric | value |
| --- | ---: |
| duration | 113.7 us |
| compute throughput | 8.03% |
| memory throughput | 19.56% |
| L1 hit rate | 0% |
| L2 hit rate | 11.57% |
| registers/thread | 216 |
| dynamic shared memory/block | 227.33 KiB |
| waves/SM | 2 |

This is also latency/shape-limited rather than saturated. Decode M is 32 while the CTA tile names M128, so at most one
quarter of the M rows are useful. The grid has only two waves per 20-SM GPU and each CTA consumes very high register/shared
memory resources. Global accesses inside CUTLASS are vectorized, but the tile shape causes underfilled work and weak reuse;
coalescing alone cannot recover it.

The next GEMM experiment should force or add an M32/M64 decode tactic, or a GEMV-like NVFP4 path, while retaining the
current M128 tactic for prefill. A D64 engine may also improve this exact kernel by filling half rather than one quarter of
the M tile, which is a stronger rationale for P4/D64 than scheduler-level batching alone.

## FP16 and LM-head implications

The FP16 kernel name exposes an M64 CTA tile and accounts for 21.0% of decode. At M32 it is also half-filled. It needs the
same Launch/Occupancy counter pass before modification. The LM-head uses two long GEMM launches and accounts for 7.0%; it
is important but secondary to 48 GDN layers plus their projection GEMMs.

## Prioritized experiments

1. **GDN wait-group fix:** change large and varlen large kernels from wait-group 0 to 1, rebuild the SM110 CuTe DSL
   artifact, run numerical tests, then compare GDN latency and D32 E2E.
2. **GDN N32 grid crossover:** export both small and large symbols and benchmark N=16/32/48/64 on Thor. Do not reuse the
   generic threshold of 32 without Thor data.
3. **NVFP4 small-M tactic:** compare M32/M64 tiles against M128 for the Qwen projection shapes. Record useful-M fraction,
   waves/SM, registers, and shared memory.
4. **D64 engine:** D64 fills more of M64/M128 GEMM tiles and may amortize per-CTA state/GEMM latency. Measure aggregate
   `batch / latency`, not latency alone.
5. Only after these, consider a state-layout transpose or fusion across layers; current evidence does not justify those
   contract changes.

Expected ceiling: a 20% GDN improvement saves about 6.7% D32 E2E; a 20% projection-GEMM improvement saves another 6-7%.
Those two targets are materially larger than the host/sampling micro-optimizations already rejected.

## Reproduction

The Systems report is `/tmp/qwen38_d32_past128.nsys-rep` on the Thor node. The base command uses graph-off
`llm_bench --mode decode --batchSize 32 --pastKVLen 128 --warmup 1 --iterations 1 --noCudaGraph`. Nsight Compute requires
the profiling container to have `SYS_ADMIN` capability for GPU performance counters; no host setting was changed.
