# Cosmos phase execution support (2026-08-06)

Cosmos Reason2-2B's exported text decoder is a Qwen3-VL-style decoder. Its
engine configuration has `indexed_kv_cache=false`, `num_deepstack_features=3`,
and an M-RoPE configuration. The phase benchmark now accepts this combination
for a fixed microbenchmark.

## What changed

- `HybridCacheManager` reuses the existing indexed gather/increment CUDA
  kernels with the legacy fixed-row length tensor when indexed KV is disabled.
- `PhaseBatchState` has an indexed/non-indexed mode. In non-indexed mode it
  binds only `kvcache_start_index` (the engine has no `kv_slot_ids` input) and
  treats the caller-provided row IDs as fixed physical rows.
- `llm_phase_bench` no longer rejects deepstack engines. Text-only Cosmos
  requests clear per-phase deepstack tensors and bind them through
  `DeepstackBinding`, so a long prefill never reads past the one-token shared
  dummy buffer.
- The non-indexed path explicitly rejects real-request admission, continuous
  load, context packing, and eviction. Those operations need stable indexed
  ownership; a legacy linear cache would have to copy KV rows.

## RTX 3080 smoke results

All runs used TensorRT 11.0.0.114 / CUDA 13.3 in
`nvcr.io/nvidia/tensorrt:26.06-py3`, warmup 0 and one measured iteration. The
table is retained as historical smoke data only: sequential was always first,
so its first sample included TensorRT lazy tactic/auxiliary-stream
initialization while concurrent mode ran second.

| engine | prefill/decode | context mode | sequential ms | scheduled ms | speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| `engine-fp16-b2-i1024-kv2048` | 1 / 1 | independent | 116.2353 | 17.3908 | 6.6837x |
| `engine-fp16-b2-i1024-kv2048` | 1 / 1 | shared | 287.7672 | 21.6262 | 13.3064x |
| `engine-fp16-b4-i1024-kv2048` | 2 / 2 | independent | 118.6825 | 20.0817 | 5.9100x |
| `engine-fp16-b8-i1024-kv2048` | 1 / 4 | independent | 114.7828 | 17.4594 | 6.5743x |

The logs verified one CUDA context and distinct TensorRT execution-context
addresses in independent mode. Three deepstack buffers were allocated for
each phase, and no KV compaction or slot allocator was used in the non-indexed
Cosmos runs.

The benchmark now primes both modes before collecting samples. With the same
Cosmos `b2` engine, `prefill=1`, `decode=1`, `input=128`, `pastKV=128`,
`warmup=0`, and `iterations=1`, the corrected result is:

| mode | makespan median | prefill median | decode median |
| --- | ---: | ---: | ---: |
| sequential | 20.5199 ms | 14.4701 ms | 6.0467 ms |
| independent concurrent | 17.2791 ms | 17.2564 ms | 8.4705 ms |

The corrected makespan speedup is **1.1876x**, consistent with phase overlap
rather than a kernel-level 6--13x improvement.
For the same workload with a shared TensorRT context, sequential and scheduled
makespans were 21.0473 ms and 21.0342 ms respectively (1.0006x).

This is text-only deepstack validation. The phase runtime now has a generic
`PhaseVisionAdapter`, the existing `Gemma4PhaseVisionAdapter`, and a
`Qwen3VLPhaseVisionAdapter` that carries M-RoPE tensors and raw deepstack
features into packed prefill. The checked-in Cosmos engines are still
non-indexed, so the real-image async trace remains intentionally rejected until
an indexed Cosmos decoder is exported; this avoids silently turning a fixed
physical row into a movable admission slot.
