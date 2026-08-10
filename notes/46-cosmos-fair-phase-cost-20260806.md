# Cosmos-Reason2-2B fair phase retest and kernel-group cost table

## Scope

This is a Cosmos-only retest.  The checkpoint is `nvidia/Cosmos-Reason2-2B`,
FP16, text-only, with the existing non-indexed fixed-linear KV path.  The
engine was rebuilt in the TensorRT 26.06 container (TensorRT 11.0.0 / CUDA
13.3) with:

```text
engine: .local/cosmos-reason2-2b/engine-fp16-b16-i1024-kv2048
maxBatchSize: 16
maxPrefillBatchSize: 16
maxDecodeBatchSize: 16
maxInputLen: 1024
maxKVCacheCapacity: 2048
KV dtype: FP16
engine size: 3.3 GiB
```

The previous Cosmos measurements used different cold-start and workload
conditions.  This run primes both execution modes, then uses the same
`inputLen=512`, `prefillChunkSize=128`, `pastKVLen=512`, 20 warmup iterations,
and 100 measured iterations for every point.  Each sample therefore contains
four fixed-128 prefill chunks and four decode steps.  CUDA graph is disabled.

The reusable harness is
[`run_fair_phase_cost_suite.py`](../scripts/cosmos_reason2/run_fair_phase_cost_suite.py).
It writes the raw phase samples and the aggregated tables under
`.local/cosmos-reason2-2b/cost-table-fair-20260806/`:

- [`phase-summary.csv`](../.local/cosmos-reason2-2b/cost-table-fair-20260806/phase-summary.csv)
- [`kernel-cost-table.csv`](../.local/cosmos-reason2-2b/cost-table-fair-20260806/kernel-cost-table.csv)

## Fair independent overlap results

Only pairs with `prefill batch + decode batch <= 16` are included in the
independent-context overlap comparison.  This is required by the 16 physical
fixed-linear KV slots; it prevents the two streams from writing the same slot.
The speedup is sequential median divided by independent concurrent median.

| Pair | Sequential median / p95 (ms) | Independent median / p95 (ms) | Median speedup | p95 speedup | Measured overlap |
|---|---:|---:|---:|---:|---:|
| p1 / d1 | 83.046 / 83.187 | 71.107 / 71.377 | 1.168x | 1.165x | 35.0% |
| p2 / d2 | 93.334 / 93.641 | 80.501 / 81.144 | 1.159x | 1.154x | 36.7% |
| p4 / d4 | 128.786 / 129.412 | 115.337 / 115.932 | 1.117x | 1.116x | 36.4% |
| p8 / d8 | 206.684 / 207.342 | 194.331 / 194.973 | 1.064x | 1.063x | 42.6% |

The overlap benefit decreases as the prefill batch grows: prefill occupies
more SM and memory bandwidth, leaving less useful concurrency for decode.
This is a real effect in the corrected, warm measurement—not the earlier
multi-x cold-start artifact.

## Doubled phase-capacity results

The BS16 capacity points use the shared TensorRT context and serialized slot
reuse.  They are valid phase-cost measurements, but are deliberately not
reported as independent overlap because a 16+16 pair needs 32 physical slots.

| Prefill batch | Prefill median / p95 (ms) | Decode batch | Decode median / p95 (ms) |
|---:|---:|---:|---:|
| 1 | 58.353 / 58.480 | 1 | 25.193 / 25.255 |
| 2 | 67.838 / 68.100 | 2 | 25.304 / 25.342 |
| 4 | 102.575 / 102.953 | 4 | 25.775 / 25.908 |
| 8 | 178.826 / 179.443 | 8 | 27.193 / 27.292 |
| 16 | 348.809 / 349.300 | 16 | 25.606 / 25.655 |

Prefill scales with batch and is the dominant cost.  Decode is nearly flat in
this fixed-input test because each decode kernel is memory/attention bound and
the measured decode phase has four one-token rounds.

## Kernel-group cost table

The fixed path now uses the same CUDA-event recorder as the serving path.  The
groups are `prefill_prepare`, `prefill_engine`, `prefill_cache_commit`,
`decode_prepare`, `decode_engine`, and `decode_cache_commit`; the new decode
commit group is declared in
[`phaseKernelGroupRecorder.h`](../cpp/runtime/scheduling/phaseKernelGroupRecorder.h).
The benchmark instrumentation is opt-in: without `--kernelGroupCsv`, the old
fast path is unchanged.

Representative median GPU costs (ms per recorded chunk/round) are:

| Scenario / execution | Prefill prepare | Prefill engine | Prefill commit | Decode prepare | Decode engine | Decode commit |
|---|---:|---:|---:|---:|---:|---:|
| p1/d1 sequential | 0.018 | 14.927 | 0.004 | 0.009 | 6.193 | 0.005 |
| p1/d1 scheduled | 0.019 | 17.687 | 0.004 | 0.015 | 9.228 | 0.009 |
| p8/d8 sequential | 0.085 | 48.054 | 0.004 | 0.010 | 6.826 | 0.006 |
| prefill BS16 serialized | 0.156 | 93.325 | 0.004 | 0.014 | 6.367 | 0.006 |
| decode BS16 serialized | 0.018 | 14.879 | 0.004 | 0.015 | 6.310 | 0.006 |

For `inputLen=512` and `chunk=128`, each prefill-engine row occurs four times
per sample; the decode-engine row also occurs four times.  The CSV retains all
100 iterations, both execution modes, group identity, batch sizes, token
lengths, median, p95, and max.  The cost table is therefore directly usable
as the lookup table for a later queue scheduler.

## Asymmetric profile follow-up

The builder was corrected so all batch-dependent inputs use the phase-specific
limit: M-RoPE and KV-cache bindings use prefill/decode limits, and Cosmos
deepstack bindings now follow the same rule.  The following engine builds
successfully:

```text
--maxBatchSize 16 --maxPrefillBatchSize 8 --maxDecodeBatchSize 16
```

Its runtime config reports `maxBatch=16, maxPrefillBatch=8,
maxDecodeBatch=16`.  p8/d8 independent execution succeeds, and p8/d16 runs
successfully with the shared serialized context.  A second engine with
`maxBatchSize=24` (8 prefill slots + 16 decode slots) also builds, but p8/d16
independent execution exhausts the RTX 3080 10 GiB device during allocation of
the two TensorRT contexts plus the 24-slot FP16 KV cache.  Therefore the
asymmetric profile is now supported by the builder, while true p8+d16
independent overlap requires either a larger GPU, smaller KV capacity, or
quantized/optimized memory ownership.

As an intermediate independent check, p4/d12 on the 16-slot asymmetric engine
uses disjoint slots and completes successfully: 27.772 ms sequential versus
23.767 ms independent concurrent (1.169x, five measured iterations).  This
confirms that asymmetry itself does not disable independent execution; the
limiting condition is the physical slot sum and available device memory.
