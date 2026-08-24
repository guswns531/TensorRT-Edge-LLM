# Thor GDN grid experiments and Qwen3.8 D64 engine

## Decisions

- GDN `cp_async_wait_group(1)`: rejected after 50-run D32 regression.
- GDN small grid at N32: rejected; large remains faster.
- GDN small grid at N48/N64: promising for a future second AOT symbol.
- NVFP4 M32/M64 custom tactic: TensorRT internal kernel, no editable source in this repository.
- P4/D64 engine: accepted at component level; aggregate decode throughput improves 41.4%.
- Recurrent-state transpose: rejected for now because current global state transactions are already coalesced.

## GDN wait-group A/B

The large and varlen-large kernels were rebuilt with `wait_group(NUM_STAGES - 1)`. Numerical GDN tests passed 7 with 1
architecture skip. A short 10-run test suggested a small win, but the required 50-run test reversed it:

| artifact | D32 past-128 latency |
| --- | ---: |
| existing `wait_group(0)` | 172.852 ms |
| candidate `wait_group(1)` | 174.198 ms |

The candidate regressed 0.78% and was removed. Keeping the future cp.async group outstanding increased contention enough
to outweigh overlap on Thor.

## GDN small/large crossover

Standalone CuTe DSL tests used Qwen shape `H16/HV48/K128/V128`; N32 large and small both matched the NumPy output/state
reference with max errors about 3e-6 and 3.4e-5.

| N | large | small (8 CTA/state) | winner |
| ---: | ---: | ---: | --- |
| 16 | 468.0 us | 475.2 us | large +1.5% |
| 32 | 1,189.5 us | 1,213.7 us | large +2.0% |
| 48 | 1,937.4 us | 1,876.1 us | small +3.2% |
| 64 | 2,508.4 us | 2,450.9 us | small +2.3% |

The current AOT artifact exports only the large wrapper. D32 should remain large. A D64 follow-up can export a second
small symbol and select it only for N>=48; expected whole-engine gain is under 1% because GDN is 33.6% of D32 time.

## NVFP4 small-M tactic

Nsight showed M128 tiles at decode M32, but the kernel is TensorRT's internal
`FlashInferCutlassNvFp4LinearKernel`. TensorRT exposes no tile selector through `llm_build`, and its implementation is not
in this source tree. A repository-local M32/M64 replacement would be a new linear plugin and weight-layout contract, not a
surgical tactic edit. D64 is the available way to improve useful M occupancy without replacing the kernel.

## P4/D64 engine

Built engine:

```text
data/qwen38/engines/phase-b64-p4-d64-kv4096
global B64, dense P4, D64, input 2048, KV 4096, 2048 pages
engine generation 414.3 s
activation memory about 1.01 GiB
weights memory about 20.55 GiB
```

Graph-off past-128 component results:

| batch | latency | aggregate throughput |
| ---: | ---: | ---: |
| 32 | 175.974 ms | 181.8 tok/s |
| 64 | 248.819 ms | 257.2 tok/s |

D64 improves aggregate throughput 41.4% over B32 in the same engine. The launcher now provides
`thor-throughput-d64`, using 64 stable/in-flight slots and the existing Thor throughput scheduling policy.

## State transpose decision

The recurrent state is `[N,HV,K,V]`; cp.async and stores traverse V-contiguous 128-byte warp segments. Nsight indicates
register-limited L1TEX scoreboard stalls, not evidence of uncoalesced sectors. A state transpose would change TensorRT
bindings, prefill/decode compatibility, snapshot storage, and MTP state semantics without addressing the measured
occupancy limit. It is deferred.

## D64 HTTP gate

The D64 server ran one c64 warmup followed by three streaming HTTP repeats. Every run completed all 128 requests with the
exact fixed output-token count. Median results:

| workload | D32/c32 reference | D64/c64 | throughput change |
| --- | ---: | ---: | ---: |
| short burst | 156.76 tok/s | 203.40 tok/s | +29.75% |
| wave burst | 177.02 tok/s | 245.56 tok/s | +38.72% |
| decode heavy | 213.11 tok/s | 307.90 tok/s | +44.48% |

D64 is a throughput preset, not a latency preset. Short TTFT/TPOT/E2E were 1,993/287/20,086 ms, versus the lower D32
latencies obtained with half as many concurrent requests. Wave TTFT/TPOT/E2E were 1,949/234/24,691 ms. Decode-heavy was
1,807/201/53,170 ms.

The decode state view recorded 2,631 prepares and 2,557 residency hits (97.2%). All workloads had zero failed requests.
Mean/max GPU power was 40.5/79.4 W, mean/max board power was 95.3/144.9 W, and maximum GPU temperature was 67.5 C.

Machine-readable results are in `data/qwen38/results/next/phase-p4-d64-c64/summary.json`.

The D64 HTTP throughput gate passes. Exporting the small GDN symbol remains optional because its expected whole-engine
gain is below 1%; D64 itself supplies the material improvement.
