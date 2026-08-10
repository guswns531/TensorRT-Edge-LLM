# Longer decode and phase overlap workloads

All runs use the 512-capacity indexed engine, 32 slots, fixed-128 chunks,
shared CUDA context, and independent prefill/decode TensorRT contexts.

## Long decode

- `dynamic-decode256-*`: 32 requests, prompt 128, output 256
- 251 full D32 dispatches out of 260 decode dispatches
- throughput: 1,946.693 tokens/s
- TTFT median/p95: 167.295 / 300.618 ms
- E2E median/p95: 4,168.784 / 4,181.161 ms
- independent-context speedup: 1.160x

This confirms that the D32 queue remains full for hundreds of decode steps,
rather than only at the beginning of a short workload.

## Prefill/decode overlap

### P4/D16

- `overlap-p4-d16-*`: 32 requests, output 128
- 8 dispatches contained both prefill and decode work
- overlap ratio mean/p95: 0.430 / 0.443
- concurrent makespan: 44.026 ms vs sequential 56.997 ms (1.295x)
- E2E p95: 3,946.581 ms; throughput 1,029.578 tokens/s

### P8/D32

- `overlap-p8-d32-*`: 32 requests, output 128
- 4 dispatches contained both prefill and decode work
- overlap ratio mean/p95: 0.414 / 0.454
- concurrent makespan: 78.108 ms vs sequential 90.796 ms (1.162x)
- E2E p95: 2,197.613 ms; throughput 1,841.217 tokens/s

The overlap dispatches include P4/D16 and P8/D32 combinations, while the
steady-state decode dispatches reach their configured limits. Kernel-group
CUDA-event samples are stored next to each workload's `*-kernels.csv` file.
