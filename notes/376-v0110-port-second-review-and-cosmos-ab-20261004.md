# 376 — v0.11.0 port: second external review and Cosmos balanced A/B (2026-10-04)

Revises note 375's open item on Cosmos balanced.

## Outcome
- With the second-review fixes (`d075df4d`), Cosmos balanced is above frozen vLLM again: same-day interleaved median
  4448.8 tok/s vs vLLM 4315.8 (+3.1%), all three runs above it (4448.8, 4355.7, 4576.4), and level with the v0.10.1
  tip median of 4446.7 (note 369).
- The 1.9% drop reported in note 375 between `9f911c16` and `f0849333` is mostly day-to-day variance: interleaved on
  the same day the two binaries differ by 0.5% (4282.6 vs 4261.5).
- Output tokens are unchanged: Cosmos short and Gemma short are token-exact (48/48 each) against the final full24 run.
- Not yet measured: full24 x3 and MMLU on `d075df4d`; the A/B covers only Cosmos balanced.

## Same-day A/B (Cosmos balanced, generated tok/s, interleaved r1→r3 across the three binaries)
| Binary | r1 | r2 | r3 | Median | vs vLLM |
|---|---:|---:|---:|---:|---:|
| `9f911c16` (note 374) | 4267.4 | 4282.6 | 4335.9 | 4282.6 | −0.8% |
| `f0849333` (note 375) | 4261.5 | 4365.5 | 4258.6 | 4261.5 | −1.3% |
| `d075df4d` (this note) | 4448.8 | 4355.7 | 4576.4 | 4448.8 | +3.1% |

The two changes on the Cosmos decode path between `f0849333` and `d075df4d` are the removal of the attention plugin's
per-enqueue workspace recomputation (`61ea0836`) and skipping legacy `kvcache_start_index` length construction and
upload on ragged engines (`d075df4d`). The A/B does not separate their contributions.

## Second review (Codex GPT-6.1-Sol high, five read-only areas, on `10194199`)
43 findings. Port-owned items fixed:
| Commit | Finding |
|---|---|
| `61ea0836` | getWorkspaceSize sized packed scratch from Tmax/Bmax while enqueue uses {logicalBatch, chunkLimit, Hq, D}; the enqueue guard checked a per-call recomputation, not the allocated size, and missed the packed carve-outs. Workspace is now sized by the chunk limit and the guard is removed |
| `e04bac8f` | CMake reconfigure treated the compiler-written cache entry as a user override and dropped the multi-SM default |
| `42fb14e4` | Ordinary packed prefill copied padded rows for thinker hidden-state capture |
| `d075df4d` | Legacy KV lengths built and uploaded on every dispatch with no engine consumer |
| `9fddd697` | Methodology corrections: fixed warmup budget without validation (36/36 Gemma and 4/36 Cosmos cells unconverged), non-comparable memory-peak windows, achieved concurrency below the ceiling on Cosmos multi-image (5/64), short (48/64), late-vision (32/64), wave-drain (5) and Gemma multi-image (20/24), wave-drain (5/24), Cosmos frozen vLLM 0.27.1 with warmup 16 on five workloads, workload composition, mean vs median reducers, MMLU parser false positives (corrected 51.25% serving vs 51.26% reference) |

Rejected: skipping invariant ragged-metadata uploads on incremental decode, because `uploadStateIndices()` writes
`stateIndices` independently and the saving is about 1.3 KB per step.

Left open by decision (inherited from fork `fca7bd0`): nine scheduler defects (encoder wedge on mixed chat options,
stranded batch on enqueue failure, leaked admission on invalid hints, impossible lengths retried forever, headroom on
physical rather than allocatable pages, destructor use-after-free, page-multiple prefix hit never admitted, cache pages
blocking admission, unbounded metrics vector) and twelve front-end defects (backend termination on empty messages or
oversized vision, blocking image paths, moved-from retry, unbounded vision ingress, per-token UTF-8 decoding, scalar
EOS, ignored stop strings, cancel acknowledgement on a reused id, backend death reported healthy, disconnect before
headers, malformed HTTP bodies). These matter for production serving, not for the fixed-output benchmark.

## Tests (`d075df4d`)
12 C++ unit-test executables, 2302 tests, 0 failures; `test_attention_plugin.py` 182 passed, 13 skipped, 1 known
upstream failure.

## Retained paths
- `.local/results/v0110-port-cosmos-balanced-ab-20261004` (diagnostic; A/B and smoke)
- `.local/results/v0110-port-codex-review2-20261004` (diagnostic; prompts, five reports, triage)
- `.local/baselines/v0110-port-d075df4d-20261004` (frozen binary with manifest)
