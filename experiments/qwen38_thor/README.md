<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3.8-27B Thor comparison

This experiment compares three implementations on one Jetson AGX Thor:

1. TensorRT Edge-LLM `v0.10.0` (`71dd1bae`)
2. the independent-phase branch built on that tag
3. stock vLLM with the official Qwen3.8 NVFP4 recipe

## Fairness contract

- Model family: `Qwen/Qwen3.8-27B`, text-only first
- Weight precision: NVFP4
- KV precision: FP8 when both runtimes support it; FP16 is a separate A/B
- Context cap: 4,096 tokens (the shared engine and vLLM serving cap)
- Sampling: greedy, EOS ignored, fixed output length
- Prefix caching and speculative decoding: disabled for the vanilla comparison
- MTP: reported as a separate best-throughput comparison
- Hardware: one Thor GPU, MAXN, no unrelated CUDA workloads
- Metrics: output token/s, TTFT, TPOT/ITL, E2E, peak unified memory, and exact token count
- Repetitions: one warmup followed by three steady-state HTTP runs per backend

The upstream and phase images use the same TensorRT 10.16, CUDA 13.2, compiler,
Python dependencies, and the SM110 CuTe DSL archive tracked by v0.10.0. This
avoids reusing the older v0.9.1 archive whose generated function ABI does not
match the v0.10 runner headers.

## Build images

Create the detached upstream worktree at the path used by the script, then run:

```bash
experiments/qwen38_thor/build_images.sh
```

The default output tags are:

- `tensorrt-edge-llm:qwen38-upstream-v010`
- `tensorrt-edge-llm:qwen38-phase`

The vLLM baseline starts from `nvcr.io/nvidia/vllm:26.06-py3`. Qwen's current
recipe uses an NVFP4 checkpoint, FP8 KV cache, and optionally three MTP draft
tokens. Validate model loading in the stock image before deciding whether a
newer source overlay is required.

The original checkpoint has 18 weight shards totaling 55,562,855,904 bytes.
Keep one local copy for Edge-LLM quantization/export; do not duplicate it per
runtime.

## Implemented phase path

The independent coordinator now accepts dense-prefill engines as well as the
existing packed-prefill layout. Dense Qwen3.8 prefills use the full engine input
profile while decode-active overlap remains chunkable. The 48 GDN recurrent and
convolution state pairs have stable request ownership plus independent prefill
and decode active views.

The decode view retains unchanged rows on the GPU across dispatches. Stable
slot generations prevent a released and reused slot from inheriting stale
state, while partial cohort remaps copy only changed rows. Hybrid serving uses a
fixed decode cohort by default so normal token steps do not gather or scatter
recurrent state.

The HTTP harness supports streaming and non-streaming OpenAI chat requests. The
workload matrix in `workloads.json` contains short burst/steady, balanced
burst/Poisson, prefill-heavy, decode-heavy, bimodal, wave, shared-prefix, and
multi-turn cases. Prompts are deterministic natural-language requests and every
result records exact usage token counts.

## Initial Thor results

The shared Inferact checkpoint is 26,381,249,312 bytes across seven weight
files. It loads in both runtimes and produces the exact greedy smoke response
`THOR READY`.

The upstream engine uses B8, input 2,048, KV capacity 4,096, and 256 FP16 KV
pages. The phase candidate uses global B32, dense P1, D32, the same input/KV
limits, and 1,024 pages. Decode throughput below is aggregate batch throughput;
`llm_bench` prints `1 / latency` even for a multi-row batch.

| engine | phase | batch | E2E time | aggregate throughput |
| --- | --- | ---: | ---: | ---: |
| upstream v0.10 | prefill 2,048 | 1 | 770.9 ms | 2,656.6 tok/s |
| upstream v0.10 | prefill 2,048 | 8 | 12,592.5 ms | 162.6 tok/s |
| upstream v0.10 | decode, past 2,048 | 1 | 96.7 ms | 10.3 tok/s |
| upstream v0.10 | decode, past 2,048 | 8 | 109.9 ms | 72.8 tok/s |
| phase P1/D32 | prefill 2,048 | 1 | 594.1 ms | 3,447.0 tok/s |
| phase P1/D32 | decode, past 2,048 | 8 | 149.2 ms | 53.6 tok/s |
| phase P1/D32 | decode, past 2,048 | 16 | 174.3 ms | 91.8 tok/s |
| phase P1/D32 | decode, past 2,048 | 32 | 215.2 ms | 148.7 tok/s |

NVFP4 uses the real `FlashInferCutlassNvFp4LinearKernel`; the requested SM110
CuTeDSL GDN prefill backend is rejected and falls back to Triton/FLA.

## Repeated HTTP results

Measurements below were collected on 2026-08-24. Each cell is the median of
three steady-state runs after server startup/warmup. All runs completed every
request and the exact configured output-token count. Prefix caching and
speculative decoding were disabled for the vanilla table.

| workload | upstream v0.10 | phase P1/D32 | vLLM vanilla | phase vs vLLM |
| --- | ---: | ---: | ---: | ---: |
| short burst, 64 x 128->64, c32 | 65.63 tok/s | 143.70 tok/s | 131.40 tok/s | +9.36% |
| prefill heavy, 24 x 2048->32, c8 | 15.70 tok/s | 22.21 tok/s | 27.46 tok/s | -19.11% |
| decode heavy, 64 x 128->256, c32 | 66.41 tok/s | 193.49 tok/s | 137.46 tok/s | +40.76% |

The upstream server's supported throughput path is non-streaming B8 HTTP
batching. Its streaming path admits only the single runtime slot and returns
503 backpressure for a c32 burst, so it is excluded from the streaming latency
table.

| short-burst streaming median | phase P1/D32 | vLLM vanilla |
| --- | ---: | ---: |
| output throughput | 144.35 tok/s | 129.35 tok/s |
| TTFT | 2,354.3 ms | 938.4 ms |
| TPOT | 187.2 ms | 121.9 ms |
| E2E | 13,809.0 ms | 8,559.0 ms |

The phase path wins aggregate throughput on short and decode-heavy traffic,
but vLLM retains better per-request latency and wins the pure long-prefill case.
This distinction is intentional: throughput and latency are not collapsed into
one score.

## Separate vLLM MTP comparison

The checkpoint's native MTP head was tested only after the vanilla comparison.
MTP-1 is the better configuration on Thor; MTP-3 repeats the same MTP layer
three times and its later draft positions have low acceptance.

| backend | short burst | decode heavy | draft acceptance |
| --- | ---: | ---: | ---: |
| phase vanilla | 143.70 tok/s | 193.49 tok/s | n/a |
| vLLM vanilla | 131.40 tok/s | 137.46 tok/s | n/a |
| vLLM MTP-1 | 114.14 tok/s | 119.25 tok/s | 73.1% |
| vLLM MTP-3 | 68.57 tok/s | not repeated | 46.9% |

For this model, hardware, and natural-language workload, native MTP reduces
throughput rather than raising it. MTP-3 position acceptance was 71.6%, 44.4%,
and 24.8%, respectively.

## Locked-clock Thor optimization

The node was rebooted after the initial comparison. All subsequent measurements
use a stricter hardware contract: MAXN, CPU 2.601 GHz, GPU GPC 1.575 GHz, GPU
NVD 1.692 GHz, and EMC 4.266 GHz, with every minimum equal to its maximum.
Results before and after the reboot are not mixed.

Several RTX 3080 assumptions were rejected by direct Thor A/B:

- Logical-vocabulary strided sampling was neutral and was reverted.
- Phase CUDA Graph replay reduced throughput and remains disabled.
- D16 and low-admission latency modes reduced both service rate and tail
  latency.
- Unprofiled prefill/decode overlap improved median TTFT but hurt throughput and
  p95 TTFT.

The asymmetric hybrid builder now gives recurrent and convolution states
phase-specific batch profiles. This permits real P4/D32 and P8/D32 engines
instead of relying on a global B32 state profile.

| prefill shape | latency | aggregate input throughput |
| --- | ---: | ---: |
| P1 x 128 | 158.3 ms | 808.7 tok/s |
| P4 x 128 | 212.1 ms | 2,413.6 tok/s |
| P8 x 128 | 397.3 ms | 2,577.7 tok/s |
| P1 x 2,048 | 585.3 ms | 3,498.9 tok/s |
| P4 x 2,048 | 2,908.4 ms | 2,816.6 tok/s |

P4 is the best short/balanced engine on Thor. P8 adds only 0.35% in the HTTP
short burst while doubling prefill activation memory. P1 remains the correct
long-prefill engine.

Two SM110 presets are available through TRT_EDGELLM_SERVING_PRESET and the
launcher:

    IGNORE_EOS=1 experiments/qwen38_thor/run_phase_server.sh thor-throughput
    experiments/qwen38_thor/run_phase_server.sh thor-latency

- thor-throughput: P4/D32, prefill/decode queue targets 150/300 ms.
- thor-latency: P4/D32, queue targets 300/250 ms.

Both keep D32. Reducing the decode batch to D16 was counterproductive on Thor's
20 SMs. The throughput preset reached 95.3% full-cohort residency and 97.0%
row-level recurrent-state residency on short burst.

### Locked-clock comparison

The table below is the median of three warmed streaming HTTP runs collected
after reboot under the locked-clock contract.

| workload | phase P4/D32 throughput | vLLM vanilla | phase advantage |
| --- | ---: | ---: | ---: |
| short burst, 128->64, c32 | 187.43 tok/s | 97.21 tok/s | +92.82% |
| decode heavy, 128->256, c32 | 213.11 tok/s | 101.23 tok/s | +110.53% |
| balanced burst, 512->128, c32 | 169.04 tok/s | not repeated | n/a |

### Full phase workload matrix

Every configured workload completed with zero failed requests.

| workload | output tok/s | TTFT median | TPOT median |
| --- | ---: | ---: | ---: |
| short burst | 153.09 | 1,209 ms | 193 ms |
| short steady | 47.74 | 464 ms | 311 ms |
| balanced burst | 140.17 | 3,408 ms | 203 ms |
| balanced Poisson | 62.13 | 571 ms | 351 ms |
| prefill heavy | 18.65 | 4,512 ms | 300 ms |
| decode heavy | 212.37 | 1,119 ms | 147 ms |
| bimodal burst | 107.06 | 4,174 ms | 215 ms |
| wave burst | 177.15 | 748 ms | 168 ms |
| shared prefix | 46.19 | 3,553 ms | 294 ms |
| multi-turn | 49.30 | 872 ms | 291 ms |

The full matrix collected 654 one-second telemetry samples. Mean/max GPU power
was 37.5/80.2 W, mean/max board input power was 90.0/165.0 W, and maximum GPU
temperature was 67.5 C.

Run the reproducible matrix with:

    python3 experiments/qwen38_thor/run_http_matrix.py \
      --base-url http://127.0.0.1:8001 \
      --backend phase-p4-d32-thor-throughput \
      --results-dir data/qwen38/results/next/full-matrix-phase-throughput \
      --repeats 3

The runner refuses to start unless GPU and EMC clocks are locked.

### Long context

The phase-b32-p1-d32-input8192-kv32768-p2048 engine supports input 8K and KV
capacity 32K. Its 2,048-page undercommitted pool supports either 32 concurrent
8K contexts, 16 concurrent 16K contexts, or 8 concurrent 32K contexts.

| component | latency | aggregate throughput |
| --- | ---: | ---: |
| P1 prefill 8K | 2,757.3 ms | 2,971.0 input tok/s |
| D32, past 8K | 183.6 ms | 174.3 tok/s |
| D16, past 16K | 153.6 ms | 104.2 tok/s |
| D8, past 32K | 139.0 ms | 57.5 tok/s |

Machine-readable results are in results_2026-08-24_thor_next.json.

## Final hybrid c64 result

The final core-workload preset combines an input-256 FP8-KV prefill engine with a weight-only INT4-RTN GDN-projection
decode engine, the N16 small-grid GDN crossover, and warmed production CUDA Graphs.

| workload | Phase hybrid | vLLM vanilla | Phase advantage |
| --- | ---: | ---: | ---: |
| short burst c64 | 286.78 tok/s | 235.27 tok/s | +21.89% |
| wave burst c64 | 275.04 tok/s | 256.56 tok/s | +7.20% |
| decode heavy c64 | 341.33 tok/s | 333.19 tok/s | +2.45% |

See `notes/102-thor-hybrid-int4-fp8kv-vllm-win-20260824.md` for the engine contract, accuracy caveats, rejected A/Bs,
graph-cache statistics, and reproduction procedure.

### Optimized vLLM MTP-1 follow-up

The separate warmed c64 MTP-1 comparison uses text-only throughput mode, a 16K scheduler budget, a 16 GiB explicit KV
cache, FP8 KV, and the checkpoint's native one-token draft head. The cumulative draft acceptance rate was 72.01%.

| workload | Phase hybrid | vLLM MTP-1 | Phase advantage |
| --- | ---: | ---: | ---: |
| short burst c64 | 286.78 tok/s | 302.88 tok/s | -5.32% |
| wave burst c64 | 275.04 tok/s | 276.60 tok/s | -0.56% |
| decode heavy c64 | 341.33 tok/s | 340.48 tok/s | +0.25% |

The Phase hybrid still wins all three vanilla comparisons, but not the optimized MTP-1 short case. The eight-prompt
FP16-projection versus INT4-RTN regression gate found no semantically incorrect answer in either engine. See
`notes/103-thor-quality-and-vllm-mtp-c64-20260824.md` for the full server contract, UMA KV-cache sizing analysis, latency,
power, quality outputs, and remaining accuracy caveats.

## Validation

- Focused phase/state/scheduler tests: 15/15 passed.
- Locked-clock next-stage builder/profile/scheduler/state tests: 91/91 passed.
- TensorRT integration builds succeeded for P4/D32, P8/D32, and
  input-8K/KV-32K P1/D32 undercommitted profiles.
- Pre-commit formatting and lint hooks: passed on all changed files.
- Full C++ suite: 1,149 passed, 5 skipped, and 3 failed on the first run. The
  flaky `SigmoidGroupTopkTest.LargeScale` passed on rerun. The two reproducible
  failures are pre-existing SM110 RoPE tolerance checks (`InitializeYarnRopeCosSin`
  and `InitializeMRopeCosSin`) whose absolute error is about 1.2-1.3e-3 against
  a 1e-3 threshold; none touches the changed phase code.

## Development overlay

After changing the builder or runtime validation, rebuild only the affected
targets on top of the complete phase image:

```bash
docker build \
  -f experiments/qwen38_thor/Dockerfile.overlay \
  -t tensorrt-edge-llm:qwen38-phase-dev .
```
