# 250. Exact merger GELU: explicit FP32 erf realization

Date: 2026-09-08. Source parent: `04cde77`, branch `codex/v0101-phase-forward-port`.
Results: `.local/results/v0101-forward-port/erf-merger-20260908/`.

## Question and controlled change

Note249 identified an upstream patch-merger activation mismatch, corrected it to exact GELU,
and measured about 120 MiB more TensorRT device memory in the native ONNX Gelu engine.
This experiment keeps exact GELU and tests the old direct builder's explicit realization:

```text
FP16 fc1 output -> Cast FP32 -> divide by sqrt(2) -> erf
                   |                              |
                   +---------- multiply by ((1 + erf) * 0.5)
                                      -> Cast FP16 -> fc2
```

The multiplication order and FP32 constants match `v0.10.0:experimental/builder/ops/backend.py`
`gelu()`. This is mathematical equivalence, not a guarantee of bitwise equality with a native
Gelu kernel: rounding/fusion/tactics can differ. All four final/deepstack mergers change; the
24 vision-block tanh GELUs remain unchanged. No full FP32 merger GEMM, KV adjustment, scheduler
policy change, or text-engine rebuild is made. The production frontend stays on exact Gelu;
the explicit realization is opt-in experimental export pending serving gates.

`export_vision_precision_probe.py --merger-erf-fp32` validates the merger chain and inserts
the explicit operations. It rejects simultaneous `--merger-fp32`, which changes the GEMM
boundary contract. The original native exact and tanh experiment flags remain reproducible.

## Reproduction and identities

Root below means `/home/sslab/TensorRT-Edge-LLM`, mounted as `/workspace` in containers.
Use the existing PyTorch 25.12 container and `PYTHONPATH=/workspace/.local/v0101-export-deps`:

```bash
python benchmarks/phase_serving/export_vision_precision_probe.py \
  --onnx-dir /workspace/.local/cosmos-reason2-2b/onnx-fp16-visual-opt/visual \
  --checkpoint-config /workspace/.local/atomic-packed-vision-runtime-20260824/direct-visual-max2048/visual/config.json \
  --output-dir /workspace/.local/scratch/vision-erf-fp32-20260908/onnx \
  --externalize-norms --merger-erf-fp32
```

The source ONNX SHA256 remains
`7a18305999f93a112bb01d1804a84cfcc1162e1f0ec7f2170892ad7c3e7f6d86`.
The manifest records 312 runtime checkpoint inputs and four decomposed mergers.
This is retained-ONNX transformation/export -> build -> actual inference, not a new full checkpoint
CLI export. No model/checkpoint data are duplicated.

Build in `nvcr.io/nvidia/tensorrt:26.06-py3` with GPU access, `TRT_PACKAGE_DIR=/opt/tensorrt`,
`LD_LIBRARY_PATH=/opt/tensorrt/lib:/usr/local/cuda/lib64:/workspace/.local/v0101-forward-build-make/examples/llm`,
and `EDGELLM_PLUGIN_PATH=/workspace/.local/v0101-forward-build-make/libNvInfer_edgellm_plugin.so.1.0`:

```bash
/workspace/.local/v0101-forward-build-make/examples/multimodal/visual_build \
  --onnxDir /workspace/.local/scratch/vision-erf-fp32-20260908/onnx \
  --engineDir /workspace/.local/scratch/vision-erf-fp32-20260908/engine \
  --minImageTokens 4 --maxImageTokens 2048 --maxImageTokensPerImage 512 --profilingDetailed
```

Serving uses the same V1 scalar binary, P8/D64/E4, text chunk128, slots80, FP16 KV256,
client cap64, and generic calibration as note249. The only backend environment difference is
`TRT_EDGELLM_VISION_ENGINE_DIR=/workspace/.local/scratch/vision-erf-fp32-20260908/engine/visual`.
Resolved request commands and aggregates accompany each run. Reuse frozen vLLM only under the
unchanged workload/output/concurrency contract. Performance comparisons are not correctness tests.

## Validation

The initial CPU test run passed all eight tests: exact frontend final/deepstack GELU, unchanged
block MLP, checkpoint bindings and shape guards, plus explicit erf graph type inference and an
FP16 numerical reference check. Logs: `tests.log`, `export.log`, `build.log`.
The explicit graph preserves FP16 public/merger boundaries and computes intermediate erf in FP32.

The pre-existing C++ telemetry changes and draft note248 are outside this experiment and preserved.

## Compiled memory and fusion result

The builder exited successfully, recording engine generation in 66.133 seconds. The new engine SHA256 is
`ddde528bd8836aa6b78a02e1d5d5bdb73108ace924d328cf3f50b4e36cdf832a`.
`inspector.json` records 322 I/O tensors, 370 layers, and 449072128 bytes of TensorRT device memory.

| Realization | TRT device memory MiB | Layer count |
|---|---:|---:|
| Old direct exact builder | 424.2646 | 368 |
| Checkpoint tanh, runtime norms | 428.2686 | 370 |
| Checkpoint native exact Gelu, runtime norms | 548.2666 | 392 |
| Explicit FP32 erf, runtime norms | 428.2686 | 370 |

The native-exact to explicit-erf reduction is 119.9980 MiB. It restores the matched tanh memory
contract without changing activation semantics back to tanh. It does not eliminate the remaining
approximately 4 MiB difference versus the old direct builder, or establish complete graph equivalence.

The inspector identifies four `FcCastMulErfAddMulMulCast` fusion layers, each associated with a
final/deepstack merger fc1. This is direct evidence that the compiler fused the explicit exact
activation with fc1. Layer count and memory recovered together. The old/native/tanh inspection
artifacts are retained under note249's result directory; no inference profiling ran concurrently
with serving measurements. These are TensorRT execution-memory requirements, not total VRAM or KV.

For the final merger, the native engine first consumes `merger.linear_fc1.weight` in a `Move`
layer, changing its strides from `[1,4096]` to `[4096,1]`. The explicit engine consumes that
same input directly in the fused fc1/erf layer, using an SM80 FP16 XMMA fusion tactic. This
supports a layout/materialization explanation rather than a KV explanation. Inspector metadata
alone does not prove that all 120 MiB belong to weight copies; allocation lifetime reuse also matters.

## Actual HTTP serving: four workloads, three fresh-process repeats

`vlm-r3/commands.json` records all resolved commands. The matrix exited successfully, completing
591 measured requests and 30672 requested/generated/captured output tokens (calibration excluded).
Every workload has identical token trace hashes across its three repeats. None of the four hashes
matches the native exact engine: this confirms repeat stability, **not cross-engine greedy identity**.
Rounding/tactic effects remain relevant even for mathematically equivalent activation formulas.

All latencies below are send-relative milliseconds, not scheduled-arrival queue-inclusive latency.
The existing aggregate uses the median of per-run means for columns labelled mean and the median
of per-run p95 values for p95. They are not pooled-request quantiles. Token/s is run median.

| Workload | Variant | Runs | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mixed | Prior Current | 3 | 1078.53 | 714.19 | 2213.64 | 33.16 | 40.55 | 2273.13 | 2649.30 |
| mixed | Native exact | 1 | 1032.01 | 750.66 | 2275.65 | 34.54 | 41.27 | 2378.52 | 2761.06 |
| mixed | Explicit erf | 3 | 1056.15 | 727.36 | 2252.65 | 35.21 | 43.98 | 2386.35 | 2708.67 |
| mixed | Frozen vLLM | 3 | 921.48 | 874.56 | 2541.43 | 46.97 | 84.02 | 3008.38 | 3140.85 |
| poisson | Prior Current | 3 | 1878.52 | 218.41 | 764.79 | 22.81 | 41.01 | 1660.85 | 2125.02 |
| poisson | Native exact | 1 | 1839.59 | 254.35 | 1000.62 | 22.30 | 40.75 | 1712.11 | 2176.63 |
| poisson | Explicit erf | 3 | 1833.19 | 218.06 | 907.91 | 23.46 | 42.32 | 1731.01 | 2199.21 |
| poisson | Frozen vLLM | 3 | 1800.07 | 438.11 | 902.68 | 22.19 | 45.67 | 1800.22 | 2266.55 |
| vision-heavy | Prior Current | 3 | 668.94 | 1342.22 | 3167.36 | 30.52 | 37.38 | 2532.63 | 3606.61 |
| vision-heavy | Native exact | 1 | 635.13 | 1443.18 | 3415.08 | 31.92 | 39.14 | 2688.94 | 3810.97 |
| vision-heavy | Explicit erf | 3 | 628.72 | 1434.00 | 3402.18 | 35.26 | 50.18 | 2780.63 | 3850.47 |
| vision-heavy | Frozen vLLM | 3 | 579.20 | 1710.70 | 3691.37 | 63.70 | 119.58 | 4119.14 | 4229.37 |
| multi-image | Prior Current | 3 | 298.40 | 233.13 | 317.70 | 9.40 | 12.39 | 524.70 | 535.91 |
| multi-image | Native exact | 1 | 291.79 | 248.92 | 329.16 | 9.47 | 12.71 | 542.39 | 547.92 |
| multi-image | Explicit erf | 3 | 292.86 | 247.18 | 327.17 | 9.46 | 12.68 | 540.33 | 546.09 |
| multi-image | Frozen vLLM | 3 | 244.52 | 259.81 | 402.58 | 12.42 | 16.32 | 644.30 | 653.90 |

Prior Current/frozen vLLM: `.local/results/v0101-forward-port/244-old-current-vllm-latency-audit.csv`.
Native exact: note249 `exact-gelu-full12-r1`. These are frozen comparisons, not fresh paired repeats.
Only the explicit-erf four-case matrix was run three times this turn. This is **not a new full12 gate**.
The benchmark holds output lengths and token counts fixed, but exact token content differs across engines.

| Workload | Explicit erf token/s repeats | Change vs native exact | Peak VRAM median MiB |
|---|---|---:|---:|
| mixed | 1056.15 / 1084.02 / 1053.61 | +2.34% | 9397 |
| poisson | 1872.84 / 1795.19 / 1833.19 | -0.35% | 9411 |
| vision-heavy | 628.72 / 636.88 / 626.74 | -1.01% | 9385 |
| multi-image | 293.54 / 292.86 / 290.84 | +0.37% | 9323 |

The maximum of workload peak medians is 9411 MiB; it is not the maximum individual-run peak.
Inspector accounting is the stronger evidence for the 120 MiB reduction. Serving numbers establish
successful inference and a mixed performance outcome, not statistically significant improvements.

## Interpretation and next gate

1. The extra 120 MiB is recoverable through operator realization/fusion with exact math intact.
   This supports addressing the compiled graph rather than shrinking KV or changing the phase policy.
2. Memory recovery alone does not recover best serving performance. Mixed throughput improves versus
   native exact, but TPOT p95 worsens from 41.27 to 43.98 ms; vision-heavy rises from 39.14 to 50.18 ms.
   Three-repeat erf versus single-run native is insufficient to attribute these tails solely to fusion.
3. All four throughput medians and E2E mean/p95 values are better than the frozen vLLM reference.
   This is not an all-metric win: poisson TPOT mean and TTFT p95 remain worse than frozen vLLM.
4. Do not promote the experimental engine as a universal replacement yet. Preserve native exact
   frontend semantics and the old performance reference; do not revert to tanh to regain speed.
5. Next, run alternating native-exact/explicit-erf repeats at fixed encoder batch and identical input
   bindings to isolate E service cost from scheduling. Then correlate E completion with P/D gaps in
   vision-heavy. Resolve cross-engine numerical fidelity separately and only then run a full12 promotion
   gate. No unsupported claim is made that the old direct builder graph or all v0.10.1 regressions are fixed.
