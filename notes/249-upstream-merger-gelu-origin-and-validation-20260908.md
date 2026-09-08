# 249. Upstream merger GELU mismatch: attribution and actual inference validation

Date: 2026-09-08. Branch: `codex/v0101-phase-forward-port`. Parent source: `962ab1b`.
Results: `.local/results/v0101-forward-port/engine-recovery-20260908/`.

## 1. Attribution: upstream frontend, not our scheduler patch

The Qwen3-VL patch-merger activation differs between the checkpoint-to-ONNX frontend and
the reference model/direct builder. This specific mismatch was not introduced by our phase,
KV, graph, or scheduling changes.

| Source checked locally | Merger activation |
|---|---|
| Upstream tag v0.10.1 (`e8b29522938901f6df19ebeedd4b69bc8edbcd97`) | `F.gelu(..., approximate="tanh")` |
| Our committed frontend before this fix (`962ab1b`) | Identical to the upstream file |
| Upstream tag v0.10.0 checkpoint frontend | Also `approximate="tanh"` |
| Upstream v0.10.0 experimental direct builder | `.gelu()`, implemented through FP32 erf and cast back |
| Local Hugging Face Qwen3-VL implementation | `nn.GELU()` (exact / `approximate="none"`) |
| Retained old vision engine inspector | Four merger erf fusions,24 vision-block tanh fusions |

The entire frontend file, not only its activation line, has the same Git blob hash in upstream
v0.10.1 and our pre-fix HEAD: `d3db5a0286ef0db3e9a2d3d052d96a919184126b`.
SHA256 of the file contents is `fd3efa0bcdb217496f89525ff590e1f7b143b542df884b19d6901c4a147d0868`.
`git diff v0.10.1 HEAD -- tensorrt_edgellm/models/qwen3_vl/modeling_qwen3_vl_visual.py`
was empty before the fix. `git blame` follows the merger implementation to upstream `11c4a95`.
This is therefore **not established as a newly introduced v0.10.1-only bug**: the v0.10.0
checkpoint frontend already contains it. It is an upstream representation mismatch across paths.

Reproduce the attribution without relying on a moving GitHub main branch:

```bash
git show v0.10.1:tensorrt_edgellm/models/qwen3_vl/modeling_qwen3_vl_visual.py
git show v0.10.0:tensorrt_edgellm/models/qwen3_vl/modeling_qwen3_vl_visual.py
git show v0.10.0:experimental/builder/ops/vision.py
git show v0.10.0:experimental/builder/ops/backend.py
git blame 962ab1b -L 346,354 -- tensorrt_edgellm/models/qwen3_vl/modeling_qwen3_vl_visual.py
```

The local reference is `.local/v0101-export-deps/transformers/models/qwen3_vl/modeling_qwen3_vl.py`,
class `Qwen3VLVisionPatchMerger`, with `self.act_fn = nn.GELU()`.
The clean checkout `.local/upstream-v0101` also retains the tanh line, and remains unmodified.

This attribution does **not** transfer all responsibility for our end-to-end regressions upstream.
The sampler, graph policies, cohort formation, and instrumentation have their own effects.
Nor does it prove that exact GELU eliminates the old engine's within-engine numerical variation.
Approximate and exact GELU differ deterministically; shape-dependent repeat variation is a separate question.

## 2. Correction to prior numerical interpretation

Note248 correctly distinguished the208-input and312-input compiled boundaries, but did not identify
the different activation function in the old engine versus the checkpoint ONNX. Its old/new numerical
differences cannot be attributed solely to TensorRT tactic selection or norm constant folding.

The two nonlinearities are:

```text
exact:       x/2 * (1 + erf(x/sqrt(2)))
tanh approx: x/2 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
```

The24 visual transformer block MLPs legitimately use tanh GELU for this model. The final patch merger
and three deepstack mergers use exact GELU in the reference. Changing all28 would be another model
error. No new workload rule, RLS adjustment, or phase preference is involved in this fix.

## 3. Implemented changes

- `tensorrt_edgellm/models/qwen3_vl/modeling_qwen3_vl_visual.py`: change only
  `Qwen3VLPatchMerger.forward()` to `approximate="none"`; leave `Qwen3VLMLP` unchanged.
- `benchmarks/phase_serving/export_vision_precision_probe.py`: add explicit
  `--exact-merger-gelu` for a paired experiment on the retained ONNX. Match merger fc1 -> Gelu -> fc2
  topology by weight names, reject unsupported chains, and record changed-node count in the manifest.
  The old probe default remains unchanged so prior experiments are reproducible.
- `tests/python-unittests/test_qwen3_vl_merger_activation.py`: actual frontend forward tests for both
  pre-shuffle final and post-shuffle deepstack normalization, plus an unchanged vision-MLP tanh test.
- `tests/python-unittests/test_vision_precision_probe.py`: verify only merger activations change and
  reject a broken topology. All seven activation/probe tests pass in the existing CPU PyTorch container.
- `benchmarks/phase_serving/inspect_vision_engines.py`: retain TensorRT engine identity, I/O and layer
  descriptions without inference. Detailed tactic metadata is absent in the old engine; do not invent it.

The new tests compare the merger against `nn.GELU()` and also verify it is not bit-identical to the tanh
alternative on the fixture. This catches the original line rather than merely testing the new flag.

## 4. Engine and serving contract

The retained ONNX was transformed with `--externalize-norms --exact-merger-gelu`:312 runtime checkpoint
weights, exactly4 activations changed, FP16 public I/O, no FP32-merger option, no checkpoint changes.
Export/ONNX checking completed, an actual TensorRT engine was built, and actual HTTP VLM inference ran.
This is a retained-graph transformation pipeline, not a claim that a fresh full-model export CLI was run.

- ONNX: `.local/scratch/vision-exact-gelu-20260908/onnx`.
- Visual engine: `.local/scratch/vision-exact-gelu-20260908/engine/visual`.
- Visual engine SHA256: `4ede4ddfcea4dadf8d5c507cf1c99bc6c5e3a36197ba4ba2a53e11f72434e2c4`.
- Source ONNX SHA256: `7a18305999f93a112bb01d1804a84cfcc1162e1f0ec7f2170892ad7c3e7f6d86`.
- Runtime binary SHA256: `2d81b00180d05a2bfe8603e65c76c9fb71ab778a5d7b98c74544f8c65660bfbd`.
- Text engine SHA256 remains `084d039248e08dc192a57ebf38d521ee1fdca2f037a865f55ac0b0b904214b0b`.
- Same Cosmos FP16, P8/D64/E4, context2048, text chunk128, stable slots80, KV256 /3584MiB.
- Same V1 scalar policy, generic calibration239 text /319 VLM, HTTP client cap64.
- Payload replay/logit capture disabled; ordinary adapter ordering; no graph policy change.

Builder uses min image tokens4, max total image tokens2048, max per-image512 and detailed profiling.
The interrupted build session did not retain a complete terminal exit log. The generated engine/config
were subsequently loaded and successfully exercised through real serving; file existence alone was
not treated as inference evidence. Retain the truncated `build-exact-gelu.log` honestly.

Compare against the existing312-input tanh rebuild for a close activation A/B. The208-input control
also differs in norm externalization; old direct-builder engines differ in additional graph/tactic details.
Performance differences against those cannot be assigned entirely to GELU.

## 5. Actual multi-image check: three fresh runs

Five requests per run,3075 prompt tokens,160 requested/generated/captured tokens per run.
Generic calibration is outside measurement. All three whole-trace hashes are:
`075bcf0b98510bbeec6355fdc972281ad0eeafb71de6ff66a2685c428eb628ed`.
This is an internal repeat-identity pass for this trace, not old/new cross-engine exact identity.
The new hash differs from both the older tanh control and the retained old engine trajectories.

| Metric | Exact-GELU3-run median |
|---|---:|
| Output token/s | 292.19 |
| TTFT mean / p95, ms | 247.10 /328.44 |
| TPOT mean / p95, ms | 9.47 /12.71 |
| E2E mean / p95, ms | 541.54 /547.04 |
| Sampled peak GPU memory, MiB | 9443 |

The prior208-input tanh control was293.04 token/s. This is not a demonstrated performance recovery.
The unchanged-contract frozen vLLM multi-image reference remains244.52 token/s, TTFT259.81/402.58,
TPOT12.42/16.32, E2E644.30/653.90ms (3-run reference, not a fresh run).

The actual raw results and commands live in `exact-gelu-multi-r3/`. Full12 screening is retained
separately in `exact-gelu-full12-r1/`; it must not inherit the three-repeat claim from multi-image.

## 6. Remaining interpretation boundaries

Fixing a real upstream mathematical mismatch does not by itself satisfy every serving gate.
Keep old/new cross-engine identity, repeated stability, semantic quality, no-regression performance,
and unrestricted sanitizer as separate conclusions. Do not use this discovery to waive our runtime
regressions, nor use a workload-specific policy to hide them. The stable production winner has not
been selected merely because this source correction is justified.

## 7. Full12 serving screen and matched vLLM references

All1513 requests completed;188872 requested/generated/captured output tokens. Corrected merger
rows are one run each, with a separate three-run multi-image result above. Prior Current is the
earlier v0.10.1 eager3-run median. vLLM uses note247's equal-client-cap64 runs for the four large text
traces and the unchanged-contract frozen3-run references for the other eight. No vLLM rerun was
needed for an unchanged request contract. All latency units are ms, measured from client send;
scheduled-arrival admission waiting is separate. Repeated means/p95 are medians of run statistics.

| Workload | Variant | Runs | token/s | TTFT mean | p95 | TPOT mean | p95 | E2E mean | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced | Prior Current | 3 | 4164.17 | 71.21 | 174.29 | 13.28 | 15.03 | 1201.16 | 1882.52 |
| balanced | Corrected merger | 1 | 4218.48 | 72.07 | 176.91 | 13.06 | 14.49 | 1184.37 | 1852.77 |
| balanced | vLLM | 1 | 4318.34 | 110.37 | 247.20 | 12.23 | 13.71 | 1154.26 | 1771.51 |
| bimodal | Prior Current | 3 | 1869.52 | 1987.13 | 4537.04 | 18.16 | 26.00 | 4504.74 | 9628.46 |
| bimodal | Corrected merger | 1 | 1865.03 | 1957.13 | 4753.63 | 18.12 | 25.53 | 4485.65 | 9842.16 |
| bimodal | vLLM | 1 | 1852.31 | 1573.46 | 2609.84 | 22.46 | 36.31 | 4689.19 | 9283.93 |
| decode-heavy | Prior Current | 3 | 4898.64 | 71.79 | 207.18 | 11.37 | 12.24 | 3009.52 | 4639.49 |
| decode-heavy | Corrected merger | 1 | 4929.22 | 72.18 | 198.91 | 11.32 | 12.31 | 2992.46 | 4615.30 |
| decode-heavy | vLLM | 1 | 4943.41 | 113.96 | 310.66 | 11.03 | 11.60 | 2969.50 | 4486.02 |
| late-vision | Prior Current | 3 | 2399.25 | 130.03 | 493.15 | 9.81 | 9.86 | 1535.31 | 1922.13 |
| late-vision | Corrected merger | 1 | 2347.30 | 141.99 | 510.13 | 10.04 | 10.08 | 1579.89 | 1964.51 |
| late-vision | vLLM | 3 | 2359.23 | 153.40 | 631.51 | 9.90 | 9.93 | 1576.03 | 1954.46 |
| long-prefill | Prior Current | 3 | 1223.75 | 2068.31 | 2727.59 | 25.34 | 29.45 | 4221.91 | 5870.92 |
| long-prefill | Corrected merger | 1 | 1230.80 | 2065.82 | 2662.93 | 25.30 | 29.44 | 4212.53 | 5702.41 |
| long-prefill | vLLM | 1 | 1127.42 | 1925.36 | 3027.37 | 31.92 | 36.89 | 4643.91 | 6579.62 |
| mixed | Prior Current | 3 | 1078.53 | 714.19 | 2213.64 | 33.16 | 40.55 | 2273.13 | 2649.30 |
| mixed | Corrected merger | 1 | 1032.01 | 750.66 | 2275.65 | 34.54 | 41.27 | 2378.52 | 2761.06 |
| mixed | vLLM | 3 | 921.48 | 874.56 | 2541.43 | 46.97 | 84.02 | 3008.38 | 3140.85 |
| multi-image | Prior Current | 3 | 298.40 | 233.13 | 317.70 | 9.40 | 12.39 | 524.70 | 535.91 |
| multi-image | Corrected merger | 1 | 291.79 | 248.92 | 329.16 | 9.47 | 12.71 | 542.39 | 547.92 |
| multi-image | vLLM | 3 | 244.52 | 259.81 | 402.58 | 12.42 | 16.32 | 644.30 | 653.90 |
| poisson | Prior Current | 3 | 1878.52 | 218.41 | 764.79 | 22.81 | 41.01 | 1660.85 | 2125.02 |
| poisson | Corrected merger | 1 | 1839.59 | 254.35 | 1000.62 | 22.30 | 40.75 | 1712.11 | 2176.63 |
| poisson | vLLM | 3 | 1800.07 | 438.11 | 902.68 | 22.19 | 45.67 | 1800.22 | 2266.55 |
| short | Prior Current | 3 | 2347.45 | 109.05 | 188.46 | 12.92 | 22.27 | 352.18 | 434.29 |
| short | Corrected merger | 1 | 2386.54 | 100.60 | 179.92 | 13.09 | 22.52 | 344.87 | 426.71 |
| short | vLLM | 3 | 1983.53 | 174.92 | 263.97 | 13.36 | 24.88 | 426.71 | 503.71 |
| text-heavy | Prior Current | 3 | 1872.49 | 314.58 | 1107.21 | 26.65 | 39.82 | 1687.72 | 1773.05 |
| text-heavy | Corrected merger | 1 | 1839.21 | 324.59 | 1110.47 | 27.46 | 41.64 | 1738.07 | 1813.24 |
| text-heavy | vLLM | 3 | 1634.76 | 421.58 | 1232.20 | 29.21 | 47.36 | 1943.42 | 2037.81 |
| vision-heavy | Prior Current | 3 | 668.94 | 1342.22 | 3167.36 | 30.52 | 37.38 | 2532.63 | 3606.61 |
| vision-heavy | Corrected merger | 1 | 635.13 | 1443.18 | 3415.08 | 31.92 | 39.14 | 2688.94 | 3810.97 |
| vision-heavy | vLLM | 3 | 579.20 | 1710.70 | 3691.37 | 63.70 | 119.58 | 4119.14 | 4229.37 |
| wave-drain | Prior Current | 3 | 96.31 | 248.30 | 398.29 | 9.89 | 13.90 | 555.92 | 615.70 |
| wave-drain | Corrected merger | 1 | 97.34 | 266.46 | 402.43 | 9.61 | 13.76 | 564.24 | 618.59 |
| wave-drain | vLLM | 3 | 95.85 | 252.76 | 418.60 | 12.43 | 17.26 | 637.86 | 649.42 |

The corrected model still has material VLM regressions against Prior Current, including mixed and
vision-heavy. Throughput does not establish all-metric superiority: the matched vLLM text comparison
retains the TTFT versus decode/E2E trade-offs. This is a model-contract correction with a measured
serving candidate, not evidence that every runtime problem was caused by the activation mismatch.

## 8. Same312-input tanh versus exact-GELU screen

Both engines retain the same runtime checkpoint boundary, shapes and FP16 I/O. Tanh engine SHA256:
`98fecd0fe67cde5dd8eb7990d14ce63cfc3b8c3c3d9d77ab8bb620d2615c4ad0` (note248 dynamic-norm build).
One fresh run each, same runtime and calibration, four VLM traces. This controls the208/312-input
confound but includes the compiler/tactic consequences of changing the four activation operators.

| Workload | Activation | token/s | TTFT mean | p95 | TPOT mean | p95 | E2E mean | p95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | tanh | 1041.65 | 758.45 | 2315.36 | 33.70 | 41.73 | 2364.41 | 2744.31 |
| mixed | exact | 1032.01 | 750.66 | 2275.65 | 34.54 | 41.27 | 2378.52 | 2761.06 |
| vision-heavy | tanh | 628.72 | 1448.08 | 3458.80 | 35.43 | 53.36 | 2811.65 | 3834.87 |
| vision-heavy | exact | 635.13 | 1443.18 | 3415.08 | 31.92 | 39.14 | 2688.94 | 3810.97 |
| poisson | tanh | 1796.67 | 256.10 | 1041.34 | 23.43 | 42.81 | 1776.79 | 2244.51 |
| poisson | exact | 1839.59 | 254.35 | 1000.62 | 22.30 | 40.75 | 1712.11 | 2176.63 |
| multi-image | tanh | 290.28 | 250.28 | 331.78 | 9.51 | 12.80 | 545.15 | 550.68 |
| multi-image | exact | 291.79 | 248.92 | 329.16 | 9.47 | 12.71 | 542.39 | 547.92 |

No broad throughput recovery is established by this single-run paired screen. Correct model semantics
and best compiled performance remain distinct goals. Raw commands/results: `matched-tanh-vlm-r1/`
and `exact-gelu-full12-r1/`.

### Actual compiled memory contract

After performance measurements ended, both engines were inspected through TensorRT, not inferred
from nvidia-smi sampling alone:

| Engine | Runtime weight inputs | All I/O tensors | Layers | TRT device memory MiB |
|---|---:|---:|---:|---:|
| Old direct builder | 312 | 322 | 368 | 424.2646 |
| New tanh control | 208 | 218 | 368 | 428.2607 |
| New tanh, runtime norms | 312 | 322 | 370 | 428.2686 |
| New exact GELU, runtime norms | 312 | 322 | 392 | 548.2666 |

The exact-GELU build requires about120 MiB more TensorRT device memory than its matched tanh build.
This is not KV growth: KV256 and FP16 KV are unchanged. The full12 sampled peak was9555 MiB,
leaving685 MiB on the10240 MiB GPU; these sampled peaks do not replace allocator accounting.
The difference follows from compiled operator realization in this pair. It does not establish that
exact GELU inherently needs this memory: the old direct builder also computes exact GELU with a
smaller memory contract. A reference-equivalent explicit erf decomposition and fusion inspection are
therefore a useful next controlled compiler experiment, without changing model math or scheduler rules.

### Semantic spot check and validation scope

All five multi-image requests returned HTTP200. The first answers identify dog, red panda, giant panda,
Entity Relationship Diagram, and the paired dog/red-panda image respectively. Fixed-output benchmarks
ignore EOS and include subsequent generated control tokens; the natural first-answer check is not
equivalent to a complete accuracy benchmark or cross-engine greedy identity.

All seven activation/export-probe unit tests passed. Full12 and the paired four-case serving runs
completed. Formatting/static checks accompany the commit. The existing C++ compact-telemetry edit
and draft note248 were preserved separately; this change does not claim to finish their pending gates.
