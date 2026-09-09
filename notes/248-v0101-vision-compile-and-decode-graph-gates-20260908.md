# 248. Vision compile boundary, decode-only graphs, and deferred compact telemetry

Date: 2026-09-08. Branch: `codex/v0101-phase-forward-port`; starting commit `01c7f8a`.
Continues note247. This report distinguishes a reproducible candidate from a promoted production default.
Results: `.local/results/v0101-forward-port/vision-precision-20260908/`.

Status: retained incomplete draft. The compact-telemetry result placeholder in Section 6 was not materialized in
this note; later service-clock and telemetry validation is recorded by notes 271 and 272.

## 1. Experiment contract

Cosmos-Reason2-2B FP16, the same text engine, P8/D64, KV256 / 3584 MiB, stable slots80,
context2048, fixed text chunk128. Main comparisons keep E cap4. E cap1 is used only as a numerical
boundary diagnostic and is not a performance workaround. Generic calibration and retained HTTP
traces match note247. No logits rounding, answer substitution, image cache replay, or KV reduction.

Measured initial binary: `8dfd17b241ef996d2248bdbd0a05ebf95fe2fe233af668556389c97869b43217`.
The later compact-telemetry source change is not built into these initial runs. Its binary SHA256 is
`2d81b00180d05a2bfe8603e65c76c9fb71ab778a5d7b98c74544f8c65660bfbd`.
Exact per-case commands are retained next to each result.

## 2. Paired vision exports and actual builds

`benchmarks/phase_serving/export_vision_precision_probe.py` produces paired experimental ONNX exports
from the retained `onnx-fp16-visual-opt/visual/model.onnx`:

1. **Control:** retains the original FP16 merger operations.
2. **FP32-merger:** casts the final merger's two GEMM inputs/weights/biases and intervening GELU to
   FP32, then casts its public output back to FP16. The preceding24 visual blocks are unchanged.

Both exports externalize the same208 untransformed FP16 weights using the existing checkpoint binding
configuration. Shape/dtype checks reject incompatible bindings. The graph's104 folded norm tensors
remain constants; the old retained vision engine externalized all312 tensors, including those norms.
Other retained initializers include patch/position embeddings. Each export is checked, built using the
current `visual_build`, and executed with actual HTTP VLM requests. No export-only success claim.

Crucially, the control is **not a byte-identical rebuild of the previous312-input engine**. Its
constant-vs-runtime-weight boundary and compiler optimization opportunities differ. Comparing control
to FP32-merger isolates the explicit precision edit much more closely; comparing either to the old
engine includes compilation/externalization differences and must be described that way.

Audit: all104 retained norm constants exactly equal the current checkpoint's BF16 values converted
to FP16 then FP32. There were0 mismatches. See `norm-checkpoint-audit.json`. Thus preserving those
constants does not intentionally introduce different norm values. It does change the compiler-visible
representation, and floating-point operation order/tactics can change.

| Engine | SHA256 | Artifact directory |
|---|---|---|
| Control | `f1b315f2a86d2a7769f1bdd87d8bb26d31133bec169ebd5306b3bd40602db1f3` | `.local/scratch/vision-precision-20260908/control-engine/visual` |
| FP32-merger | `a4637ae06cc85605d222cd9b38f9a6bd194a945224c1de8324ee9f3c6fce92d3` | `.local/scratch/vision-precision-20260908/fp32-engine/visual` |

Build profiles: min image tokens4, max total image tokens2048, max per-image512, detailed profiling.
The large checkpoint/ONNX weight files are not duplicated. ONNX exports, build logs and engine configs
are kept under `.local/scratch/vision-precision-20260908/`; despite the scratch location, these specific
artifacts are referenced by this retained experiment and must not be deleted without reviewing this note.

The FP32-merger build reported591673344 bytes of activation memory. Its diagnostic serving sampled
peak9459 MiB, versus9323 MiB for control: approximately136 MiB additional peak usage in that screen.
These are diagnostic runs and are not evidence of final serving performance.

## 3. Canonical numerical boundary test

Target request2 is the giant-panda single-image request in the5-request multi-image trace. The whole
trace includes a two-image request. Diagnostics capture its final embedding, deepstack tensors and
first32 generated-step logits. Use `TRT_EDGELLM_CANONICAL_VISION_ADAPTER_ORDER=1`.

| Configuration | Runs | Observed maximum encoder request batch per run | Whole5-request token hash prefix |
|---|---:|---|---|
| Control, E cap4 | 3 | 2 / 3 / 4 | `48f1e184f86c` in all3 |
| Control, E cap1 | 1 | 1 | `48f1e184f86c` |
| FP32-merger, E cap4 | 3 | 2 / 3 / 3 | `9f801809d957` in all3 |
| FP32-merger, E cap1 | 1 | 1 | `9f801809d957` |

Request2's final embedding is byte-identical across the tested cap/batch conditions within each engine:

- Control: `c40a285065a4207c34feaa574408b2b6d59d29ef1f4db7422807ae15d25b05ed`.
- FP32-merger: `0658da317b57b7e5a04e1f1ff42c589601f8f2834a60d065b0ef5dd81045324d`.

An initial ordinary-adapter screen also ran3 repeats per engine and obtained the same respective
embedding/token hashes. Its command included an unrecognized `TRT_EDGELLM_CANONICAL_REQUEST_ADAPTER`
variable, which has no implementation effect; those runs are **not canonical-adapter evidence**.
The subsequent canonical runs above use the correct variable. An initial E1 startup failed because its
credit target remained4; the validated E1 cases set both cap and credit target to1. No failed startup
is included in the numerical table.

Semantic spot check before natural EOS: dog, red panda, giant panda, Entity Relationship Diagram, and
the paired dog/red-panda answer are present in the control outputs. The fixed-length benchmark ignores
EOS and continues generating control tokens afterwards. Token identity comparisons include that suffix;
semantic plausibility does not waive the exact-output gate.

### Interpretation

The simple claim "final merger must be FP32" is **not supported**: the FP16 control is already stable
in this test. FP32 is not promoted just because it is a richer numerical representation. The lower-cost
control is the first candidate for diagnostic-free full-workload validation.

The new control's deepstack tensors differ numerically from the old engine's tensors, and control and
FP32-merger have different final token hashes. Therefore this is not a cross-engine greedy-identity
pass. The existing observation that an old-engine final payload replay eliminates downstream divergence
still stands, but it does not prove that only the final merger caused the old engine's variation.

### 312-input same-source ablation

The subsequent `--externalize-norms` variant restores all312 runtime checkpoint inputs. Its104 norm
inputs are FP16 and explicitly cast to the original folded FP32 names before use. The checkpoint values
and final FP16 merger are unchanged. Engine SHA256:
`98fecd0fe67cde5dd8eb7990d14ce63cfc3b8c3c3d9d77ab8bb620d2615c4ad0`.
Artifacts: `.local/scratch/vision-precision-20260908/dynamic-norm-engine/visual`.

Canonical E cap4 ran3 times with observed maximum batches3/2/3; E cap1 ran once. All four target
embeddings equal the control's `c40a2850...` hash, and all four whole-trace token hashes equal
`48f1e184...`. Thus constant-folded norms are not necessary for stability in these tested conditions.
This does not isolate a specific old-engine tactic or prove every possible encoder batch invariant.
The more defensible result is same-source rebuild stability; exact old-to-new outputs still differ.
These runs use the later binary SHA256
`2d81b00180d05a2bfe8603e65c76c9fb71ab778a5d7b98c74544f8c65660bfbd` with telemetry disabled.

## 4. Diagnostic-free control full12

All12 retained workloads are rerun3 times with the control vision engine, ordinary adapter settings,
and every payload/logit/replay diagnostic disabled. Text engine, scheduler policy, admission, KV and
calibration are unchanged. Results are in `control-full12/`.

| Workload | token/s | vs prior Current | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 | Repeat identity |
|---|---:|---:|---:|---:|---:|---|
| balanced | 4202.55 | +0.92% | 67.71 / 176.07 | 13.14 / 15.16 | 1187.38 / 1856.88 | True |
| bimodal | 1851.92 | -0.94% | 1961.35 / 4098.10 | 18.41 / 26.89 | 4517.13 / 9312.95 | True |
| decode-heavy | 4945.17 | +0.95% | 70.32 / 196.63 | 11.28 / 12.15 | 2983.36 / 4586.50 | True |
| late-vision | 2340.70 | -2.44% | 136.17 / 527.50 | 10.07 / 10.12 | 1581.72 / 1969.96 | True |
| long-prefill | 1222.57 | -0.10% | 2074.78 / 2681.67 | 25.42 / 29.61 | 4232.63 / 5791.51 | True |
| mixed | 1021.89 | -5.25% | 757.94 / 2348.14 | 34.70 / 41.94 | 2402.49 / 2797.78 | True |
| multi-image | 293.04 | -1.80% | 255.42 / 326.55 | 9.50 / 12.77 | 540.22 / 545.53 | True |
| poisson | 1741.94 | -7.27% | 243.89 / 973.91 | 23.71 / 39.43 | 1778.19 / 2225.59 | True |
| short | 2357.68 | +0.44% | 104.00 / 184.41 | 13.30 / 27.33 | 350.23 / 431.78 | True |
| text-heavy | 1804.76 | -3.62% | 331.73 / 1114.19 | 27.82 / 42.23 | 1766.00 / 1842.45 | True |
| vision-heavy | 628.29 | -6.08% | 1465.65 / 3420.68 | 35.84 / 52.32 | 2841.64 / 3841.60 | True |
| wave-drain | 97.32 | +1.04% | 269.03 / 404.93 | 9.64 / 13.88 | 566.10 / 621.48 | True |

Throughput geometric mean versus the previous Current is -2.05%. All12 workloads are internally repeat-identical across3 runs (4539 requests / 566616 generated and captured output tokens). This is not cross-engine exact identity.

## 5. Decode-only graph isolation

This uses the **old retained vision engine**, not the new control, to avoid mixing two changes.
The existing graph mechanism is configured with:

```text
TRT_EDGELLM_CAPTURE_PHASE_GRAPHS=1
TRT_EDGELLM_CAPTURE_CALIBRATION_GRAPHS=1
TRT_EDGELLM_MAX_PREFILL_GRAPHS=0
TRT_EDGELLM_MAX_DECODE_GRAPHS=64
```

The coordinator's existing zero-prefill-cache limit prevents new P captures and trims primed P graphs.
D captures remain binding-checked and bounded; no stale-address replay or weakened cache key.
This tests whether the earlier long-prefill/mixed graph regressions require P graph activity, rather
than assuming that higher graph hit rate automatically improves end-to-end service.

| Workload | token/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|
| balanced | 4324.97 | 70.81 / 170.93 | 12.73 / 14.31 | 1154.36 / 1794.46 |
| bimodal | 1865.84 | 1995.75 / 4203.27 | 18.52 / 27.56 | 4548.03 / 9455.21 |
| decode-heavy | 5052.13 | 71.46 / 191.33 | 11.02 / 11.82 | 2916.54 / 4482.70 |
| late-vision | 2446.82 | 139.48 / 534.53 | 9.62 / 9.66 | 1517.44 / 1884.66 |
| long-prefill | 1224.83 | 2064.42 / 2713.29 | 25.56 / 31.27 | 4234.66 / 5816.55 |
| mixed | 1147.71 | 705.97 / 1909.88 | 36.86 / 64.55 | 2386.59 / 2542.63 |
| multi-image | 303.98 | 232.42 / 314.55 | 9.10 / 12.01 | 514.53 / 525.83 |
| poisson | 1794.96 | 234.84 / 913.41 | 23.24 / 38.53 | 1739.99 / 2176.96 |
| short | 2397.12 | 107.49 / 184.40 | 12.63 / 22.35 | 344.44 / 423.56 |
| text-heavy | 1927.76 | 279.84 / 999.67 | 26.48 / 40.44 | 1637.17 / 1735.69 |
| vision-heavy | 681.86 | 1317.85 / 3103.32 | 30.63 / 37.19 | 2510.22 / 3531.50 |
| wave-drain | 97.33 | 231.17 / 360.59 | 9.77 / 12.59 | 534.06 / 589.51 |

All12 cases completed once. Long-prefill returns to1224.83 token/s, versus1144.81 in the earlier P+D
capture experiment: the long-prefill throughput regression does not persist with D-only capture.
However mixed TPOT p95 is64.55 ms, still materially above the previous eager40.55 ms. Poisson
throughput1794.96 is below the previous eager1878.52. Therefore disabling P capture is not a complete
graph-regression fix, and this configuration is not promoted globally. Higher throughput does not waive
latency-tail regressions. Balanced/decode throughput exceeds the retained equal-cap vLLM point, but
those are single confirmations, not repeated all-metric wins.

### Retained vLLM comparison contract

Balanced, decode-heavy, long-prefill and bimodal use note247's fresh equal-client-cap64 result (one run).
The other eight use the unchanged trace contract's frozen three-repeat reference from note244.
No vLLM experiment was rerun merely because the Current engine or telemetry implementation changed.
The following are matched-reference metrics, not a simultaneous randomized comparison. Engine outputs
can differ across frameworks; request/output-length contracts remain fixed. Latency units are ms.

| Workload | vLLM repeats | token/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|---:|
| balanced | 1 | 4318.34 | 110.37 / 247.20 | 12.23 / 13.71 | 1154.26 / 1771.51 |
| bimodal | 1 | 1852.31 | 1573.46 / 2609.84 | 22.46 / 36.31 | 4689.19 / 9283.93 |
| decode-heavy | 1 | 4943.41 | 113.96 / 310.66 | 11.03 / 11.60 | 2969.50 / 4486.02 |
| late-vision | 3 | 2359.23 | 153.40 / 631.51 | 9.90 / 9.93 | 1576.03 / 1954.46 |
| long-prefill | 1 | 1127.42 | 1925.36 / 3027.37 | 31.92 / 36.89 | 4643.91 / 6579.62 |
| mixed | 3 | 921.48 | 874.56 / 2541.43 | 46.97 / 84.02 | 3008.38 / 3140.85 |
| multi-image | 3 | 244.52 | 259.81 / 402.58 | 12.42 / 16.32 | 644.30 / 653.90 |
| poisson | 3 | 1800.07 | 438.11 / 902.68 | 22.19 / 45.67 | 1800.22 / 2266.55 |
| short | 3 | 1983.53 | 174.92 / 263.97 | 13.36 / 24.88 | 426.71 / 503.71 |
| text-heavy | 3 | 1634.76 | 421.58 / 1232.20 | 29.21 / 47.36 | 1943.42 / 2037.81 |
| vision-heavy | 3 | 579.20 | 1710.70 / 3691.37 | 63.70 / 119.58 | 4119.14 / 4229.37 |
| wave-drain | 3 | 95.85 | 252.76 / 418.60 | 12.43 / 17.26 | 637.86 / 649.42 |

Compare with sections4/5 rather than pooling these variants: the numerical-control experiment and
decode-graph experiment change different components. Neither wins every workload and every latency
metric. In particular graph mixed improves throughput versus prior Current but harms its TPOT tail;
control vision-heavy is below both prior Current and the frozen vLLM throughput.

## 6. Compact telemetry change

`examples/llm/llm_phase_context_smoke.cpp` defers compact `dispatch` JSON construction until the
three-phase/server request batch is empty. It uses already-retained dispatch metrics and their original
execution timestamps. No new device buffers, CUDA synchronization, scheduler rule, or live request
ordering is added. Default telemetry-off behavior and full/research/counterfactual levels are unchanged.

This changes **delivery time**, not the measured event time: compact metrics are an offline drain-time
stream, not a live per-dispatch feed. The final metrics are serialized before final token/completion
records, so the existing HTTP client's completion cannot silently bypass the final metric batch.
Continuous traffic without an idle boundary delays compact metric delivery. This is an explicit trade-off,
not a claim of a real-time telemetry channel. Existing epoch-retained metrics storage is reused.

TELEMETRY_RESULTS

## 7. Validation and promotion ledger

- Export probe tests:3 passed, including inferred FP32 internals / preserved FP16 output, bad topology,
  and checkpoint shape rejection. Initial host test attempts hit log-directory permissions and a bare
  container lacked YAML; the existing configured test environment passed. Neither setup failure was
  reclassified as a test pass.
- Export -> build -> actual VLM inference completed for both experimental engines.
- Changed-file pre-commit passed for the exporter and its tests.
- The exporter, warmup matrix, warmup sweep and logit-comparison Python tests passed together:
  22 tests, including the3 new exporter tests. This set is not the same22-test set in note247.
- The phase smoke binary rebuilt successfully with `TRT_PACKAGE_DIR=/opt/tensorrt`; changed-file
  pre-commit also passed for its C++ change. The runtime library has no source changes in this revision.

| Gate | Status | Evidence / restriction |
|---|---|---|
| Paired precision export/build/inference | Passed | Control, FP32 merger,312-input rebuild all executed actual VLM requests |
| Control full12 internal repeat identity | Passed in measured suite | 12 workloads x3; not arbitrary-shape or old-engine cross identity |
| FP32 required for stability | Rejected hypothesis | FP16 control and312-input FP16 rebuild also stable |
| Constant norms required for stability | Rejected in tested conditions |312-input rebuild matches208-input control payload/token hashes |
| New vision engine no-regression promotion | Not passed | Mixed/poisson/text-heavy/vision-heavy throughput regressions |
| Old-to-new exact greedy identity | Not passed | Different compiled outputs; no replay/rounding used to disguise this |
| D-only graphs universal latency/throughput gate | Not passed | Long-prefill recovered; mixed TPOT p95 and poisson still regress |
| Full unrestricted concurrent sanitizer | Still open | Note247 unrestricted tool/runtime crash; bounded-sync pass is not a substitute |

The highest-priority remaining problem is not KV capacity: these tests keep the same KV256 pool and
FP16 cache. The next useful causal experiment is an old/new vision tactic/layer comparison at fixed
shape and request membership, followed by diagnostic-free VLM checks. For graphs, separate D service
time from P/E scheduling delay and D cohort intervals in mixed; do not enable graphs globally based
only on the balanced/decode throughput points. The unrestricted sanitizer gate requires a working
instrumented driver/tool/runtime path; no kernel suppression or forced serialization may be reported
as unrestricted success. Production defaults remain unchanged while these gates are open.
