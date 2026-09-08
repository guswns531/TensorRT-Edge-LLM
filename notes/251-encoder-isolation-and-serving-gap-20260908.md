# 251. Encoder isolation and serving gap: native exact vs explicit erf

Date: 2026-09-08. Parent source: `c567529`, branch `codex/v0101-phase-forward-port`.
Artifacts: `.local/results/v0101-forward-port/encoder-isolation-20260908/`.

## Question

Note250 recovered 120 MiB of execution memory with exact FP32 erf fusion, but vision-heavy TPOT p95
was worse than a single frozen native-exact run. Separate isolated E execution from E/P/D serving
trajectory changes before modifying the scheduler or promoting the engine.

## Fixed-tensor benchmark contract

`benchmarks/phase_serving/benchmark_vision_engine_pair.py` loads the real Cosmos vision checkpoint
weights once, shares GPU input/weight addresses across two TensorRT contexts, and runs only one
context at a time on an explicit nondefault stream. Both engines remain resident; this is not a
production total-VRAM comparison. Engine deserialization, allocation, checkpoint conversion and
input copies are outside timing. No P/D contexts, image preprocessing, HTTP or sampling are active.

The inputs are deliberately **synthetic fixed tensors**, not images: seed0 normalized patch data,
zero rotary angles, fixed position indices/weights, and distinct cu_seqlens boundaries per image.
This is a compiled encoder performance test, not semantic accuracy or real-image preprocessing validation.
The script verifies finite outputs and records input/output hashes, all CUDA event samples and engine hashes.

For E1/E2/E4, test both 512 and 2048 raw patches per image (128 and 512 merged vision tokens).
Each shape/engine gets 20 warmups and 100 measurements, repeated three times in A/B, B/A, A/B order.
CUDA event spans bracket `execute_async_v3`; they include stream launch gaps and all engine work,
not a sum of isolated kernel durations. Per-iteration end-event synchronization prevents cross-iteration overlap.

Checkpoint: `.local/cosmos-reason2-2b/hf/model.safetensors`, BF16 weights converted to the engines'
declared FP16 bindings. Only single-key, matching-shape FP16 bindings are accepted. Both configs have
the same 312 bindings; runtime tensor addresses are shared. No new model/engine build was needed.

Native exact engine SHA256: `4ede4ddfcea4dadf8d5c507cf1c99bc6c5e3a36197ba4ba2a53e11f72434e2c4`.
Explicit erf engine SHA256: `ddde528bd8836aa6b78a02e1d5d5bdb73108ace924d328cf3f50b4e36cdf832a`.
Their export/build/inference provenance is in notes249–250; no engine or runtime policy changed here.

Reproduce inside the existing TensorRT26.06 container, with the repository mounted at `/workspace`,
GPU access and `LD_LIBRARY_PATH=/opt/tensorrt/lib:/usr/local/cuda/lib64:/workspace/.local/v0101-forward-build-make/examples/llm`:

```bash
python3 /workspace/.local/v0101-forward-port/benchmarks/phase_serving/benchmark_vision_engine_pair.py \
  --engine /workspace/.local/scratch/vision-exact-gelu-20260908/engine/visual/visual.engine \
  --engine /workspace/.local/scratch/vision-erf-fp32-20260908/engine/visual/visual.engine \
  --checkpoint /workspace/.local/cosmos-reason2-2b/hf/model.safetensors \
  --plugin /workspace/.local/v0101-forward-build-make/libNvInfer_edgellm_plugin.so.1.0 \
  --patches-per-image 2048 \
  --output /workspace/.local/results/v0101-forward-port/encoder-isolation-20260908/encoder-pair-2048.json
```

Use 512 and a distinct output path for the smaller shape. The first 512-patch run preceded the
explicit CLI parameter addition; its recorded batch/patch fields specify the same setting unambiguously.
The production frontend and existing C++ telemetry worktree edits remain unchanged.

## Isolated encoder results

Both benchmark processes exited successfully. Table values are medians across three run medians/p95s,
in milliseconds; 100 timed enqueues per run, 3600 total timed enqueues across both image sizes.

| Raw patches/image | Batch | Native median | Native p95 | Erf median | Erf p95 | Median time change |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | E1 | 9.4745 | 9.5161 | 9.0235 | 9.0538 | -4.76% |
| 512 | E2 | 13.7438 | 13.7884 | 13.2736 | 13.3103 | -3.42% |
| 512 | E4 | 21.8792 | 22.0551 | 21.3883 | 21.5649 | -2.24% |
| 2048 | E1 | 26.9849 | 27.1801 | 26.5149 | 26.5577 | -1.74% |
| 2048 | E2 | 53.2812 | 53.6147 | 52.6403 | 52.8065 | -1.20% |
| 2048 | E4 | 104.9919 | 105.5931 | 103.7297 | 104.2061 | -1.20% |

Every engine/shape has one output hash across its three repeats. Native and erf output hashes differ;
all outputs are finite. No bitwise cross-engine equivalence claim follows. Identical input hashes and
shared addresses are enforced by the paired harness. The larger shape reaches 8192 patches at E4,
matching the profile's 2048 total merged-token capacity, unlike the smaller 2048-patch E4 point.

The result rejects the narrow hypothesis that erf's worse serving TPOT was simply caused by a slower
isolated encoder on these shapes. It does not establish better E/P/D overlap, accuracy, or request
latency. GPU clock/thermal state is not locked; alternating order limits, but does not eliminate, drift.

## Uninstrumented Vision-heavy HTTP cross-over

All six fresh-process runs completed in this order: native1, erf1, erf2, native2, native3, erf3.
Each uses the identical vision-heavy trace, 319 generic calibration requests, V1 scalar, P8/D64/E4,
FP16 KV256, client cap64 and the same runtime binary as note250. No phase telemetry/activity capture
is enabled in these primary measurements. Each run has 64 measured requests and 2464 generated and
captured tokens. Per-engine output hashes agree across all three runs; cross-engine hashes differ.

Reproduction uses `run_policy_warmup_matrix.py` with note250's base commands and generic traces,
`--cases vision-heavy --repeats 1 --modes generic --policy-variant v1_scalar`, one distinct output
directory per run, and only `TRT_EDGELLM_VISION_ENGINE_DIR` changed. Exact resolved commands are in
`http-{native,erf}-{1,2,3}/commands.json`.

| Run | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native1 | 632.67 | 1422.04 | 3373.02 | 35.87 | 50.77 | 2807.38 | 3819.44 | 9509 |
| erf1 | 632.94 | 1418.81 | 3370.75 | 35.83 | 50.71 | 2802.71 | 3817.48 | 9389 |
| erf2 | 642.30 | 1396.55 | 3350.61 | 31.65 | 39.01 | 2633.49 | 3761.69 | 9405 |
| native2 | 630.55 | 1429.48 | 3432.28 | 32.82 | 39.22 | 2700.53 | 3836.68 | 9505 |
| native3 | 642.46 | 1407.45 | 3331.71 | 30.12 | 38.95 | 2596.64 | 3769.07 | 9525 |
| erf3 | 645.28 | 1415.59 | 3346.46 | 31.17 | 38.80 | 2636.20 | 3753.94 | 9395 |

| Variant | Runs | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fresh native exact | 3 | 632.67 | 1422.04 | 3373.02 | 32.82 | 39.22 | 2700.53 | 3819.44 |
| Fresh explicit erf | 3 | 642.30 | 1415.59 | 3350.61 | 31.65 | 39.01 | 2636.20 | 3761.69 |
| Frozen prior Current | 3 | 668.94 | 1342.22 | 3167.36 | 30.52 | 37.38 | 2532.63 | 3606.61 |
| Frozen vLLM | 3 | 579.20 | 1710.70 | 3691.37 | 63.70 | 119.58 | 4119.14 | 4229.37 |

Latency units are milliseconds, send-relative, using median of run means and median of run p95s;
not pooled quantiles or queue-inclusive arrival latency. Frozen references are note244's retained CSV,
unchanged contract, not new vLLM measurements. Fresh erf is +1.52% token/s versus fresh native,
with E2E mean -2.38% and E2E p95 -1.51%; three repetitions do not establish statistical significance.
Peak VRAM medians are 9509 versus 9395 MiB, consistent with the separately measured 120 MiB engine
requirement reduction but not an exact allocation-accounting difference.

The previous 39.14->50.18 ms TPOT-p95 comparison is not a reproducible engine-only effect here.
Both first-pair runs reach about 50 ms; both later engines reach about 39 ms. Do not declare this
thermal drift, calibration instability or formation fragmentation without additional causal evidence.
Fresh erf still trails the frozen prior Current; this experiment does not close the original full12 gate.

## Separate instrumented execution: formation and GPU gaps

After the primary runs, one additional run per engine enables `TRT_EDGELLM_EMIT_PHASE_METRICS=1`,
`TRT_EDGELLM_PHASE_TELEMETRY_LEVEL=research`, a unique `TRT_EDGELLM_PHASE_TELEMETRY_PATH`, and a unique
`TRT_EDGELLM_PHASE_ACTIVITY_PREFIX`. Commands are retained in `diagnostic-{native,erf}/commands.json`.
These results are diagnostic, not additional repetitions pooled into the uninstrumented performance table.
The CUDA activity recorder resets after calibration. The analyzer requires exactly one measurement epoch
and discards all events before it. All 177 native and 185 erf dispatches have corresponding completions.

Reproduce `gap-analysis-v2.json` with `analyze_encoder_serving_pair.py --events <native-events.jsonl>
--events <erf-events.jsonl> --output <new-output.json>`. The earlier `gap-analysis.json` contains the same
raw numbers but misleadingly names the host snapshot interval as visibility; v2 supersedes its interpretation.

| Diagnostic metric | Native exact | Explicit erf |
|---|---:|---:|
| E executions | 23 | 20 |
| E total rows / mean batch | 48 / 2.087 | 48 / 2.400 |
| E GPU total ms | 1443.98 | 1425.90 |
| P executions | 47 | 49 |
| P total rows / mean batch | 66 / 1.404 | 66 / 1.347 |
| P GPU total ms | 1281.58 | 1299.12 |
| D executions | 107 | 116 |
| D total rows / mean batch | 2400 / 22.430 | 2400 / 20.690 |
| D GPU total ms | 873.48 | 932.22 |
| D GPU service median / p95 ms | 8.280 / 8.499 | 8.234 / 8.613 |
| E-end to first subsequent P-start mean / p95 ms | 102.98 / 256.67 | 45.29 / 114.34 |
| E-end to first subsequent D-start mean / p95 ms | 204.24 / 373.29 | 148.19 / 305.29 |
| Global D GPU end-to-next-start median / p95 ms | 1.149 / 137.52 | 1.067 / 195.58 |

The E->P/D distributions match all 48 encoded request IDs to their first subsequent phase execution,
not unrelated global P/D activity. Global D gaps include time without a ready D request and therefore
are not themselves request-level TPOT or pure scheduler overhead.

The same 2400 decode rows require nine more dispatches in this erf diagnostic, despite slightly lower
median D GPU service per dispatch. E saves 18.08 ms, while P adds 17.54 ms and D adds 58.74 ms.
This is an observed formation/trajectory difference, not evidence that the isolated encoder is slower.
The E cohort distribution also changes: native E1/E2/E3/E4 counts 11/2/7/3, erf 5/4/9/2.
Erf advances encoded requests to P sooner but forms smaller D cohorts overall in this run.
It is compatible with execution–formation coupling, but a single instrumented pair is not a forced
same-snapshot causal replay and cannot isolate telemetry perturbation or online policy state.

Diagnostic serving metrics (milliseconds):

| Variant | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Native, instrumented | 639.27 | 1437.70 | 3419.65 | 30.24 | 39.46 | 2633.47 | 3787.95 |
| Erf, instrumented | 630.25 | 1427.42 | 3380.85 | 35.43 | 50.49 | 2789.93 | 3834.52 |

The diagnostic pair reproduces the adverse tail pattern but the uninstrumented three-repeat median
does not. This distinction prevents selecting the instrumented slow case as the headline result.

## GPU active spans are not idle host time

Activity CSV windows extend from first to last recorded GPU activity after measurement reset,
not from scheduled HTTP arrival. Sampling intervals are included in phase duty totals, so these
totals differ slightly from the dispatch-only GPU sums above.

| Active-span metric | Native | Erf |
|---|---:|---:|
| Window ms | 3851.54 | 3906.70 |
| All recorded streams idle | 6.21% | 6.08% |
| E only | 37.49% | 36.50% |
| P only | 33.05% | 32.99% |
| D only | 22.88% | 24.01% |
| P+D active span | 0.368% | 0.424% |
| E+D active span | 0.0015% | 0.0015% |

There are no recorded E+P or triple spans. No separate Copy spans were recorded in these runs;
that does not mean that no DMA or copies occurred. These are stream-active spans, not SM utilization
or proof that kernels execute concurrently throughout every span. Neither case shows a large increase
in all-stream idle time. Most of the long D gaps contain E/P work rather than a globally empty GPU.

## Host telemetry caveat discovered and corrected in analysis

`phaseThreeCoordinator.cpp` assigns `completionVisibleHostNs = current.hostSnapshotNs` in its unified
snapshot-difference emitter. This is the time the upper-level snapshot notices completion, not the first
CPU observation of the CUDA event. Subtracting it from the next D's enqueue time produces 32 negative
gaps out of 106 for native and 46 out of 115 for erf. Do not clamp these values to zero and call the
result runtime turnaround latency. `analyze_encoder_serving_pair.py` explicitly calls the field
`decode_snapshot_notice_to_next_enqueue_ms`, retains negatives, and warns against that interpretation.
No CPU/GPU clock subtraction is used. All GPU gaps above use the same CUDA epoch.

## Decision and next work

1. Keep the explicit-erf engine as the lower-memory candidate: exact math retained, six isolated shapes
   faster, and the fresh three-repeat serving medians mildly better than native exact. Do not yet change
   the production default: cross-engine greedy identity and full12 promotion remain open.
2. The previous engine-only TPOT regression claim is weakened by fresh cross-over results. The data
   instead show run-dependent cohort differences that can consume the small E service saving.
3. Before adding policy heuristics, record **actual** sampling completion/ready publication/next enqueue
   timestamps at the producer, rather than repurposing unified snapshot timestamps. Then compare
   high-tail and low-tail trajectories with identical instrumentation and output contract.
4. Preserve E/P/D formation invariants and investigate whether the extra small D cohorts are avoidable
   without delaying first-token progress. Do not hard-code a vision-heavy rule or sacrifice exact GELU.
5. This is a targeted E isolation plus vision-heavy investigation, not a fresh 12-workload victory or
   proof that all performance gaps versus the best frozen Current are closed. Frozen vLLM is reused;
   its uninstrumented result is not an encoder-only baseline.

Validation: both fixed-tensor GPU processes, all six uninstrumented HTTP runs, both diagnostic HTTP
runs, and three analysis unit tests passed. Project pre-commit checks passed for the new scripts/tests.
Existing C++ telemetry changes and draft note248 were left separate, not staged with this experiment.
