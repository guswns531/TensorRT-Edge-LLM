# Gemma 4 V3: isolate KV admission, vision lifetime, and P formation

Date: 2026-09-13

## Scope and comparison contract

This campaign follows [note 299](299-gemma4-v3-mask-and-admission-bottleneck-20260913.md).
The purpose is to separate capacity-induced blocking from policy/model changes. V3 remains
`service-scaled-transition`, with P8/D24, fixed chunk128, 24 stable slots and 24 client in-flight requests.
The backbone remains INT4-AWQ; KV, embedding, PLE and LM head remain FP16. No SLO numbers are added to requests.
Encoder execution remains E4. Increasing encoded capacity means retaining more completed vision payloads;
it does **not** increase the encoder engine's execution batch to E8/E12.

The preserved comparator is the corrected vLLM seq24/KV480 MiB/P4096/sparse graph full-12 from note 295:
`.local/results/gemma4-vllm-capacity-sweep-20260912/selected-seq24-kv480-p4096-g24-full12`.
Its model, requests, arrival timestamps, output limits and client concurrency are unchanged, so it is reused.
The repeated capacity A/B is the causal evidence; a comparison with frozen vLLM is contextual, not a fresh
paired confidence-interval comparison.

Artifacts live under `.local/results/gemma4-v3-capacity-ab-20260913/`. Source HEAD at campaign start is
`a516dcdc194b44ea7cddd2ae08c84f1f1b2d4ca3`, with the allocation-budget patch described below. Manifests include
the tracked diff hash, binary/plugin/engine/config identities and workload hashes. Runs are diagnostic, not
production promotion or sanitizer evidence.

## Mechanism change: separate physical allocation from admission budget

`StableKVPageManager::Config` gains `allocatablePages`. Zero resolves to `numPages` and preserves the default.
The host free list exposes only pages `[0, allocatablePages)`. Physical pool tensors are not resized:

```text
Physical K/V pool: 192 pages per half
  K page IDs: [0,192)
  V page IDs: K_ID + 192

Budget 96                       Budget 192
  allocatable K: [0,96)            allocatable K: [0,192)
  withheld K: [96,192)             withheld K: none
  V stride: 192                   V stride: 192
  TRT engine: identical           TRT engine: identical
  allocated device bytes: same    allocated device bytes: same
```

The budget is initialized with the physical pool and can also change at a **fully drained, zero-lease**
measurement boundary. This is not live device allocation resizing, host offload or incremental reservation.
Full prompt-plus-output reservation remains unchanged. Eviction still returns pages without
copying their KV payloads. A rejected grow is transactional. Prefix reference counts remain physical-pool
sized; releasing shared pages still waits for the final owner.

Implementation:

- `cpp/runtime/state/stableKVPageManager.{h,cpp}`: budget validation, resolved default and bounded free list.
- `examples/llm/llm_phase_context_smoke.cpp`: opt-in `TRT_EDGELLM_ALLOCATABLE_KV_PAGES`, pool logging and logical
  memory-horizon accounting. `TRT_EDGELLM_MEASUREMENT_KV_PAGES` changes the budget after calibration streams drain.
- `cpp/runtime/scheduling/phaseServingRuntime.cpp`: logical memory-horizon accounting. Page-table construction
  continues to use physical `numPages`.
- `unittests/cpp/runtime/state/stableKVPageManagerTest.cpp`: default, exhaustion/reuse, physical V stride,
  invalid-budget and zero-lease-only change checks.

The production runtime does not gain a workload-specific policy branch. No attention/plugin kernel changes
are needed for the admission-budget experiment.

## Export/build/inference evidence and memory ceiling

The existing exported packed P128 ONNX is reused. `build_gemma_kv_capacity_engine.sh` builds a new 192-page
engine with the same P8/D24/input1024/KV capacity2048/profile limits and `--allowKVPoolUndercommit`. Config
comparison shows only `builder_config.max_kv_pool_pages` changes, 96 to 192. This does not establish identical
TensorRT tactics between independently built engines; the primary A/B therefore uses one 192-page engine.

Existing ONNX sidecars are hard-linked into the new engine directory. The builder recognizes equivalent
files and skips payload copying. This avoids duplicate embedding, external-FFN and PLE sidecars.

The first integrated 192-page/E4 inference attempt reached P/D shape warm-up:

```text
mode=generic batches=32 requests=632 overlap_samples=60 safe_probes=20
```

It then failed while allocating vision resources:

```text
Vision context workspace: required=313528320 prefill_available=183647744 bytes
CUDA runtime error in cudaMalloc(&data, memoryCapacity): out of memory
backend exited with code 139
```

This is an OOM before HTTP measurement, not an observed performance result or proof of an attention OOB.
The failure is retained at `kv/p192-budget96-encoded4/repeat-001/balanced/` and in `campaign-kv.log`.
No integrated 192-page VLM support is claimed. Disabling the encoder for both sides permits the text-only KV
budget A/B. Because that also removes the three-phase encoder route, those absolute results must not be
presented as a same-contract production VLM-engine replacement.

## Initialization confound found and corrected

The first text-only A/B applied the budget before startup calibration. Even balanced, which does not exhaust
96 pages during measurement, changed strongly. Driver logs exposed different generic-calibration P+D safe
probe counts: 25 versus 53 in representative runs. Equal calibration input/count was **not** equal acquisition
trajectory: the smaller budget had already restricted warm-up formation.

Those runs under `cells/` are retained only as capacity-plus-learning diagnostics. They are not the primary
same-calibration comparison. The campaign terminated with a driver status 127 after the repeat-2 balanced
cell while its runner was being revised; it did not complete three repeats. That is a harness interruption,
not a model-runtime failure. Subsequent campaigns execute immutable runner snapshots stored in their result
root and must not mix those partial runs into a completed matrix.

The corrected primary campaign uses `fair-cells/`: both conditions start with all 192 pages exposed, perform
the same startup and 49-request calibration procedure, then apply 96/192 at calibration end. Existing stream
barriers establish GPU completion; the allocator additionally rejects a change if even one stable lease is
still live. Physical K/V tensors and strides are unchanged. As with independent fresh-server repeats, the
exact timing-dependent posterior is not asserted byte-identical; the capacity knob no longer deliberately
limits warm-up acquisition. Calibration counts and convergence remain observable in every driver log.

## Validation before capacity measurements

- Common scheduler/async-server regression: 179 tests passed.
- `unitTestRuntimeState`, filtered to stable KV/page-table tests: 22 tests passed, including all 9 stable KV
  manager tests and the physical V-stride budget test.
- These are unit/runtime checks, not CUDA sanitizer or semantic cross-engine exact-output promotion.
- Final measurement-budget version: 23 stable-KV/page-table tests plus 180 scheduler/async-server tests passed
  (203 total). This includes the zero-lease budget guard and the small-wavefront counterexample.

## Experiment sequence

1. Same physical 192-page text-only engine, budgets 96/192: balanced, long-prefill and bimodal, three fresh-server
   repeats each. Variant order is reversed on even repeats. Each server performs the existing generic startup
   shape warm-up and 49 HTTP calibration requests with the full pool. Budget activation and activity reset occur
   only at the drained measurement boundary.
2. Existing physical 96-page integrated E4 engine, encoded downstream capacity 4/8/12: mixed, vision-heavy and
   multi-image. KV reservation, E4 batch limit, 25 ms E formation wait, P8/D24 and chunk128 stay fixed. Encoded
   capacity is a startup configuration, so its effect on generic-calibration formation/online learning is part
   of this exploratory result, not a frozen-posterior policy-only comparison.
3. P mechanism inspection/A/B: enable/disable the existing persistent wavefront cohort on six workloads,
   keeping chunk128, engine/capacity, ragged packing and initial/continuation compatibility unchanged. This
   switch is also applied at startup, so warm-up acquisition may change. No selector weights are tuned.

Run tooling:

```bash
python3 benchmarks/phase_serving/run_gemma_capacity_ab.py --stage kv --repeats 3 \
  --result-root .local/results/gemma4-v3-capacity-ab-20260913/fair-cells
python3 benchmarks/phase_serving/run_gemma_capacity_ab.py --stage vision --repeats 1 \
  --result-root .local/results/gemma4-v3-capacity-ab-20260913/fair-cells
python3 benchmarks/phase_serving/run_gemma_capacity_ab.py --stage formation --repeats 1 \
  --result-root .local/results/gemma4-v3-capacity-ab-20260913/fair-cells
```

Per-cell raw HTTP requests, phase intervals and gateway traces are retained because admission, lifecycle
and cohort conclusions depend on them. Summary JSON reports TTFT/TPOT/E2E mean and p95, generated-token and
request throughput, phase masks, dispatch distributions, reconstructed reservation and request-stage waits.
Activity masks are phase work envelopes, not SM utilization. Zero dedicated Copy intervals does not imply
zero memory traffic.

## Corrected KV results: three repeats per condition

Both columns use the same physical 192-page text-only engine and full-pool calibration. Numbers are the mean
of the three run-level values; p95 columns average the three run-level p95s, not a pooled request percentile.

| Workload | Allocatable pages | Output tok/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---:|---:|---:|---:|---:|
| balanced | 96 | 1191.78 | 89.61 / 229.65 | 16.48 / 18.05 | 1460.76 / 2243.32 |
| balanced | 192 | 1193.29 | 89.45 / 230.06 | 16.48 / 18.10 | 1460.01 / 2237.12 |
| long-prefill | 96 | 425.16 | 2425.27 / 3724.49 | 21.78 / 25.47 | 4265.25 / 6117.39 |
| long-prefill | 192 | 307.12 | 4782.12 / 8489.06 | 11.53 / 16.54 | 5723.15 / 9660.26 |
| bimodal | 96 | 571.74 | 2321.94 / 5713.38 | 18.95 / 28.07 | 4917.18 / 10627.12 |
| bimodal | 192 | 673.69 | 1699.04 / 4300.70 | 20.83 / 35.63 | 4347.87 / 9653.13 |

Percent changes below are 192/96 minus one. Positive throughput is good; positive latency is bad.

| Workload | tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | +0.13% | -0.18% | +0.18% | +0.02% | +0.25% | -0.05% | -0.28% |
| long-prefill | -27.76% | +97.18% | +127.93% | -47.05% | -35.04% | +34.18% | +57.91% |
| bimodal | +17.83% | -26.83% | -24.73% | +9.90% | +26.90% | -11.58% | -9.17% |

### Why the larger pool is not promoted

| Workload | Mean admission wait, 96 / 192 ms | Mean P batch, 96 / 192 | Mean D batch, 96 / 192 |
|---|---:|---:|---:|
| balanced | 0.01 / 0.01 | 1.89 / 1.91 | 18.08 / 18.10 |
| long-prefill | 1929.91 / 3.73 | 1.54 / 5.10 | 8.31 / 3.68 |
| bimodal | 2008.37 / 105.03 | 1.53 / 2.37 | 8.92 / 10.49 |

1. **The balanced negative control is now effectively unchanged.** Its old pre-calibration capacity A/B
   difference disappeared when capacity no longer constrained acquisition. This validates the comparison
   correction, not a universal statement that calibration never matters.
2. **Long-prefill proves that capacity-induced admission delay is real but insufficient to explain performance.**
   Larger capacity removes almost two seconds of admission wait, yet first-token latency doubles. P groups
   become larger, but decode groups shrink sharply. This is an execution/formation trade-off after admission,
   not an argument that the extra pages were unavailable. TPOT improves while TTFT/E2E deteriorate; no single
   latency statistic captures the result.
3. **Bimodal has potential but is unstable.** Budget-192 throughput is 808.34, 399.89 and 812.85 tok/s across runs.
   Budget-96 gives 581.67, 585.29 and 548.27. The average improvement is not a reliable uniform win.
   In the slow budget-192 run D dispatches increase from 714 to 2141 and mean D batch falls from 13.47 to 4.49;
   P dispatches fall from 165 to 74 while mean P batch increases from 1.64 to 3.66. Generic calibration reports
   the same 49 requests, 53 P+D safe probes and one required/one calibrated exact key in those two runs.
   Thus different acquisition **counts** are not the remaining explanation. Timing-dependent costs/posteriors
   and early trajectory formation are hypotheses; exact causal attribution needs a frozen-state replay.
4. **Physical memory is identical inside this A/B.** Both conditions retain 192-page buffers. The experiment
   isolates exposure to admission, not the memory savings of a smaller device allocation. Integrated E4 plus
   the physical 192-page pool still fails initialization with OOM.

The default/current integrated engine pointer remains on the original 96-page engine. No workload-specific
capacity override or new scheduler score is installed. This campaign is not a full-12 promotion gate.

Structured results: `fair-cells/summary-kv.json`, `fair-cells/manifest-kv.json`, and the individual repeat roots.
The partial pre-calibration runs under `cells/` are excluded from all tables above.

## Vision lifetime capacity: one exploratory repeat per cell

This matrix retains the integrated E4/P8/D24 route, physical KV96, full reservation and wavefront enabled.
Only encoded downstream capacity changes. Capacity applies during initialization/calibration as described
above; this is not a frozen-posterior policy-only experiment. All nine cells completed their fixed outputs.

| Workload | Encoded capacity | Output tok/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|
| mixed | 4 | 497.20 | 730.16 / 2159.86 | 20.21 / 26.29 | 1690.64 / 2590.87 | 9375 |
| mixed | 8 | 625.83 | 373.28 / 1292.58 | 24.44 / 31.35 | 1500.38 / 1999.65 | 9381 |
| mixed | 12 | 665.22 | 243.85 / 517.76 | 27.92 / 37.13 | 1512.22 / 2387.27 | 9385 |
| vision-heavy | 4 | 329.12 | 1635.08 / 3453.14 | 14.97 / 22.92 | 2240.97 / 3798.76 | 9377 |
| vision-heavy | 8 | 409.72 | 980.98 / 2322.00 | 23.21 / 29.25 | 1882.39 / 3014.32 | 9383 |
| vision-heavy | 12 | 466.91 | 655.93 / 1559.96 | 27.91 / 34.83 | 1725.52 / 2429.71 | 9385 |
| multi-image | 4 | 200.25 | 1066.99 / 2145.03 | 11.67 / 16.85 | 1428.81 / 2428.88 | 9375 |
| multi-image | 8 | 273.50 | 682.53 / 1302.66 | 18.61 / 21.79 | 1259.48 / 1572.89 | 9379 |
| multi-image | 12 | 304.24 | 559.89 / 982.96 | 21.31 / 28.15 | 1220.56 / 1569.35 | 9385 |

| Workload | Capacity12 throughput change | TTFT mean change | E2E mean change | E2E p95 change | TPOT mean change |
|---|---:|---:|---:|---:|---:|
| mixed | +33.79% | -66.60% | -10.55% | -7.86% | +38.16% |
| vision-heavy | +41.87% | -59.88% | -23.00% | -36.04% | +86.42% |
| multi-image | +51.94% | -47.53% | -14.58% | -35.39% | +82.59% |

### Mechanism evidence

| Workload | Capacity | Mean E queue wait ms | Mean E batch | Mean P batch | Mean D batch |
|---|---:|---:|---:|---:|---:|
| mixed | 4 | 1101.73 | 1.28 | 1.44 | 8.45 |
| mixed | 8 | 328.98 | 1.39 | 1.76 | 15.48 |
| mixed | 12 | 78.32 | 1.52 | 1.63 | 15.65 |
| vision-heavy | 4 | 1885.90 | 1.37 | 1.47 | 4.62 |
| vision-heavy | 8 | 952.96 | 1.45 | 1.65 | 8.99 |
| vision-heavy | 12 | 518.86 | 1.50 | 1.67 | 11.82 |
| multi-image | 4 | 822.85 | 1.54 | 1.66 | 2.41 |
| multi-image | 8 | 422.00 | 2.00 | 1.84 | 5.69 |
| multi-image | 12 | 230.54 | 2.00 | 2.00 | 7.95 |

The larger window mainly removes lifetime backpressure and enables earlier downstream progress/cohort
formation; it does not turn E4 into a larger engine. E batch increases are modest compared with E wait and D
batch changes. Multi-image execution remains subject to the engine's media/input-token limits even when more
completed requests can wait in P. Approximately ten additional peak MiB are enough to expose substantial
serving opportunity here; KV pool size does not change.

The result is not Pareto dominance. D batches are denser and total completion improves, but per-token latency
increases. In mixed, capacity8 has better E2E mean/p95 than capacity12 despite lower throughput. A global
capacity12 setting cannot be justified solely from token throughput; repeat and full-12 tail validation remain
necessary. No per-workload setting is installed.

### Frozen vLLM remains stronger on throughput

The preserved comparator produces 703.81 / 559.83 / 381.34 tok/s for mixed / vision-heavy / multi-image.
Capacity12 gives 665.22 / 466.91 / 304.24, approximately -5.48% / -16.60% / -20.22%. The opportunity is real,
but this experiment does **not** establish a vLLM win or a complete solution to the VLM first-token deficit.

Structured results: `fair-cells/summary-vision.json`, `fair-cells/manifest-vision.json` and per-cell raw traces.

## P mechanism counterexample

`SmallWavefrontCohortExcludesLaterCompatibleContinuations` starts a P1 cohort, completes its first chunk, and
then adds three otherwise compatible continuation requests. With wavefront enabled, the next dispatch remains
P1 on the original request. With it disabled, the next dispatch is P4. Both use chunk128 and the same legal
initial/continuation rules; no RLS estimator is involved. This establishes a real mechanism limitation but
does not yet prove that disabling wavefront improves the production traces: cohort continuity can also help
decode formation and fairness, so its serving trade-off must be measured.

## Correctness defect exposed by wavefront-off

The first integrated wavefront-off condition failed during generic HTTP calibration, before balanced
measurement, with:

```text
independentPhaseCoordinator.cpp:189: Independent prefill batch cannot mix producer classes
```

`isPrefillBatchCompatible()` already required matching `prefillClass`, but `popBatch()`'s separate
`collectCompatible()` path omitted that constraint. Wavefront membership happened to prefilter the producer
family and hide the omission. Removing the cohort restriction exposed invalid mixing of token and external
embedding inputs. This is a defect in the custom common phase scheduler, not evidence of an upstream
TensorRT/v0.10.1 attention defect or a Gemma-only incompatibility.

The fix adds the same producer-class equality constraint before collecting rows. Class is a binding/producer
mechanism distinction, not a workload label or performance heuristic. `RaggedPrefillWithoutWavefrontKeepsProducerClassesSeparate`
regresses four interleaved text/external rows and requires a homogeneous two-row batch.

After the fix, 23 KV/page-table plus 181 scheduler/async-server tests pass (204 total). The new inference
binary is SHA-256 `7c9e8fd6b764f3898f459f0f2a3de17408eac1e2014eaed4662d245c9413e67b`.
KV and vision matrices above used the preceding budget binary
`3cfeec1bf12b1a2f9c4b115b9e3f50e68b02bfbf3c1d3ccb1cac0032af1b859e`.

The failed pair under `fair-cells/formation/` is diagnostic only. Its on condition completed six workloads;
its off condition produced no valid measured balanced result. The corrected on/off matrix is rebuilt under
`class-fixed/` using the same new binary on both sides. Results from before and after this source change
must not be spliced into one same-binary A/B.

## Corrected wavefront A/B: six real-request workloads

Both sides use the producer-class-fixed binary, physical/allocatable KV96, encoded capacity4, E4/P8/D24 and
chunk128. These are one-repeat exploratory results, not stable promotion evidence.

| Workload | Wavefront | Output tok/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---|---:|---:|---:|---:|
| balanced | on | 1179.25 | 86.58 / 217.31 | 16.65 / 18.12 | 1474.21 / 2269.58 |
| balanced | off | 1193.81 | 98.37 / 213.85 | 16.27 / 17.49 | 1460.82 / 2259.05 |
| long-prefill | on | 433.54 | 2225.64 / 3127.68 | 23.28 / 26.74 | 4166.79 / 5968.09 |
| long-prefill | off | 448.33 | 2150.81 / 2880.73 | 21.80 / 24.61 | 3984.99 / 5708.16 |
| bimodal | on | 585.41 | 2192.11 / 5421.85 | 19.89 / 32.40 | 4820.21 / 10281.58 |
| bimodal | off | 591.24 | 2149.80 / 5421.54 | 19.34 / 29.44 | 4765.49 / 10139.72 |
| mixed | on | 500.68 | 672.59 / 2250.54 | 19.58 / 25.17 | 1602.58 / 2662.64 |
| mixed | off | 567.92 | 641.91 / 1821.13 | 17.85 / 23.92 | 1496.20 / 2087.87 |
| vision-heavy | on | 343.67 | 1562.02 / 3336.61 | 14.05 / 22.86 | 2138.50 / 3665.19 |
| vision-heavy | off | 331.25 | 1805.56 / 2781.44 | 13.24 / 17.79 | 2322.36 / 3174.75 |
| multi-image | on | 166.25 | 1461.14 / 2840.52 | 12.41 / 17.22 | 1845.89 / 3097.58 |
| multi-image | off | 243.99 | 845.82 / 1448.83 | 11.23 / 13.84 | 1193.95 / 1791.40 |

| Workload | off/on tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | +1.24% | +13.62% | -1.59% | -2.28% | -3.50% | -0.91% | -0.46% |
| long-prefill | +3.41% | -3.36% | -7.90% | -6.37% | -7.97% | -4.36% | -4.36% |
| bimodal | +1.00% | -1.93% | -0.01% | -2.78% | -9.14% | -1.14% | -1.38% |
| mixed | +13.43% | -4.56% | -19.08% | -8.83% | -4.96% | -6.64% | -21.59% |
| vision-heavy | -3.61% | +15.59% | -16.64% | -5.81% | -22.21% | +8.60% | -13.38% |
| multi-image | +46.76% | -42.11% | -48.99% | -9.52% | -19.67% | -35.32% | -42.17% |

### Cohort/frontier evidence

| Workload | P dispatches on / off | Mean P batch on / off | D dispatches on / off | Mean D batch on / off | P1 with multiple ready, on / off |
|---|---:|---:|---:|---:|---:|
| balanced | 45 / 37 | 1.87 / 2.27 | 297 / 294 | 18.10 / 18.29 | 8 / 0 |
| long-prefill | 337 / 277 | 1.34 / 1.63 | 577 / 553 | 9.32 / 9.72 | 69 / 31 |
| bimodal | 183 / 175 | 1.48 / 1.55 | 1007 / 1020 | 9.55 / 9.43 | 41 / 17 |
| mixed | 103 / 68 | 1.44 / 2.18 | 350 / 291 | 8.18 / 9.84 | 30 / 8 |
| vision-heavy | 112 / 89 | 1.62 / 2.04 | 482 / 527 | 4.98 / 4.55 | 34 / 3 |
| multi-image | 48 / 39 | 1.42 / 1.74 | 349 / 175 | 1.78 / 3.54 | 23 / 0 |

Almost every counted P1/multiple-ready decision still has no larger legal P candidate in its recorded frontier.
Turning wavefront off reduces many such snapshots but does not erase them: initial/continuation, producer
class, ragged shape and preparation readiness still matter. A ready request count alone does not establish
that a larger homogeneous batch was legal at that boundary.

Multi-image's large percentage is especially provisional: the new on control is 166.25 tok/s, whereas other
fresh on controls in this campaign were near 200. The direction agrees with fewer P/D dispatches, but an
unrepeated +46.76% is not a stable effect estimate. Vision-heavy is the counterexample to universal removal:
larger P batches coexist with more fragmented D and worse mean TTFT/E2E, even though p95 improves.

No wavefront default is changed. The next mechanism candidate is bounded augmentation of compatible
continuations while retaining a continuity candidate, rather than deleting continuity protection everywhere.
Neither a workload name nor a shape-specific performance rule is added to the runtime.

## Frozen vLLM reference: all latency metrics

These are the retained optimized comparator values, not new runs. Current conditions are in the tables above.
Text-only KV diagnostics deliberately omit the encoder route and cannot be promoted as an equal-production
VLM replacement solely from these absolute comparisons. Current full telemetry also differs from vLLM's
client-facing telemetry. TTFT/TPOT/E2E are HTTP-send-based service latencies; scheduled-arrival/client-dispatch
delay is separately retained and must not be folded into them without changing the evaluation contract.

| Workload | Output tok/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms | Peak MiB |
|---|---:|---:|---:|---:|---:|
| balanced | 771.46 | 134.27 / 234.33 | 23.76 / 24.56 | 2128.03 / 3244.79 | 8559 |
| long-prefill | 500.26 | 543.92 / 1442.01 | 36.37 / 44.60 | 3554.56 / 5866.97 | 8841 |
| bimodal | 600.16 | 317.41 / 878.39 | 28.83 / 37.42 | 4389.95 / 9582.02 | 8841 |
| mixed | 703.81 | 276.41 / 410.58 | 26.48 / 32.39 | 1473.77 / 2162.56 | 8843 |
| vision-heavy | 559.83 | 280.57 / 390.78 | 30.59 / 40.54 | 1420.68 / 2138.33 | 8843 |
| multi-image | 381.34 | 188.86 / 227.16 | 29.70 / 35.54 | 1109.61 / 1300.30 | 8843 |

| Workload | Wavefront-off / vLLM tok/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | +54.75% | -26.74% | -8.74% | -31.50% | -28.81% | -31.35% | -30.38% |
| long-prefill | -10.38% | +295.42% | +99.77% | -40.07% | -44.83% | +12.11% | -2.71% |
| bimodal | -1.49% | +577.30% | +517.22% | -32.92% | -21.32% | +8.55% | +5.82% |
| mixed | -19.31% | +132.23% | +343.55% | -32.60% | -26.14% | +1.52% | -3.45% |
| vision-heavy | -40.83% | +543.54% | +611.76% | -56.73% | -56.13% | +63.47% | +48.47% |
| multi-image | -36.02% | +347.86% | +537.79% | -62.19% | -61.07% | +7.60% | +37.77% |

The throughput deficit remains on five of these six workloads for wavefront-off alone. Some E2E tails and
all mean/p95 TPOT values are better, but neither isolated knob establishes general vLLM superiority.

## Masks: capacity changes trajectories, not just idle coverage

Percentages are phase-event envelopes. E/P/D duty may sum above 100% due to overlap; Copy is zero in these
direct-output runs but memory traffic on other streams still exists.

| Workload | Encoded capacity | E duty % | P duty % | D duty % | 0000 % | E/P/D overlap % |
|---|---:|---:|---:|---:|---:|---:|
| mixed | 4 | 10.53 | 36.60 | 63.92 | 2.30 | 13.35 |
| mixed | 8 | 13.36 | 40.81 | 64.32 | 1.88 | 20.37 |
| mixed | 12 | 14.25 | 47.80 | 65.77 | 2.04 | 29.85 |
| vision-heavy | 4 | 12.59 | 38.05 | 62.92 | 2.61 | 16.18 |
| vision-heavy | 8 | 15.66 | 42.18 | 61.05 | 2.02 | 20.91 |
| vision-heavy | 12 | 17.56 | 47.26 | 60.58 | 2.14 | 27.53 |
| multi-image | 4 | 12.22 | 30.12 | 61.97 | 2.39 | 6.70 |
| multi-image | 8 | 16.06 | 40.58 | 56.18 | 1.83 | 14.63 |
| multi-image | 12 | 18.42 | 47.11 | 52.43 | 1.79 | 19.74 |

The 0000 ratio changes by less than one percentage point, while throughput changes by 34--52%. Therefore
the gain cannot be explained simply as filling previously uncovered phase intervals. Earlier E/P readiness,
denser decode cohorts and different overlap placement matter. These envelopes do not measure SM-active
cycles or true kernel-only occupancy. All sixteen masks and segment timestamps are retained in each cell's
analysis JSON/CSV; no fresh Nsight kernel-level characterization is claimed.

## Learning and promotion limits

All conditions use a fresh server, generic P/D startup calibration, then 49 generic HTTP requests. No target
workload trace is used to pretrain the runtime. Nevertheless, identical request counts are not identical
learned parameters or authority. Representative mixed calibration statuses illustrate this explicitly:

| Encoded capacity | HTTP calibration requests | E+P probes | E+D probes | P+D safe probes | Converged status |
|---|---:|---:|---:|---:|---|
| 4 | 49 | 2 | 0 | 42 | false |
| 8 | 49 | 4 | 0 | 38 | true |
| 12 | 49 | 5 | 0 | 35 | true |

The larger window changes acquisition opportunities. Its apparent gain may include better E authority in
addition to lifetime relief. Converged status can come from contextual directions rather than complete exact
key coverage; it is not a certificate of held-out accuracy on every production shape. Separating the two
effects requires one common post-calibration state and a drained-boundary encoded-window change/replay.

Final integrity checks pass on 39 measured cells: KV18, vision9, corrected formation12. All fixed requested
outputs complete; lifecycle counts match request counts; masks sum to 100%; reconstructed leases end at zero;
D useful-token count equals output tokens minus first tokens. Tests pass 204/204. These are not sanitizer,
semantic cross-engine exact-greedy or repeated full-12 promotion gates.

## Next implementation order

1. Make encoded-window activation a drained measurement-boundary option or replay a common calibrated state;
   repeat the promising 4/8/12 cases without letting capacity change startup acquisition.
2. Offer a bounded compatible-continuation augmentation candidate alongside the current continuity candidate.
   Retain producer-class, initial/continuation, profile, ownership and single-inflight correctness guards.
3. Evaluate one global capacity/formation configuration and its joint interaction, not a best setting selected
   by workload label. Repeat regressions first, then run all twelve workloads with TTFT/TPOT/E2E mean/p95.
4. Attribute the remaining first-token deficit using admission/ready/dispatch timelines and the decode cycle,
   then test existing graph/submission optimizations without changing KV semantics opportunistically.

Current defaults remain KV96, encoded4, wavefront on, chunk128. The producer-class correctness fix and optional
KV diagnostic budget are implemented. No pooled-KV device resizing, mandatory offline registry, new explicit
SLO, workload-specific policy, or automatic vLLM-win claim is introduced by this campaign.
