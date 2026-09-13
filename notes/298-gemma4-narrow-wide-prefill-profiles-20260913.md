# Gemma 4 narrow/wide packed-prefill profiles in one engine

## Question and scope

Note 297 found that a P512 engine improves long-prefill progress but regresses
multi-image throughput even when its runtime chunk remains 128. This experiment
asks whether one engine can retain a narrow P128 optimization profile while
supporting P512 rows. It does not change weights, KV allocation, phase scheduling
policy, encoder batching, or the 128-token overlap frontier.

The existing optional auxiliary prefill profile is sufficient for this proof.
Its serialized names still contain `vision`; the bindings and packed chunk-limit
carrier are producer-independent. No new exporter format or attention kernel is
introduced.

## Engine contract

```text
one TensorRT engine / one shared set of weights
├── profile 0: P8, logical row <=512, carrier <=4096
├── profile 1: D24, one token per row
└── profile 2: P8, logical row <=128, carrier <=1024

one CUDA context
├── P stream: primary and auxiliary TRT contexts serialize
│             both borrow the same P activation arena
├── D stream: independent TRT decode context
└── E stream: retained E4 visual engine
              tiered VLM mode borrows the P arena and serializes E/P
```

The primary P512 profile needs 734,580,224 activation bytes. The D profile needs
28,401,664 bytes. The auxiliary P128 profile needs 183,647,744 bytes, exactly
matching the previous standalone P128 profile's reported workspace requirement.
Workspace identity alone is not proof of tactic identity or throughput parity.

The shared P arena remains the maximum of both profile requirements:
734,580,224 bytes. The auxiliary profile does **not** remove the P512 memory cost.
This campaign first tests tactic/shape isolation, not dynamic arena resizing.

Artifacts:

```text
.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/
├── onnx-int4-awq-packed-p512/llm
└── engine-profiled-p8x512-p8x128-d24-kv2048-p96

.local/results/gemma4-profiled-prefill-20260913/
└── manifest.json
```

The builder used a digest-pinned local NVIDIA TensorRT 26.06 image with network
disabled, read-only root filesystem, read-only binary/ONNX mounts, and only the
new engine directory writable. Sidecars are hardlinks to existing files. The
build completed in 108.294 seconds and reported 9,217 MiB peak TRT GPU allocator
usage. The temporary `/tmp` filesystem was limited to 256 MiB.

## Runtime implementation

- `cpp/runtime/config/llmEngineConfig.{h,cpp}` adds
  `prefersAuxiliaryPackedPrefillProfile(batch,maxRowTokens)`. It rejects invalid
  or incompatible shapes and prefers a compatible auxiliary profile only when
  its maximum token carrier is smaller than the primary profile's carrier.
- `cpp/runtime/scheduling/independentPhaseCoordinator.cpp` selects the profile
  after the logical row lengths are known. Text rows use the narrower profile
  when available; larger rows use the primary profile. External-producer rows
  retain their dedicated executor and use the auxiliary profile when compatible,
  otherwise the primary profile. This permits P512 vision chunks despite a
  narrower auxiliary profile.
- The selected profile's dims recipe also supplies the profile-local packed
  attention-mask chunk limit. Profile index is already part of the graph shape
  and executor binding hash.
- Text execution with no auxiliary executor, including shared-context mode,
  retains the primary profile. Engines without an auxiliary profile are unchanged.

This is shape-capability routing, not a workload-name policy. Runtime fixed
chunk remains an explicit opt-in; the default stays 128. E/P serialization in
tiered mode and P/D independence are unchanged. Stable KV pages and ownership
are shared across profile switches without compaction or KV relocation.

## Initial validation

203 config/scheduler unit tests passed, including invalid shapes, missing profile,
equal/wider auxiliary carriers, and 128/512 profile-local attention-mask limits.
Pre-commit checks passed for all modified source files.

A BS1 request with 263 prompt tokens and 32 greedy output tokens used profile 2
for its 128-token chunks, confirmed by the existing opt-in graph-binding trace.
Its output hash matched both prior standalone engine references exactly:

```text
bdc36081e70e7b17391c9d5a4c28f5ab064bdf1c77acdd042146906b1832cf01
```

The trace-enabled diagnostic is not used for performance ranking.

## Performance protocol

Same newly built runtime binary; Exact policy; P8/D24; 24 client in-flight limit;
96 KV page bundles; 2,048-token KV capacity; fresh process for every repetition;
three repetitions per cell. Text calibration is the retained 35-request generic
text trace. VLM calibration is the retained 49-request generic E/P/D trace.
Calibration is held constant within each workload comparison, not between text
and vision workloads. Performance runs disable binding tracing.

Compared cells:

1. standalone P128 engine, runtime chunk 128;
2. profiled P512/P128 engine, runtime chunk 128;
3. profiled P512/P128 engine, runtime chunk 512.

The workloads are retained real-request HTTP `long-prefill` and `multi-image`
traces. VLM runs all enable the same tiered E/P arena. Every raw command,
artifact identity, summary path, and repeat count is retained in the campaign
manifest. Existing vLLM numbers are reused only for the unchanged request/client
contract; they are not same-engine profile ablations.

The HTTP aggregate reports the median across fresh-process repetitions for each
run-level metric, including fields named `*_mean_of_run_means_ms`. A latency mean
in the tables below is therefore the median of three per-run request means, not
a pooled mean over all requests.

## Text results: three fresh process runs per cell

| Engine / runtime chunk | Tok/s | TTFT mean / p95 (ms) | TPOT mean / p95 (ms) | E2E mean / p95 (ms) | Peak MiB |
|---|---:|---:|---:|---:|---:|
| Standalone P128 / 128 | 445.13 | 2,155.75 / 3,168.79 | 22.52 / 25.12 | 4,043.92 / 5,662.53 | 8,585 |
| Profiled P512+P128 / 128 | 447.41 | 2,147.28 / 3,157.42 | 22.32 / 25.40 | 4,008.55 / 5,678.67 | 9,223 |
| Profiled P512+P128 / 512 | 490.24 | 1,845.38 / 2,668.32 | 21.59 / 24.22 | 3,643.93 / 5,211.03 | 9,223 |

All three cells generated 5,440/5,440 output tokens per repeat. At runtime chunk
128, throughput changes +0.51% and every mean/p95 latency changes by less than
3%. This supports narrow-profile performance preservation on long-prefill;
the small positive difference is not evidence of a meaningful speedup.

At runtime chunk 512, throughput improves 10.13% over standalone P128, TTFT mean
falls 14.40%, TTFT p95 falls 15.79%, TPOT mean falls 4.13%, TPOT p95 falls 3.56%,
E2E mean falls 9.89%, and E2E p95 falls 7.97%. Peak memory still rises 638 MiB.

## VLM capability confound found and corrected

The first profiled-engine/chunk128 VLM run completed generic calibration but
failed during multi-image measurement with the deterministic guard:

```text
packed prefill token carrier exceeds the configured profile limit
```

This is a composition-root capability error, not evidence of a CUDA OOB or a KV
allocator failure. `llm_phase_context_smoke.cpp` disabled chunked vision prefill
whenever `hasVisionPrefillProfile()` was true. That condition assumed the optional
profile was a wider whole-prompt vision profile. In this experiment profile 2 is
instead a narrower P128 profile, so a complete multi-image prompt was incorrectly
submitted without chunking and exceeded even the primary P512 row limit.

The correction enables chunked vision prefill when the auxiliary row limit is
no greater than the primary row limit. Existing engines without an auxiliary
profile keep their previous behavior; existing wider whole-prompt vision
profiles also keep their previous behavior. The production `PhaseServingRuntime`
already derives chunking from its explicit serving capability flag rather than
this composition-root shortcut.

The failed run remains a diagnostic, not a latency/throughput datapoint. Text
results use the pre-correction binary (preserved under
`.local/artifacts/binaries/profiled-prefill-20260913/`). VLM cells are rerun with
one common corrected binary, including a fresh standalone P128 control.


## VLM results after the capability correction

| Engine / runtime chunk | Tok/s | TTFT mean / p95 (ms) | TPOT mean / p95 (ms) | E2E mean / p95 (ms) | Peak MiB |
|---|---:|---:|---:|---:|---:|
| Standalone P128 / 128 | 244.34 | 950.57 / 1585.77 | 9.88 / 14.70 | 1256.96 / 1852.38 | 9,199 |
| Profiled P512+P128 / 128 | 244.84 | 936.15 / 1576.92 | 9.89 / 14.82 | 1242.90 / 1846.57 | 9,705 |
| Profiled P512+P128 / 512 | 220.50 | 1078.62 / 1892.17 | 12.62 / 16.88 | 1468.99 / 2142.85 | 9,707 |

Each cell completed three fresh runs and generated 640/640 tokens per run.
Compared with the common-binary standalone P128 control, profiled/chunk128
changes throughput +0.20%, TTFT mean -1.52%, TTFT p95 -0.56%, TPOT mean +0.11%,
TPOT p95 +0.82%, E2E mean -1.12%, and E2E p95 -0.31%. All median metrics are
inside the 3% preservation gate on this trace. This is parity, not a demonstrated
speedup.

Profiled/chunk512 regresses throughput 9.76%, TTFT mean 13.47%, TTFT p95 19.32%,
TPOT mean 27.65%, TPOT p95 14.81%, E2E mean 16.87%, and E2E p95 15.68%. Peak
memory differs by only 2 MiB from profiled/chunk128, so allocation size/OOM is
not an explanation for this within-engine performance regression.

The 128- and 512-chunk VLM peaks leave 535 and 533 MiB against the 10,240 MiB
device total. This satisfies the 512 MiB characterization headroom target in
these runs, but with very little margin. Fully independent E/P workspace
coexistence is not restored: all VLM cells still use tiered E/P serialization.

## Dispatch and formation observations

The table below excludes calibration. Counts are the differences between the
executor graph-cache `misses` counters at measurement start and shutdown.
Graph hits/captures were zero, so these are eager P/D execution counts, not
unique-shape counts or individual CUDA kernel counts. P stats aggregate primary
and auxiliary executors.

| Cell | P dispatches per repeat | D dispatches per repeat | Tok/s range |
|---|---|---|---|
| p128-c128-text | 334 / 354 / 324 | 568 / 570 / 580 | 444.68–448.59 |
| profiled-c128-text | 317 / 366 / 325 | 580 / 574 / 584 | 439.05–448.82 |
| profiled-c512-text | 98 / 101 / 98 | 541 / 542 / 541 | 489.76–490.77 |
| p128-c128-vlm-fixed | 28 / 32 / 28 | 182 / 232 / 199 | 222.76–257.79 |
| profiled-c128-vlm-fixed | 27 / 30 / 27 | 183 / 215 / 196 | 232.27–256.65 |
| profiled-c512-vlm-fixed | 14 / 14 / 17 | 253 / 283 / 245 | 206.77–223.07 |

Long-prefill median P executions fall 334→98 (-70.66%) with chunk512; median D
executions fall 570→541. On multi-image, median P executions fall 28→14 but D
executions rise 199→253 (+27.14%). With 620 post-first-token decode row steps,
mean useful D fill falls from 3.12 to 2.45 rows/dispatch. These observations
support a trajectory/cohort explanation for the VLM loss; they do not isolate
GPU interference from host/completion gaps. No Nsight characterization was run
in this campaign.

## Calibration and statistical limits

The predictor remains Exact; contextual RLS policy authority is disabled. No
trace-derived calibration or workload-specific rule was added. All repeats
use the same retained generic trace and request count within a workload.

- Text chunk128 repeats converge with four required exact cost keys; text
  chunk512 converges with one required key.
- VLM chunk128 repeats converge with three required exact cost keys.
- VLM chunk512 has zero required/calibrated keys and reports
  `calibration_converged=false` in all three repeats. Safe probes occur, but
  those observations do not satisfy its calibration contract. This cell is
  diagnostic only and cannot be promoted on its performance result.

The VLM repeats vary substantially: standalone P128 spans 222.76–257.79 tok/s,
profiled/chunk128 spans 232.27–256.65, and profiled/chunk512 spans 206.77–223.07.
An earlier pre-correction P128 control also had a different median despite the
correction being inactive for that engine. Consequently the small parity gains
must not be advertised as speedups. These are three-run medians, not confidence
intervals; randomized/paired repeat order with at least five repetitions is
needed before a promotion claim.

## Frozen vLLM context

The retained selected seq24/KV480/P4096/sparse-graph result from note 297 is
reused; its workload and output contract did not change. It is not a fresh
repeat or a TensorRT profile ablation.

| Workload | Runtime | Tok/s | TTFT mean / p95 (ms) | TPOT mean / p95 (ms) | E2E mean / p95 (ms) |
|---|---|---:|---:|---:|---:|
| long-prefill | Profiled / chunk512 | 490.24 | 1,845.38 / 2,668.32 | 21.59 / 24.22 | 3,643.93 / 5,211.03 |
| long-prefill | Frozen vLLM | 500.26 | 543.92 / 1,442.01 | 36.37 / 44.60 | 3,554.56 / 5,866.97 |
| multi-image | Profiled / chunk128 | 244.84 | 936.15 / 1,576.92 | 9.89 / 14.82 | 1,242.90 / 1,846.57 |
| multi-image | Frozen vLLM | 381.34 | 188.86 / 227.16 | 29.70 / 35.54 | 1,109.61 / 1,300.30 |

The text chunk512 point remains 2.01% behind vLLM throughput, with better TPOT
and E2E p95 but worse TTFT and mean E2E. The parity-preserving VLM chunk128 point
remains 35.79% behind vLLM throughput. This campaign does not establish a vLLM
win across metrics or across 12 workloads.

## Final correctness and next decision

The corrected BS1/chunk512 request used profile 0 with a `[1,263,hidden]` carrier
in the measurement epoch and reproduced the same 32-token greedy hash quoted
above. The chunk128 diagnostic used profile 2 for its final seven-token chunk.
Thus both profile selections have controlled exact-output evidence, although
batched cross-policy exact identity remains the pre-existing separate gate.

All successful VLM repeats retained direct vision output with zero vision D2D
operations/bytes. The data needed to audit dispatch counts, calibration, token
hashes, and copy behavior is retained in the campaign manifest and summaries.

The result is a useful common-engine capability substrate, not a new default:

1. Keep the existing standalone P128 engine and chunk128 as the serving default.
2. Retain profiled/chunk128 as a small-shape performance-preserving capability
   candidate; verify the remaining workload suite before replacing an engine.
3. Retain profiled/chunk512 as a long-prefill diagnostic option. Do not use
   workload names to select it in the runtime.
4. Fix generic P512 VLM calibration coverage before attributing any policy gain
   or promoting that configuration.
5. Investigate measured D fragmentation/continuity with longer chunks rather
   than assuming fewer P launches necessarily improve VLM serving.
6. Separately evaluate whether small-profile execution can borrow a smaller
   arena without invalidating profile switches, outstanding GPU consumers, or
   captured graphs. This is the remaining memory/frontier problem, not a KV
   pool reduction.
