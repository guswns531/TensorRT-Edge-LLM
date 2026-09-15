# Dynamic prefill transition screen

## Outcome

The P512/P128 multi-profile engine was rebuilt from the retained ONNX and the existing ready-state chunk mechanism
was exposed to the real HTTP harness. Eight successful cells compare fixed P128, fixed P512, a D-ready P128 cap,
and a full-P8-cohort exception on long-prefill and multi-image. Every successful cell ran three fresh processes and
generated the requested output count.

No dynamic candidate is promoted. Fixed P512 is best for long-prefill, while a D-ready P128 cap is best for
multi-image. Applying the latter to all prefills destroys long-prefill progress, and a queue-count exception loses
part of the multi-image benefit. A short queue-state rule therefore cannot replace transition-aware chunk selection.
The retained P128 graph/lifetime configuration in note 310 remains the default full-12 champion.

## Implementation

- `TRT_EDGELLM_DECODE_ACTIVE_PREFILL_CHUNK` exposes the existing scheduler mechanism that bounds a P turn only
  while D work is ready.
- `TRT_EDGELLM_LARGE_PREFILL_CHUNK_QUEUE_THRESHOLD` exposes the existing full-backlog bypass for controlled tests.
- The benchmark runner now accepts the engine/build roots, max chunk, token budget, decode-active cap, queue
  threshold, graph mode, encoded-admission contract, and tiered E/P workspace mode without editing source.
- `build_gemma_profiled_prefill_engine.sh` reproducibly rebuilds the P512 primary / P128 auxiliary / D24 engine
  from retained ONNX using the digest-pinned TensorRT 26.06 image.
- A producer-class-specific cap was prototyped and measured. It was removed because it did not reproduce the
  all-prefill D-ready result and added policy surface without a demonstrated benefit.

All options are off by default. No workload name enters the scheduler.

## Engine and memory result

The rebuilt engine has SHA256
`6437ec2a11d702eefb4df099e2c85962dbfde0dcbf69570c8aa865f847ef2bf2`. TensorRT reported the same activation
requirements as the earlier build: 734,580,224 bytes for P512, 28,401,664 for D24, and 183,647,744 for the P128
auxiliary profile. Engine build peak was 9,217 MiB.

P512 plus independent vision context and calibration CUDA graphs did not fit on the 10 GiB RTX 3080. The process
completed generic calibration, then failed while allocating the vision context. Eager VLM tests therefore use the
documented tiered E/P workspace: TensorRT contexts remain distinct, but E/P cannot execute concurrently. Their peak
was 9,809--9,813 MiB. Text-only peak was 9,323 MiB. This is a workspace/graph frontier, not a KV-cache regression.

## Three-run results

Latency values are medians of run-level request statistics in milliseconds.

### Long-prefill

| Policy | tok/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | Peak MiB |
|---|---:|---:|---:|---:|---:|
| fixed P128 | 404.94 | 2569.3 / 3716.6 | 22.34 / 27.58 | 4490.5 / 6352.2 | 9323 |
| fixed P512 | **470.62** | **1939.4 / 2782.3** | 22.24 / **24.66** | **3805.8 / 5642.9** | 9323 |
| D-ready P128 | 252.78 | 5753.6 / 8732.2 | **11.10 / 17.81** | 6647.5 / 9746.0 | 9323 |
| P8-cohort exception | 445.91 | 2129.7 / 2950.6 | 22.66 / 26.05 | 4021.6 / 5696.3 | 9323 |

Relative to fixed P128, fixed P512 raises throughput 16.22%, reduces TTFT mean/p95 24.51%/25.14%, and reduces E2E
mean/p95 15.25%/11.17%. The unconditional D-ready cap instead lowers throughput 37.58% and more than doubles TTFT,
even though it improves TPOT. It protects resident decode so strongly that new long prompts stop making useful
first-token progress.

### Multi-image

| Policy | tok/s | TTFT mean / p95 | TPOT mean / p95 | E2E mean / p95 | Peak MiB |
|---|---:|---:|---:|---:|---:|
| fixed P128 | 374.87 | 473.6 / 759.3 | **24.89** / 39.87 | 1245.2 / 1497.1 | 9813 |
| fixed P512 | 408.54 | 293.5 / 466.2 | 26.38 / 36.67 | 1110.8 / 1379.3 | 9809 |
| D-ready P128 | **418.41** | **290.6 / 342.9** | 24.92 / **35.06** | **1063.1 / 1334.3** | 9809 |
| P8-cohort exception | 386.20 | 359.9 / 546.0 | 25.77 / 37.25 | 1171.5 / 1401.6 | 9811 |

Relative to fixed P128, the D-ready cap raises throughput 11.61%, reduces TTFT mean/p95 38.64%/54.84%, and reduces
E2E mean/p95 14.63%/10.88%. It also beats the unchanged vLLM reference throughput by 9.72%, but vLLM still has
lower TTFT (188.9/227.2 ms). This targeted eager/tiered result must not be substituted for the full-12 graph result.

## Why the simple policies fail

```text
fixed P512
  + amortizes long prompt launches
  - can delay or fragment an already-forming D cohort

D-ready P128
  + shortens interference and preserves VLM D formation
  - reacts to the first D row as if every remaining P row were low value

P8-cohort exception
  + restores bulk text drain
  - request count is not the same as transition value or future D cost
```

The producer-specific prototype also exposed a learning confound. Generic calibration executed a different chunk
sequence, which changed contextual RLS evidence and later action choices. Identical serving-ready state is not enough
for a fair policy comparison when the physical model was calibrated under a different chunk policy.

## Architectural conclusion

Chunk length must be an action dimension in the global selector rather than a queue-local threshold. Each candidate
must use a shared physical calibration contract and compare:

```text
measured current P cost
+ protected D completion/interference
+ deterministic next-ready cohort formation
+ workspace/graph feasibility
```

The bounded candidate set can remain small: P128, P512, D, P128+D, and WAIT when a concrete completion event exists.
Formation and ownership transitions should remain deterministic; RLS should predict physical cost, not memorize a
workload-specific chunk choice. Unknown P512 overlap remains serial until calibrated safely.

## Promotion gates

1. Capture P128 and P512 physical evidence under one policy-neutral generic calibration sequence.
2. Put chunk length in global candidate identity and evaluate the next D-ready boundary before dispatch.
3. Restore graph feasibility by sharing or leasing the P512 activation arena instead of adding another permanent
   workspace allocation.
4. Re-run long-prefill, bimodal, mixed, vision-heavy, and multi-image first. A candidate must preserve the note 310
   full-12 configuration within 3% before a full gate.
5. Repeat the winning Current and unchanged vLLM comparator at least five times on the remaining loss workloads.

Artifacts are under `.local/results/gemma4-dynamic-prefill-transition-20260915/`.
The run-002/run-003 raw gateway logs were removed after analysis (351,364,570 bytes); all summaries, aggregates,
derived activity data, run-001 raw logs, and OOM evidence remain.

The final cleaned source was rebuilt after removing the rejected producer-specific prototype. The related scheduler,
three-phase, memory, contextual-model, and global-selector suite passes 246/246; shell syntax, Python compilation,
JSON validation, and `git diff --check` also pass.
