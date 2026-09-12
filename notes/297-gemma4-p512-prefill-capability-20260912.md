# Gemma 4 P512 packed-prefill capability

## Scope

This campaign evaluates a wider packed-prefill engine contract for the local
Gemma 4 E2B INT4-AWQ runtime. It changes the maximum logical prefill row from
128 to 512 tokens while preserving the existing P8/D24 batch limits, 2,048-token
KV capacity, 96-page indexed KV pool, and Exact scheduling policy.

The goal is capability characterization, not promotion of a new global default.
The retained P128 engine remains the default serving contract.

## Implementation

The composition root previously hard-coded a 128-token scheduler chunk even
when the engine advertised a wider packed-prefill profile. It now:

- defaults to `min(128, engineChunkLimit)`, preserving every existing engine;
- accepts `TRT_EDGELLM_FIXED_PREFILL_CHUNK` as an opt-in runtime chunk;
- rejects non-positive values and values above the engine contract;
- keeps the overlap frontier at 128 tokens unless
  `TRT_EDGELLM_MAX_OVERLAP_PREFILL_TOKENS` explicitly changes it.

This separates two controls:

```text
engine max chunk       shape capability established at build time
runtime fixed chunk    selected serial prefill progress per dispatch
overlap prefill limit  independently bounded interference frontier
```

The export/build artifacts are:

```text
.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/
├── onnx-int4-awq-packed-p512/llm
└── engine-packed-p8x512-d24-kv2048-p96
```

All 35 attention nodes advertise packed prefill, a 512-token maximum chunk,
and profile-local packed-prefill limits. ONNX checker validation passed. The
engine reports P8/D24, `maxPackedPrefillChunk=512`, KV capacity 2,048, and 96
KV page bundles.

The P512 prefill execution context requires 734,580,224 bytes of activation
workspace, compared with 183,647,744 bytes for P128. The decode context remains
28,401,664 bytes. The model weights and external sidecars are byte-identical to
the P128 artifacts and are hard-linked instead of duplicated.

## Memory feasibility

A fully independent E/P/D VLM startup failed because the P512 prefill workspace
and the 313,528,320-byte vision workspace could not coexist with the retained
model, KV, and runtime allocations on the 10 GiB RTX 3080.

The existing tiered context-memory mode makes the P and E contexts share a
734,580,224-byte arena:

```text
fully independent
  P workspace 734.6 MB + E workspace 313.5 MB -> startup OOM

tiered arena
  max(P workspace, E workspace) = 734.6 MB -> startup succeeds
  E and P serialize; E+D and P+D remain possible
```

The P512 tiered VLM runs peaked at 9,735 MiB. This leaves about 505 MiB by the
10,240 MiB device total, narrowly below the earlier 512 MiB desired headroom.
The result is feasible for characterization but not yet a comfortable
production memory contract.

## Text-only same-engine chunk sweep

The same P512 engine was run with the visual engine disabled. Only the runtime
chunk changed, so TensorRT tactics and workspace are identical.

| Chunk | Generated tok/s | TTFT mean (ms) | TTFT p95 (ms) | TPOT mean (ms) | E2E mean (ms) | E2E p95 (ms) | P prepares |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 440.27 | 2,182.92 | 3,199.24 | 22.77 | 4,088.34 | 5,757.86 | 528 |
| 256 | 474.14 | 1,951.26 | 2,771.78 | 21.94 | 3,785.05 | 5,377.71 | 336 |
| 512 | 496.79 | 1,817.27 | 2,620.59 | 21.29 | 3,594.57 | 5,218.96 | 210 |

P512 versus P128 on this same engine improves generated-token throughput by
12.84%. The original P128 engine produced 436.06 tok/s, so the end-to-end
cross-engine improvement is 13.93%. The longer chunk also reduces TTFT mean by
16.84%, TPOT mean by 7.61%, and E2E mean by 12.69% relative to the original
P128-engine run.

These text results show a real long-prefill opportunity: fewer TensorRT
dispatches dominate the longer individual P action when no vision producer is
sharing the critical path.

## Multi-image tiered-arena sweep

The fair VLM comparison uses the same Exact policy, trace, warmup, E/P shared
arena mode, direct vision output, P8/D24 limits, and three fresh process runs.
Reported values are run medians.

| Engine / chunk | Generated tok/s | TTFT mean (ms) | TTFT p95 (ms) | TPOT mean (ms) | TPOT p95 (ms) | E2E mean (ms) | E2E p95 (ms) | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P128 / 128 | 239.39 | 945.42 | 1,641.01 | 10.51 | 14.68 | 1,271.30 | 1,913.69 | 9,199 |
| P512 / 128 | 230.90 | 970.91 | 1,693.26 | 10.93 | 14.68 | 1,309.87 | 2,011.84 | 9,733 |
| P512 / 256 | 218.40 | 1,104.62 | 1,918.41 | 12.22 | 14.60 | 1,483.44 | 2,170.86 | 9,735 |
| P512 / 512 | 234.09 | 896.26 | 1,719.99 | 11.50 | 16.79 | 1,220.87 | 1,981.50 | 9,735 |

Two effects are separated by these controls.

1. With chunk 128 fixed, widening the TensorRT profile reduces throughput by
   3.54% and increases peak memory by 534 MiB. This is an engine tactic/profile
   and workspace cost, not a KV-cache change.
2. Within the P512 engine, selecting chunk 512 instead of 128 recovers 1.38%
   throughput and improves mean TTFT/E2E, but worsens TPOT and tail latency.

Compared with the P128/P128 control, P512/P512 changes:

- throughput: -2.21%;
- TTFT mean: -5.20%, but TTFT p95: +4.81%;
- TPOT mean: +9.41% and TPOT p95: +14.40%;
- E2E mean: -3.97%, but E2E p95: +3.54%.

The P256 point is not a useful compromise on this trace. It is 8.77% slower
than P128 and regresses most request-level latency metrics.

## Correctness and memory-copy observations

Every retained run generated the complete requested token count. Multi-image
runs produced 640/640 tokens and long-prefill runs produced 5,440/5,440 tokens.
The direct vision-output path remained active:

```text
direct_output_bytes = 30,277,632
vision D2D operations = 0
vision D2D bytes = 0
action fidelity violations = 0
```

Fresh-process aggregate token hashes vary even for the unchanged P128 path.
The P512 experiments reproduced hashes seen in P128 runs, so aggregate hash
variation remains the pre-existing scheduling/FP16 ordering issue rather than
evidence of a new P512 ownership error.

A controlled BS1 request with a 263-token prompt and 32-token greedy decode was
then executed once through each engine. P128 required three prefill turns while
P512 completed prefill in one turn. Both engines generated all 32 tokens with
the identical aggregate token hash:

```text
bdc36081e70e7b17391c9d5a4c28f5ab064bdf1c77acdd042146906b1832cf01
```

In that controlled run P512 reduced TTFT from 56.38 to 34.81 ms and E2E from
268.78 to 246.28 ms. This passes the deterministic BS1 cross-engine greedy
identity gate.

## Decision

Do not replace P128 with P512 globally.

- Keep P128 as the VLM and general mixed-serving default.
- Retain P512 as an opt-in engine capability for long-prefill characterization.
- Do not enlarge the P+D overlap frontier beyond 128 without direct cost
  evidence; the current implementation deliberately keeps it separate.
- A future profile-free selector should choose among engine-supported chunk
  shapes from current prefill backlog, decode service pressure, and measured
  phase cost. It must not use a workload name such as `text-only` or `VLM`.
- Before such a selector is activated, build a profile-local engine whose small
  P128 tactic does not regress when P512 support is added. The present wide
  profile loses 3.54% on the same P128 runtime shape and consumes 421 MiB more
  prefill workspace.

## Validation

- TensorRT engine build: pass.
- ONNX checker and 35-node packed-prefill contract audit: pass.
- P512 text-only long-prefill inference: pass.
- P512 tiered VLM multi-image inference: pass.
- Direct vision output and zero vision-output D2D: pass.
- Scheduler unit tests: 151/151 pass.
- Fully independent P512 E/P/D VLM startup: expected OOM under the current
  10 GiB memory contract.
- Deterministic BS1 cross-engine greedy identity: pass for a 263-token prompt
  and 32 generated tokens.

Retained results are indexed under:

```text
.local/results/gemma4-packed-prefill-g4-20260912/
├── p512-active-smoke
├── p128-tiered-vlm-3x
├── p512engine-p128chunk-tiered-vlm-3x
├── p256-tiered-vlm-3x
├── p512-tiered-vlm-3x
├── bs1-p128-correctness
└── bs1-p512-correctness
```
