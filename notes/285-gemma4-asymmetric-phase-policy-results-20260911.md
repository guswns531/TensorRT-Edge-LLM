# Gemma 4 E2B AWQ asymmetric phase-policy results

Date: 2026-09-11

## Outcome

Gemma 4 E2B now runs real text and VLM requests through an asymmetric phase engine on the RTX 3080. The engine has
four stable KV owners, a prefill profile capped at two rows, and a decode profile capped at four rows. The same
binary, engine, request files, graph behavior, and memory limits were used for V0 through V3; only the phase policy
changed.

The main result is workload-dependent but not workload-labelled policy behavior:

- The short text trace is effectively at parity across V0--V3. V3 changes mean TTFT by -1.96% versus V0, but its
  E2E p95 is 4.10% worse.
- On the mixed text/VLM trace, V3 improves token throughput by 12.75%, mean TTFT by 16.35%, mean TPOT by 9.53%,
  and mean E2E latency by 11.63% versus V0.
- V3 reduces mixed-trace decode dispatches from 160 to 130 while E and P dispatch counts remain unchanged. The gain
  is primarily better decode cohort preservation rather than a large amount of GPU overlap.
- All 12 measured outputs per workload, across four policies and three repeats, are exact string matches.

This is a validation campaign for a new model and a small engine, not a replacement for the retained 12-workload
Cosmos campaign. A same-model vLLM load check was attempted, but vLLM 0.27.1 rejects Gemma 4's heterogeneous
per-layer head dimensions before loading weights. Larger Gemma profiles remain future work.

## Artifact contract

| Item | Value |
|---|---|
| Checkpoint | `Chunity/gemma-4-E2B-it-AWQ-4bit` |
| Revision | `79fb33d4c2d52338e9e6de36021d1bba8342db91` |
| Quantization | decoder INT4 AWQ, group size 128, symmetric |
| FP16 state | embedding, LM head, PLE, KV, vision, audio |
| ONNX | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/onnx-int4-awq-p128` |
| Engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-asym-p2-d4-kv2048` |
| Results | `.local/results/gemma4-e2b-awq-asym-p2-d4-20260911` |
| GPU | GeForce RTX 3080, 10,240 MiB |
| Driver / CUDA / TensorRT | 610.43.02 / 13.3 / 11.0.0.114 |

The ONNX directory name retains the historical `p128` suffix, but this Gemma ONNX uses non-packed prefill. The
current packed-prefill kernel requires head dimension 128, while Gemma 4 has heterogeneous 256/512-dimensional
attention heads. The engine therefore uses the non-packed prefill profile and the version-1 INT4 GEMM plugin.

## Engine construction

The LLM engine was built with:

```text
--maxInputLen 1024
--maxKVCacheCapacity 2048
--maxBatchSize 4
--maxPrefillBatchSize 2
--maxDecodeBatchSize 4
--maxKVPoolPages 64
--maxPrefillChunkTokens 128
```

The 64 physical KV pages are exactly four stable slots times sixteen 128-token pages per 2,048-token slot. The
builder reported about 350 MiB of activation memory for the prefill profile and 16.5 MiB for decode. The runtime
created distinct TensorRT execution contexts and workspaces for E, P, and D.

The visual engine already supports two 512-token images through its 1,024-token aggregate profile, so it is reused.
The visual engine and the three large immutable LLM sidecars are hard-linked to the BS1 validation engine after
byte-for-byte comparison. The second engine bundle has a 7.5 GiB apparent size but adds only the new 1.22 GiB LLM
engine and small metadata physically.

## Performance matrix

Each cell is the mean of three fresh-process runs. Every request generates up to 64 tokens. The text trace has eight
text requests. The mixed trace has four text and four single-image requests. These are closed-loop continuous
admission tests: all logical requests are submitted, while stable ownership limits active work to four requests.

### Text

| Policy | Token/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---:|---:|---:|---:|
| V0 Exact | 411.583 | 170.413 / 358.217 | 7.154 / 8.358 | 621.097 / 790.386 |
| V1 Scalar | 410.041 | 172.294 / 362.531 | 7.161 / 8.382 | 623.435 / 795.111 |
| V2 Scalar+Transition | 410.895 | 171.532 / 360.860 | 7.152 / 8.363 | 622.137 / 792.904 |
| V3 Service-scaled Transition | 410.154 | 167.077 / 361.832 | 7.181 / 8.282 | 619.489 / 822.829 |

V3 versus V0 is -0.35% token throughput, -1.96% mean TTFT, +1.01% TTFT p95, +0.38% mean TPOT, -0.92% TPOT
p95, -0.26% mean E2E, and +4.10% E2E p95. This is a parity result with an unresolved text tail regression, not a
V3 win.

### Mixed text/VLM

| Policy | Token/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---:|---:|---:|---:|
| V0 Exact | 286.373 | 274.811 / 494.519 | 9.626 / 11.015 | 870.228 / 1070.639 |
| V1 Scalar | 285.930 | 275.148 / 495.548 | 9.642 / 11.025 | 871.580 / 1072.867 |
| V2 Scalar+Transition | 285.470 | 220.125 / 385.585 | 10.538 / 12.114 | 873.007 / 1076.272 |
| V3 Service-scaled Transition | 322.875 | 229.872 / 467.998 | 8.709 / 10.406 | 769.045 / 1012.429 |

V3 versus V0 is +12.75% token throughput, -16.35% mean TTFT, -5.36% TTFT p95, -9.53% mean TPOT, -5.53%
TPOT p95, -11.63% mean E2E, and -5.44% E2E p95. V2 obtains the best TTFT, but fragments decode service enough
to lose its first-token gain in E2E. V3 gives up part of V2's TTFT improvement to restore decode continuity.

## E/P/D/Copy CUDA-event activity

The production `llm_inference --phaseServing` path can now accept
`TRT_EDGELLM_PHASE_ACTIVITY_PREFIX=<prefix>`. It attaches the existing CUDA-event recorder to the actual
`PhaseServingRuntime` and writes interval, 4-bit mask segment, and summary CSV files after the serving runtime drains.
This instrumentation is opt-in because it creates two timing events per interval. Activity runs are diagnostic and
are not used for the three-repeat performance headline.

The mask remains:

```text
E = 0001
P = 0010
D = 0100
C = 1000
```

Final same-binary diagnostic results:

| Workload / policy | Idle | E+P | E+D | P+D | E+P+D | E dispatch | P dispatch | D dispatch | C dispatch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Text V0 | 0.224% | 0% | 0% | 1.039% | 0% | 0 | 8 | 130 | 0 |
| Text V1 | 0.222% | 0% | 0% | 1.035% | 0% | 0 | 8 | 130 | 0 |
| Text V2 | 0.221% | 0% | 0% | 1.038% | 0% | 0 | 8 | 130 | 0 |
| Text V3 | 0.261% | 0% | 0% | 1.035% | 0% | 0 | 8 | 130 | 0 |
| Mixed V0 | 0.538% | 0% | 0.017% | 3.563% | 0% | 3 | 8 | 160 | 3 |
| Mixed V1 | 0.543% | 0% | 0.018% | 3.540% | 0% | 3 | 8 | 160 | 3 |
| Mixed V2 | 0.501% | 0% | 0.057% | 3.590% | 0% | 3 | 8 | 160 | 3 |
| Mixed V3 | 0.591% | 1.770% | 0.019% | 3.970% | 0.016% | 3 | 8 | 130 | 3 |

The percentages are active-span CUDA-event occupancy, not SM utilization. They show when phase streams have queued
GPU work. The near-zero idle means there is little empty-device time to recover. V3's large mixed gain instead
correlates with 18.75% fewer decode dispatches, while the amount of explicit phase overlap remains modest. Copy duty
is below 0.01% because only the encoder-output transfer is classified as C; PLE gathering and other phase-local
copies remain inside their owning P/D activity interval.

## Memory

The opt-in profile path now records peak process GPU and CPU allocations in the phase-serving output JSON. The
final mixed V3 diagnostic measured:

| Metric | Value |
|---|---:|
| Peak GPU allocation | 9,268 MiB |
| RTX 3080 headroom | 972 MiB |
| Peak CPU allocation | 5.36 GiB |
| P workspace | 350.3 MiB |
| D workspace | 16.5 MiB |
| E workspace | 270.0 MiB |

This passes the original 512 MiB headroom gate. It is not enough headroom to double the engine profiles without a
new memory plan. The KV pool is not the dominant new cost: increasing from 16 to 64 pages is required for four
stable owners, while the PLE table, embedding, model weights, TensorRT engine state, and three phase workspaces
dominate residency.

## vLLM compatibility control

The retained `vllm/vllm-openai:v0.27.1` image was started with the same local checkpoint, FP16 activation dtype,
2,048-token context, four maximum sequences, and 90% GPU-memory utilization. It failed before model-weight loading:

```text
AmbiguousGlobalPerLayerAttributeError: 'head_dim' is a per-layer attribute and may vary across layers
```

vLLM's model-architecture converter accesses one global `head_dim`; Gemma 4 exposes heterogeneous d256/d512 layer
configuration. TensorRT-Edge-LLM's engine config and attention path already carry 35 per-layer KV/head descriptors.
Consequently, there is no honest same-checkpoint vLLM number for this campaign. Substituting another model,
quantization, or homogeneous-head configuration would change the contract and is intentionally not reported.

## Validation

- LLM engine build: pass.
- Text V0/V1/V2/V3, three repeats each: pass.
- Mixed V0/V1/V2/V3, three repeats each: pass.
- Policy output identity: 12/12 for text and 12/12 for mixed.
- Opt-in E/P/D/Copy activity CSV generation: pass.
- Phase-serving peak-memory JSON fields: pass.
- vLLM 0.27.1 same-checkpoint load: unsupported heterogeneous `head_dim` contract.
- `unitTestRuntime`: 649 passed, 2 skipped.

## Interpretation and next steps

This campaign establishes that the generic V3 policy transfers to Gemma 4 without a Gemma-specific workload label
or per-trace rule and can exploit a real E/P/D opportunity. It does not establish universal superiority: the text
tail remains worse than V0 and the profile is deliberately small.

Next steps, in order:

1. Add a text-tail protection that is derived from observed service scale rather than a Gemma-specific constant,
   then rerun these exact two traces.
2. Recheck a future vLLM release only after its model-architecture converter supports heterogeneous per-layer
   `head_dim`; do not patch the config to pretend Gemma 4 is homogeneous.
3. Build a controlled E1/E2 plus P2/D4 opportunity trace and measure requested versus realized overlap.
4. Expand the request count and output length while keeping the four-slot engine fixed, so steady-state D4 rather
   than startup dominates.
5. Only after the 972 MiB budget is rebalanced, test P4/D8 or more KV pages. Packed prefill remains unavailable until
   the d256/d512 Gemma attention path is implemented.
