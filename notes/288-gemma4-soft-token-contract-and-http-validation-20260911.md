# Gemma 4 soft-token contract and HTTP validation

Date: 2026-09-11

## Outcome

The Gemma 4 comparison had a hidden image-token contract mismatch. The first TensorRT visual engine was built with
the sample default of 512 soft tokens per image, while Gemma 4 declares `vision_config.default_output_length=280`
and vLLM follows that model value. On the same four-image mixed trace, TensorRT therefore processed 2,101 prompt
tokens while vLLM processed 1,187. This was not a KV-cache or scheduler-policy difference: the visual builder had
silently inherited a generic sample default that was inappropriate for Gemma 4.

The builder now resolves the per-image limit from Gemma 4's exported model config unless the user explicitly passes
`--maxImageTokensPerImage`. A corrected E2 visual engine produces 1,193 prompt tokens for the same trace, only six
tokens (0.51%) above vLLM. The remaining differences are chat-template and image-aspect rounding details, not a
second 512-versus-280 mismatch.

The production-style phase HTTP path also exposed two gaps in the standalone phase smoke backend:

1. Gemma 4 PLE bindings were created but not populated on its independent prefill/decode contexts.
2. A dedicated external-prefill context was auto-created only for packed prefill, although dense VLM prefill also
   needs that independent context when the engine has no dedicated vision-prefill optimization profile.

Both are fixed. Real OpenAI-compatible HTTP/SSE requests now pass through the async request adapter, independent
E/P/D contexts, stable KV ownership, vision queue, and V0--V3 selectors with Gemma 4 PLE enabled.

The best current candidate remains V3 Service-scaled Transition. Under an eight-request trace-derived warmup, V3
improves mixed token throughput by 22.37% and mean E2E latency by 23.67% over V0 Exact. Text remains within 1% of
V0. Compared with eager vLLM 0.28, V3 has much higher throughput and much lower TPOT/E2E, but vLLM still has better
TTFT on the text trace and mixed TTFT tail. The comparison is a useful same-checkpoint control, not a claim of full
contract equivalence: vLLM's compiled path OOMs on the 10 GB GPU, its active limit is eight versus four TensorRT KV
owners, and the final prompt-token totals differ by six.

## Corrected visual-token contract

| Runtime/engine | Per-image cap | Aggregate cap | Mixed prompt tokens |
|---|---:|---:|---:|
| Old TensorRT visual engine | 512 | 1,024 | 2,101 |
| Corrected TensorRT E2 engine | 280 | 560 | 1,193 |
| Diagnostic TensorRT E4-capacity engine | 280 | 1,120 | 1,193 |
| vLLM 0.28 eager | model-derived, about 256--266 after aspect handling | dynamic | 1,187 |

The corrected TensorRT and vLLM per-request prompt counts are:

| Request | Class | TensorRT | vLLM | Delta |
|---:|---|---:|---:|---:|
| 0 | vision | 278 | 276 | +2 |
| 1 | text | 17 | 17 | 0 |
| 2 | vision | 285 | 285 | 0 |
| 3 | text | 21 | 20 | +1 |
| 4 | vision | 275 | 274 | +1 |
| 5 | text | 19 | 18 | +1 |
| 6 | vision | 280 | 279 | +1 |
| 7 | text | 18 | 18 | 0 |

The visual builder retains an explicit override. The new resolution order is:

```text
explicit --maxImageTokensPerImage
              │ present
              ▼
       use explicit value
              │ absent
              ▼
model_type == gemma4_vision and vision_config is present?
        ├── yes → default_output_length (280)
        └── no  → legacy sample default (512)
```

This is a model capability contract, not a workload-specific serving heuristic.

## Phase HTTP implementation

The backend now mirrors `PhaseServingRuntime` for Gemma 4:

```text
immutable PLE table, 4.70 GB
             │ shared ownership
      ┌──────┴──────┐
      ▼             ▼
prefill PLE       decode PLE
outputs           outputs
      │             │
P TensorMap      D TensorMap
      │             │
dense text P     autoregressive D
external VLM P
```

Token IDs are staged and the PLE gather is enqueued on the same explicit stream as the consuming TensorRT context.
The corrected context topology is:

```text
one CUDA context
├── E stream → visual TensorRT context
├── P stream → text-prefill TensorRT context
├── P stream → external-prefill TensorRT context
│              (created for dense or packed VLM when no integrated profile exists)
├── D stream → decode TensorRT context
└── C stream → encoder-output/vision-payload copies
```

The text-prefill and external-prefill contexts share the prefill workspace and therefore do not execute
concurrently. Decode retains an independent context/workspace and may overlap where the global action permits it.
Stable indexed KV ownership remains four physical owners with 64 pages. No KV allocation or page geometry changed
in this correction.

## Direct request-level control

The corrected visual engine was first evaluated through the direct phase-serving request path. Values are means of
three fresh-process runs.

| Policy | Token/s | Req/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---:|---:|---:|---:|---:|
| V0 Exact | 319.868 | 4.998 | 228.479 / 433.465 | 9.023 / 9.987 | 796.953 / 989.756 |
| V1 Scalar | 318.026 | 4.969 | 233.204 / 442.849 | 9.022 / 9.982 | 801.572 / 999.322 |
| V2 Scalar+Transition | 317.490 | 4.961 | 209.018 / 394.843 | 9.428 / 10.812 | 803.005 / 1,002.005 |
| V3 Service-scaled Transition | 360.505 | 5.633 | 207.968 / 428.742 | 7.929 / 9.416 | 707.514 / 931.782 |

Relative to the old 512-token visual contract, the corrected V3 improves token throughput from 322.875 to 360.505
token/s (+11.66%) and mean E2E from 769.045 to 707.514 ms (-8.00%). This gain is mostly removed excess vision
prefill work, not a new scheduling-policy optimization.

All direct outputs are exact across the four policies and three repeats. The corrected image descriptions remain
semantic and no request asks for a missing image.

## HTTP/SSE cold-policy matrix

This matrix resets learned execution-cost and policy posterior state after internal shape priming. It does not issue
client-level vision warmup requests. Every value is the median or median-of-run statistic from three fresh backend
processes.

### Text

| Policy | Token/s | Req/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---:|---:|---:|---:|---:|
| V0 Exact | 503.652 | 7.870 | 295.485 / 557.283 | 7.568 / 7.904 | 771.295 / 1,014.454 |
| V1 Scalar | 498.403 | 7.788 | 298.664 / 568.826 | 7.609 / 7.901 | 778.044 / 1,025.453 |
| V2 Scalar+Transition | 501.072 | 7.829 | 300.138 / 567.028 | 7.545 / 7.827 | 775.202 / 1,019.890 |
| V3 Service-scaled Transition | 505.271 | 7.895 | 291.991 / 556.738 | 7.520 / 7.824 | 765.756 / 1,011.728 |

V3 is effectively at text parity with V0: +0.32% token/s and -0.72% mean E2E.

### Mixed text/VLM

| Policy | Token/s | Req/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---:|---:|---:|---:|---:|
| V0 Exact | 321.266 | 5.020 | 567.133 / 1,118.070 | 11.224 / 14.615 | 1,275.833 / 1,592.223 |
| V1 Scalar | 320.920 | 5.014 | 562.864 / 1,120.915 | 11.225 / 14.551 | 1,270.751 / 1,594.281 |
| V2 Scalar+Transition | 320.024 | 5.000 | 567.855 / 1,124.510 | 11.003 / 14.151 | 1,261.038 / 1,598.428 |
| V3 Service-scaled Transition | 429.675 | 6.714 | 376.273 / 716.636 | 8.408 / 9.078 | 909.592 / 1,189.969 |

V3 versus V0 is +33.74% token/s, -33.65% mean TTFT, -25.09% mean TPOT, and -28.71% mean E2E. This is partly a
cold-path robustness result: V0/V1 allow the first vision execution to dominate the measured trace, whereas V3's
service-scaled action ordering hides or avoids much of that exposed delay. It must not be substituted for the
steady-state policy comparison below.

## HTTP/SSE warmup-matched matrix

vLLM's retained results issue eight warmup requests with a 32-token output cap. The second matrix uses the same
client-level warmup count and excludes those requests from measurement. The warmup is trace-derived and therefore
represents a steady-state/upper-bound comparison, not the primary proof of workload-agnostic calibration.

### Text

| Policy | Token/s | Req/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---:|---:|---:|---:|---:|
| V0 Exact | 507.452 | 7.929 | 290.166 / 556.243 | 7.445 / 7.592 | 759.220 / 1,007.392 |
| V1 Scalar | 504.173 | 7.878 | 293.053 / 560.721 | 7.489 / 7.651 | 764.879 / 1,014.110 |
| V2 Scalar+Transition | 504.365 | 7.881 | 291.203 / 559.662 | 7.439 / 7.675 | 764.177 / 1,013.749 |
| V3 Service-scaled Transition | 503.250 | 7.863 | 297.953 / 562.431 | 7.482 / 7.705 | 763.913 / 1,016.287 |

V3 remains within 1% of V0 on throughput and E2E, but V0 is the best text-only point in this small trace. V3 versus
V0 is -0.83% token/s and +0.62% mean E2E latency.

### Mixed text/VLM

| Policy | Token/s | Req/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms |
|---|---:|---:|---:|---:|---:|
| V0 Exact | 356.781 | 5.575 | 494.079 / 959.143 | 10.323 / 13.101 | 1,144.423 / 1,433.660 |
| V1 Scalar | 354.631 | 5.541 | 492.209 / 967.833 | 10.406 / 13.110 | 1,147.793 / 1,442.450 |
| V2 Scalar+Transition | 332.093 | 5.189 | 549.805 / 1,069.114 | 10.999 / 14.263 | 1,247.439 / 1,540.706 |
| V3 Service-scaled Transition | 436.588 | 6.822 | 362.961 / 698.144 | 8.108 / 8.798 | 873.581 / 1,168.698 |

V3 versus V0 is +22.37% token/s, -26.54% mean TTFT, -21.46% mean TPOT, -23.67% mean E2E, and -18.48% E2E p95.
Warmup improves mixed V0/V1 throughput by about 11%, V2 by 3.77%, and V3 by only 1.61%. V3 is therefore both the
best steady-state policy and the least dependent on a vision warmup in this trace.

## Same-checkpoint vLLM 0.28 comparison

The retained vLLM server uses eager execution because the compiled path requests an additional 4.38 GiB during
Inductor autotuning and OOMs even after reducing batch, token range, and KV capacity. vLLM uses eight active
sequences; TensorRT admits all eight HTTP requests but has four stable KV owners and continuously refills them.

| Workload/runtime | Token/s | Req/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
| Text vLLM eager, native 8 | 208.328 | 3.255 | 116.665 / 121.933 | 37.136 / 37.444 | 2,456.415 / 2,457.293 | 7,795 |
| Text TensorRT V3 | 503.250 | 7.863 | 297.953 / 562.431 | 7.482 / 7.705 | 763.913 / 1,016.287 | 9,043 |
| Mixed vLLM eager, native 8 | 182.274 | 2.916 | 339.601 / 451.484 | 37.684 / 38.734 | 2,657.057 / 2,730.633 | 7,859 |
| Mixed TensorRT V3 | 436.588 | 6.822 | 362.961 / 698.144 | 8.108 / 8.798 | 873.581 / 1,168.698 | 9,047 |

TensorRT V3 versus vLLM:

| Workload | Token/s | Req/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 | Memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Text | +141.57% | +141.57% | -155.39% | -361.26% | +79.85% | +79.42% | +68.90% | +58.64% | +1,248 MiB |
| Mixed | +139.52% | +133.91% | -6.88% | -54.63% | +78.49% | +77.29% | +67.12% | +57.20% | +1,188 MiB |

Positive latency percentages mean lower TensorRT latency. The negative TTFT entries therefore mean vLLM is
better. TensorRT's four-owner admission creates a queue before first service but then decodes much faster; vLLM's
eight active sequences expose lower TTFT but substantially higher TPOT. The current optimization target is to
preserve TensorRT's decode/E2E advantage while opening first-token admission, not to trade away the decode engine.

The memory comparison also has to remain explicit. TensorRT spends about 1.2 GiB more because it holds independent
TensorRT contexts/workspaces and phase-local PLE outputs. The KV pool is not the source of this delta: vLLM reserves
160 MiB for KV and TensorRT's 64-page stable pool is also small relative to its 4.70 GB immutable PLE and compiled
engine/context state.

## E2 versus E4-capacity visual profile

The visual engine was also rebuilt with an aggregate 1,120-token profile. The LLM engine remained P2/D4 and the
runtime encoder batch limit is bounded by the prefill profile, so this profile cannot form an actual E4 downstream
cohort. Three V3 runs measured:

| Profile | Token/s | Req/s | TTFT mean / p95 ms | TPOT mean / p95 ms | E2E mean / p95 ms | Peak GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
| E2 aggregate (560) | 360.247 | 5.629 | 208.093 / 428.949 | 7.934 / 9.416 | 707.999 / 932.399 | 9,090 |
| E4 aggregate (1,120) | 360.643 | 5.635 | 208.421 / 429.779 | 7.919 / 9.413 | 707.262 / 932.575 | 9,278 |

The +0.11% throughput is noise-level while residency increases by 188 MiB. The E2 engine is retained; a useful E4
experiment requires a larger LLM prefill/external-prefill profile and a new memory budget, not only a larger visual
input profile.

## Correctness and validation

| Check | Result |
|---|---|
| Corrected visual engine build, 280 per image / 560 total | Pass |
| Gemma default-resolution smoke | Pass; builder reads 280 from config |
| `visual_build` rebuild | Pass |
| `llm_phase_context_smoke` rebuild | Pass |
| PLE + async server + phase queue + three-phase policy unit tests | 207/207 pass |
| Direct mixed V0--V3, three repeats | Exact output identity |
| HTTP text V0--V3, cold and warm, three repeats | Exact token identity |
| HTTP mixed cold V0/V1/V3 | Exact token identity |
| HTTP mixed cold V2 | Two stable greedy branches; 2/3 majority identity |
| HTTP mixed warm V0--V3 | Two stable greedy branches under every policy |
| Invalid slot/profile/OOM | None in retained runs |
| Peak TensorRT HTTP memory | 9,049 MiB, at least 1,191 MiB nominal headroom |

The two mixed hashes differ only in request 0 after a plausible FP16 greedy branch. The other seven requests are
token-identical. Both branches correctly describe the person/dog image. This is not evidence of KV ownership
corruption, but it fails the strict production exact-repeat gate and must remain disclosed. The next determinism
test should bind a canonical row order and compare request-0 logits around the first divergent token.

vLLM's retained HTTP client did not capture server token IDs (`captured_token_ids_per_run=[0,0,0]`), so its reported
empty-trace hash cannot establish cross-framework token identity. Cross-framework correctness is limited to request
success, generated-token count, and semantic output inspection in this campaign.

## Artifacts

| Artifact | Path |
|---|---|
| Original AWQ checkpoint | `.local/artifacts/models/gemma-4-e2b-it-awq/hf` |
| vLLM metadata compatibility view | `.local/artifacts/models/gemma-4-e2b-it-awq-vllm028-compat/hf` |
| Exported ONNX | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/onnx-int4-awq-p128` |
| Retained corrected E2 engine | `.local/artifacts/v0101-forward-port/gemma-4-e2b-it-awq/engine-asym-p2-d4-kv2048-soft280` |
| Direct corrected matrix | `.local/results/gemma4-e2b-awq-soft280-20260911` |
| HTTP cold-policy matrix | `.local/results/gemma4-e2b-awq-soft280-http-20260911/matrix` |
| HTTP warmup-matched matrix | `.local/results/gemma4-e2b-awq-soft280-http-20260911/trace-derived-matrix` |
| Frozen vLLM results | `.local/results/gemma4-e2b-awq-vllm028-equal-http-20260911` |

Engine identities:

```text
LLM engine     360ec2062e99db3fa9ba6da2fed0e884437bc141aee33da4537dce8e51ecbf5a
Visual engine  f09bbd50a34f7ee4b4bc6070ff531075c720a3811f69976d9de4e3b0ba1b5b6f
```

## Next steps

1. Add a generic, workload-independent Gemma calibration trace that primes one text P/D shape and representative
   E1/E2 image geometry, then compare zero-start, generic, and trace-derived V3 without changing policy constants.
2. Diagnose request-0 mixed greedy divergence with canonical row-order telemetry and first-divergent-token logits.
3. Open TTFT capacity without changing KV size: distinguish queued-but-unowned requests from four GPU-resident KV
   owners, and admit E/P preparation earlier while preserving decode ownership.
4. Build an actual P4/D8 Gemma engine only after reducing independent-context memory or moving the immutable PLE
   binding strategy; the E4 visual-only profile is not useful.
5. Implement d256/d512 packed/chunked prefill before claiming parity with the earlier Cosmos P8/D64 scheduler
   frontier. Gemma currently exercises dense P2/D4 and is a model-port validation, not the 12-workload replacement.
6. Re-run vLLM only if the execution contract changes, such as a compiled path that fits, a different quantization,
   or a matched active-sequence/admission experiment. Reuse the frozen result otherwise.
