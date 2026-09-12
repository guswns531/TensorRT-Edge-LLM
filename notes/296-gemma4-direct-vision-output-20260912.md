# Gemma 4 Direct Vision Output

## Scope

This step removes the encoder-output device-to-device copy from the Gemma 4 phase runtime. It does not change the
vision engine, image preprocessing, request DAG, prefill engine, KV layout, or scheduling policy.

The retained comparison uses the packed Gemma 4 E2B AWQ engine with E4/P8/D24, fixed 128-token prefill chunks, the
same generic calibration, and the same 20-request multi-image HTTP trace.

## Implementation

The common `PhaseVisionAdapter` already allocates request-owned output storage and asks each multimodal runner to bind
it before encoder enqueue. Qwen runners implement that contract, but `Gemma4ViTRunner` previously returned the base
class fallback and therefore required a post-encoder copy.

Gemma now implements the same ownership contract:

1. Preserve the active output shape independently from the backing allocation.
2. Release the runner-owned maximum-size output buffer when the phase adapter assumes output ownership.
3. Keep updating the active shape during full and token-length-only preprocessing.
4. Bind the request-owned FP16 GPU tensor directly to the TensorRT `visual_output` binding before each enqueue.
5. Reject deepstack outputs because Gemma 4 E2B does not expose them through this visual engine.

The output lifetime is therefore:

```text
Before

TensorRT E output
       │ runner-owned maximum buffer
       ▼
encoder stream completion event
       ▼
copy stream D2D
       ▼
request-owned vision lease
       ▼
P consumes and releases lease

After

request-owned vision lease
       │ bound as TensorRT E output
       ▼
encoder writes final storage directly
       ▼
encoder completion event
       ▼
P consumes and releases lease
```

The encoder and prefill remain independently enqueued. Cross-stream safety is unchanged: the payload ready event is
recorded after encoder completion on the encoder stream and prefill waits on that event before consuming the lease.

## Validation

### Build and unit tests

- `llm_stream`: built successfully
- `llm_phase_context_smoke`: built successfully
- `unitTestRuntime`: 650 passed, 2 environment-dependent tests skipped

### Physical copy contract

| Mode | Direct-output batches | Direct-output bytes | Encoder D2D operations | Encoder D2D bytes |
|---|---:|---:|---:|---:|
| Previous exact, 1 run | 0 | 0 | 24 | 30,277,632 |
| Direct-output exact, 3 runs | 21 / 21 / 22 | 30,277,632 each | 0 | 0 |
| Direct-output V3, 3 runs | 23 / 24 / 25 | 30,277,632 each | 0 | 0 |

The implementation removes all encoder-output D2D traffic for this trace. Peak memory changed only slightly because
the request-owned output lease existed in both paths and the removed runner buffer is small relative to engine,
context, workspace, and KV allocations.

### Exact-policy performance

The prior exact result is a retained single run; the direct-output result is the median of three fresh runs. It is a
useful implementation gate, not yet a publication-quality confidence interval.

| Metric | Previous copy path | Direct output | Change |
|---|---:|---:|---:|
| Generated token throughput | 207.23 tok/s | 218.44 tok/s | +5.41% |
| TTFT mean | 1,162.58 ms | 983.65 ms | -15.39% |
| TPOT mean | 13.59 ms | 12.48 ms | -8.18% |
| E2E mean | 1,583.89 ms | 1,368.89 ms | -13.57% |
| Peak GPU memory | 9,379 MiB | 9,375 MiB | -4 MiB |

### V3 scheduling sensitivity

Direct-output V3 produced 152.66--203.20 tok/s across the retained three runs, with median 179.84 tok/s. The earlier
single-run V3 result was 194.22 tok/s. Prefill and decode dispatch counts also moved substantially between runs. The
copy removal advances encoder completion and therefore changes the ready boundaries at which V3 forms later cohorts.
This is execution--formation coupling, so the V3 numbers do not isolate copy cost.

The aggregate token-trace hash also varied across independently launched runs under both exact and V3 scheduling.
The old exact run's hash appears among the new direct-output runs, and every run generated all 640 requested tokens.
This does not identify direct output as the source of the pre-existing cross-run ordering/numerical variation, but an
exact per-request replay remains a separate promotion gate.

## Conclusion and next step

Gemma direct output is physically working and beneficial under the policy-stable exact comparison. It removes a
model-specific runtime gap that was absent from Qwen/Cosmos without introducing a workload rule.

The dominant remaining Gemma gap is the E-to-P critical path: the current engine can dispatch at most eight packed
rows of 128 tokens, whereas the selected vLLM configuration admits 4,096 scheduled tokens. The next step is to add
larger compiled packed-prefill quanta and let the existing global selector choose among supported shapes. The first
engine frontier should preserve 128-token row granularity while adding 256/512-token per-row profiles and a bounded
larger total packed-token profile; it must be evaluated for build feasibility, memory, TTFT, and long-prefill/VLM
throughput before becoming the default.

## Retained artifacts

- Previous exact: `.local/results/gemma4-packed-prefill-g4-20260912/sentinel-1x/exact/multi-image`
- Direct-output exact: `.local/results/gemma4-packed-prefill-g4-20260912/direct-output-exact-3x/exact/multi-image`
- Direct-output V3: `.local/results/gemma4-packed-prefill-g4-20260912/direct-output-v3-3x/scalar-transition/multi-image`
- Initial direct-output smoke: `.local/results/gemma4-packed-prefill-g4-20260912/direct-output-smoke/scalar-transition/multi-image`
