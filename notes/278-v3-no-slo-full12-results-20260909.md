# 278. V3 no-SLO full-12 result and promotion decision

## Outcome

V3 `service-scaled-transition` now runs without a configured TTFT or TPOT
target. In that mode it does not synthesize a deadline. It protects forward
progress with immutable, request-local phase service epochs measured in units
of robust isolated GPU service, then applies the existing V2 contextual Scalar
and bounded transition value to the remaining frontier.

The final same-binary, same-engine, generic-calibration full-12 comparison is:

```text
V2 Scalar + Transition
        -> fixed legacy internal time scales and inherited fallback deadlines

V3 Service-Scaled Transition
        -> explicit SLO safety only when supplied
        -> no-SLO service recovery in frozen E/P/D service quanta
        -> overdue service suppresses unknown overlap exploration
        -> V2 Scalar + deterministic transition efficiency selection
```

Against V2, V3 wins request throughput on 9/12 workloads and improves the
12-workload request-throughput geometric mean by **7.26%**. Against the retained
vLLM measurements it wins token throughput on 12/12 workloads, with a geometric
mean gain of **9.21%**. The vLLM result is a frozen comparison, not a fresh run;
the four older text traces used client cap 80 for vLLM and cap 64 for this phase
runtime, as documented in note244.

V3 is promoted as the best profile-free/no-SLO policy candidate. It is not
claimed to dominate V2 on every scalar metric. Without an external utility or
SLO, throughput, first-token progress, and decode continuity form a Pareto
surface rather than one mathematically unique optimum.

## Implementation

### 1. Explicit contract and internal service are separate

Each protected phase carries both fields rather than encoding one as the other:

```text
hasExplicitSlo + absoluteSlackUs
    -> absolute request contract, when one was supplied

readyWaitUs / frozenReferenceUs
    -> internal service age, always available after a reference is established
```

`service-scaled-transition` assigns infinite absolute slack to an unspecified
text TTFT or decode TPOT. Explicit request metadata and explicitly configured
composition-root targets retain their original microsecond semantics.

The retained replay tool exposes `--no-explicit-slo`, which removes only:

- `TRT_EDGELLM_VISION_TTFT_TARGET_MS`;
- `TRT_EDGELLM_VISION_DECODE_TPOT_TARGET_MS`;
- `TRT_EDGELLM_GLOBAL_DECODE_TPOT_TARGET_US`.

It preserves engines, requests, calibration, batching, memory, CUDA graphs,
and every other scheduler mechanism.

### 2. Immutable service epochs

The denominator cannot change while a request waits:

```text
E epoch: E-ready -> encoder progress/cancel
P epoch: P-ready -> real prefill progress/transition/cancel
D epoch: token commit -> next token commit/completion/cancel
```

Runtime exact/covering isolated costs are preferred. Static and visible cold
fallback references preserve bounded startup behavior. A new timing observation
can affect only the next epoch.

Prefill service reference covers the complete remaining prefill milestone, not
only the next 128-token chunk:

```text
P reference = robust chunk service * remaining chunk turns
```

This prevents a long prefill from appearing overdue after only one chunk-scale
wait.

### 3. Mechanism-only candidate preservation

In V3, an inherited pseudo-expiration cannot remove standalone P or D from the
global frontier. Feasibility still enforces dependency, TensorRT shape,
single-inflight context, stable ownership, and memory limits.

The decision order is:

```text
hard feasibility
    -> explicit absolute-SLO safety, if present
    -> bounded no-SLO service recovery
    -> contextual Scalar + deterministic transition efficiency
```

When the oldest no-SLO phase has waited at least one frozen service quantum,
recovery retains candidates within one projected service quantum of the best
maximum age. This is a bounded eligibility filter, not strict oldest-first
authority.

Unknown overlap exploration is not allowed to override an already overdue
no-SLO service. Audit telemetry had shown a D request at 5.97 service quanta
while the selected reason was `bounded_exploration`; the final selector closes
that authority inversion.

### 4. Vision ownership safety

An earlier no-SLO VLM run exposed that count-only encoded-vision expansion from
16 to 80 could exceed the 10 GiB device even though each local action passed its
own feasibility checks. The final invariant is:

```text
throughputMaxEncodedInFlight > maxEncodedInFlight
    requires maxEncodedBytes > 0
```

Without a byte-level persistent-ownership contract, V3 clamps the expanded
capacity to the base capacity. This is independent of KV allocation; the OOM
was caused by request-owned vision payload lifetime, not indexed KV growth.

## Full-12 V2 versus V3

All numbers are medians of three fresh runs. Latency cells report the V3 change
relative to V2; negative is better. Both policies use the same v0.10.1 binary,
Cosmos engine, P128, P8/D64/E4 capacities, generic calibration, fixed-output
contract, and memory settings.

| Workload | V2 req/s | V3 req/s | Req/s delta | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|---:|---:|
| short | 112.406 | 108.537 | -3.44% | +3.33% / +11.10% | -6.32% / -11.25% | -0.55% / +1.23% |
| balanced | 49.994 | 50.274 | +0.56% | -2.76% / -8.42% | -0.72% / -2.13% | -0.79% / -1.88% |
| decode-heavy | 19.921 | 19.798 | -0.61% | -2.12% / -10.77% | +0.54% / +0.95% | +0.50% / +0.20% |
| long-prefill | 13.990 | 15.527 | +10.99% | -5.19% / +1.34% | -15.42% / -14.18% | -10.60% / -13.71% |
| bimodal | 12.716 | 12.959 | +1.91% | -0.12% / -3.89% | -5.88% / +3.38% | -2.83% / -1.66% |
| text-heavy | 30.855 | 37.502 | +21.54% | -29.04% / -25.31% | +4.51% / +25.96% | +0.14% / -17.63% |
| mixed | 16.139 | 20.407 | +26.44% | -29.40% / -19.53% | +49.51% / -0.58% | +2.95% / -21.20% |
| vision-heavy | 13.418 | 16.366 | +21.97% | -23.84% / -20.05% | +21.62% / +0.09% | -5.95% / -18.78% |
| poisson | 24.402 | 25.213 | +3.32% | -18.48% / -26.37% | +5.82% / +4.61% | -5.47% / -3.44% |
| wave-drain | 2.882 | 3.052 | +5.93% | -45.11% / -33.04% | -8.60% / -10.62% | -32.03% / -25.15% |
| multi-image | 9.014 | 9.372 | +3.97% | +0.77% / -1.15% | -19.43% / -18.91% | -0.89% / -2.98% |
| late-vision | 17.031 | 16.969 | -0.36% | +2.54% / +4.05% | +0.13% / +0.06% | +0.34% / +0.37% |

All 12 V2 and V3 traces are token-deterministic across their three repeats.
Peak memory is unchanged at 9,399 MiB for text-only and 9,611--9,635 MiB for
the VLM traces.

The three throughput losses need different interpretations:

- `decode-heavy` and `late-vision` are below 0.7% and do not establish a
  material regression at three repeats;
- `short` is a real throughput/TPOT trade-off: V3 loses 3.44% request throughput
  but improves TPOT mean by 6.32% and TPOT p95 by 11.25%, with E2E mean 0.55%
  lower;
- reintroducing an implicit latency threshold to force the V2 point would
  violate the no-SLO contract and hide this Pareto choice.

## V3 versus frozen vLLM

| Workload | V3 token/s | Frozen vLLM token/s | V3 delta |
|---|---:|---:|---:|
| short | 2,351.6 | 1,983.5 | +18.56% |
| balanced | 4,357.1 | 4,319.9 | +0.86% |
| decode-heavy | 5,147.6 | 4,854.3 | +6.04% |
| long-prefill | 1,345.7 | 1,120.9 | +20.06% |
| bimodal | 1,987.0 | 1,868.3 | +6.35% |
| text-heavy | 1,987.6 | 1,634.8 | +21.58% |
| mixed | 933.6 | 921.5 | +1.32% |
| vision-heavy | 630.1 | 579.2 | +8.79% |
| poisson | 1,840.5 | 1,800.1 | +2.25% |
| wave-drain | 97.7 | 95.9 | +1.91% |
| multi-image | 299.9 | 244.5 | +22.65% |
| late-vision | 2,447.8 | 2,359.2 | +3.75% |

This comparison supports current engineering direction but is not the final
publication comparison. Fresh vLLM repetitions are required if the request,
client-cap, EOS, engine/model, or HTTP contract changes.

## Rejected active variants

The following experiments were run and removed from source:

| Variant | Result | Decision |
|---|---|---|
| WAIT aggregate flow-time guard | text-heavy 33.027 req/s, mixed 20.739 req/s; TPOT not consistently better | reject |
| WAIT service-epoch metadata | text-heavy 34.002 req/s, mixed 21.768 req/s; old V3 remained faster or better balanced | reject |
| decode-only overdue exploration suppression | worse than phase-generic suppression | reject |
| allow D-serving unknown overlap while overdue | worse than suppressing unknown exploration for any overdue phase | reject |
| full normalized completion authority | short V2 gap improved to -1.06%, but text-heavy fell 16.85% and mixed 39.92% versus final V3 | reject |

The last failure is particularly useful: strict request-completion Pareto
authority overprotects individual service at high load and destroys productive
phase overlap. Service normalization should bound starvation, not replace the
Scalar/transition efficiency authority.

## Retained evidence

Primary results:

- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-recovery-before-exploration-core2-v3-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-recovery-before-exploration-core4-rest-v3-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-recovery-before-exploration-full12-rest-v3-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-safe-ownership-core4-v2-v3-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-safe-ownership-full12-rest-v2-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-v2-v3-full12-3x`

Diagnostics and rejected variants:

- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-v2-v3-regression-audit`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-v3-mixed-audit`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-wait-flow-core2-v3-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-wait-service-core2-v3-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-decode-recovery-before-exploration-core2-v3-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-service-preserving-exploration-core2-v3-3x`
- `.local/results/v0101-forward-port/v3-service-scale-20260909/no-slo-normalized-authority-core7-v3-3x`

## Verification

- Release build: `llm_phase_context_smoke`, `unitTestRuntime` pass.
- Runtime unit tests: 646 run, 644 pass, 2 expected skips.
- Replay contract: 17/17 pass under `unittest`.
- Service-normalized shadow: 16/16 pass under `unittest`.
- Full-12 fixed-output identity: 12/12 token-deterministic for V2 and V3.
- `git diff --check`: pass.

## Remaining work

1. Repeat the V3 no-SLO full-12 campaign on a second GPU and model. Service
   scaling is intended to remove device-specific milliseconds; portability is
   not proven on one RTX 3080/Cosmos pair.
2. Add a controlled low-load trace that reports the complete Pareto frontier
   rather than forcing a hidden throughput/latency preference into V3.
3. Fresh-run vLLM under an exactly equal client-cap and EOS contract for final
   publication numbers.
4. Measure scheduler decision p50/p95 and service-recovery invocation rate with
   dispatch telemetry disabled for performance and enabled only for diagnosis.
5. Keep explicit SLO support as an optional contract. No-SLO mode is a robust
   default, not a replacement for a user's real deadline when one exists.
