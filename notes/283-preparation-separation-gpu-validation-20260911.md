# Preparation/Formation Separation: GPU Validation and Promotion Decision

## 1. Scope and contract

This campaign validates commit `751bc65` after the P0--P3 preparation
separation in `282-preparation-formation-separation-p0-p3-20260911.md`.
Every primary comparison uses the same:

- Cosmos-Reason2-2B text and vision engines;
- Release binary and TensorRT/CUDA container;
- fixed-output request traces and token identities;
- generic workload-agnostic calibration;
- P8/D64/E4 engine capability and P128 prefill chunk;
- memory, admission, and request-concurrency limits.

Only the two new runtime switches differ:

| Variant | P/D during preparation | Pure E execution cost |
|---|---:|---:|
| compatibility | off | off |
| P/D-dispatch | on | off |
| separate-cost | off | on |
| full | on | on |

Each of mixed, vision-heavy, wave-drain, and multi-image ran three times. All
variants produced deterministic token traces identical to compatibility.
Retained inputs, commands, manifests, logs, and aggregate files are under:

```text
.local/results/v0101-forward-port/preparation-separation-20260911/
```

The machine exposed an RTX 3080 through the NVIDIA container runtime.

## 2. Compatibility parity with the frozen V3

The persistent preparation worker preserves the old policy contract. Positive
throughput means compatibility is faster; positive latency means lower.

| Workload | req/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | +1.00% | +0.94% | +1.15% | -0.47% | -0.38% | +0.65% | +0.91% |
| vision-heavy | +0.55% | +1.01% | +3.80% | +4.59% | -3.14% | +1.09% | +0.79% |
| wave-drain | +0.04% | +0.24% | -0.29% | +0.17% | -6.27% | +0.21% | -0.11% |
| multi-image | -0.19% | -0.18% | -0.15% | +0.64% | +3.92% | -0.14% | -0.13% |

The central metrics and token contract are at parity. The small wave TPOT-p95
change is not accompanied by throughput, TTFT, or E2E movement and remains a
tail-repeat item rather than evidence of a mechanism change.

## 3. Absolute primary results

### 3.1 Mixed

| Variant | req/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | peak MiB |
|---|---:|---:|---:|---:|---:|
| compatibility | 24.232 | 776.13 / 2316.61 | 29.17 / 40.46 | 2192.94 / 2581.74 | 9609 |
| P/D-dispatch | 24.068 | 765.86 / 2317.74 | 29.22 / 41.37 | 2223.31 / 2597.27 | 9601 |
| separate-cost | 23.981 | 782.24 / 2307.22 | 29.75 / 42.64 | 2227.45 / 2606.58 | 9609 |
| full | 24.058 | 761.22 / 2321.38 | 30.72 / 41.62 | 2244.59 / 2600.12 | 9601 |

### 3.2 Vision-heavy

| Variant | req/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | peak MiB |
|---|---:|---:|---:|---:|---:|
| compatibility | 16.959 | 1454.07 / 3292.59 | 26.40 / 39.03 | 2520.49 / 3699.30 | 9601 |
| P/D-dispatch | 16.836 | 1384.88 / 3281.37 | 29.95 / 38.48 | 2557.96 / 3706.11 | 9601 |
| separate-cost | 16.945 | 1492.25 / 3401.71 | 26.28 / 38.67 | 2544.36 / 3710.04 | 9601 |
| full | 16.736 | 1472.45 / 3463.45 | 27.02 / 38.51 | 2546.14 / 3759.35 | 9601 |

### 3.3 Wave-drain

| Variant | req/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | peak MiB |
|---|---:|---:|---:|---:|---:|
| compatibility | 3.053 | 277.25 / 317.04 | 7.89 / 9.84 | 521.78 / 529.69 | 9601 |
| P/D-dispatch | 3.051 | 262.82 / 318.43 | 8.65 / 12.44 | 522.49 / 530.95 | 9601 |
| separate-cost | 3.047 | 254.38 / 322.95 | 8.79 / 11.73 | 525.29 / 535.89 | 9603 |
| full | 3.052 | 262.03 / 319.11 | 8.60 / 11.86 | 528.31 / 547.87 | 9601 |

### 3.4 Multi-image

| Variant | req/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms | peak MiB |
|---|---:|---:|---:|---:|---:|
| compatibility | 9.391 | 284.32 / 318.52 | 7.90 / 8.89 | 529.11 / 531.64 | 9629 |
| P/D-dispatch | 9.262 | 243.77 / 326.47 | 8.67 / 11.84 | 533.94 / 539.63 | 9601 |
| separate-cost | 9.553 | 269.70 / 310.68 | 7.95 / 9.23 | 516.04 / 522.90 | 9607 |
| full | 9.372 | 272.38 / 319.74 | 8.32 / 11.02 | 530.36 / 532.97 | 9601 |

## 4. Causal interpretation

Compatibility blocked otherwise-ready P/D behind preparation for only about
48--70 ms over a complete measured-plus-calibration process. Removing that
logical block reduced several E or prefill queue tails, but preparation itself
is not free. Its CUDA interval reached about 6--8 ms, and host publication
reached roughly 9--16 ms depending on the resulting contention.

The P/D-dispatch variant therefore trades first-token progress for resident
decode continuity:

- mixed TTFT mean improves 1.32%, but TPOT p95 regresses 2.23%;
- vision-heavy TTFT mean improves 4.76%, but TPOT mean regresses 13.47%;
- wave TTFT mean improves 5.21%, but TPOT p95 regresses 26.34%;
- multi-image TTFT mean improves 14.26%, but TPOT p95 regresses 33.21%.

This is not a memory effect. Peak memory is unchanged within 28 MiB and the
direct-output runner records no output D2D-copy path. It is a placement and
cohort-continuity effect.

Pure execution-cost accounting is independently useful but not universally
dominant. It is best on multi-image, improving req/s by 1.73%, TTFT p95 by
2.46%, E2E mean by 2.47%, and E2E p95 by 1.64%, while mixed and wave regress.
This option remains an experimental physical-model correction rather than a
new default.

## 5. CUDA activity audit

Diagnostic one-run multi-image traces used full lifecycle telemetry to remove
generic warmup. These numbers are not mixed into the primary latency table.

| Metric | compatibility | full |
|---|---:|---:|
| measured active span | 466.05 ms | 403.38 ms |
| all-idle ratio | 2.10% | 2.09% |
| E active | 99.54 ms | 50.97 ms |
| P active | 124.60 ms | 169.99 ms |
| D active | 232.19 ms | 219.17 ms |
| E/P/D overlap | 0.05 ms | 45.18 ms |
| D dispatches | 34 | 32 |
| D gap p95 | 1.55 ms | 0.31 ms |

The full mode demonstrably creates real E+P overlap; it does not merely change
host action labels. Nevertheless, the repeated primary run shows worse TPOT
tail. A shorter GPU active span in one diagnostic run is therefore not enough
to promote concurrency. The scheduling objective must protect request service,
not maximize overlap occupancy.

## 6. P6: removing the 25 ms wait and arbiter

The diagnostic full mode was also run with both compatibility guards removed.
Relative to compatibility:

| Workload | req/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mixed | -0.71% | +4.98% | +0.13% | -7.01% | -2.03% | -3.10% | -0.59% |
| vision-heavy | -1.58% | -1.08% | -3.40% | -5.60% | +1.30% | -3.50% | -1.79% |
| wave-drain | +0.09% | +16.47% | +1.16% | -10.48% | -22.35% | +2.07% | +0.49% |
| multi-image | +2.26% | +21.31% | +5.25% | -11.31% | -23.58% | +2.82% | +2.21% |

The deletion is not safe. It advances new vision requests, especially on wave
and multi-image, but sacrifices ongoing decode service. The old controls are
still compensating for the absence of request-local preparation ownership and
late E packing. They must not be removed by retuning another fixed timer.

## 7. P4/P7 implementation decision

P4 cannot be implemented as a coordinator-only prepared queue. Qwen/Cosmos
`preprocess()` mutates reusable runner-owned packed patches, cumulative sequence
lengths, fast-position inputs, M-RoPE state, and TensorRT binding shapes. A
second singleton preparation would overwrite the first. A correct P4 requires
a detached request input lease and an explicit later materialization step.

The P3 gate also shows that preparation competes for real GPU/copy/host
resources. Therefore P4 is not promoted speculatively in this campaign. The
minimum safe implementation is:

```text
request-local preparation
    -> detached packed-patch/input lease + ready event
prepared pool
    -> resource- and ownership-visible entries
late materialization
    -> concatenate leases, rebuild cumulative offsets/auxiliary bindings
E submit
    -> one packed TensorRT invocation
```

Cosmos uses M-RoPE, so text-prefix-before-vision currently fails a correctness
capability check before the `minPrefixBeforeVisionTokens` threshold. Sweeping
that threshold would be a no-op, not a policy experiment. Prefix P remains a
future correctness feature for this model, while P/D local candidate-membership
audits remain separate from this preparation campaign.

## 8. Comparison with frozen vLLM

vLLM was not rerun because its model, request, fixed-output, concurrency, and
memory contract did not change. The frozen comparison remains valid. The
canonical V3 already led frozen vLLM on token throughput by 18.87% on mixed,
12.50% on vision-heavy, 1.92% on wave-drain, and 23.44% on multi-image. This
campaign does not replace those headline results because no new mode passes the
cross-metric promotion gate.

## 9. Promotion decision and next steps

1. Keep the persistent worker and preparation/execution observability.
2. Keep both new behavior switches opt-in; compatibility remains the default.
3. Do not remove the 25 ms wait or arbiter yet.
4. Retain separate-cost as the most promising narrow branch and repeat it on a
   scaled multi-image trace before any activation change.
5. Implement detached Qwen/Cosmos input ownership as an isolated mechanism PR,
   with exact singleton-to-packed output identity before scheduler integration.
6. Only then add a prepared pool and late E packing, first in compatibility
   emulation and then with known-event decisions.
7. Run VLM6, full-12, and the frozen vLLM comparison only after the mechanism
   gate passes.

The result narrows the design rather than invalidating it:

```text
preparation is a first-class transition with a resource footprint;
unconditional overlap is not its scheduler policy.
```
