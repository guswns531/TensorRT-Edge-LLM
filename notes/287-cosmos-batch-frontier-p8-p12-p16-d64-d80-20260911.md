# Cosmos Batch Frontier: P8/P12/P16 and D64/D80

## 1. Decision

The global production default remains **P8/D64/E4 with a 128-token prefill
chunk**. More free memory does not make a larger TensorRT optimization profile
automatically faster.

- D80 is rejected. It reduced balanced throughput and made decode-heavy
  throughput 15.94% worse than D64 on the same P16/D80-capable engine.
- P12/D64 is infeasible with the retained 256-page FP16 KV contract. The engine
  selected a 4.70 GiB device-weight plan and the full runtime failed its initial
  allocation.
- P16/D64 is a real but workload-dependent opportunity. It improves mixed and
  vision-heavy throughput by 4--5%, but regresses long-prefill throughput by
  about 9% in the clean same-builder comparison.

This campaign changes neither KV layout nor KV capacity. All runnable variants
use stable indexed ownership, 80 logical slots, 256 FP16 KV pages (about
3.5 GiB), a 2,048-token context, and the same V3 scheduler. The experiment is
therefore an engine-profile and scheduler-capability comparison, not a KV-cache
optimization.

## 2. Reproducibility contract

| Item | Value |
|---|---|
| Source branch | `codex/v0101-phase-forward-port` |
| Source commit | `dce81ca4592348df23534e1c767cbbff5f77c0a6` |
| Source state during measurement | clean |
| Model | `nvidia/Cosmos-Reason2-2B`, FP16 weights and FP16 KV |
| GPU | RTX 3080 10 GiB |
| Runtime | frozen V3 Release binary |
| Runtime SHA-256 | `6f1b066ae1de44cf99adf3d60ae8fb1f9575b8f9ecc9ac0ed20975550e6ddefd` |
| Plugin SHA-256 | `daa90edae0c6ea4f749e9a5514867ba716ed195007eebab8f8aaaf4e39c75f09` |
| Admission / client concurrency | 80 for the initial cap screen; 64 for canonical comparisons |
| Policy initialization | generic, workload-agnostic calibration |
| Output contract | fixed output count, ignore EOS, captured token IDs |
| Retained results | `.local/results/v0101-forward-port/batch-frontier-20260911/` |

The current checkout's newly rebuilt runtime was also tested once with the
canonical engine, but it produced only 4,038 generated token/s and 46.60 req/s
on balanced. Its binary SHA differed from the frozen V3 binary, so that run is
excluded from the batch-size conclusion. All tables below use the frozen V3
runtime unless stated otherwise.

## 3. Engine profiles and memory

TensorRT selected materially different plans even though the ONNX graph, model,
KV capacity, chunk size, and builder invocation family were held fixed.

| Engine | Plan bytes | Device weight bytes | P activation bytes | D activation bytes | Load result |
|---|---:|---:|---:|---:|---|
| same-builder P8/D64 control | 3,081,868,764 | 3,053,761,792 | 79,695,360 | 21,548,544 | pass |
| P12/D64 | 4,721,145,276 | 4,697,941,248 | 119,541,248 | 21,548,544 | full-KV OOM |
| P16/D64 | 2,846,900,948 | 2,818,897,152 | 150,998,528 | 21,548,544 | pass |
| P16/D80 | 2,847,169,636 | 2,818,897,152 | 150,998,528 | 26,934,784 | pass |

The P12 failure is not caused by additional KV pages. A four-page smoke can
start, but restoring the common 256-page pool fails in `cudaMalloc` with backend
exit code 139. The relevant log is:

```text
.local/results/v0101-forward-port/batch-frontier-20260911/
  p12-d64-load-gate/balanced/run-001/gateway.log
```

The non-monotonic 3.05/4.70/2.82 GiB weight allocations for P8/P12/P16 show why
free framebuffer alone cannot predict a feasible or fast maximum profile.
TensorRT tactic selection and packed-weight representation are part of the
batch-size contract.

Measured residency is similarly non-monotonic:

| Runtime path | Ready / peak MiB | Nominal headroom |
|---|---:|---:|
| canonical P8/D64 text | 9,399 | 841 MiB |
| same-builder P8/D64 control | 9,401 | 839 MiB |
| P16/D64 text | 9,211 | 1,029 MiB |
| P16/D64 mixed peak | 9,423 | 817 MiB |
| P16/D64 vision-heavy peak | 9,419 | 821 MiB |

P16 uses a larger prefill activation buffer but a smaller packed-weight plan,
so its total residency is lower. This is an engine-build effect, not a dynamic
KV-pool saving.

## 4. Decode frontier: D64 is still the optimum tested cap

The first screen reused one P16/D80-capable engine and changed only runtime
caps at admission 80.

| Balanced cap | generated token/s | req/s | Relative to P8/D64 |
|---|---:|---:|---:|
| P8/D64 | 3,861.57 | 44.557 | reference |
| P16/D64 | 4,066.74 | 46.924 | +5.31% |
| P8/D80 | 3,666.42 | 42.305 | -5.05% |
| P16/D80 | 3,660.19 | 42.233 | -5.22% |

Decode-heavy makes the D80 failure unambiguous:

| Cap | token/s | req/s | TTFT mean ms | TPOT mean ms | E2E mean ms | D dispatches |
|---|---:|---:|---:|---:|---:|---:|
| P16/D64 | 4,727.65 | 18.183 | 409.4 | 13.31 | 3,844 | 1,455 |
| P16/D80 | 3,974.29 | 15.286 | 544.4 | 15.74 | 4,674 | 1,907 |

D80 loses 15.94% throughput and creates more, smaller effective decode
dispatches. The profile adds only about 5.1 MiB of decode activation space, so
memory is not the limiting factor; cohort formation and compiled execution cost
are.

## 5. Clean P8 versus P16 comparison

The strongest causal control rebuilt P8/D64 and P16/D64 through the same
builder lifecycle and ran the same frozen V3 binary, requests, calibration,
KV pool, admission, and client concurrency.

| Workload | Engine | token/s | req/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---|---:|---:|---:|---:|---:|
| balanced | P8/D64 | 4,321.38 | 49.862 | 60.10 / 160.03 | 12.84 / 14.50 | 1,150.90 / 1,827.58 |
| balanced | P16/D64 | 4,341.66 | 50.096 | 68.07 / 125.74 | 12.77 / 14.31 | 1,151.36 / 1,778.01 |
| long-prefill | P8/D64 | 1,356.45 | 15.651 | 1,943.49 / 2,632.73 | 21.88 / 26.65 | 3,807.46 / 5,286.76 |
| long-prefill | P16/D64 | 1,234.80 | 14.248 | 2,110.71 / 2,667.91 | 24.45 / 29.53 | 4,185.76 / 5,743.76 |

On balanced, P16 is essentially throughput-neutral (+0.47%). It trades a 13.3%
worse mean TTFT for a 21.4% better TTFT p95 and a 2.7% better E2E p95. On
long-prefill, however, P16 loses 8.97% throughput, raises mean TPOT by 11.72%,
and raises mean E2E by 9.93%. This violates the no-regression promotion gate.

## 6. Five-workload directional screen against canonical P8 V3

The canonical P8 column is the retained three-run median. P16 is a new one-run
screen, so these numbers establish direction rather than a citable confidence
interval. Positive throughput means P16 is faster; positive latency means P16
is lower.

| Workload | req/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | +1.16% | -6.76% | +22.87% | +0.99% | +1.46% | +0.84% | +2.41% |
| decode-heavy | +1.58% | -14.15% | +8.43% | +1.67% | +3.32% | +1.53% | +1.95% |
| long-prefill | -8.32% | -7.21% | +0.24% | -12.76% | -14.75% | -9.49% | -10.82% |
| mixed | +5.37% | +0.30% | +5.84% | +14.07% | +5.03% | +6.47% | +5.12% |
| vision-heavy | +4.01% | +3.74% | +3.52% | +2.79% | +1.45% | +2.22% | +3.98% |

The five-workload request-throughput geometric mean is only about +0.6% because
the long-prefill loss consumes most of the mixed/VLM gain. A full-12 campaign
would therefore spend substantial time without passing the already-failed
promotion gate.

vLLM was not rerun: the model, request traces, KV capacity, image contract, and
SLO-free evaluation contract are unchanged. The retained fresh vLLM 0.28.0
campaign remains the external reference; this campaign isolates Current's
compiled batch frontier.

## 7. Why P16 regresses on long prefill

Full activity telemetry confirms that the scheduler actually uses the larger
cap. It is not an inactive-option result.

| Long-prefill telemetry | P8/D64 | P16/D64 | P16 change |
|---|---:|---:|---:|
| P dispatches | 796 | 744 | -6.53% |
| Mean P batch | 2.261 | 2.419 | +6.99% |
| Maximum P batch | 8 | 16 | +100% |
| Mean useful P tokens/dispatch | 267.62 | 286.32 | +6.99% |
| Total P GPU time | 16,119.6 ms | 17,656.4 ms | +9.53% |
| P GPU ms / 1k useful tokens | 75.67 | 82.88 | +9.53% |

P16 formed an actual batch of 16 eleven times and reduced dispatch count. The
larger profile nevertheless increased total prefill GPU work and cost per useful
token. The regression is therefore compiled-kernel/tactic inefficiency, not a
failure to generate large batches.

## 8. Architecture consequence and next step

The tested single-profile choices expose a genuine frontier:

```text
P8 profile
  + efficient long-prefill kernels
  - cannot exploit occasional dense P9--P16 cohorts

P16 profile
  + helps mixed and vision-heavy bursts
  - changes tactics and hurts sustained long-prefill
```

The next credible optimization is not a workload-specific `if mixed then P16`
rule. It is a shape-local capability that preserves the P8 fast path and uses a
separately validated P16 path only when the current ready snapshot produces a
dense, profitable large cohort. Candidate designs are:

1. two prefill optimization profiles or specialized prefill engines, P1--P8 and
   P9--P16;
2. automatic profile selection from current candidate shape and measured
   isolated service cost;
3. a memory/build gate that rejects profile combinations which duplicate packed
   weights or violate the common 256-page KV contract;
4. graph/profile identity included in the online physical model and CUDA graph
   cache key.

Before implementing this, TensorRT profile packing must be measured because the
P12 result proves that profile composition can change weight memory by gigabytes.
Until a dual-profile design preserves P8 long-prefill cost, **P8/D64 remains the
best robust global configuration** and P16/D64 remains a diagnostic candidate.

## 9. Artifact disposition

The result root is retained as `diagnostic` with a campaign manifest. The P12
OOM engine and same-builder P8 control are reproducible scratch artifacts and
are the first cleanup candidates; P16/D64 is retained for specialized-profile
work. No engine is deleted by this campaign because engine cleanup requires an
explicit allowlist under the repository retention policy.
