# 273. Canonical build and service-normalized authority P0--P4

## Outcome

P0 through P4 were executed on the v0.10.1 forward-port tree. The canonical
CUDA 13.3/SM86 Release build and rebuilt vision engine are valid, and the
bounded transition/DAG mechanism remains correct. A service-normalized selector
was connected to the production frontier behind an explicit opt-in.

The first active selector was too permissive. It ran before the existing
efficiency Pareto filter and failed the four-workload promotion gate: it traded
lower mean TTFT for substantially worse decode continuity on text-heavy and
mixed, and regressed multi-image throughput. It was therefore not promoted.

The implementation was changed to a fail-closed form. Service-normalized
authority is now considered only after hard feasibility, SLO safety, and the
existing dominance filter. It may replace the V2 choice only when every
canonical protected request has a no-worse normalized projected service age
and at least one is strictly better. No workload name, model name, batch-size
rule, or new SLO threshold is used.

On one audit repeat of each of the four gate workloads, the final conservative
form found zero safe replacement opportunities. This is a valid negative
promotion result: the mechanism is implemented and tested, but it provides no
natural performance benefit on this gate and remains opt-in. The production
default stays V0/V1/V2 without service-normalized authority.

## P0: canonical Release/SM86 substrate

The retained performance build is now:

```text
.local/builds/v0101-release
  CMAKE_BUILD_TYPE=Release
  CMAKE_CUDA_ARCHITECTURES=86
  CUDA_CTK_VERSION=13.3
  CUDA_DIR=/usr/local/cuda-13.3
  TRT_PACKAGE_DIR=/opt/tensorrt
  BUILD_UNIT_TESTS=ON
  ENABLE_CUTE_DSL=fmha
```

The missing SM86 CuTeDSL FMHA artifact was regenerated with 25 variants. The
first stale CUDA 12.8 CMake attempt was moved under
`.local/scratch/failed-builds/`; it is not a retained baseline.

Canonical artifact identities:

| Artifact | SHA256 |
|---|---|
| `llm_phase_context_smoke` | `9959b6097df5dba67a531b901c6308d1594c4aa38fbaef6ac82e3b8798cc5598` |
| `libNvInfer_edgellm_plugin.so.1.0` | `627ea41795979a7e012ab95e0e9b229a22b782fd6dea6b8a44167148592912df` |
| text `llm.engine` | `084d039248e08dc192a57ebf38d521ee1fdca2f037a865f55ac0b0b904214b0b` |

The initial broad GPU test selected 338 tests: 337 passed and the optional
`PhaseKVActiveViewTest.IsolatedMetadataCostBenchmark` was skipped. After the
selector changes, 67 focused selector/formation/unified-event tests passed, and
the final broad run selected 339 tests: 338 passed with the same optional skip.

`.local/current/build` now resolves to the canonical Release build.

## P1: canonical vision engine

The exact-GELU vision ONNX was rebuilt with the canonical Release `visual_build`
and stored under:

```text
.local/artifacts/v0101-forward-port/cosmos-reason2-2b/
  vision-canonical-release/
```

The old and canonical engines have the same 574,899,200-byte activation report.
The engine hashes differ because the builder lifecycle differs, but paired
synthetic E1/E2/E4 output hashes are exact. Median-of-repeat CUDA times are:

| Encoder shape | previous engine ms | canonical engine ms | delta |
|---|---:|---:|---:|
| E1 / 512 patches | 9.5284 | 9.5297 | +0.013% |
| E2 / 1024 patches | 13.7283 | 13.7296 | +0.010% |
| E4 / 2048 patches | 21.7767 | 21.7777 | +0.005% |

The full paired record is
`.local/results/v0101-forward-port/p0-p4-20260909/vision-current-vs-canonical-512.json`.
`.local/current/vision` and `.local/current/vision-engine` now resolve to this
canonical artifact.

## P2: bounded transition replay gate

The existing deterministic transition mechanism was revalidated rather than
reimplemented:

```text
candidate action
  -> concrete completion boundary
  -> request DAG transition
  -> newly ready E/P/D work
  -> canonical cohort formation
  -> KV/vision ownership transition
  -> second bounded boundary
```

Seventeen `PhaseFormationPlannerTest` cases passed, including physical
completion-order replay into the request DAG and ownership, final-decode KV
release, action-induced fragmentation, equal-work horizons, and rejection of an
illegal DAG completion.

This is a deterministic in-memory transition evaluator. It is not a complete
checkpoint/restore of live TensorRT contexts, KV contents, sampler state, and
queues. A true same-GPU physical alternative branch remains a later causal
experiment and is not claimed here.

## P3: service-normalized selector

Each request-local protected milestone now carries:

```text
request ID
isolated service reference and provenance
elapsed service time at this decision boundary
predicted completion and uncertainty
```

The normalized projected age is:

```text
(elapsed service + predicted completion + uncertainty)
------------------------------------------------------
            immutable isolated service
```

First-token elapsed service starts at request submission and spans E to P.
Decode elapsed service starts at the previous token commit. Runtime exact,
interpolated, covering, static-profile, and derived-isolated references are
eligible; unknown and cold fallback references fail closed.

Authority requires all alternatives to expose the same canonical request set,
reference value, provenance, and elapsed clock. The final selector order is:

```text
hard feasibility
  -> explicit SLO-safe frontier
  -> existing execution/memory dominance pruning
  -> existing V2 selection
  -> optional all-request Pareto-safe service replacement
```

The option is enabled only for V2 with
`TRT_EDGELLM_SERVICE_NORMALIZED_AUTHORITY=1`. Audit telemetry records
`elapsed_service_us`, `service_normalized_authority`, and
`max_normalized_service_age`.

The replay harness was also repaired so retained commands always use the
validated `--build-cache` build, preserve the original guarded HTTP client
implementation through nested manifests, and can explicitly enable this
authority. These changes prevent a stale development binary from contaminating
the comparison.

## P4: four-workload promotion gate

All rows use the same canonical binary, text engine, canonical vision engine,
request traces, generic calibration, fixed output contract, memory limits, and
legacy pair-eligibility mechanism. Each policy/case was measured three times
without per-dispatch telemetry. All 24 runs were token deterministic and control
versus candidate token hashes matched exactly.

The first active selector produced the following candidate/control deltas:

| Workload | req/s | TTFT mean/p95 | TPOT mean/p95 | E2E mean/p95 |
|---|---:|---:|---:|---:|
| balanced | +1.37% | +1.67% / -0.44% | -1.50% / -1.37% | -1.35% / -1.04% |
| text-heavy | -1.37% | -13.32% / -0.22% | +5.00% / +10.48% | +1.51% / +1.20% |
| mixed | -0.67% | -4.04% / +5.28% | +6.61% / +23.10% | +2.48% / +0.60% |
| multi-image | -3.76% | +3.39% / +6.28% | +4.18% / +1.04% | +3.67% / +3.86% |

Lower latency is better. This failed promotion decisively. In particular, mean
TTFT improvement alone did not justify text-heavy/mixed decode service loss.
Multi-image also contained one large candidate outlier, but the median still
regressed, so it was not discarded.

The corrected all-request Pareto-safe selector was then audited once per case:

| Workload | decision events | service replacements |
|---|---:|---:|
| balanced | 1,046 | 0 |
| text-heavy | 563 | 0 |
| mixed | 605 | 0 |
| multi-image | 568 | 0 |

Because the corrected policy made no replacements, another full performance
sweep would measure only run-to-run noise. Per the promotion protocol, the
full12 expansion was stopped. The initial failing screen and final opportunity
audit are retained under
`.local/results/v0101-forward-port/p0-p4-20260909/`.

## Current V2 and frozen vLLM context

The service-normalized experiment does not replace the current V2 baseline.
For context, canonical V2 control from this gate remains ahead of the frozen
equal-capacity vLLM result whose model/request/KV-capacity contract is unchanged:

| Workload | current V2 token/s | frozen vLLM token/s | delta |
|---|---:|---:|---:|
| balanced | 4454.53 | 4318.34 | +3.15% |
| text-heavy | 2001.06 | 1634.76 | +22.41% |
| mixed | 1111.62 | 921.48 | +20.63% |
| multi-image | 307.14 | 244.52 | +25.61% |

This is a fresh three-repeat current run versus the retained vLLM measurements,
not an interleaved fresh vLLM campaign. vLLM was not rerun because the serving
contract did not change.

## Decision

- Promote the canonical Release build and canonical vision engine as local
  artifact pointers.
- Keep V0 Exact, V1 Scalar, and V2 Scalar+Transition as the supported policy
  variants.
- Keep service-normalized authority disabled by default and experimental.
- Do not add workload-specific exceptions to make the four-case table pass.
- Do not run full12 for a policy with zero natural safe replacements.

The next useful step is not more tuning of the normalized scalar. It is to
capture strict same-state alternatives for the previously observed shadow/live
disagreements and determine whether a non-trading, transition-level improvement
exists. If it does not, service normalization should remain diagnostic telemetry
rather than a production selector.
