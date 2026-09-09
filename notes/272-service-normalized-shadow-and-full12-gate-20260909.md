# 272. Service-normalized runtime shadow and full12 gate

## Outcome

The bounded service-normalized evaluator is now connected to real v0.10.1
decision snapshots as **shadow-only telemetry**. It has no production selection
authority. The connection preserves candidates removed by the legacy expired-P
guard, identifies the protected request explicitly, records an immutable
isolated-service denominator and its provenance, and refuses cold or aggregate
substitutes.

On a fresh mixed V2 run, real scorable decisions increased from 0/115 to
110/111. The one unsupported residual decision remained missing. The new shadow
choice disagreed with the live policy in 23/110 scorable decisions, including 13
live-P versus shadow-D decisions. This is an opportunity count, not proof that
the alternative would have completed faster.

The live policy remains V1 Scalar or V2 Scalar+Transition. No SLO threshold,
RLS feature/reward, candidate admission, KV ownership, CUDA dependency, engine,
or model was changed.

## Runtime architecture

```text
ready E/P/D + outstanding contexts + ownership
                    |
                    v
       mechanism-only bounded frontier
       dependency / TRT shape / inflight / memory
                    |
          +---------+------------------+
          |                            |
          v                            v
 existing policy pruning       immutable shadow copy
 SLO / overlap evidence          suppressed D retained
 Scalar or V2 selection          contextual authority removed
          |                            |
          v                            v
       dispatch                 request-local projections
                                + service clocks
                                + isolated references/provenance
                                          |
                                          v
                                offline normalized Pareto audit
                                production_authority = false
```

The shadow protects at most one stable, tie-broken critical request per active
phase. It does not duplicate the oldest-D estimate across every ready row. A
request without a supported request-local projection makes that alternative
incomplete. This is deliberate fail-closed behavior.

First-token age uses original submission across E to P. Decode age uses the last
host token-commit timestamp. These are host-monotonic service clocks, not CUDA
completion, sampling-ready, or HTTP-delivery timestamps.

## Implementation

- `phaseDeadline.h`: request-local protected completion fields and explicit
  isolated-reference provenance.
- `phaseReadySnapshot.h` and `phaseQueueScheduler.cpp`: stable request ID for the
  minimum decode slack; canonical P/D service reference; policy-neutral shadow D
  construction before the expired-P guard removes it.
- `phaseGlobalScheduler.h`: full mechanism candidate objects in the diagnostic
  audit. They are copies and contextual decision authority is stripped.
- `phaseThreeCoordinator.cpp`: merges P/D mechanism candidates with E/E+P/E+D,
  applies dependency/shape/inflight/memory legality, preserves original encoder
  submission clocks, and attaches request-local E first-token estimates.
- `phaseUnifiedEvent.h` and `llm_phase_context_smoke.cpp`: separate
  `mechanism_candidates`, cost provenance, protected request ID/reference, and
  service clocks in counterfactual telemetry.
- `service_normalized_shadow.py`: reconstructs a common protected-work target from
  a real event, rejects action-dependent denominators and unsupported residuals,
  and reports a Pareto frontier plus a diagnostic lexicographic choice.
- `replay_retained_policy_commands.py`: exposes the runtime's compact
  `counterfactual` telemetry level so candidate evidence does not require full
  kernel/request traces.

Accepted isolated-reference provenance is runtime exact/interpolated/covering,
static profile, or a documented composition of isolated phase costs. Cold
fallback, queue residence, explicit SLO target, and selected overlap timing are
not accepted denominators.

## Real shadow coverage

Result root:
`.local/results/v0101-forward-port/service-shadow-v2-20260909/`.

Same Cosmos FP16/native vision, P8/D64/E4, chunk128, KV256x128, graphs off,
legacy pair eligibility, 319 generic warmup requests, 64 mixed measurement
requests, and 2928 fixed output tokens. Output SHA256 remains
`73155475f432ffd2f98e5840347b702466850a69581fa1b9698b675b22908085`.

| Coverage | Count |
|---|---:|
| decisions | 111 |
| active policy candidate snapshots | 191 |
| mechanism-only snapshots | 224 |
| extra shadow candidates | 33 |
| suppressed standalone D retained | 33 |
| service snapshots reconstructed/scored | 110 |
| unsupported cold/residual snapshots | 1 |
| shadow/live choice disagreements | 23 |

Cost/reference provenance pairs over 224 mechanism candidates:

| predicted / reference source | candidates |
|---|---:|
| runtime exact / runtime exact | 118 |
| static profile / static profile | 40 |
| derived isolated / derived isolated | 36 |
| runtime covering / runtime covering | 16 |
| runtime exact / derived isolated | 10 |
| cold fallback / cold fallback | 3 |
| derived isolated / runtime exact | 1 |

Cold candidates remain visible for diagnosis but are not normalized or granted
new authority.

## Alternating-order mixed repeats

All rows use the same rebuilt binary and output hash. AB and BA roots reverse the
process order. `n` is the number of independent measurement repeats pooled across
both orders. Values are medians of run statistics.

### Telemetry off, n=6 per policy

| policy | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|
| V1 Scalar | 1108.79 | 752.39 / 2044.34 | 36.60 / 60.30 | 2449.90 / 2610.12 |
| V2 Scalar+Transition | 1107.87 | 704.56 / 2079.79 | 38.88 / 62.46 | 2468.01 / 2615.61 |
| V2 vs V1 | -0.08% | -6.36% / +1.73% | +6.21% / +3.58% | +0.74% / +0.21% |

Throughput ranges are V1 1070.43--1121.64 and V2 1098.05--1121.04 token/s.
V2 improves mean TTFT but does not recover decode continuity; it is not a
uniform winner.

### Counterfactual telemetry on, n=4 per policy

| policy | token/s | TTFT mean/p95 ms | TPOT mean/p95 ms | E2E mean/p95 ms |
|---|---:|---:|---:|---:|
| V1 Scalar | 1100.77 | 746.12 / 2216.08 | 34.23 / 42.97 | 2329.00 / 2618.01 |
| V2 Scalar+Transition | 1097.51 | 696.22 / 2192.57 | 39.46 / 59.97 | 2498.69 / 2649.11 |

Throughput medians are 0.72% and 0.94% below telemetry-off V1/V2, respectively,
but the run ranges overlap and latency statistics vary substantially. The data
supports keeping this instrumentation out of headline performance runs; it does
not support subtracting one constant overhead.

Across these four diagnostic repeats, 473/476 V1 and 464/465 V2 decisions were
scored. V1 has no V2 formation evaluation by design. V2 had 455 valid existing
formation evaluations. Missing rows were unsupported/cold rather than filled by
an aggregate estimate.

## Fresh no-telemetry full12 screen

This is one fresh process per workload and policy, not a replacement for the
retained three-repeat suite in note262. V1 and V2 generated identical token traces
in 12/12 workloads. Latencies are mean/p95 in milliseconds.

| workload | V1 tok/s | V2 tok/s (delta) | V1 TTFT | V2 TTFT | V1 TPOT | V2 TPOT | V1 E2E | V2 E2E | frozen vLLM tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 2492.52 | 2478.38 (-0.57%) | 84.54/167.32 | 88.66/168.33 | 13.26/21.80 | 13.22/21.83 | 326.99/408.22 | 329.07/410.46 | 1983.53 |
| long-prefill | 1305.47 | 1273.95 (-2.41%) | 1937.64/2433.88 | 1981.21/2531.32 | 23.76/27.32 | 24.43/28.46 | 3956.40/5375.88 | 4056.30/5693.48 | 1127.42 |
| bimodal | 1919.05 | 1945.19 (+1.36%) | 1914.02/4157.09 | 1858.08/4028.60 | 18.07/27.08 | 17.85/27.26 | 4384.33/9189.97 | 4275.99/8936.91 | 1852.31 |
| balanced | 4461.75 | 4348.29 (-2.54%) | 61.53/170.07 | 64.35/173.78 | 12.38/13.72 | 12.71/14.43 | 1114.95/1752.09 | 1145.74/1815.68 | 4318.34 |
| decode-heavy | 5263.04 | 5180.87 (-1.56%) | 61.55/180.95 | 62.57/184.69 | 10.57/11.15 | 10.75/11.41 | 2791.45/4270.19 | 2837.46/4352.97 | 4943.41 |
| text-heavy | 2016.65 | 2027.39 (+0.53%) | 350.99/1137.90 | 374.25/1090.57 | 23.23/34.93 | 22.55/36.36 | 1562.32/1659.00 | 1552.16/1657.54 | 1634.76 |
| mixed | 1060.04 | 1100.74 (+3.84%) | 757.20/2102.71 | 812.81/2020.07 | 38.95/60.41 | 34.95/53.79 | 2531.85/2699.46 | 2416.90/2606.85 | 921.48 |
| poisson | 1952.18 | 1888.28 (-3.27%) | 231.97/771.30 | 245.82/893.97 | 21.09/40.27 | 21.97/37.74 | 1576.49/2013.55 | 1668.16/2100.27 | 1800.07 |
| vision-heavy | 642.69 | 685.88 (+6.72%) | 1415.40/3365.85 | 1334.92/3178.68 | 35.10/54.98 | 54.07/87.04 | 2828.59/3765.28 | 3378.67/3552.38 | 579.20 |
| wave-drain | 97.79 | 97.62 (-0.17%) | 275.64/307.41 | 277.87/333.10 | 7.56/8.73 | 7.98/8.95 | 509.95/519.57 | 525.13/544.63 | 95.85 |
| late-vision | 2482.53 | 2475.06 (-0.30%) | 120.63/433.91 | 126.38/446.00 | 9.49/9.53 | 9.50/9.57 | 1479.85/1857.23 | 1488.04/1864.21 | 2359.23 |
| multi-image | 239.15 | 299.68 (+25.31%) | 367.65/459.51 | 284.35/321.33 | 8.68/11.04 | 7.90/9.49 | 636.86/668.79 | 529.12/533.62 | 244.52 |

V2/V1 throughput geometric mean is +2.00%; V1/frozen-vLLM is +9.51% with
11/12 throughput wins, and V2/frozen-vLLM is +11.70% with 12/12 wins. The vLLM
rows are frozen note262 results because model, engine-facing request contract,
fixed output, memory capacity, and workload are unchanged. They were not rerun.

Against frozen vLLM, the fresh V1/V2 screen has lower E2E mean in 12/12 and
lower E2E p95 in 11/12. Geometric-mean latency changes are V1 -13.84%/-10.46%
and V2 -13.26%/-11.48% for E2E mean/p95. TTFT mean wins are 8/12 for both;
TTFT p95 wins are 10/12 and 11/12. TPOT mean/p95 wins are 11/12 for both.
These remain one-run-current versus retained-three-run-vLLM comparisons, not
confidence intervals from a fresh paired campaign.

This one-run screen is noisy: multi-image has known high variance and dominates
part of V2's aggregate gain. V2 beats V1 in only 3/12 TTFT means, 5/12 TTFT p95,
5/12 TPOT means, 3/12 TPOT p95, 4/12 E2E means, and 5/12 E2E p95. Therefore V2
is not promoted over V1 as a universal policy.

## Verification

- Release build: `llm_phase_context_smoke` and `unitTestRuntime` passed.
- C++ `Phase*:*Independent*`: 327 run, 326 passed, one optional metadata
  benchmark skipped.
- Python service-normalized tests: 16 passed.
- Replay-contract tests: 15 passed.
- New output and source formatting checks passed after hook rewrites.
- Runtime binary SHA256:
  `c183b3b9bf580528960fd1f5fc7f1c4c6b611133e6699efb97fb9f0dbc4d0d83`.

## What remains intentionally inactive

The requested implementation sequence is complete through real shadow scoring,
alternating-order overhead checks, and a fresh full12 screen. One item cannot be
honestly called complete: a physical same-state alternative branch. The runtime
does not yet checkpoint and restore live KV contents, vision payloads, sampler
state, queues, posterior, and TensorRT bindings. Independent repeats and predicted
shadow alternatives are not that experiment. Note267 documents the required
bounded in-memory checkpoint contract.

Do not grant service-normalized selection authority until a forced equal-work
branch validates the 23 disagreements, especially the 13 suppressed-D choices.
The current evidence says the objective exposes real policy tension, not that its
lexicographic diagnostic is already the best production policy. The next safe
promotion sequence is:

1. implement in-memory quiescent checkpoint/restore for one bounded working set;
2. alternate selected versus shadow branch through reconvergence;
3. measure request-local service, D cohort sequence, GPU work, and ownership
   lifetime;
4. activate only a validated subset in regression workloads;
5. rerun three-repeat full12 and fresh vLLM only if the serving contract changes.
