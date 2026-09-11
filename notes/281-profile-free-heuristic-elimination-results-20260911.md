# 281. Profile-free heuristic elimination results

## Outcome

This campaign audited every remaining V3 serving-policy constant against the
same Cosmos-Reason2-2B, RTX 3080, fixed-output HTTP contract. It did not remove
an option merely because it looked heuristic. Each option was classified as a
capability, inactive compatibility setting, promotable policy removal, or an
unresolved guard whose proposed replacement failed its workload gate.

The promoted V3 contract is simpler without losing aggregate performance:

```text
removed from the normal path
  implicit service recovery
  legacy P/D pair eligibility
  encoder credit timer and target
  vision input-token exclusivity threshold
  no-SLO TPOT hysteresis, capacity dwell, and backlog triggers
  fixed warmup request count

retained as engine/resource capabilities
  E/P/D maximum binding batches
  P128 compiled chunk contract
  maximum input/media/sequence sizes
  KV and vision byte capacity

retained pending a better replacement
  25 ms encoder formation safety cap
  encoder preparation arbitration
```

The three-repeat canonical full-12 campaign is under:

```text
.local/results/v0101-forward-port/heuristic-elimination-20260911/
  v3-profile-free-clean-full12-3x/
```

All 36 measured runs completed, no run OOMed, and every workload produced a
deterministic fixed-output token trace across its three repetitions.
The campaign manifest records source commit `ae36f65`, a clean worktree, and
runtime executable SHA-256 `6f1b066a...e6ddefd`.

## 1. Implementation

### 1.1 Recovery is opt-in

`PhaseGlobalSchedulerConfig::enableServiceRecovery` is now false by default.
Selecting the service-scaled V3 representation no longer silently enables the
old age-1.0/band-1.0 recovery filter. The diagnostic environment variable is
now `TRT_EDGELLM_ENABLE_SERVICE_RECOVERY=1`; the old disable variable remains
accepted only so retained commands can be replayed safely.

This separates two ideas that had been conflated:

```text
service-scaled state representation  -> retained
fixed recovery threshold policy      -> disabled by default
```

### 1.2 No-SLO vision capacity bypasses SLO hysteresis

When the service-scaled policy has neither an explicit request SLO nor an
explicit vision TTFT contract, encoded-vision capacity is selected directly
from the byte-safe configured capacity. Decode-TPOT pressure ratios, dwell
timers, and backlog counts no longer change it. Explicit-SLO modes retain the
old contract-aware behavior.

The current engine has the same byte-safe base and high count capacity, so this
is a semantic simplification rather than the source of a performance change.
Exact KV/page and vision byte ownership checks remain authoritative.

### 1.3 Calibration stops on runtime authority

The replay harness now reads the complete generic calibration program and uses
one complete program as a convergence observation. It stops after the first
cycle whose active contextual model reports all required directions ready. A
cycle cap is only a failure-safety bound.

The stable signature contains decision authority and readiness, not sample
counts. Sample growth therefore cannot force useless extra cycles after the
active model has converged. Exact-key readiness is used only when the active
policy has not reached contextual authority.

The canonical campaign observed:

| Workload class | Program size | Normal stop | Observed extension |
|---|---:|---:|---:|
| text-only | 239 requests | 239 | none in 15 runs |
| VLM | 319 requests | 319 | mixed extended to 638 in 1 of 3 runs |

This is model/GPU calibration, not workload-specific training. The measured
trace starts only after calibration ends.

### 1.4 Reproducible campaign manifests

Every replayed campaign now records source commit and dirty state, build cache
and executable hashes, base command hash, policy/cases/repeats, explicit-SLO
state, calibration contract, telemetry level, and added/removed runtime
settings. Retained environment settings can be removed without editing an old
command file in place.

### 1.5 Additional observability

The final vision summary reports encoder-arrival wait periods, expirations, and
the last predicted wait. This makes the remaining E-formation guard measurable
instead of inferring it from latency changes.

## 2. H1: service recovery

Recovery OFF versus a fresh same-day recovery-ON V3 full-12 campaign produced
the following geometric-mean changes. Positive means recovery OFF is better.

| token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---:|---:|---:|---:|---:|---:|---:|
| +3.74% | -0.21% | +4.40% | +4.37% | +7.54% | +2.56% | +3.19% |

Recovery was not a rare last-resort invariant; it altered normal operation and
hurt the aggregate. It is therefore no longer implicit in V3.

## 3. H2: legal frontier and legacy eligibility

Expanding the whole candidate frontier and removing legacy eligibility in one
step gave only +0.11% token/s and +0.37% E2E-p95 geometric-mean improvement,
but caused workload-local tail exchanges. The split ablation showed:

- removing legacy eligibility alone is near aggregate parity;
- global frontier expansion causes most of the multi-image and text tail
  volatility;
- bounded candidate generation remains necessary; legal does not mean useful
  enough to materialize on every decision.

Legacy eligibility is absent from the new canonical command, but unrestricted
frontier expansion was not promoted. Feasibility remains mechanism-only while
candidate materialization stays bounded.

## 4. H3--H4: encoder timers, arbitration, and exclusivity

### 4.1 Inactive settings

The 12,000-token vision exclusivity threshold never fired in the retained VLM
campaigns:

```text
exclusive_batches=0
exclusive_prefill_deferrals=0
```

Removing it changed no action and exposed run-to-run trajectory variance. The
encoder credit wait/target is also inactive when the global scheduler owns the
decision. These settings were removed from the canonical command.

### 4.2 Failed dynamic replacement

A proposed replacement removed both the 25 ms encoder wait cap and old encoder
arbiter. It waited only when the observed interarrival ETA plus predicted
`E(batch+1)->P` horizon beat dispatch-now plus the residual singleton horizon.
It used no workload label or explicit SLO.

Against the fixed-wait/arbiter canonical VLM6 result, the three-repeat change
was:

| token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---:|---:|---:|---:|---:|---:|---:|
| -0.98% | +0.39% | -0.58% | -3.92% | -10.54% | +0.22% | -1.19% |

The mean TTFT gain came mainly from wave-drain and multi-image, but their TPOT
tails regressed by approximately 42% and 22%. Two one-repeat split experiments
showed that neither component can simply be deleted:

| Variant | wave TPOT p95 | multi-image TPOT p95 | Interpretation |
|---|---:|---:|---|
| automatic wait, arbiter retained | -30.3% | -39.0% | arrival prediction alone is insufficient |
| fixed 25 ms wait, arbiter removed | -32.3% | -24.4% | preparation placement protects D continuity |
| automatic wait, arbiter removed, 3x | -42.0% | -21.5% | errors compound |

Negative values mean worse latency. The automatic replacement was reverted.
The remaining two guards are explicitly recorded as unresolved compatibility
guards, not claimed as profile-free policy. Their correct replacement requires
an E-formation action inside the global selector and a bounded event-derived D
continuity cost, rather than prediction of an unobserved future arrival.

## 5. H5: ownership-driven capacity

The current no-SLO contract never activated the old TPOT-pressure, dwell, or
backlog transitions. They are now bypassed in the no-SLO service-scaled path.
This does not weaken memory safety:

```text
hard feasibility = current bytes + guaranteed growth <= byte budget
ranking signal   = immediate/near reclaim at known completion events
```

Count capacity remains because the TensorRT/KV engine has a finite number of
stable slots. It is a resource capability, not a workload policy.

## 6. H6: automatic-calibration validation

An initial two-stable-cycle implementation incorrectly treated expanding exact
key sets as instability and ran 957 requests. The authority-specific signature
fixed this: all four VLM screen workloads stopped after 319 requests. The final
full-12 campaign then demonstrated the 239/319/638 adaptive behavior described
above.

The former fixed warmup counts are therefore removed from the experiment API's
decision contract. The complete generic program still has a concrete size, as
any finite calibration workload must, but the number of repetitions is chosen
from measured convergence.

## 7. Canonical V3 versus fresh same-day V3

The comparison below is against the fresh same-day no-recovery full-12 control.
Positive means the simplified canonical is better.

| Workload | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | -0.74% | -1.79% | -1.34% | -0.38% | +11.66% | -1.93% | -0.90% |
| balanced | +0.34% | -4.58% | -2.90% | +0.71% | +0.96% | +0.27% | +1.59% |
| decode-heavy | -0.16% | +4.99% | -1.58% | +0.03% | -0.37% | -0.19% | +0.29% |
| long-prefill | -0.12% | -0.77% | -2.03% | +1.37% | +2.06% | +0.07% | +1.14% |
| bimodal | -0.43% | -1.62% | +3.58% | +1.38% | +7.18% | -0.95% | +1.27% |
| text-heavy | -0.78% | -9.00% | +1.87% | +2.78% | +2.30% | -0.78% | -0.64% |
| mixed | -0.83% | +2.38% | -1.05% | -1.75% | +1.04% | -0.44% | -0.75% |
| vision-heavy | -0.95% | -1.56% | -1.64% | -6.00% | +1.92% | -1.37% | -1.15% |
| poisson | +2.74% | +8.84% | +5.16% | +0.58% | +0.55% | +3.75% | +3.01% |
| wave-drain | -0.00% | -0.14% | -0.01% | -0.09% | -4.44% | -0.11% | -0.02% |
| multi-image | +0.79% | +0.06% | +1.08% | -0.19% | +2.68% | +0.46% | +0.75% |
| late-vision | +0.42% | +1.39% | +0.51% | +0.19% | +0.11% | +0.34% | +0.40% |
| geometric mean | **+0.02%** | **-0.06%** | **+0.17%** | **-0.09%** | **+2.21%** | **-0.06%** | **+0.42%** |

The aggregate is performance-neutral. Large single-metric exchanges in
balanced, bimodal, text-heavy, vision-heavy, and poisson show why citable
evaluation must retain all request metrics and three repeats rather than report
only token throughput.

The clean campaign also reproduced the pre-commit validation campaign within
-0.06% token/s and +0.04% E2E-p95 geometric mean. The clean manifest, rather
than the dirty validation manifest, is the retained citable result.

## 8. Canonical V3 versus frozen equal-contract vLLM

vLLM was not rerun because model, request traces, fixed-output/EOS behavior,
concurrency, precision, and memory contract did not change. Positive means the
canonical V3 is better.

| Workload | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | +14.40% | +47.50% | +22.77% | +6.54% | +28.95% | +19.23% | +15.30% |
| balanced | -0.56% | +43.20% | +35.83% | -5.71% | -6.92% | -0.48% | -2.86% |
| decode-heavy | +3.08% | +45.25% | +48.25% | +0.98% | -0.37% | +2.80% | +1.45% |
| long-prefill | +19.84% | -2.93% | +9.26% | +32.60% | +31.13% | +18.03% | +21.31% |
| bimodal | +5.14% | -24.55% | -55.81% | +27.82% | +42.48% | +8.05% | +0.46% |
| text-heavy | +53.12% | +60.26% | +47.23% | -28.78% | +8.30% | +18.41% | +33.04% |
| mixed | +18.87% | +9.81% | +7.82% | +38.61% | +51.96% | +26.38% | +16.81% |
| vision-heavy | +12.50% | +9.93% | +3.43% | +57.54% | +68.66% | +37.66% | +12.06% |
| poisson | +12.25% | +45.12% | +21.82% | +7.56% | +14.94% | +15.99% | +13.75% |
| wave-drain | +1.92% | -9.25% | +24.88% | +36.53% | +46.38% | +18.37% | +18.67% |
| multi-image | +23.44% | -9.13% | +20.88% | +35.92% | +43.14% | +18.12% | +18.87% |
| late-vision | +15.20% | +46.85% | +46.45% | +12.32% | +11.77% | +17.21% | +13.18% |
| geometric mean | **+14.20%** | **+26.84%** | **+23.16%** | **+21.75%** | **+32.14%** | **+17.22%** | **+14.04%** |

Canonical V3 wins token throughput in 11/12 workloads, TTFT p95 in 11/12,
TPOT p95 in 10/12, E2E mean in 11/12, and E2E p95 in 11/12. It is not a
pointwise latency dominator: balanced/decode-heavy TPOT and bimodal TTFT remain
the clearest gaps.

## 9. Final parameter classification

| Item | Status | Reason |
|---|---|---|
| service recovery age/band | default removed | full-12 aggregate regression |
| legacy pair eligibility | canonical removed | near aggregate parity; bounded frontier remains |
| E credit wait/target | canonical removed | inactive under global ownership |
| vision exclusivity threshold | canonical removed | zero activations in retained workloads |
| no-SLO TPOT hysteresis/dwell/backlog | bypassed | no explicit contract; byte safety is sufficient |
| fixed warmup repetitions | replaced | runtime authority controls stop |
| E formation 25 ms cap | retained guard | automatic interarrival replacement failed TPOT gate |
| encoder arbiter | retained guard | independent removal failed wave/multi TPOT gate |
| E/P/D maximum batches | retained capability | TensorRT profile/binding limit |
| P128 chunk | retained capability | compiled/validated execution contract |
| stable slots/KV pages/byte budgets | retained capability | hard memory feasibility |
| formation realized-dispatch horizon | bounded mechanism | must be evaluated by ready-boundary rollout later |

## 10. Validation

- Python replay/calibration contract tests: 21 passed.
- C++ scheduler tests: 221 passed across `PhaseGlobalSchedulerTest`,
  `PhaseQueueSchedulerTest`, and `PhaseThreeCoordinatorPolicyTest`.
- Release targets `llm_phase_context_smoke` and `unitTestRuntime` built in
  TensorRT 11.0/CUDA 13.3 container.
- Canonical full-12: 36/36 measured runs complete and token deterministic.
- No vLLM rerun: equal-contract frozen result reused as required by the local
  research workflow.

## 11. Next work

The next change should not tune the retained 25 ms or arbiter constants. It
should move E formation/preparation into the same action semantics as P/D:

1. represent `E-now`, `prepared-E`, and an event-backed `WAIT` as global
   candidates;
2. wait only on already-outstanding preparation or GPU completion events, not
   on predicted future network arrivals;
3. add a decode-continuity terminal cost using immutable service references;
4. run forced wave/multi counterfactuals to verify that the model predicts D
   fragmentation before enabling it;
5. promote only after VLM6 three-repeat parity and the full-12 gate;
6. evaluate a second model/GPU before calling the remaining controller
   generally profile-free.

Until that work passes, the honest architecture claim is:

> V3 is workload-label-free and explicit-SLO-free in its active global P/D
> policy, with automatic model/GPU calibration. Two E preparation/formation
> compatibility guards remain and are disclosed rather than hidden as learned
> behavior.
