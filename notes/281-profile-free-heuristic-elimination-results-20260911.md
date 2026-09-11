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
  v3-profile-free-canonical-full12-3x/
```

All 36 measured runs completed, no run OOMed, and every workload produced a
deterministic fixed-output token trace across its three repetitions.

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
| short | -0.04% | +0.36% | -0.16% | -0.22% | -0.58% | -0.01% | -0.01% |
| balanced | +0.79% | -6.49% | -2.08% | +1.24% | +1.05% | +0.86% | +2.20% |
| decode-heavy | -1.31% | -0.16% | -3.81% | -1.10% | -1.24% | -1.48% | -1.28% |
| long-prefill | -0.15% | +0.18% | -1.06% | +0.57% | +2.40% | -0.06% | +0.09% |
| bimodal | -0.94% | +0.61% | +1.53% | -1.90% | -9.86% | -0.21% | +1.08% |
| text-heavy | +1.61% | -4.46% | -2.53% | +3.79% | +6.99% | +1.19% | +1.59% |
| mixed | +0.21% | +2.43% | +0.89% | -0.82% | +0.38% | +0.07% | +0.34% |
| vision-heavy | -0.15% | +2.72% | +2.89% | -7.60% | +0.19% | -2.56% | -0.07% |
| poisson | +1.84% | +7.48% | +4.35% | +0.91% | +1.67% | +2.41% | +1.53% |
| wave-drain | +0.01% | -0.13% | +0.11% | +0.24% | +0.12% | +0.03% | +0.04% |
| multi-image | -0.83% | +0.11% | -1.04% | -0.35% | -0.15% | -0.34% | -0.92% |
| late-vision | -0.08% | +1.25% | +0.33% | -0.21% | +0.07% | -0.03% | -0.03% |
| geometric mean | **+0.08%** | **+0.38%** | **-0.02%** | **-0.42%** | **+0.15%** | **-0.00%** | **+0.38%** |

The aggregate is performance-neutral. Large single-metric exchanges in
balanced, bimodal, text-heavy, vision-heavy, and poisson show why citable
evaluation must retain all request metrics and three repeats rather than report
only token throughput.

## 8. Canonical V3 versus frozen equal-contract vLLM

vLLM was not rerun because model, request traces, fixed-output/EOS behavior,
concurrency, precision, and memory contract did not change. Positive means the
canonical V3 is better.

| Workload | token/s | TTFT mean | TTFT p95 | TPOT mean | TPOT p95 | E2E mean | E2E p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| short | +15.20% | +48.61% | +23.67% | +6.69% | +19.11% | +20.75% | +16.04% |
| balanced | -0.11% | +42.16% | +36.34% | -5.14% | -6.82% | +0.12% | -2.22% |
| decode-heavy | +1.89% | +42.29% | +47.12% | -0.14% | -1.23% | +1.56% | -0.10% |
| long-prefill | +19.81% | -1.96% | +10.12% | +32.05% | +31.37% | +17.92% | +20.47% |
| bimodal | +4.61% | -21.81% | -59.13% | +25.42% | +31.92% | +8.72% | +0.26% |
| text-heavy | +56.80% | +61.91% | +44.86% | -27.45% | +12.70% | +20.00% | +34.52% |
| mixed | +20.13% | +9.86% | +9.59% | +39.17% | +51.64% | +26.75% | +17.71% |
| vision-heavy | +13.41% | +13.73% | +7.74% | +56.90% | +68.11% | +36.93% | +13.00% |
| poisson | +11.28% | +44.30% | +21.15% | +7.87% | +15.90% | +14.82% | +12.43% |
| wave-drain | +1.94% | -9.24% | +24.96% | +36.74% | +48.72% | +18.48% | +18.72% |
| multi-image | +21.47% | -9.08% | +19.18% | +35.82% | +41.49% | +17.46% | +17.50% |
| late-vision | +14.62% | +46.77% | +46.36% | +11.96% | +11.74% | +16.90% | +12.81% |
| geometric mean | **+14.27%** | **+27.16%** | **+23.01%** | **+21.50%** | **+30.71%** | **+17.27%** | **+14.01%** |

Canonical V3 wins token throughput in 11/12 workloads, TTFT p95 in 11/12,
TPOT p95 in 10/12, E2E mean in 12/12, and E2E p95 in 10/12. It is not a
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
