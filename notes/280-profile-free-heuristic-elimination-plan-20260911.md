# 280. Profile-free heuristic elimination plan

Implementation and validation results are recorded in
[281-profile-free-heuristic-elimination-results-20260911.md](281-profile-free-heuristic-elimination-results-20260911.md).

## Objective

The current V3 scheduler removes explicit serving SLOs and normalizes request
urgency in measured phase-service units, but it is not parameter-free. This
campaign removes policy constants whose meaning changes with the model, engine,
GPU, or arrival process while preserving hard TensorRT and memory capabilities.

The target policy contract is:

```text
legal actions         <- dependency, TensorRT shape, context inflight, memory
observable state      <- ready work, completion events, ownership, service age
physical estimate     <- exact CUDA observations + contextual online model
action selection      <- robust completion, transition value, bounded uncertainty
```

No workload name, post-hoc SLO threshold, or trace-derived parameter may enter
the selector.

## Parameter ledger

### Retained capabilities

These values describe a compiled engine or a finite resource. They are not
scheduler preferences and remain explicit:

- maximum encoder, prefill, and decode binding batch sizes;
- maximum sequence and input lengths;
- fixed prefill chunk of 128 tokens for the current engine contract;
- maximum media count and encoder input supported by the engine;
- KV page size, KV pool budget, vision allocation budget, and safety reserve;
- number of execution contexts and the single-inflight invariant per context.

Changing one of these creates a different engine or memory contract and needs a
separate capacity experiment.

### Policy constants to eliminate

| Existing mechanism | Current constant | Replacement signal | Promotion evidence |
|---|---:|---|---|
| service recovery | age 1.0, band 1.0 | ordinary continuous service age plus a last-resort progress invariant | full-12 no-recovery A/B |
| legacy pair eligibility | Boolean compatibility path | all legal bounded candidates followed by dominance pruning | frontier identity and scheduler-cost gate |
| encoder batch wait | 25 ms | nearest observable completion and measured marginal batch payoff | E1/E2/E4 plus VLM sentinels |
| encoder credit wait | 25 ms, target 4 | queue mass and event-derived producer opportunity | wave-drain, multi-image, vision-heavy |
| vision exclusive threshold | 12,000 tokens | robust predicted overlap outcome and actual outstanding configuration | controlled E+P/E+D matrix |
| TPOT pressure hysteresis | 1.10/0.85 | predicted protected completion and uncertainty interval | text-heavy, mixed, vision-heavy |
| encoded-capacity dwell | 100 ms, backlog 4 | exact byte lease, completion event, and reclaim horizon | constrained-memory pressure trace |
| formation horizon | 4 realized dispatches | request-ready boundaries and trajectory reconvergence | forced branch replay |
| fixed warmup count | 239 text, 319 VLM | feature coverage, uncertainty, and action-rank stability | cold-start/time-to-benefit campaign |

## Staged implementation

### H0: freeze and audit

1. Record one effective runtime-policy manifest for every run.
2. Distinguish engine capabilities, memory budgets, safety invariants, and
   policy parameters in the manifest.
3. Retain the current V3 full-12, fresh vLLM, and output-identity results as the
   frozen comparison.

### H1: remove normal-path service recovery

Run the same V3 binary over all 12 traces with recovery enabled and disabled.
Promote no-recovery only if aggregate throughput and request latency improve
without a starvation or output-identity failure. Recovery may remain temporarily
as a diagnostic ablation, but must not be active by default.

### H2: expose the complete legal P/D frontier

Remove legacy pair-lineage filtering from the policy path. Candidate generation
stays bounded: one urgent, one dense, and one event-refill candidate per phase,
plus legal pair actions. Dominance pruning removes actions that advance no more
work, finish no protected request earlier, release no more ownership, and cost
no less than another candidate.

The selector must remain below 50 microseconds at p95 after cached snapshot and
feature reuse. Until that optimization is complete, the current measured p95 is
reported rather than hidden.

### H3: replace encoder timers with observable-event WAIT

`WAIT` is generated only for a completion event that can make compatible work
ready. Its value compares equal work:

```text
dispatch now + residual work
versus
event ETA + enlarged action + residual work
```

The wait is rejected if its robust completion delays the oldest protected
request more than the best immediate action. No fixed batch target or timer is
used by the selector.

### H4: replace vision pressure and exclusivity rules

E+P and E+D legality remains deterministic. Profitability uses actual or
predicted common-epoch completions, uncertainty, and the deterministic successor
state. An exclusive action is selected only when every feasible overlap action
is dominated or unsafe; input-token thresholds do not decide exclusivity.

### H5: make memory decisions ownership-event driven

Admission and encoded-vision capacity use byte leases, not request-count
watermarks. Each action projects allocation, guaranteed growth, and reclaimable
bytes at already-outstanding completion events. Shrink/grow decisions occur at
ownership transitions, eliminating time dwell and backlog-count hysteresis.

### H6: automatic calibration authority

Generic calibration traverses engine-supported action families until all of the
following hold for the reachable feature region:

- isolated reference coverage exists;
- overlap uncertainty is below the action-rank margin or the conservative action
  already dominates;
- action ordering is stable over consecutive validation windows;
- additional samples change neither the selected action nor its robust rank.

A maximum calibration budget is a safety cap and is reported, not used as the
normal stopping criterion. Zero-start and trace-derived calibration remain
separate research conditions.

### H7: final validation

1. Unit tests for candidate legality, dominance, event-WAIT equal-work
   accounting, ownership transitions, and calibration stopping.
2. Controlled E1/E2/E4/E8 overlap matrix.
3. Three-repeat 12-workload gate with fixed outputs and greedy identity.
4. Saturation and constrained-memory traces.
5. Reuse the frozen vLLM baseline when its contract is unchanged; rerun it only
   if engine, request, memory, or EOS contracts change.
6. Unseen holdout traces and a second GPU/model before claiming generality.

## Promotion gates

Each stage is compared with the immediately preceding stage using the same
binary where possible. A stage is promoted only when:

- fixed-output token identity is unchanged;
- no workload loses more than 3% token throughput unless it has a statistically
  significant joint request-latency improvement;
- geometric-mean token throughput and E2E p95 do not regress;
- starvation, ownership, OOM, and context-inflight invariants remain clean;
- policy decision p95 and candidate counts are recorded;
- an observed gain survives at least three independent runs.

The campaign is intentionally sequential. Combining several removals before a
gate would make a regression impossible to attribute and would recreate the
same heuristic-tuning problem under a new score.
