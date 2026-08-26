# Global phase-action scheduler

## Scope

The global phase-action scheduler is an opt-in policy layer for one CUDA primary context with independent TensorRT
execution contexts and CUDA streams for vision encoding (E), prefill (P), and decode (D). It does not classify a workload
as latency-, throughput-, or vision-oriented. It selects from current ready work, request slack, measured GPU cost, and
persistent-memory ownership.

The initial bounded action set is:

- `E`, `P`, or `D`;
- `E+P`, `E+D`, or `P+D` when the overlap cost is known or a safe probe is permitted; and
- `WAIT` for a concrete outstanding sampling CUDA event.

Triple overlap remains deliberately excluded. TensorRT optimization profiles remain shape-support mechanisms; they are
not serving workload profiles.

## Architecture

```text
request DAGs + E/P/D ready queues + CUDA completions + stable ownership
                              |
                              v
                     bounded candidates (<= 11)
                              |
                              v
 dependency/context/shape/memory feasibility
                              |
                              v
       request deadline violation (median + uncertainty guard)
                              |
                              v
 GPU service compression + near reclaim + request service lag
                              |
                              v
             serial / overlap / concrete WAIT dispatch
```

`PhaseThreeCoordinator`, `IndependentPhaseAsyncServer`, and `PhaseQueueScheduler` retain transition, completion, batching,
and admission mechanisms. `PhaseGlobalScheduler` owns policy when active. This separates correctness from optimization.

Three invariants are checked before ranking:

1. P cannot consume an unfinished E dependency.
2. Persistent storage cannot be reclaimed until every GPU consumer has completed.
3. At most one enqueue may be outstanding on each TensorRT execution context.

If no candidate is hard-feasible, active mode dispatches nothing. It never falls through to an unsafe legacy decision.

## Cost learning and bounded exploration

Each action key contains action kind, batch sizes, chunk length, and context buckets. A bounded sample window records CUDA
event makespan and isolated reference work. Selection uses median cost and an uncertainty guard equal to the larger of the
p95-minus-median spread and a cold-start margin that shrinks with sample count.

Unknown overlap is not generally eligible. Production-request exploration is disabled by default. A controlled warm-up
may enable a safe probe only when protected request slack exceeds a configured multiple of predicted serial work and the
probe interval has elapsed. After enough samples, overlap remains eligible only when its robust makespan provides the
configured gain over isolated work.

P/D CUDA observations use dispatch start/end events. E uses encoder CUDA events. E+P and E+D preserve the matching P/D
duration even when another dispatch completes before E, then record the maximum of the two adjacent stream spans as the
cross-stream makespan estimate. A shared-origin event is the next measurement refinement if sub-millisecond launch skew
matters.

An unknown safe probe is ranked using `max(E, P/D)` as its optimistic makespan while the difference to the robust serial
upper bound is retained as uncertainty. Therefore deadline feasibility stays conservative without making exploration
impossible. The vision encoder cost JSON may optionally contain direct `encoder_prefill` and `encoder_decode` tables;
otherwise only the slack-gated, rate-limited online path can bootstrap those shapes.

## Stable ownership and memory horizon

Candidates carry stable request IDs. The memory supplier resolves those IDs to actual request-owned vision payloads and KV
leases rather than queue-row positions.

Hard feasibility uses only:

```text
managed - observed immediate reclaim + candidate allocation + guaranteed growth <= budget
```

Expected near-term reclaim never creates hard capacity. It is only an efficiency tie-breaker. For P and P+D, vision
embedding/deepstack bytes become near reclaim when prefill-storage release is enabled. Arithmetic saturates rather than
wrapping at `size_t` limits.

Admission remains subject to stable page reservations. In active mode, an available covered decode p95 curve also limits
the active cohort to the largest batch that fits the request TPOT target. Static workload labels and fixed C16/C80 serving
profiles do not participate in the active decision.

## WAIT and host progress

`WAIT` is an explicit action, not a fixed sleep. The scheduler compares dispatch-now with:

```text
predicted event wait + predicted denser decode step
```

and accepts it only after the same feasibility and deadline checks. The current implementation creates WAIT candidates for
outstanding sampling CUDA events. Prefill-formation and encoder-credit timers remain legacy/shadow fallbacks and are bypassed
by active mode.

Completion polling and dispatch are separate operations. Empty queues are not evaluated, and an already in-flight E asks
for a decode-only preview because a new E+P action can only be launched when both contexts are idle. This keeps one global
decision per applied P/D dispatch rather than spinning on an action that cannot be launched.

## Modes and observability

`globalSchedulerMode` supports:

- `disabled`: unchanged legacy behavior;
- `shadow`: evaluate and count disagreements while preserving legacy dispatch; and
- `active`: apply the global decision.

The real-request harness exposes the same choices through `TRT_EDGELLM_GLOBAL_SCHEDULER=disabled|shadow|active`. Phase
metrics include selected actions, decisions, applied decisions, shadow disagreements, safe probes, WAIT decisions,
deadline violation, service compression, E+P/E+D action counts, CUDA group times, and stable request IDs.

## Initial real-request gate

The gate used `nvidia/Cosmos-Reason2-2B`, RTX 3080 10 GB, P8/D16 runtime caps, E4, fixed prefill chunk 128, 16 requests
(eight 256-token text decodes followed by eight real-image requests), and three fresh processes per variant. All runs
generated 2,056 tokens with the same token-ID trace SHA-256.

| system | generated token/s | text TPOT p95 | text E2E p95 | vision TTFT p95 | peak VRAM |
|---|---:|---:|---:|---:|---:|
| legacy disabled | 974.04 | 8.171 ms | 2110.31 ms | 478.09 ms | 9659 MiB |
| global active | 972.77 | 8.178 ms | 2113.15 ms | 530.91 ms | 9623 MiB |
| vLLM, same trace | 906.92 | 8.779 ms | 2266.74 ms | 584.73 ms | not captured |

Global active was within 0.2% of legacy for throughput, text TPOT, and text E2E, while using 36 MiB less peak memory in
this sample. Relative to the same-trace vLLM run, it delivered about 7.3% more generated tokens/s, 6.8% lower text TPOT
p95, 6.8% lower text E2E p95, and 9.2% lower vision TTFT p95.

Vision TTFT p95 regressed about 11.0% versus legacy in this initial gate, which motivated the E+P action. A controlled E4
+ P2 safe probe subsequently measured P at 121.84 ms instead of its roughly 34 ms uncontended value. The overlap
makespan was only slightly below the serial upper bound while the older P request completed much later. E+P is therefore
supported but remains direct-cost and slack gated. Production-request probing is disabled by default after the broader
gate showed that even one apparently slack-safe probe can dominate a small workload's tail.

The broader 12-workload gate found that active scheduling is not ready for default promotion. It retained exact token
identity for all five text-only traces and one late-vision trace, but decode-heavy throughput regressed because global
WAIT/refill formed many more, smaller decode cohorts. Mixed/VLM outputs remained semantically correct but were not exact
cross-policy token traces. The next promotion requirement is a completion-aware decode refill action that recovers legacy
cohort density before further overlap exploration.
