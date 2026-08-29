# Process-local phase cost tracking

The global phase scheduler adapts to measured GPU service costs without classifying the workload or persisting a
trained policy. One `PhaseRuntimeCostTracker` is shared by the encoder, prefill, and decode coordinators for the
lifetime of a serving process.

## Decision loop

Each scheduling iteration follows one feedback loop:

1. Harvest completed CUDA events.
2. Validate the completed action ticket against the actual outstanding execution set.
3. Add valid E, P, D, E+D, or P+D timings to bounded process-local sample windows.
4. Apply request-DAG transitions and release completed ownership.
5. Build a bounded set of deterministic phase candidates from the current queues.
6. Reject candidates that violate dependencies, TensorRT shapes, context ownership, or memory feasibility.
7. Protect first-token and next-token deadlines.
8. Select among the remaining candidates by robust measured service cost and dispatch it.

The newly completed observation is available to the next scheduling decision. There is no background training or
periodic policy update.

## Observation keys

Action timings are separated by action kind, batch sizes, chunk length, context buckets, producer class, and eager or
CUDA-graph execution variant. Decode component timings additionally distinguish encoder and prefill contention. This
prevents a contended D measurement from replacing an isolated D measurement while keeping all observations in one
tracker.

Only a plan-bound observation is accepted: the selected candidate, actual outstanding context set, execution shape,
and completion metrics must agree. A fidelity violation remains telemetry and does not update the cost window.

## Confidence and cold start

Each action key is in one of three states:

- `Unknown`: no direct observation; use the conservative configured fallback.
- `Warming`: observations exist but have not reached the common confidence threshold.
- `Ready`: at least the configured number of observations is available.

Serial E/P/D warmup populates common shapes before production traffic. Unknown overlap remains ineligible except for a
bounded safe probe with sufficient request slack. Overlap becomes eligible only after its direct sample threshold and
only when robust isolated-work compression exceeds the configured minimum gain.

Measured P/D batch formation is an opt-in evaluation path and requires a `Ready` estimate. The production default
keeps the deterministic candidate builder and its configured costs unchanged. The global selector may use a warming
estimate with its uncertainty margin, so early measurements cannot silently acquire mechanism authority.

## Replacement by observation count

Action and decode-component histories are fixed-size sample windows. New observations evict the oldest observations by
count. This reacts to clock, thermal, and contention changes without wall-clock expiration.

The process starts with empty histories and warmup repopulates them. Runtime costs are not loaded from or written to a
fleet registry, build bundle, or node journal. Consequently the runtime has no TTL, startup anchor, drift quarantine,
or promotion metadata.

## Responsibility boundaries

- Candidate builders own compatible batching and canonical row order.
- The memory broker owns KV and vision allocation feasibility and lifetime accounting.
- The runtime cost tracker owns recent CUDA timing statistics only.
- The global selector owns serial, overlap, and bounded-WAIT policy.
- The executor owns action tickets, CUDA events, and action-fidelity validation.

Correctness and feature availability are release properties rather than runtime cost metadata. Changes are promoted by
the project regression suite using throughput, E2E, TTFT, TPOT, output correctness, action fidelity, and memory-safety
gates across the shared workload set.
