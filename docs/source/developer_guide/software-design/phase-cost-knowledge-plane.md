<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Phase cost knowledge plane

## Goal

The phase scheduler must not depend on workload labels or a single machine's
hard-coded latency table. It consumes action-shape cost with conservative
uncertainty while request slack, ready state, ownership and completion events
remain the policy inputs.

```text
build CUDA calibration ──┐
                         ├─> PhaseCostOracle ─> Global phase-action scheduler
fleet aggregate ─────────┤          ^
                         │          │
node snapshot ───────────┘     direct CUDA events
                                      │
                                      └─> async node journal
```

No registry or filesystem access is permitted on the dispatch hot path.

## Portable bundle

`PhaseCostBundle` contains five independent contracts:

1. `PhaseDeploymentFingerprint`: model, ONNX, engine, external-weight and
   plugin hashes plus GPU/software signatures.
2. `PhaseCostCapability`: P/D/E profile bounds, fixed chunk, KV page facts and
   CUDA Graph shapes. These facts are not learned.
3. `PhaseGlobalActionKey`: E/P/D or overlap shape, context buckets and
   eager/graph execution variant.
4. bounded raw `(reference_work_ms, makespan_ms)` observations, rather than an
   average of node p95 values;
5. source and version provenance (`build`, `fleet`, or `node`).

GPU UUID is retained only as provenance. It is not a compatibility key.

## Compatibility

| grade | requirement | use |
|---|---|---|
| exact | engine/plugin/software/GPU signature and shape contract match | absolute prior |
| compatible | shape contract and GPU compute capability match | scaled prior, wider uncertainty |
| shape-only | shape contract matches on another GPU architecture | high-uncertainty prior |
| incompatible | model, precision, KV dtype or profile contract differs | reject |

Unknown fingerprint fields are not treated as proof of equality. Production
deployment should provide engine and plugin hashes to permit exact node-cache
restore.

## Oracle order and safety

The oracle selects estimates in this order:

```text
at least four node-local samples
  -> compatible fleet prior
  -> build prior
  -> sparse node-local observation
  -> existing analytical/static fallback in the caller
```

Compatibility widens uncertainty, and phase-specific anchor scales can be
injected independently for encoder, prefill, decode and overlap. Direct CUDA
observations collected during controlled startup warmup also update those four
scales automatically. The estimator uses a bounded median ratio window and
adds its robust dispersion to prediction uncertainty; implausible ratios are
discarded. Until the minimum sample count is reached the scale remains one.
Unknown overlap remains ineligible unless the existing safe-probe contract
permits a bounded calibration probe.

Memory feasibility is unchanged. Predicted completion or reclaim time may
rank actions but cannot create hard memory capacity.

## Node persistence

`PhaseNodeCostJournal::enqueue()` only copies an observation into a bounded
host queue. A background worker:

- appends JSON lines to `observations.journal`;
- retains at most the configured raw samples per key;
- periodically writes `local_cost_snapshot.json.tmp` and atomically renames it;
- drops observations instead of blocking when its host queue is full;
- contains filesystem failures without terminating serving.

The engine directory remains immutable. Node state belongs under a separate
writable cache directory.

## Runtime wiring

`PhaseQueueScheduler` and `PhaseThreeCoordinator` now receive the same shared
`PhaseCostOracle`. This combines E, P, D, E+P, E+D and P+D observations without
sharing scheduler policy or request state.

The phase server composition root accepts:

```text
TRT_EDGELLM_PHASE_BUILD_COST_BUNDLE
TRT_EDGELLM_PHASE_FLEET_COST_BUNDLE
TRT_EDGELLM_PHASE_NODE_COST_CACHE
TRT_EDGELLM_PHASE_COST_SNAPSHOT_SAMPLES
TRT_EDGELLM_PHASE_COST_SCALE_ENCODER
TRT_EDGELLM_PHASE_COST_SCALE_PREFILL
TRT_EDGELLM_PHASE_COST_SCALE_DECODE
TRT_EDGELLM_PHASE_COST_SCALE_OVERLAP
TRT_EDGELLM_PHASE_COST_ANCHOR_MIN_SAMPLES
TRT_EDGELLM_DISABLE_PHASE_COST_AUTO_ANCHOR
TRT_EDGELLM_PHASE_WRITE_BUILD_COST_BUNDLE
```

Optional hash/signature fields use the `TRT_EDGELLM_PHASE_*_HASH` variables in
the composition root. The production runtime itself does not read environment
variables.

With none of these controls set, the scheduler constructs an in-memory oracle
and preserves the prior behavior.

## Calibration and fleet aggregation tool

`scripts/phase_cost_bundle.py` provides three offline operations:

```text
fingerprint  engine/config/plugin -> deployment fingerprint
build        PHASE_METRIC records -> bounded build bundle
aggregate    compatible bundles -> bounded fleet bundle
```

Only records with an applied Global action and positive CUDA reference/makespan
are imported. Encoder batch metrics are imported as isolated E observations.
Context and execution-variant keys match the runtime's cost keys.

The aggregate command rejects heterogeneous model/precision/KV/profile shape
contracts. It merges bounded raw observations instead of averaging per-node
quantiles.

The phase server can also write the direct controlled-warmup observations as a
build bundle with `TRT_EDGELLM_PHASE_WRITE_BUILD_COST_BUNDLE`. This avoids
parsing production logs and preserves the exact runtime action keys. The output
path must be outside the immutable engine directory in normal deployment; the
artifact pipeline may copy the validated bundle into the engine package later.

## Remaining work

The first implementation establishes the data plane, persistence boundary and
offline fleet merge. The following are intentionally separate promotion
steps:

1. replace the compatibility composition's static decode table with a
   generated build bundle after decision-identity validation;
2. extend startup coverage with explicit E anchors when the normal VLM warmup
   does not exercise every encoder shape;
3. add TTL/drift state and local-only fallback when recent observations exceed
   a robust prior bound;
4. schedule uncertainty-guided overlap probes only during controlled warmup or
   verified idle windows;
5. connect upload/download transport to a versioned external registry. Remote
   access must remain outside inference.

These steps do not introduce workload modes. The registry distributes GPU
action cost; the live scheduler still decides from request/DAG/GPU state.
