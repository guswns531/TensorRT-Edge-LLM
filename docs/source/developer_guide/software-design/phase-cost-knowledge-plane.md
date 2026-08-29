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

The design has no external cost registry. Build/fleet bundles are local,
explicitly supplied artifacts, and the node snapshot stays on the serving
node. Filesystem access is not permitted on the dispatch hot path.

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

Portable overlap observations are useful as a timing prior, but they do not
prove that concurrency is profitable on the current node. Controlled warmup
therefore checks exact node-local sample coverage separately. It remeasures a
known portable E+D or P+D point until the local minimum is reached, while
production traffic continues to use the normal slack and probe-interval
guards.

## TTL and drift fallback

Portable fleet and non-exact build timing expires after 30 days by default.
An exact build prior remains tied to its engine/plugin fingerprint and is kept;
startup anchors and runtime drift still guard node-level changes. Node-local
action records expire after seven days.

Outside a controlled calibration epoch, the oracle compares direct CUDA
makespan with the already anchor-scaled prior for the same action key. A bounded
phase-level window uses hysteresis:

```text
8 samples and median relative error > 20% -> phase portable timing local-only
median relative error < 10%              -> portable timing eligible again
fresh controlled phase anchor            -> clear that phase's drift state
```

Local-only affects only E, P, D, or overlap timing for the drifted phase. It
does not disable other phases, discard exact-key local samples, or alter memory
feasibility. If no local estimate exists, the caller reaches its existing
analytical/static fallback. Production observations cannot retune the portable
phase-wide scale.

The asynchronous node snapshot persists the latest drift state. A restarted
process restores it only while it is fresh; stale drift state is ignored so a
temporary interference incident cannot permanently quarantine a prior.

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
TRT_EDGELLM_PHASE_COST_DRIFT_MIN_SAMPLES
TRT_EDGELLM_PHASE_COST_DRIFT_ENTER_RATIO
TRT_EDGELLM_PHASE_COST_DRIFT_EXIT_RATIO
TRT_EDGELLM_PHASE_COST_PRIOR_TTL_HOURS
TRT_EDGELLM_PHASE_COST_NODE_TTL_HOURS
TRT_EDGELLM_PHASE_ENCODER_CALIBRATION_IMAGE
TRT_EDGELLM_PHASE_ENCODER_CALIBRATION_BATCHES
TRT_EDGELLM_PHASE_ENCODER_CALIBRATION_SAMPLES
TRT_EDGELLM_DISABLE_PHASE_ENCODER_DECODE_CALIBRATION
```

Optional hash/signature fields use the `TRT_EDGELLM_PHASE_*_HASH` variables in
the composition root. The production runtime itself does not read environment
variables.

With none of these controls set, the scheduler constructs an in-memory oracle
and preserves the prior behavior.

`TRT_EDGELLM_PHASE_ENCODER_CALIBRATION_IMAGE` opts the IPC composition root
into controlled E and E+D startup calibration. The default shape set is the
power-of-two encoder batches plus the exact deployment maximum. The maximum is
first bounded by the configured encoder capacity and by the image's measured
input-token footprint, so calibration cannot request a shape outside the
visual TensorRT profile. An explicit batch list is validated against that same
physical bound.

Synthetic calibration runs in a temporary three-phase coordinator. It shares
the real execution contexts and cost oracle, then drains every synthetic
request and resets scheduler history before the production coordinator is
created. Consequently CUDA observations and graph state may be retained while
synthetic queue delay, telemetry and request state cannot enter production.
The build-bundle snapshot is written only after both P/D and optional E/E+D
calibration have completed.

## Calibration and offline fleet aggregation tool

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

There is intentionally no uploader, downloader or remote lookup service. A
validated build bundle may be packaged beside the engine. An offline fleet
bundle remains an optional manually generated input for controlled experiments;
production correctness and startup do not depend on it.

## Promotion state and remaining work

The data plane, node persistence, explicit E shape anchors and bounded
controlled-warmup E+D/P+D probes are implemented. None introduces a workload
mode: local artifacts seed GPU action cost, while the live scheduler still
decides from request/DAG/GPU state.

The remaining promotion step is to replace the compatibility composition's
static decode table with a generated build bundle after decision-identity and
the full workload regression gates pass. Verified-idle production probing can
remain opt-in; startup calibration is sufficient for deployments that require
zero exploration on user traffic.
