# Preparation/Formation Separation: P0--P3 Implementation

## 1. Objective

The remaining V3 compatibility behavior is not treated as a search for a
better encoder timer. The architectural target is:

```text
request preparation != E cohort formation != E GPU execution
```

This campaign begins the migration without changing the canonical policy by
default. The old 25 ms formation wait and encoder arbitration remain until the
new mechanism passes the VLM6 and full-12 gates.

## 2. P0 authority audit

The active controls fall into four categories.

| Control | Current authority | Classification | Status |
|---|---|---|---|
| TensorRT E/P/D batch and token limits | engine/config | capability | retain |
| global `maxCandidates=11` | global scheduler | bounded host cost | retain, measure recall |
| stable-slot/page and byte feasibility | ownership mechanism | correctness | retain |
| E 25 ms batch wait | coordinator | policy compatibility guard | unresolved |
| encoder arbiter/max defer | coordinator | policy compatibility guard | unresolved |
| arrival EWMA | coordinator | prediction of open future arrivals | remove after P4/P5 |
| prefix minimum 128 | coordinator submit path | hidden candidate-membership policy | activation telemetry added |
| P overlap token cap | P/D candidate builder | hidden policy unless direct cost overrides it | P7 audit target |
| P cohort/decode burst bounds | P/D scheduler | possible starvation mechanism | measure before removal |

The important distinction is not whether a value is a constant. It is whether
the value constrains a physical capability, preserves a correctness invariant,
or silently decides which candidate the global selector is allowed to see.

New counters expose how often the prefix path is affected:

```text
vision_prefix_plans
vision_prefix_submissions
vision_prefix_threshold_suppressions
```

## 3. P1 preparation characterization

The previous `encoder_gpu_ms` was not a pure encoder-engine duration. Its CUDA
events covered the stream interval from preparation through output readiness,
including any time a prepared batch waited before E submission. This polluted
the physical E cost used by the global policy.

Every request payload now carries three CUDA measurements:

```text
preparation_gpu_ms
    first preparation stream event -> prepared-state stream event

execution_gpu_ms
    E submission stream event -> retained output readiness

gpu_ms
    original preparation -> output envelope, retained for compatibility
```

The batch and final summary telemetry exports all three. Host preparation is
still reported separately by `vision_encoder_preparation_ms`. The preparation
interval can include CPU work, H2D/copy-engine work, CUDA preprocessing,
allocation, and runner binding. Therefore it must not be treated as a free CPU
side action.

The opt-in `TRT_EDGELLM_VISION_SEPARATE_PREPARATION_COST=1` makes online E
action learning use `execution_gpu_ms` after preparation has been separated.
The default preserves the prior whole-envelope cost for compatibility A/B.

Historical diagnostic telemetry also confirms that host placement matters.
The old async activity campaign reported encoder prepare publication medians
of about 1.09 ms (mixed) and 1.01 ms (vision-heavy), while a separate sync
preparation campaign reported about 6.17 ms and 6.08 ms. These campaigns were
not produced by the new binary and are not a causal comparison; they only
justify retaining host preparation as an explicit measured interval.

## 4. P2 persistent lifecycle

Per-batch `std::async` creation was replaced by one coordinator-owned worker.
The worker retains the shared CUDA primary context and has explicit states:

```text
idle -> queued -> running -> prepared
                         \-> failed
```

The state is exported as an additive preparation snapshot in the existing
unified event schema:

```text
stage
request_ids
start/end host timestamps
uses_host_thread
may_use_copy_engine
uses_encoder_stream
holds_device_memory
```

Preparation remains outside the E/P/D execution bitmask. This is deliberate:
it is a side-resource state, not an encoder TensorRT execution. The snapshot
makes that state observable before it is admitted into policy authority.

## 5. P3 independent P/D progress during preparation

The new opt-in switch is:

```text
TRT_EDGELLM_VISION_PREPARATION_PD_DISPATCH=1
```

When disabled, the new worker emulates the previous behavior. When enabled,
an unprepared E batch no longer prevents the global coordinator from
dispatching a ready P/D action. E is not exposed as executable until the
prepared batch is published.

The causal metric is:

```text
vision_encoder_preparation_pd_block_periods
vision_encoder_preparation_pd_block_polls
vision_encoder_preparation_pd_blocked_ms
vision_encoder_preparation_pd_blocked_max_ms
```

It measures an idle P/D dispatch boundary with queued P or D work that the
compatibility preparation rule would suppress. Busy P/D time is not counted as
blocked time.

This mode is experimental because current multimodal `preprocess()` calls can
use the encoder CUDA stream and copy engine. Enabling it can therefore create
real preparation/P/D contention. It must be promoted only after activity and
Nsight traces show that the gained dispatch time exceeds this interference.

## 6. Compatibility contract

All new behavior defaults off except the persistent worker implementation,
which preserves the existing preparation ordering:

```text
allow P/D during preparation = false
separate E action cost       = false
25 ms E formation wait       = unchanged
encoder arbiter              = unchanged
```

This is P5 compatibility emulation for the implemented P0--P3 substrate. A
same-command run must preserve request membership, row ordering, E/P/D action
IDs, output tokens, and performance within the existing repeat variance before
either opt-in is promoted.

## 7. Why P4 is a runner boundary, not a queue-only patch

`PhaseVisionAdapter::prepare()` currently invokes the model-specific
`MultimodalRunner::preprocess()` and `prepareInference()` for a complete batch.
The runner writes reusable member input tensors and TensorRT binding state.
Examples include packed image patches, cumulative sequence lengths, rotary
inputs, and model-specific auxiliary bindings.

Consequently, this unsafe transformation is rejected:

```text
prepare E1 into runner-owned buffers
prepare another E1 into the same buffers
late-pack both as E2
```

The second preparation overwrites the first. Correct late packing requires a
detached input lease owned by each prepared request or prepared group:

```text
prepareHostRequest(request)
    -> formatted text and immutable media metadata

prepareMediaRequest(request, preparation stream)
    -> detached patch/input lease + ready event

materializeBatch(prepared leases)
    -> packed runner input lease and TRT bindings

submitPreparedBatch(batch lease)
    -> E enqueue
```

This API must first be implemented for the shared Qwen/Cosmos runner path and
then either generalized or rejected explicitly by other model runners. Until
that ownership split exists, claiming a general prepared-request pool would be
incorrect.

## 8. Validation completed

Environment:

```text
nvcr.io/nvidia/tensorrt:26.06-py3
TensorRT 11.0.0
CUDA 13.3 build environment
Release build
```

Completed:

- `llm_phase_context_smoke` linked successfully;
- `unitTestRuntime` linked successfully;
- 27/27 `PhaseThreeCoordinatorPolicyTest` passed;
- 11/11 `PhaseUnifiedEventTest` passed;
- a wider CPU-capable phase run passed 340 tests; its nine failures all
  required a CUDA device and reported `no CUDA-capable device is detected`;
- pre-commit checks passed for every modified source/test file.

The initial host-side GPU check was blocked because `/dev/nvidia*` was absent.
GPU access was subsequently restored through the configured NVIDIA container
runtime. The causal GPU validation and its promotion decision are recorded in
`283-preparation-separation-gpu-validation-20260911.md`.

## 9. Original GPU gate sequence

The GPU campaign subsequently executed this sequence:

1. compatibility mode, three repeats on mixed and vision-heavy;
2. preparation timing/resource characterization for E1/E2/E4;
3. enable P/D-during-preparation only, three repeats on wave-drain,
   multi-image, mixed, and vision-heavy;
4. enable separated E cost only, then both switches together;
5. inspect E/P/D/Copy activity masks and Nsight preparation interference;
6. promote only if token identity holds and no VLM6 latency metric regresses
   beyond the established gate;
7. implement detached Qwen/Cosmos prepared-input leases and late packing only
   if the preceding mechanism gates justify the new ownership boundary;
8. reproduce compatibility again on the new lease architecture;
9. replace arrival EWMA, 25 ms wait, and arbiter with prepared-state and
   known-event decisions;
10. audit/remove prefix and P/D local candidate-membership heuristics;
11. rerun full-12 and reuse frozen vLLM only if the HTTP/model/memory contract
    remains identical.

The intended final boundary remains:

```text
deterministic mechanism: dependency, ownership, TRT shape, resource feasibility
global policy: prepared E / P / D / profitable pairs / known-event WAIT
side state: preparation stage and explicit CPU/copy/GPU/memory footprint
```
