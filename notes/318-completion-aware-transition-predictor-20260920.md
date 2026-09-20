# 318. Completion-aware transition predictor and dynamic heuristic elimination

Date: 2026-09-20. Branch: `codex/v0101-phase-forward-port`.

## 1. Scope and motivation

Prior to this work, `PhaseQueueScheduler` relied on static heuristics to govern phase interleaving:
- **`decodeBurstLimit = 8`** (hardcoded in `phaseQueueScheduler.h:576`): Fixed limit on consecutive decode batches
  before forcing a prefill turn, regardless of whether a prefill was ready, how long the prefill-to-decode handoff
  took, or current decode queue depth.
- **`maxOverlapPrefillTokens = 128`** (hardcoded in `phaseQueueScheduler.h:442`): Static upper bound on prefill tokens
  admitted into concurrent P+D execution, ignoring live KV cache pressure and actual observed interference.
- **Fixed transition penalty models**: Static delay assumptions without tracking true event makespans or handoff latency.

Under diverse workloads (such as `vision-heavy` and `multi-image`), static burst and overlap limits caused sub-optimal
interleaving: either starving decodes during long vision prefills, or prematurely capping overlap concurrency when the
GPU had sufficient execution slack.

This note documents the implementation and integration of `PhaseTransitionPredictor`: an online learned, Recursive
Least Squares (RLS) transition estimator that replaces hardcoded constants with dynamic, context-aware decisions.

## 2. Architecture and formulation

### 2.1 RLS transition estimator (`PhaseTransitionPredictor`)

`PhaseTransitionPredictor` tracks three distinct transition classes:
1. `kEncoderToPrefill`: Handoff delay from vision encoder completion to prefill queue readiness.
2. `kPrefillToDecode`: Handoff delay from prefill completion to first decode token enqueue.
3. `kActionMakespan`: End-to-end execution makespan for single and concurrent phase actions.

The state is parameterized by a 6-dimensional feature vector $\mathbf{x} \in \mathbb{R}^6$:
$$\mathbf{x} = \begin{bmatrix} 1.0 & \text{queueDepth}/64 & \text{tokens}/1024 & \text{isOverlap} & \text{kvUtil} & \text{contention} \end{bmatrix}^T$$

Parameters $\boldsymbol{\theta}$ and covariance matrix $\mathbf{P}$ are updated online via exponential forgetting RLS:
$$\mathbf{k}_t = \frac{\mathbf{P}_{t-1} \mathbf{x}_t}{\lambda + \mathbf{x}_t^T \mathbf{P}_{t-1} \mathbf{x}_t}$$
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} + \mathbf{k}_t (y_t - \mathbf{x}_t^T \boldsymbol{\theta}_{t-1})$$
$$\mathbf{P}_t = \frac{1}{\lambda} \left(\mathbf{P}_{t-1} - \mathbf{k}_t \mathbf{x}_t^T \mathbf{P}_{t-1}\right)$$

Uncertainty is quantified through running residual variance and feature leverage:
$$\sigma_t = \sqrt{s^2 (1 + \mathbf{x}_t^T \mathbf{P}_t \mathbf{x}_t)}$$
$$\text{UCB}_t = \hat{y}_t + \beta \sigma_t$$

### 2.2 Dynamic decode burst adaptation

Instead of the static `decodeBurstLimit = 8`, the effective burst limit adapts to predicted P->D transition delay
and decode step duration:

$$\text{burst}^* = \begin{cases}
2 & \text{if } \text{decodeQueue} \le 2 \\
\text{clamp}\left(\left\lfloor \frac{\hat{D}_{P \to D}}{\hat{T}_{\text{decode}}} \right\rfloor, 2, 16\right) & \text{if } \hat{D}_{P \to D} > \hat{T}_{\text{decode}} \\
\min(\text{decodeQueue}, 8) & \text{otherwise}
\end{cases}$$

When prefill-to-decode handoff is delayed (e.g. multi-image vision prefill), decode burst expands up to 16 to keep
decode throughput high and prevent idle bubbles. When decode queue is shallow, burst contracts to 2 to prioritize
incoming prefills.

### 2.3 Dynamic overlap prefill token bounding

Instead of the static `maxOverlapPrefillTokens = 128`, the effective overlap token budget adapts dynamically:
- **Throttles to 64 tokens**: If decode queue length exceeds 16 or predicted P->D delay exceeds 1,000 µs, minimizing
  interference on latency-critical decodes.
- **Expands to 256 tokens**: If decode queue is empty and handoff delay is minimal (< 100 µs), maximizing prefill progress.
- **Defaults to 128 tokens**: Under balanced queue conditions.

## 3. Integration into `PhaseQueueScheduler`

1. **Config & Lifecycle**:
   - `PhaseQueueSchedulerConfig::enableTransitionPredictor` (default `true`, overridable via `TRT_EDGELLM_ENABLE_TRANSITION_PREDICTOR`).
   - `PhaseQueueSchedulerConfig::transitionPredictorConfig`.
   - `mTransitionPredictor.reset()` integrated into `resetPolicyPosterior()` and `resetHistory()`.
2. **Telemetry Ingestion**:
   - In `PhaseQueueScheduler::observeMetrics()`, observed `makespanGpuMs`, `decodeQueueWaitUs`, and `prefillQueueWaitUs`
     are fed into `mTransitionPredictor.observe()` after each CUDA dispatch.
3. **Decision Points Updated**:
   - `defaultDecision()`: Uses `effectiveDecodeBurstLimit(state)` and `effectiveOverlapPrefillTokens(state)`.
   - `metricsDecision()`: Uses `effectiveDecodeBurstLimit(state)`.
   - `applyExternalDrainPreference()`: Uses `effectiveDecodeBurstLimit(state)`.
   - `selectPrefillBatchSize()`: Uses `effectiveOverlapPrefillTokens(state)`.
   - `next()`: Evaluates `overlapEvaluatedByCost` against `effectiveOverlapPrefillTokens(state)`.

## 4. Verification

- **Unit tests**:
  - `PhaseTransitionPredictorTest` (6 tests in `unittests/cpp/runtime/scheduling/phaseTransitionPredictorTest.cpp`):
    - `InitialState`, `LearnsEncoderToPrefillDelay`, `LearnsPrefillToDecodeDelay`,
      `UncertaintyDecreasesWithObservations`, `DynamicDecodeBurstAdaptation`, `DynamicOverlapPrefillTokens`.
  - `PhaseQueueSchedulerTest.AdaptsDecodeBurstAndOverlapWithTransitionPredictor` (in `phaseQueueSchedulerTest.cpp`):
    - Verified observation ingestion, dynamic burst/overlap adaptation, and clean reset under `resetPolicyPosterior()`.
- **Regression test suite**:
  - All 154 existing `PhaseQueueSchedulerTest` cases pass with 0 regressions.
  - Full suite passes: 160/160 tests clean.
