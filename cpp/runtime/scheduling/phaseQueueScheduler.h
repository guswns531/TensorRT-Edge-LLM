/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "runtime/phase/cost/phaseRuntimeCostTracker.h"
#include "runtime/phase/mechanism/phaseReadySnapshot.h"
#include "runtime/phase/policy/phaseGlobalScheduler.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Per-request policy hints. Zero SLO targets inherit the scheduler defaults.
struct PhaseSchedulingHints
{
    int32_t priority{};
    double ttftTargetUs{};
    double tpotTargetUs{};
    //! Original host admission time. An empty value is filled on first enqueue.
    std::chrono::steady_clock::time_point submittedAt;
};

//! Logical prefill producer class. TensorRT execution may remain shared while
//! queue compatibility and profiled costs stay producer-specific.
enum class PhasePrefillClass
{
    kAny,
    kText,
    kExternal,
};

//! A unit of phase work. For prefill, tokenCount is the remaining prompt
//! length while queued and the dispatched chunk length while in flight.
//! For decode it is the current KV length. kvSlotId identifies stable physical
//! cache ownership; tokenOffset and promptTokenCount describe chunk progress.
struct PhaseWorkItem
{
    PhaseWorkItem() = default;
    PhaseWorkItem(uint64_t requestId, int32_t tokenCount, int32_t kvSlotId = -1, int32_t tokenOffset = 0,
        int32_t promptTokenCount = 0, bool allowChunkedPrefill = true, PhaseSchedulingHints scheduling = {},
        bool exclusivePrefill = false, PhasePrefillClass prefillClass = PhasePrefillClass::kText)
        : requestId(requestId)
        , tokenCount(tokenCount)
        , kvSlotId(kvSlotId)
        , tokenOffset(tokenOffset)
        , promptTokenCount(promptTokenCount)
        , allowChunkedPrefill(allowChunkedPrefill)
        , scheduling(scheduling)
        , exclusivePrefill(exclusivePrefill)
        , prefillClass(prefillClass)
    {
    }

    uint64_t requestId{};
    int32_t tokenCount{};
    int32_t kvSlotId{-1};
    int32_t tokenOffset{};
    int32_t promptTokenCount{};
    //! Gemma4 vision-block attention currently requires one atomic prefill.
    bool allowChunkedPrefill{true};
    PhaseSchedulingHints scheduling;
    //! Keep this request in a one-row prefill batch while still permitting chunking.
    bool exclusivePrefill{};
    //! Keep text and external-producer rows in separate logical prefill batches.
    PhasePrefillClass prefillClass{PhasePrefillClass::kText};
};

enum class PhaseDispatchKind
{
    kNone,
    kPrefill,
    kDecode,
    kOverlap,
};

//! External resource-pressure hint for draining one runnable phase without overriding expired SLOs.
enum class PhaseDrainPreference
{
    kNone,
    kPrefill,
    kDecode,
};

char const* phaseDrainPreferenceName(PhaseDrainPreference preference) noexcept;

struct PhaseDispatchMetrics
{
    size_t dispatchIndex{};
    PhaseDispatchKind kind{PhaseDispatchKind::kNone};
    PhasePrefillClass prefillClass{PhasePrefillClass::kAny};
    //! Snapshot, candidate selection, and concrete batch materialization time.
    double hostSchedulerDecisionUs{};
    //! Host monotonic timestamps bracketing this asynchronous dispatch plan.
    uint64_t hostDispatchStartNs{};
    uint64_t hostSubmissionEndNs{};
    uint64_t hostCompletionNs{};
    //! Stable request membership used to join dispatch metrics with request timelines.
    std::vector<uint64_t> prefillRequestIds;
    std::vector<uint64_t> decodeRequestIds;
    int32_t prefillBatchSize{};
    int32_t decodeBatchSize{};
    int32_t prefillTokens{};
    //! Runtime B*S footprint and unused right-padding tokens for ragged prefill.
    int32_t prefillPaddedTokens{};
    int32_t prefillPaddingTokens{};
    float prefillPackingEfficiency{};
    int32_t prefillInitialRows{};
    int32_t prefillContinuationRows{};
    int32_t prefillFinalRows{};
    int32_t prefillPastKVMin{};
    int32_t prefillPastKVMean{};
    int32_t prefillPastKVMax{};
    int32_t prefillPastKVSpread{};
    int32_t prefillRemainingTokens{};
    double prefillOldestRequestAgeUs{};
    double prefillMinTtftSlackUs{};
    float predictedPrefillGpuMs{};
    float predictedDecodeSlowdownMs{};
    double predictedDecodeDebtUs{};
    int32_t consecutiveOverlapBatches{};
    bool prefillDeferredForTpot{};
    bool prefillCostCoverageMiss{};
    bool overlapEvaluatedByCost{};
    bool latencySafeFallback{};
    int32_t prefillCostLookupRows{};
    int32_t prefillCostLookupChunkLength{};
    int32_t prefillCostLookupMaxPastKVLength{};
    int32_t prefillShapeCandidatesEvaluated{};
    float predictedPrefillShapeScore{};
    bool prefillShapeDrainMode{};
    float adaptiveChunkDecodeQueuePressure{};
    float adaptiveChunkObservedTpotPressure{};
    float adaptiveChunkCombinedPressure{};
    int32_t plannedDecodeBatchSize{};
    int64_t plannedDecodeContextTokens{};
    int32_t plannedDecodeMaxContextLength{};
    int32_t predictedDecodeReplacementRows{};
    int32_t prefillCohortSize{};
    int32_t decodeCohortSize{};
    int32_t decodeTokens{};
    int32_t decodeContextTokens{};
    double prefillQueueWaitUs{};
    double decodeQueueWaitUs{};
    float prefillGpuMs{};
    float decodeGpuMs{};
    //! Component completion relative to the current H1 dispatch/augmentation
    //! boundary, rather than the component's own CUDA start event.
    float prefillCompletionMs{};
    float decodeCompletionMs{};
    float makespanGpuMs{};
    float overlapRatio{};
    //! External vision encoder state captured when this dispatch was selected.
    bool externalEncoderActive{};
    //! True when the decode cost was observed concurrently with prefill.
    bool concurrentPrefillActive{};
    //! Predicted cost and number of turns required to service the runnable decode rows.
    float predictedDecodeDrainGpuMs{};
    int32_t predictedDecodeDrainTurns{};
    //! Host page-pool snapshot after the dispatch completion; zero for linear caches.
    int32_t pagePoolTotalBundles{};
    int32_t pagePoolAllocatedBundles{};
    int32_t pagePoolAvailableBundles{};
    //! Logical page-growth controller state after this dispatch.
    int32_t pageGrowthRequestLimit{};
    int32_t pageGrowthRequestOwners{};
    float pageGrowthTpotPressure{};
    PhaseDrainPreference drainPreference{PhaseDrainPreference::kNone};
    bool drainPreferenceApplied{};
    bool globalDecisionEvaluated{};
    bool globalDecisionApplied{};
    bool globalSafeProbe{};
    uint64_t globalCandidateId{};
    bool globalCandidateParity{};
    uint64_t globalPlanId{};
    uint64_t globalSnapshotEpoch{};
    PhaseExecutionSet globalAllowedOutstanding{PhaseExecutionSet::kNone};
    PhaseExecutionSet globalLaunched{PhaseExecutionSet::kNone};
    bool globalActionFidelity{};
    PhaseExecutionVariant globalExecutionVariant{PhaseExecutionVariant::kEager};
    PhaseGlobalActionKey globalSelectedAction{};
    //! Actual phase already executing when this dispatch was augmented.
    PhaseGlobalResidualAnchor globalObservedResidualAnchor{PhaseGlobalResidualAnchor::kNone};
    PhaseGlobalDecisionReason globalDecisionReason{PhaseGlobalDecisionReason::kNoCandidate};
    double globalPredictedViolationUs{};
    double globalServiceCompression{};
    double globalReferenceWorkMs{};
    PhaseContextualPdFeatures contextualPdFeatures{};
    bool contextualPdFeatureValid{};
    bool contextualPdExploration{};
    double contextualPdMean{};
    double contextualPdUncertainty{};
    double contextualPdLowerConfidenceBound{};
    double contextualCompletionIncumbentReferenceUs{};
    double contextualCompletionNewcomerReferenceUs{};
    double contextualCompletionMinimumSlackUs{std::numeric_limits<double>::infinity()};
};

struct PhaseSchedulerTelemetry
{
    size_t sampleCount{};
    size_t overlapSampleCount{};
    size_t decodeTpotSampleCount{};
    size_t runtimeDecodeCostSampleCount{};
    size_t runtimeDecodeCostBucketCount{};
    size_t encoderContendedDecodeCostSampleCount{};
    size_t prefillContendedDecodeCostSampleCount{};
    float prefillGpuMsPerToken{};
    float decodeGpuMsPerContextToken{};
    float overlapRatio{};
    double recentDecodeTpotP95Us{};
    float recentDecodeTpotPressure{};
    bool latencySafeFallback{};
    size_t tpotHysteresisTransitions{};
    size_t drainPreferenceTransitions{};
    size_t drainPreferenceAppliedDispatches{};
    PhaseDrainPreference activeDrainPreference{PhaseDrainPreference::kNone};
    size_t globalDecisionCount{};
    size_t globalActiveDecisionCount{};
    size_t globalShadowDisagreementCount{};
    size_t globalNoFeasibleDecisionCount{};
    size_t globalSafeProbeCount{};
    size_t globalOverlapOpportunityCount{};
    size_t globalOverlapKnownCostCount{};
    size_t globalOverlapCoveringCostCount{};
    size_t globalOverlapNoSampleCount{};
    size_t globalOverlapInsufficientSampleCount{};
    size_t globalOverlapUnprofitableCount{};
    size_t globalOverlapSafeProbeEligibleCount{};
    size_t globalOverlapProbeDisabledCount{};
    size_t globalOverlapProbeIntervalBlockedCount{};
    size_t globalOverlapProbeSlackBlockedCount{};
    size_t globalOverlapSelectionCount{};
    size_t contextualPdPredictionCount{};
    size_t contextualPdReadyCount{};
    size_t contextualPdShadowDisagreementCount{};
    size_t contextualPdObservationCount{};
    size_t contextualPdRejectedObservationCount{};
    size_t contextualPdPositiveSelectionCount{};
    size_t contextualPdNegativeSelectionCount{};
    size_t contextualPdExplorationCount{};
    double contextualPdLastReward{};
    double contextualPdLastMean{};
    double contextualPdLastUncertainty{};
    double contextualPdLastLowerConfidenceBound{};
    size_t globalKnownOverlapPriorityCount{};
    size_t globalMeasuredUnprofitableOverlapSelectionCount{};
    size_t globalCostKeyObservationCount{};
    size_t globalCostKeyParityViolationCount{};
    size_t globalResidualPrefillAnchorObservationCount{};
    size_t globalResidualDecodeAnchorObservationCount{};
    size_t globalExperimentalOverlapOpportunityCount{};
    size_t globalExperimentalOverlapSelectionCount{};
    size_t globalCandidateParityViolationCount{};
    size_t globalActionFidelityViolationCount{};
    size_t globalPrefillFormationOpportunityCount{};
    size_t globalPrefillFormationDecodeSelectionCount{};
    size_t globalPrefillFormationProducerSnapshotCount{};
    size_t globalPrefillFormationCombinedCostHitCount{};
    size_t globalPrefillFormationResidualCostHitCount{};
    size_t globalPrefillFormationMaxPendingRows{};
    size_t globalWaitDecisionCount{};
    size_t globalWaitSelectedCount{};
    size_t globalWaitCandidateCount{};
    int32_t globalWaitCurrentRows{};
    int32_t globalWaitFutureRows{};
    int32_t globalWaitFutureFirstBatchRows{};
    int32_t globalWaitNowResidualRows{};
    int32_t globalWaitNowDrainTurns{};
    int32_t globalWaitFutureDrainTurns{};
    int32_t globalWaitGraphBucket{};
    uint64_t globalWaitEventId{};
    std::vector<uint64_t> globalWaitRequestIds;
    double globalWaitNowHorizonUs{};
    double globalWaitFutureHorizonUs{};
    double globalWaitPreviewBlockingUs{};
    double globalWaitPreviewUncertaintyUs{};
    double globalWaitPreviewSlackUs{};
    double globalWaitPreviewCompression{};
    PhaseGlobalActionKind lastGlobalSelectedAction{PhaseGlobalActionKind::kNone};
    PhaseGlobalDecisionReason lastGlobalDecisionReason{PhaseGlobalDecisionReason::kNoCandidate};
    double lastGlobalPredictedViolationUs{};
    double lastGlobalServiceCompression{};
    std::optional<PhaseDispatchMetrics> lastDispatch;
};

using PhaseSchedulingPolicy = std::function<PhaseDispatchKind(PhaseQueueSnapshot const&)>;
using PhaseMetricsSchedulingPolicy
    = std::function<PhaseDispatchKind(PhaseQueueSnapshot const&, PhaseSchedulerTelemetry const&)>;
using PhaseWorkEligibilityPolicy = std::function<bool(PhaseWorkItem const&, bool prefill)>;

//! Count selected decode rows whose request was absent from the prior batch.
size_t phaseDecodeReplacementRows(
    std::vector<uint64_t> const& selectedRequestIds, std::vector<uint64_t> const& previousRequestIds);

//! Conservative decode cost point loaded from offline CUDA-event profiling.
//! maxContextLength is the largest per-request KV length covered by the point.
struct PhaseDecodeBatchCost
{
    int32_t batchSize{};
    int32_t maxContextLength{};
    float p95GpuMs{};
    //! Largest sum of row context lengths covered by this point. Zero keeps
    //! legacy behavior and derives batchSize * maxContextLength.
    int64_t maxTotalContextTokens{};
};

//! Live cache and admission counters exposed to scheduler policies.
struct PhaseQueueResourceSnapshot
{
    int32_t pagePoolTotalBundles{};
    int32_t pagePoolAllocatedBundles{};
    int32_t pagePoolAvailableBundles{};
    int32_t pageReservationGuaranteedBundles{};
    int32_t pageReservationAvailableBundles{};
};

using PhaseQueueResourceSupplier = std::function<PhaseQueueResourceSnapshot()>;

//! Conservative prefill cost point loaded from offline CUDA-event profiling.
//! The point covers one uniform chunk shape up to the supplied past-KV and
//! concurrent decode batch bounds.
struct PhasePrefillBatchCost
{
    int32_t batchSize{};
    int32_t chunkLength{};
    int32_t maxPastKVLength{};
    int32_t maxConcurrentDecodeBatchSize{};
    bool initialChunk{};
    float p95GpuMs{};
    float decodeSlowdownP95Ms{};
    PhasePrefillClass prefillClass{PhasePrefillClass::kAny};
};

//! Directly observed independent-context overlap cost. Shape bounds are
//! conservative upper buckets generated from CUDA-event dispatch samples.
struct PhaseOverlapBatchCost
{
    int32_t prefillBatchSize{};
    int32_t decodeBatchSize{};
    int32_t chunkLength{};
    int32_t maxPrefillPastKVLength{};
    int32_t maxDecodeContextLength{};
    bool initialChunk{};
    float prefillP95GpuMs{};
    float decodeP95GpuMs{};
    float makespanP95GpuMs{};
    float decodeSlowdownP95Ms{};
    PhasePrefillClass prefillClass{PhasePrefillClass::kAny};
};

enum class PhaseSchedulerProfile
{
    kCustom,
    kLatencySafe,
    kBalanced,
    kThroughputBalanced,
    kLongPrefill,
    kAuto,
};

struct PhaseQueueSchedulerConfig
{
    //! Production presets only select scheduler policy behavior. Model and
    //! engine shape limits remain explicit in the fields below.
    PhaseSchedulerProfile profile{PhaseSchedulerProfile::kCustom};
    //! Profile-free P/D action selection. Shadow mode observes the same queue
    //! state without changing legacy dispatch; active mode owns the decision.
    PhaseGlobalSchedulerMode globalSchedulerMode{PhaseGlobalSchedulerMode::kDisabled};
    //! Evaluation-only policy control over the common deterministic builders.
    PhaseGlobalSelectionMode globalSelectionMode{PhaseGlobalSelectionMode::kProfileFree};
    PhaseGlobalSchedulerConfig globalSchedulerConfig{};
    PhaseGlobalCostModelConfig globalCostModelConfig{};
    //! Optional process-local tracker shared by E/P/D schedulers. A private
    //! in-memory tracker is created when this is null.
    std::shared_ptr<PhaseRuntimeCostTracker> runtimeCostTracker;
    //! Conservative cold-start bounds used until direct CUDA observations exist.
    float globalColdPrefillMsPerToken{0.02F};
    float globalColdDecodeMs{2.0F};
    //! Permit a rate-limited unknown P+D probe only with this multiple of
    //! robust serial cost remaining as slack. A conservative bounded probe is
    //! enabled by default so an empty process-local model cannot remain stuck
    //! on serial execution. Zero disables production probes.
    float globalSafeProbeSlackMultiplier{3.0F};
    size_t globalSafeProbeInterval{32U};
    //! Research-only deterministic P+D opportunity sweep. Minus one keeps
    //! production policy; 0--100 selects that percentage of hard-feasible
    //! overlap opportunities without applying the deadline/cost objective.
    int32_t globalExperimentalOverlapPercent{-1};
    //! Maximum distinct P+D shapes targeted by one calibration epoch.
    size_t globalCalibrationMaxOverlapKeys{16U};
    //! Optional ownership-aware memory horizon in one caller-defined unit.
    //! Every field returned by one invocation must use the same unit.
    std::function<PhaseActionMemoryHorizon(PhaseGlobalActionKey const&, std::vector<uint64_t> const& requestIds)>
        globalMemoryHorizonSupplier{};
    //! The caller reserves every phase allocation before work becomes
    //! runnable. This permits a single runnable phase to bypass a vacuous
    //! policy comparison without bypassing admission-time memory safety.
    bool globalDispatchUsesPreReservedMemory{};
    int32_t maxPrefillBatchSize{1};
    //! Optional row cap for external-producer prefills when their TensorRT
    //! optimization profile is narrower than the text-prefill profile. Zero
    //! inherits maxPrefillBatchSize.
    int32_t maxExternalPrefillBatchSize{};
    int32_t maxDecodeBatchSize{4};
    //! Optional row cap for continuation chunks (tokenOffset > 0). Zero
    //! inherits maxPrefillBatchSize. This keeps wide initial/atomic prefills
    //! while bounding the higher KV-traffic cost of continuation batches.
    int32_t maxContinuationPrefillBatchSize{};
    //! Optional row cap used only when prefill and decode execute concurrently.
    //! Zero inherits maxPrefillBatchSize. This lets a wide standalone prefill
    //! profile coexist with a smaller, memory-safe overlap shape.
    int32_t maxOverlapPrefillBatchSize{};
    //! Default policy only overlaps short prefills. The initial value comes from
    //! the Gemma4 E2B RTX 3080 crossover benchmark and remains configurable.
    int32_t maxOverlapPrefillTokens{128};
    //! Maximum tokens dispatched per request in one prefill turn. Zero keeps
    //! the legacy whole-prompt behavior.
    int32_t maxPrefillChunkTokens{};
    //! Optional steady-state cap while decode work is queued. This permits a
    //! larger queue-drain chunk when decode is empty without imposing that
    //! interference on active decodes. Zero inherits maxPrefillChunkTokens.
    int32_t decodeActivePrefillChunkTokens{};
    //! Keep using maxPrefillChunkTokens while at least this many prefill
    //! requests remain queued, even when decode is active. This drains a large
    //! arrival burst before switching to the steady-state cap. Zero disables
    //! backlog-triggered large chunks.
    size_t largePrefillChunkQueueThreshold{};
    //! Optional total-token budget for one compatible prefill batch. Zero disables it.
    int32_t maxPrefillBatchTokens{};
    //! Combine different text chunk lengths in one right-padded TensorRT batch.
    //! Initial and continuation chunks remain separate execution classes.
    bool enableRaggedPrefillBatching{};
    //! Pack all valid text tokens into one [1,totalTokens] carrier. Requires
    //! an indexed-paged packed-prefill engine with a compatible chunk limit.
    bool enablePackedPrefillTokenLayout{};
    //! Maximum virtual useful-token credit for each continuation row that
    //! completes a request's prefill. The scheduler caps the credit at the
    //! unused portion of one configured chunk and only grants it below half a
    //! chunk, so large tails do not preempt productive initial work. Zero
    //! preserves pure token selection.
    int32_t prefillCompletionBonusTokens{};
    //! Select a decode batch cap from measured p95 costs and current TPOT
    //! pressure. Empty costs preserve the legacy largest-available behavior.
    bool enableDynamicDecodeBatching{};
    //! Retain one stable decode cohort and replace rows only as requests finish.
    bool enableDecodeCohortBatching{};
    //! One-shot GPU metadata/cache movement cost charged for each newly introduced decode row.
    //! Zero preserves the kernel-only decode cost model.
    float decodeRowReplacementCostMs{};
    std::vector<PhaseDecodeBatchCost> decodeBatchCosts;
    //! Use confident process-local decode observations as the dynamic batching
    //! cost source. Sparse coverage never shrinks the runnable batch.
    bool enableMeasuredDecodeBatching{};
    //! Refine static decode costs from context-bucketed decode-component observations.
    bool enableDecodeComponentObservation{};
    size_t decodeComponentMinSamples{8U};
    size_t decodeComponentWindow{32U};
    int32_t runtimeDecodeContextBucketTokens{512};
    //! Bound a runtime p95 correction relative to its static prior.
    float decodeComponentMaxAdjustmentRatio{0.25F};
    //! Contended observations may be much slower than an isolated static prior.
    //! This multiplier bounds their upper correction without polluting isolated buckets.
    float decodeContentionCostMaxMultiplier{16.0F};
    //! Switch from deadline fitting to throughput-efficient backlog recovery
    //! before the TPOT deadline is fully exhausted.
    float decodeRecoveryPressureThreshold{1.0F};
    //! Select a prefill row count from profiled p95 cost and decode slack.
    bool enableDynamicPrefillBatching{};
    //! Use confident process-local prefill observations for dynamic P batch
    //! formation. Producer class is part of the cost key.
    bool enableMeasuredPrefillBatching{};
    //! Minimum dynamic prefill batch while at least this many compatible rows exist.
    int32_t minDynamicPrefillBatchSize{1};
    //! Let an expired TTFT override decode interference while decode remains within SLO.
    bool enablePrefillSloRecovery{};
    std::vector<PhasePrefillBatchCost> prefillBatchCosts;
    //! Prefer direct Co(P,D,shape) measurements over slowdown inferred from
    //! separately sampled prefill/decode rows.
    std::vector<PhaseOverlapBatchCost> overlapBatchCosts;
    //! Prevent a predicted TPOT violation by converting overlap to decode-only.
    bool enableTpotHardGuard{};
    //! If enabled, an uncovered overlap shape is unsafe instead of falling back
    //! to the indirect prefill cost table.
    bool requireDirectOverlapCost{};
    //! Let direct overlap costs replace the static maxOverlapPrefillTokens gate.
    //! Prefills already covered by the static gate preserve legacy scheduling.
    //! Larger candidates require direct coverage and the TPOT hard guard so an
    //! unsafe candidate becomes decode-only.
    bool enableCostAwareOverlapAdmission{};
    //! Disable cap-exceeding cost-aware overlap when recent decode TPOT p95
    //! reaches the enter ratio, and restore it only below the exit ratio.
    bool enableTpotHysteresis{};
    //! Collect decode TPOT pressure for external admission policies without changing phase decisions.
    bool enableDecodeTpotTelemetry{};
    float tpotHysteresisEnterRatio{0.8F};
    float tpotHysteresisExitRatio{0.6F};
    size_t tpotHysteresisWindow{32};
    size_t minTpotHysteresisSamples{8};
    int32_t maxConsecutiveOverlapBatches{4};
    double maxPredictedDecodeDebtUs{50000.0};
    int64_t autoLongPrefillBacklogTokens{4096};
    float autoDecodePressureLimit{0.5F};
    //! Keep a bounded set of requests advancing at similar chunk frontiers.
    bool enableWavefrontPrefillBatching{};
    int32_t maxPrefillCohortSize{8};
    int32_t maxPrefillCohortTurns{8};
    float decodeSlackSafetyFactor{0.8F};
    //! Model contract gate. A model with atomic multimodal prefill can disable
    //! chunking for every work item; per-request allowChunkedPrefill remains
    //! the narrower override.
    bool supportsChunkedPrefill{true};
    //! Adapt each text-prefill turn between minPrefillChunkTokens and
    //! maxPrefillChunkTokens using observed CUDA cost and overlap efficiency.
    //! Queue deadlines belong to phase selection so they cannot fragment an
    //! otherwise efficient prefill batch into extra TensorRT executions.
    bool enableAdaptivePrefillChunking{};
    int32_t minPrefillChunkTokens{32};
    int32_t prefillChunkAlignment{8};
    //! Optional profiled chunk shapes for bounded adaptive selection. Values
    //! must be strictly increasing and no larger than maxPrefillChunkTokens.
    //! An empty vector preserves the legacy aligned continuous selection.
    //! A request's final tail may be smaller than the first candidate.
    std::vector<int32_t> adaptivePrefillChunkCandidates;
    //! Jointly select a profiled (prefill batch, chunk) shape instead of
    //! choosing the chunk from queue pressure before dynamic batching. This
    //! requires bounded chunk candidates and profiled prefill costs.
    bool enableCostAwarePrefillShapeSelection{};
    //! Weight applied to predicted decode slowdown, scaled by live decode
    //! queue occupancy, when comparing feasible prefill shapes. Zero
    //! maximizes prefill throughput only.
    float prefillShapeDecodePenaltyWeight{1.0F};
    //! Optional host/enqueue cost charged once per candidate dispatch.
    float prefillShapeEnqueueCostMs{};
    //! Prefer maximum productive tokens among feasible profiled shapes once
    //! the queued prefill backlog reaches this size. Zero disables the drain
    //! regime.
    int64_t prefillShapeDrainBacklogTokens{};
    //! Select the smallest bounded candidate once decode queue occupancy
    //! multiplied by observed TPOT pressure reaches this ratio. Requiring
    //! both signals avoids shrinking chunks merely because a healthy decode
    //! batch is full.
    float adaptivePrefillChunkDecodePressureThreshold{0.8F};
    //! Permit pressure to split a row that would otherwise finish within the
    //! maximum chunk. Disabled by default because an extra TensorRT enqueue
    //! can cost more than the shorter interference window.
    bool allowAdaptivePrefillCompletionSplit{};
    //! Admit one prefill batch after this many decode-only decisions so a
    //! continuous decode queue cannot starve new requests forever.
    int32_t decodeBurstLimit{8};
    //! Opt in to the provided queue-deadline + EWMA GPU-cost policy.
    bool enableMetricsPolicy{};
    //! Dispatch expired prefill work before decode even when decode queue pressure is numerically larger.
    //! This is useful when a request TTFT target is an end-to-end hard bound while the decode target is a soft goal.
    bool enablePrefillTtftHardGuard{};
    //! Skip policy candidate construction when exactly one local phase is
    //! runnable and memory safety is either local or guaranteed by a
    //! pre-reserved ownership contract. The mechanism batch and online
    //! observation paths remain unchanged.
    bool elideVacuousGlobalDecisions{};
    double prefillQueueWaitTargetUs{5000.0};
    double decodeQueueWaitTargetUs{2000.0};
    //! Default next-token deadline for bounded WAIT/refill decisions when a
    //! request does not carry an explicit TPOT target.
    double globalDecodeTpotTargetUs{20000.0};
    float maxPredictedOverlapPrefillMs{30.0F};
    float minObservedOverlapRatio{0.05F};
    //! At or above this page-pool pressure, prefer draining decode work when
    //! neither queue has already violated its SLO. Zero disables the rule.
    float pagePressureDecodeThreshold{0.8F};
    //! Permit a scheduler-external memory broker to bias safe P/D drain decisions.
    bool enableExternalDrainPreference{};
    //! Keep an active preference for at least this many dispatches before switching or clearing it.
    size_t externalDrainPreferenceMinDwellDispatches{2U};
    //! Bound consecutive preference-selected dispatches so the other runnable phase cannot starve.
    size_t externalDrainPreferenceMaxConsecutiveDispatches{2U};
    //! Ignore a prefill drain preference at or above this observed decode TPOT pressure. Zero disables the guard.
    float externalPrefillDrainDecodePressureLimit{0.8F};
    size_t minMetricsSamples{2};
    float metricsEwmaAlpha{0.2F};
    //! Priority is constrained to [0, maxPriority]. Its contribution is bounded
    //! so an overdue lower-priority phase cannot be starved indefinitely.
    int32_t maxPriority{3};
    double priorityPressureWeight{0.25};
    //! Select higher-priority requests first within each phase batch. Waiting
    //! requests gain one effective priority class per priorityAgingUs.
    bool enablePriorityBatching{};
    double priorityAgingUs{1000000.0};
    //! Optional complete replacement for the provided metrics policy.
    PhaseMetricsSchedulingPolicy metricsPolicy{};
    //! Legacy queue-only policy, used when metrics policy is disabled.
    PhaseSchedulingPolicy policy{};
    //! Optional admission-growth gate. Ineligible work remains queued and keeps
    //! its residence timestamp until the cache owner permits further growth.
    PhaseWorkEligibilityPolicy eligibilityPolicy{};
    //! Optional live page-pool/admission snapshot used by scheduling policy.
    PhaseQueueResourceSupplier resourceSupplier{};
};

//! One concrete sampling completion horizon. Request IDs are stable ownership
//! identities expected to re-enter decode after this event. For an ordered
//! decode stream, later previews may contain the cumulative earlier cohorts.
struct PhaseDecodeCompletionPreview
{
    uint64_t eventId{};
    double predictedWaitUs{};
    double waitUncertaintyUs{};
    std::vector<uint64_t> requestIds;
    //! Next-decode context for each request ID at this completion horizon.
    std::vector<int32_t> contextLengths;
    //! Optional stable ownership and original request scheduling metadata.
    //! Empty vectors preserve the synthetic/unit-test preview contract.
    std::vector<int32_t> stableSlotIds;
    std::vector<PhaseSchedulingHints> schedulingHints;
    //! Next-decode service budget measured from event readiness.
    double serviceBudgetUs{};
};

struct PhaseDispatchPlan
{
    PhaseDispatchKind kind{PhaseDispatchKind::kNone};
    //! Snapshot, candidate selection, and concrete batch materialization time.
    double hostSchedulerDecisionUs{};
    std::vector<PhaseWorkItem> prefillBatch;
    std::vector<PhaseWorkItem> decodeBatch;
    //! Oldest selected row's host queue residence before dispatch.
    double prefillQueueWaitUs{};
    double decodeQueueWaitUs{};
    float predictedPrefillGpuMs{};
    float predictedDecodeSlowdownMs{};
    double predictedDecodeDebtUs{};
    int32_t consecutiveOverlapBatches{};
    int32_t plannedDecodeBatchSize{};
    int64_t plannedDecodeContextTokens{};
    int32_t plannedDecodeMaxContextLength{};
    int32_t predictedDecodeReplacementRows{};
    bool externalEncoderActive{};
    bool concurrentPrefillActive{};
    float predictedDecodeDrainGpuMs{};
    int32_t predictedDecodeDrainTurns{};
    bool prefillDeferredForTpot{};
    bool prefillCostCoverageMiss{};
    bool overlapEvaluatedByCost{};
    bool latencySafeFallback{};
    int32_t prefillCostLookupRows{};
    int32_t prefillCostLookupChunkLength{};
    int32_t prefillCostLookupMaxPastKVLength{};
    int32_t prefillShapeCandidatesEvaluated{};
    float predictedPrefillShapeScore{};
    bool prefillShapeDrainMode{};
    float adaptiveChunkDecodeQueuePressure{};
    float adaptiveChunkObservedTpotPressure{};
    float adaptiveChunkCombinedPressure{};
    int32_t prefillCohortSize{};
    PhaseDrainPreference drainPreference{PhaseDrainPreference::kNone};
    bool drainPreferenceApplied{};
    bool globalDecisionEvaluated{};
    bool globalDecisionApplied{};
    bool globalSafeProbe{};
    uint64_t globalCandidateId{};
    bool globalCandidateParity{};
    uint64_t globalPlanId{};
    uint64_t globalSnapshotEpoch{};
    PhaseExecutionSet globalAllowedOutstanding{PhaseExecutionSet::kNone};
    bool globalActionFidelity{};
    PhaseGlobalActionKey globalSelectedAction{};
    PhaseGlobalDecisionReason globalDecisionReason{PhaseGlobalDecisionReason::kNoCandidate};
    double globalPredictedViolationUs{};
    double globalServiceCompression{};
    double globalReferenceWorkMs{};
    PhaseContextualPdFeatures contextualPdFeatures{};
    bool contextualPdFeatureValid{};
    bool contextualPdExploration{};
    double contextualCompletionIncumbentReferenceUs{};
    double contextualCompletionNewcomerReferenceUs{};
    double contextualCompletionMinimumSlackUs{std::numeric_limits<double>::infinity()};
    double contextualPdMean{};
    double contextualPdUncertainty{};
    double contextualPdLowerConfidenceBound{};
    //! Complete policy candidate retained while this plan is in flight. The
    //! execution worker exposes it read-only so the same global policy can
    //! evaluate adding an idle peer context at a later host decision boundary.
    std::optional<PhaseGlobalActionCandidate> globalCandidate;
};

struct PhaseGlobalResidualSelection
{
    PhaseGlobalActionCandidate missingPhase;
    PhaseGlobalActionCandidate aggregate;
    bool selected{};
};

//! Host-side two-queue batch scheduler for phase-separated, dual-stream inference.
//!
//! This class intentionally owns no CUDA or TensorRT objects. The execution
//! layer consumes a DispatchPlan, binds each batch's stable KV slots to its
//! phase-local TensorMap, and records CUDA events around the two enqueues.
class PhaseQueueScheduler
//! A shared TensorRT execution context must serialize those enqueues; independent
//! contexts may opt into overlap.
{
public:
    explicit PhaseQueueScheduler(PhaseQueueSchedulerConfig config = {});

    void enqueuePrefill(PhaseWorkItem item);
    void enqueueDecode(PhaseWorkItem item);

    //! Cancel queued work. In-flight requests cannot be cancelled until their event completes.
    bool cancel(uint64_t requestId);

    PhaseDispatchPlan next();

    //! Complete one dispatched prefill chunk. An unfinished prompt is put back
    //! on the prefill queue; the final chunk transitions to decode.
    void completePrefill(PhaseWorkItem item, int32_t resultingKVLength, bool finished = false);

    //! Complete one decode turn. Unfinished requests are requeued for decode;
    //! finished requests leave the scheduler.
    void completeDecode(PhaseWorkItem item, int32_t resultingKVLength, bool finished);

    size_t prefillQueueSize() const noexcept;
    size_t decodeQueueSize() const noexcept;
    size_t decodeCohortSize() const noexcept;
    bool empty() const noexcept;
    bool hasRequest(uint64_t requestId) const noexcept;

    //! Update scheduling telemetry after one CUDA-complete dispatch.
    void observeMetrics(PhaseDispatchMetrics const& metrics);
    PhaseSchedulerTelemetry const& telemetry() const noexcept;
    //! Cost-key coverage accumulated during the current or most recent calibration epoch.
    std::vector<PhaseGlobalOverlapCostRecord> globalCalibrationDiagnostics() const;
    //! Return a read-only scheduling snapshot for an upstream phase arbiter.
    //! Return aggregate queue state for the scheduling hot path. Ready-row
    //! vectors are materialized only for opt-in decision/event capture.
    PhaseQueueSnapshot queueSnapshot(bool includeReadyDetails = false) const;
    PhaseGlobalSchedulerMode globalSchedulerMode() const noexcept;
    PhaseGlobalSelectionMode globalSelectionMode() const noexcept;
    //! Return the last IDs consumed by direct or externally coordinated dispatch.
    uint64_t globalPlanSequence() const noexcept;
    uint64_t globalSnapshotEpoch() const noexcept;
    //! Compare D-now with at most two concrete WAIT(event)+D-future actions.
    //! Shadow mode records the decision without delaying dispatch.
    bool shouldWaitForDecodeEvents(std::vector<PhaseDecodeCompletionPreview> const& previews);
    double defaultDecodeTpotTargetUs() const noexcept
    {
        return mConfig.globalDecodeTpotTargetUs;
    }
    //! Preview the best current P/D action without removing queue entries.
    std::optional<PhaseGlobalActionCandidate> previewGlobalAction();
    //! Return the complete immutable P/D frontier captured by the preceding
    //! previewGlobalAction() call. This is telemetry/evaluation evidence only;
    //! the selected action remains owned by the existing queue policy.
    std::vector<PhaseGlobalActionCandidate> const& lastGlobalPreviewCandidates() const noexcept;
    //! Preview only the current prefill action for an external E+P candidate.
    std::optional<PhaseGlobalActionCandidate> previewGlobalPrefillAction();
    //! Preview decode only while an external encoder is already in flight.
    std::optional<PhaseGlobalActionCandidate> previewGlobalDecodeAction();
    //! Compare a live single-phase action with adding the currently ready peer
    //! context. This is the incremental counterpart of previewGlobalAction().
    std::optional<PhaseGlobalResidualSelection> previewGlobalResidualAction(
        PhaseGlobalActionCandidate const& launched, double elapsedUs);
    //! Robust cost for a future prefill that has not entered the queue yet.
    //! This lets an upstream encoder protect the complete E->P first-token path
    //! without a workload label or a separate offline-only policy.
    PhaseGlobalCostEstimate estimateGlobalPrefillCost(
        int32_t batchSize, int32_t chunkLength, int32_t pastKVLength, PhasePrefillClass prefillClass) const;
    //! Sum the robust per-turn costs required to reach the first decode token.
    PhaseGlobalCostEstimate estimateGlobalPrefillDrainCost(
        int32_t batchSize, int32_t promptTokens, PhasePrefillClass prefillClass) const;
    //! Observed p95 completion of D while it shares the GPU with P.
    std::optional<float> estimateGlobalDecodeComponentP95(
        int32_t batchSize, int32_t maxContextLength, bool prefillActive) const;
    //! Consume one externally selected P/D action at the next dispatch boundary.
    void setNextGlobalAction(PhaseGlobalActionCandidate candidate, uint64_t planId = 0U, uint64_t snapshotEpoch = 0U);
    void setGlobalMemoryHorizonSupplier(
        std::function<PhaseActionMemoryHorizon(PhaseGlobalActionKey const&, std::vector<uint64_t> const& requestIds)>
            supplier);
    //! Resolve whether the exact P/D action shape is expected to use an
    //! already-captured CUDA graph. This is a mechanism capability, not a
    //! workload policy input.
    void setGlobalExecutionVariantSupplier(
        std::function<PhaseExecutionVariant(PhaseGlobalActionKey const& key, int32_t primaryTokenCount)> supplier);
    //! Largest dense decode cohort whose covered p95 GPU step fits one TPOT target.
    size_t decodeAdmissionLimitForTpot(double targetUs, int32_t maxContextLength) const noexcept;
    //! Keep runtime decode refinement out of latency mode while retaining recent samples.
    void setDecodeComponentObservationActive(bool active) noexcept;
    //! Update a scheduler-external resource drain hint. Disabled schedulers retain legacy decisions.
    void setExternalDrainPreference(PhaseDrainPreference preference) noexcept;
    //! Temporarily exclude prefill dispatch while an external encoder owns overlapping context memory.
    void setPrefillDispatchBlocked(bool blocked) noexcept;
    //! Temporarily exclude every new P/D dispatch while an external phase crosses a latency deadline.
    void setDispatchBlocked(bool blocked) noexcept;
    //! Identify external vision-encoder contention for decode cost observation and selection.
    void setExternalEncoderActive(bool active) noexcept;
    //! Publish known host-side producers so Global can price a bounded
    //! D-first prefill-formation opportunity without a workload label.
    void setPendingPrefillProducerRows(size_t rows) noexcept;
    //! Publish only producer rows unlocked by one concrete completion event.
    void setPendingPrefillProducerRows(size_t textRows, size_t externalRows, double predictedWaitUs,
        double waitUncertaintyUs, uint64_t eventId) noexcept;
    //! Reset recent scheduling history between benchmark epochs.
    //!
    //! Queue ownership is unchanged. The scheduler must be idle so a reset
    //! cannot invalidate fairness or overlap debt for active requests.
    //! Reset queue-policy history between serving epochs. Shape warmup may
    //! preserve direct Global CUDA observations while still clearing request
    //! age, fairness, and admission state.
    void resetHistory(bool preserveRuntimeCosts = false);

    //! Permit deterministic synthetic startup traffic to collect unknown P+D
    //! costs without applying production-request slack. Disable before serving.
    void setGlobalWarmupProbeMode(bool active);

private:
    struct GlobalQueueSelection
    {
        PhaseDispatchKind kind{PhaseDispatchKind::kNone};
        PhaseGlobalActionCandidate candidate;
        PhaseGlobalDecision decision;
        bool safeProbe{};
        std::vector<PhaseGlobalActionCandidate> candidateFrontier;
    };

    std::optional<GlobalQueueSelection> selectGlobalQueueAction(PhaseQueueSnapshot const& snapshot,
        bool allowPrefill = true, bool allowDecode = true, bool allowOverlap = true,
        std::optional<PhaseDispatchKind> compatibilityKind = std::nullopt);
    PhaseDispatchPlan previewMechanismPlan(PhaseDispatchKind kind) const;
    PhaseDispatchKind legacyQueueDecision(PhaseQueueSnapshot const& snapshot) const;
    PhaseGlobalActionKey globalActionKey(PhaseDispatchMetrics const& metrics) const noexcept;
    PhaseDispatchKind defaultDecision(PhaseQueueSnapshot const& snapshot) const noexcept;
    PhaseDispatchKind metricsDecision(
        PhaseQueueSnapshot const& snapshot, PhaseSchedulerTelemetry const& telemetry) const noexcept;
    void refreshExternalDrainPreference() noexcept;
    PhaseDispatchKind applyExternalDrainPreference(
        PhaseQueueSnapshot const& snapshot, PhaseDispatchKind baseline, bool& applied) const noexcept;
    int32_t selectDecodeBatchSize(PhaseQueueSnapshot const& snapshot, bool concurrentPrefill,
        float& predictedDrainGpuMs, int32_t& predictedDrainTurns) const;
    std::optional<float> decodeComponentP95(
        int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive) const;
    std::pair<int64_t, int32_t> decodeCandidateShape(int32_t maxRows) const;
    std::vector<PhaseWorkItem const*> decodeCandidateRows(int32_t maxRows) const;
    int32_t decodeCandidateReplacementRows(int32_t maxRows) const;
    std::optional<float> measuredDecodeP95(int32_t batchSize, int32_t maxContextLength) const;
    std::optional<float> measuredPrefillP95(int32_t batchSize, int32_t chunkLength, int32_t maxPastKVLength,
        PhasePrefillClass prefillClass, int32_t usefulTokens) const;
    //! Returns -1 when the TPOT guard requires decode-only, zero when no
    //! profiled dynamic decision is available, and a positive selected batch.
    int32_t selectPrefillBatchSize(std::vector<PhaseWorkItem const*> const& candidates, int32_t chunkLength,
        bool initialChunk, bool overlap, int32_t plannedDecodeBatchSize, int32_t plannedDecodeMaxContextLength,
        PhaseQueueSnapshot const& snapshot, bool preferMaximumProgress, float& predictedGpuMs,
        float& predictedDecodeSlowdownMs, bool& costCoverageMiss) const noexcept;
    PhaseQueueSnapshot snapshot(bool includeReadyDetails = false) const;
    int32_t prefillBatchLimit(PhasePrefillClass prefillClass) const noexcept;
    int32_t dispatchedPrefillTokens(PhaseWorkItem const& item) const noexcept;
    int32_t costAwarePrefillTokens(PhaseWorkItem const& item, int32_t chunkLimit) const noexcept;
    bool isPrefillBatchCompatible(PhaseWorkItem const& item, PhaseWorkItem const& seed, int32_t paddedChunkLength,
        bool initialChunk, bool allowRaggedBatch) const noexcept;
    bool isEligible(PhaseWorkItem const& item, bool prefill) const;
    std::vector<PhaseWorkItem> popBatch(std::deque<PhaseWorkItem>& queue, int32_t maxBatchSize, bool chunkPrefill,
        PhaseQueueSnapshot const& snapshot, PhaseDispatchPlan& plan);
    void enqueueKnownPrefill(PhaseWorkItem item);
    void enqueueKnownDecode(PhaseWorkItem item);

    PhaseQueueSchedulerConfig mConfig;
    PhaseGlobalScheduler mGlobalScheduler;
    std::shared_ptr<PhaseRuntimeCostTracker> mRuntimeCostTracker;
    std::deque<PhaseWorkItem> mPrefillQueue;
    std::deque<PhaseWorkItem> mDecodeQueue;
    std::unordered_set<uint64_t> mActiveRequestIds;
    std::unordered_set<uint64_t> mInFlightRequestIds;
    std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> mQueuedSince;
    PhaseSchedulerTelemetry mTelemetry;
    using RecentDecodeTpot = std::deque<double>;
    //! Mechanism previews only read recent histories through the shared tracker.
    std::shared_ptr<RecentDecodeTpot> mRecentDecodeTpotUs;
    bool mDecodeComponentObservationActive{};
    bool mLatencySafeFallback{};
    int32_t mConsecutiveDecodeBatches{};
    int32_t mConsecutiveOverlapBatches{};
    double mPredictedDecodeDebtUs{};
    std::unordered_set<uint64_t> mPrefillCohortIds;
    int32_t mPrefillCohortTurns{};
    std::unordered_set<uint64_t> mDecodeCohortIds;
    std::vector<uint64_t> mPreviousDecodeSelectionIds;
    PhaseDrainPreference mRequestedDrainPreference{PhaseDrainPreference::kNone};
    PhaseDrainPreference mActiveDrainPreference{PhaseDrainPreference::kNone};
    size_t mDrainPreferenceDispatches{};
    size_t mConsecutiveDrainPreferenceDispatches{};
    bool mPrefillDispatchBlocked{};
    bool mDispatchBlocked{};
    bool mExternalEncoderActive{};
    size_t mPendingPrefillProducerRows{};
    size_t mPendingTextPrefillProducerRows{};
    size_t mPendingExternalPrefillProducerRows{};
    double mPendingPrefillProducerWaitUs{};
    double mPendingPrefillProducerUncertaintyUs{};
    uint64_t mPendingPrefillProducerEventId{};
    bool mPendingPrefillProducerRowsClassified{};
    size_t mGlobalDecisionSequence{};
    uint64_t mGlobalPlanSequence{};
    uint64_t mGlobalSnapshotEpoch{};
    size_t mLastGlobalSafeProbeSequence{};
    size_t mGlobalExperimentalOverlapAccumulator{};
    bool mGlobalWarmupProbeMode{};
    std::vector<PhaseGlobalActionKey> mGlobalCalibrationKeys;
    std::vector<size_t> mGlobalCalibrationOpportunities;
    std::optional<PhaseGlobalActionCandidate> mNextGlobalAction;
    std::optional<PhaseGlobalDispatchPlan> mNextGlobalDispatchPlan;
    std::vector<PhaseGlobalActionCandidate> mLastGlobalPreviewCandidates;
    std::function<PhaseExecutionVariant(PhaseGlobalActionKey const& key, int32_t primaryTokenCount)>
        mGlobalExecutionVariantSupplier;
};

} // namespace rt
} // namespace trt_edgellm
