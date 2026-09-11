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

#include "runtime/phase/policy/phaseFormationPlanner.h"
#include "runtime/phase/policy/phasePolicyMode.h"
#include "runtime/scheduling/independentPhaseAsyncServer.h"
#include "runtime/scheduling/phaseMemoryBroker.h"
#include "runtime/scheduling/phaseVisionAdapter.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm::rt
{

class PhaseEncoderPreparationWorker;

enum class PhaseThreeSubmissionStatus
{
    kEncoding,
    kQueued,
    kDuplicateRequest,
};

//! Conservative vision encoder cost point measured with CUDA events.
struct PhaseVisionEncoderBatchCost
{
    size_t batchSize{};
    size_t maxInputTokens{};
    float p95GpuMs{};
};

struct PhaseVisionEncoderBatchChoice
{
    size_t batchSize{};
    float predictedDrainGpuMs{};
    size_t predictedDrainTurns{};
    bool coverageMiss{};
};

//! Power-of-two encoder shapes plus the deployment limit for controlled calibration.
std::vector<size_t> phaseEncoderCalibrationBatchSizes(
    size_t maxEncoderBatchSize, std::vector<size_t> requestedBatchSizes = {});

//! A phase without an external SLO regains a standalone candidate after the configured measured service age.
bool phaseNoSloServiceRecoveryDue(PhaseServiceState const& service, double ageQuanta = 1.0) noexcept;

//! Do not expand persistent vision ownership without a byte-level feasibility contract.
size_t phaseVisionSafeThroughputCapacity(
    size_t baseCapacity, size_t throughputCapacity, size_t maxEncodedBytes) noexcept;

//! Direct E+D overlap point used until enough online observations exist.
struct PhaseEncoderDecodeBatchCost
{
    size_t encoderBatchSize{};
    int32_t decodeBatchSize{};
    size_t maxEncoderInputTokens{};
    int32_t maxDecodeContextLength{};
    float makespanP95GpuMs{};
};

//! Direct E+P overlap point used until enough online observations exist.
struct PhaseEncoderPrefillBatchCost
{
    size_t encoderBatchSize{};
    int32_t prefillBatchSize{};
    size_t maxEncoderInputTokens{};
    int32_t maxPrefillChunkLength{};
    int32_t maxPrefillPastKVLength{};
    float makespanP95GpuMs{};
};

//! Select the first encoder batch that minimizes the measured cost of draining the current FIFO candidates.
PhaseVisionEncoderBatchChoice phaseVisionSelectEncoderBatch(
    std::vector<size_t> const& candidateInputTokens, std::vector<PhaseVisionEncoderBatchCost> const& costs);

struct PhaseThreeCoordinatorConfig
{
    PhasePolicyMode policyMode{PhasePolicyMode::kExact};
    PhaseGlobalSchedulerMode globalSchedulerMode{PhaseGlobalSchedulerMode::kDisabled};
    PhaseGlobalSchedulerConfig globalSchedulerConfig{};
    PhaseGlobalCostModelConfig globalCostModelConfig{};
    //! Share one process-local E/P/D cost tracker with the downstream server.
    std::shared_ptr<PhaseRuntimeCostTracker> runtimeCostTracker;
    double globalVisionPrefillColdStartUs{50000.0};
    //! Conservative slack multiple for bounded process-local overlap probes.
    //! Zero explicitly disables production probes.
    float globalSafeProbeSlackMultiplier{3.0F};
    size_t globalSafeProbeInterval{32U};
    //! Maximum distinct E+P/E+D shapes targeted by one calibration epoch.
    size_t globalCalibrationMaxOverlapKeys{16U};
    int32_t globalDecodeContextBucketTokens{512};
    std::vector<PhaseEncoderPrefillBatchCost> globalEncoderPrefillCosts;
    std::vector<PhaseEncoderDecodeBatchCost> globalEncoderDecodeCosts;
    //! Keep the initial bounded action space at E/P/D, E+D, P+D, and WAIT.
    bool enableGlobalEncoderPrefillAction{};
    //! Retain one standalone P and D alternative at the final E/P/D selector.
    bool enableGlobalPdFrontier{};
    //! Restrict residual external-prefill authority and couple E/P and E/D candidate eligibility.
    bool preserveLegacyPairEligibility{};

    bool contextualResidualEligible(bool externalPrefill) const noexcept
    {
        return !preserveLegacyPairEligibility || !externalPrefill;
    }

    bool encoderDecodeEligible(bool encoderPrefillExclusive) const noexcept
    {
        return !serializeAllEncoderDecode && (!preserveLegacyPairEligibility || !encoderPrefillExclusive);
    }
    //! Number of actual dispatches attributed after an H=2/myopic selection
    //! change. The selected action is the first dispatch in the horizon.
    size_t globalFormationRealizedDispatches{4U};
    //! Bound request-owned GPU vision payloads waiting in or running through the LLM phases.
    size_t maxEncodedInFlight{2U};
    //! Optional larger downstream capacity enabled only with a byte-level ownership budget.
    size_t throughputMaxEncodedInFlight{};
    //! Optional byte budget for downstream request-owned vision payloads. Zero disables the byte gate.
    size_t maxEncodedBytes{};
    //! Maximum logical requests coalesced into one vision encoder execution. One preserves legacy behavior.
    size_t maxEncoderBatchSize{1U};
    //! Physical P/D engine capacities used only to normalize continuous
    //! contextual features. They are capabilities, not policy batch targets.
    int32_t contextualPrefillBatchCapacity{8};
    int32_t contextualDecodeBatchCapacity{64};
    //! Optional media-item cap for a coalesced encoder batch. Zero disables this guard.
    size_t maxEncoderMediaItems{};
    //! Optional raw image/video input byte cap for one encoder batch. Zero disables this guard.
    size_t maxEncoderInputBytes{};
    //! Optional model-specific encoder input-token cap. Zero inherits the physical runner limit when exposed.
    size_t maxEncoderInputTokens{};
    //! Maximum time to wait for encoder batch formation. Zero dispatches immediately.
    double encoderBatchWaitUs{};
    //! Maximum time to accumulate queued requests and downstream credits for a larger encoder batch.
    double encoderCreditWaitUs{};
    //! Desired encoder batch size while accumulating downstream credits. Zero uses the physical candidate size.
    size_t encoderCreditTargetBatchSize{};
    //! Choose E2/E4/E8 from profiled batch/token costs instead of always using the largest physical batch.
    bool enableCostAwareEncoderBatching{};
    std::vector<PhaseVisionEncoderBatchCost> encoderBatchCosts;
    //! Coalesce only requests with identical media geometry, while retaining the oldest request as the FIFO anchor.
    bool enableHomogeneousEncoderBatching{};
    //! Skip non-fitting queued requests while retaining the oldest request as the FIFO anchor.
    bool enableEncoderFitLookahead{};
    //! Maximum queued requests examined by fit lookahead. Zero examines the complete queue.
    size_t maxEncoderLookahead{};
    //! Prefill the causal text prefix into stable KV ownership while the vision encoder is still pending.
    bool enablePrefixBeforeVisionPrefill{};
    //! Avoid a separate prefix launch below this token count. Zero accepts every non-empty prefix.
    size_t minPrefixBeforeVisionTokens{128U};
    //! Maximum encoded requests released together into the independent prefill scheduler. Zero inherits encoder BS.
    size_t maxPrefillBatchSize{};
    //! Optional prompt-token budget for one release into the prefill scheduler. Zero disables the token gate.
    size_t maxPrefillBatchTokens{};
    //! Maximum time an encoded request waits for prefill batch formation. Zero dispatches immediately.
    double prefillBatchWaitUs{};
    //! Adapt P-ready batching to arrival backlog, stable-slot capacity, byte pressure, and decode TPOT pressure.
    bool enableAdaptivePrefillAdmission{};
    //! Smallest ready backlog that may be released as a throughput batch.
    size_t adaptivePrefillMinBatchSize{2U};
    //! Collapse adaptive P-ready admission to P1 at or above this decode pressure. Zero disables the guard.
    float prefillDecodeTpotPressureLimit{0.8F};
    //! Temporarily defer P-ready admission instead of releasing P1 while decode is above the pressure limit.
    bool enableDecodeProtectedPrefillDeferral{};
    //! Bound one decode-protected deferral period to avoid starving vision TTFT. Zero disables the time bound.
    double maxDecodeProtectedPrefillWaitUs{250000.0};
    //! Bypass batch waiting above this fraction of maxEncodedBytes. Zero disables the byte-pressure guard.
    double prefillReadyBytePressureRatio{0.8};
    //! Default end-to-end image TTFT SLO, including encoder queue and execution. Zero inherits the LLM default.
    double visionTtftTargetUs{2500000.0};
    //! Distinguish an external contract from the legacy built-in fallback.
    bool visionTtftTargetExplicit{};
    //! Escalate lookahead after this fraction of the oldest vision request's TTFT target. Zero disables age escalation.
    double lookaheadEscalationRatio{0.4};
    //! Contract encoded capacity at or above this normalized decode TPOT pressure. Zero disables contraction.
    float lookaheadDecodeTpotPressureLimit{0.8F};
    //! Recover encoded capacity at or below this normalized decode TPOT pressure. Zero allows immediate recovery.
    float lookaheadDecodeTpotPressureRecoveryLimit{0.6F};
    //! Fallback decode TPOT target when requests do not provide one. Zero treats pressure as unknown.
    double encodedCapacityDecodeTpotTargetUs{};
    //! Minimum wall-clock dwell between encoded-capacity transitions.
    double encodedCapacityMinDwellUs{};
    //! Latch throughput capacity at this upstream vision backlog. Zero disables backlog-triggered growth.
    size_t encodedCapacityBacklogEnterRequests{};
    //! Move encoder enqueue behind the LLM poll and gate it by measured phase debt.
    bool enableEncoderDispatchArbitration{};
    //! Run vision preprocessing on a worker while the coordinator continues polling prefill and decode.
    bool enableAsyncEncoderPreparation{};
    //! Permit independent P/D dispatch while encoder preprocessing is active.
    bool allowPdDispatchDuringEncoderPreparation{};
    //! Exclude the already-completed preparation envelope from E action learning.
    bool separateEncoderPreparationCost{};
    //! Initial encoder cost used before the first CUDA-event sample is available.
    double encoderDispatchInitialCostUs{50000.0};
    //! Margin added to the latest encoder cost when protecting text TTFT.
    double encoderDispatchCostSafetyMarginUs{5000.0};
    //! Protect text requests once age plus predicted encoder cost reaches this bound. Zero disables the guard.
    double encoderDispatchTextGuardAgeUs{250000.0};
    //! Defer encoder overlap at or above this observed decode TPOT pressure. Zero disables the guard.
    float encoderDispatchDecodeTpotPressureLimit{0.9F};
    //! Force bounded encoder progress after this oldest vision queue age. Zero disables forcing.
    double encoderDispatchMaxDeferUs{500000.0};
    //! Minimum spacing between age-forced encoder batches. Zero drains the overdue FIFO.
    double encoderDispatchForcedIntervalUs{};
    //! Serialize deadline-risked encoder work at a completed P/D boundary instead of contending with it.
    bool enableDeadlineAwareEncoderSerialization{};
    //! Start serialized service when predicted encoder completion crosses this TTFT fraction.
    double encoderSerializationDeadlineRatio{1.0};
    //! Maximum consecutive serialized encoder batches before yielding one P/D dispatch.
    size_t encoderSerializationMaxBurst{2U};
    //! Opt-in memory-aware encoder admission coupled to the E/P/D scheduler.
    PhaseMemoryBrokerConfig memoryBroker;
    //! Above this raw encoder input-token count, the encoder exclusively owns the shared E/P arena.
    size_t exclusiveEncoderInputTokenThreshold{};
    //! Every encoder dispatch exclusively owns the shared E/P arena.
    bool serializeAllEncoderPrefill{};
    //! Every encoder dispatch exclusively owns the shared E/D arena.
    bool serializeAllEncoderDecode{};
};

struct PhaseThreeCoordinatorMetrics
{
    size_t pendingVisionRequests{};
    size_t pendingPrefillReadyRequests{};
    size_t pendingPrefillReadyTokens{};
    size_t admissionProfilePrefillTokens{};
    size_t pendingPrefillReadyBytes{};
    size_t downstreamEncodedRequests{};
    size_t downstreamEncodedBytes{};
    size_t prefillStorageReleases{};
    size_t prefillStorageReleasedBytes{};
    size_t visionPrefixPlans{};
    size_t visionPrefixSubmissions{};
    size_t visionPrefixThresholdSuppressions{};
    size_t encoderStarts{};
    size_t encoderCompletions{};
    size_t encoderBatches{};
    size_t lastEncoderBatchSize{};
    size_t maxEncoderBatchSize{};
    size_t lastEncoderInputBytes{};
    size_t maxEncoderInputBytes{};
    size_t lastEncoderInputTokens{};
    size_t maxEncoderInputTokens{};
    double oldestPendingAgeUs{};
    double lastEncoderQueueWaitUs{};
    double maxEncoderQueueWaitUs{};
    float lastEncoderGpuMs{};
    float maxEncoderGpuMs{};
    float lastEncoderPreparationGpuMs{};
    float maxEncoderPreparationGpuMs{};
    float lastEncoderExecutionGpuMs{};
    float maxEncoderExecutionGpuMs{};
    size_t prefillAdmissionBatches{};
    size_t lastPrefillAdmissionBatchSize{};
    size_t maxPrefillAdmissionBatchSize{};
    size_t adaptivePrefillAdmissions{};
    size_t lowLoadPrefillAdmissions{};
    size_t backlogPrefillAdmissions{};
    size_t decodeProtectedPrefillAdmissions{};
    size_t decodeDeferredPrefillPeriods{};
    size_t capacityProtectedPrefillAdmissions{};
    size_t ageForcedPrefillAdmissions{};
    size_t byteForcedPrefillAdmissions{};
    double lastPrefillReadyQueueWaitUs{};
    double maxPrefillReadyQueueWaitUs{};
    size_t availablePrefillAdmissionSlots{};
    int32_t availableKVPages{};
    size_t effectiveEncodedCapacity{};
    size_t maxEffectiveEncodedCapacity{};
    size_t lookaheadEscalations{};
    size_t encodedCapacityContractions{};
    size_t encodedCapacityDwellBlocks{};
    float decodeTpotPressure{};
    size_t encoderDispatchDeferrals{};
    size_t encoderTextGuardDeferrals{};
    size_t encoderPrefillGuardDeferrals{};
    size_t encoderDecodeGuardDeferrals{};
    size_t encoderAgeForcedStarts{};
    size_t encoderSerializedStarts{};
    size_t encoderSerializationBursts{};
    size_t encoderSerializationBoundaryWaits{};
    size_t encoderCreditWaitPeriods{};
    size_t encoderCreditAgeReleases{};
    size_t encoderCostAwareSelections{};
    size_t encoderCostCoverageMisses{};
    float lastPredictedEncoderDrainGpuMs{};
    size_t lastPredictedEncoderDrainTurns{};
    size_t encoderPreparationStarts{};
    size_t encoderPreparationCompletions{};
    double lastEncoderPreparationUs{};
    double maxEncoderPreparationUs{};
    size_t encoderPreparationPdBlockPeriods{};
    size_t encoderPreparationPdBlockPolls{};
    double encoderPreparationPdBlockedUs{};
    double maxEncoderPreparationPdBlockedUs{};
    size_t memoryBrokerDecisions{};
    size_t memoryBrokerEncoderReductions{};
    size_t memoryBrokerBackpressure{};
    size_t memoryBrokerIdleReclaims{};
    size_t memoryBrokerPrefillPreferences{};
    size_t memoryBrokerDecodePreferences{};
    size_t memoryBrokerLastPredictedBytes{};
    PhaseMemoryBrokerReason memoryBrokerLastReason{PhaseMemoryBrokerReason::kDisabled};
    PhaseDrainPreference activeMemoryDrainPreference{PhaseDrainPreference::kNone};
    size_t memoryDrainPreferenceTransitions{};
    size_t memoryDrainPreferenceAppliedDispatches{};
    size_t exclusiveEncoderBatches{};
    size_t unknownPayloadBootstrapSelections{};
    size_t exclusiveEncoderPrefillDeferrals{};
    size_t globalDecisions{};
    size_t globalEncoderSelections{};
    size_t globalEncoderPrefillSelections{};
    size_t globalEncoderDecodeSelections{};
    size_t globalResidualAugmentationOpportunities{};
    size_t globalResidualEncoderPrefillSelections{};
    size_t globalResidualEncoderDecodeSelections{};
    size_t globalResidualAugmentationUnknownCostRejects{};
    size_t globalResidualPrefillDecodeOpportunities{};
    size_t globalResidualPrefillDecodeSelections{};
    size_t globalResidualPrefillDecodeUnknownCostRejects{};
    size_t globalResidualPrefillAnchorOpportunities{};
    size_t globalResidualPrefillAnchorSelections{};
    size_t globalResidualDecodeAnchorOpportunities{};
    size_t globalResidualDecodeAnchorSelections{};
    size_t globalResidualMeasuredUnprofitableOpportunities{};
    size_t globalResidualMeasuredUnprofitableSelections{};
    size_t globalResidualCoveringCostHits{};
    size_t globalPdSelections{};
    size_t globalSafeProbes{};
    size_t globalEncoderOverlapOpportunities{};
    size_t globalEncoderOverlapKnownCosts{};
    size_t globalEncoderOverlapNoSamples{};
    size_t globalEncoderOverlapInsufficientSamples{};
    size_t globalEncoderOverlapUnprofitable{};
    size_t globalEncoderOverlapSafeProbeEligible{};
    size_t globalEncoderOverlapProbeDisabled{};
    size_t globalEncoderOverlapProbeIntervalBlocked{};
    size_t globalEncoderOverlapProbeSlackBlocked{};
    size_t globalWarmupDecisions{};
    size_t globalWarmupPrefillCandidates{};
    size_t globalWarmupDecodeCandidates{};
    size_t globalActionFidelityViolations{};
    size_t globalHostDecisionSamples{};
    double globalHostDecisionMeanUs{};
    double globalHostDecisionP95Us{};
    double globalHostDecisionMaxUs{};
    double lastGlobalFirstTokenCriticalPathUs{};
    size_t globalEncoderArrivalWaitPeriods{};
    size_t globalEncoderArrivalWaitExpirations{};
    double lastGlobalEncoderArrivalWaitUs{};
    size_t globalFormationLookaheads{};
    size_t globalFormationPredictedRows{};
    size_t globalFormationSelectionChanges{};
    size_t globalFormationPdSelections{};
    size_t globalFormationOverlapSelections{};
    size_t globalFormationH2Agreements{};
    size_t globalFormationPostPolicyOverrides{};
    size_t globalFormationRegretSamples{};
    size_t globalFormationPositiveRegrets{};
    double globalFormationPredictedRegretUs{};
    double maxGlobalFormationPredictedRegretUs{};
    size_t lastGlobalFormationPredictedRows{};
    double lastGlobalFormationHorizonUs{};
    double lastGlobalFormationCostGapUs{};
    double lastGlobalFormationPlannerUs{};
    uint64_t lastGlobalFormationSnapshotId{};
    PhaseGlobalActionKind lastGlobalFormationH2Action{PhaseGlobalActionKind::kNone};
    PhaseGlobalActionKind lastGlobalFormationOracleAction{PhaseGlobalActionKind::kNone};
    double lastGlobalFormationDecodeViolationUs{};
    size_t globalFormationRealizedEpisodesStarted{};
    size_t globalFormationRealizedEpisodesCompleted{};
    size_t globalFormationRealizedEpisodesTruncated{};
    size_t globalFormationRealizedDecodeServices{};
    size_t globalFormationRealizedDecodeBudgets{};
    size_t globalFormationRealizedDecodeServiceViolations{};
    double globalFormationRealizedDecodeServiceGapUs{};
    double maxGlobalFormationRealizedDecodeServiceGapUs{};
    double globalFormationRealizedDecodeServiceViolationUs{};
    double maxGlobalFormationRealizedDecodeServiceViolationUs{};
    uint64_t lastGlobalFormationRealizedEpisodeId{};
    uint64_t lastGlobalFormationRealizedSnapshotId{};
    PhaseGlobalActionKind lastGlobalFormationRealizedSelectedAction{PhaseGlobalActionKind::kNone};
    PhaseGlobalActionKind lastGlobalFormationRealizedMyopicAction{PhaseGlobalActionKind::kNone};
    PhaseGlobalActionKind lastGlobalFormationRealizedOracleAction{PhaseGlobalActionKind::kNone};
    size_t lastGlobalFormationRealizedDispatches{};
    size_t lastGlobalFormationRealizedEncoderRows{};
    size_t lastGlobalFormationRealizedPrefillRows{};
    size_t lastGlobalFormationRealizedDecodeRows{};
    size_t lastGlobalFormationRealizedFirstDecodeRows{};
    size_t lastGlobalFormationRealizedMaxDecodeRows{};
    bool lastGlobalFormationRealizedDecodeServiced{};
    bool lastGlobalFormationRealizedDecodeBudgetKnown{};
    bool lastGlobalFormationRealizedTruncated{};
    double lastGlobalFormationRealizedPredictedRegretUs{};
    double lastGlobalFormationRealizedDecodeBudgetUs{};
    double lastGlobalFormationRealizedDecodeServiceGapUs{};
    double lastGlobalFormationRealizedDecodeCompletionVisibleUs{};
    double lastGlobalFormationRealizedHorizonCompletionVisibleUs{};
    double lastGlobalFormationRealizedDecodeServiceViolationUs{};
    size_t contextualEpReady{};
    size_t contextualEpDecisionDisagreements{};
    size_t contextualEpPredictions{};
    size_t contextualEpObservations{};
    size_t contextualEpRejectedObservations{};
    size_t contextualEpPositiveSelections{};
    size_t contextualEpNegativeSelections{};
    size_t contextualEpExplorations{};
    double contextualEpLastReward{};
    double contextualEpLastMean{};
    double contextualEpLastUncertainty{};
    double contextualEpLastLowerConfidenceBound{};
    size_t contextualEdReady{};
    size_t contextualEdDecisionDisagreements{};
    size_t contextualEdPredictions{};
    size_t contextualEdObservations{};
    size_t contextualEdRejectedObservations{};
    size_t contextualEdPositiveSelections{};
    size_t contextualEdNegativeSelections{};
    size_t contextualEdExplorations{};
    double contextualEdLastReward{};
    double contextualEdLastMean{};
    double contextualEdLastUncertainty{};
    double contextualEdLastLowerConfidenceBound{};
    uint64_t activeGlobalPlanId{};
    PhaseExecutionSet globalPlannedOutstanding{PhaseExecutionSet::kNone};
    PhaseExecutionSet globalObservedOutstanding{PhaseExecutionSet::kNone};
    PhaseGlobalActionKind lastGlobalAction{PhaseGlobalActionKind::kNone};
};

//! One completed vision encoder batch measured by CUDA events.
struct PhaseVisionEncoderBatchMetric
{
    size_t batchIndex{};
    size_t batchSize{};
    size_t inputBytes{};
    size_t inputTokens{};
    float preparationGpuMs{};
    float executionGpuMs{};
    float gpuMs{};
};

enum class PhaseVisionEncoderDispatchReason
{
    kLegacy,
    kAllowed,
    kTextGuard,
    kPrefillGuard,
    kDecodeGuard,
    kAgeForced,
};

struct PhaseVisionEncoderDispatchDecision
{
    bool allowed{};
    PhaseVisionEncoderDispatchReason reason{PhaseVisionEncoderDispatchReason::kAllowed};
};

//! Gate a pending encoder enqueue without coupling CUDA-event readiness to scheduling policy.
PhaseVisionEncoderDispatchDecision phaseVisionEncoderDispatchDecision(bool enabled, double oldestVisionAgeUs,
    double sinceLastForcedStartUs, double maxDeferUs, double forcedIntervalUs, double oldestTextAgeUs,
    double predictedEncoderCostUs, double textGuardAgeUs, float decodeTpotPressure, float decodeTpotPressureLimit,
    bool textPrefillInFlight, bool decodeInFlight, double prefillMinTtftSlackUs = 0.0,
    bool prefillInFlight = false) noexcept;

//! Select latency or throughput vision lookahead from queue age and observed decode pressure.
size_t phaseVisionEffectiveEncodedCapacity(size_t latencyCapacity, size_t throughputCapacity, bool throughputMode,
    double oldestVisionAgeUs, double visionTtftTargetUs, double escalationRatio, float decodeTpotPressure,
    float decodeTpotPressureLimit) noexcept;

//! Apply normalized decode-TPOT hysteresis to the latency/throughput encoded-capacity states.
size_t phaseVisionNextEncodedCapacity(size_t currentCapacity, size_t latencyCapacity, size_t throughputCapacity,
    bool throughputMode, double oldestVisionAgeUs, double visionTtftTargetUs, double escalationRatio,
    float decodeTpotPressure, float pressureEnterRatio, float pressureExitRatio) noexcept;

//! Normalize an observed decode TPOT against a request or configured target. Zero means pressure is unknown.
float phaseVisionDecodeTpotPressure(double observedTpotUs, double targetTpotUs) noexcept;

//! Return true when the oldest vision request cannot meet its TTFT fraction after one predicted encoder batch.
bool phaseVisionEncoderSerializationDue(
    double oldestVisionAgeUs, double visionTtftTargetUs, double deadlineRatio, double predictedEncoderCostUs) noexcept;

//! Normalize image scheduling at HTTP arrival so encoder time remains part of TTFT age.
PhaseSchedulingHints phaseVisionSchedulingHints(PhaseSchedulingHints scheduling, double defaultTtftTargetUs,
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

//! Count and byte admission gate for starting another encoder request.
bool phaseVisionEncoderCapacityAvailable(size_t downstreamRequests, size_t maxDownstreamRequests,
    size_t downstreamBytes, size_t maxDownstreamBytes, size_t estimatedPayloadBytes,
    size_t additionalRequests = 1U) noexcept;

struct PhaseVisionEncoderInput
{
    size_t mediaItems{};
    size_t inputBytes{};
    size_t inputTokens{};
    std::vector<int64_t> mediaGeometry;
};

//! Select encoder queue indices, optionally looking ahead for media geometry compatible with the FIFO anchor.
std::vector<size_t> phaseVisionEncoderBatchIndices(std::vector<PhaseVisionEncoderInput> const& inputs,
    size_t maxBatchSize, size_t maxMediaItems, size_t maxInputBytes, size_t maxInputTokens = 0U,
    bool requireHomogeneousGeometry = false, bool enableFitLookahead = false, size_t maxLookahead = 0U);

//! Select the next encoder cohort after removing the indices already assigned to the current execution.
std::vector<size_t> phaseVisionNextQueuedEncoderBatchIndices(std::vector<PhaseVisionEncoderInput> const& inputs,
    std::vector<size_t> const& selectedIndices, size_t maxBatchSize, size_t maxMediaItems, size_t maxInputBytes,
    size_t maxInputTokens = 0U, bool requireHomogeneousGeometry = false, bool enableFitLookahead = false,
    size_t maxLookahead = 0U);

//! Select an encoder batch bounded independently by requests, media items, and raw bytes.
size_t phaseVisionEncoderBatchSize(std::vector<PhaseVisionEncoderInput> const& inputs, size_t maxBatchSize,
    size_t maxMediaItems, size_t maxInputBytes, size_t maxInputTokens = 0U, bool requireHomogeneousGeometry = false,
    bool enableFitLookahead = false, size_t maxLookahead = 0U);

//! Decide whether a partial encoder candidate should wait for queue and downstream admission credits.
bool phaseVisionShouldAccumulateEncoderCredits(size_t admittedBatchSize, size_t candidateBatchSize,
    size_t targetBatchSize, double oldestWaitUs, double maxWaitUs) noexcept;

//! Compare dispatch-now with one bounded, arrival-predicted encoder coalescing
//! interval while protecting the oldest request's complete E->P path.
bool phaseVisionShouldWaitForGlobalEncoderArrival(double predictedWaitUs, double oldestSlackUs,
    double robustFutureCriticalPathUs, double dispatchNowHorizonUs, double waitHorizonUs) noexcept;

//! Whether batch-coupled encoder preparation would suppress otherwise runnable P/D work.
bool phaseEncoderPreparationBlocksPd(bool preparationActive, bool allowConcurrentPd, bool serverBusy,
    size_t prefillQueued, size_t decodeQueued) noexcept;

//! Select the measured interval that belongs to the E action under the active preparation contract.
float phaseEncoderActionGpuMs(
    bool asyncPreparation, bool separatePreparationCost, float pipelineGpuMs, float executionGpuMs) noexcept;

//! Select the FIFO prefix released from the encoded-ready queue into the prefill scheduler.
size_t phaseVisionReadyPrefillBatchSize(std::vector<int32_t> const& promptTokenCounts, size_t maxBatchSize,
    size_t maxBatchTokens, double oldestWaitUs, double batchWaitUs) noexcept;

enum class PhaseVisionPrefillAdmissionReason
{
    kNone,
    kLegacy,
    kLowLoad,
    kBacklog,
    kDecodeProtection,
    kDecodeDeferral,
    kAge,
    kBytePressure,
    kCapacity,
};

struct PhaseVisionPrefillAdmissionDecision
{
    size_t batchSize{};
    PhaseVisionPrefillAdmissionReason reason{PhaseVisionPrefillAdmissionReason::kNone};
};

//! Select a load-aware P-ready FIFO prefix without acquiring a KV lease.
PhaseVisionPrefillAdmissionDecision phaseVisionAdaptiveReadyPrefillDecision(
    std::vector<int32_t> const& promptTokenCounts, size_t maxBatchSize, size_t maxBatchTokens, double oldestWaitUs,
    double batchWaitUs, bool enabled, size_t minBacklogBatchSize, size_t upstreamVisionRequests,
    size_t admissibleRequests, int32_t availableKVPages, float decodeTpotPressure, float decodeTpotPressureLimit,
    size_t readyBytes, size_t maxReadyBytes, double readyBytePressureRatio, bool enableDecodeProtectedDeferral = false,
    double maxDecodeProtectedWaitUs = 0.0) noexcept;

//! Encoder -> prefill -> decode coordinator over three independent contexts.
class PhaseThreeCoordinator
{
public:
    PhaseThreeCoordinator(
        PhaseVisionAdapter& vision, IndependentPhaseAsyncServer& server, PhaseThreeCoordinatorConfig config = {});
    ~PhaseThreeCoordinator() noexcept;

    PhaseThreeSubmissionStatus submit(uint64_t requestId, LLMGenerationRequest request, int32_t maxOutputTokens,
        PhaseSchedulingHints scheduling = {});
    bool cancel(uint64_t requestId);
    bool poll();
    bool empty() const noexcept;
    PhaseThreeCoordinatorMetrics metrics() const noexcept;
    //! Cost-key coverage accumulated during the current or most recent calibration epoch.
    std::vector<PhaseGlobalOverlapCostRecord> globalCalibrationDiagnostics() const;
    //! Enable controlled unknown E+P/E+D probes while the coordinator is idle.
    void setGlobalWarmupProbeMode(bool active);
    //! Start a fresh low-overhead decision-latency epoch.
    void resetGlobalDecisionCostTelemetry() noexcept;
    //! Enable optional request-level encoder and prefill-handoff telemetry.
    void setTimelineCallback(std::function<void(PhaseTimelineEvent const&)> timelineCallback);
    //! Enable one shared epoch-relative E/P/D/C activity recorder while idle.
    void setActivityTimeline(PhaseActivityTimelineRecorder* timeline);
    //! Emit schema-versioned decision/dispatch/completion records without changing policy.
    void setUnifiedEventCallback(
        std::function<void(PhaseUnifiedEvent const&)> unifiedEventCallback, bool detailedDecisionSnapshots = true);

    //! Observe each completed encoder batch exactly once.
    void setEncoderBatchMetricCallback(
        std::function<void(PhaseVisionEncoderBatchMetric const&)> encoderBatchMetricCallback);
    //! Observe each completed bounded formation-attribution episode exactly once.
    void setFormationEpisodeCallback(
        std::function<void(PhaseFormationRealizedEpisode const&)> formationEpisodeCallback);
    //! Deliver server tokens and completions without an intermediate polling queue.
    void setEventCallbacks(std::function<void(IndependentPhaseServerToken&&)> tokenCallback,
        std::function<void(IndependentPhaseServerCompletion&&)> completionCallback);

    std::optional<IndependentPhaseServerToken> tryPopToken();
    std::optional<IndependentPhaseServerCompletion> tryPopCompletion();

private:
    struct PendingVisionRequest
    {
        uint64_t requestId{};
        LLMGenerationRequest request;
        int32_t maxOutputTokens{};
        PhaseSchedulingHints scheduling;
        size_t inputTokens{};
        size_t estimatedPayloadBytes{};
        std::vector<int64_t> mediaGeometry;
        bool prefixSubmitted{};
    };

    struct ReadyPrefillRequest
    {
        uint64_t requestId{};
        std::vector<int32_t> promptTokens;
        std::shared_ptr<PhaseVisionPayload> payload;
        int32_t maxOutputTokens{};
        PhaseSchedulingHints scheduling;
        size_t payloadBytes{};
        std::chrono::steady_clock::time_point encodedAt;
        bool prefixSubmitted{};
    };

    struct EncoderServiceEpochRecord
    {
        std::chrono::steady_clock::time_point startedAt;
        PhaseServiceReference reference;
    };

    bool startNextEncoder();
    PhaseServiceReference makeEncoderServiceReference(size_t inputTokens);
    PhaseServiceState encoderServiceState() const;
    bool dispatchGlobalAction();
    bool dispatchGlobalPrefillDecodeResidual(IndependentPhaseServerArbitrationSnapshot const& serverState);
    PhaseExecutionSet observedGlobalExecution() const noexcept;
    void refreshGlobalExecutionLease();
    PhaseGlobalDispatchPlan beginGlobalExecutionLease(PhaseGlobalActionCandidate const& candidate,
        std::vector<PhaseGlobalActionCandidate> const* candidateFrontier = nullptr,
        PhaseGlobalSelectionAudit const* selectorAudit = nullptr);
    void validateGlobalExecutionLaunch();
    void abandonGlobalExecutionLease() noexcept;
    void emitCompletedFormationEpisodes();
    void completeGlobalOverlapObservation();
    void recordGlobalDecisionCost(std::chrono::steady_clock::time_point startedAt) noexcept;
    bool completeEncoder();
    bool completeEncoderPreparation();
    bool encoderPreparationActive() const noexcept;
    float lastEncoderActionGpuMs() const noexcept;
    PhasePreparationSnapshot encoderPreparationSnapshot() const;
    void observeEncoderPreparationPdBlock(bool blocked) noexcept;
    bool submitPreparedEncoder();
    void markUnifiedEncoderSubmitted();
    bool dispatchReadyPrefill();
    std::vector<size_t> nextEncoderBatchIndices();
    PhaseVisionPrefillAdmissionDecision nextReadyPrefillDecision() const noexcept;
    bool encoderCapacityAvailable(size_t additionalRequests = 1U) const noexcept;
    size_t effectiveEncodedCapacity() const noexcept;
    float decodeTpotPressure() const noexcept;
    void refreshEffectiveEncodedCapacity() noexcept;
    PhaseVisionEncoderDispatchDecision nextEncoderDispatchDecision() const noexcept;
    bool encoderSerializationDue() const noexcept;
    void refreshEncoderSerializationGate() noexcept;
    void eraseTpotTarget(uint64_t requestId);
    void observeServerCompletion(uint64_t requestId);
    PhaseInFlightSnapshot unifiedInFlightSnapshot(uint64_t hostSnapshotNs = 0U, bool includeRequestIds = true) const;
    void recordUnifiedDecision(PhaseGlobalActionCandidate const& candidate, PhaseGlobalDispatchPlan const& plan,
        std::vector<PhaseGlobalActionCandidate> const* candidateFrontier = nullptr,
        PhaseGlobalSelectionAudit const* selectorAudit = nullptr);
    void observeUnifiedInFlightTransitions();
    void emitUnifiedEvent(PhaseUnifiedEvent event);
    void recordTimeline(uint64_t requestId, PhaseTimelineStage stage, size_t batchSize = 0U, int32_t kvSlotId = -1,
        uint64_t timestampNs = 0U) const;
    static size_t mediaItemCount(PendingVisionRequest const& pending) noexcept;
    static size_t mediaInputBytes(PendingVisionRequest const& pending) noexcept;
    static std::vector<int64_t> mediaGeometry(LLMGenerationRequest const& request);

    PhaseVisionAdapter& mVision;
    IndependentPhaseAsyncServer& mServer;
    PhaseActivityTimelineRecorder* mActivityTimeline{};
    PhaseThreeCoordinatorConfig mConfig;
    PhaseGlobalScheduler mGlobalScheduler;
    std::shared_ptr<PhaseRuntimeCostTracker> mRuntimeCostTracker;
    PhaseMemoryBroker mMemoryBroker;
    std::deque<PendingVisionRequest> mPending;
    std::unordered_map<uint64_t, EncoderServiceEpochRecord> mEncoderServiceEpochs;
    uint64_t mNextEncoderServiceEpoch{1U};
    std::vector<PendingVisionRequest> mEncoding;
    std::unique_ptr<PhaseEncoderPreparationWorker> mEncoderPreparationWorker;
    //! A CPU/preprocess-complete batch awaiting an E/P/D scheduling decision.
    std::shared_ptr<PhaseVisionPreparedBatch> mPreparedEncoder;
    std::deque<ReadyPrefillRequest> mReadyPrefill;
    std::unordered_set<uint64_t> mRequestIds;
    std::unordered_map<uint64_t, double> mRequestTpotTargets;
    std::multiset<double> mTpotTargets;
    std::unordered_map<uint64_t, size_t> mDownstreamRequestBytes;
    std::unordered_set<uint64_t> mCancelRequested;
    std::optional<std::vector<size_t>> mGlobalEncoderBatchIndices;
    std::optional<PhaseGlobalActionKey> mInFlightGlobalEncoderKey;
    double mInFlightGlobalEncoderReferenceMs{};
    struct PendingGlobalOverlapObservation
    {
        PhaseGlobalActionKey key;
        float referenceWorkMs{};
        float encoderGpuMs{};
        float phaseGpuMs{};
        float phaseElapsedMs{};
        bool residualAugmentation{};
        PhaseContextualPdFeatures contextualFeatures{};
        PhaseContextualPairDirection contextualDirection{PhaseContextualPairDirection::kEncoderToPrefill};
        bool contextualFeatureValid{};
        bool contextualExploration{};
        float contextualReferenceWorkMs{};
        double contextualLowerConfidenceBound{};
        //! Mechanism-owned plan joining the independently completed E and P/D
        //! measurements. Runtime state such as externalEncoderActive may be
        //! cleared before the slower member completes and is not a stable join
        //! key.
        uint64_t planId{};
    };
    struct ActiveGlobalPdExecution
    {
        PhaseGlobalActionCandidate candidate;
        std::chrono::steady_clock::time_point startedAt;
        //! A residual E augmentation is evaluated once per distinct encoder
        //! cohort while this P/D lease remains outstanding. Tight host polling
        //! must not turn one GPU decision boundary into thousands of policy
        //! evaluations.
        std::vector<uint64_t> lastResidualEncoderRequestIds;
        //! One live P/D lease evaluates a given missing P or D cohort once.
        //! Repeated host polls are not new GPU scheduling boundaries.
        uint64_t lastResidualPdCandidateId{};
    };
    std::optional<PendingGlobalOverlapObservation> mPendingGlobalOverlapObservation;
    std::optional<PhaseGlobalDispatchPlan> mGlobalExecutionLease;
    std::optional<ActiveGlobalPdExecution> mActiveGlobalPdExecution;
    PhaseFormationRealizedTracker mGlobalFormationRealizedTracker;
    std::optional<PhaseFormationRealizedEpisodeStart> mPendingGlobalFormationRealizedEpisode;
    std::vector<PhaseGlobalActionKey> mGlobalCalibrationKeys;
    std::vector<size_t> mGlobalCalibrationOpportunities;
    std::function<void(PhaseTimelineEvent const&)> mTimelineCallback;
    std::function<void(PhaseUnifiedEvent const&)> mUnifiedEventCallback;
    bool mUnifiedDetailedDecisionSnapshots{true};
    std::function<void(PhaseVisionEncoderBatchMetric const&)> mEncoderBatchMetricCallback;
    std::function<void(PhaseFormationRealizedEpisode const&)> mFormationEpisodeCallback;
    size_t mEstimatedEncodedBytes{};
    size_t mVisionPrefixPlans{};
    size_t mVisionPrefixSubmissions{};
    size_t mVisionPrefixThresholdSuppressions{};
    size_t mEncoderStarts{};
    size_t mEncoderCompletions{};
    size_t mEncoderBatches{};
    size_t mLastEncoderBatchSize{};
    size_t mMaxEncoderBatchSize{};
    size_t mLastEncoderInputBytes{};
    size_t mMaxEncoderInputBytes{};
    size_t mLastEncoderInputTokens{};
    size_t mMaxEncoderInputTokens{};
    double mLastEncoderQueueWaitUs{};
    double mMaxEncoderQueueWaitUs{};
    float mLastEncoderGpuMs{};
    float mMaxEncoderGpuMs{};
    float mLastEncoderPreparationGpuMs{};
    float mMaxEncoderPreparationGpuMs{};
    float mLastEncoderExecutionGpuMs{};
    float mMaxEncoderExecutionGpuMs{};
    bool mEncoderGpuSubmitted{};
    uint64_t mEncoderDispatchHostNs{};
    uint64_t mEncoderPrepareStartHostNs{};
    uint64_t mEncoderPrepareEndHostNs{};
    uint64_t mEncoderExecuteStartHostNs{};
    uint64_t mEncoderExecuteEndHostNs{};
    uint64_t mEncoderExecutionId{};
    uint64_t mEncoderActivityCorrelationId{};
    uint64_t mEncoderPlanId{};
    uint64_t mEncoderActionId{};
    PhaseGlobalActionKind mEncoderActionKind{PhaseGlobalActionKind::kNone};
    size_t mReadyPrefillTokens{};
    size_t mAdmissionProfilePrefillTokens{};
    size_t mEstimatedPromptTokens{};
    size_t mReadyPrefillBytes{};
    size_t mPrefillAdmissionBatches{};
    size_t mLastPrefillAdmissionBatchSize{};
    size_t mMaxPrefillAdmissionBatchSize{};
    size_t mAdaptivePrefillAdmissions{};
    size_t mLowLoadPrefillAdmissions{};
    size_t mBacklogPrefillAdmissions{};
    size_t mDecodeProtectedPrefillAdmissions{};
    size_t mDecodeDeferredPrefillPeriods{};
    size_t mCapacityProtectedPrefillAdmissions{};
    size_t mAgeForcedPrefillAdmissions{};
    size_t mByteForcedPrefillAdmissions{};
    double mLastPrefillReadyQueueWaitUs{};
    double mMaxPrefillReadyQueueWaitUs{};
    size_t mMaxEffectiveEncodedCapacity{};
    size_t mLookaheadEscalations{};
    size_t mEncodedCapacityContractions{};
    size_t mEncodedCapacityDwellBlocks{};
    size_t mEffectiveEncodedCapacity{};
    std::chrono::steady_clock::time_point mEncodedCapacityLastTransition;
    bool mEncodedCapacityDemandActive{};
    bool mEncodedCapacityDwellDeferred{};
    bool mDecodePrefillDeferred{};
    size_t mEncoderDispatchDeferrals{};
    size_t mEncoderTextGuardDeferrals{};
    size_t mEncoderPrefillGuardDeferrals{};
    size_t mEncoderDecodeGuardDeferrals{};
    size_t mEncoderAgeForcedStarts{};
    size_t mEncoderSerializedStarts{};
    size_t mEncoderSerializationBursts{};
    size_t mEncoderSerializationBoundaryWaits{};
    size_t mEncoderCreditWaitPeriods{};
    size_t mEncoderCreditAgeReleases{};
    size_t mEncoderCostAwareSelections{};
    size_t mEncoderCostCoverageMisses{};
    float mLastPredictedEncoderDrainGpuMs{};
    size_t mLastPredictedEncoderDrainTurns{};
    size_t mEncoderPreparationStarts{};
    size_t mEncoderPreparationCompletions{};
    double mLastEncoderPreparationUs{};
    double mMaxEncoderPreparationUs{};
    std::chrono::steady_clock::time_point mEncoderPreparationStartedAt;
    size_t mEncoderPreparationPdBlockPeriods{};
    size_t mEncoderPreparationPdBlockPolls{};
    double mEncoderPreparationPdBlockedUs{};
    double mMaxEncoderPreparationPdBlockedUs{};
    std::optional<std::chrono::steady_clock::time_point> mEncoderPreparationPdBlockStartedAt;
    size_t mExclusiveEncoderBatches{};
    size_t mUnknownPayloadBootstrapSelections{};
    size_t mExclusiveEncoderPrefillDeferrals{};
    bool mExclusiveEncoderPrefillInFlight{};
    bool mExclusiveEncoderDecodeInFlight{};
    bool mEncoderSerializationGate{};
    bool mSerializedEncoderInFlight{};
    bool mEncoderSerializationYieldPending{};
    size_t mEncoderSerializationBurstSize{};
    bool mEncoderCreditDeferred{};
    size_t mMemoryBrokerDecisions{};
    size_t mMemoryBrokerEncoderReductions{};
    size_t mMemoryBrokerBackpressure{};
    size_t mMemoryBrokerIdleReclaims{};
    size_t mMemoryBrokerPrefillPreferences{};
    size_t mMemoryBrokerDecodePreferences{};
    size_t mMemoryBrokerLastPredictedBytes{};
    PhaseMemoryBrokerReason mMemoryBrokerLastReason{PhaseMemoryBrokerReason::kDisabled};
    std::chrono::steady_clock::time_point mLastForcedEncoderStart;
    size_t mGlobalDecisions{};
    size_t mGlobalEncoderSelections{};
    size_t mGlobalEncoderPrefillSelections{};
    size_t mGlobalEncoderDecodeSelections{};
    size_t mGlobalResidualAugmentationOpportunities{};
    size_t mGlobalResidualEncoderPrefillSelections{};
    size_t mGlobalResidualEncoderDecodeSelections{};
    size_t mGlobalResidualAugmentationUnknownCostRejects{};
    size_t mGlobalResidualPrefillDecodeOpportunities{};
    size_t mGlobalResidualPrefillDecodeSelections{};
    size_t mGlobalResidualPrefillDecodeUnknownCostRejects{};
    size_t mGlobalResidualPrefillAnchorOpportunities{};
    size_t mGlobalResidualPrefillAnchorSelections{};
    size_t mGlobalResidualDecodeAnchorOpportunities{};
    size_t mGlobalResidualDecodeAnchorSelections{};
    size_t mGlobalResidualMeasuredUnprofitableOpportunities{};
    size_t mGlobalResidualMeasuredUnprofitableSelections{};
    size_t mGlobalResidualCoveringCostHits{};
    size_t mGlobalPdSelections{};
    size_t mGlobalSafeProbes{};
    size_t mGlobalEncoderOverlapOpportunities{};
    size_t mGlobalEncoderOverlapKnownCosts{};
    size_t mGlobalEncoderOverlapNoSamples{};
    size_t mGlobalEncoderOverlapInsufficientSamples{};
    size_t mGlobalEncoderOverlapUnprofitable{};
    size_t mGlobalEncoderOverlapSafeProbeEligible{};
    size_t mGlobalEncoderOverlapProbeDisabled{};
    size_t mGlobalEncoderOverlapProbeIntervalBlocked{};
    size_t mGlobalEncoderOverlapProbeSlackBlocked{};
    size_t mGlobalWarmupDecisions{};
    size_t mGlobalWarmupPrefillCandidates{};
    size_t mGlobalWarmupDecodeCandidates{};
    size_t mGlobalActionFidelityViolations{};
    static constexpr size_t kGLOBAL_DECISION_COST_WINDOW = 4096U;
    std::array<double, kGLOBAL_DECISION_COST_WINDOW> mGlobalDecisionCostsUs{};
    size_t mGlobalDecisionCostSamples{};
    size_t mGlobalDecisionCostCursor{};
    double mGlobalDecisionCostSumUs{};
    double mGlobalDecisionCostMaxUs{};
    double mLastGlobalFirstTokenCriticalPathUs{};
    size_t mGlobalEncoderArrivalWaitPeriods{};
    size_t mGlobalEncoderArrivalWaitExpirations{};
    double mLastGlobalEncoderArrivalWaitUs{};
    size_t mGlobalFormationLookaheads{};
    size_t mGlobalFormationPredictedRows{};
    size_t mGlobalFormationSelectionChanges{};
    size_t mGlobalFormationPdSelections{};
    size_t mGlobalFormationOverlapSelections{};
    size_t mGlobalFormationH2Agreements{};
    size_t mGlobalFormationPostPolicyOverrides{};
    size_t mGlobalFormationRegretSamples{};
    size_t mGlobalFormationPositiveRegrets{};
    double mGlobalFormationPredictedRegretUs{};
    double mMaxGlobalFormationPredictedRegretUs{};
    size_t mContextualEpReady{};
    size_t mContextualEpDecisionDisagreements{};
    size_t mContextualEdReady{};
    size_t mContextualEdDecisionDisagreements{};
    size_t mLastGlobalFormationPredictedRows{};
    double mLastGlobalFormationHorizonUs{};
    double mLastGlobalFormationCostGapUs{};
    double mLastGlobalFormationPlannerUs{};
    uint64_t mLastGlobalFormationSnapshotId{};
    PhaseGlobalActionKind mLastGlobalFormationH2Action{PhaseGlobalActionKind::kNone};
    PhaseGlobalActionKind mLastGlobalFormationOracleAction{PhaseGlobalActionKind::kNone};
    double mLastGlobalFormationDecodeViolationUs{};
    PhaseModelFormationSnapshot mLastScalarFormation;
    bool mGlobalEncoderArrivalWaitDeferred{};
    std::chrono::steady_clock::time_point mLastVisionArrival;
    double mVisionInterarrivalEwmaUs{};
    size_t mVisionInterarrivalSamples{};
    PhaseGlobalActionKind mLastGlobalAction{PhaseGlobalActionKind::kNone};
    size_t mGlobalDecisionSequence{};
    uint64_t mGlobalPlanSequence{};
    uint64_t mGlobalSnapshotEpoch{};
    uint64_t mUnifiedEventSequence{};
    uint64_t mUnifiedEncoderExecutionSequence{};
    uint64_t mUnifiedFallbackPlanSequence{};
    std::optional<PhaseInFlightSnapshot> mPreviousUnifiedInFlight;
    std::unordered_map<uint64_t, PhaseUnifiedEvent> mUnifiedDecisionByPlan;
    std::unordered_map<uint64_t, PhaseExecutionSet> mUnifiedAllowedOutstandingByExecution;
    size_t mLastGlobalSafeProbeSequence{};
    bool mGlobalWarmupProbeMode{};
};

} // namespace trt_edgellm::rt
