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

#include "runtime/scheduling/independentPhaseAsyncServer.h"
#include "runtime/scheduling/phaseMemoryBroker.h"
#include "runtime/scheduling/phaseVisionAdapter.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm::rt
{

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
    PhaseGlobalSchedulerMode globalSchedulerMode{PhaseGlobalSchedulerMode::kDisabled};
    PhaseGlobalSchedulerConfig globalSchedulerConfig{};
    PhaseGlobalCostModelConfig globalCostModelConfig{};
    double globalVisionPrefillColdStartUs{50000.0};
    //! Zero disables production-request exploration; enable only for controlled warm-up probes.
    float globalSafeProbeSlackMultiplier{};
    size_t globalSafeProbeInterval{32U};
    int32_t globalDecodeContextBucketTokens{512};
    std::vector<PhaseEncoderPrefillBatchCost> globalEncoderPrefillCosts;
    std::vector<PhaseEncoderDecodeBatchCost> globalEncoderDecodeCosts;
    //! Keep the initial bounded action space at E/P/D, E+D, P+D, and WAIT.
    bool enableGlobalEncoderPrefillAction{};
    //! Bound request-owned GPU vision payloads waiting in or running through the LLM phases.
    size_t maxEncodedInFlight{2U};
    //! Optional larger downstream capacity enabled only by the vision-age/decode-TPOT guard.
    size_t throughputMaxEncodedInFlight{};
    //! Optional byte budget for downstream request-owned vision payloads. Zero disables the byte gate.
    size_t maxEncodedBytes{};
    //! Maximum logical requests coalesced into one vision encoder execution. One preserves legacy behavior.
    size_t maxEncoderBatchSize{1U};
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
    size_t exclusiveEncoderPrefillDeferrals{};
    size_t globalDecisions{};
    size_t globalShadowDisagreements{};
    size_t globalEncoderSelections{};
    size_t globalEncoderPrefillSelections{};
    size_t globalEncoderDecodeSelections{};
    size_t globalPdSelections{};
    size_t globalSafeProbes{};
    size_t globalActionFidelityViolations{};
    double lastGlobalFirstTokenCriticalPathUs{};
    size_t globalEncoderArrivalWaitPeriods{};
    size_t globalEncoderArrivalWaitExpirations{};
    double lastGlobalEncoderArrivalWaitUs{};
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

    PhaseThreeSubmissionStatus submit(uint64_t requestId, LLMGenerationRequest request, int32_t maxOutputTokens,
        PhaseSchedulingHints scheduling = {});
    bool cancel(uint64_t requestId);
    bool poll();
    bool empty() const noexcept;
    PhaseThreeCoordinatorMetrics metrics() const noexcept;
    //! Enable optional request-level encoder and prefill-handoff telemetry.
    void setTimelineCallback(std::function<void(PhaseTimelineEvent const&)> timelineCallback);

    //! Observe each completed encoder batch exactly once.
    void setEncoderBatchMetricCallback(
        std::function<void(PhaseVisionEncoderBatchMetric const&)> encoderBatchMetricCallback);

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

    bool startNextEncoder();
    bool dispatchGlobalAction();
    PhaseExecutionSet observedGlobalExecution() const noexcept;
    void refreshGlobalExecutionLease();
    PhaseGlobalDispatchPlan beginGlobalExecutionLease(PhaseGlobalActionCandidate const& candidate);
    void validateGlobalExecutionLaunch();
    void abandonGlobalExecutionLease() noexcept;
    void completeGlobalOverlapObservation();
    bool completeEncoder();
    bool completeEncoderPreparation();
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
    void recordTimeline(uint64_t requestId, PhaseTimelineStage stage, size_t batchSize = 0U, int32_t kvSlotId = -1,
        uint64_t timestampNs = 0U) const;
    static size_t mediaItemCount(PendingVisionRequest const& pending) noexcept;
    static size_t mediaInputBytes(PendingVisionRequest const& pending) noexcept;
    static std::vector<int64_t> mediaGeometry(LLMGenerationRequest const& request);

    PhaseVisionAdapter& mVision;
    IndependentPhaseAsyncServer& mServer;
    PhaseThreeCoordinatorConfig mConfig;
    PhaseGlobalScheduler mGlobalScheduler;
    PhaseGlobalCostModel mGlobalCostModel;
    PhaseMemoryBroker mMemoryBroker;
    std::deque<PendingVisionRequest> mPending;
    std::vector<PendingVisionRequest> mEncoding;
    std::future<std::shared_ptr<PhaseVisionPreparedBatch>> mEncoderPreparation;
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
    };
    std::optional<PendingGlobalOverlapObservation> mPendingGlobalOverlapObservation;
    std::optional<PhaseGlobalDispatchPlan> mGlobalExecutionLease;
    std::function<void(PhaseTimelineEvent const&)> mTimelineCallback;
    std::function<void(PhaseVisionEncoderBatchMetric const&)> mEncoderBatchMetricCallback;
    size_t mEstimatedEncodedBytes{};
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
    size_t mExclusiveEncoderBatches{};
    size_t mExclusiveEncoderPrefillDeferrals{};
    bool mExclusiveEncoderInFlight{};
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
    size_t mGlobalShadowDisagreements{};
    size_t mGlobalEncoderSelections{};
    size_t mGlobalEncoderPrefillSelections{};
    size_t mGlobalEncoderDecodeSelections{};
    size_t mGlobalPdSelections{};
    size_t mGlobalSafeProbes{};
    size_t mGlobalActionFidelityViolations{};
    double mLastGlobalFirstTokenCriticalPathUs{};
    size_t mGlobalEncoderArrivalWaitPeriods{};
    size_t mGlobalEncoderArrivalWaitExpirations{};
    double mLastGlobalEncoderArrivalWaitUs{};
    bool mGlobalEncoderArrivalWaitDeferred{};
    std::chrono::steady_clock::time_point mLastVisionArrival;
    double mVisionInterarrivalEwmaUs{};
    size_t mVisionInterarrivalSamples{};
    PhaseGlobalActionKind mLastGlobalAction{PhaseGlobalActionKind::kNone};
    size_t mGlobalDecisionSequence{};
    uint64_t mGlobalPlanSequence{};
    uint64_t mGlobalSnapshotEpoch{};
    size_t mLastGlobalSafeProbeSequence{};
};

} // namespace trt_edgellm::rt
