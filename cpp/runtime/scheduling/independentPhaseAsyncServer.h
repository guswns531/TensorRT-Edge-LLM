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

#include "runtime/scheduling/independentPhaseCoordinator.h"
#include "runtime/scheduling/phasePrefixReuseCache.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm::rt
{

struct PhaseVisionPayload;

//! Align admission to complete decode cohorts without exceeding the requested request capacity.
size_t phaseDecodeAlignedAdmissionCapacity(size_t requestedCapacity, size_t decodeBatchCapacity) noexcept;
//! Expose at most one complete prefill cohort to each serving-loop arbitration point.
size_t phaseServingIngressQuantum(size_t maxPendingRequests, size_t prefillBatchCapacity) noexcept;
//! Pure decision helper for sampling-aware decode-tail refill.
bool shouldDeferDecodeForSamplingRefill(
    size_t targetRows, size_t prefillRows, size_t decodeRows, size_t pendingDecodeSamplingRows) noexcept;
//! Bound a partial prefill cohort while known upstream producers can add rows.
bool shouldDeferPrefillForMicrobatchFormation(size_t targetRows, size_t prefillRows, size_t pendingProducerRows,
    double minTtftSlackUs, double ttftGuardUs, double elapsedUs, double formationWindowUs) noexcept;
//! Restrict formation to short-output cohorts when a positive limit is configured.
bool phasePrefillFormationSupportsOutputLength(int32_t maxOutputTokens, int32_t cohortMaxOutputTokens) noexcept;

//! One measured request-class region where bounded prefill formation improved throughput without violating latency.
struct IndependentPhaseFormationCost
{
    int32_t maxPromptTokens{};
    int32_t maxOutputTokens{};
    size_t maxDecodeRows{};
    float maxDecodeTpotPressure{};
    size_t targetBatchSize{};
    double windowUs{};
    double ttftTargetUs{};
    double ttftGuardUs{1000.0};
    double throughputGainPct{};
};

//! Select the highest-gain measured formation region containing the current live workload.
std::optional<size_t> phasePrefillFormationCostIndex(std::vector<IndependentPhaseFormationCost> const& costs,
    int32_t maxPromptTokens, int32_t maxOutputTokens, size_t decodeRows, float decodeTpotPressure) noexcept;
//! Hysteretic queue-pressure transition for adaptive admission.
bool nextAdaptiveThroughputMode(bool currentThroughputMode, size_t pendingRequests, size_t activeRequests,
    size_t latencyInFlightLimit, size_t backlogEnterThreshold) noexcept;
//! Move one admission step from backlog, saturation, decode pressure, and page headroom.
size_t nextStepwiseAdmissionLimit(size_t currentLimit, size_t latencyLimit, size_t throughputLimit, size_t step,
    size_t pendingRequests, size_t activeRequests, size_t backlogEnterThreshold, int32_t availablePages,
    int32_t minFreePages, float decodeTpotPressure, float pressureEnterRatio, float pressureExitRatio) noexcept;

struct IndependentPhaseAdmissionCost
{
    size_t inFlightLimit{};
    double tpotP95Us{};
};

//! Highest profiled admission limit that fits the effective TPOT budget.
size_t phaseAdmissionLimitForTpotBudget(std::vector<IndependentPhaseAdmissionCost> const& costs, size_t latencyLimit,
    size_t throughputLimit, double tpotBudgetUs) noexcept;
//! Whether the least expensive profiled admission point can meet the TPOT budget.
bool phaseAdmissionTpotBudgetSatisfiable(
    std::vector<IndependentPhaseAdmissionCost> const& costs, size_t latencyLimit, double tpotBudgetUs) noexcept;
//! Select an external-phase profile from its live request share and ready-prefill token pressure.
bool phaseAdmissionUsesExternalProfile(size_t externalRequests, size_t totalRequests, size_t externalPrefillTokens,
    double minExternalRequestFraction, size_t minExternalPrefillTokens) noexcept;
//! Latch the external workload profile for one non-idle epoch; partial ingress remains provisional.
std::optional<bool> phaseAdmissionExternalProfileForEpoch(std::optional<bool> currentSelection, size_t externalRequests,
    size_t totalRequests, size_t externalPrefillTokens, double minExternalRequestFraction,
    size_t minExternalPrefillTokens) noexcept;
//! Choose a workload-specific fallback TPOT budget without weakening an explicit per-request target.
double phaseAdmissionProfileTpotBudget(
    double defaultBudgetUs, double externalBudgetUs, bool externalProfileActive) noexcept;
//! Representative decode buckets to prime before opening a persistent serving endpoint.
std::vector<int32_t> phaseServingWarmupBatchSizes(
    int32_t maxDecodeBatchSize, std::vector<int32_t> requestedBatchSizes = {});

//! Logical page reservation used to guarantee bounded incremental KV growth.
struct IndependentPhasePageReservation
{
    uint64_t requestId{};
    int32_t basePages{};
    int32_t fullPages{};
};

//! Sum base reservations and the largest growth tails that must be simultaneously drainable.
int32_t phasePageReservationGuaranteedPages(
    std::vector<IndependentPhasePageReservation> const& reservations, int32_t maxGrowthRequests);
//! Test whether a set of incremental reservations is safe for one physical page budget.
bool phasePageReservationsFit(
    int32_t pageBudget, std::vector<IndependentPhasePageReservation> const& reservations, int32_t maxGrowthRequests);
//! Count the FIFO candidate prefix that fits the current page reservation set and request-count limit.
size_t phaseAdmissiblePageReservationPrefix(std::vector<IndependentPhasePageReservation> existing,
    std::vector<IndependentPhasePageReservation> const& candidates, int32_t pageBudget, int32_t maxGrowthRequests,
    size_t maxRequests);
//! Retain valid sticky owners, then fill free growth leases by descending tail size.
std::vector<uint64_t> selectPhasePageGrowthOwners(std::vector<IndependentPhasePageReservation> const& reservations,
    std::vector<uint64_t> const& currentOwners, int32_t maxGrowthRequests);
//! Wait until every not-yet-started growth owner reaches its first page boundary.
bool shouldDeferPhasePageGrowthCohort(
    size_t growthOwners, size_t startedGrowthOwners, size_t waitingUnstartedOwners) noexcept;

//! Request view passed to a model-specific text or multimodal adapter.
struct IndependentPhaseRequestView
{
    uint64_t requestId{};
    //! Row occupied by this request in the phase TensorRT invocation. A
    //! completion subset keeps this index so adapters can gather the matching
    //! logits instead of assuming completed rows form a dense prefix.
    size_t phaseBatchRow{};
    PhaseWorkItem work;
    std::vector<int32_t> const* promptTokens{};
    std::vector<int32_t> const* generatedTokens{};
    PhaseVisionPayload* visionPayload{};
};

//! CUDA event plus host collection callback for asynchronous greedy sampling.
//! collect() is called only after ready has completed and returns one token per requestId.
struct IndependentPhaseSampleTicket
{
    cudaEvent_t ready{};
    //! Filled by the server when the adapter returns the ticket.
    uint64_t sequenceId{};
    std::chrono::steady_clock::time_point submittedAt;
    bool fromPrefill{};
    std::vector<uint64_t> requestIds;
    std::function<std::vector<int32_t>()> collect;
    //! Return adapter-owned event/host staging resources after collection or shutdown.
    std::function<void()> release;
};

//! Model-specific seam for token staging, embeddings, deepstack/M-RoPE binding, and sampling.
struct IndependentPhaseRequestAdapter
{
    using StageCallback
        = std::function<void(std::vector<IndependentPhaseRequestView> const&, PipelineIO&, TensorMap&, cudaStream_t)>;
    using SampleCallback = std::function<std::unique_ptr<IndependentPhaseSampleTicket>(
        std::vector<IndependentPhaseRequestView> const&, PipelineIO&, cudaStream_t, bool fromPrefill)>;

    StageCallback stagePrefill;
    StageCallback stageDecode;
    SampleCallback submitSampling;
    //! The adapter supplies absolute position/RoPE metadata for a reused suffix.
    bool supportsPageAlignedPrefixReuse{};
};

enum class IndependentPhaseServerStatus
{
    kAdmitted,
    kQueued,
    kBackpressure,
    kDuplicateRequest,
    kCompleted,
    kCancelled,
};

enum class IndependentPhasePageReservationMode
{
    kFull,
    kHeadroom,
};

struct IndependentPhaseServerConfig
{
    size_t maxInFlightRequests{};
    int32_t defaultMaxOutputTokens{128};
    std::vector<int32_t> eosTokenIds;
    bool enablePrefixReuse{};
    bool enableCudaGraphs{};
    size_t maxPendingRequests{};
    size_t maxPrefillGraphs{4U};
    size_t maxDecodeGraphs{8U};
    IndependentPhasePageReservationMode pageReservationMode{IndependentPhasePageReservationMode::kFull};
    int32_t outputHeadroomTokens{128};
    //! Headroom mode guarantees this many sticky requests can grow to their full declared length.
    int32_t maxConcurrentPageGrowthRequests{8};
    //! Defer a partial decode tail while completed decode sampling tickets can refill this many rows.
    size_t decodeRefillBatchSize{};
    //! Replace the fixed refill decision with a global WAIT action over one
    //! concrete sampling event and its predicted future decode batch.
    bool enableGlobalWaitActions{};
    double globalSamplingColdStartUs{200.0};
    size_t globalSamplingLatencyWindow{32U};
    //! Defer a partial prefill cohort while known upstream producers can fill this many rows.
    size_t prefillFormationBatchSize{};
    //! Maximum non-blocking residence of one partial prefill cohort. Zero disables formation.
    double prefillFormationWindowUs{};
    //! Optional end-to-end TTFT target used only by formation, without changing scheduler policy.
    double prefillFormationTtftTargetUs{};
    //! Optional maximum declared output length for a formation-eligible live cohort.
    int32_t prefillFormationMaxOutputTokens{};
    //! Do not defer when the minimum remaining TTFT slack reaches this guard band.
    double prefillFormationTtftGuardUs{1000.0};
    //! Measured request-class regions. When non-empty, these replace the fixed formation parameters above.
    std::vector<IndependentPhaseFormationCost> prefillFormationCosts;
    //! Switch between latencyInFlightRequests/refill-off and maxInFlightRequests/refill-on from pending backlog.
    bool enableAdaptiveAdmission{};
    size_t latencyInFlightRequests{};
    size_t adaptiveBacklogEnterRequests{1U};
    //! Replace the latency/throughput jump with bounded admission steps.
    bool enableStepwiseAdaptiveAdmission{};
    size_t adaptiveAdmissionStep{16U};
    //! Minimum completed phase dispatches between admission-limit changes.
    size_t adaptiveAdmissionDwellSamples{4U};
    //! A zero enter ratio disables decode-pressure contraction.
    float adaptiveAdmissionTpotPressureEnterRatio{};
    float adaptiveAdmissionTpotPressureExitRatio{};
    int32_t adaptiveAdmissionMinFreePages{};
    //! Optional admission-level TPOT p95 costs. Empty costs disable predictive gating.
    std::vector<IndependentPhaseAdmissionCost> adaptiveAdmissionCosts;
    //! Fallback budget for requests without an explicit scheduling TPOT target.
    double adaptiveAdmissionTpotBudgetUs{};
    //! Optional profile for workloads dominated by an external producer such as a vision encoder.
    std::vector<IndependentPhaseAdmissionCost> adaptiveAdmissionExternalCosts;
    //! Optional fallback TPOT budget used while the external admission profile is active.
    double adaptiveAdmissionExternalTpotBudgetUs{};
    //! Minimum live-request share required to select adaptiveAdmissionExternalCosts.
    double adaptiveAdmissionExternalRequestFraction{};
    //! Minimum completed external prefill tokens required to select the external profile.
    size_t adaptiveAdmissionExternalPrefillTokens{};
    //! Keep a partial ingress cohort provisional until it positively identifies the external profile.
    bool enableDelayedExternalProfileSelection{};
    //! Multimodal prefill remains atomic unless an engine contract explicitly proves chunk correctness.
    bool allowChunkedVisionPrefill{};
    //! Packed-prefill engines may batch multiple complete multimodal prompts without chunking them.
    bool allowBatchedVisionPrefill{};
    //! Release embedding/deepstack leases after the final prefill event while retaining M-RoPE for decode.
    bool releaseVisionPrefillStorage{};
};

struct IndependentPhaseServerSubmission
{
    uint64_t requestId{};
    IndependentPhaseServerStatus status{IndependentPhaseServerStatus::kBackpressure};
    int32_t kvSlotId{-1};
    int32_t reusedPrefixTokens{};
};

//! Read-only LLM state used by an upstream encoder dispatch arbiter.
struct IndependentPhaseServerArbitrationSnapshot
{
    bool busy{};
    PhaseDispatchKind inFlightKind{PhaseDispatchKind::kNone};
    PhasePrefillClass inFlightPrefillClass{PhasePrefillClass::kAny};
    size_t prefillQueued{};
    size_t decodeQueued{};
    double prefillOldestRequestAgeUs{};
    double prefillMinTtftSlackUs{};
    double decodeOldestWaitUs{};
    double oldestTextWithoutTokenAgeUs{};
    double recentDecodeTpotP95Us{};
    float recentDecodeTpotPressure{};
};

struct IndependentPhaseServerCompletion
{
    uint64_t requestId{};
    std::vector<int32_t> generatedTokens;
    int32_t promptTokens{};
    double latencyMs{};
    bool stoppedByEos{};
};

struct IndependentPhaseServerToken
{
    uint64_t requestId{};
    int32_t tokenId{};
    int32_t outputIndex{};
    bool isEos{};
    double elapsedMs{};
};

struct IndependentPhaseAdmissionRequest
{
    uint64_t requestId{};
    int32_t promptTokens{};
    int32_t maxOutputTokens{};
};

//! Production-facing event-loop facade over independent prefill/decode contexts.
//!
//! The facade owns request admission, stable-slot leases, prefix sharing, and
//! asynchronous sampling tickets. Model-specific CUDA work remains in the adapter.
class IndependentPhaseAsyncServer
{
public:
    IndependentPhaseAsyncServer(IndependentPhaseServerConfig config, IndependentPhaseCoordinator& coordinator,
        StableKVPageManager& ownership, IndependentPhaseRequestAdapter adapter,
        PhasePrefixReuseCache* prefixCache = nullptr);
    ~IndependentPhaseAsyncServer() noexcept;

    IndependentPhaseAsyncServer(IndependentPhaseAsyncServer const&) = delete;
    IndependentPhaseAsyncServer& operator=(IndependentPhaseAsyncServer const&) = delete;

    IndependentPhaseServerSubmission submit(uint64_t requestId, std::vector<int32_t> promptTokens,
        int32_t maxOutputTokens = 0, PhaseSchedulingHints scheduling = {});
    //! Queue a request when slots/pages are temporarily unavailable.
    IndependentPhaseServerSubmission submitOrQueue(uint64_t requestId, std::vector<int32_t> promptTokens,
        int32_t maxOutputTokens = 0, PhaseSchedulingHints scheduling = {});
    IndependentPhaseServerSubmission submitWithVision(uint64_t requestId, std::vector<int32_t> promptTokens,
        std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens = 0,
        PhaseSchedulingHints scheduling = {});
    //! Admit a text prefix into stable KV ownership before its encoder payload is ready.
    IndependentPhaseServerSubmission submitVisionPrefix(uint64_t requestId, std::vector<int32_t> prefixTokens,
        int32_t estimatedFinalPromptTokens, int32_t maxOutputTokens = 0, PhaseSchedulingHints scheduling = {});
    //! Attach the expanded multimodal prompt and encoder payload to a previously admitted prefix.
    IndependentPhaseServerSubmission attachVisionSuffix(
        uint64_t requestId, std::vector<int32_t> promptTokens, std::shared_ptr<PhaseVisionPayload> visionPayload);
    IndependentPhaseServerSubmission submitOrQueueWithVision(uint64_t requestId, std::vector<int32_t> promptTokens,
        std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens = 0,
        PhaseSchedulingHints scheduling = {});
    bool cancel(uint64_t requestId);
    //! Capture the currently prepared phase shapes for later execute() replay.
    bool capturePreparedGraphs();
    //! Deliver ready events directly after the next GPU dispatch is enqueued.
    //! Empty callbacks preserve the polling queues.
    void setEventCallbacks(std::function<void(IndependentPhaseServerToken&&)> tokenCallback,
        std::function<void(IndependentPhaseServerCompletion&&)> completionCallback);
    //! Enable optional request-level host transition telemetry while the server is idle.
    void setTimelineCallback(std::function<void(PhaseTimelineEvent const&)> timelineCallback);
    bool poll();
    //! Progress admission, CUDA completions, sampling, and callbacks without
    //! selecting a new P/D action.
    bool pollCompletions();
    //! Select and enqueue at most one ready P/D action.
    bool dispatchReady();
    std::optional<PhaseGlobalActionCandidate> previewGlobalAction();
    std::optional<PhaseGlobalActionCandidate> previewGlobalPrefillAction();
    std::optional<PhaseGlobalActionCandidate> previewGlobalDecodeAction();
    PhaseGlobalCostEstimate estimateGlobalPrefillCost(
        int32_t batchSize, int32_t chunkLength, int32_t pastKVLength, PhasePrefillClass prefillClass) const;
    PhaseGlobalCostEstimate estimateGlobalPrefillDrainCost(
        int32_t batchSize, int32_t promptTokens, PhasePrefillClass prefillClass) const;
    //! Apply the bounded completion-aware WAIT/refill decision before an
    //! externally coordinated global P/D dispatch.
    bool shouldWaitForGlobalDecodeRefill();
    bool dispatchGlobalAction(PhaseGlobalActionCandidate candidate, uint64_t planId = 0U, uint64_t snapshotEpoch = 0U);
    void runUntilIdle(size_t maxPolls);

    std::optional<IndependentPhaseServerToken> tryPopToken();
    std::optional<IndependentPhaseServerCompletion> tryPopCompletion();
    size_t inFlightCount() const noexcept;
    size_t pendingCount() const noexcept;
    //! Return true while a concrete CUDA sampling completion can refill decode.
    bool hasPendingSampling() const noexcept;
    //! Include encoder and encoded-ready work that has not entered this server yet.
    void setExternalPendingRequests(size_t pendingRequests, double minTpotTargetUs = 0.0, size_t externalRequests = 0U,
        size_t externalPrefillTokens = 0U) noexcept;
    //! Include text/request-adapter work that has not reached admission yet.
    void setPendingAdapterRequests(size_t pendingRequests) noexcept;
    //! Forward a memory-broker drain preference into the common P/D scheduler.
    void setExternalDrainPreference(PhaseDrainPreference preference) noexcept;
    //! Exclude or restore prefill dispatch while an external phase owns overlapping workspace.
    void setPrefillDispatchBlocked(bool blocked) noexcept;
    //! Exclude or restore every new prefill/decode dispatch while an external phase owns the GPU boundary.
    void setDispatchBlocked(bool blocked) noexcept;
    //! Identify active external encoder execution for contention-aware decode cost learning.
    void setExternalEncoderActive(bool active) noexcept;
    PhaseDrainPreference activeDrainPreference() const noexcept;
    size_t drainPreferenceTransitionCount() const noexcept;
    size_t drainPreferenceAppliedDispatchCount() const noexcept;
    //! Stable slots that can be admitted immediately under the current latency/throughput limit.
    size_t availableAdmissionSlots() const noexcept;
    //! Physical KV pages currently free in the shared stable page pool.
    int32_t availableKVPages() const noexcept;
    //! FIFO candidate count that can be admitted under stable-slot and page-reservation constraints.
    size_t admissibleRequestPrefix(std::vector<IndependentPhaseAdmissionRequest> const& candidates) const;
    size_t decodeRefillWaitCount() const noexcept;
    size_t prefillFormationWaitPeriodCount() const noexcept;
    size_t prefillFormationDeferralCount() const noexcept;
    size_t prefillFormationProfileSelectionCount() const noexcept;
    size_t prefillFormationProfileMissCount() const noexcept;
    size_t pageGrowthWaitCount() const noexcept;
    size_t pendingPageGrowthCount() const noexcept;
    size_t pageGrowthOwnerCount() const noexcept;
    int32_t pageReservationBasePages() const noexcept;
    int32_t pageReservationGuaranteedPages() const;
    float decodeTpotPressure() const noexcept;
    size_t visionPayloadBytes() const noexcept;
    size_t visionPayloadBytes(std::vector<uint64_t> const& requestIds) const noexcept;
    bool releasesVisionPrefillStorage() const noexcept;
    void setGlobalMemoryHorizonSupplier(
        std::function<PhaseActionMemoryHorizon(PhaseGlobalActionKey const&, std::vector<uint64_t> const& requestIds)>
            supplier);
    size_t visionPrefillReleaseCount() const noexcept;
    size_t visionPrefillReleasedBytes() const noexcept;
    bool throughputMode() const noexcept;
    size_t throughputModeTransitionCount() const noexcept;
    size_t adaptiveAdmissionLimit() const noexcept;
    size_t adaptiveAdmissionIncreaseCount() const noexcept;
    size_t adaptiveAdmissionDecreaseCount() const noexcept;
    size_t adaptiveAdmissionCostLimit() const noexcept;
    size_t adaptiveAdmissionCostBlockCount() const noexcept;
    double adaptiveAdmissionTpotBudgetUs() const noexcept;
    bool adaptiveAdmissionTpotBudgetSatisfiable() const noexcept;
    bool adaptiveAdmissionExternalProfileActive() const noexcept;
    size_t adaptiveAdmissionExternalProfileSelectionCount() const noexcept;
    size_t adaptiveAdmissionUnsatisfiableDecisionCount() const noexcept;
    float decodeAdmissionTpotPressure() const noexcept;
    IndependentPhaseServerArbitrationSnapshot arbitrationSnapshot() const noexcept;
    bool empty() const noexcept;
    CUcontext cudaContext() const noexcept;
    PhaseGlobalSchedulerMode globalSchedulerMode() const noexcept;
    PhaseSchedulerTelemetry const& schedulerTelemetry() const noexcept;

private:
    struct RequestState
    {
        std::vector<int32_t> promptTokens;
        std::vector<int32_t> generatedTokens;
        int32_t maxOutputTokens{};
        int32_t kvSlotId{-1};
        PhaseSchedulingHints scheduling;
        std::chrono::steady_clock::time_point submittedAt;
        std::shared_ptr<PhaseVisionPayload> visionPayload;
        int32_t baseReservedPages{};
        int32_t fullReservedPages{};
        bool pageGrowthStarted{};
        bool externalProducer{};
        bool awaitingVisionPayload{};
        bool visionPrefixComplete{};
        std::vector<int32_t> pendingVisionPromptTokens;
        std::shared_ptr<PhaseVisionPayload> pendingVisionPayload;
    };

    struct PendingRequest
    {
        uint64_t requestId{};
        std::vector<int32_t> promptTokens;
        int32_t maxOutputTokens{};
        PhaseSchedulingHints scheduling;
        std::shared_ptr<PhaseVisionPayload> visionPayload;
    };

    IndependentPhaseCoordinatorCallbacks makeCallbacks();
    IndependentPhaseServerSubmission submitImpl(uint64_t requestId, std::vector<int32_t> promptTokens,
        std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens, PhaseSchedulingHints scheduling,
        int32_t reservationPromptTokens = 0, bool deferredVisionPrefix = false);
    IndependentPhaseServerSubmission submitOrQueueImpl(uint64_t requestId, std::vector<int32_t> promptTokens,
        std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens, PhaseSchedulingHints scheduling);
    std::vector<IndependentPhaseRequestView> makeViews(std::vector<PhaseWorkItem> const& batch) const;
    bool isEos(int32_t tokenId) const noexcept;
    bool admitPendingRequests();
    bool resumePendingDecodeRequests();
    bool enqueueDecodeOrWait(uint64_t requestId, RequestState& state);
    void activateVisionSuffix(uint64_t requestId, RequestState& state);
    bool activateReadyVisionSuffixes();
    IndependentPhasePageReservation makePageReservation(
        uint64_t requestId, int32_t promptTokens, int32_t maxOutputTokens) const;
    bool hasPageReservationCapacity(IndependentPhasePageReservation const& reservation) const;
    int32_t pageReservationBudget() const;
    void refreshPageGrowthOwners();
    bool shouldWaitForPrefillFormation();
    size_t admissionLimit() const noexcept;
    double effectiveAdmissionTpotBudgetUs() const noexcept;
    std::vector<IndependentPhaseAdmissionCost> const& activeAdmissionCosts() const noexcept;
    size_t costLimitedAdmissionLimit() const noexcept;
    void updateAdaptiveAdmissionMode() noexcept;
    bool processSamplingTickets();
    void enqueueSamplingTicket(std::unique_ptr<IndependentPhaseSampleTicket> ticket);
    bool flushEventCallbacks();
    void processTicket(std::unique_ptr<IndependentPhaseSampleTicket> ticket);
    void finishRequest(uint64_t requestId, bool stoppedByEos);
    void destroyTicketEvent(IndependentPhaseSampleTicket& ticket) noexcept;
    void recordTimeline(uint64_t requestId, PhaseTimelineStage stage, int32_t kvSlotId = -1) const;

    IndependentPhaseServerConfig mConfig;
    IndependentPhaseCoordinator& mCoordinator;
    StableKVPageManager& mOwnership;
    IndependentPhaseRequestAdapter mAdapter;
    PhasePrefixReuseCache* mPrefixCache{};
    std::unordered_map<uint64_t, RequestState> mRequests;
    std::multiset<int32_t> mActivePromptTokens;
    std::multiset<int32_t> mActiveOutputTokens;
    std::deque<PendingRequest> mPendingRequests;
    std::unordered_set<uint64_t> mPendingRequestIds;
    std::deque<uint64_t> mPendingDecodeRequests;
    std::unordered_set<uint64_t> mPendingDecodeRequestIds;
    std::unordered_set<uint64_t> mPageGrowthRequestIds;
    std::deque<std::unique_ptr<IndependentPhaseSampleTicket>> mSamplingTickets;
    std::deque<double> mSamplingLatencyUs;
    std::deque<IndependentPhaseServerToken> mTokenEvents;
    std::deque<IndependentPhaseServerCompletion> mCompletions;
    std::function<void(IndependentPhaseServerToken&&)> mTokenCallback;
    std::function<void(IndependentPhaseServerCompletion&&)> mCompletionCallback;
    std::function<void(PhaseTimelineEvent const&)> mTimelineCallback;
    std::unordered_set<uint64_t> mTimelineDecodeStarted;
    std::unordered_set<uint64_t> mTimelineDecodeCompleted;
    size_t mDecodeRefillWaitCount{};
    uint64_t mNextSamplingTicketSequence{1U};
    size_t mPrefillFormationWaitPeriodCount{};
    size_t mPrefillFormationDeferralCount{};
    size_t mPrefillFormationProfileSelectionCount{};
    size_t mPrefillFormationProfileMissCount{};
    size_t mPageGrowthWaitCount{};
    size_t mVisionPrefillReleaseCount{};
    size_t mVisionPrefillReleasedBytes{};
    bool mThroughputMode{};
    size_t mThroughputModeTransitionCount{};
    size_t mExternalPendingRequests{};
    size_t mPendingAdapterRequests{};
    double mExternalMinTpotTargetUs{};
    size_t mExternalRequests{};
    size_t mExternalPrefillTokens{};
    size_t mAdaptiveAdmissionLimit{};
    size_t mAdaptiveAdmissionIncreaseCount{};
    size_t mAdaptiveAdmissionDecreaseCount{};
    size_t mAdaptiveAdmissionCostBlockCount{};
    size_t mAdaptiveAdmissionExternalProfileSelectionCount{};
    size_t mAdaptiveAdmissionUnsatisfiableDecisionCount{};
    size_t mLastAdmissionDecisionSample{};
    bool mLastAdmissionExternalProfileActive{};
    std::optional<bool> mAdmissionExternalProfileEpochSelection;
    std::optional<std::chrono::steady_clock::time_point> mPrefillFormationStartedAt;
};

} // namespace trt_edgellm::rt
