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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm::rt
{

struct PhaseVisionPayload;

//! Pure decision helper for sampling-aware decode-tail refill.
bool shouldDeferDecodeForSamplingRefill(
    size_t targetRows, size_t prefillRows, size_t decodeRows, size_t pendingDecodeSamplingRows) noexcept;
//! Hysteretic queue-pressure transition for adaptive admission.
bool nextAdaptiveThroughputMode(bool currentThroughputMode, size_t pendingRequests, size_t activeRequests,
    size_t latencyInFlightLimit, size_t backlogEnterThreshold) noexcept;
//! Move one admission step from backlog, saturation, decode pressure, and page headroom.
size_t nextStepwiseAdmissionLimit(size_t currentLimit, size_t latencyLimit, size_t throughputLimit, size_t step,
    size_t pendingRequests, size_t activeRequests, size_t backlogEnterThreshold, int32_t availablePages,
    int32_t minFreePages, float decodeTpotPressure, float pressureEnterRatio, float pressureExitRatio) noexcept;
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
    bool poll();
    void runUntilIdle(size_t maxPolls);

    std::optional<IndependentPhaseServerToken> tryPopToken();
    std::optional<IndependentPhaseServerCompletion> tryPopCompletion();
    size_t inFlightCount() const noexcept;
    size_t pendingCount() const noexcept;
    //! Include encoder and encoded-ready work that has not entered this server yet.
    void setExternalPendingRequests(size_t pendingRequests) noexcept;
    //! Stable slots that can be admitted immediately under the current latency/throughput limit.
    size_t availableAdmissionSlots() const noexcept;
    //! Physical KV pages currently free in the shared stable page pool.
    int32_t availableKVPages() const noexcept;
    size_t decodeRefillWaitCount() const noexcept;
    size_t pageGrowthWaitCount() const noexcept;
    size_t pendingPageGrowthCount() const noexcept;
    size_t pageGrowthOwnerCount() const noexcept;
    int32_t pageReservationBasePages() const noexcept;
    int32_t pageReservationGuaranteedPages() const;
    float decodeTpotPressure() const noexcept;
    size_t visionPayloadBytes() const noexcept;
    size_t visionPrefillReleaseCount() const noexcept;
    size_t visionPrefillReleasedBytes() const noexcept;
    bool throughputMode() const noexcept;
    size_t throughputModeTransitionCount() const noexcept;
    size_t adaptiveAdmissionLimit() const noexcept;
    size_t adaptiveAdmissionIncreaseCount() const noexcept;
    size_t adaptiveAdmissionDecreaseCount() const noexcept;
    bool empty() const noexcept;
    CUcontext cudaContext() const noexcept;

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
        std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens, PhaseSchedulingHints scheduling);
    IndependentPhaseServerSubmission submitOrQueueImpl(uint64_t requestId, std::vector<int32_t> promptTokens,
        std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens, PhaseSchedulingHints scheduling);
    std::vector<IndependentPhaseRequestView> makeViews(std::vector<PhaseWorkItem> const& batch) const;
    bool isEos(int32_t tokenId) const noexcept;
    bool admitPendingRequests();
    bool resumePendingDecodeRequests();
    bool enqueueDecodeOrWait(uint64_t requestId, RequestState& state);
    IndependentPhasePageReservation makePageReservation(
        uint64_t requestId, int32_t promptTokens, int32_t maxOutputTokens) const;
    bool hasPageReservationCapacity(IndependentPhasePageReservation const& reservation) const;
    int32_t pageReservationBudget() const;
    void refreshPageGrowthOwners();
    bool shouldWaitForDecodeRefill() const noexcept;
    size_t admissionLimit() const noexcept;
    void updateAdaptiveAdmissionMode() noexcept;
    bool processSamplingTickets();
    bool flushEventCallbacks();
    void processTicket(std::unique_ptr<IndependentPhaseSampleTicket> ticket);
    void finishRequest(uint64_t requestId, bool stoppedByEos);
    void destroyTicketEvent(IndependentPhaseSampleTicket& ticket) noexcept;

    IndependentPhaseServerConfig mConfig;
    IndependentPhaseCoordinator& mCoordinator;
    StableKVPageManager& mOwnership;
    IndependentPhaseRequestAdapter mAdapter;
    PhasePrefixReuseCache* mPrefixCache{};
    std::unordered_map<uint64_t, RequestState> mRequests;
    std::deque<PendingRequest> mPendingRequests;
    std::unordered_set<uint64_t> mPendingRequestIds;
    std::deque<uint64_t> mPendingDecodeRequests;
    std::unordered_set<uint64_t> mPendingDecodeRequestIds;
    std::unordered_set<uint64_t> mPageGrowthRequestIds;
    std::deque<std::unique_ptr<IndependentPhaseSampleTicket>> mSamplingTickets;
    std::deque<IndependentPhaseServerToken> mTokenEvents;
    std::deque<IndependentPhaseServerCompletion> mCompletions;
    std::function<void(IndependentPhaseServerToken&&)> mTokenCallback;
    std::function<void(IndependentPhaseServerCompletion&&)> mCompletionCallback;
    size_t mDecodeRefillWaitCount{};
    size_t mPageGrowthWaitCount{};
    size_t mVisionPrefillReleaseCount{};
    size_t mVisionPrefillReleasedBytes{};
    bool mThroughputMode{};
    size_t mThroughputModeTransitionCount{};
    size_t mExternalPendingRequests{};
    size_t mAdaptiveAdmissionLimit{};
    size_t mAdaptiveAdmissionIncreaseCount{};
    size_t mAdaptiveAdmissionDecreaseCount{};
    size_t mLastAdmissionTransitionSample{};
};

} // namespace trt_edgellm::rt
