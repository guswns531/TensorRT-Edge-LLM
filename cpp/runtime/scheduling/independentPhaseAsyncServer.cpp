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

#include "runtime/scheduling/independentPhaseAsyncServer.h"

#include "runtime/scheduling/phaseVisionAdapter.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace trt_edgellm::rt
{

bool shouldDeferDecodeForSamplingRefill(
    size_t targetRows, size_t prefillRows, size_t decodeRows, size_t pendingDecodeSamplingRows) noexcept
{
    return targetRows > 0 && prefillRows == 0 && decodeRows > 0 && decodeRows < targetRows
        && decodeRows + pendingDecodeSamplingRows >= targetRows;
}

bool nextAdaptiveThroughputMode(bool currentThroughputMode, size_t pendingRequests, size_t activeRequests,
    size_t latencyInFlightLimit, size_t backlogEnterThreshold) noexcept
{
    if (!currentThroughputMode)
    {
        return pendingRequests >= backlogEnterThreshold;
    }
    return pendingRequests > 0 || activeRequests > latencyInFlightLimit;
}

std::vector<int32_t> phaseServingWarmupBatchSizes(int32_t maxDecodeBatchSize, std::vector<int32_t> requestedBatchSizes)
{
    ELLM_CHECK(maxDecodeBatchSize > 0, "Phase serving warmup requires a positive decode batch limit");
    if (requestedBatchSizes.empty())
    {
        requestedBatchSizes = {std::max(1, maxDecodeBatchSize / 8), std::max(1, maxDecodeBatchSize / 4),
            std::max(1, maxDecodeBatchSize / 2), std::max(1, 3 * maxDecodeBatchSize / 4), maxDecodeBatchSize};
    }
    for (int32_t const batchSize : requestedBatchSizes)
    {
        ELLM_CHECK(batchSize > 0 && batchSize <= maxDecodeBatchSize,
            "Phase serving warmup batch is outside the decode profile");
    }
    std::sort(requestedBatchSizes.begin(), requestedBatchSizes.end());
    requestedBatchSizes.erase(
        std::unique(requestedBatchSizes.begin(), requestedBatchSizes.end()), requestedBatchSizes.end());
    return requestedBatchSizes;
}

IndependentPhaseAsyncServer::IndependentPhaseAsyncServer(IndependentPhaseServerConfig config,
    IndependentPhaseCoordinator& coordinator, StableKVPageManager& ownership, IndependentPhaseRequestAdapter adapter,
    PhasePrefixReuseCache* prefixCache)
    : mConfig(std::move(config))
    , mCoordinator(coordinator)
    , mOwnership(ownership)
    , mAdapter(std::move(adapter))
    , mPrefixCache(prefixCache)
{
    ELLM_CHECK(mConfig.maxInFlightRequests > 0, "Independent phase server request capacity must be positive");
    ELLM_CHECK(mConfig.defaultMaxOutputTokens > 0, "Independent phase server output capacity must be positive");
    ELLM_CHECK(mConfig.outputHeadroomTokens > 0, "Independent phase server output headroom must be positive");
    ELLM_CHECK(mConfig.decodeRefillBatchSize <= mConfig.maxInFlightRequests,
        "Decode refill batch cannot exceed the server in-flight capacity");
    ELLM_CHECK(!mConfig.enableAdaptiveAdmission
            || (mConfig.latencyInFlightRequests > 0 && mConfig.latencyInFlightRequests <= mConfig.maxInFlightRequests
                && mConfig.adaptiveBacklogEnterRequests > 0),
        "Adaptive admission requires valid latency and backlog thresholds");
    ELLM_CHECK(static_cast<bool>(mAdapter.submitSampling), "Independent phase server requires a sampling adapter");
    mCoordinator.setGraphCaptureLimits(mConfig.maxPrefillGraphs, mConfig.maxDecodeGraphs);
    mCoordinator.setGraphCaptureEnabled(mConfig.enableCudaGraphs);
    mCoordinator.scheduler().setOnlineDecodeCostLearningActive(!mConfig.enableAdaptiveAdmission);
    mCoordinator.setCallbacks(makeCallbacks());
}

IndependentPhaseAsyncServer::~IndependentPhaseAsyncServer() noexcept
{
    for (auto& ticket : mSamplingTickets)
    {
        destroyTicketEvent(*ticket);
    }
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submit(
    uint64_t requestId, std::vector<int32_t> promptTokens, int32_t maxOutputTokens, PhaseSchedulingHints scheduling)
{
    return submitImpl(requestId, std::move(promptTokens), nullptr, maxOutputTokens, scheduling);
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitWithVision(uint64_t requestId,
    std::vector<int32_t> promptTokens, std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens,
    PhaseSchedulingHints scheduling)
{
    ELLM_CHECK(visionPayload != nullptr, "Vision phase submission requires an encoded payload");
    return submitImpl(requestId, std::move(promptTokens), std::move(visionPayload), maxOutputTokens, scheduling);
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitImpl(uint64_t requestId,
    std::vector<int32_t> promptTokens, std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens,
    PhaseSchedulingHints scheduling)
{
    bool const allowChunkedPrefill = true;
    bool const exclusivePrefill = visionPayload != nullptr;
    IndependentPhaseServerSubmission result{requestId};
    if (mRequests.find(requestId) != mRequests.end() || mPendingRequestIds.find(requestId) != mPendingRequestIds.end()
        || promptTokens.empty())
    {
        result.status = IndependentPhaseServerStatus::kDuplicateRequest;
        return result;
    }
    if (mRequests.size() >= admissionLimit() || mOwnership.availableSlots() == 0)
    {
        return result;
    }
    maxOutputTokens = maxOutputTokens > 0 ? maxOutputTokens : mConfig.defaultMaxOutputTokens;

    int32_t const slot = mOwnership.reserve();
    int32_t reusedPrefixTokens{};
    try
    {
        if (mConfig.enablePrefixReuse && mAdapter.supportsPageAlignedPrefixReuse && mPrefixCache != nullptr)
        {
            if (auto const match = mPrefixCache->lookup(promptTokens); match.has_value())
            {
                mOwnership.sharePrefix(match->sourceSlot, slot, match->matchedTokens);
                reusedPrefixTokens = match->matchedTokens;
            }
        }
        int32_t const reservedOutput = mConfig.pageReservationMode == IndependentPhasePageReservationMode::kFull
            ? maxOutputTokens
            : std::min(maxOutputTokens, mConfig.outputHeadroomTokens);
        mOwnership.ensureCapacity(slot, static_cast<int32_t>(promptTokens.size()) + reservedOutput);
    }
    catch (std::runtime_error const&)
    {
        mOwnership.release(slot);
        return result;
    }

    RequestState state;
    state.promptTokens = std::move(promptTokens);
    state.maxOutputTokens = maxOutputTokens;
    state.kvSlotId = slot;
    state.scheduling = scheduling;
    state.submittedAt = std::chrono::steady_clock::now();
    state.visionPayload = std::move(visionPayload);
    mRequests.emplace(requestId, std::move(state));
    int32_t const remaining = static_cast<int32_t>(mRequests.at(requestId).promptTokens.size()) - reusedPrefixTokens;
    mCoordinator.enqueuePrefill({requestId, remaining, slot, reusedPrefixTokens,
        static_cast<int32_t>(mRequests.at(requestId).promptTokens.size()), allowChunkedPrefill, scheduling,
        exclusivePrefill});
    result.status = IndependentPhaseServerStatus::kAdmitted;
    result.kvSlotId = slot;
    result.reusedPrefixTokens = reusedPrefixTokens;
    return result;
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitOrQueue(
    uint64_t requestId, std::vector<int32_t> promptTokens, int32_t maxOutputTokens, PhaseSchedulingHints scheduling)
{
    return submitOrQueueImpl(requestId, std::move(promptTokens), nullptr, maxOutputTokens, scheduling);
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitOrQueueWithVision(uint64_t requestId,
    std::vector<int32_t> promptTokens, std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens,
    PhaseSchedulingHints scheduling)
{
    ELLM_CHECK(visionPayload != nullptr, "Queued vision phase submission requires an encoded payload");
    return submitOrQueueImpl(requestId, std::move(promptTokens), std::move(visionPayload), maxOutputTokens, scheduling);
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitOrQueueImpl(uint64_t requestId,
    std::vector<int32_t> promptTokens, std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens,
    PhaseSchedulingHints scheduling)
{
    IndependentPhaseServerSubmission result
        = submitImpl(requestId, promptTokens, visionPayload, maxOutputTokens, scheduling);
    if (result.status != IndependentPhaseServerStatus::kBackpressure)
    {
        return result;
    }
    if (mConfig.maxPendingRequests == 0 || mPendingRequests.size() >= mConfig.maxPendingRequests
        || mPendingRequestIds.find(requestId) != mPendingRequestIds.end()
        || mRequests.find(requestId) != mRequests.end() || promptTokens.empty())
    {
        return result;
    }
    mPendingRequests.push_back(
        {requestId, std::move(promptTokens), maxOutputTokens, scheduling, std::move(visionPayload)});
    mPendingRequestIds.insert(requestId);
    result.status = IndependentPhaseServerStatus::kQueued;
    return result;
}

bool IndependentPhaseAsyncServer::cancel(uint64_t requestId)
{
    auto it = mRequests.find(requestId);
    if (it == mRequests.end())
    {
        if (mPendingRequestIds.erase(requestId) == 0)
        {
            return false;
        }
        auto const pending = std::find_if(mPendingRequests.begin(), mPendingRequests.end(),
            [&](PendingRequest const& request) { return request.requestId == requestId; });
        ELLM_CHECK(pending != mPendingRequests.end(), "Pending request index is inconsistent");
        mPendingRequests.erase(pending);
        return true;
    }
    if (mPendingDecodeRequestIds.erase(requestId) > 0)
    {
        auto const waiting = std::find(mPendingDecodeRequests.begin(), mPendingDecodeRequests.end(), requestId);
        ELLM_CHECK(waiting != mPendingDecodeRequests.end(), "Pending decode request index is inconsistent");
        mPendingDecodeRequests.erase(waiting);
    }
    else if (!mCoordinator.scheduler().cancel(requestId))
    {
        return false;
    }
    mOwnership.release(it->second.kvSlotId);
    mRequests.erase(it);
    return true;
}

bool IndependentPhaseAsyncServer::capturePreparedGraphs()
{
    ELLM_CHECK(
        mRequests.empty() && mSamplingTickets.empty(), "Phase graphs cannot be captured while requests are active");
    return mCoordinator.capturePreparedGraphs();
}

void IndependentPhaseAsyncServer::setEventCallbacks(std::function<void(IndependentPhaseServerToken&&)> tokenCallback,
    std::function<void(IndependentPhaseServerCompletion&&)> completionCallback)
{
    ELLM_CHECK(mRequests.empty() && mPendingRequests.empty() && mSamplingTickets.empty(),
        "Phase event callbacks can only change while the server is idle");
    mTokenCallback = std::move(tokenCallback);
    mCompletionCallback = std::move(completionCallback);
    static_cast<void>(flushEventCallbacks());
}

bool IndependentPhaseAsyncServer::poll()
{
    updateAdaptiveAdmissionMode();
    bool progressed = admitPendingRequests();
    progressed = resumePendingDecodeRequests() || progressed;
    progressed = mCoordinator.poll() || progressed;
    progressed = processSamplingTickets() || progressed;
    progressed = resumePendingDecodeRequests() || progressed;
    progressed = admitPendingRequests() || progressed;
    updateAdaptiveAdmissionMode();
    bool const waitForDecodeRefill = shouldWaitForDecodeRefill();
    if (waitForDecodeRefill)
    {
        ++mDecodeRefillWaitCount;
    }
    if (!mCoordinator.busy() && !mCoordinator.empty() && !waitForDecodeRefill)
    {
        progressed = mCoordinator.dispatchNext() || progressed;
    }
    progressed = flushEventCallbacks() || progressed;
    return progressed;
}

bool IndependentPhaseAsyncServer::shouldWaitForDecodeRefill() const noexcept
{
    size_t const queuedDecode = mCoordinator.scheduler().decodeQueueSize();
    size_t pendingDecodeRows{};
    for (auto const& ticket : mSamplingTickets)
    {
        if (!ticket->fromPrefill)
        {
            pendingDecodeRows += ticket->requestIds.size();
        }
    }
    size_t const refillTarget
        = !mConfig.enableAdaptiveAdmission || mThroughputMode ? mConfig.decodeRefillBatchSize : 0U;
    return shouldDeferDecodeForSamplingRefill(
        refillTarget, mCoordinator.scheduler().prefillQueueSize(), queuedDecode, pendingDecodeRows);
}

size_t IndependentPhaseAsyncServer::admissionLimit() const noexcept
{
    return !mConfig.enableAdaptiveAdmission || mThroughputMode ? mConfig.maxInFlightRequests
                                                               : mConfig.latencyInFlightRequests;
}

void IndependentPhaseAsyncServer::updateAdaptiveAdmissionMode() noexcept
{
    if (!mConfig.enableAdaptiveAdmission)
    {
        return;
    }
    bool const next = nextAdaptiveThroughputMode(mThroughputMode, mPendingRequests.size(), mRequests.size(),
        mConfig.latencyInFlightRequests, mConfig.adaptiveBacklogEnterRequests);
    if (next != mThroughputMode)
    {
        mThroughputMode = next;
        ++mThroughputModeTransitionCount;
    }
    mCoordinator.scheduler().setOnlineDecodeCostLearningActive(mThroughputMode);
}

void IndependentPhaseAsyncServer::runUntilIdle(size_t maxPolls)
{
    for (size_t pollCount{}; pollCount < maxPolls && !empty(); ++pollCount)
    {
        if (!poll())
        {
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    ELLM_CHECK(empty(), "Independent phase server exceeded its poll guard");
}

std::optional<IndependentPhaseServerCompletion> IndependentPhaseAsyncServer::tryPopCompletion()
{
    if (mCompletions.empty())
    {
        return std::nullopt;
    }
    IndependentPhaseServerCompletion result = std::move(mCompletions.front());
    mCompletions.pop_front();
    return result;
}

std::optional<IndependentPhaseServerToken> IndependentPhaseAsyncServer::tryPopToken()
{
    if (mTokenEvents.empty())
    {
        return std::nullopt;
    }
    IndependentPhaseServerToken result = std::move(mTokenEvents.front());
    mTokenEvents.pop_front();
    return result;
}

size_t IndependentPhaseAsyncServer::inFlightCount() const noexcept
{
    return mRequests.size();
}

size_t IndependentPhaseAsyncServer::pendingCount() const noexcept
{
    return mPendingRequests.size();
}

size_t IndependentPhaseAsyncServer::decodeRefillWaitCount() const noexcept
{
    return mDecodeRefillWaitCount;
}

bool IndependentPhaseAsyncServer::throughputMode() const noexcept
{
    return mThroughputMode;
}

size_t IndependentPhaseAsyncServer::throughputModeTransitionCount() const noexcept
{
    return mThroughputModeTransitionCount;
}

IndependentPhaseServerTimingStats const& IndependentPhaseAsyncServer::timingStats() const noexcept
{
    return mTimingStats;
}

bool IndependentPhaseAsyncServer::empty() const noexcept
{
    return mRequests.empty() && mPendingRequests.empty() && mPendingDecodeRequests.empty() && mSamplingTickets.empty()
        && mCoordinator.empty();
}

CUcontext IndependentPhaseAsyncServer::cudaContext() const noexcept
{
    return mCoordinator.cudaContext();
}

bool IndependentPhaseAsyncServer::admitPendingRequests()
{
    bool admitted{};
    while (!mPendingRequests.empty())
    {
        PendingRequest request = std::move(mPendingRequests.front());
        mPendingRequests.pop_front();
        mPendingRequestIds.erase(request.requestId);
        IndependentPhaseServerSubmission const result = submitImpl(request.requestId, request.promptTokens,
            request.visionPayload, request.maxOutputTokens, request.scheduling);
        if (result.status == IndependentPhaseServerStatus::kAdmitted)
        {
            admitted = true;
            continue;
        }
        ELLM_CHECK(result.status == IndependentPhaseServerStatus::kBackpressure,
            "Pending phase request became invalid during admission");
        mPendingRequests.push_front(std::move(request));
        mPendingRequestIds.insert(mPendingRequests.front().requestId);
        break;
    }
    return admitted;
}

bool IndependentPhaseAsyncServer::resumePendingDecodeRequests()
{
    bool resumed{};
    while (!mPendingDecodeRequests.empty())
    {
        uint64_t const requestId = mPendingDecodeRequests.front();
        auto it = mRequests.find(requestId);
        ELLM_CHECK(it != mRequests.end(), "Pending decode request is missing");
        if (!enqueueDecodeOrWait(requestId, it->second))
        {
            break;
        }
        mPendingDecodeRequests.pop_front();
        mPendingDecodeRequestIds.erase(requestId);
        resumed = true;
    }
    return resumed;
}

bool IndependentPhaseAsyncServer::enqueueDecodeOrWait(uint64_t requestId, RequestState& state)
{
    try
    {
        mOwnership.ensureCapacity(state.kvSlotId, mOwnership.length(state.kvSlotId) + 1);
    }
    catch (std::runtime_error const&)
    {
        if (mPendingDecodeRequestIds.insert(requestId).second)
        {
            mPendingDecodeRequests.push_back(requestId);
        }
        return false;
    }
    mCoordinator.enqueueDecode(
        {requestId, mOwnership.length(state.kvSlotId), state.kvSlotId, 0, 0, true, state.scheduling});
    return true;
}

IndependentPhaseCoordinatorCallbacks IndependentPhaseAsyncServer::makeCallbacks()
{
    IndependentPhaseCoordinatorCallbacks callbacks;
    callbacks.stagePrefill = [this](std::vector<PhaseWorkItem> const& batch, PipelineIO& io, cudaStream_t stream) {
        if (mAdapter.stagePrefill)
        {
            mAdapter.stagePrefill(makeViews(batch), io, mCoordinator.prefillTensorMap(), stream);
        }
    };
    callbacks.stageDecode = [this](std::vector<PhaseWorkItem> const& batch, PipelineIO& io, cudaStream_t stream) {
        if (mConfig.enableTimingMetrics)
        {
            auto const dispatchedAt = std::chrono::steady_clock::now();
            if (mTimingStats.decodeBatchHistogram.size() <= batch.size())
            {
                mTimingStats.decodeBatchHistogram.resize(batch.size() + 1U);
            }
            ++mTimingStats.decodeBatchHistogram[batch.size()];
            for (PhaseWorkItem const& item : batch)
            {
                auto const request = mRequests.find(item.requestId);
                ELLM_CHECK(request != mRequests.end(), "Decode timing requested an unknown request");
                if (request->second.decodeReady)
                {
                    double const delayUs
                        = std::chrono::duration<double, std::micro>(dispatchedAt - request->second.decodeReadyAt).count();
                    mTimingStats.readyToDispatchUsTotal += delayUs;
                    mTimingStats.readyToDispatchUsMax = std::max(mTimingStats.readyToDispatchUsMax, delayUs);
                    ++mTimingStats.decodeRowsDispatched;
                    request->second.decodeReady = false;
                }
            }
        }
        if (mAdapter.stageDecode)
        {
            mAdapter.stageDecode(makeViews(batch), io, mCoordinator.decodeTensorMap(), stream);
        }
    };
    callbacks.completePrefillBatch
        = [this](std::vector<PhaseWorkItem> const& batch, PipelineIO& io, cudaStream_t stream) {
              std::vector<IndependentPhaseRequestView> const views = makeViews(batch);
              std::vector<IndependentPhaseRequestView> finalViews;
              for (auto const& view : views)
              {
                  if (view.work.tokenOffset + view.work.tokenCount == static_cast<int32_t>(view.promptTokens->size()))
                  {
                      finalViews.push_back(view);
                  }
              }
              if (!finalViews.empty())
              {
                  std::unique_ptr<IndependentPhaseSampleTicket> ticket
                      = mAdapter.submitSampling(finalViews, io, stream, true);
                  ELLM_CHECK(ticket != nullptr, "Prefill sampling adapter returned no completion ticket");
                  if (mConfig.enableTimingMetrics)
                  {
                      ticket->submittedAt = std::chrono::steady_clock::now();
                  }
                  mSamplingTickets.push_back(std::move(ticket));
              }
          };
    callbacks.completeDecodeBatch
        = [this](std::vector<PhaseWorkItem> const& batch, PipelineIO& io, cudaStream_t stream) {
              std::unique_ptr<IndependentPhaseSampleTicket> ticket
                  = mAdapter.submitSampling(makeViews(batch), io, stream, false);
              ELLM_CHECK(ticket != nullptr, "Decode sampling adapter returned no completion ticket");
              if (mConfig.enableTimingMetrics)
              {
                  ticket->submittedAt = std::chrono::steady_clock::now();
              }
              mSamplingTickets.push_back(std::move(ticket));
          };
    callbacks.isPrefillFinished = [](PhaseWorkItem const& item, int32_t) {
        return item.tokenOffset + item.tokenCount == item.promptTokenCount;
    };
    callbacks.isDecodeFinished = [](PhaseWorkItem const&, int32_t) { return true; };
    return callbacks;
}

std::vector<IndependentPhaseRequestView> IndependentPhaseAsyncServer::makeViews(
    std::vector<PhaseWorkItem> const& batch) const
{
    std::vector<IndependentPhaseRequestView> views;
    views.reserve(batch.size());
    for (PhaseWorkItem const& work : batch)
    {
        auto const it = mRequests.find(work.requestId);
        ELLM_CHECK(it != mRequests.end(), "Phase adapter requested an unknown request");
        views.push_back({work.requestId, work, &it->second.promptTokens, &it->second.generatedTokens,
            it->second.visionPayload.get()});
    }
    return views;
}

bool IndependentPhaseAsyncServer::isEos(int32_t tokenId) const noexcept
{
    return std::find(mConfig.eosTokenIds.begin(), mConfig.eosTokenIds.end(), tokenId) != mConfig.eosTokenIds.end();
}

bool IndependentPhaseAsyncServer::processSamplingTickets()
{
    std::vector<std::unique_ptr<IndependentPhaseSampleTicket>> readyTickets;
    for (auto ticket = mSamplingTickets.begin(); ticket != mSamplingTickets.end();)
    {
        cudaError_t const status = cudaEventQuery((*ticket)->ready);
        if (status == cudaErrorNotReady)
        {
            ++ticket;
            continue;
        }
        CUDA_CHECK(status);
        if (mConfig.enableTimingMetrics)
        {
            auto const readyAt = std::chrono::steady_clock::now();
            double const delayUs
                = std::chrono::duration<double, std::micro>(readyAt - (*ticket)->submittedAt).count();
            mTimingStats.samplingReadyUsTotal += delayUs;
            mTimingStats.samplingReadyUsMax = std::max(mTimingStats.samplingReadyUsMax, delayUs);
            ++mTimingStats.samplingTickets;
        }
        readyTickets.push_back(std::move(*ticket));
        ticket = mSamplingTickets.erase(ticket);
    }
    for (auto& ticket : readyTickets)
    {
        processTicket(std::move(ticket));
    }
    return !readyTickets.empty();
}

bool IndependentPhaseAsyncServer::flushEventCallbacks()
{
    bool delivered{};
    if (mTokenCallback)
    {
        while (!mTokenEvents.empty())
        {
            IndependentPhaseServerToken event = std::move(mTokenEvents.front());
            mTokenEvents.pop_front();
            mTokenCallback(std::move(event));
            delivered = true;
        }
    }
    if (mCompletionCallback)
    {
        while (!mCompletions.empty())
        {
            IndependentPhaseServerCompletion event = std::move(mCompletions.front());
            mCompletions.pop_front();
            mCompletionCallback(std::move(event));
            delivered = true;
        }
    }
    return delivered;
}

void IndependentPhaseAsyncServer::processTicket(std::unique_ptr<IndependentPhaseSampleTicket> ticket)
{
    std::vector<int32_t> const tokens = ticket->collect();
    ELLM_CHECK(tokens.size() == ticket->requestIds.size(), "Phase sampling ticket returned an invalid row count");
    for (size_t index{}; index < tokens.size(); ++index)
    {
        uint64_t const requestId = ticket->requestIds[index];
        auto it = mRequests.find(requestId);
        if (it == mRequests.end())
        {
            continue;
        }
        RequestState& state = it->second;
        state.generatedTokens.push_back(tokens[index]);
        bool const eos = isEos(tokens[index]);
        double const elapsedMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - state.submittedAt).count();
        mTokenEvents.push_back(
            {requestId, tokens[index], static_cast<int32_t>(state.generatedTokens.size() - 1U), eos, elapsedMs});
        if (eos || static_cast<int32_t>(state.generatedTokens.size()) >= state.maxOutputTokens)
        {
            finishRequest(requestId, eos);
        }
        else
        {
            if (mConfig.enableTimingMetrics)
            {
                state.decodeReadyAt = std::chrono::steady_clock::now();
                state.decodeReady = true;
            }
            static_cast<void>(enqueueDecodeOrWait(requestId, state));
        }
    }
    destroyTicketEvent(*ticket);
}

void IndependentPhaseAsyncServer::finishRequest(uint64_t requestId, bool stoppedByEos)
{
    auto it = mRequests.find(requestId);
    ELLM_CHECK(it != mRequests.end(), "Finished phase request is missing");
    RequestState& state = it->second;
    if (mConfig.enablePrefixReuse && mAdapter.supportsPageAlignedPrefixReuse && mPrefixCache != nullptr
        && state.promptTokens.size() >= 128U)
    {
        mPrefixCache->publish(state.kvSlotId, state.promptTokens, mOwnership.length(state.kvSlotId));
    }
    else
    {
        mOwnership.release(state.kvSlotId);
    }
    double const latencyMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - state.submittedAt).count();
    mCompletions.push_back({requestId, std::move(state.generatedTokens),
        static_cast<int32_t>(state.promptTokens.size()), latencyMs, stoppedByEos});
    mRequests.erase(it);
}

void IndependentPhaseAsyncServer::destroyTicketEvent(IndependentPhaseSampleTicket& ticket) noexcept
{
    if (ticket.release)
    {
        ticket.release();
        ticket.release = {};
        ticket.ready = nullptr;
        return;
    }
    if (ticket.ready != nullptr)
    {
        static_cast<void>(cudaEventDestroy(ticket.ready));
        ticket.ready = nullptr;
    }
}

} // namespace trt_edgellm::rt
