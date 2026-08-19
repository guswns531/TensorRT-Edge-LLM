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

#include "common/checkMacros.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace trt_edgellm::rt
{

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
    ELLM_CHECK(static_cast<bool>(mAdapter.submitSampling), "Independent phase server requires a sampling adapter");
    mCoordinator.setGraphCaptureEnabled(mConfig.enableCudaGraphs);
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
    IndependentPhaseServerSubmission result{requestId};
    if (mRequests.find(requestId) != mRequests.end() || mPendingRequestIds.find(requestId) != mPendingRequestIds.end()
        || promptTokens.empty())
    {
        result.status = IndependentPhaseServerStatus::kDuplicateRequest;
        return result;
    }
    if (mRequests.size() >= mConfig.maxInFlightRequests || mOwnership.availableSlots() == 0)
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
        mOwnership.ensureCapacity(slot, static_cast<int32_t>(promptTokens.size()) + maxOutputTokens);
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
    mRequests.emplace(requestId, std::move(state));
    int32_t const remaining = static_cast<int32_t>(mRequests.at(requestId).promptTokens.size()) - reusedPrefixTokens;
    mCoordinator.enqueuePrefill({requestId, remaining, slot, reusedPrefixTokens,
        static_cast<int32_t>(mRequests.at(requestId).promptTokens.size()), true, scheduling});
    result.status = IndependentPhaseServerStatus::kAdmitted;
    result.kvSlotId = slot;
    result.reusedPrefixTokens = reusedPrefixTokens;
    return result;
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitOrQueue(
    uint64_t requestId, std::vector<int32_t> promptTokens, int32_t maxOutputTokens, PhaseSchedulingHints scheduling)
{
    IndependentPhaseServerSubmission result = submit(requestId, promptTokens, maxOutputTokens, scheduling);
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
    mPendingRequests.push_back({requestId, std::move(promptTokens), maxOutputTokens, scheduling});
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
    if (!mCoordinator.scheduler().cancel(requestId))
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

bool IndependentPhaseAsyncServer::poll()
{
    bool progressed = admitPendingRequests();
    progressed = mCoordinator.poll() || progressed;
    processSamplingTickets();
    progressed = admitPendingRequests() || progressed;
    if (!mCoordinator.busy() && !mCoordinator.empty())
    {
        progressed = mCoordinator.dispatchNext() || progressed;
    }
    return progressed;
}

void IndependentPhaseAsyncServer::runUntilIdle(size_t maxPolls)
{
    for (size_t pollCount{}; pollCount < maxPolls && !empty(); ++pollCount)
    {
        if (!poll())
        {
            std::this_thread::yield();
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

bool IndependentPhaseAsyncServer::empty() const noexcept
{
    return mRequests.empty() && mPendingRequests.empty() && mSamplingTickets.empty() && mCoordinator.empty();
}

bool IndependentPhaseAsyncServer::admitPendingRequests()
{
    bool admitted{};
    while (!mPendingRequests.empty())
    {
        PendingRequest request = std::move(mPendingRequests.front());
        mPendingRequests.pop_front();
        mPendingRequestIds.erase(request.requestId);
        IndependentPhaseServerSubmission const result
            = submit(request.requestId, request.promptTokens, request.maxOutputTokens, request.scheduling);
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
                  mSamplingTickets.push_back(std::move(ticket));
              }
          };
    callbacks.completeDecodeBatch
        = [this](std::vector<PhaseWorkItem> const& batch, PipelineIO& io, cudaStream_t stream) {
              std::unique_ptr<IndependentPhaseSampleTicket> ticket
                  = mAdapter.submitSampling(makeViews(batch), io, stream, false);
              ELLM_CHECK(ticket != nullptr, "Decode sampling adapter returned no completion ticket");
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
        views.push_back({work.requestId, work, &it->second.promptTokens, &it->second.generatedTokens});
    }
    return views;
}

bool IndependentPhaseAsyncServer::isEos(int32_t tokenId) const noexcept
{
    return std::find(mConfig.eosTokenIds.begin(), mConfig.eosTokenIds.end(), tokenId) != mConfig.eosTokenIds.end();
}

void IndependentPhaseAsyncServer::processSamplingTickets()
{
    while (!mSamplingTickets.empty())
    {
        std::unique_ptr<IndependentPhaseSampleTicket>& ticket = mSamplingTickets.front();
        cudaError_t const status = cudaEventQuery(ticket->ready);
        if (status == cudaErrorNotReady)
        {
            break;
        }
        CUDA_CHECK(status);
        std::unique_ptr<IndependentPhaseSampleTicket> ready = std::move(ticket);
        mSamplingTickets.pop_front();
        processTicket(std::move(ready));
    }
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
            mCoordinator.enqueueDecode(
                {requestId, mOwnership.length(state.kvSlotId), state.kvSlotId, 0, 0, true, state.scheduling});
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
    mCompletions.push_back({requestId, std::move(state.generatedTokens), latencyMs, stoppedByEos});
    mRequests.erase(it);
}

void IndependentPhaseAsyncServer::destroyTicketEvent(IndependentPhaseSampleTicket& ticket) noexcept
{
    if (ticket.ready != nullptr)
    {
        static_cast<void>(cudaEventDestroy(ticket.ready));
        ticket.ready = nullptr;
    }
}

} // namespace trt_edgellm::rt
