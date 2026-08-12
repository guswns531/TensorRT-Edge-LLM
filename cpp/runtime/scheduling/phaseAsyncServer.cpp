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

#include "runtime/scheduling/phaseAsyncServer.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "sampler/sampling.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

PhaseAsyncServer::PhaseAsyncServer(PhaseAsyncServerConfig config, PhaseContextServingFacade& servingFacade,
    tokenizer::Tokenizer const& tokenizer, cudaStream_t requestStream)
    : mConfig(config)
    , mServingFacade(servingFacade)
    , mTokenizer(tokenizer)
    , mRequestStream(requestStream)
{
    check::check(mConfig.maxInFlightRequests > 0, "Async server maxInFlightRequests must be positive.");
    mServingFacade.configurePageReservation(mConfig.pageReservation);
    check::check(mRequestStream != nullptr, "Async server requires an explicit request CUDA stream.");
    CUcontext requestContext{};
    CUDA_DRIVER_CHECK(cuStreamGetCtx(mRequestStream, &requestContext));
    check::check(requestContext != nullptr && requestContext == mServingFacade.cudaContext(),
        "Async text server and both LLM phase streams must share one CUDA context.");
    mTerminalObserverId = mServingFacade.addTerminalObserver(
        [this](PhaseRequestSnapshot const& snapshot) { mTerminalSnapshots.push_back(snapshot); });
}

PhaseAsyncServer::PhaseAsyncServer(PhaseAsyncServerConfig config, PhaseOnlineCoordinator& coordinator,
    PhaseEncoderDispatchWorker& encoderWorker, PhaseContextServingFacade& servingFacade,
    tokenizer::Tokenizer const& tokenizer, cudaStream_t requestStream, PhaseVisionAdapter* visionAdapter)
    : mConfig(config)
    , mCoordinator(&coordinator)
    , mEncoderWorker(&encoderWorker)
    , mServingFacade(servingFacade)
    , mTokenizer(tokenizer)
    , mRequestStream(requestStream)
    , mVisionAdapter(visionAdapter)
{
    check::check(mConfig.maxInFlightRequests > 0, "Async server maxInFlightRequests must be positive.");
    mServingFacade.configurePageReservation(mConfig.pageReservation);
    check::check(mRequestStream != nullptr, "Async server requires an explicit request CUDA stream.");
    CUcontext requestContext{};
    CUDA_DRIVER_CHECK(cuStreamGetCtx(mRequestStream, &requestContext));
    check::check(requestContext != nullptr && requestContext == mServingFacade.cudaContext()
            && requestContext == mEncoderWorker->cudaContext(),
        "Async server and all phase workers must share one CUDA context.");
    mTerminalObserverId = mServingFacade.addTerminalObserver(
        [this](PhaseRequestSnapshot const& snapshot) { mTerminalSnapshots.push_back(snapshot); });
}

PhaseAsyncServer::~PhaseAsyncServer() noexcept
{
    // Server-owned contexts must not disappear while a phase worker still borrows them.
    // Convert every outstanding request to cancellation and synchronously retire GPU work.
    try
    {
        for (auto& [requestId, state] : mRequests)
        {
            static_cast<void>(requestId);
            state->cancelRequested = true;
        }
        while (!mRequests.empty())
        {
            processCancellations();
            processTerminals();
            if (mEncoderWorker != nullptr && mEncoderWorker->busy())
            {
                mEncoderWorker->wait();
            }
            else if (mServingFacade.busy())
            {
                mServingFacade.wait();
            }
            else
            {
                if (mCoordinator != nullptr)
                {
                    static_cast<void>(mCoordinator->step());
                }
                else if (mServingFacade.hasQueuedPhaseWork())
                {
                    static_cast<void>(mServingFacade.dispatchNext());
                }
            }
            processTerminals();
        }
    }
    catch (...)
    {
        // Destructors cannot report runtime failures; normal production shutdown must drain explicitly.
    }
    mServingFacade.removeTerminalObserver(mTerminalObserverId);
}

void PhaseAsyncServer::validateRequest(LLMGenerationRequest const& request) const
{
    check::check(request.requests.size() == 1, "Async phase server accepts one logical request per submission.");
    check::check(request.maxGenerateLength > 0 && request.maxGenerateLength <= std::numeric_limits<int32_t>::max(),
        "Async phase request maxGenerateLength is invalid.");
    check::check(request.loraWeightsName.empty(), "Async phase server v1 does not support LoRA.");
    check::check(!request.saveSystemPromptKVCache, "Async phase server v1 does not support system-prompt caching.");
    check::check(request.disableSpecDecode, "Async phase server v1 requires speculative decoding to be disabled.");
    check::check(!shouldUseNonGreedySampling(request.temperature, request.topK, request.topP),
        "Async phase server v1 supports greedy sampling only.");
    check::check(request.numLogprobs == 0, "Async phase server v1 does not support logprobs.");
    check::check(request.streamChannels.empty() && !request.onTokenGenerated.has_value(),
        "Async phase server owns completion delivery and does not accept legacy streaming callbacks.");
    check::check(!request.generateAudio && request.requests.front().audioBuffers.empty(),
        "Async phase server v1 does not support audio.");
    check::check(request.requests.front().stopStrings.empty(), "Async phase server v1 does not support stop strings.");
    check::check(request.requests.front().logitBias.empty(), "Async phase server v1 does not support logit bias.");
    check::check(!request.requests.front().pastTrajectory.has_value(),
        "Async phase server v1 does not support trajectory inputs.");
}

void PhaseAsyncServer::prepareRequest(RequestState& state)
{
    LLMGenerationRequest& request = state.request;
    request.formattedRequests.resize(1);
    check::check(mTokenizer.applyChatTemplate(request.requests.front(), request.formattedRequests.front(),
                     request.applyChatTemplate, request.addGenerationPrompt, request.enableThinking),
        "Failed to apply the chat template for an async phase request.");

    state.context.initialize(1, static_cast<int32_t>(request.maxGenerateLength), std::nullopt, OptionalInputTensors{},
        request.loraWeightsName, mRequestStream);
    state.context.temperature = request.temperature;
    state.context.topP = request.topP;
    state.context.topK = request.topK;
    state.context.systemPrompts.front() = request.formattedRequests.front().formattedSystemPrompt;
    state.usesVision = !request.requests.front().imageBuffers.empty();
    if (!state.usesVision)
    {
        std::vector<int32_t> tokenIds
            = mTokenizer.encode(request.formattedRequests.front().formattedCompleteRequest, false);
        check::check(!tokenIds.empty(), "Async phase request produced an empty text prompt.");
        state.context.rawBatchedInputIds = {std::move(tokenIds)};
        state.context.tokenIds = state.context.rawBatchedInputIds;
    }
    else
    {
        check::check(mVisionAdapter != nullptr, "Async phase request contains images but no vision adapter exists.");
    }
}

PhaseRequestStatus PhaseAsyncServer::admit(uint64_t requestId, RequestState& state)
{
    if (!state.usesVision)
    {
        int32_t const promptTokens = static_cast<int32_t>(state.context.rawBatchedInputIds.front().size());
        PhaseAdmissionResult const result
            = mServingFacade.submitOrQueue(requestId, state.context, 0, promptTokens, state.scheduling);
        return result.status == PhaseAdmissionStatus::kAdmitted ? PhaseRequestStatus::kPrefill
                                                                : PhaseRequestStatus::kPending;
    }

    if (mServingFacade.availableSlotCount() == 0)
    {
        state.waitingForVisionSlot = true;
        mWaitingVisionAdmissions.push_back(requestId);
        return PhaseRequestStatus::kPending;
    }

    std::vector<int32_t> const estimate
        = mTokenizer.encode(state.request.formattedRequests.front().formattedCompleteRequest, false);
    int32_t const promptEstimate = std::max<int32_t>(1, static_cast<int32_t>(estimate.size()));
    int32_t const kvSlot
        = mServingFacade.reserveForEncoder(requestId, state.context, 0, promptEstimate, state.scheduling);
    try
    {
        mVisionAdapter->registerRequest(requestId, state.request, state.context);
        int32_t const imageCount = static_cast<int32_t>(state.request.requests.front().imageBuffers.size());
        check::check(mEncoderWorker != nullptr, "Async phase request contains images but no encoder worker exists.");
        mEncoderWorker->submit({requestId, std::max(1, imageCount), kvSlot});
    }
    catch (...)
    {
        if (mVisionAdapter->hasRequest(requestId))
        {
            mVisionAdapter->release(requestId);
        }
        static_cast<void>(mServingFacade.cancel(requestId));
        throw;
    }
    state.waitingForVisionSlot = false;
    return PhaseRequestStatus::kEncoder;
}

PhaseAsyncSubmission PhaseAsyncServer::submit(LLMGenerationRequest request, PhaseSchedulingHints scheduling)
{
    validateRequest(request);
    check::check(mRequests.size() < mConfig.maxInFlightRequests, "Async server admission capacity is full.");
    uint64_t const requestId = mNextRequestId++;
    auto state = std::make_unique<RequestState>();
    state->request = std::move(request);
    state->submittedAt = std::chrono::steady_clock::now();
    scheduling.submittedAt = state->submittedAt;
    state->scheduling = scheduling;
    prepareRequest(*state);
    auto const [it, inserted] = mRequests.emplace(requestId, std::move(state));
    check::check(inserted, "Async server generated a duplicate request ID.");
    try
    {
        PhaseRequestStatus const requestStatus = admit(requestId, *it->second);
        return {requestId, requestStatus};
    }
    catch (...)
    {
        mRequests.erase(it);
        throw;
    }
}

bool PhaseAsyncServer::cancel(uint64_t requestId)
{
    auto const found = mRequests.find(requestId);
    if (found == mRequests.end())
    {
        return false;
    }
    found->second->cancelRequested = true;
    static_cast<void>(processCancellations());
    processTerminals();
    return true;
}

bool PhaseAsyncServer::admitWaitingVisionRequests()
{
    bool progressed{};
    while (!mWaitingVisionAdmissions.empty() && mServingFacade.availableSlotCount() > 0)
    {
        uint64_t const requestId = mWaitingVisionAdmissions.front();
        mWaitingVisionAdmissions.pop_front();
        auto const found = mRequests.find(requestId);
        if (found == mRequests.end() || found->second->cancelRequested)
        {
            continue;
        }
        static_cast<void>(admit(requestId, *found->second));
        progressed = true;
    }
    return progressed;
}

bool PhaseAsyncServer::processCancellations()
{
    bool progressed{};
    std::vector<uint64_t> cancelWithoutLease;
    for (auto& [requestId, state] : mRequests)
    {
        if (!state->cancelRequested)
        {
            continue;
        }
        if (state->waitingForVisionSlot)
        {
            state->waitingForVisionSlot = false;
            cancelWithoutLease.push_back(requestId);
            continue;
        }
        if (mEncoderWorker != nullptr && mEncoderWorker->hasRequest(requestId) && !mEncoderWorker->cancel(requestId))
        {
            continue;
        }
        if (mServingFacade.cancel(requestId))
        {
            progressed = true;
        }
    }
    for (uint64_t const requestId : cancelWithoutLease)
    {
        complete(requestId, PhaseRequestStatus::kCancelled);
        progressed = true;
    }
    return progressed;
}

void PhaseAsyncServer::processTerminals()
{
    while (!mTerminalSnapshots.empty())
    {
        PhaseRequestSnapshot const snapshot = mTerminalSnapshots.front();
        mTerminalSnapshots.pop_front();
        complete(snapshot.requestId, snapshot.status);
    }
}

void PhaseAsyncServer::complete(uint64_t requestId, PhaseRequestStatus status)
{
    auto const found = mRequests.find(requestId);
    if (found == mRequests.end())
    {
        return;
    }
    RequestState& state = *found->second;
    LLMGenerationResponse response;
    response.outputIds.resize(1);
    response.outputTexts.resize(1);
    response.logprobs.resize(1);
    response.outputTrajectories.resize(1);
    response.finishReasons.resize(1, FinishReason::kCancelled);
    if (status != PhaseRequestStatus::kCancelled)
    {
        int32_t const generateLength = state.context.currentGenerateLengths.front();
        std::vector<int32_t> const& allTokens = state.context.tokenIds.front();
        check::check(generateLength >= 0 && static_cast<size_t>(generateLength) <= allTokens.size(),
            "Async phase completion has an invalid generated-token count.");
        response.outputIds.front().assign(allTokens.end() - generateLength, allTokens.end());
        response.outputTexts.front() = mTokenizer.decode(response.outputIds.front(), true);
        bool const endedByEos
            = !response.outputIds.front().empty() && mTokenizer.isEosToken(response.outputIds.front().back());
        response.finishReasons.front() = endedByEos ? FinishReason::kEndId : FinishReason::kLength;
    }
    if (state.usesVision && mVisionAdapter->hasRequest(requestId))
    {
        mVisionAdapter->release(requestId);
    }
    double const latencyMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - state.submittedAt).count();
    mCompletions.push_back({requestId, std::move(response), latencyMs});
    mRequests.erase(found);
}

bool PhaseAsyncServer::poll()
{
    bool progressed = admitWaitingVisionRequests();
    progressed = processCancellations() || progressed;
    if (mCoordinator != nullptr)
    {
        progressed = mCoordinator->step() || progressed;
    }
    else
    {
        if (mServingFacade.busy())
        {
            progressed = mServingFacade.poll() || progressed;
        }
        if (!mServingFacade.busy() && mServingFacade.hasQueuedPhaseWork())
        {
            progressed = mServingFacade.dispatchNext() || progressed;
        }
    }
    processTerminals();
    progressed = processCancellations() || progressed;
    progressed = admitWaitingVisionRequests() || progressed;
    return progressed;
}

std::optional<PhaseAsyncCompletion> PhaseAsyncServer::tryPopCompletion()
{
    if (mCompletions.empty())
    {
        return std::nullopt;
    }
    PhaseAsyncCompletion result = std::move(mCompletions.front());
    mCompletions.pop_front();
    return result;
}

std::optional<PhaseRequestStatus> PhaseAsyncServer::status(uint64_t requestId) const
{
    auto const found = mRequests.find(requestId);
    if (found == mRequests.end())
    {
        return std::nullopt;
    }
    if (found->second->waitingForVisionSlot)
    {
        return PhaseRequestStatus::kPending;
    }
    std::optional<PhaseRequestSnapshot> const snapshot = mServingFacade.request(requestId);
    return snapshot.has_value() ? std::optional{snapshot->status} : std::optional{PhaseRequestStatus::kEncoder};
}

size_t PhaseAsyncServer::inFlightCount() const noexcept
{
    return mRequests.size();
}

size_t PhaseAsyncServer::completionCount() const noexcept
{
    return mCompletions.size();
}

bool PhaseAsyncServer::empty() const noexcept
{
    return mRequests.empty();
}

} // namespace rt
} // namespace trt_edgellm
