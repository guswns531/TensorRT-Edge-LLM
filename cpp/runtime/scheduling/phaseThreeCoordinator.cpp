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

#include "runtime/scheduling/phaseThreeCoordinator.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <memory>
#include <utility>

namespace trt_edgellm::rt
{

PhaseThreeCoordinator::PhaseThreeCoordinator(
    PhaseVisionAdapter& vision, IndependentPhaseAsyncServer& server, PhaseThreeCoordinatorConfig config)
    : mVision(vision)
    , mServer(server)
    , mConfig(config)
{
    ELLM_CHECK(mConfig.maxEncodedInFlight > 0, "Three-phase encoded request capacity must be positive");
    ELLM_CHECK(mVision.cudaContext() == mServer.cudaContext(),
        "Encoder and LLM phase server must share one CUDA primary context");
}

PhaseThreeSubmissionStatus PhaseThreeCoordinator::submit(
    uint64_t requestId, LLMGenerationRequest request, int32_t maxOutputTokens, PhaseSchedulingHints scheduling)
{
    if (!mRequestIds.insert(requestId).second)
    {
        return PhaseThreeSubmissionStatus::kDuplicateRequest;
    }
    PendingVisionRequest pending{requestId, std::move(request), maxOutputTokens, scheduling};
    if (!mEncoding.has_value() && !mVision.busy() && mDownstreamRequestIds.size() < mConfig.maxEncodedInFlight)
    {
        mEncoding = std::move(pending);
        ELLM_CHECK(mVision.submit(requestId, mEncoding->request), "Failed to submit phase encoder request");
        return PhaseThreeSubmissionStatus::kEncoding;
    }
    mPending.push_back(std::move(pending));
    return PhaseThreeSubmissionStatus::kQueued;
}

bool PhaseThreeCoordinator::cancel(uint64_t requestId)
{
    auto const pending = std::find_if(mPending.begin(), mPending.end(),
        [&](PendingVisionRequest const& request) { return request.requestId == requestId; });
    if (pending != mPending.end())
    {
        mPending.erase(pending);
        mRequestIds.erase(requestId);
        return true;
    }
    if (mEncoding.has_value() && mEncoding->requestId == requestId)
    {
        if (mVision.cancel(requestId))
        {
            mEncoding.reset();
            mRequestIds.erase(requestId);
            return true;
        }
        mCancelRequested.insert(requestId);
        return true;
    }
    bool const cancelled = mServer.cancel(requestId);
    if (cancelled)
    {
        mRequestIds.erase(requestId);
        mDownstreamRequestIds.erase(requestId);
    }
    return cancelled;
}

bool PhaseThreeCoordinator::poll()
{
    bool progressed = completeEncoder();
    progressed = startNextEncoder() || progressed;
    progressed = mServer.poll() || progressed;
    return progressed;
}

bool PhaseThreeCoordinator::empty() const noexcept
{
    return mPending.empty() && !mEncoding.has_value() && mServer.empty();
}

std::optional<IndependentPhaseServerToken> PhaseThreeCoordinator::tryPopToken()
{
    return mServer.tryPopToken();
}

std::optional<IndependentPhaseServerCompletion> PhaseThreeCoordinator::tryPopCompletion()
{
    auto completion = mServer.tryPopCompletion();
    if (completion.has_value())
    {
        mRequestIds.erase(completion->requestId);
        mDownstreamRequestIds.erase(completion->requestId);
    }
    return completion;
}

bool PhaseThreeCoordinator::startNextEncoder()
{
    if (mEncoding.has_value() || mPending.empty() || mDownstreamRequestIds.size() >= mConfig.maxEncodedInFlight)
    {
        return false;
    }
    mEncoding = std::move(mPending.front());
    mPending.pop_front();
    ELLM_CHECK(mVision.submit(mEncoding->requestId, mEncoding->request), "Failed to start queued encoder request");
    return true;
}

bool PhaseThreeCoordinator::completeEncoder()
{
    if (!mEncoding.has_value() || !mVision.ready(mEncoding->requestId))
    {
        return false;
    }
    uint64_t const requestId = mEncoding->requestId;
    std::unique_ptr<PhaseVisionPayload> encoded = mVision.take(requestId);
    if (mCancelRequested.erase(requestId) > 0)
    {
        mRequestIds.erase(requestId);
        mEncoding.reset();
        return true;
    }
    ELLM_CHECK(encoded->tokenIds.size() == 1U && !encoded->tokenIds.front().empty(),
        "Phase encoder must produce one non-empty token row per logical request");
    auto sharedPayload = std::shared_ptr<PhaseVisionPayload>(std::move(encoded));
    std::vector<int32_t> promptTokens = sharedPayload->tokenIds.front();
    IndependentPhaseServerSubmission const submitted = mServer.submitOrQueueWithVision(requestId,
        std::move(promptTokens), std::move(sharedPayload), mEncoding->maxOutputTokens, mEncoding->scheduling);
    ELLM_CHECK(submitted.status == IndependentPhaseServerStatus::kAdmitted
            || submitted.status == IndependentPhaseServerStatus::kQueued,
        "Encoded phase request could not enter the LLM admission queue");
    ELLM_CHECK(mDownstreamRequestIds.insert(requestId).second, "Encoded phase request is already downstream");
    mEncoding.reset();
    return true;
}

} // namespace trt_edgellm::rt
