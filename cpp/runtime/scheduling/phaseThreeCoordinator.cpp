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
#include <chrono>
#include <cmath>
#include <memory>
#include <utility>

namespace trt_edgellm::rt
{

PhaseSchedulingHints phaseVisionSchedulingHints(
    PhaseSchedulingHints scheduling, double defaultTtftTargetUs, std::chrono::steady_clock::time_point now)
{
    if (scheduling.submittedAt == std::chrono::steady_clock::time_point{})
    {
        scheduling.submittedAt = now;
    }
    if (scheduling.ttftTargetUs == 0.0)
    {
        scheduling.ttftTargetUs = defaultTtftTargetUs;
    }
    return scheduling;
}

bool phaseVisionEncoderCapacityAvailable(size_t downstreamRequests, size_t maxDownstreamRequests,
    size_t downstreamBytes, size_t maxDownstreamBytes, size_t estimatedPayloadBytes) noexcept
{
    if (downstreamRequests >= maxDownstreamRequests)
    {
        return false;
    }
    if (maxDownstreamBytes == 0 || downstreamRequests == 0 || estimatedPayloadBytes == 0)
    {
        return true;
    }
    return downstreamBytes <= maxDownstreamBytes && estimatedPayloadBytes <= maxDownstreamBytes - downstreamBytes;
}

PhaseThreeCoordinator::PhaseThreeCoordinator(
    PhaseVisionAdapter& vision, IndependentPhaseAsyncServer& server, PhaseThreeCoordinatorConfig config)
    : mVision(vision)
    , mServer(server)
    , mConfig(config)
{
    ELLM_CHECK(mConfig.maxEncodedInFlight > 0, "Three-phase encoded request capacity must be positive");
    ELLM_CHECK(std::isfinite(mConfig.visionTtftTargetUs) && mConfig.visionTtftTargetUs >= 0.0,
        "Three-phase vision TTFT target must be finite and non-negative");
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
    scheduling = phaseVisionSchedulingHints(scheduling, mConfig.visionTtftTargetUs);
    PendingVisionRequest pending{requestId, std::move(request), maxOutputTokens, scheduling};
    if (!mEncoding.has_value() && !mVision.busy() && encoderCapacityAvailable())
    {
        mEncoding = std::move(pending);
        mLastEncoderQueueWaitUs = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - mEncoding->scheduling.submittedAt)
                                      .count();
        mMaxEncoderQueueWaitUs = std::max(mMaxEncoderQueueWaitUs, mLastEncoderQueueWaitUs);
        ELLM_CHECK(mVision.submit(requestId, mEncoding->request), "Failed to submit phase encoder request");
        ++mEncoderStarts;
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
        auto const downstream = mDownstreamRequestBytes.find(requestId);
        if (downstream != mDownstreamRequestBytes.end())
        {
            mDownstreamEncodedBytes -= downstream->second;
            mDownstreamRequestBytes.erase(downstream);
        }
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

PhaseThreeCoordinatorMetrics PhaseThreeCoordinator::metrics() const noexcept
{
    PhaseThreeCoordinatorMetrics result;
    result.pendingVisionRequests = mPending.size();
    result.downstreamEncodedRequests = mDownstreamRequestBytes.size();
    result.downstreamEncodedBytes = mDownstreamEncodedBytes;
    result.encoderStarts = mEncoderStarts;
    result.encoderCompletions = mEncoderCompletions;
    result.lastEncoderQueueWaitUs = mLastEncoderQueueWaitUs;
    result.maxEncoderQueueWaitUs = mMaxEncoderQueueWaitUs;
    result.lastEncoderGpuMs = mLastEncoderGpuMs;
    result.maxEncoderGpuMs = mMaxEncoderGpuMs;
    if (!mPending.empty())
    {
        result.oldestPendingAgeUs = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - mPending.front().scheduling.submittedAt)
                                        .count();
    }
    return result;
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
        auto const downstream = mDownstreamRequestBytes.find(completion->requestId);
        if (downstream != mDownstreamRequestBytes.end())
        {
            mDownstreamEncodedBytes -= downstream->second;
            mDownstreamRequestBytes.erase(downstream);
        }
    }
    return completion;
}

bool PhaseThreeCoordinator::startNextEncoder()
{
    if (mEncoding.has_value() || mPending.empty() || !encoderCapacityAvailable())
    {
        return false;
    }
    mEncoding = std::move(mPending.front());
    mPending.pop_front();
    mLastEncoderQueueWaitUs = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - mEncoding->scheduling.submittedAt)
                                  .count();
    mMaxEncoderQueueWaitUs = std::max(mMaxEncoderQueueWaitUs, mLastEncoderQueueWaitUs);
    ELLM_CHECK(mVision.submit(mEncoding->requestId, mEncoding->request), "Failed to start queued encoder request");
    ++mEncoderStarts;
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
    ++mEncoderCompletions;
    mLastEncoderGpuMs = encoded->encoderGpuMs;
    mMaxEncoderGpuMs = std::max(mMaxEncoderGpuMs, mLastEncoderGpuMs);
    if (mCancelRequested.erase(requestId) > 0)
    {
        mRequestIds.erase(requestId);
        mEncoding.reset();
        return true;
    }
    ELLM_CHECK(encoded->tokenIds.size() == 1U && !encoded->tokenIds.front().empty(),
        "Phase encoder must produce one non-empty token row per logical request");
    auto sharedPayload = std::shared_ptr<PhaseVisionPayload>(std::move(encoded));
    size_t const encodedBytes = sharedPayload->byteSize();
    std::vector<int32_t> promptTokens = sharedPayload->tokenIds.front();
    IndependentPhaseServerSubmission const submitted = mServer.submitOrQueueWithVision(requestId,
        std::move(promptTokens), std::move(sharedPayload), mEncoding->maxOutputTokens, mEncoding->scheduling);
    ELLM_CHECK(submitted.status == IndependentPhaseServerStatus::kAdmitted
            || submitted.status == IndependentPhaseServerStatus::kQueued,
        "Encoded phase request could not enter the LLM admission queue");
    ELLM_CHECK(
        mDownstreamRequestBytes.emplace(requestId, encodedBytes).second, "Encoded phase request is already downstream");
    mDownstreamEncodedBytes += encodedBytes;
    mEstimatedEncodedBytes = std::max(mEstimatedEncodedBytes, encodedBytes);
    mEncoding.reset();
    return true;
}

bool PhaseThreeCoordinator::encoderCapacityAvailable() const noexcept
{
    return phaseVisionEncoderCapacityAvailable(mDownstreamRequestBytes.size(), mConfig.maxEncodedInFlight,
        mDownstreamEncodedBytes, mConfig.maxEncodedBytes, mEstimatedEncodedBytes);
}

} // namespace trt_edgellm::rt
