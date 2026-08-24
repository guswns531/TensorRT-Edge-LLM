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
    size_t downstreamBytes, size_t maxDownstreamBytes, size_t estimatedPayloadBytes, size_t additionalRequests) noexcept
{
    if (additionalRequests == 0 || downstreamRequests > maxDownstreamRequests
        || additionalRequests > maxDownstreamRequests - downstreamRequests)
    {
        return false;
    }
    if (maxDownstreamBytes == 0 || estimatedPayloadBytes == 0)
    {
        return true;
    }
    if (downstreamRequests == 0 && additionalRequests == 1U)
    {
        return true;
    }
    if (downstreamBytes > maxDownstreamBytes)
    {
        return false;
    }
    size_t const remainingBytes = maxDownstreamBytes - downstreamBytes;
    return estimatedPayloadBytes <= remainingBytes / additionalRequests;
}

PhaseThreeCoordinator::PhaseThreeCoordinator(
    PhaseVisionAdapter& vision, IndependentPhaseAsyncServer& server, PhaseThreeCoordinatorConfig config)
    : mVision(vision)
    , mServer(server)
    , mConfig(config)
{
    ELLM_CHECK(mConfig.maxEncodedInFlight > 0, "Three-phase encoded request capacity must be positive");
    ELLM_CHECK(mConfig.maxEncoderBatchSize > 0, "Three-phase encoder batch size must be positive");
    ELLM_CHECK(mConfig.maxEncoderBatchSize <= mConfig.maxEncodedInFlight,
        "Three-phase encoder batch size cannot exceed downstream encoded capacity");
    ELLM_CHECK(std::isfinite(mConfig.encoderBatchWaitUs) && mConfig.encoderBatchWaitUs >= 0.0,
        "Three-phase encoder batch wait must be finite and non-negative");
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
    mPending.push_back({requestId, std::move(request), maxOutputTokens, scheduling});
    bool const started = startNextEncoder();
    bool const encodingThisRequest = started
        && std::any_of(mEncoding.begin(), mEncoding.end(),
            [&](PendingVisionRequest const& encoding) { return encoding.requestId == requestId; });
    return encodingThisRequest ? PhaseThreeSubmissionStatus::kEncoding : PhaseThreeSubmissionStatus::kQueued;
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
    auto const encoding = std::find_if(mEncoding.begin(), mEncoding.end(),
        [&](PendingVisionRequest const& request) { return request.requestId == requestId; });
    if (encoding != mEncoding.end())
    {
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
    mVision.reclaimIdleStorage();
    return progressed;
}

bool PhaseThreeCoordinator::empty() const noexcept
{
    return mPending.empty() && mEncoding.empty() && mServer.empty();
}

PhaseThreeCoordinatorMetrics PhaseThreeCoordinator::metrics() const noexcept
{
    PhaseThreeCoordinatorMetrics result;
    result.pendingVisionRequests = mPending.size();
    result.downstreamEncodedRequests = mDownstreamRequestBytes.size();
    result.downstreamEncodedBytes = mDownstreamEncodedBytes;
    result.encoderStarts = mEncoderStarts;
    result.encoderCompletions = mEncoderCompletions;
    result.encoderBatches = mEncoderBatches;
    result.lastEncoderBatchSize = mLastEncoderBatchSize;
    result.maxEncoderBatchSize = mMaxEncoderBatchSize;
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
    if (!mEncoding.empty() || mPending.empty() || mVision.busy())
    {
        return false;
    }
    size_t const batchSize = nextEncoderBatchSize();
    if (batchSize == 0)
    {
        return false;
    }

    std::vector<PhaseVisionSubmission> submissions;
    submissions.reserve(batchSize);
    auto const now = std::chrono::steady_clock::now();
    for (size_t index = 0; index < batchSize; ++index)
    {
        mEncoding.push_back(std::move(mPending.front()));
        mPending.pop_front();
        PendingVisionRequest& encoding = mEncoding.back();
        double const queueWaitUs
            = std::chrono::duration<double, std::micro>(now - encoding.scheduling.submittedAt).count();
        mLastEncoderQueueWaitUs = queueWaitUs;
        mMaxEncoderQueueWaitUs = std::max(mMaxEncoderQueueWaitUs, queueWaitUs);
        submissions.push_back({encoding.requestId, std::move(encoding.request)});
    }
    ELLM_CHECK(mVision.submit(std::move(submissions)), "Failed to start queued encoder batch");
    mEncoderStarts += batchSize;
    ++mEncoderBatches;
    mLastEncoderBatchSize = batchSize;
    mMaxEncoderBatchSize = std::max(mMaxEncoderBatchSize, batchSize);
    return true;
}

bool PhaseThreeCoordinator::completeEncoder()
{
    if (mEncoding.empty())
    {
        return false;
    }
    bool const batchReady = std::all_of(mEncoding.begin(), mEncoding.end(),
        [&](PendingVisionRequest const& encoding) { return mVision.ready(encoding.requestId); });
    if (!batchReady)
    {
        return false;
    }
    for (PendingVisionRequest& encoding : mEncoding)
    {
        uint64_t const requestId = encoding.requestId;
        std::unique_ptr<PhaseVisionPayload> encoded = mVision.take(requestId);
        ++mEncoderCompletions;
        mLastEncoderGpuMs = encoded->encoderGpuMs;
        mMaxEncoderGpuMs = std::max(mMaxEncoderGpuMs, mLastEncoderGpuMs);
        if (mCancelRequested.erase(requestId) > 0)
        {
            mRequestIds.erase(requestId);
            continue;
        }
        ELLM_CHECK(encoded->tokenIds.size() == 1U && !encoded->tokenIds.front().empty(),
            "Phase encoder must produce one non-empty token row per logical request");
        auto sharedPayload = std::shared_ptr<PhaseVisionPayload>(std::move(encoded));
        size_t const encodedBytes = sharedPayload->byteSize();
        std::vector<int32_t> promptTokens = sharedPayload->tokenIds.front();
        IndependentPhaseServerSubmission const submitted = mServer.submitOrQueueWithVision(requestId,
            std::move(promptTokens), std::move(sharedPayload), encoding.maxOutputTokens, encoding.scheduling);
        ELLM_CHECK(submitted.status == IndependentPhaseServerStatus::kAdmitted
                || submitted.status == IndependentPhaseServerStatus::kQueued,
            "Encoded phase request could not enter the LLM admission queue");
        ELLM_CHECK(mDownstreamRequestBytes.emplace(requestId, encodedBytes).second,
            "Encoded phase request is already downstream");
        mDownstreamEncodedBytes += encodedBytes;
        mEstimatedEncodedBytes = std::max(mEstimatedEncodedBytes, encodedBytes);
    }
    mEncoding.clear();
    return true;
}

size_t PhaseThreeCoordinator::nextEncoderBatchSize() const noexcept
{
    size_t batchSize{};
    size_t mediaItems{};
    size_t const limit = std::min(mConfig.maxEncoderBatchSize, mPending.size());
    for (size_t index = 0; index < limit; ++index)
    {
        size_t const candidateMediaItems = mediaItemCount(mPending[index]);
        bool const exceedsMediaLimit = mConfig.maxEncoderMediaItems > 0 && batchSize > 0
            && candidateMediaItems > mConfig.maxEncoderMediaItems - std::min(mediaItems, mConfig.maxEncoderMediaItems);
        if (exceedsMediaLimit || !encoderCapacityAvailable(batchSize + 1U))
        {
            break;
        }
        mediaItems += candidateMediaItems;
        ++batchSize;
    }
    if (batchSize == 0)
    {
        return 0;
    }

    bool const batchFull = batchSize == mConfig.maxEncoderBatchSize;
    bool const mediaFull = mConfig.maxEncoderMediaItems > 0 && mediaItems >= mConfig.maxEncoderMediaItems;
    bool const capacityFull = !encoderCapacityAvailable(batchSize + 1U);
    double const oldestWaitUs = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - mPending.front().scheduling.submittedAt)
                                    .count();
    if (!batchFull && !mediaFull && !capacityFull && oldestWaitUs < mConfig.encoderBatchWaitUs)
    {
        return 0;
    }
    return batchSize;
}

bool PhaseThreeCoordinator::encoderCapacityAvailable(size_t additionalRequests) const noexcept
{
    return phaseVisionEncoderCapacityAvailable(mDownstreamRequestBytes.size(), mConfig.maxEncodedInFlight,
        mDownstreamEncodedBytes, mConfig.maxEncodedBytes, mEstimatedEncodedBytes, additionalRequests);
}

size_t PhaseThreeCoordinator::mediaItemCount(PendingVisionRequest const& pending) noexcept
{
    size_t result{};
    for (LLMGenerationRequest::Request const& request : pending.request.requests)
    {
        result += request.imageBuffers.size();
    }
    return result;
}

} // namespace trt_edgellm::rt
