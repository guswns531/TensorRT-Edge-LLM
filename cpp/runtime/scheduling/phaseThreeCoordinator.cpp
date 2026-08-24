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

size_t phaseVisionReadyPrefillBatchSize(std::vector<int32_t> const& promptTokenCounts, size_t maxBatchSize,
    size_t maxBatchTokens, double oldestWaitUs, double batchWaitUs) noexcept
{
    if (promptTokenCounts.empty() || maxBatchSize == 0)
    {
        return 0;
    }

    size_t batchSize{};
    size_t batchTokens{};
    size_t const countLimit = std::min(maxBatchSize, promptTokenCounts.size());
    while (batchSize < countLimit)
    {
        int32_t const promptTokens = promptTokenCounts[batchSize];
        if (promptTokens <= 0)
        {
            break;
        }
        size_t const candidateTokens = static_cast<size_t>(promptTokens);
        if (maxBatchTokens > 0 && batchSize > 0
            && candidateTokens > maxBatchTokens - std::min(batchTokens, maxBatchTokens))
        {
            break;
        }
        batchTokens += candidateTokens;
        ++batchSize;
    }
    if (batchSize == 0)
    {
        return 0;
    }
    bool const batchFull = batchSize == maxBatchSize;
    bool const tokenFull = batchSize < promptTokenCounts.size();
    return batchFull || tokenFull || oldestWaitUs >= batchWaitUs ? batchSize : 0;
}

PhaseVisionPrefillAdmissionDecision phaseVisionAdaptiveReadyPrefillDecision(
    std::vector<int32_t> const& promptTokenCounts, size_t maxBatchSize, size_t maxBatchTokens, double oldestWaitUs,
    double batchWaitUs, bool enabled, size_t minBacklogBatchSize, size_t upstreamVisionRequests,
    size_t availableAdmissionSlots, int32_t availableKVPages, float decodeTpotPressure, float decodeTpotPressureLimit,
    size_t readyBytes, size_t maxReadyBytes, double readyBytePressureRatio) noexcept
{
    if (promptTokenCounts.empty() || maxBatchSize == 0 || availableAdmissionSlots == 0 || availableKVPages <= 0)
    {
        return {};
    }
    size_t const admissionLimit = std::min(maxBatchSize, availableAdmissionSlots);
    if (!enabled)
    {
        return {phaseVisionReadyPrefillBatchSize(
                    promptTokenCounts, admissionLimit, maxBatchTokens, oldestWaitUs, batchWaitUs),
            PhaseVisionPrefillAdmissionReason::kLegacy};
    }

    bool const decodeProtected = decodeTpotPressureLimit > 0.0F && decodeTpotPressure >= decodeTpotPressureLimit;
    if (decodeProtected)
    {
        return {phaseVisionReadyPrefillBatchSize(promptTokenCounts, 1U, maxBatchTokens, oldestWaitUs, 0.0),
            PhaseVisionPrefillAdmissionReason::kDecodeProtection};
    }
    bool const bytePressure = maxReadyBytes > 0 && readyBytePressureRatio > 0.0
        && static_cast<double>(readyBytes) >= static_cast<double>(maxReadyBytes) * readyBytePressureRatio;
    if (bytePressure)
    {
        return {phaseVisionReadyPrefillBatchSize(promptTokenCounts, admissionLimit, maxBatchTokens, oldestWaitUs, 0.0),
            PhaseVisionPrefillAdmissionReason::kBytePressure};
    }
    if (promptTokenCounts.size() == 1U && upstreamVisionRequests == 0)
    {
        return {phaseVisionReadyPrefillBatchSize(promptTokenCounts, 1U, maxBatchTokens, oldestWaitUs, 0.0),
            PhaseVisionPrefillAdmissionReason::kLowLoad};
    }
    bool const slotPressure
        = availableAdmissionSlots < maxBatchSize || availableAdmissionSlots - maxBatchSize < maxBatchSize;
    if (slotPressure)
    {
        return {phaseVisionReadyPrefillBatchSize(promptTokenCounts, 1U, maxBatchTokens, oldestWaitUs, 0.0),
            PhaseVisionPrefillAdmissionReason::kCapacity};
    }
    size_t const backlogThreshold = std::max(minBacklogBatchSize, size_t{1});
    if (promptTokenCounts.size() >= backlogThreshold)
    {
        size_t const backlogBatchSize = std::min(admissionLimit, promptTokenCounts.size());
        return {
            phaseVisionReadyPrefillBatchSize(promptTokenCounts, backlogBatchSize, maxBatchTokens, oldestWaitUs, 0.0),
            PhaseVisionPrefillAdmissionReason::kBacklog};
    }
    if (oldestWaitUs >= batchWaitUs)
    {
        return {phaseVisionReadyPrefillBatchSize(promptTokenCounts, admissionLimit, maxBatchTokens, oldestWaitUs, 0.0),
            PhaseVisionPrefillAdmissionReason::kAge};
    }
    return {};
}

size_t phaseVisionEffectiveEncodedCapacity(size_t latencyCapacity, size_t throughputCapacity, bool throughputMode,
    double oldestVisionAgeUs, double visionTtftTargetUs, double escalationRatio, float decodeTpotPressure,
    float decodeTpotPressureLimit) noexcept
{
    size_t const highCapacity = std::max(latencyCapacity, throughputCapacity);
    if (highCapacity == latencyCapacity)
    {
        return latencyCapacity;
    }
    bool const decodeProtected = decodeTpotPressureLimit > 0.0F && decodeTpotPressure >= decodeTpotPressureLimit;
    bool const visionLate = escalationRatio > 0.0 && visionTtftTargetUs > 0.0
        && oldestVisionAgeUs >= visionTtftTargetUs * escalationRatio;
    return !decodeProtected && (throughputMode || visionLate) ? highCapacity : latencyCapacity;
}

PhaseThreeCoordinator::PhaseThreeCoordinator(
    PhaseVisionAdapter& vision, IndependentPhaseAsyncServer& server, PhaseThreeCoordinatorConfig config)
    : mVision(vision)
    , mServer(server)
    , mConfig(config)
{
    ELLM_CHECK(mConfig.maxEncodedInFlight > 0, "Three-phase encoded request capacity must be positive");
    ELLM_CHECK(
        mConfig.throughputMaxEncodedInFlight == 0 || mConfig.throughputMaxEncodedInFlight >= mConfig.maxEncodedInFlight,
        "Three-phase throughput encoded capacity cannot be below the latency capacity");
    ELLM_CHECK(mConfig.maxEncoderBatchSize > 0, "Three-phase encoder batch size must be positive");
    ELLM_CHECK(mConfig.maxEncoderBatchSize <= mConfig.maxEncodedInFlight,
        "Three-phase encoder batch size cannot exceed downstream encoded capacity");
    if (mConfig.maxPrefillBatchSize == 0)
    {
        mConfig.maxPrefillBatchSize = mConfig.maxEncoderBatchSize;
    }
    size_t const maxEncodedCapacity = std::max(mConfig.maxEncodedInFlight, mConfig.throughputMaxEncodedInFlight);
    ELLM_CHECK(mConfig.maxPrefillBatchSize <= maxEncodedCapacity,
        "Three-phase prefill batch size cannot exceed downstream encoded capacity");
    ELLM_CHECK(std::isfinite(mConfig.encoderBatchWaitUs) && mConfig.encoderBatchWaitUs >= 0.0,
        "Three-phase encoder batch wait must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.prefillBatchWaitUs) && mConfig.prefillBatchWaitUs >= 0.0,
        "Three-phase prefill batch wait must be finite and non-negative");
    ELLM_CHECK(
        mConfig.adaptivePrefillMinBatchSize > 0, "Three-phase adaptive prefill minimum batch size must be positive");
    ELLM_CHECK(std::isfinite(mConfig.prefillDecodeTpotPressureLimit) && mConfig.prefillDecodeTpotPressureLimit >= 0.0F,
        "Three-phase prefill decode pressure limit must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.prefillReadyBytePressureRatio) && mConfig.prefillReadyBytePressureRatio >= 0.0
            && mConfig.prefillReadyBytePressureRatio <= 1.0,
        "Three-phase prefill ready byte pressure ratio must be in [0, 1]");
    ELLM_CHECK(std::isfinite(mConfig.visionTtftTargetUs) && mConfig.visionTtftTargetUs >= 0.0,
        "Three-phase vision TTFT target must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.lookaheadEscalationRatio) && mConfig.lookaheadEscalationRatio >= 0.0,
        "Three-phase lookahead escalation ratio must be finite and non-negative");
    ELLM_CHECK(
        std::isfinite(mConfig.lookaheadDecodeTpotPressureLimit) && mConfig.lookaheadDecodeTpotPressureLimit >= 0.0F,
        "Three-phase lookahead decode pressure limit must be finite and non-negative");
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
    auto const ready = std::find_if(mReadyPrefill.begin(), mReadyPrefill.end(),
        [&](ReadyPrefillRequest const& request) { return request.requestId == requestId; });
    if (ready != mReadyPrefill.end())
    {
        ELLM_CHECK(ready->payloadBytes <= mReadyPrefillBytes, "Ready prefill byte accounting underflow");
        mReadyPrefillBytes -= ready->payloadBytes;
        mReadyPrefill.erase(ready);
        mDownstreamRequestBytes.erase(requestId);
        mRequestIds.erase(requestId);
        return true;
    }
    bool const cancelled = mServer.cancel(requestId);
    if (cancelled)
    {
        mRequestIds.erase(requestId);
        auto const downstream = mDownstreamRequestBytes.find(requestId);
        if (downstream != mDownstreamRequestBytes.end())
        {
            mDownstreamRequestBytes.erase(downstream);
        }
    }
    return cancelled;
}

bool PhaseThreeCoordinator::poll()
{
    bool progressed = completeEncoder();
    progressed = startNextEncoder() || progressed;
    progressed = dispatchReadyPrefill() || progressed;
    progressed = mServer.poll() || progressed;
    mVision.reclaimIdleStorage();
    return progressed;
}

bool PhaseThreeCoordinator::empty() const noexcept
{
    return mPending.empty() && mEncoding.empty() && mReadyPrefill.empty() && mServer.empty();
}

PhaseThreeCoordinatorMetrics PhaseThreeCoordinator::metrics() const noexcept
{
    PhaseThreeCoordinatorMetrics result;
    result.pendingVisionRequests = mPending.size();
    result.pendingPrefillReadyRequests = mReadyPrefill.size();
    result.pendingPrefillReadyBytes = mReadyPrefillBytes;
    result.downstreamEncodedRequests = mDownstreamRequestBytes.size();
    result.downstreamEncodedBytes = mReadyPrefillBytes + mServer.visionPayloadBytes();
    result.prefillStorageReleases = mServer.visionPrefillReleaseCount();
    result.prefillStorageReleasedBytes = mServer.visionPrefillReleasedBytes();
    result.encoderStarts = mEncoderStarts;
    result.encoderCompletions = mEncoderCompletions;
    result.encoderBatches = mEncoderBatches;
    result.lastEncoderBatchSize = mLastEncoderBatchSize;
    result.maxEncoderBatchSize = mMaxEncoderBatchSize;
    result.lastEncoderQueueWaitUs = mLastEncoderQueueWaitUs;
    result.maxEncoderQueueWaitUs = mMaxEncoderQueueWaitUs;
    result.lastEncoderGpuMs = mLastEncoderGpuMs;
    result.maxEncoderGpuMs = mMaxEncoderGpuMs;
    result.prefillAdmissionBatches = mPrefillAdmissionBatches;
    result.lastPrefillAdmissionBatchSize = mLastPrefillAdmissionBatchSize;
    result.maxPrefillAdmissionBatchSize = mMaxPrefillAdmissionBatchSize;
    result.adaptivePrefillAdmissions = mAdaptivePrefillAdmissions;
    result.lowLoadPrefillAdmissions = mLowLoadPrefillAdmissions;
    result.backlogPrefillAdmissions = mBacklogPrefillAdmissions;
    result.decodeProtectedPrefillAdmissions = mDecodeProtectedPrefillAdmissions;
    result.capacityProtectedPrefillAdmissions = mCapacityProtectedPrefillAdmissions;
    result.ageForcedPrefillAdmissions = mAgeForcedPrefillAdmissions;
    result.byteForcedPrefillAdmissions = mByteForcedPrefillAdmissions;
    result.lastPrefillReadyQueueWaitUs = mLastPrefillReadyQueueWaitUs;
    result.maxPrefillReadyQueueWaitUs = mMaxPrefillReadyQueueWaitUs;
    result.availablePrefillAdmissionSlots = mServer.availableAdmissionSlots();
    result.availableKVPages = mServer.availableKVPages();
    result.effectiveEncodedCapacity = effectiveEncodedCapacity();
    result.maxEffectiveEncodedCapacity = mMaxEffectiveEncodedCapacity;
    result.lookaheadEscalations = mLookaheadEscalations;
    result.decodeTpotPressure = mServer.decodeTpotPressure();
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

    size_t const encodedCapacity = effectiveEncodedCapacity();
    if (encodedCapacity > mLastEffectiveEncodedCapacity && mLastEffectiveEncodedCapacity > 0)
    {
        ++mLookaheadEscalations;
    }
    mLastEffectiveEncodedCapacity = encodedCapacity;
    mMaxEffectiveEncodedCapacity = std::max(mMaxEffectiveEncodedCapacity, encodedCapacity);

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
        ELLM_CHECK(mDownstreamRequestBytes.emplace(requestId, encodedBytes).second,
            "Encoded phase request is already downstream");
        mReadyPrefill.push_back({requestId, std::move(promptTokens), std::move(sharedPayload), encoding.maxOutputTokens,
            encoding.scheduling, encodedBytes, std::chrono::steady_clock::now()});
        mReadyPrefillBytes += encodedBytes;
        mEstimatedEncodedBytes = std::max(mEstimatedEncodedBytes, encodedBytes);
    }
    mEncoding.clear();
    return true;
}

bool PhaseThreeCoordinator::dispatchReadyPrefill()
{
    PhaseVisionPrefillAdmissionDecision const decision = nextReadyPrefillDecision();
    size_t const batchSize = decision.batchSize;
    if (batchSize == 0)
    {
        return false;
    }

    size_t submitted{};
    auto const now = std::chrono::steady_clock::now();
    while (submitted < batchSize && !mReadyPrefill.empty())
    {
        ReadyPrefillRequest& ready = mReadyPrefill.front();
        IndependentPhaseServerSubmission const result = mServer.submitWithVision(
            ready.requestId, ready.promptTokens, ready.payload, ready.maxOutputTokens, ready.scheduling);
        if (result.status == IndependentPhaseServerStatus::kBackpressure)
        {
            break;
        }
        ELLM_CHECK(result.status == IndependentPhaseServerStatus::kAdmitted
                || result.status == IndependentPhaseServerStatus::kQueued,
            "Ready vision request could not enter the LLM admission queue");
        double const queueWaitUs = std::chrono::duration<double, std::micro>(now - ready.encodedAt).count();
        mLastPrefillReadyQueueWaitUs = queueWaitUs;
        mMaxPrefillReadyQueueWaitUs = std::max(mMaxPrefillReadyQueueWaitUs, queueWaitUs);
        ELLM_CHECK(ready.payloadBytes <= mReadyPrefillBytes, "Ready prefill byte accounting underflow");
        mReadyPrefillBytes -= ready.payloadBytes;
        mReadyPrefill.pop_front();
        ++submitted;
    }
    if (submitted > 0)
    {
        ++mPrefillAdmissionBatches;
        mLastPrefillAdmissionBatchSize = submitted;
        mMaxPrefillAdmissionBatchSize = std::max(mMaxPrefillAdmissionBatchSize, submitted);
        if (mConfig.enableAdaptivePrefillAdmission)
        {
            ++mAdaptivePrefillAdmissions;
            switch (decision.reason)
            {
            case PhaseVisionPrefillAdmissionReason::kLowLoad: ++mLowLoadPrefillAdmissions; break;
            case PhaseVisionPrefillAdmissionReason::kBacklog: ++mBacklogPrefillAdmissions; break;
            case PhaseVisionPrefillAdmissionReason::kDecodeProtection: ++mDecodeProtectedPrefillAdmissions; break;
            case PhaseVisionPrefillAdmissionReason::kCapacity: ++mCapacityProtectedPrefillAdmissions; break;
            case PhaseVisionPrefillAdmissionReason::kAge: ++mAgeForcedPrefillAdmissions; break;
            case PhaseVisionPrefillAdmissionReason::kBytePressure: ++mByteForcedPrefillAdmissions; break;
            case PhaseVisionPrefillAdmissionReason::kLegacy:
            case PhaseVisionPrefillAdmissionReason::kNone: break;
            }
        }
    }
    return submitted > 0;
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

PhaseVisionPrefillAdmissionDecision PhaseThreeCoordinator::nextReadyPrefillDecision() const noexcept
{
    if (mReadyPrefill.empty())
    {
        return {};
    }
    std::vector<int32_t> promptTokenCounts;
    promptTokenCounts.reserve(mReadyPrefill.size());
    for (ReadyPrefillRequest const& ready : mReadyPrefill)
    {
        promptTokenCounts.push_back(static_cast<int32_t>(ready.promptTokens.size()));
    }
    double const oldestWaitUs
        = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - mReadyPrefill.front().encodedAt)
              .count();
    return phaseVisionAdaptiveReadyPrefillDecision(promptTokenCounts, mConfig.maxPrefillBatchSize,
        mConfig.maxPrefillBatchTokens, oldestWaitUs, mConfig.prefillBatchWaitUs, mConfig.enableAdaptivePrefillAdmission,
        mConfig.adaptivePrefillMinBatchSize, mPending.size() + mEncoding.size(), mServer.availableAdmissionSlots(),
        mServer.availableKVPages(), mServer.decodeTpotPressure(), mConfig.prefillDecodeTpotPressureLimit,
        mReadyPrefillBytes, mConfig.maxEncodedBytes, mConfig.prefillReadyBytePressureRatio);
}

bool PhaseThreeCoordinator::encoderCapacityAvailable(size_t additionalRequests) const noexcept
{
    return phaseVisionEncoderCapacityAvailable(mDownstreamRequestBytes.size(), effectiveEncodedCapacity(),
        mReadyPrefillBytes + mServer.visionPayloadBytes(), mConfig.maxEncodedBytes, mEstimatedEncodedBytes,
        additionalRequests);
}

size_t PhaseThreeCoordinator::effectiveEncodedCapacity() const noexcept
{
    double oldestVisionAgeUs{};
    double visionTtftTargetUs = mConfig.visionTtftTargetUs;
    if (!mPending.empty())
    {
        PendingVisionRequest const& oldest = mPending.front();
        oldestVisionAgeUs = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - oldest.scheduling.submittedAt)
                                .count();
        if (oldest.scheduling.ttftTargetUs > 0.0)
        {
            visionTtftTargetUs = oldest.scheduling.ttftTargetUs;
        }
    }
    return phaseVisionEffectiveEncodedCapacity(mConfig.maxEncodedInFlight, mConfig.throughputMaxEncodedInFlight,
        mServer.throughputMode(), oldestVisionAgeUs, visionTtftTargetUs, mConfig.lookaheadEscalationRatio,
        mServer.decodeTpotPressure(), mConfig.lookaheadDecodeTpotPressureLimit);
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
