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

#include "runtime/scheduling/phaseQueueScheduler.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

PhaseQueueScheduler::PhaseQueueScheduler(PhaseQueueSchedulerConfig config)
    : mConfig(std::move(config))
{
    check::check(mConfig.maxPrefillBatchSize > 0, "maxPrefillBatchSize must be positive");
    check::check(mConfig.maxDecodeBatchSize > 0, "maxDecodeBatchSize must be positive");
    check::check(mConfig.maxOverlapPrefillTokens >= 0, "maxOverlapPrefillTokens must be non-negative");
    check::check(mConfig.maxPrefillChunkTokens >= 0, "maxPrefillChunkTokens must be non-negative");
    check::check(mConfig.minPrefillChunkTokens > 0, "minPrefillChunkTokens must be positive");
    check::check(mConfig.prefillChunkAlignment > 0, "prefillChunkAlignment must be positive");
    check::check(!mConfig.enableAdaptivePrefillChunking || mConfig.maxPrefillChunkTokens > 0,
        "Adaptive prefill chunking requires maxPrefillChunkTokens");
    check::check(
        !mConfig.enableAdaptivePrefillChunking || mConfig.minPrefillChunkTokens <= mConfig.maxPrefillChunkTokens,
        "Adaptive minimum prefill chunk exceeds the maximum");
    check::check(mConfig.decodeBurstLimit > 0, "decodeBurstLimit must be positive");
    check::check(mConfig.prefillQueueWaitTargetUs > 0.0, "prefillQueueWaitTargetUs must be positive");
    check::check(mConfig.decodeQueueWaitTargetUs > 0.0, "decodeQueueWaitTargetUs must be positive");
    check::check(mConfig.maxPredictedOverlapPrefillMs >= 0.0F, "maxPredictedOverlapPrefillMs must be non-negative");
    check::check(mConfig.minObservedOverlapRatio >= 0.0F && mConfig.minObservedOverlapRatio <= 1.0F,
        "minObservedOverlapRatio must be in [0, 1]");
    check::check(
        mConfig.metricsEwmaAlpha > 0.0F && mConfig.metricsEwmaAlpha <= 1.0F, "metricsEwmaAlpha must be in (0, 1]");
    check::check(mConfig.maxPriority > 0, "maxPriority must be positive");
    check::check(std::isfinite(mConfig.priorityPressureWeight) && mConfig.priorityPressureWeight >= 0.0,
        "priorityPressureWeight must be finite and non-negative");
    check::check(std::isfinite(mConfig.priorityAgingUs) && mConfig.priorityAgingUs > 0.0,
        "priorityAgingUs must be finite and positive");
}

namespace
{

void validateSchedulingHints(PhaseSchedulingHints const& hints, int32_t maxPriority)
{
    check::check(
        hints.priority >= 0 && hints.priority <= maxPriority, "Request priority is outside the configured range");
    check::check(std::isfinite(hints.ttftTargetUs) && hints.ttftTargetUs >= 0.0,
        "Request TTFT target must be finite and non-negative");
    check::check(std::isfinite(hints.tpotTargetUs) && hints.tpotTargetUs >= 0.0,
        "Request TPOT target must be finite and non-negative");
}

} // namespace

void PhaseQueueScheduler::enqueuePrefill(PhaseWorkItem item)
{
    check::check(item.tokenCount > 0, "Prefill tokenCount must be positive");
    check::check(item.tokenOffset >= 0, "Prefill tokenOffset must be non-negative");
    validateSchedulingHints(item.scheduling, mConfig.maxPriority);
    check::check(mActiveRequestIds.insert(item.requestId).second, "Request is already active");
    if (item.promptTokenCount == 0)
    {
        item.promptTokenCount = item.tokenOffset + item.tokenCount;
    }
    check::check(
        item.promptTokenCount >= item.tokenOffset + item.tokenCount, "Prefill work exceeds the request prompt length");
    enqueueKnownPrefill(item);
}

void PhaseQueueScheduler::enqueueDecode(PhaseWorkItem item)
{
    check::check(item.tokenCount >= 0, "Decode tokenCount must be non-negative");
    validateSchedulingHints(item.scheduling, mConfig.maxPriority);
    check::check(mActiveRequestIds.insert(item.requestId).second, "Request is already active");
    enqueueKnownDecode(item);
}

bool PhaseQueueScheduler::cancel(uint64_t requestId)
{
    if (mInFlightRequestIds.find(requestId) != mInFlightRequestIds.end())
    {
        return false;
    }

    auto eraseRequest = [requestId](std::deque<PhaseWorkItem>& queue) {
        auto const it = std::find_if(
            queue.begin(), queue.end(), [requestId](PhaseWorkItem const& item) { return item.requestId == requestId; });
        if (it == queue.end())
        {
            return false;
        }
        queue.erase(it);
        return true;
    };

    bool const erased = eraseRequest(mPrefillQueue) || eraseRequest(mDecodeQueue);
    if (erased)
    {
        check::check(mActiveRequestIds.erase(requestId) == 1, "Cancelled request is not active");
        check::check(mQueuedSince.erase(requestId) == 1, "Cancelled request has no queue timestamp");
    }
    return erased;
}
void PhaseQueueScheduler::enqueueKnownPrefill(PhaseWorkItem item)
{
    mPrefillQueue.push_back(item);
    mQueuedSince[item.requestId] = std::chrono::steady_clock::now();
}

void PhaseQueueScheduler::enqueueKnownDecode(PhaseWorkItem item)
{
    mDecodeQueue.push_back(item);
    mQueuedSince[item.requestId] = std::chrono::steady_clock::now();
}

PhaseQueueSnapshot PhaseQueueScheduler::snapshot() const
{
    PhaseQueueSnapshot result{};
    result.prefillQueued = mPrefillQueue.size();
    result.decodeQueued = mDecodeQueue.size();
    result.consecutiveDecodeBatches = mConsecutiveDecodeBatches;
    if (!mPrefillQueue.empty())
    {
        int32_t const bucketTokens = dispatchedPrefillTokens(mPrefillQueue.front());
        bool const bucketInitial = mPrefillQueue.front().tokenOffset == 0;
        int32_t bucketRows{};
        for (PhaseWorkItem const& item : mPrefillQueue)
        {
            if (bucketRows < mConfig.maxPrefillBatchSize && dispatchedPrefillTokens(item) == bucketTokens
                && (item.tokenOffset == 0) == bucketInitial)
            {
                result.prefillCandidateTokens += bucketTokens;
                ++bucketRows;
            }
        }
    }
    int32_t const decodeCount = std::min<int32_t>(mConfig.maxDecodeBatchSize, mDecodeQueue.size());
    for (int32_t i = 0; i < decodeCount; ++i)
    {
        result.decodeCandidateTokens += mDecodeQueue[i].tokenCount;
    }
    auto const now = std::chrono::steady_clock::now();
    auto summarizeQueue = [&](std::deque<PhaseWorkItem> const& queue, bool prefill, double& oldestWaitUs,
                              double& maxSloPressure, int32_t& highestPriority) {
        double waitUs{};
        for (PhaseWorkItem const& item : queue)
        {
            auto const timestamp = mQueuedSince.find(item.requestId);
            check::check(timestamp != mQueuedSince.end(), "Queued request has no residence timestamp");
            double const itemWaitUs = std::chrono::duration<double, std::micro>(now - timestamp->second).count();
            waitUs = std::max(waitUs, itemWaitUs);
            double const requestTarget = prefill ? item.scheduling.ttftTargetUs : item.scheduling.tpotTargetUs;
            double const target = requestTarget > 0.0
                ? requestTarget
                : (prefill ? mConfig.prefillQueueWaitTargetUs : mConfig.decodeQueueWaitTargetUs);
            maxSloPressure = std::max(maxSloPressure, itemWaitUs / target);
            highestPriority = std::max(highestPriority, item.scheduling.priority);
        }
        oldestWaitUs = waitUs;
    };
    summarizeQueue(
        mPrefillQueue, true, result.prefillOldestWaitUs, result.prefillMaxSloPressure, result.prefillHighestPriority);
    summarizeQueue(
        mDecodeQueue, false, result.decodeOldestWaitUs, result.decodeMaxSloPressure, result.decodeHighestPriority);
    return result;
}

PhaseDispatchKind PhaseQueueScheduler::defaultDecision(PhaseQueueSnapshot const& state) const noexcept
{
    if (state.prefillQueued == 0 && state.decodeQueued == 0)
    {
        return PhaseDispatchKind::kNone;
    }
    if (state.prefillQueued == 0)
    {
        return PhaseDispatchKind::kDecode;
    }
    if (state.decodeQueued == 0)
    {
        return PhaseDispatchKind::kPrefill;
    }
    if (state.consecutiveDecodeBatches >= mConfig.decodeBurstLimit)
    {
        return PhaseDispatchKind::kPrefill;
    }
    if (state.prefillCandidateTokens <= mConfig.maxOverlapPrefillTokens)
    {
        return PhaseDispatchKind::kOverlap;
    }
    return PhaseDispatchKind::kDecode;
}

PhaseDispatchKind PhaseQueueScheduler::metricsDecision(
    PhaseQueueSnapshot const& state, PhaseSchedulerTelemetry const& telemetry) const noexcept
{
    if (state.prefillQueued == 0 || state.decodeQueued == 0)
    {
        return defaultDecision(state);
    }

    double const prefillPressure = state.prefillMaxSloPressure;
    double const decodePressure = state.decodeMaxSloPressure;
    if (prefillPressure >= 1.0 || decodePressure >= 1.0)
    {
        auto score = [this](double pressure, int32_t priority) {
            return pressure
                + mConfig.priorityPressureWeight * static_cast<double>(priority)
                / static_cast<double>(mConfig.maxPriority);
        };
        return score(prefillPressure, state.prefillHighestPriority) > score(decodePressure, state.decodeHighestPriority)
            ? PhaseDispatchKind::kPrefill
            : PhaseDispatchKind::kDecode;
    }
    if (state.consecutiveDecodeBatches >= mConfig.decodeBurstLimit)
    {
        return PhaseDispatchKind::kPrefill;
    }
    if (telemetry.sampleCount < mConfig.minMetricsSamples || telemetry.prefillGpuMsPerToken <= 0.0F)
    {
        return defaultDecision(state);
    }

    float const predictedPrefillMs = telemetry.prefillGpuMsPerToken * static_cast<float>(state.prefillCandidateTokens);
    bool const overlapEfficient
        = telemetry.overlapSampleCount == 0 || telemetry.overlapRatio >= mConfig.minObservedOverlapRatio;
    if (predictedPrefillMs <= mConfig.maxPredictedOverlapPrefillMs && overlapEfficient)
    {
        return PhaseDispatchKind::kOverlap;
    }
    return PhaseDispatchKind::kDecode;
}

int32_t PhaseQueueScheduler::dispatchedPrefillTokens(PhaseWorkItem const& item) const noexcept
{
    if (!item.allowChunkedPrefill || !mConfig.supportsChunkedPrefill || mConfig.maxPrefillChunkTokens == 0)
    {
        return item.tokenCount;
    }

    int32_t const maximum = std::min(item.tokenCount, mConfig.maxPrefillChunkTokens);
    if (!mConfig.enableAdaptivePrefillChunking || mDecodeQueue.empty())
    {
        return maximum;
    }

    int32_t const minimum = std::min(maximum, mConfig.minPrefillChunkTokens);
    int32_t selected = maximum;
    if (mTelemetry.prefillGpuMsPerToken > 0.0F)
    {
        float const budgetTokens = mConfig.maxPredictedOverlapPrefillMs / mTelemetry.prefillGpuMsPerToken;
        if (budgetTokens <= static_cast<float>(minimum))
        {
            selected = minimum;
        }
        else if (budgetTokens < static_cast<float>(maximum))
        {
            selected = static_cast<int32_t>(budgetTokens);
        }
    }

    if (mTelemetry.overlapSampleCount > 0 && mTelemetry.overlapRatio < mConfig.minObservedOverlapRatio)
    {
        selected = minimum;
    }

    selected = std::clamp(selected, minimum, maximum);
    int32_t const aligned = selected / mConfig.prefillChunkAlignment * mConfig.prefillChunkAlignment;
    return std::max(minimum, aligned);
}

std::vector<PhaseWorkItem> PhaseQueueScheduler::popBatch(
    std::deque<PhaseWorkItem>& queue, int32_t maxBatchSize, bool chunkPrefill, double& queueWaitUs)
{
    auto const now = std::chrono::steady_clock::now();
    auto priorityRank = [&](PhaseWorkItem const& item) {
        auto const timestamp = mQueuedSince.find(item.requestId);
        check::check(timestamp != mQueuedSince.end(), "Prioritized request has no queue timestamp");
        double const waitUs = std::chrono::duration<double, std::micro>(now - timestamp->second).count();
        return static_cast<double>(item.scheduling.priority) + waitUs / mConfig.priorityAgingUs;
    };
    auto higherPriority = [&](PhaseWorkItem const& lhs, PhaseWorkItem const& rhs) {
        double const lhsRank = priorityRank(lhs);
        double const rhsRank = priorityRank(rhs);
        if (lhsRank != rhsRank)
        {
            return lhsRank < rhsRank;
        }
        return mQueuedSince.at(lhs.requestId) > mQueuedSince.at(rhs.requestId);
    };
    auto recordQueueWait = [&](uint64_t requestId) {
        auto const timestamp = mQueuedSince.find(requestId);
        check::check(timestamp != mQueuedSince.end(), "Dispatched request has no queue timestamp");
        queueWaitUs = std::max(queueWaitUs, std::chrono::duration<double, std::micro>(now - timestamp->second).count());
        mQueuedSince.erase(timestamp);
    };
    int32_t const count = std::min<int32_t>(maxBatchSize, queue.size());
    std::vector<PhaseWorkItem> batch;
    batch.reserve(count);
    if (!chunkPrefill)
    {
        for (int32_t i = 0; i < count; ++i)
        {
            auto selected = queue.begin();
            if (mConfig.enablePriorityBatching)
            {
                selected = std::max_element(queue.begin(), queue.end(), higherPriority);
            }
            PhaseWorkItem item = *selected;
            queue.erase(selected);
            check::check(mInFlightRequestIds.insert(item.requestId).second, "Request is already in flight");
            recordQueueWait(item.requestId);
            batch.push_back(item);
        }
        return batch;
    }

    auto bucketSeed = queue.begin();
    if (mConfig.enablePriorityBatching)
    {
        bucketSeed = std::max_element(queue.begin(), queue.end(), higherPriority);
    }
    int32_t const bucketTokens = dispatchedPrefillTokens(*bucketSeed);
    bool const bucketInitial = bucketSeed->tokenOffset == 0;
    while (static_cast<int32_t>(batch.size()) < maxBatchSize)
    {
        auto selected = queue.end();
        for (auto it = queue.begin(); it != queue.end(); ++it)
        {
            if (dispatchedPrefillTokens(*it) == bucketTokens && (it->tokenOffset == 0) == bucketInitial
                && (selected == queue.end() || (mConfig.enablePriorityBatching && higherPriority(*selected, *it))))
            {
                selected = it;
            }
        }
        if (selected == queue.end())
        {
            break;
        }
        PhaseWorkItem item = *selected;
        queue.erase(selected);
        item.tokenCount = bucketTokens;
        check::check(mInFlightRequestIds.insert(item.requestId).second, "Request is already in flight");
        recordQueueWait(item.requestId);
        batch.push_back(item);
    }
    return batch;
}

PhaseDispatchPlan PhaseQueueScheduler::next()
{
    PhaseQueueSnapshot const state = snapshot();
    PhaseDispatchKind const kind = mConfig.metricsPolicy
        ? mConfig.metricsPolicy(state, mTelemetry)
        : (mConfig.enableMetricsPolicy ? metricsDecision(state, mTelemetry)
                                       : (mConfig.policy ? mConfig.policy(state) : defaultDecision(state)));
    check::check(kind != PhaseDispatchKind::kPrefill || state.prefillQueued > 0,
        "Scheduling policy selected an empty prefill queue");
    check::check(kind != PhaseDispatchKind::kDecode || state.decodeQueued > 0,
        "Scheduling policy selected an empty decode queue");
    check::check(kind != PhaseDispatchKind::kOverlap || (state.prefillQueued > 0 && state.decodeQueued > 0),
        "Scheduling policy selected overlap without work in both queues");

    PhaseDispatchPlan plan;
    plan.kind = kind;
    if (kind == PhaseDispatchKind::kPrefill || kind == PhaseDispatchKind::kOverlap)
    {
        plan.prefillBatch = popBatch(mPrefillQueue, mConfig.maxPrefillBatchSize, true, plan.prefillQueueWaitUs);
    }
    if (kind == PhaseDispatchKind::kDecode || kind == PhaseDispatchKind::kOverlap)
    {
        plan.decodeBatch = popBatch(mDecodeQueue, mConfig.maxDecodeBatchSize, false, plan.decodeQueueWaitUs);
    }
    if (kind == PhaseDispatchKind::kDecode)
    {
        ++mConsecutiveDecodeBatches;
    }
    else if (kind == PhaseDispatchKind::kPrefill || kind == PhaseDispatchKind::kOverlap)
    {
        mConsecutiveDecodeBatches = 0;
    }
    return plan;
}

void PhaseQueueScheduler::completePrefill(PhaseWorkItem item, int32_t resultingKVLength, bool finished)
{
    check::check(
        mInFlightRequestIds.erase(item.requestId) == 1, "Completed prefill request is not currently in flight");
    check::check(item.tokenCount > 0, "Completed prefill chunk must contain tokens");
    check::check(item.promptTokenCount > 0, "Completed prefill request must have a prompt length");
    int32_t const nextOffset = item.tokenOffset + item.tokenCount;
    check::check(nextOffset <= item.promptTokenCount, "Completed prefill chunk exceeds the prompt length");
    check::check(resultingKVLength >= nextOffset, "Resulting KV length is behind completed prompt progress");

    if (finished)
    {
        check::check(nextOffset == item.promptTokenCount, "A request cannot finish before its final prefill chunk");
        check::check(mActiveRequestIds.erase(item.requestId) == 1, "Finished prefill request is not active");
        return;
    }

    item.tokenOffset = nextOffset;
    if (nextOffset < item.promptTokenCount)
    {
        item.tokenCount = item.promptTokenCount - nextOffset;
        enqueueKnownPrefill(item);
        return;
    }

    item.tokenCount = resultingKVLength;
    enqueueKnownDecode(item);
}

void PhaseQueueScheduler::completeDecode(PhaseWorkItem item, int32_t resultingKVLength, bool finished)
{
    check::check(mInFlightRequestIds.erase(item.requestId) == 1, "Completed decode request is not currently in flight");
    check::check(resultingKVLength >= item.tokenCount, "Resulting KV length cannot move backwards");
    if (finished)
    {
        check::check(mActiveRequestIds.erase(item.requestId) == 1, "Finished decode request is not active");
        return;
    }
    item.tokenCount = resultingKVLength;
    enqueueKnownDecode(item);
}

size_t PhaseQueueScheduler::prefillQueueSize() const noexcept
{
    return mPrefillQueue.size();
}

size_t PhaseQueueScheduler::decodeQueueSize() const noexcept
{
    return mDecodeQueue.size();
}

bool PhaseQueueScheduler::empty() const noexcept
{
    return mPrefillQueue.empty() && mDecodeQueue.empty();
}

bool PhaseQueueScheduler::hasRequest(uint64_t requestId) const noexcept
{
    return mActiveRequestIds.find(requestId) != mActiveRequestIds.end();
}

void PhaseQueueScheduler::observeMetrics(PhaseDispatchMetrics const& metrics)
{
    auto updateEwma = [alpha = mConfig.metricsEwmaAlpha](float& average, float sample) {
        average = average > 0.0F ? alpha * sample + (1.0F - alpha) * average : sample;
    };
    if (metrics.prefillTokens > 0 && metrics.prefillGpuMs > 0.0F)
    {
        updateEwma(mTelemetry.prefillGpuMsPerToken, metrics.prefillGpuMs / static_cast<float>(metrics.prefillTokens));
    }
    if (metrics.decodeContextTokens > 0 && metrics.decodeGpuMs > 0.0F)
    {
        updateEwma(mTelemetry.decodeGpuMsPerContextToken,
            metrics.decodeGpuMs / static_cast<float>(metrics.decodeContextTokens));
    }
    if (metrics.kind == PhaseDispatchKind::kOverlap && metrics.prefillBatchSize > 0 && metrics.decodeBatchSize > 0)
    {
        updateEwma(mTelemetry.overlapRatio, metrics.overlapRatio);
        ++mTelemetry.overlapSampleCount;
    }
    ++mTelemetry.sampleCount;
    mTelemetry.lastDispatch = metrics;
}

PhaseSchedulerTelemetry const& PhaseQueueScheduler::telemetry() const noexcept
{
    return mTelemetry;
}

} // namespace rt
} // namespace trt_edgellm
