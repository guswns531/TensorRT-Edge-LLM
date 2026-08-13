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
#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{

void applySchedulerProfile(PhaseQueueSchedulerConfig& config)
{
    if (config.profile == PhaseSchedulerProfile::kCustom)
    {
        return;
    }
    if (config.profile == PhaseSchedulerProfile::kThroughputBalanced)
    {
        config.enableTpotHardGuard = true;
        config.requireDirectOverlapCost = true;
        config.enableCostAwareOverlapAdmission = true;
        config.enableTpotHysteresis = true;
        return;
    }

    config.enableDynamicDecodeBatching = true;
    config.enableDynamicPrefillBatching = true;
    config.enableTpotHardGuard = true;
    config.requireDirectOverlapCost
        = config.requireDirectOverlapCost || config.profile == PhaseSchedulerProfile::kLatencySafe;
    config.enablePrefillSloRecovery = config.profile == PhaseSchedulerProfile::kLongPrefill;
    config.enableWavefrontPrefillBatching = config.profile == PhaseSchedulerProfile::kLongPrefill;
    config.decodeRecoveryPressureThreshold = config.profile == PhaseSchedulerProfile::kLatencySafe ? 0.9F : 0.8F;
    if (config.profile == PhaseSchedulerProfile::kLatencySafe)
    {
        config.maxConsecutiveOverlapBatches = 1;
        config.maxPredictedDecodeDebtUs = 10000.0;
    }
    else if (config.profile == PhaseSchedulerProfile::kBalanced || config.profile == PhaseSchedulerProfile::kAuto)
    {
        config.maxConsecutiveOverlapBatches = 2;
        config.maxPredictedDecodeDebtUs = 25000.0;
    }
    if (config.profile == PhaseSchedulerProfile::kBalanced || config.profile == PhaseSchedulerProfile::kAuto)
    {
        config.minDynamicPrefillBatchSize = std::min(2, config.maxPrefillBatchSize);
    }
    else if (config.profile == PhaseSchedulerProfile::kLatencySafe)
    {
        config.minDynamicPrefillBatchSize = 1;
    }
}

} // namespace

PhaseQueueScheduler::PhaseQueueScheduler(PhaseQueueSchedulerConfig config)
    : mConfig(std::move(config))
{
    applySchedulerProfile(mConfig);
    check::check(mConfig.maxPrefillBatchSize > 0, "maxPrefillBatchSize must be positive");
    check::check(mConfig.maxDecodeBatchSize > 0, "maxDecodeBatchSize must be positive");
    check::check(mConfig.maxOverlapPrefillTokens >= 0, "maxOverlapPrefillTokens must be non-negative");
    check::check(mConfig.maxPrefillChunkTokens >= 0, "maxPrefillChunkTokens must be non-negative");
    check::check(mConfig.maxPrefillBatchTokens >= 0, "maxPrefillBatchTokens must be non-negative");
    check::check(mConfig.prefillCompletionBonusTokens >= 0, "prefillCompletionBonusTokens must be non-negative");
    for (PhaseDecodeBatchCost const& cost : mConfig.decodeBatchCosts)
    {
        check::check(cost.batchSize > 0, "Decode cost batch size must be positive");
        check::check(cost.maxContextLength > 0, "Decode cost context length must be positive");
        check::check(std::isfinite(cost.p95GpuMs) && cost.p95GpuMs > 0.0F,
            "Decode cost p95 GPU time must be finite and positive");
    }
    for (PhasePrefillBatchCost const& cost : mConfig.prefillBatchCosts)
    {
        check::check(cost.batchSize > 0, "Prefill cost batch size must be positive");
        check::check(cost.chunkLength > 0 && cost.maxPastKVLength >= 0 && cost.maxConcurrentDecodeBatchSize >= 0,
            "Prefill cost shape bounds are invalid");
        check::check(std::isfinite(cost.p95GpuMs) && cost.p95GpuMs > 0.0F && std::isfinite(cost.decodeSlowdownP95Ms)
                && cost.decodeSlowdownP95Ms >= 0.0F,
            "Prefill cost timings must be finite and non-negative");
    }
    for (PhaseOverlapBatchCost const& cost : mConfig.overlapBatchCosts)
    {
        check::check(cost.prefillBatchSize > 0 && cost.decodeBatchSize > 0 && cost.chunkLength > 0
                && cost.maxPrefillPastKVLength >= 0 && cost.maxDecodeContextLength > 0,
            "Overlap cost shape bounds are invalid");
        check::check(std::isfinite(cost.prefillP95GpuMs) && cost.prefillP95GpuMs > 0.0F
                && std::isfinite(cost.decodeP95GpuMs) && cost.decodeP95GpuMs > 0.0F
                && std::isfinite(cost.makespanP95GpuMs) && cost.makespanP95GpuMs > 0.0F
                && std::isfinite(cost.decodeSlowdownP95Ms) && cost.decodeSlowdownP95Ms >= 0.0F,
            "Overlap cost timings must be finite and non-negative");
    }
    check::check(!mConfig.enableDynamicPrefillBatching || !mConfig.prefillBatchCosts.empty(),
        "Dynamic prefill batching requires profiled prefill costs");
    check::check(std::isfinite(mConfig.decodeRecoveryPressureThreshold)
            && mConfig.decodeRecoveryPressureThreshold > 0.0F && mConfig.decodeRecoveryPressureThreshold <= 1.0F,
        "Decode recovery pressure threshold must be in (0, 1]");
    check::check(!mConfig.requireDirectOverlapCost || !mConfig.overlapBatchCosts.empty(),
        "Direct overlap cost enforcement requires profiled overlap costs");
    check::check(
        !mConfig.enableCostAwareOverlapAdmission || (mConfig.enableTpotHardGuard && mConfig.requireDirectOverlapCost),
        "Cost-aware overlap admission requires direct costs and the TPOT hard guard");
    check::check(!mConfig.enableTpotHysteresis || mConfig.enableCostAwareOverlapAdmission,
        "TPOT hysteresis requires cost-aware overlap admission");
    check::check(std::isfinite(mConfig.tpotHysteresisEnterRatio) && std::isfinite(mConfig.tpotHysteresisExitRatio)
            && mConfig.tpotHysteresisExitRatio > 0.0F
            && mConfig.tpotHysteresisExitRatio < mConfig.tpotHysteresisEnterRatio
            && mConfig.tpotHysteresisEnterRatio <= 1.0F,
        "TPOT hysteresis ratios must satisfy 0 < exit < enter <= 1");
    check::check(mConfig.tpotHysteresisWindow > 0 && mConfig.minTpotHysteresisSamples > 0
            && mConfig.minTpotHysteresisSamples <= mConfig.tpotHysteresisWindow,
        "TPOT hysteresis sample bounds are invalid");
    check::check(mConfig.maxConsecutiveOverlapBatches > 0, "Maximum consecutive overlap batches must be positive");
    check::check(std::isfinite(mConfig.maxPredictedDecodeDebtUs) && mConfig.maxPredictedDecodeDebtUs >= 0.0,
        "Maximum predicted decode debt must be finite and non-negative");
    check::check(mConfig.autoLongPrefillBacklogTokens > 0, "Auto-profile prefill backlog threshold must be positive");
    check::check(std::isfinite(mConfig.autoDecodePressureLimit) && mConfig.autoDecodePressureLimit >= 0.0F
            && mConfig.autoDecodePressureLimit <= 1.0F,
        "Auto-profile decode pressure limit must be in [0, 1]");
    check::check(mConfig.maxPrefillCohortSize > 0, "Prefill cohort size must be positive");
    check::check(mConfig.maxPrefillCohortTurns > 0, "Prefill cohort turn limit must be positive");
    check::check(std::isfinite(mConfig.decodeSlackSafetyFactor) && mConfig.decodeSlackSafetyFactor > 0.0F
            && mConfig.decodeSlackSafetyFactor <= 1.0F,
        "Decode slack safety factor must be in (0, 1]");
    check::check(
        mConfig.minDynamicPrefillBatchSize > 0 && mConfig.minDynamicPrefillBatchSize <= mConfig.maxPrefillBatchSize,
        "Minimum dynamic prefill batch size must be positive and within the prefill batch limit");
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
    check::check(mConfig.pagePressureDecodeThreshold >= 0.0F && mConfig.pagePressureDecodeThreshold <= 1.0F,
        "pagePressureDecodeThreshold must be in [0, 1]");
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
        mPrefillCohortIds.erase(requestId);
    }
    return erased;
}
void PhaseQueueScheduler::enqueueKnownPrefill(PhaseWorkItem item)
{
    if (item.scheduling.submittedAt == std::chrono::steady_clock::time_point{})
    {
        item.scheduling.submittedAt = std::chrono::steady_clock::now();
    }
    mPrefillQueue.push_back(item);
    mQueuedSince[item.requestId] = std::chrono::steady_clock::now();
}

void PhaseQueueScheduler::enqueueKnownDecode(PhaseWorkItem item)
{
    if (item.scheduling.submittedAt == std::chrono::steady_clock::time_point{})
    {
        item.scheduling.submittedAt = std::chrono::steady_clock::now();
    }
    mDecodeQueue.push_back(item);
    mQueuedSince[item.requestId] = std::chrono::steady_clock::now();
}

PhaseQueueSnapshot PhaseQueueScheduler::snapshot() const
{
    PhaseQueueSnapshot result{};
    result.consecutiveDecodeBatches = mConsecutiveDecodeBatches;
    auto const prefillSeed = std::find_if(mPrefillQueue.begin(), mPrefillQueue.end(),
        [this](PhaseWorkItem const& item) { return isEligible(item, true); });
    if (prefillSeed != mPrefillQueue.end())
    {
        int32_t const bucketTokens = dispatchedPrefillTokens(*prefillSeed);
        bool const bucketInitial = prefillSeed->tokenOffset == 0;
        bool const allowRaggedBatch = prefillSeed->allowChunkedPrefill;
        int32_t bucketRows{};
        for (PhaseWorkItem const& item : mPrefillQueue)
        {
            if (isEligible(item, true) && bucketRows < mConfig.maxPrefillBatchSize
                && isPrefillBatchCompatible(item, bucketTokens, bucketInitial, allowRaggedBatch))
            {
                ++bucketRows;
            }
        }
        result.prefillCandidateTokens = bucketRows * bucketTokens;
    }
    result.prefillQueued = static_cast<size_t>(std::count_if(mPrefillQueue.begin(), mPrefillQueue.end(),
        [this](PhaseWorkItem const& item) { return isEligible(item, true); }));
    for (PhaseWorkItem const& item : mPrefillQueue)
    {
        if (isEligible(item, true))
        {
            result.prefillRemainingTokens += item.tokenCount;
            result.prefillContinuationRows += item.tokenOffset > 0 ? 1 : 0;
        }
    }
    for (PhaseWorkItem const& item : mDecodeQueue)
    {
        if (isEligible(item, false))
        {
            ++result.decodeQueued;
            if (result.decodeQueued <= static_cast<size_t>(mConfig.maxDecodeBatchSize))
            {
                result.decodeCandidateTokens += item.tokenCount;
            }
        }
    }
    auto const now = std::chrono::steady_clock::now();
    result.prefillMinTtftSlackUs = std::numeric_limits<double>::max();
    auto summarizeQueue = [&](std::deque<PhaseWorkItem> const& queue, bool prefill, double& oldestWaitUs,
                              double& maxSloPressure, int32_t& highestPriority) {
        double waitUs{};
        for (PhaseWorkItem const& item : queue)
        {
            if (!isEligible(item, prefill))
            {
                continue;
            }
            auto const timestamp = mQueuedSince.find(item.requestId);
            check::check(timestamp != mQueuedSince.end(), "Queued request has no residence timestamp");
            double const itemWaitUs = std::chrono::duration<double, std::micro>(now - timestamp->second).count();
            waitUs = std::max(waitUs, itemWaitUs);
            double const requestTarget = prefill ? item.scheduling.ttftTargetUs : item.scheduling.tpotTargetUs;
            double const target = requestTarget > 0.0
                ? requestTarget
                : (prefill ? mConfig.prefillQueueWaitTargetUs : mConfig.decodeQueueWaitTargetUs);
            double const requestAgeUs
                = std::chrono::duration<double, std::micro>(now - item.scheduling.submittedAt).count();
            double const sloAgeUs = prefill ? requestAgeUs : itemWaitUs;
            maxSloPressure = std::max(maxSloPressure, sloAgeUs / target);
            if (prefill)
            {
                result.prefillOldestRequestAgeUs = std::max(result.prefillOldestRequestAgeUs, requestAgeUs);
                result.prefillMinTtftSlackUs = std::min(result.prefillMinTtftSlackUs, target - requestAgeUs);
            }
            highestPriority = std::max(highestPriority, item.scheduling.priority);
        }
        oldestWaitUs = waitUs;
    };
    summarizeQueue(
        mPrefillQueue, true, result.prefillOldestWaitUs, result.prefillMaxSloPressure, result.prefillHighestPriority);
    summarizeQueue(
        mDecodeQueue, false, result.decodeOldestWaitUs, result.decodeMaxSloPressure, result.decodeHighestPriority);
    if (result.prefillQueued == 0)
    {
        result.prefillMinTtftSlackUs = 0.0;
    }
    return result;
}

bool PhaseQueueScheduler::isEligible(PhaseWorkItem const& item, bool prefill) const
{
    return !mConfig.eligibilityPolicy || mConfig.eligibilityPolicy(item, prefill);
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
    if (mConfig.enableCostAwareOverlapAdmission && !mLatencySafeFallback)
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
    if (mConfig.pagePressureDecodeThreshold > 0.0F && telemetry.lastDispatch.has_value()
        && telemetry.lastDispatch->pagePoolTotalBundles > 0)
    {
        float const pagePressure = static_cast<float>(telemetry.lastDispatch->pagePoolAllocatedBundles)
            / static_cast<float>(telemetry.lastDispatch->pagePoolTotalBundles);
        if (pagePressure >= mConfig.pagePressureDecodeThreshold)
        {
            return PhaseDispatchKind::kDecode;
        }
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

bool PhaseQueueScheduler::isPrefillBatchCompatible(
    PhaseWorkItem const& item, int32_t paddedChunkLength, bool initialChunk, bool allowRaggedBatch) const noexcept
{
    int32_t const itemTokens = dispatchedPrefillTokens(item);
    if ((item.tokenOffset == 0) != initialChunk || itemTokens > paddedChunkLength)
    {
        return false;
    }
    if (itemTokens == paddedChunkLength)
    {
        return true;
    }
    return mConfig.enableRaggedPrefillBatching && allowRaggedBatch && item.allowChunkedPrefill;
}

int32_t PhaseQueueScheduler::selectPrefillBatchSize(std::vector<PhaseWorkItem const*> const& candidates,
    int32_t chunkLength, bool initialChunk, bool overlap, int32_t plannedDecodeBatchSize,
    int32_t plannedDecodeMaxContextLength, PhaseQueueSnapshot const& state, float& predictedGpuMs,
    float& predictedDecodeSlowdownMs, bool& costCoverageMiss) const noexcept
{
    int32_t const available = std::min<int32_t>(mConfig.maxPrefillBatchSize, candidates.size());
    bool const evaluateOversizedOverlap = overlap && mConfig.enableCostAwareOverlapAdmission && !mLatencySafeFallback
        && state.prefillCandidateTokens > mConfig.maxOverlapPrefillTokens;
    if (available <= 0 || (!mConfig.enableDynamicPrefillBatching && !evaluateOversizedOverlap))
    {
        return 0;
    }

    struct Candidate
    {
        int32_t batchSize{};
        int32_t usefulTokens{};
        float gpuMs{};
        float decodeInterferenceMs{};
    };
    std::vector<Candidate> profiled;
    int32_t const firstBatchSize = std::min(available, mConfig.minDynamicPrefillBatchSize);
    for (int32_t batchSize = firstBatchSize; batchSize <= available; ++batchSize)
    {
        int32_t maxPastKV{};
        for (int32_t index = 0; index < batchSize; ++index)
        {
            maxPastKV = std::max(maxPastKV, candidates[static_cast<size_t>(index)]->tokenOffset);
        }
        PhasePrefillBatchCost const* selected{};
        for (PhasePrefillBatchCost const& cost : mConfig.prefillBatchCosts)
        {
            if (cost.batchSize != batchSize || cost.chunkLength < chunkLength || cost.initialChunk != initialChunk
                || cost.maxPastKVLength < maxPastKV
                || cost.maxConcurrentDecodeBatchSize
                    < std::min<int32_t>(mConfig.maxDecodeBatchSize, static_cast<int32_t>(state.decodeQueued)))
            {
                continue;
            }
            if (selected == nullptr || cost.chunkLength < selected->chunkLength
                || (cost.chunkLength == selected->chunkLength && cost.maxPastKVLength < selected->maxPastKVLength)
                || (cost.chunkLength == selected->chunkLength && cost.maxPastKVLength == selected->maxPastKVLength
                    && cost.maxConcurrentDecodeBatchSize < selected->maxConcurrentDecodeBatchSize))
            {
                selected = &cost;
            }
        }
        PhaseOverlapBatchCost const* selectedOverlap{};
        if (overlap)
        {
            for (PhaseOverlapBatchCost const& cost : mConfig.overlapBatchCosts)
            {
                if (cost.prefillBatchSize != batchSize || cost.chunkLength < chunkLength
                    || cost.initialChunk != initialChunk || cost.decodeBatchSize < plannedDecodeBatchSize
                    || cost.maxPrefillPastKVLength < maxPastKV
                    || cost.maxDecodeContextLength < plannedDecodeMaxContextLength)
                {
                    continue;
                }
                if (selectedOverlap == nullptr || cost.chunkLength < selectedOverlap->chunkLength
                    || (cost.chunkLength == selectedOverlap->chunkLength
                        && cost.decodeBatchSize < selectedOverlap->decodeBatchSize)
                    || (cost.chunkLength == selectedOverlap->chunkLength
                        && cost.decodeBatchSize == selectedOverlap->decodeBatchSize
                        && cost.maxPrefillPastKVLength < selectedOverlap->maxPrefillPastKVLength)
                    || (cost.chunkLength == selectedOverlap->chunkLength
                        && cost.decodeBatchSize == selectedOverlap->decodeBatchSize
                        && cost.maxPrefillPastKVLength == selectedOverlap->maxPrefillPastKVLength
                        && cost.maxDecodeContextLength < selectedOverlap->maxDecodeContextLength))
                {
                    selectedOverlap = &cost;
                }
            }
        }
        if (selected != nullptr && (!overlap || selectedOverlap != nullptr || !mConfig.requireDirectOverlapCost))
        {
            int32_t usefulTokens{};
            for (int32_t index{}; index < batchSize; ++index)
            {
                usefulTokens += dispatchedPrefillTokens(*candidates[static_cast<size_t>(index)]);
            }
            float const gpuMs = selectedOverlap != nullptr ? selectedOverlap->prefillP95GpuMs : selected->p95GpuMs;
            float const interference = selectedOverlap != nullptr
                ? selectedOverlap->decodeSlowdownP95Ms
                : (overlap ? selected->decodeSlowdownP95Ms : selected->p95GpuMs);
            profiled.push_back({batchSize, usefulTokens, gpuMs, interference});
        }
    }
    if (profiled.empty())
    {
        costCoverageMiss = true;
        return overlap && mConfig.enableTpotHardGuard && mConfig.requireDirectOverlapCost ? -1 : 0;
    }

    double const remainingDecodeUs
        = std::max(0.0, mConfig.decodeQueueWaitTargetUs * (1.0 - state.decodeMaxSloPressure));
    double const allowedInterferenceUs = remainingDecodeUs * mConfig.decodeSlackSafetyFactor;
    bool const automaticLongPrefillRecovery = mConfig.profile == PhaseSchedulerProfile::kAuto
        && state.prefillRemainingTokens >= mConfig.autoLongPrefillBacklogTokens
        && state.decodeMaxSloPressure < mConfig.autoDecodePressureLimit;
    bool const prefillRecovery = (mConfig.enablePrefillSloRecovery || automaticLongPrefillRecovery)
        && state.prefillMaxSloPressure >= 1.0 && state.decodeMaxSloPressure < 1.0;
    if (overlap && mConfig.enableTpotHardGuard && mConsecutiveOverlapBatches >= mConfig.maxConsecutiveOverlapBatches)
    {
        return -1;
    }
    Candidate const* selected{};
    for (Candidate const& candidate : profiled)
    {
        double const candidateInterferenceUs = static_cast<double>(candidate.decodeInterferenceMs) * 1000.0;
        bool const debtFeasible = mConfig.maxPredictedDecodeDebtUs == 0.0
            || mPredictedDecodeDebtUs + candidateInterferenceUs <= mConfig.maxPredictedDecodeDebtUs;
        bool const feasible
            = (prefillRecovery || state.decodeQueued == 0 || candidateInterferenceUs <= allowedInterferenceUs)
            && (!overlap || !mConfig.enableTpotHardGuard || debtFeasible);
        if (!feasible)
        {
            continue;
        }
        double const efficiency = static_cast<double>(candidate.usefulTokens) / candidate.gpuMs;
        double const selectedEfficiency
            = selected == nullptr ? 0.0 : static_cast<double>(selected->usefulTokens) / selected->gpuMs;
        if (selected == nullptr || efficiency > selectedEfficiency
            || (efficiency == selectedEfficiency && candidate.batchSize > selected->batchSize))
        {
            selected = &candidate;
        }
    }
    if (selected == nullptr)
    {
        if (overlap && mConfig.enableTpotHardGuard)
        {
            return -1;
        }
        selected = &*std::min_element(profiled.begin(), profiled.end(), [](Candidate const& lhs, Candidate const& rhs) {
            return lhs.decodeInterferenceMs < rhs.decodeInterferenceMs
                || (lhs.decodeInterferenceMs == rhs.decodeInterferenceMs && lhs.gpuMs < rhs.gpuMs);
        });
    }
    predictedGpuMs = selected->gpuMs;
    predictedDecodeSlowdownMs = selected->decodeInterferenceMs;
    return selected->batchSize;
}

std::vector<PhaseWorkItem> PhaseQueueScheduler::popBatch(std::deque<PhaseWorkItem>& queue, int32_t maxBatchSize,
    bool chunkPrefill, PhaseQueueSnapshot const& state, PhaseDispatchPlan& plan)
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
    auto ttftSlack = [&](PhaseWorkItem const& item) {
        double const target
            = item.scheduling.ttftTargetUs > 0.0 ? item.scheduling.ttftTargetUs : mConfig.prefillQueueWaitTargetUs;
        double const ageUs = std::chrono::duration<double, std::micro>(now - item.scheduling.submittedAt).count();
        return target - ageUs;
    };
    auto moreUrgentPrefill = [&](PhaseWorkItem const& lhs, PhaseWorkItem const& rhs) {
        if (mConfig.enablePriorityBatching && priorityRank(lhs) != priorityRank(rhs))
        {
            return priorityRank(lhs) > priorityRank(rhs);
        }
        double const lhsSlack = ttftSlack(lhs);
        double const rhsSlack = ttftSlack(rhs);
        if (lhsSlack != rhsSlack)
        {
            return lhsSlack < rhsSlack;
        }
        return mQueuedSince.at(lhs.requestId) < mQueuedSince.at(rhs.requestId);
    };
    auto recordQueueWait = [&](uint64_t requestId) {
        auto const timestamp = mQueuedSince.find(requestId);
        check::check(timestamp != mQueuedSince.end(), "Dispatched request has no queue timestamp");
        double& queueWaitUs = chunkPrefill ? plan.prefillQueueWaitUs : plan.decodeQueueWaitUs;
        queueWaitUs = std::max(queueWaitUs, std::chrono::duration<double, std::micro>(now - timestamp->second).count());
        mQueuedSince.erase(timestamp);
    };
    auto orderPrefillRow = [&](PhaseWorkItem const* lhs, PhaseWorkItem const* rhs, int32_t paddedChunkLength) {
        int32_t const lhsTokens = dispatchedPrefillTokens(*lhs);
        int32_t const rhsTokens = dispatchedPrefillTokens(*rhs);
        bool const lhsExact = lhsTokens == paddedChunkLength;
        bool const rhsExact = rhsTokens == paddedChunkLength;
        if (lhsExact != rhsExact)
        {
            return lhsExact;
        }
        if (!lhsExact && lhsTokens != rhsTokens)
        {
            return lhsTokens > rhsTokens;
        }
        return moreUrgentPrefill(*lhs, *rhs);
    };
    int32_t const count = std::min<int32_t>(maxBatchSize,
        std::count_if(queue.begin(), queue.end(),
            [this, chunkPrefill](PhaseWorkItem const& item) { return isEligible(item, chunkPrefill); }));
    std::vector<PhaseWorkItem> batch;
    batch.reserve(count);
    if (!chunkPrefill)
    {
        for (int32_t i = 0; i < count; ++i)
        {
            auto selected = std::find_if(
                queue.begin(), queue.end(), [this](PhaseWorkItem const& item) { return isEligible(item, false); });
            check::check(selected != queue.end(), "Eligible decode work disappeared during batch selection");
            if (mConfig.enablePriorityBatching)
            {
                for (auto it = std::next(selected); it != queue.end(); ++it)
                {
                    if (isEligible(*it, false) && higherPriority(*selected, *it))
                    {
                        selected = it;
                    }
                }
            }
            PhaseWorkItem item = *selected;
            queue.erase(selected);
            check::check(mInFlightRequestIds.insert(item.requestId).second, "Request is already in flight");
            recordQueueWait(item.requestId);
            batch.push_back(item);
        }
        return batch;
    }

    if (mConfig.enableWavefrontPrefillBatching && mPrefillCohortTurns >= mConfig.maxPrefillCohortTurns)
    {
        mPrefillCohortIds.clear();
        mPrefillCohortTurns = 0;
    }
    auto inActiveCohort = [&](PhaseWorkItem const& item) {
        return !mConfig.enableWavefrontPrefillBatching || mPrefillCohortIds.empty()
            || mPrefillCohortIds.find(item.requestId) != mPrefillCohortIds.end();
    };
    auto bucketSeed = std::find_if(queue.cbegin(), queue.cend(),
        [&](PhaseWorkItem const& item) { return isEligible(item, true) && inActiveCohort(item); });
    if (bucketSeed == queue.cend() && !mPrefillCohortIds.empty())
    {
        mPrefillCohortIds.clear();
        mPrefillCohortTurns = 0;
        bucketSeed = std::find_if(
            queue.cbegin(), queue.cend(), [this](PhaseWorkItem const& item) { return isEligible(item, true); });
    }
    check::check(bucketSeed != queue.cend(), "Eligible prefill work disappeared during batch selection");
    if (mConfig.enablePriorityBatching || mConfig.enableWavefrontPrefillBatching)
    {
        for (auto it = std::next(bucketSeed); it != queue.cend(); ++it)
        {
            if (isEligible(*it, true) && inActiveCohort(*it) && moreUrgentPrefill(*it, *bucketSeed))
            {
                bucketSeed = it;
            }
        }
    }
    if ((mConfig.maxPrefillBatchTokens > 0 || mConfig.enableRaggedPrefillBatching)
        && !mConfig.enableWavefrontPrefillBatching)
    {
        struct BucketScore
        {
            int32_t usefulTokens{};
            int32_t paddedTokens{};
            int32_t rows{};
            int32_t finalContinuationRows{};
            int64_t completionCreditTokens{};
            double priority{};

            int64_t productivity() const
            {
                return static_cast<int64_t>(usefulTokens) + completionCreditTokens;
            }
        };
        auto bucketScore = [&](PhaseWorkItem const& candidate) {
            int32_t const candidateTokens = dispatchedPrefillTokens(candidate);
            bool const candidateInitial = candidate.tokenOffset == 0;
            std::vector<PhaseWorkItem const*> candidateRows;
            for (PhaseWorkItem const& item : queue)
            {
                if (isEligible(item, true)
                    && isPrefillBatchCompatible(item, candidateTokens, candidateInitial, candidate.allowChunkedPrefill))
                {
                    candidateRows.push_back(&item);
                }
            }
            std::stable_sort(
                candidateRows.begin(), candidateRows.end(), [&](PhaseWorkItem const* lhs, PhaseWorkItem const* rhs) {
                    return orderPrefillRow(lhs, rhs, candidateTokens);
                });
            int32_t const budgetRows = mConfig.maxPrefillBatchTokens > 0
                ? std::max(1, mConfig.maxPrefillBatchTokens / std::max(1, candidateTokens))
                : maxBatchSize;
            int32_t const selectedRows
                = std::min({maxBatchSize, static_cast<int32_t>(candidateRows.size()), budgetRows});
            int32_t usefulTokens{};
            int32_t finalContinuationRows{};
            int64_t completionCreditTokens{};
            for (int32_t index{}; index < selectedRows; ++index)
            {
                PhaseWorkItem const& row = *candidateRows[static_cast<size_t>(index)];
                int32_t const rowTokens = dispatchedPrefillTokens(row);
                usefulTokens += rowTokens;
                bool const finalContinuation
                    = row.tokenOffset > 0 && row.tokenOffset + rowTokens == row.promptTokenCount;
                finalContinuationRows += finalContinuation;
                if (finalContinuation)
                {
                    int32_t const unusedChunkTokens = std::max(0, mConfig.maxPrefillChunkTokens - rowTokens);
                    if (rowTokens < unusedChunkTokens)
                    {
                        completionCreditTokens += std::min(mConfig.prefillCompletionBonusTokens, unusedChunkTokens);
                    }
                }
            }
            return BucketScore{usefulTokens, selectedRows * candidateTokens, selectedRows, finalContinuationRows,
                completionCreditTokens, priorityRank(candidate)};
        };
        auto const lowerBucketScore = [&](PhaseWorkItem const& lhs, PhaseWorkItem const& rhs) {
            auto const lhsScore = bucketScore(lhs);
            auto const rhsScore = bucketScore(rhs);
            int64_t const lhsProductivity = lhsScore.productivity();
            int64_t const rhsProductivity = rhsScore.productivity();
            if (lhsProductivity != rhsProductivity)
            {
                return lhsProductivity < rhsProductivity;
            }
            if (mConfig.prefillCompletionBonusTokens > 0
                && lhsScore.finalContinuationRows != rhsScore.finalContinuationRows)
            {
                return lhsScore.finalContinuationRows < rhsScore.finalContinuationRows;
            }
            int64_t const lhsEfficiency
                = static_cast<int64_t>(lhsScore.usefulTokens) * std::max(1, rhsScore.paddedTokens);
            int64_t const rhsEfficiency
                = static_cast<int64_t>(rhsScore.usefulTokens) * std::max(1, lhsScore.paddedTokens);
            if (lhsEfficiency != rhsEfficiency)
            {
                return lhsEfficiency < rhsEfficiency;
            }
            if (lhsScore.rows != rhsScore.rows)
            {
                return lhsScore.rows < rhsScore.rows;
            }
            return lhsScore.priority < rhsScore.priority;
        };
        for (auto it = queue.cbegin(); it != queue.cend(); ++it)
        {
            if (isEligible(*it, true) && lowerBucketScore(*bucketSeed, *it))
            {
                bucketSeed = it;
            }
        }
    }
    int32_t const bucketTokens = dispatchedPrefillTokens(*bucketSeed);
    bool const bucketInitial = bucketSeed->tokenOffset == 0;
    bool const allowRaggedBatch = bucketSeed->allowChunkedPrefill;
    if (mConfig.enableWavefrontPrefillBatching && mPrefillCohortIds.empty())
    {
        std::vector<PhaseWorkItem const*> compatible;
        for (PhaseWorkItem const& item : queue)
        {
            if (isEligible(item, true) && isPrefillBatchCompatible(item, bucketTokens, bucketInitial, allowRaggedBatch))
            {
                compatible.push_back(&item);
            }
        }
        std::stable_sort(compatible.begin(), compatible.end(), [&](PhaseWorkItem const* lhs, PhaseWorkItem const* rhs) {
            return orderPrefillRow(lhs, rhs, bucketTokens);
        });
        int32_t const cohortLimit
            = std::min({mConfig.maxPrefillCohortSize, maxBatchSize, static_cast<int32_t>(compatible.size())});
        for (int32_t index = 0; index < cohortLimit; ++index)
        {
            mPrefillCohortIds.insert(compatible[static_cast<size_t>(index)]->requestId);
        }
    }
    std::vector<PhaseWorkItem const*> compatible;
    for (PhaseWorkItem const& item : queue)
    {
        bool const cohortEligible = !mConfig.enableWavefrontPrefillBatching
            || mPrefillCohortIds.find(item.requestId) != mPrefillCohortIds.end();
        if (cohortEligible && isEligible(item, true)
            && isPrefillBatchCompatible(item, bucketTokens, bucketInitial, allowRaggedBatch))
        {
            compatible.push_back(&item);
        }
    }
    std::stable_sort(compatible.begin(), compatible.end(),
        [&](PhaseWorkItem const* lhs, PhaseWorkItem const* rhs) { return orderPrefillRow(lhs, rhs, bucketTokens); });
    int32_t const tokenBudget = mConfig.maxPrefillBatchTokens > 0
        ? std::max(mConfig.maxPrefillBatchTokens, bucketTokens)
        : std::numeric_limits<int32_t>::max();
    int32_t const budgetRows = std::max(1, tokenBudget / std::max(1, bucketTokens));
    int32_t batchLimit = std::min({maxBatchSize, budgetRows, static_cast<int32_t>(compatible.size())});
    float predictedGpuMs{};
    float predictedDecodeSlowdownMs{};
    plan.prefillCostLookupRows = std::min<int32_t>(mConfig.maxPrefillBatchSize, compatible.size());
    plan.prefillCostLookupChunkLength = bucketTokens;
    for (int32_t index = 0; index < plan.prefillCostLookupRows; ++index)
    {
        plan.prefillCostLookupMaxPastKVLength
            = std::max(plan.prefillCostLookupMaxPastKVLength, compatible[static_cast<size_t>(index)]->tokenOffset);
    }
    int32_t const dynamicLimit = selectPrefillBatchSize(compatible, bucketTokens, bucketInitial,
        plan.kind == PhaseDispatchKind::kOverlap, plan.plannedDecodeBatchSize, plan.plannedDecodeMaxContextLength,
        state, predictedGpuMs, predictedDecodeSlowdownMs, plan.prefillCostCoverageMiss);
    if (dynamicLimit < 0)
    {
        plan.prefillDeferredForTpot = true;
        plan.predictedDecodeDebtUs = mPredictedDecodeDebtUs;
        plan.consecutiveOverlapBatches = mConsecutiveOverlapBatches;
        return batch;
    }
    if (dynamicLimit > 0)
    {
        batchLimit = std::min(batchLimit, dynamicLimit);
        plan.predictedPrefillGpuMs = predictedGpuMs;
        plan.predictedDecodeSlowdownMs = predictedDecodeSlowdownMs;
    }
    std::vector<uint64_t> selectedRequestIds;
    for (int32_t index = 0; index < batchLimit; ++index)
    {
        selectedRequestIds.push_back(compatible[static_cast<size_t>(index)]->requestId);
    }
    for (uint64_t const requestId : selectedRequestIds)
    {
        auto selected = std::find_if(
            queue.begin(), queue.end(), [requestId](PhaseWorkItem const& item) { return item.requestId == requestId; });
        check::check(selected != queue.end(), "Selected prefill request disappeared before dispatch");
        PhaseWorkItem item = *selected;
        queue.erase(selected);
        item.tokenCount = dispatchedPrefillTokens(item);
        check::check(mInFlightRequestIds.insert(item.requestId).second, "Request is already in flight");
        recordQueueWait(item.requestId);
        batch.push_back(item);
    }
    if (mConfig.enableWavefrontPrefillBatching)
    {
        ++mPrefillCohortTurns;
        plan.prefillCohortSize = static_cast<int32_t>(mPrefillCohortIds.size());
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
    plan.latencySafeFallback = mLatencySafeFallback;
    plan.overlapEvaluatedByCost = kind == PhaseDispatchKind::kOverlap && mConfig.enableCostAwareOverlapAdmission
        && !mLatencySafeFallback && state.prefillCandidateTokens > mConfig.maxOverlapPrefillTokens;
    plan.plannedDecodeBatchSize
        = kind == PhaseDispatchKind::kDecode || kind == PhaseDispatchKind::kOverlap ? selectDecodeBatchSize(state) : 0;
    plan.plannedDecodeMaxContextLength = plan.plannedDecodeBatchSize > 0 ? decodeMaxContextLength() : 0;
    if (kind == PhaseDispatchKind::kPrefill || kind == PhaseDispatchKind::kOverlap)
    {
        plan.prefillBatch = popBatch(mPrefillQueue, mConfig.maxPrefillBatchSize, true, state, plan);
    }
    if (kind == PhaseDispatchKind::kOverlap && plan.prefillDeferredForTpot)
    {
        plan.kind = PhaseDispatchKind::kDecode;
    }
    if (kind == PhaseDispatchKind::kDecode || kind == PhaseDispatchKind::kOverlap)
    {
        plan.decodeBatch = popBatch(mDecodeQueue, plan.plannedDecodeBatchSize, false, state, plan);
    }
    if (plan.kind == PhaseDispatchKind::kDecode)
    {
        ++mConsecutiveDecodeBatches;
        mConsecutiveOverlapBatches = 0;
        mPredictedDecodeDebtUs = 0.0;
    }
    else if (plan.kind == PhaseDispatchKind::kOverlap)
    {
        mConsecutiveDecodeBatches = 0;
        ++mConsecutiveOverlapBatches;
        mPredictedDecodeDebtUs += static_cast<double>(plan.predictedDecodeSlowdownMs) * 1000.0;
    }
    else if (plan.kind == PhaseDispatchKind::kPrefill)
    {
        mConsecutiveDecodeBatches = 0;
        mConsecutiveOverlapBatches = 0;
        mPredictedDecodeDebtUs = 0.0;
    }
    plan.predictedDecodeDebtUs = mPredictedDecodeDebtUs;
    plan.consecutiveOverlapBatches = mConsecutiveOverlapBatches;
    return plan;
}

int32_t PhaseQueueScheduler::decodeMaxContextLength() const noexcept
{
    int32_t maximum{};
    for (PhaseWorkItem const& item : mDecodeQueue)
    {
        if (isEligible(item, false))
        {
            maximum = std::max(maximum, item.tokenCount);
        }
    }
    return maximum;
}

int32_t PhaseQueueScheduler::selectDecodeBatchSize(PhaseQueueSnapshot const& state) const noexcept
{
    int32_t const available = std::min<int32_t>(mConfig.maxDecodeBatchSize, static_cast<int32_t>(state.decodeQueued));
    if (available <= 1 || !mConfig.enableDynamicDecodeBatching || mConfig.decodeBatchCosts.empty())
    {
        return available;
    }

    int32_t const maxContextLength = decodeMaxContextLength();

    struct Candidate
    {
        int32_t batchSize{};
        float p95GpuMs{};
        int32_t contextLimit{};
        int32_t profiledBatchLimit{};
    };
    std::vector<Candidate> candidates;
    int32_t coveringConfiguredBatch{std::numeric_limits<int32_t>::max()};
    int32_t largestConfiguredBatch{};
    for (PhaseDecodeBatchCost const& cost : mConfig.decodeBatchCosts)
    {
        largestConfiguredBatch = std::max(largestConfiguredBatch, cost.batchSize);
        if (cost.batchSize >= available)
        {
            coveringConfiguredBatch = std::min(coveringConfiguredBatch, cost.batchSize);
        }
    }
    if (coveringConfiguredBatch == std::numeric_limits<int32_t>::max())
    {
        coveringConfiguredBatch = largestConfiguredBatch;
    }
    for (PhaseDecodeBatchCost const& cost : mConfig.decodeBatchCosts)
    {
        if (cost.maxContextLength < maxContextLength)
        {
            continue;
        }
        int32_t const dispatchedBatchSize = std::min(cost.batchSize, available);
        auto existing = std::find_if(candidates.begin(), candidates.end(),
            [&](Candidate const& candidate) { return candidate.batchSize == dispatchedBatchSize; });
        if (existing == candidates.end())
        {
            candidates.push_back({dispatchedBatchSize, cost.p95GpuMs, cost.maxContextLength, cost.batchSize});
        }
        else if (cost.batchSize < existing->profiledBatchLimit
            || (cost.batchSize == existing->profiledBatchLimit && cost.maxContextLength < existing->contextLimit))
        {
            *existing = {dispatchedBatchSize, cost.p95GpuMs, cost.maxContextLength, cost.batchSize};
        }
    }
    if (candidates.empty())
    {
        return available;
    }
    bool const largestShapeCovered
        = std::any_of(candidates.begin(), candidates.end(), [coveringConfiguredBatch](Candidate const& candidate) {
              return candidate.profiledBatchLimit == coveringConfiguredBatch;
          });
    if (!largestShapeCovered)
    {
        // A smaller batch measured at this context is not evidence that a
        // larger runnable batch is unsafe. Preserve legacy largest-available
        // behavior instead of creating a low-BS backlog from sparse profiles.
        return available;
    }

    bool const urgent = state.decodeMaxSloPressure >= mConfig.decodeRecoveryPressureThreshold;
    double const remainingUs = std::max(0.0, mConfig.decodeQueueWaitTargetUs * (1.0 - state.decodeMaxSloPressure));
    Candidate const* selected{};
    for (Candidate const& candidate : candidates)
    {
        double const costUs = static_cast<double>(candidate.p95GpuMs) * 1000.0;
        if (urgent)
        {
            // Once the queue is already overdue, minimize work per token so
            // overload can recover. Repeatedly choosing the shortest absolute
            // kernel (usually BS1) makes queue growth unbounded.
            double const efficiency = static_cast<double>(candidate.batchSize) / candidate.p95GpuMs;
            double const selectedEfficiency
                = selected == nullptr ? 0.0 : static_cast<double>(selected->batchSize) / selected->p95GpuMs;
            if (selected == nullptr || efficiency > selectedEfficiency
                || (efficiency == selectedEfficiency && candidate.batchSize > selected->batchSize))
            {
                selected = &candidate;
            }
            continue;
        }
        if (costUs > remainingUs)
        {
            continue;
        }
        double const efficiency = static_cast<double>(candidate.batchSize) / candidate.p95GpuMs;
        double const selectedEfficiency
            = selected == nullptr ? 0.0 : static_cast<double>(selected->batchSize) / selected->p95GpuMs;
        if (selected == nullptr || efficiency > selectedEfficiency
            || (efficiency == selectedEfficiency && candidate.batchSize > selected->batchSize))
        {
            selected = &candidate;
        }
    }
    if (selected != nullptr)
    {
        return selected->batchSize;
    }

    auto const fastest
        = std::min_element(candidates.begin(), candidates.end(), [](Candidate const& lhs, Candidate const& rhs) {
              return lhs.p95GpuMs < rhs.p95GpuMs || (lhs.p95GpuMs == rhs.p95GpuMs && lhs.batchSize > rhs.batchSize);
          });
    return fastest->batchSize;
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
        mPrefillCohortIds.erase(item.requestId);
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
    mPrefillCohortIds.erase(item.requestId);
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
    if (mConfig.enableTpotHysteresis && metrics.decodeBatchSize > 0 && metrics.decodeGpuMs > 0.0F)
    {
        double const sampleUs = metrics.decodeQueueWaitUs + static_cast<double>(metrics.decodeGpuMs) * 1000.0;
        mRecentDecodeTpotUs.push_back(sampleUs);
        if (mRecentDecodeTpotUs.size() > mConfig.tpotHysteresisWindow)
        {
            mRecentDecodeTpotUs.pop_front();
        }
        ++mTelemetry.decodeTpotSampleCount;
        if (mRecentDecodeTpotUs.size() >= mConfig.minTpotHysteresisSamples)
        {
            std::vector<double> ordered(mRecentDecodeTpotUs.begin(), mRecentDecodeTpotUs.end());
            std::sort(ordered.begin(), ordered.end());
            size_t const p95Index = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(ordered.size()))) - 1;
            mTelemetry.recentDecodeTpotP95Us = ordered[p95Index];
            mTelemetry.recentDecodeTpotPressure
                = static_cast<float>(mTelemetry.recentDecodeTpotP95Us / mConfig.decodeQueueWaitTargetUs);
            bool const previousFallback = mLatencySafeFallback;
            if (!mLatencySafeFallback && mTelemetry.recentDecodeTpotPressure >= mConfig.tpotHysteresisEnterRatio)
            {
                mLatencySafeFallback = true;
            }
            else if (mLatencySafeFallback && mTelemetry.recentDecodeTpotPressure <= mConfig.tpotHysteresisExitRatio)
            {
                mLatencySafeFallback = false;
            }
            if (previousFallback != mLatencySafeFallback)
            {
                ++mTelemetry.tpotHysteresisTransitions;
            }
        }
    }
    mTelemetry.latencySafeFallback = mLatencySafeFallback;
    ++mTelemetry.sampleCount;
    mTelemetry.lastDispatch = metrics;
}

PhaseSchedulerTelemetry const& PhaseQueueScheduler::telemetry() const noexcept
{
    return mTelemetry;
}

void PhaseQueueScheduler::resetHistory()
{
    check::check(empty() && mActiveRequestIds.empty() && mInFlightRequestIds.empty(),
        "Scheduling history can only be reset while the scheduler is idle");
    mTelemetry = {};
    mRecentDecodeTpotUs.clear();
    mLatencySafeFallback = false;
    mConsecutiveDecodeBatches = 0;
    mConsecutiveOverlapBatches = 0;
    mPredictedDecodeDebtUs = 0.0;
    mPrefillCohortIds.clear();
    mPrefillCohortTurns = 0;
}

} // namespace rt
} // namespace trt_edgellm
