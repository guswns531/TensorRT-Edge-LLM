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
#include <tuple>
#include <unordered_set>
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

size_t phaseDecodeReplacementRows(
    std::vector<uint64_t> const& selectedRequestIds, std::vector<uint64_t> const& previousRequestIds)
{
    std::unordered_set<uint64_t> const previous(previousRequestIds.begin(), previousRequestIds.end());
    return static_cast<size_t>(std::count_if(selectedRequestIds.begin(), selectedRequestIds.end(),
        [&](uint64_t requestId) { return previous.find(requestId) == previous.end(); }));
}

char const* phaseDrainPreferenceName(PhaseDrainPreference preference) noexcept
{
    char const* result = "unknown";
    switch (preference)
    {
    case PhaseDrainPreference::kNone: result = "none"; break;
    case PhaseDrainPreference::kPrefill: result = "prefill"; break;
    case PhaseDrainPreference::kDecode: result = "decode"; break;
    }
    return result;
}

PhaseQueueScheduler::PhaseQueueScheduler(PhaseQueueSchedulerConfig config)
    : mConfig(std::move(config))
    , mOnlineDecodeCostLearningActive(mConfig.enableOnlineDecodeCostLearning)
{
    applySchedulerProfile(mConfig);
    check::check(mConfig.maxPrefillBatchSize > 0, "maxPrefillBatchSize must be positive");
    check::check(mConfig.maxDecodeBatchSize > 0, "maxDecodeBatchSize must be positive");
    check::check(
        mConfig.maxOverlapPrefillBatchSize >= 0 && mConfig.maxOverlapPrefillBatchSize <= mConfig.maxPrefillBatchSize,
        "maxOverlapPrefillBatchSize must be zero or no greater than maxPrefillBatchSize");
    check::check(mConfig.maxOverlapPrefillTokens >= 0, "maxOverlapPrefillTokens must be non-negative");
    check::check(mConfig.maxPrefillChunkTokens >= 0, "maxPrefillChunkTokens must be non-negative");
    check::check(mConfig.decodeActivePrefillChunkTokens >= 0
            && (mConfig.decodeActivePrefillChunkTokens == 0
                || mConfig.decodeActivePrefillChunkTokens <= mConfig.maxPrefillChunkTokens),
        "decodeActivePrefillChunkTokens must be zero or within the maximum prefill chunk length");
    check::check(!mConfig.enablePackedPrefillTokenLayout || mConfig.maxPrefillChunkTokens > 0,
        "Packed prefill token layout requires a positive maximum chunk length");
    check::check(mConfig.maxPrefillBatchTokens >= 0, "maxPrefillBatchTokens must be non-negative");
    check::check(mConfig.prefillCompletionBonusTokens >= 0, "prefillCompletionBonusTokens must be non-negative");
    check::check(std::isfinite(mConfig.decodeRowReplacementCostMs) && mConfig.decodeRowReplacementCostMs >= 0.0F,
        "Decode row replacement cost must be finite and non-negative");
    for (PhaseDecodeBatchCost const& cost : mConfig.decodeBatchCosts)
    {
        check::check(cost.batchSize > 0, "Decode cost batch size must be positive");
        check::check(cost.maxContextLength > 0, "Decode cost context length must be positive");
        check::check(cost.maxTotalContextTokens >= 0, "Decode cost total context tokens cannot be negative");
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
    check::check(
        mConfig.onlineDecodeCostMinSamples > 0 && mConfig.onlineDecodeCostMinSamples <= mConfig.onlineDecodeCostWindow,
        "Online decode cost sample bounds are invalid");
    check::check(mConfig.onlineDecodeContextBucketTokens > 0, "Online decode context bucket must be positive");
    check::check(std::isfinite(mConfig.onlineDecodeCostMaxAdjustmentRatio)
            && mConfig.onlineDecodeCostMaxAdjustmentRatio >= 0.0F && mConfig.onlineDecodeCostMaxAdjustmentRatio < 1.0F,
        "Online decode cost adjustment ratio must be finite and in [0, 1)");
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
    check::check(mConfig.adaptivePrefillChunkCandidates.empty() || mConfig.enableAdaptivePrefillChunking,
        "Adaptive prefill chunk candidates require adaptive prefill chunking");
    check::check(!mConfig.enableCostAwarePrefillShapeSelection
            || (mConfig.enableDynamicPrefillBatching && mConfig.enableAdaptivePrefillChunking
                && !mConfig.adaptivePrefillChunkCandidates.empty()),
        "Cost-aware prefill shape selection requires dynamic batching and bounded adaptive chunks");
    check::check(std::isfinite(mConfig.prefillShapeDecodePenaltyWeight)
            && mConfig.prefillShapeDecodePenaltyWeight >= 0.0F && std::isfinite(mConfig.prefillShapeEnqueueCostMs)
            && mConfig.prefillShapeEnqueueCostMs >= 0.0F,
        "Cost-aware prefill shape penalties must be finite and non-negative");
    check::check(
        mConfig.prefillShapeDrainBacklogTokens >= 0, "Cost-aware prefill shape drain backlog must be non-negative");
    check::check(!mConfig.allowAdaptivePrefillCompletionSplit
            || (mConfig.enableAdaptivePrefillChunking && !mConfig.adaptivePrefillChunkCandidates.empty()),
        "Adaptive prefill completion splitting requires bounded adaptive prefill chunking");
    int32_t previousChunkCandidate{};
    for (int32_t const candidate : mConfig.adaptivePrefillChunkCandidates)
    {
        check::check(candidate > previousChunkCandidate,
            "Adaptive prefill chunk candidates must be positive, unique, and strictly increasing");
        check::check(candidate <= mConfig.maxPrefillChunkTokens,
            "Adaptive prefill chunk candidate exceeds the maximum prefill chunk length");
        previousChunkCandidate = candidate;
    }
    check::check(std::isfinite(mConfig.adaptivePrefillChunkDecodePressureThreshold)
            && mConfig.adaptivePrefillChunkDecodePressureThreshold > 0.0F
            && mConfig.adaptivePrefillChunkDecodePressureThreshold <= 1.0F,
        "Adaptive prefill chunk decode pressure threshold must be in (0, 1]");
    check::check(mConfig.decodeBurstLimit > 0, "decodeBurstLimit must be positive");
    check::check(mConfig.prefillQueueWaitTargetUs > 0.0, "prefillQueueWaitTargetUs must be positive");
    check::check(mConfig.decodeQueueWaitTargetUs > 0.0, "decodeQueueWaitTargetUs must be positive");
    check::check(mConfig.maxPredictedOverlapPrefillMs >= 0.0F, "maxPredictedOverlapPrefillMs must be non-negative");
    check::check(mConfig.minObservedOverlapRatio >= 0.0F && mConfig.minObservedOverlapRatio <= 1.0F,
        "minObservedOverlapRatio must be in [0, 1]");
    check::check(mConfig.pagePressureDecodeThreshold >= 0.0F && mConfig.pagePressureDecodeThreshold <= 1.0F,
        "pagePressureDecodeThreshold must be in [0, 1]");
    check::check(mConfig.externalDrainPreferenceMaxConsecutiveDispatches > 0U,
        "External drain preference consecutive dispatch limit must be positive");
    check::check(std::isfinite(mConfig.externalPrefillDrainDecodePressureLimit)
            && mConfig.externalPrefillDrainDecodePressureLimit >= 0.0F
            && mConfig.externalPrefillDrainDecodePressureLimit <= 1.0F,
        "External prefill drain decode pressure limit must be in [0, 1]");
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
        mDecodeCohortIds.erase(requestId);
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
    bool const hasDecodeWork = std::any_of(mDecodeQueue.begin(), mDecodeQueue.end(),
        [this](PhaseWorkItem const& item) { return isEligible(item, false); });
    int32_t const overlapPrefillBatchSize
        = mConfig.maxOverlapPrefillBatchSize > 0 ? mConfig.maxOverlapPrefillBatchSize : mConfig.maxPrefillBatchSize;
    int32_t const candidatePrefillBatchSize = hasDecodeWork ? overlapPrefillBatchSize : mConfig.maxPrefillBatchSize;
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
            if (isEligible(item, true) && bucketRows < candidatePrefillBatchSize
                && isPrefillBatchCompatible(item, *prefillSeed, bucketTokens, bucketInitial, allowRaggedBatch))
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
                result.decodeCandidateContextTokens += item.tokenCount;
                result.decodeCandidateMaxContextLength
                    = std::max(result.decodeCandidateMaxContextLength, item.tokenCount);
            }
        }
    }
    result.decodeCandidateTokens = static_cast<int32_t>(std::min<int64_t>(
        result.decodeCandidateContextTokens, static_cast<int64_t>(std::numeric_limits<int32_t>::max())));
    if (mConfig.resourceSupplier)
    {
        PhaseQueueResourceSnapshot const resources = mConfig.resourceSupplier();
        result.pagePoolTotalBundles = resources.pagePoolTotalBundles;
        result.pagePoolAllocatedBundles = resources.pagePoolAllocatedBundles;
        result.pagePoolAvailableBundles = resources.pagePoolAvailableBundles;
        result.pageReservationGuaranteedBundles = resources.pageReservationGuaranteedBundles;
        result.pageReservationAvailableBundles = resources.pageReservationAvailableBundles;
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

PhaseQueueSnapshot PhaseQueueScheduler::queueSnapshot() const
{
    return snapshot();
}

bool PhaseQueueScheduler::isEligible(PhaseWorkItem const& item, bool prefill) const
{
    return !(prefill && mPrefillDispatchBlocked)
        && (!mConfig.eligibilityPolicy || mConfig.eligibilityPolicy(item, prefill));
}

void PhaseQueueScheduler::setPrefillDispatchBlocked(bool blocked) noexcept
{
    mPrefillDispatchBlocked = blocked;
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
    int32_t const pagePoolTotal = state.pagePoolTotalBundles > 0
        ? state.pagePoolTotalBundles
        : (telemetry.lastDispatch.has_value() ? telemetry.lastDispatch->pagePoolTotalBundles : 0);
    int32_t const pagePoolAllocated = state.pagePoolTotalBundles > 0
        ? state.pagePoolAllocatedBundles
        : (telemetry.lastDispatch.has_value() ? telemetry.lastDispatch->pagePoolAllocatedBundles : 0);
    if (mConfig.pagePressureDecodeThreshold > 0.0F && pagePoolTotal > 0)
    {
        float const pagePressure = static_cast<float>(pagePoolAllocated) / static_cast<float>(pagePoolTotal);
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

    int32_t maximum = std::min(item.tokenCount, mConfig.maxPrefillChunkTokens);
    bool const largePrefillBacklog = mConfig.largePrefillChunkQueueThreshold > 0
        && mPrefillQueue.size() >= mConfig.largePrefillChunkQueueThreshold;
    if (!mDecodeQueue.empty() && !largePrefillBacklog && mConfig.decodeActivePrefillChunkTokens > 0)
    {
        maximum = std::min(maximum, mConfig.decodeActivePrefillChunkTokens);
    }
    if (!mConfig.enableAdaptivePrefillChunking || mDecodeQueue.empty() || mConfig.enableCostAwarePrefillShapeSelection)
    {
        return maximum;
    }

    int32_t const minimum = std::min(maximum, mConfig.minPrefillChunkTokens);
    int32_t selected = maximum;
    if (mConfig.adaptivePrefillChunkCandidates.empty() && mTelemetry.prefillGpuMsPerToken > 0.0F)
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

    if (mConfig.adaptivePrefillChunkCandidates.empty() && mTelemetry.overlapSampleCount > 0
        && mTelemetry.overlapRatio < mConfig.minObservedOverlapRatio)
    {
        selected = minimum;
    }

    float const decodeQueuePressure
        = std::min(1.0F, static_cast<float>(mDecodeQueue.size()) / static_cast<float>(mConfig.maxDecodeBatchSize));
    float const decodePressure = decodeQueuePressure * mTelemetry.recentDecodeTpotPressure;
    if (!mConfig.adaptivePrefillChunkCandidates.empty() && maximum == item.tokenCount
        && (!mConfig.allowAdaptivePrefillCompletionSplit
            || decodePressure < mConfig.adaptivePrefillChunkDecodePressureThreshold))
    {
        return maximum;
    }
    if (!mConfig.adaptivePrefillChunkCandidates.empty()
        && decodePressure >= mConfig.adaptivePrefillChunkDecodePressureThreshold)
    {
        selected = minimum;
    }

    selected = std::clamp(selected, minimum, maximum);
    if (!mConfig.adaptivePrefillChunkCandidates.empty())
    {
        int32_t boundedCandidate{};
        int32_t smallestRunnableCandidate{};
        for (int32_t const candidate : mConfig.adaptivePrefillChunkCandidates)
        {
            if (candidate > maximum)
            {
                break;
            }
            if (smallestRunnableCandidate == 0)
            {
                smallestRunnableCandidate = candidate;
            }
            if (candidate <= selected)
            {
                boundedCandidate = candidate;
            }
        }
        if (boundedCandidate > 0)
        {
            return boundedCandidate;
        }
        if (smallestRunnableCandidate > 0)
        {
            return smallestRunnableCandidate;
        }
        // Only the final request tail may be smaller than the profiled set.
        return maximum;
    }
    int32_t const aligned = selected / mConfig.prefillChunkAlignment * mConfig.prefillChunkAlignment;
    return std::max(minimum, aligned);
}

int32_t PhaseQueueScheduler::costAwarePrefillTokens(PhaseWorkItem const& item, int32_t chunkLimit) const noexcept
{
    if (!item.allowChunkedPrefill || !mConfig.supportsChunkedPrefill || mConfig.maxPrefillChunkTokens == 0)
    {
        return item.tokenCount;
    }

    int32_t maximum = std::min(item.tokenCount, mConfig.maxPrefillChunkTokens);
    bool const largePrefillBacklog = mConfig.largePrefillChunkQueueThreshold > 0
        && mPrefillQueue.size() >= mConfig.largePrefillChunkQueueThreshold;
    if (!mDecodeQueue.empty() && !largePrefillBacklog && mConfig.decodeActivePrefillChunkTokens > 0)
    {
        maximum = std::min(maximum, mConfig.decodeActivePrefillChunkTokens);
    }
    if (maximum == item.tokenCount && !mConfig.allowAdaptivePrefillCompletionSplit)
    {
        return maximum;
    }
    return std::min(maximum, chunkLimit);
}

bool PhaseQueueScheduler::isPrefillBatchCompatible(PhaseWorkItem const& item, PhaseWorkItem const& seed,
    int32_t paddedChunkLength, bool initialChunk, bool allowRaggedBatch) const noexcept
{
    if (item.prefillClass != seed.prefillClass)
    {
        return false;
    }
    if (seed.exclusivePrefill || item.exclusivePrefill)
    {
        return seed.exclusivePrefill && item.requestId == seed.requestId;
    }
    int32_t const itemTokens = dispatchedPrefillTokens(item);
    if ((item.tokenOffset == 0) != initialChunk || itemTokens > paddedChunkLength)
    {
        return false;
    }
    if (itemTokens == paddedChunkLength)
    {
        return true;
    }
    bool const bothAtomic = !seed.allowChunkedPrefill && !item.allowChunkedPrefill;
    return mConfig.enableRaggedPrefillBatching && (bothAtomic || (allowRaggedBatch && item.allowChunkedPrefill));
}

int32_t PhaseQueueScheduler::selectPrefillBatchSize(std::vector<PhaseWorkItem const*> const& candidates,
    int32_t chunkLength, bool initialChunk, bool overlap, int32_t plannedDecodeBatchSize,
    int32_t plannedDecodeMaxContextLength, PhaseQueueSnapshot const& state, bool preferMaximumProgress,
    float& predictedGpuMs, float& predictedDecodeSlowdownMs, bool& costCoverageMiss) const noexcept
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
    int32_t const requiredConcurrentDecodeBatchSize = overlap ? plannedDecodeBatchSize : 0;
    PhasePrefillClass const prefillClass = candidates.front()->prefillClass;
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
                || cost.maxConcurrentDecodeBatchSize < requiredConcurrentDecodeBatchSize
                || (cost.prefillClass != PhasePrefillClass::kAny && cost.prefillClass != prefillClass))
            {
                continue;
            }
            if (selected == nullptr || cost.chunkLength < selected->chunkLength
                || (cost.chunkLength == selected->chunkLength && cost.maxPastKVLength < selected->maxPastKVLength)
                || (cost.chunkLength == selected->chunkLength && cost.maxPastKVLength == selected->maxPastKVLength
                    && cost.maxConcurrentDecodeBatchSize < selected->maxConcurrentDecodeBatchSize)
                || (cost.chunkLength == selected->chunkLength && cost.maxPastKVLength == selected->maxPastKVLength
                    && cost.maxConcurrentDecodeBatchSize == selected->maxConcurrentDecodeBatchSize
                    && selected->prefillClass == PhasePrefillClass::kAny && cost.prefillClass == prefillClass))
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
                    || cost.maxDecodeContextLength < plannedDecodeMaxContextLength
                    || (cost.prefillClass != PhasePrefillClass::kAny && cost.prefillClass != prefillClass))
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
                        && cost.maxDecodeContextLength < selectedOverlap->maxDecodeContextLength)
                    || (cost.chunkLength == selectedOverlap->chunkLength
                        && cost.decodeBatchSize == selectedOverlap->decodeBatchSize
                        && cost.maxPrefillPastKVLength == selectedOverlap->maxPrefillPastKVLength
                        && cost.maxDecodeContextLength == selectedOverlap->maxDecodeContextLength
                        && selectedOverlap->prefillClass == PhasePrefillClass::kAny
                        && cost.prefillClass == prefillClass))
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
                usefulTokens += std::min(dispatchedPrefillTokens(*candidates[static_cast<size_t>(index)]), chunkLength);
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
        if (selected == nullptr || (preferMaximumProgress && candidate.usefulTokens > selected->usefulTokens)
            || (preferMaximumProgress && candidate.usefulTokens == selected->usefulTokens
                && efficiency > selectedEfficiency)
            || (!preferMaximumProgress && efficiency > selectedEfficiency)
            || (!preferMaximumProgress && efficiency == selectedEfficiency
                && candidate.batchSize > selected->batchSize))
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
        if (mConfig.enableDecodeCohortBatching)
        {
            for (PhaseWorkItem const& item : queue)
            {
                if (static_cast<int32_t>(mDecodeCohortIds.size()) >= mConfig.maxDecodeBatchSize)
                {
                    break;
                }
                if (isEligible(item, false))
                {
                    mDecodeCohortIds.insert(item.requestId);
                }
            }
        }
        for (int32_t i = 0; i < count; ++i)
        {
            auto selected = std::find_if(queue.begin(), queue.end(), [this](PhaseWorkItem const& item) {
                return isEligible(item, false)
                    && (!mConfig.enableDecodeCohortBatching
                        || mDecodeCohortIds.find(item.requestId) != mDecodeCohortIds.end());
            });
            check::check(selected != queue.end(), "Eligible decode work disappeared during batch selection");
            if (mConfig.enablePriorityBatching)
            {
                for (auto it = std::next(selected); it != queue.end(); ++it)
                {
                    if (isEligible(*it, false)
                        && (!mConfig.enableDecodeCohortBatching
                            || mDecodeCohortIds.find(it->requestId) != mDecodeCohortIds.end())
                        && higherPriority(*selected, *it))
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
        mPreviousDecodeSelectionIds.clear();
        mPreviousDecodeSelectionIds.reserve(batch.size());
        for (PhaseWorkItem const& item : batch)
        {
            mPreviousDecodeSelectionIds.push_back(item.requestId);
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
                    && isPrefillBatchCompatible(
                        item, candidate, candidateTokens, candidateInitial, candidate.allowChunkedPrefill))
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
    if (mConfig.enableRaggedPrefillBatching && !bucketSeed->allowChunkedPrefill && !bucketSeed->exclusivePrefill)
    {
        bool const initialChunk = bucketSeed->tokenOffset == 0;
        for (auto it = queue.cbegin(); it != queue.cend(); ++it)
        {
            bool const compatibleAtomic = isEligible(*it, true) && !it->allowChunkedPrefill && !it->exclusivePrefill
                && (it->tokenOffset == 0) == initialChunk;
            if (compatibleAtomic && dispatchedPrefillTokens(*it) > dispatchedPrefillTokens(*bucketSeed))
            {
                bucketSeed = it;
            }
        }
    }
    int32_t bucketTokens = dispatchedPrefillTokens(*bucketSeed);
    bool const bucketInitial = bucketSeed->tokenOffset == 0;
    bool const allowRaggedBatch = bucketSeed->allowChunkedPrefill;
    if (mConfig.enableWavefrontPrefillBatching && mPrefillCohortIds.empty())
    {
        std::vector<PhaseWorkItem const*> compatible;
        for (PhaseWorkItem const& item : queue)
        {
            if (isEligible(item, true)
                && isPrefillBatchCompatible(item, *bucketSeed, bucketTokens, bucketInitial, allowRaggedBatch))
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
    auto collectCompatible = [&](int32_t paddedChunkLength, bool costAwareShape) {
        std::vector<PhaseWorkItem const*> rows;
        for (PhaseWorkItem const& item : queue)
        {
            bool const cohortEligible = !mConfig.enableWavefrontPrefillBatching
                || mPrefillCohortIds.find(item.requestId) != mPrefillCohortIds.end();
            if (!cohortEligible || !isEligible(item, true) || (item.tokenOffset == 0) != bucketInitial)
            {
                continue;
            }
            if ((bucketSeed->exclusivePrefill || item.exclusivePrefill)
                && !(bucketSeed->exclusivePrefill && item.requestId == bucketSeed->requestId))
            {
                continue;
            }
            int32_t const itemTokens
                = costAwareShape ? costAwarePrefillTokens(item, paddedChunkLength) : dispatchedPrefillTokens(item);
            bool const bothAtomic = !bucketSeed->allowChunkedPrefill && !item.allowChunkedPrefill;
            bool const compatibleShape = itemTokens == paddedChunkLength
                || (itemTokens < paddedChunkLength && mConfig.enableRaggedPrefillBatching
                    && (bothAtomic || (allowRaggedBatch && item.allowChunkedPrefill)));
            if (compatibleShape)
            {
                rows.push_back(&item);
            }
        }
        auto orderRows = [&](PhaseWorkItem const* lhs, PhaseWorkItem const* rhs) {
            if (!costAwareShape)
            {
                return orderPrefillRow(lhs, rhs, paddedChunkLength);
            }
            int32_t const lhsTokens = costAwarePrefillTokens(*lhs, paddedChunkLength);
            int32_t const rhsTokens = costAwarePrefillTokens(*rhs, paddedChunkLength);
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
        std::stable_sort(rows.begin(), rows.end(), orderRows);
        return rows;
    };
    auto batchLimitFor = [&](int32_t paddedChunkLength, size_t compatibleRows) {
        int32_t const tokenBudget = mConfig.maxPrefillBatchTokens > 0
            ? std::max(mConfig.maxPrefillBatchTokens, paddedChunkLength)
            : std::numeric_limits<int32_t>::max();
        int32_t const budgetRows = std::max(1, tokenBudget / std::max(1, paddedChunkLength));
        return std::min({maxBatchSize, budgetRows, static_cast<int32_t>(compatibleRows)});
    };
    std::vector<PhaseWorkItem const*> compatible = collectCompatible(bucketTokens, false);
    int32_t batchLimit = batchLimitFor(bucketTokens, compatible.size());
    int32_t costLookupRows = std::min<int32_t>(mConfig.maxPrefillBatchSize, compatible.size());
    int32_t costLookupMaxPastKVLength{};
    for (int32_t index = 0; index < costLookupRows; ++index)
    {
        costLookupMaxPastKVLength
            = std::max(costLookupMaxPastKVLength, compatible[static_cast<size_t>(index)]->tokenOffset);
    }
    float predictedGpuMs{};
    float predictedDecodeSlowdownMs{};
    bool shapeSelected{};
    bool shapeRejectedForTpot{};
    bool const selectedProductiveChunk = !mConfig.adaptivePrefillChunkCandidates.empty()
        && bucketTokens == mConfig.adaptivePrefillChunkCandidates.back();
    bool const drainPrefillBacklog = mConfig.prefillShapeDrainBacklogTokens > 0
        && state.prefillRemainingTokens >= mConfig.prefillShapeDrainBacklogTokens;
    plan.prefillShapeDrainMode
        = mConfig.enableCostAwarePrefillShapeSelection && selectedProductiveChunk && drainPrefillBacklog;
    if (mConfig.enableCostAwarePrefillShapeSelection && selectedProductiveChunk)
    {
        struct JointShape
        {
            int32_t chunkLength{};
            int32_t batchSize{};
            float gpuMs{};
            float decodeSlowdownMs{};
            float score{};
            int32_t usefulTokens{};
            int32_t costLookupRows{};
            int32_t costLookupMaxPastKVLength{};
            std::vector<PhaseWorkItem const*> rows;
        };
        std::optional<JointShape> selectedShape;
        for (int32_t const candidateChunk : mConfig.adaptivePrefillChunkCandidates)
        {
            std::vector<PhaseWorkItem const*> candidateRows = collectCompatible(candidateChunk, true);
            int32_t const candidateBatchLimit = batchLimitFor(candidateChunk, candidateRows.size());
            if (candidateBatchLimit < mConfig.minDynamicPrefillBatchSize)
            {
                continue;
            }
            candidateRows.resize(static_cast<size_t>(candidateBatchLimit));
            int32_t candidateMaxPastKVLength{};
            for (PhaseWorkItem const* row : candidateRows)
            {
                candidateMaxPastKVLength = std::max(candidateMaxPastKVLength, row->tokenOffset);
            }
            float candidateGpuMs{};
            float candidateDecodeSlowdownMs{};
            bool candidateCoverageMiss{};
            int32_t const candidateBatch = selectPrefillBatchSize(candidateRows, candidateChunk, bucketInitial,
                plan.kind == PhaseDispatchKind::kOverlap, plan.plannedDecodeBatchSize,
                plan.plannedDecodeMaxContextLength, state, drainPrefillBacklog, candidateGpuMs,
                candidateDecodeSlowdownMs, candidateCoverageMiss);
            plan.prefillCostCoverageMiss = plan.prefillCostCoverageMiss || candidateCoverageMiss;
            ++plan.prefillShapeCandidatesEvaluated;
            if (candidateBatch < 0)
            {
                shapeRejectedForTpot = true;
                continue;
            }
            if (candidateBatch == 0)
            {
                continue;
            }
            int32_t usefulTokens{};
            for (int32_t index{}; index < candidateBatch; ++index)
            {
                usefulTokens += costAwarePrefillTokens(*candidateRows[static_cast<size_t>(index)], candidateChunk);
            }
            float const decodeQueuePressure = std::min(
                1.0F, static_cast<float>(state.decodeQueued) / static_cast<float>(mConfig.maxDecodeBatchSize));
            float const effectiveMs = candidateGpuMs
                + mConfig.prefillShapeDecodePenaltyWeight * decodeQueuePressure * candidateDecodeSlowdownMs
                + mConfig.prefillShapeEnqueueCostMs;
            float const score = static_cast<float>(usefulTokens) / effectiveMs;
            if (!selectedShape.has_value() || (drainPrefillBacklog && usefulTokens > selectedShape->usefulTokens)
                || (drainPrefillBacklog && usefulTokens == selectedShape->usefulTokens && score > selectedShape->score)
                || (!drainPrefillBacklog && score > selectedShape->score)
                || (!drainPrefillBacklog && score == selectedShape->score && usefulTokens > selectedShape->usefulTokens)
                || (score == selectedShape->score && usefulTokens == selectedShape->usefulTokens
                    && candidateChunk > selectedShape->chunkLength))
            {
                candidateRows.resize(static_cast<size_t>(candidateBatch));
                selectedShape = JointShape{candidateChunk, candidateBatch, candidateGpuMs, candidateDecodeSlowdownMs,
                    score, usefulTokens, candidateBatchLimit, candidateMaxPastKVLength, std::move(candidateRows)};
            }
        }
        if (selectedShape.has_value())
        {
            bucketTokens = selectedShape->chunkLength;
            batchLimit = selectedShape->batchSize;
            predictedGpuMs = selectedShape->gpuMs;
            predictedDecodeSlowdownMs = selectedShape->decodeSlowdownMs;
            costLookupRows = selectedShape->costLookupRows;
            costLookupMaxPastKVLength = selectedShape->costLookupMaxPastKVLength;
            compatible = std::move(selectedShape->rows);
            plan.predictedPrefillShapeScore = selectedShape->score;
            shapeSelected = true;
        }
    }
    if (!shapeSelected)
    {
        if (shapeRejectedForTpot && plan.kind == PhaseDispatchKind::kOverlap && mConfig.enableTpotHardGuard)
        {
            plan.prefillDeferredForTpot = true;
            plan.predictedDecodeDebtUs = mPredictedDecodeDebtUs;
            plan.consecutiveOverlapBatches = mConsecutiveOverlapBatches;
            return batch;
        }
        int32_t dynamicLimit{};
        if (!mConfig.enableCostAwarePrefillShapeSelection || selectedProductiveChunk)
        {
            std::vector<PhaseWorkItem const*> costCandidates = compatible;
            costCandidates.resize(static_cast<size_t>(batchLimit));
            dynamicLimit = selectPrefillBatchSize(costCandidates, bucketTokens, bucketInitial,
                plan.kind == PhaseDispatchKind::kOverlap, plan.plannedDecodeBatchSize,
                plan.plannedDecodeMaxContextLength, state, false, predictedGpuMs, predictedDecodeSlowdownMs,
                plan.prefillCostCoverageMiss);
        }
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
        }
    }
    plan.prefillCostLookupRows = costLookupRows;
    plan.prefillCostLookupChunkLength = bucketTokens;
    plan.prefillCostLookupMaxPastKVLength = costLookupMaxPastKVLength;
    if (shapeSelected || predictedGpuMs > 0.0F)
    {
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
        item.tokenCount = std::min(item.tokenCount, bucketTokens);
        queue.erase(selected);
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
    refreshExternalDrainPreference();
    PhaseQueueSnapshot const state = snapshot();
    bool const expiredPrefillHardGuard
        = mConfig.enablePrefillTtftHardGuard && state.prefillQueued > 0U && state.prefillMinTtftSlackUs <= 0.0;
    PhaseDispatchKind const baseline = expiredPrefillHardGuard
        ? PhaseDispatchKind::kPrefill
        : (mConfig.metricsPolicy
                  ? mConfig.metricsPolicy(state, mTelemetry)
                  : (mConfig.enableMetricsPolicy ? metricsDecision(state, mTelemetry)
                                                 : (mConfig.policy ? mConfig.policy(state) : defaultDecision(state))));
    bool drainPreferenceApplied{};
    PhaseDispatchKind const kind = applyExternalDrainPreference(state, baseline, drainPreferenceApplied);
    check::check(kind != PhaseDispatchKind::kPrefill || state.prefillQueued > 0,
        "Scheduling policy selected an empty prefill queue");
    check::check(kind != PhaseDispatchKind::kDecode || state.decodeQueued > 0,
        "Scheduling policy selected an empty decode queue");
    check::check(kind != PhaseDispatchKind::kOverlap || (state.prefillQueued > 0 && state.decodeQueued > 0),
        "Scheduling policy selected overlap without work in both queues");

    PhaseDispatchPlan plan;
    plan.kind = kind;
    plan.drainPreference = mActiveDrainPreference;
    plan.drainPreferenceApplied = drainPreferenceApplied;
    plan.adaptiveChunkDecodeQueuePressure
        = std::min(1.0F, static_cast<float>(mDecodeQueue.size()) / static_cast<float>(mConfig.maxDecodeBatchSize));
    plan.adaptiveChunkObservedTpotPressure = mTelemetry.recentDecodeTpotPressure;
    plan.adaptiveChunkCombinedPressure = plan.adaptiveChunkDecodeQueuePressure * plan.adaptiveChunkObservedTpotPressure;
    plan.latencySafeFallback = mLatencySafeFallback;
    plan.overlapEvaluatedByCost = kind == PhaseDispatchKind::kOverlap && mConfig.enableCostAwareOverlapAdmission
        && !mLatencySafeFallback && state.prefillCandidateTokens > mConfig.maxOverlapPrefillTokens;
    plan.plannedDecodeBatchSize
        = kind == PhaseDispatchKind::kDecode || kind == PhaseDispatchKind::kOverlap ? selectDecodeBatchSize(state) : 0;
    if (plan.plannedDecodeBatchSize > 0)
    {
        std::tie(plan.plannedDecodeContextTokens, plan.plannedDecodeMaxContextLength)
            = decodeCandidateShape(plan.plannedDecodeBatchSize);
        plan.predictedDecodeReplacementRows = decodeCandidateReplacementRows(plan.plannedDecodeBatchSize);
    }
    if (kind == PhaseDispatchKind::kPrefill || kind == PhaseDispatchKind::kOverlap)
    {
        int32_t const overlapPrefillBatchSize
            = mConfig.maxOverlapPrefillBatchSize > 0 ? mConfig.maxOverlapPrefillBatchSize : mConfig.maxPrefillBatchSize;
        int32_t const prefillBatchSize
            = kind == PhaseDispatchKind::kOverlap ? overlapPrefillBatchSize : mConfig.maxPrefillBatchSize;
        plan.prefillBatch = popBatch(mPrefillQueue, prefillBatchSize, true, state, plan);
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
    if (plan.kind != PhaseDispatchKind::kNone && mActiveDrainPreference != PhaseDrainPreference::kNone)
    {
        ++mDrainPreferenceDispatches;
    }
    if (drainPreferenceApplied)
    {
        ++mConsecutiveDrainPreferenceDispatches;
        ++mTelemetry.drainPreferenceAppliedDispatches;
    }
    else
    {
        mConsecutiveDrainPreferenceDispatches = 0U;
    }
    mTelemetry.activeDrainPreference = mActiveDrainPreference;
    return plan;
}

void PhaseQueueScheduler::refreshExternalDrainPreference() noexcept
{
    if (!mConfig.enableExternalDrainPreference)
    {
        mRequestedDrainPreference = PhaseDrainPreference::kNone;
        mActiveDrainPreference = PhaseDrainPreference::kNone;
        mDrainPreferenceDispatches = 0U;
        mConsecutiveDrainPreferenceDispatches = 0U;
        mTelemetry.activeDrainPreference = PhaseDrainPreference::kNone;
        return;
    }
    if (mRequestedDrainPreference == mActiveDrainPreference)
    {
        return;
    }
    bool const canTransition = mActiveDrainPreference == PhaseDrainPreference::kNone
        || mDrainPreferenceDispatches >= mConfig.externalDrainPreferenceMinDwellDispatches;
    if (!canTransition)
    {
        return;
    }
    mActiveDrainPreference = mRequestedDrainPreference;
    mDrainPreferenceDispatches = 0U;
    mConsecutiveDrainPreferenceDispatches = 0U;
    ++mTelemetry.drainPreferenceTransitions;
    mTelemetry.activeDrainPreference = mActiveDrainPreference;
}

PhaseDispatchKind PhaseQueueScheduler::applyExternalDrainPreference(
    PhaseQueueSnapshot const& state, PhaseDispatchKind baseline, bool& applied) const noexcept
{
    applied = false;
    if (!mConfig.enableExternalDrainPreference || mActiveDrainPreference == PhaseDrainPreference::kNone
        || state.prefillQueued == 0U || state.decodeQueued == 0U || state.prefillMaxSloPressure >= 1.0
        || state.decodeMaxSloPressure >= 1.0
        || mConsecutiveDrainPreferenceDispatches >= mConfig.externalDrainPreferenceMaxConsecutiveDispatches)
    {
        return baseline;
    }
    if (mActiveDrainPreference == PhaseDrainPreference::kPrefill)
    {
        bool const pagePressureBlocked = mConfig.pagePressureDecodeThreshold > 0.0F && state.pagePoolTotalBundles > 0
            && static_cast<float>(state.pagePoolAllocatedBundles) / static_cast<float>(state.pagePoolTotalBundles)
                >= mConfig.pagePressureDecodeThreshold;
        bool const decodePressureBlocked = mConfig.externalPrefillDrainDecodePressureLimit > 0.0F
            && mTelemetry.recentDecodeTpotPressure >= mConfig.externalPrefillDrainDecodePressureLimit;
        if (pagePressureBlocked || decodePressureBlocked)
        {
            return baseline;
        }
        applied = true;
        return PhaseDispatchKind::kPrefill;
    }
    if (state.consecutiveDecodeBatches >= mConfig.decodeBurstLimit)
    {
        return baseline;
    }
    applied = true;
    return PhaseDispatchKind::kDecode;
}

std::pair<int64_t, int32_t> PhaseQueueScheduler::decodeCandidateShape(int32_t maxRows) const
{
    int64_t total{};
    int32_t maximum{};
    for (PhaseWorkItem const* item : decodeCandidateRows(maxRows))
    {
        total += item->tokenCount;
        maximum = std::max(maximum, item->tokenCount);
    }
    return {total, maximum};
}

std::vector<PhaseWorkItem const*> PhaseQueueScheduler::decodeCandidateRows(int32_t maxRows) const
{
    std::unordered_set<uint64_t> cohort = mDecodeCohortIds;
    if (mConfig.enableDecodeCohortBatching)
    {
        for (PhaseWorkItem const& item : mDecodeQueue)
        {
            if (static_cast<int32_t>(cohort.size()) >= mConfig.maxDecodeBatchSize)
            {
                break;
            }
            if (isEligible(item, false))
            {
                cohort.insert(item.requestId);
            }
        }
    }

    std::vector<PhaseWorkItem const*> selected;
    selected.reserve(static_cast<size_t>(maxRows));
    for (PhaseWorkItem const& item : mDecodeQueue)
    {
        bool const cohortEligible = !mConfig.enableDecodeCohortBatching || cohort.find(item.requestId) != cohort.end();
        if (isEligible(item, false) && cohortEligible)
        {
            selected.push_back(&item);
        }
    }
    if (mConfig.enablePriorityBatching)
    {
        auto const now = std::chrono::steady_clock::now();
        auto priorityRank = [&](PhaseWorkItem const* item) {
            auto const timestamp = mQueuedSince.find(item->requestId);
            check::check(timestamp != mQueuedSince.end(), "Prioritized decode request has no queue timestamp");
            double const waitUs = std::chrono::duration<double, std::micro>(now - timestamp->second).count();
            return static_cast<double>(item->scheduling.priority) + waitUs / mConfig.priorityAgingUs;
        };
        std::stable_sort(selected.begin(), selected.end(), [&](PhaseWorkItem const* lhs, PhaseWorkItem const* rhs) {
            double const lhsRank = priorityRank(lhs);
            double const rhsRank = priorityRank(rhs);
            if (lhsRank != rhsRank)
            {
                return lhsRank > rhsRank;
            }
            return mQueuedSince.at(lhs->requestId) < mQueuedSince.at(rhs->requestId);
        });
    }
    if (static_cast<int32_t>(selected.size()) > maxRows)
    {
        selected.resize(static_cast<size_t>(maxRows));
    }
    return selected;
}

int32_t PhaseQueueScheduler::decodeCandidateReplacementRows(int32_t maxRows) const
{
    std::vector<uint64_t> selected;
    for (PhaseWorkItem const* item : decodeCandidateRows(maxRows))
    {
        selected.push_back(item->requestId);
    }
    return static_cast<int32_t>(phaseDecodeReplacementRows(selected, mPreviousDecodeSelectionIds));
}

int32_t PhaseQueueScheduler::selectDecodeBatchSize(PhaseQueueSnapshot const& state) const
{
    int32_t const available = std::min<int32_t>(mConfig.maxDecodeBatchSize, static_cast<int32_t>(state.decodeQueued));
    if (available <= 1 || !mConfig.enableDynamicDecodeBatching || mConfig.decodeBatchCosts.empty())
    {
        return available;
    }

    struct Candidate
    {
        int32_t batchSize{};
        float p95GpuMs{};
        int32_t contextLimit{};
        int64_t totalContextLimit{};
        int32_t profiledBatchLimit{};
    };
    std::vector<int64_t> contextTotals(static_cast<size_t>(available) + 1U);
    std::vector<int32_t> maxContextLengths(static_cast<size_t>(available) + 1U);
    int32_t runnableRows{};
    for (PhaseWorkItem const* item : decodeCandidateRows(available))
    {
        ++runnableRows;
        contextTotals[static_cast<size_t>(runnableRows)]
            = contextTotals[static_cast<size_t>(runnableRows - 1)] + item->tokenCount;
        maxContextLengths[static_cast<size_t>(runnableRows)]
            = std::max(maxContextLengths[static_cast<size_t>(runnableRows - 1)], item->tokenCount);
    }
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
        int32_t const dispatchedBatchSize = std::min(cost.batchSize, available);
        int64_t const totalContextTokens = contextTotals[static_cast<size_t>(dispatchedBatchSize)];
        int32_t const maxContextLength = maxContextLengths[static_cast<size_t>(dispatchedBatchSize)];
        int64_t const totalContextLimit = cost.maxTotalContextTokens > 0
            ? cost.maxTotalContextTokens
            : static_cast<int64_t>(cost.batchSize) * cost.maxContextLength;
        if (cost.maxContextLength < maxContextLength || totalContextLimit < totalContextTokens)
        {
            continue;
        }
        auto existing = std::find_if(candidates.begin(), candidates.end(),
            [&](Candidate const& candidate) { return candidate.batchSize == dispatchedBatchSize; });
        if (existing == candidates.end())
        {
            candidates.push_back(
                {dispatchedBatchSize, cost.p95GpuMs, cost.maxContextLength, totalContextLimit, cost.batchSize});
        }
        else if (cost.batchSize < existing->profiledBatchLimit
            || (cost.batchSize == existing->profiledBatchLimit && cost.maxContextLength < existing->contextLimit)
            || (cost.batchSize == existing->profiledBatchLimit && cost.maxContextLength == existing->contextLimit
                && totalContextLimit < existing->totalContextLimit))
        {
            *existing = {dispatchedBatchSize, cost.p95GpuMs, cost.maxContextLength, totalContextLimit, cost.batchSize};
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

    for (Candidate& candidate : candidates)
    {
        if (candidate.profiledBatchLimit <= candidate.batchSize)
        {
            continue;
        }
        Candidate const* lower{};
        for (Candidate const& other : candidates)
        {
            if (other.batchSize >= candidate.batchSize || (lower != nullptr && other.batchSize <= lower->batchSize))
            {
                continue;
            }
            lower = &other;
        }
        if (lower == nullptr)
        {
            continue;
        }
        float const fraction = static_cast<float>(candidate.batchSize - lower->batchSize)
            / static_cast<float>(candidate.profiledBatchLimit - lower->batchSize);
        candidate.p95GpuMs = lower->p95GpuMs + fraction * (candidate.p95GpuMs - lower->p95GpuMs);
    }

    if (mOnlineDecodeCostLearningActive)
    {
        for (Candidate& candidate : candidates)
        {
            std::optional<float> const observed
                = onlineDecodeP95(candidate.batchSize, maxContextLengths[static_cast<size_t>(candidate.batchSize)]);
            if (!observed.has_value())
            {
                continue;
            }
            float const lower = candidate.p95GpuMs * (1.0F - mConfig.onlineDecodeCostMaxAdjustmentRatio);
            float const upper = candidate.p95GpuMs * (1.0F + mConfig.onlineDecodeCostMaxAdjustmentRatio);
            candidate.p95GpuMs = std::clamp(*observed, lower, upper);
        }
    }

    if (mConfig.decodeRowReplacementCostMs > 0.0F)
    {
        for (Candidate& candidate : candidates)
        {
            candidate.p95GpuMs += mConfig.decodeRowReplacementCostMs
                * static_cast<float>(decodeCandidateReplacementRows(candidate.batchSize));
        }
    }

    float const minimumCandidateCostMs
        = std::min_element(candidates.begin(), candidates.end(), [](Candidate const& lhs, Candidate const& rhs) {
              return lhs.p95GpuMs < rhs.p95GpuMs;
          })->p95GpuMs;
    auto estimatedDrainCostMs = [&](Candidate const& candidate) {
        int32_t const remainingRows = available - candidate.batchSize;
        if (remainingRows <= 0)
        {
            return candidate.p95GpuMs;
        }
        float remainderCostMs = std::numeric_limits<float>::max();
        for (Candidate const& remainder : candidates)
        {
            if (remainder.batchSize >= remainingRows)
            {
                remainderCostMs = std::min(remainderCostMs, remainder.p95GpuMs);
            }
        }
        if (!std::isfinite(remainderCostMs) || remainderCostMs == std::numeric_limits<float>::max())
        {
            remainderCostMs = minimumCandidateCostMs;
        }
        return candidate.p95GpuMs + remainderCostMs;
    };
    auto const minimumDrainCandidate
        = std::min_element(candidates.begin(), candidates.end(), [&](Candidate const& lhs, Candidate const& rhs) {
              return estimatedDrainCostMs(lhs) < estimatedDrainCostMs(rhs);
          });
    float const minimumDrainCostMs = estimatedDrainCostMs(*minimumDrainCandidate);
    bool const deadlineInfeasibleAtIdle
        = static_cast<double>(minimumDrainCostMs) * 1000.0 > mConfig.decodeQueueWaitTargetUs;
    bool const urgent
        = state.decodeMaxSloPressure >= mConfig.decodeRecoveryPressureThreshold || deadlineInfeasibleAtIdle;
    double const remainingUs = std::max(0.0, mConfig.decodeQueueWaitTargetUs * (1.0 - state.decodeMaxSloPressure));
    Candidate const* selected{};
    for (Candidate const& candidate : candidates)
    {
        double const drainCostUs = static_cast<double>(estimatedDrainCostMs(candidate)) * 1000.0;
        if (urgent)
        {
            // Account for the extra TensorRT turn needed by a remainder. A
            // slightly cheaper partial batch can otherwise alternate with a
            // singleton forever and make queue drain slower than one dense turn.
            if (selected == nullptr || estimatedDrainCostMs(candidate) < estimatedDrainCostMs(*selected)
                || (estimatedDrainCostMs(candidate) == estimatedDrainCostMs(*selected)
                    && candidate.batchSize > selected->batchSize))
            {
                selected = &candidate;
            }
            continue;
        }
        if (drainCostUs > remainingUs)
        {
            continue;
        }
        if (selected == nullptr || estimatedDrainCostMs(candidate) < estimatedDrainCostMs(*selected)
            || (estimatedDrainCostMs(candidate) == estimatedDrainCostMs(*selected)
                && candidate.batchSize > selected->batchSize))
        {
            selected = &candidate;
        }
    }
    if (selected != nullptr)
    {
        return selected->batchSize;
    }

    // No shape can drain the current queue before its remaining deadline.
    // Minimize total recovery time instead of issuing the shortest first turn
    // and leaving a more expensive remainder behind.
    return minimumDrainCandidate->batchSize;
}

uint64_t PhaseQueueScheduler::onlineDecodeCostKey(int32_t batchSize, int32_t maxContextLength) const noexcept
{
    int32_t const contextBucket = std::max(
        1, (maxContextLength + mConfig.onlineDecodeContextBucketTokens - 1) / mConfig.onlineDecodeContextBucketTokens);
    return (static_cast<uint64_t>(static_cast<uint32_t>(batchSize)) << 32U) | static_cast<uint32_t>(contextBucket);
}

std::optional<float> PhaseQueueScheduler::onlineDecodeP95(int32_t batchSize, int32_t maxContextLength) const
{
    auto const found = mOnlineDecodeGpuMs.find(onlineDecodeCostKey(batchSize, maxContextLength));
    if (found == mOnlineDecodeGpuMs.end() || found->second.size() < mConfig.onlineDecodeCostMinSamples)
    {
        return std::nullopt;
    }
    std::vector<float> ordered(found->second.begin(), found->second.end());
    std::sort(ordered.begin(), ordered.end());
    size_t const p95Index = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(ordered.size()))) - 1U;
    return ordered[p95Index];
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
        mDecodeCohortIds.erase(item.requestId);
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

size_t PhaseQueueScheduler::decodeCohortSize() const noexcept
{
    return mDecodeCohortIds.size();
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
    bool const representativePrefillCostSample = mConfig.adaptivePrefillChunkCandidates.empty()
        || metrics.prefillTokens >= mConfig.adaptivePrefillChunkCandidates.back();
    if (metrics.prefillTokens > 0 && metrics.prefillGpuMs > 0.0F && representativePrefillCostSample)
    {
        updateEwma(mTelemetry.prefillGpuMsPerToken, metrics.prefillGpuMs / static_cast<float>(metrics.prefillTokens));
    }
    if (metrics.decodeContextTokens > 0 && metrics.decodeGpuMs > 0.0F)
    {
        updateEwma(mTelemetry.decodeGpuMsPerContextToken,
            metrics.decodeGpuMs / static_cast<float>(metrics.decodeContextTokens));
    }
    if (mOnlineDecodeCostLearningActive && metrics.kind == PhaseDispatchKind::kDecode && metrics.decodeBatchSize > 0
        && metrics.decodeGpuMs > 0.0F && metrics.plannedDecodeMaxContextLength > 0)
    {
        auto& samples
            = mOnlineDecodeGpuMs[onlineDecodeCostKey(metrics.decodeBatchSize, metrics.plannedDecodeMaxContextLength)];
        samples.push_back(metrics.decodeGpuMs);
        if (samples.size() > mConfig.onlineDecodeCostWindow)
        {
            samples.pop_front();
        }
        ++mTelemetry.onlineDecodeCostSampleCount;
        mTelemetry.onlineDecodeCostBucketCount = mOnlineDecodeGpuMs.size();
    }
    if (metrics.kind == PhaseDispatchKind::kOverlap && metrics.prefillBatchSize > 0 && metrics.decodeBatchSize > 0)
    {
        updateEwma(mTelemetry.overlapRatio, metrics.overlapRatio);
        ++mTelemetry.overlapSampleCount;
    }
    bool const collectDecodeTpot = mConfig.enableTpotHysteresis || mConfig.enableDecodeTpotTelemetry
        || (mConfig.enableAdaptivePrefillChunking && !mConfig.adaptivePrefillChunkCandidates.empty());
    if (collectDecodeTpot && metrics.decodeBatchSize > 0 && metrics.decodeGpuMs > 0.0F)
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
            if (mConfig.enableTpotHysteresis)
            {
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
    }
    mTelemetry.latencySafeFallback = mLatencySafeFallback;
    ++mTelemetry.sampleCount;
    mTelemetry.lastDispatch = metrics;
}

PhaseSchedulerTelemetry const& PhaseQueueScheduler::telemetry() const noexcept
{
    return mTelemetry;
}

void PhaseQueueScheduler::setOnlineDecodeCostLearningActive(bool active) noexcept
{
    mOnlineDecodeCostLearningActive = mConfig.enableOnlineDecodeCostLearning && active;
}

void PhaseQueueScheduler::setExternalDrainPreference(PhaseDrainPreference preference) noexcept
{
    mRequestedDrainPreference = mConfig.enableExternalDrainPreference ? preference : PhaseDrainPreference::kNone;
}

void PhaseQueueScheduler::resetHistory()
{
    check::check(empty() && mActiveRequestIds.empty() && mInFlightRequestIds.empty(),
        "Scheduling history can only be reset while the scheduler is idle");
    mTelemetry = {};
    mRecentDecodeTpotUs.clear();
    mOnlineDecodeGpuMs.clear();
    mLatencySafeFallback = false;
    mConsecutiveDecodeBatches = 0;
    mConsecutiveOverlapBatches = 0;
    mPredictedDecodeDebtUs = 0.0;
    mPrefillCohortIds.clear();
    mPrefillCohortTurns = 0;
    mDecodeCohortIds.clear();
    mPreviousDecodeSelectionIds.clear();
    mRequestedDrainPreference = PhaseDrainPreference::kNone;
    mActiveDrainPreference = PhaseDrainPreference::kNone;
    mDrainPreferenceDispatches = 0U;
    mConsecutiveDrainPreferenceDispatches = 0U;
}

} // namespace rt
} // namespace trt_edgellm
