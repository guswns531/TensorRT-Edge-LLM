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
    , mGlobalScheduler(mConfig.globalSchedulerConfig)
    , mRuntimeCostTracker(mConfig.runtimeCostTracker != nullptr
              ? mConfig.runtimeCostTracker
              : std::make_shared<PhaseRuntimeCostTracker>([&] {
                    PhaseRuntimeCostTrackerConfig trackerConfig;
                    trackerConfig.action = mConfig.globalCostModelConfig;
                    trackerConfig.decodeMinimumSamples = mConfig.decodeComponentMinSamples;
                    trackerConfig.decodeWindowSize = mConfig.decodeComponentWindow;
                    trackerConfig.decodeContextBucketTokens = mConfig.runtimeDecodeContextBucketTokens;
                    return trackerConfig;
                }()))
    , mRecentDecodeTpotUs(std::make_shared<RecentDecodeTpot>())
    , mDecodeComponentObservationActive(mConfig.enableDecodeComponentObservation)
{
    check::check(mConfig.maxPrefillBatchSize > 0, "maxPrefillBatchSize must be positive");
    check::check(
        mConfig.maxExternalPrefillBatchSize >= 0 && mConfig.maxExternalPrefillBatchSize <= mConfig.maxPrefillBatchSize,
        "maxExternalPrefillBatchSize must be zero or no greater than maxPrefillBatchSize");
    check::check(mConfig.maxDecodeBatchSize > 0, "maxDecodeBatchSize must be positive");
    check::check(std::isfinite(mConfig.globalColdPrefillMsPerToken) && mConfig.globalColdPrefillMsPerToken > 0.0F,
        "Global cold-start prefill cost must be finite and positive");
    check::check(std::isfinite(mConfig.globalColdDecodeMs) && mConfig.globalColdDecodeMs > 0.0F,
        "Global cold-start decode cost must be finite and positive");
    check::check(
        std::isfinite(mConfig.globalSafeProbeSlackMultiplier) && mConfig.globalSafeProbeSlackMultiplier >= 0.0F,
        "Global safe-probe slack multiplier must be finite and non-negative");
    check::check(mConfig.globalCalibrationMaxOverlapKeys > 0U, "Global calibration overlap-key limit must be positive");
    check::check(mConfig.maxContinuationPrefillBatchSize >= 0
            && mConfig.maxContinuationPrefillBatchSize <= mConfig.maxPrefillBatchSize,
        "maxContinuationPrefillBatchSize must be zero or no greater than maxPrefillBatchSize");
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
    check::check(!mConfig.enableDynamicPrefillBatching || mConfig.enableMeasuredPrefillBatching
            || !mConfig.prefillBatchCosts.empty(),
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
        mConfig.decodeComponentMinSamples > 0 && mConfig.decodeComponentMinSamples <= mConfig.decodeComponentWindow,
        "Runtime decode cost sample bounds are invalid");
    check::check(mConfig.runtimeDecodeContextBucketTokens > 0, "Runtime decode context bucket must be positive");
    check::check(std::isfinite(mConfig.decodeComponentMaxAdjustmentRatio)
            && mConfig.decodeComponentMaxAdjustmentRatio >= 0.0F && mConfig.decodeComponentMaxAdjustmentRatio < 1.0F,
        "Runtime decode cost adjustment ratio must be finite and in [0, 1)");
    check::check(
        std::isfinite(mConfig.decodeContentionCostMaxMultiplier) && mConfig.decodeContentionCostMaxMultiplier >= 1.0F,
        "Runtime contended decode cost multiplier must be finite and at least one");
    check::check(std::isfinite(mConfig.maxPredictedDecodeDebtUs) && mConfig.maxPredictedDecodeDebtUs >= 0.0,
        "Maximum predicted decode debt must be finite and non-negative");
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
    check::check(std::isfinite(mConfig.globalDecodeTpotTargetUs) && mConfig.globalDecodeTpotTargetUs > 0.0,
        "globalDecodeTpotTargetUs must be finite and positive");
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

PhaseQueueSnapshot PhaseQueueScheduler::snapshot(bool includeReadyDetails) const
{
    PhaseQueueSnapshot result{};
    result.prefillPendingProducerRows = mPendingPrefillProducerRows;
    result.prefillPendingTextProducerRows = mPendingTextPrefillProducerRows;
    result.prefillPendingExternalProducerRows = mPendingExternalPrefillProducerRows;
    result.prefillProducerReadyWaitUs = mPendingPrefillProducerWaitUs;
    result.prefillProducerReadyUncertaintyUs = mPendingPrefillProducerUncertaintyUs;
    result.prefillProducerReadyEventId = mPendingPrefillProducerEventId;
    result.prefillProducerRowsClassified = mPendingPrefillProducerRowsClassified;
    result.consecutiveDecodeBatches = mConsecutiveDecodeBatches;
    bool const hasDecodeWork = std::any_of(mDecodeQueue.begin(), mDecodeQueue.end(),
        [this](PhaseWorkItem const& item) { return isEligible(item, false); });
    int32_t const overlapPrefillBatchSize
        = mConfig.maxOverlapPrefillBatchSize > 0 ? mConfig.maxOverlapPrefillBatchSize : mConfig.maxPrefillBatchSize;
    int32_t candidatePrefillBatchSize = hasDecodeWork ? overlapPrefillBatchSize : mConfig.maxPrefillBatchSize;
    auto const prefillSeed = std::find_if(mPrefillQueue.begin(), mPrefillQueue.end(),
        [this](PhaseWorkItem const& item) { return isEligible(item, true); });
    if (prefillSeed != mPrefillQueue.end())
    {
        candidatePrefillBatchSize = std::min(candidatePrefillBatchSize, prefillBatchLimit(prefillSeed->prefillClass));
        int32_t const bucketTokens = dispatchedPrefillTokens(*prefillSeed);
        bool const bucketInitial = prefillSeed->tokenOffset == 0;
        if (!bucketInitial && mConfig.maxContinuationPrefillBatchSize > 0)
        {
            candidatePrefillBatchSize = std::min(candidatePrefillBatchSize, mConfig.maxContinuationPrefillBatchSize);
        }
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
    result.externalPrefillQueued
        = std::any_of(mPrefillQueue.begin(), mPrefillQueue.end(), [this](PhaseWorkItem const& item) {
              return isEligible(item, true) && item.prefillClass == PhasePrefillClass::kExternal;
          });
    for (PhaseWorkItem const& item : mPrefillQueue)
    {
        if (isEligible(item, true))
        {
            if (includeReadyDetails)
            {
                result.prefillRequestIds.push_back(item.requestId);
                result.prefillTokenCounts.push_back(item.tokenCount);
            }
            result.prefillRemainingTokens += item.tokenCount;
            result.prefillContinuationRows += item.tokenOffset > 0 ? 1 : 0;
        }
    }
    std::vector<PhaseWorkItem const*> const decodeRows = decodeCandidateRows(mConfig.maxDecodeBatchSize);
    result.decodeQueued = decodeRows.size();
    for (PhaseWorkItem const* item : decodeRows)
    {
        if (includeReadyDetails)
        {
            result.decodeRequestIds.push_back(item->requestId);
            result.decodeContextLengths.push_back(item->tokenCount);
        }
        result.decodeCandidateContextTokens += item->tokenCount;
        result.decodeCandidateMaxContextLength = std::max(result.decodeCandidateMaxContextLength, item->tokenCount);
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
    result.decodeMinTpotSlackUs = std::numeric_limits<double>::max();
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
                double const slackUs = target - requestAgeUs;
                if (slackUs < result.prefillMinTtftSlackUs
                    || (slackUs == result.prefillMinTtftSlackUs
                        && (result.prefillMinimumSlackRequestId == 0U
                            || item.requestId < result.prefillMinimumSlackRequestId)))
                {
                    result.prefillMinTtftSlackUs = slackUs;
                    result.prefillMinimumSlackRequestId = item.requestId;
                    result.prefillCriticalPathRemainingTokens = item.tokenCount;
                }
            }
            else
            {
                double const tpotTarget = requestTarget > 0.0 ? requestTarget : mConfig.globalDecodeTpotTargetUs;
                result.decodeMinTpotSlackUs = std::min(result.decodeMinTpotSlackUs, tpotTarget - itemWaitUs);
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
    if (result.decodeQueued == 0)
    {
        result.decodeMinTpotSlackUs = 0.0;
    }
    return result;
}

int32_t PhaseQueueScheduler::prefillBatchLimit(PhasePrefillClass prefillClass) const noexcept
{
    if (prefillClass == PhasePrefillClass::kExternal && mConfig.maxExternalPrefillBatchSize > 0)
    {
        return mConfig.maxExternalPrefillBatchSize;
    }
    return mConfig.maxPrefillBatchSize;
}

PhaseQueueSnapshot PhaseQueueScheduler::queueSnapshot(bool includeReadyDetails) const
{
    return snapshot(includeReadyDetails);
}

PhaseGlobalSchedulerMode PhaseQueueScheduler::globalSchedulerMode() const noexcept
{
    return mConfig.globalSchedulerMode;
}

uint64_t PhaseQueueScheduler::globalPlanSequence() const noexcept
{
    return mGlobalPlanSequence;
}

uint64_t PhaseQueueScheduler::globalSnapshotEpoch() const noexcept
{
    return mGlobalSnapshotEpoch;
}

bool PhaseQueueScheduler::isEligible(PhaseWorkItem const& item, bool prefill) const
{
    return !mDispatchBlocked && !(prefill && mPrefillDispatchBlocked) && !(!prefill && mDecodeDispatchBlocked)
        && (!mConfig.eligibilityPolicy || mConfig.eligibilityPolicy(item, prefill));
}

void PhaseQueueScheduler::setPrefillDispatchBlocked(bool blocked) noexcept
{
    mPrefillDispatchBlocked = blocked;
}

void PhaseQueueScheduler::setDecodeDispatchBlocked(bool blocked) noexcept
{
    mDecodeDispatchBlocked = blocked;
}

void PhaseQueueScheduler::setDispatchBlocked(bool blocked) noexcept
{
    mDispatchBlocked = blocked;
}

void PhaseQueueScheduler::setExternalEncoderActive(bool active) noexcept
{
    mExternalEncoderActive = active;
}

void PhaseQueueScheduler::setPendingPrefillProducerRows(size_t rows) noexcept
{
    mPendingPrefillProducerRows = rows;
    mPendingTextPrefillProducerRows = rows;
    mPendingExternalPrefillProducerRows = rows;
    mPendingPrefillProducerWaitUs = 0.0;
    mPendingPrefillProducerUncertaintyUs = 0.0;
    mPendingPrefillProducerEventId = 0U;
    mPendingPrefillProducerRowsClassified = false;
}

void PhaseQueueScheduler::setPendingPrefillProducerRows(
    size_t textRows, size_t externalRows, double predictedWaitUs, double waitUncertaintyUs, uint64_t eventId) noexcept
{
    mPendingPrefillProducerRows = textRows + externalRows;
    mPendingTextPrefillProducerRows = textRows;
    mPendingExternalPrefillProducerRows = externalRows;
    mPendingPrefillProducerWaitUs = std::max(0.0, predictedWaitUs);
    mPendingPrefillProducerUncertaintyUs = std::max(0.0, waitUncertaintyUs);
    mPendingPrefillProducerEventId = eventId;
    mPendingPrefillProducerRowsClassified = true;
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
    if (candidates.empty())
    {
        return 0;
    }
    int32_t const available = std::min<int32_t>(prefillBatchLimit(candidates.front()->prefillClass), candidates.size());
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
        int32_t usefulTokens{};
        for (int32_t index = 0; index < batchSize; ++index)
        {
            maxPastKV = std::max(maxPastKV, candidates[static_cast<size_t>(index)]->tokenOffset);
            usefulTokens += std::min(dispatchedPrefillTokens(*candidates[static_cast<size_t>(index)]), chunkLength);
        }
        std::optional<float> const measuredGpuMs
            = measuredPrefillP95(batchSize, chunkLength, maxPastKV, prefillClass, usefulTokens);
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
        if ((selected != nullptr || measuredGpuMs.has_value())
            && (!overlap || selectedOverlap != nullptr || !mConfig.requireDirectOverlapCost))
        {
            float const isolatedPrefillMs = measuredGpuMs.has_value() ? *measuredGpuMs : selected->p95GpuMs;
            float const gpuMs = selectedOverlap != nullptr ? selectedOverlap->prefillP95GpuMs : isolatedPrefillMs;
            float const interference = selectedOverlap != nullptr
                ? selectedOverlap->decodeSlowdownP95Ms
                : (overlap && selected != nullptr ? selected->decodeSlowdownP95Ms : isolatedPrefillMs);
            profiled.push_back({batchSize, usefulTokens, gpuMs, interference});
        }
    }
    if (profiled.empty())
    {
        costCoverageMiss = true;
        return overlap && mConfig.enableTpotHardGuard && mConfig.requireDirectOverlapCost ? -1 : 0;
    }
    bool const largestShapeCovered = std::any_of(profiled.begin(), profiled.end(),
        [available](Candidate const& candidate) { return candidate.batchSize == available; });
    if (mConfig.enableMeasuredPrefillBatching && !largestShapeCovered)
    {
        // Partial portable P coverage is not evidence that fragmenting a
        // larger compatible batch is profitable. Preserve the mechanism's
        // largest runnable shape until its direct or static cost is known.
        costCoverageMiss = true;
        return overlap && mConfig.enableTpotHardGuard && mConfig.requireDirectOverlapCost ? -1 : 0;
    }

    double const remainingDecodeUs
        = std::max(0.0, mConfig.decodeQueueWaitTargetUs * (1.0 - state.decodeMaxSloPressure));
    double const allowedInterferenceUs = remainingDecodeUs * mConfig.decodeSlackSafetyFactor;
    bool const prefillRecovery
        = mConfig.enablePrefillSloRecovery && state.prefillMaxSloPressure >= 1.0 && state.decodeMaxSloPressure < 1.0;
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
        auto const lhsQueued = mQueuedSince.at(lhs.requestId);
        auto const rhsQueued = mQueuedSince.at(rhs.requestId);
        if (lhsQueued != rhsQueued)
        {
            return lhsQueued > rhsQueued;
        }
        return lhs.requestId > rhs.requestId;
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
        auto const lhsQueued = mQueuedSince.at(lhs.requestId);
        auto const rhsQueued = mQueuedSince.at(rhs.requestId);
        if (lhsQueued != rhsQueued)
        {
            return lhsQueued < rhsQueued;
        }
        return lhs.requestId < rhs.requestId;
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
    int32_t count = std::min<int32_t>(maxBatchSize,
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
            count = std::min<int32_t>(
                maxBatchSize, std::count_if(queue.begin(), queue.end(), [this](PhaseWorkItem const& item) {
                    return isEligible(item, false) && mDecodeCohortIds.find(item.requestId) != mDecodeCohortIds.end();
                }));
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
    maxBatchSize = std::min(maxBatchSize, prefillBatchLimit(bucketSeed->prefillClass));
    if (!bucketInitial && mConfig.maxContinuationPrefillBatchSize > 0)
    {
        maxBatchSize = std::min(maxBatchSize, mConfig.maxContinuationPrefillBatchSize);
    }
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
    int32_t costLookupRows = std::min<int32_t>(maxBatchSize, compatible.size());
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

std::vector<PhaseWorkItem> PhaseQueueScheduler::popGlobalCandidateBatch(std::deque<PhaseWorkItem>& queue,
    PhaseGlobalActionCandidate const& candidate, bool primary, bool chunkPrefill, PhaseDispatchPlan& plan)
{
    std::vector<uint64_t> const& requestIds = primary ? candidate.primaryRequestIds : candidate.secondaryRequestIds;
    std::vector<int32_t> const& stableSlotIds
        = primary ? candidate.primaryStableSlotIds : candidate.secondaryStableSlotIds;
    check::check(requestIds.size() == stableSlotIds.size(), "Global candidate row and stable-slot counts differ");
    auto const now = std::chrono::steady_clock::now();
    std::vector<PhaseWorkItem> batch;
    batch.reserve(requestIds.size());
    for (size_t index{}; index < requestIds.size(); ++index)
    {
        auto selected = std::find_if(queue.begin(), queue.end(), [&](PhaseWorkItem const& item) {
            return item.requestId == requestIds[index] && item.kvSlotId == stableSlotIds[index];
        });
        check::check(selected != queue.end(), "Globally leased request disappeared before dispatch");
        check::check(isEligible(*selected, chunkPrefill), "Globally leased request became ineligible before dispatch");
        PhaseWorkItem item = *selected;
        if (chunkPrefill)
        {
            item.tokenCount = std::min(item.tokenCount, candidate.key.chunkLength);
        }
        queue.erase(selected);
        check::check(mInFlightRequestIds.insert(item.requestId).second, "Request is already in flight");
        auto const queued = mQueuedSince.find(item.requestId);
        check::check(queued != mQueuedSince.end(), "Globally leased request has no queue timestamp");
        double& queueWaitUs = chunkPrefill ? plan.prefillQueueWaitUs : plan.decodeQueueWaitUs;
        queueWaitUs = std::max(queueWaitUs, std::chrono::duration<double, std::micro>(now - queued->second).count());
        mQueuedSince.erase(queued);
        batch.push_back(std::move(item));
    }
    if (chunkPrefill)
    {
        if (mConfig.enableWavefrontPrefillBatching)
        {
            if (mPrefillCohortIds.empty())
            {
                mPrefillCohortIds.insert(requestIds.begin(), requestIds.end());
            }
            ++mPrefillCohortTurns;
            plan.prefillCohortSize = static_cast<int32_t>(mPrefillCohortIds.size());
        }
        plan.prefillCostLookupRows = candidate.key.primaryBatchSize;
        plan.prefillCostLookupChunkLength = candidate.key.chunkLength;
    }
    else
    {
        if (mConfig.enableDecodeCohortBatching)
        {
            mDecodeCohortIds.insert(requestIds.begin(), requestIds.end());
        }
        mPreviousDecodeSelectionIds = requestIds;
    }
    return batch;
}

PhaseDispatchPlan PhaseQueueScheduler::previewMechanismPlan(PhaseDispatchKind kind) const
{
    PhaseQueueScheduler preview = *this;
    preview.mConfig.globalSchedulerMode = PhaseGlobalSchedulerMode::kDisabled;
    preview.mConfig.enablePrefillTtftHardGuard = false;
    preview.mConfig.enableMetricsPolicy = false;
    preview.mConfig.metricsPolicy = {};
    preview.mConfig.enableExternalDrainPreference = false;
    preview.mConfig.policy = [kind](PhaseQueueSnapshot const&) { return kind; };
    preview.mNextGlobalAction.reset();
    return preview.next();
}

std::optional<PhaseQueueScheduler::GlobalQueueSelection> PhaseQueueScheduler::selectGlobalQueueAction(
    PhaseQueueSnapshot const& state, bool allowPrefill, bool allowDecode, bool allowOverlap,
    std::optional<PhaseDispatchKind> requiredKind, PhaseGlobalSelectionAudit* audit)
{
    if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kDisabled)
    {
        return std::nullopt;
    }

    struct Prediction
    {
        double makespanUs{};
        double uncertaintyUs{};
        double referenceWorkUs{};
        bool directlyKnown{};
    };
    auto fromOnline = [&](PhaseGlobalActionKey const& key) -> std::optional<Prediction> {
        std::optional<PhaseGlobalCostEstimate> const estimate = mRuntimeCostTracker->estimate(key);
        if (!estimate.has_value())
        {
            return std::nullopt;
        }
        return Prediction{static_cast<double>(estimate->makespanMedianMs) * 1000.0,
            static_cast<double>(estimate->uncertaintyMs) * 1000.0,
            static_cast<double>(estimate->referenceWorkMedianMs) * 1000.0, true};
    };
    int32_t const contextBucketTokens = std::max(1, mConfig.runtimeDecodeContextBucketTokens);
    auto contextBucket = [contextBucketTokens](int32_t tokens) {
        return (std::max(0, tokens) + contextBucketTokens - 1) / contextBucketTokens;
    };

    bool const prefillDeadlineExpired
        = mConfig.enablePrefillTtftHardGuard && state.prefillQueued > 0U && state.prefillMinTtftSlackUs <= 0.0;
    PhaseDispatchPlan const prefillPlan
        = state.prefillQueued > 0U ? previewMechanismPlan(PhaseDispatchKind::kPrefill) : PhaseDispatchPlan{};
    int32_t const prefillRows = static_cast<int32_t>(prefillPlan.prefillBatch.size());
    int32_t prefillChunk{};
    int32_t prefillPastKV{};
    int32_t prefillUsefulTokens{};
    bool prefillInitial{};
    bool prefillAllFinal{!prefillPlan.prefillBatch.empty()};
    PhasePrefillClass prefillClass{PhasePrefillClass::kAny};
    std::vector<uint64_t> prefillRequestIds;
    std::vector<int32_t> prefillStableSlotIds;
    for (PhaseWorkItem const& item : prefillPlan.prefillBatch)
    {
        prefillChunk = std::max(prefillChunk, item.tokenCount);
        prefillPastKV = std::max(prefillPastKV, item.tokenOffset);
        prefillUsefulTokens += item.tokenCount;
        prefillAllFinal = prefillAllFinal && item.tokenOffset + item.tokenCount == item.promptTokenCount;
        prefillRequestIds.push_back(item.requestId);
        prefillStableSlotIds.push_back(item.kvSlotId);
    }
    if (!prefillPlan.prefillBatch.empty())
    {
        prefillInitial = prefillPlan.prefillBatch.front().tokenOffset == 0;
        prefillClass = prefillPlan.prefillBatch.front().prefillClass;
    }
    PhaseDispatchPlan const decodePlan
        = state.decodeQueued > 0U ? previewMechanismPlan(PhaseDispatchKind::kDecode) : PhaseDispatchPlan{};
    int32_t const decodeRows = static_cast<int32_t>(decodePlan.decodeBatch.size());
    int64_t decodeContextTokens{};
    int32_t decodeMaxContext{};
    std::vector<uint64_t> decodeRequestIds;
    std::vector<int32_t> decodeStableSlotIds;
    for (PhaseWorkItem const& item : decodePlan.decodeBatch)
    {
        decodeContextTokens += item.tokenCount;
        decodeMaxContext = std::max(decodeMaxContext, item.tokenCount);
        decodeRequestIds.push_back(item.requestId);
        decodeStableSlotIds.push_back(item.kvSlotId);
    }

    PhaseGlobalActionKey prefillKey{
        PhaseGlobalActionKind::kPrefill, prefillRows, 0, prefillChunk, contextBucket(prefillPastKV), 0};
    prefillKey.primaryWorkClass = static_cast<int32_t>(prefillClass);
    PhaseGlobalActionKey decodeKey{
        PhaseGlobalActionKind::kDecode, decodeRows, 0, 1, contextBucket(decodeMaxContext), 0};

    PhaseDispatchPlan const overlapPlan = state.prefillQueued > 0U && state.decodeQueued > 0U
        ? previewMechanismPlan(PhaseDispatchKind::kOverlap)
        : PhaseDispatchPlan{};
    int32_t const overlapPrefillRows = static_cast<int32_t>(overlapPlan.prefillBatch.size());
    int32_t const overlapDecodeRows = static_cast<int32_t>(overlapPlan.decodeBatch.size());
    int32_t overlapPrefillChunk{};
    int32_t overlapPrefillPastKV{};
    int32_t overlapDecodeMaxContext{};
    int32_t overlapPrefillUsefulTokens{};
    bool overlapPrefillInitial{};
    PhasePrefillClass overlapPrefillClass{PhasePrefillClass::kAny};
    std::vector<uint64_t> overlapPrefillRequestIds;
    std::vector<uint64_t> overlapDecodeRequestIds;
    std::vector<int32_t> overlapPrefillStableSlotIds;
    std::vector<int32_t> overlapDecodeStableSlotIds;
    for (PhaseWorkItem const& item : overlapPlan.prefillBatch)
    {
        overlapPrefillChunk = std::max(overlapPrefillChunk, item.tokenCount);
        overlapPrefillPastKV = std::max(overlapPrefillPastKV, item.tokenOffset);
        overlapPrefillUsefulTokens += item.tokenCount;
        overlapPrefillRequestIds.push_back(item.requestId);
        overlapPrefillStableSlotIds.push_back(item.kvSlotId);
    }
    for (PhaseWorkItem const& item : overlapPlan.decodeBatch)
    {
        overlapDecodeMaxContext = std::max(overlapDecodeMaxContext, item.tokenCount);
        overlapDecodeRequestIds.push_back(item.requestId);
        overlapDecodeStableSlotIds.push_back(item.kvSlotId);
    }
    if (!overlapPlan.prefillBatch.empty())
    {
        overlapPrefillInitial = overlapPlan.prefillBatch.front().tokenOffset == 0;
        overlapPrefillClass = overlapPlan.prefillBatch.front().prefillClass;
    }

    auto executionVariant = [&](PhaseGlobalActionKey const& key, int32_t primaryTokens) {
        return mGlobalExecutionVariantSupplier && key.primaryBatchSize > 0
            ? mGlobalExecutionVariantSupplier(key, primaryTokens)
            : PhaseExecutionVariant::kEager;
    };
    prefillKey.executionVariant = executionVariant(prefillKey, prefillUsefulTokens);
    decodeKey.executionVariant = executionVariant(decodeKey, 0);

    auto predictPrefill = [&](PhaseGlobalActionKey const& key, int32_t rows, int32_t chunk, int32_t pastKV,
                              bool initial, PhasePrefillClass phaseClass, int32_t candidateTokens,
                              bool allowBatchInterpolation = false) -> Prediction {
        if (std::optional<Prediction> const online = fromOnline(key))
        {
            return *online;
        }
        if (allowBatchInterpolation)
        {
            std::optional<PhaseGlobalCostEstimate> const interpolated
                = mRuntimeCostTracker->estimateInterpolatedPrimaryBatch(key);
            if (interpolated.has_value())
            {
                return {static_cast<double>(interpolated->makespanMedianMs) * 1000.0,
                    static_cast<double>(interpolated->uncertaintyMs) * 1000.0,
                    static_cast<double>(interpolated->referenceWorkMedianMs) * 1000.0, true};
            }
        }
        if (std::optional<PhaseGlobalCostEstimate> const covering = mRuntimeCostTracker->estimateCoveringPrimary(key))
        {
            return {static_cast<double>(covering->makespanMedianMs) * 1000.0,
                static_cast<double>(covering->uncertaintyMs) * 1000.0,
                static_cast<double>(covering->referenceWorkMedianMs) * 1000.0, true};
        }
        PhasePrefillBatchCost const* selected{};
        PhasePrefillBatchCost const* singleton{};
        for (PhasePrefillBatchCost const& cost : mConfig.prefillBatchCosts)
        {
            bool const classCompatible
                = cost.prefillClass == PhasePrefillClass::kAny || cost.prefillClass == phaseClass;
            if (!classCompatible || cost.initialChunk != initial || cost.chunkLength < chunk
                || cost.maxPastKVLength < pastKV || cost.maxConcurrentDecodeBatchSize != 0)
            {
                continue;
            }
            if (cost.batchSize >= rows
                && (selected == nullptr || cost.batchSize < selected->batchSize
                    || (cost.batchSize == selected->batchSize && cost.p95GpuMs < selected->p95GpuMs)))
            {
                selected = &cost;
            }
            if (cost.batchSize == 1
                && (singleton == nullptr || cost.chunkLength < singleton->chunkLength
                    || (cost.chunkLength == singleton->chunkLength && cost.p95GpuMs < singleton->p95GpuMs)))
            {
                singleton = &cost;
            }
        }
        double const fallbackMs = std::max(0.001,
            static_cast<double>(mConfig.globalColdPrefillMsPerToken)
                * static_cast<double>(std::max(1, candidateTokens)));
        std::optional<PhaseGlobalCostEstimate> const launchFloor = mRuntimeCostTracker->estimatePrimaryLaunchFloor(key);
        double const launchFloorMs = launchFloor.has_value() ? launchFloor->makespanMedianMs : 0.0;
        double const makespanMs = std::max(selected != nullptr ? selected->p95GpuMs : fallbackMs, launchFloorMs);
        double const referenceMs = singleton != nullptr ? static_cast<double>(singleton->p95GpuMs) * rows : makespanMs;
        double const uncertaintyMs = launchFloor.has_value() ? launchFloor->uncertaintyMs
            : selected != nullptr                            ? 0.0
                                                             : mConfig.globalCostModelConfig.coldStartUncertaintyMs;
        return {makespanMs * 1000.0, uncertaintyMs * 1000.0, referenceMs * 1000.0, selected != nullptr};
    };
    auto predictDecode
        = [&](PhaseGlobalActionKey const& key, int32_t rows, int32_t maxContext, int64_t contextTokens) -> Prediction {
        if (std::optional<Prediction> const online = fromOnline(key))
        {
            return *online;
        }
        PhaseDecodeBatchCost const* selected{};
        PhaseDecodeBatchCost const* singleton{};
        for (PhaseDecodeBatchCost const& cost : mConfig.decodeBatchCosts)
        {
            if (cost.batchSize == 1 && cost.maxContextLength >= maxContext
                && (singleton == nullptr || cost.maxContextLength < singleton->maxContextLength
                    || (cost.maxContextLength == singleton->maxContextLength && cost.p95GpuMs < singleton->p95GpuMs)))
            {
                singleton = &cost;
            }
            int64_t const totalLimit = cost.maxTotalContextTokens > 0
                ? cost.maxTotalContextTokens
                : static_cast<int64_t>(cost.batchSize) * cost.maxContextLength;
            if (cost.maxContextLength < maxContext || totalLimit < contextTokens)
            {
                continue;
            }
            if (cost.batchSize >= rows
                && (selected == nullptr || cost.batchSize < selected->batchSize
                    || (cost.batchSize == selected->batchSize && cost.p95GpuMs < selected->p95GpuMs)))
            {
                selected = &cost;
            }
        }
        double const makespanMs = selected != nullptr ? selected->p95GpuMs : mConfig.globalColdDecodeMs;
        double const referenceMs = singleton != nullptr ? static_cast<double>(singleton->p95GpuMs) * rows : makespanMs;
        double const uncertaintyMs = selected != nullptr ? 0.0 : mConfig.globalCostModelConfig.coldStartUncertaintyMs;
        return {makespanMs * 1000.0, uncertaintyMs * 1000.0, referenceMs * 1000.0, selected != nullptr};
    };
    auto predictDecodeSuccessor
        = [&](PhaseGlobalActionKey const& key, int32_t rows, int32_t maxContext, int64_t contextTokens) {
              Prediction prediction = predictDecode(key, rows, maxContext, contextTokens);
              if (prediction.directlyKnown)
              {
                  return prediction;
              }
              std::optional<PhaseGlobalCostEstimate> const covering
                  = mRuntimeCostTracker->trustedEstimatePrimaryBatchCoveringContext(key);
              if (!covering.has_value())
              {
                  return prediction;
              }
              return Prediction{static_cast<double>(covering->makespanMedianMs) * 1000.0,
                  static_cast<double>(covering->uncertaintyMs) * 1000.0,
                  static_cast<double>(covering->referenceWorkMedianMs) * 1000.0, true};
          };

    std::optional<Prediction> const prefill = prefillRows > 0
        ? std::optional<Prediction>(predictPrefill(
              prefillKey, prefillRows, prefillChunk, prefillPastKV, prefillInitial, prefillClass, prefillUsefulTokens))
        : std::nullopt;
    std::optional<Prediction> const decode = decodeRows > 0
        ? std::optional<Prediction>(predictDecode(decodeKey, decodeRows, decodeMaxContext, decodeContextTokens))
        : std::nullopt;
    std::optional<Prediction> overlapPrefill;
    std::optional<Prediction> overlapDecode;
    if (overlapPrefillRows > 0 && overlapDecodeRows > 0)
    {
        PhaseGlobalActionKey overlapPrefillKey{PhaseGlobalActionKind::kPrefill, overlapPrefillRows, 0,
            overlapPrefillChunk, contextBucket(overlapPrefillPastKV), 0};
        overlapPrefillKey.primaryWorkClass = static_cast<int32_t>(overlapPrefillClass);
        overlapPrefillKey.executionVariant = executionVariant(overlapPrefillKey, overlapPrefillUsefulTokens);
        overlapPrefill = predictPrefill(overlapPrefillKey, overlapPrefillRows, overlapPrefillChunk,
            overlapPrefillPastKV, overlapPrefillInitial, overlapPrefillClass, overlapPrefillUsefulTokens);
        PhaseGlobalActionKey overlapDecodeKey{
            PhaseGlobalActionKind::kDecode, overlapDecodeRows, 0, 1, contextBucket(overlapDecodeMaxContext), 0};
        overlapDecodeKey.executionVariant = executionVariant(overlapDecodeKey, 0);
        int64_t overlapDecodeContextTokens{};
        for (PhaseWorkItem const& item : overlapPlan.decodeBatch)
        {
            overlapDecodeContextTokens += item.tokenCount;
        }
        overlapDecode
            = predictDecode(overlapDecodeKey, overlapDecodeRows, overlapDecodeMaxContext, overlapDecodeContextTokens);
    }
    struct DecodeFormationPrediction
    {
        Prediction combined;
        Prediction newlyProduced;
    };
    std::optional<DecodeFormationPrediction> decodeFormation;
    if (mConfig.enableDecodeFormationHorizon && prefill.has_value() && decode.has_value() && prefillAllFinal
        && decodeRows < mConfig.maxDecodeBatchSize)
    {
        ++mTelemetry.globalDecodeFormationSnapshotCount;
        int32_t combinedRows = decodeRows;
        int64_t combinedContextTokens = decodeContextTokens;
        int32_t combinedMaxContext = decodeMaxContext;
        int32_t producedRows{};
        int64_t producedContextTokens{};
        int32_t producedMaxContext{};
        for (PhaseWorkItem const& item : prefillPlan.prefillBatch)
        {
            if (combinedRows >= mConfig.maxDecodeBatchSize)
            {
                break;
            }
            int32_t const resultingContext
                = item.promptTokenCount > 0 ? item.promptTokenCount : item.tokenOffset + item.tokenCount;
            ++combinedRows;
            combinedContextTokens += resultingContext;
            combinedMaxContext = std::max(combinedMaxContext, resultingContext);
            ++producedRows;
            producedContextTokens += resultingContext;
            producedMaxContext = std::max(producedMaxContext, resultingContext);
        }
        if (producedRows > 0)
        {
            PhaseGlobalActionKey combinedKey{
                PhaseGlobalActionKind::kDecode, combinedRows, 0, 1, contextBucket(combinedMaxContext), 0};
            combinedKey.executionVariant = executionVariant(combinedKey, 0);
            PhaseGlobalActionKey producedKey{
                PhaseGlobalActionKind::kDecode, producedRows, 0, 1, contextBucket(producedMaxContext), 0};
            producedKey.executionVariant = executionVariant(producedKey, 0);
            Prediction const combined
                = predictDecodeSuccessor(combinedKey, combinedRows, combinedMaxContext, combinedContextTokens);
            Prediction const produced
                = predictDecodeSuccessor(producedKey, producedRows, producedMaxContext, producedContextTokens);
            mTelemetry.globalDecodeFormationCombinedCostHitCount += combined.directlyKnown ? 1U : 0U;
            mTelemetry.globalDecodeFormationProducedCostHitCount += produced.directlyKnown ? 1U : 0U;
            // The bounded transition is used only when both successor shapes
            // have measured coverage. A cold process must not manufacture a
            // formation benefit from fallback costs.
            if (combined.directlyKnown && produced.directlyKnown)
            {
                decodeFormation = DecodeFormationPrediction{combined, produced};
                ++mTelemetry.globalDecodeFormationOpportunityCount;
                mTelemetry.globalDecodeFormationMaxProducedRows
                    = std::max(mTelemetry.globalDecodeFormationMaxProducedRows, static_cast<size_t>(producedRows));
            }
        }
    }
    struct PrefillFormationPrediction
    {
        Prediction combined;
        Prediction residual;
        double producerWaitUs{};
        double producerUncertaintyUs{};
    };
    std::optional<PrefillFormationPrediction> prefillFormation;
    size_t compatibleProducerRows = state.prefillPendingProducerRows;
    if (state.prefillProducerRowsClassified)
    {
        compatibleProducerRows = prefillClass == PhasePrefillClass::kExternal ? state.prefillPendingExternalProducerRows
                                                                              : state.prefillPendingTextProducerRows;
    }
    if (prefill.has_value() && decode.has_value() && compatibleProducerRows > 0U)
    {
        ++mTelemetry.globalPrefillFormationProducerSnapshotCount;
        mTelemetry.globalPrefillFormationMaxPendingRows
            = std::max(mTelemetry.globalPrefillFormationMaxPendingRows, compatibleProducerRows);
        int32_t formationCapacityRows = prefillBatchLimit(prefillClass);
        if (mConfig.maxPrefillBatchTokens > 0)
        {
            int32_t const tokenCapacityRows = std::max(1, mConfig.maxPrefillBatchTokens / std::max(1, prefillChunk));
            formationCapacityRows = std::min(formationCapacityRows, tokenCapacityRows);
        }
        size_t const boundedPendingRows
            = std::min(compatibleProducerRows, static_cast<size_t>(std::numeric_limits<int32_t>::max()));
        int32_t const pendingRows = static_cast<int32_t>(boundedPendingRows);
        int32_t const combinedRows = std::min(formationCapacityRows, prefillRows + pendingRows);
        int32_t const residualRows = combinedRows - prefillRows;
        if (residualRows > 0)
        {
            PhaseGlobalActionKey combinedKey = prefillKey;
            combinedKey.primaryBatchSize = combinedRows;
            int32_t const combinedTokens = prefillUsefulTokens + residualRows * std::max(1, prefillChunk);
            combinedKey.executionVariant = executionVariant(combinedKey, combinedTokens);
            Prediction const combined = predictPrefill(combinedKey, combinedRows, prefillChunk, prefillPastKV,
                prefillInitial, prefillClass, combinedTokens, true);
            mTelemetry.globalPrefillFormationCombinedCostHitCount += combined.directlyKnown ? 1U : 0U;

            PhaseGlobalActionKey residualKey = prefillKey;
            residualKey.primaryBatchSize = residualRows;
            int32_t const residualTokens = residualRows * std::max(1, prefillChunk);
            residualKey.executionVariant = executionVariant(residualKey, residualTokens);
            Prediction const residual = predictPrefill(residualKey, residualRows, prefillChunk, prefillPastKV,
                prefillInitial, prefillClass, residualTokens, true);
            mTelemetry.globalPrefillFormationResidualCostHitCount += residual.directlyKnown ? 1U : 0U;
            if (combined.directlyKnown && residual.directlyKnown)
            {
                prefillFormation = PrefillFormationPrediction{
                    combined, residual, state.prefillProducerReadyWaitUs, state.prefillProducerReadyUncertaintyUs};
                ++mTelemetry.globalPrefillFormationOpportunityCount;
            }
        }
    }
    double const prefillSlack
        = state.prefillQueued > 0U ? state.prefillMinTtftSlackUs : std::numeric_limits<double>::infinity();
    // Queue formation latency controls when a D cohort is released; it is not
    // the request's next-token deadline. Global E/P/D selection must protect
    // the explicit per-request TPOT slack summarized in the snapshot, just as
    // the bounded WAIT comparison below does.
    double const decodeSlack
        = state.decodeQueued > 0U ? state.decodeMinTpotSlackUs : std::numeric_limits<double>::infinity();

    auto memoryFor = [&](PhaseGlobalActionKey const& key, std::vector<uint64_t> const& requestIds) {
        return mConfig.globalMemoryHorizonSupplier ? mConfig.globalMemoryHorizonSupplier(key, requestIds)
                                                   : PhaseActionMemoryHorizon{};
    };
    auto protectedPrefillAdvance = [&](std::vector<PhaseWorkItem> const& batch) {
        auto const selected = std::find_if(batch.begin(), batch.end(),
            [&](PhaseWorkItem const& item) { return item.requestId == state.prefillMinimumSlackRequestId; });
        return selected != batch.end() ? selected->tokenCount : 0;
    };
    auto protectedPrefillCompletion = [&](Prediction const& action, int32_t advancedTokens) {
        int32_t const remainingTokens
            = std::max(0, state.prefillCriticalPathRemainingTokens - std::max(0, advancedTokens));
        int32_t const futureChunkTokens = std::max(1, mConfig.maxPrefillChunkTokens);
        int32_t const residualTurns = (remainingTokens + futureChunkTokens - 1) / futureChunkTokens;
        return PhaseProtectedCompletion{prefillSlack,
            action.makespanUs + static_cast<double>(residualTurns) * prefill->makespanUs,
            action.uncertaintyUs + static_cast<double>(residualTurns) * prefill->uncertaintyUs,
            PhaseProtectedKind::kPrefill};
    };
    auto protect = [&](PhaseGlobalActionCandidate& candidate, int32_t advancedPrefillTokens, Prediction const& action) {
        if (prefill.has_value())
        {
            candidate.protectedCompletions.push_back(protectedPrefillCompletion(action, advancedPrefillTokens));
        }
        if (decode.has_value())
        {
            bool const advancesDecode = candidate.key.kind == PhaseGlobalActionKind::kDecode
                || candidate.key.kind == PhaseGlobalActionKind::kPrefillDecode;
            double const completion = action.makespanUs + (advancesDecode ? 0.0 : decode->makespanUs);
            double const uncertainty = action.uncertaintyUs + (advancesDecode ? 0.0 : decode->uncertaintyUs);
            candidate.protectedCompletions.push_back(
                {decodeSlack, completion, uncertainty, PhaseProtectedKind::kDecode});
        }
    };
    std::vector<PhaseGlobalActionCandidate> candidates;
    if (allowPrefill && prefill.has_value())
    {
        PhaseGlobalActionCandidate candidate;
        candidate.key = prefillKey;
        candidate.primaryRequestIds = prefillRequestIds;
        candidate.primaryStableSlotIds = prefillStableSlotIds;
        phaseGlobalFinalizeCandidate(candidate);
        candidate.predictedBlockingUs = prefill->makespanUs;
        candidate.predictedMakespanUs = prefill->makespanUs;
        candidate.uncertaintyUs = prefill->uncertaintyUs;
        candidate.referenceWorkUs = prefill->referenceWorkUs;
        candidate.requestServiceLagUs = state.prefillOldestRequestAgeUs;
        candidate.memory = memoryFor(candidate.key, candidate.requestIds);
        protect(candidate, protectedPrefillAdvance(prefillPlan.prefillBatch), *prefill);
        candidates.push_back(std::move(candidate));
    }
    bool const preserveExpiredDecode
        = mConfig.preserveExpiredDecodeCandidate && state.decodeQueued > 0U && state.decodeMinTpotSlackUs <= 0.0;
    if (allowDecode && decode.has_value() && (!prefillDeadlineExpired || preserveExpiredDecode))
    {
        PhaseGlobalActionCandidate candidate;
        candidate.key = decodeKey;
        candidate.primaryRequestIds = decodeRequestIds;
        candidate.primaryStableSlotIds = decodeStableSlotIds;
        phaseGlobalFinalizeCandidate(candidate);
        candidate.predictedBlockingUs = decode->makespanUs;
        candidate.predictedMakespanUs = decode->makespanUs;
        candidate.uncertaintyUs = decode->uncertaintyUs;
        candidate.referenceWorkUs = decode->referenceWorkUs;
        candidate.requestServiceLagUs = state.decodeOldestWaitUs;
        candidate.memory = memoryFor(candidate.key, candidate.requestIds);
        protect(candidate, 0, *decode);
        candidates.push_back(std::move(candidate));
    }
    if (allowPrefill && allowDecode && prefill.has_value() && decode.has_value())
    {
        // Compare serial P and D choices over the same bounded amount of work.
        // One-action compression is not comparable when the alternatives
        // advance different phases: it can prefer a locally efficient small
        // launch even though that choice fragments the other phase's next
        // cohort. The common two-action horizon changes only ranking; request
        // rows, deadline protection, and the dispatched action remain exact.
        double prefillFirstHorizonUs = prefill->makespanUs + decode->makespanUs;
        double decodeFirstHorizonUs = prefillFirstHorizonUs;
        double serialReferenceWorkUs = prefill->makespanUs + decode->makespanUs;
        if (prefillFormation.has_value())
        {
            // A known host-side producer can make more compatible rows ready
            // while D is useful GPU work. Compare the same rows in both
            // orders: P-now + D + residual-P versus D + combined-P. Unknown
            // shapes retain the ordinary two-action horizon.
            prefillFirstHorizonUs = std::max(prefillFirstHorizonUs, prefillFormation->producerWaitUs)
                + prefillFormation->residual.makespanUs;
            decodeFirstHorizonUs = std::max(decode->makespanUs, prefillFormation->producerWaitUs)
                + prefillFormation->combined.makespanUs;
            serialReferenceWorkUs = decode->makespanUs
                + std::max(
                    prefill->makespanUs + prefillFormation->residual.makespanUs, prefillFormation->combined.makespanUs);
        }
        if (decodeFormation.has_value())
        {
            // Compare one observable P -> D transition over equal useful
            // work. Completing final P rows makes them decode-ready without
            // predicting a future arrival. P-first can consume the incumbent
            // and newly produced rows in one combined D launch; D-first must
            // still service the produced rows after P completes.
            prefillFirstHorizonUs = prefill->makespanUs + decodeFormation->combined.makespanUs;
            decodeFirstHorizonUs = decode->makespanUs + prefill->makespanUs + decodeFormation->newlyProduced.makespanUs;
            serialReferenceWorkUs
                = prefill->makespanUs + decode->makespanUs + decodeFormation->newlyProduced.makespanUs;
        }
        for (PhaseGlobalActionCandidate& candidate : candidates)
        {
            if (candidate.key.kind == PhaseGlobalActionKind::kPrefill)
            {
                candidate.predictedHorizonUs = prefillFirstHorizonUs;
                candidate.horizonReferenceWorkUs = serialReferenceWorkUs;
            }
            else if (candidate.key.kind == PhaseGlobalActionKind::kDecode)
            {
                candidate.predictedHorizonUs = decodeFirstHorizonUs;
                candidate.horizonReferenceWorkUs = serialReferenceWorkUs;
                if (prefillFormation.has_value() && !candidate.protectedCompletions.empty())
                {
                    int32_t const futureChunkTokens = std::max(1, mConfig.maxPrefillChunkTokens);
                    int32_t const residualTokens
                        = std::max(0, state.prefillCriticalPathRemainingTokens - futureChunkTokens);
                    int32_t const residualTurns = (residualTokens + futureChunkTokens - 1) / futureChunkTokens;
                    candidate.protectedCompletions.front().predictedCompletionUs = decode->makespanUs
                        + prefillFormation->combined.makespanUs
                        + static_cast<double>(residualTurns) * prefill->makespanUs;
                    candidate.protectedCompletions.front().uncertaintyUs = decode->uncertaintyUs
                        + prefillFormation->combined.uncertaintyUs + prefillFormation->producerUncertaintyUs
                        + static_cast<double>(residualTurns) * prefill->uncertaintyUs;
                }
            }
        }
    }
    if (allowOverlap && prefill.has_value() && decode.has_value() && overlapPrefill.has_value()
        && overlapDecode.has_value())
    {
        ++mTelemetry.globalOverlapOpportunityCount;
        PhaseGlobalActionKey overlapKey{PhaseGlobalActionKind::kPrefillDecode, overlapPrefillRows, overlapDecodeRows,
            overlapPrefillChunk, contextBucket(overlapPrefillPastKV), contextBucket(overlapDecodeMaxContext)};
        overlapKey.primaryWorkClass = static_cast<int32_t>(overlapPrefillClass);
        overlapKey.executionVariant = executionVariant(overlapKey, overlapPrefillUsefulTokens);
        Prediction overlap{};
        bool overlapMeasured{};
        bool overlapProfitable{};
        bool overlapCovered{};
        if (std::optional<Prediction> const online = fromOnline(overlapKey))
        {
            overlap = *online;
            overlapMeasured = mRuntimeCostTracker->confidence(overlapKey) == PhaseRuntimeCostConfidence::kReady;
            overlapProfitable = mRuntimeCostTracker->overlapEligible(overlapKey);
        }
        else
        {
            PhaseOverlapBatchCost const* selected{};
            for (PhaseOverlapBatchCost const& cost : mConfig.overlapBatchCosts)
            {
                bool const classCompatible
                    = cost.prefillClass == PhasePrefillClass::kAny || cost.prefillClass == overlapPrefillClass;
                if (classCompatible && cost.initialChunk == overlapPrefillInitial
                    && cost.prefillBatchSize >= overlapPrefillRows && cost.decodeBatchSize >= overlapDecodeRows
                    && cost.chunkLength >= overlapPrefillChunk && cost.maxPrefillPastKVLength >= overlapPrefillPastKV
                    && cost.maxDecodeContextLength >= overlapDecodeMaxContext
                    && (selected == nullptr || cost.makespanP95GpuMs < selected->makespanP95GpuMs))
                {
                    selected = &cost;
                }
            }
            overlap.makespanUs = selected != nullptr ? static_cast<double>(selected->makespanP95GpuMs) * 1000.0
                                                     : overlapPrefill->makespanUs + overlapDecode->makespanUs;
            overlap.uncertaintyUs
                = selected != nullptr ? 0.0 : overlapPrefill->uncertaintyUs + overlapDecode->uncertaintyUs;
            overlap.referenceWorkUs = overlapPrefill->makespanUs + overlapDecode->makespanUs;
            overlap.directlyKnown = selected != nullptr;
            overlapMeasured = selected != nullptr;
            overlapProfitable = selected != nullptr;
        }
        if (!overlapMeasured && mRuntimeCostTracker->contextualPdConfig().mode == PhaseContextualPdMode::kDisabled)
        {
            std::optional<PhaseGlobalCostEstimate> const covering
                = mRuntimeCostTracker->trustedEstimateCoveringOverlap(overlapKey);
            if (covering.has_value())
            {
                overlap.makespanUs = static_cast<double>(covering->makespanMedianMs) * 1000.0;
                overlap.uncertaintyUs = static_cast<double>(covering->uncertaintyMs) * 1000.0;
                overlap.referenceWorkUs = overlapPrefill->makespanUs + overlapDecode->makespanUs;
                overlap.directlyKnown = false;
                overlapMeasured = true;
                double const robustMakespanUs = overlap.makespanUs + overlap.uncertaintyUs;
                overlapProfitable = overlap.referenceWorkUs >= robustMakespanUs
                        * (1.0 + static_cast<double>(mConfig.globalCostModelConfig.minimumOverlapGainRatio));
                overlapCovered = true;
            }
        }
        PhaseGlobalOverlapCostDiagnostic const localDiagnostic = mRuntimeCostTracker->overlapDiagnostic(overlapKey);
        overlapMeasured = overlapMeasured || localDiagnostic.status == PhaseGlobalOverlapCostStatus::kEligible
            || localDiagnostic.status == PhaseGlobalOverlapCostStatus::kUnprofitable;
        switch (localDiagnostic.status)
        {
        case PhaseGlobalOverlapCostStatus::kNoSamples: ++mTelemetry.globalOverlapNoSampleCount; break;
        case PhaseGlobalOverlapCostStatus::kInsufficientSamples:
            ++mTelemetry.globalOverlapInsufficientSampleCount;
            break;
        case PhaseGlobalOverlapCostStatus::kEligible: ++mTelemetry.globalOverlapKnownCostCount; break;
        case PhaseGlobalOverlapCostStatus::kUnprofitable: ++mTelemetry.globalOverlapUnprofitableCount; break;
        }
        mTelemetry.globalOverlapCoveringCostCount += overlapCovered ? 1U : 0U;
        bool const needsLocalCalibration = localDiagnostic.status == PhaseGlobalOverlapCostStatus::kNoSamples
            || localDiagnostic.status == PhaseGlobalOverlapCostStatus::kInsufficientSamples;
        bool calibrationTarget = !mGlobalWarmupProbeMode;
        if (mGlobalWarmupProbeMode)
        {
            PhaseGlobalActionKey const calibrationKey = phaseGlobalCanonicalOverlapCostKey(overlapKey);
            auto const tracked
                = std::find(mGlobalCalibrationKeys.begin(), mGlobalCalibrationKeys.end(), calibrationKey);
            if (tracked != mGlobalCalibrationKeys.end())
            {
                size_t const index = static_cast<size_t>(std::distance(mGlobalCalibrationKeys.begin(), tracked));
                ++mGlobalCalibrationOpportunities[index];
                calibrationTarget = true;
            }
            else if (mGlobalCalibrationKeys.size() < mConfig.globalCalibrationMaxOverlapKeys)
            {
                mGlobalCalibrationKeys.push_back(calibrationKey);
                mGlobalCalibrationOpportunities.push_back(1U);
                calibrationTarget = true;
            }
        }
        double const robustSerialUs = overlapPrefill->makespanUs + overlapPrefill->uncertaintyUs
            + overlapDecode->makespanUs + overlapDecode->uncertaintyUs;
        bool const probeIntervalReady = mLastGlobalSafeProbeSequence == 0U
            || mGlobalDecisionSequence - mLastGlobalSafeProbeSequence >= mConfig.globalSafeProbeInterval;
        bool const probeSlackSafe = std::min(prefillSlack, decodeSlack)
            >= static_cast<double>(mConfig.globalSafeProbeSlackMultiplier) * robustSerialUs;
        // Once serial service is itself predicted to miss the protected
        // request, refusing every unknown overlap makes that violation
        // self-sustaining. The cold-start overlap bound below is exactly the
        // robust serial cost, so one rate-limited probe cannot be ranked as
        // safer than serial but may discover a recovery action.
        bool const deadlineRecoveryProbe = std::min(prefillSlack, decodeSlack) < robustSerialUs;
        bool const calibrationProbe = mGlobalWarmupProbeMode && calibrationTarget && needsLocalCalibration;
        bool const safeProbe = calibrationProbe
            || (!overlapMeasured && needsLocalCalibration && mConfig.globalSafeProbeSlackMultiplier > 0.0F
                && probeIntervalReady && (probeSlackSafe || deadlineRecoveryProbe));
        if (safeProbe)
        {
            ++mTelemetry.globalOverlapSafeProbeEligibleCount;
        }
        else if (!overlapMeasured && needsLocalCalibration && !mGlobalWarmupProbeMode)
        {
            if (mConfig.globalSafeProbeSlackMultiplier <= 0.0F)
            {
                ++mTelemetry.globalOverlapProbeDisabledCount;
            }
            else if (!probeIntervalReady)
            {
                ++mTelemetry.globalOverlapProbeIntervalBlockedCount;
            }
            else if (!probeSlackSafe)
            {
                ++mTelemetry.globalOverlapProbeSlackBlockedCount;
            }
        }
        PhaseGlobalActionCandidate candidate;
        candidate.key = overlapKey;
        candidate.primaryRequestIds = overlapPrefillRequestIds;
        candidate.secondaryRequestIds = overlapDecodeRequestIds;
        candidate.primaryStableSlotIds = overlapPrefillStableSlotIds;
        candidate.secondaryStableSlotIds = overlapDecodeStableSlotIds;
        phaseGlobalFinalizeCandidate(candidate);
        candidate.overlapCostKnown = overlapMeasured;
        candidate.overlapCostProfitable = overlapProfitable;
        candidate.safeProbeEligible = safeProbe;
        candidate.calibrationProbe = calibrationProbe;
        candidate.predictedBlockingUs = overlap.makespanUs;
        candidate.predictedMakespanUs = overlap.makespanUs;
        candidate.uncertaintyUs = overlap.uncertaintyUs;
        // Overlap profitability compares the exact batched P and D actions
        // against executing those same actions serially. Singleton-scaled
        // work is useful for batch efficiency, but it would classify large
        // contended overlaps as profitable merely because batching itself is
        // efficient.
        candidate.referenceWorkUs = overlapPrefill->makespanUs + overlapDecode->makespanUs;
        candidate.requestServiceLagUs = std::max(state.prefillOldestRequestAgeUs, state.decodeOldestWaitUs);
        candidate.memory = memoryFor(candidate.key, candidate.requestIds);
        PhaseContextualPdMode const contextualMode = mRuntimeCostTracker->contextualPdConfig().mode;
        PhaseContextualPairDirection const contextualDirection
            = phaseContextualPairDirection(overlapKey.kind, overlapKey.residualAnchor);
        if (contextualMode != PhaseContextualPdMode::kDisabled)
        {
            // Once E has completed, an external/VLM prefill is ordinary P
            // work for this pair action. Its TTFT remains protected below,
            // while an outstanding or pending producer still prevents the
            // local P+D head from taking policy authority. Keeping external
            // P out of the feature space made generic text observations
            // unusable precisely when a VLM request reached P+D.
            PhaseContextualPdInput contextualInput{overlapPrefill->makespanUs, overlapDecode->makespanUs,
                std::min(prefillSlack, decodeSlack), overlapPrefillRows, overlapDecodeRows, overlapPrefillChunk,
                overlapKey.primaryContextBucket, overlapKey.secondaryContextBucket, overlapKey.executionVariant,
                overlapKey.residualAugmentation, overlapKey.residualAnchor};
            contextualInput.prefillBatchCapacity = mConfig.maxPrefillBatchSize;
            contextualInput.decodeBatchCapacity = mConfig.maxDecodeBatchSize;
            candidate.contextualPdFeatures = phaseContextualPdFeatures(contextualInput);
            candidate.contextualPdFeatureValid = true;
            PhaseContextualPdEstimate const contextual
                = mRuntimeCostTracker->predictContextualDirection(contextualDirection, candidate.contextualPdFeatures);
            candidate.contextualPdMean = contextual.mean;
            candidate.contextualPdUncertainty = contextual.uncertainty;
            candidate.contextualPdLowerConfidenceBound = contextual.lowerConfidenceBound;
            candidate.contextualPdExploration
                = !contextual.ready && probeIntervalReady && (probeSlackSafe || deadlineRecoveryProbe);
            bool const producerCriticalPath = phaseContextualPdProducerCriticalPath(
                mExternalEncoderActive || state.prefillPendingProducerRows > 0U, false);
            if (contextualMode == PhaseContextualPdMode::kActive && contextual.ready
                && phaseContextualPdControlsDecision(producerCriticalPath))
            {
                candidate.decisionCostKnown = true;
                candidate.contextualScalarAuthorityApplied = true;
                candidate.decisionMakespanUs
                    = phaseContextualDecisionMakespanUs(candidate.referenceWorkUs, contextual.lowerConfidenceBound);
            }
            ++mTelemetry.contextualPdPredictionCount;
            mTelemetry.contextualPdReadyCount += contextual.ready ? 1U : 0U;
        }
        candidate.protectedCompletions.push_back(
            protectedPrefillCompletion(overlap, protectedPrefillAdvance(overlapPlan.prefillBatch)));
        double overlapDecodeCompletionUs = overlap.makespanUs;
        double overlapDecodeUncertaintyUs = overlap.uncertaintyUs;
        if (std::optional<float> const decodeP95
            = estimateGlobalDecodeComponentP95(overlapDecodeRows, overlapDecodeMaxContext, true))
        {
            overlapDecodeCompletionUs = static_cast<double>(*decodeP95) * 1000.0;
            overlapDecodeUncertaintyUs = 0.0;
        }
        candidate.protectedCompletions.push_back(
            {decodeSlack, overlapDecodeCompletionUs, overlapDecodeUncertaintyUs, PhaseProtectedKind::kDecode});
        if (decodeFormation.has_value())
        {
            double const currentActionUs
                = candidate.decisionCostKnown ? candidate.decisionMakespanUs : candidate.predictedMakespanUs;
            candidate.predictedHorizonUs = currentActionUs + decodeFormation->newlyProduced.makespanUs;
            candidate.horizonReferenceWorkUs
                = prefill->makespanUs + decode->makespanUs + decodeFormation->newlyProduced.makespanUs;
        }
        candidates.push_back(std::move(candidate));
    }

    ++mGlobalDecisionSequence;
    PhaseGlobalDecision decision;
    if (mGlobalWarmupProbeMode)
    {
        decision.inputCandidates = candidates.size();
        decision.hardFeasibleCandidates = candidates.size();
        decision.deadlineSafeCandidates = candidates.size();
        auto const selected = std::find_if(candidates.begin(), candidates.end(), [](auto const& candidate) {
            return candidate.key.kind == PhaseGlobalActionKind::kPrefillDecode && candidate.calibrationProbe;
        });
        if (selected != candidates.end())
        {
            decision.selectedIndex = static_cast<size_t>(std::distance(candidates.begin(), selected));
            decision.reason = PhaseGlobalDecisionReason::kDeadlineSafeEfficiency;
            decision.serviceCompression = selected->referenceWorkUs
                / std::max(selected->predictedMakespanUs, std::numeric_limits<double>::epsilon());
        }
        else
        {
            decision = mGlobalScheduler.select(candidates);
        }
    }
    else if (requiredKind.has_value())
    {
        auto candidateKind = [](PhaseGlobalActionKind kind) {
            PhaseDispatchKind result{PhaseDispatchKind::kNone};
            switch (kind)
            {
            case PhaseGlobalActionKind::kPrefill: result = PhaseDispatchKind::kPrefill; break;
            case PhaseGlobalActionKind::kDecode: result = PhaseDispatchKind::kDecode; break;
            case PhaseGlobalActionKind::kPrefillDecode: result = PhaseDispatchKind::kOverlap; break;
            case PhaseGlobalActionKind::kNone:
            case PhaseGlobalActionKind::kEncoder:
            case PhaseGlobalActionKind::kEncoderPrefill:
            case PhaseGlobalActionKind::kEncoderDecode:
            case PhaseGlobalActionKind::kWait: break;
            }
            return result;
        };
        decision.inputCandidates = candidates.size();
        decision.hardFeasibleCandidates = candidates.size();
        decision.deadlineSafeCandidates = candidates.size();
        auto const selected = std::find_if(candidates.begin(), candidates.end(),
            [&](auto const& candidate) { return candidateKind(candidate.key.kind) == *requiredKind; });
        if (selected != candidates.end())
        {
            decision.selectedIndex = static_cast<size_t>(std::distance(candidates.begin(), selected));
            decision.reason = PhaseGlobalDecisionReason::kDeadlineSafeEfficiency;
            double const makespanUs
                = selected->predictedHorizonUs > 0.0 ? selected->predictedHorizonUs : selected->predictedMakespanUs;
            double const referenceUs
                = selected->horizonReferenceWorkUs > 0.0 ? selected->horizonReferenceWorkUs : selected->referenceWorkUs;
            decision.serviceCompression = referenceUs / std::max(makespanUs, std::numeric_limits<double>::epsilon());
        }
        else
        {
            decision.reason = PhaseGlobalDecisionReason::kNoHardFeasibleCandidate;
        }
    }
    else
    {
        decision = mGlobalScheduler.select(candidates, audit != nullptr ? &audit->inputs : nullptr);
    }
    if (audit != nullptr)
    {
        audit->decision = decision;
    }
    if (!mGlobalWarmupProbeMode && mRuntimeCostTracker->contextualPdConfig().mode != PhaseContextualPdMode::kDisabled)
    {
        auto const overlap = std::find_if(candidates.begin(), candidates.end(), [](auto const& candidate) {
            return candidate.key.kind == PhaseGlobalActionKind::kPrefillDecode && candidate.contextualPdFeatureValid;
        });
        if (overlap != candidates.end())
        {
            bool const producerCriticalPath = phaseContextualPdProducerCriticalPath(
                mExternalEncoderActive || state.prefillPendingProducerRows > 0U, false);
            bool const selectedOverlap = decision.selectedIndex.has_value()
                && candidates[*decision.selectedIndex].key.kind == PhaseGlobalActionKind::kPrefillDecode;
            bool const contextualControlsDecision = phaseContextualPdControlsDecision(producerCriticalPath);
            bool const contextualOverlap = overlap->contextualPdLowerConfidenceBound > 0.0;
            mTelemetry.contextualPdDecisionDisagreementCount
                += contextualControlsDecision && contextualOverlap != selectedOverlap ? 1U : 0U;
            if (contextualControlsDecision)
            {
                PhaseContextualPairDirection const direction
                    = phaseContextualPairDirection(overlap->key.kind, overlap->key.residualAnchor);
                mRuntimeCostTracker->recordContextualDirectionSelection(
                    direction, selectedOverlap, overlap->contextualPdExploration);
            }
        }
    }
    ++mTelemetry.globalDecisionCount;
    mTelemetry.lastGlobalDecisionReason = decision.reason;
    mTelemetry.lastGlobalPredictedViolationUs = decision.predictedViolationUs;
    mTelemetry.lastGlobalServiceCompression = decision.serviceCompression;
    if (!decision.selectedIndex.has_value())
    {
        ++mTelemetry.globalNoFeasibleDecisionCount;
        return std::nullopt;
    }
    PhaseGlobalActionCandidate const& selected = candidates[*decision.selectedIndex];
    if (prefillFormation.has_value() && selected.key.kind == PhaseGlobalActionKind::kDecode)
    {
        ++mTelemetry.globalPrefillFormationDecodeSelectionCount;
    }
    if (decodeFormation.has_value())
    {
        if (selected.key.kind == PhaseGlobalActionKind::kPrefill)
        {
            ++mTelemetry.globalDecodeFormationPrefillSelectionCount;
        }
        else if (selected.key.kind == PhaseGlobalActionKind::kDecode)
        {
            ++mTelemetry.globalDecodeFormationDecodeSelectionCount;
        }
        else if (selected.key.kind == PhaseGlobalActionKind::kPrefillDecode)
        {
            ++mTelemetry.globalDecodeFormationOverlapSelectionCount;
        }
    }
    PhaseDispatchKind kind{PhaseDispatchKind::kNone};
    switch (selected.key.kind)
    {
    case PhaseGlobalActionKind::kPrefill: kind = PhaseDispatchKind::kPrefill; break;
    case PhaseGlobalActionKind::kDecode: kind = PhaseDispatchKind::kDecode; break;
    case PhaseGlobalActionKind::kPrefillDecode: kind = PhaseDispatchKind::kOverlap; break;
    default: break;
    }
    bool const safeProbe = selected.calibrationProbe || (selected.safeProbeEligible && !selected.overlapCostKnown);
    if (safeProbe && mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive)
    {
        mLastGlobalSafeProbeSequence = mGlobalDecisionSequence;
        ++mTelemetry.globalSafeProbeCount;
    }
    if (selected.key.kind == PhaseGlobalActionKind::kPrefillDecode)
    {
        ++mTelemetry.globalOverlapSelectionCount;
        mTelemetry.globalKnownOverlapPriorityCount += selected.overlapCostProfitable ? 1U : 0U;
        mTelemetry.globalMeasuredUnprofitableOverlapSelectionCount
            += selected.overlapCostKnown && !selected.overlapCostProfitable ? 1U : 0U;
    }
    mTelemetry.lastGlobalSelectedAction = selected.key.kind;
    return GlobalQueueSelection{kind, selected, decision, safeProbe, std::move(candidates)};
}

bool PhaseQueueScheduler::shouldWaitForDecodeEvents(std::vector<PhaseDecodeCompletionPreview> const& previews)
{
    constexpr size_t kMaxWaitCandidates{2U};
    if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kDisabled || previews.empty())
    {
        return false;
    }
    PhaseQueueSnapshot const state = snapshot();
    if (state.prefillQueued > 0U || state.decodeQueued >= static_cast<size_t>(mConfig.maxDecodeBatchSize))
    {
        return false;
    }
    if (state.decodeQueued == 0U)
    {
        PhaseDecodeCompletionPreview const& next = previews.front();
        ++mTelemetry.globalWaitDecisionCount;
        ++mTelemetry.globalWaitSelectedCount;
        ++mTelemetry.globalWaitCandidateCount;
        mTelemetry.globalWaitCurrentRows = 0;
        mTelemetry.globalWaitFutureRows
            = static_cast<int32_t>(std::min(next.requestIds.size(), static_cast<size_t>(mConfig.maxDecodeBatchSize)));
        mTelemetry.globalWaitEventId = next.eventId;
        mTelemetry.globalWaitRequestIds = next.requestIds;
        mTelemetry.lastGlobalSelectedAction = PhaseGlobalActionKind::kWait;
        mTelemetry.lastGlobalDecisionReason = PhaseGlobalDecisionReason::kDeadlineSafeEfficiency;
        return mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive;
    }
    std::vector<PhaseGlobalActionCandidate> candidates;
    candidates.reserve(2U * std::min(kMaxWaitCandidates, previews.size()));
    int32_t const contextBucketTokens = std::max(1, mConfig.runtimeDecodeContextBucketTokens);
    mTelemetry.globalWaitPreviewBlockingUs = 0.0;
    mTelemetry.globalWaitPreviewUncertaintyUs = 0.0;
    mTelemetry.globalWaitPreviewSlackUs = state.decodeMinTpotSlackUs;
    mTelemetry.globalWaitPreviewCompression = 0.0;
    float predictedDrainGpuMs{};
    int32_t predictedDrainTurns{};
    int32_t const currentBatchSize = selectDecodeBatchSize(state, false, predictedDrainGpuMs, predictedDrainTurns);
    std::vector<PhaseWorkItem const*> const selectedCurrentRows = decodeCandidateRows(currentBatchSize);
    if (selectedCurrentRows.empty())
    {
        return false;
    }
    std::vector<PhaseWorkItem const*> currentRows = selectedCurrentRows;
    mTelemetry.globalWaitCurrentRows = static_cast<int32_t>(currentRows.size());

    struct DecodePrediction
    {
        double makespanUs{};
        double uncertaintyUs{};
        double referenceUs{};
        int32_t graphBucket{};
    };
    auto predictDecode = [&](int32_t rows, int64_t totalContextTokens, int32_t maxContextLength) {
        int32_t const contextBucket = (std::max(0, maxContextLength) + contextBucketTokens - 1) / contextBucketTokens;
        PhaseGlobalActionKey key{PhaseGlobalActionKind::kDecode, rows, 0, 1, contextBucket, 0};
        key.executionVariant
            = mGlobalExecutionVariantSupplier ? mGlobalExecutionVariantSupplier(key, 0) : PhaseExecutionVariant::kEager;
        DecodePrediction prediction{static_cast<double>(mConfig.globalColdDecodeMs) * 1000.0,
            static_cast<double>(mConfig.globalCostModelConfig.coldStartUncertaintyMs) * 1000.0,
            static_cast<double>(mConfig.globalColdDecodeMs) * 1000.0, rows};
        if (std::optional<PhaseGlobalCostEstimate> const online = mRuntimeCostTracker->estimate(key))
        {
            prediction.makespanUs = static_cast<double>(online->makespanMedianMs) * 1000.0;
            prediction.uncertaintyUs = static_cast<double>(online->uncertaintyMs) * 1000.0;
            prediction.referenceUs = static_cast<double>(online->referenceWorkMedianMs) * 1000.0;
            return prediction;
        }

        PhaseDecodeBatchCost const* selected{};
        PhaseDecodeBatchCost const* singleton{};
        for (PhaseDecodeBatchCost const& cost : mConfig.decodeBatchCosts)
        {
            if (cost.batchSize == 1 && cost.maxContextLength >= maxContextLength
                && (singleton == nullptr || cost.maxContextLength < singleton->maxContextLength
                    || (cost.maxContextLength == singleton->maxContextLength && cost.p95GpuMs < singleton->p95GpuMs)))
            {
                singleton = &cost;
            }
            int64_t const totalLimit = cost.maxTotalContextTokens > 0
                ? cost.maxTotalContextTokens
                : static_cast<int64_t>(cost.batchSize) * cost.maxContextLength;
            if (cost.maxContextLength < maxContextLength || totalLimit < totalContextTokens)
            {
                continue;
            }
            if (cost.batchSize >= rows
                && (selected == nullptr || cost.batchSize < selected->batchSize
                    || (cost.batchSize == selected->batchSize && cost.p95GpuMs < selected->p95GpuMs)))
            {
                selected = &cost;
            }
        }
        if (selected != nullptr)
        {
            prediction.graphBucket = selected->batchSize;
            prediction.makespanUs = static_cast<double>(selected->p95GpuMs) * 1000.0;
            prediction.uncertaintyUs = 0.0;
            prediction.referenceUs = singleton != nullptr ? static_cast<double>(singleton->p95GpuMs) * 1000.0 * rows
                                                          : prediction.makespanUs;
        }
        return prediction;
    };

    int64_t currentContextTokens{};
    int32_t currentMaxContextLength{};
    PhaseGlobalActionCandidate currentCandidate;
    for (PhaseWorkItem const* item : currentRows)
    {
        currentContextTokens += item->tokenCount;
        currentMaxContextLength = std::max(currentMaxContextLength, item->tokenCount);
        currentCandidate.primaryRequestIds.push_back(item->requestId);
        currentCandidate.primaryStableSlotIds.push_back(item->kvSlotId);
    }
    DecodePrediction const currentPrediction = predictDecode(
        static_cast<int32_t>(selectedCurrentRows.size()), currentContextTokens, currentMaxContextLength);
    int32_t const currentContextBucket
        = (std::max(0, currentMaxContextLength) + contextBucketTokens - 1) / contextBucketTokens;
    currentCandidate.key = {PhaseGlobalActionKind::kDecode, static_cast<int32_t>(selectedCurrentRows.size()), 0, 1,
        currentContextBucket, 0};
    currentCandidate.key.executionVariant = mGlobalExecutionVariantSupplier
        ? mGlobalExecutionVariantSupplier(currentCandidate.key, 0)
        : PhaseExecutionVariant::kEager;
    currentCandidate.predictedBlockingUs = currentPrediction.makespanUs;
    currentCandidate.predictedMakespanUs = currentPrediction.makespanUs;
    currentCandidate.uncertaintyUs = currentPrediction.uncertaintyUs;
    currentCandidate.referenceWorkUs = currentPrediction.referenceUs;
    currentCandidate.requestServiceLagUs = state.decodeOldestWaitUs;
    phaseGlobalFinalizeCandidate(currentCandidate);
    currentCandidate.protectedCompletions.push_back({state.decodeMinTpotSlackUs, currentPrediction.makespanUs,
        currentPrediction.uncertaintyUs, PhaseProtectedKind::kDecode});
    if (mConfig.globalMemoryHorizonSupplier)
    {
        currentCandidate.memory
            = mConfig.globalMemoryHorizonSupplier(currentCandidate.key, currentCandidate.requestIds);
    }

    // Production uses the bounded one-step refill horizon. The rejected
    // full-drain experiment is removed in the Global-only scheduler stage.
    {
        size_t waitCandidates{};
        for (PhaseDecodeCompletionPreview const& preview : previews)
        {
            if (waitCandidates >= kMaxWaitCandidates || preview.eventId == 0U || preview.predictedWaitUs < 0.0
                || preview.waitUncertaintyUs < 0.0 || preview.requestIds.empty()
                || preview.requestIds.size() != preview.contextLengths.size()
                || (!preview.stableSlotIds.empty() && preview.requestIds.size() != preview.stableSlotIds.size()))
            {
                continue;
            }
            std::vector<uint64_t> futureRequestIds = currentCandidate.requestIds;
            std::vector<int32_t> futureStableSlotIds = currentCandidate.primaryStableSlotIds;
            int64_t futureContextTokens = currentContextTokens;
            int32_t futureMaxContextLength = currentMaxContextLength;
            int64_t residualContextTokens{};
            int32_t residualMaxContextLength{};
            int32_t residualRows{};
            for (size_t index{}; index < preview.requestIds.size()
                && futureRequestIds.size() < static_cast<size_t>(mConfig.maxDecodeBatchSize);
                ++index)
            {
                if (preview.contextLengths[index] < 0
                    || std::find(futureRequestIds.begin(), futureRequestIds.end(), preview.requestIds[index])
                        != futureRequestIds.end())
                {
                    continue;
                }
                futureRequestIds.push_back(preview.requestIds[index]);
                futureStableSlotIds.push_back(preview.stableSlotIds.empty() ? -1 : preview.stableSlotIds[index]);
                futureContextTokens += preview.contextLengths[index];
                futureMaxContextLength = std::max(futureMaxContextLength, preview.contextLengths[index]);
                residualContextTokens += preview.contextLengths[index];
                residualMaxContextLength = std::max(residualMaxContextLength, preview.contextLengths[index]);
                ++residualRows;
            }
            int32_t const futureRows = static_cast<int32_t>(futureRequestIds.size());
            if (futureRows <= currentCandidate.key.primaryBatchSize)
            {
                continue;
            }

            int32_t const contextBucket
                = (std::max(0, futureMaxContextLength) + contextBucketTokens - 1) / contextBucketTokens;
            DecodePrediction const future = predictDecode(futureRows, futureContextTokens, futureMaxContextLength);
            DecodePrediction const residual
                = predictDecode(residualRows, residualContextTokens, residualMaxContextLength);
            double const horizonReferenceUs = currentCandidate.referenceWorkUs + residual.referenceUs;

            PhaseGlobalActionCandidate nowCandidate = currentCandidate;
            nowCandidate.predictedHorizonUs
                = std::max(nowCandidate.predictedMakespanUs, preview.predictedWaitUs) + residual.makespanUs;
            nowCandidate.horizonReferenceWorkUs = horizonReferenceUs;
            nowCandidate.uncertaintyUs += preview.waitUncertaintyUs + residual.uncertaintyUs;
            nowCandidate.protectedCompletions.clear();
            nowCandidate.protectedCompletions.push_back({state.decodeMinTpotSlackUs,
                currentCandidate.predictedBlockingUs, currentCandidate.uncertaintyUs, PhaseProtectedKind::kDecode});
            candidates.push_back(std::move(nowCandidate));

            PhaseGlobalActionCandidate wait;
            wait.key = {PhaseGlobalActionKind::kWait, futureRows, future.graphBucket, 1, contextBucket, 0};
            wait.concreteWaitEvent = true;
            wait.waitEventId = preview.eventId;
            wait.predictedBlockingUs = preview.predictedWaitUs + future.makespanUs;
            wait.predictedMakespanUs = wait.predictedBlockingUs;
            wait.predictedHorizonUs = wait.predictedBlockingUs;
            wait.uncertaintyUs = preview.waitUncertaintyUs + future.uncertaintyUs;
            wait.referenceWorkUs = future.referenceUs;
            wait.horizonReferenceWorkUs = horizonReferenceUs;
            wait.requestServiceLagUs = state.decodeOldestWaitUs;
            wait.primaryRequestIds = std::move(futureRequestIds);
            wait.primaryStableSlotIds = std::move(futureStableSlotIds);
            phaseGlobalFinalizeCandidate(wait);
            wait.protectedCompletions.push_back({state.decodeMinTpotSlackUs, wait.predictedBlockingUs,
                wait.uncertaintyUs, PhaseProtectedKind::kDecode});
            if (mConfig.globalMemoryHorizonSupplier)
            {
                wait.memory = mConfig.globalMemoryHorizonSupplier(wait.key, wait.requestIds);
            }
            double const compression
                = wait.referenceWorkUs / std::max(wait.predictedMakespanUs, std::numeric_limits<double>::epsilon());
            if (compression > mTelemetry.globalWaitPreviewCompression)
            {
                mTelemetry.globalWaitPreviewBlockingUs = wait.predictedBlockingUs;
                mTelemetry.globalWaitPreviewUncertaintyUs = wait.uncertaintyUs;
                mTelemetry.globalWaitPreviewCompression = compression;
            }
            candidates.push_back(std::move(wait));
            ++waitCandidates;
        }
        if (waitCandidates == 0U)
        {
            return false;
        }

        PhaseGlobalDecision const decision = mGlobalScheduler.select(candidates);
        ++mTelemetry.globalWaitDecisionCount;
        mTelemetry.globalWaitCandidateCount += waitCandidates;
        bool const selectedWait = decision.selectedIndex.has_value()
            && candidates[*decision.selectedIndex].key.kind == PhaseGlobalActionKind::kWait;
        mTelemetry.globalWaitSelectedCount += selectedWait ? 1U : 0U;
        if (selectedWait)
        {
            PhaseGlobalActionCandidate const& selected = candidates[*decision.selectedIndex];
            mTelemetry.lastGlobalSelectedAction = PhaseGlobalActionKind::kWait;
            mTelemetry.lastGlobalDecisionReason = decision.reason;
            mTelemetry.lastGlobalPredictedViolationUs = decision.predictedViolationUs;
            mTelemetry.lastGlobalServiceCompression = decision.serviceCompression;
            mTelemetry.globalWaitFutureRows = selected.key.primaryBatchSize;
            mTelemetry.globalWaitGraphBucket = selected.key.secondaryBatchSize;
            mTelemetry.globalWaitEventId = selected.waitEventId;
            mTelemetry.globalWaitRequestIds = selected.requestIds;
        }
        return selectedWait && mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive;
    }
}

std::optional<PhaseGlobalActionCandidate> PhaseQueueScheduler::previewGlobalAction(PhaseGlobalSelectionAudit* audit)
{
    if (audit != nullptr)
    {
        *audit = {};
    }
    PhaseQueueSnapshot const state = snapshot();
    std::optional<GlobalQueueSelection> selection
        = selectGlobalQueueAction(state, true, true, true, std::nullopt, audit);
    mLastGlobalPreviewCandidates
        = selection.has_value() ? std::move(selection->candidateFrontier) : std::vector<PhaseGlobalActionCandidate>{};
    return selection.has_value() ? std::optional<PhaseGlobalActionCandidate>(std::move(selection->candidate))
                                 : std::nullopt;
}

std::vector<PhaseGlobalActionCandidate> const& PhaseQueueScheduler::lastGlobalPreviewCandidates() const noexcept
{
    return mLastGlobalPreviewCandidates;
}

std::optional<PhaseGlobalActionCandidate> PhaseQueueScheduler::previewGlobalPrefillAction()
{
    std::optional<GlobalQueueSelection> const selection
        = selectGlobalQueueAction(snapshot(), true, false, false, PhaseDispatchKind::kPrefill);
    return selection.has_value() ? std::optional<PhaseGlobalActionCandidate>(selection->candidate) : std::nullopt;
}

std::optional<PhaseGlobalActionCandidate> PhaseQueueScheduler::previewGlobalDecodeAction()
{
    std::optional<GlobalQueueSelection> const selection
        = selectGlobalQueueAction(snapshot(), false, true, false, PhaseDispatchKind::kDecode);
    return selection.has_value() ? std::optional<PhaseGlobalActionCandidate>(selection->candidate) : std::nullopt;
}

std::optional<PhaseGlobalResidualSelection> PhaseQueueScheduler::previewGlobalResidualAction(
    PhaseGlobalActionCandidate const& launched, double elapsedUs)
{
    if (mConfig.globalSchedulerMode != PhaseGlobalSchedulerMode::kActive
        || (launched.key.kind != PhaseGlobalActionKind::kPrefill
            && launched.key.kind != PhaseGlobalActionKind::kDecode))
    {
        return std::nullopt;
    }

    bool const addDecode = launched.key.kind == PhaseGlobalActionKind::kPrefill;
    std::optional<PhaseGlobalActionCandidate> missing
        = addDecode ? previewGlobalDecodeAction() : previewGlobalPrefillAction();
    if (!missing.has_value())
    {
        return std::nullopt;
    }

    PhaseGlobalActionCandidate const active = phaseGlobalResidualCandidate(launched, elapsedUs);
    PhaseGlobalActionCandidate const& prefill = addDecode ? active : *missing;
    PhaseGlobalActionCandidate const& decode = addDecode ? *missing : active;
    PhaseGlobalActionCandidate overlap;
    overlap.key = {PhaseGlobalActionKind::kPrefillDecode, prefill.key.primaryBatchSize, decode.key.primaryBatchSize,
        prefill.key.chunkLength, prefill.key.primaryContextBucket, decode.key.primaryContextBucket};
    overlap.key.executionVariant
        = phaseExecutionVariant(phaseExecutionVariantUsesPrimaryGraph(prefill.key.executionVariant),
            phaseExecutionVariantUsesPrimaryGraph(decode.key.executionVariant));
    overlap.key.primaryWorkClass = prefill.key.primaryWorkClass;
    overlap.key.residualAugmentation = true;
    overlap.key.residualAnchor = addDecode ? PhaseGlobalResidualAnchor::kPrefill : PhaseGlobalResidualAnchor::kDecode;
    overlap.primaryRequestIds = prefill.primaryRequestIds;
    overlap.primaryStableSlotIds = prefill.primaryStableSlotIds;
    overlap.secondaryRequestIds = decode.primaryRequestIds;
    overlap.secondaryStableSlotIds = decode.primaryStableSlotIds;
    phaseGlobalFinalizeCandidate(overlap);
    overlap.referenceWorkUs = active.predictedMakespanUs + missing->predictedMakespanUs;
    overlap.requestServiceLagUs = std::max(active.requestServiceLagUs, missing->requestServiceLagUs);
    if (mConfig.globalMemoryHorizonSupplier)
    {
        overlap.memory = mConfig.globalMemoryHorizonSupplier(overlap.key, overlap.requestIds);
    }

    double const robustSerialUs
        = active.predictedMakespanUs + active.uncertaintyUs + missing->predictedMakespanUs + missing->uncertaintyUs;
    PhaseGlobalOverlapCostDiagnostic diagnostic = mRuntimeCostTracker->overlapDiagnostic(overlap.key);
    bool overlapKnown = diagnostic.status == PhaseGlobalOverlapCostStatus::kEligible
        || diagnostic.status == PhaseGlobalOverlapCostStatus::kUnprofitable;
    bool overlapProfitable = diagnostic.status == PhaseGlobalOverlapCostStatus::kEligible;
    std::optional<PhaseGlobalCostEstimate> estimate
        = overlapKnown ? mRuntimeCostTracker->estimate(overlap.key) : std::nullopt;
    if (!estimate.has_value())
    {
        PhaseGlobalActionKey completeKey = overlap.key;
        completeKey.residualAugmentation = false;
        completeKey.residualAnchor = PhaseGlobalResidualAnchor::kNone;
        PhaseGlobalOverlapCostDiagnostic const completeDiagnostic = mRuntimeCostTracker->overlapDiagnostic(completeKey);
        bool const completeKnown = completeDiagnostic.status == PhaseGlobalOverlapCostStatus::kEligible
            || completeDiagnostic.status == PhaseGlobalOverlapCostStatus::kUnprofitable;
        estimate = completeKnown ? mRuntimeCostTracker->estimate(completeKey)
                                 : mRuntimeCostTracker->trustedEstimateCoveringOverlap(completeKey);
        if (estimate.has_value())
        {
            overlapKnown = true;
            double const robustMakespanUs
                = static_cast<double>(estimate->makespanMedianMs + estimate->uncertaintyMs) * 1000.0;
            overlapProfitable = overlap.referenceWorkUs >= robustMakespanUs
                    * (1.0 + static_cast<double>(mConfig.globalCostModelConfig.minimumOverlapGainRatio));
        }
    }
    overlap.overlapCostKnown = overlapKnown;
    overlap.overlapCostProfitable = overlapProfitable;
    if (estimate.has_value())
    {
        overlap.predictedMakespanUs = static_cast<double>(estimate->makespanMedianMs) * 1000.0;
        overlap.uncertaintyUs = static_cast<double>(estimate->uncertaintyMs) * 1000.0;
    }
    else
    {
        overlap.predictedMakespanUs = std::max(active.predictedMakespanUs, missing->predictedMakespanUs);
        overlap.uncertaintyUs = std::max(0.0, robustSerialUs - overlap.predictedMakespanUs);
    }
    overlap.predictedBlockingUs = overlap.predictedMakespanUs;

    double protectedSlackUs = std::numeric_limits<double>::infinity();
    for (PhaseProtectedCompletion const& completion : active.protectedCompletions)
    {
        protectedSlackUs = std::min(protectedSlackUs, completion.slackUs);
    }
    for (PhaseProtectedCompletion const& completion : missing->protectedCompletions)
    {
        protectedSlackUs = std::min(protectedSlackUs, completion.slackUs);
    }
    bool const probeIntervalReady = mLastGlobalSafeProbeSequence == 0U
        || mGlobalDecisionSequence - mLastGlobalSafeProbeSequence >= mConfig.globalSafeProbeInterval;
    bool const deadlineRecovery = protectedSlackUs < robustSerialUs;
    bool const slackSafe
        = protectedSlackUs >= static_cast<double>(mConfig.globalSafeProbeSlackMultiplier) * robustSerialUs;
    overlap.safeProbeEligible = !overlapKnown && mConfig.globalSafeProbeSlackMultiplier > 0.0F && probeIntervalReady
        && (deadlineRecovery || slackSafe);

    PhaseContextualPdMode const contextualMode = mRuntimeCostTracker->contextualPdConfig().mode;
    if (contextualMode != PhaseContextualPdMode::kDisabled)
    {
        // Residual P+D augmentation is producer-agnostic once the P row is
        // runnable. External/VLM rows have already satisfied their E -> P
        // dependency here.
        PhaseContextualPdInput contextualInput{prefill.predictedMakespanUs, decode.predictedMakespanUs,
            protectedSlackUs, prefill.key.primaryBatchSize, decode.key.primaryBatchSize, prefill.key.chunkLength,
            prefill.key.primaryContextBucket, decode.key.primaryContextBucket, overlap.key.executionVariant, true,
            overlap.key.residualAnchor, elapsedUs, std::max(launched.predictedMakespanUs, launched.predictedBlockingUs),
            launched.predictedMakespanUs > 0.0 ? elapsedUs / launched.predictedMakespanUs : -1.0,
            addDecode ? PhaseExecutionSet::kPrefill : PhaseExecutionSet::kDecode};
        contextualInput.prefillBatchCapacity = mConfig.maxPrefillBatchSize;
        contextualInput.decodeBatchCapacity = mConfig.maxDecodeBatchSize;
        overlap.contextualPdFeatures = phaseContextualPdFeatures(contextualInput);
        overlap.contextualPdFeatureValid = true;
        PhaseContextualPairDirection const direction
            = phaseContextualPairDirection(overlap.key.kind, overlap.key.residualAnchor);
        PhaseContextualPdEstimate const contextual
            = mRuntimeCostTracker->predictContextualDirection(direction, overlap.contextualPdFeatures);
        overlap.contextualPdMean = contextual.mean;
        overlap.contextualPdUncertainty = contextual.uncertainty;
        overlap.contextualPdLowerConfidenceBound = contextual.lowerConfidenceBound;
        overlap.contextualPdExploration = !contextual.ready && probeIntervalReady && (deadlineRecovery || slackSafe);
        if (contextualMode == PhaseContextualPdMode::kActive && contextual.ready)
        {
            overlap.decisionCostKnown = true;
            overlap.contextualScalarAuthorityApplied = true;
            overlap.decisionMakespanUs
                = phaseContextualDecisionMakespanUs(overlap.referenceWorkUs, contextual.lowerConfidenceBound);
        }
    }

    double overlapDecodeCompletionUs = overlap.predictedMakespanUs;
    double overlapDecodeUncertaintyUs = overlap.uncertaintyUs;
    int32_t const decodeContextLength
        = decode.key.primaryContextBucket * std::max(1, mConfig.runtimeDecodeContextBucketTokens);
    if (std::optional<float> const decodeP95
        = estimateGlobalDecodeComponentP95(decode.key.primaryBatchSize, decodeContextLength, true))
    {
        overlapDecodeCompletionUs = static_cast<double>(*decodeP95) * 1000.0;
        overlapDecodeUncertaintyUs = 0.0;
    }
    auto appendProtected = [&](PhaseGlobalActionCandidate const& source) {
        for (PhaseProtectedCompletion completion : source.protectedCompletions)
        {
            completion.predictedCompletionUs = completion.kind == PhaseProtectedKind::kDecode
                ? overlapDecodeCompletionUs
                : overlap.predictedMakespanUs;
            completion.uncertaintyUs
                = completion.kind == PhaseProtectedKind::kDecode ? overlapDecodeUncertaintyUs : overlap.uncertaintyUs;
            overlap.protectedCompletions.push_back(completion);
        }
    };
    appendProtected(active);
    appendProtected(*missing);
    // Compare equal work horizons. Leaving the incumbent single phase active
    // does not discard the ready peer phase; it executes that phase at the
    // immediately following boundary. A one-action incumbent would therefore
    // make serial look artificially cheap and suppress every profitable
    // residual overlap.
    PhaseGlobalActionCandidate serial = active;
    serial.predictedHorizonUs = active.predictedMakespanUs + missing->predictedMakespanUs;
    serial.horizonReferenceWorkUs = overlap.referenceWorkUs;
    auto hasProtectedKind = [&](PhaseProtectedKind kind) {
        return std::any_of(serial.protectedCompletions.begin(), serial.protectedCompletions.end(),
            [&](PhaseProtectedCompletion const& completion) { return completion.kind == kind; });
    };
    for (PhaseProtectedCompletion completion : missing->protectedCompletions)
    {
        if (hasProtectedKind(completion.kind))
        {
            continue;
        }
        completion.predictedCompletionUs += active.predictedMakespanUs;
        completion.uncertaintyUs += active.uncertaintyUs;
        serial.protectedCompletions.push_back(completion);
    }

    ++mGlobalDecisionSequence;
    PhaseGlobalDecision const decision = mGlobalScheduler.select({serial, overlap});
    bool const selectedOverlap = decision.selectedIndex.has_value() && *decision.selectedIndex == 1U;
    if (overlap.contextualPdFeatureValid)
    {
        PhaseContextualPairDirection const direction
            = phaseContextualPairDirection(overlap.key.kind, overlap.key.residualAnchor);
        mRuntimeCostTracker->recordContextualDirectionSelection(
            direction, selectedOverlap, overlap.contextualPdExploration);
    }
    if (selectedOverlap && overlap.safeProbeEligible && !overlap.overlapCostKnown)
    {
        mLastGlobalSafeProbeSequence = mGlobalDecisionSequence;
        ++mTelemetry.globalSafeProbeCount;
    }
    mTelemetry.globalOverlapSelectionCount += selectedOverlap ? 1U : 0U;
    return PhaseGlobalResidualSelection{std::move(*missing), std::move(overlap), selectedOverlap};
}

PhaseGlobalCostEstimate PhaseQueueScheduler::estimateGlobalPrefillCost(
    int32_t batchSize, int32_t chunkLength, int32_t pastKVLength, PhasePrefillClass prefillClass) const
{
    check::check(batchSize > 0, "A future prefill estimate requires a positive batch size");
    check::check(chunkLength > 0, "A future prefill estimate requires a positive chunk length");
    check::check(pastKVLength >= 0, "A future prefill estimate requires a non-negative past-KV length");
    int32_t const contextBucketTokens = std::max(1, mConfig.runtimeDecodeContextBucketTokens);
    int32_t const contextBucket = (pastKVLength + contextBucketTokens - 1) / contextBucketTokens;
    PhaseGlobalActionKey key{PhaseGlobalActionKind::kPrefill, batchSize, 0, chunkLength, contextBucket, 0};
    key.primaryWorkClass = static_cast<int32_t>(prefillClass);
    if (std::optional<PhaseGlobalCostEstimate> const online = mRuntimeCostTracker->estimate(key))
    {
        return *online;
    }
    if (std::optional<PhaseGlobalCostEstimate> const covering = mRuntimeCostTracker->estimateCoveringPrimary(key))
    {
        return *covering;
    }
    std::optional<PhaseGlobalCostEstimate> const launchFloor = mRuntimeCostTracker->estimatePrimaryLaunchFloor(key);

    PhasePrefillBatchCost const* selected{};
    PhasePrefillBatchCost const* singleton{};
    bool const initialChunk = pastKVLength == 0;
    for (PhasePrefillBatchCost const& cost : mConfig.prefillBatchCosts)
    {
        bool const classCompatible = cost.prefillClass == PhasePrefillClass::kAny || cost.prefillClass == prefillClass;
        if (!classCompatible || cost.initialChunk != initialChunk || cost.chunkLength < chunkLength
            || cost.maxPastKVLength < pastKVLength || cost.maxConcurrentDecodeBatchSize != 0)
        {
            continue;
        }
        if (cost.batchSize >= batchSize
            && (selected == nullptr || cost.batchSize < selected->batchSize
                || (cost.batchSize == selected->batchSize && cost.p95GpuMs < selected->p95GpuMs)))
        {
            selected = &cost;
        }
        if (cost.batchSize == 1
            && (singleton == nullptr || cost.chunkLength < singleton->chunkLength
                || (cost.chunkLength == singleton->chunkLength && cost.p95GpuMs < singleton->p95GpuMs)))
        {
            singleton = &cost;
        }
    }
    if (selected != nullptr)
    {
        float const referenceMs
            = singleton != nullptr ? singleton->p95GpuMs * static_cast<float>(batchSize) : selected->p95GpuMs;
        float const makespanMs = launchFloor.has_value() ? std::max(selected->p95GpuMs, launchFloor->makespanMedianMs)
                                                         : selected->p95GpuMs;
        float const uncertaintyMs = launchFloor.has_value() ? launchFloor->uncertaintyMs : 0.0F;
        return {launchFloor.has_value() ? launchFloor->sampleCount : 0U, std::max(referenceMs, makespanMs), makespanMs,
            std::max(makespanMs, launchFloor.has_value() ? launchFloor->makespanP95Ms : makespanMs), uncertaintyMs};
    }

    double const totalTokens = static_cast<double>(batchSize) * static_cast<double>(chunkLength);
    float const coldMs
        = static_cast<float>(std::max(0.001, static_cast<double>(mConfig.globalColdPrefillMsPerToken) * totalTokens));
    if (launchFloor.has_value())
    {
        float const makespanMs = std::max(coldMs, launchFloor->makespanMedianMs);
        return {launchFloor->sampleCount, makespanMs, makespanMs, std::max(makespanMs, launchFloor->makespanP95Ms),
            launchFloor->uncertaintyMs};
    }
    return {0U, coldMs, coldMs, coldMs, mConfig.globalCostModelConfig.coldStartUncertaintyMs};
}

PhaseGlobalCostEstimate PhaseQueueScheduler::estimateGlobalPrefillDrainCost(
    int32_t batchSize, int32_t promptTokens, PhasePrefillClass prefillClass) const
{
    check::check(batchSize > 0, "A future prefill drain estimate requires a positive batch size");
    check::check(promptTokens > 0, "A future prefill drain estimate requires positive prompt tokens");
    int32_t const maxChunk = mConfig.supportsChunkedPrefill && mConfig.maxPrefillChunkTokens > 0
        ? mConfig.maxPrefillChunkTokens
        : promptTokens;
    PhaseGlobalCostEstimate result;
    result.makespanP95Ms = 0.0F;
    for (int32_t offset{}; offset < promptTokens;)
    {
        int32_t const chunk = std::min(maxChunk, promptTokens - offset);
        PhaseGlobalCostEstimate const turn = estimateGlobalPrefillCost(batchSize, chunk, offset, prefillClass);
        result.sampleCount
            = result.sampleCount == 0U ? turn.sampleCount : std::min(result.sampleCount, turn.sampleCount);
        result.referenceWorkMedianMs += turn.referenceWorkMedianMs;
        result.makespanMedianMs += turn.makespanMedianMs;
        result.makespanP95Ms += turn.makespanP95Ms;
        result.uncertaintyMs += turn.uncertaintyMs;
        offset += chunk;
    }
    return result;
}

void PhaseQueueScheduler::setNextGlobalAction(
    PhaseGlobalActionCandidate candidate, uint64_t planId, uint64_t snapshotEpoch)
{
    check::check(!mNextGlobalAction.has_value(), "A global P/D action is already pending dispatch");
    check::check(!mNextGlobalDispatchPlan.has_value(), "A global P/D execution lease is already pending dispatch");
    check::check(candidate.key.kind == PhaseGlobalActionKind::kPrefill
            || candidate.key.kind == PhaseGlobalActionKind::kDecode
            || candidate.key.kind == PhaseGlobalActionKind::kPrefillDecode,
        "External global selection is not a P/D dispatch action");
    if (candidate.candidateId == 0U)
    {
        phaseGlobalFinalizeCandidate(candidate);
    }
    check::check(planId == 0U || planId > mGlobalPlanSequence, "A global dispatch plan ID is stale");
    check::check(
        snapshotEpoch == 0U || snapshotEpoch > mGlobalSnapshotEpoch, "A global dispatch snapshot epoch is stale");
    planId = planId > 0U ? planId : mGlobalPlanSequence + 1U;
    snapshotEpoch = snapshotEpoch > 0U ? snapshotEpoch : mGlobalSnapshotEpoch + 1U;
    mGlobalPlanSequence = planId;
    mGlobalSnapshotEpoch = snapshotEpoch;
    mNextGlobalDispatchPlan = phaseGlobalDispatchPlan(planId, snapshotEpoch, candidate);
    mNextGlobalAction = std::move(candidate);
}

void PhaseQueueScheduler::setGlobalMemoryHorizonSupplier(
    std::function<PhaseActionMemoryHorizon(PhaseGlobalActionKey const&, std::vector<uint64_t> const& requestIds)>
        supplier)
{
    check::check(empty() && mActiveRequestIds.empty() && mInFlightRequestIds.empty(),
        "Global memory supplier can only change while the scheduler is idle");
    mConfig.globalMemoryHorizonSupplier = std::move(supplier);
}

void PhaseQueueScheduler::setGlobalExecutionVariantSupplier(
    std::function<PhaseExecutionVariant(PhaseGlobalActionKey const& key, int32_t primaryTokenCount)> supplier)
{
    check::check(empty() && mActiveRequestIds.empty() && mInFlightRequestIds.empty(),
        "Global execution-variant supplier can only change while the scheduler is idle");
    mGlobalExecutionVariantSupplier = std::move(supplier);
}

size_t PhaseQueueScheduler::decodeAdmissionLimitForTpot(double targetUs, int32_t maxContextLength) const noexcept
{
    if (targetUs <= 0.0)
    {
        return static_cast<size_t>(mConfig.maxDecodeBatchSize);
    }
    if (mConfig.enableMeasuredDecodeBatching
        && measuredDecodeP95(mConfig.maxDecodeBatchSize, maxContextLength).has_value())
    {
        size_t limit{1U};
        for (int32_t rows = 1; rows <= mConfig.maxDecodeBatchSize; ++rows)
        {
            std::optional<float> const cost = measuredDecodeP95(rows, maxContextLength);
            if (cost.has_value() && static_cast<double>(*cost) * 1000.0 <= targetUs)
            {
                limit = static_cast<size_t>(rows);
            }
        }
        return limit;
    }
    if (mConfig.decodeBatchCosts.empty())
    {
        return static_cast<size_t>(mConfig.maxDecodeBatchSize);
    }
    size_t limit{1U};
    for (int32_t rows = 1; rows <= mConfig.maxDecodeBatchSize; ++rows)
    {
        PhaseDecodeBatchCost const* selected{};
        for (PhaseDecodeBatchCost const& cost : mConfig.decodeBatchCosts)
        {
            int64_t const totalLimit = cost.maxTotalContextTokens > 0
                ? cost.maxTotalContextTokens
                : static_cast<int64_t>(cost.batchSize) * cost.maxContextLength;
            int64_t const requiredTotal = static_cast<int64_t>(rows) * std::max(1, maxContextLength);
            if (cost.batchSize < rows || cost.maxContextLength < maxContextLength || totalLimit < requiredTotal)
            {
                continue;
            }
            if (selected == nullptr || cost.batchSize < selected->batchSize
                || (cost.batchSize == selected->batchSize && cost.maxContextLength < selected->maxContextLength)
                || (cost.batchSize == selected->batchSize && cost.maxContextLength == selected->maxContextLength
                    && cost.p95GpuMs < selected->p95GpuMs))
            {
                selected = &cost;
            }
        }
        if (selected != nullptr && static_cast<double>(selected->p95GpuMs) * 1000.0 <= targetUs)
        {
            limit = static_cast<size_t>(rows);
        }
    }
    return limit;
}

PhaseDispatchPlan PhaseQueueScheduler::next()
{
    auto const decisionStart = std::chrono::steady_clock::now();
    refreshExternalDrainPreference();
    PhaseQueueSnapshot const state = snapshot();
    bool const profileFreeProduction
        = mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive && !mConfig.enableExternalDrainPreference;
    PhaseDispatchKind const baseline = profileFreeProduction ? PhaseDispatchKind::kNone : legacyQueueDecision(state);
    bool drainPreferenceApplied{};
    PhaseDispatchKind const legacyKind = profileFreeProduction
        ? PhaseDispatchKind::kNone
        : applyExternalDrainPreference(state, baseline, drainPreferenceApplied);
    PhaseDispatchKind kind = legacyKind;
    PhaseDispatchPlan plan;
    std::optional<PhaseGlobalActionCandidate> appliedGlobalAction;
    // A profile-free policy has no decision to make when exactly one local
    // phase is runnable. Avoid cloning the scheduler to preview the only
    // feasible mechanism batch. Memory-horizon users still pass through the
    // feasibility filter unless admission already reserved every dispatch
    // allocation, and an externally staged E/P/D action still owns the
    // dispatch boundary through mNextGlobalAction.
    bool const singleLocalPhase = (state.prefillQueued > 0U) != (state.decodeQueued > 0U);
    bool const workConservingSinglePhase = mConfig.elideVacuousGlobalDecisions
        && mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive && singleLocalPhase
        && (!mConfig.globalMemoryHorizonSupplier || mConfig.globalDispatchUsesPreReservedMemory)
        && !mGlobalWarmupProbeMode && !mNextGlobalAction.has_value();
    if (mNextGlobalAction.has_value())
    {
        check::check(mNextGlobalDispatchPlan.has_value(), "Global P/D action is missing its execution lease");
        PhaseGlobalActionCandidate const global = std::move(*mNextGlobalAction);
        PhaseGlobalDispatchPlan const globalPlan = std::move(*mNextGlobalDispatchPlan);
        mNextGlobalAction.reset();
        mNextGlobalDispatchPlan.reset();
        appliedGlobalAction = global;
        plan.globalCandidate = global;
        plan.globalDecisionEvaluated = true;
        plan.globalDecisionApplied = true;
        plan.globalSelectedAction = global.key;
        plan.globalReferenceWorkMs = global.referenceWorkUs / 1000.0;
        plan.contextualPdFeatures = global.contextualPdFeatures;
        plan.contextualPdFeatureValid = global.contextualPdFeatureValid;
        plan.contextualPdExploration = global.contextualPdExploration;
        plan.contextualPdMean = global.contextualPdMean;
        plan.contextualPdUncertainty = global.contextualPdUncertainty;
        plan.contextualPdLowerConfidenceBound = global.contextualPdLowerConfidenceBound;
        double const predictedMakespanUs
            = global.predictedMakespanUs > 0.0 ? global.predictedMakespanUs : global.predictedBlockingUs;
        plan.globalServiceCompression
            = global.referenceWorkUs / std::max(predictedMakespanUs, std::numeric_limits<double>::epsilon());
        plan.globalSafeProbe = global.calibrationProbe || (global.safeProbeEligible && !global.overlapCostKnown);
        plan.globalCandidateId = global.candidateId;
        plan.hostSchedulerDecisionUs = global.hostDecisionUs;
        plan.globalPlanId = globalPlan.planId;
        plan.globalSnapshotEpoch = globalPlan.snapshotEpoch;
        plan.globalAllowedOutstanding = globalPlan.allowedOutstanding;
        switch (global.key.kind)
        {
        case PhaseGlobalActionKind::kPrefill: kind = PhaseDispatchKind::kPrefill; break;
        case PhaseGlobalActionKind::kDecode: kind = PhaseDispatchKind::kDecode; break;
        case PhaseGlobalActionKind::kPrefillDecode: kind = PhaseDispatchKind::kOverlap; break;
        default: break;
        }
        ++mTelemetry.globalActiveDecisionCount;
        drainPreferenceApplied = false;
    }
    else if (workConservingSinglePhase)
    {
        // Keep mechanism formation, canonical row ordering, and runtime cost
        // observation unchanged. Only the vacuous policy comparison is elided.
        kind = state.prefillQueued > 0U ? PhaseDispatchKind::kPrefill : PhaseDispatchKind::kDecode;
        drainPreferenceApplied = false;
    }
    else if (std::optional<GlobalQueueSelection> const global = selectGlobalQueueAction(state, true, true, true))
    {
        plan.globalDecisionEvaluated = true;
        plan.globalSelectedAction = global->candidate.key;
        plan.globalDecisionReason = global->decision.reason;
        plan.globalPredictedViolationUs = global->decision.predictedViolationUs;
        plan.globalServiceCompression = global->decision.serviceCompression;
        plan.globalReferenceWorkMs = global->candidate.referenceWorkUs / 1000.0;
        plan.contextualPdFeatures = global->candidate.contextualPdFeatures;
        plan.contextualPdFeatureValid = global->candidate.contextualPdFeatureValid;
        plan.contextualPdExploration = global->candidate.contextualPdExploration;
        plan.contextualPdMean = global->candidate.contextualPdMean;
        plan.contextualPdUncertainty = global->candidate.contextualPdUncertainty;
        plan.contextualPdLowerConfidenceBound = global->candidate.contextualPdLowerConfidenceBound;
        plan.globalSafeProbe = global->safeProbe;
        if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive)
        {
            kind = global->kind;
            appliedGlobalAction = global->candidate;
            plan.globalCandidate = global->candidate;
            plan.globalDecisionApplied = true;
            plan.globalCandidateId = global->candidate.candidateId;
            PhaseGlobalDispatchPlan const globalPlan
                = phaseGlobalDispatchPlan(++mGlobalPlanSequence, ++mGlobalSnapshotEpoch, global->candidate);
            plan.globalPlanId = globalPlan.planId;
            plan.globalSnapshotEpoch = globalPlan.snapshotEpoch;
            plan.globalAllowedOutstanding = globalPlan.allowedOutstanding;
            ++mTelemetry.globalActiveDecisionCount;
            drainPreferenceApplied = false;
        }
    }
    else if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive)
    {
        kind = PhaseDispatchKind::kNone;
        plan.globalDecisionEvaluated = true;
        plan.globalDecisionApplied = true;
        plan.globalDecisionReason = PhaseGlobalDecisionReason::kNoHardFeasibleCandidate;
        drainPreferenceApplied = false;
    }
    check::check(kind != PhaseDispatchKind::kPrefill || state.prefillQueued > 0,
        "Scheduling policy selected an empty prefill queue");
    check::check(kind != PhaseDispatchKind::kDecode || state.decodeQueued > 0,
        "Scheduling policy selected an empty decode queue");
    check::check(kind != PhaseDispatchKind::kOverlap || (state.prefillQueued > 0 && state.decodeQueued > 0),
        "Scheduling policy selected overlap without work in both queues");

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
    plan.externalEncoderActive = mExternalEncoderActive;
    plan.concurrentPrefillActive = kind == PhaseDispatchKind::kOverlap;
    plan.plannedDecodeBatchSize = kind == PhaseDispatchKind::kDecode || kind == PhaseDispatchKind::kOverlap
        ? selectDecodeBatchSize(
              state, plan.concurrentPrefillActive, plan.predictedDecodeDrainGpuMs, plan.predictedDecodeDrainTurns)
        : 0;
    if (plan.plannedDecodeBatchSize > 0)
    {
        std::tie(plan.plannedDecodeContextTokens, plan.plannedDecodeMaxContextLength)
            = decodeCandidateShape(plan.plannedDecodeBatchSize);
        plan.predictedDecodeReplacementRows = decodeCandidateReplacementRows(plan.plannedDecodeBatchSize);
    }
    if (kind == PhaseDispatchKind::kPrefill || kind == PhaseDispatchKind::kOverlap)
    {
        if (appliedGlobalAction.has_value())
        {
            plan.prefillBatch = popGlobalCandidateBatch(mPrefillQueue, *appliedGlobalAction, true, true, plan);
        }
        else
        {
            int32_t const overlapPrefillBatchSize = mConfig.maxOverlapPrefillBatchSize > 0
                ? mConfig.maxOverlapPrefillBatchSize
                : mConfig.maxPrefillBatchSize;
            int32_t const prefillBatchSize
                = kind == PhaseDispatchKind::kOverlap ? overlapPrefillBatchSize : mConfig.maxPrefillBatchSize;
            plan.prefillBatch = popBatch(mPrefillQueue, prefillBatchSize, true, state, plan);
        }
    }
    if (kind == PhaseDispatchKind::kOverlap && plan.prefillDeferredForTpot)
    {
        plan.kind = PhaseDispatchKind::kDecode;
        plan.concurrentPrefillActive = false;
    }
    if (kind == PhaseDispatchKind::kDecode || kind == PhaseDispatchKind::kOverlap)
    {
        plan.decodeBatch = appliedGlobalAction.has_value()
            ? popGlobalCandidateBatch(mDecodeQueue, *appliedGlobalAction,
                  appliedGlobalAction->key.kind == PhaseGlobalActionKind::kDecode, false, plan)
            : popBatch(mDecodeQueue, plan.plannedDecodeBatchSize, false, state, plan);
    }
    if (appliedGlobalAction.has_value())
    {
        auto requestIds = [](std::vector<PhaseWorkItem> const& batch) {
            std::vector<uint64_t> result;
            result.reserve(batch.size());
            for (PhaseWorkItem const& item : batch)
            {
                result.push_back(item.requestId);
            }
            return result;
        };
        auto stableSlotIds = [](std::vector<PhaseWorkItem> const& batch) {
            std::vector<int32_t> result;
            result.reserve(batch.size());
            for (PhaseWorkItem const& item : batch)
            {
                result.push_back(item.kvSlotId);
            }
            return result;
        };
        std::vector<uint64_t> const actualPrimary = appliedGlobalAction->key.kind == PhaseGlobalActionKind::kDecode
            ? requestIds(plan.decodeBatch)
            : requestIds(plan.prefillBatch);
        std::vector<uint64_t> const actualSecondary
            = appliedGlobalAction->key.kind == PhaseGlobalActionKind::kPrefillDecode ? requestIds(plan.decodeBatch)
                                                                                     : std::vector<uint64_t>{};
        std::vector<int32_t> const actualPrimarySlots = appliedGlobalAction->key.kind == PhaseGlobalActionKind::kDecode
            ? stableSlotIds(plan.decodeBatch)
            : stableSlotIds(plan.prefillBatch);
        std::vector<int32_t> const actualSecondarySlots
            = appliedGlobalAction->key.kind == PhaseGlobalActionKind::kPrefillDecode ? stableSlotIds(plan.decodeBatch)
                                                                                     : std::vector<int32_t>{};
        plan.globalCandidateParity = actualPrimary == appliedGlobalAction->primaryRequestIds
            && actualSecondary == appliedGlobalAction->secondaryRequestIds
            && actualPrimarySlots == appliedGlobalAction->primaryStableSlotIds
            && actualSecondarySlots == appliedGlobalAction->secondaryStableSlotIds;
        plan.globalActionFidelity
            = phaseExecutionSetForAction(appliedGlobalAction->key.kind) == plan.globalAllowedOutstanding;
        if (!plan.globalCandidateParity)
        {
            ++mTelemetry.globalCandidateParityViolationCount;
        }
        check::check(plan.globalCandidateParity, "Selected global candidate changed before dispatch");
        if (!plan.globalActionFidelity)
        {
            ++mTelemetry.globalActionFidelityViolationCount;
        }
        check::check(plan.globalActionFidelity, "Global P/D execution lease does not match the selected action");
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
    plan.hostSchedulerDecisionUs
        += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - decisionStart).count();
    return plan;
}

PhaseDispatchKind PhaseQueueScheduler::legacyQueueDecision(PhaseQueueSnapshot const& state) const
{
    if (mConfig.enablePrefillTtftHardGuard && state.prefillQueued > 0U && state.prefillMinTtftSlackUs <= 0.0)
    {
        return PhaseDispatchKind::kPrefill;
    }
    return mConfig.metricsPolicy
        ? mConfig.metricsPolicy(state, mTelemetry)
        : (mConfig.enableMetricsPolicy ? metricsDecision(state, mTelemetry)
                                       : (mConfig.policy ? mConfig.policy(state) : defaultDecision(state)));
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
            auto const lhsQueued = mQueuedSince.at(lhs->requestId);
            auto const rhsQueued = mQueuedSince.at(rhs->requestId);
            if (lhsQueued != rhsQueued)
            {
                return lhsQueued < rhsQueued;
            }
            return lhs->requestId < rhs->requestId;
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

int32_t PhaseQueueScheduler::selectDecodeBatchSize(PhaseQueueSnapshot const& state, bool concurrentPrefill,
    float& predictedDrainGpuMs, int32_t& predictedDrainTurns) const
{
    int32_t const available = std::min<int32_t>(mConfig.maxDecodeBatchSize, static_cast<int32_t>(state.decodeQueued));
    bool const hasDecodeCostSource = mConfig.enableMeasuredDecodeBatching || !mConfig.decodeBatchCosts.empty();
    if (available <= 1 || !mConfig.enableDynamicDecodeBatching || !hasDecodeCostSource)
    {
        predictedDrainTurns = available > 0 ? 1 : 0;
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
    bool measuredCoverage{};
    if (mConfig.enableMeasuredDecodeBatching)
    {
        for (int32_t rows = 1; rows <= available; ++rows)
        {
            std::optional<float> const cost = measuredDecodeP95(rows, maxContextLengths[static_cast<size_t>(rows)]);
            if (!cost.has_value())
            {
                continue;
            }
            candidates.push_back({rows, *cost, maxContextLengths[static_cast<size_t>(rows)],
                contextTotals[static_cast<size_t>(rows)], rows});
        }
        measuredCoverage = std::any_of(candidates.begin(), candidates.end(),
            [available](Candidate const& candidate) { return candidate.batchSize == available; });
        if (!measuredCoverage)
        {
            // Sparse portable or node-local coverage is not evidence that a
            // smaller cohort is better. Fall back to the compatibility table
            // when present, otherwise preserve largest-available dispatch.
            candidates.clear();
            if (mConfig.decodeBatchCosts.empty())
            {
                predictedDrainTurns = available > 0 ? 1 : 0;
                return available;
            }
        }
    }
    int32_t coveringConfiguredBatch{std::numeric_limits<int32_t>::max()};
    int32_t largestConfiguredBatch{};
    if (!measuredCoverage)
    {
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
                *existing
                    = {dispatchedBatchSize, cost.p95GpuMs, cost.maxContextLength, totalContextLimit, cost.batchSize};
            }
        }
    }
    if (candidates.empty())
    {
        predictedDrainTurns = available > 0 ? 1 : 0;
        return available;
    }
    bool const largestShapeCovered = measuredCoverage
        || std::any_of(candidates.begin(), candidates.end(), [coveringConfiguredBatch](Candidate const& candidate) {
               return candidate.profiledBatchLimit == coveringConfiguredBatch;
           });
    if (!largestShapeCovered)
    {
        // A smaller batch measured at this context is not evidence that a
        // larger runnable batch is unsafe. Preserve legacy largest-available
        // behavior instead of creating a low-BS backlog from sparse profiles.
        predictedDrainTurns = available > 0 ? 1 : 0;
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

    if (mDecodeComponentObservationActive)
    {
        for (Candidate& candidate : candidates)
        {
            std::optional<float> const observed = decodeComponentP95(candidate.batchSize,
                maxContextLengths[static_cast<size_t>(candidate.batchSize)], mExternalEncoderActive, concurrentPrefill);
            if (!observed.has_value())
            {
                continue;
            }
            float const lower = candidate.p95GpuMs * (1.0F - mConfig.decodeComponentMaxAdjustmentRatio);
            bool const contended = mExternalEncoderActive || concurrentPrefill;
            float const upper = candidate.p95GpuMs
                * (contended ? mConfig.decodeContentionCostMaxMultiplier
                             : 1.0F + mConfig.decodeComponentMaxAdjustmentRatio);
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

    std::vector<float> minimumTailCostMs(static_cast<size_t>(available) + 1U, std::numeric_limits<float>::infinity());
    std::vector<int32_t> minimumTailTurns(static_cast<size_t>(available) + 1U, std::numeric_limits<int32_t>::max());
    minimumTailCostMs.front() = 0.0F;
    minimumTailTurns.front() = 0;
    for (int32_t rows = 1; rows <= available; ++rows)
    {
        for (Candidate const& candidate : candidates)
        {
            int32_t const turnRows = std::min(rows, candidate.batchSize);
            size_t const tailIndex = static_cast<size_t>(rows - turnRows);
            if (!std::isfinite(minimumTailCostMs[tailIndex]))
            {
                continue;
            }
            float const costMs = candidate.p95GpuMs + minimumTailCostMs[tailIndex];
            int32_t const turns = 1 + minimumTailTurns[tailIndex];
            size_t const rowsIndex = static_cast<size_t>(rows);
            if (costMs < minimumTailCostMs[rowsIndex]
                || (costMs == minimumTailCostMs[rowsIndex] && turns < minimumTailTurns[rowsIndex]))
            {
                minimumTailCostMs[rowsIndex] = costMs;
                minimumTailTurns[rowsIndex] = turns;
            }
        }
    }
    auto estimatedDrainCostMs = [&](Candidate const& candidate) {
        int32_t const remainingRows = available - candidate.batchSize;
        return candidate.p95GpuMs + minimumTailCostMs[static_cast<size_t>(std::max(remainingRows, 0))];
    };
    auto estimatedDrainTurns = [&](Candidate const& candidate) {
        int32_t const remainingRows = available - candidate.batchSize;
        return 1 + minimumTailTurns[static_cast<size_t>(std::max(remainingRows, 0))];
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
        predictedDrainGpuMs = estimatedDrainCostMs(*selected);
        predictedDrainTurns = estimatedDrainTurns(*selected);
        return selected->batchSize;
    }

    // No shape can drain the current queue before its remaining deadline.
    // Minimize total recovery time instead of issuing the shortest first turn
    // and leaving a more expensive remainder behind.
    predictedDrainGpuMs = estimatedDrainCostMs(*minimumDrainCandidate);
    predictedDrainTurns = estimatedDrainTurns(*minimumDrainCandidate);
    return minimumDrainCandidate->batchSize;
}

std::optional<float> PhaseQueueScheduler::measuredDecodeP95(int32_t batchSize, int32_t maxContextLength) const
{
    if (!mConfig.enableMeasuredDecodeBatching || batchSize <= 0 || maxContextLength < 0)
    {
        return std::nullopt;
    }
    int32_t const contextBucketTokens = std::max(1, mConfig.runtimeDecodeContextBucketTokens);
    int32_t const contextBucket = (maxContextLength + contextBucketTokens - 1) / contextBucketTokens;
    PhaseGlobalActionKey key{PhaseGlobalActionKind::kDecode, batchSize, 0, 1, contextBucket, 0};
    key.executionVariant
        = mGlobalExecutionVariantSupplier ? mGlobalExecutionVariantSupplier(key, 0) : PhaseExecutionVariant::kEager;
    std::optional<PhaseGlobalCostEstimate> estimate = mRuntimeCostTracker->trustedEstimate(key);
    if (!estimate.has_value())
    {
        estimate = mRuntimeCostTracker->trustedEstimatePrimaryBatchCoveringContext(key);
    }
    if (!estimate.has_value())
    {
        return std::nullopt;
    }
    return std::max(estimate->makespanP95Ms, estimate->makespanMedianMs + estimate->uncertaintyMs);
}

std::optional<float> PhaseQueueScheduler::measuredPrefillP95(int32_t batchSize, int32_t chunkLength,
    int32_t maxPastKVLength, PhasePrefillClass prefillClass, int32_t usefulTokens) const
{
    if (!mConfig.enableMeasuredPrefillBatching || batchSize <= 0 || chunkLength <= 0 || maxPastKVLength < 0)
    {
        return std::nullopt;
    }
    int32_t const contextBucketTokens = std::max(1, mConfig.runtimeDecodeContextBucketTokens);
    int32_t const contextBucket = (maxPastKVLength + contextBucketTokens - 1) / contextBucketTokens;
    PhaseGlobalActionKey key{PhaseGlobalActionKind::kPrefill, batchSize, 0, chunkLength, contextBucket, 0};
    key.primaryWorkClass = static_cast<int32_t>(prefillClass);
    key.executionVariant = mGlobalExecutionVariantSupplier ? mGlobalExecutionVariantSupplier(key, usefulTokens)
                                                           : PhaseExecutionVariant::kEager;
    std::optional<PhaseGlobalCostEstimate> estimate = mRuntimeCostTracker->trustedEstimate(key);
    if (!estimate.has_value())
    {
        estimate = mRuntimeCostTracker->trustedEstimatePrimaryBatchCoveringContext(key);
    }
    if (!estimate.has_value())
    {
        return std::nullopt;
    }
    return std::max(estimate->makespanP95Ms, estimate->makespanMedianMs + estimate->uncertaintyMs);
}

std::optional<float> PhaseQueueScheduler::decodeComponentP95(
    int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive) const
{
    return mRuntimeCostTracker->decodeP95(batchSize, maxContextLength, encoderActive, prefillActive);
}

std::optional<float> PhaseQueueScheduler::estimateGlobalDecodeComponentP95(
    int32_t batchSize, int32_t maxContextLength, bool prefillActive) const
{
    std::optional<float> result = decodeComponentP95(batchSize, maxContextLength, false, prefillActive);
    if (!result.has_value())
    {
        result = mRuntimeCostTracker->decodeCoveringP95(batchSize, maxContextLength, false, prefillActive);
    }
    return result;
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

PhaseGlobalActionKey PhaseQueueScheduler::globalActionKey(PhaseDispatchMetrics const& metrics) const noexcept
{
    int32_t const contextBucketTokens = std::max(1, mConfig.runtimeDecodeContextBucketTokens);
    auto contextBucket = [contextBucketTokens](int32_t tokens) {
        return (std::max(0, tokens) + contextBucketTokens - 1) / contextBucketTokens;
    };
    int32_t const chunkLength
        = metrics.prefillBatchSize > 0 ? std::max(1, metrics.prefillPaddedTokens / metrics.prefillBatchSize) : 1;
    if (metrics.kind == PhaseDispatchKind::kOverlap)
    {
        PhaseGlobalActionKey key{PhaseGlobalActionKind::kPrefillDecode, metrics.prefillBatchSize,
            metrics.decodeBatchSize, chunkLength, contextBucket(metrics.prefillPastKVMax),
            contextBucket(metrics.plannedDecodeMaxContextLength), metrics.globalExecutionVariant};
        key.primaryWorkClass = static_cast<int32_t>(metrics.prefillClass);
        key.residualAugmentation = metrics.globalDecisionApplied
            && metrics.globalSelectedAction.kind == PhaseGlobalActionKind::kPrefillDecode
            && metrics.globalSelectedAction.residualAugmentation;
        key.residualAnchor
            = key.residualAugmentation ? metrics.globalObservedResidualAnchor : PhaseGlobalResidualAnchor::kNone;
        return key;
    }
    if (metrics.kind == PhaseDispatchKind::kPrefill)
    {
        PhaseGlobalActionKey key{PhaseGlobalActionKind::kPrefill, metrics.prefillBatchSize, 0, chunkLength,
            contextBucket(metrics.prefillPastKVMax), 0, metrics.globalExecutionVariant};
        key.primaryWorkClass = static_cast<int32_t>(metrics.prefillClass);
        return key;
    }
    if (metrics.kind == PhaseDispatchKind::kDecode)
    {
        return {PhaseGlobalActionKind::kDecode, metrics.decodeBatchSize, 0, 1,
            contextBucket(metrics.plannedDecodeMaxContextLength), 0, metrics.globalExecutionVariant};
    }
    return {};
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
    if (mDecodeComponentObservationActive && metrics.decodeBatchSize > 0 && metrics.decodeGpuMs > 0.0F
        && metrics.plannedDecodeMaxContextLength > 0)
    {
        mRuntimeCostTracker->observeDecode(metrics.decodeBatchSize, metrics.plannedDecodeMaxContextLength,
            metrics.externalEncoderActive, metrics.concurrentPrefillActive, metrics.decodeGpuMs);
        ++mTelemetry.runtimeDecodeCostSampleCount;
        mTelemetry.encoderContendedDecodeCostSampleCount += metrics.externalEncoderActive ? 1U : 0U;
        mTelemetry.prefillContendedDecodeCostSampleCount += metrics.concurrentPrefillActive ? 1U : 0U;
        mTelemetry.runtimeDecodeCostBucketCount = mRuntimeCostTracker->decodeBucketCount();
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
        if (!mRecentDecodeTpotUs.unique())
        {
            mRecentDecodeTpotUs = std::make_shared<RecentDecodeTpot>(*mRecentDecodeTpotUs);
        }
        double const sampleUs = metrics.decodeQueueWaitUs + static_cast<double>(metrics.decodeGpuMs) * 1000.0;
        mRecentDecodeTpotUs->push_back(sampleUs);
        if (mRecentDecodeTpotUs->size() > mConfig.tpotHysteresisWindow)
        {
            mRecentDecodeTpotUs->pop_front();
        }
        ++mTelemetry.decodeTpotSampleCount;
        if (mRecentDecodeTpotUs->size() >= mConfig.minTpotHysteresisSamples)
        {
            std::vector<double> ordered(mRecentDecodeTpotUs->begin(), mRecentDecodeTpotUs->end());
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
    if (metrics.globalDecisionApplied && !metrics.globalActionFidelity)
    {
        ++mTelemetry.globalActionFidelityViolationCount;
    }
    if (mConfig.globalSchedulerMode != PhaseGlobalSchedulerMode::kDisabled && metrics.makespanGpuMs > 0.0F)
    {
        PhaseGlobalActionKey const observedKey = globalActionKey(metrics);
        if (metrics.globalDecisionApplied && observedKey.kind != PhaseGlobalActionKind::kNone)
        {
            ++mTelemetry.globalCostKeyObservationCount;
            bool const exactKeyParity = metrics.globalSelectedAction == observedKey;
            mTelemetry.globalCostKeyParityViolationCount += exactKeyParity ? 0U : 1U;
            if (observedKey.residualAnchor == PhaseGlobalResidualAnchor::kPrefill)
            {
                ++mTelemetry.globalResidualPrefillAnchorObservationCount;
            }
            else if (observedKey.residualAnchor == PhaseGlobalResidualAnchor::kDecode)
            {
                ++mTelemetry.globalResidualDecodeAnchorObservationCount;
            }
        }
        bool const planBoundSample = !metrics.globalDecisionApplied
            || (metrics.globalCandidateParity && metrics.globalActionFidelity
                && metrics.globalSelectedAction == observedKey);
        if (observedKey.kind != PhaseGlobalActionKind::kNone && planBoundSample)
        {
            float referenceWorkMs = metrics.prefillGpuMs + metrics.decodeGpuMs;
            if (metrics.globalReferenceWorkMs > 0.0 && metrics.globalSelectedAction == observedKey)
            {
                referenceWorkMs = static_cast<float>(metrics.globalReferenceWorkMs);
            }
            mRuntimeCostTracker->observe(observedKey, {referenceWorkMs, metrics.makespanGpuMs});
            if (observedKey.kind == PhaseGlobalActionKind::kPrefillDecode && metrics.contextualPdFeatureValid
                && referenceWorkMs > 0.0F)
            {
                double const reward = (static_cast<double>(referenceWorkMs) - metrics.makespanGpuMs)
                    / static_cast<double>(referenceWorkMs);
                PhaseContextualPairDirection const direction
                    = phaseContextualPairDirection(observedKey.kind, observedKey.residualAnchor);
                if (mRuntimeCostTracker->observeContextualDirection(direction, metrics.contextualPdFeatures, reward))
                {
                    ++mTelemetry.contextualPdObservationCount;
                }
                else
                {
                    ++mTelemetry.contextualPdRejectedObservationCount;
                }
            }
        }
    }
    PhaseContextualPdTelemetry const& contextualTelemetry = mRuntimeCostTracker->contextualPdTelemetry();
    mTelemetry.contextualPdPositiveSelectionCount = contextualTelemetry.positiveSelections;
    mTelemetry.contextualPdNegativeSelectionCount = contextualTelemetry.negativeSelections;
    mTelemetry.contextualPdExplorationCount = contextualTelemetry.explorations;
    mTelemetry.contextualPdLastReward = contextualTelemetry.lastReward;
    mTelemetry.contextualPdLastMean = contextualTelemetry.lastMean;
    mTelemetry.contextualPdLastUncertainty = contextualTelemetry.lastUncertainty;
    mTelemetry.contextualPdLastLowerConfidenceBound = contextualTelemetry.lastLowerConfidenceBound;
    ++mTelemetry.sampleCount;
    mTelemetry.lastDispatch = metrics;
}

PhaseSchedulerTelemetry const& PhaseQueueScheduler::telemetry() const noexcept
{
    return mTelemetry;
}

std::vector<PhaseGlobalOverlapCostRecord> PhaseQueueScheduler::globalCalibrationDiagnostics() const
{
    std::vector<PhaseGlobalOverlapCostRecord> result;
    result.reserve(mGlobalCalibrationKeys.size());
    for (size_t index{}; index < mGlobalCalibrationKeys.size(); ++index)
    {
        PhaseGlobalActionKey const& key = mGlobalCalibrationKeys[index];
        size_t const opportunities = mGlobalCalibrationOpportunities[index];
        result.push_back({key, mRuntimeCostTracker->overlapDiagnostic(key), opportunities,
            opportunities >= mConfig.globalCostModelConfig.overlapMinSamples});
    }
    return result;
}

void PhaseQueueScheduler::setDecodeComponentObservationActive(bool active) noexcept
{
    mDecodeComponentObservationActive = mConfig.enableDecodeComponentObservation && active;
}

void PhaseQueueScheduler::setExternalDrainPreference(PhaseDrainPreference preference) noexcept
{
    mRequestedDrainPreference = mConfig.enableExternalDrainPreference ? preference : PhaseDrainPreference::kNone;
}

void PhaseQueueScheduler::resetSchedulingHistory()
{
    check::check(empty() && mActiveRequestIds.empty() && mInFlightRequestIds.empty(),
        "Scheduling history can only be reset while the scheduler is idle");
    mTelemetry = {};
    mRecentDecodeTpotUs = std::make_shared<RecentDecodeTpot>();
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
    mGlobalDecisionSequence = 0U;
    mGlobalPlanSequence = 0U;
    mGlobalSnapshotEpoch = 0U;
    mLastGlobalSafeProbeSequence = 0U;
    mNextGlobalAction.reset();
    mNextGlobalDispatchPlan.reset();
}

void PhaseQueueScheduler::resetPolicyPosterior()
{
    check::check(empty() && mActiveRequestIds.empty() && mInFlightRequestIds.empty(),
        "Policy posterior can only be reset while the scheduler is idle");
    mRuntimeCostTracker->resetPolicyPosterior();
}

void PhaseQueueScheduler::resetExecutionCostHistory()
{
    check::check(empty() && mActiveRequestIds.empty() && mInFlightRequestIds.empty(),
        "Execution cost history can only be reset while the scheduler is idle");
    mRuntimeCostTracker->resetExecutionCostHistory();
}

void PhaseQueueScheduler::resetHistory(bool preserveRuntimeCosts)
{
    resetSchedulingHistory();
    if (!preserveRuntimeCosts)
    {
        mRuntimeCostTracker->reset();
    }
}

void PhaseQueueScheduler::setGlobalWarmupProbeMode(bool active)
{
    check::check(empty() && mActiveRequestIds.empty() && mInFlightRequestIds.empty(),
        "Global warmup probe mode can only change while the scheduler is idle");
    if (active)
    {
        mGlobalCalibrationKeys.clear();
        mGlobalCalibrationOpportunities.clear();
    }
    mGlobalWarmupProbeMode = active;
}

} // namespace rt
} // namespace trt_edgellm
