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
#include "common/cudaMacros.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace trt_edgellm::rt
{
namespace
{

//! Keep coordinator-owned E/P/D leases disjoint from P/D-only plans created
//! during engine warmup or standalone server operation.
constexpr uint64_t kTHREE_PHASE_PLAN_NAMESPACE = uint64_t{1U} << 63U;
class ScopedCudaContext
{
public:
    explicit ScopedCudaContext(CUcontext context)
        : mContext(context)
    {
        ELLM_CHECK(mContext != nullptr, "Async vision preparation requires a CUDA context");
        CUcontext current{};
        CUDA_DRIVER_CHECK(cuCtxGetCurrent(&current));
        if (current != mContext)
        {
            CUDA_DRIVER_CHECK(cuCtxPushCurrent(mContext));
            mPushed = true;
        }
    }

    ~ScopedCudaContext() noexcept
    {
        if (mPushed)
        {
            CUcontext popped{};
            static_cast<void>(cuCtxPopCurrent(&popped));
        }
    }

    ScopedCudaContext(ScopedCudaContext const&) = delete;
    ScopedCudaContext& operator=(ScopedCudaContext const&) = delete;

private:
    CUcontext mContext{};
    bool mPushed{};
};

size_t saturatedAdd(size_t left, size_t right) noexcept
{
    return right > std::numeric_limits<size_t>::max() - left ? std::numeric_limits<size_t>::max() : left + right;
}

size_t saturatedMultiply(size_t left, size_t right) noexcept
{
    if (left == 0U || right == 0U)
    {
        return 0U;
    }
    return right > std::numeric_limits<size_t>::max() / left ? std::numeric_limits<size_t>::max() : left * right;
}
} // namespace

PhaseVisionEncoderBatchChoice phaseVisionSelectEncoderBatch(
    std::vector<size_t> const& candidateInputTokens, std::vector<PhaseVisionEncoderBatchCost> const& costs)
{
    PhaseVisionEncoderBatchChoice result;
    size_t const candidateCount = candidateInputTokens.size();
    if (candidateCount == 0U)
    {
        return result;
    }
    if (costs.empty())
    {
        result.batchSize = candidateCount;
        result.coverageMiss = true;
        return result;
    }

    std::vector<size_t> prefixTokens(candidateCount + 1U);
    for (size_t index = 0; index < candidateCount; ++index)
    {
        size_t const previous = prefixTokens[index];
        size_t const tokens = candidateInputTokens[index];
        prefixTokens[index + 1U] = tokens > std::numeric_limits<size_t>::max() - previous
            ? std::numeric_limits<size_t>::max()
            : previous + tokens;
    }
    std::vector<float> minimumCost(candidateCount + 1U, std::numeric_limits<float>::infinity());
    std::vector<size_t> minimumTurns(candidateCount + 1U, std::numeric_limits<size_t>::max());
    std::vector<size_t> firstBatch(candidateCount);
    minimumCost[candidateCount] = 0.0F;
    minimumTurns[candidateCount] = 0U;
    for (size_t offset = candidateCount; offset-- > 0U;)
    {
        for (PhaseVisionEncoderBatchCost const& cost : costs)
        {
            if (cost.batchSize == 0U || offset + cost.batchSize > candidateCount)
            {
                continue;
            }
            size_t const next = offset + cost.batchSize;
            size_t const inputTokens = prefixTokens[next] == std::numeric_limits<size_t>::max()
                ? std::numeric_limits<size_t>::max()
                : prefixTokens[next] - prefixTokens[offset];
            if (cost.maxInputTokens < inputTokens || !std::isfinite(minimumCost[next]))
            {
                continue;
            }
            float const totalCost = cost.p95GpuMs + minimumCost[next];
            size_t const turns = 1U + minimumTurns[next];
            if (totalCost < minimumCost[offset]
                || (totalCost == minimumCost[offset] && cost.batchSize > firstBatch[offset]))
            {
                minimumCost[offset] = totalCost;
                minimumTurns[offset] = turns;
                firstBatch[offset] = cost.batchSize;
            }
        }
    }
    if (!std::isfinite(minimumCost.front()) || firstBatch.front() == 0U)
    {
        result.batchSize = candidateCount;
        result.coverageMiss = true;
        return result;
    }
    result.batchSize = firstBatch.front();
    result.predictedDrainGpuMs = minimumCost.front();
    result.predictedDrainTurns = minimumTurns.front();
    return result;
}

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

std::vector<size_t> phaseVisionEncoderBatchIndices(std::vector<PhaseVisionEncoderInput> const& inputs,
    size_t maxBatchSize, size_t maxMediaItems, size_t maxInputBytes, size_t maxInputTokens,
    bool requireHomogeneousGeometry, bool enableFitLookahead, size_t maxLookahead)
{
    std::vector<size_t> indices;
    indices.reserve(std::min(maxBatchSize, inputs.size()));
    size_t mediaItems{};
    size_t inputBytes{};
    size_t inputTokens{};
    size_t const candidateLimit
        = !enableFitLookahead || maxLookahead == 0 ? inputs.size() : std::min(maxLookahead, inputs.size());
    for (size_t index{}; index < candidateLimit && indices.size() < maxBatchSize; ++index)
    {
        PhaseVisionEncoderInput const& candidate = inputs[index];
        if (requireHomogeneousGeometry && index > 0 && candidate.mediaGeometry != inputs.front().mediaGeometry)
        {
            continue;
        }
        bool const mediaOverflow
            = maxMediaItems > 0 && candidate.mediaItems > maxMediaItems - std::min(mediaItems, maxMediaItems);
        bool const byteOverflow
            = maxInputBytes > 0 && candidate.inputBytes > maxInputBytes - std::min(inputBytes, maxInputBytes);
        bool const tokenOverflow
            = maxInputTokens > 0 && candidate.inputTokens > maxInputTokens - std::min(inputTokens, maxInputTokens);
        if (!indices.empty() && (mediaOverflow || byteOverflow || tokenOverflow))
        {
            if (enableFitLookahead)
            {
                continue;
            }
            break;
        }
        mediaItems += candidate.mediaItems;
        inputBytes += candidate.inputBytes;
        inputTokens += candidate.inputTokens;
        indices.push_back(index);
    }
    return indices;
}

size_t phaseVisionEncoderBatchSize(std::vector<PhaseVisionEncoderInput> const& inputs, size_t maxBatchSize,
    size_t maxMediaItems, size_t maxInputBytes, size_t maxInputTokens, bool requireHomogeneousGeometry,
    bool enableFitLookahead, size_t maxLookahead)
{
    return phaseVisionEncoderBatchIndices(inputs, maxBatchSize, maxMediaItems, maxInputBytes, maxInputTokens,
        requireHomogeneousGeometry, enableFitLookahead, maxLookahead)
        .size();
}

bool phaseVisionShouldAccumulateEncoderCredits(size_t admittedBatchSize, size_t candidateBatchSize,
    size_t targetBatchSize, double oldestWaitUs, double maxWaitUs) noexcept
{
    if (maxWaitUs <= 0.0 || oldestWaitUs >= maxWaitUs)
    {
        return false;
    }
    size_t const effectiveTarget = targetBatchSize == 0U ? candidateBatchSize : targetBatchSize;
    return admittedBatchSize < effectiveTarget;
}

bool phaseVisionShouldWaitForGlobalEncoderArrival(double predictedWaitUs, double oldestSlackUs,
    double robustFutureCriticalPathUs, double dispatchNowHorizonUs, double waitHorizonUs) noexcept
{
    if (!std::isfinite(predictedWaitUs) || predictedWaitUs <= 0.0 || !std::isfinite(robustFutureCriticalPathUs)
        || robustFutureCriticalPathUs < 0.0 || !std::isfinite(dispatchNowHorizonUs) || dispatchNowHorizonUs <= 0.0
        || !std::isfinite(waitHorizonUs) || waitHorizonUs <= 0.0 || waitHorizonUs >= dispatchNowHorizonUs)
    {
        return false;
    }
    return !std::isfinite(oldestSlackUs) || predictedWaitUs + robustFutureCriticalPathUs <= oldestSlackUs;
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
    size_t admissibleRequests, int32_t availableKVPages, float decodeTpotPressure, float decodeTpotPressureLimit,
    size_t readyBytes, size_t maxReadyBytes, double readyBytePressureRatio, bool enableDecodeProtectedDeferral,
    double maxDecodeProtectedWaitUs) noexcept
{
    if (promptTokenCounts.empty() || maxBatchSize == 0 || admissibleRequests == 0 || availableKVPages <= 0)
    {
        return {};
    }
    size_t const admissionLimit = std::min(maxBatchSize, admissibleRequests);
    bool const decodeProtected = decodeTpotPressureLimit > 0.0F && decodeTpotPressure >= decodeTpotPressureLimit;
    bool const withinDeferralBound = maxDecodeProtectedWaitUs <= 0.0 || oldestWaitUs < maxDecodeProtectedWaitUs;
    if (enableDecodeProtectedDeferral && decodeProtected && withinDeferralBound)
    {
        return {0U, PhaseVisionPrefillAdmissionReason::kDecodeDeferral};
    }
    if (!enabled)
    {
        return {phaseVisionReadyPrefillBatchSize(
                    promptTokenCounts, admissionLimit, maxBatchTokens, oldestWaitUs, batchWaitUs),
            PhaseVisionPrefillAdmissionReason::kLegacy};
    }

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
    size_t const requestedBatchSize = std::min(maxBatchSize, promptTokenCounts.size());
    if (admissionLimit < requestedBatchSize)
    {
        return {phaseVisionReadyPrefillBatchSize(promptTokenCounts, admissionLimit, maxBatchTokens, oldestWaitUs, 0.0),
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

size_t phaseVisionNextEncodedCapacity(size_t currentCapacity, size_t latencyCapacity, size_t throughputCapacity,
    bool throughputMode, double oldestVisionAgeUs, double visionTtftTargetUs, double escalationRatio,
    float decodeTpotPressure, float pressureEnterRatio, float pressureExitRatio) noexcept
{
    size_t const highCapacity = std::max(latencyCapacity, throughputCapacity);
    if (highCapacity == latencyCapacity)
    {
        return latencyCapacity;
    }
    bool const visionLate = escalationRatio > 0.0 && visionTtftTargetUs > 0.0
        && oldestVisionAgeUs >= visionTtftTargetUs * escalationRatio;
    if (!throughputMode && !visionLate)
    {
        return latencyCapacity;
    }
    bool const highCapacityActive = currentCapacity > latencyCapacity;
    if (highCapacityActive)
    {
        bool const contract = pressureEnterRatio > 0.0F && decodeTpotPressure >= pressureEnterRatio;
        return contract ? latencyCapacity : highCapacity;
    }
    bool const recoveryBlocked = pressureExitRatio > 0.0F && decodeTpotPressure > pressureExitRatio;
    return recoveryBlocked ? latencyCapacity : highCapacity;
}

float phaseVisionDecodeTpotPressure(double observedTpotUs, double targetTpotUs) noexcept
{
    if (!std::isfinite(observedTpotUs) || !std::isfinite(targetTpotUs) || observedTpotUs <= 0.0 || targetTpotUs <= 0.0)
    {
        return 0.0F;
    }
    return static_cast<float>(observedTpotUs / targetTpotUs);
}

bool phaseVisionEncoderSerializationDue(
    double oldestVisionAgeUs, double visionTtftTargetUs, double deadlineRatio, double predictedEncoderCostUs) noexcept
{
    if (oldestVisionAgeUs < 0.0 || visionTtftTargetUs <= 0.0 || deadlineRatio <= 0.0 || predictedEncoderCostUs < 0.0)
    {
        return false;
    }
    return oldestVisionAgeUs + predictedEncoderCostUs >= visionTtftTargetUs * deadlineRatio;
}

PhaseVisionEncoderDispatchDecision phaseVisionEncoderDispatchDecision(bool enabled, double oldestVisionAgeUs,
    double sinceLastForcedStartUs, double maxDeferUs, double forcedIntervalUs, double oldestTextAgeUs,
    double predictedEncoderCostUs, double textGuardAgeUs, float decodeTpotPressure, float decodeTpotPressureLimit,
    bool textPrefillInFlight, bool decodeInFlight, double prefillMinTtftSlackUs, bool prefillInFlight) noexcept
{
    if (!enabled)
    {
        return {true, PhaseVisionEncoderDispatchReason::kLegacy};
    }
    bool const forceDue = maxDeferUs > 0.0 && oldestVisionAgeUs >= maxDeferUs
        && (forcedIntervalUs == 0.0 || sinceLastForcedStartUs >= forcedIntervalUs);
    if (forceDue)
    {
        return {true, PhaseVisionEncoderDispatchReason::kAgeForced};
    }
    bool const textLate
        = textGuardAgeUs > 0.0 && oldestTextAgeUs > 0.0 && oldestTextAgeUs + predictedEncoderCostUs >= textGuardAgeUs;
    if (textPrefillInFlight || textLate)
    {
        return {false, PhaseVisionEncoderDispatchReason::kTextGuard};
    }
    bool const prefillDeadlineAtRisk = prefillMinTtftSlackUs > 0.0 && prefillMinTtftSlackUs <= predictedEncoderCostUs;
    if (prefillInFlight || prefillDeadlineAtRisk)
    {
        return {false, PhaseVisionEncoderDispatchReason::kPrefillGuard};
    }
    if (decodeInFlight || (decodeTpotPressureLimit > 0.0F && decodeTpotPressure >= decodeTpotPressureLimit))
    {
        return {false, PhaseVisionEncoderDispatchReason::kDecodeGuard};
    }
    return {true, PhaseVisionEncoderDispatchReason::kAllowed};
}

PhaseThreeCoordinator::PhaseThreeCoordinator(
    PhaseVisionAdapter& vision, IndependentPhaseAsyncServer& server, PhaseThreeCoordinatorConfig config)
    : mVision(vision)
    , mServer(server)
    , mConfig(config)
    , mGlobalScheduler(mConfig.globalSchedulerConfig)
    , mGlobalCostModel(mConfig.globalCostModelConfig)
    , mMemoryBroker(mConfig.memoryBroker)
{
    ELLM_CHECK(mConfig.maxEncodedInFlight > 0, "Three-phase encoded request capacity must be positive");
    ELLM_CHECK(
        mConfig.throughputMaxEncodedInFlight == 0 || mConfig.throughputMaxEncodedInFlight >= mConfig.maxEncodedInFlight,
        "Three-phase throughput encoded capacity cannot be below the latency capacity");
    ELLM_CHECK(mConfig.maxEncoderBatchSize > 0, "Three-phase encoder batch size must be positive");
    ELLM_CHECK(std::isfinite(mConfig.globalVisionPrefillColdStartUs) && mConfig.globalVisionPrefillColdStartUs >= 0.0,
        "Global vision-prefill cold-start cost must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.globalSafeProbeSlackMultiplier) && mConfig.globalSafeProbeSlackMultiplier >= 0.0F,
        "Global overlap safe-probe slack multiplier must be finite and non-negative");
    ELLM_CHECK(mConfig.globalDecodeContextBucketTokens > 0, "Global overlap context bucket must be positive");
    for (PhaseEncoderPrefillBatchCost const& cost : mConfig.globalEncoderPrefillCosts)
    {
        ELLM_CHECK(cost.encoderBatchSize > 0U && cost.prefillBatchSize > 0 && cost.maxEncoderInputTokens > 0U
                && cost.maxPrefillChunkLength > 0 && cost.maxPrefillPastKVLength >= 0
                && std::isfinite(cost.makespanP95GpuMs) && cost.makespanP95GpuMs > 0.0F,
            "Global E+P cost point is invalid");
    }
    for (PhaseEncoderDecodeBatchCost const& cost : mConfig.globalEncoderDecodeCosts)
    {
        ELLM_CHECK(cost.encoderBatchSize > 0U && cost.decodeBatchSize > 0 && cost.maxEncoderInputTokens > 0U
                && cost.maxDecodeContextLength > 0 && std::isfinite(cost.makespanP95GpuMs)
                && cost.makespanP95GpuMs > 0.0F,
            "Global E+D cost point is invalid");
    }
    ELLM_CHECK(mConfig.maxEncoderBatchSize <= mConfig.maxEncodedInFlight,
        "Three-phase encoder batch size cannot exceed downstream encoded capacity");
    size_t const runnerInputTokens = mVision.maxInputTokens();
    if (mConfig.maxEncoderInputTokens == 0)
    {
        mConfig.maxEncoderInputTokens = runnerInputTokens;
    }
    else if (runnerInputTokens > 0)
    {
        mConfig.maxEncoderInputTokens = std::min(mConfig.maxEncoderInputTokens, runnerInputTokens);
    }
    if (mConfig.maxPrefillBatchSize == 0)
    {
        mConfig.maxPrefillBatchSize = mConfig.maxEncoderBatchSize;
    }
    size_t const maxEncodedCapacity = std::max(mConfig.maxEncodedInFlight, mConfig.throughputMaxEncodedInFlight);
    ELLM_CHECK(mConfig.maxPrefillBatchSize <= maxEncodedCapacity,
        "Three-phase prefill batch size cannot exceed downstream encoded capacity");
    ELLM_CHECK(std::isfinite(mConfig.encoderBatchWaitUs) && mConfig.encoderBatchWaitUs >= 0.0,
        "Three-phase encoder batch wait must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.encoderCreditWaitUs) && mConfig.encoderCreditWaitUs >= 0.0,
        "Three-phase encoder credit wait must be finite and non-negative");
    ELLM_CHECK(mConfig.encoderCreditTargetBatchSize == 0
            || mConfig.encoderCreditTargetBatchSize <= mConfig.maxEncoderBatchSize,
        "Three-phase encoder credit target cannot exceed the encoder batch size");
    ELLM_CHECK(!mConfig.enableCostAwareEncoderBatching || !mConfig.encoderBatchCosts.empty(),
        "Cost-aware encoder batching requires a cost table");
    for (PhaseVisionEncoderBatchCost const& cost : mConfig.encoderBatchCosts)
    {
        ELLM_CHECK(cost.batchSize > 0U && cost.batchSize <= mConfig.maxEncoderBatchSize && cost.maxInputTokens > 0U
                && std::isfinite(cost.p95GpuMs) && cost.p95GpuMs > 0.0F,
            "Three-phase encoder cost point is invalid");
    }
    ELLM_CHECK(std::isfinite(mConfig.prefillBatchWaitUs) && mConfig.prefillBatchWaitUs >= 0.0,
        "Three-phase prefill batch wait must be finite and non-negative");
    ELLM_CHECK(
        mConfig.adaptivePrefillMinBatchSize > 0, "Three-phase adaptive prefill minimum batch size must be positive");
    ELLM_CHECK(std::isfinite(mConfig.prefillDecodeTpotPressureLimit) && mConfig.prefillDecodeTpotPressureLimit >= 0.0F,
        "Three-phase prefill decode pressure limit must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.maxDecodeProtectedPrefillWaitUs) && mConfig.maxDecodeProtectedPrefillWaitUs >= 0.0,
        "Three-phase decode-protected prefill wait must be finite and non-negative");
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
    ELLM_CHECK(std::isfinite(mConfig.lookaheadDecodeTpotPressureRecoveryLimit)
            && mConfig.lookaheadDecodeTpotPressureRecoveryLimit >= 0.0F
            && (mConfig.lookaheadDecodeTpotPressureLimit == 0.0F
                || mConfig.lookaheadDecodeTpotPressureRecoveryLimit <= mConfig.lookaheadDecodeTpotPressureLimit),
        "Three-phase lookahead decode pressure recovery limit must not exceed the contraction limit");
    ELLM_CHECK(
        std::isfinite(mConfig.encodedCapacityDecodeTpotTargetUs) && mConfig.encodedCapacityDecodeTpotTargetUs >= 0.0,
        "Three-phase encoded-capacity decode TPOT target must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.encodedCapacityMinDwellUs) && mConfig.encodedCapacityMinDwellUs >= 0.0,
        "Three-phase encoded-capacity dwell must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.encoderDispatchInitialCostUs) && mConfig.encoderDispatchInitialCostUs >= 0.0,
        "Three-phase encoder dispatch initial cost must be finite and non-negative");
    ELLM_CHECK(
        std::isfinite(mConfig.encoderDispatchCostSafetyMarginUs) && mConfig.encoderDispatchCostSafetyMarginUs >= 0.0,
        "Three-phase encoder dispatch safety margin must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.encoderDispatchTextGuardAgeUs) && mConfig.encoderDispatchTextGuardAgeUs >= 0.0,
        "Three-phase encoder text guard age must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.encoderDispatchDecodeTpotPressureLimit)
            && mConfig.encoderDispatchDecodeTpotPressureLimit >= 0.0F,
        "Three-phase encoder decode pressure limit must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.encoderDispatchMaxDeferUs) && mConfig.encoderDispatchMaxDeferUs >= 0.0,
        "Three-phase encoder max defer must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.encoderDispatchForcedIntervalUs) && mConfig.encoderDispatchForcedIntervalUs >= 0.0,
        "Three-phase encoder forced interval must be finite and non-negative");
    ELLM_CHECK(
        std::isfinite(mConfig.encoderSerializationDeadlineRatio) && mConfig.encoderSerializationDeadlineRatio > 0.0,
        "Three-phase encoder serialization deadline ratio must be finite and positive");
    ELLM_CHECK(
        mConfig.encoderSerializationMaxBurst > 0, "Three-phase encoder serialization burst size must be positive");
    ELLM_CHECK(mVision.cudaContext() == mServer.cudaContext(),
        "Encoder and LLM phase server must share one CUDA primary context");
    ELLM_CHECK(mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kDisabled
            || mServer.globalSchedulerMode() != PhaseGlobalSchedulerMode::kDisabled,
        "Global E/P/D scheduling requires the P/D global scheduler");
    if (mConfig.globalSchedulerMode != PhaseGlobalSchedulerMode::kDisabled)
    {
        mServer.setGlobalMemoryHorizonSupplier([this](PhaseGlobalActionKey const& key,
                                                   std::vector<uint64_t> const& requestIds) {
            PhaseActionMemoryHorizon horizon;
            PhaseMemoryBrokerConfig const& memory = mMemoryBroker.config();
            size_t const visionBytes = saturatedAdd(mReadyPrefillBytes, mServer.visionPayloadBytes());
            if (memory.maxManagedBytes > 0U)
            {
                size_t const committedKVBytes = memory.bytesPerKVPage > 0U
                    ? saturatedMultiply(static_cast<size_t>(memory.committedKVPages), memory.bytesPerKVPage)
                    : 0U;
                horizon.managedBytes = saturatedAdd(committedKVBytes, visionBytes);
                horizon.budgetBytes = memory.maxManagedBytes - memory.safetyReserveBytes;
            }
            else
            {
                horizon.managedBytes = visionBytes;
                horizon.budgetBytes = mConfig.maxEncodedBytes;
            }
            if (mServer.releasesVisionPrefillStorage()
                && (key.kind == PhaseGlobalActionKind::kPrefill || key.kind == PhaseGlobalActionKind::kPrefillDecode))
            {
                size_t const prefillRows
                    = std::min(requestIds.size(), static_cast<size_t>(std::max(0, key.primaryBatchSize)));
                std::vector<uint64_t> const prefillRequestIds(
                    requestIds.begin(), requestIds.begin() + static_cast<std::ptrdiff_t>(prefillRows));
                horizon.nearReclaimBytes = mServer.visionPayloadBytes(prefillRequestIds);
            }
            return horizon;
        });
    }
    mEffectiveEncodedCapacity = mConfig.maxEncodedInFlight;
    mMaxEffectiveEncodedCapacity = mEffectiveEncodedCapacity;
}

PhaseThreeSubmissionStatus PhaseThreeCoordinator::submit(
    uint64_t requestId, LLMGenerationRequest request, int32_t maxOutputTokens, PhaseSchedulingHints scheduling)
{
    if (!mRequestIds.insert(requestId).second)
    {
        return PhaseThreeSubmissionStatus::kDuplicateRequest;
    }
    auto const arrival = std::chrono::steady_clock::now();
    if (mLastVisionArrival != std::chrono::steady_clock::time_point{})
    {
        constexpr double kInterarrivalEwmaAlpha = 0.2;
        double const sampleUs = std::chrono::duration<double, std::micro>(arrival - mLastVisionArrival).count();
        if (sampleUs > 0.0)
        {
            mVisionInterarrivalEwmaUs = mVisionInterarrivalSamples == 0U
                ? sampleUs
                : kInterarrivalEwmaAlpha * sampleUs + (1.0 - kInterarrivalEwmaAlpha) * mVisionInterarrivalEwmaUs;
            ++mVisionInterarrivalSamples;
        }
    }
    mLastVisionArrival = arrival;
    scheduling = phaseVisionSchedulingHints(scheduling, mConfig.visionTtftTargetUs);
    if (scheduling.tpotTargetUs > 0.0)
    {
        mRequestTpotTargets.emplace(requestId, scheduling.tpotTargetUs);
        mTpotTargets.insert(scheduling.tpotTargetUs);
    }
    size_t const inputTokens = mVision.estimateInputTokens(request);
    size_t const estimatedPayloadBytes = mVision.estimatePayloadBytes(request);
    std::vector<int64_t> const geometry = mediaGeometry(request);
    bool prefixSubmitted{};
    if (mConfig.enablePrefixBeforeVisionPrefill)
    {
        if (auto plan = mVision.makePrefixPlan(request); plan.has_value())
        {
            if (plan->prefixTokens.size() >= mConfig.minPrefixBeforeVisionTokens)
            {
                IndependentPhaseServerSubmission const prefix = mServer.submitVisionPrefix(requestId,
                    std::move(plan->prefixTokens), plan->estimatedFinalPromptTokens, maxOutputTokens, scheduling);
                prefixSubmitted = prefix.status == IndependentPhaseServerStatus::kAdmitted;
            }
        }
    }
    mPending.push_back({requestId, std::move(request), maxOutputTokens, scheduling, inputTokens, estimatedPayloadBytes,
        geometry, prefixSubmitted});
    recordTimeline(requestId, PhaseTimelineStage::kVisionQueued);
    bool const started = mConfig.globalSchedulerMode != PhaseGlobalSchedulerMode::kActive
        && !mConfig.enableEncoderDispatchArbitration && startNextEncoder();
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
        if (pending->prefixSubmitted)
        {
            ELLM_CHECK(mServer.cancel(requestId), "Deferred vision prefix could not be cancelled");
        }
        mPending.erase(pending);
        mRequestIds.erase(requestId);
        eraseTpotTarget(requestId);
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
        ELLM_CHECK(ready->promptTokens.size() <= mReadyPrefillTokens, "Ready prefill token accounting underflow");
        mReadyPrefillTokens -= ready->promptTokens.size();
        ELLM_CHECK(ready->payloadBytes <= mReadyPrefillBytes, "Ready prefill byte accounting underflow");
        mReadyPrefillBytes -= ready->payloadBytes;
        mReadyPrefill.erase(ready);
        mDownstreamRequestBytes.erase(requestId);
        mRequestIds.erase(requestId);
        eraseTpotTarget(requestId);
        return true;
    }
    bool const cancelled = mServer.cancel(requestId);
    if (cancelled)
    {
        mRequestIds.erase(requestId);
        eraseTpotTarget(requestId);
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
    if (!mConfig.memoryBroker.enabled || mPending.empty())
    {
        mServer.setExternalDrainPreference(PhaseDrainPreference::kNone);
    }
    bool progressed = completeEncoder();
    refreshGlobalExecutionLease();
    // Consume the E+D sample before a newly dispatched D action can replace the
    // server's last completed dispatch telemetry.
    completeGlobalOverlapObservation();
    refreshEffectiveEncodedCapacity();
    refreshEncoderSerializationGate();
    bool const globalScheduling = mConfig.globalSchedulerMode != PhaseGlobalSchedulerMode::kDisabled;
    if (!globalScheduling && mEncoderSerializationGate && !mServer.arbitrationSnapshot().busy)
    {
        progressed = startNextEncoder() || progressed;
    }
    if (!globalScheduling && !mConfig.enableEncoderDispatchArbitration)
    {
        progressed = startNextEncoder() || progressed;
    }
    progressed = dispatchReadyPrefill() || progressed;
    double const minTpotTargetUs = mTpotTargets.empty() ? 0.0 : *mTpotTargets.begin();
    size_t const upstreamRequests = mPending.size() + mEncoding.size();
    mAdmissionProfilePrefillTokens = mReadyPrefillTokens + upstreamRequests * mEstimatedPromptTokens;
    mServer.setExternalPendingRequests(
        upstreamRequests + mReadyPrefill.size(), minTpotTargetUs, mRequestIds.size(), mAdmissionProfilePrefillTokens);
    progressed = (globalScheduling ? mServer.pollCompletions() : mServer.poll()) || progressed;
    refreshGlobalExecutionLease();
    refreshEffectiveEncodedCapacity();
    if (mEncoderSerializationYieldPending)
    {
        mEncoderSerializationYieldPending = false;
    }
    refreshEncoderSerializationGate();
    if (globalScheduling)
    {
        bool const globalProgress = dispatchGlobalAction();
        progressed = globalProgress || progressed;
        if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kShadow)
        {
            if (!mConfig.enableEncoderDispatchArbitration)
            {
                progressed = startNextEncoder() || progressed;
            }
            progressed = mServer.dispatchReady() || progressed;
            if (mConfig.enableEncoderDispatchArbitration)
            {
                progressed = startNextEncoder() || progressed;
            }
        }
    }
    else if (mConfig.enableEncoderDispatchArbitration)
    {
        progressed = startNextEncoder() || progressed;
    }
    if (!mEncoderPreparation.valid())
    {
        mVision.reclaimIdleStorage();
    }
    completeGlobalOverlapObservation();
    refreshGlobalExecutionLease();
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
    result.pendingPrefillReadyTokens = mReadyPrefillTokens;
    result.admissionProfilePrefillTokens = mAdmissionProfilePrefillTokens;
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
    result.lastEncoderInputBytes = mLastEncoderInputBytes;
    result.maxEncoderInputBytes = mMaxEncoderInputBytes;
    result.lastEncoderInputTokens = mLastEncoderInputTokens;
    result.maxEncoderInputTokens = mMaxEncoderInputTokens;
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
    result.decodeDeferredPrefillPeriods = mDecodeDeferredPrefillPeriods;
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
    result.encodedCapacityContractions = mEncodedCapacityContractions;
    result.encodedCapacityDwellBlocks = mEncodedCapacityDwellBlocks;
    result.decodeTpotPressure = decodeTpotPressure();
    result.encoderDispatchDeferrals = mEncoderDispatchDeferrals;
    result.encoderTextGuardDeferrals = mEncoderTextGuardDeferrals;
    result.encoderPrefillGuardDeferrals = mEncoderPrefillGuardDeferrals;
    result.encoderDecodeGuardDeferrals = mEncoderDecodeGuardDeferrals;
    result.encoderAgeForcedStarts = mEncoderAgeForcedStarts;
    result.encoderSerializedStarts = mEncoderSerializedStarts;
    result.encoderSerializationBursts = mEncoderSerializationBursts;
    result.encoderSerializationBoundaryWaits = mEncoderSerializationBoundaryWaits;
    result.encoderCreditWaitPeriods = mEncoderCreditWaitPeriods;
    result.encoderCreditAgeReleases = mEncoderCreditAgeReleases;
    result.encoderCostAwareSelections = mEncoderCostAwareSelections;
    result.encoderCostCoverageMisses = mEncoderCostCoverageMisses;
    result.lastPredictedEncoderDrainGpuMs = mLastPredictedEncoderDrainGpuMs;
    result.lastPredictedEncoderDrainTurns = mLastPredictedEncoderDrainTurns;
    result.encoderPreparationStarts = mEncoderPreparationStarts;
    result.encoderPreparationCompletions = mEncoderPreparationCompletions;
    result.lastEncoderPreparationUs = mLastEncoderPreparationUs;
    result.maxEncoderPreparationUs = mMaxEncoderPreparationUs;
    result.memoryBrokerDecisions = mMemoryBrokerDecisions;
    result.memoryBrokerEncoderReductions = mMemoryBrokerEncoderReductions;
    result.memoryBrokerBackpressure = mMemoryBrokerBackpressure;
    result.memoryBrokerIdleReclaims = mMemoryBrokerIdleReclaims;
    result.memoryBrokerPrefillPreferences = mMemoryBrokerPrefillPreferences;
    result.memoryBrokerDecodePreferences = mMemoryBrokerDecodePreferences;
    result.memoryBrokerLastPredictedBytes = mMemoryBrokerLastPredictedBytes;
    result.memoryBrokerLastReason = mMemoryBrokerLastReason;
    result.activeMemoryDrainPreference = mServer.activeDrainPreference();
    result.memoryDrainPreferenceTransitions = mServer.drainPreferenceTransitionCount();
    result.memoryDrainPreferenceAppliedDispatches = mServer.drainPreferenceAppliedDispatchCount();
    result.exclusiveEncoderBatches = mExclusiveEncoderBatches;
    result.exclusiveEncoderPrefillDeferrals = mExclusiveEncoderPrefillDeferrals;
    result.globalDecisions = mGlobalDecisions;
    result.globalShadowDisagreements = mGlobalShadowDisagreements;
    result.globalEncoderSelections = mGlobalEncoderSelections;
    result.globalEncoderPrefillSelections = mGlobalEncoderPrefillSelections;
    result.globalEncoderDecodeSelections = mGlobalEncoderDecodeSelections;
    result.globalPdSelections = mGlobalPdSelections;
    result.globalSafeProbes = mGlobalSafeProbes;
    result.globalActionFidelityViolations = mGlobalActionFidelityViolations;
    result.lastGlobalFirstTokenCriticalPathUs = mLastGlobalFirstTokenCriticalPathUs;
    result.globalEncoderArrivalWaitPeriods = mGlobalEncoderArrivalWaitPeriods;
    result.globalEncoderArrivalWaitExpirations = mGlobalEncoderArrivalWaitExpirations;
    result.lastGlobalEncoderArrivalWaitUs = mLastGlobalEncoderArrivalWaitUs;
    result.activeGlobalPlanId = mGlobalExecutionLease.has_value() ? mGlobalExecutionLease->planId : 0U;
    result.globalPlannedOutstanding = mGlobalExecutionLease.has_value()
        ? mGlobalExecutionLease->allowedOutstanding
        : PhaseExecutionSet::kNone;
    result.globalObservedOutstanding = observedGlobalExecution();
    result.lastGlobalAction = mLastGlobalAction;
    if (!mPending.empty())
    {
        result.oldestPendingAgeUs = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - mPending.front().scheduling.submittedAt)
                                        .count();
    }
    return result;
}

void PhaseThreeCoordinator::setTimelineCallback(std::function<void(PhaseTimelineEvent const&)> timelineCallback)
{
    ELLM_CHECK(empty(), "Three-phase timeline callback can only change while the coordinator is idle");
    mTimelineCallback = std::move(timelineCallback);
}

void PhaseThreeCoordinator::setEncoderBatchMetricCallback(
    std::function<void(PhaseVisionEncoderBatchMetric const&)> encoderBatchMetricCallback)
{
    ELLM_CHECK(empty(), "Three-phase encoder metric callback can only change while the coordinator is idle");
    mEncoderBatchMetricCallback = std::move(encoderBatchMetricCallback);
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
        eraseTpotTarget(completion->requestId);
        auto const downstream = mDownstreamRequestBytes.find(completion->requestId);
        if (downstream != mDownstreamRequestBytes.end())
        {
            mDownstreamRequestBytes.erase(downstream);
        }
    }
    return completion;
}

PhaseExecutionSet PhaseThreeCoordinator::observedGlobalExecution() const noexcept
{
    PhaseExecutionSet result{PhaseExecutionSet::kNone};
    if (!mEncoding.empty() || mVision.busy() || mEncoderPreparation.valid())
    {
        result = result | PhaseExecutionSet::kEncoder;
    }
    IndependentPhaseServerArbitrationSnapshot const server = mServer.arbitrationSnapshot();
    if (server.busy)
    {
        if (server.inFlightKind == PhaseDispatchKind::kPrefill || server.inFlightKind == PhaseDispatchKind::kOverlap)
        {
            result = result | PhaseExecutionSet::kPrefill;
        }
        if (server.inFlightKind == PhaseDispatchKind::kDecode || server.inFlightKind == PhaseDispatchKind::kOverlap)
        {
            result = result | PhaseExecutionSet::kDecode;
        }
    }
    return result;
}

void PhaseThreeCoordinator::refreshGlobalExecutionLease()
{
    if (!mGlobalExecutionLease.has_value())
    {
        return;
    }
    PhaseExecutionSet const observed = observedGlobalExecution();
    if (!mGlobalExecutionLease->permits(observed))
    {
        ++mGlobalActionFidelityViolations;
        ELLM_CHECK(false, "Observed phases exceed the active global execution lease");
    }
    if (observed == PhaseExecutionSet::kNone)
    {
        mGlobalExecutionLease.reset();
    }
}

PhaseGlobalDispatchPlan PhaseThreeCoordinator::beginGlobalExecutionLease(
    PhaseGlobalActionCandidate const& candidate)
{
    ELLM_CHECK(!mGlobalExecutionLease.has_value(), "A global execution lease is already active");
    PhaseGlobalDispatchPlan plan = phaseGlobalDispatchPlan(kTHREE_PHASE_PLAN_NAMESPACE | ++mGlobalPlanSequence,
        kTHREE_PHASE_PLAN_NAMESPACE | ++mGlobalSnapshotEpoch, candidate);
    ELLM_CHECK(plan.allowedOutstanding != PhaseExecutionSet::kNone, "A dispatch lease requires executable phases");
    mGlobalExecutionLease = plan;
    return plan;
}

void PhaseThreeCoordinator::validateGlobalExecutionLaunch()
{
    ELLM_CHECK(mGlobalExecutionLease.has_value(), "Global execution launch has no active lease");
    PhaseExecutionSet const observed = observedGlobalExecution();
    if (!mGlobalExecutionLease->launchMatches(observed))
    {
        ++mGlobalActionFidelityViolations;
        ELLM_CHECK(false, "Global action does not match the launched outstanding phase set");
    }
}

void PhaseThreeCoordinator::abandonGlobalExecutionLease() noexcept
{
    mGlobalExecutionLease.reset();
}

bool PhaseThreeCoordinator::dispatchGlobalAction()
{
    if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kDisabled)
    {
        return false;
    }
    refreshGlobalExecutionLease();
    if (mGlobalExecutionLease.has_value())
    {
        return false;
    }
    IndependentPhaseServerArbitrationSnapshot const serverState = mServer.arbitrationSnapshot();
    if (serverState.busy)
    {
        return false;
    }
    if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive && mServer.shouldWaitForGlobalDecodeRefill())
    {
        return false;
    }

    if (!mEncoding.empty() || mVision.busy() || mEncoderPreparation.valid())
    {
        return false;
    }

    std::optional<PhaseGlobalActionCandidate> pd;
    if (serverState.prefillQueued > 0U || serverState.decodeQueued > 0U)
    {
        pd = mServer.previewGlobalAction();
    }

    std::vector<size_t> encoderBatchIndices;
    if (!mPending.empty())
    {
        encoderBatchIndices = nextEncoderBatchIndices();
    }
    if (encoderBatchIndices.empty() && !pd.has_value())
    {
        return false;
    }
    if (encoderBatchIndices.empty())
    {
        if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive)
        {
            ++mGlobalPdSelections;
            PhaseGlobalDispatchPlan const executionPlan = beginGlobalExecutionLease(*pd);
            bool const started
                = mServer.dispatchGlobalAction(std::move(*pd), executionPlan.planId, executionPlan.snapshotEpoch);
            if (!started)
            {
                abandonGlobalExecutionLease();
                return false;
            }
            validateGlobalExecutionLaunch();
            return true;
        }
        return false;
    }

    size_t encoderInputTokens{};
    size_t encoderPayloadBytes{};
    double encoderSlackUs{std::numeric_limits<double>::infinity()};
    double encoderServiceLagUs{};
    auto const now = std::chrono::steady_clock::now();
    for (size_t const index : encoderBatchIndices)
    {
        PendingVisionRequest const& request = mPending[index];
        encoderInputTokens += request.inputTokens;
        encoderPayloadBytes
            += request.estimatedPayloadBytes > 0U ? request.estimatedPayloadBytes : mEstimatedEncodedBytes;
        double const ageUs = std::chrono::duration<double, std::micro>(now - request.scheduling.submittedAt).count();
        double const targetUs
            = request.scheduling.ttftTargetUs > 0.0 ? request.scheduling.ttftTargetUs : mConfig.visionTtftTargetUs;
        encoderSlackUs
            = std::min(encoderSlackUs, targetUs > 0.0 ? targetUs - ageUs : std::numeric_limits<double>::infinity());
        encoderServiceLagUs = std::max(encoderServiceLagUs, ageUs);
    }
    constexpr size_t kEncoderTokenBucket = 1024U;
    int32_t const encoderContextBucket
        = static_cast<int32_t>((encoderInputTokens + kEncoderTokenBucket - 1U) / kEncoderTokenBucket);
    PhaseGlobalActionKey const encoderKey{PhaseGlobalActionKind::kEncoder,
        static_cast<int32_t>(encoderBatchIndices.size()), 0, 0, encoderContextBucket, 0};

    double encoderMakespanUs = mLastEncoderGpuMs > 0.0F ? static_cast<double>(mLastEncoderGpuMs) * 1000.0
                                                        : mConfig.encoderDispatchInitialCostUs;
    double encoderUncertaintyUs = static_cast<double>(mConfig.globalCostModelConfig.coldStartUncertaintyMs) * 1000.0;
    double encoderReferenceUs = encoderMakespanUs;
    if (std::optional<PhaseGlobalCostEstimate> const online = mGlobalCostModel.estimate(encoderKey))
    {
        encoderMakespanUs = static_cast<double>(online->makespanMedianMs) * 1000.0;
        encoderUncertaintyUs = static_cast<double>(online->uncertaintyMs) * 1000.0;
        encoderReferenceUs = static_cast<double>(online->referenceWorkMedianMs) * 1000.0;
    }
    else
    {
        PhaseVisionEncoderBatchCost const* selected{};
        PhaseVisionEncoderBatchCost const* singleton{};
        for (PhaseVisionEncoderBatchCost const& cost : mConfig.encoderBatchCosts)
        {
            if (cost.maxInputTokens < encoderInputTokens)
            {
                continue;
            }
            if (cost.batchSize >= encoderBatchIndices.size()
                && (selected == nullptr || cost.batchSize < selected->batchSize
                    || (cost.batchSize == selected->batchSize && cost.p95GpuMs < selected->p95GpuMs)))
            {
                selected = &cost;
            }
            if (cost.batchSize == 1U && (singleton == nullptr || cost.maxInputTokens < singleton->maxInputTokens))
            {
                singleton = &cost;
            }
        }
        if (selected != nullptr)
        {
            encoderMakespanUs = static_cast<double>(selected->p95GpuMs) * 1000.0;
            encoderUncertaintyUs = 0.0;
            encoderReferenceUs = singleton != nullptr
                ? static_cast<double>(singleton->p95GpuMs) * 1000.0 * encoderBatchIndices.size()
                : encoderMakespanUs;
        }
    }

    size_t const estimatedPromptTokensPerRequest = std::max<size_t>(1U,
        mEstimatedPromptTokens > 0U ? mEstimatedPromptTokens
                                    : (encoderInputTokens + encoderBatchIndices.size() - 1U)
                / encoderBatchIndices.size());
    PhaseGlobalCostEstimate const visionPrefill = mServer.estimateGlobalPrefillDrainCost(
        static_cast<int32_t>(encoderBatchIndices.size()),
        static_cast<int32_t>(std::min<size_t>(estimatedPromptTokensPerRequest,
            static_cast<size_t>(std::numeric_limits<int32_t>::max()))),
        PhasePrefillClass::kExternal);
    double const visionPrefillMakespanUs = visionPrefill.makespanMedianMs > 0.0F
        ? static_cast<double>(visionPrefill.makespanMedianMs) * 1000.0
        : mConfig.globalVisionPrefillColdStartUs;
    double const visionPrefillUncertaintyUs = static_cast<double>(visionPrefill.uncertaintyMs) * 1000.0;
    mLastGlobalFirstTokenCriticalPathUs
        = encoderMakespanUs + encoderUncertaintyUs + visionPrefillMakespanUs + visionPrefillUncertaintyUs;

    PhaseGlobalActionCandidate encoder;
    encoder.key = encoderKey;
    for (size_t const index : encoderBatchIndices)
    {
        encoder.primaryRequestIds.push_back(mPending[index].requestId);
    }
    phaseGlobalFinalizeCandidate(encoder);
    encoder.predictedBlockingUs = encoderMakespanUs;
    encoder.predictedMakespanUs = encoderMakespanUs;
    encoder.uncertaintyUs = encoderUncertaintyUs;
    encoder.referenceWorkUs = encoderReferenceUs;
    encoder.requestServiceLagUs = encoderServiceLagUs;
    encoder.protectedCompletions.push_back({encoderSlackUs, encoderMakespanUs + visionPrefillMakespanUs,
        encoderUncertaintyUs + visionPrefillUncertaintyUs});
    PhaseMemoryBrokerConfig const& memoryConfig = mMemoryBroker.config();
    size_t const committedKVBytes = memoryConfig.bytesPerKVPage > 0U
        ? saturatedMultiply(static_cast<size_t>(memoryConfig.committedKVPages), memoryConfig.bytesPerKVPage)
        : 0U;
    encoder.memory.managedBytes
        = saturatedAdd(saturatedAdd(committedKVBytes, mReadyPrefillBytes), mServer.visionPayloadBytes());
    encoder.memory.allocateBytes = encoderPayloadBytes;
    encoder.memory.budgetBytes = memoryConfig.maxManagedBytes > 0U
        ? memoryConfig.maxManagedBytes - memoryConfig.safetyReserveBytes
        : mConfig.maxEncodedBytes;
    if (pd.has_value())
    {
        for (PhaseProtectedCompletion completion : pd->protectedCompletions)
        {
            completion.predictedCompletionUs += encoderMakespanUs;
            completion.uncertaintyUs += encoderUncertaintyUs;
            encoder.protectedCompletions.push_back(completion);
        }
        pd->protectedCompletions.push_back(
            {encoderSlackUs, pd->predictedMakespanUs + encoderMakespanUs + visionPrefillMakespanUs,
                pd->uncertaintyUs + encoderUncertaintyUs + visionPrefillUncertaintyUs});
    }

    std::vector<PhaseGlobalActionCandidate> candidates;
    candidates.push_back(encoder);
    if (pd.has_value())
    {
        candidates.push_back(*pd);
    }
    bool const encoderExclusive = mConfig.exclusiveEncoderInputTokenThreshold > 0
        && encoderInputTokens > mConfig.exclusiveEncoderInputTokenThreshold;
    auto addEncoderOverlap = [&](PhaseGlobalActionKind kind) {
        ELLM_CHECK(pd.has_value(), "An encoder overlap requires a P/D candidate");
        int32_t const chunkLength = kind == PhaseGlobalActionKind::kEncoderPrefill ? pd->key.chunkLength : 0;
        PhaseGlobalActionKey overlapKey{kind, static_cast<int32_t>(encoderBatchIndices.size()),
            pd->key.primaryBatchSize, chunkLength, encoderContextBucket, pd->key.primaryContextBucket};
        overlapKey.executionVariant
            = phaseExecutionVariant(false, phaseExecutionVariantUsesPrimaryGraph(pd->key.executionVariant));
        double overlapMakespanUs = encoderMakespanUs + pd->predictedMakespanUs;
        double overlapUncertaintyUs = encoderUncertaintyUs + pd->uncertaintyUs;
        bool overlapKnown{};
        if (std::optional<PhaseGlobalCostEstimate> const online = mGlobalCostModel.estimate(overlapKey))
        {
            overlapMakespanUs = static_cast<double>(online->makespanMedianMs) * 1000.0;
            overlapUncertaintyUs = static_cast<double>(online->uncertaintyMs) * 1000.0;
            overlapKnown = mGlobalCostModel.overlapEligible(overlapKey);
        }
        else
        {
            if (kind == PhaseGlobalActionKind::kEncoderPrefill)
            {
                for (PhaseEncoderPrefillBatchCost const& cost : mConfig.globalEncoderPrefillCosts)
                {
                    if (cost.encoderBatchSize >= encoderBatchIndices.size()
                        && cost.prefillBatchSize >= pd->key.primaryBatchSize
                        && cost.maxEncoderInputTokens >= encoderInputTokens
                        && cost.maxPrefillChunkLength >= pd->key.chunkLength
                        && cost.maxPrefillPastKVLength
                            >= pd->key.primaryContextBucket * mConfig.globalDecodeContextBucketTokens
                        && static_cast<double>(cost.makespanP95GpuMs) * 1000.0 < overlapMakespanUs)
                    {
                        overlapMakespanUs = static_cast<double>(cost.makespanP95GpuMs) * 1000.0;
                        overlapUncertaintyUs = 0.0;
                        overlapKnown = true;
                    }
                }
            }
            else
            {
                for (PhaseEncoderDecodeBatchCost const& cost : mConfig.globalEncoderDecodeCosts)
                {
                    if (cost.encoderBatchSize >= encoderBatchIndices.size()
                        && cost.decodeBatchSize >= pd->key.primaryBatchSize
                        && cost.maxEncoderInputTokens >= encoderInputTokens
                        && cost.maxDecodeContextLength
                            >= pd->key.primaryContextBucket * mConfig.globalDecodeContextBucketTokens
                        && static_cast<double>(cost.makespanP95GpuMs) * 1000.0 < overlapMakespanUs)
                    {
                        overlapMakespanUs = static_cast<double>(cost.makespanP95GpuMs) * 1000.0;
                        overlapUncertaintyUs = 0.0;
                        overlapKnown = true;
                    }
                }
            }
        }
        double protectedSlackUs = encoderSlackUs;
        for (PhaseProtectedCompletion const& completion : pd->protectedCompletions)
        {
            protectedSlackUs = std::min(protectedSlackUs, completion.slackUs);
        }
        double const robustSerialUs
            = encoderMakespanUs + encoderUncertaintyUs + pd->predictedMakespanUs + pd->uncertaintyUs;
        bool const probeIntervalReady = mLastGlobalSafeProbeSequence == 0U
            || mGlobalDecisionSequence - mLastGlobalSafeProbeSequence >= mConfig.globalSafeProbeInterval;
        bool const safeProbe = !overlapKnown && mConfig.globalSafeProbeSlackMultiplier > 0.0F && probeIntervalReady
            && protectedSlackUs >= static_cast<double>(mConfig.globalSafeProbeSlackMultiplier) * robustSerialUs;
        if (safeProbe)
        {
            double const optimisticMakespanUs = std::max(encoderMakespanUs, pd->predictedMakespanUs);
            overlapMakespanUs = optimisticMakespanUs;
            overlapUncertaintyUs = std::max(0.0, robustSerialUs - optimisticMakespanUs);
        }
        PhaseGlobalActionCandidate overlap;
        overlap.key = overlapKey;
        overlap.overlapCostKnown = overlapKnown;
        overlap.safeProbeEligible = safeProbe;
        overlap.predictedBlockingUs = overlapMakespanUs;
        overlap.predictedMakespanUs = overlapMakespanUs;
        overlap.uncertaintyUs = overlapUncertaintyUs;
        overlap.referenceWorkUs = encoderReferenceUs + pd->referenceWorkUs;
        overlap.requestServiceLagUs = std::max(encoderServiceLagUs, pd->requestServiceLagUs);
        overlap.memory = encoder.memory;
        overlap.memory.allocateBytes = saturatedAdd(overlap.memory.allocateBytes, pd->memory.allocateBytes);
        overlap.memory.guaranteedGrowthBytes
            = saturatedAdd(overlap.memory.guaranteedGrowthBytes, pd->memory.guaranteedGrowthBytes);
        overlap.memory.nearReclaimBytes = saturatedAdd(overlap.memory.nearReclaimBytes, pd->memory.nearReclaimBytes);
        overlap.primaryRequestIds = encoder.primaryRequestIds;
        overlap.secondaryRequestIds = pd->requestIds;
        phaseGlobalFinalizeCandidate(overlap);
        overlap.protectedCompletions.push_back({encoderSlackUs, overlapMakespanUs + visionPrefillMakespanUs,
            overlapUncertaintyUs + visionPrefillUncertaintyUs});
        for (PhaseProtectedCompletion completion : pd->protectedCompletions)
        {
            completion.predictedCompletionUs = overlapMakespanUs;
            completion.uncertaintyUs = overlapUncertaintyUs;
            overlap.protectedCompletions.push_back(completion);
        }
        candidates.push_back(std::move(overlap));
    };
    if (pd.has_value() && !encoderExclusive && !mConfig.enableAsyncEncoderPreparation)
    {
        if (pd->key.kind == PhaseGlobalActionKind::kPrefill && mConfig.enableGlobalEncoderPrefillAction)
        {
            addEncoderOverlap(PhaseGlobalActionKind::kEncoderPrefill);
        }
        else if (pd->key.kind == PhaseGlobalActionKind::kDecode)
        {
            addEncoderOverlap(PhaseGlobalActionKind::kEncoderDecode);
        }
    }

    ++mGlobalDecisionSequence;
    PhaseGlobalDecision const decision = mGlobalScheduler.select(candidates);
    ++mGlobalDecisions;
    if (!decision.selectedIndex.has_value())
    {
        return false;
    }
    PhaseGlobalActionCandidate selected = candidates[*decision.selectedIndex];
    mLastGlobalAction = selected.key.kind;
    bool const safeProbe = selected.safeProbeEligible && !selected.overlapCostKnown;
    if (safeProbe && mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive)
    {
        mLastGlobalSafeProbeSequence = mGlobalDecisionSequence;
        ++mGlobalSafeProbes;
    }
    if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kShadow)
    {
        PhaseGlobalActionKind const legacyAction = pd.has_value() ? pd->key.kind : PhaseGlobalActionKind::kEncoder;
        mGlobalShadowDisagreements += selected.key.kind != legacyAction ? 1U : 0U;
        return false;
    }

    if (selected.key.kind == PhaseGlobalActionKind::kEncoder)
    {
        static_cast<void>(beginGlobalExecutionLease(selected));
        mGlobalEncoderBatchIndices = encoderBatchIndices;
        mInFlightGlobalEncoderKey = encoderKey;
        mInFlightGlobalEncoderReferenceMs = encoderReferenceUs / 1000.0;
        ++mGlobalEncoderSelections;
        bool const started = startNextEncoder();
        if (!started)
        {
            abandonGlobalExecutionLease();
            return false;
        }
        validateGlobalExecutionLaunch();
        return true;
    }
    if (selected.key.kind == PhaseGlobalActionKind::kEncoderPrefill
        || selected.key.kind == PhaseGlobalActionKind::kEncoderDecode)
    {
        PhaseGlobalDispatchPlan const executionPlan = beginGlobalExecutionLease(selected);
        mGlobalEncoderBatchIndices = encoderBatchIndices;
        mInFlightGlobalEncoderKey = encoderKey;
        mInFlightGlobalEncoderReferenceMs = encoderReferenceUs / 1000.0;
        mPendingGlobalOverlapObservation = PendingGlobalOverlapObservation{
            selected.key, static_cast<float>(selected.referenceWorkUs / 1000.0), 0.0F, 0.0F};
        bool const encoderStarted = startNextEncoder();
        bool const phaseStarted = encoderStarted
            && mServer.dispatchGlobalAction(std::move(*pd), executionPlan.planId, executionPlan.snapshotEpoch);
        if (!encoderStarted || !phaseStarted)
        {
            mPendingGlobalOverlapObservation.reset();
            if (!encoderStarted)
            {
                abandonGlobalExecutionLease();
                return false;
            }
            ++mGlobalActionFidelityViolations;
            ELLM_CHECK(false, "Global overlap launched only a subset of the selected phases");
        }
        validateGlobalExecutionLaunch();
        if (selected.key.kind == PhaseGlobalActionKind::kEncoderPrefill)
        {
            ++mGlobalEncoderPrefillSelections;
        }
        else
        {
            ++mGlobalEncoderDecodeSelections;
        }
        return true;
    }
    ++mGlobalPdSelections;
    PhaseGlobalDispatchPlan const executionPlan = beginGlobalExecutionLease(selected);
    bool const started
        = mServer.dispatchGlobalAction(std::move(selected), executionPlan.planId, executionPlan.snapshotEpoch);
    if (!started)
    {
        abandonGlobalExecutionLease();
        return false;
    }
    validateGlobalExecutionLaunch();
    return true;
}

void PhaseThreeCoordinator::completeGlobalOverlapObservation()
{
    if (!mPendingGlobalOverlapObservation.has_value())
    {
        return;
    }
    std::optional<PhaseDispatchMetrics> const& dispatch = mServer.schedulerTelemetry().lastDispatch;
    PhaseGlobalActionKey const& key = mPendingGlobalOverlapObservation->key;
    if (mPendingGlobalOverlapObservation->phaseGpuMs <= 0.0F && dispatch.has_value() && dispatch->externalEncoderActive)
    {
        if (key.kind == PhaseGlobalActionKind::kEncoderPrefill && dispatch->prefillGpuMs > 0.0F
            && dispatch->prefillBatchSize == key.secondaryBatchSize && dispatch->prefillPaddedTokens > 0
            && dispatch->prefillPaddedTokens / dispatch->prefillBatchSize == key.chunkLength)
        {
            int32_t const contextBucket = (dispatch->prefillPastKVMax + mConfig.globalDecodeContextBucketTokens - 1)
                / mConfig.globalDecodeContextBucketTokens;
            if (contextBucket == key.secondaryContextBucket)
            {
                mPendingGlobalOverlapObservation->phaseGpuMs = dispatch->prefillGpuMs;
            }
        }
        else if (key.kind == PhaseGlobalActionKind::kEncoderDecode && dispatch->decodeGpuMs > 0.0F
            && dispatch->decodeBatchSize == key.secondaryBatchSize)
        {
            int32_t const contextBucket
                = (dispatch->plannedDecodeMaxContextLength + mConfig.globalDecodeContextBucketTokens - 1)
                / mConfig.globalDecodeContextBucketTokens;
            if (contextBucket == key.secondaryContextBucket)
            {
                mPendingGlobalOverlapObservation->phaseGpuMs = dispatch->decodeGpuMs;
            }
        }
    }
    if (mPendingGlobalOverlapObservation->encoderGpuMs <= 0.0F || mPendingGlobalOverlapObservation->phaseGpuMs <= 0.0F)
    {
        return;
    }
    float const makespanMs
        = std::max(mPendingGlobalOverlapObservation->encoderGpuMs, mPendingGlobalOverlapObservation->phaseGpuMs);
    mGlobalCostModel.observe(
        mPendingGlobalOverlapObservation->key, {mPendingGlobalOverlapObservation->referenceWorkMs, makespanMs});
    mPendingGlobalOverlapObservation.reset();
}

bool PhaseThreeCoordinator::startNextEncoder()
{
    if (!mEncoding.empty() || mPending.empty() || mVision.busy())
    {
        return false;
    }
    bool const globalAuthorized = mGlobalEncoderBatchIndices.has_value();
    PhaseVisionEncoderDispatchDecision const dispatchDecision = globalAuthorized
        ? PhaseVisionEncoderDispatchDecision{true, PhaseVisionEncoderDispatchReason::kAllowed}
        : mEncoderSerializationGate
        ? PhaseVisionEncoderDispatchDecision{true, PhaseVisionEncoderDispatchReason::kAllowed}
        : nextEncoderDispatchDecision();
    if (mEncoderSerializationGate && mServer.arbitrationSnapshot().busy)
    {
        ++mEncoderSerializationBoundaryWaits;
        return false;
    }
    if (!dispatchDecision.allowed)
    {
        ++mEncoderDispatchDeferrals;
        if (dispatchDecision.reason == PhaseVisionEncoderDispatchReason::kTextGuard)
        {
            ++mEncoderTextGuardDeferrals;
        }
        else if (dispatchDecision.reason == PhaseVisionEncoderDispatchReason::kPrefillGuard)
        {
            ++mEncoderPrefillGuardDeferrals;
        }
        else if (dispatchDecision.reason == PhaseVisionEncoderDispatchReason::kDecodeGuard)
        {
            ++mEncoderDecodeGuardDeferrals;
        }
        return false;
    }
    std::vector<size_t> const batchIndices
        = globalAuthorized ? std::move(*mGlobalEncoderBatchIndices) : nextEncoderBatchIndices();
    mGlobalEncoderBatchIndices.reset();
    size_t const batchSize = batchIndices.size();
    if (batchSize == 0)
    {
        return false;
    }
    size_t candidateInputTokens{};
    for (size_t const pendingIndex : batchIndices)
    {
        size_t const requestInputTokens = mPending[pendingIndex].inputTokens;
        candidateInputTokens = requestInputTokens > std::numeric_limits<size_t>::max() - candidateInputTokens
            ? std::numeric_limits<size_t>::max()
            : candidateInputTokens + requestInputTokens;
    }
    bool const exclusiveEncoder = mConfig.exclusiveEncoderInputTokenThreshold > 0
        && candidateInputTokens > mConfig.exclusiveEncoderInputTokenThreshold;
    if (exclusiveEncoder)
    {
        IndependentPhaseServerArbitrationSnapshot const snapshot = mServer.arbitrationSnapshot();
        bool const prefillInFlight = snapshot.busy
            && (snapshot.inFlightKind == PhaseDispatchKind::kPrefill
                || snapshot.inFlightKind == PhaseDispatchKind::kOverlap);
        if (prefillInFlight)
        {
            ++mExclusiveEncoderPrefillDeferrals;
            return false;
        }
        mServer.setPrefillDispatchBlocked(true);
        mExclusiveEncoderInFlight = true;
    }

    std::vector<PhaseVisionSubmission> submissions;
    submissions.reserve(batchSize);
    size_t encoderInputBytes{};
    size_t encoderInputTokens{};
    auto const now = std::chrono::steady_clock::now();
    if (dispatchDecision.reason == PhaseVisionEncoderDispatchReason::kAgeForced)
    {
        mLastForcedEncoderStart = now;
        ++mEncoderAgeForcedStarts;
    }
    if (mEncoderSerializationGate)
    {
        if (mEncoderSerializationBurstSize == 0)
        {
            ++mEncoderSerializationBursts;
        }
        ++mEncoderSerializationBurstSize;
        ++mEncoderSerializedStarts;
        mSerializedEncoderInFlight = true;
    }
    for (size_t selectedIndex = 0; selectedIndex < batchSize; ++selectedIndex)
    {
        size_t const pendingIndex = batchIndices[selectedIndex] - selectedIndex;
        PendingVisionRequest pending = std::move(mPending[pendingIndex]);
        mPending.erase(mPending.begin() + static_cast<std::ptrdiff_t>(pendingIndex));
        size_t const requestInputBytes = mediaInputBytes(pending);
        encoderInputBytes = requestInputBytes > std::numeric_limits<size_t>::max() - encoderInputBytes
            ? std::numeric_limits<size_t>::max()
            : encoderInputBytes + requestInputBytes;
        size_t const requestInputTokens = pending.inputTokens;
        encoderInputTokens = requestInputTokens > std::numeric_limits<size_t>::max() - encoderInputTokens
            ? std::numeric_limits<size_t>::max()
            : encoderInputTokens + requestInputTokens;
        mEncoding.push_back(std::move(pending));
        PendingVisionRequest& encoding = mEncoding.back();
        double const queueWaitUs
            = std::chrono::duration<double, std::micro>(now - encoding.scheduling.submittedAt).count();
        mLastEncoderQueueWaitUs = queueWaitUs;
        mMaxEncoderQueueWaitUs = std::max(mMaxEncoderQueueWaitUs, queueWaitUs);
        submissions.push_back({encoding.requestId, std::move(encoding.request)});
    }
    uint64_t const timelineTimestampNs = phaseTimelineNowNs();
    for (PhaseVisionSubmission const& submission : submissions)
    {
        recordTimeline(submission.requestId, PhaseTimelineStage::kEncoderStart, batchSize, -1, timelineTimestampNs);
    }
    try
    {
        if (mConfig.enableAsyncEncoderPreparation)
        {
            ELLM_CHECK(!mEncoderPreparation.valid(), "An async encoder preparation is already active");
            CUcontext const cudaContext = mVision.cudaContext();
            mEncoderPreparationStartedAt = std::chrono::steady_clock::now();
            mEncoderPreparation
                = std::async(std::launch::async, [this, cudaContext, submissions = std::move(submissions)]() mutable {
                      ScopedCudaContext context(cudaContext);
                      return mVision.prepare(std::move(submissions));
                  });
            ++mEncoderPreparationStarts;
        }
        else
        {
            ELLM_CHECK(mVision.submit(std::move(submissions)), "Failed to start queued encoder batch");
            mServer.setExternalEncoderActive(true);
        }
    }
    catch (...)
    {
        mServer.setExternalEncoderActive(false);
        if (mExclusiveEncoderInFlight)
        {
            mServer.setPrefillDispatchBlocked(false);
            mExclusiveEncoderInFlight = false;
        }
        throw;
    }
    mEncoderStarts += batchSize;
    ++mEncoderBatches;
    mLastEncoderBatchSize = batchSize;
    mMaxEncoderBatchSize = std::max(mMaxEncoderBatchSize, batchSize);
    mLastEncoderInputBytes = encoderInputBytes;
    mMaxEncoderInputBytes = std::max(mMaxEncoderInputBytes, encoderInputBytes);
    mLastEncoderInputTokens = encoderInputTokens;
    mMaxEncoderInputTokens = std::max(mMaxEncoderInputTokens, encoderInputTokens);
    if (exclusiveEncoder)
    {
        ++mExclusiveEncoderBatches;
    }
    return true;
}

bool PhaseThreeCoordinator::completeEncoderPreparation()
{
    if (!mEncoderPreparation.valid())
    {
        return true;
    }
    if (mEncoderPreparation.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        return false;
    }
    try
    {
        std::shared_ptr<PhaseVisionPreparedBatch> prepared = mEncoderPreparation.get();
        mLastEncoderPreparationUs
            = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - mEncoderPreparationStartedAt)
                  .count();
        mMaxEncoderPreparationUs = std::max(mMaxEncoderPreparationUs, mLastEncoderPreparationUs);
        ++mEncoderPreparationCompletions;
        ELLM_CHECK(mVision.submitPrepared(std::move(prepared)), "Failed to submit prepared encoder batch");
        mServer.setExternalEncoderActive(true);
    }
    catch (...)
    {
        mServer.setExternalEncoderActive(false);
        if (mExclusiveEncoderInFlight)
        {
            mServer.setPrefillDispatchBlocked(false);
            mExclusiveEncoderInFlight = false;
        }
        throw;
    }
    return true;
}

bool PhaseThreeCoordinator::completeEncoder()
{
    if (mEncoding.empty())
    {
        return false;
    }
    if (!completeEncoderPreparation())
    {
        return false;
    }
    bool const batchReady = std::all_of(mEncoding.begin(), mEncoding.end(),
        [&](PendingVisionRequest const& encoding) { return mVision.ready(encoding.requestId); });
    if (!batchReady)
    {
        return false;
    }
    mServer.setExternalEncoderActive(false);
    if (mExclusiveEncoderInFlight)
    {
        // ready() is driven by the encoder CUDA completion event, so the
        // overlapping arena is safe for the next prefill enqueue now.
        mServer.setPrefillDispatchBlocked(false);
        mExclusiveEncoderInFlight = false;
    }
    size_t const batchSize = mEncoding.size();
    for (PendingVisionRequest& encoding : mEncoding)
    {
        uint64_t const requestId = encoding.requestId;
        std::unique_ptr<PhaseVisionPayload> encoded = mVision.take(requestId);
        recordTimeline(requestId, PhaseTimelineStage::kEncoderDone, mEncoding.size());
        ++mEncoderCompletions;
        mLastEncoderGpuMs = encoded->encoderGpuMs;
        mMaxEncoderGpuMs = std::max(mMaxEncoderGpuMs, mLastEncoderGpuMs);
        if (mCancelRequested.erase(requestId) > 0)
        {
            if (encoding.prefixSubmitted)
            {
                ELLM_CHECK(mServer.cancel(requestId), "Cancelled encoder prefix could not be released");
            }
            mRequestIds.erase(requestId);
            eraseTpotTarget(requestId);
            continue;
        }
        ELLM_CHECK(encoded->tokenIds.size() == 1U && !encoded->tokenIds.front().empty(),
            "Phase encoder must produce one non-empty token row per logical request");
        auto sharedPayload = std::shared_ptr<PhaseVisionPayload>(std::move(encoded));
        size_t const encodedBytes = sharedPayload->byteSize();
        std::vector<int32_t> promptTokens = sharedPayload->tokenIds.front();
        mEstimatedPromptTokens = std::max(mEstimatedPromptTokens, promptTokens.size());
        ELLM_CHECK(mDownstreamRequestBytes.emplace(requestId, encodedBytes).second,
            "Encoded phase request is already downstream");
        mReadyPrefill.push_back({requestId, std::move(promptTokens), std::move(sharedPayload), encoding.maxOutputTokens,
            encoding.scheduling, encodedBytes, std::chrono::steady_clock::now(), encoding.prefixSubmitted});
        recordTimeline(requestId, PhaseTimelineStage::kPrefillReady, mEncoding.size());
        mReadyPrefillTokens += mReadyPrefill.back().promptTokens.size();
        mReadyPrefillBytes += encodedBytes;
        mEstimatedEncodedBytes = std::max(mEstimatedEncodedBytes, encodedBytes);
    }
    if (mEncoderBatchMetricCallback)
    {
        mEncoderBatchMetricCallback(
            {mEncoderBatches, batchSize, mLastEncoderInputBytes, mLastEncoderInputTokens, mLastEncoderGpuMs});
    }
    if (mInFlightGlobalEncoderKey.has_value() && mLastEncoderGpuMs > 0.0F && mInFlightGlobalEncoderReferenceMs > 0.0)
    {
        mGlobalCostModel.observe(
            *mInFlightGlobalEncoderKey, {static_cast<float>(mInFlightGlobalEncoderReferenceMs), mLastEncoderGpuMs});
    }
    mInFlightGlobalEncoderKey.reset();
    mInFlightGlobalEncoderReferenceMs = 0.0;
    if (mPendingGlobalOverlapObservation.has_value())
    {
        mPendingGlobalOverlapObservation->encoderGpuMs = mLastEncoderGpuMs;
    }
    mEncoding.clear();
    if (mSerializedEncoderInFlight)
    {
        mSerializedEncoderInFlight = false;
        if (mEncoderSerializationBurstSize >= mConfig.encoderSerializationMaxBurst)
        {
            mEncoderSerializationGate = false;
            mServer.setDispatchBlocked(false);
            mEncoderSerializationBurstSize = 0;
            mEncoderSerializationYieldPending = true;
        }
    }
    return true;
}

bool PhaseThreeCoordinator::dispatchReadyPrefill()
{
    PhaseVisionPrefillAdmissionDecision const decision = nextReadyPrefillDecision();
    size_t const batchSize = decision.batchSize;
    if (batchSize == 0)
    {
        bool const decodeDeferred = decision.reason == PhaseVisionPrefillAdmissionReason::kDecodeDeferral;
        if (decodeDeferred && !mDecodePrefillDeferred)
        {
            ++mDecodeDeferredPrefillPeriods;
        }
        mDecodePrefillDeferred = decodeDeferred;
        return false;
    }
    mDecodePrefillDeferred = false;

    size_t submitted{};
    auto const now = std::chrono::steady_clock::now();
    while (submitted < batchSize && !mReadyPrefill.empty())
    {
        ReadyPrefillRequest& ready = mReadyPrefill.front();
        IndependentPhaseServerSubmission const result = ready.prefixSubmitted
            ? mServer.attachVisionSuffix(ready.requestId, ready.promptTokens, ready.payload)
            : mServer.submitWithVision(
                  ready.requestId, ready.promptTokens, ready.payload, ready.maxOutputTokens, ready.scheduling);
        if (result.status == IndependentPhaseServerStatus::kBackpressure)
        {
            break;
        }
        ELLM_CHECK(result.status == IndependentPhaseServerStatus::kAdmitted
                || result.status == IndependentPhaseServerStatus::kQueued,
            "Ready vision request could not enter the LLM admission queue");
        recordTimeline(ready.requestId, PhaseTimelineStage::kPrefillRelease, batchSize, result.kvSlotId);
        double const queueWaitUs = std::chrono::duration<double, std::micro>(now - ready.encodedAt).count();
        mLastPrefillReadyQueueWaitUs = queueWaitUs;
        mMaxPrefillReadyQueueWaitUs = std::max(mMaxPrefillReadyQueueWaitUs, queueWaitUs);
        ELLM_CHECK(ready.payloadBytes <= mReadyPrefillBytes, "Ready prefill byte accounting underflow");
        ELLM_CHECK(ready.promptTokens.size() <= mReadyPrefillTokens, "Ready prefill token accounting underflow");
        mReadyPrefillTokens -= ready.promptTokens.size();
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
            case PhaseVisionPrefillAdmissionReason::kDecodeDeferral: break;
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

void PhaseThreeCoordinator::recordTimeline(
    uint64_t requestId, PhaseTimelineStage stage, size_t batchSize, int32_t kvSlotId, uint64_t timestampNs) const
{
    if (mTimelineCallback)
    {
        timestampNs = timestampNs > 0U ? timestampNs : phaseTimelineNowNs();
        mTimelineCallback({requestId, stage, timestampNs, 0U, static_cast<int32_t>(batchSize), kvSlotId});
    }
}

std::vector<size_t> PhaseThreeCoordinator::nextEncoderBatchIndices()
{
    bool const globalActive = mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive;
    std::vector<PhaseVisionEncoderInput> inputs;
    inputs.reserve(mPending.size());
    for (PendingVisionRequest const& pending : mPending)
    {
        inputs.push_back(
            {mediaItemCount(pending), mediaInputBytes(pending), pending.inputTokens, pending.mediaGeometry});
    }
    double const oldestWaitUs = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - mPending.front().scheduling.submittedAt)
                                    .count();
    size_t potentialBatchSize{};
    if (!globalActive && mConfig.encoderCreditWaitUs > 0.0)
    {
        potentialBatchSize = phaseVisionEncoderBatchSize(inputs, mConfig.maxEncoderBatchSize,
            mConfig.maxEncoderMediaItems, mConfig.maxEncoderInputBytes, mConfig.maxEncoderInputTokens,
            mConfig.enableHomogeneousEncoderBatching, mConfig.enableEncoderFitLookahead, mConfig.maxEncoderLookahead);
    }
    size_t capacityLimit{};
    while (capacityLimit < mConfig.maxEncoderBatchSize && encoderCapacityAvailable(capacityLimit + 1U))
    {
        ++capacityLimit;
    }
    std::vector<size_t> batchIndices = phaseVisionEncoderBatchIndices(inputs, capacityLimit,
        mConfig.maxEncoderMediaItems, mConfig.maxEncoderInputBytes, mConfig.maxEncoderInputTokens,
        mConfig.enableHomogeneousEncoderBatching, mConfig.enableEncoderFitLookahead, mConfig.maxEncoderLookahead);
    size_t batchSize = batchIndices.size();
    if (batchSize == 0)
    {
        bool const waitForCredits = !globalActive
            && phaseVisionShouldAccumulateEncoderCredits(0U, potentialBatchSize, mConfig.encoderCreditTargetBatchSize,
                oldestWaitUs, mConfig.encoderCreditWaitUs);
        if (waitForCredits && !mEncoderCreditDeferred)
        {
            ++mEncoderCreditWaitPeriods;
        }
        mEncoderCreditDeferred = waitForCredits;
        return {};
    }

    std::vector<size_t> candidatePayloadBytes;
    candidatePayloadBytes.reserve(batchSize);
    for (size_t const index : batchIndices)
    {
        size_t const estimate = mPending[index].estimatedPayloadBytes;
        candidatePayloadBytes.push_back(estimate > 0U ? estimate : mEstimatedEncodedBytes);
    }
    PhaseVisionMemoryStats const& visionMemory = mVision.memoryStats();
    PhaseMemoryBrokerDecision const memoryDecision = mMemoryBroker.planEncoder(
        {mServer.availableKVPages(), mReadyPrefillBytes + mServer.visionPayloadBytes(), visionMemory.idleStorageBytes},
        candidatePayloadBytes);
    PhaseDrainPreference drainPreference = PhaseDrainPreference::kNone;
    if (memoryDecision.preferPrefill)
    {
        drainPreference = PhaseDrainPreference::kPrefill;
    }
    if (memoryDecision.preferDecode)
    {
        drainPreference = PhaseDrainPreference::kDecode;
    }
    mServer.setExternalDrainPreference(globalActive ? PhaseDrainPreference::kNone : drainPreference);
    if (mConfig.memoryBroker.enabled)
    {
        ++mMemoryBrokerDecisions;
        mMemoryBrokerLastReason = memoryDecision.reason;
        mMemoryBrokerLastPredictedBytes = memoryDecision.predictedManagedBytes;
        if (memoryDecision.encoderBatchSize < batchSize)
        {
            ++mMemoryBrokerEncoderReductions;
        }
        if (memoryDecision.encoderBatchSize == 0U)
        {
            ++mMemoryBrokerBackpressure;
        }
        if (memoryDecision.preferPrefill)
        {
            ++mMemoryBrokerPrefillPreferences;
        }
        if (memoryDecision.preferDecode)
        {
            ++mMemoryBrokerDecodePreferences;
        }
        if (memoryDecision.reclaimIdleVision)
        {
            mVision.reclaimIdleStorage(true);
            ++mMemoryBrokerIdleReclaims;
        }
    }
    batchIndices.resize(memoryDecision.encoderBatchSize);
    batchSize = batchIndices.size();
    if (batchSize == 0U)
    {
        mEncoderCreditDeferred = false;
        return {};
    }

    bool const memoryAllowsCreditWait = memoryDecision.reason == PhaseMemoryBrokerReason::kDisabled
        || memoryDecision.reason == PhaseMemoryBrokerReason::kAllowed;
    size_t const creditTargetBatchSize
        = mConfig.encoderCreditTargetBatchSize == 0U ? potentialBatchSize : mConfig.encoderCreditTargetBatchSize;
    bool const waitForCredits = !globalActive && memoryAllowsCreditWait
        && phaseVisionShouldAccumulateEncoderCredits(
            batchSize, potentialBatchSize, creditTargetBatchSize, oldestWaitUs, mConfig.encoderCreditWaitUs);
    if (waitForCredits)
    {
        if (!mEncoderCreditDeferred)
        {
            ++mEncoderCreditWaitPeriods;
        }
        mEncoderCreditDeferred = true;
        return {};
    }
    if (mEncoderCreditDeferred && creditTargetBatchSize > batchSize && oldestWaitUs >= mConfig.encoderCreditWaitUs)
    {
        ++mEncoderCreditAgeReleases;
    }
    mEncoderCreditDeferred = false;

    if (mConfig.enableCostAwareEncoderBatching)
    {
        std::vector<size_t> candidateInputTokens;
        candidateInputTokens.reserve(batchIndices.size());
        for (size_t const index : batchIndices)
        {
            candidateInputTokens.push_back(inputs[index].inputTokens);
        }
        PhaseVisionEncoderBatchChoice const encoderChoice
            = phaseVisionSelectEncoderBatch(candidateInputTokens, mConfig.encoderBatchCosts);
        if (encoderChoice.coverageMiss)
        {
            ++mEncoderCostCoverageMisses;
        }
        else
        {
            ELLM_CHECK(encoderChoice.batchSize > 0U && encoderChoice.batchSize <= batchIndices.size(),
                "Cost-aware encoder batch selection is outside the candidate range");
            batchIndices.resize(encoderChoice.batchSize);
            batchSize = batchIndices.size();
            ++mEncoderCostAwareSelections;
            mLastPredictedEncoderDrainGpuMs = encoderChoice.predictedDrainGpuMs;
            mLastPredictedEncoderDrainTurns = encoderChoice.predictedDrainTurns;
        }
    }

    size_t mediaItems{};
    size_t inputBytes{};
    size_t inputTokens{};
    for (size_t index = 0; index < batchSize; ++index)
    {
        PhaseVisionEncoderInput const& input = inputs[batchIndices[index]];
        mediaItems += input.mediaItems;
        inputBytes += input.inputBytes;
        inputTokens += input.inputTokens;
    }
    bool const batchFull = batchSize == mConfig.maxEncoderBatchSize;
    bool const mediaFull = mConfig.maxEncoderMediaItems > 0 && mediaItems >= mConfig.maxEncoderMediaItems;
    bool const inputFull = mConfig.maxEncoderInputBytes > 0 && inputBytes >= mConfig.maxEncoderInputBytes;
    bool const tokenFull = mConfig.maxEncoderInputTokens > 0 && inputTokens >= mConfig.maxEncoderInputTokens;
    bool const resourceLimited = batchSize < inputs.size();
    bool const capacityFull = !encoderCapacityAvailable(batchSize + 1U);
    if (!globalActive && !batchFull && !mediaFull && !inputFull && !tokenFull && !resourceLimited && !capacityFull
        && oldestWaitUs < mConfig.encoderBatchWaitUs)
    {
        return {};
    }
    if (globalActive && !batchFull && !mediaFull && !inputFull && !tokenFull && !resourceLimited && !capacityFull
        && batchSize == inputs.size() && mVisionInterarrivalSamples > 0U && mVisionInterarrivalEwmaUs > 0.0
        && mLastVisionArrival != std::chrono::steady_clock::time_point{})
    {
        double const sinceArrivalUs = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - mLastVisionArrival)
                                          .count();
        double predictedWaitUs = mVisionInterarrivalEwmaUs - sinceArrivalUs;
        if (mConfig.encoderBatchWaitUs > 0.0)
        {
            predictedWaitUs = std::min(predictedWaitUs, mConfig.encoderBatchWaitUs - oldestWaitUs);
        }
        auto estimateEncoder = [&](size_t rows, size_t totalInputTokens) -> std::optional<double> {
            constexpr size_t kEncoderTokenBucket = 1024U;
            int32_t const contextBucket
                = static_cast<int32_t>((totalInputTokens + kEncoderTokenBucket - 1U) / kEncoderTokenBucket);
            PhaseGlobalActionKey const key{
                PhaseGlobalActionKind::kEncoder, static_cast<int32_t>(rows), 0, 0, contextBucket, 0};
            if (std::optional<PhaseGlobalCostEstimate> const online = mGlobalCostModel.estimate(key))
            {
                return static_cast<double>(online->makespanMedianMs + online->uncertaintyMs) * 1000.0;
            }
            PhaseVisionEncoderBatchCost const* selected{};
            for (PhaseVisionEncoderBatchCost const& cost : mConfig.encoderBatchCosts)
            {
                if (cost.batchSize >= rows && cost.maxInputTokens >= totalInputTokens
                    && (selected == nullptr || cost.batchSize < selected->batchSize
                        || (cost.batchSize == selected->batchSize && cost.p95GpuMs < selected->p95GpuMs)))
                {
                    selected = &cost;
                }
            }
            return selected != nullptr ? std::optional<double>(static_cast<double>(selected->p95GpuMs) * 1000.0)
                                       : std::nullopt;
        };
        size_t const averageInputTokens = (inputTokens + batchSize - 1U) / batchSize;
        std::optional<double> const currentEncoder = estimateEncoder(batchSize, inputTokens);
        std::optional<double> const singletonEncoder = estimateEncoder(1U, averageInputTokens);
        std::optional<double> const futureEncoder
            = estimateEncoder(batchSize + 1U, inputTokens + averageInputTokens);
        if (predictedWaitUs > 0.0 && currentEncoder.has_value() && singletonEncoder.has_value()
            && futureEncoder.has_value())
        {
            size_t const estimatedPromptTokens = std::max<size_t>(1U,
                mEstimatedPromptTokens > 0U ? mEstimatedPromptTokens : averageInputTokens);
            int32_t const promptTokens = static_cast<int32_t>(std::min<size_t>(
                estimatedPromptTokens, static_cast<size_t>(std::numeric_limits<int32_t>::max())));
            PhaseGlobalCostEstimate const currentPrefill = mServer.estimateGlobalPrefillDrainCost(
                static_cast<int32_t>(batchSize), promptTokens, PhasePrefillClass::kExternal);
            PhaseGlobalCostEstimate const singletonPrefill
                = mServer.estimateGlobalPrefillDrainCost(1, promptTokens, PhasePrefillClass::kExternal);
            PhaseGlobalCostEstimate const futurePrefill = mServer.estimateGlobalPrefillDrainCost(
                static_cast<int32_t>(batchSize + 1U), promptTokens, PhasePrefillClass::kExternal);
            auto robustPrefillUs = [](PhaseGlobalCostEstimate const& estimate) {
                return static_cast<double>(estimate.makespanMedianMs + estimate.uncertaintyMs) * 1000.0;
            };
            double const currentCriticalUs = *currentEncoder + robustPrefillUs(currentPrefill);
            double const singletonCriticalUs = *singletonEncoder + robustPrefillUs(singletonPrefill);
            double const futureCriticalUs = *futureEncoder + robustPrefillUs(futurePrefill);
            PendingVisionRequest const& oldest = mPending.front();
            double const targetUs
                = oldest.scheduling.ttftTargetUs > 0.0 ? oldest.scheduling.ttftTargetUs : mConfig.visionTtftTargetUs;
            double const oldestSlackUs
                = targetUs > 0.0 ? targetUs - oldestWaitUs : std::numeric_limits<double>::infinity();
            double const dispatchNowHorizonUs = currentCriticalUs + singletonCriticalUs;
            double const waitHorizonUs = predictedWaitUs + futureCriticalUs;
            if (phaseVisionShouldWaitForGlobalEncoderArrival(
                    predictedWaitUs, oldestSlackUs, futureCriticalUs, dispatchNowHorizonUs, waitHorizonUs))
            {
                if (!mGlobalEncoderArrivalWaitDeferred)
                {
                    ++mGlobalEncoderArrivalWaitPeriods;
                }
                mGlobalEncoderArrivalWaitDeferred = true;
                mLastGlobalEncoderArrivalWaitUs = predictedWaitUs;
                mLastGlobalAction = PhaseGlobalActionKind::kWait;
                return {};
            }
        }
        if (mGlobalEncoderArrivalWaitDeferred && sinceArrivalUs >= mVisionInterarrivalEwmaUs)
        {
            ++mGlobalEncoderArrivalWaitExpirations;
        }
    }
    mGlobalEncoderArrivalWaitDeferred = false;
    return batchIndices;
}

PhaseVisionPrefillAdmissionDecision PhaseThreeCoordinator::nextReadyPrefillDecision() const noexcept
{
    if (mReadyPrefill.empty())
    {
        return {};
    }
    std::vector<int32_t> promptTokenCounts;
    std::vector<IndependentPhaseAdmissionRequest> admissionRequests;
    promptTokenCounts.reserve(mReadyPrefill.size());
    admissionRequests.reserve(mReadyPrefill.size());
    size_t attachedPrefix{};
    bool encounteredUnattached{};
    for (ReadyPrefillRequest const& ready : mReadyPrefill)
    {
        promptTokenCounts.push_back(static_cast<int32_t>(ready.promptTokens.size()));
        if (ready.prefixSubmitted)
        {
            if (!encounteredUnattached)
            {
                ++attachedPrefix;
            }
        }
        else
        {
            encounteredUnattached = true;
            admissionRequests.push_back(
                {ready.requestId, static_cast<int32_t>(ready.promptTokens.size()), ready.maxOutputTokens});
        }
    }
    double const oldestWaitUs
        = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - mReadyPrefill.front().encodedAt)
              .count();
    bool const globalActive = mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive;
    return phaseVisionAdaptiveReadyPrefillDecision(promptTokenCounts, mConfig.maxPrefillBatchSize,
        mConfig.maxPrefillBatchTokens, oldestWaitUs, globalActive ? 0.0 : mConfig.prefillBatchWaitUs,
        globalActive ? false : mConfig.enableAdaptivePrefillAdmission, mConfig.adaptivePrefillMinBatchSize,
        mPending.size() + mEncoding.size(), attachedPrefix + mServer.admissibleRequestPrefix(admissionRequests),
        attachedPrefix > 0U ? std::max(mServer.availableKVPages(), 1) : mServer.availableKVPages(),
        decodeTpotPressure(), mConfig.prefillDecodeTpotPressureLimit, mReadyPrefillBytes, mConfig.maxEncodedBytes,
        mConfig.prefillReadyBytePressureRatio,
        !globalActive && mConfig.enableDecodeProtectedPrefillDeferral
            && mServer.adaptiveAdmissionExternalProfileActive() && !mServer.adaptiveAdmissionTpotBudgetSatisfiable(),
        mConfig.maxDecodeProtectedPrefillWaitUs);
}

bool PhaseThreeCoordinator::encoderCapacityAvailable(size_t additionalRequests) const noexcept
{
    return phaseVisionEncoderCapacityAvailable(mDownstreamRequestBytes.size(), effectiveEncodedCapacity(),
        mReadyPrefillBytes + mServer.visionPayloadBytes(), mConfig.maxEncodedBytes, mEstimatedEncodedBytes,
        additionalRequests);
}

PhaseVisionEncoderDispatchDecision PhaseThreeCoordinator::nextEncoderDispatchDecision() const noexcept
{
    if (mPending.empty())
    {
        return {};
    }
    auto const now = std::chrono::steady_clock::now();
    double const oldestVisionAgeUs
        = std::chrono::duration<double, std::micro>(now - mPending.front().scheduling.submittedAt).count();
    double const sinceLastForcedStartUs = mLastForcedEncoderStart == std::chrono::steady_clock::time_point{}
        ? std::numeric_limits<double>::infinity()
        : std::chrono::duration<double, std::micro>(now - mLastForcedEncoderStart).count();
    IndependentPhaseServerArbitrationSnapshot const snapshot = mServer.arbitrationSnapshot();
    double const predictedEncoderCostUs
        = std::max(mConfig.encoderDispatchInitialCostUs, static_cast<double>(mLastEncoderGpuMs) * 1000.0)
        + mConfig.encoderDispatchCostSafetyMarginUs;
    bool const textPrefillInFlight = snapshot.busy
        && (snapshot.inFlightKind == PhaseDispatchKind::kPrefill
            || snapshot.inFlightKind == PhaseDispatchKind::kOverlap)
        && snapshot.inFlightPrefillClass == PhasePrefillClass::kText;
    bool const prefillInFlight = snapshot.busy
        && (snapshot.inFlightKind == PhaseDispatchKind::kPrefill
            || snapshot.inFlightKind == PhaseDispatchKind::kOverlap);
    bool const decodeInFlight = snapshot.busy
        && (snapshot.inFlightKind == PhaseDispatchKind::kDecode
            || snapshot.inFlightKind == PhaseDispatchKind::kOverlap);
    return phaseVisionEncoderDispatchDecision(mConfig.enableEncoderDispatchArbitration, oldestVisionAgeUs,
        sinceLastForcedStartUs, mConfig.encoderDispatchMaxDeferUs, mConfig.encoderDispatchForcedIntervalUs,
        snapshot.oldestTextWithoutTokenAgeUs, predictedEncoderCostUs, mConfig.encoderDispatchTextGuardAgeUs,
        decodeTpotPressure(), mConfig.encoderDispatchDecodeTpotPressureLimit, textPrefillInFlight, decodeInFlight,
        snapshot.prefillMinTtftSlackUs, prefillInFlight);
}

bool PhaseThreeCoordinator::encoderSerializationDue() const noexcept
{
    if (!mConfig.enableDeadlineAwareEncoderSerialization || mPending.empty() || !encoderCapacityAvailable())
    {
        return false;
    }
    PendingVisionRequest const& oldest = mPending.front();
    auto const now = std::chrono::steady_clock::now();
    double const oldestVisionAgeUs
        = std::chrono::duration<double, std::micro>(now - oldest.scheduling.submittedAt).count();
    double const targetUs
        = oldest.scheduling.ttftTargetUs > 0.0 ? oldest.scheduling.ttftTargetUs : mConfig.visionTtftTargetUs;
    double const predictedEncoderCostUs
        = std::max(mConfig.encoderDispatchInitialCostUs, static_cast<double>(mLastEncoderGpuMs) * 1000.0)
        + mConfig.encoderDispatchCostSafetyMarginUs;
    return phaseVisionEncoderSerializationDue(
        oldestVisionAgeUs, targetUs, mConfig.encoderSerializationDeadlineRatio, predictedEncoderCostUs);
}

void PhaseThreeCoordinator::refreshEncoderSerializationGate() noexcept
{
    if (!mConfig.enableDeadlineAwareEncoderSerialization)
    {
        return;
    }
    if (mSerializedEncoderInFlight || mEncoderSerializationYieldPending)
    {
        return;
    }
    bool const shouldBlock = encoderSerializationDue();
    if (shouldBlock == mEncoderSerializationGate)
    {
        return;
    }
    mEncoderSerializationGate = shouldBlock;
    if (!shouldBlock)
    {
        mEncoderSerializationBurstSize = 0;
    }
    mServer.setDispatchBlocked(shouldBlock);
}

size_t PhaseThreeCoordinator::effectiveEncodedCapacity() const noexcept
{
    return mEffectiveEncodedCapacity;
}

float PhaseThreeCoordinator::decodeTpotPressure() const noexcept
{
    double const requestTargetUs = mTpotTargets.empty() ? 0.0 : *mTpotTargets.begin();
    double const targetUs = requestTargetUs > 0.0 ? requestTargetUs : mConfig.encodedCapacityDecodeTpotTargetUs;
    return phaseVisionDecodeTpotPressure(mServer.arbitrationSnapshot().recentDecodeTpotP95Us, targetUs);
}

void PhaseThreeCoordinator::refreshEffectiveEncodedCapacity() noexcept
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
    bool const visionLate = mConfig.lookaheadEscalationRatio > 0.0 && visionTtftTargetUs > 0.0
        && oldestVisionAgeUs >= visionTtftTargetUs * mConfig.lookaheadEscalationRatio;
    size_t const upstreamBacklog = mPending.size() + mEncoding.size() + mReadyPrefill.size();
    bool const backlogDemand = mConfig.encodedCapacityBacklogEnterRequests > 0U
        && upstreamBacklog >= mConfig.encodedCapacityBacklogEnterRequests;
    if (mServer.throughputMode() || visionLate || backlogDemand)
    {
        mEncodedCapacityDemandActive = true;
    }
    bool const visionPipelineDrained
        = mPending.empty() && mEncoding.empty() && mReadyPrefill.empty() && mDownstreamRequestBytes.empty();
    if (visionPipelineDrained)
    {
        mEncodedCapacityDemandActive = false;
    }
    size_t const nextCapacity = phaseVisionNextEncodedCapacity(mEffectiveEncodedCapacity, mConfig.maxEncodedInFlight,
        mConfig.throughputMaxEncodedInFlight, mEncodedCapacityDemandActive, oldestVisionAgeUs, visionTtftTargetUs,
        mConfig.lookaheadEscalationRatio, decodeTpotPressure(), mConfig.lookaheadDecodeTpotPressureLimit,
        mConfig.lookaheadDecodeTpotPressureRecoveryLimit);
    if (nextCapacity == mEffectiveEncodedCapacity)
    {
        return;
    }
    auto const now = std::chrono::steady_clock::now();
    double const dwellUs = mEncodedCapacityLastTransition == std::chrono::steady_clock::time_point{}
        ? std::numeric_limits<double>::infinity()
        : std::chrono::duration<double, std::micro>(now - mEncodedCapacityLastTransition).count();
    if (dwellUs < mConfig.encodedCapacityMinDwellUs)
    {
        if (!mEncodedCapacityDwellDeferred)
        {
            ++mEncodedCapacityDwellBlocks;
            mEncodedCapacityDwellDeferred = true;
        }
        return;
    }
    mEncodedCapacityDwellDeferred = false;
    if (nextCapacity > mEffectiveEncodedCapacity)
    {
        ++mLookaheadEscalations;
    }
    else
    {
        ++mEncodedCapacityContractions;
    }
    mEffectiveEncodedCapacity = nextCapacity;
    mMaxEffectiveEncodedCapacity = std::max(mMaxEffectiveEncodedCapacity, nextCapacity);
    mEncodedCapacityLastTransition = now;
}

void PhaseThreeCoordinator::eraseTpotTarget(uint64_t requestId)
{
    auto const requestTarget = mRequestTpotTargets.find(requestId);
    if (requestTarget == mRequestTpotTargets.end())
    {
        return;
    }
    auto const target = mTpotTargets.find(requestTarget->second);
    ELLM_CHECK(target != mTpotTargets.end(), "Three-phase TPOT target index is inconsistent");
    mTpotTargets.erase(target);
    mRequestTpotTargets.erase(requestTarget);
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

size_t PhaseThreeCoordinator::mediaInputBytes(PendingVisionRequest const& pending) noexcept
{
    size_t result{};
    for (LLMGenerationRequest::Request const& request : pending.request.requests)
    {
        for (imageUtils::ImageData const& image : request.imageBuffers)
        {
            int64_t const bytesPerFrame = image.bytesPerFrame();
            if (bytesPerFrame <= 0 || image.frames <= 0)
            {
                continue;
            }
            size_t const frameBytes = static_cast<size_t>(bytesPerFrame);
            size_t const frames = static_cast<size_t>(image.frames);
            if (frames > std::numeric_limits<size_t>::max() / frameBytes)
            {
                return std::numeric_limits<size_t>::max();
            }
            size_t const bytes = frames * frameBytes;
            if (bytes > std::numeric_limits<size_t>::max() - result)
            {
                return std::numeric_limits<size_t>::max();
            }
            result += bytes;
        }
    }
    return result;
}

std::vector<int64_t> PhaseThreeCoordinator::mediaGeometry(LLMGenerationRequest const& request)
{
    std::vector<int64_t> result;
    result.push_back(static_cast<int64_t>(request.requests.size()));
    for (LLMGenerationRequest::Request const& logicalRequest : request.requests)
    {
        result.push_back(static_cast<int64_t>(logicalRequest.imageBuffers.size()));
        for (imageUtils::ImageData const& image : logicalRequest.imageBuffers)
        {
            result.insert(result.end(),
                {image.frames, image.height, image.width, image.channels, image.doResize ? 1 : 0,
                    image.isVideo ? 1 : 0});
        }
    }
    return result;
}

} // namespace trt_edgellm::rt
