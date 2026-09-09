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
#include "runtime/phase/policy/phaseFormationPlanner.h"
#include "runtime/scheduling/phaseActivityTimeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>

namespace trt_edgellm::rt
{
namespace
{

//! Keep coordinator-owned E/P/D leases disjoint from P/D-only plans created
//! during engine warmup or standalone server operation.
constexpr uint64_t kTHREE_PHASE_PLAN_NAMESPACE = uint64_t{1U} << 63U;
constexpr uint64_t kTHREE_PHASE_EXECUTION_NAMESPACE = uint64_t{1U} << 62U;

PhaseUnifiedActionDirection phaseInitialDirection(PhaseGlobalActionKind action) noexcept
{
    PhaseUnifiedActionDirection canonical{PhaseUnifiedActionDirection::kIdleLaunch};
    switch (action)
    {
    case PhaseGlobalActionKind::kEncoderPrefill: canonical = PhaseUnifiedActionDirection::kEncoderToPrefill; break;
    case PhaseGlobalActionKind::kEncoderDecode: canonical = PhaseUnifiedActionDirection::kEncoderToDecode; break;
    case PhaseGlobalActionKind::kPrefillDecode: canonical = PhaseUnifiedActionDirection::kPrefillToDecode; break;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kEncoder:
    case PhaseGlobalActionKind::kPrefill:
    case PhaseGlobalActionKind::kDecode:
    case PhaseGlobalActionKind::kWait: break;
    }
    return canonical;
}

PhaseStartSkewBucket phaseRequestedStartSkew(double elapsedUs = 0.0, double referenceUs = 0.0) noexcept
{
    if (referenceUs > 0.0)
    {
        return phaseStartSkewBucketFromFraction(elapsedUs / referenceUs, true);
    }
    return PhaseStartSkewBucket::kImmediate;
}

std::optional<PhaseActivityKind> unifiedPhaseActivityKind(PhaseUnifiedPhase phase) noexcept
{
    switch (phase)
    {
    case PhaseUnifiedPhase::kEncoder: return PhaseActivityKind::kEncoder;
    case PhaseUnifiedPhase::kPrefill: return PhaseActivityKind::kPrefill;
    case PhaseUnifiedPhase::kDecode: return PhaseActivityKind::kDecode;
    case PhaseUnifiedPhase::kCopy: return PhaseActivityKind::kCopy;
    case PhaseUnifiedPhase::kNone: return std::nullopt;
    }
    return std::nullopt;
}

bool unifiedPhaseActivityMatches(PhaseUnifiedPhase phase, PhaseActivityInterval const& interval)
{
    std::optional<PhaseActivityKind> const kind = unifiedPhaseActivityKind(phase);
    if (!kind.has_value() || interval.kind != *kind)
    {
        return false;
    }
    switch (phase)
    {
    case PhaseUnifiedPhase::kEncoder: return interval.name == "encoder_engine";
    case PhaseUnifiedPhase::kPrefill:
        return interval.name == "prefill_dispatch" || interval.name == "prefill_residual_dispatch";
    case PhaseUnifiedPhase::kDecode:
        return interval.name == "decode_dispatch" || interval.name == "decode_residual_dispatch";
    case PhaseUnifiedPhase::kCopy: return interval.name == "encoder_output_copy";
    case PhaseUnifiedPhase::kNone: return false;
    }
    return false;
}

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

double steadyTimestampUs() noexcept
{
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

PhaseFormationWork realizedDispatchWork(PhaseGlobalDispatchPlan const& plan) noexcept
{
    PhaseFormationWork work;
    switch (plan.action)
    {
    case PhaseGlobalActionKind::kEncoder: work.encoderRows = plan.primaryRequestIds.size(); break;
    case PhaseGlobalActionKind::kPrefill: work.prefillRows = plan.primaryRequestIds.size(); break;
    case PhaseGlobalActionKind::kDecode: work.decodeRows = plan.primaryRequestIds.size(); break;
    case PhaseGlobalActionKind::kEncoderPrefill:
        work.encoderRows = plan.primaryRequestIds.size();
        work.prefillRows = plan.secondaryRequestIds.size();
        break;
    case PhaseGlobalActionKind::kEncoderDecode:
        work.encoderRows = plan.primaryRequestIds.size();
        work.decodeRows = plan.secondaryRequestIds.size();
        break;
    case PhaseGlobalActionKind::kPrefillDecode:
        work.prefillRows = plan.primaryRequestIds.size();
        work.decodeRows = plan.secondaryRequestIds.size();
        break;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kWait: break;
    }
    return work;
}

PhaseUnifiedWork unifiedCandidateWork(PhaseGlobalActionCandidate const& candidate) noexcept
{
    PhaseUnifiedWork work;
    switch (candidate.key.kind)
    {
    case PhaseGlobalActionKind::kEncoder: work.encoderRows = candidate.key.primaryBatchSize; break;
    case PhaseGlobalActionKind::kPrefill:
        work.prefillRows = candidate.key.primaryBatchSize;
        work.prefillTokens = candidate.key.primaryBatchSize * candidate.key.chunkLength;
        break;
    case PhaseGlobalActionKind::kDecode: work.decodeRows = candidate.key.primaryBatchSize; break;
    case PhaseGlobalActionKind::kEncoderPrefill:
        work.encoderRows = candidate.key.primaryBatchSize;
        work.prefillRows = candidate.key.secondaryBatchSize;
        work.prefillTokens = candidate.key.secondaryBatchSize * candidate.key.chunkLength;
        break;
    case PhaseGlobalActionKind::kEncoderDecode:
        work.encoderRows = candidate.key.primaryBatchSize;
        work.decodeRows = candidate.key.secondaryBatchSize;
        break;
    case PhaseGlobalActionKind::kPrefillDecode:
        work.prefillRows = candidate.key.primaryBatchSize;
        work.prefillTokens = candidate.key.primaryBatchSize * candidate.key.chunkLength;
        work.decodeRows = candidate.key.secondaryBatchSize;
        break;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kWait: break;
    }
    return work;
}

PhaseModelFormationSnapshot modelFormationSnapshot(
    PhaseFormationOracleResult const& result, std::vector<PhaseGlobalActionCandidate> const& candidates) noexcept
{
    PhaseModelFormationSnapshot snapshot;
    snapshot.evaluated = true;
    if (!result.selectedAction.has_value() || *result.selectedAction >= result.sequences.size()
        || *result.selectedAction >= candidates.size())
    {
        return snapshot;
    }
    PhaseFormationSequence const& sequence = result.sequences[*result.selectedAction];
    if (!sequence.feasible || !std::isfinite(sequence.makespanUs))
    {
        return snapshot;
    }
    snapshot.valid = true;
    snapshot.selectedActionId = candidates[*result.selectedAction].candidateId;
    snapshot.selectedHorizonUs = sequence.makespanUs;
    snapshot.selectedDecodeViolationUs = sequence.decodeServiceViolationUs;
    snapshot.selectedProtectedViolationUs = sequence.protectedViolationUs;
    return snapshot;
}

} // namespace

std::vector<size_t> phaseEncoderCalibrationBatchSizes(
    size_t maxEncoderBatchSize, std::vector<size_t> requestedBatchSizes)
{
    ELLM_CHECK(maxEncoderBatchSize > 0U, "Phase encoder calibration requires a positive batch limit");
    if (requestedBatchSizes.empty())
    {
        size_t batchSize{1U};
        while (batchSize < maxEncoderBatchSize)
        {
            requestedBatchSizes.push_back(batchSize);
            if (batchSize > maxEncoderBatchSize / 2U)
            {
                break;
            }
            batchSize *= 2U;
        }
        requestedBatchSizes.push_back(maxEncoderBatchSize);
    }
    for (size_t const batchSize : requestedBatchSizes)
    {
        ELLM_CHECK(batchSize > 0U && batchSize <= maxEncoderBatchSize,
            "Phase encoder calibration batch is outside the encoder profile");
    }
    std::sort(requestedBatchSizes.begin(), requestedBatchSizes.end());
    requestedBatchSizes.erase(
        std::unique(requestedBatchSizes.begin(), requestedBatchSizes.end()), requestedBatchSizes.end());
    return requestedBatchSizes;
}

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

std::vector<size_t> phaseVisionNextQueuedEncoderBatchIndices(std::vector<PhaseVisionEncoderInput> const& inputs,
    std::vector<size_t> const& selectedIndices, size_t maxBatchSize, size_t maxMediaItems, size_t maxInputBytes,
    size_t maxInputTokens, bool requireHomogeneousGeometry, bool enableFitLookahead, size_t maxLookahead)
{
    std::vector<bool> selected(inputs.size());
    for (size_t const index : selectedIndices)
    {
        ELLM_CHECK(index < inputs.size(), "Selected encoder queue index is outside the pending input range");
        selected[index] = true;
    }

    std::vector<PhaseVisionEncoderInput> remainingInputs;
    std::vector<size_t> remainingIndices;
    remainingInputs.reserve(inputs.size());
    remainingIndices.reserve(inputs.size());
    for (size_t index{}; index < inputs.size(); ++index)
    {
        if (!selected[index])
        {
            remainingInputs.push_back(inputs[index]);
            remainingIndices.push_back(index);
        }
    }

    std::vector<size_t> const relativeIndices = phaseVisionEncoderBatchIndices(remainingInputs, maxBatchSize,
        maxMediaItems, maxInputBytes, maxInputTokens, requireHomogeneousGeometry, enableFitLookahead, maxLookahead);
    std::vector<size_t> result;
    result.reserve(relativeIndices.size());
    for (size_t const relativeIndex : relativeIndices)
    {
        result.push_back(remainingIndices[relativeIndex]);
    }
    return result;
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
    , mRuntimeCostTracker(mConfig.runtimeCostTracker != nullptr
              ? mConfig.runtimeCostTracker
              : std::make_shared<PhaseRuntimeCostTracker>([&] {
                    PhaseRuntimeCostTrackerConfig trackerConfig;
                    trackerConfig.action = mConfig.globalCostModelConfig;
                    trackerConfig.decodeContextBucketTokens = mConfig.globalDecodeContextBucketTokens;
                    return trackerConfig;
                }()))
    , mMemoryBroker(mConfig.memoryBroker)
    , mGlobalFormationRealizedTracker({mConfig.globalFormationRealizedDispatches, 8U})
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
    ELLM_CHECK(mConfig.globalCalibrationMaxOverlapKeys > 0U, "Global calibration overlap-key limit must be positive");
    ELLM_CHECK(mConfig.globalFormationRealizedDispatches > 0U, "Global formation realized horizon must be positive");
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
    bool const pdOnlyFastPath = mConfig.globalSchedulerMode != PhaseGlobalSchedulerMode::kDisabled
        && mRequestIds.empty() && mPending.empty() && mEncoding.empty() && mReadyPrefill.empty()
        && mDownstreamRequestBytes.empty() && !mVision.busy() && !mEncoderPreparation.valid()
        && mPreparedEncoder == nullptr && !mGlobalExecutionLease.has_value()
        && !mPendingGlobalOverlapObservation.has_value() && !phasePolicyUsesTransition(mConfig.policyMode)
        && !mUnifiedEventCallback && !mFormationEpisodeCallback;
    if (pdOnlyFastPath)
    {
        // Project an E-empty global state directly onto the common P/D actor.
        // This preserves the same queue builders, global selector, WAIT
        // actions, ownership, and CUDA observations while avoiding repeated
        // E-state materialization on every host poll. A newly submitted vision
        // request makes mRequestIds non-empty and exits this path immediately.
        mServer.setExternalDrainPreference(PhaseDrainPreference::kNone);
        mServer.setExternalPendingRequests(0U);
        bool const progressed = mServer.poll();
        // The direct P/D fast path and the three-phase coordinator share the
        // scheduler's monotonic execution-lease namespace. A late vision
        // arrival can switch policy ownership after direct residual P/D
        // augmentation, so synchronize the local counters before issuing the
        // next explicit E/P/D plan.
        uint64_t const planSequence = mServer.globalPlanSequence();
        uint64_t const snapshotEpoch = mServer.globalSnapshotEpoch();
        mGlobalPlanSequence = std::max(mGlobalPlanSequence, planSequence & ~kTHREE_PHASE_PLAN_NAMESPACE);
        mGlobalSnapshotEpoch = std::max(mGlobalSnapshotEpoch, snapshotEpoch & ~kTHREE_PHASE_PLAN_NAMESPACE);
        return progressed;
    }
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
        if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive && mConfig.enableAsyncEncoderPreparation
            && mEncoding.empty() && !mEncoderPreparation.valid() && mPreparedEncoder == nullptr && !mPending.empty()
            && !mVision.busy())
        {
            // Preparation is a mechanism stage, not an E execution action.
            // Materialize the real encoder cohort while P/D continue, then
            // expose the prepared E batch to the global action selector.
            progressed = startNextEncoder() || progressed;
        }
        bool const globalProgress = dispatchGlobalAction();
        progressed = globalProgress || progressed;
    }
    else if (mConfig.enableEncoderDispatchArbitration)
    {
        progressed = startNextEncoder() || progressed;
    }
    if (!mEncoderPreparation.valid() && mPreparedEncoder == nullptr)
    {
        mVision.reclaimIdleStorage();
    }
    completeGlobalOverlapObservation();
    refreshGlobalExecutionLease();
    if (empty() && !mGlobalExecutionLease.has_value())
    {
        mGlobalFormationRealizedTracker.flush(steadyTimestampUs());
        emitCompletedFormationEpisodes();
    }
    observeUnifiedInFlightTransitions();
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
    result.unknownPayloadBootstrapSelections = mUnknownPayloadBootstrapSelections;
    result.exclusiveEncoderPrefillDeferrals = mExclusiveEncoderPrefillDeferrals;
    result.globalDecisions = mGlobalDecisions;
    result.globalEncoderSelections = mGlobalEncoderSelections;
    result.globalEncoderPrefillSelections = mGlobalEncoderPrefillSelections;
    result.globalEncoderDecodeSelections = mGlobalEncoderDecodeSelections;
    result.globalResidualAugmentationOpportunities = mGlobalResidualAugmentationOpportunities;
    result.globalResidualEncoderPrefillSelections = mGlobalResidualEncoderPrefillSelections;
    result.globalResidualEncoderDecodeSelections = mGlobalResidualEncoderDecodeSelections;
    result.globalResidualAugmentationUnknownCostRejects = mGlobalResidualAugmentationUnknownCostRejects;
    result.globalResidualPrefillDecodeOpportunities = mGlobalResidualPrefillDecodeOpportunities;
    result.globalResidualPrefillDecodeSelections = mGlobalResidualPrefillDecodeSelections;
    result.globalResidualPrefillDecodeUnknownCostRejects = mGlobalResidualPrefillDecodeUnknownCostRejects;
    result.globalResidualPrefillAnchorOpportunities = mGlobalResidualPrefillAnchorOpportunities;
    result.globalResidualPrefillAnchorSelections = mGlobalResidualPrefillAnchorSelections;
    result.globalResidualDecodeAnchorOpportunities = mGlobalResidualDecodeAnchorOpportunities;
    result.globalResidualDecodeAnchorSelections = mGlobalResidualDecodeAnchorSelections;
    result.globalResidualMeasuredUnprofitableOpportunities = mGlobalResidualMeasuredUnprofitableOpportunities;
    result.globalResidualMeasuredUnprofitableSelections = mGlobalResidualMeasuredUnprofitableSelections;
    result.globalResidualCoveringCostHits = mGlobalResidualCoveringCostHits;
    result.globalPdSelections = mGlobalPdSelections;
    result.globalSafeProbes = mGlobalSafeProbes;
    result.globalEncoderOverlapOpportunities = mGlobalEncoderOverlapOpportunities;
    result.globalEncoderOverlapKnownCosts = mGlobalEncoderOverlapKnownCosts;
    result.globalEncoderOverlapNoSamples = mGlobalEncoderOverlapNoSamples;
    result.globalEncoderOverlapInsufficientSamples = mGlobalEncoderOverlapInsufficientSamples;
    result.globalEncoderOverlapUnprofitable = mGlobalEncoderOverlapUnprofitable;
    result.globalEncoderOverlapSafeProbeEligible = mGlobalEncoderOverlapSafeProbeEligible;
    result.globalEncoderOverlapProbeDisabled = mGlobalEncoderOverlapProbeDisabled;
    result.globalEncoderOverlapProbeIntervalBlocked = mGlobalEncoderOverlapProbeIntervalBlocked;
    result.globalEncoderOverlapProbeSlackBlocked = mGlobalEncoderOverlapProbeSlackBlocked;
    result.globalWarmupDecisions = mGlobalWarmupDecisions;
    result.globalWarmupPrefillCandidates = mGlobalWarmupPrefillCandidates;
    result.globalWarmupDecodeCandidates = mGlobalWarmupDecodeCandidates;
    result.globalActionFidelityViolations = mGlobalActionFidelityViolations;
    result.globalHostDecisionSamples = mGlobalDecisionCostSamples;
    size_t const retainedDecisionCosts
        = std::min(mGlobalDecisionCostSamples, static_cast<size_t>(kGLOBAL_DECISION_COST_WINDOW));
    result.globalHostDecisionMeanUs
        = retainedDecisionCosts > 0U ? mGlobalDecisionCostSumUs / static_cast<double>(retainedDecisionCosts) : 0.0;
    result.globalHostDecisionMaxUs = mGlobalDecisionCostMaxUs;
    if (retainedDecisionCosts > 0U)
    {
        std::array<double, kGLOBAL_DECISION_COST_WINDOW> sorted = mGlobalDecisionCostsUs;
        std::sort(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(retainedDecisionCosts));
        size_t const p95Index = (95U * retainedDecisionCosts + 99U) / 100U - 1U;
        result.globalHostDecisionP95Us = sorted[p95Index];
    }
    result.lastGlobalFirstTokenCriticalPathUs = mLastGlobalFirstTokenCriticalPathUs;
    result.globalEncoderArrivalWaitPeriods = mGlobalEncoderArrivalWaitPeriods;
    result.globalEncoderArrivalWaitExpirations = mGlobalEncoderArrivalWaitExpirations;
    result.lastGlobalEncoderArrivalWaitUs = mLastGlobalEncoderArrivalWaitUs;
    result.globalFormationLookaheads = mGlobalFormationLookaheads;
    result.globalFormationPredictedRows = mGlobalFormationPredictedRows;
    result.globalFormationSelectionChanges = mGlobalFormationSelectionChanges;
    result.globalFormationPdSelections = mGlobalFormationPdSelections;
    result.globalFormationOverlapSelections = mGlobalFormationOverlapSelections;
    result.globalFormationH2Agreements = mGlobalFormationH2Agreements;
    result.globalFormationPostPolicyOverrides = mGlobalFormationPostPolicyOverrides;
    result.globalFormationRegretSamples = mGlobalFormationRegretSamples;
    result.globalFormationPositiveRegrets = mGlobalFormationPositiveRegrets;
    result.globalFormationPredictedRegretUs = mGlobalFormationPredictedRegretUs;
    result.maxGlobalFormationPredictedRegretUs = mMaxGlobalFormationPredictedRegretUs;
    result.lastGlobalFormationPredictedRows = mLastGlobalFormationPredictedRows;
    result.lastGlobalFormationHorizonUs = mLastGlobalFormationHorizonUs;
    result.lastGlobalFormationCostGapUs = mLastGlobalFormationCostGapUs;
    result.lastGlobalFormationPlannerUs = mLastGlobalFormationPlannerUs;
    result.lastGlobalFormationSnapshotId = mLastGlobalFormationSnapshotId;
    result.lastGlobalFormationH2Action = mLastGlobalFormationH2Action;
    result.lastGlobalFormationOracleAction = mLastGlobalFormationOracleAction;
    result.lastGlobalFormationDecodeViolationUs = mLastGlobalFormationDecodeViolationUs;
    PhaseFormationRealizedTelemetry const& realized = mGlobalFormationRealizedTracker.telemetry();
    result.globalFormationRealizedEpisodesStarted = realized.episodesStarted;
    result.globalFormationRealizedEpisodesCompleted = realized.episodesCompleted;
    result.globalFormationRealizedEpisodesTruncated = realized.episodesTruncated;
    result.globalFormationRealizedDecodeServices = realized.decodeServices;
    result.globalFormationRealizedDecodeBudgets = realized.decodeBudgets;
    result.globalFormationRealizedDecodeServiceViolations = realized.decodeServiceViolations;
    result.globalFormationRealizedDecodeServiceGapUs = realized.decodeServiceGapUs;
    result.maxGlobalFormationRealizedDecodeServiceGapUs = realized.maxDecodeServiceGapUs;
    result.globalFormationRealizedDecodeServiceViolationUs = realized.decodeServiceViolationUs;
    result.maxGlobalFormationRealizedDecodeServiceViolationUs = realized.maxDecodeServiceViolationUs;
    if (realized.lastEpisode.has_value())
    {
        PhaseFormationRealizedEpisode const& episode = *realized.lastEpisode;
        result.lastGlobalFormationRealizedEpisodeId = episode.episodeId;
        result.lastGlobalFormationRealizedSnapshotId = episode.snapshotId;
        result.lastGlobalFormationRealizedSelectedAction = episode.selectedAction;
        result.lastGlobalFormationRealizedMyopicAction = episode.myopicAction;
        result.lastGlobalFormationRealizedOracleAction = episode.oracleAction;
        result.lastGlobalFormationRealizedDispatches = episode.dispatches.size();
        result.lastGlobalFormationRealizedEncoderRows = episode.encoderRows;
        result.lastGlobalFormationRealizedPrefillRows = episode.prefillRows;
        result.lastGlobalFormationRealizedDecodeRows = episode.decodeRows;
        result.lastGlobalFormationRealizedFirstDecodeRows = episode.firstDecodeRows;
        result.lastGlobalFormationRealizedMaxDecodeRows = episode.maxDecodeRows;
        result.lastGlobalFormationRealizedDecodeServiced = episode.decodeServiced;
        result.lastGlobalFormationRealizedDecodeBudgetKnown = std::isfinite(episode.decodeServiceBudgetUs);
        result.lastGlobalFormationRealizedTruncated = episode.truncated;
        result.lastGlobalFormationRealizedPredictedRegretUs = episode.predictedRegretUs;
        result.lastGlobalFormationRealizedDecodeBudgetUs
            = std::isfinite(episode.decodeServiceBudgetUs) ? episode.decodeServiceBudgetUs : 0.0;
        result.lastGlobalFormationRealizedDecodeServiceGapUs = episode.decodeServiceGapUs;
        result.lastGlobalFormationRealizedDecodeCompletionVisibleUs = episode.decodeCompletionVisibleUs;
        result.lastGlobalFormationRealizedHorizonCompletionVisibleUs = episode.horizonCompletionVisibleUs;
        result.lastGlobalFormationRealizedDecodeServiceViolationUs = episode.decodeServiceViolationUs;
    }
    PhaseContextualPdTelemetry const& epTelemetry
        = mRuntimeCostTracker->contextualPairTelemetry(PhaseContextualPairKind::kEncoderPrefill);
    result.contextualEpReady = mContextualEpReady;
    result.contextualEpDecisionDisagreements = mContextualEpDecisionDisagreements;
    result.contextualEpPredictions = epTelemetry.predictions;
    result.contextualEpObservations = epTelemetry.observations;
    result.contextualEpRejectedObservations = epTelemetry.rejectedObservations;
    result.contextualEpPositiveSelections = epTelemetry.positiveSelections;
    result.contextualEpNegativeSelections = epTelemetry.negativeSelections;
    result.contextualEpExplorations = epTelemetry.explorations;
    result.contextualEpLastReward = epTelemetry.lastReward;
    result.contextualEpLastMean = epTelemetry.lastMean;
    result.contextualEpLastUncertainty = epTelemetry.lastUncertainty;
    result.contextualEpLastLowerConfidenceBound = epTelemetry.lastLowerConfidenceBound;
    PhaseContextualPdTelemetry const& edTelemetry
        = mRuntimeCostTracker->contextualPairTelemetry(PhaseContextualPairKind::kEncoderDecode);
    result.contextualEdReady = mContextualEdReady;
    result.contextualEdDecisionDisagreements = mContextualEdDecisionDisagreements;
    result.contextualEdPredictions = edTelemetry.predictions;
    result.contextualEdObservations = edTelemetry.observations;
    result.contextualEdRejectedObservations = edTelemetry.rejectedObservations;
    result.contextualEdPositiveSelections = edTelemetry.positiveSelections;
    result.contextualEdNegativeSelections = edTelemetry.negativeSelections;
    result.contextualEdExplorations = edTelemetry.explorations;
    result.contextualEdLastReward = edTelemetry.lastReward;
    result.contextualEdLastMean = edTelemetry.lastMean;
    result.contextualEdLastUncertainty = edTelemetry.lastUncertainty;
    result.contextualEdLastLowerConfidenceBound = edTelemetry.lastLowerConfidenceBound;
    result.activeGlobalPlanId = mGlobalExecutionLease.has_value() ? mGlobalExecutionLease->planId : 0U;
    result.globalPlannedOutstanding
        = mGlobalExecutionLease.has_value() ? mGlobalExecutionLease->allowedOutstanding : PhaseExecutionSet::kNone;
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

void PhaseThreeCoordinator::resetGlobalDecisionCostTelemetry() noexcept
{
    mGlobalDecisionCostsUs.fill(0.0);
    mGlobalDecisionCostSamples = 0U;
    mGlobalDecisionCostCursor = 0U;
    mGlobalDecisionCostSumUs = 0.0;
    mGlobalDecisionCostMaxUs = 0.0;
}

void PhaseThreeCoordinator::recordGlobalDecisionCost(std::chrono::steady_clock::time_point startedAt) noexcept
{
    double const elapsedUs
        = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - startedAt).count();
    if (mGlobalDecisionCostSamples >= kGLOBAL_DECISION_COST_WINDOW)
    {
        mGlobalDecisionCostSumUs -= mGlobalDecisionCostsUs[mGlobalDecisionCostCursor];
    }
    mGlobalDecisionCostsUs[mGlobalDecisionCostCursor] = elapsedUs;
    mGlobalDecisionCostCursor = (mGlobalDecisionCostCursor + 1U) % kGLOBAL_DECISION_COST_WINDOW;
    ++mGlobalDecisionCostSamples;
    mGlobalDecisionCostSumUs += elapsedUs;
    mGlobalDecisionCostMaxUs = std::max(mGlobalDecisionCostMaxUs, elapsedUs);
}

std::vector<PhaseGlobalOverlapCostRecord> PhaseThreeCoordinator::globalCalibrationDiagnostics() const
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

void PhaseThreeCoordinator::setGlobalWarmupProbeMode(bool active)
{
    ELLM_CHECK(empty(), "Global encoder overlap warmup mode may only change while the coordinator is idle");
    if (active)
    {
        mGlobalCalibrationKeys.clear();
        mGlobalCalibrationOpportunities.clear();
    }
    mGlobalWarmupProbeMode = active;
}

void PhaseThreeCoordinator::setTimelineCallback(std::function<void(PhaseTimelineEvent const&)> timelineCallback)
{
    ELLM_CHECK(empty(), "Three-phase timeline callback can only change while the coordinator is idle");
    mTimelineCallback = std::move(timelineCallback);
}

void PhaseThreeCoordinator::setActivityTimeline(PhaseActivityTimelineRecorder* timeline)
{
    ELLM_CHECK(empty(), "Three-phase activity timeline can only change while idle");
    mActivityTimeline = timeline;
    mVision.setActivityTimeline(timeline);
    mServer.setActivityTimeline(timeline);
}

void PhaseThreeCoordinator::setUnifiedEventCallback(
    std::function<void(PhaseUnifiedEvent const&)> unifiedEventCallback, bool detailedDecisionSnapshots)
{
    ELLM_CHECK(empty(), "Three-phase unified event callback can only change while idle");
    mUnifiedEventCallback = std::move(unifiedEventCallback);
    mUnifiedDetailedDecisionSnapshots = detailedDecisionSnapshots;
    mPreviousUnifiedInFlight.reset();
    mUnifiedDecisionByPlan.clear();
    mUnifiedAllowedOutstandingByExecution.clear();
}

void PhaseThreeCoordinator::setEncoderBatchMetricCallback(
    std::function<void(PhaseVisionEncoderBatchMetric const&)> encoderBatchMetricCallback)
{
    ELLM_CHECK(empty(), "Three-phase encoder metric callback can only change while the coordinator is idle");
    mEncoderBatchMetricCallback = std::move(encoderBatchMetricCallback);
}

void PhaseThreeCoordinator::setFormationEpisodeCallback(
    std::function<void(PhaseFormationRealizedEpisode const&)> formationEpisodeCallback)
{
    ELLM_CHECK(empty(), "Three-phase formation episode callback can only change while idle");
    mFormationEpisodeCallback = std::move(formationEpisodeCallback);
}

void PhaseThreeCoordinator::setEventCallbacks(std::function<void(IndependentPhaseServerToken&&)> tokenCallback,
    std::function<void(IndependentPhaseServerCompletion&&)> completionCallback)
{
    ELLM_CHECK(empty(), "Three-phase event callbacks can only change while the coordinator is idle");
    mServer.setEventCallbacks(std::move(tokenCallback),
        [this, completionCallback = std::move(completionCallback)](
            IndependentPhaseServerCompletion&& completion) mutable {
            observeServerCompletion(completion.requestId);
            if (completionCallback)
            {
                completionCallback(std::move(completion));
            }
        });
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
        observeServerCompletion(completion->requestId);
    }
    return completion;
}

void PhaseThreeCoordinator::observeServerCompletion(uint64_t requestId)
{
    mRequestIds.erase(requestId);
    eraseTpotTarget(requestId);
    mDownstreamRequestBytes.erase(requestId);
}

void PhaseThreeCoordinator::emitUnifiedEvent(PhaseUnifiedEvent event)
{
    if (!mUnifiedEventCallback)
    {
        return;
    }
    event.eventId = ++mUnifiedEventSequence;
    event.hostMonotonicNs = event.hostMonotonicNs > 0U ? event.hostMonotonicNs : phaseTimelineNowNs();
    mUnifiedEventCallback(event);
}

PhaseInFlightSnapshot PhaseThreeCoordinator::unifiedInFlightSnapshot(
    uint64_t hostSnapshotNs, bool includeRequestIds) const
{
    uint64_t const timestampNs = hostSnapshotNs > 0U ? hostSnapshotNs : phaseTimelineNowNs();
    PhaseInFlightSnapshot result = mServer.arbitrationSnapshot(includeRequestIds).inFlight;
    result.hostSnapshotNs = timestampNs;
    for (PhaseInFlightWorkSnapshot& work : result.work)
    {
        work.dispatchAgeUs = timestampNs >= work.dispatchHostNs
            ? static_cast<double>(timestampNs - work.dispatchHostNs) / 1000.0
            : 0.0;
    }
    if (mEncoderGpuSubmitted && !mEncoding.empty())
    {
        PhaseInFlightWorkSnapshot encoder;
        encoder.phase = PhaseUnifiedPhase::kEncoder;
        encoder.status = mVision.ready(mEncoding.front().requestId) ? PhaseInFlightStatus::kCompletionReady
                                                                    : PhaseInFlightStatus::kRunning;
        encoder.executionId = mEncoderExecutionId;
        encoder.activityCorrelationId = mEncoderActivityCorrelationId;
        encoder.planId = mEncoderPlanId;
        encoder.actionId = mEncoderActionId;
        encoder.dispatchHostNs = mEncoderDispatchHostNs;
        encoder.prepareStartHostNs = mEncoderPrepareStartHostNs;
        encoder.prepareEndHostNs = mEncoderPrepareEndHostNs;
        encoder.executeStartHostNs = mEncoderExecuteStartHostNs;
        encoder.executeEndHostNs = mEncoderExecuteEndHostNs;
        encoder.dispatchAgeUs = timestampNs >= mEncoderDispatchHostNs
            ? static_cast<double>(timestampNs - mEncoderDispatchHostNs) / 1000.0
            : 0.0;
        if (includeRequestIds)
        {
            encoder.requestIds.reserve(mEncoding.size());
            for (PendingVisionRequest const& request : mEncoding)
            {
                encoder.requestIds.push_back(request.requestId);
            }
        }
        encoder.work.encoderRows = static_cast<int32_t>(mEncoding.size());
        result.work.push_back(encoder);
        result.outstanding = result.outstanding | PhaseExecutionSet::kEncoder;
    }
    return result;
}

void PhaseThreeCoordinator::recordUnifiedDecision(PhaseGlobalActionCandidate const& candidate,
    PhaseGlobalDispatchPlan const& plan, std::vector<PhaseGlobalActionCandidate> const* candidateFrontier,
    PhaseGlobalSelectionAudit const* selectorAudit)
{
    if (!mUnifiedEventCallback)
    {
        return;
    }
    PhaseUnifiedEvent event;
    if (selectorAudit != nullptr)
    {
        event.selectorAudit = std::make_shared<PhaseGlobalSelectionAudit>(*selectorAudit);
    }
    event.kind = PhaseUnifiedEventKind::kDecision;
    event.decisionId = plan.snapshotEpoch;
    event.policyDecisionSequence = mGlobalDecisionSequence;
    event.snapshotId = plan.snapshotEpoch;
    event.planId = plan.planId;
    event.actionId = candidate.candidateId;
    event.incrementalActionId = plan.incrementalAction.actionId;
    event.requestedStartSkewPercent = static_cast<int32_t>(plan.incrementalAction.key.startSkew);
    event.requestedDirection = plan.incrementalAction.key.direction;
    event.actionKind = candidate.key.kind;
    event.selectedActionId = candidate.candidateId;
    event.cohort = unifiedCandidateWork(candidate);
    event.requestIds = candidate.requestIds;
    event.inFlight = unifiedInFlightSnapshot(0U, mUnifiedDetailedDecisionSnapshots);
    // The dispatch plan is the mechanism authority for this transition. A
    // residual P/D augmentation may be recorded after enqueue has returned
    // but before the server's next arbitration snapshot exposes the incumbent
    // context, so a fresh observational snapshot can transiently be empty.
    event.outstandingBefore = plan.incrementalAction.key.outstandingBefore;
    event.dispatchMode = phaseUnifiedDispatchMode(event.outstandingBefore, event.actionKind);
    event.incumbentPhase = phaseUnifiedDirectionIncumbentPhase(event.requestedDirection);
    event.newcomerPhase = phaseUnifiedDirectionNewcomerPhase(event.requestedDirection);
    auto const incumbent = std::find_if(event.inFlight.work.begin(), event.inFlight.work.end(),
        [&](auto const& work) { return work.phase == event.incumbentPhase; });
    if (incumbent != event.inFlight.work.end())
    {
        event.incumbentExecutionId = incumbent->executionId;
        event.incumbentDispatchAgeUs = incumbent->dispatchAgeUs;
    }
    event.plannedOutstanding = plan.allowedOutstanding;
    event.ready.encoderRows = static_cast<int32_t>(mPending.size());
    IndependentPhaseServerArbitrationSnapshot const server
        = mServer.arbitrationSnapshot(mUnifiedDetailedDecisionSnapshots);
    event.ready.prefillRows = static_cast<int32_t>(server.prefillQueued);
    event.ready.prefillTokens = server.prefillCandidateTokens;
    event.ready.decodeRows = static_cast<int32_t>(server.decodeQueued);
    event.ready.decodeContextTokens = server.decodeCandidateTokens;
    if (mUnifiedDetailedDecisionSnapshots)
    {
        event.serviceClocks = server.serviceClocks;
        auto appendServiceClock = [&](uint64_t requestId, PhaseSchedulingHints const& hints) {
            auto const existing = std::find_if(event.serviceClocks.begin(), event.serviceClocks.end(),
                [requestId](PhaseServiceClock const& clock) { return clock.requestId == requestId; });
            if (existing == event.serviceClocks.end())
            {
                event.serviceClocks.push_back({requestId, phaseServiceHostNs(hints.submittedAt), 0U});
            }
        };
        event.readyEncoderRequestIds.reserve(mPending.size());
        for (PendingVisionRequest const& request : mPending)
        {
            event.readyEncoderRequestIds.push_back(request.requestId);
            appendServiceClock(request.requestId, request.scheduling);
        }
        for (PendingVisionRequest const& request : mEncoding)
        {
            appendServiceClock(request.requestId, request.scheduling);
        }
        event.readyPrefillRequestIds = server.prefillRequestIds;
        event.readyPrefillTokenCounts = server.prefillTokenCounts;
        event.readyPrefillRequestIds.reserve(event.readyPrefillRequestIds.size() + mReadyPrefill.size());
        event.readyPrefillTokenCounts.reserve(event.readyPrefillTokenCounts.size() + mReadyPrefill.size());
        for (ReadyPrefillRequest const& request : mReadyPrefill)
        {
            event.readyPrefillRequestIds.push_back(request.requestId);
            appendServiceClock(request.requestId, request.scheduling);
            event.readyPrefillTokenCounts.push_back(static_cast<int32_t>(request.promptTokens.size()));
        }
        event.readyDecodeRequestIds = server.decodeRequestIds;
        event.readyDecodeContextLengths = server.decodeContextLengths;
    }
    event.pagePoolAllocatedBundles = server.pagePoolAllocatedBundles;
    event.pageReservationGuaranteedBundles = server.pageReservationGuaranteedBundles;
    event.visionPayloadBytes = server.visionPayloadBytes + mReadyPrefillBytes + mEstimatedEncodedBytes;
    event.kvOwnershipSignature = server.kvOwnershipSignature;
    event.visionLeaseSignature = server.visionLeaseSignature;
    auto appendCandidate = [&event](PhaseGlobalActionCandidate const& source) {
        PhaseUnifiedCandidateSnapshot snapshot;
        snapshot.actionId = source.candidateId;
        snapshot.key = source.key;
        size_t committedBytes = source.memory.managedBytes;
        if (source.memory.immediateReclaimObserved)
        {
            committedBytes = committedBytes > source.memory.immediateReclaimBytes
                ? committedBytes - source.memory.immediateReclaimBytes
                : 0U;
        }
        size_t const peakBytes = saturatedAdd(
            saturatedAdd(committedBytes, source.memory.allocateBytes), source.memory.guaranteedGrowthBytes);
        bool const memorySafe = source.memory.budgetBytes == 0U || peakBytes <= source.memory.budgetBytes;
        bool const waitSafe = source.key.kind != PhaseGlobalActionKind::kWait || source.concreteWaitEvent;
        snapshot.legal = source.dependencySafe && source.contextSafe && source.shapeSafe && memorySafe && waitSafe;
        snapshot.requestIds = source.requestIds;
        snapshot.predictedCompletionUs = source.predictedHorizonUs > 0.0 ? source.predictedHorizonUs
            : source.predictedMakespanUs > 0.0                           ? source.predictedMakespanUs
                                                                         : source.predictedBlockingUs;
        snapshot.uncertaintyUs = source.uncertaintyUs;
        snapshot.predictedCostSource = source.predictedCostSource;
        snapshot.referenceCostSource = source.referenceCostSource;
        snapshot.scalarDecisionCostKnown = source.decisionCostKnown;
        snapshot.contextualScalarAuthorityApplied = source.contextualScalarAuthorityApplied;
        snapshot.scalarDecisionMakespanUs = source.decisionMakespanUs;
        snapshot.scalarProtectedCompletions = source.protectedCompletions;
        event.candidates.push_back(std::move(snapshot));
    };
    if (mUnifiedDetailedDecisionSnapshots && candidateFrontier != nullptr)
    {
        for (PhaseGlobalActionCandidate const& frontierCandidate : *candidateFrontier)
        {
            appendCandidate(frontierCandidate);
        }
    }
    else if (mUnifiedDetailedDecisionSnapshots)
    {
        appendCandidate(candidate);
    }
    if (mUnifiedDetailedDecisionSnapshots && selectorAudit != nullptr)
    {
        std::vector<PhaseUnifiedCandidateSnapshot> activeCandidates = std::move(event.candidates);
        for (PhaseGlobalActionCandidate const& mechanismCandidate : selectorAudit->mechanismInputs)
        {
            appendCandidate(mechanismCandidate);
        }
        event.mechanismCandidates = std::move(event.candidates);
        event.candidates = std::move(activeCandidates);
    }
    if (candidateFrontier != nullptr)
    {
        PhaseGlobalDecision const scalarDecision = mGlobalScheduler.select(*candidateFrontier);
        if (scalarDecision.selectedIndex.has_value())
        {
            event.scalarSelectedActionId = (*candidateFrontier)[*scalarDecision.selectedIndex].candidateId;
        }
        std::vector<PhaseGlobalActionCandidate> nonContextualFrontier = *candidateFrontier;
        for (PhaseGlobalActionCandidate& fallback : nonContextualFrontier)
        {
            phaseRestoreNonContextualPolicy(fallback);
        }
        PhaseGlobalDecision const fallbackDecision = mGlobalScheduler.select(nonContextualFrontier);
        if (fallbackDecision.selectedIndex.has_value())
        {
            event.nonContextualSelectedActionId = nonContextualFrontier[*fallbackDecision.selectedIndex].candidateId;
        }
    }
    event.scalarFormation = mLastScalarFormation;
    event.snapshotSignature = phaseUnifiedSnapshotSignature(event);
    event.scalarPolicyStateSignature = phaseUnifiedScalarPolicyStateSignature(event);
    event.strictSnapshotSignature = phaseUnifiedStrictSnapshotSignature(event);
    event.dispatchSignature = phaseUnifiedDispatchSignature(event);
    mUnifiedDecisionByPlan[plan.planId] = event;
    emitUnifiedEvent(std::move(event));
}

void PhaseThreeCoordinator::observeUnifiedInFlightTransitions()
{
    if (!mUnifiedEventCallback)
    {
        return;
    }
    if (mActivityTimeline != nullptr)
    {
        static_cast<void>(mActivityTimeline->poll());
    }
    PhaseInFlightSnapshot const current = unifiedInFlightSnapshot();
    PhaseInFlightSnapshot previous = mPreviousUnifiedInFlight.value_or(PhaseInFlightSnapshot{});
    auto sameWork = [](PhaseInFlightWorkSnapshot const& left, PhaseInFlightWorkSnapshot const& right) {
        return left.phase == right.phase && left.executionId == right.executionId;
    };
    auto allowOutstanding = [this](uint64_t executionId, PhaseExecutionSet allowed) {
        mUnifiedAllowedOutstandingByExecution[executionId]
            = mUnifiedAllowedOutstandingByExecution[executionId] | allowed;
    };

    // A residual action can extend an incumbent execution's plan without
    // resubmitting that phase. Promote only plans whose newly launched member
    // is observable; a decision that failed before dispatch must not relax
    // fidelity for its incumbent execution.
    for (auto const& [planId, decision] : mUnifiedDecisionByPlan)
    {
        bool const launched = std::any_of(current.work.begin(), current.work.end(),
            [planId](PhaseInFlightWorkSnapshot const& work) { return work.planId == planId; });
        if (!launched)
        {
            continue;
        }
        for (PhaseInFlightWorkSnapshot const& incumbent : decision.inFlight.work)
        {
            PhaseExecutionSet const incumbentPhase = phaseExecutionSetForUnifiedPhase(incumbent.phase);
            if (phaseExecutionSetContains(decision.plannedOutstanding, incumbentPhase))
            {
                allowOutstanding(incumbent.executionId, decision.plannedOutstanding);
            }
        }
        // Compact telemetry deliberately omits detailed in-flight snapshots.
        // Once the newcomer is observable, extend every retained incumbent
        // that belongs to the transition's authoritative before-set. This is
        // the same residual lease, not an unplanned overlap.
        for (PhaseInFlightWorkSnapshot const& incumbent : previous.work)
        {
            PhaseExecutionSet const incumbentPhase = phaseExecutionSetForUnifiedPhase(incumbent.phase);
            if (phaseExecutionSetContains(decision.outstandingBefore, incumbentPhase)
                && phaseExecutionSetContains(decision.plannedOutstanding, incumbentPhase))
            {
                allowOutstanding(incumbent.executionId, decision.plannedOutstanding);
            }
        }
        for (PhaseInFlightWorkSnapshot const& work : current.work)
        {
            if (work.planId == planId)
            {
                allowOutstanding(work.executionId, decision.plannedOutstanding);
            }
        }
    }

    for (PhaseInFlightWorkSnapshot const& prior : previous.work)
    {
        bool const retained = std::any_of(current.work.begin(), current.work.end(),
            [&](PhaseInFlightWorkSnapshot const& work) { return sameWork(prior, work); });
        if (retained)
        {
            continue;
        }
        PhaseUnifiedEvent completion;
        completion.kind = PhaseUnifiedEventKind::kCompletion;
        completion.executionId = prior.executionId;
        completion.planId = prior.planId;
        completion.actionId = prior.actionId;
        completion.phase = prior.phase;
        completion.observedOutstanding = current.outstanding;
        completion.cohort = prior.work;
        completion.requestIds = prior.requestIds;
        completion.completionVisibleHostNs = current.hostSnapshotNs;
        auto const decision = mUnifiedDecisionByPlan.find(prior.planId);
        if (decision != mUnifiedDecisionByPlan.end())
        {
            completion.decisionId = decision->second.decisionId;
            completion.snapshotId = decision->second.snapshotId;
            completion.strictSnapshotSignature = decision->second.strictSnapshotSignature;
            completion.dispatchSignature = decision->second.dispatchSignature;
            completion.actionKind = decision->second.actionKind;
            completion.incrementalActionId = decision->second.incrementalActionId;
            completion.requestedStartSkewPercent = decision->second.requestedStartSkewPercent;
            completion.requestedDirection = decision->second.requestedDirection;
            completion.direction = decision->second.requestedDirection;
            completion.dispatchMode = decision->second.dispatchMode;
            completion.incumbentPhase = decision->second.incumbentPhase;
            completion.incumbentExecutionId = decision->second.incumbentExecutionId;
            completion.incumbentDispatchAgeUs = decision->second.incumbentDispatchAgeUs;
            completion.newcomerPhase = decision->second.newcomerPhase;
            completion.plannedOutstanding = decision->second.plannedOutstanding;
            auto const augmentedOutstanding = mUnifiedAllowedOutstandingByExecution.find(prior.executionId);
            PhaseExecutionSet const allowedOutstanding
                = augmentedOutstanding != mUnifiedAllowedOutstandingByExecution.end()
                ? augmentedOutstanding->second
                : decision->second.plannedOutstanding;
            completion.actionFidelity = phaseExecutionSetIsSubset(previous.outstanding, allowedOutstanding);
            bool const actionIdentityMatches = phaseUnifiedActionIdentityMatches(
                decision->second.dispatchMode, decision->second.actionId, prior.actionId);
            if (!actionIdentityMatches)
            {
                completion.actionFidelity = false;
                completion.actionFidelityReason = PhaseUnifiedFidelityReason::kActionIdMismatch;
            }
            else if (!completion.actionFidelity)
            {
                completion.actionFidelityReason = PhaseUnifiedFidelityReason::kOutstandingMismatch;
            }
        }
        else
        {
            completion.actionFidelity = false;
            completion.actionFidelityReason = PhaseUnifiedFidelityReason::kMissingDecision;
        }
        if (prior.phase == PhaseUnifiedPhase::kEncoder)
        {
            completion.gpuDurationUs = static_cast<double>(mLastEncoderGpuMs) * 1000.0;
        }
        else if (std::optional<PhaseDispatchMetrics> const& metrics = mServer.schedulerTelemetry().lastDispatch;
            metrics.has_value() && (metrics->globalPlanId == prior.planId || metrics->dispatchIndex == prior.planId))
        {
            completion.gpuDurationUs = prior.phase == PhaseUnifiedPhase::kPrefill
                ? static_cast<double>(metrics->prefillGpuMs) * 1000.0
                : static_cast<double>(metrics->decodeGpuMs) * 1000.0;
        }
        if (mActivityTimeline != nullptr)
        {
            std::vector<PhaseActivityInterval> const intervals = mActivityTimeline->intervals();
            auto const interval = std::find_if(intervals.rbegin(), intervals.rend(), [&](auto const& candidate) {
                return unifiedPhaseActivityMatches(prior.phase, candidate)
                    && candidate.correlationId == prior.activityCorrelationId;
            });
            if (interval != intervals.rend())
            {
                completion.gpuStartUs = static_cast<double>(interval->startMs) * 1000.0;
                completion.gpuEndUs = static_cast<double>(interval->endMs) * 1000.0;
                completion.gpuDurationUs = *completion.gpuEndUs - *completion.gpuStartUs;
            }
            if (decision != mUnifiedDecisionByPlan.end()
                && completion.dispatchMode != PhaseUnifiedDispatchMode::kSingle)
            {
                auto findExecution = [&](PhaseUnifiedPhase phase) -> PhaseInFlightWorkSnapshot const* {
                    auto const priorWork = std::find_if(previous.work.begin(), previous.work.end(),
                        [phase](PhaseInFlightWorkSnapshot const& work) { return work.phase == phase; });
                    if (priorWork != previous.work.end())
                    {
                        return &*priorWork;
                    }
                    auto const decisionWork
                        = std::find_if(decision->second.inFlight.work.begin(), decision->second.inFlight.work.end(),
                            [phase](PhaseInFlightWorkSnapshot const& work) { return work.phase == phase; });
                    return decisionWork != decision->second.inFlight.work.end() ? &*decisionWork : nullptr;
                };
                auto findInterval = [&](PhaseInFlightWorkSnapshot const* work) -> PhaseActivityInterval const* {
                    if (work == nullptr)
                    {
                        return nullptr;
                    }
                    auto const match = std::find_if(intervals.rbegin(), intervals.rend(), [&](auto const& candidate) {
                        return unifiedPhaseActivityMatches(work->phase, candidate)
                            && candidate.correlationId == work->activityCorrelationId;
                    });
                    return match != intervals.rend() ? &*match : nullptr;
                };
                PhaseInFlightWorkSnapshot const* incumbentWork = findExecution(completion.incumbentPhase);
                PhaseInFlightWorkSnapshot const* newcomerWork = findExecution(completion.newcomerPhase);
                if (incumbentWork != nullptr)
                {
                    completion.incumbentExecutionId = incumbentWork->executionId;
                }
                if (newcomerWork != nullptr)
                {
                    completion.newcomerExecutionId = newcomerWork->executionId;
                }
                PhaseActivityInterval const* incumbentInterval = findInterval(incumbentWork);
                PhaseActivityInterval const* newcomerInterval = findInterval(newcomerWork);
                if (incumbentInterval != nullptr)
                {
                    completion.incumbentGpuCompletionUs = static_cast<double>(incumbentInterval->endMs) * 1000.0;
                }
                if (newcomerInterval != nullptr)
                {
                    completion.newcomerGpuCompletionUs = static_cast<double>(newcomerInterval->endMs) * 1000.0;
                }
                if (incumbentInterval != nullptr && newcomerInterval != nullptr)
                {
                    double const incumbentDurationUs = std::max(
                        1.0, static_cast<double>(incumbentInterval->endMs - incumbentInterval->startMs) * 1000.0);
                    double const startDeltaUs = std::max(
                        0.0, static_cast<double>(newcomerInterval->startMs - incumbentInterval->startMs) * 1000.0);
                    bool const intervalsOverlap = newcomerInterval->startMs < incumbentInterval->endMs;
                    completion.observedStartSkewPercent = static_cast<int32_t>(
                        phaseStartSkewBucketFromFraction(startDeltaUs / incumbentDurationUs, intervalsOverlap));
                }
            }
        }
        emitUnifiedEvent(std::move(completion));
        mUnifiedAllowedOutstandingByExecution.erase(prior.executionId);
    }

    std::vector<PhaseInFlightWorkSnapshot> added;
    for (PhaseInFlightWorkSnapshot const& work : current.work)
    {
        bool const retained = std::any_of(previous.work.begin(), previous.work.end(),
            [&](PhaseInFlightWorkSnapshot const& prior) { return sameWork(prior, work); });
        if (!retained)
        {
            added.push_back(work);
        }
    }
    std::sort(added.begin(), added.end(), [](auto const& left, auto const& right) {
        if (left.dispatchHostNs != right.dispatchHostNs)
        {
            return left.dispatchHostNs < right.dispatchHostNs;
        }
        return static_cast<uint8_t>(left.phase) < static_cast<uint8_t>(right.phase);
    });
    PhaseExecutionSet observedBefore{PhaseExecutionSet::kNone};
    std::vector<PhaseInFlightWorkSnapshot> observedWork;
    for (PhaseInFlightWorkSnapshot const& prior : previous.work)
    {
        bool const retained = std::any_of(current.work.begin(), current.work.end(),
            [&](PhaseInFlightWorkSnapshot const& work) { return sameWork(prior, work); });
        if (retained)
        {
            observedBefore = observedBefore | phaseExecutionSetForUnifiedPhase(prior.phase);
            observedWork.push_back(prior);
        }
    }
    for (PhaseInFlightWorkSnapshot const& work : added)
    {
        PhaseUnifiedEvent dispatch;
        dispatch.kind = PhaseUnifiedEventKind::kDispatch;
        dispatch.executionId = work.executionId;
        dispatch.planId = work.planId;
        dispatch.actionId = work.actionId;
        dispatch.phase = work.phase;
        dispatch.outstandingBefore = observedBefore;
        dispatch.direction = phaseUnifiedActionDirection(observedBefore, work.phase);
        PhaseUnifiedPhase const incumbentPhase = phaseUnifiedDirectionIncumbentPhase(dispatch.direction);
        auto const incumbent = std::find_if(observedWork.rbegin(), observedWork.rend(),
            [incumbentPhase](PhaseInFlightWorkSnapshot const& candidate) { return candidate.phase == incumbentPhase; });
        if (incumbent != observedWork.rend())
        {
            dispatch.incumbentPhase = incumbent->phase;
            dispatch.incumbentExecutionId = incumbent->executionId;
        }
        dispatch.cohort = work.work;
        dispatch.requestIds = work.requestIds;
        dispatch.enqueueHostNs = work.dispatchHostNs;
        dispatch.prepareStartHostNs = work.prepareStartHostNs;
        dispatch.prepareEndHostNs = work.prepareEndHostNs;
        dispatch.executeStartHostNs = work.executeStartHostNs;
        dispatch.executeEndHostNs = work.executeEndHostNs;
        dispatch.graphReplay = work.graphReplay;
        dispatch.observedOutstanding = current.outstanding;
        auto const decision = mUnifiedDecisionByPlan.find(work.planId);
        if (decision != mUnifiedDecisionByPlan.end())
        {
            dispatch.decisionId = decision->second.decisionId;
            dispatch.snapshotId = decision->second.snapshotId;
            dispatch.actionKind = decision->second.actionKind;
            dispatch.incrementalActionId = decision->second.incrementalActionId;
            dispatch.requestedStartSkewPercent = decision->second.requestedStartSkewPercent;
            dispatch.requestedDirection = decision->second.requestedDirection;
            dispatch.dispatchMode = decision->second.dispatchMode;
            dispatch.direction = decision->second.requestedDirection;
            dispatch.incumbentPhase = decision->second.incumbentPhase;
            dispatch.incumbentExecutionId = decision->second.incumbentExecutionId;
            dispatch.incumbentDispatchAgeUs = decision->second.incumbentDispatchAgeUs;
            dispatch.newcomerPhase = decision->second.newcomerPhase;
            auto const phaseWork = [&](PhaseUnifiedPhase phase) {
                return std::find_if(current.work.begin(), current.work.end(),
                    [phase](PhaseInFlightWorkSnapshot const& candidate) { return candidate.phase == phase; });
            };
            auto const incumbentWork = phaseWork(dispatch.incumbentPhase);
            if (incumbentWork != current.work.end())
            {
                dispatch.incumbentExecutionId = incumbentWork->executionId;
                dispatch.incumbentDispatchAgeUs = current.hostSnapshotNs >= incumbentWork->dispatchHostNs
                    ? static_cast<double>(current.hostSnapshotNs - incumbentWork->dispatchHostNs) / 1000.0
                    : 0.0;
            }
            auto const newcomerWork = phaseWork(dispatch.newcomerPhase);
            if (newcomerWork != current.work.end())
            {
                dispatch.newcomerExecutionId = newcomerWork->executionId;
            }
            dispatch.plannedOutstanding = decision->second.plannedOutstanding;
            dispatch.actionFidelity = phaseExecutionSetIsSubset(current.outstanding, dispatch.plannedOutstanding);
            bool const actionIdentityMatches = phaseUnifiedActionIdentityMatches(
                decision->second.dispatchMode, decision->second.actionId, work.actionId);
            if (!actionIdentityMatches)
            {
                dispatch.actionFidelity = false;
                dispatch.actionFidelityReason = PhaseUnifiedFidelityReason::kActionIdMismatch;
            }
            else if (!dispatch.actionFidelity)
            {
                dispatch.actionFidelityReason = PhaseUnifiedFidelityReason::kOutstandingMismatch;
            }
            else
            {
                allowOutstanding(work.executionId, dispatch.plannedOutstanding);
            }
        }
        else
        {
            dispatch.plannedOutstanding = current.outstanding;
            dispatch.actionFidelity = false;
            dispatch.actionFidelityReason = PhaseUnifiedFidelityReason::kMissingDecision;
        }
        emitUnifiedEvent(std::move(dispatch));
        observedBefore = observedBefore | phaseExecutionSetForUnifiedPhase(work.phase);
        observedWork.push_back(work);
    }

    std::vector<uint64_t> completedPlans;
    for (auto const& [planId, decision] : mUnifiedDecisionByPlan)
    {
        static_cast<void>(decision);
        bool const active = std::any_of(current.work.begin(), current.work.end(),
            [planId](PhaseInFlightWorkSnapshot const& work) { return work.planId == planId; });
        if (!active)
        {
            completedPlans.push_back(planId);
        }
    }
    for (uint64_t const planId : completedPlans)
    {
        mUnifiedDecisionByPlan.erase(planId);
    }
    mPreviousUnifiedInFlight = current;
}

PhaseExecutionSet PhaseThreeCoordinator::observedGlobalExecution() const noexcept
{
    PhaseExecutionSet result{PhaseExecutionSet::kNone};
    // Host/preprocess preparation only materializes a future E action. It is
    // not an outstanding TensorRT encoder context and must not consume the E
    // bit in an execution lease before submitPreparedEncoder().
    if (mEncoderGpuSubmitted || mVision.busy())
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
    if (!phaseExecutionSetContains(observed, PhaseExecutionSet::kPrefill)
        && !phaseExecutionSetContains(observed, PhaseExecutionSet::kDecode))
    {
        mActiveGlobalPdExecution.reset();
    }
    if (!mGlobalExecutionLease->permits(observed))
    {
        ++mGlobalActionFidelityViolations;
        ELLM_CHECK(false, "Observed phases exceed the active global execution lease");
    }
    if (observed == PhaseExecutionSet::kNone)
    {
        mGlobalFormationRealizedTracker.observeCompletion(mGlobalExecutionLease->planId, steadyTimestampUs());
        emitCompletedFormationEpisodes();
        mGlobalExecutionLease.reset();
    }
}

PhaseGlobalDispatchPlan PhaseThreeCoordinator::beginGlobalExecutionLease(PhaseGlobalActionCandidate const& candidate,
    std::vector<PhaseGlobalActionCandidate> const* candidateFrontier, PhaseGlobalSelectionAudit const* selectorAudit)
{
    ELLM_CHECK(!mGlobalExecutionLease.has_value(), "A global execution lease is already active");
    PhaseExecutionSet const outstandingBefore = observedGlobalExecution();
    PhaseUnifiedActionDirection const direction = phaseInitialDirection(candidate.key.kind);
    PhaseStartSkewBucket const startSkew = phaseRequestedStartSkew();
    PhaseGlobalDispatchPlan plan = phaseGlobalDispatchPlan(kTHREE_PHASE_PLAN_NAMESPACE | ++mGlobalPlanSequence,
        kTHREE_PHASE_PLAN_NAMESPACE | ++mGlobalSnapshotEpoch, candidate, outstandingBefore, direction, startSkew);
    ELLM_CHECK(plan.allowedOutstanding != PhaseExecutionSet::kNone, "A dispatch lease requires executable phases");
    if (!plan.incrementalAction.legal())
    {
        LOG_ERROR(
            "Illegal global lease: action=%d outstanding_before=%d planned=%d direction=%d reason=%s prepared_e=%s",
            static_cast<int32_t>(candidate.key.kind), static_cast<int32_t>(outstandingBefore),
            static_cast<int32_t>(plan.allowedOutstanding), static_cast<int32_t>(direction),
            phaseIncrementalLegalityReasonName(plan.incrementalAction.legality),
            mPreparedEncoder != nullptr ? "yes" : "no");
    }
    ELLM_CHECK(plan.incrementalAction.legal(), "A dispatch lease violates incremental action legality");
    mGlobalExecutionLease = plan;
    recordUnifiedDecision(candidate, plan, candidateFrontier, selectorAudit);
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
    double const timestampUs = steadyTimestampUs();
    PhaseGlobalDispatchPlan const& plan = *mGlobalExecutionLease;
    PhaseFormationWork const work = realizedDispatchWork(plan);
    mGlobalFormationRealizedTracker.observeDispatch(plan.planId, plan.action, work, timestampUs);
    if (mPendingGlobalFormationRealizedEpisode.has_value())
    {
        mPendingGlobalFormationRealizedEpisode->timestampUs = timestampUs;
        mGlobalFormationRealizedTracker.startEpisode(
            *mPendingGlobalFormationRealizedEpisode, plan.planId, plan.action, work);
        mPendingGlobalFormationRealizedEpisode.reset();
    }
    emitCompletedFormationEpisodes();
}

void PhaseThreeCoordinator::abandonGlobalExecutionLease() noexcept
{
    if (mGlobalExecutionLease.has_value())
    {
        mUnifiedDecisionByPlan.erase(mGlobalExecutionLease->planId);
    }
    mGlobalExecutionLease.reset();
    mPendingGlobalFormationRealizedEpisode.reset();
}

void PhaseThreeCoordinator::emitCompletedFormationEpisodes()
{
    std::vector<PhaseFormationRealizedEpisode> completed = mGlobalFormationRealizedTracker.takeCompletedEpisodes();
    if (!mFormationEpisodeCallback)
    {
        return;
    }
    for (PhaseFormationRealizedEpisode const& episode : completed)
    {
        mFormationEpisodeCallback(episode);
    }
}

bool PhaseThreeCoordinator::dispatchGlobalPrefillDecodeResidual(
    IndependentPhaseServerArbitrationSnapshot const& serverState)
{
    if (mConfig.globalSchedulerMode != PhaseGlobalSchedulerMode::kActive || !mGlobalExecutionLease.has_value()
        || !mActiveGlobalPdExecution.has_value() || !serverState.busy || !mPending.empty() || !mEncoding.empty()
        || mVision.busy() || mEncoderPreparation.valid())
    {
        return false;
    }
    PhaseGlobalActionKind const activeKind = mActiveGlobalPdExecution->candidate.key.kind;
    bool const addDecode = activeKind == PhaseGlobalActionKind::kPrefill && serverState.decodeQueued > 0U;
    bool const addPrefill = activeKind == PhaseGlobalActionKind::kDecode && serverState.prefillQueued > 0U;
    if (!addDecode && !addPrefill)
    {
        return false;
    }
    std::optional<PhaseGlobalActionCandidate> missing
        = addDecode ? mServer.previewGlobalDecodeAction() : mServer.previewGlobalPrefillAction();
    if (!missing.has_value())
    {
        return false;
    }
    if (missing->candidateId == mActiveGlobalPdExecution->lastResidualPdCandidateId)
    {
        return false;
    }
    mActiveGlobalPdExecution->lastResidualPdCandidateId = missing->candidateId;
    ++mGlobalResidualPrefillDecodeOpportunities;

    double const elapsedUs = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - mActiveGlobalPdExecution->startedAt)
                                 .count();
    double const incumbentReferenceUs = std::max(mActiveGlobalPdExecution->candidate.predictedMakespanUs,
        mActiveGlobalPdExecution->candidate.predictedBlockingUs);
    PhaseGlobalActionCandidate const active
        = phaseGlobalResidualCandidate(mActiveGlobalPdExecution->candidate, elapsedUs);
    PhaseGlobalActionCandidate const& prefill = addPrefill ? *missing : active;
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
    if (addDecode)
    {
        ++mGlobalResidualPrefillAnchorOpportunities;
    }
    else
    {
        ++mGlobalResidualDecodeAnchorOpportunities;
    }
    overlap.primaryRequestIds = prefill.primaryRequestIds;
    overlap.primaryStableSlotIds = prefill.primaryStableSlotIds;
    overlap.secondaryRequestIds = decode.primaryRequestIds;
    overlap.secondaryStableSlotIds = decode.primaryStableSlotIds;
    phaseGlobalFinalizeCandidate(overlap);
    // Residual overlap profitability must compare the remaining P and D
    // actions against executing those exact actions serially. Phase-local
    // reference work may be singleton-scaled to describe batching efficiency;
    // adding it here would make large residual cohorts look artificially
    // profitable and contaminate their online overlap observations.
    overlap.referenceWorkUs = active.predictedMakespanUs + missing->predictedMakespanUs;
    overlap.requestServiceLagUs = std::max(active.requestServiceLagUs, missing->requestServiceLagUs);
    overlap.memory = prefill.memory;
    overlap.memory.allocateBytes = saturatedAdd(overlap.memory.allocateBytes, decode.memory.allocateBytes);
    overlap.memory.guaranteedGrowthBytes
        = saturatedAdd(overlap.memory.guaranteedGrowthBytes, decode.memory.guaranteedGrowthBytes);
    overlap.memory.nearReclaimBytes = saturatedAdd(overlap.memory.nearReclaimBytes, decode.memory.nearReclaimBytes);

    double const activeRobustUs = active.predictedMakespanUs + active.uncertaintyUs;
    double const missingRobustUs = missing->predictedMakespanUs + missing->uncertaintyUs;
    double const robustSerialUs = activeRobustUs + missingRobustUs;
    std::optional<PhaseGlobalCostEstimate> estimate = mRuntimeCostTracker->estimate(overlap.key);
    PhaseGlobalOverlapCostDiagnostic diagnostic = mRuntimeCostTracker->overlapDiagnostic(overlap.key);
    bool directMeasured = diagnostic.status == PhaseGlobalOverlapCostStatus::kEligible
        || diagnostic.status == PhaseGlobalOverlapCostStatus::kUnprofitable;
    if (!directMeasured)
    {
        estimate.reset();
    }
    bool overlapProfitable = diagnostic.status == PhaseGlobalOverlapCostStatus::kEligible;
    if (!estimate.has_value() && mRuntimeCostTracker->contextualPdConfig().mode == PhaseContextualPdMode::kDisabled)
    {
        // A complete P+D observation is a conservative upper bound for adding
        // the same idle context after its peer has already made progress.
        PhaseGlobalActionKey completeOverlapKey = overlap.key;
        completeOverlapKey.residualAugmentation = false;
        completeOverlapKey.residualAnchor = PhaseGlobalResidualAnchor::kNone;
        PhaseGlobalOverlapCostDiagnostic const completeDiagnostic
            = mRuntimeCostTracker->overlapDiagnostic(completeOverlapKey);
        bool const completeMeasured = completeDiagnostic.status == PhaseGlobalOverlapCostStatus::kEligible
            || completeDiagnostic.status == PhaseGlobalOverlapCostStatus::kUnprofitable;
        std::optional<PhaseGlobalCostEstimate> const completeEstimate
            = completeMeasured ? mRuntimeCostTracker->estimate(completeOverlapKey) : std::nullopt;
        if (completeEstimate.has_value())
        {
            diagnostic = completeDiagnostic;
            estimate = completeEstimate;
            overlapProfitable = completeDiagnostic.status == PhaseGlobalOverlapCostStatus::kEligible;
        }
    }
    if (!estimate.has_value() && mRuntimeCostTracker->contextualPdConfig().mode == PhaseContextualPdMode::kDisabled)
    {
        estimate = mRuntimeCostTracker->trustedEstimateCoveringOverlap(overlap.key);
        if (estimate.has_value())
        {
            double const robustMakespanUs
                = static_cast<double>(estimate->makespanMedianMs + estimate->uncertaintyMs) * 1000.0;
            overlapProfitable = overlap.referenceWorkUs >= robustMakespanUs
                    * (1.0 + static_cast<double>(mConfig.globalCostModelConfig.minimumOverlapGainRatio));
            ++mGlobalResidualCoveringCostHits;
        }
    }
    if (!estimate.has_value())
    {
        PhaseGlobalActionKey completeOverlapKey = overlap.key;
        completeOverlapKey.residualAugmentation = false;
        completeOverlapKey.residualAnchor = PhaseGlobalResidualAnchor::kNone;
        estimate = mRuntimeCostTracker->trustedEstimateCoveringOverlap(completeOverlapKey);
        if (estimate.has_value())
        {
            double const robustMakespanUs
                = static_cast<double>(estimate->makespanMedianMs + estimate->uncertaintyMs) * 1000.0;
            overlapProfitable = overlap.referenceWorkUs >= robustMakespanUs
                    * (1.0 + static_cast<double>(mConfig.globalCostModelConfig.minimumOverlapGainRatio));
            ++mGlobalResidualCoveringCostHits;
        }
    }
    bool const overlapMeasured = estimate.has_value();
    bool const needsCalibration = !overlapMeasured;
    if (overlapMeasured && !overlapProfitable)
    {
        ++mGlobalResidualMeasuredUnprofitableOpportunities;
    }
    bool const probeIntervalReady = mLastGlobalSafeProbeSequence == 0U
        || mGlobalDecisionSequence - mLastGlobalSafeProbeSequence >= mConfig.globalSafeProbeInterval;
    double protectedSlackUs = std::numeric_limits<double>::infinity();
    for (PhaseProtectedCompletion const& completion : active.protectedCompletions)
    {
        protectedSlackUs = std::min(protectedSlackUs, completion.slackUs);
    }
    for (PhaseProtectedCompletion const& completion : missing->protectedCompletions)
    {
        protectedSlackUs = std::min(protectedSlackUs, completion.slackUs);
    }
    bool const safeProbe = !overlapMeasured && needsCalibration && mConfig.globalSafeProbeSlackMultiplier > 0.0F
        && probeIntervalReady
        && (protectedSlackUs >= static_cast<double>(mConfig.globalSafeProbeSlackMultiplier) * robustSerialUs
            || protectedSlackUs < robustSerialUs);
    overlap.overlapCostKnown = overlapMeasured;
    overlap.overlapCostProfitable = overlapProfitable;
    overlap.safeProbeEligible = safeProbe;
    PhaseContextualPdMode const contextualMode = mRuntimeCostTracker->contextualPdConfig().mode;
    bool const externalPrefillLineage
        = prefill.key.primaryWorkClass == static_cast<int32_t>(PhasePrefillClass::kExternal);
    bool const contextualEligible = mConfig.contextualResidualEligible(externalPrefillLineage);
    bool const producerCriticalPath = phaseContextualPdProducerCriticalPath(
        !mPending.empty() || !mEncoding.empty() || mVision.busy() || mEncoderPreparation.valid(),
        mConfig.preserveLegacyPairEligibility && externalPrefillLineage);
    if (contextualMode != PhaseContextualPdMode::kDisabled && contextualEligible)
    {
        PhaseContextualPdInput contextualInput{prefill.predictedMakespanUs, decode.predictedMakespanUs,
            protectedSlackUs, prefill.key.primaryBatchSize, decode.key.primaryBatchSize, prefill.key.chunkLength,
            prefill.key.primaryContextBucket, decode.key.primaryContextBucket, overlap.key.executionVariant, true,
            overlap.key.residualAnchor, elapsedUs, incumbentReferenceUs,
            incumbentReferenceUs > 0.0 ? elapsedUs / incumbentReferenceUs : -1.0,
            addDecode ? PhaseExecutionSet::kPrefill : PhaseExecutionSet::kDecode};
        contextualInput.prefillBatchCapacity = mConfig.contextualPrefillBatchCapacity;
        contextualInput.decodeBatchCapacity = mConfig.contextualDecodeBatchCapacity;
        overlap.contextualPdFeatures = phaseContextualPdFeatures(contextualInput);
        overlap.contextualPdFeatureValid = true;
        PhaseContextualPairDirection const contextualDirection
            = phaseContextualPairDirection(overlap.key.kind, overlap.key.residualAnchor);
        PhaseContextualPdEstimate const contextual
            = mRuntimeCostTracker->predictContextualDirection(contextualDirection, overlap.contextualPdFeatures);
        overlap.contextualPdMean = contextual.mean;
        overlap.contextualPdUncertainty = contextual.uncertainty;
        overlap.contextualPdLowerConfidenceBound = contextual.lowerConfidenceBound;
        overlap.contextualPdExploration = !contextual.ready && probeIntervalReady
            && (protectedSlackUs >= static_cast<double>(mConfig.globalSafeProbeSlackMultiplier) * robustSerialUs
                || protectedSlackUs < robustSerialUs);
        if (contextualMode == PhaseContextualPdMode::kActive && contextual.ready
            && phaseContextualPdControlsDecision(producerCriticalPath))
        {
            overlap.decisionCostKnown = true;
            overlap.decisionMakespanUs
                = phaseContextualDecisionMakespanUs(overlap.referenceWorkUs, contextual.lowerConfidenceBound);
        }
    }
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
    double overlapDecodeCompletionUs = overlap.predictedMakespanUs;
    double overlapDecodeUncertaintyUs = overlap.uncertaintyUs;
    int32_t const decodeRows = decode.key.primaryBatchSize;
    int32_t const decodeContextLength
        = decode.key.primaryContextBucket * std::max(1, mConfig.globalDecodeContextBucketTokens);
    if (std::optional<float> const decodeP95
        = mServer.estimateGlobalDecodeComponentP95(decodeRows, decodeContextLength, true))
    {
        overlapDecodeCompletionUs = static_cast<double>(*decodeP95) * 1000.0;
        overlapDecodeUncertaintyUs = 0.0;
    }
    for (PhaseProtectedCompletion completion : active.protectedCompletions)
    {
        completion.predictedCompletionUs
            = completion.kind == PhaseProtectedKind::kDecode ? overlapDecodeCompletionUs : overlap.predictedMakespanUs;
        completion.uncertaintyUs
            = completion.kind == PhaseProtectedKind::kDecode ? overlapDecodeUncertaintyUs : overlap.uncertaintyUs;
        overlap.protectedCompletions.push_back(completion);
    }
    for (PhaseProtectedCompletion completion : missing->protectedCompletions)
    {
        completion.predictedCompletionUs
            = completion.kind == PhaseProtectedKind::kDecode ? overlapDecodeCompletionUs : overlap.predictedMakespanUs;
        completion.uncertaintyUs
            = completion.kind == PhaseProtectedKind::kDecode ? overlapDecodeUncertaintyUs : overlap.uncertaintyUs;
        overlap.protectedCompletions.push_back(completion);
    }
    ++mGlobalDecisionSequence;
    ++mGlobalDecisions;
    std::vector<PhaseGlobalActionCandidate> const candidates{active, overlap};
    PhaseGlobalDecision decision = mGlobalScheduler.select(candidates);
    if (contextualMode != PhaseContextualPdMode::kDisabled)
    {
        bool const selectedOverlap = decision.selectedIndex.has_value() && *decision.selectedIndex == 1U;
        bool const contextualControlsDecision = phaseContextualPdControlsDecision(producerCriticalPath);
        if (contextualControlsDecision)
        {
            PhaseContextualPairDirection const direction
                = phaseContextualPairDirection(overlap.key.kind, overlap.key.residualAnchor);
            mRuntimeCostTracker->recordContextualDirectionSelection(
                direction, selectedOverlap, overlap.contextualPdExploration);
        }
    }
    bool const effectiveSafeProbe = safeProbe;
    if (!decision.selectedIndex.has_value() || *decision.selectedIndex != 1U)
    {
        mGlobalResidualPrefillDecodeUnknownCostRejects += !overlapMeasured && !effectiveSafeProbe ? 1U : 0U;
        return false;
    }
    if (effectiveSafeProbe)
    {
        mLastGlobalSafeProbeSequence = mGlobalDecisionSequence;
        ++mGlobalSafeProbes;
    }
    uint64_t const planId = kTHREE_PHASE_PLAN_NAMESPACE | ++mGlobalPlanSequence;
    uint64_t const snapshotEpoch = kTHREE_PHASE_PLAN_NAMESPACE | ++mGlobalSnapshotEpoch;
    double const referenceUs = std::max(mActiveGlobalPdExecution->candidate.predictedMakespanUs,
        mActiveGlobalPdExecution->candidate.predictedBlockingUs);
    PhaseStartSkewBucket const startSkew = phaseRequestedStartSkew(elapsedUs, referenceUs);
    std::optional<PhaseGlobalDispatchPlan> const augmented
        = phaseGlobalAugmentedDispatchPlan(planId, snapshotEpoch, *mGlobalExecutionLease, overlap, startSkew);
    ELLM_CHECK(augmented.has_value(), "Residual P/D augmentation does not match the active phase rows");
    recordUnifiedDecision(overlap, *augmented, &candidates);
    bool const started = mServer.augmentGlobalAction(std::move(*missing), overlap, planId, snapshotEpoch);
    if (!started)
    {
        mUnifiedDecisionByPlan.erase(planId);
        return false;
    }
    mGlobalExecutionLease = *augmented;
    mActiveGlobalPdExecution.reset();
    mLastGlobalAction = PhaseGlobalActionKind::kPrefillDecode;
    ++mGlobalResidualPrefillDecodeSelections;
    if (addDecode)
    {
        ++mGlobalResidualPrefillAnchorSelections;
    }
    else
    {
        ++mGlobalResidualDecodeAnchorSelections;
    }
    if (overlapMeasured && !overlapProfitable)
    {
        ++mGlobalResidualMeasuredUnprofitableSelections;
    }
    validateGlobalExecutionLaunch();
    return true;
}

bool PhaseThreeCoordinator::dispatchGlobalAction()
{
    auto const decisionStart = std::chrono::steady_clock::now();
    if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kDisabled)
    {
        return false;
    }
    refreshGlobalExecutionLease();
    IndependentPhaseServerArbitrationSnapshot const serverState = mServer.arbitrationSnapshot();
    if (dispatchGlobalPrefillDecodeResidual(serverState))
    {
        return true;
    }
    bool const preparedEncoderReady = mPreparedEncoder != nullptr && !mEncoding.empty();
    bool const residualAugmentation = mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive
        && mGlobalExecutionLease.has_value() && mActiveGlobalPdExecution.has_value() && serverState.busy
        && (preparedEncoderReady || (!mPending.empty() && mEncoding.empty())) && !mVision.busy()
        && !mEncoderPreparation.valid()
        && (mActiveGlobalPdExecution->candidate.key.kind == PhaseGlobalActionKind::kPrefill
            || mActiveGlobalPdExecution->candidate.key.kind == PhaseGlobalActionKind::kDecode);
    if (mGlobalExecutionLease.has_value() && !residualAugmentation)
    {
        return false;
    }
    if (serverState.busy && !residualAugmentation)
    {
        return false;
    }
    if (!residualAugmentation && mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive
        && mServer.shouldWaitForGlobalDecodeRefill())
    {
        return false;
    }

    if ((!mEncoding.empty() && !preparedEncoderReady) || mVision.busy() || mEncoderPreparation.valid())
    {
        return false;
    }

    double residualElapsedUs{};
    std::optional<PhaseGlobalActionCandidate> pd;
    PhaseGlobalSelectionAudit pdAudit;
    std::vector<PhaseGlobalActionCandidate> pdCandidateFrontier;
    std::optional<PhaseGlobalActionCandidate> prefillForEncoder;
    std::optional<PhaseGlobalActionCandidate> decodeForEncoder;
    if (residualAugmentation)
    {
        residualElapsedUs = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - mActiveGlobalPdExecution->startedAt)
                                .count();
        pd = phaseGlobalResidualCandidate(mActiveGlobalPdExecution->candidate, residualElapsedUs);
        if (pd->key.kind == PhaseGlobalActionKind::kPrefill)
        {
            prefillForEncoder = *pd;
        }
        else
        {
            decodeForEncoder = *pd;
        }
    }
    else if (serverState.prefillQueued > 0U || serverState.decodeQueued > 0U)
    {
        pd = mServer.previewGlobalAction(mUnifiedEventCallback ? &pdAudit : nullptr);
        pdCandidateFrontier = mServer.lastGlobalPreviewCandidates();
    }
    auto const frontierCandidate = [&](PhaseGlobalActionKind kind) -> std::optional<PhaseGlobalActionCandidate> {
        auto const candidate = std::find_if(pdCandidateFrontier.begin(), pdCandidateFrontier.end(),
            [kind](PhaseGlobalActionCandidate const& value) { return value.key.kind == kind; });
        return candidate != pdCandidateFrontier.end() ? std::optional<PhaseGlobalActionCandidate>(*candidate)
                                                      : std::nullopt;
    };
    if (!residualAugmentation && serverState.prefillQueued > 0U && mConfig.enableGlobalEncoderPrefillAction)
    {
        prefillForEncoder = frontierCandidate(PhaseGlobalActionKind::kPrefill);
    }
    if (!residualAugmentation && serverState.decodeQueued > 0U)
    {
        decodeForEncoder = frontierCandidate(PhaseGlobalActionKind::kDecode);
    }
    std::vector<size_t> encoderBatchIndices;
    if (preparedEncoderReady)
    {
        encoderBatchIndices.resize(mEncoding.size());
        std::iota(encoderBatchIndices.begin(), encoderBatchIndices.end(), size_t{0});
    }
    else if (!mConfig.enableAsyncEncoderPreparation && !mPending.empty())
    {
        encoderBatchIndices = nextEncoderBatchIndices();
    }
    auto const& encoderRequest = [&](size_t index) -> PendingVisionRequest const& {
        return preparedEncoderReady ? mEncoding[index] : mPending[index];
    };
    if (encoderBatchIndices.empty() && !pd.has_value())
    {
        return false;
    }
    if (mGlobalWarmupProbeMode)
    {
        ++mGlobalWarmupDecisions;
        mGlobalWarmupPrefillCandidates += prefillForEncoder.has_value() ? 1U : 0U;
        mGlobalWarmupDecodeCandidates += decodeForEncoder.has_value() ? 1U : 0U;
        if (!prefillForEncoder.has_value() && !decodeForEncoder.has_value() && mServer.hasPendingSampling())
        {
            // The completion event is a concrete WAIT target. Let sampling
            // materialize its D row before choosing an otherwise serial E.
            return false;
        }
    }
    if (encoderBatchIndices.empty())
    {
        if (residualAugmentation)
        {
            return false;
        }
        if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive)
        {
            ++mGlobalPdSelections;
            pd->hostDecisionUs
                = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - decisionStart).count();
            std::vector<PhaseGlobalActionCandidate> const* candidateFrontier
                = pdCandidateFrontier.empty() ? nullptr : &pdCandidateFrontier;
            recordGlobalDecisionCost(decisionStart);
            PhaseGlobalDispatchPlan const executionPlan
                = beginGlobalExecutionLease(*pd, candidateFrontier, pdAudit.inputs.empty() ? nullptr : &pdAudit);
            PhaseGlobalActionCandidate launched = *pd;
            auto const launchedAt = std::chrono::steady_clock::now();
            bool const started
                = mServer.dispatchGlobalAction(std::move(*pd), executionPlan.planId, executionPlan.snapshotEpoch);
            if (!started)
            {
                abandonGlobalExecutionLease();
                return false;
            }
            if (launched.key.kind == PhaseGlobalActionKind::kPrefill
                || launched.key.kind == PhaseGlobalActionKind::kDecode)
            {
                mActiveGlobalPdExecution = ActiveGlobalPdExecution{std::move(launched), launchedAt, {}};
            }
            validateGlobalExecutionLaunch();
            return true;
        }
        return false;
    }

    if (residualAugmentation)
    {
        std::vector<uint64_t> encoderRequestIds;
        encoderRequestIds.reserve(encoderBatchIndices.size());
        for (size_t const index : encoderBatchIndices)
        {
            encoderRequestIds.push_back(encoderRequest(index).requestId);
        }
        if (encoderRequestIds == mActiveGlobalPdExecution->lastResidualEncoderRequestIds)
        {
            return false;
        }
        mActiveGlobalPdExecution->lastResidualEncoderRequestIds = std::move(encoderRequestIds);
        ++mGlobalResidualAugmentationOpportunities;
    }

    size_t encoderInputTokens{};
    size_t encoderPayloadBytes{};
    double encoderSlackUs{std::numeric_limits<double>::infinity()};
    double encoderServiceLagUs{};
    uint64_t encoderProtectedRequestId{};
    auto const now = std::chrono::steady_clock::now();
    for (size_t const index : encoderBatchIndices)
    {
        PendingVisionRequest const& request = encoderRequest(index);
        encoderInputTokens += request.inputTokens;
        encoderPayloadBytes
            += request.estimatedPayloadBytes > 0U ? request.estimatedPayloadBytes : mEstimatedEncodedBytes;
        double const ageUs = std::chrono::duration<double, std::micro>(now - request.scheduling.submittedAt).count();
        double const targetUs
            = request.scheduling.ttftTargetUs > 0.0 ? request.scheduling.ttftTargetUs : mConfig.visionTtftTargetUs;
        double const slackUs = targetUs > 0.0 ? targetUs - ageUs : std::numeric_limits<double>::infinity();
        if (slackUs < encoderSlackUs
            || (slackUs == encoderSlackUs
                && (encoderProtectedRequestId == 0U || request.requestId < encoderProtectedRequestId)))
        {
            encoderSlackUs = slackUs;
            encoderProtectedRequestId = request.requestId;
        }
        encoderServiceLagUs = std::max(encoderServiceLagUs, ageUs);
    }
    struct EncoderCostPrediction
    {
        PhaseGlobalActionKey key;
        double makespanUs{};
        double uncertaintyUs{};
        double referenceUs{};
        PhaseServiceReferenceSource costSource{PhaseServiceReferenceSource::kColdFallback};
        PhaseServiceReferenceSource referenceSource{PhaseServiceReferenceSource::kColdFallback};
    };
    constexpr size_t kEncoderTokenBucket = 1024U;
    auto const predictEncoderCost = [&](size_t batchSize, size_t inputTokens) {
        int32_t const contextBucket
            = static_cast<int32_t>((inputTokens + kEncoderTokenBucket - 1U) / kEncoderTokenBucket);
        EncoderCostPrediction prediction{
            {PhaseGlobalActionKind::kEncoder, static_cast<int32_t>(batchSize), 0, 0, contextBucket, 0},
            mLastEncoderGpuMs > 0.0F ? static_cast<double>(mLastEncoderGpuMs) * 1000.0
                                     : mConfig.encoderDispatchInitialCostUs,
            static_cast<double>(mConfig.globalCostModelConfig.coldStartUncertaintyMs) * 1000.0, 0.0};
        prediction.referenceUs = prediction.makespanUs;
        if (std::optional<PhaseGlobalCostEstimate> const online = mRuntimeCostTracker->estimate(prediction.key))
        {
            prediction.makespanUs = static_cast<double>(online->makespanMedianMs) * 1000.0;
            prediction.uncertaintyUs = static_cast<double>(online->uncertaintyMs) * 1000.0;
            prediction.referenceUs = static_cast<double>(online->referenceWorkMedianMs) * 1000.0;
            prediction.costSource = PhaseServiceReferenceSource::kRuntimeExact;
            prediction.referenceSource = PhaseServiceReferenceSource::kRuntimeExact;
            return prediction;
        }

        PhaseVisionEncoderBatchCost const* selected{};
        PhaseVisionEncoderBatchCost const* singleton{};
        for (PhaseVisionEncoderBatchCost const& cost : mConfig.encoderBatchCosts)
        {
            if (cost.maxInputTokens < inputTokens)
            {
                continue;
            }
            if (cost.batchSize >= batchSize
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
            prediction.makespanUs = static_cast<double>(selected->p95GpuMs) * 1000.0;
            prediction.uncertaintyUs = 0.0;
            prediction.referenceUs = singleton != nullptr
                ? static_cast<double>(singleton->p95GpuMs) * 1000.0 * batchSize
                : prediction.makespanUs;
            prediction.costSource = PhaseServiceReferenceSource::kStaticProfile;
            prediction.referenceSource = singleton != nullptr ? PhaseServiceReferenceSource::kStaticProfile
                                                              : PhaseServiceReferenceSource::kDerivedIsolated;
        }
        return prediction;
    };
    EncoderCostPrediction const encoderPrediction = predictEncoderCost(encoderBatchIndices.size(), encoderInputTokens);
    PhaseGlobalActionKey const encoderKey = encoderPrediction.key;
    double const encoderMakespanUs = encoderPrediction.makespanUs;
    double const encoderUncertaintyUs = encoderPrediction.uncertaintyUs;
    double const encoderReferenceUs = encoderPrediction.referenceUs;
    int32_t const encoderContextBucket = encoderKey.primaryContextBucket;

    size_t const estimatedPromptTokensPerRequest = std::max<size_t>(1U,
        mEstimatedPromptTokens > 0U
            ? mEstimatedPromptTokens
            : (encoderInputTokens + encoderBatchIndices.size() - 1U) / encoderBatchIndices.size());
    PhaseGlobalCostEstimate const visionPrefill
        = mServer.estimateGlobalPrefillDrainCost(static_cast<int32_t>(encoderBatchIndices.size()),
            static_cast<int32_t>(std::min<size_t>(
                estimatedPromptTokensPerRequest, static_cast<size_t>(std::numeric_limits<int32_t>::max()))),
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
        encoder.primaryRequestIds.push_back(encoderRequest(index).requestId);
    }
    phaseGlobalFinalizeCandidate(encoder);
    encoder.predictedBlockingUs = encoderMakespanUs;
    encoder.predictedMakespanUs = encoderMakespanUs;
    encoder.uncertaintyUs = encoderUncertaintyUs;
    encoder.referenceWorkUs = encoderReferenceUs;
    encoder.predictedCostSource = encoderPrediction.costSource;
    encoder.referenceCostSource = encoderPrediction.referenceSource;
    encoder.requestServiceLagUs = encoderServiceLagUs;
    encoder.protectedCompletions.push_back({encoderSlackUs, encoderMakespanUs + visionPrefillMakespanUs,
        encoderUncertaintyUs + visionPrefillUncertaintyUs, PhaseProtectedKind::kEncoder, encoderProtectedRequestId,
        std::max(1.0,
            encoderReferenceUs / static_cast<double>(std::max<size_t>(1U, encoderBatchIndices.size()))
                + visionPrefillMakespanUs),
        PhaseServiceReferenceSource::kDerivedIsolated, encoderServiceLagUs});
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
                pd->uncertaintyUs + encoderUncertaintyUs + visionPrefillUncertaintyUs, PhaseProtectedKind::kEncoder,
                encoderProtectedRequestId,
                std::max(1.0,
                    encoderReferenceUs / static_cast<double>(std::max<size_t>(1U, encoderBatchIndices.size()))
                        + visionPrefillMakespanUs),
                PhaseServiceReferenceSource::kDerivedIsolated, encoderServiceLagUs});
    }

    std::vector<PhaseGlobalActionCandidate> candidates;
    if (!residualAugmentation)
    {
        candidates.push_back(encoder);
    }
    if (pd.has_value())
    {
        candidates.push_back(*pd);
    }
    if (mConfig.enableGlobalPdFrontier && !residualAugmentation && !mGlobalWarmupProbeMode)
    {
        for (PhaseGlobalActionKind const kind : {PhaseGlobalActionKind::kPrefill, PhaseGlobalActionKind::kDecode})
        {
            auto alternative = frontierCandidate(kind);
            if (!alternative.has_value() || (pd.has_value() && alternative->candidateId == pd->candidateId))
            {
                continue;
            }
            alternative->protectedCompletions.push_back(
                {encoderSlackUs, alternative->predictedMakespanUs + encoderMakespanUs + visionPrefillMakespanUs,
                    alternative->uncertaintyUs + encoderUncertaintyUs + visionPrefillUncertaintyUs,
                    PhaseProtectedKind::kEncoder, encoderProtectedRequestId,
                    std::max(1.0,
                        encoderReferenceUs / static_cast<double>(std::max<size_t>(1U, encoderBatchIndices.size()))
                            + visionPrefillMakespanUs),
                    PhaseServiceReferenceSource::kDerivedIsolated, encoderServiceLagUs});
            candidates.push_back(std::move(*alternative));
        }
    }
    bool const encoderPrefillExclusive = mConfig.serializeAllEncoderPrefill
        || (mConfig.exclusiveEncoderInputTokenThreshold > 0
            && encoderInputTokens > mConfig.exclusiveEncoderInputTokenThreshold);
    auto addEncoderOverlap = [&](PhaseGlobalActionKind kind, PhaseGlobalActionCandidate const& phase) {
        ++mGlobalEncoderOverlapOpportunities;
        int32_t const chunkLength = kind == PhaseGlobalActionKind::kEncoderPrefill ? phase.key.chunkLength : 0;
        PhaseGlobalActionKey overlapKey{kind, static_cast<int32_t>(encoderBatchIndices.size()),
            phase.key.primaryBatchSize, chunkLength, encoderContextBucket, phase.key.primaryContextBucket};
        overlapKey.primaryWorkClass = phase.key.primaryWorkClass;
        overlapKey.residualAugmentation = residualAugmentation;
        if (residualAugmentation)
        {
            overlapKey.residualAnchor = phase.key.kind == PhaseGlobalActionKind::kPrefill
                ? PhaseGlobalResidualAnchor::kPrefill
                : PhaseGlobalResidualAnchor::kDecode;
        }
        overlapKey.executionVariant
            = phaseExecutionVariant(false, phaseExecutionVariantUsesPrimaryGraph(phase.key.executionVariant));
        double overlapMakespanUs = encoderMakespanUs + phase.predictedMakespanUs;
        double overlapUncertaintyUs = encoderUncertaintyUs + phase.uncertaintyUs;
        bool overlapKnown{};
        double phaseOverlapCompletionUs{};
        double phaseOverlapCompletionUncertaintyUs{};
        bool residualDerivedFromFullCost{};
        std::optional<PhaseGlobalCostEstimate> online = mRuntimeCostTracker->estimate(overlapKey);
        if (!online.has_value() && residualAugmentation)
        {
            PhaseGlobalActionKey fullKey = overlapKey;
            fullKey.residualAugmentation = false;
            fullKey.residualAnchor = PhaseGlobalResidualAnchor::kNone;
            online = mRuntimeCostTracker->estimate(fullKey);
            residualDerivedFromFullCost = online.has_value();
        }
        if (online.has_value())
        {
            overlapMakespanUs = static_cast<double>(online->makespanMedianMs) * 1000.0;
            overlapUncertaintyUs = static_cast<double>(online->uncertaintyMs) * 1000.0;
            PhaseGlobalActionKey eligibilityKey = overlapKey;
            eligibilityKey.residualAugmentation = residualAugmentation && !residualDerivedFromFullCost;
            if (residualDerivedFromFullCost)
            {
                eligibilityKey.residualAnchor = PhaseGlobalResidualAnchor::kNone;
            }
            overlapKnown = mRuntimeCostTracker->overlapEligible(eligibilityKey);
        }
        else
        {
            if (kind == PhaseGlobalActionKind::kEncoderPrefill)
            {
                for (PhaseEncoderPrefillBatchCost const& cost : mConfig.globalEncoderPrefillCosts)
                {
                    if (cost.encoderBatchSize >= encoderBatchIndices.size()
                        && cost.prefillBatchSize >= phase.key.primaryBatchSize
                        && cost.maxEncoderInputTokens >= encoderInputTokens
                        && cost.maxPrefillChunkLength >= phase.key.chunkLength
                        && cost.maxPrefillPastKVLength
                            >= phase.key.primaryContextBucket * mConfig.globalDecodeContextBucketTokens
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
                        && cost.decodeBatchSize >= phase.key.primaryBatchSize
                        && cost.maxEncoderInputTokens >= encoderInputTokens
                        && cost.maxDecodeContextLength
                            >= phase.key.primaryContextBucket * mConfig.globalDecodeContextBucketTokens
                        && static_cast<double>(cost.makespanP95GpuMs) * 1000.0 < overlapMakespanUs)
                    {
                        overlapMakespanUs = static_cast<double>(cost.makespanP95GpuMs) * 1000.0;
                        overlapUncertaintyUs = 0.0;
                        overlapKnown = true;
                    }
                }
            }
        }
        if (residualAugmentation && overlapKnown)
        {
            double const fullPhaseUs = mActiveGlobalPdExecution->candidate.predictedMakespanUs > 0.0
                ? mActiveGlobalPdExecution->candidate.predictedMakespanUs
                : mActiveGlobalPdExecution->candidate.predictedBlockingUs;
            double const residualPhaseUs = phase.predictedMakespanUs;
            double const interferenceUs = std::max(0.0, overlapMakespanUs - std::max(encoderMakespanUs, fullPhaseUs));
            overlapMakespanUs = std::max(encoderMakespanUs, residualPhaseUs) + interferenceUs;
            phaseOverlapCompletionUs = residualPhaseUs + interferenceUs;
            phaseOverlapCompletionUncertaintyUs = phase.uncertaintyUs;
        }
        PhaseGlobalOverlapCostDiagnostic const localDiagnostic = mRuntimeCostTracker->overlapDiagnostic(overlapKey);
        mGlobalEncoderOverlapKnownCosts += overlapKnown ? 1U : 0U;
        switch (localDiagnostic.status)
        {
        case PhaseGlobalOverlapCostStatus::kNoSamples: ++mGlobalEncoderOverlapNoSamples; break;
        case PhaseGlobalOverlapCostStatus::kInsufficientSamples: ++mGlobalEncoderOverlapInsufficientSamples; break;
        case PhaseGlobalOverlapCostStatus::kEligible: break;
        case PhaseGlobalOverlapCostStatus::kUnprofitable: ++mGlobalEncoderOverlapUnprofitable; break;
        }
        PhaseContextualPairDirection const contextualDirection
            = phaseContextualPairDirection(kind, overlapKey.residualAnchor);
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
        double protectedSlackUs = encoderSlackUs;
        for (PhaseProtectedCompletion const& completion : phase.protectedCompletions)
        {
            protectedSlackUs = std::min(protectedSlackUs, completion.slackUs);
        }
        double const robustSerialUs
            = encoderMakespanUs + encoderUncertaintyUs + phase.predictedMakespanUs + phase.uncertaintyUs;
        bool const probeIntervalReady = mLastGlobalSafeProbeSequence == 0U
            || mGlobalDecisionSequence - mLastGlobalSafeProbeSequence >= mConfig.globalSafeProbeInterval;
        double phaseProtectedSlackUs = std::numeric_limits<double>::infinity();
        for (PhaseProtectedCompletion const& completion : phase.protectedCompletions)
        {
            phaseProtectedSlackUs = std::min(phaseProtectedSlackUs, completion.slackUs);
        }
        if (residualAugmentation && phase.key.kind == PhaseGlobalActionKind::kDecode)
        {
            // The decode candidate's original protection is its queue-formation
            // deadline. Once that candidate is already running, reuse it only
            // for the residual cost and protect the continuation with the
            // request TPOT deadline instead. Otherwise a 1--2 ms batching wait
            // target permanently prevents a safe E+D calibration probe even
            // when the serving SLO has tens of milliseconds of headroom.
            double const requestTpotTargetUs = mTpotTargets.empty() ? 0.0 : *mTpotTargets.begin();
            double const continuationTargetUs
                = requestTpotTargetUs > 0.0 ? requestTpotTargetUs : mConfig.encodedCapacityDecodeTpotTargetUs;
            if (continuationTargetUs > 0.0)
            {
                double const elapsedUs = std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - mActiveGlobalPdExecution->startedAt)
                                             .count();
                phaseProtectedSlackUs = std::max(0.0, continuationTargetUs - elapsedUs);
            }
        }
        double const residualPhaseProbeUs
            = phase.predictedMakespanUs + phase.uncertaintyUs + mConfig.encoderDispatchCostSafetyMarginUs;
        bool const residualProbeSafe = residualAugmentation && probeIntervalReady
            && mConfig.globalSafeProbeSlackMultiplier > 0.0F
            && encoderSlackUs >= static_cast<double>(mConfig.globalSafeProbeSlackMultiplier)
                    * (encoderMakespanUs + encoderUncertaintyUs)
            && phaseProtectedSlackUs
                >= static_cast<double>(mConfig.globalSafeProbeSlackMultiplier) * residualPhaseProbeUs;
        bool const calibrationProbe = mGlobalWarmupProbeMode && calibrationTarget && needsLocalCalibration;
        bool const safeProbe = calibrationProbe
            || (!overlapKnown && needsLocalCalibration
                && ((mConfig.globalSafeProbeSlackMultiplier > 0.0F && probeIntervalReady
                        && protectedSlackUs
                            >= static_cast<double>(mConfig.globalSafeProbeSlackMultiplier) * robustSerialUs)
                    || residualProbeSafe));
        if (safeProbe)
        {
            ++mGlobalEncoderOverlapSafeProbeEligible;
        }
        else if (!overlapKnown && needsLocalCalibration && !mGlobalWarmupProbeMode)
        {
            if (mConfig.globalSafeProbeSlackMultiplier <= 0.0F)
            {
                ++mGlobalEncoderOverlapProbeDisabled;
            }
            else if (!probeIntervalReady)
            {
                ++mGlobalEncoderOverlapProbeIntervalBlocked;
            }
            else
            {
                ++mGlobalEncoderOverlapProbeSlackBlocked;
            }
        }
        if (safeProbe)
        {
            double const optimisticMakespanUs = std::max(encoderMakespanUs, phase.predictedMakespanUs);
            overlapMakespanUs = optimisticMakespanUs;
            overlapUncertaintyUs = std::max(0.0, robustSerialUs - optimisticMakespanUs);
            if (residualAugmentation)
            {
                overlapUncertaintyUs
                    = std::max(encoderUncertaintyUs, phase.uncertaintyUs) + mConfig.encoderDispatchCostSafetyMarginUs;
                phaseOverlapCompletionUs = residualPhaseProbeUs;
                phaseOverlapCompletionUncertaintyUs = phase.uncertaintyUs;
            }
        }
        PhaseGlobalActionCandidate overlap;
        overlap.key = overlapKey;
        overlap.overlapCostKnown = overlapKnown;
        overlap.safeProbeEligible = safeProbe;
        overlap.calibrationProbe = calibrationProbe;
        overlap.predictedBlockingUs = overlapMakespanUs;
        overlap.predictedMakespanUs = overlapMakespanUs;
        overlap.uncertaintyUs = overlapUncertaintyUs;
        overlap.referenceWorkUs = encoderReferenceUs + phase.referenceWorkUs;
        overlap.predictedCostSource
            = overlapKnown ? PhaseServiceReferenceSource::kRuntimeExact : PhaseServiceReferenceSource::kDerivedIsolated;
        overlap.referenceCostSource = PhaseServiceReferenceSource::kDerivedIsolated;
        overlap.requestServiceLagUs = std::max(encoderServiceLagUs, phase.requestServiceLagUs);
        overlap.memory = encoder.memory;
        overlap.memory.allocateBytes = saturatedAdd(overlap.memory.allocateBytes, phase.memory.allocateBytes);
        overlap.memory.guaranteedGrowthBytes
            = saturatedAdd(overlap.memory.guaranteedGrowthBytes, phase.memory.guaranteedGrowthBytes);
        overlap.memory.nearReclaimBytes = saturatedAdd(overlap.memory.nearReclaimBytes, phase.memory.nearReclaimBytes);
        overlap.primaryRequestIds = encoder.primaryRequestIds;
        overlap.secondaryRequestIds = phase.requestIds;
        overlap.secondaryStableSlotIds = phase.primaryStableSlotIds;
        phaseGlobalFinalizeCandidate(overlap);
        overlap.protectedCompletions.push_back({encoderSlackUs, overlapMakespanUs + visionPrefillMakespanUs,
            overlapUncertaintyUs + visionPrefillUncertaintyUs, PhaseProtectedKind::kEncoder, encoderProtectedRequestId,
            std::max(1.0,
                encoderReferenceUs / static_cast<double>(std::max<size_t>(1U, encoderBatchIndices.size()))
                    + visionPrefillMakespanUs),
            PhaseServiceReferenceSource::kDerivedIsolated, encoderServiceLagUs});
        for (PhaseProtectedCompletion completion : phase.protectedCompletions)
        {
            completion.predictedCompletionUs = residualAugmentation ? phaseOverlapCompletionUs : overlapMakespanUs;
            completion.uncertaintyUs
                = residualAugmentation ? phaseOverlapCompletionUncertaintyUs : overlapUncertaintyUs;
            overlap.protectedCompletions.push_back(completion);
        }
        PhaseContextualPairKind const pairKind = kind == PhaseGlobalActionKind::kEncoderPrefill
            ? PhaseContextualPairKind::kEncoderPrefill
            : PhaseContextualPairKind::kEncoderDecode;
        PhaseContextualPdMode const contextualMode = mRuntimeCostTracker->contextualPairConfig(pairKind).mode;
        if (contextualMode != PhaseContextualPdMode::kDisabled)
        {
            constexpr int32_t kChunkQuantum = 128;
            int32_t const encoderBatchCapacity = static_cast<int32_t>(std::min<size_t>(
                mConfig.maxEncoderBatchSize, static_cast<size_t>(std::numeric_limits<int32_t>::max())));
            int32_t const phaseBatchCapacity = kind == PhaseGlobalActionKind::kEncoderPrefill
                ? mConfig.contextualPrefillBatchCapacity
                : mConfig.contextualDecodeBatchCapacity;
            double const incumbentReferenceUs = residualAugmentation && mActiveGlobalPdExecution.has_value()
                ? std::max(mActiveGlobalPdExecution->candidate.predictedMakespanUs,
                      mActiveGlobalPdExecution->candidate.predictedBlockingUs)
                : 0.0;
            double const requestedSkewFraction
                = residualAugmentation && incumbentReferenceUs > 0.0 ? residualElapsedUs / incumbentReferenceUs : -1.0;
            PhaseExecutionSet const contextualOutstanding = residualAugmentation
                ? phase.key.kind == PhaseGlobalActionKind::kPrefill ? PhaseExecutionSet::kPrefill
                                                                    : PhaseExecutionSet::kDecode
                : PhaseExecutionSet::kNone;
            PhaseContextualPairInput const contextualInput{encoderMakespanUs, phase.predictedMakespanUs,
                protectedSlackUs, static_cast<int32_t>(encoderBatchIndices.size()), phase.key.primaryBatchSize,
                encoderBatchCapacity, phaseBatchCapacity, chunkLength, kChunkQuantum, encoderContextBucket,
                phase.key.primaryContextBucket, overlapKey.executionVariant, residualAugmentation,
                overlapKey.residualAnchor, residualElapsedUs, incumbentReferenceUs, requestedSkewFraction,
                contextualOutstanding};
            overlap.contextualEncoderPairFeatures = phaseContextualPairFeatures(contextualInput);
            overlap.contextualEncoderPairDirection = contextualDirection;
            overlap.contextualEncoderPairFeatureValid = true;
            PhaseContextualPdEstimate const contextual = mRuntimeCostTracker->predictContextualDirection(
                overlap.contextualEncoderPairDirection, overlap.contextualEncoderPairFeatures);
            overlap.contextualEncoderPairReady = contextual.ready;
            overlap.contextualEncoderPairMean = contextual.mean;
            overlap.contextualEncoderPairUncertainty = contextual.uncertainty;
            overlap.contextualEncoderPairLowerConfidenceBound = contextual.lowerConfidenceBound;
            overlap.contextualEncoderPairReferenceWorkUs = encoderMakespanUs + phase.predictedMakespanUs;
            if (contextualMode == PhaseContextualPdMode::kActive && contextual.ready)
            {
                overlap.decisionCostKnown = true;
                overlap.contextualScalarAuthorityApplied = true;
                overlap.decisionMakespanUs = phaseContextualDecisionMakespanUs(
                    overlap.contextualEncoderPairReferenceWorkUs, contextual.lowerConfidenceBound);
            }
            overlap.contextualEncoderPairExploration = !contextual.ready && safeProbe;
            if (kind == PhaseGlobalActionKind::kEncoderPrefill)
            {
                mContextualEpReady += contextual.ready ? 1U : 0U;
            }
            else
            {
                mContextualEdReady += contextual.ready ? 1U : 0U;
            }
        }
        candidates.push_back(std::move(overlap));
    };
    if (!encoderPrefillExclusive && prefillForEncoder.has_value())
    {
        addEncoderOverlap(PhaseGlobalActionKind::kEncoderPrefill, *prefillForEncoder);
    }
    if (mConfig.encoderDecodeEligible(encoderPrefillExclusive) && decodeForEncoder.has_value())
    {
        addEncoderOverlap(PhaseGlobalActionKind::kEncoderDecode, *decodeForEncoder);
    }

    std::vector<PhaseGlobalActionCandidate> unifiedCandidateFrontier = pdCandidateFrontier;
    for (PhaseGlobalActionCandidate const& candidate : candidates)
    {
        bool const present = std::any_of(unifiedCandidateFrontier.begin(), unifiedCandidateFrontier.end(),
            [&](PhaseGlobalActionCandidate const& existing) { return existing.candidateId == candidate.candidateId; });
        if (!present)
        {
            unifiedCandidateFrontier.push_back(candidate);
        }
    }

    PhaseGlobalSelectionAudit selectorAudit;
    selectorAudit.mechanismInputs = pdAudit.mechanismInputs;
    for (PhaseGlobalActionCandidate& mechanism : selectorAudit.mechanismInputs)
    {
        bool const protectsEncoder = std::any_of(mechanism.protectedCompletions.begin(),
            mechanism.protectedCompletions.end(),
            [](PhaseProtectedCompletion const& completion) { return completion.kind == PhaseProtectedKind::kEncoder; });
        if (!protectsEncoder && encoderProtectedRequestId != 0U)
        {
            mechanism.protectedCompletions.push_back(
                {encoderSlackUs, mechanism.predictedMakespanUs + encoderMakespanUs + visionPrefillMakespanUs,
                    mechanism.uncertaintyUs + encoderUncertaintyUs + visionPrefillUncertaintyUs,
                    PhaseProtectedKind::kEncoder, encoderProtectedRequestId,
                    std::max(1.0,
                        encoderReferenceUs / static_cast<double>(std::max<size_t>(1U, encoderBatchIndices.size()))
                            + visionPrefillMakespanUs),
                    PhaseServiceReferenceSource::kDerivedIsolated, encoderServiceLagUs});
        }
    }
    for (PhaseGlobalActionCandidate candidate : candidates)
    {
        phaseRestoreNonContextualPolicy(candidate);
        auto const present = std::find_if(selectorAudit.mechanismInputs.begin(), selectorAudit.mechanismInputs.end(),
            [&](PhaseGlobalActionCandidate const& existing) { return existing.candidateId == candidate.candidateId; });
        if (present == selectorAudit.mechanismInputs.end())
        {
            selectorAudit.mechanismInputs.push_back(std::move(candidate));
        }
        else
        {
            *present = std::move(candidate);
        }
    }
    PhaseGlobalDecision const myopicDecision
        = mGlobalScheduler.select(candidates, mUnifiedEventCallback ? &selectorAudit.inputs : nullptr);
    mLastGlobalFormationPredictedRows = 0U;
    mLastGlobalFormationHorizonUs = 0.0;
    mLastGlobalFormationCostGapUs = 0.0;
    mLastGlobalFormationPlannerUs = 0.0;
    mLastGlobalFormationSnapshotId = 0U;
    mLastGlobalFormationH2Action = PhaseGlobalActionKind::kNone;
    mLastGlobalFormationOracleAction = PhaseGlobalActionKind::kNone;
    mLastGlobalFormationDecodeViolationUs = 0.0;
    mLastScalarFormation = {};
    double formationDecodeServiceBudgetUs = std::numeric_limits<double>::infinity();
    bool formationEvaluated{};
    PhaseFormationOracleResult formationOracle;
    PhaseFormationWork formationTarget;
    for (PhaseGlobalActionCandidate const& candidate : candidates)
    {
        PhaseFormationWork const work = phaseFormationWork(candidate);
        formationTarget.encoderRows = std::max(formationTarget.encoderRows, work.encoderRows);
        formationTarget.prefillRows = std::max(formationTarget.prefillRows, work.prefillRows);
        formationTarget.decodeRows = std::max(formationTarget.decodeRows, work.decodeRows);
    }
    size_t const formationActivePhases = static_cast<size_t>(formationTarget.encoderRows > 0U)
        + static_cast<size_t>(formationTarget.prefillRows > 0U) + static_cast<size_t>(formationTarget.decodeRows > 0U);
    bool const formationSelectionActive = phasePolicyUsesTransition(mConfig.policyMode)
        && mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive;
    bool const formationEvaluationEnabled = formationSelectionActive;
    if (formationEvaluationEnabled && !residualAugmentation && formationActivePhases >= 2U)
    {
        // Compare the largest exact phase-local cohorts exposed by the current
        // mechanism frontier. This now covers ordinary P+D snapshots as well
        // as E plus P/D; no workload mode or future arrival enters the target.
        PhaseFormationWork const target = formationTarget;
        for (PhaseGlobalActionCandidate const& candidate : candidates)
        {
            for (PhaseProtectedCompletion const& completion : candidate.protectedCompletions)
            {
                if (completion.kind == PhaseProtectedKind::kDecode)
                {
                    formationDecodeServiceBudgetUs = std::min(formationDecodeServiceBudgetUs, completion.slackUs);
                }
            }
        }
        std::vector<PhaseFormationKnownCompletion> knownCompletions;
        for (PhaseDecodeCompletionPreview const& preview : mServer.previewPendingDecodeCompletions(1U))
        {
            int32_t maxContextLength{};
            for (int32_t const contextLength : preview.contextLengths)
            {
                maxContextLength = std::max(maxContextLength, contextLength);
            }
            std::optional<float> const serviceMs = mServer.estimateGlobalDecodeComponentP95(
                static_cast<int32_t>(preview.requestIds.size()), maxContextLength, false);
            if (!serviceMs.has_value() || *serviceMs <= 0.0F)
            {
                continue;
            }
            knownCompletions.push_back({preview.eventId, preview.predictedWaitUs, {0U, 0U, preview.requestIds.size()},
                preview.waitUncertaintyUs, static_cast<double>(*serviceMs) * 1000.0, 0.0, preview.serviceBudgetUs});
            if (preview.serviceBudgetUs > 0.0)
            {
                formationDecodeServiceBudgetUs
                    = std::min(formationDecodeServiceBudgetUs, preview.predictedWaitUs + preview.serviceBudgetUs);
            }
        }
        PhaseFormationSnapshot const snapshot{
            mGlobalSnapshotEpoch + 1U, target, std::move(knownCompletions), formationDecodeServiceBudgetUs};
        double encoderReferenceUs{};
        double prefillReferenceUs{};
        double decodeReferenceUs{};
        for (PhaseGlobalActionCandidate const& candidate : candidates)
        {
            double const reference
                = candidate.referenceWorkUs > 0.0 ? candidate.referenceWorkUs : candidate.predictedMakespanUs;
            switch (candidate.key.kind)
            {
            case PhaseGlobalActionKind::kEncoder: encoderReferenceUs = std::max(encoderReferenceUs, reference); break;
            case PhaseGlobalActionKind::kPrefill: prefillReferenceUs = std::max(prefillReferenceUs, reference); break;
            case PhaseGlobalActionKind::kDecode: decodeReferenceUs = std::max(decodeReferenceUs, reference); break;
            case PhaseGlobalActionKind::kNone:
            case PhaseGlobalActionKind::kEncoderPrefill:
            case PhaseGlobalActionKind::kEncoderDecode:
            case PhaseGlobalActionKind::kPrefillDecode:
            case PhaseGlobalActionKind::kWait: break;
            }
        }
        double const referenceWorkUs = std::max(1.0, encoderReferenceUs + prefillReferenceUs + decodeReferenceUs);
        auto const plannerStart = std::chrono::steady_clock::now();
        formationOracle = phaseFormationApplyH2(candidates, snapshot, target, referenceWorkUs);
        mLastScalarFormation = modelFormationSnapshot(formationOracle, candidates);
        mLastGlobalFormationPlannerUs
            = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - plannerStart).count();
        mLastGlobalFormationSnapshotId = phaseFormationSnapshotId(snapshot);
        formationEvaluated = true;
        ++mGlobalFormationLookaheads;
        if (formationOracle.selectedAction.has_value())
        {
            mLastGlobalFormationHorizonUs = formationOracle.sequences[*formationOracle.selectedAction].makespanUs;
            mLastGlobalFormationOracleAction = candidates[*formationOracle.selectedAction].key.kind;
        }
    }

    ++mGlobalDecisionSequence;
    PhaseGlobalDecision const decision = formationEvaluated
        ? mGlobalScheduler.select(candidates, mUnifiedEventCallback ? &selectorAudit.inputs : nullptr)
        : myopicDecision;
    selectorAudit.decision = decision;
    selectorAudit.pdInputs = std::move(pdAudit.inputs);
    selectorAudit.pdDecision = pdAudit.decision;
    selectorAudit.pdDecodeGuard = pdAudit.decodeGuard;
    std::optional<size_t> const h2SelectedIndex
        = formationSelectionActive ? decision.selectedIndex : formationOracle.selectedAction;
    if (formationEvaluated && h2SelectedIndex.has_value())
    {
        mLastGlobalFormationH2Action = candidates[*h2SelectedIndex].key.kind;
    }
    ++mGlobalDecisions;
    std::optional<size_t> selectedIndex = decision.selectedIndex;
    std::optional<PhaseFormationRegret> myopicFormationRegret;
    if (formationEvaluated && myopicDecision.selectedIndex.has_value())
    {
        PhaseFormationRegret const regret
            = phaseFormationPredictedRegret(formationOracle, *myopicDecision.selectedIndex);
        if (regret.valid)
        {
            myopicFormationRegret = regret;
        }
    }
    // H=2 is allowed to change production ordering only when the same robust
    // frontier predicts a strict improvement over the myopic action. Equal
    // violation/equal-horizon ties preserve stable mechanism ordering.
    if (formationSelectionActive && formationEvaluated && selectedIndex.has_value()
        && myopicDecision.selectedIndex.has_value()
        && candidates[*selectedIndex].candidateId != candidates[*myopicDecision.selectedIndex].candidateId
        && !phaseFormationShouldReplaceMyopic(formationOracle, *myopicDecision.selectedIndex, *selectedIndex))
    {
        selectedIndex = myopicDecision.selectedIndex;
    }
    if (myopicDecision.selectedIndex.has_value() && selectedIndex.has_value())
    {
        PhaseGlobalActionCandidate const& myopic = candidates[*myopicDecision.selectedIndex];
        PhaseGlobalActionCandidate const& formationAware = candidates[*selectedIndex];
        mGlobalFormationSelectionChanges += myopic.candidateId != formationAware.candidateId ? 1U : 0U;
    }
    if (mGlobalWarmupProbeMode && mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive)
    {
        auto const calibration
            = std::min_element(candidates.begin(), candidates.end(), [&](auto const& left, auto const& right) {
                  auto calibrationProgress = [&](auto const& candidate) {
                      if (!candidate.calibrationProbe)
                      {
                          return std::numeric_limits<size_t>::max();
                      }
                      return mRuntimeCostTracker->overlapDiagnostic(candidate.key).sampleCount;
                  };
                  return calibrationProgress(left) < calibrationProgress(right);
              });
        if (calibration != candidates.end())
        {
            if (calibration->calibrationProbe)
            {
                selectedIndex = static_cast<size_t>(std::distance(candidates.begin(), calibration));
            }
        }
    }
    if (!mGlobalWarmupProbeMode)
    {
        auto pairKind = [](PhaseGlobalActionKind kind) {
            return kind == PhaseGlobalActionKind::kEncoderPrefill ? PhaseContextualPairKind::kEncoderPrefill
                                                                  : PhaseContextualPairKind::kEncoderDecode;
        };
        auto isEncoderOverlap = [](PhaseGlobalActionKind kind) {
            return kind == PhaseGlobalActionKind::kEncoderPrefill || kind == PhaseGlobalActionKind::kEncoderDecode;
        };
        auto contextualReady = [&](PhaseGlobalActionCandidate const& candidate) {
            return isEncoderOverlap(candidate.key.kind) && candidate.contextualEncoderPairFeatureValid
                && candidate.contextualEncoderPairReady
                && !(candidate.safeProbeEligible && !candidate.overlapCostKnown);
        };
        for (PhaseGlobalActionCandidate const& candidate : candidates)
        {
            if (!contextualReady(candidate))
            {
                continue;
            }
            bool const selectedOverlap
                = selectedIndex.has_value() && candidates[*selectedIndex].candidateId == candidate.candidateId;
            bool const contextualOverlap = candidate.contextualEncoderPairLowerConfidenceBound > 0.0;
            PhaseContextualPairKind const kind = pairKind(candidate.key.kind);
            if (selectedOverlap != contextualOverlap)
            {
                if (kind == PhaseContextualPairKind::kEncoderPrefill)
                {
                    ++mContextualEpDecisionDisagreements;
                }
                else
                {
                    ++mContextualEdDecisionDisagreements;
                }
            }
            mRuntimeCostTracker->recordContextualDirectionSelection(
                candidate.contextualEncoderPairDirection, selectedOverlap, candidate.contextualEncoderPairExploration);
        }
    }
    std::optional<PhaseFormationRegret> formationRegret;
    if (formationEvaluated && selectedIndex.has_value())
    {
        PhaseFormationRegret const regret = phaseFormationPredictedRegret(formationOracle, *selectedIndex);
        if (regret.valid)
        {
            formationRegret = regret;
            ++mGlobalFormationRegretSamples;
            mGlobalFormationPredictedRegretUs += regret.predictedRegretUs;
            mMaxGlobalFormationPredictedRegretUs
                = std::max(mMaxGlobalFormationPredictedRegretUs, regret.predictedRegretUs);
            mLastGlobalFormationCostGapUs = regret.predictedRegretUs;
            mLastGlobalFormationDecodeViolationUs = regret.selectedDecodeViolationUs;
            mGlobalFormationPositiveRegrets += regret.predictedRegretUs > 0.0 ? 1U : 0U;
        }
    }
    if (!selectedIndex.has_value())
    {
        if (residualAugmentation)
        {
            ++mGlobalResidualAugmentationUnknownCostRejects;
        }
        return false;
    }
    PhaseGlobalActionCandidate selected = candidates[*selectedIndex];
    if (formationEvaluated)
    {
        mLastGlobalFormationHorizonUs = selected.predictedHorizonUs;
        if (selected.key.kind == pd->key.kind)
        {
            ++mGlobalFormationPdSelections;
        }
        else if (selected.key.kind == PhaseGlobalActionKind::kEncoderPrefill
            || selected.key.kind == PhaseGlobalActionKind::kEncoderDecode)
        {
            ++mGlobalFormationOverlapSelections;
        }
        if (h2SelectedIndex.has_value())
        {
            if (*h2SelectedIndex == *selectedIndex)
            {
                ++mGlobalFormationH2Agreements;
            }
            else
            {
                ++mGlobalFormationPostPolicyOverrides;
                mLastGlobalFormationCostGapUs
                    = std::max(0.0, selected.predictedHorizonUs - candidates[*h2SelectedIndex].predictedHorizonUs);
            }
        }
    }
    mLastGlobalAction = selected.key.kind;
    bool const safeProbe = selected.calibrationProbe || (selected.safeProbeEligible && !selected.overlapCostKnown);
    if (safeProbe && mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive)
    {
        mLastGlobalSafeProbeSequence = mGlobalDecisionSequence;
        ++mGlobalSafeProbes;
    }
    if (formationEvaluated && myopicDecision.selectedIndex.has_value()
        && candidates[*myopicDecision.selectedIndex].candidateId != selected.candidateId)
    {
        mPendingGlobalFormationRealizedEpisode = PhaseFormationRealizedEpisodeStart{mLastGlobalFormationSnapshotId,
            selected.key.kind, candidates[*myopicDecision.selectedIndex].key.kind, mLastGlobalFormationOracleAction,
            myopicFormationRegret.has_value() ? myopicFormationRegret->predictedRegretUs : 0.0,
            formationDecodeServiceBudgetUs, 0.0};
    }

    // Stop the policy timer before lease materialization, telemetry JSON, TRT
    // preparation, and CUDA enqueue. Those are decision-realization costs and
    // are measured separately from scheduler hot-path latency.
    recordGlobalDecisionCost(decisionStart);

    if (residualAugmentation)
    {
        if (selected.key.kind == PhaseGlobalActionKind::kPrefill || selected.key.kind == PhaseGlobalActionKind::kDecode)
        {
            bool const overlapKnown = std::any_of(candidates.begin(), candidates.end(), [](auto const& candidate) {
                return (candidate.key.kind == PhaseGlobalActionKind::kEncoderPrefill
                           || candidate.key.kind == PhaseGlobalActionKind::kEncoderDecode)
                    && (candidate.overlapCostKnown || candidate.safeProbeEligible);
            });
            mGlobalResidualAugmentationUnknownCostRejects += overlapKnown ? 0U : 1U;
            return false;
        }
        ELLM_CHECK(selected.key.kind == PhaseGlobalActionKind::kEncoderPrefill
                || selected.key.kind == PhaseGlobalActionKind::kEncoderDecode,
            "Residual lease augmentation selected an unsupported action");
        ELLM_CHECK(mGlobalExecutionLease.has_value() && mActiveGlobalPdExecution.has_value(),
            "Residual lease augmentation lost its active phase");
        double const phaseElapsedMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - mActiveGlobalPdExecution->startedAt)
                                          .count();
        double const phaseReferenceUs = std::max(mActiveGlobalPdExecution->candidate.predictedMakespanUs,
            mActiveGlobalPdExecution->candidate.predictedBlockingUs);
        PhaseStartSkewBucket const startSkew = phaseRequestedStartSkew(phaseElapsedMs * 1000.0, phaseReferenceUs);
        std::optional<PhaseGlobalDispatchPlan> const augmented
            = phaseGlobalAugmentedDispatchPlan(kTHREE_PHASE_PLAN_NAMESPACE | ++mGlobalPlanSequence,
                kTHREE_PHASE_PLAN_NAMESPACE | ++mGlobalSnapshotEpoch, *mGlobalExecutionLease, selected, startSkew);
        ELLM_CHECK(augmented.has_value(), "Residual lease augmentation does not match the active phase rows");
        recordUnifiedDecision(selected, *augmented, &unifiedCandidateFrontier, &selectorAudit);
        if (!preparedEncoderReady)
        {
            mGlobalEncoderBatchIndices = encoderBatchIndices;
        }
        mInFlightGlobalEncoderKey = encoderKey;
        mInFlightGlobalEncoderReferenceMs = encoderReferenceUs / 1000.0;
        PendingGlobalOverlapObservation observation{selected.key, static_cast<float>(selected.referenceWorkUs / 1000.0),
            0.0F, 0.0F, static_cast<float>(phaseElapsedMs), true};
        observation.contextualFeatures = selected.contextualEncoderPairFeatures;
        observation.contextualDirection = selected.contextualEncoderPairDirection;
        observation.contextualFeatureValid = selected.contextualEncoderPairFeatureValid;
        observation.contextualExploration = selected.contextualEncoderPairExploration;
        observation.contextualReferenceWorkMs
            = static_cast<float>(selected.contextualEncoderPairReferenceWorkUs / 1000.0);
        observation.contextualLowerConfidenceBound = selected.contextualEncoderPairLowerConfidenceBound;
        observation.planId = augmented->planId;
        mPendingGlobalOverlapObservation = observation;
        bool const started = preparedEncoderReady ? submitPreparedEncoder() : startNextEncoder();
        if (!started)
        {
            mPendingGlobalOverlapObservation.reset();
            mUnifiedDecisionByPlan.erase(augmented->planId);
            return false;
        }
        mGlobalFormationRealizedTracker.remapPlan(mGlobalExecutionLease->planId, augmented->planId);
        mGlobalExecutionLease = *augmented;
        mEncoderPlanId = augmented->planId;
        mEncoderActionId = selected.candidateId;
        mEncoderActionKind = selected.key.kind;
        validateGlobalExecutionLaunch();
        if (selected.key.kind == PhaseGlobalActionKind::kEncoderPrefill)
        {
            ++mGlobalResidualEncoderPrefillSelections;
        }
        else
        {
            ++mGlobalResidualEncoderDecodeSelections;
        }
        return true;
    }

    if (selected.key.kind == PhaseGlobalActionKind::kEncoder)
    {
        static_cast<void>(beginGlobalExecutionLease(selected, &unifiedCandidateFrontier, &selectorAudit));
        if (!preparedEncoderReady)
        {
            mGlobalEncoderBatchIndices = encoderBatchIndices;
        }
        mInFlightGlobalEncoderKey = encoderKey;
        mInFlightGlobalEncoderReferenceMs = encoderReferenceUs / 1000.0;
        ++mGlobalEncoderSelections;
        bool const started = preparedEncoderReady ? submitPreparedEncoder() : startNextEncoder();
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
        std::optional<PhaseGlobalActionCandidate>& phase
            = selected.key.kind == PhaseGlobalActionKind::kEncoderPrefill ? prefillForEncoder : decodeForEncoder;
        ELLM_CHECK(phase.has_value(), "Selected encoder overlap has no matching phase action");
        PhaseGlobalDispatchPlan const executionPlan
            = beginGlobalExecutionLease(selected, &unifiedCandidateFrontier, &selectorAudit);
        if (!preparedEncoderReady)
        {
            mGlobalEncoderBatchIndices = encoderBatchIndices;
        }
        mInFlightGlobalEncoderKey = encoderKey;
        mInFlightGlobalEncoderReferenceMs = encoderReferenceUs / 1000.0;
        PendingGlobalOverlapObservation observation{
            selected.key, static_cast<float>(selected.referenceWorkUs / 1000.0), 0.0F, 0.0F, 0.0F, false};
        observation.contextualFeatures = selected.contextualEncoderPairFeatures;
        observation.contextualDirection = selected.contextualEncoderPairDirection;
        observation.contextualFeatureValid = selected.contextualEncoderPairFeatureValid;
        observation.contextualExploration = selected.contextualEncoderPairExploration;
        observation.contextualReferenceWorkMs
            = static_cast<float>(selected.contextualEncoderPairReferenceWorkUs / 1000.0);
        observation.contextualLowerConfidenceBound = selected.contextualEncoderPairLowerConfidenceBound;
        observation.planId = executionPlan.planId;
        mPendingGlobalOverlapObservation = observation;
        bool const encoderStarted = preparedEncoderReady ? submitPreparedEncoder() : startNextEncoder();
        bool const phaseStarted = encoderStarted
            && mServer.dispatchGlobalAction(std::move(*phase), executionPlan.planId, executionPlan.snapshotEpoch);
        if (!encoderStarted || !phaseStarted)
        {
            mPendingGlobalOverlapObservation.reset();
            if (!encoderStarted && !phaseStarted)
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
    PhaseGlobalDispatchPlan const executionPlan
        = beginGlobalExecutionLease(selected, &unifiedCandidateFrontier, &selectorAudit);
    PhaseGlobalActionCandidate launched = selected;
    auto const launchedAt = std::chrono::steady_clock::now();
    bool const started
        = mServer.dispatchGlobalAction(std::move(selected), executionPlan.planId, executionPlan.snapshotEpoch);
    if (!started)
    {
        abandonGlobalExecutionLease();
        return false;
    }
    if (launched.key.kind == PhaseGlobalActionKind::kPrefill || launched.key.kind == PhaseGlobalActionKind::kDecode)
    {
        mActiveGlobalPdExecution = ActiveGlobalPdExecution{std::move(launched), launchedAt, {}};
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
    if (mPendingGlobalOverlapObservation->phaseGpuMs <= 0.0F && dispatch.has_value()
        && dispatch->globalPlanId == mPendingGlobalOverlapObservation->planId)
    {
        if (key.kind == PhaseGlobalActionKind::kEncoderPrefill && dispatch->prefillGpuMs > 0.0F
            && dispatch->prefillBatchSize == key.secondaryBatchSize && dispatch->prefillPaddedTokens > 0
            && dispatch->prefillPaddedTokens / dispatch->prefillBatchSize == key.chunkLength)
        {
            int32_t const contextBucket = (dispatch->prefillPastKVMax + mConfig.globalDecodeContextBucketTokens - 1)
                / mConfig.globalDecodeContextBucketTokens;
            if (contextBucket == key.secondaryContextBucket)
            {
                mPendingGlobalOverlapObservation->phaseGpuMs
                    = std::max(0.0F, dispatch->prefillGpuMs - mPendingGlobalOverlapObservation->phaseElapsedMs);
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
                mPendingGlobalOverlapObservation->phaseGpuMs
                    = std::max(0.0F, dispatch->decodeGpuMs - mPendingGlobalOverlapObservation->phaseElapsedMs);
            }
        }
    }
    if (mPendingGlobalOverlapObservation->encoderGpuMs <= 0.0F || mPendingGlobalOverlapObservation->phaseGpuMs <= 0.0F)
    {
        if (observedGlobalExecution() == PhaseExecutionSet::kNone)
        {
            mPendingGlobalOverlapObservation.reset();
        }
        return;
    }
    float const makespanMs
        = std::max(mPendingGlobalOverlapObservation->encoderGpuMs, mPendingGlobalOverlapObservation->phaseGpuMs);
    mRuntimeCostTracker->observe(
        mPendingGlobalOverlapObservation->key, {mPendingGlobalOverlapObservation->referenceWorkMs, makespanMs});
    if (mPendingGlobalOverlapObservation->contextualFeatureValid
        && mPendingGlobalOverlapObservation->contextualReferenceWorkMs > 0.0F)
    {
        double const reward
            = (static_cast<double>(mPendingGlobalOverlapObservation->contextualReferenceWorkMs) - makespanMs)
            / static_cast<double>(mPendingGlobalOverlapObservation->contextualReferenceWorkMs);
        static_cast<void>(
            mRuntimeCostTracker->observeContextualDirection(mPendingGlobalOverlapObservation->contextualDirection,
                mPendingGlobalOverlapObservation->contextualFeatures, reward));
    }
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
    bool const exclusiveEncoderPrefill = mConfig.serializeAllEncoderPrefill
        || (mConfig.exclusiveEncoderInputTokenThreshold > 0
            && candidateInputTokens > mConfig.exclusiveEncoderInputTokenThreshold);
    bool const exclusiveEncoderDecode = mConfig.serializeAllEncoderDecode;
    if (exclusiveEncoderPrefill || exclusiveEncoderDecode)
    {
        IndependentPhaseServerArbitrationSnapshot const snapshot = mServer.arbitrationSnapshot();
        bool const prefillInFlight = snapshot.busy
            && (snapshot.inFlightKind == PhaseDispatchKind::kPrefill
                || snapshot.inFlightKind == PhaseDispatchKind::kOverlap);
        bool const decodeInFlight = snapshot.busy
            && (snapshot.inFlightKind == PhaseDispatchKind::kDecode
                || snapshot.inFlightKind == PhaseDispatchKind::kOverlap);
        if ((exclusiveEncoderPrefill && prefillInFlight) || (exclusiveEncoderDecode && decodeInFlight))
        {
            ++mExclusiveEncoderPrefillDeferrals;
            return false;
        }
        if (exclusiveEncoderPrefill)
        {
            mServer.setPrefillDispatchBlocked(true);
            mExclusiveEncoderPrefillInFlight = true;
        }
        if (exclusiveEncoderDecode)
        {
            mServer.setDecodeDispatchBlocked(true);
            mExclusiveEncoderDecodeInFlight = true;
        }
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
        mEncoderPrepareStartHostNs = phaseTimelineNowNs();
        mEncoderPrepareEndHostNs = 0U;
        mEncoderExecuteStartHostNs = 0U;
        mEncoderExecuteEndHostNs = 0U;
        mEncoderDispatchHostNs = 0U;
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
            mEncoderDispatchHostNs = phaseTimelineNowNs();
            mEncoderExecuteStartHostNs = mEncoderDispatchHostNs;
            ELLM_CHECK(mVision.submit(std::move(submissions)), "Failed to start queued encoder batch");
            mEncoderPrepareEndHostNs = phaseTimelineNowNs();
            mEncoderExecuteEndHostNs = mEncoderPrepareEndHostNs;
            mServer.setExternalEncoderActive(true);
            markUnifiedEncoderSubmitted();
        }
    }
    catch (...)
    {
        mServer.setExternalEncoderActive(false);
        if (mExclusiveEncoderPrefillInFlight)
        {
            mServer.setPrefillDispatchBlocked(false);
            mExclusiveEncoderPrefillInFlight = false;
        }
        if (mExclusiveEncoderDecodeInFlight)
        {
            mServer.setDecodeDispatchBlocked(false);
            mExclusiveEncoderDecodeInFlight = false;
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
    if (exclusiveEncoderPrefill || exclusiveEncoderDecode)
    {
        ++mExclusiveEncoderBatches;
    }
    return true;
}

void PhaseThreeCoordinator::markUnifiedEncoderSubmitted()
{
    mEncoderGpuSubmitted = true;
    mEncoderDispatchHostNs = mEncoderDispatchHostNs > 0U ? mEncoderDispatchHostNs : phaseTimelineNowNs();
    mEncoderExecutionId = kTHREE_PHASE_EXECUTION_NAMESPACE | ++mUnifiedEncoderExecutionSequence;
    mEncoderActivityCorrelationId = mEncoding.front().requestId;
    if (mGlobalExecutionLease.has_value()
        && phaseExecutionSetContains(mGlobalExecutionLease->allowedOutstanding, PhaseExecutionSet::kEncoder))
    {
        mEncoderPlanId = mGlobalExecutionLease->planId;
        mEncoderActionId = mGlobalExecutionLease->candidateId;
        mEncoderActionKind = mGlobalExecutionLease->action;
        return;
    }
    mEncoderPlanId = kTHREE_PHASE_PLAN_NAMESPACE | ++mUnifiedFallbackPlanSequence;
    mEncoderActionId = mEncoderPlanId;
    mEncoderActionKind = PhaseGlobalActionKind::kEncoder;
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
        mEncoderPrepareEndHostNs = phaseTimelineNowNs();
        mLastEncoderPreparationUs
            = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - mEncoderPreparationStartedAt)
                  .count();
        mMaxEncoderPreparationUs = std::max(mMaxEncoderPreparationUs, mLastEncoderPreparationUs);
        ++mEncoderPreparationCompletions;
        if (mConfig.globalSchedulerMode == PhaseGlobalSchedulerMode::kActive)
        {
            ELLM_CHECK(mPreparedEncoder == nullptr, "A prepared encoder batch is already awaiting dispatch");
            mPreparedEncoder = std::move(prepared);
        }
        else
        {
            ELLM_CHECK(mVision.submitPrepared(std::move(prepared)), "Failed to submit prepared encoder batch");
            mServer.setExternalEncoderActive(true);
            markUnifiedEncoderSubmitted();
        }
    }
    catch (...)
    {
        mServer.setExternalEncoderActive(false);
        if (mExclusiveEncoderPrefillInFlight)
        {
            mServer.setPrefillDispatchBlocked(false);
            mExclusiveEncoderPrefillInFlight = false;
        }
        if (mExclusiveEncoderDecodeInFlight)
        {
            mServer.setDecodeDispatchBlocked(false);
            mExclusiveEncoderDecodeInFlight = false;
        }
        throw;
    }
    return true;
}

bool PhaseThreeCoordinator::submitPreparedEncoder()
{
    if (mPreparedEncoder == nullptr || mEncoding.empty() || mVision.busy())
    {
        return false;
    }
    std::shared_ptr<PhaseVisionPreparedBatch> prepared = std::move(mPreparedEncoder);
    mEncoderDispatchHostNs = phaseTimelineNowNs();
    mEncoderExecuteStartHostNs = mEncoderDispatchHostNs;
    ELLM_CHECK(mVision.submitPrepared(std::move(prepared)), "Failed to submit staged encoder batch");
    mEncoderExecuteEndHostNs = phaseTimelineNowNs();
    mServer.setExternalEncoderActive(true);
    markUnifiedEncoderSubmitted();
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
    if (mPreparedEncoder != nullptr && !mEncoderGpuSubmitted)
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
    if (mExclusiveEncoderPrefillInFlight)
    {
        // ready() is driven by the encoder CUDA completion event, so the
        // overlapping arena is safe for the next prefill enqueue now.
        mServer.setPrefillDispatchBlocked(false);
        mExclusiveEncoderPrefillInFlight = false;
    }
    if (mExclusiveEncoderDecodeInFlight)
    {
        mServer.setDecodeDispatchBlocked(false);
        mExclusiveEncoderDecodeInFlight = false;
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
        mRuntimeCostTracker->observe(
            *mInFlightGlobalEncoderKey, {static_cast<float>(mInFlightGlobalEncoderReferenceMs), mLastEncoderGpuMs});
    }
    mInFlightGlobalEncoderKey.reset();
    mInFlightGlobalEncoderReferenceMs = 0.0;
    if (mPendingGlobalOverlapObservation.has_value())
    {
        mPendingGlobalOverlapObservation->encoderGpuMs = mLastEncoderGpuMs;
    }
    mEncoding.clear();
    mEncoderGpuSubmitted = false;
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
    bool const containsUnmeasuredPayload = std::any_of(batchIndices.begin(), batchIndices.end(),
        [&](size_t index) { return mPending[index].estimatedPayloadBytes == 0U; });
    if (mEstimatedEncodedBytes == 0U && containsUnmeasuredPayload && batchIndices.size() > 1U)
    {
        ++mUnknownPayloadBootstrapSelections;
        batchIndices.resize(1U);
    }
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
    if (globalActive && mConfig.encoderBatchWaitUs > 0.0 && !batchFull && !mediaFull && !inputFull && !tokenFull
        && !resourceLimited && !capacityFull && batchSize == inputs.size() && mVisionInterarrivalSamples > 0U
        && mVisionInterarrivalEwmaUs > 0.0 && mLastVisionArrival != std::chrono::steady_clock::time_point{})
    {
        double const sinceArrivalUs
            = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - mLastVisionArrival).count();
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
            if (std::optional<PhaseGlobalCostEstimate> const online = mRuntimeCostTracker->estimate(key))
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
        std::optional<double> const futureEncoder = estimateEncoder(batchSize + 1U, inputTokens + averageInputTokens);
        if (predictedWaitUs > 0.0 && currentEncoder.has_value() && singletonEncoder.has_value()
            && futureEncoder.has_value())
        {
            size_t const estimatedPromptTokens
                = std::max<size_t>(1U, mEstimatedPromptTokens > 0U ? mEstimatedPromptTokens : averageInputTokens);
            int32_t const promptTokens = static_cast<int32_t>(
                std::min<size_t>(estimatedPromptTokens, static_cast<size_t>(std::numeric_limits<int32_t>::max())));
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
