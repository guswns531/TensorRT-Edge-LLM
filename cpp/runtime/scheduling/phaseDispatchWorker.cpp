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

#include "runtime/scheduling/phaseDispatchWorker.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "runtime/scheduling/phaseActivityTimeline.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <unordered_map>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{

uint64_t remainingDirectionalDelayUs(uint64_t requestedDelayUs, uint64_t incumbentEnqueueHostNs) noexcept
{
    uint64_t const currentTimestampNs = phaseTimelineNowNs();
    uint64_t const elapsedUs
        = currentTimestampNs > incumbentEnqueueHostNs ? (currentTimestampNs - incumbentEnqueueHostNs) / 1000U : 0U;
    return requestedDelayUs > elapsedUs ? requestedDelayUs - elapsedUs : 0U;
}

CUcontext getStreamCudaContext(cudaStream_t stream)
{
    check::check(stream != nullptr, "Phase execution requires explicit non-default CUDA streams.");
    CUcontext context{};
    CUDA_DRIVER_CHECK(cuStreamGetCtx(stream, &context));
    check::check(context != nullptr, "Phase CUDA stream has no owning CUDA context.");
    return context;
}

void validatePrimaryCudaContext(CUcontext context)
{
    CUcontext current{};
    CUDA_DRIVER_CHECK(cuCtxGetCurrent(&current));
    check::check(current == context, "Phase streams must belong to the thread's current CUDA context.");

    CUdevice device{};
    CUDA_DRIVER_CHECK(cuCtxGetDevice(&device));
    CUcontext primary{};
    CUDA_DRIVER_CHECK(cuDevicePrimaryCtxRetain(&primary, device));
    bool const isPrimary = primary == context;
    CUDA_DRIVER_CHECK(cuDevicePrimaryCtxRelease(device));
    check::check(isPrimary, "Phase execution must use the device CUDA primary context.");
}

} // namespace

void preservePhaseBatchRowAffinity(
    std::vector<PhaseWorkItem>& batch, std::vector<uint64_t> const& previousRowRequestIds)
{
    if (batch.empty())
    {
        return;
    }
    auto const canonicalLess = [](PhaseWorkItem const& left, PhaseWorkItem const& right) {
        int32_t constexpr kCONTEXT_BUCKET = 128;
        int32_t const leftBucket = left.tokenCount / kCONTEXT_BUCKET;
        int32_t const rightBucket = right.tokenCount / kCONTEXT_BUCKET;
        return leftBucket != rightBucket ? leftBucket < rightBucket : left.requestId < right.requestId;
    };
    if (previousRowRequestIds.empty())
    {
        std::stable_sort(batch.begin(), batch.end(), canonicalLess);
        return;
    }
    std::unordered_map<uint64_t, size_t> selectedRows;
    selectedRows.reserve(batch.size());
    for (size_t index{}; index < batch.size(); ++index)
    {
        selectedRows.emplace(batch[index].requestId, index);
    }
    std::vector<PhaseWorkItem> ordered(batch.size());
    std::vector<uint8_t> assigned(batch.size());
    std::vector<uint8_t> consumed(batch.size());
    size_t const retainedRows = std::min(batch.size(), previousRowRequestIds.size());
    for (size_t row{}; row < retainedRows; ++row)
    {
        auto const selected = selectedRows.find(previousRowRequestIds[row]);
        if (selected == selectedRows.end())
        {
            continue;
        }
        ordered[row] = batch[selected->second];
        assigned[row] = 1U;
        consumed[selected->second] = 1U;
    }
    std::vector<size_t> canonicalSources;
    canonicalSources.reserve(batch.size());
    for (size_t index{}; index < batch.size(); ++index)
    {
        if (consumed[index] == 0U)
        {
            canonicalSources.push_back(index);
        }
    }
    std::stable_sort(canonicalSources.begin(), canonicalSources.end(),
        [&](size_t left, size_t right) { return canonicalLess(batch[left], batch[right]); });
    size_t source{};
    for (size_t row{}; row < ordered.size(); ++row)
    {
        if (assigned[row] != 0U)
        {
            continue;
        }
        ordered[row] = batch[canonicalSources[source++]];
    }
    batch = std::move(ordered);
}

PhaseExecutionSafetyContract PhaseExecutionSafetyContract::shared(void const* tensorRTExecutionContext) noexcept
{
    PhaseExecutionSafetyContract result;
    result.prefill.tensorRTExecutionContext = tensorRTExecutionContext;
    result.decode.tensorRTExecutionContext = tensorRTExecutionContext;
    return result;
}

PhaseExecutionSafetyContract PhaseExecutionSafetyContract::independent(
    PhaseExecutionResourceIdentity prefill, PhaseExecutionResourceIdentity decode) noexcept
{
    return PhaseExecutionSafetyContract{prefill, decode};
}

bool PhaseExecutionSafetyContract::provesIndependentResources() const noexcept
{
    return prefill.tensorRTExecutionContext != nullptr && decode.tensorRTExecutionContext != nullptr
        && prefill.workspace != nullptr && decode.workspace != nullptr && prefill.ioBuffers != nullptr
        && decode.ioBuffers != nullptr && prefill.tensorRTExecutionContext != decode.tensorRTExecutionContext
        && prefill.workspace != decode.workspace && prefill.ioBuffers != decode.ioBuffers;
}

void PhaseExecutionSafetyContract::validate(PhaseTensorRTContextMode mode) const
{
    if (mode == PhaseTensorRTContextMode::kIndependentConcurrent)
    {
        check::check(provesIndependentResources(),
            "Concurrent phase execution requires distinct non-null TensorRT context, workspace, and I/O identities.");
        return;
    }
    if (prefill.tensorRTExecutionContext != nullptr || decode.tensorRTExecutionContext != nullptr)
    {
        check::check(prefill.tensorRTExecutionContext != nullptr
                && prefill.tensorRTExecutionContext == decode.tensorRTExecutionContext,
            "Shared TensorRT context mode requires one identical IExecutionContext identity.");
    }
}

PhaseDispatchWorker::PhaseDispatchWorker(PhaseQueueScheduler& scheduler, PhaseDispatchWorkerCallbacks callbacks,
    cudaStream_t prefillStream, cudaStream_t decodeStream, PhaseTensorRTContextMode executionMode,
    PhaseExecutionSafetyContract safetyContract)
    : mScheduler(scheduler)
    , mCallbacks(std::move(callbacks))
    , mPrefillStream(prefillStream)
    , mDecodeStream(decodeStream)
    , mExecutionMode(executionMode)
    , mSafetyContract(safetyContract)
{
    mSafetyContract.validate(mExecutionMode);
    CUcontext const prefillCudaContext = getStreamCudaContext(mPrefillStream);
    CUcontext const decodeCudaContext = getStreamCudaContext(mDecodeStream);
    check::check(prefillCudaContext == decodeCudaContext, "Prefill and decode streams must share one CUDA context.");
    if (mExecutionMode == PhaseTensorRTContextMode::kIndependentConcurrent)
    {
        check::check(mPrefillStream != mDecodeStream, "Independent TensorRT contexts require distinct CUDA streams.");
    }
    validatePrimaryCudaContext(prefillCudaContext);
    mCudaContext = prefillCudaContext;
    check::check(static_cast<bool>(mCallbacks.enqueuePrefill), "Prefill enqueue callback is required.");
    check::check(static_cast<bool>(mCallbacks.enqueueDecode), "Decode enqueue callback is required.");
    check::check(static_cast<bool>(mCallbacks.completePrefill), "Prefill completion callback is required.");
    check::check(static_cast<bool>(mCallbacks.completeDecode), "Decode completion callback is required.");
    CUDA_CHECK(cudaEventCreate(&mDispatchStart));
    CUDA_CHECK(cudaEventCreate(&mAugmentationStart));
    CUDA_CHECK(cudaEventCreate(&mPrefillStart));
    CUDA_CHECK(cudaEventCreate(&mPrefillDone));
    CUDA_CHECK(cudaEventCreate(&mDecodeStart));
    CUDA_CHECK(cudaEventCreate(&mDecodeDone));
}

PhaseDispatchWorker::~PhaseDispatchWorker() noexcept
{
    static_cast<void>(cudaEventDestroy(mDispatchStart));
    static_cast<void>(cudaEventDestroy(mAugmentationStart));
    static_cast<void>(cudaEventDestroy(mPrefillStart));
    static_cast<void>(cudaEventDestroy(mPrefillDone));
    static_cast<void>(cudaEventDestroy(mDecodeStart));
    static_cast<void>(cudaEventDestroy(mDecodeDone));
}

void PhaseDispatchWorker::setDirectionalInjectionControl(PhaseDirectionalInjectionControl control)
{
    check::check(!mBusy, "Phase directional injection control cannot change while work is in flight.");
    check::check(!control.enabled() || mExecutionMode == PhaseTensorRTContextMode::kIndependentConcurrent,
        "Directional injection requires independent TensorRT execution contexts.");
    check::check(control.targetFraction >= 0.0 && control.targetFraction <= 1.0,
        "Directional injection target fraction must be within [0, 1].");
    mDirectionalInjection = control;
    mDirectionalInjectionConsumed = false;
}

cudaStream_t PhaseDispatchWorker::phaseStream(PhaseUnifiedPhase phase) const noexcept
{
    return phase == PhaseUnifiedPhase::kPrefill ? mPrefillStream
        : phase == PhaseUnifiedPhase::kDecode   ? mDecodeStream
                                                : nullptr;
}

cudaEvent_t PhaseDispatchWorker::phaseStartEvent(PhaseUnifiedPhase phase) const noexcept
{
    return phase == PhaseUnifiedPhase::kPrefill ? mPrefillStart
        : phase == PhaseUnifiedPhase::kDecode   ? mDecodeStart
                                                : nullptr;
}

void PhaseDispatchWorker::setNextDispatchPreamble(PhaseUnifiedPhase phase, std::function<void(cudaStream_t)> preamble)
{
    check::check(!mBusy, "A phase dispatch preamble can only be installed while the worker is idle");
    check::check(phase == PhaseUnifiedPhase::kPrefill || phase == PhaseUnifiedPhase::kDecode,
        "A phase dispatch preamble only supports prefill or decode");
    std::function<void(cudaStream_t)>& target
        = phase == PhaseUnifiedPhase::kPrefill ? mNextPrefillDispatchPreamble : mNextDecodeDispatchPreamble;
    check::check(!target, "A phase dispatch preamble is already installed");
    target = std::move(preamble);
}

bool PhaseDispatchWorker::dispatchNext()
{
    check::check(!mBusy, "Cannot dispatch while another phase plan is in flight.");
    mInFlight = mScheduler.next();
    if (mInFlight.kind == PhaseDispatchKind::kNone)
    {
        return false;
    }

    mHasPrefill = !mInFlight.prefillBatch.empty();
    mHasDecode = !mInFlight.decodeBatch.empty();
    if (mHasPrefill)
    {
        preservePhaseBatchRowAffinity(mInFlight.prefillBatch, {});
    }
    if (mHasDecode)
    {
        preservePhaseBatchRowAffinity(mInFlight.decodeBatch, mPreviousDecodeRowRequestIds);
        mPreviousDecodeRowRequestIds.clear();
        mPreviousDecodeRowRequestIds.reserve(mInFlight.decodeBatch.size());
        for (PhaseWorkItem const& item : mInFlight.decodeBatch)
        {
            mPreviousDecodeRowRequestIds.push_back(item.requestId);
        }
    }
    mCurrentMetrics = PhaseDispatchMetrics{};
    mPrefillEnqueueHostNs = 0U;
    mDecodeEnqueueHostNs = 0U;
    mPrefillPlanId = 0U;
    mDecodePlanId = 0U;
    mPrefillActionId = 0U;
    mDecodeActionId = 0U;
    PhaseExecutionSet launched{PhaseExecutionSet::kNone};
    if (mHasPrefill)
    {
        launched = launched | PhaseExecutionSet::kPrefill;
    }
    if (mHasDecode)
    {
        launched = launched | PhaseExecutionSet::kDecode;
    }
    bool const actionFidelity = !mInFlight.globalDecisionApplied
        || (phaseExecutionSetIsSubset(launched, mInFlight.globalAllowedOutstanding)
            && launched == phaseExecutionSetForAction(mInFlight.globalSelectedAction.kind));
    check::check(actionFidelity, "Global P/D plan does not match the launched phase set");
    mCurrentMetrics.dispatchIndex = mDispatchCount + 1;
    mCurrentMetrics.kind = mInFlight.kind;
    mCurrentMetrics.hostSchedulerDecisionUs = mInFlight.hostSchedulerDecisionUs;
    mCurrentMetrics.hostDispatchStartNs = phaseTimelineNowNs();
    mCurrentMetrics.prefillRequestIds.reserve(mInFlight.prefillBatch.size());
    for (PhaseWorkItem const& item : mInFlight.prefillBatch)
    {
        mCurrentMetrics.prefillRequestIds.push_back(item.requestId);
    }
    mCurrentMetrics.decodeRequestIds.reserve(mInFlight.decodeBatch.size());
    for (PhaseWorkItem const& item : mInFlight.decodeBatch)
    {
        mCurrentMetrics.decodeRequestIds.push_back(item.requestId);
    }
    if (!mInFlight.prefillBatch.empty())
    {
        mCurrentMetrics.prefillClass = mInFlight.prefillBatch.front().prefillClass;
    }
    mCurrentMetrics.prefillBatchSize = static_cast<int32_t>(mInFlight.prefillBatch.size());
    mCurrentMetrics.decodeBatchSize = static_cast<int32_t>(mInFlight.decodeBatch.size());
    mCurrentMetrics.predictedPrefillGpuMs = mInFlight.predictedPrefillGpuMs;
    mCurrentMetrics.predictedDecodeSlowdownMs = mInFlight.predictedDecodeSlowdownMs;
    mCurrentMetrics.predictedDecodeDebtUs = mInFlight.predictedDecodeDebtUs;
    mCurrentMetrics.consecutiveOverlapBatches = mInFlight.consecutiveOverlapBatches;
    mCurrentMetrics.prefillDeferredForTpot = mInFlight.prefillDeferredForTpot;
    mCurrentMetrics.prefillCostCoverageMiss = mInFlight.prefillCostCoverageMiss;
    mCurrentMetrics.overlapEvaluatedByCost = mInFlight.overlapEvaluatedByCost;
    mCurrentMetrics.latencySafeFallback = mInFlight.latencySafeFallback;
    mCurrentMetrics.prefillCostLookupRows = mInFlight.prefillCostLookupRows;
    mCurrentMetrics.prefillCostLookupChunkLength = mInFlight.prefillCostLookupChunkLength;
    mCurrentMetrics.prefillCostLookupMaxPastKVLength = mInFlight.prefillCostLookupMaxPastKVLength;
    mCurrentMetrics.prefillShapeCandidatesEvaluated = mInFlight.prefillShapeCandidatesEvaluated;
    mCurrentMetrics.predictedPrefillShapeScore = mInFlight.predictedPrefillShapeScore;
    mCurrentMetrics.prefillShapeDrainMode = mInFlight.prefillShapeDrainMode;
    mCurrentMetrics.adaptiveChunkDecodeQueuePressure = mInFlight.adaptiveChunkDecodeQueuePressure;
    mCurrentMetrics.adaptiveChunkObservedTpotPressure = mInFlight.adaptiveChunkObservedTpotPressure;
    mCurrentMetrics.adaptiveChunkCombinedPressure = mInFlight.adaptiveChunkCombinedPressure;
    mCurrentMetrics.drainPreference = mInFlight.drainPreference;
    mCurrentMetrics.drainPreferenceApplied = mInFlight.drainPreferenceApplied;
    mCurrentMetrics.plannedDecodeBatchSize = mInFlight.plannedDecodeBatchSize;
    mCurrentMetrics.plannedDecodeContextTokens = mInFlight.plannedDecodeContextTokens;
    mCurrentMetrics.plannedDecodeMaxContextLength = mInFlight.plannedDecodeMaxContextLength;
    mCurrentMetrics.predictedDecodeReplacementRows = mInFlight.predictedDecodeReplacementRows;
    mCurrentMetrics.externalEncoderActive = mInFlight.externalEncoderActive;
    mCurrentMetrics.concurrentPrefillActive = mInFlight.concurrentPrefillActive;
    mCurrentMetrics.predictedDecodeDrainGpuMs = mInFlight.predictedDecodeDrainGpuMs;
    mCurrentMetrics.predictedDecodeDrainTurns = mInFlight.predictedDecodeDrainTurns;
    mCurrentMetrics.prefillCohortSize = mInFlight.prefillCohortSize;
    mCurrentMetrics.globalDecisionEvaluated = mInFlight.globalDecisionEvaluated;
    mCurrentMetrics.globalDecisionApplied = mInFlight.globalDecisionApplied;
    mCurrentMetrics.globalSafeProbe = mInFlight.globalSafeProbe;
    mCurrentMetrics.globalCandidateId = mInFlight.globalCandidateId;
    mCurrentMetrics.globalCandidateParity = mInFlight.globalCandidateParity;
    mCurrentMetrics.globalPlanId = mInFlight.globalPlanId;
    mCurrentMetrics.globalSnapshotEpoch = mInFlight.globalSnapshotEpoch;
    mCurrentMetrics.globalAllowedOutstanding = mInFlight.globalAllowedOutstanding;
    mCurrentMetrics.globalLaunched = launched;
    mCurrentMetrics.globalActionFidelity = actionFidelity;
    mCurrentMetrics.globalSelectedAction = mInFlight.globalSelectedAction;
    mCurrentMetrics.globalDecisionReason = mInFlight.globalDecisionReason;
    mCurrentMetrics.globalPredictedViolationUs = mInFlight.globalPredictedViolationUs;
    mCurrentMetrics.globalServiceCompression = mInFlight.globalServiceCompression;
    mCurrentMetrics.globalReferenceWorkMs = mInFlight.globalReferenceWorkMs;
    mCurrentMetrics.contextualPdFeatures = mInFlight.contextualPdFeatures;
    mCurrentMetrics.contextualPdFeatureValid = mInFlight.contextualPdFeatureValid;
    mCurrentMetrics.contextualCompletionFeatures = mInFlight.contextualCompletionFeatures;
    mCurrentMetrics.contextualCompletionFeatureValid = mInFlight.contextualCompletionFeatureValid;
    mCurrentMetrics.contextualPdExploration = mInFlight.contextualPdExploration;
    mCurrentMetrics.contextualPdMean = mInFlight.contextualPdMean;
    mCurrentMetrics.contextualPdUncertainty = mInFlight.contextualPdUncertainty;
    mCurrentMetrics.contextualPdLowerConfidenceBound = mInFlight.contextualPdLowerConfidenceBound;
    mCurrentMetrics.contextualCompletionIncumbentReferenceUs = mInFlight.contextualCompletionIncumbentReferenceUs;
    mCurrentMetrics.contextualCompletionNewcomerReferenceUs = mInFlight.contextualCompletionNewcomerReferenceUs;
    mCurrentMetrics.contextualCompletionMinimumSlackUs = mInFlight.contextualCompletionMinimumSlackUs;
    mCurrentMetrics.decodeCohortSize = static_cast<int32_t>(mScheduler.decodeCohortSize());
    int64_t prefillPastKVSum{};
    mCurrentMetrics.prefillPastKVMin = mInFlight.prefillBatch.empty() ? 0 : std::numeric_limits<int32_t>::max();
    mCurrentMetrics.prefillMinTtftSlackUs = mInFlight.prefillBatch.empty() ? 0.0 : std::numeric_limits<double>::max();
    auto const now = std::chrono::steady_clock::now();
    for (PhaseWorkItem const& item : mInFlight.prefillBatch)
    {
        mCurrentMetrics.prefillTokens += item.tokenCount;
        bool const initial = item.tokenOffset == 0;
        bool const final = item.tokenOffset + item.tokenCount == item.promptTokenCount;
        mCurrentMetrics.prefillInitialRows += initial ? 1 : 0;
        mCurrentMetrics.prefillContinuationRows += initial ? 0 : 1;
        mCurrentMetrics.prefillFinalRows += final ? 1 : 0;
        mCurrentMetrics.prefillPastKVMin = std::min(mCurrentMetrics.prefillPastKVMin, item.tokenOffset);
        mCurrentMetrics.prefillPastKVMax = std::max(mCurrentMetrics.prefillPastKVMax, item.tokenOffset);
        prefillPastKVSum += item.tokenOffset;
        mCurrentMetrics.prefillRemainingTokens += item.promptTokenCount - item.tokenOffset - item.tokenCount;
        double const requestAgeUs
            = std::chrono::duration<double, std::micro>(now - item.scheduling.submittedAt).count();
        double const targetUs = item.scheduling.ttftTargetUs;
        mCurrentMetrics.prefillOldestRequestAgeUs = std::max(mCurrentMetrics.prefillOldestRequestAgeUs, requestAgeUs);
        if (targetUs > 0.0)
        {
            mCurrentMetrics.prefillMinTtftSlackUs
                = std::min(mCurrentMetrics.prefillMinTtftSlackUs, targetUs - requestAgeUs);
        }
    }
    if (mCurrentMetrics.prefillBatchSize > 0)
    {
        int32_t const paddedChunkLength = std::max_element(mInFlight.prefillBatch.begin(), mInFlight.prefillBatch.end(),
            [](PhaseWorkItem const& lhs, PhaseWorkItem const& rhs) {
                return lhs.tokenCount < rhs.tokenCount;
            })->tokenCount;
        mCurrentMetrics.prefillPaddedTokens = paddedChunkLength * mCurrentMetrics.prefillBatchSize;
        mCurrentMetrics.prefillPaddingTokens = mCurrentMetrics.prefillPaddedTokens - mCurrentMetrics.prefillTokens;
        mCurrentMetrics.prefillPackingEfficiency = static_cast<float>(mCurrentMetrics.prefillTokens)
            / static_cast<float>(mCurrentMetrics.prefillPaddedTokens);
        mCurrentMetrics.prefillPastKVMean = static_cast<int32_t>(prefillPastKVSum / mCurrentMetrics.prefillBatchSize);
        mCurrentMetrics.prefillPastKVSpread = mCurrentMetrics.prefillPastKVMax - mCurrentMetrics.prefillPastKVMin;
        if (mCurrentMetrics.prefillMinTtftSlackUs == std::numeric_limits<double>::max())
        {
            mCurrentMetrics.prefillMinTtftSlackUs = 0.0;
        }
    }
    mCurrentMetrics.decodeTokens = static_cast<int32_t>(mInFlight.decodeBatch.size());
    for (PhaseWorkItem const& item : mInFlight.decodeBatch)
    {
        mCurrentMetrics.decodeContextTokens += item.tokenCount;
    }
    mCurrentMetrics.prefillQueueWaitUs = mInFlight.prefillQueueWaitUs;
    mCurrentMetrics.decodeQueueWaitUs = mInFlight.decodeQueueWaitUs;
    if (mCallbacks.onDispatch)
    {
        mCallbacks.onDispatch(mCurrentMetrics);
    }
    CUDA_CHECK(cudaEventRecord(mDispatchStart, mPrefillStream));
    auto enqueuePrefill = [&] {
        if (mNextPrefillDispatchPreamble)
        {
            std::function<void(cudaStream_t)> preamble = std::move(mNextPrefillDispatchPreamble);
            preamble(mPrefillStream);
        }
        mPrefillPlanId = mInFlight.globalPlanId > 0U ? mInFlight.globalPlanId : mCurrentMetrics.dispatchIndex;
        mPrefillActionId = mInFlight.globalCandidateId > 0U ? mInFlight.globalCandidateId : mPrefillPlanId;
        mPrefillEnqueueHostNs = phaseTimelineNowNs();
        recordTimeline(mInFlight.prefillBatch, PhaseTimelineStage::kPrefillStart);
        CUDA_CHECK(cudaEventRecord(mPrefillStart, mPrefillStream));
        mCurrentMetrics.prefillHostExecution = enqueueActivity(PhaseActivityKind::kPrefill, "prefill_dispatch",
            mInFlight.prefillBatch, mCallbacks.enqueuePrefill, mPrefillStream);
        CUDA_CHECK(cudaEventRecord(mPrefillDone, mPrefillStream));
    };
    auto enqueueDecode = [&] {
        if (mNextDecodeDispatchPreamble)
        {
            std::function<void(cudaStream_t)> preamble = std::move(mNextDecodeDispatchPreamble);
            preamble(mDecodeStream);
        }
        mDecodePlanId = mInFlight.globalPlanId > 0U ? mInFlight.globalPlanId : mCurrentMetrics.dispatchIndex;
        mDecodeActionId = mInFlight.globalCandidateId > 0U ? mInFlight.globalCandidateId : mDecodePlanId;
        bool const serializeSharedContext
            = mHasPrefill && mExecutionMode == PhaseTensorRTContextMode::kSharedSerialized;
        if (serializeSharedContext)
        {
            // A stream wait does not protect host-side TensorRT context state.
            // Defer prepare/execute until the prefill event has completed.
            mDecodeDeferred = true;
        }
        else
        {
            CUDA_CHECK(cudaStreamWaitEvent(mDecodeStream, mDispatchStart));
            mDecodeEnqueueHostNs = phaseTimelineNowNs();
            recordTimeline(mInFlight.decodeBatch, PhaseTimelineStage::kDecodeStart);
            CUDA_CHECK(cudaEventRecord(mDecodeStart, mDecodeStream));
            mCurrentMetrics.decodeHostExecution = enqueueActivity(PhaseActivityKind::kDecode, "decode_dispatch",
                mInFlight.decodeBatch, mCallbacks.enqueueDecode, mDecodeStream);
            CUDA_CHECK(cudaEventRecord(mDecodeDone, mDecodeStream));
        }
    };
    bool const decodeFirst = mHasPrefill && mHasDecode
        && mExecutionMode == PhaseTensorRTContextMode::kIndependentConcurrent
        && mDirectionalInjection.direction == PhaseUnifiedActionDirection::kDecodeToPrefill;
    if (decodeFirst)
    {
        enqueueDecode();
        if (mDirectionalInjection.enabled()
            && mDirectionalInjection.direction == PhaseUnifiedActionDirection::kDecodeToPrefill
            && !mDirectionalInjectionConsumed)
        {
            mDirectionalCudaGate.enqueue(mPrefillStream, mDecodeStart,
                remainingDirectionalDelayUs(mDirectionalInjection.requestedDelayUs, mDecodeEnqueueHostNs));
            mDirectionalInjectionConsumed = true;
        }
        enqueuePrefill();
    }
    else
    {
        if (mHasPrefill)
        {
            enqueuePrefill();
        }
        if (mHasPrefill && mHasDecode)
        {
            if (mDirectionalInjection.enabled()
                && mDirectionalInjection.direction == PhaseUnifiedActionDirection::kPrefillToDecode
                && !mDirectionalInjectionConsumed)
            {
                mDirectionalCudaGate.enqueue(mDecodeStream, mPrefillStart,
                    remainingDirectionalDelayUs(mDirectionalInjection.requestedDelayUs, mPrefillEnqueueHostNs));
                mDirectionalInjectionConsumed = true;
            }
        }
        if (mHasDecode)
        {
            enqueueDecode();
        }
    }
    mCurrentMetrics.hostSubmissionEndNs = phaseTimelineNowNs();
    mBusy = true;
    ++mDispatchCount;
    return true;
}

bool PhaseDispatchWorker::augmentNext(PhaseGlobalActionCandidate missingPhase, PhaseGlobalActionCandidate aggregate,
    uint64_t planId, uint64_t snapshotEpoch)
{
    if (!mBusy || mExecutionMode != PhaseTensorRTContextMode::kIndependentConcurrent || mHasPrefill == mHasDecode
        || mDecodeDeferred)
    {
        return false;
    }
    check::check(aggregate.key.kind == PhaseGlobalActionKind::kPrefillDecode,
        "Residual P/D augmentation requires a P+D aggregate action.");
    PhaseGlobalActionKind const expectedMissing
        = mHasPrefill ? PhaseGlobalActionKind::kDecode : PhaseGlobalActionKind::kPrefill;
    check::check(missingPhase.key.kind == expectedMissing, "Residual P/D augmentation selected the active phase.");
    PhaseGlobalResidualAnchor const observedAnchor
        = mHasPrefill ? PhaseGlobalResidualAnchor::kPrefill : PhaseGlobalResidualAnchor::kDecode;
    check::check(aggregate.key.residualAugmentation && aggregate.key.residualAnchor == observedAnchor,
        "Residual P/D cost key does not match the phase that is already executing.");

    mScheduler.setNextGlobalAction(std::move(missingPhase), planId, snapshotEpoch);
    PhaseDispatchPlan additional = mScheduler.next();
    uint64_t const effectivePlanId = additional.globalPlanId;
    uint64_t const effectiveSnapshotEpoch = additional.globalSnapshotEpoch;
    bool const addsPrefill = !additional.prefillBatch.empty() && additional.decodeBatch.empty();
    bool const addsDecode = additional.prefillBatch.empty() && !additional.decodeBatch.empty();
    check::check((mHasPrefill && addsDecode) || (mHasDecode && addsPrefill),
        "Residual P/D augmentation did not materialize exactly the idle phase.");

    if (addsPrefill)
    {
        if (mDirectionalInjection.enabled()
            && mDirectionalInjection.direction == PhaseUnifiedActionDirection::kDecodeToPrefill
            && !mDirectionalInjectionConsumed)
        {
            mDirectionalCudaGate.enqueue(mPrefillStream, mDecodeStart,
                remainingDirectionalDelayUs(mDirectionalInjection.requestedDelayUs, mDecodeEnqueueHostNs));
            mDirectionalInjectionConsumed = true;
        }
        mInFlight.prefillBatch = std::move(additional.prefillBatch);
        mHasPrefill = true;
        mPrefillPlanId = effectivePlanId;
        mPrefillActionId = aggregate.candidateId;
        mPrefillEnqueueHostNs = phaseTimelineNowNs();
        CUDA_CHECK(cudaEventRecord(mAugmentationStart, mPrefillStream));
        recordTimeline(mInFlight.prefillBatch, PhaseTimelineStage::kPrefillStart);
        CUDA_CHECK(cudaEventRecord(mPrefillStart, mPrefillStream));
        mCurrentMetrics.prefillHostExecution = enqueueActivity(PhaseActivityKind::kPrefill, "prefill_residual_dispatch",
            mInFlight.prefillBatch, mCallbacks.enqueuePrefill, mPrefillStream);
        CUDA_CHECK(cudaEventRecord(mPrefillDone, mPrefillStream));
    }
    else
    {
        if (mDirectionalInjection.enabled()
            && mDirectionalInjection.direction == PhaseUnifiedActionDirection::kPrefillToDecode
            && !mDirectionalInjectionConsumed)
        {
            mDirectionalCudaGate.enqueue(mDecodeStream, mPrefillStart,
                remainingDirectionalDelayUs(mDirectionalInjection.requestedDelayUs, mPrefillEnqueueHostNs));
            mDirectionalInjectionConsumed = true;
        }
        mInFlight.decodeBatch = std::move(additional.decodeBatch);
        preservePhaseBatchRowAffinity(mInFlight.decodeBatch, mPreviousDecodeRowRequestIds);
        mPreviousDecodeRowRequestIds.clear();
        mPreviousDecodeRowRequestIds.reserve(mInFlight.decodeBatch.size());
        for (PhaseWorkItem const& item : mInFlight.decodeBatch)
        {
            mPreviousDecodeRowRequestIds.push_back(item.requestId);
        }
        mHasDecode = true;
        mDecodePlanId = effectivePlanId;
        mDecodeActionId = aggregate.candidateId;
        mDecodeEnqueueHostNs = phaseTimelineNowNs();
        CUDA_CHECK(cudaEventRecord(mAugmentationStart, mDecodeStream));
        recordTimeline(mInFlight.decodeBatch, PhaseTimelineStage::kDecodeStart);
        CUDA_CHECK(cudaEventRecord(mDecodeStart, mDecodeStream));
        mCurrentMetrics.decodeHostExecution = enqueueActivity(PhaseActivityKind::kDecode, "decode_residual_dispatch",
            mInFlight.decodeBatch, mCallbacks.enqueueDecode, mDecodeStream);
        CUDA_CHECK(cudaEventRecord(mDecodeDone, mDecodeStream));
    }
    mCurrentMetrics.hostSubmissionEndNs = phaseTimelineNowNs();
    mCurrentMetrics.globalObservedResidualAnchor = observedAnchor;
    mResidualAugmentation = true;
    mergeAugmentedMetrics(additional, aggregate, effectivePlanId, effectiveSnapshotEpoch);
    return true;
}

void PhaseDispatchWorker::mergeAugmentedMetrics(PhaseDispatchPlan const& additional,
    PhaseGlobalActionCandidate const& aggregate, uint64_t planId, uint64_t snapshotEpoch)
{
    mInFlight.kind = PhaseDispatchKind::kOverlap;
    mInFlight.globalDecisionEvaluated = true;
    mInFlight.globalDecisionApplied = true;
    mInFlight.globalSafeProbe
        = aggregate.calibrationProbe || (aggregate.safeProbeEligible && !aggregate.overlapCostKnown);
    mInFlight.globalCandidateId = aggregate.candidateId;
    mInFlight.globalPlanId = planId;
    mInFlight.globalSnapshotEpoch = snapshotEpoch;
    mInFlight.globalAllowedOutstanding = PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode;
    mInFlight.globalActionFidelity = true;
    mInFlight.globalSelectedAction = aggregate.key;
    mInFlight.globalCandidate = aggregate;
    mInFlight.globalReferenceWorkMs = aggregate.referenceWorkUs / 1000.0;
    mInFlight.contextualPdFeatures = aggregate.contextualPdFeatures;
    mInFlight.contextualPdFeatureValid = aggregate.contextualPdFeatureValid;
    mInFlight.contextualCompletionFeatures = aggregate.contextualCompletionFeatures;
    mInFlight.contextualCompletionFeatureValid = aggregate.contextualCompletionFeatureValid;
    mInFlight.contextualPdExploration = aggregate.contextualPdExploration;
    mInFlight.contextualPdMean = aggregate.contextualPdMean;
    mInFlight.contextualPdUncertainty = aggregate.contextualPdUncertainty;
    mInFlight.contextualPdLowerConfidenceBound = aggregate.contextualPdLowerConfidenceBound;
    mInFlight.contextualCompletionIncumbentReferenceUs = aggregate.contextualCompletionIncumbentReferenceUs;
    mInFlight.contextualCompletionNewcomerReferenceUs = aggregate.contextualCompletionNewcomerReferenceUs;
    mInFlight.contextualCompletionMinimumSlackUs = aggregate.contextualCompletionMinimumSlackUs;
    mInFlight.globalServiceCompression
        = aggregate.referenceWorkUs / std::max(aggregate.predictedMakespanUs, std::numeric_limits<double>::epsilon());
    mInFlight.concurrentPrefillActive = true;
    mInFlight.prefillQueueWaitUs = std::max(mInFlight.prefillQueueWaitUs, additional.prefillQueueWaitUs);
    mInFlight.decodeQueueWaitUs = std::max(mInFlight.decodeQueueWaitUs, additional.decodeQueueWaitUs);
    if (!additional.prefillBatch.empty())
    {
        mInFlight.predictedPrefillGpuMs = additional.predictedPrefillGpuMs;
        mInFlight.prefillCostLookupRows = additional.prefillCostLookupRows;
        mInFlight.prefillCostLookupChunkLength = additional.prefillCostLookupChunkLength;
        mInFlight.prefillCostLookupMaxPastKVLength = additional.prefillCostLookupMaxPastKVLength;
    }
    if (!additional.decodeBatch.empty())
    {
        mInFlight.plannedDecodeBatchSize = additional.plannedDecodeBatchSize;
        mInFlight.plannedDecodeContextTokens = additional.plannedDecodeContextTokens;
        mInFlight.plannedDecodeMaxContextLength = additional.plannedDecodeMaxContextLength;
        mInFlight.predictedDecodeReplacementRows = additional.predictedDecodeReplacementRows;
    }

    mCurrentMetrics.kind = PhaseDispatchKind::kOverlap;
    mCurrentMetrics.prefillRequestIds.clear();
    mCurrentMetrics.decodeRequestIds.clear();
    for (PhaseWorkItem const& item : mInFlight.prefillBatch)
    {
        mCurrentMetrics.prefillRequestIds.push_back(item.requestId);
    }
    for (PhaseWorkItem const& item : mInFlight.decodeBatch)
    {
        mCurrentMetrics.decodeRequestIds.push_back(item.requestId);
    }
    mCurrentMetrics.prefillBatchSize = static_cast<int32_t>(mInFlight.prefillBatch.size());
    mCurrentMetrics.decodeBatchSize = static_cast<int32_t>(mInFlight.decodeBatch.size());
    mCurrentMetrics.prefillClass = mInFlight.prefillBatch.front().prefillClass;
    mCurrentMetrics.prefillTokens = 0;
    mCurrentMetrics.prefillPaddedTokens = 0;
    mCurrentMetrics.prefillPaddingTokens = 0;
    mCurrentMetrics.prefillPackingEfficiency = 0.0F;
    mCurrentMetrics.prefillInitialRows = 0;
    mCurrentMetrics.prefillContinuationRows = 0;
    mCurrentMetrics.prefillFinalRows = 0;
    mCurrentMetrics.prefillPastKVMin = std::numeric_limits<int32_t>::max();
    mCurrentMetrics.prefillPastKVMean = 0;
    mCurrentMetrics.prefillPastKVMax = 0;
    mCurrentMetrics.prefillRemainingTokens = 0;
    mCurrentMetrics.prefillOldestRequestAgeUs = 0.0;
    mCurrentMetrics.prefillMinTtftSlackUs = std::numeric_limits<double>::max();
    int64_t prefillPastKVSum{};
    auto const now = std::chrono::steady_clock::now();
    for (PhaseWorkItem const& item : mInFlight.prefillBatch)
    {
        mCurrentMetrics.prefillTokens += item.tokenCount;
        bool const initial = item.tokenOffset == 0;
        bool const final = item.tokenOffset + item.tokenCount == item.promptTokenCount;
        mCurrentMetrics.prefillInitialRows += initial ? 1 : 0;
        mCurrentMetrics.prefillContinuationRows += initial ? 0 : 1;
        mCurrentMetrics.prefillFinalRows += final ? 1 : 0;
        mCurrentMetrics.prefillPastKVMin = std::min(mCurrentMetrics.prefillPastKVMin, item.tokenOffset);
        mCurrentMetrics.prefillPastKVMax = std::max(mCurrentMetrics.prefillPastKVMax, item.tokenOffset);
        prefillPastKVSum += item.tokenOffset;
        mCurrentMetrics.prefillRemainingTokens += item.promptTokenCount - item.tokenOffset - item.tokenCount;
        mCurrentMetrics.prefillPaddedTokens
            = std::max(mCurrentMetrics.prefillPaddedTokens, item.tokenCount * mCurrentMetrics.prefillBatchSize);
        double const requestAgeUs
            = std::chrono::duration<double, std::micro>(now - item.scheduling.submittedAt).count();
        mCurrentMetrics.prefillOldestRequestAgeUs = std::max(mCurrentMetrics.prefillOldestRequestAgeUs, requestAgeUs);
        if (item.scheduling.ttftTargetUs > 0.0)
        {
            mCurrentMetrics.prefillMinTtftSlackUs
                = std::min(mCurrentMetrics.prefillMinTtftSlackUs, item.scheduling.ttftTargetUs - requestAgeUs);
        }
    }
    mCurrentMetrics.prefillPaddingTokens = mCurrentMetrics.prefillPaddedTokens - mCurrentMetrics.prefillTokens;
    mCurrentMetrics.prefillPackingEfficiency
        = static_cast<float>(mCurrentMetrics.prefillTokens) / static_cast<float>(mCurrentMetrics.prefillPaddedTokens);
    mCurrentMetrics.prefillPastKVMean = static_cast<int32_t>(prefillPastKVSum / mCurrentMetrics.prefillBatchSize);
    mCurrentMetrics.prefillPastKVSpread = mCurrentMetrics.prefillPastKVMax - mCurrentMetrics.prefillPastKVMin;
    if (mCurrentMetrics.prefillMinTtftSlackUs == std::numeric_limits<double>::max())
    {
        mCurrentMetrics.prefillMinTtftSlackUs = 0.0;
    }
    mCurrentMetrics.decodeContextTokens = 0;
    mCurrentMetrics.plannedDecodeMaxContextLength = 0;
    for (PhaseWorkItem const& item : mInFlight.decodeBatch)
    {
        mCurrentMetrics.decodeContextTokens += item.tokenCount;
        mCurrentMetrics.plannedDecodeMaxContextLength
            = std::max(mCurrentMetrics.plannedDecodeMaxContextLength, item.tokenCount);
    }
    mCurrentMetrics.decodeTokens = mCurrentMetrics.decodeBatchSize;
    mCurrentMetrics.concurrentPrefillActive = true;
    mCurrentMetrics.globalDecisionEvaluated = true;
    mCurrentMetrics.globalDecisionApplied = true;
    mCurrentMetrics.globalSafeProbe = mInFlight.globalSafeProbe;
    mCurrentMetrics.globalCandidateId = aggregate.candidateId;
    mCurrentMetrics.globalCandidateParity = true;
    mCurrentMetrics.globalPlanId = planId;
    mCurrentMetrics.globalSnapshotEpoch = snapshotEpoch;
    mCurrentMetrics.globalAllowedOutstanding = PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode;
    mCurrentMetrics.globalLaunched = PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode;
    mCurrentMetrics.globalActionFidelity = true;
    mCurrentMetrics.globalSelectedAction = aggregate.key;
    mCurrentMetrics.globalReferenceWorkMs = mInFlight.globalReferenceWorkMs;
    mCurrentMetrics.globalServiceCompression = mInFlight.globalServiceCompression;
    mCurrentMetrics.contextualPdFeatures = mInFlight.contextualPdFeatures;
    mCurrentMetrics.contextualPdFeatureValid = mInFlight.contextualPdFeatureValid;
    mCurrentMetrics.contextualCompletionFeatures = mInFlight.contextualCompletionFeatures;
    mCurrentMetrics.contextualCompletionFeatureValid = mInFlight.contextualCompletionFeatureValid;
    mCurrentMetrics.contextualPdExploration = mInFlight.contextualPdExploration;
    mCurrentMetrics.contextualPdMean = mInFlight.contextualPdMean;
    mCurrentMetrics.contextualPdUncertainty = mInFlight.contextualPdUncertainty;
    mCurrentMetrics.contextualPdLowerConfidenceBound = mInFlight.contextualPdLowerConfidenceBound;
    mCurrentMetrics.contextualCompletionIncumbentReferenceUs = mInFlight.contextualCompletionIncumbentReferenceUs;
    mCurrentMetrics.contextualCompletionNewcomerReferenceUs = mInFlight.contextualCompletionNewcomerReferenceUs;
    mCurrentMetrics.contextualCompletionMinimumSlackUs = mInFlight.contextualCompletionMinimumSlackUs;
}

bool PhaseDispatchWorker::eventReady(cudaEvent_t event) const
{
    cudaError_t const status = cudaEventQuery(event);
    if (status == cudaSuccess)
    {
        return true;
    }
    if (status == cudaErrorNotReady)
    {
        return false;
    }
    CUDA_CHECK(status);
    return false;
}

bool PhaseDispatchWorker::poll()
{
    if (!mBusy)
    {
        return false;
    }
    if (mDecodeDeferred)
    {
        if (!eventReady(mPrefillDone))
        {
            return false;
        }
        completePrefillInFlight();
        enqueueDeferredDecode();
        return false;
    }
    if ((mHasPrefill && !eventReady(mPrefillDone)) || (mHasDecode && !eventReady(mDecodeDone)))
    {
        return false;
    }
    completeInFlight();
    return true;
}

void PhaseDispatchWorker::wait()
{
    check::check(mBusy, "PhaseDispatchWorker has no in-flight plan to wait for.");
    if (mHasPrefill)
    {
        CUDA_CHECK(cudaEventSynchronize(mPrefillDone));
    }
    if (mDecodeDeferred)
    {
        completePrefillInFlight();
        enqueueDeferredDecode();
    }
    if (mHasDecode)
    {
        CUDA_CHECK(cudaEventSynchronize(mDecodeDone));
    }
    completeInFlight();
}

void PhaseDispatchWorker::enqueueDeferredDecode()
{
    check::check(mDecodeDeferred && mHasDecode, "No deferred decode batch is available.");
    mDecodeEnqueueHostNs = phaseTimelineNowNs();
    recordTimeline(mInFlight.decodeBatch, PhaseTimelineStage::kDecodeStart);
    CUDA_CHECK(cudaEventRecord(mDecodeStart, mDecodeStream));
    mCurrentMetrics.decodeHostExecution = enqueueActivity(
        PhaseActivityKind::kDecode, "decode_dispatch", mInFlight.decodeBatch, mCallbacks.enqueueDecode, mDecodeStream);
    CUDA_CHECK(cudaEventRecord(mDecodeDone, mDecodeStream));
    mCurrentMetrics.hostSubmissionEndNs = phaseTimelineNowNs();
    mDecodeDeferred = false;
}

PhaseHostExecutionTiming PhaseDispatchWorker::enqueueActivity(PhaseActivityKind kind, char const* name,
    std::vector<PhaseWorkItem> const& batch, PhaseEnqueueCallback const& callback, cudaStream_t stream)
{
    if (mActivityTimeline == nullptr)
    {
        return callback(batch, stream);
    }
    PhaseActivityTimelineRecorder::Token const token
        = mActivityTimeline->begin(kind, stream, name, mCurrentMetrics.dispatchIndex);
    try
    {
        PhaseHostExecutionTiming const timing = callback(batch, stream);
        mActivityTimeline->end(token, stream);
        return timing;
    }
    catch (...)
    {
        mActivityTimeline->cancel(token);
        throw;
    }
}

void PhaseDispatchWorker::completePrefillInFlight()
{
    if (!mHasPrefill)
    {
        return;
    }
    if (mCallbacks.completePrefillBatch)
    {
        mCallbacks.completePrefillBatch(mInFlight.prefillBatch);
    }
    recordTimeline(mInFlight.prefillBatch, PhaseTimelineStage::kPrefillDone);
    for (PhaseWorkItem const& item : mInFlight.prefillBatch)
    {
        PhasePrefillCompletion const completion = mCallbacks.completePrefill(item);
        mScheduler.completePrefill(item, completion.resultingKVLength, completion.finished);
    }
    mInFlight.prefillBatch.clear();
    mHasPrefill = false;
}

void PhaseDispatchWorker::completeDecodeInFlight()
{
    if (!mHasDecode)
    {
        return;
    }
    if (mCallbacks.completeDecodeBatch)
    {
        mCallbacks.completeDecodeBatch(mInFlight.decodeBatch);
    }
    recordTimeline(mInFlight.decodeBatch, PhaseTimelineStage::kDecodeDone);
    for (PhaseWorkItem const& item : mInFlight.decodeBatch)
    {
        PhaseDecodeCompletion const completion = mCallbacks.completeDecode(item);
        mScheduler.completeDecode(item, completion.resultingKVLength, completion.finished);
    }
    mInFlight.decodeBatch.clear();
    mHasDecode = false;
}

void PhaseDispatchWorker::completeInFlight()
{
    completePrefillInFlight();
    completeDecodeInFlight();
    mCurrentMetrics.hostCompletionNs = phaseTimelineNowNs();
    collectMetrics();
    if (mActivityTimeline != nullptr)
    {
        static_cast<void>(mActivityTimeline->poll());
    }
    mInFlight = PhaseDispatchPlan{};
    mBusy = false;
    mHasPrefill = false;
    mHasDecode = false;
    mDecodeDeferred = false;
    mResidualAugmentation = false;
}

void PhaseDispatchWorker::recordTimeline(
    std::vector<PhaseWorkItem> const& batch, PhaseTimelineStage stage, uint64_t timestampNs) const
{
    if (!mCallbacks.onTimeline)
    {
        return;
    }
    timestampNs = timestampNs > 0U ? timestampNs : phaseTimelineNowNs();
    int32_t const batchSize = static_cast<int32_t>(batch.size());
    for (PhaseWorkItem const& item : batch)
    {
        mCallbacks.onTimeline(
            {item.requestId, stage, timestampNs, mCurrentMetrics.dispatchIndex, batchSize, item.kvSlotId});
    }
}

void PhaseDispatchWorker::collectMetrics()
{
    if (mCallbacks.executionVariant)
    {
        mCurrentMetrics.globalExecutionVariant = mCallbacks.executionVariant(mCurrentMetrics.kind);
    }
    float phaseEndMs{};
    cudaEvent_t const makespanStart = mResidualAugmentation ? mAugmentationStart : mDispatchStart;
    if (mCurrentMetrics.prefillBatchSize > 0)
    {
        CUDA_CHECK(cudaEventElapsedTime(&mCurrentMetrics.prefillGpuMs, mPrefillStart, mPrefillDone));
        CUDA_CHECK(cudaEventElapsedTime(&phaseEndMs, makespanStart, mPrefillDone));
        phaseEndMs = std::max(0.0F, phaseEndMs);
        mCurrentMetrics.prefillCompletionMs = phaseEndMs;
        mCurrentMetrics.makespanGpuMs = std::max(mCurrentMetrics.makespanGpuMs, phaseEndMs);
    }
    if (mCurrentMetrics.decodeBatchSize > 0)
    {
        CUDA_CHECK(cudaEventElapsedTime(&mCurrentMetrics.decodeGpuMs, mDecodeStart, mDecodeDone));
        CUDA_CHECK(cudaEventElapsedTime(&phaseEndMs, makespanStart, mDecodeDone));
        phaseEndMs = std::max(0.0F, phaseEndMs);
        mCurrentMetrics.decodeCompletionMs = phaseEndMs;
        mCurrentMetrics.makespanGpuMs = std::max(mCurrentMetrics.makespanGpuMs, phaseEndMs);
    }
    float const phaseSum = mCurrentMetrics.prefillGpuMs + mCurrentMetrics.decodeGpuMs;
    if (phaseSum > 0.0F)
    {
        mCurrentMetrics.overlapRatio = std::clamp(1.0F - mCurrentMetrics.makespanGpuMs / phaseSum, 0.0F, 1.0F);
    }
    mLastMetrics = mCurrentMetrics;
    mScheduler.observeMetrics(*mLastMetrics);
    if (mCallbacks.onMetrics)
    {
        mCallbacks.onMetrics(*mLastMetrics);
    }
}

void PhaseDispatchWorker::runUntilIdle(size_t maxDispatches)
{
    check::check(maxDispatches > 0, "PhaseDispatchWorker maxDispatches must be positive.");
    size_t dispatches{};
    while (mBusy || !mScheduler.empty())
    {
        if (!mBusy)
        {
            check::check(dispatches < maxDispatches, "PhaseDispatchWorker exceeded its dispatch limit.");
            check::check(dispatchNext(), "PhaseDispatchWorker failed to dispatch queued work.");
            ++dispatches;
        }
        wait();
    }
}

bool PhaseDispatchWorker::busy() const noexcept
{
    return mBusy;
}

bool PhaseDispatchWorker::empty() const noexcept
{
    return !mBusy && mScheduler.empty();
}

PhaseDispatchKind PhaseDispatchWorker::inFlightKind() const noexcept
{
    return mBusy ? mInFlight.kind : PhaseDispatchKind::kNone;
}

PhasePrefillClass PhaseDispatchWorker::inFlightPrefillClass() const noexcept
{
    if (!mBusy || mInFlight.prefillBatch.empty())
    {
        return PhasePrefillClass::kAny;
    }
    return mInFlight.prefillBatch.front().prefillClass;
}

PhaseGlobalActionCandidate const* PhaseDispatchWorker::inFlightGlobalCandidate() const noexcept
{
    if (!mBusy || !mInFlight.globalCandidate.has_value())
    {
        return nullptr;
    }
    return &*mInFlight.globalCandidate;
}

size_t PhaseDispatchWorker::dispatchCount() const noexcept
{
    return mDispatchCount;
}

std::optional<PhaseDispatchMetrics> const& PhaseDispatchWorker::lastMetrics() const noexcept
{
    return mLastMetrics;
}

PhaseExecutionSafetyContract const& PhaseDispatchWorker::safetyContract() const noexcept
{
    return mSafetyContract;
}

PhaseInFlightSnapshot PhaseDispatchWorker::inFlightSnapshot(uint64_t hostSnapshotNs) const noexcept
{
    constexpr uint64_t kEXECUTION_ID_SHIFT = 2U;
    constexpr uint64_t kPREFILL_EXECUTION_TAG = 1U;
    constexpr uint64_t kDECODE_EXECUTION_TAG = 2U;
    PhaseInFlightSnapshot result;
    result.hostSnapshotNs = hostSnapshotNs > 0U ? hostSnapshotNs : phaseTimelineNowNs();
    if (!mBusy)
    {
        return result;
    }

    uint64_t const fallbackPlanId
        = mCurrentMetrics.globalPlanId > 0U ? mCurrentMetrics.globalPlanId : mCurrentMetrics.dispatchIndex;
    uint64_t const fallbackActionId
        = mCurrentMetrics.globalCandidateId > 0U ? mCurrentMetrics.globalCandidateId : fallbackPlanId;
    auto const status = [this](PhaseUnifiedPhase phase) {
        if (phase == PhaseUnifiedPhase::kDecode && mDecodeDeferred)
        {
            return PhaseInFlightStatus::kSubmitted;
        }
        cudaEvent_t const done = phase == PhaseUnifiedPhase::kPrefill ? mPrefillDone : mDecodeDone;
        return cudaEventQuery(done) == cudaSuccess ? PhaseInFlightStatus::kCompletionReady
                                                   : PhaseInFlightStatus::kRunning;
    };
    if (mHasPrefill)
    {
        PhaseInFlightWorkSnapshot work;
        work.phase = PhaseUnifiedPhase::kPrefill;
        work.status = status(work.phase);
        work.executionId = (mCurrentMetrics.dispatchIndex << kEXECUTION_ID_SHIFT) | kPREFILL_EXECUTION_TAG;
        work.activityCorrelationId = mCurrentMetrics.dispatchIndex;
        work.planId = mPrefillPlanId > 0U ? mPrefillPlanId : fallbackPlanId;
        work.actionId = mPrefillActionId > 0U ? mPrefillActionId : fallbackActionId;
        work.dispatchHostNs = mPrefillEnqueueHostNs > 0U ? mPrefillEnqueueHostNs : mCurrentMetrics.hostDispatchStartNs;
        work.prepareStartHostNs = mCurrentMetrics.prefillHostExecution.prepareStartHostNs;
        work.prepareEndHostNs = mCurrentMetrics.prefillHostExecution.prepareEndHostNs;
        work.executeStartHostNs = mCurrentMetrics.prefillHostExecution.executeStartHostNs;
        work.executeEndHostNs = mCurrentMetrics.prefillHostExecution.executeEndHostNs;
        work.graphReplay = mCurrentMetrics.prefillHostExecution.graphReplay;
        work.dispatchAgeUs = result.hostSnapshotNs >= work.dispatchHostNs
            ? static_cast<double>(result.hostSnapshotNs - work.dispatchHostNs) / 1000.0
            : 0.0;
        work.requestIds = mCurrentMetrics.prefillRequestIds;
        work.work.prefillRows = mCurrentMetrics.prefillBatchSize;
        work.work.prefillTokens = mCurrentMetrics.prefillTokens;
        result.work.push_back(work);
        result.outstanding = result.outstanding | PhaseExecutionSet::kPrefill;
    }
    if (mHasDecode)
    {
        PhaseInFlightWorkSnapshot work;
        work.phase = PhaseUnifiedPhase::kDecode;
        work.status = status(work.phase);
        work.executionId = (mCurrentMetrics.dispatchIndex << kEXECUTION_ID_SHIFT) | kDECODE_EXECUTION_TAG;
        work.activityCorrelationId = mCurrentMetrics.dispatchIndex;
        work.planId = mDecodePlanId > 0U ? mDecodePlanId : fallbackPlanId;
        work.actionId = mDecodeActionId > 0U ? mDecodeActionId : fallbackActionId;
        work.dispatchHostNs = mDecodeEnqueueHostNs > 0U ? mDecodeEnqueueHostNs : mCurrentMetrics.hostDispatchStartNs;
        work.prepareStartHostNs = mCurrentMetrics.decodeHostExecution.prepareStartHostNs;
        work.prepareEndHostNs = mCurrentMetrics.decodeHostExecution.prepareEndHostNs;
        work.executeStartHostNs = mCurrentMetrics.decodeHostExecution.executeStartHostNs;
        work.executeEndHostNs = mCurrentMetrics.decodeHostExecution.executeEndHostNs;
        work.graphReplay = mCurrentMetrics.decodeHostExecution.graphReplay;
        work.dispatchAgeUs = result.hostSnapshotNs >= work.dispatchHostNs
            ? static_cast<double>(result.hostSnapshotNs - work.dispatchHostNs) / 1000.0
            : 0.0;
        work.requestIds = mCurrentMetrics.decodeRequestIds;
        work.work.decodeRows = mCurrentMetrics.decodeBatchSize;
        work.work.decodeContextTokens = mCurrentMetrics.decodeContextTokens;
        result.work.push_back(work);
        result.outstanding = result.outstanding | PhaseExecutionSet::kDecode;
    }
    return result;
}

void PhaseDispatchWorker::setActivityTimeline(PhaseActivityTimelineRecorder* timeline)
{
    check::check(!mBusy, "Phase activity timeline cannot change while a dispatch is in flight.");
    mActivityTimeline = timeline;
}

CUcontext PhaseDispatchWorker::cudaContext() const noexcept
{
    return mCudaContext;
}

} // namespace rt
} // namespace trt_edgellm
