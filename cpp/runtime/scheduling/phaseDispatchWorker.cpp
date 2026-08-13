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

#include <limits>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{

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
    CUDA_CHECK(cudaEventCreate(&mPrefillStart));
    CUDA_CHECK(cudaEventCreate(&mPrefillDone));
    CUDA_CHECK(cudaEventCreate(&mDecodeStart));
    CUDA_CHECK(cudaEventCreate(&mDecodeDone));
}

PhaseDispatchWorker::~PhaseDispatchWorker() noexcept
{
    static_cast<void>(cudaEventDestroy(mDispatchStart));
    static_cast<void>(cudaEventDestroy(mPrefillStart));
    static_cast<void>(cudaEventDestroy(mPrefillDone));
    static_cast<void>(cudaEventDestroy(mDecodeStart));
    static_cast<void>(cudaEventDestroy(mDecodeDone));
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
    mCurrentMetrics = PhaseDispatchMetrics{};
    mCurrentMetrics.dispatchIndex = mDispatchCount + 1;
    mCurrentMetrics.kind = mInFlight.kind;
    mCurrentMetrics.prefillBatchSize = static_cast<int32_t>(mInFlight.prefillBatch.size());
    mCurrentMetrics.decodeBatchSize = static_cast<int32_t>(mInFlight.decodeBatch.size());
    mCurrentMetrics.predictedPrefillGpuMs = mInFlight.predictedPrefillGpuMs;
    mCurrentMetrics.predictedDecodeSlowdownMs = mInFlight.predictedDecodeSlowdownMs;
    mCurrentMetrics.predictedDecodeDebtUs = mInFlight.predictedDecodeDebtUs;
    mCurrentMetrics.consecutiveOverlapBatches = mInFlight.consecutiveOverlapBatches;
    mCurrentMetrics.prefillDeferredForTpot = mInFlight.prefillDeferredForTpot;
    mCurrentMetrics.prefillCohortSize = mInFlight.prefillCohortSize;
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
    if (mHasPrefill)
    {
        CUDA_CHECK(cudaEventRecord(mPrefillStart, mPrefillStream));
        mCallbacks.enqueuePrefill(mInFlight.prefillBatch, mPrefillStream);
        CUDA_CHECK(cudaEventRecord(mPrefillDone, mPrefillStream));
    }
    if (mHasDecode)
    {
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
            CUDA_CHECK(cudaEventRecord(mDecodeStart, mDecodeStream));
            mCallbacks.enqueueDecode(mInFlight.decodeBatch, mDecodeStream);
            CUDA_CHECK(cudaEventRecord(mDecodeDone, mDecodeStream));
        }
    }
    mBusy = true;
    ++mDispatchCount;
    return true;
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
    CUDA_CHECK(cudaEventRecord(mDecodeStart, mDecodeStream));
    mCallbacks.enqueueDecode(mInFlight.decodeBatch, mDecodeStream);
    CUDA_CHECK(cudaEventRecord(mDecodeDone, mDecodeStream));
    mDecodeDeferred = false;
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
    collectMetrics();
    mInFlight = PhaseDispatchPlan{};
    mBusy = false;
    mHasPrefill = false;
    mHasDecode = false;
    mDecodeDeferred = false;
}

void PhaseDispatchWorker::collectMetrics()
{
    float phaseEndMs{};
    if (mCurrentMetrics.prefillBatchSize > 0)
    {
        CUDA_CHECK(cudaEventElapsedTime(&mCurrentMetrics.prefillGpuMs, mPrefillStart, mPrefillDone));
        CUDA_CHECK(cudaEventElapsedTime(&phaseEndMs, mDispatchStart, mPrefillDone));
        mCurrentMetrics.makespanGpuMs = std::max(mCurrentMetrics.makespanGpuMs, phaseEndMs);
    }
    if (mCurrentMetrics.decodeBatchSize > 0)
    {
        CUDA_CHECK(cudaEventElapsedTime(&mCurrentMetrics.decodeGpuMs, mDecodeStart, mDecodeDone));
        CUDA_CHECK(cudaEventElapsedTime(&phaseEndMs, mDispatchStart, mDecodeDone));
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

CUcontext PhaseDispatchWorker::cudaContext() const noexcept
{
    return mCudaContext;
}

} // namespace rt
} // namespace trt_edgellm
