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

#include <utility>

namespace trt_edgellm
{
namespace rt
{

PhaseDispatchWorker::PhaseDispatchWorker(PhaseQueueScheduler& scheduler, PhaseDispatchWorkerCallbacks callbacks,
    cudaStream_t prefillStream, cudaStream_t decodeStream, PhaseStreamExecutionMode executionMode)
    : mScheduler(scheduler)
    , mCallbacks(std::move(callbacks))
    , mPrefillStream(prefillStream)
    , mDecodeStream(decodeStream)
    , mExecutionMode(executionMode)
{
    check::check(static_cast<bool>(mCallbacks.enqueuePrefill), "Prefill enqueue callback is required.");
    check::check(static_cast<bool>(mCallbacks.enqueueDecode), "Decode enqueue callback is required.");
    check::check(static_cast<bool>(mCallbacks.completePrefill), "Prefill completion callback is required.");
    check::check(static_cast<bool>(mCallbacks.completeDecode), "Decode completion callback is required.");
    CUDA_CHECK(cudaEventCreateWithFlags(&mDispatchStart, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&mPrefillDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&mDecodeDone, cudaEventDisableTiming));
}

PhaseDispatchWorker::~PhaseDispatchWorker() noexcept
{
    static_cast<void>(cudaEventDestroy(mDispatchStart));
    static_cast<void>(cudaEventDestroy(mPrefillDone));
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
    CUDA_CHECK(cudaEventRecord(mDispatchStart, mPrefillStream));
    if (mHasPrefill)
    {
        mCallbacks.enqueuePrefill(mInFlight.prefillBatch, mPrefillStream);
        CUDA_CHECK(cudaEventRecord(mPrefillDone, mPrefillStream));
    }
    if (mHasDecode)
    {
        bool const serializeSharedContext
            = mHasPrefill && mExecutionMode == PhaseStreamExecutionMode::kSharedContextSerialized;
        if (serializeSharedContext)
        {
            // A stream wait does not protect host-side TensorRT context state.
            // Defer prepare/execute until the prefill event has completed.
            mDecodeDeferred = true;
        }
        else
        {
            CUDA_CHECK(cudaStreamWaitEvent(mDecodeStream, mDispatchStart));
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
    mInFlight = PhaseDispatchPlan{};
    mBusy = false;
    mHasPrefill = false;
    mHasDecode = false;
    mDecodeDeferred = false;
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

size_t PhaseDispatchWorker::dispatchCount() const noexcept
{
    return mDispatchCount;
}

} // namespace rt
} // namespace trt_edgellm
