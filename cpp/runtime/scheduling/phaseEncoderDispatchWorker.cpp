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

#include "runtime/scheduling/phaseEncoderDispatchWorker.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <algorithm>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{

bool completeIdentity(PhaseExecutionResourceIdentity const& identity) noexcept
{
    return identity.tensorRTExecutionContext != nullptr && identity.workspace != nullptr
        && identity.ioBuffers != nullptr;
}

} // namespace

void PhaseEncoderExecutionSafetyContract::validate() const
{
    check::check(completeIdentity(encoder),
        "Concurrent encoder execution requires non-null TensorRT context, workspace, and I/O identities.");
    check::check(!llmResources.empty(), "Concurrent encoder execution requires at least one LLM resource identity.");
    for (PhaseExecutionResourceIdentity const& llm : llmResources)
    {
        check::check(completeIdentity(llm),
            "Concurrent encoder execution requires complete LLM context, workspace, and I/O identities.");
        check::check(encoder.tensorRTExecutionContext != llm.tensorRTExecutionContext
                && encoder.workspace != llm.workspace && encoder.ioBuffers != llm.ioBuffers,
            "Encoder TensorRT context, workspace, and I/O must not alias an overlapping LLM phase.");
    }
}

PhaseEncoderDispatchWorker::PhaseEncoderDispatchWorker(PhaseQueueScheduler& prefillScheduler,
    PhaseEncoderQueueConfig config, PhaseEncoderDispatchWorkerCallbacks callbacks, cudaStream_t encoderStream,
    PhaseEncoderExecutionSafetyContract safetyContract)
    : mPrefillScheduler(prefillScheduler)
    , mConfig(config)
    , mCallbacks(std::move(callbacks))
    , mEncoderStream(encoderStream)
    , mSafetyContract(std::move(safetyContract))
{
    check::check(mConfig.maxBatchSize > 0, "Encoder maxBatchSize must be positive.");
    check::check(static_cast<bool>(mCallbacks.enqueueEncoder), "Encoder enqueue callback is required.");
    check::check(static_cast<bool>(mCallbacks.completeEncoder), "Encoder completion callback is required.");
    mSafetyContract.validate();
    check::check(mEncoderStream != nullptr, "Encoder execution requires an explicit non-default CUDA stream.");
    CUDA_DRIVER_CHECK(cuStreamGetCtx(mEncoderStream, &mCudaContext));
    check::check(mCudaContext != nullptr, "Encoder CUDA stream has no owning CUDA context.");
    CUcontext current{};
    CUDA_DRIVER_CHECK(cuCtxGetCurrent(&current));
    check::check(current == mCudaContext, "Encoder stream must belong to the thread's current CUDA context.");
    CUDA_CHECK(cudaEventCreate(&mEncoderStart));
    CUDA_CHECK(cudaEventCreate(&mEncoderDone));
}

PhaseEncoderDispatchWorker::~PhaseEncoderDispatchWorker() noexcept
{
    static_cast<void>(cudaEventDestroy(mEncoderStart));
    static_cast<void>(cudaEventDestroy(mEncoderDone));
}

void PhaseEncoderDispatchWorker::submit(PhaseEncoderWorkItem item)
{
    check::check(item.inputUnits > 0, "Encoder inputUnits must be positive.");
    check::check(item.kvSlotId >= 0, "Encoder handoff requires a stable non-negative KV slot.");
    check::check(!mPrefillScheduler.hasRequest(item.requestId), "Request is already active in the LLM scheduler.");
    check::check(mActiveRequestIds.find(item.requestId) == mActiveRequestIds.end(),
        "Request is already active in the encoder queue.");
    check::check(mActiveKVSlotIds.find(item.kvSlotId) == mActiveKVSlotIds.end(),
        "KV slot is already active in the encoder queue.");
    check::check(mConfig.maxQueuedRequests == 0 || mQueue.size() < mConfig.maxQueuedRequests,
        "Encoder admission queue is full.");
    mActiveRequestIds.insert(item.requestId);
    mActiveKVSlotIds.insert(item.kvSlotId);
    mQueuedSince[item.requestId] = std::chrono::steady_clock::now();
    mQueue.push_back(item);
}

bool PhaseEncoderDispatchWorker::cancel(uint64_t requestId)
{
    if (mBusy && std::any_of(mInFlight.begin(), mInFlight.end(), [requestId](PhaseEncoderWorkItem const& item) {
            return item.requestId == requestId;
        }))
    {
        return false;
    }
    auto const it = std::find_if(mQueue.begin(), mQueue.end(),
        [requestId](PhaseEncoderWorkItem const& item) { return item.requestId == requestId; });
    if (it == mQueue.end())
    {
        return false;
    }
    int32_t const kvSlotId = it->kvSlotId;
    mQueue.erase(it);
    check::check(mQueuedSince.erase(requestId) == 1, "Cancelled encoder request has no queue timestamp.");
    check::check(mActiveRequestIds.erase(requestId) == 1, "Cancelled encoder request is not active.");
    check::check(mActiveKVSlotIds.erase(kvSlotId) == 1, "Cancelled encoder KV slot is not active.");
    return true;
}

bool PhaseEncoderDispatchWorker::dispatchNext()
{
    check::check(!mBusy, "Cannot dispatch while an encoder batch is in flight.");
    if (mQueue.empty())
    {
        return false;
    }

    auto const now = std::chrono::steady_clock::now();
    int32_t const count = std::min<int32_t>(mConfig.maxBatchSize, mQueue.size());
    mInFlight.clear();
    mInFlight.reserve(count);
    mCurrentMetrics = PhaseEncoderDispatchMetrics{};
    mCurrentMetrics.dispatchIndex = mDispatchCount + 1;
    for (int32_t index = 0; index < count; ++index)
    {
        PhaseEncoderWorkItem item = mQueue.front();
        mQueue.pop_front();
        auto const timestamp = mQueuedSince.find(item.requestId);
        check::check(timestamp != mQueuedSince.end(), "Dispatched encoder request has no queue timestamp.");
        mCurrentMetrics.queueWaitUs = std::max(
            mCurrentMetrics.queueWaitUs, std::chrono::duration<double, std::micro>(now - timestamp->second).count());
        mQueuedSince.erase(timestamp);
        mCurrentMetrics.inputUnits += item.inputUnits;
        mInFlight.push_back(item);
    }
    mCurrentMetrics.batchSize = static_cast<int32_t>(mInFlight.size());
    CUDA_CHECK(cudaEventRecord(mEncoderStart, mEncoderStream));
    mCallbacks.enqueueEncoder(mInFlight, mEncoderStream);
    CUDA_CHECK(cudaEventRecord(mEncoderDone, mEncoderStream));
    mBusy = true;
    ++mDispatchCount;
    return true;
}

bool PhaseEncoderDispatchWorker::eventReady() const
{
    cudaError_t const status = cudaEventQuery(mEncoderDone);
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

bool PhaseEncoderDispatchWorker::poll()
{
    if (!mBusy || !eventReady())
    {
        return false;
    }
    completeInFlight();
    return true;
}

void PhaseEncoderDispatchWorker::wait()
{
    check::check(mBusy, "PhaseEncoderDispatchWorker has no in-flight batch to wait for.");
    CUDA_CHECK(cudaEventSynchronize(mEncoderDone));
    completeInFlight();
}

void PhaseEncoderDispatchWorker::completeInFlight()
{
    if (mCallbacks.completeEncoderBatch)
    {
        mCallbacks.completeEncoderBatch(mInFlight);
    }

    std::vector<PhaseWorkItem> prefillWork;
    prefillWork.reserve(mInFlight.size());
    for (PhaseEncoderWorkItem const& encoderItem : mInFlight)
    {
        PhaseWorkItem item = mCallbacks.completeEncoder(encoderItem);
        check::check(item.requestId == encoderItem.requestId, "Encoder handoff changed the request ID.");
        check::check(item.kvSlotId == encoderItem.kvSlotId, "Encoder handoff changed the stable KV slot.");
        check::check(item.tokenOffset == 0, "Encoder handoff must start at prefill token offset zero.");
        check::check(item.tokenCount > 0, "Encoder handoff requires a non-empty prefill prompt.");
        if (item.promptTokenCount == 0)
        {
            item.promptTokenCount = item.tokenCount;
        }
        check::check(item.promptTokenCount >= item.tokenCount,
            "Encoder handoff prompt length is smaller than its initial prefill work.");
        check::check(!mPrefillScheduler.hasRequest(item.requestId),
            "Encoder handoff request is already active in the LLM scheduler.");
        prefillWork.push_back(item);
    }
    for (PhaseWorkItem const& item : prefillWork)
    {
        mPrefillScheduler.enqueuePrefill(item);
        check::check(mActiveRequestIds.erase(item.requestId) == 1, "Completed encoder request is not active.");
        check::check(mActiveKVSlotIds.erase(item.kvSlotId) == 1, "Completed encoder KV slot is not active.");
    }

    CUDA_CHECK(cudaEventElapsedTime(&mCurrentMetrics.gpuMs, mEncoderStart, mEncoderDone));
    mLastMetrics = mCurrentMetrics;
    if (mCallbacks.onMetrics)
    {
        mCallbacks.onMetrics(*mLastMetrics);
    }
    mInFlight.clear();
    mBusy = false;
}

void PhaseEncoderDispatchWorker::runUntilIdle(size_t maxDispatches)
{
    size_t dispatches{};
    while (mBusy || !mQueue.empty())
    {
        if (!mBusy)
        {
            check::check(dispatches < maxDispatches, "Encoder dispatch limit exceeded.");
            check::check(dispatchNext(), "Encoder queue is non-empty but no batch was dispatched.");
            ++dispatches;
        }
        wait();
    }
}

size_t PhaseEncoderDispatchWorker::queueSize() const noexcept
{
    return mQueue.size();
}

bool PhaseEncoderDispatchWorker::busy() const noexcept
{
    return mBusy;
}

bool PhaseEncoderDispatchWorker::empty() const noexcept
{
    return mQueue.empty() && !mBusy;
}

bool PhaseEncoderDispatchWorker::hasRequest(uint64_t requestId) const noexcept
{
    return mActiveRequestIds.find(requestId) != mActiveRequestIds.end();
}

size_t PhaseEncoderDispatchWorker::dispatchCount() const noexcept
{
    return mDispatchCount;
}

std::optional<PhaseEncoderDispatchMetrics> const& PhaseEncoderDispatchWorker::lastMetrics() const noexcept
{
    return mLastMetrics;
}

PhaseEncoderExecutionSafetyContract const& PhaseEncoderDispatchWorker::safetyContract() const noexcept
{
    return mSafetyContract;
}

CUcontext PhaseEncoderDispatchWorker::cudaContext() const noexcept
{
    return mCudaContext;
}

} // namespace rt
} // namespace trt_edgellm
