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

#include "runtime/scheduling/phaseRequestLifecycle.h"

#include "common/checkMacros.h"

#include <utility>

namespace trt_edgellm
{
namespace rt
{

PhaseRequestLifecycle::PhaseRequestLifecycle(int32_t maxSlots, PhaseQueueSchedulerConfig schedulerConfig,
    PhaseRequestLifecycleCallbacks callbacks, cudaStream_t prefillStream, cudaStream_t decodeStream,
    PhaseTensorRTContextMode executionMode, PhaseExecutionSafetyContract safetyContract)
    : mScheduler(std::move(schedulerConfig))
    , mSlotAllocator(maxSlots)
    , mCallbacks(std::move(callbacks))
{
    check::check(static_cast<bool>(mCallbacks.execution.enqueuePrefill), "Prefill enqueue callback is required.");
    check::check(static_cast<bool>(mCallbacks.execution.enqueueDecode), "Decode enqueue callback is required.");
    check::check(static_cast<bool>(mCallbacks.execution.completePrefill), "Prefill completion callback is required.");
    check::check(static_cast<bool>(mCallbacks.execution.completeDecode), "Decode completion callback is required.");
    mWorker = std::make_unique<PhaseDispatchWorker>(
        mScheduler, makeWorkerCallbacks(), prefillStream, decodeStream, executionMode, safetyContract);
}

PhaseDispatchWorkerCallbacks PhaseRequestLifecycle::makeWorkerCallbacks()
{
    PhaseDispatchWorkerCallbacks result;
    result.enqueuePrefill = [this](std::vector<PhaseWorkItem> const& batch, cudaStream_t stream) {
        mCallbacks.execution.enqueuePrefill(batch, stream);
    };
    result.enqueueDecode = [this](std::vector<PhaseWorkItem> const& batch, cudaStream_t stream) {
        mCallbacks.execution.enqueueDecode(batch, stream);
    };
    result.completePrefillBatch = [this](std::vector<PhaseWorkItem> const& batch) {
        if (mCallbacks.execution.completePrefillBatch)
        {
            mCallbacks.execution.completePrefillBatch(batch);
        }
    };
    result.completeDecodeBatch = [this](std::vector<PhaseWorkItem> const& batch) {
        if (mCallbacks.execution.completeDecodeBatch)
        {
            mCallbacks.execution.completeDecodeBatch(batch);
        }
    };
    result.completePrefill = [this](PhaseWorkItem const& item) {
        PhasePrefillCompletion const completion = mCallbacks.execution.completePrefill(item);
        int32_t const resultingKVLength = completion.resultingKVLength;
        int32_t const completedPromptLength = item.tokenOffset + item.tokenCount;
        check::check(resultingKVLength >= completedPromptLength, "Prefill completion moved KV length backwards.");
        auto const it = mRequests.find(item.requestId);
        check::check(it != mRequests.end(), "Prefill completed for an unknown request.");
        PhaseRequestSnapshot& snapshot = it->second.snapshot;
        snapshot.kvLength = resultingKVLength;
        if (completion.finished)
        {
            check::check(completedPromptLength == item.promptTokenCount,
                "Phase request cannot finish before its final prefill chunk.");
            releaseSlot(snapshot);
            snapshot.status = PhaseRequestStatus::kFinished;
            if (mCallbacks.onTerminal)
            {
                mCallbacks.onTerminal(snapshot);
            }
        }
        else if (completedPromptLength == item.promptTokenCount)
        {
            snapshot.status = PhaseRequestStatus::kDecode;
        }
        return completion;
    };
    result.onDispatch = mCallbacks.execution.onDispatch;
    result.onMetrics = mCallbacks.execution.onMetrics;
    result.completeDecode = [this](PhaseWorkItem const& item) {
        PhaseDecodeCompletion const completion = mCallbacks.execution.completeDecode(item);
        check::check(completion.resultingKVLength >= item.tokenCount, "Decode completion moved KV length backwards.");
        auto const it = mRequests.find(item.requestId);
        check::check(it != mRequests.end(), "Decode completed for an unknown request.");
        PhaseRequestSnapshot& snapshot = it->second.snapshot;
        snapshot.kvLength = completion.resultingKVLength;
        snapshot.status = PhaseRequestStatus::kDecode;
        if (completion.finished)
        {
            releaseSlot(snapshot);
            snapshot.status = PhaseRequestStatus::kFinished;
            if (mCallbacks.onTerminal)
            {
                mCallbacks.onTerminal(snapshot);
            }
        }
        return completion;
    };
    return result;
}

void PhaseRequestLifecycle::releaseSlot(PhaseRequestSnapshot& snapshot)
{
    int32_t const slot = snapshot.kvSlotId;
    check::check(slot >= 0, "Phase request has no stable slot to release.");
    if (mCallbacks.onSlotRelease)
    {
        mCallbacks.onSlotRelease(slot);
    }
    mSlotAllocator.release(slot);
    snapshot.kvSlotId = -1;
}

int32_t PhaseRequestLifecycle::reserveForEncoder(
    uint64_t requestId, int32_t promptTokenCountEstimate, PhaseSchedulingHints scheduling)
{
    check::check(promptTokenCountEstimate > 0, "Phase request prompt length estimate must be positive.");
    check::check(mRequests.find(requestId) == mRequests.end(), "Phase request ID has already been used.");
    int32_t const slot = mSlotAllocator.reserve();
    PhaseRequestSnapshot snapshot{
        requestId, slot, promptTokenCountEstimate, 0, PhaseRequestStatus::kEncoder, scheduling};
    try
    {
        mRequests.emplace(requestId, RequestState{snapshot});
    }
    catch (...)
    {
        mRequests.erase(requestId);
        mSlotAllocator.release(slot);
        throw;
    }
    return slot;
}

void PhaseRequestLifecycle::beginPrefill(uint64_t requestId, int32_t promptTokenCount, bool allowChunkedPrefill)
{
    check::check(promptTokenCount > 0, "Phase request prompt length must be positive.");
    auto const it = mRequests.find(requestId);
    check::check(it != mRequests.end(), "Encoder completed for an unknown phase request.");
    PhaseRequestSnapshot& snapshot = it->second.snapshot;
    check::check(snapshot.status == PhaseRequestStatus::kEncoder, "Phase request is not waiting for encoder handoff.");
    snapshot.promptTokenCount = promptTokenCount;
    snapshot.status = PhaseRequestStatus::kPrefill;
    try
    {
        PhaseWorkItem item{requestId, promptTokenCount, snapshot.kvSlotId, 0, promptTokenCount, allowChunkedPrefill};
        item.scheduling = snapshot.scheduling;
        mScheduler.enqueuePrefill(std::move(item));
    }
    catch (...)
    {
        snapshot.status = PhaseRequestStatus::kEncoder;
        throw;
    }
}

int32_t PhaseRequestLifecycle::submit(uint64_t requestId, int32_t promptTokenCount, PhaseSchedulingHints scheduling)
{
    int32_t const slot = reserveForEncoder(requestId, promptTokenCount, scheduling);
    try
    {
        beginPrefill(requestId, promptTokenCount);
    }
    catch (...)
    {
        mSlotAllocator.release(slot);
        mRequests.erase(requestId);
        throw;
    }
    return slot;
}

bool PhaseRequestLifecycle::cancel(uint64_t requestId)
{
    auto const it = mRequests.find(requestId);
    if (it == mRequests.end())
    {
        return false;
    }
    PhaseRequestSnapshot& snapshot = it->second.snapshot;
    if (snapshot.status == PhaseRequestStatus::kFinished || snapshot.status == PhaseRequestStatus::kCancelled)
    {
        return false;
    }
    if (snapshot.status != PhaseRequestStatus::kEncoder && !mScheduler.cancel(requestId))
    {
        return false;
    }
    releaseSlot(snapshot);
    snapshot.status = PhaseRequestStatus::kCancelled;
    if (mCallbacks.onTerminal)
    {
        mCallbacks.onTerminal(snapshot);
    }
    return true;
}

bool PhaseRequestLifecycle::dispatchNext()
{
    return mWorker->dispatchNext();
}

bool PhaseRequestLifecycle::poll()
{
    return mWorker->poll();
}

void PhaseRequestLifecycle::wait()
{
    mWorker->wait();
}

void PhaseRequestLifecycle::runUntilIdle(size_t maxDispatches)
{
    mWorker->runUntilIdle(maxDispatches);
}

bool PhaseRequestLifecycle::empty() const noexcept
{
    return mScheduler.empty() && !mWorker->busy() && activeRequestCount() == 0;
}

bool PhaseRequestLifecycle::hasQueuedWork() const noexcept
{
    return !mScheduler.empty();
}

bool PhaseRequestLifecycle::busy() const noexcept
{
    return mWorker->busy();
}

size_t PhaseRequestLifecycle::activeRequestCount() const noexcept
{
    size_t count{};
    for (auto const& [requestId, state] : mRequests)
    {
        static_cast<void>(requestId);
        PhaseRequestStatus const status = state.snapshot.status;
        if (status == PhaseRequestStatus::kEncoder || status == PhaseRequestStatus::kPrefill
            || status == PhaseRequestStatus::kDecode)
        {
            ++count;
        }
    }
    return count;
}

int32_t PhaseRequestLifecycle::availableSlotCount() const noexcept
{
    return mSlotAllocator.available();
}

std::optional<PhaseRequestSnapshot> PhaseRequestLifecycle::request(uint64_t requestId) const
{
    auto const it = mRequests.find(requestId);
    if (it == mRequests.end())
    {
        return std::nullopt;
    }
    return it->second.snapshot;
}

CUcontext PhaseRequestLifecycle::cudaContext() const noexcept
{
    return mWorker->cudaContext();
}

} // namespace rt
} // namespace trt_edgellm
