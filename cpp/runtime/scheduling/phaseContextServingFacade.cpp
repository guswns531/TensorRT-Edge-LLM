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

#include "runtime/scheduling/phaseContextServingFacade.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

PhaseContextServingFacade::PhaseContextServingFacade(int32_t maxSlots, PhaseQueueSchedulerConfig schedulerConfig,
    PhaseContextServingCallbacks callbacks, HybridCacheManager& cacheManager, TensorMap& decodeTensorMap,
    cudaStream_t prefillStream, cudaStream_t decodeStream, PhaseTensorRTContextMode executionMode,
    TensorMap* prefillTensorMap, int32_t maxPrefillChunkTokens, size_t maxPendingAdmissions,
    PhaseExecutionSafetyContract safetyContract)
    : mCallbacks(std::move(callbacks))
    , mCacheManager(cacheManager)
    , mHostAdmissionSlotIds(
          {maxSlots}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "phase_serving_host_admission_slots")
    , mDeviceAdmissionSlotIds({maxSlots}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "phase_serving_admission_slots")
    , mDecodeAdapter(schedulerConfig.maxDecodeBatchSize, cacheManager, decodeTensorMap, "phase_serving_decode")
    , mMaxPendingAdmissions(maxPendingAdmissions)
{
    check::check(maxSlots <= cacheManager.getGlobalKVCacheLengths().getShape()[0],
        "Serving facade slot count exceeds the indexed KV cache capacity.");
    bool const usesLegacyPrefill = static_cast<bool>(mCallbacks.enqueuePrefill);
    bool const usesPackedPrefill = static_cast<bool>(mCallbacks.enqueuePackedPrefill);
    check::check(usesLegacyPrefill != usesPackedPrefill,
        "Serving facade requires exactly one legacy or packed prefill enqueue callback.");
    if (usesPackedPrefill)
    {
        check::check(prefillTensorMap != nullptr, "Packed prefill serving requires a prefill TensorMap.");
        check::check(maxPrefillChunkTokens > 0, "Packed prefill serving requires a positive maximum chunk length.");
        mPrefillAdapter = std::make_unique<PhasePrefillContextBatchAdapter>(schedulerConfig.maxPrefillBatchSize,
            maxPrefillChunkTokens, cacheManager, *prefillTensorMap, "phase_serving_prefill");
    }
    check::check(static_cast<bool>(mCallbacks.completePrefill), "Serving prefill completion callback is required.");
    bool const usesLegacyDecode = static_cast<bool>(mCallbacks.enqueueDecode);
    bool const usesPackedDecode = static_cast<bool>(mCallbacks.enqueuePackedDecode);
    check::check(usesLegacyDecode != usesPackedDecode,
        "Serving facade requires exactly one legacy or packed decode enqueue callback.");
    check::check(!usesLegacyDecode || static_cast<bool>(mCallbacks.completeDecode),
        "Legacy serving decode completion callback is required.");
    check::check(!usesPackedDecode || static_cast<bool>(mCallbacks.completePackedDecode),
        "Packed serving decode completion callback is required.");
    mLifecycle = std::make_unique<PhaseRequestLifecycle>(maxSlots, std::move(schedulerConfig), makeLifecycleCallbacks(),
        prefillStream, decodeStream, executionMode, safetyContract);
}

PhaseRequestLifecycleCallbacks PhaseContextServingFacade::makeLifecycleCallbacks()
{
    PhaseRequestLifecycleCallbacks result;
    result.execution.onMetrics = mCallbacks.onDispatchMetrics;
    result.execution.enqueuePrefill
        = [this](std::vector<PhaseWorkItem> const& batch, cudaStream_t stream) { enqueuePrefillBatch(batch, stream); };
    result.execution.completePrefillBatch = [this](std::vector<PhaseWorkItem> const& batch) {
        if (mPrefillAdapter)
        {
            check::check(mPrefillAdapter->packed(), "Serving prefill completion has no packed context.");
            if (mCallbacks.completePackedPrefill)
            {
                mCallbacks.completePackedPrefill(*mPrefillAdapter);
            }
            mPrefillAdapter->complete();
        }
        if (mCallbacks.completePrefillBatch)
        {
            mCallbacks.completePrefillBatch(batch);
        }
    };
    result.execution.completePrefill = [this](PhaseWorkItem const& item) {
        int32_t const resultingKVLength = mCallbacks.completePrefill(item);
        bool finished{};
        if (item.tokenOffset + item.tokenCount == item.promptTokenCount)
        {
            Registration const& source = registration(item.requestId);
            finished = mCallbacks.isPrefillFinished
                ? mCallbacks.isPrefillFinished(item.requestId, *source.context, source.contextRow)
                : source.context->finishedStates[static_cast<size_t>(source.contextRow)] != 0;
        }
        return PhasePrefillCompletion{resultingKVLength, finished};
    };
    result.execution.enqueueDecode
        = [this](std::vector<PhaseWorkItem> const& batch, cudaStream_t stream) { enqueueDecodeBatch(batch, stream); };
    result.execution.completeDecodeBatch
        = [this](std::vector<PhaseWorkItem> const& batch) { completeDecodeBatch(batch); };
    result.execution.completeDecode = [this](PhaseWorkItem const& item) {
        Registration const& source = registration(item.requestId);
        bool const finished = mCallbacks.isDecodeFinished
            ? mCallbacks.isDecodeFinished(item.requestId, *source.context, source.contextRow)
            : source.context->finishedStates[static_cast<size_t>(source.contextRow)] != 0;
        return PhaseDecodeCompletion{item.tokenCount + 1, finished};
    };
    result.onTerminal = [this](PhaseRequestSnapshot const& snapshot) {
        check::check(mRegistrations.find(snapshot.requestId) != mRegistrations.end(),
            "Terminal phase request has no registered source context.");
        mRegistrations.erase(snapshot.requestId);
        if (mCallbacks.onTerminal)
        {
            mCallbacks.onTerminal(snapshot);
        }
        mPendingAdmissionRequired = true;
    };
    return result;
}

void PhaseContextServingFacade::registerSource(
    uint64_t requestId, DecodingInferenceContext& context, int32_t contextRow)
{
    check::check(contextRow >= 0 && contextRow < context.activeBatchSize,
        "Serving source row is outside the active request context.");
    check::check(context.phaseBatchState == nullptr, "A packed phase context cannot be registered as a source.");
    check::check(mRegistrations.find(requestId) == mRegistrations.end(), "Serving request ID is already registered.");
    for (auto const& [registeredId, source] : mRegistrations)
    {
        static_cast<void>(registeredId);
        check::check(source.context != &context || source.contextRow != contextRow,
            "Serving source context row is already registered.");
    }
    mRegistrations.emplace(requestId, Registration{&context, contextRow});
}

int32_t PhaseContextServingFacade::submit(uint64_t requestId, DecodingInferenceContext& context, int32_t contextRow,
    int32_t promptTokenCount, PhaseSchedulingHints scheduling)
{
    registerSource(requestId, context, contextRow);
    try
    {
        return mLifecycle->submit(requestId, promptTokenCount, scheduling);
    }
    catch (...)
    {
        mRegistrations.erase(requestId);
        throw;
    }
}

int32_t PhaseContextServingFacade::reserveForEncoder(uint64_t requestId, DecodingInferenceContext& context,
    int32_t contextRow, int32_t promptTokenCountEstimate, PhaseSchedulingHints scheduling)
{
    registerSource(requestId, context, contextRow);
    try
    {
        return mLifecycle->reserveForEncoder(requestId, promptTokenCountEstimate, scheduling);
    }
    catch (...)
    {
        mRegistrations.erase(requestId);
        throw;
    }
}

void PhaseContextServingFacade::beginPrefillAfterEncoder(PhaseWorkItem const& item)
{
    auto const snapshot = mLifecycle->request(item.requestId);
    check::check(snapshot.has_value(), "Encoder handoff has no phase request reservation.");
    check::check(snapshot->status == PhaseRequestStatus::kEncoder, "Encoder handoff request is not encoder-pending.");
    check::check(snapshot->kvSlotId == item.kvSlotId, "Encoder handoff changed the stable KV slot.");
    check::check(item.tokenOffset == 0 && item.tokenCount > 0,
        "Encoder handoff must provide a non-empty prompt at token offset zero.");
    int32_t const promptTokenCount = item.promptTokenCount > 0 ? item.promptTokenCount : item.tokenCount;
    check::check(promptTokenCount == item.tokenCount, "Encoder handoff must provide the complete prompt in V1.");
    mLifecycle->beginPrefill(item.requestId, promptTokenCount, item.allowChunkedPrefill);
}

PhaseAdmissionResult PhaseContextServingFacade::submitOrQueue(uint64_t requestId, DecodingInferenceContext& context,
    int32_t contextRow, int32_t promptTokenCount, PhaseSchedulingHints scheduling)
{
    check::check(promptTokenCount > 0, "Phase request prompt length must be positive.");
    registerSource(requestId, context, contextRow);
    PhaseAdmissionResult result{requestId, -1, PhaseAdmissionStatus::kPending};
    try
    {
        if (mLifecycle->availableSlotCount() > 0)
        {
            result.kvSlotId = mLifecycle->submit(requestId, promptTokenCount, scheduling);
            result.status = PhaseAdmissionStatus::kAdmitted;
        }
        else
        {
            check::check(mPendingAdmissions.size() < mMaxPendingAdmissions, "Serving pending admission queue is full.");
            mPendingAdmissions.push_back({requestId, promptTokenCount, scheduling});
        }
    }
    catch (...)
    {
        mRegistrations.erase(requestId);
        throw;
    }
    if (mCallbacks.onAdmission)
    {
        mCallbacks.onAdmission(result);
    }
    return result;
}

void PhaseContextServingFacade::admitPendingRequests()
{
    if (mLifecycle->busy())
    {
        mPendingAdmissionRequired = !mPendingAdmissions.empty();
        return;
    }
    mPendingAdmissionRequired = false;
    while (!mPendingAdmissions.empty() && mLifecycle->availableSlotCount() > 0)
    {
        PendingAdmission const admission = mPendingAdmissions.front();
        int32_t const slot = mLifecycle->submit(admission.requestId, admission.promptTokenCount, admission.scheduling);
        mPendingAdmissions.pop_front();
        if (mCallbacks.onAdmission)
        {
            mCallbacks.onAdmission({admission.requestId, slot, PhaseAdmissionStatus::kAdmitted});
        }
    }
}

bool PhaseContextServingFacade::cancel(uint64_t requestId)
{
    auto const pending = std::find_if(mPendingAdmissions.begin(), mPendingAdmissions.end(),
        [requestId](PendingAdmission const& admission) { return admission.requestId == requestId; });
    if (pending != mPendingAdmissions.end())
    {
        int32_t const promptTokenCount = pending->promptTokenCount;
        mPendingAdmissions.erase(pending);
        check::check(mRegistrations.erase(requestId) == 1, "Pending phase request has no registered source context.");
        if (mCallbacks.onTerminal)
        {
            mCallbacks.onTerminal({requestId, -1, promptTokenCount, 0, PhaseRequestStatus::kCancelled});
        }
        return true;
    }

    bool const cancelled = mLifecycle->cancel(requestId);
    if (cancelled && mPendingAdmissionRequired)
    {
        admitPendingRequests();
    }
    return cancelled;
}

void PhaseContextServingFacade::enqueuePrefillBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream)
{
    int32_t newSlotCount{};
    int32_t* hostSlotIds = mHostAdmissionSlotIds.dataPointer<int32_t>();
    for (PhaseWorkItem const& item : batch)
    {
        if (item.tokenOffset == 0)
        {
            hostSlotIds[newSlotCount++] = item.kvSlotId;
        }
    }
    if (newSlotCount > 0)
    {
        check::check(mHostAdmissionSlotIds.reshape({newSlotCount}), "Host admission slot IDs reshape failed.");
        check::check(mDeviceAdmissionSlotIds.reshape({newSlotCount}), "Admission slot IDs reshape failed.");
        CUDA_CHECK(cudaMemcpyAsync(mDeviceAdmissionSlotIds.rawPointer(), mHostAdmissionSlotIds.rawPointer(),
            newSlotCount * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        mCacheManager.clearPhaseKVCacheLengths(mDeviceAdmissionSlotIds, stream);
    }
    if (!mPrefillAdapter)
    {
        mCallbacks.enqueuePrefill(batch, stream);
        return;
    }

    std::vector<PhasePrefillContextRow> rows;
    rows.reserve(batch.size());
    for (PhaseWorkItem const& item : batch)
    {
        Registration& source = registration(item.requestId);
        rows.push_back({item.requestId, source.context, source.contextRow, item.kvSlotId, item.tokenOffset,
            item.tokenCount, item.promptTokenCount});
    }
    mPrefillAdapter->pack(rows, stream);
    mCallbacks.enqueuePackedPrefill(*mPrefillAdapter);
}

void PhaseContextServingFacade::enqueueDecodeBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream)
{
    std::vector<PhaseContextRow> rows;
    rows.reserve(batch.size());
    for (PhaseWorkItem const& item : batch)
    {
        Registration& source = registration(item.requestId);
        rows.push_back({item.requestId, source.context, source.contextRow, item.kvSlotId, item.tokenCount});
    }
    mDecodeAdapter.packDecode(rows, stream);
    if (mCallbacks.enqueuePackedDecode)
    {
        mCallbacks.enqueuePackedDecode(mDecodeAdapter);
    }
    else
    {
        mCallbacks.enqueueDecode(mDecodeAdapter.packedContext());
    }
}

void PhaseContextServingFacade::completeDecodeBatch(std::vector<PhaseWorkItem> const& batch)
{
    check::check(mDecodeAdapter.packed(), "Serving decode completion has no packed context.");
    check::check(mDecodeAdapter.workItems().size() == batch.size(),
        "Serving decode completion batch size does not match its packed context.");
    if (mCallbacks.completePackedDecode)
    {
        mCallbacks.completePackedDecode(mDecodeAdapter);
    }
    else
    {
        mCallbacks.completeDecode(mDecodeAdapter.packedContext());
    }
    mDecodeAdapter.scatterDecode();
}

PhaseContextServingFacade::Registration& PhaseContextServingFacade::registration(uint64_t requestId)
{
    auto const it = mRegistrations.find(requestId);
    check::check(it != mRegistrations.end(), "Phase request has no registered source context.");
    return it->second;
}

PhaseContextServingFacade::Registration const& PhaseContextServingFacade::registration(uint64_t requestId) const
{
    auto const it = mRegistrations.find(requestId);
    check::check(it != mRegistrations.end(), "Phase request has no registered source context.");
    return it->second;
}

bool PhaseContextServingFacade::dispatchNext()
{
    admitPendingRequests();
    return mLifecycle->dispatchNext();
}

bool PhaseContextServingFacade::poll()
{
    bool const completed = mLifecycle->poll();
    if (completed && mPendingAdmissionRequired)
    {
        admitPendingRequests();
    }
    return completed;
}

void PhaseContextServingFacade::wait()
{
    mLifecycle->wait();
    if (mPendingAdmissionRequired)
    {
        admitPendingRequests();
    }
}

void PhaseContextServingFacade::runUntilIdle(size_t maxDispatches)
{
    check::check(maxDispatches > 0, "Serving facade maxDispatches must be positive.");
    size_t dispatches{};
    while (!empty())
    {
        if (!busy())
        {
            check::check(dispatches < maxDispatches, "Serving facade exceeded its dispatch limit.");
            check::check(dispatchNext(), "Serving facade failed to dispatch queued work.");
            ++dispatches;
        }
        wait();
    }
}

bool PhaseContextServingFacade::empty() const noexcept
{
    return mPendingAdmissions.empty() && mLifecycle->empty();
}

bool PhaseContextServingFacade::hasQueuedPhaseWork() const noexcept
{
    return mLifecycle->hasQueuedWork();
}

bool PhaseContextServingFacade::busy() const noexcept
{
    return mLifecycle->busy();
}

size_t PhaseContextServingFacade::activeRequestCount() const noexcept
{
    return mLifecycle->activeRequestCount();
}

size_t PhaseContextServingFacade::pendingRequestCount() const noexcept
{
    return mPendingAdmissions.size();
}

size_t PhaseContextServingFacade::registeredRequestCount() const noexcept
{
    return mRegistrations.size();
}

int32_t PhaseContextServingFacade::availableSlotCount() const noexcept
{
    return mLifecycle->availableSlotCount();
}

std::optional<PhaseRequestSnapshot> PhaseContextServingFacade::request(uint64_t requestId) const
{
    auto const pending = std::find_if(mPendingAdmissions.begin(), mPendingAdmissions.end(),
        [requestId](PendingAdmission const& admission) { return admission.requestId == requestId; });
    if (pending != mPendingAdmissions.end())
    {
        return PhaseRequestSnapshot{
            requestId, -1, pending->promptTokenCount, 0, PhaseRequestStatus::kPending, pending->scheduling};
    }
    return mLifecycle->request(requestId);
}

CUcontext PhaseContextServingFacade::cudaContext() const noexcept
{
    return mLifecycle->cudaContext();
}

} // namespace rt
} // namespace trt_edgellm
