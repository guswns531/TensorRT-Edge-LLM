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

#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

PhaseContextServingFacade::PhaseContextServingFacade(int32_t maxSlots, PhaseQueueSchedulerConfig schedulerConfig,
    PhaseContextServingCallbacks callbacks, HybridCacheManager& cacheManager, TensorMap& decodeTensorMap,
    cudaStream_t prefillStream, cudaStream_t decodeStream, PhaseStreamExecutionMode executionMode,
    TensorMap* prefillTensorMap, int32_t maxPrefillChunkTokens)
    : mCallbacks(std::move(callbacks))
    , mCacheManager(cacheManager)
    , mHostAdmissionSlotIds(
          {maxSlots}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "phase_serving_host_admission_slots")
    , mDeviceAdmissionSlotIds({maxSlots}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "phase_serving_admission_slots")
    , mDecodeAdapter(schedulerConfig.maxDecodeBatchSize, cacheManager, decodeTensorMap, "phase_serving_decode")
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
    check::check(static_cast<bool>(mCallbacks.enqueueDecode), "Serving decode enqueue callback is required.");
    check::check(static_cast<bool>(mCallbacks.completeDecode), "Serving decode completion callback is required.");
    mLifecycle = std::make_unique<PhaseRequestLifecycle>(
        maxSlots, std::move(schedulerConfig), makeLifecycleCallbacks(), prefillStream, decodeStream, executionMode);
}

PhaseRequestLifecycleCallbacks PhaseContextServingFacade::makeLifecycleCallbacks()
{
    PhaseRequestLifecycleCallbacks result;
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
    result.execution.completePrefill = [this](PhaseWorkItem const& item) { return mCallbacks.completePrefill(item); };
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
    };
    return result;
}

int32_t PhaseContextServingFacade::submit(
    uint64_t requestId, DecodingInferenceContext& context, int32_t contextRow, int32_t promptTokenCount)
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
    try
    {
        return mLifecycle->submit(requestId, promptTokenCount);
    }
    catch (...)
    {
        mRegistrations.erase(requestId);
        throw;
    }
}

bool PhaseContextServingFacade::cancel(uint64_t requestId)
{
    return mLifecycle->cancel(requestId);
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
    mCallbacks.enqueueDecode(mDecodeAdapter.packedContext());
}

void PhaseContextServingFacade::completeDecodeBatch(std::vector<PhaseWorkItem> const& batch)
{
    check::check(mDecodeAdapter.packed(), "Serving decode completion has no packed context.");
    check::check(mDecodeAdapter.workItems().size() == batch.size(),
        "Serving decode completion batch size does not match its packed context.");
    mCallbacks.completeDecode(mDecodeAdapter.packedContext());
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
    return mLifecycle->dispatchNext();
}

bool PhaseContextServingFacade::poll()
{
    return mLifecycle->poll();
}

void PhaseContextServingFacade::wait()
{
    mLifecycle->wait();
}

void PhaseContextServingFacade::runUntilIdle(size_t maxDispatches)
{
    mLifecycle->runUntilIdle(maxDispatches);
}

bool PhaseContextServingFacade::empty() const noexcept
{
    return mLifecycle->empty();
}

bool PhaseContextServingFacade::busy() const noexcept
{
    return mLifecycle->busy();
}

size_t PhaseContextServingFacade::activeRequestCount() const noexcept
{
    return mLifecycle->activeRequestCount();
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
    return mLifecycle->request(requestId);
}

} // namespace rt
} // namespace trt_edgellm
