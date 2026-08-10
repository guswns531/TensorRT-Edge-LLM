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

#include "runtime/scheduling/phaseBatchState.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <unordered_set>

namespace
{

int32_t validateMaxBatchSize(int32_t maxBatchSize)
{
    trt_edgellm::check::check(maxBatchSize > 0, "PhaseBatchState maxBatchSize must be positive.");
    return maxBatchSize;
}

} // namespace

namespace trt_edgellm
{
namespace rt
{

PhaseBatchState::PhaseBatchState(int32_t maxBatchSize, std::string const& name, bool indexedKVCache)
    : mMaxBatchSize(validateMaxBatchSize(maxBatchSize))
    , mHostSlotIds({mMaxBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, name + "_host_slot_ids")
    , mDeviceSlotIds({mMaxBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, name + "_slot_ids")
    , mDeviceLengths({mMaxBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32, name + "_lengths")
    , mIndexedKVCache(indexedKVCache)
{
}

void PhaseBatchState::bind(TensorMap& tensorMap)
{
    if (mIndexedKVCache)
    {
        tensorMap.set(binding_names::kKVSlotIds, mDeviceSlotIds);
    }
    tensorMap.set(binding_names::kKVCacheStartIndex, mDeviceLengths);
}

void PhaseBatchState::prepare(
    std::vector<PhaseWorkItem> const& batch, HybridCacheManager& cacheManager, cudaStream_t stream, bool decode)
{
    check::check(!batch.empty(), "PhaseBatchState cannot prepare an empty batch.");
    check::check(mIndexedKVCache == cacheManager.isIndexedKVCache(),
        "PhaseBatchState cache mode does not match the cache manager.");
    check::check(
        static_cast<int32_t>(batch.size()) <= mMaxBatchSize, "PhaseBatchState batch exceeds its configured maximum.");
    mBatchSize = static_cast<int32_t>(batch.size());
    check::check(mHostSlotIds.reshape({mBatchSize}), "Host slot IDs reshape failed.");
    check::check(mDeviceSlotIds.reshape({mBatchSize}), "Device slot IDs reshape failed.");
    check::check(mDeviceLengths.reshape({mBatchSize}), "Device phase lengths reshape failed.");

    int32_t* hostSlotIds = mHostSlotIds.dataPointer<int32_t>();
    rt::Tensor const& physicalLengths
        = mIndexedKVCache ? cacheManager.getGlobalKVCacheLengths() : cacheManager.getKVCacheLengths();
    int32_t const maxSlots = static_cast<int32_t>(physicalLengths.getShape()[0]);
    std::unordered_set<int32_t> uniqueSlots;
    for (int32_t row = 0; row < mBatchSize; ++row)
    {
        int32_t const slot = batch[row].kvSlotId;
        check::check(slot >= 0, "Phase work item has no physical KV slot lease.");
        check::check(slot < maxSlots, "Phase work item physical KV slot is out of range.");
        check::check(uniqueSlots.insert(slot).second, "Phase batch contains a duplicate physical KV slot.");
        hostSlotIds[row] = slot;
    }

    CUDA_CHECK(cudaMemcpyAsync(mDeviceSlotIds.rawPointer(), mHostSlotIds.rawPointer(), mBatchSize * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    cacheManager.preparePagedKVCapacity(batch, decode, stream);
    cacheManager.preparePhaseKVCacheLengths(mDeviceSlotIds, mDeviceLengths, stream);
}

void PhaseBatchState::commit(HybridCacheManager& cacheManager, int32_t increment, cudaStream_t stream)
{
    check::check(mBatchSize > 0, "PhaseBatchState must be prepared before commit.");
    cacheManager.commitPhaseSequenceLength(mDeviceSlotIds, mDeviceLengths, increment, stream);
}

void PhaseBatchState::commit(HybridCacheManager& cacheManager, Tensor const& increments, cudaStream_t stream)
{
    check::check(mBatchSize > 0, "PhaseBatchState must be prepared before commit.");
    check::check(increments.getShape()[0] == mBatchSize, "Phase increments do not match the prepared batch.");
    cacheManager.commitPhaseSequenceLength(mDeviceSlotIds, mDeviceLengths, increments, stream);
}

Tensor& PhaseBatchState::slotIds() noexcept
{
    return mDeviceSlotIds;
}

Tensor& PhaseBatchState::lengths() noexcept
{
    return mDeviceLengths;
}

int32_t PhaseBatchState::batchSize() const noexcept
{
    return mBatchSize;
}

} // namespace rt
} // namespace trt_edgellm
