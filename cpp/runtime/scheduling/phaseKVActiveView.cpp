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

#include "runtime/scheduling/phaseKVActiveView.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"

#include <algorithm>

namespace trt_edgellm
{
namespace rt
{

PhaseKVActiveView::PhaseKVActiveView(
    int32_t maxActiveRows, StableKVPageManager& ownership, TensorMap& tensorMap, std::string const& name)
    : mMaxActiveRows(maxActiveRows)
    , mOwnership(ownership)
    , mTensorMap(tensorMap)
    , mPageTable(maxActiveRows, ownership.maxPagesPerSequence(), ownership.config().numPages)
    , mHostLengths({maxActiveRows}, DeviceType::kCPU, nvinfer1::DataType::kINT32, name + "_host_active_kv_lengths")
    , mDeviceLengths({maxActiveRows}, DeviceType::kGPU, nvinfer1::DataType::kINT32, name + "_active_kv_lengths")
{
    ELLM_CHECK(maxActiveRows > 0 && maxActiveRows <= ownership.config().maxActiveRows,
        "Phase KV active-row capacity is invalid");
}

PhaseKVActiveView::~PhaseKVActiveView() noexcept
{
    restoreBindings();
}

void PhaseKVActiveView::prepare(std::vector<int32_t> const& activeStableSlots, cudaStream_t stream)
{
    ELLM_CHECK(!mPrepared, "Phase KV active view is already prepared");
    ELLM_CHECK(!activeStableSlots.empty(), "Phase KV active view cannot prepare an empty batch");
    ELLM_CHECK(static_cast<int32_t>(activeStableSlots.size()) <= mMaxActiveRows,
        "Phase KV active batch exceeds its configured capacity");

    mPreviousLengths = mTensorMap.get(binding_names::kKVCacheStartIndex);
    mPreviousPageTable = mTensorMap.get(binding_names::kKVPageTable);
    ELLM_CHECK(mPreviousLengths != nullptr && mPreviousPageTable != nullptr,
        "Phase KV active view requires existing length and page-table bindings");

    mOwnership.bindActiveRows(activeStableSlots, mPageTable, stream);
    std::vector<int32_t> const lengths = mOwnership.makeActiveLengths(activeStableSlots);
    ELLM_CHECK(mHostLengths.reshape({static_cast<int64_t>(lengths.size())}), "Phase KV host length reshape failed");
    ELLM_CHECK(mDeviceLengths.reshape({static_cast<int64_t>(lengths.size())}), "Phase KV device length reshape failed");
    std::copy(lengths.begin(), lengths.end(), mHostLengths.dataPointer<int32_t>());
    CUDA_CHECK(cudaMemcpyAsync(mDeviceLengths.rawPointer(), mHostLengths.rawPointer(), lengths.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));

    mTensorMap.set(binding_names::kKVCacheStartIndex, mDeviceLengths);
    mTensorMap.set(binding_names::kKVPageTable, mPageTable.kernelView());
    mActiveStableSlots = activeStableSlots;
    mPrepared = true;
}

void PhaseKVActiveView::complete()
{
    ELLM_CHECK(mPrepared, "Phase KV active view is not prepared");
    restoreBindings();
    mActiveStableSlots.clear();
    mPrepared = false;
}

void PhaseKVActiveView::commitLengths(std::vector<int32_t> const& resultingLengths)
{
    ELLM_CHECK(mPrepared, "Phase KV active view is not prepared");
    ELLM_CHECK(resultingLengths.size() == mActiveStableSlots.size(),
        "Phase KV committed length count does not match the active batch");
    for (size_t row = 0; row < resultingLengths.size(); ++row)
    {
        mOwnership.setLength(mActiveStableSlots[row], resultingLengths[row]);
    }
}

KVPageTable& PhaseKVActiveView::pageTable() noexcept
{
    return mPageTable;
}

Tensor& PhaseKVActiveView::activeLengths() noexcept
{
    return mDeviceLengths;
}

std::vector<int32_t> const& PhaseKVActiveView::activeStableSlots() const noexcept
{
    return mActiveStableSlots;
}

bool PhaseKVActiveView::prepared() const noexcept
{
    return mPrepared;
}

void PhaseKVActiveView::restoreBindings() noexcept
{
    if (!mPrepared)
    {
        return;
    }
    try
    {
        mTensorMap.set(binding_names::kKVCacheStartIndex, *mPreviousLengths);
        mTensorMap.set(binding_names::kKVPageTable, *mPreviousPageTable);
        mPreviousLengths = nullptr;
        mPreviousPageTable = nullptr;
    }
    catch (...)
    {
    }
}

} // namespace rt
} // namespace trt_edgellm
