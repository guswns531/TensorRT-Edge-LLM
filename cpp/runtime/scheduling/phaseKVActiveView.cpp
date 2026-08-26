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
    ++mMemoryStats.prepareCalls;
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
    ++mMemoryStats.lengthH2DOperations;
    mMemoryStats.lengthH2DBytes += lengths.size() * sizeof(int32_t);

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

void PhaseKVActiveView::preparePrefillMetadata(
    PipelineIO& io, std::vector<int32_t> const& chunkLengths, cudaStream_t stream, bool packedTokenLayout) const
{
    ELLM_CHECK(mPrepared, "Phase KV active view is not prepared");
    ELLM_CHECK(
        chunkLengths.size() == mActiveStableSlots.size(), "Phase prefill chunk count does not match the active batch");
    int32_t const batchSize = static_cast<int32_t>(chunkLengths.size());
    Coords const selectShape = packedTokenLayout ? Coords{1, batchSize} : Coords{batchSize, 1};
    ELLM_CHECK(io.selectTokenIndices.reshape(selectShape), "Phase prefill select-token reshape failed");
    ELLM_CHECK(io.contextLengths.reshape({batchSize}), "Phase prefill context-length reshape failed");
    ELLM_CHECK(io.hostSelectTokenIndices.reshape(selectShape), "Phase prefill host select-token reshape failed");
    ELLM_CHECK(io.hostContextLengths.reshape({batchSize}), "Phase prefill host context-length reshape failed");

    int64_t* selectTokenIndices = io.hostSelectTokenIndices.dataPointer<int64_t>();
    int32_t* contextLengths = io.hostContextLengths.dataPointer<int32_t>();
    int64_t packedTokenOffset{};
    for (int32_t row = 0; row < batchSize; ++row)
    {
        int32_t const chunkLength = chunkLengths[static_cast<size_t>(row)];
        ELLM_CHECK(chunkLength > 0, "Phase prefill chunk length must be positive");
        int32_t const stableSlot = mActiveStableSlots[static_cast<size_t>(row)];
        int32_t const resultingLength = mOwnership.length(stableSlot) + chunkLength;
        ELLM_CHECK(resultingLength <= mOwnership.config().maxSequenceLength,
            "Phase prefill metadata exceeds the stable slot sequence capacity");
        selectTokenIndices[row] = packedTokenLayout ? packedTokenOffset + chunkLength - 1 : chunkLength - 1;
        // The prefill binding carries the valid Q length for this invocation,
        // not the resulting cumulative KV length. The attention plugin combines
        // it with kvcache_start_index when it builds cumulative KV lengths. Using
        // resultingLength here makes a continuation chunk overstate cuQSeqLens
        // and can make packed multi-row FMHA read an unused page-table entry.
        contextLengths[row] = chunkLength;
        packedTokenOffset += chunkLength;
    }
    CUDA_CHECK(cudaMemcpyAsync(io.selectTokenIndices.rawPointer(), io.hostSelectTokenIndices.rawPointer(),
        static_cast<size_t>(batchSize) * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(io.contextLengths.rawPointer(), io.hostContextLengths.rawPointer(),
        static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    mMemoryStats.prefillMetadataH2DOperations += 2U;
    mMemoryStats.prefillMetadataH2DBytes += static_cast<size_t>(batchSize) * (sizeof(int64_t) + sizeof(int32_t));
}

void PhaseKVActiveView::prepareDecodeMetadata(PipelineIO& io, cudaStream_t stream) const
{
    ELLM_CHECK(mPrepared, "Phase KV active view is not prepared");
    int32_t const batchSize = static_cast<int32_t>(mActiveStableSlots.size());
    ELLM_CHECK(io.selectTokenIndices.reshape({batchSize, 1}), "Phase decode select-token reshape failed");
    ELLM_CHECK(io.contextLengths.reshape({batchSize}), "Phase decode context-length reshape failed");
    ELLM_CHECK(io.hostContextLengths.reshape({batchSize}), "Phase decode host context-length reshape failed");

    CUDA_CHECK(cudaMemsetAsync(
        io.selectTokenIndices.rawPointer(), 0, static_cast<size_t>(batchSize) * sizeof(int64_t), stream));
    ++mMemoryStats.decodeMemsetOperations;
    mMemoryStats.decodeMemsetBytes += static_cast<size_t>(batchSize) * sizeof(int64_t);
    int32_t* contextLengths = io.hostContextLengths.dataPointer<int32_t>();
    for (int32_t row = 0; row < batchSize; ++row)
    {
        int32_t const stableSlot = mActiveStableSlots[static_cast<size_t>(row)];
        contextLengths[row] = mOwnership.length(stableSlot) + 1;
    }
    CUDA_CHECK(cudaMemcpyAsync(io.contextLengths.rawPointer(), io.hostContextLengths.rawPointer(),
        static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    ++mMemoryStats.decodeMetadataH2DOperations;
    mMemoryStats.decodeMetadataH2DBytes += static_cast<size_t>(batchSize) * sizeof(int32_t);
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

PhaseKVMemoryStats const& PhaseKVActiveView::memoryStats() const noexcept
{
    return mMemoryStats;
}

KVPageTableUploadStats const& PhaseKVActiveView::pageTableUploadStats() const noexcept
{
    return mPageTable.uploadStats();
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
