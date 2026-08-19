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

#include "runtime/scheduling/packedPrefillActiveView.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/pagedKvTypes.h"

#include <algorithm>
#include <cstddef>

namespace trt_edgellm::rt
{

PackedPrefillActiveView::PackedPrefillActiveView(LLMEngineConfig const& config, KVPageTable& sourcePageTable,
    HybridCacheManager& cacheManager, TensorMap& tensorMap, PipelineIO& pipelineIO)
    : mConfig(config)
    , mSourcePageTable(sourcePageTable)
    , mCacheManager(cacheManager)
    , mTensorMap(tensorMap)
    , mPipelineIO(pipelineIO)
    , mActivePageTable(config.maxSupportedBatchSize, sourcePageTable.maxPagesPerSeq(), sourcePageTable.numPages())
    , mHostActiveLengths({config.maxSupportedBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32,
          "PackedPrefillActiveView::hostActiveLengths")
    , mDeviceActiveLengths({config.maxSupportedBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
          "PackedPrefillActiveView::deviceActiveLengths")
    , mHostGlobalIncrements({config.maxSupportedBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32,
          "PackedPrefillActiveView::hostGlobalIncrements")
    , mDeviceGlobalIncrements({config.maxSupportedBatchSize}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
          "PackedPrefillActiveView::deviceGlobalIncrements")
{
    ELLM_CHECK(config.packedPrefill, "Packed prefill active view requires a packed engine");
}

PackedPrefillActiveView::~PackedPrefillActiveView() noexcept
{
    restoreBindings();
}

void PackedPrefillActiveView::prepare(
    std::vector<int32_t> const& sourceRows, std::vector<int32_t> const& sourceLengths, cudaStream_t stream)
{
    ELLM_CHECK(!mPrepared, "Packed prefill active view is already prepared");
    ELLM_CHECK(!sourceRows.empty() && sourceRows.size() == sourceLengths.size(),
        "Packed prefill active rows and lengths must be non-empty and aligned");
    ELLM_CHECK(static_cast<int32_t>(sourceRows.size()) <= mConfig.maxSupportedBatchSize,
        "Packed prefill active wave exceeds the engine batch limit");

    std::vector<KVPageTableRowUpdate> updates;
    updates.reserve(sourceRows.size());
    std::vector<uint8_t> seen(static_cast<size_t>(mConfig.maxSupportedBatchSize), 0U);
    for (size_t activeRow = 0; activeRow < sourceRows.size(); ++activeRow)
    {
        int32_t const sourceRow = sourceRows[activeRow];
        ELLM_CHECK(sourceRow >= 0 && sourceRow < mConfig.maxSupportedBatchSize,
            "Packed prefill source row is outside the engine batch limit");
        ELLM_CHECK(seen[static_cast<size_t>(sourceRow)]++ == 0U, "Packed prefill source row is duplicated");
        ELLM_CHECK(sourceLengths[activeRow] >= 0 && sourceLengths[activeRow] <= mConfig.maxKVCacheCapacity,
            "Packed prefill source length is outside the KV capacity");
        int32_t const* pages = mSourcePageTable.hostRow(sourceRow);
        int32_t pageCount{};
        while (pageCount < mSourcePageTable.maxPagesPerSeq() && pages[pageCount] != kUNUSED_PAGE_ENTRY)
        {
            ++pageCount;
        }
        updates.push_back(KVPageTableRowUpdate{static_cast<int32_t>(activeRow), pages, pageCount});
    }

    mPreviousPageTable = mTensorMap.get(binding_names::kKVPageTable);
    mPreviousLengths = mTensorMap.get(binding_names::kKVCacheStartIndex);
    ELLM_CHECK(mPreviousPageTable != nullptr && mPreviousLengths != nullptr,
        "Packed prefill active view requires page-table and length bindings");
    mActivePageTable.setRows(updates);
    mActivePageTable.upload(stream);

    int64_t const activeCount = static_cast<int64_t>(sourceRows.size());
    ELLM_CHECK(mHostActiveLengths.reshape({activeCount}) && mDeviceActiveLengths.reshape({activeCount}),
        "Packed prefill active length reshape failed");
    std::copy(sourceLengths.begin(), sourceLengths.end(), mHostActiveLengths.dataPointer<int32_t>());
    CUDA_CHECK(cudaMemcpyAsync(mDeviceActiveLengths.rawPointer(), mHostActiveLengths.rawPointer(),
        sourceLengths.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    mTensorMap.set(binding_names::kKVPageTable, mActivePageTable.kernelView());
    mTensorMap.set(binding_names::kKVCacheStartIndex, mDeviceActiveLengths);
    // Packed v1 admits only text-only requests. Text M-RoPE rows are identical,
    // so the existing binding's first activeCount rows remain valid even when
    // the selected KV rows are non-contiguous.
    ELLM_CHECK(mConfig.ropeConfig.type != RopeType::kMRope
            || mTensorMap.get(binding_names::kRopeCosSin) == &mPipelineIO.mropeCosSin,
        "Packed prefill text M-RoPE binding was unexpectedly replaced");
    mSourceRows = sourceRows;
    mPrepared = true;
}

void PackedPrefillActiveView::commitChunkLengths(std::vector<int32_t> const& chunkLengths, cudaStream_t stream)
{
    ELLM_CHECK(mPrepared && chunkLengths.size() == mSourceRows.size(),
        "Packed prefill committed chunks must match the active wave");
    int32_t const globalBatchSize = mCacheManager.getActiveBatchSize();
    ELLM_CHECK(globalBatchSize > 0 && globalBatchSize <= mConfig.maxSupportedBatchSize,
        "Packed prefill global batch size is invalid");
    ELLM_CHECK(mHostGlobalIncrements.reshape({globalBatchSize}) && mDeviceGlobalIncrements.reshape({globalBatchSize}),
        "Packed prefill global increment reshape failed");
    int32_t* increments = mHostGlobalIncrements.dataPointer<int32_t>();
    std::fill(increments, increments + globalBatchSize, 0);
    for (size_t activeRow = 0; activeRow < mSourceRows.size(); ++activeRow)
    {
        ELLM_CHECK(chunkLengths[activeRow] > 0, "Packed prefill chunk length must be positive");
        increments[mSourceRows[activeRow]] = chunkLengths[activeRow];
    }
    CUDA_CHECK(cudaMemcpyAsync(mDeviceGlobalIncrements.rawPointer(), mHostGlobalIncrements.rawPointer(),
        static_cast<size_t>(globalBatchSize) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    mCacheManager.commitSequenceLength(mDeviceGlobalIncrements, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void PackedPrefillActiveView::complete()
{
    ELLM_CHECK(mPrepared, "Packed prefill active view is not prepared");
    restoreBindings();
    mSourceRows.clear();
    mPrepared = false;
}

bool PackedPrefillActiveView::prepared() const noexcept
{
    return mPrepared;
}

void PackedPrefillActiveView::restoreBindings() noexcept
{
    if (!mPrepared)
    {
        return;
    }
    try
    {
        mTensorMap.set(binding_names::kKVPageTable, *mPreviousPageTable);
        mTensorMap.set(binding_names::kKVCacheStartIndex, *mPreviousLengths);
        mPreviousPageTable = nullptr;
        mPreviousLengths = nullptr;
    }
    catch (...)
    {
    }
}

} // namespace trt_edgellm::rt
