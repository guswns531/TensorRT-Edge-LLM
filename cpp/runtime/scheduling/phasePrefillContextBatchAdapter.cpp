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

#include "runtime/scheduling/phasePrefillContextBatchAdapter.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <algorithm>
#include <set>
#include <unordered_set>

namespace trt_edgellm
{
namespace rt
{

PhasePrefillContextBatchAdapter::PhasePrefillContextBatchAdapter(int32_t maxBatchSize, int32_t maxChunkTokens,
    HybridCacheManager& cacheManager, TensorMap& tensorMap, std::string const& name, bool enableRaggedPrefill)
    : mMaxBatchSize(maxBatchSize)
    , mMaxChunkTokens(maxChunkTokens)
    , mCacheManager(cacheManager)
    , mTensorMap(tensorMap)
    , mHostTokenIds(
          {maxBatchSize, maxChunkTokens}, DeviceType::kCPU, nvinfer1::DataType::kINT32, name + "_host_token_ids")
    , mDeviceTokenIds({maxBatchSize, maxChunkTokens}, DeviceType::kGPU, nvinfer1::DataType::kINT32, name + "_token_ids")
    , mBatchState(maxBatchSize, name + "_batch", cacheManager.isIndexedKVCache())
    , mEnableRaggedPrefill(enableRaggedPrefill)
{
    check::check(mMaxBatchSize > 0, "Phase prefill adapter max batch size must be positive.");
    check::check(mMaxChunkTokens > 0, "Phase prefill adapter max chunk length must be positive.");
}

PhasePrefillContextBatchAdapter::~PhasePrefillContextBatchAdapter() noexcept
{
    restoreBindings();
}

void PhasePrefillContextBatchAdapter::validateRow(PhasePrefillContextRow const& row) const
{
    check::check(row.context != nullptr, "Phase prefill row has no source context.");
    check::check(row.contextRow >= 0 && row.contextRow < row.context->activeBatchSize,
        "Phase prefill source row is outside the active batch.");
    check::check(row.kvSlotId >= 0, "Phase prefill row has no physical KV slot.");
    check::check(row.tokenOffset >= 0 && row.tokenCount > 0, "Phase prefill token range is invalid.");
    check::check(
        row.promptTokenCount >= row.tokenOffset + row.tokenCount, "Phase prefill chunk exceeds its prompt length.");

    size_t const sourceRow = static_cast<size_t>(row.contextRow);
    check::check(sourceRow < row.context->rawBatchedInputIds.size(), "Phase prefill raw input row is missing.");
    std::vector<int32_t> const& prompt = row.context->rawBatchedInputIds[sourceRow];
    check::check(static_cast<int32_t>(prompt.size()) == row.promptTokenCount,
        "Phase prefill v1 requires the full text prompt without prefix-cache reuse.");
    check::check(
        !row.context->audioEmbeddings.has_value(), "Phase prefill adapter v1 does not support audio features.");
    if (row.context->visualEmbeddings.has_value())
    {
        check::check(row.tokenOffset == 0 && row.tokenCount == row.promptTokenCount,
            "Multimodal phase prefill must process the complete prompt atomically.");
    }
    check::check(row.context->loraWeightsName.empty(), "Phase prefill adapter v1 does not support LoRA.");
}

void PhasePrefillContextBatchAdapter::pack(std::vector<PhasePrefillContextRow> const& rows, cudaStream_t stream)
{
    check::check(!mPacked, "A phase prefill context batch is already packed.");
    check::check(!rows.empty(), "Phase prefill adapter cannot pack an empty batch.");
    check::check(
        static_cast<int32_t>(rows.size()) <= mMaxBatchSize, "Phase prefill batch exceeds its configured maximum.");

    int32_t const chunkLength = std::max_element(rows.begin(), rows.end(), [](auto const& lhs, auto const& rhs) {
        return lhs.tokenCount < rhs.tokenCount;
    })->tokenCount;
    check::check(chunkLength <= mMaxChunkTokens, "Phase prefill chunk exceeds its configured maximum.");
    std::unordered_set<uint64_t> requestIds;
    std::unordered_set<int32_t> slots;
    std::set<std::pair<DecodingInferenceContext*, int32_t>> sourceRows;
    for (PhasePrefillContextRow const& row : rows)
    {
        validateRow(row);
        check::check(mEnableRaggedPrefill || row.tokenCount == chunkLength,
            "Phase prefill batch mixes different chunk lengths without ragged prefill enabled.");
        check::check(requestIds.insert(row.requestId).second, "Phase prefill batch contains a duplicate request ID.");
        check::check(slots.insert(row.kvSlotId).second, "Phase prefill batch contains a duplicate KV slot.");
        check::check(sourceRows.insert({row.context, row.contextRow}).second,
            "Phase prefill batch contains a duplicate source row.");
    }
    bool const hasVisualEmbeddings = rows.front().context->visualEmbeddings.has_value();
    check::check(std::all_of(rows.begin(), rows.end(),
                     [hasVisualEmbeddings](PhasePrefillContextRow const& row) {
                         return row.context->visualEmbeddings.has_value() == hasVisualEmbeddings;
                     }),
        "Phase prefill batch cannot mix text-only and multimodal rows.");
    check::check(!hasVisualEmbeddings || rows.size() == 1,
        "Phase prefill adapter v1 supports one multimodal request per batch.");
    bool const hasDeepstack = !rows.front().context->deepstackFeatures.empty();
    check::check(std::all_of(rows.begin(), rows.end(),
                     [hasDeepstack](PhasePrefillContextRow const& row) {
                         return (!row.context->deepstackFeatures.empty()) == hasDeepstack;
                     }),
        "Phase prefill batch cannot mix deepstack and non-deepstack rows.");
    check::check(
        !hasDeepstack || rows.size() == 1, "Phase prefill adapter v1 supports one deepstack request per batch.");

    bool const initialChunk = rows.front().tokenOffset == 0;
    check::check(
        std::all_of(rows.begin(), rows.end(),
            [initialChunk](PhasePrefillContextRow const& row) { return (row.tokenOffset == 0) == initialChunk; }),
        "Phase prefill batch mixes initial and continuation chunks.");
    mBatchSize = static_cast<int32_t>(rows.size());
    mChunkLength = chunkLength;
    mInitialChunk = initialChunk;
    mStream = stream;
    check::check(mHostTokenIds.reshape({mBatchSize, mChunkLength}), "Host prefill token IDs reshape failed.");
    check::check(mDeviceTokenIds.reshape({mBatchSize, mChunkLength}), "Device prefill token IDs reshape failed.");

    int32_t* hostTokens = mHostTokenIds.dataPointer<int32_t>();
    std::fill_n(hostTokens, static_cast<size_t>(mBatchSize) * mChunkLength, 0);
    mRows = rows;
    mWorkItems.clear();
    mWorkItems.reserve(rows.size());
    for (int32_t packedRow = 0; packedRow < mBatchSize; ++packedRow)
    {
        PhasePrefillContextRow const& row = rows[static_cast<size_t>(packedRow)];
        std::vector<int32_t> const& prompt = row.context->rawBatchedInputIds[static_cast<size_t>(row.contextRow)];
        std::copy_n(prompt.begin() + row.tokenOffset, row.tokenCount,
            hostTokens + static_cast<size_t>(packedRow) * mChunkLength);
        mWorkItems.push_back({row.requestId, row.tokenCount, row.kvSlotId, row.tokenOffset, row.promptTokenCount});
    }
    CUDA_CHECK(cudaMemcpyAsync(mDeviceTokenIds.rawPointer(), mHostTokenIds.rawPointer(),
        static_cast<size_t>(mBatchSize) * mChunkLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

    mBatchState.prepare(mWorkItems, mCacheManager, stream);
    mPreviousSlotIds = mTensorMap.get(binding_names::kKVSlotIds);
    mPreviousLengths = mTensorMap.get(binding_names::kKVCacheStartIndex);
    check::check(mPreviousSlotIds != nullptr && mPreviousLengths != nullptr,
        "Phase prefill adapter requires existing KV slot and length bindings.");
    mBatchState.bind(mTensorMap);
    mPacked = true;
}

void PhasePrefillContextBatchAdapter::complete()
{
    check::check(mPacked, "No phase prefill context batch is packed.");
    restoreBindings();
    mRows.clear();
    mWorkItems.clear();
    mBatchSize = 0;
    mChunkLength = 0;
    mInitialChunk = false;
    mStream = nullptr;
    mPacked = false;
}

void PhasePrefillContextBatchAdapter::restoreBindings() noexcept
{
    if (!mPacked)
    {
        return;
    }
    try
    {
        mTensorMap.set(binding_names::kKVSlotIds, *mPreviousSlotIds);
        mTensorMap.set(binding_names::kKVCacheStartIndex, *mPreviousLengths);
        mPreviousSlotIds = nullptr;
        mPreviousLengths = nullptr;
    }
    catch (...)
    {
    }
}

Tensor& PhasePrefillContextBatchAdapter::tokenIds() noexcept
{
    return mDeviceTokenIds;
}

Tensor const& PhasePrefillContextBatchAdapter::hostTokenIds() const noexcept
{
    return mHostTokenIds;
}

OptionalInputTensor PhasePrefillContextBatchAdapter::visualEmbeddings() const noexcept
{
    if (!mPacked || mRows.empty())
    {
        return std::nullopt;
    }
    return mRows.front().context->visualEmbeddings;
}

OptionalInputTensors const& PhasePrefillContextBatchAdapter::deepstackFeatures() const noexcept
{
    static OptionalInputTensors const empty;
    if (!mPacked || mRows.empty())
    {
        return empty;
    }
    return mRows.front().context->deepstackFeatures;
}

OptionalInputTensor PhasePrefillContextBatchAdapter::mropeCosSin() const noexcept
{
    if (!mPacked || mRows.empty())
    {
        return std::nullopt;
    }
    return mRows.front().context->mropeCosSin;
}

PhaseBatchState& PhasePrefillContextBatchAdapter::phaseBatchState() noexcept
{
    return mBatchState;
}

std::vector<PhasePrefillContextRow> const& PhasePrefillContextBatchAdapter::rows() const noexcept
{
    return mRows;
}

std::vector<PhaseWorkItem> const& PhasePrefillContextBatchAdapter::workItems() const noexcept
{
    return mWorkItems;
}

int32_t PhasePrefillContextBatchAdapter::batchSize() const noexcept
{
    return mBatchSize;
}

int32_t PhasePrefillContextBatchAdapter::chunkLength() const noexcept
{
    return mChunkLength;
}

bool PhasePrefillContextBatchAdapter::initialChunk() const noexcept
{
    return mInitialChunk;
}

cudaStream_t PhasePrefillContextBatchAdapter::stream() const noexcept
{
    return mStream;
}

bool PhasePrefillContextBatchAdapter::packed() const noexcept
{
    return mPacked;
}

} // namespace rt
} // namespace trt_edgellm
