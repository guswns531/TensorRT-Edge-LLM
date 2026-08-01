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

#include "runtime/scheduling/phaseContextBatchAdapter.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"

#include <set>
#include <unordered_set>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

PhaseContextBatchAdapter::PhaseContextBatchAdapter(
    int32_t maxBatchSize, HybridCacheManager& cacheManager, TensorMap& tensorMap, std::string const& name)
    : mMaxBatchSize(maxBatchSize)
    , mCacheManager(cacheManager)
    , mTensorMap(tensorMap)
    , mBatchState(maxBatchSize, name)
{
    check::check(mMaxBatchSize > 0, "Phase context adapter max batch size must be positive.");
    static_cast<void>(mCacheManager.getGlobalKVCacheLengths());
}

PhaseContextBatchAdapter::~PhaseContextBatchAdapter() noexcept
{
    restoreBindings();
}

void PhaseContextBatchAdapter::validateSourceRow(PhaseContextRow const& row) const
{
    check::check(row.context != nullptr, "Phase context row has no source context.");
    check::check(row.contextRow >= 0 && row.contextRow < row.context->activeBatchSize,
        "Phase context source row is outside the active batch.");
    check::check(row.kvLength >= 0, "Phase context row KV length must be non-negative.");

    size_t const sourceRow = static_cast<size_t>(row.contextRow);
    check::check(sourceRow < row.context->systemPrompts.size(), "Source context system prompt row is missing.");
    check::check(sourceRow < row.context->rawBatchedInputIds.size(), "Source context raw input row is missing.");
    check::check(sourceRow < row.context->tokenIds.size(), "Source context token row is missing.");
    check::check(sourceRow < row.context->currentGenerateLengths.size(), "Source context generation row is missing.");
    check::check(sourceRow < row.context->effectivePrefillLengths.size(), "Source context prefill row is missing.");
    check::check(sourceRow < row.context->finishedStates.size(), "Source context finish row is missing.");
    check::check(sourceRow < row.context->batchIndexMapping.size(), "Source context batch mapping row is missing.");
    check::check(sourceRow < row.context->slotStreams.size(), "Source context streaming row is missing.");
    check::check(sourceRow < row.context->stopStringsPerSlot.size(), "Source context stop-string row is missing.");
    check::check(sourceRow < row.context->logitBiasPerSlot.size(), "Source context logit-bias row is missing.");
    check::check(sourceRow < row.context->stepLogprobs.size(), "Source context logprobs row is missing.");

    check::check(!row.context->finishedStates[sourceRow], "Finished source rows cannot be packed for decode.");
    check::check(!row.context->tokenIds[sourceRow].empty(), "Decode source row has no input token.");
    check::check(!row.context->visualEmbeddings.has_value() && !row.context->audioEmbeddings.has_value()
            && row.context->deepstackFeatures.empty(),
        "Phase context adapter v1 supports text-only decode.");
    check::check(row.context->loraWeightsName.empty(), "Phase context adapter v1 does not support LoRA.");
    check::check(row.context->numLogprobs == 0, "Phase context adapter v1 does not support logprobs.");
    check::check(!row.context->hasLogitBias && row.context->logitBiasPerSlot[sourceRow].empty(),
        "Phase context adapter v1 does not support logit bias.");
    check::check(!row.context->outputThinkerEmbeddings,
        "Phase context adapter v1 does not support thinker hidden-state output.");
    check::check(!row.context->onTokenGenerated.has_value(),
        "Phase context adapter v1 does not support per-request token callbacks.");
    check::check(row.context->layerDebugger == nullptr, "Phase context adapter v1 does not support layer debugging.");
    check::check(row.context->phaseBatchState == nullptr,
        "A phase-packed context cannot be nested in another phase context batch.");
    check::check(row.context->slotStreams[sourceRow].channel == nullptr,
        "Phase context adapter v1 does not support streaming channels.");
}

void PhaseContextBatchAdapter::packDecode(std::vector<PhaseContextRow> const& rows, cudaStream_t stream)
{
    check::check(!mPacked, "A phase context batch is already packed.");
    check::check(!rows.empty(), "Phase context adapter cannot pack an empty batch.");
    check::check(
        static_cast<int32_t>(rows.size()) <= mMaxBatchSize, "Phase context batch exceeds its configured maximum.");

    std::unordered_set<uint64_t> requestIds;
    std::unordered_set<int32_t> slots;
    std::set<std::pair<DecodingInferenceContext*, int32_t>> sourceRows;
    for (PhaseContextRow const& row : rows)
    {
        validateSourceRow(row);
        check::check(requestIds.insert(row.requestId).second, "Phase context batch contains a duplicate request ID.");
        check::check(slots.insert(row.kvSlotId).second, "Phase context batch contains a duplicate KV slot.");
        check::check(sourceRows.insert({row.context, row.contextRow}).second,
            "Phase context batch contains a duplicate source row.");
    }

    DecodingInferenceContext const& first = *rows.front().context;
    for (PhaseContextRow const& row : rows)
    {
        DecodingInferenceContext const& source = *row.context;
        check::check(source.temperature == first.temperature && source.topP == first.topP && source.topK == first.topK,
            "Phase context decode rows have incompatible sampling parameters.");
        check::check(source.maxGenerateLength == first.maxGenerateLength,
            "Phase context decode rows have incompatible maximum generation lengths.");
    }

    mPackedContext = DecodingInferenceContext{};
    mPackedContext.initialize(static_cast<int32_t>(rows.size()), first.maxGenerateLength, std::nullopt,
        OptionalInputTensors{}, first.loraWeightsName, stream);
    mPackedContext.temperature = first.temperature;
    mPackedContext.topP = first.topP;
    mPackedContext.topK = first.topK;
    mPackedContext.numLogprobs = 0;
    mPackedContext.generationRound = first.generationRound;

    mRows = rows;
    mWorkItems.clear();
    mWorkItems.reserve(rows.size());
    for (size_t packedRow = 0; packedRow < rows.size(); ++packedRow)
    {
        PhaseContextRow const& row = rows[packedRow];
        DecodingInferenceContext const& source = *row.context;
        size_t const sourceRow = static_cast<size_t>(row.contextRow);
        mPackedContext.tokenIds[packedRow] = {source.tokenIds[sourceRow].back()};
        mPackedContext.currentGenerateLengths[packedRow] = source.currentGenerateLengths[sourceRow];
        mPackedContext.effectivePrefillLengths[packedRow] = source.effectivePrefillLengths[sourceRow];
        mPackedContext.finishedStates[packedRow] = source.finishedStates[sourceRow];
        mWorkItems.push_back({row.requestId, row.kvLength, row.kvSlotId});
    }

    mBatchState.prepare(mWorkItems, mCacheManager, stream);
    mBatchState.bind(mTensorMap);
    mPackedContext.phaseBatchState = &mBatchState;
    mPacked = true;
}

void PhaseContextBatchAdapter::scatterDecode()
{
    check::check(mPacked, "No phase context decode batch is available to scatter.");
    check::check(mPackedContext.activeBatchSize == static_cast<int32_t>(mRows.size()),
        "Packed phase context changed batch size before scatter.");

    for (size_t packedRow = 0; packedRow < mRows.size(); ++packedRow)
    {
        PhaseContextRow const& row = mRows[packedRow];
        validateSourceRow(row);
        DecodingInferenceContext& destination = *row.context;
        size_t const destinationRow = static_cast<size_t>(row.contextRow);
        std::vector<int32_t> const& packedTokens = mPackedContext.tokenIds[packedRow];
        check::check(!packedTokens.empty() && packedTokens.front() == destination.tokenIds[destinationRow].back(),
            "Packed decode token prefix no longer matches its source context.");
        destination.tokenIds[destinationRow].insert(
            destination.tokenIds[destinationRow].end(), packedTokens.begin() + 1, packedTokens.end());
        destination.currentGenerateLengths[destinationRow] = mPackedContext.currentGenerateLengths[packedRow];
        destination.finishedStates[destinationRow] = mPackedContext.finishedStates[packedRow];
    }

    restoreBindings();
    mRows.clear();
    mWorkItems.clear();
    mPackedContext = DecodingInferenceContext{};
    mPacked = false;
}

void PhaseContextBatchAdapter::restoreBindings() noexcept
{
    if (!mPacked)
    {
        return;
    }
    try
    {
        mTensorMap.set(binding_names::kKVSlotIds, mCacheManager.getKVSlotIds());
        mTensorMap.set(binding_names::kKVCacheStartIndex, mCacheManager.getKVCacheLengths());
    }
    catch (...)
    {
    }
}

DecodingInferenceContext& PhaseContextBatchAdapter::packedContext()
{
    check::check(mPacked, "No phase context decode batch is packed.");
    return mPackedContext;
}

std::vector<PhaseWorkItem> const& PhaseContextBatchAdapter::workItems() const noexcept
{
    return mWorkItems;
}

bool PhaseContextBatchAdapter::packed() const noexcept
{
    return mPacked;
}

} // namespace rt
} // namespace trt_edgellm
