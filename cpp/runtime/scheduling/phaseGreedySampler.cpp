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

#include "runtime/scheduling/phaseGreedySampler.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

PhaseGreedySampler::PhaseGreedySampler(
    int32_t maxBatchSize, int32_t vocabSize, std::vector<int32_t> eosTokenIds, std::string const& name)
    : mMaxBatchSize(maxBatchSize)
    , mVocabSize(vocabSize)
    , mEosTokenIds(std::move(eosTokenIds))
    , mSelectedTokenIds({maxBatchSize, 1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, name + "_selected_ids")
    , mHostSelectedTokenIds({maxBatchSize}, DeviceType::kCPU, nvinfer1::DataType::kINT32, name + "_host_selected_ids")
    , mWorkspace({static_cast<int64_t>(getSelectAllTopKWorkspaceSize(maxBatchSize, vocabSize, 1))}, DeviceType::kGPU,
          nvinfer1::DataType::kINT8, name + "_workspace")
{
    check::check(mMaxBatchSize > 0, "Phase greedy sampler max batch size must be positive.");
    check::check(mVocabSize > 0, "Phase greedy sampler vocabulary size must be positive.");
    check::check(std::all_of(mEosTokenIds.begin(), mEosTokenIds.end(),
                     [this](int32_t tokenId) { return tokenId >= 0 && tokenId < mVocabSize; }),
        "Phase greedy sampler EOS token is outside the output vocabulary.");
}

void PhaseGreedySampler::enqueue(Tensor const& logits, int32_t batchSize, cudaStream_t stream)
{
    check::check(mPendingBatchSize == 0, "Phase greedy sampler already has an in-flight selection.");
    check::check(batchSize > 0 && batchSize <= mMaxBatchSize, "Phase greedy sampler batch size is invalid.");
    Coords const shape = logits.getShape();
    check::check(logits.getDeviceType() == DeviceType::kGPU && logits.getDataType() == nvinfer1::DataType::kFLOAT,
        "Phase greedy sampler logits must be GPU FP32.");
    check::check(shape.getNumDims() == 2 && shape[0] == batchSize && shape[1] == mVocabSize,
        "Phase greedy sampler logits shape must be [batch, vocab].");
    check::check(mSelectedTokenIds.reshape({batchSize, 1}), "Phase selected token reshape failed.");
    check::check(mHostSelectedTokenIds.reshape({batchSize}), "Phase host selected token reshape failed.");
    selectAllTopK(logits, std::nullopt, mSelectedTokenIds, 1, mWorkspace, stream);
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mSelectedTokenIds.rawPointer(),
        batchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    mPendingBatchSize = batchSize;
}

void PhaseGreedySampler::completeRow(
    DecodingInferenceContext& context, int32_t row, int32_t tokenId, int32_t maxGenerateLength) const
{
    check::check(row >= 0 && row < context.activeBatchSize, "Phase sampled row is outside the active batch.");
    size_t const index = static_cast<size_t>(row);
    check::check(!context.finishedStates[index], "Cannot append a token to a finished phase row.");
    check::check(maxGenerateLength > 0, "Phase sampled context has no generation budget.");
    context.tokenIds[index].push_back(tokenId);
    ++context.currentGenerateLengths[index];
    bool const reachedEos = std::find(mEosTokenIds.begin(), mEosTokenIds.end(), tokenId) != mEosTokenIds.end();
    bool const reachedLength = context.currentGenerateLengths[index] >= maxGenerateLength;
    context.finishedStates[index] = reachedEos || reachedLength;
}

void PhaseGreedySampler::finishCompletion(int32_t batchSize)
{
    check::check(mPendingBatchSize == batchSize, "Phase sampling completion batch does not match its enqueue.");
    mPendingBatchSize = 0;
}

void PhaseGreedySampler::completePrefill(PhasePrefillContextBatchAdapter const& adapter)
{
    check::check(adapter.batchSize() == mPendingBatchSize, "Phase prefill sampling batch mismatch.");
    int32_t const* selected = mHostSelectedTokenIds.dataPointer<int32_t>();
    for (int32_t packedRow = 0; packedRow < adapter.batchSize(); ++packedRow)
    {
        PhasePrefillContextRow const& row = adapter.rows()[static_cast<size_t>(packedRow)];
        if (row.tokenOffset + row.tokenCount == row.promptTokenCount)
        {
            completeRow(*row.context, row.contextRow, selected[packedRow], row.context->maxGenerateLength);
        }
    }
    finishCompletion(adapter.batchSize());
}

void PhaseGreedySampler::completeDecode(DecodingInferenceContext& context)
{
    check::check(context.activeBatchSize == mPendingBatchSize, "Phase decode sampling batch mismatch.");
    int32_t const* selected = mHostSelectedTokenIds.dataPointer<int32_t>();
    for (int32_t row = 0; row < context.activeBatchSize; ++row)
    {
        completeRow(context, row, selected[row], context.maxGenerateLength);
    }
    finishCompletion(context.activeBatchSize);
}

void PhaseGreedySampler::completeDecode(PhaseContextBatchAdapter& adapter)
{
    DecodingInferenceContext& packed = adapter.packedContext();
    check::check(packed.activeBatchSize == mPendingBatchSize, "Phase decode sampling batch mismatch.");
    check::check(static_cast<int32_t>(adapter.rows().size()) == mPendingBatchSize,
        "Phase decode sampling source-row mismatch.");
    int32_t const* selected = mHostSelectedTokenIds.dataPointer<int32_t>();
    for (int32_t row = 0; row < packed.activeBatchSize; ++row)
    {
        PhaseContextRow const& source = adapter.rows()[static_cast<size_t>(row)];
        completeRow(packed, row, selected[row], source.context->maxGenerateLength);
    }
    finishCompletion(packed.activeBatchSize);
}

bool PhaseGreedySampler::pending() const noexcept
{
    return mPendingBatchSize > 0;
}

} // namespace rt
} // namespace trt_edgellm
