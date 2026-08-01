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

#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

#include <cstdint>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

namespace
{

std::shared_ptr<Tensor> loadPleTable(std::filesystem::path const& engineDir, LLMEngineConfig const& config,
    cudaStream_t stream)
{
    std::filesystem::path const plePath = engineDir / binding_names::kPleEmbeddingFileName;
    std::vector<Tensor> pleTensors;
    ELLM_CHECK(safetensors::loadSafetensors(plePath, pleTensors, stream),
        "Failed to load " + std::string(binding_names::kPleEmbeddingFileName)
            + " from model directory: " + engineDir.string());
    ELLM_CHECK(pleTensors.size() == 1, "ple_embedding.safetensors must contain exactly one tensor named weight");
    ELLM_CHECK(pleTensors[0].getName() == "weight", "ple_embedding.safetensors tensor must be named weight");

    auto const pleShape = pleTensors[0].getShape();
    ELLM_CHECK(pleShape.getNumDims() == 2, "PLE table must be 2D [vocab, num_layers * hidden]");
    ELLM_CHECK(pleShape[1] == static_cast<int64_t>(config.numPleInputs) * config.pleHiddenSize,
        "PLE table second dimension must equal num_ple_inputs * ple_hidden_size");
    ELLM_CHECK(pleTensors[0].getDataType() == nvinfer1::DataType::kHALF
            || pleTensors[0].getDataType() == nvinfer1::DataType::kBF16,
        "PLE table must be FP16 or BF16");
    return std::make_shared<Tensor>(std::move(pleTensors[0]));
}

} // namespace

Gemma4EmbeddingPreprocessor::Gemma4EmbeddingPreprocessor(std::filesystem::path const& engineDir,
    LLMEngineConfig const& config, int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream)
    : mConfig(config)
    , mPleTable(loadPleTable(engineDir, config, stream))
{
    ELLM_CHECK(mConfig.pleEnabled, "Gemma4EmbeddingPreprocessor constructed while PLE is disabled");
    initializeOutputs(maxBatchSize, maxSeqLen, tensorMap);
}

Gemma4EmbeddingPreprocessor::Gemma4EmbeddingPreprocessor(LLMEngineConfig const& config,
    std::shared_ptr<Tensor> pleTable, int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap)
    : mConfig(config)
    , mPleTable(std::move(pleTable))
{
    ELLM_CHECK(mPleTable != nullptr, "Gemma4 PLE sibling requires a shared immutable table");
    initializeOutputs(maxBatchSize, maxSeqLen, tensorMap);
}

void Gemma4EmbeddingPreprocessor::initializeOutputs(
    int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap)
{
    ELLM_CHECK(maxBatchSize > 0, "Gemma4EmbeddingPreprocessor requires positive max batch size");
    ELLM_CHECK(maxSeqLen > 0, "Gemma4EmbeddingPreprocessor requires positive max sequence length");

    mPleOutputBuffer = Tensor({mConfig.numPleInputs, maxBatchSize, maxSeqLen, mConfig.pleHiddenSize}, DeviceType::kGPU,
        mPleTable->getDataType(), "Gemma4EmbeddingPreprocessor::mPleOutputBuffer");

    mPleOutputViews.reserve(mConfig.numPleInputs);
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews.emplace_back(makeOutputViewForLayer(idx, maxBatchSize, maxSeqLen));
    }
    bindOutputs(tensorMap);

    LOG_INFO("Initialized Gemma4 PLE preprocessor: table=%s outputBuffer=%s numPleInputs=%d pleHiddenSize=%d",
        mPleTable->getShape().formatString().c_str(), mPleOutputBuffer.getShape().formatString().c_str(),
        mConfig.numPleInputs, mConfig.pleHiddenSize);
}

std::unique_ptr<Gemma4EmbeddingPreprocessor> Gemma4EmbeddingPreprocessor::createSibling(
    int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap) const
{
    return std::unique_ptr<Gemma4EmbeddingPreprocessor>(
        new Gemma4EmbeddingPreprocessor(mConfig, mPleTable, maxBatchSize, maxSeqLen, tensorMap));
}

void Gemma4EmbeddingPreprocessor::bindOutputs(TensorMap& tensorMap)
{
    for (Tensor& output : mPleOutputViews)
    {
        tensorMap.set(output.getName(), output);
    }
}

Tensor Gemma4EmbeddingPreprocessor::makeOutputViewForLayer(int32_t layerIdx, int64_t batchSize, int64_t seqLen)
{
    ELLM_CHECK(layerIdx >= 0 && layerIdx < mConfig.numPleInputs, "Gemma4 PLE layer index out of range");
    auto const outputShape = mPleOutputBuffer.getShape();
    ELLM_CHECK(batchSize > 0, "Gemma4 PLE batch size must be positive");
    ELLM_CHECK(seqLen > 0, "Gemma4 PLE sequence length must be positive");
    ELLM_CHECK(batchSize <= outputShape[1], "Gemma4 PLE batch size exceeds buffer capacity");
    ELLM_CHECK(seqLen <= outputShape[2], "Gemma4 PLE sequence length exceeds buffer capacity");

    int64_t const layerOutputCapacityBytes = outputShape[1] * outputShape[2] * mConfig.pleHiddenSize
        * static_cast<int64_t>(utils::getTypeSize(mPleTable->getDataType()));
    void* const layerOutputPtr
        = static_cast<void*>(static_cast<char*>(mPleOutputBuffer.rawPointer()) + layerIdx * layerOutputCapacityBytes);
    return Tensor(layerOutputPtr, Coords{batchSize, seqLen, mConfig.pleHiddenSize}, DeviceType::kGPU,
        mPleTable->getDataType(), binding_names::formatPleTokenEmbedsName(layerIdx));
}

void Gemma4EmbeddingPreprocessor::reshapeOutputs(int64_t batchSize, int64_t seqLen)
{
    for (int32_t idx = 0; idx < mConfig.numPleInputs; ++idx)
    {
        mPleOutputViews[idx] = makeOutputViewForLayer(idx, batchSize, seqLen);
    }
}

void Gemma4EmbeddingPreprocessor::embed(Tensor const& tokenIds, cudaStream_t stream)
{
    auto const tokenShape = tokenIds.getShape();
    ELLM_CHECK(tokenShape.getNumDims() == 2, "Gemma4 PLE token IDs must be [batch, seq_len]");
    reshapeOutputs(tokenShape[0], tokenShape[1]);
    kernel::gemma4PleGather(tokenIds, *mPleTable, mPleOutputBuffer, mConfig.numPleInputs, mConfig.pleHiddenSize,
        mConfig.imageTokenId, mConfig.audioTokenId, stream);
}

void const* Gemma4EmbeddingPreprocessor::tableDataIdentity() const noexcept
{
    return mPleTable->rawPointer();
}

void const* Gemma4EmbeddingPreprocessor::outputDataIdentity() const noexcept
{
    return mPleOutputBuffer.rawPointer();
}

} // namespace rt
} // namespace trt_edgellm
