/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/preprocess/embeddingPreprocessor.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include "common/logger.h"
#include "kernels/embeddingKernels/embeddingKernels.h"

#include <cstddef>
#include <cstring>

namespace trt_edgellm
{
namespace rt
{

EmbeddingPreprocessor::EmbeddingPreprocessor(EmbeddingData const& embedding, LLMEngineConfig const& config)
    : mEmbedding(embedding)
    , mConfig(config)
{
}

void EmbeddingPreprocessor::embed(Tensor const& tokenIds, OptionalInputTensor visionEmbeds,
    OptionalInputTensor audioEmbeds, PipelineIO& io, cudaStream_t stream, OptionalInputTensor precomputedIndices,
    int32_t const* imageBaseOffsets, int32_t const* audioBaseOffsets)
{
    if (visionEmbeds.has_value() || audioEmbeds.has_value())
    {
        std::optional<int32_t> audioTokenOpt
            = (mConfig.audioTokenId >= 0) ? std::optional{mConfig.audioTokenId} : std::nullopt;
        std::optional<int32_t> imageTokenOpt
            = (mConfig.imageTokenId >= 0) ? std::optional{mConfig.imageTokenId} : std::nullopt;

        if (precomputedIndices.has_value())
        {
            mIndicesPtr = &precomputedIndices.value().get();
        }
        else
        {
            mOwnedIndices = Tensor(tokenIds.getShape(), DeviceType::kGPU, tokenIds.getDataType());
            kernel::generateMultimodalIndices(
                tokenIds, mOwnedIndices, imageTokenOpt, audioTokenOpt, stream, imageBaseOffsets, audioBaseOffsets);
            mIndicesPtr = &mOwnedIndices;
        }

        kernel::embeddingLookup(tokenIds, mEmbedding.table, mEmbedding.scalesAsOptional(), io.inputsEmbeds, stream,
            std::optional{std::ref(*mIndicesPtr)}, imageTokenOpt, visionEmbeds, audioTokenOpt, audioEmbeds);
    }
    else
    {
        kernel::embeddingLookup(tokenIds, mEmbedding.table, mEmbedding.scalesAsOptional(), io.inputsEmbeds, stream);
    }
}

namespace
{
void materializeSegments(
    OptionalInputTensors const& segments, Tensor& destination, char const* name, cudaStream_t stream)
{
    check::check(!segments.empty(), "Segmented embedding input cannot be empty");
    int64_t hiddenSize{};
    int64_t totalRows{};
    for (Tensor const& segment : segments)
    {
        Coords const shape = segment.getShape();
        check::check(
            shape.getNumDims() == 2 && shape[0] > 0, "Segmented embedding inputs must be non-empty 2D tensors");
        hiddenSize = hiddenSize == 0 ? shape[1] : hiddenSize;
        check::check(shape[1] == hiddenSize, "Segmented embedding hidden dimensions must match");
        check::check(segment.getDataType() == nvinfer1::DataType::kHALF, "Segmented embedding inputs must use FP16");
        totalRows += shape[0];
    }
    Coords const destinationShape{totalRows, hiddenSize};
    int64_t const requiredBytes = destinationShape.volume() * static_cast<int64_t>(sizeof(half));
    if (destination.isEmpty() || destination.getMemoryCapacity() < requiredBytes)
    {
        destination = Tensor(destinationShape, DeviceType::kGPU, nvinfer1::DataType::kHALF, name);
    }
    else
    {
        check::check(destination.reshape(destinationShape), "Failed to reshape segmented embedding scratch");
    }
    auto* output = destination.dataPointer<uint8_t>();
    size_t offset{};
    for (Tensor const& segment : segments)
    {
        size_t const bytes = static_cast<size_t>(segment.getShape().volume()) * sizeof(half);
        CUDA_CHECK(cudaMemcpyAsync(output + offset, segment.rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
        offset += bytes;
    }
}
} // namespace

void EmbeddingPreprocessor::embedSegmentedVision(
    Tensor const& tokenIds, OptionalInputTensors const& visionSegments, PipelineIO& io, cudaStream_t stream)
{
    materializeSegments(visionSegments, mSegmentedVisionScratch, "segmented_vision_scratch", stream);
    embed(tokenIds, std::cref(mSegmentedVisionScratch), std::nullopt, io, stream);
}

OptionalInputTensors EmbeddingPreprocessor::assembleDeepstack(
    Tensor const& tokenIds, OptionalInputTensors const& features, PipelineIO& io, cudaStream_t stream)
{
    OptionalInputTensors deepstackEmbeds{};

    if (features.empty())
    {
        return deepstackEmbeds;
    }

    auto const inputShape = tokenIds.getShape();
    int64_t const activeBatchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];

    check::check(mIndicesPtr != nullptr && mIndicesPtr->getShape().volume() == inputShape.volume(),
        "assembleDeepstack requires embed() to have computed multimodal indices for the same tokens first");
    OptionalInputTensor deepstackMultimodalIndices{std::ref(*mIndicesPtr)};

    for (int32_t idx = 0; idx < static_cast<int32_t>(features.size()); ++idx)
    {
        Tensor const& featureTensor = features[idx].get();

        // Reshape the output slot to match the current batch/sequence dimensions
        check::check(
            io.deepstackEmbeds[idx].reshape({activeBatchSize, seqLen, mConfig.hiddenSize}), "Tensor reshape failed");
        kernel::assembleDeepstackEmbedding(
            tokenIds, featureTensor, io.deepstackEmbeds[idx], stream, mConfig.imageTokenId, deepstackMultimodalIndices);

        deepstackEmbeds.push_back(std::ref(io.deepstackEmbeds[idx]));
    }

    return deepstackEmbeds;
}

void EmbeddingPreprocessor::prepareDeepstack(
    Tensor const& tokenIds, OptionalInputTensors const& features, PipelineIO& io, cudaStream_t stream)
{
    if (mConfig.numDeepstackFeatures == 0)
    {
        return;
    }

    if (!features.empty())
    {
        assembleDeepstack(tokenIds, features, io, stream);
        return;
    }

    // Text-only request on a VLM engine: zero the deepstack slots so the engine reads known-zero bytes.
    LOG_DEBUG("Deepstack features configured but not available for this text-only request, using zero tensors.");
    auto const inputShape = tokenIds.getShape();
    int64_t const activeBatchSize = inputShape[0];
    int64_t const seqLen = inputShape[1];
    for (int32_t idx = 0; idx < mConfig.numDeepstackFeatures; ++idx)
    {
        check::check(
            io.deepstackEmbeds[idx].reshape({activeBatchSize, seqLen, mConfig.hiddenSize}), "Tensor reshape failed");
        CUDA_CHECK(cudaMemsetAsync(
            io.deepstackEmbeds[idx].rawPointer(), 0, io.deepstackEmbeds[idx].getMemoryCapacity(), stream));
    }
}

void EmbeddingPreprocessor::prepareSegmentedDeepstack(Tensor const& tokenIds,
    std::vector<OptionalInputTensors> const& featureSegments, PipelineIO& io, cudaStream_t stream)
{
    if (mConfig.numDeepstackFeatures == 0)
    {
        return;
    }
    check::check(featureSegments.size() == static_cast<size_t>(mConfig.numDeepstackFeatures),
        "Segmented deepstack feature count does not match model configuration");
    mSegmentedDeepstackScratch.resize(featureSegments.size());
    OptionalInputTensors contiguousFeatures;
    contiguousFeatures.reserve(featureSegments.size());
    for (size_t index{}; index < featureSegments.size(); ++index)
    {
        materializeSegments(
            featureSegments[index], mSegmentedDeepstackScratch[index], "segmented_deepstack_scratch", stream);
        contiguousFeatures.push_back(std::cref(mSegmentedDeepstackScratch[index]));
    }
    prepareDeepstack(tokenIds, contiguousFeatures, io, stream);
}

} // namespace rt
} // namespace trt_edgellm
