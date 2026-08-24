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

#include "runtime/scheduling/phaseVisionAdapter.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <utility>

namespace trt_edgellm::rt
{

PhaseVisionPayload::~PhaseVisionPayload() noexcept
{
    if (startEvent != nullptr)
    {
        static_cast<void>(cudaEventDestroy(startEvent));
    }
    if (readyEvent != nullptr)
    {
        static_cast<void>(cudaEventDestroy(readyEvent));
    }
}

size_t PhaseVisionPayload::byteSize() const noexcept
{
    auto tensorBytes = [](Tensor const& tensor) {
        return tensor.isEmpty()
            ? size_t{}
            : static_cast<size_t>(tensor.getShape().volume()) * utils::getTypeSize(tensor.getDataType());
    };
    size_t result = tensorBytes(outputEmbedding) + tensorBytes(mropeCosSin);
    for (Tensor const& feature : deepstackFeatures)
    {
        result += tensorBytes(feature);
    }
    return result;
}

PhaseVisionAdapter::PhaseVisionAdapter(
    MultimodalRunner& runner, tokenizer::Tokenizer const& tokenizer, LLMEngineConfig const& config, cudaStream_t stream)
    : mRunner(runner)
    , mTokenizer(tokenizer)
    , mConfig(config)
    , mStream(stream)
{
    ELLM_CHECK(mStream != nullptr, "Phase vision adapter requires an explicit CUDA stream");
    CUDA_DRIVER_CHECK(cuStreamGetCtx(mStream, &mCudaContext));
    ELLM_CHECK(mCudaContext != nullptr, "Phase vision stream has no CUDA context");
}

Tensor PhaseVisionAdapter::copyTensor(Tensor const& source, std::string const& name, cudaStream_t stream)
{
    ELLM_CHECK(!source.isEmpty() && source.getDeviceType() == DeviceType::kGPU,
        "Phase vision output must be a non-empty GPU tensor");
    Tensor result(source.getShape(), DeviceType::kGPU, source.getDataType(), name);
    size_t const bytes = static_cast<size_t>(source.getShape().volume()) * utils::getTypeSize(source.getDataType());
    CUDA_CHECK(cudaMemcpyAsync(result.rawPointer(), source.rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
    return result;
}

bool PhaseVisionAdapter::submit(uint64_t requestId, LLMGenerationRequest const& request)
{
    ELLM_CHECK(mRequests.empty(), "Phase vision adapter currently permits one in-flight encoder request");
    request.formattedRequests.resize(request.requests.size());
    for (size_t index = 0; index < request.requests.size(); ++index)
    {
        ELLM_CHECK(mTokenizer.applyChatTemplate(request.requests[index], request.formattedRequests[index],
                       request.applyChatTemplate, request.addGenerationPrompt, request.enableThinking),
            "Failed to format phase vision request");
    }
    auto payload = std::make_unique<PhaseVisionPayload>();
    CUDA_CHECK(cudaEventCreate(&payload->startEvent));
    CUDA_CHECK(cudaEventCreate(&payload->readyEvent));
    CUDA_CHECK(cudaEventRecord(payload->startEvent, mStream));
    if (mConfig.ropeConfig.type == RopeType::kMRope)
    {
        int64_t const activeBatchSize = static_cast<int64_t>(request.requests.size());
        ELLM_CHECK(activeBatchSize > 0 && activeBatchSize <= mConfig.maxSupportedBatchSize,
            "Phase vision M-RoPE batch is outside the engine profile");
        payload->mropeCosSin = Tensor({activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim},
            DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "phase_vision_mrope");
    }
    OptionalOutputTensor mrope
        = payload->mropeCosSin.isEmpty() ? std::nullopt : OptionalOutputTensor{std::ref(payload->mropeCosSin)};
    ELLM_CHECK(mRunner.preprocess(request, payload->tokenIds, &mTokenizer, mrope, mStream),
        "Phase vision preprocessing failed");
    ELLM_CHECK(mRunner.infer(mStream), "Phase vision inference failed");
    payload->outputEmbedding = copyTensor(mRunner.getOutputEmbedding(), "phase_vision_output", mStream);
    for (Tensor const& feature : mRunner.getDeepstackFeatures())
    {
        payload->deepstackFeatures.push_back(copyTensor(feature, "phase_vision_deepstack", mStream));
    }
    CUDA_CHECK(cudaEventRecord(payload->readyEvent, mStream));
    return mRequests.emplace(requestId, std::move(payload)).second;
}

bool PhaseVisionAdapter::ready(uint64_t requestId) const
{
    auto const it = mRequests.find(requestId);
    ELLM_CHECK(it != mRequests.end(), "Unknown phase vision request");
    cudaError_t const status = cudaEventQuery(it->second->readyEvent);
    if (status == cudaErrorNotReady)
    {
        return false;
    }
    CUDA_CHECK(status);
    return true;
}

std::unique_ptr<PhaseVisionPayload> PhaseVisionAdapter::take(uint64_t requestId)
{
    auto it = mRequests.find(requestId);
    ELLM_CHECK(it != mRequests.end(), "Unknown phase vision request");
    ELLM_CHECK(ready(requestId), "Phase vision request is not complete");
    CUDA_CHECK(cudaEventElapsedTime(&it->second->encoderGpuMs, it->second->startEvent, it->second->readyEvent));
    std::unique_ptr<PhaseVisionPayload> result = std::move(it->second);
    mRequests.erase(it);
    return result;
}

bool PhaseVisionAdapter::cancel(uint64_t requestId)
{
    auto it = mRequests.find(requestId);
    if (it == mRequests.end() || !ready(requestId))
    {
        return false;
    }
    mRequests.erase(it);
    return true;
}

bool PhaseVisionAdapter::busy() const noexcept
{
    return !mRequests.empty();
}

CUcontext PhaseVisionAdapter::cudaContext() const noexcept
{
    return mCudaContext;
}

} // namespace trt_edgellm::rt
