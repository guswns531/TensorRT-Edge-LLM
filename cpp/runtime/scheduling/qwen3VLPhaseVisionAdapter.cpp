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

#include "runtime/scheduling/qwen3VLPhaseVisionAdapter.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <functional>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

Qwen3VLPhaseVisionAdapter::Qwen3VLPhaseVisionAdapter(MultimodalRunner& runner, tokenizer::Tokenizer const& tokenizer,
    LLMEngineConfig const& config, PhaseKernelGroupRecorder* kernelGroupRecorder)
    : mRunner(runner)
    , mTokenizer(tokenizer)
    , mConfig(config)
    , mKernelGroupRecorder(kernelGroupRecorder)
{
    check::check(mRunner.getModelType() == multimodal::ModelType::QWEN3_VL
            || mRunner.getModelType() == multimodal::ModelType::QWEN3_5,
        "Qwen3-VL phase vision adapter requires a Qwen3-VL visual runner.");
    check::check(mConfig.ropeConfig.type == RopeType::kMRope,
        "Qwen3-VL phase vision adapter requires an M-RoPE decoder engine.");
}

void Qwen3VLPhaseVisionAdapter::registerRequest(
    uint64_t requestId, LLMGenerationRequest const& request, DecodingInferenceContext& context)
{
    check::check(request.requests.size() == 1U, "Qwen3-VL phase vision requires one logical sequence per request.");
    check::check(context.activeBatchSize == 1, "Qwen3-VL phase vision target context must have batch size one.");
    check::check(
        mRegistrations.find(requestId) == mRegistrations.end(), "Qwen3-VL phase vision request is already registered.");

    Registration registration;
    registration.request = &request;
    registration.context = &context;
    registration.mropeCosSin = Tensor({1, mConfig.maxKVCacheCapacity, mConfig.rotaryDim}, DeviceType::kGPU,
        nvinfer1::DataType::kFLOAT, "phase_qwen3vl_mrope_cos_sin");
    mRegistrations.emplace(requestId, std::move(registration));
}

void Qwen3VLPhaseVisionAdapter::release(uint64_t requestId)
{
    check::check(mRegistrations.erase(requestId) == 1U, "Qwen3-VL phase vision request is not registered.");
}

bool Qwen3VLPhaseVisionAdapter::hasRequest(uint64_t requestId) const noexcept
{
    return mRegistrations.find(requestId) != mRegistrations.end();
}

PhaseEncoderDispatchWorkerCallbacks Qwen3VLPhaseVisionAdapter::makeCallbacks()
{
    PhaseEncoderDispatchWorkerCallbacks callbacks;
    callbacks.enqueueEncoder
        = [this](std::vector<PhaseEncoderWorkItem> const& batch, cudaStream_t stream) { enqueue(batch, stream); };
    callbacks.completeEncoder = [this](PhaseEncoderWorkItem const& item) { return complete(item); };
    return callbacks;
}

void Qwen3VLPhaseVisionAdapter::copyTensor(Tensor const& source, Tensor& target, char const* name, cudaStream_t stream)
{
    target = Tensor(source.getShape(), DeviceType::kGPU, source.getDataType(), name);
    int64_t const bytes = source.getShape().volume() * utils::getTypeSize(source.getDataType());
    CUDA_CHECK(cudaMemcpyAsync(target.rawPointer(), source.rawPointer(), bytes, cudaMemcpyDeviceToDevice, stream));
}

void Qwen3VLPhaseVisionAdapter::enqueue(std::vector<PhaseEncoderWorkItem> const& batch, cudaStream_t stream)
{
    check::check(batch.size() == 1U, "Qwen3-VL phase vision currently supports encoder batch size one.");
    Registration& state = registration(batch.front().requestId);
    std::vector<std::vector<int32_t>> batchedInputIds;
    PhaseKernelSegment const preprocessSegment{
        PhaseKernelGroup::kEncoderPreprocess, {}, stream, [&](cudaStream_t segmentStream) {
            check::check(mRunner.preprocess(*state.request, batchedInputIds, &mTokenizer,
                             OptionalOutputTensor{std::ref(state.mropeCosSin)}, segmentStream),
                "Qwen3-VL visual preprocessing failed.");
            check::check(batchedInputIds.size() == 1U && !batchedInputIds.front().empty(),
                "Qwen3-VL visual preprocessing produced an invalid token batch.");
        }};
    PhaseKernelSegment const encoderSegment{
        PhaseKernelGroup::kEncoderEngine, {}, stream, [&](cudaStream_t segmentStream) {
            check::check(mRunner.infer(segmentStream), "Qwen3-VL visual TensorRT execution failed.");
            copyTensor(
                mRunner.getOutputEmbedding(), state.visualEmbedding, "phase_qwen3vl_visual_embedding", segmentStream);
            OptionalInputTensors const deepstack = mRunner.getDeepstackFeatures();
            check::check(static_cast<int32_t>(deepstack.size()) == mConfig.numDeepstackFeatures,
                "Qwen3-VL visual encoder deepstack feature count does not match the decoder engine contract.");
            state.deepstackFeatures.clear();
            state.deepstackFeatures.reserve(deepstack.size());
            for (size_t index = 0; index < deepstack.size(); ++index)
            {
                Tensor feature;
                copyTensor(deepstack[index].get(), feature, "phase_qwen3vl_deepstack", segmentStream);
                state.deepstackFeatures.emplace_back(std::move(feature));
            }
        }};
    std::vector<PhaseKernelSegment> const segments{preprocessSegment, encoderSegment};
    if (mKernelGroupRecorder != nullptr)
    {
        mKernelGroupRecorder->execute(mKernelGroupDispatchIndex++, segments);
    }
    else
    {
        for (PhaseKernelSegment const& segment : segments)
        {
            segment.enqueue(segment.stream);
        }
    }
    state.tokenIds = std::move(batchedInputIds.front());
}

PhaseWorkItem Qwen3VLPhaseVisionAdapter::complete(PhaseEncoderWorkItem const& item)
{
    Registration& state = registration(item.requestId);
    check::check(!state.tokenIds.empty(), "Qwen3-VL visual completion has no tokenized prompt.");
    DecodingInferenceContext& context = *state.context;
    context.rawBatchedInputIds = {state.tokenIds};
    context.tokenIds = context.rawBatchedInputIds;
    context.effectivePrefillLengths = {static_cast<int32_t>(state.tokenIds.size())};
    context.visualEmbeddings = std::cref(state.visualEmbedding);
    context.deepstackFeatures.clear();
    for (Tensor const& feature : state.deepstackFeatures)
    {
        context.deepstackFeatures.emplace_back(std::cref(feature));
    }
    context.mropeCosSin = std::cref(state.mropeCosSin);

    int32_t const promptTokens = static_cast<int32_t>(state.tokenIds.size());
    PhaseWorkItem work{item.requestId, promptTokens, item.kvSlotId, 0, promptTokens};
    work.allowChunkedPrefill = false;
    return work;
}

Qwen3VLPhaseVisionAdapter::Registration& Qwen3VLPhaseVisionAdapter::registration(uint64_t requestId)
{
    auto const found = mRegistrations.find(requestId);
    check::check(found != mRegistrations.end(), "Qwen3-VL phase vision request is not registered.");
    return found->second;
}

} // namespace rt
} // namespace trt_edgellm
