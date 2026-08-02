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

#include "runtime/scheduling/gemma4PhaseVisionAdapter.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"
#include <functional>

namespace trt_edgellm
{
namespace rt
{

Gemma4PhaseVisionAdapter::Gemma4PhaseVisionAdapter(
    MultimodalRunner& runner, tokenizer::Tokenizer const& tokenizer, PhaseKernelGroupRecorder* kernelGroupRecorder)
    : mRunner(runner)
    , mTokenizer(tokenizer)
    , mKernelGroupRecorder(kernelGroupRecorder)
{
    check::check(mRunner.getModelType() == multimodal::ModelType::GEMMA4_VISION,
        "Gemma4 phase vision adapter requires a Gemma4 visual runner.");
}

void Gemma4PhaseVisionAdapter::registerRequest(
    uint64_t requestId, LLMGenerationRequest const& request, DecodingInferenceContext& context)
{
    check::check(request.requests.size() == 1U, "Gemma4 phase vision V1 requires one logical sequence per request.");
    check::check(context.activeBatchSize == 1, "Gemma4 phase vision target context must have batch size one.");
    check::check(
        mRegistrations.find(requestId) == mRegistrations.end(), "Gemma4 phase vision request is already registered.");
    Registration registration;
    registration.request = &request;
    registration.context = &context;
    mRegistrations.emplace(requestId, std::move(registration));
}

void Gemma4PhaseVisionAdapter::release(uint64_t requestId)
{
    check::check(mRegistrations.erase(requestId) == 1U, "Gemma4 phase vision request is not registered.");
}

bool Gemma4PhaseVisionAdapter::hasRequest(uint64_t requestId) const noexcept
{
    return mRegistrations.find(requestId) != mRegistrations.end();
}

PhaseEncoderDispatchWorkerCallbacks Gemma4PhaseVisionAdapter::makeCallbacks()
{
    PhaseEncoderDispatchWorkerCallbacks callbacks;
    callbacks.enqueueEncoder
        = [this](std::vector<PhaseEncoderWorkItem> const& batch, cudaStream_t stream) { enqueue(batch, stream); };
    callbacks.completeEncoder = [this](PhaseEncoderWorkItem const& item) { return complete(item); };
    return callbacks;
}

void Gemma4PhaseVisionAdapter::enqueue(std::vector<PhaseEncoderWorkItem> const& batch, cudaStream_t stream)
{
    check::check(batch.size() == 1U, "Gemma4 phase vision V1 supports encoder batch size one.");
    Registration& state = registration(batch.front().requestId);
    std::vector<std::vector<int32_t>> batchedInputIds;
    std::vector<PhaseKernelSegment> const segments{
        {PhaseKernelGroup::kEncoderPreprocess, {}, stream,
            [&](cudaStream_t segmentStream) {
                check::check(
                    mRunner.preprocess(*state.request, batchedInputIds, &mTokenizer, std::nullopt, segmentStream),
                    "Gemma4 visual preprocessing failed.");
                check::check(batchedInputIds.size() == 1U && !batchedInputIds.front().empty(),
                    "Gemma4 visual preprocessing produced an invalid token batch.");
            }},
        {PhaseKernelGroup::kEncoderEngine, {}, stream,
            [&](cudaStream_t segmentStream) {
                check::check(mRunner.infer(segmentStream), "Gemma4 visual TensorRT execution failed.");
                Tensor& source = mRunner.getOutputEmbedding();
                state.visualEmbedding = Tensor(
                    source.getShape(), DeviceType::kGPU, source.getDataType(), "phase_gemma4_visual_embedding");
                int64_t const copyBytes = source.getShape().volume() * utils::getTypeSize(source.getDataType());
                CUDA_CHECK(cudaMemcpyAsync(state.visualEmbedding.rawPointer(), source.rawPointer(), copyBytes,
                    cudaMemcpyDeviceToDevice, segmentStream));
            }},
    };
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

PhaseWorkItem Gemma4PhaseVisionAdapter::complete(PhaseEncoderWorkItem const& item)
{
    Registration& state = registration(item.requestId);
    check::check(!state.tokenIds.empty(), "Gemma4 visual completion has no tokenized prompt.");
    DecodingInferenceContext& context = *state.context;
    context.rawBatchedInputIds = {state.tokenIds};
    context.tokenIds = context.rawBatchedInputIds;
    context.effectivePrefillLengths = {static_cast<int32_t>(state.tokenIds.size())};
    context.visualEmbeddings = std::ref(state.visualEmbedding);

    int32_t const promptTokens = static_cast<int32_t>(state.tokenIds.size());
    PhaseWorkItem work{item.requestId, promptTokens, item.kvSlotId, 0, promptTokens};
    work.allowChunkedPrefill = false;
    return work;
}

Gemma4PhaseVisionAdapter::Registration& Gemma4PhaseVisionAdapter::registration(uint64_t requestId)
{
    auto const found = mRegistrations.find(requestId);
    check::check(found != mRegistrations.end(), "Gemma4 phase vision request is not registered.");
    return found->second;
}

} // namespace rt
} // namespace trt_edgellm
