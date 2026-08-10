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

#pragma once

#include "multimodal/multimodalRunner.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/scheduling/phaseKernelGroupRecorder.h"
#include "runtime/scheduling/phaseVisionAdapter.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Qwen3-VL/Cosmos encoder adapter for phase execution.
//!
//! Unlike the Gemma adapter, this owns request-scoped copies of the main
//! visual embedding, every raw deepstack feature, and the M-RoPE cache until
//! packed prefill has consumed them.
class Qwen3VLPhaseVisionAdapter : public PhaseVisionAdapter
{
public:
    Qwen3VLPhaseVisionAdapter(MultimodalRunner& runner, tokenizer::Tokenizer const& tokenizer,
        LLMEngineConfig const& config, PhaseKernelGroupRecorder* kernelGroupRecorder = nullptr);
    ~Qwen3VLPhaseVisionAdapter() noexcept override = default;

    void registerRequest(
        uint64_t requestId, LLMGenerationRequest const& request, DecodingInferenceContext& context) override;
    void release(uint64_t requestId) override;
    bool hasRequest(uint64_t requestId) const noexcept override;
    PhaseEncoderDispatchWorkerCallbacks makeCallbacks() override;

private:
    struct Registration
    {
        LLMGenerationRequest const* request{};
        DecodingInferenceContext* context{};
        std::vector<int32_t> tokenIds;
        Tensor visualEmbedding;
        std::vector<Tensor> deepstackFeatures;
        Tensor mropeCosSin;
    };

    void enqueue(std::vector<PhaseEncoderWorkItem> const& batch, cudaStream_t stream);
    PhaseWorkItem complete(PhaseEncoderWorkItem const& item);
    Registration& registration(uint64_t requestId);
    static void copyTensor(Tensor const& source, Tensor& target, char const* name, cudaStream_t stream);

    MultimodalRunner& mRunner;
    tokenizer::Tokenizer const& mTokenizer;
    LLMEngineConfig mConfig;
    PhaseKernelGroupRecorder* mKernelGroupRecorder{};
    size_t mKernelGroupDispatchIndex{};
    std::unordered_map<uint64_t, Registration> mRegistrations;
};

} // namespace rt
} // namespace trt_edgellm
