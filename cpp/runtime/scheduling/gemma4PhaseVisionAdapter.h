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
#include "runtime/scheduling/phaseEncoderDispatchWorker.h"
#include "runtime/scheduling/phaseKernelGroupRecorder.h"
#include "runtime/state/decodingInferenceContext.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Connects the real Gemma4 visual TensorRT runner to encoder phase dispatch.
//!
//! V1 accepts one logical sequence per encoder dispatch. The visual embedding
//! is copied to request-owned GPU storage on the encoder stream before the
//! worker records its completion event, so a later encoder batch cannot
//! overwrite embeddings still consumed by prefill.
class Gemma4PhaseVisionAdapter
{
public:
    Gemma4PhaseVisionAdapter(MultimodalRunner& runner, tokenizer::Tokenizer const& tokenizer,
        PhaseKernelGroupRecorder* kernelGroupRecorder = nullptr);

    void registerRequest(uint64_t requestId, LLMGenerationRequest const& request, DecodingInferenceContext& context);
    void release(uint64_t requestId);
    bool hasRequest(uint64_t requestId) const noexcept;

    PhaseEncoderDispatchWorkerCallbacks makeCallbacks();

private:
    struct Registration
    {
        LLMGenerationRequest const* request{};
        DecodingInferenceContext* context{};
        std::vector<int32_t> tokenIds;
        Tensor visualEmbedding;
    };

    void enqueue(std::vector<PhaseEncoderWorkItem> const& batch, cudaStream_t stream);
    PhaseWorkItem complete(PhaseEncoderWorkItem const& item);
    Registration& registration(uint64_t requestId);

    MultimodalRunner& mRunner;
    tokenizer::Tokenizer const& mTokenizer;
    PhaseKernelGroupRecorder* mKernelGroupRecorder{};
    size_t mKernelGroupDispatchIndex{};
    std::unordered_map<uint64_t, Registration> mRegistrations;
};

} // namespace rt
} // namespace trt_edgellm
