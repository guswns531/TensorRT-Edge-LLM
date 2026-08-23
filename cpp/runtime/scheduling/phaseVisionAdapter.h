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

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace trt_edgellm::rt
{

//! Request-owned encoder output retained until the corresponding prefill completes.
struct PhaseVisionPayload
{
    ~PhaseVisionPayload() noexcept;

    size_t byteSize() const noexcept;

    std::vector<std::vector<int32_t>> tokenIds;
    Tensor outputEmbedding;
    std::vector<Tensor> deepstackFeatures;
    Tensor mropeCosSin;
    float encoderGpuMs{};
    cudaEvent_t startEvent{};
    cudaEvent_t readyEvent{};
};

//! Model-neutral wrapper around a v0.10 MultimodalRunner execution context.
//!
//! One encoder request is in flight at a time because the borrowed runner owns
//! reusable output buffers. Results are copied to request-owned GPU tensors on
//! the encoder stream before readyEvent is recorded.
class PhaseVisionAdapter
{
public:
    PhaseVisionAdapter(MultimodalRunner& runner, tokenizer::Tokenizer const& tokenizer, LLMEngineConfig const& config,
        cudaStream_t stream);

    PhaseVisionAdapter(PhaseVisionAdapter const&) = delete;
    PhaseVisionAdapter& operator=(PhaseVisionAdapter const&) = delete;

    bool submit(uint64_t requestId, LLMGenerationRequest const& request);
    bool ready(uint64_t requestId) const;
    std::unique_ptr<PhaseVisionPayload> take(uint64_t requestId);
    bool cancel(uint64_t requestId);
    bool busy() const noexcept;
    CUcontext cudaContext() const noexcept;

private:
    static Tensor copyTensor(Tensor const& source, std::string const& name, cudaStream_t stream);

    MultimodalRunner& mRunner;
    tokenizer::Tokenizer const& mTokenizer;
    LLMEngineConfig mConfig;
    cudaStream_t mStream{};
    CUcontext mCudaContext{};
    std::unordered_map<uint64_t, std::unique_ptr<PhaseVisionPayload>> mRequests;
};

} // namespace trt_edgellm::rt
