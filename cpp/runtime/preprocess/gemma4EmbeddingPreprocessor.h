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

#include "common/tensor.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/tensorMap.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Runtime preprocessor for Gemma4 E-model per-layer embeddings (PLE).
//!
//! Loads the token-identity PLE table from ple_embedding.safetensors, gathers
//! one [batch, seq_len, ple_hidden_size] tensor per decoder layer from token
//! IDs, and binds those tensors as ple_token_embeds_{layer_idx} engine inputs.
class Gemma4EmbeddingPreprocessor
{
public:
    Gemma4EmbeddingPreprocessor(std::filesystem::path const& engineDir, LLMEngineConfig const& config,
        int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream);

    //! Create phase-local mutable outputs while sharing the immutable PLE table.
    std::unique_ptr<Gemma4EmbeddingPreprocessor> createSibling(
        int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap) const;

    //! Gather PLE tensors for the current token-id tensor shape.
    void embed(Tensor const& tokenIds, cudaStream_t stream);

    //! Reshape already-bound output tensors for a CUDA-graph capture shape.
    void reshapeOutputs(int64_t batchSize, int64_t seqLen);

    //! Bind the same owned PLE output views into an additional serialized tensor map.
    void bindOutputs(TensorMap& tensorMap);

    //! Return the shared immutable PLE table allocation identity.
    void const* tableDataIdentity() const noexcept;

    //! Return this preprocessor's mutable phase-local output allocation identity.
    void const* outputDataIdentity() const noexcept;

private:
    Gemma4EmbeddingPreprocessor(LLMEngineConfig const& config, std::shared_ptr<Tensor> pleTable,
        int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap);
    void initializeOutputs(int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap);

    LLMEngineConfig mConfig{};
    std::shared_ptr<Tensor> mPleTable{};
    Tensor mPleOutputBuffer{}; //!< Unified owned backing buffer for all PLE layer outputs.
    //! Non-owned tensor views into mPleOutputBuffer. TensorMap stores pointers to these stable objects.
    std::vector<Tensor> mPleOutputViews{};

    //! Construct a non-owned tensor view for one layer output inside mPleOutputBuffer.
    Tensor makeOutputViewForLayer(int32_t layerIdx, int64_t batchSize, int64_t seqLen);
};

} // namespace rt
} // namespace trt_edgellm
