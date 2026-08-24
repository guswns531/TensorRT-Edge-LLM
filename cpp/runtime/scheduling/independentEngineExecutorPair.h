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
#include "runtime/exec/engineExecutor.h"

#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>

namespace trt_edgellm
{
namespace rt
{

//! @brief Resources required to run two phases concurrently on one CUDA context.
struct IndependentEngineExecutorPairConfig
{
    int32_t prefillProfile{0};
    int32_t decodeProfile{1};
    cudaStream_t setupStream{};
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    //! Reuse one TensorRT execution context/workspace and serialize phase enqueues.
    bool sharedExecutionContext{};
};

//! @brief Two independent TensorRT contexts over one deserialized engine.
//!
//! The pair is deliberately model agnostic. Callers construct the first
//! EngineExecutor with the model-specific registry/factory, then pass it here.
//! The sibling shares immutable TRT weights and the ICudaEngine, but owns a
//! different IExecutionContext, auxiliary streams, CUDA graph cache, and
//! profile-sized workspace. This is the resource boundary required before
//! prefill and decode can be enqueued concurrently.
class IndependentEngineExecutorPair
{
public:
    //! Build a pair from an already configured executor.
    static std::unique_ptr<IndependentEngineExecutorPair> create(
        std::unique_ptr<EngineExecutor> prefillExecutor, IndependentEngineExecutorPairConfig config);

    IndependentEngineExecutorPair(IndependentEngineExecutorPair const&) = delete;
    IndependentEngineExecutorPair& operator=(IndependentEngineExecutorPair const&) = delete;

    EngineExecutor& prefillExecutor() noexcept;
    EngineExecutor const& prefillExecutor() const noexcept;
    EngineExecutor& decodeExecutor() noexcept;
    EngineExecutor const& decodeExecutor() const noexcept;

    //! Profile-specific USER_MANAGED TensorRT context memory.
    Tensor& prefillContextMemory() noexcept;
    Tensor& decodeContextMemory() noexcept;

    //! The CUDA context owning all three supplied streams.
    CUcontext cudaContext() const noexcept;

    IndependentEngineExecutorPairConfig const& config() const noexcept;
    bool sharedExecutionContext() const noexcept;

private:
    IndependentEngineExecutorPair(
        std::unique_ptr<EngineExecutor> prefillExecutor, IndependentEngineExecutorPairConfig config);

    static CUcontext streamContext(cudaStream_t stream);
    static void validateStreams(IndependentEngineExecutorPairConfig const& config, CUcontext& context);

    Tensor mPrefillContextMemory;
    Tensor mDecodeContextMemory;
    std::unique_ptr<EngineExecutor> mPrefillExecutor;
    std::unique_ptr<EngineExecutor> mDecodeExecutor;
    IndependentEngineExecutorPairConfig mConfig;
    CUcontext mCudaContext{};
};

} // namespace rt
} // namespace trt_edgellm
