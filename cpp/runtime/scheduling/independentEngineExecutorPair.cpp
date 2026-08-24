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

#include "runtime/scheduling/independentEngineExecutorPair.h"

#include "common/checkMacros.h"

#include <utility>

namespace trt_edgellm
{
namespace rt
{

std::unique_ptr<IndependentEngineExecutorPair> IndependentEngineExecutorPair::create(
    std::unique_ptr<EngineExecutor> prefillExecutor, IndependentEngineExecutorPairConfig config)
{
    return std::unique_ptr<IndependentEngineExecutorPair>(
        new IndependentEngineExecutorPair(std::move(prefillExecutor), nullptr, config));
}

std::unique_ptr<IndependentEngineExecutorPair> IndependentEngineExecutorPair::create(
    std::unique_ptr<EngineExecutor> prefillExecutor, std::unique_ptr<EngineExecutor> decodeExecutor,
    IndependentEngineExecutorPairConfig config)
{
    ELLM_CHECK(decodeExecutor != nullptr, "Independent phase execution requires a decode executor");
    return std::unique_ptr<IndependentEngineExecutorPair>(
        new IndependentEngineExecutorPair(std::move(prefillExecutor), std::move(decodeExecutor), config));
}

IndependentEngineExecutorPair::IndependentEngineExecutorPair(
    std::unique_ptr<EngineExecutor> prefillExecutor, std::unique_ptr<EngineExecutor> decodeExecutor,
    IndependentEngineExecutorPairConfig config)
    : mPrefillExecutor(std::move(prefillExecutor))
    , mDecodeExecutor(std::move(decodeExecutor))
    , mConfig(config)
{
    ELLM_CHECK(mPrefillExecutor != nullptr, "Independent phase execution requires a prefill executor");
    ELLM_CHECK(mConfig.prefillProfile >= 0 && mConfig.decodeProfile >= 0,
        "Independent phase profile indices must be non-negative");
    int32_t const prefillProfileCount = mPrefillExecutor->getEngine().getNbOptimizationProfiles();
    ELLM_CHECK(mConfig.prefillProfile < prefillProfileCount,
        "Independent prefill profile index is not present in the TensorRT engine");
    validateStreams(mConfig, mCudaContext);

    if (mDecodeExecutor == nullptr)
    {
        mDecodeExecutor = mPrefillExecutor->createSibling();
    }
    ELLM_CHECK(mDecodeExecutor != nullptr, "Failed to create the independent decode executor");
    ELLM_CHECK(mConfig.decodeProfile < mDecodeExecutor->getEngine().getNbOptimizationProfiles(),
        "Independent decode profile index is not present in the TensorRT engine");
    ELLM_CHECK(mPrefillExecutor->getExecutionContextIdentity() != mDecodeExecutor->getExecutionContextIdentity(),
        "Independent phase executors unexpectedly share a TensorRT execution context");

    int64_t const prefillBytes = mPrefillExecutor->getRequiredContextMemorySizeForProfile(mConfig.prefillProfile);
    int64_t const decodeBytes = mDecodeExecutor->getRequiredContextMemorySizeForProfile(mConfig.decodeProfile);
    ELLM_CHECK(prefillBytes > 0 && decodeBytes > 0,
        "TensorRT returned an empty profile workspace for independent phase execution");

    mPrefillContextMemory = Tensor({prefillBytes}, DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "IndependentEngineExecutorPair::prefillContextMemory");
    mDecodeContextMemory = Tensor({decodeBytes}, DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "IndependentEngineExecutorPair::decodeContextMemory");

    ELLM_CHECK(mPrefillExecutor->setContextMemoryForProfile(
                   mConfig.prefillProfile, mPrefillContextMemory, mConfig.setupStream),
        "Failed to assign the prefill TensorRT profile workspace");
    ELLM_CHECK(
        mDecodeExecutor->setContextMemoryForProfile(mConfig.decodeProfile, mDecodeContextMemory, mConfig.setupStream),
        "Failed to assign the decode TensorRT profile workspace");
}

CUcontext IndependentEngineExecutorPair::streamContext(cudaStream_t stream)
{
    ELLM_CHECK(stream != nullptr, "Independent phase execution requires explicit non-default CUDA streams");
    CUcontext context{};
    CUDA_DRIVER_CHECK(cuStreamGetCtx(stream, &context));
    ELLM_CHECK(context != nullptr, "CUDA stream has no owning CUDA context");
    return context;
}

void IndependentEngineExecutorPair::validateStreams(
    IndependentEngineExecutorPairConfig const& config, CUcontext& context)
{
    CUcontext const setupContext = streamContext(config.setupStream);
    CUcontext const prefillContext = streamContext(config.prefillStream);
    CUcontext const decodeContext = streamContext(config.decodeStream);
    ELLM_CHECK(setupContext == prefillContext && setupContext == decodeContext,
        "Independent phase streams must share one CUDA context");
    ELLM_CHECK(config.prefillStream != config.decodeStream,
        "Independent TensorRT contexts require distinct prefill and decode streams");
    context = setupContext;
}

EngineExecutor& IndependentEngineExecutorPair::prefillExecutor() noexcept
{
    return *mPrefillExecutor;
}

EngineExecutor const& IndependentEngineExecutorPair::prefillExecutor() const noexcept
{
    return *mPrefillExecutor;
}

EngineExecutor& IndependentEngineExecutorPair::decodeExecutor() noexcept
{
    return *mDecodeExecutor;
}

EngineExecutor const& IndependentEngineExecutorPair::decodeExecutor() const noexcept
{
    return *mDecodeExecutor;
}

Tensor& IndependentEngineExecutorPair::prefillContextMemory() noexcept
{
    return mPrefillContextMemory;
}

Tensor& IndependentEngineExecutorPair::decodeContextMemory() noexcept
{
    return mDecodeContextMemory;
}

CUcontext IndependentEngineExecutorPair::cudaContext() const noexcept
{
    return mCudaContext;
}

IndependentEngineExecutorPairConfig const& IndependentEngineExecutorPair::config() const noexcept
{
    return mConfig;
}

} // namespace rt
} // namespace trt_edgellm
