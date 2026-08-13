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
        new IndependentEngineExecutorPair(std::move(prefillExecutor), config));
}

IndependentEngineExecutorPair::IndependentEngineExecutorPair(
    std::unique_ptr<EngineExecutor> prefillExecutor, IndependentEngineExecutorPairConfig config)
    : mPrefillExecutor(std::move(prefillExecutor))
    , mConfig(config)
{
    ELLM_CHECK(mPrefillExecutor != nullptr, "Independent phase execution requires a prefill executor");
    ELLM_CHECK(mConfig.prefillProfile >= 0 && mConfig.decodeProfile >= 0,
        "Independent phase profile indices must be non-negative");
    int32_t const profileCount = mPrefillExecutor->getEngine().getNbOptimizationProfiles();
    ELLM_CHECK(mConfig.prefillProfile < profileCount && mConfig.decodeProfile < profileCount,
        "Independent phase profile index is not present in the TensorRT engine");
    validateStreams(mConfig, mCudaContext);

    mDecodeExecutor = mPrefillExecutor->createSibling();
    ELLM_CHECK(mDecodeExecutor != nullptr, "Failed to create the independent decode executor");
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
    if (mConfig.enableCudaGraph)
    {
        ELLM_CHECK(mConfig.maxPrefillCudaGraphs > 0U && mConfig.maxDecodeCudaGraphs > 0U,
            "Independent CUDA graph cache limits must be positive");
        ELLM_CHECK(
            mConfig.minimumCudaGraphChargeBytes > 0U, "Independent CUDA graph minimum memory charge must be positive");
        mPrefillExecutor->enableAutomaticCudaGraphCapture(mConfig.maxPrefillCudaGraphs,
            mConfig.maxPrefillCudaGraphBytes, mConfig.minimumCudaGraphChargeBytes, mConfig.minimumCudaFreeMemoryBytes);
        mDecodeExecutor->enableAutomaticCudaGraphCapture(mConfig.maxDecodeCudaGraphs, mConfig.maxDecodeCudaGraphBytes,
            mConfig.minimumCudaGraphChargeBytes, mConfig.minimumCudaFreeMemoryBytes);
    }
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
