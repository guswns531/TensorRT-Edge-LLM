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
#include "multimodal/multimodalRunner.h"

#include <algorithm>
#include <cstddef>
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

    int64_t const prefillBytes = mPrefillExecutor->getRequiredContextMemorySizeForProfile(mConfig.prefillProfile);
    int64_t const decodeBytes = mPrefillExecutor->getRequiredContextMemorySizeForProfile(mConfig.decodeProfile);
    ELLM_CHECK(prefillBytes > 0 && decodeBytes > 0, "TensorRT returned an empty profile workspace for phase execution");

    if (mConfig.sharedExecutionContext)
    {
        int64_t const sharedBytes = std::max(prefillBytes, decodeBytes);
        mPrefillContextMemory = Tensor({sharedBytes}, DeviceType::kGPU, nvinfer1::DataType::kUINT8,
            "IndependentEngineExecutorPair::sharedContextMemory");
        ELLM_CHECK(mPrefillExecutor->setContextMemoryForProfile(
                       mConfig.prefillProfile, mPrefillContextMemory, mConfig.setupStream),
            "Failed to assign the shared TensorRT profile workspace");
        return;
    }

    mDecodeExecutor = mPrefillExecutor->createSibling();
    ELLM_CHECK(mDecodeExecutor != nullptr, "Failed to create the independent decode executor");
    ELLM_CHECK(mPrefillExecutor->getExecutionContextIdentity() != mDecodeExecutor->getExecutionContextIdentity(),
        "Independent phase executors unexpectedly share a TensorRT execution context");

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
    return mConfig.sharedExecutionContext ? *mPrefillExecutor : *mDecodeExecutor;
}

EngineExecutor const& IndependentEngineExecutorPair::decodeExecutor() const noexcept
{
    return mConfig.sharedExecutionContext ? *mPrefillExecutor : *mDecodeExecutor;
}

Tensor& IndependentEngineExecutorPair::prefillContextMemory() noexcept
{
    return mPrefillContextMemory;
}

Tensor& IndependentEngineExecutorPair::decodeContextMemory() noexcept
{
    return mConfig.sharedExecutionContext ? mPrefillContextMemory : mDecodeContextMemory;
}

TieredVisionContextMemoryInfo IndependentEngineExecutorPair::configureTieredVisionContextMemory(
    MultimodalRunner& vision, int32_t smallVisionProfile, int32_t largeVisionProfile)
{
    ELLM_CHECK(!mConfig.sharedExecutionContext,
        "Tiered E/P context memory requires independent prefill and decode execution contexts");
    ELLM_CHECK(smallVisionProfile >= 0 && smallVisionProfile < vision.getOptimizationProfileCount(),
        "Small vision optimization profile is out of range");
    ELLM_CHECK(largeVisionProfile >= 0 && largeVisionProfile < vision.getOptimizationProfileCount(),
        "Large vision optimization profile is out of range");
    ELLM_CHECK(smallVisionProfile != largeVisionProfile,
        "Tiered E/P context memory requires distinct small and large vision profiles");

    int64_t const prefillBytes = mPrefillExecutor->getRequiredContextMemorySizeForProfile(mConfig.prefillProfile);
    int64_t const smallVisionBytes = vision.getRequiredContextMemorySizeForProfile(smallVisionProfile);
    int64_t const largeVisionBytes = vision.getRequiredContextMemorySizeForProfile(largeVisionProfile);
    constexpr int64_t kContextAlignment = 256;
    int64_t const prefillSpanBytes = ((prefillBytes + kContextAlignment - 1) / kContextAlignment) * kContextAlignment;
    int64_t const arenaBytes = std::max(prefillSpanBytes + smallVisionBytes, largeVisionBytes);

    // Release the old prefill allocation before acquiring the replacement arena,
    // avoiding a transient peak on memory-constrained edge GPUs.
    mPrefillContextMemory = Tensor{};
    mTieredContextMemoryArena = Tensor{};
    mTieredContextMemoryArena = Tensor({arenaBytes}, DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "IndependentEngineExecutorPair::tieredVisionContextMemory");
    auto* const arenaBase = static_cast<std::byte*>(mTieredContextMemoryArena.rawPointer());
    mPrefillContextMemory = Tensor(arenaBase, {prefillBytes}, DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "IndependentEngineExecutorPair::tieredPrefillContextMemory");
    Tensor smallVisionMemory(arenaBase + prefillSpanBytes, {smallVisionBytes}, DeviceType::kGPU,
        nvinfer1::DataType::kUINT8, "IndependentEngineExecutorPair::tieredSmallVisionContextMemory");
    Tensor largeVisionMemory(arenaBase, {largeVisionBytes}, DeviceType::kGPU, nvinfer1::DataType::kUINT8,
        "IndependentEngineExecutorPair::tieredLargeVisionContextMemory");

    ELLM_CHECK(vision.setContextMemoryForProfile(smallVisionProfile, smallVisionMemory, mConfig.setupStream),
        "Failed to assign the small vision profile workspace");
    ELLM_CHECK(vision.setContextMemoryForProfile(largeVisionProfile, largeVisionMemory, mConfig.setupStream),
        "Failed to assign the large vision profile workspace");
    ELLM_CHECK(mPrefillExecutor->setContextMemoryForProfile(
                   mConfig.prefillProfile, mPrefillContextMemory, mConfig.setupStream),
        "Failed to rebind prefill to the tiered E/P context arena");

    return {arenaBytes, prefillBytes, smallVisionBytes, largeVisionBytes};
}

CUcontext IndependentEngineExecutorPair::cudaContext() const noexcept
{
    return mCudaContext;
}

IndependentEngineExecutorPairConfig const& IndependentEngineExecutorPair::config() const noexcept
{
    return mConfig;
}

bool IndependentEngineExecutorPair::sharedExecutionContext() const noexcept
{
    return mConfig.sharedExecutionContext;
}

} // namespace rt
} // namespace trt_edgellm
