/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "runtime/exec/engineExecutor.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "runtime/exec/registryBuilder.h"
#include <limits>
#include <stdexcept>
#include <string_view>

namespace trt_edgellm
{
namespace rt
{
struct EngineExecutor::SharedEngineState
{
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
};

namespace
{
bool engineHasIOTensor(nvinfer1::ICudaEngine const& engine, char const* tensorName)
{
    if (tensorName == nullptr)
    {
        return false;
    }

    int32_t const numIO = engine.getNbIOTensors();
    for (int32_t i = 0; i < numIO; ++i)
    {
        char const* const name = engine.getIOTensorName(i);
        if (name != nullptr && std::string_view{name} == tensorName)
        {
            return true;
        }
    }
    return false;
}

bool engineHasInputTensor(nvinfer1::ICudaEngine const& engine, char const* tensorName)
{
    if (tensorName == nullptr)
    {
        return false;
    }

    int32_t const numIO = engine.getNbIOTensors();
    for (int32_t i = 0; i < numIO; ++i)
    {
        char const* const name = engine.getIOTensorName(i);
        if (name != nullptr && std::string_view{name} == tensorName)
        {
            return engine.getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        }
    }
    return false;
}
} // namespace

EngineExecutor::EngineExecutor(std::filesystem::path const& enginePath, TensorRegistry registry)
    : mRegistry(std::move(registry))
{
    LOG_INFO("loading engine file: %s", enginePath.string().c_str());

    mEngineState = std::make_shared<SharedEngineState>();
    mEngineState->runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
    ELLM_CHECK(mEngineState->runtime != nullptr, "failed to create TRT IRuntime");

    mEngineState->engine = deserializeCudaEngineFromFile(*mEngineState->runtime, enginePath);
    createExecutionContext();

    auto registerOptionalTreeMetadata = [&](char const* name) {
        if (engineHasInputTensor(*mEngineState->engine, name) && !mRegistry.contains(name))
        {
            mRegistry.addTensor({name, TensorIO::kInput, nvinfer1::DataType::kINT32,
                {sym(&InferenceDims::batch), sym(&InferenceDims::attnMaskSeqLen)}});
        }
    };
    registerOptionalTreeMetadata(binding_names::kTreeParentIds);
    registerOptionalTreeMetadata(binding_names::kTreeDepths);

    LOG_INFO("engine loaded successfully (%d I/O tensors)", mEngineState->engine->getNbIOTensors());
}

EngineExecutor::EngineExecutor(std::shared_ptr<SharedEngineState> engineState, TensorRegistry registry)
    : mEngineState(std::move(engineState))
    , mRegistry(std::move(registry))
{
    ELLM_CHECK(mEngineState != nullptr && mEngineState->engine != nullptr, "invalid shared TRT engine state");
    createExecutionContext();
}

void EngineExecutor::createExecutionContext()
{
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(
        mEngineState->engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    ELLM_CHECK(mContext != nullptr, "failed to create execution context");

    setNonBlockingAuxStreams(mContext.get(), mEngineState->engine.get(), mAuxStreams);
}

std::unique_ptr<EngineExecutor> EngineExecutor::createForLLM(std::filesystem::path const& enginePath,
    LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim)
{
    auto registry = buildRegistryForLLM(cfg, specDecodeBaseOutputHiddenDim);
    return std::unique_ptr<EngineExecutor>(new EngineExecutor(enginePath, std::move(registry)));
}

std::unique_ptr<EngineExecutor> EngineExecutor::createForDraft(
    std::filesystem::path const& enginePath, DeploymentConfig const& bundle)
{
    switch (bundle.specDecodeMode())
    {
    case SpecDecodeMode::kMTP:
    case SpecDecodeMode::kEAGLE:
    {
        auto registry = buildRegistryForSpecDecodeDraft(bundle);
        return std::unique_ptr<EngineExecutor>(new EngineExecutor(enginePath, std::move(registry)));
    }
    case SpecDecodeMode::kDFlash:
    {
        auto registry = buildRegistryForDFlashDraft(bundle);
        return std::unique_ptr<EngineExecutor>(new EngineExecutor(enginePath, std::move(registry)));
    }
    case SpecDecodeMode::kGemma4MTP:
    {
        auto registry = buildRegistryForGemma4MTPDraft(bundle);
        return std::unique_ptr<EngineExecutor>(new EngineExecutor(enginePath, std::move(registry)));
    }
    case SpecDecodeMode::kNONE:
    default: ELLM_CHECK(false, "createForDraft requires a speculative decoding deployment with a draft engine.");
    }
    return nullptr;
}

std::unique_ptr<EngineExecutor> EngineExecutor::createSibling() const
{
    return std::unique_ptr<EngineExecutor>(new EngineExecutor(mEngineState, mRegistry));
}

EngineExecutor::~EngineExecutor() noexcept
{
    for (auto& [hash, cg] : mGraphs)
    {
        if (cg.exec)
        {
            cudaGraphExecDestroy(cg.exec);
        }
        if (cg.graph)
        {
            cudaGraphDestroy(cg.graph);
        }
    }
    mGraphs.clear();
}

bool EngineExecutor::prepare(int32_t profileIndex, InferenceDims const& dims, TensorMap const& map, cudaStream_t stream)
{
    if (auto bad = firstInvalidMember(dims, mRegistry.referencedMembers()); bad != nullptr)
    {
        auto const name = dimName(bad);
        LOG_ERROR(
            "EngineExecutor::prepare: InferenceDims.%.*s = %lld (must be > 0); "
            "caller likely bypassed an LLMEngineConfig recipe method",
            static_cast<int>(name.size()), name.data(), static_cast<long long>(dims.*bad));
        return false;
    }

    if (!mContext->setOptimizationProfileAsync(profileIndex, stream))
    {
        LOG_ERROR("failed to set optimization profile %d", profileIndex);
        return false;
    }

    if (!mRegistry.bindAll(mContext.get(), map, dims))
    {
        LOG_ERROR("bindAll failed for profile %d", profileIndex);
        return false;
    }

    // Bind any TensorMap entries not covered by the registry. We rebind
    // address+shape on every call (not just when getTensorAddress is null)
    // because TRT does NOT clear bindings on setOptimizationProfileAsync —
    // shapes and addresses persist across profile switches and across
    // execute() calls. Unregistered tensors that change shape per step
    // (notably kvcache_start_index, which uses [0] for initial prefill and
    // [batch] for chunked prefill / decode) would otherwise stick at
    // whatever shape was set by the previous step.
    //
    // LoRA weights are model-dependent and populated into the TensorMap by
    // LoRAManager::refreshTensorMap() before prepare() is called.
    int32_t const numIO = mEngineState->engine->getNbIOTensors();
    for (int32_t i = 0; i < numIO; ++i)
    {
        char const* name = mEngineState->engine->getIOTensorName(i);
        if (mRegistry.contains(name))
        {
            // Already bound by bindAll above; leave alone.
            continue;
        }
        Tensor* tensor = map.get(name);
        if (tensor == nullptr)
        {
            LOG_ERROR(
                "engine binding '%s' is neither registry-bound nor present in the TensorMap; "
                "every engine I/O tensor must be handled by the registry, by LoRAManager, or by an explicit "
                "TensorMap entry. Most likely the runtime is missing a binding for a model variant.",
                name);
            return false;
        }
        if (!mContext->setTensorAddress(name, tensor->rawPointer()))
        {
            LOG_ERROR("setTensorAddress failed for binding '%s'", name);
            return false;
        }
        if (mEngineState->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            if (!mContext->setInputShape(name, tensor->getTRTDims()))
            {
                LOG_ERROR("setInputShape failed for binding '%s'", name);
                return false;
            }
        }
    }

    return true;
}

bool EngineExecutor::execute(cudaStream_t stream)
{
    size_t const hash = computeBindingHash();
    BindingSnapshot const current = snapshotBindings();
    auto it = mGraphs.find(hash);
    if (it != mGraphs.end())
    {
        if (current == it->second.snapshot)
        {
            cudaError_t const err = cudaGraphLaunch(it->second.exec, stream);
            if (err == cudaSuccess)
            {
                ++mCudaGraphStats.graphLaunches;
                mLastSuccessfulBindings = current;
                return true;
            }
            ++mCudaGraphStats.graphLaunchFailures;
            LOG_WARNING("cudaGraphLaunch failed (%s), falling back to enqueueV3", cudaGetErrorString(err));
            eraseCapturedGraph(it);
            mUncapturableBindings[hash] = current;
            return enqueueAndRemember(hash, current, stream);
        }
    }

    bool const cacheHasCapacity = mGraphs.size() < mMaxAutomaticGraphs;
    bool const memoryHasCapacity = graphMemoryBudgetHasCapacity();
    bool const repeatedBindings = mLastSuccessfulBindings.has_value() && current == mLastSuccessfulBindings.value();
    auto const observed = mObservedBindings.find(hash);
    bool const observedPreviously = observed != mObservedBindings.end() && current == observed->second;
    auto const uncapturable = mUncapturableBindings.find(hash);
    bool const capturePreviouslyFailed = uncapturable != mUncapturableBindings.end() && current == uncapturable->second;
    auto const budgetRejected = mBudgetRejectedBindings.find(hash);
    bool const captureExceededBudget
        = budgetRejected != mBudgetRejectedBindings.end() && current == budgetRejected->second;
    size_t const estimatedNextGraphBytes = mGraphs.empty() || mCachedGraphBytes == 0U
        ? mMinimumAutomaticGraphBytes
        : std::max((mCachedGraphBytes + mGraphs.size() - 1U) / mGraphs.size(), mMinimumAutomaticGraphBytes);
    bool const globalMemoryHasCapacity = globalMemoryReserveHasCapacity(estimatedNextGraphBytes);
    if (mAutomaticGraphCaptureEnabled && cacheHasCapacity && memoryHasCapacity && globalMemoryHasCapacity
        && repeatedBindings && !capturePreviouslyFailed && !captureExceededBudget)
    {
        return captureLaunchAndCache(hash, current, stream);
    }
    if (mAutomaticGraphCaptureEnabled && cacheHasCapacity && memoryHasCapacity && globalMemoryHasCapacity
        && observedPreviously && !capturePreviouslyFailed && !captureExceededBudget)
    {
        return enqueueCaptureAndCache(hash, current, stream);
    }
    if (mAutomaticGraphCaptureEnabled && !cacheHasCapacity)
    {
        ++mCudaGraphStats.cacheLimitBypasses;
    }
    if (mAutomaticGraphCaptureEnabled && cacheHasCapacity && !memoryHasCapacity)
    {
        ++mCudaGraphStats.graphMemoryBudgetBypasses;
    }
    if (mAutomaticGraphCaptureEnabled && cacheHasCapacity && memoryHasCapacity && !globalMemoryHasCapacity)
    {
        ++mCudaGraphStats.globalMemoryReserveBypasses;
    }

    return enqueueAndRemember(hash, current, stream);
}

bool EngineExecutor::captureGraph(cudaStream_t stream)
{
    // Warmup: run one enqueue to ensure all internal TRT state is initialized.
    if (!mContext->enqueueV3(stream))
    {
        LOG_ERROR("warmup enqueueV3 failed");
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t freeBytesBefore{};
    size_t totalBytes{};
    CUDA_CHECK(cudaMemGetInfo(&freeBytesBefore, &totalBytes));

    auto result = captureTRTCudaGraph(mContext.get(), stream);
    if (!result.has_value())
    {
        LOG_WARNING("CUDA graph capture failed");
        return false;
    }

    size_t const hash = computeBindingHash();
    BindingSnapshot const snap = snapshotBindings();

    size_t const graphBytes = measureCapturedGraphBytes(freeBytesBefore);
    if (!cacheCapturedGraph(hash, snap, *result, graphBytes))
    {
        return false;
    }
    ++mCudaGraphStats.enqueueExecutions;
    ++mCudaGraphStats.captures;
    mLastSuccessfulBindings = snap;

    LOG_INFO("captured graph (hash=0x%zx)", hash);
    return true;
}

void EngineExecutor::enableAutomaticCudaGraphCapture(
    size_t maxCachedGraphs, size_t maxCachedGraphBytes, size_t minimumGraphChargeBytes, size_t minimumFreeMemoryBytes)
{
    ELLM_CHECK(maxCachedGraphs > 0U, "Automatic CUDA graph capture requires a non-zero cache limit");
    ELLM_CHECK(
        maxCachedGraphs <= std::numeric_limits<size_t>::max() / 4U, "Automatic CUDA graph cache limit is too large");
    mAutomaticGraphCaptureEnabled = true;
    mMaxAutomaticGraphs = maxCachedGraphs;
    mMaxAutomaticGraphBytes = maxCachedGraphBytes;
    mMinimumAutomaticGraphBytes = minimumGraphChargeBytes;
    mMinimumFreeMemoryBytes = minimumFreeMemoryBytes;
    mMaxObservedBindings = maxCachedGraphs * 4U;
    mCudaGraphStats.automaticCaptureEnabled = true;
}

EngineExecutor::CudaGraphCacheStats EngineExecutor::getCudaGraphCacheStats() const noexcept
{
    CudaGraphCacheStats result = mCudaGraphStats;
    result.cachedGraphs = mGraphs.size();
    result.cachedGraphBytes = mCachedGraphBytes;
    result.maxCachedGraphBytes = mMaxAutomaticGraphBytes;
    result.minimumGraphChargeBytes = mMinimumAutomaticGraphBytes;
    result.minimumFreeMemoryBytes = mMinimumFreeMemoryBytes;
    result.automaticCaptureEnabled = mAutomaticGraphCaptureEnabled;
    return result;
}

int64_t EngineExecutor::getRequiredContextMemorySize() const
{
    // Use getDeviceMemorySizeV2() to get the max across ALL profiles.
    // SpecDecode base engines have multiple profiles (prefill + verification) with
    // different memory requirements. Using per-profile size can underallocate.
    return mEngineState->engine->getDeviceMemorySizeV2();
}

int64_t EngineExecutor::getRequiredContextMemorySizeForProfile(int32_t profileIndex) const
{
    ELLM_CHECK(profileIndex >= 0 && profileIndex < mEngineState->engine->getNbOptimizationProfiles(),
        "TensorRT optimization profile index is out of range");
    return mEngineState->engine->getDeviceMemorySizeForProfileV2(profileIndex);
}

bool EngineExecutor::setContextMemory(Tensor& sharedMem)
{
    mContext->setDeviceMemoryV2(sharedMem.rawPointer(), sharedMem.getMemoryCapacity());
    return true;
}

bool EngineExecutor::setContextMemoryForProfile(int32_t profileIndex, Tensor& sharedMem, cudaStream_t stream)
{
    ELLM_CHECK(profileIndex >= 0 && profileIndex < mEngineState->engine->getNbOptimizationProfiles(),
        "TensorRT optimization profile index is out of range");
    if (!mContext->setOptimizationProfileAsync(profileIndex, stream))
    {
        LOG_ERROR("failed to select fixed optimization profile %d for context memory", profileIndex);
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int64_t const requiredBytes = getRequiredContextMemorySizeForProfile(profileIndex);
    ELLM_CHECK(sharedMem.getMemoryCapacity() >= requiredBytes, "Profile-specific TensorRT context memory is too small");
    mContext->setDeviceMemoryV2(sharedMem.rawPointer(), sharedMem.getMemoryCapacity());
    return true;
}

int32_t EngineExecutor::getNumIOTensors() const
{
    return mEngineState->engine->getNbIOTensors();
}

char const* EngineExecutor::getIOTensorName(int32_t index) const
{
    return mEngineState->engine->getIOTensorName(index);
}

bool EngineExecutor::hasIOTensor(char const* name) const
{
    return engineHasIOTensor(*mEngineState->engine, name);
}

nvinfer1::DataType EngineExecutor::getBindingDataType(char const* name) const
{
    return mEngineState->engine->getTensorDataType(name);
}

nvinfer1::Dims EngineExecutor::getProfileShape(
    char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector) const
{
    return mEngineState->engine->getProfileShape(name, profileIndex, selector);
}

void EngineExecutor::setProfiler(nvinfer1::IProfiler* profiler) noexcept
{
    mContext->setProfiler(profiler);
}

nvinfer1::ICudaEngine const& EngineExecutor::getEngine() const noexcept
{
    return *mEngineState->engine;
}

nvinfer1::IExecutionContext const* EngineExecutor::getExecutionContextIdentity() const noexcept
{
    return mContext.get();
}

// ---------------------------------------------------------------------------
// BindingSnapshot
// ---------------------------------------------------------------------------

bool EngineExecutor::BindingSnapshot::operator==(BindingSnapshot const& rhs) const noexcept
{
    if (bindings.size() != rhs.bindings.size())
    {
        return false;
    }
    for (size_t i = 0; i < bindings.size(); ++i)
    {
        if (bindings[i].first != rhs.bindings[i].first)
        {
            return false;
        }
        if (!dimsEqual(bindings[i].second, rhs.bindings[i].second))
        {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

size_t EngineExecutor::computeBindingHash() const
{
    size_t seed = 0;
    int32_t const numIO = mEngineState->engine->getNbIOTensors();
    for (int32_t i = 0; i < numIO; ++i)
    {
        char const* name = mEngineState->engine->getIOTensorName(i);
        auto const addr = reinterpret_cast<uintptr_t>(mContext->getTensorAddress(name));
        nvinfer1::Dims const shape = mContext->getTensorShape(name);

        hash_utils::hashCombine(seed, addr);
        hash_utils::hashCombine(seed, shape.nbDims);
        for (int32_t d = 0; d < shape.nbDims; ++d)
        {
            hash_utils::hashCombine(seed, shape.d[d]);
        }
    }
    return seed;
}

EngineExecutor::BindingSnapshot EngineExecutor::snapshotBindings() const
{
    BindingSnapshot snap;
    int32_t const numIO = mEngineState->engine->getNbIOTensors();
    snap.bindings.reserve(numIO);
    for (int32_t i = 0; i < numIO; ++i)
    {
        char const* name = mEngineState->engine->getIOTensorName(i);
        auto const addr = reinterpret_cast<uintptr_t>(mContext->getTensorAddress(name));
        nvinfer1::Dims const shape = mContext->getTensorShape(name);
        snap.bindings.emplace_back(addr, shape);
    }
    return snap;
}

bool EngineExecutor::enqueueAndRemember(size_t hash, BindingSnapshot const& snapshot, cudaStream_t stream)
{
    bool const success = mContext->enqueueV3(stream);
    ++mCudaGraphStats.enqueueExecutions;
    if (success)
    {
        mLastSuccessfulBindings = snapshot;
        rememberBindingObservation(hash, snapshot);
    }
    return success;
}

bool EngineExecutor::captureLaunchAndCache(size_t hash, BindingSnapshot const& snapshot, cudaStream_t stream)
{
    // TensorRT requires one enqueue after a dynamic-shape change before graph
    // capture. execute() only reaches here when the immediately preceding
    // successful execution used the identical binding snapshot. Synchronizing
    // this phase stream also keeps the context out of concurrent enqueue/capture.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t freeBytesBefore{};
    size_t totalBytes{};
    CUDA_CHECK(cudaMemGetInfo(&freeBytesBefore, &totalBytes));
    auto result = captureTRTCudaGraph(mContext.get(), stream);
    if (!result.has_value())
    {
        ++mCudaGraphStats.captureFailures;
        mUncapturableBindings[hash] = snapshot;
        return enqueueAndRemember(hash, snapshot, stream);
    }

    size_t const graphBytes = measureCapturedGraphBytes(freeBytesBefore);
    if (!cacheCapturedGraph(hash, snapshot, *result, graphBytes))
    {
        mBudgetRejectedBindings[hash] = snapshot;
        return enqueueAndRemember(hash, snapshot, stream);
    }

    auto const graph = mGraphs.find(hash);
    check::check(graph != mGraphs.end(), "Captured CUDA graph disappeared before launch");
    cudaError_t const launchStatus = cudaGraphLaunch(graph->second.exec, stream);
    if (launchStatus != cudaSuccess)
    {
        ++mCudaGraphStats.graphLaunchFailures;
        LOG_WARNING(
            "Captured CUDA graph launch failed (%s), falling back to enqueueV3", cudaGetErrorString(launchStatus));
        eraseCapturedGraph(graph);
        mUncapturableBindings[hash] = snapshot;
        return enqueueAndRemember(hash, snapshot, stream);
    }

    ++mCudaGraphStats.captures;
    ++mCudaGraphStats.graphLaunches;
    mLastSuccessfulBindings = snapshot;
    LOG_INFO("captured and launched graph (hash=0x%zx, cache=%zu/%zu)", hash, mGraphs.size(), mMaxAutomaticGraphs);
    return true;
}

bool EngineExecutor::enqueueCaptureAndCache(size_t hash, BindingSnapshot const& snapshot, cudaStream_t stream)
{
    // The same snapshot was observed before, but another shape has run since.
    // Execute this logical inference normally so TensorRT applies the current
    // dynamic shape. Once it completes, capture the now-warm state without
    // launching it; the current inference must never update live KV twice.
    if (!enqueueAndRemember(hash, snapshot, stream))
    {
        return false;
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    size_t freeBytesBefore{};
    size_t totalBytes{};
    CUDA_CHECK(cudaMemGetInfo(&freeBytesBefore, &totalBytes));
    auto result = captureTRTCudaGraph(mContext.get(), stream);
    if (!result.has_value())
    {
        ++mCudaGraphStats.captureFailures;
        mUncapturableBindings[hash] = snapshot;
        return true;
    }

    size_t const graphBytes = measureCapturedGraphBytes(freeBytesBefore);
    if (!cacheCapturedGraph(hash, snapshot, *result, graphBytes))
    {
        mBudgetRejectedBindings[hash] = snapshot;
        return true;
    }
    ++mCudaGraphStats.captures;
    ++mCudaGraphStats.postEnqueueCaptures;
    LOG_INFO("captured recurring graph after enqueue (hash=0x%zx, cache=%zu/%zu)", hash, mGraphs.size(),
        mMaxAutomaticGraphs);
    return true;
}

bool EngineExecutor::cacheCapturedGraph(size_t hash, BindingSnapshot const& snapshot,
    std::pair<cudaGraph_t, cudaGraphExec_t> const& capturedGraph, size_t estimatedDeviceBytes)
{
    auto existing = mGraphs.find(hash);
    size_t const replacedBytes = existing == mGraphs.end() ? 0U : existing->second.estimatedDeviceBytes;
    size_t const retainedBytes = mCachedGraphBytes - replacedBytes;
    if (mAutomaticGraphCaptureEnabled && mMaxAutomaticGraphBytes > 0U
        && estimatedDeviceBytes > mMaxAutomaticGraphBytes - std::min(retainedBytes, mMaxAutomaticGraphBytes))
    {
        static_cast<void>(cudaGraphExecDestroy(capturedGraph.second));
        static_cast<void>(cudaGraphDestroy(capturedGraph.first));
        ++mCudaGraphStats.graphMemoryBudgetRejections;
        return false;
    }

    if (!globalMemoryReserveHasCapacity(estimatedDeviceBytes))
    {
        static_cast<void>(cudaGraphExecDestroy(capturedGraph.second));
        static_cast<void>(cudaGraphDestroy(capturedGraph.first));
        ++mCudaGraphStats.globalMemoryReserveRejections;
        return false;
    }

    CapturedGraph captured{};
    captured.graph = capturedGraph.first;
    captured.exec = capturedGraph.second;
    captured.snapshot = snapshot;
    captured.estimatedDeviceBytes = estimatedDeviceBytes;
    if (existing != mGraphs.end())
    {
        eraseCapturedGraph(existing);
    }
    mGraphs[hash] = captured;
    mCachedGraphBytes += estimatedDeviceBytes;
    return true;
}

void EngineExecutor::eraseCapturedGraph(std::unordered_map<size_t, CapturedGraph>::iterator graph) noexcept
{
    mCachedGraphBytes -= std::min(mCachedGraphBytes, graph->second.estimatedDeviceBytes);
    static_cast<void>(cudaGraphExecDestroy(graph->second.exec));
    static_cast<void>(cudaGraphDestroy(graph->second.graph));
    mGraphs.erase(graph);
}

bool EngineExecutor::graphMemoryBudgetHasCapacity() const noexcept
{
    if (mMaxAutomaticGraphBytes == 0U)
    {
        return true;
    }
    if (mCachedGraphBytes >= mMaxAutomaticGraphBytes)
    {
        return false;
    }
    if (mGraphs.empty() || mCachedGraphBytes == 0U)
    {
        return mMinimumAutomaticGraphBytes <= mMaxAutomaticGraphBytes;
    }
    size_t const averageGraphBytes = (mCachedGraphBytes + mGraphs.size() - 1U) / mGraphs.size();
    size_t const nextGraphBytes = std::max(averageGraphBytes, mMinimumAutomaticGraphBytes);
    return nextGraphBytes <= mMaxAutomaticGraphBytes - mCachedGraphBytes;
}

bool EngineExecutor::globalMemoryReserveHasCapacity(size_t estimatedGraphBytes) const noexcept
{
    if (mMinimumFreeMemoryBytes == 0U)
    {
        return true;
    }
    size_t freeBytes{};
    size_t totalBytes{};
    if (cudaMemGetInfo(&freeBytes, &totalBytes) != cudaSuccess)
    {
        return false;
    }
    return freeBytes >= mMinimumFreeMemoryBytes && freeBytes - mMinimumFreeMemoryBytes >= estimatedGraphBytes;
}

size_t EngineExecutor::measureCapturedGraphBytes(size_t freeBytesBefore) const
{
    size_t freeBytesAfter{};
    size_t totalBytes{};
    CUDA_CHECK(cudaMemGetInfo(&freeBytesAfter, &totalBytes));
    size_t const measuredBytes = freeBytesBefore > freeBytesAfter ? freeBytesBefore - freeBytesAfter : 0U;
    if (measuredBytes > 0U || mGraphs.empty())
    {
        return std::max(measuredBytes, mMinimumAutomaticGraphBytes);
    }
    size_t const averageGraphBytes = (mCachedGraphBytes + mGraphs.size() - 1U) / mGraphs.size();
    return std::max(averageGraphBytes, mMinimumAutomaticGraphBytes);
}

void EngineExecutor::rememberBindingObservation(size_t hash, BindingSnapshot const& snapshot)
{
    if (!mAutomaticGraphCaptureEnabled)
    {
        return;
    }
    auto const existing = mObservedBindings.find(hash);
    if (existing != mObservedBindings.end())
    {
        existing->second = snapshot;
        return;
    }
    while (mObservedBindings.size() >= mMaxObservedBindings && !mObservedBindingOrder.empty())
    {
        size_t const oldest = mObservedBindingOrder.front();
        mObservedBindingOrder.pop_front();
        mObservedBindings.erase(oldest);
        ++mCudaGraphStats.observationEvictions;
    }
    mObservedBindings.emplace(hash, snapshot);
    mObservedBindingOrder.push_back(hash);
}

} // namespace rt
} // namespace trt_edgellm
