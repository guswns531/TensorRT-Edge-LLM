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

#pragma once

#include "common/tensor.h"
#include "common/trtUtils.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/exec/tensorRegistry.h"
#include <NvInferRuntime.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*!
 * @brief Thin TRT wrapper with prepare/execute split.
 *
 * EngineExecutor owns a TRT execution context and shares the TRT runtime and engine
 * with any sibling executors. It replaces both
 * LLMEngineRunner and EagleDraftEngineRunner with a single model-agnostic
 * wrapper (~300 LOC).
 *
 *  - @c prepare() sets the optimization profile and delegates binding to
 *    TensorRegistry::bindAll().
 *  - @c execute() replays a cached CUDA graph when the binding state matches,
 *    otherwise falls back to enqueueV3.
 *  - @c captureGraph() captures a new CUDA graph for the current bindings.
 *
 * EngineExecutor knows nothing about models, phases, or features.
 */
class EngineExecutor
{
public:
    //! @brief CUDA graph cache counters for one TensorRT execution context.
    struct CudaGraphCacheStats
    {
        uint64_t enqueueExecutions{};
        uint64_t graphLaunches{};
        uint64_t captures{};
        uint64_t captureFailures{};
        uint64_t graphLaunchFailures{};
        uint64_t cacheLimitBypasses{};
        uint64_t postEnqueueCaptures{};
        uint64_t observationEvictions{};
        uint64_t graphMemoryBudgetBypasses{};
        uint64_t graphMemoryBudgetRejections{};
        uint64_t globalMemoryReserveBypasses{};
        uint64_t globalMemoryReserveRejections{};
        size_t cachedGraphs{};
        size_t observedBindingShapes{};
        size_t uncapturableBindingShapes{};
        size_t budgetRejectedBindingShapes{};
        size_t cachedGraphBytes{};
        size_t maxCachedGraphBytes{};
        size_t minimumGraphChargeBytes{};
        size_t minimumFreeMemoryBytes{};
        bool automaticCaptureEnabled{};
    };

    //! @brief Destructor — destroys all captured CUDA graphs.
    ~EngineExecutor() noexcept;

    //! Build an EngineExecutor for a vanilla single-engine LLM or a SpecDecode
    //! base engine. The factory builds the TensorRegistry internally via
    //! `buildRegistryForLLM(cfg)`.
    static std::unique_ptr<EngineExecutor> createForLLM(std::filesystem::path const& enginePath,
        LLMEngineConfig const& cfg, std::optional<int32_t> specDecodeBaseOutputHiddenDim = std::nullopt);

    //! Build an EngineExecutor for a speculative decoding draft engine. The
    //! factory chooses the draft binding registry from `bundle.specDecodeMode()`.
    static std::unique_ptr<EngineExecutor> createForDraft(
        std::filesystem::path const& enginePath, DeploymentConfig const& bundle);

    //! @brief Create an executor with an independent execution context over the same engine.
    //!
    //! The sibling has its own USER_MANAGED context, auxiliary streams, and CUDA graph
    //! cache. Serialized weights and the ICudaEngine are shared, allowing two phase
    //! streams to enqueue concurrently without deserializing a second engine.
    std::unique_ptr<EngineExecutor> createSibling() const;

    EngineExecutor(EngineExecutor const&) = delete;
    EngineExecutor& operator=(EngineExecutor const&) = delete;

    /*!
     * @brief Switch optimization profile, resolve shapes, bind all tensors.
     *
     * @param profileIndex TRT optimization profile index
     * @param dims Symbolic dimension values for this step
     * @param map Name-to-tensor mapping
     * @param stream CUDA stream for the async profile switch
     * @return True on success
     */
    bool prepare(int32_t profileIndex, InferenceDims const& dims, TensorMap const& map, cudaStream_t stream);

    /*!
     * @brief Execute inference.
     *
     * Replays a cached CUDA graph if one matches the current bindings,
     * otherwise falls back to enqueueV3.
     *
     * @param stream CUDA stream
     * @return True on success
     */
    bool execute(cudaStream_t stream);

    /*!
     * @brief Capture a CUDA graph for the current binding state (after prepare()).
     *
     * Performs a warmup enqueue, then captures via cudaStreamBeginCapture.
     * The captured graph is keyed by a binding hash with full snapshot
     * verification.
     *
     * @param stream CUDA stream (must not be the default stream)
     * @return True if capture succeeded
     */
    bool captureGraph(cudaStream_t stream);

    //! @brief Enable capture-on-repeat for stable binding states.
    //!
    //! The first execution of a binding state uses enqueueV3 to let TensorRT
    //! apply deferred dynamic-shape updates. A consecutive recurrence is
    //! captured and launched once. A non-consecutive recurrence is enqueued
    //! once, then captured without launch for future reuse. Neither path adds
    //! an extra inference on live KV-cache state. A zero byte budget is
    //! unlimited. minimumGraphChargeBytes accounts conservatively for CUDA
    //! allocations that become visible only on first graph launch.
    void enableAutomaticCudaGraphCapture(size_t maxCachedGraphs, size_t maxCachedGraphBytes = 0U,
        size_t minimumGraphChargeBytes = 0U, size_t minimumFreeMemoryBytes = 0U);

    //! @brief Return CUDA graph cache counters for this execution context.
    CudaGraphCacheStats getCudaGraphCacheStats() const noexcept;

    /*!
     * @brief Query required device memory for the execution context.
     * @return Required memory size in bytes
     */
    int64_t getRequiredContextMemorySize() const;

    //! @brief Query the upper-bound context memory for one optimization profile.
    int64_t getRequiredContextMemorySizeForProfile(int32_t profileIndex) const;

    /*!
     * @brief Provide shared device memory for the execution context.
     *
     * @param sharedMem Tensor whose memory will back the TRT context
     * @return True on success
     */
    bool setContextMemory(Tensor& sharedMem);

    //! @brief Select one fixed profile before assigning profile-sized context memory.
    bool setContextMemoryForProfile(int32_t profileIndex, Tensor& sharedMem, cudaStream_t stream);

    //! @brief Return the number of I/O tensors in the engine.
    int32_t getNumIOTensors() const;

    //! @brief Return the name of the i-th I/O tensor.
    char const* getIOTensorName(int32_t index) const;

    //! @brief Return whether the engine exposes a named I/O tensor.
    bool hasIOTensor(char const* name) const;

    //! @brief Return the data type of a named binding.
    nvinfer1::DataType getBindingDataType(char const* name) const;

    //! @brief Return a profile shape (min/opt/max) for a named binding.
    nvinfer1::Dims getProfileShape(char const* name, int32_t profileIndex, nvinfer1::OptProfileSelector selector) const;

    //! @brief Attach a TRT profiler to the execution context.
    //!
    //! The profiler receives per-layer timing callbacks during enqueueV3.
    //! Must be called before execute() for the profiler to receive data.
    //! Passing nullptr detaches any previously set profiler.
    void setProfiler(nvinfer1::IProfiler* profiler) noexcept;

    //! @brief Access the underlying TRT engine for generic introspection.
    nvinfer1::ICudaEngine const& getEngine() const noexcept;

    //! @brief Return the owned TensorRT execution context identity.
    nvinfer1::IExecutionContext const* getExecutionContextIdentity() const noexcept;

    //! @brief Snapshot of all binding addresses and shapes — used for graph-cache verification.
    struct BindingSnapshot
    {
        std::vector<std::pair<uintptr_t, nvinfer1::Dims>> bindings;

        bool operator==(BindingSnapshot const& rhs) const noexcept;
    };

private:
    /*!
     * @brief Construct a EngineExecutor from a serialized TRT engine file.
     *
     * Reads the engine, creates an IRuntime, deserializes the engine,
     * and creates an IExecutionContext with USER_MANAGED allocation.
     *
     * Private — use `createForLLM` / `createForDraft` factories.
     *
     * @param enginePath Path to the serialized TRT engine file
     * @param registry TensorRegistry describing the binding layout
     * @throws std::runtime_error On I/O or deserialization failure
     */
    struct SharedEngineState;

    EngineExecutor(std::filesystem::path const& enginePath, TensorRegistry registry);
    EngineExecutor(std::shared_ptr<SharedEngineState> engineState, TensorRegistry registry);

    void createExecutionContext();

    AuxStreamSet mAuxStreams{};
    std::shared_ptr<SharedEngineState> mEngineState;
    std::unique_ptr<nvinfer1::IExecutionContext> mContext;
    TensorRegistry mRegistry;

    //! A captured CUDA graph together with its binding snapshot for verification.
    struct CapturedGraph
    {
        cudaGraph_t graph{nullptr};
        cudaGraphExec_t exec{nullptr};
        BindingSnapshot snapshot;
        size_t estimatedDeviceBytes{};
    };

    //! Graph cache keyed by a hash of all binding addresses + shapes.
    std::unordered_map<size_t, CapturedGraph> mGraphs;

    bool mAutomaticGraphCaptureEnabled{};
    size_t mMaxAutomaticGraphs{};
    size_t mMaxAutomaticGraphBytes{};
    size_t mMinimumAutomaticGraphBytes{};
    size_t mMinimumFreeMemoryBytes{};
    size_t mCachedGraphBytes{};
    size_t mMaxObservedBindings{};
    std::optional<BindingSnapshot> mLastSuccessfulBindings;
    std::unordered_map<size_t, BindingSnapshot> mObservedBindings;
    std::deque<size_t> mObservedBindingOrder;
    std::unordered_map<size_t, BindingSnapshot> mUncapturableBindings;
    std::unordered_map<size_t, BindingSnapshot> mBudgetRejectedBindings;
    CudaGraphCacheStats mCudaGraphStats;

    //! Hash all current binding addresses and shapes into a single key.
    size_t computeBindingHash() const;

    //! Build a full snapshot of the current binding state.
    BindingSnapshot snapshotBindings() const;

    //! Enqueue normally and remember the binding state if successful.
    bool enqueueAndRemember(size_t hash, BindingSnapshot const& snapshot, cudaStream_t stream);

    //! Capture the current execution, launch it once, and cache it on success.
    bool captureLaunchAndCache(size_t hash, BindingSnapshot const& snapshot, cudaStream_t stream);

    //! Execute the current inference once, then capture without launching the
    //! graph. This safely materializes a non-consecutive recurring shape.
    bool enqueueCaptureAndCache(size_t hash, BindingSnapshot const& snapshot, cudaStream_t stream);

    bool cacheCapturedGraph(size_t hash, BindingSnapshot const& snapshot,
        std::pair<cudaGraph_t, cudaGraphExec_t> const& captured, size_t estimatedDeviceBytes);
    void eraseCapturedGraph(std::unordered_map<size_t, CapturedGraph>::iterator graph) noexcept;
    bool graphMemoryBudgetHasCapacity() const noexcept;
    bool globalMemoryReserveHasCapacity(size_t estimatedGraphBytes) const noexcept;
    size_t measureCapturedGraphBytes(size_t freeBytesBefore) const;
    void rememberBindingObservation(size_t hash, BindingSnapshot const& snapshot);
};

} // namespace rt
} // namespace trt_edgellm
