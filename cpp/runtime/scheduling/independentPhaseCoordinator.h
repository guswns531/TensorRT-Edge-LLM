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

#include "runtime/config/llmEngineConfig.h"
#include "runtime/scheduling/independentEngineExecutorPair.h"
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/scheduling/phaseKVActiveView.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/stableKVPageManager.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace trt_edgellm::rt
{

using IndependentPhaseInputCallback = std::function<void(std::vector<PhaseWorkItem> const&, PipelineIO&, cudaStream_t)>;
using IndependentPhaseBatchCompletionCallback
    = std::function<void(std::vector<PhaseWorkItem> const&, PipelineIO&, cudaStream_t)>;
using IndependentPhaseFinishedCallback = std::function<bool(PhaseWorkItem const&, int32_t resultingKVLength)>;

struct IndependentPhaseCoordinatorCallbacks
{
    IndependentPhaseInputCallback stagePrefill;
    IndependentPhaseInputCallback stageDecode;
    IndependentPhaseBatchCompletionCallback completePrefillBatch;
    IndependentPhaseBatchCompletionCallback completeDecodeBatch;
    IndependentPhaseFinishedCallback isPrefillFinished;
    IndependentPhaseFinishedCallback isDecodeFinished;
    std::function<void(PhaseDispatchMetrics const&)> onMetrics;
    std::function<void(PhaseTimelineEvent const&)> onTimeline;
};

//! Reusable queue-to-engine coordinator for independent prefill/decode TensorRT contexts.
//!
//! The coordinator owns scheduler/worker state and phase-local stable-KV views.
//! Request payload preparation remains injectable so text token embedding,
//! multimodal staging, or synthetic benchmark inputs can share the same queue
//! and execution path.
class IndependentPhaseCoordinator
{
public:
    IndependentPhaseCoordinator(LLMEngineConfig const& config, PhaseQueueSchedulerConfig schedulerConfig,
        IndependentEngineExecutorPair& executors, StableKVPageManager& ownership, PipelineIO& prefillIO,
        PipelineIO& decodeIO, TensorMap& prefillMap, TensorMap& decodeMap, cudaStream_t prefillStream,
        cudaStream_t decodeStream, IndependentPhaseCoordinatorCallbacks callbacks);
    ~IndependentPhaseCoordinator() noexcept = default;

    IndependentPhaseCoordinator(IndependentPhaseCoordinator const&) = delete;
    IndependentPhaseCoordinator& operator=(IndependentPhaseCoordinator const&) = delete;

    //! Replace request payload/completion hooks before the first dispatch.
    void setCallbacks(IndependentPhaseCoordinatorCallbacks callbacks);

    void enqueuePrefill(PhaseWorkItem item);
    void enqueueDecode(PhaseWorkItem item);
    bool dispatchNext();
    bool poll();
    void wait();
    void runUntilIdle(size_t maxDispatches);

    //! Capture graphs for the currently prepared phase bindings. Callers must
    //! prepare both phase views with stable shapes before invoking this method.
    bool capturePreparedGraphs();
    //! Capture a graph the first time each production phase shape is observed.
    void setGraphCaptureEnabled(bool enabled) noexcept;
    //! Bound production graph-cache shape counts per phase.
    void setGraphCaptureLimits(size_t maxPrefillGraphs, size_t maxDecodeGraphs) noexcept;
    EngineExecutor::GraphCacheStats prefillGraphCacheStats() const noexcept;
    EngineExecutor::GraphCacheStats decodeGraphCacheStats() const noexcept;

    bool empty() const noexcept;
    bool busy() const noexcept;
    TensorMap& prefillTensorMap() noexcept;
    TensorMap& decodeTensorMap() noexcept;
    PhaseQueueScheduler& scheduler() noexcept;
    std::vector<PhaseDispatchMetrics> const& metrics() const noexcept;
    //! Enable external metric retention; scheduler telemetry remains active either way.
    void setMetricsCollectionEnabled(bool enabled) noexcept;
    CUcontext cudaContext() const noexcept;
    PhaseKVMemoryStats const& prefillKVMemoryStats() const noexcept;
    PhaseKVMemoryStats const& decodeKVMemoryStats() const noexcept;
    KVPageTableUploadStats const& prefillPageTableUploadStats() const noexcept;
    KVPageTableUploadStats const& decodePageTableUploadStats() const noexcept;

private:
    PhaseDispatchWorkerCallbacks makeWorkerCallbacks();
    void enqueuePrefillBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream);
    void enqueueDecodeBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream);
    void completePrefillBatch(std::vector<PhaseWorkItem> const& batch);
    void completeDecodeBatch(std::vector<PhaseWorkItem> const& batch);

    LLMEngineConfig mConfig;
    IndependentEngineExecutorPair& mExecutors;
    StableKVPageManager& mOwnership;
    PipelineIO& mPrefillIO;
    PipelineIO& mDecodeIO;
    TensorMap& mPrefillMap;
    TensorMap& mDecodeMap;
    cudaStream_t mPrefillStream{};
    cudaStream_t mDecodeStream{};
    IndependentPhaseCoordinatorCallbacks mCallbacks;
    PhaseKVActiveView mPrefillKV;
    PhaseKVActiveView mDecodeKV;
    PhaseQueueScheduler mScheduler;
    std::unique_ptr<PhaseDispatchWorker> mWorker;
    std::vector<PhaseDispatchMetrics> mMetrics;
    bool mMetricsCollectionEnabled{true};
    std::unordered_set<std::string> mCapturedPrefillShapes;
    std::unordered_set<std::string> mCapturedDecodeShapes;
    bool mGraphCaptureEnabled{};
    size_t mMaxPrefillGraphs{};
    size_t mMaxDecodeGraphs{};
};

} // namespace trt_edgellm::rt
