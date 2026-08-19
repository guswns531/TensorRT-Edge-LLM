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
#include <vector>

namespace trt_edgellm::rt
{

using IndependentPhaseInputCallback = std::function<void(std::vector<PhaseWorkItem> const&, PipelineIO&, cudaStream_t)>;
using IndependentPhaseFinishedCallback = std::function<bool(PhaseWorkItem const&, int32_t resultingKVLength)>;

struct IndependentPhaseCoordinatorCallbacks
{
    IndependentPhaseInputCallback stagePrefill;
    IndependentPhaseInputCallback stageDecode;
    IndependentPhaseFinishedCallback isPrefillFinished;
    IndependentPhaseFinishedCallback isDecodeFinished;
    std::function<void(PhaseDispatchMetrics const&)> onMetrics;
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

    void enqueuePrefill(PhaseWorkItem item);
    void enqueueDecode(PhaseWorkItem item);
    bool dispatchNext();
    bool poll();
    void wait();
    void runUntilIdle(size_t maxDispatches);

    bool empty() const noexcept;
    bool busy() const noexcept;
    PhaseQueueScheduler& scheduler() noexcept;
    std::vector<PhaseDispatchMetrics> const& metrics() const noexcept;
    CUcontext cudaContext() const noexcept;

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
};

} // namespace trt_edgellm::rt
