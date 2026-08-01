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

#include "runtime/scheduling/phaseDispatchWorker.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <deque>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Encoder work that keeps the request's future KV slot lease across handoff.
struct PhaseEncoderWorkItem
{
    uint64_t requestId{};
    int32_t inputUnits{};
    int32_t kvSlotId{-1};
};

struct PhaseEncoderDispatchMetrics
{
    size_t dispatchIndex{};
    int32_t batchSize{};
    int32_t inputUnits{};
    double queueWaitUs{};
    float gpuMs{};
};

struct PhaseEncoderQueueConfig
{
    int32_t maxBatchSize{1};
    //! Zero means unbounded. In production this should match admission capacity.
    size_t maxQueuedRequests{};
};

//! Proves that an encoder context can overlap every listed LLM resource.
struct PhaseEncoderExecutionSafetyContract
{
    PhaseExecutionResourceIdentity encoder;
    std::vector<PhaseExecutionResourceIdentity> llmResources;

    void validate() const;
};

using PhaseEncoderEnqueueCallback
    = std::function<void(std::vector<PhaseEncoderWorkItem> const&, cudaStream_t)>;
using PhaseEncoderBatchCompletionCallback
    = std::function<void(std::vector<PhaseEncoderWorkItem> const&)>;
//! Performs host-side encoder output finalization and returns ready prefill work.
using PhaseEncoderCompletionCallback = std::function<PhaseWorkItem(PhaseEncoderWorkItem const&)>;

struct PhaseEncoderDispatchWorkerCallbacks
{
    PhaseEncoderEnqueueCallback enqueueEncoder;
    PhaseEncoderBatchCompletionCallback completeEncoderBatch;
    PhaseEncoderCompletionCallback completeEncoder;
    std::function<void(PhaseEncoderDispatchMetrics const&)> onMetrics;
};

//! Bounded encoder queue, CUDA-stream dispatch, event completion, and prefill handoff.
//!
//! One encoder batch may be in flight. The worker does not own model-specific
//! tensors: callbacks pack encoder input and finalize output after the event.
class PhaseEncoderDispatchWorker
{
public:
    PhaseEncoderDispatchWorker(PhaseQueueScheduler& prefillScheduler, PhaseEncoderQueueConfig config,
        PhaseEncoderDispatchWorkerCallbacks callbacks, cudaStream_t encoderStream,
        PhaseEncoderExecutionSafetyContract safetyContract);
    ~PhaseEncoderDispatchWorker() noexcept;

    PhaseEncoderDispatchWorker(PhaseEncoderDispatchWorker const&) = delete;
    PhaseEncoderDispatchWorker& operator=(PhaseEncoderDispatchWorker const&) = delete;
    PhaseEncoderDispatchWorker(PhaseEncoderDispatchWorker&&) = delete;
    PhaseEncoderDispatchWorker& operator=(PhaseEncoderDispatchWorker&&) = delete;

    void submit(PhaseEncoderWorkItem item);
    //! Queued work is cancellable; in-flight work must reach its CUDA event.
    bool cancel(uint64_t requestId);
    bool dispatchNext();
    bool poll();
    void wait();
    void runUntilIdle(size_t maxDispatches);

    size_t queueSize() const noexcept;
    bool busy() const noexcept;
    bool empty() const noexcept;
    bool hasRequest(uint64_t requestId) const noexcept;
    size_t dispatchCount() const noexcept;
    std::optional<PhaseEncoderDispatchMetrics> const& lastMetrics() const noexcept;
    PhaseEncoderExecutionSafetyContract const& safetyContract() const noexcept;

private:
    void completeInFlight();
    bool eventReady() const;

    PhaseQueueScheduler& mPrefillScheduler;
    PhaseEncoderQueueConfig mConfig;
    PhaseEncoderDispatchWorkerCallbacks mCallbacks;
    cudaStream_t mEncoderStream{};
    PhaseEncoderExecutionSafetyContract mSafetyContract;
    cudaEvent_t mEncoderStart{};
    cudaEvent_t mEncoderDone{};
    std::deque<PhaseEncoderWorkItem> mQueue;
    std::unordered_set<uint64_t> mActiveRequestIds;
    std::unordered_set<int32_t> mActiveKVSlotIds;
    std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> mQueuedSince;
    std::vector<PhaseEncoderWorkItem> mInFlight;
    PhaseEncoderDispatchMetrics mCurrentMetrics;
    std::optional<PhaseEncoderDispatchMetrics> mLastMetrics;
    bool mBusy{};
    size_t mDispatchCount{};
};

} // namespace rt
} // namespace trt_edgellm
