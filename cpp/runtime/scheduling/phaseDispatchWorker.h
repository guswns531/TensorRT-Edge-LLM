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

#include "runtime/scheduling/phaseQueueScheduler.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <functional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct PhasePrefillCompletion
{
    int32_t resultingKVLength{};
    bool finished{};
};

struct PhaseDecodeCompletion
{
    int32_t resultingKVLength{};
    bool finished{};
};

using PhaseEnqueueCallback = std::function<void(std::vector<PhaseWorkItem> const&, cudaStream_t)>;
using PhaseBatchCompletionCallback = std::function<void(std::vector<PhaseWorkItem> const&)>;
using PrefillCompletionCallback = std::function<PhasePrefillCompletion(PhaseWorkItem const&)>;
using DecodeCompletionCallback = std::function<PhaseDecodeCompletion(PhaseWorkItem const&)>;

struct PhaseDispatchWorkerCallbacks
{
    PhaseEnqueueCallback enqueuePrefill;
    PhaseEnqueueCallback enqueueDecode;
    //! Optional batch-level host completion after the corresponding CUDA event.
    PhaseBatchCompletionCallback completePrefillBatch;
    PhaseBatchCompletionCallback completeDecodeBatch;
    PrefillCompletionCallback completePrefill;
    DecodeCompletionCallback completeDecode;
};

//! Controls whether phase enqueues may overlap across streams.
enum class PhaseStreamExecutionMode
{
    //! One TensorRT execution context is shared, so engine enqueues are ordered.
    kSharedContextSerialized,
    //! Prefill and decode own independent execution contexts and may overlap.
    kIndependentContextsConcurrent,
};

//! CUDA-event handoff between PhaseQueueScheduler and two execution streams.
//!
//! V1 permits one DispatchPlan in flight. The default shared-context mode uses
//! separate phase streams but orders decode after prefill because one TensorRT
//! execution context cannot be enqueued concurrently. Independent contexts may
//! opt into concurrent phase execution.
class PhaseDispatchWorker
{
public:
    PhaseDispatchWorker(PhaseQueueScheduler& scheduler, PhaseDispatchWorkerCallbacks callbacks,
        cudaStream_t prefillStream, cudaStream_t decodeStream,
        PhaseStreamExecutionMode executionMode = PhaseStreamExecutionMode::kSharedContextSerialized);
    ~PhaseDispatchWorker() noexcept;

    PhaseDispatchWorker(PhaseDispatchWorker const&) = delete;
    PhaseDispatchWorker& operator=(PhaseDispatchWorker const&) = delete;
    PhaseDispatchWorker(PhaseDispatchWorker&&) = delete;
    PhaseDispatchWorker& operator=(PhaseDispatchWorker&&) = delete;

    //! Dispatch the next scheduler plan without synchronizing the streams.
    //! Returns false when no queued work exists.
    bool dispatchNext();

    //! Non-blocking event query. Returns true when an in-flight plan completed.
    bool poll();

    //! Wait for and complete the current in-flight plan.
    void wait();

    //! Run until queues and in-flight work are empty, with a runaway guard.
    void runUntilIdle(size_t maxDispatches);

    bool busy() const noexcept;
    size_t dispatchCount() const noexcept;

private:
    void enqueueDeferredDecode();
    void completePrefillInFlight();
    void completeDecodeInFlight();
    void completeInFlight();
    bool eventReady(cudaEvent_t event) const;

    PhaseQueueScheduler& mScheduler;
    PhaseDispatchWorkerCallbacks mCallbacks;
    cudaStream_t mPrefillStream{};
    cudaStream_t mDecodeStream{};
    PhaseStreamExecutionMode mExecutionMode{PhaseStreamExecutionMode::kSharedContextSerialized};
    cudaEvent_t mDispatchStart{};
    cudaEvent_t mPrefillDone{};
    cudaEvent_t mDecodeDone{};
    PhaseDispatchPlan mInFlight;
    bool mBusy{};
    bool mHasPrefill{};
    bool mHasDecode{};
    bool mDecodeDeferred{};
    size_t mDispatchCount{};
};

} // namespace rt
} // namespace trt_edgellm
