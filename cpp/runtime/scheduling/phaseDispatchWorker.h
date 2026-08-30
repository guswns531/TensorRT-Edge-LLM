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
#include "runtime/scheduling/phaseTimeline.h"

#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <functional>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

enum class PhaseActivityKind : uint8_t;
class PhaseActivityTimelineRecorder;

//! Reorder one selected phase batch to retain prior request-to-row placement where possible.
//! The selected request set is unchanged; unassigned rows preserve their current relative order.
void preservePhaseBatchRowAffinity(
    std::vector<PhaseWorkItem>& batch, std::vector<uint64_t> const& previousRowRequestIds);

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
    //! Observe the selected batch composition immediately before phase enqueue.
    std::function<void(PhaseDispatchMetrics const&)> onDispatch;
    //! Observe one immutable timing record after all phase completions.
    std::function<void(PhaseDispatchMetrics const&)> onMetrics;
    //! Optional request-level host transition telemetry. Empty keeps the hot path allocation-free.
    std::function<void(PhaseTimelineEvent const&)> onTimeline;
    //! Exact eager/graph replay path observed by the executor for this plan.
    std::function<PhaseExecutionVariant(PhaseDispatchKind kind)> executionVariant;
};

//! Controls whether phase enqueues may overlap across streams.
enum class PhaseTensorRTContextMode
{
    //! One TensorRT execution context is shared, so engine enqueues are ordered.
    kSharedSerialized,
    //! Prefill and decode own independent execution contexts and may overlap.
    kIndependentConcurrent,
};

struct PhaseExecutionResourceIdentity
{
    void const* tensorRTExecutionContext{};
    void const* workspace{};
    void const* ioBuffers{};
};

//! Declares the resource-aliasing facts required by a phase execution mode.
struct PhaseExecutionSafetyContract
{
    PhaseExecutionResourceIdentity prefill;
    PhaseExecutionResourceIdentity decode;

    static PhaseExecutionSafetyContract shared(void const* tensorRTExecutionContext) noexcept;
    static PhaseExecutionSafetyContract independent(
        PhaseExecutionResourceIdentity prefill, PhaseExecutionResourceIdentity decode) noexcept;
    void validate(PhaseTensorRTContextMode mode) const;
    bool provesIndependentResources() const noexcept;
};

//! CUDA-event handoff between PhaseQueueScheduler and two execution streams in one CUDA primary context.
//!
//! V1 permits one DispatchPlan in flight. The default shared TensorRT context mode uses
//! separate phase streams but orders decode after prefill because one TensorRT
//! execution context cannot be enqueued concurrently. Independent TensorRT contexts may
//! opt into concurrent phase execution.
class PhaseDispatchWorker
{
public:
    PhaseDispatchWorker(PhaseQueueScheduler& scheduler, PhaseDispatchWorkerCallbacks callbacks,
        cudaStream_t prefillStream, cudaStream_t decodeStream,
        PhaseTensorRTContextMode executionMode = PhaseTensorRTContextMode::kSharedSerialized,
        PhaseExecutionSafetyContract safetyContract = {});
    ~PhaseDispatchWorker() noexcept;

    PhaseDispatchWorker(PhaseDispatchWorker const&) = delete;
    PhaseDispatchWorker& operator=(PhaseDispatchWorker const&) = delete;
    PhaseDispatchWorker(PhaseDispatchWorker&&) = delete;
    PhaseDispatchWorker& operator=(PhaseDispatchWorker&&) = delete;

    //! Dispatch the next scheduler plan without synchronizing the streams.
    //! Returns false when no queued work exists.
    bool dispatchNext();

    //! Add the currently idle P or D context to one live single-phase plan.
    //! The global scheduler must explicitly authorize the resulting P+D set.
    bool augmentNext(PhaseGlobalActionCandidate missingPhase, PhaseGlobalActionCandidate aggregate, uint64_t planId,
        uint64_t snapshotEpoch);

    //! Non-blocking event query. Returns true when an in-flight plan completed.
    bool poll();

    //! Wait for and complete the current in-flight plan.
    void wait();

    //! Run until queues and in-flight work are empty, with a runaway guard.
    void runUntilIdle(size_t maxDispatches);

    bool busy() const noexcept;
    bool empty() const noexcept;
    PhaseDispatchKind inFlightKind() const noexcept;
    PhasePrefillClass inFlightPrefillClass() const noexcept;
    size_t dispatchCount() const noexcept;
    std::optional<PhaseDispatchMetrics> const& lastMetrics() const noexcept;
    PhaseExecutionSafetyContract const& safetyContract() const noexcept;

    //! Enable opt-in epoch-relative P/D stream activity recording while idle.
    void setActivityTimeline(PhaseActivityTimelineRecorder* timeline);

    //! CUDA primary context shared by the two phase streams.
    CUcontext cudaContext() const noexcept;

private:
    void enqueueDeferredDecode();
    void mergeAugmentedMetrics(PhaseDispatchPlan const& additional, PhaseGlobalActionCandidate const& aggregate,
        uint64_t planId, uint64_t snapshotEpoch);
    void enqueueActivity(PhaseActivityKind kind, char const* name, std::vector<PhaseWorkItem> const& batch,
        PhaseEnqueueCallback const& callback, cudaStream_t stream);
    void completePrefillInFlight();
    void completeDecodeInFlight();
    void completeInFlight();
    void collectMetrics();
    void recordTimeline(
        std::vector<PhaseWorkItem> const& batch, PhaseTimelineStage stage, uint64_t timestampNs = 0U) const;
    bool eventReady(cudaEvent_t event) const;

    PhaseQueueScheduler& mScheduler;
    PhaseDispatchWorkerCallbacks mCallbacks;
    cudaStream_t mPrefillStream{};
    cudaStream_t mDecodeStream{};
    CUcontext mCudaContext{};
    PhaseTensorRTContextMode mExecutionMode{PhaseTensorRTContextMode::kSharedSerialized};
    PhaseExecutionSafetyContract mSafetyContract;
    cudaEvent_t mDispatchStart{};
    cudaEvent_t mAugmentationStart{};
    cudaEvent_t mPrefillStart{};
    cudaEvent_t mPrefillDone{};
    cudaEvent_t mDecodeStart{};
    cudaEvent_t mDecodeDone{};
    PhaseDispatchPlan mInFlight;
    PhaseDispatchMetrics mCurrentMetrics;
    std::optional<PhaseDispatchMetrics> mLastMetrics;
    bool mBusy{};
    bool mHasPrefill{};
    bool mHasDecode{};
    bool mDecodeDeferred{};
    bool mResidualAugmentation{};
    size_t mDispatchCount{};
    std::vector<uint64_t> mPreviousDecodeRowRequestIds;
    PhaseActivityTimelineRecorder* mActivityTimeline{};
};

} // namespace rt
} // namespace trt_edgellm
