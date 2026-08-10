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

#include "runtime/scheduling/phaseContextBatchAdapter.h"
#include "runtime/scheduling/phasePrefillContextBatchAdapter.h"
#include "runtime/scheduling/phaseRequestLifecycle.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

using PhasePackedContextCallback = std::function<void(DecodingInferenceContext&)>;
using PhasePackedPrefillCallback = std::function<void(PhasePrefillContextBatchAdapter&)>;
using PhasePackedDecodeCallback = std::function<void(PhaseContextBatchAdapter&)>;
using PhaseContextFinishedCallback
    = std::function<bool(uint64_t requestId, DecodingInferenceContext const& context, int32_t contextRow)>;

enum class PhaseAdmissionStatus
{
    kAdmitted,
    kPending,
};

struct PhaseAdmissionResult
{
    uint64_t requestId{};
    int32_t kvSlotId{-1};
    PhaseAdmissionStatus status{PhaseAdmissionStatus::kPending};
    int32_t availableSlots{};
    size_t pendingQueueDepth{};
    KVPagePoolStats pagePool;
};

struct PhaseContextServingCallbacks
{
    //! Legacy work-item-only prefill enqueue callback.
    PhaseEnqueueCallback enqueuePrefill;
    //! Production prefill callback with packed source tokens and stable KV bindings.
    PhasePackedPrefillCallback enqueuePackedPrefill;
    //! Run packed prefill completion after its CUDA event and before binding restore.
    PhasePackedPrefillCallback completePackedPrefill;
    //! Run batch-level host completion after the prefill CUDA event.
    PhaseBatchCompletionCallback completePrefillBatch;
    //! Return the KV length after an individual prefill chunk completes.
    std::function<int32_t(PhaseWorkItem const&)> completePrefill;
    //! Decide whether final-prefill sampling reached EOS or the generation limit.
    PhaseContextFinishedCallback isPrefillFinished;
    //! Legacy packed-context decode enqueue callback.
    PhasePackedContextCallback enqueueDecode;
    //! Production decode callback with device current-token staging and stable KV bindings.
    PhasePackedDecodeCallback enqueuePackedDecode;
    //! Legacy packed-context host completion after the decode CUDA event.
    PhasePackedContextCallback completeDecode;
    //! Production decode completion before source-row scatter.
    PhasePackedDecodeCallback completePackedDecode;
    //! Decide whether a scattered source row has reached its terminal state.
    PhaseContextFinishedCallback isDecodeFinished;
    //! Observe finished or cancelled requests after their stable slot is released.
    std::function<void(PhaseRequestSnapshot const&)> onTerminal;
    //! Observe immediate or deferred stable-slot admission.
    std::function<void(PhaseAdmissionResult const&)> onAdmission;
    //! Observe queue residence and CUDA-event timing for each dispatch.
    std::function<void(PhaseDispatchMetrics const&)> onDispatchMetrics;
    //! Observe selected P/D batch composition immediately before enqueue.
    std::function<void(PhaseDispatchMetrics const&)> onDispatch;
};

//! Connects continuous request admission to stable-slot packed decode execution.
//!
//! Source DecodingInferenceContext objects are borrowed and must outlive their
//! registration. One packed decode batch may be in flight at a time, matching
//! PhaseDispatchWorker's V1 execution contract.
class PhaseContextServingFacade
{
public:
    PhaseContextServingFacade(int32_t maxSlots, PhaseQueueSchedulerConfig schedulerConfig,
        PhaseContextServingCallbacks callbacks, HybridCacheManager& cacheManager, TensorMap& decodeTensorMap,
        cudaStream_t prefillStream, cudaStream_t decodeStream,
        PhaseTensorRTContextMode executionMode = PhaseTensorRTContextMode::kSharedSerialized,
        TensorMap* prefillTensorMap = nullptr, int32_t maxPrefillChunkTokens = 0, size_t maxPendingAdmissions = 0,
        PhaseExecutionSafetyContract safetyContract = {});

    PhaseContextServingFacade(PhaseContextServingFacade const&) = delete;
    PhaseContextServingFacade& operator=(PhaseContextServingFacade const&) = delete;
    PhaseContextServingFacade(PhaseContextServingFacade&&) = delete;
    PhaseContextServingFacade& operator=(PhaseContextServingFacade&&) = delete;

    //! Register a borrowed source row and reserve its stable physical KV slot.
    //! The source context must remain alive until terminal completion or cancellation.
    //! @return The leased physical KV slot ID.
    int32_t submit(uint64_t requestId, DecodingInferenceContext& context, int32_t contextRow, int32_t promptTokenCount,
        PhaseSchedulingHints scheduling = {});
    //! Register source state and lease a slot before asynchronous encoder execution.
    int32_t reserveForEncoder(uint64_t requestId, DecodingInferenceContext& context, int32_t contextRow,
        int32_t promptTokenCountEstimate, PhaseSchedulingHints scheduling = {});
    //! Make encoder-produced tokens and embeddings runnable by the prefill scheduler.
    void beginPrefillAfterEncoder(PhaseWorkItem const& item);
    //! Admit immediately when a slot is free, otherwise apply bounded queue backpressure.
    PhaseAdmissionResult submitOrQueue(uint64_t requestId, DecodingInferenceContext& context, int32_t contextRow,
        int32_t promptTokenCount, PhaseSchedulingHints scheduling = {});
    //! Cancel queued work. In-flight work can be cancelled after its event completes.
    bool cancel(uint64_t requestId);

    bool dispatchNext();
    bool poll();
    void wait();
    void runUntilIdle(size_t maxDispatches);

    bool empty() const noexcept;
    bool hasQueuedPhaseWork() const noexcept;
    bool busy() const noexcept;
    size_t activeRequestCount() const noexcept;
    size_t pendingRequestCount() const noexcept;
    size_t registeredRequestCount() const noexcept;
    int32_t availableSlotCount() const noexcept;
    std::optional<PhaseRequestSnapshot> request(uint64_t requestId) const;
    CUcontext cudaContext() const noexcept;
    //! Add a terminal observer without replacing the callbacks installed by the executor owner.
    size_t addTerminalObserver(std::function<void(PhaseRequestSnapshot const&)> observer);
    void removeTerminalObserver(size_t observerId) noexcept;

private:
    struct Registration
    {
        DecodingInferenceContext* context{};
        int32_t contextRow{-1};
    };

    struct PendingAdmission
    {
        uint64_t requestId{};
        int32_t promptTokenCount{};
        PhaseSchedulingHints scheduling;
    };

    PhaseRequestLifecycleCallbacks makeLifecycleCallbacks();
    void enqueuePrefillBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream);
    void enqueueDecodeBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream);
    void completeDecodeBatch(std::vector<PhaseWorkItem> const& batch);
    void registerSource(uint64_t requestId, DecodingInferenceContext& context, int32_t contextRow);
    void admitPendingRequests();
    Registration& registration(uint64_t requestId);
    Registration const& registration(uint64_t requestId) const;

    PhaseContextServingCallbacks mCallbacks;
    HybridCacheManager& mCacheManager;
    Tensor mHostAdmissionSlotIds;
    Tensor mDeviceAdmissionSlotIds;
    std::unique_ptr<PhasePrefillContextBatchAdapter> mPrefillAdapter;
    PhaseContextBatchAdapter mDecodeAdapter;
    std::unordered_map<uint64_t, Registration> mRegistrations;
    std::deque<PendingAdmission> mPendingAdmissions;
    size_t mMaxPendingAdmissions{};
    bool mPendingAdmissionRequired{};
    size_t mNextTerminalObserverId{1};
    std::unordered_map<size_t, std::function<void(PhaseRequestSnapshot const&)>> mTerminalObservers;
    std::unique_ptr<PhaseRequestLifecycle> mLifecycle;
};

} // namespace rt
} // namespace trt_edgellm
