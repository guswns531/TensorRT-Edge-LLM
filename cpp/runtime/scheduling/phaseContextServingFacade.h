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
#include "runtime/scheduling/phaseRequestLifecycle.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

using PhasePackedContextCallback = std::function<void(DecodingInferenceContext&)>;
using PhaseContextFinishedCallback
    = std::function<bool(uint64_t requestId, DecodingInferenceContext const& context, int32_t contextRow)>;

struct PhaseContextServingCallbacks
{
    //! Enqueue one prefill chunk batch on the provided phase stream.
    PhaseEnqueueCallback enqueuePrefill;
    //! Run batch-level host completion after the prefill CUDA event.
    PhaseBatchCompletionCallback completePrefillBatch;
    //! Return the KV length after an individual prefill chunk completes.
    PrefillCompletionCallback completePrefill;
    //! Enqueue exactly one vanilla decode token for the packed context.
    PhasePackedContextCallback enqueueDecode;
    //! Update packed token state after the decode CUDA event.
    PhasePackedContextCallback completeDecode;
    //! Decide whether a scattered source row has reached its terminal state.
    PhaseContextFinishedCallback isDecodeFinished;
    //! Observe finished or cancelled requests after their stable slot is released.
    std::function<void(PhaseRequestSnapshot const&)> onTerminal;
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
        PhaseStreamExecutionMode executionMode = PhaseStreamExecutionMode::kSharedContextSerialized);

    PhaseContextServingFacade(PhaseContextServingFacade const&) = delete;
    PhaseContextServingFacade& operator=(PhaseContextServingFacade const&) = delete;
    PhaseContextServingFacade(PhaseContextServingFacade&&) = delete;
    PhaseContextServingFacade& operator=(PhaseContextServingFacade&&) = delete;

    //! Register a borrowed source row and reserve its stable physical KV slot.
    //! The source context must remain alive until terminal completion or cancellation.
    //! @return The leased physical KV slot ID.
    int32_t submit(uint64_t requestId, DecodingInferenceContext& context, int32_t contextRow, int32_t promptTokenCount);
    //! Cancel queued work. In-flight work can be cancelled after its event completes.
    bool cancel(uint64_t requestId);

    bool dispatchNext();
    bool poll();
    void wait();
    void runUntilIdle(size_t maxDispatches);

    bool empty() const noexcept;
    bool busy() const noexcept;
    size_t activeRequestCount() const noexcept;
    size_t registeredRequestCount() const noexcept;
    int32_t availableSlotCount() const noexcept;
    std::optional<PhaseRequestSnapshot> request(uint64_t requestId) const;

private:
    struct Registration
    {
        DecodingInferenceContext* context{};
        int32_t contextRow{-1};
    };

    PhaseRequestLifecycleCallbacks makeLifecycleCallbacks();
    void enqueuePrefillBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream);
    void enqueueDecodeBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream);
    void completeDecodeBatch(std::vector<PhaseWorkItem> const& batch);
    Registration& registration(uint64_t requestId);
    Registration const& registration(uint64_t requestId) const;

    PhaseContextServingCallbacks mCallbacks;
    HybridCacheManager& mCacheManager;
    Tensor mHostAdmissionSlotIds;
    Tensor mDeviceAdmissionSlotIds;
    PhaseContextBatchAdapter mDecodeAdapter;
    std::unordered_map<uint64_t, Registration> mRegistrations;
    std::unique_ptr<PhaseRequestLifecycle> mLifecycle;
};

} // namespace rt
} // namespace trt_edgellm
