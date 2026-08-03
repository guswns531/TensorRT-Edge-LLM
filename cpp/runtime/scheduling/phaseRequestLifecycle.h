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

#include "runtime/kvSlotAllocator.h"
#include "runtime/scheduling/phaseDispatchWorker.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

enum class PhaseRequestStatus
{
    kPending,
    kEncoder,
    kPrefill,
    kDecode,
    kFinished,
    kCancelled,
};

struct PhaseRequestSnapshot
{
    PhaseRequestSnapshot() = default;
    PhaseRequestSnapshot(uint64_t requestId, int32_t kvSlotId, int32_t promptTokenCount, int32_t kvLength,
        PhaseRequestStatus status, PhaseSchedulingHints scheduling = {})
        : requestId(requestId)
        , kvSlotId(kvSlotId)
        , promptTokenCount(promptTokenCount)
        , kvLength(kvLength)
        , status(status)
        , scheduling(scheduling)
    {
    }

    uint64_t requestId{};
    int32_t kvSlotId{-1};
    int32_t promptTokenCount{};
    int32_t kvLength{};
    PhaseRequestStatus status{PhaseRequestStatus::kPrefill};
    PhaseSchedulingHints scheduling;
};

struct PhaseRequestLifecycleCallbacks
{
    PhaseDispatchWorkerCallbacks execution;
    std::function<void(PhaseRequestSnapshot const&)> onTerminal;
};

//! Owns request-to-slot leases and continuous prefill/decode queue transitions.
class PhaseRequestLifecycle
{
public:
    PhaseRequestLifecycle(int32_t maxSlots, PhaseQueueSchedulerConfig schedulerConfig,
        PhaseRequestLifecycleCallbacks callbacks, cudaStream_t prefillStream, cudaStream_t decodeStream,
        PhaseTensorRTContextMode executionMode = PhaseTensorRTContextMode::kSharedSerialized,
        PhaseExecutionSafetyContract safetyContract = {});

    //! Reserve stable KV ownership without making prefill runnable yet.
    int32_t reserveForEncoder(
        uint64_t requestId, int32_t promptTokenCountEstimate, PhaseSchedulingHints scheduling = {});
    //! Transition a reserved encoder request to the prefill queue.
    void beginPrefill(uint64_t requestId, int32_t promptTokenCount, bool allowChunkedPrefill = true);
    int32_t submit(uint64_t requestId, int32_t promptTokenCount, PhaseSchedulingHints scheduling = {});
    bool cancel(uint64_t requestId);

    bool dispatchNext();
    bool poll();
    void wait();
    void runUntilIdle(size_t maxDispatches);

    bool empty() const noexcept;
    bool hasQueuedWork() const noexcept;
    bool busy() const noexcept;
    size_t activeRequestCount() const noexcept;
    int32_t availableSlotCount() const noexcept;
    std::optional<PhaseRequestSnapshot> request(uint64_t requestId) const;
    CUcontext cudaContext() const noexcept;

private:
    PhaseDispatchWorkerCallbacks makeWorkerCallbacks();

    struct RequestState
    {
        PhaseRequestSnapshot snapshot;
    };

    PhaseQueueScheduler mScheduler;
    KVSlotAllocator mSlotAllocator;
    PhaseRequestLifecycleCallbacks mCallbacks;
    std::unordered_map<uint64_t, RequestState> mRequests;
    std::unique_ptr<PhaseDispatchWorker> mWorker;
};

} // namespace rt
} // namespace trt_edgellm
