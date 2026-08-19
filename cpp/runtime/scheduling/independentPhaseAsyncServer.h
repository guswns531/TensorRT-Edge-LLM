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

#include "runtime/scheduling/independentPhaseCoordinator.h"
#include "runtime/scheduling/phasePrefixReuseCache.h"

#include <cuda_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm::rt
{

//! Request view passed to a model-specific text or multimodal adapter.
struct IndependentPhaseRequestView
{
    uint64_t requestId{};
    PhaseWorkItem work;
    std::vector<int32_t> const* promptTokens{};
    std::vector<int32_t> const* generatedTokens{};
};

//! CUDA event plus host collection callback for asynchronous greedy sampling.
//! collect() is called only after ready has completed and returns one token per requestId.
struct IndependentPhaseSampleTicket
{
    cudaEvent_t ready{};
    bool fromPrefill{};
    std::vector<uint64_t> requestIds;
    std::function<std::vector<int32_t>()> collect;
};

//! Model-specific seam for token staging, embeddings, deepstack/M-RoPE binding, and sampling.
struct IndependentPhaseRequestAdapter
{
    using StageCallback
        = std::function<void(std::vector<IndependentPhaseRequestView> const&, PipelineIO&, TensorMap&, cudaStream_t)>;
    using SampleCallback = std::function<std::unique_ptr<IndependentPhaseSampleTicket>(
        std::vector<IndependentPhaseRequestView> const&, PipelineIO&, cudaStream_t, bool fromPrefill)>;

    StageCallback stagePrefill;
    StageCallback stageDecode;
    SampleCallback submitSampling;
    //! The adapter supplies absolute position/RoPE metadata for a reused suffix.
    bool supportsPageAlignedPrefixReuse{};
};

enum class IndependentPhaseServerStatus
{
    kAdmitted,
    kBackpressure,
    kDuplicateRequest,
    kCompleted,
    kCancelled,
};

struct IndependentPhaseServerConfig
{
    size_t maxInFlightRequests{};
    int32_t defaultMaxOutputTokens{128};
    std::vector<int32_t> eosTokenIds;
    bool enablePrefixReuse{};
};

struct IndependentPhaseServerSubmission
{
    uint64_t requestId{};
    IndependentPhaseServerStatus status{IndependentPhaseServerStatus::kBackpressure};
    int32_t kvSlotId{-1};
    int32_t reusedPrefixTokens{};
};

struct IndependentPhaseServerCompletion
{
    uint64_t requestId{};
    std::vector<int32_t> generatedTokens;
    double latencyMs{};
};

//! Production-facing event-loop facade over independent prefill/decode contexts.
//!
//! The facade owns request admission, stable-slot leases, prefix sharing, and
//! asynchronous sampling tickets. Model-specific CUDA work remains in the adapter.
class IndependentPhaseAsyncServer
{
public:
    IndependentPhaseAsyncServer(IndependentPhaseServerConfig config, IndependentPhaseCoordinator& coordinator,
        StableKVPageManager& ownership, IndependentPhaseRequestAdapter adapter,
        PhasePrefixReuseCache* prefixCache = nullptr);
    ~IndependentPhaseAsyncServer() noexcept;

    IndependentPhaseAsyncServer(IndependentPhaseAsyncServer const&) = delete;
    IndependentPhaseAsyncServer& operator=(IndependentPhaseAsyncServer const&) = delete;

    IndependentPhaseServerSubmission submit(uint64_t requestId, std::vector<int32_t> promptTokens,
        int32_t maxOutputTokens = 0, PhaseSchedulingHints scheduling = {});
    bool cancel(uint64_t requestId);
    //! Capture the currently prepared phase shapes for later execute() replay.
    bool capturePreparedGraphs();
    bool poll();
    void runUntilIdle(size_t maxPolls);

    std::optional<IndependentPhaseServerCompletion> tryPopCompletion();
    size_t inFlightCount() const noexcept;
    bool empty() const noexcept;

private:
    struct RequestState
    {
        std::vector<int32_t> promptTokens;
        std::vector<int32_t> generatedTokens;
        int32_t maxOutputTokens{};
        int32_t kvSlotId{-1};
        int32_t sharedPrefixSourceSlot{-1};
        PhaseSchedulingHints scheduling;
        std::chrono::steady_clock::time_point submittedAt;
    };

    IndependentPhaseCoordinatorCallbacks makeCallbacks();
    std::vector<IndependentPhaseRequestView> makeViews(std::vector<PhaseWorkItem> const& batch) const;
    bool isEos(int32_t tokenId) const noexcept;
    void processSamplingTickets();
    void processTicket(std::unique_ptr<IndependentPhaseSampleTicket> ticket);
    void finishRequest(uint64_t requestId);
    void destroyTicketEvent(IndependentPhaseSampleTicket& ticket) noexcept;

    IndependentPhaseServerConfig mConfig;
    IndependentPhaseCoordinator& mCoordinator;
    StableKVPageManager& mOwnership;
    IndependentPhaseRequestAdapter mAdapter;
    PhasePrefixReuseCache* mPrefixCache{};
    std::unordered_map<uint64_t, RequestState> mRequests;
    std::unordered_set<int32_t> mActiveSharedPrefixSources;
    std::deque<std::unique_ptr<IndependentPhaseSampleTicket>> mSamplingTickets;
    std::deque<IndependentPhaseServerCompletion> mCompletions;
};

} // namespace trt_edgellm::rt
