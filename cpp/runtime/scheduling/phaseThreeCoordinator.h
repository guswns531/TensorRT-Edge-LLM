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

#include "runtime/scheduling/independentPhaseAsyncServer.h"
#include "runtime/scheduling/phaseVisionAdapter.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace trt_edgellm::rt
{

enum class PhaseThreeSubmissionStatus
{
    kEncoding,
    kQueued,
    kDuplicateRequest,
};

struct PhaseThreeCoordinatorConfig
{
    //! Bound request-owned GPU vision payloads waiting in or running through the LLM phases.
    size_t maxEncodedInFlight{2U};
    //! Optional byte budget for downstream request-owned vision payloads. Zero disables the byte gate.
    size_t maxEncodedBytes{};
    //! Default end-to-end image TTFT SLO, including encoder queue and execution. Zero inherits the LLM default.
    double visionTtftTargetUs{2500000.0};
};

struct PhaseThreeCoordinatorMetrics
{
    size_t pendingVisionRequests{};
    size_t downstreamEncodedRequests{};
    size_t downstreamEncodedBytes{};
    size_t encoderStarts{};
    size_t encoderCompletions{};
    double oldestPendingAgeUs{};
    double lastEncoderQueueWaitUs{};
    double maxEncoderQueueWaitUs{};
    float lastEncoderGpuMs{};
    float maxEncoderGpuMs{};
};

//! Normalize image scheduling at HTTP arrival so encoder time remains part of TTFT age.
PhaseSchedulingHints phaseVisionSchedulingHints(PhaseSchedulingHints scheduling, double defaultTtftTargetUs,
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

//! Count and byte admission gate for starting another encoder request.
bool phaseVisionEncoderCapacityAvailable(size_t downstreamRequests, size_t maxDownstreamRequests,
    size_t downstreamBytes, size_t maxDownstreamBytes, size_t estimatedPayloadBytes) noexcept;

//! Encoder -> prefill -> decode coordinator over three independent contexts.
class PhaseThreeCoordinator
{
public:
    PhaseThreeCoordinator(
        PhaseVisionAdapter& vision, IndependentPhaseAsyncServer& server, PhaseThreeCoordinatorConfig config = {});

    PhaseThreeSubmissionStatus submit(uint64_t requestId, LLMGenerationRequest request, int32_t maxOutputTokens,
        PhaseSchedulingHints scheduling = {});
    bool cancel(uint64_t requestId);
    bool poll();
    bool empty() const noexcept;
    PhaseThreeCoordinatorMetrics metrics() const noexcept;

    std::optional<IndependentPhaseServerToken> tryPopToken();
    std::optional<IndependentPhaseServerCompletion> tryPopCompletion();

private:
    struct PendingVisionRequest
    {
        uint64_t requestId{};
        LLMGenerationRequest request;
        int32_t maxOutputTokens{};
        PhaseSchedulingHints scheduling;
    };

    bool startNextEncoder();
    bool completeEncoder();
    bool encoderCapacityAvailable() const noexcept;

    PhaseVisionAdapter& mVision;
    IndependentPhaseAsyncServer& mServer;
    PhaseThreeCoordinatorConfig mConfig;
    std::deque<PendingVisionRequest> mPending;
    std::optional<PendingVisionRequest> mEncoding;
    std::unordered_set<uint64_t> mRequestIds;
    std::unordered_map<uint64_t, size_t> mDownstreamRequestBytes;
    std::unordered_set<uint64_t> mCancelRequested;
    size_t mDownstreamEncodedBytes{};
    size_t mEstimatedEncodedBytes{};
    size_t mEncoderStarts{};
    size_t mEncoderCompletions{};
    double mLastEncoderQueueWaitUs{};
    double mMaxEncoderQueueWaitUs{};
    float mLastEncoderGpuMs{};
    float mMaxEncoderGpuMs{};
};

} // namespace trt_edgellm::rt
