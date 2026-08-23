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
};

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

    PhaseVisionAdapter& mVision;
    IndependentPhaseAsyncServer& mServer;
    PhaseThreeCoordinatorConfig mConfig;
    std::deque<PendingVisionRequest> mPending;
    std::optional<PendingVisionRequest> mEncoding;
    std::unordered_set<uint64_t> mRequestIds;
    std::unordered_set<uint64_t> mDownstreamRequestIds;
    std::unordered_set<uint64_t> mCancelRequested;
};

} // namespace trt_edgellm::rt
