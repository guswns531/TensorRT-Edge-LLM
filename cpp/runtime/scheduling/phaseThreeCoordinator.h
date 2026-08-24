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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    //! Optional larger downstream capacity enabled only by the vision-age/decode-TPOT guard.
    size_t throughputMaxEncodedInFlight{};
    //! Optional byte budget for downstream request-owned vision payloads. Zero disables the byte gate.
    size_t maxEncodedBytes{};
    //! Maximum logical requests coalesced into one vision encoder execution. One preserves legacy behavior.
    size_t maxEncoderBatchSize{1U};
    //! Optional media-item cap for a coalesced encoder batch. Zero disables this guard.
    size_t maxEncoderMediaItems{};
    //! Maximum time to wait for encoder batch formation. Zero dispatches immediately.
    double encoderBatchWaitUs{};
    //! Maximum encoded requests released together into the independent prefill scheduler. Zero inherits encoder BS.
    size_t maxPrefillBatchSize{};
    //! Optional prompt-token budget for one release into the prefill scheduler. Zero disables the token gate.
    size_t maxPrefillBatchTokens{};
    //! Maximum time an encoded request waits for prefill batch formation. Zero dispatches immediately.
    double prefillBatchWaitUs{};
    //! Default end-to-end image TTFT SLO, including encoder queue and execution. Zero inherits the LLM default.
    double visionTtftTargetUs{2500000.0};
    //! Escalate lookahead after this fraction of the oldest vision request's TTFT target. Zero disables age escalation.
    double lookaheadEscalationRatio{0.4};
    //! Do not escalate lookahead at or above this observed decode TPOT pressure. Zero disables the guard.
    float lookaheadDecodeTpotPressureLimit{0.8F};
};

struct PhaseThreeCoordinatorMetrics
{
    size_t pendingVisionRequests{};
    size_t pendingPrefillReadyRequests{};
    size_t pendingPrefillReadyBytes{};
    size_t downstreamEncodedRequests{};
    size_t downstreamEncodedBytes{};
    size_t prefillStorageReleases{};
    size_t prefillStorageReleasedBytes{};
    size_t encoderStarts{};
    size_t encoderCompletions{};
    size_t encoderBatches{};
    size_t lastEncoderBatchSize{};
    size_t maxEncoderBatchSize{};
    double oldestPendingAgeUs{};
    double lastEncoderQueueWaitUs{};
    double maxEncoderQueueWaitUs{};
    float lastEncoderGpuMs{};
    float maxEncoderGpuMs{};
    size_t prefillAdmissionBatches{};
    size_t lastPrefillAdmissionBatchSize{};
    size_t maxPrefillAdmissionBatchSize{};
    double lastPrefillReadyQueueWaitUs{};
    double maxPrefillReadyQueueWaitUs{};
    size_t effectiveEncodedCapacity{};
    size_t maxEffectiveEncodedCapacity{};
    size_t lookaheadEscalations{};
    float decodeTpotPressure{};
};

//! Select latency or throughput vision lookahead from queue age and observed decode pressure.
size_t phaseVisionEffectiveEncodedCapacity(size_t latencyCapacity, size_t throughputCapacity, bool throughputMode,
    double oldestVisionAgeUs, double visionTtftTargetUs, double escalationRatio, float decodeTpotPressure,
    float decodeTpotPressureLimit) noexcept;

//! Normalize image scheduling at HTTP arrival so encoder time remains part of TTFT age.
PhaseSchedulingHints phaseVisionSchedulingHints(PhaseSchedulingHints scheduling, double defaultTtftTargetUs,
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now());

//! Count and byte admission gate for starting another encoder request.
bool phaseVisionEncoderCapacityAvailable(size_t downstreamRequests, size_t maxDownstreamRequests,
    size_t downstreamBytes, size_t maxDownstreamBytes, size_t estimatedPayloadBytes,
    size_t additionalRequests = 1U) noexcept;

//! Select the FIFO prefix released from the encoded-ready queue into the prefill scheduler.
size_t phaseVisionReadyPrefillBatchSize(std::vector<int32_t> const& promptTokenCounts, size_t maxBatchSize,
    size_t maxBatchTokens, double oldestWaitUs, double batchWaitUs) noexcept;

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

    struct ReadyPrefillRequest
    {
        uint64_t requestId{};
        std::vector<int32_t> promptTokens;
        std::shared_ptr<PhaseVisionPayload> payload;
        int32_t maxOutputTokens{};
        PhaseSchedulingHints scheduling;
        size_t payloadBytes{};
        std::chrono::steady_clock::time_point encodedAt;
    };

    bool startNextEncoder();
    bool completeEncoder();
    bool dispatchReadyPrefill();
    size_t nextEncoderBatchSize() const noexcept;
    size_t nextReadyPrefillBatchSize() const noexcept;
    bool encoderCapacityAvailable(size_t additionalRequests = 1U) const noexcept;
    size_t effectiveEncodedCapacity() const noexcept;
    static size_t mediaItemCount(PendingVisionRequest const& pending) noexcept;

    PhaseVisionAdapter& mVision;
    IndependentPhaseAsyncServer& mServer;
    PhaseThreeCoordinatorConfig mConfig;
    std::deque<PendingVisionRequest> mPending;
    std::vector<PendingVisionRequest> mEncoding;
    std::deque<ReadyPrefillRequest> mReadyPrefill;
    std::unordered_set<uint64_t> mRequestIds;
    std::unordered_map<uint64_t, size_t> mDownstreamRequestBytes;
    std::unordered_set<uint64_t> mCancelRequested;
    size_t mEstimatedEncodedBytes{};
    size_t mEncoderStarts{};
    size_t mEncoderCompletions{};
    size_t mEncoderBatches{};
    size_t mLastEncoderBatchSize{};
    size_t mMaxEncoderBatchSize{};
    double mLastEncoderQueueWaitUs{};
    double mMaxEncoderQueueWaitUs{};
    float mLastEncoderGpuMs{};
    float mMaxEncoderGpuMs{};
    size_t mReadyPrefillBytes{};
    size_t mPrefillAdmissionBatches{};
    size_t mLastPrefillAdmissionBatchSize{};
    size_t mMaxPrefillAdmissionBatchSize{};
    double mLastPrefillReadyQueueWaitUs{};
    double mMaxPrefillReadyQueueWaitUs{};
    size_t mMaxEffectiveEncodedCapacity{};
    size_t mLookaheadEscalations{};
    size_t mLastEffectiveEncodedCapacity{};
};

} // namespace trt_edgellm::rt
