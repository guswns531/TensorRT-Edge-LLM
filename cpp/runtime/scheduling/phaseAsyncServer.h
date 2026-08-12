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

#include "runtime/llmRuntimeUtils.h"
#include "runtime/scheduling/phaseThreeCoordinator.h"
#include "runtime/scheduling/phaseVisionAdapter.h"
#include "runtime/streaming.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

namespace trt_edgellm
{
namespace rt
{

struct PhaseAsyncServerConfig
{
    //! Total requests owned by the server, including GPU work and admission queues.
    size_t maxInFlightRequests{};
    //! Whole-request KV admission policy shared by text and multimodal requests.
    PhasePageReservationConfig pageReservation;
};

struct PhaseAsyncSubmission
{
    uint64_t requestId{};
    PhaseRequestStatus status{PhaseRequestStatus::kPending};
};

struct PhaseAsyncCompletion
{
    uint64_t requestId{};
    LLMGenerationResponse response;
    double latencyMs{};
};

//! Event-loop API that owns request state across asynchronous encoder/prefill/decode execution.
//!
//! All methods must be called from one host thread with the CUDA context used to construct the
//! phase workers current on that thread. submit() accepts one logical request; continuous batching
//! happens independently inside the encoder, prefill, and decode queues. V1 is greedy-only and
//! rejects speculative decoding, LoRA, audio, logprobs, streaming channels, and stop strings.
class PhaseAsyncServer
{
public:
    //! Construct a text-only server over independent prefill/decode queues.
    PhaseAsyncServer(PhaseAsyncServerConfig config, PhaseContextServingFacade& servingFacade,
        tokenizer::Tokenizer const& tokenizer, cudaStream_t requestStream);
    //! Construct an encoder + prefill + decode server for multimodal requests.
    PhaseAsyncServer(PhaseAsyncServerConfig config, PhaseOnlineCoordinator& coordinator,
        PhaseEncoderDispatchWorker& encoderWorker, PhaseContextServingFacade& servingFacade,
        tokenizer::Tokenizer const& tokenizer, cudaStream_t requestStream, PhaseVisionAdapter* visionAdapter = nullptr);
    ~PhaseAsyncServer() noexcept;

    PhaseAsyncServer(PhaseAsyncServer const&) = delete;
    PhaseAsyncServer& operator=(PhaseAsyncServer const&) = delete;
    PhaseAsyncServer(PhaseAsyncServer&&) = delete;
    PhaseAsyncServer& operator=(PhaseAsyncServer&&) = delete;

    //! Transfer request ownership to the server and return immediately after admission.
    PhaseAsyncSubmission submit(LLMGenerationRequest request, PhaseSchedulingHints scheduling = {});
    //! Request cancellation. In-flight GPU work completes before the stable slot is released.
    bool cancel(uint64_t requestId);
    //! Advance CUDA-event completions and launch all currently runnable phases without synchronizing.
    bool poll();
    std::optional<PhaseAsyncCompletion> tryPopCompletion();

    std::optional<PhaseRequestStatus> status(uint64_t requestId) const;
    size_t inFlightCount() const noexcept;
    size_t completionCount() const noexcept;
    bool empty() const noexcept;

private:
    struct RequestState
    {
        LLMGenerationRequest request;
        DecodingInferenceContext context;
        PhaseSchedulingHints scheduling;
        std::chrono::steady_clock::time_point submittedAt;
        bool usesVision{};
        bool waitingForVisionSlot{};
        bool cancelRequested{};
    };

    void validateRequest(LLMGenerationRequest const& request) const;
    void prepareRequest(RequestState& state);
    PhaseRequestStatus admit(uint64_t requestId, RequestState& state);
    bool admitWaitingVisionRequests();
    bool processCancellations();
    void processTerminals();
    void complete(uint64_t requestId, PhaseRequestStatus status);

    PhaseAsyncServerConfig mConfig;
    PhaseOnlineCoordinator* mCoordinator{};
    PhaseEncoderDispatchWorker* mEncoderWorker{};
    PhaseContextServingFacade& mServingFacade;
    tokenizer::Tokenizer const& mTokenizer;
    cudaStream_t mRequestStream{};
    PhaseVisionAdapter* mVisionAdapter{};
    uint64_t mNextRequestId{1};
    size_t mTerminalObserverId{};
    std::unordered_map<uint64_t, std::unique_ptr<RequestState>> mRequests;
    std::deque<uint64_t> mWaitingVisionAdmissions;
    std::deque<PhaseRequestSnapshot> mTerminalSnapshots;
    std::deque<PhaseAsyncCompletion> mCompletions;
};

} // namespace rt
} // namespace trt_edgellm
