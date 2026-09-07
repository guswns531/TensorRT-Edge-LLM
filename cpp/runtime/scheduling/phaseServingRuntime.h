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

#include "runtime/phase/policy/phasePolicyMode.h"
#include "runtime/scheduling/independentPhaseAsyncServer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace trt_edgellm::rt
{

class EngineExecutor;
struct EmbeddingData;
struct LLMEngineConfig;
struct SharedResources;

struct PhaseServingRuntimeConfig
{
    PhasePolicyMode policyMode{PhasePolicyMode::kContextualScalarTransition};
    int32_t maxStableSlots{};
    size_t maxInFlightRequests{};
    size_t maxPendingRequests{};
    int32_t maxPrefillChunkTokens{128};
    int32_t maxPrefillBatchTokens{};
    bool enableCudaGraphs{true};
    bool enablePersistentDecodeSelect{true};
    bool enablePersistentPageBindings{true};
    bool sharedExecutionContext{};
};

//! Single-rank asynchronous text serving over independent prefill/decode contexts.
//!
//! The runtime consumes one base executor and shares immutable weights and the
//! physical paged-KV backing owned by SharedResources. Logical request leases,
//! active page-table rows, phase I/O, streams, TensorRT contexts, and workspaces
//! are owned independently from the legacy request loop.
class PhaseServingRuntime
{
public:
    static std::unique_ptr<PhaseServingRuntime> create(PhaseServingRuntimeConfig config,
        LLMEngineConfig const& engineConfig, std::unique_ptr<EngineExecutor> executor, SharedResources& resources,
        EmbeddingData const& embedding, cudaStream_t setupStream);

    ~PhaseServingRuntime() noexcept;

    PhaseServingRuntime(PhaseServingRuntime const&) = delete;
    PhaseServingRuntime& operator=(PhaseServingRuntime const&) = delete;

    IndependentPhaseServerSubmission submit(uint64_t requestId, std::vector<int32_t> promptTokens,
        int32_t maxOutputTokens = 0, PhaseSchedulingHints scheduling = {});
    IndependentPhaseServerSubmission submitOrQueue(uint64_t requestId, std::vector<int32_t> promptTokens,
        int32_t maxOutputTokens = 0, PhaseSchedulingHints scheduling = {});
    bool cancel(uint64_t requestId);
    bool poll();
    void runUntilIdle(size_t maxPolls);
    std::optional<IndependentPhaseServerToken> tryPopToken();
    std::optional<IndependentPhaseServerCompletion> tryPopCompletion();
    bool empty() const noexcept;
    size_t inFlightCount() const noexcept;
    size_t pendingCount() const noexcept;
    CUcontext cudaContext() const noexcept;
    PhaseSchedulerTelemetry const& schedulerTelemetry() const noexcept;

private:
    class Impl;

    explicit PhaseServingRuntime(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> mImpl;
};

} // namespace trt_edgellm::rt
