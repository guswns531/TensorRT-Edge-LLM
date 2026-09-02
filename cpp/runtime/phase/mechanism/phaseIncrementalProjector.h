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

#include "runtime/phase/mechanism/phaseIncrementalAction.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

namespace trt_edgellm::rt
{

//! Request-local DAG state used by the deterministic H1 replay projector.
enum class PhaseProjectedRequestStage : uint8_t
{
    kEncoder,
    kPrefill,
    kDecode,
    kComplete,
};

//! Stable request state. Ownership flags describe persistent allocations at
//! the projection boundary, not temporary TensorRT workspace.
struct PhaseProjectedRequest
{
    uint64_t requestId{};
    PhaseProjectedRequestStage stage{PhaseProjectedRequestStage::kPrefill};
    int32_t stableKvSlotId{-1};
    int32_t decodeStepsRemaining{};
    int32_t generatedTokens{};
    size_t visionLeaseBytes{};
    size_t kvLeaseBytes{};
    bool visionOwned{};
    bool kvOwned{};
    bool firstTokenObserved{};
};

struct PhaseProjectedOwnership
{
    size_t visionBytes{};
    size_t kvBytes{};
    size_t reclaimedVisionBytes{};
    size_t reclaimedKvBytes{};
};

//! Post-dispatch state to advance to exactly one completion boundary.
struct PhaseIncrementalProjectionSnapshot
{
    uint64_t epoch{};
    PhaseInFlightSnapshot inFlight;
    std::vector<PhaseProjectedRequest> requests;
    PhaseProjectedOwnership ownership;
};

//! One predicted or replayed context completion relative to the current
//! decision boundary. There is exactly one component per outstanding context.
struct PhaseCompletionComponent
{
    PhaseUnifiedPhase phase{PhaseUnifiedPhase::kNone};
    uint64_t executionId{};
    double completionUs{};
    double uncertaintyUs{};
    bool incumbent{};
};

struct PhaseCompletionVector
{
    uint64_t actionId{};
    std::vector<PhaseCompletionComponent> components;
};

enum class PhaseProjectionReason : uint8_t
{
    kProjected,
    kIllegalAction,
    kActionMismatch,
    kInvalidOutstandingSet,
    kInvalidCompletionVector,
    kMissingExecution,
    kMissingRequest,
    kRequestStageMismatch,
};

//! Result after the earliest component only. A remaining context stays in
//! flight and its residual completion remains in completionVector.
struct PhaseIncrementalProjection
{
    bool valid{};
    PhaseProjectionReason reason{PhaseProjectionReason::kInvalidCompletionVector};
    double boundaryUs{};
    double robustBoundaryUs{};
    PhaseCompletionComponent completed;
    PhaseIncrementalProjectionSnapshot successor;
    PhaseCompletionVector completionVector;
    std::vector<PhaseIncrementalReadyAction> ready;
    std::vector<PhaseIncrementalAction> legalActions;
};

char const* phaseProjectionReasonName(PhaseProjectionReason reason) noexcept;

//! Recompute persistent ownership from request-local stable leases.
PhaseProjectedOwnership phaseProjectedOwnership(std::vector<PhaseProjectedRequest> const& requests) noexcept;

//! Build one canonical phase-local cohort per ready phase. Requests that are
//! still represented by in-flight work are excluded.
std::vector<PhaseIncrementalReadyAction> phaseProjectedReadyActions(
    PhaseIncrementalProjectionSnapshot const& snapshot) noexcept;

//! Advance to the earliest robust component completion and rebuild the next
//! bounded 0/1/2-context action frontier. uncertaintyScale is used only for
//! ordering and SLO projection; simulated clock progress uses the component's
//! mean completion.
PhaseIncrementalProjection phaseProjectEarliestCompletion(PhaseIncrementalProjectionSnapshot const& snapshot,
    PhaseIncrementalAction const& action, PhaseCompletionVector const& completionVector,
    double uncertaintyScale = 1.0) noexcept;

//! Measured/replay predictor. It deliberately has no generalization: M4 uses
//! it to validate projector semantics before a learned predictor is promoted.
class PhaseReplayCompletionPredictor
{
public:
    bool insert(PhaseCompletionVector completionVector);
    std::optional<PhaseCompletionVector> predict(PhaseIncrementalAction const& action) const;
    size_t size() const noexcept;

private:
    std::unordered_map<uint64_t, PhaseCompletionVector> mVectors;
};

} // namespace trt_edgellm::rt
