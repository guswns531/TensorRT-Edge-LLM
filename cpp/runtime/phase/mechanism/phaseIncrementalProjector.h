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
    //! A request can own persistent state while waiting for sampling or an
    //! external producer. Only ready requests participate in the successor
    //! cohort builder; in-flight membership is represented separately.
    bool ready{true};
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

//! Fidelity of the physical outcome injected into an immutable replay. The
//! transition mechanism never changes with this value; it is telemetry for
//! predictor ablation and authority gates.
enum class PhaseOutcomeFidelity : uint8_t
{
    kScalarEnvelope,
    kEffectVector,
    kCompletionVector,
    kMeasuredReplay,
};

//! One isolated component reference at the frozen dispatch boundary.
struct PhaseOutcomeComponentReference
{
    PhaseUnifiedPhase phase{PhaseUnifiedPhase::kNone};
    uint64_t executionId{};
    double isolatedReferenceUs{};
    bool incumbent{};
};

//! A predictor may return more than one physically plausible completion
//! order. Scalar prediction intentionally returns both orders; richer models
//! may collapse the envelope only when their confidence excludes reversal.
struct PhaseOutcomeEnvelope
{
    uint64_t actionId{};
    PhaseOutcomeFidelity fidelity{PhaseOutcomeFidelity::kScalarEnvelope};
    std::vector<PhaseCompletionVector> alternatives;
    bool ready{};
};

//! Immutable, in-process decision state. The live coordinator and allocators
//! are never referenced after construction, which makes branch comparison
//! exact and repeatable within one process.
struct PhaseFrozenDecisionSnapshot
{
    uint64_t snapshotId{};
    uint64_t scalarPolicyStateSignature{};
    PhaseIncrementalProjectionSnapshot projection;
    std::vector<PhaseIncrementalAction> frontier;
    //! Work that each frontier action adds to the pre-dispatch projection.
    //! Empty launch work preserves the legacy already-dispatched replay
    //! contract used by measured completion tests.
    std::unordered_map<uint64_t, std::vector<PhaseInFlightWorkSnapshot>> launchedWork;
};

enum class PhaseFrozenReplayReason : uint8_t
{
    kReplayed,
    kInvalidSnapshot,
    kMissingAction,
    kActionNotLegal,
    kEnvelopeNotReady,
    kEnvelopeActionMismatch,
    kInvalidOutcome,
};

struct PhaseFrozenReplayTrajectory
{
    bool valid{};
    std::vector<PhaseIncrementalProjection> boundaries;
    double robustHorizonUs{};
    size_t encoderReadyRows{};
    size_t prefillReadyRows{};
    size_t decodeReadyRows{};
    PhaseProjectedOwnership ownership;
};

struct PhaseFrozenReplayResult
{
    bool valid{};
    PhaseFrozenReplayReason reason{PhaseFrozenReplayReason::kInvalidSnapshot};
    uint64_t snapshotId{};
    uint64_t actionId{};
    PhaseOutcomeFidelity fidelity{PhaseOutcomeFidelity::kScalarEnvelope};
    std::vector<PhaseFrozenReplayTrajectory> trajectories;
    double worstCaseRobustHorizonUs{};
};

//! Decision-relevant effect estimate used to materialize an outcome envelope.
//! Margins are normalized by the sum of isolated component references.
struct PhaseEffectOutcomeEstimate
{
    double compressionMean{};
    double compressionUncertainty{};
    double incumbentStretchMean{};
    double incumbentStretchUncertainty{};
    double orderMarginMean{};
    double orderMarginUncertainty{};
    bool ready{};
};

char const* phaseFrozenReplayReasonName(PhaseFrozenReplayReason reason) noexcept;

//! Validate and freeze a policy snapshot. The returned ID covers request/DAG
//! state, canonical row order, ownership, in-flight work, candidate frontier,
//! and the scalar model state supplied by the caller.
std::optional<PhaseFrozenDecisionSnapshot> phaseFreezeDecisionSnapshot(PhaseIncrementalProjectionSnapshot projection,
    std::vector<PhaseIncrementalAction> frontier, uint64_t scalarPolicyStateSignature) noexcept;

//! Freeze a true pre-dispatch decision frontier. Every launch entry must name
//! one frontier action and contain exactly the new contexts introduced by
//! that action. The same immutable snapshot can therefore replay mutually
//! exclusive branches without cloning live queues or allocators.
std::optional<PhaseFrozenDecisionSnapshot> phaseFreezeDecisionSnapshot(PhaseIncrementalProjectionSnapshot projection,
    std::vector<PhaseIncrementalAction> frontier,
    std::unordered_map<uint64_t, std::vector<PhaseInFlightWorkSnapshot>> launchedWork,
    uint64_t scalarPolicyStateSignature) noexcept;

//! Build a conservative two-order envelope from a scalar makespan. This never
//! invents an external arrival or claims to know which component finishes first.
PhaseOutcomeEnvelope phaseScalarOutcomeEnvelope(uint64_t actionId,
    std::vector<PhaseOutcomeComponentReference> const& components, double actionMakespanUs,
    double uncertaintyUs) noexcept;

//! Build the smallest completion envelope consistent with three normalized
//! effects. If the completion-order confidence interval crosses zero, both
//! physical orders are retained.
PhaseOutcomeEnvelope phaseEffectOutcomeEnvelope(uint64_t actionId,
    std::vector<PhaseOutcomeComponentReference> const& components, PhaseEffectOutcomeEstimate const& estimate) noexcept;

//! Wrap a directly predicted or measured physical completion vector.
PhaseOutcomeEnvelope phaseCompletionOutcomeEnvelope(
    PhaseCompletionVector completion, PhaseOutcomeFidelity fidelity, bool ready = true) noexcept;

//! Replay every envelope alternative for at most two request-ready boundaries.
//! The worst-case robust horizon is suitable for the SLO guard; individual
//! trajectories remain available for dominance and diagnostic analysis.
PhaseFrozenReplayResult phaseReplayFrozenOutcome(PhaseFrozenDecisionSnapshot const& snapshot, uint64_t actionId,
    PhaseOutcomeEnvelope const& envelope, double uncertaintyScale = 1.0) noexcept;

} // namespace trt_edgellm::rt
