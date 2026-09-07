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

#include "runtime/phase/execution/phaseActionPlan.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <vector>

namespace trt_edgellm::rt
{

//! Phase work represented only by canonical ready rows. Tensor and ownership
//! compatibility remains the responsibility of the mechanism candidate builder.
struct PhaseFormationWork
{
    size_t encoderRows{};
    size_t prefillRows{};
    size_t decodeRows{};

    bool empty() const noexcept;
};

//! Work that will become ready from an event that is already outstanding at
//! the snapshot boundary. This never represents a predicted external arrival.
struct PhaseFormationKnownCompletion
{
    uint64_t eventId{};
    double readyAfterUs{};
    PhaseFormationWork work;
    double readyUncertaintyUs{};
    //! Conservative cost of serving this concrete cohort at the next action
    //! boundary. Zero leaves the completion visible without extending the
    //! evaluated service horizon.
    double serviceMakespanUs{};
    double serviceUncertaintyUs{};
    //! Service budget measured from event readiness. Infinite means that this
    //! completion carries no protected next-token deadline.
    double serviceBudgetUs{std::numeric_limits<double>::infinity()};
};

//! Immutable input to deterministic bounded-horizon replay.
struct PhaseFormationSnapshot
{
    uint64_t epoch{};
    PhaseFormationWork ready;
    std::vector<PhaseFormationKnownCompletion> knownCompletions;
    //! Oldest resident decode row's remaining TPOT budget. Infinite means
    //! that no decode row needs protection in this snapshot.
    double decodeServiceBudgetUs{std::numeric_limits<double>::infinity()};
};

//! One measured or predicted execution action in the replay catalog.
struct PhaseFormationAction
{
    size_t candidateIndex{};
    PhaseFormationWork consumes;
    double makespanUs{};
    double uncertaintyUs{};
    //! Only actions materialized by the current candidate builder may be the
    //! first action. Successor-only entries can model a concrete event result.
    bool initialEligible{true};
    //! Completion of the oldest protected resident decode row relative to the
    //! action start. Infinite means that this action has no decode-service
    //! estimate. It may include one mechanism-known follow-up action.
    double decodeServiceUs{std::numeric_limits<double>::infinity()};
    double decodeServiceUncertaintyUs{};
    //! Robust violation already implied by this concrete candidate's complete
    //! protected E/P/D path. This keeps the H=2 oracle inside the same SLO-safe
    //! action set as the production selector.
    double protectedViolationUs{};
};

struct PhaseFormationTransition
{
    bool feasible{};
    PhaseFormationSnapshot successor;
    PhaseFormationWork completed;
};

struct PhaseFormationSequence
{
    bool feasible{};
    size_t firstAction{};
    std::optional<size_t> successorAction;
    double makespanUs{std::numeric_limits<double>::infinity()};
    double uncertaintyUs{};
    PhaseFormationWork completed;
    double decodeServiceUs{std::numeric_limits<double>::infinity()};
    double decodeServiceUncertaintyUs{};
    double decodeServiceViolationUs{};
    double protectedViolationUs{};
};

struct PhaseFormationOracleResult
{
    std::optional<size_t> selectedAction;
    std::vector<PhaseFormationSequence> sequences;
};

//! Predicted equal-work regret against the oracle action from the same
//! immutable snapshot. This is counterfactual telemetry, not an observed CUDA
//! duration for the action that was not executed.
struct PhaseFormationRegret
{
    bool valid{};
    size_t selectedAction{};
    size_t oracleAction{};
    double selectedHorizonUs{};
    double oracleHorizonUs{};
    double predictedRegretUs{};
    double selectedDecodeViolationUs{};
    double oracleDecodeViolationUs{};
    double selectedProtectedViolationUs{};
    double oracleProtectedViolationUs{};
};

//! Request-local state used by the physical-completion transition replay.
//! It is independent of live queue objects and therefore cannot mutate
//! ownership or dispatch order.
enum class PhaseFormationRequestStage : uint8_t
{
    kEncoderReady,
    kPrefillReady,
    kDecodeReady,
    kComplete,
};

struct PhaseFormationRequestState
{
    uint64_t requestId{};
    PhaseFormationRequestStage stage{PhaseFormationRequestStage::kEncoderReady};
    size_t decodeStepsRemaining{};
    size_t visionBytes{};
    size_t kvBytes{};
};

//! One component of a measured or predicted physical completion vector.
struct PhaseFormationPhysicalCompletion
{
    PhaseGlobalActionKind phase{PhaseGlobalActionKind::kNone};
    std::vector<uint64_t> requestIds;
    double completionUs{};
    double uncertaintyUs{};
};

struct PhaseFormationReadyBoundary
{
    double completionUs{};
    double uncertaintyUs{};
    std::vector<uint64_t> encoderRequestIds;
    std::vector<uint64_t> prefillRequestIds;
    std::vector<uint64_t> decodeRequestIds;
    size_t releasedVisionBytes{};
    size_t releasedKvBytes{};
};

//! Exactly two request-ready boundaries produced by one physical completion
//! vector. Equal-time components form one boundary. With only one distinct
//! completion time, second repeats first rather than inventing future work.
struct PhaseFormationTwoBoundaryResult
{
    bool feasible{};
    PhaseFormationReadyBoundary first;
    PhaseFormationReadyBoundary second;
    std::vector<PhaseFormationRequestState> successorRequests;
};

//! Metadata fixed when formation-aware selection changes the concrete action
//! chosen by the otherwise identical myopic frontier.
struct PhaseFormationRealizedEpisodeStart
{
    uint64_t snapshotId{};
    PhaseGlobalActionKind selectedAction{PhaseGlobalActionKind::kNone};
    PhaseGlobalActionKind myopicAction{PhaseGlobalActionKind::kNone};
    PhaseGlobalActionKind oracleAction{PhaseGlobalActionKind::kNone};
    double predictedRegretUs{};
    double decodeServiceBudgetUs{std::numeric_limits<double>::infinity()};
    double timestampUs{};
};

//! One actual dispatch observed after a formation-aware action change. Times
//! are relative to the episode start and use host completion visibility.
struct PhaseFormationRealizedDispatch
{
    uint64_t planId{};
    PhaseGlobalActionKind action{PhaseGlobalActionKind::kNone};
    PhaseFormationWork work;
    double dispatchedAfterUs{};
    double completionVisibleAfterUs{std::numeric_limits<double>::infinity()};
};

//! Bounded realized transition. It does not claim an unexecuted action's
//! counterfactual runtime; it records only the selected path's actual cohort
//! formation and host-visible completion boundaries.
struct PhaseFormationRealizedEpisode
{
    uint64_t episodeId{};
    uint64_t snapshotId{};
    PhaseGlobalActionKind selectedAction{PhaseGlobalActionKind::kNone};
    PhaseGlobalActionKind myopicAction{PhaseGlobalActionKind::kNone};
    PhaseGlobalActionKind oracleAction{PhaseGlobalActionKind::kNone};
    double predictedRegretUs{};
    double decodeServiceBudgetUs{std::numeric_limits<double>::infinity()};
    double startedAtUs{};
    std::vector<PhaseFormationRealizedDispatch> dispatches;
    size_t encoderRows{};
    size_t prefillRows{};
    size_t decodeRows{};
    size_t firstDecodeRows{};
    size_t maxDecodeRows{};
    bool decodeServiced{};
    double decodeServiceGapUs{};
    double decodeCompletionVisibleUs{};
    double horizonCompletionVisibleUs{};
    double decodeServiceViolationUs{};
    bool truncated{};
};

struct PhaseFormationRealizedTelemetry
{
    size_t episodesStarted{};
    size_t episodesCompleted{};
    size_t episodesTruncated{};
    size_t decodeServices{};
    size_t decodeBudgets{};
    size_t decodeServiceViolations{};
    double decodeServiceGapUs{};
    double maxDecodeServiceGapUs{};
    double decodeServiceViolationUs{};
    double maxDecodeServiceViolationUs{};
    std::optional<PhaseFormationRealizedEpisode> lastEpisode;
};

struct PhaseFormationRealizedTrackerConfig
{
    //! Selected dispatch plus at most the next three global dispatches.
    size_t maxDispatches{4U};
    //! Bound overlapping attribution episodes under rapid selection changes.
    size_t maxActiveEpisodes{8U};
};

//! Attribute the selected path's actual near-future cohort formation without
//! changing scheduler policy or predicting requests that have not arrived.
class PhaseFormationRealizedTracker
{
public:
    explicit PhaseFormationRealizedTracker(PhaseFormationRealizedTrackerConfig config = {});

    void observeDispatch(uint64_t planId, PhaseGlobalActionKind action, PhaseFormationWork work, double timestampUs);
    void startEpisode(PhaseFormationRealizedEpisodeStart const& start, uint64_t planId, PhaseGlobalActionKind action,
        PhaseFormationWork work);
    void remapPlan(uint64_t oldPlanId, uint64_t newPlanId) noexcept;
    void observeCompletion(uint64_t planId, double timestampUs);
    void flush(double timestampUs);

    PhaseFormationRealizedTelemetry const& telemetry() const noexcept;
    std::vector<PhaseFormationRealizedEpisode> takeCompletedEpisodes();

private:
    void appendDispatch(PhaseFormationRealizedEpisode& episode, uint64_t planId, PhaseGlobalActionKind action,
        PhaseFormationWork work, double timestampUs);
    void finishReadyEpisodes(double timestampUs, bool force);
    void finishEpisode(PhaseFormationRealizedEpisode episode, double timestampUs, bool truncated);

    PhaseFormationRealizedTrackerConfig mConfig;
    uint64_t mNextEpisodeId{};
    std::deque<PhaseFormationRealizedEpisode> mActive;
    std::deque<PhaseFormationRealizedEpisode> mCompleted;
    PhaseFormationRealizedTelemetry mTelemetry;
};

PhaseFormationWork phaseFormationWork(PhaseGlobalActionCandidate const& candidate) noexcept;

//! Stable identity for replay/debugging. The fingerprint deliberately excludes
//! wall-clock time and includes only observable ready work and concrete events.
uint64_t phaseFormationSnapshotId(PhaseFormationSnapshot const& snapshot) noexcept;

//! Apply one action without mutating a live queue. Concrete completion events
//! whose horizon expires during the action are moved into successor.ready.
PhaseFormationTransition phaseFormationTransition(
    PhaseFormationSnapshot const& snapshot, PhaseFormationAction const& action) noexcept;

//! Compare first actions over exactly the same target work using at most one
//! successor action. Sequences that cannot cover the target are ineligible.
PhaseFormationOracleResult phaseFormationEvaluateH2(PhaseFormationSnapshot const& snapshot,
    std::vector<PhaseFormationAction> const& actions, PhaseFormationWork target) noexcept;

//! Populate the existing global selector's equal-work horizon fields. Returns
//! the offline H=2 choice for shadow regret telemetry.
PhaseFormationOracleResult phaseFormationApplyH2(std::vector<PhaseGlobalActionCandidate>& candidates,
    PhaseFormationSnapshot const& snapshot, PhaseFormationWork target, double targetReferenceWorkUs) noexcept;

//! Compare one selected action with the H=2 oracle over the same work frontier.
PhaseFormationRegret phaseFormationPredictedRegret(
    PhaseFormationOracleResult const& result, size_t selectedAction) noexcept;
//! Preserve myopic ordering on equal robust horizons; H=2 may replace it only
//! when the bounded observable frontier predicts a strict improvement.
bool phaseFormationShouldReplaceMyopic(
    PhaseFormationOracleResult const& result, size_t myopicAction, size_t selectedAction) noexcept;

//! Replay E->P, P->D, and D->D/complete transitions at the first two physical
//! completion boundaries. No future arrival, learned formation rule, or live
//! queue mutation is permitted.
PhaseFormationTwoBoundaryResult phaseFormationEvaluateCompletionBoundaries(
    std::vector<PhaseFormationRequestState> requests,
    std::vector<PhaseFormationPhysicalCompletion> completions) noexcept;

} // namespace trt_edgellm::rt
