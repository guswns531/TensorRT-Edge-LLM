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

#include "runtime/phase/mechanism/phaseIncrementalProjector.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace trt_edgellm::rt
{
namespace
{

PhaseInFlightWorkSnapshot inFlight(PhaseUnifiedPhase phase, uint64_t executionId, std::vector<uint64_t> requestIds)
{
    PhaseInFlightWorkSnapshot result;
    result.phase = phase;
    result.status = PhaseInFlightStatus::kRunning;
    result.executionId = executionId;
    result.requestIds = std::move(requestIds);
    return result;
}

PhaseIncrementalAction pairAction(PhaseGlobalActionKind kind, PhaseUnifiedActionDirection direction)
{
    return phaseIncrementalActionForDispatch(
        PhaseExecutionSet::kNone, 77U, kind, direction, PhaseStartSkewBucket::kImmediate);
}

bool hasNewcomer(std::vector<PhaseIncrementalAction> const& actions, PhaseUnifiedPhase newcomer)
{
    return std::any_of(actions.begin(), actions.end(), [newcomer](PhaseIncrementalAction const& action) {
        return !action.key.noDispatch && action.key.newcomerPhase == newcomer;
    });
}

PhaseIncrementalProjectionSnapshot encoderPrefillProjection()
{
    PhaseIncrementalProjectionSnapshot snapshot;
    snapshot.epoch = 9U;
    snapshot.inFlight.hostSnapshotNs = 1000U;
    snapshot.inFlight.outstanding = PhaseExecutionSet::kEncoder | PhaseExecutionSet::kPrefill;
    snapshot.inFlight.work
        = {inFlight(PhaseUnifiedPhase::kEncoder, 71U, {701U}), inFlight(PhaseUnifiedPhase::kPrefill, 72U, {702U})};
    snapshot.requests = {
        {701U, PhaseProjectedRequestStage::kEncoder, 1, 2, 0, 100U, 1000U},
        {702U, PhaseProjectedRequestStage::kPrefill, 2, 2, 0, 200U, 2000U, true},
    };
    snapshot.ownership = phaseProjectedOwnership(snapshot.requests);
    return snapshot;
}

TEST(PhaseIncrementalProjectorTest, ReplayPredictorReturnsOnlyExactLegalActionIdentity)
{
    PhaseIncrementalAction const action
        = pairAction(PhaseGlobalActionKind::kPrefillDecode, PhaseUnifiedActionDirection::kPrefillToDecode);
    PhaseReplayCompletionPredictor predictor;
    EXPECT_TRUE(predictor.insert({action.actionId,
        {{PhaseUnifiedPhase::kPrefill, 1U, 8.0, 0.5, true}, {PhaseUnifiedPhase::kDecode, 2U, 3.0, 0.2, false}}}));
    ASSERT_TRUE(predictor.predict(action).has_value());
    EXPECT_EQ(predictor.predict(action)->components.size(), 2U);
    EXPECT_EQ(predictor.size(), 1U);

    PhaseIncrementalAction other = action;
    ++other.actionId;
    EXPECT_FALSE(predictor.predict(other).has_value());
    EXPECT_FALSE(predictor.insert({0U, {}}));
}

TEST(PhaseIncrementalProjectorTest, DecodeCompletesBeforeEncoderWithoutCompletingWholePair)
{
    PhaseIncrementalAction const action
        = pairAction(PhaseGlobalActionKind::kEncoderDecode, PhaseUnifiedActionDirection::kEncoderToDecode);
    PhaseIncrementalProjectionSnapshot snapshot;
    snapshot.epoch = 5U;
    snapshot.inFlight.hostSnapshotNs = 1000000U;
    snapshot.inFlight.outstanding = PhaseExecutionSet::kEncoder | PhaseExecutionSet::kDecode;
    snapshot.inFlight.work
        = {inFlight(PhaseUnifiedPhase::kEncoder, 11U, {101U}), inFlight(PhaseUnifiedPhase::kDecode, 12U, {102U})};
    snapshot.requests = {
        {101U, PhaseProjectedRequestStage::kEncoder, -1, 2, 0, 4096U, 512U},
        {102U, PhaseProjectedRequestStage::kDecode, 3, 0, 4, 0U, 1024U, false, true, true},
        {103U, PhaseProjectedRequestStage::kPrefill, 4, 2, 0, 0U, 512U},
    };
    snapshot.ownership = phaseProjectedOwnership(snapshot.requests);
    PhaseCompletionVector const completion{action.actionId,
        {{PhaseUnifiedPhase::kEncoder, 11U, 10.0, 0.5, true}, {PhaseUnifiedPhase::kDecode, 12U, 2.0, 0.1, false}}};

    PhaseIncrementalProjection const result = phaseProjectEarliestCompletion(snapshot, action, completion);

    ASSERT_TRUE(result.valid) << phaseProjectionReasonName(result.reason);
    EXPECT_EQ(result.completed.phase, PhaseUnifiedPhase::kDecode);
    EXPECT_DOUBLE_EQ(result.boundaryUs, 2.0);
    EXPECT_DOUBLE_EQ(result.robustBoundaryUs, 2.1);
    EXPECT_EQ(result.successor.inFlight.outstanding, PhaseExecutionSet::kEncoder);
    ASSERT_EQ(result.successor.inFlight.work.size(), 1U);
    EXPECT_EQ(result.successor.inFlight.work.front().executionId, 11U);
    ASSERT_EQ(result.completionVector.components.size(), 1U);
    EXPECT_DOUBLE_EQ(result.completionVector.components.front().completionUs, 8.0);
    EXPECT_EQ(result.successor.requests[1].stage, PhaseProjectedRequestStage::kComplete);
    EXPECT_EQ(result.successor.ownership.kvBytes, 0U);
    EXPECT_EQ(result.successor.ownership.reclaimedKvBytes, 1024U);
    EXPECT_TRUE(hasNewcomer(result.legalActions, PhaseUnifiedPhase::kPrefill));
}

TEST(PhaseIncrementalProjectorTest, EncoderCompletionMakesPrefillReadyWhileDecodeRemainsOutstanding)
{
    PhaseIncrementalAction const action
        = pairAction(PhaseGlobalActionKind::kEncoderDecode, PhaseUnifiedActionDirection::kDecodeToEncoder);
    PhaseIncrementalProjectionSnapshot snapshot;
    snapshot.inFlight.outstanding = PhaseExecutionSet::kEncoder | PhaseExecutionSet::kDecode;
    snapshot.inFlight.work
        = {inFlight(PhaseUnifiedPhase::kDecode, 21U, {201U}), inFlight(PhaseUnifiedPhase::kEncoder, 22U, {202U})};
    snapshot.requests = {
        {201U, PhaseProjectedRequestStage::kDecode, 1, 3, 2, 0U, 2048U, false, true, true},
        {202U, PhaseProjectedRequestStage::kEncoder, 2, 2, 0, 8192U, 1024U},
    };
    snapshot.ownership = phaseProjectedOwnership(snapshot.requests);

    PhaseIncrementalProjection const result = phaseProjectEarliestCompletion(snapshot, action,
        {action.actionId,
            {{PhaseUnifiedPhase::kDecode, 21U, 8.0, 0.2, true}, {PhaseUnifiedPhase::kEncoder, 22U, 3.0, 0.1, false}}});

    ASSERT_TRUE(result.valid) << phaseProjectionReasonName(result.reason);
    EXPECT_EQ(result.completed.phase, PhaseUnifiedPhase::kEncoder);
    EXPECT_EQ(result.successor.inFlight.outstanding, PhaseExecutionSet::kDecode);
    EXPECT_EQ(result.successor.requests[1].stage, PhaseProjectedRequestStage::kPrefill);
    EXPECT_TRUE(result.successor.requests[1].visionOwned);
    EXPECT_EQ(result.successor.ownership.visionBytes, 8192U);
    ASSERT_EQ(result.ready.size(), 1U);
    EXPECT_EQ(result.ready.front().phase, PhaseUnifiedPhase::kPrefill);
    EXPECT_TRUE(hasNewcomer(result.legalActions, PhaseUnifiedPhase::kPrefill));
}

TEST(PhaseIncrementalProjectorTest, PrefillDecodeBoundaryCanRefillDecodeBeforePrefillCompletes)
{
    PhaseIncrementalAction const action
        = pairAction(PhaseGlobalActionKind::kPrefillDecode, PhaseUnifiedActionDirection::kPrefillToDecode);
    PhaseIncrementalProjectionSnapshot snapshot;
    snapshot.inFlight.outstanding = PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode;
    snapshot.inFlight.work
        = {inFlight(PhaseUnifiedPhase::kPrefill, 31U, {301U}), inFlight(PhaseUnifiedPhase::kDecode, 32U, {302U})};
    snapshot.requests = {
        {301U, PhaseProjectedRequestStage::kPrefill, 1, 4, 0, 4096U, 1024U, true},
        {302U, PhaseProjectedRequestStage::kDecode, 2, 2, 7, 0U, 1024U, false, true, true},
    };
    snapshot.ownership = phaseProjectedOwnership(snapshot.requests);

    PhaseIncrementalProjection const result = phaseProjectEarliestCompletion(snapshot, action,
        {action.actionId,
            {{PhaseUnifiedPhase::kPrefill, 31U, 9.0, 0.4, true}, {PhaseUnifiedPhase::kDecode, 32U, 2.5, 0.1, false}}});

    ASSERT_TRUE(result.valid) << phaseProjectionReasonName(result.reason);
    EXPECT_EQ(result.successor.inFlight.outstanding, PhaseExecutionSet::kPrefill);
    EXPECT_EQ(result.successor.requests[1].stage, PhaseProjectedRequestStage::kDecode);
    EXPECT_EQ(result.successor.requests[1].decodeStepsRemaining, 1);
    EXPECT_TRUE(hasNewcomer(result.legalActions, PhaseUnifiedPhase::kDecode));
}

TEST(PhaseIncrementalProjectorTest, SerialPrefillAndFinalDecodeApplyOwnershipAtTheirOwnBoundaries)
{
    PhaseIncrementalProjectionSnapshot prefillSnapshot;
    prefillSnapshot.inFlight.outstanding = PhaseExecutionSet::kPrefill;
    prefillSnapshot.inFlight.work = {inFlight(PhaseUnifiedPhase::kPrefill, 41U, {401U})};
    prefillSnapshot.requests = {
        {401U, PhaseProjectedRequestStage::kPrefill, 9, 1, 0, 2048U, 4096U, true},
    };
    prefillSnapshot.ownership = phaseProjectedOwnership(prefillSnapshot.requests);
    PhaseIncrementalAction const prefillAction = phaseIncrementalActionForDispatch(PhaseExecutionSet::kNone, 91U,
        PhaseGlobalActionKind::kPrefill, PhaseUnifiedActionDirection::kIdleLaunch, PhaseStartSkewBucket::kImmediate);

    PhaseIncrementalProjection const prefill = phaseProjectEarliestCompletion(prefillSnapshot, prefillAction,
        {prefillAction.actionId, {{PhaseUnifiedPhase::kPrefill, 41U, 4.0, 0.0, false}}});
    ASSERT_TRUE(prefill.valid);
    EXPECT_TRUE(prefill.successor.requests.front().firstTokenObserved);
    EXPECT_EQ(prefill.successor.requests.front().stage, PhaseProjectedRequestStage::kDecode);
    EXPECT_FALSE(prefill.successor.requests.front().visionOwned);
    EXPECT_TRUE(prefill.successor.requests.front().kvOwned);
    EXPECT_EQ(prefill.successor.ownership.visionBytes, 0U);
    EXPECT_EQ(prefill.successor.ownership.kvBytes, 4096U);
    EXPECT_EQ(prefill.successor.ownership.reclaimedVisionBytes, 2048U);

    PhaseIncrementalProjectionSnapshot decodeSnapshot = prefill.successor;
    decodeSnapshot.inFlight.outstanding = PhaseExecutionSet::kDecode;
    decodeSnapshot.inFlight.work = {inFlight(PhaseUnifiedPhase::kDecode, 42U, {401U})};
    PhaseIncrementalAction const decodeAction = phaseIncrementalActionForDispatch(PhaseExecutionSet::kNone, 92U,
        PhaseGlobalActionKind::kDecode, PhaseUnifiedActionDirection::kIdleLaunch, PhaseStartSkewBucket::kImmediate);
    PhaseIncrementalProjection const decode = phaseProjectEarliestCompletion(
        decodeSnapshot, decodeAction, {decodeAction.actionId, {{PhaseUnifiedPhase::kDecode, 42U, 2.0, 0.0, false}}});

    ASSERT_TRUE(decode.valid);
    EXPECT_EQ(decode.successor.requests.front().stage, PhaseProjectedRequestStage::kComplete);
    EXPECT_EQ(decode.successor.requests.front().generatedTokens, 2);
    EXPECT_FALSE(decode.successor.requests.front().kvOwned);
    EXPECT_EQ(decode.successor.ownership.kvBytes, 0U);
    EXPECT_EQ(decode.successor.ownership.reclaimedKvBytes, 4096U);
}

TEST(PhaseIncrementalProjectorTest, RejectsWholeActionCompletionVectorThatOmitsOneContext)
{
    PhaseIncrementalAction const action
        = pairAction(PhaseGlobalActionKind::kEncoderDecode, PhaseUnifiedActionDirection::kEncoderToDecode);
    PhaseIncrementalProjectionSnapshot snapshot;
    snapshot.inFlight.outstanding = PhaseExecutionSet::kEncoder | PhaseExecutionSet::kDecode;
    snapshot.inFlight.work
        = {inFlight(PhaseUnifiedPhase::kEncoder, 51U, {501U}), inFlight(PhaseUnifiedPhase::kDecode, 52U, {502U})};
    snapshot.requests = {
        {501U, PhaseProjectedRequestStage::kEncoder},
        {502U, PhaseProjectedRequestStage::kDecode, 2, 1, 0, 0U, 64U, false, true, true},
    };
    snapshot.ownership = phaseProjectedOwnership(snapshot.requests);

    PhaseIncrementalProjection const result = phaseProjectEarliestCompletion(
        snapshot, action, {action.actionId, {{PhaseUnifiedPhase::kEncoder, 51U, 10.0, 0.0, true}}});

    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason, PhaseProjectionReason::kInvalidCompletionVector);
}

TEST(PhaseIncrementalProjectorTest, RejectsACompletionVectorWithReversedIncumbentIdentity)
{
    PhaseIncrementalAction const action
        = pairAction(PhaseGlobalActionKind::kPrefillDecode, PhaseUnifiedActionDirection::kPrefillToDecode);
    PhaseIncrementalProjectionSnapshot snapshot;
    snapshot.inFlight.outstanding = PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode;
    snapshot.inFlight.work
        = {inFlight(PhaseUnifiedPhase::kPrefill, 61U, {601U}), inFlight(PhaseUnifiedPhase::kDecode, 62U, {602U})};
    snapshot.requests = {
        {601U, PhaseProjectedRequestStage::kPrefill},
        {602U, PhaseProjectedRequestStage::kDecode, 2, 1, 0, 0U, 64U, false, true, true},
    };

    PhaseIncrementalProjection const result = phaseProjectEarliestCompletion(snapshot, action,
        {action.actionId,
            {{PhaseUnifiedPhase::kPrefill, 61U, 6.0, 0.0, false}, {PhaseUnifiedPhase::kDecode, 62U, 2.0, 0.0, true}}});

    EXPECT_FALSE(result.valid);
    EXPECT_EQ(result.reason, PhaseProjectionReason::kInvalidCompletionVector);
}

TEST(PhaseFrozenReplayTest, FreezesCompletePolicyStateAndRejectsOwnershipMismatch)
{
    PhaseIncrementalAction const action
        = pairAction(PhaseGlobalActionKind::kEncoderPrefill, PhaseUnifiedActionDirection::kEncoderToPrefill);
    PhaseIncrementalProjectionSnapshot projection = encoderPrefillProjection();
    std::optional<PhaseFrozenDecisionSnapshot> const frozen = phaseFreezeDecisionSnapshot(projection, {action}, 1234U);

    ASSERT_TRUE(frozen.has_value());
    EXPECT_NE(frozen->snapshotId, 0U);
    EXPECT_EQ(frozen->scalarPolicyStateSignature, 1234U);
    projection.ownership.visionBytes += 1U;
    EXPECT_FALSE(phaseFreezeDecisionSnapshot(projection, {action}, 1234U).has_value());
}

TEST(PhaseFrozenReplayTest, ScalarEnvelopePreservesBothCompletionOrders)
{
    PhaseIncrementalAction const action
        = pairAction(PhaseGlobalActionKind::kEncoderPrefill, PhaseUnifiedActionDirection::kEncoderToPrefill);
    std::optional<PhaseFrozenDecisionSnapshot> const frozen
        = phaseFreezeDecisionSnapshot(encoderPrefillProjection(), {action}, 11U);
    ASSERT_TRUE(frozen.has_value());
    PhaseOutcomeEnvelope const envelope = phaseScalarOutcomeEnvelope(action.actionId,
        {{PhaseUnifiedPhase::kEncoder, 71U, 6.0, true}, {PhaseUnifiedPhase::kPrefill, 72U, 4.0, false}}, 10.0, 0.5);

    ASSERT_TRUE(envelope.ready);
    ASSERT_EQ(envelope.alternatives.size(), 2U);
    PhaseFrozenReplayResult const replay = phaseReplayFrozenOutcome(*frozen, action.actionId, envelope);

    ASSERT_TRUE(replay.valid) << phaseFrozenReplayReasonName(replay.reason);
    ASSERT_EQ(replay.trajectories.size(), 2U);
    ASSERT_EQ(replay.trajectories[0].boundaries.size(), 2U);
    ASSERT_EQ(replay.trajectories[1].boundaries.size(), 2U);
    EXPECT_EQ(replay.trajectories[0].boundaries.front().completed.phase, PhaseUnifiedPhase::kEncoder);
    EXPECT_EQ(replay.trajectories[1].boundaries.front().completed.phase, PhaseUnifiedPhase::kPrefill);
    for (PhaseFrozenReplayTrajectory const& trajectory : replay.trajectories)
    {
        EXPECT_EQ(trajectory.prefillReadyRows, 1U);
        EXPECT_EQ(trajectory.decodeReadyRows, 1U);
        EXPECT_EQ(trajectory.ownership.visionBytes, 100U);
        EXPECT_EQ(trajectory.ownership.kvBytes, 2000U);
        EXPECT_EQ(trajectory.ownership.reclaimedVisionBytes, 200U);
    }
}

TEST(PhaseFrozenReplayTest, EffectEnvelopeRetainsOrderUncertainty)
{
    PhaseIncrementalAction const action
        = pairAction(PhaseGlobalActionKind::kEncoderPrefill, PhaseUnifiedActionDirection::kEncoderToPrefill);
    PhaseEffectOutcomeEstimate const uncertain{0.2, 0.02, 0.1, 0.03, 0.01, 0.05, true};
    PhaseOutcomeEnvelope const envelope = phaseEffectOutcomeEnvelope(action.actionId,
        {{PhaseUnifiedPhase::kEncoder, 71U, 6.0, true}, {PhaseUnifiedPhase::kPrefill, 72U, 4.0, false}}, uncertain);

    ASSERT_TRUE(envelope.ready);
    ASSERT_EQ(envelope.alternatives.size(), 2U);
    EXPECT_LT(envelope.alternatives[0].components[1].completionUs, envelope.alternatives[0].components[0].completionUs);
    EXPECT_GT(envelope.alternatives[1].components[1].completionUs, envelope.alternatives[1].components[0].completionUs);
}

TEST(PhaseFrozenReplayTest, RejectsOutcomeForADifferentAction)
{
    PhaseIncrementalAction const action
        = pairAction(PhaseGlobalActionKind::kEncoderPrefill, PhaseUnifiedActionDirection::kEncoderToPrefill);
    std::optional<PhaseFrozenDecisionSnapshot> const frozen
        = phaseFreezeDecisionSnapshot(encoderPrefillProjection(), {action}, 11U);
    ASSERT_TRUE(frozen.has_value());
    PhaseOutcomeEnvelope envelope = phaseScalarOutcomeEnvelope(action.actionId,
        {{PhaseUnifiedPhase::kEncoder, 71U, 6.0, true}, {PhaseUnifiedPhase::kPrefill, 72U, 4.0, false}}, 10.0, 0.5);
    ++envelope.actionId;

    PhaseFrozenReplayResult const replay = phaseReplayFrozenOutcome(*frozen, action.actionId, envelope);
    EXPECT_FALSE(replay.valid);
    EXPECT_EQ(replay.reason, PhaseFrozenReplayReason::kEnvelopeActionMismatch);
}

} // namespace
} // namespace trt_edgellm::rt
