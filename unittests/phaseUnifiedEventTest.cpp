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

#include "runtime/phase/mechanism/phaseUnifiedEvent.h"

#include <gtest/gtest.h>

namespace trt_edgellm::rt
{
namespace
{

TEST(PhaseUnifiedEventTest, ReportsStableSchemaAndNames)
{
    EXPECT_EQ(kPHASE_UNIFIED_EVENT_SCHEMA_VERSION, 1U);
    EXPECT_STREQ(phaseUnifiedEventKindName(PhaseUnifiedEventKind::kDecision), "decision");
    EXPECT_STREQ(phaseUnifiedEventKindName(PhaseUnifiedEventKind::kDispatch), "dispatch");
    EXPECT_STREQ(phaseUnifiedEventKindName(PhaseUnifiedEventKind::kCompletion), "completion");
    EXPECT_STREQ(phaseUnifiedPhaseName(PhaseUnifiedPhase::kEncoder), "encoder");
    EXPECT_STREQ(phaseUnifiedPhaseName(PhaseUnifiedPhase::kPrefill), "prefill");
    EXPECT_STREQ(phaseUnifiedPhaseName(PhaseUnifiedPhase::kDecode), "decode");
    EXPECT_STREQ(phaseInFlightStatusName(PhaseInFlightStatus::kCompletionReady), "completion_ready");
    EXPECT_EQ(phaseUnifiedActionDirectionFromName("prefill_to_decode"), PhaseUnifiedActionDirection::kPrefillToDecode);
    EXPECT_FALSE(phaseUnifiedActionDirectionFromName("prefill_decode").has_value());
    EXPECT_STREQ(phaseUnifiedDispatchModeName(PhaseUnifiedDispatchMode::kCoLaunch), "co_launch");
    EXPECT_STREQ(
        phaseUnifiedDispatchModeName(PhaseUnifiedDispatchMode::kResidualAugmentation), "residual_augmentation");
    EXPECT_STREQ(
        phaseUnifiedFidelityReasonName(PhaseUnifiedFidelityReason::kOutstandingMismatch), "outstanding_mismatch");
}

TEST(PhaseUnifiedEventTest, SeparatesCoLaunchFromResidualAugmentation)
{
    EXPECT_EQ(phaseUnifiedDispatchMode(PhaseExecutionSet::kNone, PhaseGlobalActionKind::kPrefillDecode),
        PhaseUnifiedDispatchMode::kCoLaunch);
    EXPECT_EQ(phaseUnifiedDispatchMode(PhaseExecutionSet::kPrefill, PhaseGlobalActionKind::kPrefillDecode),
        PhaseUnifiedDispatchMode::kResidualAugmentation);
    EXPECT_EQ(phaseUnifiedDispatchMode(PhaseExecutionSet::kNone, PhaseGlobalActionKind::kDecode),
        PhaseUnifiedDispatchMode::kSingle);
    EXPECT_EQ(phaseUnifiedDispatchMode(PhaseExecutionSet::kNone, PhaseGlobalActionKind::kWait),
        PhaseUnifiedDispatchMode::kNone);
}

TEST(PhaseUnifiedEventTest, UsesPlanIdentityForMultiPhaseMemberActions)
{
    EXPECT_TRUE(phaseUnifiedActionIdentityMatches(PhaseUnifiedDispatchMode::kSingle, 7U, 7U));
    EXPECT_FALSE(phaseUnifiedActionIdentityMatches(PhaseUnifiedDispatchMode::kSingle, 7U, 8U));
    EXPECT_TRUE(phaseUnifiedActionIdentityMatches(PhaseUnifiedDispatchMode::kCoLaunch, 7U, 8U));
    EXPECT_TRUE(phaseUnifiedActionIdentityMatches(PhaseUnifiedDispatchMode::kResidualAugmentation, 7U, 8U));
}

TEST(PhaseUnifiedEventTest, DerivesIncrementalActionDirectionFromOutstandingWork)
{
    EXPECT_EQ(phaseUnifiedActionDirection(PhaseExecutionSet::kNone, PhaseUnifiedPhase::kEncoder),
        PhaseUnifiedActionDirection::kIdleLaunch);
    EXPECT_EQ(phaseUnifiedActionDirection(PhaseExecutionSet::kEncoder, PhaseUnifiedPhase::kPrefill),
        PhaseUnifiedActionDirection::kEncoderToPrefill);
    EXPECT_EQ(phaseUnifiedActionDirection(PhaseExecutionSet::kEncoder, PhaseUnifiedPhase::kDecode),
        PhaseUnifiedActionDirection::kEncoderToDecode);
    EXPECT_EQ(phaseUnifiedActionDirection(PhaseExecutionSet::kPrefill, PhaseUnifiedPhase::kEncoder),
        PhaseUnifiedActionDirection::kPrefillToEncoder);
    EXPECT_EQ(phaseUnifiedActionDirection(PhaseExecutionSet::kPrefill, PhaseUnifiedPhase::kDecode),
        PhaseUnifiedActionDirection::kPrefillToDecode);
    EXPECT_EQ(phaseUnifiedActionDirection(PhaseExecutionSet::kDecode, PhaseUnifiedPhase::kEncoder),
        PhaseUnifiedActionDirection::kDecodeToEncoder);
    EXPECT_EQ(phaseUnifiedActionDirection(PhaseExecutionSet::kDecode, PhaseUnifiedPhase::kPrefill),
        PhaseUnifiedActionDirection::kDecodeToPrefill);
}

TEST(PhaseUnifiedEventTest, AccumulatesObservedPhaseMask)
{
    PhaseExecutionSet observed = phaseExecutionSetForUnifiedPhase(PhaseUnifiedPhase::kEncoder)
        | phaseExecutionSetForUnifiedPhase(PhaseUnifiedPhase::kDecode);
    EXPECT_TRUE(phaseExecutionSetContains(observed, PhaseExecutionSet::kEncoder));
    EXPECT_TRUE(phaseExecutionSetContains(observed, PhaseExecutionSet::kDecode));
    EXPECT_FALSE(phaseExecutionSetContains(observed, PhaseExecutionSet::kPrefill));
    EXPECT_EQ(phaseExecutionSetForUnifiedPhase(PhaseUnifiedPhase::kCopy), PhaseExecutionSet::kNone);
}

TEST(PhaseUnifiedEventTest, ReportsDirectionalInjectionEndpoints)
{
    EXPECT_EQ(phaseUnifiedDirectionIncumbentPhase(PhaseUnifiedActionDirection::kEncoderToPrefill),
        PhaseUnifiedPhase::kEncoder);
    EXPECT_EQ(phaseUnifiedDirectionNewcomerPhase(PhaseUnifiedActionDirection::kEncoderToPrefill),
        PhaseUnifiedPhase::kPrefill);
    EXPECT_EQ(
        phaseUnifiedDirectionIncumbentPhase(PhaseUnifiedActionDirection::kDecodeToEncoder), PhaseUnifiedPhase::kDecode);
    EXPECT_EQ(
        phaseUnifiedDirectionNewcomerPhase(PhaseUnifiedActionDirection::kDecodeToEncoder), PhaseUnifiedPhase::kEncoder);
    EXPECT_TRUE(phaseUnifiedDirectionsSharePair(
        PhaseUnifiedActionDirection::kEncoderToDecode, PhaseUnifiedActionDirection::kDecodeToEncoder));
    EXPECT_FALSE(phaseUnifiedDirectionsSharePair(
        PhaseUnifiedActionDirection::kEncoderToDecode, PhaseUnifiedActionDirection::kEncoderToPrefill));
}

TEST(PhaseUnifiedEventTest, SnapshotSignatureIgnoresTransientIdsAndCanonicalizesInflightOrder)
{
    PhaseUnifiedEvent left;
    left.outstandingBefore = PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode;
    left.ready.prefillRows = 2;
    left.ready.prefillTokens = 256;
    left.ready.decodeRows = 3;
    left.ready.decodeContextTokens = 3072;
    left.readyEncoderRequestIds = {9U};
    left.readyPrefillRequestIds = {2U, 3U};
    left.readyPrefillTokenCounts = {128, 64};
    left.readyDecodeRequestIds = {4U, 5U, 6U};
    left.readyDecodeContextLengths = {1024, 896, 768};
    left.pagePoolAllocatedBundles = 11;
    left.pageReservationGuaranteedBundles = 3;
    left.visionPayloadBytes = 4096U;
    PhaseInFlightWorkSnapshot prefill;
    prefill.phase = PhaseUnifiedPhase::kPrefill;
    prefill.status = PhaseInFlightStatus::kRunning;
    prefill.executionId = 77U;
    prefill.planId = 88U;
    prefill.dispatchHostNs = 100U;
    prefill.dispatchAgeUs = 5.0;
    prefill.requestIds = {7U};
    prefill.work.prefillRows = 1;
    prefill.work.prefillTokens = 128;
    PhaseInFlightWorkSnapshot decode;
    decode.phase = PhaseUnifiedPhase::kDecode;
    decode.status = PhaseInFlightStatus::kSubmitted;
    decode.executionId = 79U;
    decode.requestIds = {8U};
    decode.work.decodeRows = 1;
    decode.work.decodeContextTokens = 1024;
    left.inFlight.work = {prefill, decode};

    PhaseUnifiedEvent right = left;
    right.inFlight.work = {decode, prefill};
    right.inFlight.work[0].executionId = 1001U;
    right.inFlight.work[1].executionId = 1002U;
    right.inFlight.work[1].dispatchHostNs = 9999U;
    right.inFlight.work[1].dispatchAgeUs = 999.0;
    EXPECT_EQ(phaseUnifiedSnapshotSignature(left), phaseUnifiedSnapshotSignature(right));

    right.readyDecodeRequestIds = {4U, 6U, 5U};
    EXPECT_NE(phaseUnifiedSnapshotSignature(left), phaseUnifiedSnapshotSignature(right));
    right = left;
    ++right.pagePoolAllocatedBundles;
    EXPECT_NE(phaseUnifiedSnapshotSignature(left), phaseUnifiedSnapshotSignature(right));
    right = left;
    ++right.readyDecodeContextLengths.front();
    EXPECT_NE(phaseUnifiedSnapshotSignature(left), phaseUnifiedSnapshotSignature(right));
}

} // namespace
} // namespace trt_edgellm::rt
