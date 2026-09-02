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

#include "runtime/phase/mechanism/phaseIncrementalAction.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace trt_edgellm::rt
{
namespace
{

std::vector<PhaseIncrementalReadyAction> readyMask()
{
    return {{PhaseUnifiedPhase::kEncoder, 11U}, {PhaseUnifiedPhase::kPrefill, 22U}, {PhaseUnifiedPhase::kDecode, 33U}};
}

TEST(PhaseIncrementalActionTest, EnumeratesBoundedIdleFrontierWithBothPairOrders)
{
    PhaseInFlightSnapshot snapshot;
    std::vector<PhaseIncrementalAction> const actions
        = phaseEnumerateIncrementalActions(snapshot, readyMask(), PhaseStartSkewBucket::kImmediate);

    ASSERT_EQ(actions.size(), 10U);
    EXPECT_EQ(
        std::count_if(actions.begin(), actions.end(), [](auto const& action) { return action.key.noDispatch; }), 1);
    EXPECT_EQ(
        std::count_if(actions.begin(), actions.end(),
            [](auto const& action) { return action.key.direction == PhaseUnifiedActionDirection::kEncoderToDecode; }),
        1);
    EXPECT_EQ(
        std::count_if(actions.begin(), actions.end(),
            [](auto const& action) { return action.key.direction == PhaseUnifiedActionDirection::kDecodeToEncoder; }),
        1);
    EXPECT_TRUE(std::all_of(actions.begin(), actions.end(), [](auto const& action) { return action.legal(); }));
}

TEST(PhaseIncrementalActionTest, OneOutstandingContextAddsOnlyDistinctNewcomers)
{
    PhaseInFlightSnapshot snapshot;
    snapshot.outstanding = PhaseExecutionSet::kEncoder;
    snapshot.work.push_back({PhaseUnifiedPhase::kEncoder, PhaseInFlightStatus::kRunning, 7U, 0U, 1U, 11U});
    std::vector<PhaseIncrementalAction> const actions
        = phaseEnumerateIncrementalActions(snapshot, readyMask(), PhaseStartSkewBucket::kHalf);

    ASSERT_EQ(actions.size(), 3U);
    EXPECT_EQ(std::count_if(actions.begin(), actions.end(),
                  [](auto const& action) {
                      return action.key.direction == PhaseUnifiedActionDirection::kEncoderToPrefill
                          && action.key.startSkew == PhaseStartSkewBucket::kHalf;
                  }),
        1);
    EXPECT_EQ(std::count_if(actions.begin(), actions.end(),
                  [](auto const& action) {
                      return action.key.direction == PhaseUnifiedActionDirection::kEncoderToDecode
                          && action.key.startSkew == PhaseStartSkewBucket::kHalf;
                  }),
        1);
}

TEST(PhaseIncrementalActionTest, TwoOutstandingContextsPermitOnlyNoDispatch)
{
    PhaseInFlightSnapshot snapshot;
    snapshot.outstanding = PhaseExecutionSet::kEncoder | PhaseExecutionSet::kDecode;
    std::vector<PhaseIncrementalAction> const actions = phaseEnumerateIncrementalActions(snapshot, readyMask());

    ASSERT_EQ(actions.size(), 1U);
    EXPECT_TRUE(actions.front().key.noDispatch);
    EXPECT_EQ(actions.front().key.plannedOutstanding, snapshot.outstanding);
}

TEST(PhaseIncrementalActionTest, RejectsDuplicatePhaseLocalFrontierEntries)
{
    PhaseInFlightSnapshot snapshot;
    std::vector<PhaseIncrementalReadyAction> duplicate{
        {PhaseUnifiedPhase::kDecode, 1U}, {PhaseUnifiedPhase::kDecode, 2U}};

    EXPECT_TRUE(phaseEnumerateIncrementalActions(snapshot, duplicate).empty());
}

TEST(PhaseIncrementalActionTest, RejectsTripleAndSameContextTransitionsExplicitly)
{
    PhaseIncrementalActionKey same;
    same.outstandingBefore = PhaseExecutionSet::kDecode;
    same.plannedOutstanding = PhaseExecutionSet::kDecode;
    same.incumbentPhase = PhaseUnifiedPhase::kDecode;
    same.newcomerPhase = PhaseUnifiedPhase::kDecode;
    same.direction = PhaseUnifiedActionDirection::kDecodeToPrefill;
    EXPECT_EQ(phaseIncrementalActionLegality(same), PhaseIncrementalLegalityReason::kSameContextReenqueue);

    PhaseIncrementalActionKey third;
    third.outstandingBefore = PhaseExecutionSet::kEncoder | PhaseExecutionSet::kDecode;
    third.plannedOutstanding = PhaseExecutionSet::kEncoder | PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode;
    third.incumbentPhase = PhaseUnifiedPhase::kDecode;
    third.newcomerPhase = PhaseUnifiedPhase::kPrefill;
    third.direction = PhaseUnifiedActionDirection::kDecodeToPrefill;
    EXPECT_EQ(phaseIncrementalActionLegality(third), PhaseIncrementalLegalityReason::kThirdContext);

    PhaseInFlightSnapshot invalid;
    invalid.outstanding = PhaseExecutionSet::kEncoder | PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode;
    EXPECT_TRUE(phaseEnumerateIncrementalActions(invalid, readyMask()).empty());
}

TEST(PhaseIncrementalActionTest, DirectionAndSkewArePartOfStableIdentity)
{
    PhaseIncrementalActionKey key;
    key.cohortId = 19U;
    key.incumbentPhase = PhaseUnifiedPhase::kPrefill;
    key.newcomerPhase = PhaseUnifiedPhase::kDecode;
    key.direction = PhaseUnifiedActionDirection::kPrefillToDecode;
    key.plannedOutstanding = PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode;
    key.startSkew = PhaseStartSkewBucket::kQuarter;
    uint64_t const base = phaseIncrementalActionId(key);

    key.startSkew = PhaseStartSkewBucket::kHalf;
    EXPECT_NE(phaseIncrementalActionId(key), base);
    key.startSkew = PhaseStartSkewBucket::kQuarter;
    key.incumbentPhase = PhaseUnifiedPhase::kDecode;
    key.newcomerPhase = PhaseUnifiedPhase::kPrefill;
    key.direction = PhaseUnifiedActionDirection::kDecodeToPrefill;
    EXPECT_NE(phaseIncrementalActionId(key), base);
}

TEST(PhaseIncrementalActionTest, ClassifiesMeasuredAndSerialStartSkew)
{
    EXPECT_EQ(phaseStartSkewBucketFromFraction(0.48, true), PhaseStartSkewBucket::kHalf);
    EXPECT_EQ(phaseStartSkewBucketFromFraction(0.91, true), PhaseStartSkewBucket::kNearComplete);
    EXPECT_EQ(phaseStartSkewBucketFromFraction(0.50, false), PhaseStartSkewBucket::kSerial);
    EXPECT_STREQ(phaseStartSkewBucketName(PhaseStartSkewBucket::kSerial), "serial_realization");
}

} // namespace
} // namespace trt_edgellm::rt
