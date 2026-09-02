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

#include "runtime/phase/policy/phaseOracleH1Selector.h"

#include <gtest/gtest.h>

namespace trt_edgellm::rt
{
namespace
{

PhaseOracleH1Candidate candidate(uint64_t actionId, double boundaryUs, double uncertaintyUs, double slackUs,
    double progress, double referenceWorkUs, size_t sourceIndex = 0U)
{
    PhaseOracleH1Candidate result;
    result.sourceIndex = sourceIndex;
    result.action.actionId = actionId;
    result.action.legality = PhaseIncrementalLegalityReason::kLegal;
    result.projection.valid = true;
    result.projection.reason = PhaseProjectionReason::kProjected;
    result.projection.boundaryUs = boundaryUs;
    result.projection.robustBoundaryUs = boundaryUs + uncertaintyUs;
    result.projection.completionVector.actionId = actionId;
    result.milestones = {{1U, slackUs, boundaryUs, uncertaintyUs, progress, progress > 0.0}};
    result.referenceWorkUs = referenceWorkUs;
    return result;
}

TEST(PhaseOracleH1SelectorTest, RejectsInvalidProjectionAndMemoryOversubscription)
{
    PhaseOracleH1Candidate invalid = candidate(1U, 2.0, 0.0, 10.0, 1.0, 2.0);
    invalid.projection.valid = false;
    PhaseOracleH1Candidate memory = candidate(2U, 2.0, 0.0, 10.0, 1.0, 2.0);
    memory.hardPeakManagedBytes = 11U;
    memory.memoryBudgetBytes = 10U;

    PhaseOracleH1Decision const decision = PhaseOracleH1Selector{}.select({invalid, memory});
    EXPECT_FALSE(decision.selectedIndex.has_value());
    EXPECT_EQ(decision.reason, PhaseOracleH1DecisionReason::kNoHardFeasibleCandidate);
}

TEST(PhaseOracleH1SelectorTest, RobustSloViolationPrecedesProgressAndEfficiency)
{
    PhaseOracleH1Candidate safe = candidate(1U, 8.0, 1.0, 10.0, 1.0, 8.0);
    PhaseOracleH1Candidate fastButLate = candidate(2U, 4.0, 1.0, 3.0, 100.0, 400.0);

    PhaseOracleH1Decision const decision = PhaseOracleH1Selector{}.select({fastButLate, safe});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_DOUBLE_EQ(decision.selectedValue.robustViolationUs, 0.0);
}

TEST(PhaseOracleH1SelectorTest, UrgencyNormalizedProgressPrecedesGpuEfficiency)
{
    PhaseOracleH1Candidate efficient = candidate(1U, 5.0, 0.0, 20.0, 1.0, 100.0);
    PhaseOracleH1Candidate progress = candidate(2U, 5.0, 0.0, 20.0, 4.0, 5.0);

    PhaseOracleH1Decision const decision = PhaseOracleH1Selector{}.select({efficient, progress});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
}

TEST(PhaseOracleH1SelectorTest, EfficiencyThenOwnershipThenStableIdentityBreakTies)
{
    PhaseOracleH1Candidate lowEfficiency = candidate(4U, 5.0, 0.0, 20.0, 1.0, 5.0);
    PhaseOracleH1Candidate highEfficiency = candidate(3U, 5.0, 0.0, 20.0, 1.0, 10.0);
    PhaseOracleH1Decision decision = PhaseOracleH1Selector{}.select({lowEfficiency, highEfficiency});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);

    PhaseOracleH1Candidate lessReclaim = candidate(4U, 5.0, 0.0, 20.0, 1.0, 10.0);
    PhaseOracleH1Candidate moreReclaim = candidate(3U, 5.0, 0.0, 20.0, 1.0, 10.0);
    moreReclaim.releasedOwnershipBytes = 1024U;
    decision = PhaseOracleH1Selector{}.select({lessReclaim, moreReclaim});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);

    moreReclaim.releasedOwnershipBytes = 0U;
    decision = PhaseOracleH1Selector{}.select({lessReclaim, moreReclaim});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
}

TEST(PhaseOracleH1SelectorTest, ControlledCounterexampleUsesEarliestBoundaryNotWholePairMakespan)
{
    // A whole-action model sees only the 15 ms pair makespan. H1 observes the
    // D milestone at 6 ms while P remains outstanding, so the protected D
    // request stays within its 8 ms slack.
    PhaseOracleH1Candidate serialPrefill = candidate(10U, 10.0, 0.0, 8.0, 0.0, 10.0);
    PhaseOracleH1Candidate prefillDecode = candidate(11U, 6.0, 0.0, 8.0, 1.0, 9.0);
    prefillDecode.projection.completionVector.components = {
        {PhaseUnifiedPhase::kPrefill, 1U, 15.0, 0.0, true},
        {PhaseUnifiedPhase::kDecode, 2U, 6.0, 0.0, false},
    };

    PhaseOracleH1Decision const decision = PhaseOracleH1Selector{}.select({serialPrefill, prefillDecode});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_EQ(decision.reason, PhaseOracleH1DecisionReason::kDeadlineSafeProgressEfficiency);
}

TEST(PhaseOracleH1SelectorTest, SelectsMinimumViolationWhenEveryCandidateIsLate)
{
    PhaseOracleH1Candidate lessLate = candidate(1U, 7.0, 0.0, 5.0, 1.0, 7.0);
    PhaseOracleH1Candidate moreLate = candidate(2U, 9.0, 0.0, 5.0, 100.0, 900.0);

    PhaseOracleH1Decision const decision = PhaseOracleH1Selector{}.select({moreLate, lessLate});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_EQ(decision.reason, PhaseOracleH1DecisionReason::kMinimumRobustViolation);
    EXPECT_DOUBLE_EQ(decision.selectedValue.robustViolationUs, 2.0);
}

} // namespace
} // namespace trt_edgellm::rt
