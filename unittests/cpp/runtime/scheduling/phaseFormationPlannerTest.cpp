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

#include "runtime/phase/policy/phaseFormationPlanner.h"

#include <gtest/gtest.h>

namespace trt_edgellm::rt
{
namespace
{

TEST(PhaseFormationPlannerTest, MapsCanonicalRowsWithoutChangingTheirOwnership)
{
    PhaseGlobalActionCandidate candidate;
    candidate.key.kind = PhaseGlobalActionKind::kEncoderPrefill;
    candidate.primaryRequestIds = {7U, 3U};
    candidate.secondaryRequestIds = {11U, 5U, 2U};

    PhaseFormationWork const work = phaseFormationWork(candidate);

    EXPECT_EQ(work.encoderRows, 2U);
    EXPECT_EQ(work.prefillRows, 3U);
    EXPECT_EQ(work.decodeRows, 0U);
    EXPECT_EQ(candidate.primaryRequestIds, (std::vector<uint64_t>{7U, 3U}));
    EXPECT_EQ(candidate.secondaryRequestIds, (std::vector<uint64_t>{11U, 5U, 2U}));
}

TEST(PhaseFormationPlannerTest, AppliesOnlyConcreteCompletionsReachedByTheAction)
{
    PhaseFormationSnapshot const snapshot{9U, {1U, 0U, 2U}, {{17U, 4.0, {0U, 0U, 3U}}, {19U, 12.0, {2U, 0U, 0U}}}};
    PhaseFormationAction const decode{0U, {0U, 0U, 2U}, 5.0, 0.2, true};

    PhaseFormationTransition const result = phaseFormationTransition(snapshot, decode);

    ASSERT_TRUE(result.feasible);
    EXPECT_EQ(result.successor.ready.encoderRows, 1U);
    EXPECT_EQ(result.successor.ready.decodeRows, 3U);
    ASSERT_EQ(result.successor.knownCompletions.size(), 1U);
    EXPECT_EQ(result.successor.knownCompletions.front().eventId, 19U);
    EXPECT_DOUBLE_EQ(result.successor.knownCompletions.front().readyAfterUs, 7.0);
}

TEST(PhaseFormationPlannerTest, UsesEqualWorkInsteadOfRewardingASmallerAction)
{
    PhaseFormationSnapshot const snapshot{1U, {1U, 1U, 0U}, {}};
    std::vector<PhaseFormationAction> const actions{
        {0U, {1U, 0U, 0U}, 4.0, 0.0, true},
        {1U, {0U, 1U, 0U}, 5.0, 0.0, true},
        {2U, {1U, 1U, 0U}, 7.0, 0.0, true},
    };

    PhaseFormationOracleResult const result = phaseFormationEvaluateH2(snapshot, actions, {1U, 1U, 0U});

    ASSERT_TRUE(result.selectedAction.has_value());
    EXPECT_EQ(*result.selectedAction, 2U);
    EXPECT_DOUBLE_EQ(result.sequences[0].makespanUs, 9.0);
    EXPECT_DOUBLE_EQ(result.sequences[1].makespanUs, 9.0);
    EXPECT_DOUBLE_EQ(result.sequences[2].makespanUs, 7.0);
}

TEST(PhaseFormationPlannerTest, KnownEventCanExposeActionInducedFragmentation)
{
    // Overlap is the best immediate action (6 us versus 5+8 us serial for the
    // currently ready E1/P1), but it consumes E1 before the concrete event.
    // P-first lets that event form E2 and finishes the fixed E2/P1 frontier in
    // 12 us instead of overlap+E1 in 14 us.
    PhaseFormationSnapshot const snapshot{3U, {1U, 1U, 0U}, {{23U, 5.0, {1U, 0U, 0U}}}};
    std::vector<PhaseFormationAction> const actions{
        {0U, {1U, 1U, 0U}, 6.0, 0.0, true},
        {1U, {0U, 1U, 0U}, 5.0, 0.0, true},
        {2U, {1U, 0U, 0U}, 8.0, 0.0, false},
        {3U, {2U, 0U, 0U}, 7.0, 0.0, false},
    };

    PhaseFormationOracleResult const result = phaseFormationEvaluateH2(snapshot, actions, {2U, 1U, 0U});

    ASSERT_TRUE(result.selectedAction.has_value());
    EXPECT_EQ(*result.selectedAction, 1U);
    EXPECT_DOUBLE_EQ(result.sequences[0].makespanUs, 14.0);
    EXPECT_DOUBLE_EQ(result.sequences[1].makespanUs, 12.0);
    ASSERT_TRUE(result.sequences[1].successorAction.has_value());
    EXPECT_EQ(*result.sequences[1].successorAction, 3U);
}

TEST(PhaseFormationPlannerTest, ReplayIdentityDependsOnlyOnObservableState)
{
    PhaseFormationSnapshot const snapshot{41U, {4U, 8U, 32U}, {{5U, 120.0, {0U, 0U, 16U}}}};
    EXPECT_EQ(phaseFormationSnapshotId(snapshot), phaseFormationSnapshotId(snapshot));

    PhaseFormationSnapshot changed = snapshot;
    changed.knownCompletions.front().readyAfterUs = 121.0;
    EXPECT_NE(phaseFormationSnapshotId(snapshot), phaseFormationSnapshotId(changed));

    changed = snapshot;
    changed.decodeServiceBudgetUs = 500.0;
    EXPECT_NE(phaseFormationSnapshotId(snapshot), phaseFormationSnapshotId(changed));
}

TEST(PhaseFormationPlannerTest, ProtectsResidentDecodeBeforeMinimizingEqualWorkHorizon)
{
    PhaseFormationSnapshot const snapshot{12U, {1U, 1U, 0U}, {}, 10.0};
    std::vector<PhaseFormationAction> const actions{
        // Immediate E+P has the shortest horizon but delays a resident D row
        // past its robust TPOT budget.
        {0U, {1U, 1U, 0U}, 5.0, 0.0, true, 14.0, 0.0},
        // P-first preserves D service and then completes E as its successor.
        {1U, {0U, 1U, 0U}, 6.0, 0.0, true, 8.0, 0.0},
        {2U, {1U, 0U, 0U}, 4.0, 0.0, false},
    };

    PhaseFormationOracleResult const result = phaseFormationEvaluateH2(snapshot, actions, {1U, 1U, 0U});

    ASSERT_TRUE(result.selectedAction.has_value());
    EXPECT_EQ(*result.selectedAction, 1U);
    EXPECT_DOUBLE_EQ(result.sequences[0].decodeServiceViolationUs, 4.0);
    EXPECT_DOUBLE_EQ(result.sequences[1].decodeServiceViolationUs, 0.0);
    EXPECT_DOUBLE_EQ(result.sequences[1].makespanUs, 10.0);

    PhaseFormationRegret const regret = phaseFormationPredictedRegret(result, 0U);
    ASSERT_TRUE(regret.valid);
    EXPECT_EQ(regret.selectedAction, 0U);
    EXPECT_EQ(regret.oracleAction, 1U);
    EXPECT_DOUBLE_EQ(regret.predictedRegretUs, 4.0);
}

TEST(PhaseFormationPlannerTest, AppendsOnlyTheFirstConcreteDecodeBoundary)
{
    PhaseFormationSnapshot const snapshot{14U, {1U, 1U, 0U},
        {{31U, 3.0, {0U, 0U, 8U}, 1.0, 2.0, 0.5, 5.0}, {32U, 6.0, {0U, 0U, 16U}, 1.0, 4.0, 0.5, 5.0}}};
    std::vector<PhaseFormationAction> const actions{
        {0U, {1U, 1U, 0U}, 7.0, 0.0, true},
        {1U, {1U, 0U, 0U}, 5.0, 0.0, true},
        {2U, {0U, 1U, 0U}, 5.0, 0.0, true},
    };

    PhaseFormationOracleResult const result = phaseFormationEvaluateH2(snapshot, actions, {1U, 1U, 0U});

    ASSERT_TRUE(result.selectedAction.has_value());
    EXPECT_EQ(*result.selectedAction, 0U);
    EXPECT_DOUBLE_EQ(result.sequences[0].makespanUs, 9.0);
    EXPECT_DOUBLE_EQ(result.sequences[0].uncertaintyUs, 1.5);
    EXPECT_DOUBLE_EQ(result.sequences[0].decodeServiceViolationUs, 2.5);
    // The cumulative second preview is not appended as a second D action.
    EXPECT_LT(result.sequences[0].makespanUs, 13.0);
}

TEST(PhaseFormationPlannerTest, CounterfactualOraclePreservesAllProtectedDeadlines)
{
    PhaseFormationSnapshot const snapshot{13U, {1U, 1U, 0U}, {}};
    std::vector<PhaseFormationAction> const actions{
        {0U, {1U, 1U, 0U}, 5.0, 0.0, true, std::numeric_limits<double>::infinity(), 0.0, 3.0},
        {1U, {0U, 1U, 0U}, 6.0, 0.0, true},
        {2U, {1U, 0U, 0U}, 4.0, 0.0, false},
    };

    PhaseFormationOracleResult const result = phaseFormationEvaluateH2(snapshot, actions, {1U, 1U, 0U});

    ASSERT_TRUE(result.selectedAction.has_value());
    EXPECT_EQ(*result.selectedAction, 1U);
    EXPECT_DOUBLE_EQ(result.sequences[0].protectedViolationUs, 3.0);
    EXPECT_DOUBLE_EQ(result.sequences[1].protectedViolationUs, 0.0);
    PhaseFormationRegret const regret = phaseFormationPredictedRegret(result, 0U);
    ASSERT_TRUE(regret.valid);
    EXPECT_DOUBLE_EQ(regret.predictedRegretUs, 3.0);
}

TEST(PhaseFormationPlannerTest, ReplacesMyopicOnlyForStrictRobustImprovement)
{
    PhaseFormationOracleResult tied;
    tied.selectedAction = 1U;
    tied.sequences.resize(2U);
    for (PhaseFormationSequence& sequence : tied.sequences)
    {
        sequence.feasible = true;
        sequence.makespanUs = 10.0;
    }
    EXPECT_FALSE(phaseFormationShouldReplaceMyopic(tied, 0U, 1U));

    tied.sequences[0].makespanUs = 12.0;
    EXPECT_TRUE(phaseFormationShouldReplaceMyopic(tied, 0U, 1U));
    EXPECT_FALSE(phaseFormationShouldReplaceMyopic(tied, 1U, 1U));

    tied.sequences.push_back(tied.sequences[0]);
    tied.sequences[2].makespanUs = 11.0;
    EXPECT_TRUE(phaseFormationShouldReplaceMyopic(tied, 0U, 2U));
    tied.sequences[2].makespanUs = 13.0;
    EXPECT_FALSE(phaseFormationShouldReplaceMyopic(tied, 0U, 2U));
}

TEST(PhaseFormationPlannerTest, PopulatesGlobalCandidateHorizonsWithOneReferenceFrontier)
{
    PhaseGlobalActionCandidate encoder;
    encoder.key.kind = PhaseGlobalActionKind::kEncoder;
    encoder.primaryRequestIds = {1U};
    encoder.predictedMakespanUs = 4.0;
    encoder.referenceWorkUs = 4.0;
    PhaseGlobalActionCandidate prefill;
    prefill.key.kind = PhaseGlobalActionKind::kPrefill;
    prefill.primaryRequestIds = {2U};
    prefill.predictedMakespanUs = 5.0;
    prefill.referenceWorkUs = 5.0;
    PhaseGlobalActionCandidate overlap;
    overlap.key.kind = PhaseGlobalActionKind::kEncoderPrefill;
    overlap.primaryRequestIds = {1U};
    overlap.secondaryRequestIds = {2U};
    overlap.predictedMakespanUs = 7.0;
    overlap.referenceWorkUs = 9.0;
    std::vector<PhaseGlobalActionCandidate> candidates{encoder, prefill, overlap};

    PhaseFormationOracleResult const result
        = phaseFormationApplyH2(candidates, {7U, {1U, 1U, 0U}, {}}, {1U, 1U, 0U}, 9.0);

    ASSERT_TRUE(result.selectedAction.has_value());
    EXPECT_EQ(*result.selectedAction, 2U);
    for (PhaseGlobalActionCandidate const& candidate : candidates)
    {
        EXPECT_DOUBLE_EQ(candidate.horizonReferenceWorkUs, 9.0);
    }
    EXPECT_DOUBLE_EQ(candidates[0].predictedHorizonUs, 9.0);
    EXPECT_DOUBLE_EQ(candidates[1].predictedHorizonUs, 9.0);
    EXPECT_DOUBLE_EQ(candidates[2].predictedHorizonUs, 7.0);
}

TEST(PhaseFormationPlannerTest, UsesDecisionCostWithoutReplacingExactExecutionCost)
{
    PhaseGlobalActionCandidate encoder;
    encoder.key.kind = PhaseGlobalActionKind::kEncoder;
    encoder.primaryRequestIds = {1U};
    encoder.predictedMakespanUs = 5.0;
    PhaseGlobalActionCandidate prefill;
    prefill.key.kind = PhaseGlobalActionKind::kPrefill;
    prefill.primaryRequestIds = {2U};
    prefill.predictedMakespanUs = 5.0;
    PhaseGlobalActionCandidate overlap;
    overlap.key.kind = PhaseGlobalActionKind::kEncoderPrefill;
    overlap.primaryRequestIds = {1U};
    overlap.secondaryRequestIds = {2U};
    overlap.predictedMakespanUs = 12.0;
    overlap.decisionCostKnown = true;
    overlap.decisionMakespanUs = 8.0;
    std::vector<PhaseGlobalActionCandidate> candidates{encoder, prefill, overlap};

    PhaseFormationOracleResult const result
        = phaseFormationApplyH2(candidates, {8U, {1U, 1U, 0U}, {}}, {1U, 1U, 0U}, 10.0);

    ASSERT_TRUE(result.selectedAction.has_value());
    EXPECT_EQ(*result.selectedAction, 2U);
    EXPECT_DOUBLE_EQ(candidates[2].predictedMakespanUs, 12.0);
    EXPECT_DOUBLE_EQ(candidates[2].predictedHorizonUs, 8.0);
}

TEST(PhaseFormationPlannerTest, AttributesFourActualDispatchesAndResidentDecodeService)
{
    PhaseFormationRealizedTracker tracker({4U, 2U});
    PhaseFormationRealizedEpisodeStart const start{17U, PhaseGlobalActionKind::kEncoder,
        PhaseGlobalActionKind::kEncoderPrefill, PhaseGlobalActionKind::kEncoder, 3.0, 15.0, 100.0};

    tracker.startEpisode(start, 1U, PhaseGlobalActionKind::kEncoder, {1U, 0U, 0U});
    tracker.observeCompletion(1U, 105.0);
    tracker.observeDispatch(2U, PhaseGlobalActionKind::kDecode, {0U, 0U, 4U}, 110.0);
    tracker.observeCompletion(2U, 112.0);
    tracker.observeDispatch(3U, PhaseGlobalActionKind::kPrefill, {0U, 2U, 0U}, 115.0);
    tracker.observeCompletion(3U, 120.0);
    tracker.observeDispatch(4U, PhaseGlobalActionKind::kDecode, {0U, 0U, 8U}, 121.0);
    tracker.observeCompletion(4U, 130.0);

    PhaseFormationRealizedTelemetry const& telemetry = tracker.telemetry();
    EXPECT_EQ(telemetry.episodesStarted, 1U);
    EXPECT_EQ(telemetry.episodesCompleted, 1U);
    EXPECT_EQ(telemetry.episodesTruncated, 0U);
    EXPECT_EQ(telemetry.decodeServices, 1U);
    ASSERT_TRUE(telemetry.lastEpisode.has_value());
    PhaseFormationRealizedEpisode const& episode = *telemetry.lastEpisode;
    EXPECT_EQ(episode.dispatches.size(), 4U);
    EXPECT_EQ(episode.encoderRows, 1U);
    EXPECT_EQ(episode.prefillRows, 2U);
    EXPECT_EQ(episode.decodeRows, 12U);
    EXPECT_EQ(episode.firstDecodeRows, 4U);
    EXPECT_EQ(episode.maxDecodeRows, 8U);
    EXPECT_DOUBLE_EQ(episode.decodeServiceGapUs, 10.0);
    EXPECT_DOUBLE_EQ(episode.decodeCompletionVisibleUs, 12.0);
    EXPECT_DOUBLE_EQ(episode.horizonCompletionVisibleUs, 30.0);
    EXPECT_DOUBLE_EQ(episode.decodeServiceViolationUs, 0.0);
    std::vector<PhaseFormationRealizedEpisode> completed = tracker.takeCompletedEpisodes();
    ASSERT_EQ(completed.size(), 1U);
    EXPECT_EQ(completed.front().episodeId, episode.episodeId);
    EXPECT_TRUE(tracker.takeCompletedEpisodes().empty());
}

TEST(PhaseFormationPlannerTest, RemapsAnAugmentedLeaseBeforeCompletionVisibility)
{
    PhaseFormationRealizedTracker tracker({1U, 2U});
    PhaseFormationRealizedEpisodeStart const start{19U, PhaseGlobalActionKind::kPrefill,
        PhaseGlobalActionKind::kEncoderPrefill, PhaseGlobalActionKind::kPrefill, 0.0,
        std::numeric_limits<double>::infinity(), 40.0};

    tracker.startEpisode(start, 7U, PhaseGlobalActionKind::kPrefill, {0U, 1U, 0U});
    tracker.remapPlan(7U, 9U);
    tracker.observeCompletion(9U, 46.0);

    ASSERT_TRUE(tracker.telemetry().lastEpisode.has_value());
    EXPECT_EQ(tracker.telemetry().episodesCompleted, 1U);
    EXPECT_DOUBLE_EQ(tracker.telemetry().lastEpisode->horizonCompletionVisibleUs, 6.0);
}

TEST(PhaseFormationPlannerTest, RecordsAnUnservicedDecodeBudgetViolation)
{
    PhaseFormationRealizedTracker tracker({1U, 2U});
    PhaseFormationRealizedEpisodeStart const start{23U, PhaseGlobalActionKind::kEncoder,
        PhaseGlobalActionKind::kEncoderDecode, PhaseGlobalActionKind::kEncoder, 0.0, 1.0, 10.0};

    tracker.startEpisode(start, 5U, PhaseGlobalActionKind::kEncoder, {1U, 0U, 0U});
    tracker.observeCompletion(5U, 12.0);

    PhaseFormationRealizedTelemetry const& telemetry = tracker.telemetry();
    ASSERT_TRUE(telemetry.lastEpisode.has_value());
    EXPECT_FALSE(telemetry.lastEpisode->decodeServiced);
    EXPECT_DOUBLE_EQ(telemetry.lastEpisode->decodeServiceGapUs, 2.0);
    EXPECT_DOUBLE_EQ(telemetry.lastEpisode->decodeServiceViolationUs, 1.0);
    EXPECT_EQ(telemetry.decodeServiceViolations, 1U);
}

TEST(PhaseFormationPlannerTest, ReplaysPhysicalCompletionOrderIntoRequestDagAndOwnership)
{
    std::vector<PhaseFormationRequestState> requests{
        {1U, PhaseFormationRequestStage::kEncoderReady, 2U, 100U, 0U},
        {2U, PhaseFormationRequestStage::kPrefillReady, 3U, 200U, 20U},
    };
    std::vector<PhaseFormationPhysicalCompletion> completions{
        {PhaseGlobalActionKind::kEncoder, {1U}, 8.0, 1.0},
        {PhaseGlobalActionKind::kPrefill, {2U}, 5.0, 0.5},
    };

    PhaseFormationTwoBoundaryResult const result
        = phaseFormationEvaluateCompletionBoundaries(std::move(requests), std::move(completions));

    ASSERT_TRUE(result.feasible);
    EXPECT_DOUBLE_EQ(result.first.completionUs, 5.0);
    EXPECT_EQ(result.first.encoderRequestIds, (std::vector<uint64_t>{1U}));
    EXPECT_EQ(result.first.decodeRequestIds, (std::vector<uint64_t>{2U}));
    EXPECT_EQ(result.first.releasedVisionBytes, 200U);
    EXPECT_DOUBLE_EQ(result.second.completionUs, 8.0);
    EXPECT_EQ(result.second.prefillRequestIds, (std::vector<uint64_t>{1U}));
    EXPECT_EQ(result.second.decodeRequestIds, (std::vector<uint64_t>{2U}));
    EXPECT_EQ(result.second.releasedVisionBytes, 200U);
}

TEST(PhaseFormationPlannerTest, KeepsDecodeReadyUntilItsLastStepAndThenReleasesKv)
{
    std::vector<PhaseFormationRequestState> requests{
        {7U, PhaseFormationRequestStage::kDecodeReady, 2U, 0U, 4096U},
        {9U, PhaseFormationRequestStage::kDecodeReady, 1U, 0U, 8192U},
    };
    std::vector<PhaseFormationPhysicalCompletion> completions{
        {PhaseGlobalActionKind::kDecode, {7U}, 3.0, 0.2},
        {PhaseGlobalActionKind::kDecode, {9U}, 6.0, 0.4},
    };

    PhaseFormationTwoBoundaryResult const result
        = phaseFormationEvaluateCompletionBoundaries(std::move(requests), std::move(completions));

    ASSERT_TRUE(result.feasible);
    EXPECT_EQ(result.first.decodeRequestIds, (std::vector<uint64_t>{7U, 9U}));
    EXPECT_EQ(result.first.releasedKvBytes, 0U);
    EXPECT_EQ(result.second.decodeRequestIds, (std::vector<uint64_t>{7U}));
    EXPECT_EQ(result.second.releasedKvBytes, 8192U);
}

TEST(PhaseFormationPlannerTest, RejectsACompletionThatViolatesTheRequestDag)
{
    std::vector<PhaseFormationRequestState> requests{
        {1U, PhaseFormationRequestStage::kEncoderReady, 2U, 100U, 0U},
    };
    std::vector<PhaseFormationPhysicalCompletion> completions{
        {PhaseGlobalActionKind::kPrefill, {1U}, 4.0, 0.0},
    };

    EXPECT_FALSE(phaseFormationEvaluateCompletionBoundaries(std::move(requests), std::move(completions)).feasible);
}

} // namespace
} // namespace trt_edgellm::rt
