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

#include "runtime/scheduling/phaseGlobalScheduler.h"

#include <gtest/gtest.h>

namespace trt_edgellm::rt
{
namespace
{

PhaseGlobalActionCandidate candidate(PhaseGlobalActionKind kind, double workUs, double blockingUs, double slackUs)
{
    PhaseGlobalActionCandidate result;
    result.key.kind = kind;
    result.referenceWorkUs = workUs;
    result.predictedBlockingUs = blockingUs;
    result.minimumProtectedSlackUs = slackUs;
    return result;
}

TEST(PhaseGlobalSchedulerTest, RejectsBrokenMechanismInvariants)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate invalid = candidate(PhaseGlobalActionKind::kPrefill, 1000.0, 1000.0, 10000.0);
    invalid.dependencySafe = false;
    PhaseGlobalDecision const decision = scheduler.select({invalid});
    EXPECT_FALSE(decision.selectedIndex.has_value());
    EXPECT_EQ(decision.reason, PhaseGlobalDecisionReason::kNoHardFeasibleCandidate);
}

TEST(PhaseGlobalSchedulerTest, DoesNotCountNearReclaimAsHardCapacity)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate encoder = candidate(PhaseGlobalActionKind::kEncoder, 1000.0, 1000.0, 10000.0);
    encoder.memory = {900U, 200U, 0U, 0U, 500U, 1000U, false};
    PhaseGlobalDecision const decision = scheduler.select({encoder});
    EXPECT_FALSE(decision.selectedIndex.has_value());
}

TEST(PhaseGlobalSchedulerTest, CountsOnlyObservedImmediateReclaim)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate prefill = candidate(PhaseGlobalActionKind::kPrefill, 1000.0, 1000.0, 10000.0);
    prefill.memory = {900U, 200U, 200U, 0U, 0U, 1000U, true};
    PhaseGlobalDecision const decision = scheduler.select({prefill});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(decision.hardPeakManagedBytes, 900U);
}

TEST(PhaseGlobalSchedulerTest, RanksNearReclaimWithoutUsingItForCapacity)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate decode = candidate(PhaseGlobalActionKind::kDecode, 1000.0, 1000.0, 10000.0);
    PhaseGlobalActionCandidate prefill = candidate(PhaseGlobalActionKind::kPrefill, 1000.0, 1000.0, 10000.0);
    prefill.memory.nearReclaimBytes = 4096U;

    PhaseGlobalDecision const decision = scheduler.select({decode, prefill});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
}

TEST(PhaseGlobalSchedulerTest, ProtectsRobustSlackBeforeEfficiency)
{
    PhaseGlobalScheduler scheduler({11U, 100.0});
    PhaseGlobalActionCandidate efficient = candidate(PhaseGlobalActionKind::kDecode, 10000.0, 1000.0, 1050.0);
    PhaseGlobalActionCandidate safe = candidate(PhaseGlobalActionKind::kPrefill, 1000.0, 500.0, 5000.0);
    PhaseGlobalDecision const decision = scheduler.select({efficient, safe});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
}

TEST(PhaseGlobalSchedulerTest, UsesMinimumViolationWhenEveryActionIsLate)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate first = candidate(PhaseGlobalActionKind::kEncoder, 1000.0, 3000.0, 1000.0);
    PhaseGlobalActionCandidate second = candidate(PhaseGlobalActionKind::kDecode, 1000.0, 2000.0, 1000.0);
    PhaseGlobalDecision const decision = scheduler.select({first, second});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_EQ(decision.reason, PhaseGlobalDecisionReason::kMinimumViolation);
}

TEST(PhaseGlobalSchedulerTest, ProtectsUnservedPhaseWithFollowUpCompletion)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate decode = candidate(PhaseGlobalActionKind::kDecode, 1000.0, 1000.0, 1.0);
    decode.predictedMakespanUs = 1000.0;
    decode.protectedCompletions = {{1200.0, 1000.0, 0.0}, {1500.0, 4000.0, 0.0}};
    PhaseGlobalActionCandidate prefill = candidate(PhaseGlobalActionKind::kPrefill, 2000.0, 2000.0, 1.0);
    prefill.predictedMakespanUs = 2000.0;
    prefill.protectedCompletions = {{1200.0, 3000.0, 0.0}, {1500.0, 2000.0, 0.0}};

    PhaseGlobalDecision const decision = scheduler.select({decode, prefill});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_DOUBLE_EQ(decision.predictedViolationUs, 1800.0);
}

TEST(PhaseGlobalSchedulerTest, UsesConstituentReferenceWorkForBatchCompression)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate encoder = candidate(PhaseGlobalActionKind::kEncoder, 92000.0, 90000.0, 1000000.0);
    PhaseGlobalActionCandidate decode = candidate(PhaseGlobalActionKind::kDecode, 396800.0, 8500.0, 1000000.0);
    PhaseGlobalDecision const decision = scheduler.select({encoder, decode});
    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_GT(decision.serviceCompression, 40.0);
}

TEST(PhaseGlobalSchedulerTest, UsesEqualWorkHorizonForWaitComparison)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate now = candidate(PhaseGlobalActionKind::kDecode, 2000.0, 2000.0, 10000.0);
    now.predictedMakespanUs = 2000.0;
    now.predictedHorizonUs = 4100.0;
    now.horizonReferenceWorkUs = 8000.0;
    PhaseGlobalActionCandidate wait = candidate(PhaseGlobalActionKind::kWait, 8000.0, 4300.0, 10000.0);
    wait.predictedMakespanUs = 4300.0;
    wait.predictedHorizonUs = 4300.0;
    wait.horizonReferenceWorkUs = 8000.0;
    wait.concreteWaitEvent = true;
    wait.waitEventId = 7U;

    PhaseGlobalDecision const decision = scheduler.select({now, wait});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 0U);
}

TEST(PhaseGlobalSchedulerTest, RequiresConcreteWaitEvent)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate wait = candidate(PhaseGlobalActionKind::kWait, 10000.0, 1000.0, 10000.0);
    EXPECT_FALSE(scheduler.select({wait}).selectedIndex.has_value());
    wait.concreteWaitEvent = true;
    wait.waitEventId = 7U;
    EXPECT_TRUE(scheduler.select({wait}).selectedIndex.has_value());
}

TEST(PhaseGlobalSchedulerTest, UnknownOverlapRequiresExplicitSafeProbe)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate overlap = candidate(PhaseGlobalActionKind::kPrefillDecode, 2000.0, 1500.0, 10000.0);
    overlap.overlapCostKnown = false;
    EXPECT_FALSE(scheduler.select({overlap}).selectedIndex.has_value());
    overlap.safeProbeEligible = true;
    EXPECT_TRUE(scheduler.select({overlap}).selectedIndex.has_value());
}

TEST(PhaseGlobalSchedulerTest, TreatsEncoderPrefillAsMeasuredOverlap)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate overlap = candidate(PhaseGlobalActionKind::kEncoderPrefill, 4000.0, 2500.0, 10000.0);
    overlap.overlapCostKnown = false;
    EXPECT_FALSE(scheduler.select({overlap}).selectedIndex.has_value());
    overlap.safeProbeEligible = true;
    EXPECT_TRUE(scheduler.select({overlap}).selectedIndex.has_value());
    EXPECT_STREQ(phaseGlobalActionKindName(overlap.key.kind), "encoder_prefill");
}

TEST(PhaseGlobalSchedulerTest, ScoresSafeProbeByOptimisticMakespanWithSerialUncertainty)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate encoder = candidate(PhaseGlobalActionKind::kEncoder, 200000.0, 94000.0, 330000.0);
    encoder.predictedMakespanUs = 94000.0;
    encoder.uncertaintyUs = 2000.0;
    PhaseGlobalActionCandidate probe = candidate(PhaseGlobalActionKind::kEncoderPrefill, 260000.0, 94000.0, 330000.0);
    probe.predictedMakespanUs = 94000.0;
    probe.uncertaintyUs = 60000.0;
    probe.overlapCostKnown = false;
    probe.safeProbeEligible = true;

    PhaseGlobalDecision const decision = scheduler.select({encoder, probe});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
}

TEST(PhaseGlobalSchedulerTest, RejectsUnboundedCandidateExplosion)
{
    PhaseGlobalScheduler scheduler({2U, 0.0});
    std::vector<PhaseGlobalActionCandidate> candidates(3U, candidate(PhaseGlobalActionKind::kDecode, 1.0, 1.0, 10.0));
    EXPECT_THROW(scheduler.select(candidates), std::runtime_error);
}

TEST(PhaseGlobalSchedulerTest, MapsActionsToExplicitOutstandingSets)
{
    PhaseExecutionSet const encoderDecode = phaseExecutionSetForAction(PhaseGlobalActionKind::kEncoderDecode);
    EXPECT_TRUE(phaseExecutionSetContains(encoderDecode, PhaseExecutionSet::kEncoder));
    EXPECT_TRUE(phaseExecutionSetContains(encoderDecode, PhaseExecutionSet::kDecode));
    EXPECT_FALSE(phaseExecutionSetContains(encoderDecode, PhaseExecutionSet::kPrefill));
    EXPECT_TRUE(phaseExecutionSetIsSubset(PhaseExecutionSet::kEncoder, encoderDecode));
    EXPECT_FALSE(phaseExecutionSetIsSubset(PhaseExecutionSet::kPrefill, encoderDecode));
}

TEST(PhaseGlobalSchedulerTest, MapsPrimaryAndSecondaryGraphVariants)
{
    PhaseExecutionVariant const both = phaseExecutionVariant(true, true);
    EXPECT_EQ(both, PhaseExecutionVariant::kBothGraph);
    EXPECT_TRUE(phaseExecutionVariantUsesPrimaryGraph(both));
    EXPECT_TRUE(phaseExecutionVariantUsesSecondaryGraph(both));
    EXPECT_FALSE(phaseExecutionVariantUsesSecondaryGraph(PhaseExecutionVariant::kPrimaryGraph));
    EXPECT_STREQ(phaseExecutionVariantName(both), "both_graph");
}

TEST(PhaseGlobalSchedulerTest, MaterializesStableDispatchLease)
{
    PhaseGlobalActionCandidate action = candidate(PhaseGlobalActionKind::kPrefillDecode, 2000.0, 1500.0, 10000.0);
    action.primaryRequestIds = {3U, 1U};
    action.secondaryRequestIds = {8U, 5U};
    action.requestIds = {3U, 1U, 8U, 5U};
    action.candidateId = phaseGlobalCandidateId(action);

    PhaseGlobalDispatchPlan const plan = phaseGlobalDispatchPlan(7U, 11U, action);
    EXPECT_EQ(plan.planId, 7U);
    EXPECT_EQ(plan.snapshotEpoch, 11U);
    EXPECT_EQ(plan.candidateId, action.candidateId);
    EXPECT_TRUE(plan.launchMatches(PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode));
    EXPECT_FALSE(plan.permits(PhaseExecutionSet::kEncoder));
    EXPECT_EQ(plan.primaryRequestIds, action.primaryRequestIds);
    EXPECT_EQ(plan.secondaryRequestIds, action.secondaryRequestIds);
}

TEST(PhaseGlobalSchedulerTest, CandidateIdentityPreservesCanonicalRowOrder)
{
    PhaseGlobalActionCandidate first = candidate(PhaseGlobalActionKind::kDecode, 1000.0, 1000.0, 10000.0);
    first.primaryRequestIds = {1U, 2U, 3U};
    PhaseGlobalActionCandidate second = first;
    second.primaryRequestIds = {2U, 1U, 3U};
    EXPECT_NE(phaseGlobalCandidateId(first), phaseGlobalCandidateId(second));
    EXPECT_EQ(phaseGlobalCandidateId(first), phaseGlobalCandidateId(first));
}

TEST(PhaseGlobalCostModelTest, LearnsRobustDirectOverlapEligibility)
{
    PhaseGlobalCostModel model({8U, 4U, 2.0F, 0.02F});
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kEncoderDecode, 2, 16, 0, 512, 512};
    EXPECT_FALSE(model.overlapEligible(key));
    for (int32_t sample{}; sample < 4; ++sample)
    {
        model.observe(key, {40.0F, 30.0F});
    }
    std::optional<PhaseGlobalCostEstimate> const estimate = model.estimate(key);
    ASSERT_TRUE(estimate.has_value());
    EXPECT_EQ(estimate->sampleCount, 4U);
    EXPECT_GT(estimate->uncertaintyMs, 0.0F);
    EXPECT_TRUE(model.overlapEligible(key));
}

TEST(PhaseGlobalCostModelTest, RejectsOverlapWithoutRobustGain)
{
    PhaseGlobalCostModel model({8U, 2U, 0.0F, 0.02F});
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kPrefillDecode, 2, 16, 128, 0, 512};
    model.observe(key, {30.0F, 29.8F});
    model.observe(key, {30.0F, 30.2F});
    EXPECT_FALSE(model.overlapEligible(key));
}

TEST(PhaseGlobalCostModelTest, LearnsEncoderPrefillCostByFullShapeKey)
{
    PhaseGlobalCostModel model({8U, 2U, 0.0F, 0.02F});
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kEncoderPrefill, 4, 2, 128, 8, 0};
    model.observe(key, {120.0F, 96.0F});
    model.observe(key, {120.0F, 98.0F});
    EXPECT_TRUE(model.overlapEligible(key));

    PhaseGlobalActionKey const differentPastKV{PhaseGlobalActionKind::kEncoderPrefill, 4, 2, 128, 8, 1};
    EXPECT_FALSE(model.overlapEligible(differentPastKV));
}

TEST(PhaseGlobalCostModelTest, SeparatesEagerAndGraphReplaySamples)
{
    PhaseGlobalCostModel model;
    PhaseGlobalActionKey eager{PhaseGlobalActionKind::kDecode, 16, 0, 1, 4, 0};
    PhaseGlobalActionKey graph = eager;
    graph.executionVariant = PhaseExecutionVariant::kPrimaryGraph;

    model.observe(eager, {8.0F, 7.0F});
    model.observe(graph, {8.0F, 5.0F});

    ASSERT_TRUE(model.estimate(eager).has_value());
    ASSERT_TRUE(model.estimate(graph).has_value());
    EXPECT_FLOAT_EQ(model.estimate(eager)->makespanMedianMs, 7.0F);
    EXPECT_FLOAT_EQ(model.estimate(graph)->makespanMedianMs, 5.0F);
    PhaseGlobalActionCandidate eagerCandidate = candidate(PhaseGlobalActionKind::kDecode, 1.0, 1.0, 10.0);
    eagerCandidate.key = eager;
    PhaseGlobalActionCandidate graphCandidate = eagerCandidate;
    graphCandidate.key = graph;
    EXPECT_NE(phaseGlobalCandidateId(eagerCandidate), phaseGlobalCandidateId(graphCandidate));
}

} // namespace
} // namespace trt_edgellm::rt
