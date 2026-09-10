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

#include "runtime/phase/policy/phaseContextualPdModel.h"

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

TEST(PhaseGlobalSchedulerTest, AuditReportsActualInputsWithoutChangingDecision)
{
    PhaseGlobalScheduler scheduler({11U, 20.0});
    auto safe = candidate(PhaseGlobalActionKind::kDecode, 100.0, 100.0, 1000.0);
    auto late = candidate(PhaseGlobalActionKind::kEncoder, 500.0, 500.0, 100.0);
    auto invalid = safe;
    safe.candidateId = 1U;
    late.candidateId = 2U;
    invalid.candidateId = 3U;
    invalid.contextSafe = false;
    std::vector<PhaseGlobalActionCandidate> inputs{safe, late, invalid};
    std::vector<PhaseGlobalCandidateAudit> audit;
    auto const reference = scheduler.select(inputs);
    auto const measured = scheduler.select(inputs, &audit);
    ASSERT_EQ(audit.size(), inputs.size());
    EXPECT_EQ(measured.selectedIndex, reference.selectedIndex);
    EXPECT_EQ(measured.reason, reference.reason);
    EXPECT_EQ(measured.predictedViolationUs, reference.predictedViolationUs);
    EXPECT_EQ(audit[0].candidateId, 1U);
    EXPECT_TRUE(audit[0].frontierEligible);
    EXPECT_DOUBLE_EQ(audit[0].predictedViolationUs, 0.0);
    EXPECT_DOUBLE_EQ(audit[1].predictedViolationUs, 420.0);
    EXPECT_FALSE(audit[1].frontierEligible);
    EXPECT_FALSE(audit[2].hardFeasible);
    EXPECT_FALSE(audit[2].frontierEligible);
    scheduler.select({}, &audit);
    EXPECT_TRUE(audit.empty());
}

TEST(PhaseGlobalSchedulerTest, AuditUsesProtectedCompletionUncertainty)
{
    PhaseGlobalScheduler scheduler({11U, 20.0});
    auto input = candidate(PhaseGlobalActionKind::kDecode, 100.0, 100.0, 10000.0);
    input.protectedCompletions.push_back({100.0, 150.0, 30.0, PhaseProtectedKind::kDecode});
    std::vector<PhaseGlobalCandidateAudit> audit;
    auto const decision = scheduler.select({input}, &audit);
    ASSERT_EQ(audit.size(), 1U);
    EXPECT_DOUBLE_EQ(audit[0].predictedViolationUs, 100.0);
    EXPECT_DOUBLE_EQ(decision.predictedViolationUs, audit[0].predictedViolationUs);
}

TEST(PhaseGlobalSchedulerTest, AdditionalViolationAuditExcludesExistingLateness)
{
    PhaseGlobalScheduler scheduler({11U, 20.0});
    auto input = candidate(PhaseGlobalActionKind::kDecode, 100.0, 100.0, -1000.0);
    input.key.primaryBatchSize = 8;
    std::vector<PhaseGlobalCandidateAudit> audit;
    scheduler.select({input}, &audit);
    EXPECT_DOUBLE_EQ(audit[0].predictedViolationUs, 1120.0);
    EXPECT_DOUBLE_EQ(audit[0].additionalViolationUs, 120.0);
    EXPECT_DOUBLE_EQ(audit[0].serviceCompression, 1.0);
    EXPECT_EQ(audit[0].primaryBatchSize, 8);
    input.protectedCompletions
        = {{-1000.0, 150.0, 30.0, PhaseProtectedKind::kDecode}, {100.0, 250.0, 0.0, PhaseProtectedKind::kPrefill}};
    auto const plain = scheduler.select({input});
    auto const measured = scheduler.select({input}, &audit);
    EXPECT_EQ(plain.selectedIndex, measured.selectedIndex);
    EXPECT_DOUBLE_EQ(audit[0].predictedViolationUs, 1200.0);
    EXPECT_DOUBLE_EQ(audit[0].additionalViolationUs, 200.0);
}

TEST(PhaseGlobalSchedulerTest, FinalProtectionCanInvalidateSameIdPreview)
{
    PhaseGlobalScheduler scheduler;
    auto preview = candidate(PhaseGlobalActionKind::kDecode, 100.0, 100.0, 1000.0);
    preview.candidateId = 7U;
    auto finalInput = preview;
    finalInput.protectedCompletions.push_back({1000.0, 100.0, 0.0, PhaseProtectedKind::kDecode});
    finalInput.protectedCompletions.push_back({200.0, 800.0, 50.0, PhaseProtectedKind::kEncoder});
    std::vector<PhaseGlobalCandidateAudit> previewAudit;
    std::vector<PhaseGlobalCandidateAudit> finalAudit;
    scheduler.select({preview}, &previewAudit);
    scheduler.select({finalInput}, &finalAudit);
    ASSERT_EQ(previewAudit.size(), 1U);
    ASSERT_EQ(finalAudit.size(), 1U);
    EXPECT_EQ(previewAudit[0].candidateId, finalAudit[0].candidateId);
    EXPECT_DOUBLE_EQ(previewAudit[0].predictedViolationUs, 0.0);
    EXPECT_DOUBLE_EQ(finalAudit[0].predictedViolationUs, 650.0);
}

TEST(PhaseGlobalSchedulerTest, ContextualDecisionCostCanGeneralizeAnUnknownOverlap)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate serial = candidate(PhaseGlobalActionKind::kPrefill, 2000.0, 2000.0, 10000.0);
    PhaseGlobalActionCandidate overlap = candidate(PhaseGlobalActionKind::kPrefillDecode, 4000.0, 4000.0, 10000.0);
    overlap.overlapCostKnown = false;
    overlap.decisionCostKnown = true;
    overlap.decisionMakespanUs = 3000.0;

    PhaseGlobalDecision const decision = scheduler.select({serial, overlap});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
}

TEST(PhaseGlobalSchedulerTest, RestoresNonContextualFallbackWithoutDroppingExactCosts)
{
    PhaseGlobalActionCandidate learned = candidate(PhaseGlobalActionKind::kPrefillDecode, 5000.0, 4000.0, 10000.0);
    learned.decisionCostKnown = true;
    learned.decisionMakespanUs = 3000.0;
    learned.contextualScalarAuthorityApplied = true;

    phaseRestoreNonContextualPolicy(learned);

    EXPECT_FALSE(learned.decisionCostKnown);
    EXPECT_DOUBLE_EQ(learned.decisionMakespanUs, 0.0);
    PhaseGlobalActionCandidate exact = candidate(PhaseGlobalActionKind::kPrefillDecode, 5000.0, 4000.0, 10000.0);
    exact.decisionCostKnown = true;
    exact.decisionMakespanUs = 3500.0;
    phaseRestoreNonContextualPolicy(exact);
    EXPECT_TRUE(exact.decisionCostKnown);
    EXPECT_DOUBLE_EQ(exact.decisionMakespanUs, 3500.0);
}

TEST(PhaseGlobalSchedulerTest, ContextualDecisionCostCannotBypassExactDeadlineProtection)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate safe = candidate(PhaseGlobalActionKind::kPrefill, 2000.0, 2000.0, 2500.0);
    PhaseGlobalActionCandidate overlap = candidate(PhaseGlobalActionKind::kPrefillDecode, 4000.0, 4000.0, 2500.0);
    overlap.overlapCostKnown = false;
    overlap.decisionCostKnown = true;
    overlap.decisionMakespanUs = 1000.0;

    PhaseGlobalDecision const decision = scheduler.select({safe, overlap});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 0U);
}

TEST(PhaseGlobalSchedulerTest, ServiceNormalizationRequiresOneCanonicalRequestFrontier)
{
    PhaseGlobalScheduler scheduler({11U, 0.0, true});
    auto prefill = candidate(PhaseGlobalActionKind::kPrefill, 3000.0, 1000.0, 10000.0);
    auto decode = candidate(PhaseGlobalActionKind::kDecode, 2000.0, 800.0, 10000.0);
    prefill.protectedCompletions = {
        {10000.0, 1000.0, 0.0, PhaseProtectedKind::kPrefill, 7U, 1000.0, PhaseServiceReferenceSource::kRuntimeExact,
            4000.0},
        {10000.0, 1800.0, 0.0, PhaseProtectedKind::kDecode, 9U, 500.0, PhaseServiceReferenceSource::kRuntimeExact,
            1000.0},
    };
    decode.protectedCompletions = {
        {10000.0, 800.0, 0.0, PhaseProtectedKind::kPrefill, 7U, 1000.0, PhaseServiceReferenceSource::kRuntimeExact,
            4000.0},
        {10000.0, 800.0, 0.0, PhaseProtectedKind::kDecode, 9U, 500.0, PhaseServiceReferenceSource::kRuntimeExact,
            1000.0},
    };

    PhaseGlobalDecision const decision = scheduler.select({prefill, decode});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_TRUE(decision.serviceNormalizedAuthorityApplied);
    EXPECT_DOUBLE_EQ(decision.maxNormalizedServiceAge, 4.8);

    decode.protectedCompletions.back().predictedCompletionUs = 2800.0;
    PhaseGlobalDecision const tradeoff = scheduler.select({prefill, decode});
    ASSERT_TRUE(tradeoff.selectedIndex.has_value());
    EXPECT_EQ(*tradeoff.selectedIndex, 0U);
    EXPECT_FALSE(tradeoff.serviceNormalizedAuthorityApplied);

    decode.protectedCompletions.pop_back();
    PhaseGlobalDecision const incomplete = scheduler.select({prefill, decode});
    EXPECT_FALSE(incomplete.serviceNormalizedAuthorityApplied);
}

TEST(PhaseGlobalSchedulerTest, ServiceRecoveryProtectsOverdueNoSloWorkWithinSafeFrontier)
{
    PhaseGlobalSchedulerConfig config;
    config.enableServiceRecovery = true;
    PhaseGlobalScheduler scheduler(config);
    auto prefill = candidate(PhaseGlobalActionKind::kPrefill, 2000.0, 1000.0, 10000.0);
    auto decode = candidate(PhaseGlobalActionKind::kDecode, 4000.0, 1000.0, 10000.0);
    prefill.protectedCompletions = {
        {std::numeric_limits<double>::infinity(), 1000.0, 0.0, PhaseProtectedKind::kPrefill, 7U, 1000.0,
            PhaseServiceReferenceSource::kRuntimeExact, 3000.0},
        {10000.0, 2000.0, 0.0, PhaseProtectedKind::kDecode, 9U, 1000.0, PhaseServiceReferenceSource::kRuntimeExact,
            0.0},
    };
    decode.protectedCompletions = {
        {std::numeric_limits<double>::infinity(), 3000.0, 0.0, PhaseProtectedKind::kPrefill, 7U, 1000.0,
            PhaseServiceReferenceSource::kRuntimeExact, 3000.0},
        {10000.0, 1000.0, 0.0, PhaseProtectedKind::kDecode, 9U, 1000.0, PhaseServiceReferenceSource::kRuntimeExact,
            0.0},
    };

    PhaseGlobalDecision const decision = scheduler.select({prefill, decode});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 0U);
    EXPECT_TRUE(decision.serviceRecoveryApplied);
    EXPECT_EQ(decision.serviceRecoveryCandidates, 1U);
    EXPECT_DOUBLE_EQ(decision.maxNormalizedServiceAge, 4.0);
}

TEST(PhaseGlobalSchedulerTest, ServiceRecoveryKeepsEfficiencyWithinOneQuantumBand)
{
    PhaseGlobalSchedulerConfig config;
    config.enableServiceRecovery = true;
    PhaseGlobalScheduler scheduler(config);
    auto prefill = candidate(PhaseGlobalActionKind::kPrefill, 2000.0, 1000.0, 10000.0);
    auto decode = candidate(PhaseGlobalActionKind::kDecode, 4000.0, 1000.0, 10000.0);
    prefill.protectedCompletions = {
        {std::numeric_limits<double>::infinity(), 1000.0, 0.0, PhaseProtectedKind::kPrefill, 7U, 10000.0,
            PhaseServiceReferenceSource::kRuntimeExact, 10000.0},
    };
    decode.protectedCompletions = {
        {std::numeric_limits<double>::infinity(), 1500.0, 0.0, PhaseProtectedKind::kPrefill, 7U, 10000.0,
            PhaseServiceReferenceSource::kRuntimeExact, 10000.0},
    };

    PhaseGlobalDecision const decision = scheduler.select({prefill, decode});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_FALSE(decision.serviceRecoveryApplied);
    EXPECT_EQ(decision.serviceRecoveryCandidates, 2U);
}

TEST(PhaseGlobalSchedulerTest, ServiceRecoveryUsesConfiguredAgeAndBand)
{
    PhaseGlobalSchedulerConfig config;
    config.enableServiceRecovery = true;
    config.serviceRecoveryAgeQuanta = 2.0;
    config.serviceRecoveryBandQuanta = 0.0;
    PhaseGlobalScheduler scheduler(config);
    auto prefill = candidate(PhaseGlobalActionKind::kPrefill, 2000.0, 1000.0, 10000.0);
    auto decode = candidate(PhaseGlobalActionKind::kDecode, 4000.0, 1000.0, 10000.0);
    prefill.protectedCompletions = {
        {std::numeric_limits<double>::infinity(), 1000.0, 0.0, PhaseProtectedKind::kPrefill, 7U, 1000.0,
            PhaseServiceReferenceSource::kRuntimeExact, 1500.0},
    };
    decode.protectedCompletions = prefill.protectedCompletions;

    PhaseGlobalDecision const beforeThreshold = scheduler.select({prefill, decode});
    EXPECT_FALSE(beforeThreshold.serviceRecoveryApplied);

    prefill.protectedCompletions.front().elapsedServiceUs = 3000.0;
    decode.protectedCompletions.front().elapsedServiceUs = 3000.0;
    decode.protectedCompletions.front().predictedCompletionUs = 2000.0;
    PhaseGlobalDecision const afterThreshold = scheduler.select({prefill, decode});
    EXPECT_TRUE(afterThreshold.serviceRecoveryApplied);
    EXPECT_EQ(afterThreshold.serviceRecoveryCandidates, 1U);
}

TEST(PhaseGlobalSchedulerTest, ServiceRecoveryDoesNotReplaceExplicitDeadlineSafety)
{
    PhaseGlobalSchedulerConfig config;
    config.enableServiceRecovery = true;
    PhaseGlobalScheduler scheduler(config);
    auto prefill = candidate(PhaseGlobalActionKind::kPrefill, 4000.0, 2000.0, 10000.0);
    auto decode = candidate(PhaseGlobalActionKind::kDecode, 2000.0, 1000.0, 10000.0);
    prefill.protectedCompletions = {
        {std::numeric_limits<double>::infinity(), 2000.0, 0.0, PhaseProtectedKind::kPrefill, 7U, 1000.0,
            PhaseServiceReferenceSource::kRuntimeExact, 3000.0},
        {1000.0, 2000.0, 0.0, PhaseProtectedKind::kDecode, 9U, 1000.0, PhaseServiceReferenceSource::kRuntimeExact, 0.0},
    };
    decode.protectedCompletions = {
        {std::numeric_limits<double>::infinity(), 4000.0, 0.0, PhaseProtectedKind::kPrefill, 7U, 1000.0,
            PhaseServiceReferenceSource::kRuntimeExact, 3000.0},
        {1000.0, 1000.0, 0.0, PhaseProtectedKind::kDecode, 9U, 1000.0, PhaseServiceReferenceSource::kRuntimeExact, 0.0},
    };

    PhaseGlobalDecision const decision = scheduler.select({prefill, decode});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_FALSE(decision.serviceRecoveryApplied);
}

TEST(PhaseGlobalSchedulerTest, SelectsBoundedUnknownProbeInsideTheSingleSelector)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate serial = candidate(PhaseGlobalActionKind::kPrefill, 5000.0, 1000.0, 10000.0);
    PhaseGlobalActionCandidate probe = candidate(PhaseGlobalActionKind::kPrefillDecode, 2000.0, 1500.0, 10000.0);
    probe.overlapCostKnown = false;
    probe.safeProbeEligible = true;
    probe.uncertaintyUs = 100.0;

    PhaseGlobalDecision const decision = scheduler.select({serial, probe});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_EQ(decision.reason, PhaseGlobalDecisionReason::kBoundedExploration);
}

TEST(PhaseGlobalSchedulerTest, OverdueNoSloServiceSuppressesUnknownOverlapExploration)
{
    PhaseGlobalSchedulerConfig config;
    config.enableServiceRecovery = true;
    PhaseGlobalScheduler scheduler(config);
    auto decode = candidate(PhaseGlobalActionKind::kDecode, 1000.0, 1000.0, 10000.0);
    decode.protectedCompletions = {
        {std::numeric_limits<double>::infinity(), 1000.0, 0.0, PhaseProtectedKind::kDecode, 9U, 1000.0,
            PhaseServiceReferenceSource::kRuntimeExact, 2000.0},
    };
    auto probe = candidate(PhaseGlobalActionKind::kEncoderPrefill, 5000.0, 500.0, 10000.0);
    probe.safeProbeEligible = true;
    probe.overlapCostKnown = false;
    probe.protectedCompletions = {
        {std::numeric_limits<double>::infinity(), 3000.0, 0.0, PhaseProtectedKind::kDecode, 9U, 1000.0,
            PhaseServiceReferenceSource::kRuntimeExact, 2000.0},
    };

    PhaseGlobalDecision const decision = scheduler.select({decode, probe});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 0U);
    EXPECT_NE(decision.reason, PhaseGlobalDecisionReason::kBoundedExploration);
    EXPECT_TRUE(decision.serviceRecoveryApplied);
}

TEST(PhaseGlobalSchedulerTest, UsesOnlyExplicitlyEligibleProbeWhenEveryActionIsLate)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate serial = candidate(PhaseGlobalActionKind::kPrefill, 5000.0, 1000.0, 3000.0);
    PhaseGlobalActionCandidate probe = candidate(PhaseGlobalActionKind::kPrefillDecode, 2000.0, 2500.0, 3000.0);
    probe.overlapCostKnown = false;
    probe.safeProbeEligible = false;
    probe.uncertaintyUs = 1000.0;

    PhaseGlobalDecision const decision = scheduler.select({serial, probe});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 0U);
}

TEST(PhaseGlobalSchedulerTest, ExploresLateProbeThatPassedTheCandidateRecoveryGuard)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate serial = candidate(PhaseGlobalActionKind::kDecode, 5000.0, 4000.0, 3000.0);
    PhaseGlobalActionCandidate probe = candidate(PhaseGlobalActionKind::kPrefillDecode, 6000.0, 4000.0, 3000.0);
    probe.overlapCostKnown = false;
    probe.safeProbeEligible = true;
    probe.uncertaintyUs = 500.0;

    PhaseGlobalDecision const decision = scheduler.select({serial, probe});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_EQ(decision.reason, PhaseGlobalDecisionReason::kBoundedExploration);
}

TEST(PhaseContextualPdModelTest, MapsConservativeAdvantageToBoundedDecisionMakespan)
{
    EXPECT_DOUBLE_EQ(phaseContextualDecisionMakespanUs(1000.0, 0.25), 750.0);
    EXPECT_NEAR(phaseContextualDecisionMakespanUs(1000.0, 2.0), 50.0, 1.0e-9);
    EXPECT_DOUBLE_EQ(phaseContextualDecisionMakespanUs(1000.0, -2.0), 2000.0);
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

TEST(PhaseGlobalSchedulerTest, PreservesMostConstrainedSlackWithinSafeCandidates)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate now = candidate(PhaseGlobalActionKind::kDecode, 8000.0, 2000.0, 10000.0);
    now.predictedMakespanUs = 2000.0;
    now.predictedHorizonUs = 5000.0;
    now.horizonReferenceWorkUs = 8000.0;
    now.protectedCompletions = {{10000.0, 2000.0, 0.0, PhaseProtectedKind::kDecode}};

    PhaseGlobalActionCandidate wait = candidate(PhaseGlobalActionKind::kWait, 8000.0, 4000.0, 10000.0);
    wait.predictedMakespanUs = 4000.0;
    wait.predictedHorizonUs = 5000.0;
    wait.horizonReferenceWorkUs = 8000.0;
    wait.concreteWaitEvent = true;
    wait.waitEventId = 7U;
    wait.protectedCompletions = {{10000.0, 4000.0, 0.0, PhaseProtectedKind::kDecode}};

    PhaseGlobalDecision const decision = scheduler.select({now, wait});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 0U);
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

TEST(PhaseGlobalSchedulerTest, FormationHorizonCanRejectMyopicOverlap)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate serial = candidate(PhaseGlobalActionKind::kPrefill, 16000.0, 10000.0, 100000.0);
    serial.predictedMakespanUs = 10000.0;
    PhaseGlobalActionCandidate overlap = candidate(PhaseGlobalActionKind::kEncoderPrefill, 16000.0, 8000.0, 100000.0);
    overlap.predictedMakespanUs = 8000.0;

    PhaseGlobalDecision const myopic = scheduler.select({serial, overlap});
    ASSERT_TRUE(myopic.selectedIndex.has_value());
    EXPECT_EQ(*myopic.selectedIndex, 1U);

    // Both alternatives cover identical current plus successor work. Starting
    // the partial encoder cohort in the overlap alternative requires one extra
    // successor launch, while P-first preserves the larger encoder cohort.
    serial.predictedHorizonUs = 18000.0;
    serial.horizonReferenceWorkUs = 24000.0;
    overlap.predictedHorizonUs = 21000.0;
    overlap.horizonReferenceWorkUs = 24000.0;
    PhaseGlobalDecision const formationAware = scheduler.select({serial, overlap});

    ASSERT_TRUE(formationAware.selectedIndex.has_value());
    EXPECT_EQ(*formationAware.selectedIndex, 0U);
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
    EXPECT_TRUE(plan.incrementalAction.legal());
    EXPECT_EQ(plan.incrementalAction.key.direction, PhaseUnifiedActionDirection::kPrefillToDecode);
    EXPECT_EQ(plan.incrementalAction.key.startSkew, PhaseStartSkewBucket::kImmediate);
    EXPECT_EQ(plan.primaryRequestIds, action.primaryRequestIds);
    EXPECT_EQ(plan.secondaryRequestIds, action.secondaryRequestIds);
}

TEST(PhaseGlobalSchedulerTest, ComputesRobustResidualWithoutChangingRows)
{
    PhaseGlobalActionCandidate action = candidate(PhaseGlobalActionKind::kDecode, 8000.0, 6000.0, 10000.0);
    action.predictedMakespanUs = 6000.0;
    action.uncertaintyUs = 1000.0;
    action.primaryRequestIds = {3U, 1U};
    action.primaryStableSlotIds = {4, 2};
    action.protectedCompletions = {{9000.0, 8000.0, 1000.0}};
    phaseGlobalFinalizeCandidate(action);

    PhaseGlobalActionCandidate const residual = phaseGlobalResidualCandidate(action, 2500.0);

    EXPECT_DOUBLE_EQ(residual.predictedMakespanUs, 3500.0);
    EXPECT_DOUBLE_EQ(residual.uncertaintyUs, 1000.0);
    EXPECT_DOUBLE_EQ(residual.referenceWorkUs, 8000.0 * 4500.0 / 7000.0);
    ASSERT_EQ(residual.protectedCompletions.size(), 1U);
    EXPECT_DOUBLE_EQ(residual.protectedCompletions.front().predictedCompletionUs, 5500.0);
    EXPECT_EQ(residual.primaryRequestIds, action.primaryRequestIds);
    EXPECT_EQ(residual.primaryStableSlotIds, action.primaryStableSlotIds);
}

TEST(PhaseGlobalSchedulerTest, UpgradesOnlySinglePdLeaseToMatchingEncoderOverlap)
{
    PhaseGlobalActionCandidate decode = candidate(PhaseGlobalActionKind::kDecode, 8000.0, 6000.0, 10000.0);
    decode.primaryRequestIds = {9U, 5U};
    decode.primaryStableSlotIds = {3, 1};
    phaseGlobalFinalizeCandidate(decode);
    PhaseGlobalDispatchPlan const active = phaseGlobalDispatchPlan(4U, 7U, decode);

    PhaseGlobalActionCandidate overlap = candidate(PhaseGlobalActionKind::kEncoderDecode, 18000.0, 9000.0, 10000.0);
    overlap.primaryRequestIds = {20U};
    overlap.secondaryRequestIds = decode.primaryRequestIds;
    overlap.secondaryStableSlotIds = decode.primaryStableSlotIds;
    phaseGlobalFinalizeCandidate(overlap);

    std::optional<PhaseGlobalDispatchPlan> const upgraded = phaseGlobalAugmentedDispatchPlan(5U, 8U, active, overlap);
    ASSERT_TRUE(upgraded.has_value());
    EXPECT_EQ(upgraded->action, PhaseGlobalActionKind::kEncoderDecode);
    EXPECT_TRUE(upgraded->launchMatches(PhaseExecutionSet::kEncoder | PhaseExecutionSet::kDecode));
    EXPECT_FALSE(upgraded->permits(PhaseExecutionSet::kPrefill));
    EXPECT_TRUE(upgraded->incrementalAction.legal());
    EXPECT_EQ(upgraded->incrementalAction.key.direction, PhaseUnifiedActionDirection::kDecodeToEncoder);

    overlap.key.kind = PhaseGlobalActionKind::kEncoderPrefill;
    phaseGlobalFinalizeCandidate(overlap);
    EXPECT_FALSE(phaseGlobalAugmentedDispatchPlan(6U, 9U, active, overlap).has_value());
}

TEST(PhaseGlobalSchedulerTest, RejectsTriplePhaseLeaseAugmentation)
{
    PhaseGlobalActionCandidate pd = candidate(PhaseGlobalActionKind::kPrefillDecode, 8000.0, 6000.0, 10000.0);
    pd.primaryRequestIds = {9U};
    pd.secondaryRequestIds = {5U};
    phaseGlobalFinalizeCandidate(pd);
    PhaseGlobalDispatchPlan const active = phaseGlobalDispatchPlan(4U, 7U, pd);

    PhaseGlobalActionCandidate overlap = candidate(PhaseGlobalActionKind::kEncoderDecode, 18000.0, 9000.0, 10000.0);
    overlap.primaryRequestIds = {20U};
    overlap.secondaryRequestIds = {5U};
    phaseGlobalFinalizeCandidate(overlap);
    EXPECT_FALSE(phaseGlobalAugmentedDispatchPlan(5U, 8U, active, overlap).has_value());
}

TEST(PhaseGlobalSchedulerTest, UpgradesSinglePdLeaseToMatchingPrefillDecodeOverlap)
{
    PhaseGlobalActionCandidate decode = candidate(PhaseGlobalActionKind::kDecode, 8000.0, 6000.0, 10000.0);
    decode.primaryRequestIds = {9U, 5U};
    decode.primaryStableSlotIds = {3, 1};
    phaseGlobalFinalizeCandidate(decode);
    PhaseGlobalDispatchPlan const active = phaseGlobalDispatchPlan(4U, 7U, decode);

    PhaseGlobalActionCandidate overlap = candidate(PhaseGlobalActionKind::kPrefillDecode, 18000.0, 9000.0, 10000.0);
    overlap.primaryRequestIds = {20U};
    overlap.primaryStableSlotIds = {8};
    overlap.secondaryRequestIds = decode.primaryRequestIds;
    overlap.secondaryStableSlotIds = decode.primaryStableSlotIds;
    phaseGlobalFinalizeCandidate(overlap);

    std::optional<PhaseGlobalDispatchPlan> const upgraded = phaseGlobalAugmentedDispatchPlan(5U, 8U, active, overlap);
    ASSERT_TRUE(upgraded.has_value());
    EXPECT_EQ(upgraded->action, PhaseGlobalActionKind::kPrefillDecode);
    EXPECT_TRUE(upgraded->launchMatches(PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode));
    overlap.secondaryRequestIds = {7U};
    overlap.secondaryStableSlotIds = {1};
    phaseGlobalFinalizeCandidate(overlap);
    EXPECT_FALSE(phaseGlobalAugmentedDispatchPlan(6U, 9U, active, overlap).has_value());
}

TEST(PhaseGlobalSchedulerTest, SelectsKnownResidualAugmentationForFirstTokenDeadline)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate continuation = candidate(PhaseGlobalActionKind::kDecode, 4000.0, 4000.0, 10000.0);
    continuation.predictedMakespanUs = 4000.0;
    continuation.protectedCompletions = {{5000.0, 11000.0, 0.0, PhaseProtectedKind::kPrefill}};
    PhaseGlobalActionCandidate augmentation
        = candidate(PhaseGlobalActionKind::kEncoderDecode, 14000.0, 7000.0, 10000.0);
    augmentation.key.residualAugmentation = true;
    augmentation.predictedMakespanUs = 7000.0;
    augmentation.protectedCompletions = {{5000.0, 7000.0, 0.0, PhaseProtectedKind::kPrefill}};

    PhaseGlobalDecision const decision = scheduler.select({continuation, augmentation});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_EQ(decision.reason, PhaseGlobalDecisionReason::kAllLateEfficiencyRecovery);
}

TEST(PhaseGlobalSchedulerTest, RecoversEfficiencyWhenEveryActionMissesTheSameProtectedPhase)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate serial = candidate(PhaseGlobalActionKind::kPrefill, 20000.0, 20000.0, 5000.0);
    serial.predictedHorizonUs = 20000.0;
    serial.horizonReferenceWorkUs = 20000.0;
    serial.protectedCompletions = {{5000.0, 10000.0, 0.0, PhaseProtectedKind::kPrefill}};
    PhaseGlobalActionCandidate overlap = candidate(PhaseGlobalActionKind::kPrefillDecode, 20000.0, 10000.0, 5000.0);
    overlap.overlapCostKnown = true;
    overlap.overlapCostProfitable = true;
    overlap.protectedCompletions = {{5000.0, 12000.0, 0.0, PhaseProtectedKind::kPrefill}};

    PhaseGlobalDecision const decision = scheduler.select({serial, overlap});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 1U);
    EXPECT_EQ(decision.reason, PhaseGlobalDecisionReason::kAllLateEfficiencyRecovery);
}

TEST(PhaseGlobalSchedulerTest, SelectsMeasuredUnprofitableResidualForSmallerDeadlineViolation)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate continuation = candidate(PhaseGlobalActionKind::kDecode, 4000.0, 4000.0, 10000.0);
    continuation.predictedMakespanUs = 4000.0;
    continuation.protectedCompletions = {{5000.0, 11000.0, 0.0, PhaseProtectedKind::kPrefill}};
    PhaseGlobalActionCandidate augmentation = candidate(PhaseGlobalActionKind::kPrefillDecode, 6000.0, 7000.0, 10000.0);
    augmentation.key.residualAugmentation = true;
    augmentation.key.residualAnchor = PhaseGlobalResidualAnchor::kDecode;
    augmentation.overlapCostKnown = true;
    augmentation.overlapCostProfitable = false;
    augmentation.predictedMakespanUs = 7000.0;
    augmentation.protectedCompletions = {{5000.0, 7000.0, 0.0, PhaseProtectedKind::kPrefill}};

    PhaseGlobalDecision const decision = scheduler.select({continuation, augmentation});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 0U);
    EXPECT_EQ(decision.reason, PhaseGlobalDecisionReason::kAllLateEfficiencyRecovery);
}

TEST(PhaseGlobalSchedulerTest, KeepsActivePhaseWhenResidualOverlapCostIsUnknown)
{
    PhaseGlobalScheduler scheduler;
    PhaseGlobalActionCandidate continuation = candidate(PhaseGlobalActionKind::kPrefill, 4000.0, 4000.0, 10000.0);
    PhaseGlobalActionCandidate augmentation
        = candidate(PhaseGlobalActionKind::kEncoderPrefill, 14000.0, 7000.0, 10000.0);
    augmentation.key.residualAugmentation = true;
    augmentation.overlapCostKnown = false;

    PhaseGlobalDecision const decision = scheduler.select({continuation, augmentation});

    ASSERT_TRUE(decision.selectedIndex.has_value());
    EXPECT_EQ(*decision.selectedIndex, 0U);
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

TEST(PhaseGlobalSchedulerTest, CandidateIdentityChangesOnlyForConcreteResidualAnchor)
{
    PhaseGlobalActionCandidate ordinary = candidate(PhaseGlobalActionKind::kDecode, 1000.0, 1000.0, 10000.0);
    PhaseGlobalActionCandidate implicitNone = ordinary;
    implicitNone.key.residualAnchor = PhaseGlobalResidualAnchor::kNone;
    EXPECT_EQ(phaseGlobalCandidateId(ordinary), phaseGlobalCandidateId(implicitNone));

    PhaseGlobalActionCandidate residualPrefill = ordinary;
    residualPrefill.key.residualAnchor = PhaseGlobalResidualAnchor::kPrefill;
    PhaseGlobalActionCandidate residualDecode = ordinary;
    residualDecode.key.residualAnchor = PhaseGlobalResidualAnchor::kDecode;
    EXPECT_NE(phaseGlobalCandidateId(ordinary), phaseGlobalCandidateId(residualPrefill));
    EXPECT_NE(phaseGlobalCandidateId(residualPrefill), phaseGlobalCandidateId(residualDecode));
}

TEST(PhaseGlobalCostModelTest, RecordsRobustDirectOverlapEligibility)
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

TEST(PhaseGlobalCostModelTest, SeparatesResidualAnchorDirections)
{
    PhaseGlobalCostModel model({8U, 1U, 0.0F, 0.0F});
    PhaseGlobalActionKey prefillAnchor{PhaseGlobalActionKind::kPrefillDecode, 2, 16, 128, 1, 2};
    prefillAnchor.residualAugmentation = true;
    prefillAnchor.residualAnchor = PhaseGlobalResidualAnchor::kPrefill;
    PhaseGlobalActionKey decodeAnchor = prefillAnchor;
    decodeAnchor.residualAnchor = PhaseGlobalResidualAnchor::kDecode;

    model.observe(prefillAnchor, {10.0F, 8.0F});

    EXPECT_TRUE(model.estimate(prefillAnchor).has_value());
    EXPECT_FALSE(model.estimate(decodeAnchor).has_value());
    PhaseGlobalActionCandidate prefillCandidate;
    prefillCandidate.key = prefillAnchor;
    PhaseGlobalActionCandidate decodeCandidate;
    decodeCandidate.key = decodeAnchor;
    EXPECT_NE(phaseGlobalCandidateId(prefillCandidate), phaseGlobalCandidateId(decodeCandidate));
}

TEST(PhaseGlobalCostModelTest, CopyOnWritePreservesPreviewIsolation)
{
    PhaseGlobalCostModel original;
    PhaseGlobalActionKey const prefill{PhaseGlobalActionKind::kPrefill, 2, 0, 128, 0, 0};
    original.observe(prefill, {2.0F, 1.0F});

    PhaseGlobalCostModel preview = original;
    PhaseGlobalActionKey const decode{PhaseGlobalActionKind::kDecode, 4, 0, 1, 2, 0};
    preview.observe(decode, {4.0F, 2.0F});

    EXPECT_TRUE(original.estimate(prefill).has_value());
    EXPECT_FALSE(original.estimate(decode).has_value());
    EXPECT_TRUE(preview.estimate(prefill).has_value());
    EXPECT_TRUE(preview.estimate(decode).has_value());

    preview.reset();
    EXPECT_FALSE(preview.estimate(prefill).has_value());
    EXPECT_TRUE(original.estimate(prefill).has_value());
}

TEST(PhaseGlobalCostModelTest, RejectsOverlapWithoutRobustGain)
{
    PhaseGlobalCostModel model({8U, 2U, 0.0F, 0.02F});
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kPrefillDecode, 2, 16, 128, 0, 512};
    model.observe(key, {30.0F, 29.8F});
    model.observe(key, {30.0F, 30.2F});
    EXPECT_FALSE(model.overlapEligible(key));
    PhaseGlobalOverlapCostDiagnostic const diagnostic = model.overlapDiagnostic(key);
    EXPECT_EQ(diagnostic.status, PhaseGlobalOverlapCostStatus::kUnprofitable);
    EXPECT_EQ(diagnostic.sampleCount, 2U);
    EXPECT_LT(diagnostic.robustCompression, 1.02F);
}

TEST(PhaseGlobalCostModelTest, DistinguishesUnknownFromInsufficientOverlapCost)
{
    PhaseGlobalCostModel model({8U, 2U, 0.0F, 0.02F});
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kEncoderPrefill, 2, 2, 128, 4, 0};
    EXPECT_EQ(model.overlapDiagnostic(key).status, PhaseGlobalOverlapCostStatus::kNoSamples);
    model.observe(key, {20.0F, 15.0F});
    PhaseGlobalOverlapCostDiagnostic const diagnostic = model.overlapDiagnostic(key);
    EXPECT_EQ(diagnostic.status, PhaseGlobalOverlapCostStatus::kInsufficientSamples);
    EXPECT_EQ(diagnostic.sampleCount, 1U);
    EXPECT_GT(diagnostic.robustCompression, 1.0F);
}

TEST(PhaseGlobalCostModelTest, RecordsEncoderPrefillCostByFullShapeKey)
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

TEST(PhaseGlobalCostModelTest, InterpolatesOnlyBetweenCompatibleObservedBatchSizes)
{
    PhaseGlobalCostModel model({8U, 2U, 0.0F, 0.02F});
    PhaseGlobalActionKey lower{PhaseGlobalActionKind::kPrefill, 1, 0, 128, 0, 0};
    PhaseGlobalActionKey upper = lower;
    upper.primaryBatchSize = 3;
    model.observe(lower, {10.0F, 10.0F});
    model.observe(upper, {24.0F, 12.0F});

    PhaseGlobalActionKey middle = lower;
    middle.primaryBatchSize = 2;
    std::optional<PhaseGlobalCostEstimate> const estimate = model.estimateInterpolatedPrimaryBatch(middle);

    ASSERT_TRUE(estimate.has_value());
    EXPECT_FLOAT_EQ(estimate->referenceWorkMedianMs, 17.0F);
    EXPECT_FLOAT_EQ(estimate->makespanMedianMs, 11.0F);
    EXPECT_FLOAT_EQ(estimate->uncertaintyMs, 0.5F);
    EXPECT_FLOAT_EQ(estimate->makespanP95Ms, 11.5F);

    PhaseGlobalActionKey outside = upper;
    outside.primaryBatchSize = 4;
    EXPECT_FALSE(model.estimateInterpolatedPrimaryBatch(outside).has_value());
    PhaseGlobalActionKey differentChunk = middle;
    differentChunk.chunkLength = 64;
    EXPECT_FALSE(model.estimateInterpolatedPrimaryBatch(differentChunk).has_value());
}

TEST(PhaseGlobalCostModelTest, UsesSmallestObservedContextThatCoversDecodeRequest)
{
    PhaseGlobalCostModel model({8U, 2U, 0.0F, 0.02F});
    PhaseGlobalActionKey largerContext{PhaseGlobalActionKind::kDecode, 4, 0, 1, 4, 0};
    PhaseGlobalActionKey largestContext = largerContext;
    largestContext.primaryContextBucket = 8;
    model.observe(largerContext, {8.0F, 3.0F});
    model.observe(largestContext, {12.0F, 5.0F});

    PhaseGlobalActionKey requested = largerContext;
    requested.primaryContextBucket = 2;
    std::optional<PhaseGlobalCostEstimate> const estimate = model.estimatePrimaryBatchCoveringContext(requested);

    ASSERT_TRUE(estimate.has_value());
    EXPECT_FLOAT_EQ(estimate->makespanMedianMs, 3.0F);
    requested.primaryContextBucket = 5;
    ASSERT_TRUE(model.estimatePrimaryBatchCoveringContext(requested).has_value());
    EXPECT_FLOAT_EQ(model.estimatePrimaryBatchCoveringContext(requested)->makespanMedianMs, 5.0F);
    requested.primaryContextBucket = 9;
    EXPECT_FALSE(model.estimatePrimaryBatchCoveringContext(requested).has_value());
}

TEST(PhaseGlobalCostModelTest, PreservesMeasuredLaunchCostForRaggedPrefill)
{
    PhaseGlobalCostModel model({8U, 1U, 0.0F, 0.02F});
    PhaseGlobalActionKey observed{PhaseGlobalActionKind::kPrefill, 1, 0, 64, 1, 0};
    observed.primaryWorkClass = 1;
    model.observe(observed, {10.0F, 10.0F});

    PhaseGlobalActionKey requested = observed;
    requested.chunkLength = 3;
    requested.primaryContextBucket = 0;
    std::optional<PhaseGlobalCostEstimate> const estimate = model.estimateCoveringPrimary(requested);

    ASSERT_TRUE(estimate.has_value());
    EXPECT_FLOAT_EQ(estimate->makespanMedianMs, 10.0F);
    PhaseGlobalActionKey larger = requested;
    larger.chunkLength = 65;
    EXPECT_FALSE(model.estimateCoveringPrimary(larger).has_value());
    PhaseGlobalActionKey wrongClass = requested;
    wrongClass.primaryWorkClass = 2;
    EXPECT_FALSE(model.estimateCoveringPrimary(wrongClass).has_value());
}

TEST(PhaseGlobalCostModelTest, ConservativelyMergesIncomparablePrimaryCovers)
{
    PhaseGlobalCostModel model({8U, 1U, 0.0F, 0.02F});
    PhaseGlobalActionKey requested{PhaseGlobalActionKind::kPrefill, 1, 0, 32, 0, 0};
    PhaseGlobalActionKey moreBatch = requested;
    moreBatch.primaryBatchSize = 8;
    PhaseGlobalActionKey moreTokens = requested;
    moreTokens.chunkLength = 128;
    model.observe(moreBatch, {12.0F, 12.0F});
    model.observe(moreTokens, {10.0F, 10.0F});

    std::optional<PhaseGlobalCostEstimate> const estimate = model.estimateCoveringPrimary(requested);

    ASSERT_TRUE(estimate.has_value());
    EXPECT_FLOAT_EQ(estimate->makespanMedianMs, 12.0F);
}

TEST(PhaseGlobalCostModelTest, LearnsGeometryIndependentPrimaryLaunchFloor)
{
    PhaseGlobalCostModel model({8U, 1U, 0.0F, 0.02F});
    PhaseGlobalActionKey initial{PhaseGlobalActionKind::kPrefill, 1, 0, 33, 0, 0};
    initial.primaryWorkClass = 1;
    PhaseGlobalActionKey large = initial;
    large.primaryBatchSize = 8;
    large.chunkLength = 128;
    model.observe(initial, {10.0F, 10.0F});
    model.observe(large, {20.0F, 20.0F});

    PhaseGlobalActionKey continuation = initial;
    continuation.primaryBatchSize = 2;
    continuation.chunkLength = 3;
    continuation.primaryContextBucket = 4;
    std::optional<PhaseGlobalCostEstimate> const floor = model.estimatePrimaryLaunchFloor(continuation);

    ASSERT_TRUE(floor.has_value());
    EXPECT_FLOAT_EQ(floor->makespanMedianMs, 10.0F);
    continuation.executionVariant = PhaseExecutionVariant::kPrimaryGraph;
    EXPECT_FALSE(model.estimatePrimaryLaunchFloor(continuation).has_value());
}

TEST(PhaseGlobalCostModelTest, ConservativelyMergesNearestCoveringOverlapShapes)
{
    PhaseGlobalCostModel model({8U, 1U, 0.0F, 0.02F});
    PhaseGlobalActionKey requested{PhaseGlobalActionKind::kPrefillDecode, 2, 16, 64, 1, 2};
    requested.residualAugmentation = true;
    requested.residualAnchor = PhaseGlobalResidualAnchor::kPrefill;
    EXPECT_FALSE(model.estimateCoveringOverlap(requested).has_value());

    PhaseGlobalActionKey moreDecode = requested;
    moreDecode.primaryBatchSize = 4;
    moreDecode.secondaryBatchSize = 64;
    moreDecode.chunkLength = 128;
    moreDecode.primaryContextBucket = 2;
    moreDecode.secondaryContextBucket = 4;
    PhaseGlobalActionKey morePrefill = requested;
    morePrefill.primaryBatchSize = 8;
    morePrefill.secondaryBatchSize = 32;
    morePrefill.chunkLength = 128;
    morePrefill.primaryContextBucket = 4;
    morePrefill.secondaryContextBucket = 4;
    model.observe(moreDecode, {30.0F, 12.0F});
    model.observe(morePrefill, {28.0F, 15.0F});

    std::optional<PhaseGlobalCostEstimate> const estimate = model.estimateCoveringOverlap(requested);
    ASSERT_TRUE(estimate.has_value());
    EXPECT_FLOAT_EQ(estimate->referenceWorkMedianMs, 28.0F);
    EXPECT_FLOAT_EQ(estimate->makespanMedianMs, 15.0F);

    PhaseGlobalActionKey wrongAnchor = requested;
    wrongAnchor.residualAnchor = PhaseGlobalResidualAnchor::kDecode;
    EXPECT_FALSE(model.estimateCoveringOverlap(wrongAnchor).has_value());
    PhaseGlobalActionKey largerThanEveryObservation = requested;
    largerThanEveryObservation.secondaryBatchSize = 128;
    EXPECT_FALSE(model.estimateCoveringOverlap(largerThanEveryObservation).has_value());
}

TEST(PhaseGlobalCostModelTest, SharesOverlapSamplesWithinConservativeShapeBuckets)
{
    PhaseGlobalCostModel model({8U, 2U, 0.0F, 0.02F});
    PhaseGlobalActionKey const first{PhaseGlobalActionKind::kPrefillDecode, 3, 5, 90, 0, 1};
    PhaseGlobalActionKey const second{PhaseGlobalActionKind::kPrefillDecode, 4, 8, 128, 0, 1};
    model.observe(first, {20.0F, 15.0F});
    model.observe(second, {20.0F, 16.0F});

    ASSERT_TRUE(model.estimate(first).has_value());
    EXPECT_EQ(model.estimate(first)->sampleCount, 2U);
    EXPECT_EQ(model.estimate(second)->sampleCount, 2U);
    EXPECT_TRUE(model.overlapEligible(first));
    EXPECT_EQ(phaseGlobalCanonicalOverlapCostKey(first), phaseGlobalCanonicalOverlapCostKey(second));
}

TEST(PhaseGlobalCostModelTest, SharesOverlapSamplesWithinConservativeContextBuckets)
{
    PhaseGlobalCostModel model({8U, 2U, 0.0F, 0.02F});
    PhaseGlobalActionKey const first{PhaseGlobalActionKind::kEncoderDecode, 2, 7, 0, 3, 5};
    PhaseGlobalActionKey const second{PhaseGlobalActionKind::kEncoderDecode, 2, 7, 0, 4, 8};
    model.observe(first, {24.0F, 18.0F});
    model.observe(second, {24.0F, 17.0F});

    EXPECT_EQ(phaseGlobalCanonicalOverlapCostKey(first), phaseGlobalCanonicalOverlapCostKey(second));
    ASSERT_TRUE(model.estimate(first).has_value());
    EXPECT_EQ(model.estimate(second)->sampleCount, 2U);
    EXPECT_TRUE(model.overlapEligible(second));
}

} // namespace
} // namespace trt_edgellm::rt
