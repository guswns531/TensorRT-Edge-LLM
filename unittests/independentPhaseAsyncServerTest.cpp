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

#include "runtime/scheduling/independentPhaseAsyncServer.h"

#include <gtest/gtest.h>

namespace trt_edgellm::rt
{

TEST(IndependentPhaseAsyncServerTest, AlignsAdmissionToCompleteDecodeCohorts)
{
    EXPECT_EQ(phaseDecodeAlignedAdmissionCapacity(80, 64), 64U);
    EXPECT_EQ(phaseDecodeAlignedAdmissionCapacity(80, 32), 64U);
    EXPECT_EQ(phaseDecodeAlignedAdmissionCapacity(80, 16), 80U);
    EXPECT_EQ(phaseDecodeAlignedAdmissionCapacity(128, 64), 128U);
}

TEST(IndependentPhaseAsyncServerTest, PreservesSubCohortAndInvalidCapacities)
{
    EXPECT_EQ(phaseDecodeAlignedAdmissionCapacity(48, 64), 48U);
    EXPECT_EQ(phaseDecodeAlignedAdmissionCapacity(64, 64), 64U);
    EXPECT_EQ(phaseDecodeAlignedAdmissionCapacity(0, 64), 0U);
    EXPECT_EQ(phaseDecodeAlignedAdmissionCapacity(80, 0), 80U);
}

TEST(IndependentPhaseAsyncServerTest, SynchronizesDecodeCompletionOnlyWithoutProducerCriticalPath)
{
    EXPECT_TRUE(phaseShouldSynchronizeDecodeSampling(true, false, false, 0));
    EXPECT_FALSE(phaseShouldSynchronizeDecodeSampling(false, false, false, 0));
    EXPECT_FALSE(phaseShouldSynchronizeDecodeSampling(true, true, false, 0));
    EXPECT_FALSE(phaseShouldSynchronizeDecodeSampling(true, false, true, 0));
    EXPECT_FALSE(phaseShouldSynchronizeDecodeSampling(true, false, false, 1));
}

TEST(IndependentPhaseAsyncServerTest, ExposesOneCompletePrefillCohortPerIngressTurn)
{
    EXPECT_EQ(phaseServingIngressQuantum(1024, 8), 8U);
    EXPECT_EQ(phaseServingIngressQuantum(4, 8), 4U);
    EXPECT_EQ(phaseServingIngressQuantum(1024, 0), 1U);
    EXPECT_EQ(phaseServingIngressQuantum(0, 8), 0U);
}

TEST(IndependentPhaseAsyncServerTest, DefersOnlyARefillableDecodeTail)
{
    EXPECT_TRUE(shouldDeferDecodeForSamplingRefill(64, 0, 16, 64));
    EXPECT_TRUE(shouldDeferDecodeForSamplingRefill(64, 0, 63, 1));
    EXPECT_FALSE(shouldDeferDecodeForSamplingRefill(0, 0, 16, 64));
    EXPECT_FALSE(shouldDeferDecodeForSamplingRefill(64, 1, 16, 64));
    EXPECT_FALSE(shouldDeferDecodeForSamplingRefill(64, 0, 0, 64));
    EXPECT_FALSE(shouldDeferDecodeForSamplingRefill(64, 0, 64, 64));
    EXPECT_FALSE(shouldDeferDecodeForSamplingRefill(64, 0, 16, 47));
}

TEST(IndependentPhaseAsyncServerTest, PrefillFormationRequiresKnownUpstreamRows)
{
    EXPECT_TRUE(shouldDeferPrefillForMicrobatchFormation(8, 2, 6, 4000.0, 1000.0, 100.0, 250.0));
    EXPECT_FALSE(shouldDeferPrefillForMicrobatchFormation(8, 2, 5, 4000.0, 1000.0, 100.0, 250.0));
    EXPECT_FALSE(shouldDeferPrefillForMicrobatchFormation(0, 2, 6, 4000.0, 1000.0, 100.0, 250.0));
    EXPECT_FALSE(shouldDeferPrefillForMicrobatchFormation(8, 0, 6, 4000.0, 1000.0, 100.0, 250.0));
    EXPECT_FALSE(shouldDeferPrefillForMicrobatchFormation(8, 8, 6, 4000.0, 1000.0, 100.0, 250.0));
    EXPECT_FALSE(shouldDeferPrefillForMicrobatchFormation(8, 2, 0, 4000.0, 1000.0, 100.0, 250.0));
}

TEST(IndependentPhaseAsyncServerTest, PrefillFormationHonorsWindowAndTtftGuard)
{
    EXPECT_TRUE(shouldDeferPrefillForMicrobatchFormation(8, 2, 6, 4000.0, 1000.0, 249.0, 250.0));
    EXPECT_FALSE(shouldDeferPrefillForMicrobatchFormation(8, 2, 6, 4000.0, 1000.0, 250.0, 250.0));
    EXPECT_FALSE(shouldDeferPrefillForMicrobatchFormation(8, 2, 6, 1000.0, 1000.0, 100.0, 250.0));
    EXPECT_FALSE(shouldDeferPrefillForMicrobatchFormation(8, 2, 6, 4000.0, 1000.0, 0.0, 0.0));
}

TEST(IndependentPhaseAsyncServerTest, AdmissionRefillWaitsForOneCompletePrefillCohort)
{
    EXPECT_TRUE(shouldDeferAdmissionForPrefillRefill(8, 32, 1, 63, 1000.0, 10000.0));
    EXPECT_TRUE(shouldDeferAdmissionForPrefillRefill(8, 8, 7, 57, 9999.0, 10000.0));
    EXPECT_FALSE(shouldDeferAdmissionForPrefillRefill(8, 32, 8, 56, 1000.0, 10000.0));
    EXPECT_FALSE(shouldDeferAdmissionForPrefillRefill(8, 7, 1, 63, 1000.0, 10000.0));
    EXPECT_FALSE(shouldDeferAdmissionForPrefillRefill(8, 32, 1, 63, 10000.0, 10000.0));
}

TEST(IndependentPhaseAsyncServerTest, AdmissionRefillNeverBlocksDrainOrDisabledMode)
{
    EXPECT_FALSE(shouldDeferAdmissionForPrefillRefill(0, 32, 1, 63, 1000.0, 10000.0));
    EXPECT_FALSE(shouldDeferAdmissionForPrefillRefill(1, 32, 1, 63, 1000.0, 10000.0));
    EXPECT_FALSE(shouldDeferAdmissionForPrefillRefill(8, 32, 0, 64, 1000.0, 10000.0));
    EXPECT_FALSE(shouldDeferAdmissionForPrefillRefill(8, 32, 1, 0, 1000.0, 10000.0));
    EXPECT_FALSE(shouldDeferAdmissionForPrefillRefill(8, 32, 1, 63, 0.0, 0.0));
}

TEST(IndependentPhaseAsyncServerTest, PrefillFormationCanSelectShortOutputCohorts)
{
    EXPECT_TRUE(phasePrefillFormationSupportsOutputLength(0, 128));
    EXPECT_TRUE(phasePrefillFormationSupportsOutputLength(32, 32));
    EXPECT_FALSE(phasePrefillFormationSupportsOutputLength(32, 33));
}

TEST(IndependentPhaseAsyncServerTest, PrefillFormationCostSelectsHighestMeasuredGain)
{
    std::vector<IndependentPhaseFormationCost> const costs{
        {256, 32, 32, 2.0F, 8, 50.0, 100000.0, 1000.0, 2.6},
        {512, 64, 64, 3.0F, 8, 75.0, 120000.0, 1000.0, 1.1},
        {256, 32, 32, 2.0F, 4, 25.0, 100000.0, 1000.0, 3.0},
    };
    EXPECT_EQ(phasePrefillFormationCostIndex(costs, 128, 16, 8, 1.0F), 2U);
    EXPECT_EQ(phasePrefillFormationCostIndex(costs, 384, 48, 48, 2.5F), 1U);
}

TEST(IndependentPhaseAsyncServerTest, PrefillFormationCostRejectsUnprofiledOrNonPositiveRegions)
{
    std::vector<IndependentPhaseFormationCost> const costs{
        {256, 32, 32, 2.0F, 8, 50.0, 100000.0, 1000.0, 2.6},
        {1024, 128, 64, 4.0F, 8, 50.0, 100000.0, 1000.0, -1.0},
    };
    EXPECT_FALSE(phasePrefillFormationCostIndex(costs, 257, 32, 32, 2.0F).has_value());
    EXPECT_FALSE(phasePrefillFormationCostIndex(costs, 256, 33, 32, 2.0F).has_value());
    EXPECT_FALSE(phasePrefillFormationCostIndex(costs, 256, 32, 33, 2.0F).has_value());
    EXPECT_FALSE(phasePrefillFormationCostIndex(costs, 256, 32, 32, 2.1F).has_value());
}

TEST(IndependentPhaseAsyncServerTest, AdaptiveAdmissionUsesBacklogHysteresis)
{
    EXPECT_FALSE(nextAdaptiveThroughputMode(false, 0, 64, 64, 1));
    EXPECT_TRUE(nextAdaptiveThroughputMode(false, 1, 64, 64, 1));
    EXPECT_TRUE(nextAdaptiveThroughputMode(true, 0, 80, 64, 1));
    EXPECT_TRUE(nextAdaptiveThroughputMode(true, 1, 64, 64, 1));
    EXPECT_FALSE(nextAdaptiveThroughputMode(true, 0, 64, 64, 1));
}

TEST(IndependentPhaseAsyncServerTest, StepwiseAdmissionRequiresSaturatedBacklog)
{
    EXPECT_EQ(nextStepwiseAdmissionLimit(16, 16, 64, 16, 8, 16, 1, 128, 8, 2.0F, 3.3F, 3.0F), 32);
    EXPECT_EQ(nextStepwiseAdmissionLimit(32, 16, 64, 16, 8, 31, 1, 128, 8, 2.0F, 3.3F, 3.0F), 32);
    EXPECT_EQ(nextStepwiseAdmissionLimit(48, 16, 64, 16, 8, 48, 1, 128, 8, 2.0F, 3.3F, 3.0F), 64);
    EXPECT_EQ(nextStepwiseAdmissionLimit(64, 16, 64, 16, 8, 64, 1, 128, 8, 2.0F, 3.3F, 3.0F), 64);
}

TEST(IndependentPhaseAsyncServerTest, StepwiseAdmissionContractsForDecodeOrPagePressure)
{
    EXPECT_EQ(nextStepwiseAdmissionLimit(64, 16, 64, 16, 8, 64, 1, 128, 8, 3.3F, 3.3F, 3.0F), 48);
    EXPECT_EQ(nextStepwiseAdmissionLimit(48, 16, 64, 16, 8, 48, 1, 7, 8, 2.0F, 3.3F, 3.0F), 32);
    EXPECT_EQ(nextStepwiseAdmissionLimit(32, 16, 64, 16, 0, 16, 1, 128, 8, 2.0F, 3.3F, 3.0F), 16);
    EXPECT_EQ(nextStepwiseAdmissionLimit(16, 16, 64, 16, 8, 16, 1, 128, 8, 3.4F, 3.3F, 3.0F), 16);
}

TEST(IndependentPhaseAsyncServerTest, StepwiseAdmissionUsesTpotHysteresisForGrowth)
{
    EXPECT_EQ(nextStepwiseAdmissionLimit(32, 16, 64, 16, 8, 32, 1, 128, 8, 3.1F, 3.3F, 3.0F), 32);
    EXPECT_EQ(nextStepwiseAdmissionLimit(32, 16, 64, 16, 8, 32, 1, 128, 8, 3.0F, 3.3F, 3.0F), 48);
    EXPECT_EQ(nextStepwiseAdmissionLimit(32, 16, 64, 16, 8, 32, 1, 128, 8, 0.0F, 3.3F, 3.0F), 48);
}

TEST(IndependentPhaseAsyncServerTest, PredictiveAdmissionSelectsHighestLimitWithinTpotBudget)
{
    std::vector<IndependentPhaseAdmissionCost> const costs{{16, 23403.0}, {32, 28984.0}, {48, 33871.0}, {64, 36324.0}};
    EXPECT_EQ(phaseAdmissionLimitForTpotBudget(costs, 16, 64, 40000.0), 64);
    EXPECT_EQ(phaseAdmissionLimitForTpotBudget(costs, 16, 64, 34000.0), 48);
    EXPECT_EQ(phaseAdmissionLimitForTpotBudget(costs, 16, 64, 30000.0), 32);
    EXPECT_EQ(phaseAdmissionLimitForTpotBudget(costs, 16, 64, 20000.0), 16);
}

TEST(IndependentPhaseAsyncServerTest, PredictiveAdmissionDisablesWithoutCostsOrBudget)
{
    std::vector<IndependentPhaseAdmissionCost> const costs{{16, 23403.0}, {32, 28984.0}};
    EXPECT_EQ(phaseAdmissionLimitForTpotBudget({}, 16, 64, 30000.0), 64);
    EXPECT_EQ(phaseAdmissionLimitForTpotBudget(costs, 16, 64, 0.0), 64);
    EXPECT_EQ(phaseAdmissionLimitForTpotBudget(costs, 24, 64, 25000.0), 24);
}

TEST(IndependentPhaseAsyncServerTest, PredictiveAdmissionReportsUnsatisfiableBudget)
{
    std::vector<IndependentPhaseAdmissionCost> const costs{{16, 60680.0}, {32, 74090.0}};
    EXPECT_FALSE(phaseAdmissionTpotBudgetSatisfiable(costs, 16, 34000.0));
    EXPECT_TRUE(phaseAdmissionTpotBudgetSatisfiable(costs, 16, 70000.0));
    EXPECT_TRUE(phaseAdmissionTpotBudgetSatisfiable({}, 16, 34000.0));
    EXPECT_TRUE(phaseAdmissionTpotBudgetSatisfiable(costs, 16, 0.0));
}

TEST(IndependentPhaseAsyncServerTest, PredictiveAdmissionSelectsExternalWorkloadProfile)
{
    EXPECT_TRUE(phaseAdmissionUsesExternalProfile(48, 64, 0, 0.5, 0));
    EXPECT_FALSE(phaseAdmissionUsesExternalProfile(16, 64, 0, 0.5, 0));
    EXPECT_TRUE(phaseAdmissionUsesExternalProfile(32, 64, 2048, 0.5, 1024));
    EXPECT_FALSE(phaseAdmissionUsesExternalProfile(32, 64, 512, 0.5, 1024));
    EXPECT_FALSE(phaseAdmissionUsesExternalProfile(0, 0, 2048, 0.0, 1024));
}

TEST(IndependentPhaseAsyncServerTest, PredictiveAdmissionLatchesProfileForOneBusyEpoch)
{
    std::optional<bool> selection;
    selection = phaseAdmissionExternalProfileForEpoch(selection, 4, 16, 512, 0.5, 1024);
    EXPECT_FALSE(selection.has_value());
    selection = phaseAdmissionExternalProfileForEpoch(selection, 4, 16, 2048, 0.5, 1024);
    EXPECT_FALSE(selection.has_value());
    selection = phaseAdmissionExternalProfileForEpoch(selection, 4, 4, 2048, 0.5, 1024);
    ASSERT_TRUE(selection.has_value());
    EXPECT_TRUE(*selection);
    selection = phaseAdmissionExternalProfileForEpoch(selection, 4, 16, 2048, 0.5, 1024);
    ASSERT_TRUE(selection.has_value());
    EXPECT_TRUE(*selection);
    selection = phaseAdmissionExternalProfileForEpoch(selection, 0, 0, 0, 0.5, 1024);
    EXPECT_FALSE(selection.has_value());
    selection = phaseAdmissionExternalProfileForEpoch(selection, 12, 16, 2048, 0.5, 1024);
    ASSERT_TRUE(selection.has_value());
    EXPECT_TRUE(*selection);
}

TEST(IndependentPhaseAsyncServerTest, PredictiveAdmissionUsesExternalFallbackBudget)
{
    EXPECT_DOUBLE_EQ(phaseAdmissionProfileTpotBudget(34000.0, 75000.0, true), 75000.0);
    EXPECT_DOUBLE_EQ(phaseAdmissionProfileTpotBudget(34000.0, 75000.0, false), 34000.0);
    EXPECT_DOUBLE_EQ(phaseAdmissionProfileTpotBudget(34000.0, 0.0, true), 34000.0);
}

TEST(IndependentPhaseAsyncServerTest, ServingWarmupCoversRepresentativeDecodeBuckets)
{
    EXPECT_EQ(phaseServingWarmupBatchSizes(64), (std::vector<int32_t>{8, 16, 32, 48, 64}));
    EXPECT_EQ(phaseServingWarmupBatchSizes(8), (std::vector<int32_t>{1, 2, 4, 6, 8}));
    EXPECT_EQ(phaseServingWarmupBatchSizes(1), (std::vector<int32_t>{1}));
    EXPECT_EQ(phaseServingWarmupBatchSizes(64, {48, 12, 48, 64}), (std::vector<int32_t>{12, 48, 64}));
    EXPECT_THROW(phaseServingWarmupBatchSizes(0), std::runtime_error);
    EXPECT_THROW(phaseServingWarmupBatchSizes(64, {65}), std::runtime_error);
}

TEST(IndependentPhaseAsyncServerTest, IncrementalPageReservationGuaranteesADrainableCohort)
{
    std::vector<IndependentPhasePageReservation> reservations;
    for (uint64_t requestId{}; requestId < 80U; ++requestId)
    {
        reservations.push_back({requestId, 2, 13});
    }
    EXPECT_EQ(phasePageReservationGuaranteedPages(reservations, 8), 248);
    EXPECT_TRUE(phasePageReservationsFit(256, reservations, 8));
    EXPECT_FALSE(phasePageReservationsFit(256, reservations, 9));
}

TEST(IndependentPhaseAsyncServerTest, AdmitsOnlyTheFifoPageReservationPrefixThatFits)
{
    std::vector<IndependentPhasePageReservation> const existing{{10, 2, 8}};
    std::vector<IndependentPhasePageReservation> const candidates{{20, 2, 8}, {21, 2, 8}, {22, 2, 8}};
    EXPECT_EQ(phaseAdmissiblePageReservationPrefix(existing, candidates, 18, 2, 3), 2U);
    EXPECT_EQ(phaseAdmissiblePageReservationPrefix(existing, candidates, 18, 2, 1), 1U);
    EXPECT_EQ(phaseAdmissiblePageReservationPrefix(existing, candidates, 7, 2, 3), 0U);
}

TEST(IndependentPhaseAsyncServerTest, PageGrowthOwnersRemainStickyAndDeterministic)
{
    std::vector<IndependentPhasePageReservation> const reservations{
        {10, 2, 8}, {11, 2, 13}, {12, 2, 10}, {13, 2, 12}, {14, 2, 9}};
    EXPECT_EQ(selectPhasePageGrowthOwners(reservations, {14, 10}, 4), (std::vector<uint64_t>{10, 14, 11, 13}));
    EXPECT_EQ(selectPhasePageGrowthOwners(reservations, {99, 11}, 2), (std::vector<uint64_t>{11, 13}));
}

TEST(IndependentPhaseAsyncServerTest, PageReservationRejectsInvalidAndDuplicateInputs)
{
    EXPECT_THROW(phasePageReservationGuaranteedPages({{0, 3, 2}}, 1), std::runtime_error);
    EXPECT_THROW(phasePageReservationGuaranteedPages({}, 0), std::runtime_error);
    EXPECT_THROW(selectPhasePageGrowthOwners({{0, 1, 2}, {0, 1, 2}}, {}, 1), std::runtime_error);
}

TEST(IndependentPhaseAsyncServerTest, GrowthCohortWaitsOnlyForOwnersThatHaveNotStarted)
{
    EXPECT_TRUE(shouldDeferPhasePageGrowthCohort(13, 0, 12));
    EXPECT_FALSE(shouldDeferPhasePageGrowthCohort(13, 0, 13));
    EXPECT_TRUE(shouldDeferPhasePageGrowthCohort(13, 12, 0));
    EXPECT_FALSE(shouldDeferPhasePageGrowthCohort(13, 12, 1));
    EXPECT_FALSE(shouldDeferPhasePageGrowthCohort(13, 13, 0));
}

} // namespace trt_edgellm::rt
