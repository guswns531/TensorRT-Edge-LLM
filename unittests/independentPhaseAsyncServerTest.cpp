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

TEST(IndependentPhaseAsyncServerTest, AdaptiveAdmissionUsesBacklogHysteresis)
{
    EXPECT_FALSE(nextAdaptiveThroughputMode(false, 0, 64, 64, 1));
    EXPECT_TRUE(nextAdaptiveThroughputMode(false, 1, 64, 64, 1));
    EXPECT_TRUE(nextAdaptiveThroughputMode(true, 0, 80, 64, 1));
    EXPECT_TRUE(nextAdaptiveThroughputMode(true, 1, 64, 64, 1));
    EXPECT_FALSE(nextAdaptiveThroughputMode(true, 0, 64, 64, 1));
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
