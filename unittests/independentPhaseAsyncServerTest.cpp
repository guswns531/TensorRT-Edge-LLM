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

} // namespace trt_edgellm::rt
