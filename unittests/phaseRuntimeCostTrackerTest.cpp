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

#include "runtime/phase/cost/phaseRuntimeCostTracker.h"

#include <gtest/gtest.h>

namespace trt_edgellm::rt
{
namespace
{

TEST(PhaseRuntimeCostTrackerTest, MovesFromUnknownThroughWarmingToReady)
{
    PhaseRuntimeCostTrackerConfig config;
    config.actionMinimumSamples = 3U;
    PhaseRuntimeCostTracker tracker(config);
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kDecode, 4, 0, 1, 1, 0};

    EXPECT_EQ(tracker.confidence(key), PhaseRuntimeCostConfidence::kUnknown);
    tracker.observe(key, {8.0F, 3.0F});
    EXPECT_EQ(tracker.confidence(key), PhaseRuntimeCostConfidence::kWarming);
    EXPECT_FALSE(tracker.trustedEstimate(key).has_value());
    tracker.observe(key, {8.0F, 3.1F});
    tracker.observe(key, {8.0F, 2.9F});
    EXPECT_EQ(tracker.confidence(key), PhaseRuntimeCostConfidence::kReady);
    EXPECT_TRUE(tracker.trustedEstimate(key).has_value());
}

TEST(PhaseRuntimeCostTrackerTest, ReplacesOldActionObservationsBySampleCount)
{
    PhaseRuntimeCostTrackerConfig config;
    config.action.windowSize = 3U;
    config.actionMinimumSamples = 1U;
    config.action.coldStartUncertaintyMs = 0.0F;
    PhaseRuntimeCostTracker tracker(config);
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kPrefill, 1, 0, 128, 0, 0};

    tracker.observe(key, {10.0F, 10.0F});
    tracker.observe(key, {10.0F, 10.0F});
    tracker.observe(key, {2.0F, 2.0F});
    tracker.observe(key, {2.0F, 2.0F});

    std::optional<PhaseGlobalCostEstimate> const estimate = tracker.estimate(key);
    ASSERT_TRUE(estimate.has_value());
    EXPECT_EQ(estimate->sampleCount, 3U);
    EXPECT_FLOAT_EQ(estimate->makespanMedianMs, 2.0F);
}

TEST(PhaseRuntimeCostTrackerTest, KeepsDecodeContentionBucketsSeparate)
{
    PhaseRuntimeCostTrackerConfig config;
    config.decodeMinimumSamples = 2U;
    config.decodeWindowSize = 3U;
    PhaseRuntimeCostTracker tracker(config);

    tracker.observeDecode(8, 700, false, false, 4.0F);
    tracker.observeDecode(8, 700, false, false, 5.0F);
    tracker.observeDecode(8, 700, true, false, 9.0F);
    tracker.observeDecode(8, 700, true, false, 10.0F);

    EXPECT_FLOAT_EQ(*tracker.decodeP95(8, 700, false, false), 5.0F);
    EXPECT_FLOAT_EQ(*tracker.decodeP95(8, 700, true, false), 10.0F);
    EXPECT_FALSE(tracker.decodeP95(8, 700, false, true).has_value());
    EXPECT_EQ(tracker.decodeBucketCount(), 2U);
}

TEST(PhaseRuntimeCostTrackerTest, ResetDropsAllProcessLocalMeasurements)
{
    PhaseRuntimeCostTrackerConfig config;
    config.actionMinimumSamples = 1U;
    config.decodeMinimumSamples = 1U;
    PhaseRuntimeCostTracker tracker(config);
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kEncoder, 1, 0, 0, 1, 0};
    tracker.observe(key, {5.0F, 5.0F});
    tracker.observeDecode(1, 128, false, false, 1.0F);

    tracker.reset();

    EXPECT_EQ(tracker.confidence(key), PhaseRuntimeCostConfidence::kUnknown);
    EXPECT_FALSE(tracker.decodeP95(1, 128, false, false).has_value());
    EXPECT_EQ(tracker.decodeBucketCount(), 0U);
}

} // namespace
} // namespace trt_edgellm::rt
