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

#include "runtime/scheduling/phaseQueueScheduler.h"

#include <gtest/gtest.h>

namespace trt_edgellm
{
namespace rt
{
namespace
{

TEST(PhaseQueueSchedulerTest, BatchesQueuesIndependently)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.maxDecodeBatchSize = 3;
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 32});
    scheduler.enqueuePrefill({2, 32});
    scheduler.enqueueDecode({3, 128});
    scheduler.enqueueDecode({4, 128});
    scheduler.enqueueDecode({5, 128});

    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.kind, PhaseDispatchKind::kOverlap);
    ASSERT_EQ(plan.prefillBatch.size(), 2U);
    ASSERT_EQ(plan.decodeBatch.size(), 3U);
    EXPECT_EQ(plan.prefillBatch[0].requestId, 1U);
    EXPECT_EQ(plan.decodeBatch[0].requestId, 3U);
}

TEST(PhaseQueueSchedulerTest, GivesLongPrefillDecodePriority)
{
    PhaseQueueScheduler scheduler;
    scheduler.enqueuePrefill({1, 512});
    scheduler.enqueueDecode({2, 512});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.kind, PhaseDispatchKind::kDecode);
    EXPECT_TRUE(plan.prefillBatch.empty());
    EXPECT_EQ(plan.decodeBatch.size(), 1U);
}

TEST(PhaseQueueSchedulerTest, BurstLimitPreventsPrefillStarvation)
{
    PhaseQueueSchedulerConfig config;
    config.decodeBurstLimit = 2;
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 512});
    scheduler.enqueueDecode({2, 512});
    EXPECT_EQ(scheduler.next().kind, PhaseDispatchKind::kDecode);
    scheduler.enqueueDecode({3, 512});
    EXPECT_EQ(scheduler.next().kind, PhaseDispatchKind::kDecode);
    scheduler.enqueueDecode({4, 512});
    EXPECT_EQ(scheduler.next().kind, PhaseDispatchKind::kPrefill);
}

TEST(PhaseQueueSchedulerTest, SupportsCustomPolicy)
{
    PhaseQueueSchedulerConfig config;
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kPrefill; };
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 1024});
    scheduler.enqueueDecode({2, 128});

    EXPECT_EQ(scheduler.next().kind, PhaseDispatchKind::kPrefill);
}

TEST(PhaseQueueSchedulerTest, RejectsDuplicateQueuedRequest)
{
    PhaseQueueScheduler scheduler;
    scheduler.enqueuePrefill({7, 32});
    EXPECT_THROW(scheduler.enqueueDecode({7, 32}), std::runtime_error);
}

TEST(PhaseQueueSchedulerTest, RequeuesChunksAndTransitionsToDecode)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 128;
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({42, 300, 3});

    PhaseDispatchPlan first = scheduler.next();
    ASSERT_EQ(first.prefillBatch.size(), 1U);
    EXPECT_EQ(first.prefillBatch[0].kvSlotId, 3);
    EXPECT_EQ(first.prefillBatch[0].tokenOffset, 0);
    EXPECT_EQ(first.prefillBatch[0].tokenCount, 128);
    EXPECT_EQ(first.prefillBatch[0].promptTokenCount, 300);
    scheduler.completePrefill(first.prefillBatch[0], 128);

    PhaseDispatchPlan second = scheduler.next();
    ASSERT_EQ(second.prefillBatch.size(), 1U);
    EXPECT_EQ(second.prefillBatch[0].tokenOffset, 128);
    EXPECT_EQ(second.prefillBatch[0].tokenCount, 128);
    scheduler.completePrefill(second.prefillBatch[0], 256);

    PhaseDispatchPlan third = scheduler.next();
    ASSERT_EQ(third.prefillBatch.size(), 1U);
    EXPECT_EQ(third.prefillBatch[0].tokenOffset, 256);
    EXPECT_EQ(third.prefillBatch[0].tokenCount, 44);
    scheduler.completePrefill(third.prefillBatch[0], 300);

    EXPECT_EQ(scheduler.prefillQueueSize(), 0U);
    EXPECT_EQ(scheduler.decodeQueueSize(), 1U);
    PhaseDispatchPlan decode = scheduler.next();
    ASSERT_EQ(decode.decodeBatch.size(), 1U);
    EXPECT_EQ(decode.decodeBatch[0].kvSlotId, 3);
    EXPECT_EQ(decode.decodeBatch[0].tokenCount, 300);
    scheduler.completeDecode(decode.decodeBatch[0], 301, true);
    EXPECT_FALSE(scheduler.hasRequest(42));
}

TEST(PhaseQueueSchedulerTest, UsesChunkCostForOverlapDecision)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 128;
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 1024, 0});
    scheduler.enqueueDecode({2, 512, 1});

    EXPECT_EQ(scheduler.next().kind, PhaseDispatchKind::kOverlap);
}

TEST(PhaseQueueSchedulerTest, KeepsInFlightRequestUnique)
{
    PhaseQueueScheduler scheduler;
    scheduler.enqueuePrefill({7, 32});
    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 1U);
    EXPECT_THROW(scheduler.enqueueDecode({7, 32}), std::runtime_error);
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
