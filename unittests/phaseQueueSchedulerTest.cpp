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

TEST(PhaseQueueSchedulerTest, UsesEwmaCostToAvoidExpensiveOverlap)
{
    PhaseQueueSchedulerConfig config;
    config.enableMetricsPolicy = true;
    config.minMetricsSamples = 1;
    config.metricsEwmaAlpha = 1.0F;
    config.maxPredictedOverlapPrefillMs = 5.0F;
    config.prefillQueueWaitTargetUs = 1.0e9;
    config.decodeQueueWaitTargetUs = 1.0e9;
    PhaseQueueScheduler scheduler(config);
    PhaseDispatchMetrics sample;
    sample.kind = PhaseDispatchKind::kPrefill;
    sample.prefillTokens = 100;
    sample.prefillGpuMs = 10.0F;
    scheduler.observeMetrics(sample);
    scheduler.enqueuePrefill({1, 128});
    scheduler.enqueueDecode({2, 512});

    EXPECT_EQ(scheduler.next().kind, PhaseDispatchKind::kDecode);
    EXPECT_EQ(scheduler.telemetry().sampleCount, 1U);
    EXPECT_FLOAT_EQ(scheduler.telemetry().prefillGpuMsPerToken, 0.1F);
}

TEST(PhaseQueueSchedulerTest, SupportsCustomMetricsPolicyAndEwmaTelemetry)
{
    bool called{};
    PhaseQueueSchedulerConfig config;
    config.metricsEwmaAlpha = 0.5F;
    config.metricsPolicy = [&](PhaseQueueSnapshot const& state, PhaseSchedulerTelemetry const& telemetry) {
        called = true;
        EXPECT_EQ(state.prefillQueued, 1U);
        EXPECT_EQ(state.decodeQueued, 1U);
        EXPECT_EQ(telemetry.sampleCount, 2U);
        EXPECT_NEAR(telemetry.prefillGpuMsPerToken, 0.15F, 1.0e-6F);
        EXPECT_NEAR(telemetry.decodeGpuMsPerContextToken, 0.15F, 1.0e-6F);
        EXPECT_NEAR(telemetry.overlapRatio, 0.3F, 1.0e-6F);
        return PhaseDispatchKind::kPrefill;
    };
    PhaseQueueScheduler scheduler(config);
    PhaseDispatchMetrics first;
    first.kind = PhaseDispatchKind::kOverlap;
    first.prefillBatchSize = 1;
    first.decodeBatchSize = 1;
    first.prefillTokens = 100;
    first.decodeContextTokens = 100;
    first.prefillGpuMs = 10.0F;
    first.decodeGpuMs = 10.0F;
    first.overlapRatio = 0.4F;
    scheduler.observeMetrics(first);
    PhaseDispatchMetrics second = first;
    second.prefillGpuMs = 20.0F;
    second.decodeGpuMs = 20.0F;
    second.overlapRatio = 0.2F;
    scheduler.observeMetrics(second);
    scheduler.enqueuePrefill({1, 128});
    scheduler.enqueueDecode({2, 512});

    EXPECT_EQ(scheduler.next().kind, PhaseDispatchKind::kPrefill);
    EXPECT_TRUE(called);
    ASSERT_TRUE(scheduler.telemetry().lastDispatch.has_value());
    EXPECT_FLOAT_EQ(scheduler.telemetry().lastDispatch->overlapRatio, 0.2F);
}

TEST(PhaseQueueSchedulerTest, UsesPerRequestSloAndBoundedPriorityForUrgentQueues)
{
    PhaseQueueSchedulerConfig config;
    config.enableMetricsPolicy = true;
    config.prefillQueueWaitTargetUs = 1.0e9;
    config.decodeQueueWaitTargetUs = 1.0e9;
    config.priorityPressureWeight = 0.25;
    PhaseQueueScheduler scheduler(config);
    PhaseWorkItem prefill{1, 512};
    prefill.scheduling = {3, 0.001, 0.0};
    PhaseWorkItem decode{2, 128};
    decode.scheduling = {0, 0.0, 0.001};
    scheduler.enqueuePrefill(prefill);
    scheduler.enqueueDecode(decode);

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.kind, PhaseDispatchKind::kPrefill);
}

TEST(PhaseQueueSchedulerTest, ExposesSloAndPriorityToCustomPolicy)
{
    bool called{};
    PhaseQueueSchedulerConfig config;
    config.metricsPolicy = [&](PhaseQueueSnapshot const& state, PhaseSchedulerTelemetry const&) {
        called = true;
        EXPECT_EQ(state.prefillHighestPriority, 1);
        EXPECT_EQ(state.decodeHighestPriority, 2);
        EXPECT_GT(state.prefillMaxSloPressure, 0.0);
        EXPECT_GT(state.decodeMaxSloPressure, 0.0);
        return PhaseDispatchKind::kDecode;
    };
    PhaseQueueScheduler scheduler(config);
    PhaseWorkItem prefill{1, 32};
    prefill.scheduling = {1, 1000000.0, 0.0};
    PhaseWorkItem decode{2, 128};
    decode.scheduling = {2, 0.0, 1000000.0};
    scheduler.enqueuePrefill(prefill);
    scheduler.enqueueDecode(decode);

    EXPECT_EQ(scheduler.next().kind, PhaseDispatchKind::kDecode);
    EXPECT_TRUE(called);
}

TEST(PhaseQueueSchedulerTest, BatchesHigherPriorityWithinEachPhase)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.enablePriorityBatching = true;
    PhaseQueueScheduler scheduler(config);
    PhaseWorkItem low{1, 32};
    low.scheduling.priority = 0;
    PhaseWorkItem high{2, 64};
    high.scheduling.priority = 3;
    PhaseWorkItem medium{3, 64};
    medium.scheduling.priority = 2;
    scheduler.enqueuePrefill(low);
    scheduler.enqueuePrefill(high);
    scheduler.enqueuePrefill(medium);

    PhaseDispatchPlan const first = scheduler.next();
    ASSERT_EQ(first.prefillBatch.size(), 2U);
    EXPECT_EQ(first.prefillBatch[0].requestId, 2U);
    EXPECT_EQ(first.prefillBatch[1].requestId, 3U);
    PhaseDispatchPlan const second = scheduler.next();
    ASSERT_EQ(second.prefillBatch.size(), 1U);
    EXPECT_EQ(second.prefillBatch[0].requestId, 1U);
}

TEST(PhaseQueueSchedulerTest, RejectsDuplicateQueuedRequest)
{
    PhaseQueueScheduler scheduler;
    scheduler.enqueuePrefill({7, 32});
    EXPECT_THROW(scheduler.enqueueDecode({7, 32}), std::runtime_error);
}

TEST(PhaseQueueSchedulerTest, BucketsPrefillByChunkLengthAndInitialState)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 3;
    config.maxPrefillChunkTokens = 128;
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 300, 0, 0, 300});
    scheduler.enqueuePrefill({2, 64, 1, 0, 64});
    scheduler.enqueuePrefill({3, 128, 2, 128, 256});
    scheduler.enqueuePrefill({4, 256, 3, 0, 256});

    PhaseDispatchPlan const initial128 = scheduler.next();
    ASSERT_EQ(initial128.prefillBatch.size(), 2U);
    EXPECT_EQ(initial128.prefillBatch[0].requestId, 1U);
    EXPECT_EQ(initial128.prefillBatch[1].requestId, 4U);
    EXPECT_EQ(initial128.prefillBatch[0].tokenCount, 128);
    EXPECT_EQ(initial128.prefillBatch[1].tokenCount, 128);

    PhaseDispatchPlan const initial64 = scheduler.next();
    ASSERT_EQ(initial64.prefillBatch.size(), 1U);
    EXPECT_EQ(initial64.prefillBatch[0].requestId, 2U);
    EXPECT_EQ(initial64.prefillBatch[0].tokenCount, 64);

    PhaseDispatchPlan const continuation128 = scheduler.next();
    ASSERT_EQ(continuation128.prefillBatch.size(), 1U);
    EXPECT_EQ(continuation128.prefillBatch[0].requestId, 3U);
    EXPECT_EQ(continuation128.prefillBatch[0].tokenOffset, 128);
}

TEST(PhaseQueueSchedulerTest, AppliesPrefillTokenBudgetWithoutChangingChunkCompatibility)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 4;
    config.maxPrefillChunkTokens = 128;
    config.maxPrefillBatchTokens = 256;
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 128, 0, 0, 128});
    scheduler.enqueuePrefill({2, 128, 1, 0, 128});
    scheduler.enqueuePrefill({3, 128, 2, 0, 128});

    PhaseDispatchPlan const first = scheduler.next();
    ASSERT_EQ(first.prefillBatch.size(), 2U);
    EXPECT_EQ(first.prefillBatch[0].tokenCount, 128);
    EXPECT_EQ(first.prefillBatch[1].tokenCount, 128);
    EXPECT_EQ(scheduler.prefillQueueSize(), 1U);
}

TEST(PhaseQueueSchedulerTest, TokenBudgetSelectsTheMostProductiveCompatibleBucket)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 4;
    config.maxPrefillChunkTokens = 128;
    config.maxPrefillBatchTokens = 256;
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 64, 0, 0, 64});
    scheduler.enqueuePrefill({2, 128, 1, 0, 128});
    scheduler.enqueuePrefill({3, 128, 2, 0, 128});

    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 2U);
    EXPECT_EQ(plan.prefillBatch[0].tokenCount, 128);
    EXPECT_EQ(plan.prefillBatch[1].tokenCount, 128);
}

TEST(PhaseQueueSchedulerTest, DynamicDecodeUsesMostEfficientBatchWithinDeadline)
{
    PhaseQueueSchedulerConfig config;
    config.maxDecodeBatchSize = 4;
    config.enableDynamicDecodeBatching = true;
    config.decodeQueueWaitTargetUs = 1.0e9;
    config.decodeBatchCosts = {{1, 512, 1.0F}, {2, 512, 1.5F}, {4, 512, 2.0F}};
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueueDecode({1, 256});
    scheduler.enqueueDecode({2, 256});
    scheduler.enqueueDecode({3, 256});
    scheduler.enqueueDecode({4, 256});

    EXPECT_EQ(scheduler.next().decodeBatch.size(), 4U);
}

TEST(PhaseQueueSchedulerTest, DynamicDecodeUsesMostEfficientBatchToRecoverAfterDeadline)
{
    PhaseQueueSchedulerConfig config;
    config.maxDecodeBatchSize = 4;
    config.enableDynamicDecodeBatching = true;
    config.decodeQueueWaitTargetUs = 0.001;
    config.decodeBatchCosts = {{1, 512, 1.0F}, {2, 512, 1.5F}, {4, 512, 2.0F}};
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueueDecode({1, 256});
    scheduler.enqueueDecode({2, 256});
    scheduler.enqueueDecode({3, 256});
    scheduler.enqueueDecode({4, 256});

    EXPECT_EQ(scheduler.next().decodeBatch.size(), 4U);
}

TEST(PhaseQueueSchedulerTest, PagePressurePrefersDecodeWithoutOverridingAnExpiredPrefill)
{
    PhaseQueueSchedulerConfig config;
    config.enableMetricsPolicy = true;
    config.prefillQueueWaitTargetUs = 1.0e9;
    config.decodeQueueWaitTargetUs = 1.0e9;
    PhaseQueueScheduler scheduler(config);
    PhaseDispatchMetrics pressure;
    pressure.pagePoolTotalBundles = 100;
    pressure.pagePoolAllocatedBundles = 90;
    scheduler.observeMetrics(pressure);
    scheduler.enqueuePrefill({1, 128});
    scheduler.enqueueDecode({2, 128});

    EXPECT_EQ(scheduler.next().kind, PhaseDispatchKind::kDecode);
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

TEST(PhaseQueueSchedulerTest, AdaptsChunkSizeFromObservedGpuCost)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 128;
    config.enableAdaptivePrefillChunking = true;
    config.minPrefillChunkTokens = 16;
    config.prefillChunkAlignment = 16;
    config.maxPredictedOverlapPrefillMs = 5.0F;
    config.metricsEwmaAlpha = 1.0F;
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kPrefill; };
    PhaseQueueScheduler scheduler(config);
    PhaseDispatchMetrics sample;
    sample.prefillTokens = 100;
    sample.prefillGpuMs = 10.0F;
    scheduler.observeMetrics(sample);
    scheduler.enqueuePrefill({1, 512, 0, 0, 512});
    scheduler.enqueueDecode({2, 512, 1});

    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 1U);
    EXPECT_EQ(plan.prefillBatch.front().tokenCount, 48);
}

TEST(PhaseQueueSchedulerTest, KeepsNonChunkableMultimodalPrefillAtomic)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 128;
    config.enableAdaptivePrefillChunking = true;
    PhaseQueueScheduler scheduler(config);
    PhaseWorkItem multimodal{1, 300, 0, 0, 300};
    multimodal.allowChunkedPrefill = false;
    scheduler.enqueuePrefill(multimodal);

    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 1U);
    EXPECT_EQ(plan.prefillBatch.front().tokenCount, 300);
}

TEST(PhaseQueueSchedulerTest, DoesNotFragmentEfficientChunkForDecodeQueueWait)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 1;
    config.maxDecodeBatchSize = 1;
    config.maxPrefillChunkTokens = 128;
    config.minPrefillChunkTokens = 32;
    config.prefillChunkAlignment = 16;
    config.enableAdaptivePrefillChunking = true;
    config.decodeQueueWaitTargetUs = 0.001;
    config.maxPredictedOverlapPrefillMs = 30.0F;
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 256, 0, 0, 256});
    scheduler.enqueueDecode({2, 128, 1});
    PhaseDispatchMetrics observed;
    observed.prefillTokens = 128;
    observed.prefillGpuMs = 12.8F;
    observed.decodeContextTokens = 128;
    observed.decodeGpuMs = 4.0F;
    scheduler.observeMetrics(observed);

    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 1U);
    EXPECT_EQ(plan.prefillBatch.front().tokenCount, 128);
}

TEST(PhaseQueueSchedulerTest, KeepsInFlightRequestUnique)
{
    PhaseQueueScheduler scheduler;
    scheduler.enqueuePrefill({7, 32});
    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 1U);
    EXPECT_THROW(scheduler.enqueueDecode({7, 32}), std::runtime_error);
}

TEST(PhaseQueueSchedulerTest, KeepsIneligibleWorkQueuedUntilItsGrowthLeaseOpens)
{
    uint64_t eligibleRequest{2};
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.eligibilityPolicy = [&](PhaseWorkItem const& item, bool prefill) {
        EXPECT_TRUE(prefill);
        return item.requestId == eligibleRequest;
    };
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 32});
    scheduler.enqueuePrefill({2, 32});

    PhaseDispatchPlan first = scheduler.next();
    ASSERT_EQ(first.prefillBatch.size(), 1U);
    EXPECT_EQ(first.prefillBatch.front().requestId, 2U);
    scheduler.completePrefill(first.prefillBatch.front(), 32, true);

    eligibleRequest = 1;
    PhaseDispatchPlan second = scheduler.next();
    ASSERT_EQ(second.prefillBatch.size(), 1U);
    EXPECT_EQ(second.prefillBatch.front().requestId, 1U);
    scheduler.completePrefill(second.prefillBatch.front(), 32, true);
    EXPECT_TRUE(scheduler.empty());
}

TEST(PhaseQueueSchedulerTest, PreservesCumulativeTtftAgeAcrossChunkRequeue)
{
    int32_t observations{};
    PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 128;
    config.metricsPolicy = [&](PhaseQueueSnapshot const& state, PhaseSchedulerTelemetry const&) {
        ++observations;
        EXPECT_GT(state.prefillOldestRequestAgeUs, 1000000.0);
        EXPECT_LT(state.prefillMinTtftSlackUs, 0.0);
        EXPECT_GT(state.prefillMaxSloPressure, 1.0);
        return PhaseDispatchKind::kPrefill;
    };
    PhaseQueueScheduler scheduler(config);
    PhaseWorkItem item{1, 256, 0, 0, 256};
    item.scheduling.ttftTargetUs = 1000000.0;
    item.scheduling.submittedAt = std::chrono::steady_clock::now() - std::chrono::seconds(2);
    scheduler.enqueuePrefill(item);

    PhaseDispatchPlan first = scheduler.next();
    scheduler.completePrefill(first.prefillBatch.front(), 128);
    PhaseDispatchPlan const second = scheduler.next();
    EXPECT_EQ(second.prefillBatch.front().tokenOffset, 128);
    EXPECT_EQ(observations, 2);
}

TEST(PhaseQueueSchedulerTest, WavefrontKeepsARequestCohortAcrossChunks)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.maxPrefillChunkTokens = 128;
    config.enableWavefrontPrefillBatching = true;
    config.maxPrefillCohortSize = 2;
    config.maxPrefillCohortTurns = 4;
    PhaseQueueScheduler scheduler(config);
    for (uint64_t requestId = 1; requestId <= 4; ++requestId)
    {
        scheduler.enqueuePrefill({requestId, 256, static_cast<int32_t>(requestId - 1), 0, 256});
    }

    PhaseDispatchPlan first = scheduler.next();
    ASSERT_EQ(first.prefillBatch.size(), 2U);
    EXPECT_EQ(first.prefillCohortSize, 2);
    scheduler.completePrefill(first.prefillBatch[0], 128);
    scheduler.completePrefill(first.prefillBatch[1], 128);
    PhaseDispatchPlan const second = scheduler.next();
    ASSERT_EQ(second.prefillBatch.size(), 2U);
    EXPECT_EQ(second.prefillBatch[0].requestId, first.prefillBatch[0].requestId);
    EXPECT_EQ(second.prefillBatch[1].requestId, first.prefillBatch[1].requestId);
    EXPECT_EQ(second.prefillBatch[0].tokenOffset, 128);
}

TEST(PhaseQueueSchedulerTest, DynamicPrefillUsesLargestBatchInsideDecodeSlack)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 4;
    config.maxDecodeBatchSize = 4;
    config.maxPrefillChunkTokens = 128;
    config.enableDynamicPrefillBatching = true;
    config.decodeQueueWaitTargetUs = 5000.0;
    config.decodeSlackSafetyFactor = 1.0F;
    config.prefillBatchCosts
        = {{1, 128, 0, 4, true, 10.0F, 1.0F}, {2, 128, 0, 4, true, 15.0F, 4.0F}, {4, 128, 0, 4, true, 20.0F, 10.0F}};
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kOverlap; };
    PhaseQueueScheduler scheduler(config);
    for (uint64_t requestId = 1; requestId <= 4; ++requestId)
    {
        scheduler.enqueuePrefill({requestId, 128, static_cast<int32_t>(requestId - 1), 0, 128});
    }
    scheduler.enqueueDecode({10, 128, 4});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.prefillBatch.size(), 2U);
    EXPECT_FLOAT_EQ(plan.predictedPrefillGpuMs, 15.0F);
    EXPECT_FLOAT_EQ(plan.predictedDecodeSlowdownMs, 4.0F);
}

TEST(PhaseQueueSchedulerTest, DynamicPrefillUsesThroughputEfficientBatchToRecoverExpiredTtft)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 4;
    config.maxDecodeBatchSize = 4;
    config.maxPrefillChunkTokens = 128;
    config.enableDynamicPrefillBatching = true;
    config.enablePrefillSloRecovery = true;
    config.prefillQueueWaitTargetUs = 1.0;
    config.decodeQueueWaitTargetUs = 5000.0;
    config.prefillBatchCosts
        = {{1, 128, 0, 4, true, 10.0F, 1.0F}, {2, 128, 0, 4, true, 15.0F, 4.0F}, {4, 128, 0, 4, true, 20.0F, 10.0F}};
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kOverlap; };
    PhaseQueueScheduler scheduler(config);
    PhaseSchedulingHints overdue;
    overdue.submittedAt = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    for (uint64_t requestId = 1; requestId <= 4; ++requestId)
    {
        scheduler.enqueuePrefill({requestId, 128, static_cast<int32_t>(requestId - 1), 0, 128, true, overdue});
    }
    scheduler.enqueueDecode({10, 128, 4});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.prefillBatch.size(), 4U);
    EXPECT_FLOAT_EQ(plan.predictedPrefillGpuMs, 20.0F);
    EXPECT_FLOAT_EQ(plan.predictedDecodeSlowdownMs, 10.0F);
}

TEST(PhaseQueueSchedulerTest, DynamicPrefillHonorsConfiguredBatchFloor)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.maxDecodeBatchSize = 2;
    config.maxPrefillChunkTokens = 128;
    config.enableDynamicPrefillBatching = true;
    config.minDynamicPrefillBatchSize = 2;
    config.decodeQueueWaitTargetUs = 2000.0;
    config.decodeSlackSafetyFactor = 1.0F;
    config.prefillBatchCosts = {{1, 128, 0, 2, true, 10.0F, 1.0F}, {2, 128, 0, 2, true, 15.0F, 4.0F}};
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kOverlap; };
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 128, 0, 0, 128});
    scheduler.enqueuePrefill({2, 128, 1, 0, 128});
    scheduler.enqueueDecode({10, 128, 2});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.prefillBatch.size(), 2U);
    EXPECT_FLOAT_EQ(plan.predictedPrefillGpuMs, 15.0F);
}

TEST(PhaseQueueSchedulerTest, DynamicPrefillDoesNotOverrideAnExpiredDecodeSlo)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 4;
    config.maxDecodeBatchSize = 4;
    config.maxPrefillChunkTokens = 128;
    config.enableDynamicPrefillBatching = true;
    config.minDynamicPrefillBatchSize = 2;
    config.enablePrefillSloRecovery = true;
    config.prefillQueueWaitTargetUs = 1.0;
    config.decodeQueueWaitTargetUs = 0.000001;
    config.prefillBatchCosts = {{2, 128, 0, 4, true, 15.0F, 4.0F}, {4, 128, 0, 4, true, 20.0F, 10.0F}};
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kOverlap; };
    PhaseQueueScheduler scheduler(config);
    PhaseSchedulingHints overdue;
    overdue.submittedAt = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    for (uint64_t requestId = 1; requestId <= 4; ++requestId)
    {
        scheduler.enqueuePrefill({requestId, 128, static_cast<int32_t>(requestId - 1), 0, 128, true, overdue});
    }
    scheduler.enqueueDecode({10, 128, 4, 0, 0, true, overdue});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.prefillBatch.size(), 2U);
    EXPECT_FLOAT_EQ(plan.predictedPrefillGpuMs, 15.0F);
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
