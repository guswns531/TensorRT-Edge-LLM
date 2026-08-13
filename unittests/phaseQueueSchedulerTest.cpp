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

#include <algorithm>

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

TEST(PhaseQueueSchedulerTest, ResetsLearnedHistoryOnlyWhileIdle)
{
    PhaseQueueScheduler scheduler;
    PhaseDispatchMetrics sample;
    sample.kind = PhaseDispatchKind::kOverlap;
    sample.prefillBatchSize = 1;
    sample.decodeBatchSize = 1;
    sample.prefillTokens = 128;
    sample.decodeContextTokens = 256;
    sample.prefillGpuMs = 4.0F;
    sample.decodeGpuMs = 2.0F;
    sample.overlapRatio = 0.5F;
    scheduler.observeMetrics(sample);
    ASSERT_EQ(scheduler.telemetry().sampleCount, 1U);

    scheduler.resetHistory();

    EXPECT_EQ(scheduler.telemetry().sampleCount, 0U);
    EXPECT_EQ(scheduler.telemetry().overlapSampleCount, 0U);
    EXPECT_FLOAT_EQ(scheduler.telemetry().prefillGpuMsPerToken, 0.0F);
    EXPECT_FLOAT_EQ(scheduler.telemetry().decodeGpuMsPerContextToken, 0.0F);
    EXPECT_FALSE(scheduler.telemetry().lastDispatch.has_value());

    scheduler.enqueuePrefill({1, 128});
    EXPECT_THROW(scheduler.resetHistory(), std::runtime_error);
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

TEST(PhaseQueueSchedulerTest, PackedPrefillAcceptsAnyPositiveEngineChunkContract)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 0;
    config.enablePackedPrefillTokenLayout = true;
    EXPECT_THROW(PhaseQueueScheduler scheduler(config), std::runtime_error);

    config.maxPrefillChunkTokens = 256;
    EXPECT_NO_THROW(PhaseQueueScheduler scheduler(config));
}

TEST(PhaseQueueSchedulerTest, RightPadsRaggedPrefillWithinPaddedTokenBudget)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 4;
    config.maxPrefillChunkTokens = 128;
    config.maxPrefillBatchTokens = 384;
    config.enableRaggedPrefillBatching = true;
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 128, 0, 0, 128});
    scheduler.enqueuePrefill({2, 96, 1, 0, 96});
    scheduler.enqueuePrefill({3, 64, 2, 0, 64});
    scheduler.enqueuePrefill({4, 128, 3, 128, 256});

    PhaseDispatchPlan const initial = scheduler.next();
    ASSERT_EQ(initial.prefillBatch.size(), 3U);
    EXPECT_EQ(initial.prefillBatch[0].tokenCount, 128);
    EXPECT_EQ(initial.prefillBatch[1].tokenCount, 96);
    EXPECT_EQ(initial.prefillBatch[2].tokenCount, 64);
    EXPECT_TRUE(std::all_of(initial.prefillBatch.begin(), initial.prefillBatch.end(),
        [](PhaseWorkItem const& item) { return item.tokenOffset == 0; }));

    PhaseDispatchPlan const continuation = scheduler.next();
    ASSERT_EQ(continuation.prefillBatch.size(), 1U);
    EXPECT_EQ(continuation.prefillBatch[0].requestId, 4U);
    EXPECT_EQ(continuation.prefillBatch[0].tokenOffset, 128);
}

TEST(PhaseQueueSchedulerTest, RaggedPrefillDoesNotPadAtomicRows)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.maxPrefillChunkTokens = 128;
    config.enableRaggedPrefillBatching = true;
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 128, 0, 0, 128, true});
    scheduler.enqueuePrefill({2, 64, 1, 0, 64, false});

    PhaseDispatchPlan const text = scheduler.next();
    ASSERT_EQ(text.prefillBatch.size(), 1U);
    EXPECT_EQ(text.prefillBatch[0].requestId, 1U);
    PhaseDispatchPlan const atomic = scheduler.next();
    ASSERT_EQ(atomic.prefillBatch.size(), 1U);
    EXPECT_EQ(atomic.prefillBatch[0].requestId, 2U);
}

TEST(PhaseQueueSchedulerTest, RaggedOverlapPressureUsesPaddedTokenFootprint)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.maxDecodeBatchSize = 1;
    config.maxPrefillChunkTokens = 128;
    config.enableRaggedPrefillBatching = true;
    config.policy = [](PhaseQueueSnapshot const& state) {
        EXPECT_EQ(state.prefillCandidateTokens, 160);
        return PhaseDispatchKind::kOverlap;
    };
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 80, 0, 0, 80});
    scheduler.enqueuePrefill({2, 40, 1, 0, 40});
    scheduler.enqueueDecode({3, 128, 2});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.kind, PhaseDispatchKind::kOverlap);
    EXPECT_EQ(plan.prefillBatch.size(), 2U);
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

TEST(PhaseQueueSchedulerTest, CompletionBonusSelectsFinalContinuationBucket)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 4;
    config.maxPrefillChunkTokens = 128;
    config.maxPrefillBatchTokens = 512;
    config.enableRaggedPrefillBatching = true;
    config.prefillCompletionBonusTokens = 128;
    PhaseQueueScheduler scheduler(config);
    for (uint64_t requestId = 1; requestId <= 4; ++requestId)
    {
        scheduler.enqueuePrefill({requestId, 128, static_cast<int32_t>(requestId - 1), 0, 256});
    }
    for (uint64_t requestId = 5; requestId <= 8; ++requestId)
    {
        scheduler.enqueuePrefill({requestId, 3, static_cast<int32_t>(requestId - 1), 128, 131});
    }

    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 4U);
    for (PhaseWorkItem const& item : plan.prefillBatch)
    {
        EXPECT_EQ(item.tokenOffset, 128);
        EXPECT_EQ(item.tokenCount, 3);
    }
}

TEST(PhaseQueueSchedulerTest, CompletionBonusDoesNotPromoteNearlyFullContinuationChunks)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 4;
    config.maxPrefillChunkTokens = 128;
    config.maxPrefillBatchTokens = 512;
    config.enableRaggedPrefillBatching = true;
    config.prefillCompletionBonusTokens = 128;
    PhaseQueueScheduler scheduler(config);
    for (uint64_t requestId = 1; requestId <= 2; ++requestId)
    {
        scheduler.enqueuePrefill({requestId, 109, static_cast<int32_t>(requestId - 1), 0, 109});
    }
    for (uint64_t requestId = 5; requestId <= 6; ++requestId)
    {
        scheduler.enqueuePrefill({requestId, 108, static_cast<int32_t>(requestId - 1), 128, 236});
    }

    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 2U);
    for (PhaseWorkItem const& item : plan.prefillBatch)
    {
        EXPECT_EQ(item.tokenOffset, 0);
        EXPECT_EQ(item.tokenCount, 109);
    }
}

TEST(PhaseQueueSchedulerTest, UsesLargerQueueDrainChunkOnlyWhileDecodeIsEmpty)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 1;
    config.maxDecodeBatchSize = 1;
    config.maxOverlapPrefillTokens = 256;
    config.maxPrefillChunkTokens = 256;
    config.decodeActivePrefillChunkTokens = 128;

    PhaseQueueScheduler queueDrainScheduler(config);
    queueDrainScheduler.enqueuePrefill({1, 256, 0, 0, 256});
    PhaseDispatchPlan const queueDrain = queueDrainScheduler.next();
    ASSERT_EQ(queueDrain.prefillBatch.size(), 1U);
    EXPECT_EQ(queueDrain.prefillBatch.front().tokenCount, 256);

    PhaseQueueScheduler steadyStateScheduler(config);
    steadyStateScheduler.enqueuePrefill({1, 256, 0, 0, 256});
    steadyStateScheduler.enqueueDecode({2, 512, 1});
    PhaseDispatchPlan const steadyState = steadyStateScheduler.next();
    ASSERT_EQ(steadyState.prefillBatch.size(), 1U);
    EXPECT_EQ(steadyState.prefillBatch.front().tokenCount, 128);
}

TEST(PhaseQueueSchedulerTest, KeepsLargeChunksUntilPrefillBacklogDrains)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 1;
    config.maxDecodeBatchSize = 1;
    config.maxOverlapPrefillTokens = 256;
    config.maxPrefillChunkTokens = 256;
    config.decodeActivePrefillChunkTokens = 128;
    config.largePrefillChunkQueueThreshold = 2;

    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 256, 0, 0, 256});
    scheduler.enqueuePrefill({2, 256, 1, 0, 256});
    scheduler.enqueueDecode({3, 512, 2});

    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 1U);
    EXPECT_EQ(plan.prefillBatch.front().tokenCount, 256);
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

TEST(PhaseQueueSchedulerTest, DynamicDecodeDoesNotShrinkFromSparseContextCoverage)
{
    PhaseQueueSchedulerConfig config;
    config.maxDecodeBatchSize = 4;
    config.enableDynamicDecodeBatching = true;
    config.decodeQueueWaitTargetUs = 1.0e9;
    config.decodeBatchCosts = {{1, 2048, 1.0F}, {2, 512, 1.5F}, {4, 512, 2.0F}};
    PhaseQueueScheduler scheduler(config);
    for (uint64_t requestId = 1; requestId <= 4; ++requestId)
    {
        scheduler.enqueueDecode({requestId, 1024});
    }

    EXPECT_EQ(scheduler.next().decodeBatch.size(), 4U);
}

TEST(PhaseQueueSchedulerTest, DynamicDecodeUsesUpperBucketForCurrentActiveRows)
{
    PhaseQueueSchedulerConfig config;
    config.maxDecodeBatchSize = 4;
    config.enableDynamicDecodeBatching = true;
    config.decodeQueueWaitTargetUs = 1.0e9;
    config.decodeBatchCosts = {{2, 512, 1.5F}, {4, 512, 2.0F}};
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueueDecode({1, 256});
    scheduler.enqueueDecode({2, 256});
    scheduler.enqueueDecode({3, 256});

    EXPECT_EQ(scheduler.next().decodeBatch.size(), 3U);
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

TEST(PhaseQueueSchedulerTest, DynamicRaggedPrefillUsesUsefulTokenEfficiency)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.maxPrefillChunkTokens = 128;
    config.enableRaggedPrefillBatching = true;
    config.enableDynamicPrefillBatching = true;
    config.prefillBatchCosts = {{1, 128, 0, 0, true, 10.0F, 0.0F}, {2, 128, 0, 0, true, 11.0F, 0.0F}};
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kPrefill; };
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 128, 0, 0, 128});
    scheduler.enqueuePrefill({2, 1, 1, 0, 1});

    PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 1U);
    EXPECT_EQ(plan.prefillBatch.front().requestId, 1U);
    EXPECT_FLOAT_EQ(plan.predictedPrefillGpuMs, 10.0F);
}

TEST(PhaseQueueSchedulerTest, PrefillOnlyDynamicBatchIgnoresQueuedDecodeCoverage)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 8;
    config.maxDecodeBatchSize = 32;
    config.maxPrefillChunkTokens = 128;
    config.enableDynamicPrefillBatching = true;
    config.decodeQueueWaitTargetUs = 100000.0;
    config.prefillBatchCosts = {{4, 128, 0, 32, true, 20.0F, 10.0F}, {8, 128, 0, 0, true, 32.0F, 0.0F}};
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kPrefill; };
    PhaseQueueScheduler scheduler(config);
    for (uint64_t requestId = 1; requestId <= 8; ++requestId)
    {
        scheduler.enqueuePrefill({requestId, 128, static_cast<int32_t>(requestId - 1), 0, 128});
    }
    for (uint64_t requestId = 9; requestId <= 40; ++requestId)
    {
        scheduler.enqueueDecode({requestId, 128, static_cast<int32_t>(requestId - 9)});
    }

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.prefillBatch.size(), 8U);
    EXPECT_FLOAT_EQ(plan.predictedPrefillGpuMs, 32.0F);
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

TEST(PhaseQueueSchedulerTest, TpotHardGuardConvertsUnsafeOverlapToDecodeOnly)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.maxDecodeBatchSize = 2;
    config.maxPrefillChunkTokens = 128;
    config.enableDynamicPrefillBatching = true;
    config.enableTpotHardGuard = true;
    config.requireDirectOverlapCost = true;
    config.decodeQueueWaitTargetUs = 1000.0;
    config.decodeSlackSafetyFactor = 1.0F;
    config.prefillBatchCosts = {{1, 128, 0, 2, true, 10.0F, 4.0F}, {2, 128, 0, 2, true, 15.0F, 6.0F}};
    config.overlapBatchCosts = {
        {1, 2, 128, 0, 512, true, 10.0F, 5.0F, 10.0F, 4.0F},
        {2, 2, 128, 0, 512, true, 15.0F, 7.0F, 15.0F, 6.0F},
    };
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kOverlap; };
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 128, 0, 0, 128});
    scheduler.enqueuePrefill({2, 128, 1, 0, 128});
    scheduler.enqueueDecode({10, 128, 2});
    scheduler.enqueueDecode({11, 128, 3});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.kind, PhaseDispatchKind::kDecode);
    EXPECT_TRUE(plan.prefillBatch.empty());
    EXPECT_EQ(plan.decodeBatch.size(), 2U);
    EXPECT_TRUE(plan.prefillDeferredForTpot);
}

TEST(PhaseQueueSchedulerTest, CostAwareAdmissionCanExceedStaticOverlapCap)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.maxDecodeBatchSize = 1;
    config.maxPrefillChunkTokens = 128;
    config.maxOverlapPrefillTokens = 128;
    config.enableDynamicPrefillBatching = true;
    config.enableTpotHardGuard = true;
    config.requireDirectOverlapCost = true;
    config.enableCostAwareOverlapAdmission = true;
    config.decodeQueueWaitTargetUs = 10000.0;
    config.prefillBatchCosts = {{1, 128, 0, 1, true, 8.0F, 0.5F}, {2, 128, 0, 1, true, 12.0F, 0.8F}};
    config.overlapBatchCosts = {
        {1, 1, 128, 0, 512, true, 8.0F, 2.0F, 8.0F, 0.5F},
        {2, 1, 128, 0, 512, true, 12.0F, 2.5F, 12.0F, 0.8F},
    };
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 128, 0, 0, 128});
    scheduler.enqueuePrefill({2, 128, 1, 0, 128});
    scheduler.enqueueDecode({3, 128, 2});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.kind, PhaseDispatchKind::kOverlap);
    EXPECT_EQ(plan.prefillBatch.size(), 2U);
    EXPECT_TRUE(plan.overlapEvaluatedByCost);
}

TEST(PhaseQueueSchedulerTest, CostAwareAdmissionDefersUncoveredShape)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.maxDecodeBatchSize = 1;
    config.maxPrefillChunkTokens = 128;
    config.maxOverlapPrefillTokens = 64;
    config.enableDynamicPrefillBatching = true;
    config.enableTpotHardGuard = true;
    config.requireDirectOverlapCost = true;
    config.enableCostAwareOverlapAdmission = true;
    config.prefillBatchCosts = {{1, 128, 0, 1, true, 8.0F, 0.5F}};
    config.overlapBatchCosts = {{1, 1, 64, 0, 512, true, 4.0F, 2.0F, 4.0F, 0.5F}};
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 128, 0, 0, 128});
    scheduler.enqueueDecode({2, 128, 1});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.kind, PhaseDispatchKind::kDecode);
    EXPECT_TRUE(plan.prefillBatch.empty());
    EXPECT_TRUE(plan.prefillDeferredForTpot);
    EXPECT_TRUE(plan.prefillCostCoverageMiss);
    EXPECT_TRUE(plan.overlapEvaluatedByCost);
}

TEST(PhaseQueueSchedulerTest, CostAwareAdmissionPreservesLegacyOverlapInsideStaticCap)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 2;
    config.maxDecodeBatchSize = 1;
    config.maxPrefillChunkTokens = 128;
    config.maxOverlapPrefillTokens = 128;
    config.enableTpotHardGuard = true;
    config.requireDirectOverlapCost = true;
    config.enableCostAwareOverlapAdmission = true;
    config.prefillBatchCosts = {{1, 128, 0, 1, true, 8.0F, 0.5F}};
    config.overlapBatchCosts = {{1, 1, 128, 0, 512, true, 8.0F, 2.0F, 8.0F, 0.5F}};
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 64, 0, 0, 64});
    scheduler.enqueuePrefill({2, 64, 1, 0, 64});
    scheduler.enqueueDecode({3, 128, 2});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.kind, PhaseDispatchKind::kOverlap);
    EXPECT_EQ(plan.prefillBatch.size(), 2U);
    EXPECT_FALSE(plan.prefillDeferredForTpot);
    EXPECT_FALSE(plan.prefillCostCoverageMiss);
    EXPECT_FALSE(plan.overlapEvaluatedByCost);
}

TEST(PhaseQueueSchedulerTest, ThroughputBalancedProfileEnablesCostAwareOverlap)
{
    PhaseQueueSchedulerConfig config;
    config.profile = PhaseSchedulerProfile::kThroughputBalanced;
    config.maxPrefillBatchSize = 2;
    config.maxDecodeBatchSize = 1;
    config.maxPrefillChunkTokens = 128;
    config.maxOverlapPrefillTokens = 128;
    config.prefillBatchCosts = {{1, 128, 0, 1, true, 8.0F, 0.5F}, {2, 128, 0, 1, true, 12.0F, 0.8F}};
    config.overlapBatchCosts = {
        {1, 1, 128, 0, 512, true, 8.0F, 2.0F, 8.0F, 0.5F},
        {2, 1, 128, 0, 512, true, 12.0F, 2.5F, 12.0F, 0.8F},
    };
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 128, 0, 0, 128});
    scheduler.enqueuePrefill({2, 128, 1, 0, 128});
    scheduler.enqueueDecode({3, 128, 2});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.kind, PhaseDispatchKind::kOverlap);
    EXPECT_EQ(plan.prefillBatch.size(), 2U);
    EXPECT_TRUE(plan.overlapEvaluatedByCost);
    EXPECT_FALSE(plan.latencySafeFallback);
}

TEST(PhaseQueueSchedulerTest, TpotHysteresisFallsBackAndRecovers)
{
    PhaseQueueSchedulerConfig config;
    config.profile = PhaseSchedulerProfile::kThroughputBalanced;
    config.maxPrefillBatchSize = 2;
    config.maxDecodeBatchSize = 1;
    config.maxPrefillChunkTokens = 128;
    config.maxOverlapPrefillTokens = 128;
    config.decodeQueueWaitTargetUs = 50000.0;
    config.tpotHysteresisEnterRatio = 0.8F;
    config.tpotHysteresisExitRatio = 0.5F;
    config.tpotHysteresisWindow = 3;
    config.minTpotHysteresisSamples = 3;
    config.prefillBatchCosts = {{1, 128, 0, 1, true, 8.0F, 0.5F}, {2, 128, 0, 1, true, 12.0F, 0.8F}};
    config.overlapBatchCosts = {
        {1, 1, 128, 0, 512, true, 8.0F, 2.0F, 8.0F, 0.5F},
        {2, 1, 128, 0, 512, true, 12.0F, 2.5F, 12.0F, 0.8F},
    };
    PhaseQueueScheduler scheduler(config);
    PhaseDispatchMetrics highTpot;
    highTpot.decodeBatchSize = 1;
    highTpot.decodeContextTokens = 128;
    highTpot.decodeQueueWaitUs = 35000.0;
    highTpot.decodeGpuMs = 10.0F;
    scheduler.observeMetrics(highTpot);
    scheduler.observeMetrics(highTpot);
    scheduler.observeMetrics(highTpot);
    EXPECT_TRUE(scheduler.telemetry().latencySafeFallback);
    EXPECT_NEAR(scheduler.telemetry().recentDecodeTpotPressure, 0.9F, 1.0e-6F);
    EXPECT_EQ(scheduler.telemetry().tpotHysteresisTransitions, 1U);

    scheduler.enqueuePrefill({1, 128, 0, 0, 128});
    scheduler.enqueuePrefill({2, 128, 1, 0, 128});
    scheduler.enqueueDecode({3, 128, 2});
    PhaseDispatchPlan const fallback = scheduler.next();
    EXPECT_EQ(fallback.kind, PhaseDispatchKind::kDecode);
    EXPECT_TRUE(fallback.latencySafeFallback);
    EXPECT_FALSE(fallback.overlapEvaluatedByCost);
    ASSERT_EQ(fallback.decodeBatch.size(), 1U);
    scheduler.completeDecode(fallback.decodeBatch.front(), 129, false);

    PhaseDispatchMetrics recoveredTpot = highTpot;
    recoveredTpot.decodeQueueWaitUs = 0.0;
    recoveredTpot.decodeGpuMs = 10.0F;
    scheduler.observeMetrics(recoveredTpot);
    scheduler.observeMetrics(recoveredTpot);
    scheduler.observeMetrics(recoveredTpot);
    EXPECT_FALSE(scheduler.telemetry().latencySafeFallback);
    EXPECT_NEAR(scheduler.telemetry().recentDecodeTpotPressure, 0.2F, 1.0e-6F);
    EXPECT_EQ(scheduler.telemetry().tpotHysteresisTransitions, 2U);

    PhaseDispatchPlan const restored = scheduler.next();
    EXPECT_EQ(restored.kind, PhaseDispatchKind::kOverlap);
    EXPECT_TRUE(restored.overlapEvaluatedByCost);
    EXPECT_FALSE(restored.latencySafeFallback);
}

TEST(PhaseQueueSchedulerTest, DynamicPrefillUsesDirectCostForPlannedDecodeBatch)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 1;
    config.maxDecodeBatchSize = 2;
    config.maxPrefillChunkTokens = 128;
    config.enableDynamicPrefillBatching = true;
    config.enableTpotHardGuard = true;
    config.requireDirectOverlapCost = true;
    config.decodeQueueWaitTargetUs = 10000.0;
    config.prefillBatchCosts = {{1, 128, 0, 2, true, 10.0F, 1.0F}};
    config.overlapBatchCosts = {
        {1, 1, 128, 0, 512, true, 9.0F, 2.0F, 9.0F, 0.5F},
        {1, 2, 128, 0, 512, true, 11.0F, 5.0F, 11.0F, 4.0F},
    };
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kOverlap; };
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 128, 0, 0, 128});
    scheduler.enqueueDecode({10, 128, 1});
    scheduler.enqueueDecode({11, 128, 2});

    PhaseDispatchPlan const plan = scheduler.next();
    EXPECT_EQ(plan.kind, PhaseDispatchKind::kOverlap);
    EXPECT_EQ(plan.plannedDecodeBatchSize, 2);
    EXPECT_FLOAT_EQ(plan.predictedPrefillGpuMs, 11.0F);
    EXPECT_FLOAT_EQ(plan.predictedDecodeSlowdownMs, 4.0F);
}

TEST(PhaseQueueSchedulerTest, TpotHardGuardBoundsConsecutiveOverlapTurns)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillBatchSize = 1;
    config.maxDecodeBatchSize = 1;
    config.maxPrefillChunkTokens = 128;
    config.enableDynamicPrefillBatching = true;
    config.enableTpotHardGuard = true;
    config.requireDirectOverlapCost = true;
    config.maxConsecutiveOverlapBatches = 1;
    config.decodeQueueWaitTargetUs = 10000.0;
    config.prefillBatchCosts = {{1, 128, 0, 1, true, 10.0F, 1.0F}, {1, 128, 128, 1, false, 12.0F, 1.0F}};
    config.overlapBatchCosts = {
        {1, 1, 128, 0, 512, true, 10.0F, 3.0F, 10.0F, 1.0F},
        {1, 1, 128, 128, 512, false, 12.0F, 3.0F, 12.0F, 1.0F},
    };
    config.policy = [](PhaseQueueSnapshot const&) { return PhaseDispatchKind::kOverlap; };
    PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 256, 0, 0, 256});
    scheduler.enqueueDecode({10, 128, 1});

    PhaseDispatchPlan first = scheduler.next();
    ASSERT_EQ(first.kind, PhaseDispatchKind::kOverlap);
    scheduler.completePrefill(first.prefillBatch.front(), 128);
    scheduler.completeDecode(first.decodeBatch.front(), 129, false);
    PhaseDispatchPlan const second = scheduler.next();
    EXPECT_EQ(second.kind, PhaseDispatchKind::kDecode);
    EXPECT_TRUE(second.prefillBatch.empty());
    EXPECT_TRUE(second.prefillDeferredForTpot);
}

} // namespace
} // namespace rt
} // namespace trt_edgellm
