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

#include "runtime/phase/policy/phaseTransitionPredictor.h"

#include <gtest/gtest.h>

#include <cmath>

namespace trt_edgellm::rt
{
namespace
{

TEST(PhaseTransitionPredictorTest, InitialState)
{
    PhaseTransitionPredictor predictor;
    auto const estimate = predictor.predictDecodeQueueWait({32, 32768});
    EXPECT_FALSE(estimate.ready);
    EXPECT_EQ(estimate.observations, 0U);
    EXPECT_GE(estimate.meanUs, 0.0);
}

TEST(PhaseTransitionPredictorTest, NormalizesBatchRowsAndSummedKvContext)
{
    PhaseTransitionFeatures const expected{1.0, 0.5, 0.25};
    EXPECT_EQ(phaseDecodeQueueFeatures({32, 16384}), expected);
}

TEST(PhaseTransitionPredictorTest, LearnsDecodeReadyQueueResidence)
{
    PhaseTransitionPredictorConfig config;
    config.minimumObservations = 4U;
    config.forgettingFactor = 0.99;
    PhaseTransitionPredictor predictor(config);

    PhaseDecodeQueueState const state{16, 32768};
    constexpr double kTargetLatencyUs = 850.0;

    for (size_t i{}; i < 30U; ++i)
    {
        EXPECT_TRUE(predictor.observeDecodeQueueWait(state, kTargetLatencyUs));
    }

    auto const estimate = predictor.predictDecodeQueueWait(state);
    EXPECT_TRUE(estimate.ready);
    EXPECT_EQ(estimate.observations, 30U);
    EXPECT_NEAR(estimate.meanUs, kTargetLatencyUs, 50.0);
    EXPECT_LT(estimate.uncertaintyUs, 200.0);
    EXPECT_GE(estimate.upperConfidenceBoundUs, estimate.meanUs);

    auto const& telemetry = predictor.telemetry();
    EXPECT_EQ(telemetry.observations, 30U);
    EXPECT_EQ(telemetry.rejectedObservations, 0U);
    EXPECT_NEAR(telemetry.lastMeasuredUs, kTargetLatencyUs, 1e-3);
}

TEST(PhaseTransitionPredictorTest, AcceptsZeroResidenceAndRejectsInvalidSamples)
{
    PhaseTransitionPredictor predictor;
    PhaseDecodeQueueState const state{4, 2048};
    EXPECT_TRUE(predictor.observeDecodeQueueWait(state, 0.0));
    EXPECT_FALSE(predictor.observeDecodeQueueWait({0, 0}, 100.0));
    EXPECT_FALSE(predictor.observeDecodeQueueWait({4, -1}, 100.0));
    EXPECT_FALSE(predictor.observeDecodeQueueWait(state, -1.0));
    EXPECT_FALSE(predictor.observeDecodeQueueWait(state, std::numeric_limits<double>::quiet_NaN()));
    EXPECT_FALSE(predictor.observeDecodeQueueWait(state, std::numeric_limits<double>::infinity()));
    EXPECT_EQ(predictor.telemetry().observations, 1U);
    EXPECT_EQ(predictor.telemetry().rejectedObservations, 5U);
    EXPECT_DOUBLE_EQ(predictor.predictDecodeQueueWait(state).meanUs, 0.0);
    EXPECT_FALSE(predictor.predictDecodeQueueWait({0, 0}).ready);
}

TEST(PhaseTransitionPredictorTest, UncertaintyDecreasesWithObservations)
{
    PhaseTransitionPredictorConfig config;
    config.initialResidualVariance = 2000.0 * 2000.0;
    PhaseTransitionPredictor predictor(config);
    PhaseDecodeQueueState const state{8, 8192};

    auto const initial = predictor.predictDecodeQueueWait(state);

    for (size_t i{}; i < 20U; ++i)
    {
        predictor.observeDecodeQueueWait(state, 1500.0);
    }

    auto const updated = predictor.predictDecodeQueueWait(state);
    EXPECT_LT(updated.uncertaintyUs, initial.uncertaintyUs);
}

TEST(PhaseTransitionPredictorTest, ShadowQueueLearningCannotChangeBurstOrOverlap)
{
    PhaseTransitionPredictor predictor;
    PhaseOptimizationContext ctx{};
    ctx.decodeQueued = 16U;
    ctx.prefillQueued = 8U;
    ctx.prefillWaitUs = 22000.0;
    ctx.predictedDecodeStepUs = 1000.0;
    ctx.predictedPrefillStepUs = 3000.0;
    size_t const burst = predictor.recommendedDecodeBurst(ctx);
    int32_t const overlap = predictor.recommendedOverlapPrefillTokens(ctx);
    for (size_t index{}; index < 40U; ++index)
    {
        ASSERT_TRUE(predictor.observeDecodeQueueWait({16, 16384}, 1000000.0));
    }
    EXPECT_EQ(predictor.recommendedDecodeBurst(ctx), burst);
    EXPECT_EQ(predictor.recommendedOverlapPrefillTokens(ctx), overlap);
}

TEST(PhaseTransitionPredictorTest, OptimalDecodeBurstUnderBalancedWorkload)
{
    PhaseTransitionPredictor predictor;

    // Balanced workload: 16 decode rows active, no prefill queued -> maximum burst to saturate SMs
    PhaseOptimizationContext ctx{};
    ctx.decodeQueued = 16U;
    ctx.prefillQueued = 0U;
    ctx.decodeTokens = 16;
    ctx.predictedDecodeStepUs = 500.0;
    ctx.dispatchOverheadUs = 300.0;

    EXPECT_EQ(predictor.recommendedDecodeBurst(ctx), 16U);

    // Fresh prefill arrival (wait time = 0) -> burst remains high (>= 8) to amortize phase transition
    ctx.prefillQueued = 1U;
    ctx.prefillTokens = 128;
    ctx.prefillWaitUs = 0.0;
    ctx.predictedPrefillStepUs = 2000.0;

    size_t const burst = predictor.recommendedDecodeBurst(ctx);
    EXPECT_GE(burst, 8U);
}

TEST(PhaseTransitionPredictorTest, OptimalDecodeBurstUnderPoissonBurst)
{
    PhaseTransitionPredictor predictor;

    // Poisson burst: prefill requests have been waiting in queue (20ms) -> yields immediately (burst = 1)
    PhaseOptimizationContext ctx{};
    ctx.decodeQueued = 8U;
    ctx.prefillQueued = 4U;
    ctx.decodeTokens = 8;
    ctx.prefillWaitUs = 20000.0; // 20ms queue wait
    ctx.predictedDecodeStepUs = 1000.0;
    ctx.dispatchOverheadUs = 500.0;
    ctx.predictedPrefillStepUs = 3000.0;

    EXPECT_LE(predictor.recommendedDecodeBurst(ctx), 3U);

    // When TTFT slack is tight (only 1 decode step remaining before deadline violation) -> strictly 1
    ctx.prefillSlackUs = 21000.0;
    EXPECT_EQ(predictor.recommendedDecodeBurst(ctx), 1U);
}

TEST(PhaseTransitionPredictorTest, OptimalOverlapTokensWithSlack)
{
    PhaseTransitionPredictor predictor;

    PhaseOptimizationContext ctx{};
    ctx.decodeQueued = 8U;
    ctx.prefillQueued = 1U;
    ctx.predictedDecodeStepUs = 500.0;

    // Ample decode slack -> full 512 tokens allowed
    ctx.decodeSlackUs = 2000.0;
    EXPECT_EQ(predictor.recommendedOverlapPrefillTokens(ctx, 512), 512);

    // Tight decode slack (50us < 20% slowdown of 500us = 100us) -> throttled to 64
    ctx.decodeSlackUs = 50.0;
    EXPECT_EQ(predictor.recommendedOverlapPrefillTokens(ctx, 512), 64);
}

} // namespace
} // namespace trt_edgellm::rt
