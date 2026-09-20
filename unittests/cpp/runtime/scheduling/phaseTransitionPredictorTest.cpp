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
    PhaseTransitionFeatures features{1.0, 0.5, 0.5, 0.0, 0.5, 0.0};

    auto const ep = predictor.predict(PhaseTransitionDelayKind::kEncoderToPrefill, features);
    EXPECT_FALSE(ep.ready);
    EXPECT_EQ(ep.observations, 0U);
    EXPECT_GE(ep.meanUs, 0.0);

    auto const pd = predictor.predict(PhaseTransitionDelayKind::kPrefillToDecode, features);
    EXPECT_FALSE(pd.ready);
    EXPECT_EQ(pd.observations, 0U);

    EXPECT_STREQ(phaseTransitionDelayKindName(PhaseTransitionDelayKind::kEncoderToPrefill), "encoder_to_prefill");
    EXPECT_STREQ(phaseTransitionDelayKindName(PhaseTransitionDelayKind::kPrefillToDecode), "prefill_to_decode");
    EXPECT_STREQ(phaseTransitionDelayKindName(PhaseTransitionDelayKind::kActionMakespan), "action_makespan");
}

TEST(PhaseTransitionPredictorTest, LearnsEncoderToPrefillDelay)
{
    PhaseTransitionPredictorConfig config;
    config.minimumObservations = 4U;
    config.forgettingFactor = 0.99;
    PhaseTransitionPredictor predictor(config);

    PhaseTransitionFeatures features{1.0, 0.25, 0.5, 0.0, 0.3, 0.1};
    constexpr double kTargetLatencyUs = 850.0;

    for (size_t i{}; i < 30U; ++i)
    {
        EXPECT_TRUE(predictor.observe(PhaseTransitionDelayKind::kEncoderToPrefill, features, kTargetLatencyUs));
    }

    auto const estimate = predictor.predict(PhaseTransitionDelayKind::kEncoderToPrefill, features);
    EXPECT_TRUE(estimate.ready);
    EXPECT_EQ(estimate.observations, 30U);
    EXPECT_NEAR(estimate.meanUs, kTargetLatencyUs, 50.0);
    EXPECT_LT(estimate.uncertaintyUs, 200.0);
    EXPECT_GE(estimate.upperConfidenceBoundUs, estimate.meanUs);

    auto const& telemetry = predictor.telemetry(PhaseTransitionDelayKind::kEncoderToPrefill);
    EXPECT_EQ(telemetry.observations, 30U);
    EXPECT_EQ(telemetry.rejectedObservations, 0U);
    EXPECT_NEAR(telemetry.lastMeasuredUs, kTargetLatencyUs, 1e-3);
}

TEST(PhaseTransitionPredictorTest, LearnsPrefillToDecodeDelay)
{
    PhaseTransitionPredictor predictor;
    PhaseTransitionFeatures features{1.0, 0.5, 0.25, 1.0, 0.6, 0.2};
    constexpr double kTargetLatencyUs = 320.0;

    for (size_t i{}; i < 40U; ++i)
    {
        EXPECT_TRUE(predictor.observe(PhaseTransitionDelayKind::kPrefillToDecode, features, kTargetLatencyUs));
    }

    auto const estimate = predictor.predict(PhaseTransitionDelayKind::kPrefillToDecode, features);
    EXPECT_TRUE(estimate.ready);
    EXPECT_NEAR(estimate.meanUs, kTargetLatencyUs, 30.0);
}

TEST(PhaseTransitionPredictorTest, UncertaintyDecreasesWithObservations)
{
    PhaseTransitionPredictorConfig config;
    config.initialResidualVariance = 2000.0 * 2000.0;
    PhaseTransitionPredictor predictor(config);
    PhaseTransitionFeatures features{1.0, 0.1, 0.1, 0.0, 0.1, 0.0};

    auto const initial = predictor.predict(PhaseTransitionDelayKind::kActionMakespan, features);

    for (size_t i{}; i < 20U; ++i)
    {
        predictor.observe(PhaseTransitionDelayKind::kActionMakespan, features, 1500.0);
    }

    auto const updated = predictor.predict(PhaseTransitionDelayKind::kActionMakespan, features);
    EXPECT_LT(updated.uncertaintyUs, initial.uncertaintyUs);
}

TEST(PhaseTransitionPredictorTest, DynamicDecodeBurstAdaptation)
{
    PhaseTransitionPredictor predictor;

    // Small decode queue -> small burst
    EXPECT_EQ(predictor.recommendedDecodeBurst(2U, 500.0, 250.0), 2U);
    EXPECT_EQ(predictor.recommendedDecodeBurst(1U, 500.0, 250.0), 2U);

    // Large decode queue and significant P->D handoff delay -> burst expansion
    size_t const burst = predictor.recommendedDecodeBurst(32U, 2000.0, 200.0);
    EXPECT_GE(burst, 8U);
    EXPECT_LE(burst, 16U);

    // Zero decode duration fallback -> bounded to 8
    EXPECT_EQ(predictor.recommendedDecodeBurst(16U, 500.0, 0.0), 8U);
}

TEST(PhaseTransitionPredictorTest, DynamicOverlapPrefillTokens)
{
    PhaseTransitionPredictor predictor;

    // Normal case -> default 128
    EXPECT_EQ(predictor.recommendedOverlapPrefillTokens(4U, 200.0, 128), 128);

    // Heavy decode queue (> 16) or high handoff latency -> throttle to 64
    EXPECT_EQ(predictor.recommendedOverlapPrefillTokens(20U, 200.0, 128), 64);
    EXPECT_EQ(predictor.recommendedOverlapPrefillTokens(4U, 1500.0, 128), 64);

    // Empty decode queue and minimal handoff delay -> expand to 256
    EXPECT_EQ(predictor.recommendedOverlapPrefillTokens(0U, 50.0, 128), 256);
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
    ctx.predictedTransitionDelayUs = 300.0;

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
    ctx.predictedTransitionDelayUs = 500.0;
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
