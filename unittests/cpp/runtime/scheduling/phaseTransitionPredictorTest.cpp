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

#include <limits>
#include <stdexcept>

namespace trt_edgellm::rt
{
namespace
{

TEST(PhaseTransitionPredictorTest, RejectsInvalidBurstGracePeriod)
{
    PhaseTransitionPredictorConfig config;
    config.burstGracePeriodUs = -1.0;
    EXPECT_THROW(PhaseTransitionPredictor{config}, std::runtime_error);
    config.burstGracePeriodUs = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(PhaseTransitionPredictor{config}, std::runtime_error);
    config.burstGracePeriodUs = std::numeric_limits<double>::infinity();
    EXPECT_THROW(PhaseTransitionPredictor{config}, std::runtime_error);
    config.burstGracePeriodUs = 0.0;
    EXPECT_NO_THROW(PhaseTransitionPredictor{config});
}

TEST(PhaseTransitionPredictorTest, HandlesEmptyAndSmallDecodeQueues)
{
    PhaseTransitionPredictor predictor;
    PhaseOptimizationContext ctx{};
    EXPECT_EQ(predictor.recommendedDecodeBurst(ctx), 0U);
    EXPECT_EQ(predictor.recommendedOverlapPrefillTokens(ctx), 256);
    ctx.decodeQueued = 1U;
    EXPECT_EQ(predictor.recommendedDecodeBurst(ctx), 16U);
    ctx.prefillQueued = 1U;
    EXPECT_EQ(predictor.recommendedDecodeBurst(ctx), 2U);
    ctx.decodeQueued = 2U;
    EXPECT_EQ(predictor.recommendedDecodeBurst(ctx), 2U);
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
