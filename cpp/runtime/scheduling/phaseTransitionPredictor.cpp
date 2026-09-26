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

#include "common/checkMacros.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace trt_edgellm::rt
{

PhaseTransitionPredictor::PhaseTransitionPredictor(PhaseTransitionPredictorConfig config)
    : mConfig(config)
{
    ELLM_CHECK(std::isfinite(mConfig.burstGracePeriodUs) && mConfig.burstGracePeriodUs >= 0.0,
        "Burst grace period must be finite and non-negative");
}

size_t PhaseTransitionPredictor::recommendedDecodeBurst(PhaseOptimizationContext const& ctx) const noexcept
{
    if (ctx.decodeQueued == 0U)
    {
        return 0U;
    }
    if (ctx.prefillQueued == 0U)
    {
        return 16U;
    }

    if (ctx.decodeQueued <= 2U)
    {
        return 2U;
    }

    double const decodeStepUs = std::max(10.0, ctx.predictedDecodeStepUs);
    double const dispatchOverheadUs = std::max(10.0, ctx.dispatchOverheadUs);
    double const prefillStepUs = std::max(50.0, ctx.predictedPrefillStepUs);
    double const decodeTokens = static_cast<double>(
        std::max(1, ctx.decodeTokens > 0 ? ctx.decodeTokens : static_cast<int32_t>(ctx.decodeQueued)));

    size_t bestBurst = 1U;
    double maxObjective = -std::numeric_limits<double>::infinity();

    for (size_t burst = 1U; burst <= 16U; ++burst)
    {
        double const burstTimeUs = static_cast<double>(burst) * decodeStepUs;
        double const cycleTimeUs = burstTimeUs + dispatchOverheadUs;
        double const tokens = static_cast<double>(burst) * decodeTokens;
        double const serviceRate = tokens / cycleTimeUs;

        double const projectedPrefillWaitUs = ctx.prefillWaitUs + burstTimeUs;

        double slackPenalty = 0.0;
        if (std::isfinite(ctx.prefillSlackUs))
        {
            double const violationUs = std::max(0.0, projectedPrefillWaitUs - ctx.prefillSlackUs);
            slackPenalty = violationUs / prefillStepUs;
        }

        double const gracePeriodUs = (std::isfinite(ctx.prefillSlackUs) && ctx.prefillSlackUs > 0.0)
            ? std::max(mConfig.burstGracePeriodUs, ctx.prefillSlackUs * 0.75)
            : mConfig.burstGracePeriodUs;
        double const urgency = (projectedPrefillWaitUs > gracePeriodUs)
            ? (projectedPrefillWaitUs - gracePeriodUs) / (prefillStepUs + dispatchOverheadUs)
            : 0.0;
        double const penaltyFactor = 1.0 + slackPenalty + urgency;

        double const objective = serviceRate / penaltyFactor;
        if (objective > maxObjective)
        {
            maxObjective = objective;
            bestBurst = burst;
        }
    }

    return bestBurst;
}

int32_t PhaseTransitionPredictor::recommendedOverlapPrefillTokens(
    PhaseOptimizationContext const& ctx, int32_t defaultTokens) const noexcept
{
    int32_t const maxTokens = std::max(64, defaultTokens);
    if (ctx.decodeQueued == 0U)
    {
        return std::max(maxTokens, 256);
    }

    double const decodeStepUs = std::max(10.0, ctx.predictedDecodeStepUs);
    double const estimatedSlowdownUs = 0.20 * decodeStepUs;

    if (std::isfinite(ctx.decodeSlackUs) && estimatedSlowdownUs > ctx.decodeSlackUs)
    {
        return std::min(maxTokens, 64);
    }

    return maxTokens;
}

} // namespace trt_edgellm::rt
