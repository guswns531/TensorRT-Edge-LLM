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

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace trt_edgellm::rt
{

struct PhaseTransitionPredictorConfig
{
    double burstGracePeriodUs{20000.0};
};

struct PhaseOptimizationContext
{
    size_t decodeQueued{};
    size_t prefillQueued{};
    int32_t decodeTokens{};
    double prefillWaitUs{};
    double prefillSlackUs{std::numeric_limits<double>::infinity()};
    double decodeSlackUs{std::numeric_limits<double>::infinity()};
    //! Uncalibrated dispatch-amortization baseline; never fitted from ready-queue residence.
    double dispatchOverheadUs{200.0};
    double predictedDecodeStepUs{200.0};
    double predictedPrefillStepUs{1000.0};
};

//! Stateless burst and overlap-token controller using observed service costs and request slack.
class PhaseTransitionPredictor
{
public:
    explicit PhaseTransitionPredictor(PhaseTransitionPredictorConfig config = {});

    size_t recommendedDecodeBurst(PhaseOptimizationContext const& ctx) const noexcept;

    int32_t recommendedOverlapPrefillTokens(
        PhaseOptimizationContext const& ctx, int32_t defaultTokens = 128) const noexcept;

private:
    PhaseTransitionPredictorConfig mConfig;
};

} // namespace trt_edgellm::rt
