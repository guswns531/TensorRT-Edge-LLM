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

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace trt_edgellm::rt
{

constexpr size_t kPHASE_TRANSITION_FEATURES{3U};
using PhaseTransitionFeatures = std::array<double, kPHASE_TRANSITION_FEATURES>;

//! Decode candidate shape; contextTokens is summed KV length, not generated tokens.
struct PhaseDecodeQueueState
{
    int32_t batchRows{};
    int64_t contextTokens{};
};

//! Shared normalization for completed dispatch observations and ready-candidate predictions.
PhaseTransitionFeatures phaseDecodeQueueFeatures(PhaseDecodeQueueState const& state) noexcept;

struct PhaseTransitionPredictorConfig
{
    size_t minimumObservations{4U};
    double confidenceBeta{0.5};
    double initialCovariance{10.0};
    double initialResidualVariance{1.0};
    double forgettingFactor{0.99};
    double maxLatencyClipUs{5000000.0};
    double burstGracePeriodUs{20000.0};
};

struct PhaseTransitionEstimate
{
    double meanUs{};
    double uncertaintyUs{};
    double upperConfidenceBoundUs{};
    size_t observations{};
    bool ready{};
};

struct PhaseTransitionTelemetry
{
    size_t observations{};
    size_t rejectedObservations{};
    double lastMeanUs{};
    double lastMeasuredUs{};
    double lastErrorUs{};
    double absoluteErrorSumUs{};
};

struct PhaseOptimizationContext
{
    size_t decodeQueued{};
    size_t prefillQueued{};
    int32_t decodeTokens{};
    int32_t prefillTokens{};
    double prefillWaitUs{};
    double prefillSlackUs{std::numeric_limits<double>::infinity()};
    double decodeSlackUs{std::numeric_limits<double>::infinity()};
    //! Uncalibrated dispatch-amortization baseline; never fitted from ready-queue residence.
    double dispatchOverheadUs{200.0};
    double predictedDecodeStepUs{200.0};
    double predictedPrefillStepUs{1000.0};
};

//! Queue-pressure predictor and burst controller. Physical phase transitions are not learned here.
class PhaseTransitionPredictor
{
public:
    explicit PhaseTransitionPredictor(PhaseTransitionPredictorConfig config = {});

    PhaseTransitionEstimate predictDecodeQueueWait(PhaseDecodeQueueState const& state) const noexcept;

    //! measuredQueueWaitUs is the maximum enqueue-to-dispatch residence of the selected decode rows.
    bool observeDecodeQueueWait(PhaseDecodeQueueState const& state, double measuredQueueWaitUs) noexcept;

    size_t recommendedDecodeBurst(PhaseOptimizationContext const& ctx) const noexcept;

    int32_t recommendedOverlapPrefillTokens(
        PhaseOptimizationContext const& ctx, int32_t defaultTokens = 128) const noexcept;

    PhaseTransitionTelemetry const& telemetry() const noexcept;

    void reset() noexcept;

private:
    struct ModelCore
    {
        PhaseTransitionFeatures theta{};
        std::array<double, kPHASE_TRANSITION_FEATURES * kPHASE_TRANSITION_FEATURES> covariance{};
        double residualVariance{1.0};
        PhaseTransitionTelemetry telemetry{};
    };

    void resetCore(ModelCore& core) noexcept;

    PhaseTransitionPredictorConfig mConfig;
    ModelCore mModel{};
};

} // namespace trt_edgellm::rt
