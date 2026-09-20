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

enum class PhaseTransitionDelayKind
{
    kEncoderToPrefill,
    kPrefillToDecode,
    kActionMakespan,
};

char const* phaseTransitionDelayKindName(PhaseTransitionDelayKind kind) noexcept;

constexpr size_t kPHASE_TRANSITION_FEATURES{6U};
using PhaseTransitionFeatures = std::array<double, kPHASE_TRANSITION_FEATURES>;

struct PhaseTransitionPredictorConfig
{
    size_t minimumObservations{4U};
    double confidenceBeta{0.5};
    double initialCovariance{10.0};
    double initialResidualVariance{1.0};
    double forgettingFactor{0.99};
    double maxLatencyClipUs{5000000.0};
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
    size_t predictions{};
    size_t observations{};
    size_t rejectedObservations{};
    double lastMeanUs{};
    double lastUncertaintyUs{};
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
    double predictedTransitionDelayUs{200.0};
    double predictedDecodeStepUs{200.0};
    double predictedPrefillStepUs{1000.0};
};

class PhaseTransitionPredictor
{
public:
    explicit PhaseTransitionPredictor(PhaseTransitionPredictorConfig config = {});

    PhaseTransitionEstimate predict(
        PhaseTransitionDelayKind kind, PhaseTransitionFeatures const& features) const noexcept;

    bool observe(
        PhaseTransitionDelayKind kind, PhaseTransitionFeatures const& features, double measuredLatencyUs) noexcept;

    double predictEpTransitionDelayUs(size_t queueDepth, size_t tokens, double kvUtil) const noexcept;

    double predictPdTransitionDelayUs(size_t queueDepth, size_t tokens, double kvUtil) const noexcept;

    size_t recommendedDecodeBurst(PhaseOptimizationContext const& ctx) const noexcept;

    size_t recommendedDecodeBurst(
        size_t decodeQueueLength, double predictedPdDelayUs, double predictedDecodeDurationUs) const noexcept;

    int32_t recommendedOverlapPrefillTokens(
        PhaseOptimizationContext const& ctx, int32_t defaultTokens = 128) const noexcept;

    int32_t recommendedOverlapPrefillTokens(
        size_t decodeQueueLength, double predictedPdDelayUs, int32_t defaultTokens = 128) const noexcept;

    PhaseTransitionTelemetry const& telemetry(PhaseTransitionDelayKind kind) const noexcept;

    void reset() noexcept;

private:
    struct ModelCore
    {
        PhaseTransitionFeatures theta{};
        std::array<double, kPHASE_TRANSITION_FEATURES * kPHASE_TRANSITION_FEATURES> covariance{};
        double residualVariance{1.0};
        PhaseTransitionTelemetry telemetry{};
    };

    static size_t kindIndex(PhaseTransitionDelayKind kind) noexcept;
    void resetCore(ModelCore& core) noexcept;

    PhaseTransitionPredictorConfig mConfig;
    std::array<ModelCore, 3> mModels{};
};

} // namespace trt_edgellm::rt
