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

char const* phaseTransitionDelayKindName(PhaseTransitionDelayKind kind) noexcept
{
    switch (kind)
    {
    case PhaseTransitionDelayKind::kEncoderToPrefill: return "encoder_to_prefill";
    case PhaseTransitionDelayKind::kPrefillToDecode: return "prefill_to_decode";
    case PhaseTransitionDelayKind::kActionMakespan: return "action_makespan";
    }
    return "unknown";
}

size_t PhaseTransitionPredictor::kindIndex(PhaseTransitionDelayKind kind) noexcept
{
    switch (kind)
    {
    case PhaseTransitionDelayKind::kEncoderToPrefill: return 0U;
    case PhaseTransitionDelayKind::kPrefillToDecode: return 1U;
    case PhaseTransitionDelayKind::kActionMakespan: return 2U;
    }
    return 0U;
}

PhaseTransitionPredictor::PhaseTransitionPredictor(PhaseTransitionPredictorConfig config)
    : mConfig(config)
{
    ELLM_CHECK(mConfig.minimumObservations > 0U, "Minimum observations must be positive");
    ELLM_CHECK(std::isfinite(mConfig.confidenceBeta) && mConfig.confidenceBeta >= 0.0,
        "Confidence beta must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.initialCovariance) && mConfig.initialCovariance > 0.0,
        "Initial covariance must be finite and positive");
    ELLM_CHECK(
        std::isfinite(mConfig.forgettingFactor) && mConfig.forgettingFactor > 0.0 && mConfig.forgettingFactor <= 1.0,
        "Forgetting factor must be within (0, 1]");
    reset();
}

void PhaseTransitionPredictor::resetCore(ModelCore& core) noexcept
{
    core.theta.fill(0.0);
    core.covariance.fill(0.0);
    for (size_t i{}; i < kPHASE_TRANSITION_FEATURES; ++i)
    {
        core.covariance[i * kPHASE_TRANSITION_FEATURES + i] = mConfig.initialCovariance;
    }
    core.residualVariance = mConfig.initialResidualVariance;
    core.telemetry = {};
}

void PhaseTransitionPredictor::reset() noexcept
{
    for (auto& core : mModels)
    {
        resetCore(core);
    }
}

PhaseTransitionEstimate PhaseTransitionPredictor::predict(
    PhaseTransitionDelayKind kind, PhaseTransitionFeatures const& features) const noexcept
{
    auto const& core = mModels[kindIndex(kind)];
    PhaseTransitionFeatures projected{};
    for (size_t row{}; row < kPHASE_TRANSITION_FEATURES; ++row)
    {
        for (size_t col{}; col < kPHASE_TRANSITION_FEATURES; ++col)
        {
            projected[row] += core.covariance[row * kPHASE_TRANSITION_FEATURES + col] * features[col];
        }
    }

    double leverage{};
    double mean{};
    for (size_t i{}; i < kPHASE_TRANSITION_FEATURES; ++i)
    {
        leverage += features[i] * projected[i];
        mean += core.theta[i] * features[i];
    }
    leverage = std::max(0.0, leverage);
    mean = std::max(0.0, mean);

    double const uncertainty = std::sqrt(std::max(0.0, core.residualVariance * (1.0 + leverage)));
    double const ucb = mean + mConfig.confidenceBeta * uncertainty;
    bool const ready = core.telemetry.observations >= mConfig.minimumObservations;

    return {mean, uncertainty, ucb, core.telemetry.observations, ready};
}

bool PhaseTransitionPredictor::observe(
    PhaseTransitionDelayKind kind, PhaseTransitionFeatures const& features, double measuredLatencyUs) noexcept
{
    if (!std::isfinite(measuredLatencyUs) || measuredLatencyUs < 0.0)
    {
        auto& core = mModels[kindIndex(kind)];
        ++core.telemetry.rejectedObservations;
        return false;
    }

    double const clippedLatency = std::min(measuredLatencyUs, mConfig.maxLatencyClipUs);
    auto& core = mModels[kindIndex(kind)];

    PhaseTransitionFeatures projected{};
    for (size_t row{}; row < kPHASE_TRANSITION_FEATURES; ++row)
    {
        for (size_t col{}; col < kPHASE_TRANSITION_FEATURES; ++col)
        {
            projected[row] += core.covariance[row * kPHASE_TRANSITION_FEATURES + col] * features[col];
        }
    }

    double leverage{};
    double predictionMean{};
    for (size_t i{}; i < kPHASE_TRANSITION_FEATURES; ++i)
    {
        leverage += features[i] * projected[i];
        predictionMean += core.theta[i] * features[i];
    }
    leverage = std::max(0.0, leverage);

    double const predictionError = clippedLatency - predictionMean;
    double const denominator = mConfig.forgettingFactor + leverage;

    if (!std::isfinite(denominator) || denominator <= std::numeric_limits<double>::epsilon())
    {
        ++core.telemetry.rejectedObservations;
        return false;
    }

    PhaseTransitionFeatures gain{};
    for (size_t i{}; i < kPHASE_TRANSITION_FEATURES; ++i)
    {
        gain[i] = projected[i] / denominator;
        core.theta[i] += gain[i] * predictionError;
    }

    std::array<double, kPHASE_TRANSITION_FEATURES * kPHASE_TRANSITION_FEATURES> nextCov{};
    for (size_t row{}; row < kPHASE_TRANSITION_FEATURES; ++row)
    {
        for (size_t col{}; col < kPHASE_TRANSITION_FEATURES; ++col)
        {
            nextCov[row * kPHASE_TRANSITION_FEATURES + col]
                = (core.covariance[row * kPHASE_TRANSITION_FEATURES + col] - gain[row] * projected[col])
                / mConfig.forgettingFactor;
        }
    }
    core.covariance = nextCov;

    double const alpha = 1.0 / static_cast<double>(std::min<size_t>(core.telemetry.observations + 1U, 32U));
    core.residualVariance
        = std::max(1.0e-6, (1.0 - alpha) * core.residualVariance + alpha * predictionError * predictionError);

    ++core.telemetry.observations;
    core.telemetry.lastMeasuredUs = clippedLatency;
    core.telemetry.lastMeanUs = predictionMean;
    core.telemetry.lastErrorUs = predictionError;
    core.telemetry.absoluteErrorSumUs += std::abs(predictionError);

    return true;
}

double PhaseTransitionPredictor::predictEpTransitionDelayUs(
    size_t queueDepth, size_t tokens, double kvUtil) const noexcept
{
    PhaseTransitionFeatures features{
        1.0,
        static_cast<double>(queueDepth) / 64.0,
        static_cast<double>(tokens) / 1024.0,
        0.0,
        std::clamp(kvUtil, 0.0, 1.0),
        0.0,
    };
    auto const est = predict(PhaseTransitionDelayKind::kEncoderToPrefill, features);
    return est.ready ? est.meanUs : 500.0;
}

double PhaseTransitionPredictor::predictPdTransitionDelayUs(
    size_t queueDepth, size_t tokens, double kvUtil) const noexcept
{
    PhaseTransitionFeatures features{
        1.0,
        static_cast<double>(queueDepth) / 64.0,
        static_cast<double>(tokens) / 1024.0,
        1.0,
        std::clamp(kvUtil, 0.0, 1.0),
        0.0,
    };
    auto const est = predict(PhaseTransitionDelayKind::kPrefillToDecode, features);
    return est.ready ? est.meanUs : 200.0;
}

size_t PhaseTransitionPredictor::recommendedDecodeBurst(
    size_t decodeQueueLength, double predictedPdDelayUs, double predictedDecodeDurationUs) const noexcept
{
    if (decodeQueueLength <= 2U)
    {
        return 2U;
    }
    if (predictedDecodeDurationUs > 0.0 && predictedPdDelayUs > predictedDecodeDurationUs)
    {
        size_t const burst = static_cast<size_t>(predictedPdDelayUs / predictedDecodeDurationUs);
        return std::clamp<size_t>(burst, 2U, 16U);
    }
    return std::min<size_t>(decodeQueueLength, 8U);
}

int32_t PhaseTransitionPredictor::recommendedOverlapPrefillTokens(
    size_t decodeQueueLength, double predictedPdDelayUs, int32_t defaultTokens) const noexcept
{
    if (decodeQueueLength > 16U || predictedPdDelayUs > 1000.0)
    {
        return std::min(defaultTokens, 64);
    }
    if (decodeQueueLength == 0U && predictedPdDelayUs < 100.0)
    {
        return std::max(defaultTokens, 256);
    }
    return defaultTokens;
}

PhaseTransitionTelemetry const& PhaseTransitionPredictor::telemetry(PhaseTransitionDelayKind kind) const noexcept
{
    return mModels[kindIndex(kind)].telemetry;
}

} // namespace trt_edgellm::rt
