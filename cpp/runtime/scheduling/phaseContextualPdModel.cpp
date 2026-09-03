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

#include "runtime/phase/policy/phaseContextualPdModel.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace trt_edgellm::rt
{
namespace
{

double clampFinite(double value, double lower, double upper) noexcept
{
    return std::isfinite(value) ? std::clamp(value, lower, upper) : lower;
}

double dot(PhaseContextualPdFeatures const& left, PhaseContextualPdFeatures const& right) noexcept
{
    double result{};
    for (size_t index{}; index < left.size(); ++index)
    {
        result += left[index] * right[index];
    }
    return result;
}

PhaseContextualPdModelConfig completionModelConfig(PhaseContextualPdModelConfig config) noexcept
{
    constexpr double kCOMPLETION_RESIDUAL_CLIP = 32.0;
    config.rewardClip = std::max(config.rewardClip, kCOMPLETION_RESIDUAL_CLIP);
    return config;
}

} // namespace

char const* phaseContextualPairDirectionName(PhaseContextualPairDirection direction) noexcept
{
    switch (direction)
    {
    case PhaseContextualPairDirection::kPrefillToDecode: return "prefill_to_decode";
    case PhaseContextualPairDirection::kDecodeToPrefill: return "decode_to_prefill";
    case PhaseContextualPairDirection::kEncoderToPrefill: return "encoder_to_prefill";
    case PhaseContextualPairDirection::kPrefillToEncoder: return "prefill_to_encoder";
    case PhaseContextualPairDirection::kEncoderToDecode: return "encoder_to_decode";
    case PhaseContextualPairDirection::kDecodeToEncoder: return "decode_to_encoder";
    }
    return "unknown";
}

PhaseContextualPairKind phaseContextualPairKind(PhaseContextualPairDirection direction) noexcept
{
    switch (direction)
    {
    case PhaseContextualPairDirection::kPrefillToDecode:
    case PhaseContextualPairDirection::kDecodeToPrefill: return PhaseContextualPairKind::kPrefillDecode;
    case PhaseContextualPairDirection::kEncoderToPrefill:
    case PhaseContextualPairDirection::kPrefillToEncoder: return PhaseContextualPairKind::kEncoderPrefill;
    case PhaseContextualPairDirection::kEncoderToDecode:
    case PhaseContextualPairDirection::kDecodeToEncoder: return PhaseContextualPairKind::kEncoderDecode;
    }
    return PhaseContextualPairKind::kPrefillDecode;
}

PhaseContextualPairDirection phaseContextualPairDirection(
    PhaseGlobalActionKind kind, PhaseGlobalResidualAnchor residualAnchor) noexcept
{
    switch (kind)
    {
    case PhaseGlobalActionKind::kPrefillDecode:
        return residualAnchor == PhaseGlobalResidualAnchor::kDecode ? PhaseContextualPairDirection::kDecodeToPrefill
                                                                    : PhaseContextualPairDirection::kPrefillToDecode;
    case PhaseGlobalActionKind::kEncoderPrefill:
        return residualAnchor == PhaseGlobalResidualAnchor::kPrefill ? PhaseContextualPairDirection::kPrefillToEncoder
                                                                     : PhaseContextualPairDirection::kEncoderToPrefill;
    case PhaseGlobalActionKind::kEncoderDecode:
        return residualAnchor == PhaseGlobalResidualAnchor::kDecode ? PhaseContextualPairDirection::kDecodeToEncoder
                                                                    : PhaseContextualPairDirection::kEncoderToDecode;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kEncoder:
    case PhaseGlobalActionKind::kPrefill:
    case PhaseGlobalActionKind::kDecode:
    case PhaseGlobalActionKind::kWait: return PhaseContextualPairDirection::kPrefillToDecode;
    }
    return PhaseContextualPairDirection::kPrefillToDecode;
}

PhaseContextualPdFeatures phaseContextualPdFeatures(PhaseContextualPdInput const& input) noexcept
{
    constexpr int32_t kPrefillBatchCapacity = 8;
    constexpr int32_t kDecodeBatchCapacity = 64;
    constexpr int32_t kChunkQuantum = 128;
    return phaseContextualPairFeatures({input.prefillUs, input.decodeUs, input.minimumSlackUs, input.prefillBatchSize,
        input.decodeBatchSize, kPrefillBatchCapacity, kDecodeBatchCapacity, input.chunkLength, kChunkQuantum,
        input.prefillContextBucket, input.decodeContextBucket, input.executionVariant, input.residualAugmentation,
        input.residualAnchor, input.incumbentDispatchAgeUs, input.incumbentReferenceUs,
        input.requestedStartSkewFraction, input.outstandingBefore});
}

PhaseContextualPdFeatures phaseContextualPdCompletionFeatures(PhaseContextualPdInput const& input) noexcept
{
    constexpr int32_t kPrefillBatchCapacity = 8;
    constexpr int32_t kDecodeBatchCapacity = 64;
    constexpr int32_t kChunkQuantum = 128;
    return phaseContextualCompletionFeatures({input.prefillUs, input.decodeUs, input.minimumSlackUs,
        input.prefillBatchSize, input.decodeBatchSize, kPrefillBatchCapacity, kDecodeBatchCapacity, input.chunkLength,
        kChunkQuantum, input.prefillContextBucket, input.decodeContextBucket, input.executionVariant,
        input.residualAugmentation, input.residualAnchor, input.incumbentDispatchAgeUs, input.incumbentReferenceUs,
        input.requestedStartSkewFraction, input.outstandingBefore});
}

PhaseContextualPdFeatures phaseContextualPairFeatures(PhaseContextualPairInput const& input) noexcept
{
    double const primaryUs = std::max(0.0, input.primaryUs);
    double const secondaryUs = std::max(0.0, input.secondaryUs);
    double const serialUs = std::max(1.0, primaryUs + secondaryUs);
    double const maximumUs = std::max({1.0, primaryUs, secondaryUs});
    double const minimumUs = std::min(primaryUs, secondaryUs);
    double const slackRatio = clampFinite(input.minimumSlackUs / serialUs, -4.0, 8.0);
    bool const primaryGraph = phaseExecutionVariantUsesPrimaryGraph(input.executionVariant);
    bool const secondaryGraph = phaseExecutionVariantUsesSecondaryGraph(input.executionVariant);
    double const primaryBatchDenominator = std::log1p(std::max(1, input.primaryBatchCapacity));
    double const secondaryBatchDenominator = std::log1p(std::max(1, input.secondaryBatchCapacity));
    double const workQuantum = static_cast<double>(std::max(1, input.workQuantum));
    return {1.0, clampFinite(std::log1p(primaryUs / 1000.0) / 4.0, 0.0, 2.0),
        clampFinite(std::log1p(secondaryUs / 1000.0) / 4.0, 0.0, 2.0), primaryUs / serialUs, minimumUs / maximumUs,
        clampFinite(std::log1p(std::max(0, input.primaryBatchSize)) / primaryBatchDenominator, 0.0, 2.0),
        clampFinite(std::log1p(std::max(0, input.secondaryBatchSize)) / secondaryBatchDenominator, 0.0, 2.0),
        clampFinite(static_cast<double>(std::max(0, input.workSize)) / workQuantum, 0.0, 4.0),
        clampFinite(static_cast<double>(std::max(0, input.primaryContextBucket)) / 4.0, 0.0, 4.0),
        clampFinite(static_cast<double>(std::max(0, input.secondaryContextBucket)) / 4.0, 0.0, 4.0), slackRatio / 8.0,
        input.residualAugmentation ? 1.0 : 0.0, input.residualAnchor == PhaseGlobalResidualAnchor::kPrefill ? 1.0 : 0.0,
        input.residualAnchor == PhaseGlobalResidualAnchor::kDecode ? 1.0 : 0.0, primaryGraph ? 1.0 : 0.0,
        secondaryGraph ? 1.0 : 0.0};
}

PhaseContextualPdFeatures phaseContextualCompletionFeatures(PhaseContextualPairInput const& input) noexcept
{
    double const primaryUs = std::max(0.0, input.primaryUs);
    double const secondaryUs = std::max(0.0, input.secondaryUs);
    double const serialUs = std::max(1.0, primaryUs + secondaryUs);
    double const incumbentAgeUs = std::max(0.0, input.incumbentDispatchAgeUs);
    double const incumbentElapsedRatio
        = input.incumbentReferenceUs > 0.0 ? clampFinite(incumbentAgeUs / input.incumbentReferenceUs, 0.0, 2.0) : 0.0;
    double const primaryBatchDenominator = std::log1p(std::max(1, input.primaryBatchCapacity));
    double const secondaryBatchDenominator = std::log1p(std::max(1, input.secondaryBatchCapacity));
    double const workQuantum = static_cast<double>(std::max(1, input.workQuantum));
    double const slackRatio = clampFinite(input.minimumSlackUs / serialUs, -4.0, 8.0);
    return {1.0, clampFinite(std::log1p(primaryUs / 1000.0) / 4.0, 0.0, 2.0),
        clampFinite(std::log1p(secondaryUs / 1000.0) / 4.0, 0.0, 2.0), primaryUs / serialUs,
        clampFinite(std::log1p(std::max(0, input.primaryBatchSize)) / primaryBatchDenominator, 0.0, 2.0),
        clampFinite(std::log1p(std::max(0, input.secondaryBatchSize)) / secondaryBatchDenominator, 0.0, 2.0),
        clampFinite(static_cast<double>(std::max(0, input.workSize)) / workQuantum, 0.0, 4.0),
        clampFinite(
            static_cast<double>(std::max(input.primaryContextBucket, input.secondaryContextBucket)) / 4.0, 0.0, 4.0),
        slackRatio / 8.0, input.residualAugmentation ? 1.0 : 0.0,
        clampFinite(std::log1p(incumbentAgeUs / 1000.0) / 4.0, 0.0, 2.0), incumbentElapsedRatio,
        input.requestedStartSkewFraction >= 0.0 ? clampFinite(input.requestedStartSkewFraction, 0.0, 1.0) : -1.0,
        phaseExecutionSetContains(input.outstandingBefore, PhaseExecutionSet::kEncoder) ? 1.0 : 0.0,
        phaseExecutionSetContains(input.outstandingBefore, PhaseExecutionSet::kPrefill) ? 1.0 : 0.0,
        phaseExecutionSetContains(input.outstandingBefore, PhaseExecutionSet::kDecode) ? 1.0 : 0.0};
}

double phaseContextualDecisionMakespanUs(double referenceWorkUs, double conservativeAdvantage) noexcept
{
    double const reference = std::max(0.0, referenceWorkUs);
    // Normalized reward is (serial - observed) / serial. Bound the mapping so
    // one noisy posterior cannot manufacture a zero-cost action or an
    // unbounded penalty while it is adapting online.
    double const advantage = clampFinite(conservativeAdvantage, -1.0, 0.95);
    return reference * (1.0 - advantage);
}

double phaseContextualCompletionDecisionMakespanUs(
    PhaseContextualCompletionEstimate const& estimate, PhaseContextualPdFeatures const& features) noexcept
{
    constexpr size_t kPRODUCER_BATCH_FILL_FEATURE = 4U;
    double const formationRisk = clampFinite(features[kPRODUCER_BATCH_FILL_FEATURE], 0.0, 1.0);
    double const incumbent
        = std::max(0.0, estimate.incumbentMeanUs) + formationRisk * std::max(0.0, estimate.incumbentUncertaintyUs);
    double const newcomer
        = std::max(0.0, estimate.newcomerMeanUs) + formationRisk * std::max(0.0, estimate.newcomerUncertaintyUs);
    return std::max(incumbent, newcomer);
}

double phaseBlendContextualCompletionDecisionMakespanUs(double scalarMakespanUs,
    PhaseContextualCompletionEstimate const& estimate, PhaseContextualPdFeatures const& features,
    double completionWeight) noexcept
{
    double const weight = clampFinite(completionWeight, 0.0, 1.0);
    double const scalar = std::max(0.0, scalarMakespanUs);
    double const completion = phaseContextualCompletionDecisionMakespanUs(estimate, features);
    return scalar + weight * (completion - scalar);
}

void phaseBlendContextualCompletionComponent(double& predictedCompletionUs, double& uncertaintyUs,
    double completionMeanUs, double completionUncertaintyUs, double completionWeight) noexcept
{
    double const weight = clampFinite(completionWeight, 0.0, 1.0);
    double const baselineMean = std::max(0.0, predictedCompletionUs);
    double const baselineUncertainty = std::max(0.0, uncertaintyUs);
    predictedCompletionUs = baselineMean + weight * (std::max(0.0, completionMeanUs) - baselineMean);
    uncertaintyUs = baselineUncertainty + weight * (std::max(0.0, completionUncertaintyUs) - baselineUncertainty);
}

double phaseContextualPdDecisionValue(
    PhaseContextualPdEstimate const& estimate, bool decodeOnlyRecovery, bool producerCriticalPath) noexcept
{
    return decodeOnlyRecovery && !producerCriticalPath ? estimate.mean : estimate.lowerConfidenceBound;
}

bool phaseContextualPdDecodeDominatedRecovery(
    std::vector<PhaseProtectedCompletion> const& completions, double deadlineGuardUs) noexcept
{
    double decodeViolation{};
    double otherViolation{};
    for (PhaseProtectedCompletion const& completion : completions)
    {
        if (!std::isfinite(completion.slackUs))
        {
            continue;
        }
        double const robustCompletion
            = std::max(0.0, completion.predictedCompletionUs) + std::max(0.0, completion.uncertaintyUs);
        double const violation = robustCompletion + deadlineGuardUs - completion.slackUs;
        if (violation <= 0.0)
        {
            continue;
        }
        if (completion.kind == PhaseProtectedKind::kDecode)
        {
            decodeViolation = std::max(decodeViolation, violation);
        }
        else
        {
            otherViolation = std::max(otherViolation, violation);
        }
    }
    return decodeViolation > otherViolation;
}

bool phaseContextualPdProducerCriticalPath(bool producerOutstanding, bool externalPrefillLineage) noexcept
{
    return producerOutstanding || externalPrefillLineage;
}

bool phaseContextualPdDrainsReadyPrefill(bool allRowsFinal, int64_t remainingTokens, int32_t usefulTokens) noexcept
{
    return allRowsFinal && remainingTokens > 0 && remainingTokens == std::max(0, usefulTokens);
}

bool phaseContextualPdControlsDecision(bool producerCriticalPath) noexcept
{
    return !producerCriticalPath;
}

bool phaseContextualPdMayPromoteOverlap(bool deadlineRecovery, bool producerCriticalPath) noexcept
{
    return deadlineRecovery && !producerCriticalPath;
}

PhaseContextualPdModel::PhaseContextualPdModel(PhaseContextualPdModelConfig config)
    : mConfig(config)
{
    ELLM_CHECK(mConfig.minimumObservations > 0U, "Contextual pair minimum observations must be positive");
    ELLM_CHECK(std::isfinite(mConfig.confidenceBeta) && mConfig.confidenceBeta >= 0.0,
        "Contextual pair confidence beta must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.initialCovariance) && mConfig.initialCovariance > 0.0,
        "Contextual pair initial covariance must be finite and positive");
    ELLM_CHECK(std::isfinite(mConfig.initialResidualVariance) && mConfig.initialResidualVariance > 0.0,
        "Contextual pair initial residual variance must be finite and positive");
    ELLM_CHECK(
        std::isfinite(mConfig.forgettingFactor) && mConfig.forgettingFactor > 0.0 && mConfig.forgettingFactor <= 1.0,
        "Contextual pair forgetting factor must be within (0, 1]");
    ELLM_CHECK(std::isfinite(mConfig.rewardClip) && mConfig.rewardClip > 0.0,
        "Contextual pair reward clip must be finite and positive");
    reset();
}

PhaseContextualPdEstimate PhaseContextualPdModel::predict(PhaseContextualPdFeatures const& features)
{
    PhaseContextualPdFeatures projected{};
    for (size_t row{}; row < features.size(); ++row)
    {
        for (size_t column{}; column < features.size(); ++column)
        {
            projected[row] += mCovariance[row * features.size() + column] * features[column];
        }
    }
    double const leverage = std::max(0.0, dot(features, projected));
    double const mean = dot(mTheta, features);
    double const uncertainty = std::sqrt(std::max(0.0, mResidualVariance * (1.0 + leverage)));
    PhaseContextualPdEstimate result{mean, uncertainty, mean - mConfig.confidenceBeta * uncertainty,
        mTelemetry.observations, mTelemetry.observations >= mConfig.minimumObservations};
    ++mTelemetry.predictions;
    mTelemetry.lastMean = result.mean;
    mTelemetry.lastUncertainty = result.uncertainty;
    mTelemetry.lastLowerConfidenceBound = result.lowerConfidenceBound;
    return result;
}

bool PhaseContextualPdModel::observe(
    PhaseContextualPdFeatures const& features, double normalizedAdvantage, double weight)
{
    if (!std::isfinite(normalizedAdvantage) || !std::isfinite(weight) || weight <= 0.0)
    {
        ++mTelemetry.rejectedObservations;
        return false;
    }
    double const reward = std::clamp(normalizedAdvantage, -mConfig.rewardClip, mConfig.rewardClip);
    // Evaluate the posterior before consuming this label. Gate-C calibration
    // must measure a true online prediction, not the fit after the update.
    PhaseContextualPdFeatures predictionProjection{};
    for (size_t row{}; row < features.size(); ++row)
    {
        for (size_t column{}; column < features.size(); ++column)
        {
            predictionProjection[row] += mCovariance[row * features.size() + column] * features[column];
        }
    }
    double const predictionLeverage = std::max(0.0, dot(features, predictionProjection));
    double const predictionMean = dot(mTheta, features);
    double const predictionUncertainty = std::sqrt(std::max(0.0, mResidualVariance * (1.0 + predictionLeverage)));
    double const predictionLcb = predictionMean - mConfig.confidenceBeta * predictionUncertainty;
    double const predictionError = reward - predictionMean;
    bool const ready = mTelemetry.observations >= mConfig.minimumObservations;
    bool const intervalCovered = std::abs(predictionError) <= mConfig.confidenceBeta * predictionUncertainty;
    ++mTelemetry.calibrationObservations;
    mTelemetry.readyCalibrationObservations += ready ? 1U : 0U;
    mTelemetry.confidenceIntervalCovered += intervalCovered ? 1U : 0U;
    mTelemetry.readyConfidenceIntervalCovered += ready && intervalCovered ? 1U : 0U;
    mTelemetry.predictedSafeObservations += predictionLcb > 0.0 ? 1U : 0U;
    mTelemetry.falseSafeObservations += predictionLcb > 0.0 && reward <= 0.0 ? 1U : 0U;
    mTelemetry.absoluteErrorSum += std::abs(predictionError);
    mTelemetry.squaredErrorSum += predictionError * predictionError;
    mTelemetry.readyAbsoluteErrorSum += ready ? std::abs(predictionError) : 0.0;
    mTelemetry.readySquaredErrorSum += ready ? predictionError * predictionError : 0.0;
    mTelemetry.lastMean = predictionMean;
    mTelemetry.lastUncertainty = predictionUncertainty;
    mTelemetry.lastLowerConfidenceBound = predictionLcb;
    mTelemetry.lastPredictionError = predictionError;
    PhaseContextualPdFeatures projected{};
    for (size_t row{}; row < features.size(); ++row)
    {
        for (size_t column{}; column < features.size(); ++column)
        {
            projected[row] += mCovariance[row * features.size() + column] * features[column];
        }
    }
    double const leverage = std::max(0.0, dot(features, projected));
    double const denominator = mConfig.forgettingFactor / weight + leverage;
    if (!std::isfinite(denominator) || denominator <= std::numeric_limits<double>::epsilon())
    {
        ++mTelemetry.rejectedObservations;
        return false;
    }
    double const residual = reward - dot(mTheta, features);
    PhaseContextualPdFeatures gain{};
    for (size_t index{}; index < gain.size(); ++index)
    {
        gain[index] = projected[index] / denominator;
        mTheta[index] += gain[index] * residual;
    }
    std::array<double, kPHASE_CONTEXTUAL_PD_FEATURES * kPHASE_CONTEXTUAL_PD_FEATURES> next{};
    for (size_t row{}; row < features.size(); ++row)
    {
        for (size_t column{}; column < features.size(); ++column)
        {
            next[row * features.size() + column]
                = (mCovariance[row * features.size() + column] - gain[row] * projected[column])
                / mConfig.forgettingFactor;
        }
    }
    mCovariance = next;
    double const varianceAlpha = 1.0 / static_cast<double>(std::min<size_t>(mTelemetry.observations + 1U, 32U));
    mResidualVariance
        = std::max(1.0e-6, (1.0 - varianceAlpha) * mResidualVariance + varianceAlpha * residual * residual);
    ++mTelemetry.observations;
    mTelemetry.lastReward = reward;
    return true;
}

void PhaseContextualPdModel::recordSelection(bool overlap, bool exploration) noexcept
{
    mTelemetry.positiveSelections += overlap ? 1U : 0U;
    mTelemetry.negativeSelections += overlap ? 0U : 1U;
    mTelemetry.explorations += exploration ? 1U : 0U;
}

void PhaseContextualPdModel::reset() noexcept
{
    mTheta.fill(0.0);
    mCovariance.fill(0.0);
    for (size_t index{}; index < kPHASE_CONTEXTUAL_PD_FEATURES; ++index)
    {
        mCovariance[index * kPHASE_CONTEXTUAL_PD_FEATURES + index] = mConfig.initialCovariance;
    }
    mResidualVariance = mConfig.initialResidualVariance;
    mTelemetry = {};
}

PhaseContextualPdModelConfig const& PhaseContextualPdModel::config() const noexcept
{
    return mConfig;
}

PhaseContextualPdTelemetry const& PhaseContextualPdModel::telemetry() const noexcept
{
    return mTelemetry;
}

PhaseContextualCompletionModel::PhaseContextualCompletionModel(PhaseContextualPdModelConfig config)
    : mConfig(completionModelConfig(config))
    , mIncumbent(mConfig)
    , mNewcomer(mConfig)
{
}

PhaseContextualCompletionEstimate PhaseContextualCompletionModel::estimate(PhaseContextualPdFeatures const& features,
    double incumbentReferenceUs, double newcomerReferenceUs, bool recordPrediction)
{
    incumbentReferenceUs = std::max(0.0, incumbentReferenceUs);
    newcomerReferenceUs = std::max(0.0, newcomerReferenceUs);
    double const serialReferenceUs = std::max(1.0, incumbentReferenceUs + newcomerReferenceUs);
    PhaseContextualPdEstimate const incumbent = mIncumbent.predict(features);
    PhaseContextualPdEstimate const newcomer = mNewcomer.predict(features);
    if (recordPrediction)
    {
        ++mTelemetry.predictions;
    }
    return {std::max(0.0, incumbentReferenceUs + serialReferenceUs * incumbent.mean),
        serialReferenceUs * incumbent.uncertainty,
        std::max(0.0, newcomerReferenceUs + serialReferenceUs * newcomer.mean),
        serialReferenceUs * newcomer.uncertainty, std::min(incumbent.observations, newcomer.observations),
        incumbent.ready && newcomer.ready};
}

PhaseContextualCompletionEstimate PhaseContextualCompletionModel::predict(
    PhaseContextualPdFeatures const& features, double incumbentReferenceUs, double newcomerReferenceUs)
{
    return estimate(features, incumbentReferenceUs, newcomerReferenceUs, true);
}

bool PhaseContextualCompletionModel::observe(PhaseContextualPdFeatures const& features, double incumbentReferenceUs,
    double newcomerReferenceUs, double incumbentCompletionUs, double newcomerCompletionUs, double minimumSlackUs)
{
    if (!std::isfinite(incumbentReferenceUs) || incumbentReferenceUs <= 0.0 || !std::isfinite(newcomerReferenceUs)
        || newcomerReferenceUs <= 0.0 || !std::isfinite(incumbentCompletionUs) || incumbentCompletionUs < 0.0
        || !std::isfinite(newcomerCompletionUs) || newcomerCompletionUs < 0.0)
    {
        return false;
    }
    PhaseContextualCompletionEstimate const prediction
        = estimate(features, incumbentReferenceUs, newcomerReferenceUs, false);
    double const incumbentErrorUs = incumbentCompletionUs - prediction.incumbentMeanUs;
    double const newcomerErrorUs = newcomerCompletionUs - prediction.newcomerMeanUs;
    bool const incumbentCovered
        = std::abs(incumbentErrorUs) <= mConfig.confidenceBeta * prediction.incumbentUncertaintyUs;
    bool const newcomerCovered = std::abs(newcomerErrorUs) <= mConfig.confidenceBeta * prediction.newcomerUncertaintyUs;
    double const predictedRobustUs
        = std::max(prediction.incumbentMeanUs + mConfig.confidenceBeta * prediction.incumbentUncertaintyUs,
            prediction.newcomerMeanUs + mConfig.confidenceBeta * prediction.newcomerUncertaintyUs);
    double const observedRobustUs = std::max(incumbentCompletionUs, newcomerCompletionUs);
    bool const hasSlack = std::isfinite(minimumSlackUs);
    bool const predictedSafe = hasSlack && predictedRobustUs <= minimumSlackUs;
    ++mTelemetry.observations;
    mTelemetry.readyCalibrationObservations += prediction.ready ? 1U : 0U;
    mTelemetry.incumbentIntervalCovered += incumbentCovered ? 1U : 0U;
    mTelemetry.newcomerIntervalCovered += newcomerCovered ? 1U : 0U;
    mTelemetry.readyIncumbentIntervalCovered += prediction.ready && incumbentCovered ? 1U : 0U;
    mTelemetry.readyNewcomerIntervalCovered += prediction.ready && newcomerCovered ? 1U : 0U;
    mTelemetry.predictedSafeObservations += predictedSafe ? 1U : 0U;
    mTelemetry.falseSafeObservations += predictedSafe && observedRobustUs > minimumSlackUs ? 1U : 0U;
    mTelemetry.incumbentAbsoluteErrorUs += std::abs(incumbentErrorUs);
    mTelemetry.incumbentSquaredErrorUs += incumbentErrorUs * incumbentErrorUs;
    mTelemetry.newcomerAbsoluteErrorUs += std::abs(newcomerErrorUs);
    mTelemetry.newcomerSquaredErrorUs += newcomerErrorUs * newcomerErrorUs;
    mTelemetry.readyIncumbentAbsoluteErrorUs += prediction.ready ? std::abs(incumbentErrorUs) : 0.0;
    mTelemetry.readyIncumbentSquaredErrorUs += prediction.ready ? incumbentErrorUs * incumbentErrorUs : 0.0;
    mTelemetry.readyNewcomerAbsoluteErrorUs += prediction.ready ? std::abs(newcomerErrorUs) : 0.0;
    mTelemetry.readyNewcomerSquaredErrorUs += prediction.ready ? newcomerErrorUs * newcomerErrorUs : 0.0;
    double const serialReferenceUs = incumbentReferenceUs + newcomerReferenceUs;
    double const incumbentResidual = (incumbentCompletionUs - incumbentReferenceUs) / serialReferenceUs;
    double const newcomerResidual = (newcomerCompletionUs - newcomerReferenceUs) / serialReferenceUs;
    bool const incumbentObserved = mIncumbent.observe(features, incumbentResidual);
    bool const newcomerObserved = mNewcomer.observe(features, newcomerResidual);
    return incumbentObserved && newcomerObserved;
}

void PhaseContextualCompletionModel::reset() noexcept
{
    mIncumbent.reset();
    mNewcomer.reset();
    mTelemetry = {};
}

PhaseContextualCompletionTelemetry const& PhaseContextualCompletionModel::telemetry() const noexcept
{
    return mTelemetry;
}

PhaseContextualCompletionEstimate phaseBlendContextualCompletionEstimates(PhaseContextualCompletionEstimate const& pair,
    PhaseContextualCompletionEstimate const& direction, double directionPseudoObservations) noexcept
{
    double const pseudoObservations = std::max(1.0, directionPseudoObservations);
    double const directionWeight = static_cast<double>(direction.observations)
        / (static_cast<double>(direction.observations) + pseudoObservations);
    double const pairWeight = 1.0 - directionWeight;
    auto const blendMean = [pairWeight, directionWeight](double pairMean, double directionMean) {
        return pairWeight * pairMean + directionWeight * directionMean;
    };
    auto const blendUncertainty = [pairWeight, directionWeight](double pairMean, double pairUncertainty,
                                      double directionMean, double directionUncertainty) {
        double const disagreement = pairMean - directionMean;
        double const variance = pairWeight * pairWeight * pairUncertainty * pairUncertainty
            + directionWeight * directionWeight * directionUncertainty * directionUncertainty
            + pairWeight * directionWeight * disagreement * disagreement;
        return std::sqrt(std::max(0.0, variance));
    };
    return {blendMean(pair.incumbentMeanUs, direction.incumbentMeanUs),
        blendUncertainty(pair.incumbentMeanUs, pair.incumbentUncertaintyUs, direction.incumbentMeanUs,
            direction.incumbentUncertaintyUs),
        blendMean(pair.newcomerMeanUs, direction.newcomerMeanUs),
        blendUncertainty(
            pair.newcomerMeanUs, pair.newcomerUncertaintyUs, direction.newcomerMeanUs, direction.newcomerUncertaintyUs),
        pair.observations, pair.ready, pair.observations, direction.observations, directionWeight};
}

PhaseContextualCompletionCalibrator::PhaseContextualCompletionCalibrator(
    PhaseContextualCompletionCalibrationConfig config)
    : mConfig(config)
{
    ELLM_CHECK(std::isfinite(mConfig.targetCoverage) && mConfig.targetCoverage > 0.0 && mConfig.targetCoverage < 1.0,
        "Completion calibration target coverage must be finite and between zero and one");
    ELLM_CHECK(mConfig.minimumObservations > 0U, "Completion calibration minimum observations must be positive");
    ELLM_CHECK(mConfig.windowSize >= mConfig.minimumObservations,
        "Completion calibration window must cover the minimum observations");
    ELLM_CHECK(std::isfinite(mConfig.minimumScale) && mConfig.minimumScale > 0.0,
        "Completion calibration minimum scale must be finite and positive");
    ELLM_CHECK(std::isfinite(mConfig.maximumScale) && mConfig.maximumScale >= mConfig.minimumScale,
        "Completion calibration maximum scale must cover the minimum scale");
    mScores.reserve(mConfig.windowSize);
}

PhaseContextualCompletionCalibrationEstimate PhaseContextualCompletionCalibrator::estimate() const
{
    if (!mConfig.enabled || mScores.size() < mConfig.minimumObservations)
    {
        return {mConfig.minimumScale, mScores.size(), false};
    }
    std::vector<double> ordered = mScores;
    std::sort(ordered.begin(), ordered.end());
    double const finiteSampleRank = std::ceil((static_cast<double>(ordered.size()) + 1.0) * mConfig.targetCoverage);
    size_t const rank = std::min(ordered.size(), std::max<size_t>(1U, static_cast<size_t>(finiteSampleRank)));
    double const scale = std::clamp(ordered[rank - 1U], mConfig.minimumScale, mConfig.maximumScale);
    return {scale, mScores.size(), true};
}

PhaseContextualCompletionEstimate PhaseContextualCompletionCalibrator::apply(
    PhaseContextualCompletionEstimate prediction) const
{
    PhaseContextualCompletionCalibrationEstimate const calibration = estimate();
    prediction.incumbentUncertaintyUs *= calibration.scale;
    prediction.newcomerUncertaintyUs *= calibration.scale;
    prediction.uncertaintyScale = calibration.scale;
    prediction.uncertaintyCalibrationObservations = calibration.observations;
    prediction.uncertaintyCalibrated = calibration.ready;
    return prediction;
}

bool PhaseContextualCompletionCalibrator::observe(PhaseContextualCompletionEstimate const& rawPrediction,
    double confidenceBeta, double incumbentCompletionUs, double newcomerCompletionUs)
{
    if (!mConfig.enabled || !rawPrediction.ready || !std::isfinite(confidenceBeta) || confidenceBeta <= 0.0
        || !std::isfinite(incumbentCompletionUs) || !std::isfinite(newcomerCompletionUs))
    {
        return false;
    }
    double const incumbentDenominator = confidenceBeta * rawPrediction.incumbentUncertaintyUs;
    double const newcomerDenominator = confidenceBeta * rawPrediction.newcomerUncertaintyUs;
    if (!std::isfinite(incumbentDenominator) || incumbentDenominator <= std::numeric_limits<double>::epsilon()
        || !std::isfinite(newcomerDenominator) || newcomerDenominator <= std::numeric_limits<double>::epsilon())
    {
        return false;
    }
    double const incumbentScore
        = std::abs(incumbentCompletionUs - rawPrediction.incumbentMeanUs) / incumbentDenominator;
    double const newcomerScore = std::abs(newcomerCompletionUs - rawPrediction.newcomerMeanUs) / newcomerDenominator;
    double const score = std::max(incumbentScore, newcomerScore);
    if (!std::isfinite(score))
    {
        return false;
    }
    if (mScores.size() == mConfig.windowSize)
    {
        mScores.erase(mScores.begin());
    }
    mScores.push_back(score);
    return true;
}

void PhaseContextualCompletionCalibrator::reset() noexcept
{
    mScores.clear();
}

} // namespace trt_edgellm::rt
