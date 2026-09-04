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

#include "runtime/phase/cost/phaseRuntimeCostTracker.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

namespace trt_edgellm::rt
{

char const* phaseContextualCompletionCalibrationStageName(PhaseContextualCompletionCalibrationStage stage) noexcept
{
    switch (stage)
    {
    case PhaseContextualCompletionCalibrationStage::kPosteriorFit: return "posterior_fit";
    case PhaseContextualCompletionCalibrationStage::kUncertaintyCalibration: return "uncertainty_calibration";
    case PhaseContextualCompletionCalibrationStage::kAuthorityValidation: return "authority_validation";
    case PhaseContextualCompletionCalibrationStage::kComplete: return "complete";
    }
    return "unknown";
}

PhaseRuntimeCostTracker::PhaseRuntimeCostTracker(PhaseRuntimeCostTrackerConfig config)
    : mConfig(config)
    , mActions(config.action)
    , mContextualPd(config.contextualPd)
    , mContextualDp(config.contextualPd)
    , mContextualEp(config.contextualEp)
    , mContextualPe(config.contextualEp)
    , mContextualEd(config.contextualEd)
    , mContextualDe(config.contextualEd)
    , mCompletionPd(config.contextualPd)
    , mCompletionDp(config.contextualPd)
    , mCompletionEp(config.contextualEp)
    , mCompletionPe(config.contextualEp)
    , mCompletionEd(config.contextualEd)
    , mCompletionDe(config.contextualEd)
    , mCompletionPdPair(config.contextualPd)
    , mCompletionEpPair(config.contextualEp)
    , mCompletionEdPair(config.contextualEd)
    , mCompletionPdCalibration(config.completionCalibration)
    , mCompletionEpCalibration(config.completionCalibration)
    , mCompletionEdCalibration(config.completionCalibration)
{
    ELLM_CHECK(mConfig.actionMinimumSamples > 0U, "Runtime action minimum sample count must be positive");
    ELLM_CHECK(mConfig.actionMinimumSamples <= mConfig.action.windowSize,
        "Runtime action sample window must cover the minimum sample count");
    ELLM_CHECK(mConfig.decodeMinimumSamples > 0U, "Runtime decode minimum sample count must be positive");
    ELLM_CHECK(mConfig.decodeWindowSize >= mConfig.decodeMinimumSamples,
        "Runtime decode sample window must cover the minimum sample count");
    ELLM_CHECK(mConfig.decodeContextBucketTokens > 0, "Runtime decode context bucket must be positive");
    ELLM_CHECK(std::isfinite(mConfig.completionDirectionPseudoObservations)
            && mConfig.completionDirectionPseudoObservations > 0.0,
        "Completion direction pseudo-observation count must be finite and positive");
    ELLM_CHECK(std::isfinite(mConfig.completionCalibration.authorityCoverageTolerance)
            && mConfig.completionCalibration.authorityCoverageTolerance >= 0.0
            && mConfig.completionCalibration.authorityCoverageTolerance < 1.0,
        "Completion authority coverage tolerance must be finite and in [0, 1)");
    ELLM_CHECK(mConfig.completionCalibration.authorityMinimumObservations > 0U,
        "Completion authority minimum observations must be positive");
    ELLM_CHECK(mConfig.completionCalibration.authorityMinimumObservations <= mConfig.completionCalibration.windowSize,
        "Completion authority evidence window must cover the minimum observations");
    ELLM_CHECK(std::isfinite(mConfig.completionCalibration.authorityDemotionCoverageTolerance)
            && mConfig.completionCalibration.authorityDemotionCoverageTolerance
                >= mConfig.completionCalibration.authorityCoverageTolerance
            && mConfig.completionCalibration.authorityDemotionCoverageTolerance < 1.0,
        "Completion authority demotion tolerance must cover promotion tolerance and be in [0, 1)");
    ELLM_CHECK(std::isfinite(mConfig.completionCalibration.authorityMaximumFalseSafeRate)
            && mConfig.completionCalibration.authorityMaximumFalseSafeRate >= 0.0
            && mConfig.completionCalibration.authorityMaximumFalseSafeRate <= 1.0,
        "Completion authority false-safe rate must be finite and in [0, 1]");
    ELLM_CHECK(std::isfinite(mConfig.completionCalibration.authorityBlendWeight)
            && mConfig.completionCalibration.authorityBlendWeight >= 0.0
            && mConfig.completionCalibration.authorityBlendWeight <= 1.0,
        "Completion authority blend weight must be finite and in [0, 1]");
}

void PhaseRuntimeCostTracker::observe(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation observation)
{
    mActions.observe(key, observation);
}

std::optional<PhaseGlobalCostEstimate> PhaseRuntimeCostTracker::estimate(PhaseGlobalActionKey const& key) const
{
    return mActions.estimate(key);
}

std::optional<PhaseGlobalCostEstimate> PhaseRuntimeCostTracker::trustedEstimate(PhaseGlobalActionKey const& key) const
{
    std::optional<PhaseGlobalCostEstimate> const estimateValue = mActions.estimate(key);
    return estimateValue.has_value() && estimateValue->sampleCount >= mConfig.actionMinimumSamples ? estimateValue
                                                                                                   : std::nullopt;
}

std::optional<PhaseGlobalCostEstimate> PhaseRuntimeCostTracker::estimateInterpolatedPrimaryBatch(
    PhaseGlobalActionKey const& key) const
{
    return mActions.estimateInterpolatedPrimaryBatch(key);
}

std::optional<PhaseGlobalCostEstimate> PhaseRuntimeCostTracker::estimatePrimaryBatchCoveringContext(
    PhaseGlobalActionKey const& key) const
{
    return mActions.estimatePrimaryBatchCoveringContext(key);
}

std::optional<PhaseGlobalCostEstimate> PhaseRuntimeCostTracker::estimateCoveringPrimary(
    PhaseGlobalActionKey const& key) const
{
    return mActions.estimateCoveringPrimary(key);
}

std::optional<PhaseGlobalCostEstimate> PhaseRuntimeCostTracker::estimatePrimaryLaunchFloor(
    PhaseGlobalActionKey const& key) const
{
    return mActions.estimatePrimaryLaunchFloor(key);
}

std::optional<PhaseGlobalCostEstimate> PhaseRuntimeCostTracker::trustedEstimatePrimaryBatchCoveringContext(
    PhaseGlobalActionKey const& key) const
{
    std::optional<PhaseGlobalCostEstimate> const estimateValue = mActions.estimatePrimaryBatchCoveringContext(key);
    return estimateValue.has_value() && estimateValue->sampleCount >= mConfig.actionMinimumSamples ? estimateValue
                                                                                                   : std::nullopt;
}

std::optional<PhaseGlobalCostEstimate> PhaseRuntimeCostTracker::trustedEstimateCoveringOverlap(
    PhaseGlobalActionKey const& key) const
{
    std::optional<PhaseGlobalCostEstimate> const estimateValue = mActions.estimateCoveringOverlap(key);
    return estimateValue.has_value() && estimateValue->sampleCount >= mConfig.actionMinimumSamples ? estimateValue
                                                                                                   : std::nullopt;
}

PhaseGlobalOverlapCostDiagnostic PhaseRuntimeCostTracker::overlapDiagnostic(PhaseGlobalActionKey const& key) const
{
    return mActions.overlapDiagnostic(key);
}

size_t PhaseRuntimeCostTracker::sampleCount(PhaseGlobalActionKey const& key) const
{
    std::optional<PhaseGlobalCostEstimate> const estimateValue = mActions.estimate(key);
    return estimateValue.has_value() ? estimateValue->sampleCount : 0U;
}

PhaseRuntimeCostConfidence PhaseRuntimeCostTracker::confidence(PhaseGlobalActionKey const& key) const
{
    size_t const samples = sampleCount(key);
    if (samples == 0U)
    {
        return PhaseRuntimeCostConfidence::kUnknown;
    }
    return samples >= mConfig.actionMinimumSamples ? PhaseRuntimeCostConfidence::kReady
                                                   : PhaseRuntimeCostConfidence::kWarming;
}

bool PhaseRuntimeCostTracker::overlapEligible(PhaseGlobalActionKey const& key) const
{
    return mActions.overlapEligible(key);
}

PhaseContextualPdEstimate PhaseRuntimeCostTracker::predictContextualPd(PhaseContextualPdFeatures const& features)
{
    return mContextualPd.predict(features);
}

bool PhaseRuntimeCostTracker::observeContextualPd(
    PhaseContextualPdFeatures const& features, double normalizedAdvantage, double weight)
{
    return mContextualPd.observe(features, normalizedAdvantage, weight);
}

void PhaseRuntimeCostTracker::recordContextualPdSelection(bool overlap, bool exploration) noexcept
{
    mContextualPd.recordSelection(overlap, exploration);
}

PhaseContextualPdModelConfig const& PhaseRuntimeCostTracker::contextualPdConfig() const noexcept
{
    return mContextualPd.config();
}

PhaseContextualPdTelemetry PhaseRuntimeCostTracker::contextualPdTelemetry() const noexcept
{
    return contextualPairTelemetry(PhaseContextualPairKind::kPrefillDecode);
}

PhaseContextualPdEstimate PhaseRuntimeCostTracker::predictContextualPair(
    PhaseContextualPairKind kind, PhaseContextualPdFeatures const& features)
{
    switch (kind)
    {
    case PhaseContextualPairKind::kPrefillDecode: return mContextualPd.predict(features);
    case PhaseContextualPairKind::kEncoderPrefill: return mContextualEp.predict(features);
    case PhaseContextualPairKind::kEncoderDecode: return mContextualEd.predict(features);
    }
    ELLM_CHECK(false, "Unknown contextual pair kind");
}

bool PhaseRuntimeCostTracker::observeContextualPair(
    PhaseContextualPairKind kind, PhaseContextualPdFeatures const& features, double normalizedAdvantage, double weight)
{
    switch (kind)
    {
    case PhaseContextualPairKind::kPrefillDecode: return mContextualPd.observe(features, normalizedAdvantage, weight);
    case PhaseContextualPairKind::kEncoderPrefill: return mContextualEp.observe(features, normalizedAdvantage, weight);
    case PhaseContextualPairKind::kEncoderDecode: return mContextualEd.observe(features, normalizedAdvantage, weight);
    }
    ELLM_CHECK(false, "Unknown contextual pair kind");
}

void PhaseRuntimeCostTracker::recordContextualPairSelection(
    PhaseContextualPairKind kind, bool overlap, bool exploration) noexcept
{
    switch (kind)
    {
    case PhaseContextualPairKind::kPrefillDecode: mContextualPd.recordSelection(overlap, exploration); break;
    case PhaseContextualPairKind::kEncoderPrefill: mContextualEp.recordSelection(overlap, exploration); break;
    case PhaseContextualPairKind::kEncoderDecode: mContextualEd.recordSelection(overlap, exploration); break;
    }
}

PhaseContextualPdModelConfig const& PhaseRuntimeCostTracker::contextualPairConfig(
    PhaseContextualPairKind kind) const noexcept
{
    switch (kind)
    {
    case PhaseContextualPairKind::kPrefillDecode: return mContextualPd.config();
    case PhaseContextualPairKind::kEncoderPrefill: return mContextualEp.config();
    case PhaseContextualPairKind::kEncoderDecode: return mContextualEd.config();
    }
    return mContextualPd.config();
}

namespace
{

size_t completionDirectionIndex(PhaseContextualPairDirection direction) noexcept
{
    switch (direction)
    {
    case PhaseContextualPairDirection::kPrefillToDecode: return 0U;
    case PhaseContextualPairDirection::kDecodeToPrefill: return 1U;
    case PhaseContextualPairDirection::kEncoderToPrefill: return 2U;
    case PhaseContextualPairDirection::kPrefillToEncoder: return 3U;
    case PhaseContextualPairDirection::kEncoderToDecode: return 4U;
    case PhaseContextualPairDirection::kDecodeToEncoder: return 5U;
    }
    return 0U;
}

bool canonicalPrimaryIsIncumbent(PhaseContextualPairDirection direction) noexcept
{
    return direction == PhaseContextualPairDirection::kPrefillToDecode
        || direction == PhaseContextualPairDirection::kEncoderToPrefill
        || direction == PhaseContextualPairDirection::kEncoderToDecode;
}

PhaseContextualCompletionEstimate swapCompletionComponents(PhaseContextualCompletionEstimate value) noexcept
{
    std::swap(value.incumbentMeanUs, value.newcomerMeanUs);
    std::swap(value.incumbentUncertaintyUs, value.newcomerUncertaintyUs);
    return value;
}

void observeHierarchicalCompletionTelemetry(PhaseContextualCompletionTelemetry& telemetry,
    PhaseContextualCompletionEstimate const& prediction, double confidenceBeta, double incumbentCompletionUs,
    double newcomerCompletionUs, double minimumSlackUs) noexcept
{
    double const incumbentErrorUs = incumbentCompletionUs - prediction.incumbentMeanUs;
    double const newcomerErrorUs = newcomerCompletionUs - prediction.newcomerMeanUs;
    bool const incumbentCovered = std::abs(incumbentErrorUs) <= confidenceBeta * prediction.incumbentUncertaintyUs;
    bool const newcomerCovered = std::abs(newcomerErrorUs) <= confidenceBeta * prediction.newcomerUncertaintyUs;
    double const predictedRobustUs
        = std::max(prediction.incumbentMeanUs + confidenceBeta * prediction.incumbentUncertaintyUs,
            prediction.newcomerMeanUs + confidenceBeta * prediction.newcomerUncertaintyUs);
    bool const predictedSafe = std::isfinite(minimumSlackUs) && predictedRobustUs <= minimumSlackUs;
    ++telemetry.observations;
    telemetry.readyCalibrationObservations += prediction.ready ? 1U : 0U;
    telemetry.incumbentIntervalCovered += incumbentCovered ? 1U : 0U;
    telemetry.newcomerIntervalCovered += newcomerCovered ? 1U : 0U;
    telemetry.readyIncumbentIntervalCovered += prediction.ready && incumbentCovered ? 1U : 0U;
    telemetry.readyNewcomerIntervalCovered += prediction.ready && newcomerCovered ? 1U : 0U;
    telemetry.conformalCalibrationObservations += prediction.uncertaintyCalibrated ? 1U : 0U;
    telemetry.conformalIncumbentIntervalCovered += prediction.uncertaintyCalibrated && incumbentCovered ? 1U : 0U;
    telemetry.conformalNewcomerIntervalCovered += prediction.uncertaintyCalibrated && newcomerCovered ? 1U : 0U;
    telemetry.predictedSafeObservations += predictedSafe ? 1U : 0U;
    telemetry.falseSafeObservations
        += predictedSafe && std::max(incumbentCompletionUs, newcomerCompletionUs) > minimumSlackUs ? 1U : 0U;
    telemetry.conformalPredictedSafeObservations += prediction.uncertaintyCalibrated && predictedSafe ? 1U : 0U;
    telemetry.conformalFalseSafeObservations += prediction.uncertaintyCalibrated && predictedSafe
            && std::max(incumbentCompletionUs, newcomerCompletionUs) > minimumSlackUs
        ? 1U
        : 0U;
    telemetry.incumbentAbsoluteErrorUs += std::abs(incumbentErrorUs);
    telemetry.incumbentSquaredErrorUs += incumbentErrorUs * incumbentErrorUs;
    telemetry.newcomerAbsoluteErrorUs += std::abs(newcomerErrorUs);
    telemetry.newcomerSquaredErrorUs += newcomerErrorUs * newcomerErrorUs;
    telemetry.readyIncumbentAbsoluteErrorUs += prediction.ready ? std::abs(incumbentErrorUs) : 0.0;
    telemetry.readyIncumbentSquaredErrorUs += prediction.ready ? incumbentErrorUs * incumbentErrorUs : 0.0;
    telemetry.readyNewcomerAbsoluteErrorUs += prediction.ready ? std::abs(newcomerErrorUs) : 0.0;
    telemetry.readyNewcomerSquaredErrorUs += prediction.ready ? newcomerErrorUs * newcomerErrorUs : 0.0;
}

PhaseContextualPdTelemetry mergeContextualTelemetry(
    PhaseContextualPdTelemetry left, PhaseContextualPdTelemetry const& right) noexcept
{
    left.predictions += right.predictions;
    left.observations += right.observations;
    left.rejectedObservations += right.rejectedObservations;
    left.positiveSelections += right.positiveSelections;
    left.negativeSelections += right.negativeSelections;
    left.explorations += right.explorations;
    left.calibrationObservations += right.calibrationObservations;
    left.readyCalibrationObservations += right.readyCalibrationObservations;
    left.confidenceIntervalCovered += right.confidenceIntervalCovered;
    left.readyConfidenceIntervalCovered += right.readyConfidenceIntervalCovered;
    left.predictedSafeObservations += right.predictedSafeObservations;
    left.falseSafeObservations += right.falseSafeObservations;
    left.absoluteErrorSum += right.absoluteErrorSum;
    left.squaredErrorSum += right.squaredErrorSum;
    left.readyAbsoluteErrorSum += right.readyAbsoluteErrorSum;
    left.readySquaredErrorSum += right.readySquaredErrorSum;
    if (right.observations > 0U)
    {
        left.lastReward = right.lastReward;
        left.lastMean = right.lastMean;
        left.lastUncertainty = right.lastUncertainty;
        left.lastLowerConfidenceBound = right.lastLowerConfidenceBound;
        left.lastPredictionError = right.lastPredictionError;
    }
    return left;
}

} // namespace

PhaseContextualPdTelemetry PhaseRuntimeCostTracker::contextualPairTelemetry(PhaseContextualPairKind kind) const noexcept
{
    switch (kind)
    {
    case PhaseContextualPairKind::kPrefillDecode:
        return mergeContextualTelemetry(mContextualPd.telemetry(), mContextualDp.telemetry());
    case PhaseContextualPairKind::kEncoderPrefill:
        return mergeContextualTelemetry(mContextualEp.telemetry(), mContextualPe.telemetry());
    case PhaseContextualPairKind::kEncoderDecode:
        return mergeContextualTelemetry(mContextualEd.telemetry(), mContextualDe.telemetry());
    }
    return mContextualPd.telemetry();
}

PhaseContextualPdEstimate PhaseRuntimeCostTracker::predictContextualDirection(
    PhaseContextualPairDirection direction, PhaseContextualPdFeatures const& features)
{
    switch (direction)
    {
    case PhaseContextualPairDirection::kPrefillToDecode: return mContextualPd.predict(features);
    case PhaseContextualPairDirection::kDecodeToPrefill: return mContextualDp.predict(features);
    case PhaseContextualPairDirection::kEncoderToPrefill: return mContextualEp.predict(features);
    case PhaseContextualPairDirection::kPrefillToEncoder: return mContextualPe.predict(features);
    case PhaseContextualPairDirection::kEncoderToDecode: return mContextualEd.predict(features);
    case PhaseContextualPairDirection::kDecodeToEncoder: return mContextualDe.predict(features);
    }
    ELLM_CHECK(false, "Unknown contextual pair direction");
}

bool PhaseRuntimeCostTracker::observeContextualDirection(PhaseContextualPairDirection direction,
    PhaseContextualPdFeatures const& features, double normalizedAdvantage, double weight)
{
    switch (direction)
    {
    case PhaseContextualPairDirection::kPrefillToDecode:
        return mContextualPd.observe(features, normalizedAdvantage, weight);
    case PhaseContextualPairDirection::kDecodeToPrefill:
        return mContextualDp.observe(features, normalizedAdvantage, weight);
    case PhaseContextualPairDirection::kEncoderToPrefill:
        return mContextualEp.observe(features, normalizedAdvantage, weight);
    case PhaseContextualPairDirection::kPrefillToEncoder:
        return mContextualPe.observe(features, normalizedAdvantage, weight);
    case PhaseContextualPairDirection::kEncoderToDecode:
        return mContextualEd.observe(features, normalizedAdvantage, weight);
    case PhaseContextualPairDirection::kDecodeToEncoder:
        return mContextualDe.observe(features, normalizedAdvantage, weight);
    }
    ELLM_CHECK(false, "Unknown contextual pair direction");
}

void PhaseRuntimeCostTracker::recordContextualDirectionSelection(
    PhaseContextualPairDirection direction, bool overlap, bool exploration) noexcept
{
    switch (direction)
    {
    case PhaseContextualPairDirection::kPrefillToDecode: mContextualPd.recordSelection(overlap, exploration); break;
    case PhaseContextualPairDirection::kDecodeToPrefill: mContextualDp.recordSelection(overlap, exploration); break;
    case PhaseContextualPairDirection::kEncoderToPrefill: mContextualEp.recordSelection(overlap, exploration); break;
    case PhaseContextualPairDirection::kPrefillToEncoder: mContextualPe.recordSelection(overlap, exploration); break;
    case PhaseContextualPairDirection::kEncoderToDecode: mContextualEd.recordSelection(overlap, exploration); break;
    case PhaseContextualPairDirection::kDecodeToEncoder: mContextualDe.recordSelection(overlap, exploration); break;
    }
}

PhaseContextualPdTelemetry const& PhaseRuntimeCostTracker::contextualDirectionTelemetry(
    PhaseContextualPairDirection direction) const noexcept
{
    switch (direction)
    {
    case PhaseContextualPairDirection::kPrefillToDecode: return mContextualPd.telemetry();
    case PhaseContextualPairDirection::kDecodeToPrefill: return mContextualDp.telemetry();
    case PhaseContextualPairDirection::kEncoderToPrefill: return mContextualEp.telemetry();
    case PhaseContextualPairDirection::kPrefillToEncoder: return mContextualPe.telemetry();
    case PhaseContextualPairDirection::kEncoderToDecode: return mContextualEd.telemetry();
    case PhaseContextualPairDirection::kDecodeToEncoder: return mContextualDe.telemetry();
    }
    return mContextualPd.telemetry();
}

PhaseContextualCompletionEstimate PhaseRuntimeCostTracker::predictContextualCompletionDirection(
    PhaseContextualPairDirection direction, PhaseContextualPdFeatures const& features, double incumbentReferenceUs,
    double newcomerReferenceUs)
{
    PhaseContextualPdFeatures const policyFeatures = contextualCompletionFeaturesForPolicy(features);
    PhaseContextualCompletionModel* pair{};
    PhaseContextualCompletionModel* ordered{};
    switch (direction)
    {
    case PhaseContextualPairDirection::kPrefillToDecode:
        pair = &mCompletionPdPair;
        ordered = &mCompletionPd;
        break;
    case PhaseContextualPairDirection::kDecodeToPrefill:
        pair = &mCompletionPdPair;
        ordered = &mCompletionDp;
        break;
    case PhaseContextualPairDirection::kEncoderToPrefill:
        pair = &mCompletionEpPair;
        ordered = &mCompletionEp;
        break;
    case PhaseContextualPairDirection::kPrefillToEncoder:
        pair = &mCompletionEpPair;
        ordered = &mCompletionPe;
        break;
    case PhaseContextualPairDirection::kEncoderToDecode:
        pair = &mCompletionEdPair;
        ordered = &mCompletionEd;
        break;
    case PhaseContextualPairDirection::kDecodeToEncoder:
        pair = &mCompletionEdPair;
        ordered = &mCompletionDe;
        break;
    }
    ELLM_CHECK(pair != nullptr && ordered != nullptr, "Unknown contextual completion direction");
    bool const primaryIsIncumbent = canonicalPrimaryIsIncumbent(direction);
    double const primaryReferenceUs = primaryIsIncumbent ? incumbentReferenceUs : newcomerReferenceUs;
    double const secondaryReferenceUs = primaryIsIncumbent ? newcomerReferenceUs : incumbentReferenceUs;
    PhaseContextualCompletionEstimate pairEstimate
        = pair->predict(policyFeatures, primaryReferenceUs, secondaryReferenceUs);
    if (!primaryIsIncumbent)
    {
        pairEstimate = swapCompletionComponents(pairEstimate);
    }
    PhaseContextualCompletionEstimate const directionEstimate
        = ordered->predict(policyFeatures, incumbentReferenceUs, newcomerReferenceUs);
    ++mCompletionHierarchicalTelemetry[completionDirectionIndex(direction)].predictions;
    PhaseContextualCompletionEstimate const raw = phaseBlendContextualCompletionEstimates(
        pairEstimate, directionEstimate, mConfig.completionDirectionPseudoObservations);
    switch (phaseContextualPairKind(direction))
    {
    case PhaseContextualPairKind::kPrefillDecode: return mCompletionPdCalibration.apply(raw);
    case PhaseContextualPairKind::kEncoderPrefill: return mCompletionEpCalibration.apply(raw);
    case PhaseContextualPairKind::kEncoderDecode: return mCompletionEdCalibration.apply(raw);
    }
    return raw;
}

bool PhaseRuntimeCostTracker::observeContextualCompletionDirection(PhaseContextualPairDirection direction,
    PhaseContextualPdFeatures const& features, double incumbentReferenceUs, double newcomerReferenceUs,
    double incumbentCompletionUs, double newcomerCompletionUs, double minimumSlackUs, double scalarDecisionMakespanUs)
{
    // Keep hierarchical calibration on the exact admissibility boundary used
    // by both underlying completion models. Counting a rejected label in the
    // wrapper would make pair and direction coverage incomparable.
    if (!std::isfinite(incumbentReferenceUs) || incumbentReferenceUs <= 0.0 || !std::isfinite(newcomerReferenceUs)
        || newcomerReferenceUs <= 0.0 || !std::isfinite(incumbentCompletionUs) || incumbentCompletionUs < 0.0
        || !std::isfinite(newcomerCompletionUs) || newcomerCompletionUs < 0.0)
    {
        return false;
    }
    PhaseContextualCompletionModel* pair{};
    PhaseContextualCompletionModel* ordered{};
    PhaseContextualCompletionCalibrator* calibrator{};
    switch (direction)
    {
    case PhaseContextualPairDirection::kPrefillToDecode:
        pair = &mCompletionPdPair;
        ordered = &mCompletionPd;
        calibrator = &mCompletionPdCalibration;
        break;
    case PhaseContextualPairDirection::kDecodeToPrefill:
        pair = &mCompletionPdPair;
        ordered = &mCompletionDp;
        calibrator = &mCompletionPdCalibration;
        break;
    case PhaseContextualPairDirection::kEncoderToPrefill:
        pair = &mCompletionEpPair;
        ordered = &mCompletionEp;
        calibrator = &mCompletionEpCalibration;
        break;
    case PhaseContextualPairDirection::kPrefillToEncoder:
        pair = &mCompletionEpPair;
        ordered = &mCompletionPe;
        calibrator = &mCompletionEpCalibration;
        break;
    case PhaseContextualPairDirection::kEncoderToDecode:
        pair = &mCompletionEdPair;
        ordered = &mCompletionEd;
        calibrator = &mCompletionEdCalibration;
        break;
    case PhaseContextualPairDirection::kDecodeToEncoder:
        pair = &mCompletionEdPair;
        ordered = &mCompletionDe;
        calibrator = &mCompletionEdCalibration;
        break;
    }
    ELLM_CHECK(
        pair != nullptr && ordered != nullptr && calibrator != nullptr, "Unknown contextual completion direction");
    PhaseContextualPdFeatures const policyFeatures = contextualCompletionFeaturesForPolicy(features);
    PhaseContextualCompletionEstimate const prediction
        = predictContextualCompletionDirection(direction, policyFeatures, incumbentReferenceUs, newcomerReferenceUs);
    PhaseContextualCompletionTelemetry& telemetry
        = mCompletionHierarchicalTelemetry[completionDirectionIndex(direction)];
    // The prediction above is part of calibration, not candidate generation.
    --telemetry.predictions;
    observeHierarchicalCompletionTelemetry(telemetry, prediction,
        contextualPairConfig(phaseContextualPairKind(direction)).confidenceBeta, incumbentCompletionUs,
        newcomerCompletionUs, minimumSlackUs);
    observeContextualCompletionAuthorityEvidence(direction, prediction, policyFeatures, incumbentReferenceUs,
        newcomerReferenceUs, incumbentCompletionUs, newcomerCompletionUs, minimumSlackUs, scalarDecisionMakespanUs);
    PhaseContextualCompletionEstimate rawPrediction = prediction;
    rawPrediction.incumbentUncertaintyUs /= prediction.uncertaintyScale;
    rawPrediction.newcomerUncertaintyUs /= prediction.uncertaintyScale;
    rawPrediction.uncertaintyScale = 1.0;
    rawPrediction.uncertaintyCalibrationObservations = 0U;
    rawPrediction.uncertaintyCalibrated = false;
    static_cast<void>(
        calibrator->observe(rawPrediction, contextualPairConfig(phaseContextualPairKind(direction)).confidenceBeta,
            incumbentCompletionUs, newcomerCompletionUs));

    bool const primaryIsIncumbent = canonicalPrimaryIsIncumbent(direction);
    double const primaryReferenceUs = primaryIsIncumbent ? incumbentReferenceUs : newcomerReferenceUs;
    double const secondaryReferenceUs = primaryIsIncumbent ? newcomerReferenceUs : incumbentReferenceUs;
    double const primaryCompletionUs = primaryIsIncumbent ? incumbentCompletionUs : newcomerCompletionUs;
    double const secondaryCompletionUs = primaryIsIncumbent ? newcomerCompletionUs : incumbentCompletionUs;
    bool const pairObserved = pair->observe(policyFeatures, primaryReferenceUs, secondaryReferenceUs,
        primaryCompletionUs, secondaryCompletionUs, minimumSlackUs);
    bool const directionObserved = ordered->observe(policyFeatures, incumbentReferenceUs, newcomerReferenceUs,
        incumbentCompletionUs, newcomerCompletionUs, minimumSlackUs);
    return pairObserved && directionObserved;
}

PhaseContextualPdFeatures PhaseRuntimeCostTracker::contextualCompletionFeaturesForPolicy(
    PhaseContextualPdFeatures features) const noexcept
{
    if (!mConfig.completionCalibration.useResidualFeatures)
    {
        // Completion V2 indices 9--15 encode residual mode, incumbent age,
        // elapsed/reference ratio, requested skew, and the outstanding mask.
        // The first nine shape/cost/slack features remain identical.
        std::fill(features.begin() + 9, features.end(), 0.0);
    }
    return features;
}

PhaseContextualCompletionEstimate PhaseRuntimeCostTracker::contextualCompletionAuthorityEstimate(
    PhaseContextualPairDirection direction, PhaseContextualCompletionEstimate estimate) const noexcept
{
    if (mConfig.completionCalibration.authorityUsesUncertainty)
    {
        double const beta = contextualPairConfig(phaseContextualPairKind(direction)).confidenceBeta;
        estimate.incumbentUncertaintyUs *= beta;
        estimate.newcomerUncertaintyUs *= beta;
    }
    else
    {
        estimate.incumbentUncertaintyUs = 0.0;
        estimate.newcomerUncertaintyUs = 0.0;
    }
    return estimate;
}

bool PhaseRuntimeCostTracker::contextualCompletionAuthorityEvidenceReady(
    PhaseContextualPairDirection direction) const noexcept
{
    if (!contextualCompletionAuthorityEnabled())
    {
        return false;
    }
    return contextualCompletionAuthorityEvidence(direction).validated;
}

bool PhaseRuntimeCostTracker::contextualCompletionAuthorityEvidenceComplete(
    PhaseContextualPairDirection direction) const noexcept
{
    if (!contextualCompletionAuthorityEnabled())
    {
        return false;
    }
    return contextualCompletionAuthorityEvidence(direction).observations
        >= mConfig.completionCalibration.authorityMinimumObservations;
}

bool PhaseRuntimeCostTracker::contextualCompletionAuthorityEvidenceSatisfies(
    PhaseContextualCompletionAuthorityEvidence const& evidence, double coverageTolerance) const noexcept
{
    size_t const minimum = mConfig.completionCalibration.authorityMinimumObservations;
    if (evidence.observations < minimum)
    {
        return false;
    }
    double const incumbentCoverage
        = static_cast<double>(evidence.incumbentIntervalCovered) / static_cast<double>(evidence.observations);
    double const newcomerCoverage
        = static_cast<double>(evidence.newcomerIntervalCovered) / static_cast<double>(evidence.observations);
    double const minimumCoverage = std::max(0.0, mConfig.completionCalibration.targetCoverage - coverageTolerance);
    if (incumbentCoverage < minimumCoverage || newcomerCoverage < minimumCoverage)
    {
        return false;
    }
    double const falseSafeRate = evidence.predictedSafeObservations > 0U
        ? static_cast<double>(evidence.falseSafeObservations) / static_cast<double>(evidence.predictedSafeObservations)
        : 0.0;
    return falseSafeRate <= mConfig.completionCalibration.authorityMaximumFalseSafeRate;
}

PhaseContextualCompletionAuthorityEvidence PhaseRuntimeCostTracker::contextualCompletionAuthorityEvidence(
    PhaseContextualPairDirection direction) const noexcept
{
    return mCompletionAuthorityEvidence[completionDirectionIndex(direction)].aggregate;
}

void PhaseRuntimeCostTracker::observeContextualCompletionAuthorityEvidence(PhaseContextualPairDirection direction,
    PhaseContextualCompletionEstimate const& prediction, PhaseContextualPdFeatures const& features,
    double incumbentReferenceUs, double newcomerReferenceUs, double incumbentCompletionUs, double newcomerCompletionUs,
    double minimumSlackUs, double scalarDecisionMakespanUs) noexcept
{
    if (!prediction.uncertaintyCalibrated)
    {
        return;
    }
    double const beta = contextualPairConfig(phaseContextualPairKind(direction)).confidenceBeta;
    double const incumbentRobustUs = prediction.incumbentMeanUs + beta * prediction.incumbentUncertaintyUs;
    double const newcomerRobustUs = prediction.newcomerMeanUs + beta * prediction.newcomerUncertaintyUs;
    double const actualMakespanUs = std::max(incumbentCompletionUs, newcomerCompletionUs);
    double const completionMakespanUs = phaseContextualCompletionDecisionMakespanUs(prediction, features);
    double const scalarMakespanUs = std::isfinite(scalarDecisionMakespanUs)
        ? std::max(0.0, scalarDecisionMakespanUs)
        : std::max(incumbentReferenceUs, newcomerReferenceUs);
    CompletionAuthorityEvidenceSample const sample{
        std::abs(incumbentCompletionUs - prediction.incumbentMeanUs) <= beta * prediction.incumbentUncertaintyUs,
        std::abs(newcomerCompletionUs - prediction.newcomerMeanUs) <= beta * prediction.newcomerUncertaintyUs,
        std::isfinite(minimumSlackUs) && std::max(incumbentRobustUs, newcomerRobustUs) <= minimumSlackUs,
        std::isfinite(minimumSlackUs) && std::max(incumbentRobustUs, newcomerRobustUs) <= minimumSlackUs
            && std::max(incumbentCompletionUs, newcomerCompletionUs) > minimumSlackUs,
        std::abs(actualMakespanUs - completionMakespanUs), std::abs(actualMakespanUs - scalarMakespanUs),
        std::abs(incumbentCompletionUs - prediction.incumbentMeanUs),
        std::abs(incumbentCompletionUs - incumbentReferenceUs),
        std::abs(newcomerCompletionUs - prediction.newcomerMeanUs),
        std::abs(newcomerCompletionUs - newcomerReferenceUs)};
    CompletionAuthorityEvidenceWindow& window = mCompletionAuthorityEvidence[completionDirectionIndex(direction)];
    auto add
        = [](PhaseContextualCompletionAuthorityEvidence& aggregate, CompletionAuthorityEvidenceSample const& value) {
              ++aggregate.observations;
              aggregate.incumbentIntervalCovered += value.incumbentCovered ? 1U : 0U;
              aggregate.newcomerIntervalCovered += value.newcomerCovered ? 1U : 0U;
              aggregate.predictedSafeObservations += value.predictedSafe ? 1U : 0U;
              aggregate.falseSafeObservations += value.falseSafe ? 1U : 0U;
              aggregate.completionAbsoluteErrorUs += value.completionAbsoluteErrorUs;
              aggregate.referenceAbsoluteErrorUs += value.referenceAbsoluteErrorUs;
              aggregate.incumbentCompletionAbsoluteErrorUs += value.incumbentCompletionAbsoluteErrorUs;
              aggregate.incumbentReferenceAbsoluteErrorUs += value.incumbentReferenceAbsoluteErrorUs;
              aggregate.newcomerCompletionAbsoluteErrorUs += value.newcomerCompletionAbsoluteErrorUs;
              aggregate.newcomerReferenceAbsoluteErrorUs += value.newcomerReferenceAbsoluteErrorUs;
          };
    auto remove
        = [](PhaseContextualCompletionAuthorityEvidence& aggregate, CompletionAuthorityEvidenceSample const& value) {
              --aggregate.observations;
              aggregate.incumbentIntervalCovered -= value.incumbentCovered ? 1U : 0U;
              aggregate.newcomerIntervalCovered -= value.newcomerCovered ? 1U : 0U;
              aggregate.predictedSafeObservations -= value.predictedSafe ? 1U : 0U;
              aggregate.falseSafeObservations -= value.falseSafe ? 1U : 0U;
              aggregate.completionAbsoluteErrorUs -= value.completionAbsoluteErrorUs;
              aggregate.referenceAbsoluteErrorUs -= value.referenceAbsoluteErrorUs;
              aggregate.incumbentCompletionAbsoluteErrorUs -= value.incumbentCompletionAbsoluteErrorUs;
              aggregate.incumbentReferenceAbsoluteErrorUs -= value.incumbentReferenceAbsoluteErrorUs;
              aggregate.newcomerCompletionAbsoluteErrorUs -= value.newcomerCompletionAbsoluteErrorUs;
              aggregate.newcomerReferenceAbsoluteErrorUs -= value.newcomerReferenceAbsoluteErrorUs;
          };
    if (window.samples.size() == mConfig.completionCalibration.windowSize)
    {
        remove(window.aggregate, window.samples.front());
        window.samples.pop_front();
    }
    window.samples.push_back(sample);
    add(window.aggregate, sample);
    bool const promotionReady = contextualCompletionAuthorityEvidenceSatisfies(
        window.aggregate, mConfig.completionCalibration.authorityCoverageTolerance);
    bool const retainAuthority = contextualCompletionAuthorityEvidenceSatisfies(
        window.aggregate, mConfig.completionCalibration.authorityDemotionCoverageTolerance);
    if (!window.aggregate.validated && promotionReady)
    {
        window.aggregate.validated = true;
        ++window.aggregate.promotions;
    }
    else if (window.aggregate.validated && !retainAuthority)
    {
        window.aggregate.validated = false;
        ++window.aggregate.demotions;
    }
}

double PhaseRuntimeCostTracker::contextualCompletionAuthorityBlendWeight(
    PhaseContextualPairDirection direction) const noexcept
{
    PhaseContextualCompletionAuthorityEvidence const evidence = contextualCompletionAuthorityEvidence(direction);
    if (!evidence.validated || evidence.referenceAbsoluteErrorUs <= std::numeric_limits<double>::epsilon())
    {
        return 0.0;
    }
    double const relativeImprovement
        = (evidence.referenceAbsoluteErrorUs - evidence.completionAbsoluteErrorUs) / evidence.referenceAbsoluteErrorUs;
    size_t const minimum = mConfig.completionCalibration.authorityMinimumObservations;
    double const maturity = evidence.observations > minimum
        ? static_cast<double>(evidence.observations - minimum) / static_cast<double>(evidence.observations)
        : 0.0;
    return mConfig.completionCalibration.authorityBlendWeight * maturity * std::clamp(relativeImprovement, 0.0, 1.0);
}

double PhaseRuntimeCostTracker::contextualCompletionAuthorityComponentBlendWeight(
    PhaseContextualPairDirection direction, bool incumbent) const noexcept
{
    PhaseContextualCompletionAuthorityEvidence const evidence = contextualCompletionAuthorityEvidence(direction);
    double const completionError
        = incumbent ? evidence.incumbentCompletionAbsoluteErrorUs : evidence.newcomerCompletionAbsoluteErrorUs;
    double const referenceError
        = incumbent ? evidence.incumbentReferenceAbsoluteErrorUs : evidence.newcomerReferenceAbsoluteErrorUs;
    if (!evidence.validated || referenceError <= std::numeric_limits<double>::epsilon())
    {
        return 0.0;
    }
    double const relativeImprovement = (referenceError - completionError) / referenceError;
    size_t const minimum = mConfig.completionCalibration.authorityMinimumObservations;
    double const maturity = evidence.observations > minimum
        ? static_cast<double>(evidence.observations - minimum) / static_cast<double>(evidence.observations)
        : 0.0;
    return mConfig.completionCalibration.authorityBlendWeight * maturity * std::clamp(relativeImprovement, 0.0, 1.0);
}

bool PhaseRuntimeCostTracker::contextualCompletionAuthorityReady(
    PhaseContextualPairDirection direction, PhaseContextualCompletionEstimate const& estimate) const noexcept
{
    return estimate.ready && estimate.uncertaintyCalibrated && contextualCompletionAuthorityEvidenceReady(direction);
}

PhaseContextualCompletionTelemetry const& PhaseRuntimeCostTracker::contextualCompletionDirectionTelemetry(
    PhaseContextualPairDirection direction) const noexcept
{
    return mCompletionHierarchicalTelemetry[completionDirectionIndex(direction)];
}

PhaseContextualCompletionTelemetry const& PhaseRuntimeCostTracker::contextualCompletionPairTelemetry(
    PhaseContextualPairKind kind) const noexcept
{
    switch (kind)
    {
    case PhaseContextualPairKind::kPrefillDecode: return mCompletionPdPair.telemetry();
    case PhaseContextualPairKind::kEncoderPrefill: return mCompletionEpPair.telemetry();
    case PhaseContextualPairKind::kEncoderDecode: return mCompletionEdPair.telemetry();
    }
    return mCompletionPdPair.telemetry();
}

PhaseContextualCompletionCalibrationEstimate PhaseRuntimeCostTracker::contextualCompletionCalibration(
    PhaseContextualPairKind kind) const
{
    switch (kind)
    {
    case PhaseContextualPairKind::kPrefillDecode: return mCompletionPdCalibration.estimate();
    case PhaseContextualPairKind::kEncoderPrefill: return mCompletionEpCalibration.estimate();
    case PhaseContextualPairKind::kEncoderDecode: return mCompletionEdCalibration.estimate();
    }
    return mCompletionPdCalibration.estimate();
}

PhaseContextualCompletionCalibrationProgress PhaseRuntimeCostTracker::contextualCompletionCalibrationProgress(
    PhaseContextualPairDirection direction) const noexcept
{
    PhaseContextualPairKind const kind = phaseContextualPairKind(direction);
    size_t const posteriorObservations = contextualCompletionDirectionTelemetry(direction).observations;
    size_t const posteriorMinimumObservations = contextualPairConfig(kind).minimumObservations;
    PhaseContextualCompletionCalibrationEstimate const uncertainty = contextualCompletionCalibration(kind);
    size_t const authorityObservations = contextualCompletionAuthorityEvidence(direction).observations;
    PhaseContextualCompletionCalibrationProgress progress{PhaseContextualCompletionCalibrationStage::kComplete,
        posteriorObservations, posteriorMinimumObservations, uncertainty.observations,
        mConfig.completionCalibration.minimumObservations, authorityObservations,
        mConfig.completionCalibration.authorityMinimumObservations};
    if (posteriorObservations < posteriorMinimumObservations)
    {
        progress.stage = PhaseContextualCompletionCalibrationStage::kPosteriorFit;
    }
    else if (!uncertainty.ready)
    {
        progress.stage = PhaseContextualCompletionCalibrationStage::kUncertaintyCalibration;
    }
    else if (authorityObservations < mConfig.completionCalibration.authorityMinimumObservations)
    {
        progress.stage = PhaseContextualCompletionCalibrationStage::kAuthorityValidation;
    }
    return progress;
}

void PhaseRuntimeCostTracker::observeDecode(
    int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive, float gpuMs)
{
    ELLM_CHECK(batchSize > 0, "Runtime decode observation requires a positive batch size");
    ELLM_CHECK(maxContextLength >= 0, "Runtime decode observation requires a non-negative context length");
    ELLM_CHECK(std::isfinite(gpuMs) && gpuMs > 0.0F, "Runtime decode observation must be finite and positive");
    std::deque<float>& samples
        = mDecodeComponents[decodeKey(batchSize, maxContextLength, encoderActive, prefillActive)];
    samples.push_back(gpuMs);
    while (samples.size() > mConfig.decodeWindowSize)
    {
        samples.pop_front();
    }
}

std::optional<float> PhaseRuntimeCostTracker::decodeP95(
    int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive) const
{
    auto const found = mDecodeComponents.find(decodeKey(batchSize, maxContextLength, encoderActive, prefillActive));
    if (found == mDecodeComponents.end() || found->second.size() < mConfig.decodeMinimumSamples)
    {
        return std::nullopt;
    }
    std::vector<float> ordered(found->second.begin(), found->second.end());
    std::sort(ordered.begin(), ordered.end());
    size_t const p95Index = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(ordered.size()))) - 1U;
    return ordered[p95Index];
}

std::optional<float> PhaseRuntimeCostTracker::decodeCoveringP95(
    int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive) const
{
    DecodeKey const target = decodeKey(batchSize, maxContextLength, encoderActive, prefillActive);
    struct Cover
    {
        DecodeKey key;
        float p95{};
    };
    std::vector<Cover> covers;
    for (auto const& [key, samples] : mDecodeComponents)
    {
        if (key.encoderActive != target.encoderActive || key.prefillActive != target.prefillActive
            || key.batchSize < target.batchSize || key.contextBucket < target.contextBucket
            || samples.size() < mConfig.decodeMinimumSamples)
        {
            continue;
        }
        std::vector<float> ordered(samples.begin(), samples.end());
        std::sort(ordered.begin(), ordered.end());
        size_t const p95Index = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(ordered.size()))) - 1U;
        covers.push_back({key, ordered[p95Index]});
    }

    std::optional<float> result;
    for (Cover const& cover : covers)
    {
        bool dominated{};
        for (Cover const& other : covers)
        {
            bool const noLarger
                = other.key.batchSize <= cover.key.batchSize && other.key.contextBucket <= cover.key.contextBucket;
            bool const strictlySmaller
                = other.key.batchSize < cover.key.batchSize || other.key.contextBucket < cover.key.contextBucket;
            if (&cover != &other && noLarger && strictlySmaller)
            {
                dominated = true;
                break;
            }
        }
        if (!dominated)
        {
            result = std::max(result.value_or(0.0F), cover.p95);
        }
    }
    return result;
}

size_t PhaseRuntimeCostTracker::decodeBucketCount() const noexcept
{
    return mDecodeComponents.size();
}

void PhaseRuntimeCostTracker::resetExecutionCostHistory()
{
    mActions.reset();
    mDecodeComponents.clear();
}

void PhaseRuntimeCostTracker::resetPolicyPosterior()
{
    mContextualPd.reset();
    mContextualDp.reset();
    mContextualEp.reset();
    mContextualPe.reset();
    mContextualEd.reset();
    mContextualDe.reset();
    mCompletionPd.reset();
    mCompletionDp.reset();
    mCompletionEp.reset();
    mCompletionPe.reset();
    mCompletionEd.reset();
    mCompletionDe.reset();
    mCompletionPdPair.reset();
    mCompletionEpPair.reset();
    mCompletionEdPair.reset();
    mCompletionPdCalibration.reset();
    mCompletionEpCalibration.reset();
    mCompletionEdCalibration.reset();
    mCompletionHierarchicalTelemetry = {};
    mCompletionAuthorityEvidence = {};
}

void PhaseRuntimeCostTracker::resetCompletionAuthorityEvidence() noexcept
{
    mCompletionAuthorityEvidence = {};
}

void PhaseRuntimeCostTracker::reset()
{
    resetExecutionCostHistory();
    resetPolicyPosterior();
}

bool PhaseRuntimeCostTracker::DecodeKey::operator==(DecodeKey const& other) const noexcept
{
    return batchSize == other.batchSize && contextBucket == other.contextBucket && encoderActive == other.encoderActive
        && prefillActive == other.prefillActive;
}

size_t PhaseRuntimeCostTracker::DecodeKeyHash::operator()(DecodeKey const& key) const noexcept
{
    size_t result = std::hash<int32_t>{}(key.batchSize);
    auto combine = [&result](size_t value) { result ^= value + 0x9e3779b9U + (result << 6U) + (result >> 2U); };
    combine(std::hash<int32_t>{}(key.contextBucket));
    combine(std::hash<bool>{}(key.encoderActive));
    combine(std::hash<bool>{}(key.prefillActive));
    return result;
}

PhaseRuntimeCostTracker::DecodeKey PhaseRuntimeCostTracker::decodeKey(
    int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive) const noexcept
{
    int32_t const contextBucket
        = std::max(1, (maxContextLength + mConfig.decodeContextBucketTokens - 1) / mConfig.decodeContextBucketTokens);
    return {batchSize, contextBucket, encoderActive, prefillActive};
}

} // namespace trt_edgellm::rt
