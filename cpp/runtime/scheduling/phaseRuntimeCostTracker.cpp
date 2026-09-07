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

namespace
{

PhaseRuntimeCostTrackerConfig normalizePolicyConfig(PhaseRuntimeCostTrackerConfig config) noexcept
{
    if (!phasePolicyUsesContextualScalar(config.policyMode))
    {
        config.contextualPd.mode = PhaseContextualPdMode::kDisabled;
        config.contextualEp.mode = PhaseContextualPdMode::kDisabled;
        config.contextualEd.mode = PhaseContextualPdMode::kDisabled;
    }
    return config;
}

} // namespace

PhaseRuntimeCostTracker::PhaseRuntimeCostTracker(PhaseRuntimeCostTrackerConfig config)
    : mConfig(normalizePolicyConfig(config))
    , mActions(mConfig.action)
    , mContextualPd(mConfig.contextualPd)
    , mContextualDp(mConfig.contextualPd)
    , mContextualEp(mConfig.contextualEp)
    , mContextualPe(mConfig.contextualEp)
    , mContextualEd(mConfig.contextualEd)
    , mContextualDe(mConfig.contextualEd)
{
    ELLM_CHECK(mConfig.actionMinimumSamples > 0U, "Runtime action minimum sample count must be positive");
    ELLM_CHECK(mConfig.actionMinimumSamples <= mConfig.action.windowSize,
        "Runtime action sample window must cover the minimum sample count");
    ELLM_CHECK(mConfig.decodeMinimumSamples > 0U, "Runtime decode minimum sample count must be positive");
    ELLM_CHECK(mConfig.decodeWindowSize >= mConfig.decodeMinimumSamples,
        "Runtime decode sample window must cover the minimum sample count");
    ELLM_CHECK(mConfig.decodeContextBucketTokens > 0, "Runtime decode context bucket must be positive");
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
