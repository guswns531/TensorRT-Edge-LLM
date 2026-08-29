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

PhaseRuntimeCostTracker::PhaseRuntimeCostTracker(PhaseRuntimeCostTrackerConfig config)
    : mConfig(config)
    , mActions(config.action)
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

std::optional<PhaseGlobalCostEstimate> PhaseRuntimeCostTracker::trustedEstimatePrimaryBatchCoveringContext(
    PhaseGlobalActionKey const& key) const
{
    std::optional<PhaseGlobalCostEstimate> const estimateValue = mActions.estimatePrimaryBatchCoveringContext(key);
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

size_t PhaseRuntimeCostTracker::decodeBucketCount() const noexcept
{
    return mDecodeComponents.size();
}

void PhaseRuntimeCostTracker::reset()
{
    mActions.reset();
    mDecodeComponents.clear();
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
