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

#include "runtime/phase/policy/phaseGlobalCostModel.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>

namespace trt_edgellm::rt
{

//! Configuration for bounded, process-local CUDA timing observations.
struct PhaseRuntimeCostTrackerConfig
{
    PhaseGlobalCostModelConfig action;
    size_t actionMinimumSamples{4U};
    size_t decodeMinimumSamples{8U};
    size_t decodeWindowSize{32U};
    int32_t decodeContextBucketTokens{512};
};

enum class PhaseRuntimeCostConfidence
{
    kUnknown,
    kWarming,
    kReady,
};

//! Process-local measurements used by the next scheduling decision.
//!
//! The tracker neither persists observations nor assigns policy authority.
//! A bounded sample window replaces old measurements by observation count,
//! so no wall-clock TTL, startup anchor, or drift state is required.
class PhaseRuntimeCostTracker
{
public:
    explicit PhaseRuntimeCostTracker(PhaseRuntimeCostTrackerConfig config = {});

    void observe(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation observation);
    std::optional<PhaseGlobalCostEstimate> estimate(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> trustedEstimate(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> estimateInterpolatedPrimaryBatch(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> estimatePrimaryBatchCoveringContext(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> trustedEstimatePrimaryBatchCoveringContext(
        PhaseGlobalActionKey const& key) const;
    PhaseGlobalOverlapCostDiagnostic overlapDiagnostic(PhaseGlobalActionKey const& key) const;
    size_t sampleCount(PhaseGlobalActionKey const& key) const;
    PhaseRuntimeCostConfidence confidence(PhaseGlobalActionKey const& key) const;
    bool overlapEligible(PhaseGlobalActionKey const& key) const;

    //! Record the decode component separately from a combined action makespan.
    void observeDecode(
        int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive, float gpuMs);
    std::optional<float> decodeP95(
        int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive) const;
    size_t decodeBucketCount() const noexcept;

    void reset();

private:
    struct DecodeKey
    {
        int32_t batchSize{};
        int32_t contextBucket{};
        bool encoderActive{};
        bool prefillActive{};

        bool operator==(DecodeKey const& other) const noexcept;
    };

    struct DecodeKeyHash
    {
        size_t operator()(DecodeKey const& key) const noexcept;
    };

    DecodeKey decodeKey(
        int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive) const noexcept;

    PhaseRuntimeCostTrackerConfig mConfig;
    PhaseGlobalCostModel mActions;
    std::unordered_map<DecodeKey, std::deque<float>, DecodeKeyHash> mDecodeComponents;
};

} // namespace trt_edgellm::rt
