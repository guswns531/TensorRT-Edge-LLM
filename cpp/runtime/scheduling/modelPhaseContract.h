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

#include "runtime/config/llmEngineConfig.h"

#include <algorithm>
#include <cstdint>

namespace trt_edgellm
{
namespace rt
{

//! Model capabilities consumed by the model-neutral phase scheduler.
//! The scheduler must branch on these capabilities, never on an architecture
//! name or a model-specific class.
struct ModelPhaseContract
{
    bool indexedKVCache{};
    bool hasPLE{};
    int32_t numDeepstackFeatures{};
    bool hasMRope{};
    bool supportsChunkedPrefill{true};
    bool requiresAtomicMultimodalPrefill{};
    bool supportsDynamicAdmission{};
    int32_t maxPrefillChunkTokens{};

    bool hasDeepstack() const noexcept
    {
        return numDeepstackFeatures > 0;
    }

    bool supportsDynamicBatching() const noexcept
    {
        return supportsDynamicAdmission && indexedKVCache;
    }
};

inline ModelPhaseContract makeModelPhaseContract(LLMEngineConfig const& config)
{
    ModelPhaseContract contract;
    contract.indexedKVCache = config.indexedKVCache;
    contract.hasPLE = config.pleEnabled;
    contract.numDeepstackFeatures = config.numDeepstackFeatures;
    contract.hasMRope = config.ropeConfig.type == RopeType::kMRope;
    contract.supportsChunkedPrefill = !config.useVisionBidirectionalAttention;
    contract.requiresAtomicMultimodalPrefill = config.useVisionBidirectionalAttention;
    contract.supportsDynamicAdmission = config.indexedKVCache;
    contract.maxPrefillChunkTokens
        = config.packedPrefill ? config.maxPackedPrefillChunkTokens : std::max(1, config.maxSupportedInputLength);
    return contract;
}

} // namespace rt
} // namespace trt_edgellm
