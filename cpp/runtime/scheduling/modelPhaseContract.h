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

namespace trt_edgellm::rt
{

//! Model/engine capabilities consumed by the model-neutral phase runtime.
struct ModelPhaseContract
{
    bool pagedKVCache{};
    bool hasPLE{};
    int32_t numDeepstackFeatures{};
    bool hasMRope{};
    bool supportsChunkedPrefill{};
    bool supportsDynamicAdmission{};
    bool requiresRequestOwnedMultimodalState{};
    int32_t maxPrefillChunkTokens{};
};

inline ModelPhaseContract makeModelPhaseContract(LLMEngineConfig const& config)
{
    ModelPhaseContract contract;
    contract.pagedKVCache = config.kvPoolPages > 0;
    contract.hasPLE = config.pleEnabled;
    contract.numDeepstackFeatures = config.numDeepstackFeatures;
    contract.hasMRope = config.ropeConfig.type == RopeType::kMRope;
    contract.supportsChunkedPrefill = config.packedPrefill;
    contract.supportsDynamicAdmission = contract.pagedKVCache;
    contract.requiresRequestOwnedMultimodalState
        = contract.hasPLE || contract.numDeepstackFeatures > 0 || contract.hasMRope;
    contract.maxPrefillChunkTokens = config.maxPackedPrefillChunkTokens;
    return contract;
}

} // namespace trt_edgellm::rt
