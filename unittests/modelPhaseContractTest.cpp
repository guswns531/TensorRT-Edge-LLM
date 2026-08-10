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

#include "runtime/scheduling/modelPhaseContract.h"

#include <gtest/gtest.h>

using namespace trt_edgellm::rt;

TEST(ModelPhaseContractTest, IndexedTextSupportsDynamicChunking)
{
    LLMEngineConfig config;
    config.indexedKVCache = true;
    config.maxSupportedInputLength = 1024;

    ModelPhaseContract const contract = makeModelPhaseContract(config);

    EXPECT_TRUE(contract.indexedKVCache);
    EXPECT_TRUE(contract.supportsChunkedPrefill);
    EXPECT_TRUE(contract.supportsDynamicAdmission);
    EXPECT_TRUE(contract.supportsDynamicBatching());
    EXPECT_EQ(contract.maxPrefillChunkTokens, 1024);
}

TEST(ModelPhaseContractTest, MultimodalRequiresAtomicPrefill)
{
    LLMEngineConfig config;
    config.indexedKVCache = true;
    config.maxSupportedInputLength = 2048;
    config.numDeepstackFeatures = 3;
    config.ropeConfig.type = RopeType::kMRope;
    config.useVisionBidirectionalAttention = true;

    ModelPhaseContract const contract = makeModelPhaseContract(config);

    EXPECT_TRUE(contract.hasDeepstack());
    EXPECT_TRUE(contract.hasMRope);
    EXPECT_FALSE(contract.supportsChunkedPrefill);
    EXPECT_TRUE(contract.requiresAtomicMultimodalPrefill);
    EXPECT_TRUE(contract.supportsDynamicBatching());
}

TEST(ModelPhaseContractTest, LegacyCacheRejectsDynamicAdmission)
{
    LLMEngineConfig config;
    config.indexedKVCache = false;
    config.maxSupportedInputLength = 512;

    ModelPhaseContract const contract = makeModelPhaseContract(config);

    EXPECT_FALSE(contract.supportsDynamicAdmission);
    EXPECT_FALSE(contract.supportsDynamicBatching());
    EXPECT_EQ(contract.maxPrefillChunkTokens, 512);
}
