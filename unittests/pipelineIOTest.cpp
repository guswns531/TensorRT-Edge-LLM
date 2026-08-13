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

#include "runtime/state/pipelineIO.h"

#include <gtest/gtest.h>

using namespace trt_edgellm::rt;

namespace
{

TEST(PipelineIOTest, PackedPrefillSeparatesLogicalRowsFromTokenCarrier)
{
    constexpr int32_t kLOGICAL_BATCH{8};
    constexpr int32_t kTOTAL_TOKENS{1024};
    constexpr int32_t kHIDDEN_SIZE{16};
    constexpr int32_t kVOCAB_SIZE{32};
    constexpr int32_t kDEEPSTACK_FEATURES{3};
    LLMEngineConfig config;
    config.packedPrefill = true;
    config.maxSupportedBatchSize = kLOGICAL_BATCH;
    config.maxSupportedPrefillBatchSize = kLOGICAL_BATCH;
    config.maxSupportedInputLength = kTOTAL_TOKENS;
    config.hiddenSize = kHIDDEN_SIZE;
    config.outputVocabSize = kVOCAB_SIZE;
    config.numDeepstackFeatures = kDEEPSTACK_FEATURES;

    PipelineIO io = PipelineIO::createForPackedPrefill(config, kLOGICAL_BATCH, kTOTAL_TOKENS, nullptr);

    EXPECT_EQ(io.inputsEmbeds.getShape()[0], 1);
    EXPECT_EQ(io.inputsEmbeds.getShape()[1], kTOTAL_TOKENS);
    EXPECT_EQ(io.inputsEmbeds.getShape()[2], kHIDDEN_SIZE);
    EXPECT_EQ(io.outputLogits.getShape()[0], kLOGICAL_BATCH);
    EXPECT_EQ(io.selectTokenIndices.getShape()[1], kLOGICAL_BATCH);
    EXPECT_EQ(io.contextLengths.getShape()[0], kLOGICAL_BATCH);
    ASSERT_EQ(io.deepstackEmbeds.size(), kDEEPSTACK_FEATURES);
    EXPECT_EQ(io.deepstackEmbeds.front().getShape()[0], 1);
    EXPECT_EQ(io.deepstackEmbeds.front().getShape()[1], kTOTAL_TOKENS);
    EXPECT_EQ(io.outputHiddenStates.getShape()[0], 1);
    EXPECT_EQ(io.outputHiddenStates.getShape()[1], kTOTAL_TOKENS);
}

} // namespace
