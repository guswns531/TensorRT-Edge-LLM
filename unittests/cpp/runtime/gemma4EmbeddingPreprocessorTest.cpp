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

#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"
#include "common/bindingNames.h"

#include <gtest/gtest.h>
#include <memory>

using namespace trt_edgellm::rt;

TEST(Gemma4EmbeddingPreprocessorTest, SharesImmutableTableAcrossPhaseLocalOutputs)
{
    LLMEngineConfig config;
    config.pleEnabled = true;
    config.numPleInputs = 2;
    config.pleHiddenSize = 4;

    auto table = std::make_shared<Tensor>(
        Coords{16, 8}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "gemma4_ple_shared_table");
    TensorMap prefillMap;
    TensorMap decodeMap;
    Gemma4EmbeddingPreprocessor prefill(config, 2, 8, prefillMap, table);
    Gemma4EmbeddingPreprocessor decode(config, 4, 1, decodeMap, table);

    EXPECT_EQ(prefill.shareTable().get(), table.get());
    EXPECT_EQ(decode.shareTable().get(), table.get());
    for (int32_t layer{}; layer < config.numPleInputs; ++layer)
    {
        std::string const name = trt_edgellm::binding_names::formatPleTokenEmbedsName(layer);
        Tensor* const prefillOutput = prefillMap.get(name);
        Tensor* const decodeOutput = decodeMap.get(name);
        ASSERT_NE(prefillOutput, nullptr);
        ASSERT_NE(decodeOutput, nullptr);
        EXPECT_NE(prefillOutput->rawPointer(), decodeOutput->rawPointer());
        EXPECT_EQ(prefillOutput->getShape(), (Coords{2, 8, 4}));
        EXPECT_EQ(decodeOutput->getShape(), (Coords{4, 1, 4}));
    }
}
