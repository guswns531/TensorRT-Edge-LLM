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

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

// Mirrors a down-scaled version of a real serving contract: engine-global capacity is far
// larger than any single phase's local window (prefill chunk, decode batch, etc.).
LLMEngineConfig makePhaseCapacityConfig()
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 16;
    cfg.outputVocabSize = 32;
    cfg.numAttentionLayers = 1;
    cfg.numDecoderLayers = 1;
    cfg.maxSupportedBatchSize = 80;
    cfg.maxSupportedInputLength = 2048;
    cfg.maxKVCacheCapacity = 2048;
    cfg.maxNumSequences = 80;
    cfg.maxPhysicalTokens = 80 * 2048;
    cfg.ropeConfig.type = RopeType::kNoRope;
    return cfg;
}

} // namespace

// Phase-scoped PipelineIO (e.g. decode: batch=64, seqLen=1) must size its ragged metadata by the
// phase's own window, not by the engine-global maxPhysicalTokens/maxNumSequences capacity.
TEST(PipelineIOPhaseCapacityTest, DecodePhaseSizesRaggedMetadataByPhaseWindow)
{
    LLMEngineConfig const cfg = makePhaseCapacityConfig();
    int32_t constexpr kDecodeBatch = 64;
    int32_t constexpr kDecodeSeqLen = 1;

    PipelineIO io = PipelineIO::createForLLMPhase(cfg, kDecodeBatch, kDecodeSeqLen, /*stream=*/0);

    EXPECT_EQ(io.queryLengths.getShape()[0], kDecodeBatch);
    EXPECT_EQ(io.pastLengths.getShape()[0], kDecodeBatch);
    EXPECT_EQ(io.stateIndices.getShape()[0], kDecodeBatch);
    EXPECT_EQ(io.logitsIndices.getShape()[0], kDecodeBatch);
    EXPECT_EQ(io.queryStartOffsets.getShape()[0], kDecodeBatch + 1);
    EXPECT_EQ(io.inputsEmbeds.getShape()[0], kDecodeBatch * kDecodeSeqLen);
    EXPECT_EQ(io.positions.getShape()[0], kDecodeBatch * kDecodeSeqLen);
    EXPECT_LT(io.inputsEmbeds.getShape()[0], cfg.maxPhysicalTokens);
    EXPECT_LT(io.queryLengths.getShape()[0], cfg.maxNumSequences);
}

// A prefill phase with a packed-chunk window should likewise size ragged token buffers by its own
// token budget, not the full engine capacity.
TEST(PipelineIOPhaseCapacityTest, PrefillPhaseSizesRaggedMetadataByPhaseWindow)
{
    LLMEngineConfig const cfg = makePhaseCapacityConfig();
    int32_t constexpr kPrefillBatch = 8;
    int32_t constexpr kPrefillChunkTokens = 1024;

    PipelineIO io = PipelineIO::createForLLMPhase(cfg, kPrefillBatch, kPrefillChunkTokens, /*stream=*/0);

    EXPECT_EQ(io.queryLengths.getShape()[0], kPrefillBatch);
    EXPECT_EQ(io.inputsEmbeds.getShape()[0], kPrefillBatch * kPrefillChunkTokens);
    EXPECT_LT(io.inputsEmbeds.getShape()[0], cfg.maxPhysicalTokens);
}

// The full-engine (non-phase) path must keep allocating at engine-global capacity: it is reused by
// shared ragged-execution helpers that assume the full-capacity window.
TEST(PipelineIOPhaseCapacityTest, FullEngineIOKeepsEngineGlobalCapacity)
{
    LLMEngineConfig const cfg = makePhaseCapacityConfig();

    PipelineIO io = PipelineIO::createForLLM(cfg, /*stream=*/0);

    EXPECT_EQ(io.queryLengths.getShape()[0], cfg.maxNumSequences);
    EXPECT_EQ(io.inputsEmbeds.getShape()[0], cfg.maxPhysicalTokens);
}
