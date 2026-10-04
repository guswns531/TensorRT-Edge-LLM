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

#include <cuda_fp16.h>
#include <gtest/gtest.h>

#include <vector>

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

// Packed engines expose prefill-time inputs_embeds/outputHiddenStates as compact rows
// (sum of per-sequence lengths, no padding between sequences), not {batch, maxLen, hiddenSize}.
// StreamingPrefillBuffers::populateFromPrefill() must gather each sequence's compact rows into
// its own padded [0, q_i) slice and zero the tail so Qwen3-Omni thinker consumers see a uniform
// {batch, maxLen, hiddenSize} view indexed by sequence, not by raw packed row offset.
TEST(PipelineIOPhaseCapacityTest, StreamingPrefillGathersPackedRowsPerSequenceWithZeroedTail)
{
    int32_t constexpr kBatch = 2;
    int32_t constexpr kHiddenSize = 2;
    int32_t constexpr kMaxBatch = 2;
    int32_t constexpr kMaxSeq = 4;
    std::vector<int32_t> const sequenceLengths{3, 1};
    int32_t const totalTokens = 3 + 1;

    // Packed live buffer: row r holds value r in every hidden-size slot, so each sequence's
    // gathered rows are trivially distinguishable from both its neighbor and a zero pad.
    std::vector<__half> hostLive(static_cast<size_t>(totalTokens) * kHiddenSize);
    for (int32_t row = 0; row < totalTokens; ++row)
    {
        for (int32_t h = 0; h < kHiddenSize; ++h)
        {
            hostLive[static_cast<size_t>(row) * kHiddenSize + h] = __half(static_cast<float>(row + 1));
        }
    }
    Tensor liveInputEmbeds(
        {totalTokens, kHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "test_live_input_embeds");
    Tensor liveHiddenStates(
        {totalTokens, kHiddenSize}, DeviceType::kGPU, nvinfer1::DataType::kHALF, "test_live_hidden_states");
    ASSERT_EQ(cudaMemcpy(liveInputEmbeds.rawPointer(), hostLive.data(), hostLive.size() * sizeof(__half),
                  cudaMemcpyHostToDevice),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpy(liveHiddenStates.rawPointer(), hostLive.data(), hostLive.size() * sizeof(__half),
                  cudaMemcpyHostToDevice),
        cudaSuccess);

    StreamingPrefillBuffers streaming;
    streaming.populateFromPrefill(liveInputEmbeds, liveHiddenStates, kBatch, sequenceLengths, /*packed=*/true,
        kHiddenSize, kMaxBatch, kMaxSeq, /*stream=*/0);
    ASSERT_EQ(cudaStreamSynchronize(0), cudaSuccess);

    int32_t const maxLen = 3;
    ASSERT_EQ(streaming.inputEmbeds.getShape()[0], kBatch);
    ASSERT_EQ(streaming.inputEmbeds.getShape()[1], maxLen);
    std::vector<__half> hostGathered(static_cast<size_t>(kBatch) * maxLen * kHiddenSize);
    ASSERT_EQ(cudaMemcpy(hostGathered.data(), streaming.inputEmbeds.rawPointer(), hostGathered.size() * sizeof(__half),
                  cudaMemcpyDeviceToHost),
        cudaSuccess);

    auto const at = [&](int32_t seq, int32_t row, int32_t h) {
        return static_cast<float>(hostGathered[(static_cast<size_t>(seq) * maxLen + row) * kHiddenSize + h]);
    };
    // Sequence 0 (q=3) takes packed rows 0..2 (values 1,2,3); no padding needed.
    for (int32_t row = 0; row < 3; ++row)
    {
        EXPECT_FLOAT_EQ(at(0, row, 0), static_cast<float>(row + 1));
        EXPECT_FLOAT_EQ(at(0, row, 1), static_cast<float>(row + 1));
    }
    // Sequence 1 (q=1) takes packed row 3 (value 4) at its own row 0; rows 1..2 are zero-padded,
    // not misassigned rows that belonged to sequence 0 in the packed stream.
    EXPECT_FLOAT_EQ(at(1, 0, 0), 4.0f);
    EXPECT_FLOAT_EQ(at(1, 0, 1), 4.0f);
    for (int32_t row = 1; row < 3; ++row)
    {
        EXPECT_FLOAT_EQ(at(1, row, 0), 0.0f);
        EXPECT_FLOAT_EQ(at(1, row, 1), 0.0f);
    }
}
