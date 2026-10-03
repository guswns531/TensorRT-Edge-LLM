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

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/pagedKvTypes.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/scheduling/phaseKVActiveView.h"
#include "runtime/scheduling/phaseRaggedMetadata.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "runtime/state/stableKVPageManager.h"

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

namespace
{

// Gemma4-like prefill contract scaled down: dual RoPE, full-mode SWA, packed-prefill chunk carrier,
// text profile (batch 4, chunk 8) plus a vision profile (batch 2, chunk 16).
LLMEngineConfig makePrefillConfig()
{
    LLMEngineConfig cfg;
    cfg.hiddenSize = 16;
    cfg.outputVocabSize = 32;
    cfg.numAttentionLayers = 2;
    cfg.numDecoderLayers = 2;
    cfg.numKVHeads = 1;
    cfg.headDim = 8;
    cfg.rotaryDim = 8;
    cfg.maxSupportedBatchSize = 4;
    cfg.maxSupportedPrefillBatchSize = 4;
    cfg.maxSupportedDecodeBatchSize = 4;
    cfg.maxSupportedInputLength = 32;
    cfg.maxKVCacheCapacity = 512;
    cfg.maxNumSequences = 4;
    cfg.maxPhysicalTokens = 4 * 32;
    cfg.kvPoolPages
        = static_cast<int32_t>(computeMinimumKvPoolPages(cfg.maxSupportedBatchSize, cfg.maxKVCacheCapacity));
    cfg.numSwaPages = 256;
    cfg.kvCacheDtype = nvinfer1::DataType::kHALF;
    cfg.layerTypes.assign(2, HybridCacheManager::LayerType::kAttention);
    cfg.kvLayerConfigs = {
        KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8},
        KVLayerConfig{/*numKVHeads=*/1, /*headDim=*/8, /*kvCacheCapacity=*/129},
    };
    cfg.kvSharingDonors = {-1, -1};
    cfg.useDualRope = true;
    cfg.slidingRotaryDim = 4;
    cfg.fullRotaryDim = 8;
    cfg.slidingRopeConfig.type = RopeType::kDefault;
    cfg.fullRopeConfig.type = RopeType::kDefault;
    cfg.packedPrefill = true;
    cfg.maxPackedPrefillChunkTokens = 8;
    cfg.visionPrefillProfile = 2;
    cfg.maxSupportedVisionPrefillBatchSize = 2;
    cfg.maxVisionPackedPrefillChunkTokens = 16;
    cfg.setSwaKVCacheMode(SwaKVCacheMode::kFull);
    return cfg;
}

std::vector<int32_t> readInt32(Tensor const& device, cudaStream_t stream)
{
    size_t const count = static_cast<size_t>(device.getShape().volume());
    Tensor host({static_cast<int64_t>(count)}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "readback");
    CUDA_CHECK(cudaMemcpyAsync(
        host.rawPointer(), device.rawPointer(), count * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return {host.dataPointer<int32_t>(), host.dataPointer<int32_t>() + count};
}

std::vector<int64_t> readInt64(Tensor const& device, cudaStream_t stream)
{
    size_t const count = static_cast<size_t>(device.getShape().volume());
    Tensor host({static_cast<int64_t>(count)}, DeviceType::kCPU, nvinfer1::DataType::kINT64, "readback64");
    CUDA_CHECK(cudaMemcpyAsync(
        host.rawPointer(), device.rawPointer(), count * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return {host.dataPointer<int64_t>(), host.dataPointer<int64_t>() + count};
}

std::vector<float> readFloat(Tensor const& device, size_t count, cudaStream_t stream)
{
    Tensor host({static_cast<int64_t>(count)}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT, "readbackF");
    CUDA_CHECK(
        cudaMemcpyAsync(host.rawPointer(), device.rawPointer(), count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return {host.dataPointer<float>(), host.dataPointer<float>() + count};
}

} // namespace

// Case (a): initial packed prefill of unequal sequences uses the entry-padded layout.
TEST(PhasePrefillRaggedMetadataTest, InitialUnequalSequencesUseEntryPaddedRows)
{
    PhaseRaggedMetadataBuilder builder(4, 32);
    RaggedExecutionBatch const& batch = builder.build(SequenceWork::kContext, {{11, 0, 3}, {12, 0, 5}, {13, 0, 1}});
    EXPECT_EQ(builder.executionPhase(), ExecutionPhase::kContextPrefill);
    EXPECT_EQ(batch.shape.numSequences, 3);
    EXPECT_EQ(batch.shape.queryWidth, 5);
    EXPECT_EQ(batch.shape.physicalTokens, 15);
    EXPECT_EQ(batch.shape.validTokens, 9);
    EXPECT_EQ(batch.shape.numContextSequences, 3);
    EXPECT_EQ(batch.shape.numLogits, 3);
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 5, 10, 15}));
    EXPECT_EQ(batch.queryLengths, (std::vector<int32_t>{3, 5, 1}));
    EXPECT_EQ(batch.pastLengths, (std::vector<int32_t>{0, 0, 0}));
    EXPECT_EQ(batch.attentionSequenceLengths, (std::vector<int32_t>{3, 5, 1}));
    EXPECT_EQ(batch.stateIndices, (std::vector<int32_t>{0, 1, 2}));
    EXPECT_EQ(batch.positions, (std::vector<int32_t>{0, 1, 2, -1, -1, 0, 1, 2, 3, 4, 0, -1, -1, -1, -1}));
    EXPECT_EQ(batch.logitsIndices, (std::vector<int64_t>{2, 9, 10}));
    EXPECT_EQ(batch.logitsToSequence, (std::vector<int32_t>{0, 1, 2}));
}

// Case (b): chunk continuation carries per-row past lengths and selects the context-chunk phase.
TEST(PhasePrefillRaggedMetadataTest, ChunkContinuationCarriesPastLengths)
{
    PhaseRaggedMetadataBuilder builder(4, 32);
    RaggedExecutionBatch const& batch = builder.build(SequenceWork::kContext, {{21, 128, 4}, {22, 0, 2}});
    EXPECT_EQ(builder.executionPhase(), ExecutionPhase::kContextChunk);
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 4, 8}));
    EXPECT_EQ(batch.pastLengths, (std::vector<int32_t>{128, 0}));
    EXPECT_EQ(batch.attentionSequenceLengths, (std::vector<int32_t>{132, 2}));
    EXPECT_EQ(batch.positions, (std::vector<int32_t>{128, 129, 130, 131, 0, 1, -1, -1}));
    EXPECT_EQ(batch.logitsIndices, (std::vector<int64_t>{3, 5}));
}

// Case (c): one sequence degenerates to W = T = q.
TEST(PhasePrefillRaggedMetadataTest, SingleSequenceIsDegenerateEntryPadding)
{
    PhaseRaggedMetadataBuilder builder(4, 32);
    RaggedExecutionBatch const& batch = builder.build(SequenceWork::kContext, {{31, 16, 7}});
    EXPECT_EQ(builder.executionPhase(), ExecutionPhase::kContextChunk);
    EXPECT_EQ(batch.shape.queryWidth, 7);
    EXPECT_EQ(batch.shape.physicalTokens, 7);
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 7}));
    EXPECT_EQ(batch.logitsIndices, (std::vector<int64_t>{6}));
    EXPECT_EQ(batch.positions.front(), 16);
    EXPECT_EQ(batch.positions.back(), 22);
}

TEST(PhasePrefillRaggedMetadataTest, MatchesUpstreamValidator)
{
    PhaseRaggedMetadataBuilder builder(4, 32);
    RaggedExecutionBatch const& built = builder.build(SequenceWork::kContext, {{41, 0, 3}, {42, 9, 5}});
    RaggedExecutionBatch batch;
    batch.layout = built.layout;
    batch.shape = built.shape;
    batch.sequenceOrder = built.sequenceOrder;
    batch.positions = built.positions;
    batch.queryStartOffsets = built.queryStartOffsets;
    batch.queryLengths = built.queryLengths;
    batch.pastLengths = built.pastLengths;
    batch.attentionSequenceLengths = built.attentionSequenceLengths;
    batch.stateIndices = built.stateIndices;
    batch.logitsIndices = built.logitsIndices;
    batch.logitsToSequence = built.logitsToSequence;
    batch.sequenceWorks = built.sequenceWorks;
    batch.hostTokenIds = Tensor({batch.shape.physicalTokens}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "ids");
    std::fill_n(batch.hostTokenIds.dataPointer<int32_t>(), batch.shape.physicalTokens, 0);
    RaggedEngineContract const contract{TokenLayoutBackend::kEntryPaddedCompatibility, 4, 8, 32, 4, false};
    EXPECT_NO_THROW(RaggedBatchBuilder::validateExecutionBatch(batch, contract));
}

TEST(PhasePrefillRaggedMetadataTest, RejectsInvalidSteps)
{
    PhaseRaggedMetadataBuilder builder(2, 8);
    EXPECT_THROW(builder.build(SequenceWork::kContext, {}), std::runtime_error);
    EXPECT_THROW(builder.build(SequenceWork::kContext, {{1, 0, 0}}), std::runtime_error);
    EXPECT_THROW(builder.build(SequenceWork::kContext, {{1, -1, 2}}), std::runtime_error);
    EXPECT_THROW(builder.build(SequenceWork::kContext, {{1, 0, 1}, {2, 0, 1}, {3, 0, 1}}), std::runtime_error);
    EXPECT_THROW(builder.build(SequenceWork::kContext, {{1, 0, 5}, {2, 0, 1}}), std::runtime_error);
    EXPECT_THROW(builder.build(SequenceWork::kDecode, {{1, 4, 2}}), std::runtime_error);
}

TEST(PhasePrefillRaggedMetadataTest, PackedDimsCarryPhaseAndProfileChunkCap)
{
    LLMEngineConfig const cfg = makePrefillConfig();
    InferenceDims const text = cfg.packedPrefillDims(3, 15, 5, ExecutionPhase::kContextPrefill);
    EXPECT_EQ(text.seqLen, 15);
    EXPECT_EQ(text.selectLen, 3);
    EXPECT_EQ(text.queryOffsetLen, 4);
    EXPECT_EQ(text.contextSequenceCount, 3);
    EXPECT_EQ(text.attnMaskSeqLen, cfg.maxPackedPrefillChunkTokens);
    EXPECT_EQ(text.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kContextPrefill));

    InferenceDims const chunk = cfg.packedPrefillDims(2, 8, 4, ExecutionPhase::kContextChunk);
    EXPECT_EQ(chunk.executionPhaseLen, static_cast<int64_t>(ExecutionPhase::kContextChunk));

    // Case (d): vision profile uses its own batch and chunk caps.
    InferenceDims const vision = cfg.visionPackedPrefillDims(2, 32, 16, ExecutionPhase::kContextPrefill);
    EXPECT_EQ(vision.seqLen, 32);
    EXPECT_EQ(vision.attnMaskSeqLen, cfg.maxVisionPackedPrefillChunkTokens);
    EXPECT_THROW(cfg.visionPackedPrefillDims(3, 48, 16, ExecutionPhase::kContextPrefill), std::runtime_error);
    EXPECT_THROW(cfg.packedPrefillDims(1, 9, 9, ExecutionPhase::kContextPrefill), std::runtime_error);
    EXPECT_THROW(cfg.packedPrefillDims(1, 4, 4, ExecutionPhase::kAutoregressiveDecode), std::runtime_error);
}

// Binds the engine-style prefill input set and checks every uploaded ragged value and RoPE row.
TEST(PhasePrefillRaggedMetadataTest, UploadFillsEngineInputsAndTokenAlignedRope)
{
    LLMEngineConfig const cfg = makePrefillConfig();
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    {
        std::unordered_map<std::string, std::string> const noLoraWeights;
        std::unique_ptr<SharedResources> resources = SharedResources::createForLLM(cfg, noLoraWeights, nullptr);
        PipelineIO io = PipelineIO::createForLLMPhase(cfg, cfg.maxSupportedPrefillBatchSize, 8, stream);
        TensorMap map;
        buildTensorMap(map, io, *resources, cfg, /*kvCacheIndex=*/0);

        for (char const* name : {binding_names::kInputsEmbeds, binding_names::kRopeCosSinSliding,
                 binding_names::kRopeCosSinFull, binding_names::kPositions, binding_names::kQueryStartOffsets,
                 binding_names::kQueryLengths, binding_names::kPastLengths, binding_names::kAttentionSequenceLengths,
                 binding_names::kStateIndices, binding_names::kExecutionPhaseMarker,
                 binding_names::kContextSequenceCountCarrier, binding_names::kKVPageTable,
                 binding_names::kSwaKVPageTable, binding_names::kSwaKVCacheMode, binding_names::kLogitsIndices})
        {
            EXPECT_TRUE(map.contains(name)) << name;
        }
        EXPECT_FALSE(map.contains(binding_names::kRopeCosSin));
        EXPECT_FALSE(map.contains(binding_names::kLastTokenIds));

        PhaseRaggedMetadataBuilder builder(4, 32);
        RaggedExecutionBatch const& batch = builder.build(SequenceWork::kContext, {{1, 5, 3}, {2, 0, 2}});
        uploadPhaseRaggedMetadata(io, *resources, cfg, batch, stream);

        EXPECT_EQ(map.get(binding_names::kPositions), &io.positions);
        EXPECT_EQ(readInt32(io.positions, stream), (std::vector<int32_t>{5, 6, 7, 0, 1, -1}));
        EXPECT_EQ(readInt32(io.queryStartOffsets, stream), (std::vector<int32_t>{0, 3, 6}));
        EXPECT_EQ(readInt32(io.queryLengths, stream), (std::vector<int32_t>{3, 2}));
        EXPECT_EQ(readInt32(io.pastLengths, stream), (std::vector<int32_t>{5, 0}));
        EXPECT_EQ(readInt32(io.attentionSequenceLengths, stream), (std::vector<int32_t>{8, 2}));
        EXPECT_EQ(readInt32(io.stateIndices, stream), (std::vector<int32_t>{0, 1}));
        EXPECT_EQ(readInt64(io.logitsIndices, stream), (std::vector<int64_t>{2, 4}));
        EXPECT_EQ(io.contextSequenceCountCarrier.getShape()[0], 2);

        Tensor const& fullPool
            = resources->ropePool.getOrCreate(cfg.fullRopeConfig, cfg.fullRotaryDim, cfg.maxKVCacheCapacity, stream);
        ASSERT_EQ(io.raggedRopeCosSinFull.getShape(), Coords({6, cfg.fullRotaryDim}));
        std::vector<float> const gathered = readFloat(io.raggedRopeCosSinFull, 6 * cfg.fullRotaryDim, stream);
        std::vector<float> const pool
            = readFloat(fullPool, static_cast<size_t>(cfg.maxKVCacheCapacity) * cfg.fullRotaryDim, stream);
        std::vector<int32_t> const expectedPositions{5, 6, 7, 0, 1};
        for (size_t row = 0; row < expectedPositions.size(); ++row)
        {
            for (int32_t d = 0; d < cfg.fullRotaryDim; ++d)
            {
                EXPECT_EQ(gathered[row * cfg.fullRotaryDim + d],
                    pool[static_cast<size_t>(expectedPositions[row]) * cfg.fullRotaryDim + d])
                    << "row " << row << " dim " << d;
            }
        }
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Packed-layout counterpart of UploadFillsEngineInputsAndTokenAlignedRope: unequal rows with
// nonzero past, T=sum(q_i), confirming the host formulas reach the device unchanged under
// kNativeCompactRagged.
TEST(PhasePrefillRaggedMetadataTest, PackedUploadFillsEngineInputsAndTokenAlignedRope)
{
    LLMEngineConfig const cfg = makePrefillConfig();
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    {
        std::unordered_map<std::string, std::string> const noLoraWeights;
        std::unique_ptr<SharedResources> resources = SharedResources::createForLLM(cfg, noLoraWeights, nullptr);
        PipelineIO io = PipelineIO::createForLLMPhase(cfg, cfg.maxSupportedPrefillBatchSize, 8, stream);
        TensorMap map;
        buildTensorMap(map, io, *resources, cfg, /*kvCacheIndex=*/0);

        PhaseRaggedMetadataBuilder builder(4, 32);
        RaggedExecutionBatch const& batch
            = builder.build(SequenceWork::kContext, {{1, 5, 3}, {2, 0, 2}}, TokenLayoutBackend::kNativeCompactRagged);
        EXPECT_EQ(batch.layout, TokenLayoutBackend::kNativeCompactRagged);
        EXPECT_EQ(batch.shape.physicalTokens, 5);
        uploadPhaseRaggedMetadata(io, *resources, cfg, batch, stream);

        EXPECT_EQ(readInt32(io.positions, stream), (std::vector<int32_t>{5, 6, 7, 0, 1}));
        EXPECT_EQ(readInt32(io.queryStartOffsets, stream), (std::vector<int32_t>{0, 3, 5}));
        EXPECT_EQ(readInt32(io.queryLengths, stream), (std::vector<int32_t>{3, 2}));
        EXPECT_EQ(readInt32(io.pastLengths, stream), (std::vector<int32_t>{5, 0}));
        EXPECT_EQ(readInt32(io.attentionSequenceLengths, stream), (std::vector<int32_t>{8, 2}));
        EXPECT_EQ(readInt64(io.logitsIndices, stream), (std::vector<int64_t>{2, 4}));

        Tensor const& fullPool
            = resources->ropePool.getOrCreate(cfg.fullRopeConfig, cfg.fullRotaryDim, cfg.maxKVCacheCapacity, stream);
        ASSERT_EQ(io.raggedRopeCosSinFull.getShape(), Coords({5, cfg.fullRotaryDim}));
        std::vector<float> const gathered = readFloat(io.raggedRopeCosSinFull, 5 * cfg.fullRotaryDim, stream);
        std::vector<float> const pool
            = readFloat(fullPool, static_cast<size_t>(cfg.maxKVCacheCapacity) * cfg.fullRotaryDim, stream);
        std::vector<int32_t> const expectedPositions{5, 6, 7, 0, 1};
        for (size_t row = 0; row < expectedPositions.size(); ++row)
        {
            for (int32_t d = 0; d < cfg.fullRotaryDim; ++d)
            {
                EXPECT_EQ(gathered[row * cfg.fullRotaryDim + d],
                    pool[static_cast<size_t>(expectedPositions[row]) * cfg.fullRotaryDim + d])
                    << "row " << row << " dim " << d;
            }
        }
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// kv_page_table and swa_kv_page_table rows are indexed by active phase row: row i holds the pages of
// the stable slot of sequence i. No kvcache_start_index binding is required.
TEST(PhasePrefillRaggedMetadataTest, ActiveViewSwapsKvAndSwaTablesByActiveRow)
{
    cudaStream_t stream{};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    {
        StableKVPageManager ownership({8, 4, 64, 2048, 128});
        int32_t const slotA = ownership.reserve();
        int32_t const slotB = ownership.reserve();
        int32_t const slotC = ownership.reserve();
        ownership.ensureCapacity(slotA, 300);
        ownership.ensureCapacity(slotC, 129);
        static_cast<void>(slotB);

        TensorMap map;
        KVPageTable resident(4, 16, 64);
        map.set(binding_names::kKVPageTable, resident.kernelView());
        map.set(binding_names::kSwaKVPageTable, resident.kernelView());
        PhaseKVActiveView view(4, ownership, map, "prefill_view_test");
        view.prepare({slotC, slotA}, stream);

        EXPECT_EQ(map.get(binding_names::kKVPageTable), &view.pageTable().kernelView());
        EXPECT_EQ(map.get(binding_names::kSwaKVPageTable), &view.pageTable().kernelView());
        EXPECT_FALSE(map.contains(binding_names::kKVCacheStartIndex));
        int32_t const* row0 = view.pageTable().hostRow(0);
        int32_t const* row1 = view.pageTable().hostRow(1);
        std::vector<int32_t> const& pagesC = ownership.pages(slotC);
        std::vector<int32_t> const& pagesA = ownership.pages(slotA);
        for (size_t page = 0; page < pagesC.size(); ++page)
        {
            EXPECT_EQ(row0[page], pagesC[page]);
        }
        for (size_t page = 0; page < pagesA.size(); ++page)
        {
            EXPECT_EQ(row1[page], pagesA[page]);
        }
        view.complete();
        EXPECT_EQ(map.get(binding_names::kKVPageTable), &resident.kernelView());
        EXPECT_EQ(map.get(binding_names::kSwaKVPageTable), &resident.kernelView());
        CUDA_CHECK(cudaStreamSynchronize(stream));

        KVPageTable boundedSwa(4, 16, 64);
        map.set(binding_names::kSwaKVPageTable, boundedSwa.kernelView());
        PhaseKVActiveView boundedView(4, ownership, map, "bounded_prefill_view_test");
        EXPECT_THROW(boundedView.prepare({slotA}, stream), std::runtime_error);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
