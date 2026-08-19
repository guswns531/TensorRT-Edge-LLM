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

#include "runtime/scheduling/phaseKVActiveView.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"

#include <gtest/gtest.h>

using namespace trt_edgellm;

TEST(PhaseKVActiveViewTest, GivesConcurrentPhasesIndependentBindingsOverSharedPages)
{
    rt::StableKVPageManager ownership({4, 2, 12, 512, 128});
    int32_t const slot0 = ownership.reserve();
    int32_t const slot1 = ownership.reserve();
    int32_t const slot2 = ownership.reserve();
    ownership.ensureCapacity(slot0, 128);
    ownership.ensureCapacity(slot1, 256);
    ownership.ensureCapacity(slot2, 128);
    ownership.setLength(slot0, 64);
    ownership.setLength(slot1, 192);
    ownership.setLength(slot2, 96);

    rt::Tensor legacyLengths({2}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "legacy_lengths");
    rt::KVPageTable legacyPageTable(2, 4, 12);
    rt::TensorMap prefillMap;
    rt::TensorMap decodeMap;
    prefillMap.set(binding_names::kKVCacheStartIndex, legacyLengths);
    prefillMap.set(binding_names::kKVPageTable, legacyPageTable.kernelView());
    decodeMap.set(binding_names::kKVCacheStartIndex, legacyLengths);
    decodeMap.set(binding_names::kKVPageTable, legacyPageTable.kernelView());

    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    rt::PhaseKVActiveView prefill(2, ownership, prefillMap, "prefill");
    rt::PhaseKVActiveView decode(2, ownership, decodeMap, "decode");
    prefill.prepare({slot2}, prefillStream);
    decode.prepare({slot0, slot1}, decodeStream);
    CUDA_CHECK(cudaStreamSynchronize(prefillStream));
    CUDA_CHECK(cudaStreamSynchronize(decodeStream));

    EXPECT_NE(prefillMap.get(binding_names::kKVPageTable), decodeMap.get(binding_names::kKVPageTable));
    EXPECT_NE(prefillMap.get(binding_names::kKVCacheStartIndex), decodeMap.get(binding_names::kKVCacheStartIndex));
    EXPECT_EQ(prefill.pageTable().hostRow(0)[0], ownership.pages(slot2)[0]);
    EXPECT_EQ(decode.pageTable().hostRow(0)[0], ownership.pages(slot0)[0]);
    EXPECT_EQ(decode.pageTable().hostRow(1)[0], ownership.pages(slot1)[0]);

    std::vector<int32_t> prefillLength(1);
    std::vector<int32_t> decodeLengths(2);
    CUDA_CHECK(cudaMemcpy(
        prefillLength.data(), prefill.activeLengths().rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        decodeLengths.data(), decode.activeLengths().rawPointer(), 2 * sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(prefillLength, (std::vector<int32_t>{96}));
    EXPECT_EQ(decodeLengths, (std::vector<int32_t>{64, 192}));

    rt::PipelineIO prefillIO = rt::PipelineIO::createForLLM(
        [] {
            rt::LLMEngineConfig config;
            config.maxSupportedBatchSize = 2;
            config.maxSupportedInputLength = 128;
            config.maxKVCacheCapacity = 512;
            config.hiddenSize = 8;
            config.outputVocabSize = 16;
            config.numDeepstackFeatures = 0;
            return config;
        }(),
        prefillStream);
    prefill.preparePrefillMetadata(prefillIO, {32}, prefillStream);
    std::vector<int32_t> prefillContextLength(1);
    CUDA_CHECK(cudaMemcpy(
        prefillContextLength.data(), prefillIO.contextLengths.rawPointer(), sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(prefillContextLength, (std::vector<int32_t>{128}));

    rt::PipelineIO decodeIO = rt::PipelineIO::createForLLM(
        [] {
            rt::LLMEngineConfig config;
            config.maxSupportedBatchSize = 2;
            config.maxSupportedInputLength = 128;
            config.maxKVCacheCapacity = 512;
            config.hiddenSize = 8;
            config.outputVocabSize = 16;
            config.numDeepstackFeatures = 0;
            return config;
        }(),
        decodeStream);
    decode.prepareDecodeMetadata(decodeIO, decodeStream);
    std::vector<int32_t> decodeContextLengths(2);
    CUDA_CHECK(cudaMemcpy(decodeContextLengths.data(), decodeIO.contextLengths.rawPointer(), 2 * sizeof(int32_t),
        cudaMemcpyDeviceToHost));
    EXPECT_EQ(decodeContextLengths, (std::vector<int32_t>{65, 193}));

    prefill.commitLengths({128});
    decode.commitLengths({65, 193});
    EXPECT_EQ(ownership.length(slot2), 128);
    EXPECT_EQ(ownership.length(slot0), 65);
    EXPECT_EQ(ownership.length(slot1), 193);

    prefill.complete();
    decode.complete();
    EXPECT_EQ(prefillMap.get(binding_names::kKVCacheStartIndex), &legacyLengths);
    EXPECT_EQ(prefillMap.get(binding_names::kKVPageTable), &legacyPageTable.kernelView());
    EXPECT_EQ(decodeMap.get(binding_names::kKVCacheStartIndex), &legacyLengths);
    EXPECT_EQ(decodeMap.get(binding_names::kKVPageTable), &legacyPageTable.kernelView());

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseKVActiveViewTest, RejectsNestedPrepareAndLengthMismatch)
{
    rt::StableKVPageManager ownership({2, 1, 4, 512, 128});
    int32_t const slot = ownership.reserve();
    ownership.ensureCapacity(slot, 128);
    rt::Tensor legacyLengths({1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "legacy_lengths");
    rt::KVPageTable legacyPageTable(1, 4, 4);
    rt::TensorMap tensorMap;
    tensorMap.set(binding_names::kKVCacheStartIndex, legacyLengths);
    tensorMap.set(binding_names::kKVPageTable, legacyPageTable.kernelView());
    rt::PhaseKVActiveView view(1, ownership, tensorMap, "phase");

    view.prepare({slot}, nullptr);
    EXPECT_THROW(view.prepare({slot}, nullptr), std::runtime_error);
    EXPECT_THROW(view.commitLengths({1, 2}), std::runtime_error);
    view.complete();
    EXPECT_THROW(view.complete(), std::runtime_error);
}
