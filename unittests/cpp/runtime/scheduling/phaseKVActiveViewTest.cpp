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

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

using namespace trt_edgellm;

TEST(PhaseKVActiveViewTest, IsolatedMetadataCostBenchmark)
{
    if (std::getenv("TRT_EDGELLM_KV_METADATA_BENCH") == nullptr)
    {
        GTEST_SKIP() << "Opt-in metadata benchmark";
    }
    cudaStream_t stream{};
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    for (int32_t const batch : {1, 8, 32, 64})
    {
        for (bool const persistent : {false, true})
        {
            for (bool const churn : {false, true})
            {
                rt::StableKVPageManager ownership({80, 64, 256, 2048, 128});
                std::vector<int32_t> slots;
                for (int32_t row{}; row < batch; ++row)
                {
                    int32_t const slot = ownership.reserve();
                    ownership.ensureCapacity(slot, 256);
                    ownership.setLength(slot, 256);
                    slots.push_back(slot);
                }
                rt::TensorMap map;
                rt::Tensor lengths({64}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "bench_lengths");
                rt::KVPageTable table(64, 16, 256);
                map.set(binding_names::kKVCacheStartIndex, lengths);
                map.set(binding_names::kKVPageTable, table.kernelView());
                rt::PhaseKVActiveView view(64, ownership, map, "metadata_bench");
                view.setPersistentPageBindingsEnabled(persistent);
                std::vector<double> hostUs, leaseUs, gpuUs;
                for (int32_t step{}; step < 1100; ++step)
                {
                    auto const leaseStart = std::chrono::steady_clock::now();
                    if (churn)
                    {
                        ownership.release(slots.front());
                        slots.front() = ownership.reserve();
                        ownership.ensureCapacity(slots.front(), 256);
                        ownership.setLength(slots.front(), 256);
                        std::rotate(slots.begin(), slots.begin() + 1, slots.end());
                    }
                    auto const hostStart = std::chrono::steady_clock::now();
                    CUDA_CHECK(cudaEventRecord(start, stream));
                    view.prepare(slots, stream);
                    CUDA_CHECK(cudaEventRecord(stop, stream));
                    auto const hostStop = std::chrono::steady_clock::now();
                    CUDA_CHECK(cudaEventSynchronize(stop));
                    float elapsed{};
                    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
                    view.complete();
                    if (step >= 100)
                    {
                        hostUs.push_back(std::chrono::duration<double, std::micro>(hostStop - hostStart).count());
                        leaseUs.push_back(std::chrono::duration<double, std::micro>(hostStart - leaseStart).count());
                        gpuUs.push_back(elapsed * 1000.0);
                    }
                }
                std::sort(hostUs.begin(), hostUs.end());
                std::sort(leaseUs.begin(), leaseUs.end());
                std::sort(gpuUs.begin(), gpuUs.end());
                auto const& stats = view.pageTableUploadStats();
                std::printf(
                    "KV_BENCH batch=%d persistent=%d churn=%d host_median_us=%.3f host_p95_us=%.3f "
                    "lease_median_us=%.3f lease_p95_us=%.3f event_median_us=%.3f event_p95_us=%.3f "
                    "table_bytes=%zu host_waits=%zu\n",
                    batch, persistent, churn, hostUs[500], hostUs[949], leaseUs[500], leaseUs[949], gpuUs[500],
                    gpuUs[949], stats.copyBytes, stats.hostWaits);
                EXPECT_EQ(ownership.availablePages(), 256 - batch * 2);
            }
        }
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

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
    decode.setPersistentDecodeSelectEnabled(true);
    decode.setPersistentPageBindingsEnabled(true);
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
    EXPECT_EQ(prefillContextLength, (std::vector<int32_t>{32}));

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
    EXPECT_EQ(decode.memoryStats().decodeMemsetOperations, 1U);
    EXPECT_EQ(decode.memoryStats().decodeSelectZeroReuses, 0U);

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

    prefill.prepare({slot0, slot2}, prefillStream);
    rt::PipelineIO packedIO = rt::PipelineIO::createForLLM(
        [] {
            rt::LLMEngineConfig config;
            config.maxSupportedBatchSize = 2;
            config.maxSupportedInputLength = 128;
            config.maxKVCacheCapacity = 512;
            config.hiddenSize = 8;
            config.outputVocabSize = 16;
            return config;
        }(),
        prefillStream);
    prefill.preparePrefillMetadata(packedIO, {3, 2}, prefillStream, true);
    std::vector<int64_t> packedSelectIndices(2);
    CUDA_CHECK(cudaMemcpy(packedSelectIndices.data(), packedIO.selectTokenIndices.rawPointer(),
        packedSelectIndices.size() * sizeof(int64_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(packedSelectIndices, (std::vector<int64_t>{2, 4}));
    std::vector<int32_t> packedContextLengths(2);
    CUDA_CHECK(cudaMemcpy(packedContextLengths.data(), packedIO.contextLengths.rawPointer(),
        packedContextLengths.size() * sizeof(int32_t), cudaMemcpyDeviceToHost));
    EXPECT_EQ(packedContextLengths, (std::vector<int32_t>{3, 2}));
    EXPECT_EQ(packedIO.selectTokenIndices.getShape()[0], 1);
    EXPECT_EQ(packedIO.selectTokenIndices.getShape()[1], 2);
    prefill.complete();

    decode.prepare({slot0}, decodeStream);
    decode.prepareDecodeMetadata(decodeIO, decodeStream);
    CUDA_CHECK(cudaStreamSynchronize(decodeStream));
    EXPECT_EQ(decode.memoryStats().decodeMemsetOperations, 1U);
    EXPECT_EQ(decode.memoryStats().decodeSelectZeroReuses, 1U);
    EXPECT_EQ(decode.memoryStats().pageBindingRowReuses, 1U);
    decode.complete();

    decode.setPersistentDecodeSelectEnabled(false);
    decode.prepare({slot0}, decodeStream);
    decode.prepareDecodeMetadata(decodeIO, decodeStream);
    decode.complete();
    decode.prepare({slot0}, decodeStream);
    decode.prepareDecodeMetadata(decodeIO, decodeStream);
    decode.complete();
    EXPECT_EQ(decode.memoryStats().decodeMemsetOperations, 3U);
    EXPECT_EQ(decode.memoryStats().decodeSelectZeroReuses, 1U);

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
