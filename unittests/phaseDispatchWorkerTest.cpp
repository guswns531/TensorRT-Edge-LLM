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

#include "runtime/scheduling/phaseDispatchWorker.h"
#include "common/bindingNames.h"
#include "common/cudaUtils.h"
#include "runtime/scheduling/phaseBatchState.h"
#include "testUtils.h"

#include <cstring>
#include <gtest/gtest.h>
#include <unordered_map>
#include <utility>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

rt::HybridCacheManager makeIndexedManager(int32_t maxBatchSize)
{
    rt::HybridCacheManager::Config config{};
    config.layerTypes = {rt::HybridCacheManager::LayerType::kAttention};
    config.kvConfig = rt::KVCacheManager::Config{1, maxBatchSize, 64, {rt::KVLayerConfig{1, 64}}, DataType::kHALF};
    config.mambaConfig.maxBatchSize = maxBatchSize;
    config.indexedKVCache = true;
    config.maxBatchSize = maxBatchSize;
    return rt::HybridCacheManager(config, nullptr);
}

TEST(PhaseBatchStateTest, BindsAndGathersStableSlots)
{
    int32_t const maxBatchSize = 4;
    rt::HybridCacheManager cacheManager = makeIndexedManager(maxBatchSize);
    std::vector<int32_t> initialLengths{10, 20, 30, 40};
    rt::Tensor hostLengths({maxBatchSize}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(hostLengths.rawPointer(), initialLengths.data(), initialLengths.size() * sizeof(int32_t));
    cacheManager.resetForNewSequences(hostLengths, nullptr);

    rt::PhaseBatchState state(2, "phase_batch_test");
    rt::TensorMap tensorMap;
    state.bind(tensorMap);
    ASSERT_EQ(tensorMap.get(binding_names::kKVSlotIds), &state.slotIds());
    ASSERT_EQ(tensorMap.get(binding_names::kKVCacheStartIndex), &state.lengths());

    std::vector<rt::PhaseWorkItem> const batch{{1, 128, 3}, {2, 128, 0}};
    state.prepare(batch, cacheManager, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    EXPECT_EQ(copyDeviceToHost<int32_t>(state.slotIds()), (std::vector<int32_t>{3, 0}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(state.lengths()), (std::vector<int32_t>{40, 10}));

    state.commit(cacheManager, 128, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    EXPECT_EQ(copyDeviceToHost<int32_t>(state.lengths()), (std::vector<int32_t>{168, 138}));
    EXPECT_EQ(
        copyDeviceToHost<int32_t>(cacheManager.getGlobalKVCacheLengths()), (std::vector<int32_t>{138, 20, 30, 168}));
}

TEST(PhaseBatchStateTest, RejectsDuplicatePhysicalSlots)
{
    rt::HybridCacheManager cacheManager = makeIndexedManager(2);
    rt::PhaseBatchState state(2, "phase_batch_duplicate_test");
    std::vector<rt::PhaseWorkItem> const duplicateBatch{{1, 32, 0}, {2, 32, 0}};
    EXPECT_THROW(state.prepare(duplicateBatch, cacheManager, nullptr), std::runtime_error);
}

TEST(PhaseBatchStateTest, RejectsOutOfRangePhysicalSlot)
{
    rt::HybridCacheManager cacheManager = makeIndexedManager(2);
    rt::PhaseBatchState state(1, "phase_batch_range_test");
    std::vector<rt::PhaseWorkItem> const invalidBatch{{1, 32, 2}};
    EXPECT_THROW(state.prepare(invalidBatch, cacheManager, nullptr), std::runtime_error);
}

TEST(PhaseDispatchWorkerTest, RunsChunkCompletionAndDecodeRequeue)
{
    rt::PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 128;
    config.maxPrefillBatchSize = 1;
    config.maxDecodeBatchSize = 1;
    rt::PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 300, 0});
    scheduler.enqueueDecode({2, 10, 1});

    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    int32_t prefillEnqueues{};
    int32_t decodeEnqueues{};
    int32_t prefillBatchCompletions{};
    int32_t decodeBatchCompletions{};
    std::unordered_map<uint64_t, int32_t> decodeSteps;
    rt::PhaseDispatchWorkerCallbacks callbacks;
    callbacks.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t) {
        ASSERT_EQ(batch.size(), 1U);
        ++prefillEnqueues;
    };
    callbacks.enqueueDecode = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t) {
        ASSERT_EQ(batch.size(), 1U);
        ++decodeEnqueues;
    };
    callbacks.completePrefillBatch = [&](std::vector<rt::PhaseWorkItem> const& batch) {
        EXPECT_EQ(batch.size(), 1U);
        EXPECT_GT(prefillEnqueues, prefillBatchCompletions);
        ++prefillBatchCompletions;
    };
    callbacks.completeDecodeBatch = [&](std::vector<rt::PhaseWorkItem> const& batch) {
        EXPECT_EQ(batch.size(), 1U);
        EXPECT_GT(decodeEnqueues, decodeBatchCompletions);
        ++decodeBatchCompletions;
    };
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) { return item.tokenOffset + item.tokenCount; };
    callbacks.completeDecode = [&](rt::PhaseWorkItem const& item) {
        int32_t const step = ++decodeSteps[item.requestId];
        bool const finished = item.requestId == 1 || step == 3;
        return rt::PhaseDecodeCompletion{item.tokenCount + 1, finished};
    };

    rt::PhaseDispatchWorker worker(scheduler, std::move(callbacks), prefillStream, decodeStream);
    worker.runUntilIdle(8);

    EXPECT_EQ(prefillEnqueues, 3);
    EXPECT_EQ(decodeEnqueues, 4);
    EXPECT_EQ(prefillBatchCompletions, prefillEnqueues);
    EXPECT_EQ(decodeBatchCompletions, decodeEnqueues);
    EXPECT_EQ(worker.dispatchCount(), 4U);
    EXPECT_FALSE(worker.busy());
    EXPECT_TRUE(scheduler.empty());
    EXPECT_FALSE(scheduler.hasRequest(1));
    EXPECT_FALSE(scheduler.hasRequest(2));

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

} // namespace
