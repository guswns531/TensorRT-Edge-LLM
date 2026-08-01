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
#include "runtime/scheduling/phaseContextBatchAdapter.h"
#include "runtime/scheduling/phaseContextServingFacade.h"
#include "runtime/scheduling/phaseGreedySampler.h"
#include "runtime/scheduling/phasePrefillContextBatchAdapter.h"
#include "runtime/scheduling/phaseRequestLifecycle.h"
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
    std::vector<rt::PhaseDispatchMetrics> metrics;
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
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) {
        return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, false};
    };
    callbacks.completeDecode = [&](rt::PhaseWorkItem const& item) {
        int32_t const step = ++decodeSteps[item.requestId];
        bool const finished = item.requestId == 1 || step == 3;
        return rt::PhaseDecodeCompletion{item.tokenCount + 1, finished};
    };
    callbacks.onMetrics = [&](rt::PhaseDispatchMetrics const& sample) { metrics.push_back(sample); };

    rt::PhaseDispatchWorker worker(scheduler, std::move(callbacks), prefillStream, decodeStream);
    worker.runUntilIdle(8);

    EXPECT_EQ(prefillEnqueues, 3);
    EXPECT_EQ(decodeEnqueues, 4);
    EXPECT_EQ(prefillBatchCompletions, prefillEnqueues);
    EXPECT_EQ(decodeBatchCompletions, decodeEnqueues);
    EXPECT_EQ(worker.dispatchCount(), 4U);
    ASSERT_EQ(metrics.size(), worker.dispatchCount());
    ASSERT_TRUE(worker.lastMetrics().has_value());
    EXPECT_EQ(worker.lastMetrics()->dispatchIndex, metrics.back().dispatchIndex);
    EXPECT_EQ(metrics.front().kind, rt::PhaseDispatchKind::kOverlap);
    EXPECT_EQ(metrics.front().prefillBatchSize, 1);
    EXPECT_EQ(metrics.front().decodeBatchSize, 1);
    EXPECT_EQ(metrics.front().prefillTokens, 128);
    EXPECT_EQ(metrics.front().decodeTokens, 1);
    for (size_t index = 0; index < metrics.size(); ++index)
    {
        EXPECT_EQ(metrics[index].dispatchIndex, index + 1);
        EXPECT_GE(metrics[index].prefillQueueWaitUs, 0.0);
        EXPECT_GE(metrics[index].decodeQueueWaitUs, 0.0);
        EXPECT_GE(metrics[index].prefillGpuMs, 0.0F);
        EXPECT_GE(metrics[index].decodeGpuMs, 0.0F);
        EXPECT_GE(metrics[index].makespanGpuMs, 0.0F);
        EXPECT_GE(metrics[index].overlapRatio, 0.0F);
        EXPECT_LE(metrics[index].overlapRatio, 1.0F);
    }
    EXPECT_FALSE(worker.busy());
    EXPECT_TRUE(scheduler.empty());
    EXPECT_FALSE(scheduler.hasRequest(1));
    EXPECT_FALSE(scheduler.hasRequest(2));

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseRequestLifecycleTest, OwnsStableSlotsAcrossContinuousQueueTransitions)
{
    rt::PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 2;
    config.maxPrefillBatchSize = 2;
    config.maxDecodeBatchSize = 2;
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    std::unordered_map<uint64_t, int32_t> observedSlots;
    std::unordered_map<uint64_t, int32_t> decodeSteps;
    std::vector<rt::PhaseRequestSnapshot> terminals;
    rt::PhaseRequestLifecycleCallbacks callbacks;
    callbacks.execution.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t) {
        for (rt::PhaseWorkItem const& item : batch)
        {
            auto const [it, inserted] = observedSlots.emplace(item.requestId, item.kvSlotId);
            if (!inserted)
            {
                EXPECT_EQ(it->second, item.kvSlotId);
            }
        }
    };
    callbacks.execution.enqueueDecode = callbacks.execution.enqueuePrefill;
    callbacks.execution.completePrefill = [](rt::PhaseWorkItem const& item) {
        return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, false};
    };
    callbacks.execution.completeDecode = [&](rt::PhaseWorkItem const& item) {
        int32_t const step = ++decodeSteps[item.requestId];
        return rt::PhaseDecodeCompletion{item.tokenCount + 1, step == 2};
    };
    callbacks.onTerminal = [&](rt::PhaseRequestSnapshot const& snapshot) { terminals.push_back(snapshot); };

    {
        rt::PhaseRequestLifecycle lifecycle(2, config, std::move(callbacks), prefillStream, decodeStream);
        EXPECT_EQ(lifecycle.submit(10, 5), 0);
        EXPECT_EQ(lifecycle.submit(20, 4), 1);
        EXPECT_EQ(lifecycle.availableSlotCount(), 0);
        EXPECT_THROW(lifecycle.submit(30, 3), std::runtime_error);
        EXPECT_TRUE(lifecycle.cancel(20));
        EXPECT_EQ(lifecycle.availableSlotCount(), 1);
        EXPECT_EQ(lifecycle.submit(30, 3), 1);
        EXPECT_EQ(lifecycle.activeRequestCount(), 2U);
        lifecycle.runUntilIdle(16);
        EXPECT_TRUE(lifecycle.empty());
        EXPECT_EQ(lifecycle.activeRequestCount(), 0U);
        EXPECT_EQ(lifecycle.availableSlotCount(), 2);
        auto const request10 = lifecycle.request(10);
        auto const request20 = lifecycle.request(20);
        auto const request30 = lifecycle.request(30);
        ASSERT_TRUE(request10.has_value());
        ASSERT_TRUE(request20.has_value());
        ASSERT_TRUE(request30.has_value());
        EXPECT_EQ(request10->status, rt::PhaseRequestStatus::kFinished);
        EXPECT_EQ(request20->status, rt::PhaseRequestStatus::kCancelled);
        EXPECT_EQ(request30->status, rt::PhaseRequestStatus::kFinished);
        EXPECT_EQ(request10->kvLength, 7);
        EXPECT_EQ(request30->kvLength, 5);
        EXPECT_EQ(observedSlots.at(10), 0);
        EXPECT_EQ(observedSlots.at(30), 1);
        EXPECT_EQ(terminals.size(), 3U);
    }

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseRequestLifecycleTest, DefersInFlightCancellationUntilEventCompletion)
{
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    std::vector<rt::PhaseRequestSnapshot> terminals;
    rt::PhaseRequestLifecycleCallbacks callbacks;
    callbacks.execution.enqueuePrefill = [](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) {};
    callbacks.execution.enqueueDecode = [](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) {};
    callbacks.execution.completePrefill = [](rt::PhaseWorkItem const& item) {
        return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, false};
    };
    callbacks.execution.completeDecode
        = [](rt::PhaseWorkItem const& item) { return rt::PhaseDecodeCompletion{item.tokenCount + 1, true}; };
    callbacks.onTerminal = [&](rt::PhaseRequestSnapshot const& snapshot) { terminals.push_back(snapshot); };

    {
        rt::PhaseRequestLifecycle lifecycle(
            1, rt::PhaseQueueSchedulerConfig{}, std::move(callbacks), prefillStream, decodeStream);
        EXPECT_EQ(lifecycle.submit(7, 2), 0);
        EXPECT_TRUE(lifecycle.dispatchNext());
        EXPECT_TRUE(lifecycle.busy());
        EXPECT_FALSE(lifecycle.cancel(7));
        EXPECT_EQ(lifecycle.availableSlotCount(), 0);
        lifecycle.wait();
        auto const afterPrefill = lifecycle.request(7);
        ASSERT_TRUE(afterPrefill.has_value());
        EXPECT_EQ(afterPrefill->status, rt::PhaseRequestStatus::kDecode);
        EXPECT_TRUE(lifecycle.cancel(7));
        EXPECT_EQ(lifecycle.availableSlotCount(), 1);
        auto const cancelled = lifecycle.request(7);
        ASSERT_TRUE(cancelled.has_value());
        EXPECT_EQ(cancelled->status, rt::PhaseRequestStatus::kCancelled);
    }

    EXPECT_EQ(terminals.size(), 1U);
    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseGreedySamplerTest, SamplesActualLogitsAndAppliesEosAndLengthState)
{
    constexpr int32_t batchSize = 2;
    constexpr int32_t vocabSize = 8;
    rt::Tensor logits({batchSize, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT, "phase_sampler_logits");
    std::vector<float> hostLogits(static_cast<size_t>(batchSize * vocabSize), -10.0F);
    hostLogits[3] = 7.0F;
    hostLogits[vocabSize + 5] = 9.0F;
    CUDA_CHECK(
        cudaMemcpy(logits.rawPointer(), hostLogits.data(), hostLogits.size() * sizeof(float), cudaMemcpyHostToDevice));

    rt::DecodingInferenceContext context;
    context.initialize(batchSize, 2, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    context.rawBatchedInputIds = {{1}, {2}};
    context.tokenIds = context.rawBatchedInputIds;
    context.effectivePrefillLengths = {1, 1};

    rt::PhaseGreedySampler sampler(batchSize, vocabSize, {3}, "phase_sampler_test");
    sampler.enqueue(logits, batchSize, nullptr);
    ASSERT_TRUE(sampler.pending());
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    sampler.completeDecode(context);

    EXPECT_FALSE(sampler.pending());
    EXPECT_EQ(context.tokenIds[0], (std::vector<int32_t>{1, 3}));
    EXPECT_EQ(context.tokenIds[1], (std::vector<int32_t>{2, 5}));
    EXPECT_EQ(context.currentGenerateLengths, (std::vector<int32_t>{1, 1}));
    EXPECT_EQ(context.finishedStates, (std::vector<int8_t>{1, 0}));
}

TEST(PhaseRequestLifecycleTest, ReleasesSlotWhenFinalPrefillSampleFinishes)
{
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    int32_t decodeEnqueues{};
    std::vector<rt::PhaseRequestSnapshot> terminals;
    rt::PhaseRequestLifecycleCallbacks callbacks;
    callbacks.execution.enqueuePrefill = [](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) {};
    callbacks.execution.enqueueDecode = [&](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) { ++decodeEnqueues; };
    callbacks.execution.completePrefill = [](rt::PhaseWorkItem const& item) {
        return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, true};
    };
    callbacks.execution.completeDecode
        = [](rt::PhaseWorkItem const& item) { return rt::PhaseDecodeCompletion{item.tokenCount + 1, true}; };
    callbacks.onTerminal = [&](rt::PhaseRequestSnapshot const& snapshot) { terminals.push_back(snapshot); };

    {
        rt::PhaseRequestLifecycle lifecycle(
            1, rt::PhaseQueueSchedulerConfig{}, std::move(callbacks), prefillStream, decodeStream);
        EXPECT_EQ(lifecycle.submit(77, 2), 0);
        lifecycle.runUntilIdle(2);
        EXPECT_TRUE(lifecycle.empty());
        EXPECT_EQ(lifecycle.availableSlotCount(), 1);
        EXPECT_EQ(decodeEnqueues, 0);
        ASSERT_EQ(terminals.size(), 1U);
        EXPECT_EQ(terminals[0].requestId, 77U);
        EXPECT_EQ(terminals[0].kvSlotId, -1);
        EXPECT_EQ(terminals[0].kvLength, 2);
        EXPECT_EQ(terminals[0].status, rt::PhaseRequestStatus::kFinished);
    }

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseContextBatchAdapterTest, PacksScattersAndRestoresStableSlotBindings)
{
    int32_t const maxSlots = 4;
    rt::HybridCacheManager cacheManager = makeIndexedManager(maxSlots);
    std::vector<int32_t> initialLengths{10, 20, 30, 40};
    rt::Tensor hostLengths({maxSlots}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(hostLengths.rawPointer(), initialLengths.data(), initialLengths.size() * sizeof(int32_t));
    cacheManager.resetForNewSequences(hostLengths, nullptr);

    rt::TensorMap tensorMap;
    rt::PhaseBatchState previousBindings(maxSlots, "phase_context_previous_bindings");
    previousBindings.bind(tensorMap);

    rt::DecodingInferenceContext first;
    first.initialize(2, 8, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    first.rawBatchedInputIds = {{1, 2}, {3, 4, 5}};
    first.tokenIds = {{1, 2, 11}, {3, 4, 5, 12}};
    first.effectivePrefillLengths = {2, 3};
    first.currentGenerateLengths = {1, 1};

    rt::DecodingInferenceContext second;
    second.initialize(1, 8, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    second.rawBatchedInputIds = {{6, 7}};
    second.tokenIds = {{6, 7, 13}};
    second.effectivePrefillLengths = {2};
    second.currentGenerateLengths = {1};

    rt::PhaseContextBatchAdapter adapter(2, cacheManager, tensorMap, "phase_context_adapter_test");
    std::vector<rt::PhaseContextRow> const rows{{101, &first, 1, 3, 40}, {202, &second, 0, 0, 10}};
    adapter.packDecode(rows, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    ASSERT_TRUE(adapter.packed());
    ASSERT_EQ(adapter.workItems().size(), 2U);
    EXPECT_EQ(adapter.workItems()[0].kvSlotId, 3);
    EXPECT_EQ(adapter.workItems()[1].kvSlotId, 0);
    rt::DecodingInferenceContext& packed = adapter.packedContext();
    ASSERT_NE(packed.phaseBatchState, nullptr);
    EXPECT_EQ(packed.tokenIds[0], (std::vector<int32_t>{12}));
    EXPECT_EQ(packed.tokenIds[1], (std::vector<int32_t>{13}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(adapter.tokenIds()), (std::vector<int32_t>{12, 13}));
    EXPECT_EQ(packed.batchIndexMapping, (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(packed.phaseBatchState->slotIds()), (std::vector<int32_t>{3, 0}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(packed.phaseBatchState->lengths()), (std::vector<int32_t>{40, 10}));
    EXPECT_EQ(tensorMap.get(binding_names::kKVSlotIds), &packed.phaseBatchState->slotIds());
    EXPECT_EQ(tensorMap.get(binding_names::kKVCacheStartIndex), &packed.phaseBatchState->lengths());

    packed.tokenIds[0].push_back(91);
    packed.tokenIds[1].push_back(92);
    ++packed.currentGenerateLengths[0];
    ++packed.currentGenerateLengths[1];
    adapter.scatterDecode();

    EXPECT_FALSE(adapter.packed());
    EXPECT_EQ(first.tokenIds[0], (std::vector<int32_t>{1, 2, 11}));
    EXPECT_EQ(first.tokenIds[1], (std::vector<int32_t>{3, 4, 5, 12, 91}));
    EXPECT_EQ(second.tokenIds[0], (std::vector<int32_t>{6, 7, 13, 92}));
    EXPECT_EQ(first.currentGenerateLengths, (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(second.currentGenerateLengths, (std::vector<int32_t>{2}));
    EXPECT_EQ(tensorMap.get(binding_names::kKVSlotIds), &previousBindings.slotIds());
    EXPECT_EQ(tensorMap.get(binding_names::kKVCacheStartIndex), &previousBindings.lengths());
}

TEST(PhaseContextBatchAdapterTest, RejectsIncompatibleOrDuplicateDecodeRows)
{
    rt::HybridCacheManager cacheManager = makeIndexedManager(2);
    std::vector<int32_t> initialLengths{5, 7};
    rt::Tensor hostLengths({2}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(hostLengths.rawPointer(), initialLengths.data(), initialLengths.size() * sizeof(int32_t));
    cacheManager.resetForNewSequences(hostLengths, nullptr);
    rt::TensorMap tensorMap;
    tensorMap.set(binding_names::kKVSlotIds, cacheManager.getKVSlotIds());
    tensorMap.set(binding_names::kKVCacheStartIndex, cacheManager.getKVCacheLengths());

    rt::DecodingInferenceContext first;
    first.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    first.rawBatchedInputIds = {{1}};
    first.tokenIds = {{1, 10}};
    first.effectivePrefillLengths = {1};

    rt::DecodingInferenceContext second;
    second.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    second.rawBatchedInputIds = {{2}};
    second.tokenIds = {{2, 20}};
    second.effectivePrefillLengths = {1};
    second.temperature = 0.5F;

    rt::PhaseContextBatchAdapter adapter(2, cacheManager, tensorMap, "phase_context_adapter_reject_test");
    EXPECT_THROW(adapter.packDecode({{1, &first, 0, 0, 5}, {2, &second, 0, 1, 7}}, nullptr), std::runtime_error);

    second.temperature = first.temperature;
    EXPECT_THROW(adapter.packDecode({{1, &first, 0, 0, 5}, {2, &second, 0, 0, 5}}, nullptr), std::runtime_error);
}

TEST(PhasePrefillContextBatchAdapterTest, PacksPromptSlicesAndRestoresStableSlotBindings)
{
    int32_t const maxSlots = 4;
    rt::HybridCacheManager cacheManager = makeIndexedManager(maxSlots);
    std::vector<int32_t> initialLengths{10, 20, 30, 40};
    rt::Tensor hostLengths({maxSlots}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(hostLengths.rawPointer(), initialLengths.data(), initialLengths.size() * sizeof(int32_t));
    cacheManager.resetForNewSequences(hostLengths, nullptr);

    rt::TensorMap tensorMap;
    rt::PhaseBatchState previousBindings(maxSlots, "phase_prefill_previous_bindings");
    previousBindings.bind(tensorMap);

    rt::DecodingInferenceContext first;
    first.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    first.rawBatchedInputIds = {{10, 11, 12, 13, 14}};
    first.tokenIds = first.rawBatchedInputIds;

    rt::DecodingInferenceContext second;
    second.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    second.rawBatchedInputIds = {{20, 21, 22, 23, 24}};
    second.tokenIds = second.rawBatchedInputIds;

    rt::PhasePrefillContextBatchAdapter adapter(2, 4, cacheManager, tensorMap, "phase_prefill_adapter_test");
    std::vector<rt::PhasePrefillContextRow> const rows{{101, &first, 0, 3, 2, 2, 5}, {202, &second, 0, 0, 2, 2, 5}};
    adapter.pack(rows, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    ASSERT_TRUE(adapter.packed());
    EXPECT_EQ(adapter.batchSize(), 2);
    EXPECT_EQ(adapter.chunkLength(), 2);
    EXPECT_FALSE(adapter.initialChunk());
    EXPECT_EQ(copyDeviceToHost<int32_t>(adapter.tokenIds()), (std::vector<int32_t>{12, 13, 22, 23}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(adapter.phaseBatchState().slotIds()), (std::vector<int32_t>{3, 0}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(adapter.phaseBatchState().lengths()), (std::vector<int32_t>{40, 10}));
    EXPECT_EQ(tensorMap.get(binding_names::kKVSlotIds), &adapter.phaseBatchState().slotIds());
    EXPECT_EQ(tensorMap.get(binding_names::kKVCacheStartIndex), &adapter.phaseBatchState().lengths());

    adapter.phaseBatchState().commit(cacheManager, 2, nullptr);
    adapter.complete();
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    EXPECT_EQ(tensorMap.get(binding_names::kKVSlotIds), &previousBindings.slotIds());
    EXPECT_EQ(tensorMap.get(binding_names::kKVCacheStartIndex), &previousBindings.lengths());
    EXPECT_EQ(
        copyDeviceToHost<int32_t>(cacheManager.getGlobalKVCacheLengths()), (std::vector<int32_t>{12, 20, 30, 42}));

    EXPECT_THROW(adapter.pack({{1, &first, 0, 0, 0, 2, 5}, {2, &second, 0, 1, 0, 1, 5}}, nullptr), std::runtime_error);
    EXPECT_THROW(adapter.pack({{1, &first, 0, 0, 0, 2, 5}, {2, &second, 0, 1, 2, 2, 5}}, nullptr), std::runtime_error);
}

TEST(PhaseContextServingFacadeTest, AdmitsPacksScattersAndReusesReleasedSlots)
{
    rt::HybridCacheManager cacheManager = makeIndexedManager(2);
    // Seed stale lengths to prove first-chunk admission clears only newly leased slots.
    std::vector<int32_t> initialLengths{9, 7};
    rt::Tensor hostLengths({2}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(hostLengths.rawPointer(), initialLengths.data(), initialLengths.size() * sizeof(int32_t));

    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));
    cacheManager.resetForNewSequences(hostLengths, prefillStream);
    CUDA_CHECK(cudaStreamSynchronize(prefillStream));

    rt::TensorMap decodeTensorMap;
    decodeTensorMap.set(binding_names::kKVSlotIds, cacheManager.getKVSlotIds());
    decodeTensorMap.set(binding_names::kKVCacheStartIndex, cacheManager.getKVCacheLengths());
    rt::PhaseBatchState prefillState(2, "phase_serving_prefill_test");

    rt::DecodingInferenceContext first;
    first.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", decodeStream);
    first.rawBatchedInputIds = {{1, 2}};
    first.tokenIds = {{1, 2, 10}};
    first.effectivePrefillLengths = {2};
    first.currentGenerateLengths = {1};

    rt::DecodingInferenceContext second;
    second.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", decodeStream);
    second.rawBatchedInputIds = {{3, 4}};
    second.tokenIds = {{3, 4, 20}};
    second.effectivePrefillLengths = {2};
    second.currentGenerateLengths = {1};

    rt::DecodingInferenceContext third;
    third.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", decodeStream);
    third.rawBatchedInputIds = {{5, 6}};
    third.tokenIds = {{5, 6, 30}};
    third.effectivePrefillLengths = {2};
    third.currentGenerateLengths = {1};

    rt::DecodingInferenceContext fourth;
    fourth.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", decodeStream);
    fourth.rawBatchedInputIds = {{7, 8}};
    fourth.tokenIds = {{7, 8, 40}};
    fourth.effectivePrefillLengths = {2};
    fourth.currentGenerateLengths = {1};

    std::unordered_map<uint64_t, int32_t> const finishLengths{{101, 2}, {202, 3}, {303, 2}, {404, 2}};
    std::vector<std::vector<int32_t>> decodeSlotBatches;
    std::vector<rt::PhaseAdmissionResult> admissions;
    std::vector<rt::PhaseRequestSnapshot> terminals;
    int32_t generatedToken{100};

    rt::PhaseContextServingCallbacks callbacks;
    callbacks.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream) {
        ASSERT_FALSE(batch.empty());
        int32_t const chunkLength = batch.front().tokenCount;
        ASSERT_TRUE(std::all_of(batch.begin(), batch.end(),
            [chunkLength](rt::PhaseWorkItem const& item) { return item.tokenCount == chunkLength; }));
        prefillState.prepare(batch, cacheManager, stream);
        prefillState.commit(cacheManager, chunkLength, stream);
    };
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) { return item.tokenOffset + item.tokenCount; };
    callbacks.enqueueDecode = [&](rt::DecodingInferenceContext& packed) {
        ASSERT_NE(packed.phaseBatchState, nullptr);
        packed.phaseBatchState->commit(cacheManager, 1, packed.stream);
    };
    callbacks.completeDecode = [&](rt::DecodingInferenceContext& packed) {
        decodeSlotBatches.push_back(copyDeviceToHost<int32_t>(packed.phaseBatchState->slotIds()));
        for (int32_t row = 0; row < packed.activeBatchSize; ++row)
        {
            packed.tokenIds[row].push_back(generatedToken++);
            ++packed.currentGenerateLengths[row];
        }
    };
    callbacks.isDecodeFinished = [&](uint64_t requestId, rt::DecodingInferenceContext const& context, int32_t row) {
        return context.currentGenerateLengths[static_cast<size_t>(row)] >= finishLengths.at(requestId);
    };
    callbacks.onTerminal = [&](rt::PhaseRequestSnapshot const& snapshot) { terminals.push_back(snapshot); };
    callbacks.onAdmission = [&](rt::PhaseAdmissionResult const& result) { admissions.push_back(result); };

    rt::PhaseQueueSchedulerConfig schedulerConfig;
    schedulerConfig.maxPrefillBatchSize = 2;
    schedulerConfig.maxDecodeBatchSize = 2;
    schedulerConfig.maxPrefillChunkTokens = 1;
    rt::PhaseContextServingFacade facade(2, schedulerConfig, std::move(callbacks), cacheManager, decodeTensorMap,
        prefillStream, decodeStream, rt::PhaseStreamExecutionMode::kIndependentContextsConcurrent, nullptr, 0, 1);

    EXPECT_EQ(facade.submit(101, first, 0, 2), 0);
    EXPECT_EQ(facade.submit(202, second, 0, 2), 1);
    rt::PhaseAdmissionResult const pending = facade.submitOrQueue(303, third, 0, 2);
    EXPECT_EQ(pending.status, rt::PhaseAdmissionStatus::kPending);
    EXPECT_EQ(pending.kvSlotId, -1);
    EXPECT_EQ(facade.pendingRequestCount(), 1U);
    ASSERT_TRUE(facade.request(303).has_value());
    EXPECT_EQ(facade.request(303)->status, rt::PhaseRequestStatus::kPending);
    EXPECT_THROW(facade.submitOrQueue(404, fourth, 0, 2), std::runtime_error);
    EXPECT_EQ(facade.registeredRequestCount(), 3U);
    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();
    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();
    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();

    ASSERT_EQ(terminals.size(), 1U);
    EXPECT_EQ(terminals[0].requestId, 101U);
    EXPECT_EQ(terminals[0].status, rt::PhaseRequestStatus::kFinished);
    EXPECT_EQ(facade.availableSlotCount(), 0);
    EXPECT_EQ(facade.pendingRequestCount(), 0U);
    EXPECT_EQ(facade.registeredRequestCount(), 2U);
    ASSERT_EQ(admissions.size(), 2U);
    EXPECT_EQ(admissions[0].status, rt::PhaseAdmissionStatus::kPending);
    EXPECT_EQ(admissions[1].status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(admissions[1].requestId, 303U);
    EXPECT_EQ(admissions[1].kvSlotId, 0);

    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();
    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();
    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();

    EXPECT_TRUE(facade.empty());
    EXPECT_EQ(facade.activeRequestCount(), 0U);
    EXPECT_EQ(facade.registeredRequestCount(), 0U);
    EXPECT_EQ(facade.availableSlotCount(), 2);
    EXPECT_EQ(decodeSlotBatches, (std::vector<std::vector<int32_t>>{{0, 1}, {1}, {0}}));
    EXPECT_EQ(first.tokenIds[0].size(), 4U);
    EXPECT_EQ(second.tokenIds[0].size(), 5U);
    EXPECT_EQ(third.tokenIds[0].size(), 4U);
    EXPECT_EQ(copyDeviceToHost<int32_t>(cacheManager.getGlobalKVCacheLengths()), (std::vector<int32_t>{3, 4}));

    EXPECT_EQ(facade.submit(404, first, 0, 2), 0);
    EXPECT_EQ(facade.submit(405, fourth, 0, 2), 1);
    rt::PhaseAdmissionResult const cancelledPending = facade.submitOrQueue(505, second, 0, 2);
    EXPECT_EQ(cancelledPending.status, rt::PhaseAdmissionStatus::kPending);
    EXPECT_TRUE(facade.cancel(505));
    EXPECT_EQ(facade.pendingRequestCount(), 0U);
    EXPECT_EQ(facade.registeredRequestCount(), 2U);
    EXPECT_EQ(terminals.back().requestId, 505U);
    EXPECT_EQ(terminals.back().kvSlotId, -1);
    EXPECT_EQ(terminals.back().status, rt::PhaseRequestStatus::kCancelled);
    EXPECT_TRUE(facade.cancel(405));
    EXPECT_TRUE(facade.cancel(404));
    EXPECT_EQ(facade.registeredRequestCount(), 0U);
    EXPECT_EQ(terminals.back().requestId, 404U);
    EXPECT_EQ(terminals.back().status, rt::PhaseRequestStatus::kCancelled);

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

} // namespace
