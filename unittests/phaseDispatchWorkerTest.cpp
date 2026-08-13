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
#include "runtime/scheduling/gemma4PhaseVisionAdapter.h"
#include "runtime/scheduling/phaseBatchState.h"
#include "runtime/scheduling/phaseContextBatchAdapter.h"
#include "runtime/scheduling/phaseContextServingFacade.h"
#include "runtime/scheduling/phaseEncoderDispatchWorker.h"
#include "runtime/scheduling/phaseGreedySampler.h"
#include "runtime/scheduling/phaseKernelGroupRecorder.h"
#include "runtime/scheduling/phasePrefillContextBatchAdapter.h"
#include "runtime/scheduling/phaseRequestLifecycle.h"
#include "runtime/scheduling/phaseThreeCoordinator.h"
#include "testUtils.h"

#include <array>
#include <cstring>
#include <gtest/gtest.h>
#include <numeric>
#include <unordered_map>
#include <utility>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

class FakeGemma4VisionRunner : public rt::MultimodalRunner
{
public:
    FakeGemma4VisionRunner()
    {
        mModelType = multimodal::ModelType::GEMMA4_VISION;
        mOutputEmbedding = rt::Tensor({3, 4}, rt::DeviceType::kGPU, nvinfer1::DataType::kHALF, "fake_visual_embedding");
    }

    bool preprocess(rt::LLMGenerationRequest const&, std::vector<std::vector<int32_t>>& batchedInputIds,
        tokenizer::Tokenizer const*, rt::OptionalOutputTensor, cudaStream_t, bool) override
    {
        batchedInputIds = {{1, 2, 3, 4}};
        return true;
    }

    bool infer(cudaStream_t stream) override
    {
        CUDA_CHECK(cudaMemsetAsync(mOutputEmbedding.rawPointer(), 0, mOutputEmbedding.getMemoryCapacity(), stream));
        return true;
    }

    bool validateAndFillConfig(std::string const&) override
    {
        return true;
    }

    bool allocateBuffer(cudaStream_t) override
    {
        return true;
    }
};

rt::HybridCacheManager makeIndexedManager(int32_t maxBatchSize)
{
    rt::HybridCacheManager::Config config{};
    config.layerTypes = {rt::HybridCacheManager::LayerType::kAttention};
    config.kvConfig
        = rt::KVCacheManager::Config{1, maxBatchSize, 64, {rt::KVLayerConfig{1, 64}}, DataType::kHALF, false, 0, 128};
    config.mambaConfig.maxBatchSize = maxBatchSize;
    config.indexedKVCache = true;
    config.maxBatchSize = maxBatchSize;
    return rt::HybridCacheManager(config, nullptr);
}

rt::HybridCacheManager makeIndexedPagedManager(int32_t maxBatchSize, int32_t pageBundles)
{
    constexpr int32_t kMAX_SEQUENCE_LENGTH = 256;
    rt::HybridCacheManager::Config config{};
    config.layerTypes = {rt::HybridCacheManager::LayerType::kAttention};
    config.kvConfig = rt::KVCacheManager::Config{
        1, maxBatchSize, kMAX_SEQUENCE_LENGTH, {rt::KVLayerConfig{1, 64}}, DataType::kHALF, true, pageBundles, 128};
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
        EXPECT_EQ(prefillEnqueues, prefillBatchCompletions)
            << "Shared-context decode enqueue ran before prefill host completion";
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
    CUcontext expectedCudaContext{};
    CUDA_DRIVER_CHECK(cuStreamGetCtx(prefillStream, &expectedCudaContext));
    EXPECT_EQ(worker.cudaContext(), expectedCudaContext);
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

TEST(PhaseExecutionSafetyContractTest, ValidatesSharedAndIndependentResourceIdentity)
{
    int identities[6]{};
    auto const independent = rt::PhaseExecutionSafetyContract::independent(
        {&identities[0], &identities[1], &identities[2]}, {&identities[3], &identities[4], &identities[5]});
    EXPECT_NO_THROW(independent.validate(rt::PhaseTensorRTContextMode::kIndependentConcurrent));
    EXPECT_TRUE(independent.provesIndependentResources());

    auto const aliasedWorkspace = rt::PhaseExecutionSafetyContract::independent(
        {&identities[0], &identities[1], &identities[2]}, {&identities[3], &identities[1], &identities[5]});
    EXPECT_THROW(aliasedWorkspace.validate(rt::PhaseTensorRTContextMode::kIndependentConcurrent), std::runtime_error);
    EXPECT_NO_THROW(rt::PhaseExecutionSafetyContract{}.validate(rt::PhaseTensorRTContextMode::kSharedSerialized));
    EXPECT_NO_THROW(rt::PhaseExecutionSafetyContract::shared(&identities[0])
            .validate(rt::PhaseTensorRTContextMode::kSharedSerialized));
    auto const mismatchedShared = rt::PhaseExecutionSafetyContract::independent(
        {&identities[0], nullptr, nullptr}, {&identities[1], nullptr, nullptr});
    EXPECT_THROW(mismatchedShared.validate(rt::PhaseTensorRTContextMode::kSharedSerialized), std::runtime_error);
}

TEST(PhaseDispatchWorkerTest, ConcurrentModeEnqueuesOnlyWithIndependentResourceProof)
{
    rt::PhaseQueueScheduler scheduler;
    scheduler.enqueuePrefill({1, 32});
    scheduler.enqueueDecode({2, 64});
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    int prefillEnqueues{};
    int decodeEnqueues{};
    rt::PhaseDispatchWorkerCallbacks callbacks;
    callbacks.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) { ++prefillEnqueues; };
    callbacks.enqueueDecode = [&](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) { ++decodeEnqueues; };
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) {
        return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, true};
    };
    callbacks.completeDecode
        = [](rt::PhaseWorkItem const& item) { return rt::PhaseDecodeCompletion{item.tokenCount + 1, true}; };
    int identities[6]{};
    auto const contract = rt::PhaseExecutionSafetyContract::independent(
        {&identities[0], &identities[1], &identities[2]}, {&identities[3], &identities[4], &identities[5]});
    EXPECT_THROW(
        {
            rt::PhaseDispatchWorker invalidWorker(scheduler, callbacks, prefillStream, prefillStream,
                rt::PhaseTensorRTContextMode::kIndependentConcurrent, contract);
        },
        std::runtime_error);
    rt::PhaseDispatchWorker worker(scheduler, std::move(callbacks), prefillStream, decodeStream,
        rt::PhaseTensorRTContextMode::kIndependentConcurrent, contract);

    EXPECT_TRUE(worker.dispatchNext());
    CUcontext currentCudaContext{};
    CUDA_DRIVER_CHECK(cuCtxGetCurrent(&currentCudaContext));
    EXPECT_EQ(worker.cudaContext(), currentCudaContext);
    EXPECT_EQ(prefillEnqueues, 1);
    EXPECT_EQ(decodeEnqueues, 1);
    worker.wait();
    EXPECT_TRUE(scheduler.empty());

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseEncoderExecutionSafetyContractTest, RejectsAliasedLlmResources)
{
    int identities[9]{};
    rt::PhaseEncoderExecutionSafetyContract const valid{{&identities[0], &identities[1], &identities[2]},
        {{&identities[3], &identities[4], &identities[5]}, {&identities[6], &identities[7], &identities[8]}}};
    EXPECT_NO_THROW(valid.validate());

    rt::PhaseEncoderExecutionSafetyContract const aliased{
        {&identities[0], &identities[1], &identities[2]}, {{&identities[3], &identities[1], &identities[5]}}};
    EXPECT_THROW(aliased.validate(), std::runtime_error);
}

TEST(PhaseEncoderDispatchWorkerTest, BatchesAndHandsOffToPrefillAfterCudaEvent)
{
    rt::PhaseQueueSchedulerConfig llmConfig;
    llmConfig.maxPrefillBatchSize = 2;
    rt::PhaseQueueScheduler scheduler(llmConfig);
    cudaStream_t encoderStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&encoderStream, cudaStreamNonBlocking));
    rt::Tensor marker({4}, rt::DeviceType::kGPU, DataType::kINT32, "encoder_event_marker");

    int32_t enqueueCount{};
    int32_t batchCompletionCount{};
    std::vector<rt::PhaseEncoderDispatchMetrics> metrics;
    rt::PhaseEncoderDispatchWorkerCallbacks callbacks;
    callbacks.enqueueEncoder = [&](std::vector<rt::PhaseEncoderWorkItem> const& batch, cudaStream_t stream) {
        EXPECT_EQ(batch.size(), 2U);
        ++enqueueCount;
        CUDA_CHECK(cudaMemsetAsync(marker.rawPointer(), 0, marker.getMemoryCapacity(), stream));
    };
    callbacks.completeEncoderBatch = [&](std::vector<rt::PhaseEncoderWorkItem> const& batch) {
        EXPECT_EQ(batch.size(), 2U);
        EXPECT_EQ(enqueueCount, 1);
        ++batchCompletionCount;
    };
    callbacks.completeEncoder = [](rt::PhaseEncoderWorkItem const& item) {
        return rt::PhaseWorkItem{item.requestId, item.inputUnits, item.kvSlotId, 0, item.inputUnits};
    };
    callbacks.onMetrics = [&](rt::PhaseEncoderDispatchMetrics const& sample) { metrics.push_back(sample); };

    int identities[9]{};
    rt::PhaseEncoderExecutionSafetyContract contract{{&identities[0], &identities[1], &identities[2]},
        {{&identities[3], &identities[4], &identities[5]}, {&identities[6], &identities[7], &identities[8]}}};
    rt::PhaseEncoderQueueConfig encoderConfig;
    encoderConfig.maxBatchSize = 2;
    encoderConfig.maxQueuedRequests = 2;
    rt::PhaseEncoderDispatchWorker worker(
        scheduler, encoderConfig, std::move(callbacks), encoderStream, std::move(contract));

    worker.submit({10, 4, 0});
    worker.submit({20, 5, 1});
    EXPECT_THROW(worker.submit({30, 3, 2}), std::runtime_error);
    EXPECT_TRUE(worker.cancel(20));
    EXPECT_THROW(worker.submit({40, 2, 0}), std::runtime_error);
    worker.submit({30, 3, 2});
    EXPECT_EQ(worker.queueSize(), 2U);
    EXPECT_TRUE(worker.dispatchNext());
    EXPECT_TRUE(worker.busy());
    EXPECT_FALSE(worker.cancel(10));
    worker.wait();

    EXPECT_TRUE(worker.empty());
    EXPECT_EQ(worker.dispatchCount(), 1U);
    EXPECT_EQ(batchCompletionCount, 1);
    ASSERT_EQ(metrics.size(), 1U);
    EXPECT_EQ(metrics.front().batchSize, 2);
    EXPECT_EQ(metrics.front().inputUnits, 7);
    EXPECT_GE(metrics.front().queueWaitUs, 0.0);
    EXPECT_GE(metrics.front().gpuMs, 0.0F);
    EXPECT_EQ(scheduler.prefillQueueSize(), 2U);
    EXPECT_TRUE(scheduler.hasRequest(10));
    EXPECT_TRUE(scheduler.hasRequest(30));
    EXPECT_FALSE(scheduler.hasRequest(20));

    for (int32_t dispatch = 0; dispatch < 2; ++dispatch)
    {
        rt::PhaseDispatchPlan const plan = scheduler.next();
        ASSERT_EQ(plan.prefillBatch.size(), 1U);
        rt::PhaseWorkItem const& item = plan.prefillBatch.front();
        scheduler.completePrefill(item, item.tokenCount, true);
    }
    EXPECT_TRUE(scheduler.empty());
    CUDA_CHECK(cudaStreamDestroy(encoderStream));
}

TEST(Gemma4PhaseVisionAdapterTest, CopiesActualRunnerOutputBeforePrefillHandoff)
{
    cudaStream_t encoderStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&encoderStream, cudaStreamNonBlocking));
    FakeGemma4VisionRunner runner;
    tokenizer::Tokenizer tokenizer;
    rt::PhaseKernelGroupRecorder kernelGroups;
    rt::Gemma4PhaseVisionAdapter adapter(runner, tokenizer, &kernelGroups);
    rt::LLMGenerationRequest request;
    request.requests.resize(1);
    rt::DecodingInferenceContext context;
    context.initialize(1, 1, std::nullopt, {}, "", encoderStream);
    adapter.registerRequest(10, request, context);

    rt::PhaseQueueScheduler scheduler;
    int identities[6]{};
    rt::PhaseEncoderExecutionSafetyContract safety{
        {&identities[0], &identities[1], &identities[2]}, {{&identities[3], &identities[4], &identities[5]}}};
    rt::PhaseEncoderDispatchWorker worker(scheduler, {}, adapter.makeCallbacks(), encoderStream, std::move(safety));
    worker.submit({10, 1, 0});
    ASSERT_TRUE(worker.dispatchNext());
    worker.wait();
    kernelGroups.drain();
    ASSERT_EQ(kernelGroups.samples().size(), 2U);
    EXPECT_EQ(kernelGroups.samples()[0].group, rt::PhaseKernelGroup::kEncoderPreprocess);
    EXPECT_EQ(kernelGroups.samples()[1].group, rt::PhaseKernelGroup::kEncoderEngine);

    rt::PhaseDispatchPlan const plan = scheduler.next();
    ASSERT_EQ(plan.prefillBatch.size(), 1U);
    EXPECT_EQ(plan.prefillBatch.front().tokenCount, 4);
    EXPECT_FALSE(plan.prefillBatch.front().allowChunkedPrefill);
    ASSERT_TRUE(context.visualEmbeddings.has_value());
    EXPECT_EQ(context.visualEmbeddings->get().getShape(), rt::Coords({3, 4}));
    scheduler.completePrefill(plan.prefillBatch.front(), 4, true);
    adapter.release(10);
    CUDA_CHECK(cudaStreamDestroy(encoderStream));
}

TEST(PhaseThreeCoordinatorTest, ConnectsEncoderPrefillAndDecodeOnOneCudaContext)
{
    rt::PhaseQueueScheduler scheduler;
    cudaStream_t encoderStream{};
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&encoderStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));
    rt::Tensor marker({3}, rt::DeviceType::kGPU, DataType::kINT32, "three_phase_marker");

    int identities[9]{};
    rt::PhaseEncoderDispatchWorkerCallbacks encoderCallbacks;
    encoderCallbacks.enqueueEncoder = [&](std::vector<rt::PhaseEncoderWorkItem> const&, cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(marker.rawPointer(), 1, marker.getMemoryCapacity(), stream));
    };
    encoderCallbacks.completeEncoder = [](rt::PhaseEncoderWorkItem const& item) {
        return rt::PhaseWorkItem{item.requestId, 2, item.kvSlotId, 0, 2};
    };
    rt::PhaseEncoderExecutionSafetyContract encoderSafety{{&identities[0], &identities[1], &identities[2]},
        {{&identities[3], &identities[4], &identities[5]}, {&identities[6], &identities[7], &identities[8]}}};
    rt::PhaseEncoderDispatchWorker encoderWorker(
        scheduler, {}, std::move(encoderCallbacks), encoderStream, std::move(encoderSafety));

    rt::PhaseDispatchWorkerCallbacks llmCallbacks;
    llmCallbacks.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const&, cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(marker.rawPointer(), 2, marker.getMemoryCapacity(), stream));
    };
    llmCallbacks.enqueueDecode = [&](std::vector<rt::PhaseWorkItem> const&, cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(marker.rawPointer(), 3, marker.getMemoryCapacity(), stream));
    };
    llmCallbacks.completePrefill
        = [](rt::PhaseWorkItem const& item) { return rt::PhasePrefillCompletion{item.promptTokenCount, false}; };
    llmCallbacks.completeDecode
        = [](rt::PhaseWorkItem const& item) { return rt::PhaseDecodeCompletion{item.tokenCount + 1, true}; };
    rt::PhaseExecutionSafetyContract llmSafety = rt::PhaseExecutionSafetyContract::independent(
        {&identities[3], &identities[4], &identities[5]}, {&identities[6], &identities[7], &identities[8]});
    rt::PhaseDispatchWorker llmWorker(scheduler, std::move(llmCallbacks), prefillStream, decodeStream,
        rt::PhaseTensorRTContextMode::kIndependentConcurrent, llmSafety);
    rt::PhaseThreeCoordinator coordinator(encoderWorker, llmWorker);

    encoderWorker.submit({10, 1, 0});
    coordinator.runUntilIdle(3);
    EXPECT_TRUE(coordinator.empty());
    EXPECT_EQ(coordinator.encoderDispatchCount(), 1U);
    EXPECT_EQ(coordinator.llmDispatchCount(), 2U);
    EXPECT_FALSE(scheduler.hasRequest(10));

    CUDA_CHECK(cudaStreamDestroy(encoderStream));
    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseKernelGroupRecorderTest, SegmentsAndMeasuresCrossStreamHandoffs)
{
    cudaStream_t firstStream{};
    cudaStream_t secondStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&firstStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&secondStream, cudaStreamNonBlocking));
    rt::Tensor marker({4}, rt::DeviceType::kGPU, DataType::kINT32, "kernel_group_marker");
    rt::PhaseKernelGroupRecorder recorder;
    recorder.execute(7,
        {{rt::PhaseKernelGroup::kEncoderEngine, "vit", firstStream,
             [&](cudaStream_t stream) {
                 CUDA_CHECK(cudaMemsetAsync(marker.rawPointer(), 1, marker.getMemoryCapacity(), stream));
             }},
            {rt::PhaseKernelGroup::kPrefillEngine, "llm_prefill", secondStream,
                [&](cudaStream_t stream) {
                    CUDA_CHECK(cudaMemsetAsync(marker.rawPointer(), 2, marker.getMemoryCapacity(), stream));
                }},
            {rt::PhaseKernelGroup::kPrefillSample, "sample", secondStream, [&](cudaStream_t stream) {
                 CUDA_CHECK(cudaMemsetAsync(marker.rawPointer(), 3, marker.getMemoryCapacity(), stream));
             }}});
    EXPECT_EQ(recorder.pendingCount(), 3U);
    recorder.drain();
    ASSERT_EQ(recorder.samples().size(), 3U);
    EXPECT_EQ(recorder.samples()[0].dispatchIndex, 7U);
    EXPECT_EQ(recorder.samples()[0].name, "vit");
    EXPECT_EQ(recorder.samples()[1].group, rt::PhaseKernelGroup::kPrefillEngine);
    EXPECT_GE(recorder.samples()[2].gpuMs, 0.0F);

    CUDA_CHECK(cudaStreamDestroy(firstStream));
    CUDA_CHECK(cudaStreamDestroy(secondStream));
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
    std::vector<rt::PhaseDispatchMetrics> dispatches;
    std::vector<int32_t> releasedSlots;
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
    callbacks.execution.onDispatch = [&](rt::PhaseDispatchMetrics const& metrics) { dispatches.push_back(metrics); };
    callbacks.onSlotRelease = [&](int32_t slot) { releasedSlots.push_back(slot); };
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
        EXPECT_EQ(releasedSlots.size(), 3U);
        EXPECT_EQ(std::count(releasedSlots.begin(), releasedSlots.end(), 0), 1);
        EXPECT_EQ(std::count(releasedSlots.begin(), releasedSlots.end(), 1), 2);
        ASSERT_FALSE(dispatches.empty());
        EXPECT_TRUE(std::any_of(dispatches.begin(), dispatches.end(), [](rt::PhaseDispatchMetrics const& metrics) {
            return metrics.prefillBatchSize > 0 || metrics.decodeBatchSize > 0;
        }));
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

TEST(PhaseRequestLifecycleTest, DefersPrefillUntilEncoderHandoff)
{
    cudaStream_t encoderStream{};
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&encoderStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    int32_t prefillEnqueues{};
    rt::PhaseRequestLifecycleCallbacks lifecycleCallbacks;
    lifecycleCallbacks.execution.enqueuePrefill
        = [&](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) { ++prefillEnqueues; };
    lifecycleCallbacks.execution.enqueueDecode = [](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) {};
    lifecycleCallbacks.execution.completePrefill = [](rt::PhaseWorkItem const& item) {
        return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, true};
    };
    lifecycleCallbacks.execution.completeDecode
        = [](rt::PhaseWorkItem const& item) { return rt::PhaseDecodeCompletion{item.tokenCount + 1, true}; };

    rt::PhaseRequestLifecycle lifecycle(
        1, rt::PhaseQueueSchedulerConfig{}, std::move(lifecycleCallbacks), prefillStream, decodeStream);
    int32_t const slot = lifecycle.reserveForEncoder(55, 1);
    EXPECT_EQ(slot, 0);
    EXPECT_FALSE(lifecycle.empty());
    EXPECT_FALSE(lifecycle.hasQueuedWork());
    ASSERT_TRUE(lifecycle.request(55).has_value());
    EXPECT_EQ(lifecycle.request(55)->status, rt::PhaseRequestStatus::kEncoder);

    rt::Tensor marker({1}, rt::DeviceType::kGPU, DataType::kINT32, "deferred_encoder_marker");
    rt::PhaseEncoderDispatchWorkerCallbacks encoderCallbacks;
    encoderCallbacks.enqueueEncoder = [&](std::vector<rt::PhaseEncoderWorkItem> const&, cudaStream_t stream) {
        CUDA_CHECK(cudaMemsetAsync(marker.rawPointer(), 0, marker.getMemoryCapacity(), stream));
    };
    encoderCallbacks.completeEncoder = [](rt::PhaseEncoderWorkItem const& item) {
        rt::PhaseWorkItem work{item.requestId, 4, item.kvSlotId, 0, 4};
        work.allowChunkedPrefill = false;
        return work;
    };
    int identities[6]{};
    rt::PhaseEncoderExecutionSafetyContract safety{
        {&identities[0], &identities[1], &identities[2]}, {{&identities[3], &identities[4], &identities[5]}}};
    rt::PhaseEncoderDispatchWorker encoderWorker(
        {}, std::move(encoderCallbacks),
        [&](rt::PhaseWorkItem const& item) {
            EXPECT_EQ(item.kvSlotId, slot);
            lifecycle.beginPrefill(item.requestId, item.promptTokenCount, item.allowChunkedPrefill);
        },
        encoderStream, std::move(safety));
    encoderWorker.submit({55, 1, slot});
    ASSERT_TRUE(encoderWorker.dispatchNext());
    encoderWorker.wait();

    EXPECT_TRUE(lifecycle.hasQueuedWork());
    EXPECT_EQ(lifecycle.request(55)->status, rt::PhaseRequestStatus::kPrefill);
    lifecycle.runUntilIdle(1);
    EXPECT_EQ(prefillEnqueues, 1);
    EXPECT_TRUE(lifecycle.empty());
    EXPECT_EQ(lifecycle.availableSlotCount(), 1);

    CUDA_CHECK(cudaStreamDestroy(encoderStream));
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
    second.initialize(1, 2, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
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
    EXPECT_EQ(packed.maxGenerateLength, 8);
    EXPECT_EQ(packed.tokenIds[0], (std::vector<int32_t>{12}));
    EXPECT_EQ(packed.tokenIds[1], (std::vector<int32_t>{13}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(adapter.tokenIds()), (std::vector<int32_t>{12, 13}));
    EXPECT_EQ(packed.batchIndexMapping, (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(packed.phaseBatchState->slotIds()), (std::vector<int32_t>{3, 0}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(packed.phaseBatchState->lengths()), (std::vector<int32_t>{40, 10}));
    EXPECT_EQ(tensorMap.get(binding_names::kKVSlotIds), &packed.phaseBatchState->slotIds());
    EXPECT_EQ(tensorMap.get(binding_names::kKVCacheStartIndex), &packed.phaseBatchState->lengths());

    constexpr int32_t vocabSize = 128;
    rt::Tensor logits({2, vocabSize}, rt::DeviceType::kGPU, DataType::kFLOAT, "phase_adapter_sampler_logits");
    std::vector<float> hostLogits(2 * vocabSize, -10.0F);
    hostLogits[91] = 7.0F;
    hostLogits[vocabSize + 92] = 8.0F;
    CUDA_CHECK(
        cudaMemcpy(logits.rawPointer(), hostLogits.data(), hostLogits.size() * sizeof(float), cudaMemcpyHostToDevice));
    rt::PhaseGreedySampler sampler(2, vocabSize, {}, "phase_adapter_sampler");
    sampler.enqueue(logits, 2, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));
    sampler.completeDecode(adapter);
    adapter.scatterDecode();

    EXPECT_FALSE(adapter.packed());
    EXPECT_EQ(first.tokenIds[0], (std::vector<int32_t>{1, 2, 11}));
    EXPECT_EQ(first.tokenIds[1], (std::vector<int32_t>{3, 4, 5, 12, 91}));
    EXPECT_EQ(second.tokenIds[0], (std::vector<int32_t>{6, 7, 13, 92}));
    EXPECT_EQ(first.currentGenerateLengths, (std::vector<int32_t>{1, 2}));
    EXPECT_EQ(second.currentGenerateLengths, (std::vector<int32_t>{2}));
    EXPECT_EQ(first.finishedStates, (std::vector<int8_t>{0, 0}));
    EXPECT_EQ(second.finishedStates, (std::vector<int8_t>{1}));
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

TEST(PhasePrefillContextBatchAdapterTest, RightPadsRaggedRowsAndCommitsActualLengths)
{
    int32_t const maxSlots = 4;
    rt::HybridCacheManager cacheManager = makeIndexedManager(maxSlots);
    std::vector<int32_t> initialLengths{10, 20, 30, 40};
    rt::Tensor hostLengths({maxSlots}, rt::DeviceType::kCPU, DataType::kINT32);
    std::memcpy(hostLengths.rawPointer(), initialLengths.data(), initialLengths.size() * sizeof(int32_t));
    cacheManager.resetForNewSequences(hostLengths, nullptr);

    rt::TensorMap tensorMap;
    rt::PhaseBatchState previousBindings(maxSlots, "phase_ragged_prefill_previous_bindings");
    previousBindings.bind(tensorMap);

    rt::DecodingInferenceContext first;
    first.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    first.rawBatchedInputIds = {{10, 11, 12, 13, 14}};
    first.tokenIds = first.rawBatchedInputIds;

    rt::DecodingInferenceContext second;
    second.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    second.rawBatchedInputIds = {{20, 21, 22, 23, 24}};
    second.tokenIds = second.rawBatchedInputIds;

    rt::PhasePrefillContextBatchAdapter adapter(
        2, 4, cacheManager, tensorMap, "phase_ragged_prefill_adapter_test", true);
    adapter.pack({{101, &first, 0, 3, 2, 3, 5}, {202, &second, 0, 0, 4, 1, 5}}, nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    EXPECT_EQ(adapter.chunkLength(), 3);
    EXPECT_EQ(copyDeviceToHost<int32_t>(adapter.tokenIds()), (std::vector<int32_t>{12, 13, 14, 24, 0, 0}));

    rt::Tensor increments({2}, rt::DeviceType::kGPU, DataType::kINT32, "phase_ragged_prefill_increments");
    std::vector<int32_t> const hostIncrements{3, 1};
    CUDA_CHECK(cudaMemcpy(increments.rawPointer(), hostIncrements.data(), hostIncrements.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));
    adapter.phaseBatchState().commit(cacheManager, increments, nullptr);
    adapter.complete();
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    EXPECT_EQ(
        copyDeviceToHost<int32_t>(cacheManager.getGlobalKVCacheLengths()), (std::vector<int32_t>{11, 20, 30, 43}));
    EXPECT_EQ(tensorMap.get(binding_names::kKVSlotIds), &previousBindings.slotIds());
    EXPECT_EQ(tensorMap.get(binding_names::kKVCacheStartIndex), &previousBindings.lengths());
}

TEST(PhasePrefillContextBatchAdapterTest, PacksFixedChunksIntoOneTokenCarrierAndRestoresBindings)
{
    constexpr int32_t kMAX_SLOTS = 4;
    constexpr int32_t kCHUNK_TOKENS = 128;
    constexpr int32_t kFIRST_PROMPT_TOKENS = 200;
    constexpr int32_t kSECOND_PROMPT_TOKENS = 170;
    constexpr int32_t kFIRST_TAIL_TOKENS = kFIRST_PROMPT_TOKENS - kCHUNK_TOKENS;
    constexpr int32_t kSECOND_TAIL_TOKENS = kSECOND_PROMPT_TOKENS - kCHUNK_TOKENS;
    constexpr int32_t kTOTAL_TOKENS = kFIRST_TAIL_TOKENS + kSECOND_TAIL_TOKENS;
    constexpr int32_t kVOCAB_SIZE = 32;
    rt::HybridCacheManager cacheManager = makeIndexedPagedManager(kMAX_SLOTS, 8);
    rt::PhaseBatchState initialState(2, "phase_packed_prefill_initial_state", true);
    initialState.prepare(
        {{101, kCHUNK_TOKENS, 3, 0, kFIRST_PROMPT_TOKENS}, {202, kCHUNK_TOKENS, 0, 0, kSECOND_PROMPT_TOKENS}},
        cacheManager, nullptr);
    initialState.commit(cacheManager, kCHUNK_TOKENS, nullptr);

    rt::TensorMap tensorMap;
    rt::PhaseBatchState previousBindings(kMAX_SLOTS, "phase_packed_prefill_previous_bindings");
    previousBindings.bind(tensorMap);
    rt::Tensor previousLastTokenIds(
        {kMAX_SLOTS, 1}, rt::DeviceType::kGPU, DataType::kINT64, "phase_packed_prefill_previous_last_token_ids");
    tensorMap.set(binding_names::kLastTokenIds, previousLastTokenIds);

    rt::DecodingInferenceContext first;
    first.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    first.rawBatchedInputIds.resize(1);
    first.rawBatchedInputIds.front().resize(kFIRST_PROMPT_TOKENS);
    std::iota(first.rawBatchedInputIds.front().begin(), first.rawBatchedInputIds.front().end(), 1000);
    first.tokenIds = first.rawBatchedInputIds;

    rt::DecodingInferenceContext second;
    second.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    second.rawBatchedInputIds.resize(1);
    second.rawBatchedInputIds.front().resize(kSECOND_PROMPT_TOKENS);
    std::iota(second.rawBatchedInputIds.front().begin(), second.rawBatchedInputIds.front().end(), 2000);
    second.tokenIds = second.rawBatchedInputIds;

    rt::PhasePrefillContextBatchAdapter adapter(
        2, kCHUNK_TOKENS, cacheManager, tensorMap, "phase_packed_prefill_adapter_test", true, true);
    adapter.pack({{101, &first, 0, 3, kCHUNK_TOKENS, kFIRST_TAIL_TOKENS, kFIRST_PROMPT_TOKENS},
                     {202, &second, 0, 0, kCHUNK_TOKENS, kSECOND_TAIL_TOKENS, kSECOND_PROMPT_TOKENS}},
        nullptr);
    CUDA_CHECK(cudaStreamSynchronize(nullptr));

    EXPECT_EQ(adapter.batchSize(), 2);
    EXPECT_EQ(adapter.chunkLength(), kFIRST_TAIL_TOKENS);
    EXPECT_EQ(adapter.engineSequenceLength(), kTOTAL_TOKENS);
    EXPECT_EQ(adapter.tokenBatchSize(), 1);
    EXPECT_TRUE(adapter.usesPackedTokenLayout());
    EXPECT_EQ(adapter.tokenIds().getShape(), rt::Coords({1, kTOTAL_TOKENS}));
    EXPECT_EQ(adapter.lastTokenIds().getShape(), rt::Coords({1, 2}));
    std::vector<int32_t> expectedTokens;
    expectedTokens.insert(expectedTokens.end(), first.rawBatchedInputIds.front().begin() + kCHUNK_TOKENS,
        first.rawBatchedInputIds.front().end());
    expectedTokens.insert(expectedTokens.end(), second.rawBatchedInputIds.front().begin() + kCHUNK_TOKENS,
        second.rawBatchedInputIds.front().end());
    EXPECT_EQ(copyDeviceToHost<int32_t>(adapter.tokenIds()), expectedTokens);
    EXPECT_EQ(copyDeviceToHost<int64_t>(adapter.lastTokenIds()),
        (std::vector<int64_t>{kFIRST_TAIL_TOKENS - 1, kTOTAL_TOKENS - 1}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(adapter.phaseBatchState().slotIds()), (std::vector<int32_t>{3, 0}));
    EXPECT_EQ(copyDeviceToHost<int32_t>(adapter.phaseBatchState().lengths()),
        (std::vector<int32_t>{kCHUNK_TOKENS, kCHUNK_TOKENS}));
    EXPECT_EQ(tensorMap.get(binding_names::kLastTokenIds), &adapter.lastTokenIds());

    rt::Tensor logits({1, 2, kVOCAB_SIZE}, rt::DeviceType::kGPU, DataType::kFLOAT, "phase_packed_prefill_logits");
    adapter.reshapeOutputLogits(logits, kVOCAB_SIZE);
    EXPECT_EQ(logits.getShape(), rt::Coords({2, kVOCAB_SIZE}));
    adapter.complete();
    EXPECT_EQ(tensorMap.get(binding_names::kKVSlotIds), &previousBindings.slotIds());
    EXPECT_EQ(tensorMap.get(binding_names::kKVCacheStartIndex), &previousBindings.lengths());
    EXPECT_EQ(tensorMap.get(binding_names::kLastTokenIds), &previousLastTokenIds);
}

TEST(PhasePrefillContextBatchAdapterTest, PacksSingleAtomicMultimodalPrompt)
{
    rt::HybridCacheManager cacheManager = makeIndexedManager(2);
    rt::TensorMap tensorMap;
    rt::PhaseBatchState previousBindings(2, "phase_multimodal_prefill_previous_bindings");
    previousBindings.bind(tensorMap);
    rt::Tensor visualEmbeddings({4, 8}, rt::DeviceType::kGPU, DataType::kHALF, "phase_visual_embeddings");

    rt::DecodingInferenceContext context;
    context.initialize(1, 4, std::cref(visualEmbeddings), rt::OptionalInputTensors{}, "", nullptr);
    context.rawBatchedInputIds = {{10, 11, 12, 13}};
    context.tokenIds = context.rawBatchedInputIds;

    rt::PhasePrefillContextBatchAdapter adapter(2, 4, cacheManager, tensorMap, "phase_multimodal_prefill_test");
    adapter.pack({{101, &context, 0, 1, 0, 4, 4}}, nullptr);
    ASSERT_TRUE(adapter.visualEmbeddings().has_value());
    EXPECT_EQ(&adapter.visualEmbeddings()->get(), &visualEmbeddings);
    adapter.complete();

    EXPECT_THROW(adapter.pack({{101, &context, 0, 1, 0, 2, 4}}, nullptr), std::runtime_error);

    rt::DecodingInferenceContext textContext;
    textContext.initialize(1, 4, std::nullopt, rt::OptionalInputTensors{}, "", nullptr);
    textContext.rawBatchedInputIds = {{20, 21, 22, 23}};
    textContext.tokenIds = textContext.rawBatchedInputIds;
    EXPECT_THROW(adapter.pack({{101, &context, 0, 1, 0, 4, 4}, {202, &textContext, 0, 0, 0, 4, 4}}, nullptr),
        std::runtime_error);
}

TEST(PhaseContextBatchAdapterTest, AllowsVisualContextDuringDecode)
{
    rt::HybridCacheManager cacheManager = makeIndexedManager(1);
    rt::TensorMap tensorMap;
    rt::PhaseBatchState previousBindings(1, "phase_visual_decode_previous_bindings");
    previousBindings.bind(tensorMap);
    rt::Tensor visualEmbeddings({4, 8}, rt::DeviceType::kGPU, DataType::kHALF, "phase_decode_visual_embeddings");

    rt::DecodingInferenceContext context;
    context.initialize(1, 4, std::cref(visualEmbeddings), rt::OptionalInputTensors{}, "", nullptr);
    context.rawBatchedInputIds = {{1, 2}};
    context.tokenIds = {{1, 2, 3}};
    context.effectivePrefillLengths = {2};
    context.currentGenerateLengths = {1};

    rt::PhaseContextBatchAdapter adapter(1, cacheManager, tensorMap, "phase_visual_decode_test");
    EXPECT_NO_THROW(adapter.packDecode({{101, &context, 0, 0, 2}}, nullptr));
    adapter.scatterDecode();
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
    int resourceIdentities[6]{};
    auto const safetyContract = rt::PhaseExecutionSafetyContract::independent(
        {&resourceIdentities[0], &resourceIdentities[1], &resourceIdentities[2]},
        {&resourceIdentities[3], &resourceIdentities[4], &resourceIdentities[5]});
    rt::PhaseContextServingFacade facade(2, schedulerConfig, std::move(callbacks), cacheManager, decodeTensorMap,
        prefillStream, decodeStream, rt::PhaseTensorRTContextMode::kIndependentConcurrent, nullptr, 0, 1,
        safetyContract);
    std::vector<rt::PhaseRequestSnapshot> observedTerminals;
    size_t const observerId = facade.addTerminalObserver(
        [&](rt::PhaseRequestSnapshot const& snapshot) { observedTerminals.push_back(snapshot); });

    EXPECT_EQ(facade.submit(101, first, 0, 2), 0);
    EXPECT_EQ(facade.submit(202, second, 0, 2), 1);
    rt::PhaseSchedulingHints const pendingScheduling{2, 7000.0, 900.0};
    rt::PhaseAdmissionResult const pending = facade.submitOrQueue(303, third, 0, 2, pendingScheduling);
    EXPECT_EQ(pending.status, rt::PhaseAdmissionStatus::kPending);
    EXPECT_EQ(pending.kvSlotId, -1);
    EXPECT_EQ(facade.pendingRequestCount(), 1U);
    ASSERT_TRUE(facade.request(303).has_value());
    EXPECT_EQ(facade.request(303)->status, rt::PhaseRequestStatus::kPending);
    EXPECT_EQ(facade.request(303)->scheduling.priority, 2);
    EXPECT_DOUBLE_EQ(facade.request(303)->scheduling.ttftTargetUs, 7000.0);
    EXPECT_THROW(facade.submitOrQueue(404, fourth, 0, 2), std::runtime_error);
    EXPECT_EQ(facade.registeredRequestCount(), 3U);
    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();
    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();
    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();

    ASSERT_EQ(terminals.size(), 1U);
    ASSERT_EQ(observedTerminals.size(), 1U);
    EXPECT_EQ(terminals[0].requestId, 101U);
    EXPECT_EQ(observedTerminals[0].requestId, 101U);
    EXPECT_EQ(terminals[0].status, rt::PhaseRequestStatus::kFinished);
    EXPECT_EQ(facade.availableSlotCount(), 0);
    EXPECT_EQ(facade.pendingRequestCount(), 0U);
    EXPECT_EQ(facade.registeredRequestCount(), 2U);
    ASSERT_EQ(admissions.size(), 2U);
    EXPECT_EQ(admissions[0].status, rt::PhaseAdmissionStatus::kPending);
    EXPECT_EQ(admissions[1].status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(admissions[1].requestId, 303U);
    EXPECT_EQ(admissions[1].kvSlotId, 0);
    ASSERT_TRUE(facade.request(303).has_value());
    EXPECT_EQ(facade.request(303)->scheduling.priority, 2);
    EXPECT_DOUBLE_EQ(facade.request(303)->scheduling.tpotTargetUs, 900.0);

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

    facade.removeTerminalObserver(observerId);
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
    EXPECT_EQ(observedTerminals.size(), 3U);

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseContextServingFacadeTest, AppliesWholeRequestPagedKVAdmissionBackpressure)
{
    constexpr int32_t kSLOT_COUNT = 3;
    constexpr int32_t kPAGE_BUNDLES = 2;
    constexpr int32_t kPROMPT_TOKENS = 64;
    constexpr int32_t kOUTPUT_TOKENS = 64;
    rt::HybridCacheManager cacheManager = makeIndexedPagedManager(kSLOT_COUNT, kPAGE_BUNDLES);
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    rt::TensorMap decodeTensorMap;
    decodeTensorMap.set(binding_names::kKVSlotIds, cacheManager.getKVSlotIds());
    decodeTensorMap.set(binding_names::kKVCacheStartIndex, cacheManager.getKVCacheLengths());
    rt::PhaseBatchState prefillState(kSLOT_COUNT, "phase_page_admission_test");
    std::array<rt::DecodingInferenceContext, kSLOT_COUNT> contexts;
    for (rt::DecodingInferenceContext& context : contexts)
    {
        context.initialize(1, kOUTPUT_TOKENS, std::nullopt, rt::OptionalInputTensors{}, "", decodeStream);
        context.rawBatchedInputIds = {{1}};
        context.tokenIds = context.rawBatchedInputIds;
    }

    std::vector<rt::PhaseAdmissionResult> admissions;
    rt::PhaseContextServingCallbacks callbacks;
    callbacks.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream) {
        prefillState.prepare(batch, cacheManager, stream);
        prefillState.commit(cacheManager, kPROMPT_TOKENS, stream);
    };
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) { return item.promptTokenCount; };
    callbacks.isPrefillFinished = [](uint64_t, rt::DecodingInferenceContext const&, int32_t) { return true; };
    callbacks.enqueueDecode = [](rt::DecodingInferenceContext&) {};
    callbacks.completeDecode = [](rt::DecodingInferenceContext&) {};
    callbacks.onAdmission = [&](rt::PhaseAdmissionResult const& result) { admissions.push_back(result); };

    rt::PhaseQueueSchedulerConfig schedulerConfig;
    schedulerConfig.maxPrefillBatchSize = kSLOT_COUNT;
    schedulerConfig.maxDecodeBatchSize = kSLOT_COUNT;
    schedulerConfig.maxPrefillChunkTokens = kPROMPT_TOKENS;
    rt::PhaseContextServingFacade facade(kSLOT_COUNT, schedulerConfig, std::move(callbacks), cacheManager,
        decodeTensorMap, prefillStream, decodeStream, rt::PhaseTensorRTContextMode::kSharedSerialized, nullptr, 0,
        kSLOT_COUNT);

    EXPECT_EQ(facade.submitOrQueue(101, contexts[0], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(facade.submitOrQueue(102, contexts[1], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    rt::PhaseAdmissionResult const pending = facade.submitOrQueue(103, contexts[2], 0, kPROMPT_TOKENS);
    EXPECT_EQ(pending.status, rt::PhaseAdmissionStatus::kPending);
    EXPECT_EQ(pending.availableSlots, 1);
    EXPECT_EQ(pending.reservedPageBundles, kPAGE_BUNDLES);
    EXPECT_EQ(pending.reservationAvailableBundles, 0);
    EXPECT_EQ(facade.pendingRequestCount(), 1U);

    facade.runUntilIdle(4);
    EXPECT_TRUE(facade.empty());
    EXPECT_EQ(facade.availableSlotCount(), kSLOT_COUNT);
    EXPECT_EQ(cacheManager.getPagedKVPoolStats().allocatedBundles, 0);
    ASSERT_EQ(admissions.size(), 4U);
    EXPECT_EQ(admissions.back().requestId, 103U);
    EXPECT_EQ(admissions.back().status, rt::PhaseAdmissionStatus::kAdmitted);

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseContextServingFacadeTest, HeadroomPolicyAdmitsMoreRequestsWithOneDrainLease)
{
    constexpr int32_t kSLOT_COUNT = 3;
    constexpr int32_t kPAGE_BUNDLES = 4;
    constexpr int32_t kPROMPT_TOKENS = 64;
    constexpr int32_t kOUTPUT_TOKENS = 192;
    rt::HybridCacheManager cacheManager = makeIndexedPagedManager(kSLOT_COUNT, kPAGE_BUNDLES);
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    rt::TensorMap decodeTensorMap;
    decodeTensorMap.set(binding_names::kKVSlotIds, cacheManager.getKVSlotIds());
    decodeTensorMap.set(binding_names::kKVCacheStartIndex, cacheManager.getKVCacheLengths());
    std::array<rt::DecodingInferenceContext, kSLOT_COUNT> contexts;
    for (rt::DecodingInferenceContext& context : contexts)
    {
        context.initialize(1, kOUTPUT_TOKENS, std::nullopt, rt::OptionalInputTensors{}, "", decodeStream);
        context.rawBatchedInputIds = {{1}};
        context.tokenIds = context.rawBatchedInputIds;
    }

    rt::PhaseContextServingCallbacks callbacks;
    callbacks.enqueuePrefill = [](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) {};
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) { return item.promptTokenCount; };
    callbacks.enqueueDecode = [](rt::DecodingInferenceContext&) {};
    callbacks.completeDecode = [](rt::DecodingInferenceContext&) {};

    rt::PhaseQueueSchedulerConfig schedulerConfig;
    schedulerConfig.maxPrefillBatchSize = kSLOT_COUNT;
    schedulerConfig.maxDecodeBatchSize = kSLOT_COUNT;
    schedulerConfig.maxPrefillChunkTokens = kPROMPT_TOKENS;
    rt::PhaseContextServingFacade facade(kSLOT_COUNT, schedulerConfig, std::move(callbacks), cacheManager,
        decodeTensorMap, prefillStream, decodeStream, rt::PhaseTensorRTContextMode::kSharedSerialized, nullptr, 0,
        kSLOT_COUNT);
    facade.configurePageReservation({rt::PhasePageReservationMode::kHeadroom, 0, 0, 1});

    EXPECT_EQ(facade.submitOrQueue(101, contexts[0], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(facade.submitOrQueue(102, contexts[1], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    rt::PhaseAdmissionResult const third = facade.submitOrQueue(103, contexts[2], 0, kPROMPT_TOKENS);
    EXPECT_EQ(third.status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(third.reservedPageBundles, kPAGE_BUNDLES);
    EXPECT_EQ(third.reservationAvailableBundles, 0);
    EXPECT_EQ(facade.pendingRequestCount(), 0U);

    EXPECT_TRUE(facade.cancel(103));
    EXPECT_TRUE(facade.cancel(102));
    EXPECT_TRUE(facade.cancel(101));
    EXPECT_TRUE(facade.empty());
    EXPECT_EQ(cacheManager.getPagedKVPoolStats().allocatedBundles, 0);

    facade.configurePageReservation({rt::PhasePageReservationMode::kBoundedOvercommit, 0, 1, 1});
    EXPECT_EQ(facade.submitOrQueue(201, contexts[0], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(facade.submitOrQueue(202, contexts[1], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(facade.submitOrQueue(203, contexts[2], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_TRUE(facade.cancel(203));
    EXPECT_TRUE(facade.cancel(202));
    EXPECT_TRUE(facade.cancel(201));

    rt::PhasePageReservationConfig promptAware;
    promptAware.mode = rt::PhasePageReservationMode::kHeadroom;
    promptAware.outputHeadroomTokens = 0;
    promptAware.maxConcurrentGrowthRequests = 1;
    promptAware.fullReservationPromptThresholdTokens = kPROMPT_TOKENS;
    facade.configurePageReservation(promptAware);
    EXPECT_EQ(facade.submitOrQueue(251, contexts[0], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(facade.submitOrQueue(252, contexts[1], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(facade.submitOrQueue(253, contexts[2], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kPending);
    EXPECT_TRUE(facade.cancel(253));
    EXPECT_TRUE(facade.cancel(252));
    EXPECT_TRUE(facade.cancel(251));

    facade.configurePageReservation({rt::PhasePageReservationMode::kHeadroom, 0, 0, 2});
    EXPECT_EQ(facade.submitOrQueue(301, contexts[0], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(facade.submitOrQueue(302, contexts[1], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(facade.submitOrQueue(303, contexts[2], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kPending);
    EXPECT_TRUE(facade.cancel(303));
    EXPECT_TRUE(facade.cancel(302));
    EXPECT_TRUE(facade.cancel(301));

    rt::PhasePageReservationConfig adaptive;
    adaptive.mode = rt::PhasePageReservationMode::kHeadroom;
    adaptive.outputHeadroomTokens = 0;
    adaptive.maxConcurrentGrowthRequests = 2;
    adaptive.enableAdaptiveGrowthRequests = true;
    adaptive.minConcurrentGrowthRequests = 1;
    adaptive.growthTpotTargetUs = 0.001;
    adaptive.growthPressureEwmaAlpha = 1.0F;
    adaptive.growthScaleUpThreshold = 0.5F;
    adaptive.growthAdjustmentInterval = 1;
    adaptive.growthAdjustmentStep = 1;
    facade.configurePageReservation(adaptive);
    EXPECT_EQ(facade.submitOrQueue(401, contexts[0], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(facade.submitOrQueue(402, contexts[1], 0, kPROMPT_TOKENS).status, rt::PhaseAdmissionStatus::kAdmitted);
    EXPECT_EQ(facade.pageReservationStats().growthRequestLimit, 1);
    EXPECT_EQ(facade.pageReservationStats().growthRequestOwners, 1);
    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();
    ASSERT_TRUE(facade.dispatchNext());
    facade.wait();
    EXPECT_EQ(facade.pageReservationStats().growthRequestLimit, 2);
    EXPECT_EQ(facade.pageReservationStats().growthRequestOwners, 2);
    EXPECT_TRUE(facade.cancel(402));
    EXPECT_TRUE(facade.cancel(401));

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseContextServingFacadeTest, KeepsStableSlotsWhileDecodeBatchShrinksFourToOne)
{
    constexpr int32_t kSlotCount = 4;
    rt::HybridCacheManager cacheManager = makeIndexedManager(kSlotCount);
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    rt::TensorMap decodeTensorMap;
    decodeTensorMap.set(binding_names::kKVSlotIds, cacheManager.getKVSlotIds());
    decodeTensorMap.set(binding_names::kKVCacheStartIndex, cacheManager.getKVCacheLengths());
    rt::PhaseBatchState prefillState(kSlotCount, "phase_shrinking_prefill_test");

    std::array<rt::DecodingInferenceContext, kSlotCount> contexts;
    for (rt::DecodingInferenceContext& context : contexts)
    {
        context.initialize(1, kSlotCount, std::nullopt, rt::OptionalInputTensors{}, "", decodeStream);
        context.rawBatchedInputIds = {{1, 2}};
        context.tokenIds = context.rawBatchedInputIds;
        context.effectivePrefillLengths = {2};
        context.currentGenerateLengths = {0};
    }

    std::unordered_map<uint64_t, int32_t> const outputBudgets{{101, 1}, {102, 2}, {103, 3}, {104, 4}};
    std::vector<std::vector<int32_t>> decodeSlotBatches;
    std::vector<uint64_t> terminalRequestIds;

    rt::PhaseContextServingCallbacks callbacks;
    callbacks.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream) {
        prefillState.prepare(batch, cacheManager, stream);
        prefillState.commit(cacheManager, 2, stream);
    };
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) { return item.promptTokenCount; };
    callbacks.enqueueDecode
        = [&](rt::DecodingInferenceContext& packed) { packed.phaseBatchState->commit(cacheManager, 1, packed.stream); };
    callbacks.completeDecode = [&](rt::DecodingInferenceContext& packed) {
        decodeSlotBatches.push_back(copyDeviceToHost<int32_t>(packed.phaseBatchState->slotIds()));
        for (int32_t row = 0; row < packed.activeBatchSize; ++row)
        {
            packed.tokenIds[row].push_back(200 + packed.currentGenerateLengths[row]);
            ++packed.currentGenerateLengths[row];
        }
    };
    callbacks.isDecodeFinished = [&](uint64_t requestId, rt::DecodingInferenceContext const& context, int32_t row) {
        return context.currentGenerateLengths[static_cast<size_t>(row)] >= outputBudgets.at(requestId);
    };
    callbacks.onTerminal
        = [&](rt::PhaseRequestSnapshot const& snapshot) { terminalRequestIds.push_back(snapshot.requestId); };

    rt::PhaseQueueSchedulerConfig schedulerConfig;
    schedulerConfig.maxPrefillBatchSize = kSlotCount;
    schedulerConfig.maxDecodeBatchSize = kSlotCount;
    schedulerConfig.maxPrefillChunkTokens = 2;
    int resourceIdentities[6]{};
    auto const safetyContract = rt::PhaseExecutionSafetyContract::independent(
        {&resourceIdentities[0], &resourceIdentities[1], &resourceIdentities[2]},
        {&resourceIdentities[3], &resourceIdentities[4], &resourceIdentities[5]});
    rt::PhaseContextServingFacade facade(kSlotCount, schedulerConfig, std::move(callbacks), cacheManager,
        decodeTensorMap, prefillStream, decodeStream, rt::PhaseTensorRTContextMode::kIndependentConcurrent, nullptr, 0,
        1, safetyContract);

    for (int32_t index = 0; index < kSlotCount; ++index)
    {
        uint64_t const requestId = static_cast<uint64_t>(101 + index);
        EXPECT_EQ(facade.submit(requestId, contexts[static_cast<size_t>(index)], 0, 2), index);
    }
    while (!facade.empty())
    {
        ASSERT_TRUE(facade.dispatchNext());
        facade.wait();
    }

    EXPECT_EQ(decodeSlotBatches, (std::vector<std::vector<int32_t>>{{0, 1, 2, 3}, {1, 2, 3}, {2, 3}, {3}}));
    EXPECT_EQ(terminalRequestIds, (std::vector<uint64_t>{101, 102, 103, 104}));
    EXPECT_EQ(facade.availableSlotCount(), kSlotCount);
    EXPECT_EQ(copyDeviceToHost<int32_t>(cacheManager.getGlobalKVCacheLengths()), (std::vector<int32_t>{3, 4, 5, 6}));

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseContextServingFacadeTest, PrimesServingOwnedPhaseShapesAndReleasesSyntheticKV)
{
    constexpr int32_t kSlotCount = 4;
    rt::HybridCacheManager cacheManager = makeIndexedPagedManager(kSlotCount, 8);
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    rt::TensorMap prefillTensorMap;
    prefillTensorMap.set(binding_names::kKVSlotIds, cacheManager.getKVSlotIds());
    prefillTensorMap.set(binding_names::kKVCacheStartIndex, cacheManager.getKVCacheLengths());
    rt::TensorMap decodeTensorMap = prefillTensorMap;
    int32_t prefillEnqueues{};
    int32_t decodeEnqueues{};
    rt::PhaseContextServingCallbacks callbacks;
    callbacks.enqueuePackedPrefill = [&](rt::PhasePrefillContextBatchAdapter& packed) {
        ++prefillEnqueues;
        packed.phaseBatchState().commit(cacheManager, packed.chunkLength(), packed.stream());
    };
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) { return item.tokenOffset + item.tokenCount; };
    callbacks.enqueuePackedDecode = [&](rt::PhaseContextBatchAdapter& packed) {
        ++decodeEnqueues;
        packed.packedContext().phaseBatchState->commit(cacheManager, 1, packed.packedContext().stream);
    };
    callbacks.completePackedDecode = [](rt::PhaseContextBatchAdapter& packed) {
        rt::DecodingInferenceContext& context = packed.packedContext();
        for (int32_t row{}; row < context.activeBatchSize; ++row)
        {
            context.tokenIds[static_cast<size_t>(row)].push_back(42);
            ++context.currentGenerateLengths[static_cast<size_t>(row)];
        }
    };

    rt::PhaseQueueSchedulerConfig schedulerConfig;
    schedulerConfig.maxPrefillBatchSize = kSlotCount;
    schedulerConfig.maxDecodeBatchSize = kSlotCount;
    schedulerConfig.maxPrefillChunkTokens = 16;
    int resourceIdentities[6]{};
    auto const safetyContract = rt::PhaseExecutionSafetyContract::independent(
        {&resourceIdentities[0], &resourceIdentities[1], &resourceIdentities[2]},
        {&resourceIdentities[3], &resourceIdentities[4], &resourceIdentities[5]});
    rt::PhaseContextServingFacade facade(kSlotCount, schedulerConfig, std::move(callbacks), cacheManager,
        decodeTensorMap, prefillStream, decodeStream, rt::PhaseTensorRTContextMode::kIndependentConcurrent,
        &prefillTensorMap, 16, 0, safetyContract);

    facade.primeCudaGraphShapes({
        {rt::PhaseCudaGraphWarmupKind::kPrefill, 2, 8, 0, 2},
        {rt::PhaseCudaGraphWarmupKind::kPrefill, 2, 8, 8, 2},
        {rt::PhaseCudaGraphWarmupKind::kDecode, 4, 1, 16, 2},
    });

    EXPECT_EQ(prefillEnqueues, 4);
    EXPECT_EQ(decodeEnqueues, 2);
    EXPECT_TRUE(facade.empty());
    EXPECT_EQ(facade.registeredRequestCount(), 0U);
    rt::KVPagePoolStats const pool = cacheManager.getPagedKVPoolStats();
    EXPECT_EQ(pool.allocatedBundles, 0);
    EXPECT_EQ(pool.availableBundles, pool.totalBundles);
    EXPECT_EQ(copyDeviceToHost<int32_t>(cacheManager.getGlobalKVCacheLengths()), (std::vector<int32_t>{0, 0, 0, 0}));

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

} // namespace
