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

#include "common/checkMacros.h"
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/scheduling/phaseKernelGroupRecorder.h"

#include <gtest/gtest.h>
#include <unordered_map>
#include <utility>

using namespace trt_edgellm;

TEST(PhaseDispatchWorkerTest, PreservesSelectedDecodeRowsWithoutChangingRequestSet)
{
    std::vector<rt::PhaseWorkItem> batch{{13, 1}, {11, 1}, {14, 1}};
    rt::preservePhaseBatchRowAffinity(batch, {10, 11, 12, 13});

    ASSERT_EQ(batch.size(), 3U);
    EXPECT_EQ(batch[0].requestId, 13U);
    EXPECT_EQ(batch[1].requestId, 11U);
    EXPECT_EQ(batch[2].requestId, 14U);
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
    std::unordered_map<uint64_t, int32_t> decodeSteps;
    rt::PhaseDispatchWorkerCallbacks callbacks;
    callbacks.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) { ++prefillEnqueues; };
    callbacks.enqueueDecode = [&](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) { ++decodeEnqueues; };
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) {
        return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, false};
    };
    callbacks.completeDecode = [&](rt::PhaseWorkItem const& item) {
        int32_t const step = ++decodeSteps[item.requestId];
        return rt::PhaseDecodeCompletion{item.tokenCount + 1, item.requestId == 1 || step == 3};
    };

    rt::PhaseDispatchWorker worker(scheduler, std::move(callbacks), prefillStream, decodeStream);
    CUcontext expectedCudaContext{};
    CUDA_DRIVER_CHECK(cuStreamGetCtx(prefillStream, &expectedCudaContext));
    EXPECT_EQ(worker.cudaContext(), expectedCudaContext);
    worker.runUntilIdle(8);

    EXPECT_EQ(prefillEnqueues, 3);
    EXPECT_EQ(decodeEnqueues, 4);
    EXPECT_EQ(worker.dispatchCount(), 4U);
    EXPECT_FALSE(worker.busy());
    EXPECT_TRUE(scheduler.empty());

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseExecutionSafetyContractTest, RequiresIndependentResourcesForConcurrentMode)
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
}

TEST(PhaseDispatchWorkerTest, ConcurrentModeUsesTwoStreamsInOneCudaContext)
{
    rt::PhaseQueueScheduler scheduler;
    scheduler.enqueuePrefill({1, 32});
    scheduler.enqueueDecode({2, 64});

    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    int32_t prefillEnqueues{};
    int32_t decodeEnqueues{};
    std::vector<rt::PhaseTimelineEvent> timeline;
    std::optional<rt::PhaseDispatchMetrics> dispatchMetrics;
    rt::PhaseDispatchWorkerCallbacks callbacks;
    callbacks.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) { ++prefillEnqueues; };
    callbacks.enqueueDecode = [&](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) { ++decodeEnqueues; };
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) {
        return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, true};
    };
    callbacks.completeDecode
        = [](rt::PhaseWorkItem const& item) { return rt::PhaseDecodeCompletion{item.tokenCount + 1, true}; };
    callbacks.onTimeline = [&](rt::PhaseTimelineEvent const& event) { timeline.push_back(event); };
    callbacks.onMetrics = [&](rt::PhaseDispatchMetrics const& metrics) { dispatchMetrics = metrics; };

    int identities[6]{};
    auto const contract = rt::PhaseExecutionSafetyContract::independent(
        {&identities[0], &identities[1], &identities[2]}, {&identities[3], &identities[4], &identities[5]});
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
    ASSERT_EQ(timeline.size(), 4U);
    EXPECT_EQ(timeline[0].stage, rt::PhaseTimelineStage::kPrefillStart);
    EXPECT_EQ(timeline[0].requestId, 1U);
    EXPECT_EQ(timeline[1].stage, rt::PhaseTimelineStage::kDecodeStart);
    EXPECT_EQ(timeline[1].requestId, 2U);
    EXPECT_EQ(timeline[2].stage, rt::PhaseTimelineStage::kPrefillDone);
    EXPECT_EQ(timeline[3].stage, rt::PhaseTimelineStage::kDecodeDone);
    EXPECT_TRUE(std::all_of(timeline.begin(), timeline.end(),
        [](rt::PhaseTimelineEvent const& event) { return event.dispatchIndex == 1U && event.timestampNs > 0U; }));
    ASSERT_TRUE(dispatchMetrics.has_value());
    EXPECT_EQ(dispatchMetrics->prefillRequestIds, std::vector<uint64_t>{1U});
    EXPECT_EQ(dispatchMetrics->decodeRequestIds, std::vector<uint64_t>{2U});
    EXPECT_GT(dispatchMetrics->hostCompletionNs, dispatchMetrics->hostDispatchStartNs);

    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseKernelGroupRecorderTest, SegmentsAndMeasuresCrossStreamHandoffs)
{
    cudaStream_t firstStream{};
    cudaStream_t secondStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&firstStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&secondStream, cudaStreamNonBlocking));
    int32_t* marker{};
    CUDA_CHECK(cudaMalloc(&marker, 4 * sizeof(int32_t)));

    rt::PhaseKernelGroupRecorder recorder;
    recorder.execute(7,
        {{rt::PhaseKernelGroup::kEncoderEngine, "vit", firstStream,
             [&](cudaStream_t stream) { CUDA_CHECK(cudaMemsetAsync(marker, 1, 4 * sizeof(int32_t), stream)); }},
            {rt::PhaseKernelGroup::kPrefillEngine, "llm_prefill", secondStream,
                [&](cudaStream_t stream) { CUDA_CHECK(cudaMemsetAsync(marker, 2, 4 * sizeof(int32_t), stream)); }},
            {rt::PhaseKernelGroup::kPrefillSample, "sample", secondStream,
                [&](cudaStream_t stream) { CUDA_CHECK(cudaMemsetAsync(marker, 3, 4 * sizeof(int32_t), stream)); }}});
    EXPECT_EQ(recorder.pendingCount(), 3U);
    recorder.drain();
    ASSERT_EQ(recorder.samples().size(), 3U);
    EXPECT_EQ(recorder.samples()[0].dispatchIndex, 7U);
    EXPECT_EQ(recorder.samples()[0].name, "vit");
    EXPECT_EQ(recorder.samples()[1].group, rt::PhaseKernelGroup::kPrefillEngine);
    EXPECT_GE(recorder.samples()[2].gpuMs, 0.0F);

    CUDA_CHECK(cudaFree(marker));
    CUDA_CHECK(cudaStreamDestroy(firstStream));
    CUDA_CHECK(cudaStreamDestroy(secondStream));
}
