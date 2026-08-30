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
#include "runtime/scheduling/phaseActivityTimeline.h"
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/scheduling/phaseKernelGroupRecorder.h"

#include <gtest/gtest.h>
#include <unordered_map>
#include <utility>

using namespace trt_edgellm;

TEST(PhaseActivityTimelineTest, SweepsAllFourActivityBitsAndIdleGaps)
{
    std::vector<rt::PhaseActivityInterval> const intervals{
        {1, 0, rt::PhaseActivityKind::kDecode, "decode-0", 0.5F, 2.0F},
        {2, 0, rt::PhaseActivityKind::kEncoder, "encoder", 1.0F, 6.0F},
        {3, 0, rt::PhaseActivityKind::kPrefill, "prefill", 2.0F, 7.0F},
        {4, 0, rt::PhaseActivityKind::kDecode, "decode-1", 3.0F, 5.0F},
        {5, 0, rt::PhaseActivityKind::kCopy, "copy", 2.5F, 4.5F},
    };
    std::vector<rt::PhaseActivitySegment> const segments = rt::phaseActivitySegments(intervals, 8.0F);
    ASSERT_EQ(segments.size(), 10U);
    EXPECT_EQ(segments[0].mask, 0U);
    EXPECT_EQ(segments[1].mask, 0x04U);
    EXPECT_EQ(segments[2].mask, 0x05U);
    EXPECT_EQ(segments[3].mask, 0x03U);
    EXPECT_EQ(segments[4].mask, 0x0BU);
    EXPECT_EQ(segments[5].mask, 0x0FU);
    EXPECT_EQ(segments[6].mask, 0x07U);
    EXPECT_EQ(segments[7].mask, 0x03U);
    EXPECT_EQ(segments[8].mask, 0x02U);
    EXPECT_EQ(segments[9].mask, 0U);
    EXPECT_EQ(rt::phaseActivityMaskString(0x01U), "0001");
    EXPECT_EQ(rt::phaseActivityMaskString(0x0FU), "1111");

    rt::PhaseActivitySummary const summary = rt::phaseActivitySummary(segments);
    EXPECT_DOUBLE_EQ(summary.windowMs, 8.0);
    EXPECT_DOUBLE_EQ(summary.allIdleMs, 1.5);
    EXPECT_DOUBLE_EQ(summary.epdIdleMs, 1.5);
    EXPECT_DOUBLE_EQ(summary.anyActivityMs, 6.5);
    EXPECT_DOUBLE_EQ(summary.anyEpdMs, 6.5);
    EXPECT_DOUBLE_EQ(summary.epdTripleMs, 2.0);
    EXPECT_DOUBLE_EQ(summary.fourWayMs, 1.5);
    EXPECT_DOUBLE_EQ(summary.activityMs[0], 5.0);
    EXPECT_DOUBLE_EQ(summary.activityMs[1], 5.0);
    EXPECT_DOUBLE_EQ(summary.activityMs[2], 3.5);
    EXPECT_DOUBLE_EQ(summary.activityMs[3], 2.0);
}

TEST(PhaseActivityTimelineTest, RecordsEpochRelativeIntervalsAcrossStreams)
{
    cudaStream_t epochStream{};
    cudaStream_t encoderStream{};
    cudaStream_t copyStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&epochStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&encoderStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
    int32_t* marker{};
    CUDA_CHECK(cudaMalloc(&marker, 4 * sizeof(int32_t)));

    rt::PhaseActivityTimelineRecorder recorder(epochStream);
    auto const encoder = recorder.begin(rt::PhaseActivityKind::kEncoder, encoderStream, "vit", 41U);
    CUDA_CHECK(cudaMemsetAsync(marker, 1, 2 * sizeof(int32_t), encoderStream));
    recorder.end(encoder, encoderStream);
    auto const copy = recorder.begin(rt::PhaseActivityKind::kCopy, copyStream, "payload", 41U);
    CUDA_CHECK(cudaMemsetAsync(marker + 2, 2, 2 * sizeof(int32_t), copyStream));
    recorder.end(copy, copyStream);
    EXPECT_EQ(recorder.pendingCount(), 2U);
    recorder.drain();

    std::vector<rt::PhaseActivityInterval> const intervals = recorder.intervals();
    ASSERT_EQ(intervals.size(), 2U);
    EXPECT_EQ(intervals[0].correlationId, 41U);
    EXPECT_GE(intervals[0].startMs, 0.0F);
    EXPECT_GE(intervals[0].endMs, intervals[0].startMs);
    EXPECT_GE(intervals[1].startMs, 0.0F);
    EXPECT_GE(intervals[1].endMs, intervals[1].startMs);

    CUDA_CHECK(cudaFree(marker));
    CUDA_CHECK(cudaStreamDestroy(epochStream));
    CUDA_CHECK(cudaStreamDestroy(encoderStream));
    CUDA_CHECK(cudaStreamDestroy(copyStream));
}

TEST(PhaseActivityTimelineTest, CountsCopyOnlyTimeAsEpdIdle)
{
    std::vector<rt::PhaseActivityInterval> const intervals{
        {1, 0, rt::PhaseActivityKind::kCopy, "copy", 0.5F, 1.5F},
    };
    rt::PhaseActivitySummary const summary = rt::phaseActivitySummary(rt::phaseActivitySegments(intervals, 2.0F));

    EXPECT_DOUBLE_EQ(summary.windowMs, 2.0);
    EXPECT_DOUBLE_EQ(summary.allIdleMs, 1.0);
    EXPECT_DOUBLE_EQ(summary.epdIdleMs, 2.0);
    EXPECT_DOUBLE_EQ(summary.anyActivityMs, 1.0);
    EXPECT_DOUBLE_EQ(summary.anyEpdMs, 0.0);
    EXPECT_DOUBLE_EQ(summary.activityMs[3], 1.0);
}

TEST(PhaseActivityTimelineTest, ActiveSpanExcludesStartupAndTrailingIdleButPreservesInternalIdle)
{
    std::vector<rt::PhaseActivityInterval> const intervals{
        {1, 0, rt::PhaseActivityKind::kEncoder, "encoder", 10.0F, 11.0F},
        {2, 0, rt::PhaseActivityKind::kDecode, "decode", 12.0F, 13.0F},
    };
    std::vector<rt::PhaseActivitySegment> const segments = rt::phaseActivityActiveSpanSegments(intervals);

    ASSERT_EQ(segments.size(), 3U);
    EXPECT_EQ(segments[0].mask, 0x01U);
    EXPECT_EQ(segments[1].mask, 0x00U);
    EXPECT_EQ(segments[2].mask, 0x04U);
    EXPECT_FLOAT_EQ(segments[0].startMs, 0.0F);
    EXPECT_FLOAT_EQ(segments[2].endMs, 3.0F);
    rt::PhaseActivitySummary const summary = rt::phaseActivitySummary(segments);
    EXPECT_DOUBLE_EQ(summary.windowMs, 3.0);
    EXPECT_DOUBLE_EQ(summary.allIdleMs, 1.0);
}

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

TEST(PhaseDispatchWorkerTest, AugmentsLivePrefillWithResidualDecode)
{
    rt::PhaseQueueSchedulerConfig config;
    config.globalSchedulerMode = rt::PhaseGlobalSchedulerMode::kActive;
    rt::PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 32, 1, 0, 32});

    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));
    rt::PhaseDispatchWorkerCallbacks callbacks;
    callbacks.enqueuePrefill = [](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) {};
    callbacks.enqueueDecode = [](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) {};
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) {
        return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, true};
    };
    callbacks.completeDecode
        = [](rt::PhaseWorkItem const& item) { return rt::PhaseDecodeCompletion{item.tokenCount + 1, true}; };
    int identities[6]{};
    auto const contract = rt::PhaseExecutionSafetyContract::independent(
        {&identities[0], &identities[1], &identities[2]}, {&identities[3], &identities[4], &identities[5]});
    rt::PhaseDispatchWorker worker(scheduler, std::move(callbacks), prefillStream, decodeStream,
        rt::PhaseTensorRTContextMode::kIndependentConcurrent, contract);

    ASSERT_TRUE(worker.dispatchNext());
    scheduler.enqueueDecode({2, 64, 2});
    std::optional<rt::PhaseGlobalActionCandidate> missing = scheduler.previewGlobalDecodeAction();
    ASSERT_TRUE(missing.has_value());
    rt::PhaseGlobalActionCandidate aggregate;
    aggregate.key = {rt::PhaseGlobalActionKind::kPrefillDecode, 1, missing->key.primaryBatchSize, 32, 0,
        missing->key.primaryContextBucket};
    aggregate.key.primaryWorkClass = static_cast<int32_t>(rt::PhasePrefillClass::kText);
    aggregate.key.residualAugmentation = true;
    aggregate.primaryRequestIds = {1U};
    aggregate.primaryStableSlotIds = {1};
    aggregate.secondaryRequestIds = missing->primaryRequestIds;
    aggregate.secondaryStableSlotIds = missing->primaryStableSlotIds;
    aggregate.referenceWorkUs = 2.0;
    aggregate.predictedMakespanUs = 1.0;
    aggregate.overlapCostKnown = true;
    rt::phaseGlobalFinalizeCandidate(aggregate);

    EXPECT_TRUE(worker.augmentNext(std::move(*missing), aggregate, 100U, 100U));
    EXPECT_EQ(worker.inFlightKind(), rt::PhaseDispatchKind::kOverlap);
    worker.wait();

    ASSERT_TRUE(worker.lastMetrics().has_value());
    EXPECT_EQ(worker.lastMetrics()->kind, rt::PhaseDispatchKind::kOverlap);
    EXPECT_EQ(worker.lastMetrics()->prefillRequestIds, std::vector<uint64_t>{1U});
    EXPECT_EQ(worker.lastMetrics()->decodeRequestIds, std::vector<uint64_t>{2U});
    EXPECT_TRUE(worker.lastMetrics()->globalSelectedAction.residualAugmentation);
    EXPECT_TRUE(scheduler.empty());
    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
}

TEST(PhaseDispatchWorkerTest, RejectsResidualAugmentationWithSharedContext)
{
    rt::PhaseQueueSchedulerConfig config;
    config.globalSchedulerMode = rt::PhaseGlobalSchedulerMode::kActive;
    rt::PhaseQueueScheduler scheduler(config);
    scheduler.enqueuePrefill({1, 32, 1, 0, 32});

    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));
    rt::PhaseDispatchWorkerCallbacks callbacks;
    callbacks.enqueuePrefill = [](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) {};
    callbacks.enqueueDecode = [](std::vector<rt::PhaseWorkItem> const&, cudaStream_t) {};
    callbacks.completePrefill = [](rt::PhaseWorkItem const& item) {
        return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, true};
    };
    callbacks.completeDecode
        = [](rt::PhaseWorkItem const& item) { return rt::PhaseDecodeCompletion{item.tokenCount + 1, true}; };
    rt::PhaseDispatchWorker worker(scheduler, std::move(callbacks), prefillStream, decodeStream);

    ASSERT_TRUE(worker.dispatchNext());
    scheduler.enqueueDecode({2, 64, 2});
    std::optional<rt::PhaseGlobalActionCandidate> missing = scheduler.previewGlobalDecodeAction();
    ASSERT_TRUE(missing.has_value());
    rt::PhaseGlobalActionCandidate aggregate;
    aggregate.key.kind = rt::PhaseGlobalActionKind::kPrefillDecode;
    aggregate.primaryRequestIds = {1U};
    aggregate.secondaryRequestIds = missing->primaryRequestIds;

    EXPECT_FALSE(worker.augmentNext(std::move(*missing), aggregate, 100U, 100U));
    EXPECT_EQ(worker.inFlightKind(), rt::PhaseDispatchKind::kPrefill);
    worker.wait();

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
