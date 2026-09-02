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

#include "runtime/scheduling/independentPhaseCoordinator.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <algorithm>
#include <utility>

namespace trt_edgellm::rt
{
namespace
{

EngineExecutor::GraphCacheStats addGraphCacheStats(
    EngineExecutor::GraphCacheStats left, EngineExecutor::GraphCacheStats const& right) noexcept
{
    left.executeCalls += right.executeCalls;
    left.hits += right.hits;
    left.misses += right.misses;
    left.captures += right.captures;
    left.evictions += right.evictions;
    left.launchFailures += right.launchFailures;
    left.entries += right.entries;
    return left;
}

} // namespace

IndependentPhaseCoordinator::IndependentPhaseCoordinator(LLMEngineConfig const& config,
    PhaseQueueSchedulerConfig schedulerConfig, IndependentEngineExecutorPair& executors, StableKVPageManager& ownership,
    PipelineIO& prefillIO, PipelineIO& decodeIO, TensorMap& prefillMap, TensorMap& decodeMap,
    cudaStream_t prefillStream, cudaStream_t decodeStream, IndependentPhaseCoordinatorCallbacks callbacks)
    : mConfig(config)
    , mExecutors(executors)
    , mOwnership(ownership)
    , mPrefillIO(prefillIO)
    , mDecodeIO(decodeIO)
    , mPrefillMap(prefillMap)
    , mDecodeMap(decodeMap)
    , mPrefillStream(prefillStream)
    , mDecodeStream(decodeStream)
    , mCallbacks(std::move(callbacks))
    , mPrefillKV(config.maxSupportedPrefillBatchSize, ownership, prefillMap, "independent_coordinator_prefill")
    , mDecodeKV(config.maxSupportedDecodeBatchSize, ownership, decodeMap, "independent_coordinator_decode")
    , mScheduler(std::move(schedulerConfig))
{
    ELLM_CHECK(config.kvPoolPages > 0, "Independent phase coordinator requires a paged-KV engine");
    ELLM_CHECK(mPrefillStream != nullptr && mDecodeStream != nullptr && mPrefillStream != mDecodeStream,
        "Independent phase coordinator requires distinct explicit CUDA streams");
    ELLM_CHECK(static_cast<bool>(mCallbacks.isDecodeFinished),
        "Independent phase coordinator requires a decode termination callback");
    for (Tensor& deepstack : mPrefillIO.deepstackEmbeds)
    {
        CUDA_CHECK(cudaMemsetAsync(deepstack.rawPointer(), 0, deepstack.getMemoryCapacity(), mPrefillStream));
    }
    for (Tensor& deepstack : mDecodeIO.deepstackEmbeds)
    {
        CUDA_CHECK(cudaMemsetAsync(deepstack.rawPointer(), 0, deepstack.getMemoryCapacity(), mDecodeStream));
    }

    bool const sharedContext = mExecutors.sharedExecutionContext();
    PhaseExecutionSafetyContract const safety = sharedContext
        ? PhaseExecutionSafetyContract::shared(mExecutors.prefillExecutor().getExecutionContextIdentity())
        : PhaseExecutionSafetyContract::independent({mExecutors.prefillExecutor().getExecutionContextIdentity(),
                                                        mExecutors.prefillContextMemory().rawPointer(), &mPrefillIO},
              {mExecutors.decodeExecutor().getExecutionContextIdentity(), mExecutors.decodeContextMemory().rawPointer(),
                  &mDecodeIO});
    mWorker = std::make_unique<PhaseDispatchWorker>(mScheduler, makeWorkerCallbacks(), mPrefillStream, mDecodeStream,
        sharedContext ? PhaseTensorRTContextMode::kSharedSerialized : PhaseTensorRTContextMode::kIndependentConcurrent,
        safety);
    mScheduler.setGlobalExecutionVariantSupplier([this](PhaseGlobalActionKey const& key, int32_t primaryTokenCount) {
        int32_t const prefillGraphTokens = mConfig.packedPrefill ? primaryTokenCount : key.chunkLength;
        std::string const shapeSuffix
            = ":" + std::to_string(key.primaryBatchSize) + ":" + std::to_string(prefillGraphTokens);
        bool const prefillGraph
            = mCapturedPrefillShapes.find(std::to_string(mExecutors.config().prefillProfile) + shapeSuffix)
                != mCapturedPrefillShapes.end()
            || (mExecutors.config().visionPrefillProfile >= 0
                && mCapturedPrefillShapes.find(std::to_string(mExecutors.config().visionPrefillProfile) + shapeSuffix)
                    != mCapturedPrefillShapes.end());
        bool const primaryDecodeGraph
            = mCapturedDecodeShapes.find(std::to_string(key.primaryBatchSize)) != mCapturedDecodeShapes.end();
        bool const secondaryDecodeGraph
            = mCapturedDecodeShapes.find(std::to_string(key.secondaryBatchSize)) != mCapturedDecodeShapes.end();
        if (key.kind == PhaseGlobalActionKind::kPrefill)
        {
            return phaseExecutionVariant(prefillGraph, false);
        }
        if (key.kind == PhaseGlobalActionKind::kDecode)
        {
            return phaseExecutionVariant(primaryDecodeGraph, false);
        }
        if (key.kind == PhaseGlobalActionKind::kPrefillDecode)
        {
            return phaseExecutionVariant(prefillGraph, secondaryDecodeGraph);
        }
        return PhaseExecutionVariant::kEager;
    });
}

void IndependentPhaseCoordinator::setCallbacks(IndependentPhaseCoordinatorCallbacks callbacks)
{
    ELLM_CHECK(!busy(), "Independent phase callbacks cannot change while work is in flight");
    ELLM_CHECK(static_cast<bool>(callbacks.isDecodeFinished),
        "Independent phase coordinator requires a decode termination callback");
    mCallbacks = std::move(callbacks);
}

PhaseDispatchWorkerCallbacks IndependentPhaseCoordinator::makeWorkerCallbacks()
{
    PhaseDispatchWorkerCallbacks callbacks;
    callbacks.enqueuePrefill = [this](std::vector<PhaseWorkItem> const& batch, cudaStream_t stream) {
        return enqueuePrefillBatch(batch, stream);
    };
    callbacks.enqueueDecode = [this](std::vector<PhaseWorkItem> const& batch, cudaStream_t stream) {
        return enqueueDecodeBatch(batch, stream);
    };
    callbacks.completePrefillBatch = [this](std::vector<PhaseWorkItem> const& batch) { completePrefillBatch(batch); };
    callbacks.completeDecodeBatch = [this](std::vector<PhaseWorkItem> const& batch) { completeDecodeBatch(batch); };
    callbacks.completePrefill = [this](PhaseWorkItem const& item) {
        int32_t const resultingLength = mOwnership.length(item.kvSlotId);
        bool const finished = mCallbacks.isPrefillFinished && mCallbacks.isPrefillFinished(item, resultingLength);
        return PhasePrefillCompletion{resultingLength, finished};
    };
    callbacks.completeDecode = [this](PhaseWorkItem const& item) {
        int32_t const resultingLength = mOwnership.length(item.kvSlotId);
        return PhaseDecodeCompletion{resultingLength, mCallbacks.isDecodeFinished(item, resultingLength)};
    };
    callbacks.onMetrics = [this](PhaseDispatchMetrics const& metrics) {
        if (mMetricsCollectionEnabled)
        {
            mMetrics.push_back(metrics);
        }
        if (mCallbacks.onMetrics)
        {
            mCallbacks.onMetrics(metrics);
        }
    };
    callbacks.onTimeline = [this](PhaseTimelineEvent const& event) {
        if (mCallbacks.onTimeline)
        {
            mCallbacks.onTimeline(event);
        }
    };
    callbacks.executionVariant = [this](PhaseDispatchKind kind) {
        if (kind == PhaseDispatchKind::kPrefill)
        {
            return phaseExecutionVariant(mLastPrefillGraphReplay, false);
        }
        if (kind == PhaseDispatchKind::kDecode)
        {
            return phaseExecutionVariant(mLastDecodeGraphReplay, false);
        }
        if (kind == PhaseDispatchKind::kOverlap)
        {
            return phaseExecutionVariant(mLastPrefillGraphReplay, mLastDecodeGraphReplay);
        }
        return PhaseExecutionVariant::kEager;
    };
    return callbacks;
}

PhaseHostExecutionTiming IndependentPhaseCoordinator::enqueuePrefillBatch(
    std::vector<PhaseWorkItem> const& batch, cudaStream_t stream)
{
    PhaseHostExecutionTiming timing;
    timing.prepareStartHostNs = phaseTimelineNowNs();
    ELLM_CHECK(!batch.empty(), "Independent prefill enqueue requires a non-empty batch");
    PhasePrefillClass const prefillClass = batch.front().prefillClass;
    ELLM_CHECK(
        prefillClass != PhasePrefillClass::kAny, "Independent prefill work must identify its concrete producer class");
    ELLM_CHECK(std::all_of(batch.begin(), batch.end(),
                   [prefillClass](PhaseWorkItem const& item) { return item.prefillClass == prefillClass; }),
        "Independent prefill batch cannot mix producer classes");
    bool const externalPrefill = prefillClass == PhasePrefillClass::kExternal;
    ELLM_CHECK(!externalPrefill || mExecutors.hasExternalPrefillExecutor(),
        "External prefill requires a dedicated serialized TensorRT execution context");
    int32_t const profileIndex
        = externalPrefill ? mExecutors.externalPrefillProfile() : mExecutors.config().prefillProfile;
    ELLM_CHECK(profileIndex >= 0, "Selected prefill profile is not configured");
    EngineExecutor& executor = externalPrefill ? mExecutors.externalPrefillExecutor() : mExecutors.prefillExecutor();

    std::vector<int32_t> slots;
    std::vector<int32_t> chunks;
    int32_t totalTokens{};
    for (PhaseWorkItem const& item : batch)
    {
        slots.push_back(item.kvSlotId);
        chunks.push_back(item.tokenCount);
        totalTokens += item.tokenCount;
        mOwnership.ensureCapacity(item.kvSlotId, mOwnership.length(item.kvSlotId) + item.tokenCount);
    }
    mPrefillKV.prepare(slots, stream);
    int32_t const chunkLength = chunks.front();
    int32_t const maxRowTokens = *std::max_element(chunks.begin(), chunks.end());
    if (!mConfig.packedPrefill)
    {
        ELLM_CHECK(
            std::all_of(chunks.begin(), chunks.end(), [chunkLength](int32_t value) { return value == chunkLength; }),
            "Independent dense prefill batch requires one uniform chunk length");
    }
    Coords const inputShape = mConfig.packedPrefill
        ? Coords{1, totalTokens, mConfig.hiddenSize}
        : Coords{static_cast<int64_t>(batch.size()), chunkLength, mConfig.hiddenSize};
    ELLM_CHECK(mPrefillIO.inputsEmbeds.reshape(inputShape), "Independent prefill embedding reshape failed");
    if (mCallbacks.stagePrefill)
    {
        mCallbacks.stagePrefill(batch, mPrefillIO, stream);
    }
    else
    {
        CUDA_CHECK(cudaMemsetAsync(
            mPrefillIO.inputsEmbeds.rawPointer(), 0, mPrefillIO.inputsEmbeds.getMemoryCapacity(), stream));
    }
    mPrefillKV.preparePrefillMetadata(mPrefillIO, chunks, stream, mConfig.packedPrefill);
    bool const initialPrefill
        = std::all_of(batch.begin(), batch.end(), [](PhaseWorkItem const& item) { return item.tokenOffset == 0; });
    InferenceDims const dims = mConfig.packedPrefill
        ? (externalPrefill && mConfig.hasVisionPrefillProfile()
                  ? mConfig.visionPackedPrefillDims(static_cast<int64_t>(batch.size()), totalTokens, maxRowTokens)
                  : mConfig.packedPrefillDims(static_cast<int64_t>(batch.size()), totalTokens, maxRowTokens))
        : mConfig.prefillDims(static_cast<int64_t>(batch.size()), chunkLength, initialPrefill);
    ELLM_CHECK(executor.prepare(profileIndex, dims, mPrefillMap, stream), "Independent prefill prepare failed");
    int32_t const graphTokens = mConfig.packedPrefill ? totalTokens : chunkLength;
    std::string const graphShape
        = std::to_string(profileIndex) + ":" + std::to_string(batch.size()) + ":" + std::to_string(graphTokens);
    if (mGraphCaptureEnabled && mCapturedPrefillShapes.find(graphShape) == mCapturedPrefillShapes.end()
        && mCapturedPrefillShapes.size() < mMaxPrefillGraphs
        && ++mPrefillGraphShapeObservations[graphShape] >= mGraphCaptureMinObservations)
    {
        ELLM_CHECK(executor.captureGraph(stream), "Independent packed prefill graph capture failed");
        mCapturedPrefillShapes.insert(graphShape);
        mPrefillGraphShapeObservations.erase(graphShape);
    }
    timing.prepareEndHostNs = phaseTimelineNowNs();
    EngineExecutor::GraphCacheStats const beforeExecute = executor.graphCacheStats();
    timing.executeStartHostNs = phaseTimelineNowNs();
    ELLM_CHECK(executor.execute(stream), "Independent packed prefill execute failed");
    timing.executeEndHostNs = phaseTimelineNowNs();
    EngineExecutor::GraphCacheStats const afterExecute = executor.graphCacheStats();
    mLastPrefillGraphReplay = afterExecute.hits > beforeExecute.hits;
    timing.graphReplay = mLastPrefillGraphReplay;
    return timing;
}

PhaseHostExecutionTiming IndependentPhaseCoordinator::enqueueDecodeBatch(
    std::vector<PhaseWorkItem> const& batch, cudaStream_t stream)
{
    PhaseHostExecutionTiming timing;
    timing.prepareStartHostNs = phaseTimelineNowNs();
    std::vector<int32_t> slots;
    for (PhaseWorkItem const& item : batch)
    {
        slots.push_back(item.kvSlotId);
        mOwnership.ensureCapacity(item.kvSlotId, mOwnership.length(item.kvSlotId) + 1);
    }
    mDecodeKV.prepare(slots, stream);
    ELLM_CHECK(mDecodeIO.inputsEmbeds.reshape({static_cast<int64_t>(batch.size()), 1, mConfig.hiddenSize}),
        "Independent decode embedding reshape failed");
    if (mCallbacks.stageDecode)
    {
        mCallbacks.stageDecode(batch, mDecodeIO, stream);
    }
    else
    {
        CUDA_CHECK(cudaMemsetAsync(
            mDecodeIO.inputsEmbeds.rawPointer(), 0, mDecodeIO.inputsEmbeds.getMemoryCapacity(), stream));
    }
    mDecodeKV.prepareDecodeMetadata(mDecodeIO, stream);
    ELLM_CHECK(mExecutors.decodeExecutor().prepare(mExecutors.config().decodeProfile,
                   mConfig.decodeDims(static_cast<int64_t>(batch.size())), mDecodeMap, stream),
        "Independent decode prepare failed");
    std::string const graphShape = std::to_string(batch.size());
    if (mGraphCaptureEnabled && mCapturedDecodeShapes.find(graphShape) == mCapturedDecodeShapes.end()
        && mCapturedDecodeShapes.size() < mMaxDecodeGraphs
        && ++mDecodeGraphShapeObservations[graphShape] >= mGraphCaptureMinObservations)
    {
        ELLM_CHECK(mExecutors.decodeExecutor().captureGraph(stream), "Independent decode graph capture failed");
        mCapturedDecodeShapes.insert(graphShape);
        mDecodeGraphShapeObservations.erase(graphShape);
    }
    timing.prepareEndHostNs = phaseTimelineNowNs();
    EngineExecutor::GraphCacheStats const beforeExecute = mExecutors.decodeExecutor().graphCacheStats();
    timing.executeStartHostNs = phaseTimelineNowNs();
    ELLM_CHECK(mExecutors.decodeExecutor().execute(stream), "Independent decode execute failed");
    timing.executeEndHostNs = phaseTimelineNowNs();
    EngineExecutor::GraphCacheStats const afterExecute = mExecutors.decodeExecutor().graphCacheStats();
    mLastDecodeGraphReplay = afterExecute.hits > beforeExecute.hits;
    timing.graphReplay = mLastDecodeGraphReplay;
    return timing;
}

void IndependentPhaseCoordinator::completePrefillBatch(std::vector<PhaseWorkItem> const& batch)
{
    std::vector<int32_t> resultingLengths;
    for (PhaseWorkItem const& item : batch)
    {
        resultingLengths.push_back(mOwnership.length(item.kvSlotId) + item.tokenCount);
    }
    mPrefillKV.commitLengths(resultingLengths);
    mPrefillKV.complete();
    if (mCallbacks.completePrefillBatch)
    {
        mCallbacks.completePrefillBatch(batch, mPrefillIO, mPrefillStream);
    }
}

void IndependentPhaseCoordinator::completeDecodeBatch(std::vector<PhaseWorkItem> const& batch)
{
    std::vector<int32_t> resultingLengths;
    for (PhaseWorkItem const& item : batch)
    {
        resultingLengths.push_back(mOwnership.length(item.kvSlotId) + 1);
    }
    mDecodeKV.commitLengths(resultingLengths);
    mDecodeKV.complete();
    if (mCallbacks.completeDecodeBatch)
    {
        mCallbacks.completeDecodeBatch(batch, mDecodeIO, mDecodeStream);
    }
}

void IndependentPhaseCoordinator::enqueuePrefill(PhaseWorkItem item)
{
    mScheduler.enqueuePrefill(std::move(item));
}

void IndependentPhaseCoordinator::enqueueDecode(PhaseWorkItem item)
{
    mScheduler.enqueueDecode(std::move(item));
}

bool IndependentPhaseCoordinator::dispatchNext()
{
    return mWorker->dispatchNext();
}

bool IndependentPhaseCoordinator::augmentGlobalAction(PhaseGlobalActionCandidate missingPhase,
    PhaseGlobalActionCandidate aggregate, uint64_t planId, uint64_t snapshotEpoch)
{
    return mWorker->augmentNext(std::move(missingPhase), std::move(aggregate), planId, snapshotEpoch);
}

bool IndependentPhaseCoordinator::poll()
{
    return mWorker->poll();
}

void IndependentPhaseCoordinator::wait()
{
    mWorker->wait();
}

void IndependentPhaseCoordinator::runUntilIdle(size_t maxDispatches)
{
    mWorker->runUntilIdle(maxDispatches);
}

bool IndependentPhaseCoordinator::capturePreparedGraphs()
{
    ELLM_CHECK(!busy(), "Independent phase graphs cannot be captured while work is in flight");
    bool const prefillCaptured = mExecutors.prefillExecutor().captureGraph(mPrefillStream);
    bool const decodeCaptured = mExecutors.decodeExecutor().captureGraph(mDecodeStream);
    return prefillCaptured && decodeCaptured;
}

void IndependentPhaseCoordinator::setGraphCaptureEnabled(bool enabled) noexcept
{
    mGraphCaptureEnabled = enabled;
}

void IndependentPhaseCoordinator::setGraphCaptureMinObservations(size_t observations)
{
    ELLM_CHECK(observations > 0, "Graph capture promotion requires a positive observation count");
    mGraphCaptureMinObservations = observations;
    mPrefillGraphShapeObservations.clear();
    mDecodeGraphShapeObservations.clear();
}

void IndependentPhaseCoordinator::setGraphCaptureLimits(size_t maxPrefillGraphs, size_t maxDecodeGraphs) noexcept
{
    mMaxPrefillGraphs = maxPrefillGraphs;
    mMaxDecodeGraphs = maxDecodeGraphs;
    static_cast<void>(mExecutors.prefillExecutor().trimGraphCache(maxPrefillGraphs));
    if (mExecutors.hasExternalPrefillExecutor())
    {
        static_cast<void>(mExecutors.externalPrefillExecutor().trimGraphCache(maxPrefillGraphs));
    }
    static_cast<void>(mExecutors.decodeExecutor().trimGraphCache(maxDecodeGraphs));
}

void IndependentPhaseCoordinator::setPersistentDecodeSelectEnabled(bool enabled) noexcept
{
    mDecodeKV.setPersistentDecodeSelectEnabled(enabled);
}

void IndependentPhaseCoordinator::setPersistentPageBindingsEnabled(bool enabled) noexcept
{
    mPrefillKV.setPersistentPageBindingsEnabled(enabled);
    mDecodeKV.setPersistentPageBindingsEnabled(enabled);
}

EngineExecutor::GraphCacheStats IndependentPhaseCoordinator::prefillGraphCacheStats() const noexcept
{
    EngineExecutor::GraphCacheStats result = mExecutors.prefillExecutor().graphCacheStats();
    if (mExecutors.hasExternalPrefillExecutor())
    {
        result = addGraphCacheStats(result, mExecutors.externalPrefillExecutor().graphCacheStats());
    }
    return result;
}

EngineExecutor::GraphCacheStats IndependentPhaseCoordinator::decodeGraphCacheStats() const noexcept
{
    return mExecutors.decodeExecutor().graphCacheStats();
}

bool IndependentPhaseCoordinator::empty() const noexcept
{
    return mScheduler.empty() && !mWorker->busy();
}

bool IndependentPhaseCoordinator::busy() const noexcept
{
    return mWorker->busy();
}

PhaseDispatchKind IndependentPhaseCoordinator::inFlightKind() const noexcept
{
    return mWorker->inFlightKind();
}

PhasePrefillClass IndependentPhaseCoordinator::inFlightPrefillClass() const noexcept
{
    return mWorker->inFlightPrefillClass();
}

PhaseGlobalActionCandidate const* IndependentPhaseCoordinator::inFlightGlobalCandidate() const noexcept
{
    return mWorker->inFlightGlobalCandidate();
}

PhaseInFlightSnapshot IndependentPhaseCoordinator::inFlightSnapshot(uint64_t hostSnapshotNs) const noexcept
{
    return mWorker->inFlightSnapshot(hostSnapshotNs);
}

void IndependentPhaseCoordinator::setDirectionalInjectionControl(PhaseDirectionalInjectionControl control)
{
    ELLM_CHECK(!busy(), "Phase directional injection control cannot change while work is in flight");
    mWorker->setDirectionalInjectionControl(control);
}

TensorMap& IndependentPhaseCoordinator::prefillTensorMap() noexcept
{
    return mPrefillMap;
}

TensorMap& IndependentPhaseCoordinator::decodeTensorMap() noexcept
{
    return mDecodeMap;
}

PhaseQueueScheduler& IndependentPhaseCoordinator::scheduler() noexcept
{
    return mScheduler;
}

PhaseQueueScheduler const& IndependentPhaseCoordinator::scheduler() const noexcept
{
    return mScheduler;
}

std::vector<PhaseDispatchMetrics> const& IndependentPhaseCoordinator::metrics() const noexcept
{
    return mMetrics;
}

void IndependentPhaseCoordinator::setMetricsCollectionEnabled(bool enabled) noexcept
{
    mMetricsCollectionEnabled = enabled;
    if (!enabled)
    {
        mMetrics.clear();
    }
}

void IndependentPhaseCoordinator::setActivityTimeline(PhaseActivityTimelineRecorder* timeline)
{
    mWorker->setActivityTimeline(timeline);
}

CUcontext IndependentPhaseCoordinator::cudaContext() const noexcept
{
    return mWorker->cudaContext();
}

cudaStream_t IndependentPhaseCoordinator::phaseStream(PhaseUnifiedPhase phase) const noexcept
{
    return mWorker->phaseStream(phase);
}

cudaEvent_t IndependentPhaseCoordinator::phaseStartEvent(PhaseUnifiedPhase phase) const noexcept
{
    return mWorker->phaseStartEvent(phase);
}

void IndependentPhaseCoordinator::setNextDispatchPreamble(
    PhaseUnifiedPhase phase, std::function<void(cudaStream_t)> preamble)
{
    mWorker->setNextDispatchPreamble(phase, std::move(preamble));
}

PhaseKVMemoryStats const& IndependentPhaseCoordinator::prefillKVMemoryStats() const noexcept
{
    return mPrefillKV.memoryStats();
}

PhaseKVMemoryStats const& IndependentPhaseCoordinator::decodeKVMemoryStats() const noexcept
{
    return mDecodeKV.memoryStats();
}

KVPageTableUploadStats const& IndependentPhaseCoordinator::prefillPageTableUploadStats() const noexcept
{
    return mPrefillKV.pageTableUploadStats();
}

KVPageTableUploadStats const& IndependentPhaseCoordinator::decodePageTableUploadStats() const noexcept
{
    return mDecodeKV.pageTableUploadStats();
}

} // namespace trt_edgellm::rt
