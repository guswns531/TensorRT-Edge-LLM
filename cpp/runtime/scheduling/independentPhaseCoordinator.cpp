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

#include <utility>

namespace trt_edgellm::rt
{

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
    , mPrefillKV(config.maxSupportedBatchSize, ownership, prefillMap, "independent_coordinator_prefill")
    , mDecodeKV(config.maxSupportedBatchSize, ownership, decodeMap, "independent_coordinator_decode")
    , mScheduler(std::move(schedulerConfig))
{
    ELLM_CHECK(config.packedPrefill, "Independent phase coordinator currently requires a packed-prefill engine");
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

    PhaseExecutionSafetyContract const safety
        = PhaseExecutionSafetyContract::independent({mExecutors.prefillExecutor().getExecutionContextIdentity(),
                                                        mExecutors.prefillContextMemory().rawPointer(), &mPrefillIO},
            {mExecutors.decodeExecutor().getExecutionContextIdentity(), mExecutors.decodeContextMemory().rawPointer(),
                &mDecodeIO});
    mWorker = std::make_unique<PhaseDispatchWorker>(mScheduler, makeWorkerCallbacks(), mPrefillStream, mDecodeStream,
        PhaseTensorRTContextMode::kIndependentConcurrent, safety);
}

PhaseDispatchWorkerCallbacks IndependentPhaseCoordinator::makeWorkerCallbacks()
{
    PhaseDispatchWorkerCallbacks callbacks;
    callbacks.enqueuePrefill
        = [this](std::vector<PhaseWorkItem> const& batch, cudaStream_t stream) { enqueuePrefillBatch(batch, stream); };
    callbacks.enqueueDecode
        = [this](std::vector<PhaseWorkItem> const& batch, cudaStream_t stream) { enqueueDecodeBatch(batch, stream); };
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
        mMetrics.push_back(metrics);
        if (mCallbacks.onMetrics)
        {
            mCallbacks.onMetrics(metrics);
        }
    };
    return callbacks;
}

void IndependentPhaseCoordinator::enqueuePrefillBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream)
{
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
    ELLM_CHECK(mPrefillIO.inputsEmbeds.reshape({1, totalTokens, mConfig.hiddenSize}),
        "Independent packed prefill embedding reshape failed");
    if (mCallbacks.stagePrefill)
    {
        mCallbacks.stagePrefill(batch, mPrefillIO, stream);
    }
    else
    {
        CUDA_CHECK(cudaMemsetAsync(
            mPrefillIO.inputsEmbeds.rawPointer(), 0, mPrefillIO.inputsEmbeds.getMemoryCapacity(), stream));
    }
    mPrefillKV.preparePrefillMetadata(mPrefillIO, chunks, stream, true);
    ELLM_CHECK(mExecutors.prefillExecutor().prepare(mExecutors.config().prefillProfile,
                   mConfig.packedPrefillDims(static_cast<int64_t>(batch.size()), totalTokens), mPrefillMap, stream),
        "Independent packed prefill prepare failed");
    ELLM_CHECK(mExecutors.prefillExecutor().execute(stream), "Independent packed prefill execute failed");
}

void IndependentPhaseCoordinator::enqueueDecodeBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream)
{
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
    ELLM_CHECK(mExecutors.decodeExecutor().execute(stream), "Independent decode execute failed");
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

bool IndependentPhaseCoordinator::empty() const noexcept
{
    return mScheduler.empty() && !mWorker->busy();
}

bool IndependentPhaseCoordinator::busy() const noexcept
{
    return mWorker->busy();
}

PhaseQueueScheduler& IndependentPhaseCoordinator::scheduler() noexcept
{
    return mScheduler;
}

std::vector<PhaseDispatchMetrics> const& IndependentPhaseCoordinator::metrics() const noexcept
{
    return mMetrics;
}

CUcontext IndependentPhaseCoordinator::cudaContext() const noexcept
{
    return mWorker->cudaContext();
}

} // namespace trt_edgellm::rt
