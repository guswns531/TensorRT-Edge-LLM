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

#include "runtime/scheduling/phaseVisionAdapter.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <utility>

namespace trt_edgellm::rt
{

struct PhaseVisionBatchStorage
{
    Tensor outputEmbedding;
    std::vector<Tensor> deepstackFeatures;
    Tensor mropeCosSin;
    size_t lastUseGeneration{};
};

struct PhaseVisionMropeStorage
{
    Tensor mropeCosSin;
};

namespace
{
void resizeTensor(Tensor& destination, Tensor const& source, std::string const& name)
{
    int64_t const requiredBytes
        = source.getShape().volume() * static_cast<int64_t>(utils::getTypeSize(source.getDataType()));
    if (destination.isEmpty() || destination.getDataType() != source.getDataType()
        || destination.getMemoryCapacity() < requiredBytes)
    {
        destination = Tensor(source.getShape(), DeviceType::kGPU, source.getDataType(), name);
        return;
    }
    ELLM_CHECK(destination.reshape(source.getShape()), "Failed to reshape retained vision tensor");
}

size_t storageByteSize(PhaseVisionBatchStorage const& storage) noexcept
{
    auto capacity = [](Tensor const& tensor) {
        return tensor.isEmpty() ? size_t{} : static_cast<size_t>(tensor.getMemoryCapacity());
    };
    size_t result = capacity(storage.outputEmbedding);
    for (Tensor const& feature : storage.deepstackFeatures)
    {
        result += capacity(feature);
    }
    result += capacity(storage.mropeCosSin);
    return result;
}

PhaseVisionDebugTensor captureDebugTensor(Tensor const& tensor, cudaStream_t stream)
{
    PhaseVisionDebugTensor result;
    if (tensor.isEmpty())
    {
        return result;
    }
    Coords const shape = tensor.getShape();
    result.shape.reserve(static_cast<size_t>(shape.getNumDims()));
    for (int32_t dim{}; dim < shape.getNumDims(); ++dim)
    {
        result.shape.push_back(shape[dim]);
    }
    result.dataType = tensor.getDataType();
    size_t const bytes = static_cast<size_t>(shape.volume()) * utils::getTypeSize(result.dataType);
    result.bytes.resize(bytes);
    CUDA_CHECK(cudaMemcpyAsync(result.bytes.data(), tensor.rawPointer(), bytes, cudaMemcpyDeviceToHost, stream));
    return result;
}
} // namespace

PhaseVisionPayload::~PhaseVisionPayload() noexcept
{
    if (startEvent != nullptr)
    {
        static_cast<void>(cudaEventDestroy(startEvent));
    }
    if (readyEvent != nullptr)
    {
        static_cast<void>(cudaEventDestroy(readyEvent));
    }
}

size_t PhaseVisionPayload::byteSize() const noexcept
{
    auto tensorBytes = [](Tensor const& tensor) {
        return tensor.isEmpty()
            ? size_t{}
            : static_cast<size_t>(tensor.getShape().volume()) * utils::getTypeSize(tensor.getDataType());
    };
    size_t result = tensorBytes(outputEmbedding) + tensorBytes(mropeCosSin);
    for (Tensor const& feature : deepstackFeatures)
    {
        result += tensorBytes(feature);
    }
    return result;
}

size_t PhaseVisionPayload::prefillByteSize() const noexcept
{
    auto tensorBytes = [](Tensor const& tensor) {
        return tensor.isEmpty()
            ? size_t{}
            : static_cast<size_t>(tensor.getShape().volume()) * utils::getTypeSize(tensor.getDataType());
    };
    size_t result = tensorBytes(outputEmbedding);
    for (Tensor const& feature : deepstackFeatures)
    {
        result += tensorBytes(feature);
    }
    return result;
}

size_t PhaseVisionPayload::releasePrefillStorage() noexcept
{
    // A legacy payload keeps M-RoPE in the same slab as the prefill tensors. Retaining the whole slab is required
    // until decode finishes; only the split-lease path can safely release it after the final prefill.
    if (!mropeCosSin.isEmpty() && mropeStorageOwner == nullptr)
    {
        return 0;
    }
    size_t const releasedBytes = prefillByteSize();
    outputEmbedding = Tensor{};
    deepstackFeatures.clear();
    storageOwner.reset();
    return releasedBytes;
}

std::vector<int64_t> phaseVisionEmbeddingRows(std::vector<std::vector<int32_t>> const& tokenIds, int32_t imageTokenId)
{
    ELLM_CHECK(imageTokenId >= 0, "Phase vision batching requires a configured image token ID");
    std::vector<int64_t> result;
    result.reserve(tokenIds.size());
    for (auto const& row : tokenIds)
    {
        result.push_back(static_cast<int64_t>(std::count(row.begin(), row.end(), imageTokenId)));
    }
    return result;
}

PhaseMropeStagingRange phaseMropeStagingRange(
    bool ownerChanged, int32_t validPositions, int32_t requiredPositions, int32_t capacity, int32_t granularity)
{
    ELLM_CHECK(capacity > 0 && granularity > 0, "M-RoPE staging capacity and granularity must be positive");
    ELLM_CHECK(requiredPositions >= 0 && requiredPositions <= capacity,
        "M-RoPE staging requirement is outside the cache capacity");
    ELLM_CHECK(
        validPositions >= 0 && validPositions <= capacity, "M-RoPE staging valid prefix is outside the cache capacity");
    int32_t const copyOffset = ownerChanged ? 0 : validPositions;
    if (requiredPositions <= copyOffset)
    {
        return {copyOffset, 0, copyOffset};
    }
    int64_t const rounded = (static_cast<int64_t>(requiredPositions) + granularity - 1) / granularity * granularity;
    int32_t const copyEnd = static_cast<int32_t>(std::min<int64_t>(capacity, rounded));
    return {copyOffset, copyEnd - copyOffset, copyEnd};
}

PhaseVisionAdapter::PhaseVisionAdapter(MultimodalRunner& runner, tokenizer::Tokenizer const& tokenizer,
    LLMEngineConfig const& config, cudaStream_t stream, PhaseVisionStoragePolicy storagePolicy)
    : mRunner(runner)
    , mTokenizer(tokenizer)
    , mConfig(config)
    , mStoragePolicy(storagePolicy)
    , mStream(stream)
{
    ELLM_CHECK(mStream != nullptr, "Phase vision adapter requires an explicit CUDA stream");
    CUDA_DRIVER_CHECK(cuStreamGetCtx(mStream, &mCudaContext));
    ELLM_CHECK(mCudaContext != nullptr, "Phase vision stream has no CUDA context");
}

Tensor PhaseVisionAdapter::viewTensorRows(Tensor& source, int64_t rowOffset, int64_t rowCount, std::string const& name)
{
    ELLM_CHECK(!source.isEmpty() && source.getDeviceType() == DeviceType::kGPU,
        "Phase vision output must be a non-empty GPU tensor");
    Coords const sourceShape = source.getShape();
    ELLM_CHECK(sourceShape.getNumDims() > 0 && rowOffset >= 0 && rowCount > 0 && rowOffset + rowCount <= sourceShape[0],
        "Phase vision row slice is outside the source tensor");
    std::vector<int64_t> resultShape;
    resultShape.reserve(static_cast<size_t>(sourceShape.getNumDims()));
    resultShape.push_back(rowCount);
    for (int32_t dim = 1; dim < sourceShape.getNumDims(); ++dim)
    {
        resultShape.push_back(sourceShape[dim]);
    }
    int64_t const elementsPerRow = sourceShape.volume() / sourceShape[0];
    size_t const elementBytes = utils::getTypeSize(source.getDataType());
    size_t const offsetBytes = static_cast<size_t>(rowOffset * elementsPerRow) * elementBytes;
    auto* sourceBytes = static_cast<std::byte*>(source.rawPointer());
    return Tensor(sourceBytes + offsetBytes, Coords(resultShape), DeviceType::kGPU, source.getDataType(), name);
}

std::shared_ptr<PhaseVisionBatchStorage> PhaseVisionAdapter::acquireBatchStorage()
{
    reclaimIdleStorage();
    std::shared_ptr<PhaseVisionBatchStorage> storage;
    for (auto const& candidate : mStoragePool)
    {
        if (candidate.use_count() == 1)
        {
            storage = candidate;
            ++mMemoryStats.batchStorageReuses;
            break;
        }
    }
    if (storage == nullptr)
    {
        storage = std::make_shared<PhaseVisionBatchStorage>();
        mStoragePool.push_back(storage);
        ++mMemoryStats.batchStorageAllocations;
    }

    storage->lastUseGeneration = ++mStorageGeneration;
    refreshIdleStorageStats();
    return storage;
}

void PhaseVisionAdapter::copyRunnerOutputs(
    PhaseVisionBatchStorage& storage, Tensor const& outputEmbedding, OptionalInputTensors const& deepstackFeatures)
{
    auto retain = [&](Tensor const& source, Tensor& destination, std::string const& name) {
        resizeTensor(destination, source, name);
        size_t const copyBytes
            = static_cast<size_t>(source.getShape().volume()) * utils::getTypeSize(source.getDataType());
        CUDA_CHECK(cudaMemcpyAsync(
            destination.rawPointer(), source.rawPointer(), copyBytes, cudaMemcpyDeviceToDevice, mStream));
        ++mMemoryStats.deviceCopyOperations;
        mMemoryStats.deviceCopyBytes += copyBytes;
    };
    retain(outputEmbedding, storage.outputEmbedding, "phase_vision_batch_output");
    storage.deepstackFeatures.resize(deepstackFeatures.size());
    for (size_t index{}; index < deepstackFeatures.size(); ++index)
    {
        retain(deepstackFeatures[index], storage.deepstackFeatures[index], "phase_vision_batch_deepstack");
    }
}

bool PhaseVisionAdapter::submit(uint64_t requestId, LLMGenerationRequest const& request)
{
    return submit(std::vector<PhaseVisionSubmission>{{requestId, request}});
}

bool PhaseVisionAdapter::submit(std::vector<PhaseVisionSubmission> submissions)
{
    ELLM_CHECK(mRequests.empty(), "Phase vision adapter currently permits one in-flight encoder batch");
    ELLM_CHECK(!submissions.empty(), "Phase vision encoder batch cannot be empty");
    std::unordered_set<uint64_t> requestIds;
    requestIds.reserve(submissions.size());
    for (auto const& submission : submissions)
    {
        ELLM_CHECK(requestIds.insert(submission.requestId).second, "Duplicate request ID in phase vision batch");
        ELLM_CHECK(submission.request.requests.size() == 1U,
            "Each phase vision submission must contain exactly one logical request");
    }

    LLMGenerationRequest batchedRequest = std::move(submissions.front().request);
    for (size_t index = 1; index < submissions.size(); ++index)
    {
        LLMGenerationRequest& request = submissions[index].request;
        ELLM_CHECK(request.applyChatTemplate == batchedRequest.applyChatTemplate
                && request.addGenerationPrompt == batchedRequest.addGenerationPrompt
                && request.enableThinking == batchedRequest.enableThinking,
            "Phase vision encoder batch requires identical chat formatting options");
        batchedRequest.requests.push_back(std::move(request.requests.front()));
    }
    batchedRequest.formattedRequests.resize(batchedRequest.requests.size());
    batchedRequest.streamChannels.clear();
    for (size_t index = 0; index < batchedRequest.requests.size(); ++index)
    {
        ELLM_CHECK(
            mTokenizer.applyChatTemplate(batchedRequest.requests[index], batchedRequest.formattedRequests[index],
                batchedRequest.applyChatTemplate, batchedRequest.addGenerationPrompt, batchedRequest.enableThinking),
            "Failed to format phase vision request");
    }
    mBatchedRequest.emplace(std::move(batchedRequest));

    std::vector<std::unique_ptr<PhaseVisionPayload>> payloads;
    payloads.reserve(submissions.size());
    std::shared_ptr<PhaseVisionBatchStorage> const storage = acquireBatchStorage();
    std::shared_ptr<PhaseVisionMropeStorage> mropeStorage;
    Tensor* mropeCosSin = &storage->mropeCosSin;
    if (mStoragePolicy.splitMropeLease)
    {
        storage->mropeCosSin = Tensor{};
        mropeStorage = std::make_shared<PhaseVisionMropeStorage>();
        mropeCosSin = &mropeStorage->mropeCosSin;
    }
    try
    {
        for (size_t index = 0; index < submissions.size(); ++index)
        {
            auto payload = std::make_unique<PhaseVisionPayload>();
            CUDA_CHECK(cudaEventCreate(&payload->startEvent));
            CUDA_CHECK(cudaEventCreate(&payload->readyEvent));
            CUDA_CHECK(cudaEventRecord(payload->startEvent, mStream));
            payloads.push_back(std::move(payload));
        }

        if (mConfig.ropeConfig.type == RopeType::kMRope)
        {
            int64_t const activeBatchSize = static_cast<int64_t>(submissions.size());
            ELLM_CHECK(activeBatchSize <= mConfig.maxSupportedBatchSize,
                "Phase vision M-RoPE batch is outside the engine profile");
            Coords const requiredShape{activeBatchSize, mConfig.maxKVCacheCapacity, mConfig.rotaryDim};
            int64_t const requiredBytes
                = requiredShape.volume() * static_cast<int64_t>(utils::getTypeSize(nvinfer1::DataType::kFLOAT));
            if (mropeCosSin->isEmpty() || mropeCosSin->getMemoryCapacity() < requiredBytes)
            {
                *mropeCosSin
                    = Tensor(requiredShape, DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "phase_vision_batched_mrope");
            }
            else
            {
                ELLM_CHECK(mropeCosSin->reshape(requiredShape), "Failed to reshape batched vision M-RoPE storage");
            }
        }
        else
        {
            *mropeCosSin = Tensor{};
        }
        OptionalOutputTensor mrope
            = mropeCosSin->isEmpty() ? std::nullopt : OptionalOutputTensor{std::ref(*mropeCosSin)};
        std::vector<std::vector<int32_t>> batchedTokenIds;
        ELLM_CHECK(mRunner.preprocess(*mBatchedRequest, batchedTokenIds, &mTokenizer, mrope, mStream),
            "Phase vision preprocessing failed");
        ELLM_CHECK(batchedTokenIds.size() == submissions.size(),
            "Phase vision preprocessing returned the wrong logical batch size");

        std::vector<int64_t> const embeddingRows = phaseVisionEmbeddingRows(batchedTokenIds, mConfig.imageTokenId);
        int64_t const totalEmbeddingRows = std::accumulate(embeddingRows.begin(), embeddingRows.end(), int64_t{});
        Tensor const& outputEmbedding = mRunner.getOutputEmbedding();
        ELLM_CHECK(outputEmbedding.getShape().getNumDims() > 0 && outputEmbedding.getShape()[0] == totalEmbeddingRows,
            "Phase vision embedding rows do not match expanded image-token rows");
        OptionalInputTensors const deepstackFeatures = mRunner.getDeepstackFeatures();
        for (Tensor const& feature : deepstackFeatures)
        {
            ELLM_CHECK(feature.getShape().getNumDims() > 0 && feature.getShape()[0] == totalEmbeddingRows,
                "Phase vision deepstack rows do not match expanded image-token rows");
        }
        resizeTensor(storage->outputEmbedding, outputEmbedding, "phase_vision_batch_output");
        storage->deepstackFeatures.resize(deepstackFeatures.size());
        std::vector<std::reference_wrapper<Tensor>> externalDeepstack;
        externalDeepstack.reserve(deepstackFeatures.size());
        for (size_t index{}; index < deepstackFeatures.size(); ++index)
        {
            resizeTensor(storage->deepstackFeatures[index], deepstackFeatures[index], "phase_vision_batch_deepstack");
            externalDeepstack.emplace_back(storage->deepstackFeatures[index]);
        }
        bool const directOutput = mRunner.bindExternalOutputStorage(storage->outputEmbedding, externalDeepstack);
        ELLM_CHECK(mRunner.infer(mStream), "Phase vision inference failed");
        if (directOutput)
        {
            auto activeBytes = [](Tensor const& tensor) {
                return static_cast<size_t>(tensor.getShape().volume()) * utils::getTypeSize(tensor.getDataType());
            };
            ++mMemoryStats.directOutputBatches;
            mMemoryStats.directOutputBytes += activeBytes(storage->outputEmbedding);
            for (Tensor const& feature : storage->deepstackFeatures)
            {
                mMemoryStats.directOutputBytes += activeBytes(feature);
            }
            if (!storage->mropeCosSin.isEmpty())
            {
                mMemoryStats.directOutputBytes += activeBytes(storage->mropeCosSin);
            }
        }
        else
        {
            copyRunnerOutputs(*storage, outputEmbedding, deepstackFeatures);
        }

        int64_t embeddingOffset{};
        std::vector<PhaseVisionDebugSnapshot> debugSnapshots;
        if (mDebugCallback)
        {
            debugSnapshots.reserve(submissions.size());
        }
        for (size_t index = 0; index < submissions.size(); ++index)
        {
            int64_t const rowCount = embeddingRows[index];
            ELLM_CHECK(rowCount > 0, "Phase vision logical request produced no image embedding rows");
            PhaseVisionPayload& payload = *payloads[index];
            payload.storageOwner = storage;
            payload.mropeStorageOwner = mropeStorage;
            payload.tokenIds.push_back(std::move(batchedTokenIds[index]));
            payload.outputEmbedding
                = viewTensorRows(storage->outputEmbedding, embeddingOffset, rowCount, "phase_vision_output");
            for (Tensor& feature : storage->deepstackFeatures)
            {
                payload.deepstackFeatures.push_back(
                    viewTensorRows(feature, embeddingOffset, rowCount, "phase_vision_deepstack"));
            }
            if (!mropeCosSin->isEmpty())
            {
                payload.mropeCosSin
                    = viewTensorRows(*mropeCosSin, static_cast<int64_t>(index), 1, "phase_vision_mrope");
            }
            if (mDebugCallback)
            {
                PhaseVisionDebugSnapshot snapshot;
                snapshot.requestId = submissions[index].requestId;
                snapshot.encoderBatchSize = submissions.size();
                snapshot.encoderBatchIndex = index;
                snapshot.tokenIds = payload.tokenIds.front();
                snapshot.outputEmbedding = captureDebugTensor(payload.outputEmbedding, mStream);
                snapshot.mropeCosSin = captureDebugTensor(payload.mropeCosSin, mStream);
                debugSnapshots.push_back(std::move(snapshot));
            }
            embeddingOffset += rowCount;
            CUDA_CHECK(cudaEventRecord(payload.readyEvent, mStream));
        }
        ELLM_CHECK(embeddingOffset == totalEmbeddingRows, "Phase vision embedding slicing did not consume all rows");
        if (!debugSnapshots.empty())
        {
            CUDA_CHECK(cudaStreamSynchronize(mStream));
            for (PhaseVisionDebugSnapshot const& snapshot : debugSnapshots)
            {
                mDebugCallback(snapshot);
            }
        }
    }
    catch (...)
    {
        static_cast<void>(cudaStreamSynchronize(mStream));
        mBatchedRequest.reset();
        throw;
    }

    for (size_t index = 0; index < submissions.size(); ++index)
    {
        ELLM_CHECK(mRequests.emplace(submissions[index].requestId, std::move(payloads[index])).second,
            "Failed to register phase vision batch request");
    }
    return true;
}

bool PhaseVisionAdapter::ready(uint64_t requestId) const
{
    auto const it = mRequests.find(requestId);
    ELLM_CHECK(it != mRequests.end(), "Unknown phase vision request");
    cudaError_t const status = cudaEventQuery(it->second->readyEvent);
    if (status == cudaErrorNotReady)
    {
        return false;
    }
    CUDA_CHECK(status);
    return true;
}

std::unique_ptr<PhaseVisionPayload> PhaseVisionAdapter::take(uint64_t requestId)
{
    auto it = mRequests.find(requestId);
    ELLM_CHECK(it != mRequests.end(), "Unknown phase vision request");
    ELLM_CHECK(ready(requestId), "Phase vision request is not complete");
    CUDA_CHECK(cudaEventElapsedTime(&it->second->encoderGpuMs, it->second->startEvent, it->second->readyEvent));
    std::unique_ptr<PhaseVisionPayload> result = std::move(it->second);
    mRequests.erase(it);
    releaseBatchStorageIfIdle();
    return result;
}

bool PhaseVisionAdapter::cancel(uint64_t requestId)
{
    auto it = mRequests.find(requestId);
    if (it == mRequests.end() || !ready(requestId))
    {
        return false;
    }
    mRequests.erase(it);
    releaseBatchStorageIfIdle();
    return true;
}

bool PhaseVisionAdapter::busy() const noexcept
{
    return !mRequests.empty();
}

size_t PhaseVisionAdapter::estimateInputTokens(LLMGenerationRequest const& request)
{
    int64_t const tokens = mRunner.estimateInputTokens(request);
    ELLM_CHECK(tokens >= 0, "Vision runner returned a negative input-token estimate");
    return static_cast<size_t>(tokens);
}

size_t PhaseVisionAdapter::maxInputTokens() const noexcept
{
    int64_t const tokens = mRunner.maxInputTokens();
    return tokens > 0 ? static_cast<size_t>(tokens) : 0U;
}

CUcontext PhaseVisionAdapter::cudaContext() const noexcept
{
    return mCudaContext;
}

PhaseVisionMemoryStats const& PhaseVisionAdapter::memoryStats() const noexcept
{
    return mMemoryStats;
}

void PhaseVisionAdapter::setDebugCallback(std::function<void(PhaseVisionDebugSnapshot const&)> callback)
{
    ELLM_CHECK(!busy(), "Phase vision debug callback can only change while the adapter is idle");
    mDebugCallback = std::move(callback);
}

void PhaseVisionAdapter::refreshIdleStorageStats() noexcept
{
    mMemoryStats.idleStorageBatches = 0;
    mMemoryStats.idleStorageBytes = 0;
    for (auto const& storage : mStoragePool)
    {
        if (storage.use_count() == 1)
        {
            ++mMemoryStats.idleStorageBatches;
            mMemoryStats.idleStorageBytes += storageByteSize(*storage);
        }
    }
}

void PhaseVisionAdapter::reclaimIdleStorage()
{
    refreshIdleStorageStats();
    auto overBudget = [&] {
        return mMemoryStats.idleStorageBatches > mStoragePolicy.maxIdleBatches
            || mMemoryStats.idleStorageBytes > mStoragePolicy.maxIdleBytes;
    };
    while (overBudget())
    {
        auto oldest = mStoragePool.end();
        for (auto it = mStoragePool.begin(); it != mStoragePool.end(); ++it)
        {
            if (it->use_count() != 1)
            {
                continue;
            }
            if (oldest == mStoragePool.end() || (*it)->lastUseGeneration < (*oldest)->lastUseGeneration)
            {
                oldest = it;
            }
        }
        if (oldest == mStoragePool.end())
        {
            break;
        }
        size_t const bytes = storageByteSize(**oldest);
        mStoragePool.erase(oldest);
        ++mMemoryStats.batchStorageReclaims;
        mMemoryStats.reclaimedBytes += bytes;
        refreshIdleStorageStats();
    }
}

void PhaseVisionAdapter::releaseBatchStorageIfIdle()
{
    if (!mRequests.empty())
    {
        return;
    }
    mBatchedRequest.reset();
    reclaimIdleStorage();
}

} // namespace trt_edgellm::rt
