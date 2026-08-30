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

#pragma once

#include "multimodal/multimodalRunner.h"
#include "runtime/config/llmEngineConfig.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace trt_edgellm::rt
{

struct PhaseVisionBatchStorage;
struct PhaseVisionMropeStorage;
struct PhaseVisionPreparedBatch;
enum class PhaseActivityKind : uint8_t;
class PhaseActivityTimelineRecorder;

//! High-water policy for encoder-output slabs that are no longer request-owned.
struct PhaseVisionStoragePolicy
{
    size_t maxIdleBatches{4U};
    size_t maxIdleBytes{256U * 1024U * 1024U};
    bool splitMropeLease{};
};

//! GPU-memory operations performed while retaining encoder output for downstream prefill.
struct PhaseVisionMemoryStats
{
    size_t batchStorageAllocations{};
    size_t batchStorageReuses{};
    size_t batchStorageReclaims{};
    size_t reclaimedBytes{};
    size_t directOutputBatches{};
    size_t directOutputBytes{};
    size_t deviceCopyOperations{};
    size_t deviceCopyBytes{};
    size_t idleStorageBatches{};
    size_t idleStorageBytes{};
};

//! Incremental row interval required when staging request-owned M-RoPE data.
struct PhaseMropeStagingRange
{
    int32_t offsetPositions{};
    int32_t countPositions{};
    int32_t validPositions{};
};

//! Compute the newly required M-RoPE prefix, rounded to a bounded copy granularity.
PhaseMropeStagingRange phaseMropeStagingRange(
    bool ownerChanged, int32_t validPositions, int32_t requiredPositions, int32_t capacity, int32_t granularity);
//! Whether the model's positional encoding supports a text-only KV prefix followed by a deferred vision suffix.
bool phaseVisionSupportsPrefixBeforeVision(RopeType ropeType) noexcept;

//! Request-owned encoder output retained until the corresponding prefill completes.
struct PhaseVisionPayload
{
    ~PhaseVisionPayload() noexcept;

    size_t byteSize() const noexcept;
    size_t prefillByteSize() const noexcept;
    size_t releasePrefillStorage() noexcept;

    std::vector<std::vector<int32_t>> tokenIds;
    Tensor outputEmbedding;
    std::vector<Tensor> deepstackFeatures;
    Tensor mropeCosSin;
    std::shared_ptr<PhaseVisionBatchStorage> storageOwner;
    std::shared_ptr<PhaseVisionMropeStorage> mropeStorageOwner;
    float encoderGpuMs{};
    cudaEvent_t startEvent{};
    cudaEvent_t readyEvent{};
};

//! One logical request submitted as part of a shared vision-encoder batch.
struct PhaseVisionSubmission
{
    uint64_t requestId{};
    LLMGenerationRequest request;
};

struct PhaseVisionPrefixPlan
{
    std::vector<int32_t> prefixTokens;
    int32_t estimatedFinalPromptTokens{};
};

//! Host copy of one encoder boundary tensor for opt-in numerical bisecting.
struct PhaseVisionDebugTensor
{
    std::vector<int64_t> shape;
    nvinfer1::DataType dataType{nvinfer1::DataType::kFLOAT};
    std::vector<std::byte> bytes;
};

//! Per-request encoder outputs captured before downstream prefill ownership transfer.
struct PhaseVisionDebugSnapshot
{
    uint64_t requestId{};
    size_t encoderBatchSize{};
    size_t encoderBatchIndex{};
    std::vector<int32_t> tokenIds;
    PhaseVisionDebugTensor outputEmbedding;
    PhaseVisionDebugTensor mropeCosSin;
};

//! Return the contiguous vision-embedding row count for every expanded token row.
std::vector<int64_t> phaseVisionEmbeddingRows(std::vector<std::vector<int32_t>> const& tokenIds, int32_t imageTokenId);

//! Model-neutral wrapper around a v0.10 MultimodalRunner execution context.
//!
//! One encoder batch is in flight at a time because the borrowed runner owns
//! reusable output buffers. Runners that implement external output binding write
//! directly into retained slabs; other runners fall back to one batch-level copy.
class PhaseVisionAdapter
{
public:
    PhaseVisionAdapter(MultimodalRunner& runner, tokenizer::Tokenizer const& tokenizer, LLMEngineConfig const& config,
        cudaStream_t stream, PhaseVisionStoragePolicy storagePolicy = {}, cudaStream_t copyStream = nullptr);
    ~PhaseVisionAdapter() noexcept;

    PhaseVisionAdapter(PhaseVisionAdapter const&) = delete;
    PhaseVisionAdapter& operator=(PhaseVisionAdapter const&) = delete;

    bool submit(uint64_t requestId, LLMGenerationRequest const& request);
    bool submit(std::vector<PhaseVisionSubmission> submissions);
    //! Prepare text, image tensors, and M-RoPE state without enqueuing the encoder engine.
    std::shared_ptr<PhaseVisionPreparedBatch> prepare(std::vector<PhaseVisionSubmission> submissions);
    //! Bind retained outputs and enqueue an already prepared encoder batch.
    bool submitPrepared(std::shared_ptr<PhaseVisionPreparedBatch> prepared);
    bool ready(uint64_t requestId) const;
    std::unique_ptr<PhaseVisionPayload> take(uint64_t requestId);
    bool cancel(uint64_t requestId);
    bool busy() const noexcept;
    size_t estimateInputTokens(LLMGenerationRequest const& request);
    //! Estimate request-owned encoder output, deepstack, and M-RoPE bytes retained through prefill.
    size_t estimatePayloadBytes(LLMGenerationRequest const& request);
    //! Tokenize the causal text prefix before the first image placeholder without launching encoder work.
    std::optional<PhaseVisionPrefixPlan> makePrefixPlan(LLMGenerationRequest const& request);
    size_t maxInputTokens() const noexcept;
    CUcontext cudaContext() const noexcept;
    PhaseVisionMemoryStats const& memoryStats() const noexcept;
    //! Reclaim idle slabs above the configured high-water mark, or every idle slab under broker pressure.
    void reclaimIdleStorage(bool force = false);
    //! Enable synchronous debug capture only while the adapter is idle.
    void setDebugCallback(std::function<void(PhaseVisionDebugSnapshot const&)> callback);
    //! Enable opt-in epoch-relative E/C stream activity recording while idle.
    void setActivityTimeline(PhaseActivityTimelineRecorder* timeline);

private:
    static Tensor viewTensorRows(Tensor& source, int64_t rowOffset, int64_t rowCount, std::string const& name);
    std::shared_ptr<PhaseVisionBatchStorage> acquireBatchStorage();
    void copyRunnerOutputs(PhaseVisionBatchStorage& storage, Tensor const& outputEmbedding,
        OptionalInputTensors const& deepstackFeatures, cudaStream_t stream);
    void recordActivity(PhaseActivityKind kind, char const* name, uint64_t correlationId, cudaStream_t stream,
        std::function<void()> const& enqueue);
    void refreshIdleStorageStats() noexcept;
    void releaseBatchStorageIfIdle();

    MultimodalRunner& mRunner;
    tokenizer::Tokenizer const& mTokenizer;
    LLMEngineConfig mConfig;
    PhaseVisionStoragePolicy mStoragePolicy;
    cudaStream_t mStream{};
    cudaStream_t mCopyStream{};
    CUcontext mCudaContext{};
    cudaEvent_t mEncoderDoneEvent{};
    std::unordered_map<uint64_t, std::unique_ptr<PhaseVisionPayload>> mRequests;
    std::optional<LLMGenerationRequest> mBatchedRequest;
    std::vector<std::shared_ptr<PhaseVisionBatchStorage>> mStoragePool;
    size_t mStorageGeneration{};
    PhaseVisionMemoryStats mMemoryStats;
    std::function<void(PhaseVisionDebugSnapshot const&)> mDebugCallback;
    PhaseActivityTimelineRecorder* mActivityTimeline{};
};

} // namespace trt_edgellm::rt
