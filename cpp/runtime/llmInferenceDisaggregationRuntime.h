/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "profiling/metrics.h"
#include "runtime/llmEngineRunner.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

/*! \brief Structure to hold cached system prompt and its KV cache
 */
struct DisaggregationSystemPromptKVCache
{
    std::string systemPrompt;                     //!< The system prompt text
    std::vector<tokenizer::Rank> tokenizedPrompt; //!< Tokenized version of the system prompt
    rt::Tensor kvCacheContent;                    //!< Cached KV cache content for the system prompt
};

/*! \brief LLM disaggregation runtime with 3-stage worker pipeline
 */
class LLMInferenceDisaggregationRuntime
{
public:
    struct AsyncRequestResult
    {
        bool success{false};
        LLMGenerationResponse response{};
    };

    /*! \brief Construct a disaggregation runtime
     *  \param engineDir Directory containing the LLM engine
     *  \param multimodalEngineDir Directory containing the multimodal engine
     *  \param loraWeightsMap Map of LoRA weights names to paths
     *  \param stream CUDA stream for initialization
     */
    LLMInferenceDisaggregationRuntime(std::string const& engineDir, std::string const& multimodalEngineDir,
        std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream,
        int32_t decodeTpcCount = -1);

    //! \brief Destructor
    ~LLMInferenceDisaggregationRuntime();

    //! \brief Submit asynchronous request through disaggregation pipeline
    std::future<AsyncRequestResult> submitRequestAsync(LLMGenerationRequest const& request);

    //! \brief Handle request and wait until done
    bool handleRequest(LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream);

    //! \brief Capture decode CUDA graph for runtime decode stage
    bool captureDecodingCUDAGraph(cudaStream_t stream);

    //! \brief Generate and save system prompt KV cache
    bool genAndSaveSystemPromptKVCache(
        std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream);

    //! \brief Get prefill metrics
    metrics::LLMPrefillMetrics const& getPrefillMetrics() const
    {
        return mPrefillMetrics;
    }

    //! \brief Get generation metrics
    metrics::LLMGenerationMetrics const& getGenerationMetrics() const
    {
        return mGenerationMetrics;
    }

    //! \brief Get multimodal metrics
    metrics::MultimodalMetrics getMultimodalMetrics() const
    {
        return mMultimodalRunner ? mMultimodalRunner->getMultimodalMetrics() : metrics::MultimodalMetrics{};
    }

private:
    template <typename T>
    class BlockingQueue
    {
    public:
        void push(T&& value)
        {
            {
                std::lock_guard<std::mutex> lock(mMutex);
                if (mStopped)
                {
                    return;
                }
                mQueue.emplace_back(std::move(value));
            }
            mCv.notify_one();
        }

        bool pop(T& value)
        {
            std::unique_lock<std::mutex> lock(mMutex);
            mCv.wait(lock, [&]() { return mStopped || !mQueue.empty(); });
            if (mQueue.empty())
            {
                return false;
            }
            value = std::move(mQueue.front());
            mQueue.pop_front();
            return true;
        }

        void stop()
        {
            {
                std::lock_guard<std::mutex> lock(mMutex);
                mStopped = true;
            }
            mCv.notify_all();
        }

    private:
        std::mutex mMutex;
        std::condition_variable mCv;
        std::deque<T> mQueue;
        bool mStopped{false};
    };

    struct TokenCountInfo
    {
        int32_t totalReusedTokens{0};
        int32_t totalComputedTokens{0};
    };

    struct StageContext
    {
        uint64_t requestId{0};
        LLMGenerationRequest ownedRequest{};
        LLMGenerationRequest const* request{nullptr};
        LLMGenerationResponse response{};
        std::promise<AsyncRequestResult> completionPromise;
        std::atomic<bool> completionSet{false};
        bool status{false};
        std::string errorMessage;

        std::vector<std::vector<int32_t>> batchedInputIds;
        std::vector<std::string> batchSystemPrompts;
        std::string loraWeightsName;
        TokenCountInfo tokenCount{};

        int32_t activeBatchSize{0};
        int32_t maxGenerationLength{0};
        int32_t unFinishedBatchNum{0};
        int32_t generationIter{0};
        int32_t slotOffset{0};

        rt::Tensor samplingWorkspace{};
        rt::Tensor inputIds{};
        rt::Tensor hostPackedInputIds{};
        rt::Tensor hostContextLengths{};
        rt::Tensor outputLogits{};
        rt::Tensor selectedIndices{};
        rt::Tensor hostSelectedTokenIds{};
        rt::Tensor hostReuseKVCacheLengths{};

        std::vector<std::vector<int32_t>> outputIds;
        std::vector<bool> finishedStates;

        cudaEvent_t multimodalDone{nullptr};
        cudaEvent_t prefillDone{nullptr};
        std::mutex prefillSyncMutex;
        std::condition_variable prefillSyncCv;
        bool prefillStageCompleted{false};
    };

    bool examineRequest(LLMGenerationRequest const& request);
    bool setUpForPrefillExecution(StageContext& context, cudaStream_t stream);
    TokenCountInfo calculateTokenCounts(std::vector<std::vector<int32_t>> const& batchedInputIds,
        std::vector<std::string> const& systemPrompts, std::string const& loraWeightsName) const;
    bool sampleTokens(StageContext& context, cudaStream_t stream);
    void setFailedResult(StageContext& context, std::string const& message);

    void multimodalWorkerMain();
    void prefillWorkerMain();
    void decodeWorkerMain();
    void finalizeContext(StageContext& context);
    void stopWorkers();
    int32_t allocateSlotRange(int32_t slotCount);
    void releaseSlotRange(int32_t slotOffset, int32_t slotCount);
    bool applyDisaggregatedTpcMasks(int32_t decodeTpcCount);
    bool applyTpcMaskToStream(cudaStream_t stream, __uint128_t mask, char const* streamName) const;
    static std::vector<uint32_t> getJetsonThorTpcOrderFromGpcMasks();

    std::unique_ptr<LLMEngineRunner> mLLMEngineRunner{nullptr};
    std::unique_ptr<MultimodalRunner> mMultimodalRunner{nullptr};
    std::unique_ptr<tokenizer::Tokenizer> mTokenizer{nullptr};
    std::unordered_map<size_t, DisaggregationSystemPromptKVCache> mSystemPromptKVCache{};

    rt::Tensor mSamplingWorkspace{};
    rt::Tensor mInputIds{};
    rt::Tensor mHostPackedInputIds{};
    rt::Tensor mHostContextLengths{};
    rt::Tensor mOutputLogits{};
    rt::Tensor mSelectedIndices{};
    rt::Tensor mHostSelectedTokenIds{};
    rt::Tensor mHostReuseKVCacheLengths{};
    rt::Tensor mVocabMappingTable{};
    int64_t mMaxSamplingWorkspaceSize{0};
    std::string mEmptyLoraWeightsName{""};
    LLMEngineRunnerConfig mEngineConfig{};

    metrics::LLMPrefillMetrics mPrefillMetrics;
    metrics::LLMGenerationMetrics mGenerationMetrics;

    cudaStream_t mMultimodalStream{nullptr};
    cudaStream_t mPrefillStream{nullptr};
    cudaStream_t mDecodeStream{nullptr};

    BlockingQueue<std::shared_ptr<StageContext>> mMultimodalQueue;
    BlockingQueue<std::shared_ptr<StageContext>> mPrefillQueue;
    BlockingQueue<std::shared_ptr<StageContext>> mDecodeQueue;

    std::thread mMultimodalWorker;
    std::thread mPrefillWorker;
    std::thread mDecodeWorker;

    std::mutex mSlotAllocatorMutex;
    std::condition_variable mSlotAllocatorCv;
    std::vector<bool> mSlotUsage;

    std::mutex mRunnerPrefillMutex;
    std::mutex mRunnerDecodeMutex;
    std::atomic<uint64_t> mRequestCounter{0};
    std::atomic<bool> mSystemPromptKVCacheDisabledWarningLogged{false};
};

} // namespace rt
} // namespace trt_edgellm
