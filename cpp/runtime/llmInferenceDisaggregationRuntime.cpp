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

#include "runtime/llmInferenceDisaggregationRuntime.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/hashUtils.h"
#include "common/logger.h"
#include "common/safetensorsUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "profiling/timer.h"
#include "sampler/sampling.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace
{

size_t hashSystemPromptWithLoraWeights(std::string const& systemPrompt, std::string const& loraWeightsName)
{
    size_t hashValue = 0;
    hash_utils::hashCombine(hashValue, systemPrompt);
    hash_utils::hashCombine(hashValue, loraWeightsName);
    return hashValue;
}

} // namespace

namespace rt
{

LLMInferenceDisaggregationRuntime::LLMInferenceDisaggregationRuntime(std::string const& engineDir,
    std::string const& multimodalEngineDir, std::unordered_map<std::string, std::string> const& loraWeightsMap,
    cudaStream_t stream)
{
    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / "llm.engine";
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";

    mLLMEngineRunner = std::make_unique<LLMEngineRunner>(enginePath, configPath, loraWeightsMap, stream);
    mEngineConfig = mLLMEngineRunner->getEngineConfig();

    int32_t const defaultTopK{0};
    float const defaultTopP{0.9F};
    trt_edgellm::SamplingParams samplingParams(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize, 1.0f, defaultTopK, defaultTopP);
    int64_t maxSamplingWorkspaceSize = static_cast<int64_t>(trt_edgellm::getTopKtopPSamplingWorkspaceSize(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize, samplingParams));

    mSamplingWorkspace = rt::Tensor(
        {maxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8, "DisaggRuntime::mSamplingWorkspace");
    mInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
        rt::DeviceType::kGPU, DataType::kINT32, "DisaggRuntime::mInputIds");
    mHostPackedInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
        rt::DeviceType::kCPU, DataType::kINT32, "DisaggRuntime::mHostPackedInputIds");
    mOutputLogits = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.vocabSize}, rt::DeviceType::kGPU,
        DataType::kFLOAT, "DisaggRuntime::mOutputLogits");
    mSelectedIndices = rt::Tensor(
        {mEngineConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU, DataType::kINT32, "DisaggRuntime::mSelected");
    mHostSelectedTokenIds = rt::Tensor(
        {mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32, "DisaggRuntime::mHostSelected");
    mHostContextLengths = rt::Tensor(
        {mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32, "DisaggRuntime::mContextLen");
    mHostReuseKVCacheLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32,
        "DisaggRuntime::mReuseKVCacheLengths");

    mTokenizer = std::make_unique<tokenizer::Tokenizer>();
    if (!mTokenizer->loadFromHF(engineDir))
    {
        throw std::runtime_error("Failed to load tokenizer from model directory: " + engineDir);
    }

    if (mEngineConfig.reducedVocabSize > 0)
    {
        std::filesystem::path const vocabMapPath = std::filesystem::path(engineDir) / binding_names::kVocabMapFileName;
        std::vector<rt::Tensor> vocabMapTensors;
        check::check(safetensors::loadSafetensors(vocabMapPath, vocabMapTensors, stream),
            std::string("Failed to load ") + binding_names::kVocabMapFileName + " from model directory: " + engineDir);
        check::check(vocabMapTensors.size() == 1, std::string(binding_names::kVocabMapFileName) + " should have 1 tensor");
        mVocabMappingTable = std::move(vocabMapTensors[0]);
    }

    if (!multimodalEngineDir.empty())
    {
        mMultimodalRunner = MultimodalRunner::create(
            multimodalEngineDir, mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxKVCacheCapacity, stream);
    }

    CUDA_CHECK(cudaStreamCreate(&mMultimodalStream));
    CUDA_CHECK(cudaStreamCreate(&mPrefillStream));
    CUDA_CHECK(cudaStreamCreate(&mDecodeStream));

    mMultimodalWorker = std::thread(&LLMInferenceDisaggregationRuntime::multimodalWorkerMain, this);
    mPrefillWorker = std::thread(&LLMInferenceDisaggregationRuntime::prefillWorkerMain, this);
    mDecodeWorker = std::thread(&LLMInferenceDisaggregationRuntime::decodeWorkerMain, this);
}

LLMInferenceDisaggregationRuntime::~LLMInferenceDisaggregationRuntime()
{
    stopWorkers();

    if (mMultimodalStream)
    {
        CUDA_CHECK(cudaStreamDestroy(mMultimodalStream));
        mMultimodalStream = nullptr;
    }
    if (mPrefillStream)
    {
        CUDA_CHECK(cudaStreamDestroy(mPrefillStream));
        mPrefillStream = nullptr;
    }
    if (mDecodeStream)
    {
        CUDA_CHECK(cudaStreamDestroy(mDecodeStream));
        mDecodeStream = nullptr;
    }
}

void LLMInferenceDisaggregationRuntime::stopWorkers()
{
    mMultimodalQueue.stop();
    mPrefillQueue.stop();
    mDecodeQueue.stop();

    if (mMultimodalWorker.joinable())
    {
        mMultimodalWorker.join();
    }
    if (mPrefillWorker.joinable())
    {
        mPrefillWorker.join();
    }
    if (mDecodeWorker.joinable())
    {
        mDecodeWorker.join();
    }
}

std::future<LLMInferenceDisaggregationRuntime::AsyncRequestResult> LLMInferenceDisaggregationRuntime::submitRequestAsync(
    LLMGenerationRequest const& request)
{
    auto context = std::make_shared<StageContext>();
    context->request = &request;
    auto future = context->completionPromise.get_future();
    mMultimodalQueue.push(std::move(context));
    return future;
}

bool LLMInferenceDisaggregationRuntime::handleRequest(
    LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream)
{
    (void) stream;
    auto future = submitRequestAsync(request);
    auto result = future.get();
    response = std::move(result.response);
    return result.success;
}

bool LLMInferenceDisaggregationRuntime::examineRequest(LLMGenerationRequest const& request)
{
    int32_t const activeBatchSize = static_cast<int32_t>(request.requests.size());
    if (activeBatchSize == 0)
    {
        LOG_ERROR("LLMInferenceDisaggregationRuntime(): The request is empty with no requests supplied.");
        return false;
    }
    if (activeBatchSize > mEngineConfig.maxSupportedBatchSize)
    {
        LOG_ERROR("LLMInferenceDisaggregationRuntime(): batch size (%d) exceeds max (%d).", activeBatchSize,
            mEngineConfig.maxSupportedBatchSize);
        return false;
    }
    for (auto const& oneRequest : request.requests)
    {
        if (oneRequest.messages.empty())
        {
            LOG_ERROR("LLMInferenceDisaggregationRuntime(): there is an empty request in batch.");
            return false;
        }
    }
    return true;
}

bool LLMInferenceDisaggregationRuntime::setUpForPrefillExecution(std::vector<std::vector<int32_t>> const& batchedInputIds,
    std::vector<std::string> const& systemPrompts, std::string const& loraWeightsName, cudaStream_t stream)
{
    std::vector<std::vector<int32_t>> processedInputIds;
    std::vector<int32_t> processedIdsLengths;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());

    rt::LinearKVCache& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    rt::Tensor kvCacheBuffer = linearKVCache.getKVCacheBuffer();

    check::check(mHostReuseKVCacheLengths.reshape({activeBatchSize}), "Failed to reshape host reuse KV cache lengths.");
    int32_t* reuseKVCacheLengthsData = mHostReuseKVCacheLengths.dataPointer<int32_t>();

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto promptHash = hashSystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
        if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
        {
            auto& precachedKVCache = mSystemPromptKVCache[promptHash];
            auto const& kvCacheContent = precachedKVCache.kvCacheContent;
            kernel::instantiateKVCacheFromTensor(kvCacheBuffer, kvCacheContent, i, stream);
            int32_t reuseLength = static_cast<int32_t>(kvCacheContent.getShape()[3]);
            processedInputIds.emplace_back(batchedInputIds[i].begin() + reuseLength, batchedInputIds[i].end());
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size() - reuseLength));
            reuseKVCacheLengthsData[i] = reuseLength;
        }
        else
        {
            processedInputIds.emplace_back(batchedInputIds[i]);
            processedIdsLengths.emplace_back(static_cast<int32_t>(batchedInputIds[i].size()));
            reuseKVCacheLengthsData[i] = 0;
        }
    }

    int32_t const maxInputLength = *std::max_element(processedIdsLengths.begin(), processedIdsLengths.end());
    if (maxInputLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR("LLMInferenceDisaggregationRuntime(): max input length (%d) exceeds max supported (%d).", maxInputLength,
            mEngineConfig.maxSupportedInputLength);
        return false;
    }

    int32_t const packedInputLength = std::max(maxInputLength, mEngineConfig.minSupportedInputLength);
    check::check(mHostPackedInputIds.reshape({activeBatchSize, packedInputLength}), "Failed to reshape packed input IDs.");
    int32_t* packedInputIdsData = mHostPackedInputIds.dataPointer<int32_t>();
    std::fill(packedInputIdsData, packedInputIdsData + activeBatchSize * packedInputLength, mTokenizer->getPadId());
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        std::copy(processedInputIds[i].begin(), processedInputIds[i].end(), packedInputIdsData + i * packedInputLength);
    }

    linearKVCache.resetForNewSequences(mHostReuseKVCacheLengths, stream);
    check::check(mInputIds.reshape({activeBatchSize, packedInputLength}), "Failed to reshape input IDs.");
    check::check(mHostContextLengths.reshape({activeBatchSize}), "Failed to reshape context lengths.");
    check::check(mOutputLogits.reshape({activeBatchSize, mEngineConfig.outputVocabSize}), "Failed to reshape output logits.");

    CUDA_CHECK(cudaMemcpyAsync(mInputIds.rawPointer(), mHostPackedInputIds.rawPointer(),
        activeBatchSize * packedInputLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    memcpy(mHostContextLengths.dataPointer<int32_t>(), processedIdsLengths.data(), activeBatchSize * sizeof(int32_t));

    if (mEngineConfig.maxSupportedLoraRank > 0 && !mLLMEngineRunner->switchLoraWeights(loraWeightsName, stream))
    {
        LOG_ERROR("Failed to switch LoRA weights to %s", loraWeightsName.c_str());
        return false;
    }
    return true;
}

LLMInferenceDisaggregationRuntime::TokenCountInfo LLMInferenceDisaggregationRuntime::calculateTokenCounts(
    std::vector<std::vector<int32_t>> const& batchedInputIds, std::vector<std::string> const& systemPrompts,
    std::string const& loraWeightsName) const
{
    TokenCountInfo tokenCount;
    int32_t const activeBatchSize = static_cast<int32_t>(batchedInputIds.size());
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        int32_t contextLength = static_cast<int32_t>(batchedInputIds[i].size());
        auto promptHash = hashSystemPromptWithLoraWeights(systemPrompts[i], loraWeightsName);
        if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
        {
            int32_t reusedLength = static_cast<int32_t>(mSystemPromptKVCache.at(promptHash).tokenizedPrompt.size());
            tokenCount.totalReusedTokens += reusedLength;
            tokenCount.totalComputedTokens += (contextLength - reusedLength);
        }
        else
        {
            tokenCount.totalComputedTokens += contextLength;
        }
    }
    return tokenCount;
}

bool LLMInferenceDisaggregationRuntime::sampleTokens(StageContext& context, cudaStream_t stream)
{
    SamplingParams params(context.activeBatchSize, mEngineConfig.outputVocabSize, context.request->temperature,
        context.request->topK, context.request->topP);
    trt_edgellm::topKtopPSamplingFromLogits(mOutputLogits, mSelectedIndices, params, mSamplingWorkspace, stream);
    if (mEngineConfig.reducedVocabSize > 0)
    {
        trt_edgellm::mapReducedVocabToFullVocab(mSelectedIndices, mVocabMappingTable, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(mHostSelectedTokenIds.rawPointer(), mSelectedIndices.rawPointer(),
        context.activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int32_t* hostSelectedTokenIdsData = mHostSelectedTokenIds.dataPointer<int32_t>();
    for (int32_t i = 0; i < context.activeBatchSize; ++i)
    {
        if (!context.finishedStates[i])
        {
            context.outputIds[i].push_back(hostSelectedTokenIdsData[i]);
            context.finishedStates[i] = hostSelectedTokenIdsData[i] == mTokenizer->getEosId();
            if (context.finishedStates[i])
            {
                context.unFinishedBatchNum--;
            }
        }
    }
    ++context.generationIter;
    return true;
}

void LLMInferenceDisaggregationRuntime::setFailedResult(StageContext& context, std::string const& message)
{
    context.status = false;
    context.errorMessage = message;
}

void LLMInferenceDisaggregationRuntime::multimodalWorkerMain()
{
    std::shared_ptr<StageContext> context;
    while (mMultimodalQueue.pop(context))
    {
        context->pipelineLock = std::make_unique<std::unique_lock<std::mutex>>(mExecutionMutex);
        if (!examineRequest(*context->request))
        {
            setFailedResult(*context, "request validation failed");
            finalizeContext(*context);
            continue;
        }

        context->activeBatchSize = static_cast<int32_t>(context->request->requests.size());
        context->loraWeightsName = context->request->loraWeightsName;
        context->request->formattedRequests.resize(context->activeBatchSize);
        context->batchSystemPrompts.reserve(context->activeBatchSize);

        for (int32_t i = 0; i < context->activeBatchSize; ++i)
        {
            mTokenizer->applyChatTemplate(context->request->requests[i], context->request->formattedRequests[i],
                context->request->applyChatTemplate, context->request->addGenerationPrompt, context->request->enableThinking);
            context->batchSystemPrompts.emplace_back(context->request->formattedRequests[i].formattedSystemPrompt);

            if (context->request->saveSystemPromptKVCache)
            {
                if (mMultimodalRunner)
                {
                    mMultimodalRunner->preprocessSystemPrompt(context->batchSystemPrompts[i], mTokenizer.get(),
                        mLLMEngineRunner->getRopeCosSinCacheTensor(), mMultimodalStream);
                }
                if (!genAndSaveSystemPromptKVCache(context->batchSystemPrompts[i], context->loraWeightsName, mMultimodalStream))
                {
                    LOG_WARNING("Failed to save system prompt KVCache. Proceed without saving cache.");
                }
            }
        }

        if (!mMultimodalRunner)
        {
            context->batchedInputIds.reserve(context->activeBatchSize);
            for (int32_t i = 0; i < context->activeBatchSize; ++i)
            {
                context->batchedInputIds.emplace_back(
                    mTokenizer->encode(context->request->formattedRequests[i].formattedCompleteRequest, true));
            }
        }
        else
        {
            if (!mMultimodalRunner->preprocess(*context->request, context->batchedInputIds, mTokenizer.get(),
                    mLLMEngineRunner->getRopeCosSinCacheTensor(), mMultimodalStream))
            {
                setFailedResult(*context, "multimodal preprocess failed");
                finalizeContext(*context);
                continue;
            }
            if (!mMultimodalRunner->infer(mMultimodalStream))
            {
                setFailedResult(*context, "multimodal inference failed");
                finalizeContext(*context);
                continue;
            }
        }

        context->tokenCount
            = calculateTokenCounts(context->batchedInputIds, context->batchSystemPrompts, context->loraWeightsName);
        CUDA_CHECK(cudaEventCreateWithFlags(&context->multimodalDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(context->multimodalDone, mMultimodalStream));
        mPrefillQueue.push(std::move(context));
    }
}

void LLMInferenceDisaggregationRuntime::prefillWorkerMain()
{
    std::shared_ptr<StageContext> context;
    while (mPrefillQueue.pop(context))
    {
        CUDA_CHECK(cudaStreamWaitEvent(mPrefillStream, context->multimodalDone, 0));

        if (!setUpForPrefillExecution(
                context->batchedInputIds, context->batchSystemPrompts, context->loraWeightsName, mPrefillStream))
        {
            setFailedResult(*context, "prefill setup failed");
            finalizeContext(*context);
            continue;
        }

        int32_t const maxInputIdsLength = mInputIds.getShape()[1];
        context->maxGenerationLength = context->request->maxGenerateLength;
        if (maxInputIdsLength + context->maxGenerationLength > mEngineConfig.maxKVCacheCapacity)
        {
            context->maxGenerationLength = mEngineConfig.maxKVCacheCapacity - maxInputIdsLength;
        }

        context->unFinishedBatchNum = context->activeBatchSize;
        context->generationIter = 0;
        context->outputIds.assign(context->activeBatchSize, std::vector<int32_t>{});
        context->finishedStates.assign(context->activeBatchSize, false);
        check::check(mSelectedIndices.reshape({context->activeBatchSize, 1}), "Failed to reshape selected indices.");
        check::check(mHostSelectedTokenIds.reshape({context->activeBatchSize}), "Failed to reshape host selected token IDs.");

        rt::OptionalInputTensor multimodalEmbeddings
            = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;
        rt::OptionalInputTensors extraVisualFeatures
            = mMultimodalRunner ? mMultimodalRunner->getExtraVisualFeatures() : rt::OptionalInputTensors{};
        rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
        {
            TIME_STAGE(metrics::StageNames::kLLM_PREFILL, mPrefillStream);
            if (!mLLMEngineRunner->executePrefillStep(mInputIds, mHostContextLengths, multimodalEmbeddings, extraVisualFeatures,
                    mOutputLogits, outputHiddenStates, mPrefillStream))
            {
                setFailedResult(*context, "prefill execution failed");
                finalizeContext(*context);
                continue;
            }
            sampleTokens(*context, mPrefillStream);
        }
        mPrefillMetrics.recordRun(context->tokenCount.totalReusedTokens, context->tokenCount.totalComputedTokens);

        check::check(mInputIds.reshape({context->activeBatchSize, 1}), "Failed to reshape decode input IDs.");
        CUDA_CHECK(cudaEventCreateWithFlags(&context->prefillDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(context->prefillDone, mPrefillStream));
        mDecodeQueue.push(std::move(context));
    }
}

void LLMInferenceDisaggregationRuntime::decodeWorkerMain()
{
    std::shared_ptr<StageContext> context;
    while (mDecodeQueue.pop(context))
    {
        CUDA_CHECK(cudaStreamWaitEvent(mDecodeStream, context->prefillDone, 0));
        {
            TIME_STAGE(metrics::StageNames::kLLM_GENERATION, mDecodeStream);
            while (context->unFinishedBatchNum > 0 && context->generationIter < context->maxGenerationLength)
            {
                if (!mLLMEngineRunner->executeVanillaDecodingStep(mSelectedIndices, mOutputLogits, mDecodeStream))
                {
                    setFailedResult(*context, "decode execution failed");
                    break;
                }
                sampleTokens(*context, mDecodeStream);
            }
        }

        if (context->errorMessage.empty())
        {
            int32_t totalGeneratedTokens = 0;
            for (int32_t i = 0; i < context->activeBatchSize; ++i)
            {
                totalGeneratedTokens += static_cast<int32_t>(context->outputIds[i].size() - 1);
            }
            if (totalGeneratedTokens > 0)
            {
                mGenerationMetrics.recordRun(totalGeneratedTokens);
            }

            context->response.outputIds.clear();
            context->response.outputTexts.clear();
            for (int32_t i = 0; i < context->activeBatchSize; ++i)
            {
                context->response.outputIds.emplace_back(context->outputIds[i]);
                context->response.outputTexts.emplace_back(mTokenizer->decode(context->outputIds[i], true));
            }
            context->status = true;
        }

        finalizeContext(*context);
    }
}

void LLMInferenceDisaggregationRuntime::finalizeContext(StageContext& context)
{
    if (context.multimodalDone)
    {
        CUDA_CHECK(cudaEventDestroy(context.multimodalDone));
        context.multimodalDone = nullptr;
    }
    if (context.prefillDone)
    {
        CUDA_CHECK(cudaEventDestroy(context.prefillDone));
        context.prefillDone = nullptr;
    }

    AsyncRequestResult result;
    result.success = context.status;
    result.response = std::move(context.response);
    context.completionPromise.set_value(std::move(result));
    context.pipelineLock.reset();
}

bool LLMInferenceDisaggregationRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    bool captureStatus{true};
    int32_t const minSupportedBatchSize = 1;
    for (int32_t batchSize = minSupportedBatchSize; batchSize <= mEngineConfig.maxSupportedBatchSize; ++batchSize)
    {
        check::check(mSelectedIndices.reshape({batchSize, 1}), "Failed to reshape selected indices for CUDA graph capture.");
        check::check(
            mOutputLogits.reshape({batchSize, mEngineConfig.outputVocabSize}), "Failed to reshape output logits for capture.");
        captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(
            mSelectedIndices, mOutputLogits, mEmptyLoraWeightsName, stream);
        if (mEngineConfig.maxSupportedLoraRank > 0)
        {
            for (auto const& loraWeightsName : mLLMEngineRunner->getAvailableLoraWeights())
            {
                captureStatus &= mLLMEngineRunner->captureVanillaDecodingCudaGraph(
                    mSelectedIndices, mOutputLogits, loraWeightsName, stream);
            }
        }
    }
    return captureStatus;
}

bool LLMInferenceDisaggregationRuntime::genAndSaveSystemPromptKVCache(
    std::string const& prompt, std::string const& loraWeightsName, cudaStream_t stream)
{
    if (prompt.empty())
    {
        return true;
    }

    size_t const promptHash = hashSystemPromptWithLoraWeights(prompt, loraWeightsName);
    if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
    {
        return true;
    }

    auto tokenizedPrompt = mTokenizer->encode(prompt, true);
    int32_t const promptIdsLength = static_cast<int32_t>(tokenizedPrompt.size());
    if (promptIdsLength > mEngineConfig.maxSupportedInputLength)
    {
        LOG_ERROR("LLMInferenceDisaggregationRuntime(): prompt length exceeds max supported input.");
        return false;
    }

    std::vector<std::vector<int32_t>> batchedInputIds(1, tokenizedPrompt);
    std::vector<std::string> batchedSystemPrompts(1, prompt);
    if (!setUpForPrefillExecution(batchedInputIds, batchedSystemPrompts, loraWeightsName, stream))
    {
        return false;
    }

    rt::OptionalInputTensor multimodalEmbeddings
        = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensors extraVisualFeatures
        = mMultimodalRunner ? mMultimodalRunner->getExtraVisualFeatures() : rt::OptionalInputTensors{};
    rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
    if (!mLLMEngineRunner->executePrefillStep(
            mInputIds, mHostContextLengths, multimodalEmbeddings, extraVisualFeatures, mOutputLogits, outputHiddenStates, stream))
    {
        return false;
    }

    auto& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    auto cacheConfig = linearKVCache.getConfig();
    auto kvCacheBuffer = linearKVCache.getKVCacheBuffer();
    rt::Coords savedKVCacheShape{
        cacheConfig.numDecoderLayers, 2, cacheConfig.numKVHeads, promptIdsLength, cacheConfig.headDim};

    DisaggregationSystemPromptKVCache savedKVCache;
    savedKVCache.systemPrompt = prompt;
    savedKVCache.tokenizedPrompt = tokenizedPrompt;
    savedKVCache.kvCacheContent = rt::Tensor(savedKVCacheShape, rt::DeviceType::kGPU, rt::LinearKVCache::KVCacheTypeTRT,
        "DisaggRuntime::savedKVCache.kvCacheContent");

    constexpr int32_t cacheBatchIdx{0};
    kernel::saveKVCacheIntoTensor(savedKVCache.kvCacheContent, kvCacheBuffer, cacheBatchIdx, stream);
    mSystemPromptKVCache.insert({promptHash, std::move(savedKVCache)});
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return true;
}

} // namespace rt
} // namespace trt_edgellm
