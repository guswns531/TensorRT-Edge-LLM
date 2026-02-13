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
#include <array>
#include <cstring>
#include <filesystem>
#include <functional>
#include <stdexcept>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace
{
constexpr size_t kCuda130MaskOffsetJetson = 0x54c;
constexpr std::array<uint32_t, 3> kJetsonThorGpcMasks{
    0x049, // GPC0 -> TPC {0,3,6}
    0x092, // GPC1 -> TPC {1,4,7}
    0x324 // GPC2 -> TPC {2,5,8,9}
};

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
    cudaStream_t stream, int32_t decodeTpcCount)
{
    std::filesystem::path const enginePath = std::filesystem::path(engineDir) / "llm.engine";
    std::filesystem::path const configPath = std::filesystem::path(engineDir) / "config.json";

    mLLMEngineRunner = std::make_unique<LLMEngineRunner>(enginePath, configPath, loraWeightsMap, stream);
    mEngineConfig = mLLMEngineRunner->getEngineConfig();
    mSlotUsage.assign(static_cast<size_t>(mEngineConfig.maxSupportedBatchSize), false);

    int32_t const defaultTopK{0};
    float const defaultTopP{0.9F};
    trt_edgellm::SamplingParams samplingParams(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize, 1.0f, defaultTopK, defaultTopP);
    int64_t maxSamplingWorkspaceSize = static_cast<int64_t>(trt_edgellm::getTopKtopPSamplingWorkspaceSize(
        mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize, samplingParams));
    mMaxSamplingWorkspaceSize = maxSamplingWorkspaceSize;

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
    if (decodeTpcCount > 0 && !applyDisaggregatedTpcMasks(decodeTpcCount))
    {
        throw std::runtime_error("Failed to apply disaggregated TPC masks to runtime streams.");
    }

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

std::vector<uint32_t> LLMInferenceDisaggregationRuntime::getJetsonThorTpcOrderFromGpcMasks()
{
    std::vector<uint32_t> order;
    order.reserve(10);
    for (auto const gpcMask : kJetsonThorGpcMasks)
    {
        for (uint32_t bit = 0; bit < 32; ++bit)
        {
            if (gpcMask & (1u << bit))
            {
                order.push_back(bit);
            }
        }
    }
    return order;
}

bool LLMInferenceDisaggregationRuntime::applyTpcMaskToStream(
    cudaStream_t stream, __uint128_t mask, char const* streamName) const
{
    int cudaRuntimeVersion = 0;
    if (cudaRuntimeGetVersion(&cudaRuntimeVersion) != cudaSuccess)
    {
        LOG_ERROR("Failed to get CUDA runtime version for TPC mask.");
        return false;
    }
    if (cudaRuntimeVersion / 1000 != 13)
    {
        LOG_ERROR("TPC mask currently supports CUDA 13.x only. Detected runtime version=%d", cudaRuntimeVersion);
        return false;
    }

    auto streamStructBase = *reinterpret_cast<char**>(stream);
    if (streamStructBase == nullptr)
    {
        LOG_ERROR("Failed to resolve internal stream struct for TPC mask (%s).", streamName);
        return false;
    }

    struct StreamSmMaskV2
    {
        uint32_t enabled;
        uint32_t mask[4];
    };

    auto* hwMask = reinterpret_cast<StreamSmMaskV2*>(streamStructBase + kCuda130MaskOffsetJetson);
    hwMask->enabled = 1;
    hwMask->mask[0] = static_cast<uint32_t>(mask);
    hwMask->mask[1] = static_cast<uint32_t>(mask >> 32);
    hwMask->mask[2] = static_cast<uint32_t>(mask >> 64);
    hwMask->mask[3] = static_cast<uint32_t>(mask >> 96);

    LOG_INFO("Applied disaggregated TPC mask to %s stream: mask64=0x%016llx", streamName,
        static_cast<unsigned long long>(mask));
    return true;
}

bool LLMInferenceDisaggregationRuntime::applyDisaggregatedTpcMasks(int32_t decodeTpcCount)
{
    auto const tpcOrder = getJetsonThorTpcOrderFromGpcMasks();
    int32_t const totalTpcCount = static_cast<int32_t>(tpcOrder.size());
    if (decodeTpcCount <= 0 || decodeTpcCount >= totalTpcCount)
    {
        LOG_ERROR("Invalid decode tpcCount=%d. Expected range is [1, %d).", decodeTpcCount, totalTpcCount);
        return false;
    }

    // libsmctrl semantics: 0-bit means enabled TPC, 1-bit means disabled TPC.
    __uint128_t decodeMask = ~static_cast<__uint128_t>(0);
    __uint128_t prefillEncodingMask = ~static_cast<__uint128_t>(0);

    for (int32_t i = 0; i < decodeTpcCount; ++i)
    {
        decodeMask &= ~(static_cast<__uint128_t>(1) << tpcOrder[static_cast<size_t>(i)]);
    }
    for (int32_t i = decodeTpcCount; i < totalTpcCount; ++i)
    {
        prefillEncodingMask &= ~(static_cast<__uint128_t>(1) << tpcOrder[static_cast<size_t>(i)]);
    }

    bool const decodeStatus = applyTpcMaskToStream(mDecodeStream, decodeMask, "decode");
    bool const prefillStatus = applyTpcMaskToStream(mPrefillStream, prefillEncodingMask, "prefill");
    bool const multimodalStatus = applyTpcMaskToStream(mMultimodalStream, prefillEncodingMask, "multimodal");
    if (!decodeStatus || !prefillStatus || !multimodalStatus)
    {
        return false;
    }

    LOG_INFO("TPC split applied. decode tpcCount=%d, prefill/encoding tpcCount=%d", decodeTpcCount,
        totalTpcCount - decodeTpcCount);
    return true;
}

std::future<LLMInferenceDisaggregationRuntime::AsyncRequestResult> LLMInferenceDisaggregationRuntime::submitRequestAsync(
    LLMGenerationRequest const& request)
{
    auto context = std::make_shared<StageContext>();
    context->requestId = mRequestCounter.fetch_add(1, std::memory_order_relaxed) + 1;
    context->request = &request;
    LOG_INFO("[Disagg][Req=%llu] submitted: batch=%zu", static_cast<unsigned long long>(context->requestId),
        request.requests.size());
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

bool LLMInferenceDisaggregationRuntime::setUpForPrefillExecution(StageContext& context, cudaStream_t stream)
{
    std::vector<std::vector<int32_t>> processedInputIds;
    std::vector<int32_t> processedIdsLengths;
    int32_t const activeBatchSize = static_cast<int32_t>(context.batchedInputIds.size());

    rt::LinearKVCache& linearKVCache = mLLMEngineRunner->getLinearKVCache();
    rt::Tensor kvCacheBuffer = linearKVCache.getKVCacheBuffer();

    check::check(
        context.hostReuseKVCacheLengths.reshape({activeBatchSize}), "Failed to reshape host reuse KV cache lengths.");
    int32_t* reuseKVCacheLengthsData = context.hostReuseKVCacheLengths.dataPointer<int32_t>();

    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        auto promptHash = hashSystemPromptWithLoraWeights(context.batchSystemPrompts[i], context.loraWeightsName);
        if (mSystemPromptKVCache.find(promptHash) != mSystemPromptKVCache.end())
        {
            auto& precachedKVCache = mSystemPromptKVCache[promptHash];
            auto const& kvCacheContent = precachedKVCache.kvCacheContent;
            kernel::instantiateKVCacheFromTensor(kvCacheBuffer, kvCacheContent, i, stream);
            int32_t reuseLength = static_cast<int32_t>(kvCacheContent.getShape()[3]);
            processedInputIds.emplace_back(
                context.batchedInputIds[i].begin() + reuseLength, context.batchedInputIds[i].end());
            processedIdsLengths.emplace_back(static_cast<int32_t>(context.batchedInputIds[i].size() - reuseLength));
            reuseKVCacheLengthsData[i] = reuseLength;
        }
        else
        {
            processedInputIds.emplace_back(context.batchedInputIds[i]);
            processedIdsLengths.emplace_back(static_cast<int32_t>(context.batchedInputIds[i].size()));
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
    check::check(
        context.hostPackedInputIds.reshape({activeBatchSize, packedInputLength}), "Failed to reshape packed input IDs.");
    int32_t* packedInputIdsData = context.hostPackedInputIds.dataPointer<int32_t>();
    std::fill(packedInputIdsData, packedInputIdsData + activeBatchSize * packedInputLength, mTokenizer->getPadId());
    for (int32_t i = 0; i < activeBatchSize; ++i)
    {
        std::copy(processedInputIds[i].begin(), processedInputIds[i].end(), packedInputIdsData + i * packedInputLength);
    }

    linearKVCache.resetForNewSequences(context.hostReuseKVCacheLengths, context.slotOffset, stream);
    check::check(context.inputIds.reshape({activeBatchSize, packedInputLength}), "Failed to reshape input IDs.");
    check::check(context.hostContextLengths.reshape({activeBatchSize}), "Failed to reshape context lengths.");
    check::check(
        context.outputLogits.reshape({activeBatchSize, mEngineConfig.outputVocabSize}), "Failed to reshape output logits.");

    CUDA_CHECK(cudaMemcpyAsync(context.inputIds.rawPointer(), context.hostPackedInputIds.rawPointer(),
        activeBatchSize * packedInputLength * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    memcpy(context.hostContextLengths.dataPointer<int32_t>(), processedIdsLengths.data(), activeBatchSize * sizeof(int32_t));

    if (mEngineConfig.maxSupportedLoraRank > 0
        && mLLMEngineRunner->getActiveLoraWeightsName() != context.loraWeightsName
        && !mLLMEngineRunner->switchLoraWeights(context.loraWeightsName, stream))
    {
        LOG_ERROR("Failed to switch LoRA weights to %s", context.loraWeightsName.c_str());
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
    trt_edgellm::topKtopPSamplingFromLogits(
        context.outputLogits, context.selectedIndices, params, context.samplingWorkspace, stream);
    if (mEngineConfig.reducedVocabSize > 0)
    {
        trt_edgellm::mapReducedVocabToFullVocab(context.selectedIndices, mVocabMappingTable, stream);
    }
    CUDA_CHECK(cudaMemcpyAsync(context.hostSelectedTokenIds.rawPointer(), context.selectedIndices.rawPointer(),
        context.activeBatchSize * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int32_t* hostSelectedTokenIdsData = context.hostSelectedTokenIds.dataPointer<int32_t>();
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

int32_t LLMInferenceDisaggregationRuntime::allocateSlotRange(int32_t slotCount)
{
    check::check(slotCount > 0 && slotCount <= mEngineConfig.maxSupportedBatchSize, "Invalid slotCount.");
    std::unique_lock<std::mutex> lock(mSlotAllocatorMutex);
    auto hasRange = [&]() {
        int32_t consecutive = 0;
        for (bool used : mSlotUsage)
        {
            consecutive = used ? 0 : consecutive + 1;
            if (consecutive >= slotCount)
            {
                return true;
            }
        }
        return false;
    };
    mSlotAllocatorCv.wait(lock, hasRange);
    int32_t consecutive = 0;
    for (int32_t i = 0; i < mEngineConfig.maxSupportedBatchSize; ++i)
    {
        consecutive = mSlotUsage[static_cast<size_t>(i)] ? 0 : consecutive + 1;
        if (consecutive >= slotCount)
        {
            int32_t const slotOffset = i - slotCount + 1;
            for (int32_t s = slotOffset; s < slotOffset + slotCount; ++s)
            {
                mSlotUsage[static_cast<size_t>(s)] = true;
            }
            return slotOffset;
        }
    }
    return 0;
}

void LLMInferenceDisaggregationRuntime::releaseSlotRange(int32_t slotOffset, int32_t slotCount)
{
    if (slotCount <= 0)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mSlotAllocatorMutex);
    for (int32_t s = slotOffset; s < slotOffset + slotCount; ++s)
    {
        if (s >= 0 && s < mEngineConfig.maxSupportedBatchSize)
        {
            mSlotUsage[static_cast<size_t>(s)] = false;
        }
    }
    mSlotAllocatorCv.notify_all();
}

void LLMInferenceDisaggregationRuntime::multimodalWorkerMain()
{
    std::shared_ptr<StageContext> context;
    while (mMultimodalQueue.pop(context))
    {
        try
        {
        if (!examineRequest(*context->request))
        {
            setFailedResult(*context, "request validation failed");
            finalizeContext(*context);
            continue;
        }

        context->activeBatchSize = static_cast<int32_t>(context->request->requests.size());
        context->slotOffset = allocateSlotRange(context->activeBatchSize);
        context->loraWeightsName = context->request->loraWeightsName;
        LOG_INFO("[Disagg][Req=%llu] encoding(start): batch=%d slotOffset=%d",
            static_cast<unsigned long long>(context->requestId), context->activeBatchSize, context->slotOffset);
        context->samplingWorkspace = rt::Tensor({mMaxSamplingWorkspaceSize}, rt::DeviceType::kGPU, DataType::kINT8,
            "DisaggRuntime::ctxSamplingWorkspace");
        context->inputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kGPU, DataType::kINT32, "DisaggRuntime::ctxInputIds");
        context->hostPackedInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
            rt::DeviceType::kCPU, DataType::kINT32, "DisaggRuntime::ctxHostPackedInputIds");
        context->hostContextLengths = rt::Tensor(
            {mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32, "DisaggRuntime::ctxContextLen");
        context->outputLogits = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize},
            rt::DeviceType::kGPU, DataType::kFLOAT, "DisaggRuntime::ctxOutputLogits");
        context->selectedIndices = rt::Tensor(
            {mEngineConfig.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU, DataType::kINT32, "DisaggRuntime::ctxSelected");
        context->hostSelectedTokenIds = rt::Tensor(
            {mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32, "DisaggRuntime::ctxHostSelected");
        context->hostReuseKVCacheLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            DataType::kINT32, "DisaggRuntime::ctxReuseKVCacheLengths");
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
        LOG_INFO("[Disagg][Req=%llu] encoding(end): reusedTokens=%d computedTokens=%d",
            static_cast<unsigned long long>(context->requestId), context->tokenCount.totalReusedTokens,
            context->tokenCount.totalComputedTokens);
        CUDA_CHECK(cudaEventCreateWithFlags(&context->multimodalDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(context->multimodalDone, mMultimodalStream));
        mPrefillQueue.push(std::move(context));
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("multimodalWorkerMain exception: %s", e.what());
            setFailedResult(*context, e.what());
            finalizeContext(*context);
        }
        catch (...)
        {
            LOG_ERROR("multimodalWorkerMain unknown exception.");
            setFailedResult(*context, "unknown exception in multimodal worker");
            finalizeContext(*context);
        }
    }
}

void LLMInferenceDisaggregationRuntime::prefillWorkerMain()
{
    std::shared_ptr<StageContext> context;
    while (mPrefillQueue.pop(context))
    {
        try
        {
        CUDA_CHECK(cudaStreamWaitEvent(mPrefillStream, context->multimodalDone, 0));

        if (!setUpForPrefillExecution(*context, mPrefillStream))
        {
            setFailedResult(*context, "prefill setup failed");
            finalizeContext(*context);
            continue;
        }

        int32_t const maxInputIdsLength = context->inputIds.getShape()[1];
        context->maxGenerationLength = context->request->maxGenerateLength;
        if (maxInputIdsLength + context->maxGenerationLength > mEngineConfig.maxKVCacheCapacity)
        {
            context->maxGenerationLength = mEngineConfig.maxKVCacheCapacity - maxInputIdsLength;
        }

        context->unFinishedBatchNum = context->activeBatchSize;
        context->generationIter = 0;
        context->outputIds.assign(context->activeBatchSize, std::vector<int32_t>{});
        context->finishedStates.assign(context->activeBatchSize, false);
        check::check(context->selectedIndices.reshape({context->activeBatchSize, 1}), "Failed to reshape selected indices.");
        check::check(
            context->hostSelectedTokenIds.reshape({context->activeBatchSize}), "Failed to reshape host selected token IDs.");

        rt::OptionalInputTensor multimodalEmbeddings
            = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;
        rt::OptionalInputTensors extraVisualFeatures
            = mMultimodalRunner ? mMultimodalRunner->getExtraVisualFeatures() : rt::OptionalInputTensors{};
        rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
        LOG_INFO("[Disagg][Req=%llu] prefill(start): batch=%d slotOffset=%d inputLen=%d maxGen=%d",
            static_cast<unsigned long long>(context->requestId), context->activeBatchSize, context->slotOffset,
            maxInputIdsLength, context->maxGenerationLength);
        {
            TIME_STAGE(metrics::StageNames::kLLM_PREFILL, mPrefillStream);
            std::lock_guard<std::mutex> runnerLock(mRunnerPrefillMutex);
            if (!mLLMEngineRunner->executePrefillStep(context->inputIds, context->hostContextLengths, multimodalEmbeddings,
                    extraVisualFeatures, context->outputLogits, outputHiddenStates, context->slotOffset, mPrefillStream))
            {
                setFailedResult(*context, "prefill execution failed");
                finalizeContext(*context);
                continue;
            }
            sampleTokens(*context, mPrefillStream);
        }
        LOG_INFO("[Disagg][Req=%llu] prefill(end): generationIter=%d unfinished=%d",
            static_cast<unsigned long long>(context->requestId), context->generationIter, context->unFinishedBatchNum);
        mPrefillMetrics.recordRun(context->tokenCount.totalReusedTokens, context->tokenCount.totalComputedTokens);

        check::check(context->inputIds.reshape({context->activeBatchSize, 1}), "Failed to reshape decode input IDs.");
        CUDA_CHECK(cudaEventCreateWithFlags(&context->prefillDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(context->prefillDone, mPrefillStream));
        mDecodeQueue.push(std::move(context));
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("prefillWorkerMain exception: %s", e.what());
            setFailedResult(*context, e.what());
            finalizeContext(*context);
        }
        catch (...)
        {
            LOG_ERROR("prefillWorkerMain unknown exception.");
            setFailedResult(*context, "unknown exception in prefill worker");
            finalizeContext(*context);
        }
    }
}

void LLMInferenceDisaggregationRuntime::decodeWorkerMain()
{
    std::shared_ptr<StageContext> context;
    while (mDecodeQueue.pop(context))
    {
        try
        {
        CUDA_CHECK(cudaStreamWaitEvent(mDecodeStream, context->prefillDone, 0));
        LOG_INFO("[Disagg][Req=%llu] decode(start): batch=%d slotOffset=%d maxGen=%d",
            static_cast<unsigned long long>(context->requestId), context->activeBatchSize, context->slotOffset,
            context->maxGenerationLength);
        {
            TIME_STAGE(metrics::StageNames::kLLM_GENERATION, mDecodeStream);
            while (context->unFinishedBatchNum > 0 && context->generationIter < context->maxGenerationLength)
            {
                std::lock_guard<std::mutex> runnerLock(mRunnerDecodeMutex);
                if (!mLLMEngineRunner->executeVanillaDecodingStep(
                        context->selectedIndices, context->outputLogits, context->slotOffset, mDecodeStream))
                {
                    setFailedResult(*context, "decode execution failed");
                    break;
                }
                sampleTokens(*context, mDecodeStream);
            }
        }
        LOG_INFO("[Disagg][Req=%llu] decode(end): generationIter=%d unfinished=%d status=%s",
            static_cast<unsigned long long>(context->requestId), context->generationIter, context->unFinishedBatchNum,
            context->errorMessage.empty() ? "ok" : "failed");

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
        catch (std::exception const& e)
        {
            LOG_ERROR("decodeWorkerMain exception: %s", e.what());
            setFailedResult(*context, e.what());
            finalizeContext(*context);
        }
        catch (...)
        {
            LOG_ERROR("decodeWorkerMain unknown exception.");
            setFailedResult(*context, "unknown exception in decode worker");
            finalizeContext(*context);
        }
    }
}

void LLMInferenceDisaggregationRuntime::finalizeContext(StageContext& context)
{
    bool expected = false;
    if (!context.completionSet.compare_exchange_strong(expected, true))
    {
        return;
    }

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
    try
    {
        context.completionPromise.set_value(std::move(result));
    }
    catch (std::exception const& e)
    {
        LOG_WARNING("finalizeContext promise set_value exception: %s", e.what());
    }
    releaseSlotRange(context.slotOffset, context.activeBatchSize);
}

bool LLMInferenceDisaggregationRuntime::captureDecodingCUDAGraph(cudaStream_t stream)
{
    (void) stream;
    LOG_WARNING("Disaggregation runtime disables decode CUDA graph for stability.");
    return false;
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
    StageContext cacheContext;
    cacheContext.activeBatchSize = 1;
    cacheContext.slotOffset = 0;
    cacheContext.batchedInputIds = batchedInputIds;
    cacheContext.batchSystemPrompts = batchedSystemPrompts;
    cacheContext.loraWeightsName = loraWeightsName;
    cacheContext.inputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
        rt::DeviceType::kGPU, DataType::kINT32, "DisaggRuntime::cacheInputIds");
    cacheContext.hostPackedInputIds = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.maxSupportedInputLength},
        rt::DeviceType::kCPU, DataType::kINT32, "DisaggRuntime::cacheHostPackedInputIds");
    cacheContext.hostContextLengths = rt::Tensor(
        {mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU, DataType::kINT32, "DisaggRuntime::cacheContextLen");
    cacheContext.outputLogits = rt::Tensor({mEngineConfig.maxSupportedBatchSize, mEngineConfig.outputVocabSize},
        rt::DeviceType::kGPU, DataType::kFLOAT, "DisaggRuntime::cacheOutputLogits");
    cacheContext.hostReuseKVCacheLengths = rt::Tensor({mEngineConfig.maxSupportedBatchSize}, rt::DeviceType::kCPU,
        DataType::kINT32, "DisaggRuntime::cacheReuseKVCacheLengths");

    if (!setUpForPrefillExecution(cacheContext, stream))
    {
        return false;
    }

    rt::OptionalInputTensor multimodalEmbeddings
        = mMultimodalRunner ? std::optional{std::ref(mMultimodalRunner->getOutputEmbedding())} : std::nullopt;
    rt::OptionalInputTensors extraVisualFeatures
        = mMultimodalRunner ? mMultimodalRunner->getExtraVisualFeatures() : rt::OptionalInputTensors{};
    rt::OptionalOutputTensor outputHiddenStates{std::nullopt};
    if (!mLLMEngineRunner->executePrefillStep(cacheContext.inputIds, cacheContext.hostContextLengths, multimodalEmbeddings,
            extraVisualFeatures, cacheContext.outputLogits, outputHiddenStates, cacheContext.slotOffset, stream))
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
