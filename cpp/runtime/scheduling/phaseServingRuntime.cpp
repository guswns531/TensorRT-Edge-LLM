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

#include "runtime/scheduling/phaseServingRuntime.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "multimodal/common/multimodalRunner.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/phase/cost/phaseRuntimeCostTracker.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/scheduling/independentEngineExecutorPair.h"
#include "runtime/scheduling/independentPhaseCoordinator.h"
#include "runtime/scheduling/phaseVisionAdapter.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "runtime/state/stableKVPageManager.h"
#include "sampler/sampling.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <thread>
#include <utility>

namespace trt_edgellm::rt
{
namespace
{

class PhaseSamplingSlotPool
{
public:
    struct Slot
    {
        explicit Slot(int32_t maxRows)
            : hostIds({maxRows}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "phase_serving_host_sample_ids")
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
        }

        ~Slot() noexcept
        {
            if (ready != nullptr)
            {
                static_cast<void>(cudaEventDestroy(ready));
            }
        }

        Tensor hostIds;
        cudaEvent_t ready{};
        bool busy{};
    };

    explicit PhaseSamplingSlotPool(int32_t maxRows)
        : mMaxRows(maxRows)
    {
    }

    Slot& acquire()
    {
        for (auto& slot : mSlots)
        {
            if (!slot->busy)
            {
                slot->busy = true;
                return *slot;
            }
        }
        mSlots.push_back(std::make_unique<Slot>(mMaxRows));
        mSlots.back()->busy = true;
        return *mSlots.back();
    }

    void release(Slot& slot) noexcept
    {
        slot.busy = false;
    }

private:
    int32_t mMaxRows{};
    std::vector<std::unique_ptr<Slot>> mSlots;
};

PhaseQueueSchedulerConfig makeSchedulerConfig(PhaseServingRuntimeConfig const& serving, LLMEngineConfig const& engine)
{
    PhaseQueueSchedulerConfig config;
    config.policyMode = serving.policyMode;
    config.globalSchedulerMode = PhaseGlobalSchedulerMode::kActive;
    config.maxPrefillBatchSize = engine.maxSupportedPrefillBatchSize;
    config.maxExternalPrefillBatchSize = engine.maxSupportedVisionPrefillBatchSize;
    config.maxDecodeBatchSize = engine.maxSupportedDecodeBatchSize;
    int32_t const engineChunkLimit
        = engine.packedPrefill ? engine.maxPackedPrefillChunkTokens : engine.maxSupportedInputLength;
    config.maxPrefillChunkTokens = std::min(serving.maxPrefillChunkTokens, engineChunkLimit);
    config.maxOverlapPrefillTokens = config.maxPrefillChunkTokens;
    config.maxPrefillBatchTokens = serving.maxPrefillBatchTokens > 0
        ? serving.maxPrefillBatchTokens
        : config.maxPrefillBatchSize * config.maxPrefillChunkTokens;
    config.enableRaggedPrefillBatching = engine.packedPrefill;
    config.enablePackedPrefillTokenLayout = engine.packedPrefill;
    config.prefillCompletionBonusTokens = config.maxPrefillChunkTokens;
    config.enableWavefrontPrefillBatching = true;
    config.maxPrefillCohortSize = config.maxPrefillBatchSize;
    config.enableMetricsPolicy = true;
    config.elideVacuousGlobalDecisions = true;
    config.enableDecodeFormationHorizon = true;
    config.supportsChunkedPrefill = engine.packedPrefill;
    config.globalDispatchUsesPreReservedMemory = true;

    PhaseRuntimeCostTrackerConfig trackerConfig;
    trackerConfig.policyMode = serving.policyMode;
    trackerConfig.action = config.globalCostModelConfig;
    trackerConfig.decodeMinimumSamples = config.decodeComponentMinSamples;
    trackerConfig.decodeWindowSize = config.decodeComponentWindow;
    trackerConfig.decodeContextBucketTokens = config.runtimeDecodeContextBucketTokens;
    if (phasePolicyUsesContextualScalar(serving.policyMode))
    {
        trackerConfig.contextualPd.mode = PhaseContextualPdMode::kActive;
        trackerConfig.contextualEp.mode = PhaseContextualPdMode::kActive;
        trackerConfig.contextualEd.mode = PhaseContextualPdMode::kActive;
    }
    config.runtimeCostTracker = std::make_shared<PhaseRuntimeCostTracker>(std::move(trackerConfig));
    return config;
}

IndependentPhaseServerConfig makeServerConfig(
    PhaseServingRuntimeConfig const& serving, LLMEngineConfig const& engine, int32_t maxStableSlots)
{
    IndependentPhaseServerConfig config;
    config.maxInFlightRequests
        = serving.maxInFlightRequests > 0 ? serving.maxInFlightRequests : static_cast<size_t>(maxStableSlots);
    config.maxPendingRequests
        = serving.maxPendingRequests > 0 ? serving.maxPendingRequests : static_cast<size_t>(maxStableSlots);
    config.defaultMaxOutputTokens = 128;
    config.eosTokenIds = engine.eosTokenIds;
    config.enableCudaGraphs = serving.enableCudaGraphs;
    config.pageReservationMode = IndependentPhasePageReservationMode::kHeadroom;
    config.outputHeadroomTokens = 128;
    config.maxConcurrentPageGrowthRequests = std::max(1, engine.maxSupportedDecodeBatchSize);
    config.enableGlobalWaitActions = true;
    config.enableGlobalWaitAuthority = true;
    config.enableCompletionAwareAdmissionProjection = phasePolicyUsesTransition(serving.policyMode);
    config.allowChunkedVisionPrefill = serving.enableChunkedVisionPrefill && engine.packedPrefill;
    config.allowBatchedVisionPrefill = serving.enableBatchedVisionPrefill;
    config.releaseVisionPrefillStorage = serving.releaseVisionPrefillStorage;
    return config;
}

PhaseThreeCoordinatorConfig makeVisionConfig(PhaseServingRuntimeConfig const& serving, LLMEngineConfig const& engine,
    int32_t maxStableSlots, std::shared_ptr<PhaseRuntimeCostTracker> runtimeCostTracker)
{
    PhaseThreeCoordinatorConfig config;
    config.policyMode = serving.policyMode;
    config.globalSchedulerMode = PhaseGlobalSchedulerMode::kActive;
    config.runtimeCostTracker = std::move(runtimeCostTracker);
    config.enableGlobalEncoderPrefillAction = true;
    config.maxEncodedInFlight
        = serving.maxEncodedVisionRequests > 0 ? serving.maxEncodedVisionRequests : static_cast<size_t>(maxStableSlots);
    config.maxEncodedBytes = serving.maxEncodedVisionBytes;
    config.maxEncoderBatchSize = serving.maxEncoderBatchSize > 0
        ? serving.maxEncoderBatchSize
        : static_cast<size_t>(engine.maxSupportedPrefillBatchSize);
    config.maxEncoderBatchSize = std::min(config.maxEncoderBatchSize, config.maxEncodedInFlight);
    config.contextualPrefillBatchCapacity = engine.maxSupportedPrefillBatchSize;
    config.contextualDecodeBatchCapacity = engine.maxSupportedDecodeBatchSize;
    config.maxEncoderMediaItems = serving.maxEncoderMediaItems;
    config.maxEncoderInputBytes = serving.maxEncoderInputBytes;
    config.maxEncoderInputTokens = serving.maxEncoderInputTokens;
    config.encoderBatchWaitUs = serving.encoderBatchWaitUs;
    config.maxPrefillBatchSize
        = static_cast<size_t>(engine.hasVisionPrefillProfile() ? engine.maxSupportedVisionPrefillBatchSize
                                                               : engine.maxSupportedPrefillBatchSize);
    config.maxPrefillBatchSize = std::min(config.maxPrefillBatchSize, config.maxEncodedInFlight);
    config.maxPrefillBatchTokens = serving.maxPrefillBatchTokens > 0
        ? static_cast<size_t>(serving.maxPrefillBatchTokens)
        : static_cast<size_t>(engine.maxSupportedInputLength) * config.maxPrefillBatchSize;
    config.prefillBatchWaitUs = serving.visionPrefillBatchWaitUs;
    config.enableAdaptivePrefillAdmission = true;
    config.enablePrefixBeforeVisionPrefill = serving.enableVisionPrefixPrefill;
    config.enableAsyncEncoderPreparation = serving.enableAsyncEncoderPreparation;
    config.allowPdDispatchDuringEncoderPreparation = serving.allowPdDispatchDuringEncoderPreparation;
    config.separateEncoderPreparationCost = serving.separateEncoderPreparationCost;
    config.visionTtftTargetUs = serving.visionTtftTargetUs;
    config.visionTtftTargetExplicit = serving.visionTtftTargetExplicit;
    return config;
}

} // namespace

class PhaseServingRuntime::Impl
{
public:
    Impl(PhaseServingRuntimeConfig servingConfig, LLMEngineConfig const& engineConfig,
        std::unique_ptr<EngineExecutor> executor, SharedResources& resources, EmbeddingData const& embedding,
        cudaStream_t setupStream, std::unique_ptr<MultimodalRunner> visionRunner, tokenizer::Tokenizer const* tokenizer)
        : mServingConfig(std::move(servingConfig))
        , mEngineConfig(engineConfig)
        , mResources(resources)
        , mVisionRunner(std::move(visionRunner))
        , mEmbeddingPreprocessor(embedding, engineConfig)
        , mSamplingSlots(std::max(engineConfig.maxSupportedPrefillBatchSize, engineConfig.maxSupportedDecodeBatchSize))
    {
        ELLM_CHECK(executor != nullptr, "Phase serving requires a base executor");
        ELLM_CHECK(setupStream != nullptr, "Phase serving requires an explicit setup stream");
        ELLM_CHECK(engineConfig.kvPoolPages > 0, "Phase serving requires a paged-KV engine");
        ELLM_CHECK(!engineConfig.isSpecDecodeBase && !engineConfig.isDiffusionBackbone,
            "Phase serving supports vanilla autoregressive engines only");
        ELLM_CHECK(engineConfig.numLinearAttnLayers == 0, "Phase serving does not yet support recurrent layers");
        ELLM_CHECK(!engineConfig.pleEnabled, "Phase serving does not yet support PLE inputs");
        ELLM_CHECK(engineConfig.maxSupportedLoraRank == 0, "Phase serving does not yet support LoRA switching");
        ELLM_CHECK(mServingConfig.maxPrefillChunkTokens > 0, "Phase serving prefill chunk must be positive");

        CUDA_CHECK(cudaStreamCreateWithFlags(&mPrefillStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&mDecodeStream, cudaStreamNonBlocking));
        if (mVisionRunner != nullptr)
        {
            ELLM_CHECK(tokenizer != nullptr, "Phase vision serving requires a tokenizer");
            CUDA_CHECK(cudaStreamCreateWithFlags(&mEncoderStream, cudaStreamNonBlocking));
            CUDA_CHECK(cudaStreamCreateWithFlags(&mCopyStream, cudaStreamNonBlocking));
        }

        IndependentEngineExecutorPairConfig pairConfig;
        pairConfig.setupStream = setupStream;
        pairConfig.prefillStream = mPrefillStream;
        pairConfig.decodeStream = mDecodeStream;
        pairConfig.visionPrefillProfile = engineConfig.visionPrefillProfile;
        pairConfig.sharedExecutionContext = mServingConfig.sharedExecutionContext;
        pairConfig.dedicatedExternalPrefillContext
            = mVisionRunner != nullptr && engineConfig.packedPrefill && engineConfig.visionPrefillProfile < 0;
        mExecutors = IndependentEngineExecutorPair::create(std::move(executor), pairConfig);
        int32_t decodeBatchCapacity = engineConfig.maxSupportedDecodeBatchSize;
        size_t exclusiveEncoderInputTokenThreshold{};
        bool serializeAllEncoderPrefill{};
        bool serializeAllEncoderDecode{};
        if (mVisionRunner != nullptr)
        {
            int32_t const profileCount = mVisionRunner->getOptimizationProfileCount();
            ELLM_CHECK(profileCount > 0, "Phase vision serving requires at least one encoder profile");
            size_t freeBytes{};
            size_t totalBytes{};
            CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
            LOG_INFO("Phase workspace requirements: prefill=%zu decode=%zu vision=%lld profiles=%d free=%zu total=%zu",
                mExecutors->prefillContextMemory().getMemoryCapacity(),
                mExecutors->decodeContextMemory().getMemoryCapacity(),
                static_cast<long long>(mVisionRunner->getRequiredContextMemorySize()), profileCount, freeBytes,
                totalBytes);
            int64_t const prefillBytes = static_cast<int64_t>(mExecutors->prefillContextMemory().getMemoryCapacity());
            int64_t const visionBytes = mVisionRunner->getRequiredContextMemorySize();
            ELLM_CHECK(visionBytes > 0, "TensorRT returned an empty vision workspace for phase execution");
            constexpr size_t kWORKSPACE_HEADROOM_BYTES = 96U * 1024U * 1024U;
            size_t const independentVisionBytes = static_cast<size_t>(visionBytes);
            bool const independentVisionFits = freeBytes >= independentVisionBytes
                && freeBytes - independentVisionBytes >= kWORKSPACE_HEADROOM_BYTES;
            size_t const prefillShareGrowth = static_cast<size_t>(std::max<int64_t>(0, visionBytes - prefillBytes));
            bool const prefillShareFits
                = freeBytes >= prefillShareGrowth && freeBytes - prefillShareGrowth >= kWORKSPACE_HEADROOM_BYTES;
            if (independentVisionFits)
            {
                mVisionRunner->allocateContextMemory();
                LOG_INFO("Phase workspace mode: independent E/P/D arenas; all pairwise overlap remains available");
            }
            else if (!prefillShareFits)
            {
                TieredVisionContextMemoryInfo const info
                    = mExecutors->configureSharedVisionDecodeContextMemory(*mVisionRunner);
                serializeAllEncoderDecode = true;
                constexpr int32_t kMEMORY_CONSTRAINED_DECODE_BATCH = 32;
                decodeBatchCapacity = std::min(decodeBatchCapacity, kMEMORY_CONSTRAINED_DECODE_BATCH);
                LOG_INFO("Phase workspace mode: shared E/D arena=%lld bytes; E/P overlap remains available",
                    static_cast<long long>(info.arenaBytes));
                LOG_INFO("Phase memory-constrained decode capacity: %d rows", decodeBatchCapacity);
            }
            else if (profileCount == 1)
            {
                TieredVisionContextMemoryInfo const info
                    = mExecutors->configureSharedVisionContextMemory(*mVisionRunner, 0);
                serializeAllEncoderPrefill = true;
                LOG_INFO("Phase workspace mode: shared E/P arena=%lld bytes; E/D overlap remains available",
                    static_cast<long long>(info.arenaBytes));
            }
            else
            {
                TieredVisionContextMemoryInfo const info
                    = mExecutors->configureTieredVisionContextMemory(*mVisionRunner, 0, profileCount - 1);
                exclusiveEncoderInputTokenThreshold
                    = static_cast<size_t>(mVisionRunner->getInputTokenLimitForProfile(0));
                LOG_INFO("Phase workspace mode: tiered E/P arena=%lld bytes, exclusive input threshold=%zu",
                    static_cast<long long>(info.arenaBytes), exclusiveEncoderInputTokenThreshold);
            }
        }

        int32_t const prefillSequenceCapacity = engineConfig.packedPrefill
            ? std::min(mServingConfig.maxPrefillChunkTokens, engineConfig.maxPackedPrefillChunkTokens)
            : engineConfig.maxSupportedInputLength;
        mPrefillIO = std::make_unique<PipelineIO>(PipelineIO::createForLLMPhase(
            engineConfig, engineConfig.maxSupportedPrefillBatchSize, prefillSequenceCapacity, setupStream));
        mDecodeIO = std::make_unique<PipelineIO>(
            PipelineIO::createForLLMPhase(engineConfig, decodeBatchCapacity, 1, setupStream));
        buildTensorMap(mPrefillMap, *mPrefillIO, resources, engineConfig, 0);
        buildTensorMap(mDecodeMap, *mDecodeIO, resources, engineConfig, 0);
        if (!resources.externalWeightManager->validated())
        {
            resources.externalWeightManager->validateAgainstEngine(mExecutors->prefillExecutor(), "phase-base");
        }
        resources.externalWeightManager->registerTensorMapEntries({&mPrefillMap, &mDecodeMap});

        int32_t const maxPhaseBatch = std::max(engineConfig.maxSupportedPrefillBatchSize, decodeBatchCapacity);
        int32_t const maxStableSlots
            = mServingConfig.maxStableSlots > 0 ? mServingConfig.maxStableSlots : engineConfig.maxSupportedBatchSize;
        ELLM_CHECK(maxStableSlots >= maxPhaseBatch, "Phase stable-slot capacity must cover the largest phase batch");
        mOwnership = std::make_unique<StableKVPageManager>(StableKVPageManager::Config{
            maxStableSlots, maxPhaseBatch, engineConfig.kvPoolPages, engineConfig.maxKVCacheCapacity, 128});

        int32_t const prefillTokenCapacity = engineConfig.packedPrefill ? engineConfig.maxPackedPrefillChunkTokens
                                                                        : engineConfig.maxSupportedInputLength;
        mHostPrefillIds = Tensor({engineConfig.maxSupportedPrefillBatchSize, prefillTokenCapacity}, DeviceType::kCPU,
            nvinfer1::DataType::kINT32, "phase_serving_host_prefill_ids");
        mDevicePrefillIds = Tensor({engineConfig.maxSupportedPrefillBatchSize, prefillTokenCapacity}, DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "phase_serving_prefill_ids");
        mHostDecodeIds = Tensor(
            {decodeBatchCapacity, 1}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "phase_serving_host_decode_ids");
        mDeviceDecodeIds = Tensor(
            {decodeBatchCapacity, 1}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "phase_serving_decode_ids");
        mPrefillSelectedIds = Tensor({engineConfig.maxSupportedPrefillBatchSize, 1}, DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "phase_serving_prefill_selected_ids");
        mDecodeSelectedIds = Tensor({decodeBatchCapacity, 1}, DeviceType::kGPU, nvinfer1::DataType::kINT32,
            "phase_serving_decode_selected_ids");
        mPrefillCompactedLogits = Tensor({engineConfig.maxSupportedPrefillBatchSize, engineConfig.outputVocabSize},
            DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "phase_serving_prefill_compacted_logits");

        if (engineConfig.ropeConfig.type == RopeType::kMRope)
        {
            mTextMropeTemplate = Tensor({1, engineConfig.maxKVCacheCapacity, engineConfig.rotaryDim}, DeviceType::kGPU,
                nvinfer1::DataType::kFLOAT, "phase_serving_text_mrope_template");
            kernel::initializeTextOnlyMRopeCosSin(mTextMropeTemplate.dataPointer<float>(),
                engineConfig.ropeConfig.rotaryTheta, engineConfig.rotaryDim, engineConfig.maxKVCacheCapacity, 1,
                setupStream);
            mPrefillMropeOwners.resize(static_cast<size_t>(engineConfig.maxSupportedPrefillBatchSize));
            mDecodeMropeOwners.resize(static_cast<size_t>(decodeBatchCapacity));
            mPrefillMropeValid.resize(static_cast<size_t>(engineConfig.maxSupportedPrefillBatchSize));
            mDecodeMropeValid.resize(static_cast<size_t>(decodeBatchCapacity));
        }

        LLMEngineConfig phaseEngineConfig = engineConfig;
        phaseEngineConfig.maxSupportedDecodeBatchSize = decodeBatchCapacity;
        PhaseQueueSchedulerConfig schedulerConfig = makeSchedulerConfig(mServingConfig, phaseEngineConfig);
        if (schedulerConfig.maxExternalPrefillBatchSize <= 0)
        {
            schedulerConfig.maxExternalPrefillBatchSize = schedulerConfig.maxPrefillBatchSize;
        }
        std::shared_ptr<PhaseRuntimeCostTracker> runtimeCostTracker = schedulerConfig.runtimeCostTracker;
        schedulerConfig.globalMemoryHorizonSupplier = [this](
                                                          PhaseGlobalActionKey const&, std::vector<uint64_t> const&) {
            PhaseActionMemoryHorizon horizon;
            horizon.managedBytes = static_cast<size_t>(mOwnership->config().numPages - mOwnership->availablePages());
            horizon.budgetBytes = static_cast<size_t>(mOwnership->config().numPages);
            return horizon;
        };
        IndependentPhaseCoordinatorCallbacks seedCallbacks;
        seedCallbacks.isDecodeFinished = [](PhaseWorkItem const&, int32_t) { return true; };
        mCoordinator = std::make_unique<IndependentPhaseCoordinator>(phaseEngineConfig, std::move(schedulerConfig),
            *mExecutors, *mOwnership, *mPrefillIO, *mDecodeIO, mPrefillMap, mDecodeMap, mPrefillStream, mDecodeStream,
            std::move(seedCallbacks));
        mCoordinator->setGraphCaptureEnabled(mServingConfig.enableCudaGraphs);
        mCoordinator->setPersistentDecodeSelectEnabled(mServingConfig.enablePersistentDecodeSelect);
        mCoordinator->setPersistentPageBindingsEnabled(mServingConfig.enablePersistentPageBindings);

        IndependentPhaseRequestAdapter adapter;
        adapter.stagePrefill = [this](std::vector<IndependentPhaseRequestView> const& views, PipelineIO& io,
                                   TensorMap& map, cudaStream_t stream) { stageTokens(views, io, map, stream, true); };
        adapter.stageDecode = [this](std::vector<IndependentPhaseRequestView> const& views, PipelineIO& io,
                                  TensorMap& map, cudaStream_t stream) { stageTokens(views, io, map, stream, false); };
        adapter.submitSampling
            = [this](std::vector<IndependentPhaseRequestView> const& views, PipelineIO& io, cudaStream_t stream,
                  bool fromPrefill) { return submitSampling(views, io, stream, fromPrefill); };
        IndependentPhaseServerConfig serverConfig = makeServerConfig(mServingConfig, phaseEngineConfig, maxStableSlots);
        mServer = std::make_unique<IndependentPhaseAsyncServer>(
            std::move(serverConfig), *mCoordinator, *mOwnership, std::move(adapter));
        if (mVisionRunner != nullptr)
        {
            PhaseVisionStoragePolicy storagePolicy;
            storagePolicy.splitMropeLease = engineConfig.ropeConfig.type == RopeType::kMRope;
            mVisionAdapter = std::make_unique<PhaseVisionAdapter>(
                *mVisionRunner, *tokenizer, engineConfig, mEncoderStream, storagePolicy, mCopyStream);
            PhaseThreeCoordinatorConfig visionConfig
                = makeVisionConfig(mServingConfig, phaseEngineConfig, maxStableSlots, std::move(runtimeCostTracker));
            visionConfig.exclusiveEncoderInputTokenThreshold = exclusiveEncoderInputTokenThreshold;
            visionConfig.serializeAllEncoderPrefill = serializeAllEncoderPrefill;
            visionConfig.serializeAllEncoderDecode = serializeAllEncoderDecode;
            mThreePhase = std::make_unique<PhaseThreeCoordinator>(*mVisionAdapter, *mServer, std::move(visionConfig));
        }
        CUDA_CHECK(cudaStreamSynchronize(setupStream));
    }

    ~Impl() noexcept
    {
        if (mPrefillStream != nullptr)
        {
            static_cast<void>(cudaStreamSynchronize(mPrefillStream));
        }
        if (mDecodeStream != nullptr)
        {
            static_cast<void>(cudaStreamSynchronize(mDecodeStream));
        }
        if (mEncoderStream != nullptr)
        {
            static_cast<void>(cudaStreamSynchronize(mEncoderStream));
        }
        if (mCopyStream != nullptr)
        {
            static_cast<void>(cudaStreamSynchronize(mCopyStream));
        }
        mThreePhase.reset();
        mVisionAdapter.reset();
        mServer.reset();
        mCoordinator.reset();
        mOwnership.reset();
        mExecutors.reset();
        mPrefillIO.reset();
        mDecodeIO.reset();
        if (mPrefillStream != nullptr)
        {
            static_cast<void>(cudaStreamDestroy(mPrefillStream));
        }
        if (mDecodeStream != nullptr)
        {
            static_cast<void>(cudaStreamDestroy(mDecodeStream));
        }
        if (mEncoderStream != nullptr)
        {
            static_cast<void>(cudaStreamDestroy(mEncoderStream));
        }
        if (mCopyStream != nullptr)
        {
            static_cast<void>(cudaStreamDestroy(mCopyStream));
        }
    }

    void stageTokens(std::vector<IndependentPhaseRequestView> const& views, PipelineIO& io, TensorMap& map,
        cudaStream_t stream, bool prefill)
    {
        ELLM_CHECK(!views.empty(), "Phase token staging requires a non-empty batch");
        int32_t totalTokens{};
        for (IndependentPhaseRequestView const& view : views)
        {
            totalTokens += prefill ? view.work.tokenCount : 1;
        }
        Coords const tokenShape = prefill
            ? (mEngineConfig.packedPrefill ? Coords{1, totalTokens}
                                           : Coords{static_cast<int64_t>(views.size()), views.front().work.tokenCount})
            : Coords{static_cast<int64_t>(views.size()), 1};
        Tensor& hostIds = prefill ? mHostPrefillIds : mHostDecodeIds;
        Tensor& deviceIds = prefill ? mDevicePrefillIds : mDeviceDecodeIds;
        ELLM_CHECK(hostIds.reshape(tokenShape) && deviceIds.reshape(tokenShape), "Phase token staging reshape failed");

        int32_t* destination = hostIds.dataPointer<int32_t>();
        int32_t destinationOffset{};
        for (IndependentPhaseRequestView const& view : views)
        {
            if (prefill)
            {
                ELLM_CHECK(view.promptTokens != nullptr, "Phase prefill request has no prompt tokens");
                std::copy_n(view.promptTokens->begin() + view.work.tokenOffset, view.work.tokenCount,
                    destination + destinationOffset);
                destinationOffset += view.work.tokenCount;
            }
            else
            {
                ELLM_CHECK(view.generatedTokens != nullptr && !view.generatedTokens->empty(),
                    "Phase decode request has no sampled input token");
                destination[destinationOffset++] = view.generatedTokens->back();
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(deviceIds.rawPointer(), hostIds.rawPointer(),
            static_cast<size_t>(totalTokens) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));

        std::vector<Tensor> visionViews;
        std::vector<Tensor> deepstackViews;
        OptionalInputTensors visionSegments;
        std::vector<OptionalInputTensors> deepstackSegments(static_cast<size_t>(mEngineConfig.numDeepstackFeatures));
        size_t activeVisionRows{};
        for (IndependentPhaseRequestView const& view : views)
        {
            if (view.visionPayload != nullptr)
            {
                ++activeVisionRows;
            }
        }
        visionViews.reserve(activeVisionRows);
        deepstackViews.reserve(activeVisionRows * static_cast<size_t>(mEngineConfig.numDeepstackFeatures));
        visionSegments.reserve(activeVisionRows);
        for (auto& segments : deepstackSegments)
        {
            segments.reserve(activeVisionRows);
        }

        auto makeFeatureView = [](Tensor& feature, int64_t rowOffset, int64_t rows, std::string const& name) {
            Coords const shape = feature.getShape();
            ELLM_CHECK(shape.getNumDims() == 2 && rowOffset >= 0 && rows > 0 && rowOffset + rows <= shape[0],
                "Phase vision chunk is outside its request-owned feature buffer");
            size_t const rowBytes = static_cast<size_t>(shape[1]) * utils::getTypeSize(feature.getDataType());
            auto* const data = static_cast<std::byte*>(feature.rawPointer()) + rowOffset * rowBytes;
            return Tensor(data, {rows, shape[1]}, DeviceType::kGPU, feature.getDataType(), name);
        };
        for (IndependentPhaseRequestView const& view : views)
        {
            if (view.visionPayload == nullptr || !prefill)
            {
                continue;
            }
            auto const chunkBegin = view.promptTokens->begin() + view.work.tokenOffset;
            auto const chunkEnd = chunkBegin + view.work.tokenCount;
            int64_t const imageOffset = std::count(view.promptTokens->begin(), chunkBegin, mEngineConfig.imageTokenId);
            int64_t const imageRows = std::count(chunkBegin, chunkEnd, mEngineConfig.imageTokenId);
            if (imageRows == 0)
            {
                continue;
            }
            PhaseVisionPayload& payload = *view.visionPayload;
            visionViews.push_back(
                makeFeatureView(payload.outputEmbedding, imageOffset, imageRows, "phase_serving_vision_segment"));
            visionSegments.push_back(std::cref(visionViews.back()));
            ELLM_CHECK(payload.deepstackFeatures.size() == deepstackSegments.size(),
                "Phase vision request has the wrong deepstack feature count");
            for (size_t index{}; index < payload.deepstackFeatures.size(); ++index)
            {
                deepstackViews.push_back(makeFeatureView(
                    payload.deepstackFeatures[index], imageOffset, imageRows, "phase_serving_deepstack_segment"));
                deepstackSegments[index].push_back(std::cref(deepstackViews.back()));
            }
        }

        if (mEngineConfig.ropeConfig.type == RopeType::kMRope)
        {
            auto& owners = prefill ? mPrefillMropeOwners : mDecodeMropeOwners;
            auto& validPositions = prefill ? mPrefillMropeValid : mDecodeMropeValid;
            size_t const positionBytes
                = static_cast<size_t>(mEngineConfig.rotaryDim) * utils::getTypeSize(nvinfer1::DataType::kFLOAT);
            size_t const rowBytes = static_cast<size_t>(mEngineConfig.maxKVCacheCapacity) * positionBytes;
            constexpr int32_t kCopyGranularity = 128;
            for (size_t row{}; row < views.size(); ++row)
            {
                PhaseVisionPayload* const payload = views[row].visionPayload;
                std::optional<uint64_t> const desiredOwner = payload != nullptr && !payload->mropeCosSin.isEmpty()
                    ? std::optional<uint64_t>{views[row].requestId}
                    : std::nullopt;
                bool const ownerChanged = owners[row] != desiredOwner;
                int32_t const requiredPositions = prefill ? views[row].work.tokenOffset + views[row].work.tokenCount
                                                          : views[row].work.tokenCount + 1;
                PhaseMropeStagingRange const range = phaseMropeStagingRange(ownerChanged, validPositions[row],
                    requiredPositions, mEngineConfig.maxKVCacheCapacity, kCopyGranularity);
                owners[row] = desiredOwner;
                validPositions[row] = range.validPositions;
                if (range.countPositions == 0)
                {
                    continue;
                }
                Tensor const& source = desiredOwner.has_value() ? payload->mropeCosSin : mTextMropeTemplate;
                size_t const copyOffset = static_cast<size_t>(range.offsetPositions) * positionBytes;
                size_t const copyBytes = static_cast<size_t>(range.countPositions) * positionBytes;
                auto* const destination
                    = static_cast<std::byte*>(io.mropeCosSin.rawPointer()) + row * rowBytes + copyOffset;
                auto const* sourceBytes = static_cast<std::byte const*>(source.rawPointer()) + copyOffset;
                CUDA_CHECK(cudaMemcpyAsync(destination, sourceBytes, copyBytes, cudaMemcpyDeviceToDevice, stream));
            }
            map.set(binding_names::kRopeCosSin, io.mropeCosSin);
        }

        if (!visionSegments.empty())
        {
            mEmbeddingPreprocessor.embedSegmentedVision(deviceIds, visionSegments, io, stream);
            mEmbeddingPreprocessor.prepareSegmentedDeepstack(deviceIds, deepstackSegments, io, stream);
        }
        else
        {
            mEmbeddingPreprocessor.embed(deviceIds, std::nullopt, std::nullopt, io, stream);
            mEmbeddingPreprocessor.prepareDeepstack(deviceIds, {}, io, stream);
        }
        if (prefill)
        {
            for (int32_t index = 0; index < static_cast<int32_t>(io.deepstackEmbeds.size()); ++index)
            {
                map.set(binding_names::formatDeepstackEmbedsName(index), io.deepstackEmbeds[index]);
            }
        }
    }

    std::unique_ptr<IndependentPhaseSampleTicket> submitSampling(
        std::vector<IndependentPhaseRequestView> const& views, PipelineIO& io, cudaStream_t stream, bool fromPrefill)
    {
        int32_t const batchSize = static_cast<int32_t>(views.size());
        Tensor& selectedIds = fromPrefill ? mPrefillSelectedIds : mDecodeSelectedIds;
        PhaseSamplingSlotPool::Slot& slot = mSamplingSlots.acquire();
        ELLM_CHECK(io.outputLogits.reshape({batchSize, mEngineConfig.outputVocabSize})
                && selectedIds.reshape({batchSize, 1}) && slot.hostIds.reshape({batchSize}),
            "Phase sampling reshape failed");
        Tensor* logits = &io.outputLogits;
        bool const denseRows = std::all_of(views.begin(), views.end(),
            [row = size_t{}](auto const& view) mutable { return view.phaseBatchRow == row++; });
        if (fromPrefill && !denseRows)
        {
            ELLM_CHECK(mPrefillCompactedLogits.reshape({batchSize, mEngineConfig.outputVocabSize}),
                "Phase compacted-logits reshape failed");
            size_t const rowBytes = static_cast<size_t>(mEngineConfig.outputVocabSize) * sizeof(float);
            auto const* source = static_cast<std::byte const*>(io.outputLogits.rawPointer());
            auto* destination = static_cast<std::byte*>(mPrefillCompactedLogits.rawPointer());
            for (size_t row{}; row < views.size(); ++row)
            {
                CUDA_CHECK(cudaMemcpyAsync(destination + row * rowBytes, source + views[row].phaseBatchRow * rowBytes,
                    rowBytes, cudaMemcpyDeviceToDevice, stream));
            }
            logits = &mPrefillCompactedLogits;
        }
        selectArgmax(*logits, selectedIds, stream);
        CUDA_CHECK(cudaMemcpyAsync(slot.hostIds.rawPointer(), selectedIds.rawPointer(),
            static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(slot.ready, stream));

        auto ticket = std::make_unique<IndependentPhaseSampleTicket>();
        ticket->ready = slot.ready;
        ticket->fromPrefill = fromPrefill;
        ticket->requestIds.reserve(views.size());
        for (IndependentPhaseRequestView const& view : views)
        {
            ticket->requestIds.push_back(view.requestId);
        }
        ticket->collect = [&slot, batchSize]() {
            int32_t const* selected = slot.hostIds.dataPointer<int32_t>();
            return std::vector<int32_t>(selected, selected + batchSize);
        };
        ticket->release = [this, &slot]() { mSamplingSlots.release(slot); };
        return ticket;
    }

    PhaseServingRuntimeConfig mServingConfig;
    LLMEngineConfig mEngineConfig;
    SharedResources& mResources;
    cudaStream_t mPrefillStream{};
    cudaStream_t mDecodeStream{};
    cudaStream_t mEncoderStream{};
    cudaStream_t mCopyStream{};
    std::unique_ptr<IndependentEngineExecutorPair> mExecutors;
    std::unique_ptr<MultimodalRunner> mVisionRunner;
    std::unique_ptr<PipelineIO> mPrefillIO;
    std::unique_ptr<PipelineIO> mDecodeIO;
    TensorMap mPrefillMap;
    TensorMap mDecodeMap;
    std::unique_ptr<StableKVPageManager> mOwnership;
    EmbeddingPreprocessor mEmbeddingPreprocessor;
    Tensor mHostPrefillIds;
    Tensor mDevicePrefillIds;
    Tensor mHostDecodeIds;
    Tensor mDeviceDecodeIds;
    Tensor mPrefillSelectedIds;
    Tensor mDecodeSelectedIds;
    Tensor mPrefillCompactedLogits;
    Tensor mTextMropeTemplate;
    std::vector<std::optional<uint64_t>> mPrefillMropeOwners;
    std::vector<std::optional<uint64_t>> mDecodeMropeOwners;
    std::vector<int32_t> mPrefillMropeValid;
    std::vector<int32_t> mDecodeMropeValid;
    PhaseSamplingSlotPool mSamplingSlots;
    std::unique_ptr<IndependentPhaseCoordinator> mCoordinator;
    std::unique_ptr<IndependentPhaseAsyncServer> mServer;
    std::unique_ptr<PhaseVisionAdapter> mVisionAdapter;
    std::unique_ptr<PhaseThreeCoordinator> mThreePhase;
};

std::unique_ptr<PhaseServingRuntime> PhaseServingRuntime::create(PhaseServingRuntimeConfig config,
    LLMEngineConfig const& engineConfig, std::unique_ptr<EngineExecutor> executor, SharedResources& resources,
    EmbeddingData const& embedding, cudaStream_t setupStream, std::unique_ptr<MultimodalRunner> visionRunner,
    tokenizer::Tokenizer const* tokenizer)
{
    return std::unique_ptr<PhaseServingRuntime>(new PhaseServingRuntime(std::make_unique<Impl>(std::move(config),
        engineConfig, std::move(executor), resources, embedding, setupStream, std::move(visionRunner), tokenizer)));
}

PhaseServingRuntime::PhaseServingRuntime(std::unique_ptr<Impl> impl)
    : mImpl(std::move(impl))
{
}

PhaseServingRuntime::~PhaseServingRuntime() noexcept = default;

IndependentPhaseServerSubmission PhaseServingRuntime::submit(
    uint64_t requestId, std::vector<int32_t> promptTokens, int32_t maxOutputTokens, PhaseSchedulingHints scheduling)
{
    return mImpl->mServer->submit(requestId, std::move(promptTokens), maxOutputTokens, scheduling);
}

IndependentPhaseServerSubmission PhaseServingRuntime::submitOrQueue(
    uint64_t requestId, std::vector<int32_t> promptTokens, int32_t maxOutputTokens, PhaseSchedulingHints scheduling)
{
    return mImpl->mServer->submitOrQueue(requestId, std::move(promptTokens), maxOutputTokens, scheduling);
}

PhaseThreeSubmissionStatus PhaseServingRuntime::submitVision(
    uint64_t requestId, LLMGenerationRequest request, int32_t maxOutputTokens, PhaseSchedulingHints scheduling)
{
    ELLM_CHECK(mImpl->mThreePhase != nullptr, "Phase vision serving is not enabled");
    return mImpl->mThreePhase->submit(requestId, std::move(request), maxOutputTokens, scheduling);
}

bool PhaseServingRuntime::cancel(uint64_t requestId)
{
    return mImpl->mThreePhase != nullptr ? mImpl->mThreePhase->cancel(requestId) : mImpl->mServer->cancel(requestId);
}

bool PhaseServingRuntime::poll()
{
    return mImpl->mThreePhase != nullptr ? mImpl->mThreePhase->poll() : mImpl->mServer->poll();
}

void PhaseServingRuntime::runUntilIdle(size_t maxPolls)
{
    if (mImpl->mThreePhase == nullptr)
    {
        mImpl->mServer->runUntilIdle(maxPolls);
        return;
    }
    for (size_t pollCount{}; pollCount < maxPolls && !mImpl->mThreePhase->empty(); ++pollCount)
    {
        if (!mImpl->mThreePhase->poll())
        {
            std::this_thread::yield();
        }
    }
    ELLM_CHECK(mImpl->mThreePhase->empty(), "Phase vision serving did not become idle within the poll limit");
}

std::optional<IndependentPhaseServerToken> PhaseServingRuntime::tryPopToken()
{
    return mImpl->mThreePhase != nullptr ? mImpl->mThreePhase->tryPopToken() : mImpl->mServer->tryPopToken();
}

std::optional<IndependentPhaseServerCompletion> PhaseServingRuntime::tryPopCompletion()
{
    return mImpl->mThreePhase != nullptr ? mImpl->mThreePhase->tryPopCompletion() : mImpl->mServer->tryPopCompletion();
}

bool PhaseServingRuntime::empty() const noexcept
{
    return mImpl->mThreePhase != nullptr ? mImpl->mThreePhase->empty() : mImpl->mServer->empty();
}

size_t PhaseServingRuntime::inFlightCount() const noexcept
{
    return mImpl->mServer->inFlightCount();
}

size_t PhaseServingRuntime::pendingCount() const noexcept
{
    return mImpl->mServer->pendingCount();
}

bool PhaseServingRuntime::visionEnabled() const noexcept
{
    return mImpl->mThreePhase != nullptr;
}

std::optional<PhaseThreeCoordinatorMetrics> PhaseServingRuntime::visionMetrics() const noexcept
{
    return mImpl->mThreePhase != nullptr ? std::optional{mImpl->mThreePhase->metrics()} : std::nullopt;
}

CUcontext PhaseServingRuntime::cudaContext() const noexcept
{
    return mImpl->mServer->cudaContext();
}

PhaseSchedulerTelemetry const& PhaseServingRuntime::schedulerTelemetry() const noexcept
{
    return mImpl->mServer->schedulerTelemetry();
}

} // namespace trt_edgellm::rt
