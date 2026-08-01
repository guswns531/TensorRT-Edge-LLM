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

#include "benchRunner.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "kernels/kvCacheUtilKernels/kvCacheUtilsKernels.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"
#include "runtime/scheduling/phaseBatchState.h"
#include "runtime/scheduling/phaseContextBatchAdapter.h"
#include "runtime/scheduling/phaseContextServingFacade.h"
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/scheduling/phaseGreedySampler.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};

struct Args
{
    std::string engineDir;
    std::string outputCsv;
    int32_t prefillBatch{1};
    int32_t decodeBatch{1};
    int32_t inputLen{512};
    int32_t prefillChunkSize{};
    int32_t pastKVLen{512};
    int32_t warmup{20};
    int32_t iterations{100};
    bool sharedContext{};
    bool contextAdapter{};
};

struct Sample
{
    bool concurrent{};
    float makespanMs{};
    float prefillMs{};
    float decodeMs{};
    float contextPackUs{};
    float contextScatterUs{};
};

void printUsage(char const* program)
{
    LOG_INFO(
        "Usage: %s --engineDir DIR [--prefillBatch N] [--decodeBatch N] [--inputLen N] "
        "[--prefillChunkSize N] [--pastKVLen N] [--warmup N] [--iterations N] [--sharedContext] "
        "[--contextAdapter] [--outputCsv FILE]",
        program);
}

bool parseArgs(Args& args, int argc, char** argv)
{
    enum OptionId : int
    {
        kEngineDir = 801,
        kPrefillBatch,
        kDecodeBatch,
        kInputLen,
        kPrefillChunkSize,
        kPastKVLen,
        kWarmup,
        kIterations,
        kOutputCsv,
        kSharedContext,
        kContextAdapter,
        kHelp,
    };
    option const options[] = {{"engineDir", required_argument, nullptr, kEngineDir},
        {"prefillBatch", required_argument, nullptr, kPrefillBatch},
        {"decodeBatch", required_argument, nullptr, kDecodeBatch}, {"inputLen", required_argument, nullptr, kInputLen},
        {"prefillChunkSize", required_argument, nullptr, kPrefillChunkSize},
        {"pastKVLen", required_argument, nullptr, kPastKVLen}, {"warmup", required_argument, nullptr, kWarmup},
        {"iterations", required_argument, nullptr, kIterations}, {"outputCsv", required_argument, nullptr, kOutputCsv},
        {"sharedContext", no_argument, nullptr, kSharedContext},
        {"contextAdapter", no_argument, nullptr, kContextAdapter}, {"help", no_argument, nullptr, kHelp}, {}};

    int optionId{};
    while ((optionId = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (optionId)
        {
        case kEngineDir: args.engineDir = optarg; break;
        case kPrefillBatch: args.prefillBatch = std::stoi(optarg); break;
        case kDecodeBatch: args.decodeBatch = std::stoi(optarg); break;
        case kInputLen: args.inputLen = std::stoi(optarg); break;
        case kPrefillChunkSize: args.prefillChunkSize = std::stoi(optarg); break;
        case kPastKVLen: args.pastKVLen = std::stoi(optarg); break;
        case kWarmup: args.warmup = std::stoi(optarg); break;
        case kIterations: args.iterations = std::stoi(optarg); break;
        case kOutputCsv: args.outputCsv = optarg; break;
        case kSharedContext: args.sharedContext = true; break;
        case kContextAdapter: args.contextAdapter = true; break;
        case kHelp: printUsage(argv[0]); return false;
        default: return false;
        }
    }
    return !args.engineDir.empty() && args.prefillBatch > 0 && args.decodeBatch > 0 && args.inputLen > 0
        && args.prefillChunkSize >= 0 && args.prefillChunkSize <= args.inputLen && args.pastKVLen >= 0
        && args.warmup >= 0 && args.iterations > 0;
}

void uploadInt32(rt::Tensor& tensor, std::vector<int32_t> const& values, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(
        tensor.rawPointer(), values.data(), values.size() * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
}

void uploadInt64(rt::Tensor& tensor, std::vector<int64_t> const& values, cudaStream_t stream)
{
    CUDA_CHECK(cudaMemcpyAsync(
        tensor.rawPointer(), values.data(), values.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream));
}

float percentile(std::vector<float> values, float quantile)
{
    std::sort(values.begin(), values.end());
    size_t const index = static_cast<size_t>(quantile * static_cast<float>(values.size() - 1));
    return values[index];
}

void logSummary(char const* name, std::vector<Sample> const& samples, bool concurrent)
{
    std::vector<float> makespans;
    std::vector<float> prefill;
    std::vector<float> decode;
    std::vector<float> contextPack;
    std::vector<float> contextScatter;
    for (Sample const& sample : samples)
    {
        if (sample.concurrent == concurrent)
        {
            makespans.push_back(sample.makespanMs);
            prefill.push_back(sample.prefillMs);
            decode.push_back(sample.decodeMs);
            contextPack.push_back(sample.contextPackUs);
            contextScatter.push_back(sample.contextScatterUs);
        }
    }
    LOG_INFO(
        "%s: makespan median=%.4f ms p95=%.4f ms, prefill median=%.4f ms, decode median=%.4f ms, "
        "context pack/scatter median=%.3f/%.3f us",
        name, percentile(makespans, 0.5F), percentile(makespans, 0.95F), percentile(prefill, 0.5F),
        percentile(decode, 0.5F), percentile(contextPack, 0.5F), percentile(contextScatter, 0.5F));
}

void writeCsv(std::filesystem::path const& path, std::vector<Sample> const& samples, std::string const& scheduledMode)
{
    std::ofstream output(path);
    ELLM_CHECK(output.is_open(), "Failed to open phase benchmark CSV: " + path.string());
    output << "iteration,mode,makespan_ms,prefill_ms,decode_ms,overlap_fraction,context_pack_us,context_scatter_us\n";
    int32_t sequentialIndex{};
    int32_t concurrentIndex{};
    output << std::fixed << std::setprecision(6);
    for (Sample const& sample : samples)
    {
        int32_t const iteration = sample.concurrent ? concurrentIndex++ : sequentialIndex++;
        float const phaseSum = sample.prefillMs + sample.decodeMs;
        float const overlap = phaseSum > 0.0F ? std::max(0.0F, 1.0F - sample.makespanMs / phaseSum) : 0.0F;
        output << iteration << ',' << (sample.concurrent ? scheduledMode : "sequential") << ',' << sample.makespanMs
               << ',' << sample.prefillMs << ',' << sample.decodeMs << ',' << overlap << ',' << sample.contextPackUs
               << ',' << sample.contextScatterUs << '\n';
    }
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parseArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    auto pluginHandles = loadEdgellmPluginLib();
    static_cast<void>(pluginHandles);
    std::filesystem::path const engineDir{args.engineDir};
    rt::DeploymentConfig deployment = rt::createDeploymentConfig(engineDir / "config.json", std::nullopt, std::nullopt);
    rt::LLMEngineConfig const& config = deployment.base;
    ELLM_CHECK(config.indexedKVCache, "llm_phase_bench requires an indexed_kv_cache engine");
    ELLM_CHECK(!deployment.draft.has_value(), "llm_phase_bench v1 supports vanilla inference only");
    ELLM_CHECK(config.numDeepstackFeatures == 0, "llm_phase_bench v1 does not support deepstack inputs");
    ELLM_CHECK(args.prefillBatch + args.decodeBatch <= config.maxSupportedBatchSize,
        "Prefill and decode physical slots exceed maxSupportedBatchSize");
    ELLM_CHECK(args.inputLen <= config.maxSupportedInputLength, "inputLen exceeds maxSupportedInputLength");
    int32_t const configuredChunkSize = args.prefillChunkSize > 0 ? args.prefillChunkSize : args.inputLen;
    int32_t const phaseRounds = (args.inputLen + configuredChunkSize - 1) / configuredChunkSize;
    ELLM_CHECK(args.pastKVLen + phaseRounds <= config.maxKVCacheCapacity, "pastKVLen exceeds KV capacity");
    LOG_INFO("Phase benchmark work per sample: %d prefill chunk(s), %d decode step(s)", phaseRounds, phaseRounds);

    cudaStream_t setupStream{};
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&setupStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    std::filesystem::path const enginePath = engineDir / "llm.engine";
    auto prefillExecutor = rt::EngineExecutor::createForLLM(enginePath, config);
    std::unique_ptr<rt::EngineExecutor> decodeExecutor;
    rt::EngineExecutor* decodeRunner = prefillExecutor.get();
    if (!args.sharedContext)
    {
        decodeExecutor = prefillExecutor->createSibling();
        decodeRunner = decodeExecutor.get();
    }
    rt::validateAgainstEngine(config, *prefillExecutor, "phase-prefill");

    std::unordered_map<std::string, std::string> const emptyLoraMap;
    auto resources = rt::SharedResources::createForLLM(config, emptyLoraMap, setupStream);
    auto prefillIO = rt::PipelineIO::createForLLM(config, setupStream);
    auto decodeIO = rt::PipelineIO::createForLLM(config, setupStream);
    rt::TensorMap prefillMap;
    rt::TensorMap decodeMap;
    rt::buildTensorMap(prefillMap, prefillIO, *resources, config, 0);
    rt::buildTensorMap(decodeMap, decodeIO, *resources, config, 0);

    resources->externalWeightManager->load(engineDir, engineDir / "config.json", setupStream);
    resources->externalWeightManager->validateAgainstEngine(*prefillExecutor, "phase-shared");
    resources->externalWeightManager->registerTensorMapEntries(prefillMap);
    resources->externalWeightManager->registerAdditionalTensorMapEntries(decodeMap);

    int64_t const contextBytes = prefillExecutor->getRequiredContextMemorySize();
    rt::Tensor prefillContext(
        {contextBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "phase_prefill_context_memory");
    prefillExecutor->setContextMemory(prefillContext);
    std::unique_ptr<rt::Tensor> decodeContext;
    if (decodeExecutor)
    {
        decodeContext = std::make_unique<rt::Tensor>(
            rt::Coords{contextBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "phase_decode_context_memory");
        decodeExecutor->setContextMemory(*decodeContext);
    }

    std::vector<int32_t> prefillSlots(args.prefillBatch);
    std::iota(prefillSlots.begin(), prefillSlots.end(), 0);
    std::vector<int32_t> decodeSlots(args.decodeBatch);
    std::iota(decodeSlots.begin(), decodeSlots.end(), args.prefillBatch);
    std::vector<rt::PhaseWorkItem> prefillBatch;
    std::vector<rt::PhaseWorkItem> decodeBatch;
    for (int32_t row = 0; row < args.prefillBatch; ++row)
    {
        prefillBatch.push_back({static_cast<uint64_t>(row), args.inputLen, prefillSlots[row], 0, args.inputLen});
    }
    for (int32_t row = 0; row < args.decodeBatch; ++row)
    {
        decodeBatch.push_back({static_cast<uint64_t>(args.prefillBatch + row), args.pastKVLen, decodeSlots[row]});
    }

    rt::PhaseBatchState prefillBatchState(args.prefillBatch, "phase_prefill");
    rt::PhaseBatchState decodeBatchState(args.decodeBatch, "phase_decode");
    prefillBatchState.bind(prefillMap);
    decodeBatchState.bind(decodeMap);

    int32_t const phaseSlotCount = args.prefillBatch + args.decodeBatch;
    std::vector<int32_t> initialSlotLengths(phaseSlotCount, 0);
    std::fill(initialSlotLengths.begin() + args.prefillBatch, initialSlotLengths.end(), args.pastKVLen);
    rt::Tensor hostInitialSlotLengths(
        {phaseSlotCount}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "phase_initial_slot_lengths");
    std::copy(initialSlotLengths.begin(), initialSlotLengths.end(), hostInitialSlotLengths.dataPointer<int32_t>());
    auto& cacheManager = *resources->cacheManagers[0];
    cacheManager.resetForNewSequences(hostInitialSlotLengths, setupStream);

    std::vector<std::unique_ptr<rt::DecodingInferenceContext>> decodeSourceContexts;
    std::vector<rt::PhaseContextRow> decodeContextRows;
    std::unique_ptr<rt::PhaseContextBatchAdapter> decodeContextAdapter;
    if (args.contextAdapter)
    {
        decodeSourceContexts.reserve(args.decodeBatch);
        decodeContextRows.reserve(args.decodeBatch);
        for (int32_t row = 0; row < args.decodeBatch; ++row)
        {
            auto source = std::make_unique<rt::DecodingInferenceContext>();
            source->initialize(1, phaseRounds + 1, std::nullopt, rt::OptionalInputTensors{}, "", decodeStream);
            source->rawBatchedInputIds = {{0}};
            source->tokenIds = {{0}};
            source->effectivePrefillLengths = {args.pastKVLen};
            decodeContextRows.push_back(
                {decodeBatch[row].requestId, source.get(), 0, decodeBatch[row].kvSlotId, args.pastKVLen});
            decodeSourceContexts.push_back(std::move(source));
        }
        decodeContextAdapter = std::make_unique<rt::PhaseContextBatchAdapter>(
            args.decodeBatch, cacheManager, decodeMap, "phase_decode_context_adapter");
    }

    uploadInt32(prefillIO.contextLengths, std::vector<int32_t>(args.prefillBatch, args.inputLen), setupStream);
    uploadInt32(decodeIO.contextLengths, std::vector<int32_t>(args.decodeBatch, args.pastKVLen + 1), setupStream);
    uploadInt64(prefillIO.selectTokenIndices, std::vector<int64_t>(args.prefillBatch, args.inputLen - 1), setupStream);
    uploadInt64(decodeIO.selectTokenIndices, std::vector<int64_t>(args.decodeBatch, 0), setupStream);
    fillRandomData(prefillIO.inputsEmbeds, -1.0F, 1.0F, nvinfer1::DataType::kHALF, 0);
    fillRandomData(decodeIO.inputsEmbeds, -1.0F, 1.0F, nvinfer1::DataType::kHALF, 1);

    std::unique_ptr<rt::Gemma4EmbeddingPreprocessor> gemma4Ple;
    if (config.pleEnabled)
    {
        int32_t const maxPleSeqLen = std::max(args.inputLen, 1);
        gemma4Ple = std::make_unique<rt::Gemma4EmbeddingPreprocessor>(
            engineDir, config, config.maxSupportedBatchSize, maxPleSeqLen, prefillMap, setupStream);
        gemma4Ple->bindOutputs(decodeMap);
        rt::Tensor pleTokenIds({config.maxSupportedBatchSize, maxPleSeqLen}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "phase_ple_token_ids");
        CUDA_CHECK(cudaMemsetAsync(pleTokenIds.rawPointer(), 0, pleTokenIds.getMemoryCapacity(), setupStream));
        gemma4Ple->embed(pleTokenIds, setupStream);
    }
    CUDA_CHECK(cudaStreamSynchronize(setupStream));

    auto enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream) {
        ELLM_CHECK(static_cast<int32_t>(batch.size()) == args.prefillBatch, "Unexpected prefill batch size");
        prefillBatchState.prepare(batch, cacheManager, stream);
        CUDA_CHECK(cudaMemsetAsync(
            prefillIO.selectTokenIndices.rawPointer(), 0, prefillIO.selectTokenIndices.getMemoryCapacity(), stream));
        for (int32_t chunkOffset = 0; chunkOffset < args.inputLen; chunkOffset += configuredChunkSize)
        {
            int32_t const chunkLength = std::min(configuredChunkSize, args.inputLen - chunkOffset);
            check::check(prefillIO.inputsEmbeds.reshape({args.prefillBatch, chunkLength, config.hiddenSize}),
                "Prefill input reshape failed");
            check::check(
                prefillIO.contextLengths.reshape({args.prefillBatch}), "Prefill context lengths reshape failed");
            CUDA_CHECK(cudaMemsetAsync(
                prefillIO.contextLengths.rawPointer(), 0, prefillIO.contextLengths.getMemoryCapacity(), stream));
            kernel::incrementLengthTensor(prefillIO.contextLengths, chunkLength, stream);
            if (gemma4Ple)
            {
                gemma4Ple->reshapeOutputs(args.prefillBatch, chunkLength);
            }
            bool const initialChunk = chunkOffset == 0;
            auto const prefillDims = config.prefillDims(args.prefillBatch, chunkLength, initialChunk);
            if (!prefillExecutor->prepare(kPrefillProfile, prefillDims, prefillMap, stream)
                || !prefillExecutor->execute(stream))
            {
                return false;
            }
            prefillBatchState.commit(cacheManager, chunkLength, stream);
        }
        return true;
    };
    auto executeDecode = [&](rt::PhaseBatchState& activeBatchState, int32_t rounds, cudaStream_t stream) {
        int32_t const batchSize = activeBatchState.lengths().getShape()[0];
        check::check(decodeIO.contextLengths.reshape({batchSize}), "Decode context lengths reshape failed");
        check::check(
            decodeIO.outputLogits.reshape({batchSize, config.outputVocabSize}), "Decode logits reshape failed");
        CUDA_CHECK(cudaMemcpyAsync(decodeIO.contextLengths.rawPointer(), activeBatchState.lengths().rawPointer(),
            batchSize * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream));
        kernel::incrementLengthTensor(decodeIO.contextLengths, 1, stream);
        if (gemma4Ple)
        {
            gemma4Ple->reshapeOutputs(batchSize, 1);
        }
        auto const decodeDims = config.decodeDims(batchSize);
        for (int32_t round = 0; round < rounds; ++round)
        {
            if (!decodeRunner->prepare(kDecodeProfile, decodeDims, decodeMap, stream) || !decodeRunner->execute(stream))
            {
                return false;
            }
            activeBatchState.commit(cacheManager, 1, stream);
            kernel::incrementLengthTensor(decodeIO.contextLengths, 1, stream);
        }
        return true;
    };
    float contextPackUs{};
    auto enqueueDecode = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream, bool useContextAdapter) {
        ELLM_CHECK(static_cast<int32_t>(batch.size()) == args.decodeBatch, "Unexpected decode batch size");
        int32_t const currentKVLength = batch.front().tokenCount;
        ELLM_CHECK(std::all_of(batch.begin(), batch.end(),
                       [currentKVLength](rt::PhaseWorkItem const& item) { return item.tokenCount == currentKVLength; }),
            "Phase benchmark decode batch must have uniform KV lengths");

        rt::PhaseBatchState* activeBatchState = &decodeBatchState;
        if (useContextAdapter)
        {
            ELLM_CHECK(decodeContextAdapter != nullptr, "Context adapter was not initialized");
            for (int32_t row = 0; row < args.decodeBatch; ++row)
            {
                decodeContextRows[row].requestId = batch[row].requestId;
                decodeContextRows[row].kvSlotId = batch[row].kvSlotId;
                decodeContextRows[row].kvLength = batch[row].tokenCount;
            }
            auto const packStart = std::chrono::steady_clock::now();
            decodeContextAdapter->packDecode(decodeContextRows, stream);
            auto const packEnd = std::chrono::steady_clock::now();
            contextPackUs = std::chrono::duration<float, std::micro>(packEnd - packStart).count();
            activeBatchState = decodeContextAdapter->packedContext().phaseBatchState;
        }
        else
        {
            contextPackUs = 0.0F;
            decodeBatchState.prepare(batch, cacheManager, stream);
        }
        return executeDecode(*activeBatchState, phaseRounds, stream);
    };

    // Exercise continuous source-context admission outside the timed samples.
    if (args.prefillBatch == args.decodeBatch)
    {
        std::vector<std::unique_ptr<rt::DecodingInferenceContext>> facadeContexts;
        facadeContexts.reserve(args.prefillBatch);
        for (int32_t row = 0; row < args.prefillBatch; ++row)
        {
            auto source = std::make_unique<rt::DecodingInferenceContext>();
            source->initialize(1, phaseRounds + 1, std::nullopt, rt::OptionalInputTensors{}, "", decodeStream);
            source->rawBatchedInputIds = {std::vector<int32_t>(args.inputLen, 0)};
            source->tokenIds = source->rawBatchedInputIds;
            source->effectivePrefillLengths = {args.inputLen};
            source->currentGenerateLengths = {0};
            facadeContexts.push_back(std::move(source));
        }

        rt::PhaseGreedySampler prefillSampler(
            args.prefillBatch, config.outputVocabSize, config.eosTokenIds, "phase_serving_prefill_sampler");
        rt::PhaseGreedySampler decodeSampler(
            args.decodeBatch, config.outputVocabSize, config.eosTokenIds, "phase_serving_decode_sampler");

        rt::PhaseQueueSchedulerConfig facadeSchedulerConfig;
        facadeSchedulerConfig.maxPrefillBatchSize = args.prefillBatch;
        facadeSchedulerConfig.maxDecodeBatchSize = args.decodeBatch;
        facadeSchedulerConfig.maxPrefillChunkTokens = configuredChunkSize;
        rt::PhaseContextServingCallbacks facadeCallbacks;
        facadeCallbacks.enqueuePackedPrefill = [&](rt::PhasePrefillContextBatchAdapter& packed) {
            int32_t const batchSize = packed.batchSize();
            int32_t const chunkLength = packed.chunkLength();
            check::check(prefillIO.inputsEmbeds.reshape({batchSize, chunkLength, config.hiddenSize}),
                "Serving prefill input reshape failed");
            check::check(
                prefillIO.contextLengths.reshape({batchSize}), "Serving prefill context lengths reshape failed");
            check::check(
                prefillIO.selectTokenIndices.reshape({batchSize, 1}), "Serving prefill select indices reshape failed");
            check::check(
                prefillIO.hostContextLengths.reshape({batchSize}), "Serving host prefill lengths reshape failed");
            check::check(prefillIO.hostSelectTokenIndices.reshape({batchSize, 1}),
                "Serving host prefill indices reshape failed");
            std::fill_n(prefillIO.hostContextLengths.dataPointer<int32_t>(), batchSize, chunkLength);
            std::fill_n(prefillIO.hostSelectTokenIndices.dataPointer<int64_t>(), batchSize,
                static_cast<int64_t>(chunkLength - 1));
            CUDA_CHECK(cudaMemcpyAsync(prefillIO.contextLengths.rawPointer(), prefillIO.hostContextLengths.rawPointer(),
                batchSize * sizeof(int32_t), cudaMemcpyHostToDevice, packed.stream()));
            CUDA_CHECK(cudaMemcpyAsync(prefillIO.selectTokenIndices.rawPointer(),
                prefillIO.hostSelectTokenIndices.rawPointer(), batchSize * sizeof(int64_t), cudaMemcpyHostToDevice,
                packed.stream()));
            if (gemma4Ple)
            {
                gemma4Ple->embed(packed.tokenIds(), packed.stream());
                gemma4Ple->reshapeOutputs(batchSize, chunkLength);
            }
            check::check(prefillIO.outputLogits.reshape({batchSize, config.outputVocabSize}),
                "Serving prefill logits reshape failed");
            auto const dims = config.prefillDims(batchSize, chunkLength, packed.initialChunk());
            ELLM_CHECK(prefillExecutor->prepare(kPrefillProfile, dims, prefillMap, packed.stream())
                    && prefillExecutor->execute(packed.stream()),
                "Serving facade packed prefill enqueue failed");
            packed.phaseBatchState().commit(cacheManager, chunkLength, packed.stream());
            bool const hasFinalPromptRow
                = std::any_of(packed.rows().begin(), packed.rows().end(), [](rt::PhasePrefillContextRow const& row) {
                      return row.tokenOffset + row.tokenCount == row.promptTokenCount;
                  });
            if (hasFinalPromptRow)
            {
                prefillSampler.enqueue(prefillIO.outputLogits, batchSize, packed.stream());
            }
        };
        facadeCallbacks.completePackedPrefill = [&](rt::PhasePrefillContextBatchAdapter& packed) {
            if (prefillSampler.pending())
            {
                prefillSampler.completePrefill(packed);
            }
        };
        facadeCallbacks.completePrefill
            = [](rt::PhaseWorkItem const& item) { return item.tokenOffset + item.tokenCount; };
        facadeCallbacks.isPrefillFinished = [](uint64_t, rt::DecodingInferenceContext const& context, int32_t row) {
            return context.finishedStates[static_cast<size_t>(row)] != 0;
        };
        facadeCallbacks.enqueuePackedDecode = [&](rt::PhaseContextBatchAdapter& adapter) {
            rt::DecodingInferenceContext& packed = adapter.packedContext();
            ELLM_CHECK(packed.phaseBatchState != nullptr, "Serving facade decode has no phase batch state");
            ELLM_CHECK(gemma4Ple != nullptr, "Serving facade actual token decode requires Gemma 4 PLE");
            gemma4Ple->embed(adapter.tokenIds(), packed.stream);
            gemma4Ple->reshapeOutputs(packed.activeBatchSize, 1);
            ELLM_CHECK(
                executeDecode(*packed.phaseBatchState, 1, packed.stream), "Serving facade decode enqueue failed");
            decodeSampler.enqueue(decodeIO.outputLogits, packed.activeBatchSize, packed.stream);
        };
        facadeCallbacks.completePackedDecode
            = [&](rt::PhaseContextBatchAdapter& adapter) { decodeSampler.completeDecode(adapter.packedContext()); };
        facadeCallbacks.isDecodeFinished = [](uint64_t, rt::DecodingInferenceContext const& context, int32_t row) {
            return context.finishedStates[static_cast<size_t>(row)] != 0;
        };
        auto const facadeMode = args.sharedContext ? rt::PhaseStreamExecutionMode::kSharedContextSerialized
                                                   : rt::PhaseStreamExecutionMode::kIndependentContextsConcurrent;
        rt::PhaseContextServingFacade facade(phaseSlotCount, facadeSchedulerConfig, std::move(facadeCallbacks),
            cacheManager, decodeMap, prefillStream, decodeStream, facadeMode, &prefillMap, configuredChunkSize);
        for (int32_t row = 0; row < args.prefillBatch; ++row)
        {
            uint64_t const requestId = static_cast<uint64_t>(1000 + row);
            int32_t const slot = facade.submit(requestId, *facadeContexts[static_cast<size_t>(row)], 0, args.inputLen);
            ELLM_CHECK(slot == row, "Serving facade did not preserve deterministic stable slot allocation");
        }
        facade.runUntilIdle(static_cast<size_t>(2 * phaseRounds + 2));
        ELLM_CHECK(facade.empty(), "Serving facade engine smoke did not drain all phase queues");
        ELLM_CHECK(facade.availableSlotCount() == phaseSlotCount,
            "Serving facade engine smoke did not release all stable slots");
        ELLM_CHECK(facade.registeredRequestCount() == 0, "Serving facade retained source registrations");
        LOG_INFO(
            "Serving facade engine smoke passed: %d request context(s), stable admission -> actual greedy sampling -> "
            "repeated packed decode -> scatter -> slot release",
            args.prefillBatch);
    }

    cudaEvent_t start{};
    cudaEvent_t prefillBegin{};
    cudaEvent_t prefillEnd{};
    cudaEvent_t decodeBegin{};
    cudaEvent_t decodeEnd{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&prefillBegin));
    CUDA_CHECK(cudaEventCreate(&prefillEnd));
    CUDA_CHECK(cudaEventCreate(&decodeBegin));
    CUDA_CHECK(cudaEventCreate(&decodeEnd));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto runOnce = [&](bool concurrent) {
        cacheManager.resetForNewSequences(hostInitialSlotLengths, setupStream);
        CUDA_CHECK(cudaEventRecord(start, setupStream));
        if (concurrent)
        {
            CUDA_CHECK(cudaStreamWaitEvent(prefillStream, start));
            rt::PhaseQueueSchedulerConfig schedulerConfig;
            schedulerConfig.maxPrefillBatchSize = args.prefillBatch;
            schedulerConfig.maxDecodeBatchSize = args.decodeBatch;
            schedulerConfig.policy = [](rt::PhaseQueueSnapshot const&) { return rt::PhaseDispatchKind::kOverlap; };
            rt::PhaseQueueScheduler scheduler(schedulerConfig);
            for (rt::PhaseWorkItem const& item : prefillBatch)
            {
                scheduler.enqueuePrefill(item);
            }
            for (rt::PhaseWorkItem const& item : decodeBatch)
            {
                scheduler.enqueueDecode(item);
            }

            rt::PhaseDispatchWorkerCallbacks callbacks;
            callbacks.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream) {
                CUDA_CHECK(cudaEventRecord(prefillBegin, stream));
                ELLM_CHECK(enqueuePrefill(batch, stream), "Prefill enqueue failed");
                CUDA_CHECK(cudaEventRecord(prefillEnd, stream));
            };
            callbacks.enqueueDecode = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream) {
                CUDA_CHECK(cudaEventRecord(decodeBegin, stream));
                ELLM_CHECK(enqueueDecode(batch, stream, args.contextAdapter), "Decode enqueue failed");
                CUDA_CHECK(cudaEventRecord(decodeEnd, stream));
            };
            callbacks.completePrefill = [](rt::PhaseWorkItem const& item) {
                return rt::PhasePrefillCompletion{item.tokenOffset + item.tokenCount, false};
            };
            callbacks.completeDecode = [phaseRounds](rt::PhaseWorkItem const& item) {
                return rt::PhaseDecodeCompletion{item.tokenCount + phaseRounds, true};
            };
            auto const executionMode = args.sharedContext
                ? rt::PhaseStreamExecutionMode::kSharedContextSerialized
                : rt::PhaseStreamExecutionMode::kIndependentContextsConcurrent;
            rt::PhaseDispatchWorker worker(scheduler, std::move(callbacks), prefillStream, decodeStream, executionMode);
            ELLM_CHECK(worker.dispatchNext(), "Phase worker failed to dispatch overlap plan");
            worker.wait();
            CUDA_CHECK(cudaStreamWaitEvent(setupStream, prefillEnd));
            CUDA_CHECK(cudaStreamWaitEvent(setupStream, decodeEnd));
        }
        else
        {
            CUDA_CHECK(cudaEventRecord(prefillBegin, setupStream));
            ELLM_CHECK(enqueuePrefill(prefillBatch, setupStream), "Sequential prefill enqueue failed");
            CUDA_CHECK(cudaEventRecord(prefillEnd, setupStream));
            if (args.sharedContext)
            {
                CUDA_CHECK(cudaEventSynchronize(prefillEnd));
            }
            CUDA_CHECK(cudaEventRecord(decodeBegin, setupStream));
            ELLM_CHECK(
                enqueueDecode(decodeBatch, setupStream, args.contextAdapter), "Sequential decode enqueue failed");
            CUDA_CHECK(cudaEventRecord(decodeEnd, setupStream));
        }
        CUDA_CHECK(cudaEventRecord(stop, setupStream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        Sample sample{concurrent};
        sample.contextPackUs = contextPackUs;
        if (args.contextAdapter)
        {
            auto const scatterStart = std::chrono::steady_clock::now();
            decodeContextAdapter->scatterDecode();
            auto const scatterEnd = std::chrono::steady_clock::now();
            sample.contextScatterUs = std::chrono::duration<float, std::micro>(scatterEnd - scatterStart).count();
        }
        CUDA_CHECK(cudaEventElapsedTime(&sample.makespanMs, start, stop));
        CUDA_CHECK(cudaEventElapsedTime(&sample.prefillMs, prefillBegin, prefillEnd));
        CUDA_CHECK(cudaEventElapsedTime(&sample.decodeMs, decodeBegin, decodeEnd));
        return sample;
    };

    for (int32_t i = 0; i < args.warmup; ++i)
    {
        static_cast<void>(runOnce(false));
        static_cast<void>(runOnce(true));
    }
    std::vector<Sample> samples;
    samples.reserve(static_cast<size_t>(args.iterations) * 2U);
    for (int32_t i = 0; i < args.iterations; ++i)
    {
        samples.push_back(runOnce((i % 2) != 0));
        samples.push_back(runOnce((i % 2) == 0));
    }

    logSummary("Sequential", samples, false);
    char const* const scheduledName
        = args.sharedContext ? "Shared-context scheduled" : "Independent-context concurrent";
    logSummary(scheduledName, samples, true);
    std::vector<float> sequentialMakespans;
    std::vector<float> concurrentMakespans;
    for (Sample const& sample : samples)
    {
        (sample.concurrent ? concurrentMakespans : sequentialMakespans).push_back(sample.makespanMs);
    }
    float const speedup = percentile(sequentialMakespans, 0.5F) / percentile(concurrentMakespans, 0.5F);
    LOG_INFO("%s makespan speedup: %.4fx", scheduledName, speedup);
    if (!args.outputCsv.empty())
    {
        writeCsv(args.outputCsv, samples, args.sharedContext ? "scheduled_shared" : "concurrent_independent");
        LOG_INFO("Raw CUDA-event samples written to %s", args.outputCsv.c_str());
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(prefillBegin));
    CUDA_CHECK(cudaEventDestroy(prefillEnd));
    CUDA_CHECK(cudaEventDestroy(decodeBegin));
    CUDA_CHECK(cudaEventDestroy(decodeEnd));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
    CUDA_CHECK(cudaStreamDestroy(setupStream));
    return EXIT_SUCCESS;
}
