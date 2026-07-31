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
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"

#include <algorithm>
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
};

struct Sample
{
    bool concurrent{};
    float makespanMs{};
    float prefillMs{};
    float decodeMs{};
};

void printUsage(char const* program)
{
    LOG_INFO(
        "Usage: %s --engineDir DIR [--prefillBatch N] [--decodeBatch N] [--inputLen N] "
        "[--prefillChunkSize N] [--pastKVLen N] [--warmup N] [--iterations N] [--outputCsv FILE]",
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
        kHelp,
    };
    option const options[] = {{"engineDir", required_argument, nullptr, kEngineDir},
        {"prefillBatch", required_argument, nullptr, kPrefillBatch},
        {"decodeBatch", required_argument, nullptr, kDecodeBatch}, {"inputLen", required_argument, nullptr, kInputLen},
        {"prefillChunkSize", required_argument, nullptr, kPrefillChunkSize},
        {"pastKVLen", required_argument, nullptr, kPastKVLen}, {"warmup", required_argument, nullptr, kWarmup},
        {"iterations", required_argument, nullptr, kIterations}, {"outputCsv", required_argument, nullptr, kOutputCsv},
        {"help", no_argument, nullptr, kHelp}, {}};

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
    for (Sample const& sample : samples)
    {
        if (sample.concurrent == concurrent)
        {
            makespans.push_back(sample.makespanMs);
            prefill.push_back(sample.prefillMs);
            decode.push_back(sample.decodeMs);
        }
    }
    LOG_INFO("%s: makespan median=%.4f ms p95=%.4f ms, prefill median=%.4f ms, decode median=%.4f ms", name,
        percentile(makespans, 0.5F), percentile(makespans, 0.95F), percentile(prefill, 0.5F), percentile(decode, 0.5F));
}

void writeCsv(std::filesystem::path const& path, std::vector<Sample> const& samples)
{
    std::ofstream output(path);
    ELLM_CHECK(output.is_open(), "Failed to open phase benchmark CSV: " + path.string());
    output << "iteration,mode,makespan_ms,prefill_ms,decode_ms,overlap_fraction\n";
    int32_t sequentialIndex{};
    int32_t concurrentIndex{};
    output << std::fixed << std::setprecision(6);
    for (Sample const& sample : samples)
    {
        int32_t const iteration = sample.concurrent ? concurrentIndex++ : sequentialIndex++;
        float const phaseSum = sample.prefillMs + sample.decodeMs;
        float const overlap = phaseSum > 0.0F ? std::max(0.0F, 1.0F - sample.makespanMs / phaseSum) : 0.0F;
        output << iteration << ',' << (sample.concurrent ? "concurrent" : "sequential") << ',' << sample.makespanMs
               << ',' << sample.prefillMs << ',' << sample.decodeMs << ',' << overlap << '\n';
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
    auto decodeExecutor = prefillExecutor->createSibling();
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
    rt::Tensor decodeContext(
        {contextBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8, "phase_decode_context_memory");
    prefillExecutor->setContextMemory(prefillContext);
    decodeExecutor->setContextMemory(decodeContext);

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

    auto const decodeDims = config.decodeDims(args.decodeBatch);
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
    auto enqueueDecode = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream) {
        ELLM_CHECK(static_cast<int32_t>(batch.size()) == args.decodeBatch, "Unexpected decode batch size");
        decodeBatchState.prepare(batch, cacheManager, stream);
        CUDA_CHECK(cudaMemsetAsync(
            decodeIO.contextLengths.rawPointer(), 0, decodeIO.contextLengths.getMemoryCapacity(), stream));
        kernel::incrementLengthTensor(decodeIO.contextLengths, args.pastKVLen + 1, stream);
        if (gemma4Ple)
        {
            gemma4Ple->reshapeOutputs(args.decodeBatch, 1);
        }
        for (int32_t round = 0; round < phaseRounds; ++round)
        {
            if (!decodeExecutor->prepare(kDecodeProfile, decodeDims, decodeMap, stream)
                || !decodeExecutor->execute(stream))
            {
                return false;
            }
            decodeBatchState.commit(cacheManager, 1, stream);
            kernel::incrementLengthTensor(decodeIO.contextLengths, 1, stream);
        }
        return true;
    };

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
                ELLM_CHECK(enqueueDecode(batch, stream), "Decode enqueue failed");
                CUDA_CHECK(cudaEventRecord(decodeEnd, stream));
            };
            callbacks.completePrefill
                = [](rt::PhaseWorkItem const& item) { return item.tokenOffset + item.tokenCount; };
            callbacks.completeDecode
                = [](rt::PhaseWorkItem const& item) { return rt::PhaseDecodeCompletion{item.tokenCount + 1, true}; };
            rt::PhaseDispatchWorker worker(scheduler, std::move(callbacks), prefillStream, decodeStream);
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
            CUDA_CHECK(cudaEventRecord(decodeBegin, setupStream));
            ELLM_CHECK(enqueueDecode(decodeBatch, setupStream), "Sequential decode enqueue failed");
            CUDA_CHECK(cudaEventRecord(decodeEnd, setupStream));
        }
        CUDA_CHECK(cudaEventRecord(stop, setupStream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        Sample sample{concurrent};
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
    logSummary("Concurrent", samples, true);
    std::vector<float> sequentialMakespans;
    std::vector<float> concurrentMakespans;
    for (Sample const& sample : samples)
    {
        (sample.concurrent ? concurrentMakespans : sequentialMakespans).push_back(sample.makespanMs);
    }
    float const speedup = percentile(sequentialMakespans, 0.5F) / percentile(concurrentMakespans, 0.5F);
    LOG_INFO("Concurrent makespan speedup: %.4fx", speedup);
    if (!args.outputCsv.empty())
    {
        writeCsv(args.outputCsv, samples);
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
