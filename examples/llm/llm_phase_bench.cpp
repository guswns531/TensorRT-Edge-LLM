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
#include "runtime/scheduling/phaseContinuousLoadGenerator.h"
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/scheduling/phaseGreedySampler.h"
#include "runtime/scheduling/phaseKernelGroupRecorder.h"
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
#include <map>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kPrefillProfile{0};
constexpr int32_t kDecodeProfile{1};
constexpr uint64_t kLoadFirstRequestId{10000};

struct Args
{
    std::string engineDir;
    std::string outputCsv;
    std::string loadCsv;
    std::string kernelGroupCsv;
    int32_t prefillBatch{1};
    int32_t decodeBatch{1};
    int32_t inputLen{512};
    int32_t prefillChunkSize{};
    int32_t pastKVLen{512};
    int32_t warmup{20};
    int32_t iterations{100};
    int32_t loadRequests{};
    double arrivalRate{1000.0};
    int32_t loadPromptMin{};
    int32_t loadPromptMax{};
    int32_t loadOutputMin{8};
    int32_t loadOutputMax{8};
    int32_t maxOverlapPrefillTokens{128};
    double ttftTargetMs{500.0};
    double tpotTargetMs{50.0};
    int32_t loadPriorityClasses{1};
    uint32_t loadSeed{};
    rt::PhaseTensorRTContextMode trtContextMode{rt::PhaseTensorRTContextMode::kIndependentConcurrent};
    bool contextAdapter{};
    bool adaptiveScheduler{};
    bool adaptiveChunking{};
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

struct LoadRequestSample
{
    uint64_t requestId{};
    int64_t scheduledArrivalUs{};
    int64_t submittedUs{-1};
    int64_t admittedUs{-1};
    int64_t firstTokenUs{-1};
    int64_t terminalUs{-1};
    int32_t promptTokens{};
    int32_t maxOutputTokens{};
    int32_t generatedTokens{};
    int32_t priority{};
    bool initiallyPending{};
};

void printUsage(char const* program)
{
    LOG_INFO(
        "Usage: %s --engineDir DIR [--prefillBatch N] [--decodeBatch N] [--inputLen N] "
        "[--prefillChunkSize N] [--pastKVLen N] [--warmup N] [--iterations N] "
        "[--trtContextMode shared|independent] "
        "[--contextAdapter] [--adaptiveScheduler] [--adaptiveChunking] [--outputCsv FILE] "
        "[--kernelGroupCsv FILE] "
        "[--loadRequests N --arrivalRate R --loadPromptMin N --loadPromptMax N "
        "--loadOutputMin N --loadOutputMax N --maxOverlapPrefillTokens N --ttftTargetMs F --tpotTargetMs F "
        "--loadPriorityClasses N --loadSeed N --loadCsv FILE]",
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
        kTensorRTContextMode,
        kLegacySharedContext,
        kContextAdapter,
        kAdaptiveScheduler,
        kAdaptiveChunking,
        kLoadRequests,
        kArrivalRate,
        kLoadPromptMin,
        kLoadPromptMax,
        kLoadOutputMin,
        kLoadOutputMax,
        kMaxOverlapPrefillTokens,
        kTtftTargetMs,
        kTpotTargetMs,
        kLoadPriorityClasses,
        kLoadSeed,
        kLoadCsv,
        kKernelGroupCsv,
        kHelp,
    };
    option const options[] = {{"engineDir", required_argument, nullptr, kEngineDir},
        {"prefillBatch", required_argument, nullptr, kPrefillBatch},
        {"decodeBatch", required_argument, nullptr, kDecodeBatch}, {"inputLen", required_argument, nullptr, kInputLen},
        {"prefillChunkSize", required_argument, nullptr, kPrefillChunkSize},
        {"pastKVLen", required_argument, nullptr, kPastKVLen}, {"warmup", required_argument, nullptr, kWarmup},
        {"iterations", required_argument, nullptr, kIterations}, {"outputCsv", required_argument, nullptr, kOutputCsv},
        {"trtContextMode", required_argument, nullptr, kTensorRTContextMode},
        {"sharedContext", no_argument, nullptr, kLegacySharedContext},
        {"contextAdapter", no_argument, nullptr, kContextAdapter},
        {"adaptiveScheduler", no_argument, nullptr, kAdaptiveScheduler},
        {"adaptiveChunking", no_argument, nullptr, kAdaptiveChunking},
        {"loadRequests", required_argument, nullptr, kLoadRequests},
        {"arrivalRate", required_argument, nullptr, kArrivalRate},
        {"loadPromptMin", required_argument, nullptr, kLoadPromptMin},
        {"loadPromptMax", required_argument, nullptr, kLoadPromptMax},
        {"loadOutputMin", required_argument, nullptr, kLoadOutputMin},
        {"loadOutputMax", required_argument, nullptr, kLoadOutputMax},
        {"maxOverlapPrefillTokens", required_argument, nullptr, kMaxOverlapPrefillTokens},
        {"ttftTargetMs", required_argument, nullptr, kTtftTargetMs},
        {"tpotTargetMs", required_argument, nullptr, kTpotTargetMs},
        {"loadPriorityClasses", required_argument, nullptr, kLoadPriorityClasses},
        {"loadSeed", required_argument, nullptr, kLoadSeed}, {"loadCsv", required_argument, nullptr, kLoadCsv},
        {"kernelGroupCsv", required_argument, nullptr, kKernelGroupCsv}, {"help", no_argument, nullptr, kHelp}, {}};

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
        case kTensorRTContextMode:
        {
            std::string const mode{optarg};
            if (mode == "shared")
            {
                args.trtContextMode = rt::PhaseTensorRTContextMode::kSharedSerialized;
            }
            else if (mode == "independent")
            {
                args.trtContextMode = rt::PhaseTensorRTContextMode::kIndependentConcurrent;
            }
            else
            {
                return false;
            }
            break;
        }
        case kLegacySharedContext: args.trtContextMode = rt::PhaseTensorRTContextMode::kSharedSerialized; break;
        case kContextAdapter: args.contextAdapter = true; break;
        case kAdaptiveScheduler: args.adaptiveScheduler = true; break;
        case kAdaptiveChunking: args.adaptiveChunking = true; break;
        case kLoadRequests: args.loadRequests = std::stoi(optarg); break;
        case kArrivalRate: args.arrivalRate = std::stod(optarg); break;
        case kLoadPromptMin: args.loadPromptMin = std::stoi(optarg); break;
        case kLoadPromptMax: args.loadPromptMax = std::stoi(optarg); break;
        case kLoadOutputMin: args.loadOutputMin = std::stoi(optarg); break;
        case kLoadOutputMax: args.loadOutputMax = std::stoi(optarg); break;
        case kMaxOverlapPrefillTokens: args.maxOverlapPrefillTokens = std::stoi(optarg); break;
        case kTtftTargetMs: args.ttftTargetMs = std::stod(optarg); break;
        case kTpotTargetMs: args.tpotTargetMs = std::stod(optarg); break;
        case kLoadPriorityClasses: args.loadPriorityClasses = std::stoi(optarg); break;
        case kLoadSeed: args.loadSeed = static_cast<uint32_t>(std::stoul(optarg)); break;
        case kLoadCsv: args.loadCsv = optarg; break;
        case kKernelGroupCsv: args.kernelGroupCsv = optarg; break;
        case kHelp: printUsage(argv[0]); return false;
        default: return false;
        }
    }
    return !args.engineDir.empty() && args.prefillBatch > 0 && args.decodeBatch > 0 && args.inputLen > 0
        && args.prefillChunkSize >= 0 && args.prefillChunkSize <= args.inputLen && args.pastKVLen >= 0
        && args.warmup >= 0 && args.iterations > 0 && args.loadRequests >= 0 && args.arrivalRate > 0.0
        && args.loadPromptMin >= 0 && args.loadPromptMax >= 0 && args.loadOutputMin > 0
        && args.loadOutputMin <= args.loadOutputMax && args.maxOverlapPrefillTokens >= 0 && args.ttftTargetMs > 0.0
        && args.tpotTargetMs > 0.0 && args.loadPriorityClasses > 0 && args.loadPriorityClasses <= 4;
}

bool usesSharedTensorRTContext(Args const& args) noexcept
{
    return args.trtContextMode == rt::PhaseTensorRTContextMode::kSharedSerialized;
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

void writeDispatchMetrics(std::filesystem::path const& path, std::vector<rt::PhaseDispatchMetrics> const& metrics)
{
    std::ofstream output(path);
    ELLM_CHECK(output.good(), "Failed to open dispatch metrics CSV: " + path.string());
    output << "dispatch_index,kind,prefill_batch,decode_batch,prefill_tokens,decode_tokens,"
              "prefill_queue_wait_us,decode_queue_wait_us,prefill_gpu_ms,decode_gpu_ms,makespan_gpu_ms,overlap_ratio\n";
    output << std::fixed << std::setprecision(6);
    for (rt::PhaseDispatchMetrics const& sample : metrics)
    {
        output << sample.dispatchIndex << ',' << static_cast<int32_t>(sample.kind) << ',' << sample.prefillBatchSize
               << ',' << sample.decodeBatchSize << ',' << sample.prefillTokens << ',' << sample.decodeTokens << ','
               << sample.prefillQueueWaitUs << ',' << sample.decodeQueueWaitUs << ',' << sample.prefillGpuMs << ','
               << sample.decodeGpuMs << ',' << sample.makespanGpuMs << ',' << sample.overlapRatio << '\n';
    }
}

void writeLoadMetrics(std::filesystem::path const& path, std::vector<LoadRequestSample> const& metrics)
{
    std::ofstream output(path);
    ELLM_CHECK(output.good(), "Failed to open request load metrics CSV: " + path.string());
    output << "request_id,scheduled_arrival_us,submitted_us,admitted_us,first_token_us,terminal_us,"
              "prompt_tokens,max_output_tokens,generated_tokens,priority,initial_admission,ttft_us,e2e_us,tpot_us\n";
    output << std::fixed << std::setprecision(6);
    for (LoadRequestSample const& sample : metrics)
    {
        int64_t const ttft = sample.firstTokenUs - sample.scheduledArrivalUs;
        int64_t const e2e = sample.terminalUs - sample.scheduledArrivalUs;
        double const tpot = sample.generatedTokens > 1
            ? static_cast<double>(sample.terminalUs - sample.firstTokenUs) / (sample.generatedTokens - 1)
            : 0.0;
        output << sample.requestId << ',' << sample.scheduledArrivalUs << ',' << sample.submittedUs << ','
               << sample.admittedUs << ',' << sample.firstTokenUs << ',' << sample.terminalUs << ','
               << sample.promptTokens << ',' << sample.maxOutputTokens << ',' << sample.generatedTokens << ','
               << sample.priority << ',' << (sample.initiallyPending ? "pending" : "admitted") << ',' << ttft << ','
               << e2e << ',' << tpot << '\n';
    }
}

void logDispatchHistogram(std::vector<rt::PhaseDispatchMetrics> const& metrics)
{
    std::map<std::pair<int32_t, int32_t>, size_t> histogram;
    for (rt::PhaseDispatchMetrics const& sample : metrics)
    {
        ++histogram[{sample.prefillBatchSize, sample.decodeBatchSize}];
    }
    for (auto const& [batchSizes, count] : histogram)
    {
        LOG_INFO("Continuous-load batch histogram: prefill=%d decode=%d count=%zu", batchSizes.first, batchSizes.second,
            count);
    }
}

int64_t elapsedMicroseconds(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();
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
    int32_t const loadPromptMin = args.loadPromptMin > 0 ? args.loadPromptMin : args.inputLen;
    int32_t const loadPromptMax = args.loadPromptMax > 0 ? args.loadPromptMax : args.inputLen;
    if (args.loadRequests > 0)
    {
        ELLM_CHECK(!args.loadCsv.empty(), "Continuous-load mode requires --loadCsv");
        ELLM_CHECK(loadPromptMin <= loadPromptMax, "Continuous-load prompt range is invalid");
        ELLM_CHECK(loadPromptMax <= args.inputLen,
            "Continuous-load prompt maximum exceeds the benchmark prefill buffer capacity");
        ELLM_CHECK(loadPromptMax + args.loadOutputMax <= config.maxKVCacheCapacity,
            "Continuous-load prompt plus output maximum exceeds KV capacity");
    }
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
    if (!usesSharedTensorRTContext(args))
    {
        decodeExecutor = prefillExecutor->createSibling();
        decodeRunner = decodeExecutor.get();
    }
    bool const sharedTensorRTContext
        = prefillExecutor->getExecutionContextIdentity() == decodeRunner->getExecutionContextIdentity();
    ELLM_CHECK(sharedTensorRTContext == usesSharedTensorRTContext(args),
        "TensorRT execution-context identity does not match --trtContextMode");
    CUcontext prefillCudaContext{};
    CUcontext decodeCudaContext{};
    CUDA_DRIVER_CHECK(cuStreamGetCtx(prefillStream, &prefillCudaContext));
    CUDA_DRIVER_CHECK(cuStreamGetCtx(decodeStream, &decodeCudaContext));
    ELLM_CHECK(prefillCudaContext != nullptr && prefillCudaContext == decodeCudaContext,
        "Phase streams must share one CUDA primary context");
    LOG_INFO("Phase execution topology: CUDA context=%p (shared), TensorRT prefill=%p, decode=%p (%s)",
        static_cast<void*>(prefillCudaContext),
        static_cast<void const*>(prefillExecutor->getExecutionContextIdentity()),
        static_cast<void const*>(decodeRunner->getExecutionContextIdentity()),
        sharedTensorRTContext ? "shared_serialized" : "independent_concurrent");
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

    std::unique_ptr<rt::Gemma4EmbeddingPreprocessor> prefillGemma4Ple;
    std::unique_ptr<rt::Gemma4EmbeddingPreprocessor> decodeGemma4PleOwner;
    rt::Gemma4EmbeddingPreprocessor* decodeGemma4Ple{};
    if (config.pleEnabled)
    {
        int32_t const maxPleSeqLen = std::max(args.inputLen, 1);
        prefillGemma4Ple = std::make_unique<rt::Gemma4EmbeddingPreprocessor>(
            engineDir, config, config.maxSupportedBatchSize, maxPleSeqLen, prefillMap, setupStream);
        if (usesSharedTensorRTContext(args))
        {
            prefillGemma4Ple->bindOutputs(decodeMap);
            decodeGemma4Ple = prefillGemma4Ple.get();
        }
        else
        {
            decodeGemma4PleOwner = prefillGemma4Ple->createSibling(config.maxSupportedBatchSize, 1, decodeMap);
            decodeGemma4Ple = decodeGemma4PleOwner.get();
            ELLM_CHECK(prefillGemma4Ple->tableDataIdentity() == decodeGemma4Ple->tableDataIdentity(),
                "Independent phase PLE preprocessors did not share the immutable table");
            ELLM_CHECK(prefillGemma4Ple->outputDataIdentity() != decodeGemma4Ple->outputDataIdentity(),
                "Independent phase PLE preprocessors aliased mutable outputs");
        }
        rt::Tensor prefillPleTokenIds({config.maxSupportedBatchSize, maxPleSeqLen}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "phase_prefill_ple_token_ids");
        rt::Tensor decodePleTokenIds({config.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "phase_decode_ple_token_ids");
        CUDA_CHECK(
            cudaMemsetAsync(prefillPleTokenIds.rawPointer(), 0, prefillPleTokenIds.getMemoryCapacity(), setupStream));
        CUDA_CHECK(
            cudaMemsetAsync(decodePleTokenIds.rawPointer(), 0, decodePleTokenIds.getMemoryCapacity(), setupStream));
        prefillGemma4Ple->embed(prefillPleTokenIds, setupStream);
        decodeGemma4Ple->embed(decodePleTokenIds, setupStream);
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
            if (prefillGemma4Ple)
            {
                prefillGemma4Ple->reshapeOutputs(args.prefillBatch, chunkLength);
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
        if (decodeGemma4Ple)
        {
            decodeGemma4Ple->reshapeOutputs(batchSize, 1);
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

    // Exercise actual source-context admission outside the fixed-shape timed samples.
    bool const continuousLoad = args.loadRequests > 0;
    if (continuousLoad || args.prefillBatch == args.decodeBatch)
    {
        std::optional<rt::PhaseContinuousLoadGenerator> loadGenerator;
        std::vector<rt::PhaseLoadRequest> servingRequests;
        if (continuousLoad)
        {
            loadGenerator.emplace(rt::PhaseContinuousLoadConfig{args.loadRequests, args.arrivalRate, loadPromptMin,
                loadPromptMax, args.loadOutputMin, args.loadOutputMax, args.loadSeed, kLoadFirstRequestId});
            servingRequests = loadGenerator->schedule();
        }
        else
        {
            servingRequests.reserve(args.prefillBatch);
            for (int32_t row = 0; row < args.prefillBatch; ++row)
            {
                servingRequests.push_back({static_cast<uint64_t>(1000 + row), 0, args.inputLen, phaseRounds + 1});
            }
        }

        std::vector<std::unique_ptr<rt::DecodingInferenceContext>> facadeContexts;
        facadeContexts.reserve(servingRequests.size());
        for (rt::PhaseLoadRequest const& request : servingRequests)
        {
            auto source = std::make_unique<rt::DecodingInferenceContext>();
            source->initialize(1, request.maxOutputTokens, std::nullopt, rt::OptionalInputTensors{}, "", decodeStream);
            source->rawBatchedInputIds = {std::vector<int32_t>(request.promptTokenCount, 0)};
            source->tokenIds = source->rawBatchedInputIds;
            source->effectivePrefillLengths = {request.promptTokenCount};
            source->currentGenerateLengths = {0};
            facadeContexts.push_back(std::move(source));
        }

        std::vector<int32_t> const servingEosTokenIds = continuousLoad ? std::vector<int32_t>{} : config.eosTokenIds;
        rt::PhaseGreedySampler prefillSampler(
            args.prefillBatch, config.outputVocabSize, servingEosTokenIds, "phase_serving_prefill_sampler");
        rt::PhaseGreedySampler decodeSampler(
            args.decodeBatch, config.outputVocabSize, servingEosTokenIds, "phase_serving_decode_sampler");

        std::vector<rt::PhaseDispatchMetrics> facadeDispatchMetrics;
        std::vector<LoadRequestSample> loadMetrics;
        std::unordered_map<uint64_t, size_t> loadMetricIndices;
        if (continuousLoad)
        {
            loadMetrics.reserve(servingRequests.size());
            for (size_t index = 0; index < servingRequests.size(); ++index)
            {
                rt::PhaseLoadRequest const& request = servingRequests[index];
                loadMetricIndices.emplace(request.requestId, loadMetrics.size());
                loadMetrics.push_back({request.requestId, request.arrivalOffsetUs, -1, -1, -1, -1,
                    request.promptTokenCount, request.maxOutputTokens, 0,
                    static_cast<int32_t>(index % static_cast<size_t>(args.loadPriorityClasses)), false});
            }
        }
        std::chrono::steady_clock::time_point loadStart;
        rt::PhaseQueueSchedulerConfig facadeSchedulerConfig;
        facadeSchedulerConfig.maxPrefillBatchSize = args.prefillBatch;
        facadeSchedulerConfig.maxDecodeBatchSize = args.decodeBatch;
        facadeSchedulerConfig.maxOverlapPrefillTokens = args.maxOverlapPrefillTokens;
        facadeSchedulerConfig.maxPrefillChunkTokens = configuredChunkSize;
        facadeSchedulerConfig.enableMetricsPolicy = args.adaptiveScheduler;
        facadeSchedulerConfig.prefillQueueWaitTargetUs = args.ttftTargetMs * 1000.0;
        facadeSchedulerConfig.decodeQueueWaitTargetUs = args.tpotTargetMs * 1000.0;
        facadeSchedulerConfig.enableAdaptivePrefillChunking = args.adaptiveChunking && configuredChunkSize > 0;
        facadeSchedulerConfig.minPrefillChunkTokens = std::min(32, std::max(1, configuredChunkSize));
        rt::PhaseKernelGroupRecorder kernelGroupRecorder;
        size_t kernelGroupDispatchIndex{};
        auto executeKernelSegments = [&](std::vector<rt::PhaseKernelSegment> const& segments) {
            if (args.kernelGroupCsv.empty())
            {
                for (rt::PhaseKernelSegment const& segment : segments)
                {
                    segment.enqueue(segment.stream);
                }
            }
            else
            {
                kernelGroupRecorder.execute(kernelGroupDispatchIndex++, segments);
            }
        };
        rt::PhaseContextServingCallbacks facadeCallbacks;
        facadeCallbacks.enqueuePackedPrefill = [&](rt::PhasePrefillContextBatchAdapter& packed) {
            int32_t const batchSize = packed.batchSize();
            int32_t const chunkLength = packed.chunkLength();
            bool const hasFinalPromptRow
                = std::any_of(packed.rows().begin(), packed.rows().end(), [](rt::PhasePrefillContextRow const& row) {
                      return row.tokenOffset + row.tokenCount == row.promptTokenCount;
                  });
            std::vector<rt::PhaseKernelSegment> segments;
            segments.push_back(
                {rt::PhaseKernelGroup::kPrefillPrepare, {}, packed.stream(), [&](cudaStream_t stream) {
                     check::check(prefillIO.inputsEmbeds.reshape({batchSize, chunkLength, config.hiddenSize}),
                         "Serving prefill input reshape failed");
                     check::check(prefillIO.contextLengths.reshape({batchSize}),
                         "Serving prefill context lengths reshape failed");
                     check::check(prefillIO.selectTokenIndices.reshape({batchSize, 1}),
                         "Serving prefill select indices reshape failed");
                     check::check(prefillIO.hostContextLengths.reshape({batchSize}),
                         "Serving host prefill lengths reshape failed");
                     check::check(prefillIO.hostSelectTokenIndices.reshape({batchSize, 1}),
                         "Serving host prefill indices reshape failed");
                     std::fill_n(prefillIO.hostContextLengths.dataPointer<int32_t>(), batchSize, chunkLength);
                     std::fill_n(prefillIO.hostSelectTokenIndices.dataPointer<int64_t>(), batchSize,
                         static_cast<int64_t>(chunkLength - 1));
                     CUDA_CHECK(cudaMemcpyAsync(prefillIO.contextLengths.rawPointer(),
                         prefillIO.hostContextLengths.rawPointer(), batchSize * sizeof(int32_t), cudaMemcpyHostToDevice,
                         stream));
                     CUDA_CHECK(cudaMemcpyAsync(prefillIO.selectTokenIndices.rawPointer(),
                         prefillIO.hostSelectTokenIndices.rawPointer(), batchSize * sizeof(int64_t),
                         cudaMemcpyHostToDevice, stream));
                     if (prefillGemma4Ple)
                     {
                         prefillGemma4Ple->embed(packed.tokenIds(), stream);
                         prefillGemma4Ple->reshapeOutputs(batchSize, chunkLength);
                     }
                     check::check(prefillIO.outputLogits.reshape({batchSize, config.outputVocabSize}),
                         "Serving prefill logits reshape failed");
                 }});
            segments.push_back({rt::PhaseKernelGroup::kPrefillEngine, {}, packed.stream(), [&](cudaStream_t stream) {
                                    auto const dims = config.prefillDims(batchSize, chunkLength, packed.initialChunk());
                                    ELLM_CHECK(prefillExecutor->prepare(kPrefillProfile, dims, prefillMap, stream)
                                            && prefillExecutor->execute(stream),
                                        "Serving facade packed prefill enqueue failed");
                                }});
            segments.push_back({rt::PhaseKernelGroup::kPrefillCacheCommit, {}, packed.stream(),
                [&](cudaStream_t stream) { packed.phaseBatchState().commit(cacheManager, chunkLength, stream); }});
            if (hasFinalPromptRow)
            {
                segments.push_back({rt::PhaseKernelGroup::kPrefillSample, {}, packed.stream(),
                    [&](cudaStream_t stream) { prefillSampler.enqueue(prefillIO.outputLogits, batchSize, stream); }});
            }
            executeKernelSegments(segments);
        };
        facadeCallbacks.completePackedPrefill = [&](rt::PhasePrefillContextBatchAdapter& packed) {
            if (prefillSampler.pending())
            {
                prefillSampler.completePrefill(packed);
                if (continuousLoad)
                {
                    int64_t const completionUs = elapsedMicroseconds(loadStart);
                    for (rt::PhasePrefillContextRow const& row : packed.rows())
                    {
                        if (row.tokenOffset + row.tokenCount == row.promptTokenCount)
                        {
                            LoadRequestSample& sample = loadMetrics.at(loadMetricIndices.at(row.requestId));
                            if (sample.firstTokenUs < 0)
                            {
                                sample.firstTokenUs = completionUs;
                            }
                        }
                    }
                }
            }
            kernelGroupRecorder.poll();
        };
        facadeCallbacks.completePrefill
            = [](rt::PhaseWorkItem const& item) { return item.tokenOffset + item.tokenCount; };
        facadeCallbacks.isPrefillFinished = [](uint64_t, rt::DecodingInferenceContext const& context, int32_t row) {
            return context.finishedStates[static_cast<size_t>(row)] != 0;
        };
        facadeCallbacks.enqueuePackedDecode = [&](rt::PhaseContextBatchAdapter& adapter) {
            rt::DecodingInferenceContext& packed = adapter.packedContext();
            ELLM_CHECK(packed.phaseBatchState != nullptr, "Serving facade decode has no phase batch state");
            ELLM_CHECK(decodeGemma4Ple != nullptr, "Serving facade actual token decode requires Gemma 4 PLE");
            executeKernelSegments({
                {rt::PhaseKernelGroup::kDecodePrepare, {}, packed.stream,
                    [&](cudaStream_t stream) {
                        decodeGemma4Ple->embed(adapter.tokenIds(), stream);
                        decodeGemma4Ple->reshapeOutputs(packed.activeBatchSize, 1);
                    }},
                {rt::PhaseKernelGroup::kDecodeEngine, {}, packed.stream,
                    [&](cudaStream_t stream) {
                        ELLM_CHECK(
                            executeDecode(*packed.phaseBatchState, 1, stream), "Serving facade decode enqueue failed");
                    }},
                {rt::PhaseKernelGroup::kDecodeSample, {}, packed.stream,
                    [&](cudaStream_t stream) {
                        decodeSampler.enqueue(decodeIO.outputLogits, packed.activeBatchSize, stream);
                    }},
            });
        };
        facadeCallbacks.completePackedDecode = [&](rt::PhaseContextBatchAdapter& adapter) {
            decodeSampler.completeDecode(adapter);
            kernelGroupRecorder.poll();
        };
        facadeCallbacks.isDecodeFinished = [](uint64_t, rt::DecodingInferenceContext const& context, int32_t row) {
            return context.finishedStates[static_cast<size_t>(row)] != 0;
        };
        facadeCallbacks.onDispatchMetrics
            = [&](rt::PhaseDispatchMetrics const& sample) { facadeDispatchMetrics.push_back(sample); };
        facadeCallbacks.onAdmission = [&](rt::PhaseAdmissionResult const& admission) {
            if (!continuousLoad)
            {
                return;
            }
            LoadRequestSample& sample = loadMetrics.at(loadMetricIndices.at(admission.requestId));
            if (admission.status == rt::PhaseAdmissionStatus::kPending)
            {
                sample.initiallyPending = true;
            }
            else if (sample.admittedUs < 0)
            {
                sample.admittedUs = elapsedMicroseconds(loadStart);
            }
        };
        facadeCallbacks.onTerminal = [&](rt::PhaseRequestSnapshot const& snapshot) {
            if (!continuousLoad)
            {
                return;
            }
            LoadRequestSample& sample = loadMetrics.at(loadMetricIndices.at(snapshot.requestId));
            size_t const index = loadMetricIndices.at(snapshot.requestId);
            sample.terminalUs = elapsedMicroseconds(loadStart);
            sample.generatedTokens = facadeContexts.at(index)->currentGenerateLengths[0];
        };
        auto const facadeMode = usesSharedTensorRTContext(args) ? rt::PhaseTensorRTContextMode::kSharedSerialized
                                                                : rt::PhaseTensorRTContextMode::kIndependentConcurrent;
        auto const facadeSafety = usesSharedTensorRTContext(args)
            ? rt::PhaseExecutionSafetyContract::shared(prefillExecutor->getExecutionContextIdentity())
            : rt::PhaseExecutionSafetyContract::independent(
                  {prefillExecutor->getExecutionContextIdentity(), &prefillContext, &prefillIO},
                  {decodeRunner->getExecutionContextIdentity(), decodeContext.get(), &decodeIO});
        rt::PhaseContextServingFacade facade(phaseSlotCount, facadeSchedulerConfig, std::move(facadeCallbacks),
            cacheManager, decodeMap, prefillStream, decodeStream, facadeMode, &prefillMap, configuredChunkSize,
            continuousLoad ? servingRequests.size() : 0, facadeSafety);
        if (continuousLoad)
        {
            ELLM_CHECK(enqueuePrefill(prefillBatch, prefillStream), "Continuous-load prefill warmup failed");
            CUDA_CHECK(cudaStreamSynchronize(prefillStream));
            ELLM_CHECK(enqueueDecode(decodeBatch, decodeStream, false), "Continuous-load decode warmup failed");
            CUDA_CHECK(cudaStreamSynchronize(decodeStream));

            rt::Tensor hostZeroLengths(
                {phaseSlotCount}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "load_zero_slot_lengths");
            std::fill_n(hostZeroLengths.dataPointer<int32_t>(), phaseSlotCount, 0);
            cacheManager.resetForNewSequences(hostZeroLengths, setupStream);
            CUDA_CHECK(cudaStreamSynchronize(setupStream));

            loadStart = std::chrono::steady_clock::now();
            size_t submittedRequests{};
            constexpr int64_t kLoadTimeoutUs = 300000000;
            while (!loadGenerator->done() || !facade.empty())
            {
                int64_t const elapsedUs = elapsedMicroseconds(loadStart);
                ELLM_CHECK(elapsedUs < kLoadTimeoutUs, "Continuous-load run exceeded its timeout");
                std::vector<rt::PhaseLoadRequest> const ready = loadGenerator->popReady(elapsedUs);
                bool madeProgress = !ready.empty();
                for (rt::PhaseLoadRequest const& request : ready)
                {
                    size_t const index = loadMetricIndices.at(request.requestId);
                    loadMetrics[index].submittedUs = elapsedMicroseconds(loadStart);
                    rt::PhaseSchedulingHints scheduling;
                    scheduling.priority = loadMetrics[index].priority;
                    scheduling.ttftTargetUs = args.ttftTargetMs * 1000.0;
                    scheduling.tpotTargetUs = args.tpotTargetMs * 1000.0;
                    static_cast<void>(facade.submitOrQueue(
                        request.requestId, *facadeContexts[index], 0, request.promptTokenCount, scheduling));
                    ++submittedRequests;
                }
                if (facade.busy())
                {
                    madeProgress = facade.poll() || madeProgress;
                }
                if (!facade.busy() && !facade.empty())
                {
                    ELLM_CHECK(facade.dispatchNext(), "Continuous-load facade failed to dispatch queued work");
                    madeProgress = true;
                }
                if (!madeProgress)
                {
                    std::this_thread::yield();
                }
            }
            ELLM_CHECK(submittedRequests == servingRequests.size(),
                "Continuous-load generator did not submit every scheduled request");
            for (LoadRequestSample const& sample : loadMetrics)
            {
                ELLM_CHECK(sample.submittedUs >= sample.scheduledArrivalUs && sample.admittedUs >= sample.submittedUs,
                    "Continuous-load admission timestamps are invalid");
                ELLM_CHECK(sample.firstTokenUs >= sample.admittedUs && sample.terminalUs >= sample.firstTokenUs,
                    "Continuous-load completion timestamps are invalid");
                ELLM_CHECK(sample.generatedTokens == sample.maxOutputTokens,
                    "Continuous-load request did not generate its configured token count");
            }
        }
        else
        {
            for (int32_t row = 0; row < args.prefillBatch; ++row)
            {
                rt::PhaseLoadRequest const& request = servingRequests[static_cast<size_t>(row)];
                int32_t const slot = facade.submit(
                    request.requestId, *facadeContexts[static_cast<size_t>(row)], 0, request.promptTokenCount);
                ELLM_CHECK(slot == row, "Serving facade did not preserve deterministic stable slot allocation");
            }
            facade.runUntilIdle(static_cast<size_t>(2 * phaseRounds + 2));
        }
        ELLM_CHECK(facade.empty(), "Serving facade engine smoke did not drain all phase queues");
        ELLM_CHECK(facade.availableSlotCount() == phaseSlotCount,
            "Serving facade engine smoke did not release all stable slots");
        ELLM_CHECK(facade.registeredRequestCount() == 0, "Serving facade retained source registrations");
        ELLM_CHECK(!facadeDispatchMetrics.empty(), "Serving facade emitted no dispatch metrics");
        std::filesystem::path dispatchCsv;
        if (continuousLoad)
        {
            writeLoadMetrics(args.loadCsv, loadMetrics);
            dispatchCsv = args.loadCsv;
            dispatchCsv.replace_filename(dispatchCsv.stem().string() + "-dispatch.csv");
            logDispatchHistogram(facadeDispatchMetrics);
            std::vector<float> ttftUs;
            std::vector<float> e2eUs;
            int64_t terminalUs{};
            int32_t generatedTokens{};
            for (LoadRequestSample const& sample : loadMetrics)
            {
                ttftUs.push_back(static_cast<float>(sample.firstTokenUs - sample.scheduledArrivalUs));
                e2eUs.push_back(static_cast<float>(sample.terminalUs - sample.scheduledArrivalUs));
                terminalUs = std::max(terminalUs, sample.terminalUs);
                generatedTokens += sample.generatedTokens;
            }
            LOG_INFO(
                "Continuous-load completed: requests=%zu rate=%.3f req/s pending_initial=%zu "
                "achieved=%.3f req/s %.3f token/s TTFT median/p95=%.3f/%.3f ms "
                "E2E median/p95=%.3f/%.3f ms",
                loadMetrics.size(), args.arrivalRate,
                static_cast<size_t>(std::count_if(loadMetrics.begin(), loadMetrics.end(),
                    [](LoadRequestSample const& sample) { return sample.initiallyPending; })),
                static_cast<double>(loadMetrics.size()) * 1000000.0 / terminalUs,
                static_cast<double>(generatedTokens) * 1000000.0 / terminalUs, percentile(ttftUs, 0.5F) / 1000.0F,
                percentile(ttftUs, 0.95F) / 1000.0F, percentile(e2eUs, 0.5F) / 1000.0F,
                percentile(e2eUs, 0.95F) / 1000.0F);
            LOG_INFO("Continuous-load request metrics written to %s", args.loadCsv.c_str());
        }
        else if (!args.outputCsv.empty())
        {
            dispatchCsv = args.outputCsv;
            dispatchCsv.replace_filename(dispatchCsv.stem().string() + "-dispatch.csv");
        }
        if (!dispatchCsv.empty())
        {
            writeDispatchMetrics(dispatchCsv, facadeDispatchMetrics);
            LOG_INFO("Serving dispatch metrics written to %s", dispatchCsv.c_str());
        }
        LOG_INFO("Serving facade scheduler policy: %s", args.adaptiveScheduler ? "adaptive_metrics" : "queue_default");
        LOG_INFO("Serving facade prefill chunk policy: %s", args.adaptiveChunking ? "adaptive_cuda_cost" : "fixed");
        LOG_INFO(
            "Serving facade engine run passed: %zu request context(s), stable admission -> actual greedy "
            "sampling -> repeated packed decode -> scatter -> slot release",
            servingRequests.size());
        if (!args.kernelGroupCsv.empty())
        {
            kernelGroupRecorder.drain();
            kernelGroupRecorder.writeCsv(args.kernelGroupCsv);
            LOG_INFO("Kernel-group CUDA-event samples written to %s", args.kernelGroupCsv.c_str());
        }
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
            auto const executionMode = usesSharedTensorRTContext(args)
                ? rt::PhaseTensorRTContextMode::kSharedSerialized
                : rt::PhaseTensorRTContextMode::kIndependentConcurrent;
            auto const safetyContract = usesSharedTensorRTContext(args)
                ? rt::PhaseExecutionSafetyContract::shared(prefillExecutor->getExecutionContextIdentity())
                : rt::PhaseExecutionSafetyContract::independent(
                      {prefillExecutor->getExecutionContextIdentity(), &prefillContext, &prefillIO},
                      {decodeRunner->getExecutionContextIdentity(), decodeContext.get(), &decodeIO});
            rt::PhaseDispatchWorker worker(
                scheduler, std::move(callbacks), prefillStream, decodeStream, executionMode, safetyContract);
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
            if (usesSharedTensorRTContext(args))
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
    char const* const scheduledName = usesSharedTensorRTContext(args) ? "Shared-TensorRT-context scheduled"
                                                                      : "Independent-TensorRT-context concurrent";
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
        char const* const csvMode
            = usesSharedTensorRTContext(args) ? "shared_trt_serialized" : "independent_trt_concurrent";
        writeCsv(args.outputCsv, samples, csvMode);
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
