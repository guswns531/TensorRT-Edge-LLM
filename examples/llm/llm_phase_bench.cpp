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
#include "multimodal/multimodalRunner.h"
#include "requestFileParser.h"
#include "runtime/config/deploymentConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/features/deepstackBinding.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"
#include "runtime/scheduling/gemma4PhaseVisionAdapter.h"
#include "runtime/scheduling/independentEngineExecutorPair.h"
#include "runtime/scheduling/modelPhaseContract.h"
#include "runtime/scheduling/phaseAsyncServer.h"
#include "runtime/scheduling/phaseBatchState.h"
#include "runtime/scheduling/phaseContextBatchAdapter.h"
#include "runtime/scheduling/phaseContextServingFacade.h"
#include "runtime/scheduling/phaseContinuousLoadGenerator.h"
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/scheduling/phaseGreedySampler.h"
#include "runtime/scheduling/phaseKernelGroupRecorder.h"
#include "runtime/scheduling/phaseThreeCoordinator.h"
#include "runtime/scheduling/qwen3VLPhaseVisionAdapter.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <getopt.h>
#include <iomanip>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
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
    std::string inputFile;
    std::string multimodalEngineDir;
    std::string traceCsv;
    std::string schedulerCostJson;
    int32_t prefillBatch{1};
    int32_t decodeBatch{1};
    int32_t slotCount{};
    int32_t inputLen{512};
    int32_t prefillChunkSize{};
    int32_t pastKVLen{512};
    int32_t warmup{20};
    int32_t iterations{100};
    int32_t loadRequests{};
    double arrivalRate{1000.0};
    double traceArrivalRate{10.0};
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
    bool cudaGraph{};
    int32_t maxCudaGraphs{128};
    int32_t maxPrefillCudaGraphs{-1};
    int32_t maxDecodeCudaGraphs{-1};
    int32_t maxCudaGraphMiB{};
    int32_t maxPrefillCudaGraphMiB{-1};
    int32_t maxDecodeCudaGraphMiB{-1};
    int32_t cudaGraphChargeMiB{4};
    int32_t cudaGraphReserveMiB{};
    int32_t prefillTokenBudget{};
    bool dynamicDecodeBatching{};
    rt::PhasePageReservationMode pageReservationMode{rt::PhasePageReservationMode::kFull};
    int32_t pageReservationHeadroomTokens{128};
    int32_t pageReservationOvercommitBundles{1};
    int32_t pageReservationGrowthRequests{8};
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

struct TraceRequestSample
{
    uint64_t requestId{};
    int64_t scheduledArrivalUs{};
    int64_t submittedUs{-1};
    int64_t admittedUs{-1};
    int64_t firstTokenUs{-1};
    int64_t completedUs{-1};
    int32_t promptTokens{};
    int32_t maxOutputTokens{};
    size_t imageCount{};
    rt::PhaseRequestStatus admissionStatus{rt::PhaseRequestStatus::kPending};
    int32_t admissionAvailableSlots{};
    size_t admissionPendingQueueDepth{};
    rt::KVPagePoolStats admissionPagePool;
    int32_t admissionReservedPageBundles{};
    int32_t admissionReservationAvailableBundles{};
    rt::PhaseAsyncCompletion completion;
};

struct TraceRequestMetadata
{
    int64_t arrivalOffsetUs{};
    std::optional<int32_t> maxGenerateLength;
    int32_t priority{};
};

std::vector<TraceRequestMetadata> readTraceMetadata(
    std::filesystem::path const& inputFile, size_t requestCount, double defaultArrivalRate)
{
    std::ifstream input(inputFile);
    ELLM_CHECK(input.is_open(), "Failed to open trace JSON: " + inputFile.string());
    nlohmann::json const root = nlohmann::json::parse(input);
    auto const& requests = root.at("requests");
    ELLM_CHECK(requests.size() == requestCount, "Trace parser and arrival metadata request counts differ");
    bool const hasExplicitOffsets = std::all_of(requests.begin(), requests.end(),
        [](nlohmann::json const& request) { return request.contains("arrival_offset_us"); });
    std::vector<TraceRequestMetadata> result(requestCount);
    int64_t const defaultIntervalUs = static_cast<int64_t>(1000000.0 / defaultArrivalRate);
    for (size_t index = 0; index < requestCount; ++index)
    {
        TraceRequestMetadata& metadata = result[index];
        metadata.arrivalOffsetUs = hasExplicitOffsets ? requests[index].at("arrival_offset_us").get<int64_t>()
                                                      : static_cast<int64_t>(index) * defaultIntervalUs;
        if (requests[index].contains("max_generate_length"))
        {
            metadata.maxGenerateLength = requests[index].at("max_generate_length").get<int32_t>();
            ELLM_CHECK(*metadata.maxGenerateLength > 0, "Trace request max_generate_length must be positive");
        }
        metadata.priority = requests[index].value("priority", 0);
        ELLM_CHECK(metadata.priority >= 0, "Trace request priority must be non-negative");
        ELLM_CHECK(metadata.arrivalOffsetUs >= 0
                && (index == 0 || metadata.arrivalOffsetUs >= result[index - 1].arrivalOffsetUs),
            "Trace arrival offsets must be non-negative and nondecreasing");
    }
    return result;
}

std::string csvQuote(std::string value)
{
    size_t offset{};
    while ((offset = value.find('"', offset)) != std::string::npos)
    {
        value.insert(offset, 1, '"');
        offset += 2;
    }
    return '"' + value + '"';
}

void writeTraceMetrics(std::filesystem::path const& path, std::vector<TraceRequestSample> const& samples)
{
    if (path.has_parent_path())
    {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream output(path);
    ELLM_CHECK(output.is_open(), "Failed to open phase trace CSV: " + path.string());
    output << "request_id,scheduled_arrival_us,submitted_us,admitted_us,first_token_us,completed_us,queue_delay_us,"
              "admission_delay_us,prompt_tokens,max_output_tokens,output_tokens,ttft_us,tpot_us,e2e_ms,image_count,"
              "admission_status,admission_available_slots,admission_pending_queue_depth,"
              "admission_page_pool_total_bundles,admission_page_pool_allocated_bundles,"
              "admission_page_pool_available_bundles,admission_page_pool_pressure,"
              "admission_reserved_page_bundles,admission_reservation_available_bundles,"
              "admission_reserved_page_pressure,"
              "finish_reason,output_text\n";
    output << std::fixed << std::setprecision(6);
    for (TraceRequestSample const& sample : samples)
    {
        rt::LLMGenerationResponse const& response = sample.completion.response;
        size_t const outputTokens = response.outputIds[0].size();
        int64_t const ttftUs = sample.firstTokenUs - sample.scheduledArrivalUs;
        double const tpotUs = outputTokens > 1
            ? static_cast<double>(sample.completedUs - sample.firstTokenUs) / static_cast<double>(outputTokens - 1)
            : 0.0;
        output << sample.requestId << ',' << sample.scheduledArrivalUs << ',' << sample.submittedUs << ','
               << sample.admittedUs << ',' << sample.firstTokenUs << ',' << sample.completedUs << ','
               << sample.submittedUs - sample.scheduledArrivalUs << ',' << sample.admittedUs - sample.submittedUs << ','
               << sample.promptTokens << ',' << sample.maxOutputTokens << ',' << outputTokens << ',' << ttftUs << ','
               << tpotUs << ',' << sample.completion.latencyMs << ',' << sample.imageCount << ','
               << static_cast<int>(sample.admissionStatus) << ',' << sample.admissionAvailableSlots << ','
               << sample.admissionPendingQueueDepth << ',' << sample.admissionPagePool.totalBundles << ','
               << sample.admissionPagePool.allocatedBundles << ',' << sample.admissionPagePool.availableBundles << ','
               << (sample.admissionPagePool.totalBundles > 0
                          ? static_cast<double>(sample.admissionPagePool.allocatedBundles)
                              / static_cast<double>(sample.admissionPagePool.totalBundles)
                          : 0.0)
               << ',' << sample.admissionReservedPageBundles << ',' << sample.admissionReservationAvailableBundles
               << ','
               << (sample.admissionPagePool.totalBundles > 0 ? static_cast<double>(sample.admissionReservedPageBundles)
                              / static_cast<double>(sample.admissionPagePool.totalBundles)
                                                             : 0.0)
               << ',' << rt::finishReasonName(response.finishReasons[0]) << ',' << csvQuote(response.outputTexts[0])
               << '\n';
    }
}

void printUsage(char const* program)
{
    LOG_INFO(
        "Usage: %s --engineDir DIR [--prefillBatch N] [--decodeBatch N] [--inputLen N] "
        "[--prefillChunkSize N] [--pastKVLen N] [--warmup N] [--iterations N] "
        "[--trtContextMode shared|independent] "
        "[--cudaGraph --maxCudaGraphs N --maxPrefillCudaGraphs N --maxDecodeCudaGraphs N "
        "--maxCudaGraphMiB N --maxPrefillCudaGraphMiB N --maxDecodeCudaGraphMiB N] "
        "[--cudaGraphChargeMiB N --cudaGraphReserveMiB N] "
        "[--slotCount N] "
        "[--contextAdapter] [--adaptiveScheduler] [--adaptiveChunking] [--prefillTokenBudget N] "
        "[--dynamicDecodeBatching --schedulerCostJson FILE] [--outputCsv FILE] "
        "[--kernelGroupCsv FILE] "
        "[--inputFile FILE --multimodalEngineDir DIR --traceCsv FILE --traceArrivalRate R] "
        "[--pageReservationMode full|headroom|bounded-overcommit "
        "--pageReservationHeadroomTokens N --pageReservationOvercommitBundles N "
        "--pageReservationGrowthRequests N] "
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
        kSlotCount,
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
        kCudaGraph,
        kMaxCudaGraphs,
        kMaxPrefillCudaGraphs,
        kMaxDecodeCudaGraphs,
        kMaxCudaGraphMiB,
        kMaxPrefillCudaGraphMiB,
        kMaxDecodeCudaGraphMiB,
        kCudaGraphChargeMiB,
        kCudaGraphReserveMiB,
        kPrefillTokenBudget,
        kDynamicDecodeBatching,
        kSchedulerCostJson,
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
        kInputFile,
        kMultimodalEngineDir,
        kTraceCsv,
        kTraceArrivalRate,
        kPageReservationMode,
        kPageReservationHeadroomTokens,
        kPageReservationOvercommitBundles,
        kPageReservationGrowthRequests,
        kHelp,
    };
    option const options[] = {{"engineDir", required_argument, nullptr, kEngineDir},
        {"prefillBatch", required_argument, nullptr, kPrefillBatch},
        {"decodeBatch", required_argument, nullptr, kDecodeBatch}, {"inputLen", required_argument, nullptr, kInputLen},
        {"slotCount", required_argument, nullptr, kSlotCount},
        {"prefillChunkSize", required_argument, nullptr, kPrefillChunkSize},
        {"pastKVLen", required_argument, nullptr, kPastKVLen}, {"warmup", required_argument, nullptr, kWarmup},
        {"iterations", required_argument, nullptr, kIterations}, {"outputCsv", required_argument, nullptr, kOutputCsv},
        {"trtContextMode", required_argument, nullptr, kTensorRTContextMode},
        {"sharedContext", no_argument, nullptr, kLegacySharedContext},
        {"contextAdapter", no_argument, nullptr, kContextAdapter},
        {"adaptiveScheduler", no_argument, nullptr, kAdaptiveScheduler},
        {"adaptiveChunking", no_argument, nullptr, kAdaptiveChunking}, {"cudaGraph", no_argument, nullptr, kCudaGraph},
        {"maxCudaGraphs", required_argument, nullptr, kMaxCudaGraphs},
        {"maxPrefillCudaGraphs", required_argument, nullptr, kMaxPrefillCudaGraphs},
        {"maxDecodeCudaGraphs", required_argument, nullptr, kMaxDecodeCudaGraphs},
        {"maxCudaGraphMiB", required_argument, nullptr, kMaxCudaGraphMiB},
        {"maxPrefillCudaGraphMiB", required_argument, nullptr, kMaxPrefillCudaGraphMiB},
        {"maxDecodeCudaGraphMiB", required_argument, nullptr, kMaxDecodeCudaGraphMiB},
        {"cudaGraphChargeMiB", required_argument, nullptr, kCudaGraphChargeMiB},
        {"cudaGraphReserveMiB", required_argument, nullptr, kCudaGraphReserveMiB},
        {"prefillTokenBudget", required_argument, nullptr, kPrefillTokenBudget},
        {"dynamicDecodeBatching", no_argument, nullptr, kDynamicDecodeBatching},
        {"schedulerCostJson", required_argument, nullptr, kSchedulerCostJson},
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
        {"kernelGroupCsv", required_argument, nullptr, kKernelGroupCsv},
        {"inputFile", required_argument, nullptr, kInputFile},
        {"multimodalEngineDir", required_argument, nullptr, kMultimodalEngineDir},
        {"traceCsv", required_argument, nullptr, kTraceCsv},
        {"traceArrivalRate", required_argument, nullptr, kTraceArrivalRate},
        {"pageReservationMode", required_argument, nullptr, kPageReservationMode},
        {"pageReservationHeadroomTokens", required_argument, nullptr, kPageReservationHeadroomTokens},
        {"pageReservationOvercommitBundles", required_argument, nullptr, kPageReservationOvercommitBundles},
        {"pageReservationGrowthRequests", required_argument, nullptr, kPageReservationGrowthRequests},
        {"help", no_argument, nullptr, kHelp}, {}};

    int optionId{};
    while ((optionId = getopt_long(argc, argv, "", options, nullptr)) != -1)
    {
        switch (optionId)
        {
        case kEngineDir: args.engineDir = optarg; break;
        case kPrefillBatch: args.prefillBatch = std::stoi(optarg); break;
        case kDecodeBatch: args.decodeBatch = std::stoi(optarg); break;
        case kSlotCount: args.slotCount = std::stoi(optarg); break;
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
        case kCudaGraph: args.cudaGraph = true; break;
        case kMaxCudaGraphs: args.maxCudaGraphs = std::stoi(optarg); break;
        case kMaxPrefillCudaGraphs: args.maxPrefillCudaGraphs = std::stoi(optarg); break;
        case kMaxDecodeCudaGraphs: args.maxDecodeCudaGraphs = std::stoi(optarg); break;
        case kMaxCudaGraphMiB: args.maxCudaGraphMiB = std::stoi(optarg); break;
        case kMaxPrefillCudaGraphMiB: args.maxPrefillCudaGraphMiB = std::stoi(optarg); break;
        case kMaxDecodeCudaGraphMiB: args.maxDecodeCudaGraphMiB = std::stoi(optarg); break;
        case kCudaGraphChargeMiB: args.cudaGraphChargeMiB = std::stoi(optarg); break;
        case kCudaGraphReserveMiB: args.cudaGraphReserveMiB = std::stoi(optarg); break;
        case kPrefillTokenBudget: args.prefillTokenBudget = std::stoi(optarg); break;
        case kDynamicDecodeBatching: args.dynamicDecodeBatching = true; break;
        case kSchedulerCostJson: args.schedulerCostJson = optarg; break;
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
        case kInputFile: args.inputFile = optarg; break;
        case kMultimodalEngineDir: args.multimodalEngineDir = optarg; break;
        case kTraceCsv: args.traceCsv = optarg; break;
        case kTraceArrivalRate: args.traceArrivalRate = std::stod(optarg); break;
        case kPageReservationMode:
        {
            std::string const mode{optarg};
            if (mode == "full")
            {
                args.pageReservationMode = rt::PhasePageReservationMode::kFull;
            }
            else if (mode == "headroom")
            {
                args.pageReservationMode = rt::PhasePageReservationMode::kHeadroom;
            }
            else if (mode == "bounded-overcommit")
            {
                args.pageReservationMode = rt::PhasePageReservationMode::kBoundedOvercommit;
            }
            else
            {
                return false;
            }
            break;
        }
        case kPageReservationHeadroomTokens: args.pageReservationHeadroomTokens = std::stoi(optarg); break;
        case kPageReservationOvercommitBundles: args.pageReservationOvercommitBundles = std::stoi(optarg); break;
        case kPageReservationGrowthRequests: args.pageReservationGrowthRequests = std::stoi(optarg); break;
        case kHelp: printUsage(argv[0]); return false;
        default: return false;
        }
    }
    return !args.engineDir.empty() && args.prefillBatch > 0 && args.decodeBatch > 0 && args.slotCount >= 0
        && args.inputLen > 0 && args.prefillChunkSize >= 0 && args.prefillChunkSize <= args.inputLen
        && args.pastKVLen >= 0 && args.warmup >= 0 && args.iterations > 0 && args.loadRequests >= 0
        && args.arrivalRate > 0.0 && args.loadPromptMin >= 0 && args.loadPromptMax >= 0 && args.loadOutputMin > 0
        && args.loadOutputMin <= args.loadOutputMax && args.maxOverlapPrefillTokens >= 0 && args.ttftTargetMs > 0.0
        && args.tpotTargetMs > 0.0 && args.loadPriorityClasses > 0 && args.loadPriorityClasses <= 4
        && args.traceArrivalRate > 0.0 && args.maxCudaGraphs > 0 && args.maxPrefillCudaGraphs != 0
        && args.maxPrefillCudaGraphs >= -1 && args.maxDecodeCudaGraphs != 0 && args.maxDecodeCudaGraphs >= -1
        && args.maxCudaGraphMiB >= 0 && args.maxPrefillCudaGraphMiB >= -1 && args.maxDecodeCudaGraphMiB >= -1
        && args.cudaGraphChargeMiB > 0 && args.cudaGraphReserveMiB >= 0 && args.prefillTokenBudget >= 0
        && args.pageReservationHeadroomTokens >= 0 && args.pageReservationOvercommitBundles >= 0
        && args.pageReservationGrowthRequests > 0 && (!args.dynamicDecodeBatching || !args.schedulerCostJson.empty())
        && (args.inputFile.empty() || !args.traceCsv.empty());
}

std::vector<rt::PhaseDecodeBatchCost> loadDecodeBatchCosts(std::filesystem::path const& path)
{
    std::ifstream stream(path);
    ELLM_CHECK(stream.good(), "Failed to open scheduler cost model");
    nlohmann::json const root = nlohmann::json::parse(stream);
    ELLM_CHECK(
        root.contains("decode") && root.at("decode").is_array(), "Scheduler cost model must contain a decode array");
    std::vector<rt::PhaseDecodeBatchCost> costs;
    for (nlohmann::json const& point : root.at("decode"))
    {
        costs.push_back({point.at("batch_size").get<int32_t>(), point.at("max_context_length").get<int32_t>(),
            point.at("p95_gpu_ms").get<float>()});
    }
    ELLM_CHECK(!costs.empty(), "Scheduler cost model contains no decode points");
    return costs;
}

bool usesSharedTensorRTContext(Args const& args) noexcept
{
    return args.trtContextMode == rt::PhaseTensorRTContextMode::kSharedSerialized;
}

void logCudaGraphStats(char const* phase, rt::EngineExecutor const& executor)
{
    rt::EngineExecutor::CudaGraphCacheStats const stats = executor.getCudaGraphCacheStats();
    LOG_INFO(
        "%s CUDA graph: enabled=%s cached=%zu captures=%lu launches=%lu enqueue=%lu "
        "capture_failures=%lu launch_failures=%lu cache_limit_bypasses=%lu post_enqueue_captures=%lu "
        "observation_evictions=%lu graph_bytes=%zu graph_budget_bytes=%zu graph_min_charge_bytes=%zu "
        "global_reserve_bytes=%zu "
        "graph_budget_bypasses=%lu "
        "graph_budget_rejections=%lu global_reserve_bypasses=%lu global_reserve_rejections=%lu",
        phase, stats.automaticCaptureEnabled ? "true" : "false", stats.cachedGraphs,
        static_cast<unsigned long>(stats.captures), static_cast<unsigned long>(stats.graphLaunches),
        static_cast<unsigned long>(stats.enqueueExecutions), static_cast<unsigned long>(stats.captureFailures),
        static_cast<unsigned long>(stats.graphLaunchFailures), static_cast<unsigned long>(stats.cacheLimitBypasses),
        static_cast<unsigned long>(stats.postEnqueueCaptures), static_cast<unsigned long>(stats.observationEvictions),
        stats.cachedGraphBytes, stats.maxCachedGraphBytes, stats.minimumGraphChargeBytes, stats.minimumFreeMemoryBytes,
        static_cast<unsigned long>(stats.graphMemoryBudgetBypasses),
        static_cast<unsigned long>(stats.graphMemoryBudgetRejections),
        static_cast<unsigned long>(stats.globalMemoryReserveBypasses),
        static_cast<unsigned long>(stats.globalMemoryReserveRejections));
}

void logCudaMemory(char const* point)
{
    size_t freeBytes{};
    size_t totalBytes{};
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    constexpr double kMIB_BYTES{1024.0 * 1024.0};
    LOG_INFO("CUDA memory %s: used=%.1f MiB free=%.1f MiB total=%.1f MiB", point,
        static_cast<double>(totalBytes - freeBytes) / kMIB_BYTES, static_cast<double>(freeBytes) / kMIB_BYTES,
        static_cast<double>(totalBytes) / kMIB_BYTES);
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
              "prefill_queue_wait_us,decode_queue_wait_us,prefill_gpu_ms,decode_gpu_ms,makespan_gpu_ms,overlap_ratio,"
              "page_pool_total_bundles,page_pool_allocated_bundles,page_pool_available_bundles\n";
    output << std::fixed << std::setprecision(6);
    for (rt::PhaseDispatchMetrics const& sample : metrics)
    {
        output << sample.dispatchIndex << ',' << static_cast<int32_t>(sample.kind) << ',' << sample.prefillBatchSize
               << ',' << sample.decodeBatchSize << ',' << sample.prefillTokens << ',' << sample.decodeTokens << ','
               << sample.prefillQueueWaitUs << ',' << sample.decodeQueueWaitUs << ',' << sample.prefillGpuMs << ','
               << sample.decodeGpuMs << ',' << sample.makespanGpuMs << ',' << sample.overlapRatio << ','
               << sample.pagePoolTotalBundles << ',' << sample.pagePoolAllocatedBundles << ','
               << sample.pagePoolAvailableBundles << '\n';
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
    rt::ModelPhaseContract const phaseContract = rt::makeModelPhaseContract(config);
    ELLM_CHECK(!deployment.draft.has_value(), "llm_phase_bench v1 supports vanilla inference only");
    if (!phaseContract.supportsDynamicAdmission)
    {
        // A legacy fixed-linear cache has no stable-slot allocator. It is
        // useful for Cosmos text smoke tests, but admission/eviction would
        // require copying KV rows and is intentionally not enabled here.
        ELLM_CHECK(args.loadRequests == 0 && args.inputFile.empty() && !args.contextAdapter,
            "Non-indexed phase execution supports only the fixed microbenchmark path");
    }
    ELLM_CHECK(args.prefillBatch <= config.maxSupportedPrefillBatchSize,
        "prefillBatch exceeds the engine prefill profile limit");
    ELLM_CHECK(
        args.decodeBatch <= config.maxSupportedDecodeBatchSize, "decodeBatch exceeds the engine decode profile limit");
    if (args.loadRequests == 0 && args.inputFile.empty())
    {
        // A shared TensorRT context serializes the two phases.  It is safe to
        // reuse the same physical slots there, which lets the capacity sweep
        // measure a phase at BS16 on a BS16 fixed-linear engine.  Independent
        // overlap still requires disjoint slots and retains the strict sum
        // check below.
        bool const serializedSlotReuse = usesSharedTensorRTContext(args) && args.slotCount > 0;
        ELLM_CHECK(args.prefillBatch + args.decodeBatch <= config.maxSupportedBatchSize || serializedSlotReuse,
            "Fixed microbenchmark prefill and decode slots exceed maxSupportedBatchSize");
    }
    int32_t const phaseSlotCount = args.slotCount > 0 ? args.slotCount : args.prefillBatch + args.decodeBatch;
    ELLM_CHECK(phaseSlotCount <= config.maxSupportedBatchSize, "Physical slot count exceeds maxSupportedBatchSize");
    ELLM_CHECK(phaseSlotCount >= std::max(args.prefillBatch, args.decodeBatch),
        "Physical slot count is smaller than a phase batch limit");
    rt::LLMEngineConfig resourceConfig = config;
    resourceConfig.maxSupportedBatchSize = phaseSlotCount;
    ELLM_CHECK(args.inputLen <= config.maxSupportedInputLength, "inputLen exceeds maxSupportedInputLength");
    int32_t const loadPromptMin = args.loadPromptMin > 0 ? args.loadPromptMin : args.inputLen;
    int32_t const loadPromptMax = args.loadPromptMax > 0 ? args.loadPromptMax : args.inputLen;
    if (args.loadRequests > 0)
    {
        ELLM_CHECK(!args.loadCsv.empty(), "Continuous-load mode requires --loadCsv");
        ELLM_CHECK(loadPromptMin <= loadPromptMax, "Continuous-load prompt range is invalid");
        ELLM_CHECK(
            loadPromptMax <= args.inputLen || (args.prefillChunkSize > 0 && args.prefillChunkSize <= args.inputLen),
            "Continuous-load prompt maximum requires chunking within the prefill buffer capacity");
        ELLM_CHECK(loadPromptMax + args.loadOutputMax <= config.maxKVCacheCapacity,
            "Continuous-load prompt plus output maximum exceeds KV capacity");
    }
    int32_t const configuredChunkSize = args.prefillChunkSize > 0 ? args.prefillChunkSize : args.inputLen;
    int32_t const phaseRounds = (args.inputLen + configuredChunkSize - 1) / configuredChunkSize;
    ELLM_CHECK(args.pastKVLen + phaseRounds <= config.maxKVCacheCapacity, "pastKVLen exceeds KV capacity");
    ELLM_CHECK(!args.cudaGraph || !usesSharedTensorRTContext(args),
        "--cudaGraph requires --trtContextMode independent so each phase owns its graph cache");
    LOG_INFO("Phase benchmark work per sample: %d prefill chunk(s), %d decode step(s)", phaseRounds, phaseRounds);

    cudaStream_t setupStream{};
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&setupStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    std::filesystem::path const enginePath = engineDir / "llm.engine";
    std::unique_ptr<rt::EngineExecutor> sharedExecutor;
    std::unique_ptr<rt::IndependentEngineExecutorPair> independentExecutors;
    rt::EngineExecutor* prefillRunner{};
    rt::EngineExecutor* decodeRunner{};
    if (usesSharedTensorRTContext(args))
    {
        sharedExecutor = rt::EngineExecutor::createForLLM(enginePath, config);
        prefillRunner = sharedExecutor.get();
        decodeRunner = prefillRunner;
    }
    else
    {
        auto executor = rt::EngineExecutor::createForLLM(enginePath, config);
        rt::IndependentEngineExecutorPairConfig const pairConfig{
            kPrefillProfile, kDecodeProfile, setupStream, prefillStream, decodeStream};
        independentExecutors = rt::IndependentEngineExecutorPair::create(std::move(executor), pairConfig);
        prefillRunner = &independentExecutors->prefillExecutor();
        decodeRunner = &independentExecutors->decodeExecutor();
    }
    if (args.cudaGraph)
    {
        int32_t const prefillLimit = args.maxPrefillCudaGraphs > 0 ? args.maxPrefillCudaGraphs : args.maxCudaGraphs;
        int32_t const decodeLimit = args.maxDecodeCudaGraphs > 0 ? args.maxDecodeCudaGraphs : args.maxCudaGraphs;
        int32_t const prefillMiB
            = args.maxPrefillCudaGraphMiB >= 0 ? args.maxPrefillCudaGraphMiB : args.maxCudaGraphMiB;
        int32_t const decodeMiB = args.maxDecodeCudaGraphMiB >= 0 ? args.maxDecodeCudaGraphMiB : args.maxCudaGraphMiB;
        constexpr size_t kMIB_BYTES{1024U * 1024U};
        size_t const graphChargeBytes = static_cast<size_t>(args.cudaGraphChargeMiB) * kMIB_BYTES;
        size_t const graphReserveBytes = static_cast<size_t>(args.cudaGraphReserveMiB) * kMIB_BYTES;
        prefillRunner->enableAutomaticCudaGraphCapture(static_cast<size_t>(prefillLimit),
            static_cast<size_t>(prefillMiB) * kMIB_BYTES, graphChargeBytes, graphReserveBytes);
        decodeRunner->enableAutomaticCudaGraphCapture(static_cast<size_t>(decodeLimit),
            static_cast<size_t>(decodeMiB) * kMIB_BYTES, graphChargeBytes, graphReserveBytes);
    }
    bool const sharedTensorRTContext
        = prefillRunner->getExecutionContextIdentity() == decodeRunner->getExecutionContextIdentity();
    ELLM_CHECK(sharedTensorRTContext == usesSharedTensorRTContext(args),
        "TensorRT execution-context identity does not match --trtContextMode");
    CUcontext prefillCudaContext{};
    CUcontext decodeCudaContext{};
    CUDA_DRIVER_CHECK(cuStreamGetCtx(prefillStream, &prefillCudaContext));
    CUDA_DRIVER_CHECK(cuStreamGetCtx(decodeStream, &decodeCudaContext));
    ELLM_CHECK(prefillCudaContext != nullptr && prefillCudaContext == decodeCudaContext,
        "Phase streams must share one CUDA primary context");
    LOG_INFO("Phase execution topology: CUDA context=%p (shared), TensorRT prefill=%p, decode=%p (%s)",
        static_cast<void*>(prefillCudaContext), static_cast<void const*>(prefillRunner->getExecutionContextIdentity()),
        static_cast<void const*>(decodeRunner->getExecutionContextIdentity()),
        sharedTensorRTContext ? "shared_serialized" : "independent_concurrent");
    rt::validateAgainstEngine(config, *prefillRunner, "phase-prefill");

    std::unordered_map<std::string, std::string> const emptyLoraMap;
    auto resources = rt::SharedResources::createForLLM(resourceConfig, emptyLoraMap, setupStream);
    auto prefillIO = rt::PipelineIO::createForLLM(config, args.prefillBatch, args.inputLen, setupStream);
    auto decodeIO = rt::PipelineIO::createForLLM(config, args.decodeBatch, 1, setupStream);
    rt::TensorMap prefillMap;
    rt::TensorMap decodeMap;
    rt::buildTensorMap(prefillMap, prefillIO, *resources, resourceConfig, 0);
    rt::buildTensorMap(decodeMap, decodeIO, *resources, resourceConfig, 0);

    // Cosmos/Qwen-VL decoder layers always consume deepstack inputs. A
    // text-only phase run binds the per-phase tensors and clears them, which
    // is equivalent to the runtime's zero-feature path without relying on a
    // one-token shared dummy buffer for a long prefill sequence.
    std::unique_ptr<rt::DeepstackBinding> prefillDeepstack;
    std::unique_ptr<rt::DeepstackBinding> decodeDeepstack;
    if (phaseContract.hasDeepstack())
    {
        prefillDeepstack = std::make_unique<rt::DeepstackBinding>(prefillIO.deepstackEmbeds, resources->zeroBuffer);
        decodeDeepstack = std::make_unique<rt::DeepstackBinding>(decodeIO.deepstackEmbeds, resources->zeroBuffer);
        prefillDeepstack->useRealFeatures(prefillMap);
        decodeDeepstack->useRealFeatures(decodeMap);
    }

    resources->externalWeightManager->load(engineDir, engineDir / "config.json", setupStream);
    resources->externalWeightManager->validateAgainstEngine(*prefillRunner, "phase-shared");
    resources->externalWeightManager->registerTensorMapEntries(prefillMap);
    resources->externalWeightManager->registerAdditionalTensorMapEntries(decodeMap);

    rt::EmbeddingData embedding = rt::loadEmbeddingTable(engineDir / "embedding.safetensors", setupStream);
    rt::EmbeddingPreprocessor prefillEmbedding(embedding, config);
    rt::EmbeddingPreprocessor decodeEmbedding(embedding, config);

    int64_t const prefillContextBytes = prefillRunner->getRequiredContextMemorySizeForProfile(kPrefillProfile);
    int64_t const decodeContextBytes = decodeRunner->getRequiredContextMemorySizeForProfile(kDecodeProfile);
    LOG_INFO("Phase context workspaces: prefill=%.1f MiB decode=%.1f MiB all-profiles=%.1f MiB",
        static_cast<double>(prefillContextBytes) / (1024.0 * 1024.0),
        static_cast<double>(decodeContextBytes) / (1024.0 * 1024.0),
        static_cast<double>(prefillRunner->getRequiredContextMemorySize()) / (1024.0 * 1024.0));
    std::unique_ptr<rt::Tensor> ownedPrefillContext;
    rt::Tensor* prefillContext{};
    rt::Tensor* decodeContext{};
    if (independentExecutors)
    {
        prefillContext = &independentExecutors->prefillContextMemory();
        decodeContext = &independentExecutors->decodeContextMemory();
    }
    else
    {
        ownedPrefillContext = std::make_unique<rt::Tensor>(rt::Coords{prefillContextBytes}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kUINT8, "phase_prefill_context_memory");
        ELLM_CHECK(prefillRunner->setContextMemoryForProfile(kPrefillProfile, *ownedPrefillContext, setupStream),
            "Failed to assign prefill profile context memory");
        prefillContext = ownedPrefillContext.get();
    }

    std::vector<int32_t> prefillSlots(args.prefillBatch);
    std::iota(prefillSlots.begin(), prefillSlots.end(), 0);
    std::vector<int32_t> decodeSlots(args.decodeBatch);
    bool const reuseWarmupSlots = args.prefillBatch + args.decodeBatch > phaseSlotCount;
    std::iota(decodeSlots.begin(), decodeSlots.end(), reuseWarmupSlots ? 0 : args.prefillBatch);
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

    rt::PhaseBatchState prefillBatchState(args.prefillBatch, "phase_prefill", phaseContract.indexedKVCache);
    rt::PhaseBatchState decodeBatchState(args.decodeBatch, "phase_decode", phaseContract.indexedKVCache);
    prefillBatchState.bind(prefillMap);
    decodeBatchState.bind(decodeMap);

    std::vector<int32_t> initialSlotLengths(phaseSlotCount, 0);
    if (!reuseWarmupSlots)
    {
        for (int32_t const slot : decodeSlots)
        {
            initialSlotLengths[slot] = args.pastKVLen;
        }
    }
    rt::Tensor hostInitialSlotLengths(
        {phaseSlotCount}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "phase_initial_slot_lengths");
    std::copy(initialSlotLengths.begin(), initialSlotLengths.end(), hostInitialSlotLengths.dataPointer<int32_t>());
    auto& cacheManager = *resources->cacheManagers[0];
    cacheManager.resetForNewSequences(hostInitialSlotLengths, setupStream);
    auto resetForDecodeWarmup = [&](cudaStream_t stream) {
        if (!reuseWarmupSlots)
        {
            return;
        }
        rt::Tensor hostDecodeLengths(
            {phaseSlotCount}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "warmup_decode_slot_lengths");
        std::fill_n(hostDecodeLengths.dataPointer<int32_t>(), phaseSlotCount, 0);
        for (int32_t const slot : decodeSlots)
        {
            hostDecodeLengths.dataPointer<int32_t>()[slot] = args.pastKVLen;
        }
        cacheManager.resetForNewSequences(hostDecodeLengths, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };

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

    // The fixed-shape benchmark normally records only phase-level CUDA events.
    // Keep a separate recorder for this path so Cosmos (which does not expose
    // the indexed serving facade) can produce the same kernel-group cost data
    // as the dynamic serving path without changing the untimed fast path.
    rt::PhaseKernelGroupRecorder fixedKernelGroupRecorder;
    size_t fixedKernelGroupDispatchIndex{};
    rt::PhaseKernelDispatchMetadata fixedKernelDispatchMetadata;
    bool const recordFixedKernelGroups = !args.kernelGroupCsv.empty();

    uploadInt32(prefillIO.contextLengths, std::vector<int32_t>(args.prefillBatch, args.inputLen), setupStream);
    uploadInt32(decodeIO.contextLengths, std::vector<int32_t>(args.decodeBatch, args.pastKVLen + 1), setupStream);
    uploadInt64(prefillIO.selectTokenIndices, std::vector<int64_t>(args.prefillBatch, args.inputLen - 1), setupStream);
    uploadInt64(decodeIO.selectTokenIndices, std::vector<int64_t>(args.decodeBatch, 0), setupStream);
    if (config.useVisionBidirectionalAttention)
    {
        CUDA_CHECK(cudaMemsetAsync(
            prefillIO.visionBlockIds.rawPointer(), 0xFF, prefillIO.visionBlockIds.getMemoryCapacity(), setupStream));
        CUDA_CHECK(cudaMemsetAsync(
            decodeIO.visionBlockIds.rawPointer(), 0xFF, decodeIO.visionBlockIds.getMemoryCapacity(), setupStream));
    }
    fillRandomData(prefillIO.inputsEmbeds, -1.0F, 1.0F, nvinfer1::DataType::kHALF, 0);
    fillRandomData(decodeIO.inputsEmbeds, -1.0F, 1.0F, nvinfer1::DataType::kHALF, 1);

    std::unique_ptr<rt::Gemma4EmbeddingPreprocessor> prefillGemma4Ple;
    std::unique_ptr<rt::Gemma4EmbeddingPreprocessor> decodeGemma4PleOwner;
    rt::Gemma4EmbeddingPreprocessor* decodeGemma4Ple{};
    if (config.pleEnabled)
    {
        int32_t const maxPleSeqLen = std::max(args.inputLen, 1);
        prefillGemma4Ple = std::make_unique<rt::Gemma4EmbeddingPreprocessor>(
            engineDir, config, args.prefillBatch, maxPleSeqLen, prefillMap, setupStream);
        // PLE output buffers are mutable phase-local state. Keep a sibling for
        // decode even when TensorRT itself is serialized; otherwise an
        // asymmetric prefill/decode batch would exceed the prefill-sized view.
        decodeGemma4PleOwner = prefillGemma4Ple->createSibling(args.decodeBatch, 1, decodeMap);
        decodeGemma4Ple = decodeGemma4PleOwner.get();
        ELLM_CHECK(prefillGemma4Ple->tableDataIdentity() == decodeGemma4Ple->tableDataIdentity(),
            "Phase PLE preprocessors did not share the immutable table");
        ELLM_CHECK(prefillGemma4Ple->outputDataIdentity() != decodeGemma4Ple->outputDataIdentity(),
            "Phase PLE preprocessors aliased mutable outputs");
        rt::Tensor prefillPleTokenIds({args.prefillBatch, maxPleSeqLen}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "phase_prefill_ple_token_ids");
        rt::Tensor decodePleTokenIds(
            {args.decodeBatch, 1}, rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "phase_decode_ple_token_ids");
        CUDA_CHECK(
            cudaMemsetAsync(prefillPleTokenIds.rawPointer(), 0, prefillPleTokenIds.getMemoryCapacity(), setupStream));
        CUDA_CHECK(
            cudaMemsetAsync(decodePleTokenIds.rawPointer(), 0, decodePleTokenIds.getMemoryCapacity(), setupStream));
        prefillGemma4Ple->embed(prefillPleTokenIds, setupStream);
        decodeGemma4Ple->embed(decodePleTokenIds, setupStream);
    }
    CUDA_CHECK(cudaStreamSynchronize(setupStream));
    logCudaMemory("before phase execution");

    auto enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream) {
        ELLM_CHECK(static_cast<int32_t>(batch.size()) == args.prefillBatch, "Unexpected prefill batch size");
        if (!recordFixedKernelGroups)
        {
            prefillBatchState.prepare(batch, cacheManager, stream);
            CUDA_CHECK(cudaMemsetAsync(prefillIO.selectTokenIndices.rawPointer(), 0,
                prefillIO.selectTokenIndices.getMemoryCapacity(), stream));
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
                for (rt::Tensor& deepstack : prefillIO.deepstackEmbeds)
                {
                    check::check(deepstack.reshape({args.prefillBatch, chunkLength, config.hiddenSize}),
                        "Prefill deepstack reshape failed");
                    CUDA_CHECK(cudaMemsetAsync(deepstack.rawPointer(), 0, deepstack.getMemoryCapacity(), stream));
                }
                bool const initialChunk = chunkOffset == 0;
                auto const prefillDims = config.prefillDims(args.prefillBatch, chunkLength, initialChunk);
                if (!prefillRunner->prepare(kPrefillProfile, prefillDims, prefillMap, stream)
                    || !prefillRunner->execute(stream))
                {
                    return false;
                }
                prefillBatchState.commit(cacheManager, chunkLength, stream);
            }
            return true;
        }

        std::vector<rt::PhaseKernelSegment> segments;
        segments.reserve(static_cast<size_t>(args.inputLen / configuredChunkSize) * 3U + 3U);
        bool prefillOk{true};
        for (int32_t chunkOffset = 0; chunkOffset < args.inputLen; chunkOffset += configuredChunkSize)
        {
            int32_t const chunkLength = std::min(configuredChunkSize, args.inputLen - chunkOffset);
            bool const initialChunk = chunkOffset == 0;
            segments.push_back({rt::PhaseKernelGroup::kPrefillPrepare, {}, stream,
                [&, chunkLength, initialChunk](cudaStream_t segmentStream) {
                    if (initialChunk)
                    {
                        prefillBatchState.prepare(batch, cacheManager, segmentStream);
                        CUDA_CHECK(cudaMemsetAsync(prefillIO.selectTokenIndices.rawPointer(), 0,
                            prefillIO.selectTokenIndices.getMemoryCapacity(), segmentStream));
                    }
                    check::check(prefillIO.inputsEmbeds.reshape({args.prefillBatch, chunkLength, config.hiddenSize}),
                        "Prefill input reshape failed");
                    check::check(prefillIO.contextLengths.reshape({args.prefillBatch}),
                        "Prefill context lengths reshape failed");
                    CUDA_CHECK(cudaMemsetAsync(prefillIO.contextLengths.rawPointer(), 0,
                        prefillIO.contextLengths.getMemoryCapacity(), segmentStream));
                    kernel::incrementLengthTensor(prefillIO.contextLengths, chunkLength, segmentStream);
                    if (prefillGemma4Ple)
                    {
                        prefillGemma4Ple->reshapeOutputs(args.prefillBatch, chunkLength);
                    }
                    for (rt::Tensor& deepstack : prefillIO.deepstackEmbeds)
                    {
                        check::check(deepstack.reshape({args.prefillBatch, chunkLength, config.hiddenSize}),
                            "Prefill deepstack reshape failed");
                        CUDA_CHECK(
                            cudaMemsetAsync(deepstack.rawPointer(), 0, deepstack.getMemoryCapacity(), segmentStream));
                    }
                }});
            segments.push_back({rt::PhaseKernelGroup::kPrefillEngine, {}, stream,
                [&, chunkLength, initialChunk](cudaStream_t segmentStream) {
                    auto const prefillDims = config.prefillDims(args.prefillBatch, chunkLength, initialChunk);
                    if (!prefillRunner->prepare(kPrefillProfile, prefillDims, prefillMap, segmentStream)
                        || !prefillRunner->execute(segmentStream))
                    {
                        prefillOk = false;
                    }
                }});
            segments.push_back(
                {rt::PhaseKernelGroup::kPrefillCacheCommit, {}, stream, [&, chunkLength](cudaStream_t segmentStream) {
                     if (prefillOk)
                     {
                         prefillBatchState.commit(cacheManager, chunkLength, segmentStream);
                     }
                 }});
        }
        fixedKernelGroupRecorder.execute(fixedKernelGroupDispatchIndex++, segments, fixedKernelDispatchMetadata);
        fixedKernelGroupRecorder.poll();
        return prefillOk;
    };
    auto executeDecode = [&](rt::PhaseBatchState& activeBatchState, int32_t rounds, cudaStream_t stream) {
        int32_t const batchSize = activeBatchState.lengths().getShape()[0];
        if (!recordFixedKernelGroups)
        {
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
            for (rt::Tensor& deepstack : decodeIO.deepstackEmbeds)
            {
                check::check(deepstack.reshape({batchSize, 1, config.hiddenSize}), "Decode deepstack reshape failed");
                CUDA_CHECK(cudaMemsetAsync(deepstack.rawPointer(), 0, deepstack.getMemoryCapacity(), stream));
            }
            auto const decodeDims = config.decodeDims(batchSize);
            for (int32_t round = 0; round < rounds; ++round)
            {
                if (!decodeRunner->prepare(kDecodeProfile, decodeDims, decodeMap, stream)
                    || !decodeRunner->execute(stream))
                {
                    return false;
                }
                activeBatchState.commit(cacheManager, 1, stream);
                kernel::incrementLengthTensor(decodeIO.contextLengths, 1, stream);
            }
            return true;
        }

        std::vector<rt::PhaseKernelSegment> segments;
        segments.reserve(static_cast<size_t>(rounds) * 3U + 1U);
        bool decodeOk{true};
        segments.push_back(
            {rt::PhaseKernelGroup::kDecodePrepare, {}, stream, [&, batchSize](cudaStream_t segmentStream) {
                 check::check(decodeIO.contextLengths.reshape({batchSize}), "Decode context lengths reshape failed");
                 check::check(decodeIO.outputLogits.reshape({batchSize, config.outputVocabSize}),
                     "Decode logits reshape failed");
                 CUDA_CHECK(
                     cudaMemcpyAsync(decodeIO.contextLengths.rawPointer(), activeBatchState.lengths().rawPointer(),
                         batchSize * sizeof(int32_t), cudaMemcpyDeviceToDevice, segmentStream));
                 kernel::incrementLengthTensor(decodeIO.contextLengths, 1, segmentStream);
                 if (decodeGemma4Ple)
                 {
                     decodeGemma4Ple->reshapeOutputs(batchSize, 1);
                 }
                 for (rt::Tensor& deepstack : decodeIO.deepstackEmbeds)
                 {
                     check::check(
                         deepstack.reshape({batchSize, 1, config.hiddenSize}), "Decode deepstack reshape failed");
                     CUDA_CHECK(
                         cudaMemsetAsync(deepstack.rawPointer(), 0, deepstack.getMemoryCapacity(), segmentStream));
                 }
             }});
        auto const decodeDims = config.decodeDims(batchSize);
        for (int32_t round = 0; round < rounds; ++round)
        {
            segments.push_back(
                {rt::PhaseKernelGroup::kDecodeEngine, {}, stream, [&, decodeDims](cudaStream_t segmentStream) {
                     if (!decodeRunner->prepare(kDecodeProfile, decodeDims, decodeMap, segmentStream)
                         || !decodeRunner->execute(segmentStream))
                     {
                         decodeOk = false;
                     }
                 }});
            segments.push_back({rt::PhaseKernelGroup::kDecodeCacheCommit, {}, stream, [&](cudaStream_t segmentStream) {
                                    if (decodeOk)
                                    {
                                        activeBatchState.commit(cacheManager, 1, segmentStream);
                                        kernel::incrementLengthTensor(decodeIO.contextLengths, 1, segmentStream);
                                    }
                                }});
        }
        fixedKernelGroupRecorder.execute(fixedKernelGroupDispatchIndex++, segments, fixedKernelDispatchMetadata);
        fixedKernelGroupRecorder.poll();
        return decodeOk;
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
            decodeBatchState.prepare(batch, cacheManager, stream, /*decode=*/true);
        }
        return executeDecode(*activeBatchState, phaseRounds, stream);
    };

    // Exercise actual source-context admission outside the fixed-shape timed samples.
    bool const continuousLoad = args.loadRequests > 0;
    bool const realRequestTrace = !args.inputFile.empty();
    std::vector<rt::LLMGenerationRequest> traceRequests;
    std::vector<TraceRequestMetadata> traceMetadata;
    bool traceHasVision{};
    bool traceHasPriority{};
    if (realRequestTrace)
    {
        auto parsed = exampleUtils::parseRequestFile(args.inputFile, 1, -1, 0);
        ELLM_CHECK(parsed.first.empty(), "Real phase trace v1 does not support LoRA weights");
        traceRequests = std::move(parsed.second);
        ELLM_CHECK(!traceRequests.empty(), "Real phase trace contains no requests");
        traceHasVision = std::any_of(traceRequests.begin(), traceRequests.end(),
            [](auto const& request) { return !request.requests.front().imageBuffers.empty(); });
        ELLM_CHECK(!traceHasVision || !usesSharedTensorRTContext(args),
            "Real multimodal trace requires independent TensorRT contexts");
        ELLM_CHECK(!traceHasVision || !args.multimodalEngineDir.empty(),
            "Real multimodal trace requires --multimodalEngineDir");
        traceMetadata = readTraceMetadata(args.inputFile, traceRequests.size(), args.traceArrivalRate);
        traceHasPriority = std::any_of(
            traceMetadata.begin(), traceMetadata.end(), [](auto const& metadata) { return metadata.priority > 0; });
    }
    if (continuousLoad || realRequestTrace
        || (args.prefillBatch == args.decodeBatch && phaseContract.supportsDynamicBatching()))
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
        std::chrono::steady_clock::time_point traceStart;
        std::vector<TraceRequestSample> traceSamples(traceRequests.size());
        std::unordered_map<uint64_t, size_t> traceIndices;
        rt::PhaseQueueSchedulerConfig facadeSchedulerConfig;
        facadeSchedulerConfig.maxPrefillBatchSize = args.prefillBatch;
        facadeSchedulerConfig.maxDecodeBatchSize = args.decodeBatch;
        facadeSchedulerConfig.maxOverlapPrefillTokens = args.maxOverlapPrefillTokens;
        facadeSchedulerConfig.maxPrefillChunkTokens
            = std::min(configuredChunkSize, phaseContract.maxPrefillChunkTokens);
        facadeSchedulerConfig.maxPrefillBatchTokens = args.prefillTokenBudget;
        facadeSchedulerConfig.enableDynamicDecodeBatching = args.dynamicDecodeBatching;
        if (args.dynamicDecodeBatching)
        {
            facadeSchedulerConfig.decodeBatchCosts = loadDecodeBatchCosts(args.schedulerCostJson);
        }
        facadeSchedulerConfig.supportsChunkedPrefill = phaseContract.supportsChunkedPrefill;
        facadeSchedulerConfig.enableMetricsPolicy = args.adaptiveScheduler;
        facadeSchedulerConfig.prefillQueueWaitTargetUs = args.ttftTargetMs * 1000.0;
        facadeSchedulerConfig.decodeQueueWaitTargetUs = args.tpotTargetMs * 1000.0;
        facadeSchedulerConfig.enablePriorityBatching = args.loadPriorityClasses > 1 || traceHasPriority;
        facadeSchedulerConfig.enableAdaptivePrefillChunking = args.adaptiveChunking && configuredChunkSize > 0;
        facadeSchedulerConfig.minPrefillChunkTokens = std::min(32, std::max(1, configuredChunkSize));
        rt::PhaseKernelGroupRecorder kernelGroupRecorder;
        size_t kernelGroupDispatchIndex{};
        rt::PhaseKernelDispatchMetadata kernelDispatchMetadata;
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
                kernelGroupRecorder.execute(kernelGroupDispatchIndex++, segments, kernelDispatchMetadata);
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
            segments.push_back({rt::PhaseKernelGroup::kPrefillPrepare, {}, packed.stream(),
                [&](cudaStream_t stream) {
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
                    prefillEmbedding.embed(
                        packed.tokenIds(), packed.visualEmbeddings(), std::nullopt, prefillIO, stream);
                    prefillEmbedding.prepareDeepstack(packed.tokenIds(), packed.deepstackFeatures(), prefillIO, stream);
                    if (phaseContract.hasMRope && packed.mropeCosSin().has_value())
                    {
                        rt::OptionalInputTensor const mropeCosSin = packed.mropeCosSin();
                        ELLM_CHECK(mropeCosSin.has_value(), "Packed M-RoPE cache disappeared before copy");
                        rt::Tensor const& source = mropeCosSin->get();
                        ELLM_CHECK(source.getDataType() == nvinfer1::DataType::kFLOAT,
                            "Multimodal M-RoPE cache must use FLOAT");
                        CUDA_CHECK(cudaMemsetAsync(
                            prefillIO.mropeCosSin.rawPointer(), 0, prefillIO.mropeCosSin.getMemoryCapacity(), stream));
                        CUDA_CHECK(cudaMemcpyAsync(prefillIO.mropeCosSin.rawPointer(), source.rawPointer(),
                            source.getShape().volume() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                    }
                    if (config.useVisionBidirectionalAttention)
                    {
                        check::check(prefillIO.visionBlockIds.reshape({batchSize, chunkLength}),
                            "Serving prefill vision block IDs reshape failed");
                        rt::Tensor hostVisionBlockIds
                            = rt::generateVisionBlockIds(packed.hostTokenIds(), config.imageTokenId);
                        CUDA_CHECK(cudaMemcpy(prefillIO.visionBlockIds.rawPointer(), hostVisionBlockIds.rawPointer(),
                            batchSize * chunkLength * sizeof(int32_t), cudaMemcpyHostToDevice));
                    }
                    if (prefillGemma4Ple)
                    {
                        prefillGemma4Ple->reshapeOutputs(batchSize, chunkLength);
                        prefillGemma4Ple->embed(packed.tokenIds(), stream);
                    }
                    check::check(prefillIO.outputLogits.reshape({batchSize, config.outputVocabSize}),
                        "Serving prefill logits reshape failed");
                }});
            segments.push_back({rt::PhaseKernelGroup::kPrefillEngine, {}, packed.stream(), [&](cudaStream_t stream) {
                                    auto const dims = config.prefillDims(batchSize, chunkLength, packed.initialChunk());
                                    ELLM_CHECK(prefillRunner->prepare(kPrefillProfile, dims, prefillMap, stream)
                                            && prefillRunner->execute(stream),
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
                if (continuousLoad || realRequestTrace)
                {
                    int64_t const completionUs = elapsedMicroseconds(continuousLoad ? loadStart : traceStart);
                    for (rt::PhasePrefillContextRow const& row : packed.rows())
                    {
                        if (row.tokenOffset + row.tokenCount == row.promptTokenCount)
                        {
                            if (continuousLoad)
                            {
                                LoadRequestSample& sample = loadMetrics.at(loadMetricIndices.at(row.requestId));
                                if (sample.firstTokenUs < 0)
                                {
                                    sample.firstTokenUs = completionUs;
                                }
                            }
                            else
                            {
                                TraceRequestSample& sample = traceSamples.at(traceIndices.at(row.requestId));
                                if (sample.firstTokenUs < 0)
                                {
                                    sample.firstTokenUs = completionUs;
                                }
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
            executeKernelSegments({
                {rt::PhaseKernelGroup::kDecodePrepare, {}, packed.stream,
                    [&](cudaStream_t stream) {
                        check::check(decodeIO.inputsEmbeds.reshape({packed.activeBatchSize, 1, config.hiddenSize}),
                            "Serving decode input reshape failed");
                        decodeEmbedding.embed(adapter.tokenIds(), std::nullopt, std::nullopt, decodeIO, stream);
                        if (decodeGemma4Ple)
                        {
                            decodeGemma4Ple->reshapeOutputs(packed.activeBatchSize, 1);
                            decodeGemma4Ple->embed(adapter.tokenIds(), stream);
                        }
                        if (phaseContract.hasMRope && adapter.mropeCosSin().has_value())
                        {
                            rt::OptionalInputTensor const mropeCosSin = adapter.mropeCosSin();
                            ELLM_CHECK(mropeCosSin.has_value(), "Packed M-RoPE cache disappeared before decode copy");
                            rt::Tensor const& source = mropeCosSin->get();
                            ELLM_CHECK(source.getDataType() == nvinfer1::DataType::kFLOAT,
                                "Multimodal M-RoPE cache must use FLOAT");
                            CUDA_CHECK(cudaMemsetAsync(decodeIO.mropeCosSin.rawPointer(), 0,
                                decodeIO.mropeCosSin.getMemoryCapacity(), stream));
                            CUDA_CHECK(cudaMemcpyAsync(decodeIO.mropeCosSin.rawPointer(), source.rawPointer(),
                                source.getShape().volume() * sizeof(float), cudaMemcpyDeviceToDevice, stream));
                        }
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
        facadeCallbacks.onDispatchMetrics = [&](rt::PhaseDispatchMetrics const& sample) {
            rt::PhaseDispatchMetrics telemetry = sample;
            rt::KVPagePoolStats const pool = cacheManager.getPagedKVPoolStats();
            telemetry.pagePoolTotalBundles = pool.totalBundles;
            telemetry.pagePoolAllocatedBundles = pool.allocatedBundles;
            telemetry.pagePoolAvailableBundles = pool.availableBundles;
            facadeDispatchMetrics.push_back(telemetry);
        };
        facadeCallbacks.onDispatch = [&](rt::PhaseDispatchMetrics const& sample) {
            kernelDispatchMetadata.schedulerDispatchIndex = sample.dispatchIndex;
            kernelDispatchMetadata.schedulerKind = static_cast<int32_t>(sample.kind);
            kernelDispatchMetadata.prefillBatchSize = sample.prefillBatchSize;
            kernelDispatchMetadata.decodeBatchSize = sample.decodeBatchSize;
            kernelDispatchMetadata.prefillTokens = sample.prefillTokens;
            kernelDispatchMetadata.decodeContextTokens = sample.decodeContextTokens;
        };
        facadeCallbacks.onAdmission = [&](rt::PhaseAdmissionResult const& admission) {
            if (continuousLoad)
            {
                LoadRequestSample& sample = loadMetrics.at(loadMetricIndices.at(admission.requestId));
                if (admission.status == rt::PhaseAdmissionStatus::kPending)
                {
                    sample.initiallyPending = true;
                }
                else if (sample.admittedUs < 0)
                {
                    sample.admittedUs = elapsedMicroseconds(loadStart);
                }
                return;
            }
            auto const found = traceIndices.find(admission.requestId);
            if (realRequestTrace && found != traceIndices.end())
            {
                TraceRequestSample& sample = traceSamples.at(found->second);
                sample.admissionAvailableSlots = admission.availableSlots;
                sample.admissionPendingQueueDepth = admission.pendingQueueDepth;
                sample.admissionPagePool = admission.pagePool;
                sample.admissionReservedPageBundles = admission.reservedPageBundles;
                sample.admissionReservationAvailableBundles = admission.reservationAvailableBundles;
                if (admission.status == rt::PhaseAdmissionStatus::kAdmitted && sample.admittedUs < 0)
                {
                    sample.admittedUs = elapsedMicroseconds(traceStart);
                }
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
            ? rt::PhaseExecutionSafetyContract::shared(prefillRunner->getExecutionContextIdentity())
            : rt::PhaseExecutionSafetyContract::independent(
                  {prefillRunner->getExecutionContextIdentity(), prefillContext, &prefillIO},
                  {decodeRunner->getExecutionContextIdentity(), decodeContext, &decodeIO});
        rt::PhaseContextServingFacade facade(phaseSlotCount, facadeSchedulerConfig, std::move(facadeCallbacks),
            cacheManager, decodeMap, prefillStream, decodeStream, facadeMode, &prefillMap,
            facadeSchedulerConfig.maxPrefillChunkTokens, continuousLoad ? servingRequests.size() : traceRequests.size(),
            facadeSafety);
        if (realRequestTrace)
        {
            tokenizer::Tokenizer tokenizer;
            ELLM_CHECK(tokenizer.loadFromHF(engineDir), "Failed to load phase trace tokenizer");
            tokenizer.setAdditionalEosIds(config.eosTokenIds);

            ELLM_CHECK(enqueuePrefill(prefillBatch, prefillStream), "Real trace prefill context warmup failed");
            CUDA_CHECK(cudaStreamSynchronize(prefillStream));
            resetForDecodeWarmup(decodeStream);
            ELLM_CHECK(enqueueDecode(decodeBatch, decodeStream, false), "Real trace decode context warmup failed");
            CUDA_CHECK(cudaStreamSynchronize(decodeStream));

            rt::Tensor hostZeroLengths(
                {phaseSlotCount}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "trace_zero_slot_lengths");
            std::fill_n(hostZeroLengths.dataPointer<int32_t>(), phaseSlotCount, 0);
            cacheManager.resetForNewSequences(hostZeroLengths, setupStream);
            CUDA_CHECK(cudaStreamSynchronize(setupStream));

            auto runTrace = [&](rt::PhaseAsyncServer& server) {
                traceStart = std::chrono::steady_clock::now();
                size_t nextSubmission{};
                size_t completionCount{};
                constexpr int64_t kTraceTimeoutUs = 300000000;
                while (completionCount < traceRequests.size())
                {
                    int64_t const elapsedUs = elapsedMicroseconds(traceStart);
                    ELLM_CHECK(elapsedUs < kTraceTimeoutUs, "Real phase trace exceeded its timeout");
                    bool madeProgress{};
                    while (nextSubmission < traceRequests.size()
                        && traceMetadata[nextSubmission].arrivalOffsetUs <= elapsedUs)
                    {
                        rt::LLMGenerationRequest& request = traceRequests[nextSubmission];
                        TraceRequestMetadata const& metadata = traceMetadata[nextSubmission];
                        if (metadata.maxGenerateLength.has_value())
                        {
                            request.maxGenerateLength = *metadata.maxGenerateLength;
                        }
                        request.disableSpecDecode = true;
                        request.topK = 1;
                        request.numLogprobs = 0;
                        size_t const imageCount = request.requests.front().imageBuffers.size();
                        int32_t const maxOutputTokens = static_cast<int32_t>(request.maxGenerateLength);
                        rt::PhaseSchedulingHints scheduling;
                        scheduling.priority = metadata.priority;
                        scheduling.ttftTargetUs = args.ttftTargetMs * 1000.0;
                        scheduling.tpotTargetUs = args.tpotTargetMs * 1000.0;
                        rt::PhaseAsyncSubmission const submission = server.submit(std::move(request), scheduling);
                        TraceRequestSample& sample = traceSamples[nextSubmission];
                        sample.requestId = submission.requestId;
                        sample.scheduledArrivalUs = metadata.arrivalOffsetUs;
                        sample.submittedUs = elapsedMicroseconds(traceStart);
                        if (submission.status != rt::PhaseRequestStatus::kPending)
                        {
                            sample.admittedUs = sample.submittedUs;
                        }
                        sample.imageCount = imageCount;
                        sample.maxOutputTokens = maxOutputTokens;
                        sample.admissionStatus = submission.status;
                        traceIndices.emplace(submission.requestId, nextSubmission);
                        std::optional<rt::PhaseRequestSnapshot> const snapshot = facade.request(submission.requestId);
                        ELLM_CHECK(snapshot.has_value(), "Submitted trace request is missing from the facade");
                        sample.promptTokens = snapshot->promptTokenCount;
                        ++nextSubmission;
                        madeProgress = true;
                    }
                    madeProgress = server.poll() || madeProgress;
                    while (auto completion = server.tryPopCompletion())
                    {
                        TraceRequestSample& sample = traceSamples.at(traceIndices.at(completion->requestId));
                        sample.completedUs = elapsedMicroseconds(traceStart);
                        sample.completion = std::move(*completion);
                        LOG_INFO("Trace request %lu completed in %.3f ms: %s",
                            static_cast<unsigned long>(sample.requestId), sample.completion.latencyMs,
                            sample.completion.response.outputTexts.front().c_str());
                        ++completionCount;
                        madeProgress = true;
                    }
                    if (!madeProgress)
                    {
                        std::this_thread::yield();
                    }
                }
                ELLM_CHECK(server.empty() && facade.empty(), "Real phase trace did not drain all LLM phase queues");
            };

            if (!traceHasVision)
            {
                LOG_INFO("Text-only phase topology: CUDA=%p prefill=%p decode=%p",
                    static_cast<void*>(prefillCudaContext), prefillRunner->getExecutionContextIdentity(),
                    decodeRunner->getExecutionContextIdentity());
                rt::PhaseAsyncServerConfig serverConfig;
                serverConfig.maxInFlightRequests = traceRequests.size();
                serverConfig.pageReservation = {args.pageReservationMode, args.pageReservationHeadroomTokens,
                    args.pageReservationOvercommitBundles, args.pageReservationGrowthRequests};
                rt::PhaseAsyncServer server(serverConfig, facade, tokenizer, prefillStream);
                runTrace(server);
            }
            else
            {
                cudaStream_t encoderStream{};
                CUDA_CHECK(cudaStreamCreateWithFlags(&encoderStream, cudaStreamNonBlocking));
                {
                    std::filesystem::path visionDir{args.multimodalEngineDir};
                    if (std::filesystem::exists(visionDir / "visual" / "visual.engine"))
                    {
                        visionDir /= "visual";
                    }
                    auto visionRunner = rt::MultimodalRunner::create(
                        visionDir.string(), config.maxSupportedBatchSize, config.maxKVCacheCapacity, encoderStream);
                    int64_t const encoderContextBytes = visionRunner->getRequiredContextMemorySize();
                    rt::Tensor encoderContext({encoderContextBytes}, rt::DeviceType::kGPU, nvinfer1::DataType::kUINT8,
                        "phase_encoder_context_memory");
                    ELLM_CHECK(visionRunner->setContextMemory(encoderContext),
                        "Failed to assign independent encoder context memory");
                    std::unique_ptr<rt::PhaseVisionAdapter> visionAdapter;
                    switch (visionRunner->getModelType())
                    {
                    case multimodal::ModelType::GEMMA4_VISION:
                        visionAdapter = std::make_unique<rt::Gemma4PhaseVisionAdapter>(
                            *visionRunner, tokenizer, &kernelGroupRecorder);
                        break;
                    case multimodal::ModelType::QWEN3_VL:
                    case multimodal::ModelType::QWEN3_5:
                        visionAdapter = std::make_unique<rt::Qwen3VLPhaseVisionAdapter>(
                            *visionRunner, tokenizer, config, &kernelGroupRecorder);
                        break;
                    default: ELLM_CHECK(false, "No phase vision adapter is registered for this multimodal model");
                    }
                    rt::PhaseEncoderExecutionSafetyContract encoderSafety{
                        {visionRunner->getExecutionContextIdentity(), &encoderContext, visionRunner.get()},
                        {{prefillRunner->getExecutionContextIdentity(), prefillContext, &prefillIO},
                            {decodeRunner->getExecutionContextIdentity(), decodeContext, &decodeIO}}};
                    rt::PhaseEncoderQueueConfig encoderQueueConfig;
                    encoderQueueConfig.maxBatchSize = 1;
                    encoderQueueConfig.maxQueuedRequests = traceRequests.size();
                    rt::PhaseEncoderDispatchWorker encoderWorker(
                        encoderQueueConfig, visionAdapter->makeCallbacks(),
                        [&](rt::PhaseWorkItem const& item) { facade.beginPrefillAfterEncoder(item); }, encoderStream,
                        std::move(encoderSafety));
                    rt::PhaseOnlineCoordinator coordinator(encoderWorker, facade);
                    rt::PhaseAsyncServerConfig serverConfig;
                    serverConfig.maxInFlightRequests = traceRequests.size();
                    serverConfig.pageReservation = {args.pageReservationMode, args.pageReservationHeadroomTokens,
                        args.pageReservationOvercommitBundles, args.pageReservationGrowthRequests};
                    rt::PhaseAsyncServer server(serverConfig, coordinator, encoderWorker, facade, tokenizer,
                        prefillStream, visionAdapter.get());
                    runTrace(server);
                    ELLM_CHECK(encoderWorker.empty(), "Real phase trace did not drain the encoder queue");
                }
                CUDA_CHECK(cudaStreamDestroy(encoderStream));
            }
            writeTraceMetrics(args.traceCsv, traceSamples);
            LOG_INFO("Real request phase trace metrics written to %s", args.traceCsv.c_str());
        }
        else if (continuousLoad)
        {
            ELLM_CHECK(enqueuePrefill(prefillBatch, prefillStream), "Continuous-load prefill warmup failed");
            CUDA_CHECK(cudaStreamSynchronize(prefillStream));
            resetForDecodeWarmup(decodeStream);
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
        rt::KVPagePoolStats const finalPagePool = cacheManager.getPagedKVPoolStats();
        if (finalPagePool.totalBundles > 0)
        {
            ELLM_CHECK(finalPagePool.allocatedBundles == 0,
                "Serving facade engine smoke did not release all paged KV bundles");
            ELLM_CHECK(finalPagePool.availableBundles == finalPagePool.totalBundles,
                "Serving facade engine smoke did not restore the paged KV pool");
            LOG_INFO("Serving facade final paged KV pool: allocated=%d available=%d total=%d",
                finalPagePool.allocatedBundles, finalPagePool.availableBundles, finalPagePool.totalBundles);
        }
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
        else if (realRequestTrace)
        {
            dispatchCsv = args.traceCsv;
            dispatchCsv.replace_filename(dispatchCsv.stem().string() + "-dispatch.csv");
            std::vector<float> ttftUs;
            std::vector<float> e2eUs;
            int64_t terminalUs{};
            int32_t generatedTokens{};
            for (TraceRequestSample const& sample : traceSamples)
            {
                ELLM_CHECK(sample.admittedUs >= sample.submittedUs && sample.firstTokenUs >= sample.admittedUs
                        && sample.completedUs >= sample.firstTokenUs,
                    "Real trace request timestamps are invalid");
                ttftUs.push_back(static_cast<float>(sample.firstTokenUs - sample.scheduledArrivalUs));
                e2eUs.push_back(static_cast<float>(sample.completedUs - sample.scheduledArrivalUs));
                terminalUs = std::max(terminalUs, sample.completedUs);
                generatedTokens += static_cast<int32_t>(sample.completion.response.outputIds.front().size());
            }
            LOG_INFO(
                "Real request trace completed: requests=%zu rate=%.3f req/s achieved=%.3f req/s %.3f token/s "
                "TTFT median/p95=%.3f/%.3f ms E2E median/p95=%.3f/%.3f ms",
                traceSamples.size(), args.traceArrivalRate,
                static_cast<double>(traceSamples.size()) * 1000000.0 / terminalUs,
                static_cast<double>(generatedTokens) * 1000000.0 / terminalUs, percentile(ttftUs, 0.5F) / 1000.0F,
                percentile(ttftUs, 0.95F) / 1000.0F, percentile(e2eUs, 0.5F) / 1000.0F,
                percentile(e2eUs, 0.95F) / 1000.0F);
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

    if (realRequestTrace)
    {
        logCudaGraphStats("Prefill", *prefillRunner);
        logCudaGraphStats("Decode", *decodeRunner);
        logCudaMemory("after phase execution");
        CUDA_CHECK(cudaStreamDestroy(prefillStream));
        CUDA_CHECK(cudaStreamDestroy(decodeStream));
        CUDA_CHECK(cudaStreamDestroy(setupStream));
        return EXIT_SUCCESS;
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

    size_t fixedSchedulerDispatchIndex{};
    auto runOnce = [&](bool concurrent) {
        cacheManager.resetForNewSequences(hostInitialSlotLengths, setupStream);
        fixedKernelDispatchMetadata.schedulerDispatchIndex = fixedSchedulerDispatchIndex++;
        fixedKernelDispatchMetadata.schedulerKind
            = static_cast<int32_t>(concurrent ? rt::PhaseDispatchKind::kOverlap : rt::PhaseDispatchKind::kNone);
        fixedKernelDispatchMetadata.prefillBatchSize = args.prefillBatch;
        fixedKernelDispatchMetadata.decodeBatchSize = args.decodeBatch;
        fixedKernelDispatchMetadata.prefillTokens = args.inputLen;
        fixedKernelDispatchMetadata.decodeContextTokens = args.pastKVLen;
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
                ? rt::PhaseExecutionSafetyContract::shared(prefillRunner->getExecutionContextIdentity())
                : rt::PhaseExecutionSafetyContract::independent(
                      {prefillRunner->getExecutionContextIdentity(), prefillContext, &prefillIO},
                      {decodeRunner->getExecutionContextIdentity(), decodeContext, &decodeIO});
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

    // TensorRT lazily initializes tactics and auxiliary-stream state on the
    // first enqueue. Prime both modes before collecting user-visible samples;
    // otherwise the fixed loop order (sequential first, concurrent second)
    // makes a zero-warmup run report a false multi-x speedup for concurrent mode.
    static_cast<void>(runOnce(false));
    static_cast<void>(runOnce(true));
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
    if (recordFixedKernelGroups)
    {
        fixedKernelGroupRecorder.drain();
        fixedKernelGroupRecorder.writeCsv(args.kernelGroupCsv);
        LOG_INFO("Fixed-path kernel-group CUDA-event samples written to %s", args.kernelGroupCsv.c_str());
    }
    logCudaGraphStats("Prefill", *prefillRunner);
    logCudaGraphStats("Decode", *decodeRunner);
    logCudaMemory("after phase execution");

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
