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

#include "common/trtUtils.h"
#include "memoryMonitor.h"
#include "profileFormatter.h"
#include "profiling/metrics.h"
#include "profiling/timer.h"
#include "runtime/llmInferenceDisaggregationRuntime.h"
#include "runtime/llmInferenceRuntime.h"
#include "runtime/llmInferenceSpecDecodeRuntime.h"
#include "runtime/llmRuntimeUtils.h"
#include "tokenizer/tokenizer.h"
#include <filesystem>
#include <fstream>
#include <future>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <array>
#include <cstdint>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace trt_edgellm;
using Json = nlohmann::json;

// Enum for command line option IDs (using traditional enum for C library compatibility)
enum LLMInferenceOptionId : int
{
    HELP = 900,
    INPUT_FILE = 901,
    ENGINE_DIR = 902,
    MULTIMODAL_ENGINE_DIR = 903,
    OUTPUT_FILE = 904,
    DEBUG = 905,
    DUMP_PROFILE = 906,
    PROFILE_OUTPUT_FILE = 907,
    WARMUP = 908,
    DUMP_OUTPUT = 909,
    EAGLE = 910,
    EAGLE_DRAFT_TOP_K = 911,
    EAGLE_DRAFT_STEP = 912,
    EAGLE_VERIFY_TREE_SIZE = 913,
    BATCH_SIZE = 914,
    MAX_GENERATE_LENGTH = 915,
    BENCHMARK_COUNT = 916,
    DISAGGREGATION = 917,
    TPC_COUNT = 918,
    QUIET = 919,
    DISAGG_DECODE_CUDA_GRAPH = 920
};

// Struct to hold Eagle-specific arguments for speculative decoding
struct EagleArgs
{
    bool enabled{false};

    // Number of tokens selected per drafting step from the draft model's output distribution.
    // This controls the branching factor at each level of the draft tree.
    int32_t draftTopK{10};

    // Number of drafting steps to perform with the draft model.
    // Each step extends the draft tree by one more level.
    int32_t draftStep{6};

    // Number of tokens to select from the complete draft tree for base model verification.
    // The total draft tree size is: 1 + draftTopK + (draftStep - 1) * draftTopK * draftTopK
    // This parameter should be <= total draft tree size for optimal performance.
    int32_t verifyTreeSize{60};
};

struct LLMInferenceArgs
{
    bool help{false};
    std::string engineDir;
    std::string multimodalEngineDir{""};
    std::string inputFile;
    std::string outputFile{""};
    std::string profileOutputFile{""};
    bool debug{false};
    bool quiet{false};
    bool dumpProfile{false};
    int32_t warmup{0};
    bool dumpOutput{false};
    // Override parameters (only batchSize and maxGenerateLength can be overridden via CLI)
    // For other sampling parameters (temperature, top_p, top_k), please specify them in the input JSON file
    int32_t batchSize{-1};         // -1 means use value from input file
    int64_t maxGenerateLength{-1}; // -1 means use value from input file
    int32_t tpcCount{-1};          // -1 means disabled
    int32_t benchmarkCount{1};     // Repeat full input request set N times
    bool disaggregation{false};
    bool disaggDecodeCudaGraph{false};
    EagleArgs eagleArgs;
};

namespace
{
constexpr size_t kCuda130MaskOffsetJetson = 0x54c;
constexpr std::array<uint32_t, 3> kJetsonThorGpcMasks{
    0x049, // GPC0 -> TPC {0,3,6}
    0x092, // GPC1 -> TPC {1,4,7}
    0x324 // GPC2 -> TPC {2,5,8,9}
};

struct StreamSmMaskV2
{
    uint32_t enabled;
    uint32_t mask[4];
};

std::vector<uint32_t> getJetsonThorTpcOrderFromGpcMasks()
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

bool applyTpcMaskToStream(cudaStream_t stream, int32_t tpcCount)
{
    static const std::vector<uint32_t> kJetsonThorTpcOrder = getJetsonThorTpcOrderFromGpcMasks();
    if (kJetsonThorTpcOrder.empty())
    {
        LOG_ERROR("Failed to build Jetson Thor TPC order from GPC masks.");
        return false;
    }

    if (tpcCount <= 0 || tpcCount > static_cast<int32_t>(kJetsonThorTpcOrder.size()))
    {
        LOG_ERROR("Invalid tpcCount=%d. Expected range is [1, %zu] for Jetson Thor.", tpcCount,
            kJetsonThorTpcOrder.size());
        return false;
    }

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
        LOG_ERROR("Failed to resolve internal stream struct for TPC mask.");
        return false;
    }

    auto* hwMask = reinterpret_cast<StreamSmMaskV2*>(streamStructBase + kCuda130MaskOffsetJetson);

    // libsmctrl semantics: 0-bit means enabled TPC, 1-bit means disabled TPC.
    // Start with all disabled and clear bits for selected TPCs.
    __uint128_t fullMask = ~static_cast<__uint128_t>(0);
    for (int32_t i = 0; i < tpcCount; ++i)
    {
        fullMask &= ~(static_cast<__uint128_t>(1) << kJetsonThorTpcOrder[static_cast<size_t>(i)]);
    }

    hwMask->enabled = 1;
    hwMask->mask[0] = static_cast<uint32_t>(fullMask);
    hwMask->mask[1] = static_cast<uint32_t>(fullMask >> 32);
    hwMask->mask[2] = static_cast<uint32_t>(fullMask >> 64);
    hwMask->mask[3] = static_cast<uint32_t>(fullMask >> 96);

    LOG_INFO("Applied TPC mask to stream: tpcCount=%d, mask64=0x%016llx", tpcCount,
        static_cast<unsigned long long>(fullMask));
    return true;
}
} // namespace

void printUsage(char const* programName)
{
    std::cerr << "Usage: " << programName
              << " [--help] [--engineDir=<path to engine directory>] [--multimodalEngineDir=<path to multimodal engine "
                 "directory>] [--inputFile=<path to input file>] [--outputFile=<path to output file>] "
                 "[--dumpProfile] [--profileOutputFile=<path to profile output file>] [--warmup=<number>] [--debug] "
                 "[--dumpOutput] [--batchSize=<number>] [--maxGenerateLength=<number>] [--tpcCount=<number>] [--eagle] "
                 "[--benchmarkCount=<number>] [--disaggregation] "
                 "[--quiet] [--disaggDecodeCudaGraph] "
                 "[--eagleDraftTopK=<number>] [--eagleDraftStep=<number>] "
                 "[--eagleVerifyTreeSize=<number>]"
              << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --help                    Display this help message" << std::endl;
    std::cerr << "  --inputFile               Path to input JSON file with requests" << std::endl;
    std::cerr << "  --engineDir               Path to engine directory" << std::endl;
    std::cerr << "  --multimodalEngineDir     Path to multimodal engine directory (optional)" << std::endl;
    std::cerr << "  --outputFile              Path to output JSON file (optional)" << std::endl;
    std::cerr << "  --dumpProfile             Dump profiling summary to console" << std::endl;
    std::cerr << "  --profileOutputFile       Path to profile JSON output file (optional)" << std::endl;
    std::cerr << "  --warmup                  Number of warmup runs using the first request (default: 0)" << std::endl;
    std::cerr << "  --debug                   Enable debug logging" << std::endl;
    std::cerr << "  --quiet                   Show only warning/error logs" << std::endl;
    std::cerr << "  --dumpOutput              Dump inference output to console" << std::endl;
    std::cerr << "  --batchSize               Override batch size from input file" << std::endl;
    std::cerr << "  --maxGenerateLength       Override max generate length from input file" << std::endl;
    std::cerr << "  --tpcCount                Enable first N TPCs using Jetson Thor GPC/TPC map (1..10)" << std::endl;
    std::cerr << "  --benchmarkCount          Repeat full input requests N times (default: 1)" << std::endl;
    std::cerr << "  --disaggregation          Enable disaggregation runtime (3-stage async workers)" << std::endl;
    std::cerr << "  --disaggDecodeCudaGraph   Try capture/use decode CUDA graph in disaggregation mode" << std::endl;
    std::cerr << "                            NOTE: For sampling parameters (temperature, top_p, top_k)," << std::endl;
    std::cerr << "                            please specify them in the input JSON file instead of CLI" << std::endl;
    std::cerr << "  --eagle                   Enable Eagle speculative decoding mode" << std::endl;
    std::cerr << "  --eagleDraftTopK          Number of tokens selected per drafting step (default: 10)" << std::endl;
    std::cerr << "                            Controls branching factor at each draft tree level" << std::endl;
    std::cerr << "  --eagleDraftStep          Number of drafting steps to perform (default: 6)" << std::endl;
    std::cerr << "                            Each step extends the draft tree by one more level" << std::endl;
    std::cerr << "  --eagleVerifyTreeSize     Number of tokens for base model verification (default: 60)" << std::endl;
    std::cerr << "                            Total draft tree size: 1 + topK + (step-1) * topK^2" << std::endl;
}

bool parseLLMInferenceArgs(LLMInferenceArgs& args, int argc, char* argv[])
{
    static struct option inferenceOptions[] = {{"help", no_argument, 0, LLMInferenceOptionId::HELP},
        {"inputFile", required_argument, 0, LLMInferenceOptionId::INPUT_FILE},
        {"engineDir", required_argument, 0, LLMInferenceOptionId::ENGINE_DIR},
        {"multimodalEngineDir", required_argument, 0, LLMInferenceOptionId::MULTIMODAL_ENGINE_DIR},
        {"outputFile", required_argument, 0, LLMInferenceOptionId::OUTPUT_FILE},
        {"debug", no_argument, 0, LLMInferenceOptionId::DEBUG},
        {"quiet", no_argument, 0, LLMInferenceOptionId::QUIET},
        {"dumpProfile", no_argument, 0, LLMInferenceOptionId::DUMP_PROFILE},
        {"profileOutputFile", required_argument, 0, LLMInferenceOptionId::PROFILE_OUTPUT_FILE},
        {"warmup", required_argument, 0, LLMInferenceOptionId::WARMUP},
        {"dumpOutput", no_argument, 0, LLMInferenceOptionId::DUMP_OUTPUT},
        {"eagle", no_argument, 0, LLMInferenceOptionId::EAGLE},
        {"eagleDraftTopK", required_argument, 0, LLMInferenceOptionId::EAGLE_DRAFT_TOP_K},
        {"eagleDraftStep", required_argument, 0, LLMInferenceOptionId::EAGLE_DRAFT_STEP},
        {"eagleVerifyTreeSize", required_argument, 0, LLMInferenceOptionId::EAGLE_VERIFY_TREE_SIZE},
        {"batchSize", required_argument, 0, LLMInferenceOptionId::BATCH_SIZE},
        {"maxGenerateLength", required_argument, 0, LLMInferenceOptionId::MAX_GENERATE_LENGTH},
        {"tpcCount", required_argument, 0, LLMInferenceOptionId::TPC_COUNT},
        {"benchmarkCount", required_argument, 0, LLMInferenceOptionId::BENCHMARK_COUNT},
        {"disaggregation", no_argument, 0, LLMInferenceOptionId::DISAGGREGATION},
        {"disaggDecodeCudaGraph", no_argument, 0, LLMInferenceOptionId::DISAGG_DECODE_CUDA_GRAPH},
        {0, 0, 0, 0}};

    int opt;
    while ((opt = getopt_long(argc, argv, "", inferenceOptions, nullptr)) != -1)
    {
        switch (opt)
        {
        case LLMInferenceOptionId::HELP: args.help = true; return true;
        case LLMInferenceOptionId::INPUT_FILE: args.inputFile = optarg; break;
        case LLMInferenceOptionId::ENGINE_DIR: args.engineDir = optarg; break;
        case LLMInferenceOptionId::MULTIMODAL_ENGINE_DIR: args.multimodalEngineDir = optarg; break;
        case LLMInferenceOptionId::OUTPUT_FILE: args.outputFile = optarg; break;
        case LLMInferenceOptionId::DEBUG: args.debug = true; break;
        case LLMInferenceOptionId::QUIET: args.quiet = true; break;
        case LLMInferenceOptionId::DUMP_PROFILE: args.dumpProfile = true; break;
        case LLMInferenceOptionId::PROFILE_OUTPUT_FILE: args.profileOutputFile = optarg; break;
        case LLMInferenceOptionId::WARMUP:
            try
            {
                args.warmup = std::stoi(optarg);
                if (args.warmup < 0)
                {
                    LOG_ERROR("Invalid warmup value: %s (must be non-negative)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid warmup value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DUMP_OUTPUT: args.dumpOutput = true; break;
        case LLMInferenceOptionId::EAGLE: args.eagleArgs.enabled = true; break;
        case LLMInferenceOptionId::EAGLE_DRAFT_TOP_K:
            try
            {
                args.eagleArgs.draftTopK = std::stoi(optarg);
                if (args.eagleArgs.draftTopK <= 0)
                {
                    LOG_ERROR("Invalid eagleDraftTopK value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleDraftTopK value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::EAGLE_DRAFT_STEP:
            try
            {
                args.eagleArgs.draftStep = std::stoi(optarg);
                if (args.eagleArgs.draftStep <= 0)
                {
                    LOG_ERROR("Invalid eagleDraftStep value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleDraftStep value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::EAGLE_VERIFY_TREE_SIZE:
            try
            {
                args.eagleArgs.verifyTreeSize = std::stoi(optarg);
                if (args.eagleArgs.verifyTreeSize <= 0)
                {
                    LOG_ERROR("Invalid eagleVerifyTreeSize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid eagleVerifyTreeSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::BATCH_SIZE:
            try
            {
                args.batchSize = std::stoi(optarg);
                if (args.batchSize <= 0)
                {
                    LOG_ERROR("Invalid batchSize value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid batchSize value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::MAX_GENERATE_LENGTH:
            try
            {
                args.maxGenerateLength = std::stoll(optarg);
                if (args.maxGenerateLength <= 0)
                {
                    LOG_ERROR("Invalid maxGenerateLength value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid maxGenerateLength value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::TPC_COUNT:
            try
            {
                args.tpcCount = std::stoi(optarg);
                auto const tpcLimit = static_cast<int32_t>(getJetsonThorTpcOrderFromGpcMasks().size());
                if (args.tpcCount <= 0 || args.tpcCount > tpcLimit)
                {
                    LOG_ERROR("Invalid tpcCount value: %s (must be in [1, %zu])", optarg,
                        static_cast<size_t>(tpcLimit));
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid tpcCount value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::BENCHMARK_COUNT:
            try
            {
                args.benchmarkCount = std::stoi(optarg);
                if (args.benchmarkCount <= 0)
                {
                    LOG_ERROR("Invalid benchmarkCount value: %s (must be positive)", optarg);
                    return false;
                }
            }
            catch (std::exception const& e)
            {
                LOG_ERROR("Invalid benchmarkCount value: %s", optarg);
                return false;
            }
            break;
        case LLMInferenceOptionId::DISAGGREGATION: args.disaggregation = true; break;
        case LLMInferenceOptionId::DISAGG_DECODE_CUDA_GRAPH: args.disaggDecodeCudaGraph = true; break;
        default: return false;
        }
    }

    if (args.debug && args.quiet)
    {
        LOG_ERROR("Cannot enable both --debug and --quiet at the same time.");
        return false;
    }

    if (args.quiet)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING);
    }
    else if (args.debug)
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kVERBOSE);
    }
    else
    {
        gLogger.setLevel(nvinfer1::ILogger::Severity::kINFO);
    }

    LOG_INFO("args.inputFile: %s", args.inputFile.c_str());
    if (args.inputFile.empty())
    {
        LOG_ERROR("ERROR: --inputFile is required");
        return false;
    }
    LOG_INFO("args.engineDir: %s", args.engineDir.c_str());
    if (args.engineDir.empty())
    {
        LOG_ERROR("ERROR: --engineDir is required");
        return false;
    }
    if (!args.multimodalEngineDir.empty())
    {
        LOG_INFO("args.multimodalEngineDir: %s", args.multimodalEngineDir.c_str());
    }

    if (args.outputFile.empty())
    {
        LOG_ERROR("ERROR: --outputFile is required");
        return false;
    }
    LOG_INFO("args.outputFile: %s", args.outputFile.c_str());

    if (args.dumpOutput)
    {
        LOG_INFO("args.dumpOutput: enabled");
    }

    if (!args.profileOutputFile.empty())
    {
        LOG_INFO("args.profileOutputFile: %s", args.profileOutputFile.c_str());
    }

    if (args.dumpProfile)
    {
        LOG_INFO("Profile dumping to console is enabled");
    }

    if (args.warmup > 0)
    {
        LOG_INFO("Warmup runs: %d", args.warmup);
    }
    if (args.benchmarkCount > 1)
    {
        LOG_INFO("Benchmark repetitions: %d", args.benchmarkCount);
    }
    if (args.tpcCount > 0)
    {
        if (args.disaggregation)
        {
            LOG_INFO("Disaggregation TPC split enabled: decode tpcCount=%d, prefill+encoding use remaining TPCs.",
                args.tpcCount);
        }
        else
        {
            LOG_INFO("TPC stream mask enabled with tpcCount=%d", args.tpcCount);
        }
    }

    if (args.eagleArgs.enabled)
    {
        LOG_INFO("Eagle mode enabled");
        LOG_INFO("Eagle draft topK: %d", args.eagleArgs.draftTopK);
        LOG_INFO("Eagle draft step: %d", args.eagleArgs.draftStep);
        LOG_INFO("Eagle verify tree size: %d", args.eagleArgs.verifyTreeSize);
    }
    if (args.disaggregation)
    {
        LOG_INFO("Disaggregation runtime enabled");
        if (args.disaggDecodeCudaGraph)
        {
            LOG_INFO("Disaggregation decode CUDA graph capture enabled");
        }
    }
    if (args.eagleArgs.enabled && args.disaggregation)
    {
        LOG_ERROR("Cannot enable both --eagle and --disaggregation at the same time.");
        return false;
    }
    if (args.disaggregation && args.tpcCount > 0)
    {
        auto const tpcLimit = static_cast<int32_t>(getJetsonThorTpcOrderFromGpcMasks().size());
        if (args.tpcCount >= tpcLimit)
        {
            LOG_ERROR("In disaggregation mode, --tpcCount must be in [1, %d) so prefill+encoding can use remaining TPCs.",
                tpcLimit);
            return false;
        }
    }

    return true;
}

std::pair<std::unordered_map<std::string, std::string>, std::vector<rt::LLMGenerationRequest>> parseInputFile(
    std::filesystem::path const& inputFilePath, int32_t batchSizeOverride = -1, int64_t maxGenerateLengthOverride = -1)
{
    std::vector<rt::LLMGenerationRequest> batchedRequests;

    Json inputData;
    std::ifstream inputFileStream(inputFilePath);
    if (!inputFileStream.is_open())
    {
        LOG_ERROR("Failed to open input file: %s", inputFilePath.string().c_str());
        throw std::runtime_error("Failed to open input file: " + inputFilePath.string());
    }
    try
    {
        inputData = Json::parse(inputFileStream);
        inputFileStream.close();
    }
    catch (Json::parse_error const& e)
    {
        LOG_ERROR("Failed to parse input file with error: %s", e.what());
        throw std::runtime_error("Failed to parse input file: " + inputFilePath.string());
    }

    // Extract global parameters
    int batchSize = (batchSizeOverride != -1) ? batchSizeOverride : inputData.value("batch_size", 1);
    if (batchSize <= 0)
    {
        LOG_ERROR("Invalid batch_size value: %d (must be positive)", batchSize);
        throw std::runtime_error("Invalid batch_size value (must be positive)");
    }

    float temperature = inputData.value("temperature", 1.0f);
    float topP = inputData.value("top_p", 0.8f);
    int64_t topK = inputData.value("top_k", 50);
    int64_t maxGenerateLength
        = (maxGenerateLengthOverride != -1) ? maxGenerateLengthOverride : inputData.value("max_generate_length", 256);
    if (maxGenerateLength <= 0)
    {
        LOG_ERROR(
            "Invalid max_generate_length value: %lld (must be positive)", static_cast<long long>(maxGenerateLength));
        throw std::runtime_error("Invalid max_generate_length value (must be positive)");
    }

    // Read apply_chat_template flag (defaults to true)
    bool applyChatTemplate = inputData.value("apply_chat_template", true);

    // Read add_generation_prompt flag (defaults to true)
    bool addGenerationPrompt = inputData.value("add_generation_prompt", true);

    // Read enable_thinking flag (defaults to false)
    bool enableThinking = inputData.value("enable_thinking", false);

    std::unordered_map<std::string, std::string> loraWeightsMap;
    if (inputData.contains("available_lora_weights") && inputData["available_lora_weights"].is_object())
    {
        auto const& availableLoraWeights = inputData["available_lora_weights"];
        for (auto const& [loraName, loraPath] : availableLoraWeights.items())
        {
            if (!loraPath.is_string())
            {
                LOG_ERROR("LoRA weight path for '%s' must be a string", loraName.c_str());
                throw std::runtime_error("LoRA weight path for '" + loraName + "' must be a string");
            }
            if (loraWeightsMap.find(loraName) != loraWeightsMap.end())
            {
                LOG_ERROR("Lora weights with name %s already exists", loraName.c_str());
                throw std::runtime_error("Lora weights with name " + loraName + " already exists");
            }
            loraWeightsMap[loraName] = loraPath.get<std::string>();
            LOG_INFO("Registered LoRA weights '%s' -> '%s'", loraName.c_str(), loraWeightsMap[loraName].c_str());
        }
    }

    // Parse requests and create batched requests
    if (inputData.contains("requests") && inputData["requests"].is_array())
    {
        auto& requestsArray = inputData["requests"];
        size_t numRequests = requestsArray.size();

        // Process requests in batches according to batchSize
        for (size_t startIdx = 0; startIdx < numRequests; startIdx += batchSize)
        {
            rt::LLMGenerationRequest batchRequest;
            batchRequest.temperature = temperature;
            batchRequest.topP = topP;
            batchRequest.topK = topK;
            batchRequest.maxGenerateLength = maxGenerateLength;
            batchRequest.applyChatTemplate = applyChatTemplate;
            batchRequest.addGenerationPrompt = addGenerationPrompt;
            batchRequest.enableThinking = enableThinking;

            // Track LoRA weights for validation
            std::string batchLoraWeightsName = "";
            bool firstInBatch = true;

            // Add requests to this batch (up to batchSize requests)
            size_t endIdx = std::min(startIdx + batchSize, numRequests);
            for (size_t requestIdx = startIdx; requestIdx < endIdx; ++requestIdx)
            {
                auto const& requestItem = requestsArray[requestIdx];

                // Each request must be an object with "messages" key
                if (!requestItem.is_object())
                {
                    LOG_ERROR("Each request must be an object with 'messages' key");
                    throw std::runtime_error("Each request must be an object with 'messages' key");
                }

                // Explicit query whether to save the system prompt KVCache of this message for later reuse.
                // This logic has limitation that once one prompt sets saveSystemPromptKVCache to true, all prompts in
                // the same batch will cache system prompt KVCache. Since long instruction cache saving is
                // usually done during system setup, this limitation can be resolved by issuing single batch request at
                // initialization stage for KVCache saving.
                bool saveSystemPromptKVCache = requestItem.value("save_system_prompt_kv_cache", false);
                if (saveSystemPromptKVCache)
                {
                    batchRequest.saveSystemPromptKVCache = true;
                }

                if (!requestItem.contains("messages") || !requestItem["messages"].is_array())
                {
                    LOG_ERROR("Each request object must contain a 'messages' array");
                    throw std::runtime_error("Each request object must contain a 'messages' array");
                }

                auto const& messagesArray = requestItem["messages"];

                // Get per-conversation LoRA name if present
                std::string requestLoraName = "";
                if (requestItem.contains("lora_name") && !requestItem["lora_name"].is_null())
                {
                    requestLoraName = requestItem["lora_name"].get<std::string>();

                    // Validate that the LoRA name exists in available_lora_weights
                    if (!requestLoraName.empty() && loraWeightsMap.find(requestLoraName) == loraWeightsMap.end())
                    {
                        LOG_ERROR("LoRA name '%s' not found in available_lora_weights", requestLoraName.c_str());
                        throw std::runtime_error(
                            "LoRA name '" + requestLoraName + "' not found in available_lora_weights");
                    }
                }

                // Validate that all requests in this batch use the same LoRA weights
                if (firstInBatch)
                {
                    batchLoraWeightsName = requestLoraName;
                    firstInBatch = false;
                }
                else
                {
                    if (requestLoraName != batchLoraWeightsName)
                    {
                        LOG_ERROR(
                            "All requests within the same batch must use the same LoRA weights. Batch has %d requests.",
                            static_cast<int>(endIdx - startIdx));
                        throw std::runtime_error("Different LoRA weights within the same batch are not supported");
                    }
                }

                // Parse messages into structured format
                std::vector<rt::Message> chatMessages;
                std::vector<rt::imageUtils::ImageData> imageBuffers;

                for (auto const& messageJson : messagesArray)
                {
                    if (!messageJson.contains("role") || !messageJson.contains("content"))
                    {
                        LOG_ERROR("Each message must have 'role' and 'content' fields");
                        throw std::runtime_error("Each message must have 'role' and 'content' fields");
                    }

                    rt::Message chatMsg;
                    chatMsg.role = messageJson["role"].get<std::string>();

                    auto const& contentJson = messageJson["content"];

                    // Support both string (simple text) and array (multimodal) formats
                    if (contentJson.is_string())
                    {
                        // Simple string format - treat as text content
                        rt::Message::MessageContent msgContent;
                        msgContent.type = "text";
                        msgContent.content = contentJson.get<std::string>();
                        chatMsg.contents.push_back(msgContent);
                    }
                    else if (contentJson.is_array())
                    {
                        // Array format - supports multimodal content
                        for (auto const& contentItemJson : contentJson)
                        {
                            if (!contentItemJson.contains("type"))
                            {
                                LOG_ERROR("Each content item must have a 'type' field");
                                throw std::runtime_error("Each content item must have a 'type' field");
                            }

                            rt::Message::MessageContent msgContent;
                            msgContent.type = contentItemJson["type"].get<std::string>();

                            // Based on type, extract the appropriate field
                            if (msgContent.type == "text")
                            {
                                msgContent.content = contentItemJson["text"].get<std::string>();
                            }
                            else if (msgContent.type == "image")
                            {
                                msgContent.content = contentItemJson["image"].get<std::string>();
                                // TODO: Need to consider multi-turn conversation, and whether to load all images.
                                auto image = rt::imageUtils::loadImageFromFile(msgContent.content);
                                if (image.buffer != nullptr)
                                {
                                    imageBuffers.push_back(std::move(image));
                                }
                            }
                            else
                            {
                                LOG_ERROR("Content type must be 'text', 'image', but got: %s", msgContent.type.c_str());
                                throw std::runtime_error(format::fmtstr(
                                    "Content type must be 'text', 'image', but got: %s", msgContent.type.c_str()));
                            }

                            chatMsg.contents.push_back(msgContent);
                        }
                    }
                    else
                    {
                        LOG_ERROR("Message content must be a string or an array");
                        throw std::runtime_error("Message content must be a string or an array");
                    }

                    chatMessages.push_back(chatMsg);
                }

                // Create prompt structure with structured messages
                rt::LLMGenerationRequest::Request request;
                request.messages = std::move(chatMessages);
                request.imageBuffers = std::move(imageBuffers);
                batchRequest.requests.push_back(std::move(request));
            }

            // Set the LoRA weights name for this batch (all requests in this batch use the same LoRA weights)
            if (!batchLoraWeightsName.empty())
            {
                batchRequest.loraWeightsName = batchLoraWeightsName;
            }

            batchedRequests.push_back(std::move(batchRequest));
        }
    }
    else
    {
        LOG_ERROR("'requests' array not found in input file");
        throw std::runtime_error("'requests' array not found in input file");
    }

    return std::make_pair(std::move(loraWeightsMap), std::move(batchedRequests));
}

int main(int argc, char* argv[])
{
    LLMInferenceArgs args;
    if (!parseLLMInferenceArgs(args, argc, argv))
    {
        printUsage(argv[0]);
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printUsage(argv[0]);
        return EXIT_SUCCESS;
    }
    bool benchmarkMode = args.benchmarkCount > 1;
    bool profilerEnabled = args.dumpProfile || !args.profileOutputFile.empty() || benchmarkMode;
    MemoryMonitor memoryMonitor;
    // Start memory monitoring at the beginning if profiling is enabled
    if (profilerEnabled)
    {
        memoryMonitor.start();
    }

    auto pluginHandles = loadEdgellmPluginLib();
    // load input file and parse to requests
    std::unordered_map<std::string, std::string> loraWeightsMap;
    std::vector<rt::LLMGenerationRequest> batchedRequests;
    try
    {
        std::tie(loraWeightsMap, batchedRequests)
            = parseInputFile(args.inputFile, args.batchSize, args.maxGenerateLength);
        LOG_INFO("Successfully parsed %zu LoRA weights from input file.", loraWeightsMap.size());
        LOG_INFO("Successfully parsed %zu batches of requests from input file.", batchedRequests.size());
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to parse input file: %s", e.what());
        return EXIT_FAILURE;
    }

    if (batchedRequests.empty())
    {
        LOG_ERROR("No valid requests found in input file.");
        return EXIT_FAILURE;
    }

    // Create runtime based on mode
    std::unique_ptr<rt::LLMInferenceRuntime> llmInferenceRuntime{nullptr};
    std::unique_ptr<rt::LLMInferenceDisaggregationRuntime> disaggregationRuntime{nullptr};
    std::unique_ptr<rt::LLMInferenceSpecDecodeRuntime> eagleInferenceRuntime{nullptr};
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    if (!args.disaggregation && args.tpcCount > 0 && !applyTpcMaskToStream(stream, args.tpcCount))
    {
        LOG_ERROR("Failed to apply TPC stream mask for tpcCount=%d", args.tpcCount);
        return EXIT_FAILURE;
    }

    if (args.eagleArgs.enabled)
    {
        // Eagle mode - LoRA is not supported
        if (!loraWeightsMap.empty())
        {
            LOG_WARNING("Eagle mode does not support LoRA weights. Ignoring LoRA weights.");
        }

        rt::EagleDraftingConfig draftingConfig{
            args.eagleArgs.draftTopK, args.eagleArgs.draftStep, args.eagleArgs.verifyTreeSize};
        try
        {
            eagleInferenceRuntime = std::make_unique<rt::LLMInferenceSpecDecodeRuntime>(
                args.engineDir, args.multimodalEngineDir, draftingConfig, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize LLMInferenceSpecDecodeRuntime: %s", e.what());
            return EXIT_FAILURE;
        }

        bool const draftProposalCaptureStatus = eagleInferenceRuntime->captureDraftProposalCudaGraph(stream);
        if (!draftProposalCaptureStatus)
        {
            LOG_WARNING(
                "Failed to capture CUDA graph for draft proposal usage, proceeding with normal engine execution.");
        }

        bool const draftAcceptCaptureStatus = eagleInferenceRuntime->captureDraftAcceptDecodeTokenCudaGraph(stream);
        if (!draftAcceptCaptureStatus)
        {
            LOG_WARNING(
                "Failed to capture CUDA graph for draft accept decode token usage, proceeding with normal engine "
                "execution.");
        }

        bool const baseCaptureStatus = eagleInferenceRuntime->captureBaseVerificationCudaGraph(stream);
        if (!baseCaptureStatus)
        {
            LOG_WARNING(
                "Failed to capture CUDA graph for base model verification usage, proceeding with normal engine "
                "execution.");
        }
    }
    else if (args.disaggregation)
    {
        try
        {
            disaggregationRuntime = std::make_unique<rt::LLMInferenceDisaggregationRuntime>(
                args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream, args.tpcCount,
                args.disaggDecodeCudaGraph);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize LLMInferenceDisaggregationRuntime: %s", e.what());
            return EXIT_FAILURE;
        }
        if (!disaggregationRuntime->captureDecodingCUDAGraph(stream))
        {
            LOG_WARNING("Failed to capture CUDA graph for disaggregation decode usage.");
        }
    }
    else
    {
        // Standard mode
        try
        {
            llmInferenceRuntime = std::make_unique<rt::LLMInferenceRuntime>(
                args.engineDir, args.multimodalEngineDir, loraWeightsMap, stream);
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to initialize LLMInferenceRuntime: %s", e.what());
            return EXIT_FAILURE;
        }
        if (!llmInferenceRuntime->captureDecodingCUDAGraph(stream))
        {
            LOG_WARNING("Failed to capture CUDA graph for decoding usage, proceeding with normal engine execution.");
        }
    }

    // Perform warmup runs if requested
    if (args.warmup > 0)
    {
        // Disable profiling for warmup runs
        setProfilingEnabled(false);
        LOG_INFO("Starting warmup with %d runs using the first request...", args.warmup);
        auto& firstRequest = batchedRequests[0];

        for (int32_t warmupRun = 0; warmupRun < args.warmup; ++warmupRun)
        {
            rt::LLMGenerationResponse warmupResponse;
            bool requestStatus = false;
            if (args.eagleArgs.enabled)
            {
                requestStatus = eagleInferenceRuntime->handleRequest(firstRequest, warmupResponse, stream);
            }
            else if (args.disaggregation)
            {
                auto warmupFuture = disaggregationRuntime->submitRequestAsync(firstRequest);
                auto warmupResult = warmupFuture.get();
                warmupResponse = std::move(warmupResult.response);
                requestStatus = warmupResult.success;
            }
            else
            {
                requestStatus = llmInferenceRuntime->handleRequest(firstRequest, warmupResponse, stream);
            }

            if (!requestStatus)
            {
                LOG_ERROR("Warmup run %d/%d failed", warmupRun + 1, args.warmup);
                return EXIT_FAILURE;
            }
        }
        LOG_INFO("Warmup of %d runs completed. Starting actual benchmark runs...", args.warmup);
    }

    if (profilerEnabled)
    {
        setProfilingEnabled(true);
    }

    // Structure to collect all responses for JSON export
    nlohmann::json outputData;
    outputData["input_file"] = args.inputFile;
    outputData["responses"] = nlohmann::json::array();

    bool hasFailedRequest = false;
    std::string errorMessage = "TensorRT Edge LLM cannot handle this request. Fails.";
    size_t failedCount = 0;

    size_t totalRequests = batchedRequests.size() * static_cast<size_t>(args.benchmarkCount);
    size_t processedRequests = 0;
    LOG_INFO("Processing %zu batched requests (%d repetition(s))...", totalRequests, args.benchmarkCount);
    if (args.disaggregation)
    {
        struct PendingRequest
        {
            size_t globalRequestIdx{0};
            rt::LLMGenerationRequest* request{nullptr};
            std::future<rt::LLMInferenceDisaggregationRuntime::AsyncRequestResult> future;
        };

        std::vector<PendingRequest> pendingRequests;
        pendingRequests.reserve(totalRequests);
        for (int32_t repetition = 0; repetition < args.benchmarkCount; ++repetition)
        {
            for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
            {
                auto& request = batchedRequests[requestIdx];
                size_t globalRequestIdx = static_cast<size_t>(repetition) * batchedRequests.size() + requestIdx;
                PendingRequest pending;
                pending.globalRequestIdx = globalRequestIdx;
                pending.request = &request;
                pending.future = disaggregationRuntime->submitRequestAsync(request);
                pendingRequests.emplace_back(std::move(pending));
            }
        }

        for (auto& pending : pendingRequests)
        {
            auto result = pending.future.get();
            auto& request = *pending.request;
            auto& response = result.response;
            bool const requestStatus = result.success;
            ++processedRequests;

            size_t progressInterval = std::max(size_t(1), std::min(totalRequests / 10, size_t(100)));
            if (processedRequests % progressInterval == 0 || processedRequests == 1 || processedRequests == totalRequests)
            {
                LOG_INFO("Progress: %zu/%zu (%f%%)", processedRequests, totalRequests,
                    100.0 * processedRequests / totalRequests);
            }

            if (requestStatus)
            {
                if (args.dumpOutput)
                {
                    for (size_t batchIdx = 0; batchIdx < response.outputTexts.size(); ++batchIdx)
                    {
                        LOG_INFO("Response for request %zu batch %zu: %s", pending.globalRequestIdx, batchIdx,
                            response.outputTexts[batchIdx].c_str());
                    }
                }
            }
            else
            {
                hasFailedRequest = true;
                failedCount++;
                LOG_ERROR("*** FAILED *** Request %zu failed to process!", pending.globalRequestIdx);
            }

            for (size_t batchIdx = 0; batchIdx < request.requests.size(); ++batchIdx)
            {
                nlohmann::json responseJson;
                std::string outputText = requestStatus ? response.outputTexts[batchIdx] : errorMessage;
                responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
                responseJson["request_idx"] = pending.globalRequestIdx;
                responseJson["batch_idx"] = batchIdx;
                nlohmann::json messagesJson = nlohmann::json::array();
                for (auto const& msg : request.requests[batchIdx].messages)
                {
                    nlohmann::json msgJson;
                    msgJson["role"] = msg.role;
                    msgJson["content"] = nlohmann::json::array();
                    for (auto const& content : msg.contents)
                    {
                        nlohmann::json contentJson;
                        contentJson["type"] = content.type;
                        if (content.type == "text")
                        {
                            contentJson["text"] = content.content;
                        }
                        else if (content.type == "image")
                        {
                            contentJson["image"] = content.content;
                        }
                        else if (content.type == "video")
                        {
                            contentJson["video"] = content.content;
                        }
                        msgJson["content"].push_back(contentJson);
                    }
                    messagesJson.push_back(msgJson);
                }
                responseJson["messages"] = messagesJson;
                responseJson["formatted_system_prompt"] = request.formattedRequests[batchIdx].formattedSystemPrompt;
                responseJson["formatted_complete_request"] = request.formattedRequests[batchIdx].formattedCompleteRequest;
                outputData["responses"].push_back(responseJson);
            }
        }
    }
    else for (int32_t repetition = 0; repetition < args.benchmarkCount; ++repetition)
    {
        for (size_t requestIdx = 0; requestIdx < batchedRequests.size(); ++requestIdx)
        {
            auto& request = batchedRequests[requestIdx];
            rt::LLMGenerationResponse response;
            size_t globalRequestIdx = static_cast<size_t>(repetition) * batchedRequests.size() + requestIdx;
            ++processedRequests;

            // Show progress every 10% or every 100 requests, whichever is smaller
            size_t progressInterval = std::max(size_t(1), std::min(totalRequests / 10, size_t(100)));
            if (processedRequests % progressInterval == 0 || processedRequests == 1 || processedRequests == totalRequests)
            {
                LOG_INFO("Progress: %zu/%zu (%f%%)", processedRequests, totalRequests,
                    100.0 * processedRequests / totalRequests);
            }

            bool requestStatus = false;
            if (args.eagleArgs.enabled)
            {
                requestStatus = eagleInferenceRuntime->handleRequest(request, response, stream);
            }
            else
            {
                requestStatus = llmInferenceRuntime->handleRequest(request, response, stream);
            }

            if (requestStatus)
            {
                // Display inference output to console if --dumpOutput is enabled
                if (args.dumpOutput)
                {
                    for (size_t batchIdx = 0; batchIdx < response.outputTexts.size(); ++batchIdx)
                    {
                        LOG_INFO("Response for request %zu batch %zu: %s", globalRequestIdx, batchIdx,
                            response.outputTexts[batchIdx].c_str());
                    }
                }
            }
            else
            {
                // Handle failed request - highlight failures
                hasFailedRequest = true;
                failedCount++;
                LOG_ERROR("*** FAILED *** Request %zu failed to process!", globalRequestIdx);
            }

            // Add to JSON output with UTF-8 validation on output text
            for (size_t batchIdx = 0; batchIdx < request.requests.size(); ++batchIdx)
            {
                nlohmann::json responseJson;
                std::string outputText = requestStatus ? response.outputTexts[batchIdx] : errorMessage;
                // Validate UTF-8 for output text (inputs are always valid)
                // If invalid UTF-8 detected, error message is returned and original text is logged
                responseJson["output_text"] = sanitizeUtf8ForJson(outputText);
                responseJson["request_idx"] = globalRequestIdx;
                responseJson["batch_idx"] = batchIdx;
                // Store messages for reference
                nlohmann::json messagesJson = nlohmann::json::array();
                for (auto const& msg : request.requests[batchIdx].messages)
                {
                    nlohmann::json msgJson;
                    msgJson["role"] = msg.role;
                    msgJson["content"] = nlohmann::json::array();
                    for (auto const& content : msg.contents)
                    {
                        nlohmann::json contentJson;
                        contentJson["type"] = content.type;
                        if (content.type == "text")
                        {
                            contentJson["text"] = content.content;
                        }
                        else if (content.type == "image")
                        {
                            contentJson["image"] = content.content;
                        }
                        else if (content.type == "video")
                        {
                            contentJson["video"] = content.content;
                        }
                        msgJson["content"].push_back(contentJson);
                    }
                    messagesJson.push_back(msgJson);
                }
                responseJson["messages"] = messagesJson;
                // Store formatted prompts for reference
                responseJson["formatted_system_prompt"] = request.formattedRequests[batchIdx].formattedSystemPrompt;
                responseJson["formatted_complete_request"] = request.formattedRequests[batchIdx].formattedCompleteRequest;
                outputData["responses"].push_back(responseJson);
            }
        }
    }

    // Final processing summary
    LOG_INFO("Processing complete: %zu/%zu batched requests successful", totalRequests - failedCount, totalRequests);
    if (failedCount > 0)
    {
        LOG_ERROR("*** %zu BATCHED REQUESTS FAILED ***", failedCount);
    }

    if (profilerEnabled)
    {
        // Stop memory monitoring for examples
        setProfilingEnabled(false);
        memoryMonitor.stop();
    }

    if (args.dumpProfile)
    {
        std::ostringstream profileOutput;
        profileOutput << std::endl;
        profileOutput << "=== Performance Summary ===" << std::endl;
        if (args.eagleArgs.enabled)
        {
            // Eagle runtime with detailed metrics
            auto prefillMetrics = eagleInferenceRuntime->getPrefillMetrics();
            auto eagleGenerationMetrics = eagleInferenceRuntime->getEagleGenerationMetrics();
            auto multimodalMetrics = eagleInferenceRuntime->getMultimodalMetrics();
            outputPrefillProfile(profileOutput, prefillMetrics);
            outputEagleGenerationProfile(profileOutput, eagleGenerationMetrics);
            outputMultimodalProfile(profileOutput, multimodalMetrics);
            outputMemoryProfile(profileOutput, memoryMonitor);
        }
        else if (args.disaggregation)
        {
            auto multimodalMetrics = disaggregationRuntime->getMultimodalMetrics();
            outputPrefillProfile(profileOutput, disaggregationRuntime->getPrefillMetrics());
            outputGenerationProfile(profileOutput, disaggregationRuntime->getGenerationMetrics());
            outputMultimodalProfile(profileOutput, multimodalMetrics);
            outputMemoryProfile(profileOutput, memoryMonitor);
        }
        else
        {
            auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
            outputPrefillProfile(profileOutput, llmInferenceRuntime->getPrefillMetrics());
            outputGenerationProfile(profileOutput, llmInferenceRuntime->getGenerationMetrics());
            outputMultimodalProfile(profileOutput, multimodalMetrics);
            outputMemoryProfile(profileOutput, memoryMonitor);
        }
        profileOutput << "=====================================" << std::endl;
        LOG_INFO("%s", profileOutput.str().c_str());
    }

    if (benchmarkMode)
    {
        nlohmann::json profileJson;
        if (args.eagleArgs.enabled)
        {
            auto prefillMetrics = eagleInferenceRuntime->getPrefillMetrics();
            auto eagleGenerationMetrics = eagleInferenceRuntime->getEagleGenerationMetrics();
            auto multimodalMetrics = eagleInferenceRuntime->getMultimodalMetrics();
            addJsonPrefillSummary(profileJson, prefillMetrics);
            addJsonEagleGenerationSummary(profileJson, eagleGenerationMetrics);
            addJsonMultimodalSummary(profileJson, multimodalMetrics);
            addJsonTimingStages(profileJson);
        }
        else if (args.disaggregation)
        {
            auto multimodalMetrics = disaggregationRuntime->getMultimodalMetrics();
            addJsonPrefillSummary(profileJson, disaggregationRuntime->getPrefillMetrics());
            addJsonGenerationSummary(profileJson, disaggregationRuntime->getGenerationMetrics());
            addJsonMultimodalSummary(profileJson, multimodalMetrics);
            addJsonTimingStages(profileJson);
        }
        else
        {
            auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();
            addJsonPrefillSummary(profileJson, llmInferenceRuntime->getPrefillMetrics());
            addJsonGenerationSummary(profileJson, llmInferenceRuntime->getGenerationMetrics());
            addJsonMultimodalSummary(profileJson, multimodalMetrics);
            addJsonTimingStages(profileJson);
        }

        auto getStageMetric = [&](std::string const& stageId, std::string const& metricName) -> double {
            if (!profileJson.contains("stages") || !profileJson["stages"].is_array())
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            for (auto const& stage : profileJson["stages"])
            {
                if (!stage.contains("stage_id") || !stage["stage_id"].is_string()
                    || stage["stage_id"].get<std::string>() != stageId)
                {
                    continue;
                }
                if (metricName == "avg_mean_ms")
                {
                    if (stage.contains("gpu_time_stats") && stage["gpu_time_stats"].contains("mean_ms"))
                    {
                        return stage["gpu_time_stats"]["mean_ms"].get<double>();
                    }
                    if (stage.contains("average_time_per_run_ms"))
                    {
                        return stage["average_time_per_run_ms"].get<double>();
                    }
                }
                if (metricName == "p99_ms" && stage.contains("gpu_time_stats") && stage["gpu_time_stats"].contains("p99_ms"))
                {
                    return stage["gpu_time_stats"]["p99_ms"].get<double>();
                }
                if (metricName == "median_ms" && stage.contains("gpu_time_stats")
                    && stage["gpu_time_stats"].contains("median_ms"))
                {
                    return stage["gpu_time_stats"]["median_ms"].get<double>();
                }
            }
            return std::numeric_limits<double>::quiet_NaN();
        };

        auto getStageSampleCount = [&](std::string const& stageId) -> int64_t {
            if (!profileJson.contains("stages") || !profileJson["stages"].is_array())
            {
                return -1;
            }
            for (auto const& stage : profileJson["stages"])
            {
                if (!stage.contains("stage_id") || !stage["stage_id"].is_string()
                    || stage["stage_id"].get<std::string>() != stageId)
                {
                    continue;
                }
                if (stage.contains("gpu_time_stats") && stage["gpu_time_stats"].contains("count")
                    && stage["gpu_time_stats"]["count"].is_number_integer())
                {
                    return stage["gpu_time_stats"]["count"].get<int64_t>();
                }
                if (stage.contains("total_runs") && stage["total_runs"].is_number_integer())
                {
                    return stage["total_runs"].get<int64_t>();
                }
            }
            return -1;
        };

        auto getStageTotalGpuTimeMs = [&](std::string const& stageId) -> double {
            if (!profileJson.contains("stages") || !profileJson["stages"].is_array())
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            for (auto const& stage : profileJson["stages"])
            {
                if (!stage.contains("stage_id") || !stage["stage_id"].is_string()
                    || stage["stage_id"].get<std::string>() != stageId)
                {
                    continue;
                }
                if (stage.contains("total_gpu_time_ms") && stage["total_gpu_time_ms"].is_number())
                {
                    return stage["total_gpu_time_ms"].get<double>();
                }
            }
            return std::numeric_limits<double>::quiet_NaN();
        };

        auto getDecodeTokenSampleCount = [&]() -> int64_t {
            if (profileJson.contains("generation") && profileJson["generation"].contains("generated_tokens")
                && profileJson["generation"]["generated_tokens"].is_number_integer())
            {
                return profileJson["generation"]["generated_tokens"].get<int64_t>();
            }
            return -1;
        };

        auto safeDivide = [](double numerator, double denominator) -> double {
            if (std::isnan(numerator) || std::isinf(numerator) || std::isnan(denominator) || std::isinf(denominator)
                || denominator <= 0.0)
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return numerator / denominator;
        };

        auto getTotalResponseCount = [&]() -> int64_t {
            if (outputData.contains("responses") && outputData["responses"].is_array())
            {
                return static_cast<int64_t>(outputData["responses"].size());
            }
            return -1;
        };

        auto getStageRequestMetric = [&](std::string const& stageId, std::string const& metricName) -> double {
            int64_t totalResponses = getTotalResponseCount();
            int64_t stageRuns = getStageSampleCount(stageId);
            if (totalResponses <= 0 || stageRuns <= 0)
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            double requestsPerRun = static_cast<double>(totalResponses) / static_cast<double>(stageRuns);
            if (metricName == "mean_ms")
            {
                return safeDivide(getStageMetric(stageId, "avg_mean_ms"), requestsPerRun);
            }
            if (metricName == "median_ms")
            {
                return safeDivide(getStageMetric(stageId, "median_ms"), requestsPerRun);
            }
            if (metricName == "p99_ms")
            {
                return safeDivide(getStageMetric(stageId, "p99_ms"), requestsPerRun);
            }
            return std::numeric_limits<double>::quiet_NaN();
        };

        auto getRequestTotalMetric = [&](std::string const& metricName) -> double {
            double encoding = getStageRequestMetric("multimodal_processing", metricName);
            double prefill = getStageRequestMetric("llm_prefill", metricName);
            double decode = getStageRequestMetric("llm_generation", metricName);
            if (std::isnan(encoding) || std::isnan(prefill) || std::isnan(decode) || std::isinf(encoding)
                || std::isinf(prefill) || std::isinf(decode))
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return encoding + prefill + decode;
        };

        auto getRequestTotalThroughput = [&]() -> double {
            int64_t totalResponses = getTotalResponseCount();
            double totalPipelineTimeMs = getStageTotalGpuTimeMs("multimodal_processing")
                + getStageTotalGpuTimeMs("llm_prefill") + getStageTotalGpuTimeMs("llm_generation");
            if (totalResponses <= 0 || std::isnan(totalPipelineTimeMs) || std::isinf(totalPipelineTimeMs)
                || totalPipelineTimeMs <= 0.0)
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return static_cast<double>(totalResponses) / (totalPipelineTimeMs / 1000.0);
        };

        auto getDecodeTokenMetric = [&](std::string const& metricName) -> double {
            if (args.eagleArgs.enabled || !profileJson.contains("generation"))
            {
                return std::numeric_limits<double>::quiet_NaN();
            }

            auto const& generation = profileJson["generation"];
            if (metricName == "mean_ms")
            {
                if (generation.contains("average_time_per_token_ms"))
                {
                    return generation["average_time_per_token_ms"].get<double>();
                }
                double stageMean = getStageMetric("llm_generation", "avg_mean_ms");
                double avgTokensPerRun = generation.value("average_tokens_per_run", 0.0);
                return safeDivide(stageMean, avgTokensPerRun);
            }

            double avgTokensPerRun = generation.value("average_tokens_per_run", 0.0);
            if (metricName == "median_ms")
            {
                return safeDivide(getStageMetric("llm_generation", "median_ms"), avgTokensPerRun);
            }
            if (metricName == "p99_ms")
            {
                return safeDivide(getStageMetric("llm_generation", "p99_ms"), avgTokensPerRun);
            }
            return std::numeric_limits<double>::quiet_NaN();
        };

        auto getThroughput = [&](std::string const& label) -> double {
            if (label == "prefill" && profileJson.contains("prefill") && profileJson["prefill"].contains("tokens_per_second"))
            {
                return profileJson["prefill"]["tokens_per_second"].get<double>();
            }
            if (label == "decode" && profileJson.contains("generation")
                && profileJson["generation"].contains("tokens_per_second"))
            {
                return profileJson["generation"]["tokens_per_second"].get<double>();
            }
            if (label == "encoding" && profileJson.contains("multimodal")
                && profileJson["multimodal"].contains("average_time_per_token_ms"))
            {
                double msPerToken = profileJson["multimodal"]["average_time_per_token_ms"].get<double>();
                if (msPerToken > 0.0)
                {
                    return 1000.0 / msPerToken;
                }
            }
            return std::numeric_limits<double>::quiet_NaN();
        };

        auto fmt = [](double value) -> std::string {
            if (std::isnan(value) || std::isinf(value))
            {
                return "n/a";
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << value;
            return oss.str();
        };

        auto fmtCount = [](int64_t value) -> std::string {
            if (value < 0)
            {
                return "n/a";
            }
            return std::to_string(value);
        };

        std::ostringstream benchOutput;
        benchOutput << "\n=== Benchmark Summary ===\n";
        benchOutput << "count: " << args.benchmarkCount << "\n";
        benchOutput << std::left << std::setw(10) << "stage" << std::right << std::setw(10) << "samples"
                    << std::setw(14) << "mean_ms"
                    << std::setw(12) << "median_ms" << std::setw(12) << "p99_ms" << std::setw(20) << "throughput\n";
        benchOutput << "----------------------------------------------------------------------------\n";
        benchOutput << std::left << std::setw(10) << "encoding" << std::right
                    << std::setw(10) << fmtCount(getStageSampleCount("multimodal_processing"))
                    << std::setw(14) << fmt(getStageMetric("multimodal_processing", "avg_mean_ms"))
                    << std::setw(12) << fmt(getStageMetric("multimodal_processing", "median_ms"))
                    << std::setw(12) << fmt(getStageMetric("multimodal_processing", "p99_ms")) << std::setw(20)
                    << (fmt(getThroughput("encoding")) + " imgtok/s") << "\n";
        benchOutput << std::left << std::setw(10) << "prefill" << std::right
                    << std::setw(10) << fmtCount(getStageSampleCount("llm_prefill"))
                    << std::setw(14) << fmt(getStageMetric("llm_prefill", "avg_mean_ms"))
                    << std::setw(12) << fmt(getStageMetric("llm_prefill", "median_ms"))
                    << std::setw(12) << fmt(getStageMetric("llm_prefill", "p99_ms")) << std::setw(20)
                    << (fmt(getThroughput("prefill")) + " tok/s") << "\n";
        benchOutput << std::left << std::setw(10) << "decode" << std::right
                    << std::setw(10) << fmtCount(getDecodeTokenSampleCount())
                    << std::setw(14) << fmt(getDecodeTokenMetric("mean_ms"))
                    << std::setw(12) << fmt(getDecodeTokenMetric("median_ms"))
                    << std::setw(12) << fmt(getDecodeTokenMetric("p99_ms")) << std::setw(20)
                    << (fmt(getThroughput("decode")) + " tok/s") << "\n";
        benchOutput << std::left << std::setw(10) << "request" << std::right
                    << std::setw(10) << fmtCount(getTotalResponseCount())
                    << std::setw(14) << fmt(getRequestTotalMetric("mean_ms"))
                    << std::setw(12) << fmt(getRequestTotalMetric("median_ms"))
                    << std::setw(12) << fmt(getRequestTotalMetric("p99_ms")) << std::setw(20)
                    << (fmt(getRequestTotalThroughput()) + " req/s") << "\n";
        benchOutput << "decode metrics are per-token latency (ms/token)\n";
        benchOutput << "request row is end-to-end (encoding+prefill+decode) per request\n";
        benchOutput << "=======================================\n";
        if (args.quiet)
        {
            std::cout << benchOutput.str();
        }
        else
        {
            LOG_INFO("%s", benchOutput.str().c_str());
        }
    }

    // Export profile to JSON file
    if (!args.profileOutputFile.empty())
    {
        try
        {
            nlohmann::json profileJson;

            if (args.eagleArgs.enabled)
            {
                // Eagle runtime with detailed metrics
                auto prefillMetrics = eagleInferenceRuntime->getPrefillMetrics();
                auto eagleGenerationMetrics = eagleInferenceRuntime->getEagleGenerationMetrics();
                auto multimodalMetrics = eagleInferenceRuntime->getMultimodalMetrics();

                // Add high-level metrics
                addJsonPrefillSummary(profileJson, prefillMetrics);
                addJsonEagleGenerationSummary(profileJson, eagleGenerationMetrics);
                addJsonMultimodalSummary(profileJson, multimodalMetrics);

                // Add detailed timing stages
                addJsonTimingStages(profileJson);

                // Add memory usage
                addJsonMemorySummary(profileJson, memoryMonitor);
            }
            else
            {
                auto multimodalMetrics = llmInferenceRuntime->getMultimodalMetrics();

                // Add high-level metrics
                addJsonPrefillSummary(profileJson, llmInferenceRuntime->getPrefillMetrics());
                addJsonGenerationSummary(profileJson, llmInferenceRuntime->getGenerationMetrics());
                addJsonMultimodalSummary(profileJson, multimodalMetrics);

                // Add detailed timing stages
                addJsonTimingStages(profileJson);

                // Add memory usage
                addJsonMemorySummary(profileJson, memoryMonitor);
            }

            std::ofstream profileFile(args.profileOutputFile);
            if (profileFile.is_open())
            {
                profileFile << profileJson.dump(2); // Pretty print with 2 space indentation
                profileFile.close();
                LOG_INFO("Profile data exported to: %s", args.profileOutputFile.c_str());
            }
            else
            {
                LOG_ERROR("Failed to open profile output file: %s", args.profileOutputFile.c_str());
                return EXIT_FAILURE;
            }
        }
        catch (std::exception const& e)
        {
            LOG_ERROR("Failed to write profile output file: %s", e.what());
            return EXIT_FAILURE;
        }
    }

    // Export to JSON file
    try
    {
        std::ofstream outputFile(args.outputFile);
        if (outputFile.is_open())
        {
            outputFile << outputData.dump(4); // Pretty print with 4 spaces indentation
            outputFile.close();
            LOG_INFO("All responses exported to: %s", args.outputFile.c_str());
        }
        else
        {
            LOG_ERROR("Failed to open output file: %s", args.outputFile.c_str());
            return EXIT_FAILURE;
        }
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to write output file: %s", e.what());
        return EXIT_FAILURE;
    }

    // Return false if any request failed
    return hasFailedRequest ? EXIT_FAILURE : EXIT_SUCCESS;
}
