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

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "kernels/posEncoding/initializeCosSinCache.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/imageUtils.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/scheduling/independentEngineExecutorPair.h"
#include "runtime/scheduling/independentPhaseAsyncServer.h"
#include "runtime/scheduling/independentPhaseCoordinator.h"
#include "runtime/scheduling/phaseContinuousLoadGenerator.h"
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/scheduling/phaseKVActiveView.h"
#include "runtime/scheduling/phaseThreeCoordinator.h"
#include "runtime/scheduling/phaseVisionAdapter.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "sampler/sampling.h"
#include "tokenizer/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kDEFAULT_STABLE_SLOTS = 80;
constexpr size_t kDEFAULT_MAX_INFLIGHT_REQUESTS = 16U;

struct PhaseIpcInput
{
    uint64_t requestId{};
    bool cancel{};
    bool valid{true};
    std::string error;
    rt::LLMGenerationRequest::Request request;
    int32_t maxOutputTokens{};
    rt::PhaseSchedulingHints scheduling;
    double adapterUs{};
};

PhaseIpcInput parsePhaseIpcInput(std::string const& line, int32_t defaultMaxOutputTokens)
{
    PhaseIpcInput result;
    result.maxOutputTokens = defaultMaxOutputTokens;
    try
    {
        nlohmann::json const payload = nlohmann::json::parse(line);
        result.requestId = payload.value("request_index", uint64_t{});
        result.cancel = payload.value("type", "submit") == "cancel";
        if (result.cancel)
        {
            return result;
        }
        nlohmann::json const requestPayload = payload.contains("request") ? payload.at("request") : payload;
        if (requestPayload.contains("messages") && requestPayload.at("messages").is_array())
        {
            for (auto const& message : requestPayload.at("messages"))
            {
                rt::Message parsedMessage;
                parsedMessage.role = message.value("role", "user");
                if (message.contains("content") && message.at("content").is_string())
                {
                    parsedMessage.contents.push_back({"text", message.at("content").get<std::string>()});
                }
                else if (message.contains("content") && message.at("content").is_array())
                {
                    for (auto const& part : message.at("content"))
                    {
                        if (part.value("type", "") == "text" && part.contains("text"))
                        {
                            parsedMessage.contents.push_back({"text", part.at("text").get<std::string>()});
                        }
                        else if (part.value("type", "") == "image_url" && part.contains("image_url"))
                        {
                            nlohmann::json const& imageUrl = part.at("image_url");
                            std::string path
                                = imageUrl.is_string() ? imageUrl.get<std::string>() : imageUrl.value("url", "");
                            std::string const fileScheme = "file://";
                            if (path.compare(0, fileScheme.size(), fileScheme) == 0)
                            {
                                path.erase(0, fileScheme.size());
                            }
                            if (path.empty() || path.find("://") != std::string::npos)
                            {
                                result.valid = false;
                                result.error = "phase backend accepts local file image_url paths";
                                return result;
                            }
                            parsedMessage.contents.push_back({"image", ""});
                            result.request.imageBuffers.push_back(rt::imageUtils::loadImageFromFile(path));
                        }
                    }
                }
                result.request.messages.push_back(std::move(parsedMessage));
            }
        }
        if (requestPayload.contains("max_output_tokens"))
        {
            result.maxOutputTokens = requestPayload.at("max_output_tokens").get<int32_t>();
        }
        else if (requestPayload.contains("max_tokens"))
        {
            result.maxOutputTokens = requestPayload.at("max_tokens").get<int32_t>();
        }
        else if (requestPayload.contains("max_generate_length"))
        {
            result.maxOutputTokens = requestPayload.at("max_generate_length").get<int32_t>();
        }
        if (requestPayload.contains("metadata") && requestPayload.at("metadata").is_object())
        {
            nlohmann::json const& metadata = requestPayload.at("metadata");
            nlohmann::json const& phaseScheduling
                = metadata.contains("phase_scheduling") ? metadata.at("phase_scheduling") : metadata;
            result.scheduling.priority = phaseScheduling.value("priority", 0);
            result.scheduling.ttftTargetUs = phaseScheduling.value("ttft_target_ms", 0.0) * 1000.0;
            result.scheduling.tpotTargetUs = phaseScheduling.value("tpot_target_ms", 0.0) * 1000.0;
        }
    }
    catch (std::exception const& exception)
    {
        result.valid = false;
        result.error = exception.what();
    }
    return result;
}

rt::PhasePrefillClass parsePrefillClass(nlohmann::json const& point)
{
    std::string const value = point.value("prefill_class", std::string{"any"});
    if (value == "any")
    {
        return rt::PhasePrefillClass::kAny;
    }
    if (value == "text")
    {
        return rt::PhasePrefillClass::kText;
    }
    if (value == "external")
    {
        return rt::PhasePrefillClass::kExternal;
    }
    ELLM_CHECK(false, "Unknown phase prefill class: " + value);
    return rt::PhasePrefillClass::kAny;
}

std::vector<rt::IndependentPhaseAdmissionCost> parseAdmissionCosts(std::string const& value)
{
    std::vector<rt::IndependentPhaseAdmissionCost> result;
    std::stringstream entries(value);
    std::string entry;
    while (std::getline(entries, entry, ','))
    {
        size_t const separator = entry.find(':');
        ELLM_CHECK(separator != std::string::npos && separator > 0 && separator + 1U < entry.size(),
            "Phase admission cost must use limit:tpot_us entries");
        result.push_back(
            {static_cast<size_t>(std::stoul(entry.substr(0, separator))), std::stod(entry.substr(separator + 1U))});
    }
    ELLM_CHECK(!result.empty(), "Phase admission cost table cannot be empty");
    return result;
}

void loadSchedulerCostModel(std::filesystem::path const& path, rt::PhaseQueueSchedulerConfig& config)
{
    std::ifstream stream(path);
    ELLM_CHECK(stream.good(), "Failed to open phase scheduler cost model: " + path.string());
    nlohmann::json const root = nlohmann::json::parse(stream);
    ELLM_CHECK(root.value("prefill_layout", std::string{}) == "packed",
        "Phase scheduler cost model must use packed prefill layout");
    config.decodeBatchCosts.clear();
    for (nlohmann::json const& point : root.at("decode"))
    {
        config.decodeBatchCosts.push_back(
            {point.at("batch_size").get<int32_t>(), point.at("max_context_length").get<int32_t>(),
                point.at("p95_gpu_ms").get<float>(), point.value("max_total_context_tokens", int64_t{})});
    }
    for (nlohmann::json const& point : root.at("prefill"))
    {
        config.prefillBatchCosts.push_back({point.at("batch_size").get<int32_t>(),
            point.at("chunk_length").get<int32_t>(), point.at("max_past_kv_length").get<int32_t>(),
            point.at("max_concurrent_decode_batch_size").get<int32_t>(), point.at("initial_chunk").get<bool>(),
            point.at("p95_gpu_ms").get<float>(), point.at("decode_slowdown_p95_ms").get<float>(),
            parsePrefillClass(point)});
    }
    for (nlohmann::json const& point : root.at("overlap"))
    {
        config.overlapBatchCosts.push_back({point.at("prefill_batch_size").get<int32_t>(),
            point.at("decode_batch_size").get<int32_t>(), point.at("chunk_length").get<int32_t>(),
            point.at("max_prefill_past_kv_length").get<int32_t>(), point.at("max_decode_context_length").get<int32_t>(),
            point.at("initial_chunk").get<bool>(), point.at("prefill_p95_gpu_ms").get<float>(),
            point.at("decode_p95_gpu_ms").get<float>(), point.at("makespan_p95_gpu_ms").get<float>(),
            point.at("decode_slowdown_p95_ms").get<float>(), parsePrefillClass(point)});
    }
    ELLM_CHECK(
        !config.decodeBatchCosts.empty() && !config.prefillBatchCosts.empty() && !config.overlapBatchCosts.empty(),
        "Phase scheduler cost model is incomplete");
    config.profile = rt::PhaseSchedulerProfile::kThroughputBalanced;
}

void loadPrefillFormationCostModel(std::filesystem::path const& path, rt::IndependentPhaseServerConfig& config)
{
    std::ifstream stream(path);
    ELLM_CHECK(stream.good(), "Failed to open prefill formation cost model: " + path.string());
    nlohmann::json const root = nlohmann::json::parse(stream);
    ELLM_CHECK(root.contains("formation") && root.at("formation").is_array(),
        "Prefill formation cost model has no formation table");
    config.prefillFormationCosts.clear();
    for (nlohmann::json const& point : root.at("formation"))
    {
        config.prefillFormationCosts.push_back({point.at("max_prompt_tokens").get<int32_t>(),
            point.at("max_output_tokens").get<int32_t>(), point.at("max_decode_rows").get<size_t>(),
            point.value("max_decode_tpot_pressure", 0.0F), point.at("target_batch_size").get<size_t>(),
            point.at("window_us").get<double>(), point.value("ttft_target_us", 0.0),
            point.value("ttft_guard_us", 1000.0), point.at("throughput_gain_pct").get<double>()});
    }
    ELLM_CHECK(!config.prefillFormationCosts.empty(), "Prefill formation cost table cannot be empty");
}

void loadEncoderCostModel(std::filesystem::path const& path, rt::PhaseThreeCoordinatorConfig& config)
{
    std::ifstream stream(path);
    ELLM_CHECK(stream.good(), "Failed to open phase encoder cost model: " + path.string());
    nlohmann::json const root = nlohmann::json::parse(stream);
    ELLM_CHECK(
        root.contains("encoder") && root.at("encoder").is_array(), "Phase encoder cost model has no encoder table");
    config.encoderBatchCosts.clear();
    for (nlohmann::json const& point : root.at("encoder"))
    {
        size_t const batchSize = point.at("batch_size").get<size_t>();
        if (batchSize <= config.maxEncoderBatchSize)
        {
            config.encoderBatchCosts.push_back(
                {batchSize, point.at("max_input_tokens").get<size_t>(), point.at("p95_gpu_ms").get<float>()});
        }
    }
    ELLM_CHECK(!config.encoderBatchCosts.empty(), "Phase encoder cost table cannot be empty");
    config.globalEncoderPrefillCosts.clear();
    if (root.contains("encoder_prefill"))
    {
        ELLM_CHECK(root.at("encoder_prefill").is_array(), "Phase E+P cost table must be an array");
        for (nlohmann::json const& point : root.at("encoder_prefill"))
        {
            config.globalEncoderPrefillCosts.push_back({point.at("encoder_batch_size").get<size_t>(),
                point.at("prefill_batch_size").get<int32_t>(), point.at("max_encoder_input_tokens").get<size_t>(),
                point.at("max_prefill_chunk_length").get<int32_t>(),
                point.at("max_prefill_past_kv_length").get<int32_t>(), point.at("makespan_p95_gpu_ms").get<float>()});
        }
    }
    config.globalEncoderDecodeCosts.clear();
    if (root.contains("encoder_decode"))
    {
        ELLM_CHECK(root.at("encoder_decode").is_array(), "Phase E+D cost table must be an array");
        for (nlohmann::json const& point : root.at("encoder_decode"))
        {
            config.globalEncoderDecodeCosts.push_back({point.at("encoder_batch_size").get<size_t>(),
                point.at("decode_batch_size").get<int32_t>(), point.at("max_encoder_input_tokens").get<size_t>(),
                point.at("max_decode_context_length").get<int32_t>(), point.at("makespan_p95_gpu_ms").get<float>()});
        }
    }
    config.enableCostAwareEncoderBatching = true;
}

struct PhaseTiming
{
    float prefillMs{};
    float decodeMs{};
    float makespanMs{};
};

class CudaEventAccumulator
{
public:
    explicit CudaEventAccumulator(bool enabled)
        : mEnabled(enabled)
    {
    }

    ~CudaEventAccumulator() noexcept
    {
        for (Interval const& interval : mIntervals)
        {
            static_cast<void>(cudaEventDestroy(interval.start));
            static_cast<void>(cudaEventDestroy(interval.end));
        }
    }

    void begin(cudaStream_t stream)
    {
        if (!mEnabled)
        {
            return;
        }
        Interval interval;
        CUDA_CHECK(cudaEventCreate(&interval.start));
        CUDA_CHECK(cudaEventCreate(&interval.end));
        CUDA_CHECK(cudaEventRecord(interval.start, stream));
        mIntervals.push_back(interval);
    }

    void end(cudaStream_t stream)
    {
        if (mEnabled)
        {
            CUDA_CHECK(cudaEventRecord(mIntervals.back().end, stream));
        }
    }

    float milliseconds()
    {
        float result{};
        for (Interval const& interval : mIntervals)
        {
            CUDA_CHECK(cudaEventSynchronize(interval.end));
            float elapsed{};
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, interval.start, interval.end));
            result += elapsed;
        }
        return result;
    }

    size_t size() const noexcept
    {
        return mIntervals.size();
    }

private:
    struct Interval
    {
        cudaEvent_t start{};
        cudaEvent_t end{};
    };

    bool mEnabled{};
    std::vector<Interval> mIntervals;
};

class SamplingSlotPool
{
public:
    struct Slot
    {
        explicit Slot(int32_t maxRows)
            : hostIds({maxRows}, rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "phase_sampling_host_ids")
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

        rt::Tensor hostIds;
        cudaEvent_t ready{};
        bool busy{};
    };

    explicit SamplingSlotPool(int32_t maxRows)
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
                ++mReuseCount;
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

    size_t size() const noexcept
    {
        return mSlots.size();
    }

    size_t reuseCount() const noexcept
    {
        return mReuseCount;
    }

private:
    int32_t mMaxRows{};
    std::vector<std::unique_ptr<Slot>> mSlots;
    size_t mReuseCount{};
};

PhaseTiming measureOverlap(rt::EngineExecutor& prefillExecutor, rt::EngineExecutor& decodeExecutor,
    cudaStream_t setupStream, cudaStream_t prefillStream, cudaStream_t decodeStream, int32_t warmup, int32_t iterations)
{
    cudaEvent_t gate{};
    cudaEvent_t prefillStart{};
    cudaEvent_t prefillEnd{};
    cudaEvent_t decodeStart{};
    cudaEvent_t decodeEnd{};
    cudaEvent_t done{};
    CUDA_CHECK(cudaEventCreate(&gate));
    CUDA_CHECK(cudaEventCreate(&prefillStart));
    CUDA_CHECK(cudaEventCreate(&prefillEnd));
    CUDA_CHECK(cudaEventCreate(&decodeStart));
    CUDA_CHECK(cudaEventCreate(&decodeEnd));
    CUDA_CHECK(cudaEventCreate(&done));

    PhaseTiming total;
    int32_t const totalIterations = warmup + iterations;
    for (int32_t iteration = 0; iteration < totalIterations; ++iteration)
    {
        CUDA_CHECK(cudaEventRecord(gate, setupStream));
        CUDA_CHECK(cudaStreamWaitEvent(prefillStream, gate));
        CUDA_CHECK(cudaStreamWaitEvent(decodeStream, gate));
        CUDA_CHECK(cudaEventRecord(prefillStart, prefillStream));
        ELLM_CHECK(prefillExecutor.execute(prefillStream), "Packed overlap prefill execution failed");
        CUDA_CHECK(cudaEventRecord(prefillEnd, prefillStream));
        CUDA_CHECK(cudaEventRecord(decodeStart, decodeStream));
        ELLM_CHECK(decodeExecutor.execute(decodeStream), "Packed overlap decode execution failed");
        CUDA_CHECK(cudaEventRecord(decodeEnd, decodeStream));
        CUDA_CHECK(cudaStreamWaitEvent(setupStream, prefillEnd));
        CUDA_CHECK(cudaStreamWaitEvent(setupStream, decodeEnd));
        CUDA_CHECK(cudaEventRecord(done, setupStream));
        CUDA_CHECK(cudaEventSynchronize(done));
        if (iteration >= warmup)
        {
            float prefillMs{};
            float decodeMs{};
            float makespanMs{};
            CUDA_CHECK(cudaEventElapsedTime(&prefillMs, prefillStart, prefillEnd));
            CUDA_CHECK(cudaEventElapsedTime(&decodeMs, decodeStart, decodeEnd));
            CUDA_CHECK(cudaEventElapsedTime(&makespanMs, gate, done));
            total.prefillMs += prefillMs;
            total.decodeMs += decodeMs;
            total.makespanMs += makespanMs;
        }
    }

    CUDA_CHECK(cudaEventDestroy(gate));
    CUDA_CHECK(cudaEventDestroy(prefillStart));
    CUDA_CHECK(cudaEventDestroy(prefillEnd));
    CUDA_CHECK(cudaEventDestroy(decodeStart));
    CUDA_CHECK(cudaEventDestroy(decodeEnd));
    CUDA_CHECK(cudaEventDestroy(done));
    float const scale = 1.0F / static_cast<float>(iterations);
    total.prefillMs *= scale;
    total.decodeMs *= scale;
    total.makespanMs *= scale;
    return total;
}

PhaseTiming measureSequential(rt::EngineExecutor& prefillExecutor, rt::EngineExecutor& decodeExecutor,
    cudaStream_t prefillStream, cudaStream_t decodeStream, int32_t warmup, int32_t iterations)
{
    cudaEvent_t prefillStart{};
    cudaEvent_t prefillEnd{};
    cudaEvent_t decodeStart{};
    cudaEvent_t decodeEnd{};
    CUDA_CHECK(cudaEventCreate(&prefillStart));
    CUDA_CHECK(cudaEventCreate(&prefillEnd));
    CUDA_CHECK(cudaEventCreate(&decodeStart));
    CUDA_CHECK(cudaEventCreate(&decodeEnd));

    PhaseTiming total;
    int32_t const totalIterations = warmup + iterations;
    for (int32_t iteration = 0; iteration < totalIterations; ++iteration)
    {
        CUDA_CHECK(cudaEventRecord(prefillStart, prefillStream));
        ELLM_CHECK(prefillExecutor.execute(prefillStream), "Packed sequential prefill execution failed");
        CUDA_CHECK(cudaEventRecord(prefillEnd, prefillStream));
        CUDA_CHECK(cudaEventSynchronize(prefillEnd));
        CUDA_CHECK(cudaEventRecord(decodeStart, decodeStream));
        ELLM_CHECK(decodeExecutor.execute(decodeStream), "Packed sequential decode execution failed");
        CUDA_CHECK(cudaEventRecord(decodeEnd, decodeStream));
        CUDA_CHECK(cudaEventSynchronize(decodeEnd));
        if (iteration >= warmup)
        {
            float prefillMs{};
            float decodeMs{};
            CUDA_CHECK(cudaEventElapsedTime(&prefillMs, prefillStart, prefillEnd));
            CUDA_CHECK(cudaEventElapsedTime(&decodeMs, decodeStart, decodeEnd));
            total.prefillMs += prefillMs;
            total.decodeMs += decodeMs;
            total.makespanMs += prefillMs + decodeMs;
        }
    }

    CUDA_CHECK(cudaEventDestroy(prefillStart));
    CUDA_CHECK(cudaEventDestroy(prefillEnd));
    CUDA_CHECK(cudaEventDestroy(decodeStart));
    CUDA_CHECK(cudaEventDestroy(decodeEnd));
    float const scale = 1.0F / static_cast<float>(iterations);
    total.prefillMs *= scale;
    total.decodeMs *= scale;
    total.makespanMs *= scale;
    return total;
}

} // namespace

int main(int argc, char** argv)
{
    constexpr int32_t kMIN_ARGUMENTS = 2;
    constexpr int32_t kMAX_ARGUMENTS = 3;
    if (argc < kMIN_ARGUMENTS || argc > kMAX_ARGUMENTS)
    {
        LOG_ERROR("Usage: %s <engine-dir> [checkpoint-dir]", argv[0]);
        return EXIT_FAILURE;
    }

    auto pluginHandle = loadEdgellmPluginLib();
    if (pluginHandle == nullptr)
    {
        return EXIT_FAILURE;
    }

    std::filesystem::path const engineDir{argv[1]};
    std::string const checkpointDir = argc == kMAX_ARGUMENTS ? argv[2] : "";
    rt::LLMEngineConfig const config = rt::parseEngineConfig(engineDir / "config.json");

    cudaStream_t setupStream{};
    cudaStream_t prefillStream{};
    cudaStream_t decodeStream{};
    bool const enablePhaseStreamPriorities = std::getenv("TRT_EDGELLM_PHASE_STREAM_PRIORITIES") != nullptr;
    int leastPriority{};
    int greatestPriority{};
    if (enablePhaseStreamPriorities)
    {
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    }
    int const normalPriority = std::clamp(0, greatestPriority, leastPriority);
    if (enablePhaseStreamPriorities)
    {
        CUDA_CHECK(cudaStreamCreateWithPriority(&setupStream, cudaStreamNonBlocking, normalPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&prefillStream, cudaStreamNonBlocking, normalPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&decodeStream, cudaStreamNonBlocking, greatestPriority));
        LOG_INFO("Phase stream priorities: encoder=%d prefill=%d decode=%d", leastPriority, normalPriority,
            greatestPriority);
    }
    else
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&setupStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));
    }

    {
        auto executor = rt::EngineExecutor::createForLLM(engineDir / "llm.engine", config);
        rt::IndependentEngineExecutorPairConfig pairConfig;
        pairConfig.setupStream = setupStream;
        pairConfig.prefillStream = prefillStream;
        pairConfig.decodeStream = decodeStream;
        pairConfig.sharedExecutionContext = std::getenv("TRT_EDGELLM_SHARED_PHASE_CONTEXT") != nullptr;
        auto pair = rt::IndependentEngineExecutorPair::create(std::move(executor), pairConfig);

        ELLM_CHECK(&pair->prefillExecutor().getEngine() == &pair->decodeExecutor().getEngine(),
            "Phase executors must share one TensorRT engine");
        if (pair->sharedExecutionContext())
        {
            ELLM_CHECK(pair->prefillExecutor().getExecutionContextIdentity()
                    == pair->decodeExecutor().getExecutionContextIdentity(),
                "Shared phase mode must reuse one TensorRT context");
            ELLM_CHECK(pair->prefillContextMemory().rawPointer() == pair->decodeContextMemory().rawPointer(),
                "Shared phase mode must reuse one workspace");
        }
        else
        {
            ELLM_CHECK(pair->prefillExecutor().getExecutionContextIdentity()
                    != pair->decodeExecutor().getExecutionContextIdentity(),
                "Independent phase executors must own different TensorRT contexts");
            ELLM_CHECK(pair->prefillContextMemory().rawPointer() != pair->decodeContextMemory().rawPointer(),
                "Independent phase executors must own different workspaces");
        }

        std::unordered_map<std::string, std::string> const emptyLoraMap;
        auto resources = rt::SharedResources::createForLLM(config, emptyLoraMap, setupStream);
        rt::EmbeddingData embedding = rt::loadEmbeddingTable(engineDir / "embedding.safetensors", setupStream);
        auto prefillIO = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForLLMPhase(
            config, config.maxSupportedPrefillBatchSize, config.maxSupportedInputLength, setupStream));
        auto decodeIO = std::make_unique<rt::PipelineIO>(
            rt::PipelineIO::createForLLMPhase(config, config.maxSupportedDecodeBatchSize, 1, setupStream));
        rt::TensorMap prefillMap;
        rt::TensorMap decodeMap;
        rt::buildTensorMap(prefillMap, *prefillIO, *resources, config, 0);
        rt::buildTensorMap(decodeMap, *decodeIO, *resources, config, 0);
        resources->externalWeightManager->load(
            engineDir, engineDir / "config.json", setupStream, checkpointDir, {}, &embedding.table);
        resources->externalWeightManager->validateAgainstEngine(pair->prefillExecutor(), "base");
        resources->externalWeightManager->registerTensorMapEntries(prefillMap);
        resources->externalWeightManager->registerTensorMapEntries(decodeMap);

        int32_t maxStableSlots = std::max(kDEFAULT_STABLE_SLOTS, config.maxSupportedBatchSize);
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_STABLE_SLOTS"))
        {
            maxStableSlots = std::stoi(value);
        }
        int32_t const maxPhaseBatch = std::max(config.maxSupportedPrefillBatchSize, config.maxSupportedDecodeBatchSize);
        ELLM_CHECK(maxStableSlots >= maxPhaseBatch, "Stable slot capacity must cover the largest phase batch");
        rt::StableKVPageManager ownership(
            {maxStableSlots, maxPhaseBatch, config.kvPoolPages, config.maxKVCacheCapacity, 128});
        bool const semanticOnly = std::getenv("TRT_EDGELLM_SEMANTIC_ONLY") != nullptr;
        if (!semanticOnly)
        {
            int32_t const prefillSlot0 = ownership.reserve();
            int32_t const prefillSlot1 = config.packedPrefill ? ownership.reserve() : -1;
            int32_t const decodeSlot = ownership.reserve();
            ownership.ensureCapacity(prefillSlot0, 128);
            if (config.packedPrefill)
            {
                ownership.ensureCapacity(prefillSlot1, 128);
            }
            ownership.ensureCapacity(decodeSlot, 129);
            ownership.setLength(prefillSlot0, 0);
            if (config.packedPrefill)
            {
                ownership.setLength(prefillSlot1, 0);
            }
            ownership.setLength(decodeSlot, 128);
            rt::PhaseKVActiveView prefillKV(config.maxSupportedPrefillBatchSize, ownership, prefillMap, "prefill");
            rt::PhaseKVActiveView decodeKV(config.maxSupportedDecodeBatchSize, ownership, decodeMap, "decode");
            std::vector<int32_t> const prefillSlots = config.packedPrefill
                ? std::vector<int32_t>{prefillSlot0, prefillSlot1}
                : std::vector<int32_t>{prefillSlot0};
            std::vector<int32_t> const prefillChunkLengths
                = config.packedPrefill ? std::vector<int32_t>{96, 32} : std::vector<int32_t>{128};
            prefillKV.prepare(prefillSlots, prefillStream);
            decodeKV.prepare({decodeSlot}, decodeStream);

            int32_t const prefillTotalTokens = 128;
            ELLM_CHECK(prefillIO->inputsEmbeds.reshape({1, prefillTotalTokens, config.hiddenSize}),
                "Failed to reshape prefill input embeddings");
            ELLM_CHECK(
                decodeIO->inputsEmbeds.reshape({1, 1, config.hiddenSize}), "Failed to reshape decode input embeddings");
            CUDA_CHECK(cudaMemsetAsync(
                prefillIO->inputsEmbeds.rawPointer(), 0, prefillIO->inputsEmbeds.getMemoryCapacity(), prefillStream));
            CUDA_CHECK(cudaMemsetAsync(
                decodeIO->inputsEmbeds.rawPointer(), 0, decodeIO->inputsEmbeds.getMemoryCapacity(), decodeStream));
            for (rt::Tensor& deepstack : prefillIO->deepstackEmbeds)
            {
                CUDA_CHECK(cudaMemsetAsync(deepstack.rawPointer(), 0, deepstack.getMemoryCapacity(), prefillStream));
            }
            prefillKV.preparePrefillMetadata(*prefillIO, prefillChunkLengths, prefillStream, config.packedPrefill);
            decodeKV.prepareDecodeMetadata(*decodeIO, decodeStream);

            rt::InferenceDims const prefillDims = config.packedPrefill
                ? config.packedPrefillDims(static_cast<int64_t>(prefillSlots.size()), prefillTotalTokens)
                : config.prefillDims(1, prefillTotalTokens, false);
            ELLM_CHECK(pair->prefillExecutor().prepare(0, prefillDims, prefillMap, prefillStream),
                "Failed to bind the stable paged-KV prefill view");
            ELLM_CHECK(pair->decodeExecutor().prepare(1, config.decodeDims(1), decodeMap, decodeStream),
                "Failed to bind the stable paged-KV decode view");
            if (std::getenv("TRT_EDGELLM_CAPTURE_PHASE_GRAPHS") != nullptr)
            {
                bool const prefillGraph = pair->prefillExecutor().captureGraph(prefillStream);
                bool const decodeGraph = pair->decodeExecutor().captureGraph(decodeStream);
                LOG_INFO("Prepared phase CUDA graphs: prefill=%s decode=%s", prefillGraph ? "yes" : "no",
                    decodeGraph ? "yes" : "no");
            }
            constexpr int32_t kWARMUP = 20;
            constexpr int32_t kITERATIONS = 100;
            PhaseTiming const sequential = measureSequential(
                pair->prefillExecutor(), pair->decodeExecutor(), prefillStream, decodeStream, kWARMUP, kITERATIONS);
            PhaseTiming const overlap = measureOverlap(pair->prefillExecutor(), pair->decodeExecutor(), setupStream,
                prefillStream, decodeStream, kWARMUP, kITERATIONS);
            float const speedup = sequential.makespanMs / overlap.makespanMs;
            float const overlapRatio = (overlap.prefillMs + overlap.decodeMs - overlap.makespanMs)
                / std::min(overlap.prefillMs, overlap.decodeMs);
            prefillKV.commitLengths(prefillChunkLengths);
            decodeKV.commitLengths({129});
            prefillKV.complete();
            decodeKV.complete();

            LOG_INFO(
                "Independent phase context smoke passed: prefill_workspace=%zu decode_workspace=%zu stable_pages=%d",
                pair->prefillContextMemory().getMemoryCapacity(), pair->decodeContextMemory().getMemoryCapacity(),
                config.kvPoolPages - ownership.availablePages());
            LOG_INFO(
                "Independent phase timing (mean of %d): sequential=%.4f ms overlap=%.4f ms speedup=%.3fx "
                "overlap_ratio=%.3f prefill_seq=%.4f ms decode_seq=%.4f ms prefill_overlap=%.4f ms "
                "decode_overlap=%.4f ms",
                kITERATIONS, sequential.makespanMs, overlap.makespanMs, speedup, overlapRatio, sequential.prefillMs,
                sequential.decodeMs, overlap.prefillMs, overlap.decodeMs);

            // Exercise the real queue scheduler -> dispatch worker -> independent
            // TensorRT context path. Two 256-token requests advance in fixed-128
            // waves while one request is already decoding.
            ownership.ensureCapacity(prefillSlot0, 258);
            ownership.ensureCapacity(prefillSlot1, 258);
            ownership.ensureCapacity(decodeSlot, 132);
            ownership.setLength(prefillSlot0, 0);
            ownership.setLength(prefillSlot1, 0);
            ownership.setLength(decodeSlot, 128);
            for (rt::Tensor& deepstack : decodeIO->deepstackEmbeds)
            {
                CUDA_CHECK(cudaMemsetAsync(deepstack.rawPointer(), 0, deepstack.getMemoryCapacity(), decodeStream));
            }

            rt::PhaseQueueSchedulerConfig schedulerConfig;
            schedulerConfig.maxPrefillBatchSize = 2;
            schedulerConfig.maxDecodeBatchSize = 3;
            schedulerConfig.maxPrefillChunkTokens = 128;
            schedulerConfig.maxOverlapPrefillTokens = 64;
            schedulerConfig.enablePackedPrefillTokenLayout = true;
            schedulerConfig.enableTpotHardGuard = true;
            schedulerConfig.requireDirectOverlapCost = true;
            schedulerConfig.enableCostAwareOverlapAdmission = true;
            schedulerConfig.decodeQueueWaitTargetUs = 50000.0;
            schedulerConfig.decodeSlackSafetyFactor = 0.8F;
            schedulerConfig.maxConsecutiveOverlapBatches = 2;
            schedulerConfig.maxPredictedDecodeDebtUs = 10000.0;
            schedulerConfig.prefillBatchCosts = {
                {1, 128, 2048, 3, true, 13.7F, 2.5F},
                {2, 128, 2048, 3, true, 13.7F, 2.5F},
                {1, 128, 2048, 3, false, 13.7F, 2.5F},
                {2, 128, 2048, 3, false, 13.7F, 2.5F},
            };
            schedulerConfig.overlapBatchCosts = {
                {1, 3, 128, 2048, 2048, true, 13.7F, 8.1F, 13.7F, 2.5F},
                {2, 3, 128, 2048, 2048, true, 13.7F, 8.1F, 13.7F, 2.5F},
                {1, 3, 128, 2048, 2048, false, 13.7F, 8.1F, 13.7F, 2.5F},
                {2, 3, 128, 2048, 2048, false, 13.7F, 8.1F, 13.7F, 2.5F},
            };
            std::unordered_map<uint64_t, int32_t> decodeSteps;
            rt::IndependentPhaseCoordinatorCallbacks coordinatorCallbacks;
            coordinatorCallbacks.isDecodeFinished
                = [&](rt::PhaseWorkItem const& item, int32_t) { return ++decodeSteps[item.requestId] >= 2; };
            rt::IndependentPhaseCoordinator coordinator(config, schedulerConfig, *pair, ownership, *prefillIO,
                *decodeIO, prefillMap, decodeMap, prefillStream, decodeStream, std::move(coordinatorCallbacks));

            rt::PhaseSchedulingHints decodeHints;
            decodeHints.tpotTargetUs = 50000.0;
            coordinator.enqueuePrefill({101, 256, prefillSlot0, 0, 256});
            coordinator.enqueuePrefill({102, 256, prefillSlot1, 0, 256});
            coordinator.enqueueDecode({103, 128, decodeSlot, 0, 0, true, decodeHints});
            coordinator.runUntilIdle(32);
            std::vector<rt::PhaseDispatchMetrics> const& dispatchMetrics = coordinator.metrics();
            size_t const overlapDispatches = static_cast<size_t>(std::count_if(
                dispatchMetrics.begin(), dispatchMetrics.end(), [](rt::PhaseDispatchMetrics const& metrics) {
                    return metrics.kind == rt::PhaseDispatchKind::kOverlap;
                }));
            double totalMakespanMs{};
            for (rt::PhaseDispatchMetrics const& metrics : dispatchMetrics)
            {
                totalMakespanMs += metrics.makespanGpuMs;
            }
            ELLM_CHECK(
                coordinator.empty() && overlapDispatches > 0, "Cost-aware phase queue did not drain with overlap");
            LOG_INFO(
                "Cost-aware phase queue passed: dispatches=%zu overlap_dispatches=%zu total_makespan=%.4f ms "
                "request101_kv=%d request102_kv=%d request103_kv=%d",
                dispatchMetrics.size(), overlapDispatches, totalMakespanMs, ownership.length(prefillSlot0),
                ownership.length(prefillSlot1), ownership.length(decodeSlot));

            ownership.release(prefillSlot0);
            ownership.release(prefillSlot1);
            ownership.release(decodeSlot);

            rt::PhaseContinuousLoadGenerator load({16, 1000.0, 128, 512, 2, 6, 20260819, 10000});
            std::deque<rt::PhaseLoadRequest> pendingAdmissions;
            std::unordered_map<uint64_t, int32_t> traceSlots;
            std::unordered_map<uint64_t, int32_t> traceOutputTargets;
            std::unordered_map<uint64_t, int32_t> traceDecodeSteps;
            std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> traceSubmittedAt;
            std::vector<double> traceLatenciesMs;
            size_t completedRequests{};
            rt::IndependentPhaseCoordinatorCallbacks traceCallbacks;
            traceCallbacks.isDecodeFinished = [&](rt::PhaseWorkItem const& item, int32_t) {
                int32_t const steps = ++traceDecodeSteps[item.requestId];
                bool const finished = steps >= traceOutputTargets.at(item.requestId);
                if (finished)
                {
                    auto const now = std::chrono::steady_clock::now();
                    traceLatenciesMs.push_back(
                        std::chrono::duration<double, std::milli>(now - traceSubmittedAt.at(item.requestId)).count());
                    ownership.release(traceSlots.at(item.requestId));
                    ++completedRequests;
                }
                return finished;
            };
            rt::IndependentPhaseCoordinator traceCoordinator(config, schedulerConfig, *pair, ownership, *prefillIO,
                *decodeIO, prefillMap, decodeMap, prefillStream, decodeStream, std::move(traceCallbacks));

            auto const traceStart = std::chrono::steady_clock::now();
            size_t loopIterations{};
            while (completedRequests < load.schedule().size())
            {
                int64_t const elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - traceStart)
                                              .count();
                std::vector<rt::PhaseLoadRequest> ready = load.popReady(elapsedUs);
                pendingAdmissions.insert(pendingAdmissions.end(), ready.begin(), ready.end());
                while (!pendingAdmissions.empty() && ownership.availableSlots() > 0)
                {
                    rt::PhaseLoadRequest const request = pendingAdmissions.front();
                    int32_t const slot = ownership.reserve();
                    ownership.ensureCapacity(slot, request.promptTokenCount + request.maxOutputTokens);
                    ownership.setLength(slot, 0);
                    traceSlots[request.requestId] = slot;
                    traceOutputTargets[request.requestId] = request.maxOutputTokens;
                    traceSubmittedAt[request.requestId]
                        = traceStart + std::chrono::microseconds(request.arrivalOffsetUs);
                    traceCoordinator.enqueuePrefill(
                        {request.requestId, request.promptTokenCount, slot, 0, request.promptTokenCount});
                    pendingAdmissions.pop_front();
                }
                if (traceCoordinator.busy())
                {
                    static_cast<void>(traceCoordinator.poll());
                }
                else if (!traceCoordinator.empty())
                {
                    static_cast<void>(traceCoordinator.dispatchNext());
                }
                ELLM_CHECK(++loopIterations < 10000000, "Continuous phase load loop exceeded its runaway guard");
            }
            if (traceCoordinator.busy())
            {
                traceCoordinator.wait();
            }
            auto const traceEnd = std::chrono::steady_clock::now();
            std::sort(traceLatenciesMs.begin(), traceLatenciesMs.end());
            size_t const p95Index = std::min(
                traceLatenciesMs.size() - 1, static_cast<size_t>(0.95 * static_cast<double>(traceLatenciesMs.size())));
            double totalLatencyMs{};
            for (double const latency : traceLatenciesMs)
            {
                totalLatencyMs += latency;
            }
            double const elapsedSeconds = std::chrono::duration<double>(traceEnd - traceStart).count();
            size_t const traceOverlapDispatches = static_cast<size_t>(std::count_if(traceCoordinator.metrics().begin(),
                traceCoordinator.metrics().end(), [](rt::PhaseDispatchMetrics const& metrics) {
                    return metrics.kind == rt::PhaseDispatchKind::kOverlap;
                }));
            ELLM_CHECK(traceCoordinator.empty() && ownership.availableSlots() == maxStableSlots,
                "Continuous phase load did not release every request and stable slot");
            LOG_INFO(
                "Continuous phase load passed: requests=%zu dispatches=%zu overlaps=%zu throughput=%.2f req/s "
                "mean_latency=%.3f ms p95_latency=%.3f ms",
                completedRequests, traceCoordinator.metrics().size(), traceOverlapDispatches,
                static_cast<double>(completedRequests) / elapsedSeconds,
                totalLatencyMs / static_cast<double>(traceLatenciesMs.size()), traceLatenciesMs[p95Index]);
        }

        // Real text payload path: tokenizer -> independent async server adapter ->
        // compact token staging -> embedding lookup -> TensorRT prefill/decode.
        CUDA_CHECK(cudaStreamSynchronize(setupStream));
        rt::EmbeddingPreprocessor embeddingPreprocessor(embedding, config);
        tokenizer::Tokenizer tokenizer;
        ELLM_CHECK(tokenizer.loadFromHF(engineDir), "Failed to load tokenizer for semantic phase requests");

        std::vector<std::string> const prompts{
            "Give one practical tip for reducing latency in an online inference service.",
            "In two short sentences, explain why dynamic batching can improve GPU utilization.",
            "List three concise checks for diagnosing a slow CUDA inference pipeline, focusing on kernel timing, "
            "memory transfers, and synchronization.",
        };
        std::unordered_map<uint64_t, std::vector<int32_t>> semanticPrompts;
        constexpr int32_t kSEMANTIC_OUTPUT_TOKENS = 8;
        for (size_t index = 0; index < prompts.size(); ++index)
        {
            uint64_t const requestId = 20000 + index;
            rt::LLMGenerationRequest::Request request;
            request.messages.push_back({"user", {{"text", prompts[index]}}});
            rt::LLMGenerationRequest::FormattedRequest formatted;
            ELLM_CHECK(tokenizer.applyChatTemplate(request, formatted, true, true, false),
                "Failed to format semantic phase request");
            semanticPrompts[requestId] = tokenizer.encode(formatted.formattedCompleteRequest, false);
            ELLM_CHECK(!semanticPrompts[requestId].empty(), "Semantic phase request tokenized to an empty prompt");
        }

        int32_t const prefillTokenCapacity
            = config.packedPrefill ? config.maxPackedPrefillChunkTokens : config.maxSupportedInputLength;
        bool const enableBatchedVisionPrefill = std::getenv("TRT_EDGELLM_ENABLE_BATCHED_VISION_PREFILL") != nullptr;
        ELLM_CHECK(!enableBatchedVisionPrefill || config.packedPrefill,
            "Batched vision prefill requires a packed-prefill engine");
        int32_t prefillBatchTokenBudget = config.maxSupportedPrefillBatchSize * 128;
        char const* prefillBatchTokenBudgetValue = std::getenv("TRT_EDGELLM_MAX_PREFILL_BATCH_TOKENS");
        if (prefillBatchTokenBudgetValue != nullptr)
        {
            prefillBatchTokenBudget = std::stoi(prefillBatchTokenBudgetValue);
        }
        int32_t const profileTokenCapacity = config.maxSupportedPrefillBatchSize * prefillTokenCapacity;
        ELLM_CHECK(prefillBatchTokenBudget > 0 && prefillBatchTokenBudget <= profileTokenCapacity,
            "Prefill batch token budget is outside the packed-prefill profile");
        rt::Tensor hostSemanticPrefillIds({config.maxSupportedPrefillBatchSize, prefillTokenCapacity},
            rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "semantic_phase_host_prefill_ids");
        rt::Tensor deviceSemanticPrefillIds({config.maxSupportedPrefillBatchSize, prefillTokenCapacity},
            rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "semantic_phase_prefill_ids");
        rt::Tensor hostSemanticDecodeIds({config.maxSupportedDecodeBatchSize, 1}, rt::DeviceType::kCPU,
            nvinfer1::DataType::kINT32, "semantic_phase_host_decode_ids");
        rt::Tensor deviceSemanticDecodeIds({config.maxSupportedDecodeBatchSize, 1}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "semantic_phase_decode_ids");
        size_t const samplingWorkspaceBytes = getSelectAllTopKWorkspaceSize(maxPhaseBatch, config.outputVocabSize, 1);
        rt::Tensor prefillSamplingWorkspace({static_cast<int64_t>(samplingWorkspaceBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT8, "semantic_phase_prefill_sampling_workspace");
        rt::Tensor decodeSamplingWorkspace({static_cast<int64_t>(samplingWorkspaceBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT8, "semantic_phase_decode_sampling_workspace");
        rt::Tensor prefillSelectedIds({config.maxSupportedPrefillBatchSize, 1}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "semantic_phase_prefill_selected_ids");
        rt::Tensor decodeSelectedIds({config.maxSupportedDecodeBatchSize, 1}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "semantic_phase_decode_selected_ids");
        rt::Tensor prefillCompactedLogits({config.maxSupportedPrefillBatchSize, config.outputVocabSize},
            rt::DeviceType::kGPU, nvinfer1::DataType::kFLOAT, "semantic_phase_prefill_compacted_logits");
        SamplingSlotPool samplingSlotPool(maxPhaseBatch);
        std::vector<uint64_t> lastDecodeSampleRequestIds;
        size_t tokenH2DOperations{};
        size_t tokenH2DBytes{};
        size_t decodeDeviceTokenReuseBatches{};
        size_t decodeDeviceTokenReuseRows{};
        size_t segmentedVisionBatches{};
        size_t segmentedVisionSources{};
        size_t segmentedVisionBytes{};
        size_t mropeD2DOperations{};
        size_t mropeD2DBytes{};
        size_t mropeFullRowD2DBytes{};
        int32_t mropeCopyGranularity = std::min(512, config.maxKVCacheCapacity);
        if (char const* value = std::getenv("TRT_EDGELLM_MROPE_COPY_GRANULARITY"))
        {
            mropeCopyGranularity = std::stoi(value);
        }
        ELLM_CHECK(mropeCopyGranularity > 0 && mropeCopyGranularity <= config.maxKVCacheCapacity,
            "M-RoPE copy granularity is outside the cache capacity");
        CudaEventAccumulator mropeCopyTiming(std::getenv("TRT_EDGELLM_PROFILE_MROPE_D2D") != nullptr);
        rt::Tensor textOnlyMropeTemplate;
        std::vector<std::optional<uint64_t>> prefillMropeOwners(
            static_cast<size_t>(config.maxSupportedPrefillBatchSize));
        std::vector<std::optional<uint64_t>> decodeMropeOwners(static_cast<size_t>(config.maxSupportedDecodeBatchSize));
        std::vector<int32_t> prefillMropeValid(
            static_cast<size_t>(config.maxSupportedPrefillBatchSize), config.maxKVCacheCapacity);
        std::vector<int32_t> decodeMropeValid(
            static_cast<size_t>(config.maxSupportedDecodeBatchSize), config.maxKVCacheCapacity);
        if (config.ropeConfig.type == rt::RopeType::kMRope)
        {
            textOnlyMropeTemplate = rt::Tensor({1, config.maxKVCacheCapacity, config.rotaryDim}, rt::DeviceType::kGPU,
                nvinfer1::DataType::kFLOAT, "semantic_phase_text_mrope_template");
            kernel::initializeTextOnlyMRopeCosSin(textOnlyMropeTemplate.dataPointer<float>(),
                config.ropeConfig.rotaryTheta, config.rotaryDim, config.maxKVCacheCapacity, 1, setupStream);
        }

        auto stageTokens = [&](std::vector<rt::IndependentPhaseRequestView> const& views, rt::PipelineIO& io,
                               rt::TensorMap& map, cudaStream_t stream, bool prefill) {
            rt::OptionalInputTensor visionEmbedding;
            rt::OptionalInputTensors deepstackFeatures;
            rt::Tensor visionEmbeddingView;
            std::vector<rt::Tensor> deepstackFeatureViews;
            std::vector<rt::Tensor> segmentedVisionViews;
            rt::OptionalInputTensors segmentedVisionSegments;
            std::vector<std::vector<rt::Tensor>> segmentedDeepstackViews;
            std::vector<rt::OptionalInputTensors> segmentedDeepstackSegments;
            std::vector<rt::IndependentPhaseRequestView const*> visionViews;
            if (prefill)
            {
                for (rt::IndependentPhaseRequestView const& view : views)
                {
                    if (view.visionPayload != nullptr)
                    {
                        visionViews.push_back(&view);
                    }
                }
            }
            auto imageRange = [&](rt::IndependentPhaseRequestView const& view) {
                auto const chunkBegin = view.promptTokens->begin() + view.work.tokenOffset;
                auto const chunkEnd = chunkBegin + view.work.tokenCount;
                int64_t const offset = std::count(view.promptTokens->begin(), chunkBegin, config.imageTokenId);
                int64_t const count = std::count(chunkBegin, chunkEnd, config.imageTokenId);
                return std::pair<int64_t, int64_t>{offset, count};
            };
            auto makeFeatureView = [&](rt::Tensor& feature, int64_t imageOffset, int64_t imageTokens,
                                       std::string const& name) {
                rt::Coords const shape = feature.getShape();
                ELLM_CHECK(shape.getNumDims() == 2 && imageOffset >= 0 && imageTokens > 0
                        && imageOffset + imageTokens <= shape[0],
                    "Multimodal chunk is outside its request-owned feature buffer");
                size_t const rowBytes = static_cast<size_t>(shape[1]) * rt::utils::getTypeSize(feature.getDataType());
                auto* const data = static_cast<std::byte*>(feature.rawPointer()) + imageOffset * rowBytes;
                return rt::Tensor(data, {imageTokens, shape[1]}, rt::DeviceType::kGPU, feature.getDataType(), name);
            };
            if (visionViews.size() == 1U)
            {
                rt::IndependentPhaseRequestView const& view = *visionViews.front();
                auto const [imageOffset, imageTokens] = imageRange(view);
                if (imageTokens > 0)
                {
                    rt::PhaseVisionPayload& payload = *view.visionPayload;
                    visionEmbeddingView
                        = makeFeatureView(payload.outputEmbedding, imageOffset, imageTokens, "phase_vision_chunk");
                    visionEmbedding = std::cref(visionEmbeddingView);
                    deepstackFeatureViews.reserve(payload.deepstackFeatures.size());
                    deepstackFeatures.reserve(payload.deepstackFeatures.size());
                    for (rt::Tensor& feature : payload.deepstackFeatures)
                    {
                        deepstackFeatureViews.push_back(
                            makeFeatureView(feature, imageOffset, imageTokens, "phase_deepstack_chunk"));
                        deepstackFeatures.push_back(std::cref(deepstackFeatureViews.back()));
                    }
                }
            }
            else if (!visionViews.empty())
            {
                ELLM_CHECK(enableBatchedVisionPrefill, "Multiple multimodal rows require batched vision prefill");
                int64_t totalImageTokens{};
                segmentedVisionViews.reserve(visionViews.size());
                segmentedDeepstackViews.resize(static_cast<size_t>(config.numDeepstackFeatures));
                for (auto& features : segmentedDeepstackViews)
                {
                    features.reserve(visionViews.size());
                }
                for (rt::IndependentPhaseRequestView const* view : visionViews)
                {
                    auto const [imageOffset, imageTokens] = imageRange(*view);
                    if (imageTokens == 0)
                    {
                        continue;
                    }
                    rt::PhaseVisionPayload& payload = *view->visionPayload;
                    segmentedVisionViews.push_back(
                        makeFeatureView(payload.outputEmbedding, imageOffset, imageTokens, "phase_vision_segment"));
                    ELLM_CHECK(payload.deepstackFeatures.size() == segmentedDeepstackViews.size(),
                        "Batched vision request has the wrong deepstack feature count");
                    for (size_t index{}; index < payload.deepstackFeatures.size(); ++index)
                    {
                        segmentedDeepstackViews[index].push_back(makeFeatureView(
                            payload.deepstackFeatures[index], imageOffset, imageTokens, "phase_deepstack_segment"));
                    }
                    totalImageTokens += imageTokens;
                }
                ELLM_CHECK(totalImageTokens > 0, "Segmented vision batch contains no active image rows");
                segmentedVisionSegments.reserve(segmentedVisionViews.size());
                for (rt::Tensor const& segment : segmentedVisionViews)
                {
                    segmentedVisionSegments.push_back(std::cref(segment));
                    segmentedVisionBytes += static_cast<size_t>(segment.getShape().volume())
                        * rt::utils::getTypeSize(segment.getDataType());
                }
                segmentedDeepstackSegments.resize(segmentedDeepstackViews.size());
                for (size_t featureIndex{}; featureIndex < segmentedDeepstackViews.size(); ++featureIndex)
                {
                    for (rt::Tensor const& segment : segmentedDeepstackViews[featureIndex])
                    {
                        segmentedDeepstackSegments[featureIndex].push_back(std::cref(segment));
                        segmentedVisionBytes += static_cast<size_t>(segment.getShape().volume())
                            * rt::utils::getTypeSize(segment.getDataType());
                    }
                }
                ++segmentedVisionBatches;
                segmentedVisionSources += segmentedVisionViews.size();
            }
            if (config.ropeConfig.type == rt::RopeType::kMRope)
            {
                std::vector<std::optional<uint64_t>>& owners = prefill ? prefillMropeOwners : decodeMropeOwners;
                std::vector<int32_t>& validPositions = prefill ? prefillMropeValid : decodeMropeValid;
                size_t const positionBytes
                    = static_cast<size_t>(config.rotaryDim) * rt::utils::getTypeSize(nvinfer1::DataType::kFLOAT);
                size_t const rowBytes = static_cast<size_t>(config.maxKVCacheCapacity) * config.rotaryDim
                    * rt::utils::getTypeSize(nvinfer1::DataType::kFLOAT);
                bool timingStarted{};
                for (size_t row{}; row < views.size(); ++row)
                {
                    rt::PhaseVisionPayload* const payload = views[row].visionPayload;
                    std::optional<uint64_t> const desiredOwner = payload != nullptr && !payload->mropeCosSin.isEmpty()
                        ? std::optional<uint64_t>{views[row].work.requestId}
                        : std::nullopt;
                    bool const ownerChanged = owners[row] != desiredOwner;
                    int32_t const requiredPositions = prefill ? views[row].work.tokenOffset + views[row].work.tokenCount
                                                              : views[row].work.tokenCount + 1;
                    rt::PhaseMropeStagingRange const range = rt::phaseMropeStagingRange(ownerChanged,
                        validPositions[row], requiredPositions, config.maxKVCacheCapacity, mropeCopyGranularity);
                    if (ownerChanged)
                    {
                        mropeFullRowD2DBytes += rowBytes;
                    }
                    owners[row] = desiredOwner;
                    validPositions[row] = range.validPositions;
                    if (range.countPositions > 0)
                    {
                        if (!timingStarted)
                        {
                            mropeCopyTiming.begin(stream);
                            timingStarted = true;
                        }
                        rt::Tensor const& source
                            = desiredOwner.has_value() ? payload->mropeCosSin : textOnlyMropeTemplate;
                        size_t const copyOffset = static_cast<size_t>(range.offsetPositions) * positionBytes;
                        size_t const copyBytes = static_cast<size_t>(range.countPositions) * positionBytes;
                        auto* const destination
                            = static_cast<std::byte*>(io.mropeCosSin.rawPointer()) + row * rowBytes + copyOffset;
                        auto const* const sourceBytes = static_cast<std::byte const*>(source.rawPointer()) + copyOffset;
                        CUDA_CHECK(
                            cudaMemcpyAsync(destination, sourceBytes, copyBytes, cudaMemcpyDeviceToDevice, stream));
                        ++mropeD2DOperations;
                        mropeD2DBytes += copyBytes;
                    }
                }
                if (timingStarted)
                {
                    mropeCopyTiming.end(stream);
                }
                map.set(binding_names::kRopeCosSin, io.mropeCosSin);
            }
            int32_t totalTokens{};
            for (rt::IndependentPhaseRequestView const& view : views)
            {
                totalTokens += prefill ? view.work.tokenCount : 1;
            }
            rt::Coords const tokenShape = prefill
                ? (config.packedPrefill ? rt::Coords{1, totalTokens}
                                        : rt::Coords{static_cast<int64_t>(views.size()), views.front().work.tokenCount})
                : rt::Coords{static_cast<int64_t>(views.size()), 1};
            bool const reuseDecodeSample = !prefill && lastDecodeSampleRequestIds.size() == views.size()
                && std::equal(views.begin(), views.end(), lastDecodeSampleRequestIds.begin(),
                    [](rt::IndependentPhaseRequestView const& view, uint64_t requestId) {
                        return view.requestId == requestId;
                    });
            rt::Tensor sampledIdsView;
            rt::Tensor* stagedIds{};
            if (reuseDecodeSample)
            {
                sampledIdsView = rt::Tensor(decodeSelectedIds.rawPointer(), tokenShape, rt::DeviceType::kGPU,
                    nvinfer1::DataType::kINT32, "semantic_phase_reused_decode_ids");
                stagedIds = &sampledIdsView;
                ++decodeDeviceTokenReuseBatches;
                decodeDeviceTokenReuseRows += views.size();
            }
            else
            {
                rt::Tensor& hostIds = prefill ? hostSemanticPrefillIds : hostSemanticDecodeIds;
                rt::Tensor& deviceIds = prefill ? deviceSemanticPrefillIds : deviceSemanticDecodeIds;
                ELLM_CHECK(hostIds.reshape(tokenShape) && deviceIds.reshape(tokenShape),
                    "Semantic phase token staging reshape failed");
                int32_t* destination = hostIds.dataPointer<int32_t>();
                int32_t destinationOffset{};
                for (rt::IndependentPhaseRequestView const& view : views)
                {
                    if (prefill)
                    {
                        std::copy_n(view.promptTokens->begin() + view.work.tokenOffset, view.work.tokenCount,
                            destination + destinationOffset);
                        destinationOffset += view.work.tokenCount;
                    }
                    else
                    {
                        ELLM_CHECK(
                            !view.generatedTokens->empty(), "Semantic decode request has no sampled input token");
                        destination[destinationOffset++] = view.generatedTokens->back();
                    }
                }
                size_t const copyBytes = static_cast<size_t>(totalTokens) * sizeof(int32_t);
                CUDA_CHECK(cudaMemcpyAsync(
                    deviceIds.rawPointer(), hostIds.rawPointer(), copyBytes, cudaMemcpyHostToDevice, stream));
                ++tokenH2DOperations;
                tokenH2DBytes += copyBytes;
                stagedIds = &deviceIds;
            }
            if (!segmentedVisionSegments.empty())
            {
                embeddingPreprocessor.embedSegmentedVision(*stagedIds, segmentedVisionSegments, io, stream);
                embeddingPreprocessor.prepareSegmentedDeepstack(*stagedIds, segmentedDeepstackSegments, io, stream);
            }
            else
            {
                embeddingPreprocessor.embed(*stagedIds, visionEmbedding, std::nullopt, io, stream);
                embeddingPreprocessor.prepareDeepstack(*stagedIds, deepstackFeatures, io, stream);
            }
            if (prefill)
            {
                for (int32_t index = 0; index < static_cast<int32_t>(io.deepstackEmbeds.size()); ++index)
                {
                    map.set(binding_names::formatDeepstackEmbedsName(index), io.deepstackEmbeds[index]);
                }
            }
        };

        auto submitSampling = [&](std::vector<rt::IndependentPhaseRequestView> const& views, rt::PipelineIO& io,
                                  cudaStream_t stream, bool prefill) {
            int32_t const batchSize = static_cast<int32_t>(views.size());
            rt::Tensor& selectedIds = prefill ? prefillSelectedIds : decodeSelectedIds;
            rt::Tensor& workspace = prefill ? prefillSamplingWorkspace : decodeSamplingWorkspace;
            SamplingSlotPool::Slot& slot = samplingSlotPool.acquire();
            ELLM_CHECK(io.outputLogits.reshape({batchSize, config.outputVocabSize})
                    && selectedIds.reshape({batchSize, 1}) && slot.hostIds.reshape({batchSize}),
                "Semantic phase sampling reshape failed");
            rt::Tensor* samplingLogits = &io.outputLogits;
            bool const densePrefix = std::all_of(views.begin(), views.end(),
                [row = size_t{}](auto const& view) mutable { return view.phaseBatchRow == row++; });
            if (prefill && !densePrefix)
            {
                ELLM_CHECK(prefillCompactedLogits.reshape({batchSize, config.outputVocabSize}),
                    "Semantic phase compacted prefill logits reshape failed");
                size_t const rowBytes = static_cast<size_t>(config.outputVocabSize) * sizeof(float);
                auto const* source = static_cast<std::byte const*>(io.outputLogits.rawPointer());
                auto* destination = static_cast<std::byte*>(prefillCompactedLogits.rawPointer());
                for (size_t row{}; row < views.size(); ++row)
                {
                    CUDA_CHECK(cudaMemcpyAsync(destination + row * rowBytes,
                        source + views[row].phaseBatchRow * rowBytes, rowBytes, cudaMemcpyDeviceToDevice, stream));
                }
                samplingLogits = &prefillCompactedLogits;
            }
            selectAllTopK(*samplingLogits, std::nullopt, selectedIds, 1, workspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(slot.hostIds.rawPointer(), selectedIds.rawPointer(),
                static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaEventRecord(slot.ready, stream));
            auto ticket = std::make_unique<rt::IndependentPhaseSampleTicket>();
            ticket->ready = slot.ready;
            ticket->fromPrefill = prefill;
            for (rt::IndependentPhaseRequestView const& view : views)
            {
                ticket->requestIds.push_back(view.requestId);
            }
            if (!prefill)
            {
                lastDecodeSampleRequestIds = ticket->requestIds;
            }
            ticket->collect = [&slot, batchSize]() {
                int32_t const* selected = slot.hostIds.dataPointer<int32_t>();
                return std::vector<int32_t>(selected, selected + batchSize);
            };
            ticket->release = [&samplingSlotPool, &slot]() { samplingSlotPool.release(slot); };
            return ticket;
        };

        rt::IndependentPhaseRequestAdapter semanticAdapter;
        semanticAdapter.stagePrefill
            = [&](std::vector<rt::IndependentPhaseRequestView> const& views, rt::PipelineIO& io, rt::TensorMap& map,
                  cudaStream_t stream) { stageTokens(views, io, map, stream, true); };
        semanticAdapter.stageDecode
            = [&](std::vector<rt::IndependentPhaseRequestView> const& views, rt::PipelineIO& io, rt::TensorMap& map,
                  cudaStream_t stream) { stageTokens(views, io, map, stream, false); };
        semanticAdapter.submitSampling = submitSampling;
        bool const enablePrefixReuse = std::getenv("TRT_EDGELLM_ENABLE_PREFIX_REUSE") != nullptr;
        semanticAdapter.supportsPageAlignedPrefixReuse = enablePrefixReuse;

        rt::IndependentPhaseCoordinatorCallbacks seedCallbacks;
        seedCallbacks.isDecodeFinished = [](rt::PhaseWorkItem const&, int32_t) { return true; };
        rt::PhaseQueueSchedulerConfig semanticSchedulerConfig;
        semanticSchedulerConfig.maxPrefillBatchSize = config.maxSupportedPrefillBatchSize;
        semanticSchedulerConfig.maxDecodeBatchSize = config.maxSupportedDecodeBatchSize;
        if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_SCHEDULER"))
        {
            std::string const mode(value);
            ELLM_CHECK(mode == "disabled" || mode == "shadow" || mode == "active",
                "TRT_EDGELLM_GLOBAL_SCHEDULER must be disabled, shadow, or active");
            semanticSchedulerConfig.globalSchedulerMode = mode == "active" ? rt::PhaseGlobalSchedulerMode::kActive
                : mode == "shadow"                                         ? rt::PhaseGlobalSchedulerMode::kShadow
                                                                           : rt::PhaseGlobalSchedulerMode::kDisabled;
        }
        if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_DECODE_TPOT_TARGET_US"))
        {
            semanticSchedulerConfig.globalDecodeTpotTargetUs = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_DECODE_ROW_REPLACEMENT_COST_MS"))
        {
            semanticSchedulerConfig.decodeRowReplacementCostMs = std::stof(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_PREFILL_BATCH"))
        {
            semanticSchedulerConfig.maxPrefillBatchSize = std::stoi(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_CONTINUATION_PREFILL_BATCH"))
        {
            semanticSchedulerConfig.maxContinuationPrefillBatchSize = std::stoi(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_DECODE_BATCH"))
        {
            semanticSchedulerConfig.maxDecodeBatchSize = std::stoi(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_OVERLAP_PREFILL_BATCH"))
        {
            semanticSchedulerConfig.maxOverlapPrefillBatchSize = std::stoi(value);
        }
        ELLM_CHECK(semanticSchedulerConfig.maxPrefillBatchSize > 0
                && semanticSchedulerConfig.maxPrefillBatchSize <= config.maxSupportedPrefillBatchSize,
            "Semantic prefill batch cap is outside the engine profile");
        ELLM_CHECK(semanticSchedulerConfig.maxContinuationPrefillBatchSize >= 0
                && semanticSchedulerConfig.maxContinuationPrefillBatchSize
                    <= semanticSchedulerConfig.maxPrefillBatchSize,
            "Semantic continuation prefill batch cap is outside the configured prefill limit");
        ELLM_CHECK(semanticSchedulerConfig.maxDecodeBatchSize > 0
                && semanticSchedulerConfig.maxDecodeBatchSize <= config.maxSupportedDecodeBatchSize,
            "Semantic decode batch cap is outside the engine profile");
        semanticSchedulerConfig.maxPrefillChunkTokens = 128;
        semanticSchedulerConfig.maxOverlapPrefillTokens = 128;
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_OVERLAP_PREFILL_TOKENS"))
        {
            semanticSchedulerConfig.maxOverlapPrefillTokens = std::stoi(value);
        }
        semanticSchedulerConfig.maxPrefillBatchTokens = prefillBatchTokenBudget;
        semanticSchedulerConfig.enableRaggedPrefillBatching = config.packedPrefill;
        semanticSchedulerConfig.enablePackedPrefillTokenLayout = config.packedPrefill;
        semanticSchedulerConfig.prefillCompletionBonusTokens = 128;
        semanticSchedulerConfig.enableWavefrontPrefillBatching
            = std::getenv("TRT_EDGELLM_DISABLE_WAVEFRONT_PREFILL") == nullptr;
        semanticSchedulerConfig.maxPrefillCohortSize = semanticSchedulerConfig.maxPrefillBatchSize;
        semanticSchedulerConfig.maxPrefillCohortTurns = 8;
        semanticSchedulerConfig.enableAdaptivePrefillChunking = true;
        semanticSchedulerConfig.enableDecodeTpotTelemetry
            = std::getenv("TRT_EDGELLM_THROUGHPUT_MAX_ENCODED_VISION") != nullptr
            || std::getenv("TRT_EDGELLM_STEPWISE_ADMISSION") != nullptr
            || std::getenv("TRT_EDGELLM_PHASE_MEMORY_BROKER") != nullptr;
        semanticSchedulerConfig.minPrefillChunkTokens = 32;
        semanticSchedulerConfig.prefillChunkAlignment = 8;
        semanticSchedulerConfig.adaptivePrefillChunkCandidates = {32, 64, 128};
        if (char const* value = std::getenv("TRT_EDGELLM_FIXED_PREFILL_CHUNK"))
        {
            int32_t const fixedChunk = std::stoi(value);
            ELLM_CHECK(fixedChunk > 0 && fixedChunk <= 128, "Fixed prefill chunk is outside the engine profile");
            semanticSchedulerConfig.maxPrefillChunkTokens = fixedChunk;
            semanticSchedulerConfig.minPrefillChunkTokens = fixedChunk;
            semanticSchedulerConfig.enableAdaptivePrefillChunking = false;
            semanticSchedulerConfig.adaptivePrefillChunkCandidates.clear();
            if (prefillBatchTokenBudgetValue == nullptr)
            {
                semanticSchedulerConfig.maxPrefillBatchTokens
                    = semanticSchedulerConfig.maxPrefillBatchSize * fixedChunk;
            }
        }
        semanticSchedulerConfig.enableMetricsPolicy = std::getenv("TRT_EDGELLM_DISABLE_METRICS_POLICY") == nullptr;
        semanticSchedulerConfig.enablePrefillTtftHardGuard
            = std::getenv("TRT_EDGELLM_PREFILL_TTFT_HARD_GUARD") != nullptr;
        semanticSchedulerConfig.enableExternalDrainPreference
            = std::getenv("TRT_EDGELLM_PHASE_MEMORY_BROKER") != nullptr
            && std::getenv("TRT_EDGELLM_DISABLE_PHASE_MEMORY_DRAIN") == nullptr;
        if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_DRAIN_DWELL_DISPATCHES"))
        {
            semanticSchedulerConfig.externalDrainPreferenceMinDwellDispatches = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_DRAIN_MAX_CONSECUTIVE_DISPATCHES"))
        {
            semanticSchedulerConfig.externalDrainPreferenceMaxConsecutiveDispatches
                = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_PREFILL_DECODE_PRESSURE_LIMIT"))
        {
            semanticSchedulerConfig.externalPrefillDrainDecodePressureLimit = std::stof(value);
        }
        semanticSchedulerConfig.minMetricsSamples = 2;
        semanticSchedulerConfig.prefillQueueWaitTargetUs = 5000.0;
        semanticSchedulerConfig.decodeQueueWaitTargetUs = 2000.0;
        if (char const* value = std::getenv("TRT_EDGELLM_PREFILL_QUEUE_TARGET_US"))
        {
            semanticSchedulerConfig.prefillQueueWaitTargetUs = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_DECODE_QUEUE_TARGET_US"))
        {
            semanticSchedulerConfig.decodeQueueWaitTargetUs = std::stod(value);
        }
        bool const forceDynamicDecode = std::getenv("TRT_EDGELLM_ENABLE_DYNAMIC_DECODE") != nullptr;
        semanticSchedulerConfig.enableDynamicDecodeBatching = forceDynamicDecode
            || (config.packedPrefill && std::getenv("TRT_EDGELLM_DISABLE_DYNAMIC_DECODE") == nullptr);
        semanticSchedulerConfig.enableOnlineDecodeCostLearning = semanticSchedulerConfig.enableDynamicDecodeBatching
            && std::getenv("TRT_EDGELLM_DISABLE_ONLINE_COST") == nullptr;
        bool const asymmetricDecodeStaging = maxStableSlots > semanticSchedulerConfig.maxDecodeBatchSize
            && semanticSchedulerConfig.maxDecodeBatchSize > semanticSchedulerConfig.maxPrefillBatchSize;
        semanticSchedulerConfig.enableDecodeCohortBatching = semanticSchedulerConfig.enableDynamicDecodeBatching
            && asymmetricDecodeStaging && std::getenv("TRT_EDGELLM_DISABLE_DECODE_COHORT") == nullptr;
        if (std::getenv("TRT_EDGELLM_ENABLE_DECODE_COHORT") != nullptr)
        {
            semanticSchedulerConfig.enableDecodeCohortBatching = true;
        }
        semanticSchedulerConfig.decodeBatchCosts = {
            {1, 2048, 6.238F},
            {2, 2048, 6.294F},
            {3, 2048, 6.370F},
            {4, 2048, 6.321F},
            {5, 2048, 6.402F},
            {6, 2048, 6.360F},
            {7, 2048, 6.467F},
            {8, 2048, 6.467F},
            {9, 2048, 6.680F},
            {10, 2048, 6.617F},
            {11, 2048, 6.767F},
            {12, 2048, 6.712F},
            {13, 2048, 6.924F},
            {14, 2048, 6.790F},
            {15, 2048, 6.904F},
            {16, 2048, 6.862F},
            {17, 1024, 6.927F},
            {18, 1024, 6.988F},
            {19, 1024, 7.016F},
            {20, 1024, 7.129F},
            {21, 1024, 7.105F},
            {22, 1024, 7.082F},
            {23, 1024, 7.106F},
            {24, 1024, 7.178F},
            {25, 1024, 7.291F},
            {26, 1024, 7.278F},
            {27, 1024, 7.341F},
            {28, 1024, 7.264F},
            {29, 1024, 7.374F},
            {30, 1024, 7.435F},
            {31, 1024, 7.523F},
            {32, 1024, 7.484F},
        };
        if (config.maxSupportedDecodeBatchSize > 32)
        {
            // P8/D64 undercommitted-pool profile, measured from decode-only CUDA-event samples.
            semanticSchedulerConfig.decodeBatchCosts.insert(semanticSchedulerConfig.decodeBatchCosts.end(),
                {{40, 2048, 7.949F}, {47, 2048, 8.126F}, {63, 2048, 9.861F}, {64, 2048, 9.764F}});
        }
        if (char const* value = std::getenv("TRT_EDGELLM_SCHEDULER_COST_JSON"))
        {
            loadSchedulerCostModel(value, semanticSchedulerConfig);
        }
        if (semanticSchedulerConfig.globalSchedulerMode != rt::PhaseGlobalSchedulerMode::kDisabled)
        {
            semanticSchedulerConfig.globalMemoryHorizonSupplier
                = [&ownership](rt::PhaseGlobalActionKey const&, std::vector<uint64_t> const&) {
                      rt::StableKVPageManager::Config const& ownershipConfig = ownership.config();
                      rt::PhaseActionMemoryHorizon horizon;
                      horizon.managedBytes = static_cast<size_t>(ownershipConfig.numPages - ownership.availablePages());
                      horizon.budgetBytes = static_cast<size_t>(ownershipConfig.numPages);
                      // Request admission and growth-owner leases reserve physical
                      // pages before a phase becomes runnable, so dispatch itself
                      // introduces no unreserved KV allocation here.
                      return horizon;
                  };
        }
        rt::IndependentPhaseCoordinator semanticCoordinator(config, semanticSchedulerConfig, *pair, ownership,
            *prefillIO, *decodeIO, prefillMap, decodeMap, prefillStream, decodeStream, std::move(seedCallbacks));
        if (std::getenv("TRT_EDGELLM_ENABLE_PERSISTENT_DECODE_SELECT") != nullptr)
        {
            semanticCoordinator.setPersistentDecodeSelectEnabled(true);
        }
        if (std::getenv("TRT_EDGELLM_ENABLE_PERSISTENT_PAGE_BINDINGS") != nullptr)
        {
            semanticCoordinator.setPersistentPageBindingsEnabled(true);
        }
        rt::IndependentPhaseServerConfig serverConfig;
        size_t maxInFlightRequests = std::min(kDEFAULT_MAX_INFLIGHT_REQUESTS, static_cast<size_t>(maxStableSlots));
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_INFLIGHT"))
        {
            maxInFlightRequests = static_cast<size_t>(std::stoul(value));
        }
        ELLM_CHECK(maxInFlightRequests > 0 && maxInFlightRequests <= static_cast<size_t>(maxStableSlots),
            "Phase server in-flight capacity must be in the stable slot range");
        serverConfig.maxInFlightRequests = maxInFlightRequests;
        serverConfig.defaultMaxOutputTokens = kSEMANTIC_OUTPUT_TOKENS;
        if (std::getenv("TRT_EDGELLM_IGNORE_EOS") == nullptr)
        {
            serverConfig.eosTokenIds = config.eosTokenIds;
        }
        serverConfig.enablePrefixReuse = enablePrefixReuse;
        serverConfig.enableCudaGraphs = std::getenv("TRT_EDGELLM_CAPTURE_PHASE_GRAPHS") != nullptr;
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_PREFILL_GRAPHS"))
        {
            serverConfig.maxPrefillGraphs = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_DECODE_GRAPHS"))
        {
            serverConfig.maxDecodeGraphs = static_cast<size_t>(std::stoul(value));
        }
        serverConfig.allowBatchedVisionPrefill = enableBatchedVisionPrefill;
        serverConfig.releaseVisionPrefillStorage = std::getenv("TRT_EDGELLM_RELEASE_VISION_PREFILL_STORAGE") != nullptr;
        serverConfig.maxPendingRequests = 1024;
        serverConfig.enableGlobalWaitActions
            = semanticSchedulerConfig.globalSchedulerMode != rt::PhaseGlobalSchedulerMode::kDisabled;
        if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_SAMPLING_COLD_START_US"))
        {
            serverConfig.globalSamplingColdStartUs = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_SAMPLING_WINDOW"))
        {
            serverConfig.globalSamplingLatencyWindow = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_DECODE_REFILL_BATCH"))
        {
            serverConfig.decodeRefillBatchSize = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PREFILL_FORMATION_BATCH"))
        {
            serverConfig.prefillFormationBatchSize = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PREFILL_FORMATION_WINDOW_US"))
        {
            serverConfig.prefillFormationWindowUs = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PREFILL_FORMATION_TTFT_TARGET_US"))
        {
            serverConfig.prefillFormationTtftTargetUs = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PREFILL_FORMATION_MAX_OUTPUT_TOKENS"))
        {
            serverConfig.prefillFormationMaxOutputTokens = std::stoi(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PREFILL_FORMATION_TTFT_GUARD_US"))
        {
            serverConfig.prefillFormationTtftGuardUs = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PREFILL_FORMATION_COST_JSON"))
        {
            loadPrefillFormationCostModel(value, serverConfig);
        }
        serverConfig.enableAdaptiveAdmission = std::getenv("TRT_EDGELLM_ADAPTIVE_ADMISSION") != nullptr;
        if (char const* value = std::getenv("TRT_EDGELLM_LATENCY_INFLIGHT"))
        {
            serverConfig.latencyInFlightRequests = static_cast<size_t>(std::stoul(value));
        }
        serverConfig.enableStepwiseAdaptiveAdmission = std::getenv("TRT_EDGELLM_STEPWISE_ADMISSION") != nullptr;
        serverConfig.enableDelayedExternalProfileSelection
            = std::getenv("TRT_EDGELLM_DELAY_EXTERNAL_PROFILE_SELECTION") != nullptr;
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_STEP"))
        {
            serverConfig.adaptiveAdmissionStep = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_DWELL_SAMPLES"))
        {
            serverConfig.adaptiveAdmissionDwellSamples = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_TPOT_ENTER"))
        {
            serverConfig.adaptiveAdmissionTpotPressureEnterRatio = std::stof(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_TPOT_EXIT"))
        {
            serverConfig.adaptiveAdmissionTpotPressureExitRatio = std::stof(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_MIN_FREE_PAGES"))
        {
            serverConfig.adaptiveAdmissionMinFreePages = std::stoi(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_TPOT_COSTS"))
        {
            serverConfig.adaptiveAdmissionCosts = parseAdmissionCosts(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_TPOT_BUDGET_US"))
        {
            serverConfig.adaptiveAdmissionTpotBudgetUs = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_EXTERNAL_TPOT_COSTS"))
        {
            serverConfig.adaptiveAdmissionExternalCosts = parseAdmissionCosts(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_EXTERNAL_TPOT_BUDGET_US"))
        {
            serverConfig.adaptiveAdmissionExternalTpotBudgetUs = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_EXTERNAL_MIN_SHARE"))
        {
            serverConfig.adaptiveAdmissionExternalRequestFraction = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_EXTERNAL_MIN_PREFILL_TOKENS"))
        {
            serverConfig.adaptiveAdmissionExternalPrefillTokens = static_cast<size_t>(std::stoull(value));
        }
        if (char const* reservation = std::getenv("TRT_EDGELLM_PAGE_RESERVATION");
            reservation != nullptr && std::string(reservation) == "headroom")
        {
            serverConfig.pageReservationMode = rt::IndependentPhasePageReservationMode::kHeadroom;
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PAGE_RESERVATION_HEADROOM_TOKENS"))
        {
            serverConfig.outputHeadroomTokens = std::stoi(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_PAGE_GROWTH_REQUESTS"))
        {
            serverConfig.maxConcurrentPageGrowthRequests = std::stoi(value);
        }
        std::unique_ptr<rt::PhasePrefixReuseCache> semanticPrefixCache;
        if (enablePrefixReuse)
        {
            semanticPrefixCache = std::make_unique<rt::PhasePrefixReuseCache>(ownership, 1);
        }
        rt::IndependentPhaseAsyncServer semanticServer(
            serverConfig, semanticCoordinator, ownership, std::move(semanticAdapter), semanticPrefixCache.get());
        bool const ipcMode = std::getenv("TRT_EDGELLM_PHASE_IPC") != nullptr;
        bool const prefixReuseGate = std::getenv("TRT_EDGELLM_PREFIX_REUSE_GATE") != nullptr;
        char const* visionEngineDir = std::getenv("TRT_EDGELLM_VISION_ENGINE_DIR");
        char const* visionImagePath = std::getenv("TRT_EDGELLM_VISION_IMAGE");
        rt::PhaseVisionStoragePolicy visionStoragePolicy;
        visionStoragePolicy.splitMropeLease = serverConfig.releaseVisionPrefillStorage;
        if (char const* value = std::getenv("TRT_EDGELLM_VISION_IDLE_SLABS"))
        {
            visionStoragePolicy.maxIdleBatches = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_VISION_IDLE_SLAB_BYTES"))
        {
            visionStoragePolicy.maxIdleBytes = static_cast<size_t>(std::stoull(value));
        }
        size_t tieredVisionExclusiveInputTokens{};
        auto configureVisionContextMemory = [&](rt::MultimodalRunner& runner) {
            int64_t const requiredBytes = runner.getRequiredContextMemorySize();
            LOG_INFO("Vision context workspace: required=%lld prefill_available=%zu bytes",
                static_cast<long long>(requiredBytes), pair->prefillContextMemory().getMemoryCapacity());
            if (std::getenv("TRT_EDGELLM_TIERED_VISION_CONTEXT_MEMORY") != nullptr)
            {
                int32_t const profileCount = runner.getOptimizationProfileCount();
                ELLM_CHECK(profileCount >= 2, "Tiered vision context memory requires at least two visual profiles");
                int32_t const largeProfile = profileCount - 1;
                rt::TieredVisionContextMemoryInfo const info
                    = pair->configureTieredVisionContextMemory(runner, 0, largeProfile);
                tieredVisionExclusiveInputTokens = static_cast<size_t>(runner.getInputTokenLimitForProfile(0));
                LOG_INFO(
                    "Tiered E/P context arena: total=%lld prefill=%lld small_vision=%lld large_vision=%lld "
                    "exclusive_above_input_tokens=%zu",
                    static_cast<long long>(info.arenaBytes), static_cast<long long>(info.prefillBytes),
                    static_cast<long long>(info.smallVisionBytes), static_cast<long long>(info.largeVisionBytes),
                    tieredVisionExclusiveInputTokens);
                return;
            }
            runner.allocateContextMemory();
        };
        if (visionEngineDir != nullptr && visionImagePath != nullptr)
        {
            cudaStream_t encoderStream{};
            if (enablePhaseStreamPriorities)
            {
                CUDA_CHECK(cudaStreamCreateWithPriority(&encoderStream, cudaStreamNonBlocking, leastPriority));
            }
            else
            {
                CUDA_CHECK(cudaStreamCreateWithFlags(&encoderStream, cudaStreamNonBlocking));
            }
            {
                auto runner = rt::MultimodalRunner::create(visionEngineDir, config.maxSupportedBatchSize,
                    config.maxKVCacheCapacity, encoderStream, checkpointDir);
                configureVisionContextMemory(*runner);
                rt::PhaseVisionAdapter visionAdapter(*runner, tokenizer, config, encoderStream, visionStoragePolicy);
                rt::PhaseThreeCoordinator threePhase(visionAdapter, semanticServer);

                rt::LLMGenerationRequest request{};
                rt::LLMGenerationRequest::Request logicalRequest;
                logicalRequest.messages.push_back(
                    {"user", {{"image", visionImagePath}, {"text", "Describe the image briefly."}}});
                logicalRequest.imageBuffers.push_back(rt::imageUtils::loadImageFromFile(visionImagePath));
                request.requests.push_back(std::move(logicalRequest));
                request.temperature = 0.0F;
                request.topP = 1.0F;
                request.topK = 1;
                request.maxGenerateLength = kSEMANTIC_OUTPUT_TOKENS;
                request.applyChatTemplate = true;
                request.addGenerationPrompt = true;
                auto const submitted = threePhase.submit(23000, std::move(request), kSEMANTIC_OUTPUT_TOKENS);
                ELLM_CHECK(submitted == rt::PhaseThreeSubmissionStatus::kEncoding,
                    "Three-phase vision request did not enter the encoder");
                size_t pollCount{};
                while (!threePhase.empty())
                {
                    static_cast<void>(threePhase.poll());
                    ELLM_CHECK(++pollCount < 1000000U, "Three-phase vision request exceeded its poll guard");
                }
                auto completion = threePhase.tryPopCompletion();
                ELLM_CHECK(completion.has_value() && !completion->generatedTokens.empty(),
                    "Three-phase vision request produced no output");
                LOG_INFO("Three-phase vision request passed: output='%s'",
                    tokenizer.decode(completion->generatedTokens, false).c_str());
            }
            CUDA_CHECK(cudaStreamDestroy(encoderStream));
        }
        else if (prefixReuseGate)
        {
            ELLM_CHECK(enablePrefixReuse && semanticPrefixCache != nullptr,
                "Prefix reuse gate requires TRT_EDGELLM_ENABLE_PREFIX_REUSE=1");
            std::string longPrompt;
            for (int32_t index{}; index < 160; ++index)
            {
                longPrompt += " reusable";
            }
            longPrompt += " Explain how stable prefixes reduce latency.";
            rt::LLMGenerationRequest::Request request;
            request.messages.push_back({"user", {{"text", longPrompt}}});
            rt::LLMGenerationRequest::FormattedRequest formatted;
            ELLM_CHECK(tokenizer.applyChatTemplate(request, formatted, true, true, false),
                "Failed to format prefix reuse gate request");
            std::vector<int32_t> const tokenIds = tokenizer.encode(formatted.formattedCompleteRequest, false);
            auto const sourceSubmission = semanticServer.submit(22000, tokenIds, kSEMANTIC_OUTPUT_TOKENS);
            ELLM_CHECK(sourceSubmission.status == rt::IndependentPhaseServerStatus::kAdmitted
                    && sourceSubmission.reusedPrefixTokens == 0,
                "Prefix reuse gate source admission failed");
            semanticServer.runUntilIdle(100000);
            auto sourceCompletion = semanticServer.tryPopCompletion();
            ELLM_CHECK(sourceCompletion.has_value(), "Prefix reuse gate source did not complete");
            while (semanticServer.tryPopToken().has_value())
            {
            }
            auto const targetSubmission = semanticServer.submit(22001, tokenIds, kSEMANTIC_OUTPUT_TOKENS);
            ELLM_CHECK(targetSubmission.status == rt::IndependentPhaseServerStatus::kAdmitted
                    && targetSubmission.reusedPrefixTokens >= 128,
                "Prefix reuse gate target did not reuse a complete page");
            semanticServer.runUntilIdle(100000);
            auto targetCompletion = semanticServer.tryPopCompletion();
            ELLM_CHECK(
                targetCompletion.has_value() && targetCompletion->generatedTokens == sourceCompletion->generatedTokens,
                "Prefix reuse changed greedy output");
            semanticPrefixCache->clear();
            ELLM_CHECK(
                ownership.availableSlots() == maxStableSlots, "Prefix reuse gate did not release every stable slot");
            LOG_INFO(
                "Prefix reuse off/on greedy output gate passed: reused_tokens=%d", targetSubmission.reusedPrefixTokens);
        }
        else if (ipcMode)
        {
            // Bound synchronous request parsing before polling the GPU phases. In
            // particular, image loading must not drain an entire burst while the
            // device is idle. Half a prefill cohort leaves enough rows for useful
            // batching while overlapping the remaining ingress work with CUDA.
            size_t ipcIngressQuantum = std::min(serverConfig.maxPendingRequests,
                std::max<size_t>(1U, static_cast<size_t>(semanticSchedulerConfig.maxPrefillBatchSize) / 2U));
            if (char const* value = std::getenv("TRT_EDGELLM_IPC_INGRESS_QUANTUM"))
            {
                ipcIngressQuantum = static_cast<size_t>(std::stoul(value));
            }
            ELLM_CHECK(ipcIngressQuantum > 0, "Phase IPC ingress quantum must be positive");
            bool const emitPhaseMetrics = std::getenv("TRT_EDGELLM_EMIT_PHASE_METRICS") != nullptr;
            semanticCoordinator.setMetricsCollectionEnabled(emitPhaseMetrics);
            size_t const warmupAdmissionLimit = serverConfig.enableAdaptiveAdmission
                ? serverConfig.latencyInFlightRequests
                : serverConfig.maxInFlightRequests;
            int32_t const warmupBatchLimit
                = std::min(semanticSchedulerConfig.maxDecodeBatchSize, static_cast<int32_t>(warmupAdmissionLimit));
            std::vector<int32_t> requestedWarmupBatchSizes;
            if (char const* value = std::getenv("TRT_EDGELLM_IPC_WARMUP_DECODE_BATCHES"))
            {
                std::stringstream stream(value);
                std::string batchSize;
                while (std::getline(stream, batchSize, ','))
                {
                    ELLM_CHECK(!batchSize.empty(), "Phase IPC warmup batch list contains an empty entry");
                    requestedWarmupBatchSizes.push_back(std::stoi(batchSize));
                }
            }
            std::vector<int32_t> const warmupBatchSizes = std::getenv("TRT_EDGELLM_DISABLE_IPC_SHAPE_WARMUP") == nullptr
                ? rt::phaseServingWarmupBatchSizes(warmupBatchLimit, std::move(requestedWarmupBatchSizes))
                : std::vector<int32_t>{};
            uint64_t warmupRequestId = 1000000;
            size_t warmedRequests{};
            for (int32_t const batchSize : warmupBatchSizes)
            {
                for (int32_t row{}; row < batchSize; ++row)
                {
                    auto const submission = semanticServer.submit(warmupRequestId++, semanticPrompts.at(20000), 2);
                    ELLM_CHECK(submission.status == rt::IndependentPhaseServerStatus::kAdmitted,
                        "Phase IPC shape warmup request was not admitted");
                }
                semanticServer.runUntilIdle(1000000);
                size_t completed{};
                while (semanticServer.tryPopCompletion().has_value())
                {
                    ++completed;
                }
                while (semanticServer.tryPopToken().has_value())
                {
                }
                if (semanticPrefixCache != nullptr)
                {
                    semanticPrefixCache->clear();
                }
                ELLM_CHECK(completed == static_cast<size_t>(batchSize),
                    "Phase IPC shape warmup did not complete every request");
                warmedRequests += completed;
            }
            ELLM_CHECK(semanticServer.empty() && ownership.availableSlots() == maxStableSlots,
                "Phase IPC shape warmup did not release every stable slot");
            // Shape priming is not production traffic. Keep graph entries, but
            // do not let synthetic queue waits drive adaptive admission.
            semanticCoordinator.scheduler().resetHistory();
            if (serverConfig.enableCudaGraphs && std::getenv("TRT_EDGELLM_ONLINE_GRAPH_CAPTURE") == nullptr)
            {
                // Retain the primed graph cache, but do not synchronously capture
                // unseen production shapes on their latency-critical first request.
                semanticCoordinator.setGraphCaptureEnabled(false);
            }
            else if (serverConfig.enableCudaGraphs)
            {
                size_t graphCaptureMinObservations = 8U;
                if (char const* value = std::getenv("TRT_EDGELLM_GRAPH_CAPTURE_MIN_OBSERVATIONS"))
                {
                    graphCaptureMinObservations = static_cast<size_t>(std::stoul(value));
                }
                semanticCoordinator.setGraphCaptureMinObservations(graphCaptureMinObservations);
            }
            LOG_INFO("Phase IPC shape warmup: batches=%zu requests=%zu", warmupBatchSizes.size(), warmedRequests);
            cudaStream_t ipcEncoderStream{};
            std::unique_ptr<rt::MultimodalRunner> ipcVisionRunner;
            std::unique_ptr<rt::PhaseVisionAdapter> ipcVisionAdapter;
            std::unique_ptr<rt::PhaseThreeCoordinator> ipcThreePhase;
            if (visionEngineDir != nullptr)
            {
                ELLM_CHECK(
                    !config.packedPrefill || config.maxPackedPrefillChunkTokens >= config.maxSupportedInputLength,
                    "Three-phase packed vision requires an atomic packed-prefill chunk covering maxInputLength");
                if (enablePhaseStreamPriorities)
                {
                    CUDA_CHECK(cudaStreamCreateWithPriority(&ipcEncoderStream, cudaStreamNonBlocking, leastPriority));
                }
                else
                {
                    CUDA_CHECK(cudaStreamCreateWithFlags(&ipcEncoderStream, cudaStreamNonBlocking));
                }
                ipcVisionRunner = rt::MultimodalRunner::create(visionEngineDir, config.maxSupportedBatchSize,
                    config.maxKVCacheCapacity, ipcEncoderStream, checkpointDir);
                configureVisionContextMemory(*ipcVisionRunner);
                ipcVisionAdapter = std::make_unique<rt::PhaseVisionAdapter>(
                    *ipcVisionRunner, tokenizer, config, ipcEncoderStream, visionStoragePolicy);
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_DEBUG_DIR"))
                {
                    std::filesystem::path const debugDirectory(value);
                    std::filesystem::create_directories(debugDirectory);
                    ipcVisionAdapter->setDebugCallback([debugDirectory](rt::PhaseVisionDebugSnapshot const& snapshot) {
                        std::string const stem = "request-" + std::to_string(snapshot.requestId) + "-bs"
                            + std::to_string(snapshot.encoderBatchSize) + "-row"
                            + std::to_string(snapshot.encoderBatchIndex);
                        auto const writeTensor = [&](std::string const& name,
                                                     rt::PhaseVisionDebugTensor const& tensor) {
                            std::filesystem::path const path = debugDirectory / (stem + "-" + name + ".bin");
                            std::ofstream stream(path, std::ios::binary);
                            ELLM_CHECK(stream.good(), "Failed to open phase vision debug tensor: " + path.string());
                            stream.write(reinterpret_cast<char const*>(tensor.bytes.data()),
                                static_cast<std::streamsize>(tensor.bytes.size()));
                            ELLM_CHECK(stream.good(), "Failed to write phase vision debug tensor: " + path.string());
                            return path.filename().string();
                        };
                        std::string const embeddingFile = writeTensor("embedding", snapshot.outputEmbedding);
                        std::string const mropeFile = writeTensor("mrope", snapshot.mropeCosSin);
                        nlohmann::json const metadata{{"request_id", snapshot.requestId},
                            {"encoder_batch_size", snapshot.encoderBatchSize},
                            {"encoder_batch_index", snapshot.encoderBatchIndex}, {"token_ids", snapshot.tokenIds},
                            {"embedding_file", embeddingFile}, {"embedding_shape", snapshot.outputEmbedding.shape},
                            {"embedding_data_type", static_cast<int32_t>(snapshot.outputEmbedding.dataType)},
                            {"mrope_file", mropeFile}, {"mrope_shape", snapshot.mropeCosSin.shape},
                            {"mrope_data_type", static_cast<int32_t>(snapshot.mropeCosSin.dataType)}};
                        std::filesystem::path const metadataPath = debugDirectory / (stem + ".json");
                        std::ofstream metadataStream(metadataPath);
                        ELLM_CHECK(metadataStream.good(),
                            "Failed to open phase vision debug metadata: " + metadataPath.string());
                        metadataStream << metadata.dump(2) << '\n';
                        ELLM_CHECK(metadataStream.good(),
                            "Failed to write phase vision debug metadata: " + metadataPath.string());
                    });
                }
                rt::PhaseThreeCoordinatorConfig threePhaseConfig;
                threePhaseConfig.globalSchedulerMode = semanticSchedulerConfig.globalSchedulerMode;
                if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER"))
                {
                    threePhaseConfig.globalSafeProbeSlackMultiplier = std::stof(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_SAFE_PROBE_INTERVAL"))
                {
                    threePhaseConfig.globalSafeProbeInterval = static_cast<size_t>(std::stoull(value));
                }
                threePhaseConfig.exclusiveEncoderInputTokenThreshold = tieredVisionExclusiveInputTokens;
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_EXCLUSIVE_INPUT_TOKENS"))
                {
                    threePhaseConfig.exclusiveEncoderInputTokenThreshold = static_cast<size_t>(std::stoull(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_MAX_ENCODED_VISION"))
                {
                    threePhaseConfig.maxEncodedInFlight = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_THROUGHPUT_MAX_ENCODED_VISION"))
                {
                    threePhaseConfig.throughputMaxEncodedInFlight = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_MAX_ENCODED_VISION_BYTES"))
                {
                    threePhaseConfig.maxEncodedBytes = static_cast<size_t>(std::stoull(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_BATCH_SIZE"))
                {
                    threePhaseConfig.maxEncoderBatchSize = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_MAX_MEDIA"))
                {
                    threePhaseConfig.maxEncoderMediaItems = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_MAX_INPUT_BYTES"))
                {
                    threePhaseConfig.maxEncoderInputBytes = static_cast<size_t>(std::stoull(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_MAX_INPUT_TOKENS"))
                {
                    threePhaseConfig.maxEncoderInputTokens = static_cast<size_t>(std::stoull(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_BATCH_WAIT_US"))
                {
                    threePhaseConfig.encoderBatchWaitUs = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_CREDIT_WAIT_US"))
                {
                    threePhaseConfig.encoderCreditWaitUs = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_CREDIT_TARGET"))
                {
                    threePhaseConfig.encoderCreditTargetBatchSize = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_HOMOGENEOUS_BATCHING"))
                {
                    threePhaseConfig.enableHomogeneousEncoderBatching = std::stoi(value) != 0;
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_FIT_LOOKAHEAD"))
                {
                    threePhaseConfig.enableEncoderFitLookahead = std::stoi(value) != 0;
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_MAX_LOOKAHEAD"))
                {
                    threePhaseConfig.maxEncoderLookahead = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_PREFIX_PREFILL"))
                {
                    threePhaseConfig.enablePrefixBeforeVisionPrefill = std::stoi(value) != 0;
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_PREFIX_MIN_TOKENS"))
                {
                    threePhaseConfig.minPrefixBeforeVisionTokens = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_PREFILL_BATCH_SIZE"))
                {
                    threePhaseConfig.maxPrefillBatchSize = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_PREFILL_BATCH_TOKENS"))
                {
                    threePhaseConfig.maxPrefillBatchTokens = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_PREFILL_BATCH_WAIT_US"))
                {
                    threePhaseConfig.prefillBatchWaitUs = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ADAPTIVE_PREFILL"))
                {
                    threePhaseConfig.enableAdaptivePrefillAdmission = std::stoi(value) != 0;
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ADAPTIVE_PREFILL_MIN_BATCH"))
                {
                    threePhaseConfig.adaptivePrefillMinBatchSize = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_PREFILL_TPOT_PRESSURE_LIMIT"))
                {
                    threePhaseConfig.prefillDecodeTpotPressureLimit = std::stof(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_PREFILL_DEFER_ON_TPOT"))
                {
                    threePhaseConfig.enableDecodeProtectedPrefillDeferral = std::stoi(value) != 0;
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_PREFILL_MAX_DEFER_US"))
                {
                    threePhaseConfig.maxDecodeProtectedPrefillWaitUs = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_PREFILL_BYTE_PRESSURE_RATIO"))
                {
                    threePhaseConfig.prefillReadyBytePressureRatio = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_TTFT_TARGET_MS"))
                {
                    threePhaseConfig.visionTtftTargetUs = std::stod(value) * 1000.0;
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_LOOKAHEAD_ESCALATION_RATIO"))
                {
                    threePhaseConfig.lookaheadEscalationRatio = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_LOOKAHEAD_TPOT_PRESSURE_LIMIT"))
                {
                    threePhaseConfig.lookaheadDecodeTpotPressureLimit = std::stof(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_LOOKAHEAD_TPOT_PRESSURE_RECOVERY_LIMIT"))
                {
                    threePhaseConfig.lookaheadDecodeTpotPressureRecoveryLimit = std::stof(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_DECODE_TPOT_TARGET_MS"))
                {
                    threePhaseConfig.encodedCapacityDecodeTpotTargetUs = std::stod(value) * 1000.0;
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODED_CAPACITY_DWELL_MS"))
                {
                    threePhaseConfig.encodedCapacityMinDwellUs = std::stod(value) * 1000.0;
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODED_CAPACITY_BACKLOG"))
                {
                    threePhaseConfig.encodedCapacityBacklogEnterRequests = static_cast<size_t>(std::stoul(value));
                }
                threePhaseConfig.enableEncoderDispatchArbitration
                    = std::getenv("TRT_EDGELLM_VISION_ENCODER_ARBITER") != nullptr;
                threePhaseConfig.enableAsyncEncoderPreparation
                    = std::getenv("TRT_EDGELLM_VISION_ASYNC_PREPARATION") != nullptr;
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_INITIAL_COST_US"))
                {
                    threePhaseConfig.encoderDispatchInitialCostUs = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_COST_MARGIN_US"))
                {
                    threePhaseConfig.encoderDispatchCostSafetyMarginUs = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_TEXT_GUARD_US"))
                {
                    threePhaseConfig.encoderDispatchTextGuardAgeUs = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_DECODE_PRESSURE_LIMIT"))
                {
                    threePhaseConfig.encoderDispatchDecodeTpotPressureLimit = std::stof(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_MAX_DEFER_US"))
                {
                    threePhaseConfig.encoderDispatchMaxDeferUs = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_FORCE_INTERVAL_US"))
                {
                    threePhaseConfig.encoderDispatchForcedIntervalUs = std::stod(value);
                }
                threePhaseConfig.enableDeadlineAwareEncoderSerialization
                    = std::getenv("TRT_EDGELLM_VISION_ENCODER_DEADLINE_SERIALIZATION") != nullptr;
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_DEADLINE_RATIO"))
                {
                    threePhaseConfig.encoderSerializationDeadlineRatio = std::stod(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_VISION_ENCODER_SERIALIZED_BURST"))
                {
                    threePhaseConfig.encoderSerializationMaxBurst = static_cast<size_t>(std::stoul(value));
                }
                threePhaseConfig.memoryBroker.enabled = std::getenv("TRT_EDGELLM_PHASE_MEMORY_BROKER") != nullptr;
                threePhaseConfig.memoryBroker.committedKVPages = config.kvPoolPages;
                constexpr size_t kTOKENS_PER_KV_PAGE = 128U;
                threePhaseConfig.memoryBroker.bytesPerKVPage = static_cast<size_t>(config.numAttentionLayers) * 2U
                    * kTOKENS_PER_KV_PAGE * static_cast<size_t>(config.numKVHeads) * static_cast<size_t>(config.headDim)
                    * rt::utils::getTypeSize(config.kvCacheDtype);
                if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_KV_RESERVE_PAGES"))
                {
                    threePhaseConfig.memoryBroker.kvReservePages = std::stoi(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_KV_PRESSURE_PAGES"))
                {
                    threePhaseConfig.memoryBroker.kvPressurePages = std::stoi(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_PRESSURE_ENCODER_BATCH"))
                {
                    threePhaseConfig.memoryBroker.pressureMaxEncoderBatchSize = static_cast<size_t>(std::stoul(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_MAX_BYTES"))
                {
                    threePhaseConfig.memoryBroker.maxManagedBytes = static_cast<size_t>(std::stoull(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_PHASE_MEMORY_SAFETY_BYTES"))
                {
                    threePhaseConfig.memoryBroker.safetyReserveBytes = static_cast<size_t>(std::stoull(value));
                }
                char const* encoderCostPath = std::getenv("TRT_EDGELLM_VISION_ENCODER_COST_JSON");
                if (encoderCostPath == nullptr)
                {
                    encoderCostPath = std::getenv("TRT_EDGELLM_SCHEDULER_COST_JSON");
                }
                if (encoderCostPath != nullptr
                    && std::getenv("TRT_EDGELLM_DISABLE_COST_AWARE_ENCODER_BATCHING") == nullptr)
                {
                    loadEncoderCostModel(encoderCostPath, threePhaseConfig);
                }
                ipcThreePhase
                    = std::make_unique<rt::PhaseThreeCoordinator>(*ipcVisionAdapter, semanticServer, threePhaseConfig);
            }
            std::deque<rt::PhaseTimelineEvent> phaseTimelineEvents;
            std::deque<rt::PhaseVisionEncoderBatchMetric> encoderBatchMetrics;
            if (emitPhaseMetrics)
            {
                auto const timelineCallback
                    = [&](rt::PhaseTimelineEvent const& event) { phaseTimelineEvents.push_back(event); };
                semanticServer.setTimelineCallback(timelineCallback);
                if (ipcThreePhase != nullptr)
                {
                    ipcThreePhase->setTimelineCallback(timelineCallback);
                    ipcThreePhase->setEncoderBatchMetricCallback([&](rt::PhaseVisionEncoderBatchMetric const& metric) {
                        encoderBatchMetrics.push_back(metric);
                    });
                }
            }
            bool const asyncRequestAdapter = std::getenv("TRT_EDGELLM_IPC_ASYNC_REQUEST_ADAPTER") != nullptr;
            size_t requestAdapterWorkers = 4U;
            if (char const* value = std::getenv("TRT_EDGELLM_IPC_REQUEST_ADAPTER_WORKERS"))
            {
                requestAdapterWorkers = static_cast<size_t>(std::stoull(value));
            }
            ELLM_CHECK(requestAdapterWorkers > 0, "IPC request adapter worker count must be positive");
            std::deque<std::string> pendingLines;
            std::deque<PhaseIpcInput> pendingInputs;
            std::deque<std::future<PhaseIpcInput>> pendingInputTasks;
            std::mutex pendingMutex;
            std::condition_variable inputAdapterReady;
            bool inputClosed{};
            using PhaseOutputRecord
                = std::variant<std::string, rt::IndependentPhaseServerToken, rt::IndependentPhaseServerCompletion>;
            std::deque<PhaseOutputRecord> outputRecords;
            std::mutex outputMutex;
            std::condition_variable outputReady;
            bool outputClosed{};
            size_t outputWriteBatches{};
            size_t outputWriteRecords{};
            size_t outputWriteBytes{};
            double outputSerializationUs{};
            std::unordered_map<int32_t, std::string> decodedTokenTextCache;
            decodedTokenTextCache.reserve(4096U);
            size_t decodedTokenTextCacheHits{};
            size_t decodedTokenTextCacheMisses{};
            std::thread outputWriter([&]() {
                while (true)
                {
                    std::unique_lock<std::mutex> lock(outputMutex);
                    outputReady.wait(lock, [&]() { return outputClosed || !outputRecords.empty(); });
                    std::deque<PhaseOutputRecord> readyRecords;
                    readyRecords.swap(outputRecords);
                    bool const closed = outputClosed;
                    lock.unlock();
                    if (!readyRecords.empty())
                    {
                        auto const serializationStart = std::chrono::steady_clock::now();
                        std::vector<std::string> readyLines;
                        readyLines.reserve(readyRecords.size());
                        for (PhaseOutputRecord& record : readyRecords)
                        {
                            if (auto* line = std::get_if<std::string>(&record))
                            {
                                readyLines.push_back(std::move(*line));
                            }
                            else if (auto* token = std::get_if<rt::IndependentPhaseServerToken>(&record))
                            {
                                auto [decoded, inserted] = decodedTokenTextCache.try_emplace(token->tokenId);
                                if (inserted)
                                {
                                    decoded->second = tokenizer.decode(std::vector<int32_t>{token->tokenId}, false);
                                    ++decodedTokenTextCacheMisses;
                                }
                                else
                                {
                                    ++decodedTokenTextCacheHits;
                                }
                                nlohmann::json const event{{"type", "token"}, {"request_index", token->requestId},
                                    {"token_id", token->tokenId}, {"text", decoded->second},
                                    {"output_index", token->outputIndex}, {"elapsed_ms", token->elapsedMs}};
                                readyLines.push_back("PHASE_EVENT\t" + event.dump());
                            }
                            else
                            {
                                auto& completion = std::get<rt::IndependentPhaseServerCompletion>(record);
                                nlohmann::json const event{{"type", "completion"},
                                    {"request_index", completion.requestId},
                                    {"finish_reason", completion.stoppedByEos ? "end-of-sequence" : "length"},
                                    {"prompt_tokens", completion.promptTokens},
                                    {"output_tokens", completion.generatedTokens.size()},
                                    {"latency_ms", completion.latencyMs}};
                                readyLines.push_back("PHASE_EVENT\t" + event.dump());
                            }
                        }
                        outputSerializationUs += std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - serializationStart)
                                                     .count();
                        size_t bytes{};
                        for (std::string const& line : readyLines)
                        {
                            bytes += line.size() + 1U;
                        }
                        std::string payload;
                        payload.reserve(bytes);
                        for (std::string& line : readyLines)
                        {
                            payload.append(line);
                            payload.push_back('\n');
                        }
                        std::cout.write(payload.data(), static_cast<std::streamsize>(payload.size()));
                        std::cout.flush();
                        ++outputWriteBatches;
                        outputWriteRecords += readyRecords.size();
                        outputWriteBytes += payload.size();
                    }
                    if (closed)
                    {
                        break;
                    }
                }
            });
            auto emitOutputRecords = [&](std::vector<PhaseOutputRecord> records) {
                if (records.empty())
                {
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(outputMutex);
                    for (PhaseOutputRecord& record : records)
                    {
                        outputRecords.push_back(std::move(record));
                    }
                }
                outputReady.notify_one();
            };
            auto emitSerializedRecords = [&](std::vector<std::string> records) {
                std::vector<PhaseOutputRecord> output;
                output.reserve(records.size());
                for (std::string& record : records)
                {
                    output.emplace_back(std::move(record));
                }
                emitOutputRecords(std::move(output));
            };
            auto emitRecord = [&](std::string const& prefix, nlohmann::json const& event) {
                emitSerializedRecords({prefix + event.dump()});
            };
            auto emitEvent = [&](nlohmann::json const& event) { emitRecord("PHASE_EVENT\t", event); };
            std::deque<rt::IndependentPhaseServerToken> nativeTokenEvents;
            std::deque<rt::IndependentPhaseServerCompletion> nativeCompletionEvents;
            bool const nativeEventCallbacks = ipcThreePhase == nullptr;
            if (nativeEventCallbacks)
            {
                semanticServer.setEventCallbacks(
                    [&](rt::IndependentPhaseServerToken&& event) { nativeTokenEvents.push_back(std::move(event)); },
                    [&](rt::IndependentPhaseServerCompletion&& event) {
                        nativeCompletionEvents.push_back(std::move(event));
                    });
            }
            auto popTokenEvent = [&]() -> std::optional<rt::IndependentPhaseServerToken> {
                if (!nativeEventCallbacks)
                {
                    return ipcThreePhase != nullptr ? ipcThreePhase->tryPopToken() : semanticServer.tryPopToken();
                }
                if (nativeTokenEvents.empty())
                {
                    return std::nullopt;
                }
                rt::IndependentPhaseServerToken event = std::move(nativeTokenEvents.front());
                nativeTokenEvents.pop_front();
                return event;
            };
            auto popCompletionEvent = [&]() -> std::optional<rt::IndependentPhaseServerCompletion> {
                if (!nativeEventCallbacks)
                {
                    return ipcThreePhase != nullptr ? ipcThreePhase->tryPopCompletion()
                                                    : semanticServer.tryPopCompletion();
                }
                if (nativeCompletionEvents.empty())
                {
                    return std::nullopt;
                }
                rt::IndependentPhaseServerCompletion event = std::move(nativeCompletionEvents.front());
                nativeCompletionEvents.pop_front();
                return event;
            };
            double ipcRequestAdapterUs{};
            size_t ipcRequestAdapterInputs{};
            std::thread inputReader([&]() {
                std::string line;
                while (std::getline(std::cin, line))
                {
                    if (!line.empty())
                    {
                        if (asyncRequestAdapter)
                        {
                            std::unique_lock<std::mutex> lock(pendingMutex);
                            inputAdapterReady.wait(
                                lock, [&]() { return pendingInputTasks.size() < requestAdapterWorkers; });
                            pendingInputTasks.push_back(std::async(std::launch::async,
                                [line = std::move(line),
                                    defaultMaxOutputTokens = serverConfig.defaultMaxOutputTokens]() {
                                    auto const adapterStart = std::chrono::steady_clock::now();
                                    PhaseIpcInput input = parsePhaseIpcInput(line, defaultMaxOutputTokens);
                                    input.adapterUs = std::chrono::duration<double, std::micro>(
                                        std::chrono::steady_clock::now() - adapterStart)
                                                          .count();
                                    return input;
                                }));
                        }
                        else
                        {
                            std::lock_guard<std::mutex> lock(pendingMutex);
                            pendingLines.push_back(std::move(line));
                        }
                    }
                }
                std::lock_guard<std::mutex> lock(pendingMutex);
                inputClosed = true;
            });
            emitEvent({{"type", "ready"}});
            size_t emittedMetrics = semanticCoordinator.metrics().size();
            double ipcIngressUs{};
            double ipcPollUs{};
            double ipcSerializationUs{};
            size_t ipcPollCalls{};
            while (true)
            {
                std::deque<std::string> lines;
                std::deque<PhaseIpcInput> inputs;
                {
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    lines.swap(pendingLines);
                    inputs.swap(pendingInputs);
                }
                while (true)
                {
                    std::future<PhaseIpcInput> task;
                    {
                        std::lock_guard<std::mutex> lock(pendingMutex);
                        if (pendingInputTasks.empty()
                            || pendingInputTasks.front().wait_for(std::chrono::seconds(0)) != std::future_status::ready)
                        {
                            break;
                        }
                        task = std::move(pendingInputTasks.front());
                        pendingInputTasks.pop_front();
                    }
                    inputAdapterReady.notify_one();
                    PhaseIpcInput input = task.get();
                    ipcRequestAdapterUs += input.adapterUs;
                    ++ipcRequestAdapterInputs;
                    inputs.push_back(std::move(input));
                }
                bool madeProgress = !lines.empty() || !inputs.empty();
                auto const ingressStart = std::chrono::steady_clock::now();
                size_t ingestedLines{};
                while ((!lines.empty() || !inputs.empty()) && ingestedLines < ipcIngressQuantum)
                {
                    PhaseIpcInput input = !inputs.empty()
                        ? std::move(inputs.front())
                        : parsePhaseIpcInput(lines.front(), serverConfig.defaultMaxOutputTokens);
                    if (!inputs.empty())
                    {
                        inputs.pop_front();
                    }
                    else
                    {
                        lines.pop_front();
                    }
                    uint64_t const requestId = input.requestId;
                    if (!input.valid)
                    {
                        emitEvent({{"type", "error"}, {"request_index", requestId}, {"message", input.error}});
                        ++ingestedLines;
                        continue;
                    }
                    if (input.cancel)
                    {
                        bool const cancelled = ipcThreePhase != nullptr ? ipcThreePhase->cancel(requestId)
                                                                        : semanticServer.cancel(requestId);
                        nlohmann::json const cancelEvent{
                            {"type", "cancelled"}, {"request_index", requestId}, {"cancelled", cancelled}};
                        emitEvent(cancelEvent);
                        ++ingestedLines;
                        continue;
                    }
                    rt::LLMGenerationRequest::Request request = std::move(input.request);
                    int32_t const maxOutputTokens = input.maxOutputTokens;
                    rt::PhaseSchedulingHints const scheduling = input.scheduling;
                    bool accepted{};
                    if (!request.imageBuffers.empty())
                    {
                        if (ipcThreePhase == nullptr)
                        {
                            emitEvent({{"type", "error"}, {"request_index", requestId},
                                {"message", "vision engine is not configured"}});
                            accepted = true;
                        }
                        else
                        {
                            rt::LLMGenerationRequest generation{};
                            generation.requests.push_back(std::move(request));
                            generation.temperature = 0.0F;
                            generation.topP = 1.0F;
                            generation.topK = 1;
                            generation.maxGenerateLength = maxOutputTokens;
                            generation.applyChatTemplate = true;
                            generation.addGenerationPrompt = true;
                            generation.disableSpecDecode = true;
                            rt::PhaseThreeSubmissionStatus const submission
                                = ipcThreePhase->submit(requestId, std::move(generation), maxOutputTokens, scheduling);
                            accepted = submission == rt::PhaseThreeSubmissionStatus::kEncoding
                                || submission == rt::PhaseThreeSubmissionStatus::kQueued;
                        }
                    }
                    else
                    {
                        rt::LLMGenerationRequest::FormattedRequest formatted;
                        ELLM_CHECK(tokenizer.applyChatTemplate(request, formatted, true, true, false),
                            "Failed to format IPC phase request");
                        std::vector<int32_t> const tokenIds
                            = tokenizer.encode(formatted.formattedCompleteRequest, false);
                        auto const submission
                            = semanticServer.submitOrQueue(requestId, tokenIds, maxOutputTokens, scheduling);
                        accepted = submission.status == rt::IndependentPhaseServerStatus::kAdmitted
                            || submission.status == rt::IndependentPhaseServerStatus::kQueued;
                    }
                    if (!accepted)
                    {
                        inputs.push_front(std::move(input));
                        break;
                    }
                    ++ingestedLines;
                }
                ipcIngressUs
                    += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ingressStart)
                           .count();
                size_t pendingAdapterRequests = lines.size() + inputs.size();
                {
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    pendingAdapterRequests += pendingLines.size() + pendingInputs.size() + pendingInputTasks.size();
                    // The reader may be between publishing bounded adapter
                    // cohorts while the long-lived ingress stream is open.
                    // Retain its worker cohort as near-term producer capacity
                    // across that handoff boundary.
                    if (!inputClosed)
                    {
                        pendingAdapterRequests = std::max(pendingAdapterRequests, requestAdapterWorkers);
                    }
                }
                semanticServer.setPendingAdapterRequests(pendingAdapterRequests);
                auto const pollStart = std::chrono::steady_clock::now();
                madeProgress
                    = (ipcThreePhase != nullptr ? ipcThreePhase->poll() : semanticServer.poll()) || madeProgress;
                ipcPollUs
                    += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - pollStart).count();
                ++ipcPollCalls;
                auto const serializationStart = std::chrono::steady_clock::now();
                std::vector<std::string> serializedRecords;
                if (!emitPhaseMetrics)
                {
                    emittedMetrics = semanticCoordinator.metrics().size();
                }
                while (emitPhaseMetrics && emittedMetrics < semanticCoordinator.metrics().size())
                {
                    madeProgress = true;
                    rt::PhaseDispatchMetrics const& metrics = semanticCoordinator.metrics()[emittedMetrics++];
                    auto const prefillGraphs = semanticCoordinator.prefillGraphCacheStats();
                    auto const decodeGraphs = semanticCoordinator.decodeGraphCacheStats();
                    rt::PhaseThreeCoordinatorMetrics const visionMetrics
                        = ipcThreePhase != nullptr ? ipcThreePhase->metrics() : rt::PhaseThreeCoordinatorMetrics{};
                    nlohmann::json const metricEvent{{"dispatch_index", metrics.dispatchIndex},
                        {"kind", static_cast<int32_t>(metrics.kind)}, {"prefill_batch", metrics.prefillBatchSize},
                        {"host_dispatch_start_us", static_cast<double>(metrics.hostDispatchStartNs) / 1000.0},
                        {"host_completion_us", static_cast<double>(metrics.hostCompletionNs) / 1000.0},
                        {"prefill_request_ids", metrics.prefillRequestIds},
                        {"decode_request_ids", metrics.decodeRequestIds},
                        {"prefill_class", static_cast<int32_t>(metrics.prefillClass)},
                        {"decode_batch", metrics.decodeBatchSize}, {"prefill_tokens", metrics.prefillTokens},
                        {"prefill_chunk_length", metrics.prefillCostLookupChunkLength},
                        {"prefill_initial_rows", metrics.prefillInitialRows},
                        {"prefill_continuation_rows", metrics.prefillContinuationRows},
                        {"prefill_past_kv_max", metrics.prefillPastKVMax}, {"decode_tokens", metrics.decodeTokens},
                        {"prefill_gpu_ms", metrics.prefillGpuMs},
                        {"decode_context_tokens", metrics.decodeContextTokens},
                        {"decode_context_max", metrics.plannedDecodeMaxContextLength},
                        {"decode_replacement_rows", metrics.predictedDecodeReplacementRows},
                        {"external_encoder_active", metrics.externalEncoderActive},
                        {"concurrent_prefill_active", metrics.concurrentPrefillActive},
                        {"predicted_decode_drain_gpu_ms", metrics.predictedDecodeDrainGpuMs},
                        {"predicted_decode_drain_turns", metrics.predictedDecodeDrainTurns},
                        {"decode_cohort_size", metrics.decodeCohortSize}, {"decode_gpu_ms", metrics.decodeGpuMs},
                        {"makespan_gpu_ms", metrics.makespanGpuMs}, {"overlap_ratio", metrics.overlapRatio},
                        {"memory_drain_preference", rt::phaseDrainPreferenceName(metrics.drainPreference)},
                        {"memory_drain_preference_applied", metrics.drainPreferenceApplied},
                        {"global_decision_evaluated", metrics.globalDecisionEvaluated},
                        {"global_decision_applied", metrics.globalDecisionApplied},
                        {"global_safe_probe", metrics.globalSafeProbe},
                        {"global_action", rt::phaseGlobalActionKindName(metrics.globalSelectedAction.kind)},
                        {"global_decision_reason", static_cast<int32_t>(metrics.globalDecisionReason)},
                        {"global_predicted_violation_us", metrics.globalPredictedViolationUs},
                        {"global_service_compression", metrics.globalServiceCompression},
                        {"global_reference_work_ms", metrics.globalReferenceWorkMs},
                        {"global_decisions", semanticCoordinator.scheduler().telemetry().globalDecisionCount},
                        {"global_active_decisions",
                            semanticCoordinator.scheduler().telemetry().globalActiveDecisionCount},
                        {"global_shadow_disagreements",
                            semanticCoordinator.scheduler().telemetry().globalShadowDisagreementCount},
                        {"global_wait_decisions", semanticCoordinator.scheduler().telemetry().globalWaitDecisionCount},
                        {"global_wait_selected", semanticCoordinator.scheduler().telemetry().globalWaitSelectedCount},
                        {"global_wait_candidates",
                            semanticCoordinator.scheduler().telemetry().globalWaitCandidateCount},
                        {"global_wait_future_rows", semanticCoordinator.scheduler().telemetry().globalWaitFutureRows},
                        {"global_wait_graph_bucket", semanticCoordinator.scheduler().telemetry().globalWaitGraphBucket},
                        {"global_wait_event_id", semanticCoordinator.scheduler().telemetry().globalWaitEventId},
                        {"global_wait_request_ids", semanticCoordinator.scheduler().telemetry().globalWaitRequestIds},
                        {"global_wait_preview_blocking_us",
                            semanticCoordinator.scheduler().telemetry().globalWaitPreviewBlockingUs},
                        {"global_wait_preview_uncertainty_us",
                            semanticCoordinator.scheduler().telemetry().globalWaitPreviewUncertaintyUs},
                        {"global_wait_preview_slack_us",
                            semanticCoordinator.scheduler().telemetry().globalWaitPreviewSlackUs},
                        {"global_wait_preview_compression",
                            semanticCoordinator.scheduler().telemetry().globalWaitPreviewCompression},
                        {"adaptive_throughput_mode", semanticServer.throughputMode()},
                        {"adaptive_transitions", semanticServer.throughputModeTransitionCount()},
                        {"adaptive_admission_limit", semanticServer.adaptiveAdmissionLimit()},
                        {"adaptive_admission_increases", semanticServer.adaptiveAdmissionIncreaseCount()},
                        {"adaptive_admission_decreases", semanticServer.adaptiveAdmissionDecreaseCount()},
                        {"adaptive_admission_cost_limit", semanticServer.adaptiveAdmissionCostLimit()},
                        {"adaptive_admission_cost_blocks", semanticServer.adaptiveAdmissionCostBlockCount()},
                        {"adaptive_admission_tpot_budget_us", semanticServer.adaptiveAdmissionTpotBudgetUs()},
                        {"adaptive_admission_tpot_satisfiable",
                            semanticServer.adaptiveAdmissionTpotBudgetSatisfiable()},
                        {"adaptive_admission_external_profile",
                            semanticServer.adaptiveAdmissionExternalProfileActive()},
                        {"adaptive_admission_external_profile_selections",
                            semanticServer.adaptiveAdmissionExternalProfileSelectionCount()},
                        {"adaptive_admission_unsatisfiable_decisions",
                            semanticServer.adaptiveAdmissionUnsatisfiableDecisionCount()},
                        {"decode_admission_tpot_pressure", semanticServer.decodeAdmissionTpotPressure()},
                        {"decode_refill_waits", semanticServer.decodeRefillWaitCount()},
                        {"prefill_formation_periods", semanticServer.prefillFormationWaitPeriodCount()},
                        {"prefill_formation_deferrals", semanticServer.prefillFormationDeferralCount()},
                        {"prefill_formation_profile_selections",
                            semanticServer.prefillFormationProfileSelectionCount()},
                        {"prefill_formation_profile_misses", semanticServer.prefillFormationProfileMissCount()},
                        {"page_growth_waits", semanticServer.pageGrowthWaitCount()},
                        {"page_growth_pending", semanticServer.pendingPageGrowthCount()},
                        {"page_growth_owners", semanticServer.pageGrowthOwnerCount()},
                        {"page_reservation_base", semanticServer.pageReservationBasePages()},
                        {"page_reservation_guaranteed", semanticServer.pageReservationGuaranteedPages()},
                        {"online_decode_cost_samples",
                            semanticCoordinator.scheduler().telemetry().onlineDecodeCostSampleCount},
                        {"online_decode_cost_buckets",
                            semanticCoordinator.scheduler().telemetry().onlineDecodeCostBucketCount},
                        {"encoder_contended_decode_cost_samples",
                            semanticCoordinator.scheduler().telemetry().encoderContendedDecodeCostSampleCount},
                        {"prefill_contended_decode_cost_samples",
                            semanticCoordinator.scheduler().telemetry().prefillContendedDecodeCostSampleCount},
                        {"prefill_graph_hits", prefillGraphs.hits}, {"prefill_graph_misses", prefillGraphs.misses},
                        {"decode_graph_hits", decodeGraphs.hits}, {"decode_graph_misses", decodeGraphs.misses},
                        {"sampling_event_slots", samplingSlotPool.size()},
                        {"sampling_event_reuses", samplingSlotPool.reuseCount()},
                        {"vision_pending", visionMetrics.pendingVisionRequests},
                        {"vision_prefill_ready", visionMetrics.pendingPrefillReadyRequests},
                        {"vision_prefill_ready_tokens", visionMetrics.pendingPrefillReadyTokens},
                        {"vision_admission_profile_prefill_tokens", visionMetrics.admissionProfilePrefillTokens},
                        {"vision_prefill_ready_bytes", visionMetrics.pendingPrefillReadyBytes},
                        {"vision_downstream", visionMetrics.downstreamEncodedRequests},
                        {"vision_downstream_bytes", visionMetrics.downstreamEncodedBytes},
                        {"vision_effective_encoded_capacity", visionMetrics.effectiveEncodedCapacity},
                        {"vision_effective_encoded_capacity_max", visionMetrics.maxEffectiveEncodedCapacity},
                        {"vision_encoded_capacity_escalations", visionMetrics.lookaheadEscalations},
                        {"vision_encoded_capacity_contractions", visionMetrics.encodedCapacityContractions},
                        {"vision_encoded_capacity_dwell_blocks", visionMetrics.encodedCapacityDwellBlocks},
                        {"vision_decode_tpot_pressure", visionMetrics.decodeTpotPressure},
                        {"vision_prefill_storage_releases", visionMetrics.prefillStorageReleases},
                        {"vision_prefill_storage_released_bytes", visionMetrics.prefillStorageReleasedBytes},
                        {"vision_encoder_starts", visionMetrics.encoderStarts},
                        {"vision_encoder_completions", visionMetrics.encoderCompletions},
                        {"vision_encoder_batches", visionMetrics.encoderBatches},
                        {"vision_encoder_batch", visionMetrics.lastEncoderBatchSize},
                        {"vision_encoder_batch_max", visionMetrics.maxEncoderBatchSize},
                        {"vision_encoder_input_bytes", visionMetrics.lastEncoderInputBytes},
                        {"vision_encoder_input_bytes_max", visionMetrics.maxEncoderInputBytes},
                        {"vision_encoder_input_tokens", visionMetrics.lastEncoderInputTokens},
                        {"vision_encoder_input_tokens_max", visionMetrics.maxEncoderInputTokens},
                        {"vision_oldest_pending_ms", visionMetrics.oldestPendingAgeUs / 1000.0},
                        {"vision_encoder_queue_wait_ms", visionMetrics.lastEncoderQueueWaitUs / 1000.0},
                        {"vision_encoder_queue_wait_max_ms", visionMetrics.maxEncoderQueueWaitUs / 1000.0},
                        {"vision_encoder_gpu_ms", visionMetrics.lastEncoderGpuMs},
                        {"vision_encoder_gpu_max_ms", visionMetrics.maxEncoderGpuMs},
                        {"vision_encoder_dispatch_deferrals", visionMetrics.encoderDispatchDeferrals},
                        {"vision_encoder_text_guard_deferrals", visionMetrics.encoderTextGuardDeferrals},
                        {"vision_encoder_prefill_guard_deferrals", visionMetrics.encoderPrefillGuardDeferrals},
                        {"vision_encoder_decode_guard_deferrals", visionMetrics.encoderDecodeGuardDeferrals},
                        {"vision_encoder_age_forced_starts", visionMetrics.encoderAgeForcedStarts},
                        {"vision_encoder_serialized_starts", visionMetrics.encoderSerializedStarts},
                        {"vision_encoder_serialization_bursts", visionMetrics.encoderSerializationBursts},
                        {"vision_encoder_serialization_boundary_waits",
                            visionMetrics.encoderSerializationBoundaryWaits},
                        {"vision_encoder_credit_wait_periods", visionMetrics.encoderCreditWaitPeriods},
                        {"vision_encoder_credit_age_releases", visionMetrics.encoderCreditAgeReleases},
                        {"vision_encoder_cost_aware_selections", visionMetrics.encoderCostAwareSelections},
                        {"vision_encoder_cost_coverage_misses", visionMetrics.encoderCostCoverageMisses},
                        {"vision_encoder_predicted_drain_gpu_ms", visionMetrics.lastPredictedEncoderDrainGpuMs},
                        {"vision_encoder_predicted_drain_turns", visionMetrics.lastPredictedEncoderDrainTurns},
                        {"vision_encoder_preparation_starts", visionMetrics.encoderPreparationStarts},
                        {"vision_encoder_preparation_completions", visionMetrics.encoderPreparationCompletions},
                        {"vision_encoder_preparation_ms", visionMetrics.lastEncoderPreparationUs / 1000.0},
                        {"vision_encoder_preparation_max_ms", visionMetrics.maxEncoderPreparationUs / 1000.0},
                        {"vision_encoder_exclusive_batches", visionMetrics.exclusiveEncoderBatches},
                        {"vision_encoder_exclusive_prefill_deferrals", visionMetrics.exclusiveEncoderPrefillDeferrals},
                        {"vision_global_decisions", visionMetrics.globalDecisions},
                        {"vision_global_shadow_disagreements", visionMetrics.globalShadowDisagreements},
                        {"vision_global_encoder_selections", visionMetrics.globalEncoderSelections},
                        {"vision_global_encoder_prefill_selections", visionMetrics.globalEncoderPrefillSelections},
                        {"vision_global_encoder_decode_selections", visionMetrics.globalEncoderDecodeSelections},
                        {"vision_global_pd_selections", visionMetrics.globalPdSelections},
                        {"vision_global_safe_probes", visionMetrics.globalSafeProbes},
                        {"vision_global_action", rt::phaseGlobalActionKindName(visionMetrics.lastGlobalAction)},
                        {"phase_memory_broker_decisions", visionMetrics.memoryBrokerDecisions},
                        {"phase_memory_encoder_reductions", visionMetrics.memoryBrokerEncoderReductions},
                        {"phase_memory_backpressure", visionMetrics.memoryBrokerBackpressure},
                        {"phase_memory_idle_reclaims", visionMetrics.memoryBrokerIdleReclaims},
                        {"phase_memory_prefill_preferences", visionMetrics.memoryBrokerPrefillPreferences},
                        {"phase_memory_decode_preferences", visionMetrics.memoryBrokerDecodePreferences},
                        {"phase_memory_predicted_bytes", visionMetrics.memoryBrokerLastPredictedBytes},
                        {"phase_memory_reason", rt::phaseMemoryBrokerReasonName(visionMetrics.memoryBrokerLastReason)},
                        {"phase_memory_drain_active",
                            rt::phaseDrainPreferenceName(visionMetrics.activeMemoryDrainPreference)},
                        {"phase_memory_drain_transitions", visionMetrics.memoryDrainPreferenceTransitions},
                        {"phase_memory_drain_applied_dispatches", visionMetrics.memoryDrainPreferenceAppliedDispatches},
                        {"vision_prefill_admission_batches", visionMetrics.prefillAdmissionBatches},
                        {"vision_prefill_admission_batch", visionMetrics.lastPrefillAdmissionBatchSize},
                        {"vision_prefill_admission_batch_max", visionMetrics.maxPrefillAdmissionBatchSize},
                        {"vision_prefill_adaptive_admissions", visionMetrics.adaptivePrefillAdmissions},
                        {"vision_prefill_low_load_admissions", visionMetrics.lowLoadPrefillAdmissions},
                        {"vision_prefill_backlog_admissions", visionMetrics.backlogPrefillAdmissions},
                        {"vision_prefill_decode_protected_admissions", visionMetrics.decodeProtectedPrefillAdmissions},
                        {"vision_prefill_decode_deferred_periods", visionMetrics.decodeDeferredPrefillPeriods},
                        {"vision_prefill_capacity_protected_admissions",
                            visionMetrics.capacityProtectedPrefillAdmissions},
                        {"vision_prefill_age_forced_admissions", visionMetrics.ageForcedPrefillAdmissions},
                        {"vision_prefill_byte_forced_admissions", visionMetrics.byteForcedPrefillAdmissions},
                        {"vision_prefill_available_slots", visionMetrics.availablePrefillAdmissionSlots},
                        {"vision_available_kv_pages", visionMetrics.availableKVPages},
                        {"vision_prefill_ready_wait_ms", visionMetrics.lastPrefillReadyQueueWaitUs / 1000.0},
                        {"vision_prefill_ready_wait_max_ms", visionMetrics.maxPrefillReadyQueueWaitUs / 1000.0}};
                    serializedRecords.push_back("PHASE_METRIC\t" + metricEvent.dump());
                }
                while (emitPhaseMetrics && !phaseTimelineEvents.empty())
                {
                    madeProgress = true;
                    rt::PhaseTimelineEvent const event = phaseTimelineEvents.front();
                    phaseTimelineEvents.pop_front();
                    nlohmann::json const timelineEvent{{"type", "timeline"}, {"request_index", event.requestId},
                        {"stage", std::string(rt::phaseTimelineStageName(event.stage))},
                        {"timestamp_us", static_cast<double>(event.timestampNs) / 1000.0},
                        {"dispatch_index", event.dispatchIndex}, {"batch_size", event.batchSize},
                        {"kv_slot_id", event.kvSlotId}};
                    serializedRecords.push_back("PHASE_TIMELINE\t" + timelineEvent.dump());
                }
                while (emitPhaseMetrics && !encoderBatchMetrics.empty())
                {
                    madeProgress = true;
                    rt::PhaseVisionEncoderBatchMetric const metric = encoderBatchMetrics.front();
                    encoderBatchMetrics.pop_front();
                    nlohmann::json const metricEvent{{"batch_index", metric.batchIndex},
                        {"batch_size", metric.batchSize}, {"input_bytes", metric.inputBytes},
                        {"input_tokens", metric.inputTokens}, {"gpu_ms", metric.gpuMs}};
                    serializedRecords.push_back("PHASE_ENCODER_METRIC\t" + metricEvent.dump());
                }
                std::vector<PhaseOutputRecord> phaseOutputRecords;
                while (auto token = popTokenEvent())
                {
                    madeProgress = true;
                    phaseOutputRecords.emplace_back(std::move(*token));
                }
                while (auto completion = popCompletionEvent())
                {
                    madeProgress = true;
                    phaseOutputRecords.emplace_back(std::move(*completion));
                }
                emitSerializedRecords(std::move(serializedRecords));
                emitOutputRecords(std::move(phaseOutputRecords));
                ipcSerializationUs
                    += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - serializationStart)
                           .count();
                bool closed{};
                {
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    closed = inputClosed;
                }
                bool noPendingInput{};
                {
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    noPendingInput = pendingLines.empty() && pendingInputs.empty() && pendingInputTasks.empty();
                }
                bool const serverEmpty = ipcThreePhase != nullptr ? ipcThreePhase->empty() : semanticServer.empty();
                if (closed && lines.empty() && inputs.empty() && noPendingInput && serverEmpty)
                {
                    break;
                }
                if (!lines.empty())
                {
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    while (!lines.empty())
                    {
                        pendingLines.push_front(std::move(lines.back()));
                        lines.pop_back();
                    }
                }
                if (!inputs.empty())
                {
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    while (!inputs.empty())
                    {
                        pendingInputs.push_front(std::move(inputs.back()));
                        inputs.pop_back();
                    }
                }
                if (!madeProgress)
                {
                    std::this_thread::yield();
                }
            }
            inputReader.join();
            LOG_INFO(
                "Phase formation: decode_refill_waits=%zu prefill_wait_periods=%zu prefill_deferrals=%zu "
                "prefill_profile_selections=%zu prefill_profile_misses=%zu "
                "adaptive_transitions=%zu throughput_mode=%s "
                "admission_limit=%zu admission_increases=%zu admission_decreases=%zu cost_limit=%zu "
                "cost_blocks=%zu tpot_budget_us=%.3f tpot_satisfiable=%s external_profile=%s "
                "external_profile_selections=%zu unsatisfiable_decisions=%zu",
                semanticServer.decodeRefillWaitCount(), semanticServer.prefillFormationWaitPeriodCount(),
                semanticServer.prefillFormationDeferralCount(), semanticServer.prefillFormationProfileSelectionCount(),
                semanticServer.prefillFormationProfileMissCount(), semanticServer.throughputModeTransitionCount(),
                semanticServer.throughputMode() ? "yes" : "no", semanticServer.adaptiveAdmissionLimit(),
                semanticServer.adaptiveAdmissionIncreaseCount(), semanticServer.adaptiveAdmissionDecreaseCount(),
                semanticServer.adaptiveAdmissionCostLimit(), semanticServer.adaptiveAdmissionCostBlockCount(),
                semanticServer.adaptiveAdmissionTpotBudgetUs(),
                semanticServer.adaptiveAdmissionTpotBudgetSatisfiable() ? "yes" : "no",
                semanticServer.adaptiveAdmissionExternalProfileActive() ? "yes" : "no",
                semanticServer.adaptiveAdmissionExternalProfileSelectionCount(),
                semanticServer.adaptiveAdmissionUnsatisfiableDecisionCount());
            LOG_INFO(
                "Sampling event pool: slots=%zu reuses=%zu", samplingSlotPool.size(), samplingSlotPool.reuseCount());
            LOG_INFO(
                "Phase token memory ops: h2d_operations=%zu h2d_bytes=%zu decode_device_reuse_batches=%zu "
                "decode_device_reuse_rows=%zu",
                tokenH2DOperations, tokenH2DBytes, decodeDeviceTokenReuseBatches, decodeDeviceTokenReuseRows);
            LOG_INFO(
                "Phase staging memory ops: segmented_vision_batches=%zu segmented_vision_sources=%zu "
                "segmented_vision_bytes=%zu vision_pack_d2d_operations=0 vision_pack_d2d_bytes=0 "
                "mrope_d2d_operations=%zu mrope_d2d_bytes=%zu mrope_full_row_d2d_bytes=%zu "
                "mrope_copy_granularity=%d mrope_timed_batches=%zu mrope_gpu_ms=%.3f",
                segmentedVisionBatches, segmentedVisionSources, segmentedVisionBytes, mropeD2DOperations, mropeD2DBytes,
                mropeFullRowD2DBytes, mropeCopyGranularity, mropeCopyTiming.size(), mropeCopyTiming.milliseconds());
            auto logKVMemoryOps = [](char const* phase, rt::PhaseKVMemoryStats const& memory,
                                      rt::KVPageTableUploadStats const& pageTable) {
                LOG_INFO(
                    "Phase %s KV memory ops: prepares=%zu length_h2d_ops=%zu length_h2d_bytes=%zu "
                    "metadata_h2d_ops=%zu metadata_h2d_bytes=%zu memset_ops=%zu memset_bytes=%zu "
                    "decode_select_zero_reuses=%zu page_binding_updates=%zu page_binding_reuses=%zu "
                    "page_calls=%zu page_uploads=%zu page_copy_ops=%zu page_copy_bytes=%zu page_host_waits=%zu "
                    "page_stream_waits=%zu",
                    phase, memory.prepareCalls, memory.lengthH2DOperations, memory.lengthH2DBytes,
                    memory.prefillMetadataH2DOperations + memory.decodeMetadataH2DOperations,
                    memory.prefillMetadataH2DBytes + memory.decodeMetadataH2DBytes, memory.decodeMemsetOperations,
                    memory.decodeMemsetBytes, memory.decodeSelectZeroReuses, memory.pageBindingRowUpdates,
                    memory.pageBindingRowReuses, pageTable.calls, pageTable.uploads, pageTable.copyOperations,
                    pageTable.copyBytes, pageTable.hostWaits, pageTable.streamWaits);
            };
            logKVMemoryOps("prefill", semanticCoordinator.prefillKVMemoryStats(),
                semanticCoordinator.prefillPageTableUploadStats());
            logKVMemoryOps(
                "decode", semanticCoordinator.decodeKVMemoryStats(), semanticCoordinator.decodePageTableUploadStats());
            LOG_INFO("Phase IPC policy: ingress_quantum=%zu emit_metrics=%s", ipcIngressQuantum,
                emitPhaseMetrics ? "yes" : "no");
            LOG_INFO("Phase IPC response path: %s", nativeEventCallbacks ? "native_callback" : "polling_queue");
            LOG_INFO(
                "Phase page reservation: mode=%s base=%d guaranteed=%d growth_owners=%zu growth_pending=%zu "
                "growth_waits=%zu",
                serverConfig.pageReservationMode == rt::IndependentPhasePageReservationMode::kFull ? "full"
                                                                                                   : "headroom",
                semanticServer.pageReservationBasePages(), semanticServer.pageReservationGuaranteedPages(),
                semanticServer.pageGrowthOwnerCount(), semanticServer.pendingPageGrowthCount(),
                semanticServer.pageGrowthWaitCount());
            if (ipcThreePhase != nullptr)
            {
                rt::PhaseThreeCoordinatorMetrics const visionMetrics = ipcThreePhase->metrics();
                LOG_INFO(
                    "Phase vision cost: starts=%zu completions=%zu batches=%zu batch_last=%zu batch_max=%zu "
                    "encoder_input_bytes_last=%zu encoder_input_bytes_max=%zu "
                    "encoder_input_tokens_last=%zu encoder_input_tokens_max=%zu "
                    "pending=%zu prefill_ready=%zu prefill_ready_bytes=%zu downstream=%zu bytes=%zu "
                    "prefill_admission_batches=%zu prefill_admission_last=%zu prefill_admission_max=%zu "
                    "prefill_adaptive=%zu prefill_low_load=%zu prefill_backlog=%zu prefill_decode_protected=%zu "
                    "prefill_decode_deferred=%zu prefill_capacity_protected=%zu prefill_age_forced=%zu "
                    "prefill_byte_forced=%zu "
                    "available_prefill_slots=%zu "
                    "available_kv_pages=%d "
                    "prefill_releases=%zu prefill_released_bytes=%zu "
                    "queue_wait_last=%.3f ms queue_wait_max=%.3f ms encoder_gpu_last=%.3f ms encoder_gpu_max=%.3f ms "
                    "prefill_ready_wait_last=%.3f ms prefill_ready_wait_max=%.3f ms "
                    "encoded_capacity=%zu encoded_capacity_max=%zu lookahead_escalations=%zu "
                    "capacity_contractions=%zu capacity_dwell_blocks=%zu "
                    "decode_tpot_pressure=%.3f async_preparations=%zu/%zu preparation_last=%.3f ms "
                    "preparation_max=%.3f ms exclusive_batches=%zu exclusive_prefill_deferrals=%zu",
                    visionMetrics.encoderStarts, visionMetrics.encoderCompletions, visionMetrics.encoderBatches,
                    visionMetrics.lastEncoderBatchSize, visionMetrics.maxEncoderBatchSize,
                    visionMetrics.lastEncoderInputBytes, visionMetrics.maxEncoderInputBytes,
                    visionMetrics.lastEncoderInputTokens, visionMetrics.maxEncoderInputTokens,
                    visionMetrics.pendingVisionRequests, visionMetrics.pendingPrefillReadyRequests,
                    visionMetrics.pendingPrefillReadyBytes, visionMetrics.downstreamEncodedRequests,
                    visionMetrics.downstreamEncodedBytes, visionMetrics.prefillAdmissionBatches,
                    visionMetrics.lastPrefillAdmissionBatchSize, visionMetrics.maxPrefillAdmissionBatchSize,
                    visionMetrics.adaptivePrefillAdmissions, visionMetrics.lowLoadPrefillAdmissions,
                    visionMetrics.backlogPrefillAdmissions, visionMetrics.decodeProtectedPrefillAdmissions,
                    visionMetrics.decodeDeferredPrefillPeriods, visionMetrics.capacityProtectedPrefillAdmissions,
                    visionMetrics.ageForcedPrefillAdmissions, visionMetrics.byteForcedPrefillAdmissions,
                    visionMetrics.availablePrefillAdmissionSlots, visionMetrics.availableKVPages,
                    visionMetrics.prefillStorageReleases, visionMetrics.prefillStorageReleasedBytes,
                    visionMetrics.lastEncoderQueueWaitUs / 1000.0, visionMetrics.maxEncoderQueueWaitUs / 1000.0,
                    visionMetrics.lastEncoderGpuMs, visionMetrics.maxEncoderGpuMs,
                    visionMetrics.lastPrefillReadyQueueWaitUs / 1000.0,
                    visionMetrics.maxPrefillReadyQueueWaitUs / 1000.0, visionMetrics.effectiveEncodedCapacity,
                    visionMetrics.maxEffectiveEncodedCapacity, visionMetrics.lookaheadEscalations,
                    visionMetrics.encodedCapacityContractions, visionMetrics.encodedCapacityDwellBlocks,
                    visionMetrics.decodeTpotPressure, visionMetrics.encoderPreparationStarts,
                    visionMetrics.encoderPreparationCompletions, visionMetrics.lastEncoderPreparationUs / 1000.0,
                    visionMetrics.maxEncoderPreparationUs / 1000.0, visionMetrics.exclusiveEncoderBatches,
                    visionMetrics.exclusiveEncoderPrefillDeferrals);
                rt::PhaseVisionMemoryStats const& memoryStats = ipcVisionAdapter->memoryStats();
                LOG_INFO(
                    "Phase vision memory ops: slab_allocations=%zu slab_reuses=%zu slab_reclaims=%zu "
                    "reclaimed_bytes=%zu direct_output_batches=%zu direct_output_bytes=%zu "
                    "d2d_operations=%zu d2d_bytes=%zu idle_slabs=%zu idle_bytes=%zu",
                    memoryStats.batchStorageAllocations, memoryStats.batchStorageReuses,
                    memoryStats.batchStorageReclaims, memoryStats.reclaimedBytes, memoryStats.directOutputBatches,
                    memoryStats.directOutputBytes, memoryStats.deviceCopyOperations, memoryStats.deviceCopyBytes,
                    memoryStats.idleStorageBatches, memoryStats.idleStorageBytes);
            }
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                outputClosed = true;
            }
            outputReady.notify_one();
            outputWriter.join();
            LOG_INFO("Phase IPC token text cache: entries=%zu hits=%zu misses=%zu", decodedTokenTextCache.size(),
                decodedTokenTextCacheHits, decodedTokenTextCacheMisses);
            LOG_INFO(
                "Phase IPC host cost: polls=%zu ingress=%.3f ms adapter_workers=%zu adapter_inputs=%zu "
                "adapter=%.3f ms "
                "poll=%.3f ms serialize=%.3f ms output_serialize=%.3f ms output_batches=%zu output_records=%zu "
                "output_bytes=%zu",
                ipcPollCalls, ipcIngressUs / 1000.0, asyncRequestAdapter ? requestAdapterWorkers : 0U,
                ipcRequestAdapterInputs, ipcRequestAdapterUs / 1000.0, ipcPollUs / 1000.0, ipcSerializationUs / 1000.0,
                outputSerializationUs / 1000.0, outputWriteBatches, outputWriteRecords, outputWriteBytes);
            auto const prefillGraphStats = semanticCoordinator.prefillGraphCacheStats();
            auto const decodeGraphStats = semanticCoordinator.decodeGraphCacheStats();
            LOG_INFO(
                "Phase CUDA graph cache: prefill entries=%zu hits=%zu misses=%zu captures=%zu evictions=%zu; "
                "decode entries=%zu hits=%zu misses=%zu captures=%zu evictions=%zu",
                prefillGraphStats.entries, prefillGraphStats.hits, prefillGraphStats.misses, prefillGraphStats.captures,
                prefillGraphStats.evictions, decodeGraphStats.entries, decodeGraphStats.hits, decodeGraphStats.misses,
                decodeGraphStats.captures, decodeGraphStats.evictions);
            ipcThreePhase.reset();
            ipcVisionAdapter.reset();
            ipcVisionRunner.reset();
            if (ipcEncoderStream != nullptr)
            {
                CUDA_CHECK(cudaStreamDestroy(ipcEncoderStream));
            }
        }
        else
        {
            for (size_t index = 0; index < prompts.size(); ++index)
            {
                uint64_t const requestId = 20000 + index;
                auto const submission = semanticServer.submit(requestId, semanticPrompts.at(requestId));
                ELLM_CHECK(submission.status == rt::IndependentPhaseServerStatus::kAdmitted,
                    "Semantic phase request admission failed");
            }
            semanticServer.runUntilIdle(100000);

            std::unordered_map<uint64_t, std::string> semanticTexts;
            while (auto completion = semanticServer.tryPopCompletion())
            {
                semanticTexts[completion->requestId] = tokenizer.decode(completion->generatedTokens, false);
            }
            ELLM_CHECK(semanticTexts.size() == prompts.size(), "Semantic phase completion count mismatch");
            LOG_INFO("Semantic phase outputs: output0='%s' output1='%s' output2='%s'", semanticTexts.at(20000).c_str(),
                semanticTexts.at(20001).c_str(), semanticTexts.at(20002).c_str());
            ELLM_CHECK(semanticTexts.at(20000).find("asynchronous") != std::string::npos
                    && semanticTexts.at(20001).find("Dynamic batching") != std::string::npos
                    && semanticTexts.at(20002).find("Kernel") != std::string::npos,
                "Semantic phase outputs do not match the expected Cosmos responses");
            ELLM_CHECK(semanticServer.empty() && ownership.availableSlots() == maxStableSlots,
                "Semantic phase requests did not drain and release every slot");
            LOG_INFO("Semantic phase requests passed through IndependentPhaseAsyncServer");
        }
    }

    CUDA_CHECK(cudaStreamDestroy(setupStream));
    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
    return EXIT_SUCCESS;
}
