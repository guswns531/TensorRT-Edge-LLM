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
#include "runtime/scheduling/phaseActivityTimeline.h"
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
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
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

enum class PhaseIpcKind
{
    kSubmit,
    kCancel,
    kCalibrationBegin,
    kCalibrationStatus,
    kCalibrationEnd,
};

enum class PhasePolicyWarmupMode
{
    kGraphOnly,
    kGeneric,
    kTraceDerived,
    kZeroStart,
};

PhasePolicyWarmupMode phasePolicyWarmupMode()
{
    std::string const value = std::getenv("TRT_EDGELLM_POLICY_WARMUP_MODE") != nullptr
        ? std::getenv("TRT_EDGELLM_POLICY_WARMUP_MODE")
        : "trace_derived";
    ELLM_CHECK(value == "graph_only" || value == "generic" || value == "trace_derived" || value == "zero_start",
        "TRT_EDGELLM_POLICY_WARMUP_MODE must be graph_only, generic, trace_derived, or zero_start");
    if (value == "graph_only")
    {
        return PhasePolicyWarmupMode::kGraphOnly;
    }
    if (value == "generic")
    {
        return PhasePolicyWarmupMode::kGeneric;
    }
    if (value == "zero_start")
    {
        return PhasePolicyWarmupMode::kZeroStart;
    }
    return PhasePolicyWarmupMode::kTraceDerived;
}

char const* phasePolicyWarmupModeName(PhasePolicyWarmupMode mode) noexcept
{
    switch (mode)
    {
    case PhasePolicyWarmupMode::kGraphOnly: return "graph_only";
    case PhasePolicyWarmupMode::kGeneric: return "generic";
    case PhasePolicyWarmupMode::kTraceDerived: return "trace_derived";
    case PhasePolicyWarmupMode::kZeroStart: return "zero_start";
    }
    return "unknown";
}

struct PhaseIpcInput
{
    uint64_t ingressSequence{};
    uint64_t requestId{};
    PhaseIpcKind kind{PhaseIpcKind::kSubmit};
    bool valid{true};
    std::string error;
    rt::LLMGenerationRequest::Request request;
    int32_t maxOutputTokens{};
    rt::PhaseSchedulingHints scheduling;
    double adapterUs{};
};

enum class PhaseIpcOrderingClass : uint8_t
{
    kText,
    kVision,
    kBarrier,
};

struct PendingPhaseIpcInput
{
    PhaseIpcOrderingClass orderingClass{PhaseIpcOrderingClass::kText};
    std::future<PhaseIpcInput> task;
};

PhaseIpcOrderingClass phaseIpcOrderingClass(std::string const& line) noexcept
{
    if (line.find("\"type\":\"calibration_") != std::string::npos
        || line.find("\"type\": \"calibration_") != std::string::npos
        || line.find("\"type\":\"cancel\"") != std::string::npos
        || line.find("\"type\": \"cancel\"") != std::string::npos)
    {
        return PhaseIpcOrderingClass::kBarrier;
    }
    if (line.find("\"image_url\"") != std::string::npos)
    {
        return PhaseIpcOrderingClass::kVision;
    }
    return PhaseIpcOrderingClass::kText;
}

PhaseIpcInput parsePhaseIpcInput(std::string const& line, int32_t defaultMaxOutputTokens)
{
    PhaseIpcInput result;
    result.maxOutputTokens = defaultMaxOutputTokens;
    try
    {
        nlohmann::json const payload = nlohmann::json::parse(line);
        result.requestId = payload.value("request_index", uint64_t{});
        std::string const type = payload.value("type", "submit");
        if (type == "cancel")
        {
            result.kind = PhaseIpcKind::kCancel;
        }
        else if (type == "calibration_begin")
        {
            result.kind = PhaseIpcKind::kCalibrationBegin;
        }
        else if (type == "calibration_end")
        {
            result.kind = PhaseIpcKind::kCalibrationEnd;
        }
        else if (type == "calibration_status")
        {
            result.kind = PhaseIpcKind::kCalibrationStatus;
        }
        if (result.kind != PhaseIpcKind::kSubmit)
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
    float overlapMs{};
    float overlapPairFraction{};
};

struct ExecutorTiming
{
    float meanMs{};
    float medianMs{};
    float p95Ms{};
    float maxMs{};
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

PhaseTiming measureControlledOverlap(rt::EngineExecutor& prefillExecutor, rt::EngineExecutor& decodeExecutor,
    cudaStream_t setupStream, cudaStream_t prefillStream, cudaStream_t decodeStream, int32_t warmup, int32_t iterations,
    int32_t overlapPercent)
{
    ELLM_CHECK(overlapPercent >= 0 && overlapPercent <= 100, "Controlled overlap percentage is outside [0, 100]");
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
    int32_t overlapAccumulator{};
    int32_t measuredOverlapPairs{};
    int32_t const totalIterations = warmup + iterations;
    for (int32_t iteration = 0; iteration < totalIterations; ++iteration)
    {
        overlapAccumulator += overlapPercent;
        bool const concurrent = overlapAccumulator >= 100;
        overlapAccumulator -= concurrent ? 100 : 0;
        if (concurrent)
        {
            CUDA_CHECK(cudaEventRecord(gate, setupStream));
            CUDA_CHECK(cudaStreamWaitEvent(prefillStream, gate));
            CUDA_CHECK(cudaStreamWaitEvent(decodeStream, gate));
            CUDA_CHECK(cudaEventRecord(prefillStart, prefillStream));
            ELLM_CHECK(prefillExecutor.execute(prefillStream), "Controlled overlap prefill execution failed");
            CUDA_CHECK(cudaEventRecord(prefillEnd, prefillStream));
            CUDA_CHECK(cudaEventRecord(decodeStart, decodeStream));
            ELLM_CHECK(decodeExecutor.execute(decodeStream), "Controlled overlap decode execution failed");
            CUDA_CHECK(cudaEventRecord(decodeEnd, decodeStream));
            CUDA_CHECK(cudaStreamWaitEvent(setupStream, prefillEnd));
            CUDA_CHECK(cudaStreamWaitEvent(setupStream, decodeEnd));
            CUDA_CHECK(cudaEventRecord(done, setupStream));
            CUDA_CHECK(cudaEventSynchronize(done));
        }
        else
        {
            CUDA_CHECK(cudaEventRecord(prefillStart, prefillStream));
            ELLM_CHECK(prefillExecutor.execute(prefillStream), "Controlled sequential prefill execution failed");
            CUDA_CHECK(cudaEventRecord(prefillEnd, prefillStream));
            CUDA_CHECK(cudaEventSynchronize(prefillEnd));
            CUDA_CHECK(cudaEventRecord(decodeStart, decodeStream));
            ELLM_CHECK(decodeExecutor.execute(decodeStream), "Controlled sequential decode execution failed");
            CUDA_CHECK(cudaEventRecord(decodeEnd, decodeStream));
            CUDA_CHECK(cudaEventSynchronize(decodeEnd));
        }
        if (iteration < warmup)
        {
            continue;
        }

        float prefillMs{};
        float decodeMs{};
        CUDA_CHECK(cudaEventElapsedTime(&prefillMs, prefillStart, prefillEnd));
        CUDA_CHECK(cudaEventElapsedTime(&decodeMs, decodeStart, decodeEnd));
        total.prefillMs += prefillMs;
        total.decodeMs += decodeMs;
        if (concurrent)
        {
            float makespanMs{};
            float prefillStartMs{};
            float prefillEndMs{};
            float decodeStartMs{};
            float decodeEndMs{};
            CUDA_CHECK(cudaEventElapsedTime(&makespanMs, gate, done));
            CUDA_CHECK(cudaEventElapsedTime(&prefillStartMs, gate, prefillStart));
            CUDA_CHECK(cudaEventElapsedTime(&prefillEndMs, gate, prefillEnd));
            CUDA_CHECK(cudaEventElapsedTime(&decodeStartMs, gate, decodeStart));
            CUDA_CHECK(cudaEventElapsedTime(&decodeEndMs, gate, decodeEnd));
            total.makespanMs += makespanMs;
            total.overlapMs
                += std::max(0.0F, std::min(prefillEndMs, decodeEndMs) - std::max(prefillStartMs, decodeStartMs));
            ++measuredOverlapPairs;
        }
        else
        {
            total.makespanMs += prefillMs + decodeMs;
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
    total.overlapMs *= scale;
    total.overlapPairFraction = static_cast<float>(measuredOverlapPairs) * scale;
    return total;
}

ExecutorTiming measureExecutor(rt::EngineExecutor& executor, cudaStream_t stream, int32_t warmup, int32_t iterations)
{
    ELLM_CHECK(warmup >= 0 && iterations > 0, "Executor timing requires non-negative warmup and positive iterations");
    cudaEvent_t start{};
    cudaEvent_t end{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    std::vector<float> samples;
    samples.reserve(static_cast<size_t>(iterations));
    int32_t const totalIterations = warmup + iterations;
    for (int32_t iteration = 0; iteration < totalIterations; ++iteration)
    {
        CUDA_CHECK(cudaEventRecord(start, stream));
        ELLM_CHECK(executor.execute(stream), "Isolated engine execution failed");
        CUDA_CHECK(cudaEventRecord(end, stream));
        CUDA_CHECK(cudaEventSynchronize(end));
        if (iteration >= warmup)
        {
            float elapsedMs{};
            CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, end));
            samples.push_back(elapsedMs);
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    std::sort(samples.begin(), samples.end());
    float const sum = std::accumulate(samples.begin(), samples.end(), 0.0F);
    size_t const medianIndex = samples.size() / 2U;
    size_t const p95Index = std::min(
        samples.size() - 1U, static_cast<size_t>(std::ceil(static_cast<double>(samples.size()) * 0.95)) - 1U);
    return ExecutorTiming{
        sum / static_cast<float>(samples.size()), samples[medianIndex], samples[p95Index], samples.back()};
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
    cudaStream_t copyStream{};
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
        CUDA_CHECK(cudaStreamCreateWithPriority(&copyStream, cudaStreamNonBlocking, normalPriority));
        LOG_INFO("Phase stream priorities: encoder=%d prefill=%d decode=%d", leastPriority, normalPriority,
            greatestPriority);
    }
    else
    {
        CUDA_CHECK(cudaStreamCreateWithFlags(&setupStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
    }

    {
        auto executor = rt::EngineExecutor::createForLLM(engineDir / "llm.engine", config);
        rt::IndependentEngineExecutorPairConfig pairConfig;
        pairConfig.visionPrefillProfile = config.visionPrefillProfile;
        pairConfig.dedicatedExternalPrefillContext
            = std::getenv("TRT_EDGELLM_DEDICATED_EXTERNAL_PREFILL_CONTEXT") != nullptr;
        pairConfig.setupStream = setupStream;
        pairConfig.prefillStream = prefillStream;
        pairConfig.decodeStream = decodeStream;
        pairConfig.sharedExecutionContext = std::getenv("TRT_EDGELLM_SHARED_PHASE_CONTEXT") != nullptr;
        auto pair = rt::IndependentEngineExecutorPair::create(std::move(executor), pairConfig);

        ELLM_CHECK(&pair->prefillExecutor().getEngine() == &pair->decodeExecutor().getEngine(),
            "Phase executors must share one TensorRT engine");
        ELLM_CHECK(&pair->prefillExecutor().getEngine() == &pair->externalPrefillExecutor().getEngine(),
            "External-prefill executor must share the TensorRT engine");
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
            if (pair->hasExternalPrefillExecutor())
            {
                ELLM_CHECK(pair->prefillExecutor().getExecutionContextIdentity()
                        != pair->externalPrefillExecutor().getExecutionContextIdentity(),
                    "Text and external prefill must own different TensorRT contexts");
            }
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
            bool const controlledOverlapSweep = std::getenv("TRT_EDGELLM_PHASE_OVERLAP_SWEEP") != nullptr;
            int32_t controlledDecodeBatchSize = 1;
            if (controlledOverlapSweep)
            {
                if (char const* value = std::getenv("TRT_EDGELLM_PHASE_OVERLAP_SWEEP_DECODE_BATCH"))
                {
                    controlledDecodeBatchSize = std::stoi(value);
                }
            }
            ELLM_CHECK(controlledDecodeBatchSize > 0 && controlledDecodeBatchSize <= config.maxSupportedDecodeBatchSize,
                "Controlled overlap decode batch is outside the engine profile");
            int32_t const prefillSlot0 = ownership.reserve();
            int32_t const prefillSlot1 = config.packedPrefill ? ownership.reserve() : -1;
            std::vector<int32_t> decodeSlots;
            decodeSlots.reserve(static_cast<size_t>(controlledDecodeBatchSize));
            for (int32_t row{}; row < controlledDecodeBatchSize; ++row)
            {
                decodeSlots.push_back(ownership.reserve());
            }
            int32_t const decodeSlot = decodeSlots.front();
            ownership.ensureCapacity(prefillSlot0, 128);
            if (config.packedPrefill)
            {
                ownership.ensureCapacity(prefillSlot1, 128);
            }
            ownership.setLength(prefillSlot0, 0);
            if (config.packedPrefill)
            {
                ownership.setLength(prefillSlot1, 0);
            }
            for (int32_t const slot : decodeSlots)
            {
                ownership.ensureCapacity(slot, 129);
                ownership.setLength(slot, 128);
            }
            rt::PhaseKVActiveView prefillKV(config.maxSupportedPrefillBatchSize, ownership, prefillMap, "prefill");
            rt::PhaseKVActiveView decodeKV(config.maxSupportedDecodeBatchSize, ownership, decodeMap, "decode");
            std::vector<int32_t> const prefillSlots = config.packedPrefill
                ? std::vector<int32_t>{prefillSlot0, prefillSlot1}
                : std::vector<int32_t>{prefillSlot0};
            std::vector<int32_t> const prefillChunkLengths
                = config.packedPrefill ? std::vector<int32_t>{96, 32} : std::vector<int32_t>{128};
            prefillKV.prepare(prefillSlots, prefillStream);
            decodeKV.prepare(decodeSlots, decodeStream);

            int32_t const prefillTotalTokens = 128;
            ELLM_CHECK(prefillIO->inputsEmbeds.reshape({1, prefillTotalTokens, config.hiddenSize}),
                "Failed to reshape prefill input embeddings");
            ELLM_CHECK(decodeIO->inputsEmbeds.reshape({controlledDecodeBatchSize, 1, config.hiddenSize}),
                "Failed to reshape decode input embeddings");
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
                ? config.packedPrefillDims(
                      static_cast<int64_t>(prefillSlots.size()), prefillTotalTokens, /*maxRowTokens=*/96)
                : config.prefillDims(1, prefillTotalTokens, false);
            ELLM_CHECK(pair->prefillExecutor().prepare(0, prefillDims, prefillMap, prefillStream),
                "Failed to bind the stable paged-KV prefill view");
            ELLM_CHECK(pair->decodeExecutor().prepare(
                           1, config.decodeDims(controlledDecodeBatchSize), decodeMap, decodeStream),
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
            decodeKV.commitLengths(std::vector<int32_t>(static_cast<size_t>(controlledDecodeBatchSize), 129));
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

            if (controlledOverlapSweep)
            {
                std::array<int32_t, 5> constexpr kOVERLAP_PERCENTAGES{0, 25, 50, 75, 100};
                float controlledSequentialMakespanMs{};
                for (int32_t const requestedPercent : kOVERLAP_PERCENTAGES)
                {
                    PhaseTiming const controlled
                        = measureControlledOverlap(pair->prefillExecutor(), pair->decodeExecutor(), setupStream,
                            prefillStream, decodeStream, kWARMUP, kITERATIONS, requestedPercent);
                    if (requestedPercent == 0)
                    {
                        controlledSequentialMakespanMs = controlled.makespanMs;
                    }
                    float const controlledSpeedup = controlledSequentialMakespanMs / controlled.makespanMs;
                    float const activeWallOverlap
                        = controlled.overlapMs / std::max(controlled.makespanMs, std::numeric_limits<float>::epsilon());
                    LOG_INFO(
                        "Controlled phase overlap: decode_batch=%d requested_pairs=%d%% observed_pairs=%.1f%% "
                        "active_wall_overlap=%.3f%% makespan=%.4f ms speedup=%.3fx prefill=%.4f ms "
                        "decode=%.4f ms overlap=%.4f ms",
                        controlledDecodeBatchSize, requestedPercent, controlled.overlapPairFraction * 100.0F,
                        activeWallOverlap * 100.0F, controlled.makespanMs, controlledSpeedup, controlled.prefillMs,
                        controlled.decodeMs, controlled.overlapMs);
                }
                LOG_INFO("Controlled phase overlap sweep completed; skipping unrelated semantic smoke stages");
                CUDA_CHECK(cudaStreamDestroy(setupStream));
                CUDA_CHECK(cudaStreamDestroy(prefillStream));
                CUDA_CHECK(cudaStreamDestroy(decodeStream));
                CUDA_CHECK(cudaStreamDestroy(copyStream));
                return EXIT_SUCCESS;
            }

            // Activate the optional external-prefill context before the server
            // advertises readiness. Text and external prefill share one arena
            // but retain fixed profiles on distinct serialized contexts.
            if (pair->hasExternalPrefillExecutor())
            {
                int32_t const visionWarmupTokens = config.hasVisionPrefillProfile()
                    ? config.maxVisionPackedPrefillChunkTokens
                    : config.maxPackedPrefillChunkTokens;
                ownership.setLength(prefillSlot0, 0);
                ownership.ensureCapacity(prefillSlot0, visionWarmupTokens);
                prefillKV.prepare({prefillSlot0}, prefillStream);
                ELLM_CHECK(prefillIO->inputsEmbeds.reshape({1, visionWarmupTokens, config.hiddenSize}),
                    "Failed to reshape external-prefill warmup embeddings");
                CUDA_CHECK(cudaMemsetAsync(prefillIO->inputsEmbeds.rawPointer(), 0,
                    prefillIO->inputsEmbeds.getMemoryCapacity(), prefillStream));
                for (rt::Tensor& deepstack : prefillIO->deepstackEmbeds)
                {
                    CUDA_CHECK(
                        cudaMemsetAsync(deepstack.rawPointer(), 0, deepstack.getMemoryCapacity(), prefillStream));
                }
                prefillKV.preparePrefillMetadata(*prefillIO, {visionWarmupTokens}, prefillStream, true);
                rt::InferenceDims const externalDims = config.hasVisionPrefillProfile()
                    ? config.visionPackedPrefillDims(1, visionWarmupTokens, visionWarmupTokens)
                    : config.packedPrefillDims(1, visionWarmupTokens, visionWarmupTokens);
                ELLM_CHECK(pair->externalPrefillExecutor().prepare(
                               pair->externalPrefillProfile(), externalDims, prefillMap, prefillStream),
                    "Failed to prepare the external-prefill profile warmup");
                ELLM_CHECK(pair->externalPrefillExecutor().execute(prefillStream),
                    "Failed to execute the external-prefill profile warmup");
                CUDA_CHECK(cudaStreamSynchronize(prefillStream));
                prefillKV.complete();
                LOG_INFO("Primed external-prefill profile %d with %d tokens", pair->externalPrefillProfile(),
                    visionWarmupTokens);

                if (char const* value = std::getenv("TRT_EDGELLM_EXTERNAL_PREFILL_BENCH_TOKENS"))
                {
                    constexpr int32_t kDEFAULT_BENCH_WARMUP = 20;
                    constexpr int32_t kDEFAULT_BENCH_ITERATIONS = 100;
                    int32_t const benchTokens = std::stoi(value);
                    int32_t const benchWarmup = std::getenv("TRT_EDGELLM_EXTERNAL_PREFILL_BENCH_WARMUP") != nullptr
                        ? std::stoi(std::getenv("TRT_EDGELLM_EXTERNAL_PREFILL_BENCH_WARMUP"))
                        : kDEFAULT_BENCH_WARMUP;
                    int32_t const benchIterations
                        = std::getenv("TRT_EDGELLM_EXTERNAL_PREFILL_BENCH_ITERATIONS") != nullptr
                        ? std::stoi(std::getenv("TRT_EDGELLM_EXTERNAL_PREFILL_BENCH_ITERATIONS"))
                        : kDEFAULT_BENCH_ITERATIONS;
                    ELLM_CHECK(benchTokens > 0 && benchTokens <= visionWarmupTokens,
                        "External-prefill benchmark tokens exceed the selected profile limit");
                    ownership.setLength(prefillSlot0, 0);
                    prefillKV.prepare({prefillSlot0}, prefillStream);
                    ELLM_CHECK(prefillIO->inputsEmbeds.reshape({1, benchTokens, config.hiddenSize}),
                        "Failed to reshape isolated external-prefill embeddings");
                    prefillKV.preparePrefillMetadata(*prefillIO, {benchTokens}, prefillStream, true);
                    rt::InferenceDims const benchDims = config.hasVisionPrefillProfile()
                        ? config.visionPackedPrefillDims(1, benchTokens, benchTokens)
                        : config.packedPrefillDims(1, benchTokens, benchTokens);
                    ELLM_CHECK(pair->externalPrefillExecutor().prepare(
                                   pair->externalPrefillProfile(), benchDims, prefillMap, prefillStream),
                        "Failed to prepare the isolated external-prefill benchmark");
                    ExecutorTiming const timing
                        = measureExecutor(pair->externalPrefillExecutor(), prefillStream, benchWarmup, benchIterations);
                    prefillKV.complete();
                    LOG_INFO(
                        "External-prefill isolated timing: profile=%d tokens=%d warmup=%d iterations=%d "
                        "mean=%.4f ms median=%.4f ms p95=%.4f ms max=%.4f ms",
                        pair->externalPrefillProfile(), benchTokens, benchWarmup, benchIterations, timing.meanMs,
                        timing.medianMs, timing.p95Ms, timing.maxMs);
                }
            }

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

        int32_t const prefillTokenCapacity = config.packedPrefill
            ? std::max(config.maxPackedPrefillChunkTokens, config.maxVisionPackedPrefillChunkTokens)
            : config.maxSupportedInputLength;
        bool const enableBatchedVisionPrefill = std::getenv("TRT_EDGELLM_ENABLE_BATCHED_VISION_PREFILL") != nullptr;
        ELLM_CHECK(!enableBatchedVisionPrefill || config.packedPrefill,
            "Batched vision prefill requires a packed-prefill engine");
        int32_t prefillBatchTokenBudget = config.maxSupportedPrefillBatchSize * 128;
        char const* prefillBatchTokenBudgetValue = std::getenv("TRT_EDGELLM_MAX_PREFILL_BATCH_TOKENS");
        if (prefillBatchTokenBudgetValue != nullptr)
        {
            prefillBatchTokenBudget = std::stoi(prefillBatchTokenBudgetValue);
        }
        int32_t const textProfileTokenCapacity
            = config.maxSupportedPrefillBatchSize * config.maxPackedPrefillChunkTokens;
        int32_t const visionProfileTokenCapacity
            = config.maxSupportedVisionPrefillBatchSize * config.maxVisionPackedPrefillChunkTokens;
        int32_t const profileTokenCapacity = config.packedPrefill
            ? std::max(textProfileTokenCapacity, visionProfileTokenCapacity)
            : config.maxSupportedPrefillBatchSize * prefillTokenCapacity;
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
            selectArgmax(*samplingLogits, selectedIds, stream);
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
#include "phaseSchedulerOptions.inc"
        PhasePolicyWarmupMode const policyWarmupMode = phasePolicyWarmupMode();
        rt::PhaseRuntimeCostTrackerConfig runtimeCostConfig;
        runtimeCostConfig.action = semanticSchedulerConfig.globalCostModelConfig;
        runtimeCostConfig.decodeMinimumSamples = semanticSchedulerConfig.decodeComponentMinSamples;
        runtimeCostConfig.decodeWindowSize = semanticSchedulerConfig.decodeComponentWindow;
        runtimeCostConfig.decodeContextBucketTokens = semanticSchedulerConfig.runtimeDecodeContextBucketTokens;
        if (char const* value = std::getenv("TRT_EDGELLM_CONTEXTUAL_PD"))
        {
            std::string const mode(value);
            ELLM_CHECK(mode == "disabled" || mode == "shadow" || mode == "active",
                "TRT_EDGELLM_CONTEXTUAL_PD must be disabled, shadow, or active");
            runtimeCostConfig.contextualPd.mode = mode == "active" ? rt::PhaseContextualPdMode::kActive
                : mode == "shadow"                                 ? rt::PhaseContextualPdMode::kShadow
                                                                   : rt::PhaseContextualPdMode::kDisabled;
        }
        if (char const* value = std::getenv("TRT_EDGELLM_CONTEXTUAL_PD_MIN_OBSERVATIONS"))
        {
            runtimeCostConfig.contextualPd.minimumObservations = static_cast<size_t>(std::stoull(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_CONTEXTUAL_PD_CONFIDENCE_BETA"))
        {
            runtimeCostConfig.contextualPd.confidenceBeta = std::stod(value);
        }
        auto configureContextualEncoderPair = [](char const* modeVariable, char const* observationsVariable,
                                                  char const* betaVariable, rt::PhaseContextualPdModelConfig& config) {
            if (char const* value = std::getenv(modeVariable))
            {
                std::string const mode(value);
                ELLM_CHECK(mode == "disabled" || mode == "shadow" || mode == "active",
                    std::string(modeVariable) + " must be disabled, shadow, or active");
                config.mode = mode == "active" ? rt::PhaseContextualPdMode::kActive
                    : mode == "shadow"         ? rt::PhaseContextualPdMode::kShadow
                                               : rt::PhaseContextualPdMode::kDisabled;
            }
            if (char const* value = std::getenv(observationsVariable))
            {
                config.minimumObservations = static_cast<size_t>(std::stoull(value));
            }
            if (char const* value = std::getenv(betaVariable))
            {
                config.confidenceBeta = std::stod(value);
            }
        };
        configureContextualEncoderPair("TRT_EDGELLM_CONTEXTUAL_EP", "TRT_EDGELLM_CONTEXTUAL_EP_MIN_OBSERVATIONS",
            "TRT_EDGELLM_CONTEXTUAL_EP_CONFIDENCE_BETA", runtimeCostConfig.contextualEp);
        configureContextualEncoderPair("TRT_EDGELLM_CONTEXTUAL_ED", "TRT_EDGELLM_CONTEXTUAL_ED_MIN_OBSERVATIONS",
            "TRT_EDGELLM_CONTEXTUAL_ED_CONFIDENCE_BETA", runtimeCostConfig.contextualEd);
        if (policyWarmupMode == PhasePolicyWarmupMode::kGraphOnly)
        {
            runtimeCostConfig.contextualPd.mode = rt::PhaseContextualPdMode::kDisabled;
            runtimeCostConfig.contextualEp.mode = rt::PhaseContextualPdMode::kDisabled;
            runtimeCostConfig.contextualEd.mode = rt::PhaseContextualPdMode::kDisabled;
        }
        if (char const* value = std::getenv("TRT_EDGELLM_COMPLETION_CONFORMAL"))
        {
            std::string const enabled(value);
            ELLM_CHECK(enabled == "0" || enabled == "1", "TRT_EDGELLM_COMPLETION_CONFORMAL must be 0 or 1");
            runtimeCostConfig.completionCalibration.enabled = enabled == "1";
        }
        if (char const* value = std::getenv("TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE"))
        {
            std::string const active(value);
            ELLM_CHECK(active == "0" || active == "1", "TRT_EDGELLM_COMPLETION_CONFORMAL_ACTIVE must be 0 or 1");
            runtimeCostConfig.completionCalibration.active = active == "1";
            ELLM_CHECK(
                !runtimeCostConfig.completionCalibration.active || runtimeCostConfig.completionCalibration.enabled,
                "Active completion conformal authority requires TRT_EDGELLM_COMPLETION_CONFORMAL=1");
        }
        if (char const* value = std::getenv("TRT_EDGELLM_COMPLETION_CONFORMAL_MIN_OBSERVATIONS"))
        {
            runtimeCostConfig.completionCalibration.minimumObservations = static_cast<size_t>(std::stoull(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_COMPLETION_CONFORMAL_WINDOW"))
        {
            runtimeCostConfig.completionCalibration.windowSize = static_cast<size_t>(std::stoull(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_COMPLETION_CONFORMAL_TARGET"))
        {
            runtimeCostConfig.completionCalibration.targetCoverage = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_COMPLETION_AUTHORITY_COVERAGE_TOLERANCE"))
        {
            runtimeCostConfig.completionCalibration.authorityCoverageTolerance = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_COMPLETION_AUTHORITY_MIN_OBSERVATIONS"))
        {
            runtimeCostConfig.completionCalibration.authorityMinimumObservations
                = static_cast<size_t>(std::stoull(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_COMPLETION_AUTHORITY_DEMOTION_TOLERANCE"))
        {
            runtimeCostConfig.completionCalibration.authorityDemotionCoverageTolerance = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_COMPLETION_AUTHORITY_MAX_FALSE_SAFE_RATE"))
        {
            runtimeCostConfig.completionCalibration.authorityMaximumFalseSafeRate = std::stod(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_COMPLETION_AUTHORITY_BLEND_WEIGHT"))
        {
            runtimeCostConfig.completionCalibration.authorityBlendWeight = std::stod(value);
        }
        auto completionAblation = [](char const* variable, bool& setting) {
            if (char const* value = std::getenv(variable))
            {
                std::string const enabled(value);
                ELLM_CHECK(enabled == "0" || enabled == "1", std::string(variable) + " must be 0 or 1");
                setting = enabled == "1";
            }
        };
        completionAblation("TRT_EDGELLM_COMPLETION_AUTHORITY_USES_UNCERTAINTY",
            runtimeCostConfig.completionCalibration.authorityUsesUncertainty);
        completionAblation("TRT_EDGELLM_COMPLETION_USE_RESIDUAL_FEATURES",
            runtimeCostConfig.completionCalibration.useResidualFeatures);
        completionAblation("TRT_EDGELLM_COMPLETION_AUTHORITY_PREDICTS_INCUMBENT",
            runtimeCostConfig.completionCalibration.authorityPredictsIncumbent);
        auto runtimeCostTracker = std::make_shared<rt::PhaseRuntimeCostTracker>(runtimeCostConfig);
        semanticSchedulerConfig.runtimeCostTracker = runtimeCostTracker;
        if (semanticSchedulerConfig.globalSchedulerMode != rt::PhaseGlobalSchedulerMode::kDisabled)
        {
            semanticSchedulerConfig.globalDispatchUsesPreReservedMemory = true;
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
        // Stable indexed ownership makes an in-flight limit independent of the
        // largest decode cohort. Keep the old decode-aligned cap only as an
        // explicit compatibility experiment; applying it by default can leave
        // valid stable slots idle and inflate admission/TTFT latency.
        if (std::getenv("TRT_EDGELLM_ENABLE_DECODE_ALIGNED_ADMISSION") != nullptr
            && std::getenv("TRT_EDGELLM_DISABLE_DECODE_ALIGNED_ADMISSION") == nullptr)
        {
            size_t const requestedCapacity = maxInFlightRequests;
            maxInFlightRequests = rt::phaseDecodeAlignedAdmissionCapacity(
                requestedCapacity, static_cast<size_t>(semanticSchedulerConfig.maxDecodeBatchSize));
            if (maxInFlightRequests != requestedCapacity)
            {
                LOG_INFO("Decode-aligned admission: requested=%zu effective=%zu decode_batch=%d", requestedCapacity,
                    maxInFlightRequests, semanticSchedulerConfig.maxDecodeBatchSize);
            }
        }
        serverConfig.maxInFlightRequests = maxInFlightRequests;
        serverConfig.defaultMaxOutputTokens = kSEMANTIC_OUTPUT_TOKENS;
        if (std::getenv("TRT_EDGELLM_IGNORE_EOS") == nullptr)
        {
            serverConfig.eosTokenIds = config.eosTokenIds;
        }
        serverConfig.enablePrefixReuse = enablePrefixReuse;
        serverConfig.enableCudaGraphs = std::getenv("TRT_EDGELLM_CAPTURE_PHASE_GRAPHS") != nullptr;
        serverConfig.synchronizeDecodeSampling = std::getenv("TRT_EDGELLM_SYNCHRONIZE_DECODE_SAMPLING") != nullptr;
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_REFILL_BATCH"))
        {
            serverConfig.admissionRefillBatchSize = static_cast<size_t>(std::stoul(value));
        }
        if (char const* value = std::getenv("TRT_EDGELLM_ADMISSION_REFILL_WINDOW_US"))
        {
            serverConfig.admissionRefillWindowUs = std::stod(value);
        }
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
        serverConfig.enableCompletionAwareAdmissionProjection
            = std::getenv("TRT_EDGELLM_COMPLETION_AWARE_ADMISSION_PROJECTION") != nullptr;
        if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_WAIT_AUTHORITY"))
        {
            serverConfig.enableGlobalWaitAuthority = std::stoi(value) != 0;
        }
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
        char const* activityPrefixValue = std::getenv("TRT_EDGELLM_PHASE_ACTIVITY_PREFIX");
        std::unique_ptr<rt::PhaseActivityTimelineRecorder> activityTimeline;
        std::filesystem::path activityPrefix;
        if (activityPrefixValue != nullptr)
        {
            activityPrefix = activityPrefixValue;
            if (!activityPrefix.parent_path().empty())
            {
                std::filesystem::create_directories(activityPrefix.parent_path());
            }
            activityTimeline = std::make_unique<rt::PhaseActivityTimelineRecorder>(setupStream);
            semanticServer.setActivityTimeline(activityTimeline.get());
            LOG_INFO("Phase activity timeline enabled: prefix=%s", activityPrefix.string().c_str());
        }
        bool const ipcMode = std::getenv("TRT_EDGELLM_PHASE_IPC") != nullptr;
        bool const prefixReuseGate = std::getenv("TRT_EDGELLM_PREFIX_REUSE_GATE") != nullptr;
        char const* visionEngineDir = std::getenv("TRT_EDGELLM_VISION_ENGINE_DIR");
        char const* visionImagePath = std::getenv("TRT_EDGELLM_VISION_IMAGE");
        char const* encoderCalibrationImage = std::getenv("TRT_EDGELLM_PHASE_ENCODER_CALIBRATION_IMAGE");
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
        bool serializeAllEncoderPrefill{};
        auto configureVisionContextMemory = [&](rt::MultimodalRunner& runner) {
            int64_t const requiredBytes = runner.getRequiredContextMemorySize();
            LOG_INFO("Vision context workspace: required=%lld prefill_available=%zu bytes",
                static_cast<long long>(requiredBytes), pair->prefillContextMemory().getMemoryCapacity());
            if (std::getenv("TRT_EDGELLM_TIERED_VISION_CONTEXT_MEMORY") != nullptr)
            {
                int32_t const profileCount = runner.getOptimizationProfileCount();
                ELLM_CHECK(profileCount > 0, "Tiered vision context memory requires a visual profile");
                if (profileCount == 1)
                {
                    rt::TieredVisionContextMemoryInfo const info = pair->configureSharedVisionContextMemory(runner, 0);
                    serializeAllEncoderPrefill = true;
                    LOG_INFO("Shared E/P context arena: total=%lld prefill=%lld vision=%lld serialize_all_encoder=yes",
                        static_cast<long long>(info.arenaBytes), static_cast<long long>(info.prefillBytes),
                        static_cast<long long>(info.largeVisionBytes));
                    return;
                }
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
                rt::PhaseVisionAdapter visionAdapter(
                    *runner, tokenizer, config, encoderStream, visionStoragePolicy, copyStream);
                rt::PhaseThreeCoordinator threePhase(visionAdapter, semanticServer);
                if (activityTimeline != nullptr)
                {
                    threePhase.setActivityTimeline(activityTimeline.get());
                }

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
            // Publish one complete prefill cohort before each arbitration point.
            // Parsing remains asynchronous, so expensive image preparation does
            // not drain an entire burst while the device is idle.
            size_t ipcIngressQuantum = rt::phaseServingIngressQuantum(
                serverConfig.maxPendingRequests, static_cast<size_t>(semanticSchedulerConfig.maxPrefillBatchSize));
            if (char const* value = std::getenv("TRT_EDGELLM_IPC_INGRESS_QUANTUM"))
            {
                ipcIngressQuantum = static_cast<size_t>(std::stoul(value));
            }
            ELLM_CHECK(ipcIngressQuantum > 0, "Phase IPC ingress quantum must be positive");
            bool const emitPhaseMetrics = std::getenv("TRT_EDGELLM_EMIT_PHASE_METRICS") != nullptr;
            std::string const phaseTelemetryLevel = std::getenv("TRT_EDGELLM_PHASE_TELEMETRY_LEVEL") != nullptr
                ? std::getenv("TRT_EDGELLM_PHASE_TELEMETRY_LEVEL")
                : "full";
            ELLM_CHECK(phaseTelemetryLevel == "full" || phaseTelemetryLevel == "research"
                    || phaseTelemetryLevel == "counterfactual",
                "TRT_EDGELLM_PHASE_TELEMETRY_LEVEL must be full, research, or counterfactual");
            bool const emitPhaseRequestTimeline = emitPhaseMetrics && phaseTelemetryLevel == "full";
            bool const emitFullUnifiedSnapshots = phaseTelemetryLevel != "research";
            bool const collectPhaseDispatchMetrics = emitPhaseMetrics && phaseTelemetryLevel == "full";
            std::string const schedulerRunId = std::getenv("TRT_EDGELLM_PHASE_RUN_ID") != nullptr
                ? std::getenv("TRT_EDGELLM_PHASE_RUN_ID")
                : "phase-ipc";
            semanticCoordinator.setMetricsCollectionEnabled(collectPhaseDispatchMetrics);
            size_t const warmupAdmissionLimit = serverConfig.enableAdaptiveAdmission
                ? serverConfig.latencyInFlightRequests
                : serverConfig.maxInFlightRequests;
            int32_t const warmupBatchLimit
                = std::min(semanticSchedulerConfig.maxDecodeBatchSize, static_cast<int32_t>(warmupAdmissionLimit));
            struct DecodeWarmupShape
            {
                int32_t batchSize{};
                int32_t promptTokens{};
                //! Zero preserves the dense paired warmup. Positive values
                //! add low-dimensional contextual coverage without becoming
                //! a workload-specific action table.
                int32_t overlapPrefillRows{};
            };
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
            std::vector<DecodeWarmupShape> warmupShapes;
            if (char const* value = std::getenv("TRT_EDGELLM_IPC_WARMUP_DECODE_SHAPES"))
            {
                std::stringstream stream(value);
                std::string shape;
                while (std::getline(stream, shape, ','))
                {
                    size_t const separator = shape.find(':');
                    ELLM_CHECK(separator != std::string::npos && separator > 0U && separator + 1U < shape.size(),
                        "Phase IPC decode warmup shape must be batch:prompt_tokens");
                    DecodeWarmupShape const parsed{
                        std::stoi(shape.substr(0U, separator)), std::stoi(shape.substr(separator + 1U))};
                    ELLM_CHECK(parsed.batchSize > 0 && parsed.batchSize <= warmupBatchLimit,
                        "Phase IPC decode warmup batch exceeds the active admission/profile limit");
                    ELLM_CHECK(parsed.promptTokens > 0 && parsed.promptTokens + 4 <= config.maxKVCacheCapacity,
                        "Phase IPC decode warmup prompt exceeds the KV capacity");
                    warmupShapes.push_back(parsed);
                }
                ELLM_CHECK(!warmupShapes.empty(), "Phase IPC decode warmup shape list cannot be empty");
            }
            else if (std::getenv("TRT_EDGELLM_DISABLE_IPC_SHAPE_WARMUP") == nullptr)
            {
                std::vector<int32_t> const warmupBatchSizes
                    = rt::phaseServingWarmupBatchSizes(warmupBatchLimit, std::move(requestedWarmupBatchSizes));
                for (int32_t const batchSize : warmupBatchSizes)
                {
                    warmupShapes.push_back({batchSize, 0, 0});
                }
            }
            bool const globalOverlapWarmup
                = semanticSchedulerConfig.globalSchedulerMode == rt::PhaseGlobalSchedulerMode::kActive
                && std::getenv("TRT_EDGELLM_DISABLE_GLOBAL_OVERLAP_WARMUP") == nullptr;
            size_t globalOverlapWarmupSamples = 4U;
            if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_OVERLAP_WARMUP_SAMPLES"))
            {
                globalOverlapWarmupSamples = static_cast<size_t>(std::stoull(value));
            }
            ELLM_CHECK(!globalOverlapWarmup || globalOverlapWarmupSamples > 0U,
                "Phase Global overlap warmup samples must be positive");
            size_t shapeWarmupSamples = globalOverlapWarmup ? globalOverlapWarmupSamples : 1U;
            if (char const* value = std::getenv("TRT_EDGELLM_IPC_WARMUP_SHAPE_SAMPLES"))
            {
                shapeWarmupSamples = static_cast<size_t>(std::stoull(value));
            }
            ELLM_CHECK(shapeWarmupSamples > 0U, "Phase IPC warmup shape sample count must be positive");
            std::vector<DecodeWarmupShape> executionWarmupShapes;
            for (DecodeWarmupShape const& shape : warmupShapes)
            {
                executionWarmupShapes.insert(executionWarmupShapes.end(), shapeWarmupSamples, shape);
            }
            if (globalOverlapWarmup && !warmupShapes.empty())
            {
                DecodeWarmupShape const& largestDecode = *std::max_element(warmupShapes.begin(), warmupShapes.end(),
                    [](DecodeWarmupShape const& lhs, DecodeWarmupShape const& rhs) {
                        return lhs.batchSize < rhs.batchSize;
                    });
                int32_t const prefillLimit = semanticSchedulerConfig.maxPrefillBatchSize;
                for (int32_t prefillRows = 1; prefillRows < prefillLimit; prefillRows *= 2)
                {
                    DecodeWarmupShape contextualShape = largestDecode;
                    contextualShape.overlapPrefillRows = prefillRows;
                    executionWarmupShapes.insert(
                        executionWarmupShapes.end(), globalOverlapWarmupSamples, contextualShape);
                }
            }
            semanticCoordinator.scheduler().setGlobalWarmupProbeMode(globalOverlapWarmup);
            uint64_t warmupRequestId = 1000000;
            size_t warmedRequests{};
            size_t warmupPollGuard = 100000000U;
            if (char const* value = std::getenv("TRT_EDGELLM_IPC_WARMUP_POLL_GUARD"))
            {
                warmupPollGuard = static_cast<size_t>(std::stoull(value));
            }
            ELLM_CHECK(warmupPollGuard > 0U, "Phase IPC warmup poll guard must be positive");
            for (DecodeWarmupShape const& shape : executionWarmupShapes)
            {
                int32_t const batchSize = shape.batchSize;
                std::vector<int32_t> warmupPrompt = semanticPrompts.at(20000);
                if (shape.promptTokens > 0)
                {
                    ELLM_CHECK(!warmupPrompt.empty(), "Phase IPC warmup seed prompt cannot be empty");
                    warmupPrompt.resize(static_cast<size_t>(shape.promptTokens), warmupPrompt.back());
                }
                for (int32_t row{}; row < batchSize; ++row)
                {
                    int32_t const outputTokens = globalOverlapWarmup ? 4 : 2;
                    auto const submission = semanticServer.submit(warmupRequestId++, warmupPrompt, outputTokens);
                    ELLM_CHECK(submission.status == rt::IndependentPhaseServerStatus::kAdmitted,
                        "Phase IPC shape warmup request was not admitted");
                }
                size_t overlapWarmupRows{};
                if (globalOverlapWarmup)
                {
                    // Leave a decode cohort resident, then admit a bounded P
                    // cohort. Synthetic requests have no user SLO, so the
                    // scheduler can safely collect a direct P+D CUDA sample
                    // without exploring on production traffic.
                    size_t pollCount{};
                    while (true)
                    {
                        static_cast<void>(semanticServer.poll());
                        rt::IndependentPhaseServerArbitrationSnapshot const snapshot
                            = semanticServer.arbitrationSnapshot();
                        if (snapshot.busy && snapshot.inFlightKind == rt::PhaseDispatchKind::kDecode)
                        {
                            break;
                        }
                        ELLM_CHECK(!semanticServer.empty() && ++pollCount < 1000000U,
                            "Phase Global overlap warmup did not reach decode");
                    }
                    size_t const availableRows = warmupAdmissionLimit > static_cast<size_t>(batchSize)
                        ? warmupAdmissionLimit - static_cast<size_t>(batchSize)
                        : 0U;
                    size_t const requestedPrefillRows = shape.overlapPrefillRows > 0
                        ? static_cast<size_t>(shape.overlapPrefillRows)
                        : static_cast<size_t>(batchSize);
                    overlapWarmupRows = std::min({availableRows, requestedPrefillRows,
                        static_cast<size_t>(semanticSchedulerConfig.maxPrefillBatchSize)});
                    for (size_t row{}; row < overlapWarmupRows; ++row)
                    {
                        auto const submission = semanticServer.submit(warmupRequestId++, warmupPrompt, 2);
                        ELLM_CHECK(submission.status == rt::IndependentPhaseServerStatus::kAdmitted,
                            "Phase Global overlap warmup request was not admitted");
                    }
                }
                semanticServer.runUntilIdle(warmupPollGuard);
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
                ELLM_CHECK(completed == static_cast<size_t>(batchSize) + overlapWarmupRows,
                    "Phase IPC shape warmup did not complete every request");
                warmedRequests += completed;
            }
            ELLM_CHECK(semanticServer.empty() && ownership.availableSlots() == maxStableSlots,
                "Phase IPC shape warmup did not release every stable slot");
            CUDA_CHECK(cudaStreamSynchronize(prefillStream));
            CUDA_CHECK(cudaStreamSynchronize(decodeStream));
            CUDA_CHECK(cudaStreamSynchronize(copyStream));
            CUDA_CHECK(cudaStreamSynchronize(setupStream));
            semanticCoordinator.scheduler().setGlobalWarmupProbeMode(false);
            rt::PhaseSchedulerTelemetry const warmupTelemetry = semanticCoordinator.scheduler().telemetry();
            // Shape priming is not production traffic. Keep graph entries, but
            // do not let synthetic queue waits drive adaptive admission.
            semanticCoordinator.scheduler().resetSchedulingHistory();
            // Completion-authority evidence is deliberately retained. It is
            // collected from held-out warmup observations and belongs to the
            // calibrated physical model, not to serving-policy telemetry.
            if (policyWarmupMode == PhasePolicyWarmupMode::kGraphOnly
                || policyWarmupMode == PhasePolicyWarmupMode::kZeroStart)
            {
                semanticCoordinator.scheduler().resetExecutionCostHistory();
                semanticCoordinator.scheduler().resetPolicyPosterior();
            }
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
            LOG_INFO("Phase IPC shape warmup: mode=%s batches=%zu requests=%zu overlap_samples=%zu safe_probes=%zu",
                phasePolicyWarmupModeName(policyWarmupMode), executionWarmupShapes.size(), warmedRequests,
                warmupTelemetry.overlapSampleCount, warmupTelemetry.globalSafeProbeCount);
            if (activityTimeline != nullptr)
            {
                activityTimeline->reset(setupStream);
            }
            cudaStream_t ipcEncoderStream{};
            std::unique_ptr<rt::MultimodalRunner> ipcVisionRunner;
            std::unique_ptr<rt::PhaseVisionAdapter> ipcVisionAdapter;
            std::unique_ptr<rt::PhaseThreeCoordinator> ipcThreePhase;
            if (visionEngineDir != nullptr)
            {
                bool const hasAtomicExternalPrefill = pair->hasExternalPrefillExecutor()
                    && ((config.hasVisionPrefillProfile()
                            && config.maxVisionPackedPrefillChunkTokens >= config.maxSupportedInputLength)
                        || (!config.hasVisionPrefillProfile()
                            && config.maxPackedPrefillChunkTokens >= config.maxSupportedInputLength));
                ELLM_CHECK(!config.packedPrefill || hasAtomicExternalPrefill,
                    "Three-phase packed vision requires a dedicated atomic external-prefill context covering "
                    "maxInputLength");
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
                    *ipcVisionRunner, tokenizer, config, ipcEncoderStream, visionStoragePolicy, copyStream);
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
                threePhaseConfig.runtimeCostTracker = runtimeCostTracker;
                threePhaseConfig.globalExperimentalOverlapPercent
                    = semanticSchedulerConfig.globalExperimentalOverlapPercent;
                if (char const* value = std::getenv("TRT_EDGELLM_EXPERIMENTAL_ENCODER_PREFILL_OVERLAP_PERCENT"))
                {
                    threePhaseConfig.globalExperimentalEncoderPrefillOverlapPercent = std::stoi(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_EXPERIMENTAL_ENCODER_DECODE_OVERLAP_PERCENT"))
                {
                    threePhaseConfig.globalExperimentalEncoderDecodeOverlapPercent = std::stoi(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_REPLAY_DECISION_SEQUENCE"))
                {
                    threePhaseConfig.globalReplayDecisionSequence = std::stoull(value);
                    char const* action = std::getenv("TRT_EDGELLM_REPLAY_ACTION_KIND");
                    ELLM_CHECK(action != nullptr, "Causal replay decision requires an action kind");
                    std::optional<rt::PhaseGlobalActionKind> const kind = rt::phaseGlobalActionKindFromName(action);
                    ELLM_CHECK(kind.has_value() && *kind != rt::PhaseGlobalActionKind::kNone,
                        "Unknown or empty causal replay action kind");
                    threePhaseConfig.globalReplayActionKind = *kind;
                }
                if (char const* value = std::getenv("TRT_EDGELLM_DIRECTIONAL_INJECTION"))
                {
                    std::optional<rt::PhaseUnifiedActionDirection> const direction
                        = rt::phaseUnifiedActionDirectionFromName(value);
                    ELLM_CHECK(direction.has_value(), "Unknown directional-injection phase order");
                    threePhaseConfig.directionalInjection.direction = *direction;
                    char const* target = std::getenv("TRT_EDGELLM_DIRECTIONAL_INJECTION_TARGET");
                    char const* incumbent = std::getenv("TRT_EDGELLM_DIRECTIONAL_INJECTION_INCUMBENT_US");
                    char const* newcomer = std::getenv("TRT_EDGELLM_DIRECTIONAL_INJECTION_NEWCOMER_US");
                    ELLM_CHECK(target != nullptr && incumbent != nullptr && newcomer != nullptr,
                        "Directional injection requires target, incumbent, and newcomer references");
                    threePhaseConfig.directionalInjection.targetFraction = std::stod(target);
                    threePhaseConfig.directionalInjection.incumbentReferenceUs = std::stod(incumbent);
                    threePhaseConfig.directionalInjection.newcomerReferenceUs = std::stod(newcomer);
                    if (char const* delay = std::getenv("TRT_EDGELLM_DIRECTIONAL_INJECTION_DELAY_US"))
                    {
                        threePhaseConfig.directionalInjection.requestedDelayUs = std::stoull(delay);
                    }
                    else
                    {
                        threePhaseConfig.directionalInjection.requestedDelayUs
                            = static_cast<uint64_t>(std::llround(threePhaseConfig.directionalInjection.targetFraction
                                * threePhaseConfig.directionalInjection.incumbentReferenceUs));
                    }
                }
                threePhaseConfig.enableGlobalEncoderPrefillAction
                    = semanticSchedulerConfig.globalSchedulerMode == rt::PhaseGlobalSchedulerMode::kActive
                    && std::getenv("TRT_EDGELLM_DISABLE_GLOBAL_ENCODER_PREFILL_ACTION") == nullptr;
                threePhaseConfig.enableGlobalFormationAwareSelection
                    = semanticSchedulerConfig.globalSchedulerMode == rt::PhaseGlobalSchedulerMode::kActive
                    && std::getenv("TRT_EDGELLM_ENABLE_GLOBAL_FORMATION_AWARE") != nullptr
                    && std::getenv("TRT_EDGELLM_DISABLE_GLOBAL_FORMATION_AWARE") == nullptr;
                if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_FORMATION_REALIZED_DISPATCHES"))
                {
                    threePhaseConfig.globalFormationRealizedDispatches = static_cast<size_t>(std::stoull(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_SAFE_PROBE_SLACK_MULTIPLIER"))
                {
                    threePhaseConfig.globalSafeProbeSlackMultiplier = std::stof(value);
                }
                if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_SAFE_PROBE_INTERVAL"))
                {
                    threePhaseConfig.globalSafeProbeInterval = static_cast<size_t>(std::stoull(value));
                }
                if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_OVERLAP_MIN_SAMPLES"))
                {
                    threePhaseConfig.globalCostModelConfig.overlapMinSamples = static_cast<size_t>(std::stoull(value));
                }
                threePhaseConfig.exclusiveEncoderInputTokenThreshold = tieredVisionExclusiveInputTokens;
                threePhaseConfig.serializeAllEncoderPrefill = serializeAllEncoderPrefill;
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
                else if (char const* value = std::getenv("TRT_EDGELLM_GLOBAL_DECODE_TPOT_TARGET_US"))
                {
                    // The global queue scheduler and residual E+D protection
                    // must use the same profile-free decode service target.
                    threePhaseConfig.encodedCapacityDecodeTpotTargetUs = std::stod(value);
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
                if (encoderCalibrationImage != nullptr)
                {
                    ELLM_CHECK(semanticSchedulerConfig.globalSchedulerMode == rt::PhaseGlobalSchedulerMode::kActive,
                        "Encoder calibration requires the active Global scheduler");
                    std::vector<size_t> requestedEncoderBatches;
                    if (char const* value = std::getenv("TRT_EDGELLM_PHASE_ENCODER_CALIBRATION_BATCHES"))
                    {
                        std::stringstream stream(value);
                        std::string batchSize;
                        while (std::getline(stream, batchSize, ','))
                        {
                            ELLM_CHECK(!batchSize.empty(), "Encoder calibration batch list contains an empty entry");
                            requestedEncoderBatches.push_back(static_cast<size_t>(std::stoull(batchSize)));
                        }
                    }
                    size_t encoderCalibrationSamples = threePhaseConfig.globalCostModelConfig.overlapMinSamples;
                    if (char const* value = std::getenv("TRT_EDGELLM_PHASE_ENCODER_CALIBRATION_SAMPLES"))
                    {
                        encoderCalibrationSamples = static_cast<size_t>(std::stoull(value));
                    }
                    ELLM_CHECK(encoderCalibrationSamples > 0U, "Encoder calibration sample count must be positive");
                    bool const calibrateEncoderDecode
                        = std::getenv("TRT_EDGELLM_DISABLE_PHASE_ENCODER_DECODE_CALIBRATION") == nullptr;
                    rt::imageUtils::ImageData const calibrationImage
                        = rt::imageUtils::loadImageFromFile(encoderCalibrationImage);
                    auto makeCalibrationRequest = [&]() {
                        rt::LLMGenerationRequest request{};
                        rt::LLMGenerationRequest::Request logicalRequest;
                        logicalRequest.messages.push_back(
                            {"user", {{"image", encoderCalibrationImage}, {"text", "Describe the image briefly."}}});
                        logicalRequest.imageBuffers.push_back(calibrationImage);
                        request.requests.push_back(std::move(logicalRequest));
                        request.temperature = 0.0F;
                        request.topP = 1.0F;
                        request.topK = 1;
                        request.maxGenerateLength = 2;
                        request.applyChatTemplate = true;
                        request.addGenerationPrompt = true;
                        return request;
                    };
                    size_t const encoderInputTokensPerRequest
                        = ipcVisionAdapter->estimateInputTokens(makeCalibrationRequest());
                    ELLM_CHECK(encoderInputTokensPerRequest > 0U,
                        "Encoder calibration image produced no encoder input tokens");
                    size_t encoderCalibrationLimit
                        = std::min(threePhaseConfig.maxEncoderBatchSize, threePhaseConfig.maxEncodedInFlight);
                    size_t const encoderInputTokenLimit = ipcVisionAdapter->maxInputTokens();
                    if (encoderInputTokenLimit > 0U)
                    {
                        encoderCalibrationLimit
                            = std::min(encoderCalibrationLimit, encoderInputTokenLimit / encoderInputTokensPerRequest);
                    }
                    ELLM_CHECK(encoderCalibrationLimit > 0U,
                        "Encoder calibration image exceeds the physical encoder token profile");
                    std::vector<size_t> const encoderCalibrationBatches
                        = rt::phaseEncoderCalibrationBatchSizes(encoderCalibrationLimit, requestedEncoderBatches);
                    constexpr size_t kEncoderContextBucketTokens = 1024U;
                    constexpr int32_t kEncoderCalibrationOutputTokens = 2;
                    constexpr int32_t kDecodeCalibrationOutputTokens = 32;
                    uint64_t encoderCalibrationRequestId = 1100000U;
                    size_t encoderCalibrationExecutions{};
                    size_t encoderDecodeCalibrationExecutions{};
                    size_t encoderCalibrationRequests{};
                    auto drainCalibration = [&](rt::PhaseThreeCoordinator& calibration, size_t expectedCompletions) {
                        size_t pollCount{};
                        while (!calibration.empty())
                        {
                            static_cast<void>(calibration.poll());
                            ELLM_CHECK(++pollCount < 2000000U, "Encoder calibration exceeded its poll guard");
                        }
                        size_t completions{};
                        while (calibration.tryPopCompletion().has_value())
                        {
                            ++completions;
                        }
                        while (calibration.tryPopToken().has_value())
                        {
                        }
                        if (semanticPrefixCache != nullptr)
                        {
                            semanticPrefixCache->clear();
                        }
                        ELLM_CHECK(completions == expectedCompletions,
                            "Encoder calibration did not complete every synthetic request");
                    };
                    auto submitEncoderCalibrationBatch = [&](rt::PhaseThreeCoordinator& calibration, size_t batchSize) {
                        for (size_t row{}; row < batchSize; ++row)
                        {
                            rt::PhaseThreeSubmissionStatus const submitted
                                = calibration.submit(encoderCalibrationRequestId++, makeCalibrationRequest(),
                                    kEncoderCalibrationOutputTokens);
                            ELLM_CHECK(submitted != rt::PhaseThreeSubmissionStatus::kDuplicateRequest,
                                "Encoder calibration generated a duplicate request ID");
                        }
                        encoderCalibrationRequests += batchSize;
                    };

                    semanticCoordinator.scheduler().setGlobalWarmupProbeMode(true);
                    {
                        rt::PhaseThreeCoordinator calibration(*ipcVisionAdapter, semanticServer, threePhaseConfig);
                        calibration.setGlobalWarmupProbeMode(true);
                        for (size_t const batchSize : encoderCalibrationBatches)
                        {
                            size_t const totalInputTokens = encoderInputTokensPerRequest * batchSize;
                            int32_t const contextBucket = static_cast<int32_t>(
                                (totalInputTokens + kEncoderContextBucketTokens - 1U) / kEncoderContextBucketTokens);
                            rt::PhaseGlobalActionKey const encoderKey{rt::PhaseGlobalActionKind::kEncoder,
                                static_cast<int32_t>(batchSize), 0, 0, contextBucket, 0};
                            size_t const existingSamples = runtimeCostTracker->sampleCount(encoderKey);
                            size_t const requiredExecutions = encoderCalibrationSamples > existingSamples
                                ? encoderCalibrationSamples - existingSamples
                                : 0U;
                            for (size_t sample{}; sample < requiredExecutions; ++sample)
                            {
                                submitEncoderCalibrationBatch(calibration, batchSize);
                                drainCalibration(calibration, batchSize);
                                ++encoderCalibrationExecutions;
                            }
                        }
                        if (calibrateEncoderDecode)
                        {
                            for (size_t const encoderBatchSize : encoderCalibrationBatches)
                            {
                                for (size_t sample{}; sample < encoderCalibrationSamples; ++sample)
                                {
                                    std::vector<rt::PhaseGlobalOverlapCostRecord> const diagnostics
                                        = calibration.globalCalibrationDiagnostics();
                                    bool const calibrated = std::any_of(diagnostics.begin(), diagnostics.end(),
                                        [&](rt::PhaseGlobalOverlapCostRecord const& record) {
                                            return record.key.kind == rt::PhaseGlobalActionKind::kEncoderDecode
                                                && record.key.primaryBatchSize == static_cast<int32_t>(encoderBatchSize)
                                                && (record.diagnostic.status
                                                        == rt::PhaseGlobalOverlapCostStatus::kEligible
                                                    || record.diagnostic.status
                                                        == rt::PhaseGlobalOverlapCostStatus::kUnprofitable);
                                        });
                                    if (calibrated)
                                    {
                                        break;
                                    }
                                    size_t const decodeRows = std::max<size_t>(1U,
                                        std::min({encoderBatchSize, warmupAdmissionLimit,
                                            static_cast<size_t>(semanticSchedulerConfig.maxDecodeBatchSize)}));
                                    for (size_t row{}; row < decodeRows; ++row)
                                    {
                                        auto const submission = semanticServer.submit(encoderCalibrationRequestId++,
                                            semanticPrompts.at(20000), kDecodeCalibrationOutputTokens);
                                        ELLM_CHECK(submission.status == rt::IndependentPhaseServerStatus::kAdmitted,
                                            "Encoder-decode calibration seed was not admitted");
                                    }
                                    size_t pollCount{};
                                    while (true)
                                    {
                                        static_cast<void>(calibration.poll());
                                        rt::IndependentPhaseServerArbitrationSnapshot const snapshot
                                            = semanticServer.arbitrationSnapshot();
                                        if (snapshot.busy && snapshot.inFlightKind == rt::PhaseDispatchKind::kDecode)
                                        {
                                            break;
                                        }
                                        ELLM_CHECK(!semanticServer.empty() && ++pollCount < 1000000U,
                                            "Encoder-decode calibration did not reach decode");
                                    }
                                    submitEncoderCalibrationBatch(calibration, encoderBatchSize);
                                    drainCalibration(calibration, decodeRows + encoderBatchSize);
                                    ++encoderDecodeCalibrationExecutions;
                                }
                            }
                        }
                        calibration.setGlobalWarmupProbeMode(false);
                    }
                    semanticCoordinator.scheduler().setGlobalWarmupProbeMode(false);
                    CUDA_CHECK(cudaStreamSynchronize(ipcEncoderStream));
                    CUDA_CHECK(cudaStreamSynchronize(prefillStream));
                    CUDA_CHECK(cudaStreamSynchronize(decodeStream));
                    CUDA_CHECK(cudaStreamSynchronize(copyStream));
                    CUDA_CHECK(cudaStreamSynchronize(setupStream));
                    semanticCoordinator.scheduler().resetSchedulingHistory();
                    // Preserve held-out completion evidence across the epoch
                    // transition; only serving-policy history is reset.
                    if (policyWarmupMode == PhasePolicyWarmupMode::kGraphOnly
                        || policyWarmupMode == PhasePolicyWarmupMode::kZeroStart)
                    {
                        semanticCoordinator.scheduler().resetExecutionCostHistory();
                        semanticCoordinator.scheduler().resetPolicyPosterior();
                    }
                    LOG_INFO(
                        "Phase encoder calibration: shapes=%zu isolated=%zu encoder_decode=%zu vision_requests=%zu",
                        encoderCalibrationBatches.size(), encoderCalibrationExecutions,
                        encoderDecodeCalibrationExecutions, encoderCalibrationRequests);
                }
                ipcThreePhase
                    = std::make_unique<rt::PhaseThreeCoordinator>(*ipcVisionAdapter, semanticServer, threePhaseConfig);
                if (activityTimeline != nullptr)
                {
                    ipcThreePhase->setActivityTimeline(activityTimeline.get());
                }
            }
            else
            {
                ELLM_CHECK(encoderCalibrationImage == nullptr,
                    "Encoder calibration image requires TRT_EDGELLM_VISION_ENGINE_DIR");
            }
            std::deque<rt::PhaseTimelineEvent> phaseTimelineEvents;
            std::deque<rt::PhaseVisionEncoderBatchMetric> encoderBatchMetrics;
            std::deque<rt::PhaseFormationRealizedEpisode> formationEpisodes;
            std::deque<rt::PhaseUnifiedEvent> unifiedSchedulerEvents;
            if (emitPhaseMetrics)
            {
                if (emitPhaseRequestTimeline)
                {
                    auto const timelineCallback
                        = [&](rt::PhaseTimelineEvent const& event) { phaseTimelineEvents.push_back(event); };
                    semanticServer.setTimelineCallback(timelineCallback);
                    if (ipcThreePhase != nullptr)
                    {
                        ipcThreePhase->setTimelineCallback(timelineCallback);
                    }
                }
                if (ipcThreePhase != nullptr)
                {
                    ipcThreePhase->setEncoderBatchMetricCallback([&](rt::PhaseVisionEncoderBatchMetric const& metric) {
                        encoderBatchMetrics.push_back(metric);
                    });
                    ipcThreePhase->setFormationEpisodeCallback([&](rt::PhaseFormationRealizedEpisode const& episode) {
                        formationEpisodes.push_back(episode);
                    });
                    ipcThreePhase->setUnifiedEventCallback(
                        [&](rt::PhaseUnifiedEvent const& event) { unifiedSchedulerEvents.push_back(event); },
                        emitFullUnifiedSnapshots);
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
            std::deque<PendingPhaseIpcInput> pendingInputTasks;
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
            size_t telemetryWriteBytes{};
            double outputSerializationUs{};
            std::ofstream phaseTelemetryOutput;
            if (char const* value = std::getenv("TRT_EDGELLM_PHASE_TELEMETRY_PATH"))
            {
                std::filesystem::path const path(value);
                if (path.has_parent_path())
                {
                    std::filesystem::create_directories(path.parent_path());
                }
                phaseTelemetryOutput.open(path, std::ios::out | std::ios::trunc);
                ELLM_CHECK(phaseTelemetryOutput.good(), "Failed to open phase telemetry output path");
            }
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
                        size_t responseBytes{};
                        size_t telemetryBytes{};
                        for (std::string const& line : readyLines)
                        {
                            bool const telemetry = phaseTelemetryOutput.is_open() && line.rfind("PHASE_", 0U) == 0U
                                && line.rfind("PHASE_EVENT\t", 0U) != 0U;
                            (telemetry ? telemetryBytes : responseBytes) += line.size() + 1U;
                        }
                        std::string responsePayload;
                        std::string telemetryPayload;
                        responsePayload.reserve(responseBytes);
                        telemetryPayload.reserve(telemetryBytes);
                        for (std::string& line : readyLines)
                        {
                            bool const telemetry = phaseTelemetryOutput.is_open() && line.rfind("PHASE_", 0U) == 0U
                                && line.rfind("PHASE_EVENT\t", 0U) != 0U;
                            std::string& payload = telemetry ? telemetryPayload : responsePayload;
                            payload.append(line);
                            payload.push_back('\n');
                        }
                        if (!telemetryPayload.empty())
                        {
                            phaseTelemetryOutput.write(
                                telemetryPayload.data(), static_cast<std::streamsize>(telemetryPayload.size()));
                            telemetryWriteBytes += telemetryPayload.size();
                        }
                        if (!responsePayload.empty())
                        {
                            std::cout.write(
                                responsePayload.data(), static_cast<std::streamsize>(responsePayload.size()));
                            std::cout.flush();
                        }
                        ++outputWriteBatches;
                        outputWriteRecords += readyRecords.size();
                        outputWriteBytes += responsePayload.size();
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
            auto tokenCallback
                = [&](rt::IndependentPhaseServerToken&& event) { nativeTokenEvents.push_back(std::move(event)); };
            auto completionCallback = [&](rt::IndependentPhaseServerCompletion&& event) {
                nativeCompletionEvents.push_back(std::move(event));
            };
            if (ipcThreePhase != nullptr)
            {
                ipcThreePhase->setEventCallbacks(std::move(tokenCallback), std::move(completionCallback));
            }
            else
            {
                semanticServer.setEventCallbacks(std::move(tokenCallback), std::move(completionCallback));
            }
            auto popTokenEvent = [&]() -> std::optional<rt::IndependentPhaseServerToken> {
                if (nativeTokenEvents.empty())
                {
                    return std::nullopt;
                }
                rt::IndependentPhaseServerToken event = std::move(nativeTokenEvents.front());
                nativeTokenEvents.pop_front();
                return event;
            };
            auto popCompletionEvent = [&]() -> std::optional<rt::IndependentPhaseServerCompletion> {
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
            size_t ipcRequestAdapterOutOfOrderCompletions{};
            size_t ipcRequestAdapterMaxReadyBypass{};
            bool allowOutOfOrderAdapterCompletion{true};
            char const* const canonicalVisionOrder = std::getenv("TRT_EDGELLM_CANONICAL_VISION_ADAPTER_ORDER");
            char const* const disableOutOfOrder = std::getenv("TRT_EDGELLM_DISABLE_IPC_ADAPTER_OUT_OF_ORDER");
            ELLM_CHECK(canonicalVisionOrder == nullptr || disableOutOfOrder == nullptr,
                "TRT_EDGELLM_CANONICAL_VISION_ADAPTER_ORDER and "
                "TRT_EDGELLM_DISABLE_IPC_ADAPTER_OUT_OF_ORDER cannot both be set");
            if (canonicalVisionOrder != nullptr)
            {
                std::string const enabled(canonicalVisionOrder);
                ELLM_CHECK(
                    enabled == "0" || enabled == "1", "TRT_EDGELLM_CANONICAL_VISION_ADAPTER_ORDER must be 0 or 1");
                allowOutOfOrderAdapterCompletion = enabled == "0";
            }
            else if (disableOutOfOrder != nullptr)
            {
                std::string const disabled(disableOutOfOrder);
                ELLM_CHECK(
                    disabled == "0" || disabled == "1", "TRT_EDGELLM_DISABLE_IPC_ADAPTER_OUT_OF_ORDER must be 0 or 1");
                allowOutOfOrderAdapterCompletion = disabled == "0";
            }
            std::thread inputReader([&]() {
                uint64_t ingressSequence{};
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
                            uint64_t const sequence = ingressSequence++;
                            PhaseIpcOrderingClass const orderingClass = phaseIpcOrderingClass(line);
                            pendingInputTasks.push_back({orderingClass,
                                std::async(std::launch::async,
                                    [line = std::move(line), sequence,
                                        defaultMaxOutputTokens = serverConfig.defaultMaxOutputTokens]() {
                                        auto const adapterStart = std::chrono::steady_clock::now();
                                        PhaseIpcInput input = parsePhaseIpcInput(line, defaultMaxOutputTokens);
                                        input.ingressSequence = sequence;
                                        input.adapterUs = std::chrono::duration<double, std::micro>(
                                            std::chrono::steady_clock::now() - adapterStart)
                                                              .count();
                                        return input;
                                    })});
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
            size_t measurementEpoch{};
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
                    size_t readyTaskIndex{};
                    {
                        std::lock_guard<std::mutex> lock(pendingMutex);
                        if (pendingInputTasks.empty())
                        {
                            break;
                        }
                        auto readyTask = pendingInputTasks.begin();
                        if (allowOutOfOrderAdapterCompletion)
                        {
                            readyTask = std::find_if(pendingInputTasks.begin(), pendingInputTasks.end(),
                                [](PendingPhaseIpcInput& candidate) {
                                    return candidate.task.wait_for(std::chrono::seconds(0))
                                        == std::future_status::ready;
                                });
                        }
                        else
                        {
                            auto const firstVision = std::find_if(pendingInputTasks.begin(), pendingInputTasks.end(),
                                [](PendingPhaseIpcInput const& candidate) {
                                    return candidate.orderingClass == PhaseIpcOrderingClass::kVision;
                                });
                            auto const barrier = std::find_if(pendingInputTasks.begin(), pendingInputTasks.end(),
                                [](PendingPhaseIpcInput const& candidate) {
                                    return candidate.orderingClass == PhaseIpcOrderingClass::kBarrier;
                                });
                            readyTask = std::find_if(pendingInputTasks.begin(),
                                barrier == pendingInputTasks.end() ? pendingInputTasks.end() : std::next(barrier),
                                [&](PendingPhaseIpcInput& candidate) {
                                    bool const orderedVision = candidate.orderingClass != PhaseIpcOrderingClass::kVision
                                        || &candidate == &*firstVision;
                                    bool const orderedBarrier
                                        = candidate.orderingClass != PhaseIpcOrderingClass::kBarrier
                                        || &candidate == &pendingInputTasks.front();
                                    return orderedVision && orderedBarrier
                                        && candidate.task.wait_for(std::chrono::seconds(0))
                                        == std::future_status::ready;
                                });
                        }
                        if (readyTask == pendingInputTasks.end())
                        {
                            break;
                        }
                        readyTaskIndex = static_cast<size_t>(std::distance(pendingInputTasks.begin(), readyTask));
                        task = std::move(readyTask->task);
                        pendingInputTasks.erase(readyTask);
                    }
                    ipcRequestAdapterOutOfOrderCompletions += readyTaskIndex > 0U ? 1U : 0U;
                    ipcRequestAdapterMaxReadyBypass = std::max(ipcRequestAdapterMaxReadyBypass, readyTaskIndex);
                    inputAdapterReady.notify_one();
                    PhaseIpcInput input = task.get();
                    ipcRequestAdapterUs += input.adapterUs;
                    ++ipcRequestAdapterInputs;
                    inputs.push_back(std::move(input));
                }
                if (inputs.size() > 1U)
                {
                    std::stable_sort(
                        inputs.begin(), inputs.end(), [](PhaseIpcInput const& left, PhaseIpcInput const& right) {
                            return left.ingressSequence < right.ingressSequence;
                        });
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
                    if (input.kind == PhaseIpcKind::kCalibrationBegin || input.kind == PhaseIpcKind::kCalibrationStatus
                        || input.kind == PhaseIpcKind::kCalibrationEnd)
                    {
                        bool const active = input.kind == PhaseIpcKind::kCalibrationBegin;
                        bool const idle = ipcThreePhase != nullptr ? ipcThreePhase->empty() : semanticServer.empty();
                        if (!idle)
                        {
                            emitEvent({{"type", "error"}, {"request_index", requestId},
                                {"message", "calibration epoch may only change while the backend is idle"}});
                            ++ingestedLines;
                            continue;
                        }
                        rt::PhaseThreeCoordinatorMetrics const calibrationMetrics
                            = ipcThreePhase != nullptr ? ipcThreePhase->metrics() : rt::PhaseThreeCoordinatorMetrics{};
                        rt::PhaseSchedulerTelemetry const calibrationQueueMetrics
                            = semanticCoordinator.scheduler().telemetry();
                        std::vector<rt::PhaseGlobalOverlapCostRecord> calibrationCosts
                            = semanticCoordinator.scheduler().globalCalibrationDiagnostics();
                        if (ipcThreePhase != nullptr)
                        {
                            std::vector<rt::PhaseGlobalOverlapCostRecord> encoderCosts
                                = ipcThreePhase->globalCalibrationDiagnostics();
                            calibrationCosts.insert(calibrationCosts.end(), encoderCosts.begin(), encoderCosts.end());
                        }
                        nlohmann::json calibrationCostKeys = nlohmann::json::array();
                        size_t calibratedCostKeys{};
                        size_t requiredCostKeys{};
                        for (rt::PhaseGlobalOverlapCostRecord const& cost : calibrationCosts)
                        {
                            bool const calibrated
                                = cost.diagnostic.status == rt::PhaseGlobalOverlapCostStatus::kEligible
                                || cost.diagnostic.status == rt::PhaseGlobalOverlapCostStatus::kUnprofitable;
                            requiredCostKeys += cost.required ? 1U : 0U;
                            calibratedCostKeys += calibrated && cost.required ? 1U : 0U;
                            calibrationCostKeys.push_back({{"action", rt::phaseGlobalActionKindName(cost.key.kind)},
                                {"primary_batch_size", cost.key.primaryBatchSize},
                                {"secondary_batch_size", cost.key.secondaryBatchSize},
                                {"chunk_length", cost.key.chunkLength},
                                {"primary_context_bucket", cost.key.primaryContextBucket},
                                {"secondary_context_bucket", cost.key.secondaryContextBucket},
                                {"execution_variant", rt::phaseExecutionVariantName(cost.key.executionVariant)},
                                {"residual_anchor", rt::phaseGlobalResidualAnchorName(cost.key.residualAnchor)},
                                {"status", rt::phaseGlobalOverlapCostStatusName(cost.diagnostic.status)},
                                {"sample_count", cost.diagnostic.sampleCount},
                                {"opportunity_count", cost.opportunityCount}, {"required", cost.required},
                                {"robust_compression", cost.diagnostic.robustCompression}});
                        }
                        size_t contextualRequiredDirections{};
                        size_t contextualReadyDirections{};
                        size_t completionRequiredDirections{};
                        size_t completionCompleteDirections{};
                        auto const contextualDirectionCalibration = [&](rt::PhaseContextualPairDirection direction,
                                                                        size_t minimumObservations) {
                            rt::PhaseContextualPdTelemetry const& telemetry
                                = runtimeCostTracker->contextualDirectionTelemetry(direction);
                            rt::PhaseContextualCompletionTelemetry const& completion
                                = runtimeCostTracker->contextualCompletionDirectionTelemetry(direction);
                            rt::PhaseContextualCompletionAuthorityEvidence const authority
                                = runtimeCostTracker->contextualCompletionAuthorityEvidence(direction);
                            // Candidate generation may predict a residual or
                            // reverse direction that never becomes a legal
                            // dispatched action.  Such a direction remains on
                            // scalar fallback and must not prevent a bounded
                            // calibration epoch from terminating.
                            bool const required = telemetry.observations > 0U;
                            bool const ready = required && telemetry.observations >= minimumObservations;
                            rt::PhaseContextualCompletionCalibrationProgress const progress
                                = runtimeCostTracker->contextualCompletionCalibrationProgress(direction);
                            bool const completionRequired
                                = runtimeCostTracker->contextualCompletionAuthorityEnabled() && required;
                            bool const completionComplete = completionRequired
                                && runtimeCostTracker->contextualCompletionAuthorityEvidenceComplete(direction);
                            contextualRequiredDirections += required ? 1U : 0U;
                            contextualReadyDirections += ready ? 1U : 0U;
                            completionRequiredDirections += completionRequired ? 1U : 0U;
                            completionCompleteDirections += completionComplete ? 1U : 0U;
                            return nlohmann::json{{"direction", rt::phaseContextualPairDirectionName(direction)},
                                {"predictions", telemetry.predictions}, {"observations", telemetry.observations},
                                {"minimum_observations", minimumObservations}, {"required", required}, {"ready", ready},
                                {"completion_ready_observations", completion.readyCalibrationObservations},
                                {"completion_conformal_observations", completion.conformalCalibrationObservations},
                                {"completion_conformal_false_safe", completion.conformalFalseSafeObservations},
                                {"completion_authority_window_observations", authority.observations},
                                {"completion_authority_minimum_observations",
                                    runtimeCostConfig.completionCalibration.authorityMinimumObservations},
                                {"completion_authority_blend_weight",
                                    runtimeCostConfig.completionCalibration.authorityBlendWeight},
                                {"completion_authority_incumbent_covered", authority.incumbentIntervalCovered},
                                {"completion_authority_newcomer_covered", authority.newcomerIntervalCovered},
                                {"completion_authority_predicted_safe", authority.predictedSafeObservations},
                                {"completion_authority_false_safe", authority.falseSafeObservations},
                                {"completion_authority_completion_absolute_error_us",
                                    authority.completionAbsoluteErrorUs},
                                {"completion_authority_reference_absolute_error_us",
                                    authority.referenceAbsoluteErrorUs},
                                {"completion_authority_empirical_blend_weight",
                                    runtimeCostTracker->contextualCompletionAuthorityBlendWeight(direction)},
                                {"completion_authority_incumbent_completion_absolute_error_us",
                                    authority.incumbentCompletionAbsoluteErrorUs},
                                {"completion_authority_incumbent_reference_absolute_error_us",
                                    authority.incumbentReferenceAbsoluteErrorUs},
                                {"completion_authority_newcomer_completion_absolute_error_us",
                                    authority.newcomerCompletionAbsoluteErrorUs},
                                {"completion_authority_newcomer_reference_absolute_error_us",
                                    authority.newcomerReferenceAbsoluteErrorUs},
                                {"completion_authority_incumbent_blend_weight",
                                    runtimeCostTracker->contextualCompletionAuthorityComponentBlendWeight(
                                        direction, true)},
                                {"completion_authority_newcomer_blend_weight",
                                    runtimeCostTracker->contextualCompletionAuthorityComponentBlendWeight(
                                        direction, false)},
                                {"completion_authority_promotions", authority.promotions},
                                {"completion_authority_demotions", authority.demotions},
                                {"completion_authority_validated", authority.validated},
                                {"completion_authority_evidence_ready",
                                    runtimeCostTracker->contextualCompletionAuthorityEvidenceReady(direction)},
                                {"completion_calibration_stage",
                                    rt::phaseContextualCompletionCalibrationStageName(progress.stage)},
                                {"completion_candidate_seen", telemetry.predictions > 0U},
                                {"completion_posterior_observations", progress.posteriorObservations},
                                {"completion_posterior_minimum_observations", progress.posteriorMinimumObservations},
                                {"completion_uncertainty_observations", progress.uncertaintyObservations},
                                {"completion_uncertainty_minimum_observations",
                                    progress.uncertaintyMinimumObservations},
                                {"completion_authority_evidence_complete", completionComplete}};
                        };
                        auto const contextualFamilyCalibration = [&](rt::PhaseContextualPairKind kind,
                                                                     rt::PhaseContextualPairDirection first,
                                                                     rt::PhaseContextualPairDirection second) {
                            rt::PhaseContextualPdModelConfig const& config
                                = runtimeCostTracker->contextualPairConfig(kind);
                            nlohmann::json directions = nlohmann::json::array();
                            size_t const requiredBefore = contextualRequiredDirections;
                            size_t const readyBefore = contextualReadyDirections;
                            if (config.mode != rt::PhaseContextualPdMode::kDisabled)
                            {
                                directions.push_back(contextualDirectionCalibration(first, config.minimumObservations));
                                directions.push_back(
                                    contextualDirectionCalibration(second, config.minimumObservations));
                            }
                            size_t const requiredDirections = contextualRequiredDirections - requiredBefore;
                            size_t const readyDirections = contextualReadyDirections - readyBefore;
                            return nlohmann::json{{"enabled", config.mode != rt::PhaseContextualPdMode::kDisabled},
                                {"minimum_observations", config.minimumObservations},
                                {"required_directions", requiredDirections}, {"ready_directions", readyDirections},
                                {"converged", requiredDirections > 0U && readyDirections == requiredDirections},
                                {"directions", std::move(directions)}};
                        };
                        nlohmann::json contextualCalibration
                            = {{"prefill_decode",
                                   contextualFamilyCalibration(rt::PhaseContextualPairKind::kPrefillDecode,
                                       rt::PhaseContextualPairDirection::kPrefillToDecode,
                                       rt::PhaseContextualPairDirection::kDecodeToPrefill)},
                                {"encoder_prefill",
                                    contextualFamilyCalibration(rt::PhaseContextualPairKind::kEncoderPrefill,
                                        rt::PhaseContextualPairDirection::kEncoderToPrefill,
                                        rt::PhaseContextualPairDirection::kPrefillToEncoder)},
                                {"encoder_decode",
                                    contextualFamilyCalibration(rt::PhaseContextualPairKind::kEncoderDecode,
                                        rt::PhaseContextualPairDirection::kEncoderToDecode,
                                        rt::PhaseContextualPairDirection::kDecodeToEncoder)}};
                        bool const exactCostCalibrationConverged
                            = requiredCostKeys > 0U && calibratedCostKeys == requiredCostKeys;
                        bool const contextualPolicyCalibrationConverged = contextualRequiredDirections > 0U
                            && contextualReadyDirections == contextualRequiredDirections;
                        bool const completionPolicyCalibrationConverged = completionRequiredDirections > 0U
                            && completionCompleteDirections == completionRequiredDirections;
                        bool const calibrationConverged = runtimeCostTracker->contextualCompletionAuthorityEnabled()
                            ? contextualPolicyCalibrationConverged && completionPolicyCalibrationConverged
                            : exactCostCalibrationConverged || contextualPolicyCalibrationConverged;
                        bool const changesCalibration = input.kind != PhaseIpcKind::kCalibrationStatus;
                        if (changesCalibration)
                        {
                            semanticCoordinator.scheduler().setGlobalWarmupProbeMode(active);
                            if (ipcThreePhase != nullptr)
                            {
                                ipcThreePhase->setGlobalWarmupProbeMode(active);
                            }
                        }
                        if (input.kind == PhaseIpcKind::kCalibrationEnd)
                        {
                            CUDA_CHECK(cudaStreamSynchronize(prefillStream));
                            CUDA_CHECK(cudaStreamSynchronize(decodeStream));
                            if (ipcEncoderStream != nullptr)
                            {
                                CUDA_CHECK(cudaStreamSynchronize(ipcEncoderStream));
                            }
                            CUDA_CHECK(cudaStreamSynchronize(copyStream));
                            CUDA_CHECK(cudaStreamSynchronize(setupStream));
                            if (activityTimeline != nullptr)
                            {
                                activityTimeline->drain();
                                ELLM_CHECK(activityTimeline->pendingCount() == 0U,
                                    "Calibration ended with an outstanding E/P/D/Copy activity interval");
                            }
                            semanticCoordinator.scheduler().resetSchedulingHistory();
                            // All physical observations are now complete and
                            // immutable. Retain validation evidence while
                            // starting serving-policy telemetry at zero.
                            if (policyWarmupMode == PhasePolicyWarmupMode::kGraphOnly
                                || policyWarmupMode == PhasePolicyWarmupMode::kZeroStart)
                            {
                                semanticCoordinator.scheduler().resetExecutionCostHistory();
                                semanticCoordinator.scheduler().resetPolicyPosterior();
                            }
                            ++measurementEpoch;
                            if (ipcThreePhase != nullptr)
                            {
                                ipcThreePhase->resetGlobalDecisionCostTelemetry();
                            }
                            emittedMetrics = semanticCoordinator.metrics().size();
                            phaseTimelineEvents.clear();
                            encoderBatchMetrics.clear();
                            formationEpisodes.clear();
                            if (activityTimeline != nullptr)
                            {
                                // Generic/trace-derived HTTP calibration uses
                                // the real phase streams. Start the measured
                                // activity window only after that work drains.
                                activityTimeline->reset(setupStream);
                            }
                            emitRecord("PHASE_EPOCH\t", {{"epoch", measurementEpoch}, {"kind", "measurement"}});
                        }
                        char const* calibrationAction = input.kind == PhaseIpcKind::kCalibrationBegin ? "begin"
                            : input.kind == PhaseIpcKind::kCalibrationEnd                             ? "end"
                                                                                                      : "status";
                        emitEvent({{"type", "control"}, {"request_index", requestId},
                            {"calibration", calibrationAction}, {"epoch", measurementEpoch},
                            {"policy_warmup_mode", phasePolicyWarmupModeName(policyWarmupMode)},
                            {"encoder_prefill_probes", calibrationMetrics.globalEncoderPrefillSelections},
                            {"residual_augmentation_opportunities",
                                calibrationMetrics.globalResidualAugmentationOpportunities},
                            {"residual_encoder_prefill_selections",
                                calibrationMetrics.globalResidualEncoderPrefillSelections},
                            {"residual_encoder_decode_selections",
                                calibrationMetrics.globalResidualEncoderDecodeSelections},
                            {"residual_unknown_cost_rejects",
                                calibrationMetrics.globalResidualAugmentationUnknownCostRejects},
                            {"encoder_decode_probes", calibrationMetrics.globalEncoderDecodeSelections},
                            {"encoder_safe_probes", calibrationMetrics.globalSafeProbes},
                            {"warmup_decisions", calibrationMetrics.globalWarmupDecisions},
                            {"warmup_prefill_candidates", calibrationMetrics.globalWarmupPrefillCandidates},
                            {"warmup_decode_candidates", calibrationMetrics.globalWarmupDecodeCandidates},
                            {"prefill_decode_safe_probes", calibrationQueueMetrics.globalSafeProbeCount},
                            {"calibration_cost_keys", std::move(calibrationCostKeys)},
                            {"calibrated_cost_keys", calibratedCostKeys}, {"required_cost_keys", requiredCostKeys},
                            {"calibration_target_keys", calibrationCosts.size()},
                            {"exact_cost_calibration_converged", exactCostCalibrationConverged},
                            {"contextual_policy_calibration", std::move(contextualCalibration)},
                            {"contextual_required_directions", contextualRequiredDirections},
                            {"contextual_ready_directions", contextualReadyDirections},
                            {"contextual_policy_calibration_converged", contextualPolicyCalibrationConverged},
                            {"completion_required_directions", completionRequiredDirections},
                            {"completion_complete_directions", completionCompleteDirections},
                            {"completion_policy_calibration_converged", completionPolicyCalibrationConverged},
                            {"calibration_converged", calibrationConverged}});
                        ++ingestedLines;
                        continue;
                    }
                    if (input.kind == PhaseIpcKind::kCancel)
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
                if (!collectPhaseDispatchMetrics)
                {
                    emittedMetrics = semanticCoordinator.metrics().size();
                }
                while (collectPhaseDispatchMetrics && emittedMetrics < semanticCoordinator.metrics().size())
                {
                    madeProgress = true;
                    rt::PhaseDispatchMetrics const& metrics = semanticCoordinator.metrics()[emittedMetrics++];
                    auto const prefillGraphs = semanticCoordinator.prefillGraphCacheStats();
                    auto const decodeGraphs = semanticCoordinator.decodeGraphCacheStats();
                    rt::PhaseThreeCoordinatorMetrics const visionMetrics
                        = ipcThreePhase != nullptr ? ipcThreePhase->metrics() : rt::PhaseThreeCoordinatorMetrics{};
                    rt::PhaseContextualPdTelemetry const contextualPdCalibration
                        = runtimeCostTracker->contextualPairTelemetry(rt::PhaseContextualPairKind::kPrefillDecode);
                    rt::PhaseContextualPdTelemetry const contextualEpCalibration
                        = runtimeCostTracker->contextualPairTelemetry(rt::PhaseContextualPairKind::kEncoderPrefill);
                    rt::PhaseContextualPdTelemetry const contextualEdCalibration
                        = runtimeCostTracker->contextualPairTelemetry(rt::PhaseContextualPairKind::kEncoderDecode);
                    rt::PhaseVisionMemoryStats const visionMemory
                        = ipcVisionAdapter != nullptr ? ipcVisionAdapter->memoryStats() : rt::PhaseVisionMemoryStats{};
                    char const* visionCopyPath = visionMemory.deviceCopyOperations > 0U ? "explicit_d2d"
                        : visionMemory.directOutputBatches > 0U                         ? "direct_output"
                                                                                        : "not_issued";
                    auto const directionObservations = [&](rt::PhaseContextualPairDirection direction) {
                        return runtimeCostTracker->contextualDirectionTelemetry(direction).observations;
                    };
                    auto const completionTelemetryJson = [](rt::PhaseContextualCompletionTelemetry const& telemetry) {
                        return nlohmann::json{{"predictions", telemetry.predictions},
                            {"observations", telemetry.observations},
                            {"ready_calibration_observations", telemetry.readyCalibrationObservations},
                            {"incumbent_interval_covered", telemetry.incumbentIntervalCovered},
                            {"newcomer_interval_covered", telemetry.newcomerIntervalCovered},
                            {"ready_incumbent_interval_covered", telemetry.readyIncumbentIntervalCovered},
                            {"ready_newcomer_interval_covered", telemetry.readyNewcomerIntervalCovered},
                            {"conformal_calibration_observations", telemetry.conformalCalibrationObservations},
                            {"conformal_incumbent_interval_covered", telemetry.conformalIncumbentIntervalCovered},
                            {"conformal_newcomer_interval_covered", telemetry.conformalNewcomerIntervalCovered},
                            {"conformal_predicted_safe", telemetry.conformalPredictedSafeObservations},
                            {"conformal_false_safe", telemetry.conformalFalseSafeObservations},
                            {"predicted_safe", telemetry.predictedSafeObservations},
                            {"false_safe", telemetry.falseSafeObservations},
                            {"incumbent_absolute_error_us", telemetry.incumbentAbsoluteErrorUs},
                            {"incumbent_squared_error_us", telemetry.incumbentSquaredErrorUs},
                            {"newcomer_absolute_error_us", telemetry.newcomerAbsoluteErrorUs},
                            {"newcomer_squared_error_us", telemetry.newcomerSquaredErrorUs},
                            {"ready_incumbent_absolute_error_us", telemetry.readyIncumbentAbsoluteErrorUs},
                            {"ready_incumbent_squared_error_us", telemetry.readyIncumbentSquaredErrorUs},
                            {"ready_newcomer_absolute_error_us", telemetry.readyNewcomerAbsoluteErrorUs},
                            {"ready_newcomer_squared_error_us", telemetry.readyNewcomerSquaredErrorUs}};
                    };
                    nlohmann::json completionCalibration = nlohmann::json::object();
                    for (rt::PhaseContextualPairDirection const direction :
                        {rt::PhaseContextualPairDirection::kPrefillToDecode,
                            rt::PhaseContextualPairDirection::kDecodeToPrefill,
                            rt::PhaseContextualPairDirection::kEncoderToPrefill,
                            rt::PhaseContextualPairDirection::kPrefillToEncoder,
                            rt::PhaseContextualPairDirection::kEncoderToDecode,
                            rt::PhaseContextualPairDirection::kDecodeToEncoder})
                    {
                        rt::PhaseContextualCompletionTelemetry const& telemetry
                            = runtimeCostTracker->contextualCompletionDirectionTelemetry(direction);
                        completionCalibration[rt::phaseContextualPairDirectionName(direction)]
                            = completionTelemetryJson(telemetry);
                    }
                    nlohmann::json completionPairCalibration
                        = {{"prefill_decode",
                               completionTelemetryJson(runtimeCostTracker->contextualCompletionPairTelemetry(
                                   rt::PhaseContextualPairKind::kPrefillDecode))},
                            {"encoder_prefill",
                                completionTelemetryJson(runtimeCostTracker->contextualCompletionPairTelemetry(
                                    rt::PhaseContextualPairKind::kEncoderPrefill))},
                            {"encoder_decode",
                                completionTelemetryJson(runtimeCostTracker->contextualCompletionPairTelemetry(
                                    rt::PhaseContextualPairKind::kEncoderDecode))}};
                    auto const conformalCalibrationJson = [&](rt::PhaseContextualPairKind kind) {
                        rt::PhaseContextualCompletionCalibrationEstimate const estimate
                            = runtimeCostTracker->contextualCompletionCalibration(kind);
                        return nlohmann::json{{"scale", estimate.scale}, {"observations", estimate.observations},
                            {"ready", estimate.ready}};
                    };
                    nlohmann::json completionConformalCalibration
                        = {{"prefill_decode", conformalCalibrationJson(rt::PhaseContextualPairKind::kPrefillDecode)},
                            {"encoder_prefill", conformalCalibrationJson(rt::PhaseContextualPairKind::kEncoderPrefill)},
                            {"encoder_decode", conformalCalibrationJson(rt::PhaseContextualPairKind::kEncoderDecode)}};
                    nlohmann::json const metricEvent{{"dispatch_index", metrics.dispatchIndex},
                        {"measurement_epoch", measurementEpoch},
                        {"policy_warmup_mode", phasePolicyWarmupModeName(policyWarmupMode)},
                        {"kind", static_cast<int32_t>(metrics.kind)}, {"prefill_batch", metrics.prefillBatchSize},
                        {"host_scheduler_decision_us", metrics.hostSchedulerDecisionUs},
                        {"host_dispatch_start_us", static_cast<double>(metrics.hostDispatchStartNs) / 1000.0},
                        {"host_submission_end_us", static_cast<double>(metrics.hostSubmissionEndNs) / 1000.0},
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
                        {"prefill_completion_ms", metrics.prefillCompletionMs},
                        {"decode_context_tokens", metrics.decodeContextTokens},
                        {"decode_context_max", metrics.plannedDecodeMaxContextLength},
                        {"decode_replacement_rows", metrics.predictedDecodeReplacementRows},
                        {"external_encoder_active", metrics.externalEncoderActive},
                        {"concurrent_prefill_active", metrics.concurrentPrefillActive},
                        {"predicted_decode_drain_gpu_ms", metrics.predictedDecodeDrainGpuMs},
                        {"predicted_decode_drain_turns", metrics.predictedDecodeDrainTurns},
                        {"decode_cohort_size", metrics.decodeCohortSize}, {"decode_gpu_ms", metrics.decodeGpuMs},
                        {"decode_completion_ms", metrics.decodeCompletionMs},
                        {"makespan_gpu_ms", metrics.makespanGpuMs}, {"overlap_ratio", metrics.overlapRatio},
                        {"vision_copy_path", visionCopyPath},
                        {"vision_direct_output_batches", visionMemory.directOutputBatches},
                        {"vision_direct_output_bytes", visionMemory.directOutputBytes},
                        {"vision_d2d_operations", visionMemory.deviceCopyOperations},
                        {"vision_d2d_bytes", visionMemory.deviceCopyBytes},
                        {"memory_drain_preference", rt::phaseDrainPreferenceName(metrics.drainPreference)},
                        {"memory_drain_preference_applied", metrics.drainPreferenceApplied},
                        {"global_decision_evaluated", metrics.globalDecisionEvaluated},
                        {"global_decision_applied", metrics.globalDecisionApplied},
                        {"global_safe_probe", metrics.globalSafeProbe},
                        {"global_action", rt::phaseGlobalActionKindName(metrics.globalSelectedAction.kind)},
                        {"global_execution_variant", rt::phaseExecutionVariantName(metrics.globalExecutionVariant)},
                        {"global_selected_residual_anchor",
                            rt::phaseGlobalResidualAnchorName(metrics.globalSelectedAction.residualAnchor)},
                        {"global_observed_residual_anchor",
                            rt::phaseGlobalResidualAnchorName(metrics.globalObservedResidualAnchor)},
                        {"global_decision_reason", static_cast<int32_t>(metrics.globalDecisionReason)},
                        {"global_action_fidelity_violations",
                            semanticCoordinator.scheduler().telemetry().globalActionFidelityViolationCount},
                        {"global_predicted_violation_us", metrics.globalPredictedViolationUs},
                        {"global_service_compression", metrics.globalServiceCompression},
                        {"global_reference_work_ms", metrics.globalReferenceWorkMs},
                        {"contextual_pd_feature_valid", metrics.contextualPdFeatureValid},
                        {"contextual_pd_exploration", metrics.contextualPdExploration},
                        {"contextual_pd_mean", metrics.contextualPdMean},
                        {"contextual_pd_uncertainty", metrics.contextualPdUncertainty},
                        {"contextual_pd_lcb", metrics.contextualPdLowerConfidenceBound},
                        {"contextual_completion_incumbent_reference_us",
                            metrics.contextualCompletionIncumbentReferenceUs},
                        {"contextual_completion_newcomer_reference_us",
                            metrics.contextualCompletionNewcomerReferenceUs},
                        {"contextual_completion", std::move(completionCalibration)},
                        {"contextual_completion_pair", std::move(completionPairCalibration)},
                        {"contextual_completion_conformal", std::move(completionConformalCalibration)},
                        {"global_decisions", semanticCoordinator.scheduler().telemetry().globalDecisionCount},
                        {"global_active_decisions",
                            semanticCoordinator.scheduler().telemetry().globalActiveDecisionCount},
                        {"global_shadow_disagreements",
                            semanticCoordinator.scheduler().telemetry().globalShadowDisagreementCount},
                        {"global_overlap_opportunities",
                            semanticCoordinator.scheduler().telemetry().globalOverlapOpportunityCount},
                        {"global_overlap_known_costs",
                            semanticCoordinator.scheduler().telemetry().globalOverlapKnownCostCount},
                        {"global_overlap_covering_costs",
                            semanticCoordinator.scheduler().telemetry().globalOverlapCoveringCostCount},
                        {"global_overlap_no_samples",
                            semanticCoordinator.scheduler().telemetry().globalOverlapNoSampleCount},
                        {"global_overlap_insufficient_samples",
                            semanticCoordinator.scheduler().telemetry().globalOverlapInsufficientSampleCount},
                        {"global_overlap_unprofitable",
                            semanticCoordinator.scheduler().telemetry().globalOverlapUnprofitableCount},
                        {"global_overlap_safe_probe_eligible",
                            semanticCoordinator.scheduler().telemetry().globalOverlapSafeProbeEligibleCount},
                        {"global_overlap_probe_disabled",
                            semanticCoordinator.scheduler().telemetry().globalOverlapProbeDisabledCount},
                        {"global_overlap_probe_interval_blocked",
                            semanticCoordinator.scheduler().telemetry().globalOverlapProbeIntervalBlockedCount},
                        {"global_overlap_probe_slack_blocked",
                            semanticCoordinator.scheduler().telemetry().globalOverlapProbeSlackBlockedCount},
                        {"global_overlap_selections",
                            semanticCoordinator.scheduler().telemetry().globalOverlapSelectionCount},
                        {"contextual_pd_predictions",
                            semanticCoordinator.scheduler().telemetry().contextualPdPredictionCount},
                        {"contextual_pd_ready", semanticCoordinator.scheduler().telemetry().contextualPdReadyCount},
                        {"contextual_pd_shadow_disagreements",
                            semanticCoordinator.scheduler().telemetry().contextualPdShadowDisagreementCount},
                        {"contextual_pd_observations",
                            semanticCoordinator.scheduler().telemetry().contextualPdObservationCount},
                        {"contextual_pd_rejected_observations",
                            semanticCoordinator.scheduler().telemetry().contextualPdRejectedObservationCount},
                        {"contextual_pd_positive_selections",
                            semanticCoordinator.scheduler().telemetry().contextualPdPositiveSelectionCount},
                        {"contextual_pd_negative_selections",
                            semanticCoordinator.scheduler().telemetry().contextualPdNegativeSelectionCount},
                        {"contextual_pd_explorations",
                            semanticCoordinator.scheduler().telemetry().contextualPdExplorationCount},
                        {"contextual_pd_last_reward",
                            semanticCoordinator.scheduler().telemetry().contextualPdLastReward},
                        {"contextual_pd_last_mean", semanticCoordinator.scheduler().telemetry().contextualPdLastMean},
                        {"contextual_pd_last_uncertainty",
                            semanticCoordinator.scheduler().telemetry().contextualPdLastUncertainty},
                        {"contextual_pd_last_lcb",
                            semanticCoordinator.scheduler().telemetry().contextualPdLastLowerConfidenceBound},
                        {"contextual_pd_calibration_observations", contextualPdCalibration.calibrationObservations},
                        {"contextual_pd_ready_calibration_observations",
                            contextualPdCalibration.readyCalibrationObservations},
                        {"contextual_pd_interval_covered", contextualPdCalibration.confidenceIntervalCovered},
                        {"contextual_pd_ready_interval_covered",
                            contextualPdCalibration.readyConfidenceIntervalCovered},
                        {"contextual_pd_predicted_safe", contextualPdCalibration.predictedSafeObservations},
                        {"contextual_pd_false_safe", contextualPdCalibration.falseSafeObservations},
                        {"contextual_pd_absolute_error_sum", contextualPdCalibration.absoluteErrorSum},
                        {"contextual_pd_squared_error_sum", contextualPdCalibration.squaredErrorSum},
                        {"contextual_pd_ready_absolute_error_sum", contextualPdCalibration.readyAbsoluteErrorSum},
                        {"contextual_pd_ready_squared_error_sum", contextualPdCalibration.readySquaredErrorSum},
                        {"contextual_prefill_to_decode_observations",
                            directionObservations(rt::PhaseContextualPairDirection::kPrefillToDecode)},
                        {"contextual_decode_to_prefill_observations",
                            directionObservations(rt::PhaseContextualPairDirection::kDecodeToPrefill)},
                        {"contextual_ep_predictions", visionMetrics.contextualEpPredictions},
                        {"contextual_ep_ready", visionMetrics.contextualEpReady},
                        {"contextual_ep_shadow_disagreements", visionMetrics.contextualEpShadowDisagreements},
                        {"contextual_ep_observations", visionMetrics.contextualEpObservations},
                        {"contextual_ep_rejected_observations", visionMetrics.contextualEpRejectedObservations},
                        {"contextual_ep_positive_selections", visionMetrics.contextualEpPositiveSelections},
                        {"contextual_ep_negative_selections", visionMetrics.contextualEpNegativeSelections},
                        {"contextual_ep_explorations", visionMetrics.contextualEpExplorations},
                        {"contextual_ep_last_reward", visionMetrics.contextualEpLastReward},
                        {"contextual_ep_last_mean", visionMetrics.contextualEpLastMean},
                        {"contextual_ep_last_uncertainty", visionMetrics.contextualEpLastUncertainty},
                        {"contextual_ep_last_lcb", visionMetrics.contextualEpLastLowerConfidenceBound},
                        {"contextual_ep_calibration_observations", contextualEpCalibration.calibrationObservations},
                        {"contextual_ep_ready_calibration_observations",
                            contextualEpCalibration.readyCalibrationObservations},
                        {"contextual_ep_interval_covered", contextualEpCalibration.confidenceIntervalCovered},
                        {"contextual_ep_ready_interval_covered",
                            contextualEpCalibration.readyConfidenceIntervalCovered},
                        {"contextual_ep_predicted_safe", contextualEpCalibration.predictedSafeObservations},
                        {"contextual_ep_false_safe", contextualEpCalibration.falseSafeObservations},
                        {"contextual_ep_absolute_error_sum", contextualEpCalibration.absoluteErrorSum},
                        {"contextual_ep_squared_error_sum", contextualEpCalibration.squaredErrorSum},
                        {"contextual_ep_ready_absolute_error_sum", contextualEpCalibration.readyAbsoluteErrorSum},
                        {"contextual_ep_ready_squared_error_sum", contextualEpCalibration.readySquaredErrorSum},
                        {"contextual_encoder_to_prefill_observations",
                            directionObservations(rt::PhaseContextualPairDirection::kEncoderToPrefill)},
                        {"contextual_prefill_to_encoder_observations",
                            directionObservations(rt::PhaseContextualPairDirection::kPrefillToEncoder)},
                        {"contextual_ed_predictions", visionMetrics.contextualEdPredictions},
                        {"contextual_ed_ready", visionMetrics.contextualEdReady},
                        {"contextual_ed_shadow_disagreements", visionMetrics.contextualEdShadowDisagreements},
                        {"contextual_ed_observations", visionMetrics.contextualEdObservations},
                        {"contextual_ed_rejected_observations", visionMetrics.contextualEdRejectedObservations},
                        {"contextual_ed_positive_selections", visionMetrics.contextualEdPositiveSelections},
                        {"contextual_ed_negative_selections", visionMetrics.contextualEdNegativeSelections},
                        {"contextual_ed_explorations", visionMetrics.contextualEdExplorations},
                        {"contextual_ed_last_reward", visionMetrics.contextualEdLastReward},
                        {"contextual_ed_last_mean", visionMetrics.contextualEdLastMean},
                        {"contextual_ed_last_uncertainty", visionMetrics.contextualEdLastUncertainty},
                        {"contextual_ed_last_lcb", visionMetrics.contextualEdLastLowerConfidenceBound},
                        {"contextual_ed_calibration_observations", contextualEdCalibration.calibrationObservations},
                        {"contextual_ed_ready_calibration_observations",
                            contextualEdCalibration.readyCalibrationObservations},
                        {"contextual_ed_interval_covered", contextualEdCalibration.confidenceIntervalCovered},
                        {"contextual_ed_ready_interval_covered",
                            contextualEdCalibration.readyConfidenceIntervalCovered},
                        {"contextual_ed_predicted_safe", contextualEdCalibration.predictedSafeObservations},
                        {"contextual_ed_false_safe", contextualEdCalibration.falseSafeObservations},
                        {"contextual_ed_absolute_error_sum", contextualEdCalibration.absoluteErrorSum},
                        {"contextual_ed_squared_error_sum", contextualEdCalibration.squaredErrorSum},
                        {"contextual_ed_ready_absolute_error_sum", contextualEdCalibration.readyAbsoluteErrorSum},
                        {"contextual_ed_ready_squared_error_sum", contextualEdCalibration.readySquaredErrorSum},
                        {"contextual_encoder_to_decode_observations",
                            directionObservations(rt::PhaseContextualPairDirection::kEncoderToDecode)},
                        {"contextual_decode_to_encoder_observations",
                            directionObservations(rt::PhaseContextualPairDirection::kDecodeToEncoder)},
                        {"global_known_overlap_priorities",
                            semanticCoordinator.scheduler().telemetry().globalKnownOverlapPriorityCount},
                        {"global_measured_unprofitable_overlap_selections",
                            semanticCoordinator.scheduler()
                                .telemetry()
                                .globalMeasuredUnprofitableOverlapSelectionCount},
                        {"global_cost_key_observations",
                            semanticCoordinator.scheduler().telemetry().globalCostKeyObservationCount},
                        {"global_cost_key_parity_violations",
                            semanticCoordinator.scheduler().telemetry().globalCostKeyParityViolationCount},
                        {"global_residual_prefill_anchor_observations",
                            semanticCoordinator.scheduler().telemetry().globalResidualPrefillAnchorObservationCount},
                        {"global_residual_decode_anchor_observations",
                            semanticCoordinator.scheduler().telemetry().globalResidualDecodeAnchorObservationCount},
                        {"global_experimental_overlap_opportunities",
                            semanticCoordinator.scheduler().telemetry().globalExperimentalOverlapOpportunityCount},
                        {"global_experimental_overlap_selections",
                            semanticCoordinator.scheduler().telemetry().globalExperimentalOverlapSelectionCount},
                        {"global_prefill_formation_opportunities",
                            semanticCoordinator.scheduler().telemetry().globalPrefillFormationOpportunityCount},
                        {"global_prefill_formation_decode_selections",
                            semanticCoordinator.scheduler().telemetry().globalPrefillFormationDecodeSelectionCount},
                        {"global_prefill_formation_producer_snapshots",
                            semanticCoordinator.scheduler().telemetry().globalPrefillFormationProducerSnapshotCount},
                        {"global_prefill_formation_combined_cost_hits",
                            semanticCoordinator.scheduler().telemetry().globalPrefillFormationCombinedCostHitCount},
                        {"global_prefill_formation_residual_cost_hits",
                            semanticCoordinator.scheduler().telemetry().globalPrefillFormationResidualCostHitCount},
                        {"global_prefill_formation_max_pending_rows",
                            semanticCoordinator.scheduler().telemetry().globalPrefillFormationMaxPendingRows},
                        {"global_decode_formation_snapshots",
                            semanticCoordinator.scheduler().telemetry().globalDecodeFormationSnapshotCount},
                        {"global_decode_formation_opportunities",
                            semanticCoordinator.scheduler().telemetry().globalDecodeFormationOpportunityCount},
                        {"global_decode_formation_combined_cost_hits",
                            semanticCoordinator.scheduler().telemetry().globalDecodeFormationCombinedCostHitCount},
                        {"global_decode_formation_produced_cost_hits",
                            semanticCoordinator.scheduler().telemetry().globalDecodeFormationProducedCostHitCount},
                        {"global_decode_formation_prefill_selections",
                            semanticCoordinator.scheduler().telemetry().globalDecodeFormationPrefillSelectionCount},
                        {"global_decode_formation_decode_selections",
                            semanticCoordinator.scheduler().telemetry().globalDecodeFormationDecodeSelectionCount},
                        {"global_decode_formation_overlap_selections",
                            semanticCoordinator.scheduler().telemetry().globalDecodeFormationOverlapSelectionCount},
                        {"global_decode_formation_max_produced_rows",
                            semanticCoordinator.scheduler().telemetry().globalDecodeFormationMaxProducedRows},
                        {"global_wait_decisions", semanticCoordinator.scheduler().telemetry().globalWaitDecisionCount},
                        {"global_wait_selected", semanticCoordinator.scheduler().telemetry().globalWaitSelectedCount},
                        {"global_wait_candidates",
                            semanticCoordinator.scheduler().telemetry().globalWaitCandidateCount},
                        {"global_wait_current_rows", semanticCoordinator.scheduler().telemetry().globalWaitCurrentRows},
                        {"global_wait_future_rows", semanticCoordinator.scheduler().telemetry().globalWaitFutureRows},
                        {"global_wait_future_first_batch_rows",
                            semanticCoordinator.scheduler().telemetry().globalWaitFutureFirstBatchRows},
                        {"global_wait_now_residual_rows",
                            semanticCoordinator.scheduler().telemetry().globalWaitNowResidualRows},
                        {"global_wait_now_drain_turns",
                            semanticCoordinator.scheduler().telemetry().globalWaitNowDrainTurns},
                        {"global_wait_future_drain_turns",
                            semanticCoordinator.scheduler().telemetry().globalWaitFutureDrainTurns},
                        {"global_wait_graph_bucket", semanticCoordinator.scheduler().telemetry().globalWaitGraphBucket},
                        {"global_wait_event_id", semanticCoordinator.scheduler().telemetry().globalWaitEventId},
                        {"global_wait_request_ids", semanticCoordinator.scheduler().telemetry().globalWaitRequestIds},
                        {"global_wait_now_horizon_us",
                            semanticCoordinator.scheduler().telemetry().globalWaitNowHorizonUs},
                        {"global_wait_future_horizon_us",
                            semanticCoordinator.scheduler().telemetry().globalWaitFutureHorizonUs},
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
                        {"runtime_decode_cost_samples",
                            semanticCoordinator.scheduler().telemetry().runtimeDecodeCostSampleCount},
                        {"runtime_decode_cost_buckets",
                            semanticCoordinator.scheduler().telemetry().runtimeDecodeCostBucketCount},
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
                        {"vision_global_residual_augmentation_opportunities",
                            visionMetrics.globalResidualAugmentationOpportunities},
                        {"vision_global_residual_encoder_prefill_selections",
                            visionMetrics.globalResidualEncoderPrefillSelections},
                        {"vision_global_residual_encoder_decode_selections",
                            visionMetrics.globalResidualEncoderDecodeSelections},
                        {"vision_global_residual_unknown_cost_rejects",
                            visionMetrics.globalResidualAugmentationUnknownCostRejects},
                        {"vision_global_residual_prefill_decode_opportunities",
                            visionMetrics.globalResidualPrefillDecodeOpportunities},
                        {"vision_global_residual_prefill_decode_selections",
                            visionMetrics.globalResidualPrefillDecodeSelections},
                        {"vision_global_residual_prefill_decode_unknown_cost_rejects",
                            visionMetrics.globalResidualPrefillDecodeUnknownCostRejects},
                        {"vision_global_residual_prefill_anchor_opportunities",
                            visionMetrics.globalResidualPrefillAnchorOpportunities},
                        {"vision_global_residual_prefill_anchor_selections",
                            visionMetrics.globalResidualPrefillAnchorSelections},
                        {"vision_global_residual_decode_anchor_opportunities",
                            visionMetrics.globalResidualDecodeAnchorOpportunities},
                        {"vision_global_residual_decode_anchor_selections",
                            visionMetrics.globalResidualDecodeAnchorSelections},
                        {"vision_global_residual_measured_unprofitable_opportunities",
                            visionMetrics.globalResidualMeasuredUnprofitableOpportunities},
                        {"vision_global_residual_measured_unprofitable_selections",
                            visionMetrics.globalResidualMeasuredUnprofitableSelections},
                        {"vision_global_residual_covering_cost_hits", visionMetrics.globalResidualCoveringCostHits},
                        {"vision_global_experimental_residual_prefill_decode_opportunities",
                            visionMetrics.globalExperimentalResidualPrefillDecodeOpportunities},
                        {"vision_global_experimental_residual_prefill_decode_selections",
                            visionMetrics.globalExperimentalResidualPrefillDecodeSelections},
                        {"vision_global_experimental_encoder_prefill_opportunities",
                            visionMetrics.globalExperimentalEncoderPrefillOpportunities},
                        {"vision_global_experimental_encoder_prefill_selections",
                            visionMetrics.globalExperimentalEncoderPrefillSelections},
                        {"vision_global_experimental_encoder_decode_opportunities",
                            visionMetrics.globalExperimentalEncoderDecodeOpportunities},
                        {"vision_global_experimental_encoder_decode_selections",
                            visionMetrics.globalExperimentalEncoderDecodeSelections},
                        {"vision_global_encoder_decode_selections", visionMetrics.globalEncoderDecodeSelections},
                        {"vision_global_pd_selections", visionMetrics.globalPdSelections},
                        {"vision_global_safe_probes", visionMetrics.globalSafeProbes},
                        {"vision_global_overlap_opportunities", visionMetrics.globalEncoderOverlapOpportunities},
                        {"vision_global_overlap_known_costs", visionMetrics.globalEncoderOverlapKnownCosts},
                        {"vision_global_overlap_no_samples", visionMetrics.globalEncoderOverlapNoSamples},
                        {"vision_global_overlap_insufficient_samples",
                            visionMetrics.globalEncoderOverlapInsufficientSamples},
                        {"vision_global_overlap_unprofitable", visionMetrics.globalEncoderOverlapUnprofitable},
                        {"vision_global_overlap_safe_probe_eligible",
                            visionMetrics.globalEncoderOverlapSafeProbeEligible},
                        {"vision_global_overlap_probe_disabled", visionMetrics.globalEncoderOverlapProbeDisabled},
                        {"vision_global_overlap_probe_interval_blocked",
                            visionMetrics.globalEncoderOverlapProbeIntervalBlocked},
                        {"vision_global_overlap_probe_slack_blocked",
                            visionMetrics.globalEncoderOverlapProbeSlackBlocked},
                        {"vision_global_formation_lookaheads", visionMetrics.globalFormationLookaheads},
                        {"vision_global_formation_predicted_rows", visionMetrics.globalFormationPredictedRows},
                        {"vision_global_formation_selection_changes", visionMetrics.globalFormationSelectionChanges},
                        {"vision_global_formation_pd_selections", visionMetrics.globalFormationPdSelections},
                        {"vision_global_formation_overlap_selections", visionMetrics.globalFormationOverlapSelections},
                        {"vision_global_formation_h2_agreements", visionMetrics.globalFormationH2Agreements},
                        {"vision_global_formation_post_policy_overrides",
                            visionMetrics.globalFormationPostPolicyOverrides},
                        {"vision_global_formation_regret_samples", visionMetrics.globalFormationRegretSamples},
                        {"vision_global_formation_positive_regrets", visionMetrics.globalFormationPositiveRegrets},
                        {"vision_global_formation_predicted_regret_ms",
                            visionMetrics.globalFormationPredictedRegretUs / 1000.0},
                        {"vision_global_formation_max_predicted_regret_ms",
                            visionMetrics.maxGlobalFormationPredictedRegretUs / 1000.0},
                        {"vision_global_formation_last_predicted_rows", visionMetrics.lastGlobalFormationPredictedRows},
                        {"vision_global_formation_last_horizon_ms",
                            visionMetrics.lastGlobalFormationHorizonUs / 1000.0},
                        {"vision_global_formation_last_cost_gap_ms",
                            visionMetrics.lastGlobalFormationCostGapUs / 1000.0},
                        {"vision_global_formation_last_planner_us", visionMetrics.lastGlobalFormationPlannerUs},
                        {"vision_global_formation_last_snapshot_id", visionMetrics.lastGlobalFormationSnapshotId},
                        {"vision_global_formation_last_h2_action",
                            rt::phaseGlobalActionKindName(visionMetrics.lastGlobalFormationH2Action)},
                        {"vision_global_formation_last_oracle_action",
                            rt::phaseGlobalActionKindName(visionMetrics.lastGlobalFormationOracleAction)},
                        {"vision_global_formation_last_decode_violation_ms",
                            visionMetrics.lastGlobalFormationDecodeViolationUs / 1000.0},
                        {"vision_global_formation_realized_episodes_started",
                            visionMetrics.globalFormationRealizedEpisodesStarted},
                        {"vision_global_formation_realized_episodes_completed",
                            visionMetrics.globalFormationRealizedEpisodesCompleted},
                        {"vision_global_formation_realized_episodes_truncated",
                            visionMetrics.globalFormationRealizedEpisodesTruncated},
                        {"vision_global_formation_realized_decode_services",
                            visionMetrics.globalFormationRealizedDecodeServices},
                        {"vision_global_formation_realized_decode_budgets",
                            visionMetrics.globalFormationRealizedDecodeBudgets},
                        {"vision_global_formation_realized_decode_service_violations",
                            visionMetrics.globalFormationRealizedDecodeServiceViolations},
                        {"vision_global_formation_realized_decode_service_gap_ms",
                            visionMetrics.globalFormationRealizedDecodeServiceGapUs / 1000.0},
                        {"vision_global_formation_realized_decode_service_gap_max_ms",
                            visionMetrics.maxGlobalFormationRealizedDecodeServiceGapUs / 1000.0},
                        {"vision_global_formation_realized_decode_service_violation_ms",
                            visionMetrics.globalFormationRealizedDecodeServiceViolationUs / 1000.0},
                        {"vision_global_formation_realized_decode_service_violation_max_ms",
                            visionMetrics.maxGlobalFormationRealizedDecodeServiceViolationUs / 1000.0},
                        {"vision_global_formation_realized_last_episode_id",
                            visionMetrics.lastGlobalFormationRealizedEpisodeId},
                        {"vision_global_formation_realized_last_snapshot_id",
                            visionMetrics.lastGlobalFormationRealizedSnapshotId},
                        {"vision_global_formation_realized_last_selected_action",
                            rt::phaseGlobalActionKindName(visionMetrics.lastGlobalFormationRealizedSelectedAction)},
                        {"vision_global_formation_realized_last_myopic_action",
                            rt::phaseGlobalActionKindName(visionMetrics.lastGlobalFormationRealizedMyopicAction)},
                        {"vision_global_formation_realized_last_oracle_action",
                            rt::phaseGlobalActionKindName(visionMetrics.lastGlobalFormationRealizedOracleAction)},
                        {"vision_global_formation_realized_last_dispatches",
                            visionMetrics.lastGlobalFormationRealizedDispatches},
                        {"vision_global_formation_realized_last_encoder_rows",
                            visionMetrics.lastGlobalFormationRealizedEncoderRows},
                        {"vision_global_formation_realized_last_prefill_rows",
                            visionMetrics.lastGlobalFormationRealizedPrefillRows},
                        {"vision_global_formation_realized_last_decode_rows",
                            visionMetrics.lastGlobalFormationRealizedDecodeRows},
                        {"vision_global_formation_realized_last_first_decode_rows",
                            visionMetrics.lastGlobalFormationRealizedFirstDecodeRows},
                        {"vision_global_formation_realized_last_max_decode_rows",
                            visionMetrics.lastGlobalFormationRealizedMaxDecodeRows},
                        {"vision_global_formation_realized_last_decode_serviced",
                            visionMetrics.lastGlobalFormationRealizedDecodeServiced},
                        {"vision_global_formation_realized_last_decode_budget_known",
                            visionMetrics.lastGlobalFormationRealizedDecodeBudgetKnown},
                        {"vision_global_formation_realized_last_truncated",
                            visionMetrics.lastGlobalFormationRealizedTruncated},
                        {"vision_global_formation_realized_last_predicted_regret_ms",
                            visionMetrics.lastGlobalFormationRealizedPredictedRegretUs / 1000.0},
                        {"vision_global_formation_realized_last_decode_budget_ms",
                            visionMetrics.lastGlobalFormationRealizedDecodeBudgetUs / 1000.0},
                        {"vision_global_formation_realized_last_decode_service_gap_ms",
                            visionMetrics.lastGlobalFormationRealizedDecodeServiceGapUs / 1000.0},
                        {"vision_global_formation_realized_last_decode_completion_visible_ms",
                            visionMetrics.lastGlobalFormationRealizedDecodeCompletionVisibleUs / 1000.0},
                        {"vision_global_formation_realized_last_horizon_completion_visible_ms",
                            visionMetrics.lastGlobalFormationRealizedHorizonCompletionVisibleUs / 1000.0},
                        {"vision_global_formation_realized_last_decode_service_violation_ms",
                            visionMetrics.lastGlobalFormationRealizedDecodeServiceViolationUs / 1000.0},
                        {"vision_global_action_fidelity_violations", visionMetrics.globalActionFidelityViolations},
                        {"vision_global_planned_outstanding",
                            static_cast<uint8_t>(visionMetrics.globalPlannedOutstanding)},
                        {"vision_global_observed_outstanding",
                            static_cast<uint8_t>(visionMetrics.globalObservedOutstanding)},
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
                while (emitPhaseMetrics && !unifiedSchedulerEvents.empty())
                {
                    madeProgress = true;
                    rt::PhaseUnifiedEvent const event = std::move(unifiedSchedulerEvents.front());
                    unifiedSchedulerEvents.pop_front();
                    auto const workJson = [](rt::PhaseUnifiedWork const& work) {
                        return nlohmann::json{{"encoder_rows", work.encoderRows}, {"prefill_rows", work.prefillRows},
                            {"prefill_tokens", work.prefillTokens}, {"decode_rows", work.decodeRows},
                            {"decode_context_tokens", work.decodeContextTokens}};
                    };
                    nlohmann::json record{{"schema_version", event.schemaVersion},
                        {"event_kind", rt::phaseUnifiedEventKindName(event.kind)}, {"event_id", event.eventId},
                        {"run_id", schedulerRunId}, {"host_monotonic_ns", event.hostMonotonicNs}};
                    if (event.kind == rt::PhaseUnifiedEventKind::kDecision)
                    {
                        nlohmann::json inflight = nlohmann::json::array();
                        for (rt::PhaseInFlightWorkSnapshot const& work : event.inFlight.work)
                        {
                            inflight.push_back({{"phase", rt::phaseUnifiedPhaseName(work.phase)},
                                {"execution_id", work.executionId}, {"plan_id", work.planId},
                                {"action_id", work.actionId}, {"activity_correlation_id", work.activityCorrelationId},
                                {"status", rt::phaseInFlightStatusName(work.status)},
                                {"dispatch_host_ns", work.dispatchHostNs}, {"dispatch_age_us", work.dispatchAgeUs},
                                {"request_ids", work.requestIds}, {"cohort", workJson(work.work)}});
                        }
                        nlohmann::json candidates = nlohmann::json::array();
                        for (rt::PhaseUnifiedCandidateSnapshot const& candidate : event.candidates)
                        {
                            auto protectedCompletions = [](auto const& completions) {
                                nlohmann::json result = nlohmann::json::array();
                                for (rt::PhaseProtectedCompletion const& completion : completions)
                                {
                                    result.push_back({{"kind", rt::phaseProtectedKindName(completion.kind)},
                                        {"slack_us", completion.slackUs},
                                        {"predicted_completion_us", completion.predictedCompletionUs},
                                        {"uncertainty_us", completion.uncertaintyUs}});
                                }
                                return result;
                            };
                            candidates.push_back({{"action_id", candidate.actionId},
                                {"action_kind", rt::phaseGlobalActionKindName(candidate.key.kind)},
                                {"primary_batch_size", candidate.key.primaryBatchSize},
                                {"secondary_batch_size", candidate.key.secondaryBatchSize},
                                {"chunk_length", candidate.key.chunkLength},
                                {"primary_context_bucket", candidate.key.primaryContextBucket},
                                {"secondary_context_bucket", candidate.key.secondaryContextBucket},
                                {"execution_variant", rt::phaseExecutionVariantName(candidate.key.executionVariant)},
                                {"primary_work_class", candidate.key.primaryWorkClass},
                                {"action_direction",
                                    candidate.key.kind == rt::PhaseGlobalActionKind::kEncoderPrefill
                                            || candidate.key.kind == rt::PhaseGlobalActionKind::kEncoderDecode
                                            || candidate.key.kind == rt::PhaseGlobalActionKind::kPrefillDecode
                                        ? rt::phaseContextualPairDirectionName(candidate.contextualDirection)
                                        : "none"},
                                {"residual_augmentation", candidate.key.residualAugmentation},
                                {"residual_anchor", rt::phaseGlobalResidualAnchorName(candidate.key.residualAnchor)},
                                {"legal", candidate.legal}, {"request_ids", candidate.requestIds},
                                {"predicted_completion_us", nlohmann::json::array({candidate.predictedCompletionUs})},
                                {"uncertainty_us", nlohmann::json::array({candidate.uncertaintyUs})},
                                {"max_slo_violation_us", candidate.predictedSloViolationUs},
                                {"contextual_completion_valid", candidate.contextualCompletionValid},
                                {"contextual_completion_feature_v2_valid",
                                    candidate.contextualCompletionFeatureV2Valid},
                                {"contextual_completion_features", candidate.contextualCompletionFeatures},
                                {"contextual_direction",
                                    rt::phaseContextualPairDirectionName(candidate.contextualDirection)},
                                {"contextual_completion_ready", candidate.contextualCompletion.ready},
                                {"contextual_effect_valid", candidate.contextualEffectValid},
                                {"contextual_effect_ready", candidate.contextualEffect.ready()},
                                {"contextual_effect_compression_mean", candidate.contextualEffect.compression.mean},
                                {"contextual_effect_compression_uncertainty",
                                    candidate.contextualEffect.compression.uncertainty},
                                {"contextual_effect_incumbent_stretch_mean",
                                    candidate.contextualEffect.incumbentStretch.mean},
                                {"contextual_effect_incumbent_stretch_uncertainty",
                                    candidate.contextualEffect.incumbentStretch.uncertainty},
                                {"contextual_effect_order_margin_mean",
                                    candidate.contextualEffect.completionOrderMargin.mean},
                                {"contextual_effect_order_margin_uncertainty",
                                    candidate.contextualEffect.completionOrderMargin.uncertainty},
                                {"contextual_incumbent_mean_us", candidate.contextualCompletion.incumbentMeanUs},
                                {"contextual_incumbent_uncertainty_us",
                                    candidate.contextualCompletion.incumbentUncertaintyUs},
                                {"contextual_newcomer_mean_us", candidate.contextualCompletion.newcomerMeanUs},
                                {"contextual_newcomer_uncertainty_us",
                                    candidate.contextualCompletion.newcomerUncertaintyUs},
                                {"contextual_pair_observations", candidate.contextualCompletion.pairObservations},
                                {"contextual_direction_observations",
                                    candidate.contextualCompletion.directionObservations},
                                {"contextual_direction_weight", candidate.contextualCompletion.directionWeight},
                                {"contextual_uncertainty_scale", candidate.contextualCompletion.uncertaintyScale},
                                {"contextual_uncertainty_calibration_observations",
                                    candidate.contextualCompletion.uncertaintyCalibrationObservations},
                                {"contextual_uncertainty_calibrated",
                                    candidate.contextualCompletion.uncertaintyCalibrated},
                                {"contextual_incumbent_reference_us", candidate.contextualIncumbentReferenceUs},
                                {"contextual_newcomer_reference_us", candidate.contextualNewcomerReferenceUs},
                                {"completion_policy_evaluated", candidate.completionPolicyEvaluated},
                                {"completion_authority_ready", candidate.completionAuthorityReady},
                                {"completion_authority_applied", candidate.completionAuthorityApplied},
                                {"scalar_decision_cost_known", candidate.scalarDecisionCostKnown},
                                {"active_decision_cost_known", candidate.activeDecisionCostKnown},
                                {"scalar_decision_makespan_us", candidate.scalarDecisionMakespanUs},
                                {"active_decision_makespan_us", candidate.activeDecisionMakespanUs},
                                {"completion_aggregate_blend_weight", candidate.completionAggregateBlendWeight},
                                {"completion_incumbent_blend_weight", candidate.completionIncumbentBlendWeight},
                                {"completion_newcomer_blend_weight", candidate.completionNewcomerBlendWeight},
                                {"scalar_protected_completions",
                                    protectedCompletions(candidate.scalarProtectedCompletions)},
                                {"active_protected_completions",
                                    protectedCompletions(candidate.activeProtectedCompletions)}});
                            if (std::isfinite(candidate.contextualMinimumSlackUs))
                            {
                                candidates.back()["contextual_minimum_slack_us"] = candidate.contextualMinimumSlackUs;
                            }
                        }
                        record.update({{"decision_id", event.decisionId},
                            {"policy_decision_sequence", event.policyDecisionSequence},
                            {"snapshot_id", event.snapshotId}, {"snapshot_signature", event.snapshotSignature},
                            {"plan_id", event.planId}, {"strict_snapshot_signature", event.strictSnapshotSignature},
                            {"kv_ownership_signature", event.kvOwnershipSignature},
                            {"scalar_policy_state_signature", event.scalarPolicyStateSignature},
                            {"vision_lease_signature", event.visionLeaseSignature},
                            {"causal_replay_forced", event.causalReplayForced}, {"action_id", event.actionId},
                            {"incremental_action_id", event.incrementalActionId},
                            {"requested_start_skew_percent", event.requestedStartSkewPercent},
                            {"requested_action_direction",
                                rt::phaseUnifiedActionDirectionName(event.requestedDirection)},
                            {"dispatch_mode", rt::phaseUnifiedDispatchModeName(event.dispatchMode)},
                            {"action_kind", rt::phaseGlobalActionKindName(event.actionKind)},
                            {"outstanding_before_mask", static_cast<uint8_t>(event.outstandingBefore)},
                            {"planned_outstanding_mask", static_cast<uint8_t>(event.plannedOutstanding)},
                            {"ready", workJson(event.ready)}, {"selected_cohort", workJson(event.cohort)},
                            {"ready_encoder_request_ids", event.readyEncoderRequestIds},
                            {"ready_prefill_request_ids", event.readyPrefillRequestIds},
                            {"ready_prefill_token_counts", event.readyPrefillTokenCounts},
                            {"ready_decode_request_ids", event.readyDecodeRequestIds},
                            {"ready_decode_context_lengths", event.readyDecodeContextLengths},
                            {"page_pool_allocated_bundles", event.pagePoolAllocatedBundles},
                            {"page_reservation_guaranteed_bundles", event.pageReservationGuaranteedBundles},
                            {"vision_payload_bytes", event.visionPayloadBytes}, {"request_ids", event.requestIds},
                            {"inflight", std::move(inflight)}, {"candidates", std::move(candidates)},
                            {"selected_action_id", event.selectedActionId},
                            {"active_h1_selected_action_id", event.activeH1SelectedActionId},
                            {"scalar_h1_selected_action_id", event.scalarSelectedActionId},
                            {"completion_changed_h1_action",
                                event.scalarSelectedActionId > 0U && event.activeH1SelectedActionId > 0U
                                    && event.scalarSelectedActionId != event.activeH1SelectedActionId}});
                    }
                    else if (event.kind == rt::PhaseUnifiedEventKind::kDispatch)
                    {
                        record.update({{"decision_id", event.decisionId}, {"snapshot_id", event.snapshotId},
                            {"execution_id", event.executionId}, {"plan_id", event.planId},
                            {"action_id", event.actionId}, {"phase", rt::phaseUnifiedPhaseName(event.phase)},
                            {"incremental_action_id", event.incrementalActionId},
                            {"requested_start_skew_percent", event.requestedStartSkewPercent},
                            {"requested_action_direction",
                                rt::phaseUnifiedActionDirectionName(event.requestedDirection)},
                            {"action_kind", rt::phaseGlobalActionKindName(event.actionKind)},
                            {"action_direction", rt::phaseUnifiedActionDirectionName(event.direction)},
                            {"dispatch_mode", rt::phaseUnifiedDispatchModeName(event.dispatchMode)},
                            {"incumbent_dispatch_age_us", event.incumbentDispatchAgeUs},
                            {"observed_start_skew_percent", event.observedStartSkewPercent},
                            {"outstanding_before_mask", static_cast<uint8_t>(event.outstandingBefore)},
                            {"planned_outstanding_mask", static_cast<uint8_t>(event.plannedOutstanding)},
                            {"observed_outstanding_mask", static_cast<uint8_t>(event.observedOutstanding)},
                            {"cohort", workJson(event.cohort)}, {"request_ids", event.requestIds},
                            {"enqueue_host_ns", event.enqueueHostNs},
                            {"prepare_start_host_ns", event.prepareStartHostNs},
                            {"prepare_end_host_ns", event.prepareEndHostNs},
                            {"execute_start_host_ns", event.executeStartHostNs},
                            {"execute_end_host_ns", event.executeEndHostNs}, {"graph_replay", event.graphReplay},
                            {"action_fidelity", event.actionFidelity},
                            {"action_fidelity_reason",
                                rt::phaseUnifiedFidelityReasonName(event.actionFidelityReason)}});
                        if (event.incumbentExecutionId > 0U)
                        {
                            record["incumbent_phase"] = rt::phaseUnifiedPhaseName(event.incumbentPhase);
                            record["incumbent_execution_id"] = event.incumbentExecutionId;
                        }
                        if (event.newcomerExecutionId > 0U)
                        {
                            record["newcomer_phase"] = rt::phaseUnifiedPhaseName(event.newcomerPhase);
                            record["newcomer_execution_id"] = event.newcomerExecutionId;
                        }
                        if (event.injectionTargetFraction.has_value())
                        {
                            record["injection_target_fraction"] = *event.injectionTargetFraction;
                            record["injection_requested_direction"]
                                = rt::phaseUnifiedActionDirectionName(event.injectionRequestedDirection);
                            record["injection_incumbent_reference_us"] = event.injectionIncumbentReferenceUs;
                            record["injection_newcomer_reference_us"] = event.injectionNewcomerReferenceUs;
                            record["requested_injection_delay_us"] = event.requestedInjectionDelayUs;
                        }
                    }
                    else
                    {
                        record.update({{"decision_id", event.decisionId}, {"snapshot_id", event.snapshotId},
                            {"execution_id", event.executionId}, {"plan_id", event.planId},
                            {"action_id", event.actionId}, {"phase", rt::phaseUnifiedPhaseName(event.phase)},
                            {"incremental_action_id", event.incrementalActionId},
                            {"requested_start_skew_percent", event.requestedStartSkewPercent},
                            {"requested_action_direction",
                                rt::phaseUnifiedActionDirectionName(event.requestedDirection)},
                            {"action_kind", rt::phaseGlobalActionKindName(event.actionKind)},
                            {"action_direction", rt::phaseUnifiedActionDirectionName(event.direction)},
                            {"dispatch_mode", rt::phaseUnifiedDispatchModeName(event.dispatchMode)},
                            {"incumbent_dispatch_age_us", event.incumbentDispatchAgeUs},
                            {"observed_start_skew_percent", event.observedStartSkewPercent},
                            {"observed_outstanding_mask", static_cast<uint8_t>(event.observedOutstanding)},
                            {"cohort", workJson(event.cohort)}, {"request_ids", event.requestIds},
                            {"gpu_duration_us", event.gpuDurationUs},
                            {"completion_visible_host_ns", event.completionVisibleHostNs},
                            {"completion_status", "success"}, {"action_fidelity", event.actionFidelity},
                            {"action_fidelity_reason",
                                rt::phaseUnifiedFidelityReasonName(event.actionFidelityReason)}});
                        if (event.incumbentExecutionId > 0U)
                        {
                            record["incumbent_phase"] = rt::phaseUnifiedPhaseName(event.incumbentPhase);
                            record["incumbent_execution_id"] = event.incumbentExecutionId;
                        }
                        if (event.newcomerExecutionId > 0U)
                        {
                            record["newcomer_phase"] = rt::phaseUnifiedPhaseName(event.newcomerPhase);
                            record["newcomer_execution_id"] = event.newcomerExecutionId;
                        }
                        if (event.incumbentGpuCompletionUs.has_value())
                        {
                            record["incumbent_gpu_completion_us"] = *event.incumbentGpuCompletionUs;
                        }
                        if (event.newcomerGpuCompletionUs.has_value())
                        {
                            record["newcomer_gpu_completion_us"] = *event.newcomerGpuCompletionUs;
                        }
                        if (event.gpuStartUs.has_value() && event.gpuEndUs.has_value())
                        {
                            record["gpu_start_us"] = *event.gpuStartUs;
                            record["gpu_end_us"] = *event.gpuEndUs;
                        }
                    }
                    serializedRecords.push_back("PHASE_SCHEDULER_EVENT\t" + record.dump());
                }
                while (emitPhaseRequestTimeline && !phaseTimelineEvents.empty())
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
                while (emitPhaseMetrics && !formationEpisodes.empty())
                {
                    madeProgress = true;
                    rt::PhaseFormationRealizedEpisode const episode = std::move(formationEpisodes.front());
                    formationEpisodes.pop_front();
                    nlohmann::json dispatches = nlohmann::json::array();
                    for (rt::PhaseFormationRealizedDispatch const& dispatch : episode.dispatches)
                    {
                        dispatches.push_back(
                            {{"plan_id", dispatch.planId}, {"action", rt::phaseGlobalActionKindName(dispatch.action)},
                                {"encoder_rows", dispatch.work.encoderRows},
                                {"prefill_rows", dispatch.work.prefillRows}, {"decode_rows", dispatch.work.decodeRows},
                                {"dispatch_after_ms", dispatch.dispatchedAfterUs / 1000.0},
                                {"completion_visible_after_ms", dispatch.completionVisibleAfterUs / 1000.0}});
                    }
                    nlohmann::json const episodeEvent{{"episode_id", episode.episodeId},
                        {"snapshot_id", episode.snapshotId},
                        {"selected_action", rt::phaseGlobalActionKindName(episode.selectedAction)},
                        {"myopic_action", rt::phaseGlobalActionKindName(episode.myopicAction)},
                        {"oracle_action", rt::phaseGlobalActionKindName(episode.oracleAction)},
                        {"predicted_regret_ms", episode.predictedRegretUs / 1000.0},
                        {"decode_budget_ms",
                            std::isfinite(episode.decodeServiceBudgetUs) ? episode.decodeServiceBudgetUs / 1000.0
                                                                         : 0.0},
                        {"decode_budget_known", std::isfinite(episode.decodeServiceBudgetUs)},
                        {"encoder_rows", episode.encoderRows}, {"prefill_rows", episode.prefillRows},
                        {"decode_rows", episode.decodeRows}, {"first_decode_rows", episode.firstDecodeRows},
                        {"max_decode_rows", episode.maxDecodeRows}, {"decode_serviced", episode.decodeServiced},
                        {"decode_service_gap_ms", episode.decodeServiceGapUs / 1000.0},
                        {"decode_completion_visible_ms", episode.decodeCompletionVisibleUs / 1000.0},
                        {"horizon_completion_visible_ms", episode.horizonCompletionVisibleUs / 1000.0},
                        {"decode_service_violation_ms", episode.decodeServiceViolationUs / 1000.0},
                        {"truncated", episode.truncated}, {"dispatches", std::move(dispatches)}};
                    serializedRecords.push_back("PHASE_FORMATION_EPISODE\t" + episodeEvent.dump());
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
                "admission_refill_wait_periods=%zu admission_refill_deferrals=%zu "
                "adaptive_transitions=%zu throughput_mode=%s "
                "admission_limit=%zu admission_increases=%zu admission_decreases=%zu cost_limit=%zu "
                "cost_blocks=%zu tpot_budget_us=%.3f tpot_satisfiable=%s external_profile=%s "
                "external_profile_selections=%zu unsatisfiable_decisions=%zu",
                semanticServer.decodeRefillWaitCount(), semanticServer.prefillFormationWaitPeriodCount(),
                semanticServer.prefillFormationDeferralCount(), semanticServer.prefillFormationProfileSelectionCount(),
                semanticServer.prefillFormationProfileMissCount(), semanticServer.admissionRefillWaitPeriodCount(),
                semanticServer.admissionRefillDeferralCount(), semanticServer.throughputModeTransitionCount(),
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
            LOG_INFO(
                "Phase IPC policy: ingress_quantum=%zu emit_metrics=%s collect_dispatch_metrics=%s telemetry_level=%s",
                ipcIngressQuantum, emitPhaseMetrics ? "yes" : "no", collectPhaseDispatchMetrics ? "yes" : "no",
                phaseTelemetryLevel.c_str());
            LOG_INFO("Phase IPC response path: native_callback");
            LOG_INFO(
                "Phase IPC request adapter: inputs=%zu total=%.3f ms vision_canonical=%s "
                "unrestricted_out_of_order=%s "
                "out_of_order_completions=%zu max_ready_bypass=%zu",
                ipcRequestAdapterInputs, ipcRequestAdapterUs / 1000.0, allowOutOfOrderAdapterCompletion ? "no" : "yes",
                allowOutOfOrderAdapterCompletion ? "yes" : "no", ipcRequestAdapterOutOfOrderCompletions,
                ipcRequestAdapterMaxReadyBypass);
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
                rt::PhaseSchedulerTelemetry const& schedulerMetrics = semanticCoordinator.scheduler().telemetry();
                LOG_INFO("Phase global scheduler decision cost: samples=%zu mean=%.3f us p95=%.3f us max=%.3f us",
                    visionMetrics.globalHostDecisionSamples, visionMetrics.globalHostDecisionMeanUs,
                    visionMetrics.globalHostDecisionP95Us, visionMetrics.globalHostDecisionMaxUs);
                LOG_INFO(
                    "Phase global scheduler summary: decisions=%zu active=%zu overlaps=%zu known_profitable=%zu "
                    "known_unprofitable_selected=%zu safe_probes=%zu wait_decisions=%zu wait_selected=%zu "
                    "contextual_predictions=%zu contextual_ready=%zu contextual_disagreements=%zu "
                    "action_fidelity_violations=%zu",
                    schedulerMetrics.globalDecisionCount, schedulerMetrics.globalActiveDecisionCount,
                    schedulerMetrics.globalOverlapSelectionCount, schedulerMetrics.globalKnownOverlapPriorityCount,
                    schedulerMetrics.globalMeasuredUnprofitableOverlapSelectionCount,
                    schedulerMetrics.globalSafeProbeCount, schedulerMetrics.globalWaitDecisionCount,
                    schedulerMetrics.globalWaitSelectedCount, schedulerMetrics.contextualPdPredictionCount,
                    schedulerMetrics.contextualPdReadyCount, schedulerMetrics.contextualPdShadowDisagreementCount,
                    schedulerMetrics.globalActionFidelityViolationCount);
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
            if (phaseTelemetryOutput.is_open())
            {
                phaseTelemetryOutput.flush();
                phaseTelemetryOutput.close();
            }
            LOG_INFO("Phase IPC token text cache: entries=%zu hits=%zu misses=%zu", decodedTokenTextCache.size(),
                decodedTokenTextCacheHits, decodedTokenTextCacheMisses);
            LOG_INFO(
                "Phase IPC host cost: polls=%zu ingress=%.3f ms adapter_workers=%zu adapter_inputs=%zu "
                "adapter=%.3f ms "
                "poll=%.3f ms serialize=%.3f ms output_serialize=%.3f ms output_batches=%zu output_records=%zu "
                "output_bytes=%zu telemetry_bytes=%zu",
                ipcPollCalls, ipcIngressUs / 1000.0, asyncRequestAdapter ? requestAdapterWorkers : 0U,
                ipcRequestAdapterInputs, ipcRequestAdapterUs / 1000.0, ipcPollUs / 1000.0, ipcSerializationUs / 1000.0,
                outputSerializationUs / 1000.0, outputWriteBatches, outputWriteRecords, outputWriteBytes,
                telemetryWriteBytes);
            auto const prefillGraphStats = semanticCoordinator.prefillGraphCacheStats();
            auto const decodeGraphStats = semanticCoordinator.decodeGraphCacheStats();
            LOG_INFO(
                "Phase CUDA graph cache: prefill entries=%zu hits=%zu misses=%zu captures=%zu evictions=%zu; "
                "decode entries=%zu hits=%zu misses=%zu captures=%zu evictions=%zu",
                prefillGraphStats.entries, prefillGraphStats.hits, prefillGraphStats.misses, prefillGraphStats.captures,
                prefillGraphStats.evictions, decodeGraphStats.entries, decodeGraphStats.hits, decodeGraphStats.misses,
                decodeGraphStats.captures, decodeGraphStats.evictions);
            if (ipcThreePhase != nullptr)
            {
                // PhaseThree's completion wrapper owns request-lifetime cleanup.
                // Remove it before destroying the wrapper around the longer-lived server.
                semanticServer.setEventCallbacks({}, {});
            }
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
        if (activityTimeline != nullptr)
        {
            activityTimeline->drain();
            activityTimeline->writeCsv(activityPrefix);
            rt::PhaseActivitySummary const activity = activityTimeline->summary();
            double const ratioScale = activity.windowMs > 0.0 ? 100.0 / activity.windowMs : 0.0;
            LOG_INFO(
                "Phase activity: window=%.3f ms any_epd=%.2f%% all_idle=%.2f%% epd_idle=%.2f%% "
                "epd_triple=%.2f%% four_way=%.2f%% E=%.2f%% P=%.2f%% D=%.2f%% C=%.2f%%",
                activity.windowMs, activity.anyEpdMs * ratioScale, activity.allIdleMs * ratioScale,
                activity.epdIdleMs * ratioScale, activity.epdTripleMs * ratioScale, activity.fourWayMs * ratioScale,
                activity.activityMs[0] * ratioScale, activity.activityMs[1] * ratioScale,
                activity.activityMs[2] * ratioScale, activity.activityMs[3] * ratioScale);
            semanticServer.setActivityTimeline(nullptr);
        }
    }

    CUDA_CHECK(cudaStreamDestroy(setupStream));
    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
    CUDA_CHECK(cudaStreamDestroy(copyStream));
    return EXIT_SUCCESS;
}
