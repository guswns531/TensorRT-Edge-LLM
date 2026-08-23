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
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kDEFAULT_STABLE_SLOTS = 80;
constexpr size_t kDEFAULT_MAX_INFLIGHT_REQUESTS = 16U;

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
            point.at("p95_gpu_ms").get<float>(), point.at("decode_slowdown_p95_ms").get<float>()});
    }
    for (nlohmann::json const& point : root.at("overlap"))
    {
        config.overlapBatchCosts.push_back(
            {point.at("prefill_batch_size").get<int32_t>(), point.at("decode_batch_size").get<int32_t>(),
                point.at("chunk_length").get<int32_t>(), point.at("max_prefill_past_kv_length").get<int32_t>(),
                point.at("max_decode_context_length").get<int32_t>(), point.at("initial_chunk").get<bool>(),
                point.at("prefill_p95_gpu_ms").get<float>(), point.at("decode_p95_gpu_ms").get<float>(),
                point.at("makespan_p95_gpu_ms").get<float>(), point.at("decode_slowdown_p95_ms").get<float>()});
    }
    ELLM_CHECK(
        !config.decodeBatchCosts.empty() && !config.prefillBatchCosts.empty() && !config.overlapBatchCosts.empty(),
        "Phase scheduler cost model is incomplete");
    config.profile = rt::PhaseSchedulerProfile::kThroughputBalanced;
}

struct PhaseTiming
{
    float prefillMs{};
    float decodeMs{};
    float makespanMs{};
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
    CUDA_CHECK(cudaStreamCreateWithFlags(&setupStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&prefillStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&decodeStream, cudaStreamNonBlocking));

    {
        auto executor = rt::EngineExecutor::createForLLM(engineDir / "llm.engine", config);
        rt::IndependentEngineExecutorPairConfig pairConfig;
        pairConfig.setupStream = setupStream;
        pairConfig.prefillStream = prefillStream;
        pairConfig.decodeStream = decodeStream;
        auto pair = rt::IndependentEngineExecutorPair::create(std::move(executor), pairConfig);

        ELLM_CHECK(&pair->prefillExecutor().getEngine() == &pair->decodeExecutor().getEngine(),
            "Independent phase executors must share one TensorRT engine");
        ELLM_CHECK(pair->prefillExecutor().getExecutionContextIdentity()
                != pair->decodeExecutor().getExecutionContextIdentity(),
            "Independent phase executors must own different TensorRT contexts");
        ELLM_CHECK(pair->prefillContextMemory().rawPointer() != pair->decodeContextMemory().rawPointer(),
            "Independent phase executors must own different workspaces");

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

        rt::Tensor hostSemanticPrefillIds({config.maxSupportedPrefillBatchSize, config.maxPackedPrefillChunkTokens},
            rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "semantic_phase_host_prefill_ids");
        rt::Tensor deviceSemanticPrefillIds({config.maxSupportedPrefillBatchSize, config.maxPackedPrefillChunkTokens},
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
        SamplingSlotPool samplingSlotPool(maxPhaseBatch);

        auto stageTokens = [&](std::vector<rt::IndependentPhaseRequestView> const& views, rt::PipelineIO& io,
                               rt::TensorMap& map, cudaStream_t stream, bool prefill) {
            rt::OptionalInputTensor visionEmbedding;
            rt::OptionalInputTensors deepstackFeatures;
            rt::Tensor visionEmbeddingView;
            std::vector<rt::Tensor> deepstackFeatureViews;
            if (prefill && !views.empty() && views.front().visionPayload != nullptr)
            {
                ELLM_CHECK(views.size() == 1U, "Multimodal phase v1 requires an atomic single-row prefill");
                rt::PhaseVisionPayload& payload = *views.front().visionPayload;
                auto const chunkBegin = views.front().promptTokens->begin() + views.front().work.tokenOffset;
                auto const chunkEnd = chunkBegin + views.front().work.tokenCount;
                int64_t const imageOffset
                    = std::count(views.front().promptTokens->begin(), chunkBegin, config.imageTokenId);
                int64_t const imageTokens = std::count(chunkBegin, chunkEnd, config.imageTokenId);
                if (imageTokens > 0)
                {
                    auto makeFeatureView = [&](rt::Tensor& feature, std::string const& name) {
                        rt::Coords const shape = feature.getShape();
                        ELLM_CHECK(shape.getNumDims() == 2 && imageOffset + imageTokens <= shape[0],
                            "Multimodal chunk is outside its request-owned feature buffer");
                        size_t const rowBytes
                            = static_cast<size_t>(shape[1]) * rt::utils::getTypeSize(feature.getDataType());
                        auto* const data = static_cast<std::byte*>(feature.rawPointer()) + imageOffset * rowBytes;
                        return rt::Tensor(data, {shape[0] - imageOffset, shape[1]}, rt::DeviceType::kGPU,
                            feature.getDataType(), name);
                    };
                    visionEmbeddingView = makeFeatureView(payload.outputEmbedding, "phase_vision_chunk");
                    visionEmbedding = std::cref(visionEmbeddingView);
                    deepstackFeatureViews.reserve(payload.deepstackFeatures.size());
                    deepstackFeatures.reserve(payload.deepstackFeatures.size());
                    for (rt::Tensor& feature : payload.deepstackFeatures)
                    {
                        deepstackFeatureViews.push_back(makeFeatureView(feature, "phase_deepstack_chunk"));
                        deepstackFeatures.push_back(std::cref(deepstackFeatureViews.back()));
                    }
                }
                if (!payload.mropeCosSin.isEmpty())
                {
                    map.set(binding_names::kRopeCosSin, payload.mropeCosSin);
                }
            }
            int32_t totalTokens{};
            for (rt::IndependentPhaseRequestView const& view : views)
            {
                totalTokens += prefill ? view.work.tokenCount : 1;
            }
            rt::Coords const tokenShape
                = prefill ? rt::Coords{1, totalTokens} : rt::Coords{static_cast<int64_t>(views.size()), 1};
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
                    ELLM_CHECK(!view.generatedTokens->empty(), "Semantic decode request has no sampled input token");
                    destination[destinationOffset++] = view.generatedTokens->back();
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(deviceIds.rawPointer(), hostIds.rawPointer(),
                static_cast<size_t>(totalTokens) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
            embeddingPreprocessor.embed(deviceIds, visionEmbedding, std::nullopt, io, stream);
            embeddingPreprocessor.prepareDeepstack(deviceIds, deepstackFeatures, io, stream);
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
            selectAllTopK(io.outputLogits, std::nullopt, selectedIds, 1, workspace, stream);
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
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_PREFILL_BATCH"))
        {
            semanticSchedulerConfig.maxPrefillBatchSize = std::stoi(value);
        }
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_DECODE_BATCH"))
        {
            semanticSchedulerConfig.maxDecodeBatchSize = std::stoi(value);
        }
        ELLM_CHECK(semanticSchedulerConfig.maxPrefillBatchSize > 0
                && semanticSchedulerConfig.maxPrefillBatchSize <= config.maxSupportedPrefillBatchSize,
            "Semantic prefill batch cap is outside the engine profile");
        ELLM_CHECK(semanticSchedulerConfig.maxDecodeBatchSize > 0
                && semanticSchedulerConfig.maxDecodeBatchSize <= config.maxSupportedDecodeBatchSize,
            "Semantic decode batch cap is outside the engine profile");
        semanticSchedulerConfig.maxPrefillChunkTokens = 128;
        semanticSchedulerConfig.maxOverlapPrefillTokens = 128;
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_OVERLAP_PREFILL_TOKENS"))
        {
            semanticSchedulerConfig.maxOverlapPrefillTokens = std::stoi(value);
        }
        semanticSchedulerConfig.maxPrefillBatchTokens = semanticSchedulerConfig.maxPrefillBatchSize * 128;
        semanticSchedulerConfig.enableRaggedPrefillBatching = true;
        semanticSchedulerConfig.enablePackedPrefillTokenLayout = true;
        semanticSchedulerConfig.prefillCompletionBonusTokens = 128;
        semanticSchedulerConfig.enableWavefrontPrefillBatching
            = std::getenv("TRT_EDGELLM_DISABLE_WAVEFRONT_PREFILL") == nullptr;
        semanticSchedulerConfig.maxPrefillCohortSize = semanticSchedulerConfig.maxPrefillBatchSize;
        semanticSchedulerConfig.maxPrefillCohortTurns = 8;
        semanticSchedulerConfig.enableAdaptivePrefillChunking = true;
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
            semanticSchedulerConfig.maxPrefillBatchTokens = semanticSchedulerConfig.maxPrefillBatchSize * fixedChunk;
        }
        semanticSchedulerConfig.enableMetricsPolicy = std::getenv("TRT_EDGELLM_DISABLE_METRICS_POLICY") == nullptr;
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
        semanticSchedulerConfig.enableDynamicDecodeBatching
            = std::getenv("TRT_EDGELLM_DISABLE_DYNAMIC_DECODE") == nullptr;
        semanticSchedulerConfig.enableOnlineDecodeCostLearning
            = std::getenv("TRT_EDGELLM_DISABLE_ONLINE_COST") == nullptr;
        bool const asymmetricDecodeStaging = maxStableSlots > semanticSchedulerConfig.maxDecodeBatchSize
            && semanticSchedulerConfig.maxDecodeBatchSize > semanticSchedulerConfig.maxPrefillBatchSize;
        semanticSchedulerConfig.enableDecodeCohortBatching
            = asymmetricDecodeStaging && std::getenv("TRT_EDGELLM_DISABLE_DECODE_COHORT") == nullptr;
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
        rt::IndependentPhaseCoordinator semanticCoordinator(config, semanticSchedulerConfig, *pair, ownership,
            *prefillIO, *decodeIO, prefillMap, decodeMap, prefillStream, decodeStream, std::move(seedCallbacks));
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
        serverConfig.maxPendingRequests = 1024;
        if (char const* value = std::getenv("TRT_EDGELLM_DECODE_REFILL_BATCH"))
        {
            serverConfig.decodeRefillBatchSize = static_cast<size_t>(std::stoul(value));
        }
        serverConfig.enableAdaptiveAdmission = std::getenv("TRT_EDGELLM_ADAPTIVE_ADMISSION") != nullptr;
        if (char const* value = std::getenv("TRT_EDGELLM_LATENCY_INFLIGHT"))
        {
            serverConfig.latencyInFlightRequests = static_cast<size_t>(std::stoul(value));
        }
        if (char const* reservation = std::getenv("TRT_EDGELLM_PAGE_RESERVATION");
            reservation != nullptr && std::string(reservation) == "headroom")
        {
            serverConfig.pageReservationMode = rt::IndependentPhasePageReservationMode::kHeadroom;
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
        if (visionEngineDir != nullptr && visionImagePath != nullptr)
        {
            cudaStream_t encoderStream{};
            CUDA_CHECK(cudaStreamCreateWithFlags(&encoderStream, cudaStreamNonBlocking));
            {
                auto runner = rt::MultimodalRunner::create(visionEngineDir, config.maxSupportedBatchSize,
                    config.maxKVCacheCapacity, encoderStream, checkpointDir);
                runner->allocateContextMemory();
                rt::PhaseVisionAdapter visionAdapter(*runner, tokenizer, config, encoderStream);
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
            size_t ipcIngressQuantum = serverConfig.maxPendingRequests;
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
            if (serverConfig.enableCudaGraphs)
            {
                // Retain the primed graph cache, but do not synchronously capture
                // unseen production shapes on their latency-critical first request.
                semanticCoordinator.setGraphCaptureEnabled(false);
            }
            LOG_INFO("Phase IPC shape warmup: batches=%zu requests=%zu", warmupBatchSizes.size(), warmedRequests);
            cudaStream_t ipcEncoderStream{};
            std::unique_ptr<rt::MultimodalRunner> ipcVisionRunner;
            std::unique_ptr<rt::PhaseVisionAdapter> ipcVisionAdapter;
            std::unique_ptr<rt::PhaseThreeCoordinator> ipcThreePhase;
            if (visionEngineDir != nullptr)
            {
                CUDA_CHECK(cudaStreamCreateWithFlags(&ipcEncoderStream, cudaStreamNonBlocking));
                ipcVisionRunner = rt::MultimodalRunner::create(visionEngineDir, config.maxSupportedBatchSize,
                    config.maxKVCacheCapacity, ipcEncoderStream, checkpointDir);
                ipcVisionRunner->allocateContextMemory();
                ipcVisionAdapter
                    = std::make_unique<rt::PhaseVisionAdapter>(*ipcVisionRunner, tokenizer, config, ipcEncoderStream);
                ipcThreePhase = std::make_unique<rt::PhaseThreeCoordinator>(*ipcVisionAdapter, semanticServer);
            }
            std::deque<std::string> pendingLines;
            std::mutex pendingMutex;
            bool inputClosed{};
            std::deque<std::string> outputLines;
            std::mutex outputMutex;
            std::condition_variable outputReady;
            bool outputClosed{};
            size_t outputWriteBatches{};
            size_t outputWriteRecords{};
            size_t outputWriteBytes{};
            std::thread outputWriter([&]() {
                while (true)
                {
                    std::unique_lock<std::mutex> lock(outputMutex);
                    outputReady.wait(lock, [&]() { return outputClosed || !outputLines.empty(); });
                    std::deque<std::string> readyLines;
                    readyLines.swap(outputLines);
                    bool const closed = outputClosed;
                    lock.unlock();
                    if (!readyLines.empty())
                    {
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
                        outputWriteRecords += readyLines.size();
                        outputWriteBytes += payload.size();
                    }
                    if (closed)
                    {
                        break;
                    }
                }
            });
            auto emitSerializedRecords = [&](std::vector<std::string> records) {
                if (records.empty())
                {
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(outputMutex);
                    for (std::string& record : records)
                    {
                        outputLines.push_back(std::move(record));
                    }
                }
                outputReady.notify_one();
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
                    return semanticServer.tryPopToken();
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
                    return semanticServer.tryPopCompletion();
                }
                if (nativeCompletionEvents.empty())
                {
                    return std::nullopt;
                }
                rt::IndependentPhaseServerCompletion event = std::move(nativeCompletionEvents.front());
                nativeCompletionEvents.pop_front();
                return event;
            };
            std::thread inputReader([&]() {
                std::string line;
                while (std::getline(std::cin, line))
                {
                    if (!line.empty())
                    {
                        std::lock_guard<std::mutex> lock(pendingMutex);
                        pendingLines.push_back(std::move(line));
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
                {
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    lines.swap(pendingLines);
                }
                bool madeProgress = !lines.empty();
                auto const ingressStart = std::chrono::steady_clock::now();
                size_t ingestedLines{};
                while (!lines.empty() && ingestedLines < ipcIngressQuantum)
                {
                    nlohmann::json const payload = nlohmann::json::parse(lines.front());
                    uint64_t const requestId = payload.value("request_index", uint64_t{});
                    if (payload.value("type", "submit") == "cancel")
                    {
                        bool const cancelled = ipcThreePhase != nullptr ? ipcThreePhase->cancel(requestId)
                                                                        : semanticServer.cancel(requestId);
                        nlohmann::json const cancelEvent{
                            {"type", "cancelled"}, {"request_index", requestId}, {"cancelled", cancelled}};
                        emitEvent(cancelEvent);
                        lines.pop_front();
                        ++ingestedLines;
                        continue;
                    }
                    nlohmann::json const requestPayload = payload.contains("request") ? payload.at("request") : payload;
                    rt::LLMGenerationRequest::Request request;
                    bool validRequest{true};
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
                                        std::string path = imageUrl.is_string() ? imageUrl.get<std::string>()
                                                                                : imageUrl.value("url", std::string{});
                                        std::string const fileScheme = "file://";
                                        if (path.compare(0, fileScheme.size(), fileScheme) == 0)
                                        {
                                            path.erase(0, fileScheme.size());
                                        }
                                        if (path.empty() || path.find("://") != std::string::npos)
                                        {
                                            validRequest = false;
                                            emitEvent({{"type", "error"}, {"request_index", requestId},
                                                {"message", "phase backend accepts local file image_url paths"}});
                                            break;
                                        }
                                        parsedMessage.contents.push_back({"image", ""});
                                        request.imageBuffers.push_back(rt::imageUtils::loadImageFromFile(path));
                                    }
                                }
                            }
                            request.messages.push_back(std::move(parsedMessage));
                            if (!validRequest)
                            {
                                break;
                            }
                        }
                    }
                    if (!validRequest)
                    {
                        lines.pop_front();
                        ++ingestedLines;
                        continue;
                    }
                    int32_t maxOutputTokens = serverConfig.defaultMaxOutputTokens;
                    if (requestPayload.contains("max_output_tokens"))
                    {
                        maxOutputTokens = requestPayload.at("max_output_tokens").get<int32_t>();
                    }
                    else if (requestPayload.contains("max_tokens"))
                    {
                        maxOutputTokens = requestPayload.at("max_tokens").get<int32_t>();
                    }
                    else if (requestPayload.contains("max_generate_length"))
                    {
                        maxOutputTokens = requestPayload.at("max_generate_length").get<int32_t>();
                    }
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
                                = ipcThreePhase->submit(requestId, std::move(generation), maxOutputTokens);
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
                        auto const submission = semanticServer.submitOrQueue(requestId, tokenIds, maxOutputTokens);
                        accepted = submission.status == rt::IndependentPhaseServerStatus::kAdmitted
                            || submission.status == rt::IndependentPhaseServerStatus::kQueued;
                    }
                    if (!accepted)
                    {
                        break;
                    }
                    lines.pop_front();
                    ++ingestedLines;
                }
                ipcIngressUs
                    += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ingressStart)
                           .count();
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
                    nlohmann::json const metricEvent{{"dispatch_index", metrics.dispatchIndex},
                        {"kind", static_cast<int32_t>(metrics.kind)}, {"prefill_batch", metrics.prefillBatchSize},
                        {"decode_batch", metrics.decodeBatchSize}, {"prefill_tokens", metrics.prefillTokens},
                        {"prefill_chunk_length", metrics.prefillCostLookupChunkLength},
                        {"prefill_initial_rows", metrics.prefillInitialRows},
                        {"prefill_continuation_rows", metrics.prefillContinuationRows},
                        {"prefill_past_kv_max", metrics.prefillPastKVMax}, {"decode_tokens", metrics.decodeTokens},
                        {"prefill_gpu_ms", metrics.prefillGpuMs},
                        {"decode_context_tokens", metrics.decodeContextTokens},
                        {"decode_context_max", metrics.plannedDecodeMaxContextLength},
                        {"decode_cohort_size", metrics.decodeCohortSize}, {"decode_gpu_ms", metrics.decodeGpuMs},
                        {"makespan_gpu_ms", metrics.makespanGpuMs}, {"overlap_ratio", metrics.overlapRatio},
                        {"adaptive_throughput_mode", semanticServer.throughputMode()},
                        {"adaptive_transitions", semanticServer.throughputModeTransitionCount()},
                        {"decode_refill_waits", semanticServer.decodeRefillWaitCount()},
                        {"online_decode_cost_samples",
                            semanticCoordinator.scheduler().telemetry().onlineDecodeCostSampleCount},
                        {"online_decode_cost_buckets",
                            semanticCoordinator.scheduler().telemetry().onlineDecodeCostBucketCount},
                        {"prefill_graph_hits", prefillGraphs.hits}, {"prefill_graph_misses", prefillGraphs.misses},
                        {"decode_graph_hits", decodeGraphs.hits}, {"decode_graph_misses", decodeGraphs.misses},
                        {"sampling_event_slots", samplingSlotPool.size()},
                        {"sampling_event_reuses", samplingSlotPool.reuseCount()}};
                    serializedRecords.push_back("PHASE_METRIC\t" + metricEvent.dump());
                }
                while (auto token = popTokenEvent())
                {
                    madeProgress = true;
                    nlohmann::json const tokenEvent{{"type", "token"}, {"request_index", token->requestId},
                        {"token_id", token->tokenId},
                        {"text", tokenizer.decode(std::vector<int32_t>{token->tokenId}, false)},
                        {"output_index", token->outputIndex}, {"elapsed_ms", token->elapsedMs}};
                    serializedRecords.push_back("PHASE_EVENT\t" + tokenEvent.dump());
                }
                while (auto completion = popCompletionEvent())
                {
                    madeProgress = true;
                    nlohmann::json const completionEvent{{"type", "completion"},
                        {"request_index", completion->requestId},
                        {"finish_reason", completion->stoppedByEos ? "end-of-sequence" : "length"},
                        {"prompt_tokens", completion->promptTokens},
                        {"output_tokens", completion->generatedTokens.size()}, {"latency_ms", completion->latencyMs}};
                    serializedRecords.push_back("PHASE_EVENT\t" + completionEvent.dump());
                }
                emitSerializedRecords(std::move(serializedRecords));
                ipcSerializationUs
                    += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - serializationStart)
                           .count();
                bool closed{};
                {
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    closed = inputClosed;
                }
                bool noPendingLines{};
                {
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    noPendingLines = pendingLines.empty();
                }
                bool const serverEmpty = ipcThreePhase != nullptr ? ipcThreePhase->empty() : semanticServer.empty();
                if (closed && lines.empty() && noPendingLines && serverEmpty)
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
                if (!madeProgress)
                {
                    std::this_thread::yield();
                }
            }
            inputReader.join();
            LOG_INFO("Sampling-aware decode refill waits: %zu adaptive_transitions=%zu throughput_mode=%s",
                semanticServer.decodeRefillWaitCount(), semanticServer.throughputModeTransitionCount(),
                semanticServer.throughputMode() ? "yes" : "no");
            LOG_INFO(
                "Sampling event pool: slots=%zu reuses=%zu", samplingSlotPool.size(), samplingSlotPool.reuseCount());
            LOG_INFO("Phase IPC policy: ingress_quantum=%zu emit_metrics=%s", ipcIngressQuantum,
                emitPhaseMetrics ? "yes" : "no");
            LOG_INFO("Phase IPC response path: %s", nativeEventCallbacks ? "native_callback" : "polling_queue");
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                outputClosed = true;
            }
            outputReady.notify_one();
            outputWriter.join();
            LOG_INFO(
                "Phase IPC host cost: polls=%zu ingress=%.3f ms poll=%.3f ms serialize=%.3f ms output_batches=%zu "
                "output_records=%zu output_bytes=%zu",
                ipcPollCalls, ipcIngressUs / 1000.0, ipcPollUs / 1000.0, ipcSerializationUs / 1000.0,
                outputWriteBatches, outputWriteRecords, outputWriteBytes);
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
