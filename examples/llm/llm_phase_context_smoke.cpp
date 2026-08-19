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
#include "runtime/llmRuntimeUtils.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/scheduling/independentEngineExecutorPair.h"
#include "runtime/scheduling/independentPhaseAsyncServer.h"
#include "runtime/scheduling/independentPhaseCoordinator.h"
#include "runtime/scheduling/phaseContinuousLoadGenerator.h"
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/scheduling/phaseKVActiveView.h"
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
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

constexpr int32_t kDEFAULT_STABLE_SLOTS = 80;
constexpr size_t kDEFAULT_MAX_INFLIGHT_REQUESTS = 16U;

struct PhaseTiming
{
    float prefillMs{};
    float decodeMs{};
    float makespanMs{};
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
        auto prefillIO = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForLLM(config, setupStream));
        auto decodeIO = std::make_unique<rt::PipelineIO>(rt::PipelineIO::createForLLM(config, setupStream));
        rt::TensorMap prefillMap;
        rt::TensorMap decodeMap;
        rt::buildTensorMap(prefillMap, *prefillIO, *resources, config, 0);
        rt::buildTensorMap(decodeMap, *decodeIO, *resources, config, 0);
        resources->externalWeightManager->load(engineDir, engineDir / "config.json", setupStream, checkpointDir);
        resources->externalWeightManager->validateAgainstEngine(pair->prefillExecutor(), "base");
        resources->externalWeightManager->registerTensorMapEntries(prefillMap);
        resources->externalWeightManager->registerTensorMapEntries(decodeMap);

        int32_t maxStableSlots = std::max(kDEFAULT_STABLE_SLOTS, config.maxSupportedBatchSize);
        if (char const* value = std::getenv("TRT_EDGELLM_MAX_STABLE_SLOTS"))
        {
            maxStableSlots = std::stoi(value);
        }
        ELLM_CHECK(maxStableSlots >= config.maxSupportedBatchSize,
            "Stable slot capacity must be at least the engine active batch size");
        rt::StableKVPageManager ownership(
            {maxStableSlots, config.maxSupportedBatchSize, config.kvPoolPages, config.maxKVCacheCapacity, 128});
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
            rt::PhaseKVActiveView prefillKV(config.maxSupportedBatchSize, ownership, prefillMap, "prefill");
            rt::PhaseKVActiveView decodeKV(config.maxSupportedBatchSize, ownership, decodeMap, "decode");
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
        rt::EmbeddingData embedding = rt::loadEmbeddingTable(engineDir / "embedding.safetensors", setupStream);
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

        rt::Tensor hostSemanticPrefillIds({config.maxSupportedBatchSize, config.maxPackedPrefillChunkTokens},
            rt::DeviceType::kCPU, nvinfer1::DataType::kINT32, "semantic_phase_host_prefill_ids");
        rt::Tensor deviceSemanticPrefillIds({config.maxSupportedBatchSize, config.maxPackedPrefillChunkTokens},
            rt::DeviceType::kGPU, nvinfer1::DataType::kINT32, "semantic_phase_prefill_ids");
        rt::Tensor hostSemanticDecodeIds({config.maxSupportedBatchSize, 1}, rt::DeviceType::kCPU,
            nvinfer1::DataType::kINT32, "semantic_phase_host_decode_ids");
        rt::Tensor deviceSemanticDecodeIds({config.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "semantic_phase_decode_ids");
        size_t const samplingWorkspaceBytes
            = getSelectAllTopKWorkspaceSize(config.maxSupportedBatchSize, config.outputVocabSize, 1);
        rt::Tensor prefillSamplingWorkspace({static_cast<int64_t>(samplingWorkspaceBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT8, "semantic_phase_prefill_sampling_workspace");
        rt::Tensor decodeSamplingWorkspace({static_cast<int64_t>(samplingWorkspaceBytes)}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT8, "semantic_phase_decode_sampling_workspace");
        rt::Tensor prefillSelectedIds({config.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "semantic_phase_prefill_selected_ids");
        rt::Tensor decodeSelectedIds({config.maxSupportedBatchSize, 1}, rt::DeviceType::kGPU,
            nvinfer1::DataType::kINT32, "semantic_phase_decode_selected_ids");
        rt::Tensor hostPrefillSelectedIds({config.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            nvinfer1::DataType::kINT32, "semantic_phase_host_prefill_selected_ids");
        rt::Tensor hostDecodeSelectedIds({config.maxSupportedBatchSize}, rt::DeviceType::kCPU,
            nvinfer1::DataType::kINT32, "semantic_phase_host_decode_selected_ids");

        auto stageTokens = [&](std::vector<rt::IndependentPhaseRequestView> const& views, rt::PipelineIO& io,
                               rt::TensorMap& map, cudaStream_t stream, bool prefill) {
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
            embeddingPreprocessor.embed(deviceIds, std::nullopt, std::nullopt, io, stream);
            embeddingPreprocessor.prepareDeepstack(deviceIds, rt::OptionalInputTensors{}, io, stream);
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
            rt::Tensor& hostSelectedIds = prefill ? hostPrefillSelectedIds : hostDecodeSelectedIds;
            rt::Tensor& workspace = prefill ? prefillSamplingWorkspace : decodeSamplingWorkspace;
            ELLM_CHECK(io.outputLogits.reshape({batchSize, config.outputVocabSize})
                    && selectedIds.reshape({batchSize, 1}) && hostSelectedIds.reshape({batchSize}),
                "Semantic phase sampling reshape failed");
            selectAllTopK(io.outputLogits, std::nullopt, selectedIds, 1, workspace, stream);
            CUDA_CHECK(cudaMemcpyAsync(hostSelectedIds.rawPointer(), selectedIds.rawPointer(),
                static_cast<size_t>(batchSize) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
            cudaEvent_t ready{};
            CUDA_CHECK(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(ready, stream));
            rt::Tensor* const hostSelectedIdsPtr = &hostSelectedIds;
            auto ticket = std::make_unique<rt::IndependentPhaseSampleTicket>();
            ticket->ready = ready;
            ticket->fromPrefill = prefill;
            for (rt::IndependentPhaseRequestView const& view : views)
            {
                ticket->requestIds.push_back(view.requestId);
            }
            ticket->collect = [hostSelectedIdsPtr, batchSize]() {
                int32_t const* selected = hostSelectedIdsPtr->dataPointer<int32_t>();
                return std::vector<int32_t>(selected, selected + batchSize);
            };
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
        semanticSchedulerConfig.maxPrefillBatchSize = std::min(8, config.maxSupportedBatchSize);
        semanticSchedulerConfig.maxDecodeBatchSize = config.maxSupportedBatchSize;
        semanticSchedulerConfig.maxPrefillChunkTokens = 128;
        semanticSchedulerConfig.maxOverlapPrefillTokens = 128;
        semanticSchedulerConfig.enablePackedPrefillTokenLayout = true;
        semanticSchedulerConfig.enableAdaptivePrefillChunking = true;
        semanticSchedulerConfig.minPrefillChunkTokens = 32;
        semanticSchedulerConfig.prefillChunkAlignment = 8;
        semanticSchedulerConfig.adaptivePrefillChunkCandidates = {32, 64, 128};
        semanticSchedulerConfig.enableMetricsPolicy = true;
        semanticSchedulerConfig.minMetricsSamples = 2;
        semanticSchedulerConfig.prefillQueueWaitTargetUs = 5000.0;
        semanticSchedulerConfig.decodeQueueWaitTargetUs = 2000.0;
        semanticSchedulerConfig.enableDynamicDecodeBatching = true;
        semanticSchedulerConfig.decodeBatchCosts = {
            {1, 2048, 5.877F},
            {2, 2048, 5.901F},
            {3, 2048, 5.931F},
            {4, 2048, 5.957F},
            {5, 2048, 5.972F},
            {6, 2048, 6.048F},
            {7, 2048, 6.080F},
            {8, 2048, 6.107F},
        };
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
        serverConfig.eosTokenIds = config.eosTokenIds;
        serverConfig.enablePrefixReuse = enablePrefixReuse;
        serverConfig.enableCudaGraphs = std::getenv("TRT_EDGELLM_CAPTURE_PHASE_GRAPHS") != nullptr;
        std::unique_ptr<rt::PhasePrefixReuseCache> semanticPrefixCache;
        if (enablePrefixReuse)
        {
            semanticPrefixCache = std::make_unique<rt::PhasePrefixReuseCache>(ownership, 1);
        }
        rt::IndependentPhaseAsyncServer semanticServer(
            serverConfig, semanticCoordinator, ownership, std::move(semanticAdapter), semanticPrefixCache.get());
        bool const ipcMode = std::getenv("TRT_EDGELLM_PHASE_IPC") != nullptr;
        bool const prefixReuseGate = std::getenv("TRT_EDGELLM_PREFIX_REUSE_GATE") != nullptr;
        if (prefixReuseGate)
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
            std::deque<std::string> pendingLines;
            std::mutex pendingMutex;
            bool inputClosed{};
            std::deque<std::string> outputLines;
            std::mutex outputMutex;
            std::condition_variable outputReady;
            bool outputClosed{};
            std::thread outputWriter([&]() {
                while (true)
                {
                    std::unique_lock<std::mutex> lock(outputMutex);
                    outputReady.wait(lock, [&]() { return outputClosed || !outputLines.empty(); });
                    while (!outputLines.empty())
                    {
                        std::string line = std::move(outputLines.front());
                        outputLines.pop_front();
                        lock.unlock();
                        std::cout << line << std::endl;
                        lock.lock();
                    }
                    if (outputClosed)
                    {
                        break;
                    }
                }
            });
            auto emitRecord = [&](std::string const& prefix, nlohmann::json const& event) {
                {
                    std::lock_guard<std::mutex> lock(outputMutex);
                    outputLines.push_back(prefix + event.dump());
                }
                outputReady.notify_one();
            };
            auto emitEvent = [&](nlohmann::json const& event) { emitRecord("PHASE_EVENT\t", event); };
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
            std::unordered_map<uint64_t, int32_t> promptLengths;
            size_t emittedMetrics{};
            while (true)
            {
                std::deque<std::string> lines;
                {
                    std::lock_guard<std::mutex> lock(pendingMutex);
                    lines.swap(pendingLines);
                }
                while (!lines.empty())
                {
                    nlohmann::json const payload = nlohmann::json::parse(lines.front());
                    uint64_t const requestId = payload.value("request_index", uint64_t{});
                    if (payload.value("type", "submit") == "cancel")
                    {
                        bool const cancelled = semanticServer.cancel(requestId);
                        nlohmann::json const cancelEvent{
                            {"type", "cancelled"}, {"request_index", requestId}, {"cancelled", cancelled}};
                        emitEvent(cancelEvent);
                        lines.pop_front();
                        continue;
                    }
                    nlohmann::json const requestPayload = payload.contains("request") ? payload.at("request") : payload;
                    rt::LLMGenerationRequest::Request request;
                    if (requestPayload.contains("messages") && requestPayload.at("messages").is_array())
                    {
                        for (auto const& message : requestPayload.at("messages"))
                        {
                            std::string content;
                            if (message.contains("content") && message.at("content").is_string())
                            {
                                content = message.at("content").get<std::string>();
                            }
                            else if (message.contains("content") && message.at("content").is_array())
                            {
                                for (auto const& part : message.at("content"))
                                {
                                    if (part.value("type", "") == "text" && part.contains("text"))
                                    {
                                        content += part.at("text").get<std::string>();
                                    }
                                }
                            }
                            std::string const role = message.value("role", "user");
                            request.messages.push_back({role, {{"text", content}}});
                        }
                    }
                    rt::LLMGenerationRequest::FormattedRequest formatted;
                    ELLM_CHECK(tokenizer.applyChatTemplate(request, formatted, true, true, false),
                        "Failed to format IPC phase request");
                    std::vector<int32_t> const tokenIds = tokenizer.encode(formatted.formattedCompleteRequest, false);
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
                    auto const submission = semanticServer.submit(requestId, tokenIds, maxOutputTokens);
                    if (submission.status == rt::IndependentPhaseServerStatus::kAdmitted)
                    {
                        promptLengths[requestId] = static_cast<int32_t>(tokenIds.size());
                        lines.pop_front();
                    }
                    else
                    {
                        break;
                    }
                }
                static_cast<void>(semanticServer.poll());
                while (emittedMetrics < semanticCoordinator.metrics().size())
                {
                    rt::PhaseDispatchMetrics const& metrics = semanticCoordinator.metrics()[emittedMetrics++];
                    nlohmann::json const metricEvent{{"dispatch_index", metrics.dispatchIndex},
                        {"kind", static_cast<int32_t>(metrics.kind)}, {"prefill_batch", metrics.prefillBatchSize},
                        {"decode_batch", metrics.decodeBatchSize}, {"prefill_tokens", metrics.prefillTokens},
                        {"decode_tokens", metrics.decodeTokens}, {"prefill_gpu_ms", metrics.prefillGpuMs},
                        {"decode_gpu_ms", metrics.decodeGpuMs}, {"makespan_gpu_ms", metrics.makespanGpuMs},
                        {"overlap_ratio", metrics.overlapRatio}};
                    emitRecord("PHASE_METRIC\t", metricEvent);
                }
                while (auto token = semanticServer.tryPopToken())
                {
                    nlohmann::json const tokenEvent{{"type", "token"}, {"request_index", token->requestId},
                        {"token_id", token->tokenId},
                        {"text", tokenizer.decode(std::vector<int32_t>{token->tokenId}, false)},
                        {"output_index", token->outputIndex}, {"elapsed_ms", token->elapsedMs}};
                    emitEvent(tokenEvent);
                }
                while (auto completion = semanticServer.tryPopCompletion())
                {
                    int32_t const promptLength = promptLengths[completion->requestId];
                    nlohmann::json const completionEvent{{"type", "completion"},
                        {"request_index", completion->requestId},
                        {"finish_reason", completion->stoppedByEos ? "end-of-sequence" : "length"},
                        {"prompt_tokens", promptLength}, {"output_tokens", completion->generatedTokens.size()},
                        {"latency_ms", completion->latencyMs}};
                    emitEvent(completionEvent);
                    promptLengths.erase(completion->requestId);
                }
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
                if (closed && lines.empty() && noPendingLines && semanticServer.empty())
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
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            inputReader.join();
            {
                std::lock_guard<std::mutex> lock(outputMutex);
                outputClosed = true;
            }
            outputReady.notify_one();
            outputWriter.join();
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
