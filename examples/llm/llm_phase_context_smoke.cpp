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

#include "common/checkMacros.h"
#include "common/logger.h"
#include "common/trtUtils.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/scheduling/independentEngineExecutorPair.h"
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/scheduling/phaseKVActiveView.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace trt_edgellm;

namespace
{

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

        rt::StableKVPageManager ownership({config.maxSupportedBatchSize, config.maxSupportedBatchSize,
            config.kvPoolPages, config.maxKVCacheCapacity, 128});
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

        LOG_INFO("Independent phase context smoke passed: prefill_workspace=%zu decode_workspace=%zu stable_pages=%d",
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
        rt::PhaseQueueScheduler scheduler(schedulerConfig);

        std::unordered_map<uint64_t, int32_t> decodeSteps;
        std::vector<rt::PhaseDispatchMetrics> dispatchMetrics;
        rt::PhaseDispatchWorkerCallbacks callbacks;
        callbacks.enqueuePrefill = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream) {
            std::vector<int32_t> slots;
            std::vector<int32_t> chunks;
            int32_t totalTokens{};
            for (rt::PhaseWorkItem const& item : batch)
            {
                slots.push_back(item.kvSlotId);
                chunks.push_back(item.tokenCount);
                totalTokens += item.tokenCount;
                ownership.ensureCapacity(item.kvSlotId, ownership.length(item.kvSlotId) + item.tokenCount);
            }
            prefillKV.prepare(slots, stream);
            ELLM_CHECK(prefillIO->inputsEmbeds.reshape({1, totalTokens, config.hiddenSize}),
                "Queued packed prefill embedding reshape failed");
            prefillKV.preparePrefillMetadata(*prefillIO, chunks, stream, true);
            ELLM_CHECK(
                pair->prefillExecutor().prepare(
                    0, config.packedPrefillDims(static_cast<int64_t>(batch.size()), totalTokens), prefillMap, stream),
                "Queued packed prefill prepare failed");
            ELLM_CHECK(pair->prefillExecutor().execute(stream), "Queued packed prefill execute failed");
        };
        callbacks.completePrefillBatch = [&](std::vector<rt::PhaseWorkItem> const& batch) {
            std::vector<int32_t> resultingLengths;
            for (rt::PhaseWorkItem const& item : batch)
            {
                resultingLengths.push_back(ownership.length(item.kvSlotId) + item.tokenCount);
            }
            prefillKV.commitLengths(resultingLengths);
            prefillKV.complete();
        };
        callbacks.completePrefill = [&](rt::PhaseWorkItem const& item) {
            return rt::PhasePrefillCompletion{ownership.length(item.kvSlotId), false};
        };
        callbacks.enqueueDecode = [&](std::vector<rt::PhaseWorkItem> const& batch, cudaStream_t stream) {
            std::vector<int32_t> slots;
            for (rt::PhaseWorkItem const& item : batch)
            {
                slots.push_back(item.kvSlotId);
                ownership.ensureCapacity(item.kvSlotId, ownership.length(item.kvSlotId) + 1);
            }
            decodeKV.prepare(slots, stream);
            ELLM_CHECK(decodeIO->inputsEmbeds.reshape({static_cast<int64_t>(batch.size()), 1, config.hiddenSize}),
                "Queued decode embedding reshape failed");
            decodeKV.prepareDecodeMetadata(*decodeIO, stream);
            ELLM_CHECK(pair->decodeExecutor().prepare(
                           1, config.decodeDims(static_cast<int64_t>(batch.size())), decodeMap, stream),
                "Queued decode prepare failed");
            ELLM_CHECK(pair->decodeExecutor().execute(stream), "Queued decode execute failed");
        };
        callbacks.completeDecodeBatch = [&](std::vector<rt::PhaseWorkItem> const& batch) {
            std::vector<int32_t> resultingLengths;
            for (rt::PhaseWorkItem const& item : batch)
            {
                resultingLengths.push_back(ownership.length(item.kvSlotId) + 1);
            }
            decodeKV.commitLengths(resultingLengths);
            decodeKV.complete();
        };
        callbacks.completeDecode = [&](rt::PhaseWorkItem const& item) {
            int32_t const steps = ++decodeSteps[item.requestId];
            return rt::PhaseDecodeCompletion{ownership.length(item.kvSlotId), steps >= 2};
        };
        callbacks.onMetrics = [&](rt::PhaseDispatchMetrics const& metrics) { dispatchMetrics.push_back(metrics); };

        rt::PhaseExecutionSafetyContract const safety = rt::PhaseExecutionSafetyContract::independent(
            {pair->prefillExecutor().getExecutionContextIdentity(), pair->prefillContextMemory().rawPointer(),
                prefillIO.get()},
            {pair->decodeExecutor().getExecutionContextIdentity(), pair->decodeContextMemory().rawPointer(),
                decodeIO.get()});
        rt::PhaseDispatchWorker worker(scheduler, std::move(callbacks), prefillStream, decodeStream,
            rt::PhaseTensorRTContextMode::kIndependentConcurrent, safety);

        rt::PhaseSchedulingHints decodeHints;
        decodeHints.tpotTargetUs = 50000.0;
        scheduler.enqueuePrefill({101, 256, prefillSlot0, 0, 256});
        scheduler.enqueuePrefill({102, 256, prefillSlot1, 0, 256});
        scheduler.enqueueDecode({103, 128, decodeSlot, 0, 0, true, decodeHints});
        worker.runUntilIdle(32);
        size_t const overlapDispatches = static_cast<size_t>(std::count_if(dispatchMetrics.begin(),
            dispatchMetrics.end(),
            [](rt::PhaseDispatchMetrics const& metrics) { return metrics.kind == rt::PhaseDispatchKind::kOverlap; }));
        double totalMakespanMs{};
        for (rt::PhaseDispatchMetrics const& metrics : dispatchMetrics)
        {
            totalMakespanMs += metrics.makespanGpuMs;
        }
        ELLM_CHECK(scheduler.empty() && overlapDispatches > 0, "Cost-aware phase queue did not drain with overlap");
        LOG_INFO(
            "Cost-aware phase queue passed: dispatches=%zu overlap_dispatches=%zu total_makespan=%.4f ms "
            "request101_kv=%d request102_kv=%d request103_kv=%d",
            dispatchMetrics.size(), overlapDispatches, totalMakespanMs, ownership.length(prefillSlot0),
            ownership.length(prefillSlot1), ownership.length(decodeSlot));
    }

    CUDA_CHECK(cudaStreamDestroy(setupStream));
    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
    return EXIT_SUCCESS;
}
