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
    }

    CUDA_CHECK(cudaStreamDestroy(setupStream));
    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
    return EXIT_SUCCESS;
}
