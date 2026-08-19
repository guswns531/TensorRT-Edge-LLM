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

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>

using namespace trt_edgellm;

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
        int32_t const prefillSlot = ownership.reserve();
        int32_t const decodeSlot = ownership.reserve();
        ownership.ensureCapacity(prefillSlot, 128);
        ownership.ensureCapacity(decodeSlot, 129);
        ownership.setLength(prefillSlot, 0);
        ownership.setLength(decodeSlot, 128);
        rt::PhaseKVActiveView prefillKV(config.maxSupportedBatchSize, ownership, prefillMap, "prefill");
        rt::PhaseKVActiveView decodeKV(config.maxSupportedBatchSize, ownership, decodeMap, "decode");
        prefillKV.prepare({prefillSlot}, prefillStream);
        decodeKV.prepare({decodeSlot}, decodeStream);

        ELLM_CHECK(
            prefillIO->inputsEmbeds.reshape({1, 128, config.hiddenSize}), "Failed to reshape prefill input embeddings");
        ELLM_CHECK(
            decodeIO->inputsEmbeds.reshape({1, 1, config.hiddenSize}), "Failed to reshape decode input embeddings");
        CUDA_CHECK(cudaMemsetAsync(
            prefillIO->inputsEmbeds.rawPointer(), 0, prefillIO->inputsEmbeds.getMemoryCapacity(), prefillStream));
        CUDA_CHECK(cudaMemsetAsync(
            decodeIO->inputsEmbeds.rawPointer(), 0, decodeIO->inputsEmbeds.getMemoryCapacity(), decodeStream));
        prefillKV.preparePrefillMetadata(*prefillIO, {128}, prefillStream);
        decodeKV.prepareDecodeMetadata(*decodeIO, decodeStream);

        ELLM_CHECK(pair->prefillExecutor().prepare(0, config.prefillDims(1, 128, false), prefillMap, prefillStream),
            "Failed to bind the stable paged-KV prefill view");
        ELLM_CHECK(pair->decodeExecutor().prepare(1, config.decodeDims(1), decodeMap, decodeStream),
            "Failed to bind the stable paged-KV decode view");
        ELLM_CHECK(pair->prefillExecutor().execute(prefillStream), "Failed to execute stable paged-KV prefill");
        ELLM_CHECK(pair->decodeExecutor().execute(decodeStream), "Failed to execute stable paged-KV decode");
        CUDA_CHECK(cudaStreamSynchronize(prefillStream));
        CUDA_CHECK(cudaStreamSynchronize(decodeStream));
        prefillKV.commitLengths({128});
        decodeKV.commitLengths({129});
        prefillKV.complete();
        decodeKV.complete();

        LOG_INFO("Independent phase context smoke passed: prefill_workspace=%zu decode_workspace=%zu stable_pages=%d",
            pair->prefillContextMemory().getMemoryCapacity(), pair->decodeContextMemory().getMemoryCapacity(),
            config.kvPoolPages - ownership.availablePages());
    }

    CUDA_CHECK(cudaStreamDestroy(setupStream));
    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
    return EXIT_SUCCESS;
}
