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

#include <cstdlib>
#include <filesystem>
#include <memory>

using namespace trt_edgellm;

int main(int argc, char** argv)
{
    constexpr int32_t kEXPECTED_ARGUMENTS = 2;
    if (argc != kEXPECTED_ARGUMENTS)
    {
        LOG_ERROR("Usage: %s <engine-dir>", argv[0]);
        return EXIT_FAILURE;
    }

    auto pluginHandle = loadEdgellmPluginLib();
    if (pluginHandle == nullptr)
    {
        return EXIT_FAILURE;
    }

    std::filesystem::path const engineDir{argv[1]};
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

        LOG_INFO("Independent phase context smoke passed: prefill_workspace=%zu decode_workspace=%zu",
            pair->prefillContextMemory().getMemoryCapacity(), pair->decodeContextMemory().getMemoryCapacity());
    }

    CUDA_CHECK(cudaStreamDestroy(setupStream));
    CUDA_CHECK(cudaStreamDestroy(prefillStream));
    CUDA_CHECK(cudaStreamDestroy(decodeStream));
    return EXIT_SUCCESS;
}
