/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "runtime/llmInferenceRuntime.h"
#include "common/bindingNames.h"
#include <memory>
#include <mutex>
#include "common/logger.h"
#include "common/trtUtils.h"
#include <vector>
#include <nlohmann/json.hpp>
#include <memory>

namespace trt_edgellm
{
namespace rt
{

class EngineManager
{
public:
    EngineManager(std::string const& engineDir, int numWorkers, cudaStream_t stream, std::string const& multimodalEngineDir = "");
    ~EngineManager();
    bool handleRequest(int workerIdx, LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream, std::function<void(int, std::string const&, bool)> streamCallback = nullptr);

    LLMEngineRunnerConfig getEngineConfig() const { return mConfig; }

private:
    std::unique_ptr<void, DlDeleter> mPluginHandle;
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    LLMEngineRunnerConfig mConfig;
    nlohmann::json mConfigJson;
    std::vector<std::unique_ptr<LLMInferenceRuntime>> mWorkers;
    std::vector<cudaStream_t> mStreams;
};

} // namespace rt
} // namespace trt_edgellm
