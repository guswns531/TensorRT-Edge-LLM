/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/engine_manager.h"
#include "common/logger.h"
#include "runtime/llmEngineRunner.h"
#include <filesystem>

namespace trt_edgellm
{
namespace rt
{

EngineManager::EngineManager(std::string const& engineDir, int numWorkers, cudaStream_t stream)
    : mPluginHandle(loadEdgellmPluginLib())
{
    (void)stream;
    std::filesystem::path root(engineDir);
    std::filesystem::path enginePath = root / "llm.engine";
    std::filesystem::path configPath = root / "config.json";

    LOG_INFO("EngineManager initializing with %d workers from %s", numWorkers, engineDir.c_str());

    if (!LLMEngineRunner::loadEngineAndConfig(enginePath, configPath, mEngine, mConfig, mConfigJson))
    {
        LOG_ERROR("Failed to load engine or config.");
        throw std::runtime_error("EngineManager initialization failed.");
    }

    mWorkers.reserve(numWorkers);
    mStreams.reserve(numWorkers);

    for (int i = 0; i < numWorkers; ++i)
    {
        cudaStream_t workerStream;
        cudaStreamCreate(&workerStream);
        mStreams.push_back(workerStream);

        // Share the engine! Each runner gets its own ExecutionContext and LinearKVCache.
        auto runner = std::make_unique<LLMEngineRunner>(mEngine, mConfig, mConfigJson, std::unordered_map<std::string, std::string>{}, workerStream);
        
        // Load tokenizer (shared across all workers)
        static std::shared_ptr<tokenizer::Tokenizer> sharedTokenizer = nullptr;
        static std::mutex tokenizerMutex;
        
        {
            std::lock_guard<std::mutex> lock(tokenizerMutex);
            if (!sharedTokenizer) {
                sharedTokenizer = std::make_shared<tokenizer::Tokenizer>();
                sharedTokenizer->loadFromHF(engineDir);
                LOG_INFO("Tokenizer loaded and shared.");
            }
        }

        // Wrap it in LLMInferenceRuntime using the new constructor
        mWorkers.push_back(std::make_unique<LLMInferenceRuntime>(std::move(runner), sharedTokenizer, workerStream));
        
        LOG_INFO("Worker %d initialized.", i);
    }
}

EngineManager::~EngineManager()
{
    mWorkers.clear(); // Explicitly destroy workers first
    for (auto stream : mStreams)
    {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
}

bool EngineManager::handleRequest(int workerIdx, LLMGenerationRequest const& request, LLMGenerationResponse& response, cudaStream_t stream, std::function<void(std::string const&, bool)> streamCallback)
{
    (void)stream;
    if (workerIdx < 0 || workerIdx >= static_cast<int>(mWorkers.size()))
    {
        LOG_ERROR("Invalid worker index: %d", workerIdx);
        return false;
    }

    // Capture the worker's stream
    cudaStream_t workerStream = mStreams[workerIdx];

    // Standard LLMInferenceRuntime::handleRequest uses the provided stream.
    // In our case, we want to use the worker's dedicated stream for execution.
    // The 'stream' parameter from Go will likely be the same or we use it for sync.
    bool success = mWorkers[workerIdx]->handleRequest(request, response, workerStream, streamCallback);
    
    // Synchronize to ensure Go gets the data immediately
    if (success)
    {
        cudaStreamSynchronize(workerStream);
    }

    return success;
}

} // namespace rt
} // namespace trt_edgellm
