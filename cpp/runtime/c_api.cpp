/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/c_api.h"
#include "runtime/engine_manager.h"
#include "common/logger.h"
#include <cstring>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

EdgeLLMManagerHandle EdgeLLMManagerCreate(const char* engineDir, int numWorkers)
{
    try
    {
        // For simplicity, we use the default stream for initialization
        return reinterpret_cast<EdgeLLMManagerHandle>(new EngineManager(engineDir, numWorkers, nullptr));
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("C-API EdgeLLMManagerCreate failed: %s", e.what());
        return nullptr;
    }
}

bool EdgeLLMManagerInfer(EdgeLLMManagerHandle handle, int workerIdx, const EdgeLLMRequest* request, EdgeLLMResponse* response, EdgeLLMStreamCallback callback, void* ctx)
{
    if (!handle || !request || !response) return false;

    EngineManager* manager = reinterpret_cast<EngineManager*>(handle);
    
    LLMGenerationRequest cppRequest;
    LLMGenerationRequest::Request subReq;
    Message msg;
    msg.role = "user";
    msg.contents.push_back({"text", request->prompt});
    subReq.messages.push_back(msg);
    cppRequest.requests.push_back(subReq);

    cppRequest.maxGenerateLength = request->max_new_tokens;
    cppRequest.temperature = request->temperature;
    cppRequest.topP = request->top_p;
    cppRequest.topK = request->top_k;

    // Bridge C++ std::function to C callback
    std::function<void(std::string const&, bool)> streamLambda = nullptr;
    if (callback) {
        streamLambda = [callback, ctx](std::string const& token, bool isFinished) {
            callback(token.c_str(), isFinished, ctx);
        };
    }

    LLMGenerationResponse cppResponse;
    if (manager->handleRequest(workerIdx, cppRequest, cppResponse, nullptr, streamLambda))
    {
        if (!cppResponse.outputTexts.empty()) {
            response->text = strdup(cppResponse.outputTexts[0].c_str());
            if (!cppResponse.outputIds.empty()) {
                response->num_tokens = static_cast<int32_t>(cppResponse.outputIds[0].size());
            } else {
                response->num_tokens = 0;
            }
            return true;
        }
    }

    return false;
}

void EdgeLLMManagerDestroy(EdgeLLMManagerHandle handle)
{
    if (handle)
    {
        delete reinterpret_cast<EngineManager*>(handle);
    }
}

void EdgeLLMFreeResponse(EdgeLLMResponse* response)
{
    if (response && response->text)
    {
        free(response->text);
        response->text = nullptr;
    }
}
