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

    // Delegate to batch infer with size 1
    // But we need to adapt the callback signature.
    // Actually, let's just keep the implementation simple here.
    
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

    // Bridge C++ std::function to C callback (ignoring batchIdx)
    std::function<void(int, std::string const&, bool)> streamLambda = nullptr;
    if (callback) {
        streamLambda = [callback, ctx](int batchIdx, std::string const& token, bool isFinished) {
            (void)batchIdx;
            // The single-request callback signature does NOT have batchIdx (Wait, check c_api.h)
            // Ah, I UPDATED EdgeLLMStreamCallback definition in previous step to INCLUDE batchIdx!
            // So existing code using this typedef MUST pass batchIdx.
            // "typedef void (*EdgeLLMStreamCallback)(int batchIndex, ...)"
            // So if I use the SAME typedef, I must pass 0.
            callback(0, token.c_str(), isFinished, ctx);
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

bool EdgeLLMManagerInferBatch(EdgeLLMManagerHandle handle, int workerIdx, const EdgeLLMRequest* requests, int numRequests, EdgeLLMResponse* responses, EdgeLLMStreamCallback callback, void* ctx)
{
    if (!handle || !requests || !responses || numRequests <= 0) return false;

    EngineManager* manager = reinterpret_cast<EngineManager*>(handle);
    
    LLMGenerationRequest cppRequest;
    
    // Use the params from the first request for the whole batch (TensorRT limitation usually requires uniform sampling params or padding)
    // Actually, TRT-LLM usually supports per-request sampling params if configured.
    // But LLMGenerationRequest struct has global params?
    // Let's check LLMGenerationRequest definition.
    // It has `maxGenerateLength`, `temperature`, etc. as members of `LLMGenerationRequest`.
    // It does NOT have per-request sampling params in `requests` vector (which is `LLMGenerationRequest::Request`).
    // So all requests in the batch MUST share generation config.
    // For now, I will use params from requests[0].
    
    cppRequest.maxGenerateLength = requests[0].max_new_tokens;
    cppRequest.temperature = requests[0].temperature;
    cppRequest.topP = requests[0].top_p;
    cppRequest.topK = requests[0].top_k;

    for (int i = 0; i < numRequests; ++i) {
        LLMGenerationRequest::Request subReq;
        Message msg;
        msg.role = "user";
        msg.contents.push_back({"text", requests[i].prompt});
        subReq.messages.push_back(msg);
        cppRequest.requests.push_back(subReq);
    }

    // Bridge C++ std::function to C callback
    std::function<void(int, std::string const&, bool)> streamLambda = nullptr;
    if (callback) {
        streamLambda = [callback, ctx](int batchIdx, std::string const& token, bool isFinished) {
            callback(batchIdx, token.c_str(), isFinished, ctx);
        };
    }

    LLMGenerationResponse cppResponse;
    if (manager->handleRequest(workerIdx, cppRequest, cppResponse, nullptr, streamLambda))
    {
        if (cppResponse.outputTexts.size() != static_cast<size_t>(numRequests)) {
             LOG_ERROR("Mismatch in response size: expected %d, got %zu", numRequests, cppResponse.outputTexts.size());
             return false;
        }

        for (int i = 0; i < numRequests; ++i) {
             responses[i].text = strdup(cppResponse.outputTexts[i].c_str());
             if (i < (int)cppResponse.outputIds.size()) {
                 responses[i].num_tokens = static_cast<int32_t>(cppResponse.outputIds[i].size());
             } else {
                 responses[i].num_tokens = 0;
             }
        }
        return true;
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
