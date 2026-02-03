/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/c_api.h"
#include "runtime/engine_manager.h"
#include "runtime/imageUtils.h"
#include "common/logger.h"
#include <cstring>

using namespace trt_edgellm;
using namespace trt_edgellm::rt;

EdgeLLMManagerHandle EdgeLLMManagerCreate(const char* engineDir, const char* multimodalEngineDir, int numWorkers)
{
    try
    {
        // For simplicity, we use the default stream for initialization
        // Pass multimodalEngineDir to EngineManager constructor
        std::string mmDir = multimodalEngineDir ? multimodalEngineDir : "";
        return reinterpret_cast<EdgeLLMManagerHandle>(new EngineManager(engineDir, numWorkers, nullptr, mmDir));
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("C-API EdgeLLMManagerCreate failed: %s", e.what());
        return nullptr;
    }
}

bool EdgeLLMManagerInfer(EdgeLLMManagerHandle handle, int workerIdx, const EdgeLLMRequest* request, EdgeLLMResponse* response, EdgeLLMStreamCallback callback, void* ctx) {
    // Legacy single request wrapper - NOT IMPLEMENTED for VLM fully yet, focusing on Batch API
    // but we should at least support basic redirection.
    return false; // Deprecated or unimplemented for complex inputs in this short path
}

bool EdgeLLMManagerInferBatch(EdgeLLMManagerHandle handle, int workerIdx, const EdgeLLMRequest* requests, int numRequests, EdgeLLMResponse* responses, EdgeLLMStreamCallback callback, void* ctx)
{
    if (!handle || !requests || !responses || numRequests <= 0) return false;

    EngineManager* manager = reinterpret_cast<EngineManager*>(handle);
    
    LLMGenerationRequest cppRequest;
    
    // Sampling Params from first request
    cppRequest.maxGenerateLength = requests[0].max_new_tokens;
    cppRequest.temperature = requests[0].temperature;
    cppRequest.topP = requests[0].top_p;
    cppRequest.topK = requests[0].top_k;

    for (int i = 0; i < numRequests; ++i) {
        if (i > 0) {
            if (requests[i].temperature != requests[0].temperature ||
                requests[i].top_p != requests[0].top_p ||
                requests[i].top_k != requests[0].top_k ||
                requests[i].max_new_tokens != requests[0].max_new_tokens) {
                 LOG_WARNING("Batch request %d has different sampling params. Current runtime forces params from request 0.", i);
            }
        }

        LLMGenerationRequest::Request subReq;
        Message msg;
        msg.role = "user";
        
        // Parse Contents
        for (int c = 0; c < requests[i].num_contents; ++c) {
            const auto& content = requests[i].contents[c];
            std::string type(content.type);
            std::string data(content.data, content.data_len);
            
            if (type == "image") {
                try {
                    // Populate imageBuffers for multimodal runner
                    // Using loadImageFromFile assuming 'data' is a file path
                    // If it were raw bytes, we'd need a different handling or flag
                    auto image = imageUtils::loadImageFromFile(data);
                    subReq.imageBuffers.push_back(std::move(image));
                } catch (const std::exception& e) {
                    LOG_ERROR("Failed to load image from '%s': %s", data.c_str(), e.what());
                    // Skipping push_back, which might cause token imbalance and error later,
                    // but better not to crash here.
                    // Ideally we should return false.
                    return false;
                }
            }

            msg.contents.push_back({type, data});
        }
        
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
