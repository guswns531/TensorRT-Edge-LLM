/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

typedef void* EdgeLLMManagerHandle;

typedef struct {
    const char* prompt;
    int32_t max_new_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    bool stream_output;
} EdgeLLMRequest;

typedef void (*EdgeLLMStreamCallback)(const char* token, bool is_finished, void* ctx);

typedef struct {
    char* text;
    int32_t num_tokens;
} EdgeLLMResponse;

EdgeLLMManagerHandle EdgeLLMManagerCreate(const char* engineDir, int numWorkers);

bool EdgeLLMManagerInfer(EdgeLLMManagerHandle handle, int workerIdx, const EdgeLLMRequest* request, EdgeLLMResponse* response, EdgeLLMStreamCallback callback, void* ctx);

void EdgeLLMManagerDestroy(EdgeLLMManagerHandle handle);

void EdgeLLMFreeResponse(EdgeLLMResponse* response);

#ifdef __cplusplus
}
#endif
