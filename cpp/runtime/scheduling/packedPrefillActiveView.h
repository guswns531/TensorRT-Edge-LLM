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

#pragma once

#include "runtime/config/llmEngineConfig.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/state/kvPageTable.h"
#include "runtime/state/pipelineIO.h"

#include <cstdint>
#include <vector>

namespace trt_edgellm::rt
{

//! Phase-local bindings for one packed-prefill wave over a subset of the
//! request batch. Physical KV pages stay in the source page table; only rows,
//! active lengths, and optional M-RoPE caches are gathered.
class PackedPrefillActiveView
{
public:
    PackedPrefillActiveView(LLMEngineConfig const& config, KVPageTable& sourcePageTable,
        HybridCacheManager& cacheManager, TensorMap& tensorMap, PipelineIO& pipelineIO);
    ~PackedPrefillActiveView() noexcept;

    PackedPrefillActiveView(PackedPrefillActiveView const&) = delete;
    PackedPrefillActiveView& operator=(PackedPrefillActiveView const&) = delete;

    void prepare(
        std::vector<int32_t> const& sourceRows, std::vector<int32_t> const& sourceLengths, cudaStream_t stream);
    void commitChunkLengths(std::vector<int32_t> const& chunkLengths, cudaStream_t stream);
    void complete();

    bool prepared() const noexcept;

private:
    void restoreBindings() noexcept;

    LLMEngineConfig mConfig;
    KVPageTable& mSourcePageTable;
    HybridCacheManager& mCacheManager;
    TensorMap& mTensorMap;
    PipelineIO& mPipelineIO;
    KVPageTable mActivePageTable;
    Tensor mHostActiveLengths;
    Tensor mDeviceActiveLengths;
    Tensor mHostGlobalIncrements;
    Tensor mDeviceGlobalIncrements;
    std::vector<int32_t> mSourceRows;
    Tensor* mPreviousPageTable{};
    Tensor* mPreviousLengths{};
    bool mPrepared{};
};

} // namespace trt_edgellm::rt
