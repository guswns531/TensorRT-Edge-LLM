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

#include "runtime/scheduling/phaseBatchState.h"
#include "runtime/state/decodingInferenceContext.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct PhasePrefillContextRow
{
    uint64_t requestId{};
    DecodingInferenceContext* context{};
    int32_t contextRow{-1};
    int32_t kvSlotId{-1};
    int32_t tokenOffset{};
    int32_t tokenCount{};
    int32_t promptTokenCount{};
};

//! Packs equal-length prompt chunks from independent request contexts.
//!
//! The adapter owns pinned/device token staging and phase-local indexed KV
//! bindings. Source contexts and the TensorMap must outlive an in-flight pack.
class PhasePrefillContextBatchAdapter
{
public:
    PhasePrefillContextBatchAdapter(int32_t maxBatchSize, int32_t maxChunkTokens, HybridCacheManager& cacheManager,
        TensorMap& tensorMap, std::string const& name);
    ~PhasePrefillContextBatchAdapter() noexcept;

    PhasePrefillContextBatchAdapter(PhasePrefillContextBatchAdapter const&) = delete;
    PhasePrefillContextBatchAdapter& operator=(PhasePrefillContextBatchAdapter const&) = delete;
    PhasePrefillContextBatchAdapter(PhasePrefillContextBatchAdapter&&) = delete;
    PhasePrefillContextBatchAdapter& operator=(PhasePrefillContextBatchAdapter&&) = delete;

    //! Pack one uniform chunk-size bucket and bind its stable KV slots.
    void pack(std::vector<PhasePrefillContextRow> const& rows, cudaStream_t stream);

    //! Restore the exact bindings that were active before pack().
    void complete();

    Tensor& tokenIds() noexcept;
    PhaseBatchState& phaseBatchState() noexcept;
    std::vector<PhasePrefillContextRow> const& rows() const noexcept;
    std::vector<PhaseWorkItem> const& workItems() const noexcept;
    int32_t batchSize() const noexcept;
    int32_t chunkLength() const noexcept;
    bool initialChunk() const noexcept;
    cudaStream_t stream() const noexcept;
    bool packed() const noexcept;

private:
    void validateRow(PhasePrefillContextRow const& row) const;
    void restoreBindings() noexcept;

    int32_t mMaxBatchSize{};
    int32_t mMaxChunkTokens{};
    HybridCacheManager& mCacheManager;
    TensorMap& mTensorMap;
    Tensor mHostTokenIds;
    Tensor mDeviceTokenIds;
    PhaseBatchState mBatchState;
    std::vector<PhasePrefillContextRow> mRows;
    std::vector<PhaseWorkItem> mWorkItems;
    Tensor* mPreviousSlotIds{};
    Tensor* mPreviousLengths{};
    int32_t mBatchSize{};
    int32_t mChunkLength{};
    bool mInitialChunk{};
    cudaStream_t mStream{};
    bool mPacked{};
};

} // namespace rt
} // namespace trt_edgellm
