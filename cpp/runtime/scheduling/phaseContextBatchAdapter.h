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

struct PhaseContextRow
{
    uint64_t requestId{};
    DecodingInferenceContext* context{};
    int32_t contextRow{-1};
    int32_t kvSlotId{-1};
    int32_t kvLength{};
};

//! Packs rows from independent request contexts into one production decode batch.
//!
//! V1 intentionally supports vanilla text decode without streaming callbacks,
//! logprobs, LoRA, multimodal tensors, or logit bias. Those features own
//! request-scoped or runtime-global buffers that require explicit fan-in/fan-out.
class PhaseContextBatchAdapter
{
public:
    PhaseContextBatchAdapter(
        int32_t maxBatchSize, HybridCacheManager& cacheManager, TensorMap& tensorMap, std::string const& name);
    ~PhaseContextBatchAdapter() noexcept;

    PhaseContextBatchAdapter(PhaseContextBatchAdapter const&) = delete;
    PhaseContextBatchAdapter& operator=(PhaseContextBatchAdapter const&) = delete;
    PhaseContextBatchAdapter(PhaseContextBatchAdapter&&) = delete;
    PhaseContextBatchAdapter& operator=(PhaseContextBatchAdapter&&) = delete;

    //! Pack compatible source rows and bind their stable KV slots for decode.
    void packDecode(std::vector<PhaseContextRow> const& rows, cudaStream_t stream);

    //! Scatter decode completion state back to each source context and restore legacy bindings.
    void scatterDecode();

    DecodingInferenceContext& packedContext();
    //! Device INT32 [batch] containing each source row's current decode token.
    Tensor& tokenIds() noexcept;
    //! Request-owned M-RoPE cache for a packed multimodal decode batch.
    OptionalInputTensor mropeCosSin() const noexcept;
    //! Source rows corresponding to packed row order.
    std::vector<PhaseContextRow> const& rows() const noexcept;
    std::vector<PhaseWorkItem> const& workItems() const noexcept;
    bool packed() const noexcept;

private:
    void restoreBindings() noexcept;
    void validateSourceRow(PhaseContextRow const& row) const;

    int32_t mMaxBatchSize{};
    HybridCacheManager& mCacheManager;
    TensorMap& mTensorMap;
    Tensor mHostTokenIds;
    Tensor mDeviceTokenIds;
    PhaseBatchState mBatchState;
    DecodingInferenceContext mPackedContext;
    std::vector<PhaseContextRow> mRows;
    std::vector<PhaseWorkItem> mWorkItems;
    Tensor* mPreviousSlotIds{};
    Tensor* mPreviousLengths{};
    bool mPacked{};
};

} // namespace rt
} // namespace trt_edgellm
