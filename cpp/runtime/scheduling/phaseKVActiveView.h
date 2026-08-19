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

#include "common/tensor.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/state/kvPageTable.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/stableKVPageManager.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Phase-local active-row view over shared stable paged-KV ownership.
//!
//! Prefill and decode create separate instances so their TensorRT contexts never
//! alias page-table or active-length buffers. Both views still reference the
//! same physical KV pool through StableKVPageManager page IDs.
class PhaseKVActiveView
{
public:
    PhaseKVActiveView(
        int32_t maxActiveRows, StableKVPageManager& ownership, TensorMap& tensorMap, std::string const& name);
    ~PhaseKVActiveView() noexcept;

    PhaseKVActiveView(PhaseKVActiveView const&) = delete;
    PhaseKVActiveView& operator=(PhaseKVActiveView const&) = delete;

    //! Bind active row -> stable slot pages and lengths into this phase's TensorMap.
    void prepare(std::vector<int32_t> const& activeStableSlots, cudaStream_t stream);

    //! Restore bindings that were active before prepare().
    void complete();

    //! Commit one resulting length per active row into stable ownership.
    void commitLengths(std::vector<int32_t> const& resultingLengths);

    //! Prepare select-token and context-length metadata for dense or compact packed prefill rows.
    void preparePrefillMetadata(PipelineIO& io, std::vector<int32_t> const& chunkLengths, cudaStream_t stream,
        bool packedTokenLayout = false) const;

    //! Prepare select-token and context-length metadata for one-token decode rows.
    void prepareDecodeMetadata(PipelineIO& io, cudaStream_t stream) const;

    KVPageTable& pageTable() noexcept;
    Tensor& activeLengths() noexcept;
    std::vector<int32_t> const& activeStableSlots() const noexcept;
    bool prepared() const noexcept;

private:
    void restoreBindings() noexcept;

    int32_t mMaxActiveRows{};
    StableKVPageManager& mOwnership;
    TensorMap& mTensorMap;
    KVPageTable mPageTable;
    Tensor mHostLengths;
    Tensor mDeviceLengths;
    std::vector<int32_t> mActiveStableSlots;
    Tensor* mPreviousLengths{};
    Tensor* mPreviousPageTable{};
    bool mPrepared{};
};

} // namespace rt
} // namespace trt_edgellm
