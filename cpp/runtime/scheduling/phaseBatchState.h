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
#include "runtime/hybridCacheManager.h"
#include "runtime/scheduling/phaseQueueScheduler.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Stable phase-local bindings for indexed KV cache execution.
//!
//! Each TensorRT execution context owns one instance. The device tensors stay
//! at stable addresses after bind() while prepare() reshapes and fills the
//! active prefix for each dispatched batch.
class PhaseBatchState
{
public:
    PhaseBatchState(int32_t maxBatchSize, std::string const& name);

    PhaseBatchState(PhaseBatchState const&) = delete;
    PhaseBatchState& operator=(PhaseBatchState const&) = delete;
    PhaseBatchState(PhaseBatchState&&) = delete;
    PhaseBatchState& operator=(PhaseBatchState&&) = delete;

    //! Bind kv_slot_ids and kvcache_start_index to stable member tensors.
    void bind(TensorMap& tensorMap);

    //! Upload batch slot IDs and gather physical lengths into the local view.
    void prepare(std::vector<PhaseWorkItem> const& batch, HybridCacheManager& cacheManager, cudaStream_t stream);

    //! Commit a scalar increment for every row in the current phase batch.
    void commit(HybridCacheManager& cacheManager, int32_t increment, cudaStream_t stream);

    //! Commit per-row increments for the current phase batch.
    void commit(HybridCacheManager& cacheManager, Tensor const& increments, cudaStream_t stream);

    Tensor& slotIds() noexcept;
    Tensor& lengths() noexcept;
    int32_t batchSize() const noexcept;

private:
    int32_t mMaxBatchSize{};
    int32_t mBatchSize{};
    Tensor mHostSlotIds;
    Tensor mDeviceSlotIds;
    Tensor mDeviceLengths;
};

} // namespace rt
} // namespace trt_edgellm
