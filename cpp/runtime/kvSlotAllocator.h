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

#include <cstdint>
#include <set>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Host-side stable physical-slot leases for indexed-linear KV cache.
class KVSlotAllocator
{
public:
    explicit KVSlotAllocator(int32_t capacity);

    //! Release all leases and reserve one slot for each logical row.
    void reset(int32_t batchSize);

    //! Reserve the lowest-numbered available physical slot.
    int32_t reserve();

    //! Release a physical slot.
    void release(int32_t slot);

    //! Apply old-row -> new-row eviction mapping without moving physical storage.
    void compact(std::vector<int32_t> const& batchMapping, int32_t newBatchSize);

    std::vector<int32_t> const& activeSlots() const noexcept;
    int32_t available() const noexcept;

private:
    int32_t mCapacity{};
    std::set<int32_t> mFreeSlots;
    std::vector<int32_t> mActiveSlots;
};

} // namespace rt
} // namespace trt_edgellm
