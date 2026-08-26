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

#include "runtime/state/kvPageTable.h"

#include <cstdint>
#include <set>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Stable request ownership over the v0.10 paged-KV pool.
//!
//! A stable slot survives active-batch row reordering. Physical K page IDs are
//! owned by stable slots, while bindActiveRows() materializes the transient
//! active-row page table consumed by TensorRT. Eviction therefore updates only
//! host ownership and page-table rows; it never copies KV payloads.
class StableKVPageManager
{
public:
    //! \cond INTERNAL
    struct Config
    {
        int32_t maxStableSlots{};
        int32_t maxActiveRows{};
        int32_t numPages{};
        int32_t maxSequenceLength{};
        int32_t tokensPerPage{128};
    };
    //! \endcond

    explicit StableKVPageManager(Config const& config);

    //! Reserve the lowest free stable slot.
    int32_t reserve();

    //! Release a slot and every physical page owned by it.
    void release(int32_t stableSlot);

    //! Transactionally grow one stable slot to cover sequenceLength tokens.
    void ensureCapacity(int32_t stableSlot, int32_t sequenceLength);

    //! Share complete physical pages from sourceSlot into an empty target slot.
    //! The prefix must be page aligned so subsequent suffix writes cannot
    //! mutate a page still referenced by the source request.
    void sharePrefix(int32_t sourceSlot, int32_t targetSlot, int32_t prefixLength);

    //! Set and query the committed global length of one stable slot.
    void setLength(int32_t stableSlot, int32_t length);
    int32_t length(int32_t stableSlot) const;

    //! Materialize active row -> stable slot ownership in a v0.10 KVPageTable.
    //! Rows not present in activeStableSlots are cleared.
    bool bindActiveRows(
        std::vector<int32_t> const& activeStableSlots, KVPageTable& pageTable, cudaStream_t stream) const;

    //! Gather committed lengths in active-row order.
    std::vector<int32_t> makeActiveLengths(std::vector<int32_t> const& activeStableSlots) const;

    std::vector<int32_t> const& pages(int32_t stableSlot) const;
    //! Monotonic identity of the current lease; changes when a slot is reused.
    uint64_t leaseGeneration(int32_t stableSlot) const;
    bool leased(int32_t stableSlot) const;
    int32_t availableSlots() const noexcept;
    int32_t availablePages() const noexcept;
    int32_t maxPagesPerSequence() const noexcept;
    Config const& config() const noexcept;

private:
    void validateSlot(int32_t stableSlot) const;
    void validateLease(int32_t stableSlot) const;
    int32_t pagesForLength(int32_t sequenceLength) const;

    Config mConfig{};
    int32_t mMaxPagesPerSequence{};
    std::set<int32_t> mFreeSlots;
    std::set<int32_t> mFreePages;
    std::vector<uint8_t> mLeased;
    std::vector<uint64_t> mLeaseGenerations;
    std::vector<int32_t> mLengths;
    std::vector<std::vector<int32_t>> mSlotPages;
    std::vector<int32_t> mPageRefCounts;
};

} // namespace rt
} // namespace trt_edgellm
