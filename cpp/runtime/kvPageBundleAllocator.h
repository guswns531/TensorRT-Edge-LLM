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

//! Host-side ownership manager for a shared paged KV-cache pool.
//!
//! A bundle ID denotes the same token range in every attention layer. Each
//! layer expands one bundle into two physical XQA pages: K=`bundle * 2` and
//! V=`bundle * 2 + 1`. Stable request slots own ordered bundle-ID lists, but
//! the physical bundles do not need to be contiguous.
class KVPageBundleAllocator
{
public:
    //! \cond INTERNAL
    struct Config
    {
        int32_t numSlots{};          //!< Number of stable logical request slots
        int32_t numPageBundles{};    //!< Number of bundles in the shared pool
        int32_t maxSequenceLength{}; //!< Maximum tokens addressable by one slot
        int32_t tokensPerPage{128};  //!< Token granularity; fixed to 128 in paged KV v1
    };
    //! \endcond

    explicit KVPageBundleAllocator(Config const& config);

    //! Release every slot and restore the deterministic free-list.
    void reset();

    //! Ensure a slot owns enough bundles for `sequenceLength` tokens.
    //!
    //! Allocation is transactional: if the pool lacks capacity, no bundle is
    //! removed from the free-list and the slot's existing ownership is intact.
    void ensureCapacity(int32_t slot, int32_t sequenceLength);

    //! Release every bundle owned by a slot.
    void release(int32_t slot);

    //! Ordered bundle IDs for one stable slot.
    std::vector<int32_t> const& bundles(int32_t slot) const;

    //! Flattened XQA page table [numSlots, 2, maxPagesPerSequence].
    //!
    //! Unallocated entries contain -1. K and V page IDs are derived from one
    //! bundle ID so allocation remains atomic across both cache planes.
    std::vector<int32_t> makePhysicalPageTable() const;

    //! One flattened XQA page-table row [2, maxPagesPerSequence].
    //!
    //! Runtime dispatch uploads this smaller view after a slot grows instead
    //! of rebuilding or copying the table for every stable slot.
    std::vector<int32_t> makePhysicalPageTableRow(int32_t slot) const;

    int32_t availableBundles() const noexcept;
    int32_t allocatedBundles() const noexcept;
    int32_t maxPagesPerSequence() const noexcept;
    Config const& getConfig() const noexcept;

private:
    void validateSlot(int32_t slot) const;
    int32_t pagesForLength(int32_t sequenceLength) const;

    Config mConfig{};
    int32_t mMaxPagesPerSequence{};
    std::set<int32_t> mFreeBundles;
    std::vector<std::vector<int32_t>> mSlotBundles;
};

} // namespace rt
} // namespace trt_edgellm
