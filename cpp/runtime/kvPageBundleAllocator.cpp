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

#include "runtime/kvPageBundleAllocator.h"

#include "common/checkMacros.h"

#include <algorithm>

namespace trt_edgellm
{
namespace rt
{

KVPageBundleAllocator::KVPageBundleAllocator(Config const& config)
    : mConfig(config)
{
    check::check(mConfig.numSlots > 0, "KV page-bundle slot count must be positive.");
    check::check(mConfig.numPageBundles > 0, "KV page-bundle pool size must be positive.");
    check::check(mConfig.maxSequenceLength > 0, "KV page-bundle maximum sequence length must be positive.");
    check::check(mConfig.tokensPerPage == 128, "Paged KV v1 requires exactly 128 tokens per page.");
    check::check(mConfig.maxSequenceLength % mConfig.tokensPerPage == 0,
        "Paged KV v1 requires maximum sequence length divisible by 128.");

    mMaxPagesPerSequence = mConfig.maxSequenceLength / mConfig.tokensPerPage;
    check::check(mConfig.numPageBundles >= mMaxPagesPerSequence,
        "KV page-bundle pool cannot hold one maximum-length sequence.");
    reset();
}

void KVPageBundleAllocator::reset()
{
    mFreeBundles.clear();
    for (int32_t page = 0; page < mConfig.numPageBundles; ++page)
    {
        mFreeBundles.insert(page);
    }
    mSlotBundles.assign(mConfig.numSlots, {});
}

void KVPageBundleAllocator::ensureCapacity(int32_t slot, int32_t sequenceLength)
{
    validateSlot(slot);
    int32_t const requiredPages = pagesForLength(sequenceLength);
    auto& owned = mSlotBundles[slot];
    int32_t const missingPages = requiredPages - static_cast<int32_t>(owned.size());
    if (missingPages <= 0)
    {
        return;
    }

    check::check(static_cast<int32_t>(mFreeBundles.size()) >= missingPages,
        "KV page-bundle pool is exhausted; allocation was not modified.");

    std::vector<int32_t> reserved;
    reserved.reserve(missingPages);
    auto it = mFreeBundles.begin();
    for (int32_t page = 0; page < missingPages; ++page)
    {
        reserved.push_back(*it);
        ++it;
    }
    for (int32_t page : reserved)
    {
        mFreeBundles.erase(page);
        owned.push_back(page);
    }
}

void KVPageBundleAllocator::release(int32_t slot)
{
    validateSlot(slot);
    auto& owned = mSlotBundles[slot];
    check::check(!owned.empty(), "KV page-bundle allocator detected a release of an empty slot.");
    for (int32_t page : owned)
    {
        bool const inserted = mFreeBundles.insert(page).second;
        check::check(inserted, "KV page-bundle allocator detected duplicate physical ownership.");
    }
    owned.clear();
}

std::vector<int32_t> const& KVPageBundleAllocator::bundles(int32_t slot) const
{
    validateSlot(slot);
    return mSlotBundles[slot];
}

std::vector<int32_t> KVPageBundleAllocator::makePhysicalPageTable() const
{
    constexpr int32_t kKV_PLANES = 2;
    std::vector<int32_t> pageTable(
        static_cast<size_t>(mConfig.numSlots) * kKV_PLANES * mMaxPagesPerSequence, -1);
    for (int32_t slot = 0; slot < mConfig.numSlots; ++slot)
    {
        std::vector<int32_t> const row = makePhysicalPageTableRow(slot);
        auto const offset = pageTable.begin() + static_cast<size_t>(slot * kKV_PLANES * mMaxPagesPerSequence);
        std::copy(row.begin(), row.end(), offset);
    }
    return pageTable;
}

std::vector<int32_t> KVPageBundleAllocator::makePhysicalPageTableRow(int32_t slot) const
{
    constexpr int32_t kKV_PLANES = 2;
    validateSlot(slot);
    std::vector<int32_t> row(static_cast<size_t>(kKV_PLANES * mMaxPagesPerSequence), -1);
    auto const& owned = mSlotBundles[slot];
    for (int32_t page = 0; page < static_cast<int32_t>(owned.size()); ++page)
    {
        int32_t const bundle = owned[page];
        row[page] = bundle * kKV_PLANES;
        row[mMaxPagesPerSequence + page] = bundle * kKV_PLANES + 1;
    }
    return row;
}

int32_t KVPageBundleAllocator::availableBundles() const noexcept
{
    return static_cast<int32_t>(mFreeBundles.size());
}

int32_t KVPageBundleAllocator::allocatedBundles() const noexcept
{
    return mConfig.numPageBundles - availableBundles();
}

int32_t KVPageBundleAllocator::maxPagesPerSequence() const noexcept
{
    return mMaxPagesPerSequence;
}

KVPageBundleAllocator::Config const& KVPageBundleAllocator::getConfig() const noexcept
{
    return mConfig;
}

void KVPageBundleAllocator::validateSlot(int32_t slot) const
{
    check::check(slot >= 0 && slot < mConfig.numSlots, "KV page-bundle slot is out of range.");
}

int32_t KVPageBundleAllocator::pagesForLength(int32_t sequenceLength) const
{
    check::check(sequenceLength >= 0 && sequenceLength <= mConfig.maxSequenceLength,
        "KV page-bundle sequence length is out of range.");
    return (sequenceLength + mConfig.tokensPerPage - 1) / mConfig.tokensPerPage;
}

} // namespace rt
} // namespace trt_edgellm
