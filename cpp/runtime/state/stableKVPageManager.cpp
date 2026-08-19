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

#include "runtime/state/stableKVPageManager.h"

#include "common/checkMacros.h"

#include <unordered_set>

namespace trt_edgellm
{
namespace rt
{

StableKVPageManager::StableKVPageManager(Config const& config)
    : mConfig(config)
{
    ELLM_CHECK(mConfig.maxStableSlots > 0, "Stable KV slot count must be positive");
    ELLM_CHECK(mConfig.maxActiveRows > 0 && mConfig.maxActiveRows <= mConfig.maxStableSlots,
        "Stable KV active-row capacity must be positive and no larger than the stable-slot count");
    ELLM_CHECK(mConfig.numPages > 0, "Stable KV page count must be positive");
    ELLM_CHECK(mConfig.maxSequenceLength > 0, "Stable KV maximum sequence length must be positive");
    ELLM_CHECK(mConfig.tokensPerPage == 128, "Stable KV v1 requires 128 tokens per page");

    mMaxPagesPerSequence = (mConfig.maxSequenceLength + mConfig.tokensPerPage - 1) / mConfig.tokensPerPage;
    ELLM_CHECK(mConfig.numPages >= mMaxPagesPerSequence, "Stable KV page pool cannot hold one maximum-length sequence");

    for (int32_t slot = 0; slot < mConfig.maxStableSlots; ++slot)
    {
        mFreeSlots.insert(slot);
    }
    for (int32_t page = 0; page < mConfig.numPages; ++page)
    {
        mFreePages.insert(page);
    }
    mLeased.assign(static_cast<size_t>(mConfig.maxStableSlots), 0U);
    mLengths.assign(static_cast<size_t>(mConfig.maxStableSlots), 0);
    mSlotPages.resize(static_cast<size_t>(mConfig.maxStableSlots));
    mPageRefCounts.assign(static_cast<size_t>(mConfig.numPages), 0);
}

int32_t StableKVPageManager::reserve()
{
    ELLM_CHECK(!mFreeSlots.empty(), "Stable KV slot pool is exhausted");
    auto const slotIt = mFreeSlots.begin();
    int32_t const slot = *slotIt;
    mFreeSlots.erase(slotIt);
    mLeased[static_cast<size_t>(slot)] = 1U;
    mLengths[static_cast<size_t>(slot)] = 0;
    return slot;
}

void StableKVPageManager::release(int32_t stableSlot)
{
    validateLease(stableSlot);
    auto& ownedPages = mSlotPages[static_cast<size_t>(stableSlot)];
    for (int32_t const page : ownedPages)
    {
        int32_t& refCount = mPageRefCounts[static_cast<size_t>(page)];
        ELLM_CHECK(refCount > 0, "Stable KV page has invalid reference count");
        --refCount;
        if (refCount == 0)
        {
            bool const inserted = mFreePages.insert(page).second;
            ELLM_CHECK(inserted, "Stable KV page has duplicate ownership");
        }
    }
    ownedPages.clear();
    mLengths[static_cast<size_t>(stableSlot)] = 0;
    mLeased[static_cast<size_t>(stableSlot)] = 0U;
    bool const inserted = mFreeSlots.insert(stableSlot).second;
    ELLM_CHECK(inserted, "Stable KV slot allocator detected a double free");
}

void StableKVPageManager::ensureCapacity(int32_t stableSlot, int32_t sequenceLength)
{
    validateLease(stableSlot);
    int32_t const requiredPages = pagesForLength(sequenceLength);
    auto& ownedPages = mSlotPages[static_cast<size_t>(stableSlot)];
    int32_t const missingPages = requiredPages - static_cast<int32_t>(ownedPages.size());
    if (missingPages <= 0)
    {
        return;
    }
    ELLM_CHECK(static_cast<int32_t>(mFreePages.size()) >= missingPages,
        "Stable KV page pool is exhausted; allocation was not modified");

    std::vector<int32_t> reserved;
    reserved.reserve(static_cast<size_t>(missingPages));
    auto pageIt = mFreePages.begin();
    for (int32_t page = 0; page < missingPages; ++page, ++pageIt)
    {
        reserved.push_back(*pageIt);
    }
    for (int32_t const page : reserved)
    {
        mFreePages.erase(page);
        ++mPageRefCounts[static_cast<size_t>(page)];
        ownedPages.push_back(page);
    }
}

void StableKVPageManager::sharePrefix(int32_t sourceSlot, int32_t targetSlot, int32_t prefixLength)
{
    validateLease(sourceSlot);
    validateLease(targetSlot);
    ELLM_CHECK(sourceSlot != targetSlot, "Stable KV prefix source and target must differ");
    ELLM_CHECK(prefixLength >= 0 && prefixLength <= mLengths[static_cast<size_t>(sourceSlot)],
        "Stable KV shared prefix exceeds the source length");
    ELLM_CHECK(
        prefixLength % mConfig.tokensPerPage == 0, "Stable KV shared prefix must end on a physical page boundary");
    auto& targetPages = mSlotPages[static_cast<size_t>(targetSlot)];
    ELLM_CHECK(targetPages.empty(), "Stable KV prefix target must not own pages");
    int32_t const sharedPages = pagesForLength(prefixLength);
    auto const& sourcePages = mSlotPages[static_cast<size_t>(sourceSlot)];
    ELLM_CHECK(sharedPages <= static_cast<int32_t>(sourcePages.size()),
        "Stable KV source does not contain enough pages for the shared prefix");
    targetPages.assign(sourcePages.begin(), sourcePages.begin() + sharedPages);
    for (int32_t const page : targetPages)
    {
        ELLM_CHECK(mPageRefCounts[static_cast<size_t>(page)] > 0, "Stable KV source page is not referenced");
        ++mPageRefCounts[static_cast<size_t>(page)];
    }
    mLengths[static_cast<size_t>(targetSlot)] = prefixLength;
}

void StableKVPageManager::setLength(int32_t stableSlot, int32_t length)
{
    validateLease(stableSlot);
    ELLM_CHECK(length >= 0 && length <= mConfig.maxSequenceLength, "Stable KV length is out of range");
    ELLM_CHECK(pagesForLength(length) <= static_cast<int32_t>(mSlotPages[static_cast<size_t>(stableSlot)].size()),
        "Stable KV length exceeds the slot's physical page capacity");
    mLengths[static_cast<size_t>(stableSlot)] = length;
}

int32_t StableKVPageManager::length(int32_t stableSlot) const
{
    validateLease(stableSlot);
    return mLengths[static_cast<size_t>(stableSlot)];
}

bool StableKVPageManager::bindActiveRows(
    std::vector<int32_t> const& activeStableSlots, KVPageTable& pageTable, cudaStream_t stream) const
{
    ELLM_CHECK(static_cast<int32_t>(activeStableSlots.size()) <= mConfig.maxActiveRows,
        "Stable KV active batch exceeds its configured row capacity");
    ELLM_CHECK(pageTable.kernelView().getShape()[0] == mConfig.maxActiveRows,
        "Stable KV page table active-row capacity does not match");
    ELLM_CHECK(
        pageTable.maxPagesPerSeq() == mMaxPagesPerSequence, "Stable KV page table sequence capacity does not match");
    ELLM_CHECK(pageTable.numPages() == mConfig.numPages, "Stable KV page table pool size does not match");

    std::unordered_set<int32_t> uniqueSlots;
    std::vector<KVPageTableRowUpdate> updates;
    updates.reserve(static_cast<size_t>(mConfig.maxActiveRows));
    for (int32_t row = 0; row < mConfig.maxActiveRows; ++row)
    {
        if (row >= static_cast<int32_t>(activeStableSlots.size()))
        {
            updates.push_back({row, nullptr, 0});
            continue;
        }
        int32_t const stableSlot = activeStableSlots[static_cast<size_t>(row)];
        validateLease(stableSlot);
        ELLM_CHECK(uniqueSlots.insert(stableSlot).second, "Stable KV active rows contain a duplicate slot");
        auto const& ownedPages = mSlotPages[static_cast<size_t>(stableSlot)];
        updates.push_back(
            {row, ownedPages.empty() ? nullptr : ownedPages.data(), static_cast<int32_t>(ownedPages.size())});
    }
    pageTable.setRows(updates);
    return pageTable.upload(stream);
}

std::vector<int32_t> StableKVPageManager::makeActiveLengths(std::vector<int32_t> const& activeStableSlots) const
{
    ELLM_CHECK(static_cast<int32_t>(activeStableSlots.size()) <= mConfig.maxActiveRows,
        "Stable KV active length batch exceeds its configured row capacity");
    std::unordered_set<int32_t> uniqueSlots;
    std::vector<int32_t> lengths;
    lengths.reserve(activeStableSlots.size());
    for (int32_t const stableSlot : activeStableSlots)
    {
        validateLease(stableSlot);
        ELLM_CHECK(uniqueSlots.insert(stableSlot).second, "Stable KV active lengths contain a duplicate slot");
        lengths.push_back(mLengths[static_cast<size_t>(stableSlot)]);
    }
    return lengths;
}

std::vector<int32_t> const& StableKVPageManager::pages(int32_t stableSlot) const
{
    validateLease(stableSlot);
    return mSlotPages[static_cast<size_t>(stableSlot)];
}

bool StableKVPageManager::leased(int32_t stableSlot) const
{
    validateSlot(stableSlot);
    return mLeased[static_cast<size_t>(stableSlot)] != 0U;
}

int32_t StableKVPageManager::availableSlots() const noexcept
{
    return static_cast<int32_t>(mFreeSlots.size());
}

int32_t StableKVPageManager::availablePages() const noexcept
{
    return static_cast<int32_t>(mFreePages.size());
}

int32_t StableKVPageManager::maxPagesPerSequence() const noexcept
{
    return mMaxPagesPerSequence;
}

StableKVPageManager::Config const& StableKVPageManager::config() const noexcept
{
    return mConfig;
}

void StableKVPageManager::validateSlot(int32_t stableSlot) const
{
    ELLM_CHECK(stableSlot >= 0 && stableSlot < mConfig.maxStableSlots, "Stable KV slot is out of range");
}

void StableKVPageManager::validateLease(int32_t stableSlot) const
{
    validateSlot(stableSlot);
    ELLM_CHECK(leased(stableSlot), "Stable KV slot is not leased");
}

int32_t StableKVPageManager::pagesForLength(int32_t sequenceLength) const
{
    ELLM_CHECK(sequenceLength >= 0 && sequenceLength <= mConfig.maxSequenceLength,
        "Stable KV sequence length is out of range");
    return (sequenceLength + mConfig.tokensPerPage - 1) / mConfig.tokensPerPage;
}

} // namespace rt
} // namespace trt_edgellm
