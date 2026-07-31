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

#include "runtime/kvSlotAllocator.h"

#include "common/checkMacros.h"

#include <utility>

namespace trt_edgellm
{
namespace rt
{

KVSlotAllocator::KVSlotAllocator(int32_t capacity)
    : mCapacity(capacity)
{
    check::check(capacity > 0, "KVSlotAllocator capacity must be positive.");
    reset(0);
}

void KVSlotAllocator::reset(int32_t batchSize)
{
    check::check(batchSize >= 0 && batchSize <= mCapacity, "KVSlotAllocator reset batch size is out of range.");
    mFreeSlots.clear();
    mActiveSlots.clear();
    for (int32_t slot = 0; slot < mCapacity; ++slot)
    {
        mFreeSlots.insert(slot);
    }
    mActiveSlots.reserve(batchSize);
    for (int32_t row = 0; row < batchSize; ++row)
    {
        mActiveSlots.push_back(reserve());
    }
}

int32_t KVSlotAllocator::reserve()
{
    check::check(!mFreeSlots.empty(), "KVSlotAllocator has no free physical slot.");
    auto const it = mFreeSlots.begin();
    int32_t const slot = *it;
    mFreeSlots.erase(it);
    return slot;
}

void KVSlotAllocator::release(int32_t slot)
{
    check::check(slot >= 0 && slot < mCapacity, "KVSlotAllocator release slot is out of range.");
    check::check(mFreeSlots.find(slot) == mFreeSlots.end(), "KVSlotAllocator detected a double free.");
    mFreeSlots.insert(slot);
}

void KVSlotAllocator::compact(std::vector<int32_t> const& batchMapping, int32_t newBatchSize)
{
    check::check(
        batchMapping.size() == mActiveSlots.size(), "KVSlotAllocator batch mapping size does not match active rows.");
    check::check(newBatchSize >= 0 && newBatchSize <= static_cast<int32_t>(mActiveSlots.size()),
        "KVSlotAllocator new batch size is out of range.");

    std::vector<int32_t> compacted(newBatchSize, -1);
    for (int32_t oldRow = 0; oldRow < static_cast<int32_t>(batchMapping.size()); ++oldRow)
    {
        int32_t const newRow = batchMapping[oldRow];
        int32_t const slot = mActiveSlots[oldRow];
        if (newRow < 0)
        {
            release(slot);
            continue;
        }
        check::check(newRow < newBatchSize, "KVSlotAllocator mapping target is out of range.");
        check::check(compacted[newRow] < 0, "KVSlotAllocator mapping contains a duplicate target row.");
        compacted[newRow] = slot;
    }
    for (int32_t slot : compacted)
    {
        check::check(slot >= 0, "KVSlotAllocator mapping leaves a hole in the active rows.");
    }
    mActiveSlots = std::move(compacted);
}

std::vector<int32_t> const& KVSlotAllocator::activeSlots() const noexcept
{
    return mActiveSlots;
}

int32_t KVSlotAllocator::available() const noexcept
{
    return static_cast<int32_t>(mFreeSlots.size());
}

} // namespace rt
} // namespace trt_edgellm
