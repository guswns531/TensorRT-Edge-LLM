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

#include "runtime/scheduling/phasePrefixReuseCache.h"

#include "common/checkMacros.h"

#include <algorithm>

namespace trt_edgellm::rt
{

PhasePrefixReuseCache::PhasePrefixReuseCache(StableKVPageManager& ownership, int32_t maxRecords)
    : mOwnership(ownership)
    , mMaxRecords(maxRecords)
{
    ELLM_CHECK(mMaxRecords >= 0, "Prefix reuse record limit must be non-negative");
}

PhasePrefixReuseCache::~PhasePrefixReuseCache() noexcept
{
    clear();
}

std::optional<PhasePrefixReuseMatch> PhasePrefixReuseCache::lookup(std::vector<int32_t> const& tokenIds) const
{
    if (tokenIds.size() <= 1U)
    {
        return std::nullopt;
    }
    PhasePrefixReuseMatch best;
    for (Entry const& entry : mEntries)
    {
        if (!mOwnership.leased(entry.sourceSlot))
        {
            continue;
        }
        size_t common{};
        size_t const maxCommon = std::min(tokenIds.size(), entry.tokenIds.size());
        while (common < maxCommon && tokenIds[common] == entry.tokenIds[common])
        {
            ++common;
        }
        int32_t const pageAligned
            = static_cast<int32_t>(common) / mOwnership.config().tokensPerPage * mOwnership.config().tokensPerPage;
        int32_t const usable = std::min(pageAligned, static_cast<int32_t>(tokenIds.size() - 1U));
        if (usable > best.matchedTokens)
        {
            best = {entry.sourceSlot, usable};
        }
    }
    if (best.matchedTokens == 0)
    {
        return std::nullopt;
    }
    return best;
}

void PhasePrefixReuseCache::publish(
    int32_t sourceSlot, std::vector<int32_t> const& tokenIds, int32_t materializedLength)
{
    ELLM_CHECK(mMaxRecords > 0, "Prefix reuse is disabled");
    ELLM_CHECK(mOwnership.leased(sourceSlot), "Prefix reuse source slot is not leased");
    ELLM_CHECK(materializedLength > 0 && materializedLength <= mOwnership.config().maxSequenceLength,
        "Prefix reuse materialized length is out of range");

    auto const same = std::find_if(
        mEntries.begin(), mEntries.end(), [&](Entry const& entry) { return entry.tokenIds == tokenIds; });
    if (same != mEntries.end())
    {
        if (same->sourceSlot != sourceSlot)
        {
            mOwnership.release(same->sourceSlot);
            same->sourceSlot = sourceSlot;
        }
        same->materializedLength = materializedLength;
        same->lastUse = ++mClock;
        return;
    }

    if (static_cast<int32_t>(mEntries.size()) >= mMaxRecords)
    {
        auto const oldest = std::min_element(mEntries.begin(), mEntries.end(),
            [](Entry const& lhs, Entry const& rhs) { return lhs.lastUse < rhs.lastUse; });
        mOwnership.release(oldest->sourceSlot);
        mEntries.erase(oldest);
    }
    mEntries.push_back({tokenIds, sourceSlot, materializedLength, ++mClock});
}

void PhasePrefixReuseCache::clear() noexcept
{
    for (Entry const& entry : mEntries)
    {
        if (mOwnership.leased(entry.sourceSlot))
        {
            mOwnership.release(entry.sourceSlot);
        }
    }
    mEntries.clear();
}

size_t PhasePrefixReuseCache::size() const noexcept
{
    return mEntries.size();
}

} // namespace trt_edgellm::rt
