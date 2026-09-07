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

#include "runtime/state/stableKVPageManager.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm::rt
{

//! A stable-slot source and page-aligned prefix length selected for reuse.
struct PhasePrefixReuseMatch
{
    int32_t sourceSlot{-1};
    int32_t matchedTokens{};
};

//! Process-local token-prefix index whose records retain stable KV slots.
//!
//! The cache is deliberately small and single-writer. A published record owns
//! its source slot until eviction, while target requests share the referenced
//! physical pages through StableKVPageManager's page reference counts.
class PhasePrefixReuseCache
{
public:
    PhasePrefixReuseCache(StableKVPageManager& ownership, int32_t maxRecords);
    ~PhasePrefixReuseCache() noexcept;

    PhasePrefixReuseCache(PhasePrefixReuseCache const&) = delete;
    PhasePrefixReuseCache& operator=(PhasePrefixReuseCache const&) = delete;

    std::optional<PhasePrefixReuseMatch> lookup(std::vector<int32_t> const& tokenIds) const;
    void publish(int32_t sourceSlot, std::vector<int32_t> const& tokenIds, int32_t materializedLength);
    void clear() noexcept;
    size_t size() const noexcept;

private:
    struct Entry
    {
        std::vector<int32_t> tokenIds;
        int32_t sourceSlot{-1};
        int32_t materializedLength{};
        uint64_t lastUse{};
    };

    StableKVPageManager& mOwnership;
    int32_t mMaxRecords{};
    uint64_t mClock{};
    std::vector<Entry> mEntries;
};

} // namespace trt_edgellm::rt
