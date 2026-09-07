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

#include <gtest/gtest.h>

#include <numeric>

using namespace trt_edgellm;

namespace
{

std::vector<int32_t> makeTokens(int32_t count)
{
    std::vector<int32_t> tokens(static_cast<size_t>(count));
    std::iota(tokens.begin(), tokens.end(), 1);
    return tokens;
}

} // namespace

TEST(PhasePrefixReuseTest, SharesPageReferencesWithoutCopyingOrPrematureFree)
{
    rt::StableKVPageManager manager({4, 3, 8, 512, 128});
    int32_t const source = manager.reserve();
    int32_t const target = manager.reserve();
    manager.ensureCapacity(source, 256);
    manager.setLength(source, 256);
    int32_t const pagesBefore = manager.availablePages();

    manager.sharePrefix(source, target, 256);
    EXPECT_EQ(manager.pages(source), manager.pages(target));
    EXPECT_EQ(manager.availablePages(), pagesBefore);

    manager.release(source);
    EXPECT_TRUE(manager.leased(target));
    EXPECT_EQ(manager.availablePages(), pagesBefore);
    manager.release(target);
    EXPECT_EQ(manager.availablePages(), 8);
}

TEST(PhasePrefixReuseTest, FindsOnlyPageAlignedCommonPrefix)
{
    rt::StableKVPageManager manager({4, 3, 8, 512, 128});
    rt::PhasePrefixReuseCache cache(manager, 2);
    int32_t const source = manager.reserve();
    manager.ensureCapacity(source, 256);
    manager.setLength(source, 256);
    std::vector<int32_t> const prefix = makeTokens(256);
    cache.publish(source, prefix, 256);

    std::vector<int32_t> query = prefix;
    query.push_back(999);
    auto const match = cache.lookup(query);
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match->sourceSlot, source);
    EXPECT_EQ(match->matchedTokens, 256);

    query[127] = 1000;
    EXPECT_FALSE(cache.lookup(query).has_value());
    cache.clear();
    EXPECT_EQ(manager.availableSlots(), 4);
}
