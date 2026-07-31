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

#include <gtest/gtest.h>

namespace trt_edgellm
{
namespace rt
{

TEST(KVSlotAllocatorTest, StableSlotsSurviveMiddleEviction)
{
    KVSlotAllocator allocator(4);
    allocator.reset(4);
    allocator.compact({0, -1, 1, 2}, 3);
    EXPECT_EQ(allocator.activeSlots(), (std::vector<int32_t>{0, 2, 3}));
    EXPECT_EQ(allocator.available(), 1);
    EXPECT_EQ(allocator.reserve(), 1);
}

TEST(KVSlotAllocatorTest, DeterministicallyReusesLowestSlot)
{
    KVSlotAllocator allocator(4);
    allocator.reset(3);
    allocator.compact({-1, 0, -1}, 1);
    EXPECT_EQ(allocator.activeSlots(), (std::vector<int32_t>{1}));
    EXPECT_EQ(allocator.reserve(), 0);
    EXPECT_EQ(allocator.reserve(), 2);
}

TEST(KVSlotAllocatorTest, RejectsExhaustionAndDoubleFree)
{
    KVSlotAllocator allocator(1);
    allocator.reset(1);
    EXPECT_ANY_THROW(allocator.reserve());
    allocator.compact({-1}, 0);
    EXPECT_ANY_THROW(allocator.release(0));
}

TEST(KVSlotAllocatorTest, SupportsNonIdentityActiveMapping)
{
    KVSlotAllocator allocator(4);
    allocator.reset(4);
    allocator.compact({1, -1, 2, 0}, 3);
    EXPECT_EQ(allocator.activeSlots(), (std::vector<int32_t>{3, 0, 2}));
}

} // namespace rt
} // namespace trt_edgellm
