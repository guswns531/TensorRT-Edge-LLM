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

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using trt_edgellm::rt::KVPageBundleAllocator;
using trt_edgellm::rt::KVPagePrefixShare;

namespace
{

KVPageBundleAllocator makeAllocator(int32_t bundles = 8)
{
    return KVPageBundleAllocator({/*numSlots=*/4, /*numPageBundles=*/bundles,
        /*maxSequenceLength=*/512, /*tokensPerPage=*/128});
}

} // namespace

TEST(KVPageBundleAllocatorTest, AllocatesOnlyWhenCrossingPageBoundary)
{
    auto allocator = makeAllocator();

    EXPECT_EQ(allocator.getConfig().tokensPerPage, 128);
    EXPECT_EQ(allocator.maxPagesPerSequence(), 4);
    allocator.ensureCapacity(3, 1);
    EXPECT_EQ(allocator.bundles(3), (std::vector<int32_t>{0}));
    allocator.ensureCapacity(3, 128);
    EXPECT_EQ(allocator.bundles(3), (std::vector<int32_t>{0}));
    allocator.ensureCapacity(3, 129);
    EXPECT_EQ(allocator.bundles(3), (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(allocator.allocatedBundles(), 2);
}

TEST(KVPageBundleAllocatorTest, ReusesLowestReleasedBundleDeterministically)
{
    auto allocator = makeAllocator();

    allocator.ensureCapacity(0, 256);
    allocator.ensureCapacity(1, 128);
    EXPECT_EQ(allocator.bundles(0), (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(allocator.bundles(1), (std::vector<int32_t>{2}));

    allocator.release(0);
    allocator.ensureCapacity(2, 256);
    EXPECT_EQ(allocator.bundles(2), (std::vector<int32_t>{0, 1}));
}

TEST(KVPageBundleAllocatorTest, ExhaustionIsTransactional)
{
    auto allocator = makeAllocator(/*bundles=*/4);

    allocator.ensureCapacity(0, 256);
    allocator.ensureCapacity(1, 128);
    EXPECT_EQ(allocator.bundles(0), (std::vector<int32_t>{0, 1}));
    EXPECT_THROW(allocator.ensureCapacity(0, 512), std::runtime_error);
    EXPECT_EQ(allocator.bundles(0), (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(allocator.bundles(1), (std::vector<int32_t>{2}));
    EXPECT_EQ(allocator.availableBundles(), 1);
}

TEST(KVPageBundleAllocatorTest, BuildsKAndVPhysicalPageTable)
{
    auto allocator = makeAllocator();
    allocator.ensureCapacity(0, 256);
    allocator.ensureCapacity(2, 128);

    std::vector<int32_t> const table = allocator.makePhysicalPageTable();
    int32_t const stride = allocator.maxPagesPerSequence();
    ASSERT_EQ(table.size(), 4U * 2U * static_cast<size_t>(stride));

    EXPECT_EQ(std::vector<int32_t>(table.begin(), table.begin() + stride), (std::vector<int32_t>{0, 2, -1, -1}));
    EXPECT_EQ(
        std::vector<int32_t>(table.begin() + stride, table.begin() + 2 * stride), (std::vector<int32_t>{1, 3, -1, -1}));
    EXPECT_EQ(allocator.makePhysicalPageTableRow(0), (std::vector<int32_t>{0, 2, -1, -1, 1, 3, -1, -1}));

    int32_t const slot2Offset = 2 * 2 * stride;
    EXPECT_EQ(table[slot2Offset], 4);
    EXPECT_EQ(table[slot2Offset + stride], 5);
}

TEST(KVPageBundleAllocatorTest, SharesFullPrefixPagesUntilLastOwnerReleases)
{
    auto allocator = makeAllocator();
    allocator.ensureCapacity(0, 256);

    KVPagePrefixShare const share = allocator.sharePrefix(0, 1, 256);
    EXPECT_EQ(share.sharedBundles, 2);
    EXPECT_EQ(share.tailTokens, 0);
    EXPECT_EQ(allocator.bundles(1), allocator.bundles(0));
    EXPECT_EQ(allocator.bundleRefCount(0), 2);
    EXPECT_EQ(allocator.bundleRefCount(1), 2);
    EXPECT_EQ(allocator.allocatedBundles(), 2);

    allocator.release(0);
    EXPECT_EQ(allocator.allocatedBundles(), 2);
    allocator.release(1);
    EXPECT_EQ(allocator.allocatedBundles(), 0);
}

TEST(KVPageBundleAllocatorTest, GivesPartialPrefixTailPrivateOwnership)
{
    auto allocator = makeAllocator();
    allocator.ensureCapacity(0, 256);

    KVPagePrefixShare const share = allocator.sharePrefix(0, 1, 160);
    EXPECT_EQ(share.sharedBundles, 1);
    EXPECT_EQ(share.sourceTailBundle, 1);
    EXPECT_EQ(share.targetTailBundle, 2);
    EXPECT_EQ(share.tailTokens, 32);
    EXPECT_EQ(allocator.bundles(1), (std::vector<int32_t>{0, 2}));
    EXPECT_EQ(allocator.bundleRefCount(0), 2);
    EXPECT_EQ(allocator.bundleRefCount(1), 1);
    EXPECT_EQ(allocator.bundleRefCount(2), 1);

    allocator.release(0);
    allocator.ensureCapacity(2, 128);
    EXPECT_EQ(allocator.bundles(2), (std::vector<int32_t>{1}));
    EXPECT_EQ(allocator.bundles(1), (std::vector<int32_t>{0, 2}));
}

TEST(KVPageBundleAllocatorTest, RejectsInvalidLifecycleAndConfiguration)
{
    EXPECT_THROW((KVPageBundleAllocator({4, 3, 512, 128})), std::runtime_error);
    EXPECT_THROW((KVPageBundleAllocator({4, 8, 512, 64})), std::runtime_error);

    auto allocator = makeAllocator();
    EXPECT_THROW(allocator.release(0), std::runtime_error);
    EXPECT_THROW(allocator.ensureCapacity(-1, 1), std::runtime_error);
    EXPECT_THROW(allocator.ensureCapacity(0, 513), std::runtime_error);
}
