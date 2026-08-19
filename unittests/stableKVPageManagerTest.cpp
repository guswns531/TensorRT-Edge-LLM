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

#include <gtest/gtest.h>

using namespace trt_edgellm;

namespace
{

rt::StableKVPageManager makeManager(int32_t pages = 12)
{
    return rt::StableKVPageManager({4, 3, pages, 512, 128});
}

} // namespace

TEST(StableKVPageManagerTest, ReusesReleasedSlotsAndPagesDeterministically)
{
    auto manager = makeManager();
    int32_t const slot0 = manager.reserve();
    int32_t const slot1 = manager.reserve();
    manager.ensureCapacity(slot0, 256);
    manager.ensureCapacity(slot1, 128);
    EXPECT_EQ(manager.pages(slot0), (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(manager.pages(slot1), (std::vector<int32_t>{2}));

    manager.release(slot0);
    int32_t const reusedSlot = manager.reserve();
    EXPECT_EQ(reusedSlot, slot0);
    manager.ensureCapacity(reusedSlot, 256);
    EXPECT_EQ(manager.pages(reusedSlot), (std::vector<int32_t>{0, 1}));
}

TEST(StableKVPageManagerTest, ExhaustionIsTransactional)
{
    auto manager = makeManager(4);
    int32_t const slot0 = manager.reserve();
    int32_t const slot1 = manager.reserve();
    manager.ensureCapacity(slot0, 256);
    manager.ensureCapacity(slot1, 128);
    EXPECT_THROW(manager.ensureCapacity(slot0, 512), std::runtime_error);
    EXPECT_EQ(manager.pages(slot0), (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(manager.pages(slot1), (std::vector<int32_t>{2}));
    EXPECT_EQ(manager.availablePages(), 1);
}

TEST(StableKVPageManagerTest, ReordersActiveRowsWithoutMovingPhysicalPages)
{
    auto manager = makeManager();
    int32_t const slot0 = manager.reserve();
    int32_t const slot1 = manager.reserve();
    int32_t const slot2 = manager.reserve();
    manager.ensureCapacity(slot0, 128);
    manager.ensureCapacity(slot1, 256);
    manager.ensureCapacity(slot2, 128);
    manager.setLength(slot0, 64);
    manager.setLength(slot1, 192);
    manager.setLength(slot2, 96);

    rt::KVPageTable pageTable(3, 4, 12);
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    EXPECT_TRUE(manager.bindActiveRows({slot2, slot0}, pageTable, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    EXPECT_EQ(
        std::vector<int32_t>(pageTable.hostRow(0), pageTable.hostRow(0) + 4), (std::vector<int32_t>{3, -1, -1, -1}));
    EXPECT_EQ(
        std::vector<int32_t>(pageTable.hostRow(1), pageTable.hostRow(1) + 4), (std::vector<int32_t>{0, -1, -1, -1}));
    EXPECT_EQ(
        std::vector<int32_t>(pageTable.hostRow(2), pageTable.hostRow(2) + 4), (std::vector<int32_t>{-1, -1, -1, -1}));
    EXPECT_EQ(manager.makeActiveLengths({slot2, slot0}), (std::vector<int32_t>{96, 64}));
    EXPECT_EQ(manager.pages(slot1), (std::vector<int32_t>{1, 2}));

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(StableKVPageManagerTest, RejectsInvalidLeaseAndDuplicateActiveRows)
{
    auto manager = makeManager();
    int32_t const slot = manager.reserve();
    rt::KVPageTable pageTable(3, 4, 12);
    EXPECT_THROW(manager.bindActiveRows({slot, slot}, pageTable, nullptr), std::runtime_error);
    manager.release(slot);
    EXPECT_THROW(manager.release(slot), std::runtime_error);
    EXPECT_THROW(manager.ensureCapacity(slot, 1), std::runtime_error);
}
