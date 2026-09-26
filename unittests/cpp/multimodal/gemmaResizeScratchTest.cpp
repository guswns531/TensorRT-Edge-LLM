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

#include "multimodal/gemma4/gemma4ViTRunner.h"

#include <gtest/gtest.h>
#include <limits>
#include <stdexcept>

using trt_edgellm::rt::Gemma4ResizeScratch;

TEST(Gemma4ResizeScratchRequirementsTest, ComputesActualRawAndHorizontalPassBytes)
{
    auto const required = Gemma4ResizeScratch::requirements(1365, 2048, 3, 624, 960);
    EXPECT_EQ(required.rawBytes, 8386560);
    EXPECT_EQ(required.temporaryBytes, 15724800);
}

TEST(Gemma4ResizeScratchRequirementsTest, IdentityCopyNeedsNoScratch)
{
    auto const required = Gemma4ResizeScratch::requirements(8192, 1, 3, 8192, 1);
    EXPECT_EQ(required.rawBytes, 0);
    EXPECT_EQ(required.temporaryBytes, 0);
}

TEST(Gemma4ResizeScratchRequirementsTest, PreservesMaximumRawDimension)
{
    auto const required = Gemma4ResizeScratch::requirements(4096, 4096, 3, 768, 768);
    EXPECT_EQ(required.rawBytes, 48 * 1024 * 1024);
    EXPECT_EQ(required.temporaryBytes, 36 * 1024 * 1024);
    EXPECT_THROW(Gemma4ResizeScratch::requirements(4097, 4096, 3, 768, 768), std::runtime_error);
    EXPECT_THROW(Gemma4ResizeScratch::requirements(4096, 4097, 3, 768, 768), std::runtime_error);
}

TEST(Gemma4ResizeScratchRequirementsTest, RejectsInvalidDimensionsAndOverflow)
{
    EXPECT_THROW(Gemma4ResizeScratch::requirements(0, 16, 3, 8, 8), std::runtime_error);
    EXPECT_THROW(Gemma4ResizeScratch::requirements(16, 16, 0, 8, 8), std::runtime_error);
    EXPECT_THROW(Gemma4ResizeScratch::requirements(16, 16, 3, 0, 8), std::runtime_error);
    EXPECT_THROW(
        Gemma4ResizeScratch::requirements(16, 16, std::numeric_limits<int64_t>::max(), 8, 8), std::runtime_error);
    EXPECT_THROW(
        Gemma4ResizeScratch::requirements(16, 16, 3, 8, std::numeric_limits<int64_t>::max()), std::runtime_error);
}

class Gemma4ResizeScratchGpuTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int deviceCount{};
        if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        {
            GTEST_SKIP() << "CUDA device is unavailable";
        }
        int device{};
        ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
        ASSERT_EQ(cudaDeviceGetAttribute(&mPoolsSupported, cudaDevAttrMemoryPoolsSupported, device), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&mFirst, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&mSecond, cudaStreamNonBlocking), cudaSuccess);
    }

    void TearDown() override
    {
        if (mFirst != nullptr)
        {
            EXPECT_EQ(cudaStreamDestroy(mFirst), cudaSuccess);
        }
        if (mSecond != nullptr)
        {
            EXPECT_EQ(cudaStreamDestroy(mSecond), cudaSuccess);
        }
    }

    cudaStream_t mFirst{};
    cudaStream_t mSecond{};
    int mPoolsSupported{};
};

TEST_F(Gemma4ResizeScratchGpuTest, ReusesSmallLargeSmallAcrossStreams)
{
    if (mPoolsSupported == 0)
    {
        GTEST_SKIP() << "CUDA memory pools are unavailable";
    }
    Gemma4ResizeScratch scratch;
    scratch.initialize(3, 645120, true);
    EXPECT_EQ(scratch.metrics().allocatedHighWaterBytes, 0);
    auto const small = Gemma4ResizeScratch::requirements(8, 8, 3, 4, 4);
    auto const large = Gemma4ResizeScratch::requirements(32, 64, 3, 16, 32);
    scratch.reserve(small, mFirst);
    ASSERT_EQ(cudaMemsetAsync(scratch.raw().rawPointer(), 0x11, small.rawBytes, mFirst), cudaSuccess);
    scratch.recordUse(mFirst);
    scratch.reserve(large, mSecond);
    auto* const largePointer = scratch.raw().rawPointer();
    ASSERT_EQ(cudaMemsetAsync(largePointer, 0x5A, large.rawBytes, mSecond), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(scratch.temporary().rawPointer(), 0x5B, large.temporaryBytes, mSecond), cudaSuccess);
    scratch.recordUse(mSecond);
    scratch.reserve(small, mFirst);
    EXPECT_EQ(scratch.raw().rawPointer(), largePointer);
    EXPECT_EQ(scratch.metrics().growthCount, 2);
    EXPECT_EQ(scratch.metrics().allocatedHighWaterBytes, large.rawBytes + large.temporaryBytes);
    trt_edgellm::rt::Tensor host({large.rawBytes}, trt_edgellm::rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    ASSERT_EQ(
        cudaMemcpyAsync(host.rawPointer(), largePointer, large.rawBytes, cudaMemcpyDeviceToHost, mFirst), cudaSuccess);
    scratch.recordUse(mFirst);
    ASSERT_EQ(cudaStreamSynchronize(mFirst), cudaSuccess);
    for (int64_t i = 0; i < large.rawBytes; ++i)
    {
        ASSERT_EQ(host.dataPointer<unsigned char>()[i], 0x5A);
    }
}

TEST_F(Gemma4ResizeScratchGpuTest, FailedSecondAllocationPreservesBothOldBuffers)
{
    if (mPoolsSupported == 0)
    {
        GTEST_SKIP() << "CUDA memory pools are unavailable";
    }
    Gemma4ResizeScratch scratch;
    scratch.initialize(3, 645120, true);
    auto const small = Gemma4ResizeScratch::requirements(8, 8, 3, 4, 4);
    scratch.reserve(small, mFirst);
    auto* const oldRaw = scratch.raw().rawPointer();
    auto* const oldTemporary = scratch.temporary().rawPointer();
    size_t freeBytes{};
    size_t totalBytes{};
    ASSERT_EQ(cudaMemGetInfo(&freeBytes, &totalBytes), cudaSuccess);
    int64_t const impossibleBytes = static_cast<int64_t>(2 * totalBytes);
    EXPECT_THROW(scratch.reserve({small.rawBytes * 2, impossibleBytes}, mSecond), std::runtime_error);
    static_cast<void>(cudaGetLastError());
    EXPECT_EQ(scratch.raw().rawPointer(), oldRaw);
    EXPECT_EQ(scratch.temporary().rawPointer(), oldTemporary);
    EXPECT_EQ(scratch.metrics().rawAllocatedBytes, small.rawBytes);
    EXPECT_EQ(scratch.metrics().temporaryAllocatedBytes, small.temporaryBytes);
    EXPECT_EQ(scratch.metrics().growthCount, 1);
    scratch.reserve(small, mFirst);
    ASSERT_EQ(cudaMemsetAsync(oldRaw, 0, small.rawBytes, mFirst), cudaSuccess);
    scratch.recordUse(mFirst);
}

TEST_F(Gemma4ResizeScratchGpuTest, StartupFallbackUsesOwnedFixedCapacity)
{
    Gemma4ResizeScratch scratch;
    scratch.initialize(3, 645120, false);
    EXPECT_FALSE(scratch.metrics().streamOrdered);
    EXPECT_EQ(scratch.metrics().rawAllocatedBytes, 48 * 1024 * 1024);
    auto* const pointer = scratch.raw().rawPointer();
    scratch.reserve(Gemma4ResizeScratch::requirements(1365, 2048, 3, 624, 960), mFirst);
    scratch.recordUse(mFirst);
    EXPECT_EQ(pointer, scratch.raw().rawPointer());
    EXPECT_EQ(scratch.metrics().growthCount, 0);
}

TEST_F(Gemma4ResizeScratchGpuTest, EarlyFailureOnAnotherStreamPreservesPreviousCompletion)
{
    if (mPoolsSupported == 0)
    {
        GTEST_SKIP() << "CUDA memory pools are unavailable";
    }
    auto const required = Gemma4ResizeScratch::requirements(8, 8, 3, 4, 4);
    trt_edgellm::rt::Tensor host({required.rawBytes}, trt_edgellm::rt::DeviceType::kCPU, nvinfer1::DataType::kUINT8);
    {
        Gemma4ResizeScratch scratch;
        scratch.initialize(3, 645120, true);
        scratch.reserve(required, mFirst);
        ASSERT_EQ(cudaMemsetAsync(scratch.raw().rawPointer(), 0x5C, required.rawBytes, mFirst), cudaSuccess);
        scratch.recordUse(mFirst);
        EXPECT_THROW(scratch.reserve({-1, 0}, mSecond), std::runtime_error);
        scratch.recordUse(mSecond);
        ASSERT_EQ(cudaMemcpyAsync(host.rawPointer(), scratch.raw().rawPointer(), required.rawBytes,
                      cudaMemcpyDeviceToHost, mSecond),
            cudaSuccess);
        scratch.recordUse(mSecond);
    }
    for (int64_t i = 0; i < required.rawBytes; ++i)
    {
        ASSERT_EQ(host.dataPointer<unsigned char>()[i], 0x5C);
    }
}

TEST_F(Gemma4ResizeScratchGpuTest, CompletionEventOutlivesCallerStream)
{
    if (mPoolsSupported == 0)
    {
        GTEST_SKIP() << "CUDA memory pools are unavailable";
    }
    Gemma4ResizeScratch scratch;
    scratch.initialize(3, 645120, true);
    auto const required = Gemma4ResizeScratch::requirements(8, 8, 3, 4, 4);
    scratch.reserve(required, mFirst);
    ASSERT_EQ(cudaMemsetAsync(scratch.raw().rawPointer(), 0, required.rawBytes, mFirst), cudaSuccess);
    scratch.recordUse(mFirst);
    ASSERT_EQ(cudaStreamDestroy(mFirst), cudaSuccess);
    mFirst = nullptr;
}
