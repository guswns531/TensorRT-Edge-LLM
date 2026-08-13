/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Unit tests for cvtKVLayoutBHSDToSplitKV.
//
// The kernel converts a KV-cache tensor [B, 2, H, S, D] into two separate
// tensors kDst [B, S, H, D] and vDst [B, S, H, D].  Two test strategies:
//
//   1. CPU reference  – fill src with known values on the host, compute the
//      expected K/V layout by hand, compare against GPU output.
//   2. FP8 dequant    – verify that scale factors are applied correctly when
//      the source is FP8 (compiled in only when SUPPORTS_FP8 == 1).

#include "common/cudaMacros.h"
#include "common/cudaUtils.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "testUtils.h"

#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace trt_edgellm;
using namespace nvinfer1;

namespace
{

// Returns the flat index into a [B, 2, H, S, D] src tensor.
size_t srcIdx(int32_t b, int32_t kv, int32_t h, int32_t s, int32_t d, int32_t H, int32_t S, int32_t D)
{
    return (((((size_t) b * 2 + kv) * H + h) * S + s) * D + d);
}

size_t indexedSrcIdx(int32_t slot, int32_t kv, int32_t h, int32_t s, int32_t d, int32_t H, int32_t S, int32_t D)
{
    constexpr int32_t kTOKENS_PER_PAGE = 128;
    int32_t const pagesPerSequence = S / kTOKENS_PER_PAGE;
    int32_t const page = (slot * 2 + kv) * pagesPerSequence + s / kTOKENS_PER_PAGE;
    int32_t const tokenInPage = s % kTOKENS_PER_PAGE;
    return (((static_cast<size_t>(page) * kTOKENS_PER_PAGE + tokenInPage) * H + h) * D + d);
}

// Returns the flat index into a [B, S, H, D] dst tensor.
size_t dstIdx(int32_t b, int32_t s, int32_t h, int32_t d, int32_t S, int32_t H, int32_t D)
{
    return ((((size_t) b * S + s) * H + h) * D + d);
}

struct SplitKVParams
{
    int32_t B, H, S, D;
};

} // namespace

// ===== 1. CPU reference test (FP16) ==============================================

class SplitKVCpuReferenceTest : public ::testing::TestWithParam<SplitKVParams>
{
};

TEST_P(SplitKVCpuReferenceTest, MatchesCpuReference)
{
    auto [B, H, S, D] = GetParam();

    cudaStream_t stream{nullptr};

    // Fill source on the host with random FP16 values.
    size_t const srcVol = (size_t) B * 2 * H * S * D;
    std::vector<half> srcHost(srcVol);
    uniformFloatInitialization(srcHost, -4.f, 4.f);

    // Upload to device.
    rt::Tensor srcTensor({B, 2, H, S, D}, rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemcpy(srcTensor.rawPointer(), srcHost.data(), srcVol * sizeof(half), cudaMemcpyHostToDevice));

    // Allocate output tensors.
    rt::Tensor kTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor emptyScale{};
    kernel::cvtKVLayoutBHSDToSplitKV(srcTensor, kTensor, vTensor, emptyScale, S, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pull results back to host.
    size_t const dstVol = (size_t) B * S * H * D;
    std::vector<half> kHost(dstVol), vHost(dstVol);
    CUDA_CHECK(cudaMemcpy(kHost.data(), kTensor.rawPointer(), dstVol * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vHost.data(), vTensor.rawPointer(), dstVol * sizeof(half), cudaMemcpyDeviceToHost));

    // Verify against CPU reference.
    auto const [rtol, atol] = getTolerance<half>();
    for (int32_t b = 0; b < B; ++b)
        for (int32_t s = 0; s < S; ++s)
            for (int32_t h = 0; h < H; ++h)
                for (int32_t d = 0; d < D; ++d)
                {
                    half expectedK = srcHost[srcIdx(b, 0, h, s, d, H, S, D)];
                    half expectedV = srcHost[srcIdx(b, 1, h, s, d, H, S, D)];
                    size_t outIdx = dstIdx(b, s, h, d, S, H, D);

                    ASSERT_TRUE(isclose(kHost[outIdx], expectedK, rtol, atol))
                        << "K mismatch at b=" << b << " s=" << s << " h=" << h << " d=" << d
                        << ": got=" << __half2float(kHost[outIdx]) << " expected=" << __half2float(expectedK);

                    ASSERT_TRUE(isclose(vHost[outIdx], expectedV, rtol, atol))
                        << "V mismatch at b=" << b << " s=" << s << " h=" << h << " d=" << d
                        << ": got=" << __half2float(vHost[outIdx]) << " expected=" << __half2float(expectedV);
                }
}

INSTANTIATE_TEST_SUITE_P(SplitKVShapes, SplitKVCpuReferenceTest,
    ::testing::Values(SplitKVParams{1, 4, 16, 64}, // small single-batch
        SplitKVParams{2, 8, 32, 128},              // multi-batch, head-size 128
        SplitKVParams{4, 2, 64, 64},               // larger sequence
        SplitKVParams{1, 1, 8, 64}                 // minimal dims
        ));

TEST(SplitKVIndexedTest, GathersStablePhysicalSlots)
{
    int32_t const physicalB = 4;
    int32_t const activeB = 3;
    int32_t const H = 8;
    int32_t const S = 256;
    int32_t const D = 8;
    std::vector<int32_t> const slotIds{3, 0, 2};

    size_t const srcVol = static_cast<size_t>(physicalB) * 2 * H * S * D;
    std::vector<half> srcHost(srcVol);
    for (int32_t b = 0; b < physicalB; ++b)
    {
        for (int32_t kv = 0; kv < 2; ++kv)
        {
            for (int32_t h = 0; h < H; ++h)
            {
                for (int32_t s = 0; s < S; ++s)
                {
                    for (int32_t d = 0; d < D; ++d)
                    {
                        float const value
                            = static_cast<float>(1000 * b + 100 * kv + 10 * h + s % 10) + static_cast<float>(d) / 16.0F;
                        srcHost[indexedSrcIdx(b, kv, h, s, d, H, S, D)] = __float2half(value);
                    }
                }
            }
        }
    }

    rt::Tensor backing({physicalB, 2, H, S, D}, rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemcpy(backing.rawPointer(), srcHost.data(), srcVol * sizeof(half), cudaMemcpyHostToDevice));
    rt::Tensor activeView(backing.rawPointer(), rt::Coords{activeB, 2, H, S, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor slotTensor({activeB}, rt::DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(
        cudaMemcpy(slotTensor.rawPointer(), slotIds.data(), slotIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    rt::Tensor kTensor({activeB, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({activeB, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);

    kernel::cvtKVLayoutBHSDToSplitKV(
        activeView, kTensor, vTensor, rt::Tensor{}, S, nullptr, slotTensor.dataPointer<int32_t>());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const kHost = copyDeviceToHost<half>(kTensor);
    auto const vHost = copyDeviceToHost<half>(vTensor);
    for (int32_t b = 0; b < activeB; ++b)
    {
        for (int32_t s = 0; s < S; ++s)
        {
            for (int32_t h = 0; h < H; ++h)
            {
                for (int32_t d = 0; d < D; ++d)
                {
                    size_t const out = dstIdx(b, s, h, d, S, H, D);
                    EXPECT_EQ(kHost[out], srcHost[indexedSrcIdx(slotIds[b], 0, h, s, d, H, S, D)]);
                    EXPECT_EQ(vHost[out], srcHost[indexedSrcIdx(slotIds[b], 1, h, s, d, H, S, D)]);
                }
            }
        }
    }
}

TEST(SplitKVIndexedTest, GathersVariablePrefixesIntoPackedWorkspace)
{
    int32_t constexpr physicalSlots = 4;
    int32_t constexpr activeRows = 3;
    int32_t constexpr numHeads = 2;
    int32_t constexpr capacity = 256;
    int32_t constexpr headDim = 8;
    int32_t constexpr tokensPerPage = 128;
    int32_t constexpr pagesPerSequence = capacity / tokensPerPage;
    int32_t constexpr physicalPages = physicalSlots * 2 * pagesPerSequence;
    std::vector<int32_t> const slotIds{3, 0, 2};
    std::vector<int32_t> const rowLengths{130, 3, 129};
    std::vector<int32_t> const cuKVSeqLens{0, 130, 133, 262};
    std::vector<int32_t> const pageIds{
        6,
        7,
        8,
        9,
        4,
        5,
        14,
        15,
        0,
        1,
        2,
        3,
        10,
        11,
        12,
        13,
    };

    size_t const srcVolume = static_cast<size_t>(physicalPages) * tokensPerPage * numHeads * headDim;
    std::vector<half> srcHost(srcVolume);
    for (int32_t page = 0; page < physicalPages; ++page)
    {
        for (int32_t token = 0; token < tokensPerPage; ++token)
        {
            for (int32_t head = 0; head < numHeads; ++head)
            {
                for (int32_t dim = 0; dim < headDim; ++dim)
                {
                    size_t const index
                        = (((static_cast<size_t>(page) * tokensPerPage + token) * numHeads + head) * headDim + dim);
                    srcHost[index] = __float2half(
                        static_cast<float>(1000 * page + 10 * token + 2 * head) + static_cast<float>(dim) / 16.0F);
                }
            }
        }
    }

    rt::Tensor backing({physicalSlots, 2, numHeads, capacity, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    CUDA_CHECK(cudaMemcpy(backing.rawPointer(), srcHost.data(), srcHost.size() * sizeof(half), cudaMemcpyHostToDevice));
    rt::Tensor activeView(
        backing.rawPointer(), {activeRows, 2, numHeads, capacity, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor slotTensor({activeRows}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor pageTensor({physicalSlots, 2, pagesPerSequence}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor({activeRows + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(
        cudaMemcpy(slotTensor.rawPointer(), slotIds.data(), slotIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(pageTensor.rawPointer(), pageIds.data(), pageIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuKVSeqLensTensor.rawPointer(), cuKVSeqLens.data(), cuKVSeqLens.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));

    rt::Tensor kTensor({activeRows * capacity, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({activeRows * capacity, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    kernel::gatherKVCacheToPackedSplitKV(activeView, kTensor, vTensor, cuKVSeqLensTensor, nullptr,
        slotTensor.dataPointer<int32_t>(), pageTensor.dataPointer<int32_t>());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto const kHost = copyDeviceToHost<half>(kTensor);
    auto const vHost = copyDeviceToHost<half>(vTensor);
    for (int32_t row = 0; row < activeRows; ++row)
    {
        int32_t const slot = slotIds[row];
        for (int32_t token = 0; token < rowLengths[row]; ++token)
        {
            int32_t const logicalPage = token / tokensPerPage;
            int32_t const tokenInPage = token % tokensPerPage;
            for (int32_t kv = 0; kv < 2; ++kv)
            {
                int32_t const page = pageIds[(slot * 2 + kv) * pagesPerSequence + logicalPage];
                for (int32_t head = 0; head < numHeads; ++head)
                {
                    for (int32_t dim = 0; dim < headDim; ++dim)
                    {
                        size_t const srcIndex
                            = (((static_cast<size_t>(page) * tokensPerPage + tokenInPage) * numHeads + head) * headDim
                                + dim);
                        size_t const dstIndex
                            = ((static_cast<size_t>(cuKVSeqLens[row] + token) * numHeads + head) * headDim + dim);
                        EXPECT_EQ(kv == 0 ? kHost[dstIndex] : vHost[dstIndex], srcHost[srcIndex]);
                    }
                }
            }
        }
    }
}

TEST(SplitKVIndexedTest, PackedPrefixGatherBenchmark)
{
    int32_t constexpr batchSize = 8;
    int32_t constexpr numHeads = 8;
    int32_t constexpr capacity = 2048;
    int32_t constexpr headDim = 128;
    int32_t constexpr tokensPerPage = 128;
    int32_t constexpr pagesPerSequence = capacity / tokensPerPage;
    int32_t constexpr maxPrefixLength = 1024;
    std::vector<int32_t> const slotIds{0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int32_t> const prefixLengths{128, 256, 384, 512, 640, 768, 896, 1024};
    std::vector<int32_t> cuKVSeqLens(batchSize + 1, 0);
    for (int32_t row = 0; row < batchSize; ++row)
    {
        cuKVSeqLens[row + 1] = cuKVSeqLens[row] + prefixLengths[row];
    }
    std::vector<int32_t> pageIds(batchSize * 2 * pagesPerSequence);
    for (int32_t page = 0; page < static_cast<int32_t>(pageIds.size()); ++page)
    {
        pageIds[page] = page;
    }

    rt::Tensor cache({batchSize, 2, numHeads, capacity, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor slotTensor({batchSize}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor pageTensor({batchSize, 2, pagesPerSequence}, rt::DeviceType::kGPU, DataType::kINT32);
    rt::Tensor cuKVSeqLensTensor({batchSize + 1}, rt::DeviceType::kGPU, DataType::kINT32);
    CUDA_CHECK(
        cudaMemcpy(slotTensor.rawPointer(), slotIds.data(), slotIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(pageTensor.rawPointer(), pageIds.data(), pageIds.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(cuKVSeqLensTensor.rawPointer(), cuKVSeqLens.data(), cuKVSeqLens.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice));

    rt::Tensor paddedK({batchSize, maxPrefixLength, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor paddedV({batchSize, maxPrefixLength, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor packedK({batchSize * capacity, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor packedV({batchSize * capacity, numHeads, headDim}, rt::DeviceType::kGPU, DataType::kHALF);
    cudaStream_t stream{nullptr};
    auto launchPadded = [&]() {
        kernel::cvtKVLayoutBHSDToSplitKV(cache, paddedK, paddedV, rt::Tensor{}, maxPrefixLength, stream,
            slotTensor.dataPointer<int32_t>(), pageTensor.dataPointer<int32_t>());
    };
    auto launchPacked = [&]() {
        kernel::gatherKVCacheToPackedSplitKV(cache, packedK, packedV, cuKVSeqLensTensor, stream,
            slotTensor.dataPointer<int32_t>(), pageTensor.dataPointer<int32_t>());
    };

    int32_t constexpr warmupIterations = 20;
    int32_t constexpr benchmarkIterations = 100;
    for (int32_t iteration = 0; iteration < warmupIterations; ++iteration)
    {
        launchPadded();
        launchPacked();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    auto measure = [&](auto const& launch) {
        CUDA_CHECK(cudaEventRecord(startEvent, stream));
        for (int32_t iteration = 0; iteration < benchmarkIterations; ++iteration)
        {
            launch();
        }
        CUDA_CHECK(cudaEventRecord(stopEvent, stream));
        CUDA_CHECK(cudaEventSynchronize(stopEvent));
        float elapsedMs{0.0F};
        CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent));
        return elapsedMs / benchmarkIterations;
    };
    float const paddedMs = measure(launchPadded);
    float const packedMs = measure(launchPacked);
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    std::cout << "Packed prefix gather benchmark: B=" << batchSize << " Hkv=" << numHeads << " D=" << headDim
              << " capacity=" << capacity << " exact_tokens=" << cuKVSeqLens.back()
              << " padded_tokens=" << batchSize * maxPrefixLength << " padded_ms=" << paddedMs
              << " packed_ms=" << packedMs << std::endl;
}

// ===== 2. FP8 dequantization test =================================================

#if SUPPORTS_FP8
TEST(SplitKVFP8Test, DequantizesWithScale)
{
    // Small shape for a targeted FP8 → FP16 dequant check.
    int32_t const B = 1, H = 2, S = 4, D = 8;

    cudaStream_t stream{nullptr};

    size_t const srcVol = (size_t) B * 2 * H * S * D;

    // Build FP8 source from known FP32 values so we can predict the dequant output.
    std::vector<float> srcFP32(srcVol);
    uniformFloatInitialization(srcFP32, 0.5f, 2.f); // positive range avoids FP8 sign edge cases

    std::vector<__nv_fp8_e4m3> srcFP8(srcVol);
    for (size_t i = 0; i < srcVol; ++i)
        srcFP8[i] = __nv_fp8_e4m3(srcFP32[i]);

    rt::Tensor srcTensor({B, 2, H, S, D}, rt::DeviceType::kGPU, DataType::kFP8);
    CUDA_CHECK(
        cudaMemcpy(srcTensor.rawPointer(), srcFP8.data(), srcVol * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));

    // Scale factors: K scale = 2.0, V scale = 0.5
    float const kScale = 2.f, vScale = 0.5f;
    std::vector<float> scalesHost = {kScale, vScale};
    rt::Tensor scaleTensor({2}, rt::DeviceType::kGPU, DataType::kFLOAT);
    CUDA_CHECK(cudaMemcpy(scaleTensor.rawPointer(), scalesHost.data(), 2 * sizeof(float), cudaMemcpyHostToDevice));

    rt::Tensor kTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    rt::Tensor vTensor({B, S, H, D}, rt::DeviceType::kGPU, DataType::kHALF);
    kernel::cvtKVLayoutBHSDToSplitKV(srcTensor, kTensor, vTensor, scaleTensor, S, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    size_t const dstVol = (size_t) B * S * H * D;
    std::vector<half> kHost(dstVol), vHost(dstVol);
    CUDA_CHECK(cudaMemcpy(kHost.data(), kTensor.rawPointer(), dstVol * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vHost.data(), vTensor.rawPointer(), dstVol * sizeof(half), cudaMemcpyDeviceToHost));

    // Expected: dequant(fp8_value) * scale, where dequant recovers the stored fp8 value.
    // FP8 → float → scale → half.  Use loose tolerance to account for FP8 quantization error.
    float const rtol = 0.1f, atol = 0.05f;
    for (int32_t b = 0; b < B; ++b)
        for (int32_t s = 0; s < S; ++s)
            for (int32_t h = 0; h < H; ++h)
                for (int32_t d = 0; d < D; ++d)
                {
                    size_t const outIdx = dstIdx(b, s, h, d, S, H, D);
                    float const fp8ValK = static_cast<float>(srcFP8[srcIdx(b, 0, h, s, d, H, S, D)]);
                    float const fp8ValV = static_cast<float>(srcFP8[srcIdx(b, 1, h, s, d, H, S, D)]);

                    ASSERT_TRUE(isclose(kHost[outIdx], __float2half(fp8ValK * kScale), rtol, atol))
                        << "FP8 K dequant mismatch at b=" << b << " s=" << s << " h=" << h << " d=" << d;
                    ASSERT_TRUE(isclose(vHost[outIdx], __float2half(fp8ValV * vScale), rtol, atol))
                        << "FP8 V dequant mismatch at b=" << b << " s=" << s << " h=" << h << " d=" << d;
                }
}
#endif // SUPPORTS_FP8
