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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernels/decodeAttentionKernels/decoderXQARunner.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void check(cudaError_t result)
{
    if (result != cudaSuccess)
    {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

class Buffer
{
public:
    explicit Buffer(size_t bytes)
    {
        check(cudaMalloc(&mDevice, bytes));
        try
        {
            check(cudaMallocHost(&mHost, bytes));
        }
        catch (...)
        {
            cudaFree(mDevice);
            throw;
        }
    }
    ~Buffer()
    {
        cudaFreeHost(mHost);
        cudaFree(mDevice);
    }
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;
    void* device() const
    {
        return mDevice;
    }
    void* host() const
    {
        return mHost;
    }

private:
    void* mDevice{};
    void* mHost{};
};

void fillHalf(Buffer& buffer, size_t count, uint32_t seed, cudaStream_t stream)
{
    auto* values = static_cast<half*>(buffer.host());
    for (size_t index{}; index < count; ++index)
    {
        seed = seed * 1664525U + 1013904223U;
        values[index] = __float2half(static_cast<float>(static_cast<int32_t>(seed >> 21) - 1024) / 1024.F);
    }
    check(cudaMemcpyAsync(buffer.device(), buffer.host(), count * sizeof(half), cudaMemcpyHostToDevice, stream));
}

void run(int32_t batch, int32_t length, std::string const& output, cudaStream_t stream)
{
    constexpr int32_t kQ_HEADS{16};
    constexpr int32_t kKV_HEADS{8};
    constexpr int32_t kHEAD_SIZE{128};
    constexpr int32_t kPAGE_TOKENS{128};
    int32_t const pagesPerRow = (length + kPAGE_TOKENS - 1) / kPAGE_TOKENS;
    int32_t const capacity = pagesPerRow * kPAGE_TOKENS;
    int32_t const pages = batch * 2 * pagesPerRow;
    size_t const qElements = static_cast<size_t>(batch) * kQ_HEADS * kHEAD_SIZE;
    size_t const kvElements = static_cast<size_t>(pages) * kPAGE_TOKENS * kKV_HEADS * kHEAD_SIZE;
    Buffer query(qElements * sizeof(half));
    Buffer cache(kvElements * sizeof(half));
    Buffer result(qElements * sizeof(half));
    Buffer table(static_cast<size_t>(pages) * sizeof(int32_t));
    Buffer lengths(static_cast<size_t>(batch) * sizeof(int32_t));
    fillHalf(query, qElements, 17U, stream);
    fillHalf(cache, kvElements, 31U, stream);
    auto* pageIds = static_cast<int32_t*>(table.host());
    for (int32_t index{}; index < pages; ++index)
    {
        pageIds[index] = pages - 1 - index;
    }
    auto* contextLengths = static_cast<int32_t*>(lengths.host());
    for (int32_t index{}; index < batch; ++index)
    {
        contextLengths[index] = std::max(1, length - index % 17);
    }
    check(cudaMemcpyAsync(
        table.device(), table.host(), static_cast<size_t>(pages) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(lengths.device(), lengths.host(), static_cast<size_t>(batch) * sizeof(int32_t),
        cudaMemcpyHostToDevice, stream));
    check(cudaMemsetAsync(result.device(), 0, qElements * sizeof(half), stream));
    check(cudaStreamSynchronize(stream));

    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0));
    int32_t const sm = properties.major * 10 + properties.minor;
#if defined(EDGELLM_PROBE_JIT)
    trt_edgellm::XQAJitKey const key{sm, nvinfer1::DataType::kHALF, nvinfer1::DataType::kHALF, kHEAD_SIZE,
        kQ_HEADS / kKV_HEADS, kPAGE_TOKENS, false, false};
    auto const compiled = trt_edgellm::compileXQAKernel(key);
    if (!trt_edgellm::DecoderXQARunner::loadDecodeXQAKernelFromCubin(key, compiled.cubin.data(), compiled.cubin.size()))
    {
        throw std::runtime_error("Cannot load XQA JIT kernel");
    }
#endif
    trt_edgellm::DecoderXQARunner runner(
        nvinfer1::DataType::kHALF, nvinfer1::DataType::kHALF, batch, kQ_HEADS, kKV_HEADS, kHEAD_SIZE, sm);
    auto params = runner.initXQAParams();
    params.qInputPtr = query.device();
    params.output = result.device();
    params.kvCache.data = cache.device();
    params.kvCache.pageList = static_cast<int32_t*>(table.device());
    params.kvCache.sequence_lengths = static_cast<int32_t*>(lengths.device());
    params.kvCache.capacity = capacity;
    params.kvCache.tokensPerPage = kPAGE_TOKENS;
    params.attentionScale = 1.F / std::sqrt(static_cast<float>(kHEAD_SIZE));
    for (int32_t index{}; index < 20; ++index)
    {
        runner.dispatchXQAKernel(params, stream);
    }
    cudaEvent_t begin{}, end{};
    check(cudaEventCreate(&begin));
    check(cudaEventCreate(&end));
    std::vector<float> times;
    for (int32_t index{}; index < 100; ++index)
    {
        check(cudaEventRecord(begin, stream));
        runner.dispatchXQAKernel(params, stream);
        check(cudaEventRecord(end, stream));
        check(cudaEventSynchronize(end));
        float ms{};
        check(cudaEventElapsedTime(&ms, begin, end));
        times.push_back(ms);
    }
    check(cudaMemcpyAsync(result.host(), result.device(), qElements * sizeof(half), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    std::ofstream file(output, std::ios::binary);
    file.write(static_cast<char const*>(result.host()), static_cast<std::streamsize>(qElements * sizeof(half)));
    if (!file)
    {
        throw std::runtime_error("Cannot write probe output");
    }
    std::sort(times.begin(), times.end());
    std::cout << "XQA_PROBE," << batch << ',' << length << ',' << (times[49] + times[50]) / 2.F << ','
              << times[94] + 0.05F * (times[95] - times[94]) << '\n';
    check(cudaEventDestroy(begin));
    check(cudaEventDestroy(end));
}
} // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: xqaParityProbe BATCH CONTEXT OUTPUT_FP16\n";
        return 2;
    }
    cudaStream_t stream{};
    try
    {
        int32_t const batch = std::stoi(argv[1]);
        int32_t const length = std::stoi(argv[2]);
        if (batch < 1 || batch > 64 || length < 1 || length > 2048)
        {
            throw std::runtime_error("Probe shape out of range");
        }
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        run(batch, length, argv[3], stream);
        check(cudaStreamDestroy(stream));
    }
    catch (std::exception const& error)
    {
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
