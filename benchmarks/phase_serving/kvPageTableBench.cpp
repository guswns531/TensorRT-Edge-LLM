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

#include "common/checkMacros.h"
#include "runtime/state/kvPageTable.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

int main()
{
    using namespace trt_edgellm;
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    for (int32_t const batch : {1, 8, 32, 64})
    {
        for (int32_t const dirtyMode : {0, 1, 2})
        {
            rt::KVPageTable table(batch, 16, 256);
            std::vector<double> samples;
            size_t uploads{};
            int32_t const dirtyRows = dirtyMode == 2 ? batch : dirtyMode;
            for (int32_t step{}; step < 1100; ++step)
            {
                auto const start = std::chrono::steady_clock::now();
                for (int32_t row{}; row < dirtyRows; ++row)
                {
                    int32_t const page = (step % 2) * 128 + row;
                    table.setRow(row, &page, 1);
                }
                bool const uploaded = table.upload(stream);
                auto const stop = std::chrono::steady_clock::now();
                if (step >= 100)
                {
                    uploads += uploaded;
                    samples.push_back(std::chrono::duration<double, std::micro>(stop - start).count());
                }
                // Bound outstanding updates without forcing every upload to complete on the host.
                if (step % 16 == 15)
                {
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                }
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::vector<int32_t> device(static_cast<size_t>(batch) * 32);
            CUDA_CHECK(cudaMemcpyAsync(device.data(), table.kernelView().rawPointer(), device.size() * sizeof(int32_t),
                cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int32_t row{}; row < dirtyRows; ++row)
            {
                ELLM_CHECK(device[row * 32] == 128 + row, "Final K page differs from committed row");
                ELLM_CHECK(device[row * 32 + 16] == 384 + row, "Final V page differs from committed row");
            }
            std::sort(samples.begin(), samples.end());
            std::printf("KV_TABLE batch=%d dirty_rows=%d median_us=%.3f p95_us=%.3f uploads=%zu\n", batch, dirtyRows,
                samples[500], samples[949], uploads);
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
