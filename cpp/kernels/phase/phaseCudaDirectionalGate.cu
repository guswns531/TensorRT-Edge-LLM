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

#include "runtime/scheduling/phaseCudaDirectionalGate.h"

#include "common/checkMacros.h"

namespace trt_edgellm::rt
{
namespace
{

__global__ void delayDirectionalGate(uint64_t cycles)
{
    uint64_t const started = clock64();
    while (clock64() - started < cycles)
    {
    }
}

} // namespace

PhaseCudaDirectionalGate::~PhaseCudaDirectionalGate() noexcept
{
    wait();
    if (mGateDone != nullptr)
    {
        static_cast<void>(cudaEventDestroy(mGateDone));
    }
    if (mGateStream != nullptr)
    {
        static_cast<void>(cudaStreamDestroy(mGateStream));
    }
}

void PhaseCudaDirectionalGate::ensureStorage()
{
    if (mGateStream != nullptr)
    {
        return;
    }
    int leastPriority{};
    int greatestPriority{};
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&mGateStream, cudaStreamNonBlocking, greatestPriority));
    CUDA_CHECK(cudaEventCreateWithFlags(&mGateDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaDeviceGetAttribute(&mClockRateKHz, cudaDevAttrClockRate, 0));
}

void PhaseCudaDirectionalGate::enqueue(cudaStream_t newcomerStream, cudaEvent_t incumbentStart, uint64_t delayUs)
{
    wait();
    ensureStorage();
    CUDA_CHECK(cudaStreamWaitEvent(mGateStream, incumbentStart));
    uint64_t const cycles = delayUs * static_cast<uint64_t>(mClockRateKHz) / 1000U;
    delayDirectionalGate<<<1, 1, 0, mGateStream>>>(cycles);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(mGateDone, mGateStream));
    CUDA_CHECK(cudaStreamWaitEvent(newcomerStream, mGateDone));
}

void PhaseCudaDirectionalGate::wait() noexcept
{
    if (mGateStream != nullptr)
    {
        static_cast<void>(cudaStreamSynchronize(mGateStream));
    }
}

} // namespace trt_edgellm::rt
