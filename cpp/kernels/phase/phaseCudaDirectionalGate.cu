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

#include <chrono>
#include <thread>

namespace trt_edgellm::rt
{

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
    if (mHostGateFlag != nullptr)
    {
        static_cast<void>(cudaFreeHost(mHostGateFlag));
    }
    if (mHostIncumbentFlag != nullptr)
    {
        static_cast<void>(cudaFreeHost(mHostIncumbentFlag));
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
    CUDA_CHECK(cudaHostAlloc(&mHostGateFlag, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&mDeviceGateFlag, mHostGateFlag, 0U));
    CUDA_CHECK(cudaHostAlloc(&mHostIncumbentFlag, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&mDeviceIncumbentFlag, mHostIncumbentFlag, 0U));
    __atomic_store_n(mHostGateFlag, 0U, __ATOMIC_RELEASE);
    __atomic_store_n(mHostIncumbentFlag, 0U, __ATOMIC_RELEASE);
}

void CUDART_CB PhaseCudaDirectionalGate::releaseHostGate(void* userData)
{
    auto& gate = *static_cast<PhaseCudaDirectionalGate*>(userData);
    if (gate.mPendingDelayUs > 0U)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(gate.mPendingDelayUs));
    }
    __atomic_store_n(gate.mHostGateFlag, 1U, __ATOMIC_RELEASE);
}

void PhaseCudaDirectionalGate::enqueue(cudaStream_t newcomerStream, cudaEvent_t incumbentStart, uint64_t delayUs)
{
    wait();
    ensureStorage();
    __atomic_store_n(mHostGateFlag, 0U, __ATOMIC_RELEASE);
    mPendingDelayUs = delayUs;
    CUDA_CHECK(cudaStreamWaitEvent(mGateStream, incumbentStart));
    CUDA_CHECK(cudaLaunchHostFunc(mGateStream, releaseHostGate, this));
    CUDA_CHECK(cudaEventRecord(mGateDone, mGateStream));
    CUstream const driverStream = reinterpret_cast<CUstream>(newcomerStream);
    CUdeviceptr const gateAddress = reinterpret_cast<CUdeviceptr>(mDeviceGateFlag);
    CUDA_DRIVER_CHECK(cuStreamWaitValue32(driverStream, gateAddress, 1U, CU_STREAM_WAIT_VALUE_EQ));
}

void PhaseCudaDirectionalGate::arm(cudaStream_t newcomerStream, uint64_t delayUs)
{
    wait();
    ensureStorage();
    __atomic_store_n(mHostGateFlag, 0U, __ATOMIC_RELEASE);
    __atomic_store_n(mHostIncumbentFlag, 0U, __ATOMIC_RELEASE);
    mPendingDelayUs = delayUs;
    CUstream const gateStream = reinterpret_cast<CUstream>(mGateStream);
    CUdeviceptr const incumbentAddress = reinterpret_cast<CUdeviceptr>(mDeviceIncumbentFlag);
    CUDA_DRIVER_CHECK(cuStreamWaitValue32(gateStream, incumbentAddress, 1U, CU_STREAM_WAIT_VALUE_EQ));
    CUDA_CHECK(cudaLaunchHostFunc(mGateStream, releaseHostGate, this));
    CUDA_CHECK(cudaEventRecord(mGateDone, mGateStream));
    CUstream const driverStream = reinterpret_cast<CUstream>(newcomerStream);
    CUdeviceptr const gateAddress = reinterpret_cast<CUdeviceptr>(mDeviceGateFlag);
    CUDA_DRIVER_CHECK(cuStreamWaitValue32(driverStream, gateAddress, 1U, CU_STREAM_WAIT_VALUE_EQ));
}

void PhaseCudaDirectionalGate::signal(cudaStream_t incumbentStream)
{
    ELLM_CHECK(mDeviceIncumbentFlag != nullptr, "Directional gate must be armed before it is signalled");
    CUstream const driverStream = reinterpret_cast<CUstream>(incumbentStream);
    CUdeviceptr const incumbentAddress = reinterpret_cast<CUdeviceptr>(mDeviceIncumbentFlag);
    CUDA_DRIVER_CHECK(cuStreamWriteValue32(driverStream, incumbentAddress, 1U, CU_STREAM_WRITE_VALUE_DEFAULT));
}

void PhaseCudaDirectionalGate::wait() noexcept
{
    if (mGateStream != nullptr)
    {
        static_cast<void>(cudaStreamSynchronize(mGateStream));
    }
}

} // namespace trt_edgellm::rt
