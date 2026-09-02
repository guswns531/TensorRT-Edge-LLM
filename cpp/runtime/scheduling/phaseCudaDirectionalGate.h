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

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace trt_edgellm::rt
{

//! Research-only CUDA stream gate used to place a newcomer at a measured
//! fraction of an incumbent execution without delaying host preparation.
class PhaseCudaDirectionalGate
{
public:
    PhaseCudaDirectionalGate() = default;
    ~PhaseCudaDirectionalGate() noexcept;

    PhaseCudaDirectionalGate(PhaseCudaDirectionalGate const&) = delete;
    PhaseCudaDirectionalGate& operator=(PhaseCudaDirectionalGate const&) = delete;

    //! Enqueue a device-side wait on newcomerStream. The gate is released only
    //! after incumbentStart is visible on the GPU plus delayUs.
    void enqueue(cudaStream_t newcomerStream, cudaEvent_t incumbentStart, uint64_t delayUs);
    //! Queue newcomer work behind a device semaphore before the incumbent is
    //! submitted. signal() later defines the incumbent GPU start boundary.
    void arm(cudaStream_t newcomerStream, uint64_t delayUs);
    void signal(cudaStream_t incumbentStream);
    void wait() noexcept;

private:
    void ensureStorage();
    static void CUDART_CB releaseHostGate(void* userData);

    cudaStream_t mGateStream{};
    cudaEvent_t mGateDone{};
    uint32_t* mHostGateFlag{};
    uint32_t* mDeviceGateFlag{};
    uint32_t* mHostIncumbentFlag{};
    uint32_t* mDeviceIncumbentFlag{};
    uint64_t mPendingDelayUs{};
};

} // namespace trt_edgellm::rt
