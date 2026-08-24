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

#include "runtime/exec/tensorMap.h"
#include "runtime/mambaCacheManager.h"
#include "runtime/state/stableKVPageManager.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_edgellm::rt
{

struct PhaseRecurrentStateStats
{
    uint64_t prepareCalls{};
    uint64_t residencyHits{};
    uint64_t gatheredSlots{};
    uint64_t scatteredSlots{};
    size_t gatheredBytes{};
    size_t scatteredBytes{};
};

//! Phase-local active rows over stable recurrent and convolution state slots.
class PhaseRecurrentStateActiveView
{
public:
    PhaseRecurrentStateActiveView(StableKVPageManager& ownership, MambaCacheManager& stableStates, TensorMap& tensorMap,
        int32_t maxActiveRows, bool retainBetweenDispatches, cudaStream_t setupStream);
    ~PhaseRecurrentStateActiveView() noexcept;

    PhaseRecurrentStateActiveView(PhaseRecurrentStateActiveView const&) = delete;
    PhaseRecurrentStateActiveView& operator=(PhaseRecurrentStateActiveView const&) = delete;

    void prepare(std::vector<int32_t> const& stableSlots, cudaStream_t stream);
    void markUpdated() noexcept;
    void complete(cudaStream_t stream);
    void flushAndInvalidate(cudaStream_t stream);

    MambaCacheManager& activeStates() noexcept;
    PhaseRecurrentStateStats const& stats() const noexcept;

private:
    struct BindingBackup
    {
        Tensor* recurrentInput{};
        Tensor* recurrentOutput{};
        Tensor* convInput{};
        Tensor* convOutput{};
    };

    size_t copySlot(MambaCacheManager& source, int32_t sourceRow, MambaCacheManager& destination,
        int32_t destinationRow, cudaStream_t stream);
    size_t zeroSlot(MambaCacheManager& destination, int32_t row, cudaStream_t stream);
    void flushResident(cudaStream_t stream);
    void restoreBindings() noexcept;

    StableKVPageManager& mOwnership;
    MambaCacheManager& mStableStates;
    TensorMap& mTensorMap;
    MambaCacheManager mActiveStates;
    std::vector<BindingBackup> mPreviousBindings;
    std::vector<int32_t> mResidentSlots;
    std::vector<uint64_t> mResidentGenerations;
    std::vector<uint8_t> mDirtyRows;
    PhaseRecurrentStateStats mStats;
    bool mRetainBetweenDispatches{};
};

} // namespace trt_edgellm::rt
