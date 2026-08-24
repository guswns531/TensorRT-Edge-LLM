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

#include "runtime/scheduling/phaseRecurrentStateActiveView.h"

#include "common/bindingNames.h"
#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <algorithm>
#include <cstddef>
#include <unordered_set>

namespace trt_edgellm::rt
{

namespace
{

size_t rowBytes(Tensor const& tensor)
{
    ELLM_CHECK(tensor.getShape().getNumDims() > 0 && tensor.getShape()[0] > 0,
        "Phase recurrent state tensor has no batch dimension");
    return tensor.getMemoryCapacity() / static_cast<size_t>(tensor.getShape()[0]);
}

void copyTensorRow(
    Tensor const& source, int32_t sourceRow, Tensor& destination, int32_t destinationRow, cudaStream_t stream)
{
    size_t const sourceBytes = rowBytes(source);
    size_t const destinationBytes = rowBytes(destination);
    ELLM_CHECK(sourceBytes == destinationBytes && source.getDataType() == destination.getDataType(),
        "Phase recurrent state row contracts do not match");
    auto const* sourceData = static_cast<std::byte const*>(source.rawPointer()) + sourceRow * sourceBytes;
    auto* destinationData = static_cast<std::byte*>(destination.rawPointer()) + destinationRow * destinationBytes;
    CUDA_CHECK(cudaMemcpyAsync(destinationData, sourceData, sourceBytes, cudaMemcpyDeviceToDevice, stream));
}

void zeroTensorRow(Tensor& tensor, int32_t row, cudaStream_t stream)
{
    size_t const bytes = rowBytes(tensor);
    auto* data = static_cast<std::byte*>(tensor.rawPointer()) + row * bytes;
    CUDA_CHECK(cudaMemsetAsync(data, 0, bytes, stream));
}

} // namespace

PhaseRecurrentStateActiveView::PhaseRecurrentStateActiveView(StableKVPageManager& ownership,
    MambaCacheManager& stableStates, TensorMap& tensorMap, int32_t maxActiveRows, bool retainBetweenDispatches,
    cudaStream_t setupStream)
    : mOwnership(ownership)
    , mStableStates(stableStates)
    , mTensorMap(tensorMap)
    , mRetainBetweenDispatches(retainBetweenDispatches)
{
    MambaCacheManager::Config activeConfig = stableStates.getConfig();
    ELLM_CHECK(activeConfig.numRecurrentLayers > 0, "Phase recurrent state view requires recurrent layers");
    ELLM_CHECK(maxActiveRows > 0 && maxActiveRows <= ownership.config().maxStableSlots,
        "Phase recurrent state active-row capacity is invalid");
    activeConfig.maxBatchSize = maxActiveRows;
    activeConfig.maxIntermediateSeqLen = 0;
    mActiveStates = MambaCacheManager(activeConfig, setupStream);

    mPreviousBindings.reserve(static_cast<size_t>(activeConfig.numRecurrentLayers));
    for (int32_t layer = 0; layer < activeConfig.numRecurrentLayers; ++layer)
    {
        std::string const recurrentInput = binding_names::formatRecurrentStateName(layer, true);
        std::string const recurrentOutput = binding_names::formatRecurrentStateName(layer, false);
        std::string const convInput = binding_names::formatConvStateName(layer, true);
        std::string const convOutput = binding_names::formatConvStateName(layer, false);
        BindingBackup backup{mTensorMap.get(recurrentInput), mTensorMap.get(recurrentOutput), mTensorMap.get(convInput),
            mTensorMap.get(convOutput)};
        ELLM_CHECK(backup.recurrentInput != nullptr && backup.recurrentOutput != nullptr && backup.convInput != nullptr
                && backup.convOutput != nullptr,
            "Phase recurrent state view requires existing state bindings");
        mPreviousBindings.push_back(backup);
        mTensorMap.set(recurrentInput, mActiveStates.getRecurrentState(layer));
        mTensorMap.set(recurrentOutput, mActiveStates.getRecurrentState(layer));
        mTensorMap.set(convInput, mActiveStates.getConvState(layer));
        mTensorMap.set(convOutput, mActiveStates.getConvState(layer));
    }
}

PhaseRecurrentStateActiveView::~PhaseRecurrentStateActiveView() noexcept
{
    restoreBindings();
}

void PhaseRecurrentStateActiveView::prepare(std::vector<int32_t> const& stableSlots, cudaStream_t stream)
{
    ELLM_CHECK(
        !stableSlots.empty() && static_cast<int32_t>(stableSlots.size()) <= mActiveStates.getConfig().maxBatchSize,
        "Phase recurrent state batch is outside the active-row capacity");
    std::unordered_set<int32_t> uniqueSlots;
    std::vector<uint64_t> generations;
    generations.reserve(stableSlots.size());
    for (int32_t const slot : stableSlots)
    {
        ELLM_CHECK(mOwnership.leased(slot), "Phase recurrent state slot is not leased");
        ELLM_CHECK(uniqueSlots.insert(slot).second, "Phase recurrent state batch contains a duplicate slot");
        generations.push_back(mOwnership.generation(slot));
    }
    ++mStats.prepareCalls;
    if (stableSlots == mResidentSlots && generations == mResidentGenerations)
    {
        ++mStats.residencyHits;
        return;
    }

    std::vector<uint8_t> preserved(stableSlots.size(), 0U);
    size_t const commonRows = std::min(stableSlots.size(), mResidentSlots.size());
    for (size_t row = 0; row < commonRows; ++row)
    {
        preserved[row]
            = stableSlots[row] == mResidentSlots[row] && generations[row] == mResidentGenerations[row] ? 1U : 0U;
    }
    for (size_t row = 0; row < mResidentSlots.size(); ++row)
    {
        if (row < preserved.size() && preserved[row] != 0U)
        {
            continue;
        }
        int32_t const slot = mResidentSlots[row];
        if (row < mDirtyRows.size() && mDirtyRows[row] != 0U && mOwnership.leased(slot)
            && mOwnership.generation(slot) == mResidentGenerations[row])
        {
            mStats.scatteredBytes += copySlot(mActiveStates, static_cast<int32_t>(row), mStableStates, slot, stream);
            ++mStats.scatteredSlots;
        }
    }
    std::vector<uint8_t> dirtyRows(stableSlots.size(), 0U);
    for (size_t row = 0; row < commonRows; ++row)
    {
        if (preserved[row] != 0U)
        {
            dirtyRows[row] = mDirtyRows[row];
        }
    }
    mResidentSlots = stableSlots;
    mResidentGenerations = generations;
    for (size_t row = 0; row < stableSlots.size(); ++row)
    {
        if (preserved[row] != 0U)
        {
            continue;
        }
        int32_t const slot = stableSlots[row];
        size_t const bytes = mOwnership.length(slot) == 0
            ? zeroSlot(mActiveStates, static_cast<int32_t>(row), stream)
            : copySlot(mStableStates, slot, mActiveStates, static_cast<int32_t>(row), stream);
        mStats.gatheredBytes += bytes;
        ++mStats.gatheredSlots;
    }
    mDirtyRows = std::move(dirtyRows);
}

void PhaseRecurrentStateActiveView::markUpdated() noexcept
{
    std::fill(mDirtyRows.begin(), mDirtyRows.end(), 1U);
}

void PhaseRecurrentStateActiveView::complete(cudaStream_t stream)
{
    if (!mRetainBetweenDispatches)
    {
        flushAndInvalidate(stream);
    }
}

void PhaseRecurrentStateActiveView::flushAndInvalidate(cudaStream_t stream)
{
    flushResident(stream);
    mResidentSlots.clear();
    mResidentGenerations.clear();
    mDirtyRows.clear();
}

MambaCacheManager& PhaseRecurrentStateActiveView::activeStates() noexcept
{
    return mActiveStates;
}

PhaseRecurrentStateStats const& PhaseRecurrentStateActiveView::stats() const noexcept
{
    return mStats;
}

size_t PhaseRecurrentStateActiveView::copySlot(MambaCacheManager& source, int32_t sourceRow,
    MambaCacheManager& destination, int32_t destinationRow, cudaStream_t stream)
{
    ELLM_CHECK(source.numLayers() == destination.numLayers(), "Phase recurrent state layer counts do not match");
    size_t bytes{};
    for (int32_t layer = 0; layer < source.numLayers(); ++layer)
    {
        Tensor& sourceRecurrent = source.getRecurrentState(layer);
        Tensor& destinationRecurrent = destination.getRecurrentState(layer);
        Tensor& sourceConv = source.getConvState(layer);
        Tensor& destinationConv = destination.getConvState(layer);
        copyTensorRow(sourceRecurrent, sourceRow, destinationRecurrent, destinationRow, stream);
        copyTensorRow(sourceConv, sourceRow, destinationConv, destinationRow, stream);
        bytes += rowBytes(sourceRecurrent) + rowBytes(sourceConv);
    }
    return bytes;
}

size_t PhaseRecurrentStateActiveView::zeroSlot(MambaCacheManager& destination, int32_t row, cudaStream_t stream)
{
    size_t bytes{};
    for (int32_t layer = 0; layer < destination.numLayers(); ++layer)
    {
        Tensor& recurrent = destination.getRecurrentState(layer);
        Tensor& conv = destination.getConvState(layer);
        zeroTensorRow(recurrent, row, stream);
        zeroTensorRow(conv, row, stream);
        bytes += rowBytes(recurrent) + rowBytes(conv);
    }
    return bytes;
}

void PhaseRecurrentStateActiveView::flushResident(cudaStream_t stream)
{
    for (size_t row = 0; row < mResidentSlots.size(); ++row)
    {
        if (row >= mDirtyRows.size() || mDirtyRows[row] == 0U)
        {
            continue;
        }
        int32_t const slot = mResidentSlots[row];
        if (!mOwnership.leased(slot) || mOwnership.generation(slot) != mResidentGenerations[row])
        {
            continue;
        }
        mStats.scatteredBytes += copySlot(mActiveStates, static_cast<int32_t>(row), mStableStates, slot, stream);
        ++mStats.scatteredSlots;
        mDirtyRows[row] = 0U;
    }
}

void PhaseRecurrentStateActiveView::restoreBindings() noexcept
{
    try
    {
        for (int32_t layer = 0; layer < static_cast<int32_t>(mPreviousBindings.size()); ++layer)
        {
            BindingBackup const& backup = mPreviousBindings[static_cast<size_t>(layer)];
            mTensorMap.set(binding_names::formatRecurrentStateName(layer, true), *backup.recurrentInput);
            mTensorMap.set(binding_names::formatRecurrentStateName(layer, false), *backup.recurrentOutput);
            mTensorMap.set(binding_names::formatConvStateName(layer, true), *backup.convInput);
            mTensorMap.set(binding_names::formatConvStateName(layer, false), *backup.convOutput);
        }
    }
    catch (...)
    {
    }
}

} // namespace trt_edgellm::rt
