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
#include "common/cudaMacros.h"

#include <gtest/gtest.h>

using namespace trt_edgellm;

namespace
{

rt::MambaCacheManager::Config makeStateConfig(int32_t maxBatchSize)
{
    return {/*.numRecurrentLayers=*/1,
        /*.maxBatchSize=*/maxBatchSize,
        /*.recurrentStateNumHeads=*/1,
        /*.recurrentStateHeadDim=*/2,
        /*.recurrentStateSize=*/2,
        /*.convDim=*/2,
        /*.convKernel=*/2,
        /*.maxIntermediateSeqLen=*/0,
        /*.recurrentStateType=*/nvinfer1::DataType::kFLOAT,
        /*.convStateType=*/nvinfer1::DataType::kHALF,
        /*.recurrentStateNumGroups=*/0,
        /*.specVerifyUsesReplay=*/false};
}

size_t rowBytes(rt::Tensor const& tensor)
{
    return tensor.getMemoryCapacity() / static_cast<size_t>(tensor.getShape()[0]);
}

void setRow(rt::Tensor& tensor, int32_t row, uint8_t value, cudaStream_t stream)
{
    size_t const bytes = rowBytes(tensor);
    auto* data = static_cast<std::byte*>(tensor.rawPointer()) + row * bytes;
    CUDA_CHECK(cudaMemsetAsync(data, value, bytes, stream));
}

uint8_t firstRowByte(rt::Tensor const& tensor, int32_t row)
{
    uint8_t value{};
    size_t const bytes = rowBytes(tensor);
    auto const* data = static_cast<std::byte const*>(tensor.rawPointer()) + row * bytes;
    CUDA_CHECK(cudaMemcpy(&value, data, sizeof(value), cudaMemcpyDeviceToHost));
    return value;
}

void bindStableStates(rt::TensorMap& map, rt::MambaCacheManager& states)
{
    map.set(binding_names::formatRecurrentStateName(0, true), states.getRecurrentState(0));
    map.set(binding_names::formatRecurrentStateName(0, false), states.getRecurrentState(0));
    map.set(binding_names::formatConvStateName(0, true), states.getConvState(0));
    map.set(binding_names::formatConvStateName(0, false), states.getConvState(0));
}

} // namespace

TEST(PhaseRecurrentStateActiveViewTest, RetainsDecodeCohortAndFlushesOnlyOnRemap)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    rt::StableKVPageManager ownership({4, 2, 8, 256, 128});
    int32_t const slot0 = ownership.reserve();
    int32_t const slot1 = ownership.reserve();
    ownership.ensureCapacity(slot0, 1);
    ownership.ensureCapacity(slot1, 1);
    ownership.setLength(slot0, 1);
    ownership.setLength(slot1, 1);
    rt::MambaCacheManager stableStates(makeStateConfig(4), stream);
    setRow(stableStates.getRecurrentState(0), slot0, 0x11, stream);
    setRow(stableStates.getConvState(0), slot0, 0x12, stream);
    setRow(stableStates.getRecurrentState(0), slot1, 0x21, stream);
    setRow(stableStates.getConvState(0), slot1, 0x22, stream);
    rt::TensorMap map;
    bindStableStates(map, stableStates);

    {
        rt::PhaseRecurrentStateActiveView view(ownership, stableStates, map, 2, true, stream);
        view.prepare({slot1, slot0}, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(firstRowByte(view.activeStates().getRecurrentState(0), 0), 0x21);
        EXPECT_EQ(firstRowByte(view.activeStates().getRecurrentState(0), 1), 0x11);

        setRow(view.activeStates().getRecurrentState(0), 0, 0x31, stream);
        setRow(view.activeStates().getConvState(0), 0, 0x32, stream);
        setRow(view.activeStates().getRecurrentState(0), 1, 0x41, stream);
        setRow(view.activeStates().getConvState(0), 1, 0x42, stream);
        view.markUpdated();
        view.prepare({slot1, slot0}, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(view.stats().residencyHits, 1U);
        EXPECT_EQ(view.stats().scatteredSlots, 0U);
        EXPECT_EQ(firstRowByte(view.activeStates().getRecurrentState(0), 0), 0x31);

        view.prepare({slot0}, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        EXPECT_EQ(view.stats().scatteredSlots, 2U);
        EXPECT_EQ(view.stats().gatheredSlots, 3U);
        EXPECT_EQ(firstRowByte(stableStates.getRecurrentState(0), slot1), 0x31);
        EXPECT_EQ(firstRowByte(view.activeStates().getRecurrentState(0), 0), 0x41);
    }

    EXPECT_EQ(map.get(binding_names::formatRecurrentStateName(0, true)), &stableStates.getRecurrentState(0));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(PhaseRecurrentStateActiveViewTest, DiscardsResidentStateAfterStableSlotReuse)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    rt::StableKVPageManager ownership({2, 1, 4, 256, 128});
    int32_t const slot = ownership.reserve();
    rt::MambaCacheManager stableStates(makeStateConfig(2), stream);
    rt::TensorMap map;
    bindStableStates(map, stableStates);
    rt::PhaseRecurrentStateActiveView view(ownership, stableStates, map, 1, true, stream);
    view.prepare({slot}, stream);
    setRow(view.activeStates().getRecurrentState(0), 0, 0x77, stream);
    view.markUpdated();

    ownership.release(slot);
    EXPECT_EQ(ownership.reserve(), slot);
    view.prepare({slot}, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(firstRowByte(view.activeStates().getRecurrentState(0), 0), 0U);
    EXPECT_EQ(view.stats().scatteredSlots, 0U);

    CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(PhaseRecurrentStateActiveViewTest, PreservesUnchangedRowsAcrossPartialCohortRemaps)
{
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    rt::StableKVPageManager ownership({4, 2, 8, 256, 128});
    int32_t const slot0 = ownership.reserve();
    int32_t const slot1 = ownership.reserve();
    ownership.ensureCapacity(slot0, 1);
    ownership.ensureCapacity(slot1, 1);
    ownership.setLength(slot0, 1);
    ownership.setLength(slot1, 1);
    rt::MambaCacheManager stableStates(makeStateConfig(4), stream);
    setRow(stableStates.getRecurrentState(0), slot0, 0x11, stream);
    setRow(stableStates.getRecurrentState(0), slot1, 0x21, stream);
    rt::TensorMap map;
    bindStableStates(map, stableStates);
    rt::PhaseRecurrentStateActiveView view(ownership, stableStates, map, 2, true, stream);

    view.prepare({slot0}, stream);
    setRow(view.activeStates().getRecurrentState(0), 0, 0x31, stream);
    view.markUpdated();
    view.prepare({slot0, slot1}, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(view.stats().gatheredSlots, 2U);
    EXPECT_EQ(view.stats().scatteredSlots, 0U);
    EXPECT_EQ(firstRowByte(view.activeStates().getRecurrentState(0), 0), 0x31);
    EXPECT_EQ(firstRowByte(view.activeStates().getRecurrentState(0), 1), 0x21);

    setRow(view.activeStates().getRecurrentState(0), 1, 0x41, stream);
    view.markUpdated();
    view.prepare({slot0}, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(view.stats().gatheredSlots, 2U);
    EXPECT_EQ(view.stats().scatteredSlots, 1U);
    EXPECT_EQ(firstRowByte(stableStates.getRecurrentState(0), slot1), 0x41);
    EXPECT_EQ(firstRowByte(view.activeStates().getRecurrentState(0), 0), 0x31);

    view.flushAndInvalidate(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    EXPECT_EQ(view.stats().scatteredSlots, 2U);
    EXPECT_EQ(firstRowByte(stableStates.getRecurrentState(0), slot0), 0x31);
    CUDA_CHECK(cudaStreamDestroy(stream));
}
