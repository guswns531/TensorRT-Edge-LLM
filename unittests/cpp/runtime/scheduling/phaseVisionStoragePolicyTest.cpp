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

#include "runtime/scheduling/phaseVisionAdapter.h"

#include <gtest/gtest.h>

#include <array>

using trt_edgellm::rt::Coords;
using trt_edgellm::rt::DeviceType;
using trt_edgellm::rt::PhaseVisionPayload;
using trt_edgellm::rt::phaseVisionPreparationWithinStorageBudget;
using trt_edgellm::rt::Tensor;

TEST(PhaseVisionStoragePolicyTest, SingleStorageWaitsForDownstreamPrefillConsumers)
{
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 1, 1, true));
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 1, 0, true));
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 1, 1, false));
    EXPECT_TRUE(phaseVisionPreparationWithinStorageBudget(true, 1, 0, false));
}

TEST(PhaseVisionStoragePolicyTest, DoubleStorageBoundsPreparationBehindPrefill)
{
    EXPECT_TRUE(phaseVisionPreparationWithinStorageBudget(true, 2, 1, true));
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 2, 2, true));
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 2, 2, false));
    EXPECT_TRUE(phaseVisionPreparationWithinStorageBudget(true, 2, 1, false));
}

TEST(PhaseVisionStoragePolicyTest, ReleaseReopensCapacityWithoutRelaxingWorkspaceExclusion)
{
    size_t retained = 2;
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 2, retained, true));
    --retained;
    EXPECT_TRUE(phaseVisionPreparationWithinStorageBudget(true, 2, retained, true));
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 0, 0, false));
}

TEST(PhaseVisionStoragePolicyTest, IndependentWorkspaceDoesNotInheritSharedStorageBarrier)
{
    EXPECT_TRUE(phaseVisionPreparationWithinStorageBudget(false, 1, 2, true));
}

TEST(PhaseVisionStoragePolicyTest, DecodeOnlyMropeDoesNotConsumeAnEncoderSlab)
{
    std::array<float, 8> mrope{};
    PhaseVisionPayload payload;
    payload.mropeCosSin = Tensor(mrope.data(), Coords{1, 4, 2}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);

    EXPECT_EQ(payload.byteSize(), sizeof(mrope));
    EXPECT_EQ(payload.prefillByteSize(), 0U);
    EXPECT_TRUE(phaseVisionPreparationWithinStorageBudget(true, 1, 0, payload.prefillByteSize() > 0U));
    EXPECT_FALSE(payload.mropeCosSin.isEmpty());
}

TEST(PhaseVisionStoragePolicyTest, UnpooledPrefillViewsStillBlockSingleStorage)
{
    std::array<float, 8> embedding{};
    PhaseVisionPayload payload;
    payload.outputEmbedding = Tensor(embedding.data(), Coords{2, 4}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 1, 0, payload.prefillByteSize() > 0U));

    payload.outputEmbedding = Tensor{};
    payload.deepstackFeatures.emplace_back(
        embedding.data(), Coords{2, 4}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 1, 0, payload.prefillByteSize() > 0U));
    EXPECT_TRUE(phaseVisionPreparationWithinStorageBudget(true, 2, 0, payload.prefillByteSize() > 0U));
}

TEST(PhaseVisionStoragePolicyTest, LegacyTiedMropeRetainsItsEncoderSlab)
{
    std::array<float, 8> embedding{};
    std::array<float, 8> mrope{};
    PhaseVisionPayload payload;
    payload.outputEmbedding = Tensor(embedding.data(), Coords{2, 4}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);
    payload.mropeCosSin = Tensor(mrope.data(), Coords{1, 4, 2}, DeviceType::kCPU, nvinfer1::DataType::kFLOAT);

    EXPECT_EQ(payload.releasePrefillStorage(), 0U);
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 1, 1, payload.prefillByteSize() > 0U));
    EXPECT_FALSE(payload.mropeCosSin.isEmpty());
    EXPECT_FALSE(payload.outputEmbedding.isEmpty());
}

TEST(PhaseVisionStoragePolicyTest, PhysicalConsumerLeaseBlocksUntilRetired)
{
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 1, 1, false));
    EXPECT_FALSE(phaseVisionPreparationWithinStorageBudget(true, 2, 2, false));
    EXPECT_TRUE(phaseVisionPreparationWithinStorageBudget(true, 1, 0, false));
    EXPECT_TRUE(phaseVisionPreparationWithinStorageBudget(true, 2, 1, false));
}
