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

using trt_edgellm::rt::phaseVisionPreparationWithinStorageBudget;

TEST(PhaseVisionStoragePolicyTest, SingleStorageWaitsForAllDownstreamConsumers)
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
