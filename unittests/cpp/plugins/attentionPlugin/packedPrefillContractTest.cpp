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

#include "plugins/attentionPlugin/packedPrefillContract.h"

#include <gtest/gtest.h>

using namespace trt_edgellm;

TEST(PackedPrefillContractTest, RetainsPrefillModeForOneTokenContinuation)
{
    EXPECT_TRUE(plugins::isPackedPrefillInvocation(true, 1, 1, 128));
    EXPECT_TRUE(plugins::isPackedPrefillInvocation(true, 1, 128, 128));
    EXPECT_TRUE(plugins::isPackedPrefillInvocation(true, 1, 2, 128));
}

TEST(PackedPrefillContractTest, DecodeCarrierKeepsSingleAndBatchedDecode)
{
    EXPECT_FALSE(plugins::isPackedPrefillInvocation(true, 1, 1, 1));
    EXPECT_FALSE(plugins::isPackedPrefillInvocation(true, 8, 1, 1));
    EXPECT_FALSE(plugins::isPackedPrefillInvocation(true, 24, 1, 1));
}

TEST(PackedPrefillContractTest, LegacyCarrierCannotDisambiguateOneToken)
{
    EXPECT_FALSE(plugins::isPackedPrefillInvocation(true, 1, 1, 0));
    EXPECT_TRUE(plugins::isPackedPrefillInvocation(true, 1, 2, 0));
    EXPECT_TRUE(plugins::isPackedPrefillInvocation(true, 1, 128, 0));
}

TEST(PackedPrefillContractTest, RequiresPackedEngineAndNonemptySinglePhysicalRow)
{
    EXPECT_FALSE(plugins::isPackedPrefillInvocation(false, 1, 1, 128));
    EXPECT_FALSE(plugins::isPackedPrefillInvocation(false, 1, 128, 128));
    EXPECT_FALSE(plugins::isPackedPrefillInvocation(true, 8, 128, 128));
    EXPECT_FALSE(plugins::isPackedPrefillInvocation(true, 1, 0, 128));
}
