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

#include "runtime/scheduling/phaseContinuousLoadGenerator.h"

#include <gtest/gtest.h>

#include <limits>

using namespace trt_edgellm;

namespace
{

rt::PhaseContinuousLoadConfig makeConfig()
{
    return {5, 4.0, 16, 64, 2, 8, 7, 900};
}

TEST(PhaseContinuousLoadGeneratorTest, RepeatsScheduleForSameSeed)
{
    rt::PhaseContinuousLoadGenerator first(makeConfig());
    rt::PhaseContinuousLoadGenerator second(makeConfig());

    EXPECT_EQ(first.schedule(), second.schedule());
    ASSERT_EQ(first.schedule().size(), 5);
    EXPECT_EQ(first.schedule()[0].requestId, 900);
    EXPECT_EQ(first.schedule()[0].arrivalOffsetUs, 0);
    EXPECT_EQ(first.schedule()[1].arrivalOffsetUs, 250000);
    EXPECT_EQ(first.schedule()[4].arrivalOffsetUs, 1000000);
    for (rt::PhaseLoadRequest const& request : first.schedule())
    {
        EXPECT_GE(request.promptTokenCount, 16);
        EXPECT_LE(request.promptTokenCount, 64);
        EXPECT_GE(request.maxOutputTokens, 2);
        EXPECT_LE(request.maxOutputTokens, 8);
    }
}

TEST(PhaseContinuousLoadGeneratorTest, ChangesLengthsForDifferentSeed)
{
    rt::PhaseContinuousLoadConfig firstConfig = makeConfig();
    rt::PhaseContinuousLoadConfig secondConfig = makeConfig();
    secondConfig.seed = firstConfig.seed + 1;
    rt::PhaseContinuousLoadGenerator first(firstConfig);
    rt::PhaseContinuousLoadGenerator second(secondConfig);

    bool hasDifferentLength{};
    for (size_t index = 0; index < first.schedule().size(); ++index)
    {
        hasDifferentLength = hasDifferentLength
            || first.schedule()[index].promptTokenCount != second.schedule()[index].promptTokenCount
            || first.schedule()[index].maxOutputTokens != second.schedule()[index].maxOutputTokens;
    }
    EXPECT_TRUE(hasDifferentLength);
}

TEST(PhaseContinuousLoadGeneratorTest, PopsOnlyReadyRequestsAtBoundaries)
{
    rt::PhaseContinuousLoadGenerator generator(makeConfig());

    EXPECT_TRUE(generator.popReady(-1).empty());
    EXPECT_EQ(generator.nextArrivalOffsetUs(), 0);
    EXPECT_EQ(generator.popReady(0).size(), 1);
    EXPECT_TRUE(generator.popReady(249999).empty());
    EXPECT_EQ(generator.popReady(500000).size(), 2);
    EXPECT_EQ(generator.remaining(), 2);
    EXPECT_EQ(generator.popReady(1000000).size(), 2);
    EXPECT_TRUE(generator.done());
    EXPECT_EQ(generator.nextArrivalOffsetUs(), std::nullopt);
}

TEST(PhaseContinuousLoadGeneratorTest, RejectsInvalidConfiguration)
{
    rt::PhaseContinuousLoadConfig config = makeConfig();
    config.requestCount = 0;
    EXPECT_THROW(rt::PhaseContinuousLoadGenerator{config}, std::runtime_error);
    config = makeConfig();
    config.requestsPerSecond = 0.0;
    EXPECT_THROW(rt::PhaseContinuousLoadGenerator{config}, std::runtime_error);
    config = makeConfig();
    config.requestsPerSecond = std::numeric_limits<double>::infinity();
    EXPECT_THROW(rt::PhaseContinuousLoadGenerator{config}, std::runtime_error);
    config = makeConfig();
    config.minPromptTokens = config.maxPromptTokens + 1;
    EXPECT_THROW(rt::PhaseContinuousLoadGenerator{config}, std::runtime_error);
    config = makeConfig();
    config.minOutputTokens = 0;
    EXPECT_THROW(rt::PhaseContinuousLoadGenerator{config}, std::runtime_error);
    config = makeConfig();
    config.firstRequestId = std::numeric_limits<uint64_t>::max() - 2;
    EXPECT_THROW(rt::PhaseContinuousLoadGenerator{config}, std::runtime_error);
}

} // namespace
