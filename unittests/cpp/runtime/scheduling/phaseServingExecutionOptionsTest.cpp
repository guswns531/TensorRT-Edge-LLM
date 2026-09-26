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

#include "runtime/scheduling/phaseServingExecutionOptions.h"
#include "runtime/scheduling/phaseQueueScheduler.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <map>
#include <string>
#include <utility>

using namespace trt_edgellm::rt;

namespace
{
PhaseEnvironmentLookup lookup(std::map<std::string, std::string> values = {})
{
    return [values = std::move(values)](char const* name) -> char const* {
        auto const found = values.find(name);
        return found == values.end() ? nullptr : found->second.c_str();
    };
}
} // namespace

TEST(PhaseServingExecutionOptionsTest, SharedDefaultsPreserveFixedChunkAndBurstGrace)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 512;
    resolvePhasePrefillExecutionOptions(config, 512, lookup());
    EXPECT_EQ(config.maxPrefillChunkTokens, 512);
    EXPECT_EQ(config.minPrefillChunkTokens, 512);
    EXPECT_EQ(config.maxOverlapPrefillTokens, 128);
    EXPECT_DOUBLE_EQ(config.transitionPredictorConfig.burstGracePeriodUs, 20000.0);
}

TEST(PhaseServingExecutionOptionsTest, AdaptiveCandidatesUseValidatedSortedCapabilities)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 128;
    resolvePhasePrefillExecutionOptions(config, 512,
        lookup({{"TRT_EDGELLM_ADAPTIVE_PREFILL_CHUNK_CANDIDATES", "512,128,256,128"},
            {"TRT_EDGELLM_DECODE_BURST_GRACE_PERIOD_US", "15000"}}));
    EXPECT_TRUE(config.enableAdaptivePrefillChunking);
    EXPECT_EQ(config.adaptivePrefillChunkCandidates, (std::vector<int32_t>{128, 256, 512}));
    EXPECT_EQ(config.minPrefillChunkTokens, 128);
    EXPECT_EQ(config.maxPrefillChunkTokens, 512);
    EXPECT_DOUBLE_EQ(config.transitionPredictorConfig.burstGracePeriodUs, 15000.0);
}

TEST(PhaseServingExecutionOptionsTest, RejectsUnsupportedChunksAndNonfiniteGrace)
{
    PhaseQueueSchedulerConfig config;
    config.maxPrefillChunkTokens = 128;
    EXPECT_ANY_THROW(resolvePhasePrefillExecutionOptions(
        config, 128, lookup({{"TRT_EDGELLM_ADAPTIVE_PREFILL_CHUNK_CANDIDATES", "128,512"}})));
    EXPECT_ANY_THROW(resolvePhasePrefillExecutionOptions(
        config, 512, lookup({{"TRT_EDGELLM_DECODE_BURST_GRACE_PERIOD_US", "nan"}})));
    EXPECT_ANY_THROW(
        resolvePhasePrefillExecutionOptions(config, 512, lookup({{"TRT_EDGELLM_MAX_OVERLAP_PREFILL_TOKENS", "1024"}})));
}

TEST(PhaseServingExecutionOptionsTest, GraphReplayDoesNotImplyOnlineCapture)
{
    auto const options = resolvePhaseGraphExecutionOptions({}, lookup({{"TRT_EDGELLM_CAPTURE_PHASE_GRAPHS", "1"}}));
    EXPECT_TRUE(options.enabled);
    EXPECT_FALSE(options.onlineCapture);
    EXPECT_EQ(options.minObservations, 8U);
    EXPECT_EQ(options.maxDecodeGraphs, 8U);
    EXPECT_EQ(options.maxPrefillGraphs, 4U);
}

TEST(PhaseServingExecutionOptionsTest, GraphBooleanZeroDisablesInsteadOfEnablingByPresence)
{
    PhaseGraphExecutionOptions initial;
    initial.enabled = true;
    auto const options = resolvePhaseGraphExecutionOptions(
        initial, lookup({{"TRT_EDGELLM_CAPTURE_PHASE_GRAPHS", "0"}, {"TRT_EDGELLM_ONLINE_GRAPH_CAPTURE", "1"}}));
    EXPECT_FALSE(options.enabled);
    EXPECT_FALSE(options.onlineCapture);
    EXPECT_ANY_THROW(resolvePhaseGraphExecutionOptions({}, lookup({{"TRT_EDGELLM_MAX_DECODE_GRAPHS", "-1"}})));
    EXPECT_ANY_THROW(
        resolvePhaseGraphExecutionOptions({}, lookup({{"TRT_EDGELLM_GRAPH_CAPTURE_MIN_OBSERVATIONS", "0"}})));
}

TEST(PhaseServingExecutionOptionsTest, GraphPrimingIncludesMaximumWithBoundedCoverage)
{
    auto const shapes = phaseDecodeGraphWarmupBatches(24);
    ASSERT_EQ(shapes.size(), 24U);
    EXPECT_EQ(shapes.front(), 24);
    EXPECT_EQ(shapes[1], 12);
    auto sorted = shapes;
    std::sort(sorted.begin(), sorted.end());
    for (int32_t index = 0; index < 24; ++index)
    {
        EXPECT_EQ(sorted[index], index + 1);
    }
    EXPECT_TRUE(phaseDecodeGraphWarmupBatches(0).empty());
}
