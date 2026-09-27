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

TEST(PhaseServingExecutionOptionsTest, StartupCalibrationIsOptInAndRejectsInvalidBudget)
{
    EXPECT_FALSE(resolvePhaseStartupCalibrationOptions(lookup()).enabled);
    auto const options = resolvePhaseStartupCalibrationOptions(
        lookup({{"TRT_EDGELLM_STARTUP_CALIBRATION", "1"}, {"TRT_EDGELLM_STARTUP_REQUIRE_COVERAGE", "1"}}));
    EXPECT_TRUE(options.enabled);
    EXPECT_TRUE(options.requireCoverage);
    EXPECT_THROW(
        resolvePhaseStartupCalibrationOptions(lookup({{"TRT_EDGELLM_STARTUP_BUDGET_MS", "nan"}})), std::exception);
    EXPECT_THROW(
        resolvePhaseStartupCalibrationOptions(lookup({{"TRT_EDGELLM_STARTUP_BUDGET_MS", "0"}})), std::exception);
    EXPECT_THROW(
        resolvePhaseStartupCalibrationOptions(lookup({{"TRT_EDGELLM_STARTUP_REQUIRE_COVERAGE", "1"}})), std::exception);
}

TEST(PhaseServingExecutionOptionsTest, StartupProbesRespectFullOutputReservation)
{
    auto const probes = phaseStartupDecodeProbes(24, 8, 128, 2048, 192, 128, 4);
    ASSERT_FALSE(probes.empty());
    EXPECT_EQ(probes.front().batchSize, 1);
    EXPECT_EQ(probes.back().batchSize, 24);
    for (auto const& probe : probes)
    {
        EXPECT_LE(probe.batchSize, 24);
        EXPECT_EQ(probe.promptTokens % 128, 0);
        EXPECT_LE(probe.promptTokens + probe.outputTokens, 2048);
        EXPECT_LE(((probe.promptTokens + probe.outputTokens + 127) / 128) * probe.batchSize, 192);
        EXPECT_GE(probe.outputTokens, 4 + 2 * ((probe.batchSize + 7) / 8));
    }
}

TEST(PhaseServingExecutionOptionsTest, StartupProbesDoNotInventCapacity)
{
    EXPECT_TRUE(phaseStartupDecodeProbes(64, 8, 128, 128, 256, 128, 4).empty());
    auto const probes = phaseStartupDecodeProbes(64, 8, 128, 2048, 4, 128, 4);
    ASSERT_FALSE(probes.empty());
    EXPECT_LE(probes.back().batchSize, 2);
    EXPECT_THROW(phaseStartupDecodeProbes(0, 8, 128, 2048, 256, 128, 4), std::exception);
    EXPECT_THROW(phaseStartupDecodeProbes(64, 8, 128, 2048, 256, 0, 4), std::exception);
}

TEST(PhaseServingExecutionOptionsTest, StartupFrontierIsBoundedWithoutModelSpecificBatchList)
{
    auto const probes = phaseStartupDecodeProbes(64, 8, 128, 2048, 256, 128, 4);
    EXPECT_LE(probes.size(), 14U);
    EXPECT_EQ(probes.back().batchSize, 64);
    EXPECT_LE(probes.back().promptTokens + probes.back().outputTokens, 512);
}

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
