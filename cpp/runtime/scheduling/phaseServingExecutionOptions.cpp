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

#include "common/checkMacros.h"
#include "common/logger.h"
#include "runtime/scheduling/phaseQueueScheduler.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

namespace trt_edgellm::rt
{
namespace
{

bool parseBoolean(char const* value)
{
    std::string const text(value);
    ELLM_CHECK(text == "1" || text == "true" || text == "0" || text == "false", "Invalid phase boolean option");
    return text == "1" || text == "true";
}

size_t parseNonnegative(char const* value)
{
    std::string const text(value);
    ELLM_CHECK(!text.empty() && text.find_first_not_of("0123456789") == std::string::npos,
        "Phase graph limits must be non-negative integers");
    return static_cast<size_t>(std::stoull(text));
}

} // namespace

void resolvePhasePrefillExecutionOptions(
    PhaseQueueSchedulerConfig& config, int32_t engineChunkLimit, PhaseEnvironmentLookup const& environment)
{
    if (char const* value = environment("TRT_EDGELLM_FIXED_PREFILL_CHUNK"))
    {
        config.maxPrefillChunkTokens = std::stoi(value);
    }
    if (char const* value = environment("TRT_EDGELLM_ENABLE_ADAPTIVE_PREFILL_CHUNKING"))
    {
        config.enableAdaptivePrefillChunking = parseBoolean(value);
    }
    if (char const* value = environment("TRT_EDGELLM_ADAPTIVE_PREFILL_CHUNK_CANDIDATES"))
    {
        config.adaptivePrefillChunkCandidates.clear();
        std::stringstream input(value);
        std::string token;
        while (std::getline(input, token, ','))
        {
            ELLM_CHECK(!token.empty(), "Adaptive prefill candidates must not contain empty entries");
            config.adaptivePrefillChunkCandidates.push_back(std::stoi(token));
        }
        ELLM_CHECK(!config.adaptivePrefillChunkCandidates.empty(), "Adaptive prefill candidates must not be empty");
        config.enableAdaptivePrefillChunking = true;
    }
    auto& candidates = config.adaptivePrefillChunkCandidates;
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    for (int32_t const candidate : candidates)
    {
        ELLM_CHECK(
            candidate > 0 && candidate <= engineChunkLimit, "Adaptive prefill candidate exceeds the engine capability");
    }
    if (config.enableAdaptivePrefillChunking && !candidates.empty())
    {
        config.maxPrefillChunkTokens = std::max(config.maxPrefillChunkTokens, candidates.back());
    }
    ELLM_CHECK(config.maxPrefillChunkTokens > 0 && config.maxPrefillChunkTokens <= engineChunkLimit,
        "Prefill chunk exceeds the engine capability");
    config.minPrefillChunkTokens = config.enableAdaptivePrefillChunking && !candidates.empty()
        ? candidates.front()
        : config.maxPrefillChunkTokens;
    config.maxOverlapPrefillTokens = std::min(128, config.maxPrefillChunkTokens);
    if (char const* value = environment("TRT_EDGELLM_MAX_OVERLAP_PREFILL_TOKENS"))
    {
        config.maxOverlapPrefillTokens = std::stoi(value);
    }
    ELLM_CHECK(config.maxOverlapPrefillTokens >= 0 && config.maxOverlapPrefillTokens <= engineChunkLimit,
        "Overlap prefill limit exceeds the engine capability");
    if (char const* value = environment("TRT_EDGELLM_ENABLE_COST_AWARE_PREFILL_SHAPE"))
    {
        config.enableCostAwarePrefillShapeSelection = parseBoolean(value);
    }
    if (char const* value = environment("TRT_EDGELLM_DECODE_BURST_GRACE_PERIOD_US"))
    {
        config.transitionPredictorConfig.burstGracePeriodUs = std::stod(value);
    }
    ELLM_CHECK(std::isfinite(config.transitionPredictorConfig.burstGracePeriodUs)
            && config.transitionPredictorConfig.burstGracePeriodUs >= 0.0,
        "Decode burst grace must be finite and non-negative");
    LOG_INFO(
        "Phase execution contract: chunk=%d min_chunk=%d overlap_tokens=%d adaptive=%d shape_cost=%d grace_us=%.3f",
        config.maxPrefillChunkTokens, config.minPrefillChunkTokens, config.maxOverlapPrefillTokens,
        config.enableAdaptivePrefillChunking, config.enableCostAwarePrefillShapeSelection,
        config.transitionPredictorConfig.burstGracePeriodUs);
}

PhaseGraphExecutionOptions resolvePhaseGraphExecutionOptions(
    PhaseGraphExecutionOptions config, PhaseEnvironmentLookup const& environment)
{
    if (char const* value = environment("TRT_EDGELLM_CAPTURE_PHASE_GRAPHS"))
    {
        config.enabled = parseBoolean(value);
    }
    if (char const* value = environment("TRT_EDGELLM_ONLINE_GRAPH_CAPTURE"))
    {
        config.onlineCapture = parseBoolean(value);
    }
    if (char const* value = environment("TRT_EDGELLM_GRAPH_CAPTURE_MIN_OBSERVATIONS"))
    {
        config.minObservations = parseNonnegative(value);
    }
    if (char const* value = environment("TRT_EDGELLM_MAX_PREFILL_GRAPHS"))
    {
        config.maxPrefillGraphs = parseNonnegative(value);
    }
    if (char const* value = environment("TRT_EDGELLM_MAX_DECODE_GRAPHS"))
    {
        config.maxDecodeGraphs = parseNonnegative(value);
    }
    ELLM_CHECK(config.minObservations > 0U, "Graph capture observations must be positive");
    config.onlineCapture = config.enabled && config.onlineCapture;
    return config;
}

std::vector<int32_t> phaseDecodeGraphWarmupBatches(int32_t maxBatchSize)
{
    std::vector<int32_t> batches;
    if (maxBatchSize <= 0)
    {
        return batches;
    }
    batches.push_back(maxBatchSize);
    for (int32_t size = maxBatchSize / 2; size > 0; size /= 2)
    {
        batches.push_back(size);
    }
    for (int32_t size = 1; size < maxBatchSize; ++size)
    {
        if (std::find(batches.begin(), batches.end(), size) == batches.end())
        {
            batches.push_back(size);
        }
    }
    return batches;
}

} // namespace trt_edgellm::rt
