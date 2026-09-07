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

#include "common/checkMacros.h"

#include <cmath>
#include <limits>
#include <random>

namespace trt_edgellm
{
namespace rt
{

namespace
{

int32_t sampleRange(std::mt19937& generator, int32_t minimum, int32_t maximum)
{
    uint64_t const span = static_cast<uint64_t>(maximum) - static_cast<uint64_t>(minimum) + 1U;
    return minimum + static_cast<int32_t>(static_cast<uint64_t>(generator()) % span);
}

} // namespace

PhaseContinuousLoadGenerator::PhaseContinuousLoadGenerator(PhaseContinuousLoadConfig config)
{
    check::check(config.requestCount > 0, "Continuous load requestCount must be positive.");
    check::check(std::isfinite(config.requestsPerSecond) && config.requestsPerSecond > 0.0,
        "Continuous load requestsPerSecond must be finite and positive.");
    check::check(config.minPromptTokens > 0 && config.minPromptTokens <= config.maxPromptTokens,
        "Continuous load prompt-token range is invalid.");
    check::check(config.minOutputTokens > 0 && config.minOutputTokens <= config.maxOutputTokens,
        "Continuous load output-token range is invalid.");
    check::check(
        config.firstRequestId <= std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(config.requestCount - 1),
        "Continuous load request IDs overflow uint64_t.");

    double const intervalUs = 1000000.0 / config.requestsPerSecond;
    double const lastArrivalUs = static_cast<double>(config.requestCount - 1) * intervalUs;
    check::check(lastArrivalUs <= static_cast<double>(std::numeric_limits<int64_t>::max()),
        "Continuous load arrival schedule overflows int64_t.");

    std::mt19937 generator(config.seed);
    mSchedule.reserve(config.requestCount);
    for (int32_t index = 0; index < config.requestCount; ++index)
    {
        mSchedule.push_back({config.firstRequestId + static_cast<uint64_t>(index),
            static_cast<int64_t>(std::llround(static_cast<double>(index) * intervalUs)),
            sampleRange(generator, config.minPromptTokens, config.maxPromptTokens),
            sampleRange(generator, config.minOutputTokens, config.maxOutputTokens)});
    }
}

std::vector<PhaseLoadRequest> PhaseContinuousLoadGenerator::popReady(int64_t elapsedUs)
{
    std::vector<PhaseLoadRequest> result;
    while (mNextRequest < mSchedule.size() && mSchedule[mNextRequest].arrivalOffsetUs <= elapsedUs)
    {
        result.push_back(mSchedule[mNextRequest]);
        ++mNextRequest;
    }
    return result;
}

std::optional<int64_t> PhaseContinuousLoadGenerator::nextArrivalOffsetUs() const noexcept
{
    if (done())
    {
        return std::nullopt;
    }
    return mSchedule[mNextRequest].arrivalOffsetUs;
}

bool PhaseContinuousLoadGenerator::done() const noexcept
{
    return mNextRequest == mSchedule.size();
}

size_t PhaseContinuousLoadGenerator::remaining() const noexcept
{
    return mSchedule.size() - mNextRequest;
}

std::vector<PhaseLoadRequest> const& PhaseContinuousLoadGenerator::schedule() const noexcept
{
    return mSchedule;
}

} // namespace rt
} // namespace trt_edgellm
