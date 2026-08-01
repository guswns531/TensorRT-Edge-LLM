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

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct PhaseLoadRequest
{
    uint64_t requestId{};
    int64_t arrivalOffsetUs{};
    int32_t promptTokenCount{};
    int32_t maxOutputTokens{};

    bool operator==(PhaseLoadRequest const& other) const noexcept
    {
        return requestId == other.requestId && arrivalOffsetUs == other.arrivalOffsetUs
            && promptTokenCount == other.promptTokenCount && maxOutputTokens == other.maxOutputTokens;
    }
};

struct PhaseContinuousLoadConfig
{
    int32_t requestCount{};
    double requestsPerSecond{};
    int32_t minPromptTokens{};
    int32_t maxPromptTokens{};
    int32_t minOutputTokens{};
    int32_t maxOutputTokens{};
    uint32_t seed{};
    uint64_t firstRequestId{10000};
};

//! Produces a reproducible fixed-interval request schedule with seed-derived lengths.
class PhaseContinuousLoadGenerator
{
public:
    explicit PhaseContinuousLoadGenerator(PhaseContinuousLoadConfig config);

    //! Remove and return all requests whose scheduled arrival is at or before elapsedUs.
    std::vector<PhaseLoadRequest> popReady(int64_t elapsedUs);

    //! Scheduled offset for the next request, or nullopt after the schedule is consumed.
    std::optional<int64_t> nextArrivalOffsetUs() const noexcept;

    bool done() const noexcept;
    size_t remaining() const noexcept;
    std::vector<PhaseLoadRequest> const& schedule() const noexcept;

private:
    std::vector<PhaseLoadRequest> mSchedule;
    size_t mNextRequest{};
};

} // namespace rt
} // namespace trt_edgellm
