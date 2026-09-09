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

#include <chrono>
#include <cstdint>

namespace trt_edgellm::rt
{

//! Host commit clocks, not CUDA completion or HTTP delivery timestamps. Zero means unavailable.
struct PhaseServiceClock
{
    uint64_t requestId{};
    uint64_t submittedHostNs{};
    uint64_t lastTokenCommittedHostNs{};
};

inline uint64_t phaseServiceHostNs(std::chrono::steady_clock::time_point timestamp) noexcept
{
    return timestamp == std::chrono::steady_clock::time_point{}
        ? 0U
        : static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count());
}

} // namespace trt_edgellm::rt
