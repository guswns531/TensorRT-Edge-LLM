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

#include <cstdint>
#include <limits>

namespace trt_edgellm::rt
{

//! Origin of an immutable isolated-service denominator. Queue residence,
//! SLO targets, and selected overlap timings are not valid sources.
enum class PhaseServiceReferenceSource
{
    kUnknown,
    kRuntimeExact,
    kRuntimeInterpolated,
    kRuntimeCovering,
    kStaticProfile,
    kColdFallback,
    kDerivedIsolated,
};

inline char const* phaseServiceReferenceSourceName(PhaseServiceReferenceSource source) noexcept
{
    switch (source)
    {
    case PhaseServiceReferenceSource::kUnknown: return "unknown";
    case PhaseServiceReferenceSource::kRuntimeExact: return "runtime_exact";
    case PhaseServiceReferenceSource::kRuntimeInterpolated: return "runtime_interpolated";
    case PhaseServiceReferenceSource::kRuntimeCovering: return "runtime_covering";
    case PhaseServiceReferenceSource::kStaticProfile: return "static_profile";
    case PhaseServiceReferenceSource::kColdFallback: return "cold_fallback";
    case PhaseServiceReferenceSource::kDerivedIsolated: return "derived_isolated";
    }
    return "unknown";
}

struct PhaseServiceReference
{
    double serviceUs{};
    PhaseServiceReferenceSource source{PhaseServiceReferenceSource::kUnknown};
    uint64_t epoch{};
    bool valid{};
};

struct PhaseServiceState
{
    uint64_t requestId{};
    double readyWaitUs{};
    double serviceAgeQuanta{};
    PhaseServiceReference reference;
    bool hasExplicitSlo{};
    double absoluteSlackUs{std::numeric_limits<double>::infinity()};
};

} // namespace trt_edgellm::rt
