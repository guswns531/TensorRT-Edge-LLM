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

enum class PhaseProtectedKind
{
    kUnknown,
    kEncoder,
    kPrefill,
    kDecode,
};

//! Origin of the immutable isolated-service denominator used by diagnostic
//! request-age normalization. Queue residence, SLO targets, and selected
//! overlap timings are deliberately not valid sources.
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

inline char const* phaseProtectedKindName(PhaseProtectedKind kind) noexcept
{
    switch (kind)
    {
    case PhaseProtectedKind::kUnknown: return "unknown";
    case PhaseProtectedKind::kEncoder: return "encoder";
    case PhaseProtectedKind::kPrefill: return "prefill";
    case PhaseProtectedKind::kDecode: return "decode";
    }
    return "unknown";
}

//! One request deadline protected while evaluating an action. Completion may
//! include a required follow-up phase when the candidate does not advance the
//! request owning this deadline.
struct PhaseProtectedCompletion
{
    double slackUs{std::numeric_limits<double>::infinity()};
    double predictedCompletionUs{};
    double uncertaintyUs{};
    PhaseProtectedKind kind{PhaseProtectedKind::kUnknown};
    //! Zero means the older aggregate-only contract. Nonzero identities make
    //! the estimate suitable for request-local shadow evaluation.
    uint64_t requestId{};
    //! Snapshot-fixed isolated service cost for this request's milestone.
    double referenceUs{};
    PhaseServiceReferenceSource referenceSource{PhaseServiceReferenceSource::kUnknown};
};

} // namespace trt_edgellm::rt
