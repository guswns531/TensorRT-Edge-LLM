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

#include "runtime/phase/mechanism/phaseServiceState.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm::rt
{

enum class PhaseResidentDecodeStage
{
    kUnknown,
    kQueued,
    kInFlight,
    kSampling,
    kCapacityWait,
};

inline char const* phaseResidentDecodeStageName(PhaseResidentDecodeStage stage) noexcept
{
    switch (stage)
    {
    case PhaseResidentDecodeStage::kQueued: return "queued";
    case PhaseResidentDecodeStage::kInFlight: return "inflight";
    case PhaseResidentDecodeStage::kSampling: return "sampling";
    case PhaseResidentDecodeStage::kCapacityWait: return "capacity_wait";
    default: return "unknown";
    }
}

struct PhaseResidentDecodeObservation
{
    uint64_t requestId{};
    PhaseResidentDecodeStage stage{PhaseResidentDecodeStage::kUnknown};
    PhaseServiceReference reference;
    int32_t referenceContextLength{};
    bool candidateReady{};
    uint64_t lastTokenCommittedHostNs{};
    uint64_t samplingSubmittedHostNs{};
};

struct PhaseResidentDecodeSnapshot
{
    uint64_t hostSnapshotNs{};
    std::vector<PhaseResidentDecodeObservation> requests;
};

//! Freeze an isolated CUDA reference at token commit; missing evidence stays missing for this epoch.
inline PhaseServiceReference phaseResidentDecodeReference(std::optional<float> isolatedGpuMs, uint64_t epoch) noexcept
{
    if (!isolatedGpuMs.has_value() || !std::isfinite(*isolatedGpuMs) || *isolatedGpuMs <= 0.0F)
    {
        return {0.0, PhaseServiceReferenceSource::kUnknown, epoch, false};
    }
    return {static_cast<double>(*isolatedGpuMs) * 1000.0, PhaseServiceReferenceSource::kRuntimeCovering, epoch, true};
}

inline std::optional<double> phaseResidentDecodeAge(
    PhaseResidentDecodeObservation const& request, uint64_t snapshotHostNs) noexcept
{
    PhaseServiceReference const& reference = request.reference;
    bool const measured = reference.source == PhaseServiceReferenceSource::kRuntimeExact
        || reference.source == PhaseServiceReferenceSource::kRuntimeCovering
        || reference.source == PhaseServiceReferenceSource::kRuntimeInterpolated;
    if (!reference.valid || !measured || !std::isfinite(reference.serviceUs) || reference.serviceUs <= 0.0
        || request.lastTokenCommittedHostNs == 0U || snapshotHostNs < request.lastTokenCommittedHostNs)
    {
        return std::nullopt;
    }
    return static_cast<double>(snapshotHostNs - request.lastTokenCommittedHostNs) / 1000.0 / reference.serviceUs;
}

struct PhaseResidentDecodeSummary
{
    std::array<size_t, 5> stageCounts{};
    size_t measuredReferences{};
    size_t candidateReady{};
    std::optional<double> maxServiceAge;
};

inline PhaseResidentDecodeSummary phaseSummarizeResidentDecode(PhaseResidentDecodeSnapshot const& snapshot) noexcept
{
    PhaseResidentDecodeSummary result;
    for (auto const& request : snapshot.requests)
    {
        ++result.stageCounts[static_cast<size_t>(request.stage)];
        result.candidateReady += request.candidateReady ? 1U : 0U;
        if (auto const age = phaseResidentDecodeAge(request, snapshot.hostSnapshotNs))
        {
            ++result.measuredReferences;
            result.maxServiceAge = result.maxServiceAge.has_value() ? std::max(*result.maxServiceAge, *age) : *age;
        }
    }
    return result;
}

} // namespace trt_edgellm::rt
