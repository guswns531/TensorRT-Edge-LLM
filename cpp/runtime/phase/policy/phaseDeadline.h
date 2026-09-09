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
    //! Time already spent waiting for this milestone at the decision boundary.
    //! First-token service uses request submission; decode uses the previous
    //! token commit. It is intentionally independent of an explicit SLO.
    double elapsedServiceUs{};
    //! Whether this milestone has a user or composition-root absolute SLO.
    //! The legacy slackUs field may still contain a policy fallback.
    bool hasExplicitSlo{};
    //! Absolute request slack only. Infinite means that no external SLO was
    //! supplied for this milestone.
    double absoluteSlackUs{std::numeric_limits<double>::infinity()};
    //! Immutable service-reference generation. Zero is the pre-V3 contract.
    uint64_t serviceEpoch{};
};

} // namespace trt_edgellm::rt
