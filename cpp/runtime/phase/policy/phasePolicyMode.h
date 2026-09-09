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

#include <optional>
#include <string_view>

namespace trt_edgellm::rt
{

//! The only supported phase-policy variants. Execution, ownership, candidate
//! formation, and SLO feasibility are shared by every mode.
enum class PhasePolicyMode
{
    kExact,
    kContextualScalar,
    kContextualScalarTransition,
    kServiceScaledTransition,
};

inline char const* phasePolicyModeName(PhasePolicyMode mode) noexcept
{
    switch (mode)
    {
    case PhasePolicyMode::kExact: return "exact";
    case PhasePolicyMode::kContextualScalar: return "scalar";
    case PhasePolicyMode::kContextualScalarTransition: return "scalar-transition";
    case PhasePolicyMode::kServiceScaledTransition: return "service-scaled-transition";
    }
    return "unknown";
}

inline std::optional<PhasePolicyMode> phasePolicyModeFromName(std::string_view name) noexcept
{
    if (name == "exact")
    {
        return PhasePolicyMode::kExact;
    }
    if (name == "scalar")
    {
        return PhasePolicyMode::kContextualScalar;
    }
    if (name == "scalar-transition")
    {
        return PhasePolicyMode::kContextualScalarTransition;
    }
    if (name == "service-scaled-transition")
    {
        return PhasePolicyMode::kServiceScaledTransition;
    }
    return std::nullopt;
}

inline bool phasePolicyUsesContextualScalar(PhasePolicyMode mode) noexcept
{
    return mode != PhasePolicyMode::kExact;
}

inline bool phasePolicyUsesTransition(PhasePolicyMode mode) noexcept
{
    return mode == PhasePolicyMode::kContextualScalarTransition || mode == PhasePolicyMode::kServiceScaledTransition;
}

inline bool phasePolicyUsesServiceScale(PhasePolicyMode mode) noexcept
{
    return mode == PhasePolicyMode::kServiceScaledTransition;
}

} // namespace trt_edgellm::rt
