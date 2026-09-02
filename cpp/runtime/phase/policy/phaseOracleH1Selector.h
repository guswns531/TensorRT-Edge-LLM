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

#include "runtime/phase/mechanism/phaseIncrementalProjector.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace trt_edgellm::rt
{

//! One protected request milestone evaluated at the candidate's first
//! concrete completion boundary. Progress units are policy-neutral work units
//! supplied by the replay contract, not workload labels.
struct PhaseOracleH1Milestone
{
    uint64_t requestId{};
    double slackUs{};
    //! Negative values inherit the projected boundary/vector uncertainty.
    double predictedCompletionUs{-1.0};
    double uncertaintyUs{-1.0};
    double progressUnits{};
    bool completedAtBoundary{};
};

//! A measured/replayed legal candidate. The deterministic M4 projector owns
//! request/DAG and ownership transitions; this structure adds only the
//! objective inputs that are not part of execution correctness.
struct PhaseOracleH1Candidate
{
    size_t sourceIndex{};
    PhaseIncrementalAction action;
    PhaseIncrementalProjection projection;
    std::vector<PhaseOracleH1Milestone> milestones;
    double referenceWorkUs{};
    size_t hardPeakManagedBytes{};
    size_t memoryBudgetBytes{};
    size_t releasedOwnershipBytes{};
    bool dependencySafe{true};
    bool contextSafe{true};
    bool shapeSafe{true};
    bool ownershipSafe{true};
};

enum class PhaseOracleH1DecisionReason : uint8_t
{
    kNoCandidate,
    kNoHardFeasibleCandidate,
    kMinimumRobustViolation,
    kDeadlineSafeProgressEfficiency,
};

struct PhaseOracleH1CandidateValue
{
    bool hardFeasible{};
    double robustViolationUs{};
    double urgencyNormalizedProgress{};
    double serviceEfficiency{};
    size_t releasedOwnershipBytes{};
};

struct PhaseOracleH1Decision
{
    std::optional<size_t> selectedIndex;
    PhaseOracleH1DecisionReason reason{PhaseOracleH1DecisionReason::kNoCandidate};
    size_t inputCandidates{};
    size_t hardFeasibleCandidates{};
    size_t deadlineSafeCandidates{};
    PhaseOracleH1CandidateValue selectedValue;
};

//! Evaluate one candidate using only measured/replayed H1 state.
PhaseOracleH1CandidateValue phaseOracleH1Value(PhaseOracleH1Candidate const& candidate) noexcept;

//! Offline/measured oracle for validating the H1 objective. It deliberately
//! does not predict an unobserved counterfactual and is not a production
//! scheduler mode.
class PhaseOracleH1Selector
{
public:
    PhaseOracleH1Decision select(std::vector<PhaseOracleH1Candidate> const& candidates) const;
};

} // namespace trt_edgellm::rt
