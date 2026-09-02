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

#include "runtime/phase/policy/phaseOracleH1Selector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace trt_edgellm::rt
{
namespace
{

bool hardFeasible(PhaseOracleH1Candidate const& candidate) noexcept
{
    bool const memorySafe
        = candidate.memoryBudgetBytes == 0U || candidate.hardPeakManagedBytes <= candidate.memoryBudgetBytes;
    return candidate.action.legal() && candidate.projection.valid
        && candidate.projection.completionVector.actionId == candidate.action.actionId && candidate.dependencySafe
        && candidate.contextSafe && candidate.shapeSafe && candidate.ownershipSafe && memorySafe;
}

double robustViolationUs(PhaseOracleH1Candidate const& candidate) noexcept
{
    double result{};
    for (PhaseOracleH1Milestone const& milestone : candidate.milestones)
    {
        if (!std::isfinite(milestone.slackUs))
        {
            continue;
        }
        double const completion = milestone.predictedCompletionUs >= 0.0 ? milestone.predictedCompletionUs
                                                                         : candidate.projection.boundaryUs;
        double const uncertainty = milestone.uncertaintyUs >= 0.0
            ? milestone.uncertaintyUs
            : candidate.projection.robustBoundaryUs - candidate.projection.boundaryUs;
        result = std::max(
            result, std::max(0.0, completion) + std::max(0.0, uncertainty) - std::max(0.0, milestone.slackUs));
    }
    return std::max(0.0, result);
}

double urgencyNormalizedProgress(PhaseOracleH1Candidate const& candidate) noexcept
{
    double result{};
    for (PhaseOracleH1Milestone const& milestone : candidate.milestones)
    {
        if (!milestone.completedAtBoundary || milestone.progressUnits <= 0.0)
        {
            continue;
        }
        double const normalization = std::isfinite(milestone.slackUs) ? std::max(1.0, milestone.slackUs) : 1.0;
        result += milestone.progressUnits / normalization;
    }
    return result;
}

double serviceEfficiency(PhaseOracleH1Candidate const& candidate) noexcept
{
    double const horizon = std::max(candidate.projection.robustBoundaryUs, std::numeric_limits<double>::epsilon());
    return std::max(0.0, candidate.referenceWorkUs) / horizon;
}

} // namespace

PhaseOracleH1CandidateValue phaseOracleH1Value(PhaseOracleH1Candidate const& candidate) noexcept
{
    PhaseOracleH1CandidateValue result;
    result.hardFeasible = hardFeasible(candidate);
    result.robustViolationUs = robustViolationUs(candidate);
    result.urgencyNormalizedProgress = urgencyNormalizedProgress(candidate);
    result.serviceEfficiency = serviceEfficiency(candidate);
    result.releasedOwnershipBytes = candidate.releasedOwnershipBytes;
    return result;
}

PhaseOracleH1Decision PhaseOracleH1Selector::select(std::vector<PhaseOracleH1Candidate> const& candidates) const
{
    PhaseOracleH1Decision result;
    result.inputCandidates = candidates.size();
    if (candidates.empty())
    {
        return result;
    }

    std::vector<PhaseOracleH1CandidateValue> values;
    values.reserve(candidates.size());
    std::vector<size_t> feasible;
    for (size_t index{}; index < candidates.size(); ++index)
    {
        values.push_back(phaseOracleH1Value(candidates[index]));
        if (values.back().hardFeasible)
        {
            feasible.push_back(index);
            if (values.back().robustViolationUs == 0.0)
            {
                ++result.deadlineSafeCandidates;
            }
        }
    }
    result.hardFeasibleCandidates = feasible.size();
    if (feasible.empty())
    {
        result.reason = PhaseOracleH1DecisionReason::kNoHardFeasibleCandidate;
        return result;
    }

    bool const hasSafe = result.deadlineSafeCandidates > 0U;
    auto better = [&](size_t left, size_t right) {
        PhaseOracleH1CandidateValue const& lhs = values[left];
        PhaseOracleH1CandidateValue const& rhs = values[right];
        if (lhs.robustViolationUs != rhs.robustViolationUs)
        {
            return lhs.robustViolationUs < rhs.robustViolationUs;
        }
        if (lhs.urgencyNormalizedProgress != rhs.urgencyNormalizedProgress)
        {
            return lhs.urgencyNormalizedProgress > rhs.urgencyNormalizedProgress;
        }
        if (lhs.serviceEfficiency != rhs.serviceEfficiency)
        {
            return lhs.serviceEfficiency > rhs.serviceEfficiency;
        }
        if (lhs.releasedOwnershipBytes != rhs.releasedOwnershipBytes)
        {
            return lhs.releasedOwnershipBytes > rhs.releasedOwnershipBytes;
        }
        return std::tie(candidates[left].action.actionId, candidates[left].sourceIndex)
            < std::tie(candidates[right].action.actionId, candidates[right].sourceIndex);
    };

    size_t selected = feasible.front();
    for (size_t const index : feasible)
    {
        if (better(index, selected))
        {
            selected = index;
        }
    }
    result.selectedIndex = selected;
    result.reason = hasSafe ? PhaseOracleH1DecisionReason::kDeadlineSafeProgressEfficiency
                            : PhaseOracleH1DecisionReason::kMinimumRobustViolation;
    result.selectedValue = values[selected];
    return result;
}

} // namespace trt_edgellm::rt
