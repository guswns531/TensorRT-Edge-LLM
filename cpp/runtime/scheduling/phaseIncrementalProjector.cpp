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

#include "runtime/phase/mechanism/phaseIncrementalProjector.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace trt_edgellm::rt
{
namespace
{

uint64_t hashCombine(uint64_t seed, uint64_t value) noexcept
{
    constexpr uint64_t kHASH_OFFSET = 0x9e3779b97f4a7c15ULL;
    seed ^= value + kHASH_OFFSET + (seed << 6U) + (seed >> 2U);
    return seed;
}

bool isExecutionPhase(PhaseUnifiedPhase phase) noexcept
{
    return phase == PhaseUnifiedPhase::kEncoder || phase == PhaseUnifiedPhase::kPrefill
        || phase == PhaseUnifiedPhase::kDecode;
}

bool stageMatches(PhaseProjectedRequestStage stage, PhaseUnifiedPhase phase) noexcept
{
    return (stage == PhaseProjectedRequestStage::kEncoder && phase == PhaseUnifiedPhase::kEncoder)
        || (stage == PhaseProjectedRequestStage::kPrefill && phase == PhaseUnifiedPhase::kPrefill)
        || (stage == PhaseProjectedRequestStage::kDecode && phase == PhaseUnifiedPhase::kDecode);
}

PhaseExecutionSet executionSet(std::vector<PhaseInFlightWorkSnapshot> const& work) noexcept
{
    PhaseExecutionSet result{PhaseExecutionSet::kNone};
    for (PhaseInFlightWorkSnapshot const& item : work)
    {
        result = result | phaseExecutionSetForUnifiedPhase(item.phase);
    }
    return result;
}

size_t executionPhaseCount(PhaseExecutionSet set) noexcept
{
    size_t result{};
    result += phaseExecutionSetContains(set, PhaseExecutionSet::kEncoder) ? 1U : 0U;
    result += phaseExecutionSetContains(set, PhaseExecutionSet::kPrefill) ? 1U : 0U;
    result += phaseExecutionSetContains(set, PhaseExecutionSet::kDecode) ? 1U : 0U;
    return result;
}

PhaseProjectedRequest* findRequest(std::vector<PhaseProjectedRequest>& requests, uint64_t requestId) noexcept
{
    auto const iterator = std::find_if(requests.begin(), requests.end(),
        [requestId](PhaseProjectedRequest const& request) { return request.requestId == requestId; });
    return iterator == requests.end() ? nullptr : &*iterator;
}

void applyCompletion(
    PhaseProjectedRequest& request, PhaseUnifiedPhase phase, PhaseProjectedOwnership& ownership) noexcept
{
    switch (phase)
    {
    case PhaseUnifiedPhase::kEncoder:
        request.stage = PhaseProjectedRequestStage::kPrefill;
        if (!request.visionOwned)
        {
            request.visionOwned = true;
            ownership.visionBytes += request.visionLeaseBytes;
        }
        break;
    case PhaseUnifiedPhase::kPrefill:
        request.firstTokenObserved = true;
        ++request.generatedTokens;
        if (request.visionOwned)
        {
            request.visionOwned = false;
            ownership.visionBytes -= request.visionLeaseBytes;
            ownership.reclaimedVisionBytes += request.visionLeaseBytes;
        }
        if (!request.kvOwned)
        {
            request.kvOwned = true;
            ownership.kvBytes += request.kvLeaseBytes;
        }
        if (request.decodeStepsRemaining > 0)
        {
            request.stage = PhaseProjectedRequestStage::kDecode;
        }
        else
        {
            request.stage = PhaseProjectedRequestStage::kComplete;
            request.kvOwned = false;
            ownership.kvBytes -= request.kvLeaseBytes;
            ownership.reclaimedKvBytes += request.kvLeaseBytes;
        }
        break;
    case PhaseUnifiedPhase::kDecode:
        ++request.generatedTokens;
        if (request.decodeStepsRemaining > 0)
        {
            --request.decodeStepsRemaining;
        }
        if (request.decodeStepsRemaining > 0)
        {
            request.stage = PhaseProjectedRequestStage::kDecode;
        }
        else
        {
            request.stage = PhaseProjectedRequestStage::kComplete;
            if (request.kvOwned)
            {
                request.kvOwned = false;
                ownership.kvBytes -= request.kvLeaseBytes;
                ownership.reclaimedKvBytes += request.kvLeaseBytes;
            }
        }
        break;
    case PhaseUnifiedPhase::kNone:
    case PhaseUnifiedPhase::kCopy: break;
    }
}

uint64_t readyCohortId(PhaseUnifiedPhase phase, std::vector<PhaseProjectedRequest> const& requests,
    std::unordered_set<uint64_t> const& inFlightRequests) noexcept
{
    uint64_t result = static_cast<uint64_t>(phase);
    PhaseProjectedRequestStage const stage = phase == PhaseUnifiedPhase::kEncoder ? PhaseProjectedRequestStage::kEncoder
        : phase == PhaseUnifiedPhase::kPrefill                                    ? PhaseProjectedRequestStage::kPrefill
                                                                                  : PhaseProjectedRequestStage::kDecode;
    bool found{};
    for (PhaseProjectedRequest const& request : requests)
    {
        if (request.stage != stage || inFlightRequests.count(request.requestId) != 0U)
        {
            continue;
        }
        found = true;
        result = hashCombine(result, request.requestId);
        result = hashCombine(result, static_cast<uint64_t>(request.stableKvSlotId + 1));
    }
    return found ? result : 0U;
}

} // namespace

char const* phaseProjectionReasonName(PhaseProjectionReason reason) noexcept
{
    switch (reason)
    {
    case PhaseProjectionReason::kProjected: return "projected";
    case PhaseProjectionReason::kIllegalAction: return "illegal_action";
    case PhaseProjectionReason::kActionMismatch: return "action_mismatch";
    case PhaseProjectionReason::kInvalidOutstandingSet: return "invalid_outstanding_set";
    case PhaseProjectionReason::kInvalidCompletionVector: return "invalid_completion_vector";
    case PhaseProjectionReason::kMissingExecution: return "missing_execution";
    case PhaseProjectionReason::kMissingRequest: return "missing_request";
    case PhaseProjectionReason::kRequestStageMismatch: return "request_stage_mismatch";
    }
    return "unknown";
}

PhaseProjectedOwnership phaseProjectedOwnership(std::vector<PhaseProjectedRequest> const& requests) noexcept
{
    PhaseProjectedOwnership result;
    for (PhaseProjectedRequest const& request : requests)
    {
        result.visionBytes += request.visionOwned ? request.visionLeaseBytes : 0U;
        result.kvBytes += request.kvOwned ? request.kvLeaseBytes : 0U;
    }
    return result;
}

std::vector<PhaseIncrementalReadyAction> phaseProjectedReadyActions(
    PhaseIncrementalProjectionSnapshot const& snapshot) noexcept
{
    std::unordered_set<uint64_t> inFlightRequests;
    for (PhaseInFlightWorkSnapshot const& work : snapshot.inFlight.work)
    {
        inFlightRequests.insert(work.requestIds.begin(), work.requestIds.end());
    }

    std::vector<PhaseIncrementalReadyAction> result;
    for (PhaseUnifiedPhase const phase :
        {PhaseUnifiedPhase::kEncoder, PhaseUnifiedPhase::kPrefill, PhaseUnifiedPhase::kDecode})
    {
        uint64_t const candidateId = readyCohortId(phase, snapshot.requests, inFlightRequests);
        if (candidateId != 0U)
        {
            result.push_back({phase, candidateId});
        }
    }
    return result;
}

PhaseIncrementalProjection phaseProjectEarliestCompletion(PhaseIncrementalProjectionSnapshot const& snapshot,
    PhaseIncrementalAction const& action, PhaseCompletionVector const& completionVector,
    double uncertaintyScale) noexcept
{
    PhaseIncrementalProjection result;
    result.successor = snapshot;
    PhaseProjectedOwnership const observedOwnership = phaseProjectedOwnership(result.successor.requests);
    result.successor.ownership.visionBytes = observedOwnership.visionBytes;
    result.successor.ownership.kvBytes = observedOwnership.kvBytes;
    result.completionVector.actionId = completionVector.actionId;
    if (!action.legal())
    {
        result.reason = PhaseProjectionReason::kIllegalAction;
        return result;
    }
    if (completionVector.actionId == 0U || completionVector.actionId != action.actionId)
    {
        result.reason = PhaseProjectionReason::kActionMismatch;
        return result;
    }
    PhaseExecutionSet const observed = executionSet(snapshot.inFlight.work);
    if (observed != snapshot.inFlight.outstanding || observed != action.key.plannedOutstanding
        || executionPhaseCount(observed) == 0U || executionPhaseCount(observed) > 2U)
    {
        result.reason = PhaseProjectionReason::kInvalidOutstandingSet;
        return result;
    }
    if (completionVector.components.size() != snapshot.inFlight.work.size())
    {
        result.reason = PhaseProjectionReason::kInvalidCompletionVector;
        return result;
    }

    uncertaintyScale = std::max(0.0, uncertaintyScale);
    size_t earliestIndex{};
    double earliestRobust = std::numeric_limits<double>::infinity();
    std::unordered_set<uint64_t> executionIds;
    std::unordered_set<PhaseUnifiedPhase> phases;
    std::unordered_set<uint64_t> inFlightRequestIds;
    for (size_t index{}; index < completionVector.components.size(); ++index)
    {
        PhaseCompletionComponent const& component = completionVector.components[index];
        if (!isExecutionPhase(component.phase) || component.executionId == 0U || !std::isfinite(component.completionUs)
            || component.completionUs < 0.0 || !std::isfinite(component.uncertaintyUs) || component.uncertaintyUs < 0.0
            || !executionIds.insert(component.executionId).second || !phases.insert(component.phase).second)
        {
            result.reason = PhaseProjectionReason::kInvalidCompletionVector;
            return result;
        }
        auto const execution = std::find_if(snapshot.inFlight.work.begin(), snapshot.inFlight.work.end(),
            [&component](PhaseInFlightWorkSnapshot const& work) {
                return work.executionId == component.executionId && work.phase == component.phase;
            });
        if (execution == snapshot.inFlight.work.end())
        {
            result.reason = PhaseProjectionReason::kMissingExecution;
            return result;
        }
        if (!action.key.noDispatch && component.incumbent != (component.phase == action.key.incumbentPhase))
        {
            result.reason = PhaseProjectionReason::kInvalidCompletionVector;
            return result;
        }
        for (uint64_t const requestId : execution->requestIds)
        {
            if (!inFlightRequestIds.insert(requestId).second)
            {
                result.reason = PhaseProjectionReason::kMissingRequest;
                return result;
            }
            PhaseProjectedRequest* request = findRequest(result.successor.requests, requestId);
            if (request == nullptr)
            {
                result.reason = PhaseProjectionReason::kMissingRequest;
                return result;
            }
            if (!stageMatches(request->stage, component.phase))
            {
                result.reason = PhaseProjectionReason::kRequestStageMismatch;
                return result;
            }
        }
        double const robust = component.completionUs + uncertaintyScale * component.uncertaintyUs;
        PhaseCompletionComponent const& earliest = completionVector.components[earliestIndex];
        if (robust < earliestRobust
            || (robust == earliestRobust
                && (component.phase < earliest.phase
                    || (component.phase == earliest.phase && component.executionId < earliest.executionId))))
        {
            earliestIndex = index;
            earliestRobust = robust;
        }
    }

    result.completed = completionVector.components[earliestIndex];
    result.boundaryUs = result.completed.completionUs;
    result.robustBoundaryUs = earliestRobust;
    auto const completedExecution = std::find_if(result.successor.inFlight.work.begin(),
        result.successor.inFlight.work.end(), [&result](PhaseInFlightWorkSnapshot const& work) {
            return work.executionId == result.completed.executionId && work.phase == result.completed.phase;
        });
    if (completedExecution == result.successor.inFlight.work.end())
    {
        result.reason = PhaseProjectionReason::kMissingExecution;
        return result;
    }

    std::unordered_set<uint64_t> completedRequests;
    for (uint64_t const requestId : completedExecution->requestIds)
    {
        if (!completedRequests.insert(requestId).second)
        {
            result.reason = PhaseProjectionReason::kMissingRequest;
            return result;
        }
        PhaseProjectedRequest* request = findRequest(result.successor.requests, requestId);
        if (request == nullptr)
        {
            result.reason = PhaseProjectionReason::kMissingRequest;
            return result;
        }
        if (!stageMatches(request->stage, result.completed.phase))
        {
            result.reason = PhaseProjectionReason::kRequestStageMismatch;
            return result;
        }
    }

    for (uint64_t const requestId : completedExecution->requestIds)
    {
        applyCompletion(
            *findRequest(result.successor.requests, requestId), result.completed.phase, result.successor.ownership);
    }
    result.successor.inFlight.work.erase(completedExecution);
    result.successor.inFlight.outstanding = executionSet(result.successor.inFlight.work);
    result.successor.inFlight.hostSnapshotNs += static_cast<uint64_t>(std::llround(result.boundaryUs * 1000.0));
    for (PhaseInFlightWorkSnapshot& work : result.successor.inFlight.work)
    {
        work.dispatchAgeUs += result.boundaryUs;
    }
    ++result.successor.epoch;

    result.completionVector.components.reserve(completionVector.components.size() - 1U);
    for (size_t index{}; index < completionVector.components.size(); ++index)
    {
        if (index == earliestIndex)
        {
            continue;
        }
        PhaseCompletionComponent residual = completionVector.components[index];
        residual.completionUs = std::max(0.0, residual.completionUs - result.boundaryUs);
        result.completionVector.components.push_back(residual);
    }
    result.ready = phaseProjectedReadyActions(result.successor);
    result.legalActions = phaseEnumerateIncrementalActions(result.successor.inFlight, result.ready);
    result.valid = true;
    result.reason = PhaseProjectionReason::kProjected;
    return result;
}

bool PhaseReplayCompletionPredictor::insert(PhaseCompletionVector completionVector)
{
    if (completionVector.actionId == 0U || completionVector.components.empty())
    {
        return false;
    }
    for (PhaseCompletionComponent const& component : completionVector.components)
    {
        if (!isExecutionPhase(component.phase) || component.executionId == 0U || !std::isfinite(component.completionUs)
            || component.completionUs < 0.0 || !std::isfinite(component.uncertaintyUs) || component.uncertaintyUs < 0.0)
        {
            return false;
        }
    }
    mVectors.insert_or_assign(completionVector.actionId, std::move(completionVector));
    return true;
}

std::optional<PhaseCompletionVector> PhaseReplayCompletionPredictor::predict(PhaseIncrementalAction const& action) const
{
    if (!action.legal())
    {
        return std::nullopt;
    }
    auto const found = mVectors.find(action.actionId);
    return found == mVectors.end() ? std::nullopt : std::optional<PhaseCompletionVector>{found->second};
}

size_t PhaseReplayCompletionPredictor::size() const noexcept
{
    return mVectors.size();
}

} // namespace trt_edgellm::rt
