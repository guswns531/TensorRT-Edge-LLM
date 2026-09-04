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

PhaseProjectedRequest const* findRequest(
    std::vector<PhaseProjectedRequest> const& requests, uint64_t requestId) noexcept
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

char const* phaseFrozenReplayReasonName(PhaseFrozenReplayReason reason) noexcept
{
    switch (reason)
    {
    case PhaseFrozenReplayReason::kReplayed: return "replayed";
    case PhaseFrozenReplayReason::kInvalidSnapshot: return "invalid_snapshot";
    case PhaseFrozenReplayReason::kMissingAction: return "missing_action";
    case PhaseFrozenReplayReason::kActionNotLegal: return "action_not_legal";
    case PhaseFrozenReplayReason::kEnvelopeNotReady: return "envelope_not_ready";
    case PhaseFrozenReplayReason::kEnvelopeActionMismatch: return "envelope_action_mismatch";
    case PhaseFrozenReplayReason::kInvalidOutcome: return "invalid_outcome";
    }
    return "unknown";
}

namespace
{

bool validFrozenProjection(PhaseIncrementalProjectionSnapshot const& projection) noexcept
{
    PhaseExecutionSet const observed = executionSet(projection.inFlight.work);
    if (observed != projection.inFlight.outstanding || executionPhaseCount(observed) > 2U)
    {
        return false;
    }
    std::unordered_set<uint64_t> requestIds;
    for (PhaseProjectedRequest const& request : projection.requests)
    {
        if (request.requestId == 0U || !requestIds.insert(request.requestId).second)
        {
            return false;
        }
    }
    std::unordered_set<uint64_t> executionIds;
    std::unordered_set<uint64_t> inFlightRequestIds;
    for (PhaseInFlightWorkSnapshot const& work : projection.inFlight.work)
    {
        if (!isExecutionPhase(work.phase) || work.executionId == 0U || work.requestIds.empty()
            || !executionIds.insert(work.executionId).second)
        {
            return false;
        }
        for (uint64_t const requestId : work.requestIds)
        {
            PhaseProjectedRequest const* request = findRequest(projection.requests, requestId);
            if (request == nullptr || !stageMatches(request->stage, work.phase)
                || !inFlightRequestIds.insert(requestId).second)
            {
                return false;
            }
        }
    }
    PhaseProjectedOwnership const ownership = phaseProjectedOwnership(projection.requests);
    return ownership.visionBytes == projection.ownership.visionBytes
        && ownership.kvBytes == projection.ownership.kvBytes;
}

uint64_t frozenSnapshotId(PhaseIncrementalProjectionSnapshot const& projection,
    std::vector<PhaseIncrementalAction> const& frontier, uint64_t scalarPolicyStateSignature) noexcept
{
    constexpr uint64_t kFROZEN_OFFSET = 1469598103934665603ULL;
    uint64_t result = hashCombine(kFROZEN_OFFSET, projection.epoch);
    result = hashCombine(result, projection.inFlight.hostSnapshotNs);
    result = hashCombine(result, static_cast<uint64_t>(projection.inFlight.outstanding));
    for (PhaseInFlightWorkSnapshot const& work : projection.inFlight.work)
    {
        result = hashCombine(result, static_cast<uint64_t>(work.phase));
        result = hashCombine(result, work.executionId);
        for (uint64_t const requestId : work.requestIds)
        {
            result = hashCombine(result, requestId);
        }
    }
    for (PhaseProjectedRequest const& request : projection.requests)
    {
        result = hashCombine(result, request.requestId);
        result = hashCombine(result, static_cast<uint64_t>(request.stage));
        result = hashCombine(result, static_cast<uint64_t>(request.stableKvSlotId + 1));
        result = hashCombine(result, static_cast<uint64_t>(request.decodeStepsRemaining));
        result = hashCombine(result, static_cast<uint64_t>(request.generatedTokens));
        result = hashCombine(result, request.visionLeaseBytes);
        result = hashCombine(result, request.kvLeaseBytes);
        result = hashCombine(result, static_cast<uint64_t>(request.visionOwned));
        result = hashCombine(result, static_cast<uint64_t>(request.kvOwned));
        result = hashCombine(result, static_cast<uint64_t>(request.firstTokenObserved));
    }
    result = hashCombine(result, projection.ownership.visionBytes);
    result = hashCombine(result, projection.ownership.kvBytes);
    result = hashCombine(result, scalarPolicyStateSignature);
    for (PhaseIncrementalAction const& action : frontier)
    {
        result = hashCombine(result, action.actionId);
    }
    return result;
}

std::optional<PhaseIncrementalAction> continuationAction(PhaseIncrementalProjection const& projection) noexcept
{
    auto const found = std::find_if(projection.legalActions.begin(), projection.legalActions.end(),
        [](PhaseIncrementalAction const& action) { return action.legal() && action.key.noDispatch; });
    return found == projection.legalActions.end() ? std::nullopt : std::optional<PhaseIncrementalAction>{*found};
}

void countReadyRows(PhaseIncrementalProjectionSnapshot const& snapshot, PhaseFrozenReplayTrajectory& trajectory)
{
    std::unordered_set<uint64_t> inFlight;
    for (PhaseInFlightWorkSnapshot const& work : snapshot.inFlight.work)
    {
        inFlight.insert(work.requestIds.begin(), work.requestIds.end());
    }
    for (PhaseProjectedRequest const& request : snapshot.requests)
    {
        if (inFlight.count(request.requestId) != 0U)
        {
            continue;
        }
        trajectory.encoderReadyRows += request.stage == PhaseProjectedRequestStage::kEncoder ? 1U : 0U;
        trajectory.prefillReadyRows += request.stage == PhaseProjectedRequestStage::kPrefill ? 1U : 0U;
        trajectory.decodeReadyRows += request.stage == PhaseProjectedRequestStage::kDecode ? 1U : 0U;
    }
}

PhaseCompletionVector physicalVector(uint64_t actionId, std::vector<PhaseOutcomeComponentReference> const& components,
    size_t firstIndex, double makespanUs, double uncertaintyUs) noexcept
{
    PhaseCompletionVector result;
    result.actionId = actionId;
    result.components.reserve(components.size());
    if (components.size() == 1U)
    {
        PhaseOutcomeComponentReference const& component = components.front();
        result.components.push_back(
            {component.phase, component.executionId, makespanUs, uncertaintyUs, component.incumbent});
        return result;
    }
    for (size_t index{}; index < components.size(); ++index)
    {
        PhaseOutcomeComponentReference const& component = components[index];
        double const completionUs
            = index == firstIndex ? std::min(makespanUs, std::max(0.0, component.isolatedReferenceUs)) : makespanUs;
        result.components.push_back(
            {component.phase, component.executionId, completionUs, uncertaintyUs, component.incumbent});
    }
    return result;
}

PhaseCompletionVector effectVector(uint64_t actionId, std::vector<PhaseOutcomeComponentReference> const& components,
    PhaseEffectOutcomeEstimate const& estimate, double orderMargin) noexcept
{
    double serialReferenceUs{};
    size_t incumbentIndex{};
    for (size_t index{}; index < components.size(); ++index)
    {
        serialReferenceUs += components[index].isolatedReferenceUs;
        if (components[index].incumbent)
        {
            incumbentIndex = index;
        }
    }
    size_t const newcomerIndex = incumbentIndex == 0U ? 1U : 0U;
    double const makespanUs = std::max(0.0, serialReferenceUs * (1.0 - estimate.compressionMean));
    double const marginUs = orderMargin * serialReferenceUs;
    double incumbentUs = std::clamp(
        components[incumbentIndex].isolatedReferenceUs * (1.0 + estimate.incumbentStretchMean), 0.0, makespanUs);
    double newcomerUs{};
    if (marginUs >= 0.0)
    {
        newcomerUs = makespanUs;
        incumbentUs = std::min(incumbentUs, std::max(0.0, makespanUs - marginUs));
    }
    else
    {
        incumbentUs = makespanUs;
        newcomerUs = std::max(0.0, makespanUs + marginUs);
    }
    double const uncertaintyUs = serialReferenceUs
        * (std::max(0.0, estimate.compressionUncertainty) + std::max(0.0, estimate.incumbentStretchUncertainty)
            + std::max(0.0, estimate.orderMarginUncertainty));
    PhaseCompletionVector result;
    result.actionId = actionId;
    result.components.resize(components.size());
    result.components[incumbentIndex]
        = {components[incumbentIndex].phase, components[incumbentIndex].executionId, incumbentUs, uncertaintyUs, true};
    result.components[newcomerIndex]
        = {components[newcomerIndex].phase, components[newcomerIndex].executionId, newcomerUs, uncertaintyUs, false};
    return result;
}

} // namespace

std::optional<PhaseFrozenDecisionSnapshot> phaseFreezeDecisionSnapshot(PhaseIncrementalProjectionSnapshot projection,
    std::vector<PhaseIncrementalAction> frontier, uint64_t scalarPolicyStateSignature) noexcept
{
    if (!validFrozenProjection(projection) || frontier.empty())
    {
        return std::nullopt;
    }
    std::unordered_set<uint64_t> actionIds;
    for (PhaseIncrementalAction const& action : frontier)
    {
        if (!action.legal() || action.actionId == 0U || !actionIds.insert(action.actionId).second)
        {
            return std::nullopt;
        }
    }
    PhaseFrozenDecisionSnapshot result;
    result.scalarPolicyStateSignature = scalarPolicyStateSignature;
    result.projection = std::move(projection);
    result.frontier = std::move(frontier);
    result.snapshotId = frozenSnapshotId(result.projection, result.frontier, scalarPolicyStateSignature);
    return result;
}

PhaseOutcomeEnvelope phaseScalarOutcomeEnvelope(uint64_t actionId,
    std::vector<PhaseOutcomeComponentReference> const& components, double actionMakespanUs,
    double uncertaintyUs) noexcept
{
    PhaseOutcomeEnvelope result;
    result.actionId = actionId;
    result.fidelity = PhaseOutcomeFidelity::kScalarEnvelope;
    if (actionId == 0U || components.empty() || components.size() > 2U || !std::isfinite(actionMakespanUs)
        || actionMakespanUs < 0.0 || !std::isfinite(uncertaintyUs) || uncertaintyUs < 0.0)
    {
        return result;
    }
    std::unordered_set<uint64_t> executionIds;
    size_t incumbents{};
    for (PhaseOutcomeComponentReference const& component : components)
    {
        if (!isExecutionPhase(component.phase) || component.executionId == 0U
            || !executionIds.insert(component.executionId).second || !std::isfinite(component.isolatedReferenceUs)
            || component.isolatedReferenceUs < 0.0)
        {
            return result;
        }
        incumbents += component.incumbent ? 1U : 0U;
    }
    if (components.size() == 2U && incumbents != 1U)
    {
        return result;
    }
    result.alternatives.push_back(physicalVector(actionId, components, 0U, actionMakespanUs, uncertaintyUs));
    if (components.size() == 2U)
    {
        result.alternatives.push_back(physicalVector(actionId, components, 1U, actionMakespanUs, uncertaintyUs));
    }
    result.ready = true;
    return result;
}

PhaseOutcomeEnvelope phaseEffectOutcomeEnvelope(uint64_t actionId,
    std::vector<PhaseOutcomeComponentReference> const& components, PhaseEffectOutcomeEstimate const& estimate) noexcept
{
    PhaseOutcomeEnvelope result;
    result.actionId = actionId;
    result.fidelity = PhaseOutcomeFidelity::kEffectVector;
    if (!estimate.ready || actionId == 0U || components.size() != 2U || !std::isfinite(estimate.compressionMean)
        || !std::isfinite(estimate.compressionUncertainty) || estimate.compressionUncertainty < 0.0
        || !std::isfinite(estimate.incumbentStretchMean) || !std::isfinite(estimate.incumbentStretchUncertainty)
        || estimate.incumbentStretchUncertainty < 0.0 || !std::isfinite(estimate.orderMarginMean)
        || !std::isfinite(estimate.orderMarginUncertainty) || estimate.orderMarginUncertainty < 0.0)
    {
        return result;
    }
    size_t const incumbents
        = static_cast<size_t>(components[0].incumbent) + static_cast<size_t>(components[1].incumbent);
    if (incumbents != 1U || components[0].executionId == 0U || components[1].executionId == 0U
        || components[0].executionId == components[1].executionId || !isExecutionPhase(components[0].phase)
        || !isExecutionPhase(components[1].phase) || components[0].phase == components[1].phase
        || !std::isfinite(components[0].isolatedReferenceUs) || components[0].isolatedReferenceUs <= 0.0
        || !std::isfinite(components[1].isolatedReferenceUs) || components[1].isolatedReferenceUs <= 0.0)
    {
        return result;
    }
    double const lower = estimate.orderMarginMean - estimate.orderMarginUncertainty;
    double const upper = estimate.orderMarginMean + estimate.orderMarginUncertainty;
    result.alternatives.push_back(effectVector(actionId, components, estimate, estimate.orderMarginMean));
    if (lower <= 0.0 && upper >= 0.0)
    {
        double const magnitude
            = std::max({std::abs(estimate.orderMarginMean), estimate.orderMarginUncertainty, 1.0e-6});
        result.alternatives.clear();
        result.alternatives.push_back(effectVector(actionId, components, estimate, -magnitude));
        result.alternatives.push_back(effectVector(actionId, components, estimate, magnitude));
    }
    result.ready = true;
    return result;
}

PhaseOutcomeEnvelope phaseCompletionOutcomeEnvelope(
    PhaseCompletionVector completion, PhaseOutcomeFidelity fidelity, bool ready) noexcept
{
    PhaseOutcomeEnvelope result;
    result.actionId = completion.actionId;
    result.fidelity = fidelity;
    result.ready = ready;
    if (ready)
    {
        result.alternatives.push_back(std::move(completion));
    }
    return result;
}

PhaseFrozenReplayResult phaseReplayFrozenOutcome(PhaseFrozenDecisionSnapshot const& snapshot, uint64_t actionId,
    PhaseOutcomeEnvelope const& envelope, double uncertaintyScale) noexcept
{
    PhaseFrozenReplayResult result;
    result.snapshotId = snapshot.snapshotId;
    result.actionId = actionId;
    result.fidelity = envelope.fidelity;
    if (snapshot.snapshotId == 0U || !validFrozenProjection(snapshot.projection) || snapshot.frontier.empty())
    {
        result.reason = PhaseFrozenReplayReason::kInvalidSnapshot;
        return result;
    }
    auto const action = std::find_if(snapshot.frontier.begin(), snapshot.frontier.end(),
        [actionId](PhaseIncrementalAction const& candidate) { return candidate.actionId == actionId; });
    if (action == snapshot.frontier.end())
    {
        result.reason = PhaseFrozenReplayReason::kMissingAction;
        return result;
    }
    if (!action->legal())
    {
        result.reason = PhaseFrozenReplayReason::kActionNotLegal;
        return result;
    }
    if (!envelope.ready || envelope.alternatives.empty())
    {
        result.reason = PhaseFrozenReplayReason::kEnvelopeNotReady;
        return result;
    }
    if (envelope.actionId != actionId)
    {
        result.reason = PhaseFrozenReplayReason::kEnvelopeActionMismatch;
        return result;
    }
    for (PhaseCompletionVector completion : envelope.alternatives)
    {
        if (completion.actionId != actionId)
        {
            result.reason = PhaseFrozenReplayReason::kEnvelopeActionMismatch;
            result.trajectories.clear();
            return result;
        }
        PhaseFrozenReplayTrajectory trajectory;
        PhaseIncrementalProjection first
            = phaseProjectEarliestCompletion(snapshot.projection, *action, completion, uncertaintyScale);
        if (!first.valid)
        {
            result.reason = PhaseFrozenReplayReason::kInvalidOutcome;
            result.trajectories.clear();
            return result;
        }
        trajectory.robustHorizonUs = first.robustBoundaryUs;
        trajectory.boundaries.push_back(first);
        if (!first.completionVector.components.empty())
        {
            std::optional<PhaseIncrementalAction> const wait = continuationAction(first);
            if (!wait.has_value())
            {
                result.reason = PhaseFrozenReplayReason::kInvalidOutcome;
                result.trajectories.clear();
                return result;
            }
            PhaseCompletionVector residual = first.completionVector;
            residual.actionId = wait->actionId;
            PhaseIncrementalProjection second
                = phaseProjectEarliestCompletion(first.successor, *wait, residual, uncertaintyScale);
            if (!second.valid)
            {
                result.reason = PhaseFrozenReplayReason::kInvalidOutcome;
                result.trajectories.clear();
                return result;
            }
            trajectory.robustHorizonUs += second.robustBoundaryUs;
            trajectory.boundaries.push_back(second);
        }
        PhaseIncrementalProjectionSnapshot const& terminal = trajectory.boundaries.back().successor;
        trajectory.ownership = terminal.ownership;
        countReadyRows(terminal, trajectory);
        result.worstCaseRobustHorizonUs = std::max(result.worstCaseRobustHorizonUs, trajectory.robustHorizonUs);
        trajectory.valid = true;
        result.trajectories.push_back(std::move(trajectory));
    }
    result.valid = true;
    result.reason = PhaseFrozenReplayReason::kReplayed;
    return result;
}

} // namespace trt_edgellm::rt
