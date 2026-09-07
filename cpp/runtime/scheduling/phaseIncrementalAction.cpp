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

#include "runtime/phase/mechanism/phaseIncrementalAction.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace trt_edgellm::rt
{
namespace
{

constexpr uint64_t kHASH_OFFSET = 1469598103934665603ULL;
constexpr uint64_t kHASH_PRIME = 1099511628211ULL;

uint64_t hashCombine(uint64_t seed, uint64_t value) noexcept
{
    seed ^= value;
    seed *= kHASH_PRIME;
    return seed;
}

uint8_t executionBits(PhaseExecutionSet phases) noexcept
{
    return static_cast<uint8_t>(phases);
}

bool validExecutionSet(PhaseExecutionSet phases) noexcept
{
    constexpr uint8_t kVALID_BITS = static_cast<uint8_t>(PhaseExecutionSet::kEncoder)
        | static_cast<uint8_t>(PhaseExecutionSet::kPrefill) | static_cast<uint8_t>(PhaseExecutionSet::kDecode);
    return (executionBits(phases) & static_cast<uint8_t>(~kVALID_BITS)) == 0U;
}

int32_t executionCount(PhaseExecutionSet phases) noexcept
{
    uint8_t bits = executionBits(phases);
    int32_t count{};
    while (bits != 0U)
    {
        count += static_cast<int32_t>(bits & 1U);
        bits >>= 1U;
    }
    return count;
}

PhaseUnifiedPhase onlyPhase(PhaseExecutionSet phases) noexcept
{
    if (phases == PhaseExecutionSet::kEncoder)
    {
        return PhaseUnifiedPhase::kEncoder;
    }
    if (phases == PhaseExecutionSet::kPrefill)
    {
        return PhaseUnifiedPhase::kPrefill;
    }
    if (phases == PhaseExecutionSet::kDecode)
    {
        return PhaseUnifiedPhase::kDecode;
    }
    return PhaseUnifiedPhase::kNone;
}

uint64_t pairCohortId(uint64_t incumbent, uint64_t newcomer) noexcept
{
    uint64_t result = hashCombine(kHASH_OFFSET, incumbent);
    return hashCombine(result, newcomer);
}

PhaseIncrementalAction buildAction(
    PhaseIncrementalActionKey key, bool dependencySafe = true, bool contextSafe = true, bool shapeSafe = true) noexcept
{
    PhaseIncrementalAction result;
    result.key = key;
    result.legality = phaseIncrementalActionLegality(key, dependencySafe, contextSafe, shapeSafe);
    result.actionId = phaseIncrementalActionId(key);
    return result;
}

PhaseUnifiedActionDirection canonicalDirection(PhaseGlobalActionKind action) noexcept
{
    switch (action)
    {
    case PhaseGlobalActionKind::kEncoderPrefill: return PhaseUnifiedActionDirection::kEncoderToPrefill;
    case PhaseGlobalActionKind::kEncoderDecode: return PhaseUnifiedActionDirection::kEncoderToDecode;
    case PhaseGlobalActionKind::kPrefillDecode: return PhaseUnifiedActionDirection::kPrefillToDecode;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kEncoder:
    case PhaseGlobalActionKind::kPrefill:
    case PhaseGlobalActionKind::kDecode:
    case PhaseGlobalActionKind::kWait: return PhaseUnifiedActionDirection::kIdleLaunch;
    }
    return PhaseUnifiedActionDirection::kNone;
}

} // namespace

bool PhaseIncrementalActionKey::operator==(PhaseIncrementalActionKey const& other) const noexcept
{
    return cohortId == other.cohortId && outstandingBefore == other.outstandingBefore
        && plannedOutstanding == other.plannedOutstanding && incumbentPhase == other.incumbentPhase
        && newcomerPhase == other.newcomerPhase && direction == other.direction && startSkew == other.startSkew
        && noDispatch == other.noDispatch;
}

char const* phaseStartSkewBucketName(PhaseStartSkewBucket bucket) noexcept
{
    switch (bucket)
    {
    case PhaseStartSkewBucket::kUnknown: return "unknown";
    case PhaseStartSkewBucket::kImmediate: return "offset_0";
    case PhaseStartSkewBucket::kQuarter: return "offset_25";
    case PhaseStartSkewBucket::kHalf: return "offset_50";
    case PhaseStartSkewBucket::kThreeQuarter: return "offset_75";
    case PhaseStartSkewBucket::kNearComplete: return "offset_90";
    case PhaseStartSkewBucket::kSerial: return "serial_realization";
    }
    return "unknown";
}

char const* phaseIncrementalLegalityReasonName(PhaseIncrementalLegalityReason reason) noexcept
{
    switch (reason)
    {
    case PhaseIncrementalLegalityReason::kLegal: return "legal";
    case PhaseIncrementalLegalityReason::kInvalidOutstandingSet: return "invalid_outstanding_set";
    case PhaseIncrementalLegalityReason::kNoDispatchMismatch: return "no_dispatch_mismatch";
    case PhaseIncrementalLegalityReason::kMissingNewcomer: return "missing_newcomer";
    case PhaseIncrementalLegalityReason::kSameContextReenqueue: return "same_context_reenqueue";
    case PhaseIncrementalLegalityReason::kThirdContext: return "third_context";
    case PhaseIncrementalLegalityReason::kIncumbentMismatch: return "incumbent_mismatch";
    case PhaseIncrementalLegalityReason::kDirectionMismatch: return "direction_mismatch";
    case PhaseIncrementalLegalityReason::kPlannedOutstandingMismatch: return "planned_outstanding_mismatch";
    case PhaseIncrementalLegalityReason::kDependencyUnsafe: return "dependency_unsafe";
    case PhaseIncrementalLegalityReason::kContextUnsafe: return "context_unsafe";
    case PhaseIncrementalLegalityReason::kShapeUnsafe: return "shape_unsafe";
    }
    return "unknown";
}

PhaseStartSkewBucket phaseStartSkewBucketFromFraction(double fraction, bool intervalsOverlap) noexcept
{
    if (!intervalsOverlap)
    {
        return PhaseStartSkewBucket::kSerial;
    }
    if (!std::isfinite(fraction))
    {
        return PhaseStartSkewBucket::kUnknown;
    }
    struct Bucket
    {
        double fraction;
        PhaseStartSkewBucket bucket;
    };
    constexpr std::array<Bucket, 5U> kBUCKETS{{{0.0, PhaseStartSkewBucket::kImmediate},
        {0.25, PhaseStartSkewBucket::kQuarter}, {0.50, PhaseStartSkewBucket::kHalf},
        {0.75, PhaseStartSkewBucket::kThreeQuarter}, {0.90, PhaseStartSkewBucket::kNearComplete}}};
    auto const closest = std::min_element(kBUCKETS.begin(), kBUCKETS.end(), [fraction](Bucket left, Bucket right) {
        return std::abs(fraction - left.fraction) < std::abs(fraction - right.fraction);
    });
    return closest->bucket;
}

PhaseIncrementalLegalityReason phaseIncrementalActionLegality(
    PhaseIncrementalActionKey const& key, bool dependencySafe, bool contextSafe, bool shapeSafe) noexcept
{
    if (!validExecutionSet(key.outstandingBefore) || !validExecutionSet(key.plannedOutstanding)
        || executionCount(key.outstandingBefore) > 2)
    {
        return PhaseIncrementalLegalityReason::kInvalidOutstandingSet;
    }
    if (key.noDispatch)
    {
        bool const exact = key.newcomerPhase == PhaseUnifiedPhase::kNone
            && key.direction == PhaseUnifiedActionDirection::kNone && key.plannedOutstanding == key.outstandingBefore;
        return exact ? PhaseIncrementalLegalityReason::kLegal : PhaseIncrementalLegalityReason::kNoDispatchMismatch;
    }
    PhaseExecutionSet const newcomer = phaseExecutionSetForUnifiedPhase(key.newcomerPhase);
    if (newcomer == PhaseExecutionSet::kNone)
    {
        return PhaseIncrementalLegalityReason::kMissingNewcomer;
    }
    if (phaseExecutionSetContains(key.outstandingBefore, newcomer))
    {
        return PhaseIncrementalLegalityReason::kSameContextReenqueue;
    }
    if (executionCount(key.outstandingBefore) >= 2)
    {
        return PhaseIncrementalLegalityReason::kThirdContext;
    }
    if (executionCount(key.plannedOutstanding) > 2)
    {
        return PhaseIncrementalLegalityReason::kThirdContext;
    }
    PhaseExecutionSet const expected = key.outstandingBefore | newcomer;
    if (key.outstandingBefore == PhaseExecutionSet::kNone && key.incumbentPhase != PhaseUnifiedPhase::kNone)
    {
        PhaseExecutionSet const incumbent = phaseExecutionSetForUnifiedPhase(key.incumbentPhase);
        if (incumbent == PhaseExecutionSet::kNone || incumbent == newcomer)
        {
            return PhaseIncrementalLegalityReason::kIncumbentMismatch;
        }
        if (key.direction != phaseUnifiedActionDirection(incumbent, key.newcomerPhase))
        {
            return PhaseIncrementalLegalityReason::kDirectionMismatch;
        }
        if (key.plannedOutstanding != (incumbent | newcomer))
        {
            return PhaseIncrementalLegalityReason::kPlannedOutstandingMismatch;
        }
    }
    else if (key.outstandingBefore == PhaseExecutionSet::kNone)
    {
        if (key.direction != PhaseUnifiedActionDirection::kIdleLaunch)
        {
            return PhaseIncrementalLegalityReason::kDirectionMismatch;
        }
        if (key.plannedOutstanding != newcomer)
        {
            return PhaseIncrementalLegalityReason::kPlannedOutstandingMismatch;
        }
    }
    else
    {
        PhaseUnifiedPhase const incumbent = onlyPhase(key.outstandingBefore);
        if (incumbent == PhaseUnifiedPhase::kNone || key.incumbentPhase != incumbent)
        {
            return PhaseIncrementalLegalityReason::kIncumbentMismatch;
        }
        if (key.direction != phaseUnifiedActionDirection(key.outstandingBefore, key.newcomerPhase))
        {
            return PhaseIncrementalLegalityReason::kDirectionMismatch;
        }
        if (key.plannedOutstanding != expected)
        {
            return PhaseIncrementalLegalityReason::kPlannedOutstandingMismatch;
        }
    }
    if (!dependencySafe)
    {
        return PhaseIncrementalLegalityReason::kDependencyUnsafe;
    }
    if (!contextSafe)
    {
        return PhaseIncrementalLegalityReason::kContextUnsafe;
    }
    if (!shapeSafe)
    {
        return PhaseIncrementalLegalityReason::kShapeUnsafe;
    }
    return PhaseIncrementalLegalityReason::kLegal;
}

uint64_t phaseIncrementalActionId(PhaseIncrementalActionKey const& key) noexcept
{
    uint64_t result = hashCombine(kHASH_OFFSET, key.cohortId);
    result = hashCombine(result, executionBits(key.outstandingBefore));
    result = hashCombine(result, executionBits(key.plannedOutstanding));
    result = hashCombine(result, static_cast<uint64_t>(key.incumbentPhase));
    result = hashCombine(result, static_cast<uint64_t>(key.newcomerPhase));
    result = hashCombine(result, static_cast<uint64_t>(key.direction));
    result = hashCombine(result, static_cast<uint64_t>(static_cast<int32_t>(key.startSkew) + 1));
    return hashCombine(result, static_cast<uint64_t>(key.noDispatch));
}

std::vector<PhaseIncrementalAction> phaseEnumerateIncrementalActions(PhaseInFlightSnapshot const& snapshot,
    std::vector<PhaseIncrementalReadyAction> const& ready, PhaseStartSkewBucket incumbentStartSkew)
{
    std::vector<PhaseIncrementalAction> result;
    if (!validExecutionSet(snapshot.outstanding) || executionCount(snapshot.outstanding) > 2)
    {
        return result;
    }
    PhaseExecutionSet readyPhases{PhaseExecutionSet::kNone};
    for (PhaseIncrementalReadyAction const& candidate : ready)
    {
        PhaseExecutionSet const phase = phaseExecutionSetForUnifiedPhase(candidate.phase);
        if (candidate.candidateId == 0U || phase == PhaseExecutionSet::kNone
            || phaseExecutionSetContains(readyPhases, phase))
        {
            return {};
        }
        readyPhases = readyPhases | phase;
    }
    uint64_t waitCohortId = kHASH_OFFSET;
    for (PhaseInFlightWorkSnapshot const& work : snapshot.work)
    {
        waitCohortId = hashCombine(waitCohortId, work.executionId);
    }
    PhaseIncrementalActionKey wait;
    wait.cohortId = waitCohortId;
    wait.outstandingBefore = snapshot.outstanding;
    wait.plannedOutstanding = snapshot.outstanding;
    wait.noDispatch = true;
    result.push_back(buildAction(wait));
    if (executionCount(snapshot.outstanding) >= 2)
    {
        return result;
    }

    if (snapshot.outstanding == PhaseExecutionSet::kNone)
    {
        for (PhaseIncrementalReadyAction const& candidate : ready)
        {
            PhaseIncrementalActionKey singleton;
            singleton.cohortId = candidate.candidateId;
            singleton.newcomerPhase = candidate.phase;
            singleton.direction = PhaseUnifiedActionDirection::kIdleLaunch;
            singleton.startSkew = PhaseStartSkewBucket::kImmediate;
            singleton.plannedOutstanding = phaseExecutionSetForUnifiedPhase(candidate.phase);
            PhaseIncrementalAction action
                = buildAction(singleton, candidate.dependencySafe, candidate.contextSafe, candidate.shapeSafe);
            if (action.legal())
            {
                result.push_back(action);
            }
        }
        for (size_t incumbentIndex{}; incumbentIndex < ready.size(); ++incumbentIndex)
        {
            for (size_t newcomerIndex{}; newcomerIndex < ready.size(); ++newcomerIndex)
            {
                if (incumbentIndex == newcomerIndex || ready[incumbentIndex].phase == ready[newcomerIndex].phase)
                {
                    continue;
                }
                PhaseIncrementalActionKey pair;
                pair.cohortId = pairCohortId(ready[incumbentIndex].candidateId, ready[newcomerIndex].candidateId);
                pair.incumbentPhase = ready[incumbentIndex].phase;
                pair.newcomerPhase = ready[newcomerIndex].phase;
                pair.direction = phaseUnifiedActionDirection(
                    phaseExecutionSetForUnifiedPhase(pair.incumbentPhase), pair.newcomerPhase);
                pair.startSkew = PhaseStartSkewBucket::kImmediate;
                pair.plannedOutstanding = phaseExecutionSetForUnifiedPhase(pair.incumbentPhase)
                    | phaseExecutionSetForUnifiedPhase(pair.newcomerPhase);
                PhaseIncrementalAction action
                    = buildAction(pair, ready[incumbentIndex].dependencySafe && ready[newcomerIndex].dependencySafe,
                        ready[incumbentIndex].contextSafe && ready[newcomerIndex].contextSafe,
                        ready[incumbentIndex].shapeSafe && ready[newcomerIndex].shapeSafe);
                if (action.legal())
                {
                    result.push_back(action);
                }
            }
        }
        return result;
    }

    PhaseUnifiedPhase const incumbent = onlyPhase(snapshot.outstanding);
    uint64_t incumbentId{};
    for (PhaseInFlightWorkSnapshot const& work : snapshot.work)
    {
        if (work.phase == incumbent)
        {
            incumbentId = work.actionId != 0U ? work.actionId : work.executionId;
            break;
        }
    }
    for (PhaseIncrementalReadyAction const& candidate : ready)
    {
        PhaseIncrementalActionKey augmentation;
        augmentation.cohortId = pairCohortId(incumbentId, candidate.candidateId);
        augmentation.outstandingBefore = snapshot.outstanding;
        augmentation.incumbentPhase = incumbent;
        augmentation.newcomerPhase = candidate.phase;
        augmentation.direction = phaseUnifiedActionDirection(snapshot.outstanding, candidate.phase);
        augmentation.startSkew = incumbentStartSkew;
        augmentation.plannedOutstanding = snapshot.outstanding | phaseExecutionSetForUnifiedPhase(candidate.phase);
        PhaseIncrementalAction action
            = buildAction(augmentation, candidate.dependencySafe, candidate.contextSafe, candidate.shapeSafe);
        if (action.legal())
        {
            result.push_back(action);
        }
    }
    return result;
}

PhaseIncrementalAction phaseIncrementalActionForDispatch(PhaseExecutionSet outstandingBefore, uint64_t cohortId,
    PhaseGlobalActionKind action, PhaseUnifiedActionDirection direction, PhaseStartSkewBucket startSkew) noexcept
{
    PhaseIncrementalActionKey key;
    key.cohortId = cohortId;
    key.outstandingBefore = outstandingBefore;
    key.plannedOutstanding = phaseExecutionSetForAction(action);
    key.direction = direction == PhaseUnifiedActionDirection::kNone ? canonicalDirection(action) : direction;
    key.startSkew = startSkew;
    if (action == PhaseGlobalActionKind::kWait || action == PhaseGlobalActionKind::kNone)
    {
        key.noDispatch = true;
        key.direction = PhaseUnifiedActionDirection::kNone;
        key.plannedOutstanding = outstandingBefore;
        return buildAction(key);
    }
    if (outstandingBefore == PhaseExecutionSet::kNone && executionCount(key.plannedOutstanding) == 1)
    {
        key.newcomerPhase = onlyPhase(key.plannedOutstanding);
        key.direction = PhaseUnifiedActionDirection::kIdleLaunch;
    }
    else
    {
        key.incumbentPhase = phaseUnifiedDirectionIncumbentPhase(key.direction);
        key.newcomerPhase = phaseUnifiedDirectionNewcomerPhase(key.direction);
    }
    return buildAction(key);
}

} // namespace trt_edgellm::rt
