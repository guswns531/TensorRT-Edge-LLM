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

#include "runtime/phase/policy/phaseContextualPdModel.h"
#include "runtime/phase/policy/phaseGlobalCostModel.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace trt_edgellm::rt
{

constexpr uint32_t kPHASE_UNIFIED_EVENT_SCHEMA_VERSION = 1U;

enum class PhaseUnifiedEventKind : uint8_t
{
    kDecision,
    kDispatch,
    kCompletion,
};

enum class PhaseUnifiedPhase : uint8_t
{
    kNone,
    kEncoder,
    kPrefill,
    kDecode,
    kCopy,
};

enum class PhaseUnifiedActionDirection : uint8_t
{
    kNone,
    kIdleLaunch,
    kEncoderToPrefill,
    kPrefillToEncoder,
    kEncoderToDecode,
    kDecodeToEncoder,
    kPrefillToDecode,
    kDecodeToPrefill,
};

enum class PhaseInFlightStatus : uint8_t
{
    kSubmitted,
    kRunning,
    kCompletionReady,
};

//! Distinguish an idle-boundary pair launch from adding work to a live context.
enum class PhaseUnifiedDispatchMode : uint8_t
{
    kNone,
    kSingle,
    kCoLaunch,
    kResidualAugmentation,
};

//! Machine-readable reason why a dispatch/completion was rejected for learning.
enum class PhaseUnifiedFidelityReason : uint8_t
{
    kNone,
    kMissingDecision,
    kActionIdMismatch,
    kOutstandingMismatch,
};

inline char const* phaseUnifiedEventKindName(PhaseUnifiedEventKind kind) noexcept
{
    switch (kind)
    {
    case PhaseUnifiedEventKind::kDecision: return "decision";
    case PhaseUnifiedEventKind::kDispatch: return "dispatch";
    case PhaseUnifiedEventKind::kCompletion: return "completion";
    }
    return "unknown";
}

inline char const* phaseUnifiedPhaseName(PhaseUnifiedPhase phase) noexcept
{
    switch (phase)
    {
    case PhaseUnifiedPhase::kNone: return "none";
    case PhaseUnifiedPhase::kEncoder: return "encoder";
    case PhaseUnifiedPhase::kPrefill: return "prefill";
    case PhaseUnifiedPhase::kDecode: return "decode";
    case PhaseUnifiedPhase::kCopy: return "copy";
    }
    return "unknown";
}

inline char const* phaseUnifiedActionDirectionName(PhaseUnifiedActionDirection direction) noexcept
{
    switch (direction)
    {
    case PhaseUnifiedActionDirection::kNone: return "none";
    case PhaseUnifiedActionDirection::kIdleLaunch: return "idle_launch";
    case PhaseUnifiedActionDirection::kEncoderToPrefill: return "encoder_to_prefill";
    case PhaseUnifiedActionDirection::kPrefillToEncoder: return "prefill_to_encoder";
    case PhaseUnifiedActionDirection::kEncoderToDecode: return "encoder_to_decode";
    case PhaseUnifiedActionDirection::kDecodeToEncoder: return "decode_to_encoder";
    case PhaseUnifiedActionDirection::kPrefillToDecode: return "prefill_to_decode";
    case PhaseUnifiedActionDirection::kDecodeToPrefill: return "decode_to_prefill";
    }
    return "unknown";
}

inline std::optional<PhaseUnifiedActionDirection> phaseUnifiedActionDirectionFromName(std::string_view name) noexcept
{
    if (name == "encoder_to_prefill")
    {
        return PhaseUnifiedActionDirection::kEncoderToPrefill;
    }
    if (name == "prefill_to_encoder")
    {
        return PhaseUnifiedActionDirection::kPrefillToEncoder;
    }
    if (name == "encoder_to_decode")
    {
        return PhaseUnifiedActionDirection::kEncoderToDecode;
    }
    if (name == "decode_to_encoder")
    {
        return PhaseUnifiedActionDirection::kDecodeToEncoder;
    }
    if (name == "prefill_to_decode")
    {
        return PhaseUnifiedActionDirection::kPrefillToDecode;
    }
    if (name == "decode_to_prefill")
    {
        return PhaseUnifiedActionDirection::kDecodeToPrefill;
    }
    return std::nullopt;
}

inline PhaseUnifiedPhase phaseUnifiedDirectionIncumbentPhase(PhaseUnifiedActionDirection direction) noexcept
{
    switch (direction)
    {
    case PhaseUnifiedActionDirection::kEncoderToPrefill:
    case PhaseUnifiedActionDirection::kEncoderToDecode: return PhaseUnifiedPhase::kEncoder;
    case PhaseUnifiedActionDirection::kPrefillToEncoder:
    case PhaseUnifiedActionDirection::kPrefillToDecode: return PhaseUnifiedPhase::kPrefill;
    case PhaseUnifiedActionDirection::kDecodeToEncoder:
    case PhaseUnifiedActionDirection::kDecodeToPrefill: return PhaseUnifiedPhase::kDecode;
    case PhaseUnifiedActionDirection::kNone:
    case PhaseUnifiedActionDirection::kIdleLaunch: return PhaseUnifiedPhase::kNone;
    }
    return PhaseUnifiedPhase::kNone;
}

inline PhaseUnifiedPhase phaseUnifiedDirectionNewcomerPhase(PhaseUnifiedActionDirection direction) noexcept
{
    switch (direction)
    {
    case PhaseUnifiedActionDirection::kPrefillToEncoder:
    case PhaseUnifiedActionDirection::kDecodeToEncoder: return PhaseUnifiedPhase::kEncoder;
    case PhaseUnifiedActionDirection::kEncoderToPrefill:
    case PhaseUnifiedActionDirection::kDecodeToPrefill: return PhaseUnifiedPhase::kPrefill;
    case PhaseUnifiedActionDirection::kEncoderToDecode:
    case PhaseUnifiedActionDirection::kPrefillToDecode: return PhaseUnifiedPhase::kDecode;
    case PhaseUnifiedActionDirection::kNone:
    case PhaseUnifiedActionDirection::kIdleLaunch: return PhaseUnifiedPhase::kNone;
    }
    return PhaseUnifiedPhase::kNone;
}

inline bool phaseUnifiedDirectionsSharePair(
    PhaseUnifiedActionDirection left, PhaseUnifiedActionDirection right) noexcept
{
    PhaseUnifiedPhase const leftIncumbent = phaseUnifiedDirectionIncumbentPhase(left);
    PhaseUnifiedPhase const leftNewcomer = phaseUnifiedDirectionNewcomerPhase(left);
    PhaseUnifiedPhase const rightIncumbent = phaseUnifiedDirectionIncumbentPhase(right);
    PhaseUnifiedPhase const rightNewcomer = phaseUnifiedDirectionNewcomerPhase(right);
    return leftIncumbent != PhaseUnifiedPhase::kNone && rightIncumbent != PhaseUnifiedPhase::kNone
        && ((leftIncumbent == rightIncumbent && leftNewcomer == rightNewcomer)
            || (leftIncumbent == rightNewcomer && leftNewcomer == rightIncumbent));
}

//! Research-only launch control for the M2 directional-injection benchmark.
//! Production leaves direction at kNone, so no launch order or delay changes.
struct PhaseDirectionalInjectionControl
{
    PhaseUnifiedActionDirection direction{PhaseUnifiedActionDirection::kNone};
    double targetFraction{};
    double incumbentReferenceUs{};
    double newcomerReferenceUs{};
    uint64_t requestedDelayUs{};

    bool enabled() const noexcept
    {
        return direction != PhaseUnifiedActionDirection::kNone;
    }
};

inline char const* phaseInFlightStatusName(PhaseInFlightStatus status) noexcept
{
    switch (status)
    {
    case PhaseInFlightStatus::kSubmitted: return "submitted";
    case PhaseInFlightStatus::kRunning: return "running";
    case PhaseInFlightStatus::kCompletionReady: return "completion_ready";
    }
    return "unknown";
}

inline char const* phaseUnifiedDispatchModeName(PhaseUnifiedDispatchMode mode) noexcept
{
    switch (mode)
    {
    case PhaseUnifiedDispatchMode::kNone: return "none";
    case PhaseUnifiedDispatchMode::kSingle: return "single";
    case PhaseUnifiedDispatchMode::kCoLaunch: return "co_launch";
    case PhaseUnifiedDispatchMode::kResidualAugmentation: return "residual_augmentation";
    }
    return "unknown";
}

inline char const* phaseUnifiedFidelityReasonName(PhaseUnifiedFidelityReason reason) noexcept
{
    switch (reason)
    {
    case PhaseUnifiedFidelityReason::kNone: return "none";
    case PhaseUnifiedFidelityReason::kMissingDecision: return "missing_decision";
    case PhaseUnifiedFidelityReason::kActionIdMismatch: return "action_id_mismatch";
    case PhaseUnifiedFidelityReason::kOutstandingMismatch: return "outstanding_mismatch";
    }
    return "unknown";
}

//! A multi-phase plan owns one stable plan/incremental identity while each
//! member execution retains its phase-local action ID. Exact member action-ID
//! equality is therefore meaningful only for a single-phase dispatch.
inline bool phaseUnifiedActionIdentityMatches(
    PhaseUnifiedDispatchMode mode, uint64_t selectedActionId, uint64_t memberActionId) noexcept
{
    return mode != PhaseUnifiedDispatchMode::kSingle || selectedActionId == memberActionId;
}

inline PhaseUnifiedDispatchMode phaseUnifiedDispatchMode(
    PhaseExecutionSet outstandingBefore, PhaseGlobalActionKind action) noexcept
{
    bool const pair = action == PhaseGlobalActionKind::kEncoderPrefill
        || action == PhaseGlobalActionKind::kEncoderDecode || action == PhaseGlobalActionKind::kPrefillDecode;
    if (!pair)
    {
        return action == PhaseGlobalActionKind::kNone || action == PhaseGlobalActionKind::kWait
            ? PhaseUnifiedDispatchMode::kNone
            : PhaseUnifiedDispatchMode::kSingle;
    }
    return outstandingBefore == PhaseExecutionSet::kNone ? PhaseUnifiedDispatchMode::kCoLaunch
                                                         : PhaseUnifiedDispatchMode::kResidualAugmentation;
}

inline PhaseExecutionSet phaseExecutionSetForUnifiedPhase(PhaseUnifiedPhase phase) noexcept
{
    switch (phase)
    {
    case PhaseUnifiedPhase::kEncoder: return PhaseExecutionSet::kEncoder;
    case PhaseUnifiedPhase::kPrefill: return PhaseExecutionSet::kPrefill;
    case PhaseUnifiedPhase::kDecode: return PhaseExecutionSet::kDecode;
    case PhaseUnifiedPhase::kNone:
    case PhaseUnifiedPhase::kCopy: return PhaseExecutionSet::kNone;
    }
    return PhaseExecutionSet::kNone;
}

inline PhaseUnifiedActionDirection phaseUnifiedActionDirection(
    PhaseExecutionSet outstandingBefore, PhaseUnifiedPhase added) noexcept
{
    if (outstandingBefore == PhaseExecutionSet::kNone)
    {
        return PhaseUnifiedActionDirection::kIdleLaunch;
    }
    if (added == PhaseUnifiedPhase::kEncoder)
    {
        if (phaseExecutionSetContains(outstandingBefore, PhaseExecutionSet::kPrefill))
        {
            return PhaseUnifiedActionDirection::kPrefillToEncoder;
        }
        if (phaseExecutionSetContains(outstandingBefore, PhaseExecutionSet::kDecode))
        {
            return PhaseUnifiedActionDirection::kDecodeToEncoder;
        }
    }
    if (added == PhaseUnifiedPhase::kPrefill)
    {
        if (phaseExecutionSetContains(outstandingBefore, PhaseExecutionSet::kEncoder))
        {
            return PhaseUnifiedActionDirection::kEncoderToPrefill;
        }
        if (phaseExecutionSetContains(outstandingBefore, PhaseExecutionSet::kDecode))
        {
            return PhaseUnifiedActionDirection::kDecodeToPrefill;
        }
    }
    if (added == PhaseUnifiedPhase::kDecode)
    {
        if (phaseExecutionSetContains(outstandingBefore, PhaseExecutionSet::kEncoder))
        {
            return PhaseUnifiedActionDirection::kEncoderToDecode;
        }
        if (phaseExecutionSetContains(outstandingBefore, PhaseExecutionSet::kPrefill))
        {
            return PhaseUnifiedActionDirection::kPrefillToDecode;
        }
    }
    return PhaseUnifiedActionDirection::kNone;
}

struct PhaseUnifiedWork
{
    int32_t encoderRows{};
    int32_t prefillRows{};
    int32_t prefillTokens{};
    int32_t decodeRows{};
    int64_t decodeContextTokens{};
};

struct PhaseInFlightWorkSnapshot
{
    PhaseUnifiedPhase phase{PhaseUnifiedPhase::kNone};
    PhaseInFlightStatus status{PhaseInFlightStatus::kSubmitted};
    uint64_t executionId{};
    uint64_t activityCorrelationId{};
    uint64_t planId{};
    uint64_t actionId{};
    uint64_t dispatchHostNs{};
    uint64_t prepareStartHostNs{};
    uint64_t prepareEndHostNs{};
    uint64_t executeStartHostNs{};
    uint64_t executeEndHostNs{};
    bool graphReplay{};
    double dispatchAgeUs{};
    std::vector<uint64_t> requestIds;
    PhaseUnifiedWork work;
};

struct PhaseInFlightSnapshot
{
    uint64_t hostSnapshotNs{};
    PhaseExecutionSet outstanding{PhaseExecutionSet::kNone};
    std::vector<PhaseInFlightWorkSnapshot> work;
};

struct PhaseUnifiedCandidateSnapshot
{
    uint64_t actionId{};
    PhaseGlobalActionKey key;
    bool legal{true};
    std::vector<uint64_t> requestIds;
    double predictedCompletionUs{};
    double uncertaintyUs{};
    double predictedSloViolationUs{};
    //! Pre-update contextual completion evidence for this exact decision
    //! frontier.  Keeping every legal candidate here lets offline replay join
    //! cross-run measured labels without reconstructing process-local model
    //! state or treating the selected action as the whole frontier.
    bool contextualCompletionValid{};
    bool contextualCompletionFeatureV2Valid{};
    PhaseContextualPdFeatures contextualCompletionFeatures{};
    PhaseContextualPairDirection contextualDirection{PhaseContextualPairDirection::kPrefillToDecode};
    PhaseContextualCompletionEstimate contextualCompletion;
    double contextualIncumbentReferenceUs{};
    double contextualNewcomerReferenceUs{};
    double contextualMinimumSlackUs{std::numeric_limits<double>::infinity()};
    bool completionPolicyEvaluated{};
    bool completionAuthorityReady{};
    bool completionAuthorityApplied{};
    bool scalarDecisionCostKnown{};
    bool activeDecisionCostKnown{};
    double scalarDecisionMakespanUs{};
    double activeDecisionMakespanUs{};
    double completionAggregateBlendWeight{};
    double completionIncumbentBlendWeight{};
    double completionNewcomerBlendWeight{};
    std::vector<PhaseProtectedCompletion> scalarProtectedCompletions;
    std::vector<PhaseProtectedCompletion> activeProtectedCompletions;
};

struct PhaseUnifiedEvent
{
    uint32_t schemaVersion{kPHASE_UNIFIED_EVENT_SCHEMA_VERSION};
    PhaseUnifiedEventKind kind{PhaseUnifiedEventKind::kDecision};
    uint64_t eventId{};
    uint64_t hostMonotonicNs{};
    uint64_t decisionId{};
    uint64_t snapshotId{};
    uint64_t executionId{};
    uint64_t planId{};
    uint64_t actionId{};
    //! Stable M3 identity over cohort, outstanding transition, direction, and skew.
    uint64_t incrementalActionId{};
    //! Requested/estimated incumbent offset bucket. Negative means unknown.
    int32_t requestedStartSkewPercent{-1};
    PhaseUnifiedPhase phase{PhaseUnifiedPhase::kNone};
    PhaseGlobalActionKind actionKind{PhaseGlobalActionKind::kNone};
    PhaseUnifiedActionDirection requestedDirection{PhaseUnifiedActionDirection::kNone};
    PhaseUnifiedActionDirection direction{PhaseUnifiedActionDirection::kNone};
    PhaseUnifiedDispatchMode dispatchMode{PhaseUnifiedDispatchMode::kNone};
    PhaseUnifiedPhase incumbentPhase{PhaseUnifiedPhase::kNone};
    uint64_t incumbentExecutionId{};
    double incumbentDispatchAgeUs{};
    PhaseUnifiedPhase newcomerPhase{PhaseUnifiedPhase::kNone};
    uint64_t newcomerExecutionId{};
    int32_t observedStartSkewPercent{-1};
    PhaseExecutionSet outstandingBefore{PhaseExecutionSet::kNone};
    PhaseExecutionSet plannedOutstanding{PhaseExecutionSet::kNone};
    PhaseExecutionSet observedOutstanding{PhaseExecutionSet::kNone};
    PhaseUnifiedWork ready;
    //! Canonical ready lineages. Transient timestamps and process-local
    //! execution IDs are deliberately excluded from the replay signature.
    std::vector<uint64_t> readyEncoderRequestIds;
    std::vector<uint64_t> readyPrefillRequestIds;
    //! Per-request remaining prompt progress parallel to readyPrefillRequestIds.
    std::vector<int32_t> readyPrefillTokenCounts;
    std::vector<uint64_t> readyDecodeRequestIds;
    //! Per-request KV progress parallel to readyDecodeRequestIds.
    std::vector<int32_t> readyDecodeContextLengths;
    int32_t pagePoolAllocatedBundles{};
    int32_t pageReservationGuaranteedBundles{};
    size_t visionPayloadBytes{};
    uint64_t snapshotSignature{};
    PhaseUnifiedWork cohort;
    std::vector<uint64_t> requestIds;
    PhaseInFlightSnapshot inFlight;
    std::vector<PhaseUnifiedCandidateSnapshot> candidates;
    uint64_t selectedActionId{};
    //! H=1 selector result with completion authority as evaluated online.
    uint64_t activeH1SelectedActionId{};
    //! H=1 selector result after restoring completion-sensitive candidate
    //! fields to their scalar values over this exact frontier.
    uint64_t scalarSelectedActionId{};
    uint64_t enqueueHostNs{};
    uint64_t prepareStartHostNs{};
    uint64_t prepareEndHostNs{};
    uint64_t executeStartHostNs{};
    uint64_t executeEndHostNs{};
    bool graphReplay{};
    std::optional<double> injectionTargetFraction;
    PhaseUnifiedActionDirection injectionRequestedDirection{PhaseUnifiedActionDirection::kNone};
    double injectionIncumbentReferenceUs{};
    double injectionNewcomerReferenceUs{};
    uint64_t requestedInjectionDelayUs{};
    std::optional<double> gpuStartUs;
    std::optional<double> gpuEndUs;
    std::optional<double> incumbentGpuCompletionUs;
    std::optional<double> newcomerGpuCompletionUs;
    double gpuDurationUs{};
    uint64_t completionVisibleHostNs{};
    bool actionFidelity{true};
    PhaseUnifiedFidelityReason actionFidelityReason{PhaseUnifiedFidelityReason::kNone};
};

//! Cross-run stable signature of an immutable observable decision snapshot.
//! Queue order is meaningful and retained. In-flight phase records are sorted
//! locally so P/D/E polling order cannot change replay identity.
inline uint64_t phaseUnifiedSnapshotSignature(PhaseUnifiedEvent const& event)
{
    constexpr uint64_t kFNV_OFFSET = 14695981039346656037ULL;
    constexpr uint64_t kFNV_PRIME = 1099511628211ULL;
    uint64_t result{kFNV_OFFSET};
    auto add = [&](uint64_t value) {
        result ^= value;
        result *= kFNV_PRIME;
    };
    auto addIds = [&](std::vector<uint64_t> const& ids) {
        add(ids.size());
        for (uint64_t const requestId : ids)
        {
            add(requestId);
        }
    };
    auto addCounts = [&](std::vector<int32_t> const& counts) {
        add(counts.size());
        for (int32_t const count : counts)
        {
            add(static_cast<uint64_t>(count));
        }
    };
    add(static_cast<uint8_t>(event.outstandingBefore));
    add(static_cast<uint64_t>(event.ready.encoderRows));
    add(static_cast<uint64_t>(event.ready.prefillRows));
    add(static_cast<uint64_t>(event.ready.prefillTokens));
    add(static_cast<uint64_t>(event.ready.decodeRows));
    add(static_cast<uint64_t>(event.ready.decodeContextTokens));
    addIds(event.readyEncoderRequestIds);
    addIds(event.readyPrefillRequestIds);
    addCounts(event.readyPrefillTokenCounts);
    addIds(event.readyDecodeRequestIds);
    addCounts(event.readyDecodeContextLengths);
    add(static_cast<uint64_t>(event.pagePoolAllocatedBundles));
    add(static_cast<uint64_t>(event.pageReservationGuaranteedBundles));
    add(event.visionPayloadBytes);

    std::vector<PhaseInFlightWorkSnapshot const*> inFlight;
    inFlight.reserve(event.inFlight.work.size());
    for (PhaseInFlightWorkSnapshot const& work : event.inFlight.work)
    {
        inFlight.push_back(&work);
    }
    std::sort(inFlight.begin(), inFlight.end(), [](auto const* left, auto const* right) {
        return static_cast<uint8_t>(left->phase) < static_cast<uint8_t>(right->phase);
    });
    add(inFlight.size());
    for (PhaseInFlightWorkSnapshot const* work : inFlight)
    {
        add(static_cast<uint8_t>(work->phase));
        add(static_cast<uint8_t>(work->status));
        add(static_cast<uint64_t>(work->work.encoderRows));
        add(static_cast<uint64_t>(work->work.prefillRows));
        add(static_cast<uint64_t>(work->work.prefillTokens));
        add(static_cast<uint64_t>(work->work.decodeRows));
        add(static_cast<uint64_t>(work->work.decodeContextTokens));
        addIds(work->requestIds);
    }
    return result;
}

} // namespace trt_edgellm::rt
