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

#include "runtime/scheduling/phaseGlobalScheduler.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>

namespace trt_edgellm::rt
{
namespace
{

bool isOverlap(PhaseGlobalActionKind kind) noexcept
{
    return kind == PhaseGlobalActionKind::kEncoderPrefill || kind == PhaseGlobalActionKind::kEncoderDecode
        || kind == PhaseGlobalActionKind::kPrefillDecode;
}

uint64_t hashCombine(uint64_t seed, uint64_t value) noexcept
{
    constexpr uint64_t kHASH_OFFSET = 0x9e3779b97f4a7c15ULL;
    seed ^= value + kHASH_OFFSET + (seed << 6U) + (seed >> 2U);
    return seed;
}

size_t saturatedAdd(size_t left, size_t right) noexcept
{
    return right > std::numeric_limits<size_t>::max() - left ? std::numeric_limits<size_t>::max() : left + right;
}

size_t saturatedSubtract(size_t value, size_t reduction) noexcept
{
    return reduction >= value ? 0U : value - reduction;
}

size_t hardPeakManagedBytes(PhaseActionMemoryHorizon const& memory) noexcept
{
    size_t committed = memory.managedBytes;
    if (memory.immediateReclaimObserved)
    {
        committed = saturatedSubtract(committed, memory.immediateReclaimBytes);
    }
    return saturatedAdd(saturatedAdd(committed, memory.allocateBytes), memory.guaranteedGrowthBytes);
}

double robustCompletionUs(PhaseGlobalActionCandidate const& candidate) noexcept
{
    return std::max(0.0, candidate.predictedBlockingUs) + std::max(0.0, candidate.uncertaintyUs);
}

double actionMakespanUs(PhaseGlobalActionCandidate const& candidate) noexcept
{
    double const makespan
        = candidate.predictedMakespanUs > 0.0 ? candidate.predictedMakespanUs : candidate.predictedBlockingUs;
    return std::max(0.0, makespan);
}

double selectionHorizonUs(PhaseGlobalActionCandidate const& candidate) noexcept
{
    return candidate.predictedHorizonUs > 0.0 ? candidate.predictedHorizonUs : actionMakespanUs(candidate);
}

double selectionReferenceWorkUs(PhaseGlobalActionCandidate const& candidate) noexcept
{
    return candidate.horizonReferenceWorkUs > 0.0 ? candidate.horizonReferenceWorkUs : candidate.referenceWorkUs;
}

double predictedViolationUs(PhaseGlobalActionCandidate const& candidate, double deadlineGuardUs) noexcept
{
    double violation{};
    for (PhaseProtectedCompletion const& protectedCompletion : candidate.protectedCompletions)
    {
        if (!std::isfinite(protectedCompletion.slackUs))
        {
            continue;
        }
        double const robustCompletion = std::max(0.0, protectedCompletion.predictedCompletionUs)
            + std::max(0.0, protectedCompletion.uncertaintyUs);
        violation = std::max(violation, robustCompletion + deadlineGuardUs - protectedCompletion.slackUs);
    }
    if (!candidate.protectedCompletions.empty())
    {
        return std::max(0.0, violation);
    }
    if (!std::isfinite(candidate.minimumProtectedSlackUs))
    {
        return 0.0;
    }
    return std::max(0.0, robustCompletionUs(candidate) + deadlineGuardUs - candidate.minimumProtectedSlackUs);
}

double serviceCompression(PhaseGlobalActionCandidate const& candidate) noexcept
{
    double const makespan = std::max(selectionHorizonUs(candidate), std::numeric_limits<double>::epsilon());
    return std::max(0.0, selectionReferenceWorkUs(candidate)) / makespan;
}

bool hardFeasible(PhaseGlobalActionCandidate const& candidate) noexcept
{
    if (!candidate.dependencySafe || !candidate.contextSafe || !candidate.shapeSafe)
    {
        return false;
    }
    if (isOverlap(candidate.key.kind) && !candidate.overlapCostKnown && !candidate.safeProbeEligible)
    {
        return false;
    }
    if (candidate.key.kind == PhaseGlobalActionKind::kWait && !candidate.concreteWaitEvent)
    {
        return false;
    }
    size_t const peak = hardPeakManagedBytes(candidate.memory);
    return candidate.memory.budgetBytes == 0U || peak <= candidate.memory.budgetBytes;
}

bool dominates(
    PhaseGlobalActionCandidate const& left, PhaseGlobalActionCandidate const& right, double deadlineGuardUs) noexcept
{
    double const leftViolation = predictedViolationUs(left, deadlineGuardUs);
    double const rightViolation = predictedViolationUs(right, deadlineGuardUs);
    size_t const leftPeak = hardPeakManagedBytes(left.memory);
    size_t const rightPeak = hardPeakManagedBytes(right.memory);
    bool const noWorse = leftViolation <= rightViolation && selectionHorizonUs(left) <= selectionHorizonUs(right)
        && selectionReferenceWorkUs(left) >= selectionReferenceWorkUs(right) && leftPeak <= rightPeak
        && left.uncertaintyUs <= right.uncertaintyUs;
    bool const strictlyBetter = leftViolation < rightViolation || selectionHorizonUs(left) < selectionHorizonUs(right)
        || selectionReferenceWorkUs(left) > selectionReferenceWorkUs(right) || leftPeak < rightPeak
        || left.uncertaintyUs < right.uncertaintyUs;
    return noWorse && strictlyBetter;
}

float percentile(std::vector<float> values, float fraction)
{
    ELLM_CHECK(!values.empty(), "A percentile requires at least one sample");
    std::sort(values.begin(), values.end());
    size_t const index = static_cast<size_t>(std::ceil(fraction * static_cast<float>(values.size()))) - 1U;
    return values[std::min(index, values.size() - 1U)];
}

} // namespace

PhaseExecutionSet operator|(PhaseExecutionSet left, PhaseExecutionSet right) noexcept
{
    return static_cast<PhaseExecutionSet>(static_cast<uint8_t>(left) | static_cast<uint8_t>(right));
}

PhaseExecutionSet operator&(PhaseExecutionSet left, PhaseExecutionSet right) noexcept
{
    return static_cast<PhaseExecutionSet>(static_cast<uint8_t>(left) & static_cast<uint8_t>(right));
}

bool phaseExecutionSetContains(PhaseExecutionSet set, PhaseExecutionSet phase) noexcept
{
    return (set & phase) == phase;
}

bool phaseExecutionSetIsSubset(PhaseExecutionSet subset, PhaseExecutionSet superset) noexcept
{
    return (subset & superset) == subset;
}

PhaseExecutionSet phaseExecutionSetForAction(PhaseGlobalActionKind kind) noexcept
{
    PhaseExecutionSet result{PhaseExecutionSet::kNone};
    switch (kind)
    {
    case PhaseGlobalActionKind::kEncoder: result = PhaseExecutionSet::kEncoder; break;
    case PhaseGlobalActionKind::kPrefill: result = PhaseExecutionSet::kPrefill; break;
    case PhaseGlobalActionKind::kDecode: result = PhaseExecutionSet::kDecode; break;
    case PhaseGlobalActionKind::kEncoderPrefill:
        result = PhaseExecutionSet::kEncoder | PhaseExecutionSet::kPrefill;
        break;
    case PhaseGlobalActionKind::kEncoderDecode:
        result = PhaseExecutionSet::kEncoder | PhaseExecutionSet::kDecode;
        break;
    case PhaseGlobalActionKind::kPrefillDecode:
        result = PhaseExecutionSet::kPrefill | PhaseExecutionSet::kDecode;
        break;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kWait: result = PhaseExecutionSet::kNone; break;
    }
    return result;
}

PhaseExecutionVariant phaseExecutionVariant(bool primaryGraph, bool secondaryGraph) noexcept
{
    uint8_t const mask = static_cast<uint8_t>(primaryGraph ? PhaseExecutionVariant::kPrimaryGraph
                                                           : PhaseExecutionVariant::kEager)
        | static_cast<uint8_t>(secondaryGraph ? PhaseExecutionVariant::kSecondaryGraph
                                              : PhaseExecutionVariant::kEager);
    return static_cast<PhaseExecutionVariant>(mask);
}

bool phaseExecutionVariantUsesPrimaryGraph(PhaseExecutionVariant variant) noexcept
{
    return (static_cast<uint8_t>(variant) & static_cast<uint8_t>(PhaseExecutionVariant::kPrimaryGraph)) != 0U;
}

bool phaseExecutionVariantUsesSecondaryGraph(PhaseExecutionVariant variant) noexcept
{
    return (static_cast<uint8_t>(variant) & static_cast<uint8_t>(PhaseExecutionVariant::kSecondaryGraph)) != 0U;
}

char const* phaseExecutionVariantName(PhaseExecutionVariant variant) noexcept
{
    char const* result = "unknown";
    switch (variant)
    {
    case PhaseExecutionVariant::kEager: result = "eager"; break;
    case PhaseExecutionVariant::kPrimaryGraph: result = "primary_graph"; break;
    case PhaseExecutionVariant::kSecondaryGraph: result = "secondary_graph"; break;
    case PhaseExecutionVariant::kBothGraph: result = "both_graph"; break;
    }
    return result;
}

uint64_t phaseGlobalCandidateId(PhaseGlobalActionCandidate const& candidate) noexcept
{
    uint64_t result = static_cast<uint64_t>(candidate.key.kind);
    result = hashCombine(result, static_cast<uint64_t>(candidate.key.primaryBatchSize));
    result = hashCombine(result, static_cast<uint64_t>(candidate.key.secondaryBatchSize));
    result = hashCombine(result, static_cast<uint64_t>(candidate.key.chunkLength));
    result = hashCombine(result, static_cast<uint64_t>(candidate.key.primaryContextBucket));
    result = hashCombine(result, static_cast<uint64_t>(candidate.key.secondaryContextBucket));
    result = hashCombine(result, static_cast<uint64_t>(candidate.key.executionVariant));
    for (uint64_t const requestId : candidate.primaryRequestIds)
    {
        result = hashCombine(result, requestId);
    }
    result = hashCombine(result, candidate.primaryRequestIds.size());
    for (int32_t const stableSlotId : candidate.primaryStableSlotIds)
    {
        result = hashCombine(result, static_cast<uint64_t>(stableSlotId));
    }
    result = hashCombine(result, candidate.primaryStableSlotIds.size());
    for (uint64_t const requestId : candidate.secondaryRequestIds)
    {
        result = hashCombine(result, requestId);
    }
    result = hashCombine(result, candidate.secondaryRequestIds.size());
    for (int32_t const stableSlotId : candidate.secondaryStableSlotIds)
    {
        result = hashCombine(result, static_cast<uint64_t>(stableSlotId));
    }
    result = hashCombine(result, candidate.secondaryStableSlotIds.size());
    result = hashCombine(result, candidate.waitEventId);
    return result;
}

void phaseGlobalFinalizeCandidate(PhaseGlobalActionCandidate& candidate)
{
    ELLM_CHECK(candidate.primaryStableSlotIds.empty()
            || candidate.primaryStableSlotIds.size() == candidate.primaryRequestIds.size(),
        "Global primary stable-slot rows do not match request rows");
    ELLM_CHECK(candidate.secondaryStableSlotIds.empty()
            || candidate.secondaryStableSlotIds.size() == candidate.secondaryRequestIds.size(),
        "Global secondary stable-slot rows do not match request rows");
    candidate.requestIds = candidate.primaryRequestIds;
    candidate.requestIds.insert(
        candidate.requestIds.end(), candidate.secondaryRequestIds.begin(), candidate.secondaryRequestIds.end());
    candidate.candidateId = phaseGlobalCandidateId(candidate);
}

bool PhaseGlobalDispatchPlan::permits(PhaseExecutionSet phases) const noexcept
{
    return phaseExecutionSetIsSubset(phases, allowedOutstanding);
}

bool PhaseGlobalDispatchPlan::launchMatches(PhaseExecutionSet phases) const noexcept
{
    return phases == launched && permits(phases);
}

PhaseGlobalDispatchPlan phaseGlobalDispatchPlan(
    uint64_t planId, uint64_t snapshotEpoch, PhaseGlobalActionCandidate const& candidate)
{
    PhaseGlobalDispatchPlan result;
    result.planId = planId;
    result.snapshotEpoch = snapshotEpoch;
    result.candidateId = candidate.candidateId != 0U ? candidate.candidateId : phaseGlobalCandidateId(candidate);
    result.action = candidate.key.kind;
    result.allowedOutstanding = phaseExecutionSetForAction(candidate.key.kind);
    result.launched = result.allowedOutstanding;
    result.waitEventId = candidate.waitEventId;
    result.primaryRequestIds = candidate.primaryRequestIds;
    result.secondaryRequestIds = candidate.secondaryRequestIds;
    result.primaryStableSlotIds = candidate.primaryStableSlotIds;
    result.secondaryStableSlotIds = candidate.secondaryStableSlotIds;
    return result;
}

char const* phaseGlobalActionKindName(PhaseGlobalActionKind kind) noexcept
{
    char const* result = "unknown";
    switch (kind)
    {
    case PhaseGlobalActionKind::kNone: result = "none"; break;
    case PhaseGlobalActionKind::kEncoder: result = "encoder"; break;
    case PhaseGlobalActionKind::kPrefill: result = "prefill"; break;
    case PhaseGlobalActionKind::kDecode: result = "decode"; break;
    case PhaseGlobalActionKind::kEncoderPrefill: result = "encoder_prefill"; break;
    case PhaseGlobalActionKind::kEncoderDecode: result = "encoder_decode"; break;
    case PhaseGlobalActionKind::kPrefillDecode: result = "prefill_decode"; break;
    case PhaseGlobalActionKind::kWait: result = "wait"; break;
    }
    return result;
}

bool PhaseGlobalActionKey::operator==(PhaseGlobalActionKey const& other) const noexcept
{
    return std::tie(kind, primaryBatchSize, secondaryBatchSize, chunkLength, primaryContextBucket,
               secondaryContextBucket, executionVariant)
        == std::tie(other.kind, other.primaryBatchSize, other.secondaryBatchSize, other.chunkLength,
            other.primaryContextBucket, other.secondaryContextBucket, other.executionVariant);
}

size_t PhaseGlobalCostModel::KeyHash::operator()(PhaseGlobalActionKey const& key) const noexcept
{
    size_t result = static_cast<size_t>(key.kind);
    auto combine = [&result](int32_t value) {
        size_t const hashed = std::hash<int32_t>{}(value);
        result ^= hashed + 0x9e3779b9U + (result << 6U) + (result >> 2U);
    };
    combine(key.primaryBatchSize);
    combine(key.secondaryBatchSize);
    combine(key.chunkLength);
    combine(key.primaryContextBucket);
    combine(key.secondaryContextBucket);
    combine(static_cast<int32_t>(key.executionVariant));
    return result;
}

PhaseGlobalCostModel::PhaseGlobalCostModel(PhaseGlobalCostModelConfig config)
    : mConfig(config)
{
    ELLM_CHECK(mConfig.windowSize > 0U, "Global phase cost window must be positive");
    ELLM_CHECK(mConfig.overlapMinSamples > 0U, "Global phase overlap sample threshold must be positive");
    ELLM_CHECK(mConfig.coldStartUncertaintyMs >= 0.0F, "Global phase cold-start uncertainty must be non-negative");
    ELLM_CHECK(mConfig.minimumOverlapGainRatio >= 0.0F, "Global phase overlap gain must be non-negative");
}

void PhaseGlobalCostModel::observe(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation observation)
{
    ELLM_CHECK(observation.referenceWorkMs > 0.0F, "Global phase reference work must be positive");
    ELLM_CHECK(observation.makespanMs > 0.0F, "Global phase makespan must be positive");
    Samples& samples = mSamples[key];
    samples.values.push_back(observation);
    while (samples.values.size() > mConfig.windowSize)
    {
        samples.values.pop_front();
    }
}

std::optional<PhaseGlobalCostEstimate> PhaseGlobalCostModel::estimate(PhaseGlobalActionKey const& key) const
{
    auto const found = mSamples.find(key);
    if (found == mSamples.end() || found->second.values.empty())
    {
        return std::nullopt;
    }
    std::vector<float> reference;
    std::vector<float> makespan;
    reference.reserve(found->second.values.size());
    makespan.reserve(found->second.values.size());
    for (PhaseGlobalCostObservation const& value : found->second.values)
    {
        reference.push_back(value.referenceWorkMs);
        makespan.push_back(value.makespanMs);
    }
    float const median = percentile(makespan, 0.5F);
    float const p95 = percentile(makespan, 0.95F);
    float const sampleMargin
        = mConfig.coldStartUncertaintyMs / std::sqrt(static_cast<float>(found->second.values.size()));
    return PhaseGlobalCostEstimate{
        found->second.values.size(), percentile(reference, 0.5F), median, p95, std::max(p95 - median, sampleMargin)};
}

bool PhaseGlobalCostModel::overlapEligible(PhaseGlobalActionKey const& key) const
{
    if (!isOverlap(key.kind))
    {
        return true;
    }
    std::optional<PhaseGlobalCostEstimate> const cost = estimate(key);
    if (!cost.has_value() || cost->sampleCount < mConfig.overlapMinSamples)
    {
        return false;
    }
    float const robustMakespan = cost->makespanMedianMs + cost->uncertaintyMs;
    float const requiredReference = robustMakespan * (1.0F + mConfig.minimumOverlapGainRatio);
    return cost->referenceWorkMedianMs >= requiredReference;
}

void PhaseGlobalCostModel::reset()
{
    mSamples.clear();
}

PhaseGlobalScheduler::PhaseGlobalScheduler(PhaseGlobalSchedulerConfig config)
    : mConfig(config)
{
    ELLM_CHECK(mConfig.maxCandidates > 0U, "Global phase candidate limit must be positive");
    ELLM_CHECK(mConfig.deadlineGuardUs >= 0.0, "Global phase deadline guard must be non-negative");
}

PhaseGlobalDecision PhaseGlobalScheduler::select(std::vector<PhaseGlobalActionCandidate> const& candidates) const
{
    ELLM_CHECK(candidates.size() <= mConfig.maxCandidates, "Global phase candidate limit exceeded");
    PhaseGlobalDecision decision;
    decision.inputCandidates = candidates.size();
    if (candidates.empty())
    {
        return decision;
    }

    std::vector<size_t> feasible;
    for (size_t index{}; index < candidates.size(); ++index)
    {
        if (hardFeasible(candidates[index]))
        {
            feasible.push_back(index);
        }
    }
    decision.hardFeasibleCandidates = feasible.size();
    if (feasible.empty())
    {
        decision.reason = PhaseGlobalDecisionReason::kNoHardFeasibleCandidate;
        return decision;
    }

    std::vector<size_t> safe;
    for (size_t const index : feasible)
    {
        if (predictedViolationUs(candidates[index], mConfig.deadlineGuardUs) == 0.0)
        {
            safe.push_back(index);
        }
    }
    decision.deadlineSafeCandidates = safe.size();
    std::vector<size_t> frontier = safe.empty() ? feasible : safe;
    std::vector<size_t> pruned;
    for (size_t const right : frontier)
    {
        bool dominated{};
        for (size_t const left : frontier)
        {
            if (left != right && dominates(candidates[left], candidates[right], mConfig.deadlineGuardUs))
            {
                dominated = true;
                break;
            }
        }
        if (dominated)
        {
            ++decision.dominatedCandidates;
        }
        else
        {
            pruned.push_back(right);
        }
    }
    ELLM_CHECK(!pruned.empty(), "Global phase dominance pruning removed every candidate");

    auto betterSafe = [&](size_t left, size_t right) {
        PhaseGlobalActionCandidate const& lhs = candidates[left];
        PhaseGlobalActionCandidate const& rhs = candidates[right];
        double const lhsCompression = serviceCompression(lhs);
        double const rhsCompression = serviceCompression(rhs);
        size_t const lhsReclaim = saturatedAdd(
            lhs.memory.immediateReclaimObserved ? lhs.memory.immediateReclaimBytes : 0U, lhs.memory.nearReclaimBytes);
        size_t const rhsReclaim = saturatedAdd(
            rhs.memory.immediateReclaimObserved ? rhs.memory.immediateReclaimBytes : 0U, rhs.memory.nearReclaimBytes);
        auto const lhsRank = std::tie(lhsCompression, lhsReclaim, lhs.requestServiceLagUs);
        auto const rhsRank = std::tie(rhsCompression, rhsReclaim, rhs.requestServiceLagUs);
        if (lhsRank != rhsRank)
        {
            return lhsRank > rhsRank;
        }
        uint64_t const lhsId = lhs.candidateId != 0U ? lhs.candidateId : phaseGlobalCandidateId(lhs);
        uint64_t const rhsId = rhs.candidateId != 0U ? rhs.candidateId : phaseGlobalCandidateId(rhs);
        return lhsId < rhsId;
    };
    auto betterViolation = [&](size_t left, size_t right) {
        PhaseGlobalActionCandidate const& lhs = candidates[left];
        PhaseGlobalActionCandidate const& rhs = candidates[right];
        double const lhsViolation = predictedViolationUs(lhs, mConfig.deadlineGuardUs);
        double const rhsViolation = predictedViolationUs(rhs, mConfig.deadlineGuardUs);
        if (lhsViolation != rhsViolation)
        {
            return lhsViolation < rhsViolation;
        }
        if (lhs.requestServiceLagUs != rhs.requestServiceLagUs)
        {
            return lhs.requestServiceLagUs > rhs.requestServiceLagUs;
        }
        double const lhsCompression = serviceCompression(lhs);
        double const rhsCompression = serviceCompression(rhs);
        if (lhsCompression != rhsCompression)
        {
            return lhsCompression > rhsCompression;
        }
        uint64_t const lhsId = lhs.candidateId != 0U ? lhs.candidateId : phaseGlobalCandidateId(lhs);
        uint64_t const rhsId = rhs.candidateId != 0U ? rhs.candidateId : phaseGlobalCandidateId(rhs);
        return lhsId < rhsId;
    };

    size_t selected = pruned.front();
    for (size_t const index : pruned)
    {
        if ((safe.empty() && betterViolation(index, selected)) || (!safe.empty() && betterSafe(index, selected)))
        {
            selected = index;
        }
    }
    PhaseGlobalActionCandidate const& candidate = candidates[selected];
    decision.selectedIndex = selected;
    decision.reason = safe.empty() ? PhaseGlobalDecisionReason::kMinimumViolation
                                   : PhaseGlobalDecisionReason::kDeadlineSafeEfficiency;
    decision.predictedViolationUs = predictedViolationUs(candidate, mConfig.deadlineGuardUs);
    decision.serviceCompression = serviceCompression(candidate);
    decision.hardPeakManagedBytes = hardPeakManagedBytes(candidate.memory);
    return decision;
}

} // namespace trt_edgellm::rt
