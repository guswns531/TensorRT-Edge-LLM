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

#include "runtime/phase/policy/phaseFormationPlanner.h"

#include <algorithm>
#include <cmath>
#include <limits>

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

bool contains(PhaseFormationWork const& available, PhaseFormationWork const& requested) noexcept
{
    return available.encoderRows >= requested.encoderRows && available.prefillRows >= requested.prefillRows
        && available.decodeRows >= requested.decodeRows;
}

PhaseFormationWork add(PhaseFormationWork left, PhaseFormationWork const& right) noexcept
{
    left.encoderRows += right.encoderRows;
    left.prefillRows += right.prefillRows;
    left.decodeRows += right.decodeRows;
    return left;
}

PhaseFormationWork subtract(PhaseFormationWork left, PhaseFormationWork const& right) noexcept
{
    left.encoderRows -= right.encoderRows;
    left.prefillRows -= right.prefillRows;
    left.decodeRows -= right.decodeRows;
    return left;
}

bool covers(PhaseFormationWork const& completed, PhaseFormationWork const& target) noexcept
{
    return contains(completed, target);
}

bool fitsRemaining(
    PhaseFormationWork const& completed, PhaseFormationWork const& consumes, PhaseFormationWork const& target) noexcept
{
    return completed.encoderRows + consumes.encoderRows <= target.encoderRows
        && completed.prefillRows + consumes.prefillRows <= target.prefillRows
        && completed.decodeRows + consumes.decodeRows <= target.decodeRows;
}

double decodeViolationUs(double completionUs, double uncertaintyUs, double budgetUs) noexcept
{
    if (!std::isfinite(completionUs) || !std::isfinite(budgetUs))
    {
        return 0.0;
    }
    return std::max(0.0, completionUs + std::max(0.0, uncertaintyUs) - budgetUs);
}

void updateDecodeService(PhaseFormationSequence& sequence, double completionUs, double uncertaintyUs) noexcept
{
    if (!std::isfinite(completionUs) || completionUs < 0.0)
    {
        return;
    }
    if (!std::isfinite(sequence.decodeServiceUs)
        || completionUs + uncertaintyUs < sequence.decodeServiceUs + sequence.decodeServiceUncertaintyUs)
    {
        sequence.decodeServiceUs = completionUs;
        sequence.decodeServiceUncertaintyUs = std::max(0.0, uncertaintyUs);
    }
}

std::optional<PhaseProtectedCompletion> protectedDecode(PhaseGlobalActionCandidate const& candidate) noexcept
{
    std::optional<PhaseProtectedCompletion> result;
    for (PhaseProtectedCompletion const& completion : candidate.protectedCompletions)
    {
        if (completion.kind != PhaseProtectedKind::kDecode)
        {
            continue;
        }
        double const robust = std::max(0.0, completion.predictedCompletionUs) + std::max(0.0, completion.uncertaintyUs);
        double const currentRobust = result.has_value()
            ? std::max(0.0, result->predictedCompletionUs) + std::max(0.0, result->uncertaintyUs)
            : std::numeric_limits<double>::infinity();
        if (robust < currentRobust)
        {
            result = completion;
        }
    }
    return result;
}

double protectedViolationUs(PhaseGlobalActionCandidate const& candidate) noexcept
{
    double result{};
    for (PhaseProtectedCompletion const& completion : candidate.protectedCompletions)
    {
        if (!std::isfinite(completion.slackUs))
        {
            continue;
        }
        double const robust = std::max(0.0, completion.predictedCompletionUs) + std::max(0.0, completion.uncertaintyUs);
        result = std::max(result, robust - completion.slackUs);
    }
    return std::max(0.0, result);
}

double appendConcreteCompletionBoundary(
    PhaseFormationSequence& sequence, PhaseFormationSnapshot const& snapshot) noexcept
{
    // The first preview is the earliest cumulative decode cohort. Following
    // previews contain that same cohort plus later events, so appending more
    // than one would double-count work. This is a causal boundary extension,
    // not an unconstrained H=3 search.
    auto const completion = std::find_if(snapshot.knownCompletions.begin(), snapshot.knownCompletions.end(),
        [](PhaseFormationKnownCompletion const& value) {
            return value.work.decodeRows > 0U && value.serviceMakespanUs > 0.0;
        });
    if (completion == snapshot.knownCompletions.end())
    {
        return 0.0;
    }

    double const priorMakespanUs = sequence.makespanUs;
    double const priorUncertaintyUs = sequence.uncertaintyUs;
    double const serviceStartUs = std::max(priorMakespanUs, std::max(0.0, completion->readyAfterUs));
    double const completionUs = serviceStartUs + completion->serviceMakespanUs;
    double const completionUncertaintyUs = priorUncertaintyUs + std::max(0.0, completion->readyUncertaintyUs)
        + std::max(0.0, completion->serviceUncertaintyUs);
    double const serviceAfterReadyUs
        = std::max(0.0, priorMakespanUs - completion->readyAfterUs) + completion->serviceMakespanUs;

    sequence.makespanUs = completionUs;
    sequence.uncertaintyUs = completionUncertaintyUs;
    updateDecodeService(sequence, completionUs, completionUncertaintyUs);
    return decodeViolationUs(serviceAfterReadyUs, completionUncertaintyUs, completion->serviceBudgetUs);
}

void applyDecodeSequenceGuard(
    PhaseGlobalActionCandidate& candidate, PhaseFormationSequence const& sequence, double budgetUs)
{
    if (!std::isfinite(sequence.decodeServiceUs) || !std::isfinite(budgetUs))
    {
        return;
    }
    auto completion = std::find_if(candidate.protectedCompletions.begin(), candidate.protectedCompletions.end(),
        [](PhaseProtectedCompletion const& value) { return value.kind == PhaseProtectedKind::kDecode; });
    double const sequenceRobust = sequence.decodeServiceUs + sequence.decodeServiceUncertaintyUs;
    if (completion == candidate.protectedCompletions.end())
    {
        candidate.protectedCompletions.push_back(
            {budgetUs, sequence.decodeServiceUs, sequence.decodeServiceUncertaintyUs, PhaseProtectedKind::kDecode});
        return;
    }
    double const currentRobust
        = std::max(0.0, completion->predictedCompletionUs) + std::max(0.0, completion->uncertaintyUs);
    completion->slackUs = std::min(completion->slackUs, budgetUs);
    if (sequenceRobust > currentRobust)
    {
        completion->predictedCompletionUs = sequence.decodeServiceUs;
        completion->uncertaintyUs = sequence.decodeServiceUncertaintyUs;
    }
}

} // namespace

bool PhaseFormationWork::empty() const noexcept
{
    return encoderRows == 0U && prefillRows == 0U && decodeRows == 0U;
}

PhaseFormationWork phaseFormationWork(PhaseGlobalActionCandidate const& candidate) noexcept
{
    PhaseFormationWork work;
    switch (candidate.key.kind)
    {
    case PhaseGlobalActionKind::kEncoder: work.encoderRows = candidate.primaryRequestIds.size(); break;
    case PhaseGlobalActionKind::kPrefill: work.prefillRows = candidate.primaryRequestIds.size(); break;
    case PhaseGlobalActionKind::kDecode: work.decodeRows = candidate.primaryRequestIds.size(); break;
    case PhaseGlobalActionKind::kEncoderPrefill:
        work.encoderRows = candidate.primaryRequestIds.size();
        work.prefillRows = candidate.secondaryRequestIds.size();
        break;
    case PhaseGlobalActionKind::kEncoderDecode:
        work.encoderRows = candidate.primaryRequestIds.size();
        work.decodeRows = candidate.secondaryRequestIds.size();
        break;
    case PhaseGlobalActionKind::kPrefillDecode:
        work.prefillRows = candidate.primaryRequestIds.size();
        work.decodeRows = candidate.secondaryRequestIds.size();
        break;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kWait: break;
    }
    return work;
}

uint64_t phaseFormationSnapshotId(PhaseFormationSnapshot const& snapshot) noexcept
{
    uint64_t result = snapshot.epoch;
    result = hashCombine(result, snapshot.ready.encoderRows);
    result = hashCombine(result, snapshot.ready.prefillRows);
    result = hashCombine(result, snapshot.ready.decodeRows);
    result = hashCombine(result,
        std::isfinite(snapshot.decodeServiceBudgetUs)
            ? static_cast<uint64_t>(std::llround(std::max(0.0, snapshot.decodeServiceBudgetUs)))
            : std::numeric_limits<uint64_t>::max());
    for (PhaseFormationKnownCompletion const& completion : snapshot.knownCompletions)
    {
        result = hashCombine(result, completion.eventId);
        result = hashCombine(result, static_cast<uint64_t>(std::llround(std::max(0.0, completion.readyAfterUs))));
        result = hashCombine(result, static_cast<uint64_t>(std::llround(std::max(0.0, completion.readyUncertaintyUs))));
        result = hashCombine(result, completion.work.encoderRows);
        result = hashCombine(result, completion.work.prefillRows);
        result = hashCombine(result, completion.work.decodeRows);
        result = hashCombine(result, static_cast<uint64_t>(std::llround(std::max(0.0, completion.serviceMakespanUs))));
        result
            = hashCombine(result, static_cast<uint64_t>(std::llround(std::max(0.0, completion.serviceUncertaintyUs))));
        result = hashCombine(result,
            std::isfinite(completion.serviceBudgetUs)
                ? static_cast<uint64_t>(std::llround(std::max(0.0, completion.serviceBudgetUs)))
                : std::numeric_limits<uint64_t>::max());
    }
    return result;
}

PhaseFormationTransition phaseFormationTransition(
    PhaseFormationSnapshot const& snapshot, PhaseFormationAction const& action) noexcept
{
    PhaseFormationTransition result;
    result.successor = snapshot;
    if (!contains(snapshot.ready, action.consumes) || action.consumes.empty() || action.makespanUs < 0.0)
    {
        return result;
    }
    result.feasible = true;
    result.completed = action.consumes;
    result.successor.ready = subtract(result.successor.ready, action.consumes);
    std::vector<PhaseFormationKnownCompletion> pending;
    pending.reserve(snapshot.knownCompletions.size());
    for (PhaseFormationKnownCompletion completion : snapshot.knownCompletions)
    {
        if (completion.readyAfterUs <= action.makespanUs)
        {
            result.successor.ready = add(result.successor.ready, completion.work);
        }
        else
        {
            completion.readyAfterUs -= action.makespanUs;
            pending.push_back(completion);
        }
    }
    result.successor.knownCompletions = std::move(pending);
    return result;
}

PhaseFormationOracleResult phaseFormationEvaluateH2(PhaseFormationSnapshot const& snapshot,
    std::vector<PhaseFormationAction> const& actions, PhaseFormationWork target) noexcept
{
    PhaseFormationOracleResult result;
    result.sequences.resize(actions.size());
    double bestCost = std::numeric_limits<double>::infinity();
    double bestUncertainty = std::numeric_limits<double>::infinity();
    double bestViolation = std::numeric_limits<double>::infinity();
    for (size_t firstIndex{}; firstIndex < actions.size(); ++firstIndex)
    {
        PhaseFormationSequence& sequence = result.sequences[firstIndex];
        sequence.firstAction = firstIndex;
        PhaseFormationAction const& first = actions[firstIndex];
        if (!first.initialEligible || !fitsRemaining({}, first.consumes, target))
        {
            continue;
        }
        PhaseFormationTransition const firstTransition = phaseFormationTransition(snapshot, first);
        if (!firstTransition.feasible)
        {
            continue;
        }
        sequence.completed = firstTransition.completed;
        sequence.makespanUs = first.makespanUs;
        sequence.uncertaintyUs = first.uncertaintyUs;
        sequence.protectedViolationUs = first.protectedViolationUs;
        updateDecodeService(sequence, first.decodeServiceUs, first.decodeServiceUncertaintyUs);
        sequence.feasible = covers(sequence.completed, target);
        if (!sequence.feasible)
        {
            double successorCost = std::numeric_limits<double>::infinity();
            double successorUncertainty = std::numeric_limits<double>::infinity();
            double successorDecodeViolation = std::numeric_limits<double>::infinity();
            std::optional<size_t> successorIndex;
            for (size_t index{}; index < actions.size(); ++index)
            {
                PhaseFormationAction const& successor = actions[index];
                if (!fitsRemaining(sequence.completed, successor.consumes, target))
                {
                    continue;
                }
                PhaseFormationTransition const transition
                    = phaseFormationTransition(firstTransition.successor, successor);
                if (!transition.feasible || !covers(add(sequence.completed, transition.completed), target))
                {
                    continue;
                }
                PhaseFormationSequence decodePreview = sequence;
                updateDecodeService(decodePreview, first.makespanUs + successor.decodeServiceUs,
                    first.uncertaintyUs + successor.decodeServiceUncertaintyUs);
                double const violation = decodeViolationUs(decodePreview.decodeServiceUs,
                    decodePreview.decodeServiceUncertaintyUs, snapshot.decodeServiceBudgetUs);
                if (violation < successorDecodeViolation
                    || (violation == successorDecodeViolation && successor.makespanUs < successorCost)
                    || (violation == successorDecodeViolation && successor.makespanUs == successorCost
                        && successor.uncertaintyUs < successorUncertainty))
                {
                    successorDecodeViolation = violation;
                    successorCost = successor.makespanUs;
                    successorUncertainty = successor.uncertaintyUs;
                    successorIndex = index;
                }
            }
            if (successorIndex.has_value())
            {
                PhaseFormationAction const& successor = actions[*successorIndex];
                updateDecodeService(sequence, first.makespanUs + successor.decodeServiceUs,
                    first.uncertaintyUs + successor.decodeServiceUncertaintyUs);
                sequence.feasible = true;
                sequence.successorAction = successorIndex;
                sequence.makespanUs += successorCost;
                sequence.uncertaintyUs += successorUncertainty;
                sequence.completed = target;
            }
        }
        if (!sequence.feasible)
        {
            sequence.makespanUs = std::numeric_limits<double>::infinity();
            sequence.uncertaintyUs = 0.0;
            continue;
        }
        double const residentDecodeViolationUs = decodeViolationUs(
            sequence.decodeServiceUs, sequence.decodeServiceUncertaintyUs, snapshot.decodeServiceBudgetUs);
        double const completionBoundaryViolationUs = appendConcreteCompletionBoundary(sequence, snapshot);
        sequence.decodeServiceViolationUs = std::max(residentDecodeViolationUs, completionBoundaryViolationUs);
        double const violation = std::max(sequence.protectedViolationUs, sequence.decodeServiceViolationUs);
        if (violation < bestViolation || (violation == bestViolation && sequence.makespanUs < bestCost)
            || (violation == bestViolation && sequence.makespanUs == bestCost
                && sequence.uncertaintyUs < bestUncertainty))
        {
            bestViolation = violation;
            bestCost = sequence.makespanUs;
            bestUncertainty = sequence.uncertaintyUs;
            result.selectedAction = firstIndex;
        }
    }
    return result;
}

PhaseFormationOracleResult phaseFormationApplyH2(std::vector<PhaseGlobalActionCandidate>& candidates,
    PhaseFormationSnapshot const& snapshot, PhaseFormationWork target, double targetReferenceWorkUs) noexcept
{
    std::vector<PhaseFormationAction> actions;
    actions.reserve(candidates.size());
    for (size_t index{}; index < candidates.size(); ++index)
    {
        PhaseGlobalActionCandidate const& candidate = candidates[index];
        double const makespan = candidate.decisionCostKnown && candidate.decisionMakespanUs > 0.0
            ? candidate.decisionMakespanUs
            : candidate.predictedMakespanUs > 0.0 ? candidate.predictedMakespanUs
                                                  : candidate.predictedBlockingUs;
        std::optional<PhaseProtectedCompletion> const decode = protectedDecode(candidate);
        actions.push_back({index, phaseFormationWork(candidate), std::max(0.0, makespan),
            std::max(0.0, candidate.uncertaintyUs), true,
            decode.has_value() ? std::max(0.0, decode->predictedCompletionUs) : std::numeric_limits<double>::infinity(),
            decode.has_value() ? std::max(0.0, decode->uncertaintyUs) : 0.0, protectedViolationUs(candidate)});
    }
    PhaseFormationOracleResult result = phaseFormationEvaluateH2(snapshot, actions, target);
    for (size_t index{}; index < candidates.size(); ++index)
    {
        if (index >= result.sequences.size() || !result.sequences[index].feasible)
        {
            continue;
        }
        candidates[index].predictedHorizonUs = result.sequences[index].makespanUs;
        candidates[index].horizonReferenceWorkUs = targetReferenceWorkUs;
        applyDecodeSequenceGuard(candidates[index], result.sequences[index], snapshot.decodeServiceBudgetUs);
    }
    return result;
}

PhaseFormationRegret phaseFormationPredictedRegret(
    PhaseFormationOracleResult const& result, size_t selectedAction) noexcept
{
    PhaseFormationRegret regret;
    if (!result.selectedAction.has_value() || selectedAction >= result.sequences.size()
        || *result.selectedAction >= result.sequences.size())
    {
        return regret;
    }
    PhaseFormationSequence const& selected = result.sequences[selectedAction];
    PhaseFormationSequence const& oracle = result.sequences[*result.selectedAction];
    if (!selected.feasible || !oracle.feasible || !std::isfinite(selected.makespanUs)
        || !std::isfinite(oracle.makespanUs))
    {
        return regret;
    }
    regret.valid = true;
    regret.selectedAction = selectedAction;
    regret.oracleAction = *result.selectedAction;
    regret.selectedHorizonUs = selected.makespanUs;
    regret.oracleHorizonUs = oracle.makespanUs;
    regret.selectedDecodeViolationUs = selected.decodeServiceViolationUs;
    regret.oracleDecodeViolationUs = oracle.decodeServiceViolationUs;
    regret.selectedProtectedViolationUs = selected.protectedViolationUs;
    regret.oracleProtectedViolationUs = oracle.protectedViolationUs;
    double const selectedViolation = std::max(selected.protectedViolationUs, selected.decodeServiceViolationUs);
    double const oracleViolation = std::max(oracle.protectedViolationUs, oracle.decodeServiceViolationUs);
    if (selectedViolation > oracleViolation)
    {
        regret.predictedRegretUs = selectedViolation - oracleViolation;
    }
    else if (selectedViolation == oracleViolation)
    {
        regret.predictedRegretUs = std::max(0.0, selected.makespanUs - oracle.makespanUs);
    }
    return regret;
}

bool phaseFormationShouldReplaceMyopic(
    PhaseFormationOracleResult const& result, size_t myopicAction, size_t selectedAction) noexcept
{
    if (myopicAction == selectedAction)
    {
        return false;
    }
    PhaseFormationRegret const myopicRegret = phaseFormationPredictedRegret(result, myopicAction);
    PhaseFormationRegret const selectedRegret = phaseFormationPredictedRegret(result, selectedAction);
    return myopicRegret.valid && selectedRegret.valid
        && selectedRegret.predictedRegretUs < myopicRegret.predictedRegretUs;
}

PhaseFormationRealizedTracker::PhaseFormationRealizedTracker(PhaseFormationRealizedTrackerConfig config)
    : mConfig(config)
{
    mConfig.maxDispatches = std::max<size_t>(1U, mConfig.maxDispatches);
    mConfig.maxActiveEpisodes = std::max<size_t>(1U, mConfig.maxActiveEpisodes);
}

void PhaseFormationRealizedTracker::appendDispatch(PhaseFormationRealizedEpisode& episode, uint64_t planId,
    PhaseGlobalActionKind action, PhaseFormationWork work, double timestampUs)
{
    if (episode.dispatches.size() >= mConfig.maxDispatches)
    {
        return;
    }
    double const dispatchedAfterUs = std::max(0.0, timestampUs - episode.startedAtUs);
    episode.dispatches.push_back({planId, action, work, dispatchedAfterUs});
    episode.encoderRows += work.encoderRows;
    episode.prefillRows += work.prefillRows;
    episode.decodeRows += work.decodeRows;
    episode.maxDecodeRows = std::max(episode.maxDecodeRows, work.decodeRows);
    if (!episode.decodeServiced && work.decodeRows > 0U)
    {
        episode.decodeServiced = true;
        episode.firstDecodeRows = work.decodeRows;
        episode.decodeServiceGapUs = dispatchedAfterUs;
    }
}

void PhaseFormationRealizedTracker::observeDispatch(
    uint64_t planId, PhaseGlobalActionKind action, PhaseFormationWork work, double timestampUs)
{
    for (PhaseFormationRealizedEpisode& episode : mActive)
    {
        appendDispatch(episode, planId, action, work, timestampUs);
    }
    finishReadyEpisodes(timestampUs, false);
}

void PhaseFormationRealizedTracker::startEpisode(PhaseFormationRealizedEpisodeStart const& start, uint64_t planId,
    PhaseGlobalActionKind action, PhaseFormationWork work)
{
    if (mActive.size() >= mConfig.maxActiveEpisodes)
    {
        PhaseFormationRealizedEpisode oldest = std::move(mActive.front());
        mActive.pop_front();
        finishEpisode(std::move(oldest), start.timestampUs, true);
    }
    PhaseFormationRealizedEpisode episode;
    episode.episodeId = ++mNextEpisodeId;
    episode.snapshotId = start.snapshotId;
    episode.selectedAction = start.selectedAction;
    episode.myopicAction = start.myopicAction;
    episode.oracleAction = start.oracleAction;
    episode.predictedRegretUs = std::max(0.0, start.predictedRegretUs);
    episode.decodeServiceBudgetUs = start.decodeServiceBudgetUs;
    episode.startedAtUs = start.timestampUs;
    appendDispatch(episode, planId, action, work, start.timestampUs);
    mActive.push_back(std::move(episode));
    ++mTelemetry.episodesStarted;
    finishReadyEpisodes(start.timestampUs, false);
}

void PhaseFormationRealizedTracker::remapPlan(uint64_t oldPlanId, uint64_t newPlanId) noexcept
{
    for (PhaseFormationRealizedEpisode& episode : mActive)
    {
        for (PhaseFormationRealizedDispatch& dispatch : episode.dispatches)
        {
            if (dispatch.planId == oldPlanId && !std::isfinite(dispatch.completionVisibleAfterUs))
            {
                dispatch.planId = newPlanId;
            }
        }
    }
}

void PhaseFormationRealizedTracker::observeCompletion(uint64_t planId, double timestampUs)
{
    for (PhaseFormationRealizedEpisode& episode : mActive)
    {
        double const completionAfterUs = std::max(0.0, timestampUs - episode.startedAtUs);
        for (PhaseFormationRealizedDispatch& dispatch : episode.dispatches)
        {
            if (dispatch.planId != planId || std::isfinite(dispatch.completionVisibleAfterUs))
            {
                continue;
            }
            dispatch.completionVisibleAfterUs = completionAfterUs;
            episode.horizonCompletionVisibleUs = std::max(episode.horizonCompletionVisibleUs, completionAfterUs);
            if (dispatch.work.decodeRows > 0U
                && (episode.decodeCompletionVisibleUs <= 0.0 || completionAfterUs < episode.decodeCompletionVisibleUs))
            {
                episode.decodeCompletionVisibleUs = completionAfterUs;
            }
        }
    }
    finishReadyEpisodes(timestampUs, false);
}

void PhaseFormationRealizedTracker::flush(double timestampUs)
{
    finishReadyEpisodes(timestampUs, true);
}

PhaseFormationRealizedTelemetry const& PhaseFormationRealizedTracker::telemetry() const noexcept
{
    return mTelemetry;
}

std::vector<PhaseFormationRealizedEpisode> PhaseFormationRealizedTracker::takeCompletedEpisodes()
{
    std::vector<PhaseFormationRealizedEpisode> result;
    result.reserve(mCompleted.size());
    while (!mCompleted.empty())
    {
        result.push_back(std::move(mCompleted.front()));
        mCompleted.pop_front();
    }
    return result;
}

void PhaseFormationRealizedTracker::finishReadyEpisodes(double timestampUs, bool force)
{
    auto episode = mActive.begin();
    while (episode != mActive.end())
    {
        bool const horizonFull = episode->dispatches.size() >= mConfig.maxDispatches;
        bool const completionsVisible = std::all_of(
            episode->dispatches.begin(), episode->dispatches.end(), [](PhaseFormationRealizedDispatch const& dispatch) {
                return std::isfinite(dispatch.completionVisibleAfterUs);
            });
        if (!force && (!horizonFull || !completionsVisible))
        {
            ++episode;
            continue;
        }
        PhaseFormationRealizedEpisode finished = std::move(*episode);
        episode = mActive.erase(episode);
        finishEpisode(std::move(finished), timestampUs, force && (!horizonFull || !completionsVisible));
    }
}

void PhaseFormationRealizedTracker::finishEpisode(
    PhaseFormationRealizedEpisode episode, double timestampUs, bool truncated)
{
    double const finishAfterUs = std::max(0.0, timestampUs - episode.startedAtUs);
    for (PhaseFormationRealizedDispatch& dispatch : episode.dispatches)
    {
        if (!std::isfinite(dispatch.completionVisibleAfterUs))
        {
            dispatch.completionVisibleAfterUs = finishAfterUs;
        }
        episode.horizonCompletionVisibleUs
            = std::max(episode.horizonCompletionVisibleUs, dispatch.completionVisibleAfterUs);
        if (dispatch.work.decodeRows > 0U
            && (episode.decodeCompletionVisibleUs <= 0.0
                || dispatch.completionVisibleAfterUs < episode.decodeCompletionVisibleUs))
        {
            episode.decodeCompletionVisibleUs = dispatch.completionVisibleAfterUs;
        }
    }
    if (!episode.decodeServiced)
    {
        episode.decodeServiceGapUs = episode.horizonCompletionVisibleUs;
    }
    if (std::isfinite(episode.decodeServiceBudgetUs))
    {
        episode.decodeServiceViolationUs = std::max(0.0, episode.decodeServiceGapUs - episode.decodeServiceBudgetUs);
    }
    episode.truncated = truncated;
    ++mTelemetry.episodesCompleted;
    mTelemetry.episodesTruncated += truncated ? 1U : 0U;
    mTelemetry.decodeServices += episode.decodeServiced ? 1U : 0U;
    mTelemetry.decodeBudgets += std::isfinite(episode.decodeServiceBudgetUs) ? 1U : 0U;
    mTelemetry.decodeServiceViolations += episode.decodeServiceViolationUs > 0.0 ? 1U : 0U;
    mTelemetry.decodeServiceGapUs += episode.decodeServiceGapUs;
    mTelemetry.maxDecodeServiceGapUs = std::max(mTelemetry.maxDecodeServiceGapUs, episode.decodeServiceGapUs);
    mTelemetry.decodeServiceViolationUs += episode.decodeServiceViolationUs;
    mTelemetry.maxDecodeServiceViolationUs
        = std::max(mTelemetry.maxDecodeServiceViolationUs, episode.decodeServiceViolationUs);
    mTelemetry.lastEpisode = episode;
    mCompleted.push_back(std::move(episode));
}

} // namespace trt_edgellm::rt
