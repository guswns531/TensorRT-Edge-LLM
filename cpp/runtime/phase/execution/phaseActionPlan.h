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

#include "runtime/phase/mechanism/phaseIncrementalAction.h"
#include "runtime/phase/ownership/phaseOwnershipHorizon.h"
#include "runtime/phase/policy/phaseContextualPdModel.h"
#include "runtime/phase/policy/phaseDeadline.h"
#include "runtime/phase/policy/phaseGlobalCostModel.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace trt_edgellm::rt
{

//! A phase-local candidate after compatibility batching but before global selection.
struct PhaseGlobalActionCandidate
{
    //! Stable identity derived from the action shape and ordered request rows.
    uint64_t candidateId{};
    //! Host time already spent forming/selecting this externally staged action.
    double hostDecisionUs{};
    PhaseGlobalActionKey key;
    //! Primary and secondary row vectors retain phase-local canonical order.
    std::vector<uint64_t> primaryRequestIds;
    std::vector<uint64_t> secondaryRequestIds;
    //! Stable physical KV ownership follows the exact corresponding row order.
    //! Encoder-only candidates may leave these vectors empty because the KV
    //! lease is acquired at the downstream prefill transition.
    std::vector<int32_t> primaryStableSlotIds;
    std::vector<int32_t> secondaryStableSlotIds;
    std::vector<uint64_t> requestIds;
    //! Invariant results supplied by mechanism-only components.
    bool dependencySafe{true};
    bool contextSafe{true};
    bool shapeSafe{true};
    //! Overlap candidates require direct observations or one explicitly safe probe.
    bool overlapCostKnown{true};
    //! A measured overlap may be feasible for deadline recovery even when it
    //! does not compress serial work. Profitability remains a ranking signal.
    bool overlapCostProfitable{true};
    bool safeProbeEligible{};
    //! Controlled calibration may remeasure a portable prior until enough
    //! exact node-local samples exist. This is never set by production policy.
    bool calibrationProbe{};
    PhaseContextualPdFeatures contextualPdFeatures{};
    bool contextualPdFeatureValid{};
    bool contextualPdExploration{};
    double contextualPdMean{};
    double contextualPdUncertainty{};
    double contextualPdLowerConfidenceBound{};
    //! Encoder pair policy evidence is consumed by the three-phase global
    //! coordinator and recorded with the matching E+P or E+D completion.
    PhaseContextualPdFeatures contextualEncoderPairFeatures{};
    bool contextualEncoderPairFeatureValid{};
    bool contextualEncoderPairReady{};
    bool contextualEncoderPairExploration{};
    double contextualEncoderPairMean{};
    double contextualEncoderPairUncertainty{};
    double contextualEncoderPairLowerConfidenceBound{};
    PhaseContextualPairDirection contextualEncoderPairDirection{PhaseContextualPairDirection::kEncoderToPrefill};
    //! Sum of the two isolated batch makespans. This is intentionally
    //! separate from throughput-oriented service reference work.
    double contextualEncoderPairReferenceWorkUs{};
    //! WAIT is valid only for an already outstanding completion source.
    bool concreteWaitEvent{};
    uint64_t waitEventId{};
    //! Robust scheduling inputs. The candidate generator may use an online estimate
    //! or a conservative cold-start bound.
    double minimumProtectedSlackUs{std::numeric_limits<double>::infinity()};
    //! Legacy scalar completion bound used when protectedCompletions is empty.
    double predictedBlockingUs{};
    double uncertaintyUs{};
    //! GPU makespan of this action only. Zero falls back to predictedBlockingUs.
    double predictedMakespanUs{};
    //! Low-dimensional policy estimate used only to rank already-safe actions.
    //! Exact CUDA costs and protected completion bounds remain authoritative
    //! for feasibility and deadline checks. This separation lets contextual
    //! evidence interpolate across shapes without pretending to be an exact
    //! execution-cost observation.
    bool decisionCostKnown{};
    double decisionMakespanUs{};
    //! The contextual scalar posterior, rather than an exact execution-cost
    //! observation, supplied decisionMakespanUs for this candidate. This bit
    //! permits same-frontier fallback attribution without changing the live
    //! policy or discarding exact CUDA costs.
    bool contextualScalarAuthorityApplied{};
    //! Bounded decision-horizon cost. WAIT comparisons use the same future work
    //! on both NOW and WAIT alternatives so dispatching work now is not treated
    //! as if it left no residual work. Zero falls back to the action makespan.
    double predictedHorizonUs{};
    std::vector<PhaseProtectedCompletion> protectedCompletions;
    double referenceWorkUs{};
    //! Reference work covered by predictedHorizonUs. Zero falls back to
    //! referenceWorkUs and therefore preserves ordinary one-action selection.
    double horizonReferenceWorkUs{};
    double requestServiceLagUs{};
    PhaseActionMemoryHorizon memory;
};

//! Remove only contextual scalar authority while retaining exact CUDA cost,
//! feasibility, SLO, ownership, and row-order state. Completion authority is
//! restored first because its scalar baseline may itself be contextual.
inline void phaseRestoreNonContextualPolicy(PhaseGlobalActionCandidate& candidate)
{
    if (candidate.contextualScalarAuthorityApplied)
    {
        candidate.decisionCostKnown = false;
        candidate.decisionMakespanUs = 0.0;
    }
}

//! Return a deterministic identity without changing phase-local row order.
uint64_t phaseGlobalCandidateId(PhaseGlobalActionCandidate const& candidate) noexcept;

//! Rebuild the aggregate ownership vector and stable identity after phase-local
//! row vectors have been finalized.
void phaseGlobalFinalizeCandidate(PhaseGlobalActionCandidate& candidate);

//! An execution lease remains authoritative until its launched phases complete
//! or a newer snapshot explicitly replaces it before enqueue.
struct PhaseGlobalDispatchPlan
{
    uint64_t planId{};
    uint64_t snapshotEpoch{};
    uint64_t candidateId{};
    PhaseGlobalActionKind action{PhaseGlobalActionKind::kNone};
    PhaseExecutionSet allowedOutstanding{PhaseExecutionSet::kNone};
    PhaseExecutionSet launched{PhaseExecutionSet::kNone};
    //! M3 action identity is the correctness authority for 0/1/2-context transitions.
    PhaseIncrementalAction incrementalAction;
    uint64_t waitEventId{};
    std::vector<uint64_t> primaryRequestIds;
    std::vector<uint64_t> secondaryRequestIds;
    std::vector<int32_t> primaryStableSlotIds;
    std::vector<int32_t> secondaryStableSlotIds;
    PhaseContextualPdFeatures contextualPdFeatures{};
    bool contextualPdFeatureValid{};
    bool contextualPdExploration{};

    bool permits(PhaseExecutionSet phases) const noexcept;
    bool launchMatches(PhaseExecutionSet phases) const noexcept;
};

//! Materialize one selected candidate into an explicit execution lease.
PhaseGlobalDispatchPlan phaseGlobalDispatchPlan(uint64_t planId, uint64_t snapshotEpoch,
    PhaseGlobalActionCandidate const& candidate, PhaseExecutionSet outstandingBefore = PhaseExecutionSet::kNone,
    PhaseUnifiedActionDirection direction = PhaseUnifiedActionDirection::kNone,
    PhaseStartSkewBucket startSkew = PhaseStartSkewBucket::kImmediate);

//! Return the unfinished portion of an already launched single-phase action.
PhaseGlobalActionCandidate phaseGlobalResidualCandidate(
    PhaseGlobalActionCandidate const& launched, double elapsedUs) noexcept;

//! Upgrade a live P or D lease to E+P, E+D, or P+D without authorizing a third phase.
std::optional<PhaseGlobalDispatchPlan> phaseGlobalAugmentedDispatchPlan(uint64_t planId, uint64_t snapshotEpoch,
    PhaseGlobalDispatchPlan const& active, PhaseGlobalActionCandidate const& augmentation,
    PhaseStartSkewBucket startSkew = PhaseStartSkewBucket::kUnknown) noexcept;

} // namespace trt_edgellm::rt
