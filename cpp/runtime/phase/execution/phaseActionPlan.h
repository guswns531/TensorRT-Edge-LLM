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

#include "runtime/phase/ownership/phaseOwnershipHorizon.h"
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
    bool safeProbeEligible{};
    //! Controlled calibration may remeasure a portable prior until enough
    //! exact node-local samples exist. This is never set by production policy.
    bool calibrationProbe{};
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
    uint64_t waitEventId{};
    std::vector<uint64_t> primaryRequestIds;
    std::vector<uint64_t> secondaryRequestIds;
    std::vector<int32_t> primaryStableSlotIds;
    std::vector<int32_t> secondaryStableSlotIds;

    bool permits(PhaseExecutionSet phases) const noexcept;
    bool launchMatches(PhaseExecutionSet phases) const noexcept;
};

//! Materialize one selected candidate into an explicit execution lease.
PhaseGlobalDispatchPlan phaseGlobalDispatchPlan(
    uint64_t planId, uint64_t snapshotEpoch, PhaseGlobalActionCandidate const& candidate);

//! Return the unfinished portion of an already launched single-phase action.
PhaseGlobalActionCandidate phaseGlobalResidualCandidate(
    PhaseGlobalActionCandidate const& launched, double elapsedUs) noexcept;

//! Upgrade a live P or D lease to E+P or E+D without authorizing a third phase.
std::optional<PhaseGlobalDispatchPlan> phaseGlobalAugmentedDispatchPlan(uint64_t planId, uint64_t snapshotEpoch,
    PhaseGlobalDispatchPlan const& active, PhaseGlobalActionCandidate const& augmentation) noexcept;

} // namespace trt_edgellm::rt
