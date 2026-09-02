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

#include "runtime/phase/mechanism/phaseUnifiedEvent.h"

#include <cstdint>
#include <vector>

namespace trt_edgellm::rt
{

//! Coarse start-skew identity used by the bounded incremental action space.
enum class PhaseStartSkewBucket : int8_t
{
    kUnknown = -1,
    kImmediate = 0,
    kQuarter = 25,
    kHalf = 50,
    kThreeQuarter = 75,
    kNearComplete = 90,
    kSerial = 100,
};

//! Deterministic legality outcome. Policy never overrides these reasons.
enum class PhaseIncrementalLegalityReason : uint8_t
{
    kLegal,
    kInvalidOutstandingSet,
    kNoDispatchMismatch,
    kMissingNewcomer,
    kSameContextReenqueue,
    kThirdContext,
    kIncumbentMismatch,
    kDirectionMismatch,
    kPlannedOutstandingMismatch,
    kDependencyUnsafe,
    kContextUnsafe,
    kShapeUnsafe,
};

//! One ready phase-local cohort supplied by a deterministic cohort builder.
struct PhaseIncrementalReadyAction
{
    PhaseUnifiedPhase phase{PhaseUnifiedPhase::kNone};
    //! Stable phase-local candidate identity, including shape and row order.
    uint64_t candidateId{};
    bool dependencySafe{true};
    bool contextSafe{true};
    bool shapeSafe{true};
};

//! Complete identity of one bounded incremental dispatch or NO_DISPATCH action.
struct PhaseIncrementalActionKey
{
    //! Stable identity of the participating phase-local cohort or pair.
    uint64_t cohortId{};
    PhaseExecutionSet outstandingBefore{PhaseExecutionSet::kNone};
    PhaseExecutionSet plannedOutstanding{PhaseExecutionSet::kNone};
    PhaseUnifiedPhase incumbentPhase{PhaseUnifiedPhase::kNone};
    PhaseUnifiedPhase newcomerPhase{PhaseUnifiedPhase::kNone};
    PhaseUnifiedActionDirection direction{PhaseUnifiedActionDirection::kNone};
    PhaseStartSkewBucket startSkew{PhaseStartSkewBucket::kUnknown};
    bool noDispatch{};

    bool operator==(PhaseIncrementalActionKey const& other) const noexcept;
};

struct PhaseIncrementalAction
{
    PhaseIncrementalActionKey key;
    uint64_t actionId{};
    PhaseIncrementalLegalityReason legality{PhaseIncrementalLegalityReason::kLegal};

    bool legal() const noexcept
    {
        return legality == PhaseIncrementalLegalityReason::kLegal;
    }
};

char const* phaseStartSkewBucketName(PhaseStartSkewBucket bucket) noexcept;
char const* phaseIncrementalLegalityReasonName(PhaseIncrementalLegalityReason reason) noexcept;

//! Map a measured/requested normalized offset to the M2/M3 action bucket.
PhaseStartSkewBucket phaseStartSkewBucketFromFraction(double fraction, bool intervalsOverlap) noexcept;

//! Validate one action independently of policy value and execution costs.
PhaseIncrementalLegalityReason phaseIncrementalActionLegality(PhaseIncrementalActionKey const& key,
    bool dependencySafe = true, bool contextSafe = true, bool shapeSafe = true) noexcept;

//! Return the stable action identity including direction, order, and start skew.
uint64_t phaseIncrementalActionId(PhaseIncrementalActionKey const& key) noexcept;

//! Enumerate the complete bounded V1 frontier. Illegal actions are not returned.
//!
//! With no outstanding context this returns NO_DISPATCH, every ready singleton,
//! and both launch orders for every ready pair. With one outstanding context it
//! returns NO_DISPATCH and every legal distinct newcomer. With two outstanding
//! contexts only NO_DISPATCH is legal. Invalid or three-phase snapshots return
//! an empty frontier.
std::vector<PhaseIncrementalAction> phaseEnumerateIncrementalActions(PhaseInFlightSnapshot const& snapshot,
    std::vector<PhaseIncrementalReadyAction> const& ready,
    PhaseStartSkewBucket incumbentStartSkew = PhaseStartSkewBucket::kUnknown);

//! Materialize and validate the incremental identity of a selected whole-action lease.
PhaseIncrementalAction phaseIncrementalActionForDispatch(PhaseExecutionSet outstandingBefore, uint64_t cohortId,
    PhaseGlobalActionKind action, PhaseUnifiedActionDirection direction, PhaseStartSkewBucket startSkew) noexcept;

} // namespace trt_edgellm::rt
