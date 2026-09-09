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

#include "runtime/phase/execution/phaseActionPlan.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace trt_edgellm::rt
{

enum class PhaseGlobalDecisionReason
{
    kNoCandidate,
    kNoHardFeasibleCandidate,
    kDeadlineSafeEfficiency,
    kBoundedExploration,
    kMinimumViolation,
    //! Every feasible action misses the same protected phase set; select the
    //! action that clears the common work horizon most efficiently.
    kAllLateEfficiencyRecovery,
};

//! Result of one bounded global scheduling decision.
struct PhaseGlobalDecision
{
    std::optional<size_t> selectedIndex;
    PhaseGlobalDecisionReason reason{PhaseGlobalDecisionReason::kNoCandidate};
    size_t inputCandidates{};
    size_t hardFeasibleCandidates{};
    size_t deadlineSafeCandidates{};
    size_t dominatedCandidates{};
    double predictedViolationUs{};
    double serviceCompression{};
    size_t hardPeakManagedBytes{};
};

struct PhaseGlobalSchedulerConfig
{
    size_t maxCandidates{11U};
    double deadlineGuardUs{};
};

//! Evaluation of one actual selector input, not a preview-frontier candidate.
struct PhaseGlobalCandidateAudit
{
    uint64_t candidateId{};
    bool hardFeasible{};
    double predictedViolationUs{};
    bool frontierEligible{};
    bool dominated{};
    PhaseGlobalActionKind kind{PhaseGlobalActionKind::kNone};
    uint32_t violationMask{};
    double additionalViolationUs{};
    double serviceCompression{};
    double selectionHorizonUs{};
    int32_t primaryBatchSize{};
    int32_t secondaryBatchSize{};
};

//! Local queue guard outcome before global candidate ranking.
struct PhaseDecodeGuardAudit
{
    bool prefillExpired{};
    bool decodeExpired{};
    bool candidateRestored{};
    bool candidateSuppressed{};
};

//! Captured at selection; a later dispatch override is not this decision.
struct PhaseGlobalSelectionAudit
{
    std::vector<PhaseGlobalCandidateAudit> inputs;
    PhaseGlobalDecision decision;
    std::vector<PhaseGlobalCandidateAudit> pdInputs;
    PhaseGlobalDecision pdDecision;
    std::optional<PhaseDecodeGuardAudit> decodeGuard;
    std::optional<PhaseDecodeGuardAudit> pdDecodeGuard;
};

//! Profile-free selector shared by text and multimodal request DAGs.
//!
//! Mechanism components generate a small candidate frontier. This selector
//! enforces dependency, context, shape, ownership, and robust deadline safety
//! before comparing reference-service compression and memory lifetime.
class PhaseGlobalScheduler
{
public:
    explicit PhaseGlobalScheduler(PhaseGlobalSchedulerConfig config = {});

    PhaseGlobalDecision select(std::vector<PhaseGlobalActionCandidate> const& candidates,
        std::vector<PhaseGlobalCandidateAudit>* audit = nullptr) const;

private:
    PhaseGlobalSchedulerConfig mConfig;
};

} // namespace trt_edgellm::rt
