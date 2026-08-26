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

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

namespace trt_edgellm::rt
{

//! One bounded scheduling action over independently enqueueable phase contexts.
enum class PhaseGlobalActionKind
{
    kNone,
    kEncoder,
    kPrefill,
    kDecode,
    kEncoderDecode,
    kPrefillDecode,
    kWait,
};

//! Staged activation keeps legacy serving behavior available while a global
//! decision stream is validated against the same live queue snapshots.
enum class PhaseGlobalSchedulerMode
{
    kDisabled,
    kShadow,
    kActive,
};

//! Stable telemetry name for a global action kind.
char const* phaseGlobalActionKindName(PhaseGlobalActionKind kind) noexcept;

//! Shape key for online action-cost observations. Bucketing belongs to the caller.
struct PhaseGlobalActionKey
{
    PhaseGlobalActionKind kind{PhaseGlobalActionKind::kNone};
    int32_t primaryBatchSize{};
    int32_t secondaryBatchSize{};
    int32_t chunkLength{};
    int32_t primaryContextBucket{};
    int32_t secondaryContextBucket{};

    bool operator==(PhaseGlobalActionKey const& other) const noexcept;
};

//! One direct CUDA-event observation for a single action key.
struct PhaseGlobalCostObservation
{
    //! Sum of isolated one-request service quanta completed by the action.
    float referenceWorkMs{};
    //! Observed end-to-end GPU makespan of the complete serial or overlap action.
    float makespanMs{};
};

//! Robust online estimate used for deadline and efficiency decisions.
struct PhaseGlobalCostEstimate
{
    size_t sampleCount{};
    float referenceWorkMedianMs{};
    float makespanMedianMs{};
    float makespanP95Ms{};
    float uncertaintyMs{};
};

struct PhaseGlobalCostModelConfig
{
    size_t windowSize{32U};
    size_t overlapMinSamples{4U};
    float coldStartUncertaintyMs{2.0F};
    float minimumOverlapGainRatio{0.02F};
};

//! Bounded direct-observation model. Unknown overlap remains ineligible unless
//! a caller explicitly marks one candidate as a safe, rate-limited probe.
class PhaseGlobalCostModel
{
public:
    explicit PhaseGlobalCostModel(PhaseGlobalCostModelConfig config = {});

    void observe(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation observation);
    std::optional<PhaseGlobalCostEstimate> estimate(PhaseGlobalActionKey const& key) const;
    bool overlapEligible(PhaseGlobalActionKey const& key) const;
    void reset();

private:
    struct Samples
    {
        std::deque<PhaseGlobalCostObservation> values;
    };

    struct KeyHash
    {
        size_t operator()(PhaseGlobalActionKey const& key) const noexcept;
    };

    PhaseGlobalCostModelConfig mConfig;
    std::unordered_map<PhaseGlobalActionKey, Samples, KeyHash> mSamples;
};

//! Ownership changes caused by one action. Near reclaim is a ranking signal and
//! never contributes to hard feasibility until its completion event is observed.
struct PhaseActionMemoryHorizon
{
    size_t managedBytes{};
    size_t allocateBytes{};
    size_t immediateReclaimBytes{};
    size_t guaranteedGrowthBytes{};
    size_t nearReclaimBytes{};
    size_t budgetBytes{};
    bool immediateReclaimObserved{};
};

//! One request deadline protected while evaluating an action. Completion may
//! include a required follow-up phase when the candidate does not advance the
//! request owning this deadline.
struct PhaseProtectedCompletion
{
    double slackUs{std::numeric_limits<double>::infinity()};
    double predictedCompletionUs{};
    double uncertaintyUs{};
};

//! A phase-local candidate after compatibility batching but before global selection.
struct PhaseGlobalActionCandidate
{
    PhaseGlobalActionKey key;
    std::vector<uint64_t> requestIds;
    //! Invariant results supplied by mechanism-only components.
    bool dependencySafe{true};
    bool contextSafe{true};
    bool shapeSafe{true};
    //! Overlap candidates require direct observations or one explicitly safe probe.
    bool overlapCostKnown{true};
    bool safeProbeEligible{};
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
    std::vector<PhaseProtectedCompletion> protectedCompletions;
    double referenceWorkUs{};
    double requestServiceLagUs{};
    PhaseActionMemoryHorizon memory;
};

enum class PhaseGlobalDecisionReason
{
    kNoCandidate,
    kNoHardFeasibleCandidate,
    kDeadlineSafeEfficiency,
    kMinimumViolation,
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

//! Profile-free selector shared by text and multimodal request DAGs.
//!
//! Mechanism components generate a small candidate frontier. This selector
//! enforces dependency, context, shape, ownership, and robust deadline safety
//! before comparing reference-service compression and memory lifetime.
class PhaseGlobalScheduler
{
public:
    explicit PhaseGlobalScheduler(PhaseGlobalSchedulerConfig config = {});

    PhaseGlobalDecision select(std::vector<PhaseGlobalActionCandidate> const& candidates) const;

private:
    PhaseGlobalSchedulerConfig mConfig;
};

} // namespace trt_edgellm::rt
