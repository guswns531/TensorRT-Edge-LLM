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
    kEncoderPrefill,
    kEncoderDecode,
    kPrefillDecode,
    kWait,
};

//! Independently outstanding TensorRT phase contexts represented as a bit set.
enum class PhaseExecutionSet : uint8_t
{
    kNone = 0U,
    kEncoder = 1U,
    kPrefill = 2U,
    kDecode = 4U,
};

PhaseExecutionSet operator|(PhaseExecutionSet left, PhaseExecutionSet right) noexcept;
PhaseExecutionSet operator&(PhaseExecutionSet left, PhaseExecutionSet right) noexcept;
bool phaseExecutionSetContains(PhaseExecutionSet set, PhaseExecutionSet phase) noexcept;
bool phaseExecutionSetIsSubset(PhaseExecutionSet subset, PhaseExecutionSet superset) noexcept;
PhaseExecutionSet phaseExecutionSetForAction(PhaseGlobalActionKind kind) noexcept;

//! Staged activation keeps legacy serving behavior available while a global
//! decision stream is validated against the same live queue snapshots.
enum class PhaseGlobalSchedulerMode
{
    kDisabled,
    kShadow,
    kActive,
};

//! Selection authority used after the common deterministic builders run.
//! Compatibility mode is an evaluation control and never classifies a
//! workload; it replays the legacy P/D phase choice on the same snapshot.
enum class PhaseGlobalSelectionMode
{
    kProfileFree,
    kLegacyCompatibility,
};

//! CUDA launch path used by one action. Primary and secondary refer to the
//! corresponding row vectors in PhaseGlobalActionCandidate. Keeping graph
//! replay in the cost key prevents eager warmup samples from contaminating
//! steady-state replay estimates.
enum class PhaseExecutionVariant : uint8_t
{
    kEager = 0U,
    kPrimaryGraph = 1U,
    kSecondaryGraph = 2U,
    kBothGraph = 3U,
};

PhaseExecutionVariant phaseExecutionVariant(bool primaryGraph, bool secondaryGraph) noexcept;
bool phaseExecutionVariantUsesPrimaryGraph(PhaseExecutionVariant variant) noexcept;
bool phaseExecutionVariantUsesSecondaryGraph(PhaseExecutionVariant variant) noexcept;
char const* phaseExecutionVariantName(PhaseExecutionVariant variant) noexcept;

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
    PhaseExecutionVariant executionVariant{PhaseExecutionVariant::kEager};
    //! Distinguish overlap that begins after the primary phase has already consumed work.
    bool residualAugmentation{};

    bool operator==(PhaseGlobalActionKey const& other) const noexcept;
};

//! Canonicalize only overlap cost-learning dimensions. Candidate identity and
//! TensorRT bindings retain their exact shapes.
PhaseGlobalActionKey phaseGlobalCanonicalOverlapCostKey(PhaseGlobalActionKey key) noexcept;

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

//! Calibration state for one overlap shape. A calibrated shape is either
//! eligible or rejected as unprofitable; only the first two states need probes.
enum class PhaseGlobalOverlapCostStatus
{
    kNoSamples,
    kInsufficientSamples,
    kEligible,
    kUnprofitable,
};

struct PhaseGlobalOverlapCostDiagnostic
{
    PhaseGlobalOverlapCostStatus status{PhaseGlobalOverlapCostStatus::kNoSamples};
    size_t sampleCount{};
    float robustCompression{};
};

struct PhaseGlobalOverlapCostRecord
{
    PhaseGlobalActionKey key;
    PhaseGlobalOverlapCostDiagnostic diagnostic;
    size_t opportunityCount{};
    bool required{};
};

//! Stable telemetry name for one overlap calibration state.
char const* phaseGlobalOverlapCostStatusName(PhaseGlobalOverlapCostStatus status) noexcept;

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
    //! Interpolate a missing primary batch size only when direct observations
    //! with otherwise identical execution keys bracket it on both sides.
    std::optional<PhaseGlobalCostEstimate> estimateInterpolatedPrimaryBatch(
        PhaseGlobalActionKey const& key) const;
    PhaseGlobalOverlapCostDiagnostic overlapDiagnostic(PhaseGlobalActionKey const& key) const;
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

enum class PhaseGlobalDecisionReason
{
    kNoCandidate,
    kNoHardFeasibleCandidate,
    kDeadlineSafeEfficiency,
    kMinimumViolation,
    kLegacyCompatibility,
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
