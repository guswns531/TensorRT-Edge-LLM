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
#include <memory>
#include <optional>
#include <string_view>
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

enum class PhaseGlobalSchedulerMode
{
    kDisabled,
    kActive,
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

//! Phase that was already outstanding when a residual overlap was added.
enum class PhaseGlobalResidualAnchor : uint8_t
{
    kNone,
    kPrefill,
    kDecode,
};

PhaseExecutionVariant phaseExecutionVariant(bool primaryGraph, bool secondaryGraph) noexcept;
bool phaseExecutionVariantUsesPrimaryGraph(PhaseExecutionVariant variant) noexcept;
bool phaseExecutionVariantUsesSecondaryGraph(PhaseExecutionVariant variant) noexcept;
char const* phaseExecutionVariantName(PhaseExecutionVariant variant) noexcept;

//! Stable telemetry name for the phase that anchors a residual action.
char const* phaseGlobalResidualAnchorName(PhaseGlobalResidualAnchor anchor) noexcept;

//! Stable telemetry name for a global action kind.
char const* phaseGlobalActionKindName(PhaseGlobalActionKind kind) noexcept;
//! Parse the stable telemetry name used by research replay controls.
std::optional<PhaseGlobalActionKind> phaseGlobalActionKindFromName(std::string_view name) noexcept;

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
    //! Producer class for prefill-bearing actions: 0=unspecified, 1=text, 2=external.
    int32_t primaryWorkClass{};
    //! Distinguish overlap that begins after the primary phase has already consumed work.
    bool residualAugmentation{};
    //! Keep P->P+D and D->P+D observations in independent online-cost buckets.
    PhaseGlobalResidualAnchor residualAnchor{PhaseGlobalResidualAnchor::kNone};

    bool operator==(PhaseGlobalActionKey const& other) const noexcept;
};

//! Canonicalize overlap batch, chunk, and context dimensions into conservative
//! upper buckets. Candidate identity and TensorRT bindings retain exact shapes.
PhaseGlobalActionKey phaseGlobalCanonicalOverlapCostKey(PhaseGlobalActionKey key) noexcept;

//! One direct CUDA-event observation for a single action key.
struct PhaseGlobalCostObservation
{
    //! Reference work represented by the action. Overlap uses the exact
    //! same-shape serial actions; phase-only samples may retain an equivalent
    //! singleton-service reference for batch-efficiency prediction.
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
    std::optional<PhaseGlobalCostEstimate> estimateInterpolatedPrimaryBatch(PhaseGlobalActionKey const& key) const;
    //! Use the smallest observed primary context bucket that conservatively
    //! covers the requested bucket, with primary-batch interpolation inside it.
    std::optional<PhaseGlobalCostEstimate> estimatePrimaryBatchCoveringContext(PhaseGlobalActionKey const& key) const;
    //! Reuse only phase-only observations whose batch, chunk, and context
    //! geometry covers the requested action. This preserves fixed launch cost
    //! for ragged prefill shapes instead of extrapolating from token count.
    std::optional<PhaseGlobalCostEstimate> estimateCoveringPrimary(PhaseGlobalActionKey const& key) const;
    //! Minimum observed phase-only launch cost for the same phase, producer
    //! class, execution variant, and residual semantics. Geometry is ignored
    //! deliberately: this is a lower-bound intercept, not a shape estimate.
    std::optional<PhaseGlobalCostEstimate> estimatePrimaryLaunchFloor(PhaseGlobalActionKey const& key) const;
    //! Reuse only observations whose complete overlap geometry covers the
    //! requested geometry. Incomparable nearest covers are merged with the
    //! slowest robust cost so the estimate cannot cherry-pick a favorable
    //! batch or context dimension.
    std::optional<PhaseGlobalCostEstimate> estimateCoveringOverlap(PhaseGlobalActionKey const& key) const;
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

    using SampleMap = std::unordered_map<PhaseGlobalActionKey, Samples, KeyHash>;

    PhaseGlobalCostModelConfig mConfig;
    //! Mechanism previews copy the complete scheduler so every hidden queue
    //! state remains identical. Samples are read-only in those previews, so
    //! share them until an actual observation mutates one copy.
    std::shared_ptr<SampleMap> mSamples;
    //! One scheduling decision queries many decode batch sizes at the same
    //! context. Cache that conservative covering curve until observations
    //! change instead of rescanning every action record once per row.
    mutable std::unordered_map<PhaseGlobalActionKey, std::vector<std::optional<PhaseGlobalCostEstimate>>, KeyHash>
        mCoveringContextCache;
    //! Includes negative lookups for phase-only covering geometry.
    mutable std::unordered_map<PhaseGlobalActionKey, std::optional<PhaseGlobalCostEstimate>, KeyHash>
        mCoveringPrimaryCache;
    mutable std::unordered_map<PhaseGlobalActionKey, std::optional<PhaseGlobalCostEstimate>, KeyHash>
        mPrimaryLaunchFloorCache;
    //! Includes negative lookups. Runtime observations invalidate the cache.
    mutable std::unordered_map<PhaseGlobalActionKey, std::optional<PhaseGlobalCostEstimate>, KeyHash>
        mCoveringOverlapCache;
};

} // namespace trt_edgellm::rt
