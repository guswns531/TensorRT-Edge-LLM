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
    //! Producer class for prefill-bearing actions: 0=unspecified, 1=text, 2=external.
    int32_t primaryWorkClass{};
    //! Distinguish overlap that begins after the primary phase has already consumed work.
    bool residualAugmentation{};

    bool operator==(PhaseGlobalActionKey const& other) const noexcept;
};

//! Canonicalize overlap batch, chunk, and context dimensions into conservative
//! upper buckets. Candidate identity and TensorRT bindings retain exact shapes.
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
    std::optional<PhaseGlobalCostEstimate> estimateInterpolatedPrimaryBatch(PhaseGlobalActionKey const& key) const;
    //! Use the smallest observed primary context bucket that conservatively
    //! covers the requested bucket, with primary-batch interpolation inside it.
    std::optional<PhaseGlobalCostEstimate> estimatePrimaryBatchCoveringContext(PhaseGlobalActionKey const& key) const;
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
};

} // namespace trt_edgellm::rt
