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

#include "runtime/phase/policy/phaseContextualPdModel.h"
#include "runtime/phase/policy/phaseGlobalCostModel.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <optional>
#include <unordered_map>

namespace trt_edgellm::rt
{

//! Configuration for bounded, process-local CUDA timing observations.
struct PhaseRuntimeCostTrackerConfig
{
    PhaseGlobalCostModelConfig action;
    size_t actionMinimumSamples{4U};
    size_t decodeMinimumSamples{8U};
    size_t decodeWindowSize{32U};
    int32_t decodeContextBucketTokens{512};
    PhaseContextualPdModelConfig contextualPd;
    PhaseContextualPdModelConfig contextualEp;
    PhaseContextualPdModelConfig contextualEd;
    //! Pair-common evidence is worth this many virtual observations when it
    //! is blended with a direction-specific completion posterior.
    double completionDirectionPseudoObservations{4.0};
    PhaseContextualCompletionCalibrationConfig completionCalibration;
};

enum class PhaseRuntimeCostConfidence
{
    kUnknown,
    kWarming,
    kReady,
};

//! Chronological completion-policy calibration stage for one ordered phase
//! direction.  A direction first fits its posterior, then contributes to the
//! pair-family conformal scale, and only then supplies held-out authority
//! evidence.  The final stage means that the bounded evidence window is full
//! enough to make a validation decision; it does not imply that validation
//! passed.
enum class PhaseContextualCompletionCalibrationStage
{
    kPosteriorFit,
    kUncertaintyCalibration,
    kAuthorityValidation,
    kComplete,
};

char const* phaseContextualCompletionCalibrationStageName(PhaseContextualCompletionCalibrationStage stage) noexcept;

struct PhaseContextualCompletionCalibrationProgress
{
    PhaseContextualCompletionCalibrationStage stage{PhaseContextualCompletionCalibrationStage::kPosteriorFit};
    size_t posteriorObservations{};
    size_t posteriorMinimumObservations{};
    size_t uncertaintyObservations{};
    size_t uncertaintyMinimumObservations{};
    size_t authorityObservations{};
    size_t authorityMinimumObservations{};
};

//! Process-local measurements used by the next scheduling decision.
//!
//! The tracker neither persists observations nor assigns policy authority.
//! A bounded sample window replaces old measurements by observation count,
//! so no wall-clock TTL, startup anchor, or drift state is required.
class PhaseRuntimeCostTracker
{
public:
    explicit PhaseRuntimeCostTracker(PhaseRuntimeCostTrackerConfig config = {});

    void observe(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation observation);
    std::optional<PhaseGlobalCostEstimate> estimate(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> trustedEstimate(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> estimateInterpolatedPrimaryBatch(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> estimatePrimaryBatchCoveringContext(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> estimateCoveringPrimary(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> estimatePrimaryLaunchFloor(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> trustedEstimatePrimaryBatchCoveringContext(
        PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> trustedEstimateCoveringOverlap(PhaseGlobalActionKey const& key) const;
    PhaseGlobalOverlapCostDiagnostic overlapDiagnostic(PhaseGlobalActionKey const& key) const;
    size_t sampleCount(PhaseGlobalActionKey const& key) const;
    PhaseRuntimeCostConfidence confidence(PhaseGlobalActionKey const& key) const;
    bool overlapEligible(PhaseGlobalActionKey const& key) const;

    PhaseContextualPdEstimate predictContextualPd(PhaseContextualPdFeatures const& features);
    bool observeContextualPd(
        PhaseContextualPdFeatures const& features, double normalizedAdvantage, double weight = 1.0);
    void recordContextualPdSelection(bool overlap, bool exploration) noexcept;
    PhaseContextualPdModelConfig const& contextualPdConfig() const noexcept;
    PhaseContextualPdTelemetry contextualPdTelemetry() const noexcept;

    PhaseContextualPdEstimate predictContextualPair(
        PhaseContextualPairKind kind, PhaseContextualPdFeatures const& features);
    bool observeContextualPair(PhaseContextualPairKind kind, PhaseContextualPdFeatures const& features,
        double normalizedAdvantage, double weight = 1.0);
    void recordContextualPairSelection(PhaseContextualPairKind kind, bool overlap, bool exploration) noexcept;
    PhaseContextualPdModelConfig const& contextualPairConfig(PhaseContextualPairKind kind) const noexcept;
    PhaseContextualPdTelemetry contextualPairTelemetry(PhaseContextualPairKind kind) const noexcept;

    PhaseContextualPdEstimate predictContextualDirection(
        PhaseContextualPairDirection direction, PhaseContextualPdFeatures const& features);
    bool observeContextualDirection(PhaseContextualPairDirection direction, PhaseContextualPdFeatures const& features,
        double normalizedAdvantage, double weight = 1.0);
    void recordContextualDirectionSelection(
        PhaseContextualPairDirection direction, bool overlap, bool exploration) noexcept;
    PhaseContextualPdTelemetry const& contextualDirectionTelemetry(
        PhaseContextualPairDirection direction) const noexcept;

    PhaseContextualCompletionEstimate predictContextualCompletionDirection(PhaseContextualPairDirection direction,
        PhaseContextualPdFeatures const& features, double incumbentReferenceUs, double newcomerReferenceUs);
    bool observeContextualCompletionDirection(PhaseContextualPairDirection direction,
        PhaseContextualPdFeatures const& features, double incumbentReferenceUs, double newcomerReferenceUs,
        double incumbentCompletionUs, double newcomerCompletionUs,
        double minimumSlackUs = std::numeric_limits<double>::infinity(),
        double scalarDecisionMakespanUs = std::numeric_limits<double>::quiet_NaN());
    PhaseContextualCompletionTelemetry const& contextualCompletionDirectionTelemetry(
        PhaseContextualPairDirection direction) const noexcept;
    PhaseContextualCompletionTelemetry const& contextualCompletionPairTelemetry(
        PhaseContextualPairKind kind) const noexcept;
    PhaseContextualCompletionAuthorityEvidence contextualCompletionAuthorityEvidence(
        PhaseContextualPairDirection direction) const noexcept;
    PhaseContextualCompletionCalibrationEstimate contextualCompletionCalibration(PhaseContextualPairKind kind) const;
    PhaseContextualCompletionCalibrationProgress contextualCompletionCalibrationProgress(
        PhaseContextualPairDirection direction) const noexcept;
    bool contextualCompletionAuthorityEnabled() const noexcept
    {
        return mConfig.completionCalibration.enabled && mConfig.completionCalibration.active;
    }
    //! Return true only after the ordered direction and its conformal
    //! uncertainty have enough held-out evidence for policy authority.
    bool contextualCompletionAuthorityEvidenceReady(PhaseContextualPairDirection direction) const noexcept;
    //! Return true once enough held-out evidence exists to either promote or
    //! reject authority.  Unlike EvidenceReady(), a completed but failed
    //! validation is terminal for bounded warmup rather than an endless probe.
    bool contextualCompletionAuthorityEvidenceComplete(PhaseContextualPairDirection direction) const noexcept;
    bool contextualCompletionAuthorityReady(
        PhaseContextualPairDirection direction, PhaseContextualCompletionEstimate const& estimate) const noexcept;
    bool contextualCompletionAuthorityPredictsIncumbent() const noexcept
    {
        return mConfig.completionCalibration.authorityPredictsIncumbent;
    }
    //! Return an empirical-risk blend. Held-out completion error must improve
    //! on the isolated scalar prior; the relative improvement continuously
    //! determines how much of the configured maximum authority is granted.
    double contextualCompletionAuthorityBlendWeight(PhaseContextualPairDirection direction) const noexcept;
    //! Grant each completion boundary authority from its own held-out risk.
    //! One accurate component cannot hide or force authority on a harmful peer.
    double contextualCompletionAuthorityComponentBlendWeight(
        PhaseContextualPairDirection direction, bool incumbent) const noexcept;
    PhaseContextualCompletionEstimate contextualCompletionAuthorityEstimate(
        PhaseContextualPairDirection direction, PhaseContextualCompletionEstimate estimate) const noexcept;

    //! Record the decode component separately from a combined action makespan.
    void observeDecode(
        int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive, float gpuMs);
    std::optional<float> decodeP95(
        int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive) const;
    //! Conservative p95 from the nearest observed batch/context buckets that
    //! both cover the requested contended decode shape.
    std::optional<float> decodeCoveringP95(
        int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive) const;
    size_t decodeBucketCount() const noexcept;

    //! Drop exact action/decode service observations while preserving the
    //! contextual policy posterior. CUDA graph/profile state is owned by the
    //! execution coordinator and is therefore unaffected.
    void resetExecutionCostHistory();
    //! Drop contextual action-value, completion, and uncertainty-calibration
    //! state while preserving exact execution observations.
    void resetPolicyPosterior();
    //! Start a new serving-distribution validation epoch while retaining the
    //! generic posterior and conformal calibration learned during warmup.
    //! Completion estimates regain authority only from held-out observations
    //! collected in the new epoch.
    void resetCompletionAuthorityEvidence() noexcept;
    //! Drop every process-local execution and policy observation.
    void reset();

private:
    PhaseContextualPdFeatures contextualCompletionFeaturesForPolicy(PhaseContextualPdFeatures features) const noexcept;

    struct CompletionAuthorityEvidenceSample
    {
        bool incumbentCovered{};
        bool newcomerCovered{};
        bool predictedSafe{};
        bool falseSafe{};
        double completionAbsoluteErrorUs{};
        double referenceAbsoluteErrorUs{};
        double incumbentCompletionAbsoluteErrorUs{};
        double incumbentReferenceAbsoluteErrorUs{};
        double newcomerCompletionAbsoluteErrorUs{};
        double newcomerReferenceAbsoluteErrorUs{};
    };

    struct CompletionAuthorityEvidenceWindow
    {
        std::deque<CompletionAuthorityEvidenceSample> samples;
        PhaseContextualCompletionAuthorityEvidence aggregate;
    };

    void observeContextualCompletionAuthorityEvidence(PhaseContextualPairDirection direction,
        PhaseContextualCompletionEstimate const& prediction, PhaseContextualPdFeatures const& features,
        double incumbentReferenceUs, double newcomerReferenceUs, double incumbentCompletionUs,
        double newcomerCompletionUs, double minimumSlackUs, double scalarDecisionMakespanUs) noexcept;
    bool contextualCompletionAuthorityEvidenceSatisfies(
        PhaseContextualCompletionAuthorityEvidence const& evidence, double coverageTolerance) const noexcept;

    struct DecodeKey
    {
        int32_t batchSize{};
        int32_t contextBucket{};
        bool encoderActive{};
        bool prefillActive{};

        bool operator==(DecodeKey const& other) const noexcept;
    };

    struct DecodeKeyHash
    {
        size_t operator()(DecodeKey const& key) const noexcept;
    };

    DecodeKey decodeKey(
        int32_t batchSize, int32_t maxContextLength, bool encoderActive, bool prefillActive) const noexcept;

    PhaseRuntimeCostTrackerConfig mConfig;
    PhaseGlobalCostModel mActions;
    PhaseContextualPdModel mContextualPd;
    PhaseContextualPdModel mContextualDp;
    PhaseContextualPdModel mContextualEp;
    PhaseContextualPdModel mContextualPe;
    PhaseContextualPdModel mContextualEd;
    PhaseContextualPdModel mContextualDe;
    PhaseContextualCompletionModel mCompletionPd;
    PhaseContextualCompletionModel mCompletionDp;
    PhaseContextualCompletionModel mCompletionEp;
    PhaseContextualCompletionModel mCompletionPe;
    PhaseContextualCompletionModel mCompletionEd;
    PhaseContextualCompletionModel mCompletionDe;
    PhaseContextualCompletionModel mCompletionPdPair;
    PhaseContextualCompletionModel mCompletionEpPair;
    PhaseContextualCompletionModel mCompletionEdPair;
    PhaseContextualCompletionCalibrator mCompletionPdCalibration;
    PhaseContextualCompletionCalibrator mCompletionEpCalibration;
    PhaseContextualCompletionCalibrator mCompletionEdCalibration;
    std::array<PhaseContextualCompletionTelemetry, 6U> mCompletionHierarchicalTelemetry{};
    std::array<CompletionAuthorityEvidenceWindow, 6U> mCompletionAuthorityEvidence{};
    std::unordered_map<DecodeKey, std::deque<float>, DecodeKeyHash> mDecodeComponents;
};

} // namespace trt_edgellm::rt
