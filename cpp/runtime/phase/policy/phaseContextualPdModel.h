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

#include "runtime/phase/policy/phaseDeadline.h"
#include "runtime/phase/policy/phaseGlobalCostModel.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace trt_edgellm::rt
{

enum class PhaseContextualPdMode
{
    kDisabled,
    kShadow,
    kActive,
};

//! Independent contextual value heads. Exact CUDA cost keys remain phase
//! specific; this enum selects only the low-dimensional policy evidence.
enum class PhaseContextualPairKind
{
    kPrefillDecode,
    kEncoderPrefill,
    kEncoderDecode,
};

//! Ordered pair identity for one contextual posterior. Direction is part of
//! the model identity rather than an exact-key bucket: adding D to an active P
//! and adding P to an active D have different interference and completion
//! vectors even when their final outstanding set is identical.
enum class PhaseContextualPairDirection
{
    kPrefillToDecode,
    kDecodeToPrefill,
    kEncoderToPrefill,
    kPrefillToEncoder,
    kEncoderToDecode,
    kDecodeToEncoder,
};

char const* phaseContextualPairDirectionName(PhaseContextualPairDirection direction) noexcept;
PhaseContextualPairKind phaseContextualPairKind(PhaseContextualPairDirection direction) noexcept;
PhaseContextualPairDirection phaseContextualPairDirection(
    PhaseGlobalActionKind kind, PhaseGlobalResidualAnchor residualAnchor) noexcept;

constexpr size_t kPHASE_CONTEXTUAL_PD_FEATURES{16U};
using PhaseContextualPdFeatures = std::array<double, kPHASE_CONTEXTUAL_PD_FEATURES>;

struct PhaseContextualPairInput
{
    double primaryUs{};
    double secondaryUs{};
    double minimumSlackUs{};
    int32_t primaryBatchSize{};
    int32_t secondaryBatchSize{};
    int32_t primaryBatchCapacity{8};
    int32_t secondaryBatchCapacity{64};
    int32_t workSize{};
    int32_t workQuantum{128};
    int32_t primaryContextBucket{};
    int32_t secondaryContextBucket{};
    PhaseExecutionVariant executionVariant{PhaseExecutionVariant::kEager};
    bool residualAugmentation{};
    PhaseGlobalResidualAnchor residualAnchor{PhaseGlobalResidualAnchor::kNone};
};

//! Project any two-phase action into the common continuous policy space. Each
//! action family owns an independent posterior, so E+P, E+D, and P+D evidence
//! cannot contaminate one another.
PhaseContextualPdFeatures phaseContextualPairFeatures(PhaseContextualPairInput const& input) noexcept;

struct PhaseContextualPdInput
{
    double prefillUs{};
    double decodeUs{};
    double minimumSlackUs{};
    int32_t prefillBatchSize{};
    int32_t decodeBatchSize{};
    int32_t chunkLength{};
    int32_t prefillContextBucket{};
    int32_t decodeContextBucket{};
    PhaseExecutionVariant executionVariant{PhaseExecutionVariant::kEager};
    bool residualAugmentation{};
    PhaseGlobalResidualAnchor residualAnchor{PhaseGlobalResidualAnchor::kNone};
};

//! Project an exact P+D action into a bounded, model- and workload-label-free
//! decision representation. The execution cost tracker retains exact keys;
//! only policy evidence is shared through this low-dimensional projection.
PhaseContextualPdFeatures phaseContextualPdFeatures(PhaseContextualPdInput const& input) noexcept;

struct PhaseContextualPdModelConfig
{
    PhaseContextualPdMode mode{PhaseContextualPdMode::kDisabled};
    size_t minimumObservations{4U};
    double confidenceBeta{0.5};
    double initialCovariance{4.0};
    double initialResidualVariance{0.04};
    double forgettingFactor{1.0};
    double rewardClip{1.0};
};

struct PhaseContextualPdEstimate
{
    double mean{};
    double uncertainty{};
    double lowerConfidenceBound{};
    size_t observations{};
    bool ready{};
};

//! Convert a normalized contextual advantage into a conservative decision
//! makespan. The estimate changes efficiency ranking only; callers must retain
//! exact/robust execution costs for SLO and memory feasibility.
double phaseContextualDecisionMakespanUs(double referenceWorkUs, double conservativeAdvantage) noexcept;

//! Select the contextual advantage used by policy. Posterior-mean recovery is
//! permitted only when every action is already late and no upstream producer
//! is on the first-token critical path; otherwise retain the conservative LCB.
double phaseContextualPdDecisionValue(
    PhaseContextualPdEstimate const& estimate, bool decodeOnlyRecovery, bool producerCriticalPath) noexcept;

//! Mean-based recovery is safe only when the robust D violation exceeds the
//! largest P/E first-token violation. This compares current deadline state;
//! it does not classify the workload.
bool phaseContextualPdDecodeDominatedRecovery(
    std::vector<PhaseProtectedCompletion> const& completions, double deadlineGuardUs) noexcept;

//! Preserve producer ownership after E completion: an external P remains on
//! the E->P first-token path until that P action consumes its payload.
bool phaseContextualPdProducerCriticalPath(bool producerOutstanding, bool externalPrefillLineage) noexcept;

//! Return whether the current P action completes every currently ready P row.
//! This is a bounded current-state transition, not a future-arrival forecast.
bool phaseContextualPdDrainsReadyPrefill(bool allRowsFinal, int64_t remainingTokens, int32_t usefulTokens) noexcept;

//! The P+D-only model does not own a decision that also requires reasoning
//! about an upstream producer phase absent from its feature representation.
bool phaseContextualPdControlsDecision(bool producerCriticalPath) noexcept;

//! Return whether the local P+D value model may promote a serial global
//! decision to overlap. The local head may always veto an overlap with
//! negative evidence, but it must not create deadline-safe work that the
//! E/P/D-aware global selector rejected.
bool phaseContextualPdMayPromoteOverlap(bool deadlineRecovery, bool producerCriticalPath) noexcept;

struct PhaseContextualPdTelemetry
{
    size_t predictions{};
    size_t observations{};
    size_t rejectedObservations{};
    size_t positiveSelections{};
    size_t negativeSelections{};
    size_t explorations{};
    //! Prediction quality is evaluated before the corresponding observation
    //! updates the posterior. This avoids reporting training-after-label
    //! accuracy as shadow calibration.
    size_t calibrationObservations{};
    size_t readyCalibrationObservations{};
    size_t confidenceIntervalCovered{};
    size_t readyConfidenceIntervalCovered{};
    size_t predictedSafeObservations{};
    size_t falseSafeObservations{};
    double absoluteErrorSum{};
    double squaredErrorSum{};
    double readyAbsoluteErrorSum{};
    double readySquaredErrorSum{};
    double lastReward{};
    double lastMean{};
    double lastUncertainty{};
    double lastLowerConfidenceBound{};
    double lastPredictionError{};
};

//! A fixed-size recursive least-squares action-value model for one pair-action
//! family. The runtime owns independent instances for P+D, E+P, and E+D. It is
//! process-local, allocation-free on the hot path, and intentionally does not
//! persist or classify workload traces.
class PhaseContextualPdModel
{
public:
    explicit PhaseContextualPdModel(PhaseContextualPdModelConfig config = {});

    PhaseContextualPdEstimate predict(PhaseContextualPdFeatures const& features);
    bool observe(PhaseContextualPdFeatures const& features, double normalizedAdvantage, double weight = 1.0);
    void recordSelection(bool overlap, bool exploration) noexcept;
    void reset() noexcept;

    PhaseContextualPdModelConfig const& config() const noexcept;
    PhaseContextualPdTelemetry const& telemetry() const noexcept;

private:
    PhaseContextualPdModelConfig mConfig;
    PhaseContextualPdFeatures mTheta{};
    std::array<double, kPHASE_CONTEXTUAL_PD_FEATURES * kPHASE_CONTEXTUAL_PD_FEATURES> mCovariance{};
    double mResidualVariance{};
    PhaseContextualPdTelemetry mTelemetry;
};

//! Prediction of the two completion boundaries created by adding a newcomer
//! context to an incumbent context. Times are relative to the newcomer launch
//! boundary used by the M4 projector.
struct PhaseContextualCompletionEstimate
{
    double incumbentMeanUs{};
    double incumbentUncertaintyUs{};
    double newcomerMeanUs{};
    double newcomerUncertaintyUs{};
    size_t observations{};
    bool ready{};
    //! Pair-common and ordered-direction evidence used by hierarchical
    //! shrinkage. These remain zero for a standalone completion model.
    size_t pairObservations{};
    size_t directionObservations{};
    double directionWeight{};
    //! Pair-family conformal multiplier learned only from chronological,
    //! pre-update completion residuals. A value of one preserves the raw RLS
    //! uncertainty.
    double uncertaintyScale{1.0};
    size_t uncertaintyCalibrationObservations{};
    bool uncertaintyCalibrated{};
};

//! Configuration for bounded online conformal calibration. The same
//! configuration is used by every pair family; no direction, shape, trace, or
//! workload label enters the calibration policy.
struct PhaseContextualCompletionCalibrationConfig
{
    bool enabled{};
    //! Permit a ready, calibrated completion vector to replace the legacy
    //! protected-completion estimate. Calibration and authority are separate
    //! so shadow collection cannot silently change scheduling decisions.
    bool active{};
    double targetCoverage{0.95};
    size_t minimumObservations{16U};
    size_t windowSize{128U};
    double minimumScale{1.0};
    double maximumScale{8.0};
};

struct PhaseContextualCompletionCalibrationEstimate
{
    double scale{1.0};
    size_t observations{};
    bool ready{};
};

//! Pair-family online conformal uncertainty calibrator. Each observation is
//! the maximum standardized error of the two completion components, so one
//! scale protects the whole completion vector. Prediction always precedes
//! insertion of the corresponding score.
class PhaseContextualCompletionCalibrator
{
public:
    explicit PhaseContextualCompletionCalibrator(PhaseContextualCompletionCalibrationConfig config = {});

    PhaseContextualCompletionCalibrationEstimate estimate() const;
    PhaseContextualCompletionEstimate apply(PhaseContextualCompletionEstimate prediction) const;
    bool observe(PhaseContextualCompletionEstimate const& rawPrediction, double confidenceBeta,
        double incumbentCompletionUs, double newcomerCompletionUs);
    void reset() noexcept;

private:
    PhaseContextualCompletionCalibrationConfig mConfig;
    std::vector<double> mScores;
};

//! Shrink an ordered-direction posterior toward its canonical pair-common
//! posterior. The disagreement term prevents a newly diverging direction
//! from becoming overconfident while it acquires its own observations.
PhaseContextualCompletionEstimate phaseBlendContextualCompletionEstimates(PhaseContextualCompletionEstimate const& pair,
    PhaseContextualCompletionEstimate const& direction, double directionPseudoObservations) noexcept;

struct PhaseContextualCompletionTelemetry
{
    size_t predictions{};
    size_t observations{};
    size_t readyCalibrationObservations{};
    size_t incumbentIntervalCovered{};
    size_t newcomerIntervalCovered{};
    size_t readyIncumbentIntervalCovered{};
    size_t readyNewcomerIntervalCovered{};
    size_t conformalCalibrationObservations{};
    size_t conformalIncumbentIntervalCovered{};
    size_t conformalNewcomerIntervalCovered{};
    size_t conformalPredictedSafeObservations{};
    size_t conformalFalseSafeObservations{};
    size_t predictedSafeObservations{};
    size_t falseSafeObservations{};
    double incumbentAbsoluteErrorUs{};
    double incumbentSquaredErrorUs{};
    double newcomerAbsoluteErrorUs{};
    double newcomerSquaredErrorUs{};
    double readyIncumbentAbsoluteErrorUs{};
    double readyIncumbentSquaredErrorUs{};
    double readyNewcomerAbsoluteErrorUs{};
    double readyNewcomerSquaredErrorUs{};
};

//! Two-output residual RLS completion predictor. The deterministic isolated
//! completion is the baseline and each head learns a normalized residual.
//! This keeps cold predictions physically meaningful while preserving a
//! compact continuous policy representation.
class PhaseContextualCompletionModel
{
public:
    explicit PhaseContextualCompletionModel(PhaseContextualPdModelConfig config = {});

    PhaseContextualCompletionEstimate predict(
        PhaseContextualPdFeatures const& features, double incumbentReferenceUs, double newcomerReferenceUs);
    bool observe(PhaseContextualPdFeatures const& features, double incumbentReferenceUs, double newcomerReferenceUs,
        double incumbentCompletionUs, double newcomerCompletionUs,
        double minimumSlackUs = std::numeric_limits<double>::infinity());
    void reset() noexcept;

    PhaseContextualCompletionTelemetry const& telemetry() const noexcept;

private:
    PhaseContextualCompletionEstimate estimate(PhaseContextualPdFeatures const& features, double incumbentReferenceUs,
        double newcomerReferenceUs, bool recordPrediction);

    PhaseContextualPdModelConfig mConfig;
    PhaseContextualPdModel mIncumbent;
    PhaseContextualPdModel mNewcomer;
    PhaseContextualCompletionTelemetry mTelemetry;
};

} // namespace trt_edgellm::rt
