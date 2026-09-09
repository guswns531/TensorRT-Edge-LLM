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

#include "runtime/phase/cost/phaseRuntimeCostTracker.h"

#include <gtest/gtest.h>

#include <limits>

namespace trt_edgellm::rt
{
namespace
{

TEST(PhaseRuntimeCostTrackerTest, MovesFromUnknownThroughWarmingToReady)
{
    PhaseRuntimeCostTrackerConfig config;
    config.actionMinimumSamples = 3U;
    PhaseRuntimeCostTracker tracker(config);
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kDecode, 4, 0, 1, 1, 0};

    EXPECT_EQ(tracker.confidence(key), PhaseRuntimeCostConfidence::kUnknown);
    tracker.observe(key, {8.0F, 3.0F});
    EXPECT_EQ(tracker.confidence(key), PhaseRuntimeCostConfidence::kWarming);
    EXPECT_FALSE(tracker.trustedEstimate(key).has_value());
    tracker.observe(key, {8.0F, 3.1F});
    tracker.observe(key, {8.0F, 2.9F});
    EXPECT_EQ(tracker.confidence(key), PhaseRuntimeCostConfidence::kReady);
    EXPECT_TRUE(tracker.trustedEstimate(key).has_value());
}

TEST(PhaseRuntimeCostTrackerTest, ReplacesOldActionObservationsBySampleCount)
{
    PhaseRuntimeCostTrackerConfig config;
    config.action.windowSize = 3U;
    config.actionMinimumSamples = 1U;
    config.action.coldStartUncertaintyMs = 0.0F;
    PhaseRuntimeCostTracker tracker(config);
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kPrefill, 1, 0, 128, 0, 0};

    tracker.observe(key, {10.0F, 10.0F});
    tracker.observe(key, {10.0F, 10.0F});
    tracker.observe(key, {2.0F, 2.0F});
    tracker.observe(key, {2.0F, 2.0F});

    std::optional<PhaseGlobalCostEstimate> const estimate = tracker.estimate(key);
    ASSERT_TRUE(estimate.has_value());
    EXPECT_EQ(estimate->sampleCount, 3U);
    EXPECT_FLOAT_EQ(estimate->makespanMedianMs, 2.0F);
}

TEST(PhaseRuntimeCostTrackerTest, KeepsDecodeContentionBucketsSeparate)
{
    PhaseRuntimeCostTrackerConfig config;
    config.decodeMinimumSamples = 2U;
    config.decodeWindowSize = 3U;
    PhaseRuntimeCostTracker tracker(config);

    tracker.observeDecode(8, 700, false, false, 4.0F);
    tracker.observeDecode(8, 700, false, false, 5.0F);
    tracker.observeDecode(8, 700, true, false, 9.0F);
    tracker.observeDecode(8, 700, true, false, 10.0F);

    EXPECT_FLOAT_EQ(*tracker.decodeP95(8, 700, false, false), 5.0F);
    EXPECT_FLOAT_EQ(*tracker.decodeP95(8, 700, true, false), 10.0F);
    EXPECT_FALSE(tracker.decodeP95(8, 700, false, true).has_value());
    EXPECT_EQ(tracker.decodeBucketCount(), 2U);
}

TEST(PhaseRuntimeCostTrackerTest, UsesNearestCoveringContendedDecodeBuckets)
{
    PhaseRuntimeCostTrackerConfig config;
    config.decodeMinimumSamples = 2U;
    config.decodeContextBucketTokens = 512;
    PhaseRuntimeCostTracker tracker(config);

    tracker.observeDecode(32, 1024, false, true, 7.0F);
    tracker.observeDecode(32, 1024, false, true, 8.0F);
    tracker.observeDecode(64, 512, false, true, 9.0F);
    tracker.observeDecode(64, 512, false, true, 10.0F);

    EXPECT_FLOAT_EQ(*tracker.decodeCoveringP95(16, 500, false, true), 10.0F);
    EXPECT_FLOAT_EQ(*tracker.decodeCoveringP95(32, 700, false, true), 8.0F);
    EXPECT_FALSE(tracker.decodeCoveringP95(65, 500, false, true).has_value());
    EXPECT_FALSE(tracker.decodeCoveringP95(16, 500, true, true).has_value());
}

TEST(PhaseRuntimeCostTrackerTest, ResetDropsAllProcessLocalMeasurements)
{
    PhaseRuntimeCostTrackerConfig config;
    config.actionMinimumSamples = 1U;
    config.decodeMinimumSamples = 1U;
    PhaseRuntimeCostTracker tracker(config);
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kEncoder, 1, 0, 0, 1, 0};
    tracker.observe(key, {5.0F, 5.0F});
    tracker.observeDecode(1, 128, false, false, 1.0F);

    tracker.reset();

    EXPECT_EQ(tracker.confidence(key), PhaseRuntimeCostConfidence::kUnknown);
    EXPECT_FALSE(tracker.decodeP95(1, 128, false, false).has_value());
    EXPECT_EQ(tracker.decodeBucketCount(), 0U);
}

TEST(PhaseRuntimeCostTrackerTest, ExecutionAndPolicyStateResetIndependently)
{
    PhaseRuntimeCostTrackerConfig config;
    config.policyMode = PhasePolicyMode::kContextualScalar;
    config.actionMinimumSamples = 1U;
    config.contextualPd.mode = PhaseContextualPdMode::kActive;
    config.contextualPd.minimumObservations = 1U;
    PhaseRuntimeCostTracker tracker(config);
    PhaseGlobalActionKey const key{PhaseGlobalActionKind::kPrefillDecode, 1, 1, 128, 1, 1};
    PhaseContextualPdFeatures const features = phaseContextualPdFeatures(
        {3000.0, 7000.0, 100000.0, 1, 1, 128, 1, 1, PhaseExecutionVariant::kEager, false, {}});
    tracker.observe(key, {10.0F, 8.0F});
    ASSERT_TRUE(tracker.observeContextualDirection(PhaseContextualPairDirection::kPrefillToDecode, features, 0.2));

    tracker.resetExecutionCostHistory();
    EXPECT_EQ(tracker.confidence(key), PhaseRuntimeCostConfidence::kUnknown);
    EXPECT_EQ(tracker.contextualDirectionTelemetry(PhaseContextualPairDirection::kPrefillToDecode).observations, 1U);

    tracker.observe(key, {10.0F, 8.0F});
    tracker.resetPolicyPosterior();
    EXPECT_EQ(tracker.confidence(key), PhaseRuntimeCostConfidence::kReady);
    EXPECT_EQ(tracker.contextualDirectionTelemetry(PhaseContextualPairDirection::kPrefillToDecode).observations, 0U);
}

TEST(PhaseContextualPdModelTest, BecomesReadyAndReducesUncertainty)
{
    PhaseContextualPdModelConfig config;
    config.mode = PhaseContextualPdMode::kActive;
    config.minimumObservations = 4U;
    PhaseContextualPdModel model(config);
    PhaseContextualPdFeatures const features = phaseContextualPdFeatures(
        {4000.0, 8000.0, 100000.0, 2, 32, 128, 1, 2, PhaseExecutionVariant::kEager, false, {}});

    PhaseContextualPdEstimate const cold = model.predict(features);
    for (size_t sample{}; sample < 8U; ++sample)
    {
        EXPECT_TRUE(model.observe(features, 0.2));
    }
    PhaseContextualPdEstimate const warm = model.predict(features);

    EXPECT_FALSE(cold.ready);
    EXPECT_TRUE(warm.ready);
    EXPECT_GT(warm.mean, 0.0);
    EXPECT_LT(warm.uncertainty, cold.uncertainty);
    EXPECT_EQ(model.telemetry().observations, 8U);
}

TEST(PhaseContextualPdModelTest, LearnsShapeDependentSignWithoutExactKeys)
{
    PhaseContextualPdModelConfig config;
    config.minimumObservations = 4U;
    config.confidenceBeta = 0.0;
    PhaseContextualPdModel model(config);
    PhaseContextualPdFeatures const small = phaseContextualPdFeatures(
        {3000.0, 8000.0, 100000.0, 1, 32, 128, 0, 2, PhaseExecutionVariant::kEager, false, {}});
    PhaseContextualPdFeatures const large = phaseContextualPdFeatures(
        {30000.0, 8000.0, 100000.0, 8, 32, 128, 0, 2, PhaseExecutionVariant::kEager, false, {}});

    for (size_t sample{}; sample < 32U; ++sample)
    {
        EXPECT_TRUE(model.observe(small, 0.2));
        EXPECT_TRUE(model.observe(large, -0.15));
    }

    EXPECT_GT(model.predict(small).mean, 0.0);
    EXPECT_LT(model.predict(large).mean, 0.0);
}

TEST(PhaseContextualPdModelTest, RejectsInvalidObservationAndResetDropsEvidence)
{
    PhaseContextualPdModel model;
    PhaseContextualPdFeatures const features = phaseContextualPdFeatures({4000.0, 8000.0, 100000.0, 2, 32, 128, 1, 2,
        PhaseExecutionVariant::kEager, true, PhaseGlobalResidualAnchor::kDecode});

    EXPECT_FALSE(model.observe(features, std::numeric_limits<double>::quiet_NaN()));
    EXPECT_TRUE(model.observe(features, 0.1));
    EXPECT_EQ(model.telemetry().observations, 1U);
    EXPECT_EQ(model.telemetry().rejectedObservations, 1U);
    model.reset();
    EXPECT_EQ(model.telemetry().observations, 0U);
    EXPECT_FALSE(model.predict(features).ready);
}

TEST(PhaseContextualPdModelTest, UsesPosteriorMeanOnlyForEligibleLateRecovery)
{
    PhaseContextualPdEstimate const estimate{0.2, 0.4, -0.1, 8U, true};
    std::vector<PhaseProtectedCompletion> const decodeLate{
        {100.0, 120.0, 0.0, PhaseProtectedKind::kDecode}, {200.0, 120.0, 0.0, PhaseProtectedKind::kPrefill}};
    std::vector<PhaseProtectedCompletion> const prefillLate{
        {200.0, 120.0, 0.0, PhaseProtectedKind::kDecode}, {100.0, 120.0, 0.0, PhaseProtectedKind::kPrefill}};

    EXPECT_DOUBLE_EQ(phaseContextualPdDecisionValue(estimate, false, false), estimate.lowerConfidenceBound);
    EXPECT_DOUBLE_EQ(phaseContextualPdDecisionValue(estimate, true, true), estimate.lowerConfidenceBound);
    EXPECT_DOUBLE_EQ(phaseContextualPdDecisionValue(estimate, true, false), estimate.mean);
    EXPECT_TRUE(phaseContextualPdDecodeDominatedRecovery(decodeLate, 0.0));
    EXPECT_FALSE(phaseContextualPdDecodeDominatedRecovery(prefillLate, 0.0));
    EXPECT_FALSE(phaseContextualPdProducerCriticalPath(false, false));
    EXPECT_TRUE(phaseContextualPdProducerCriticalPath(true, false));
    EXPECT_TRUE(phaseContextualPdProducerCriticalPath(false, true));
    EXPECT_TRUE(phaseContextualPdDrainsReadyPrefill(true, 128, 128));
    EXPECT_FALSE(phaseContextualPdDrainsReadyPrefill(false, 128, 128));
    EXPECT_FALSE(phaseContextualPdDrainsReadyPrefill(true, 256, 128));
    EXPECT_TRUE(phaseContextualPdControlsDecision(false));
    EXPECT_FALSE(phaseContextualPdControlsDecision(true));
    EXPECT_FALSE(phaseContextualPdMayPromoteOverlap(false, false));
    EXPECT_TRUE(phaseContextualPdMayPromoteOverlap(true, false));
    EXPECT_FALSE(phaseContextualPdMayPromoteOverlap(true, true));
}

TEST(PhaseContextualPairModelTest, ProjectsEncoderPairsIntoContinuousFeatures)
{
    PhaseContextualPdFeatures const encoderPrefill = phaseContextualPairFeatures(
        {3000.0, 8000.0, 100000.0, 1, 8, 8, 8, 128, 128, 1, 2, PhaseExecutionVariant::kEager});
    PhaseContextualPdFeatures const encoderDecode = phaseContextualPairFeatures(
        {3000.0, 8000.0, 100000.0, 1, 32, 8, 64, 0, 128, 1, 2, PhaseExecutionVariant::kEager});

    EXPECT_EQ(encoderPrefill.size(), kPHASE_CONTEXTUAL_PD_FEATURES);
    EXPECT_DOUBLE_EQ(encoderPrefill.front(), 1.0);
    EXPECT_NE(encoderPrefill, encoderDecode);
}

TEST(PhaseContextualPairModelTest, BatchFeaturesUseRuntimeCapacities)
{
    PhaseContextualPdInput halfFull{4000.0, 8000.0, 100000.0, 4, 32, 128, 1, 2, PhaseExecutionVariant::kEager};
    halfFull.prefillBatchCapacity = 8;
    halfFull.decodeBatchCapacity = 64;
    PhaseContextualPdInput full = halfFull;
    full.prefillBatchSize = 8;
    full.decodeBatchSize = 64;
    PhaseContextualPdInput widerEngine = full;
    widerEngine.prefillBatchCapacity = 16;
    widerEngine.decodeBatchCapacity = 128;

    PhaseContextualPdFeatures const halfFeatures = phaseContextualPdFeatures(halfFull);
    PhaseContextualPdFeatures const fullFeatures = phaseContextualPdFeatures(full);
    PhaseContextualPdFeatures const widerFeatures = phaseContextualPdFeatures(widerEngine);

    EXPECT_LT(halfFeatures[5], fullFeatures[5]);
    EXPECT_LT(halfFeatures[6], fullFeatures[6]);
    EXPECT_LT(widerFeatures[5], fullFeatures[5]);
    EXPECT_LT(widerFeatures[6], fullFeatures[6]);
}

TEST(PhaseContextualPairModelTest, KeepsEncoderPrefillAndDecodeEvidenceIndependent)
{
    PhaseRuntimeCostTrackerConfig config;
    config.policyMode = PhasePolicyMode::kContextualScalar;
    config.contextualEp.mode = PhaseContextualPdMode::kActive;
    config.contextualEp.minimumObservations = 2U;
    config.contextualEp.confidenceBeta = 0.0;
    config.contextualEd.mode = PhaseContextualPdMode::kActive;
    config.contextualEd.minimumObservations = 2U;
    config.contextualEd.confidenceBeta = 0.0;
    PhaseRuntimeCostTracker tracker(config);
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 8, 8, 8, 128, 128, 1, 2, PhaseExecutionVariant::kEager});

    for (size_t sample{}; sample < 8U; ++sample)
    {
        EXPECT_TRUE(tracker.observeContextualPair(PhaseContextualPairKind::kEncoderPrefill, features, 0.2));
        EXPECT_TRUE(tracker.observeContextualPair(PhaseContextualPairKind::kEncoderDecode, features, -0.2));
    }

    EXPECT_GT(tracker.predictContextualPair(PhaseContextualPairKind::kEncoderPrefill, features).mean, 0.0);
    EXPECT_LT(tracker.predictContextualPair(PhaseContextualPairKind::kEncoderDecode, features).mean, 0.0);
    EXPECT_EQ(tracker.contextualPairTelemetry(PhaseContextualPairKind::kPrefillDecode).observations, 0U);

    tracker.reset();
    EXPECT_EQ(tracker.contextualPairTelemetry(PhaseContextualPairKind::kEncoderPrefill).observations, 0U);
    EXPECT_EQ(tracker.contextualPairTelemetry(PhaseContextualPairKind::kEncoderDecode).observations, 0U);
}

TEST(PhaseContextualPairModelTest, KeepsOppositeDirectionsIndependent)
{
    PhaseRuntimeCostTrackerConfig config;
    config.policyMode = PhasePolicyMode::kContextualScalar;
    config.contextualPd.mode = PhaseContextualPdMode::kActive;
    config.contextualPd.minimumObservations = 2U;
    config.contextualPd.confidenceBeta = 0.0;
    PhaseRuntimeCostTracker tracker(config);
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 32, 8, 64, 128, 128, 1, 2, PhaseExecutionVariant::kEager});

    for (size_t sample{}; sample < 8U; ++sample)
    {
        EXPECT_TRUE(tracker.observeContextualDirection(PhaseContextualPairDirection::kPrefillToDecode, features, 0.2));
        EXPECT_TRUE(tracker.observeContextualDirection(PhaseContextualPairDirection::kDecodeToPrefill, features, -0.2));
    }

    EXPECT_GT(tracker.predictContextualDirection(PhaseContextualPairDirection::kPrefillToDecode, features).mean, 0.0);
    EXPECT_LT(tracker.predictContextualDirection(PhaseContextualPairDirection::kDecodeToPrefill, features).mean, 0.0);
    EXPECT_EQ(tracker.contextualPairTelemetry(PhaseContextualPairKind::kPrefillDecode).observations, 16U);
    EXPECT_STREQ(phaseContextualPairDirectionName(PhaseContextualPairDirection::kDecodeToPrefill), "decode_to_prefill");
}

TEST(PhaseContextualPairModelTest, CalibratesBeforeUpdatingAndCountsFalseSafe)
{
    PhaseContextualPdModelConfig config;
    config.minimumObservations = 1U;
    config.confidenceBeta = 0.0;
    PhaseContextualPdModel model(config);
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 32, 8, 64, 128, 128, 1, 2, PhaseExecutionVariant::kEager});

    EXPECT_TRUE(model.observe(features, 0.5));
    EXPECT_TRUE(model.observe(features, -0.5));

    PhaseContextualPdTelemetry const& telemetry = model.telemetry();
    EXPECT_EQ(telemetry.calibrationObservations, 2U);
    EXPECT_EQ(telemetry.readyCalibrationObservations, 1U);
    EXPECT_EQ(telemetry.predictedSafeObservations, 1U);
    EXPECT_EQ(telemetry.falseSafeObservations, 1U);
    EXPECT_GT(telemetry.absoluteErrorSum, 0.0);
    EXPECT_LT(telemetry.lastPredictionError, 0.0);
}

TEST(PhasePolicyModeTest, ParsesOnlyCanonicalProductionVariants)
{
    EXPECT_EQ(phasePolicyModeFromName("exact"), PhasePolicyMode::kExact);
    EXPECT_EQ(phasePolicyModeFromName("scalar"), PhasePolicyMode::kContextualScalar);
    EXPECT_EQ(phasePolicyModeFromName("scalar-transition"), PhasePolicyMode::kContextualScalarTransition);
    EXPECT_EQ(phasePolicyModeFromName("service-scaled-transition"), PhasePolicyMode::kServiceScaledTransition);
    EXPECT_FALSE(phasePolicyModeFromName("completion").has_value());
    EXPECT_FALSE(phasePolicyModeFromName("selective").has_value());

    EXPECT_FALSE(phasePolicyUsesContextualScalar(PhasePolicyMode::kExact));
    EXPECT_TRUE(phasePolicyUsesContextualScalar(PhasePolicyMode::kContextualScalar));
    EXPECT_TRUE(phasePolicyUsesContextualScalar(PhasePolicyMode::kContextualScalarTransition));
    EXPECT_TRUE(phasePolicyUsesContextualScalar(PhasePolicyMode::kServiceScaledTransition));
    EXPECT_FALSE(phasePolicyUsesTransition(PhasePolicyMode::kContextualScalar));
    EXPECT_TRUE(phasePolicyUsesTransition(PhasePolicyMode::kContextualScalarTransition));
    EXPECT_TRUE(phasePolicyUsesTransition(PhasePolicyMode::kServiceScaledTransition));
    EXPECT_FALSE(phasePolicyUsesServiceScale(PhasePolicyMode::kContextualScalarTransition));
    EXPECT_TRUE(phasePolicyUsesServiceScale(PhasePolicyMode::kServiceScaledTransition));
}

TEST(PhasePolicyModeTest, ExactModeCannotAccidentallyEnableContextualAuthority)
{
    PhaseRuntimeCostTrackerConfig config;
    config.policyMode = PhasePolicyMode::kExact;
    config.contextualPd.mode = PhaseContextualPdMode::kActive;
    config.contextualEp.mode = PhaseContextualPdMode::kActive;
    config.contextualEd.mode = PhaseContextualPdMode::kActive;

    PhaseRuntimeCostTracker tracker(config);

    EXPECT_EQ(tracker.contextualPdConfig().mode, PhaseContextualPdMode::kDisabled);
    EXPECT_EQ(
        tracker.contextualPairConfig(PhaseContextualPairKind::kEncoderPrefill).mode, PhaseContextualPdMode::kDisabled);
    EXPECT_EQ(
        tracker.contextualPairConfig(PhaseContextualPairKind::kEncoderDecode).mode, PhaseContextualPdMode::kDisabled);
}

} // namespace
} // namespace trt_edgellm::rt
