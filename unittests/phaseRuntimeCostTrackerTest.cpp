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

TEST(PhaseContextualPairModelTest, CompletionV2AddsResidualStateWithoutChangingAdvantageFeatures)
{
    PhaseContextualPairInput early{4000.0, 8000.0, 100000.0, 2, 32, 8, 64, 128, 128, 1, 2,
        PhaseExecutionVariant::kEager, true, PhaseGlobalResidualAnchor::kPrefill, 1000.0, 4000.0, 0.25,
        PhaseExecutionSet::kPrefill};
    PhaseContextualPairInput late = early;
    late.incumbentDispatchAgeUs = 3000.0;
    late.requestedStartSkewFraction = 0.75;

    EXPECT_EQ(phaseContextualPairFeatures(early), phaseContextualPairFeatures(late));
    EXPECT_NE(phaseContextualCompletionFeatures(early), phaseContextualCompletionFeatures(late));
    EXPECT_DOUBLE_EQ(phaseContextualCompletionFeatures(early)[11], 0.25);
    EXPECT_DOUBLE_EQ(phaseContextualCompletionFeatures(late)[11], 0.75);
    EXPECT_DOUBLE_EQ(phaseContextualCompletionFeatures(early)[14], 1.0);
}

TEST(PhaseContextualPairModelTest, KeepsEncoderPrefillAndDecodeEvidenceIndependent)
{
    PhaseRuntimeCostTrackerConfig config;
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
    config.contextualPd.mode = PhaseContextualPdMode::kShadow;
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

TEST(PhaseContextualCompletionModelTest, LearnsIncumbentAndNewcomerResidualsIndependently)
{
    PhaseContextualPdModelConfig config;
    config.minimumObservations = 2U;
    config.confidenceBeta = 1.96;
    PhaseContextualCompletionModel model(config);
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 32, 8, 64, 128, 128, 1, 2, PhaseExecutionVariant::kEager});

    for (size_t sample{}; sample < 16U; ++sample)
    {
        EXPECT_TRUE(model.observe(features, 4000.0, 8000.0, 5000.0, 6000.0, 20000.0));
    }
    PhaseContextualCompletionEstimate const prediction = model.predict(features, 4000.0, 8000.0);

    EXPECT_TRUE(prediction.ready);
    EXPECT_NEAR(prediction.incumbentMeanUs, 5000.0, 250.0);
    EXPECT_NEAR(prediction.newcomerMeanUs, 6000.0, 250.0);
    EXPECT_EQ(model.telemetry().observations, 16U);
    EXPECT_EQ(model.telemetry().falseSafeObservations, 0U);
}

TEST(PhaseContextualEffectModelTest, LearnsThreeDecisionRelevantEffectsIndependently)
{
    PhaseContextualPdModelConfig config;
    config.minimumObservations = 2U;
    config.confidenceBeta = 0.0;
    PhaseContextualEffectModel model(config);
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 32, 8, 64, 128, 128, 1, 2, PhaseExecutionVariant::kEager});

    for (size_t sample{}; sample < 16U; ++sample)
    {
        ASSERT_TRUE(model.observe(features, 0.25, 0.10, -0.20));
    }
    PhaseContextualEffectEstimate const estimate = model.predict(features);

    EXPECT_TRUE(estimate.ready());
    EXPECT_NEAR(estimate.compression.mean, 0.25, 0.02);
    EXPECT_NEAR(estimate.incumbentStretch.mean, 0.10, 0.02);
    EXPECT_NEAR(estimate.completionOrderMargin.mean, -0.20, 0.02);
    EXPECT_EQ(model.compressionTelemetry().observations, 16U);
    EXPECT_EQ(model.incumbentStretchTelemetry().observations, 16U);
    EXPECT_EQ(model.completionOrderTelemetry().observations, 16U);

    model.reset();
    EXPECT_EQ(model.compressionTelemetry().observations, 0U);
}

TEST(PhaseContextualEffectModelTest, RuntimeTrackerDerivesTargetsFromPhysicalCompletion)
{
    PhaseRuntimeCostTrackerConfig config;
    config.contextualPd.minimumObservations = 2U;
    config.contextualPd.confidenceBeta = 0.0;
    PhaseRuntimeCostTracker tracker(config);
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 32, 8, 64, 128, 128, 1, 2, PhaseExecutionVariant::kEager});
    PhaseContextualPairDirection const direction = PhaseContextualPairDirection::kPrefillToDecode;

    for (size_t sample{}; sample < 16U; ++sample)
    {
        ASSERT_TRUE(tracker.observeContextualCompletionDirection(direction, features, 4000.0, 8000.0, 5000.0, 6000.0));
    }
    PhaseContextualEffectEstimate const estimate = tracker.predictContextualEffectDirection(direction, features);

    EXPECT_TRUE(estimate.ready());
    EXPECT_NEAR(estimate.compression.mean, 0.5, 0.03);
    EXPECT_NEAR(estimate.incumbentStretch.mean, 0.25, 0.03);
    EXPECT_NEAR(estimate.completionOrderMargin.mean, 1.0 / 12.0, 0.02);
    EXPECT_EQ(tracker.contextualEffectDirectionModel(direction).compressionTelemetry().observations, 16U);
}

TEST(PhaseContextualCompletionModelTest, PairPosteriorWarmsColdReverseDirection)
{
    PhaseRuntimeCostTrackerConfig config;
    config.contextualPd.minimumObservations = 2U;
    config.completionDirectionPseudoObservations = 4.0;
    PhaseRuntimeCostTracker tracker(config);
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 32, 8, 64, 128, 128, 1, 2, PhaseExecutionVariant::kEager});

    for (size_t sample{}; sample < 16U; ++sample)
    {
        EXPECT_TRUE(tracker.observeContextualCompletionDirection(
            PhaseContextualPairDirection::kPrefillToDecode, features, 4000.0, 8000.0, 4500.0, 9000.0, 20000.0));
    }

    PhaseContextualCompletionEstimate const reverse = tracker.predictContextualCompletionDirection(
        PhaseContextualPairDirection::kDecodeToPrefill, features, 8000.0, 4000.0);

    EXPECT_TRUE(reverse.ready);
    EXPECT_EQ(reverse.pairObservations, 16U);
    EXPECT_EQ(reverse.directionObservations, 0U);
    EXPECT_DOUBLE_EQ(reverse.directionWeight, 0.0);
    EXPECT_NEAR(reverse.incumbentMeanUs, 9000.0, 300.0);
    EXPECT_NEAR(reverse.newcomerMeanUs, 4500.0, 300.0);
    EXPECT_EQ(
        tracker.contextualCompletionDirectionTelemetry(PhaseContextualPairDirection::kPrefillToDecode).observations,
        16U);
    EXPECT_EQ(
        tracker.contextualCompletionDirectionTelemetry(PhaseContextualPairDirection::kDecodeToPrefill).observations,
        0U);
    EXPECT_EQ(tracker.contextualCompletionPairTelemetry(PhaseContextualPairKind::kPrefillDecode).observations, 16U);
}

TEST(PhaseContextualCompletionModelTest, DirectionPosteriorGraduallyOverridesPairPosterior)
{
    PhaseContextualCompletionEstimate pair{5000.0, 1000.0, 7000.0, 800.0, 20U, true};
    PhaseContextualCompletionEstimate direction{9000.0, 1200.0, 6000.0, 900.0, 4U, true};

    PhaseContextualCompletionEstimate const result = phaseBlendContextualCompletionEstimates(pair, direction, 4.0);

    EXPECT_DOUBLE_EQ(result.directionWeight, 0.5);
    EXPECT_DOUBLE_EQ(result.incumbentMeanUs, 7000.0);
    EXPECT_DOUBLE_EQ(result.newcomerMeanUs, 6500.0);
    EXPECT_GT(result.incumbentUncertaintyUs, 2000.0);
    EXPECT_EQ(result.pairObservations, 20U);
    EXPECT_EQ(result.directionObservations, 4U);
    EXPECT_TRUE(result.ready);
}

TEST(PhaseContextualCompletionModelTest, CompletionDecisionRiskScalesWithProducerFormation)
{
    PhaseContextualCompletionEstimate const estimate{100.0, 40.0, 80.0, 20.0, 16U, true};
    PhaseContextualPdFeatures sparse{};
    PhaseContextualPdFeatures dense{};
    dense[4] = 1.0;

    EXPECT_DOUBLE_EQ(phaseContextualCompletionDecisionMakespanUs(estimate, sparse), 100.0);
    EXPECT_DOUBLE_EQ(phaseContextualCompletionDecisionMakespanUs(estimate, dense), 140.0);
}

TEST(PhaseContextualCompletionModelTest, AuthorityBlendPreservesEndpointsAndComponents)
{
    PhaseContextualCompletionEstimate const estimate{80.0, 20.0, 120.0, 40.0, 8U, true};
    PhaseContextualPdFeatures features{};
    features[4] = 0.5;

    EXPECT_DOUBLE_EQ(phaseBlendContextualCompletionDecisionMakespanUs(200.0, estimate, features, 0.0), 200.0);
    EXPECT_DOUBLE_EQ(phaseBlendContextualCompletionDecisionMakespanUs(200.0, estimate, features, 1.0), 140.0);
    EXPECT_DOUBLE_EQ(phaseBlendContextualCompletionDecisionMakespanUs(200.0, estimate, features, 0.5), 170.0);

    double completionUs{200.0};
    double uncertaintyUs{10.0};
    phaseBlendContextualCompletionComponent(completionUs, uncertaintyUs, 100.0, 30.0, 0.25);
    EXPECT_DOUBLE_EQ(completionUs, 175.0);
    EXPECT_DOUBLE_EQ(uncertaintyUs, 15.0);
}

TEST(PhaseContextualCompletionModelTest, HierarchicalTelemetryRejectsInvalidLabelsAtModelBoundary)
{
    PhaseRuntimeCostTracker tracker;
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 32, 8, 64, 128, 128, 1, 2, PhaseExecutionVariant::kEager});

    EXPECT_FALSE(tracker.observeContextualCompletionDirection(
        PhaseContextualPairDirection::kPrefillToDecode, features, 0.0, 8000.0, 4500.0, 9000.0, 20000.0));

    EXPECT_EQ(
        tracker.contextualCompletionDirectionTelemetry(PhaseContextualPairDirection::kPrefillToDecode).observations,
        0U);
    EXPECT_EQ(tracker.contextualCompletionPairTelemetry(PhaseContextualPairKind::kPrefillDecode).observations, 0U);
}

TEST(PhaseContextualCompletionCalibrationTest, UsesOnlyPriorPairFamilyResiduals)
{
    PhaseContextualCompletionCalibrationConfig config;
    config.enabled = true;
    config.targetCoverage = 0.75;
    config.minimumObservations = 2U;
    config.windowSize = 4U;
    PhaseContextualCompletionCalibrator calibrator(config);
    PhaseContextualCompletionEstimate const raw{100.0, 10.0, 200.0, 20.0, 8U, true};

    EXPECT_FALSE(calibrator.estimate().ready);
    EXPECT_TRUE(calibrator.observe(raw, 1.0, 120.0, 220.0));
    EXPECT_TRUE(calibrator.observe(raw, 1.0, 110.0, 260.0));

    PhaseContextualCompletionCalibrationEstimate const estimate = calibrator.estimate();
    EXPECT_TRUE(estimate.ready);
    EXPECT_EQ(estimate.observations, 2U);
    EXPECT_DOUBLE_EQ(estimate.scale, 3.0);
    PhaseContextualCompletionEstimate const calibrated = calibrator.apply(raw);
    EXPECT_DOUBLE_EQ(calibrated.incumbentUncertaintyUs, 30.0);
    EXPECT_DOUBLE_EQ(calibrated.newcomerUncertaintyUs, 60.0);
    EXPECT_TRUE(calibrated.uncertaintyCalibrated);
}

TEST(PhaseContextualCompletionCalibrationTest, SharesScaleAcrossOppositeDirectionsButNotPairs)
{
    PhaseRuntimeCostTrackerConfig config;
    config.contextualPd.minimumObservations = 1U;
    config.contextualEp.minimumObservations = 1U;
    config.completionCalibration.enabled = true;
    config.completionCalibration.minimumObservations = 1U;
    config.completionCalibration.authorityMinimumObservations = 1U;
    config.completionCalibration.windowSize = 4U;
    PhaseRuntimeCostTracker tracker(config);
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 32, 8, 64, 128, 128, 1, 2, PhaseExecutionVariant::kEager});

    EXPECT_TRUE(tracker.observeContextualCompletionDirection(
        PhaseContextualPairDirection::kPrefillToDecode, features, 4000.0, 8000.0, 4500.0, 9000.0));
    EXPECT_TRUE(tracker.observeContextualCompletionDirection(
        PhaseContextualPairDirection::kPrefillToDecode, features, 4000.0, 8000.0, 7000.0, 12000.0));

    PhaseContextualCompletionCalibrationEstimate const pd
        = tracker.contextualCompletionCalibration(PhaseContextualPairKind::kPrefillDecode);
    EXPECT_TRUE(pd.ready);
    EXPECT_EQ(pd.observations, 1U);
    EXPECT_EQ(tracker.contextualCompletionCalibration(PhaseContextualPairKind::kEncoderPrefill).observations, 0U);
    PhaseContextualCompletionEstimate const reverse = tracker.predictContextualCompletionDirection(
        PhaseContextualPairDirection::kDecodeToPrefill, features, 8000.0, 4000.0);
    EXPECT_TRUE(reverse.uncertaintyCalibrated);
    EXPECT_DOUBLE_EQ(reverse.uncertaintyScale, pd.scale);
}

TEST(PhaseContextualCompletionCalibrationTest, SeparatesShadowCalibrationFromSchedulingAuthority)
{
    PhaseRuntimeCostTrackerConfig shadowConfig;
    shadowConfig.completionCalibration.enabled = true;
    PhaseRuntimeCostTracker shadow(shadowConfig);
    EXPECT_FALSE(shadow.contextualCompletionAuthorityEnabled());

    PhaseRuntimeCostTrackerConfig activeConfig;
    activeConfig.completionCalibration.enabled = true;
    activeConfig.completionCalibration.active = true;
    PhaseRuntimeCostTracker active(activeConfig);
    EXPECT_TRUE(active.contextualCompletionAuthorityEnabled());
}

TEST(PhaseContextualCompletionCalibrationTest, AuthorityRequiresHeldOutDirectionalCoverage)
{
    PhaseRuntimeCostTrackerConfig config;
    config.contextualPd.minimumObservations = 1U;
    config.completionCalibration.enabled = true;
    config.completionCalibration.active = true;
    config.completionCalibration.minimumObservations = 1U;
    config.completionCalibration.authorityMinimumObservations = 1U;
    config.completionCalibration.windowSize = 4U;
    PhaseRuntimeCostTracker tracker(config);
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 32, 8, 64, 128, 128, 1, 2, PhaseExecutionVariant::kEager});
    PhaseContextualPairDirection const direction = PhaseContextualPairDirection::kPrefillToDecode;

    PhaseContextualCompletionEstimate estimate
        = tracker.predictContextualCompletionDirection(direction, features, 4000.0, 8000.0);
    EXPECT_FALSE(tracker.contextualCompletionAuthorityReady(direction, estimate));
    for (size_t sample{}; sample < 3U; ++sample)
    {
        ASSERT_TRUE(tracker.observeContextualCompletionDirection(direction, features, 4000.0, 8000.0, 4000.0, 8000.0));
    }
    estimate = tracker.predictContextualCompletionDirection(direction, features, 4000.0, 8000.0);
    EXPECT_TRUE(estimate.ready);
    EXPECT_TRUE(estimate.uncertaintyCalibrated);
    EXPECT_TRUE(tracker.contextualCompletionAuthorityReady(direction, estimate));
    EXPECT_EQ(tracker.contextualCompletionAuthorityEvidence(direction).observations, 1U);
    EXPECT_EQ(tracker.contextualCompletionAuthorityEvidence(direction).promotions, 1U);
    EXPECT_TRUE(tracker.contextualCompletionAuthorityEvidence(direction).validated);
    for (size_t sample{}; sample < 8U; ++sample)
    {
        ASSERT_TRUE(tracker.observeContextualCompletionDirection(direction, features, 4000.0, 8000.0, 4000.0, 8000.0));
    }
    PhaseContextualCompletionAuthorityEvidence const evidence
        = tracker.contextualCompletionAuthorityEvidence(direction);
    EXPECT_EQ(evidence.observations, config.completionCalibration.windowSize);
    EXPECT_EQ(evidence.incumbentIntervalCovered, evidence.observations);
    EXPECT_EQ(evidence.newcomerIntervalCovered, evidence.observations);
    tracker.resetCompletionAuthorityEvidence();
    PhaseContextualCompletionEstimate const retained
        = tracker.predictContextualCompletionDirection(direction, features, 4000.0, 8000.0);
    EXPECT_TRUE(retained.ready);
    EXPECT_TRUE(retained.uncertaintyCalibrated);
    EXPECT_EQ(tracker.contextualCompletionAuthorityEvidence(direction).observations, 0U);
    EXPECT_FALSE(tracker.contextualCompletionAuthorityReady(direction, retained));
    PhaseContextualCompletionEstimate const reverse = tracker.predictContextualCompletionDirection(
        PhaseContextualPairDirection::kDecodeToPrefill, features, 8000.0, 4000.0);
    EXPECT_FALSE(tracker.contextualCompletionAuthorityReady(PhaseContextualPairDirection::kDecodeToPrefill, reverse));
}

TEST(PhaseContextualCompletionCalibrationTest, ReportsBoundedChronologicalCalibrationStages)
{
    EXPECT_STREQ(phaseContextualCompletionCalibrationStageName(
                     PhaseContextualCompletionCalibrationStage::kUncertaintyCalibration),
        "uncertainty_calibration");
    PhaseRuntimeCostTrackerConfig config;
    config.contextualEp.minimumObservations = 2U;
    config.completionCalibration.enabled = true;
    config.completionCalibration.active = true;
    config.completionCalibration.minimumObservations = 2U;
    config.completionCalibration.authorityMinimumObservations = 2U;
    config.completionCalibration.windowSize = 4U;
    PhaseRuntimeCostTracker tracker(config);
    PhaseContextualPdFeatures const features = phaseContextualPairFeatures(
        {4000.0, 8000.0, 100000.0, 2, 8, 8, 64, 128, 128, 1, 2, PhaseExecutionVariant::kEager});
    PhaseContextualPairDirection const direction = PhaseContextualPairDirection::kEncoderToPrefill;

    auto progress = tracker.contextualCompletionCalibrationProgress(direction);
    EXPECT_EQ(progress.stage, PhaseContextualCompletionCalibrationStage::kPosteriorFit);
    EXPECT_FALSE(tracker.contextualCompletionAuthorityEvidenceComplete(direction));

    for (size_t sample{}; sample < 2U; ++sample)
    {
        ASSERT_TRUE(tracker.observeContextualCompletionDirection(direction, features, 4000.0, 8000.0, 4000.0, 8000.0));
    }
    progress = tracker.contextualCompletionCalibrationProgress(direction);
    EXPECT_EQ(progress.stage, PhaseContextualCompletionCalibrationStage::kUncertaintyCalibration);
    EXPECT_EQ(progress.posteriorObservations, 2U);

    for (size_t sample{}; sample < 2U; ++sample)
    {
        ASSERT_TRUE(tracker.observeContextualCompletionDirection(direction, features, 4000.0, 8000.0, 4000.0, 8000.0));
    }
    progress = tracker.contextualCompletionCalibrationProgress(direction);
    EXPECT_EQ(progress.stage, PhaseContextualCompletionCalibrationStage::kAuthorityValidation);
    EXPECT_EQ(progress.uncertaintyObservations, 2U);

    for (size_t sample{}; sample < 2U; ++sample)
    {
        ASSERT_TRUE(tracker.observeContextualCompletionDirection(direction, features, 4000.0, 8000.0, 4000.0, 8000.0));
    }
    progress = tracker.contextualCompletionCalibrationProgress(direction);
    EXPECT_EQ(progress.stage, PhaseContextualCompletionCalibrationStage::kComplete);
    EXPECT_EQ(progress.authorityObservations, 2U);
    EXPECT_TRUE(tracker.contextualCompletionAuthorityEvidenceComplete(direction));
    EXPECT_TRUE(tracker.contextualCompletionAuthorityEvidenceReady(direction));
}

TEST(PhaseContextualCompletionCalibrationTest, SupportsPolicyOnlyAuthorityAblations)
{
    rt::PhaseRuntimeCostTrackerConfig config;
    config.completionCalibration.enabled = true;
    config.completionCalibration.active = true;
    config.completionCalibration.authorityUsesUncertainty = false;
    config.completionCalibration.authorityPredictsIncumbent = false;
    rt::PhaseRuntimeCostTracker tracker(config);

    rt::PhaseContextualCompletionEstimate estimate;
    estimate.incumbentMeanUs = 100.0;
    estimate.incumbentUncertaintyUs = 25.0;
    estimate.newcomerMeanUs = 80.0;
    estimate.newcomerUncertaintyUs = 20.0;
    rt::PhaseContextualCompletionEstimate const authority
        = tracker.contextualCompletionAuthorityEstimate(rt::PhaseContextualPairDirection::kPrefillToDecode, estimate);

    EXPECT_TRUE(tracker.contextualCompletionAuthorityEnabled());
    EXPECT_FALSE(tracker.contextualCompletionAuthorityPredictsIncumbent());
    EXPECT_DOUBLE_EQ(authority.incumbentMeanUs, 100.0);
    EXPECT_DOUBLE_EQ(authority.newcomerMeanUs, 80.0);
    EXPECT_DOUBLE_EQ(authority.incumbentUncertaintyUs, 0.0);
    EXPECT_DOUBLE_EQ(authority.newcomerUncertaintyUs, 0.0);
}

TEST(PhaseContextualCompletionCalibrationTest, ResidualFeatureAblationProjectsToTheSamePosterior)
{
    rt::PhaseRuntimeCostTrackerConfig config;
    config.completionCalibration.useResidualFeatures = false;
    rt::PhaseRuntimeCostTracker tracker(config);
    rt::PhaseContextualPdFeatures coLaunch{};
    rt::PhaseContextualPdFeatures residual{};
    coLaunch.fill(0.25);
    residual = coLaunch;
    for (size_t index{9U}; index < residual.size(); ++index)
    {
        residual[index] = 1.0;
    }

    rt::PhaseContextualCompletionEstimate const coLaunchPrediction = tracker.predictContextualCompletionDirection(
        rt::PhaseContextualPairDirection::kPrefillToDecode, coLaunch, 100.0, 80.0);
    rt::PhaseContextualCompletionEstimate const residualPrediction = tracker.predictContextualCompletionDirection(
        rt::PhaseContextualPairDirection::kPrefillToDecode, residual, 100.0, 80.0);

    EXPECT_DOUBLE_EQ(coLaunchPrediction.incumbentMeanUs, residualPrediction.incumbentMeanUs);
    EXPECT_DOUBLE_EQ(coLaunchPrediction.newcomerMeanUs, residualPrediction.newcomerMeanUs);
    EXPECT_DOUBLE_EQ(coLaunchPrediction.incumbentUncertaintyUs, residualPrediction.incumbentUncertaintyUs);
    EXPECT_DOUBLE_EQ(coLaunchPrediction.newcomerUncertaintyUs, residualPrediction.newcomerUncertaintyUs);
}

} // namespace
} // namespace trt_edgellm::rt
