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

#include "runtime/phase/cost/phaseCostOracle.h"

#include <gtest/gtest.h>

#include <atomic>
#include <filesystem>
#include <fstream>

namespace trt_edgellm::rt
{
namespace
{

PhaseDeploymentFingerprint fingerprint(std::string computeCapability = "8.6")
{
    PhaseDeploymentFingerprint result;
    result.modelHash = "model";
    result.onnxHash = "onnx";
    result.engineHash = "engine";
    result.externalWeightHash = "weights";
    result.precision = "fp16";
    result.kvDtype = "fp16";
    result.capability.maxPrefillBatchSize = 8;
    result.capability.maxDecodeBatchSize = 64;
    result.capability.maxEncoderBatchSize = 4;
    result.capability.prefillChunkTokens = 128;
    result.capability.maxKVCacheCapacity = 2048;
    result.gpu.computeCapability = std::move(computeCapability);
    result.gpu.smCount = 68;
    result.gpu.memoryBytes = 10U * 1024U * 1024U * 1024U;
    result.gpu.productName = "RTX 3080";
    result.gpuUuid = "provenance-only";
    result.software.tensorrtVersion = "11.0.0.114";
    result.software.cudaVersion = "13.3";
    result.software.pluginHash = "plugin";
    return result;
}

PhaseGlobalActionKey decodeKey(int32_t batchSize = 32)
{
    PhaseGlobalActionKey key;
    key.kind = PhaseGlobalActionKind::kDecode;
    key.primaryBatchSize = batchSize;
    key.primaryContextBucket = 512;
    return key;
}

PhaseCostBundle bundle(PhaseCostBundleSource source, float makespanMs)
{
    PhaseCostBundle result;
    result.bundleVersion = "test-v1";
    result.source = source;
    result.deployment = fingerprint();
    result.createdAtUnixNs = 123U;
    result.records.push_back({decodeKey(), {{10.0F, makespanMs}, {10.0F, makespanMs + 1.0F}}, 122U});
    return result;
}

std::filesystem::path temporaryDirectory()
{
    static std::atomic<uint64_t> counter{};
    return std::filesystem::temp_directory_path()
        / ("phase-cost-knowledge-test-" + std::to_string(phaseCostUnixTimeNs()) + "-"
            + std::to_string(counter.fetch_add(1U)));
}

TEST(PhaseCostKnowledgeTest, ClassifiesDeploymentCompatibilityWithoutGpuUuid)
{
    PhaseDeploymentFingerprint local = fingerprint();
    PhaseDeploymentFingerprint candidate = local;
    candidate.gpuUuid = "another-node";
    EXPECT_EQ(phaseCostCompatibility(local, candidate), PhaseCostCompatibility::kExact);

    candidate.engineHash = "another-engine-tactic";
    EXPECT_EQ(phaseCostCompatibility(local, candidate), PhaseCostCompatibility::kCompatible);

    candidate.gpu.computeCapability = "9.0";
    EXPECT_EQ(phaseCostCompatibility(local, candidate), PhaseCostCompatibility::kShapeOnly);

    candidate.capability.maxDecodeBatchSize = 32;
    EXPECT_EQ(phaseCostCompatibility(local, candidate), PhaseCostCompatibility::kIncompatible);

    candidate = local;
    candidate.capability.kvBytesPerToken = 4096U;
    local.capability.kvBytesPerToken = 2048U;
    EXPECT_EQ(phaseCostCompatibility(local, candidate), PhaseCostCompatibility::kIncompatible);

    candidate = local;
    local.capability.graphShapes = {decodeKey(32)};
    candidate.capability.graphShapes = {decodeKey(64)};
    EXPECT_EQ(phaseCostCompatibility(local, candidate), PhaseCostCompatibility::kIncompatible);
}

TEST(PhaseCostKnowledgeTest, BundleRoundTripPreservesRawObservations)
{
    std::filesystem::path const directory = temporaryDirectory();
    std::filesystem::path const path = directory / "bundle.json";
    PhaseCostBundle const expected = bundle(PhaseCostBundleSource::kBuild, 7.0F);
    phaseWriteCostBundleAtomic(expected, path);
    PhaseCostBundle const actual = phaseLoadCostBundle(path);

    ASSERT_EQ(actual.records.size(), 1U);
    ASSERT_EQ(actual.records.front().observations.size(), 2U);
    EXPECT_EQ(actual.bundleVersion, expected.bundleVersion);
    EXPECT_EQ(actual.source, PhaseCostBundleSource::kBuild);
    EXPECT_EQ(actual.records.front().key, decodeKey());
    EXPECT_FLOAT_EQ(actual.records.front().observations.front().makespanMs, 7.0F);
    std::filesystem::remove_all(directory);
}

TEST(PhaseCostKnowledgeTest, OracleExportsControlledObservationsAsBuildPrior)
{
    PhaseCostOracle oracle;
    oracle.observe(decodeKey(), {10.0F, 7.0F});
    PhaseCostBundle const calibrated
        = oracle.snapshot(PhaseCostBundleSource::kBuild, fingerprint(), "startup-calibration-test", 456U);

    EXPECT_EQ(calibrated.source, PhaseCostBundleSource::kBuild);
    EXPECT_EQ(calibrated.bundleVersion, "startup-calibration-test");
    EXPECT_EQ(calibrated.createdAtUnixNs, 456U);
    ASSERT_EQ(calibrated.records.size(), 1U);
    EXPECT_EQ(calibrated.records.front().key, decodeKey());
}

TEST(PhaseCostKnowledgeTest, OraclePrefersFleetUntilLocalEvidenceIsSufficient)
{
    PhaseCostOracleConfig config;
    config.sufficientLocalSamples = 3U;
    config.model.coldStartUncertaintyMs = 0.0F;
    config.anchor.enabled = false;
    PhaseCostOracle oracle(config);
    oracle.loadPrior(bundle(PhaseCostBundleSource::kBuild, 9.0F), PhaseCostCompatibility::kExact);
    oracle.loadPrior(bundle(PhaseCostBundleSource::kFleet, 7.0F), PhaseCostCompatibility::kExact);

    ASSERT_TRUE(oracle.estimate(decodeKey()).has_value());
    EXPECT_FLOAT_EQ(oracle.estimate(decodeKey())->makespanMedianMs, 7.0F);
    oracle.observe(decodeKey(), {10.0F, 5.0F});
    oracle.observe(decodeKey(), {10.0F, 5.0F});
    EXPECT_FLOAT_EQ(oracle.estimate(decodeKey())->makespanMedianMs, 7.0F);
    oracle.observe(decodeKey(), {10.0F, 5.0F});
    EXPECT_FLOAT_EQ(oracle.estimate(decodeKey())->makespanMedianMs, 5.0F);
}

TEST(PhaseCostKnowledgeTest, CompatiblePriorScalesCostAndWidensUncertainty)
{
    PhaseCostOracleConfig config;
    config.model.coldStartUncertaintyMs = 0.0F;
    config.compatibleUncertaintyMultiplier = 2.0F;
    PhaseCostOracle oracle(config);
    PhaseCostScale scale;
    scale.decode = 1.1F;
    oracle.loadPrior(bundle(PhaseCostBundleSource::kFleet, 10.0F), PhaseCostCompatibility::kCompatible, scale);

    std::optional<PhaseGlobalCostEstimate> const estimate = oracle.estimate(decodeKey());
    ASSERT_TRUE(estimate.has_value());
    EXPECT_FLOAT_EQ(estimate->makespanMedianMs, 11.0F);
    EXPECT_FLOAT_EQ(estimate->uncertaintyMs, 2.2F);
    EXPECT_FLOAT_EQ(estimate->makespanP95Ms, 13.2F);
}

TEST(PhaseCostKnowledgeTest, StartupAnchorsAdaptPriorWithoutAWorkloadMode)
{
    PhaseCostOracleConfig config;
    config.sufficientLocalSamples = 4U;
    config.model.coldStartUncertaintyMs = 0.0F;
    config.anchor.minimumSamplesPerPhase = 2U;
    PhaseCostOracle oracle(config);
    PhaseCostBundle prior = bundle(PhaseCostBundleSource::kBuild, 10.0F);
    prior.records.front().observations = {{10.0F, 10.0F}, {10.0F, 10.0F}};
    PhaseCostRecord larger{decodeKey(64), {{20.0F, 20.0F}, {20.0F, 20.0F}}, 122U};
    prior.records.push_back(larger);
    oracle.loadPrior(std::move(prior), PhaseCostCompatibility::kCompatible);
    oracle.setAnchorCollectionActive(true);

    oracle.observe(decodeKey(), {10.0F, 12.0F});
    EXPECT_FLOAT_EQ(oracle.anchorState().scale.decode, 1.0F);
    oracle.observe(decodeKey(), {10.0F, 14.0F});

    PhaseCostAnchorState const state = oracle.anchorState();
    EXPECT_EQ(state.decodeSamples, 2U);
    EXPECT_FLOAT_EQ(state.scale.decode, 1.3F);
    EXPECT_GT(state.decodeRelativeUncertainty, config.anchor.minimumRelativeUncertainty);
    std::optional<PhaseGlobalCostEstimate> const estimate = oracle.estimate(decodeKey(64));
    ASSERT_TRUE(estimate.has_value());
    EXPECT_FLOAT_EQ(estimate->makespanMedianMs, 26.0F);
    EXPECT_GT(estimate->uncertaintyMs, 0.0F);
}

TEST(PhaseCostKnowledgeTest, StartupAnchorsRejectImplausibleScaleAndResetCleanly)
{
    PhaseCostOracleConfig config;
    config.model.coldStartUncertaintyMs = 0.0F;
    config.anchor.minimumSamplesPerPhase = 1U;
    PhaseCostOracle oracle(config);
    PhaseCostBundle prior = bundle(PhaseCostBundleSource::kBuild, 10.0F);
    prior.records.front().observations = {{10.0F, 10.0F}};
    oracle.loadPrior(std::move(prior), PhaseCostCompatibility::kExact);
    oracle.setAnchorCollectionActive(true);

    oracle.observe(decodeKey(), {10.0F, 100.0F});
    EXPECT_EQ(oracle.anchorState().decodeSamples, 0U);
    oracle.observe(decodeKey(), {10.0F, 12.0F});
    EXPECT_FLOAT_EQ(oracle.anchorState().scale.decode, 1.2F);
    oracle.resetLocal();
    EXPECT_EQ(oracle.anchorState().decodeSamples, 0U);
    EXPECT_FLOAT_EQ(oracle.anchorState().scale.decode, 1.0F);
}

TEST(PhaseCostKnowledgeTest, ProductionObservationsDoNotRetunePortablePhaseScale)
{
    PhaseCostOracleConfig config;
    config.model.coldStartUncertaintyMs = 0.0F;
    config.anchor.minimumSamplesPerPhase = 1U;
    PhaseCostOracle oracle(config);
    PhaseCostBundle prior = bundle(PhaseCostBundleSource::kBuild, 10.0F);
    prior.records.front().observations = {{10.0F, 10.0F}};
    prior.records.push_back({decodeKey(64), {{20.0F, 20.0F}}, 122U});
    oracle.loadPrior(std::move(prior), PhaseCostCompatibility::kExact);

    oracle.observe(decodeKey(), {10.0F, 12.0F});
    EXPECT_EQ(oracle.anchorState().decodeSamples, 0U);
    std::optional<PhaseGlobalCostEstimate> const estimate = oracle.estimate(decodeKey(64));
    ASSERT_TRUE(estimate.has_value());
    EXPECT_FLOAT_EQ(estimate->makespanMedianMs, 20.0F);
}

TEST(PhaseCostKnowledgeTest, NodeJournalPersistsOutsideObservationHotPath)
{
    std::filesystem::path const directory = temporaryDirectory();
    PhaseNodeCostJournalConfig config;
    config.directory = directory;
    config.snapshotEveryObservations = 1U;
    config.flushInterval = std::chrono::hours(1);
    {
        auto journal = std::make_shared<PhaseNodeCostJournal>(config, fingerprint());
        PhaseCostOracle oracle;
        oracle.attachJournal(journal);
        oracle.observe(decodeKey(), {10.0F, 6.0F});
        journal->flush();
        EXPECT_TRUE(std::filesystem::is_regular_file(journal->journalPath()));
        EXPECT_TRUE(std::filesystem::is_regular_file(journal->snapshotPath()));
    }

    PhaseCostBundle restored = phaseLoadCostBundle(directory / "local_cost_snapshot.json");
    ASSERT_EQ(restored.records.size(), 1U);
    PhaseCostOracle oracle;
    oracle.restoreNode(restored);
    ASSERT_TRUE(oracle.estimate(decodeKey()).has_value());
    EXPECT_FLOAT_EQ(oracle.estimate(decodeKey())->makespanMedianMs, 6.0F);
    std::filesystem::remove_all(directory);
}

TEST(PhaseCostKnowledgeTest, CorruptBundleUsesExplicitSafeFailure)
{
    std::filesystem::path const directory = temporaryDirectory();
    std::filesystem::create_directories(directory);
    std::filesystem::path const path = directory / "corrupt.json";
    {
        std::ofstream stream(path);
        stream << "{not-json";
    }
    PhaseCostBundle ignored;
    std::string error;
    EXPECT_FALSE(phaseTryLoadCostBundle(path, ignored, error));
    EXPECT_FALSE(error.empty());
    std::filesystem::remove_all(directory);
}

} // namespace
} // namespace trt_edgellm::rt
