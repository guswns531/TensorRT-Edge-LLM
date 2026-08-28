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

#include "runtime/phase/policy/phaseGlobalCostModel.h"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trt_edgellm::rt
{

//! Engine profile and persistent-memory facts that are safe to share without timing calibration.
struct PhaseCostCapability
{
    int32_t maxPrefillBatchSize{};
    int32_t maxDecodeBatchSize{};
    int32_t maxEncoderBatchSize{};
    int32_t prefillChunkTokens{};
    int32_t maxKVCacheCapacity{};
    int32_t kvPageTokens{128};
    size_t kvBytesPerToken{};
    size_t visionOutputBytesPerToken{};
    std::vector<PhaseGlobalActionKey> graphShapes;
};

struct PhaseCostGpuSignature
{
    std::string computeCapability;
    int32_t smCount{};
    uint64_t memoryBytes{};
    std::string productName;
};

struct PhaseCostSoftwareSignature
{
    std::string tensorrtVersion;
    std::string cudaVersion;
    std::string driverVersion;
    std::string pluginHash;
};

//! Compatibility identity. GPU UUID is provenance, not part of compatibility.
struct PhaseDeploymentFingerprint
{
    std::string modelHash;
    std::string onnxHash;
    std::string engineHash;
    std::string externalWeightHash;
    std::string precision;
    std::string kvDtype;
    PhaseCostCapability capability;
    PhaseCostGpuSignature gpu;
    PhaseCostSoftwareSignature software;
    std::string gpuUuid;
};

enum class PhaseCostCompatibility
{
    kExact,
    kCompatible,
    kShapeOnly,
    kIncompatible,
};

char const* phaseCostCompatibilityName(PhaseCostCompatibility compatibility) noexcept;
PhaseCostCompatibility phaseCostCompatibility(
    PhaseDeploymentFingerprint const& local, PhaseDeploymentFingerprint const& candidate) noexcept;

enum class PhaseCostBundleSource
{
    kBuild,
    kFleet,
    kNode,
};

char const* phaseCostBundleSourceName(PhaseCostBundleSource source) noexcept;

struct PhaseCostPhaseDriftState
{
    bool localOnly{};
    size_t sampleCount{};
    float medianRatio{1.0F};
    uint64_t observedAtUnixNs{};
};

struct PhaseCostDriftState
{
    PhaseCostPhaseDriftState encoder;
    PhaseCostPhaseDriftState prefill;
    PhaseCostPhaseDriftState decode;
    PhaseCostPhaseDriftState overlap;
};

struct PhaseCostRecord
{
    PhaseGlobalActionKey key;
    std::vector<PhaseGlobalCostObservation> observations;
    uint64_t observedAtUnixNs{};
};

//! Versioned, portable action-cost data. Records carry bounded raw observations so quantiles remain mergeable.
struct PhaseCostBundle
{
    int32_t schemaVersion{1};
    std::string bundleVersion;
    PhaseCostBundleSource source{PhaseCostBundleSource::kBuild};
    PhaseDeploymentFingerprint deployment;
    uint64_t createdAtUnixNs{};
    std::vector<PhaseCostRecord> records;
    std::optional<PhaseCostDriftState> driftState;
};

PhaseCostBundle phaseLoadCostBundle(std::filesystem::path const& path);
bool phaseTryLoadCostBundle(std::filesystem::path const& path, PhaseCostBundle& bundle, std::string& error) noexcept;
void phaseWriteCostBundleAtomic(PhaseCostBundle const& bundle, std::filesystem::path const& path);

struct PhaseCostScale
{
    float encoder{1.0F};
    float prefill{1.0F};
    float decode{1.0F};
    float overlap{1.0F};
};

struct PhaseCostAnchorConfig
{
    bool enabled{true};
    size_t minimumSamplesPerPhase{2U};
    size_t maxSamplesPerPhase{16U};
    float minimumAcceptedScale{0.25F};
    float maximumAcceptedScale{4.0F};
    float minimumAppliedScale{0.5F};
    float maximumAppliedScale{2.0F};
    float minimumRelativeUncertainty{0.02F};
};

struct PhaseCostAnchorState
{
    PhaseCostScale scale;
    float encoderRelativeUncertainty{};
    float prefillRelativeUncertainty{};
    float decodeRelativeUncertainty{};
    float overlapRelativeUncertainty{};
    size_t encoderSamples{};
    size_t prefillSamples{};
    size_t decodeSamples{};
    size_t overlapSamples{};
};

struct PhaseCostDriftConfig
{
    bool enabled{true};
    size_t minimumSamplesPerPhase{8U};
    size_t maxSamplesPerPhase{16U};
    float enterRelativeError{0.20F};
    float exitRelativeError{0.10F};
    float minimumAcceptedRatio{0.25F};
    float maximumAcceptedRatio{4.0F};
    std::chrono::nanoseconds portablePriorTtl{std::chrono::hours{24 * 30}};
    std::chrono::nanoseconds nodeObservationTtl{std::chrono::hours{24 * 7}};
    std::chrono::nanoseconds restoredDriftStateTtl{std::chrono::hours{1}};
};

struct PhaseCostHealthState
{
    PhaseCostDriftState drift;
    bool buildPriorExpired{};
    bool fleetPriorExpired{};
    size_t staleNodeRecords{};
};

struct PhaseCostOracleConfig
{
    PhaseGlobalCostModelConfig model;
    PhaseCostAnchorConfig anchor;
    PhaseCostDriftConfig drift;
    size_t sufficientLocalSamples{4U};
    float compatibleUncertaintyMultiplier{2.0F};
    float shapeOnlyUncertaintyMultiplier{4.0F};
};

class PhaseNodeCostJournal;

//! Cost selection order: sufficient node-local observations, fleet prior, build prior, sparse local observation.
class PhaseCostOracle
{
public:
    explicit PhaseCostOracle(PhaseCostOracleConfig config = {});

    void observe(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation observation);
    std::optional<PhaseGlobalCostEstimate> estimate(PhaseGlobalActionKey const& key) const;
    std::optional<PhaseGlobalCostEstimate> estimateInterpolatedPrimaryBatch(PhaseGlobalActionKey const& key) const;
    PhaseGlobalOverlapCostDiagnostic overlapDiagnostic(PhaseGlobalActionKey const& key) const;
    bool overlapEligible(PhaseGlobalActionKey const& key) const;

    void loadPrior(PhaseCostBundle bundle, PhaseCostCompatibility compatibility, PhaseCostScale scale = {});
    void restoreNode(PhaseCostBundle const& bundle);
    PhaseCostBundle snapshot(PhaseCostBundleSource source, PhaseDeploymentFingerprint deployment,
        std::string bundleVersion, uint64_t createdAtUnixNs = 0U) const;
    PhaseCostBundle snapshotNode(
        PhaseDeploymentFingerprint deployment, std::string bundleVersion, uint64_t createdAtUnixNs = 0U) const;
    void attachJournal(std::shared_ptr<PhaseNodeCostJournal> journal);
    void resetLocal();
    void setCalibrationActive(bool active) noexcept;
    PhaseCostAnchorState anchorState() const;
    PhaseCostHealthState healthState() const;

private:
    struct PriorLayer
    {
        PhaseGlobalCostModel model;
        PhaseCostCompatibility compatibility{PhaseCostCompatibility::kIncompatible};
        PhaseCostScale scale;
        bool expired{};
    };

    std::optional<PhaseGlobalCostEstimate> estimatePrior(std::optional<PriorLayer> const& layer,
        PhaseGlobalActionKey const& key, bool interpolate, bool applyAnchor = true, bool ignoreDrift = false) const;
    static PhaseGlobalCostEstimate scaledEstimate(PhaseGlobalCostEstimate estimate, PhaseGlobalActionKey const& key,
        PhaseCostScale const& scale, float uncertaintyMultiplier, float relativeUncertainty = 0.0F) noexcept;
    void observeAnchor(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation const& observation);
    void observeDrift(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation const& observation);

    struct AnchorSamples
    {
        std::deque<float> ratios;
    };

    struct DriftSamples
    {
        std::deque<float> ratios;
    };

    PhaseCostOracleConfig mConfig;
    PhaseGlobalCostModel mLocal;
    std::vector<PhaseCostRecord> mLocalRecords;
    std::optional<PriorLayer> mBuild;
    std::optional<PriorLayer> mFleet;
    AnchorSamples mEncoderAnchors;
    AnchorSamples mPrefillAnchors;
    AnchorSamples mDecodeAnchors;
    AnchorSamples mOverlapAnchors;
    PhaseCostAnchorState mAnchorState;
    DriftSamples mEncoderDrift;
    DriftSamples mPrefillDrift;
    DriftSamples mDecodeDrift;
    DriftSamples mOverlapDrift;
    PhaseCostHealthState mHealthState;
    bool mCalibrationActive{};
    std::shared_ptr<PhaseNodeCostJournal> mJournal;
};

struct PhaseNodeCostJournalConfig
{
    std::filesystem::path directory;
    size_t maxPendingObservations{4096U};
    size_t maxSamplesPerKey{32U};
    size_t snapshotEveryObservations{128U};
    std::chrono::milliseconds flushInterval{1000};
};

//! Asynchronous node-local persistence. enqueue() performs no filesystem I/O.
class PhaseNodeCostJournal
{
public:
    PhaseNodeCostJournal(PhaseNodeCostJournalConfig config, PhaseDeploymentFingerprint deployment);
    ~PhaseNodeCostJournal() noexcept;

    PhaseNodeCostJournal(PhaseNodeCostJournal const&) = delete;
    PhaseNodeCostJournal& operator=(PhaseNodeCostJournal const&) = delete;

    void enqueue(PhaseGlobalActionKey key, PhaseGlobalCostObservation observation, PhaseCostDriftState driftState,
        uint64_t observedAtUnixNs = 0U);
    void flush();
    size_t droppedObservations() const noexcept;
    std::filesystem::path snapshotPath() const;
    std::filesystem::path journalPath() const;

private:
    class Impl;
    std::unique_ptr<Impl> mImpl;
};

uint64_t phaseCostUnixTimeNs() noexcept;

} // namespace trt_edgellm::rt
