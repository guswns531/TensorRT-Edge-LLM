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

#include "common/checkMacros.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>

namespace trt_edgellm::rt
{
namespace
{

using Json = nlohmann::json;

bool sameKey(PhaseGlobalActionKey const& left, PhaseGlobalActionKey const& right) noexcept
{
    return left == right;
}

bool nonEmptyMismatch(std::string const& left, std::string const& right) noexcept
{
    return !left.empty() && !right.empty() && left != right;
}

bool sameCapability(PhaseCostCapability const& left, PhaseCostCapability const& right) noexcept
{
    auto const sameOrUnknown
        = [](int32_t first, int32_t second) { return first == 0 || second == 0 || first == second; };
    auto const sameSizeOrUnknown
        = [](size_t first, size_t second) { return first == 0U || second == 0U || first == second; };
    auto const sameGraphShapes = [&] {
        if (left.graphShapes.empty() || right.graphShapes.empty())
        {
            return true;
        }
        if (left.graphShapes.size() != right.graphShapes.size())
        {
            return false;
        }
        return std::all_of(left.graphShapes.begin(), left.graphShapes.end(), [&](PhaseGlobalActionKey const& key) {
            return std::find(right.graphShapes.begin(), right.graphShapes.end(), key) != right.graphShapes.end();
        });
    };
    return sameOrUnknown(left.maxPrefillBatchSize, right.maxPrefillBatchSize)
        && sameOrUnknown(left.maxDecodeBatchSize, right.maxDecodeBatchSize)
        && sameOrUnknown(left.maxEncoderBatchSize, right.maxEncoderBatchSize)
        && sameOrUnknown(left.prefillChunkTokens, right.prefillChunkTokens)
        && sameOrUnknown(left.maxKVCacheCapacity, right.maxKVCacheCapacity)
        && sameOrUnknown(left.kvPageTokens, right.kvPageTokens)
        && sameSizeOrUnknown(left.kvBytesPerToken, right.kvBytesPerToken)
        && sameSizeOrUnknown(left.visionOutputBytesPerToken, right.visionOutputBytesPerToken) && sameGraphShapes();
}

float actionScale(PhaseGlobalActionKind kind, PhaseCostScale const& scale) noexcept
{
    switch (kind)
    {
    case PhaseGlobalActionKind::kEncoder: return scale.encoder;
    case PhaseGlobalActionKind::kPrefill: return scale.prefill;
    case PhaseGlobalActionKind::kDecode: return scale.decode;
    case PhaseGlobalActionKind::kEncoderPrefill:
    case PhaseGlobalActionKind::kEncoderDecode:
    case PhaseGlobalActionKind::kPrefillDecode: return scale.overlap;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kWait: return 1.0F;
    }
    return 1.0F;
}

float actionRelativeUncertainty(PhaseGlobalActionKind kind, PhaseCostAnchorState const& state) noexcept
{
    switch (kind)
    {
    case PhaseGlobalActionKind::kEncoder: return state.encoderRelativeUncertainty;
    case PhaseGlobalActionKind::kPrefill: return state.prefillRelativeUncertainty;
    case PhaseGlobalActionKind::kDecode: return state.decodeRelativeUncertainty;
    case PhaseGlobalActionKind::kEncoderPrefill:
    case PhaseGlobalActionKind::kEncoderDecode:
    case PhaseGlobalActionKind::kPrefillDecode: return state.overlapRelativeUncertainty;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kWait: return 0.0F;
    }
    return 0.0F;
}

float median(std::deque<float> const& values)
{
    ELLM_CHECK(!values.empty(), "Cannot compute an empty phase anchor median");
    std::vector<float> ordered(values.begin(), values.end());
    std::sort(ordered.begin(), ordered.end());
    size_t const middle = ordered.size() / 2U;
    return ordered.size() % 2U == 0U ? (ordered[middle - 1U] + ordered[middle]) * 0.5F : ordered[middle];
}

float medianAbsoluteDeviation(std::deque<float> const& values, float center)
{
    std::deque<float> deviations;
    for (float const value : values)
    {
        deviations.push_back(std::abs(value - center));
    }
    return median(deviations);
}

PhaseGlobalActionKind parseAction(std::string const& value)
{
    if (value == "encoder")
    {
        return PhaseGlobalActionKind::kEncoder;
    }
    if (value == "prefill")
    {
        return PhaseGlobalActionKind::kPrefill;
    }
    if (value == "decode")
    {
        return PhaseGlobalActionKind::kDecode;
    }
    if (value == "encoder_prefill")
    {
        return PhaseGlobalActionKind::kEncoderPrefill;
    }
    if (value == "encoder_decode")
    {
        return PhaseGlobalActionKind::kEncoderDecode;
    }
    if (value == "prefill_decode")
    {
        return PhaseGlobalActionKind::kPrefillDecode;
    }
    if (value == "wait")
    {
        return PhaseGlobalActionKind::kWait;
    }
    ELLM_CHECK(false, "Unknown phase cost action: " + value);
    return PhaseGlobalActionKind::kNone;
}

PhaseExecutionVariant parseVariant(std::string const& value)
{
    if (value == "eager")
    {
        return PhaseExecutionVariant::kEager;
    }
    if (value == "primary_graph")
    {
        return PhaseExecutionVariant::kPrimaryGraph;
    }
    if (value == "secondary_graph")
    {
        return PhaseExecutionVariant::kSecondaryGraph;
    }
    if (value == "both_graph")
    {
        return PhaseExecutionVariant::kBothGraph;
    }
    ELLM_CHECK(false, "Unknown phase cost execution variant: " + value);
    return PhaseExecutionVariant::kEager;
}

PhaseCostBundleSource parseSource(std::string const& value)
{
    if (value == "build")
    {
        return PhaseCostBundleSource::kBuild;
    }
    if (value == "fleet")
    {
        return PhaseCostBundleSource::kFleet;
    }
    if (value == "node")
    {
        return PhaseCostBundleSource::kNode;
    }
    ELLM_CHECK(false, "Unknown phase cost bundle source: " + value);
    return PhaseCostBundleSource::kBuild;
}

Json keyJson(PhaseGlobalActionKey const& key)
{
    return {{"action", phaseGlobalActionKindName(key.kind)}, {"primary_batch_size", key.primaryBatchSize},
        {"secondary_batch_size", key.secondaryBatchSize}, {"chunk_length", key.chunkLength},
        {"primary_context_bucket", key.primaryContextBucket}, {"secondary_context_bucket", key.secondaryContextBucket},
        {"execution_variant", phaseExecutionVariantName(key.executionVariant)},
        {"residual_augmentation", key.residualAugmentation}};
}

PhaseGlobalActionKey parseKey(Json const& value)
{
    PhaseGlobalActionKey key;
    key.kind = parseAction(value.at("action").get<std::string>());
    key.primaryBatchSize = value.value("primary_batch_size", 0);
    key.secondaryBatchSize = value.value("secondary_batch_size", 0);
    key.chunkLength = value.value("chunk_length", 0);
    key.primaryContextBucket = value.value("primary_context_bucket", 0);
    key.secondaryContextBucket = value.value("secondary_context_bucket", 0);
    key.executionVariant = parseVariant(value.value("execution_variant", std::string{"eager"}));
    key.residualAugmentation = value.value("residual_augmentation", false);
    return key;
}

Json capabilityJson(PhaseCostCapability const& capability)
{
    Json graphs = Json::array();
    for (PhaseGlobalActionKey const& key : capability.graphShapes)
    {
        graphs.push_back(keyJson(key));
    }
    return {{"max_prefill_batch_size", capability.maxPrefillBatchSize},
        {"max_decode_batch_size", capability.maxDecodeBatchSize},
        {"max_encoder_batch_size", capability.maxEncoderBatchSize},
        {"prefill_chunk_tokens", capability.prefillChunkTokens},
        {"max_kv_cache_capacity", capability.maxKVCacheCapacity}, {"kv_page_tokens", capability.kvPageTokens},
        {"kv_bytes_per_token", capability.kvBytesPerToken},
        {"vision_output_bytes_per_token", capability.visionOutputBytesPerToken}, {"graph_shapes", graphs}};
}

PhaseCostCapability parseCapability(Json const& value)
{
    PhaseCostCapability capability;
    capability.maxPrefillBatchSize = value.value("max_prefill_batch_size", 0);
    capability.maxDecodeBatchSize = value.value("max_decode_batch_size", 0);
    capability.maxEncoderBatchSize = value.value("max_encoder_batch_size", 0);
    capability.prefillChunkTokens = value.value("prefill_chunk_tokens", 0);
    capability.maxKVCacheCapacity = value.value("max_kv_cache_capacity", 0);
    capability.kvPageTokens = value.value("kv_page_tokens", 128);
    capability.kvBytesPerToken = value.value("kv_bytes_per_token", size_t{});
    capability.visionOutputBytesPerToken = value.value("vision_output_bytes_per_token", size_t{});
    for (Json const& shape : value.value("graph_shapes", Json::array()))
    {
        capability.graphShapes.push_back(parseKey(shape));
    }
    return capability;
}

Json deploymentJson(PhaseDeploymentFingerprint const& deployment)
{
    return {{"model_hash", deployment.modelHash}, {"onnx_hash", deployment.onnxHash},
        {"engine_hash", deployment.engineHash}, {"external_weight_hash", deployment.externalWeightHash},
        {"precision", deployment.precision}, {"kv_dtype", deployment.kvDtype},
        {"capability", capabilityJson(deployment.capability)},
        {"gpu",
            {{"compute_capability", deployment.gpu.computeCapability}, {"sm_count", deployment.gpu.smCount},
                {"memory_bytes", deployment.gpu.memoryBytes}, {"product_name", deployment.gpu.productName},
                {"uuid", deployment.gpuUuid}}},
        {"software",
            {{"tensorrt", deployment.software.tensorrtVersion}, {"cuda", deployment.software.cudaVersion},
                {"driver", deployment.software.driverVersion}, {"plugin_hash", deployment.software.pluginHash}}}};
}

PhaseDeploymentFingerprint parseDeployment(Json const& value)
{
    PhaseDeploymentFingerprint result;
    result.modelHash = value.value("model_hash", std::string{});
    result.onnxHash = value.value("onnx_hash", std::string{});
    result.engineHash = value.value("engine_hash", std::string{});
    result.externalWeightHash = value.value("external_weight_hash", std::string{});
    result.precision = value.value("precision", std::string{});
    result.kvDtype = value.value("kv_dtype", std::string{});
    result.capability = parseCapability(value.at("capability"));
    Json const& gpu = value.at("gpu");
    result.gpu.computeCapability = gpu.value("compute_capability", std::string{});
    result.gpu.smCount = gpu.value("sm_count", 0);
    result.gpu.memoryBytes = gpu.value("memory_bytes", uint64_t{});
    result.gpu.productName = gpu.value("product_name", std::string{});
    result.gpuUuid = gpu.value("uuid", std::string{});
    Json const& software = value.at("software");
    result.software.tensorrtVersion = software.value("tensorrt", std::string{});
    result.software.cudaVersion = software.value("cuda", std::string{});
    result.software.driverVersion = software.value("driver", std::string{});
    result.software.pluginHash = software.value("plugin_hash", std::string{});
    return result;
}

Json bundleJson(PhaseCostBundle const& bundle)
{
    Json records = Json::array();
    for (PhaseCostRecord const& record : bundle.records)
    {
        Json observations = Json::array();
        for (PhaseGlobalCostObservation const& observation : record.observations)
        {
            observations.push_back(
                {{"reference_work_ms", observation.referenceWorkMs}, {"makespan_ms", observation.makespanMs}});
        }
        records.push_back({{"key", keyJson(record.key)}, {"observed_at_unix_ns", record.observedAtUnixNs},
            {"observations", observations}});
    }
    return {{"schema_version", bundle.schemaVersion}, {"bundle_version", bundle.bundleVersion},
        {"source", phaseCostBundleSourceName(bundle.source)}, {"created_at_unix_ns", bundle.createdAtUnixNs},
        {"deployment", deploymentJson(bundle.deployment)}, {"records", records}};
}

PhaseCostRecord* findRecord(std::vector<PhaseCostRecord>& records, PhaseGlobalActionKey const& key)
{
    auto const found = std::find_if(
        records.begin(), records.end(), [&](PhaseCostRecord const& record) { return sameKey(record.key, key); });
    return found == records.end() ? nullptr : &*found;
}

void mergeObservation(std::vector<PhaseCostRecord>& records, PhaseGlobalActionKey key,
    PhaseGlobalCostObservation observation, uint64_t observedAtUnixNs, size_t maxSamples)
{
    key = phaseGlobalCanonicalOverlapCostKey(key);
    PhaseCostRecord* record = findRecord(records, key);
    if (record == nullptr)
    {
        records.push_back({key, {}, observedAtUnixNs});
        record = &records.back();
    }
    record->observedAtUnixNs = std::max(record->observedAtUnixNs, observedAtUnixNs);
    record->observations.push_back(observation);
    if (record->observations.size() > maxSamples)
    {
        size_t const excess = record->observations.size() - maxSamples;
        record->observations.erase(record->observations.begin(), record->observations.begin() + excess);
    }
}

PhaseNodeCostJournalConfig validateJournalConfig(PhaseNodeCostJournalConfig config)
{
    ELLM_CHECK(!config.directory.empty(), "Node cost journal directory is empty");
    ELLM_CHECK(config.maxPendingObservations > 0U, "Node cost pending bound must be positive");
    ELLM_CHECK(config.maxSamplesPerKey > 0U, "Node cost sample bound must be positive");
    ELLM_CHECK(config.snapshotEveryObservations > 0U, "Node cost snapshot interval must be positive");
    return config;
}

} // namespace

char const* phaseCostCompatibilityName(PhaseCostCompatibility compatibility) noexcept
{
    switch (compatibility)
    {
    case PhaseCostCompatibility::kExact: return "exact";
    case PhaseCostCompatibility::kCompatible: return "compatible";
    case PhaseCostCompatibility::kShapeOnly: return "shape_only";
    case PhaseCostCompatibility::kIncompatible: return "incompatible";
    }
    return "incompatible";
}

PhaseCostCompatibility phaseCostCompatibility(
    PhaseDeploymentFingerprint const& local, PhaseDeploymentFingerprint const& candidate) noexcept
{
    if (nonEmptyMismatch(local.modelHash, candidate.modelHash) || nonEmptyMismatch(local.onnxHash, candidate.onnxHash)
        || nonEmptyMismatch(local.precision, candidate.precision) || nonEmptyMismatch(local.kvDtype, candidate.kvDtype)
        || !sameCapability(local.capability, candidate.capability))
    {
        return PhaseCostCompatibility::kIncompatible;
    }
    bool const exactEngine = !local.engineHash.empty() && local.engineHash == candidate.engineHash
        && !local.software.pluginHash.empty() && local.software.pluginHash == candidate.software.pluginHash;
    bool const exactSoftware = local.software.tensorrtVersion == candidate.software.tensorrtVersion
        && local.software.cudaVersion == candidate.software.cudaVersion;
    bool const exactGpu = local.gpu.computeCapability == candidate.gpu.computeCapability
        && local.gpu.smCount == candidate.gpu.smCount && local.gpu.memoryBytes == candidate.gpu.memoryBytes;
    if (exactEngine && exactSoftware && exactGpu)
    {
        return PhaseCostCompatibility::kExact;
    }
    if (!local.gpu.computeCapability.empty() && local.gpu.computeCapability == candidate.gpu.computeCapability)
    {
        return PhaseCostCompatibility::kCompatible;
    }
    return PhaseCostCompatibility::kShapeOnly;
}

char const* phaseCostBundleSourceName(PhaseCostBundleSource source) noexcept
{
    switch (source)
    {
    case PhaseCostBundleSource::kBuild: return "build";
    case PhaseCostBundleSource::kFleet: return "fleet";
    case PhaseCostBundleSource::kNode: return "node";
    }
    return "build";
}

PhaseCostBundle phaseLoadCostBundle(std::filesystem::path const& path)
{
    std::ifstream stream(path);
    ELLM_CHECK(stream.good(), "Failed to open phase cost bundle: " + path.string());
    Json const root = Json::parse(stream);
    PhaseCostBundle result;
    result.schemaVersion = root.at("schema_version").get<int32_t>();
    ELLM_CHECK(result.schemaVersion == 1, "Unsupported phase cost bundle schema");
    result.bundleVersion = root.value("bundle_version", std::string{});
    result.source = parseSource(root.at("source").get<std::string>());
    result.createdAtUnixNs = root.value("created_at_unix_ns", uint64_t{});
    result.deployment = parseDeployment(root.at("deployment"));
    for (Json const& value : root.at("records"))
    {
        PhaseCostRecord record;
        record.key = parseKey(value.at("key"));
        record.observedAtUnixNs = value.value("observed_at_unix_ns", uint64_t{});
        for (Json const& observation : value.at("observations"))
        {
            record.observations.push_back(
                {observation.at("reference_work_ms").get<float>(), observation.at("makespan_ms").get<float>()});
        }
        ELLM_CHECK(!record.observations.empty(), "Phase cost record has no observations");
        result.records.push_back(std::move(record));
    }
    return result;
}

bool phaseTryLoadCostBundle(std::filesystem::path const& path, PhaseCostBundle& bundle, std::string& error) noexcept
{
    try
    {
        bundle = phaseLoadCostBundle(path);
        error.clear();
        return true;
    }
    catch (std::exception const& exception)
    {
        error = exception.what();
        return false;
    }
}

void phaseWriteCostBundleAtomic(PhaseCostBundle const& bundle, std::filesystem::path const& path)
{
    ELLM_CHECK(!path.empty(), "Phase cost bundle output path is empty");
    if (!path.parent_path().empty())
    {
        std::filesystem::create_directories(path.parent_path());
    }
    std::filesystem::path temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::trunc);
        ELLM_CHECK(stream.good(), "Failed to open temporary phase cost bundle: " + temporary.string());
        stream << bundleJson(bundle).dump(2) << '\n';
        ELLM_CHECK(stream.good(), "Failed to write temporary phase cost bundle: " + temporary.string());
    }
    std::filesystem::rename(temporary, path);
}

PhaseCostOracle::PhaseCostOracle(PhaseCostOracleConfig config)
    : mConfig(config)
    , mLocal(config.model)
{
    ELLM_CHECK(mConfig.sufficientLocalSamples > 0U, "Sufficient local cost sample count must be positive");
    ELLM_CHECK(
        mConfig.compatibleUncertaintyMultiplier >= 1.0F, "Compatible cost uncertainty multiplier must be at least one");
    ELLM_CHECK(mConfig.shapeOnlyUncertaintyMultiplier >= mConfig.compatibleUncertaintyMultiplier,
        "Shape-only cost uncertainty must be no smaller than compatible uncertainty");
    ELLM_CHECK(mConfig.anchor.minimumSamplesPerPhase > 0U, "Phase anchor minimum sample count must be positive");
    ELLM_CHECK(mConfig.anchor.maxSamplesPerPhase >= mConfig.anchor.minimumSamplesPerPhase,
        "Phase anchor sample window must cover the minimum sample count");
    ELLM_CHECK(mConfig.anchor.minimumAcceptedScale > 0.0F
            && mConfig.anchor.maximumAcceptedScale >= mConfig.anchor.minimumAcceptedScale,
        "Phase anchor accepted scale range is invalid");
    ELLM_CHECK(mConfig.anchor.minimumAppliedScale > 0.0F
            && mConfig.anchor.maximumAppliedScale >= mConfig.anchor.minimumAppliedScale,
        "Phase anchor applied scale range is invalid");
    ELLM_CHECK(mConfig.anchor.minimumRelativeUncertainty >= 0.0F,
        "Phase anchor minimum relative uncertainty must be non-negative");
}

void PhaseCostOracle::observe(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation observation)
{
    observeAnchor(key, observation);
    mLocal.observe(key, observation);
    mergeObservation(mLocalRecords, key, observation, phaseCostUnixTimeNs(), mConfig.model.windowSize);
    if (mJournal != nullptr)
    {
        mJournal->enqueue(key, observation);
    }
}

std::optional<PhaseGlobalCostEstimate> PhaseCostOracle::estimate(PhaseGlobalActionKey const& key) const
{
    std::optional<PhaseGlobalCostEstimate> const local = mLocal.estimate(key);
    if (local.has_value() && local->sampleCount >= mConfig.sufficientLocalSamples)
    {
        return local;
    }
    if (std::optional<PhaseGlobalCostEstimate> const fleet = estimatePrior(mFleet, key, false))
    {
        return fleet;
    }
    if (std::optional<PhaseGlobalCostEstimate> const build = estimatePrior(mBuild, key, false))
    {
        return build;
    }
    return local;
}

std::optional<PhaseGlobalCostEstimate> PhaseCostOracle::estimateInterpolatedPrimaryBatch(
    PhaseGlobalActionKey const& key) const
{
    std::optional<PhaseGlobalCostEstimate> const local = mLocal.estimateInterpolatedPrimaryBatch(key);
    if (local.has_value() && local->sampleCount >= mConfig.sufficientLocalSamples)
    {
        return local;
    }
    if (std::optional<PhaseGlobalCostEstimate> const fleet = estimatePrior(mFleet, key, true))
    {
        return fleet;
    }
    if (std::optional<PhaseGlobalCostEstimate> const build = estimatePrior(mBuild, key, true))
    {
        return build;
    }
    return local;
}

PhaseGlobalOverlapCostDiagnostic PhaseCostOracle::overlapDiagnostic(PhaseGlobalActionKey const& key) const
{
    std::optional<PhaseGlobalCostEstimate> const estimateValue = estimate(key);
    if (!estimateValue.has_value())
    {
        return {};
    }
    float const robustMakespan = estimateValue->makespanMedianMs + estimateValue->uncertaintyMs;
    float const compression = estimateValue->referenceWorkMedianMs / std::max(robustMakespan, 1.0e-6F);
    if (estimateValue->sampleCount < mConfig.model.overlapMinSamples)
    {
        return {PhaseGlobalOverlapCostStatus::kInsufficientSamples, estimateValue->sampleCount, compression};
    }
    PhaseGlobalOverlapCostStatus const status = compression >= 1.0F + mConfig.model.minimumOverlapGainRatio
        ? PhaseGlobalOverlapCostStatus::kEligible
        : PhaseGlobalOverlapCostStatus::kUnprofitable;
    return {status, estimateValue->sampleCount, compression};
}

bool PhaseCostOracle::overlapEligible(PhaseGlobalActionKey const& key) const
{
    bool const overlap = key.kind == PhaseGlobalActionKind::kEncoderPrefill
        || key.kind == PhaseGlobalActionKind::kEncoderDecode || key.kind == PhaseGlobalActionKind::kPrefillDecode;
    return !overlap || overlapDiagnostic(key).status == PhaseGlobalOverlapCostStatus::kEligible;
}

void PhaseCostOracle::loadPrior(PhaseCostBundle bundle, PhaseCostCompatibility compatibility, PhaseCostScale scale)
{
    ELLM_CHECK(compatibility != PhaseCostCompatibility::kIncompatible, "Cannot load an incompatible phase cost prior");
    PriorLayer layer{PhaseGlobalCostModel(mConfig.model), compatibility, scale};
    for (PhaseCostRecord const& record : bundle.records)
    {
        for (PhaseGlobalCostObservation const& observation : record.observations)
        {
            layer.model.observe(record.key, observation);
        }
    }
    if (bundle.source == PhaseCostBundleSource::kFleet)
    {
        mFleet = std::move(layer);
    }
    else
    {
        mBuild = std::move(layer);
    }
}

void PhaseCostOracle::restoreNode(PhaseCostBundle const& bundle)
{
    ELLM_CHECK(bundle.source == PhaseCostBundleSource::kNode, "Only node cost bundles can restore local observations");
    for (PhaseCostRecord const& record : bundle.records)
    {
        for (PhaseGlobalCostObservation const& observation : record.observations)
        {
            mLocal.observe(record.key, observation);
            mergeObservation(mLocalRecords, record.key, observation, record.observedAtUnixNs, mConfig.model.windowSize);
        }
    }
}

PhaseCostBundle PhaseCostOracle::snapshot(PhaseCostBundleSource source, PhaseDeploymentFingerprint deployment,
    std::string bundleVersion, uint64_t createdAtUnixNs) const
{
    PhaseCostBundle result;
    result.bundleVersion = std::move(bundleVersion);
    result.source = source;
    result.deployment = std::move(deployment);
    result.createdAtUnixNs = createdAtUnixNs != 0U ? createdAtUnixNs : phaseCostUnixTimeNs();
    result.records = mLocalRecords;
    return result;
}

PhaseCostBundle PhaseCostOracle::snapshotNode(
    PhaseDeploymentFingerprint deployment, std::string bundleVersion, uint64_t createdAtUnixNs) const
{
    return snapshot(PhaseCostBundleSource::kNode, std::move(deployment), std::move(bundleVersion), createdAtUnixNs);
}

void PhaseCostOracle::attachJournal(std::shared_ptr<PhaseNodeCostJournal> journal)
{
    mJournal = std::move(journal);
}

void PhaseCostOracle::resetLocal()
{
    mLocal.reset();
    mLocalRecords.clear();
    mEncoderAnchors.ratios.clear();
    mPrefillAnchors.ratios.clear();
    mDecodeAnchors.ratios.clear();
    mOverlapAnchors.ratios.clear();
    mAnchorState = {};
}

void PhaseCostOracle::setAnchorCollectionActive(bool active) noexcept
{
    mAnchorCollectionActive = active;
}

PhaseCostAnchorState PhaseCostOracle::anchorState() const
{
    return mAnchorState;
}

std::optional<PhaseGlobalCostEstimate> PhaseCostOracle::estimatePrior(
    std::optional<PriorLayer> const& layer, PhaseGlobalActionKey const& key, bool interpolate, bool applyAnchor) const
{
    if (!layer.has_value())
    {
        return std::nullopt;
    }
    std::optional<PhaseGlobalCostEstimate> estimateValue
        = interpolate ? layer->model.estimateInterpolatedPrimaryBatch(key) : layer->model.estimate(key);
    if (!estimateValue.has_value())
    {
        return std::nullopt;
    }
    float uncertaintyMultiplier = 1.0F;
    if (layer->compatibility == PhaseCostCompatibility::kCompatible)
    {
        uncertaintyMultiplier = mConfig.compatibleUncertaintyMultiplier;
    }
    else if (layer->compatibility == PhaseCostCompatibility::kShapeOnly)
    {
        uncertaintyMultiplier = mConfig.shapeOnlyUncertaintyMultiplier;
    }
    PhaseCostScale scale = layer->scale;
    float relativeUncertainty{};
    if (applyAnchor)
    {
        float const adaptive = actionScale(key.kind, mAnchorState.scale);
        switch (key.kind)
        {
        case PhaseGlobalActionKind::kEncoder: scale.encoder *= adaptive; break;
        case PhaseGlobalActionKind::kPrefill: scale.prefill *= adaptive; break;
        case PhaseGlobalActionKind::kDecode: scale.decode *= adaptive; break;
        case PhaseGlobalActionKind::kEncoderPrefill:
        case PhaseGlobalActionKind::kEncoderDecode:
        case PhaseGlobalActionKind::kPrefillDecode: scale.overlap *= adaptive; break;
        case PhaseGlobalActionKind::kNone:
        case PhaseGlobalActionKind::kWait: break;
        }
        relativeUncertainty = actionRelativeUncertainty(key.kind, mAnchorState);
    }
    return scaledEstimate(*estimateValue, key, scale, uncertaintyMultiplier, relativeUncertainty);
}

PhaseGlobalCostEstimate PhaseCostOracle::scaledEstimate(PhaseGlobalCostEstimate estimate,
    PhaseGlobalActionKey const& key, PhaseCostScale const& scale, float uncertaintyMultiplier,
    float relativeUncertainty) noexcept
{
    float const multiplier = actionScale(key.kind, scale);
    estimate.referenceWorkMedianMs *= multiplier;
    estimate.makespanMedianMs *= multiplier;
    estimate.makespanP95Ms *= multiplier;
    estimate.uncertaintyMs *= multiplier * uncertaintyMultiplier;
    estimate.uncertaintyMs
        = std::max(estimate.uncertaintyMs, estimate.makespanMedianMs * std::max(0.0F, relativeUncertainty));
    estimate.makespanP95Ms = std::max(estimate.makespanP95Ms, estimate.makespanMedianMs + estimate.uncertaintyMs);
    return estimate;
}

void PhaseCostOracle::observeAnchor(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation const& observation)
{
    if (!mConfig.anchor.enabled || !mAnchorCollectionActive || !std::isfinite(observation.referenceWorkMs)
        || observation.referenceWorkMs <= 0.0F || !std::isfinite(observation.makespanMs)
        || observation.makespanMs <= 0.0F)
    {
        return;
    }
    std::optional<PhaseGlobalCostEstimate> prior = estimatePrior(mFleet, key, false, false);
    if (!prior.has_value())
    {
        prior = estimatePrior(mBuild, key, false, false);
    }
    if (!prior.has_value() || !std::isfinite(prior->makespanMedianMs)
        || prior->makespanMedianMs <= std::numeric_limits<float>::epsilon())
    {
        return;
    }
    float const ratio = observation.makespanMs / prior->makespanMedianMs;
    if (!std::isfinite(ratio) || ratio < mConfig.anchor.minimumAcceptedScale
        || ratio > mConfig.anchor.maximumAcceptedScale)
    {
        return;
    }

    AnchorSamples* samples{};
    float* scale{};
    float* uncertainty{};
    size_t* sampleCount{};
    switch (key.kind)
    {
    case PhaseGlobalActionKind::kEncoder:
        samples = &mEncoderAnchors;
        scale = &mAnchorState.scale.encoder;
        uncertainty = &mAnchorState.encoderRelativeUncertainty;
        sampleCount = &mAnchorState.encoderSamples;
        break;
    case PhaseGlobalActionKind::kPrefill:
        samples = &mPrefillAnchors;
        scale = &mAnchorState.scale.prefill;
        uncertainty = &mAnchorState.prefillRelativeUncertainty;
        sampleCount = &mAnchorState.prefillSamples;
        break;
    case PhaseGlobalActionKind::kDecode:
        samples = &mDecodeAnchors;
        scale = &mAnchorState.scale.decode;
        uncertainty = &mAnchorState.decodeRelativeUncertainty;
        sampleCount = &mAnchorState.decodeSamples;
        break;
    case PhaseGlobalActionKind::kEncoderPrefill:
    case PhaseGlobalActionKind::kEncoderDecode:
    case PhaseGlobalActionKind::kPrefillDecode:
        samples = &mOverlapAnchors;
        scale = &mAnchorState.scale.overlap;
        uncertainty = &mAnchorState.overlapRelativeUncertainty;
        sampleCount = &mAnchorState.overlapSamples;
        break;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kWait: return;
    }
    samples->ratios.push_back(ratio);
    if (samples->ratios.size() > mConfig.anchor.maxSamplesPerPhase)
    {
        samples->ratios.pop_front();
    }
    *sampleCount = samples->ratios.size();
    if (samples->ratios.size() < mConfig.anchor.minimumSamplesPerPhase)
    {
        return;
    }
    float const center = median(samples->ratios);
    *scale = std::clamp(center, mConfig.anchor.minimumAppliedScale, mConfig.anchor.maximumAppliedScale);
    constexpr float kNormalMadScale = 1.4826F;
    float const relativeMad = kNormalMadScale * medianAbsoluteDeviation(samples->ratios, center)
        / std::max(center, std::numeric_limits<float>::epsilon());
    *uncertainty = std::max(mConfig.anchor.minimumRelativeUncertainty, relativeMad);
}

class PhaseNodeCostJournal::Impl
{
public:
    struct Pending
    {
        PhaseGlobalActionKey key;
        PhaseGlobalCostObservation observation;
        uint64_t observedAtUnixNs{};
    };

    Impl(PhaseNodeCostJournalConfig config, PhaseDeploymentFingerprint deployment)
        : mConfig(validateJournalConfig(std::move(config)))
        , mDeployment(std::move(deployment))
        , mWorker([this] {
            try
            {
                run();
            }
            catch (...)
            {
                {
                    std::lock_guard<std::mutex> lock(mMutex);
                    mFailed = true;
                    mFlushComplete = mFlushRequest;
                }
                mFlushed.notify_all();
            }
        })
    {
    }

    ~Impl() noexcept
    {
        try
        {
            {
                std::lock_guard<std::mutex> lock(mMutex);
                mStop = true;
            }
            mCondition.notify_all();
            if (mWorker.joinable())
            {
                mWorker.join();
            }
        }
        catch (...)
        {
        }
    }

    void enqueue(PhaseGlobalActionKey key, PhaseGlobalCostObservation observation, uint64_t observedAtUnixNs)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mPending.size() >= mConfig.maxPendingObservations)
        {
            ++mDropped;
            return;
        }
        mPending.push_back({key, observation, observedAtUnixNs != 0U ? observedAtUnixNs : phaseCostUnixTimeNs()});
        mCondition.notify_one();
    }

    void flush()
    {
        std::unique_lock<std::mutex> lock(mMutex);
        uint64_t const generation = ++mFlushRequest;
        mCondition.notify_one();
        mFlushed.wait(lock, [&] { return mFailed || mFlushComplete >= generation; });
    }

    size_t dropped() const noexcept
    {
        return mDropped.load();
    }

    std::filesystem::path snapshotPath() const
    {
        return mConfig.directory / "local_cost_snapshot.json";
    }

    std::filesystem::path journalPath() const
    {
        return mConfig.directory / "observations.journal";
    }

private:
    void run()
    {
        std::filesystem::create_directories(mConfig.directory);
        PhaseCostBundle restored;
        std::string error;
        if (phaseTryLoadCostBundle(snapshotPath(), restored, error) && restored.source == PhaseCostBundleSource::kNode
            && phaseCostCompatibility(mDeployment, restored.deployment) == PhaseCostCompatibility::kExact)
        {
            mRecords = std::move(restored.records);
        }
        while (true)
        {
            std::deque<Pending> pending;
            uint64_t flushRequest{};
            bool stop{};
            {
                std::unique_lock<std::mutex> lock(mMutex);
                mCondition.wait_for(lock, mConfig.flushInterval,
                    [&] { return mStop || !mPending.empty() || mFlushRequest > mFlushComplete; });
                pending.swap(mPending);
                flushRequest = mFlushRequest;
                stop = mStop;
            }
            if (!pending.empty())
            {
                appendJournal(pending);
                for (Pending const& value : pending)
                {
                    mergeObservation(
                        mRecords, value.key, value.observation, value.observedAtUnixNs, mConfig.maxSamplesPerKey);
                }
                mSinceSnapshot += pending.size();
            }
            if ((mSinceSnapshot >= mConfig.snapshotEveryObservations || flushRequest > mFlushComplete || stop)
                && !mRecords.empty())
            {
                writeSnapshot();
                mSinceSnapshot = 0U;
            }
            {
                std::lock_guard<std::mutex> lock(mMutex);
                mFlushComplete = std::max(mFlushComplete, flushRequest);
            }
            mFlushed.notify_all();
            if (stop)
            {
                return;
            }
        }
    }

    void appendJournal(std::deque<Pending> const& pending)
    {
        std::ofstream stream(journalPath(), std::ios::app);
        ELLM_CHECK(stream.good(), "Failed to append node phase cost journal");
        for (Pending const& value : pending)
        {
            stream << Json{{"key", keyJson(value.key)}, {"observed_at_unix_ns", value.observedAtUnixNs},
                {"reference_work_ms", value.observation.referenceWorkMs}, {"makespan_ms", value.observation.makespanMs}}
                          .dump()
                   << '\n';
        }
    }

    void writeSnapshot()
    {
        PhaseCostBundle bundle;
        bundle.bundleVersion = "node-local-v1";
        bundle.source = PhaseCostBundleSource::kNode;
        bundle.deployment = mDeployment;
        bundle.createdAtUnixNs = phaseCostUnixTimeNs();
        bundle.records = mRecords;
        phaseWriteCostBundleAtomic(bundle, snapshotPath());
    }

    PhaseNodeCostJournalConfig mConfig;
    PhaseDeploymentFingerprint mDeployment;
    std::vector<PhaseCostRecord> mRecords;
    std::deque<Pending> mPending;
    mutable std::mutex mMutex;
    std::condition_variable mCondition;
    std::condition_variable mFlushed;
    bool mStop{};
    bool mFailed{};
    uint64_t mFlushRequest{};
    uint64_t mFlushComplete{};
    size_t mSinceSnapshot{};
    std::atomic<size_t> mDropped{};
    std::thread mWorker;
};

PhaseNodeCostJournal::PhaseNodeCostJournal(PhaseNodeCostJournalConfig config, PhaseDeploymentFingerprint deployment)
    : mImpl(std::make_unique<Impl>(std::move(config), std::move(deployment)))
{
}

PhaseNodeCostJournal::~PhaseNodeCostJournal() noexcept = default;

void PhaseNodeCostJournal::enqueue(
    PhaseGlobalActionKey key, PhaseGlobalCostObservation observation, uint64_t observedAtUnixNs)
{
    mImpl->enqueue(key, observation, observedAtUnixNs);
}

void PhaseNodeCostJournal::flush()
{
    mImpl->flush();
}

size_t PhaseNodeCostJournal::droppedObservations() const noexcept
{
    return mImpl->dropped();
}

std::filesystem::path PhaseNodeCostJournal::snapshotPath() const
{
    return mImpl->snapshotPath();
}

std::filesystem::path PhaseNodeCostJournal::journalPath() const
{
    return mImpl->journalPath();
}

uint64_t phaseCostUnixTimeNs() noexcept
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count());
}

} // namespace trt_edgellm::rt
