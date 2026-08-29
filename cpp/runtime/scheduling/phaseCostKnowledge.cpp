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
#include <array>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>

namespace trt_edgellm::rt
{
namespace
{

using Json = nlohmann::json;

constexpr std::array<uint32_t, 64> kSHA256_ROUND_CONSTANTS{0x428A2F98U, 0x71374491U, 0xB5C0FBCFU, 0xE9B5DBA5U,
    0x3956C25BU, 0x59F111F1U, 0x923F82A4U, 0xAB1C5ED5U, 0xD807AA98U, 0x12835B01U, 0x243185BEU, 0x550C7DC3U, 0x72BE5D74U,
    0x80DEB1FEU, 0x9BDC06A7U, 0xC19BF174U, 0xE49B69C1U, 0xEFBE4786U, 0x0FC19DC6U, 0x240CA1CCU, 0x2DE92C6FU, 0x4A7484AAU,
    0x5CB0A9DCU, 0x76F988DAU, 0x983E5152U, 0xA831C66DU, 0xB00327C8U, 0xBF597FC7U, 0xC6E00BF3U, 0xD5A79147U, 0x06CA6351U,
    0x14292967U, 0x27B70A85U, 0x2E1B2138U, 0x4D2C6DFCU, 0x53380D13U, 0x650A7354U, 0x766A0ABBU, 0x81C2C92EU, 0x92722C85U,
    0xA2BFE8A1U, 0xA81A664BU, 0xC24B8B70U, 0xC76C51A3U, 0xD192E819U, 0xD6990624U, 0xF40E3585U, 0x106AA070U, 0x19A4C116U,
    0x1E376C08U, 0x2748774CU, 0x34B0BCB5U, 0x391C0CB3U, 0x4ED8AA4AU, 0x5B9CCA4FU, 0x682E6FF3U, 0x748F82EEU, 0x78A5636FU,
    0x84C87814U, 0x8CC70208U, 0x90BEFFFAU, 0xA4506CEBU, 0xBEF9A3F7U, 0xC67178F2U};

uint32_t rotateRight(uint32_t value, uint32_t bits) noexcept
{
    return (value >> bits) | (value << (32U - bits));
}

class Sha256
{
public:
    void update(uint8_t const* data, size_t size)
    {
        mBitCount += static_cast<uint64_t>(size) * 8U;
        while (size > 0U)
        {
            size_t const copied = std::min(size, mBlock.size() - mBlockSize);
            std::copy_n(data, copied, mBlock.begin() + static_cast<std::ptrdiff_t>(mBlockSize));
            mBlockSize += copied;
            data += copied;
            size -= copied;
            if (mBlockSize == mBlock.size())
            {
                transform(mBlock.data());
                mBlockSize = 0U;
            }
        }
    }

    std::array<uint8_t, 32> finish()
    {
        uint64_t const messageBits = mBitCount;
        uint8_t const marker = 0x80U;
        updatePadding(&marker, 1U);
        uint8_t const zero{};
        while (mBlockSize != 56U)
        {
            updatePadding(&zero, 1U);
        }
        std::array<uint8_t, 8> length{};
        for (size_t index{}; index < length.size(); ++index)
        {
            length[length.size() - 1U - index] = static_cast<uint8_t>(messageBits >> (index * 8U));
        }
        updatePadding(length.data(), length.size());

        std::array<uint8_t, 32> result{};
        for (size_t index{}; index < mState.size(); ++index)
        {
            result[index * 4U] = static_cast<uint8_t>(mState[index] >> 24U);
            result[index * 4U + 1U] = static_cast<uint8_t>(mState[index] >> 16U);
            result[index * 4U + 2U] = static_cast<uint8_t>(mState[index] >> 8U);
            result[index * 4U + 3U] = static_cast<uint8_t>(mState[index]);
        }
        return result;
    }

private:
    void updatePadding(uint8_t const* data, size_t size)
    {
        while (size > 0U)
        {
            size_t const copied = std::min(size, mBlock.size() - mBlockSize);
            std::copy_n(data, copied, mBlock.begin() + static_cast<std::ptrdiff_t>(mBlockSize));
            mBlockSize += copied;
            data += copied;
            size -= copied;
            if (mBlockSize == mBlock.size())
            {
                transform(mBlock.data());
                mBlockSize = 0U;
            }
        }
    }

    void transform(uint8_t const* block)
    {
        std::array<uint32_t, 64> words{};
        for (size_t index{}; index < 16U; ++index)
        {
            words[index] = static_cast<uint32_t>(block[index * 4U]) << 24U
                | static_cast<uint32_t>(block[index * 4U + 1U]) << 16U
                | static_cast<uint32_t>(block[index * 4U + 2U]) << 8U | static_cast<uint32_t>(block[index * 4U + 3U]);
        }
        for (size_t index = 16U; index < words.size(); ++index)
        {
            uint32_t const sigma0 = rotateRight(words[index - 15U], 7U) ^ rotateRight(words[index - 15U], 18U)
                ^ (words[index - 15U] >> 3U);
            uint32_t const sigma1 = rotateRight(words[index - 2U], 17U) ^ rotateRight(words[index - 2U], 19U)
                ^ (words[index - 2U] >> 10U);
            words[index] = words[index - 16U] + sigma0 + words[index - 7U] + sigma1;
        }

        uint32_t a = mState[0];
        uint32_t b = mState[1];
        uint32_t c = mState[2];
        uint32_t d = mState[3];
        uint32_t e = mState[4];
        uint32_t f = mState[5];
        uint32_t g = mState[6];
        uint32_t h = mState[7];
        for (size_t index{}; index < words.size(); ++index)
        {
            uint32_t const sum1 = rotateRight(e, 6U) ^ rotateRight(e, 11U) ^ rotateRight(e, 25U);
            uint32_t const choose = (e & f) ^ (~e & g);
            uint32_t const temporary1 = h + sum1 + choose + kSHA256_ROUND_CONSTANTS[index] + words[index];
            uint32_t const sum0 = rotateRight(a, 2U) ^ rotateRight(a, 13U) ^ rotateRight(a, 22U);
            uint32_t const majority = (a & b) ^ (a & c) ^ (b & c);
            uint32_t const temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        mState[0] += a;
        mState[1] += b;
        mState[2] += c;
        mState[3] += d;
        mState[4] += e;
        mState[5] += f;
        mState[6] += g;
        mState[7] += h;
    }

    std::array<uint32_t, 8> mState{
        0x6A09E667U, 0xBB67AE85U, 0x3C6EF372U, 0xA54FF53AU, 0x510E527FU, 0x9B05688CU, 0x1F83D9ABU, 0x5BE0CD19U};
    std::array<uint8_t, 64> mBlock{};
    size_t mBlockSize{};
    uint64_t mBitCount{};
};

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

PhaseCostPhaseDriftState* phaseDriftState(PhaseCostDriftState& state, PhaseGlobalActionKind kind) noexcept
{
    switch (kind)
    {
    case PhaseGlobalActionKind::kEncoder: return &state.encoder;
    case PhaseGlobalActionKind::kPrefill: return &state.prefill;
    case PhaseGlobalActionKind::kDecode: return &state.decode;
    case PhaseGlobalActionKind::kEncoderPrefill:
    case PhaseGlobalActionKind::kEncoderDecode:
    case PhaseGlobalActionKind::kPrefillDecode: return &state.overlap;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kWait: return nullptr;
    }
    return nullptr;
}

PhaseCostPhaseDriftState const* phaseDriftState(PhaseCostDriftState const& state, PhaseGlobalActionKind kind) noexcept
{
    switch (kind)
    {
    case PhaseGlobalActionKind::kEncoder: return &state.encoder;
    case PhaseGlobalActionKind::kPrefill: return &state.prefill;
    case PhaseGlobalActionKind::kDecode: return &state.decode;
    case PhaseGlobalActionKind::kEncoderPrefill:
    case PhaseGlobalActionKind::kEncoderDecode:
    case PhaseGlobalActionKind::kPrefillDecode: return &state.overlap;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kWait: return nullptr;
    }
    return nullptr;
}

bool olderThan(uint64_t timestamp, std::chrono::nanoseconds ttl, uint64_t now) noexcept
{
    if (ttl.count() <= 0)
    {
        return false;
    }
    if (timestamp == 0U)
    {
        return true;
    }
    uint64_t const ttlNs = static_cast<uint64_t>(ttl.count());
    return now > timestamp && now - timestamp > ttlNs;
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
        {"primary_work_class", key.primaryWorkClass}, {"residual_augmentation", key.residualAugmentation}};
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
    key.primaryWorkClass = value.value("primary_work_class", 0);
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
        {"config_hash", deployment.configHash}, {"engine_hash", deployment.engineHash},
        {"external_weight_hash", deployment.externalWeightHash}, {"precision", deployment.precision},
        {"kv_dtype", deployment.kvDtype}, {"capability", capabilityJson(deployment.capability)},
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
    result.configHash = value.value("config_hash", std::string{});
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

Json driftPhaseJson(PhaseCostPhaseDriftState const& state)
{
    return {{"local_only", state.localOnly}, {"sample_count", state.sampleCount}, {"median_ratio", state.medianRatio},
        {"observed_at_unix_ns", state.observedAtUnixNs}};
}

PhaseCostPhaseDriftState parseDriftPhase(Json const& value)
{
    PhaseCostPhaseDriftState result;
    result.localOnly = value.value("local_only", false);
    result.sampleCount = value.value("sample_count", size_t{});
    result.medianRatio = value.value("median_ratio", 1.0F);
    result.observedAtUnixNs = value.value("observed_at_unix_ns", uint64_t{});
    return result;
}

Json driftStateJson(PhaseCostDriftState const& state)
{
    return {{"encoder", driftPhaseJson(state.encoder)}, {"prefill", driftPhaseJson(state.prefill)},
        {"decode", driftPhaseJson(state.decode)}, {"overlap", driftPhaseJson(state.overlap)}};
}

PhaseCostDriftState parseDriftState(Json const& value)
{
    PhaseCostDriftState result;
    result.encoder = parseDriftPhase(value.at("encoder"));
    result.prefill = parseDriftPhase(value.at("prefill"));
    result.decode = parseDriftPhase(value.at("decode"));
    result.overlap = parseDriftPhase(value.at("overlap"));
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
    Json result = {{"schema_version", bundle.schemaVersion}, {"bundle_version", bundle.bundleVersion},
        {"source", phaseCostBundleSourceName(bundle.source)}, {"created_at_unix_ns", bundle.createdAtUnixNs},
        {"deployment", deploymentJson(bundle.deployment)}, {"records", records}};
    if (bundle.promotion.has_value())
    {
        result["promotion"] = {{"decode_batching", bundle.promotion->decodeBatching},
            {"prefill_batching", bundle.promotion->prefillBatching},
            {"overlap_selection", bundle.promotion->overlapSelection}};
    }
    if (bundle.driftState.has_value())
    {
        result["drift_state"] = driftStateJson(*bundle.driftState);
    }
    return result;
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

std::string phaseCostFileSha256(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    ELLM_CHECK(stream.is_open(), "Failed to open phase fingerprint artifact: " + path.string());
    Sha256 digest;
    constexpr size_t kREAD_BYTES = 1024U * 1024U;
    std::vector<uint8_t> buffer(kREAD_BYTES);
    while (stream)
    {
        stream.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        std::streamsize const bytes = stream.gcount();
        if (bytes > 0)
        {
            digest.update(buffer.data(), static_cast<size_t>(bytes));
        }
    }
    ELLM_CHECK(stream.eof(), "Failed while reading phase fingerprint artifact: " + path.string());
    std::ostringstream encoded;
    encoded << std::hex << std::setfill('0');
    for (uint8_t const value : digest.finish())
    {
        encoded << std::setw(2) << static_cast<uint32_t>(value);
    }
    return encoded.str();
}

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
    bool const exactEngine = !local.configHash.empty() && local.configHash == candidate.configHash
        && !local.engineHash.empty() && local.engineHash == candidate.engineHash && !local.software.pluginHash.empty()
        && local.software.pluginHash == candidate.software.pluginHash;
    bool const exactExternalWeights = (local.externalWeightHash.empty() && candidate.externalWeightHash.empty())
        || (!local.externalWeightHash.empty() && local.externalWeightHash == candidate.externalWeightHash);
    bool const exactSoftware = local.software.tensorrtVersion == candidate.software.tensorrtVersion
        && local.software.cudaVersion == candidate.software.cudaVersion;
    bool const exactGpu = local.gpu.computeCapability == candidate.gpu.computeCapability
        && local.gpu.smCount == candidate.gpu.smCount && local.gpu.memoryBytes == candidate.gpu.memoryBytes;
    if (exactEngine && exactExternalWeights && exactSoftware && exactGpu)
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
    if (root.contains("promotion"))
    {
        Json const& promotion = root.at("promotion");
        result.promotion = PhaseCostPromotion{promotion.value("decode_batching", false),
            promotion.value("prefill_batching", false), promotion.value("overlap_selection", false)};
    }
    if (root.contains("drift_state"))
    {
        result.driftState = parseDriftState(root.at("drift_state"));
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
    ELLM_CHECK(mConfig.drift.minimumSamplesPerPhase > 0U, "Phase drift minimum sample count must be positive");
    ELLM_CHECK(mConfig.drift.maxSamplesPerPhase >= mConfig.drift.minimumSamplesPerPhase,
        "Phase drift sample window must cover the minimum sample count");
    ELLM_CHECK(mConfig.drift.enterRelativeError > 0.0F && mConfig.drift.exitRelativeError >= 0.0F
            && mConfig.drift.exitRelativeError < mConfig.drift.enterRelativeError,
        "Phase drift hysteresis must satisfy 0 <= exit < enter");
    ELLM_CHECK(mConfig.drift.minimumAcceptedRatio > 0.0F
            && mConfig.drift.maximumAcceptedRatio >= mConfig.drift.minimumAcceptedRatio,
        "Phase drift accepted ratio range is invalid");
    ELLM_CHECK(mConfig.drift.portablePriorTtl.count() >= 0 && mConfig.drift.nodeObservationTtl.count() >= 0
            && mConfig.drift.restoredDriftStateTtl.count() >= 0,
        "Phase cost TTL values must be non-negative");
}

void PhaseCostOracle::observe(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation observation)
{
    observeAnchor(key, observation);
    observeDrift(key, observation);
    mLocal.observe(key, observation);
    mergeObservation(mLocalRecords, key, observation, phaseCostUnixTimeNs(), mConfig.model.windowSize);
    if (mJournal != nullptr)
    {
        mJournal->enqueue(key, observation, mHealthState.drift);
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

std::optional<PhaseGlobalCostEstimate> PhaseCostOracle::estimatePrimaryBatchCoveringContext(
    PhaseGlobalActionKey const& key) const
{
    std::optional<PhaseGlobalCostEstimate> const local = mLocal.estimatePrimaryBatchCoveringContext(key);
    if (local.has_value() && local->sampleCount >= mConfig.sufficientLocalSamples)
    {
        return local;
    }
    if (std::optional<PhaseGlobalCostEstimate> const fleet = estimatePrior(mFleet, key, true, true, false, true))
    {
        return fleet;
    }
    if (std::optional<PhaseGlobalCostEstimate> const build = estimatePrior(mBuild, key, true, true, false, true))
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

PhaseGlobalOverlapCostDiagnostic PhaseCostOracle::localOverlapDiagnostic(PhaseGlobalActionKey const& key) const
{
    return mLocal.overlapDiagnostic(key);
}

size_t PhaseCostOracle::localSampleCount(PhaseGlobalActionKey const& key) const
{
    std::optional<PhaseGlobalCostEstimate> const estimateValue = mLocal.estimate(key);
    return estimateValue.has_value() ? estimateValue->sampleCount : 0U;
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
    ELLM_CHECK(bundle.source == PhaseCostBundleSource::kBuild || bundle.source == PhaseCostBundleSource::kFleet,
        "Only build or fleet bundles can be loaded as a portable prior");
    bool const exactBuild
        = bundle.source == PhaseCostBundleSource::kBuild && compatibility == PhaseCostCompatibility::kExact;
    bool const expired
        = !exactBuild && olderThan(bundle.createdAtUnixNs, mConfig.drift.portablePriorTtl, phaseCostUnixTimeNs());
    PriorLayer layer{PhaseGlobalCostModel(mConfig.model), compatibility, scale, expired};
    for (PhaseCostRecord const& record : bundle.records)
    {
        for (PhaseGlobalCostObservation const& observation : record.observations)
        {
            layer.model.observe(record.key, observation);
        }
    }
    if (bundle.source == PhaseCostBundleSource::kFleet)
    {
        mHealthState.fleetPriorExpired = expired;
        mFleet = std::move(layer);
    }
    else
    {
        mHealthState.buildPriorExpired = expired;
        mBuild = std::move(layer);
    }
}

void PhaseCostOracle::restoreNode(PhaseCostBundle const& bundle)
{
    ELLM_CHECK(bundle.source == PhaseCostBundleSource::kNode, "Only node cost bundles can restore local observations");
    uint64_t const now = phaseCostUnixTimeNs();
    for (PhaseCostRecord const& record : bundle.records)
    {
        if (olderThan(record.observedAtUnixNs, mConfig.drift.nodeObservationTtl, now))
        {
            ++mHealthState.staleNodeRecords;
            continue;
        }
        for (PhaseGlobalCostObservation const& observation : record.observations)
        {
            mLocal.observe(record.key, observation);
            mergeObservation(mLocalRecords, record.key, observation, record.observedAtUnixNs, mConfig.model.windowSize);
        }
    }
    if (bundle.driftState.has_value())
    {
        auto restorePhase = [&](PhaseCostPhaseDriftState& destination, PhaseCostPhaseDriftState const& source) {
            if (!olderThan(source.observedAtUnixNs, mConfig.drift.restoredDriftStateTtl, now))
            {
                destination = source;
            }
        };
        restorePhase(mHealthState.drift.encoder, bundle.driftState->encoder);
        restorePhase(mHealthState.drift.prefill, bundle.driftState->prefill);
        restorePhase(mHealthState.drift.decode, bundle.driftState->decode);
        restorePhase(mHealthState.drift.overlap, bundle.driftState->overlap);
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
    if (source == PhaseCostBundleSource::kNode)
    {
        result.driftState = mHealthState.drift;
    }
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
    mEncoderDrift.ratios.clear();
    mPrefillDrift.ratios.clear();
    mDecodeDrift.ratios.clear();
    mOverlapDrift.ratios.clear();
    mHealthState.drift = {};
    mHealthState.staleNodeRecords = 0U;
}

void PhaseCostOracle::setCalibrationActive(bool active) noexcept
{
    mCalibrationActive = active;
}

PhaseCostAnchorState PhaseCostOracle::anchorState() const
{
    return mAnchorState;
}

PhaseCostHealthState PhaseCostOracle::healthState() const
{
    return mHealthState;
}

std::optional<PhaseGlobalCostEstimate> PhaseCostOracle::estimatePrior(std::optional<PriorLayer> const& layer,
    PhaseGlobalActionKey const& key, bool interpolate, bool applyAnchor, bool ignoreDrift, bool coverContext) const
{
    PhaseCostPhaseDriftState const* drift = phaseDriftState(mHealthState.drift, key.kind);
    if (!layer.has_value() || layer->expired || (!ignoreDrift && drift != nullptr && drift->localOnly))
    {
        return std::nullopt;
    }
    auto estimateForKey = [&](PhaseGlobalActionKey const& lookupKey) {
        return coverContext ? layer->model.estimatePrimaryBatchCoveringContext(lookupKey)
                            : (interpolate ? layer->model.estimateInterpolatedPrimaryBatch(lookupKey)
                                           : layer->model.estimate(lookupKey));
    };
    std::optional<PhaseGlobalCostEstimate> estimateValue = estimateForKey(key);
    bool const containsPrefill = key.kind == PhaseGlobalActionKind::kPrefill
        || key.kind == PhaseGlobalActionKind::kEncoderPrefill || key.kind == PhaseGlobalActionKind::kPrefillDecode;
    if (!estimateValue.has_value() && containsPrefill && key.primaryWorkClass != 0)
    {
        PhaseGlobalActionKey anyClassKey = key;
        anyClassKey.primaryWorkClass = 0;
        estimateValue = estimateForKey(anyClassKey);
    }
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
    if (!mConfig.anchor.enabled || !mCalibrationActive || !std::isfinite(observation.referenceWorkMs)
        || observation.referenceWorkMs <= 0.0F || !std::isfinite(observation.makespanMs)
        || observation.makespanMs <= 0.0F)
    {
        return;
    }
    std::optional<PhaseGlobalCostEstimate> prior = estimatePrior(mFleet, key, false, false, true);
    if (!prior.has_value())
    {
        prior = estimatePrior(mBuild, key, false, false, true);
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
    DriftSamples* driftSamples{};
    PhaseCostPhaseDriftState* driftState{};
    float* scale{};
    float* uncertainty{};
    size_t* sampleCount{};
    switch (key.kind)
    {
    case PhaseGlobalActionKind::kEncoder:
        samples = &mEncoderAnchors;
        driftSamples = &mEncoderDrift;
        driftState = &mHealthState.drift.encoder;
        scale = &mAnchorState.scale.encoder;
        uncertainty = &mAnchorState.encoderRelativeUncertainty;
        sampleCount = &mAnchorState.encoderSamples;
        break;
    case PhaseGlobalActionKind::kPrefill:
        samples = &mPrefillAnchors;
        driftSamples = &mPrefillDrift;
        driftState = &mHealthState.drift.prefill;
        scale = &mAnchorState.scale.prefill;
        uncertainty = &mAnchorState.prefillRelativeUncertainty;
        sampleCount = &mAnchorState.prefillSamples;
        break;
    case PhaseGlobalActionKind::kDecode:
        samples = &mDecodeAnchors;
        driftSamples = &mDecodeDrift;
        driftState = &mHealthState.drift.decode;
        scale = &mAnchorState.scale.decode;
        uncertainty = &mAnchorState.decodeRelativeUncertainty;
        sampleCount = &mAnchorState.decodeSamples;
        break;
    case PhaseGlobalActionKind::kEncoderPrefill:
    case PhaseGlobalActionKind::kEncoderDecode:
    case PhaseGlobalActionKind::kPrefillDecode:
        samples = &mOverlapAnchors;
        driftSamples = &mOverlapDrift;
        driftState = &mHealthState.drift.overlap;
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
    driftSamples->ratios.clear();
    *driftState = {};
}

void PhaseCostOracle::observeDrift(PhaseGlobalActionKey const& key, PhaseGlobalCostObservation const& observation)
{
    if (!mConfig.drift.enabled || mCalibrationActive || !std::isfinite(observation.makespanMs)
        || observation.makespanMs <= 0.0F)
    {
        return;
    }
    std::optional<PhaseGlobalCostEstimate> prior = estimatePrior(mFleet, key, false, true, true);
    if (!prior.has_value())
    {
        prior = estimatePrior(mBuild, key, false, true, true);
    }
    if (!prior.has_value() || !std::isfinite(prior->makespanMedianMs)
        || prior->makespanMedianMs <= std::numeric_limits<float>::epsilon())
    {
        return;
    }
    float const ratio = observation.makespanMs / prior->makespanMedianMs;
    if (!std::isfinite(ratio) || ratio < mConfig.drift.minimumAcceptedRatio
        || ratio > mConfig.drift.maximumAcceptedRatio)
    {
        return;
    }

    DriftSamples* samples{};
    PhaseCostPhaseDriftState* state = phaseDriftState(mHealthState.drift, key.kind);
    switch (key.kind)
    {
    case PhaseGlobalActionKind::kEncoder: samples = &mEncoderDrift; break;
    case PhaseGlobalActionKind::kPrefill: samples = &mPrefillDrift; break;
    case PhaseGlobalActionKind::kDecode: samples = &mDecodeDrift; break;
    case PhaseGlobalActionKind::kEncoderPrefill:
    case PhaseGlobalActionKind::kEncoderDecode:
    case PhaseGlobalActionKind::kPrefillDecode: samples = &mOverlapDrift; break;
    case PhaseGlobalActionKind::kNone:
    case PhaseGlobalActionKind::kWait: return;
    }
    ELLM_CHECK(state != nullptr, "Phase drift state is missing for an executable action");
    samples->ratios.push_back(ratio);
    if (samples->ratios.size() > mConfig.drift.maxSamplesPerPhase)
    {
        samples->ratios.pop_front();
    }
    state->sampleCount = samples->ratios.size();
    state->medianRatio = median(samples->ratios);
    state->observedAtUnixNs = phaseCostUnixTimeNs();
    if (samples->ratios.size() < mConfig.drift.minimumSamplesPerPhase)
    {
        return;
    }
    float const relativeError = std::abs(state->medianRatio - 1.0F);
    if (!state->localOnly && relativeError > mConfig.drift.enterRelativeError)
    {
        state->localOnly = true;
    }
    else if (state->localOnly && relativeError < mConfig.drift.exitRelativeError)
    {
        state->localOnly = false;
    }
}

class PhaseNodeCostJournal::Impl
{
public:
    struct Pending
    {
        PhaseGlobalActionKey key;
        PhaseGlobalCostObservation observation;
        PhaseCostDriftState driftState;
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

    void enqueue(PhaseGlobalActionKey key, PhaseGlobalCostObservation observation, PhaseCostDriftState driftState,
        uint64_t observedAtUnixNs)
    {
        std::lock_guard<std::mutex> lock(mMutex);
        if (mPending.size() >= mConfig.maxPendingObservations)
        {
            ++mDropped;
            return;
        }
        mPending.push_back({key, observation, std::move(driftState),
            observedAtUnixNs != 0U ? observedAtUnixNs : phaseCostUnixTimeNs()});
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
            if (restored.driftState.has_value())
            {
                mDriftState = *restored.driftState;
            }
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
                mDriftState = pending.back().driftState;
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
        bundle.driftState = mDriftState;
        phaseWriteCostBundleAtomic(bundle, snapshotPath());
    }

    PhaseNodeCostJournalConfig mConfig;
    PhaseDeploymentFingerprint mDeployment;
    std::vector<PhaseCostRecord> mRecords;
    PhaseCostDriftState mDriftState;
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

void PhaseNodeCostJournal::enqueue(PhaseGlobalActionKey key, PhaseGlobalCostObservation observation,
    PhaseCostDriftState driftState, uint64_t observedAtUnixNs)
{
    mImpl->enqueue(key, observation, std::move(driftState), observedAtUnixNs);
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
