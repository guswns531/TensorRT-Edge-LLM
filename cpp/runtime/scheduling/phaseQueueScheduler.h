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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! A unit of phase work. For prefill, tokenCount is the remaining prompt
//! length while queued and the dispatched chunk length while in flight.
//! For decode it is the current KV length. kvSlotId identifies stable physical
//! cache ownership; tokenOffset and promptTokenCount describe chunk progress.
struct PhaseWorkItem
{
    uint64_t requestId{};
    int32_t tokenCount{};
    int32_t kvSlotId{-1};
    int32_t tokenOffset{};
    int32_t promptTokenCount{};
    //! Gemma4 vision-block attention currently requires one atomic prefill.
    bool allowChunkedPrefill{true};
};

enum class PhaseDispatchKind
{
    kNone,
    kPrefill,
    kDecode,
    kOverlap,
};

struct PhaseDispatchMetrics
{
    size_t dispatchIndex{};
    PhaseDispatchKind kind{PhaseDispatchKind::kNone};
    int32_t prefillBatchSize{};
    int32_t decodeBatchSize{};
    int32_t prefillTokens{};
    int32_t decodeTokens{};
    int32_t decodeContextTokens{};
    double prefillQueueWaitUs{};
    double decodeQueueWaitUs{};
    float prefillGpuMs{};
    float decodeGpuMs{};
    float makespanGpuMs{};
    float overlapRatio{};
};

struct PhaseSchedulerTelemetry
{
    size_t sampleCount{};
    size_t overlapSampleCount{};
    float prefillGpuMsPerToken{};
    float decodeGpuMsPerContextToken{};
    float overlapRatio{};
    std::optional<PhaseDispatchMetrics> lastDispatch;
};

//! Read-only queue summary passed to a custom scheduling policy.
struct PhaseQueueSnapshot
{
    size_t prefillQueued{};
    size_t decodeQueued{};
    int32_t prefillCandidateTokens{};
    int32_t decodeCandidateTokens{};
    int32_t consecutiveDecodeBatches{};
    double prefillOldestWaitUs{};
    double decodeOldestWaitUs{};
};

using PhaseSchedulingPolicy = std::function<PhaseDispatchKind(PhaseQueueSnapshot const&)>;
using PhaseMetricsSchedulingPolicy
    = std::function<PhaseDispatchKind(PhaseQueueSnapshot const&, PhaseSchedulerTelemetry const&)>;

struct PhaseQueueSchedulerConfig
{
    int32_t maxPrefillBatchSize{1};
    int32_t maxDecodeBatchSize{4};
    //! Default policy only overlaps short prefills. The initial value comes from
    //! the Gemma4 E2B RTX 3080 crossover benchmark and remains configurable.
    int32_t maxOverlapPrefillTokens{128};
    //! Maximum tokens dispatched per request in one prefill turn. Zero keeps
    //! the legacy whole-prompt behavior.
    int32_t maxPrefillChunkTokens{};
    //! Adapt each text-prefill turn between minPrefillChunkTokens and
    //! maxPrefillChunkTokens using observed CUDA cost and overlap efficiency.
    //! Queue deadlines belong to phase selection so they cannot fragment an
    //! otherwise efficient prefill batch into extra TensorRT executions.
    bool enableAdaptivePrefillChunking{};
    int32_t minPrefillChunkTokens{32};
    int32_t prefillChunkAlignment{8};
    //! Admit one prefill batch after this many decode-only decisions so a
    //! continuous decode queue cannot starve new requests forever.
    int32_t decodeBurstLimit{8};
    //! Opt in to the provided queue-deadline + EWMA GPU-cost policy.
    bool enableMetricsPolicy{};
    double prefillQueueWaitTargetUs{5000.0};
    double decodeQueueWaitTargetUs{2000.0};
    float maxPredictedOverlapPrefillMs{30.0F};
    float minObservedOverlapRatio{0.05F};
    size_t minMetricsSamples{2};
    float metricsEwmaAlpha{0.2F};
    //! Optional complete replacement for the provided metrics policy.
    PhaseMetricsSchedulingPolicy metricsPolicy{};
    //! Legacy queue-only policy, used when metrics policy is disabled.
    PhaseSchedulingPolicy policy{};
};

struct PhaseDispatchPlan
{
    PhaseDispatchKind kind{PhaseDispatchKind::kNone};
    std::vector<PhaseWorkItem> prefillBatch;
    std::vector<PhaseWorkItem> decodeBatch;
    //! Oldest selected row's host queue residence before dispatch.
    double prefillQueueWaitUs{};
    double decodeQueueWaitUs{};
};

//! Host-side two-queue batch scheduler for phase-separated, dual-stream inference.
//!
//! This class intentionally owns no CUDA or TensorRT objects. The execution
//! layer consumes a DispatchPlan, binds each batch's stable KV slots to its
//! phase-local TensorMap, and records CUDA events around the two enqueues.
class PhaseQueueScheduler
//! A shared TensorRT execution context must serialize those enqueues; independent
//! contexts may opt into overlap.
{
public:
    explicit PhaseQueueScheduler(PhaseQueueSchedulerConfig config = {});

    void enqueuePrefill(PhaseWorkItem item);
    void enqueueDecode(PhaseWorkItem item);

    //! Cancel queued work. In-flight requests cannot be cancelled until their event completes.
    bool cancel(uint64_t requestId);

    PhaseDispatchPlan next();

    //! Complete one dispatched prefill chunk. An unfinished prompt is put back
    //! on the prefill queue; the final chunk transitions to decode.
    void completePrefill(PhaseWorkItem item, int32_t resultingKVLength, bool finished = false);

    //! Complete one decode turn. Unfinished requests are requeued for decode;
    //! finished requests leave the scheduler.
    void completeDecode(PhaseWorkItem item, int32_t resultingKVLength, bool finished);

    size_t prefillQueueSize() const noexcept;
    size_t decodeQueueSize() const noexcept;
    bool empty() const noexcept;
    bool hasRequest(uint64_t requestId) const noexcept;

    //! Update scheduling telemetry after one CUDA-complete dispatch.
    void observeMetrics(PhaseDispatchMetrics const& metrics);
    PhaseSchedulerTelemetry const& telemetry() const noexcept;

private:
    PhaseDispatchKind defaultDecision(PhaseQueueSnapshot const& snapshot) const noexcept;
    PhaseDispatchKind metricsDecision(
        PhaseQueueSnapshot const& snapshot, PhaseSchedulerTelemetry const& telemetry) const noexcept;
    PhaseQueueSnapshot snapshot() const;
    int32_t dispatchedPrefillTokens(PhaseWorkItem const& item) const noexcept;
    std::vector<PhaseWorkItem> popBatch(
        std::deque<PhaseWorkItem>& queue, int32_t maxBatchSize, bool chunkPrefill, double& queueWaitUs);
    void enqueueKnownPrefill(PhaseWorkItem item);
    void enqueueKnownDecode(PhaseWorkItem item);

    PhaseQueueSchedulerConfig mConfig;
    std::deque<PhaseWorkItem> mPrefillQueue;
    std::deque<PhaseWorkItem> mDecodeQueue;
    std::unordered_set<uint64_t> mActiveRequestIds;
    std::unordered_set<uint64_t> mInFlightRequestIds;
    std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> mQueuedSince;
    PhaseSchedulerTelemetry mTelemetry;
    int32_t mConsecutiveDecodeBatches{};
};

} // namespace rt
} // namespace trt_edgellm
