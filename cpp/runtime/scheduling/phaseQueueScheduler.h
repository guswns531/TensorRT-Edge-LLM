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
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Per-request policy hints. Zero SLO targets inherit the scheduler defaults.
struct PhaseSchedulingHints
{
    int32_t priority{};
    double ttftTargetUs{};
    double tpotTargetUs{};
    //! Original host admission time. An empty value is filled on first enqueue.
    std::chrono::steady_clock::time_point submittedAt;
};

//! A unit of phase work. For prefill, tokenCount is the remaining prompt
//! length while queued and the dispatched chunk length while in flight.
//! For decode it is the current KV length. kvSlotId identifies stable physical
//! cache ownership; tokenOffset and promptTokenCount describe chunk progress.
struct PhaseWorkItem
{
    PhaseWorkItem() = default;
    PhaseWorkItem(uint64_t requestId, int32_t tokenCount, int32_t kvSlotId = -1, int32_t tokenOffset = 0,
        int32_t promptTokenCount = 0, bool allowChunkedPrefill = true, PhaseSchedulingHints scheduling = {},
        bool exclusivePrefill = false)
        : requestId(requestId)
        , tokenCount(tokenCount)
        , kvSlotId(kvSlotId)
        , tokenOffset(tokenOffset)
        , promptTokenCount(promptTokenCount)
        , allowChunkedPrefill(allowChunkedPrefill)
        , scheduling(scheduling)
        , exclusivePrefill(exclusivePrefill)
    {
    }

    uint64_t requestId{};
    int32_t tokenCount{};
    int32_t kvSlotId{-1};
    int32_t tokenOffset{};
    int32_t promptTokenCount{};
    //! Gemma4 vision-block attention currently requires one atomic prefill.
    bool allowChunkedPrefill{true};
    PhaseSchedulingHints scheduling;
    //! Keep this request in a one-row prefill batch while still permitting chunking.
    bool exclusivePrefill{};
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
    //! Runtime B*S footprint and unused right-padding tokens for ragged prefill.
    int32_t prefillPaddedTokens{};
    int32_t prefillPaddingTokens{};
    float prefillPackingEfficiency{};
    int32_t prefillInitialRows{};
    int32_t prefillContinuationRows{};
    int32_t prefillFinalRows{};
    int32_t prefillPastKVMin{};
    int32_t prefillPastKVMean{};
    int32_t prefillPastKVMax{};
    int32_t prefillPastKVSpread{};
    int32_t prefillRemainingTokens{};
    double prefillOldestRequestAgeUs{};
    double prefillMinTtftSlackUs{};
    float predictedPrefillGpuMs{};
    float predictedDecodeSlowdownMs{};
    double predictedDecodeDebtUs{};
    int32_t consecutiveOverlapBatches{};
    bool prefillDeferredForTpot{};
    bool prefillCostCoverageMiss{};
    bool overlapEvaluatedByCost{};
    bool latencySafeFallback{};
    int32_t prefillCostLookupRows{};
    int32_t prefillCostLookupChunkLength{};
    int32_t prefillCostLookupMaxPastKVLength{};
    int32_t prefillShapeCandidatesEvaluated{};
    float predictedPrefillShapeScore{};
    bool prefillShapeDrainMode{};
    float adaptiveChunkDecodeQueuePressure{};
    float adaptiveChunkObservedTpotPressure{};
    float adaptiveChunkCombinedPressure{};
    int32_t plannedDecodeBatchSize{};
    int64_t plannedDecodeContextTokens{};
    int32_t plannedDecodeMaxContextLength{};
    int32_t predictedDecodeReplacementRows{};
    int32_t prefillCohortSize{};
    int32_t decodeCohortSize{};
    int32_t decodeTokens{};
    int32_t decodeContextTokens{};
    double prefillQueueWaitUs{};
    double decodeQueueWaitUs{};
    float prefillGpuMs{};
    float decodeGpuMs{};
    float makespanGpuMs{};
    float overlapRatio{};
    //! Host page-pool snapshot after the dispatch completion; zero for linear caches.
    int32_t pagePoolTotalBundles{};
    int32_t pagePoolAllocatedBundles{};
    int32_t pagePoolAvailableBundles{};
    //! Logical page-growth controller state after this dispatch.
    int32_t pageGrowthRequestLimit{};
    int32_t pageGrowthRequestOwners{};
    float pageGrowthTpotPressure{};
};

struct PhaseSchedulerTelemetry
{
    size_t sampleCount{};
    size_t overlapSampleCount{};
    size_t decodeTpotSampleCount{};
    size_t onlineDecodeCostSampleCount{};
    size_t onlineDecodeCostBucketCount{};
    float prefillGpuMsPerToken{};
    float decodeGpuMsPerContextToken{};
    float overlapRatio{};
    double recentDecodeTpotP95Us{};
    float recentDecodeTpotPressure{};
    bool latencySafeFallback{};
    size_t tpotHysteresisTransitions{};
    std::optional<PhaseDispatchMetrics> lastDispatch;
};

//! Read-only queue summary passed to a custom scheduling policy.
struct PhaseQueueSnapshot
{
    size_t prefillQueued{};
    size_t decodeQueued{};
    int32_t prefillCandidateTokens{};
    int64_t prefillRemainingTokens{};
    int32_t prefillContinuationRows{};
    int32_t decodeCandidateTokens{};
    //! Total and maximum context lengths of the first runnable decode rows.
    int64_t decodeCandidateContextTokens{};
    int32_t decodeCandidateMaxContextLength{};
    int32_t consecutiveDecodeBatches{};
    double prefillOldestWaitUs{};
    double prefillOldestRequestAgeUs{};
    double prefillMinTtftSlackUs{};
    double decodeOldestWaitUs{};
    double prefillMaxSloPressure{};
    double decodeMaxSloPressure{};
    int32_t prefillHighestPriority{};
    int32_t decodeHighestPriority{};
    //! Current resource state supplied by the serving facade. Zero denotes a
    //! linear cache or a scheduler without a resource supplier.
    int32_t pagePoolTotalBundles{};
    int32_t pagePoolAllocatedBundles{};
    int32_t pagePoolAvailableBundles{};
    int32_t pageReservationGuaranteedBundles{};
    int32_t pageReservationAvailableBundles{};
};

using PhaseSchedulingPolicy = std::function<PhaseDispatchKind(PhaseQueueSnapshot const&)>;
using PhaseMetricsSchedulingPolicy
    = std::function<PhaseDispatchKind(PhaseQueueSnapshot const&, PhaseSchedulerTelemetry const&)>;
using PhaseWorkEligibilityPolicy = std::function<bool(PhaseWorkItem const&, bool prefill)>;

//! Count selected decode rows whose request was absent from the prior batch.
size_t phaseDecodeReplacementRows(
    std::vector<uint64_t> const& selectedRequestIds, std::vector<uint64_t> const& previousRequestIds);

//! Conservative decode cost point loaded from offline CUDA-event profiling.
//! maxContextLength is the largest per-request KV length covered by the point.
struct PhaseDecodeBatchCost
{
    int32_t batchSize{};
    int32_t maxContextLength{};
    float p95GpuMs{};
    //! Largest sum of row context lengths covered by this point. Zero keeps
    //! legacy behavior and derives batchSize * maxContextLength.
    int64_t maxTotalContextTokens{};
};

//! Live cache and admission counters exposed to scheduler policies.
struct PhaseQueueResourceSnapshot
{
    int32_t pagePoolTotalBundles{};
    int32_t pagePoolAllocatedBundles{};
    int32_t pagePoolAvailableBundles{};
    int32_t pageReservationGuaranteedBundles{};
    int32_t pageReservationAvailableBundles{};
};

using PhaseQueueResourceSupplier = std::function<PhaseQueueResourceSnapshot()>;

//! Conservative prefill cost point loaded from offline CUDA-event profiling.
//! The point covers one uniform chunk shape up to the supplied past-KV and
//! concurrent decode batch bounds.
struct PhasePrefillBatchCost
{
    int32_t batchSize{};
    int32_t chunkLength{};
    int32_t maxPastKVLength{};
    int32_t maxConcurrentDecodeBatchSize{};
    bool initialChunk{};
    float p95GpuMs{};
    float decodeSlowdownP95Ms{};
};

//! Directly observed independent-context overlap cost. Shape bounds are
//! conservative upper buckets generated from CUDA-event dispatch samples.
struct PhaseOverlapBatchCost
{
    int32_t prefillBatchSize{};
    int32_t decodeBatchSize{};
    int32_t chunkLength{};
    int32_t maxPrefillPastKVLength{};
    int32_t maxDecodeContextLength{};
    bool initialChunk{};
    float prefillP95GpuMs{};
    float decodeP95GpuMs{};
    float makespanP95GpuMs{};
    float decodeSlowdownP95Ms{};
};

enum class PhaseSchedulerProfile
{
    kCustom,
    kLatencySafe,
    kBalanced,
    kThroughputBalanced,
    kLongPrefill,
    kAuto,
};

struct PhaseQueueSchedulerConfig
{
    //! Production presets only select scheduler policy behavior. Model and
    //! engine shape limits remain explicit in the fields below.
    PhaseSchedulerProfile profile{PhaseSchedulerProfile::kCustom};
    int32_t maxPrefillBatchSize{1};
    int32_t maxDecodeBatchSize{4};
    //! Optional row cap used only when prefill and decode execute concurrently.
    //! Zero inherits maxPrefillBatchSize. This lets a wide standalone prefill
    //! profile coexist with a smaller, memory-safe overlap shape.
    int32_t maxOverlapPrefillBatchSize{};
    //! Default policy only overlaps short prefills. The initial value comes from
    //! the Gemma4 E2B RTX 3080 crossover benchmark and remains configurable.
    int32_t maxOverlapPrefillTokens{128};
    //! Maximum tokens dispatched per request in one prefill turn. Zero keeps
    //! the legacy whole-prompt behavior.
    int32_t maxPrefillChunkTokens{};
    //! Optional steady-state cap while decode work is queued. This permits a
    //! larger queue-drain chunk when decode is empty without imposing that
    //! interference on active decodes. Zero inherits maxPrefillChunkTokens.
    int32_t decodeActivePrefillChunkTokens{};
    //! Keep using maxPrefillChunkTokens while at least this many prefill
    //! requests remain queued, even when decode is active. This drains a large
    //! arrival burst before switching to the steady-state cap. Zero disables
    //! backlog-triggered large chunks.
    size_t largePrefillChunkQueueThreshold{};
    //! Optional total-token budget for one compatible prefill batch. Zero disables it.
    int32_t maxPrefillBatchTokens{};
    //! Combine different text chunk lengths in one right-padded TensorRT batch.
    //! Initial and continuation chunks remain separate execution classes.
    bool enableRaggedPrefillBatching{};
    //! Pack all valid text tokens into one [1,totalTokens] carrier. Requires
    //! an indexed-paged packed-prefill engine with a compatible chunk limit.
    bool enablePackedPrefillTokenLayout{};
    //! Maximum virtual useful-token credit for each continuation row that
    //! completes a request's prefill. The scheduler caps the credit at the
    //! unused portion of one configured chunk and only grants it below half a
    //! chunk, so large tails do not preempt productive initial work. Zero
    //! preserves pure token selection.
    int32_t prefillCompletionBonusTokens{};
    //! Select a decode batch cap from measured p95 costs and current TPOT
    //! pressure. Empty costs preserve the legacy largest-available behavior.
    bool enableDynamicDecodeBatching{};
    //! Retain one stable decode cohort and replace rows only as requests finish.
    bool enableDecodeCohortBatching{};
    //! One-shot GPU metadata/cache movement cost charged for each newly introduced decode row.
    //! Zero preserves the kernel-only decode cost model.
    float decodeRowReplacementCostMs{};
    std::vector<PhaseDecodeBatchCost> decodeBatchCosts;
    //! Refine static decode costs from confident, context-bucketed decode-only observations.
    bool enableOnlineDecodeCostLearning{};
    size_t onlineDecodeCostMinSamples{8U};
    size_t onlineDecodeCostWindow{32U};
    int32_t onlineDecodeContextBucketTokens{512};
    //! Bound an online p95 correction relative to its static prior.
    float onlineDecodeCostMaxAdjustmentRatio{0.25F};
    //! Switch from deadline fitting to throughput-efficient backlog recovery
    //! before the TPOT deadline is fully exhausted.
    float decodeRecoveryPressureThreshold{1.0F};
    //! Select a prefill row count from profiled p95 cost and decode slack.
    bool enableDynamicPrefillBatching{};
    //! Minimum dynamic prefill batch while at least this many compatible rows exist.
    int32_t minDynamicPrefillBatchSize{1};
    //! Let an expired TTFT override decode interference while decode remains within SLO.
    bool enablePrefillSloRecovery{};
    std::vector<PhasePrefillBatchCost> prefillBatchCosts;
    //! Prefer direct Co(P,D,shape) measurements over slowdown inferred from
    //! separately sampled prefill/decode rows.
    std::vector<PhaseOverlapBatchCost> overlapBatchCosts;
    //! Prevent a predicted TPOT violation by converting overlap to decode-only.
    bool enableTpotHardGuard{};
    //! If enabled, an uncovered overlap shape is unsafe instead of falling back
    //! to the indirect prefill cost table.
    bool requireDirectOverlapCost{};
    //! Let direct overlap costs replace the static maxOverlapPrefillTokens gate.
    //! Prefills already covered by the static gate preserve legacy scheduling.
    //! Larger candidates require direct coverage and the TPOT hard guard so an
    //! unsafe candidate becomes decode-only.
    bool enableCostAwareOverlapAdmission{};
    //! Disable cap-exceeding cost-aware overlap when recent decode TPOT p95
    //! reaches the enter ratio, and restore it only below the exit ratio.
    bool enableTpotHysteresis{};
    //! Collect decode TPOT pressure for external admission policies without changing phase decisions.
    bool enableDecodeTpotTelemetry{};
    float tpotHysteresisEnterRatio{0.8F};
    float tpotHysteresisExitRatio{0.6F};
    size_t tpotHysteresisWindow{32};
    size_t minTpotHysteresisSamples{8};
    int32_t maxConsecutiveOverlapBatches{4};
    double maxPredictedDecodeDebtUs{50000.0};
    int64_t autoLongPrefillBacklogTokens{4096};
    float autoDecodePressureLimit{0.5F};
    //! Keep a bounded set of requests advancing at similar chunk frontiers.
    bool enableWavefrontPrefillBatching{};
    int32_t maxPrefillCohortSize{8};
    int32_t maxPrefillCohortTurns{8};
    float decodeSlackSafetyFactor{0.8F};
    //! Model contract gate. A model with atomic multimodal prefill can disable
    //! chunking for every work item; per-request allowChunkedPrefill remains
    //! the narrower override.
    bool supportsChunkedPrefill{true};
    //! Adapt each text-prefill turn between minPrefillChunkTokens and
    //! maxPrefillChunkTokens using observed CUDA cost and overlap efficiency.
    //! Queue deadlines belong to phase selection so they cannot fragment an
    //! otherwise efficient prefill batch into extra TensorRT executions.
    bool enableAdaptivePrefillChunking{};
    int32_t minPrefillChunkTokens{32};
    int32_t prefillChunkAlignment{8};
    //! Optional profiled chunk shapes for bounded adaptive selection. Values
    //! must be strictly increasing and no larger than maxPrefillChunkTokens.
    //! An empty vector preserves the legacy aligned continuous selection.
    //! A request's final tail may be smaller than the first candidate.
    std::vector<int32_t> adaptivePrefillChunkCandidates;
    //! Jointly select a profiled (prefill batch, chunk) shape instead of
    //! choosing the chunk from queue pressure before dynamic batching. This
    //! requires bounded chunk candidates and profiled prefill costs.
    bool enableCostAwarePrefillShapeSelection{};
    //! Weight applied to predicted decode slowdown, scaled by live decode
    //! queue occupancy, when comparing feasible prefill shapes. Zero
    //! maximizes prefill throughput only.
    float prefillShapeDecodePenaltyWeight{1.0F};
    //! Optional host/enqueue cost charged once per candidate dispatch.
    float prefillShapeEnqueueCostMs{};
    //! Prefer maximum productive tokens among feasible profiled shapes once
    //! the queued prefill backlog reaches this size. Zero disables the drain
    //! regime.
    int64_t prefillShapeDrainBacklogTokens{};
    //! Select the smallest bounded candidate once decode queue occupancy
    //! multiplied by observed TPOT pressure reaches this ratio. Requiring
    //! both signals avoids shrinking chunks merely because a healthy decode
    //! batch is full.
    float adaptivePrefillChunkDecodePressureThreshold{0.8F};
    //! Permit pressure to split a row that would otherwise finish within the
    //! maximum chunk. Disabled by default because an extra TensorRT enqueue
    //! can cost more than the shorter interference window.
    bool allowAdaptivePrefillCompletionSplit{};
    //! Admit one prefill batch after this many decode-only decisions so a
    //! continuous decode queue cannot starve new requests forever.
    int32_t decodeBurstLimit{8};
    //! Opt in to the provided queue-deadline + EWMA GPU-cost policy.
    bool enableMetricsPolicy{};
    double prefillQueueWaitTargetUs{5000.0};
    double decodeQueueWaitTargetUs{2000.0};
    float maxPredictedOverlapPrefillMs{30.0F};
    float minObservedOverlapRatio{0.05F};
    //! At or above this page-pool pressure, prefer draining decode work when
    //! neither queue has already violated its SLO. Zero disables the rule.
    float pagePressureDecodeThreshold{0.8F};
    size_t minMetricsSamples{2};
    float metricsEwmaAlpha{0.2F};
    //! Priority is constrained to [0, maxPriority]. Its contribution is bounded
    //! so an overdue lower-priority phase cannot be starved indefinitely.
    int32_t maxPriority{3};
    double priorityPressureWeight{0.25};
    //! Select higher-priority requests first within each phase batch. Waiting
    //! requests gain one effective priority class per priorityAgingUs.
    bool enablePriorityBatching{};
    double priorityAgingUs{1000000.0};
    //! Optional complete replacement for the provided metrics policy.
    PhaseMetricsSchedulingPolicy metricsPolicy{};
    //! Legacy queue-only policy, used when metrics policy is disabled.
    PhaseSchedulingPolicy policy{};
    //! Optional admission-growth gate. Ineligible work remains queued and keeps
    //! its residence timestamp until the cache owner permits further growth.
    PhaseWorkEligibilityPolicy eligibilityPolicy{};
    //! Optional live page-pool/admission snapshot used by scheduling policy.
    PhaseQueueResourceSupplier resourceSupplier{};
};

struct PhaseDispatchPlan
{
    PhaseDispatchKind kind{PhaseDispatchKind::kNone};
    std::vector<PhaseWorkItem> prefillBatch;
    std::vector<PhaseWorkItem> decodeBatch;
    //! Oldest selected row's host queue residence before dispatch.
    double prefillQueueWaitUs{};
    double decodeQueueWaitUs{};
    float predictedPrefillGpuMs{};
    float predictedDecodeSlowdownMs{};
    double predictedDecodeDebtUs{};
    int32_t consecutiveOverlapBatches{};
    int32_t plannedDecodeBatchSize{};
    int64_t plannedDecodeContextTokens{};
    int32_t plannedDecodeMaxContextLength{};
    int32_t predictedDecodeReplacementRows{};
    bool prefillDeferredForTpot{};
    bool prefillCostCoverageMiss{};
    bool overlapEvaluatedByCost{};
    bool latencySafeFallback{};
    int32_t prefillCostLookupRows{};
    int32_t prefillCostLookupChunkLength{};
    int32_t prefillCostLookupMaxPastKVLength{};
    int32_t prefillShapeCandidatesEvaluated{};
    float predictedPrefillShapeScore{};
    bool prefillShapeDrainMode{};
    float adaptiveChunkDecodeQueuePressure{};
    float adaptiveChunkObservedTpotPressure{};
    float adaptiveChunkCombinedPressure{};
    int32_t prefillCohortSize{};
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
    size_t decodeCohortSize() const noexcept;
    bool empty() const noexcept;
    bool hasRequest(uint64_t requestId) const noexcept;

    //! Update scheduling telemetry after one CUDA-complete dispatch.
    void observeMetrics(PhaseDispatchMetrics const& metrics);
    PhaseSchedulerTelemetry const& telemetry() const noexcept;
    //! Keep online decode refinement out of latency mode while retaining learned samples.
    void setOnlineDecodeCostLearningActive(bool active) noexcept;
    //! Reset learned scheduling history between benchmark epochs.
    //!
    //! Queue ownership is unchanged. The scheduler must be idle so a reset
    //! cannot invalidate fairness or overlap debt for active requests.
    void resetHistory();

private:
    PhaseDispatchKind defaultDecision(PhaseQueueSnapshot const& snapshot) const noexcept;
    PhaseDispatchKind metricsDecision(
        PhaseQueueSnapshot const& snapshot, PhaseSchedulerTelemetry const& telemetry) const noexcept;
    int32_t selectDecodeBatchSize(PhaseQueueSnapshot const& snapshot) const;
    uint64_t onlineDecodeCostKey(int32_t batchSize, int32_t maxContextLength) const noexcept;
    std::optional<float> onlineDecodeP95(int32_t batchSize, int32_t maxContextLength) const;
    std::pair<int64_t, int32_t> decodeCandidateShape(int32_t maxRows) const;
    std::vector<PhaseWorkItem const*> decodeCandidateRows(int32_t maxRows) const;
    int32_t decodeCandidateReplacementRows(int32_t maxRows) const;
    //! Returns -1 when the TPOT guard requires decode-only, zero when no
    //! profiled dynamic decision is available, and a positive selected batch.
    int32_t selectPrefillBatchSize(std::vector<PhaseWorkItem const*> const& candidates, int32_t chunkLength,
        bool initialChunk, bool overlap, int32_t plannedDecodeBatchSize, int32_t plannedDecodeMaxContextLength,
        PhaseQueueSnapshot const& snapshot, bool preferMaximumProgress, float& predictedGpuMs,
        float& predictedDecodeSlowdownMs, bool& costCoverageMiss) const noexcept;
    PhaseQueueSnapshot snapshot() const;
    int32_t dispatchedPrefillTokens(PhaseWorkItem const& item) const noexcept;
    int32_t costAwarePrefillTokens(PhaseWorkItem const& item, int32_t chunkLimit) const noexcept;
    bool isPrefillBatchCompatible(PhaseWorkItem const& item, PhaseWorkItem const& seed, int32_t paddedChunkLength,
        bool initialChunk, bool allowRaggedBatch) const noexcept;
    bool isEligible(PhaseWorkItem const& item, bool prefill) const;
    std::vector<PhaseWorkItem> popBatch(std::deque<PhaseWorkItem>& queue, int32_t maxBatchSize, bool chunkPrefill,
        PhaseQueueSnapshot const& snapshot, PhaseDispatchPlan& plan);
    void enqueueKnownPrefill(PhaseWorkItem item);
    void enqueueKnownDecode(PhaseWorkItem item);

    PhaseQueueSchedulerConfig mConfig;
    std::deque<PhaseWorkItem> mPrefillQueue;
    std::deque<PhaseWorkItem> mDecodeQueue;
    std::unordered_set<uint64_t> mActiveRequestIds;
    std::unordered_set<uint64_t> mInFlightRequestIds;
    std::unordered_map<uint64_t, std::chrono::steady_clock::time_point> mQueuedSince;
    PhaseSchedulerTelemetry mTelemetry;
    std::deque<double> mRecentDecodeTpotUs;
    std::unordered_map<uint64_t, std::deque<float>> mOnlineDecodeGpuMs;
    bool mOnlineDecodeCostLearningActive{};
    bool mLatencySafeFallback{};
    int32_t mConsecutiveDecodeBatches{};
    int32_t mConsecutiveOverlapBatches{};
    double mPredictedDecodeDebtUs{};
    std::unordered_set<uint64_t> mPrefillCohortIds;
    int32_t mPrefillCohortTurns{};
    std::unordered_set<uint64_t> mDecodeCohortIds;
    std::vector<uint64_t> mPreviousDecodeSelectionIds;
};

} // namespace rt
} // namespace trt_edgellm
