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

#include "runtime/scheduling/independentPhaseAsyncServer.h"

#include "runtime/scheduling/phaseActivityTimeline.h"
#include "runtime/scheduling/phaseVisionAdapter.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace trt_edgellm::rt
{
namespace
{

constexpr uint64_t kSTATE_HASH_OFFSET = 14695981039346656037ULL;
constexpr uint64_t kSTATE_HASH_PRIME = 1099511628211ULL;

void addStateHash(uint64_t& hash, uint64_t value) noexcept
{
    hash ^= value;
    hash *= kSTATE_HASH_PRIME;
}

void addVisionPayloadHash(uint64_t& hash, PhaseVisionPayload const* payload) noexcept
{
    addStateHash(hash, payload != nullptr ? 1U : 0U);
    if (payload == nullptr)
    {
        return;
    }
    addStateHash(hash, payload->byteSize());
    addStateHash(hash, payload->prefillByteSize());
    addStateHash(hash, payload->tokenIds.size());
    for (std::vector<int32_t> const& ids : payload->tokenIds)
    {
        addStateHash(hash, ids.size());
        for (int32_t const id : ids)
        {
            addStateHash(hash, static_cast<uint64_t>(id));
        }
    }
}

} // namespace

size_t phaseDecodeAlignedAdmissionCapacity(size_t requestedCapacity, size_t decodeBatchCapacity) noexcept
{
    if (requestedCapacity == 0 || decodeBatchCapacity == 0 || requestedCapacity <= decodeBatchCapacity)
    {
        return requestedCapacity;
    }
    return requestedCapacity - requestedCapacity % decodeBatchCapacity;
}

bool phaseShouldSynchronizeDecodeSampling(
    bool enabled, bool fromPrefill, bool externalPrefillQueued, size_t externalProducerRows) noexcept
{
    return enabled && !fromPrefill && !externalPrefillQueued && externalProducerRows == 0U;
}

size_t phaseServingIngressQuantum(size_t maxPendingRequests, size_t prefillBatchCapacity) noexcept
{
    return std::min(maxPendingRequests, std::max(size_t{1U}, prefillBatchCapacity));
}

bool shouldDeferDecodeForSamplingRefill(
    size_t targetRows, size_t prefillRows, size_t decodeRows, size_t pendingDecodeSamplingRows) noexcept
{
    return targetRows > 0 && prefillRows == 0 && decodeRows > 0 && decodeRows < targetRows
        && decodeRows + pendingDecodeSamplingRows >= targetRows;
}

bool shouldDeferPrefillForMicrobatchFormation(size_t targetRows, size_t prefillRows, size_t pendingProducerRows,
    double minTtftSlackUs, double ttftGuardUs, double elapsedUs, double formationWindowUs) noexcept
{
    return targetRows > 0 && prefillRows > 0 && prefillRows < targetRows
        && pendingProducerRows >= targetRows - prefillRows && formationWindowUs > 0.0 && elapsedUs < formationWindowUs
        && minTtftSlackUs > ttftGuardUs;
}

bool shouldDeferAdmissionForPrefillRefill(size_t targetRows, size_t pendingRows, size_t availableSlots,
    size_t activeRows, double elapsedUs, double formationWindowUs) noexcept
{
    return targetRows > 1U && pendingRows >= targetRows && availableSlots > 0U && availableSlots < targetRows
        && activeRows > 0U && formationWindowUs > 0.0 && elapsedUs < formationWindowUs;
}

bool phasePrefillFormationSupportsOutputLength(int32_t maxOutputTokens, int32_t cohortMaxOutputTokens) noexcept
{
    return maxOutputTokens <= 0 || cohortMaxOutputTokens <= maxOutputTokens;
}

std::optional<size_t> phasePrefillFormationCostIndex(std::vector<IndependentPhaseFormationCost> const& costs,
    int32_t maxPromptTokens, int32_t maxOutputTokens, size_t decodeRows, float decodeTpotPressure) noexcept
{
    std::optional<size_t> selected;
    for (size_t index = 0; index < costs.size(); ++index)
    {
        IndependentPhaseFormationCost const& cost = costs[index];
        bool const matches = cost.throughputGainPct > 0.0 && cost.targetBatchSize > 0 && cost.windowUs > 0.0
            && (cost.maxPromptTokens <= 0 || maxPromptTokens <= cost.maxPromptTokens)
            && (cost.maxOutputTokens <= 0 || maxOutputTokens <= cost.maxOutputTokens)
            && (cost.maxDecodeRows == 0 || decodeRows <= cost.maxDecodeRows)
            && (cost.maxDecodeTpotPressure <= 0.0F || decodeTpotPressure <= cost.maxDecodeTpotPressure);
        if (matches && (!selected || cost.throughputGainPct > costs[*selected].throughputGainPct))
        {
            selected = index;
        }
    }
    return selected;
}

bool nextAdaptiveThroughputMode(bool currentThroughputMode, size_t pendingRequests, size_t activeRequests,
    size_t latencyInFlightLimit, size_t backlogEnterThreshold) noexcept
{
    if (!currentThroughputMode)
    {
        return pendingRequests >= backlogEnterThreshold;
    }
    return pendingRequests > 0 || activeRequests > latencyInFlightLimit;
}

size_t nextStepwiseAdmissionLimit(size_t currentLimit, size_t latencyLimit, size_t throughputLimit, size_t step,
    size_t pendingRequests, size_t activeRequests, size_t backlogEnterThreshold, int32_t availablePages,
    int32_t minFreePages, float decodeTpotPressure, float pressureEnterRatio, float pressureExitRatio) noexcept
{
    currentLimit = std::clamp(currentLimit, latencyLimit, throughputLimit);
    size_t const lowerLimit = currentLimit - latencyLimit > step ? currentLimit - step : latencyLimit;
    size_t const upperLimit = throughputLimit - currentLimit > step ? currentLimit + step : throughputLimit;
    bool const pagePressure = minFreePages > 0 && availablePages < minFreePages;
    bool const decodePressure = pressureEnterRatio > 0.0F && decodeTpotPressure >= pressureEnterRatio;
    if ((pagePressure || decodePressure) && currentLimit > latencyLimit)
    {
        return lowerLimit;
    }

    bool const decodeAllowsGrowth
        = pressureExitRatio <= 0.0F || decodeTpotPressure <= pressureExitRatio || decodeTpotPressure == 0.0F;
    bool const backlog = pendingRequests >= backlogEnterThreshold;
    if (backlog && activeRequests >= currentLimit && !pagePressure && decodeAllowsGrowth
        && currentLimit < throughputLimit)
    {
        return upperLimit;
    }
    if (!backlog && activeRequests <= lowerLimit && currentLimit > latencyLimit)
    {
        return lowerLimit;
    }
    return currentLimit;
}

size_t phaseAdmissionLimitForTpotBudget(std::vector<IndependentPhaseAdmissionCost> const& costs, size_t latencyLimit,
    size_t throughputLimit, double tpotBudgetUs) noexcept
{
    if (costs.empty() || tpotBudgetUs <= 0.0)
    {
        return throughputLimit;
    }
    size_t result = latencyLimit;
    for (IndependentPhaseAdmissionCost const& cost : costs)
    {
        if (cost.inFlightLimit >= latencyLimit && cost.inFlightLimit <= throughputLimit
            && cost.tpotP95Us <= tpotBudgetUs)
        {
            result = std::max(result, cost.inFlightLimit);
        }
    }
    return result;
}

bool phaseAdmissionTpotBudgetSatisfiable(
    std::vector<IndependentPhaseAdmissionCost> const& costs, size_t latencyLimit, double tpotBudgetUs) noexcept
{
    if (costs.empty() || tpotBudgetUs <= 0.0)
    {
        return true;
    }
    auto const point = std::find_if(costs.cbegin(), costs.cend(),
        [&](IndependentPhaseAdmissionCost const& cost) { return cost.inFlightLimit >= latencyLimit; });
    return point != costs.cend() && point->tpotP95Us <= tpotBudgetUs;
}

bool phaseAdmissionUsesExternalProfile(size_t externalRequests, size_t totalRequests, size_t externalPrefillTokens,
    double minExternalRequestFraction, size_t minExternalPrefillTokens) noexcept
{
    if (totalRequests == 0 || externalRequests == 0)
    {
        return false;
    }
    double const externalFraction
        = static_cast<double>(std::min(externalRequests, totalRequests)) / static_cast<double>(totalRequests);
    return externalFraction >= minExternalRequestFraction && externalPrefillTokens >= minExternalPrefillTokens;
}

std::optional<bool> phaseAdmissionExternalProfileForEpoch(std::optional<bool> currentSelection, size_t externalRequests,
    size_t totalRequests, size_t externalPrefillTokens, double minExternalRequestFraction,
    size_t minExternalPrefillTokens) noexcept
{
    if (totalRequests == 0)
    {
        return std::nullopt;
    }
    if (currentSelection.value_or(false) || externalPrefillTokens < minExternalPrefillTokens)
    {
        return currentSelection;
    }
    if (phaseAdmissionUsesExternalProfile(externalRequests, totalRequests, externalPrefillTokens,
            minExternalRequestFraction, minExternalPrefillTokens))
    {
        return true;
    }
    // A partial ingress cohort is not evidence of a text-only epoch. Keep the
    // profile provisional so later vision arrivals can still select it.
    return std::nullopt;
}

double phaseAdmissionProfileTpotBudget(
    double defaultBudgetUs, double externalBudgetUs, bool externalProfileActive) noexcept
{
    return externalProfileActive && externalBudgetUs > 0.0 ? externalBudgetUs : defaultBudgetUs;
}

std::vector<int32_t> phaseServingWarmupBatchSizes(int32_t maxDecodeBatchSize, std::vector<int32_t> requestedBatchSizes)
{
    ELLM_CHECK(maxDecodeBatchSize > 0, "Phase serving warmup requires a positive decode batch limit");
    if (requestedBatchSizes.empty())
    {
        requestedBatchSizes = {std::max(1, maxDecodeBatchSize / 8), std::max(1, maxDecodeBatchSize / 4),
            std::max(1, maxDecodeBatchSize / 2), std::max(1, 3 * maxDecodeBatchSize / 4), maxDecodeBatchSize};
    }
    for (int32_t const batchSize : requestedBatchSizes)
    {
        ELLM_CHECK(batchSize > 0 && batchSize <= maxDecodeBatchSize,
            "Phase serving warmup batch is outside the decode profile");
    }
    std::sort(requestedBatchSizes.begin(), requestedBatchSizes.end());
    requestedBatchSizes.erase(
        std::unique(requestedBatchSizes.begin(), requestedBatchSizes.end()), requestedBatchSizes.end());
    return requestedBatchSizes;
}

int32_t phasePageReservationGuaranteedPages(
    std::vector<IndependentPhasePageReservation> const& reservations, int32_t maxGrowthRequests)
{
    ELLM_CHECK(maxGrowthRequests > 0, "Phase page growth request limit must be positive");
    int64_t basePages{};
    std::vector<int32_t> tails;
    tails.reserve(reservations.size());
    for (IndependentPhasePageReservation const& reservation : reservations)
    {
        ELLM_CHECK(reservation.basePages >= 0 && reservation.fullPages >= reservation.basePages,
            "Phase page reservation is invalid");
        basePages += reservation.basePages;
        tails.push_back(reservation.fullPages - reservation.basePages);
    }
    std::sort(tails.begin(), tails.end(), std::greater<int32_t>());
    int32_t const growthCount = std::min<int32_t>(maxGrowthRequests, tails.size());
    int64_t const guaranteed = basePages + std::accumulate(tails.begin(), tails.begin() + growthCount, int64_t{});
    ELLM_CHECK(guaranteed <= std::numeric_limits<int32_t>::max(), "Phase page reservation count overflowed");
    return static_cast<int32_t>(guaranteed);
}

bool phasePageReservationsFit(
    int32_t pageBudget, std::vector<IndependentPhasePageReservation> const& reservations, int32_t maxGrowthRequests)
{
    ELLM_CHECK(pageBudget >= 0, "Phase page reservation budget must be non-negative");
    return phasePageReservationGuaranteedPages(reservations, maxGrowthRequests) <= pageBudget;
}

size_t phaseAdmissiblePageReservationPrefix(std::vector<IndependentPhasePageReservation> existing,
    std::vector<IndependentPhasePageReservation> const& candidates, int32_t pageBudget, int32_t maxGrowthRequests,
    size_t maxRequests)
{
    size_t admitted{};
    existing.reserve(existing.size() + std::min(maxRequests, candidates.size()));
    while (admitted < candidates.size() && admitted < maxRequests)
    {
        existing.push_back(candidates[admitted]);
        if (!phasePageReservationsFit(pageBudget, existing, maxGrowthRequests))
        {
            break;
        }
        ++admitted;
    }
    return admitted;
}

std::vector<uint64_t> selectPhasePageGrowthOwners(std::vector<IndependentPhasePageReservation> const& reservations,
    std::vector<uint64_t> const& currentOwners, int32_t maxGrowthRequests)
{
    ELLM_CHECK(maxGrowthRequests > 0, "Phase page growth request limit must be positive");
    std::unordered_map<uint64_t, int32_t> tails;
    tails.reserve(reservations.size());
    for (IndependentPhasePageReservation const& reservation : reservations)
    {
        ELLM_CHECK(reservation.fullPages >= reservation.basePages, "Phase page reservation tail is invalid");
        ELLM_CHECK(tails.emplace(reservation.requestId, reservation.fullPages - reservation.basePages).second,
            "Phase page reservations contain a duplicate request");
    }

    std::vector<uint64_t> owners;
    owners.reserve(static_cast<size_t>(maxGrowthRequests));
    std::unordered_set<uint64_t> selected;
    std::vector<uint64_t> sortedCurrent = currentOwners;
    std::sort(sortedCurrent.begin(), sortedCurrent.end());
    for (uint64_t const requestId : sortedCurrent)
    {
        auto const tail = tails.find(requestId);
        if (tail != tails.end() && tail->second > 0 && selected.insert(requestId).second)
        {
            owners.push_back(requestId);
            if (static_cast<int32_t>(owners.size()) == maxGrowthRequests)
            {
                return owners;
            }
        }
    }

    std::vector<std::pair<int32_t, uint64_t>> candidates;
    candidates.reserve(reservations.size());
    for (IndependentPhasePageReservation const& reservation : reservations)
    {
        int32_t const tail = reservation.fullPages - reservation.basePages;
        if (tail > 0 && selected.find(reservation.requestId) == selected.end())
        {
            candidates.emplace_back(tail, reservation.requestId);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](auto const& left, auto const& right) {
        return left.first > right.first || (left.first == right.first && left.second < right.second);
    });
    for (auto const& candidate : candidates)
    {
        owners.push_back(candidate.second);
        if (static_cast<int32_t>(owners.size()) == maxGrowthRequests)
        {
            break;
        }
    }
    return owners;
}

bool shouldDeferPhasePageGrowthCohort(
    size_t growthOwners, size_t startedGrowthOwners, size_t waitingUnstartedOwners) noexcept
{
    return startedGrowthOwners < growthOwners && waitingUnstartedOwners < growthOwners - startedGrowthOwners;
}

IndependentPhaseAsyncServer::IndependentPhaseAsyncServer(IndependentPhaseServerConfig config,
    IndependentPhaseCoordinator& coordinator, StableKVPageManager& ownership, IndependentPhaseRequestAdapter adapter,
    PhasePrefixReuseCache* prefixCache)
    : mConfig(std::move(config))
    , mCoordinator(coordinator)
    , mOwnership(ownership)
    , mAdapter(std::move(adapter))
    , mPrefixCache(prefixCache)
{
    ELLM_CHECK(mConfig.maxInFlightRequests > 0, "Independent phase server request capacity must be positive");
    ELLM_CHECK(mConfig.defaultMaxOutputTokens > 0, "Independent phase server output capacity must be positive");
    ELLM_CHECK(mConfig.outputHeadroomTokens >= 0, "Independent phase server output headroom must be non-negative");
    if (mConfig.pageReservationMode == IndependentPhasePageReservationMode::kHeadroom)
    {
        ELLM_CHECK(mConfig.maxConcurrentPageGrowthRequests > 0,
            "Independent phase server page growth request limit must be positive");
        ELLM_CHECK(static_cast<size_t>(mConfig.maxConcurrentPageGrowthRequests) <= mConfig.maxInFlightRequests,
            "Independent phase server page growth request limit exceeds in-flight capacity");
        ELLM_CHECK(!mConfig.enablePrefixReuse,
            "Incremental phase page reservation does not support retained prefix-cache slots");
    }
    ELLM_CHECK(mConfig.decodeRefillBatchSize <= mConfig.maxInFlightRequests,
        "Decode refill batch cannot exceed the server in-flight capacity");
    ELLM_CHECK(mConfig.admissionRefillBatchSize <= mConfig.maxInFlightRequests,
        "Admission refill batch cannot exceed the server in-flight capacity");
    ELLM_CHECK(std::isfinite(mConfig.admissionRefillWindowUs) && mConfig.admissionRefillWindowUs >= 0.0,
        "Admission refill window must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.globalSamplingColdStartUs) && mConfig.globalSamplingColdStartUs > 0.0,
        "Global sampling cold-start latency must be finite and positive");
    ELLM_CHECK(mConfig.globalSamplingLatencyWindow > 0U, "Global sampling latency window must be positive");
    ELLM_CHECK(mConfig.prefillFormationBatchSize <= mConfig.maxInFlightRequests,
        "Prefill formation batch cannot exceed the server in-flight capacity");
    ELLM_CHECK(std::isfinite(mConfig.prefillFormationWindowUs) && mConfig.prefillFormationWindowUs >= 0.0,
        "Prefill formation window must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.prefillFormationTtftTargetUs) && mConfig.prefillFormationTtftTargetUs >= 0.0,
        "Prefill formation TTFT target must be finite and non-negative");
    ELLM_CHECK(
        mConfig.prefillFormationMaxOutputTokens >= 0, "Prefill formation output token limit must be non-negative");
    ELLM_CHECK(std::isfinite(mConfig.prefillFormationTtftGuardUs) && mConfig.prefillFormationTtftGuardUs >= 0.0,
        "Prefill formation TTFT guard must be finite and non-negative");
    for (IndependentPhaseFormationCost const& cost : mConfig.prefillFormationCosts)
    {
        ELLM_CHECK(cost.maxPromptTokens >= 0 && cost.maxOutputTokens >= 0,
            "Prefill formation request-class limits must be non-negative");
        ELLM_CHECK(cost.targetBatchSize > 0 && cost.targetBatchSize <= mConfig.maxInFlightRequests,
            "Prefill formation profiled batch must be in the server in-flight range");
        ELLM_CHECK(std::isfinite(cost.maxDecodeTpotPressure) && cost.maxDecodeTpotPressure >= 0.0F,
            "Prefill formation TPOT pressure must be finite and non-negative");
        ELLM_CHECK(std::isfinite(cost.windowUs) && cost.windowUs > 0.0,
            "Prefill formation profiled window must be finite and positive");
        ELLM_CHECK(std::isfinite(cost.ttftTargetUs) && cost.ttftTargetUs >= 0.0,
            "Prefill formation profiled TTFT target must be finite and non-negative");
        ELLM_CHECK(std::isfinite(cost.ttftGuardUs) && cost.ttftGuardUs >= 0.0,
            "Prefill formation profiled TTFT guard must be finite and non-negative");
        ELLM_CHECK(std::isfinite(cost.throughputGainPct) && cost.throughputGainPct > 0.0,
            "Prefill formation cost table must contain only measured positive-gain regions");
    }
    ELLM_CHECK(!mConfig.enableAdaptiveAdmission
            || (mConfig.latencyInFlightRequests > 0 && mConfig.latencyInFlightRequests <= mConfig.maxInFlightRequests
                && mConfig.adaptiveBacklogEnterRequests > 0),
        "Adaptive admission requires valid latency and backlog thresholds");
    ELLM_CHECK(!mConfig.enableStepwiseAdaptiveAdmission || mConfig.enableAdaptiveAdmission,
        "Stepwise admission requires adaptive admission");
    ELLM_CHECK(!mConfig.enableStepwiseAdaptiveAdmission
            || (mConfig.adaptiveAdmissionStep > 0 && mConfig.adaptiveAdmissionDwellSamples > 0
                && mConfig.adaptiveAdmissionMinFreePages >= 0
                && (mConfig.adaptiveAdmissionTpotPressureEnterRatio == 0.0F
                    || (mConfig.adaptiveAdmissionTpotPressureExitRatio > 0.0F
                        && mConfig.adaptiveAdmissionTpotPressureExitRatio
                            <= mConfig.adaptiveAdmissionTpotPressureEnterRatio))),
        "Stepwise admission requires valid step, dwell, page, and TPOT thresholds");
    ELLM_CHECK(std::isfinite(mConfig.adaptiveAdmissionTpotBudgetUs) && mConfig.adaptiveAdmissionTpotBudgetUs >= 0.0,
        "Predictive admission TPOT budget must be finite and non-negative");
    ELLM_CHECK(std::isfinite(mConfig.adaptiveAdmissionExternalTpotBudgetUs)
            && mConfig.adaptiveAdmissionExternalTpotBudgetUs >= 0.0,
        "Predictive external admission TPOT budget must be finite and non-negative");
    auto const validateAdmissionCosts = [&](std::vector<IndependentPhaseAdmissionCost> const& costs) {
        size_t previousAdmissionCostLimit{};
        for (IndependentPhaseAdmissionCost const& cost : costs)
        {
            ELLM_CHECK(cost.inFlightLimit > previousAdmissionCostLimit
                    && cost.inFlightLimit <= mConfig.maxInFlightRequests && std::isfinite(cost.tpotP95Us)
                    && cost.tpotP95Us > 0.0,
                "Predictive admission costs must have increasing valid limits and positive finite TPOT values");
            previousAdmissionCostLimit = cost.inFlightLimit;
        }
    };
    validateAdmissionCosts(mConfig.adaptiveAdmissionCosts);
    validateAdmissionCosts(mConfig.adaptiveAdmissionExternalCosts);
    ELLM_CHECK(std::isfinite(mConfig.adaptiveAdmissionExternalRequestFraction)
            && mConfig.adaptiveAdmissionExternalRequestFraction >= 0.0
            && mConfig.adaptiveAdmissionExternalRequestFraction <= 1.0,
        "Predictive external admission request fraction must be in [0, 1]");
    if (!mConfig.adaptiveAdmissionExternalCosts.empty())
    {
        ELLM_CHECK(mConfig.adaptiveAdmissionExternalRequestFraction > 0.0
                || mConfig.adaptiveAdmissionExternalPrefillTokens > 0,
            "Predictive external admission profile requires a positive workload threshold");
    }
    ELLM_CHECK(static_cast<bool>(mAdapter.submitSampling), "Independent phase server requires a sampling adapter");
    mCoordinator.setGraphCaptureLimits(mConfig.maxPrefillGraphs, mConfig.maxDecodeGraphs);
    mCoordinator.setGraphCaptureEnabled(mConfig.enableCudaGraphs);
    mCoordinator.setCallbacks(makeCallbacks());
    mAdaptiveAdmissionLimit
        = mConfig.enableAdaptiveAdmission ? mConfig.latencyInFlightRequests : mConfig.maxInFlightRequests;
    if (mConfig.enableStepwiseAdaptiveAdmission && !mConfig.adaptiveAdmissionCosts.empty()
        && mConfig.adaptiveAdmissionTpotBudgetUs > 0.0)
    {
        mAdaptiveAdmissionLimit = phaseAdmissionLimitForTpotBudget(mConfig.adaptiveAdmissionCosts,
            mConfig.latencyInFlightRequests, mConfig.maxInFlightRequests, mConfig.adaptiveAdmissionTpotBudgetUs);
        mThroughputMode = mAdaptiveAdmissionLimit > mConfig.latencyInFlightRequests;
    }
    mCoordinator.scheduler().setDecodeComponentObservationActive(!mConfig.enableAdaptiveAdmission || mThroughputMode);
}

IndependentPhaseAsyncServer::~IndependentPhaseAsyncServer() noexcept
{
    for (auto& ticket : mSamplingTickets)
    {
        destroyTicketEvent(*ticket);
    }
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submit(
    uint64_t requestId, std::vector<int32_t> promptTokens, int32_t maxOutputTokens, PhaseSchedulingHints scheduling)
{
    recordTimeline(requestId, PhaseTimelineStage::kServerSubmit);
    return submitImpl(requestId, std::move(promptTokens), nullptr, maxOutputTokens, scheduling);
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitWithVision(uint64_t requestId,
    std::vector<int32_t> promptTokens, std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens,
    PhaseSchedulingHints scheduling)
{
    ELLM_CHECK(visionPayload != nullptr, "Vision phase submission requires an encoded payload");
    recordTimeline(requestId, PhaseTimelineStage::kServerSubmit);
    return submitImpl(requestId, std::move(promptTokens), std::move(visionPayload), maxOutputTokens, scheduling);
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitVisionPrefix(uint64_t requestId,
    std::vector<int32_t> prefixTokens, int32_t estimatedFinalPromptTokens, int32_t maxOutputTokens,
    PhaseSchedulingHints scheduling)
{
    ELLM_CHECK(!prefixTokens.empty(), "Deferred vision prefix must not be empty");
    ELLM_CHECK(estimatedFinalPromptTokens >= static_cast<int32_t>(prefixTokens.size()),
        "Deferred vision prompt estimate cannot be shorter than its prefix");
    recordTimeline(requestId, PhaseTimelineStage::kServerSubmit);
    return submitImpl(
        requestId, std::move(prefixTokens), nullptr, maxOutputTokens, scheduling, estimatedFinalPromptTokens, true);
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::attachVisionSuffix(
    uint64_t requestId, std::vector<int32_t> promptTokens, std::shared_ptr<PhaseVisionPayload> visionPayload)
{
    IndependentPhaseServerSubmission result{requestId};
    auto const found = mRequests.find(requestId);
    if (found == mRequests.end() || !found->second.awaitingVisionPayload || visionPayload == nullptr)
    {
        result.status = IndependentPhaseServerStatus::kDuplicateRequest;
        return result;
    }
    RequestState& state = found->second;
    ELLM_CHECK(promptTokens.size() >= state.promptTokens.size()
            && std::equal(state.promptTokens.begin(), state.promptTokens.end(), promptTokens.begin()),
        "Expanded vision prompt does not preserve the prefilled text prefix");
    state.pendingVisionPromptTokens = std::move(promptTokens);
    state.pendingVisionPayload = std::move(visionPayload);
    result.status = IndependentPhaseServerStatus::kAdmitted;
    result.kvSlotId = state.kvSlotId;
    return result;
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitImpl(uint64_t requestId,
    std::vector<int32_t> promptTokens, std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens,
    PhaseSchedulingHints scheduling, int32_t reservationPromptTokens, bool deferredVisionPrefix)
{
    bool const allowChunkedPrefill = visionPayload == nullptr || mConfig.allowChunkedVisionPrefill;
    bool const exclusivePrefill = visionPayload != nullptr && !mConfig.allowBatchedVisionPrefill;
    PhasePrefillClass const prefillClass
        = visionPayload == nullptr ? PhasePrefillClass::kText : PhasePrefillClass::kExternal;
    IndependentPhaseServerSubmission result{requestId};
    if (mRequests.find(requestId) != mRequests.end() || mPendingRequestIds.find(requestId) != mPendingRequestIds.end()
        || promptTokens.empty())
    {
        result.status = IndependentPhaseServerStatus::kDuplicateRequest;
        return result;
    }
    if (mRequests.size() >= admissionLimit() || mOwnership.availableSlots() == 0)
    {
        return result;
    }
    maxOutputTokens = maxOutputTokens > 0 ? maxOutputTokens : mConfig.defaultMaxOutputTokens;

    IndependentPhasePageReservation reservation;
    int32_t const reservedPromptTokens
        = reservationPromptTokens > 0 ? reservationPromptTokens : static_cast<int32_t>(promptTokens.size());
    if (mConfig.pageReservationMode == IndependentPhasePageReservationMode::kHeadroom)
    {
        reservation = makePageReservation(requestId, reservedPromptTokens, maxOutputTokens);
        if (!hasPageReservationCapacity(reservation))
        {
            return result;
        }
    }

    int32_t const slot = mOwnership.reserve();
    int32_t reusedPrefixTokens{};
    try
    {
        if (mConfig.enablePrefixReuse && mAdapter.supportsPageAlignedPrefixReuse && mPrefixCache != nullptr)
        {
            if (auto const match = mPrefixCache->lookup(promptTokens); match.has_value())
            {
                mOwnership.sharePrefix(match->sourceSlot, slot, match->matchedTokens);
                reusedPrefixTokens = match->matchedTokens;
            }
        }
        int32_t const reservedOutput = mConfig.pageReservationMode == IndependentPhasePageReservationMode::kFull
            ? maxOutputTokens
            : std::min(maxOutputTokens, mConfig.outputHeadroomTokens);
        mOwnership.ensureCapacity(slot, reservedPromptTokens + reservedOutput);
    }
    catch (std::runtime_error const&)
    {
        mOwnership.release(slot);
        return result;
    }

    RequestState state;
    state.promptTokens = std::move(promptTokens);
    state.maxOutputTokens = maxOutputTokens;
    state.kvSlotId = slot;
    state.scheduling = scheduling;
    state.submittedAt = std::chrono::steady_clock::now();
    state.externalProducer = visionPayload != nullptr || deferredVisionPrefix;
    state.visionPayload = std::move(visionPayload);
    state.baseReservedPages = reservation.basePages;
    state.fullReservedPages = reservation.fullPages;
    state.awaitingVisionPayload = deferredVisionPrefix;
    mRequests.emplace(requestId, std::move(state));
    mActivePromptTokens.insert(static_cast<int32_t>(mRequests.at(requestId).promptTokens.size()));
    mActiveOutputTokens.insert(maxOutputTokens);
    if (mConfig.pageReservationMode == IndependentPhasePageReservationMode::kHeadroom)
    {
        refreshPageGrowthOwners();
    }
    int32_t const remaining = static_cast<int32_t>(mRequests.at(requestId).promptTokens.size()) - reusedPrefixTokens;
    mCoordinator.enqueuePrefill({requestId, remaining, slot, reusedPrefixTokens,
        static_cast<int32_t>(mRequests.at(requestId).promptTokens.size()), allowChunkedPrefill, scheduling,
        exclusivePrefill, prefillClass});
    result.status = IndependentPhaseServerStatus::kAdmitted;
    result.kvSlotId = slot;
    result.reusedPrefixTokens = reusedPrefixTokens;
    recordTimeline(requestId, PhaseTimelineStage::kServerAdmit, slot);
    return result;
}

void IndependentPhaseAsyncServer::activateVisionSuffix(uint64_t requestId, RequestState& state)
{
    ELLM_CHECK(state.awaitingVisionPayload && state.visionPrefixComplete && state.pendingVisionPayload != nullptr,
        "Deferred vision suffix is not ready for activation");
    int32_t const prefixTokens = static_cast<int32_t>(state.promptTokens.size());
    int32_t const finalPromptTokens = static_cast<int32_t>(state.pendingVisionPromptTokens.size());
    ELLM_CHECK(finalPromptTokens > prefixTokens, "Deferred vision suffix must extend its text prefix");
    int32_t const reservedOutput = mConfig.pageReservationMode == IndependentPhasePageReservationMode::kFull
        ? state.maxOutputTokens
        : std::min(state.maxOutputTokens, mConfig.outputHeadroomTokens);
    mOwnership.ensureCapacity(state.kvSlotId, finalPromptTokens + reservedOutput);
    auto const promptLength = mActivePromptTokens.find(prefixTokens);
    ELLM_CHECK(promptLength != mActivePromptTokens.end(), "Active prompt-length index is inconsistent");
    mActivePromptTokens.erase(promptLength);
    state.promptTokens = std::move(state.pendingVisionPromptTokens);
    mActivePromptTokens.insert(finalPromptTokens);
    state.visionPayload = std::move(state.pendingVisionPayload);
    state.awaitingVisionPayload = false;
    bool const allowChunkedPrefill = mConfig.allowChunkedVisionPrefill;
    bool const exclusivePrefill = !mConfig.allowBatchedVisionPrefill;
    mCoordinator.enqueuePrefill({requestId, finalPromptTokens - prefixTokens, state.kvSlotId, prefixTokens,
        finalPromptTokens, allowChunkedPrefill, state.scheduling, exclusivePrefill, PhasePrefillClass::kExternal});
}

bool IndependentPhaseAsyncServer::activateReadyVisionSuffixes()
{
    bool activated{};
    for (auto& [requestId, state] : mRequests)
    {
        if (state.awaitingVisionPayload && state.visionPrefixComplete && state.pendingVisionPayload != nullptr)
        {
            activateVisionSuffix(requestId, state);
            activated = true;
        }
    }
    return activated;
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitOrQueue(
    uint64_t requestId, std::vector<int32_t> promptTokens, int32_t maxOutputTokens, PhaseSchedulingHints scheduling)
{
    recordTimeline(requestId, PhaseTimelineStage::kServerSubmit);
    return submitOrQueueImpl(requestId, std::move(promptTokens), nullptr, maxOutputTokens, scheduling);
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitOrQueueWithVision(uint64_t requestId,
    std::vector<int32_t> promptTokens, std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens,
    PhaseSchedulingHints scheduling)
{
    ELLM_CHECK(visionPayload != nullptr, "Queued vision phase submission requires an encoded payload");
    recordTimeline(requestId, PhaseTimelineStage::kServerSubmit);
    return submitOrQueueImpl(requestId, std::move(promptTokens), std::move(visionPayload), maxOutputTokens, scheduling);
}

IndependentPhaseServerSubmission IndependentPhaseAsyncServer::submitOrQueueImpl(uint64_t requestId,
    std::vector<int32_t> promptTokens, std::shared_ptr<PhaseVisionPayload> visionPayload, int32_t maxOutputTokens,
    PhaseSchedulingHints scheduling)
{
    IndependentPhaseServerSubmission result
        = submitImpl(requestId, promptTokens, visionPayload, maxOutputTokens, scheduling);
    if (result.status != IndependentPhaseServerStatus::kBackpressure)
    {
        return result;
    }
    if (mConfig.maxPendingRequests == 0 || mPendingRequests.size() >= mConfig.maxPendingRequests
        || mPendingRequestIds.find(requestId) != mPendingRequestIds.end()
        || mRequests.find(requestId) != mRequests.end() || promptTokens.empty())
    {
        return result;
    }
    if (scheduling.submittedAt == std::chrono::steady_clock::time_point{})
    {
        scheduling.submittedAt = std::chrono::steady_clock::now();
    }
    mPendingRequests.push_back(
        {requestId, std::move(promptTokens), maxOutputTokens, scheduling, std::move(visionPayload)});
    mPendingRequestIds.insert(requestId);
    result.status = IndependentPhaseServerStatus::kQueued;
    return result;
}

bool IndependentPhaseAsyncServer::cancel(uint64_t requestId)
{
    auto it = mRequests.find(requestId);
    if (it == mRequests.end())
    {
        if (mPendingRequestIds.erase(requestId) == 0)
        {
            return false;
        }
        auto const pending = std::find_if(mPendingRequests.begin(), mPendingRequests.end(),
            [&](PendingRequest const& request) { return request.requestId == requestId; });
        ELLM_CHECK(pending != mPendingRequests.end(), "Pending request index is inconsistent");
        mPendingRequests.erase(pending);
        return true;
    }
    if (mPendingDecodeRequestIds.erase(requestId) > 0)
    {
        auto const waiting = std::find(mPendingDecodeRequests.begin(), mPendingDecodeRequests.end(), requestId);
        ELLM_CHECK(waiting != mPendingDecodeRequests.end(), "Pending decode request index is inconsistent");
        mPendingDecodeRequests.erase(waiting);
    }
    else if (!mCoordinator.scheduler().cancel(requestId))
    {
        return false;
    }
    mOwnership.release(it->second.kvSlotId);
    auto const promptLength = mActivePromptTokens.find(static_cast<int32_t>(it->second.promptTokens.size()));
    ELLM_CHECK(promptLength != mActivePromptTokens.end(), "Active prompt-length index is inconsistent");
    mActivePromptTokens.erase(promptLength);
    auto const outputLength = mActiveOutputTokens.find(it->second.maxOutputTokens);
    ELLM_CHECK(outputLength != mActiveOutputTokens.end(), "Active output-length index is inconsistent");
    mActiveOutputTokens.erase(outputLength);
    mRequests.erase(it);
    if (mConfig.pageReservationMode == IndependentPhasePageReservationMode::kHeadroom)
    {
        refreshPageGrowthOwners();
    }
    return true;
}

bool IndependentPhaseAsyncServer::capturePreparedGraphs()
{
    ELLM_CHECK(
        mRequests.empty() && mSamplingTickets.empty(), "Phase graphs cannot be captured while requests are active");
    return mCoordinator.capturePreparedGraphs();
}

void IndependentPhaseAsyncServer::setEventCallbacks(std::function<void(IndependentPhaseServerToken&&)> tokenCallback,
    std::function<void(IndependentPhaseServerCompletion&&)> completionCallback)
{
    ELLM_CHECK(mRequests.empty() && mPendingRequests.empty() && mSamplingTickets.empty(),
        "Phase event callbacks can only change while the server is idle");
    mTokenCallback = std::move(tokenCallback);
    mCompletionCallback = std::move(completionCallback);
    static_cast<void>(flushEventCallbacks());
}

void IndependentPhaseAsyncServer::setTimelineCallback(std::function<void(PhaseTimelineEvent const&)> timelineCallback)
{
    ELLM_CHECK(mRequests.empty() && mPendingRequests.empty() && mSamplingTickets.empty(),
        "Phase timeline callback can only change while the server is idle");
    mTimelineCallback = std::move(timelineCallback);
}

void IndependentPhaseAsyncServer::setActivityTimeline(PhaseActivityTimelineRecorder* timeline)
{
    ELLM_CHECK(mRequests.empty() && mPendingRequests.empty() && mSamplingTickets.empty(),
        "Phase activity timeline can only change while the server is idle");
    mActivityTimeline = timeline;
    mCoordinator.setActivityTimeline(timeline);
}

void IndependentPhaseAsyncServer::setCompletionAttributionEnabled(bool enabled) noexcept
{
    mCoordinator.scheduler().setCompletionAttributionEnabled(enabled);
}

void IndependentPhaseAsyncServer::setDirectionalInjectionControl(PhaseDirectionalInjectionControl control)
{
    ELLM_CHECK(mRequests.empty() && mPendingRequests.empty() && mSamplingTickets.empty(),
        "Phase directional injection control can only change while the server is idle");
    mCoordinator.setDirectionalInjectionControl(control);
}

cudaStream_t IndependentPhaseAsyncServer::phaseStream(PhaseUnifiedPhase phase) const noexcept
{
    return mCoordinator.phaseStream(phase);
}

cudaEvent_t IndependentPhaseAsyncServer::phaseStartEvent(PhaseUnifiedPhase phase) const noexcept
{
    return mCoordinator.phaseStartEvent(phase);
}

void IndependentPhaseAsyncServer::setNextDispatchPreamble(
    PhaseUnifiedPhase phase, std::function<void(cudaStream_t)> preamble)
{
    mCoordinator.setNextDispatchPreamble(phase, std::move(preamble));
}

bool IndependentPhaseAsyncServer::poll()
{
    bool progressed = pollCompletions();
    progressed = dispatchReady() || progressed;
    progressed = flushEventCallbacks() || progressed;
    return progressed;
}

bool IndependentPhaseAsyncServer::pollCompletions()
{
    updateAdaptiveAdmissionMode();
    bool progressed = admitPendingRequests();
    progressed = resumePendingDecodeRequests() || progressed;
    progressed = mCoordinator.poll() || progressed;
    progressed = activateReadyVisionSuffixes() || progressed;
    progressed = processSamplingTickets() || progressed;
    progressed = resumePendingDecodeRequests() || progressed;
    progressed = admitPendingRequests() || progressed;
    updateAdaptiveAdmissionMode();
    progressed = flushEventCallbacks() || progressed;
    return progressed;
}

bool IndependentPhaseAsyncServer::dispatchReady()
{
    bool progressed = tryAugmentReadyAction();
    size_t producerTextRows{};
    size_t producerExternalRows{};
    double producerWaitUs{};
    double producerUncertaintyUs{};
    uint64_t producerEventId{};
    // Only a length-certain decode completion can turn an admission-waiting
    // request into a concrete future P row. Adapter backlog and possible EOS
    // completions are intentionally excluded. Prefix sharing and incremental
    // page reservation require a refcount/reservation-aware forecast, so this
    // first implementation remains disabled for those mechanisms.
    if (mConfig.enableCompletionAwareAdmissionProjection && !mPendingRequests.empty() && !mConfig.enablePrefixReuse
        && mConfig.pageReservationMode == IndependentPhasePageReservationMode::kFull)
    {
        std::vector<double> ordered(mSamplingLatencyUs.begin(), mSamplingLatencyUs.end());
        std::sort(ordered.begin(), ordered.end());
        double samplingMedianUs = mConfig.globalSamplingColdStartUs;
        double samplingP95Us = mConfig.globalSamplingColdStartUs * 2.0;
        if (!ordered.empty())
        {
            samplingMedianUs = ordered[(ordered.size() - 1U) / 2U];
            size_t const p95Index = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(ordered.size()))) - 1U;
            samplingP95Us = ordered[std::min(p95Index, ordered.size() - 1U)];
        }
        auto const now = std::chrono::steady_clock::now();
        for (auto const& ticket : mSamplingTickets)
        {
            std::vector<int32_t> completingSlots;
            for (uint64_t const requestId : ticket->requestIds)
            {
                auto const request = mRequests.find(requestId);
                if (request != mRequests.end()
                    && request->second.generatedTokens.size() + 1U
                        >= static_cast<size_t>(request->second.maxOutputTokens))
                {
                    completingSlots.push_back(request->second.kvSlotId);
                }
            }
            double const ageUs = ticket->submittedAt == std::chrono::steady_clock::time_point{}
                ? 0.0
                : std::chrono::duration<double, std::micro>(now - ticket->submittedAt).count();
            producerWaitUs = std::max(producerWaitUs, std::max(0.0, samplingMedianUs - ageUs));
            double const producerP95Us = std::max(producerWaitUs, samplingP95Us - ageUs);
            producerUncertaintyUs = std::max(producerUncertaintyUs, producerP95Us - producerWaitUs);
            if (completingSlots.empty())
            {
                continue;
            }

            size_t const activeAfterCompletion = mRequests.size() - completingSlots.size();
            size_t const admissionCapacity
                = admissionLimit() > activeAfterCompletion ? admissionLimit() - activeAfterCompletion : 0U;
            size_t const physicalCapacity = static_cast<size_t>(mOwnership.availableSlots()) + completingSlots.size();
            size_t remainingRows = std::min({mPendingRequests.size(), admissionCapacity, physicalCapacity});
            int32_t availablePages = mOwnership.availablePages();
            for (int32_t const slot : completingSlots)
            {
                availablePages += static_cast<int32_t>(mOwnership.pages(slot).size());
            }
            int32_t const tokensPerPage = mOwnership.config().tokensPerPage;
            for (PendingRequest const& request : mPendingRequests)
            {
                if (remainingRows == 0U)
                {
                    break;
                }
                int32_t const maxOutputTokens
                    = request.maxOutputTokens > 0 ? request.maxOutputTokens : mConfig.defaultMaxOutputTokens;
                int32_t const requestedTokens = static_cast<int32_t>(request.promptTokens.size()) + maxOutputTokens;
                int32_t const requestedPages = (requestedTokens + tokensPerPage - 1) / tokensPerPage;
                if (requestedPages > availablePages)
                {
                    break;
                }
                availablePages -= requestedPages;
                if (request.visionPayload == nullptr)
                {
                    ++producerTextRows;
                }
                else
                {
                    ++producerExternalRows;
                }
                --remainingRows;
            }
            producerEventId = ticket->sequenceId;
            break;
        }
    }
    mCoordinator.scheduler().setPendingPrefillProducerRows(
        producerTextRows, producerExternalRows, producerWaitUs, producerUncertaintyUs, producerEventId);
    bool const waitForDecodeRefill = shouldWaitForGlobalDecodeRefill();
    if (waitForDecodeRefill)
    {
        ++mDecodeRefillWaitCount;
    }
    bool const profileFreeGlobal = mCoordinator.scheduler().globalSchedulerMode() == PhaseGlobalSchedulerMode::kActive
        && mCoordinator.scheduler().globalSelectionMode() == PhaseGlobalSelectionMode::kProfileFree;
    bool const waitForPrefillFormation = profileFreeGlobal ? false : shouldWaitForPrefillFormation();
    bool const phaseQueued
        = mCoordinator.scheduler().prefillQueueSize() > 0U || mCoordinator.scheduler().decodeQueueSize() > 0U;
    if (!mCoordinator.busy() && phaseQueued && !waitForDecodeRefill)
    {
        if (waitForPrefillFormation)
        {
            mCoordinator.scheduler().setPrefillDispatchBlocked(true);
        }
        progressed = mCoordinator.dispatchNext() || progressed;
        if (waitForPrefillFormation)
        {
            mCoordinator.scheduler().setPrefillDispatchBlocked(false);
        }
        progressed = tryAugmentReadyAction() || progressed;
    }
    return progressed;
}

bool IndependentPhaseAsyncServer::tryAugmentReadyAction()
{
    if (!mCoordinator.busy() || mCoordinator.inFlightKind() == PhaseDispatchKind::kOverlap)
    {
        if (!mCoordinator.busy())
        {
            mResidualIncumbentCandidateId = 0U;
            mResidualMissingCandidateId = 0U;
            mResidualPrefillQueued = std::numeric_limits<size_t>::max();
            mResidualDecodeQueued = std::numeric_limits<size_t>::max();
        }
        return false;
    }
    PhaseGlobalActionCandidate const* const incumbent = mCoordinator.inFlightGlobalCandidate();
    if (incumbent == nullptr)
    {
        return false;
    }
    if (incumbent->candidateId != mResidualIncumbentCandidateId)
    {
        mResidualIncumbentCandidateId = incumbent->candidateId;
        mResidualMissingCandidateId = 0U;
        mResidualPrefillQueued = std::numeric_limits<size_t>::max();
        mResidualDecodeQueued = std::numeric_limits<size_t>::max();
    }
    PhaseQueueSnapshot const queue = mCoordinator.scheduler().queueSnapshot();
    if (queue.prefillQueued == mResidualPrefillQueued && queue.decodeQueued == mResidualDecodeQueued)
    {
        return false;
    }
    mResidualPrefillQueued = queue.prefillQueued;
    mResidualDecodeQueued = queue.decodeQueued;
    PhaseInFlightSnapshot const inFlight = mCoordinator.inFlightSnapshot();
    double elapsedUs{};
    for (PhaseInFlightWorkSnapshot const& work : inFlight.work)
    {
        elapsedUs = std::max(elapsedUs, work.dispatchAgeUs);
    }
    std::optional<PhaseGlobalResidualSelection> residual
        = mCoordinator.scheduler().previewGlobalResidualAction(*incumbent, elapsedUs);
    if (!residual.has_value() || residual->missingPhase.candidateId == mResidualMissingCandidateId)
    {
        return false;
    }
    mResidualMissingCandidateId = residual->missingPhase.candidateId;
    if (!residual->selected)
    {
        return false;
    }
    bool const started
        = mCoordinator.augmentGlobalAction(std::move(residual->missingPhase), std::move(residual->aggregate), 0U, 0U);
    if (started)
    {
        mResidualIncumbentCandidateId = 0U;
        mResidualMissingCandidateId = 0U;
        mResidualPrefillQueued = std::numeric_limits<size_t>::max();
        mResidualDecodeQueued = std::numeric_limits<size_t>::max();
    }
    return started;
}

std::optional<PhaseGlobalActionCandidate> IndependentPhaseAsyncServer::previewGlobalAction()
{
    if (mCoordinator.busy())
    {
        return std::nullopt;
    }
    return mCoordinator.scheduler().previewGlobalAction();
}

std::vector<PhaseGlobalActionCandidate> const& IndependentPhaseAsyncServer::lastGlobalPreviewCandidates() const noexcept
{
    return mCoordinator.scheduler().lastGlobalPreviewCandidates();
}

std::optional<PhaseGlobalActionCandidate> IndependentPhaseAsyncServer::previewGlobalPrefillAction()
{
    if (mCoordinator.busy() && mCoordinator.inFlightKind() != PhaseDispatchKind::kDecode)
    {
        return std::nullopt;
    }
    return mCoordinator.scheduler().previewGlobalPrefillAction();
}

std::optional<PhaseGlobalActionCandidate> IndependentPhaseAsyncServer::previewGlobalDecodeAction()
{
    if (mCoordinator.busy() && mCoordinator.inFlightKind() != PhaseDispatchKind::kPrefill)
    {
        return std::nullopt;
    }
    return mCoordinator.scheduler().previewGlobalDecodeAction();
}

PhaseGlobalCostEstimate IndependentPhaseAsyncServer::estimateGlobalPrefillCost(
    int32_t batchSize, int32_t chunkLength, int32_t pastKVLength, PhasePrefillClass prefillClass) const
{
    return mCoordinator.scheduler().estimateGlobalPrefillCost(batchSize, chunkLength, pastKVLength, prefillClass);
}

PhaseGlobalCostEstimate IndependentPhaseAsyncServer::estimateGlobalPrefillDrainCost(
    int32_t batchSize, int32_t promptTokens, PhasePrefillClass prefillClass) const
{
    return mCoordinator.scheduler().estimateGlobalPrefillDrainCost(batchSize, promptTokens, prefillClass);
}

std::optional<float> IndependentPhaseAsyncServer::estimateGlobalDecodeComponentP95(
    int32_t batchSize, int32_t maxContextLength, bool prefillActive) const
{
    return mCoordinator.scheduler().estimateGlobalDecodeComponentP95(batchSize, maxContextLength, prefillActive);
}

bool IndependentPhaseAsyncServer::dispatchGlobalAction(
    PhaseGlobalActionCandidate candidate, uint64_t planId, uint64_t snapshotEpoch)
{
    if (mCoordinator.busy())
    {
        return false;
    }
    mCoordinator.scheduler().setNextGlobalAction(std::move(candidate), planId, snapshotEpoch);
    return mCoordinator.dispatchNext();
}

bool IndependentPhaseAsyncServer::augmentGlobalAction(PhaseGlobalActionCandidate missingPhase,
    PhaseGlobalActionCandidate aggregate, uint64_t planId, uint64_t snapshotEpoch)
{
    if (!mCoordinator.busy())
    {
        return false;
    }
    return mCoordinator.augmentGlobalAction(std::move(missingPhase), std::move(aggregate), planId, snapshotEpoch);
}

bool IndependentPhaseAsyncServer::shouldWaitForPrefillFormation()
{
    if (mCoordinator.busy())
    {
        return false;
    }
    PhaseQueueSnapshot const queue = mCoordinator.scheduler().queueSnapshot();
    size_t const pendingProducerRows = mExternalPendingRequests + mPendingAdapterRequests;
    int32_t const cohortMaxPromptTokens = mActivePromptTokens.empty() ? 0 : *mActivePromptTokens.rbegin();
    int32_t const cohortMaxOutputTokens = mActiveOutputTokens.empty() ? 0 : *mActiveOutputTokens.rbegin();
    size_t targetBatchSize = mConfig.prefillFormationBatchSize;
    double formationWindowUs = mConfig.prefillFormationWindowUs;
    double ttftTargetUs = mConfig.prefillFormationTtftTargetUs;
    double ttftGuardUs = mConfig.prefillFormationTtftGuardUs;
    if (!mConfig.prefillFormationCosts.empty())
    {
        std::optional<size_t> const costIndex = phasePrefillFormationCostIndex(mConfig.prefillFormationCosts,
            cohortMaxPromptTokens, cohortMaxOutputTokens, queue.decodeQueued,
            mCoordinator.scheduler().telemetry().recentDecodeTpotPressure);
        if (!costIndex)
        {
            if (queue.prefillQueued > 0 && pendingProducerRows > 0)
            {
                ++mPrefillFormationProfileMissCount;
            }
            mPrefillFormationStartedAt.reset();
            return false;
        }
        IndependentPhaseFormationCost const& cost = mConfig.prefillFormationCosts[*costIndex];
        targetBatchSize = cost.targetBatchSize;
        formationWindowUs = cost.windowUs;
        ttftTargetUs = cost.ttftTargetUs;
        ttftGuardUs = cost.ttftGuardUs;
    }
    else if (!phasePrefillFormationSupportsOutputLength(mConfig.prefillFormationMaxOutputTokens, cohortMaxOutputTokens))
    {
        mPrefillFormationStartedAt.reset();
        return false;
    }
    double const minTtftSlackUs
        = ttftTargetUs > 0.0 ? ttftTargetUs - queue.prefillOldestRequestAgeUs : queue.prefillMinTtftSlackUs;
    auto const now = std::chrono::steady_clock::now();
    if (!mPrefillFormationStartedAt)
    {
        bool const candidate = shouldDeferPrefillForMicrobatchFormation(targetBatchSize, queue.prefillQueued,
            pendingProducerRows, minTtftSlackUs, ttftGuardUs, 0.0, formationWindowUs);
        if (!candidate)
        {
            return false;
        }
        mPrefillFormationStartedAt = now;
        ++mPrefillFormationWaitPeriodCount;
        if (!mConfig.prefillFormationCosts.empty())
        {
            ++mPrefillFormationProfileSelectionCount;
        }
    }
    double const elapsedUs = std::chrono::duration<double, std::micro>(now - *mPrefillFormationStartedAt).count();
    bool const defer = shouldDeferPrefillForMicrobatchFormation(targetBatchSize, queue.prefillQueued,
        pendingProducerRows, minTtftSlackUs, ttftGuardUs, elapsedUs, formationWindowUs);
    if (defer)
    {
        ++mPrefillFormationDeferralCount;
    }
    else
    {
        mPrefillFormationStartedAt.reset();
    }
    return defer;
}

bool IndependentPhaseAsyncServer::shouldWaitForGlobalDecodeRefill()
{
    size_t const queuedDecode = mCoordinator.scheduler().decodeQueueSize();
    size_t pendingDecodeRows{};
    for (auto const& ticket : mSamplingTickets)
    {
        if (!ticket->fromPrefill)
        {
            pendingDecodeRows += ticket->requestIds.size();
        }
    }
    bool const retainGrowthCohort = mConfig.pageReservationMode == IndependentPhasePageReservationMode::kHeadroom
        && !mPageGrowthRequestIds.empty();
    size_t const refillTarget = !mConfig.enableAdaptiveAdmission || mThroughputMode || retainGrowthCohort
        ? mConfig.decodeRefillBatchSize
        : 0U;
    bool const legacyWait = shouldDeferDecodeForSamplingRefill(
        refillTarget, mCoordinator.scheduler().prefillQueueSize(), queuedDecode, pendingDecodeRows);
    if (!mConfig.enableGlobalWaitActions || mSamplingTickets.empty()
        || mCoordinator.scheduler().globalSchedulerMode() == PhaseGlobalSchedulerMode::kDisabled)
    {
        return legacyWait;
    }
    if (mCoordinator.scheduler().prefillQueueSize() > 0U || queuedDecode == 0U || pendingDecodeRows == 0U)
    {
        return false;
    }

    std::vector<PhaseDecodeCompletionPreview> const previews = previewPendingDecodeCompletions();
    if (previews.empty())
    {
        return false;
    }
    bool const globalWait = mCoordinator.scheduler().shouldWaitForDecodeEvents(previews);
    bool const useLegacyWait = mCoordinator.scheduler().globalSchedulerMode() == PhaseGlobalSchedulerMode::kShadow
        || mCoordinator.scheduler().globalSelectionMode() == PhaseGlobalSelectionMode::kLegacyCompatibility;
    if (useLegacyWait)
    {
        return legacyWait;
    }
    return mConfig.enableGlobalWaitAuthority && globalWait;
}

std::vector<PhaseDecodeCompletionPreview> IndependentPhaseAsyncServer::previewPendingDecodeCompletions(
    size_t maxPreviews) const
{
    std::vector<PhaseDecodeCompletionPreview> previews;
    if (maxPreviews == 0U || mSamplingTickets.empty())
    {
        return previews;
    }

    std::vector<double> ordered(mSamplingLatencyUs.begin(), mSamplingLatencyUs.end());
    std::sort(ordered.begin(), ordered.end());
    double medianUs = mConfig.globalSamplingColdStartUs;
    double p95Us = mConfig.globalSamplingColdStartUs * 2.0;
    if (!ordered.empty())
    {
        medianUs = ordered[(ordered.size() - 1U) / 2U];
        size_t const p95Index = static_cast<size_t>(std::ceil(0.95 * static_cast<double>(ordered.size()))) - 1U;
        p95Us = ordered[std::min(p95Index, ordered.size() - 1U)];
    }

    auto const now = std::chrono::steady_clock::now();
    previews.reserve(maxPreviews);
    std::vector<uint64_t> cumulativeRequestIds;
    std::vector<int32_t> cumulativeContextLengths;
    std::vector<int32_t> cumulativeStableSlotIds;
    std::vector<PhaseSchedulingHints> cumulativeSchedulingHints;
    std::unordered_set<uint64_t> cumulativeOwners;
    double cumulativeWaitUs{};
    double cumulativeP95Us{};
    for (auto const& ticket : mSamplingTickets)
    {
        if (ticket->fromPrefill)
        {
            continue;
        }
        double const ageUs = ticket->submittedAt == std::chrono::steady_clock::time_point{}
            ? 0.0
            : std::chrono::duration<double, std::micro>(now - ticket->submittedAt).count();
        double const remainingMedianUs = std::max(0.0, medianUs - ageUs);
        double const remainingP95Us = std::max(remainingMedianUs, p95Us - ageUs);
        cumulativeWaitUs = std::max(cumulativeWaitUs, remainingMedianUs);
        cumulativeP95Us = std::max(cumulativeP95Us, remainingP95Us);
        for (uint64_t const requestId : ticket->requestIds)
        {
            auto const request = mRequests.find(requestId);
            if (request != mRequests.end()
                && request->second.generatedTokens.size() + 1U < static_cast<size_t>(request->second.maxOutputTokens)
                && cumulativeOwners.insert(requestId).second)
            {
                cumulativeRequestIds.push_back(requestId);
                cumulativeContextLengths.push_back(mOwnership.length(request->second.kvSlotId));
                cumulativeStableSlotIds.push_back(request->second.kvSlotId);
                cumulativeSchedulingHints.push_back(request->second.scheduling);
            }
        }
        if (!cumulativeRequestIds.empty())
        {
            previews.push_back({ticket->sequenceId, cumulativeWaitUs, std::max(0.0, cumulativeP95Us - cumulativeWaitUs),
                cumulativeRequestIds, cumulativeContextLengths, cumulativeStableSlotIds, cumulativeSchedulingHints});
            double serviceBudgetUs = mCoordinator.scheduler().defaultDecodeTpotTargetUs();
            for (PhaseSchedulingHints const& hints : cumulativeSchedulingHints)
            {
                if (hints.tpotTargetUs > 0.0)
                {
                    serviceBudgetUs = std::min(serviceBudgetUs, hints.tpotTargetUs);
                }
            }
            previews.back().serviceBudgetUs = serviceBudgetUs;
        }
        if (previews.size() == maxPreviews)
        {
            break;
        }
    }
    return previews;
}

size_t IndependentPhaseAsyncServer::admissionLimit() const noexcept
{
    if (mCoordinator.scheduler().globalSchedulerMode() == PhaseGlobalSchedulerMode::kActive)
    {
        double targetUs = mConfig.adaptiveAdmissionTpotBudgetUs;
        auto includeTarget = [&](double candidateUs) {
            if (candidateUs > 0.0 && (targetUs == 0.0 || candidateUs < targetUs))
            {
                targetUs = candidateUs;
            }
        };
        includeTarget(mExternalMinTpotTargetUs);
        int32_t maxContextLength{};
        for (auto const& request : mRequests)
        {
            includeTarget(request.second.scheduling.tpotTargetUs);
            maxContextLength = std::max(maxContextLength,
                static_cast<int32_t>(request.second.promptTokens.size()) + request.second.maxOutputTokens);
        }
        for (PendingRequest const& request : mPendingRequests)
        {
            includeTarget(request.scheduling.tpotTargetUs);
            maxContextLength = std::max(
                maxContextLength, static_cast<int32_t>(request.promptTokens.size()) + request.maxOutputTokens);
        }
        if (targetUs > 0.0)
        {
            size_t const decodeLimit = mCoordinator.scheduler().decodeAdmissionLimitForTpot(targetUs, maxContextLength);
            return std::min(mConfig.maxInFlightRequests, std::max(size_t{1U}, decodeLimit));
        }
        return mConfig.maxInFlightRequests;
    }
    if (mConfig.enableStepwiseAdaptiveAdmission)
    {
        return mAdaptiveAdmissionLimit;
    }
    return !mConfig.enableAdaptiveAdmission || mThroughputMode ? mConfig.maxInFlightRequests
                                                               : mConfig.latencyInFlightRequests;
}

double IndependentPhaseAsyncServer::effectiveAdmissionTpotBudgetUs() const noexcept
{
    double result = phaseAdmissionProfileTpotBudget(mConfig.adaptiveAdmissionTpotBudgetUs,
        mConfig.adaptiveAdmissionExternalTpotBudgetUs, adaptiveAdmissionExternalProfileActive());
    auto includeTarget = [&](double targetUs) {
        if (targetUs > 0.0 && (result == 0.0 || targetUs < result))
        {
            result = targetUs;
        }
    };
    includeTarget(mExternalMinTpotTargetUs);
    for (auto const& request : mRequests)
    {
        includeTarget(request.second.scheduling.tpotTargetUs);
    }
    for (PendingRequest const& request : mPendingRequests)
    {
        includeTarget(request.scheduling.tpotTargetUs);
    }
    return result;
}

size_t IndependentPhaseAsyncServer::costLimitedAdmissionLimit() const noexcept
{
    return phaseAdmissionLimitForTpotBudget(activeAdmissionCosts(), mConfig.latencyInFlightRequests,
        mConfig.maxInFlightRequests, effectiveAdmissionTpotBudgetUs());
}

std::vector<IndependentPhaseAdmissionCost> const& IndependentPhaseAsyncServer::activeAdmissionCosts() const noexcept
{
    return adaptiveAdmissionExternalProfileActive() ? mConfig.adaptiveAdmissionExternalCosts
                                                    : mConfig.adaptiveAdmissionCosts;
}

void IndependentPhaseAsyncServer::updateAdaptiveAdmissionMode() noexcept
{
    if (!mConfig.enableAdaptiveAdmission)
    {
        return;
    }
    if (mConfig.enableStepwiseAdaptiveAdmission)
    {
        PhaseSchedulerTelemetry const& telemetry = mCoordinator.scheduler().telemetry();
        bool const externalProfile = adaptiveAdmissionExternalProfileActive();
        if (externalProfile && !mLastAdmissionExternalProfileActive)
        {
            ++mAdaptiveAdmissionExternalProfileSelectionCount;
        }
        mLastAdmissionExternalProfileActive = externalProfile;
        size_t const costLimit = costLimitedAdmissionLimit();
        double const effectiveTpotBudgetUs = effectiveAdmissionTpotBudgetUs();
        bool const predictiveAdmission = !activeAdmissionCosts().empty() && effectiveTpotBudgetUs > 0.0;
        bool const predictiveFastStart
            = predictiveAdmission && mAdaptiveAdmissionLimit < costLimit && telemetry.decodeTpotSampleCount == 0;
        if (predictiveFastStart || telemetry.sampleCount < mLastAdmissionDecisionSample
            || telemetry.sampleCount - mLastAdmissionDecisionSample >= mConfig.adaptiveAdmissionDwellSamples)
        {
            if (!phaseAdmissionTpotBudgetSatisfiable(
                    activeAdmissionCosts(), mConfig.latencyInFlightRequests, effectiveTpotBudgetUs))
            {
                ++mAdaptiveAdmissionUnsatisfiableDecisionCount;
            }
            size_t const pendingRequests = mPendingRequests.size() + mExternalPendingRequests;
            size_t next{};
            bool const observedTpotBudgetPressure = predictiveAdmission && telemetry.recentDecodeTpotP95Us > 0.0
                && telemetry.recentDecodeTpotP95Us >= effectiveTpotBudgetUs;
            if (mAdaptiveAdmissionLimit > costLimit || observedTpotBudgetPressure)
            {
                next = mAdaptiveAdmissionLimit - mConfig.latencyInFlightRequests > mConfig.adaptiveAdmissionStep
                    ? mAdaptiveAdmissionLimit - mConfig.adaptiveAdmissionStep
                    : mConfig.latencyInFlightRequests;
            }
            else
            {
                next = nextStepwiseAdmissionLimit(mAdaptiveAdmissionLimit, mConfig.latencyInFlightRequests, costLimit,
                    mConfig.adaptiveAdmissionStep, pendingRequests, mRequests.size(),
                    mConfig.adaptiveBacklogEnterRequests, mOwnership.availablePages(),
                    mConfig.adaptiveAdmissionMinFreePages, telemetry.recentDecodeTpotPressure,
                    predictiveAdmission ? 0.0F : mConfig.adaptiveAdmissionTpotPressureEnterRatio,
                    predictiveAdmission ? 0.0F : mConfig.adaptiveAdmissionTpotPressureExitRatio);
            }
            bool const costBlocked = pendingRequests >= mConfig.adaptiveBacklogEnterRequests
                && mRequests.size() >= mAdaptiveAdmissionLimit && mAdaptiveAdmissionLimit < mConfig.maxInFlightRequests
                && costLimit <= mAdaptiveAdmissionLimit;
            if (costBlocked)
            {
                ++mAdaptiveAdmissionCostBlockCount;
            }
            if (next != mAdaptiveAdmissionLimit)
            {
                if (next > mAdaptiveAdmissionLimit)
                {
                    ++mAdaptiveAdmissionIncreaseCount;
                }
                else
                {
                    ++mAdaptiveAdmissionDecreaseCount;
                }
                bool const previousThroughputMode = mThroughputMode;
                mAdaptiveAdmissionLimit = next;
                mThroughputMode = next > mConfig.latencyInFlightRequests;
                if (previousThroughputMode != mThroughputMode)
                {
                    ++mThroughputModeTransitionCount;
                }
            }
            mLastAdmissionDecisionSample = telemetry.sampleCount;
        }
        mCoordinator.scheduler().setDecodeComponentObservationActive(mThroughputMode);
        return;
    }
    bool const next = nextAdaptiveThroughputMode(mThroughputMode, mPendingRequests.size(), mRequests.size(),
        mConfig.latencyInFlightRequests, mConfig.adaptiveBacklogEnterRequests);
    if (next != mThroughputMode)
    {
        mThroughputMode = next;
        ++mThroughputModeTransitionCount;
    }
    mCoordinator.scheduler().setDecodeComponentObservationActive(mThroughputMode);
}

void IndependentPhaseAsyncServer::runUntilIdle(size_t maxPolls)
{
    for (size_t pollCount{}; pollCount < maxPolls && !empty(); ++pollCount)
    {
        if (!poll())
        {
            std::this_thread::yield();
        }
    }
    ELLM_CHECK(empty(), "Independent phase server exceeded its poll guard");
}

std::optional<IndependentPhaseServerCompletion> IndependentPhaseAsyncServer::tryPopCompletion()
{
    if (mCompletions.empty())
    {
        return std::nullopt;
    }
    IndependentPhaseServerCompletion result = std::move(mCompletions.front());
    mCompletions.pop_front();
    return result;
}

std::optional<IndependentPhaseServerToken> IndependentPhaseAsyncServer::tryPopToken()
{
    if (mTokenEvents.empty())
    {
        return std::nullopt;
    }
    IndependentPhaseServerToken result = std::move(mTokenEvents.front());
    mTokenEvents.pop_front();
    return result;
}

size_t IndependentPhaseAsyncServer::inFlightCount() const noexcept
{
    return mRequests.size();
}

size_t IndependentPhaseAsyncServer::pendingCount() const noexcept
{
    return mPendingRequests.size();
}

bool IndependentPhaseAsyncServer::hasPendingSampling() const noexcept
{
    return !mSamplingTickets.empty();
}

void IndependentPhaseAsyncServer::setExternalPendingRequests(
    size_t pendingRequests, double minTpotTargetUs, size_t externalRequests, size_t externalPrefillTokens) noexcept
{
    mExternalPendingRequests = pendingRequests;
    mExternalMinTpotTargetUs = minTpotTargetUs;
    mExternalRequests = externalRequests;
    mExternalPrefillTokens = externalPrefillTokens;
    size_t const totalRequests = mRequests.size() + mPendingRequests.size() + mExternalPendingRequests;
    if (mConfig.enableDelayedExternalProfileSelection)
    {
        mAdmissionExternalProfileEpochSelection = phaseAdmissionExternalProfileForEpoch(
            mAdmissionExternalProfileEpochSelection, mExternalRequests, totalRequests, mExternalPrefillTokens,
            mConfig.adaptiveAdmissionExternalRequestFraction, mConfig.adaptiveAdmissionExternalPrefillTokens);
    }
    else if (totalRequests == 0)
    {
        mAdmissionExternalProfileEpochSelection = std::nullopt;
    }
    else if (!mAdmissionExternalProfileEpochSelection
        && mExternalPrefillTokens >= mConfig.adaptiveAdmissionExternalPrefillTokens)
    {
        mAdmissionExternalProfileEpochSelection
            = phaseAdmissionUsesExternalProfile(mExternalRequests, totalRequests, mExternalPrefillTokens,
                mConfig.adaptiveAdmissionExternalRequestFraction, mConfig.adaptiveAdmissionExternalPrefillTokens);
    }
}

void IndependentPhaseAsyncServer::setPendingAdapterRequests(size_t pendingRequests) noexcept
{
    mPendingAdapterRequests = pendingRequests;
}

void IndependentPhaseAsyncServer::setExternalDrainPreference(PhaseDrainPreference preference) noexcept
{
    mCoordinator.scheduler().setExternalDrainPreference(preference);
}

void IndependentPhaseAsyncServer::setPrefillDispatchBlocked(bool blocked) noexcept
{
    mCoordinator.scheduler().setPrefillDispatchBlocked(blocked);
}

void IndependentPhaseAsyncServer::setDispatchBlocked(bool blocked) noexcept
{
    mCoordinator.scheduler().setDispatchBlocked(blocked);
}

void IndependentPhaseAsyncServer::setExternalEncoderActive(bool active) noexcept
{
    mCoordinator.scheduler().setExternalEncoderActive(active);
}

PhaseDrainPreference IndependentPhaseAsyncServer::activeDrainPreference() const noexcept
{
    return mCoordinator.scheduler().telemetry().activeDrainPreference;
}

size_t IndependentPhaseAsyncServer::drainPreferenceTransitionCount() const noexcept
{
    return mCoordinator.scheduler().telemetry().drainPreferenceTransitions;
}

size_t IndependentPhaseAsyncServer::drainPreferenceAppliedDispatchCount() const noexcept
{
    return mCoordinator.scheduler().telemetry().drainPreferenceAppliedDispatches;
}

size_t IndependentPhaseAsyncServer::availableAdmissionSlots() const noexcept
{
    size_t const limit = admissionLimit();
    size_t const requestCapacity = mRequests.size() < limit ? limit - mRequests.size() : 0U;
    return std::min(requestCapacity, static_cast<size_t>(std::max(mOwnership.availableSlots(), 0)));
}

int32_t IndependentPhaseAsyncServer::availableKVPages() const noexcept
{
    return mOwnership.availablePages();
}

size_t IndependentPhaseAsyncServer::admissibleRequestPrefix(
    std::vector<IndependentPhaseAdmissionRequest> const& candidates) const
{
    std::vector<IndependentPhasePageReservation> existing;
    existing.reserve(mRequests.size());
    for (auto const& request : mRequests)
    {
        existing.push_back({request.first, request.second.baseReservedPages, request.second.fullReservedPages});
    }
    std::vector<IndependentPhasePageReservation> candidateReservations;
    candidateReservations.reserve(candidates.size());
    for (IndependentPhaseAdmissionRequest const& candidate : candidates)
    {
        candidateReservations.push_back(
            makePageReservation(candidate.requestId, candidate.promptTokens, candidate.maxOutputTokens));
    }
    return phaseAdmissiblePageReservationPrefix(std::move(existing), candidateReservations, pageReservationBudget(),
        mConfig.maxConcurrentPageGrowthRequests, availableAdmissionSlots());
}

size_t IndependentPhaseAsyncServer::decodeRefillWaitCount() const noexcept
{
    return mDecodeRefillWaitCount;
}

size_t IndependentPhaseAsyncServer::prefillFormationWaitPeriodCount() const noexcept
{
    return mPrefillFormationWaitPeriodCount;
}

size_t IndependentPhaseAsyncServer::prefillFormationDeferralCount() const noexcept
{
    return mPrefillFormationDeferralCount;
}

size_t IndependentPhaseAsyncServer::prefillFormationProfileSelectionCount() const noexcept
{
    return mPrefillFormationProfileSelectionCount;
}

size_t IndependentPhaseAsyncServer::prefillFormationProfileMissCount() const noexcept
{
    return mPrefillFormationProfileMissCount;
}

size_t IndependentPhaseAsyncServer::admissionRefillWaitPeriodCount() const noexcept
{
    return mAdmissionRefillWaitPeriodCount;
}

size_t IndependentPhaseAsyncServer::admissionRefillDeferralCount() const noexcept
{
    return mAdmissionRefillDeferralCount;
}

size_t IndependentPhaseAsyncServer::pageGrowthWaitCount() const noexcept
{
    return mPageGrowthWaitCount;
}

size_t IndependentPhaseAsyncServer::pendingPageGrowthCount() const noexcept
{
    return mPendingDecodeRequests.size();
}

size_t IndependentPhaseAsyncServer::pageGrowthOwnerCount() const noexcept
{
    return mPageGrowthRequestIds.size();
}

int32_t IndependentPhaseAsyncServer::pageReservationBasePages() const noexcept
{
    int32_t result{};
    for (auto const& request : mRequests)
    {
        result += request.second.baseReservedPages;
    }
    return result;
}

int32_t IndependentPhaseAsyncServer::pageReservationGuaranteedPages() const
{
    std::vector<IndependentPhasePageReservation> reservations;
    reservations.reserve(mRequests.size());
    for (auto const& request : mRequests)
    {
        reservations.push_back({request.first, request.second.baseReservedPages, request.second.fullReservedPages});
    }
    return phasePageReservationGuaranteedPages(reservations, mConfig.maxConcurrentPageGrowthRequests);
}

float IndependentPhaseAsyncServer::decodeTpotPressure() const noexcept
{
    return mCoordinator.scheduler().telemetry().recentDecodeTpotPressure;
}

size_t IndependentPhaseAsyncServer::visionPayloadBytes() const noexcept
{
    size_t result{};
    for (auto const& request : mRequests)
    {
        if (request.second.visionPayload != nullptr)
        {
            result += request.second.visionPayload->byteSize();
        }
    }
    for (PendingRequest const& request : mPendingRequests)
    {
        if (request.visionPayload != nullptr)
        {
            result += request.visionPayload->byteSize();
        }
    }
    return result;
}

size_t IndependentPhaseAsyncServer::visionPayloadBytes(std::vector<uint64_t> const& requestIds) const noexcept
{
    size_t result{};
    for (uint64_t const requestId : requestIds)
    {
        auto const request = mRequests.find(requestId);
        if (request != mRequests.end() && request->second.visionPayload != nullptr)
        {
            result += request->second.visionPayload->byteSize();
        }
    }
    return result;
}

bool IndependentPhaseAsyncServer::releasesVisionPrefillStorage() const noexcept
{
    return mConfig.releaseVisionPrefillStorage;
}

void IndependentPhaseAsyncServer::setGlobalMemoryHorizonSupplier(
    std::function<PhaseActionMemoryHorizon(PhaseGlobalActionKey const&, std::vector<uint64_t> const& requestIds)>
        supplier)
{
    mCoordinator.scheduler().setGlobalMemoryHorizonSupplier(std::move(supplier));
}

size_t IndependentPhaseAsyncServer::visionPrefillReleaseCount() const noexcept
{
    return mVisionPrefillReleaseCount;
}

size_t IndependentPhaseAsyncServer::visionPrefillReleasedBytes() const noexcept
{
    return mVisionPrefillReleasedBytes;
}

bool IndependentPhaseAsyncServer::throughputMode() const noexcept
{
    return mThroughputMode;
}

size_t IndependentPhaseAsyncServer::throughputModeTransitionCount() const noexcept
{
    return mThroughputModeTransitionCount;
}

size_t IndependentPhaseAsyncServer::adaptiveAdmissionLimit() const noexcept
{
    return admissionLimit();
}

size_t IndependentPhaseAsyncServer::adaptiveAdmissionIncreaseCount() const noexcept
{
    return mAdaptiveAdmissionIncreaseCount;
}

size_t IndependentPhaseAsyncServer::adaptiveAdmissionDecreaseCount() const noexcept
{
    return mAdaptiveAdmissionDecreaseCount;
}

size_t IndependentPhaseAsyncServer::adaptiveAdmissionCostLimit() const noexcept
{
    return costLimitedAdmissionLimit();
}

size_t IndependentPhaseAsyncServer::adaptiveAdmissionCostBlockCount() const noexcept
{
    return mAdaptiveAdmissionCostBlockCount;
}

double IndependentPhaseAsyncServer::adaptiveAdmissionTpotBudgetUs() const noexcept
{
    return effectiveAdmissionTpotBudgetUs();
}

bool IndependentPhaseAsyncServer::adaptiveAdmissionTpotBudgetSatisfiable() const noexcept
{
    return phaseAdmissionTpotBudgetSatisfiable(
        activeAdmissionCosts(), mConfig.latencyInFlightRequests, effectiveAdmissionTpotBudgetUs());
}

bool IndependentPhaseAsyncServer::adaptiveAdmissionExternalProfileActive() const noexcept
{
    return !mConfig.adaptiveAdmissionExternalCosts.empty() && mAdmissionExternalProfileEpochSelection.value_or(false);
}

size_t IndependentPhaseAsyncServer::adaptiveAdmissionExternalProfileSelectionCount() const noexcept
{
    return mAdaptiveAdmissionExternalProfileSelectionCount;
}

size_t IndependentPhaseAsyncServer::adaptiveAdmissionUnsatisfiableDecisionCount() const noexcept
{
    return mAdaptiveAdmissionUnsatisfiableDecisionCount;
}

float IndependentPhaseAsyncServer::decodeAdmissionTpotPressure() const noexcept
{
    double budgetUs = phaseAdmissionProfileTpotBudget(mConfig.adaptiveAdmissionTpotBudgetUs,
        mConfig.adaptiveAdmissionExternalTpotBudgetUs, adaptiveAdmissionExternalProfileActive());
    if (mExternalMinTpotTargetUs > 0.0 && (budgetUs == 0.0 || mExternalMinTpotTargetUs < budgetUs))
    {
        budgetUs = mExternalMinTpotTargetUs;
    }
    double const observedUs = mCoordinator.scheduler().telemetry().recentDecodeTpotP95Us;
    if (budgetUs > 0.0 && observedUs > 0.0)
    {
        return static_cast<float>(observedUs / budgetUs);
    }
    return decodeTpotPressure();
}

IndependentPhaseServerArbitrationSnapshot IndependentPhaseAsyncServer::arbitrationSnapshot(
    bool includeReadyDetails) const noexcept
{
    IndependentPhaseServerArbitrationSnapshot result;
    result.busy = mCoordinator.busy();
    result.inFlightKind = mCoordinator.inFlightKind();
    result.inFlightPrefillClass = mCoordinator.inFlightPrefillClass();
    if (includeReadyDetails)
    {
        result.inFlight = mCoordinator.inFlightSnapshot();
    }
    PhaseQueueSnapshot const queue = mCoordinator.scheduler().queueSnapshot(includeReadyDetails);
    result.prefillQueued = queue.prefillQueued;
    result.prefillCandidateTokens = queue.prefillCandidateTokens;
    result.decodeQueued = queue.decodeQueued;
    result.decodeCandidateTokens = queue.decodeCandidateTokens;
    result.pagePoolAllocatedBundles = queue.pagePoolAllocatedBundles;
    result.pageReservationGuaranteedBundles = queue.pageReservationGuaranteedBundles;
    if (includeReadyDetails)
    {
        result.prefillRequestIds = queue.prefillRequestIds;
        result.prefillTokenCounts = queue.prefillTokenCounts;
        result.decodeRequestIds = queue.decodeRequestIds;
        result.decodeContextLengths = queue.decodeContextLengths;
        result.visionPayloadBytes = visionPayloadBytes();

        std::vector<uint64_t> activeRequestIds;
        activeRequestIds.reserve(mRequests.size());
        for (auto const& [requestId, request] : mRequests)
        {
            static_cast<void>(request);
            activeRequestIds.push_back(requestId);
        }
        std::sort(activeRequestIds.begin(), activeRequestIds.end());
        result.kvOwnershipSignature = kSTATE_HASH_OFFSET;
        result.visionLeaseSignature = kSTATE_HASH_OFFSET;
        addStateHash(result.kvOwnershipSignature, activeRequestIds.size());
        addStateHash(result.visionLeaseSignature, activeRequestIds.size());
        for (uint64_t const requestId : activeRequestIds)
        {
            RequestState const& request = mRequests.at(requestId);
            addStateHash(result.kvOwnershipSignature, requestId);
            addStateHash(result.kvOwnershipSignature, static_cast<uint64_t>(request.kvSlotId));
            if (request.kvSlotId >= 0 && mOwnership.leased(request.kvSlotId))
            {
                addStateHash(result.kvOwnershipSignature, mOwnership.leaseGeneration(request.kvSlotId));
                addStateHash(result.kvOwnershipSignature, static_cast<uint64_t>(mOwnership.length(request.kvSlotId)));
                std::vector<int32_t> const& pages = mOwnership.pages(request.kvSlotId);
                addStateHash(result.kvOwnershipSignature, pages.size());
                for (int32_t const page : pages)
                {
                    addStateHash(result.kvOwnershipSignature, static_cast<uint64_t>(page));
                }
            }
            addStateHash(result.visionLeaseSignature, requestId);
            addStateHash(result.visionLeaseSignature, request.awaitingVisionPayload ? 1U : 0U);
            addStateHash(result.visionLeaseSignature, request.visionPrefixComplete ? 1U : 0U);
            addVisionPayloadHash(result.visionLeaseSignature, request.visionPayload.get());
            addVisionPayloadHash(result.visionLeaseSignature, request.pendingVisionPayload.get());
        }
        addStateHash(result.visionLeaseSignature, mPendingRequests.size());
        for (PendingRequest const& request : mPendingRequests)
        {
            addStateHash(result.visionLeaseSignature, request.requestId);
            addVisionPayloadHash(result.visionLeaseSignature, request.visionPayload.get());
        }
    }
    result.prefillOldestRequestAgeUs = queue.prefillOldestRequestAgeUs;
    result.prefillMinTtftSlackUs = queue.prefillMinTtftSlackUs;
    result.decodeOldestWaitUs = queue.decodeOldestWaitUs;
    PhaseSchedulerTelemetry const& telemetry = mCoordinator.scheduler().telemetry();
    result.recentDecodeTpotP95Us = telemetry.recentDecodeTpotP95Us;
    result.recentDecodeTpotPressure = telemetry.recentDecodeTpotPressure;

    auto const now = std::chrono::steady_clock::now();
    for (auto const& [requestId, request] : mRequests)
    {
        static_cast<void>(requestId);
        if (request.externalProducer || !request.generatedTokens.empty())
        {
            continue;
        }
        double const ageUs = std::chrono::duration<double, std::micro>(now - request.submittedAt).count();
        result.oldestTextWithoutTokenAgeUs = std::max(result.oldestTextWithoutTokenAgeUs, ageUs);
    }
    for (PendingRequest const& request : mPendingRequests)
    {
        if (request.visionPayload != nullptr)
        {
            continue;
        }
        double const ageUs = std::chrono::duration<double, std::micro>(now - request.scheduling.submittedAt).count();
        result.oldestTextWithoutTokenAgeUs = std::max(result.oldestTextWithoutTokenAgeUs, ageUs);
    }
    return result;
}

std::vector<PhaseProjectedRequest> IndependentPhaseAsyncServer::projectedRequests(size_t bytesPerKVPage) const noexcept
{
    PhaseQueueSnapshot const queue = mCoordinator.scheduler().queueSnapshot(true);
    PhaseInFlightSnapshot const inFlight = mCoordinator.inFlightSnapshot();
    std::unordered_set<uint64_t> const readyPrefill(queue.prefillRequestIds.begin(), queue.prefillRequestIds.end());
    std::unordered_set<uint64_t> const readyDecode(queue.decodeRequestIds.begin(), queue.decodeRequestIds.end());
    std::unordered_map<uint64_t, PhaseUnifiedPhase> inFlightPhase;
    for (PhaseInFlightWorkSnapshot const& work : inFlight.work)
    {
        for (uint64_t const requestId : work.requestIds)
        {
            inFlightPhase.emplace(requestId, work.phase);
        }
    }

    std::vector<uint64_t> requestIds;
    requestIds.reserve(mRequests.size());
    for (auto const& [requestId, request] : mRequests)
    {
        static_cast<void>(request);
        requestIds.push_back(requestId);
    }
    std::sort(requestIds.begin(), requestIds.end());

    std::vector<PhaseProjectedRequest> result;
    result.reserve(requestIds.size());
    for (uint64_t const requestId : requestIds)
    {
        RequestState const& request = mRequests.at(requestId);
        auto const running = inFlightPhase.find(requestId);
        PhaseProjectedRequestStage stage = !request.generatedTokens.empty() ? PhaseProjectedRequestStage::kDecode
                                                                            : PhaseProjectedRequestStage::kPrefill;
        if (running != inFlightPhase.end())
        {
            stage = running->second == PhaseUnifiedPhase::kDecode ? PhaseProjectedRequestStage::kDecode
                                                                  : PhaseProjectedRequestStage::kPrefill;
        }
        else if (readyDecode.count(requestId) != 0U)
        {
            stage = PhaseProjectedRequestStage::kDecode;
        }
        else if (readyPrefill.count(requestId) != 0U)
        {
            stage = PhaseProjectedRequestStage::kPrefill;
        }

        bool const prefill = stage == PhaseProjectedRequestStage::kPrefill;
        int32_t const generated = static_cast<int32_t>(request.generatedTokens.size());
        int32_t const decodeStepsRemaining = std::max(0, request.maxOutputTokens - generated - (prefill ? 1 : 0));
        size_t visionBytes{};
        if (request.visionPayload != nullptr)
        {
            visionBytes += request.visionPayload->byteSize();
        }
        if (request.pendingVisionPayload != nullptr && request.pendingVisionPayload != request.visionPayload)
        {
            visionBytes += request.pendingVisionPayload->byteSize();
        }
        bool const kvOwned = request.kvSlotId >= 0 && mOwnership.leased(request.kvSlotId);
        size_t const kvBytes
            = kvOwned ? static_cast<size_t>(mOwnership.releasablePages(request.kvSlotId)) * bytesPerKVPage : 0U;
        bool const ready = running == inFlightPhase.end()
            && (readyPrefill.count(requestId) != 0U || readyDecode.count(requestId) != 0U);
        result.push_back({requestId, stage, request.kvSlotId, decodeStepsRemaining, generated, visionBytes, kvBytes,
            visionBytes > 0U, kvOwned, generated > 0, ready});
    }
    return result;
}

bool IndependentPhaseAsyncServer::empty() const noexcept
{
    return mRequests.empty() && mPendingRequests.empty() && mPendingDecodeRequests.empty() && mSamplingTickets.empty()
        && mCoordinator.empty();
}

CUcontext IndependentPhaseAsyncServer::cudaContext() const noexcept
{
    return mCoordinator.cudaContext();
}

PhaseGlobalSchedulerMode IndependentPhaseAsyncServer::globalSchedulerMode() const noexcept
{
    return mCoordinator.scheduler().globalSchedulerMode();
}

uint64_t IndependentPhaseAsyncServer::globalPlanSequence() const noexcept
{
    return mCoordinator.scheduler().globalPlanSequence();
}

uint64_t IndependentPhaseAsyncServer::globalSnapshotEpoch() const noexcept
{
    return mCoordinator.scheduler().globalSnapshotEpoch();
}

PhaseSchedulerTelemetry const& IndependentPhaseAsyncServer::schedulerTelemetry() const noexcept
{
    return mCoordinator.scheduler().telemetry();
}

bool IndependentPhaseAsyncServer::admitPendingRequests()
{
    if (shouldWaitForAdmissionRefill())
    {
        return false;
    }
    bool admitted{};
    while (!mPendingRequests.empty())
    {
        PendingRequest request = std::move(mPendingRequests.front());
        mPendingRequests.pop_front();
        mPendingRequestIds.erase(request.requestId);
        IndependentPhaseServerSubmission const result = submitImpl(request.requestId, request.promptTokens,
            request.visionPayload, request.maxOutputTokens, request.scheduling);
        if (result.status == IndependentPhaseServerStatus::kAdmitted)
        {
            admitted = true;
            continue;
        }
        ELLM_CHECK(result.status == IndependentPhaseServerStatus::kBackpressure,
            "Pending phase request became invalid during admission");
        mPendingRequests.push_front(std::move(request));
        mPendingRequestIds.insert(mPendingRequests.front().requestId);
        break;
    }
    return admitted;
}

bool IndependentPhaseAsyncServer::shouldWaitForAdmissionRefill()
{
    if (mConfig.admissionRefillBatchSize <= 1U || mConfig.admissionRefillWindowUs <= 0.0
        || mPendingRequests.size() < mConfig.admissionRefillBatchSize || mRequests.empty())
    {
        mAdmissionRefillStartedAt.reset();
        return false;
    }
    size_t const availableSlots = availableAdmissionSlots();
    bool const eligible = availableSlots > 0U && availableSlots < mConfig.admissionRefillBatchSize;
    if (!eligible)
    {
        mAdmissionRefillStartedAt.reset();
        return false;
    }

    auto const now = std::chrono::steady_clock::now();
    if (!mAdmissionRefillStartedAt)
    {
        mAdmissionRefillStartedAt = now;
        ++mAdmissionRefillWaitPeriodCount;
    }
    double const elapsedUs = std::chrono::duration<double, std::micro>(now - *mAdmissionRefillStartedAt).count();
    bool const defer = shouldDeferAdmissionForPrefillRefill(mConfig.admissionRefillBatchSize, mPendingRequests.size(),
        availableSlots, mRequests.size(), elapsedUs, mConfig.admissionRefillWindowUs);
    if (defer)
    {
        ++mAdmissionRefillDeferralCount;
        return true;
    }
    mAdmissionRefillStartedAt.reset();
    return false;
}

bool IndependentPhaseAsyncServer::resumePendingDecodeRequests()
{
    bool resumed{};
    if (mConfig.pageReservationMode == IndependentPhasePageReservationMode::kHeadroom)
    {
        std::vector<uint64_t> owners(mPageGrowthRequestIds.begin(), mPageGrowthRequestIds.end());
        std::sort(owners.begin(), owners.end());
        size_t startedOwners{};
        size_t waitingUnstartedOwners{};
        for (uint64_t const requestId : owners)
        {
            auto const request = mRequests.find(requestId);
            ELLM_CHECK(request != mRequests.end(), "Page growth owner request is missing");
            if (request->second.pageGrowthStarted)
            {
                ++startedOwners;
            }
            else if (mPendingDecodeRequestIds.find(requestId) != mPendingDecodeRequestIds.end())
            {
                ++waitingUnstartedOwners;
            }
        }
        if (shouldDeferPhasePageGrowthCohort(owners.size(), startedOwners, waitingUnstartedOwners))
        {
            return false;
        }
        if (waitingUnstartedOwners > 0)
        {
            for (uint64_t const requestId : owners)
            {
                auto request = mRequests.find(requestId);
                if (!request->second.pageGrowthStarted
                    && mPendingDecodeRequestIds.find(requestId) != mPendingDecodeRequestIds.end())
                {
                    request->second.pageGrowthStarted = true;
                }
            }
        }
        for (uint64_t const requestId : owners)
        {
            if (mPendingDecodeRequestIds.erase(requestId) == 0)
            {
                continue;
            }
            auto const pending = std::find(mPendingDecodeRequests.begin(), mPendingDecodeRequests.end(), requestId);
            ELLM_CHECK(pending != mPendingDecodeRequests.end(), "Pending page growth request index is inconsistent");
            mPendingDecodeRequests.erase(pending);
            auto it = mRequests.find(requestId);
            ELLM_CHECK(it != mRequests.end(), "Pending page growth request is missing");
            resumed = enqueueDecodeOrWait(requestId, it->second) || resumed;
        }
        return resumed;
    }
    while (!mPendingDecodeRequests.empty())
    {
        uint64_t const requestId = mPendingDecodeRequests.front();
        auto it = mRequests.find(requestId);
        ELLM_CHECK(it != mRequests.end(), "Pending decode request is missing");
        if (!enqueueDecodeOrWait(requestId, it->second))
        {
            break;
        }
        mPendingDecodeRequests.pop_front();
        mPendingDecodeRequestIds.erase(requestId);
        resumed = true;
    }
    return resumed;
}

bool IndependentPhaseAsyncServer::enqueueDecodeOrWait(uint64_t requestId, RequestState& state)
{
    int32_t const targetLength = mOwnership.length(state.kvSlotId) + 1;
    if (mConfig.pageReservationMode == IndependentPhasePageReservationMode::kHeadroom)
    {
        int32_t const requiredPages
            = (targetLength + mOwnership.config().tokensPerPage - 1) / mOwnership.config().tokensPerPage;
        if (requiredPages > state.baseReservedPages
            && (mPageGrowthRequestIds.find(requestId) == mPageGrowthRequestIds.end() || !state.pageGrowthStarted))
        {
            if (mPendingDecodeRequestIds.insert(requestId).second)
            {
                mPendingDecodeRequests.push_back(requestId);
                ++mPageGrowthWaitCount;
            }
            return false;
        }
    }
    try
    {
        mOwnership.ensureCapacity(state.kvSlotId, targetLength);
    }
    catch (std::runtime_error const&)
    {
        if (mPendingDecodeRequestIds.insert(requestId).second)
        {
            mPendingDecodeRequests.push_back(requestId);
        }
        return false;
    }
    mCoordinator.enqueueDecode(
        {requestId, mOwnership.length(state.kvSlotId), state.kvSlotId, 0, 0, true, state.scheduling});
    recordTimeline(requestId, PhaseTimelineStage::kDecodeReady, state.kvSlotId, state.decodeProducerSequenceId);
    return true;
}

IndependentPhasePageReservation IndependentPhaseAsyncServer::makePageReservation(
    uint64_t requestId, int32_t promptTokens, int32_t maxOutputTokens) const
{
    int32_t const tokensPerPage = mOwnership.config().tokensPerPage;
    auto const pagesForTokens
        = [tokensPerPage](int32_t tokens) { return (tokens + tokensPerPage - 1) / tokensPerPage; };
    int32_t const fullPages = pagesForTokens(promptTokens + maxOutputTokens);
    int32_t const reservedOutput = mConfig.pageReservationMode == IndependentPhasePageReservationMode::kFull
        ? maxOutputTokens
        : std::min(maxOutputTokens, mConfig.outputHeadroomTokens);
    return {requestId, pagesForTokens(promptTokens + reservedOutput), fullPages};
}

bool IndependentPhaseAsyncServer::hasPageReservationCapacity(IndependentPhasePageReservation const& reservation) const
{
    std::vector<IndependentPhasePageReservation> reservations;
    reservations.reserve(mRequests.size() + 1U);
    for (auto const& request : mRequests)
    {
        reservations.push_back({request.first, request.second.baseReservedPages, request.second.fullReservedPages});
    }
    reservations.push_back(reservation);
    return phasePageReservationsFit(pageReservationBudget(), reservations, mConfig.maxConcurrentPageGrowthRequests);
}

int32_t IndependentPhaseAsyncServer::pageReservationBudget() const
{
    std::unordered_set<int32_t> requestSlots;
    requestSlots.reserve(mRequests.size());
    for (auto const& request : mRequests)
    {
        requestSlots.insert(request.second.kvSlotId);
    }
    std::unordered_set<int32_t> externalPages;
    for (int32_t slot{}; slot < mOwnership.config().maxStableSlots; ++slot)
    {
        if (!mOwnership.leased(slot) || requestSlots.find(slot) != requestSlots.end())
        {
            continue;
        }
        auto const& pages = mOwnership.pages(slot);
        externalPages.insert(pages.begin(), pages.end());
    }
    return mOwnership.config().numPages - static_cast<int32_t>(externalPages.size());
}

void IndependentPhaseAsyncServer::refreshPageGrowthOwners()
{
    if (mConfig.pageReservationMode != IndependentPhasePageReservationMode::kHeadroom)
    {
        mPageGrowthRequestIds.clear();
        return;
    }
    std::vector<IndependentPhasePageReservation> reservations;
    reservations.reserve(mRequests.size());
    for (auto const& request : mRequests)
    {
        reservations.push_back({request.first, request.second.baseReservedPages, request.second.fullReservedPages});
    }
    std::vector<uint64_t> currentOwners(mPageGrowthRequestIds.begin(), mPageGrowthRequestIds.end());
    std::vector<uint64_t> const selected
        = selectPhasePageGrowthOwners(reservations, currentOwners, mConfig.maxConcurrentPageGrowthRequests);
    mPageGrowthRequestIds.clear();
    mPageGrowthRequestIds.insert(selected.begin(), selected.end());
}

IndependentPhaseCoordinatorCallbacks IndependentPhaseAsyncServer::makeCallbacks()
{
    IndependentPhaseCoordinatorCallbacks callbacks;
    callbacks.stagePrefill = [this](std::vector<PhaseWorkItem> const& batch, PipelineIO& io, cudaStream_t stream) {
        if (mAdapter.stagePrefill)
        {
            mAdapter.stagePrefill(makeViews(batch), io, mCoordinator.prefillTensorMap(), stream);
        }
    };
    callbacks.stageDecode = [this](std::vector<PhaseWorkItem> const& batch, PipelineIO& io, cudaStream_t stream) {
        if (mAdapter.stageDecode)
        {
            mAdapter.stageDecode(makeViews(batch), io, mCoordinator.decodeTensorMap(), stream);
        }
    };
    callbacks.completePrefillBatch
        = [this](std::vector<PhaseWorkItem> const& batch, PipelineIO& io, cudaStream_t stream) {
              std::vector<IndependentPhaseRequestView> const views = makeViews(batch);
              std::vector<IndependentPhaseRequestView> finalViews;
              for (auto const& view : views)
              {
                  if (view.work.tokenOffset + view.work.tokenCount == static_cast<int32_t>(view.promptTokens->size()))
                  {
                      RequestState& state = mRequests.at(view.requestId);
                      if (state.awaitingVisionPayload)
                      {
                          state.visionPrefixComplete = true;
                          continue;
                      }
                      finalViews.push_back(view);
                  }
              }
              if (!finalViews.empty())
              {
                  std::unique_ptr<IndependentPhaseSampleTicket> ticket
                      = submitSamplingWithActivity(finalViews, io, stream, true);
                  ELLM_CHECK(ticket != nullptr, "Prefill sampling adapter returned no completion ticket");
                  enqueueSamplingTicket(std::move(ticket));
                  for (IndependentPhaseRequestView const& view : finalViews)
                  {
                      if (!mConfig.releaseVisionPrefillStorage || view.visionPayload == nullptr)
                      {
                          continue;
                      }
                      size_t const releasedBytes = view.visionPayload->releasePrefillStorage();
                      if (releasedBytes > 0)
                      {
                          ++mVisionPrefillReleaseCount;
                          mVisionPrefillReleasedBytes += releasedBytes;
                      }
                  }
              }
          };
    callbacks.completeDecodeBatch
        = [this](std::vector<PhaseWorkItem> const& batch, PipelineIO& io, cudaStream_t stream) {
              std::unique_ptr<IndependentPhaseSampleTicket> ticket
                  = submitSamplingWithActivity(makeViews(batch), io, stream, false);
              ELLM_CHECK(ticket != nullptr, "Decode sampling adapter returned no completion ticket");
              enqueueSamplingTicket(std::move(ticket));
          };
    callbacks.isPrefillFinished = [](PhaseWorkItem const& item, int32_t) {
        return item.tokenOffset + item.tokenCount == item.promptTokenCount;
    };
    callbacks.isDecodeFinished = [](PhaseWorkItem const&, int32_t) { return true; };
    callbacks.onTimeline = [this](PhaseTimelineEvent const& event) {
        if (mTimelineCallback)
        {
            mTimelineCallback(event);
        }
    };
    return callbacks;
}

std::unique_ptr<IndependentPhaseSampleTicket> IndependentPhaseAsyncServer::submitSamplingWithActivity(
    std::vector<IndependentPhaseRequestView> const& views, PipelineIO& io, cudaStream_t stream, bool fromPrefill)
{
    if (mActivityTimeline == nullptr)
    {
        return mAdapter.submitSampling(views, io, stream, fromPrefill);
    }
    PhaseActivityKind const kind = fromPrefill ? PhaseActivityKind::kPrefill : PhaseActivityKind::kDecode;
    char const* name = fromPrefill ? "prefill_sampling" : "decode_sampling";
    PhaseActivityTimelineRecorder::Token const token
        = mActivityTimeline->begin(kind, stream, name, mNextSamplingTicketSequence);
    try
    {
        std::unique_ptr<IndependentPhaseSampleTicket> ticket = mAdapter.submitSampling(views, io, stream, fromPrefill);
        mActivityTimeline->end(token, stream);
        return ticket;
    }
    catch (...)
    {
        mActivityTimeline->cancel(token);
        throw;
    }
}

std::vector<IndependentPhaseRequestView> IndependentPhaseAsyncServer::makeViews(
    std::vector<PhaseWorkItem> const& batch) const
{
    std::vector<IndependentPhaseRequestView> views;
    views.reserve(batch.size());
    for (size_t row{}; row < batch.size(); ++row)
    {
        PhaseWorkItem const& work = batch[row];
        auto const it = mRequests.find(work.requestId);
        ELLM_CHECK(it != mRequests.end(), "Phase adapter requested an unknown request");
        views.push_back({work.requestId, row, work, &it->second.promptTokens, &it->second.generatedTokens,
            it->second.visionPayload.get()});
    }
    return views;
}

bool IndependentPhaseAsyncServer::isEos(int32_t tokenId) const noexcept
{
    return std::find(mConfig.eosTokenIds.begin(), mConfig.eosTokenIds.end(), tokenId) != mConfig.eosTokenIds.end();
}

void IndependentPhaseAsyncServer::enqueueSamplingTicket(std::unique_ptr<IndependentPhaseSampleTicket> ticket)
{
    ELLM_CHECK(ticket != nullptr && ticket->ready != nullptr, "Sampling adapter returned an invalid CUDA event");
    ticket->sequenceId = mNextSamplingTicketSequence++;
    ticket->submittedAt = std::chrono::steady_clock::now();
    recordSamplingTimeline(*ticket,
        ticket->fromPrefill ? PhaseTimelineStage::kPrefillSamplingSubmit : PhaseTimelineStage::kDecodeSamplingSubmit);
    mSamplingTickets.push_back(std::move(ticket));
}

void IndependentPhaseAsyncServer::completeSamplingTicket(std::unique_ptr<IndependentPhaseSampleTicket> ticket)
{
    recordSamplingTimeline(*ticket,
        ticket->fromPrefill ? PhaseTimelineStage::kPrefillSamplingReady : PhaseTimelineStage::kDecodeSamplingReady);
    if (ticket->submittedAt != std::chrono::steady_clock::time_point{})
    {
        double const latencyUs
            = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - ticket->submittedAt).count();
        mSamplingLatencyUs.push_back(latencyUs);
        while (mSamplingLatencyUs.size() > mConfig.globalSamplingLatencyWindow)
        {
            mSamplingLatencyUs.pop_front();
        }
    }
    processTicket(std::move(ticket));
}

bool IndependentPhaseAsyncServer::processSamplingTickets()
{
    std::vector<std::unique_ptr<IndependentPhaseSampleTicket>> readyTickets;
    for (auto ticket = mSamplingTickets.begin(); ticket != mSamplingTickets.end();)
    {
        cudaError_t status = cudaEventQuery((*ticket)->ready);
        PhaseQueueSnapshot const queue = mCoordinator.scheduler().queueSnapshot();
        bool const synchronize = status == cudaErrorNotReady
            && phaseShouldSynchronizeDecodeSampling(mConfig.synchronizeDecodeSampling, (*ticket)->fromPrefill,
                queue.externalPrefillQueued, mExternalPendingRequests);
        if (synchronize)
        {
            status = cudaEventSynchronize((*ticket)->ready);
        }
        if (status == cudaErrorNotReady)
        {
            ++ticket;
            continue;
        }
        CUDA_CHECK(status);
        readyTickets.push_back(std::move(*ticket));
        ticket = mSamplingTickets.erase(ticket);
    }
    for (auto& ticket : readyTickets)
    {
        completeSamplingTicket(std::move(ticket));
    }
    return !readyTickets.empty();
}

bool IndependentPhaseAsyncServer::flushEventCallbacks()
{
    bool delivered{};
    if (mTokenCallback)
    {
        while (!mTokenEvents.empty())
        {
            IndependentPhaseServerToken event = std::move(mTokenEvents.front());
            mTokenEvents.pop_front();
            mTokenCallback(std::move(event));
            delivered = true;
        }
    }
    if (mCompletionCallback)
    {
        while (!mCompletions.empty())
        {
            IndependentPhaseServerCompletion event = std::move(mCompletions.front());
            mCompletions.pop_front();
            mCompletionCallback(std::move(event));
            delivered = true;
        }
    }
    return delivered;
}

void IndependentPhaseAsyncServer::processTicket(std::unique_ptr<IndependentPhaseSampleTicket> ticket)
{
    std::vector<int32_t> const tokens = ticket->collect();
    ELLM_CHECK(tokens.size() == ticket->requestIds.size(), "Phase sampling ticket returned an invalid row count");
    recordSamplingTimeline(*ticket,
        ticket->fromPrefill ? PhaseTimelineStage::kPrefillSamplingCollected
                            : PhaseTimelineStage::kDecodeSamplingCollected);
    for (size_t index{}; index < tokens.size(); ++index)
    {
        uint64_t const requestId = ticket->requestIds[index];
        auto it = mRequests.find(requestId);
        if (it == mRequests.end())
        {
            continue;
        }
        RequestState& state = it->second;
        state.generatedTokens.push_back(tokens[index]);
        recordTimeline(requestId,
            ticket->fromPrefill ? PhaseTimelineStage::kPrefillTokenCommitted
                                : PhaseTimelineStage::kDecodeTokenCommitted,
            state.kvSlotId, ticket->sequenceId);
        bool const eos = isEos(tokens[index]);
        double const elapsedMs
            = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - state.submittedAt).count();
        mTokenEvents.push_back(
            {requestId, tokens[index], static_cast<int32_t>(state.generatedTokens.size() - 1U), eos, elapsedMs});
        if (state.generatedTokens.size() == 1U)
        {
            recordTimeline(requestId, PhaseTimelineStage::kFirstToken, state.kvSlotId);
        }
        if (eos || static_cast<int32_t>(state.generatedTokens.size()) >= state.maxOutputTokens)
        {
            finishRequest(requestId, eos);
        }
        else
        {
            state.decodeProducerSequenceId = ticket->sequenceId;
            static_cast<void>(enqueueDecodeOrWait(requestId, state));
        }
    }
    destroyTicketEvent(*ticket);
}

void IndependentPhaseAsyncServer::finishRequest(uint64_t requestId, bool stoppedByEos)
{
    auto it = mRequests.find(requestId);
    ELLM_CHECK(it != mRequests.end(), "Finished phase request is missing");
    RequestState& state = it->second;
    if (mConfig.enablePrefixReuse && mAdapter.supportsPageAlignedPrefixReuse && mPrefixCache != nullptr
        && state.promptTokens.size() >= 128U)
    {
        mPrefixCache->publish(state.kvSlotId, state.promptTokens, mOwnership.length(state.kvSlotId));
    }
    else
    {
        mOwnership.release(state.kvSlotId);
        recordTimeline(requestId, PhaseTimelineStage::kSlotReleased, state.kvSlotId);
    }
    recordTimeline(requestId, PhaseTimelineStage::kCompletion, state.kvSlotId);
    double const latencyMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - state.submittedAt).count();
    mCompletions.push_back({requestId, std::move(state.generatedTokens),
        static_cast<int32_t>(state.promptTokens.size()), latencyMs, stoppedByEos});
    auto const promptLength = mActivePromptTokens.find(static_cast<int32_t>(state.promptTokens.size()));
    ELLM_CHECK(promptLength != mActivePromptTokens.end(), "Active prompt-length index is inconsistent");
    mActivePromptTokens.erase(promptLength);
    auto const outputLength = mActiveOutputTokens.find(state.maxOutputTokens);
    ELLM_CHECK(outputLength != mActiveOutputTokens.end(), "Active output-length index is inconsistent");
    mActiveOutputTokens.erase(outputLength);
    mRequests.erase(it);
    if (mConfig.pageReservationMode == IndependentPhasePageReservationMode::kHeadroom)
    {
        refreshPageGrowthOwners();
    }
}

void IndependentPhaseAsyncServer::recordSamplingTimeline(
    IndependentPhaseSampleTicket const& ticket, PhaseTimelineStage stage) const
{
    if (!mTimelineCallback)
    {
        return;
    }
    uint64_t const timestampNs = phaseTimelineNowNs();
    for (uint64_t const requestId : ticket.requestIds)
    {
        auto const request = mRequests.find(requestId);
        int32_t const kvSlotId = request == mRequests.end() ? -1 : request->second.kvSlotId;
        recordTimeline(requestId, stage, kvSlotId, ticket.sequenceId, timestampNs);
    }
}

void IndependentPhaseAsyncServer::recordTimeline(
    uint64_t requestId, PhaseTimelineStage stage, int32_t kvSlotId, uint64_t correlationId, uint64_t timestampNs) const
{
    if (mTimelineCallback)
    {
        mTimelineCallback(
            {requestId, stage, timestampNs > 0U ? timestampNs : phaseTimelineNowNs(), correlationId, 0, kvSlotId});
    }
}

void IndependentPhaseAsyncServer::destroyTicketEvent(IndependentPhaseSampleTicket& ticket) noexcept
{
    if (ticket.release)
    {
        ticket.release();
        ticket.release = {};
        ticket.ready = nullptr;
        return;
    }
    if (ticket.ready != nullptr)
    {
        static_cast<void>(cudaEventDestroy(ticket.ready));
        ticket.ready = nullptr;
    }
}

} // namespace trt_edgellm::rt
