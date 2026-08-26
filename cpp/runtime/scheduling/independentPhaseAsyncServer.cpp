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

bool shouldDeferDecodeForSamplingRefill(
    size_t targetRows, size_t prefillRows, size_t decodeRows, size_t pendingDecodeSamplingRows) noexcept
{
    return targetRows > 0 && prefillRows == 0 && decodeRows > 0 && decodeRows < targetRows
        && decodeRows + pendingDecodeSamplingRows >= targetRows;
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
    mCoordinator.scheduler().setOnlineDecodeCostLearningActive(!mConfig.enableAdaptiveAdmission || mThroughputMode);
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
    state.promptTokens = std::move(state.pendingVisionPromptTokens);
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

bool IndependentPhaseAsyncServer::poll()
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
    bool const waitForDecodeRefill = shouldWaitForDecodeRefill();
    if (waitForDecodeRefill)
    {
        ++mDecodeRefillWaitCount;
    }
    if (!mCoordinator.busy() && !mCoordinator.empty() && !waitForDecodeRefill)
    {
        progressed = mCoordinator.dispatchNext() || progressed;
    }
    progressed = flushEventCallbacks() || progressed;
    return progressed;
}

bool IndependentPhaseAsyncServer::shouldWaitForDecodeRefill() const noexcept
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
    return shouldDeferDecodeForSamplingRefill(
        refillTarget, mCoordinator.scheduler().prefillQueueSize(), queuedDecode, pendingDecodeRows);
}

size_t IndependentPhaseAsyncServer::admissionLimit() const noexcept
{
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
        mCoordinator.scheduler().setOnlineDecodeCostLearningActive(mThroughputMode);
        return;
    }
    bool const next = nextAdaptiveThroughputMode(mThroughputMode, mPendingRequests.size(), mRequests.size(),
        mConfig.latencyInFlightRequests, mConfig.adaptiveBacklogEnterRequests);
    if (next != mThroughputMode)
    {
        mThroughputMode = next;
        ++mThroughputModeTransitionCount;
    }
    mCoordinator.scheduler().setOnlineDecodeCostLearningActive(mThroughputMode);
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

IndependentPhaseServerArbitrationSnapshot IndependentPhaseAsyncServer::arbitrationSnapshot() const noexcept
{
    IndependentPhaseServerArbitrationSnapshot result;
    result.busy = mCoordinator.busy();
    result.inFlightKind = mCoordinator.inFlightKind();
    result.inFlightPrefillClass = mCoordinator.inFlightPrefillClass();
    PhaseQueueSnapshot const queue = mCoordinator.scheduler().queueSnapshot();
    result.prefillQueued = queue.prefillQueued;
    result.decodeQueued = queue.decodeQueued;
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

bool IndependentPhaseAsyncServer::empty() const noexcept
{
    return mRequests.empty() && mPendingRequests.empty() && mPendingDecodeRequests.empty() && mSamplingTickets.empty()
        && mCoordinator.empty();
}

CUcontext IndependentPhaseAsyncServer::cudaContext() const noexcept
{
    return mCoordinator.cudaContext();
}

bool IndependentPhaseAsyncServer::admitPendingRequests()
{
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
                      = mAdapter.submitSampling(finalViews, io, stream, true);
                  ELLM_CHECK(ticket != nullptr, "Prefill sampling adapter returned no completion ticket");
                  mSamplingTickets.push_back(std::move(ticket));
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
                  = mAdapter.submitSampling(makeViews(batch), io, stream, false);
              ELLM_CHECK(ticket != nullptr, "Decode sampling adapter returned no completion ticket");
              mSamplingTickets.push_back(std::move(ticket));
          };
    callbacks.isPrefillFinished = [](PhaseWorkItem const& item, int32_t) {
        return item.tokenOffset + item.tokenCount == item.promptTokenCount;
    };
    callbacks.isDecodeFinished = [](PhaseWorkItem const&, int32_t) { return true; };
    callbacks.onTimeline = [this](PhaseTimelineEvent const& event) {
        if (!mTimelineCallback)
        {
            return;
        }
        bool emit = true;
        if (event.stage == PhaseTimelineStage::kDecodeStart)
        {
            emit = mTimelineDecodeStarted.insert(event.requestId).second;
        }
        else if (event.stage == PhaseTimelineStage::kDecodeDone)
        {
            emit = mTimelineDecodeCompleted.insert(event.requestId).second;
        }
        if (emit)
        {
            mTimelineCallback(event);
        }
    };
    return callbacks;
}

std::vector<IndependentPhaseRequestView> IndependentPhaseAsyncServer::makeViews(
    std::vector<PhaseWorkItem> const& batch) const
{
    std::vector<IndependentPhaseRequestView> views;
    views.reserve(batch.size());
    for (PhaseWorkItem const& work : batch)
    {
        auto const it = mRequests.find(work.requestId);
        ELLM_CHECK(it != mRequests.end(), "Phase adapter requested an unknown request");
        views.push_back({work.requestId, work, &it->second.promptTokens, &it->second.generatedTokens,
            it->second.visionPayload.get()});
    }
    return views;
}

bool IndependentPhaseAsyncServer::isEos(int32_t tokenId) const noexcept
{
    return std::find(mConfig.eosTokenIds.begin(), mConfig.eosTokenIds.end(), tokenId) != mConfig.eosTokenIds.end();
}

bool IndependentPhaseAsyncServer::processSamplingTickets()
{
    std::vector<std::unique_ptr<IndependentPhaseSampleTicket>> readyTickets;
    for (auto ticket = mSamplingTickets.begin(); ticket != mSamplingTickets.end();)
    {
        cudaError_t const status = cudaEventQuery((*ticket)->ready);
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
        processTicket(std::move(ticket));
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
    recordTimeline(requestId, PhaseTimelineStage::kCompletion, state.kvSlotId);
    mTimelineDecodeStarted.erase(requestId);
    mTimelineDecodeCompleted.erase(requestId);
    if (mConfig.enablePrefixReuse && mAdapter.supportsPageAlignedPrefixReuse && mPrefixCache != nullptr
        && state.promptTokens.size() >= 128U)
    {
        mPrefixCache->publish(state.kvSlotId, state.promptTokens, mOwnership.length(state.kvSlotId));
    }
    else
    {
        mOwnership.release(state.kvSlotId);
    }
    double const latencyMs
        = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - state.submittedAt).count();
    mCompletions.push_back({requestId, std::move(state.generatedTokens),
        static_cast<int32_t>(state.promptTokens.size()), latencyMs, stoppedByEos});
    mRequests.erase(it);
    if (mConfig.pageReservationMode == IndependentPhasePageReservationMode::kHeadroom)
    {
        refreshPageGrowthOwners();
    }
}

void IndependentPhaseAsyncServer::recordTimeline(uint64_t requestId, PhaseTimelineStage stage, int32_t kvSlotId) const
{
    if (mTimelineCallback)
    {
        mTimelineCallback({requestId, stage, phaseTimelineNowNs(), 0U, 0, kvSlotId});
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
