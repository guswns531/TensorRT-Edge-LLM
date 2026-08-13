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

#include "runtime/scheduling/phaseContextServingFacade.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

PhaseContextServingFacade::PhaseContextServingFacade(int32_t maxSlots, PhaseQueueSchedulerConfig schedulerConfig,
    PhaseContextServingCallbacks callbacks, HybridCacheManager& cacheManager, TensorMap& decodeTensorMap,
    cudaStream_t prefillStream, cudaStream_t decodeStream, PhaseTensorRTContextMode executionMode,
    TensorMap* prefillTensorMap, int32_t maxPrefillChunkTokens, size_t maxPendingAdmissions,
    PhaseExecutionSafetyContract safetyContract)
    : mCallbacks(std::move(callbacks))
    , mCacheManager(cacheManager)
    , mMaxSlots(maxSlots)
    , mPrefillStream(prefillStream)
    , mDecodeStream(decodeStream)
    , mHostAdmissionSlotIds(
          {maxSlots}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "phase_serving_host_admission_slots")
    , mDeviceAdmissionSlotIds({maxSlots}, DeviceType::kGPU, nvinfer1::DataType::kINT32, "phase_serving_admission_slots")
    , mDecodeAdapter(schedulerConfig.maxDecodeBatchSize, cacheManager, decodeTensorMap, "phase_serving_decode")
    , mMaxPendingAdmissions(maxPendingAdmissions)
{
    check::check(maxSlots <= cacheManager.getGlobalKVCacheLengths().getShape()[0],
        "Serving facade slot count exceeds the indexed KV cache capacity.");
    bool const usesLegacyPrefill = static_cast<bool>(mCallbacks.enqueuePrefill);
    bool const usesPackedPrefill = static_cast<bool>(mCallbacks.enqueuePackedPrefill);
    check::check(usesLegacyPrefill != usesPackedPrefill,
        "Serving facade requires exactly one legacy or packed prefill enqueue callback.");
    if (usesPackedPrefill)
    {
        check::check(prefillTensorMap != nullptr, "Packed prefill serving requires a prefill TensorMap.");
        check::check(maxPrefillChunkTokens > 0, "Packed prefill serving requires a positive maximum chunk length.");
        mPrefillAdapter = std::make_unique<PhasePrefillContextBatchAdapter>(schedulerConfig.maxPrefillBatchSize,
            maxPrefillChunkTokens, cacheManager, *prefillTensorMap, "phase_serving_prefill",
            schedulerConfig.enableRaggedPrefillBatching, schedulerConfig.enablePackedPrefillTokenLayout);
    }
    check::check(static_cast<bool>(mCallbacks.completePrefill), "Serving prefill completion callback is required.");
    bool const usesLegacyDecode = static_cast<bool>(mCallbacks.enqueueDecode);
    bool const usesPackedDecode = static_cast<bool>(mCallbacks.enqueuePackedDecode);
    check::check(usesLegacyDecode != usesPackedDecode,
        "Serving facade requires exactly one legacy or packed decode enqueue callback.");
    check::check(!usesLegacyDecode || static_cast<bool>(mCallbacks.completeDecode),
        "Legacy serving decode completion callback is required.");
    check::check(!usesPackedDecode || static_cast<bool>(mCallbacks.completePackedDecode),
        "Packed serving decode completion callback is required.");
    PhaseWorkEligibilityPolicy const configuredEligibility = std::move(schedulerConfig.eligibilityPolicy);
    schedulerConfig.eligibilityPolicy = [this, configuredEligibility](PhaseWorkItem const& item, bool prefill) {
        return (!configuredEligibility || configuredEligibility(item, prefill)) && isPageWorkEligible(item, prefill);
    };
    mLifecycle = std::make_unique<PhaseRequestLifecycle>(maxSlots, std::move(schedulerConfig), makeLifecycleCallbacks(),
        prefillStream, decodeStream, executionMode, safetyContract);
}

void PhaseContextServingFacade::configurePageReservation(PhasePageReservationConfig config)
{
    check::check(mRegistrations.empty() && mPendingAdmissions.empty() && mPageBundleReservations.empty(),
        "Serving page reservation must be configured before request admission.");
    check::check(config.outputHeadroomTokens >= 0, "Serving output page headroom cannot be negative.");
    check::check(config.maxOvercommitPageBundles >= 0, "Serving page overcommit bound cannot be negative.");
    check::check(config.maxConcurrentGrowthRequests > 0, "Serving concurrent page growth limit must be positive.");
    check::check(config.minConcurrentGrowthRequests > 0
            && config.minConcurrentGrowthRequests <= config.maxConcurrentGrowthRequests,
        "Serving minimum concurrent page growth limit is invalid.");
    check::check(std::isfinite(config.growthTpotTargetUs) && config.growthTpotTargetUs > 0.0,
        "Serving page growth TPOT target must be finite and positive.");
    check::check(std::isfinite(config.growthPressureEwmaAlpha) && config.growthPressureEwmaAlpha > 0.0F
            && config.growthPressureEwmaAlpha <= 1.0F,
        "Serving page growth EWMA alpha must be in (0, 1].");
    check::check(std::isfinite(config.growthScaleDownThreshold) && std::isfinite(config.growthScaleUpThreshold)
            && config.growthScaleDownThreshold >= 0.0F
            && config.growthScaleDownThreshold < config.growthScaleUpThreshold,
        "Serving page growth pressure thresholds are invalid.");
    check::check(config.growthAdjustmentInterval > 0 && config.growthAdjustmentStep > 0,
        "Serving page growth adjustment cadence must be positive.");
    check::check(config.fullReservationPromptThresholdTokens >= 0,
        "Serving full-reservation prompt threshold cannot be negative.");
    mPageReservationConfig = config;
    mGrowthRequestLimit
        = config.enableAdaptiveGrowthRequests ? config.minConcurrentGrowthRequests : config.maxConcurrentGrowthRequests;
    mGrowthTpotPressure = 0.0F;
    mGrowthMetricSamples = 0;
}

void PhaseContextServingFacade::resetSchedulingHistory()
{
    check::check(empty() && mRegistrations.empty() && mPendingAdmissions.empty() && mPageBundleReservations.empty(),
        "Serving scheduling history can only be reset after all requests drain");
    mLifecycle->resetSchedulingHistory();
    mGrowthRequestLimit = mPageReservationConfig.enableAdaptiveGrowthRequests
        ? mPageReservationConfig.minConcurrentGrowthRequests
        : mPageReservationConfig.maxConcurrentGrowthRequests;
    mGrowthTpotPressure = 0.0F;
    mGrowthMetricSamples = 0;
    mDrainRequestIds.clear();
}

void PhaseContextServingFacade::primeCudaGraphShapes(std::vector<PhaseCudaGraphWarmupShape> const& shapes)
{
    check::check(empty() && mRegistrations.empty() && mPendingAdmissions.empty() && mPageBundleReservations.empty(),
        "Serving CUDA graph priming requires an idle facade");
    check::check(mPrefillAdapter != nullptr, "Serving CUDA graph priming requires packed prefill support");

    Tensor hostLengths({mMaxSlots}, DeviceType::kCPU, nvinfer1::DataType::kINT32, "phase_graph_warmup_lengths");
    PhaseBatchState seedState(mMaxSlots, "phase_graph_warmup_seed", mCacheManager.isIndexedKVCache());
    uint64_t requestId{std::numeric_limits<uint64_t>::max() / 2U};
    auto resetLengths = [&](cudaStream_t stream) {
        int32_t* lengths = hostLengths.dataPointer<int32_t>();
        std::fill_n(lengths, mMaxSlots, 0);
        mCacheManager.resetForNewSequences(hostLengths, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    auto seedLengths = [&](int32_t batchSize, int32_t contextLength, cudaStream_t stream) {
        if (contextLength == 0)
        {
            return;
        }
        std::vector<PhaseWorkItem> work;
        work.reserve(batchSize);
        for (int32_t row{}; row < batchSize; ++row)
        {
            work.push_back({requestId++, contextLength, row, 0, contextLength});
        }
        seedState.prepare(work, mCacheManager, stream);
        seedState.commit(mCacheManager, contextLength, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    };
    auto releaseSlots = [&](int32_t batchSize) {
        for (int32_t slot{}; slot < batchSize; ++slot)
        {
            mCacheManager.releasePagedKVSlot(slot);
        }
    };

    for (PhaseCudaGraphWarmupShape const& shape : shapes)
    {
        check::check(shape.batchSize > 0 && shape.batchSize <= mMaxSlots,
            "CUDA graph warmup batch size is outside the stable-slot capacity");
        check::check(shape.tokenCount > 0 && shape.contextLength >= 0 && shape.repetitions >= 2,
            "CUDA graph warmup shape values are invalid");
        for (int32_t repetition{}; repetition < shape.repetitions; ++repetition)
        {
            cudaStream_t const stream
                = shape.kind == PhaseCudaGraphWarmupKind::kPrefill ? mPrefillStream : mDecodeStream;
            resetLengths(stream);
            seedLengths(shape.batchSize, shape.contextLength, stream);
            std::vector<std::unique_ptr<DecodingInferenceContext>> contexts;
            contexts.reserve(shape.batchSize);

            if (shape.kind == PhaseCudaGraphWarmupKind::kPrefill)
            {
                std::vector<PhasePrefillContextRow> rows;
                rows.reserve(shape.batchSize);
                int32_t const promptLength = shape.contextLength + shape.tokenCount + 1;
                for (int32_t row{}; row < shape.batchSize; ++row)
                {
                    auto context = std::make_unique<DecodingInferenceContext>();
                    context->initialize(1, 4, std::nullopt, OptionalInputTensors{}, "", stream);
                    context->rawBatchedInputIds = {std::vector<int32_t>(promptLength, 0)};
                    context->tokenIds = context->rawBatchedInputIds;
                    context->effectivePrefillLengths = {promptLength};
                    rows.push_back(
                        {requestId++, context.get(), 0, row, shape.contextLength, shape.tokenCount, promptLength});
                    contexts.push_back(std::move(context));
                }
                mPrefillAdapter->pack(rows, stream);
                mCallbacks.enqueuePackedPrefill(*mPrefillAdapter);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (mCallbacks.completePackedPrefill)
                {
                    mCallbacks.completePackedPrefill(*mPrefillAdapter);
                }
                mPrefillAdapter->complete();
            }
            else
            {
                check::check(shape.tokenCount == 1, "Decode CUDA graph warmup token count must be one");
                std::vector<PhaseContextRow> rows;
                rows.reserve(shape.batchSize);
                for (int32_t row{}; row < shape.batchSize; ++row)
                {
                    auto context = std::make_unique<DecodingInferenceContext>();
                    context->initialize(1, 4, std::nullopt, OptionalInputTensors{}, "", stream);
                    context->rawBatchedInputIds = {{0}};
                    context->tokenIds = {{0}};
                    context->effectivePrefillLengths = {shape.contextLength};
                    rows.push_back({requestId++, context.get(), 0, row, shape.contextLength});
                    contexts.push_back(std::move(context));
                }
                mDecodeAdapter.packDecode(rows, stream);
                if (mCallbacks.enqueuePackedDecode)
                {
                    mCallbacks.enqueuePackedDecode(mDecodeAdapter);
                }
                else
                {
                    mCallbacks.enqueueDecode(mDecodeAdapter.packedContext());
                }
                CUDA_CHECK(cudaStreamSynchronize(stream));
                if (mCallbacks.completePackedDecode)
                {
                    mCallbacks.completePackedDecode(mDecodeAdapter);
                }
                else
                {
                    mCallbacks.completeDecode(mDecodeAdapter.packedContext());
                }
                mDecodeAdapter.scatterDecode();
            }
            releaseSlots(shape.batchSize);
        }
    }
    resetLengths(mPrefillStream);
}

PhaseRequestLifecycleCallbacks PhaseContextServingFacade::makeLifecycleCallbacks()
{
    PhaseRequestLifecycleCallbacks result;
    result.onSlotRelease = [this](int32_t slot) { mCacheManager.releasePagedKVSlot(slot); };
    result.execution.onMetrics = [this](PhaseDispatchMetrics const& metrics) {
        observePageReservationMetrics(metrics);
        if (mCallbacks.onDispatchMetrics)
        {
            PhaseDispatchMetrics enriched = metrics;
            enriched.pageGrowthRequestLimit = mGrowthRequestLimit;
            enriched.pageGrowthRequestOwners = static_cast<int32_t>(mDrainRequestIds.size());
            enriched.pageGrowthTpotPressure = mGrowthTpotPressure;
            mCallbacks.onDispatchMetrics(enriched);
        }
    };
    result.execution.onDispatch = mCallbacks.onDispatch;
    result.execution.enqueuePrefill
        = [this](std::vector<PhaseWorkItem> const& batch, cudaStream_t stream) { enqueuePrefillBatch(batch, stream); };
    result.execution.completePrefillBatch = [this](std::vector<PhaseWorkItem> const& batch) {
        if (mPrefillAdapter)
        {
            check::check(mPrefillAdapter->packed(), "Serving prefill completion has no packed context.");
            if (mCallbacks.completePackedPrefill)
            {
                mCallbacks.completePackedPrefill(*mPrefillAdapter);
            }
            mPrefillAdapter->complete();
        }
        if (mCallbacks.completePrefillBatch)
        {
            mCallbacks.completePrefillBatch(batch);
        }
    };
    result.execution.completePrefill = [this](PhaseWorkItem const& item) {
        int32_t const resultingKVLength = mCallbacks.completePrefill(item);
        bool finished{};
        if (item.tokenOffset + item.tokenCount == item.promptTokenCount)
        {
            Registration const& source = registration(item.requestId);
            finished = mCallbacks.isPrefillFinished
                ? mCallbacks.isPrefillFinished(item.requestId, *source.context, source.contextRow)
                : source.context->finishedStates[static_cast<size_t>(source.contextRow)] != 0;
        }
        return PhasePrefillCompletion{resultingKVLength, finished};
    };
    result.execution.enqueueDecode
        = [this](std::vector<PhaseWorkItem> const& batch, cudaStream_t stream) { enqueueDecodeBatch(batch, stream); };
    result.execution.completeDecodeBatch
        = [this](std::vector<PhaseWorkItem> const& batch) { completeDecodeBatch(batch); };
    result.execution.completeDecode = [this](PhaseWorkItem const& item) {
        Registration const& source = registration(item.requestId);
        bool const finished = mCallbacks.isDecodeFinished
            ? mCallbacks.isDecodeFinished(item.requestId, *source.context, source.contextRow)
            : source.context->finishedStates[static_cast<size_t>(source.contextRow)] != 0;
        return PhaseDecodeCompletion{item.tokenCount + 1, finished};
    };
    result.onTerminal = [this](PhaseRequestSnapshot const& snapshot) {
        check::check(mRegistrations.find(snapshot.requestId) != mRegistrations.end(),
            "Terminal phase request has no registered source context.");
        releasePageBundles(snapshot.requestId);
        mRegistrations.erase(snapshot.requestId);
        if (mCallbacks.onTerminal)
        {
            mCallbacks.onTerminal(snapshot);
        }
        for (auto const& [observerId, observer] : mTerminalObservers)
        {
            static_cast<void>(observerId);
            observer(snapshot);
        }
        mPendingAdmissionRequired = true;
    };
    return result;
}

void PhaseContextServingFacade::registerSource(
    uint64_t requestId, DecodingInferenceContext& context, int32_t contextRow)
{
    check::check(contextRow >= 0 && contextRow < context.activeBatchSize,
        "Serving source row is outside the active request context.");
    check::check(context.phaseBatchState == nullptr, "A packed phase context cannot be registered as a source.");
    check::check(mRegistrations.find(requestId) == mRegistrations.end(), "Serving request ID is already registered.");
    for (auto const& [registeredId, source] : mRegistrations)
    {
        static_cast<void>(registeredId);
        check::check(source.context != &context || source.contextRow != contextRow,
            "Serving source context row is already registered.");
    }
    mRegistrations.emplace(requestId, Registration{&context, contextRow});
}

PhaseContextServingFacade::PageBundleReservation PhaseContextServingFacade::makePageBundleReservation(
    DecodingInferenceContext const& context, int32_t contextRow, int32_t promptTokenCount) const
{
    if (!mCacheManager.isPagedKVCache())
    {
        return {};
    }
    check::check(contextRow >= 0 && contextRow < context.activeBatchSize,
        "Serving page reservation row is outside the active request context.");
    check::check(context.maxGenerateLength > 0, "Serving page reservation requires a positive output limit.");
    int64_t const sequenceLength = static_cast<int64_t>(promptTokenCount) + context.maxGenerateLength;
    check::check(
        sequenceLength <= std::numeric_limits<int32_t>::max(), "Serving page reservation sequence length overflowed.");
    int32_t const fullBundles = mCacheManager.getPagedKVRequiredBundles(static_cast<int32_t>(sequenceLength));
    int32_t const promptBundles = mCacheManager.getPagedKVRequiredBundles(promptTokenCount);
    if (mPageReservationConfig.fullReservationPromptThresholdTokens > 0
        && promptTokenCount >= mPageReservationConfig.fullReservationPromptThresholdTokens)
    {
        return {fullBundles, fullBundles};
    }
    int32_t baseBundles = fullBundles;
    switch (mPageReservationConfig.mode)
    {
    case PhasePageReservationMode::kFull: break;
    case PhasePageReservationMode::kHeadroom:
    {
        int32_t const headroom = std::min(context.maxGenerateLength, mPageReservationConfig.outputHeadroomTokens);
        baseBundles = mCacheManager.getPagedKVRequiredBundles(promptTokenCount + headroom);
        break;
    }
    case PhasePageReservationMode::kBoundedOvercommit:
        baseBundles = std::max(promptBundles, fullBundles - mPageReservationConfig.maxOvercommitPageBundles);
        break;
    }
    return {baseBundles, fullBundles};
}

int32_t PhaseContextServingFacade::guaranteedPageBundles() const
{
    std::vector<int32_t> tails;
    tails.reserve(mPageBundleReservations.size());
    for (auto const& [requestId, reservation] : mPageBundleReservations)
    {
        static_cast<void>(requestId);
        tails.push_back(reservation.fullBundles - reservation.baseBundles);
    }
    std::sort(tails.begin(), tails.end(), std::greater<int32_t>());
    int32_t const growthCount = std::min<int32_t>(mPageReservationConfig.maxConcurrentGrowthRequests, tails.size());
    return mBaseReservedPageBundles + std::accumulate(tails.begin(), tails.begin() + growthCount, 0);
}

int32_t PhaseContextServingFacade::guaranteedPageBundlesWithReplacement(
    uint64_t replacedRequestId, PageBundleReservation replacement) const
{
    int32_t guaranteed = mBaseReservedPageBundles;
    auto const previous = mPageBundleReservations.find(replacedRequestId);
    check::check(previous != mPageBundleReservations.end(), "Replacement page reservation request is not active.");
    guaranteed += replacement.baseBundles - previous->second.baseBundles;
    std::vector<int32_t> tails;
    tails.reserve(mPageBundleReservations.size());
    for (auto const& [requestId, reservation] : mPageBundleReservations)
    {
        PageBundleReservation const& selected = requestId == replacedRequestId ? replacement : reservation;
        tails.push_back(selected.fullBundles - selected.baseBundles);
    }
    std::sort(tails.begin(), tails.end(), std::greater<int32_t>());
    int32_t const growthCount = std::min<int32_t>(mPageReservationConfig.maxConcurrentGrowthRequests, tails.size());
    guaranteed += std::accumulate(tails.begin(), tails.begin() + growthCount, 0);
    return guaranteed;
}

bool PhaseContextServingFacade::hasPageReservationCapacity(PageBundleReservation const& reservation) const
{
    KVPagePoolStats const pagePool = mCacheManager.getPagedKVPoolStats();
    std::vector<int32_t> tails;
    tails.reserve(mPageBundleReservations.size() + 1);
    tails.push_back(reservation.fullBundles - reservation.baseBundles);
    for (auto const& [requestId, active] : mPageBundleReservations)
    {
        static_cast<void>(requestId);
        tails.push_back(active.fullBundles - active.baseBundles);
    }
    std::sort(tails.begin(), tails.end(), std::greater<int32_t>());
    int32_t const growthCount = std::min<int32_t>(mPageReservationConfig.maxConcurrentGrowthRequests, tails.size());
    int32_t const guaranteed = mBaseReservedPageBundles + reservation.baseBundles
        + std::accumulate(tails.begin(), tails.begin() + growthCount, 0);
    return pagePool.totalBundles == 0 || guaranteed <= pagePool.totalBundles;
}

void PhaseContextServingFacade::reservePageBundles(uint64_t requestId, PageBundleReservation reservation)
{
    check::check(reservation.baseBundles >= 0 && reservation.fullBundles >= reservation.baseBundles,
        "Serving page reservation is invalid.");
    check::check(hasPageReservationCapacity(reservation), "Serving paged KV admission capacity is exhausted.");
    check::check(mPageBundleReservations.emplace(requestId, reservation).second,
        "Serving request already owns a paged KV admission reservation.");
    mBaseReservedPageBundles += reservation.baseBundles;
    selectDrainOwners();
}

void PhaseContextServingFacade::resizePageBundleReservation(uint64_t requestId, PageBundleReservation replacement)
{
    auto const reservation = mPageBundleReservations.find(requestId);
    check::check(reservation != mPageBundleReservations.end(), "Serving request has no paged KV reservation.");
    check::check(replacement.baseBundles >= 0 && replacement.fullBundles >= replacement.baseBundles,
        "Serving replacement page reservation is invalid.");
    KVPagePoolStats const pagePool = mCacheManager.getPagedKVPoolStats();
    check::check(pagePool.totalBundles == 0
            || guaranteedPageBundlesWithReplacement(requestId, replacement) <= pagePool.totalBundles,
        "Serving encoder handoff exceeds the available paged KV reservation capacity.");
    mBaseReservedPageBundles += replacement.baseBundles - reservation->second.baseBundles;
    check::check(mBaseReservedPageBundles >= 0, "Serving paged KV reservation accounting underflowed.");
    reservation->second = replacement;
    if (replacement.fullBundles == replacement.baseBundles)
    {
        mDrainRequestIds.erase(requestId);
    }
    selectDrainOwners();
}

void PhaseContextServingFacade::releasePageBundles(uint64_t requestId)
{
    auto const reservation = mPageBundleReservations.find(requestId);
    if (reservation == mPageBundleReservations.end())
    {
        return;
    }
    check::check(mBaseReservedPageBundles >= reservation->second.baseBundles,
        "Serving paged KV reservation accounting underflowed.");
    mBaseReservedPageBundles -= reservation->second.baseBundles;
    mDrainRequestIds.erase(requestId);
    mPageBundleReservations.erase(reservation);
    selectDrainOwners();
}

void PhaseContextServingFacade::selectDrainOwners()
{
    for (auto owner = mDrainRequestIds.begin(); owner != mDrainRequestIds.end();)
    {
        auto const reservation = mPageBundleReservations.find(*owner);
        if (reservation == mPageBundleReservations.end()
            || reservation->second.fullBundles == reservation->second.baseBundles)
        {
            owner = mDrainRequestIds.erase(owner);
        }
        else
        {
            ++owner;
        }
    }
    std::vector<std::pair<int32_t, uint64_t>> candidates;
    candidates.reserve(mPageBundleReservations.size() - mDrainRequestIds.size());
    for (auto const& [requestId, reservation] : mPageBundleReservations)
    {
        int32_t const tail = reservation.fullBundles - reservation.baseBundles;
        if (tail > 0 && mDrainRequestIds.find(requestId) == mDrainRequestIds.end())
        {
            candidates.emplace_back(tail, requestId);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](auto const& left, auto const& right) {
        return left.first > right.first || (left.first == right.first && left.second < right.second);
    });
    int32_t const availableGrowthLeases
        = std::max(0, mGrowthRequestLimit - static_cast<int32_t>(mDrainRequestIds.size()));
    int32_t const growthCount = std::min<int32_t>(availableGrowthLeases, candidates.size());
    for (int32_t index{}; index < growthCount; ++index)
    {
        mDrainRequestIds.insert(candidates[static_cast<size_t>(index)].second);
    }
}

void PhaseContextServingFacade::observePageReservationMetrics(PhaseDispatchMetrics const& metrics)
{
    if (!mPageReservationConfig.enableAdaptiveGrowthRequests || metrics.decodeBatchSize == 0
        || mPageReservationConfig.mode == PhasePageReservationMode::kFull)
    {
        return;
    }
    double const observedTpotUs = metrics.decodeQueueWaitUs + static_cast<double>(metrics.decodeGpuMs) * 1000.0;
    float const pressure = static_cast<float>(observedTpotUs / mPageReservationConfig.growthTpotTargetUs);
    float const alpha = mPageReservationConfig.growthPressureEwmaAlpha;
    mGrowthTpotPressure = mGrowthMetricSamples > 0 ? alpha * pressure + (1.0F - alpha) * mGrowthTpotPressure : pressure;
    ++mGrowthMetricSamples;
    if (mGrowthMetricSamples % mPageReservationConfig.growthAdjustmentInterval != 0)
    {
        return;
    }
    int32_t nextLimit = mGrowthRequestLimit;
    if (mGrowthTpotPressure >= mPageReservationConfig.growthScaleUpThreshold)
    {
        nextLimit = std::min(mPageReservationConfig.maxConcurrentGrowthRequests,
            mGrowthRequestLimit + mPageReservationConfig.growthAdjustmentStep);
    }
    else if (mGrowthTpotPressure <= mPageReservationConfig.growthScaleDownThreshold)
    {
        nextLimit = std::max(mPageReservationConfig.minConcurrentGrowthRequests,
            mGrowthRequestLimit - mPageReservationConfig.growthAdjustmentStep);
    }
    if (nextLimit != mGrowthRequestLimit)
    {
        mGrowthRequestLimit = nextLimit;
        selectDrainOwners();
    }
}

bool PhaseContextServingFacade::isPageWorkEligible(PhaseWorkItem const& item, bool prefill) const
{
    if (!mCacheManager.isPagedKVCache())
    {
        return true;
    }
    auto const reservation = mPageBundleReservations.find(item.requestId);
    check::check(reservation != mPageBundleReservations.end(), "Queued phase work has no paged KV reservation.");
    int32_t const targetTokens = prefill ? item.promptTokenCount : item.tokenCount + 1;
    int32_t const targetBundles = mCacheManager.getPagedKVRequiredBundles(targetTokens);
    int32_t const limit = mDrainRequestIds.find(item.requestId) != mDrainRequestIds.end()
        ? reservation->second.fullBundles
        : reservation->second.baseBundles;
    return targetBundles <= limit;
}

void PhaseContextServingFacade::updateAdmissionPageReservation(PhaseAdmissionResult& result) const
{
    KVPagePoolStats const pagePool = mCacheManager.getPagedKVPoolStats();
    result.reservedPageBundles = guaranteedPageBundles();
    result.reservationAvailableBundles
        = pagePool.totalBundles > 0 ? pagePool.totalBundles - result.reservedPageBundles : 0;
}

int32_t PhaseContextServingFacade::submit(uint64_t requestId, DecodingInferenceContext& context, int32_t contextRow,
    int32_t promptTokenCount, PhaseSchedulingHints scheduling)
{
    registerSource(requestId, context, contextRow);
    PageBundleReservation const pageReservation = makePageBundleReservation(context, contextRow, promptTokenCount);
    try
    {
        reservePageBundles(requestId, pageReservation);
        return mLifecycle->submit(requestId, promptTokenCount, scheduling);
    }
    catch (...)
    {
        releasePageBundles(requestId);
        mRegistrations.erase(requestId);
        throw;
    }
}

int32_t PhaseContextServingFacade::reserveForEncoder(uint64_t requestId, DecodingInferenceContext& context,
    int32_t contextRow, int32_t promptTokenCountEstimate, PhaseSchedulingHints scheduling)
{
    registerSource(requestId, context, contextRow);
    PageBundleReservation const pageReservation
        = makePageBundleReservation(context, contextRow, promptTokenCountEstimate);
    try
    {
        reservePageBundles(requestId, pageReservation);
        return mLifecycle->reserveForEncoder(requestId, promptTokenCountEstimate, scheduling);
    }
    catch (...)
    {
        releasePageBundles(requestId);
        mRegistrations.erase(requestId);
        throw;
    }
}

void PhaseContextServingFacade::beginPrefillAfterEncoder(PhaseWorkItem const& item)
{
    auto const snapshot = mLifecycle->request(item.requestId);
    check::check(snapshot.has_value(), "Encoder handoff has no phase request reservation.");
    check::check(snapshot->status == PhaseRequestStatus::kEncoder, "Encoder handoff request is not encoder-pending.");
    check::check(snapshot->kvSlotId == item.kvSlotId, "Encoder handoff changed the stable KV slot.");
    check::check(item.tokenOffset == 0 && item.tokenCount > 0,
        "Encoder handoff must provide a non-empty prompt at token offset zero.");
    int32_t const promptTokenCount = item.promptTokenCount > 0 ? item.promptTokenCount : item.tokenCount;
    check::check(promptTokenCount == item.tokenCount, "Encoder handoff must provide the complete prompt in V1.");
    Registration const& source = registration(item.requestId);
    resizePageBundleReservation(
        item.requestId, makePageBundleReservation(*source.context, source.contextRow, promptTokenCount));
    mLifecycle->beginPrefill(item.requestId, promptTokenCount, item.allowChunkedPrefill);
}

PhaseAdmissionResult PhaseContextServingFacade::submitOrQueue(uint64_t requestId, DecodingInferenceContext& context,
    int32_t contextRow, int32_t promptTokenCount, PhaseSchedulingHints scheduling)
{
    check::check(promptTokenCount > 0, "Phase request prompt length must be positive.");
    registerSource(requestId, context, contextRow);
    PhaseAdmissionResult result{};
    result.requestId = requestId;
    result.kvSlotId = -1;
    result.status = PhaseAdmissionStatus::kPending;
    result.availableSlots = mLifecycle->availableSlotCount();
    result.pendingQueueDepth = mPendingAdmissions.size();
    result.pagePool = mCacheManager.getPagedKVPoolStats();
    PageBundleReservation const pageReservation = makePageBundleReservation(context, contextRow, promptTokenCount);
    check::check(result.pagePool.totalBundles == 0 || pageReservation.fullBundles <= result.pagePool.totalBundles,
        "Serving request exceeds the complete paged KV pool capacity.");
    try
    {
        if (result.availableSlots > 0 && hasPageReservationCapacity(pageReservation))
        {
            reservePageBundles(requestId, pageReservation);
            result.kvSlotId = mLifecycle->submit(requestId, promptTokenCount, scheduling);
            result.status = PhaseAdmissionStatus::kAdmitted;
        }
        else
        {
            check::check(mPendingAdmissions.size() < mMaxPendingAdmissions, "Serving pending admission queue is full.");
            mPendingAdmissions.push_back({requestId, promptTokenCount, pageReservation, scheduling});
        }
    }
    catch (...)
    {
        releasePageBundles(requestId);
        mRegistrations.erase(requestId);
        throw;
    }
    result.pendingQueueDepth = mPendingAdmissions.size();
    updateAdmissionPageReservation(result);
    if (mCallbacks.onAdmission)
    {
        mCallbacks.onAdmission(result);
    }
    return result;
}

void PhaseContextServingFacade::admitPendingRequests()
{
    if (mLifecycle->busy())
    {
        mPendingAdmissionRequired = !mPendingAdmissions.empty();
        return;
    }
    mPendingAdmissionRequired = false;
    while (!mPendingAdmissions.empty() && mLifecycle->availableSlotCount() > 0
        && hasPageReservationCapacity(mPendingAdmissions.front().pageReservation))
    {
        PendingAdmission const admission = mPendingAdmissions.front();
        reservePageBundles(admission.requestId, admission.pageReservation);
        int32_t slot{};
        try
        {
            slot = mLifecycle->submit(admission.requestId, admission.promptTokenCount, admission.scheduling);
        }
        catch (...)
        {
            releasePageBundles(admission.requestId);
            throw;
        }
        mPendingAdmissions.pop_front();
        if (mCallbacks.onAdmission)
        {
            PhaseAdmissionResult result{};
            result.requestId = admission.requestId;
            result.kvSlotId = slot;
            result.status = PhaseAdmissionStatus::kAdmitted;
            result.availableSlots = mLifecycle->availableSlotCount();
            result.pendingQueueDepth = mPendingAdmissions.size();
            result.pagePool = mCacheManager.getPagedKVPoolStats();
            updateAdmissionPageReservation(result);
            mCallbacks.onAdmission(result);
        }
    }
}

bool PhaseContextServingFacade::cancel(uint64_t requestId)
{
    auto const pending = std::find_if(mPendingAdmissions.begin(), mPendingAdmissions.end(),
        [requestId](PendingAdmission const& admission) { return admission.requestId == requestId; });
    if (pending != mPendingAdmissions.end())
    {
        int32_t const promptTokenCount = pending->promptTokenCount;
        mPendingAdmissions.erase(pending);
        check::check(mRegistrations.erase(requestId) == 1, "Pending phase request has no registered source context.");
        if (mCallbacks.onTerminal)
        {
            mCallbacks.onTerminal({requestId, -1, promptTokenCount, 0, PhaseRequestStatus::kCancelled});
        }
        return true;
    }

    bool const cancelled = mLifecycle->cancel(requestId);
    if (cancelled && mPendingAdmissionRequired)
    {
        admitPendingRequests();
    }
    return cancelled;
}

void PhaseContextServingFacade::enqueuePrefillBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream)
{
    int32_t newSlotCount{};
    int32_t* hostSlotIds = mHostAdmissionSlotIds.dataPointer<int32_t>();
    for (PhaseWorkItem const& item : batch)
    {
        if (item.tokenOffset == 0)
        {
            hostSlotIds[newSlotCount++] = item.kvSlotId;
        }
    }
    if (newSlotCount > 0)
    {
        check::check(mHostAdmissionSlotIds.reshape({newSlotCount}), "Host admission slot IDs reshape failed.");
        check::check(mDeviceAdmissionSlotIds.reshape({newSlotCount}), "Admission slot IDs reshape failed.");
        CUDA_CHECK(cudaMemcpyAsync(mDeviceAdmissionSlotIds.rawPointer(), mHostAdmissionSlotIds.rawPointer(),
            newSlotCount * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
        mCacheManager.clearPhaseKVCacheLengths(mDeviceAdmissionSlotIds, stream);
    }
    if (!mPrefillAdapter)
    {
        mCallbacks.enqueuePrefill(batch, stream);
        return;
    }

    std::vector<PhasePrefillContextRow> rows;
    rows.reserve(batch.size());
    for (PhaseWorkItem const& item : batch)
    {
        Registration& source = registration(item.requestId);
        rows.push_back({item.requestId, source.context, source.contextRow, item.kvSlotId, item.tokenOffset,
            item.tokenCount, item.promptTokenCount});
    }
    mPrefillAdapter->pack(rows, stream);
    mCallbacks.enqueuePackedPrefill(*mPrefillAdapter);
}

void PhaseContextServingFacade::enqueueDecodeBatch(std::vector<PhaseWorkItem> const& batch, cudaStream_t stream)
{
    std::vector<PhaseContextRow> rows;
    rows.reserve(batch.size());
    for (PhaseWorkItem const& item : batch)
    {
        Registration& source = registration(item.requestId);
        rows.push_back({item.requestId, source.context, source.contextRow, item.kvSlotId, item.tokenCount});
    }
    mDecodeAdapter.packDecode(rows, stream);
    if (mCallbacks.enqueuePackedDecode)
    {
        mCallbacks.enqueuePackedDecode(mDecodeAdapter);
    }
    else
    {
        mCallbacks.enqueueDecode(mDecodeAdapter.packedContext());
    }
}

void PhaseContextServingFacade::completeDecodeBatch(std::vector<PhaseWorkItem> const& batch)
{
    check::check(mDecodeAdapter.packed(), "Serving decode completion has no packed context.");
    check::check(mDecodeAdapter.workItems().size() == batch.size(),
        "Serving decode completion batch size does not match its packed context.");
    if (mCallbacks.completePackedDecode)
    {
        mCallbacks.completePackedDecode(mDecodeAdapter);
    }
    else
    {
        mCallbacks.completeDecode(mDecodeAdapter.packedContext());
    }
    mDecodeAdapter.scatterDecode();
}

PhaseContextServingFacade::Registration& PhaseContextServingFacade::registration(uint64_t requestId)
{
    auto const it = mRegistrations.find(requestId);
    check::check(it != mRegistrations.end(), "Phase request has no registered source context.");
    return it->second;
}

PhaseContextServingFacade::Registration const& PhaseContextServingFacade::registration(uint64_t requestId) const
{
    auto const it = mRegistrations.find(requestId);
    check::check(it != mRegistrations.end(), "Phase request has no registered source context.");
    return it->second;
}

bool PhaseContextServingFacade::dispatchNext()
{
    admitPendingRequests();
    return mLifecycle->dispatchNext();
}

bool PhaseContextServingFacade::poll()
{
    bool const completed = mLifecycle->poll();
    if (completed && mPendingAdmissionRequired)
    {
        admitPendingRequests();
    }
    return completed;
}

void PhaseContextServingFacade::wait()
{
    mLifecycle->wait();
    if (mPendingAdmissionRequired)
    {
        admitPendingRequests();
    }
}

void PhaseContextServingFacade::runUntilIdle(size_t maxDispatches)
{
    check::check(maxDispatches > 0, "Serving facade maxDispatches must be positive.");
    size_t dispatches{};
    while (!empty())
    {
        if (!busy())
        {
            check::check(dispatches < maxDispatches, "Serving facade exceeded its dispatch limit.");
            check::check(dispatchNext(), "Serving facade failed to dispatch queued work.");
            ++dispatches;
        }
        wait();
    }
}

bool PhaseContextServingFacade::empty() const noexcept
{
    return mPendingAdmissions.empty() && mLifecycle->empty();
}

bool PhaseContextServingFacade::hasQueuedPhaseWork() const noexcept
{
    return mLifecycle->hasQueuedWork();
}

bool PhaseContextServingFacade::busy() const noexcept
{
    return mLifecycle->busy();
}

size_t PhaseContextServingFacade::activeRequestCount() const noexcept
{
    return mLifecycle->activeRequestCount();
}

size_t PhaseContextServingFacade::pendingRequestCount() const noexcept
{
    return mPendingAdmissions.size();
}

size_t PhaseContextServingFacade::registeredRequestCount() const noexcept
{
    return mRegistrations.size();
}

int32_t PhaseContextServingFacade::availableSlotCount() const noexcept
{
    return mLifecycle->availableSlotCount();
}

PhasePageReservationStats PhaseContextServingFacade::pageReservationStats() const
{
    return {guaranteedPageBundles(), mBaseReservedPageBundles, mGrowthRequestLimit,
        static_cast<int32_t>(mDrainRequestIds.size()), mGrowthTpotPressure};
}

std::optional<PhaseRequestSnapshot> PhaseContextServingFacade::request(uint64_t requestId) const
{
    auto const pending = std::find_if(mPendingAdmissions.begin(), mPendingAdmissions.end(),
        [requestId](PendingAdmission const& admission) { return admission.requestId == requestId; });
    if (pending != mPendingAdmissions.end())
    {
        return PhaseRequestSnapshot{
            requestId, -1, pending->promptTokenCount, 0, PhaseRequestStatus::kPending, pending->scheduling};
    }
    return mLifecycle->request(requestId);
}

CUcontext PhaseContextServingFacade::cudaContext() const noexcept
{
    return mLifecycle->cudaContext();
}

size_t PhaseContextServingFacade::addTerminalObserver(std::function<void(PhaseRequestSnapshot const&)> observer)
{
    check::check(static_cast<bool>(observer), "Serving terminal observer must be callable.");
    size_t const observerId = mNextTerminalObserverId++;
    mTerminalObservers.emplace(observerId, std::move(observer));
    return observerId;
}

void PhaseContextServingFacade::removeTerminalObserver(size_t observerId) noexcept
{
    mTerminalObservers.erase(observerId);
}

} // namespace rt
} // namespace trt_edgellm
