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

#include "runtime/scheduling/phaseActivityTimeline.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <utility>

namespace trt_edgellm::rt
{
namespace
{

constexpr uint8_t kENCODER_MASK = static_cast<uint8_t>(PhaseActivityKind::kEncoder);
constexpr uint8_t kPREFILL_MASK = static_cast<uint8_t>(PhaseActivityKind::kPrefill);
constexpr uint8_t kDECODE_MASK = static_cast<uint8_t>(PhaseActivityKind::kDecode);
constexpr uint8_t kCOPY_MASK = static_cast<uint8_t>(PhaseActivityKind::kCopy);
constexpr uint8_t kEPD_MASK = kENCODER_MASK | kPREFILL_MASK | kDECODE_MASK;
constexpr uint8_t kALL_MASK = kEPD_MASK | kCOPY_MASK;

struct Boundary
{
    float timeMs{};
    uint8_t bit{};
    int32_t delta{};
};

std::filesystem::path csvPath(std::filesystem::path const& prefix, char const* suffix)
{
    return std::filesystem::path(prefix.string() + suffix);
}

} // namespace

char const* phaseActivityKindName(PhaseActivityKind kind) noexcept
{
    char const* result{"unknown"};
    switch (kind)
    {
    case PhaseActivityKind::kEncoder: result = "encoder"; break;
    case PhaseActivityKind::kPrefill: result = "prefill"; break;
    case PhaseActivityKind::kDecode: result = "decode"; break;
    case PhaseActivityKind::kCopy: result = "copy"; break;
    }
    return result;
}

std::string phaseActivityMaskString(uint8_t mask)
{
    return std::bitset<PhaseActivitySummary::kACTIVITY_COUNT>(mask & kALL_MASK).to_string();
}

std::vector<PhaseActivitySegment> phaseActivitySegments(
    std::vector<PhaseActivityInterval> const& intervals, float windowEndMs)
{
    ELLM_CHECK(
        std::isfinite(windowEndMs) && windowEndMs >= 0.0F, "Activity window end must be finite and non-negative");
    std::vector<Boundary> boundaries;
    boundaries.reserve(intervals.size() * 2U);
    float lastEndMs = windowEndMs;
    for (PhaseActivityInterval const& interval : intervals)
    {
        uint8_t const bit = static_cast<uint8_t>(interval.kind);
        ELLM_CHECK(bit == kENCODER_MASK || bit == kPREFILL_MASK || bit == kDECODE_MASK || bit == kCOPY_MASK,
            "Activity interval has an invalid kind");
        ELLM_CHECK(std::isfinite(interval.startMs) && std::isfinite(interval.endMs) && interval.startMs >= 0.0F
                && interval.endMs >= interval.startMs,
            "Activity interval has invalid epoch-relative bounds");
        boundaries.push_back({interval.startMs, bit, 1});
        boundaries.push_back({interval.endMs, bit, -1});
        lastEndMs = std::max(lastEndMs, interval.endMs);
    }
    if (lastEndMs <= 0.0F)
    {
        return {};
    }
    std::sort(boundaries.begin(), boundaries.end(),
        [](Boundary const& left, Boundary const& right) { return left.timeMs < right.timeMs; });

    std::array<int32_t, PhaseActivitySummary::kACTIVITY_COUNT> activeCounts{};
    auto currentMask = [&activeCounts]() {
        uint8_t mask{};
        constexpr std::array<uint8_t, PhaseActivitySummary::kACTIVITY_COUNT> kBITS{
            kENCODER_MASK, kPREFILL_MASK, kDECODE_MASK, kCOPY_MASK};
        for (size_t index{}; index < kBITS.size(); ++index)
        {
            if (activeCounts[index] > 0)
            {
                mask |= kBITS[index];
            }
        }
        return mask;
    };
    auto bitIndex = [](uint8_t bit) {
        if (bit == kENCODER_MASK)
        {
            return size_t{0U};
        }
        if (bit == kPREFILL_MASK)
        {
            return size_t{1U};
        }
        if (bit == kDECODE_MASK)
        {
            return size_t{2U};
        }
        return size_t{3U};
    };

    std::vector<PhaseActivitySegment> result;
    float cursorMs{};
    size_t boundaryIndex{};
    while (boundaryIndex < boundaries.size())
    {
        float const boundaryMs = boundaries[boundaryIndex].timeMs;
        if (boundaryMs > cursorMs)
        {
            result.push_back({cursorMs, boundaryMs, currentMask()});
            cursorMs = boundaryMs;
        }
        while (boundaryIndex < boundaries.size() && boundaries[boundaryIndex].timeMs == boundaryMs)
        {
            Boundary const& boundary = boundaries[boundaryIndex++];
            size_t const index = bitIndex(boundary.bit);
            activeCounts[index] += boundary.delta;
            ELLM_CHECK(activeCounts[index] >= 0, "Activity boundary closes an inactive phase");
        }
    }
    if (lastEndMs > cursorMs)
    {
        result.push_back({cursorMs, lastEndMs, currentMask()});
    }
    return result;
}

std::vector<PhaseActivitySegment> phaseActivityActiveSpanSegments(std::vector<PhaseActivityInterval> const& intervals)
{
    if (intervals.empty())
    {
        return {};
    }
    float firstStartMs = intervals.front().startMs;
    float lastEndMs = intervals.front().endMs;
    for (PhaseActivityInterval const& interval : intervals)
    {
        firstStartMs = std::min(firstStartMs, interval.startMs);
        lastEndMs = std::max(lastEndMs, interval.endMs);
    }
    std::vector<PhaseActivitySegment> const epochSegments = phaseActivitySegments(intervals, lastEndMs);
    std::vector<PhaseActivitySegment> result;
    result.reserve(epochSegments.size());
    for (PhaseActivitySegment const& segment : epochSegments)
    {
        float const croppedStartMs = std::max(segment.startMs, firstStartMs);
        float const croppedEndMs = std::min(segment.endMs, lastEndMs);
        if (croppedEndMs > croppedStartMs)
        {
            result.push_back({croppedStartMs - firstStartMs, croppedEndMs - firstStartMs, segment.mask});
        }
    }
    return result;
}

PhaseActivitySummary phaseActivitySummary(std::vector<PhaseActivitySegment> const& segments, float windowEndMs)
{
    PhaseActivitySummary result;
    for (PhaseActivitySegment const& segment : segments)
    {
        ELLM_CHECK(segment.mask < PhaseActivitySummary::kMASK_COUNT && segment.startMs >= 0.0F
                && segment.endMs >= segment.startMs,
            "Activity segment is invalid");
        double const durationMs = static_cast<double>(segment.endMs - segment.startMs);
        result.maskMs[segment.mask] += durationMs;
        result.windowMs = std::max(result.windowMs, static_cast<double>(segment.endMs));
    }
    result.windowMs = std::max(result.windowMs, static_cast<double>(windowEndMs));
    for (size_t mask{}; mask < result.maskMs.size(); ++mask)
    {
        for (size_t bit{}; bit < result.activityMs.size(); ++bit)
        {
            if ((mask & (size_t{1U} << bit)) != 0U)
            {
                result.activityMs[bit] += result.maskMs[mask];
            }
        }
    }
    result.allIdleMs = result.maskMs[0U];
    result.epdIdleMs = result.maskMs[0U] + result.maskMs[kCOPY_MASK];
    result.anyActivityMs = result.windowMs - result.allIdleMs;
    result.anyEpdMs = result.windowMs - result.epdIdleMs;
    result.epdTripleMs = result.maskMs[kEPD_MASK] + result.maskMs[kALL_MASK];
    result.fourWayMs = result.maskMs[kALL_MASK];
    return result;
}

bool phaseActivityIntervalsOverlap(
    double leftStartMs, double leftEndMs, double rightStartMs, double rightEndMs) noexcept
{
    return std::isfinite(leftStartMs) && std::isfinite(rightStartMs) && !std::isnan(leftEndMs)
        && !std::isnan(rightEndMs) && leftEndMs > leftStartMs && rightEndMs > rightStartMs && leftStartMs < rightEndMs
        && rightStartMs < leftEndMs;
}

PhaseActivityTimelineRecorder::PhaseActivityTimelineRecorder(cudaStream_t epochStream, uint8_t captureMask)
    : mCaptureMask(captureMask)
{
    ELLM_CHECK((captureMask & static_cast<uint8_t>(~kPHASE_ACTIVITY_ALL_MASK)) == 0U,
        "Activity capture mask contains unknown phase bits");
    initializeEpoch(epochStream);
}

PhaseActivityTimelineRecorder::~PhaseActivityTimelineRecorder() noexcept
{
    std::lock_guard<std::mutex> lock(mMutex);
    for (auto& entry : mPending)
    {
        if (entry.second.closed)
        {
            static_cast<void>(cudaEventSynchronize(entry.second.end));
        }
        destroy(entry.second);
    }
    if (mEpoch != nullptr)
    {
        static_cast<void>(cudaEventDestroy(mEpoch));
    }
}

void PhaseActivityTimelineRecorder::initializeEpoch(cudaStream_t epochStream)
{
    ELLM_CHECK(epochStream != nullptr, "Activity timeline requires an explicit epoch stream");
    CUcontext context{};
    CUDA_DRIVER_CHECK(cuStreamGetCtx(epochStream, &context));
    ELLM_CHECK(context != nullptr, "Activity timeline epoch stream has no CUDA context");
    if (mCudaContext != nullptr)
    {
        ELLM_CHECK(context == mCudaContext, "Activity timeline reset changed CUDA contexts");
    }
    mCudaContext = context;
    CUDA_CHECK(cudaEventCreate(&mEpoch));
    CUDA_CHECK(cudaEventRecord(mEpoch, epochStream));
    CUDA_CHECK(cudaEventSynchronize(mEpoch));
}

PhaseActivityTimelineRecorder::Token PhaseActivityTimelineRecorder::begin(
    PhaseActivityKind kind, cudaStream_t stream, std::string name, uint64_t correlationId)
{
    if ((mCaptureMask & static_cast<uint8_t>(kind)) == 0U)
    {
        return 0U;
    }
    ELLM_CHECK(stream != nullptr, "Activity interval requires an explicit CUDA stream");
    CUcontext context{};
    CUDA_DRIVER_CHECK(cuStreamGetCtx(stream, &context));
    ELLM_CHECK(context == mCudaContext, "Activity interval stream uses a different CUDA context");
    PendingInterval pending;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        pending.token = mNextToken++;
    }
    pending.correlationId = correlationId;
    pending.kind = kind;
    pending.name = name.empty() ? phaseActivityKindName(kind) : std::move(name);
    pending.stream = stream;
    CUDA_CHECK(cudaEventCreate(&pending.start));
    CUDA_CHECK(cudaEventCreate(&pending.end));
    CUDA_CHECK(cudaEventRecord(pending.start, stream));
    Token const token = pending.token;
    {
        std::lock_guard<std::mutex> lock(mMutex);
        ELLM_CHECK(mPending.emplace(token, std::move(pending)).second, "Duplicate activity interval token");
    }
    return token;
}

void PhaseActivityTimelineRecorder::end(Token token, cudaStream_t stream)
{
    if (token == 0U)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    auto const it = mPending.find(token);
    ELLM_CHECK(it != mPending.end(), "Unknown activity interval token");
    ELLM_CHECK(!it->second.closed, "Activity interval was already closed");
    ELLM_CHECK(stream == it->second.stream, "Activity interval must end on its starting stream");
    CUDA_CHECK(cudaEventRecord(it->second.end, stream));
    it->second.closed = true;
}

void PhaseActivityTimelineRecorder::cancel(Token token) noexcept
{
    if (token == 0U)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(mMutex);
    auto const it = mPending.find(token);
    if (it == mPending.end())
    {
        return;
    }
    destroy(it->second);
    mPending.erase(it);
}

bool PhaseActivityTimelineRecorder::collect(PendingInterval& pending, bool synchronize)
{
    ELLM_CHECK(pending.closed, "Cannot collect an open activity interval");
    if (synchronize)
    {
        CUDA_CHECK(cudaEventSynchronize(pending.end));
    }
    else
    {
        cudaError_t const status = cudaEventQuery(pending.end);
        if (status == cudaErrorNotReady)
        {
            return false;
        }
        CUDA_CHECK(status);
    }
    PhaseActivityInterval interval;
    interval.intervalId = pending.token;
    interval.correlationId = pending.correlationId;
    interval.kind = pending.kind;
    interval.name = std::move(pending.name);
    CUDA_CHECK(cudaEventElapsedTime(&interval.startMs, mEpoch, pending.start));
    CUDA_CHECK(cudaEventElapsedTime(&interval.endMs, mEpoch, pending.end));
    mOverlapWindow.push_back({interval.kind, interval.startMs, interval.endMs});
    // Encoder-only capture serves online cost labels and never exports history.
    if (mCaptureMask != kPHASE_ACTIVITY_ENCODER_MASK)
    {
        mIntervals.push_back(std::move(interval));
    }
    destroy(pending);
    return true;
}

size_t PhaseActivityTimelineRecorder::poll()
{
    std::lock_guard<std::mutex> lock(mMutex);
    size_t collected{};
    for (auto pending = mPending.begin(); pending != mPending.end();)
    {
        if (pending->second.closed && collect(pending->second, false))
        {
            pending = mPending.erase(pending);
            ++collected;
        }
        else
        {
            ++pending;
        }
    }
    return collected;
}

void PhaseActivityTimelineRecorder::drain()
{
    std::lock_guard<std::mutex> lock(mMutex);
    ELLM_CHECK(std::all_of(mPending.begin(), mPending.end(), [](auto const& entry) { return entry.second.closed; }),
        "Every activity interval must be closed before drain");
    for (auto& entry : mPending)
    {
        static_cast<void>(collect(entry.second, true));
    }
    mPending.clear();
}

void PhaseActivityTimelineRecorder::reset(cudaStream_t epochStream)
{
    drain();
    std::lock_guard<std::mutex> lock(mMutex);
    mIntervals.clear();
    mOverlapWindow.clear();
    mNextToken = 1U;
    if (mEpoch != nullptr)
    {
        CUDA_CHECK(cudaEventDestroy(mEpoch));
        mEpoch = nullptr;
    }
    initializeEpoch(epochStream);
}

std::vector<PhaseActivityInterval> PhaseActivityTimelineRecorder::intervals() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    std::vector<PhaseActivityInterval> result = mIntervals;
    std::sort(result.begin(), result.end(), [](PhaseActivityInterval const& left, PhaseActivityInterval const& right) {
        return left.startMs < right.startMs || (left.startMs == right.startMs && left.intervalId < right.intervalId);
    });
    return result;
}

std::optional<bool> PhaseActivityTimelineRecorder::overlaps(PhaseActivityKind kind, cudaEvent_t start, cudaEvent_t end)
{
    if ((mCaptureMask & static_cast<uint8_t>(kind)) == 0U)
    {
        return std::nullopt;
    }
    ELLM_CHECK(start != nullptr && end != nullptr, "Activity overlap requires a complete phase event pair");
    CUDA_CHECK(cudaEventQuery(start));
    CUDA_CHECK(cudaEventQuery(end));
    float actionStartMs{};
    float actionEndMs{};
    CUDA_CHECK(cudaEventElapsedTime(&actionStartMs, mEpoch, start));
    CUDA_CHECK(cudaEventElapsedTime(&actionEndMs, mEpoch, end));
    if (actionEndMs <= actionStartMs)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(mMutex);
    for (OverlapInterval const& interval : mOverlapWindow)
    {
        if (interval.kind == kind
            && phaseActivityIntervalsOverlap(interval.startMs, interval.endMs, actionStartMs, actionEndMs))
        {
            return true;
        }
    }
    bool pendingOverlapUnknown{};
    for (auto const& entry : mPending)
    {
        PendingInterval const& interval = entry.second;
        if (interval.kind != kind)
        {
            continue;
        }
        cudaError_t const startStatus = cudaEventQuery(interval.start);
        if (startStatus == cudaErrorNotReady)
        {
            continue;
        }
        CUDA_CHECK(startStatus);
        float intervalStartMs{};
        CUDA_CHECK(cudaEventElapsedTime(&intervalStartMs, mEpoch, interval.start));
        if (!interval.closed)
        {
            pendingOverlapUnknown = pendingOverlapUnknown
                || phaseActivityIntervalsOverlap(
                    intervalStartMs, std::numeric_limits<double>::infinity(), actionStartMs, actionEndMs);
            continue;
        }
        cudaError_t const endStatus = cudaEventQuery(interval.end);
        if (endStatus == cudaSuccess)
        {
            float intervalEndMs{};
            CUDA_CHECK(cudaEventElapsedTime(&intervalEndMs, mEpoch, interval.end));
            if (phaseActivityIntervalsOverlap(intervalStartMs, intervalEndMs, actionStartMs, actionEndMs))
            {
                return true;
            }
        }
        else if (endStatus == cudaErrorNotReady)
        {
            if (intervalStartMs < actionEndMs)
            {
                return true;
            }
        }
        else
        {
            CUDA_CHECK(endStatus);
        }
    }
    return pendingOverlapUnknown ? std::nullopt : std::optional<bool>{false};
}

void PhaseActivityTimelineRecorder::retireOverlapWindow(cudaEvent_t earliestActionStart)
{
    ELLM_CHECK(earliestActionStart != nullptr, "Overlap-window retirement requires an action-start event");
    CUDA_CHECK(cudaEventQuery(earliestActionStart));
    float watermarkMs{};
    CUDA_CHECK(cudaEventElapsedTime(&watermarkMs, mEpoch, earliestActionStart));
    std::lock_guard<std::mutex> lock(mMutex);
    mOverlapWindow.erase(std::remove_if(mOverlapWindow.begin(), mOverlapWindow.end(),
                             [watermarkMs](OverlapInterval const& interval) { return interval.endMs <= watermarkMs; }),
        mOverlapWindow.end());
}

std::vector<PhaseActivitySegment> PhaseActivityTimelineRecorder::segments() const
{
    return phaseActivityActiveSpanSegments(intervals());
}

PhaseActivitySummary PhaseActivityTimelineRecorder::summary() const
{
    return phaseActivitySummary(segments());
}

size_t PhaseActivityTimelineRecorder::pendingCount() const
{
    std::lock_guard<std::mutex> lock(mMutex);
    return mPending.size();
}

void PhaseActivityTimelineRecorder::writeCsv(std::filesystem::path const& prefix) const
{
    std::vector<PhaseActivityInterval> const collected = intervals();
    std::vector<PhaseActivitySegment> const maskSegments = phaseActivityActiveSpanSegments(collected);
    PhaseActivitySummary const totals = phaseActivitySummary(maskSegments);

    std::ofstream intervalOutput(csvPath(prefix, "-intervals.csv"));
    ELLM_CHECK(intervalOutput.good(), "Failed to open phase activity interval CSV");
    intervalOutput << "interval_index,interval_id,correlation_id,kind,bit,name,start_ms,end_ms,duration_ms\n";
    intervalOutput << std::fixed << std::setprecision(6);
    for (size_t index{}; index < collected.size(); ++index)
    {
        PhaseActivityInterval const& interval = collected[index];
        intervalOutput << index << ',' << interval.intervalId << ',' << interval.correlationId << ','
                       << phaseActivityKindName(interval.kind) << ','
                       << phaseActivityMaskString(static_cast<uint8_t>(interval.kind)) << ',' << interval.name << ','
                       << interval.startMs << ',' << interval.endMs << ',' << interval.endMs - interval.startMs << '\n';
    }

    std::ofstream segmentOutput(csvPath(prefix, "-segments.csv"));
    ELLM_CHECK(segmentOutput.good(), "Failed to open phase activity segment CSV");
    segmentOutput << "segment_index,active_span_start_ms,active_span_end_ms,duration_ms,mask,binary_mask,encoder,"
                     "prefill,decode,copy,"
                     "active_streams\n";
    segmentOutput << std::fixed << std::setprecision(6);
    for (size_t index{}; index < maskSegments.size(); ++index)
    {
        PhaseActivitySegment const& segment = maskSegments[index];
        int32_t activeStreams{};
        for (uint8_t bit : {kENCODER_MASK, kPREFILL_MASK, kDECODE_MASK, kCOPY_MASK})
        {
            activeStreams += (segment.mask & bit) != 0U ? 1 : 0;
        }
        segmentOutput << index << ',' << segment.startMs << ',' << segment.endMs << ','
                      << segment.endMs - segment.startMs << ',' << static_cast<int32_t>(segment.mask) << ','
                      << phaseActivityMaskString(segment.mask) << ',' << ((segment.mask & kENCODER_MASK) != 0U) << ','
                      << ((segment.mask & kPREFILL_MASK) != 0U) << ',' << ((segment.mask & kDECODE_MASK) != 0U) << ','
                      << ((segment.mask & kCOPY_MASK) != 0U) << ',' << activeStreams << '\n';
    }

    std::ofstream summaryOutput(csvPath(prefix, "-summary.csv"));
    ELLM_CHECK(summaryOutput.good(), "Failed to open phase activity summary CSV");
    summaryOutput << "metric,mask,binary_mask,duration_ms,ratio\n";
    summaryOutput << std::fixed << std::setprecision(6);
    auto const writeSummary = [&](char const* metric, int32_t mask, double durationMs) {
        double const ratio = totals.windowMs > 0.0 ? durationMs / totals.windowMs : 0.0;
        summaryOutput << metric << ',';
        if (mask >= 0)
        {
            summaryOutput << mask << ',' << phaseActivityMaskString(static_cast<uint8_t>(mask));
        }
        else
        {
            summaryOutput << ',';
        }
        summaryOutput << ',' << durationMs << ',' << ratio << '\n';
    };
    for (size_t mask{}; mask < totals.maskMs.size(); ++mask)
    {
        writeSummary("mask", static_cast<int32_t>(mask), totals.maskMs[mask]);
    }
    writeSummary("window", -1, totals.windowMs);
    writeSummary("all_idle", 0, totals.allIdleMs);
    writeSummary("epd_idle", -1, totals.epdIdleMs);
    writeSummary("any_activity", -1, totals.anyActivityMs);
    writeSummary("any_epd", -1, totals.anyEpdMs);
    writeSummary("epd_triple", -1, totals.epdTripleMs);
    writeSummary("four_way", kALL_MASK, totals.fourWayMs);
    writeSummary("encoder_duty", kENCODER_MASK, totals.activityMs[0U]);
    writeSummary("prefill_duty", kPREFILL_MASK, totals.activityMs[1U]);
    writeSummary("decode_duty", kDECODE_MASK, totals.activityMs[2U]);
    writeSummary("copy_duty", kCOPY_MASK, totals.activityMs[3U]);
}

void PhaseActivityTimelineRecorder::destroy(PendingInterval& pending) noexcept
{
    if (pending.start != nullptr)
    {
        static_cast<void>(cudaEventDestroy(pending.start));
        pending.start = nullptr;
    }
    if (pending.end != nullptr)
    {
        static_cast<void>(cudaEventDestroy(pending.end));
        pending.end = nullptr;
    }
}

} // namespace trt_edgellm::rt
