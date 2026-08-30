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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_edgellm::rt
{

//! Stable activity-mask bits. Printed binary masks use C/D/P/E order, so E is 0001.
enum class PhaseActivityKind : uint8_t
{
    kEncoder = 0x01U,
    kPrefill = 0x02U,
    kDecode = 0x04U,
    kCopy = 0x08U,
};

char const* phaseActivityKindName(PhaseActivityKind kind) noexcept;
std::string phaseActivityMaskString(uint8_t mask);

struct PhaseActivityInterval
{
    uint64_t intervalId{};
    uint64_t correlationId{};
    PhaseActivityKind kind{PhaseActivityKind::kEncoder};
    std::string name;
    float startMs{};
    float endMs{};
};

struct PhaseActivitySegment
{
    float startMs{};
    float endMs{};
    uint8_t mask{};
};

struct PhaseActivitySummary
{
    static constexpr size_t kMASK_COUNT = 16U;
    static constexpr size_t kACTIVITY_COUNT = 4U;

    std::array<double, kMASK_COUNT> maskMs{};
    std::array<double, kACTIVITY_COUNT> activityMs{};
    double windowMs{};
    double allIdleMs{};
    double epdIdleMs{};
    double anyActivityMs{};
    double anyEpdMs{};
    double epdTripleMs{};
    double fourWayMs{};
};

//! Build a non-overlapping E/P/D/C activity-mask timeline.
//!
//! The default window begins at the recorder epoch (0 ms) and ends at the last
//! interval. An explicit later window end includes trailing idle time.
std::vector<PhaseActivitySegment> phaseActivitySegments(
    std::vector<PhaseActivityInterval> const& intervals, float windowEndMs = 0.0F);
//! Crop to the first/last recorded GPU work and rebase the first activity to 0 ms.
//! Internal mask-0000 gaps remain visible; process startup and post-drain time do not.
std::vector<PhaseActivitySegment> phaseActivityActiveSpanSegments(std::vector<PhaseActivityInterval> const& intervals);
PhaseActivitySummary phaseActivitySummary(std::vector<PhaseActivitySegment> const& segments, float windowEndMs = 0.0F);

//! Opt-in CUDA-event recorder for phase-stream activity rather than hardware utilization.
//!
//! Every interval is resolved against one synchronized event epoch in the same
//! CUDA context. Recording adds two timing events per interval and must remain
//! disabled on normal serving runs that do not request activity telemetry.
class PhaseActivityTimelineRecorder
{
public:
    using Token = uint64_t;

    explicit PhaseActivityTimelineRecorder(cudaStream_t epochStream);
    ~PhaseActivityTimelineRecorder() noexcept;

    PhaseActivityTimelineRecorder(PhaseActivityTimelineRecorder const&) = delete;
    PhaseActivityTimelineRecorder& operator=(PhaseActivityTimelineRecorder const&) = delete;

    Token begin(PhaseActivityKind kind, cudaStream_t stream, std::string name = {}, uint64_t correlationId = 0U);
    void end(Token token, cudaStream_t stream);
    void cancel(Token token) noexcept;

    //! Collect every currently completed interval without blocking.
    size_t poll();
    //! Require all intervals to be closed, then synchronize and collect them.
    void drain();
    //! Clear collected data and establish a new synchronized device epoch.
    void reset(cudaStream_t epochStream);

    std::vector<PhaseActivityInterval> intervals() const;
    std::vector<PhaseActivitySegment> segments() const;
    PhaseActivitySummary summary() const;
    size_t pendingCount() const;

    //! Write <prefix>-intervals.csv, <prefix>-segments.csv, and <prefix>-summary.csv.
    void writeCsv(std::filesystem::path const& prefix) const;

private:
    struct PendingInterval
    {
        Token token{};
        uint64_t correlationId{};
        PhaseActivityKind kind{PhaseActivityKind::kEncoder};
        std::string name;
        cudaStream_t stream{};
        cudaEvent_t start{};
        cudaEvent_t end{};
        bool closed{};
    };

    void initializeEpoch(cudaStream_t epochStream);
    bool collect(PendingInterval& pending, bool synchronize);
    void destroy(PendingInterval& pending) noexcept;

    mutable std::mutex mMutex;
    CUcontext mCudaContext{};
    cudaEvent_t mEpoch{};
    std::unordered_map<Token, PendingInterval> mPending;
    std::vector<PhaseActivityInterval> mIntervals;
    Token mNextToken{1U};
};

} // namespace trt_edgellm::rt
