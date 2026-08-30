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
#include <string_view>

namespace trt_edgellm::rt
{

//! Host-side request transition recorded around the asynchronous E/P/D executors.
enum class PhaseTimelineStage
{
    kVisionQueued,
    kEncoderStart,
    kEncoderDone,
    kPrefillReady,
    kPrefillRelease,
    kServerSubmit,
    kServerAdmit,
    kPrefillStart,
    kPrefillDone,
    kPrefillSamplingSubmit,
    kPrefillSamplingReady,
    kPrefillSamplingCollected,
    kPrefillTokenCommitted,
    kDecodeStart,
    kDecodeDone,
    kDecodeSamplingSubmit,
    kDecodeSamplingReady,
    kDecodeSamplingCollected,
    kDecodeTokenCommitted,
    kDecodeReady,
    kFirstToken,
    kCompletion,
    kSlotReleased,
};

//! One immutable request transition. Repeated decode dispatches are preserved;
//! GPU durations remain in PhaseDispatchMetrics.
struct PhaseTimelineEvent
{
    uint64_t requestId{};
    PhaseTimelineStage stage{PhaseTimelineStage::kVisionQueued};
    uint64_t timestampNs{};
    size_t dispatchIndex{};
    int32_t batchSize{};
    int32_t kvSlotId{-1};
};

inline uint64_t phaseTimelineNowNs() noexcept
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline std::string_view phaseTimelineStageName(PhaseTimelineStage stage) noexcept
{
    switch (stage)
    {
    case PhaseTimelineStage::kVisionQueued: return "vision_queued";
    case PhaseTimelineStage::kEncoderStart: return "encoder_start";
    case PhaseTimelineStage::kEncoderDone: return "encoder_done";
    case PhaseTimelineStage::kPrefillReady: return "prefill_ready";
    case PhaseTimelineStage::kPrefillRelease: return "prefill_release";
    case PhaseTimelineStage::kServerSubmit: return "server_submit";
    case PhaseTimelineStage::kServerAdmit: return "server_admit";
    case PhaseTimelineStage::kPrefillStart: return "prefill_start";
    case PhaseTimelineStage::kPrefillDone: return "prefill_done";
    case PhaseTimelineStage::kPrefillSamplingSubmit: return "prefill_sampling_submit";
    case PhaseTimelineStage::kPrefillSamplingReady: return "prefill_sampling_ready";
    case PhaseTimelineStage::kPrefillSamplingCollected: return "prefill_sampling_collected";
    case PhaseTimelineStage::kPrefillTokenCommitted: return "prefill_token_committed";
    case PhaseTimelineStage::kDecodeStart: return "decode_start";
    case PhaseTimelineStage::kDecodeDone: return "decode_done";
    case PhaseTimelineStage::kDecodeSamplingSubmit: return "decode_sampling_submit";
    case PhaseTimelineStage::kDecodeSamplingReady: return "decode_sampling_ready";
    case PhaseTimelineStage::kDecodeSamplingCollected: return "decode_sampling_collected";
    case PhaseTimelineStage::kDecodeTokenCommitted: return "decode_token_committed";
    case PhaseTimelineStage::kDecodeReady: return "decode_ready";
    case PhaseTimelineStage::kFirstToken: return "first_token";
    case PhaseTimelineStage::kCompletion: return "completion";
    case PhaseTimelineStage::kSlotReleased: return "slot_released";
    }
    return "unknown";
}

} // namespace trt_edgellm::rt
