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

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trt_edgellm::rt
{

//! Scheduler-facing policy for sharing a bounded device-memory budget between stable KV and vision payloads.
struct PhaseMemoryBrokerConfig
{
    //! Preserve the legacy scheduler and fixed allocations unless explicitly enabled.
    bool enabled{};
    //! Minimum logical KV pages that encoder admission must leave available.
    int32_t kvReservePages{};
    //! At or below this free-page level, contract encoder admission. Zero disables contraction.
    int32_t kvPressurePages{};
    //! Encoder batch cap under logical KV pressure.
    size_t pressureMaxEncoderBatchSize{1U};
    //! Optional combined committed-KV and vision-payload budget. Zero disables the byte gate.
    size_t maxManagedBytes{};
    //! Bytes within maxManagedBytes that no phase may consume.
    size_t safetyReserveBytes{};
    //! Current physically committed KV pages. Fixed-pool deployments set this to the complete pool.
    int32_t committedKVPages{};
    //! Physical bytes consumed by one KV page across all attention layers.
    size_t bytesPerKVPage{};
    //! Permit the coordinator to reclaim idle vision slabs before applying byte backpressure.
    bool reclaimIdleVision{true};
};

struct PhaseMemoryBrokerSnapshot
{
    int32_t availableKVPages{};
    size_t downstreamVisionBytes{};
    size_t idleVisionBytes{};
};

enum class PhaseMemoryBrokerReason
{
    kDisabled,
    kAllowed,
    kKVPressure,
    kKVReserve,
    kManagedByteLimit,
};

//! Stable telemetry name for one broker decision reason.
char const* phaseMemoryBrokerReasonName(PhaseMemoryBrokerReason reason) noexcept;

struct PhaseMemoryBrokerDecision
{
    size_t encoderBatchSize{};
    size_t predictedManagedBytes{};
    bool reclaimIdleVision{};
    bool preferPrefill{};
    bool preferDecode{};
    PhaseMemoryBrokerReason reason{PhaseMemoryBrokerReason::kDisabled};
};

//! Decide how much of an already-valid encoder batch may enter the three-phase pipeline.
class PhaseMemoryBroker
{
public:
    explicit PhaseMemoryBroker(PhaseMemoryBrokerConfig config = {});

    //! Candidate bytes are ordered exactly like the encoder indices selected by the batching policy.
    PhaseMemoryBrokerDecision planEncoder(
        PhaseMemoryBrokerSnapshot const& snapshot, std::vector<size_t> const& candidateBytes) const noexcept;

    PhaseMemoryBrokerConfig const& config() const noexcept;

private:
    PhaseMemoryBrokerConfig mConfig;
};

} // namespace trt_edgellm::rt
