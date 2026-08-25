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

#include "runtime/scheduling/phaseMemoryBroker.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <limits>

namespace trt_edgellm::rt
{
namespace
{

size_t saturatedAdd(size_t left, size_t right) noexcept
{
    return right > std::numeric_limits<size_t>::max() - left ? std::numeric_limits<size_t>::max() : left + right;
}

size_t saturatedMultiply(size_t left, size_t right) noexcept
{
    return left > 0U && right > std::numeric_limits<size_t>::max() / left ? std::numeric_limits<size_t>::max()
                                                                          : left * right;
}

} // namespace

PhaseMemoryBroker::PhaseMemoryBroker(PhaseMemoryBrokerConfig config)
    : mConfig(config)
{
    ELLM_CHECK(mConfig.kvReservePages >= 0, "Phase memory KV reserve must be non-negative");
    ELLM_CHECK(mConfig.kvPressurePages == 0 || mConfig.kvPressurePages >= mConfig.kvReservePages,
        "Phase memory KV pressure watermark cannot be below its reserve");
    ELLM_CHECK(mConfig.pressureMaxEncoderBatchSize > 0, "Phase memory pressure encoder batch cap must be positive");
    ELLM_CHECK(mConfig.committedKVPages >= 0, "Phase memory committed KV pages must be non-negative");
    ELLM_CHECK(mConfig.maxManagedBytes == 0 || mConfig.safetyReserveBytes < mConfig.maxManagedBytes,
        "Phase memory safety reserve must be smaller than the managed byte budget");
    ELLM_CHECK(mConfig.maxManagedBytes == 0 || mConfig.committedKVPages == 0 || mConfig.bytesPerKVPage > 0,
        "Phase memory byte accounting requires bytesPerKVPage for a committed KV pool");
}

PhaseMemoryBrokerDecision PhaseMemoryBroker::planEncoder(
    PhaseMemoryBrokerSnapshot const& snapshot, std::vector<size_t> const& candidateBytes) const noexcept
{
    PhaseMemoryBrokerDecision decision;
    decision.encoderBatchSize = candidateBytes.size();
    if (!mConfig.enabled)
    {
        decision.reason = PhaseMemoryBrokerReason::kDisabled;
        return decision;
    }

    decision.reason = PhaseMemoryBrokerReason::kAllowed;
    size_t batchLimit = candidateBytes.size();
    if (snapshot.availableKVPages <= mConfig.kvReservePages)
    {
        batchLimit = 0U;
        decision.preferDecode = true;
        decision.reason = PhaseMemoryBrokerReason::kKVReserve;
    }
    else
    {
        batchLimit = std::min(batchLimit, static_cast<size_t>(snapshot.availableKVPages - mConfig.kvReservePages));
        if (mConfig.kvPressurePages > 0 && snapshot.availableKVPages <= mConfig.kvPressurePages)
        {
            batchLimit = std::min(batchLimit, mConfig.pressureMaxEncoderBatchSize);
            decision.preferDecode = true;
            decision.reason = PhaseMemoryBrokerReason::kKVPressure;
        }
    }

    size_t const committedKVBytes
        = saturatedMultiply(static_cast<size_t>(mConfig.committedKVPages), mConfig.bytesPerKVPage);
    size_t const activeBytes = saturatedAdd(committedKVBytes, snapshot.downstreamVisionBytes);
    size_t candidateTotal{};
    if (mConfig.maxManagedBytes > 0U)
    {
        size_t const usableBytes = mConfig.maxManagedBytes - mConfig.safetyReserveBytes;
        size_t const activeAndIdleBytes = saturatedAdd(activeBytes, snapshot.idleVisionBytes);
        size_t fitWithIdle{};
        size_t fitWithoutIdle{};
        size_t bytesWithIdle = activeAndIdleBytes;
        size_t bytesWithoutIdle = activeBytes;
        for (size_t index{}; index < batchLimit; ++index)
        {
            bytesWithIdle = saturatedAdd(bytesWithIdle, candidateBytes[index]);
            bytesWithoutIdle = saturatedAdd(bytesWithoutIdle, candidateBytes[index]);
            if (bytesWithIdle <= usableBytes)
            {
                fitWithIdle = index + 1U;
            }
            if (bytesWithoutIdle <= usableBytes)
            {
                fitWithoutIdle = index + 1U;
            }
        }
        size_t const byteLimit = mConfig.reclaimIdleVision ? fitWithoutIdle : fitWithIdle;
        if (mConfig.reclaimIdleVision && fitWithoutIdle > fitWithIdle)
        {
            decision.reclaimIdleVision = true;
        }
        if (byteLimit < batchLimit)
        {
            batchLimit = byteLimit;
            decision.preferPrefill = true;
            decision.reason = PhaseMemoryBrokerReason::kManagedByteLimit;
        }
    }

    for (size_t index{}; index < batchLimit; ++index)
    {
        candidateTotal = saturatedAdd(candidateTotal, candidateBytes[index]);
    }
    decision.encoderBatchSize = batchLimit;
    decision.predictedManagedBytes = saturatedAdd(activeBytes, candidateTotal);
    return decision;
}

PhaseMemoryBrokerConfig const& PhaseMemoryBroker::config() const noexcept
{
    return mConfig;
}

char const* phaseMemoryBrokerReasonName(PhaseMemoryBrokerReason reason) noexcept
{
    char const* result = "unknown";
    switch (reason)
    {
    case PhaseMemoryBrokerReason::kDisabled: result = "disabled"; break;
    case PhaseMemoryBrokerReason::kAllowed: result = "allowed"; break;
    case PhaseMemoryBrokerReason::kKVPressure: result = "kv_pressure"; break;
    case PhaseMemoryBrokerReason::kKVReserve: result = "kv_reserve"; break;
    case PhaseMemoryBrokerReason::kManagedByteLimit: result = "managed_byte_limit"; break;
    }
    return result;
}

} // namespace trt_edgellm::rt
