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

#include "runtime/scheduling/phaseQueueScheduler.h"

#include "common/checkMacros.h"

#include <algorithm>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

PhaseQueueScheduler::PhaseQueueScheduler(PhaseQueueSchedulerConfig config)
    : mConfig(std::move(config))
{
    check::check(mConfig.maxPrefillBatchSize > 0, "maxPrefillBatchSize must be positive");
    check::check(mConfig.maxDecodeBatchSize > 0, "maxDecodeBatchSize must be positive");
    check::check(mConfig.maxOverlapPrefillTokens >= 0, "maxOverlapPrefillTokens must be non-negative");
    check::check(mConfig.decodeBurstLimit > 0, "decodeBurstLimit must be positive");
}

void PhaseQueueScheduler::enqueuePrefill(PhaseWorkItem item)
{
    check::check(item.tokenCount > 0, "Prefill tokenCount must be positive");
    check::check(mQueuedRequestIds.insert(item.requestId).second, "Request is already queued");
    mPrefillQueue.push_back(item);
}

void PhaseQueueScheduler::enqueueDecode(PhaseWorkItem item)
{
    check::check(item.tokenCount >= 0, "Decode tokenCount must be non-negative");
    check::check(mQueuedRequestIds.insert(item.requestId).second, "Request is already queued");
    mDecodeQueue.push_back(item);
}

PhaseQueueSnapshot PhaseQueueScheduler::snapshot() const
{
    PhaseQueueSnapshot result{};
    result.prefillQueued = mPrefillQueue.size();
    result.decodeQueued = mDecodeQueue.size();
    result.consecutiveDecodeBatches = mConsecutiveDecodeBatches;
    int32_t const prefillCount = std::min<int32_t>(mConfig.maxPrefillBatchSize, mPrefillQueue.size());
    int32_t const decodeCount = std::min<int32_t>(mConfig.maxDecodeBatchSize, mDecodeQueue.size());
    for (int32_t i = 0; i < prefillCount; ++i)
    {
        result.prefillCandidateTokens += mPrefillQueue[i].tokenCount;
    }
    for (int32_t i = 0; i < decodeCount; ++i)
    {
        result.decodeCandidateTokens += mDecodeQueue[i].tokenCount;
    }
    return result;
}

PhaseDispatchKind PhaseQueueScheduler::defaultDecision(PhaseQueueSnapshot const& state) const noexcept
{
    if (state.prefillQueued == 0 && state.decodeQueued == 0)
    {
        return PhaseDispatchKind::kNone;
    }
    if (state.prefillQueued == 0)
    {
        return PhaseDispatchKind::kDecode;
    }
    if (state.decodeQueued == 0)
    {
        return PhaseDispatchKind::kPrefill;
    }
    if (state.consecutiveDecodeBatches >= mConfig.decodeBurstLimit)
    {
        return PhaseDispatchKind::kPrefill;
    }
    if (state.prefillCandidateTokens <= mConfig.maxOverlapPrefillTokens)
    {
        return PhaseDispatchKind::kOverlap;
    }
    return PhaseDispatchKind::kDecode;
}

std::vector<PhaseWorkItem> PhaseQueueScheduler::popBatch(std::deque<PhaseWorkItem>& queue, int32_t maxBatchSize)
{
    int32_t const count = std::min<int32_t>(maxBatchSize, queue.size());
    std::vector<PhaseWorkItem> batch;
    batch.reserve(count);
    for (int32_t i = 0; i < count; ++i)
    {
        batch.push_back(queue.front());
        mQueuedRequestIds.erase(queue.front().requestId);
        queue.pop_front();
    }
    return batch;
}

PhaseDispatchPlan PhaseQueueScheduler::next()
{
    PhaseQueueSnapshot const state = snapshot();
    PhaseDispatchKind const kind = mConfig.policy ? mConfig.policy(state) : defaultDecision(state);
    check::check(kind != PhaseDispatchKind::kPrefill || state.prefillQueued > 0,
        "Scheduling policy selected an empty prefill queue");
    check::check(kind != PhaseDispatchKind::kDecode || state.decodeQueued > 0,
        "Scheduling policy selected an empty decode queue");
    check::check(kind != PhaseDispatchKind::kOverlap || (state.prefillQueued > 0 && state.decodeQueued > 0),
        "Scheduling policy selected overlap without work in both queues");

    PhaseDispatchPlan plan;
    plan.kind = kind;
    if (kind == PhaseDispatchKind::kPrefill || kind == PhaseDispatchKind::kOverlap)
    {
        plan.prefillBatch = popBatch(mPrefillQueue, mConfig.maxPrefillBatchSize);
    }
    if (kind == PhaseDispatchKind::kDecode || kind == PhaseDispatchKind::kOverlap)
    {
        plan.decodeBatch = popBatch(mDecodeQueue, mConfig.maxDecodeBatchSize);
    }
    if (kind == PhaseDispatchKind::kDecode)
    {
        ++mConsecutiveDecodeBatches;
    }
    else if (kind == PhaseDispatchKind::kPrefill || kind == PhaseDispatchKind::kOverlap)
    {
        mConsecutiveDecodeBatches = 0;
    }
    return plan;
}

size_t PhaseQueueScheduler::prefillQueueSize() const noexcept
{
    return mPrefillQueue.size();
}

size_t PhaseQueueScheduler::decodeQueueSize() const noexcept
{
    return mDecodeQueue.size();
}

bool PhaseQueueScheduler::empty() const noexcept
{
    return mPrefillQueue.empty() && mDecodeQueue.empty();
}

} // namespace rt
} // namespace trt_edgellm
