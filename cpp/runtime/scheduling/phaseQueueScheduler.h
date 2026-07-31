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
#include <deque>
#include <functional>
#include <unordered_set>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! A unit of phase work. tokenCount is prompt length for prefill and current
//! KV length for decode; it is a scheduling cost hint, not tensor ownership.
struct PhaseWorkItem
{
    uint64_t requestId{};
    int32_t tokenCount{};
};

enum class PhaseDispatchKind
{
    kNone,
    kPrefill,
    kDecode,
    kOverlap,
};

//! Read-only queue summary passed to a custom scheduling policy.
struct PhaseQueueSnapshot
{
    size_t prefillQueued{};
    size_t decodeQueued{};
    int32_t prefillCandidateTokens{};
    int32_t decodeCandidateTokens{};
    int32_t consecutiveDecodeBatches{};
};

using PhaseSchedulingPolicy = std::function<PhaseDispatchKind(PhaseQueueSnapshot const&)>;

struct PhaseQueueSchedulerConfig
{
    int32_t maxPrefillBatchSize{1};
    int32_t maxDecodeBatchSize{4};
    //! Default policy only overlaps short prefills. The initial value comes from
    //! the Gemma4 E2B RTX 3080 crossover benchmark and remains configurable.
    int32_t maxOverlapPrefillTokens{128};
    //! Admit one prefill batch after this many decode-only decisions so a
    //! continuous decode queue cannot starve new requests forever.
    int32_t decodeBurstLimit{8};
    PhaseSchedulingPolicy policy{};
};

struct PhaseDispatchPlan
{
    PhaseDispatchKind kind{PhaseDispatchKind::kNone};
    std::vector<PhaseWorkItem> prefillBatch;
    std::vector<PhaseWorkItem> decodeBatch;
};

//! Host-side two-queue batch scheduler for same-context, dual-stream inference.
//!
//! This class intentionally owns no CUDA or TensorRT objects. The execution
//! layer consumes a DispatchPlan, binds each batch's stable KV slots to its
//! phase-local TensorMap, and records CUDA events around the two enqueues.
class PhaseQueueScheduler
{
public:
    explicit PhaseQueueScheduler(PhaseQueueSchedulerConfig config = {});

    void enqueuePrefill(PhaseWorkItem item);
    void enqueueDecode(PhaseWorkItem item);

    PhaseDispatchPlan next();

    size_t prefillQueueSize() const noexcept;
    size_t decodeQueueSize() const noexcept;
    bool empty() const noexcept;

private:
    PhaseDispatchKind defaultDecision(PhaseQueueSnapshot const& snapshot) const noexcept;
    PhaseQueueSnapshot snapshot() const;
    std::vector<PhaseWorkItem> popBatch(std::deque<PhaseWorkItem>& queue, int32_t maxBatchSize);

    PhaseQueueSchedulerConfig mConfig;
    std::deque<PhaseWorkItem> mPrefillQueue;
    std::deque<PhaseWorkItem> mDecodeQueue;
    std::unordered_set<uint64_t> mQueuedRequestIds;
    int32_t mConsecutiveDecodeBatches{};
};

} // namespace rt
} // namespace trt_edgellm
