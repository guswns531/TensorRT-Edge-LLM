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

namespace trt_edgellm::rt
{
//! Read-only queue summary passed to a custom scheduling policy.
struct PhaseQueueSnapshot
{
    size_t prefillQueued{};
    //! Requests already known to the serving facade whose host-side producer
    //! may make another prefill row ready after the current GPU action.
    size_t prefillPendingProducerRows{};
    size_t prefillPendingTextProducerRows{};
    size_t prefillPendingExternalProducerRows{};
    double prefillProducerReadyWaitUs{};
    double prefillProducerReadyUncertaintyUs{};
    uint64_t prefillProducerReadyEventId{};
    bool prefillProducerRowsClassified{};
    size_t decodeQueued{};
    int32_t prefillCandidateTokens{};
    int64_t prefillRemainingTokens{};
    int32_t prefillContinuationRows{};
    int32_t decodeCandidateTokens{};
    //! Total and maximum context lengths of the first runnable decode rows.
    int64_t decodeCandidateContextTokens{};
    int32_t decodeCandidateMaxContextLength{};
    int32_t consecutiveDecodeBatches{};
    double prefillOldestWaitUs{};
    double prefillOldestRequestAgeUs{};
    double prefillMinTtftSlackUs{};
    //! Request and remaining prompt path owning the minimum first-token slack.
    uint64_t prefillMinimumSlackRequestId{};
    int32_t prefillCriticalPathRemainingTokens{};
    double decodeOldestWaitUs{};
    //! Minimum remaining next-token slack. Request SLOs override the global
    //! fallback used only by WAIT/refill action selection.
    double decodeMinTpotSlackUs{};
    double prefillMaxSloPressure{};
    double decodeMaxSloPressure{};
    int32_t prefillHighestPriority{};
    int32_t decodeHighestPriority{};
    //! Current resource state supplied by the serving facade. Zero denotes a
    //! linear cache or a scheduler without a resource supplier.
    int32_t pagePoolTotalBundles{};
    int32_t pagePoolAllocatedBundles{};
    int32_t pagePoolAvailableBundles{};
    int32_t pageReservationGuaranteedBundles{};
    int32_t pageReservationAvailableBundles{};
};


} // namespace trt_edgellm::rt
