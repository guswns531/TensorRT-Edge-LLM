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

#include "runtime/scheduling/phaseKernelGroupRecorder.h"

#include "common/checkMacros.h"
#include "common/cudaMacros.h"

#include <fstream>
#include <iomanip>
#include <utility>

namespace trt_edgellm
{
namespace rt
{

char const* phaseKernelGroupName(PhaseKernelGroup group) noexcept
{
    char const* result{"unknown"};
    switch (group)
    {
    case PhaseKernelGroup::kEncoderPreprocess: result = "encoder_preprocess"; break;
    case PhaseKernelGroup::kEncoderEngine: result = "encoder_engine"; break;
    case PhaseKernelGroup::kPrefillPrepare: result = "prefill_prepare"; break;
    case PhaseKernelGroup::kPrefillEngine: result = "prefill_engine"; break;
    case PhaseKernelGroup::kPrefillCacheCommit: result = "prefill_cache_commit"; break;
    case PhaseKernelGroup::kPrefillSample: result = "prefill_sample"; break;
    case PhaseKernelGroup::kDecodePrepare: result = "decode_prepare"; break;
    case PhaseKernelGroup::kDecodeEngine: result = "decode_engine"; break;
    case PhaseKernelGroup::kDecodeCacheCommit: result = "decode_cache_commit"; break;
    case PhaseKernelGroup::kDecodeSample: result = "decode_sample"; break;
    }
    return result;
}

PhaseKernelGroupRecorder::~PhaseKernelGroupRecorder() noexcept
{
    for (PendingSample& pending : mPending)
    {
        static_cast<void>(cudaEventSynchronize(pending.done));
        destroy(pending);
    }
}

void PhaseKernelGroupRecorder::execute(
    size_t dispatchIndex, std::vector<PhaseKernelSegment> const& segments, PhaseKernelDispatchMetadata dispatch)
{
    check::check(!segments.empty(), "Kernel-group execution requires at least one segment.");
    cudaEvent_t previous{};
    size_t segmentIndex{};
    for (PhaseKernelSegment const& segment : segments)
    {
        check::check(segment.stream != nullptr, "Kernel-group segment requires an explicit CUDA stream.");
        check::check(static_cast<bool>(segment.enqueue), "Kernel-group segment requires an enqueue callback.");
        if (previous != nullptr)
        {
            CUDA_CHECK(cudaStreamWaitEvent(segment.stream, previous));
        }
        PendingSample pending;
        pending.sample.dispatchIndex = dispatchIndex;
        pending.sample.segmentIndex = segmentIndex++;
        pending.sample.group = segment.group;
        pending.sample.name = segment.name.empty() ? phaseKernelGroupName(segment.group) : segment.name;
        pending.sample.dispatch = dispatch;
        CUDA_CHECK(cudaEventCreate(&pending.start));
        CUDA_CHECK(cudaEventCreate(&pending.done));
        CUDA_CHECK(cudaEventRecord(pending.start, segment.stream));
        segment.enqueue(segment.stream);
        CUDA_CHECK(cudaEventRecord(pending.done, segment.stream));
        previous = pending.done;
        mPending.push_back(std::move(pending));
    }
}

bool PhaseKernelGroupRecorder::collect(PendingSample& pending, bool synchronize)
{
    if (synchronize)
    {
        CUDA_CHECK(cudaEventSynchronize(pending.done));
    }
    else
    {
        cudaError_t const status = cudaEventQuery(pending.done);
        if (status == cudaErrorNotReady)
        {
            return false;
        }
        CUDA_CHECK(status);
    }
    CUDA_CHECK(cudaEventElapsedTime(&pending.sample.gpuMs, pending.start, pending.done));
    mSamples.push_back(std::move(pending.sample));
    destroy(pending);
    return true;
}

size_t PhaseKernelGroupRecorder::poll()
{
    size_t collected{};
    for (auto pending = mPending.begin(); pending != mPending.end();)
    {
        if (collect(*pending, false))
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

void PhaseKernelGroupRecorder::drain()
{
    for (PendingSample& pending : mPending)
    {
        static_cast<void>(collect(pending, true));
    }
    mPending.clear();
}

void PhaseKernelGroupRecorder::writeCsv(std::filesystem::path const& path) const
{
    check::check(mPending.empty(), "Kernel-group samples must be drained before CSV export.");
    std::ofstream output(path);
    check::check(output.good(), "Failed to open kernel-group CSV output.");
    output << "dispatch_index,segment_index,scheduler_dispatch_index,scheduler_kind,prefill_batch,decode_batch,"
              "prefill_tokens,prefill_initial_rows,prefill_continuation_rows,prefill_final_rows,"
              "prefill_past_kv_mean,prefill_past_kv_max,prefill_past_kv_spread,decode_context_tokens,"
              "group,name,gpu_ms\n";
    output << std::fixed << std::setprecision(6);
    for (PhaseKernelGroupSample const& sample : mSamples)
    {
        output << sample.dispatchIndex << ',' << sample.segmentIndex << ',' << sample.dispatch.schedulerDispatchIndex
               << ',' << sample.dispatch.schedulerKind << ',' << sample.dispatch.prefillBatchSize << ','
               << sample.dispatch.decodeBatchSize << ',' << sample.dispatch.prefillTokens << ','
               << sample.dispatch.prefillInitialRows << ',' << sample.dispatch.prefillContinuationRows << ','
               << sample.dispatch.prefillFinalRows << ',' << sample.dispatch.prefillPastKVMean << ','
               << sample.dispatch.prefillPastKVMax << ',' << sample.dispatch.prefillPastKVSpread << ','
               << sample.dispatch.decodeContextTokens << ',' << phaseKernelGroupName(sample.group) << ',' << sample.name
               << ',' << sample.gpuMs << '\n';
    }
}

std::vector<PhaseKernelGroupSample> const& PhaseKernelGroupRecorder::samples() const noexcept
{
    return mSamples;
}

size_t PhaseKernelGroupRecorder::pendingCount() const noexcept
{
    return mPending.size();
}

void PhaseKernelGroupRecorder::destroy(PendingSample& pending) noexcept
{
    static_cast<void>(cudaEventDestroy(pending.start));
    static_cast<void>(cudaEventDestroy(pending.done));
    pending.start = nullptr;
    pending.done = nullptr;
}

} // namespace rt
} // namespace trt_edgellm
