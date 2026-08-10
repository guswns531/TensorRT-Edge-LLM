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
#include <cuda_runtime.h>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

enum class PhaseKernelGroup
{
    kEncoderPreprocess,
    kEncoderEngine,
    kPrefillPrepare,
    kPrefillEngine,
    kPrefillCacheCommit,
    kPrefillSample,
    kDecodePrepare,
    kDecodeEngine,
    kDecodeCacheCommit,
    kDecodeSample,
};

char const* phaseKernelGroupName(PhaseKernelGroup group) noexcept;

struct PhaseKernelSegment
{
    PhaseKernelGroup group{PhaseKernelGroup::kEncoderPreprocess};
    std::string name;
    cudaStream_t stream{};
    std::function<void(cudaStream_t)> enqueue;
};

struct PhaseKernelDispatchMetadata
{
    size_t schedulerDispatchIndex{};
    int32_t schedulerKind{};
    int32_t prefillBatchSize{};
    int32_t decodeBatchSize{};
    int32_t prefillTokens{};
    int32_t decodeContextTokens{};
};

struct PhaseKernelGroupSample
{
    size_t dispatchIndex{};
    size_t segmentIndex{};
    PhaseKernelGroup group{PhaseKernelGroup::kEncoderPreprocess};
    std::string name;
    PhaseKernelDispatchMetadata dispatch;
    float gpuMs{};
};

//! Executes named kernel groups with CUDA-event boundaries and explicit handoff.
//!
//! Segments in one call are ordered by cudaStreamWaitEvent, even when adjacent
//! groups use different streams. Separate execute() calls remain independent
//! and may overlap, which permits E/P/D phase concurrency.
class PhaseKernelGroupRecorder
{
public:
    PhaseKernelGroupRecorder() = default;
    ~PhaseKernelGroupRecorder() noexcept;

    PhaseKernelGroupRecorder(PhaseKernelGroupRecorder const&) = delete;
    PhaseKernelGroupRecorder& operator=(PhaseKernelGroupRecorder const&) = delete;

    void execute(size_t dispatchIndex, std::vector<PhaseKernelSegment> const& segments,
        PhaseKernelDispatchMetadata dispatch = {});
    //! Collect every currently complete segment without blocking.
    size_t poll();
    //! Synchronize and collect every pending segment.
    void drain();
    void writeCsv(std::filesystem::path const& path) const;

    std::vector<PhaseKernelGroupSample> const& samples() const noexcept;
    size_t pendingCount() const noexcept;

private:
    struct PendingSample
    {
        PhaseKernelGroupSample sample;
        cudaEvent_t start{};
        cudaEvent_t done{};
    };

    bool collect(PendingSample& pending, bool synchronize);
    void destroy(PendingSample& pending) noexcept;

    std::vector<PendingSample> mPending;
    std::vector<PhaseKernelGroupSample> mSamples;
};

} // namespace rt
} // namespace trt_edgellm
