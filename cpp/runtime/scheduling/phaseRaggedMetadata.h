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

#include "common/executionPhase.h"
#include "runtime/exec/raggedBatchBuilder.h"
#include "runtime/exec/scheduledStep.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct LLMEngineConfig;
struct PipelineIO;
struct SharedResources;

//! One logical sequence of a homogeneous phase step, listed in phase-batch row order.
struct PhaseRaggedSequence
{
    uint64_t requestId{0};  //!< Non-zero; recorded in RaggedExecutionBatch::sequenceOrder for diagnostics.
    int32_t pastLength{0};  //!< Committed KV tokens before the step (StableKVPageManager::length(slot)).
    int32_t queryLength{0}; //!< Tokens executed this step; exactly 1 for SequenceWork::kDecode.
};

//! Entry-padded query width shared by token staging and metadata: max(queryLength), > 0.
int32_t phaseRaggedQueryWidth(std::vector<PhaseRaggedSequence> const& sequences);

//! Host builder for one homogeneous phase step in the entry-padded token layout.
//! Sequence i occupies physical rows [i*W, i*W + q_i); padding rows carry position -1.
//! state_indices[i] = i because the phase page-table and M-RoPE rows are indexed by active row.
//! Every sequence selects its last valid row, so logits row i corresponds to phase batch row i.
class PhaseRaggedMetadataBuilder
{
public:
    //! Reserves every host vector once; build() never allocates when within these capacities.
    PhaseRaggedMetadataBuilder(int32_t maxSequences, int32_t maxPhysicalTokens);

    //! @throws std::runtime_error on empty input, non-positive q, decode q != 1, past < 0,
    //!         N > maxSequences, or N*W > maxPhysicalTokens.
    RaggedExecutionBatch const& build(SequenceWork work, std::vector<PhaseRaggedSequence> const& sequences);

    //! kAutoregressiveDecode for decode; kContextChunk if any pastLength > 0, else kContextPrefill.
    ExecutionPhase executionPhase() const noexcept;

    RaggedExecutionBatch const& batch() const noexcept;

private:
    //! Same-row-count decode: offsets, lengths, state/logits indices, and sequenceWorks depend only on the
    //! row index, so only positions, past/attention lengths, and sequenceOrder are rewritten.
    RaggedExecutionBatch const& buildDecodeIncremental(std::vector<PhaseRaggedSequence> const& sequences);

    int32_t mMaxSequences{};
    int32_t mMaxPhysicalTokens{};
    RaggedExecutionBatch mBatch;
    ExecutionPhase mExecutionPhase{ExecutionPhase::kContextPrefill};
    bool mHasPriorBuild{false};
    SequenceWork mPriorWork{SequenceWork::kContext};
    int32_t mPriorNumSequences{0};
};

//! Uploads batch metadata through io's pinned staging (PipelineIO::uploadRaggedMetadata) and gathers
//! token-aligned RoPE into io.raggedRopeCosSin / io.raggedRopeCosSinSliding / io.raggedRopeCosSinFull
//! (prepareRaggedRope). Never touches kv_page_table, swa_kv_page_table, or HybridCacheManager state.
void uploadPhaseRaggedMetadata(PipelineIO& io, SharedResources& resources, LLMEngineConfig const& config,
    RaggedExecutionBatch const& batch, cudaStream_t stream);

} // namespace rt
} // namespace trt_edgellm
