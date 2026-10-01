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

#include "runtime/scheduling/phaseRaggedMetadata.h"

#include "common/checkMacros.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/state/pipelineIO.h"

#include <algorithm>

namespace trt_edgellm::rt
{

int32_t phaseRaggedQueryWidth(std::vector<PhaseRaggedSequence> const& sequences)
{
    ELLM_CHECK(!sequences.empty(), "Phase ragged metadata requires at least one sequence");
    int32_t width = 0;
    for (PhaseRaggedSequence const& sequence : sequences)
    {
        ELLM_CHECK(sequence.queryLength > 0, "Phase ragged sequence query length must be positive");
        width = std::max(width, sequence.queryLength);
    }
    return width;
}

PhaseRaggedMetadataBuilder::PhaseRaggedMetadataBuilder(int32_t maxSequences, int32_t maxPhysicalTokens)
    : mMaxSequences(maxSequences)
    , mMaxPhysicalTokens(maxPhysicalTokens)
{
    ELLM_CHECK(
        mMaxSequences > 0 && mMaxPhysicalTokens > 0, "Phase ragged metadata builder capacities must be positive");
    mBatch.sequenceOrder.reserve(static_cast<size_t>(mMaxSequences));
    mBatch.positions.reserve(static_cast<size_t>(mMaxPhysicalTokens));
    mBatch.queryStartOffsets.reserve(static_cast<size_t>(mMaxSequences) + 1U);
    mBatch.queryLengths.reserve(static_cast<size_t>(mMaxSequences));
    mBatch.pastLengths.reserve(static_cast<size_t>(mMaxSequences));
    mBatch.attentionSequenceLengths.reserve(static_cast<size_t>(mMaxSequences));
    mBatch.stateIndices.reserve(static_cast<size_t>(mMaxSequences));
    mBatch.logitsIndices.reserve(static_cast<size_t>(mMaxSequences));
    mBatch.logitsToSequence.reserve(static_cast<size_t>(mMaxSequences));
    mBatch.sequenceWorks.reserve(static_cast<size_t>(mMaxSequences));
}

RaggedExecutionBatch const& PhaseRaggedMetadataBuilder::build(
    SequenceWork work, std::vector<PhaseRaggedSequence> const& sequences)
{
    ELLM_CHECK(!sequences.empty(), "Phase ragged metadata build requires a non-empty step");
    int32_t const numSequences = static_cast<int32_t>(sequences.size());
    ELLM_CHECK(numSequences <= mMaxSequences, "Phase ragged metadata step exceeds sequence capacity");

    int32_t const queryWidth = phaseRaggedQueryWidth(sequences);
    int64_t const physicalTokens64 = static_cast<int64_t>(numSequences) * queryWidth;
    ELLM_CHECK(physicalTokens64 <= mMaxPhysicalTokens, "Phase ragged metadata step exceeds physical token capacity");
    int32_t const physicalTokens = static_cast<int32_t>(physicalTokens64);

    bool anyPastLength = false;
    for (PhaseRaggedSequence const& sequence : sequences)
    {
        ELLM_CHECK(sequence.pastLength >= 0, "Phase ragged sequence past length must not be negative");
        ELLM_CHECK(work != SequenceWork::kDecode || sequence.queryLength == 1,
            "Vanilla decode requires a query length of one");
        anyPastLength = anyPastLength || sequence.pastLength > 0;
    }
    mExecutionPhase = work == SequenceWork::kDecode
        ? ExecutionPhase::kAutoregressiveDecode
        : (anyPastLength ? ExecutionPhase::kContextChunk : ExecutionPhase::kContextPrefill);

    mBatch.layout = TokenLayoutBackend::kEntryPaddedCompatibility;
    mBatch.shape = RaggedStepShape{numSequences, 0, physicalTokens, queryWidth,
        work == SequenceWork::kContext ? numSequences : 0, 0, numSequences};
    mBatch.sequenceOrder.clear();
    mBatch.positions.assign(static_cast<size_t>(physicalTokens), -1);
    mBatch.queryStartOffsets.resize(static_cast<size_t>(numSequences) + 1U);
    mBatch.queryLengths.clear();
    mBatch.pastLengths.clear();
    mBatch.attentionSequenceLengths.clear();
    mBatch.stateIndices.clear();
    mBatch.logitsIndices.clear();
    mBatch.logitsToSequence.clear();
    mBatch.sequenceWorks.clear();

    int32_t validTokens = 0;
    int32_t contextTokens = 0;
    for (int32_t row = 0; row < numSequences; ++row)
    {
        PhaseRaggedSequence const& sequence = sequences[static_cast<size_t>(row)];
        int32_t const physicalStart = row * queryWidth;
        mBatch.queryStartOffsets[static_cast<size_t>(row)] = physicalStart;
        mBatch.queryLengths.push_back(sequence.queryLength);
        mBatch.pastLengths.push_back(sequence.pastLength);
        mBatch.attentionSequenceLengths.push_back(sequence.pastLength + sequence.queryLength);
        mBatch.stateIndices.push_back(row);
        mBatch.sequenceOrder.push_back(SequenceIdentity{sequence.requestId, ResidentRef{row, 1}});
        mBatch.sequenceWorks.push_back(work);
        for (int32_t token = 0; token < sequence.queryLength; ++token)
        {
            mBatch.positions[static_cast<size_t>(physicalStart + token)] = sequence.pastLength + token;
        }
        mBatch.logitsIndices.push_back(physicalStart + sequence.queryLength - 1);
        mBatch.logitsToSequence.push_back(row);
        validTokens += sequence.queryLength;
        contextTokens += work == SequenceWork::kContext ? sequence.queryLength : 0;
    }
    mBatch.queryStartOffsets[static_cast<size_t>(numSequences)] = physicalTokens;
    mBatch.shape.validTokens = validTokens;
    mBatch.shape.numContextTokens = contextTokens;
    return mBatch;
}

ExecutionPhase PhaseRaggedMetadataBuilder::executionPhase() const noexcept
{
    return mExecutionPhase;
}

RaggedExecutionBatch const& PhaseRaggedMetadataBuilder::batch() const noexcept
{
    return mBatch;
}

void uploadPhaseRaggedMetadata(PipelineIO& io, SharedResources& resources, LLMEngineConfig const& config,
    RaggedExecutionBatch const& batch, cudaStream_t stream)
{
    io.uploadRaggedMetadata(batch, stream);
    prepareRaggedRope(io, resources, config, batch.shape.physicalTokens, batch.shape.numSequences, stream);
}

} // namespace trt_edgellm::rt
