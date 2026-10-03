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
#include "runtime/state/pipelineIO.h"

#include <algorithm>
#include <limits>

namespace trt_edgellm
{
namespace rt
{

int32_t phaseRaggedQueryWidth(std::vector<PhaseRaggedSequence> const& sequences)
{
    ELLM_CHECK(!sequences.empty(), "phase ragged metadata requires at least one sequence");
    int32_t width = 0;
    for (PhaseRaggedSequence const& sequence : sequences)
    {
        ELLM_CHECK(sequence.queryLength > 0, "phase ragged sequence query length must be positive");
        width = std::max(width, sequence.queryLength);
    }
    return width;
}

PhaseRaggedMetadataBuilder::PhaseRaggedMetadataBuilder(int32_t maxSequences, int32_t maxPhysicalTokens)
    : mMaxSequences(maxSequences)
    , mMaxPhysicalTokens(maxPhysicalTokens)
{
    ELLM_CHECK(maxSequences > 0, "phase ragged metadata builder requires a positive sequence capacity");
    ELLM_CHECK(maxPhysicalTokens > 0, "phase ragged metadata builder requires a positive token capacity");
    mBatch.sequenceOrder.reserve(static_cast<size_t>(maxSequences));
    mBatch.positions.reserve(static_cast<size_t>(maxPhysicalTokens));
    mBatch.queryStartOffsets.reserve(static_cast<size_t>(maxSequences) + 1);
    mBatch.queryLengths.reserve(static_cast<size_t>(maxSequences));
    mBatch.pastLengths.reserve(static_cast<size_t>(maxSequences));
    mBatch.attentionSequenceLengths.reserve(static_cast<size_t>(maxSequences));
    mBatch.stateIndices.reserve(static_cast<size_t>(maxSequences));
    mBatch.logitsIndices.reserve(static_cast<size_t>(maxSequences));
    mBatch.logitsToSequence.reserve(static_cast<size_t>(maxSequences));
    mBatch.sequenceWorks.reserve(static_cast<size_t>(maxSequences));
}

RaggedExecutionBatch const& PhaseRaggedMetadataBuilder::buildDecodeIncremental(
    std::vector<PhaseRaggedSequence> const& sequences)
{
    int32_t const numSequences = static_cast<int32_t>(sequences.size());
    for (int32_t row = 0; row < numSequences; ++row)
    {
        PhaseRaggedSequence const& sequence = sequences[static_cast<size_t>(row)];
        ELLM_CHECK(sequence.queryLength == 1, "vanilla decode requires query length one");
        ELLM_CHECK(sequence.pastLength >= 0, "phase ragged sequence past length must not be negative");
        mBatch.positions[static_cast<size_t>(row)] = sequence.pastLength;
        mBatch.pastLengths[static_cast<size_t>(row)] = sequence.pastLength;
        mBatch.attentionSequenceLengths[static_cast<size_t>(row)] = sequence.pastLength + 1;
        mBatch.sequenceOrder[static_cast<size_t>(row)] = SequenceIdentity{sequence.requestId, ResidentRef{row, 1}};
    }
    mExecutionPhase = ExecutionPhase::kAutoregressiveDecode;
    return mBatch;
}

RaggedExecutionBatch const& PhaseRaggedMetadataBuilder::build(
    SequenceWork work, std::vector<PhaseRaggedSequence> const& sequences, TokenLayoutBackend layout)
{
    ELLM_CHECK(!sequences.empty(), "phase ragged metadata builder requires a non-empty batch");
    int32_t const numSequences = static_cast<int32_t>(sequences.size());
    ELLM_CHECK(numSequences <= mMaxSequences, "phase ragged batch exceeds its configured sequence capacity");

    if (work == SequenceWork::kDecode && mHasPriorBuild && mPriorWork == SequenceWork::kDecode
        && mPriorNumSequences == numSequences && mBatch.layout == layout)
    {
        return buildDecodeIncremental(sequences);
    }

    bool const packed = layout == TokenLayoutBackend::kNativeCompactRagged;

    int32_t queryWidth = 0;
    int64_t sumQueryLengths = 0;
    bool anyPastNonZero = false;
    for (PhaseRaggedSequence const& sequence : sequences)
    {
        ELLM_CHECK(sequence.queryLength > 0, "phase ragged sequence query length must be positive");
        ELLM_CHECK(sequence.pastLength >= 0, "phase ragged sequence past length must not be negative");
        if (work == SequenceWork::kDecode)
        {
            ELLM_CHECK(sequence.queryLength == 1, "vanilla decode requires query length one");
        }
        anyPastNonZero = anyPastNonZero || sequence.pastLength > 0;
        queryWidth = std::max(queryWidth, sequence.queryLength);
        sumQueryLengths += sequence.queryLength;
    }

    int64_t const physicalTokens64 = packed ? sumQueryLengths : static_cast<int64_t>(numSequences) * queryWidth;
    ELLM_CHECK(physicalTokens64 <= mMaxPhysicalTokens, "phase ragged batch exceeds its configured token capacity");
    int32_t const physicalTokens = static_cast<int32_t>(physicalTokens64);

    int32_t const numContextSequences = (work == SequenceWork::kContext) ? numSequences : 0;

    mBatch.layout = layout;
    mBatch.sequenceOrder.clear();
    mBatch.positions.clear();
    mBatch.queryStartOffsets.clear();
    mBatch.queryLengths.clear();
    mBatch.pastLengths.clear();
    mBatch.attentionSequenceLengths.clear();
    mBatch.stateIndices.clear();
    mBatch.logitsIndices.clear();
    mBatch.logitsToSequence.clear();
    mBatch.sequenceWorks.clear();

    // Packed rows are contiguous [o_i, o_i+q_i) with no padding; nothing needs an init sentinel.
    mBatch.positions.assign(static_cast<size_t>(physicalTokens), packed ? 0 : -1);
    mBatch.queryStartOffsets.resize(static_cast<size_t>(numSequences) + 1);

    int64_t numContextTokens = 0;
    int32_t runningOffset = 0;
    for (int32_t row = 0; row < numSequences; ++row)
    {
        PhaseRaggedSequence const& sequence = sequences[static_cast<size_t>(row)];
        int32_t const physicalStart = packed ? runningOffset : row * queryWidth;
        mBatch.queryStartOffsets[static_cast<size_t>(row)] = physicalStart;
        mBatch.queryLengths.push_back(sequence.queryLength);
        mBatch.pastLengths.push_back(sequence.pastLength);
        mBatch.attentionSequenceLengths.push_back(sequence.pastLength + sequence.queryLength);
        mBatch.stateIndices.push_back(row);
        mBatch.sequenceOrder.push_back(SequenceIdentity{sequence.requestId, ResidentRef{row, 1}});
        mBatch.sequenceWorks.push_back(work);

        for (int32_t tokenIndex = 0; tokenIndex < sequence.queryLength; ++tokenIndex)
        {
            mBatch.positions[static_cast<size_t>(physicalStart + tokenIndex)] = sequence.pastLength + tokenIndex;
        }
        mBatch.logitsIndices.push_back(physicalStart + sequence.queryLength - 1);
        mBatch.logitsToSequence.push_back(row);
        numContextTokens += sequence.queryLength;
        runningOffset += sequence.queryLength;
    }
    mBatch.queryStartOffsets[static_cast<size_t>(numSequences)] = physicalTokens;

    mBatch.shape = RaggedStepShape{numSequences, static_cast<int32_t>(numContextTokens), physicalTokens, queryWidth,
        numContextSequences, (work == SequenceWork::kContext) ? static_cast<int32_t>(numContextTokens) : 0,
        numSequences};

    mExecutionPhase = (work == SequenceWork::kDecode)
        ? ExecutionPhase::kAutoregressiveDecode
        : (anyPastNonZero ? ExecutionPhase::kContextChunk : ExecutionPhase::kContextPrefill);

    mHasPriorBuild = true;
    mPriorWork = work;
    mPriorNumSequences = numSequences;
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

} // namespace rt
} // namespace trt_edgellm
