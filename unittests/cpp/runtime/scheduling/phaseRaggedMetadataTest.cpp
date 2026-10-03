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

#include <gtest/gtest.h>

#include <algorithm>

using namespace trt_edgellm::rt;

namespace
{

std::vector<PhaseRaggedSequence> makeRows(std::vector<int32_t> const& pastLengths, int32_t queryLength)
{
    std::vector<PhaseRaggedSequence> rows;
    rows.reserve(pastLengths.size());
    uint64_t requestId = 1;
    for (int32_t pastLength : pastLengths)
    {
        rows.push_back(PhaseRaggedSequence{requestId++, pastLength, queryLength});
    }
    return rows;
}

} // namespace

TEST(PhaseRaggedMetadataTest, DecodeRowsCarryPerRowPastLengths)
{
    PhaseRaggedMetadataBuilder builder(8, 8);
    RaggedExecutionBatch const& batch = builder.build(SequenceWork::kDecode, makeRows({5, 130, 0}, 1));

    EXPECT_EQ(batch.shape.numSequences, 3);
    EXPECT_EQ(batch.shape.physicalTokens, 3);
    EXPECT_EQ(batch.shape.numContextSequences, 0);
    EXPECT_EQ(batch.positions, (std::vector<int32_t>{5, 130, 0}));
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 1, 2, 3}));
    EXPECT_EQ(batch.pastLengths, (std::vector<int32_t>{5, 130, 0}));
    EXPECT_EQ(batch.queryLengths, (std::vector<int32_t>{1, 1, 1}));
    EXPECT_EQ(batch.attentionSequenceLengths, (std::vector<int32_t>{6, 131, 1}));
    EXPECT_EQ(batch.stateIndices, (std::vector<int32_t>{0, 1, 2}));
    EXPECT_EQ(batch.logitsIndices, (std::vector<int64_t>{0, 1, 2}));
    EXPECT_EQ(builder.executionPhase(), ExecutionPhase::kAutoregressiveDecode);
}

TEST(PhaseRaggedMetadataTest, PrefillUsesEntryPaddedRows)
{
    PhaseRaggedMetadataBuilder builder(8, 32);
    std::vector<PhaseRaggedSequence> rows{{1, 0, 3}, {2, 0, 5}, {3, 0, 1}};
    RaggedExecutionBatch const& batch = builder.build(SequenceWork::kContext, rows);

    EXPECT_EQ(batch.shape.queryWidth, 5);
    EXPECT_EQ(batch.shape.physicalTokens, 15);
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 5, 10, 15}));
    std::vector<int32_t> const expectedPositions{0, 1, 2, -1, -1, 0, 1, 2, 3, 4, 0, -1, -1, -1, -1};
    EXPECT_EQ(batch.positions, expectedPositions);
    EXPECT_EQ(batch.logitsIndices, (std::vector<int64_t>{2, 9, 10}));
    EXPECT_EQ(builder.executionPhase(), ExecutionPhase::kContextPrefill);
}

TEST(PhaseRaggedMetadataTest, ChunkContinuationSelectsContextChunk)
{
    PhaseRaggedMetadataBuilder builder(4, 512);
    std::vector<PhaseRaggedSequence> rows{{1, 128, 128}, {2, 0, 64}};
    RaggedExecutionBatch const& batch = builder.build(SequenceWork::kContext, rows);

    EXPECT_EQ(batch.attentionSequenceLengths, (std::vector<int32_t>{256, 64}));
    for (int32_t t = 0; t < 128; ++t)
    {
        EXPECT_EQ(batch.positions[static_cast<size_t>(t)], 128 + t);
    }
    EXPECT_EQ(builder.executionPhase(), ExecutionPhase::kContextChunk);
}

TEST(PhaseRaggedMetadataTest, SingleSequenceIsDegenerateEntryPadding)
{
    PhaseRaggedMetadataBuilder builder(1, 16);
    RaggedExecutionBatch const& batch = builder.build(SequenceWork::kContext, {{1, 0, 7}});

    EXPECT_EQ(batch.shape.queryWidth, 7);
    EXPECT_EQ(batch.shape.physicalTokens, 7);
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 7}));
}

TEST(PhaseRaggedMetadataTest, RejectsInvalidSteps)
{
    PhaseRaggedMetadataBuilder builder(2, 4);
    EXPECT_THROW(builder.build(SequenceWork::kContext, {}), std::runtime_error);
    EXPECT_THROW(builder.build(SequenceWork::kContext, {{1, 0, 0}}), std::runtime_error);
    EXPECT_THROW(builder.build(SequenceWork::kDecode, {{1, 0, 2}}), std::runtime_error);
    EXPECT_THROW(builder.build(SequenceWork::kContext, {{1, -1, 1}}), std::runtime_error);
    EXPECT_THROW(builder.build(SequenceWork::kContext, {{1, 0, 3}, {2, 0, 3}, {3, 0, 3}}), std::runtime_error);
}

TEST(PhaseRaggedMetadataTest, DecodeIncrementalStepMatchesFullRebuildWhenRowCountIsStable)
{
    PhaseRaggedMetadataBuilder reference(8, 8);
    PhaseRaggedMetadataBuilder incremental(8, 8);

    // Three decode steps at a stable row count (the common case the incremental path targets),
    // each advancing every row's past length by one, as a real decode loop does.
    std::vector<std::vector<int32_t>> const steps{{5, 130, 0}, {6, 131, 1}, {7, 132, 2}};
    for (std::vector<int32_t> const& pastLengths : steps)
    {
        RaggedExecutionBatch const& expected = reference.build(SequenceWork::kDecode, makeRows(pastLengths, 1));
        RaggedExecutionBatch const& actual = incremental.build(SequenceWork::kDecode, makeRows(pastLengths, 1));

        EXPECT_EQ(actual.positions, expected.positions);
        EXPECT_EQ(actual.queryStartOffsets, expected.queryStartOffsets);
        EXPECT_EQ(actual.queryLengths, expected.queryLengths);
        EXPECT_EQ(actual.pastLengths, expected.pastLengths);
        EXPECT_EQ(actual.attentionSequenceLengths, expected.attentionSequenceLengths);
        EXPECT_EQ(actual.stateIndices, expected.stateIndices);
        EXPECT_EQ(actual.logitsIndices, expected.logitsIndices);
        EXPECT_EQ(actual.sequenceOrder.size(), expected.sequenceOrder.size());
        for (size_t row = 0; row < actual.sequenceOrder.size(); ++row)
        {
            EXPECT_EQ(actual.sequenceOrder[row].requestId, expected.sequenceOrder[row].requestId);
            EXPECT_EQ(actual.sequenceOrder[row].resident.slot, expected.sequenceOrder[row].resident.slot);
        }
        EXPECT_EQ(incremental.executionPhase(), ExecutionPhase::kAutoregressiveDecode);
    }
}

TEST(PhaseRaggedMetadataTest, DecodeRowCountChangeFallsBackToFullRebuild)
{
    PhaseRaggedMetadataBuilder builder(8, 8);

    // Stable at N=3 primes the incremental path; dropping to N=2 and growing back to N=3 with
    // different request ids must not leak stale rows from the N=3 incremental snapshot.
    RaggedExecutionBatch const& wide = builder.build(SequenceWork::kDecode, makeRows({5, 130, 0}, 1));
    EXPECT_EQ(wide.shape.numSequences, 3);

    RaggedExecutionBatch const& narrow = builder.build(SequenceWork::kDecode, makeRows({42, 7}, 1));
    EXPECT_EQ(narrow.shape.numSequences, 2);
    EXPECT_EQ(narrow.positions, (std::vector<int32_t>{42, 7}));
    EXPECT_EQ(narrow.queryStartOffsets, (std::vector<int32_t>{0, 1, 2}));
    EXPECT_EQ(narrow.stateIndices, (std::vector<int32_t>{0, 1}));

    RaggedExecutionBatch const& regrown = builder.build(SequenceWork::kDecode, makeRows({9, 9, 9}, 1));
    EXPECT_EQ(regrown.shape.numSequences, 3);
    EXPECT_EQ(regrown.positions, (std::vector<int32_t>{9, 9, 9}));
    EXPECT_EQ(regrown.queryStartOffsets, (std::vector<int32_t>{0, 1, 2, 3}));
    EXPECT_EQ(regrown.stateIndices, (std::vector<int32_t>{0, 1, 2}));
    EXPECT_EQ(regrown.sequenceOrder[0].requestId, 1u);
    EXPECT_EQ(regrown.sequenceOrder[1].requestId, 2u);
    EXPECT_EQ(regrown.sequenceOrder[2].requestId, 3u);
}

TEST(PhaseRaggedMetadataTest, PrefillBetweenDecodeStepsForcesFullRebuild)
{
    PhaseRaggedMetadataBuilder builder(8, 32);

    RaggedExecutionBatch const& decodeBefore = builder.build(SequenceWork::kDecode, makeRows({5, 130, 0}, 1));
    EXPECT_EQ(decodeBefore.shape.numSequences, 3);

    // A context step reusing the same builder (as a rebinding runtime might) must not be
    // mistaken for a stable-row-count decode step.
    RaggedExecutionBatch const& context = builder.build(SequenceWork::kContext, {{9, 0, 2}, {10, 0, 2}, {11, 0, 2}});
    EXPECT_EQ(builder.executionPhase(), ExecutionPhase::kContextPrefill);
    EXPECT_EQ(context.shape.queryWidth, 2);

    RaggedExecutionBatch const& decodeAfter = builder.build(SequenceWork::kDecode, makeRows({1, 2, 3}, 1));
    EXPECT_EQ(decodeAfter.positions, (std::vector<int32_t>{1, 2, 3}));
    EXPECT_EQ(decodeAfter.queryStartOffsets, (std::vector<int32_t>{0, 1, 2, 3}));
    EXPECT_EQ(builder.executionPhase(), ExecutionPhase::kAutoregressiveDecode);
}

TEST(PhaseRaggedMetadataTest, QueryWidthHelperMatchesMaxQueryLength)
{
    std::vector<PhaseRaggedSequence> rows{{1, 0, 3}, {2, 0, 7}, {3, 0, 1}};
    EXPECT_EQ(phaseRaggedQueryWidth(rows), 7);
}

TEST(PhaseRaggedMetadataTest, PackedPrefillUsesPrefixSumOffsets)
{
    PhaseRaggedMetadataBuilder builder(8, 32);
    std::vector<PhaseRaggedSequence> rows{{1, 0, 3}, {2, 0, 5}, {3, 0, 1}};
    RaggedExecutionBatch const& batch
        = builder.build(SequenceWork::kContext, rows, TokenLayoutBackend::kNativeCompactRagged);

    EXPECT_EQ(batch.layout, TokenLayoutBackend::kNativeCompactRagged);
    EXPECT_EQ(batch.shape.queryWidth, 5);
    EXPECT_EQ(batch.shape.physicalTokens, 9);
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 3, 8, 9}));
    std::vector<int32_t> const expectedPositions{0, 1, 2, 0, 1, 2, 3, 4, 0};
    EXPECT_EQ(batch.positions, expectedPositions);
    EXPECT_EQ(batch.logitsIndices, (std::vector<int64_t>{2, 7, 8}));
    EXPECT_EQ(builder.executionPhase(), ExecutionPhase::kContextPrefill);
}

TEST(PhaseRaggedMetadataTest, PackedChunkContinuationUsesPrefixSumWithPast)
{
    PhaseRaggedMetadataBuilder builder(4, 512);
    std::vector<PhaseRaggedSequence> rows{{1, 128, 64}, {2, 0, 32}};
    RaggedExecutionBatch const& batch
        = builder.build(SequenceWork::kContext, rows, TokenLayoutBackend::kNativeCompactRagged);

    EXPECT_EQ(batch.shape.physicalTokens, 96);
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 64, 96}));
    for (int32_t t = 0; t < 64; ++t)
    {
        EXPECT_EQ(batch.positions[static_cast<size_t>(t)], 128 + t);
    }
    for (int32_t t = 0; t < 32; ++t)
    {
        EXPECT_EQ(batch.positions[static_cast<size_t>(64 + t)], t);
    }
    EXPECT_EQ(batch.logitsIndices, (std::vector<int64_t>{63, 95}));
    EXPECT_EQ(builder.executionPhase(), ExecutionPhase::kContextChunk);
}

TEST(PhaseRaggedMetadataTest, PackedSingleSequenceMatchesEntryPadded)
{
    PhaseRaggedMetadataBuilder builder(1, 16);
    RaggedExecutionBatch const& batch
        = builder.build(SequenceWork::kContext, {{1, 0, 7}}, TokenLayoutBackend::kNativeCompactRagged);

    EXPECT_EQ(batch.shape.queryWidth, 7);
    EXPECT_EQ(batch.shape.physicalTokens, 7);
    EXPECT_EQ(batch.queryStartOffsets, (std::vector<int32_t>{0, 7}));
}

TEST(PhaseRaggedMetadataTest, PackedAndEntryPaddedDecodeFormulasCoincide)
{
    PhaseRaggedMetadataBuilder entryPaddedBuilder(8, 8);
    PhaseRaggedMetadataBuilder packedBuilder(8, 8);
    std::vector<PhaseRaggedSequence> rows = makeRows({5, 130, 0}, 1);

    RaggedExecutionBatch const& entryPadded = entryPaddedBuilder.build(SequenceWork::kDecode, rows);
    RaggedExecutionBatch const& packed
        = packedBuilder.build(SequenceWork::kDecode, rows, TokenLayoutBackend::kNativeCompactRagged);

    EXPECT_EQ(entryPadded.queryStartOffsets, packed.queryStartOffsets);
    EXPECT_EQ(entryPadded.positions, packed.positions);
    EXPECT_EQ(entryPadded.logitsIndices, packed.logitsIndices);
    EXPECT_EQ(entryPadded.shape.physicalTokens, packed.shape.physicalTokens);
}

// IndependentPhaseCoordinator sizes mPrefillRaggedMetadata as
// maxSupportedPrefillBatchSize * (packed ? chunkCap : maxSupportedInputLength); a non-packed
// engine has maxPackedPrefillChunkTokens == 0, so using that term directly (pre-fix) collapsed the
// capacity to batch*1 and any multi-token prompt overflowed it.
TEST(PhaseRaggedMetadataTest, NonPackedCoordinatorCapacityFormulaFitsAMultiTokenPrompt)
{
    int32_t const prefillBatch = 8;
    int32_t const maxSupportedInputLength = 1024;
    int32_t const maxPackedPrefillChunkTokens = 0; // non-packed engines serialize this as zero.
    bool const packedPrefill = false;

    int32_t const capacity
        = prefillBatch * (packedPrefill ? std::max(maxPackedPrefillChunkTokens, 1) : maxSupportedInputLength);
    PhaseRaggedMetadataBuilder builder(prefillBatch, capacity);

    EXPECT_NO_THROW(builder.build(SequenceWork::kContext, makeRows({0}, 128)));
}
