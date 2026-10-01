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

TEST(PhaseRaggedMetadataTest, QueryWidthHelperMatchesMaxQueryLength)
{
    std::vector<PhaseRaggedSequence> rows{{1, 0, 3}, {2, 0, 7}, {3, 0, 1}};
    EXPECT_EQ(phaseRaggedQueryWidth(rows), 7);
}
