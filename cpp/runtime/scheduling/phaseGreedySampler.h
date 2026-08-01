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

#include "runtime/scheduling/phaseContextBatchAdapter.h"
#include "runtime/scheduling/phasePrefillContextBatchAdapter.h"
#include "runtime/state/decodingInferenceContext.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Owns phase-local top-1 sampling buffers and asynchronous D2H completion.
class PhaseGreedySampler
{
public:
    PhaseGreedySampler(
        int32_t maxBatchSize, int32_t vocabSize, std::vector<int32_t> eosTokenIds, std::string const& name);

    //! Enqueue top-1 selection and D2H on the phase stream without synchronizing.
    void enqueue(Tensor const& logits, int32_t batchSize, cudaStream_t stream);
    //! Append sampled tokens only for rows completing their final prompt chunk.
    void completePrefill(PhasePrefillContextBatchAdapter const& adapter);
    //! Append one sampled token to every packed decode row.
    void completeDecode(DecodingInferenceContext& context);
    //! Append one token while applying each source row's own generation budget.
    void completeDecode(PhaseContextBatchAdapter& adapter);

    bool pending() const noexcept;

private:
    void completeRow(
        DecodingInferenceContext& context, int32_t row, int32_t tokenId, int32_t maxGenerateLength) const;
    void finishCompletion(int32_t batchSize);

    int32_t mMaxBatchSize{};
    int32_t mVocabSize{};
    std::vector<int32_t> mEosTokenIds;
    Tensor mSelectedTokenIds;
    Tensor mHostSelectedTokenIds;
    Tensor mWorkspace;
    int32_t mPendingBatchSize{};
};

} // namespace rt
} // namespace trt_edgellm
