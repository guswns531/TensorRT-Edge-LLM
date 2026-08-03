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

#include "runtime/scheduling/phaseContextServingFacade.h"
#include "runtime/scheduling/phaseDispatchWorker.h"
#include "runtime/scheduling/phaseEncoderDispatchWorker.h"

#include <cstddef>

namespace trt_edgellm
{
namespace rt
{

//! Drives encoder and LLM workers without host-side stream synchronization.
//!
//! Encoder completion hands work into the shared prefill scheduler. The LLM
//! worker may independently overlap prefill/decode while the next encoder
//! batch is executing. All streams must belong to one CUDA primary context.
class PhaseThreeCoordinator
{
public:
    PhaseThreeCoordinator(PhaseEncoderDispatchWorker& encoderWorker, PhaseDispatchWorker& llmWorker);

    //! Query completed events and launch every currently available phase.
    bool step();
    void runUntilIdle(size_t maxDispatches);

    bool empty() const noexcept;
    size_t encoderDispatchCount() const noexcept;
    size_t llmDispatchCount() const noexcept;

private:
    PhaseEncoderDispatchWorker& mEncoderWorker;
    PhaseDispatchWorker& mLlmWorker;
};

//! Drives encoder work and the continuous-context serving facade together.
//!
//! Requests may hold a stable slot in kEncoder state while visual work is in
//! flight. Encoder completion uses PhaseContextServingFacade::beginPrefillAfterEncoder
//! to make the request runnable by the LLM phase queues.
class PhaseOnlineCoordinator
{
public:
    PhaseOnlineCoordinator(PhaseEncoderDispatchWorker& encoderWorker, PhaseContextServingFacade& servingFacade);

    bool step();
    void runUntilIdle(size_t maxDispatches);

    bool empty() const noexcept;
    size_t encoderDispatchCount() const noexcept;
    size_t llmDispatchCount() const noexcept;

private:
    PhaseEncoderDispatchWorker& mEncoderWorker;
    PhaseContextServingFacade& mServingFacade;
    size_t mLlmDispatchCount{};
};

} // namespace rt
} // namespace trt_edgellm
