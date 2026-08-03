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

#include "runtime/scheduling/phaseThreeCoordinator.h"

#include "common/checkMacros.h"

#include <thread>

namespace trt_edgellm
{
namespace rt
{

PhaseThreeCoordinator::PhaseThreeCoordinator(PhaseEncoderDispatchWorker& encoderWorker, PhaseDispatchWorker& llmWorker)
    : mEncoderWorker(encoderWorker)
    , mLlmWorker(llmWorker)
{
    check::check(mEncoderWorker.cudaContext() == mLlmWorker.cudaContext(),
        "Encoder, prefill, and decode streams must share one CUDA context.");
}

bool PhaseThreeCoordinator::step()
{
    bool progressed{};
    if (mEncoderWorker.busy())
    {
        progressed = mEncoderWorker.poll() || progressed;
    }
    if (mLlmWorker.busy())
    {
        progressed = mLlmWorker.poll() || progressed;
    }
    if (!mEncoderWorker.busy() && mEncoderWorker.queueSize() > 0)
    {
        progressed = mEncoderWorker.dispatchNext() || progressed;
    }
    if (!mLlmWorker.busy() && !mLlmWorker.empty())
    {
        progressed = mLlmWorker.dispatchNext() || progressed;
    }
    return progressed;
}

void PhaseThreeCoordinator::runUntilIdle(size_t maxDispatches)
{
    check::check(maxDispatches > 0, "Three-phase coordinator maxDispatches must be positive.");
    size_t const initialDispatches = encoderDispatchCount() + llmDispatchCount();
    while (!empty())
    {
        if (!step())
        {
            std::this_thread::yield();
        }
        check::check(encoderDispatchCount() + llmDispatchCount() - initialDispatches <= maxDispatches,
            "Three-phase coordinator exceeded its dispatch limit.");
    }
}

bool PhaseThreeCoordinator::empty() const noexcept
{
    return mEncoderWorker.empty() && mLlmWorker.empty();
}

size_t PhaseThreeCoordinator::encoderDispatchCount() const noexcept
{
    return mEncoderWorker.dispatchCount();
}

size_t PhaseThreeCoordinator::llmDispatchCount() const noexcept
{
    return mLlmWorker.dispatchCount();
}

PhaseOnlineCoordinator::PhaseOnlineCoordinator(
    PhaseEncoderDispatchWorker& encoderWorker, PhaseContextServingFacade& servingFacade)
    : mEncoderWorker(encoderWorker)
    , mServingFacade(servingFacade)
{
    check::check(mEncoderWorker.cudaContext() == mServingFacade.cudaContext(),
        "Encoder and LLM phase workers must share one CUDA context.");
}

bool PhaseOnlineCoordinator::step()
{
    bool progressed{};
    if (mEncoderWorker.busy())
    {
        progressed = mEncoderWorker.poll() || progressed;
    }
    if (mServingFacade.busy())
    {
        progressed = mServingFacade.poll() || progressed;
    }
    if (!mEncoderWorker.busy() && mEncoderWorker.queueSize() > 0)
    {
        progressed = mEncoderWorker.dispatchNext() || progressed;
    }
    if (!mServingFacade.busy() && mServingFacade.hasQueuedPhaseWork())
    {
        bool const dispatched = mServingFacade.dispatchNext();
        progressed = dispatched || progressed;
        if (dispatched)
        {
            ++mLlmDispatchCount;
        }
    }
    return progressed;
}

void PhaseOnlineCoordinator::runUntilIdle(size_t maxDispatches)
{
    check::check(maxDispatches > 0, "Online phase coordinator maxDispatches must be positive.");
    size_t const initialDispatches = encoderDispatchCount() + llmDispatchCount();
    while (!empty())
    {
        if (!step())
        {
            std::this_thread::yield();
        }
        check::check(encoderDispatchCount() + llmDispatchCount() - initialDispatches <= maxDispatches,
            "Online phase coordinator exceeded its dispatch limit.");
    }
}

bool PhaseOnlineCoordinator::empty() const noexcept
{
    return mEncoderWorker.empty() && mServingFacade.empty();
}

size_t PhaseOnlineCoordinator::encoderDispatchCount() const noexcept
{
    return mEncoderWorker.dispatchCount();
}

size_t PhaseOnlineCoordinator::llmDispatchCount() const noexcept
{
    return mLlmDispatchCount;
}

} // namespace rt
} // namespace trt_edgellm
