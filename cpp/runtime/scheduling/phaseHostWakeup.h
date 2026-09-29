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

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>

namespace trt_edgellm::rt
{

//! Lets an idle host polling loop sleep until another thread publishes work.
//!
//! A waiter reads generation() before checking for work and passes it to waitFor(), so a notification that
//! lands between the check and the wait is never lost.
class PhaseHostWakeup
{
public:
    void notify() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mMutex);
            ++mGeneration;
        }
        mCondition.notify_all();
    }

    uint64_t generation() const noexcept
    {
        std::lock_guard<std::mutex> lock(mMutex);
        return mGeneration;
    }

    //! Returns false when the timeout elapsed without a notification.
    bool waitFor(uint64_t seenGeneration, std::chrono::microseconds timeout)
    {
        std::unique_lock<std::mutex> lock(mMutex);
        return mCondition.wait_for(lock, timeout, [&] { return mGeneration != seenGeneration; });
    }

private:
    mutable std::mutex mMutex;
    std::condition_variable mCondition;
    uint64_t mGeneration{};
};

} // namespace trt_edgellm::rt
