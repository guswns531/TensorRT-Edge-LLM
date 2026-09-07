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

namespace trt_edgellm::rt
{

//! Ownership changes caused by one action. Near reclaim is a ranking signal and
//! never contributes to hard feasibility until its completion event is observed.
struct PhaseActionMemoryHorizon
{
    size_t managedBytes{};
    size_t allocateBytes{};
    size_t immediateReclaimBytes{};
    size_t guaranteedGrowthBytes{};
    size_t nearReclaimBytes{};
    size_t budgetBytes{};
    bool immediateReclaimObserved{};
};

} // namespace trt_edgellm::rt
