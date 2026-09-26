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
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>

namespace trt_edgellm::rt
{

struct PhaseQueueSchedulerConfig;
using PhaseEnvironmentLookup = std::function<char const*(char const*)>;

//! Resolve shared prefill/transition overrides before allocating phase I/O.
void resolvePhasePrefillExecutionOptions(PhaseQueueSchedulerConfig& config, int32_t engineChunkLimit,
    PhaseEnvironmentLookup const& environment = std::getenv);

struct PhaseGraphExecutionOptions
{
    bool enabled{};
    bool onlineCapture{};
    size_t minObservations{8U};
    size_t maxPrefillGraphs{4U}; //!< Zero disables capture and evicts cached graphs for this phase.
    size_t maxDecodeGraphs{8U};  //!< Zero disables capture and evicts cached graphs for this phase.
};

//! Resolve graph limits and opt-in serving-time capture identically in both entrypoints.
PhaseGraphExecutionOptions resolvePhaseGraphExecutionOptions(
    PhaseGraphExecutionOptions config, PhaseEnvironmentLookup const& environment = std::getenv);

//! Cover the maximum cohort and geometric smaller shapes before remaining sizes.
std::vector<int32_t> phaseDecodeGraphWarmupBatches(int32_t maxBatchSize);

} // namespace trt_edgellm::rt
