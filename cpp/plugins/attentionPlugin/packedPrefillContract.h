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

#include <cstdint>

namespace trt_edgellm
{
namespace plugins
{

//! Profile-local P bindings retain their chunk capacity even for a one-token tail;
//! D bindings use capacity one. A missing carrier (zero) cannot distinguish P1 from D1.
constexpr bool isPackedPrefillInvocation(
    bool enabled, int32_t physicalBatchSize, int32_t sequenceLength, int32_t profileChunkLimit) noexcept
{
    return enabled && physicalBatchSize == 1 && (sequenceLength > 1 || (sequenceLength == 1 && profileChunkLimit > 1));
}

} // namespace plugins
} // namespace trt_edgellm
