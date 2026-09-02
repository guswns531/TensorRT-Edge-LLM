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
#include <limits>

namespace trt_edgellm::rt
{

enum class PhaseProtectedKind
{
    kUnknown,
    kEncoder,
    kPrefill,
    kDecode,
};

//! One request deadline protected while evaluating an action. Completion may
//! include a required follow-up phase when the candidate does not advance the
//! request owning this deadline.
struct PhaseProtectedCompletion
{
    double slackUs{std::numeric_limits<double>::infinity()};
    double predictedCompletionUs{};
    double uncertaintyUs{};
    PhaseProtectedKind kind{PhaseProtectedKind::kUnknown};
};

} // namespace trt_edgellm::rt
