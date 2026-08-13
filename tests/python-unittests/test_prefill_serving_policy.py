# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from scripts.cosmos_reason2 import select_prefill_serving_policy


def test_prefers_profiled_graph_when_memory_contract_fits():
    stats = {"request_count": 288, "offered_rps": 1000.0}
    policy, _ = select_prefill_serving_policy.select_policy(
        stats,
        available_mib=927,
        graph_budget_mib=240,
        reserve_mib=256,
        graph_profile_available=True)
    assert policy == "chunk128_graph"


def test_uses_large_graph_off_chunks_for_memory_constrained_backlog():
    stats = {"request_count": 288, "offered_rps": 1000.0}
    policy, _ = select_prefill_serving_policy.select_policy(
        stats,
        available_mib=400,
        graph_budget_mib=240,
        reserve_mib=256,
        graph_profile_available=True)
    assert policy == "chunk256_graph_off"


def test_short_memory_constrained_trace_avoids_large_workspace():
    stats = {"request_count": 48, "offered_rps": 30.0}
    policy, _ = select_prefill_serving_policy.select_policy(
        stats,
        available_mib=400,
        graph_budget_mib=240,
        reserve_mib=256,
        graph_profile_available=False)
    assert policy == "chunk128_graph_off"
