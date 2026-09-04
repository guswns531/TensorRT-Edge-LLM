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

from benchmarks.phase_serving.analyze_completion_opportunities import \
    completion_authority_attribution


def test_attributes_completion_authority_on_the_same_h1_frontier() -> None:
    decisions = [{
        "selected_action_id":
        20,
        "active_h1_selected_action_id":
        20,
        "scalar_h1_selected_action_id":
        10,
        "candidates": [{
            "contextual_direction":
            "prefill_to_decode",
            "completion_policy_evaluated":
            True,
            "completion_authority_ready":
            True,
            "completion_authority_applied":
            True,
            "scalar_decision_cost_known":
            True,
            "active_decision_cost_known":
            True,
            "scalar_decision_makespan_us":
            10.0,
            "active_decision_makespan_us":
            12.0,
            "scalar_protected_completions": [{
                "kind": "decode",
                "predicted_completion_us": 7.0,
                "uncertainty_us": 2.0,
            }],
            "active_protected_completions": [{
                "kind": "decode",
                "predicted_completion_us": 8.0,
                "uncertainty_us": 3.0,
            }],
        }],
    }]

    result = completion_authority_attribution(decisions)

    assert result["evaluated_candidates"] == 1
    assert result["authority_ready_candidates"] == 1
    assert result["authority_applied_candidates"] == 1
    assert result["h1_action_attribution"]["changed_decisions"] == 1
    assert result["h1_action_attribution"]["active_to_final_changes"] == 0
    assert result["h1_action_attribution"]["scalar_to_final_changes"] == 1
    assert result["active_minus_scalar_decision_makespan_us"]["median"] == 2.0
    assert result["active_minus_scalar_protected_robust_us"]["decode"][
        "median"] == 2.0
    assert result["by_direction"]["prefill_to_decode"]["cost_increased"] == 1


def test_skips_unavailable_counterfactual_fields() -> None:
    result = completion_authority_attribution([{
        "active_h1_selected_action_id":
        0,
        "scalar_h1_selected_action_id":
        0,
        "candidates": [{
            "completion_policy_evaluated": False,
        }],
    }])

    assert result["evaluated_candidates"] == 0
    assert result["h1_action_attribution"]["comparable_decisions"] == 0
    assert result["h1_action_attribution"]["changed_fraction"] is None
