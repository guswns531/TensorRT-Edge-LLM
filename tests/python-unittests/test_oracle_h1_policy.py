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

import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[
    2] / "benchmarks/phase_serving/analyze_oracle_h1_policy.py"
SPEC = importlib.util.spec_from_file_location("analyze_oracle_h1_policy",
                                              MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def candidate(label,
              action_id,
              boundary,
              slack,
              progress,
              work,
              *,
              valid=True):
    return {
        "label":
        label,
        "action_id":
        action_id,
        "projection": {
            "valid": valid,
            "action_id": action_id,
            "boundary_us": boundary,
            "robust_boundary_us": boundary,
        },
        "milestones": [{
            "slack_us": slack,
            "progress_units": progress,
            "completed_at_boundary": progress > 0,
        }],
        "reference_work_us":
        work,
    }


def test_oracle_uses_slo_then_progress_then_efficiency():
    payload = {
        "episodes": [{
            "episode_id":
            1,
            "action_fidelity":
            True,
            "candidates": [
                candidate("late", 1, 6, 5, 100, 600),
                candidate("safe", 2, 4, 5, 1, 4),
            ],
            "selections": {
                "myopic": "late",
                "h2": "safe"
            },
        }]
    }
    result = MODULE.analyze(payload)
    assert result["episodes"][0]["oracle"] == "safe"
    assert result["comparisons"]["myopic"]["oracle_changes"] == 1
    assert result["comparisons"]["h2"]["agreements"] == 1


def test_oracle_rejects_invalid_candidate_and_reports_incomplete_gate_b():
    payload = {
        "episodes": [{
            "episode_id":
            1,
            "action_fidelity":
            True,
            "workload":
            "controlled",
            "candidates": [
                candidate("invalid", 1, 1, 10, 10, 100, valid=False),
                candidate("serial", 2, 2, 10, 1, 2),
            ],
            "selections": {
                "myopic": "serial"
            },
        }]
    }
    result = MODULE.analyze(payload)
    assert result["episodes"][0]["oracle"] == "serial"
    assert not result["coverage"]["gate_b_evaluable"]


def test_oracle_requires_a_feasible_candidate():
    payload = {
        "episodes": [{
            "episode_id":
            1,
            "action_fidelity":
            True,
            "candidates": [candidate("bad", 1, 1, 10, 1, 1, valid=False)],
            "selections": {
                "myopic": "bad"
            },
        }]
    }
    with pytest.raises(ValueError, match="no feasible candidate"):
        MODULE.analyze(payload)
