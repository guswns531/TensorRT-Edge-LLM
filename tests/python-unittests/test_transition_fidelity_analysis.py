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
import json
from pathlib import Path


def _module():
    root = Path(__file__).resolve().parents[2]
    path = root / "benchmarks/phase_serving/analyze_transition_fidelity.py"
    spec = importlib.util.spec_from_file_location("transition_fidelity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _transition(horizon: float, decode_rows: int) -> dict:
    return {
        "evaluated": True,
        "valid": True,
        "alternatives": 1,
        "worst_case_robust_horizon_us": horizon,
        "first_completed_phase_mask": 2,
        "min_first_encoder_ready_rows": 0,
        "max_first_encoder_ready_rows": 0,
        "min_first_prefill_ready_rows": 0,
        "max_first_prefill_ready_rows": 0,
        "min_first_decode_ready_rows": decode_rows,
        "max_first_decode_ready_rows": decode_rows,
        "min_encoder_ready_rows": 0,
        "max_encoder_ready_rows": 0,
        "min_prefill_ready_rows": 0,
        "max_prefill_ready_rows": 0,
        "min_decode_ready_rows": decode_rows,
        "max_decode_ready_rows": decode_rows,
        "min_reclaim_bytes": 0,
        "max_reclaim_bytes": 0,
    }


def test_analyzes_common_snapshot_and_false_safe(tmp_path):
    candidate = {
        "action_id": 7,
        "action_kind": "prefill_decode",
        "contextual_direction": "prefill_to_decode",
        "contextual_incumbent_reference_us": 6.0,
        "contextual_newcomer_reference_us": 4.0,
        "scalar_decision_cost_known": True,
        "scalar_decision_makespan_us": 8.0,
        "contextual_effect_ready": True,
        "contextual_effect_compression_mean": 0.25,
        "contextual_effect_compression_uncertainty": 0.05,
        "contextual_completion_ready": True,
        "contextual_incumbent_mean_us": 5.0,
        "contextual_incumbent_uncertainty_us": 0.2,
        "contextual_newcomer_mean_us": 7.0,
        "contextual_newcomer_uncertainty_us": 0.2,
        "scalar_transition": _transition(8.0, 1),
        "effect_transition": _transition(7.5, 2),
        "completion_transition": _transition(7.2, 2),
    }
    events = [{
        "event_kind": "decision",
        "decision_id": 1,
        "selected_action_id": 7,
        "frozen_transition_snapshot_valid": True,
        "candidates": [candidate],
    }, {
        "event_kind": "completion",
        "decision_id": 1,
        "action_kind": "prefill_decode",
        "action_fidelity": True,
        "gpu_start_us": 100.0,
        "incumbent_gpu_completion_us": 111.0,
        "newcomer_gpu_completion_us": 109.0,
    }]
    path = tmp_path / "balanced" / "run-001-events.jsonl"
    path.parent.mkdir()
    path.write_text("".join("PHASE_SCHEDULER_EVENT\t" + json.dumps(event) +
                            "\n" for event in events),
                    encoding="utf-8")

    result = _module().analyze(tmp_path)

    assert result["frozen_snapshot_coverage"] == 1.0
    assert result["transition_valid"]["effect"]["coverage"] == 1.0
    assert result["transition_disagreement"]["scalar_effect"][
        "state_disagreements"] == 1
    assert result["selected_physics"]["effect"]["false_safe"] == 1
    assert result["selected_physics"]["completion"]["false_safe"] == 1
    assert result["selected_physics"]["scalar"]["makespan_mae_us"] == 3.0
    assert not result["selected_physics"]["effect"]["promotion_gate"]["passed"]


def test_groups_warmup_matrix_by_workload(tmp_path):
    event = {
        "event_kind": "decision",
        "decision_id": 1,
        "selected_action_id": 7,
        "frozen_transition_snapshot_valid": True,
        "candidates": [],
    }
    path = (tmp_path / "generic" / "balanced" / "worker-4" / "activity" /
            "run-001-events.jsonl")
    path.parent.mkdir(parents=True)
    path.write_text("PHASE_SCHEDULER_EVENT\t" + json.dumps(event) + "\n",
                    encoding="utf-8")

    result = _module().analyze(tmp_path)

    assert list(result["by_workload"]) == ["balanced"]
    assert result["by_workload"]["balanced"]["decisions"] == 1
