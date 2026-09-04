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

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    root = Path(__file__).resolve().parents[2]
    path = root / "benchmarks/phase_serving/analyze_effect_vector_shadow.py"
    spec = importlib.util.spec_from_file_location("effect_vector_analysis",
                                                  path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_analyze_joins_selected_candidate_to_common_epoch(tmp_path: Path):
    decision = {
        "event_kind":
        "decision",
        "decision_id":
        9,
        "selected_action_id":
        77,
        "candidates": [{
            "action_id": 77,
            "contextual_effect_valid": True,
            "contextual_effect_ready": True,
            "contextual_direction": "prefill_to_decode",
            "contextual_incumbent_reference_us": 4.0,
            "contextual_newcomer_reference_us": 8.0,
            "scalar_decision_cost_known": True,
            "scalar_decision_makespan_us": 7.0,
            "predicted_completion_us": [7.0],
            "contextual_incumbent_mean_us": 5.0,
            "contextual_newcomer_mean_us": 6.0,
            "contextual_effect_compression_mean": 0.5,
            "contextual_effect_incumbent_stretch_mean": 0.25,
            "contextual_effect_order_margin_mean": 1.0 / 12.0,
            "contextual_effect_order_margin_uncertainty": 0.01,
        }],
    }
    completion = {
        "event_kind": "completion",
        "decision_id": 9,
        "action_kind": "prefill_decode",
        "action_fidelity": True,
        "gpu_start_us": 100.0,
        "incumbent_gpu_completion_us": 105.0,
        "newcomer_gpu_completion_us": 106.0,
    }
    path = tmp_path / "run-001-events.jsonl"
    path.write_text("\n".join("PHASE_SCHEDULER_EVENT\t" + json.dumps(event)
                              for event in (decision, completion)),
                    encoding="utf-8")

    result = _module().analyze(tmp_path)

    assert result["schema_version"] == 2
    assert result["common_epoch_samples"] == 1
    assert result["effect_ready_samples"] == 1
    assert result["order"]["accuracy"] == 1.0
    assert result["errors"]["effect_compression"]["mae"] == 0.0
    assert result["errors"]["effect_incumbent_stretch"]["mae"] == 0.0
    assert result["errors"]["effect_order_margin"]["mae"] == 0.0
    workload = result["by_workload"]["run-001-events"]
    assert workload["common_epoch_samples"] == 1
    assert workload["by_direction"]["prefill_to_decode"][
        "order_accuracy"] == 1.0
