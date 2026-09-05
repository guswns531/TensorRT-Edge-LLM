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

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = (Path(__file__).parents[2] / "benchmarks" / "phase_serving" /
          "analyze_multi_image_trajectory.py")
SPEC = importlib.util.spec_from_file_location("multi_trajectory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
TRAJECTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAJECTORY)


def test_analyze_run_correlates_decode_fragmentation(tmp_path: Path) -> None:
    variant = tmp_path / "generic" / "multi-image" / "worker-4"
    client = variant / "run-001" / "client"
    activity = variant / "activity"
    client.mkdir(parents=True)
    activity.mkdir()
    (client / "aggregate.json").write_text(json.dumps({
        "generated_token_s_median":
        244.0,
        "ttft_mean_of_run_means_ms":
        300.0,
        "ttft_p95_median_ms":
        330.0,
        "tpot_mean_of_run_means_ms":
        11.0,
        "tpot_p95_median_ms":
        13.0,
        "e2e_mean_of_run_means_ms":
        600.0,
        "e2e_p95_median_ms":
        620.0,
        "by_request_class": {
            "vision": {
                "requests": 5,
            },
        },
    }),
                                           encoding="utf-8")
    with (activity / "run-001-intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=("kind", "name", "duration_ms",
                                            "start_ms"))
        writer.writeheader()
        writer.writerows({
            "kind": "decode",
            "name": "decode_dispatch",
            "duration_ms": "1.5",
            "start_ms": str(100 + index)
        } for index in range(61))
    event = {
        "event_kind":
        "decision",
        "action_kind":
        "decode",
        "selected_action_id":
        "d0",
        "candidates": [{
            "action_id": "d0",
            "scalar_decision_cost_known": True,
            "contextual_direction": "prefill_to_decode",
            "contextual_direction_observations": 7,
        }],
    }
    (activity / "run-001-events.jsonl").write_text("PHASE_SCHEDULER_EVENT\t" +
                                                   json.dumps(event) + "\n",
                                                   encoding="utf-8")

    result = TRAJECTORY.analyze_run(client / "aggregate.json", 40, 50)

    assert result["trajectory_family"] == "fragmented"
    assert result["request_count"] == 5
    assert result["decode_dispatches"] == 61
    assert result["decode_gpu_ms"] == 91.5
    assert result["selected_scalar_known"] == 1
    assert result["direction_observations"]["prefill_to_decode"] == 7


def test_analyze_run_labels_scaled_trace_without_five_request_thresholds(
        tmp_path: Path) -> None:
    variant = tmp_path / "generic" / "multi-image" / "worker-4"
    client = variant / "run-001" / "client"
    activity = variant / "activity"
    client.mkdir(parents=True)
    activity.mkdir()
    (client / "aggregate.json").write_text(json.dumps({
        "generated_token_s_median":
        100.0,
        "ttft_mean_of_run_means_ms":
        10.0,
        "ttft_p95_median_ms":
        11.0,
        "tpot_mean_of_run_means_ms":
        2.0,
        "tpot_p95_median_ms":
        3.0,
        "e2e_mean_of_run_means_ms":
        20.0,
        "e2e_p95_median_ms":
        21.0,
        "by_request_class": {
            "vision": {
                "requests": 40,
            },
        },
    }),
                                           encoding="utf-8")
    with (activity / "run-001-intervals.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream,
                                fieldnames=("kind", "name", "duration_ms",
                                            "start_ms"))
        writer.writeheader()

    result = TRAJECTORY.analyze_run(client / "aggregate.json", 40, 50)

    assert result["trajectory_family"] == "scaled"
    assert result["request_count"] == 40
