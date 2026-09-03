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

SCRIPT = (Path(__file__).parents[2] / "benchmarks" / "phase_serving" /
          "analyze_contextual_adaptation.py")
SPEC = importlib.util.spec_from_file_location("adaptation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ADAPTATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ADAPTATION)


def test_adaptation_bins_use_measurement_epoch_and_cumulative_deltas(
        tmp_path: Path) -> None:
    log = tmp_path / "gateway.log"
    records = []
    for index in range(1, 9):
        record = {
            "measurement_epoch": 1,
            "policy_warmup_mode": "zero_start",
            "dispatch_index": index,
            "host_dispatch_start_us": index * 1000.0,
            "prefill_request_ids": [index],
        }
        for family in ADAPTATION.FAMILIES:
            record[f"contextual_{family}_calibration_observations"] = index
            record[f"contextual_{family}_squared_error_sum"] = index * 0.01
            record[f"contextual_{family}_absolute_error_sum"] = index * 0.1
            record[f"contextual_{family}_predicted_safe"] = index
            record[f"contextual_{family}_false_safe"] = 0
            record[f"contextual_{family}_last_mean"] = 0.2
            record[f"contextual_{family}_last_uncertainty"] = 0.1
            record[f"contextual_{family}_last_lcb"] = 0.15
        for direction in ADAPTATION.DIRECTIONS:
            record[f"contextual_{direction}_observations"] = index
        records.append(record)
    old = dict(records[0])
    old["measurement_epoch"] = 0
    for family in ADAPTATION.FAMILIES:
        old[f"contextual_{family}_calibration_observations"] = 11
        old[f"contextual_{family}_squared_error_sum"] = 0.11
        old[f"contextual_{family}_absolute_error_sum"] = 1.1
        old[f"contextual_{family}_predicted_safe"] = 11
    for direction in ADAPTATION.DIRECTIONS:
        old[f"contextual_{direction}_observations"] = 11
    for index, record in enumerate(records, start=1):
        for family in ADAPTATION.FAMILIES:
            record[f"contextual_{family}_calibration_observations"] += 11
            record[f"contextual_{family}_squared_error_sum"] += 0.11
            record[f"contextual_{family}_absolute_error_sum"] += 1.1
            record[f"contextual_{family}_predicted_safe"] += 11
        for direction in ADAPTATION.DIRECTIONS:
            record[f"contextual_{direction}_observations"] += 11
    log.write_text("PHASE_METRIC\t" + json.dumps(old) + "\n" +
                   "\n".join("PHASE_METRIC\t" + json.dumps(record)
                             for record in records) + "\n",
                   encoding="utf-8")

    result = ADAPTATION.analyze_log(log,
                                    bins=(2, 4, 6),
                                    minimum_observations=2,
                                    maximum_rmse=0.2)

    assert result["measurement_epoch"] == 1
    assert result["policy_warmup_mode"] == "zero_start"
    assert result["decisions"] == 8
    assert result["epoch_baseline"]["pd"] == 11
    assert result["bins"][0]["families"]["pd"]["observations"] == 2
    assert result["bins"][1]["families"]["pd"]["observations"] == 2
    assert result["stability"]["pd"]["stable"]
    assert result["stability"]["pd"]["decision_to_stability"] == 4
    assert result["stability"]["pd"]["time_to_stability_ms"] == 3.0
    assert result["stability"]["pd"]["request_frontier_to_stability"] == 4


def test_adaptation_does_not_report_stable_after_late_false_safe(
        tmp_path: Path) -> None:
    log = tmp_path / "gateway.log"
    records = []
    for index in range(1, 7):
        record = {
            "measurement_epoch": 0,
            "policy_warmup_mode": "zero_start",
            "contextual_pd_calibration_observations": index,
            "contextual_pd_squared_error_sum": index * 0.01,
            "contextual_pd_absolute_error_sum": index * 0.1,
            "contextual_pd_predicted_safe": index,
            "contextual_pd_false_safe": 1 if index == 6 else 0,
        }
        records.append(record)
    log.write_text("\n".join("PHASE_METRIC\t" + json.dumps(record)
                             for record in records) + "\n",
                   encoding="utf-8")

    result = ADAPTATION.analyze_log(log,
                                    bins=(2, 4),
                                    minimum_observations=2,
                                    maximum_rmse=0.2)

    assert not result["stability"]["pd"]["stable"]
    assert result["stability"]["pd"]["decision_to_stability"] == 4
    assert result["stability"]["pd"]["stability_lost_after_first"]
