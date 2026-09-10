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
import subprocess
import sys
from pathlib import Path

GOODPUT_PATH = Path(__file__).parents[
    2] / "benchmarks" / "phase_serving" / "analyze_slo_goodput.py"
GOODPUT_SPEC = importlib.util.spec_from_file_location("phase_slo_goodput",
                                                      GOODPUT_PATH)
GOODPUT = importlib.util.module_from_spec(GOODPUT_SPEC)
assert GOODPUT_SPEC.loader is not None
GOODPUT_SPEC.loader.exec_module(GOODPUT)

LOAD_PATH = Path(__file__).parents[
    2] / "benchmarks" / "phase_serving" / "materialize_load_sweep.py"
LOAD_SPEC = importlib.util.spec_from_file_location("phase_load_sweep",
                                                   LOAD_PATH)
LOAD = importlib.util.module_from_spec(LOAD_SPEC)
assert LOAD_SPEC.loader is not None
LOAD_SPEC.loader.exec_module(LOAD)


def test_scale_trace_changes_only_arrival_offsets():
    source = {
        "workload":
        "balanced",
        "requests": [{
            "arrival_offset_us": 100,
            "max_generate_length": 32
        }, {
            "arrival_offset_us": 400,
            "max_generate_length": 64
        }],
    }

    scaled = LOAD.scale_trace(source, 2.0)

    assert [item["arrival_offset_us"]
            for item in scaled["requests"]] == [50, 200]
    assert [item["max_generate_length"]
            for item in scaled["requests"]] == [32, 64]


def test_slo_goodput_counts_only_joint_slo_passes(tmp_path):
    path = tmp_path / "requests.csv"
    fieldnames = [
        "request_class", "scheduled_arrival_us", "first_token_us",
        "completed_us", "output_tokens", "http_status", "error", "ttft_ms",
        "tpot_ms", "e2e_ms"
    ]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{
            "request_class": "text",
            "scheduled_arrival_us": 0,
            "first_token_us": 100_000,
            "completed_us": 1_000_000,
            "output_tokens": 32,
            "http_status": 200,
            "error": "",
            "ttft_ms": 100,
            "tpot_ms": 20,
            "e2e_ms": 1000,
        }, {
            "request_class": "text",
            "scheduled_arrival_us": 0,
            "first_token_us": 600_000,
            "completed_us": 2_000_000,
            "output_tokens": 64,
            "http_status": 200,
            "error": "",
            "ttft_ms": 600,
            "tpot_ms": 20,
            "e2e_ms": 2000,
        }])

    result = GOODPUT.summarize(path, ttft_ms=500, tpot_ms=50, e2e_ms=0)

    assert result["passed_requests"] == 1
    assert result["pass_rate"] == 0.5
    assert result["request_goodput_per_s"] == 0.5
    assert result["token_goodput_per_s"] == 16.0
    assert result["failure_reasons"] == {"pass": 1, "ttft": 1}
    assert result["by_request_class"]["text"]["failure_reasons"] == {
        "pass": 1,
        "ttft": 1,
    }


def test_slo_goodput_attributes_joint_failures(tmp_path):
    path = tmp_path / "requests.csv"
    fieldnames = [
        "request_class", "scheduled_arrival_us", "first_token_us",
        "completed_us", "output_tokens", "http_status", "error", "ttft_ms",
        "tpot_ms", "e2e_ms"
    ]
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "request_class": "vision",
            "scheduled_arrival_us": 0,
            "first_token_us": 600_000,
            "completed_us": 3_000_000,
            "output_tokens": 32,
            "http_status": 200,
            "error": "",
            "ttft_ms": 600,
            "tpot_ms": 60,
            "e2e_ms": 3000,
        })

    result = GOODPUT.summarize(path, ttft_ms=500, tpot_ms=50, e2e_ms=2500)

    assert result["passed_requests"] == 0
    assert result["failure_reasons"] == {"ttft+tpot+e2e": 1}


def test_slo_surface_writes_every_threshold_pair(tmp_path):
    requests = tmp_path / "requests.csv"
    output_json = tmp_path / "surface.json"
    output_csv = tmp_path / "surface.csv"
    fieldnames = [
        "request_class", "scheduled_arrival_us", "first_token_us",
        "completed_us", "output_tokens", "http_status", "error", "ttft_ms",
        "tpot_ms", "e2e_ms"
    ]
    with requests.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "request_class": "text",
            "scheduled_arrival_us": 0,
            "first_token_us": 100_000,
            "completed_us": 1_000_000,
            "output_tokens": 32,
            "http_status": 200,
            "error": "",
            "ttft_ms": 100,
            "tpot_ms": 20,
            "e2e_ms": 1000,
        })

    surface_path = Path(__file__).parents[
        2] / "benchmarks" / "phase_serving" / "analyze_slo_surface.py"
    subprocess.run([
        sys.executable,
        str(surface_path), "--run", f"test={requests}", "--ttft-ms", "50",
        "150", "--tpot-ms", "10", "30", "--output-json",
        str(output_json), "--output-csv",
        str(output_csv)
    ],
                   check=True)

    artifact = json.loads(output_json.read_text())
    assert len(artifact["run_points"]) == 4
    assert len(artifact["surface"]) == 4
    assert sum(row["passed_requests"] for row in artifact["surface"]) == 1
