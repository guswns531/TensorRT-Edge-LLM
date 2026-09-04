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
import json

from benchmarks.phase_serving.analyze_completion_vllm_gate import (
    _relative_percent, build_comparison)


def _write_requests(path, duration_us, ttft_us, tpot_ms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target,
                                fieldnames=[
                                    "http_status", "error",
                                    "scheduled_arrival_us", "first_token_us",
                                    "completed_us", "tpot_ms", "output_tokens",
                                    "request_class"
                                ])
        writer.writeheader()
        writer.writerow({
            "http_status": 200,
            "error": "",
            "scheduled_arrival_us": 0,
            "first_token_us": ttft_us,
            "completed_us": duration_us,
            "tpot_ms": tpot_ms,
            "output_tokens": 10,
            "request_class": "text",
        })


def test_compares_policy_matrix_with_fresh_vllm_suite(tmp_path) -> None:
    current_aggregate = tmp_path / "current.json"
    current_aggregate.write_text(json.dumps(
        {"gpu_memory_peak_mib_median": 9000}),
                                 encoding="utf-8")
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps({
        "rows": [{
            "policy": "scalar",
            "workload": "balanced",
            "throughput_req_s": 12.0,
            "token_s": 120.0,
            "ttft_mean_ms": 10.0,
            "ttft_p95_ms": 20.0,
            "tpot_mean_ms": 2.0,
            "tpot_p95_ms": 3.0,
            "e2e_mean_ms": 30.0,
            "e2e_p95_ms": 40.0,
            "joint_slo_pass_rate": 1.0,
            "joint_slo_goodput_req_s": 12.0,
            "slo_repeats": 3,
            "aggregate": str(current_aggregate),
        }]
    }),
                      encoding="utf-8")
    vllm_root = tmp_path / "vllm"
    _write_requests(vllm_root / "balanced/run-001/client/run-001/requests.csv",
                    100000, 10000, 2.0)
    (vllm_root / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (vllm_root / "summary.json").write_text(json.dumps({
        "cases": {
            "balanced": {
                "runs": [{
                    "achieved_req_s_median": 10.0
                }],
                "generated_token_s_median": 100.0,
                "ttft_mean_ms": 12.0,
                "ttft_p95_ms": 22.0,
                "tpot_mean_ms": 2.5,
                "tpot_p95_ms": 3.5,
                "e2e_mean_ms": 35.0,
                "e2e_p95_ms": 45.0,
                "gpu_memory_peak_mib": 8000.0,
            }
        }
    }),
                                            encoding="utf-8")
    result = build_comparison(matrix, "scalar", vllm_root, 500.0, 50.0, 2500.0)
    assert result["summary"]["throughput_wins"] == 1
    assert result["summary"]["ttft_mean_ms_wins"] == 1
    assert abs(
        result["summary"]["geometric_mean_token_throughput_vs_vllm_percent"] -
        20.0) < 1e-9
    assert abs(result["rows"][0]["current_vs_vllm_token_s_percent"] -
               20.0) < 1e-9


def test_relative_percent_marks_a_zero_reference_as_unbounded() -> None:
    assert _relative_percent(1.0, 0.0) is None
    assert _relative_percent(0.0, 0.0) == 0.0
