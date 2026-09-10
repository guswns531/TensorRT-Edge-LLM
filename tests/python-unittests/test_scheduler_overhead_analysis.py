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

SCRIPT = (Path(__file__).parents[2] / "benchmarks" / "phase_serving" /
          "analyze_scheduler_overhead.py")
SPEC = importlib.util.spec_from_file_location("scheduler_overhead", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_and_aggregate_decision_cost(tmp_path):
    logs = []
    for run, mean, p95, maximum in (("run-001", 3.0, 5.0, 7.0),
                                    ("run-002", 5.0, 9.0, 13.0)):
        path = tmp_path / "v3" / "generic" / "mixed" / "worker-4" / run / "gateway.log"
        path.parent.mkdir(parents=True)
        path.write_text("Phase global scheduler decision cost: samples=10 "
                        f"mean={mean} us p95={p95} us max={maximum} us\n")
        logs.append(MODULE.load(path, tmp_path))

    result = MODULE.aggregate(logs)[0]

    assert result["policy"] == "v3"
    assert result["workload"] == "mixed"
    assert result["runs"] == 2
    assert result["samples"] == 20
    assert result["weighted_mean_us"] == 4.0
    assert result["run_p95_us_median"] == 7.0
    assert result["maximum_us"] == 13.0


def test_zero_decision_runs_remain_reportable():
    result = MODULE.aggregate([{
        "policy": "exact",
        "workload": "short",
        "samples": 0,
        "mean_us": 0.0,
        "p95_us": 0.0,
        "max_us": 0.0,
    }])[0]

    assert result["samples"] == 0
    assert result["weighted_mean_us"] == 0.0
