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
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parents[2] / "benchmarks" / "phase_serving"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "warmup_sweep", SCRIPT_DIR / "run_policy_warmup_sweep.py")
assert SPEC is not None and SPEC.loader is not None
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)
ANALYSIS_SPEC = importlib.util.spec_from_file_location(
    "warmup_analysis", SCRIPT_DIR / "analyze_policy_warmup_sweep.py")
assert ANALYSIS_SPEC is not None and ANALYSIS_SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(ANALYSIS_SPEC)
ANALYSIS_SPEC.loader.exec_module(ANALYSIS)


def _trace(path: Path) -> None:
    path.write_text(json.dumps({"requests": [{"messages": [{
        "content": "hello"
    }]} for _ in range(4)]}), encoding="utf-8")


def test_prepare_budget_command_uses_exact_budget(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    _trace(trace)
    command = [
        "python3", "bench.py", "--trace", str(trace), "--output-dir",
        "old", "--repeats", "1", "--", "docker", "run", "--rm",
        "nvcr.io/nvidia/tensorrt:26.06-py3", "binary"
    ]
    result = SWEEP.prepare_budget_command(
        {"command": command}, 10, tmp_path / "out", 2, trace, trace, "",
        "", "full_active", (), 0)

    assert result[result.index("--warmup-requests") + 1] == "10"
    assert result[result.index("--phase-calibration-min-requests") + 1] == "10"
    assert result[result.index("--phase-calibration-round-requests") + 1] == "4"


def test_parse_budgets_rejects_duplicates() -> None:
    try:
        SWEEP._parse_budgets("0,4,4")
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate budgets must be rejected")


def test_process_wall_duration_uses_gateway_timestamps(tmp_path: Path) -> None:
    run = tmp_path / "run-001"
    run.mkdir()
    (run / "gateway.log").write_text(
        "[23:59:58.500] [INFO] start\n"
        "[00:00:01.750] [INFO] finish\n", encoding="utf-8")

    result = ANALYSIS._load_process_wall(tmp_path)

    assert result["process_wall_s_median"] == 3.25
