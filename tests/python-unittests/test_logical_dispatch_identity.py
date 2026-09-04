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
          "analyze_logical_dispatch_identity.py")
SPEC = importlib.util.spec_from_file_location("logical_identity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IDENTITY)


def test_sequence_comparison_distinguishes_order_from_multiset() -> None:
    left = [("prefill", (1,)), ("decode", (1,))]
    right = list(reversed(left))

    result = IDENTITY._compare_sequence(left, right)

    assert not result["exact"]
    assert result["multiset_equal"]
    assert result["common_prefix"] == 0
    assert result["first_mismatch"] == 0


def test_load_infers_gateway_log_for_host_decision_cost(tmp_path: Path) -> None:
    activity = tmp_path / "default" / "activity"
    activity.mkdir(parents=True)
    events = activity / "run-001-events.jsonl"
    events.write_text("", encoding="utf-8")
    gateway = tmp_path / "default" / "run-001" / "gateway.log"
    gateway.parent.mkdir()
    gateway.write_text(
        "Phase global scheduler decision cost: samples=12 mean=3.5 us "
        "p95=7.0 us max=11.0 us\n", encoding="utf-8")

    result = IDENTITY.load(events)

    assert result["decision_cost"] == {
        "samples": 12,
        "mean_us": 3.5,
        "p95_us": 7.0,
        "max_us": 11.0,
    }
