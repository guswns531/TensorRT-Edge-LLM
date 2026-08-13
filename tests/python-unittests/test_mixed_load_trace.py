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

import argparse
import runpy
from pathlib import Path

import pytest

SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "cosmos_reason2" /
          "build_mixed_load_trace.py")
GLOBALS = runpy.run_path(str(SCRIPT))


def test_materialize_cycles_requests_and_preserves_phase_boundaries():
    source = {"requests": [{"prompt": "a"}, {"prompt": "b"}]}
    phases = [
        GLOBALS["LoadPhase"]("low", 2, 10.0, 0.0),
        GLOBALS["LoadPhase"]("burst", 3, 1000.0, 250.0),
    ]

    first = GLOBALS["materialize"](source, phases, 7)
    second = GLOBALS["materialize"](source, phases, 7)

    assert first == second
    assert [request["prompt"]
            for request in first["requests"]] == ["a", "b", "a", "b", "a"]
    assert first["load_phases"][0]["first_request_index"] == 0
    assert first["load_phases"][1]["first_request_index"] == 2
    low_last = first["load_phases"][0]["last_arrival_offset_us"]
    assert first["load_phases"][1]["start_offset_us"] == low_last + 250_000
    arrivals = [request["arrival_offset_us"] for request in first["requests"]]
    assert arrivals == sorted(arrivals)


def test_parse_phase_rejects_invalid_values():
    with pytest.raises(argparse.ArgumentTypeError):
        GLOBALS["parse_phase"]("burst:0:1000:0")
    with pytest.raises(argparse.ArgumentTypeError):
        GLOBALS["parse_phase"]("burst:32:fast:0")
