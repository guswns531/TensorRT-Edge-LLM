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
          "build_generic_policy_calibration_trace.py")
SPEC = importlib.util.spec_from_file_location("generic_trace", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GENERIC_TRACE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERIC_TRACE)


def test_generic_trace_is_fixed_and_optionally_covers_vision() -> None:
    text = GENERIC_TRACE.build_trace(None, cycles=1, cycle_interval_us=120_000)
    vision = GENERIC_TRACE.build_trace("file:///image.png",
                                       cycles=1,
                                       cycle_interval_us=120_000)

    assert len(text["requests"]) == 21
    assert len(vision["requests"]) == 28
    assert all(request["request_class"].startswith("generic_")
               for request in vision["requests"])
    assert [request["arrival_offset_us"] for request in vision["requests"]
            ] == sorted(request["arrival_offset_us"]
                        for request in vision["requests"])
    vision_requests = [
        request for request in vision["requests"]
        if request["request_class"] == "generic_vision"
    ]
    assert len(vision_requests) == 7
    assert sorted(
        sum(
            part.get("type") == "image_url"
            for part in request["messages"][0]["content"])
        for request in vision_requests) == [1, 1, 1, 1, 1, 1, 1]
