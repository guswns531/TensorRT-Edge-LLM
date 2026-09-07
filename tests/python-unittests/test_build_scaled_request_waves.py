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

from benchmarks.phase_serving.build_scaled_request_waves import build_waves


def test_repeats_requests_with_deterministic_wave_offsets() -> None:
    source = {
        "schema_version": 1,
        "workload": "multi-image",
        "requests": [{
            "semantic_id": "first",
            "arrival_offset_us": 7,
            "messages": [],
        }, {
            "semantic_id": "second",
            "arrival_offset_us": 11,
            "messages": [],
        }],
    }

    result = build_waves(source, 3, 100)

    assert len(result["requests"]) == 6
    assert [request["arrival_offset_us"] for request in result["requests"]
            ] == [7, 11, 107, 111, 207, 211]
    assert result["requests"][4]["semantic_id"] == "first-wave-002"
    assert result["requests"][4]["source_request_index"] == 0
    assert result["requests"][4]["wave_index"] == 2
