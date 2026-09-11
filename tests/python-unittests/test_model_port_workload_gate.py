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
          "build_model_port_workload_gate.py")
SPEC = importlib.util.spec_from_file_location("model_port_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODEL_PORT_GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL_PORT_GATE)


def test_materialize_trace_repeats_and_containerizes_images() -> None:
    source = {
        "workload": "multi-image",
        "requests": [{
            "semantic_id": "pair",
            "arrival_offset_us": 10,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": "file:///home/sslab/TensorRT-Edge-LLM/examples/multimodal/pics/red_panda.jpeg"
                    },
                }],
            }],
        }],
    }

    result = MODEL_PORT_GATE.materialize_trace(source, "multi-image", 3,
                                                100)

    assert len(result["requests"]) == 3
    assert [request["arrival_offset_us"] for request in result["requests"]
            ] == [10, 120, 230]
    assert result["requests"][2]["semantic_id"] == "pair-port-002"
    assert result["requests"][0]["messages"][0]["content"][0][
        "image_url"]["url"].startswith("file:///workspace/")
