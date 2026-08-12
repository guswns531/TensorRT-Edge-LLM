#!/usr/bin/env python3
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
"""Create reproducible scheduler workloads from a real message trace."""

import argparse
import copy
import json
from pathlib import Path


def resize_text(text: str, target_chars: int) -> str:
    """Repeat a prompt to a deterministic approximate token-length bucket."""
    if target_chars <= 0 or len(text) >= target_chars:
        return text[:target_chars] if target_chars > 0 else text
    sections = []
    while sum(len(section) for section in sections) < target_chars:
        sections.append(text)
    return "\n\nAdditional context:\n".join(sections)[:target_chars]


def transform_request(request: dict[str, object], preset: str,
                      index: int) -> dict[str, object]:
    result = copy.deepcopy(request)
    messages = result.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("every request must contain at least one message")
    content = messages[-1].get("content")
    if not isinstance(content, str):
        raise ValueError("the final message content must be text")
    original_output = int(result.get("max_generate_length", 1))

    if preset == "interactive-short":
        target_chars = min(len(content), 900)
        output_multiplier = 1
    elif preset == "chat-balanced":
        target_chars = max(len(content), 1600)
        output_multiplier = 4
    elif preset == "long-prefill":
        target_chars = max(len(content), 3800)
        output_multiplier = 4
    elif preset == "decode-heavy":
        target_chars = min(len(content), 1200)
        output_multiplier = 12
    elif preset == "bimodal-mixed":
        long_request = index % 2 == 1
        target_chars = max(len(content), 3800) if long_request else min(
            len(content), 700)
        output_multiplier = 12 if long_request else 2
    else:
        raise ValueError(f"unsupported workload preset: {preset}")

    messages[-1]["content"] = resize_text(content, target_chars)
    result["max_generate_length"] = max(1, original_output * output_multiplier)
    result.pop("arrival_offset_us", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset",
                        required=True,
                        choices=("interactive-short", "chat-balanced",
                                 "long-prefill", "decode-heavy",
                                 "bimodal-mixed"))
    args = parser.parse_args()

    root = json.loads(args.input.read_text(encoding="utf-8"))
    requests = root.get("requests")
    if not isinstance(requests, list) or not requests:
        parser.error("input trace contains no requests")
    root["requests"] = [
        transform_request(request, args.preset, index)
        for index, request in enumerate(requests)
    ]
    root["max_generate_length"] = max(
        int(request["max_generate_length"]) for request in root["requests"])
    root["workload_preset"] = args.preset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(root, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
