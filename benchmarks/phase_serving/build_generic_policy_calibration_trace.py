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
"""Build one workload-independent E/P/D contextual-policy calibration trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _text_request(prompt_tokens: int, output_tokens: int, arrival_us: int,
                  request_class: str) -> dict[str, Any]:
    unit = "Explain safe GPU phase scheduling using measured costs. "
    words = max(1, prompt_tokens // 10)
    return {
        "messages": [{
            "role": "user",
            "content": unit * words
        }],
        "max_generate_length": output_tokens,
        "arrival_offset_us": arrival_us,
        "request_class": request_class,
    }


def _vision_request(image_url: str, output_tokens: int, arrival_us: int,
                    request_class: str) -> dict[str, Any]:
    return {
        "messages": [{
            "role":
            "user",
            "content": [{
                "type": "image_url",
                "image_url": {
                    "url": image_url
                },
            }, {
                "type": "text",
                "text": "Describe the image briefly.",
            }],
        }],
        "max_generate_length":
        output_tokens,
        "arrival_offset_us":
        arrival_us,
        "request_class":
        request_class,
    }


def build_trace(image_url: str | None, cycles: int,
                cycle_interval_us: int) -> dict[str, Any]:
    """Return a fixed state-coverage trace independent of measured traffic."""
    requests: list[dict[str, Any]] = []
    for cycle in range(cycles):
        base = cycle * cycle_interval_us
        # Establish long-lived D cohorts at three useful cohort scales.
        for rows, offset in ((8, 0), (32, 1200), (64, 2400)):
            requests.extend(
                _text_request(32, 64, base + offset, "generic_resident_decode")
                for _ in range(rows))
        # Cross the resident cohorts with fixed-128 P candidates. Alternating
        # offsets exposes both incumbent directions without observing the
        # production trace being measured.
        for wave, rows in enumerate((1, 4, 8, 8, 4, 1)):
            requests.extend(
                _text_request(128, 8, base + 4000 +
                              wave * 1500, "generic_prefill")
                for _ in range(rows))
        if image_url is not None:
            for wave, rows in enumerate((1, 2, 4, 4, 2, 1)):
                requests.extend(
                    _vision_request(image_url, 16, base + 4500 +
                                    wave * 1700, "generic_vision")
                    for _ in range(rows))
    requests.sort(key=lambda request: int(request["arrival_offset_us"]))
    return {
        "schema_version": 1,
        "kind": "workload_independent_policy_calibration",
        "cycles": cycles,
        "requests": requests,
        "max_generate_length": 64,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-url")
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--cycle-interval-us", type=int, default=120_000)
    args = parser.parse_args()
    if args.cycles <= 0 or args.cycle_interval_us <= 0:
        parser.error("cycles and cycle interval must be positive")
    trace = build_trace(args.image_url, args.cycles, args.cycle_interval_us)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, indent=2) + "\n",
                           encoding="utf-8")
    print(
        json.dumps({
            "output": str(args.output),
            "requests": len(trace["requests"]),
            "vision": args.image_url is not None,
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
