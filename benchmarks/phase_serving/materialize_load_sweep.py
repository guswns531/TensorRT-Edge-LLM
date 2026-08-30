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
"""Scale only the arrival process of a materialized request trace."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def scale_trace(source: dict[str, Any],
                load_multiplier: float) -> dict[str, Any]:
    """Return a trace with arrival offsets divided by ``load_multiplier``."""
    if load_multiplier <= 0.0:
        raise ValueError("load_multiplier must be positive")
    result = copy.deepcopy(source)
    requests = result.get("requests")
    if not isinstance(requests, list):
        raise ValueError("trace must contain a requests list")
    for request in requests:
        if "arrival_offset_us" not in request:
            raise ValueError("every request must contain arrival_offset_us")
        request["arrival_offset_us"] = round(
            float(request["arrival_offset_us"]) / load_multiplier)
    result["load_multiplier"] = load_multiplier
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--multiplier",
                        action="append",
                        type=float,
                        required=True)
    args = parser.parse_args()

    source = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for multiplier in args.multiplier:
        scaled = scale_trace(source, multiplier)
        output = args.output_dir / f"load-{multiplier:g}x.json"
        output.write_text(json.dumps(scaled, indent=2) + "\n",
                          encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
