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
"""Repeat a real-request trace into deterministic arrival waves."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def build_waves(source: dict[str, Any], waves: int,
                wave_interval_us: int) -> dict[str, Any]:
    """Return repeated requests while preserving each source wave's offsets."""
    requests = source.get("requests")
    if not isinstance(requests, list) or not requests:
        raise ValueError("source trace must contain a non-empty requests list")
    if waves < 1 or wave_interval_us < 0:
        raise ValueError("waves must be positive and wave interval non-negative")
    result = copy.deepcopy(source)
    result["workload"] = f"{source.get('workload', 'trace')}-waves-{waves}"
    result["source_workload"] = source.get("workload", "")
    result["wave_count"] = waves
    result["wave_interval_us"] = wave_interval_us
    result["requests"] = []
    for wave in range(waves):
        for source_index, original in enumerate(requests):
            request = copy.deepcopy(original)
            request["arrival_offset_us"] = (int(
                request.get("arrival_offset_us", 0)) + wave * wave_interval_us)
            semantic_id = request.get("semantic_id")
            if semantic_id:
                request["semantic_id"] = f"{semantic_id}-wave-{wave:03d}"
            request["source_request_index"] = source_index
            request["wave_index"] = wave
            result["requests"].append(request)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--waves", type=int, required=True)
    parser.add_argument("--wave-interval-us", type=int, default=0)
    args = parser.parse_args()
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        result = build_waves(source, args.waves, args.wave_interval_us)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n",
                               encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
