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
"""Compute request and token SLO goodput from HTTP request-level results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _passes_slo(row: dict[str, str], ttft_ms: float, tpot_ms: float,
                e2e_ms: float) -> bool:
    if int(row.get("http_status") or 0) != 200 or row.get("error"):
        return False
    scheduled_us = float(row["scheduled_arrival_us"])
    arrival_ttft_ms = (float(row["first_token_us"]) - scheduled_us) / 1000.0
    arrival_e2e_ms = (float(row["completed_us"]) - scheduled_us) / 1000.0
    return (arrival_ttft_ms <= ttft_ms and float(row["tpot_ms"]) <= tpot_ms
            and (e2e_ms <= 0.0 or arrival_e2e_ms <= e2e_ms))


def summarize(path: Path, ttft_ms: float, tpot_ms: float,
              e2e_ms: float) -> dict[str, Any]:
    """Summarize one request CSV using request-completion wall time."""
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"{path} contains no requests")
    start_us = min(float(row["scheduled_arrival_us"]) for row in rows)
    end_us = max(float(row["completed_us"]) for row in rows)
    duration_s = max((end_us - start_us) / 1_000_000.0, 1e-9)
    passed = [
        row for row in rows if _passes_slo(row, ttft_ms, tpot_ms, e2e_ms)
    ]
    class_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        class_rows[row.get("request_class", "unknown")].append(row)

    result = {
        "requests": len(rows),
        "passed_requests": len(passed),
        "pass_rate": len(passed) / len(rows),
        "duration_s": duration_s,
        "request_goodput_per_s": len(passed) / duration_s,
        "token_goodput_per_s":
        sum(int(row["output_tokens"]) for row in passed) / duration_s,
        "request_throughput_per_s": len(rows) / duration_s,
        "token_throughput_per_s":
        sum(int(row["output_tokens"]) for row in rows) / duration_s,
        "by_request_class": {},
    }
    for request_class, members in sorted(class_rows.items()):
        class_passed = [
            row for row in members
            if _passes_slo(row, ttft_ms, tpot_ms, e2e_ms)
        ]
        result["by_request_class"][request_class] = {
            "requests": len(members),
            "passed_requests": len(class_passed),
            "pass_rate": len(class_passed) / len(members),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",
                        action="append",
                        required=True,
                        help="LABEL=path/to/requests.csv; may be repeated")
    parser.add_argument("--ttft-ms", type=float, required=True)
    parser.add_argument("--tpot-ms", type=float, required=True)
    parser.add_argument("--e2e-ms", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for spec in args.run:
        label, separator, raw_path = spec.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("--run must use LABEL=PATH")
        result = summarize(Path(raw_path), args.ttft_ms, args.tpot_ms,
                           args.e2e_ms)
        result["label"] = label
        result["requests_csv"] = raw_path
        results.append(result)
    payload = {
        "slo": {
            "ttft_ms": args.ttft_ms,
            "tpot_ms": args.tpot_ms,
            "e2e_ms": args.e2e_ms,
        },
        "runs": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
